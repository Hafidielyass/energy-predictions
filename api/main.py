"""
FastAPI Deployment API — Solar PV Forecasting & Anomaly Detection

Endpoints
---------
GET  /health                  — liveness probe
POST /predict/forecast        — 24-hour power generation forecast
POST /predict/anomaly         — daily anomaly classification
GET  /model/info              — model metadata & version
POST /predict/batch_forecast  — batch multi-day forecast

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import logging

# ── Suppress TF warnings BEFORE any tensorflow import ────────────────────────
# Fixes: "TensorFlow GPU support is not available on native Windows >= 2.11"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import warnings
warnings.filterwarnings("ignore", message=".*TensorFlow GPU.*")
warnings.filterwarnings("ignore", message=".*native Windows.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, date
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ─── App initialisation ───────────────────────────────────────────────────────

app = FastAPI(
    title="Solar PV Forecasting & Anomaly Detection API",
    description=(
        "Production REST API for next-24h AC power forecasting and "
        "automatic fault classification at a photovoltaic plant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ─── Global model state ───────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "gru_model":      None,
    "feature_scaler": None,
    "anomaly_detector": None,
    "model_meta":     {},
    "loaded_at":      None,
}


def _load_models():
    """Lazy-load models on first request (avoids cold-start penalty at import)."""
    if _state["gru_model"] is not None:
        return

    try:
        from tensorflow import keras
        gru_path     = os.getenv("GRU_MODEL_PATH",     "models/forecasting/best_gru_model.keras")
        scaler_path  = os.getenv("SCALER_PATH",        "models/forecasting/feature_scaler.pkl")
        ae_path      = os.getenv("AE_MODEL_PATH",      "models/anomaly/autoencoder.keras")
        iso_path     = os.getenv("ISO_MODEL_PATH",     "models/anomaly/iso_forest.pkl")
        ad_scaler    = os.getenv("AD_SCALER_PATH",     "models/anomaly/anomaly_scaler.pkl")

        if os.path.exists(gru_path):
            _state["gru_model"] = keras.models.load_model(gru_path)
            log.info("GRU model loaded from %s", gru_path)

        if os.path.exists(scaler_path):
            _state["feature_scaler"] = joblib.load(scaler_path)
            log.info("Feature scaler loaded.")

        if os.path.exists(ae_path) and os.path.exists(iso_path):
            from src.anomaly_detection.anomaly_detector import HybridAnomalyDetector
            from src.utils.config_loader import load_config
            cfg = load_config()
            det = HybridAnomalyDetector(cfg)
            det.load(ad_scaler, ae_path, iso_path)
            _state["anomaly_detector"] = det
            log.info("Anomaly detector loaded.")

        _state["model_meta"] = {
            "gru_version":   "1.0.0",
            "framework":     "TensorFlow 2.21 / Keras 3",
            "forecast_type": "GRU multi-step 24h",
            "sequence_len":  96,
            "horizon":       96,
            "resample_freq": "15T",
        }
        _state["loaded_at"] = datetime.now().isoformat()

    except Exception as exc:
        log.error("Model loading failed: %s", exc)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """One 15-minute sensor observation."""
    measured_on:             str   = Field(..., example="2022-06-15 08:00:00")
    ac_power__315:           float = Field(..., ge=0,      example=450.0)
    poa_irradiance__313:     float = Field(...,            example=620.0)
    ambient_temp__320:       float = Field(...,            example=28.5)
    module_temp_1__321:      float = Field(...,            example=45.2)
    dc_pos_voltage__316:     float = Field(...,            example=270.0)
    dc_pos_current__317:     float = Field(...,            example=1.8)
    dc_power__314:           float = Field(...,            example=510.0)
    ac_voltage__318:         float = Field(...,            example=122.5)
    ac_current__319:         float = Field(...,            example=3.7)
    inverter_temp__324:      float = Field(...,            example=38.0)
    power_factor__327:       float = Field(0.95, ge=-1, le=1)
    das_battery_voltage__326: float = Field(13.5, example=13.5)

    @field_validator("measured_on")
    @classmethod
    def validate_ts(cls, v):
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"Invalid timestamp: {v}")
        return v


class ForecastRequest(BaseModel):
    """24h forecast request: provide the last 96 × 15-min observations."""
    observations: List[SensorReading] = Field(
        ...,
        min_length=96,
        max_length=200,
        description="Historical sensor readings (most recent last). Min 96 steps = 24h lookback.",
    )


class ForecastResponse(BaseModel):
    status:          str
    forecast_steps:  int
    forecast_horizon_hours: int
    timestamps:      List[str]
    ac_power_w:      List[float]
    confidence_note: str
    inference_ms:    float


class DailyObservation(BaseModel):
    """Aggregated daily statistics for anomaly scoring."""
    date:               str   = Field(..., example="2022-06-15")
    daily_energy_kwh:   float = Field(..., ge=0, example=4.2)
    peak_power_kw:      float = Field(..., ge=0, example=1.1)
    performance_ratio:  float = Field(1.0,       example=0.88)
    efficiency:         float = Field(0.0,        example=0.12)
    irradiance_sum:     float = Field(0.0,        example=5800.0)
    power_ramp_std:     float = Field(0.0,        example=12.5)
    zero_power_ratio:   float = Field(0.0, ge=0, le=1, example=0.03)
    power_factor_mean:  float = Field(0.95,       example=0.94)
    temp_delta:         float = Field(0.0,        example=18.0)
    dc_ac_ratio:        float = Field(1.0,        example=1.12)
    ac_voltage_std:     float = Field(0.0,        example=0.4)
    inverter_temp_max:  float = Field(0.0,        example=42.0)


class AnomalyRequest(BaseModel):
    days: List[DailyObservation] = Field(..., min_length=1)


class AnomalyResult(BaseModel):
    date:           str
    anomaly_score:  float
    severity_label: str
    if_score:       float
    ae_score:       float
    stat_score:     float
    curve_score:    float


class AnomalyResponse(BaseModel):
    status:  str
    results: List[AnomalyResult]
    summary: Dict[str, int]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    _load_models()


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": _state["gru_model"] is not None,
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Return metadata about loaded models."""
    _load_models()
    return {
        "status": "ok",
        "meta": _state["model_meta"],
        "loaded_at": _state["loaded_at"],
    }


@app.post("/predict/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast_power(request: ForecastRequest):
    """
    Forecast AC power for the next 24 hours (96 × 15-min steps).

    Provide at least 96 consecutive 15-minute sensor readings.
    The most recent 96 readings are used as the lookback window.
    """
    t0 = time.perf_counter()
    _load_models()

    if _state["gru_model"] is None or _state["feature_scaler"] is None:
        raise HTTPException(
            status_code=503,
            detail="Forecasting model not available. Ensure model files are present.",
        )

    try:
        # ── Build DataFrame from observations ──────────────────────────────────
        obs_dicts = [o.model_dump() for o in request.observations[-96:]]
        df = pd.DataFrame(obs_dicts)
        df["measured_on"] = pd.to_datetime(df["measured_on"])
        df = df.set_index("measured_on").sort_index()

        # ── Run feature engineering ────────────────────────────────────────────
        from src.features.feature_engineering import build_features
        feat = build_features(df, cache_path=None, force_rebuild=True)

        # Drop non-numeric / target columns for the model
        drop_cols = ["ac_power__315"]
        X_cols = [c for c in feat.columns if c not in drop_cols]
        X_scaled = _state["feature_scaler"].transform(feat[X_cols].values[-96:])
        X_input = X_scaled.reshape(1, 96, -1).astype(np.float32)

        # ── Predict ────────────────────────────────────────────────────────────
        y_pred = _state["gru_model"].predict(X_input, verbose=0)[0]  # (96,)

        # ── Generate forecast timestamps ───────────────────────────────────────
        last_ts = df.index[-1]
        forecast_ts = pd.date_range(
            start=last_ts + pd.Timedelta("15min"),
            periods=96,
            freq="15min",
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        return ForecastResponse(
            status="success",
            forecast_steps=96,
            forecast_horizon_hours=24,
            timestamps=[str(ts) for ts in forecast_ts],
            ac_power_w=[max(0.0, float(v)) for v in y_pred],
            confidence_note="Point forecast — no uncertainty intervals in v1.0",
            inference_ms=round(inference_ms, 2),
        )

    except Exception as exc:
        log.error("Forecast error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/anomaly", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def classify_anomaly(request: AnomalyRequest):
    """
    Classify each day as Normal / Moderately Faulty / Severely Faulty.

    Provide one or more DailyObservation objects (one per calendar day).
    """
    t0 = time.perf_counter()
    _load_models()

    if _state["anomaly_detector"] is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector not available. Ensure model files are present.",
        )

    try:
        rows = [d.model_dump() for d in request.days]
        daily_df = pd.DataFrame(rows)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.set_index("date")

        result_df = _state["anomaly_detector"].predict(daily_df)

        results = []
        for idx, row in result_df.iterrows():
            results.append(AnomalyResult(
                date=str(idx.date()),
                anomaly_score=float(row.get("anomaly_score", 0)),
                severity_label=str(row.get("severity_label", "Unknown")),
                if_score=float(row.get("if_score",    0)),
                ae_score=float(row.get("ae_score",    0)),
                stat_score=float(row.get("stat_score",  0)),
                curve_score=float(row.get("curve_score", 0)),
            ))

        label_counts = result_df["severity_label"].value_counts().to_dict()
        summary = {
            "Normal":            label_counts.get("Normal",            0),
            "Moderately Faulty": label_counts.get("Moderately Faulty", 0),
            "Severely Faulty":   label_counts.get("Severely Faulty",   0),
        }

        inference_ms = (time.perf_counter() - t0) * 1000
        log.info("Anomaly endpoint: %d days processed in %.1f ms", len(results), inference_ms)
        return AnomalyResponse(status="success", results=results, summary=summary)

    except Exception as exc:
        log.error("Anomaly error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
