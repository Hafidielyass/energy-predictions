"""
InferencePipeline — Production-Ready End-to-End Inference
===========================================================

Encapsulates the full feature engineering → scaling → model inference
workflow so that a single call converts raw sensor readings into either:
  (a) a 24-hour ahead AC power forecast  (``run_forecast``)
  (b) a daily anomaly severity score     (``run_anomaly``)

Design goals
------------
* **Zero external dependencies at inference time** — only joblib + keras + numpy
* **Consistent preprocessing** — uses the same scaler fitted during training
* **Stateless** — no mutable global state; safe for concurrent requests
* **Validated inputs** — raises descriptive errors before model call

Usage
-----
>>> from src.deployment.inference_pipeline import InferencePipeline
>>> pipeline = InferencePipeline.from_paths(
...     gru_path    = "models/forecasting/best_gru_model.keras",
...     scaler_path = "models/forecasting/feature_scaler.pkl",
...     ae_path     = "models/anomaly/autoencoder.keras",
...     iso_path    = "models/anomaly/iso_forest.pkl",
...     ad_scaler_p = "models/anomaly/anomaly_scaler.pkl",
... )
>>> # 24-hour forecast
>>> result = pipeline.run_forecast(recent_15min_df)
>>> print(result["timestamps"], result["ac_power_w"])
>>>
>>> # Anomaly detection on daily aggregates
>>> anomaly = pipeline.run_anomaly(daily_df)
>>> print(anomaly["severity_label"], anomaly["anomaly_score"])
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class InferencePipeline:
    """
    Unified forecast + anomaly inference pipeline.

    Parameters
    ----------
    gru_model      : keras.Model  — trained GRU forecasting model
    feature_scaler : StandardScaler — fitted on training feature matrix
    anomaly_det    : HybridAnomalyDetector | None
    seq_len        : int  — lookback window in 15-min periods (default 96 = 24 h)
    horizon        : int  — forecast horizon in 15-min periods (default 96 = 24 h)
    feature_cols   : list[str]  — ordered feature column names
    """

    def __init__(
        self,
        gru_model,
        feature_scaler,
        anomaly_det=None,
        seq_len: int = 96,
        horizon: int = 96,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.gru_model     = gru_model
        self.feature_scaler = feature_scaler
        self.anomaly_det   = anomaly_det
        self.seq_len       = seq_len
        self.horizon       = horizon
        self.feature_cols  = feature_cols or []

    # ─── Factory constructors ─────────────────────────────────────────────────

    @classmethod
    def from_paths(
        cls,
        gru_path: str,
        scaler_path: str,
        ae_path: Optional[str] = None,
        iso_path: Optional[str] = None,
        ad_scaler_path: Optional[str] = None,
        config_path: str = "configs/config.yaml",
    ) -> "InferencePipeline":
        """
        Load all artefacts from file paths and return a ready pipeline.

        Parameters
        ----------
        gru_path       : Path to the saved GRU ``.keras`` model.
        scaler_path    : Path to the joblib-serialised ``StandardScaler``.
        ae_path        : (optional) Autoencoder model path.
        iso_path       : (optional) Isolation Forest joblib path.
        ad_scaler_path : (optional) Anomaly scaler + metadata joblib path.
        config_path    : Path to ``configs/config.yaml``.
        """
        from tensorflow import keras

        log.info("Loading GRU model from %s", gru_path)
        gru = keras.models.load_model(gru_path)

        log.info("Loading feature scaler from %s", scaler_path)
        scaler_data = joblib.load(scaler_path)
        # scaler_path may store bare scaler or dict
        if isinstance(scaler_data, dict):
            scaler      = scaler_data["scaler"]
            feat_cols   = scaler_data.get("feature_cols", [])
        else:
            scaler      = scaler_data
            feat_cols   = []

        anomaly_det = None
        if ae_path and iso_path and ad_scaler_path:
            if Path(ae_path).exists() and Path(iso_path).exists() and Path(ad_scaler_path).exists():
                from src.utils.config_loader import load_config
                from src.anomaly_detection.anomaly_detector import HybridAnomalyDetector
                cfg = load_config(config_path)
                anomaly_det = HybridAnomalyDetector(cfg)
                anomaly_det.load(ad_scaler_path, ae_path, iso_path)
                log.info("Anomaly detector loaded.")
            else:
                log.warning("One or more anomaly model files missing — anomaly endpoint disabled.")

        fc = cls(
            gru_model=gru,
            feature_scaler=scaler,
            anomaly_det=anomaly_det,
            feature_cols=feat_cols,
        )
        # Read seq_len / horizon from model input/output shapes
        fc.seq_len = gru.input_shape[1]
        fc.horizon = gru.output_shape[-1]
        log.info("InferencePipeline ready  seq_len=%d  horizon=%d", fc.seq_len, fc.horizon)
        return fc

    @classmethod
    def from_registry(cls, registry, version: str = "latest") -> "InferencePipeline":
        """
        Convenience constructor that resolves paths via the ModelRegistry.

        Parameters
        ----------
        registry : ModelRegistry instance
        version  : "latest" or a specific version string, e.g. "v1.2"
        """
        paths = registry.resolve(version)
        return cls.from_paths(**paths)

    # ─── Forecast endpoint ────────────────────────────────────────────────────

    def run_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Produce a 24-hour ahead AC power forecast.

        Parameters
        ----------
        df : pd.DataFrame
            At least ``seq_len`` rows of 15-min sensor readings with a
            DatetimeIndex, containing all columns required by the feature
            engineering step.

        Returns
        -------
        dict with keys:
            timestamps    : list[str]   — ISO forecast timestamps (96 steps)
            ac_power_w    : list[float] — predicted AC power in Watts (≥ 0)
            inference_ms  : float       — wall-clock latency in milliseconds
        """
        t0 = time.perf_counter()
        self._validate_df(df, min_rows=self.seq_len)

        # Feature engineering
        X_scaled = self._prepare_features(df)
        if len(X_scaled) < self.seq_len:
            raise ValueError(
                f"After feature engineering only {len(X_scaled)} rows remain "
                f"(need {self.seq_len}). Provide more history."
            )

        # Take last seq_len rows as the lookback window
        window = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1).astype(np.float32)

        # Predict
        y_pred = self.gru_model.predict(window, verbose=0)[0]  # (horizon,)
        y_pred = np.maximum(0.0, y_pred)

        # Generate forecast timestamps
        last_ts = df.index[-1]
        ts = pd.date_range(
            start=last_ts + pd.Timedelta("15T"),
            periods=self.horizon,
            freq="15T",
        )

        ms = (time.perf_counter() - t0) * 1000
        log.info("Forecast completed in %.1f ms  horizon=%d", ms, self.horizon)
        return {
            "timestamps":   [str(t) for t in ts],
            "ac_power_w":   [float(v) for v in y_pred],
            "inference_ms": round(ms, 2),
        }

    # ─── Anomaly endpoint ─────────────────────────────────────────────────────

    def run_anomaly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each day in ``daily_df`` as Normal / Moderately / Severely Faulty.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily aggregated features — output of ``build_daily_df()``.
            Must have a DatetimeIndex at daily frequency.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with anomaly scores and severity labels.
        """
        if self.anomaly_det is None:
            raise RuntimeError(
                "Anomaly detector not loaded. Provide ae_path / iso_path / ad_scaler_path "
                "when constructing InferencePipeline."
            )
        t0 = time.perf_counter()
        result = self.anomaly_det.predict(daily_df)
        ms = (time.perf_counter() - t0) * 1000
        log.info("Anomaly scoring completed in %.1f ms  (%d days)", ms, len(daily_df))
        return result

    # ─── Full pipeline ────────────────────────────────────────────────────────

    def run_full(
        self,
        df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run both forecast and anomaly detection in one call.

        Parameters
        ----------
        df       : pd.DataFrame  — 15-min time-series (recent observations)
        daily_df : pd.DataFrame | None  — pre-computed daily features.
                   If None, builds them from ``df`` automatically.

        Returns
        -------
        dict with keys ``forecast`` and ``anomaly``.
        """
        from src.preprocessing.data_loader import build_daily_df

        forecast_result = self.run_forecast(df)

        if daily_df is None:
            daily_df = build_daily_df(df)

        anomaly_result = None
        if self.anomaly_det is not None:
            anomaly_result = self.run_anomaly(daily_df)

        return {
            "forecast": forecast_result,
            "anomaly":  anomaly_result,
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _validate_df(self, df: pd.DataFrame, min_rows: int) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")
        if len(df) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows, got {len(df)}."
            )

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Run feature engineering and return the scaled feature matrix."""
        from src.features.feature_engineering import build_features

        feat = build_features(
            df,
            target_col="ac_power__315",
            lag_periods=[1, 2, 4, 8, 16, 32, 48, 96],
            rolling_windows=[4, 8, 16, 48, 96],
            cache_path=None,
            force_rebuild=True,
        )
        drop_cols = ["ac_power__315", "is_daytime"]
        feat_cols = [c for c in feat.columns if c not in drop_cols]

        # Align columns with training feature set if available
        if self.feature_cols:
            missing = set(self.feature_cols) - set(feat_cols)
            if missing:
                log.warning("Missing feature columns at inference: %s", missing)
            feat_cols = [c for c in self.feature_cols if c in feat_cols]

        X = self.feature_scaler.transform(feat[feat_cols].values).astype(np.float32)
        return X
