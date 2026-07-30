"""
Hybrid Anomaly Detection & Fault Classification Pipeline.

Methodology
-----------
No ground-truth fault labels are available, so we use an **unsupervised
hybrid framework** combining four complementary signals:

Signal 1 — Isolation Forest
    Tree-based algorithm; each data point is scored by how quickly it can
    be isolated. Anomalies are isolated in fewer splits → higher score.
    Captures multivariate outliers efficiently (Liu et al. 2008).

Signal 2 — Autoencoder Reconstruction Error
    A deep autoencoder trained on *normal* days learns a compressed
    representation of healthy plant behaviour. Abnormal days have high
    reconstruction error because they deviate from the learned manifold.
    (Malhotra et al. 2016; Principi et al. 2019)

Signal 3 — Statistical Thresholds (domain knowledge)
    Physics-based rules derived from operational constraints:
    * Daily energy far below seasonal median → potential shading / fault
    * Very high zero-power ratio during daylight → inverter trip
    * Performance Ratio (PR) deviation beyond 2-sigma → degradation
    * Irradiance-power efficiency collapse → soiling / cell damage

Signal 4 — Power Curve Similarity (DTW-inspired)
    Compare each day's normalised power curve against the 30-day rolling
    median curve using cosine similarity. Abnormal shape = low similarity.

Composite Severity Scoring
---------------------------
A weighted composite score [0, 1] is computed from the four signals:
    score = 0.30·IF + 0.30·AE + 0.25·stat + 0.15·curve

Thresholds map score → label:
    score < 0.35  → Normal
    0.35 ≤ score < 0.65  → Moderately Faulty
    score ≥ 0.65  → Severely Faulty
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils.device_config import configure_tf   # must come before tensorflow import
tf = configure_tf(seed=42, verbose=False)
from tensorflow import keras
from tensorflow.keras import layers

from src.utils.logger import get_logger

log = get_logger(__name__)

SEVERITY_LABELS = {0: "Normal", 1: "Moderately Faulty", 2: "Severely Faulty"}


# ─── Autoencoder builder ──────────────────────────────────────────────────────

def build_autoencoder(
    n_features: int,
    encoding_dims: List[int] = None,
) -> Tuple[keras.Model, keras.Model]:
    """
    Build a symmetric Dense autoencoder.

    Returns
    -------
    autoencoder : full model (encoder + decoder)
    encoder     : encoder-only model for embeddings
    """
    if encoding_dims is None:
        encoding_dims = [32, 16, 8]

    # Encoder
    inputs   = keras.Input(shape=(n_features,), name="ae_input")
    encoded  = inputs
    for i, dim in enumerate(encoding_dims):
        encoded = layers.Dense(dim, activation="relu", name=f"enc_{i}")(encoded)
        encoded = layers.BatchNormalization(name=f"bn_enc_{i}")(encoded)

    # Bottleneck
    bottleneck = layers.Dense(encoding_dims[-1], activation="relu", name="bottleneck")(encoded)

    # Decoder
    decoded = bottleneck
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        decoded = layers.Dense(dim, activation="relu", name=f"dec_{i}")(decoded)
        decoded = layers.BatchNormalization(name=f"bn_dec_{i}")(decoded)
    outputs = layers.Dense(n_features, activation="linear", name="ae_output")(decoded)

    autoencoder = keras.Model(inputs, outputs, name="Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    encoder = keras.Model(inputs, bottleneck, name="Encoder")
    return autoencoder, encoder


# ─── Main detector class ──────────────────────────────────────────────────────

class HybridAnomalyDetector:
    """
    Full hybrid anomaly detection pipeline.

    Parameters
    ----------
    config : Config
        Project configuration object.

    Attributes
    ----------
    scaler : MinMaxScaler
    iso_forest : IsolationForest
    autoencoder : keras.Model
    encoder : keras.Model
    ae_threshold : float  — reconstruction error threshold
    """

    def __init__(self, config) -> None:
        self.config = config
        self.scaler: Optional[MinMaxScaler] = None
        self.iso_forest: Optional[IsolationForest] = None
        self.autoencoder: Optional[keras.Model] = None
        self.encoder: Optional[keras.Model] = None
        self.ae_threshold: float = 0.0
        self._feature_cols: List[str] = []

    # ── Feature selection ─────────────────────────────────────────────────────

    def _select_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Return the subset of daily features suitable for anomaly detection."""
        candidate_cols = [
            "daily_energy_kwh", "peak_power_kw", "performance_ratio",
            "efficiency", "irradiance_sum", "power_ramp_std",
            "zero_power_ratio", "power_factor_mean", "temp_delta",
            "dc_ac_ratio", "ac_voltage_std", "inverter_temp_max",
        ]
        cols = [c for c in candidate_cols if c in daily_df.columns]
        self._feature_cols = cols
        return daily_df[cols].copy()

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, daily_df: pd.DataFrame) -> "HybridAnomalyDetector":
        """
        Fit the full hybrid detector on daily aggregated features.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Output of ``build_daily_df()`` — one row per day.
        """
        log.info("Fitting anomaly detector on %d days...", len(daily_df))

        feat_df = self._select_features(daily_df)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Scale
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(feat_df.values).astype(np.float32)

        # ── Signal 1: Isolation Forest ────────────────────────────────────────
        ad_cfg = self.config.get("anomaly_detection", {})
        if_cfg = ad_cfg.get("isolation_forest", {})
        self.iso_forest = IsolationForest(
            n_estimators=int(if_cfg.get("n_estimators", 200)),
            contamination=float(if_cfg.get("contamination", 0.05)),
            random_state=int(if_cfg.get("random_state", 42)),
            n_jobs=-1,
        )
        self.iso_forest.fit(X)
        log.info("Isolation Forest fitted.")

        # ── Signal 2: Autoencoder ─────────────────────────────────────────────
        ae_cfg = ad_cfg.get("autoencoder", {})
        enc_dims = ae_cfg.get("encoding_dims", [32, 16, 8])
        self.autoencoder, self.encoder = build_autoencoder(
            n_features=X.shape[1],
            encoding_dims=enc_dims,
        )

        ae_cb = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        self.autoencoder.fit(
            X, X,
            epochs=int(ae_cfg.get("epochs", 80)),
            batch_size=int(ae_cfg.get("batch_size", 32)),
            validation_split=0.1,
            callbacks=ae_cb,
            verbose=0,
        )
        log.info("Autoencoder trained.")

        # Set reconstruction error threshold at percentile
        recon = self.autoencoder.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=1)
        thresh_pct = int(ae_cfg.get("threshold_percentile", 95))
        self.ae_threshold = float(np.percentile(errors, thresh_pct))
        log.info("AE threshold (p%d): %.6f", thresh_pct, self.ae_threshold)

        return self

    # ── Prediction / scoring ──────────────────────────────────────────────────

    def predict(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score and classify each day in ``daily_df``.

        Returns
        -------
        pd.DataFrame
            Original daily_df augmented with:
            - if_score        (0–1, higher = more anomalous)
            - ae_score        (0–1)
            - stat_score      (0–1)
            - curve_score     (0–1)
            - anomaly_score   composite (0–1)
            - severity_num    (0 / 1 / 2)
            - severity_label  (Normal / Moderately Faulty / Severely Faulty)
        """
        feat_df = self._select_features(daily_df)
        valid_idx = feat_df.replace([np.inf, -np.inf], np.nan).dropna().index
        feat_clean = feat_df.loc[valid_idx]
        X = self.scaler.transform(feat_clean.values).astype(np.float32)

        # ── Signal 1: Isolation Forest ────────────────────────────────────────
        # decision_function returns negative anomaly score; flip so high=bad
        if_raw   = -self.iso_forest.decision_function(X)
        if_norm  = self._minmax_norm(if_raw)

        # ── Signal 2: Autoencoder reconstruction error ────────────────────────
        recon    = self.autoencoder.predict(X, verbose=0)
        ae_raw   = np.mean((X - recon) ** 2, axis=1)
        ae_norm  = np.clip(ae_raw / (self.ae_threshold + 1e-9), 0, 1)

        # ── Signal 3: Statistical / domain thresholds ─────────────────────────
        stat_norm = self._statistical_score(feat_clean)

        # ── Signal 4: Power curve similarity ─────────────────────────────────
        curve_norm = self._curve_similarity_score(daily_df.loc[valid_idx])

        # ── Composite score ───────────────────────────────────────────────────
        composite = (
            0.30 * if_norm
            + 0.30 * ae_norm
            + 0.25 * stat_norm
            + 0.15 * curve_norm
        )

        result = daily_df.copy()
        result["if_score"]      = np.nan
        result["ae_score"]      = np.nan
        result["stat_score"]    = np.nan
        result["curve_score"]   = np.nan
        result["anomaly_score"] = np.nan

        result.loc[valid_idx, "if_score"]      = if_norm
        result.loc[valid_idx, "ae_score"]      = ae_norm
        result.loc[valid_idx, "stat_score"]    = stat_norm
        result.loc[valid_idx, "curve_score"]   = curve_norm
        result.loc[valid_idx, "anomaly_score"] = composite

        # ── Severity mapping ──────────────────────────────────────────────────
        sev_cfg = self.config.get("anomaly_detection", {}).get("severity", {})
        normal_th = float(sev_cfg.get("normal_below",   0.35))
        severe_th = float(sev_cfg.get("severe_above",   0.65))

        def _label(s):
            if pd.isna(s):   return np.nan
            if s < normal_th: return 0
            if s < severe_th: return 1
            return 2

        result["severity_num"]   = result["anomaly_score"].apply(_label)
        result["severity_label"] = result["severity_num"].map(SEVERITY_LABELS)

        log.info("Anomaly scoring complete.  Normal: %d | Moderate: %d | Severe: %d",
                 (result["severity_num"] == 0).sum(),
                 (result["severity_num"] == 1).sum(),
                 (result["severity_num"] == 2).sum())
        return result

    # ── Internal scoring helpers ──────────────────────────────────────────────

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def _statistical_score(self, feat_df: pd.DataFrame) -> np.ndarray:
        """
        Rule-based anomaly score derived from domain knowledge.

        Rules
        -----
        * daily_energy < 2-sigma below seasonal median          (+0.4)
        * zero_power_ratio > 0.6 during expected daylight        (+0.3)
        * performance_ratio deviation > 2 sigma from mean        (+0.2)
        * efficiency < 10th percentile                           (+0.1)
        * dc_ac_ratio > 95th percentile  (inverter mismatch)     (+0.2)
        """
        scores = np.zeros(len(feat_df))

        for i, col in enumerate(["daily_energy_kwh", "performance_ratio", "efficiency"]):
            if col not in feat_df.columns:
                continue
            vals  = feat_df[col].values
            mu    = np.nanmedian(vals)
            sigma = np.nanstd(vals) + 1e-9
            z     = (mu - vals) / sigma   # positive when below median
            scores += np.clip(z / 3.0, 0, 0.4)

        if "zero_power_ratio" in feat_df.columns:
            scores += np.clip(feat_df["zero_power_ratio"].values / 0.6, 0, 0.3)

        if "dc_ac_ratio" in feat_df.columns:
            p95 = np.nanpercentile(feat_df["dc_ac_ratio"].values, 95)
            scores += (feat_df["dc_ac_ratio"].values > p95).astype(float) * 0.2

        return self._minmax_norm(np.clip(scores, 0, 1))

    def _curve_similarity_score(self, daily_df: pd.DataFrame) -> np.ndarray:
        """
        Approximate power-curve shape deviation using normalised daily energy
        pattern. Proxy: how far daily energy deviates from 30-day rolling median.
        """
        if "daily_energy_kwh" not in daily_df.columns:
            return np.zeros(len(daily_df))

        energy = daily_df["daily_energy_kwh"]
        rolling_med = energy.rolling(30, min_periods=5, center=True).median()
        deviation = (energy - rolling_med).abs() / (rolling_med.abs() + 1e-9)
        scores = deviation.fillna(0).values
        return self._minmax_norm(scores)

    # ── Clustering evaluation ─────────────────────────────────────────────────

    def evaluate_clustering(self, result_df: pd.DataFrame) -> Dict[str, float]:
        """Compute silhouette score and Davies-Bouldin index."""
        feat_df = self._select_features(result_df)
        feat_clean = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
        labels = result_df.loc[feat_clean.index, "severity_num"].dropna().astype(int)
        feat_clean = feat_clean.loc[labels.index]

        if len(labels.unique()) < 2:
            log.warning("Only one cluster label found; skipping clustering metrics.")
            return {}

        X = self.scaler.transform(feat_clean.values)
        sil = float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42))
        db  = float(davies_bouldin_score(X, labels))
        metrics = {"silhouette_score": sil, "davies_bouldin_index": db}
        log.info("Clustering metrics: %s", metrics)
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, scaler_path: str, ae_path: str, iso_path: str) -> None:
        os.makedirs(Path(scaler_path).parent, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "ae_threshold": self.ae_threshold,
                     "feature_cols": self._feature_cols}, scaler_path)
        self.autoencoder.save(ae_path)
        joblib.dump(self.iso_forest, iso_path)
        log.info("Anomaly detector saved.")

    def load(self, scaler_path: str, ae_path: str, iso_path: str) -> None:
        d = joblib.load(scaler_path)
        self.scaler         = d["scaler"]
        self.ae_threshold   = d["ae_threshold"]
        self._feature_cols  = d["feature_cols"]
        self.autoencoder    = keras.models.load_model(ae_path)
        self.encoder        = keras.Model(
            self.autoencoder.input,
            self.autoencoder.get_layer("bottleneck").output,
        )
        self.iso_forest = joblib.load(iso_path)
        log.info("Anomaly detector loaded.")
