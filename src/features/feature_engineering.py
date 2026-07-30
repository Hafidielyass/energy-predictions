"""
Feature engineering pipeline for the PV forecasting model.

Key feature families
--------------------
1. **Temporal / cyclic** — hour, day-of-year, month encoded as sin/cos pairs
   so the model sees continuity at boundaries (hour 23 → 0, Dec → Jan).
2. **Lag features** — past ac_power values at multiple lookback distances
   to give the model explicit auto-regressive information.
3. **Rolling statistics** — mean, std, min, max over sliding windows to
   capture short-term trends and variability.
4. **Domain physics features** — irradiance-to-power efficiency, DC-AC
   ratio, module-ambient temperature delta.
5. **Calendar flags** — season, is_weekend for potential behavioural shifts.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─── Cyclic encoding ─────────────────────────────────────────────────────────

def encode_cyclic(series: pd.Series, period: float) -> pd.DataFrame:
    """Encode a periodic feature as sin + cos pair."""
    rad = 2 * np.pi * series / period
    name = series.name
    return pd.DataFrame({
        f"{name}_sin": np.sin(rad),
        f"{name}_cos": np.cos(rad),
    }, index=series.index)


# ─── Main feature builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target_col: str = "ac_power__315",
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    cache_path: Optional[str] = "data/features/feature_data.parquet",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Generate the full feature matrix from the processed time-series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, resampled DataFrame with DatetimeIndex.
    target_col : str
        Name of the target column (ac_power).
    lag_periods : list[int]
        Lag offsets in resampled periods (e.g. [1,2,4,8,96]).
    rolling_windows : list[int]
        Window sizes for rolling statistics.
    cache_path : str | None
        Parquet cache path. Pass None to disable caching.
    force_rebuild : bool
        Ignore cache and recompute.

    Returns
    -------
    pd.DataFrame  — feature matrix (NaN rows from lags/rolling dropped)
    """
    from pathlib import Path

    if cache_path and Path(cache_path).exists() and not force_rebuild:
        log.info("Loading cached features from %s", cache_path)
        return pd.read_parquet(cache_path)

    if lag_periods is None:
        lag_periods = [1, 2, 4, 8, 16, 32, 48, 96]
    if rolling_windows is None:
        rolling_windows = [4, 8, 16, 48, 96]

    feat = df.copy()

    # ── 1. Temporal features ─────────────────────────────────────────────────
    idx = feat.index
    feat["hour"]        = idx.hour.astype("float32")
    feat["minute"]      = idx.minute.astype("float32")
    feat["day_of_week"] = idx.dayofweek.astype("float32")
    feat["day_of_year"] = idx.dayofyear.astype("float32")
    feat["month"]       = idx.month.astype("float32")
    feat["week"]        = idx.isocalendar().week.astype("float32")

    # Cyclic encodings (prevent discontinuity at boundary)
    for col, period in [
        ("hour",        24.0),
        ("day_of_year", 365.25),
        ("month",       12.0),
        ("day_of_week", 7.0),
    ]:
        feat = pd.concat([feat, encode_cyclic(feat[col], period)], axis=1)

    # ── 2. Calendar flags ────────────────────────────────────────────────────
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype("int8")
    feat["season"] = feat["month"].map(
        {12: 0, 1: 0, 2: 0,     # winter
         3: 1, 4: 1, 5: 1,      # spring
         6: 2, 7: 2, 8: 2,      # summer
         9: 3, 10: 3, 11: 3}    # autumn
    ).astype("int8")

    # ── 3. Lag features ──────────────────────────────────────────────────────
    for lag in lag_periods:
        feat[f"lag_{lag}"] = feat[target_col].shift(lag).astype("float32")

    # ── 4. Rolling statistics ────────────────────────────────────────────────
    for w in rolling_windows:
        r = feat[target_col].rolling(w, min_periods=max(1, w // 2))
        feat[f"roll_mean_{w}"] = r.mean().astype("float32")
        feat[f"roll_std_{w}"]  = r.std().astype("float32")
        feat[f"roll_max_{w}"]  = r.max().astype("float32")

    # ── 5. Domain physics features ───────────────────────────────────────────
    eps = 1e-6
    irr_clip = feat.get("poa_irradiance_clipped", feat["poa_irradiance__313"].clip(0))

    feat["efficiency_ratio"] = (
        feat[target_col] / (irr_clip + eps)
    ).astype("float32")

    feat["temp_delta"] = (
        feat["module_temp_1__321"] - feat["ambient_temp__320"]
    ).astype("float32")

    feat["dc_ac_ratio"] = (
        feat["dc_power__314"] / (feat[target_col] + eps)
    ).astype("float32").clip(-10, 10)

    feat["ac_apparent_power"] = (
        feat["ac_current__319"] * feat["ac_voltage__318"]
    ).astype("float32")

    # ── 6. Drop rows with NaN introduced by lags / rolling ──────────────────
    n_before = len(feat)
    feat = feat.dropna()
    log.info("Rows dropped due to lag/rolling NaNs: %d → %d", n_before, len(feat))

    # ── 7. Cache ─────────────────────────────────────────────────────────────
    if cache_path:
        import os
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        feat.to_parquet(cache_path, engine="pyarrow", compression="snappy")
        log.info("Feature matrix saved to %s  (shape: %s)", cache_path, feat.shape)

    log.info("Feature matrix shape: %s", feat.shape)
    return feat


# ─── Sequence generator for LSTM / GRU ───────────────────────────────────────

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> tuple:
    """
    Convert a 2-D feature array into overlapping 3-D windows for RNN training.

    Parameters
    ----------
    X : np.ndarray  shape (N, F)
    y : np.ndarray  shape (N,)
    seq_len : int   number of historical time steps in each input window
    horizon : int   number of future steps to predict (multi-step)

    Returns
    -------
    X_seq : np.ndarray  (samples, seq_len, F)
    y_seq : np.ndarray  (samples, horizon)
    """
    n_samples = len(X) - seq_len - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows ({len(X)}) for seq_len={seq_len} + horizon={horizon}"
        )

    X_seq = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(seq_len, X.shape[1])
    )
    # sliding_window_view returns (N-seq_len+1, 1, seq_len, F) → squeeze axis 1
    X_seq = X_seq[:n_samples, 0, :, :]   # (samples, seq_len, F)

    y_seq = np.array([
        y[i + seq_len : i + seq_len + horizon]
        for i in range(n_samples)
    ])  # (samples, horizon)

    return X_seq.astype(np.float32), y_seq.astype(np.float32)
