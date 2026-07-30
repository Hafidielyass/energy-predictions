"""
Data loading, cleaning, and resampling for the PV plant dataset.

Design decisions
----------------
* Raw data is 1-minute resolution (1.57 M rows, 259 MB CSV).
  We resample to 15-minute periods during loading to reduce memory by 15x
  while retaining sub-hourly dynamics critical for ramp detection.
* Night-time records (poa_irradiance < threshold) are kept in the full
  dataset but flagged, so the forecaster can learn the zero-power baseline.
* Negative physical values (e.g., irradiance < 0) are clipped to 0 for
  derived features but stored raw for anomaly-detection signals.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────────

SENSOR_COLS = [
    "ac_current__319",
    "ac_power__315",
    "ac_voltage__318",
    "ambient_temp__320",
    "das_battery_voltage__326",
    "das_temp__325",
    "dc_pos_current__317",
    "dc_pos_voltage__316",
    "dc_power__314",
    "inverter_temp__324",
    "module_temp_1__321",
    "module_temp_2__322",
    "module_temp_3__323",
    "poa_irradiance__313",
    "power_factor__327",
]

TARGET = "ac_power__315"
TIMESTAMP = "measured_on"


# ─── Core loader ─────────────────────────────────────────────────────────────

def load_raw(
    csv_path: str = "input/dataset.csv",
    resample_freq: str = "15min",
    cache_path: Optional[str] = "data/processed/processed_data.parquet",
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load and clean the raw PV dataset.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.
    resample_freq : str
        Pandas offset alias for resampling (e.g. ``"15min"``, ``"1h"``).
        Set to ``"1min"`` to keep original 1-minute resolution.
    cache_path : str | None
        If provided, the processed DataFrame is cached as Parquet for
        fast subsequent loads.
    force_reload : bool
        If True, ignore the Parquet cache and re-process the CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned, resampled DataFrame indexed by ``measured_on``.
    """
    # ── Try cache ────────────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists() and not force_reload:
        log.info("Loading cached processed data from %s", cache_path)
        df = pd.read_parquet(cache_path)
        log.info("Loaded %d rows from cache.", len(df))
        return df

    log.info("Reading raw CSV: %s", csv_path)
    dtype_map = {col: "float32" for col in SENSOR_COLS}
    dtype_map["system_id"] = "int8"

    df = pd.read_csv(
        csv_path,
        parse_dates=[TIMESTAMP],
        dtype=dtype_map,
        low_memory=True,
    )
    log.info("Raw shape: %s", df.shape)

    # ── Sort & deduplicate ───────────────────────────────────────────────────
    df = df.sort_values(TIMESTAMP).drop_duplicates(subset=TIMESTAMP)
    df = df.set_index(TIMESTAMP)
    df.index.name = "measured_on"

    # ── Drop system_id (single-plant dataset) ───────────────────────────────
    df = df.drop(columns=["system_id"], errors="ignore")

    # ── Handle missing values ────────────────────────────────────────────────
    # module_temp_2 has ~2 420 NaNs in the first 500 k rows; interpolate.
    df = df.interpolate(method="time", limit=10)
    df = df.ffill(limit=5).bfill(limit=5)
    null_after = df.isnull().sum().sum()
    if null_after:
        log.warning("%d null values remain after imputation; dropping rows.", null_after)
        df = df.dropna()

    # ── Resample ─────────────────────────────────────────────────────────────
    if resample_freq not in ("1min", "1T"):  # "1T" kept for backward compat
        log.info("Resampling 1-min data to %s ...", resample_freq)
        # Use mean for power/temperature/irradiance (physical averages)
        df = df.resample(resample_freq).mean()
        log.info("Shape after resampling: %s", df.shape)

    # ── Physical plausibility clip ───────────────────────────────────────────
    # Negative ac_power outside sensor noise (< -5 W) → clip to 0
    df[TARGET] = df[TARGET].clip(lower=0.0)
    # Irradiance: keep raw for anomaly detection, but also store clipped
    df["poa_irradiance_clipped"] = df["poa_irradiance__313"].clip(lower=0.0)

    # ── Daylight flag ────────────────────────────────────────────────────────
    df["is_daytime"] = (df["poa_irradiance_clipped"] > 10).astype("int8")

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
        log.info("Processed data saved to %s", cache_path)

    log.info("Final processed shape: %s  |  date range: %s -> %s",
             df.shape, df.index.min(), df.index.max())
    return df


# ─── Train / val / test split ─────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame chronologically (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
    test_frac : float
        Fraction of total data reserved for testing.
    val_frac : float
        Fraction of training data reserved for validation.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    n = len(df)
    n_test = int(n * test_frac)
    n_train_val = n - n_test
    n_val = int(n_train_val * val_frac)
    n_train = n_train_val - n_val

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    log.info("Split sizes — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    log.info("  Train: %s → %s", train.index.min(), train.index.max())
    log.info("  Val  : %s → %s", val.index.min(),   val.index.max())
    log.info("  Test : %s → %s", test.index.min(),  test.index.max())
    return train, val, test


# ─── Daily aggregation helper ─────────────────────────────────────────────────

def build_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sub-daily data to one row per calendar day for anomaly detection.

    Returns
    -------
    pd.DataFrame
        Daily statistics used as input to the fault-detection pipeline.
    """
    daily = df.resample("D").agg(
        daily_energy_kwh   = (TARGET,                   lambda x: x.clip(lower=0).sum() * (15 / 60)),  # kWh (15-min periods)
        peak_power_kw      = (TARGET,                   "max"),
        mean_power_kw      = (TARGET,                   "mean"),
        irradiance_sum     = ("poa_irradiance_clipped",  "sum"),
        irradiance_mean    = ("poa_irradiance_clipped",  "mean"),
        ambient_temp_mean  = ("ambient_temp__320",       "mean"),
        ambient_temp_max   = ("ambient_temp__320",       "max"),
        module_temp_mean   = ("module_temp_1__321",      "mean"),
        inverter_temp_max  = ("inverter_temp__324",      "max"),
        power_factor_mean  = ("power_factor__327",       "mean"),
        ac_voltage_std     = ("ac_voltage__318",         "std"),
        dc_voltage_mean    = ("dc_pos_voltage__316",     "mean"),
        dc_current_mean    = ("dc_pos_current__317",     "mean"),
        power_ramp_std     = (TARGET,                   lambda x: x.diff().abs().std()),
        zero_power_ratio   = (TARGET,                   lambda x: (x < 5).mean()),
        n_records          = (TARGET,                   "count"),
    )

    # Performance Ratio = daily_energy / (irradiance_Wh * STC_factor)
    # We normalise by the site's long-run median to get a unitless PR proxy
    median_irr = daily["irradiance_sum"].replace(0, np.nan).median()
    daily["performance_ratio"] = daily["daily_energy_kwh"] / (
        daily["irradiance_sum"].replace(0, np.nan) / median_irr
    )

    # Efficiency: ratio of AC energy to irradiance energy
    daily["efficiency"] = daily["daily_energy_kwh"] / (
        daily["irradiance_sum"].replace(0, np.nan) * 1e-3 + 1e-9
    )

    # Temp delta
    daily["temp_delta"] = daily["module_temp_mean"] - daily["ambient_temp_mean"]

    # DC-AC conversion efficiency proxy
    daily["dc_ac_ratio"] = (
        daily["dc_current_mean"] * daily["dc_voltage_mean"]
    ) / (daily["mean_power_kw"].replace(0, np.nan) + 1e-9)

    # Curve smoothness: low std in power ramp = smooth bell curve (normal)
    # Already captured in power_ramp_std above.

    daily = daily.dropna(subset=["daily_energy_kwh"])
    log.info("Daily aggregation complete: %d days", len(daily))
    return daily
