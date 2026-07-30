"""
src.preprocessing — Data Loading & Preprocessing
==================================================

Modules
-------
data_loader : Load raw CSV, clean, resample to 15-min, split, build daily aggregates

Key functions
-------------
load_raw(csv_path, resample_freq, cache_path)
    Load and clean the PV dataset. Caches result as Parquet.
    Returns a DatetimeIndex-indexed DataFrame.

chronological_split(df, test_frac, val_frac)
    Split a time-indexed DataFrame into train / val / test without shuffling.

build_daily_df(df)
    Aggregate 15-min data to one row per calendar day.
    Used as input to the anomaly detection pipeline.

Design notes
------------
* Raw data is 1.57 M rows at 1-min resolution (259 MB CSV).
  Resampling to 15-min reduces memory 15× while preserving sub-hourly dynamics.
* Physical plausibility clipping: ac_power clipped to ≥ 0.
* Night-time records are kept but flagged with ``is_daytime`` column.
"""

from src.preprocessing.data_loader import load_raw, chronological_split, build_daily_df

__all__ = ["load_raw", "chronological_split", "build_daily_df"]
