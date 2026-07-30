"""
src.features — Feature Engineering
=====================================

Modules
-------
feature_engineering : Build the full feature matrix from processed time-series data.

Feature families
----------------
1. **Temporal cyclic**      — sin/cos encoding of hour, day-of-year, month, weekday
2. **Calendar flags**       — season (0-3), is_weekend
3. **Auto-regressive lags** — lag_1 … lag_96 (15-min periods back)
4. **Rolling statistics**   — mean, std, max over 4 / 8 / 16 / 48 / 96 windows
5. **Domain physics**       — efficiency_ratio, temp_delta, dc_ac_ratio, ac_apparent_power

Key functions
-------------
build_features(df, target_col, lag_periods, rolling_windows, cache_path)
    Generate the full feature matrix. Caches result as Parquet.

make_sequences(X, y, seq_len, horizon)
    Slide a window of ``seq_len`` steps to produce (N, seq_len, F) arrays
    for GRU training.

encode_cyclic(series, period)
    Encode a periodic scalar as sin + cos pair.

Scientific justification
------------------------
Cyclic encodings prevent the model treating hour 23 and hour 0 as maximally
different. Lag_96 (same time 24 h ago) captures the dominant daily periodicity
and is consistently the top-3 SHAP feature across all runs.
"""

from src.features.feature_engineering import build_features, make_sequences, encode_cyclic

__all__ = ["build_features", "make_sequences", "encode_cyclic"]
