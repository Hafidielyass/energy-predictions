# data/features/

This directory stores the **engineered feature matrix** used for model training.

Generated files
---------------

| File | Created by | Description |
|------|-----------|-------------|
| `feature_data.parquet` | `feature_engineering.build_features()` | Full feature matrix (~52 columns). Created once and cached. |

Feature matrix columns (~52 total)
------------------------------------

**Raw sensor readings** (15 columns)
- ac_power__315, poa_irradiance__313, ambient_temp__320, module_temp_1__321, ...

**Temporal cyclic features** (8 columns)
- hour_sin, hour_cos
- day_of_year_sin, day_of_year_cos
- month_sin, month_cos
- day_of_week_sin, day_of_week_cos

**Calendar flags** (2 columns)
- season (0=winter, 1=spring, 2=summer, 3=autumn)
- is_weekend

**Auto-regressive lags** (8 columns)
- lag_1, lag_2, lag_4, lag_8, lag_16, lag_32, lag_48, lag_96

**Rolling statistics** (15 columns)
- roll_mean_4/8/16/48/96
- roll_std_4/8/16/48/96
- roll_max_4/8/16/48/96

**Domain physics features** (4 columns)
- efficiency_ratio = ac_power / (poa_irradiance + ε)
- temp_delta = module_temp_1 - ambient_temp
- dc_ac_ratio = dc_power / (ac_power + ε)
- ac_apparent = ac_current × ac_voltage

How to regenerate
-----------------
```python
from src.features.feature_engineering import build_features
from src.preprocessing.data_loader import load_raw

df = load_raw('input/dataset.csv')
feat = build_features(df, cache_path='data/features/feature_data.parquet')
print(feat.shape)  # (~104000, 52)
```

Force rebuild (ignore cache)
-----------------------------
```python
feat = build_features(df, cache_path='data/features/feature_data.parquet', force_rebuild=True)
```
