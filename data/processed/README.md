# data/processed/

This directory is **auto-populated** when you run the pipeline.

Generated files
---------------

| File | Created by | Description |
|------|-----------|-------------|
| `processed_data.parquet` | `data_loader.load_raw()` | Cleaned, 15-min resampled dataset. ~105k rows, ~14 MB. |
| `daily_features.parquet` | `data_loader.build_daily_df()` | One row per calendar day with aggregated KPIs (~1400 rows). |
| `daily_anomaly_results.parquet` | `HybridAnomalyDetector.predict()` | Daily anomaly scores + severity labels. Used by the Streamlit dashboard. |
| `forecast_results.parquet` | Notebook cell 14 | GRU predictions vs actuals on the test set. Used by the Streamlit dashboard. |
| `batch_forecast_*.parquet` | `BatchScorer.score_range()` | Rolling batch forecasts over a date range. |
| `batch_anomaly_*.parquet` | `BatchScorer.score_range()` | Anomaly scores for a scored date range. |

How to regenerate
-----------------
```bash
# Option 1: Run the full Jupyter notebook
jupyter notebook notebooks/solar_pv_complete_pipeline.ipynb

# Option 2: Python script
python -c "
from src.preprocessing.data_loader import load_raw, build_daily_df
df = load_raw('input/dataset.csv', resample_freq='15T')
daily = build_daily_df(df)
daily.to_parquet('data/processed/daily_features.parquet')
print('Done.')
"
```

Parquet format
--------------
All files use **Apache Parquet with Snappy compression** via `pyarrow`.
Load with: `pd.read_parquet('data/processed/processed_data.parquet')`
