"""
BatchScorer — Offline Batch Scoring Pipeline
=============================================

Scores an entire date range from a Parquet file, producing:
  1. Rolling 24-hour forecast for every day in the range
  2. Daily anomaly scores for every day in the range

Use cases
---------
* Back-testing: evaluate forecast quality on historical data
* Scheduled scoring: nightly job that produces next-day forecasts
* Report generation: monthly anomaly summary for maintenance teams

Output
------
``data/processed/batch_forecast_{start}_{end}.parquet``
``data/processed/batch_anomaly_{start}_{end}.parquet``

Usage
-----
>>> from src.deployment.batch_scorer import BatchScorer
>>> from src.deployment.inference_pipeline import InferencePipeline
>>>
>>> pipeline = InferencePipeline.from_paths(
...     gru_path="models/forecasting/best_gru_model.keras",
...     scaler_path="models/forecasting/feature_scaler.pkl",
... )
>>> scorer = BatchScorer(pipeline, output_dir="data/processed")
>>> fc_df, anomaly_df = scorer.score_range(
...     data_path="data/processed/processed_data.parquet",
...     start_date="2022-01-01",
...     end_date="2022-12-31",
... )
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class BatchScorer:
    """
    Offline batch scoring engine.

    Parameters
    ----------
    pipeline   : InferencePipeline  — loaded and ready inference pipeline
    output_dir : str  — directory where output Parquet files are saved
    seq_len    : int  — lookback window length (must match pipeline.seq_len)
    stride     : int  — step between forecast windows (default = horizon = 96)
    """

    def __init__(
        self,
        pipeline,
        output_dir: str = "data/processed",
        stride: Optional[int] = None,
    ) -> None:
        self.pipeline   = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len    = pipeline.seq_len
        self.horizon    = pipeline.horizon
        self.stride     = stride or self.horizon   # non-overlapping by default

    # ─── Main entry point ─────────────────────────────────────────────────────

    def score_range(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Score all windows in [start_date, end_date].

        Parameters
        ----------
        data_path  : Path to a Parquet or CSV file with a DatetimeIndex.
        start_date : First date to forecast, e.g. ``"2022-01-01"``.
        end_date   : Last date to forecast,  e.g. ``"2022-12-31"``.
        save       : If True, write output Parquet files.

        Returns
        -------
        forecast_df  : pd.DataFrame with columns [timestamp, actual, predicted]
        anomaly_df   : pd.DataFrame with daily anomaly scores (or None)
        """
        log.info("BatchScorer: loading data from %s", data_path)
        df = self._load_data(data_path)

        # Restrict range
        df_range = df.loc[start_date:end_date]
        if len(df_range) < self.seq_len + self.horizon:
            raise ValueError(
                f"Not enough data in [{start_date}, {end_date}] for one forecast window "
                f"(need {self.seq_len + self.horizon}, got {len(df_range)})."
            )

        log.info("Scoring range %s → %s  (%d rows)", start_date, end_date, len(df_range))

        # ── Rolling forecast ──────────────────────────────────────────────────
        forecast_rows = []
        total_windows = (len(df_range) - self.seq_len) // self.stride
        log.info("Total forecast windows: %d", total_windows)

        for i, start_idx in enumerate(range(0, len(df_range) - self.seq_len, self.stride)):
            end_idx   = start_idx + self.seq_len
            window_df = df_range.iloc[start_idx:end_idx]

            try:
                result = self.pipeline.run_forecast(window_df)
                for ts_str, pw in zip(result["timestamps"], result["ac_power_w"]):
                    ts = pd.Timestamp(ts_str)
                    # Look up actual value if available
                    actual = float(df.loc[ts, "ac_power__315"]) if ts in df.index else np.nan
                    forecast_rows.append({
                        "timestamp": ts,
                        "predicted": pw,
                        "actual":    actual,
                    })
            except Exception as exc:
                log.warning("Window %d failed: %s", i, exc)
                continue

            if (i + 1) % max(1, total_windows // 10) == 0:
                log.info("  Progress: %d/%d windows (%.0f%%)", i+1, total_windows,
                         100*(i+1)/total_windows)

        forecast_df = pd.DataFrame(forecast_rows).drop_duplicates("timestamp").set_index("timestamp")
        forecast_df["error"] = forecast_df["actual"] - forecast_df["predicted"]

        # ── Compute aggregate metrics ─────────────────────────────────────────
        valid = forecast_df.dropna()
        if len(valid) > 0:
            mae  = float(np.mean(np.abs(valid["error"])))
            rmse = float(np.sqrt(np.mean(valid["error"] ** 2)))
            mask = valid["actual"] > 1.0
            mape = float(np.mean(np.abs(valid.loc[mask, "error"] / valid.loc[mask, "actual"])) * 100) \
                   if mask.sum() > 0 else np.nan
            log.info("Batch forecast metrics — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
                     mae, rmse, mape)

        # ── Anomaly scoring ───────────────────────────────────────────────────
        anomaly_df = None
        if self.pipeline.anomaly_det is not None:
            from src.preprocessing.data_loader import build_daily_df
            daily_df = build_daily_df(df_range)
            if len(daily_df) > 0:
                anomaly_df = self.pipeline.run_anomaly(daily_df)

        # ── Save ──────────────────────────────────────────────────────────────
        if save:
            tag = f"{start_date}_{end_date}".replace("-", "")
            fc_path = self.output_dir / f"batch_forecast_{tag}.parquet"
            forecast_df.to_parquet(fc_path)
            log.info("Forecast saved to %s", fc_path)

            if anomaly_df is not None:
                an_path = self.output_dir / f"batch_anomaly_{tag}.parquet"
                anomaly_df.to_parquet(an_path)
                log.info("Anomaly results saved to %s", an_path)

        return forecast_df, anomaly_df

    # ─── Scheduled run helper ─────────────────────────────────────────────────

    def score_yesterday(self, data_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Convenience method: score the previous calendar day.
        Designed for a nightly cron job.
        """
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return self.score_range(data_path, yesterday, yesterday)

    def score_last_n_days(
        self, data_path: str, n: int = 30
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Score the last N calendar days."""
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - pd.Timedelta(days=n)).strftime("%Y-%m-%d")
        return self.score_range(data_path, start, end)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path.resolve()}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path, parse_dates=["measured_on"])
            return df.set_index("measured_on").sort_index()
        raise ValueError(f"Unsupported file format: {path.suffix}")
