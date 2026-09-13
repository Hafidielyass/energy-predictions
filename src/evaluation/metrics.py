"""
Scoring for day-ahead forecasts.

Everything is expressed as a percentage of installed capacity. The Belgian
fleet grew from 8.8 to 12.1 GW over the study window, so an error in raw
megawatts means something different in 2024 than in 2026; normalising makes
periods comparable and matches how the solar forecasting literature reports
results.

Metrics are computed on daylight hours only. Roughly half of all rows are
night, where both the forecast and the truth are zero — including them adds
thousands of trivially correct predictions that inflate R2 and shrink mean
error without the model having done anything.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def score(
    y_true: pd.Series,
    y_pred: pd.Series,
    is_day: Optional[pd.Series] = None,
    label: str = "model",
) -> Dict[str, float]:
    """
    Compute normalised error metrics for capacity-factor predictions.

    Inputs are capacity factors (0-1), so an absolute error of 0.03 is 3% of
    installed capacity and the metrics are already normalised by construction.
    """
    if is_day is not None:
        mask = is_day == 1
        y_true, y_pred = y_true[mask], y_pred[mask]

    err = y_true - y_pred
    return {
        "label": label,
        "n": int(len(y_true)),
        "nMAE": float(100 * err.abs().mean()),
        "nRMSE": float(100 * np.sqrt((err ** 2).mean())),
        "bias": float(100 * err.mean()),
        "R2": float(1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
    }


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """
    Fractional reduction in RMSE against a reference forecast.

    Positive means better than the reference, zero means indistinguishable,
    negative means worse. Reported against Elia's operational forecast, which
    is the only comparison that says anything about real-world usefulness —
    beating a naive persistence model would not.
    """
    return float(1.0 - model_rmse / baseline_rmse)


def report(results: list, baseline_label: str = "Elia day-ahead") -> pd.DataFrame:
    """Render a comparison table, with skill measured against the baseline row."""
    frame = pd.DataFrame(results).set_index("label")

    if baseline_label in frame.index:
        base_rmse = frame.loc[baseline_label, "nRMSE"]
        frame["skill_vs_elia"] = [
            skill_score(r, base_rmse) for r in frame["nRMSE"]
        ]

    display = frame.copy()
    for col in ("nMAE", "nRMSE", "bias"):
        display[col] = display[col].map(lambda v: f"{v:6.3f}")
    display["R2"] = display["R2"].map(lambda v: f"{v:.4f}")
    if "skill_vs_elia" in display.columns:
        display["skill_vs_elia"] = display["skill_vs_elia"].map(lambda v: f"{100*v:+6.2f}%")

    print(display.to_string())
    return frame
