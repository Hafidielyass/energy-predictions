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


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    """
    The proper scoring rule for a single quantile forecast.

    Under-prediction is penalised by `quantile` and over-prediction by
    `1 - quantile`, so a P90 model is punished nine times harder for coming in
    under the truth than over it. That asymmetry is what makes the model learn
    an actual upper bound rather than drifting back toward the mean.
    """
    err = y_true - y_pred
    return float(100 * np.maximum(quantile * err, (quantile - 1) * err).mean())


def coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """
    Fraction of observations that landed inside the interval.

    A P10-P90 band should contain 80% of outcomes. Materially below that and
    the interval is lying about its confidence; materially above and it is
    padded so wide it carries no information.
    """
    return float(100 * ((y_true >= lower) & (y_true <= upper)).mean())


def sharpness(lower: pd.Series, upper: pd.Series) -> float:
    """
    Mean interval width, as a percentage of installed capacity.

    Only meaningful alongside coverage: any model can achieve perfect coverage
    by predicting "somewhere between zero and maximum". Narrow *and* correctly
    covered is the goal.
    """
    return float(100 * (upper - lower).mean())


def score_intervals(
    y_true: pd.Series,
    lower: pd.Series,
    median: pd.Series,
    upper: pd.Series,
    is_day: Optional[pd.Series] = None,
    label: str = "model",
) -> Dict[str, float]:
    """Combined calibration and sharpness summary for a P10/P50/P90 forecast."""
    if is_day is not None:
        mask = is_day == 1
        y_true, lower, median, upper = (
            y_true[mask], lower[mask], median[mask], upper[mask]
        )

    return {
        "label": label,
        "n": int(len(y_true)),
        "coverage_%": coverage(y_true, lower, upper),
        "width_%": sharpness(lower, upper),
        "pinball_P10": pinball_loss(y_true, lower, 0.10),
        "pinball_P50": pinball_loss(y_true, median, 0.50),
        "pinball_P90": pinball_loss(y_true, upper, 0.90),
        "pinball_mean": float(np.mean([
            pinball_loss(y_true, lower, 0.10),
            pinball_loss(y_true, median, 0.50),
            pinball_loss(y_true, upper, 0.90),
        ])),
    }


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
