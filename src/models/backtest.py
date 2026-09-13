"""
Walk-forward backtest: does periodic retraining beat fitting once?

Elia's own forecast improved by roughly 10% between the development period and
the test year, so a model trained once on 2024 residuals is correcting mistakes
its target has since stopped making. This simulates operating the system month
by month and compares three retraining policies.

Every prediction is made by a model fitted only on data preceding it, so
walking through the test year this way is an honest deployment simulation
rather than a second look at held-out data.

Policies
--------
static    fit once on the development period, never updated
expanding fit on everything available up to the current month
rolling   fit on a trailing 12-month window only

Run:
    python -m src.models.backtest
"""

import json
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLS, TARGET_COL
from src.preprocessing.split import VAL_END

FEATURES_PATH = "data/dayahead/features.parquet"
PARAMS_PATH = "models/dayahead/best_params.json"

ROLLING_MONTHS = 12
TREE_SCALE = 1.15


def _fit(train: pd.DataFrame, params: Dict, n_trees: int) -> lgb.LGBMRegressor:
    day = train[train["is_day"] == 1]
    return lgb.LGBMRegressor(
        objective="l2", n_estimators=n_trees, random_state=42,
        subsample_freq=1, verbose=-1, **params,
    ).fit(day[FEATURE_COLS], day[TARGET_COL])


def _nrmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(100 * np.sqrt(((y_true - y_pred) ** 2).mean()))


def main() -> pd.DataFrame:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        chosen = json.load(f)
    params = {k: v for k, v in chosen["lightgbm"].items()
              if k not in ("cv_nrmse", "n_estimators")}
    n_trees = int(chosen["lightgbm"]["n_estimators"] * TREE_SCALE)

    df = pd.read_parquet(FEATURES_PATH)
    dev_end = pd.Timestamp(VAL_END, tz="UTC")
    dev = df[df.index < dev_end]

    # One model fitted once, as the point-forecast script does.
    static_model = _fit(dev, params, n_trees)

    months = sorted({(ts.year, ts.month) for ts in df[df.index >= dev_end].index})
    rows: List[Dict] = []

    print(f"Walking {len(months)} months, retraining monthly\n")
    print(f"{'month':<9} {'n':>5} {'Elia':>8} {'static':>8} {'expand':>8} {'roll12':>8}")
    print("-" * 54)

    for year, month in months:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        stop = start + pd.DateOffset(months=1)

        current = df[(df.index >= start) & (df.index < stop)]
        current_d = current[current["is_day"] == 1]
        if len(current_d) < 50:
            continue

        history = df[df.index < start]
        rolling_history = history[history.index >= start - pd.DateOffset(months=ROLLING_MONTHS)]

        X, y = current_d[FEATURE_COLS], current_d[TARGET_COL]

        scores = {
            "Elia": _nrmse(y, current_d["elia_dayahead_cf"].to_numpy()),
            "static": _nrmse(y, static_model.predict(X).clip(0, 1)),
            "expand": _nrmse(y, _fit(history, params, n_trees).predict(X).clip(0, 1)),
            "roll12": _nrmse(y, _fit(rolling_history, params, n_trees).predict(X).clip(0, 1)),
        }
        scores.update(month=f"{year}-{month:02d}", n=len(current_d))
        rows.append(scores)

        print(f"{scores['month']:<9} {scores['n']:>5} {scores['Elia']:>8.3f} "
              f"{scores['static']:>8.3f} {scores['expand']:>8.3f} {scores['roll12']:>8.3f}")

    frame = pd.DataFrame(rows).set_index("month")
    _summary(frame)
    return frame


def _summary(frame: pd.DataFrame) -> None:
    """Aggregate by row count so long months are not under-weighted."""
    weights = frame["n"]
    print("\nWeighted mean nRMSE across the test year:")

    elia = float((frame["Elia"] * weights).sum() / weights.sum())
    for policy in ("Elia", "static", "expand", "roll12"):
        value = float((frame[policy] * weights).sum() / weights.sum())
        skill = 100 * (1 - value / elia)
        wins = int((frame[policy] < frame["Elia"]).sum()) if policy != "Elia" else 0
        extra = "" if policy == "Elia" else f"   beats Elia in {wins}/{len(frame)} months"
        print(f"  {policy:<8} {value:6.3f}%   skill vs Elia {skill:+6.2f}%{extra}")

    best = min(("static", "expand", "roll12"),
               key=lambda p: (frame[p] * weights).sum())
    print(f"\nBest retraining policy: {best}")


if __name__ == "__main__":
    main()
