"""
Probabilistic day-ahead forecasting via quantile regression.

A single number ("6,100 MW at noon tomorrow") is not actionable on its own —
whoever schedules reserve capacity or bids into a market needs to know how
wrong it might be. This fits three LightGBM models under a quantile objective
to produce a P10/P50/P90 band, and scores them the way probabilistic forecasts
have to be scored: calibration (does the band contain the truth as often as it
claims?) and sharpness (is it narrow enough to be useful?), never one alone.

Elia publishes its own confidence bands in the same dataset, so the comparison
is against a real operational product rather than an arbitrary target.

Run:
    python -m src.models.quantile
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import score_intervals
from src.features.build_features import FEATURE_COLS, TARGET_COL
from src.preprocessing.split import VAL_END, chronological_split

FEATURES_PATH = "data/dayahead/features.parquet"
PARAMS_PATH = "models/dayahead/best_params.json"
MODEL_DIR = "models/dayahead"

QUANTILES = {"P10": 0.10, "P50": 0.50, "P90": 0.90}
TREE_SCALE = 1.15


def main() -> pd.DataFrame:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        chosen = json.load(f)

    df = pd.read_parquet(FEATURES_PATH)
    _, _, test = chronological_split(df)

    dev = df[df.index < pd.Timestamp(VAL_END, tz="UTC")]
    dev_d = dev[dev["is_day"] == 1]
    test_d = test[test["is_day"] == 1]

    X_dev, y_dev = dev_d[FEATURE_COLS], dev_d[TARGET_COL]
    X_test, y_test = test_d[FEATURE_COLS], test_d[TARGET_COL]
    print(f"daytime rows — development {len(dev_d):,} | test {len(test_d):,}\n")

    # Reuse the regularisation found for the point model. The search there was
    # over capacity control (7 leaves, min_child 80), which is driven by how
    # little data there is rather than by the loss function, so it transfers.
    base = {k: v for k, v in chosen["lightgbm"].items()
            if k not in ("cv_nrmse", "n_estimators")}
    n_trees = int(chosen["lightgbm"]["n_estimators"] * TREE_SCALE)

    models, preds = {}, {}
    for name, alpha in QUANTILES.items():
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, n_estimators=n_trees,
            random_state=42, subsample_freq=1, verbose=-1, **base,
        ).fit(X_dev, y_dev)
        models[name] = model
        preds[name] = pd.Series(model.predict(X_test).clip(0, 1), index=y_test.index)
        print(f"  fitted {name} (alpha={alpha})")

    lower, median, upper = _enforce_monotonic(preds)

    results = [
        score_intervals(y_test, lower, median, upper, label="LightGBM quantile"),
        score_intervals(
            y_test,
            test_d["dayaheadconfidence10"] / test_d["monitoredcapacity"],
            test_d["elia_dayahead_cf"],
            test_d["dayaheadconfidence90"] / test_d["monitoredcapacity"],
            label="Elia day-ahead band",
        ),
    ]

    print("\nTEST SET — probabilistic scores, daylight hours, % of capacity")
    print("=" * 96)
    frame = pd.DataFrame(results).set_index("label")
    display = frame.copy()
    for col in display.columns:
        if col != "n":
            display[col] = display[col].map(lambda v: f"{v:.3f}")
    print(display.to_string())

    _interpret(frame)
    _coverage_by_season(test_d, y_test, lower, upper)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": FEATURE_COLS, "quantiles": QUANTILES},
                f"{MODEL_DIR}/lgbm_quantile.pkl")
    print(f"\nSaved -> {MODEL_DIR}/lgbm_quantile.pkl")
    return frame


def _enforce_monotonic(preds: Dict[str, pd.Series]):
    """
    Guarantee P10 <= P50 <= P90.

    The three quantiles are fitted independently, so nothing stops the P10
    model from exceeding the P90 model on a given hour — "quantile crossing".
    Sorting each row's predictions is the standard, cheap repair and cannot
    make calibration worse.
    """
    stacked = np.sort(
        np.column_stack([preds["P10"], preds["P50"], preds["P90"]]), axis=1
    )
    index = preds["P10"].index
    crossings = int((
        (preds["P10"] > preds["P50"]) | (preds["P50"] > preds["P90"])
    ).sum())
    if crossings:
        print(f"  repaired {crossings:,} quantile crossings "
              f"({100*crossings/len(index):.2f}% of rows)")

    return (pd.Series(stacked[:, 0], index=index),
            pd.Series(stacked[:, 1], index=index),
            pd.Series(stacked[:, 2], index=index))


def _interpret(frame: pd.DataFrame) -> None:
    """State plainly whether the band is trustworthy and how it compares."""
    ours = frame.loc["LightGBM quantile"]
    elia = frame.loc["Elia day-ahead band"]

    print("\nInterpretation:")
    for label, row in (("ours", ours), ("Elia", elia)):
        gap = row["coverage_%"] - 80.0
        verdict = ("well calibrated" if abs(gap) <= 3
                   else "over-confident (band too narrow)" if gap < 0
                   else "under-confident (band too wide)")
        print(f"  {label:<5} coverage {row['coverage_%']:.1f}% vs 80% target "
              f"({gap:+.1f} pp) — {verdict}; mean width {row['width_%']:.2f}% of capacity")

    better = "ours" if ours["pinball_mean"] < elia["pinball_mean"] else "Elia"
    delta = 100 * (1 - ours["pinball_mean"] / elia["pinball_mean"])
    print(f"  mean pinball loss: ours {ours['pinball_mean']:.3f} vs "
          f"Elia {elia['pinball_mean']:.3f}  -> {better} better ({delta:+.1f}%)")


def _coverage_by_season(test_d, y_true, lower, upper) -> None:
    """Calibration that only holds on average is not calibration."""
    quarters = test_d.index.tz_convert(None).to_period("Q")
    print("\nCoverage by quarter (target 80%):")
    for q in sorted(set(quarters)):
        mask = quarters == q
        cov = 100 * ((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])).mean()
        width = 100 * (upper[mask] - lower[mask]).mean()
        print(f"  {str(q):<8} n={int(mask.sum()):>5}  coverage {cov:5.1f}%  "
              f"width {width:5.2f}%")


if __name__ == "__main__":
    main()
