"""
Final fit and single test-set evaluation.

Hyperparameters arrive already chosen from `src.models.tune`, which searched
rolling-origin folds inside the development period. This script reads the test
year exactly once, with nothing left to decide — that is what makes the number
it prints an estimate of future performance rather than a summary of past fit.

Models are fit on daylight hours only. Night output is zero by physics, not by
prediction; training on those rows teaches nothing and scoring on them flatters
every model equally.

Run:
    python -m src.models.tune     # first, to choose hyperparameters
    python -m src.models.train
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import report, score
from src.features.build_features import FEATURE_COLS, TARGET_COL
from src.preprocessing.split import VAL_END, chronological_split

FEATURES_PATH = "data/dayahead/features.parquet"
PARAMS_PATH = "models/dayahead/best_params.json"
MODEL_DIR = "models/dayahead"

# The CV folds trained on 5-12k rows; the final fit sees the whole development
# period, so slightly more trees are warranted. Scaling the averaged stopping
# point is a heuristic, but it is fixed in advance rather than tuned on test.
TREE_SCALE = 1.15


def main() -> pd.DataFrame:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        chosen = json.load(f)

    df = pd.read_parquet(FEATURES_PATH)
    train, val, test = chronological_split(df)

    # Development = everything before the test year. Folds were carved out of
    # this for tuning; the final models get all of it.
    dev = df[df.index < pd.Timestamp(VAL_END, tz="UTC")]
    dev_d = dev[dev["is_day"] == 1]
    test_d = test[test["is_day"] == 1]
    print(f"daytime rows — development {len(dev_d):,} | test {len(test_d):,}")
    print(f"test period  — {test_d.index.min():%Y-%m-%d} to {test_d.index.max():%Y-%m-%d}\n")

    X_dev, y_dev = dev_d[FEATURE_COLS], dev_d[TARGET_COL]
    X_test, y_test = test_d[FEATURE_COLS], test_d[TARGET_COL]

    results = [score(y_test, test_d["elia_dayahead_cf"], label="Elia day-ahead")]

    # ── B0: clear-sky-scaled persistence ──────────────────────────────────
    ratio = (test_d["clearsky_ghi"] /
             test_d["clearsky_ghi"].shift(48).replace(0, np.nan)).fillna(1.0)
    persistence = (test_d["cf_lag_48h"] * ratio.clip(0.5, 2.0)).clip(0, 1)
    results.append(score(y_test, persistence, label="B0 clearsky-persistence"))

    # ── B1: Ridge ─────────────────────────────────────────────────────────
    alpha = chosen["ridge"]["alpha"]
    scaler = StandardScaler().fit(X_dev)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X_dev), y_dev)
    ridge_pred = pd.Series(
        ridge.predict(scaler.transform(X_test)).clip(0, 1), index=y_test.index
    )
    results.append(score(y_test, ridge_pred, label=f"B1 Ridge (a={alpha})"))

    # ── B2: LightGBM ──────────────────────────────────────────────────────
    params = {k: v for k, v in chosen["lightgbm"].items()
              if k not in ("cv_nrmse", "n_estimators")}
    n_trees = int(chosen["lightgbm"]["n_estimators"] * TREE_SCALE)

    lgbm = lgb.LGBMRegressor(
        objective="l2", n_estimators=n_trees, random_state=42,
        subsample_freq=1, verbose=-1, **params,
    ).fit(X_dev, y_dev)
    lgbm_pred = pd.Series(lgbm.predict(X_test).clip(0, 1), index=y_test.index)
    results.append(score(y_test, lgbm_pred, label="B2 LightGBM (tuned)"))
    print(f"LightGBM: {n_trees} trees, {params}\n")

    print("TEST SET — daylight hours, errors as % of installed capacity")
    print("=" * 82)
    frame = report(results)

    _stability(test_d, y_test, {"Ridge": ridge_pred, "LightGBM": lgbm_pred})
    _shuffle_test(lgbm, X_test, y_test)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lgbm, "features": FEATURE_COLS}, f"{MODEL_DIR}/lgbm.pkl")
    joblib.dump({"model": ridge, "scaler": scaler, "features": FEATURE_COLS},
                f"{MODEL_DIR}/ridge.pkl")
    print(f"\nSaved -> {MODEL_DIR}/")
    return frame


def _stability(test_d: pd.DataFrame, y_true: pd.Series, preds: dict) -> None:
    """
    Skill against Elia, broken out by quarter.

    An aggregate improvement can hide a model that wins big in one season and
    loses in another — which would be worthless operationally, since a
    forecaster cannot choose which months to run. Consistency across quarters
    matters more than the headline average.
    """
    elia = test_d["elia_dayahead_cf"]
    quarters = test_d.index.tz_convert(None).to_period("Q")

    print("\nSkill vs Elia by quarter (positive = better than Elia):")
    header = f"{'quarter':<10} {'n':>6} " + "".join(f"{k:>14}" for k in preds)
    print(header)
    print("-" * len(header))

    for q in sorted(set(quarters)):
        mask = quarters == q
        base = np.sqrt(((y_true[mask] - elia[mask]) ** 2).mean())
        row = f"{str(q):<10} {int(mask.sum()):>6} "
        for pred in preds.values():
            rmse = np.sqrt(((y_true[mask] - pred[mask]) ** 2).mean())
            row += f"{100*(1 - rmse/base):>13.2f}%"
        print(row)


def _shuffle_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Leakage check: destroy the weather signal and confirm accuracy collapses.

    A model that scores nearly as well on shuffled forecasts as on real ones is
    not using the weather — it is reading the diurnal cycle off the calendar and
    solar-geometry columns, and would fail the moment conditions departed from
    the seasonal norm.
    """
    weather = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
               "direct_normal_irradiance", "cloud_cover", "clearsky_index",
               "physical_yield", "elia_dayahead_cf"]

    shuffled = X_test.copy()
    order = np.random.default_rng(42).permutation(len(shuffled))
    for col in weather:
        shuffled[col] = shuffled[col].to_numpy()[order]

    intact = score(y_test, pd.Series(model.predict(X_test).clip(0, 1), index=y_test.index))
    broken = score(y_test, pd.Series(model.predict(shuffled).clip(0, 1), index=y_test.index))
    ratio = broken["nRMSE"] / intact["nRMSE"]

    verdict = "PASS" if ratio > 1.5 else "SUSPICIOUS — model may ignore weather"
    print(f"\nShuffled-weather leakage test: {intact['nRMSE']:.3f}% -> "
          f"{broken['nRMSE']:.3f}% ({ratio:.2f}x worse) -> {verdict}")


if __name__ == "__main__":
    main()
