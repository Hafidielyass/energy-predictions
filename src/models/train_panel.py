
"""
Train on the provincial panel and compare against the national model.

The national model was capped at seven leaves because ~11,900 daylight rows
could not support more. With eleven times the rows that constraint should
lift, so capacity is re-searched here rather than inherited.

Two evaluations matter and they answer different questions:

  per-province  — against Elia's own forecast for each province, which is the
                  harder task, since a single province is not smoothed by
                  spatial averaging

  national      — provincial predictions summed back to a country total and
                  compared with Elia's national forecast. This is the number
                  directly comparable to everything measured so far.

Run:
    python -m src.models.train_panel
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.build_features import TARGET_COL
from src.features.build_panel_features import PANEL_FEATURE_COLS, REGION_COL
from src.preprocessing.split import VAL_END, chronological_split, rolling_origin_folds

FEATURES_PATH = "data/dayahead/panel_features.parquet"
MODEL_DIR = "models/dayahead"

# Capacity-focused search: the national tuning already settled the other knobs,
# and leaf count is the parameter data volume actually constrains.
CANDIDATES = [
    {"num_leaves": 7, "min_child_samples": 80},
    {"num_leaves": 31, "min_child_samples": 80},
    {"num_leaves": 63, "min_child_samples": 100},
    {"num_leaves": 127, "min_child_samples": 150},
    {"num_leaves": 255, "min_child_samples": 200},
]
BASE_PARAMS = {
    "objective": "l2", "learning_rate": 0.05, "reg_lambda": 1.0,
    "colsample_bytree": 0.9, "subsample": 0.8, "subsample_freq": 1,
    "random_state": 42, "verbose": -1,
}


def _daylight(frame):
    return frame[frame["is_day"] == 1]


def _nrmse(y, pred):
    return float(100 * np.sqrt(((y - pred) ** 2).mean()))


def main() -> None:
    panel = pd.read_parquet(FEATURES_PATH)
    train, val, test = chronological_split(panel)
    dev = panel[panel.index < pd.Timestamp(VAL_END, tz="UTC")]

    dev_day, test_day = _daylight(dev), _daylight(test)
    print(f"daylight rows — development {len(dev_day):,} | test {len(test_day):,}\n")

    # ── Search leaf capacity on rolling-origin folds ──────────────────────
    folds = rolling_origin_folds(panel)
    print(f"tuning capacity on {len(folds)} folds:")

    scored = []
    for candidate in CANDIDATES:
        fold_scores = []
        for fold_train, fold_val in folds:
            ft, fv = _daylight(fold_train), _daylight(fold_val)
            model = lgb.LGBMRegressor(n_estimators=3000, **BASE_PARAMS, **candidate)
            model.fit(
                ft[PANEL_FEATURE_COLS], ft[TARGET_COL],
                eval_X=fv[PANEL_FEATURE_COLS], eval_y=fv[TARGET_COL],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(0)],
            )
            pred = model.predict(fv[PANEL_FEATURE_COLS]).clip(0, 1)
            fold_scores.append((_nrmse(fv[TARGET_COL], pred), model.best_iteration_ or 200))

        mean_score = float(np.mean([s for s, _ in fold_scores]))
        mean_trees = int(np.mean([t for _, t in fold_scores]))
        scored.append((mean_score, mean_trees, candidate))
        print(f"  leaves={candidate['num_leaves']:>4}  CV nRMSE {mean_score:.4f}%  "
              f"({mean_trees} trees)")

    best_score, best_trees, best_params = min(scored, key=lambda x: x[0])
    print(f"\nbest: {best_params}  ({best_trees} trees, CV {best_score:.4f}%)\n")

    # ── Final fit ─────────────────────────────────────────────────────────
    X_dev, y_dev = dev_day[PANEL_FEATURE_COLS], dev_day[TARGET_COL]
    X_test, y_test = test_day[PANEL_FEATURE_COLS], test_day[TARGET_COL]

    lgbm = lgb.LGBMRegressor(
        n_estimators=int(best_trees * 1.15), **BASE_PARAMS, **best_params
    ).fit(X_dev, y_dev)
    lgbm_pred = pd.Series(lgbm.predict(X_test).clip(0, 1), index=y_test.index)

    scaler = StandardScaler().fit(X_dev)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(X_dev), y_dev)
    ridge_pred = pd.Series(
        ridge.predict(scaler.transform(X_test)).clip(0, 1), index=y_test.index)

    _per_province(test_day, y_test, lgbm_pred, ridge_pred)
    _national(test_day, lgbm_pred, ridge_pred)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lgbm, "features": PANEL_FEATURE_COLS},
                f"{MODEL_DIR}/lgbm_panel.pkl")
    with open(f"{MODEL_DIR}/panel_params.json", "w", encoding="utf-8") as f:
        json.dump({**BASE_PARAMS, **best_params,
                   "n_estimators": int(best_trees * 1.15)}, f, indent=2)
    print(f"\nSaved -> {MODEL_DIR}/lgbm_panel.pkl")


def _per_province(test_day, y_test, lgbm_pred, ridge_pred) -> None:
    """Each province against Elia's own forecast for that province."""
    print("PER-PROVINCE (test year, daylight, % of each province's capacity)")
    print("=" * 74)
    print(f"{'province':<18}{'Elia':>9}{'LightGBM':>11}{'Ridge':>9}{'skill LGBM':>13}")
    print("-" * 74)

    rows = []
    for region, group in test_day.groupby(REGION_COL, observed=True):
        idx = group.index
        mask = test_day[REGION_COL] == region
        elia = _nrmse(group[TARGET_COL], group["elia_dayahead_cf"])
        lg = _nrmse(group[TARGET_COL], lgbm_pred[mask.to_numpy()])
        rd = _nrmse(group[TARGET_COL], ridge_pred[mask.to_numpy()])
        rows.append((region, elia, lg, rd))
        print(f"{region:<18}{elia:>8.2f}%{lg:>10.2f}%{rd:>8.2f}%"
              f"{100 * (1 - lg / elia):>12.2f}%")

    elia_mean = np.mean([r[1] for r in rows])
    lgbm_mean = np.mean([r[2] for r in rows])
    print("-" * 74)
    print(f"{'mean':<18}{elia_mean:>8.2f}%{lgbm_mean:>10.2f}%"
          f"{np.mean([r[3] for r in rows]):>8.2f}%"
          f"{100 * (1 - lgbm_mean / elia_mean):>12.2f}%")


def _national(test_day, lgbm_pred, ridge_pred) -> None:
    """
    Provincial forecasts summed back to a country total.

    This is the figure comparable with every national result measured so far:
    the same target, the same benchmark, reached by a different route.
    """
    cap = test_day["monitoredcapacity"]
    frame = pd.DataFrame({
        "actual_mw": test_day[TARGET_COL] * cap,
        "elia_mw": test_day["elia_dayahead_cf"] * cap,
        "lgbm_mw": lgbm_pred * cap,
        "ridge_mw": ridge_pred * cap,
        "capacity_mw": cap,
    })
    national = frame.groupby(level=0).sum()
    national = national[national["capacity_mw"] > 0]

    cap_n = national["capacity_mw"]
    print("\n\nNATIONAL (provinces summed, test year, % of national capacity)")
    print("=" * 74)

    base = _nrmse(national["actual_mw"] / cap_n, national["elia_mw"] / cap_n)
    print(f"{'model':<26}{'nMAE':>9}{'nRMSE':>9}{'skill vs Elia':>16}")
    print("-" * 74)
    for label, column in [("Elia day-ahead", "elia_mw"),
                          ("LightGBM panel", "lgbm_mw"),
                          ("Ridge panel", "ridge_mw")]:
        err = (national["actual_mw"] - national[column]) / cap_n
        rmse = float(100 * np.sqrt((err ** 2).mean()))
        print(f"{label:<26}{100 * err.abs().mean():>8.3f}%{rmse:>8.3f}%"
              f"{100 * (1 - rmse / base):>15.2f}%")

    print("\nfor reference, the national-only model measured earlier:")
    print(f"{'Elia day-ahead':<26}{2.788:>8.3f}%{4.191:>8.3f}%{0.0:>15.2f}%")
    print(f"{'Ridge (national model)':<26}{2.832:>8.3f}%{4.197:>8.3f}%{-0.14:>15.2f}%")


if __name__ == "__main__":
    main()
