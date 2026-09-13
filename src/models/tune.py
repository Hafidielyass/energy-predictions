"""
Hyperparameter search over rolling-origin folds.

Selection happens entirely within the development period. The test year is not
read here at all — once a hyperparameter has been chosen because it improved a
test score, that score stops being an estimate of future performance and
becomes a description of the past.

The search space leans hard toward regularisation. With roughly 5-12k daylight
training rows, the default LightGBM configuration has far more capacity than
the data can support, which is the most likely reason the untuned model lost to
plain Ridge.

Run:
    python -m src.models.tune
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLS, TARGET_COL
from src.preprocessing.split import rolling_origin_folds

FEATURES_PATH = "data/dayahead/features.parquet"
PARAMS_PATH = "models/dayahead/best_params.json"

N_TRIALS = 40
SEED = 42

SEARCH_SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [20, 40, 80, 150],
    "learning_rate": [0.02, 0.05, 0.1],
    "reg_lambda": [0.1, 1.0, 10.0, 50.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "subsample": [0.7, 0.9],
}

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _daytime(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["is_day"] == 1]


def _nrmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(100 * np.sqrt(((y_true - y_pred) ** 2).mean()))


def cv_lightgbm(folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
                params: Dict) -> Tuple[float, int]:
    """Mean nRMSE across folds, plus the mean stopping iteration."""
    scores, iterations = [], []

    for train, val in folds:
        train_d, val_d = _daytime(train), _daytime(val)
        model = lgb.LGBMRegressor(
            objective="l2", n_estimators=3000, random_state=SEED,
            subsample_freq=1, verbose=-1, **params,
        )
        model.fit(
            train_d[FEATURE_COLS], train_d[TARGET_COL],
            eval_X=val_d[FEATURE_COLS], eval_y=val_d[TARGET_COL],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(val_d[FEATURE_COLS]).clip(0, 1)
        scores.append(_nrmse(val_d[TARGET_COL], pred))
        iterations.append(model.best_iteration_ or 100)

    return float(np.mean(scores)), int(np.mean(iterations))


def cv_ridge(folds: List[Tuple[pd.DataFrame, pd.DataFrame]], alpha: float) -> float:
    scores = []
    for train, val in folds:
        train_d, val_d = _daytime(train), _daytime(val)
        scaler = StandardScaler().fit(train_d[FEATURE_COLS])
        model = Ridge(alpha=alpha).fit(
            scaler.transform(train_d[FEATURE_COLS]), train_d[TARGET_COL]
        )
        pred = model.predict(scaler.transform(val_d[FEATURE_COLS])).clip(0, 1)
        scores.append(_nrmse(val_d[TARGET_COL], pred))
    return float(np.mean(scores))


def main() -> Dict:
    df = pd.read_parquet(FEATURES_PATH)
    folds = rolling_origin_folds(df)
    print(f"{len(folds)} rolling-origin folds\n")

    # ── Elia's forecast, scored on the same folds as a reference point ────
    elia = float(np.mean([
        _nrmse(_daytime(v)[TARGET_COL], _daytime(v)["elia_dayahead_cf"].to_numpy())
        for _, v in folds
    ]))
    print(f"Elia day-ahead, CV nRMSE: {elia:.4f}%\n")

    # ── Ridge ─────────────────────────────────────────────────────────────
    print("Ridge alpha search:")
    ridge_scores = {a: cv_ridge(folds, a) for a in RIDGE_ALPHAS}
    for alpha, s in ridge_scores.items():
        print(f"  alpha={alpha:<7} CV nRMSE {s:.4f}%")
    best_alpha = min(ridge_scores, key=ridge_scores.get)
    print(f"  -> best alpha {best_alpha} ({ridge_scores[best_alpha]:.4f}%)\n")

    # ── LightGBM random search ────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    seen, trials = set(), []

    print(f"LightGBM random search ({N_TRIALS} trials):")
    while len(trials) < N_TRIALS:
        params = {k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        cv_score, n_trees = cv_lightgbm(folds, params)
        trials.append({"params": params, "cv_nrmse": cv_score, "n_trees": n_trees})

        best_so_far = min(t["cv_nrmse"] for t in trials)
        marker = "  <-- best" if cv_score == best_so_far else ""
        print(f"  [{len(trials):>2}/{N_TRIALS}] nRMSE {cv_score:.4f}%  "
              f"leaves={params['num_leaves']:<3} min_child={params['min_child_samples']:<4} "
              f"lr={params['learning_rate']:<5} lambda={params['reg_lambda']:<5}{marker}")

    best = min(trials, key=lambda t: t["cv_nrmse"])
    print(f"\nBest LightGBM CV nRMSE: {best['cv_nrmse']:.4f}%  "
          f"({best['n_trees']} trees avg)")
    print(f"  params: {best['params']}")

    # ── Verdict on the development data only ──────────────────────────────
    print(f"\nCV comparison (development period):")
    print(f"  Elia day-ahead : {elia:.4f}%")
    print(f"  Ridge          : {ridge_scores[best_alpha]:.4f}%  "
          f"(skill {100*(1 - ridge_scores[best_alpha]/elia):+.2f}%)")
    print(f"  LightGBM       : {best['cv_nrmse']:.4f}%  "
          f"(skill {100*(1 - best['cv_nrmse']/elia):+.2f}%)")

    chosen = {
        "elia_cv_nrmse": elia,
        "ridge": {"alpha": best_alpha, "cv_nrmse": ridge_scores[best_alpha]},
        "lightgbm": {**best["params"],
                     "n_estimators": best["n_trees"],
                     "cv_nrmse": best["cv_nrmse"]},
    }

    Path(PARAMS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"\nSaved -> {PARAMS_PATH}")
    return chosen


if __name__ == "__main__":
    main()
