"""
Baseline forecasting models for comparative evaluation.

Models included
---------------
1. **Persistence** — naive: forecast = last observed value (strong daily-cycle baseline)
2. **Linear Regression** — learns linear mapping from feature vector
3. **Random Forest** — ensemble of decision trees, captures non-linearity
4. **XGBoost** — gradient boosting, state-of-the-art tabular baseline

All models share the same train/test split and feature set as the GRU,
enabling fair head-to-head comparison on MAE / RMSE / MAPE / R².
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    log.warning("XGBoost not installed. Skipping XGBoost baseline.")


# ─── Evaluation helper ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    name: str = "Model") -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, R²."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true > 1.0
    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ) if mask.sum() > 0 else np.nan
    print(f"{name:25s} | MAE={mae:7.2f} | RMSE={rmse:7.2f} | MAPE={mape:6.2f}% | R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ─── 1. Persistence model ─────────────────────────────────────────────────────

class PersistenceModel:
    """
    Forecasts future power = power value from exactly 24 h ago.

    This captures the strong daily cycle in solar data and is a much stronger
    baseline than a simple last-value-forward approach.
    """

    def __init__(self, horizon: int = 96, lag_steps: int = 96) -> None:
        self.horizon = horizon
        self.lag_steps = lag_steps   # 96 × 15 min = 24 h

    def predict(self, y_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        y_series : pd.Series  — full target time series (chronological)

        Returns
        -------
        y_true : np.ndarray  shape (N_windows, horizon)
        y_pred : np.ndarray  shape (N_windows, horizon)
        """
        values = y_series.values
        preds, trues = [], []
        step = self.horizon
        for start in range(0, len(values) - self.lag_steps - self.horizon, step):
            ref_start = start
            ref_end   = start + self.horizon
            future_start = start + self.lag_steps
            future_end   = future_start + self.horizon
            if future_end > len(values):
                break
            preds.append(values[ref_start:ref_end])
            trues.append(values[future_start:future_end])

        return np.array(trues), np.array(preds)

    def evaluate(self, y_series: pd.Series) -> Dict[str, float]:
        y_true, y_pred = self.predict(y_series)
        metrics = compute_metrics(y_true.ravel(), y_pred.ravel())
        log.info("Persistence model metrics: %s", metrics)
        return metrics


# ─── 2. Linear Regression ────────────────────────────────────────────────────

class LinearForecaster:
    """Ridge Regression on the same flat feature vector used by GRU."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Flatten sequences → (samples, seq_len * features)
        X_flat = X_train.reshape(len(X_train), -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        y_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        self.model.fit(X_scaled, y_flat)
        log.info("Linear forecaster trained on %d samples.", len(X_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_true = y_test[:, 0] if y_test.ndim > 1 else y_test
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_true, y_pred)
        log.info("Linear model metrics: %s", metrics)
        return metrics

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str) -> None:
        d = joblib.load(path)
        self.model  = d["model"]
        self.scaler = d["scaler"]


# ─── 3. Random Forest ────────────────────────────────────────────────────────

class RandomForestForecaster:
    """Random Forest Regressor — 1-step-ahead single-output version."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 20,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = X_train.reshape(len(X_train), -1)
        y_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        self.model.fit(X_flat, y_flat)
        log.info("Random Forest trained on %d samples.", len(X_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.reshape(len(X), -1))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_true = y_test[:, 0] if y_test.ndim > 1 else y_test
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_true, y_pred)
        log.info("Random Forest metrics: %s", metrics)
        return metrics

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


# ─── 4. XGBoost ──────────────────────────────────────────────────────────────

class XGBoostForecaster:
    """XGBoost Regressor — direct multi-output via sklearn wrapper."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = X_train.reshape(len(X_train), -1)
        y_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        self.model.fit(
            X_flat, y_flat,
            eval_set=[(X_flat, y_flat)],
            verbose=False,
        )
        log.info("XGBoost trained on %d samples.", len(X_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.reshape(len(X), -1))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_true = y_test[:, 0] if y_test.ndim > 1 else y_test
        y_pred = self.predict(X_test)
        metrics = compute_metrics(y_true, y_pred)
        log.info("XGBoost metrics: %s", metrics)
        return metrics

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model.load_model(path)
