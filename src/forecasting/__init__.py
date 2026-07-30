"""
src.forecasting — Power Generation Forecasting
================================================

Modules
-------
gru_model  : Stacked GRU architecture with Keras Tuner hyperparameter search
baselines  : Persistence, Ridge Regression, Random Forest, XGBoost

Model choice: GRU
-----------------
GRU was selected over LSTM based on:
* Equivalent accuracy on smooth periodic solar signals (Wang 2020; Agga 2022)
* ~25% fewer parameters → lower overfitting risk on a single-site dataset
* Faster training convergence (15–30% wall-time reduction on CPU)
* Identical gating structure sufficient for the daily periodicity in PV data

Architecture summary (default config)
--------------------------------------
  Input  (96, n_features)
    → GRU(128, return_sequences=True)  → Dropout(0.2)
    → GRU(64,  return_sequences=False) → Dropout(0.2)
    → Dense(64, ReLU) → Dropout(0.1)
    → Dense(96, ReLU)    # 24 h × 15-min forecast horizon

Loss : Huber  (robust to cloud-transient outliers)
Opt  : Adam with ReduceLROnPlateau
Stop : EarlyStopping(patience=15)

Key classes / functions
-----------------------
build_gru_model(input_shape, horizon, units, dropout, lr)  → keras.Model
GRUForecaster   — wraps build, fit, evaluate, predict, save/load
compute_metrics(y_true, y_pred, name)                      → dict[str, float]
PersistenceModel, LinearForecaster, RandomForestForecaster, XGBoostForecaster
"""

from src.forecasting.gru_model import build_gru_model, GRUForecaster, get_callbacks
from src.forecasting.baselines import (
    compute_metrics,
    PersistenceModel,
    LinearForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

__all__ = [
    "build_gru_model",
    "GRUForecaster",
    "get_callbacks",
    "compute_metrics",
    "PersistenceModel",
    "LinearForecaster",
    "RandomForestForecaster",
    "XGBoostForecaster",
]
