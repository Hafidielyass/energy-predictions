"""
Unit and integration tests for the Solar PV forecasting pipeline.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Generate a 30-day synthetic PV DataFrame at 15-min resolution."""
    n = 30 * 96  # 30 days × 96 periods/day
    idx = pd.date_range("2022-06-01", periods=n, freq="15T")
    hour = idx.hour + idx.minute / 60

    # Simulate a realistic diurnal power curve
    power = np.maximum(0, 1000 * np.sin(np.pi * (hour - 6) / 12) ** 2)
    power[hour < 6] = 0
    power[hour > 20] = 0
    power += np.random.randn(n) * 10  # sensor noise

    return pd.DataFrame({
        "ac_power__315":         power.astype("float32"),
        "poa_irradiance__313":   (power * 1.2 + np.random.randn(n) * 5).clip(0).astype("float32"),
        "poa_irradiance_clipped":(power * 1.2).clip(0).astype("float32"),
        "ambient_temp__320":     (20 + 8 * np.sin(np.pi * hour / 12)).astype("float32"),
        "module_temp_1__321":    (30 + 10 * np.sin(np.pi * hour / 12)).astype("float32"),
        "module_temp_2__322":    (30 + 10 * np.sin(np.pi * hour / 12)).astype("float32"),
        "module_temp_3__323":    (30 + 10 * np.sin(np.pi * hour / 12)).astype("float32"),
        "inverter_temp__324":    (35 + 5 * np.sin(np.pi * hour / 12)).astype("float32"),
        "dc_pos_voltage__316":   (270 + np.random.randn(n)).astype("float32"),
        "dc_pos_current__317":   (power / 270).astype("float32"),
        "dc_power__314":         (power * 1.05).astype("float32"),
        "ac_voltage__318":       (122 + np.random.randn(n) * 0.5).astype("float32"),
        "ac_current__319":       (power / 122).astype("float32"),
        "power_factor__327":     (0.95 + np.random.randn(n) * 0.01).clip(0, 1).astype("float32"),
        "das_battery_voltage__326": np.full(n, 13.5, dtype="float32"),
        "das_temp__325":         (20 + np.random.randn(n)).astype("float32"),
        "is_daytime":            (power > 10).astype("int8"),
    }, index=idx)


# ─── Feature engineering tests ────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_cyclic_encoding_range(self, sample_df):
        """Cyclic sin/cos features must lie in [-1, 1]."""
        from src.features.feature_engineering import encode_cyclic
        feat = encode_cyclic(pd.Series(range(24), name="hour"), period=24.0)
        assert feat["hour_sin"].between(-1, 1).all()
        assert feat["hour_cos"].between(-1, 1).all()

    def test_cyclic_boundary_continuity(self):
        """Cyclic encoding at period boundary should be continuous."""
        from src.features.feature_engineering import encode_cyclic
        s = pd.Series([0.0, 23.0], name="hour")
        feat = encode_cyclic(s, period=24.0)
        # sin(0) ≈ sin(24·2π/24) numerically
        assert abs(feat["hour_sin"].iloc[0] - feat["hour_sin"].iloc[1]) < 0.3

    def test_feature_matrix_no_nans(self, sample_df):
        """Feature matrix should have no NaN after build_features call."""
        from src.features.feature_engineering import build_features
        feat = build_features(
            sample_df,
            target_col="ac_power__315",
            lag_periods=[1, 4, 96],
            rolling_windows=[4, 96],
            cache_path=None,
        )
        assert feat.isnull().sum().sum() == 0

    def test_sequence_shapes(self):
        """make_sequences must produce correct (N, seq, F) shapes."""
        from src.features.feature_engineering import make_sequences
        N, F = 500, 10
        X = np.random.randn(N, F).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)
        seq_len, horizon = 48, 12
        X_seq, y_seq = make_sequences(X, y, seq_len, horizon)
        expected_samples = N - seq_len - horizon + 1
        assert X_seq.shape == (expected_samples, seq_len, F)
        assert y_seq.shape == (expected_samples, horizon)

    def test_efficiency_ratio_non_negative(self, sample_df):
        """Efficiency ratio (power / irradiance) must be >= 0 during daytime."""
        from src.features.feature_engineering import build_features
        feat = build_features(sample_df, cache_path=None, lag_periods=[1], rolling_windows=[4])
        day_feat = feat[feat["is_daytime"] == 1]
        assert (day_feat["efficiency_ratio"] >= 0).all()


# ─── Data loading tests ───────────────────────────────────────────────────────

class TestDataLoader:

    def test_build_daily_df_shape(self, sample_df):
        """Daily aggregation should return one row per calendar day."""
        from src.preprocessing.data_loader import build_daily_df
        daily = build_daily_df(sample_df)
        expected_days = sample_df.index.normalize().nunique()
        assert len(daily) == expected_days

    def test_daily_energy_non_negative(self, sample_df):
        """Daily energy in kWh must be non-negative."""
        from src.preprocessing.data_loader import build_daily_df
        daily = build_daily_df(sample_df)
        assert (daily["daily_energy_kwh"] >= 0).all()

    def test_chronological_split_no_leakage(self, sample_df):
        """Train/val/test splits must not overlap in time."""
        from src.preprocessing.data_loader import chronological_split
        train, val, test = chronological_split(sample_df, test_frac=0.2, val_frac=0.1)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_split_sizes(self, sample_df):
        """Split sizes must sum to total length."""
        from src.preprocessing.data_loader import chronological_split
        train, val, test = chronological_split(sample_df, test_frac=0.15, val_frac=0.10)
        assert len(train) + len(val) + len(test) == len(sample_df)


# ─── GRU model tests ─────────────────────────────────────────────────────────

class TestGRUModel:

    def test_model_builds(self):
        """GRU model must build without errors."""
        from src.forecasting.gru_model import build_gru_model
        model = build_gru_model(input_shape=(96, 20), horizon=96)
        assert model is not None
        assert model.count_params() > 0

    def test_model_output_shape(self):
        """GRU must output (batch, horizon) tensor."""
        from src.forecasting.gru_model import build_gru_model
        model = build_gru_model(input_shape=(48, 10), horizon=24)
        X = np.random.randn(4, 48, 10).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert out.shape == (4, 24)

    def test_model_output_non_negative(self):
        """GRU output (ReLU) must be >= 0 for any input."""
        from src.forecasting.gru_model import build_gru_model
        model = build_gru_model(input_shape=(48, 5), horizon=12)
        X = np.random.randn(10, 48, 5).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert (out >= 0).all()


# ─── Anomaly detection tests ─────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_severity_labels_valid(self, sample_df):
        """All severity labels must be in the expected set."""
        from src.preprocessing.data_loader import build_daily_df
        from src.utils.config_loader import Config

        daily = build_daily_df(sample_df)

        cfg = Config({
            "anomaly_detection": {
                "isolation_forest": {"n_estimators": 20, "contamination": 0.05, "random_state": 42},
                "autoencoder":      {"encoding_dims": [8, 4], "epochs": 5, "batch_size": 8,
                                     "threshold_percentile": 95},
                "severity":         {"normal_below": 0.35, "severe_above": 0.65},
            }
        })
        from src.anomaly_detection.anomaly_detector import HybridAnomalyDetector
        det = HybridAnomalyDetector(cfg)
        det.fit(daily)
        result = det.predict(daily)

        valid_labels = {"Normal", "Moderately Faulty", "Severely Faulty"}
        assert set(result["severity_label"].dropna().unique()).issubset(valid_labels)

    def test_anomaly_score_range(self, sample_df):
        """Composite anomaly score must lie in [0, 1]."""
        from src.preprocessing.data_loader import build_daily_df
        from src.utils.config_loader import Config
        from src.anomaly_detection.anomaly_detector import HybridAnomalyDetector

        daily = build_daily_df(sample_df)
        cfg = Config({
            "anomaly_detection": {
                "isolation_forest": {"n_estimators": 10, "contamination": 0.05, "random_state": 42},
                "autoencoder":      {"encoding_dims": [8, 4], "epochs": 3, "batch_size": 4,
                                     "threshold_percentile": 95},
                "severity":         {"normal_below": 0.35, "severe_above": 0.65},
            }
        })
        det = HybridAnomalyDetector(cfg)
        det.fit(daily)
        result = det.predict(daily)
        valid = result["anomaly_score"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# ─── Baseline tests ───────────────────────────────────────────────────────────

class TestBaselines:

    def test_persistence_metric_keys(self):
        """Persistence model must return all required metric keys."""
        from src.forecasting.baselines import PersistenceModel
        y = pd.Series(np.random.rand(500) * 500)
        pm = PersistenceModel(horizon=24, lag_steps=48)
        metrics = pm.evaluate(y)
        assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "R2"}

    def test_ridge_predict_shape(self):
        """Ridge output shape must match (N_test,)."""
        from src.forecasting.baselines import LinearForecaster
        lf = LinearForecaster()
        X_tr = np.random.randn(100, 20, 5).astype(np.float32)
        y_tr = np.random.randn(100, 1).astype(np.float32)
        X_te = np.random.randn(20, 20, 5).astype(np.float32)
        lf.fit(X_tr, y_tr)
        pred = lf.predict(X_te)
        assert pred.shape == (20,)


# ─── Configuration tests ──────────────────────────────────────────────────────

class TestConfig:

    def test_config_loads(self):
        """Config file must load without errors."""
        from src.utils.config_loader import load_config
        cfg = load_config("configs/config.yaml")
        assert "forecasting" in cfg
        assert "anomaly_detection" in cfg

    def test_config_dot_access(self):
        """Config must support dot-notation attribute access."""
        from src.utils.config_loader import Config
        cfg = Config({"forecasting": {"model_type": "GRU", "epochs": 100}})
        assert cfg.forecasting["model_type"] == "GRU"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
