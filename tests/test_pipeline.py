"""
Correctness of the data, feature, and split stages.

These cover the physical and structural invariants the modelling depends on —
the things that would produce a plausible-looking but wrong model if they
quietly broke.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLS, TARGET_COL
from src.features.solar import _clearsky_index, add_solar_features
from src.preprocessing.split import chronological_split, rolling_origin_folds

BRUSSELS_LAT, BRUSSELS_LON = 50.85, 4.35


@pytest.fixture(scope="module")
def features():
    try:
        return pd.read_parquet("data/dayahead/features.parquet")
    except FileNotFoundError:
        pytest.skip("run `python -m src.features.build_features` first")


# ── Solar geometry ───────────────────────────────────────────────────────────

def test_sun_is_up_at_midday_and_down_at_midnight():
    """Sanity-check the geometry against facts that need no dataset."""
    times = pd.to_datetime(
        ["2026-06-21 11:00", "2026-06-21 23:00", "2026-12-21 11:00"], utc=True
    )
    frame = add_solar_features(
        pd.DataFrame(index=times), BRUSSELS_LAT, BRUSSELS_LON, ghi_col=None
    )

    summer_noon, midnight, winter_noon = frame["solar_elevation"]
    assert summer_noon > 55, "midsummer midday sun should be high over Brussels"
    assert midnight < 0, "the sun is below the horizon at 23:00 UTC in June"
    assert 10 < winter_noon < 25, "midwinter midday sun should be low but up"
    assert summer_noon > winter_noon, "summer sun must exceed winter sun"


def test_clearsky_index_stays_bounded():
    """
    The ratio must not explode when the clear-sky ceiling approaches zero.

    Near sunrise the denominator tends to zero, so the index is only evaluated
    above a floor and clipped. Values slightly above 1 are physically real —
    cloud-edge reflection can briefly beat the clear-sky estimate.
    """
    ghi = pd.Series([0.0, 100.0, 500.0, 900.0, 50.0])
    clearsky = pd.Series([0.0, 5.0, 600.0, 800.0, 1000.0])

    index = _clearsky_index(ghi, clearsky)
    assert (index >= 0).all() and (index <= 1.5).all()
    assert index.iloc[0] == 0.0, "zero ceiling must give zero, not NaN or inf"
    assert index.iloc[1] == 0.0, "below the floor must be suppressed, not divided"
    assert np.isfinite(index).all()


def test_is_day_matches_solar_elevation(features):
    day = features["is_day"] == 1
    assert (features.loc[day, "solar_elevation"] > 0).all()
    assert (features.loc[~day, "solar_elevation"] <= 0).all()


# ── Physical plausibility ────────────────────────────────────────────────────

def test_capacity_factor_is_a_fraction(features):
    """Output cannot be negative, nor exceed installed capacity."""
    assert features[TARGET_COL].min() >= 0
    assert features[TARGET_COL].max() <= 1.0


def test_generation_is_zero_at_night(features):
    """Anything else means the join or the geometry is wrong."""
    night = features[features["is_day"] == 0]
    assert night[TARGET_COL].max() < 0.02, \
        "meaningful generation recorded while the sun is below the horizon"


def test_forecast_irradiance_is_physically_possible(features):
    """Surface irradiance above ~1200 W/m2 indicates a broken unit or sensor."""
    assert features["shortwave_radiation"].between(0, 1200).all()
    assert features["temperature_2m"].between(-30, 50).all()
    assert features["cloud_cover"].between(0, 100).all()


def test_cell_temperature_exceeds_air_temperature_in_sun(features):
    """Panels self-heat under irradiance; the estimate must reflect that."""
    sunny = features[features["shortwave_radiation"] > 400]
    assert len(sunny) > 100
    assert (sunny["cell_temp_est"] > sunny["temperature_2m"]).all()


# ── Data integrity ───────────────────────────────────────────────────────────

def test_hourly_index_is_complete_and_unique(features):
    assert not features.index.duplicated().any()
    assert features.index.is_monotonic_increasing
    assert str(features.index.tz) == "UTC"

    gaps = features.index.to_series().diff().dropna()
    assert gaps.max() <= pd.Timedelta(hours=24), "unexpectedly large gap in the series"


def test_no_nulls_in_the_modelling_columns(features):
    assert features[FEATURE_COLS + [TARGET_COL]].notna().all().all()


# ── Splits ───────────────────────────────────────────────────────────────────

def test_splits_are_chronological_and_disjoint(features):
    train, val, test = chronological_split(features)

    assert len(train) and len(val) and len(test)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(features)


def test_test_period_covers_a_full_year(features):
    """
    A test set that misses winter cannot support a year-round claim.

    Solar skill varies enormously by season, so a partial-year test would make
    the headline number a statement about whichever months it happened to cover.
    """
    _, _, test = chronological_split(features)
    months = {ts.month for ts in test.index}
    assert len(months) == 12, f"test set covers only {len(months)} months"


def test_rolling_folds_never_train_on_their_own_future(features):
    folds = rolling_origin_folds(features)
    assert len(folds) >= 3

    for train, val in folds:
        assert train.index.max() < val.index.min(), "fold trains on future data"

    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes), "expanding window should grow with each fold"


def test_folds_stay_out_of_the_test_period(features):
    """Hyperparameter search must never see the held-out year."""
    _, _, test = chronological_split(features)
    for train, val in rolling_origin_folds(features):
        assert val.index.max() < test.index.min()
        assert train.index.max() < test.index.min()
