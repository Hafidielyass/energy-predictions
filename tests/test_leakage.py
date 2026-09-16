"""
Guards against the ways this pipeline could silently start cheating.

Every test here corresponds to a real defect that was found by hand during
development. Hand-checking does not survive the next refactor; these do.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLS, LAG_HOURS, TARGET_COL

# Columns that describe the outcome at the moment being predicted. Any of these
# reaching the feature matrix means the model is reading the answer.
OUTCOME_COLUMNS = {
    "measured", "cf", "loadfactor", "mostrecentforecast",
    "mostrecentconfidence10", "mostrecentconfidence90",
}


@pytest.fixture(scope="module")
def features():
    try:
        return pd.read_parquet("data/dayahead/features.parquet")
    except FileNotFoundError:
        pytest.skip("run `python -m src.features.build_features` first")


def test_target_never_appears_as_a_feature():
    """The thing being predicted must not also be an input."""
    assert TARGET_COL not in FEATURE_COLS
    leaked = OUTCOME_COLUMNS & set(FEATURE_COLS)
    assert not leaked, f"outcome columns leaked into features: {leaked}"


def test_no_inverter_or_dc_columns():
    """
    No DC power, inverter telemetry, or cumulative yield counters.

    This is the defect that makes the widely-copied Kaggle solar notebooks
    report R2 = 1.00: AC power is DC power times inverter efficiency, so
    feeding DC in means fitting y = 0.975x. Cumulative yield counters leak the
    same way, since the running total at time T already contains the output at
    time T.
    """
    banned = ("dc_power", "source_key", "daily_yield", "total_yield", "inverter")
    for column in FEATURE_COLS:
        assert not any(b in column.lower() for b in banned), \
            f"{column} is inverter/DC/yield-derived and cannot be a feature"


def test_lags_clear_the_issue_time_cutoff():
    """
    Lags must be old enough to exist when the forecast is issued.

    Elia publishes the day-ahead forecast on the morning of D-1, so for a
    target late on day D the newest actual is already ~37 h old. Separately,
    the data feeds leave a ~24 h gap between the historical archive and the
    near-real-time one. 72 h clears both.
    """
    assert min(LAG_HOURS) >= 72, \
        f"shortest lag {min(LAG_HOURS)}h is not available at serving time"

    for hours in LAG_HOURS:
        assert f"cf_lag_{hours}h" in FEATURE_COLS


def test_lag_features_look_backwards(features):
    """A lag column must equal the target shifted forward in time, never back."""
    lag = min(LAG_HOURS)
    column = f"cf_lag_{lag}h"

    expected = features[TARGET_COL].shift(lag)
    both_present = features[column].notna() & expected.notna()
    assert both_present.sum() > 1000, "not enough overlap to verify"

    np.testing.assert_allclose(
        features.loc[both_present, column].to_numpy(),
        expected[both_present].to_numpy(),
        rtol=1e-4,
        err_msg=f"{column} is not the target lagged by {lag}h",
    )


def test_weather_and_generation_are_time_aligned(features):
    """
    Forecast irradiance must correlate best with generation at zero shift.

    Open-Meteo reports radiation as a backward average over the preceding hour
    while Elia labels hourly means by their start, so joining the two naively
    offsets sunshine from power by a full hour. Elia's own forecast is the
    control: it is aligned by construction, so if it fails this the join itself
    is broken rather than the shift correction.
    """
    cf = features[TARGET_COL]
    shifts = range(-3, 4)

    ghi = {s: cf.corr(features["shortwave_radiation"].shift(s)) for s in shifts}
    assert max(ghi, key=ghi.get) == 0, \
        f"forecast irradiance is misaligned; peaks at {max(ghi, key=ghi.get):+d}h"

    control = {s: cf.corr(features["elia_dayahead_cf"].shift(s)) for s in shifts}
    assert max(control, key=control.get) == 0, "the join itself is misaligned"


def test_shuffling_the_weather_destroys_accuracy(features):
    """
    A model that survives shuffled weather is not using it.

    It would be reading the diurnal cycle off the calendar and solar-geometry
    columns instead, and would fail the moment conditions departed from the
    seasonal norm.
    """
    joblib = pytest.importorskip("joblib")
    try:
        bundle = joblib.load("models/dayahead/ridge.pkl")
    except FileNotFoundError:
        pytest.skip("run `python -m src.models.train` first")

    from src.preprocessing.split import chronological_split
    _, _, test = chronological_split(features)
    day = test[test["is_day"] == 1]

    def rmse(frame):
        pred = bundle["model"].predict(bundle["scaler"].transform(frame)).clip(0, 1)
        return float(np.sqrt(((day[TARGET_COL] - pred) ** 2).mean()))

    intact = rmse(day[FEATURE_COLS])

    shuffled = day[FEATURE_COLS].copy()
    order = np.random.default_rng(0).permutation(len(shuffled))
    for column in ("shortwave_radiation", "direct_radiation", "cloud_cover",
                   "clearsky_index", "physical_yield", "elia_dayahead_cf"):
        shuffled[column] = shuffled[column].to_numpy()[order]

    assert rmse(shuffled) > 1.5 * intact, \
        "accuracy barely changed on shuffled weather — the model ignores it"
