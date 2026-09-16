"""
Serving-side invariants.

Both failures guarded here shipped at some point and were invisible to every
offline metric — they only appeared when the serving path was actually run.
"""

import pandas as pd

from src.data.build_dataset import PROVINCES
from src.data.openmeteo import BACKWARD_AVERAGED, DEFAULT_VARIABLES
from src.deployment.predict import apply_physical_guardrails
from src.features.build_features import FEATURE_COLS


def _frame(is_day, clearsky_ghi):
    index = pd.date_range("2026-06-15", periods=len(is_day), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"is_day": is_day, "clearsky_ghi": clearsky_ghi}, index=index
    )


def test_night_predictions_are_forced_to_zero():
    """A daylight-trained model emits garbage at night; physics says zero."""
    frame = _frame(is_day=[0, 0, 1, 1], clearsky_ghi=[0.0, 0.0, 800.0, 900.0])
    raw = pd.DataFrame(
        {"P10": [0.01, 0.02, 0.10, 0.15],
         "P50": [0.05, 0.06, 0.30, 0.35],
         "P90": [0.40, 0.45, 0.55, 0.60]},
        index=frame.index,
    )

    guarded = apply_physical_guardrails(raw, frame)
    assert (guarded.loc[frame["is_day"] == 0] == 0).all().all()
    assert (guarded.loc[frame["is_day"] == 1] > 0).all().all()


def test_predictions_cannot_exceed_the_clear_sky_ceiling():
    """
    Twilight is where the upper quantile used to run away.

    With a clear-sky ceiling of 20 W/m2 the maximum possible capacity factor is
    0.02, so a P90 of 0.60 is physically impossible regardless of what the
    model believes.
    """
    frame = _frame(is_day=[1, 1], clearsky_ghi=[20.0, 1000.0])
    raw = pd.DataFrame(
        {"P10": [0.01, 0.20], "P50": [0.30, 0.40], "P90": [0.60, 0.70]},
        index=frame.index,
    )

    guarded = apply_physical_guardrails(raw, frame)
    assert (guarded.iloc[0] <= 0.02 + 1e-9).all(), "twilight ceiling not enforced"
    assert guarded.iloc[1]["P90"] == 0.70, "a reachable value must pass through"


def test_guardrails_never_raise_a_prediction():
    """They are a cap, not a correction — output can only shrink."""
    frame = _frame(is_day=[1, 1, 0], clearsky_ghi=[500.0, 50.0, 0.0])
    raw = pd.DataFrame(
        {"P10": [0.10, 0.02, 0.01], "P50": [0.25, 0.30, 0.05],
         "P90": [0.40, 0.60, 0.20]},
        index=frame.index,
    )

    guarded = apply_physical_guardrails(raw, frame)
    assert (guarded <= raw + 1e-9).all().all()


def test_serving_uses_the_training_feature_set():
    """
    Training and serving must build identical features.

    Serving imports FEATURE_COLS from the training module rather than keeping
    its own list, so the two cannot drift apart. This asserts that structure
    stays in place — a local copy in the serving module would reintroduce
    exactly the skew that broke the 48-hour lag.
    """
    import src.deployment.predict as serving

    assert serving.FEATURE_COLS is FEATURE_COLS
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "duplicate feature names"


def test_serving_weights_the_same_eleven_provinces():
    """
    Training weights weather by province capacity; serving must match.

    Serving a single centroid point while training on capacity-weighted
    weather would be a skew no offline metric could catch.
    """
    assert len(PROVINCES) == 11
    total = sum(weight for _, _, weight in PROVINCES.values())
    assert 11_500 < total < 12_500, f"province capacities sum to {total:.0f} MW"

    for name, (lat, lon, weight) in PROVINCES.items():
        assert 49.4 < lat < 51.6, f"{name} latitude outside Belgium"
        assert 2.5 < lon < 6.5, f"{name} longitude outside Belgium"
        assert weight > 0


def test_backward_averaged_variables_are_a_subset_of_requested_ones():
    """The hour-shift correction must apply to variables we actually fetch."""
    assert BACKWARD_AVERAGED <= set(DEFAULT_VARIABLES)
    assert "shortwave_radiation" in BACKWARD_AVERAGED
    assert "temperature_2m" not in BACKWARD_AVERAGED, \
        "temperature is instantaneous and must not be shifted"
