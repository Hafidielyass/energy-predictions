"""
Produce a live day-ahead forecast for a future date.

Training reads *archived* forecasts (`previous-runs`, one-day lead) so that
history reflects what was genuinely knowable in advance. Serving reads the
*live* forecast API instead, because tomorrow's archived run does not exist
yet. Both describe the same thing — the weather as predicted roughly a day
out — which is what keeps training and serving consistent.

Run:
    python -m src.deployment.predict              # tomorrow
    python -m src.deployment.predict 2026-09-20
"""

import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

from src.data import elia
from src.data.build_dataset import BELGIUM_LAT, BELGIUM_LON, PROVINCES
from src.data.openmeteo import BACKWARD_AVERAGED, DEFAULT_VARIABLES
from src.features.build_features import (
    FEATURE_COLS, LAG_HOURS, _add_calendar, _add_lags, _add_physics,
)
from src.features.solar import add_solar_features

LIVE_API = "https://api.open-meteo.com/v1/forecast"
QUANTILE_MODEL = "models/dayahead/lgbm_quantile.pkl"
RIDGE_MODEL = "models/dayahead/ridge.pkl"

# Enough history behind the target date to satisfy the longest lag and the
# seven-day rolling mean, plus a margin for any late-arriving Elia rows.
HISTORY_DAYS = max(LAG_HOURS) // 24 + 10


def _fetch_live_point(lat: float, lon: float, target: pd.Timestamp) -> pd.DataFrame:
    """Live NWP for one location, converted to the training convention."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(DEFAULT_VARIABLES),
        "start_date": target.strftime("%Y-%m-%d"),
        # One extra day so the backward-average shift below still has a value
        # to pull into the target day's final hour.
        "end_date": (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    response = requests.get(LIVE_API, params=params, timeout=60)
    response.raise_for_status()

    frame = pd.DataFrame(response.json()["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")
    frame.index.name = "datetime"

    # Same backward-average correction applied during ingest; without it the
    # served features would be offset an hour from the trained ones.
    interval_cols = [c for c in frame.columns if c in BACKWARD_AVERAGED]
    frame[interval_cols] = frame[interval_cols].shift(-1)
    return frame


def fetch_live_weather(target: pd.Timestamp) -> pd.DataFrame:
    """
    Capacity-weighted live forecast across the provinces.

    Must mirror `openmeteo.fetch_weighted` exactly — same sites, same weights,
    same dispersion term. Training on capacity-weighted weather while serving a
    single point would be a train/serve skew invisible to every offline metric.
    """
    total = sum(w for _, _, w in PROVINCES.values())

    frames, weights = {}, {}
    for name, (lat, lon, weight) in PROVINCES.items():
        frames[name] = _fetch_live_point(lat, lon, target)
        weights[name] = weight / total

    columns = next(iter(frames.values())).columns
    weighted = sum(frames[n][columns] * w for n, w in weights.items())

    ghi = pd.concat({n: f["shortwave_radiation"] for n, f in frames.items()}, axis=1)
    mean = weighted["shortwave_radiation"]
    variance = sum(weights[n] * (ghi[n] - mean) ** 2 for n in frames)
    weighted["ghi_dispersion"] = np.sqrt(variance)
    return weighted


def apply_physical_guardrails(
    predictions: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    """
    Impose what physics guarantees but a daylight-trained model cannot know.

    Night is outside the distribution the models ever saw, so they emit
    arbitrary constants there — an early version predicted 76 MW at midnight
    with a P90 of 1261 MW at 3 am. Output before sunrise and after sunset is
    zero by physics, not by prediction.

    The same failure survives into twilight, where the upper quantile drifts
    far above anything the sun could supply. A cloudless sky is a hard ceiling:
    no forecast may exceed the capacity factor that clear-sky irradiance alone
    would permit at full nameplate conversion — already a generous bound, since
    real systems never convert that efficiently.
    """
    guarded = predictions.copy()
    guarded.loc[frame["is_day"] == 0, :] = 0.0

    ceiling = (frame["clearsky_ghi"] / 1000.0).clip(0.0, 1.0)
    return guarded.clip(upper=ceiling, axis=0)


def forecast(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Forecast Belgian PV output for one day, with a P10/P50/P90 band.

    Returns a frame indexed by UTC hour with megawatt predictions.
    """
    target = (pd.Timestamp(target_date, tz="UTC") if target_date
              else pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1))
    target = target.normalize()
    print(f"Forecasting {target:%Y-%m-%d} (UTC)")

    quantile_models = joblib.load(QUANTILE_MODEL)["models"]
    ridge = joblib.load(RIDGE_MODEL)

    history_start = (target - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # The historical archive runs a couple of days behind real time, so it
    # cannot supply the 48-hour lag for tomorrow on its own. The near-real-time
    # feed spans roughly the last week plus the coming one, covering both that
    # recent gap and the target day's forecast. The archive stays authoritative
    # wherever both have a value.
    history = elia.fetch(history_start, end, region="Belgium", cache_path=None)
    recent = elia.fetch_forecast(
        (target - pd.Timedelta(days=6)).strftime("%Y-%m-%d"), end, region="Belgium"
    )
    generation = elia.to_hourly(history.combine_first(recent).sort_index())

    weather = fetch_live_weather(target)

    frame = generation.join(weather, how="outer").sort_index()
    if frame.loc[frame.index.isin(weather.index), "monitoredcapacity"].isna().any():
        frame["monitoredcapacity"] = frame["monitoredcapacity"].ffill()

    capacity = frame["monitoredcapacity"]
    frame["cf"] = frame["measured"] / capacity
    frame["elia_dayahead_cf"] = frame["dayaheadforecast"] / capacity

    frame = add_solar_features(frame, BELGIUM_LAT, BELGIUM_LON)
    frame = _add_calendar(frame)
    frame = _add_physics(frame)
    frame = _add_lags(frame)

    day = frame.loc[target:target + pd.Timedelta(hours=23)]
    usable = day.dropna(subset=FEATURE_COLS)
    if usable.empty:
        missing = day[FEATURE_COLS].isna().sum()
        raise RuntimeError(
            "No hour has a complete feature set. Missing counts:\n"
            f"{missing[missing > 0].to_string()}"
        )

    X = usable[FEATURE_COLS]
    predictions = pd.DataFrame(index=usable.index)
    for name, model in quantile_models.items():
        predictions[name] = model.predict(X).clip(0, 1)

    # The central estimate averages Ridge with LightGBM rather than picking one.
    # They are statistically tied and disagree about which is better depending
    # on the period evaluated — LightGBM wins on the development years, Ridge on
    # the test year — so averaging hedges an instability we cannot resolve. It
    # also scored best on the development folds, where the choice was made
    # without consulting held-out data.
    predictions["P50"] = (
        predictions["P50"]
        + ridge["model"].predict(ridge["scaler"].transform(X)).clip(0, 1)
    ) / 2

    predictions = apply_physical_guardrails(predictions, usable)

    # Independent quantile fits can cross; sort each row to restore ordering.
    predictions[["P10", "P50", "P90"]] = pd.DataFrame(
        __import__("numpy").sort(predictions[["P10", "P50", "P90"]].to_numpy(), axis=1),
        index=predictions.index, columns=["P10", "P50", "P90"],
    )

    cap = usable["monitoredcapacity"]
    output = pd.DataFrame({
        "P10_MW": predictions["P10"] * cap,
        "P50_MW": predictions["P50"] * cap,
        "P90_MW": predictions["P90"] * cap,
        "elia_MW": usable["dayaheadforecast"],
        "capacity_MW": cap,
    })

    if len(usable) < 24:
        print(f"  warning: only {len(usable)}/24 hours produced "
              f"(incomplete Elia history for the lag features)")
    return output


if __name__ == "__main__":
    result = forecast(sys.argv[1] if len(sys.argv) > 1 else None)

    daylight = result[result["P50_MW"] > 1.0]
    print(f"\n{'hour (UTC)':<12} {'P10':>9} {'P50':>9} {'P90':>9} {'Elia':>9}")
    print("-" * 52)
    for stamp, row in daylight.iterrows():
        print(f"{stamp:%Y-%m-%d %H}h {row['P10_MW']:>9.0f} {row['P50_MW']:>9.0f} "
              f"{row['P90_MW']:>9.0f} {row['elia_MW']:>9.0f}")

    energy = result["P50_MW"].sum() / 1000
    print(f"\nExpected energy: {energy:.1f} GWh "
          f"(P10 {result['P10_MW'].sum()/1000:.1f} - "
          f"P90 {result['P90_MW'].sum()/1000:.1f})")
