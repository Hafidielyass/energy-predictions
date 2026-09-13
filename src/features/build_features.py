"""
Assemble the modelling feature matrix from the joined generation/forecast data.

Everything here must be knowable at forecast issue time. The weather columns
are already archived forecasts rather than observations, solar geometry is
deterministic, and the autoregressive lags are deliberately held back far
enough to survive the issue-time cutoff (see LAG_HOURS).

Run:
    python -m src.features.build_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.solar import add_solar_features

IN_PATH = "data/dayahead/belgium_hourly.parquet"
OUT_PATH = "data/dayahead/features.parquet"

BELGIUM_LAT, BELGIUM_LON, BELGIUM_ALT = 50.64, 4.67, 60.0

# Elia issues the day-ahead forecast during the morning of D-1, so for a target
# late on day D the most recent actual is already ~37 h old. Anything shorter
# than 48 h would be available for some hours and not others — the kind of
# quietly horizon-dependent leak that inflates offline scores and vanishes in
# production. 48 h is the shortest lag that is safe for every hour in the window.
LAG_HOURS = [48, 72, 168]

WEATHER_COLS = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "temperature_2m", "cloud_cover",
    "relative_humidity_2m", "wind_speed_10m", "precipitation",
]

SOLAR_COLS = [
    "solar_elevation", "solar_zenith", "solar_azimuth",
    "clearsky_ghi", "clearsky_dni", "clearsky_dhi", "clearsky_index",
]

CALENDAR_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

PHYSICS_COLS = ["cell_temp_est", "physical_yield"]

LAG_COLS = [f"cf_lag_{h}h" for h in LAG_HOURS] + ["cf_roll_mean_7d"]

BASELINE_COLS = ["elia_dayahead_cf"]

FEATURE_COLS = (
    WEATHER_COLS + SOLAR_COLS + CALENDAR_COLS
    + PHYSICS_COLS + LAG_COLS + BASELINE_COLS
)

TARGET_COL = "cf"


def build() -> pd.DataFrame:
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} rows  {df.index.min()} -> {df.index.max()}")

    cap = df["monitoredcapacity"]

    # Model the capacity factor rather than raw MW: the Belgian fleet grew from
    # 8.8 to 12.1 GW across this window, so raw megawatts conflate "sunnier day"
    # with "more panels installed".
    df[TARGET_COL] = (df["measured"] / cap).astype("float32")
    df["elia_dayahead_cf"] = (df["dayaheadforecast"] / cap).astype("float32")

    df = add_solar_features(df, BELGIUM_LAT, BELGIUM_LON, BELGIUM_ALT)
    df = _add_calendar(df)
    df = _add_physics(df)
    df = _add_lags(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Dropped {before - len(df):,} rows with incomplete lag history "
          f"-> {len(df):,} usable rows")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, compression="snappy")
    print(f"Saved -> {OUT_PATH}  ({len(FEATURE_COLS)} features)")
    return df


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings so hour 23 sits next to hour 0 rather than 23 units away."""
    hour = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    return df


def _add_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    A first-principles yield estimate handed to the model as a feature.

    Panel efficiency falls as the cells heat up, and cells run hotter than the
    surrounding air in proportion to irradiance. Encoding that relationship
    explicitly saves a tree ensemble from having to approximate a smooth
    physical curve through axis-aligned splits.
    """
    # NOCT-style rise: cells sit roughly (NOCT-20)/800 degrees above air
    # temperature per W/m2, with NOCT taken as a typical 45 C.
    df["cell_temp_est"] = (
        df["temperature_2m"] + 0.03125 * df["shortwave_radiation"]
    ).astype("float32")

    # Crystalline silicon loses about 0.4% of output per degree above the 25 C
    # rating point.
    derate = 1.0 - 0.004 * (df["cell_temp_est"] - 25.0)
    df["physical_yield"] = (
        (df["shortwave_radiation"] / 1000.0) * derate.clip(lower=0.0)
    ).astype("float32")
    return df


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Autoregressive history, held back to stay behind the issue-time cutoff."""
    for hours in LAG_HOURS:
        df[f"cf_lag_{hours}h"] = df[TARGET_COL].shift(hours).astype("float32")

    # Week-long average, itself shifted past the cutoff. Tracks slow drift in
    # fleet performance (soiling, seasonal snow, metering changes) that the
    # weather features cannot express.
    df["cf_roll_mean_7d"] = (
        df[TARGET_COL].shift(min(LAG_HOURS)).rolling(168, min_periods=24).mean()
    ).astype("float32")
    return df


if __name__ == "__main__":
    frame = build()
    day = frame[frame["is_day"] == 1]
    print(f"\nDaytime rows: {len(day):,} / {len(frame):,} "
          f"({100*len(day)/len(frame):.1f}%)")
    print(f"Target (capacity factor) daytime range: "
          f"{day[TARGET_COL].min():.4f} -> {day[TARGET_COL].max():.4f}")
    print(f"\nFeature columns ({len(FEATURE_COLS)}):")
    for col in FEATURE_COLS:
        print(f"  {col}")
