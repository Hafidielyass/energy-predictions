
"""
Features for the provincial panel.

Two things differ from the national build and both matter.

Solar geometry is computed per province rather than once for the country.
Luxembourg sits at 49.7N and Antwerp at 51.2N, so their sun angles and
clear-sky ceilings genuinely differ — using one national geometry would blur
the very signal the panel exists to exploit.

Lags are grouped by province. Shifting the stacked frame directly would pull
one province's history into another's features, since the panel interleaves
regions at every timestamp.

Run:
    python -m src.features.build_panel_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_dataset import PROVINCES
from src.features.build_features import (
    BASELINE_COLS, CALENDAR_COLS, LAG_HOURS, PHYSICS_COLS, SOLAR_COLS,
    TARGET_COL, WEATHER_COLS, _add_calendar, _add_physics,
)
from src.features.solar import add_solar_features

IN_PATH = "data/dayahead/panel_hourly.parquet"
OUT_PATH = "data/dayahead/panel_features.parquet"

# Province size, so the model can distinguish a 2 GW fleet from a 330 MW one.
# ghi_dispersion is national-only and has no panel equivalent, since a single
# province has no internal spread to measure.
PANEL_EXTRA_COLS = ["capacity_gw"]

PANEL_WEATHER_COLS = [c for c in WEATHER_COLS if c != "ghi_dispersion"]

PANEL_FEATURE_COLS = (
    PANEL_WEATHER_COLS + SOLAR_COLS + CALENDAR_COLS
    + PHYSICS_COLS + [f"cf_lag_{h}h" for h in LAG_HOURS] + ["cf_roll_mean_7d"]
    + BASELINE_COLS + PANEL_EXTRA_COLS
)

REGION_COL = "region"


def build() -> pd.DataFrame:
    panel = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(panel):,} rows, {panel[REGION_COL].nunique()} provinces")

    frames = []
    for region, group in panel.groupby(REGION_COL, sort=False):
        latitude, longitude, _ = PROVINCES[region]
        frames.append(_build_region(group.copy(), latitude, longitude))

    features = pd.concat(frames).sort_index()

    before = len(features)
    features = features.dropna(subset=PANEL_FEATURE_COLS + [TARGET_COL])
    print(f"Dropped {before - len(features):,} rows with incomplete lag history "
          f"-> {len(features):,} usable rows")

    features[REGION_COL] = features[REGION_COL].astype("category")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH, compression="snappy")
    print(f"Saved -> {OUT_PATH}  ({len(PANEL_FEATURE_COLS)} features)")
    return features


def _build_region(group: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    capacity = group["monitoredcapacity"]

    group[TARGET_COL] = (group["measured"] / capacity).astype("float32")
    group["elia_dayahead_cf"] = (group["dayaheadforecast"] / capacity).astype("float32")
    group["capacity_gw"] = (capacity / 1000.0).astype("float32")

    group = add_solar_features(group, latitude, longitude)
    group = _add_calendar(group)
    group = _add_physics(group)

    # Safe to shift directly: this frame holds one province only.
    for hours in LAG_HOURS:
        group[f"cf_lag_{hours}h"] = group[TARGET_COL].shift(hours).astype("float32")
    group["cf_roll_mean_7d"] = (
        group[TARGET_COL].shift(min(LAG_HOURS)).rolling(168, min_periods=24).mean()
    ).astype("float32")

    return group


if __name__ == "__main__":
    frame = build()

    day = frame[frame["is_day"] == 1]
    print(f"\nDaytime rows: {len(day):,} / {len(frame):,}")
    print(f"vs national model:  11,916 daytime rows "
          f"({len(day) / 11916:.1f}x more)")

    print(f"\nPer-province daytime rows and solar elevation at noon:")
    for region, group in day.groupby(REGION_COL, observed=True):
        noon = group[group.index.hour == 12]
        print(f"  {region:<18} {len(group):>7,} rows   "
              f"mean noon elevation {noon['solar_elevation'].mean():5.1f} deg")
