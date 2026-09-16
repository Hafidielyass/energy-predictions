
"""
Build the provincial panel: each Belgian province as its own observation.

The national model trains on ~11,900 daylight rows, which is what forced the
gradient-boosting model down to seven leaves. Elia publishes measured output,
installed capacity and its own day-ahead forecast for all eleven provinces
separately, and we already fetch weather at all eleven points — the national
pipeline just averages them away.

Pairing each province's generation with its own local weather multiplies the
training rows by eleven without a new data source. The rows are correlated,
since Belgian provinces share weather systems, so the effective sample size is
well below eleven times; the gain should be real but smaller than the raw
count suggests.

Run:
    python -m src.data.build_panel
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import elia, openmeteo
from src.data.build_dataset import END, PROVINCES, START

OUT_PATH = "data/dayahead/panel_hourly.parquet"
ELIA_CACHE = "data/dayahead/raw_elia_panel.parquet"
NWP_CACHE = "data/dayahead/raw_nwp_panel.parquet"

# Elia also publishes "Belgium", "Flanders" and "Wallonia", which are sums of
# the provinces. Including them would train the model on the same generation
# several times over at different aggregation levels.
AGGREGATE_REGIONS = {"Belgium", "Flanders", "Wallonia"}

# Elia spells one province with an accent; our weather site keys do not.
REGION_ALIASES = {"Liege": "Liège"}


def build(force: bool = False) -> pd.DataFrame:
    generation = _fetch_generation(force)
    weather = _fetch_weather(force)

    frames = []
    for region in PROVINCES:
        gen = generation[generation["region"] == region].drop(columns="region")
        wx = weather[weather["region"] == region].drop(columns="region")

        joined = elia.to_hourly(gen).drop(columns="region", errors="ignore").join(
            wx, how="inner"
        )
        joined = joined.dropna(subset=["shortwave_radiation", "temperature_2m"])
        joined["region"] = region
        frames.append(joined)
        print(f"  {region:<18} {len(joined):,} hourly rows")

    panel = pd.concat(frames).sort_index()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, compression="snappy")
    print(f"\nPanel: {len(panel):,} rows across {panel['region'].nunique()} provinces")
    print(f"Saved -> {OUT_PATH}")
    return panel


def _fetch_generation(force: bool) -> pd.DataFrame:
    if Path(ELIA_CACHE).exists() and not force:
        return pd.read_parquet(ELIA_CACHE)

    print(f"Elia ODS032 per province: {START} -> {END}")
    frames = []
    for region in PROVINCES:
        elia_name = REGION_ALIASES.get(region, region)
        frame = elia.fetch(START, END, region=elia_name, cache_path=None)
        frame["region"] = region
        frames.append(frame)
        print(f"  {region:<18} {len(frame):,} rows at 15 min")

    panel = pd.concat(frames).sort_index()
    Path(ELIA_CACHE).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(ELIA_CACHE, compression="snappy")
    return panel


def _fetch_weather(force: bool) -> pd.DataFrame:
    print(f"\nOpen-Meteo previous runs per province:")
    return openmeteo.fetch_sites(
        PROVINCES, START, END, cache_path=NWP_CACHE, force=force
    )


def report(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("PANEL REPORT")
    print("=" * 74)
    print(f"{'region':<18}{'rows':>9}{'capacity MW':>13}{'mean CF':>10}{'Elia nRMSE':>12}")
    print("-" * 74)

    for region, group in panel.groupby("region"):
        day = group[group["measured"] > 0.01 * group["monitoredcapacity"]]
        if not len(day):
            continue
        cf = day["measured"] / day["monitoredcapacity"]
        err = cf - day["dayaheadforecast"] / day["monitoredcapacity"]
        print(f"{region:<18}{len(group):>9,}{group['monitoredcapacity'].iloc[-1]:>13.1f}"
              f"{cf.mean():>10.3f}{100 * np.sqrt((err ** 2).mean()):>11.2f}%")

    print(f"\nTotal rows: {len(panel):,}")
    daylight = panel[panel["measured"] > 0.01 * panel["monitoredcapacity"]]
    print(f"Daylight rows: {len(daylight):,}")

    overlap = panel.groupby(panel.index).size()
    print(f"Provinces present per timestamp: min {overlap.min()}, max {overlap.max()}")


if __name__ == "__main__":
    frame = build()
    report(frame)
