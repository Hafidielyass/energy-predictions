"""
Build the day-ahead PV forecasting dataset: Elia generation joined to archived
one-day-lead NWP forecasts, with an integrity report.

Run:
    python -m src.data.build_dataset
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data import elia, openmeteo

# Geographic centre of Belgium, kept for solar geometry — the sun's position
# varies little across a country 300 km wide, so one point is ample there.
BELGIUM_LAT, BELGIUM_LON = 50.64, 4.67

# Weather is a different matter: cloud over Antwerp is not cloud over Arlon.
# Each of Belgium's eleven provinces is sampled at its capital and weighted by
# the PV capacity Elia reports for it, so the forecast the model sees reflects
# the weather where the panels actually are. Capacities are from Elia's own
# per-region monitoredcapacity and sum to the national 12,128 MW.
PROVINCES = {
    "East-Flanders":   (51.05, 3.72, 2039.7),
    "Antwerp":         (51.22, 4.40, 2029.0),
    "West-Flanders":   (51.21, 3.22, 1911.9),
    "Limburg":         (50.93, 5.34, 1565.2),
    "Flemish-Brabant": (50.88, 4.70, 1168.0),
    "Hainaut":         (50.45, 3.95, 1058.3),
    "Liege":           (50.63, 5.57,  854.1),
    "Namur":           (50.47, 4.87,  444.6),
    "Luxembourg":      (49.68, 5.81,  368.0),
    "Walloon-Brabant": (50.72, 4.61,  358.2),
    "Brussels":        (50.85, 4.35,  331.4),
}

# Elia reaches back to 2020-09, but the Open-Meteo previous-runs archive only
# starts in 2024 — the binding constraint on how much history is usable. Rows
# without a forecast are dropped rather than imputed: a fabricated forecast
# would defeat the entire point of the day-ahead framing.
START = "2024-01-01"
END = "2026-09-01"

OUT_PATH = "data/dayahead/belgium_hourly.parquet"
ELIA_CACHE = "data/dayahead/raw_elia.parquet"
NWP_CACHE = "data/dayahead/raw_nwp.parquet"


def build(force: bool = False) -> pd.DataFrame:
    print(f"Elia ODS032: {START} -> {END}")
    gen = elia.fetch(START, END, region="Belgium", cache_path=ELIA_CACHE, force=force)
    print(f"  {len(gen):,} rows at 15 min")

    gen_h = elia.to_hourly(gen)
    print(f"  {len(gen_h):,} rows after hourly aggregation")

    print(f"Open-Meteo previous runs (lead 1 day), {len(PROVINCES)} provinces "
          f"weighted by installed capacity:")
    nwp = openmeteo.fetch_weighted(
        PROVINCES, START, END, cache_path=NWP_CACHE, force=force,
    )
    print(f"  {len(nwp):,} hourly rows, {len(nwp.columns)} variables")

    df = gen_h.join(nwp, how="inner")
    before = len(df)
    df = df.dropna(subset=["shortwave_radiation", "temperature_2m"])
    print(f"Joined: {before:,} rows -> {len(df):,} after dropping rows with no "
          f"archived forecast ({before - len(df):,} dropped)")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, compression="snappy")
    print(f"Saved -> {OUT_PATH}")
    return df


def integrity_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("INTEGRITY REPORT")
    print("=" * 70)

    print(f"Range: {df.index.min()} -> {df.index.max()}")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    missing = expected.difference(df.index)
    print(f"Expected hours: {len(expected):,} | present: {len(df):,} | "
          f"missing: {len(missing):,} ({100*len(missing)/len(expected):.3f}%)")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"\nColumns with nulls: {nulls.to_dict() if len(nulls) else 'none'}")

    print("\nPhysical plausibility:")
    checks = {
        "measured": (0, 25000),
        "dayaheadforecast": (0, 25000),
        "shortwave_radiation": (0, 1200),
        "temperature_2m": (-25, 45),
        "cloud_cover": (0, 100),
        "relative_humidity_2m": (0, 100),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        flag = "  <-- CHECK" if bad else ""
        print(f"  {col:24s} range [{df[col].min():9.2f}, {df[col].max():9.2f}] "
              f"outside bounds: {bad}{flag}")

    _alignment_check(df)
    _baseline_skill(df)


def _alignment_check(df: pd.DataFrame) -> None:
    """
    Verify forecast irradiance is time-aligned with generation.

    Scans candidate hour shifts and checks that correlation peaks at zero. A
    peak anywhere else means the two series are offset — which is invisible in
    summary statistics but degrades every downstream model. Elia's own
    forecast acts as the control: it is aligned with `measured` by
    construction, so its best shift must come out at zero or the join itself
    is wrong.
    """
    cf = df["measured"] / df["monitoredcapacity"]
    shifts = range(-3, 4)

    ghi_corr = {s: cf.corr(df["shortwave_radiation"].shift(s)) for s in shifts}
    best_ghi = max(ghi_corr, key=ghi_corr.get)

    ref = df["dayaheadforecast"] / df["monitoredcapacity"]
    ref_corr = {s: cf.corr(ref.shift(s)) for s in shifts}
    best_ref = max(ref_corr, key=ref_corr.get)

    status = "OK" if best_ghi == 0 else f"MISALIGNED by {best_ghi:+d} h"
    print(f"\nForecast/generation alignment: best shift {best_ghi:+d} h "
          f"(corr {ghi_corr[best_ghi]:.4f})  -> {status}")
    print(f"  control - Elia forecast best shift {best_ref:+d} h "
          f"(corr {ref_corr[best_ref]:.4f})"
          f"{'' if best_ref == 0 else '  <-- JOIN ERROR'}")


def _baseline_skill(df: pd.DataFrame) -> None:
    """Elia's own day-ahead forecast error — the bar any model must clear."""
    d = df.dropna(subset=["measured", "dayaheadforecast"])
    cap = d["monitoredcapacity"]
    day = d[(d["measured"] > 0.01 * cap) | (d["dayaheadforecast"] > 0.01 * cap)]

    err = day["measured"] - day["dayaheadforecast"]
    c = day["monitoredcapacity"]
    print(f"\nElia day-ahead baseline (daytime, n={len(day):,}):")
    print(f"  nMAE  {100*(err.abs()/c).mean():5.2f} % of capacity")
    print(f"  nRMSE {100*np.sqrt(((err/c)**2).mean()):5.2f} % of capacity")
    print(f"  bias  {err.mean():+8.1f} MW")


if __name__ == "__main__":
    frame = build()
    integrity_report(frame)
