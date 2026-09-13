"""
Elia ODS032 ingest — Belgian grid PV generation, measured plus the TSO's own
day-ahead forecast.

The dataset pairs `measured` with `dayaheadforecast` on the same row, so the
forecast is by construction what was issued before the fact. No join against a
separate weather archive is needed to keep the framing leak-free, and
`dayaheadforecast` doubles as the operational baseline to beat.
"""

from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests

EXPORT_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/exports/csv"

NUMERIC_COLS = [
    "measured",
    "mostrecentforecast", "mostrecentconfidence10", "mostrecentconfidence90",
    "dayahead11hforecast", "dayahead11hconfidence10", "dayahead11hconfidence90",
    "dayaheadforecast", "dayaheadconfidence10", "dayaheadconfidence90",
    "weekaheadforecast", "weekaheadconfidence10", "weekaheadconfidence90",
    "monitoredcapacity", "loadfactor",
]


def fetch(
    start: str,
    end: str,
    region: str = "Belgium",
    cache_path: Optional[str] = None,
    force: bool = False,
    timeout: int = 600,
) -> pd.DataFrame:
    """
    Download Elia PV generation for one region.

    Parameters
    ----------
    start, end : str
        Inclusive start / exclusive end, ``"YYYY-MM-DD"``.
    region : str
        Elia region name, e.g. ``"Belgium"`` (national total), ``"Flanders"``.
    cache_path : str | None
        Parquet cache. Reused unless ``force``.

    Returns
    -------
    pd.DataFrame indexed by UTC datetime at 15-minute resolution.
    """
    if cache_path and Path(cache_path).exists() and not force:
        return pd.read_parquet(cache_path)

    where = f'region="{region}" and datetime>="{start}" and datetime<"{end}"'
    url = f"{EXPORT_URL}?where={quote(where)}&limit=-1&delimiter=%3B"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), sep=";")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="mixed")
    df = df.sort_values("datetime").set_index("datetime")
    df.index.name = "datetime"

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["resolutioncode"], errors="ignore")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression="snappy")

    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute records to hourly means.

    Hourly is the working resolution because the NWP weather features only
    update hourly; keeping 15-minute targets would imply a precision the
    inputs cannot support. Both the target and Elia's forecast are aggregated
    identically, so the skill comparison stays fair.
    """
    numeric = [c for c in NUMERIC_COLS if c in df.columns]
    hourly = df[numeric].resample("1h").mean()
    hourly["region"] = df["region"].iloc[0] if "region" in df.columns else None
    return hourly
