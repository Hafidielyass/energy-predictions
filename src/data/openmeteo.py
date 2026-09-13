"""
Open-Meteo Previous Runs ingest — archived numerical weather prediction fields
at a fixed one-day lead time.

Every variable carries the ``_previous_day1`` suffix, meaning the value is what
the model predicted 24 hours before that timestamp was valid. Using these
instead of observed weather is what keeps the day-ahead framing honest: at
issue time the true weather does not exist yet, only a forecast of it.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# cloud_cover_low/mid/high are absent from the previous-runs archive — they
# come back 100% null — so they are deliberately not requested here.
DEFAULT_VARIABLES = [
    "shortwave_radiation",      # GHI
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "temperature_2m",
    "cloud_cover",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
]

# Open-Meteo reports these as an average (or sum) over the *preceding* hour, so
# the value stamped 12:00 describes 11:00-12:00. Everything else is an
# instantaneous reading at the stamp itself. Left-labelling the interval
# variables costs one shift and gains ~0.04 correlation against generation;
# skipping it silently offsets irradiance from power by a full hour.
BACKWARD_AVERAGED = {
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "precipitation",
}


def fetch(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    variables: Optional[List[str]] = None,
    lead_days: int = 1,
    cache_path: Optional[str] = None,
    force: bool = False,
    timeout: int = 180,
) -> pd.DataFrame:
    """
    Download archived day-ahead NWP forecasts for one location.

    Parameters
    ----------
    latitude, longitude : float
        Site coordinates. Open-Meteo resolves to its nearest grid cell.
    start, end : str
        ``"YYYY-MM-DD"``, both inclusive.
    lead_days : int
        Forecast lead time in days (1 = issued ~24 h ahead). 1-7 supported.
    cache_path : str | None
        Parquet cache. Reused unless ``force``.

    Returns
    -------
    pd.DataFrame indexed by UTC datetime at hourly resolution. Column names
    have the ``_previous_dayN`` suffix stripped, so ``shortwave_radiation``
    here always means the forecast value, never an observation.
    """
    if cache_path and Path(cache_path).exists() and not force:
        return pd.read_parquet(cache_path)

    variables = variables or DEFAULT_VARIABLES
    suffix = f"_previous_day{lead_days}"
    hourly_params = [v + suffix for v in variables]

    # Chunk by calendar year: long ranges can exceed the API's per-request limit.
    frames = []
    for year_start, year_end in _year_chunks(start, end):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_params),
            "start_date": year_start,
            "end_date": year_end,
            "timezone": "UTC",
        }
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]

        chunk = pd.DataFrame(hourly)
        chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
        frames.append(chunk.set_index("time"))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"
    df.columns = [c.replace(suffix, "") for c in df.columns]

    interval_cols = [c for c in df.columns if c in BACKWARD_AVERAGED]
    df[interval_cols] = df[interval_cols].shift(-1)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, compression="snappy")

    return df


def _year_chunks(start: str, end: str):
    """Split an inclusive date range into per-calendar-year segments."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cursor = start_ts
    while cursor <= end_ts:
        year_end = min(pd.Timestamp(year=cursor.year, month=12, day=31), end_ts)
        yield cursor.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")
        cursor = year_end + pd.Timedelta(days=1)
