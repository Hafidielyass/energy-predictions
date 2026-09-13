"""
Deterministic solar geometry and clear-sky reference.

These are the only features knowable with certainty at any forecast horizon:
the sun's position and the irradiance a cloudless sky would deliver depend on
astronomy and site coordinates, not on weather. Supplying them directly means
the model spends its capacity on the genuinely uncertain part — how much cloud
attenuates that ceiling — instead of rediscovering the diurnal and seasonal
cycle from lag features.
"""

from typing import Optional

import numpy as np
import pandas as pd
import pvlib


def add_solar_features(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude: float = 60.0,
    ghi_col: Optional[str] = "shortwave_radiation",
) -> pd.DataFrame:
    """
    Append solar position, clear-sky irradiance, and a clear-sky index.

    Solar position is evaluated at the midpoint of each interval. The frame is
    left-labelled hourly (the row stamped 10:00 covers 10:00-11:00), so the
    instantaneous geometry that best represents the interval is the one at
    10:30 — using the left edge biases every value toward the start of the hour.

    Parameters
    ----------
    df : pd.DataFrame
        Left-labelled hourly frame with a tz-aware UTC index.
    latitude, longitude, altitude : float
        Site coordinates; altitude in metres.
    ghi_col : str | None
        Forecast GHI column used to derive the clear-sky index. Pass None to
        skip that step.
    """
    out = df.copy()
    midpoints = out.index + pd.Timedelta(minutes=30)

    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, tz="UTC", altitude=altitude
    )
    solpos = location.get_solarposition(midpoints)
    clearsky = location.get_clearsky(midpoints, model="ineichen")

    # pvlib indexes results by midpoint; restore the frame's own stamps.
    solpos.index = out.index
    clearsky.index = out.index

    out["solar_elevation"] = solpos["apparent_elevation"].astype("float32")
    out["solar_zenith"] = solpos["apparent_zenith"].astype("float32")
    out["solar_azimuth"] = solpos["azimuth"].astype("float32")
    out["clearsky_ghi"] = clearsky["ghi"].astype("float32")
    out["clearsky_dni"] = clearsky["dni"].astype("float32")
    out["clearsky_dhi"] = clearsky["dhi"].astype("float32")

    out["is_day"] = (out["solar_elevation"] > 0).astype("int8")

    if ghi_col and ghi_col in out.columns:
        out["clearsky_index"] = _clearsky_index(out[ghi_col], out["clearsky_ghi"])

    return out


def _clearsky_index(ghi: pd.Series, clearsky_ghi: pd.Series) -> pd.Series:
    """
    Forecast GHI as a fraction of the clear-sky ceiling — effectively "how
    cloudy does the forecast think it will be", independent of season or hour.

    Near sunrise and sunset the denominator approaches zero and the ratio
    explodes, so it is only evaluated above a 20 W/m2 floor and clipped to a
    physically sensible range (values slightly above 1 are real — cloud-edge
    reflection can briefly exceed the clear-sky estimate).
    """
    valid = clearsky_ghi > 20.0
    denominator = clearsky_ghi.where(valid, 1.0)
    ratio = np.where(valid, ghi / denominator, 0.0)
    return pd.Series(np.clip(ratio, 0.0, 1.5), index=ghi.index, dtype="float32")
