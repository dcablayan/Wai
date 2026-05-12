"""Demo and NOAA CO-OPS data loaders for Wai.

Schema contract for all DataFrames returned by this module:
    timestamp   : datetime64[ns, UTC]
    station_id  : str
    water_level : float  (meters)
    datum       : str    (e.g. MLLW, NAVD88, MSL)
    units       : str    (m or ft)
    lat         : float
    lon         : float
    source      : str    (DEMO_SYNTHETIC | NOAA_COOPS)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DEMO_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "demo" / "demo_water_levels.csv"

REQUIRED_COLUMNS = [
    "timestamp", "station_id", "water_level",
    "datum", "units", "lat", "lon", "source",
]


def load_demo_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the synthetic demo water-level dataset.

    Returns a DataFrame conforming to the Wai schema with
    UTC-aware timestamps. Data is clearly labeled as DEMO_SYNTHETIC.
    """
    p = path or DEMO_DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Demo data not found at {p}. "
            "Run: python -m scripts.prepare_demo_data"
        )
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_noaa_data(
    station_id: str,
    begin_date: str,
    end_date: str,
    product: str = "water_level",
    datum: str = "MLLW",
    units: str = "metric",
    time_zone: str = "gmt",
) -> pd.DataFrame:
    """Fetch water-level data from the NOAA CO-OPS public API.

    Parameters
    ----------
    station_id : str
        NOAA CO-OPS station ID (e.g. '9414290' for San Francisco).
    begin_date : str
        Start date in 'YYYYMMDD' format.
    end_date : str
        End date in 'YYYYMMDD' format (max 31-day window per request).
    product : str
        'water_level' for verified observations, 'predictions' for tidal
        predictions (no API key required for either).
    datum : str
        Tidal datum reference — MLLW, NAVD, MSL, MHHW, etc.
    units : str
        'metric' (meters) or 'english' (feet).
    time_zone : str
        'gmt', 'lst', or 'lst_ldt'.

    Returns
    -------
    pd.DataFrame conforming to the Wai schema.

    Example
    -------
    >>> df = load_noaa_data("9414290", "20240101", "20240131")
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "station": station_id,
        "product": product,
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": datum,
        "units": units,
        "time_zone": time_zone,
        "application": "wai_portfolio",
        "format": "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        raise ValueError(f"NOAA API error: {payload['error']['message']}")

    records = payload.get("data", [])
    if not records:
        raise ValueError(f"NOAA API returned no data for station {station_id}")

    df = pd.DataFrame(records)
    df = df.rename(columns={"t": "timestamp", "v": "water_level"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    df["station_id"] = str(station_id)
    df["datum"] = datum
    df["units"] = "m" if units == "metric" else "ft"
    df["source"] = "NOAA_COOPS"

    meta = payload.get("metadata", {})
    df["lat"] = float(meta.get("lat", float("nan")))
    df["lon"] = float(meta.get("lon", float("nan")))

    return df[REQUIRED_COLUMNS]
