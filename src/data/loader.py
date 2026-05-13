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


NOAA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def _noaa_api_params(
    station_id: str,
    begin_date: str,
    end_date: str,
    product: str,
    datum: str,
    units: str,
    time_zone: str,
) -> dict:
    """Return the NOAA CO-OPS API parameter dict (no network call)."""
    return {
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


def _parse_noaa_response(
    payload: dict,
    station_id: str,
    datum: str,
    units: str,
    source_label: str = "NOAA_COOPS",
) -> pd.DataFrame:
    """Parse a NOAA CO-OPS JSON payload into a Wai-schema DataFrame.

    Supports both ``data`` (observations/water_level product) and
    ``predictions`` (predictions product) as the records key.  Raises
    informative errors for API-reported errors or missing records.
    """
    if "error" in payload:
        msg = payload["error"].get("message", str(payload["error"]))
        raise ValueError(f"NOAA API error: {msg}")

    # The CO-OPS API uses "data" for observations and "predictions" for the
    # predictions product.  Accept either key so the loader is product-agnostic.
    records = payload.get("data") or payload.get("predictions") or []
    if not records:
        raise ValueError(f"NOAA API returned no data for station {station_id}")

    # Warn on potential datum/unit mismatches embedded in the metadata
    meta = payload.get("metadata", {})
    _check_noaa_metadata(meta, datum, units, station_id)

    df = pd.DataFrame(records)
    df = df.rename(columns={"t": "timestamp", "v": "water_level"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    df["station_id"] = str(station_id)
    df["datum"] = datum
    df["units"] = "m" if units == "metric" else "ft"
    df["source"] = source_label

    df["lat"] = float(meta.get("lat", float("nan")))
    df["lon"] = float(meta.get("lon", float("nan")))

    return df[REQUIRED_COLUMNS]


def _check_noaa_metadata(
    meta: dict,
    requested_datum: str,
    requested_units: str,
    station_id: str,
) -> None:
    """Emit warnings when the API response metadata suggests a mismatch."""
    import warnings

    api_datum = meta.get("datum", "")
    if api_datum and api_datum.upper() != requested_datum.upper():
        warnings.warn(
            f"NOAA station {station_id}: requested datum={requested_datum!r} but "
            f"response metadata reports datum={api_datum!r}. Values may be on a "
            "different vertical reference — verify before use.",
            stacklevel=4,
        )

    api_units = meta.get("units", "")
    expected_units = "metric" if requested_units == "metric" else "english"
    if api_units and api_units.lower() not in (expected_units, requested_units.lower()):
        warnings.warn(
            f"NOAA station {station_id}: requested units={requested_units!r} but "
            f"response metadata reports units={api_units!r}. Check unit conversion.",
            stacklevel=4,
        )


def load_noaa_data(
    station_id: str,
    begin_date: str,
    end_date: str,
    product: str = "water_level",
    datum: str = "MLLW",
    units: str = "metric",
    time_zone: str = "gmt",
) -> pd.DataFrame:
    """Fetch water-level observations from the NOAA CO-OPS public API.

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
    params = _noaa_api_params(
        station_id, begin_date, end_date, product, datum, units, time_zone
    )
    resp = requests.get(NOAA_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_noaa_response(resp.json(), station_id, datum, units)


def load_noaa_predictions(
    station_id: str,
    begin_date: str,
    end_date: str,
    datum: str = "MLLW",
    units: str = "metric",
    time_zone: str = "gmt",
) -> pd.DataFrame:
    """Fetch NOAA CO-OPS tidal predictions for a station and date range.

    Tidal predictions are deterministic harmonics computed from NOAA's
    constituent database. They do not include surge or meteorological effects
    and are available for all gauged stations without an API key.

    Useful as a high-quality tidal signal baseline or additional feature.

    Parameters
    ----------
    station_id : str
        NOAA CO-OPS station ID (e.g. '9414290' for San Francisco).
    begin_date : str
        Start date in 'YYYYMMDD' format.
    end_date : str
        End date in 'YYYYMMDD' format (max 31-day window per request).
    datum : str
        Tidal datum — MLLW, NAVD, MSL, MHHW, etc.
    units : str
        'metric' (meters) or 'english' (feet).
    time_zone : str
        'gmt', 'lst', or 'lst_ldt'.

    Returns
    -------
    pd.DataFrame conforming to the Wai schema with source='NOAA_PREDICTIONS'.

    Example
    -------
    >>> preds = load_noaa_predictions("9414290", "20240101", "20240131")

    How to use as a feature
    -----------------------
    Merge the predictions DataFrame onto your observations by timestamp,
    then include the 'water_level' column from predictions as a feature:

        obs = load_noaa_data("9414290", "20240101", "20240131")
        prd = load_noaa_predictions("9414290", "20240101", "20240131")
        merged = obs.merge(
            prd[["timestamp", "water_level"]].rename(
                columns={"water_level": "noaa_prediction"}
            ),
            on="timestamp", how="left",
        )
    """
    params = _noaa_api_params(
        station_id, begin_date, end_date, "predictions", datum, units, time_zone
    )
    resp = requests.get(NOAA_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_noaa_response(
        resp.json(), station_id, datum, units, source_label="NOAA_PREDICTIONS"
    )
