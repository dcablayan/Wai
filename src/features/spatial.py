"""Inverse-distance weighting (IDW) spatial interpolation across stations.

Only applicable when two or more stations have valid lat/lon coordinates.
IDW is a simple deterministic interpolation method that weights each station's
value by the inverse of its distance to the query point.

Limitations
-----------
- No nugget effect; queries exactly coinciding with a station return that
  station's value directly.
- Accuracy degrades for sparse station networks or large spatial gradients.
- No anisotropy corrections for coastal geometry (e.g. along-shore vs. cross-
  shore directions may have very different correlation lengths).
- Not designed for extrapolation beyond the station network boundary.
- Distance is computed using the haversine formula (great-circle) which is
  accurate to within ~0.5% for distances < 1000 km.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

StationPoint = Tuple[float, float, float]  # (lat, lon, value)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def idw_interpolate(
    stations: List[StationPoint],
    query_lat: float,
    query_lon: float,
    power: float = 2.0,
) -> float:
    """Inverse-distance weighted interpolation at a query location.

    Parameters
    ----------
    stations : list of (lat, lon, value)
        At least one entry required. Values must be finite floats.
    query_lat, query_lon : float
        Query location in decimal degrees.
    power : float
        Distance decay exponent (default 2 — standard IDW).

    Returns
    -------
    float
        Interpolated value. Returns the single station's value if len == 1.
        Returns nan if stations is empty.
    """
    if not stations:
        return float("nan")
    if len(stations) == 1:
        return float(stations[0][2])

    dists = [haversine_km(s[0], s[1], query_lat, query_lon) for s in stations]

    for i, d in enumerate(dists):
        if d < 1e-6:
            return float(stations[i][2])

    weights = [1.0 / (d ** power) for d in dists]
    total_w = sum(weights)
    if total_w == 0.0:
        return float("nan")
    return sum(w * s[2] for w, s in zip(weights, stations)) / total_w


def interpolate_from_records(
    station_records: List[dict],
    query_lat: float,
    query_lon: float,
    value_key: str = "value",
    power: float = 2.0,
) -> float:
    """Convenience wrapper accepting list of dicts with lat/lon/value keys.

    Parameters
    ----------
    station_records : list of dict
        Each dict must have 'lat', 'lon', and value_key fields.
    query_lat, query_lon : float
        Query location in decimal degrees.
    value_key : str
        Key for the value field in each record dict (default 'value').
    power : float
        Distance decay exponent.

    Returns
    -------
    float
        Interpolated value, or nan if records is empty.
    """
    pts: List[StationPoint] = [
        (float(r["lat"]), float(r["lon"]), float(r[value_key]))
        for r in station_records
        if r.get("lat") is not None and r.get("lon") is not None
    ]
    return idw_interpolate(pts, query_lat, query_lon, power=power)
