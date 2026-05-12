"""Tests for IDW spatial interpolation (src/features/spatial.py)."""

import math

import pytest

from src.features.spatial import (
    haversine_km,
    idw_interpolate,
    interpolate_from_records,
)


def test_haversine_same_point():
    assert haversine_km(21.3, -157.8, 21.3, -157.8) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance():
    # Honolulu to San Francisco is roughly 3855 km
    dist = haversine_km(21.307, -157.858, 37.774, -122.419)
    assert 3800 < dist < 3920


def test_idw_single_station():
    stations = [(21.3, -157.8, 1.5)]
    result = idw_interpolate(stations, 21.5, -157.9)
    assert result == pytest.approx(1.5)


def test_idw_empty_returns_nan():
    result = idw_interpolate([], 21.3, -157.8)
    assert math.isnan(result)


def test_idw_exact_coincidence():
    stations = [(21.3, -157.8, 2.0), (22.0, -158.0, 3.0)]
    result = idw_interpolate(stations, 21.3, -157.8)
    assert result == pytest.approx(2.0)


def test_idw_equidistant():
    """Two equally distant stations should return their mean."""
    # Place query midpoint at (0, 0), stations at (1, 0) and (-1, 0)
    d1 = haversine_km(1.0, 0.0, 0.0, 0.0)
    d2 = haversine_km(-1.0, 0.0, 0.0, 0.0)
    assert abs(d1 - d2) < 1.0  # within 1 km for symmetry
    stations = [(1.0, 0.0, 10.0), (-1.0, 0.0, 20.0)]
    result = idw_interpolate(stations, 0.0, 0.0)
    assert 10.0 < result < 20.0
    # Equidistant → average of values
    assert result == pytest.approx(15.0, abs=0.5)


def test_idw_closer_station_has_more_influence():
    """A nearby station should dominate over a distant one."""
    near = (21.3, -157.8, 10.0)
    far = (30.0, -140.0, 0.0)
    result = idw_interpolate([near, far], 21.31, -157.81)
    # Should be much closer to 10.0 than 0.0
    assert result > 8.0


def test_idw_three_stations():
    stations = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 2.0),
        (0.0, 1.0, 3.0),
    ]
    result = idw_interpolate(stations, 0.5, 0.5)
    assert 1.0 <= result <= 3.0
    assert math.isfinite(result)


def test_interpolate_from_records():
    records = [
        {"lat": 21.3, "lon": -157.8, "value": 1.0},
        {"lat": 22.0, "lon": -158.0, "value": 2.0},
    ]
    result = interpolate_from_records(records, 21.6, -157.9)
    assert math.isfinite(result)
    assert 1.0 <= result <= 2.0


def test_interpolate_from_records_empty():
    result = interpolate_from_records([], 21.3, -157.8)
    assert math.isnan(result)
