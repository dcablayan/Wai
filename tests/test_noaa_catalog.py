"""Tests for nationwide NOAA station discovery and snapshot validation."""

from __future__ import annotations

import json

import pytest
import requests

from src.data.noaa_catalog import (
    NOAAStationCatalogError,
    load_noaa_station_catalog,
    load_noaa_station_snapshot,
    parse_noaa_station_catalog,
    save_noaa_station_catalog,
)


def _water_payload():
    return {
        "count": 3,
        "stations": [
            {
                "id": "9414290",
                "name": "San Francisco",
                "lat": 37.8063,
                "lng": -122.4659,
                "state": "CA",
                "tidal": True,
                "greatlakes": False,
                "timezone": "PST",
                "affiliations": "NWLON",
                "tideType": "Mixed",
                "forecast": True,
            },
            {
                "id": "1612340",
                "name": "Honolulu",
                "lat": 21.3067,
                "lng": -157.8675,
                "state": "HI",
                "tidal": True,
                "greatlakes": False,
            },
            {
                "id": "9075014",
                "name": "Dry Dock",
                "lat": 46.7767,
                "lng": -92.092,
                "state": "MN",
                "tidal": False,
                "greatlakes": True,
            },
        ],
    }


def _tide_payload():
    return {
        "count": 2,
        "stations": [
            {"id": "9414290"},
            {"id": "1612340"},
        ],
    }


def test_catalog_parser_preserves_all_active_stations_and_capabilities():
    catalog = parse_noaa_station_catalog(
        _water_payload(),
        _tide_payload(),
        retrieved_at="2026-07-21T12:00:00Z",
    )

    assert catalog.count == 3
    assert catalog.tide_prediction_count == 2
    assert catalog.great_lakes_count == 1
    assert catalog.regions == ("CA", "HI", "MN")
    assert catalog.by_id["9414290"].has_tide_predictions is True
    assert catalog.by_id["9075014"].default_datum == "IGLD"
    assert catalog.by_id["9075014"].datum_options == ("IGLD", "LWD", "STND")


def test_catalog_parser_rejects_duplicate_ids_and_count_mismatch():
    duplicate = _water_payload()
    duplicate["stations"][1]["id"] = "9414290"
    with pytest.raises(NOAAStationCatalogError, match="duplicated"):
        parse_noaa_station_catalog(duplicate, _tide_payload())

    wrong_count = _water_payload()
    wrong_count["count"] = 99
    with pytest.raises(NOAAStationCatalogError, match="does not match"):
        parse_noaa_station_catalog(wrong_count, _tide_payload())


def test_catalog_snapshot_round_trip(tmp_path):
    catalog = parse_noaa_station_catalog(_water_payload(), _tide_payload())
    target = save_noaa_station_catalog(catalog, tmp_path / "stations.json")
    loaded = load_noaa_station_snapshot(target)

    assert loaded.count == catalog.count
    assert loaded.by_id["1612340"].label == "Honolulu, HI · 1612340"


def test_live_discovery_falls_back_to_bundled_noaa_snapshot(monkeypatch, tmp_path):
    catalog = parse_noaa_station_catalog(_water_payload(), _tide_payload())
    snapshot = save_noaa_station_catalog(catalog, tmp_path / "stations.json")

    class BrokenSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    import src.data.noaa_catalog as module

    monkeypatch.setattr(module, "DEFAULT_NOAA_STATION_SNAPSHOT", snapshot)
    loaded = load_noaa_station_catalog(session=BrokenSession())

    assert loaded.count == 3
    assert loaded.source == "bundled NOAA metadata snapshot"
    assert "offline" in (loaded.warning or "")


def test_snapshot_rejects_tampered_count(tmp_path):
    catalog = parse_noaa_station_catalog(_water_payload(), _tide_payload())
    target = save_noaa_station_catalog(catalog, tmp_path / "stations.json")
    payload = json.loads(target.read_text())
    payload["station_count"] = 999
    target.write_text(json.dumps(payload))

    with pytest.raises(NOAAStationCatalogError, match="count"):
        load_noaa_station_snapshot(target)
