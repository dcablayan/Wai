"""Nationwide NOAA CO-OPS active water-level station catalog.

The live dashboard discovers stations from NOAA's Metadata API and keeps a
bundled NOAA-sourced snapshot for startup and offline development. Time-series
data remains on demand: selecting a station triggers the bounded Data API
request instead of fan-out requests across the entire national network.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import requests


NOAA_METADATA_API_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
)
DEFAULT_NOAA_STATION_SNAPSHOT = (
    Path(__file__).resolve().parents[2] / "data" / "noaa_active_stations.json"
)
CATALOG_SCHEMA_VERSION = 1


class NOAAStationCatalogError(ValueError):
    """Raised when NOAA station metadata violates the catalog contract."""


@dataclass(frozen=True)
class NOAAStation:
    """One active NOAA water-level station and its dashboard capabilities."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    state: str
    tidal: bool
    great_lakes: bool
    timezone: str = ""
    affiliations: str = ""
    tide_type: str = ""
    has_tide_predictions: bool = False
    forecast_flag: bool = False

    @property
    def label(self) -> str:
        location = f"{self.name}, {self.state}" if self.state else self.name
        return f"{location} · {self.station_id}"

    @property
    def default_datum(self) -> str:
        if self.great_lakes:
            return "IGLD"
        if self.tidal:
            return "MLLW"
        return "STND"

    @property
    def datum_options(self) -> tuple[str, ...]:
        if self.great_lakes:
            return ("IGLD", "LWD", "STND")
        if self.tidal:
            return ("MLLW", "MSL", "MHHW", "NAVD", "STND")
        return ("STND", "MSL", "NAVD")


@dataclass(frozen=True)
class NOAAStationCatalog:
    """Validated collection of all active NOAA water-level stations."""

    stations: tuple[NOAAStation, ...]
    retrieved_at: str
    source: str
    source_url: str = NOAA_METADATA_API_URL
    warning: str | None = None

    @property
    def count(self) -> int:
        return len(self.stations)

    @property
    def by_id(self) -> dict[str, NOAAStation]:
        return {station.station_id: station for station in self.stations}

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(sorted({station.state for station in self.stations if station.state}))

    @property
    def tide_prediction_count(self) -> int:
        return sum(station.has_tide_predictions for station in self.stations)

    @property
    def great_lakes_count(self) -> int:
        return sum(station.great_lakes for station in self.stations)


def parse_noaa_station_catalog(
    water_level_payload: dict,
    tide_prediction_payload: dict,
    *,
    retrieved_at: object | None = None,
    source: str = "live NOAA Metadata API",
) -> NOAAStationCatalog:
    """Validate and combine active-water-level and tide-capability metadata."""

    water_rows = _station_rows(water_level_payload, "waterlevels")
    tide_rows = _station_rows(tide_prediction_payload, "tidepredictions")
    tide_ids = {
        str(row.get("id", "")).strip()
        for row in tide_rows
        if str(row.get("id", "")).strip()
    }

    stations: list[NOAAStation] = []
    seen: set[str] = set()
    for index, row in enumerate(water_rows):
        station_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        state = str(row.get("state", "")).strip()
        if not station_id or not name or not state:
            raise NOAAStationCatalogError(
                f"NOAA water-level station row {index} is missing id, name, or region"
            )
        if station_id in seen:
            raise NOAAStationCatalogError(
                f"NOAA water-level station ID is duplicated: {station_id}"
            )
        seen.add(station_id)
        try:
            latitude = float(row["lat"])
            longitude = float(row["lng"])
        except (KeyError, TypeError, ValueError) as error:
            raise NOAAStationCatalogError(
                f"NOAA station {station_id} has invalid coordinates"
            ) from error
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise NOAAStationCatalogError(
                f"NOAA station {station_id} coordinates are out of range"
            )
        stations.append(
            NOAAStation(
                station_id=station_id,
                name=name,
                latitude=latitude,
                longitude=longitude,
                state=state,
                tidal=bool(row.get("tidal")),
                great_lakes=bool(row.get("greatlakes")),
                timezone=str(row.get("timezone") or "").strip(),
                affiliations=str(row.get("affiliations") or "").strip(),
                tide_type=str(row.get("tideType") or "").strip(),
                has_tide_predictions=station_id in tide_ids,
                forecast_flag=bool(row.get("forecast")),
            )
        )

    stations.sort(key=lambda item: (item.state, item.name, item.station_id))
    timestamp = _utc_timestamp(retrieved_at)
    return NOAAStationCatalog(
        stations=tuple(stations),
        retrieved_at=timestamp,
        source=source,
    )


def fetch_noaa_station_catalog(
    *,
    session: requests.Session | None = None,
    timeout_seconds: float = 30.0,
) -> NOAAStationCatalog:
    """Fetch all active NOAA water-level stations and tide capabilities."""

    client = session or requests.Session()
    common = {"units": "metric"}
    water_response = client.get(
        NOAA_METADATA_API_URL,
        params={**common, "type": "waterlevels"},
        timeout=timeout_seconds,
    )
    water_response.raise_for_status()
    tide_response = client.get(
        NOAA_METADATA_API_URL,
        params={**common, "type": "tidepredictions"},
        timeout=timeout_seconds,
    )
    tide_response.raise_for_status()
    return parse_noaa_station_catalog(
        water_response.json(),
        tide_response.json(),
    )


def save_noaa_station_catalog(
    catalog: NOAAStationCatalog,
    path: str | Path = DEFAULT_NOAA_STATION_SNAPSHOT,
) -> Path:
    """Persist a deterministic, reviewable NOAA metadata snapshot."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "retrieved_at": catalog.retrieved_at,
        "source": catalog.source,
        "source_url": catalog.source_url,
        "station_count": catalog.count,
        "stations": [asdict(station) for station in catalog.stations],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def load_noaa_station_snapshot(
    path: str | Path | None = None,
) -> NOAAStationCatalog:
    """Load and validate the bundled nationwide NOAA station snapshot."""

    source_path = Path(path or DEFAULT_NOAA_STATION_SNAPSHOT)
    payload = json.loads(source_path.read_text())
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise NOAAStationCatalogError("Unsupported NOAA station snapshot schema")
    rows = payload.get("stations")
    if not isinstance(rows, list) or not rows:
        raise NOAAStationCatalogError("NOAA station snapshot has no station rows")
    stations = tuple(NOAAStation(**row) for row in rows)
    if len({station.station_id for station in stations}) != len(stations):
        raise NOAAStationCatalogError("NOAA station snapshot contains duplicate IDs")
    expected_count = payload.get("station_count")
    if expected_count is not None and int(expected_count) != len(stations):
        raise NOAAStationCatalogError("NOAA station snapshot count does not match rows")
    return NOAAStationCatalog(
        stations=stations,
        retrieved_at=str(payload.get("retrieved_at") or "unknown"),
        source=str(payload.get("source") or "bundled NOAA snapshot"),
        source_url=str(payload.get("source_url") or NOAA_METADATA_API_URL),
    )


def load_noaa_station_catalog(
    *,
    allow_bundled_snapshot: bool = True,
    session: requests.Session | None = None,
) -> NOAAStationCatalog:
    """Load live metadata, falling back only to the bundled NOAA snapshot."""

    try:
        return fetch_noaa_station_catalog(session=session)
    except (requests.RequestException, NOAAStationCatalogError, ValueError) as error:
        if not allow_bundled_snapshot:
            raise
        snapshot = load_noaa_station_snapshot()
        return replace(
            snapshot,
            source="bundled NOAA metadata snapshot",
            warning=f"Live NOAA station discovery failed: {error}",
        )


def _station_rows(payload: dict, station_type: str) -> list[dict]:
    if not isinstance(payload, dict):
        raise NOAAStationCatalogError(f"NOAA {station_type} payload is not an object")
    rows = payload.get("stations") or payload.get("stationList")
    if not isinstance(rows, list) or not rows:
        raise NOAAStationCatalogError(f"NOAA {station_type} payload has no stations")
    advertised_count = payload.get("count")
    if advertised_count is not None and int(advertised_count) != len(rows):
        raise NOAAStationCatalogError(
            f"NOAA {station_type} count {advertised_count} does not match {len(rows)} rows"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise NOAAStationCatalogError(f"NOAA {station_type} contains a non-object row")
    return rows


def _utc_timestamp(value: object | None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value).replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
