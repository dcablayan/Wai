"""Config-driven station catalog for any gauge provider.

Replaces the hardcoded two-entry ``DEFAULT_STATION_PAIRS`` dict with a JSON
catalog describing each station: which provider serves it, where it is, its
native cadence, datum, an optional regional reference station, and any known
vertical datum offsets.  ``StationPair`` stays as the orchestration-facing
view, produced by :meth:`StationMetadata.to_station_pair`.

Catalog file format (``data/stations.json`` by default)::

    {
      "stations": [
        {
          "station_id": "MY-GAUGE-01",
          "name": "Harbor east dock",
          "source": "csv",
          "latitude": 21.3,
          "longitude": -157.86,
          "datum": "NAVD88",
          "units": "m",
          "cadence_minutes": 5,
          "reference_station_id": "1612340",
          "reference_source": "noaa_coops",
          "residual_scale": 0.9,
          "lag_minutes": 10,
          "datum_offsets": {"NAVD88->MLLW": 0.482}
        }
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.data.station_mapping import StationPair

DEFAULT_CATALOG_PATH = Path("data/stations.json")


@dataclass(frozen=True)
class StationMetadata:
    """Everything Wai needs to know to onboard one gauge."""

    station_id: str
    source: str
    name: str = ""
    latitude: float | None = None
    longitude: float | None = None
    datum: str = "MLLW"
    units: str = "m"
    cadence_minutes: float | None = None
    reference_station_id: str | None = None
    reference_source: str | None = None
    residual_scale: float = 1.0
    lag_minutes: int = 0
    #: additive offsets in meters keyed "FROM->TO", e.g. {"NAVD88->MLLW": 0.482}
    datum_offsets: Mapping[str, float] = field(default_factory=dict)
    #: passed to the source constructor (path, base_url, column_map, ...)
    source_config: Mapping[str, object] = field(default_factory=dict)

    def to_station_pair(self) -> StationPair:
        """Orchestration-facing view; the reference defaults to self."""

        return StationPair(
            target_station_id=self.station_id,
            paired_noaa_station_id=self.reference_station_id or self.station_id,
            target_name=self.name,
            residual_scale=self.residual_scale,
            lag_minutes=self.lag_minutes,
            datum=self.datum,
        )

    def datum_offset_table(self) -> dict[tuple[str, str, str], float]:
        """Offsets in the ``src.data.datum`` key layout for this station."""

        table: dict[tuple[str, str, str], float] = {}
        for key, value in self.datum_offsets.items():
            parts = [p.strip().upper() for p in str(key).split("->")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"Station {self.station_id}: datum offset key {key!r} "
                    "must look like 'FROM->TO'"
                )
            table[(self.station_id, parts[0], parts[1])] = float(value)
        return table


class StationCatalog:
    """Lookup over station metadata loaded from config or code."""

    def __init__(self, stations: list[StationMetadata] | None = None) -> None:
        self._stations: dict[str, StationMetadata] = {}
        for station in stations or []:
            self.add(station)

    def add(self, station: StationMetadata) -> None:
        if station.station_id in self._stations:
            raise ValueError(f"Duplicate station_id in catalog: {station.station_id!r}")
        self._stations[station.station_id] = station

    def get(self, station_id: str) -> StationMetadata:
        try:
            return self._stations[str(station_id)]
        except KeyError:
            raise KeyError(
                f"Station {station_id!r} is not in the catalog; known stations: "
                f"{sorted(self._stations)}"
            ) from None

    def __contains__(self, station_id: str) -> bool:
        return str(station_id) in self._stations

    def __len__(self) -> int:
        return len(self._stations)

    def station_ids(self) -> list[str]:
        return sorted(self._stations)

    def datum_offset_table(self) -> dict[tuple[str, str, str], float]:
        """Merged datum offsets across all catalog stations."""

        merged: dict[tuple[str, str, str], float] = {}
        for station in self._stations.values():
            merged.update(station.datum_offset_table())
        return merged

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StationCatalog":
        stations_raw = payload.get("stations")
        if not isinstance(stations_raw, list):
            raise ValueError('Catalog payload must contain a "stations" list')
        stations = []
        for entry in stations_raw:
            if not isinstance(entry, dict):
                raise ValueError(f"Catalog station entry must be an object: {entry!r}")
            unknown = set(entry) - _STATION_FIELDS
            if unknown:
                raise ValueError(
                    f"Unknown station fields {sorted(unknown)}; "
                    f"allowed: {sorted(_STATION_FIELDS)}"
                )
            stations.append(StationMetadata(**entry))
        return cls(stations)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> "StationCatalog":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        stations = []
        for station in self._stations.values():
            entry = {
                "station_id": station.station_id,
                "source": station.source,
                "name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "datum": station.datum,
                "units": station.units,
                "cadence_minutes": station.cadence_minutes,
                "reference_station_id": station.reference_station_id,
                "reference_source": station.reference_source,
                "residual_scale": station.residual_scale,
                "lag_minutes": station.lag_minutes,
                "datum_offsets": dict(station.datum_offsets),
                "source_config": dict(station.source_config),
            }
            stations.append({k: v for k, v in entry.items() if v not in (None, {}, "")})
        return {"stations": stations}

    def save(self, path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


_STATION_FIELDS = {
    "station_id",
    "source",
    "name",
    "latitude",
    "longitude",
    "datum",
    "units",
    "cadence_minutes",
    "reference_station_id",
    "reference_source",
    "residual_scale",
    "lag_minutes",
    "datum_offsets",
    "source_config",
}


def default_catalog() -> StationCatalog:
    """Catalog mirroring the legacy demo pairings, for backward compatibility."""

    return StationCatalog([
        StationMetadata(
            station_id="DEMO-HNL",
            source="dataframe",
            name="Demo Honolulu local station",
            latitude=21.3067,
            longitude=-157.8675,
            cadence_minutes=6.0,
            reference_station_id="1612340",
            reference_source="noaa_coops",
            residual_scale=0.85,
        ),
        StationMetadata(
            station_id="DEMO-SFO",
            source="dataframe",
            name="Demo San Francisco local station",
            latitude=37.8063,
            longitude=-122.4659,
            cadence_minutes=6.0,
            reference_station_id="9414290",
            reference_source="noaa_coops",
            residual_scale=0.9,
        ),
    ])
