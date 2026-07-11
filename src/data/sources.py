"""Pluggable data-source abstraction and registry.

Any tide-gauge provider — a vendor API, a CSV export, an in-memory frame —
plugs into Wai by implementing :class:`DataSource` and registering under a
name.  Every source returns frames in the canonical schema
(``src/data/canonicalize.py``), so nothing downstream knows or cares which
provider produced the data.

Built-in sources:

- ``csv``: any delimited file, with a configurable column mapping.
- ``dataframe``: an in-memory frame (tests, notebooks, custom loaders).
- ``hohonu``: the Hohonu REST adapter.
- ``noaa_coops``: the NOAA CO-OPS adapter (observations and tide predictions).

Registering a new provider::

    @register_source("my_vendor")
    class MyVendorSource(DataSource):
        def fetch_observations(self, station_id, start, end, **kwargs):
            payload = ...  # provider-specific fetch
            return canonicalize_frame(payload, source="MY_VENDOR",
                                      record_type="observation", ...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.data.canonicalize import canonicalize_frame, utc_timestamp

_SOURCE_REGISTRY: dict[str, type["DataSource"]] = {}


class DataSource(ABC):
    """One provider of water-level records in the canonical schema."""

    #: registry key; set by :func:`register_source`.
    name: str = ""

    @abstractmethod
    def fetch_observations(
        self,
        station_id: str,
        start: object,
        end: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Return canonical ``record_type="observation"`` rows for a station."""

    def fetch_tide_predictions(
        self,
        station_id: str,
        start: object,
        end: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Return canonical ``record_type="tide_prediction"`` rows.

        Optional: providers without a tide-prediction product raise
        ``NotImplementedError`` and the pipeline falls back to internally
        synthesized harmonics.
        """

        raise NotImplementedError(f"{self.name or type(self).__name__} has no tide-prediction product")


def register_source(name: str) -> Callable[[type[DataSource]], type[DataSource]]:
    """Class decorator registering a :class:`DataSource` under ``name``."""

    key = name.strip().lower()

    def _register(cls: type[DataSource]) -> type[DataSource]:
        if not issubclass(cls, DataSource):
            raise TypeError(f"{cls.__name__} must subclass DataSource")
        existing = _SOURCE_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"Source name {key!r} already registered to {existing.__name__}")
        cls.name = key
        _SOURCE_REGISTRY[key] = cls
        return cls

    return _register


def get_source(name: str, **kwargs: object) -> DataSource:
    """Instantiate a registered source by name."""

    key = name.strip().lower()
    if key not in _SOURCE_REGISTRY:
        raise KeyError(
            f"Unknown data source {name!r}; registered sources: {list_sources()}"
        )
    return _SOURCE_REGISTRY[key](**kwargs)


def list_sources() -> list[str]:
    """Return registered source names, sorted."""

    return sorted(_SOURCE_REGISTRY)


@register_source("dataframe")
class DataFrameSource(DataSource):
    """Serve canonical records from in-memory provider frames.

    ``frame`` may be in the canonical vocabulary already or in any layout
    accepted by the column mapping (same knobs as :class:`CSVSource`).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        source_label: str = "DATAFRAME",
        record_type: str = "observation",
        column_map: Mapping[str, str] | None = None,
        units: str = "m",
        datum: str = "MLLW",
        latitude: float | None = None,
        longitude: float | None = None,
        station_id: str | None = None,
        tide_frame: pd.DataFrame | None = None,
    ) -> None:
        self._frame = _coerce_to_canonical(
            frame,
            source_label=source_label,
            record_type=record_type,
            column_map=column_map,
            units=units,
            datum=datum,
            latitude=latitude,
            longitude=longitude,
            station_id=station_id,
        )
        self._tide = (
            _coerce_to_canonical(
                tide_frame,
                source_label=source_label,
                record_type="tide_prediction",
                column_map=column_map,
                units=units,
                datum=datum,
                latitude=latitude,
                longitude=longitude,
                station_id=station_id,
            )
            if tide_frame is not None
            else None
        )

    def fetch_observations(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        return _window(self._frame, station_id, start, end)

    def fetch_tide_predictions(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        if self._tide is None:
            raise NotImplementedError("No tide-prediction frame supplied")
        return _window(self._tide, station_id, start, end)


@register_source("csv")
class CSVSource(DataSource):
    """Serve canonical records from any delimited gauge export.

    ``column_map`` maps Wai's expected keys (``timestamp``, ``water_level``,
    and optionally ``station_id``, ``lat``, ``lon``, ``datum``, ``units``,
    ``qc_status``, ``qc_flags``) to the file's actual column names.  Constant
    metadata missing from the file (a station id, coordinates, datum, units)
    is supplied via keyword arguments instead.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_label: str = "CSV",
        record_type: str = "observation",
        column_map: Mapping[str, str] | None = None,
        units: str = "m",
        datum: str = "MLLW",
        latitude: float | None = None,
        longitude: float | None = None,
        station_id: str | None = None,
        read_csv_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.path = Path(path)
        raw = pd.read_csv(self.path, **dict(read_csv_kwargs or {}))
        self._frame = _coerce_to_canonical(
            raw,
            source_label=source_label,
            record_type=record_type,
            column_map=column_map,
            units=units,
            datum=datum,
            latitude=latitude,
            longitude=longitude,
            station_id=station_id,
        )

    def fetch_observations(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        return _window(self._frame, station_id, start, end)


@register_source("hohonu")
class HohonuSource(DataSource):
    """Registry wrapper over :class:`src.data.hohonu.HohonuAdapter`."""

    def __init__(self, **adapter_kwargs: object) -> None:
        from src.data.hohonu import HohonuAdapter

        self._adapter = HohonuAdapter(**adapter_kwargs)

    def fetch_observations(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        return self._adapter.fetch_observations(station_id, start, end, **kwargs)


@register_source("noaa_coops")
class NOAACoopsSource(DataSource):
    """Registry wrapper over :class:`src.data.noaa.NOAACoopsAdapter`."""

    def __init__(self, **adapter_kwargs: object) -> None:
        from src.data.noaa import NOAACoopsAdapter

        self._adapter = NOAACoopsAdapter(**adapter_kwargs)

    def fetch_observations(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        return self._adapter.fetch_observations(station_id, begin=start, end=end, **kwargs)

    def fetch_tide_predictions(
        self, station_id: str, start: object, end: object, **kwargs: object
    ) -> pd.DataFrame:
        return self._adapter.fetch_tide_predictions(station_id, begin=start, end=end, **kwargs)


def _coerce_to_canonical(
    frame: pd.DataFrame,
    *,
    source_label: str,
    record_type: str,
    column_map: Mapping[str, str] | None,
    units: str,
    datum: str,
    latitude: float | None,
    longitude: float | None,
    station_id: str | None,
) -> pd.DataFrame:
    from src.data.schema import is_canonical

    if is_canonical(frame):
        return frame.copy()

    df = frame.copy()
    if column_map:
        rename = {src: dst for dst, src in column_map.items() if src in df.columns}
        df = df.rename(columns=rename)

    if "station_id" not in df.columns:
        if station_id is None:
            raise ValueError("station_id column missing and no station_id constant given")
        df["station_id"] = station_id
    if "lat" not in df.columns:
        df["lat"] = latitude if latitude is not None else float("nan")
    if "lon" not in df.columns:
        df["lon"] = longitude if longitude is not None else float("nan")
    if "datum" not in df.columns:
        df["datum"] = datum

    return canonicalize_frame(
        df,
        source=source_label,
        record_type=record_type,
        units=units,
        qc_status_col="qc_status" if "qc_status" in df.columns else None,
        qc_flags_col="qc_flags" if "qc_flags" in df.columns else None,
    )


def _window(
    frame: pd.DataFrame, station_id: str, start: object, end: object
) -> pd.DataFrame:
    mask = frame["station_id"].astype(str) == str(station_id)
    out = frame.loc[mask]
    if start is not None:
        out = out[out["timestamp_utc"] >= utc_timestamp(start)]
    if end is not None:
        out = out[out["timestamp_utc"] <= utc_timestamp(end)]
    return out.reset_index(drop=True)
