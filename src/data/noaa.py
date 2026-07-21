"""NOAA CO-OPS adapter with canonical schema output."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.data.canonicalize import CanonicalSchemaError, canonicalize_frame
from src.data.loader import NOAA_API_URL

LOGGER = logging.getLogger(__name__)

SUPPORTED_WEATHER_PRODUCTS = {
    "air_pressure",
    "air_temperature",
    "water_temperature",
    "wind",
}


def _weather_values(product: str, record: dict) -> dict[str, object]:
    """Map NOAA product-specific fields into canonical metric weather fields."""

    product = str(product).strip().lower()
    if product == "wind":
        return {
            "wind_speed_mps": record.get("s"),
            "wind_direction_deg": record.get("d"),
            "wind_gust_mps": record.get("g"),
        }
    value = record.get("v", record.get("value"))
    target = {
        "air_pressure": "air_pressure_hpa",
        "air_temperature": "air_temperature_c",
        "water_temperature": "water_temperature_c",
    }.get(product)
    if target is None:
        raise CanonicalSchemaError(f"Unsupported NOAA weather product: {product!r}")
    return {target: value}


@dataclass(frozen=True)
class NOAAConfig:
    """Configuration for NOAA CO-OPS requests."""

    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    chunk_days: int = 31
    cache_dir: Path | None = Path(".cache/wai/noaa")
    application: str = "wai_forecasting_orchestrator"


class NOAACoopsAdapter:
    """Fetch NOAA observations, tide predictions, and weather products."""

    def __init__(
        self,
        config: NOAAConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or NOAAConfig()
        self.session = session or requests.Session()

    def fetch_observations(
        self,
        station_id: str,
        begin: object,
        end: object,
        *,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        return self._fetch_product(
            station_id,
            begin,
            end,
            product="water_level",
            record_type="observation",
            latitude=latitude,
            longitude=longitude,
            datum=datum,
            source="NOAA_COOPS",
            use_cache=use_cache,
        )

    def fetch_tide_predictions(
        self,
        station_id: str,
        begin: object,
        end: object,
        *,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        return self._fetch_product(
            station_id,
            begin,
            end,
            product="predictions",
            record_type="tide_prediction",
            latitude=latitude,
            longitude=longitude,
            datum=datum,
            source="NOAA_PREDICTIONS",
            use_cache=use_cache,
        )

    def fetch_operational_forecast(
        self,
        station_id: str,
        begin: object,
        end: object,
        *,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch NOAA OFS water-level guidance when supported by a station."""

        return self._fetch_product(
            station_id,
            begin,
            end,
            product="ofs_water_level",
            record_type="forecast_guidance",
            latitude=latitude,
            longitude=longitude,
            datum=datum,
            source="NOAA_OFS_WATER_LEVEL",
            use_cache=use_cache,
        )

    def fetch_weather_observations(
        self,
        station_id: str,
        begin: object,
        end: object,
        *,
        product: str,
        latitude: float,
        longitude: float,
        datum: str = "METEOROLOGICAL",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch a supported NOAA product as canonical weather observations."""

        product = str(product).strip().lower()
        if product not in SUPPORTED_WEATHER_PRODUCTS:
            raise ValueError(
                f"Unsupported NOAA weather product {product!r}; expected one of "
                f"{sorted(SUPPORTED_WEATHER_PRODUCTS)}"
            )

        return self._fetch_product(
            station_id,
            begin,
            end,
            product=product,
            record_type="weather_observation",
            latitude=latitude,
            longitude=longitude,
            datum=datum,
            source=f"NOAA_{product.upper()}",
            use_cache=use_cache,
        )

    def canonicalize_payload(
        self,
        payload: dict,
        *,
        station_id: str,
        product: str,
        record_type: str,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        source: str = "NOAA_COOPS_MOCK",
        retrieved_at: object | None = None,
    ) -> pd.DataFrame:
        records = payload.get("data") or payload.get("predictions") or []
        if not records:
            raise ValueError(f"NOAA payload returned no records for station {station_id}")
        rows = []
        meta = payload.get("metadata", {})
        for record in records:
            row = {
                "timestamp": record.get("t") or record.get("timestamp"),
                "station_id": station_id,
                "units": "m",
                "lat": meta.get("lat", latitude),
                "lon": meta.get("lon", longitude),
                "datum": meta.get("datum", datum),
                "qc_status": record.get("q", record.get("qc_status", "unknown")),
                "qc_flags": record.get("f", record.get("qc_flags", [])),
            }
            if record_type == "weather_observation":
                row.update(_weather_values(product, record))
            else:
                row["water_level"] = record.get(
                    "v", record.get("value", record.get("water_level"))
                )
            rows.append(row)
        source_frame = pd.DataFrame(rows)
        weather_defaults = None
        water_level_col = "water_level"
        if record_type == "weather_observation":
            water_level_col = None
            weather_defaults = {
                column: source_frame[column]
                for column in (
                    "wind_speed_mps",
                    "wind_direction_deg",
                    "wind_gust_mps",
                    "air_pressure_hpa",
                    "air_temperature_c",
                    "water_temperature_c",
                )
                if column in source_frame
            }
        else:
            # NOAA uses blank/non-numeric values for individual missing samples.
            # Drop those provider sentinels while retaining valid points instead
            # of rejecting an otherwise useful station window.
            source_frame["water_level"] = pd.to_numeric(
                source_frame["water_level"], errors="coerce"
            )
            source_frame = source_frame.dropna(subset=["water_level"])
            if source_frame.empty:
                raise ValueError(
                    f"NOAA payload returned no numeric {product} values for "
                    f"station {station_id}"
                )
        return canonicalize_frame(
            source_frame,
            source=source,
            record_type=record_type,
            water_level_col=water_level_col,
            qc_status_col="qc_status",
            qc_flags_col="qc_flags",
            retrieved_at=retrieved_at,
            extra_defaults=weather_defaults,
        )

    def _fetch_product(
        self,
        station_id: str,
        begin: object,
        end: object,
        *,
        product: str,
        record_type: str,
        latitude: float,
        longitude: float,
        datum: str,
        source: str,
        use_cache: bool,
    ) -> pd.DataFrame:
        frames = []
        for chunk_start, chunk_end in _iter_date_chunks(begin, end, self.config.chunk_days):
            params = {
                "station": station_id,
                "product": product,
                "begin_date": chunk_start.strftime("%Y%m%d"),
                "end_date": chunk_end.strftime("%Y%m%d"),
                "datum": datum,
                "units": "metric",
                "time_zone": "gmt",
                "application": self.config.application,
                "format": "json",
            }
            cache_path = self._cache_path(station_id, product, params)
            if use_cache and cache_path and cache_path.exists():
                payload = json.loads(cache_path.read_text())
            else:
                payload = self._request_json(params)
                if use_cache and cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(payload, sort_keys=True))

            frames.append(self.canonicalize_payload(
                payload,
                station_id=station_id,
                product=product,
                record_type=record_type,
                latitude=latitude,
                longitude=longitude,
                datum=datum,
                source=source,
            ))
        return pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    def _request_json(self, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(
                    NOAA_API_URL,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 429:
                    delay = float(response.headers.get("Retry-After", self.config.backoff_seconds))
                    LOGGER.warning("NOAA rate limited station %s; sleeping %.2fs", params.get("station"), delay)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise RuntimeError(payload["error"].get("message", str(payload["error"])))
                return payload
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = self.config.backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "NOAA request failed for station %s product %s on attempt %s/%s: %s",
                    params.get("station"),
                    params.get("product"),
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"NOAA request failed for station {params.get('station')} "
            f"product {params.get('product')}: {last_error}"
        )

    def _cache_path(self, station_id: str, product: str, params: dict) -> Path | None:
        if self.config.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.config.cache_dir / f"{station_id}_{product}_{key}.json"


def mock_noaa_observations(
    station_id: str = "NOAA_TEST",
    *,
    start: str = "2024-01-01T00:00:00Z",
    periods: int = 240,
    freq: str = "6min",
    latitude: float = 21.3067,
    longitude: float = -157.8675,
    datum: str = "MLLW",
    residual_m: float = 0.08,
    retrieved_at: object = "2024-01-02T00:00:00Z",
) -> pd.DataFrame:
    timestamps, tide = _mock_tide(start, periods, freq)
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_id,
        "water_level": tide + residual_m,
        "units": "m",
        "lat": latitude,
        "lon": longitude,
        "datum": datum,
        "qc_status": "verified",
        "qc_flags": [[] for _ in range(periods)],
    })
    return canonicalize_frame(
        raw,
        source="NOAA_COOPS_MOCK",
        record_type="observation",
        qc_status_col="qc_status",
        qc_flags_col="qc_flags",
        retrieved_at=retrieved_at,
    )


def mock_noaa_tide_predictions(
    station_id: str = "NOAA_TEST",
    *,
    start: str = "2024-01-01T00:00:00Z",
    periods: int = 360,
    freq: str = "6min",
    latitude: float = 21.3067,
    longitude: float = -157.8675,
    datum: str = "MLLW",
    retrieved_at: object = "2024-01-01T00:00:00Z",
) -> pd.DataFrame:
    timestamps, tide = _mock_tide(start, periods, freq)
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_id,
        "water_level": tide,
        "units": "m",
        "lat": latitude,
        "lon": longitude,
        "datum": datum,
        "qc_status": "verified",
        "qc_flags": [[] for _ in range(periods)],
    })
    return canonicalize_frame(
        raw,
        source="NOAA_PREDICTIONS_MOCK",
        record_type="tide_prediction",
        qc_status_col="qc_status",
        qc_flags_col="qc_flags",
        retrieved_at=retrieved_at,
    )


def _mock_tide(start: str, periods: int, freq: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    timestamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    t_h = np.arange(periods) * (pd.Timedelta(freq).total_seconds() / 3600.0)
    tide = 0.5 * np.sin(2 * np.pi * t_h / 12.42) + 0.25 * np.sin(2 * np.pi * t_h / 24.0)
    return timestamps, tide


def _iter_date_chunks(
    begin: object,
    end: object,
    chunk_days: int,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    start = _as_utc_timestamp(begin)
    finish = _as_utc_timestamp(end)
    if finish < start:
        raise ValueError("end must be after begin")
    cursor = start.normalize()
    final = finish.normalize()
    delta = pd.Timedelta(days=chunk_days - 1)
    while cursor <= final:
        chunk_end = min(cursor + delta, final)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
