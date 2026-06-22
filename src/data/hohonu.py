"""Hohonu adapter and offline fixtures.

The live endpoint shape is intentionally narrow and configurable because Hohonu
deployments can expose customer-specific station and token arrangements.  The
adapter keeps credentials in environment variables and returns canonical Wai
observations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.data.canonicalize import canonicalize_frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HohonuConfig:
    """Configuration for the Hohonu observation adapter."""

    base_url: str = "https://api.hohonu.io"
    api_key_env: str = "HOHONU_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 0.5
    chunk_days: int = 7
    cache_dir: Path | None = Path(".cache/wai/hohonu")


class HohonuAdapter:
    """Fetch Hohonu observations and normalize them to Wai's schema."""

    def __init__(
        self,
        config: HohonuConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or HohonuConfig()
        self.session = session or requests.Session()

    def fetch_observations(
        self,
        station_id: str,
        start_time: object,
        end_time: object,
        *,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch Hohonu observations over a chunked time range."""

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Hohonu credentials are not configured. Set {self.config.api_key_env} "
                "or use mock_hohonu_observations for offline tests."
            )

        frames = []
        for chunk_start, chunk_end in _iter_chunks(start_time, end_time, self.config.chunk_days):
            cache_path = self._cache_path(station_id, chunk_start, chunk_end)
            if use_cache and cache_path and cache_path.exists():
                LOGGER.info("Reading Hohonu cache for station %s: %s", station_id, cache_path)
                payload = json.loads(cache_path.read_text())
            else:
                payload = self._request_json(
                    station_id=station_id,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    api_key=api_key,
                )
                if use_cache and cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(payload, sort_keys=True))

            frames.append(self.canonicalize_records(
                payload.get("data", payload.get("records", payload)),
                station_id=station_id,
                latitude=latitude,
                longitude=longitude,
                datum=datum,
                source="HOHONU",
            ))

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)

    def canonicalize_records(
        self,
        records: Iterable[dict],
        *,
        station_id: str,
        latitude: float,
        longitude: float,
        datum: str = "MLLW",
        source: str = "HOHONU_MOCK",
        retrieved_at: object | None = None,
    ) -> pd.DataFrame:
        """Normalize provider records without making a network call."""

        rows = []
        for record in records:
            rows.append({
                "timestamp": (
                    record.get("timestamp")
                    or record.get("timestamp_utc")
                    or record.get("observed_at")
                    or record.get("time")
                ),
                "station_id": str(record.get("station_id", station_id)),
                "water_level": (
                    record.get("water_level_m")
                    if record.get("water_level_m") is not None
                    else record.get("water_level", record.get("value"))
                ),
                "units": record.get("units", "m"),
                "lat": record.get("latitude", record.get("lat", latitude)),
                "lon": record.get("longitude", record.get("lon", longitude)),
                "datum": record.get("datum", datum),
                "qc_status": record.get("qc_status", record.get("quality", "unknown")),
                "qc_flags": record.get("qc_flags", record.get("flags", [])),
            })

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"Hohonu returned no records for station {station_id}")
        return canonicalize_frame(
            frame,
            source=source,
            record_type="observation",
            qc_status_col="qc_status",
            qc_flags_col="qc_flags",
            retrieved_at=retrieved_at,
        )

    def _request_json(
        self,
        *,
        station_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        api_key: str,
    ) -> dict:
        url = f"{self.config.base_url.rstrip('/')}/stations/{station_id}/observations"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 429:
                    delay = float(response.headers.get("Retry-After", self.config.backoff_seconds))
                    LOGGER.warning("Hohonu rate limited station %s; sleeping %.2fs", station_id, delay)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = self.config.backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Hohonu request failed for station %s on attempt %s/%s: %s",
                    station_id,
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(f"Hohonu request failed for station {station_id}: {last_error}")

    def _cache_path(
        self,
        station_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> Path | None:
        if self.config.cache_dir is None:
            return None
        key = hashlib.sha256(
            f"{station_id}|{start_time.isoformat()}|{end_time.isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        return self.config.cache_dir / f"{station_id}_{key}.json"


def mock_hohonu_observations(
    station_id: str = "HOHONU_TEST",
    *,
    start: str = "2024-01-01T00:00:00Z",
    periods: int = 240,
    freq: str = "6min",
    latitude: float = 21.3067,
    longitude: float = -157.8675,
    datum: str = "MLLW",
    qc_status: str = "pass",
    retrieved_at: object = "2024-01-02T00:00:00Z",
) -> pd.DataFrame:
    """Return deterministic local Hohonu-like observations for offline tests."""

    timestamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    t_h = np.arange(periods) * (pd.Timedelta(freq).total_seconds() / 3600.0)
    water = 0.55 * np.sin(2 * np.pi * t_h / 12.42) + 0.02 * np.sin(2 * np.pi * t_h / 3.0)
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_id,
        "water_level": water,
        "units": "m",
        "lat": latitude,
        "lon": longitude,
        "datum": datum,
        "qc_status": qc_status,
        "qc_flags": [[] for _ in range(periods)],
    })
    return canonicalize_frame(
        raw,
        source="HOHONU_MOCK",
        record_type="observation",
        qc_status_col="qc_status",
        qc_flags_col="qc_flags",
        retrieved_at=retrieved_at,
    )


def _iter_chunks(
    start_time: object,
    end_time: object,
    chunk_days: int,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_time).tz_convert("UTC") if pd.Timestamp(start_time).tzinfo else pd.Timestamp(start_time).tz_localize("UTC")
    end = pd.Timestamp(end_time).tz_convert("UTC") if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time).tz_localize("UTC")
    if end < start:
        raise ValueError("end_time must be after start_time")
    cursor = start
    delta = pd.Timedelta(days=chunk_days)
    while cursor <= end:
        chunk_end = min(cursor + delta, end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(microseconds=1)
