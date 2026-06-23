"""Forecast context construction for Wai orchestration.

``build_forecast_context`` keeps its original signature for backward
compatibility, but now delegates to :class:`PreparedStationData` so that the
single-shot path and the historical-replay path share identical, vectorized
logic.  The previous ``iterrows`` residual-trend loop and repeated full-frame
copies have been replaced by ``searchsorted`` slicing and a single
``merge_asof`` residual alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.canonicalize import is_good_qc
from src.data.station_mapping import StationPair
from src.orchestration.prepared import (
    PreparedStationData,
    _nearest_index,
    _upper_bound,
)


@dataclass
class ForecastContext:
    """All data the rule-based router and numerical experts may inspect."""

    target_station_id: str
    paired_noaa_station_id: str
    forecast_time_utc: pd.Timestamp
    target_time_utc: pd.Timestamp
    horizon_minutes: int
    station_pair: StationPair
    datum: str
    latest_hohonu_observation: dict[str, Any] | None = None
    latest_noaa_observation: dict[str, Any] | None = None
    noaa_tide_prediction: dict[str, Any] | None = None
    local_tide_prediction: dict[str, Any] | None = None
    recent_hohonu_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    recent_noaa_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    noaa_tide_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    recent_hohonu_trend_m_per_hour: float | None = None
    recent_noaa_residual_m: float | None = None
    noaa_residual_trend_m_per_hour: float | None = None
    observation_freshness_seconds: dict[str, float] = field(default_factory=dict)
    qc_status: dict[str, str] = field(default_factory=dict)
    tide_phase: str | None = None
    pressure_trend: float | None = None
    wind_speed_mps: float | None = None
    wind_direction_deg: float | None = None
    neighboring_station_signals: dict[str, float] = field(default_factory=dict)
    recent_model_performance: dict[str, float] = field(default_factory=dict)
    model_disagreement_m: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Prepared, indexed data for lag-aware lookups (not part of the public API).
    prepared: PreparedStationData | None = None

    @property
    def hohonu_is_fresh(self) -> bool:
        return self.observation_freshness_seconds.get("hohonu", float("inf")) <= 60 * 60

    @property
    def noaa_is_fresh(self) -> bool:
        return self.observation_freshness_seconds.get("noaa", float("inf")) <= 3 * 60 * 60

    @property
    def hohonu_qc_ok(self) -> bool:
        return is_good_qc(self.qc_status.get("hohonu", "unknown"))

    @property
    def noaa_qc_ok(self) -> bool:
        return is_good_qc(self.qc_status.get("noaa", "unknown"))

    def noaa_residual_at(self, when: pd.Timestamp) -> float | None:
        """Observed NOAA residual at or before ``when`` (leakage-safe)."""

        if self.prepared is None:
            return self.recent_noaa_residual_m
        return self.prepared.residual_at(when)


def build_forecast_context(
    *,
    target_station_id: str,
    paired_noaa_station_id: str | None,
    horizon_minutes: int,
    forecast_time_utc: object,
    hohonu_observations: pd.DataFrame | None = None,
    noaa_observations: pd.DataFrame | None = None,
    noaa_tide_predictions: pd.DataFrame | None = None,
    station_pair: StationPair | None = None,
    local_tide_predictions: pd.DataFrame | None = None,
    recent_hours: float = 6.0,
    recent_model_performance: dict[str, float] | None = None,
    prepared: PreparedStationData | None = None,
) -> ForecastContext:
    """Build a leakage-safe context at one forecast origin.

    ``prepared`` may be supplied to reuse an already-indexed
    :class:`PreparedStationData` (the fast historical-replay path).  When absent,
    the frames are indexed once here.
    """

    if prepared is None:
        prepared = PreparedStationData.build(
            target_station_id=target_station_id,
            paired_noaa_station_id=paired_noaa_station_id,
            hohonu_observations=_empty_if_none(hohonu_observations),
            noaa_observations=_empty_if_none(noaa_observations),
            noaa_tide_predictions=_empty_if_none(noaa_tide_predictions),
            station_pair=station_pair,
            local_tide_predictions=local_tide_predictions,
        )
    return context_from_prepared(
        prepared,
        forecast_time_utc=forecast_time_utc,
        horizon_minutes=horizon_minutes,
        recent_hours=recent_hours,
        recent_model_performance=recent_model_performance,
    )


def context_from_prepared(
    prepared: PreparedStationData,
    *,
    forecast_time_utc: object,
    horizon_minutes: int,
    recent_hours: float = 6.0,
    recent_model_performance: dict[str, float] | None = None,
) -> ForecastContext:
    """Construct a context at one origin from prepared, indexed data."""

    forecast_time = _as_utc(forecast_time_utc)
    target_time = forecast_time + pd.Timedelta(minutes=horizon_minutes)
    recent_start = forecast_time - pd.Timedelta(hours=recent_hours)

    latest_hohonu = prepared.latest_before(prepared.hohonu, prepared.hohonu_ts, forecast_time)
    latest_noaa = prepared.latest_before(prepared.noaa, prepared.noaa_ts, forecast_time)
    target_tide = prepared.nearest(prepared.tide, prepared.tide_ts, target_time)
    local_target_tide = prepared.nearest(
        prepared.local_tide, prepared.local_tide_ts, target_time
    )

    recent_hohonu = prepared.recent_slice(
        prepared.hohonu, prepared.hohonu_ts, recent_start, forecast_time
    )
    recent_noaa = prepared.recent_slice(
        prepared.noaa, prepared.noaa_ts, recent_start, forecast_time
    )

    freshness: dict[str, float] = {}
    qc: dict[str, str] = {}
    if latest_hohonu:
        freshness["hohonu"] = (forecast_time - latest_hohonu["timestamp_utc"]).total_seconds()
        qc["hohonu"] = str(latest_hohonu.get("qc_status", "unknown"))
    if latest_noaa:
        freshness["noaa"] = (forecast_time - latest_noaa["timestamp_utc"]).total_seconds()
        qc["noaa"] = str(latest_noaa.get("qc_status", "unknown"))

    # Residual at the latest NOAA observation time (tide aligned at obs time).
    noaa_residual = None
    if latest_noaa is not None:
        noaa_residual = prepared.residual_at(latest_noaa["timestamp_utc"])

    context = ForecastContext(
        target_station_id=prepared.target_station_id,
        paired_noaa_station_id=prepared.paired_noaa_station_id,
        forecast_time_utc=forecast_time,
        target_time_utc=target_time,
        horizon_minutes=int(horizon_minutes),
        station_pair=prepared.station_pair,
        datum=prepared.datum,
        latest_hohonu_observation=latest_hohonu,
        latest_noaa_observation=latest_noaa,
        noaa_tide_prediction=target_tide,
        local_tide_prediction=local_target_tide,
        recent_hohonu_observations=recent_hohonu,
        recent_noaa_observations=recent_noaa,
        noaa_tide_predictions=prepared.tide,
        recent_hohonu_trend_m_per_hour=_trend_m_per_hour(recent_hohonu),
        recent_noaa_residual_m=noaa_residual,
        noaa_residual_trend_m_per_hour=_residual_trend_m_per_hour(prepared, recent_start, forecast_time),
        observation_freshness_seconds=freshness,
        qc_status=qc,
        tide_phase=_tide_phase(prepared.tide, prepared.tide_ts, target_time),
        recent_model_performance=dict(recent_model_performance or {}),
        prepared=prepared,
        diagnostics={
            "max_hohonu_input_time_utc": _max_time_iso(prepared.hohonu, prepared.hohonu_ts, forecast_time),
            "max_noaa_input_time_utc": _max_time_iso(prepared.noaa, prepared.noaa_ts, forecast_time),
            "target_tide_time_utc": _time_iso(target_tide),
        },
    )
    return context


def _empty_if_none(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if frame is not None else pd.DataFrame()


def _trend_m_per_hour(frame: pd.DataFrame) -> float | None:
    if len(frame) < 2:
        return None
    first = frame.iloc[0]
    last = frame.iloc[-1]
    hours = (last["timestamp_utc"] - first["timestamp_utc"]).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return float((last["water_level_m"] - first["water_level_m"]) / hours)


def _residual_trend_m_per_hour(
    prepared: PreparedStationData, start: pd.Timestamp, end: pd.Timestamp
) -> float | None:
    """Slope of the precomputed NOAA residual series over the recent window."""

    res = prepared.noaa_residual
    ts = prepared.noaa_residual_ts
    if ts.size < 2:
        return None
    hi = _upper_bound(ts, end)
    lo = _upper_bound(ts, start - pd.Timedelta(nanoseconds=1))
    window = res.iloc[lo:hi]
    if len(window) < 2:
        return None
    first = window.iloc[0]
    last = window.iloc[-1]
    hours = (last["timestamp_utc"] - first["timestamp_utc"]).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return float((last["residual_m"] - first["residual_m"]) / hours)


def _tide_phase(tide: pd.DataFrame, ts: np.ndarray, target_time: pd.Timestamp) -> str | None:
    if ts.size < 2:
        return None
    n_before = _upper_bound(ts, target_time)
    if n_before == 0 or n_before >= ts.size:
        return None
    before = float(tide.iloc[n_before - 1]["water_level_m"])
    after = float(tide.iloc[n_before]["water_level_m"])
    delta = after - before
    if np.isclose(delta, 0.0, atol=1e-4):
        return "slack"
    return "rising" if delta > 0 else "falling"


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _max_time_iso(
    frame: pd.DataFrame, ts: np.ndarray, forecast_time: pd.Timestamp
) -> str | None:
    n = _upper_bound(ts, forecast_time)
    if n == 0:
        return None
    return str(frame.iloc[n - 1]["timestamp_utc"])


def _time_iso(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    return str(record.get("timestamp_utc"))
