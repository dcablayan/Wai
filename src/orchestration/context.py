"""Forecast context construction for Wai orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.canonicalize import assert_compatible_datums, is_good_qc
from src.data.station_mapping import StationPair, get_station_pair


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


def build_forecast_context(
    *,
    target_station_id: str,
    paired_noaa_station_id: str | None,
    horizon_minutes: int,
    forecast_time_utc: object,
    hohonu_observations: pd.DataFrame,
    noaa_observations: pd.DataFrame,
    noaa_tide_predictions: pd.DataFrame,
    station_pair: StationPair | None = None,
    local_tide_predictions: pd.DataFrame | None = None,
    recent_hours: float = 6.0,
) -> ForecastContext:
    """Build a leakage-safe context at one forecast origin."""

    forecast_time = _as_utc(forecast_time_utc)
    target_time = forecast_time + pd.Timedelta(minutes=horizon_minutes)
    pair = station_pair or get_station_pair(
        target_station_id,
        paired_noaa_station_id=paired_noaa_station_id,
    )

    hohonu = _canonical_subset(
        hohonu_observations,
        station_id=target_station_id,
        record_type="observation",
        end_time=forecast_time,
    )
    noaa = _canonical_subset(
        noaa_observations,
        station_id=pair.paired_noaa_station_id,
        record_type="observation",
        end_time=forecast_time,
    )
    tide = _canonical_subset(
        noaa_tide_predictions,
        station_id=pair.paired_noaa_station_id,
        record_type="tide_prediction",
        end_time=None,
    )
    local_tide = _canonical_subset(
        local_tide_predictions,
        station_id=target_station_id,
        record_type="tide_prediction",
        end_time=None,
    ) if local_tide_predictions is not None else pd.DataFrame()

    datum = assert_compatible_datums(
        [frame for frame in (hohonu, noaa, tide, local_tide) if not frame.empty],
        label="forecast context",
    )

    recent_start = forecast_time - pd.Timedelta(hours=recent_hours)
    recent_hohonu = hohonu[hohonu["timestamp_utc"] >= recent_start].copy()
    recent_noaa = noaa[noaa["timestamp_utc"] >= recent_start].copy()

    latest_hohonu = _latest_record(hohonu)
    latest_noaa = _latest_record(noaa)
    target_tide = _nearest_record(tide, target_time)
    current_tide = _nearest_record(tide, forecast_time)
    local_target_tide = _nearest_record(local_tide, target_time)

    freshness = {}
    qc = {}
    if latest_hohonu:
        freshness["hohonu"] = (forecast_time - latest_hohonu["timestamp_utc"]).total_seconds()
        qc["hohonu"] = str(latest_hohonu.get("qc_status", "unknown"))
    if latest_noaa:
        freshness["noaa"] = (forecast_time - latest_noaa["timestamp_utc"]).total_seconds()
        qc["noaa"] = str(latest_noaa.get("qc_status", "unknown"))

    noaa_residual = None
    if latest_noaa and current_tide:
        noaa_residual = float(latest_noaa["water_level_m"] - current_tide["water_level_m"])

    context = ForecastContext(
        target_station_id=target_station_id,
        paired_noaa_station_id=pair.paired_noaa_station_id,
        forecast_time_utc=forecast_time,
        target_time_utc=target_time,
        horizon_minutes=int(horizon_minutes),
        station_pair=pair,
        datum=datum,
        latest_hohonu_observation=latest_hohonu,
        latest_noaa_observation=latest_noaa,
        noaa_tide_prediction=target_tide,
        local_tide_prediction=local_target_tide,
        recent_hohonu_observations=recent_hohonu,
        recent_noaa_observations=recent_noaa,
        noaa_tide_predictions=tide,
        recent_hohonu_trend_m_per_hour=_trend_m_per_hour(recent_hohonu),
        recent_noaa_residual_m=noaa_residual,
        noaa_residual_trend_m_per_hour=_residual_trend_m_per_hour(recent_noaa, tide),
        observation_freshness_seconds=freshness,
        qc_status=qc,
        tide_phase=_tide_phase(tide, target_time),
        diagnostics={
            "max_hohonu_input_time_utc": _max_time_iso(hohonu),
            "max_noaa_input_time_utc": _max_time_iso(noaa),
            "target_tide_time_utc": _time_iso(target_tide),
        },
    )
    return context


def _canonical_subset(
    frame: pd.DataFrame | None,
    *,
    station_id: str,
    record_type: str,
    end_time: pd.Timestamp | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    mask = (df["station_id"].astype(str) == str(station_id)) & (df["record_type"] == record_type)
    if end_time is not None:
        mask &= df["timestamp_utc"] <= end_time
    return df.loc[mask].sort_values("timestamp_utc").reset_index(drop=True)


def _latest_record(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    return frame.iloc[-1].to_dict()


def _nearest_record(frame: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, Any] | None:
    if frame.empty:
        return None
    idx = (frame["timestamp_utc"] - timestamp).abs().idxmin()
    return frame.loc[idx].to_dict()


def _trend_m_per_hour(frame: pd.DataFrame) -> float | None:
    if len(frame) < 2:
        return None
    first = frame.iloc[0]
    last = frame.iloc[-1]
    hours = (last["timestamp_utc"] - first["timestamp_utc"]).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return float((last["water_level_m"] - first["water_level_m"]) / hours)


def _residual_trend_m_per_hour(obs: pd.DataFrame, tide: pd.DataFrame) -> float | None:
    if len(obs) < 2 or tide.empty:
        return None
    rows = []
    for _, row in obs.iterrows():
        nearest = _nearest_record(tide, row["timestamp_utc"])
        if nearest:
            rows.append({
                "timestamp_utc": row["timestamp_utc"],
                "residual": float(row["water_level_m"] - nearest["water_level_m"]),
            })
    if len(rows) < 2:
        return None
    residuals = pd.DataFrame(rows)
    first = residuals.iloc[0]
    last = residuals.iloc[-1]
    hours = (last["timestamp_utc"] - first["timestamp_utc"]).total_seconds() / 3600.0
    if hours <= 0:
        return None
    return float((last["residual"] - first["residual"]) / hours)


def _tide_phase(tide: pd.DataFrame, target_time: pd.Timestamp) -> str | None:
    if len(tide) < 2:
        return None
    before = tide[tide["timestamp_utc"] <= target_time].tail(1)
    after = tide[tide["timestamp_utc"] > target_time].head(1)
    if before.empty or after.empty:
        return None
    delta = float(after.iloc[0]["water_level_m"] - before.iloc[0]["water_level_m"])
    if np.isclose(delta, 0.0, atol=1e-4):
        return "slack"
    return "rising" if delta > 0 else "falling"


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _max_time_iso(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    return str(frame["timestamp_utc"].max())


def _time_iso(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    return str(record.get("timestamp_utc"))
