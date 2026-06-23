"""Tests for the prepared/indexed data layer and vectorized context build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.orchestration.context import build_forecast_context, context_from_prepared
from src.orchestration.prepared import PreparedStationData

STATION, NOAA = "HOHONU_TEST", "NOAA_TEST"


def _frames(periods=300, residual=0.1):
    return (
        mock_hohonu_observations(STATION, periods=periods),
        mock_noaa_observations(NOAA, periods=periods, residual_m=residual),
        mock_noaa_tide_predictions(NOAA, periods=int(periods * 1.4)),
    )


def test_prepared_and_single_shot_contexts_are_equivalent():
    h, n, t = _frames()
    pair = StationPair(STATION, NOAA)
    direct = build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=360,
        forecast_time_utc="2024-01-01T18:00:00Z", hohonu_observations=h,
        noaa_observations=n, noaa_tide_predictions=t, station_pair=pair,
    )
    prepared = PreparedStationData.build(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=h, noaa_observations=n, noaa_tide_predictions=t, station_pair=pair,
    )
    reused = context_from_prepared(prepared, forecast_time_utc="2024-01-01T18:00:00Z", horizon_minutes=360)

    assert reused.recent_noaa_residual_m == pytest.approx(direct.recent_noaa_residual_m)
    assert reused.tide_phase == direct.tide_phase
    assert reused.observation_freshness_seconds == direct.observation_freshness_seconds
    assert reused.latest_hohonu_observation["water_level_m"] == pytest.approx(
        direct.latest_hohonu_observation["water_level_m"]
    )
    assert reused.diagnostics["max_hohonu_input_time_utc"] == direct.diagnostics["max_hohonu_input_time_utc"]


def test_vectorized_residual_matches_manual_alignment():
    h, n, t = _frames(periods=120, residual=0.15)
    prepared = PreparedStationData.build(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=h, noaa_observations=n, noaa_tide_predictions=t,
        station_pair=StationPair(STATION, NOAA),
    )
    # Manual nearest-tide residual for every NOAA observation.
    tide = prepared.tide
    for _, obs in prepared.noaa.iterrows():
        idx = (tide["timestamp_utc"] - obs["timestamp_utc"]).abs().idxmin()
        manual = float(obs["water_level_m"]) - float(tide.loc[idx, "water_level_m"])
        got = prepared.residual_at(obs["timestamp_utc"])
        assert got == pytest.approx(manual, abs=1e-9)


def test_context_construction_is_leakage_safe():
    h, n, t = _frames()
    pair = StationPair(STATION, NOAA)
    forecast_time = pd.Timestamp("2024-01-01T12:00:00Z")
    ctx = build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=360,
        forecast_time_utc=forecast_time, hohonu_observations=h,
        noaa_observations=n, noaa_tide_predictions=t, station_pair=pair,
    )
    assert pd.Timestamp(ctx.diagnostics["max_hohonu_input_time_utc"]) <= forecast_time
    assert pd.Timestamp(ctx.diagnostics["max_noaa_input_time_utc"]) <= forecast_time
    # No recent observation may exceed the forecast origin.
    assert (ctx.recent_hohonu_observations["timestamp_utc"] <= forecast_time).all()
    assert (ctx.recent_noaa_observations["timestamp_utc"] <= forecast_time).all()


def test_future_observations_do_not_change_the_context():
    h, n, t = _frames()
    pair = StationPair(STATION, NOAA)
    kw = dict(target_station_id=STATION, paired_noaa_station_id=NOAA, horizon_minutes=180,
              forecast_time_utc="2024-01-01T06:00:00Z", noaa_tide_predictions=t, station_pair=pair)
    full = build_forecast_context(hohonu_observations=h, noaa_observations=n, **kw)
    # Truncate everything after the origin: the context must be identical.
    cutoff = pd.Timestamp("2024-01-01T06:00:00Z")
    h_trunc = h[pd.to_datetime(h["timestamp_utc"], utc=True) <= cutoff]
    n_trunc = n[pd.to_datetime(n["timestamp_utc"], utc=True) <= cutoff]
    trunc = build_forecast_context(hohonu_observations=h_trunc, noaa_observations=n_trunc, **kw)
    assert full.recent_noaa_residual_m == pytest.approx(trunc.recent_noaa_residual_m)
    assert full.latest_hohonu_observation["water_level_m"] == pytest.approx(
        trunc.latest_hohonu_observation["water_level_m"]
    )
