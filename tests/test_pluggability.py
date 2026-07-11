"""Tests for the provider-agnostic ingestion layer.

Covers the schema bridge, regularization, the data-source registry, datum
conversion, the station catalog, cadence-aware features, the harmonic
fallback expert, and the any-gauge CLI end to end.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.datum import convert_datum, harmonize_datums, lookup_offset_m
from src.data.canonicalize import DatumMismatchError
from src.data.hohonu import mock_hohonu_observations
from src.data.regularize import (
    despike_mad,
    infer_cadence_minutes,
    regularize_frame,
    steps_for_minutes,
)
from src.data.schema import to_canonical_frame, to_model_frame
from src.data.sources import (
    CSVSource,
    DataFrameSource,
    DataSource,
    get_source,
    list_sources,
    register_source,
)
from src.data.station_catalog import StationCatalog, StationMetadata
from src.data.station_mapping import StationPair
from src.features.engineering import add_lag_features, add_rolling_features
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context


def _model_frame(periods: int = 60, freq: str = "6min") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    t_h = np.arange(periods) * (pd.Timedelta(freq).total_seconds() / 3600.0)
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "GAUGE-1",
        "water_level": 0.5 * np.sin(2 * np.pi * t_h / 12.42),
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.86,
        "source": "TEST",
    })


# ---------------------------------------------------------------- schema

def test_schema_bridge_round_trip():
    model = _model_frame()
    canonical = to_canonical_frame(model, source="TEST")
    assert "timestamp_utc" in canonical.columns
    assert "water_level_m" in canonical.columns

    back = to_model_frame(canonical)
    assert list(back["water_level"]) == pytest.approx(list(model["water_level"]))
    assert (back["units"] == "m").all()


def test_to_model_frame_preserves_covariates():
    canonical = to_canonical_frame(_model_frame(), source="TEST")
    canonical["wind_speed_mps"] = 4.2
    back = to_model_frame(canonical)
    assert (back["wind_speed_mps"] == 4.2).all()


# ---------------------------------------------------------------- regularize

def test_infer_cadence_uses_mode_not_mean():
    ts = pd.Series(pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC"))
    # One 2-hour outage should not shift the modal cadence.
    ts.iloc[30:] = ts.iloc[30:] + pd.Timedelta(hours=2)
    assert infer_cadence_minutes(ts) == 5.0


def test_despike_flags_isolated_spike():
    values = pd.Series(np.sin(np.linspace(0, 6, 200)))
    values.iloc[100] += 3.0
    spikes = despike_mad(values)
    assert bool(spikes.iloc[100])
    assert spikes.sum() <= 3


def test_regularize_snaps_grid_interpolates_small_gaps_leaves_large():
    frame = _model_frame(periods=200, freq="5min")
    # Jitter one timestamp, drop one record (small gap), cut a 10-record outage.
    frame.loc[50, "timestamp"] += pd.Timedelta(seconds=20)
    frame = frame.drop(index=[80]).drop(index=range(120, 130)).reset_index(drop=True)

    out, report = regularize_frame(frame)
    assert report.cadence_minutes == 5.0
    deltas = out["timestamp"].diff().dropna().unique()
    assert len(deltas) == 1  # perfectly regular grid
    assert report.n_interpolated >= 1  # the single-record gap was filled
    assert report.n_gap_rows >= 7  # the outage stayed NaN
    assert out["is_interpolated"].sum() == report.n_interpolated


def test_regularize_works_on_canonical_frames():
    canonical = mock_hohonu_observations(periods=100)
    out, report = regularize_frame(canonical)
    assert report.cadence_minutes == 6.0
    assert "water_level_m" in out.columns


def test_steps_for_minutes():
    assert steps_for_minutes(60, 6) == 10
    assert steps_for_minutes(60, 15) == 4
    assert steps_for_minutes(1, 6) == 1


# ---------------------------------------------------------------- sources

def test_builtin_sources_registered():
    names = list_sources()
    for expected in ("csv", "dataframe", "hohonu", "noaa_coops"):
        assert expected in names


def test_csv_source_maps_columns_and_units(tmp_path):
    raw = pd.DataFrame({
        "obs_time": pd.date_range("2024-01-01", periods=10, freq="6min"),
        "level_ft": np.linspace(1.0, 2.0, 10),
    })
    path = tmp_path / "gauge.csv"
    raw.to_csv(path, index=False)

    source = CSVSource(
        path,
        column_map={"timestamp": "obs_time", "water_level": "level_ft"},
        units="ft",
        datum="NAVD88",
        station_id="GAUGE-9",
        latitude=21.0,
        longitude=-157.0,
    )
    frame = source.fetch_observations("GAUGE-9", None, None)
    assert len(frame) == 10
    assert frame["water_level_m"].iloc[0] == pytest.approx(1.0 * 0.3048)
    assert (frame["datum"] == "NAVD88").all()


def test_dataframe_source_windows_by_station_and_time():
    source = DataFrameSource(_model_frame(periods=20), source_label="TEST")
    frame = source.fetch_observations(
        "GAUGE-1", "2024-01-01T00:30:00Z", "2024-01-01T01:00:00Z"
    )
    assert len(frame) == 6
    assert source.fetch_observations("OTHER", None, None).empty


def test_register_source_rejects_duplicates_and_unknown_lookup():
    with pytest.raises(ValueError):
        @register_source("csv")
        class Clash(DataSource):  # pragma: no cover - registration must fail
            def fetch_observations(self, station_id, start, end, **kwargs):
                raise NotImplementedError

    with pytest.raises(KeyError):
        get_source("no_such_provider")


# ---------------------------------------------------------------- datum

def test_lookup_offset_derives_reverse():
    offsets = {("G1", "NAVD88", "MLLW"): 0.482}
    assert lookup_offset_m(offsets, "G1", "NAVD88", "MLLW") == 0.482
    assert lookup_offset_m(offsets, "G1", "MLLW", "NAVD88") == -0.482
    assert lookup_offset_m(offsets, "G1", "MLLW", "MLLW") == 0.0
    assert lookup_offset_m(offsets, "G2", "NAVD88", "MLLW") is None


def test_convert_datum_applies_offset():
    frame = to_canonical_frame(_model_frame(), source="TEST")
    frame["datum"] = "NAVD88"
    offsets = {("GAUGE-1", "NAVD88", "MLLW"): 0.5}
    converted = convert_datum(frame, to_datum="MLLW", offsets=offsets)
    assert (converted["datum"] == "MLLW").all()
    assert converted["water_level_m"].iloc[0] == pytest.approx(
        frame["water_level_m"].iloc[0] + 0.5
    )


def test_harmonize_datums_fails_closed_without_offsets():
    a = to_canonical_frame(_model_frame(), source="A")
    b = to_canonical_frame(_model_frame(), source="B")
    b["datum"] = "NAVD88"
    with pytest.raises(DatumMismatchError):
        harmonize_datums([a, b])


def test_harmonize_datums_converts_with_offsets():
    a = to_canonical_frame(_model_frame(), source="A")
    b = to_canonical_frame(_model_frame(), source="B")
    b["datum"] = "NAVD88"
    offsets = {("GAUGE-1", "NAVD88", "MLLW"): 0.5}
    out = harmonize_datums([a, b], to_datum="MLLW", offsets=offsets)
    assert all((f["datum"] == "MLLW").all() for f in out)


# ---------------------------------------------------------------- catalog

def test_station_catalog_round_trip(tmp_path):
    catalog = StationCatalog([
        StationMetadata(
            station_id="MY-GAUGE",
            source="csv",
            name="Test gauge",
            latitude=21.3,
            longitude=-157.86,
            datum="NAVD88",
            cadence_minutes=5.0,
            reference_station_id="1612340",
            reference_source="noaa_coops",
            residual_scale=0.9,
            datum_offsets={"NAVD88->MLLW": 0.482},
        ),
    ])
    path = tmp_path / "stations.json"
    catalog.save(path)
    loaded = StationCatalog.load(path)

    meta = loaded.get("MY-GAUGE")
    assert meta.cadence_minutes == 5.0
    pair = meta.to_station_pair()
    assert isinstance(pair, StationPair)
    assert pair.paired_noaa_station_id == "1612340"
    assert meta.datum_offset_table() == {("MY-GAUGE", "NAVD88", "MLLW"): 0.482}


def test_station_catalog_rejects_unknown_fields():
    with pytest.raises(ValueError):
        StationCatalog.from_dict(
            {"stations": [{"station_id": "X", "source": "csv", "wrong_field": 1}]}
        )


# ------------------------------------------------------ cadence-aware features

def test_lag_features_scale_with_cadence():
    df = _model_frame(periods=300)
    six = add_lag_features(df)
    fifteen = add_lag_features(df, cadence_minutes=15.0)
    # 60 physical minutes = 10 steps at 6-min cadence, 4 steps at 15-min.
    assert "water_level_lag10" in six.columns
    assert "water_level_lag4" in fifteen.columns
    assert "water_level_lag40" in six.columns  # 240 min
    assert "water_level_lag16" in fifteen.columns  # 240 min


def test_rolling_features_scale_with_cadence():
    df = _model_frame(periods=300)
    fifteen = add_rolling_features(df, cadence_minutes=15.0)
    assert "water_level_rmean4" in fifteen.columns  # 60 min
    assert "water_level_rmean96" in fifteen.columns  # 1440 min


# --------------------------------------------------- harmonic fallback expert

def _no_tide_context(periods: int = 520, horizon_minutes: int = 120):
    observations = mock_hohonu_observations("LONE-GAUGE", periods=periods)
    forecast_time = observations["timestamp_utc"].iloc[-1]
    return build_forecast_context(
        target_station_id="LONE-GAUGE",
        paired_noaa_station_id=None,
        horizon_minutes=horizon_minutes,
        forecast_time_utc=forecast_time,
        local_observations=observations,
        station_pair=StationPair("LONE-GAUGE", "LONE-GAUGE"),
    )


def test_harmonic_fallback_forecasts_without_tide_product():
    from src.experts import HarmonicFallbackExpert

    context = _no_tide_context()
    forecast = HarmonicFallbackExpert().forecast(context)
    assert forecast.ok, forecast.message
    # The mock is a pure M2-ish sine of amplitude 0.55; the fit should land
    # comfortably inside the physical range.
    assert abs(forecast.predicted_water_level_m) <= 0.7
    assert forecast.lower_m < forecast.predicted_water_level_m < forecast.upper_m
    assert "M2" in forecast.diagnostics["constituents"]


def test_harmonic_fallback_unavailable_on_short_history():
    from src.experts import HarmonicFallbackExpert

    context = _no_tide_context(periods=120)  # 12h < 48h minimum span
    forecast = HarmonicFallbackExpert().forecast(context)
    assert forecast.status == "unavailable"


def test_pipeline_serves_gauge_without_any_tide_product():
    context = _no_tide_context()
    result = ForecastPipeline(mode="mini").run(context)
    payload = result.to_dict()
    assert payload["forecast_m"] is not None
    assert payload["status"] == "available"


# ---------------------------------------------------------------- CLI

def test_run_gauge_forecast_cli_from_bare_csv(tmp_path, capsys):
    from scripts.run_gauge_forecast import main

    periods = 600
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="5min")
    t_h = np.arange(periods) * (5 / 60.0)
    raw = pd.DataFrame({
        "time": timestamps,
        "level_ft": 1.8 * np.sin(2 * np.pi * t_h / 12.42) + 3.0,
    })
    path = tmp_path / "vendor_export.csv"
    raw.to_csv(path, index=False)

    rc = main([
        "--csv", str(path),
        "--station-id", "VENDOR-GAUGE-7",
        "--timestamp-col", "time",
        "--water-level-col", "level_ft",
        "--units", "ft",
        "--horizon-minutes", "120",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["forecast_m"] is not None
    assert payload["status"] == "available"
    assert payload["input_regularization"]["cadence_minutes"] == 5.0
