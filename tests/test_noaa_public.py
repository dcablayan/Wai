"""Tests for scripts/evaluate_noaa_public.py (offline mode only — no network calls)."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_noaa_public import (
    LiveNOAAFetchError,
    NOAA_STATIONS,
    evaluate_station,
    fetch_noaa_df,
    format_results_md,
    _make_mock_noaa_df,
)


# ── Mock data helpers ─────────────────────────────────────────────────────────

def _get_mock_df(n: int = 500) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    rng = np.random.default_rng(42)
    t_h = np.arange(n) * (6 / 60)
    wl = (
        0.5 * np.sin(2 * np.pi * t_h / 12.42)
        + 0.3 * np.sin(2 * np.pi * t_h / 24.0)
        + 0.05 * rng.standard_normal(n)
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "TEST",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "NOAA_COOPS_MOCK",
    })


# ── fetch_noaa_df offline mode ─────────────────────────────────────────────────

def test_fetch_noaa_df_offline_returns_dataframe():
    df, prov = fetch_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5, offline=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert prov["mock_used"] is True
    assert prov["data_source"] == "NOAA_COOPS_MOCK"
    assert prov["station_id"] == "9414290"
    assert prov["begin_date"] == "20240101"
    assert prov["end_date"] == "20240128"


def test_fetch_noaa_df_offline_schema():
    df, _ = fetch_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5, offline=True)
    for col in ("timestamp", "station_id", "water_level", "lat", "lon"):
        assert col in df.columns, f"Missing column: {col}"


# ── fail-hard / mock-permission semantics ─────────────────────────────────────

def test_fetch_noaa_live_raises_on_failure_without_allow_mock(monkeypatch):
    """Live mode (offline=False, allow_mock=False) must raise on any fetch error.

    Reason: silent fallback to mock data would let "real NOAA evaluation"
    reports advertise metrics that were actually computed from synthetic stand-ins.
    """
    import scripts.evaluate_noaa_public as ep

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated network error")

    monkeypatch.setattr(ep, "load_noaa_data", _boom)
    with pytest.raises(LiveNOAAFetchError, match="Live NOAA fetch failed"):
        fetch_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5,
                      offline=False, allow_mock=False)


def test_fetch_noaa_live_falls_back_when_allow_mock(monkeypatch):
    """`--allow-mock` permits a per-station fallback to mock data with a flag."""
    import scripts.evaluate_noaa_public as ep

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated network error")

    monkeypatch.setattr(ep, "load_noaa_data", _boom)
    df, prov = fetch_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5,
                             offline=False, allow_mock=True)
    assert len(df) > 0
    assert prov["mock_used"] is True
    assert prov["reason"] == "live_fetch_failed"
    assert "error" in prov


def test_fetch_noaa_offline_does_not_call_network(monkeypatch):
    """`offline=True` must never hit `load_noaa_data`."""
    import scripts.evaluate_noaa_public as ep

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("network call attempted in offline mode")

    monkeypatch.setattr(ep, "load_noaa_data", _should_not_be_called)
    df, prov = fetch_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5,
                             offline=True, allow_mock=False)
    assert prov["mock_used"] is True
    assert prov["reason"] == "offline_mode"
    assert len(df) > 0


def test_make_mock_noaa_df_water_level_numeric():
    df = _make_mock_noaa_df("9414290", "20240101", "20240128", 37.8, -122.5)
    assert pd.api.types.is_float_dtype(df["water_level"])
    assert not df["water_level"].isna().all()


# ── evaluate_station ──────────────────────────────────────────────────────────

def test_evaluate_station_returns_required_keys():
    df = _get_mock_df(n=500)
    res = evaluate_station(df, station_label="TEST", holdout_type="temporal")
    for key in ("station", "holdout_type", "n_train", "n_test",
                "persistence_rolling", "harmonic_ridge", "grad_boost"):
        assert key in res, f"Missing result key: {key}"


def test_evaluate_station_harmonic_ridge_has_metrics():
    df = _get_mock_df(n=500)
    res = evaluate_station(df, station_label="TEST")
    m = res.get("harmonic_ridge", {})
    assert "mae" in m, "HarmonicRidge metrics missing 'mae'"
    assert math.isfinite(m["mae"])


def test_evaluate_station_persistence_has_metrics():
    df = _get_mock_df(n=500)
    res = evaluate_station(df, station_label="TEST")
    m = res.get("persistence_rolling", {})
    assert "mae" in m
    assert math.isfinite(m["mae"])


def test_evaluate_station_too_short_returns_error():
    df = _get_mock_df(n=50)
    res = evaluate_station(df, station_label="TOO_SHORT")
    assert "error" in res


def test_evaluate_station_split_counts_correct():
    df = _get_mock_df(n=400)
    res = evaluate_station(df, station_label="TEST")
    n_train = res.get("n_train", 0)
    n_test = res.get("n_test", 0)
    # Both should be positive; together they don't exceed len(df)
    assert n_train > 0
    assert n_test > 0
    assert n_train + n_test <= len(df)


def test_evaluate_station_ci_is_tuple():
    df = _get_mock_df(n=500)
    res = evaluate_station(df, station_label="TEST")
    m = res.get("harmonic_ridge", {})
    ci = m.get("mae_ci_95")
    if ci is not None:
        assert len(ci) == 2
        lo, hi = ci
        assert lo <= hi


# ── NOAA station registry ─────────────────────────────────────────────────────

def test_noaa_stations_has_five_entries():
    assert len(NOAA_STATIONS) >= 5


def test_noaa_stations_have_valid_coords():
    for sid, label, lat, lon in NOAA_STATIONS:
        assert -90 <= lat <= 90, f"{label}: invalid lat {lat}"
        assert -180 <= lon <= 180, f"{label}: invalid lon {lon}"


# ── format_results_md ─────────────────────────────────────────────────────────

def test_format_results_md_contains_stations():
    df = _get_mock_df(n=500)
    res = {"test_station": evaluate_station(df, station_label="My Station")}
    md = format_results_md(res)
    assert "My Station" in md
    assert "HarmonicRidge" in md
    assert "Persistence" in md


def test_format_results_md_has_notes():
    md = format_results_md({})
    assert "real NOAA data" in md.lower() or "real" in md.lower()


# ── offline integration ───────────────────────────────────────────────────────

def test_main_runs_offline(tmp_path, monkeypatch):
    """Running main() in NOAA_OFFLINE=1 mode should produce both output files."""
    import scripts.evaluate_noaa_public as ep

    monkeypatch.setattr(ep, "REPORTS_DIR", tmp_path)
    monkeypatch.setenv("NOAA_OFFLINE", "1")
    # Re-import to pick up patched REPORTS_DIR
    ep.REPORTS_DIR = tmp_path

    ep.main([])

    assert (tmp_path / "noaa_public_metrics.json").exists()
    assert (tmp_path / "noaa_public_metrics.md").exists()

    with open(tmp_path / "noaa_public_metrics.json") as f:
        data = json.load(f)
    # New schema: _meta + 5 stations + storm = at least 7 top-level keys
    assert "_meta" in data
    assert data["_meta"]["offline_mode"] is True
    assert data["_meta"]["any_mock_used"] is True
    # Each station record must carry provenance fields
    station_entries = [v for k, v in data.items() if not k.startswith("_")]
    assert len(station_entries) >= 5
    for rec in station_entries:
        assert rec["data_source"] == "NOAA_COOPS_MOCK"
        assert rec["mock_used"] is True
        assert "begin_date" in rec
        assert "end_date" in rec


def test_main_live_mode_hard_fails_when_fetch_breaks(tmp_path, monkeypatch):
    """Live default mode (no env var, no flag) raises on the first fetch error."""
    import scripts.evaluate_noaa_public as ep

    monkeypatch.setattr(ep, "REPORTS_DIR", tmp_path)
    monkeypatch.delenv("NOAA_OFFLINE", raising=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated network outage")

    monkeypatch.setattr(ep, "load_noaa_data", _boom)
    with pytest.raises(LiveNOAAFetchError):
        ep.main([])  # no --allow-mock
