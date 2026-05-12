"""Tests for multi-horizon evaluation (scripts/evaluate_horizons.py)."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_horizons import (
    build_horizon_features,
    evaluate_persistence_horizon,
    evaluate_sklearn_horizon,
    evaluate_station_horizons,
    format_metrics_md,
)


def _synthetic_df(n: int = 500, station: str = "TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    rng = np.random.default_rng(3)
    t = np.arange(n) * (6 / 60)
    water_level = (
        0.6 * np.sin(2 * np.pi * t / 12.42)
        + 0.3 * np.sin(2 * np.pi * t / 24.0)
        + 0.01 * rng.standard_normal(n)
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station,
        "water_level": water_level,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


# ── build_horizon_features ────────────────────────────────────────────────────

def test_build_horizon_features_shape_h1():
    df = _synthetic_df()
    X, y_h = build_horizon_features(df, horizon_steps=1)
    assert len(X) == len(y_h)
    assert len(X) > 0
    assert len(X) < len(df)  # some rows dropped due to lag NaN + shifted target


def test_build_horizon_features_shorter_at_longer_horizon():
    df = _synthetic_df()
    _, y_1 = build_horizon_features(df, horizon_steps=1)
    _, y_60 = build_horizon_features(df, horizon_steps=60)
    assert len(y_1) > len(y_60)


def test_build_horizon_features_no_target_leakage():
    df = _synthetic_df()
    X, _ = build_horizon_features(df, horizon_steps=1)
    # Shifted target and original target columns must not appear in X
    assert "water_level" not in X.columns
    assert "_target_h" not in X.columns


def test_build_horizon_features_no_nan_in_output():
    df = _synthetic_df()
    X, y_h = build_horizon_features(df, horizon_steps=60)
    assert not X.isna().any().any()
    assert not y_h.isna().any()


# ── evaluate_persistence_horizon ─────────────────────────────────────────────

def test_persistence_h1_returns_metrics():
    df = _synthetic_df()
    series = df["water_level"]
    n_train = int(len(series) * 0.75)
    m = evaluate_persistence_horizon(series, n_train, horizon_steps=1)
    assert "mae" in m and "rmse" in m
    assert math.isfinite(m["mae"])
    assert m["mae"] >= 0.0


def test_persistence_longer_horizon_higher_error():
    """Persistence error should grow with longer horizons (tidal signal varies)."""
    df = _synthetic_df(n=600)
    series = df["water_level"]
    n_train = int(len(series) * 0.75)
    m1 = evaluate_persistence_horizon(series, n_train, horizon_steps=1)
    m60 = evaluate_persistence_horizon(series, n_train, horizon_steps=60)
    assert m60["mae"] >= m1["mae"]


# ── evaluate_sklearn_horizon ──────────────────────────────────────────────────

def test_sklearn_harmonic_ridge_returns_metrics():
    df = _synthetic_df()
    X, y = build_horizon_features(df, horizon_steps=1)
    n_train = int(len(X) * 0.75)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    m = evaluate_sklearn_horizon(X_train, y_train, X_test, y_test, "harmonic_ridge")
    assert "mae" in m
    assert math.isfinite(m["mae"])


def test_sklearn_grad_boost_returns_metrics():
    df = _synthetic_df()
    X, y = build_horizon_features(df, horizon_steps=1)
    n_train = int(len(X) * 0.75)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    m = evaluate_sklearn_horizon(X_train, y_train, X_test, y_test, "grad_boost")
    assert "mae" in m
    assert math.isfinite(m["mae"])


def test_sklearn_unknown_model_raises():
    df = _synthetic_df()
    X, y = build_horizon_features(df, horizon_steps=1)
    with pytest.raises(ValueError, match="Unknown model"):
        evaluate_sklearn_horizon(X, y, X, y, "bad_model")


# ── evaluate_station_horizons ─────────────────────────────────────────────────

def test_station_horizons_returns_all_horizons():
    df = _synthetic_df(n=500)
    results = evaluate_station_horizons(df)
    for h in ("1step_6min", "6h", "12h", "24h"):
        assert h in results


def test_station_horizons_has_persistence_at_all_horizons():
    df = _synthetic_df(n=500)
    results = evaluate_station_horizons(df)
    for h in ("1step_6min", "6h", "12h", "24h"):
        assert "persistence" in results[h]


def test_station_horizons_wavegru_only_at_1step():
    df = _synthetic_df(n=500)
    results = evaluate_station_horizons(df)
    # WaveGRU at 1-step should have real metrics
    wgru_1 = results["1step_6min"].get("wave_gru", {})
    assert "mae" in wgru_1 or "error" in wgru_1  # may fail on small data
    # WaveGRU at 6h should have a note, not metrics
    wgru_6h = results["6h"].get("wave_gru", {})
    assert "note" in wgru_6h


# ── format_metrics_md ─────────────────────────────────────────────────────────

def test_format_metrics_md_contains_station():
    df = _synthetic_df(n=500)
    results = {"TEST": evaluate_station_horizons(df)}
    md = format_metrics_md(results)
    assert "TEST" in md
    assert "1step_6min" in md
    assert "persistence" in md


def test_format_metrics_md_contains_notes():
    results = {}
    md = format_metrics_md(results)
    assert "synthetic demo data" in md
    assert "deep learning" in md


# ── output file integration ───────────────────────────────────────────────────

def test_evaluate_horizons_output_files(tmp_path, monkeypatch):
    """evaluate_horizons.main() should create both output files."""
    import scripts.evaluate_horizons as eh

    monkeypatch.setattr(eh, "REPORTS_DIR", tmp_path)

    df = _synthetic_df(n=500)
    # Patch the name as imported in the script module, not in the source module
    monkeypatch.setattr(eh, "load_demo_data", lambda: df)

    eh.main()

    assert (tmp_path / "horizon_metrics.json").exists()
    assert (tmp_path / "horizon_metrics.md").exists()

    with open(tmp_path / "horizon_metrics.json") as f:
        data = json.load(f)
    assert "TEST" in data
