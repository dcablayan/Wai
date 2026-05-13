"""Tests for scripts/evaluate_conformal.py report generation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.evaluate_conformal import evaluate_station_conformal


def _df(n: int = 900) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t = np.arange(n) * 0.1
    wl = (
        0.5 * np.sin(2 * np.pi * t / 12.42)
        + 0.2 * np.sin(2 * np.pi * t / 24.0)
        + 0.02 * np.random.default_rng(4).standard_normal(n)
    )
    return pd.DataFrame({
        "timestamp": ts,
        "station_id": "TEST",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


def test_evaluate_station_conformal_required_fields():
    res = evaluate_station_conformal(_df(), station_id="TEST", nominal_coverage=0.9)
    assert "models" in res
    for model_name in ("harmonic_ridge", "grad_boost"):
        m = res["models"][model_name]
        for key in (
            "nominal_coverage", "empirical_coverage", "event_coverage",
            "non_event_coverage", "qhat", "n_cal", "mean_interval_width",
        ):
            assert key in m
        assert m["nominal_coverage"] == 0.9
        assert m["n_cal"] > 0
        assert m["mean_interval_width"] >= 0


def test_evaluate_conformal_main_writes_outputs(tmp_path, monkeypatch):
    import scripts.evaluate_conformal as ec

    monkeypatch.setattr(ec, "REPORTS_DIR", tmp_path)
    full = pd.concat([_df(700), _df(700).assign(station_id="TEST2")], ignore_index=True)
    monkeypatch.setattr(ec, "load_demo_data", lambda: full)

    ec.main([])

    assert (tmp_path / "conformal_metrics.json").exists()
    assert (tmp_path / "conformal_metrics.md").exists()
    data = json.loads((tmp_path / "conformal_metrics.json").read_text())
    assert data["_meta"]["nominal_coverage"] == 0.9
    assert "harmonic_ridge" in data["TEST"]["models"]
