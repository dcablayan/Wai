"""Tests for scripts/evaluate_events.py (event-holdout evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_events import evaluate_station_events


def _synthetic_event_df(
    n_days: int = 30,
    surge_test_day: float = 25.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic single-station series with a surge in the test window."""
    rng = np.random.default_rng(seed)
    n = int(n_days * 24 * 60 / 6)  # 6-min cadence
    timestamps = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t_h = np.arange(n) * (6 / 60.0)
    wl = (
        0.55 * np.sin(2 * np.pi * t_h / 12.42)
        + 0.30 * np.sin(2 * np.pi * t_h / 24.0)
        + 0.02 * rng.standard_normal(n)
    )
    # Place a clearly above-threshold surge in the test span (75% split → day ~22.5)
    surge_center_h = surge_test_day * 24.0
    wl += 0.6 * np.exp(-((t_h - surge_center_h) ** 2) / (2 * 6.0 ** 2))
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": "TEST",
        "water_level": wl,
        "datum": "MLLW",
        "units": "m",
        "lat": 21.3,
        "lon": -157.8,
        "source": "DEMO_SYNTHETIC",
    })


def test_evaluate_station_events_returns_required_keys():
    df = _synthetic_event_df()
    res = evaluate_station_events(df, station_id="TEST", threshold_k=2.0)
    for key in (
        "station_id", "train_cutoff_ts", "train_threshold_m",
        "n_train", "n_test", "test_obs_episodes",
        "persistence_rolling", "harmonic_ridge", "grad_boost",
    ):
        assert key in res


def test_event_holdout_has_observed_episodes_in_test_window():
    """The synthetic generator must put at least one episode in the test span,
    otherwise event metrics are meaningless."""
    df = _synthetic_event_df()
    res = evaluate_station_events(df, station_id="TEST", threshold_k=2.0)
    assert res["test_obs_episodes"] >= 1


def test_event_holdout_threshold_is_train_only():
    """The reference threshold must be fit on the training window alone."""
    df = _synthetic_event_df()
    n = len(df)
    n_train = int(n * 0.75)
    train_wl = df["water_level"].iloc[:n_train]
    expected = float(train_wl.mean() + 2.0 * train_wl.std())
    res = evaluate_station_events(df, station_id="TEST", threshold_k=2.0)
    assert res["train_threshold_m"] == pytest.approx(expected, abs=1e-4)


def test_event_holdout_episode_metrics_have_all_keys():
    df = _synthetic_event_df()
    res = evaluate_station_events(df, station_id="TEST")
    ep = res["harmonic_ridge"]["episode"]
    for key in (
        "n_obs_episodes", "n_pred_episodes", "n_matched",
        "episode_precision", "episode_recall", "episode_f1",
        "peak_height_error_m", "peak_time_error_s", "lead_time_error_s",
    ):
        assert key in ep


def test_evaluate_events_main_writes_outputs(tmp_path, monkeypatch):
    import scripts.evaluate_events as ee

    monkeypatch.setattr(ee, "REPORTS_DIR", tmp_path)
    df = _synthetic_event_df(n_days=20)
    df2 = df.copy()
    df2["station_id"] = "TEST2"
    full = pd.concat([df, df2], ignore_index=True)
    monkeypatch.setattr(ee, "load_demo_data", lambda: full)

    ee.main([])

    assert (tmp_path / "event_metrics.json").exists()
    assert (tmp_path / "event_metrics.md").exists()

    data = json.loads((tmp_path / "event_metrics.json").read_text())
    assert "_meta" in data
    assert data["_meta"]["threshold_k"] == 2.0
    assert "TEST" in data and "TEST2" in data
