"""Tests for rolling-origin evaluation."""

from __future__ import annotations

import json

import pandas as pd

from scripts.evaluate_rolling_origin import (
    evaluate_station_rolling_origin,
    make_rolling_origin_folds,
)


def _df(n: int = 600) -> pd.DataFrame:
    import numpy as np

    ts = pd.date_range("2024-01-01", periods=n, freq="6min", tz="UTC")
    t = np.arange(n) * 0.1
    wl = 0.5 * np.sin(2 * np.pi * t / 12.42)
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


def test_rolling_origin_fold_boundaries_have_no_leakage():
    folds = make_rolling_origin_folds(_df(), n_folds=3)
    assert len(folds) == 3
    for train, test, meta in folds:
        assert train["timestamp"].max() < test["timestamp"].min()
        assert meta["train_end"] < meta["test_start"]
        assert meta["n_train"] == len(train)
        assert meta["n_test"] == len(test)


def test_evaluate_station_rolling_origin_includes_fold_metadata():
    res = evaluate_station_rolling_origin(_df(), station_id="TEST", n_folds=2)
    assert res["n_folds"] == 2
    fold = res["folds"][0]
    for key in ("train_start", "train_end", "test_start", "test_end", "n_train", "n_test"):
        assert key in fold
    assert "rolling_persistence" in fold
    assert "harmonic_ridge" in fold


def test_rolling_origin_main_writes_outputs(tmp_path, monkeypatch):
    import scripts.evaluate_rolling_origin as ro

    monkeypatch.setattr(ro, "REPORTS_DIR", tmp_path)
    full = pd.concat([_df(500), _df(500).assign(station_id="TEST2")], ignore_index=True)
    monkeypatch.setattr(ro, "load_demo_data", lambda: full)

    ro.main(["--folds", "2"])

    assert (tmp_path / "rolling_origin_metrics.json").exists()
    assert (tmp_path / "rolling_origin_metrics.md").exists()
    data = json.loads((tmp_path / "rolling_origin_metrics.json").read_text())
    assert data["TEST"]["n_folds"] == 2
