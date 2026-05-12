"""Tests for src/models/metrics.py."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.metrics import compute_metrics, save_metrics


def test_perfect_forecast():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(a, a)
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert m["r2"] == pytest.approx(1.0, abs=1e-9)


def test_constant_offset():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f = a + 0.1
    m = compute_metrics(a, f)
    assert m["mae"] == pytest.approx(0.1, abs=1e-9)
    assert m["rmse"] == pytest.approx(0.1, abs=1e-9)


def test_corr_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(a, a)
    assert m["corr"] == pytest.approx(1.0, abs=1e-9)


def test_nan_inputs_handled():
    a = np.array([1.0, float("nan"), 3.0])
    f = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(a, f)
    assert not np.isnan(m["mae"])


def test_all_nan_returns_nan():
    a = np.array([float("nan"), float("nan")])
    f = np.array([float("nan"), float("nan")])
    m = compute_metrics(a, f)
    assert np.isnan(m["mae"])


def test_save_metrics_creates_file():
    data = {"model_a": {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "nse": 0.9, "corr": 0.95}}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub" / "metrics.json"
        save_metrics(data, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["model_a"]["mae"] == pytest.approx(0.1)
