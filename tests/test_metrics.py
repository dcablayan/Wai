"""Tests for src/models/metrics.py."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.metrics import bootstrap_ci, compute_event_metrics, compute_metrics, save_metrics


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


# ── compute_event_metrics ──────────────────────────────────────────────────────

def test_event_metrics_perfect_detection():
    """When forecast == actual, precision and recall are both 1.0."""
    a = np.array([0.5, 1.2, 1.5, 0.3, 1.1])
    threshold = 1.0
    m = compute_event_metrics(a, a, threshold)
    assert m["precision"] == pytest.approx(1.0, abs=1e-6)
    assert m["recall"] == pytest.approx(1.0, abs=1e-6)
    assert m["f1"] == pytest.approx(1.0, abs=1e-6)


def test_event_metrics_no_events():
    """When no actual samples exceed threshold, recall is NaN."""
    a = np.zeros(10)
    f = np.ones(10)
    m = compute_event_metrics(a, f, threshold=2.0)
    assert math.isnan(m["recall"]) or math.isnan(m["f1"])


def test_event_metrics_peak_error():
    """peak_error_m must be the largest abs error on actual-event steps."""
    a = np.array([0.0, 1.0, 1.5, 0.0])
    f = np.array([0.0, 0.8, 1.0, 0.0])
    threshold = 0.9
    m = compute_event_metrics(a, f, threshold)
    assert m["peak_error_m"] == pytest.approx(0.5, abs=1e-4)


def test_event_metrics_threshold_agree_all():
    """threshold_agree must be 1.0 when forecast and actual always agree on side."""
    a = np.array([0.2, 0.1, 1.2, 1.5])
    threshold = 0.5
    m = compute_event_metrics(a, a, threshold)
    assert m["threshold_agree"] == pytest.approx(1.0, abs=1e-6)


def test_event_metrics_required_keys():
    a = np.array([0.5, 1.5, 0.3])
    f = np.array([0.4, 1.6, 0.2])
    m = compute_event_metrics(a, f, threshold=1.0)
    for key in ("precision", "recall", "f1", "peak_error_m", "threshold_agree"):
        assert key in m, f"Missing key: {key}"


# ── bootstrap_ci ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_bounds_ordered():
    """Lower CI bound must be <= upper CI bound."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(200)
    f = a + 0.1 * rng.standard_normal(200)
    lo, hi = bootstrap_ci(a, f, metric="mae", n_boot=200)
    assert lo <= hi


def test_bootstrap_ci_mae_positive():
    """MAE bootstrap CI lower bound must be >= 0."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200)
    f = a + 0.05 * rng.standard_normal(200)
    lo, _ = bootstrap_ci(a, f, metric="mae", n_boot=200)
    assert lo >= 0.0


def test_bootstrap_ci_covers_point_estimate():
    """Bootstrap CI should bracket the point-estimate MAE most of the time."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal(300)
    f = a + 0.1 * rng.standard_normal(300)
    m = compute_metrics(a, f)
    lo, hi = bootstrap_ci(a, f, metric="mae", n_boot=500, seed=42)
    assert lo <= m["mae"] <= hi


def test_bootstrap_ci_empty_returns_nan():
    lo, hi = bootstrap_ci(np.array([]), np.array([]), metric="mae")
    assert math.isnan(lo) and math.isnan(hi)
