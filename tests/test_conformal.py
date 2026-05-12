"""Tests for ConformalIntervals (src/models/conformal.py)."""

import math

import numpy as np
import pytest

from src.models.conformal import ConformalIntervals


def _make_residuals(n: int = 100, scale: float = 0.1, seed: int = 0) -> tuple:
    """Return (actual, predicted) arrays with known residual scale."""
    rng = np.random.default_rng(seed)
    actual = rng.standard_normal(n)
    predicted = actual + rng.normal(0, scale, n)
    return actual, predicted


def test_calibrate_sets_qhat():
    ci = ConformalIntervals(coverage=0.90)
    actual, predicted = _make_residuals()
    ci.calibrate(actual, predicted)
    assert ci.qhat > 0.0
    assert math.isfinite(ci.qhat)


def test_intervals_shape():
    ci = ConformalIntervals(coverage=0.90)
    actual, predicted = _make_residuals()
    ci.calibrate(actual, predicted)
    preds = np.array([1.0, 2.0, 3.0])
    lower, upper = ci.intervals(preds)
    assert lower.shape == preds.shape
    assert upper.shape == preds.shape
    assert np.all(upper > lower)


def test_intervals_symmetric():
    ci = ConformalIntervals(coverage=0.90)
    actual, predicted = _make_residuals()
    ci.calibrate(actual, predicted)
    p = np.array([0.5, 1.5])
    lower, upper = ci.intervals(p)
    assert np.allclose(p - lower, upper - p)


def test_empirical_coverage_at_least_nominal():
    """Coverage on the same data used for calibration must be >= nominal."""
    ci = ConformalIntervals(coverage=0.90)
    actual, predicted = _make_residuals(n=200, scale=0.05)
    ci.calibrate(actual, predicted)
    cov = ci.empirical_coverage(actual, predicted)
    assert cov >= 0.90, f"Expected coverage >= 0.90, got {cov:.3f}"


def test_qhat_before_calibrate_raises():
    ci = ConformalIntervals(coverage=0.90)
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = ci.qhat


def test_intervals_before_calibrate_raises():
    ci = ConformalIntervals(coverage=0.90)
    with pytest.raises(RuntimeError):
        ci.intervals(np.array([1.0]))


def test_invalid_coverage_raises():
    with pytest.raises(ValueError):
        ConformalIntervals(coverage=0.0)
    with pytest.raises(ValueError):
        ConformalIntervals(coverage=1.0)
    with pytest.raises(ValueError):
        ConformalIntervals(coverage=1.5)


def test_coverage_increases_with_nominal():
    """Higher nominal coverage should produce a wider qhat."""
    actual, predicted = _make_residuals(n=200, scale=0.1)
    ci_low = ConformalIntervals(coverage=0.80)
    ci_high = ConformalIntervals(coverage=0.95)
    ci_low.calibrate(actual, predicted)
    ci_high.calibrate(actual, predicted)
    assert ci_high.qhat >= ci_low.qhat


def test_all_nan_calibration_raises():
    ci = ConformalIntervals(coverage=0.90)
    with pytest.raises(ValueError, match="No valid calibration samples"):
        ci.calibrate(np.array([float("nan")]), np.array([float("nan")]))


def test_n_cal_stored():
    ci = ConformalIntervals(coverage=0.90)
    actual, predicted = _make_residuals(n=50)
    ci.calibrate(actual, predicted)
    assert ci.n_cal == 50
