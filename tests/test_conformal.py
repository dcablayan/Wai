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


# ── finite-sample exact kth-residual semantics ────────────────────────────────

def test_qhat_is_exact_kth_smallest_residual():
    """qhat must equal the kth smallest absolute residual where
    k = ceil((1-alpha)(n+1))."""
    a = np.zeros(10)
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # scores = |a - p| = [0.0, 0.1, 0.2, ..., 0.9] (already sorted)
    ci = ConformalIntervals(coverage=0.90)
    ci.calibrate(a, p)
    n = 10
    k = int(np.ceil(0.90 * (n + 1)))  # ceil(9.9) = 10
    expected = float(np.sort(np.abs(a - p))[k - 1])
    assert ci.k == k
    assert ci.qhat == pytest.approx(expected)


def test_qhat_higher_method_not_linear_interpolation():
    """Old linear-interp behaviour can land between two residuals; the new
    higher/kth-residual must equal one of the residuals exactly."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(50)
    p = a + 0.1 * rng.standard_normal(50)
    scores = np.sort(np.abs(a - p))
    ci = ConformalIntervals(coverage=0.85)
    ci.calibrate(a, p)
    # qhat must coincide with one of the sample residuals.
    assert np.any(np.isclose(scores, ci.qhat, atol=1e-12))


def test_qhat_falls_back_to_max_when_calibration_too_small():
    """If ceil((1-alpha)(n+1)) > n, qhat must default to the maximum residual."""
    a = np.zeros(5)
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    # ceil(0.99 * 6) = 6 > 5 → fall back to max residual = 0.4
    ci = ConformalIntervals(coverage=0.99)
    ci.calibrate(a, p)
    assert ci.qhat == pytest.approx(0.4)
    assert ci.k == 6


# ── stratified coverage (event / non-event) ───────────────────────────────────

def test_stratified_coverage_overall_matches_empirical_coverage():
    a, p = _make_residuals(n=300, scale=0.1)
    ci = ConformalIntervals(coverage=0.90)
    ci.calibrate(a, p)
    rep = ci.stratified_coverage(a, p)
    assert rep["coverage_overall"] == pytest.approx(ci.empirical_coverage(a, p))
    assert rep["nominal_coverage"] == 0.90
    assert rep["qhat"] == pytest.approx(ci.qhat)
    assert rep["k"] == ci.k
    assert rep["n_cal"] == ci.n_cal


def test_stratified_coverage_splits_by_event_threshold():
    """With an event threshold, the report must include event/non-event subsets."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal(400)
    p = a + 0.05 * rng.standard_normal(400)
    ci = ConformalIntervals(coverage=0.90)
    ci.calibrate(a, p)
    rep = ci.stratified_coverage(a, p, event_threshold=1.0)
    assert "coverage_event" in rep
    assert "coverage_non_event" in rep
    assert rep["n_event_samples"] + rep["n_non_event_samples"] == len(a)
    assert rep["event_threshold"] == 1.0


def test_stratified_coverage_handles_no_events():
    """When no test sample crosses the threshold, event coverage is NaN
    but overall and non-event are well-defined."""
    a, p = _make_residuals(n=100, scale=0.01)
    ci = ConformalIntervals(coverage=0.90)
    ci.calibrate(a, p)
    rep = ci.stratified_coverage(a, p, event_threshold=1e6)
    assert math.isnan(rep["coverage_event"])
    assert not math.isnan(rep["coverage_non_event"])
    assert rep["n_event_samples"] == 0


def test_empirical_coverage_on_held_out_test_is_finite():
    """Coverage must be meaningful on a separate test slice (the post-calibration
    period), which is the intended evaluation pathway."""
    rng = np.random.default_rng(13)
    cal_a = rng.standard_normal(150)
    cal_p = cal_a + 0.1 * rng.standard_normal(150)
    test_a = rng.standard_normal(150)
    test_p = test_a + 0.1 * rng.standard_normal(150)
    ci = ConformalIntervals(coverage=0.90).calibrate(cal_a, cal_p)
    cov = ci.empirical_coverage(test_a, test_p)
    assert 0.0 <= cov <= 1.0
