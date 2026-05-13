"""Tests for src/models/metrics.py."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.metrics import (
    block_bootstrap_ci,
    bootstrap_ci,
    compute_episode_metrics,
    compute_event_metrics,
    compute_metrics,
    save_metrics,
    skill_score,
)


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


def test_skill_score_positive_for_error_reduction():
    assert skill_score(0.25, 0.5) == pytest.approx(0.5)


def test_skill_score_negative_when_worse_than_reference():
    assert skill_score(0.75, 0.5) == pytest.approx(-0.5)


def test_skill_score_invalid_reference_returns_nan():
    assert math.isnan(skill_score(0.1, 0.0))


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


# ── block_bootstrap_ci ────────────────────────────────────────────────────────

def _ar1_residuals(n: int, rho: float = 0.9, seed: int = 0) -> np.ndarray:
    """Generate AR(1) residuals — strongly autocorrelated noise."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def test_block_bootstrap_ci_returns_block_length_and_metadata():
    """Block bootstrap output must carry the block length for auditability."""
    a = _ar1_residuals(400)
    f = np.zeros_like(a)
    res = block_bootstrap_ci(a, f, metric="mae", n_boot=200)
    assert "lower" in res and "upper" in res
    assert "block_length" in res
    assert res["block_length"] >= 2
    assert res["method"] == "circular_block"
    assert res["n_samples"] == 400


def test_block_bootstrap_ci_bounds_ordered():
    a = _ar1_residuals(300)
    f = a + 0.1 * np.random.default_rng(0).standard_normal(300)
    res = block_bootstrap_ci(a, f, metric="mae", n_boot=200)
    assert res["lower"] <= res["upper"]


def test_block_bootstrap_ci_wider_than_iid_for_correlated_data():
    """Block CI should be (typically) WIDER than the iid CI on AR(1) residuals,
    because iid bootstrap fails to capture autocorrelation."""
    rng = np.random.default_rng(123)
    n = 500
    a = _ar1_residuals(n, rho=0.95, seed=1)
    f = a + 0.2 * rng.standard_normal(n)
    iid_lo, iid_hi = bootstrap_ci(a, f, metric="mae", n_boot=600, seed=7)
    block = block_bootstrap_ci(a, f, metric="mae", n_boot=600, seed=7,
                               block_length=40)
    assert (block["upper"] - block["lower"]) >= (iid_hi - iid_lo) * 0.8, (
        f"block CI width {block['upper']-block['lower']:.4f} should not be "
        f"much narrower than iid CI width {iid_hi-iid_lo:.4f}"
    )


def test_block_bootstrap_ci_respects_explicit_block_length():
    a = _ar1_residuals(200)
    f = np.zeros_like(a)
    res = block_bootstrap_ci(a, f, metric="mae", n_boot=100, block_length=25)
    assert res["block_length"] == 25


def test_block_bootstrap_ci_moving_mode():
    """The non-circular variant must report `moving_block` as its method."""
    a = _ar1_residuals(200)
    f = a + 0.05 * np.random.default_rng(0).standard_normal(200)
    res = block_bootstrap_ci(a, f, metric="mae", n_boot=100,
                             block_length=20, circular=False)
    assert res["method"] == "moving_block"


def test_block_bootstrap_ci_handles_empty():
    res = block_bootstrap_ci(np.array([]), np.array([]), metric="mae")
    assert math.isnan(res["lower"]) and math.isnan(res["upper"])
    assert res["n_samples"] == 0


def test_block_bootstrap_ci_covers_point_estimate():
    a = _ar1_residuals(400)
    f = a + 0.1 * np.random.default_rng(2).standard_normal(400)
    point = compute_metrics(a, f)["mae"]
    res = block_bootstrap_ci(a, f, metric="mae", n_boot=400, seed=42)
    assert res["lower"] <= point <= res["upper"]


# ── compute_episode_metrics ───────────────────────────────────────────────────

def _make_episode_series(events: list, n: int = 200, baseline: float = 0.0) -> np.ndarray:
    """Build an array of length n with Gaussian-shaped events at given centers.

    events: list of (center, half_width, peak)
    """
    x = np.full(n, baseline, dtype=float)
    t = np.arange(n)
    for c, hw, p in events:
        x = x + p * np.exp(-((t - c) ** 2) / (2 * hw ** 2))
    return x


def test_episode_metrics_perfect_match():
    """When forecast == actual, every metric is 0 / 1 (peak errors zero)."""
    a = _make_episode_series([(50, 4, 1.5), (150, 3, 1.2)])
    m = compute_episode_metrics(a, a, threshold=1.0)
    assert m["n_obs_episodes"] == 2
    assert m["n_pred_episodes"] == 2
    assert m["n_matched"] == 2
    assert m["episode_precision"] == pytest.approx(1.0)
    assert m["episode_recall"] == pytest.approx(1.0)
    assert m["episode_f1"] == pytest.approx(1.0)
    assert m["peak_height_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["peak_time_error_s"] == pytest.approx(0.0, abs=1e-9)
    assert m["lead_time_error_s"] == pytest.approx(0.0, abs=1e-9)


def test_episode_metrics_false_positive_drops_precision():
    """Spurious forecast peak with no observed counterpart must lower precision."""
    a = _make_episode_series([(50, 4, 1.5)])
    f = _make_episode_series([(50, 4, 1.5), (150, 3, 1.2)])
    m = compute_episode_metrics(a, f, threshold=1.0)
    assert m["n_obs_episodes"] == 1
    assert m["n_pred_episodes"] == 2
    assert m["n_matched"] == 1
    assert m["episode_precision"] == pytest.approx(0.5)
    assert m["episode_recall"] == pytest.approx(1.0)


def test_episode_metrics_missed_event_drops_recall():
    """Observed event with no predicted counterpart lowers recall."""
    a = _make_episode_series([(50, 4, 1.5), (150, 3, 1.2)])
    f = _make_episode_series([(50, 4, 1.5)])
    m = compute_episode_metrics(a, f, threshold=1.0)
    assert m["n_obs_episodes"] == 2
    assert m["n_pred_episodes"] == 1
    assert m["n_matched"] == 1
    assert m["episode_precision"] == pytest.approx(1.0)
    assert m["episode_recall"] == pytest.approx(0.5)


def test_episode_metrics_peak_height_error():
    """Peak-height error reports the mean absolute peak gap on matched pairs."""
    a = _make_episode_series([(50, 4, 1.5)])
    f = _make_episode_series([(50, 4, 1.2)])
    m = compute_episode_metrics(a, f, threshold=1.0)
    # peaks are at the same step; difference ~ 0.3 (modulo Gaussian sample)
    assert m["peak_height_error_m"] > 0.25
    assert m["peak_height_error_m"] < 0.4


def test_episode_metrics_lead_time_late_prediction():
    """A predicted peak arriving later than the observed peak has positive lead error.

    Uses wide Gaussians (half_width=8) so the two episodes overlap and the
    matcher pairs them up.
    """
    a = _make_episode_series([(50, 8, 1.5)])
    f = _make_episode_series([(56, 8, 1.5)])
    m = compute_episode_metrics(a, f, threshold=1.0, step_seconds=360.0)
    assert m["n_matched"] == 1
    # Predicted peak ~6 steps late → +6*360 ≈ 2160 s
    assert m["lead_time_error_s"] > 0
    # Peak-time error is symmetric (abs value)
    assert m["peak_time_error_s"] >= 4 * 360


def test_episode_metrics_lead_time_early_prediction():
    """A predicted peak arriving earlier than the observed peak has negative lead error."""
    a = _make_episode_series([(80, 8, 1.5)])
    f = _make_episode_series([(70, 8, 1.5)])
    m = compute_episode_metrics(a, f, threshold=1.0, step_seconds=360.0)
    assert m["n_matched"] == 1
    assert m["lead_time_error_s"] < 0


def test_episode_metrics_no_events_yields_nan_rates():
    """When neither actual nor forecast crosses threshold, P/R are NaN."""
    a = np.zeros(100)
    f = np.zeros(100)
    m = compute_episode_metrics(a, f, threshold=1.0)
    assert m["n_obs_episodes"] == 0
    assert m["n_pred_episodes"] == 0
    assert math.isnan(m["episode_precision"])
    assert math.isnan(m["episode_recall"])


def test_episode_metrics_required_keys():
    """All documented keys must be present in the result."""
    a = _make_episode_series([(50, 4, 1.5)])
    m = compute_episode_metrics(a, a, threshold=1.0)
    for key in (
        "n_obs_episodes", "n_pred_episodes", "n_matched",
        "episode_precision", "episode_recall", "episode_f1",
        "peak_height_error_m", "peak_time_error_s", "lead_time_error_s",
        "threshold_m",
    ):
        assert key in m, f"missing key: {key}"


def test_episode_metrics_uses_timestamps_when_provided():
    """Timestamp-based errors should be in seconds derived from datetimes."""
    import pandas as pd
    a = _make_episode_series([(50, 8, 1.5)])
    f = _make_episode_series([(56, 8, 1.5)])
    ts = pd.date_range("2024-01-01", periods=len(a), freq="6min", tz="UTC")
    m = compute_episode_metrics(a, f, threshold=1.0, timestamps=ts)
    assert m["n_matched"] == 1
    # 6 steps * 360 s = 2160 s — allow some boundary jitter
    assert abs(m["lead_time_error_s"] - 2160) < 720
