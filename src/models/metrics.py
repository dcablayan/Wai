"""Forecast evaluation metrics for Wai."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R², NSE, and Pearson correlation.

    NaN values in either array are excluded before computation.
    Returns a dict with keys: mae, rmse, r2, nse, corr.
    """
    a = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]

    if len(a) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "corr": float("nan")}

    diff = f - a
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    nse = r2  # Nash-Sutcliffe efficiency equals R² for point forecasts

    corr = (
        float(np.corrcoef(f, a)[0, 1])
        if np.std(f) > 0 and np.std(a) > 0
        else float("nan")
    )

    return {"mae": mae, "rmse": rmse, "r2": r2, "nse": nse, "corr": corr}


def compute_event_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute event-level threshold exceedance metrics.

    Treats each time step as a binary event (actual >= threshold) and computes
    precision, recall, and F1 over the forecast's threshold-crossing decisions.
    Also reports peak error and the fraction of samples where forecast and
    actual agree on side-of-threshold.

    Parameters
    ----------
    actual : array of observed water levels
    forecast : array of forecast water levels (same length)
    threshold : scalar threshold for alert classification

    Returns
    -------
    dict with keys:
        precision       — TP / (TP + FP); fraction of predicted events that are real
        recall          — TP / (TP + FN); fraction of real events that are predicted
        f1              — harmonic mean of precision and recall
        peak_error_m    — max(abs(actual - forecast)) across all event steps (actual>=threshold)
        threshold_agree — fraction of steps where both signals are on the same side
    """
    a = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]

    if len(a) == 0:
        nan = float("nan")
        return {"precision": nan, "recall": nan, "f1": nan,
                "peak_error_m": nan, "threshold_agree": nan}

    a_event = a >= threshold
    f_event = f >= threshold

    tp = float(np.sum(a_event & f_event))
    fp = float(np.sum(~a_event & f_event))
    fn = float(np.sum(a_event & ~f_event))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision is not float("nan") and recall is not float("nan") and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    event_mask = a_event
    peak_error = float(np.max(np.abs(a[event_mask] - f[event_mask]))) if event_mask.any() else float("nan")

    threshold_agree = float(np.mean(a_event == f_event))

    return {
        "precision": round(precision, 4) if not _isnan(precision) else float("nan"),
        "recall": round(recall, 4) if not _isnan(recall) else float("nan"),
        "f1": round(f1, 4) if not _isnan(f1) else float("nan"),
        "peak_error_m": round(peak_error, 4) if not _isnan(peak_error) else float("nan"),
        "threshold_agree": round(threshold_agree, 4),
    }


def _isnan(x: float) -> bool:
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


def bootstrap_ci(
    actual: np.ndarray,
    forecast: np.ndarray,
    metric: str = "mae",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a scalar metric.

    Resamples (actual, forecast) pairs with replacement *n_boot* times and
    reports the *ci*-level interval as (lower, upper).

    Parameters
    ----------
    actual, forecast : arrays of equal length
    metric : 'mae', 'rmse', 'r2', or 'corr'
    n_boot : number of bootstrap replications
    ci : nominal coverage (e.g. 0.95 → 95% CI)
    seed : RNG seed for reproducibility

    Returns
    -------
    (lower, upper) tuple
    """
    a = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]

    if len(a) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    stats: List[float] = []
    n = len(a)
    alpha = 1 - ci
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(a[idx], f[idx])
        val = m.get(metric, float("nan"))
        if not _isnan(val):
            stats.append(val)

    if not stats:
        return (float("nan"), float("nan"))

    stats_arr = np.array(stats)
    lo = float(np.percentile(stats_arr, 100 * alpha / 2))
    hi = float(np.percentile(stats_arr, 100 * (1 - alpha / 2)))
    return (round(lo, 6), round(hi, 6))


def _sanitize(obj):
    """Recursively replace float NaN/Inf with None for valid JSON output."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_metrics(metrics: dict, path: Path) -> None:
    """Persist a metrics dict to JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize(metrics), f, indent=2)
