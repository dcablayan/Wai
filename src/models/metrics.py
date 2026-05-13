"""Forecast evaluation metrics for Wai."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous (start, end_exclusive) index runs where mask is True."""
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_episode_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    threshold: float,
    timestamps: Optional[Sequence] = None,
    step_seconds: float = 360.0,
) -> Dict[str, float]:
    """Episode-level event metrics: precision, recall, lead/peak-time/peak-height errors.

    An *episode* is a contiguous run of timesteps where the signal stays at or
    above ``threshold``. Predicted episodes are matched to observed episodes by
    any temporal overlap (interval intersection on the integer step grid). Each
    matched pair contributes to peak-height, peak-time, and lead-time errors;
    unmatched predicted episodes count as false positives; unmatched observed
    episodes count as false negatives.

    Definitions
    -----------
    - **Episode precision** = matched_predicted / total_predicted
    - **Episode recall**    = matched_predicted / total_observed
        (matched_predicted == matched_observed for one-to-one matching)
    - **Peak-height error** = mean(|peak_pred − peak_obs|) across matches (metres)
    - **Peak-time error**   = mean(|t_peak_pred − t_peak_obs|) across matches (s)
    - **Lead-time error**   = mean(t_pred_start − t_obs_start) across matches
        (positive = prediction LATE, negative = prediction EARLY).

    Timestamps may be passed for human-readable seconds; otherwise integer step
    indices are used and the errors are multiplied by ``step_seconds``.

    Parameters
    ----------
    actual, forecast : 1-D arrays of identical length
    threshold : float
        Exceedance threshold (units match the signal — typically meters).
    timestamps : optional, datetime-like sequence aligned to actual/forecast.
    step_seconds : float, default 360.0 (6 min)
        Used when ``timestamps`` is None to convert step-count errors to seconds.

    Returns
    -------
    dict with keys:
        n_obs_episodes, n_pred_episodes, n_matched,
        episode_precision, episode_recall, episode_f1,
        peak_height_error_m, peak_time_error_s, lead_time_error_s,
        threshold_m
    """
    a = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    if len(a) != len(f):
        raise ValueError(f"actual ({len(a)}) and forecast ({len(f)}) must align")
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]
    if timestamps is not None:
        ts_arr = np.asarray(list(timestamps))[mask]
    else:
        ts_arr = None

    obs_runs = _runs(a >= threshold)
    pred_runs = _runs(f >= threshold)

    n_obs = len(obs_runs)
    n_pred = len(pred_runs)

    # One-to-one matching: each predicted episode pairs with the observed
    # episode it overlaps the most (greedy by overlap length). An observed
    # episode may be claimed by at most one predicted episode so precision
    # and recall are interpretable as fractions of distinct events.
    matches: List[Tuple[int, int, int]] = []  # (pred_idx, obs_idx, overlap)
    for p_i, (ps, pe) in enumerate(pred_runs):
        best = None
        for o_i, (os_, oe) in enumerate(obs_runs):
            ov = max(0, min(pe, oe) - max(ps, os_))
            if ov > 0 and (best is None or ov > best[2]):
                best = (p_i, o_i, ov)
        if best is not None:
            matches.append(best)

    # Resolve double-claims of observed episodes: keep the largest overlap.
    seen_obs: Dict[int, Tuple[int, int]] = {}
    for p_i, o_i, ov in matches:
        prev = seen_obs.get(o_i)
        if prev is None or ov > prev[1]:
            seen_obs[o_i] = (p_i, ov)
    final_matches = [(p_i, o_i) for o_i, (p_i, _) in seen_obs.items()]

    n_matched = len(final_matches)
    precision = n_matched / n_pred if n_pred > 0 else float("nan")
    recall = n_matched / n_obs if n_obs > 0 else float("nan")
    if (
        not _isnan(precision)
        and not _isnan(recall)
        and (precision + recall) > 0
    ):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    # Peak / lead errors for matched pairs
    peak_height_errs: List[float] = []
    peak_time_errs: List[float] = []
    lead_time_errs: List[float] = []
    for p_i, o_i in final_matches:
        ps, pe = pred_runs[p_i]
        os_, oe = obs_runs[o_i]
        peak_pred_off = ps + int(np.argmax(f[ps:pe]))
        peak_obs_off = os_ + int(np.argmax(a[os_:oe]))

        peak_height_errs.append(abs(float(f[peak_pred_off] - a[peak_obs_off])))

        if ts_arr is not None:
            t_pred_start = pd_ts_to_seconds(ts_arr[ps])
            t_obs_start = pd_ts_to_seconds(ts_arr[os_])
            t_pred_peak = pd_ts_to_seconds(ts_arr[peak_pred_off])
            t_obs_peak = pd_ts_to_seconds(ts_arr[peak_obs_off])
            peak_time_errs.append(abs(t_pred_peak - t_obs_peak))
            lead_time_errs.append(t_pred_start - t_obs_start)
        else:
            peak_time_errs.append(abs(peak_pred_off - peak_obs_off) * step_seconds)
            lead_time_errs.append((ps - os_) * step_seconds)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_obs_episodes": int(n_obs),
        "n_pred_episodes": int(n_pred),
        "n_matched": int(n_matched),
        "episode_precision": round(precision, 4) if not _isnan(precision) else float("nan"),
        "episode_recall": round(recall, 4) if not _isnan(recall) else float("nan"),
        "episode_f1": round(f1, 4) if not _isnan(f1) else float("nan"),
        "peak_height_error_m": round(_mean(peak_height_errs), 6),
        "peak_time_error_s": round(_mean(peak_time_errs), 3),
        "lead_time_error_s": round(_mean(lead_time_errs), 3),
        "threshold_m": float(threshold),
    }


def pd_ts_to_seconds(ts) -> float:
    """Convert a datetime-like value to POSIX seconds (UTC)."""
    import pandas as _pd
    return float(_pd.Timestamp(ts).timestamp())


def bootstrap_ci(
    actual: np.ndarray,
    forecast: np.ndarray,
    metric: str = "mae",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """IID bootstrap confidence interval for a scalar metric.

    .. warning::
       For time-series residuals this is a **reference baseline only** — it
       assumes independent samples and so underestimates uncertainty when
       residuals are autocorrelated (true for tidal forecast residuals).
       Prefer :func:`block_bootstrap_ci` for the headline interval.

    Resamples (actual, forecast) pairs with replacement *n_boot* times and
    reports the *ci*-level interval as (lower, upper).
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


def _suggest_block_length(n: int) -> int:
    """Default block length ≈ n**(1/3) (Politis & White rule-of-thumb floor).

    Returns at least 2 and at most n // 4 so a typical replicate still spans
    many blocks.
    """
    if n < 8:
        return max(2, n // 2)
    return int(max(2, min(n // 4, np.ceil(n ** (1 / 3)))))


def block_bootstrap_ci(
    actual: np.ndarray,
    forecast: np.ndarray,
    metric: str = "mae",
    n_boot: int = 1000,
    ci: float = 0.95,
    block_length: Optional[int] = None,
    circular: bool = True,
    seed: int = 42,
) -> Dict[str, object]:
    """Moving / circular block bootstrap CI for a scalar metric on a time series.

    Why blocks
    ----------
    Tidal-forecast residuals are autocorrelated: large negative residuals at
    one timestep are likely to be followed by large negative residuals at the
    next. An i.i.d. bootstrap shatters that dependence and produces CIs that
    are too tight. Resampling *contiguous blocks* of residuals preserves the
    short-range correlation structure inside each block, giving a more honest
    interval at the cost of slightly higher variance.

    Modes
    -----
    - ``circular=True`` (default): circular block bootstrap (Politis &
      Romano 1992). Sampling wraps around the end of the series, so every
      block has identical length and the procedure is stationary.
    - ``circular=False``: moving block bootstrap (Künsch 1989). Blocks are
      drawn only from positions where they fit without wrapping; the last
      partial block of a replicate is trimmed to length ``n``.

    Block length
    ------------
    If ``block_length`` is None, defaults to ``ceil(n**(1/3))`` (a coarse
    rule-of-thumb that grows with sample size). The returned dict reports
    the block length used so the CI is auditable.

    Returns
    -------
    dict with keys:
        lower, upper       — interval bounds
        block_length       — block length used
        n_boot             — replicates
        method             — 'circular_block' or 'moving_block'
        n_samples          — series length after NaN removal
    """
    a = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    mask = ~(np.isnan(a) | np.isnan(f))
    a, f = a[mask], f[mask]
    n = len(a)
    if n < 2:
        return {
            "lower": float("nan"), "upper": float("nan"),
            "block_length": None, "n_boot": int(n_boot),
            "method": "circular_block" if circular else "moving_block",
            "n_samples": int(n),
        }

    L = int(block_length) if block_length is not None else _suggest_block_length(n)
    L = max(1, min(L, n))
    n_blocks = int(np.ceil(n / L))

    rng = np.random.default_rng(seed)
    stats: List[float] = []
    alpha = 1 - ci

    if circular:
        max_start = n  # any start; wrap via modulo
    else:
        max_start = max(1, n - L + 1)  # only positions where block fits

    for _ in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        offsets = np.arange(L)
        if circular:
            idx = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        else:
            idx = (starts[:, None] + offsets[None, :]).reshape(-1)[:n]
        m = compute_metrics(a[idx], f[idx])
        val = m.get(metric, float("nan"))
        if not _isnan(val):
            stats.append(val)

    if not stats:
        lo = hi = float("nan")
    else:
        arr = np.array(stats)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))

    return {
        "lower": round(lo, 6) if not _isnan(lo) else float("nan"),
        "upper": round(hi, 6) if not _isnan(hi) else float("nan"),
        "block_length": int(L),
        "n_boot": int(n_boot),
        "method": "circular_block" if circular else "moving_block",
        "n_samples": int(n),
    }


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
