"""Forecast evaluation metrics for Wai."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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
