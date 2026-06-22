"""Forecast combination methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.experts.base import ExpertForecast


@dataclass
class CombinedForecast:
    """A combined forecast before final verification."""

    forecast_m: float
    lower_m: float
    upper_m: float
    confidence: float
    method: str
    experts_used: list[str]
    diagnostics: dict = field(default_factory=dict)


class ForecastCombiner:
    """Combine successful expert forecasts."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def combine(
        self,
        forecasts: Iterable[ExpertForecast],
        *,
        method: str = "weighted_median",
    ) -> CombinedForecast:
        valid = [f for f in forecasts if f.ok]
        if not valid:
            raise ValueError("No successful expert forecasts are available to combine")

        values = np.array([float(f.predicted_water_level_m) for f in valid], dtype=float)
        lowers = np.array([float(f.lower_m) for f in valid], dtype=float)
        uppers = np.array([float(f.upper_m) for f in valid], dtype=float)
        confidences = np.array([float(f.confidence) for f in valid], dtype=float)
        weights = np.array([
            max(0.0, self.weights.get(f.model_name, f.confidence)) for f in valid
        ], dtype=float)
        if np.isclose(weights.sum(), 0.0):
            weights = np.ones(len(valid), dtype=float)

        if method == "best_expert":
            idx = int(np.argmax(confidences))
            forecast = values[idx]
            lower = lowers[idx]
            upper = uppers[idx]
            confidence = confidences[idx]
        elif method == "weighted_average":
            forecast = float(np.average(values, weights=weights))
            lower = float(np.average(lowers, weights=weights))
            upper = float(np.average(uppers, weights=weights))
            confidence = float(np.average(confidences, weights=weights))
        elif method == "simple_median":
            forecast = float(np.median(values))
            lower = float(np.median(lowers))
            upper = float(np.median(uppers))
            confidence = float(np.median(confidences))
        elif method == "weighted_median":
            forecast = _weighted_median(values, weights)
            lower = _weighted_median(lowers, weights)
            upper = _weighted_median(uppers, weights)
            confidence = float(np.average(confidences, weights=weights))
        else:
            raise ValueError(f"Unknown combination method: {method}")

        if lower > upper:
            lower, upper = upper, lower
        lower = min(lower, forecast)
        upper = max(upper, forecast)

        return CombinedForecast(
            forecast_m=float(forecast),
            lower_m=float(lower),
            upper_m=float(upper),
            confidence=max(0.0, min(1.0, confidence)),
            method=method,
            experts_used=[f.model_name for f in valid],
            diagnostics={
                "n_experts": len(valid),
                "expert_values_m": dict(zip([f.model_name for f in valid], values.tolist())),
                "weights": dict(zip([f.model_name for f in valid], weights.tolist())),
            },
        )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = sorted_weights.sum() / 2.0
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])
