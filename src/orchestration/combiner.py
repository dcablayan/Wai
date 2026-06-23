"""Forecast combination methods.

Combination now depends on the number and *measured* skill of successful
experts rather than self-reported confidence alone:

- one expert: return it directly (after verification upstream);
- two experts: skill-weighted average / weighted median;
- three+ experts: weighted median (robust).

Weights default to inverse-validation-error from the :class:`SkillStore` when
supplied; otherwise they fall back to per-forecast confidence (legacy
behaviour).  ``safe_fallback`` is not averaged into an otherwise-valid ensemble
— it is a fallback/comparison baseline, not a peer model.  Final uncertainty
combines the experts' own intervals with between-expert disagreement and a
horizon term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.experts.base import ExpertForecast

SAFE_BASELINE_NAME = "safe_fallback"


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
        weights: dict[str, float] | None = None,
        horizon_minutes: int | None = None,
        drop_safe_baseline: bool = True,
        min_half_width_m: float | None = None,
    ) -> CombinedForecast:
        valid = [f for f in forecasts if f.ok]
        if not valid:
            raise ValueError("No successful expert forecasts are available to combine")

        # Do not average the safe baseline into an otherwise-valid ensemble.
        if drop_safe_baseline and len(valid) > 1:
            non_baseline = [f for f in valid if f.model_name != SAFE_BASELINE_NAME]
            if non_baseline:
                valid = non_baseline

        weight_source = weights if weights is not None else self.weights
        values = np.array([float(f.predicted_water_level_m) for f in valid], dtype=float)
        lowers = np.array([float(f.lower_m) for f in valid], dtype=float)
        uppers = np.array([float(f.upper_m) for f in valid], dtype=float)
        confidences = np.array([float(f.confidence) for f in valid], dtype=float)
        weight_arr = np.array([
            max(0.0, weight_source.get(f.model_name, f.confidence)) for f in valid
        ], dtype=float)
        if np.isclose(weight_arr.sum(), 0.0):
            weight_arr = np.ones(len(valid), dtype=float)

        # A single expert collapses every method to itself.
        if len(valid) == 1:
            method_used = "single_expert"
            forecast = values[0]
            lower = lowers[0]
            upper = uppers[0]
            confidence = confidences[0]
        else:
            method_used = method
            forecast, lower, upper, confidence = _apply_method(
                method, values, lowers, uppers, confidences, weight_arr
            )

        # Clamp the method's interval to contain the point forecast, then widen
        # for between-expert disagreement and horizon.
        if lower > upper:
            lower, upper = upper, lower
        base_half = max(forecast - lower, upper - forecast, 0.0)
        spread = float(values.max() - values.min()) if len(values) >= 2 else 0.0
        horizon_hours = (horizon_minutes / 60.0) if horizon_minutes else 0.0
        extra_half = 0.5 * spread + 0.002 * horizon_hours
        half = base_half + extra_half
        # Floor the interval at measured historical residual uncertainty so an
        # early-stopped single-expert forecast does not under-cover for lack of
        # between-expert disagreement.
        if min_half_width_m is not None:
            half = max(half, float(min_half_width_m))
        lower = float(forecast - half)
        upper = float(forecast + half)

        return CombinedForecast(
            forecast_m=float(forecast),
            lower_m=float(lower),
            upper_m=float(upper),
            confidence=max(0.0, min(1.0, float(confidence))),
            method=method_used,
            experts_used=[f.model_name for f in valid],
            diagnostics={
                "n_experts": len(valid),
                "expert_values_m": dict(zip([f.model_name for f in valid], values.tolist())),
                "weights": dict(zip([f.model_name for f in valid], weight_arr.tolist())),
                "between_expert_spread_m": spread,
            },
        )


def _apply_method(method, values, lowers, uppers, confidences, weights):
    if method == "best_expert":
        idx = int(np.argmax(confidences))
        return values[idx], lowers[idx], uppers[idx], confidences[idx]
    if method == "weighted_average":
        return (
            float(np.average(values, weights=weights)),
            float(np.average(lowers, weights=weights)),
            float(np.average(uppers, weights=weights)),
            float(np.average(confidences, weights=weights)),
        )
    if method == "simple_median":
        return (
            float(np.median(values)),
            float(np.median(lowers)),
            float(np.median(uppers)),
            float(np.median(confidences)),
        )
    if method == "weighted_median":
        return (
            _weighted_median(values, weights),
            _weighted_median(lowers, weights),
            _weighted_median(uppers, weights),
            float(np.average(confidences, weights=weights)),
        )
    raise ValueError(f"Unknown combination method: {method}")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = sorted_weights.sum() / 2.0
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])
