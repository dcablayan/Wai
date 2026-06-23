"""Terminal reward functions for Wai Ultra trajectory search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TerminalLossConfig:
    """Configurable terminal loss for workflow-level optimization."""

    peak_event_weight: float = 2.0
    undercoverage_penalty: float = 0.5
    overwide_interval_penalty: float = 0.05
    unavailable_penalty: float = 2.0
    failure_penalty: float = 1.0
    call_cost_weight: float = 0.02
    latency_weight: float = 0.0001
    peak_threshold_m: float = 0.75


def terminal_loss(
    *,
    forecast_m: float | None,
    lower_m: float | None,
    upper_m: float | None,
    actual_m: float | None,
    total_calls: int,
    total_latency_ms: float,
    failed: bool = False,
    config: TerminalLossConfig | None = None,
) -> float:
    """Compute trajectory terminal loss without exposing actuals to policy state."""

    cfg = config or TerminalLossConfig()
    if actual_m is None or forecast_m is None or lower_m is None or upper_m is None:
        return cfg.unavailable_penalty + cfg.call_cost_weight * total_calls
    abs_error = abs(float(forecast_m) - float(actual_m))
    peak_error = abs_error if abs(float(actual_m)) >= cfg.peak_threshold_m else 0.0
    covered = float(lower_m) <= float(actual_m) <= float(upper_m)
    width = max(0.0, float(upper_m) - float(lower_m))
    undercoverage = 0.0 if covered else cfg.undercoverage_penalty
    overwide = cfg.overwide_interval_penalty * max(0.0, width - 1.0)
    failure = cfg.failure_penalty if failed else 0.0
    return float(
        abs_error
        + cfg.peak_event_weight * peak_error
        + undercoverage
        + overwide
        + failure
        + cfg.call_cost_weight * total_calls
        + cfg.latency_weight * total_latency_ms
    )


def reward_from_loss(loss: float) -> float:
    return -float(loss)


def interval_coverage(candidate: dict[str, Any], actual_m: float | None) -> bool | None:
    if actual_m is None:
        return None
    if candidate.get("lower_m") is None or candidate.get("upper_m") is None:
        return False
    return float(candidate["lower_m"]) <= float(actual_m) <= float(candidate["upper_m"])
