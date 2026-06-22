"""Configuration policies for routing and verification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingPolicy:
    """Thresholds used by the first rule-based router."""

    short_horizon_minutes: int = 90
    normal_horizon_minutes: int = 24 * 60
    fresh_hohonu_seconds: float = 60 * 60
    fresh_noaa_seconds: float = 3 * 60 * 60
    large_noaa_residual_m: float = 0.25
    strong_disagreement_m: float = 0.35
    max_selected_experts: int = 3
    default_combination_method: str = "weighted_median"
    performance_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationPolicy:
    """Thresholds used by the forecast verifier."""

    max_input_staleness_seconds: float = 6 * 60 * 60
    suspicious_jump_m: float = 1.5
    plausible_min_m: float = -5.0
    plausible_max_m: float = 5.0
    high_disagreement_m: float = 0.4
    disagreement_interval_multiplier: float = 1.5
