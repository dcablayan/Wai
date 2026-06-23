"""Machine-readable capability metadata for forecasting experts.

Every expert declares an :class:`ExpertSpec` so the orchestrator can exclude
impossible experts *before* routing instead of discovering missing dependencies
by executing each expert and waiting for it to fail.  The spec also feeds the
cost-aware router score (latency class, compute cost) and the parallel executor
(thread safety).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Relative latency classes (ordering matters for cost-aware routing).
LATENCY_INSTANT = "instant"   # pure arithmetic on already-loaded context
LATENCY_FAST = "fast"         # small vectorized lookups
LATENCY_MODERATE = "moderate" # local model inference
LATENCY_SLOW = "slow"         # network / heavy compute

_LATENCY_ORDER = {
    LATENCY_INSTANT: 0.0,
    LATENCY_FAST: 1.0,
    LATENCY_MODERATE: 5.0,
    LATENCY_SLOW: 25.0,
}


@dataclass(frozen=True)
class ExpertSpec:
    """Declarative capability description for one expert.

    Attributes mirror the orchestration questions: what data must exist, what
    horizons the expert supports, whether it is a safe baseline, and how
    expensive / parallel-safe it is.
    """

    model_name: str
    required_sources: tuple[str, ...] = ()
    requires_local_obs: bool = False
    requires_noaa_obs: bool = False
    requires_tide: bool = False
    is_safe_baseline: bool = False
    min_horizon_minutes: int = 0
    max_horizon_minutes: int = 14 * 24 * 60
    latency_class: str = LATENCY_INSTANT
    compute_cost: float = 1.0
    thread_safe: bool = True
    cacheable: bool = True
    notes: str = ""

    @property
    def expected_latency_units(self) -> float:
        """Coarse expected latency for cost-aware ranking (lower is cheaper)."""

        return _LATENCY_ORDER.get(self.latency_class, 5.0)

    def supports_horizon(self, horizon_minutes: int) -> bool:
        return self.min_horizon_minutes <= horizon_minutes <= self.max_horizon_minutes

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "required_sources": list(self.required_sources),
            "requires_local_obs": self.requires_local_obs,
            "requires_noaa_obs": self.requires_noaa_obs,
            "requires_tide": self.requires_tide,
            "is_safe_baseline": self.is_safe_baseline,
            "min_horizon_minutes": self.min_horizon_minutes,
            "max_horizon_minutes": self.max_horizon_minutes,
            "latency_class": self.latency_class,
            "compute_cost": self.compute_cost,
            "thread_safe": self.thread_safe,
            "cacheable": self.cacheable,
            "notes": self.notes,
        }
