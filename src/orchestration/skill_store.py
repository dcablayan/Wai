"""Lightweight rolling skill store for experts.

Tracks recent measured skill (MAE, RMSE, interval coverage, failure rate,
latency) per (expert, station, horizon bucket, regime) using exponentially
weighted updates, with a hierarchy that falls back to coarser keys when a
specific cell has too few samples:

    station + horizon + regime
        -> station + horizon
            -> horizon
                -> global expert prior

A tiny sample is never treated as strong evidence: a specific cell's metric is
shrunk toward the coarser-level estimate by sample count, so one observation
barely moves the effective skill.  The store is in-memory and can be persisted
to / loaded from JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Cold-start priors per expert (rough expected MAE in metres).  These are only
# used until enough measured samples accumulate; measured skill then dominates.
_DEFAULT_PRIOR_MAE = {
    "local_persistence": 0.10,
    "local_tide": 0.16,
    "noaa_residual": 0.12,
    "regional_to_local_residual": 0.18,
    "safe_fallback": 0.22,
}
_GLOBAL_PRIOR_MAE = 0.20


def horizon_bucket(horizon_minutes: int) -> str:
    if horizon_minutes <= 90:
        return "short"
    if horizon_minutes <= 360:
        return "medium"
    if horizon_minutes <= 1440:
        return "day"
    return "long"


@dataclass
class SkillRecord:
    """Exponentially weighted skill metrics for one key."""

    n: int = 0
    mae: float = 0.0
    mse: float = 0.0
    coverage: float = 0.0
    failure_rate: float = 0.0
    latency_ms: float = 0.0
    last_update: str | None = None

    @property
    def rmse(self) -> float | None:
        if self.n < 2 or self.mse <= 0:
            return None
        return math.sqrt(self.mse)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rmse"] = self.rmse
        return d


@dataclass
class SkillEstimate:
    """Resolved skill estimate after hierarchical fallback + shrinkage."""

    expert: str
    mae: float
    coverage: float | None
    failure_rate: float
    latency_ms: float
    sample_count: int
    source_level: str  # which hierarchy level supplied most of the evidence


class SkillStore:
    """In-memory rolling skill metrics with optional JSON persistence."""

    def __init__(
        self,
        *,
        decay: float = 0.85,
        min_samples: int = 5,
        prior_strength: float = 4.0,
        prior_mae: dict[str, float] | None = None,
    ) -> None:
        self.decay = float(decay)
        self.min_samples = int(min_samples)
        self.prior_strength = float(prior_strength)
        self.prior_mae = dict(prior_mae or _DEFAULT_PRIOR_MAE)
        self._records: dict[tuple[str, str, str, str], SkillRecord] = {}

    # -- updates ------------------------------------------------------------

    def update(
        self,
        *,
        expert: str,
        station: str,
        horizon_minutes: int,
        regime: str,
        abs_error: float | None,
        covered: bool | None = None,
        failed: bool = False,
        latency_ms: float = 0.0,
        timestamp: str | None = None,
    ) -> None:
        """Record one outcome at every hierarchy level it informs."""

        bucket = horizon_bucket(horizon_minutes)
        keys = [
            (expert, station, bucket, regime),
            (expert, station, bucket, "*"),
            (expert, "*", bucket, "*"),
            (expert, "*", "*", "*"),
        ]
        for key in keys:
            self._update_key(key, abs_error, covered, failed, latency_ms, timestamp)

    def _update_key(self, key, abs_error, covered, failed, latency_ms, timestamp) -> None:
        rec = self._records.get(key)
        if rec is None:
            rec = SkillRecord()
            self._records[key] = rec
        a = 1.0 - self.decay if rec.n > 0 else 1.0
        rec.failure_rate = (1 - a) * rec.failure_rate + a * (1.0 if failed else 0.0)
        if not failed and abs_error is not None and math.isfinite(abs_error):
            err = abs(float(abs_error))
            rec.mae = (1 - a) * rec.mae + a * err if rec.n > 0 else err
            rec.mse = (1 - a) * rec.mse + a * (err * err) if rec.n > 0 else err * err
            if covered is not None:
                rec.coverage = (1 - a) * rec.coverage + a * (1.0 if covered else 0.0)
        if latency_ms:
            rec.latency_ms = (1 - a) * rec.latency_ms + a * float(latency_ms) if rec.n > 0 else float(latency_ms)
        rec.n += 1
        rec.last_update = timestamp

    # -- lookups ------------------------------------------------------------

    def estimate(
        self, *, expert: str, station: str, horizon_minutes: int, regime: str
    ) -> SkillEstimate:
        """Resolve skill via hierarchy + shrinkage toward coarser levels."""

        bucket = horizon_bucket(horizon_minutes)
        levels = [
            ("station_horizon_regime", (expert, station, bucket, regime)),
            ("station_horizon", (expert, station, bucket, "*")),
            ("horizon", (expert, "*", bucket, "*")),
            ("global", (expert, "*", "*", "*")),
        ]
        prior = self.prior_mae.get(expert, _GLOBAL_PRIOR_MAE)

        # Choose the finest level that meets min_samples; otherwise fall back to
        # the finest level that has any data.  The chosen cell's metric is then
        # shrunk toward the prior by *its own* sample count, so a tiny sample is
        # never strong evidence and shrinkage does not compound across levels.
        chosen_name = "prior"
        chosen = None
        finest_with_data = None
        for name, key in levels:  # finest -> coarsest
            rec = self._records.get(key)
            if rec is None or rec.n == 0:
                continue
            if finest_with_data is None:
                finest_with_data = (name, rec)
            if rec.n >= self.min_samples:
                chosen_name, chosen = name, rec
                break
        if chosen is None and finest_with_data is not None:
            chosen_name, chosen = "prior", finest_with_data[1]

        if chosen is None:
            return SkillEstimate(
                expert=expert, mae=float(prior), coverage=None, failure_rate=0.0,
                latency_ms=0.0, sample_count=0, source_level="prior",
            )

        w = chosen.n / (chosen.n + self.prior_strength)
        eff_mae = w * chosen.mae + (1 - w) * prior
        eff_fail = w * chosen.failure_rate  # failures shrink toward 0
        return SkillEstimate(
            expert=expert,
            mae=float(eff_mae),
            coverage=chosen.coverage if chosen.coverage else None,
            failure_rate=float(eff_fail),
            latency_ms=float(chosen.latency_ms),
            sample_count=int(chosen.n),
            source_level=chosen_name,
        )

    def weight(self, *, expert: str, station: str, horizon_minutes: int, regime: str) -> float:
        """Inverse-error skill weight for combination (higher is better)."""

        est = self.estimate(
            expert=expert, station=station, horizon_minutes=horizon_minutes, regime=regime
        )
        return 1.0 / max(est.mae, 1e-3)

    def has_support(self, *, station: str, horizon_minutes: int, regime: str, expert: str) -> bool:
        est = self.estimate(
            expert=expert, station=station, horizon_minutes=horizon_minutes, regime=regime
        )
        return est.sample_count >= self.min_samples

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "decay": self.decay,
            "min_samples": self.min_samples,
            "prior_strength": self.prior_strength,
            "prior_mae": self.prior_mae,
            "records": {
                "|".join(k): asdict(v) for k, v in self._records.items()
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "SkillStore":
        payload = json.loads(Path(path).read_text())
        store = cls(
            decay=payload.get("decay", 0.85),
            min_samples=payload.get("min_samples", 5),
            prior_strength=payload.get("prior_strength", 4.0),
            prior_mae=payload.get("prior_mae"),
        )
        for key, rec in payload.get("records", {}).items():
            store._records[tuple(key.split("|"))] = SkillRecord(**{
                k: v for k, v in rec.items() if k in SkillRecord.__dataclass_fields__
            })
        return store
