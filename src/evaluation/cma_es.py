"""Small NumPy-only separable CMA-ES optimizer for trajectory-level tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CMAESConfig:
    """Strict-budget configuration for optional coordinator optimization."""

    population_size: int = 8
    parents: int = 4
    sigma: float = 0.1
    max_evaluations: int = 64
    random_seed: int = 42
    checkpoint_path: str | None = None


@dataclass
class CMAESResult:
    """Optimization trace and best parameters."""

    best_params: list[float]
    best_reward: float
    evaluations: int
    training_curve: list[dict[str, float]] = field(default_factory=list)


class SeparableCMAESOptimizer:
    """Diagonal-covariance CMA-ES suitable for small coordination heads."""

    def __init__(self, config: CMAESConfig | None = None) -> None:
        self.config = config or CMAESConfig()

    def optimize(
        self,
        initial_params: np.ndarray,
        objective: Callable[[np.ndarray], float],
    ) -> CMAESResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        mean = np.array(initial_params, dtype=float)
        sigma = np.full_like(mean, cfg.sigma, dtype=float)
        weights = np.log(cfg.parents + 0.5) - np.log(np.arange(1, cfg.parents + 1))
        weights = weights / weights.sum()
        best_params = mean.copy()
        best_reward = float(objective(best_params))
        evaluations = 1
        curve = [{"evaluation": evaluations, "best_reward": best_reward}]

        while evaluations < cfg.max_evaluations:
            n = min(cfg.population_size, cfg.max_evaluations - evaluations)
            noise = rng.normal(size=(n, len(mean)))
            population = mean + noise * sigma
            rewards = np.array([float(objective(params)) for params in population], dtype=float)
            evaluations += n
            order = np.argsort(-rewards)
            selected = population[order[: cfg.parents]]
            selected_noise = noise[order[: cfg.parents]]
            mean = np.average(selected, axis=0, weights=weights[: len(selected)])
            sigma = np.sqrt(
                np.maximum(
                    1e-8,
                    np.average((selected_noise * sigma[:]) ** 2, axis=0, weights=weights[: len(selected)]),
                )
            )
            if rewards[order[0]] > best_reward:
                best_reward = float(rewards[order[0]])
                best_params = population[order[0]].copy()
            curve.append({"evaluation": evaluations, "best_reward": best_reward})
            self._checkpoint(best_params, best_reward, evaluations, curve)

        return CMAESResult(
            best_params=best_params.tolist(),
            best_reward=best_reward,
            evaluations=evaluations,
            training_curve=curve,
        )

    def _checkpoint(
        self,
        best_params: np.ndarray,
        best_reward: float,
        evaluations: int,
        curve: list[dict[str, float]],
    ) -> None:
        if not self.config.checkpoint_path:
            return
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "best_params": best_params.tolist(),
            "best_reward": best_reward,
            "evaluations": evaluations,
            "training_curve": curve,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
