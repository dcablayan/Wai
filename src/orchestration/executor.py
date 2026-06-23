"""Bounded expert execution with optional parallelism.

Cheap arithmetic experts run faster sequentially (thread setup costs more than
the work), so the default path is sequential.  Parallel escalation is available
for independent experts that perform I/O or meaningful compute, behind an
explicit flag, and provides:

- a small bounded thread pool (``max_parallelism``)
- per-expert timeout handling
- deterministic output ordering (input order, not completion order)
- exception isolation (one expert failing never breaks the batch)
- no shared mutable model state (each expert is stateless given its context)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable

from src.experts.base import ExpertForecast, ForecastExpert


@dataclass
class ExpertRun:
    """Outcome of one expert execution (success, failure, or timeout)."""

    name: str
    forecast: ExpertForecast | None
    latency_ms: float
    timed_out: bool = False
    error: str | None = None


def run_expert(expert: ForecastExpert, context) -> ExpertRun:
    """Run a single expert, capturing latency and isolating exceptions."""

    t0 = time.perf_counter()
    try:
        forecast = expert.forecast(context)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        forecast.latency_ms = latency_ms
        return ExpertRun(name=expert.model_name, forecast=forecast, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001 - isolate any expert failure
        latency_ms = (time.perf_counter() - t0) * 1000.0
        failed = expert.failed(context, str(exc))
        failed.latency_ms = latency_ms
        return ExpertRun(
            name=expert.model_name,
            forecast=failed,
            latency_ms=latency_ms,
            error=str(exc),
        )


def run_experts(
    experts: list[ForecastExpert],
    context,
    *,
    parallel: bool = False,
    max_parallelism: int = 2,
    per_expert_timeout_ms: float | None = None,
) -> list[ExpertRun]:
    """Run experts sequentially (default) or in a bounded parallel pool.

    Output order always matches input order, regardless of completion order.
    """

    if not experts:
        return []
    if not parallel or len(experts) == 1:
        return [run_expert(expert, context) for expert in experts]

    timeout_s = (per_expert_timeout_ms / 1000.0) if per_expert_timeout_ms else None
    workers = max(1, min(max_parallelism, len(experts)))
    results: list[ExpertRun | None] = [None] * len(experts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_expert, expert, context): i for i, expert in enumerate(experts)}
        for future, i in futures.items():
            try:
                results[i] = future.result(timeout=timeout_s)
            except FutureTimeout:
                future.cancel()
                results[i] = ExpertRun(
                    name=experts[i].model_name,
                    forecast=None,
                    latency_ms=per_expert_timeout_ms or 0.0,
                    timed_out=True,
                    error="expert timed out",
                )
    return [r for r in results if r is not None]
