"""Evaluation helpers for Wai Mini, Ultra, and oracle workflows."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.reward import terminal_loss
from src.forecasting.pipeline import ForecastPipeline


@dataclass
class ModeEvaluationRow:
    """One evaluated forecast result."""

    mode: str
    forecast_origin: str
    target_time: str
    actual_m: float | None
    forecast_m: float | None
    lower_m: float | None
    upper_m: float | None
    abs_error_m: float | None
    covered: bool | None
    interval_width_m: float | None
    status: str
    fallback_used: bool
    latency_ms: float
    logical_actions: int
    physical_expert_calls: int
    unique_experts: int
    turns: int
    termination_reason: str
    reward: float | None


def evaluate_modes_on_contexts(
    contexts: Iterable[Any],
    actuals_m: Iterable[float | None],
    *,
    modes: tuple[str, ...] = ("legacy", "mini", "ultra"),
) -> pd.DataFrame:
    """Run comparable modes on prebuilt contexts and actuals."""

    rows = []
    for context, actual_m in zip(contexts, actuals_m):
        for mode in modes:
            pipe = ForecastPipeline(mode=mode)
            started = time.perf_counter()
            result = pipe.run(context)
            latency_ms = (time.perf_counter() - started) * 1000.0
            abs_error = None if result.forecast_m is None or actual_m is None else abs(result.forecast_m - actual_m)
            covered = None
            width = None
            if result.lower_m is not None and result.upper_m is not None:
                width = result.upper_m - result.lower_m
                covered = None if actual_m is None else result.lower_m <= actual_m <= result.upper_m
            loss = None
            if actual_m is not None:
                loss = terminal_loss(
                    forecast_m=result.forecast_m,
                    lower_m=result.lower_m,
                    upper_m=result.upper_m,
                    actual_m=actual_m,
                    total_calls=result.logical_actions,
                    total_latency_ms=latency_ms,
                    failed=result.status != "available",
                )
            rows.append(
                ModeEvaluationRow(
                    mode=mode,
                    forecast_origin=str(context.forecast_time_utc),
                    target_time=str(context.target_time_utc),
                    actual_m=actual_m,
                    forecast_m=result.forecast_m,
                    lower_m=result.lower_m,
                    upper_m=result.upper_m,
                    abs_error_m=abs_error,
                    covered=covered,
                    interval_width_m=width,
                    status=result.status,
                    fallback_used=result.fallback_used,
                    latency_ms=latency_ms,
                    logical_actions=result.logical_actions,
                    physical_expert_calls=result.physical_expert_calls,
                    unique_experts=result.unique_experts,
                    turns=result.number_of_turns,
                    termination_reason=result.termination_reason,
                    reward=None if loss is None else -loss,
                )
            )
    return pd.DataFrame([asdict(row) for row in rows])


def summarize_mode_evaluation(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quality, coverage, latency, and compute by mode."""

    summaries = []
    for mode, group in rows.groupby("mode"):
        summaries.append({
            "mode": mode,
            "overall_mae": _mean(group["abs_error_m"]),
            "interval_coverage": _mean(group["covered"].dropna().astype(float)) if group["covered"].notna().any() else np.nan,
            "interval_width": _mean(group["interval_width_m"]),
            "unavailable_rate": float((group["status"] != "available").mean()),
            "fallback_rate": float(group["fallback_used"].mean()),
            "p50_latency_ms": float(group["latency_ms"].quantile(0.50)),
            "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
            "logical_actions": _mean(group["logical_actions"]),
            "physical_expert_calls": _mean(group["physical_expert_calls"]),
            "unique_experts": _mean(group["unique_experts"]),
            "average_turns": _mean(group["turns"]),
            "reward": _mean(group["reward"]),
        })
    return pd.DataFrame(summaries)


def ablation_rows(base_rows: pd.DataFrame) -> pd.DataFrame:
    """Return placeholders for standard Ultra ablations with explicit status."""

    ablations = [
        "no_thinker_role",
        "no_verifier_role",
        "no_transcript",
        "no_access_control",
        "no_recursive_replanning",
        "fixed_topology",
        "no_randomized_pool_training",
        "mini_cascade",
    ]
    return pd.DataFrame([
        {
            "ablation": name,
            "status": "not_run_in_lightweight_helper",
            "note": "Use the full historical replay benchmark to produce numeric ablation metrics.",
        }
        for name in ablations
    ])


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")
