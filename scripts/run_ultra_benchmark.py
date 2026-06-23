"""Run a deterministic local smoke benchmark for Wai Ultra orchestration.

This benchmark uses bundled mock data. It verifies quality/compute reporting and
mode compatibility; it is not a real-world scientific validation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.evaluation.reward import terminal_loss
from src.evaluation.trajectory_search import search_oracle_workflows
from src.evaluation.ultra_evaluation import summarize_mode_evaluation
from src.forecasting import ForecastPipeline, default_experts
from src.orchestration.context import build_forecast_context
from src.orchestration.protocol import ExecutionBudget

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> None:
    contexts, actuals = _demo_contexts()
    rows = []
    for budget in (1, 2, 3, 5):
        for context, actual_m in zip(contexts, actuals):
            for mode in ("legacy", "mini"):
                if budget != 5:
                    continue
                rows.append(_run_one(mode, context, actual_m))
            rows.append(_run_one("ultra", context, actual_m, budget=budget))
    result_frame = pd.DataFrame(rows)
    summary = summarize_mode_evaluation(result_frame)
    oracle = _oracle_rows(contexts, actuals)

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "ultra_benchmark_results.json"
    json_path.write_text(json.dumps({
        "description": "Local deterministic mock-data smoke benchmark; not a real-world validation.",
        "rows": result_frame.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "oracle": oracle,
        "ablations": [
            {"ablation": name, "status": "not_run", "reason": "requires full historical trajectory dataset"}
            for name in (
                "no_thinker_role",
                "no_verifier_role",
                "no_transcript",
                "no_access_control",
                "no_recursive_replanning",
                "fixed_topology",
                "no_randomized_pool_training",
                "mini_cascade",
            )
        ],
    }, indent=2, sort_keys=True, default=str))

    md_path = REPORTS_DIR / "ultra_benchmark_results.md"
    md_path.write_text(_markdown_report(summary, oracle))
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


def _run_one(mode: str, context, actual_m: float | None, *, budget: int | None = None) -> dict:
    kwargs = {}
    label = mode
    if mode == "ultra" and budget is not None:
        kwargs["ultra_budget"] = ExecutionBudget(max_turns=budget, deadline_ms=2500.0)
        label = f"ultra_bootstrap_{budget}turn"
    pipe = ForecastPipeline(mode=mode, **kwargs)
    started = time.perf_counter()
    result = pipe.run(context)
    latency_ms = (time.perf_counter() - started) * 1000.0
    abs_error = None if result.forecast_m is None or actual_m is None else abs(result.forecast_m - actual_m)
    covered = None
    width = None
    if result.lower_m is not None and result.upper_m is not None:
        covered = None if actual_m is None else result.lower_m <= actual_m <= result.upper_m
        width = result.upper_m - result.lower_m
    reward = None
    if actual_m is not None:
        reward = -terminal_loss(
            forecast_m=result.forecast_m,
            lower_m=result.lower_m,
            upper_m=result.upper_m,
            actual_m=actual_m,
            total_calls=result.logical_actions,
            total_latency_ms=latency_ms,
            failed=result.status != "available",
        )
    return {
        "mode": label,
        "forecast_origin": str(context.forecast_time_utc),
        "target_time": str(context.target_time_utc),
        "actual_m": actual_m,
        "forecast_m": result.forecast_m,
        "lower_m": result.lower_m,
        "upper_m": result.upper_m,
        "abs_error_m": abs_error,
        "covered": covered,
        "interval_width_m": width,
        "status": result.status,
        "fallback_used": result.fallback_used,
        "latency_ms": latency_ms,
        "logical_actions": result.logical_actions,
        "physical_expert_calls": result.physical_expert_calls,
        "unique_experts": result.unique_experts,
        "turns": result.number_of_turns,
        "termination_reason": result.termination_reason,
        "reward": reward,
    }


def _demo_contexts():
    station_id = "HOHONU_TEST"
    noaa_id = "NOAA_TEST"
    hohonu = mock_hohonu_observations(station_id, periods=520)
    tide = mock_noaa_tide_predictions(noaa_id, periods=620)
    contexts = []
    actuals = []
    cases = [
        ("2024-01-01T12:00:00Z", 60, 0.08),
        ("2024-01-01T18:00:00Z", 360, 0.08),
        ("2024-01-01T18:00:00Z", 360, 0.40),
    ]
    for origin, horizon, residual in cases:
        noaa = mock_noaa_observations(noaa_id, periods=520, residual_m=residual)
        context = build_forecast_context(
            target_station_id=station_id,
            paired_noaa_station_id=noaa_id,
            horizon_minutes=horizon,
            forecast_time_utc=origin,
            hohonu_observations=hohonu,
            noaa_observations=noaa,
            noaa_tide_predictions=tide,
            station_pair=StationPair(station_id, noaa_id),
        )
        contexts.append(context)
        actuals.append(_nearest_actual(hohonu, context.target_time_utc))
    return contexts, actuals


def _nearest_actual(frame: pd.DataFrame, target_time: pd.Timestamp) -> float | None:
    idx = (frame["timestamp_utc"] - target_time).abs().idxmin()
    return float(frame.loc[idx, "water_level_m"])


def _oracle_rows(contexts, actuals):
    experts = default_experts(include_placeholders=False)
    rows = []
    for context, actual_m in zip(contexts, actuals):
        predictions = {
            name: {
                "status": forecast.status,
                "prediction_m": forecast.predicted_water_level_m,
                "lower_m": forecast.lower_m,
                "upper_m": forecast.upper_m,
                "confidence": forecast.confidence,
            }
            for name, forecast in ((name, expert.forecast(context)) for name, expert in experts.items())
        }
        best = search_oracle_workflows(
            expert_predictions=predictions,
            actual_m=actual_m,
            max_turns=5,
            keep_alternatives=1,
        )[0]
        rows.append(best.to_dict())
    return rows


def _markdown_report(summary: pd.DataFrame, oracle: list[dict]) -> str:
    lines = [
        "# Wai Ultra Local Smoke Benchmark",
        "",
        "This benchmark uses bundled mock observations and tide predictions. It checks orchestration behavior and reporting; it is not real-world forecast validation.",
        "",
        "## Mode Summary",
        "",
        _df_to_markdown(summary),
        "",
        "## Oracle Workflow Upper Bound",
        "",
        "| workflow | loss | reward | calls |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in oracle:
        lines.append(
            f"| {item['workflow_id']} | {item['terminal_loss']:.4f} | {item['reward']:.4f} | {item['total_calls']} |"
        )
    lines.extend([
        "",
        "## Ablations",
        "",
        "Ablations are defined in code but not run here because they require a full historical trajectory dataset.",
    ])
    return "\n".join(lines) + "\n"


def _df_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
