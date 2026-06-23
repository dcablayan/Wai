"""Run a deterministic local smoke benchmark for Wai Ultra orchestration.

This benchmark uses bundled mock data. It verifies quality/compute reporting and
mode compatibility; it is not a real-world scientific validation.
"""

from __future__ import annotations

import json
import math
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
    summary = _summarize(result_frame)
    oracle = _oracle_rows(contexts, actuals)
    ablations = _numeric_ablations(result_frame, summary)

    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / "ultra_benchmark_results.json"
    json_path.write_text(json.dumps(_json_sanitize({
        "description": "Local deterministic mock-data smoke benchmark; not a real-world validation.",
        "n_forecast_origins": len(contexts),
        "rows": result_frame.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "oracle": oracle,
        "ablations": ablations,
    }), indent=2, sort_keys=True, default=str, allow_nan=False))

    md_path = REPORTS_DIR / "ultra_benchmark_results.md"
    md_path.write_text(_markdown_report(summary, oracle, ablations))
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
        "peak_event": abs(float(context.recent_noaa_residual_m or 0.0)) >= 0.25,
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
        "thinker_calls": result.thinker_calls,
        "verifier_calls": result.verifier_calls,
        "worker_calls": result.worker_calls,
        "fallback_calls": result.fallback_calls,
        "unique_experts": result.unique_experts,
        "unique_base_numerical_experts": _unique_base_numerical_experts(result),
        "replan_rate": 1.0 if "replan" in result.termination_reason else 0.0,
        "nested_replan_success": 1.0 if result.termination_reason == "child_replan_verifier_acceptance" else 0.0,
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
    cases = []
    origins = [
        "2024-01-01T06:00:00Z",
        "2024-01-01T12:00:00Z",
        "2024-01-01T18:00:00Z",
        "2024-01-02T00:00:00Z",
    ]
    for idx, origin in enumerate(origins):
        cases.append((origin, 60, 0.05 + 0.03 * idx))
        cases.append((origin, 360, 0.08 if idx % 2 == 0 else 0.35))
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


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for mode, group in rows.groupby("mode"):
        available = group[group["status"] == "available"]
        peak = group[group["peak_event"].astype(bool)]
        fallback_verified = group[
            (group["fallback_used"].astype(bool))
            & (group["status"] == "available")
            & group["termination_reason"].astype(str).str.contains("fallback|acceptance", case=False, regex=True)
        ]
        summaries.append({
            "mode": mode,
            "overall_mae": _mean(group["abs_error_m"]),
            "peak_event_mae": _mean(peak["abs_error_m"]) if len(peak) else None,
            "interval_coverage": _mean(group["covered"].dropna().astype(float)) if group["covered"].notna().any() else None,
            "interval_width": _mean(group["interval_width_m"]),
            "unavailable_rate": float((group["status"] != "available").mean()),
            "verified_fallback_rate": float(len(fallback_verified) / len(group)) if len(group) else None,
            "fallback_failure_reasons": sorted(
                str(value)
                for value in group.loc[
                    (group["fallback_used"].astype(bool)) & (group["status"] != "available"),
                    "termination_reason",
                ].dropna().unique()
            ),
            "p50_latency_ms": float(group["latency_ms"].quantile(0.50)),
            "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
            "logical_actions": _mean(group["logical_actions"]),
            "physical_worker_calls": _mean(group["physical_expert_calls"]),
            "thinker_calls": _mean(group["thinker_calls"]),
            "verifier_calls": _mean(group["verifier_calls"]),
            "unique_base_numerical_experts": _mean(group["unique_base_numerical_experts"]),
            "replan_rate": _mean(group["replan_rate"]),
            "nested_replan_success_rate": _mean(group["nested_replan_success"]),
            "routing_regret": None,
            "reward": _mean(group["reward"]),
            "n_available": int(len(available)),
            "n": int(len(group)),
        })
    return pd.DataFrame(summaries)


def _numeric_ablations(rows: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    base = _summary_row(summary, "ultra_bootstrap_5turn")
    definitions = {
        "no_thinker": "ultra_bootstrap_2turn",
        "no_verifier": "ultra_bootstrap_1turn",
        "no_result_conditioned_transcript": "legacy",
        "no_access_control": "legacy",
        "no_replan": "ultra_bootstrap_5turn",
        "fixed_topology": "legacy",
        "no_randomized_pools": "ultra_bootstrap_5turn",
        "mini_cascade": "mini",
    }
    ablations = []
    for name, mode in definitions.items():
        row = _summary_row(summary, mode)
        if row is None:
            ablations.append({"ablation": name, "mode": mode, "status": "unavailable"})
            continue
        ablations.append({
            "ablation": name,
            "mode": mode,
            "status": "numeric_local_proxy",
            "overall_mae": row.get("overall_mae"),
            "unavailable_rate": row.get("unavailable_rate"),
            "reward": row.get("reward"),
            "delta_reward_vs_ultra_5turn": None
            if base is None or row.get("reward") is None or base.get("reward") is None
            else float(row["reward"] - base["reward"]),
        })
    return ablations


def _summary_row(summary: pd.DataFrame, mode: str) -> dict | None:
    matches = summary[summary["mode"] == mode]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _markdown_report(summary: pd.DataFrame, oracle: list[dict], ablations: list[dict]) -> str:
    lines = [
        "# Wai Ultra Local Historical Benchmark",
        "",
        "This benchmark uses all bundled mock forecast origins generated by the script. It checks orchestration behavior and reporting; it is not real-world forecast validation.",
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
        "| ablation | mode/proxy | MAE | unavailable | reward delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for item in ablations:
        lines.append(
            "| {ablation} | {mode} | {mae} | {unavail} | {delta} |".format(
                ablation=item["ablation"],
                mode=item.get("mode"),
                mae=_fmt(item.get("overall_mae")),
                unavail=_fmt(item.get("unavailable_rate")),
                delta=_fmt(item.get("delta_reward_vs_ultra_5turn")),
            )
        )
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
            values.append(_fmt(row[col]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _unique_base_numerical_experts(result) -> int:
    experts = {
        expert
        for expert in result.experts_used
        if expert
        not in {
            "ensemble_synthesis",
            "safe_fallback",
            "physics_datum_verifier",
            "cross_source_verifier",
            "calibration_verifier",
            "event_risk_verifier",
        }
    }
    return len(experts)


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _json_sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None:
        return None
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


if __name__ == "__main__":
    main()
