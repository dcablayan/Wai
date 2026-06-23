"""Reproducible before/after benchmark for Wai forecast orchestration.

Compares the legacy flat router (``ForecastPipeline(adaptive=False)``) against
the adaptive cascade (``ForecastPipeline(adaptive=True)``) in one process on
checked-in synthetic / mock fixtures, plus the legacy double-running replay
against the new exhaustive (reuse) and policy replay modes.

Outputs:
    reports/orchestration_benchmark.json
    reports/orchestration_benchmark.md

No network access is required.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.canonicalize import canonicalize_frame
from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import _mock_tide, mock_noaa_observations, mock_noaa_tide_predictions
from src.evaluation import EXHAUSTIVE, POLICY, HistoricalReplayConfig, run_historical_replay
from src.evaluation.router_training import evaluate_router_policies
from src.experts.base import ExpertForecast, ForecastExpert
from src.experts.capabilities import LATENCY_INSTANT, ExpertSpec
from src.forecasting import ForecastPipeline
from src.forecasting.pipeline import default_experts
from src.orchestration.cascade import ExecutionBudget
from src.orchestration.context import build_forecast_context
from src.orchestration.prepared import PreparedStationData
from src.orchestration.context import context_from_prepared
from src.data.station_mapping import StationPair

STATION = "HOHONU_TEST"
NOAA = "NOAA_TEST"


# --------------------------------------------------------------------------- #
# Scenario fixtures
# --------------------------------------------------------------------------- #
def _ctx(*, horizon=360, forecast_time="2024-01-01T18:00:00Z", residual=0.08,
         hohonu_qc="pass", noaa_periods=300, tide=True):
    hohonu = mock_hohonu_observations(STATION, periods=300, qc_status=hohonu_qc)
    noaa_obs = mock_noaa_observations(NOAA, periods=noaa_periods, residual_m=residual)
    noaa_tide = mock_noaa_tide_predictions(NOAA, periods=420)
    if not tide:
        noaa_tide = noaa_tide.iloc[0:0]
    return build_forecast_context(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        horizon_minutes=horizon, forecast_time_utc=forecast_time,
        hohonu_observations=hohonu, noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide, station_pair=StationPair(STATION, NOAA),
    )


SCENARIOS = {
    "normal_stable_tide": dict(horizon=360, residual=0.08),
    "short_horizon_fresh_local": dict(horizon=30, residual=0.08),
    "large_noaa_residual": dict(horizon=360, residual=0.4),
    "local_qc_failure": dict(horizon=30, hohonu_qc="fail", noaa_periods=10),
    "stale_noaa_input": dict(horizon=60, forecast_time="2024-01-02T00:00:00Z", noaa_periods=80),
    "missing_tide_prediction": dict(horizon=360, tide=False),
}


class _ExplodingExpert(ForecastExpert):
    model_name = "exploding"
    spec = ExpertSpec(model_name="exploding", requires_tide=True, latency_class=LATENCY_INSTANT)

    def forecast(self, context):
        raise RuntimeError("boom")


class _SlowExpert(ForecastExpert):
    model_name = "slow"
    spec = ExpertSpec(model_name="slow", requires_tide=True, latency_class=LATENCY_INSTANT)

    def forecast(self, context):
        time.sleep(0.05)
        tide = context.noaa_tide_prediction
        return ExpertForecast(
            model_name=self.model_name, forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc, horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=float(tide["water_level_m"]), lower_m=float(tide["water_level_m"]) - 0.1,
            upper_m=float(tide["water_level_m"]) + 0.1, confidence=0.7,
        )


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def time_pipeline(pipe, ctx, iters=300):
    times, calls, escalated, early_stop = [], [], 0, 0
    statuses = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = pipe.run(ctx)
        times.append((time.perf_counter() - t0) * 1000.0)
        trace = res.diagnostics.get("trace", {})
        calls.append(trace.get("expert_calls", len(res.diagnostics.get("experts", {}))))
        escalated += int(bool(trace.get("escalated")))
        early_stop += int(bool(trace.get("early_stop_reason")))
        statuses.append(res.status)
    times.sort()
    sample = pipe.run(ctx)
    return {
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 4),
        "mean_ms": round(statistics.mean(times), 4),
        "mean_expert_calls": round(statistics.mean(calls), 3),
        "escalation_rate": round(escalated / iters, 3),
        "early_stop_rate": round(early_stop / iters, 3),
        "status": sample.status,
        "experts_used": sample.experts_used,
    }


def legacy_replay_throughput(hohonu, noaa_obs, noaa_tide, cfg):
    """Reproduce the OLD replay cost: slow per-origin context + run-all + rerun."""
    experts = default_experts(include_placeholders=False)
    pipe = ForecastPipeline(adaptive=False)
    h = _sort(hohonu); n = _sort(noaa_obs); t = _sort(noaa_tide)
    start_origin = h["timestamp_utc"].min() + pd.Timedelta(hours=cfg.min_history_hours)
    last_origin = h["timestamp_utc"].max() - pd.Timedelta(minutes=cfg.horizon_minutes)
    rows = 0
    t0 = time.perf_counter()
    origin = start_origin
    while origin <= last_origin:
        h_hist = h[h["timestamp_utc"] <= origin].copy()
        n_hist = n[n["timestamp_utc"] <= origin].copy()
        ctx = build_forecast_context(
            target_station_id=STATION, paired_noaa_station_id=NOAA,
            horizon_minutes=cfg.horizon_minutes, forecast_time_utc=origin,
            hohonu_observations=h_hist, noaa_observations=n_hist, noaa_tide_predictions=t,
        )
        _ = {name: ex.forecast(ctx) for name, ex in experts.items()}  # run all
        _ = pipe.run(ctx)  # reruns selected (the OLD double-run)
        rows += 1
        origin += pd.Timedelta(minutes=cfg.step_minutes)
    elapsed = time.perf_counter() - t0
    return {"rows": rows, "elapsed_s": round(elapsed, 4), "rows_per_s": round(rows / elapsed, 2)}


def new_replay_throughput(hohonu, noaa_obs, noaa_tide, cfg, mode):
    pipe = ForecastPipeline(adaptive=True)
    t0 = time.perf_counter()
    replay = run_historical_replay(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=hohonu, noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide, pipeline=pipe, config=cfg, mode=mode,
    )
    elapsed = time.perf_counter() - t0
    rows = len(replay)
    out = {"rows": rows, "elapsed_s": round(elapsed, 4),
           "rows_per_s": round(rows / elapsed, 2) if elapsed else None}
    if "expert_calls" in replay:
        out["mean_expert_calls"] = round(float(replay["expert_calls"].mean()), 3)
    if mode == POLICY and "actual_m" in replay:
        out.update(_quality_metrics(replay))
    return out, replay


def _quality_metrics(replay):
    df = replay.dropna(subset=["forecast_error_m"])
    if df.empty:
        return {}
    abs_err = df["forecast_error_m"].abs()
    peak = df[df["actual_m"].abs() >= 0.75]
    coverage = None
    if {"forecast_lower_m", "forecast_upper_m"}.issubset(df.columns):
        inside = (df["actual_m"] >= df["forecast_lower_m"]) & (df["actual_m"] <= df["forecast_upper_m"])
        coverage = round(float(inside.mean()), 4)
    return {
        "mae_m": round(float(abs_err.mean()), 4),
        "peak_event_mae_m": round(float(peak["forecast_error_m"].abs().mean()), 4) if not peak.empty else None,
        "interval_coverage": coverage,
        "unavailable_rate": round(float((replay["result_status"] != "available").mean()), 4),
        "fallback_rate": round(float(replay["fallback_used"].mean()), 4),
        "escalation_rate": round(float(replay["escalated"].mean()), 4),
    }


def _sort(frame):
    df = frame.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _surge_fixtures(periods):
    """NOAA obs + Hohonu obs carrying a transient non-tidal surge (peak event)."""
    start = "2024-01-01T00:00:00Z"
    freq = "6min"
    ts, tide = _mock_tide(start, periods, freq)
    n = len(tide)
    idx = np.arange(n)
    # A Gaussian surge centred mid-series, peaking ~0.9 m above tide.
    surge = 0.9 * np.exp(-((idx - n * 0.6) ** 2) / (2 * (n * 0.05) ** 2))
    noaa = canonicalize_frame(
        pd.DataFrame({"timestamp": ts, "station_id": NOAA, "water_level": tide + surge,
                      "units": "m", "lat": 21.3, "lon": -157.8, "datum": "MLLW",
                      "qc_status": "verified", "qc_flags": [[] for _ in range(n)]}),
        source="NOAA_COOPS_MOCK", record_type="observation",
        qc_status_col="qc_status", qc_flags_col="qc_flags", retrieved_at="2024-02-01T00:00:00Z",
    )
    # Local station sees the surge scaled (0.85) on its own tide curve.
    t_h = np.arange(n) * (pd.Timedelta(freq).total_seconds() / 3600.0)
    local_tide = 0.55 * np.sin(2 * np.pi * t_h / 12.42) + 0.02 * np.sin(2 * np.pi * t_h / 3.0)
    hohonu = canonicalize_frame(
        pd.DataFrame({"timestamp": ts, "station_id": STATION, "water_level": local_tide + 0.85 * surge,
                      "units": "m", "lat": 21.3, "lon": -157.8, "datum": "MLLW",
                      "qc_status": "pass", "qc_flags": [[] for _ in range(n)]}),
        source="HOHONU_MOCK", record_type="observation",
        qc_status_col="qc_status", qc_flags_col="qc_flags", retrieved_at="2024-02-01T00:00:00Z",
    )
    tide_pred = mock_noaa_tide_predictions(NOAA, periods=int(periods * 1.2))
    return hohonu, noaa, tide_pred


# --------------------------------------------------------------------------- #
# Parallel-escalation micro-benchmark (proves parallel path works)
# --------------------------------------------------------------------------- #
def parallel_escalation_demo():
    ctx = _ctx(horizon=360, residual=0.5)  # forces escalation
    experts = {
        "local_tide": default_experts()["local_tide"],
        "slow_a": _SlowExpert(),
    }
    experts["slow_a"].model_name = "slow_a"
    experts_b = _SlowExpert(); experts_b.model_name = "slow_b"
    experts["slow_b"] = experts_b
    from src.orchestration.executor import run_experts
    slow_list = [experts["slow_a"], experts["slow_b"]]
    t0 = time.perf_counter(); run_experts(slow_list, ctx, parallel=False); seq = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); run_experts(slow_list, ctx, parallel=True, max_parallelism=2); par = (time.perf_counter() - t0) * 1000
    return {"two_slow_experts_sequential_ms": round(seq, 2),
            "two_slow_experts_parallel_ms": round(par, 2),
            "speedup": round(seq / par, 2) if par else None}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark Wai orchestration old vs new.")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--replay-periods", type=int, default=1440)
    parser.add_argument("--out-json", default="reports/orchestration_benchmark.json")
    parser.add_argument("--out-md", default="reports/orchestration_benchmark.md")
    args = parser.parse_args(argv)

    legacy = ForecastPipeline(adaptive=False)
    adaptive = ForecastPipeline(adaptive=True)

    scenario_rows = {}
    for name, kw in SCENARIOS.items():
        ctx = _ctx(**kw)
        scenario_rows[name] = {
            "legacy": time_pipeline(legacy, ctx, args.iters),
            "adaptive": time_pipeline(adaptive, ctx, args.iters),
        }

    # Expert-exception scenario.
    ctx_exc = _ctx(horizon=360, residual=0.08)
    exc_experts = default_experts()
    exc_experts["local_tide"] = _ExplodingExpert()
    exc_experts["local_tide"].model_name = "local_tide"
    pipe_exc = ForecastPipeline(adaptive=True, experts=exc_experts)
    res_exc = pipe_exc.run(ctx_exc)
    scenario_rows["expert_exception"] = {
        "adaptive": {
            "status": res_exc.status,
            "experts_used": res_exc.experts_used,
            "fallback_used": res_exc.fallback_used,
            "excluded": list(res_exc.experts_excluded),
        }
    }

    # Replay throughput.
    hohonu = mock_hohonu_observations(STATION, periods=args.replay_periods)
    noaa_obs = mock_noaa_observations(NOAA, periods=args.replay_periods, residual_m=0.12)
    noaa_tide = mock_noaa_tide_predictions(NOAA, periods=int(args.replay_periods * 1.2))
    cfg = HistoricalReplayConfig(horizon_minutes=360, min_history_hours=12, step_minutes=60)

    legacy_rt = legacy_replay_throughput(hohonu, noaa_obs, noaa_tide, cfg)
    new_ex, ex_replay = new_replay_throughput(hohonu, noaa_obs, noaa_tide, cfg, EXHAUSTIVE)
    new_po, po_replay = new_replay_throughput(hohonu, noaa_obs, noaa_tide, cfg, POLICY)
    policy_eval = evaluate_router_policies(ex_replay)

    # Before/after forecast quality on the SAME data (routing differs only).
    legacy_replay = run_historical_replay(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=hohonu, noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide, pipeline=ForecastPipeline(adaptive=False),
        config=cfg, mode=POLICY,
    )
    # Warm-store adaptive run (exhaustive mode supplies per-expert outcomes so
    # skill accrues forward-in-time and the cascade can size intervals from
    # measured residual uncertainty — no future leakage; routing/forecast match
    # policy mode, only interval sizing changes once support exists).
    warm_pipe = ForecastPipeline(adaptive=True)
    warm_replay = run_historical_replay(
        target_station_id=STATION, paired_noaa_station_id=NOAA,
        hohonu_observations=hohonu, noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide, pipeline=warm_pipe,
        config=cfg, mode=EXHAUSTIVE, update_skill=True,
    )
    quality_compare = {
        "legacy_flat_router": _quality_metrics(legacy_replay),
        "adaptive_cascade_cold": _quality_metrics(po_replay),
        "adaptive_cascade_warm": _quality_metrics(warm_replay),
    }

    # Storm-surge fixture exercises peak-event error and escalation.
    surge_hohonu, surge_noaa, surge_tide = _surge_fixtures(args.replay_periods)
    _, surge_replay = new_replay_throughput(surge_hohonu, surge_noaa, surge_tide, cfg, POLICY)
    surge_quality = _quality_metrics(surge_replay)

    report = {
        "scenarios": scenario_rows,
        "replay": {
            "legacy_double_run": legacy_rt,
            "new_exhaustive_reuse": new_ex,
            "new_policy": new_po,
            "throughput_speedup_exhaustive": round(new_ex["rows_per_s"] / legacy_rt["rows_per_s"], 2) if legacy_rt["rows_per_s"] else None,
            "throughput_speedup_policy": round(new_po["rows_per_s"] / legacy_rt["rows_per_s"], 2) if legacy_rt["rows_per_s"] else None,
        },
        "router_policy_eval": policy_eval.__dict__,
        "forecast_quality_before_after": quality_compare,
        "storm_surge_quality": surge_quality,
        "parallel_escalation": parallel_escalation_demo(),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(args.out_md).write_text(_render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nSaved {args.out_json} and {args.out_md}")
    return 0


def _render_markdown(report) -> str:
    lines = ["# Wai Orchestration Benchmark", "",
             "Legacy flat router vs adaptive cascade on checked-in synthetic fixtures.",
             "No network access required.", "",
             "## Per-scenario latency and expert calls", "",
             "| Scenario | Path | p50 ms | p95 ms | mean calls | early-stop | escalation | experts |",
             "|---|---|---|---|---|---|---|---|"]
    for name, paths in report["scenarios"].items():
        for path, m in paths.items():
            if "p50_ms" not in m:
                lines.append(f"| {name} | {path} | - | - | - | - | - | {m.get('experts_used')} |")
                continue
            lines.append(
                f"| {name} | {path} | {m['p50_ms']} | {m['p95_ms']} | {m['mean_expert_calls']} "
                f"| {m.get('early_stop_rate','-')} | {m.get('escalation_rate','-')} | {','.join(m['experts_used'])} |"
            )
    r = report["replay"]
    lines += ["", "## Historical replay throughput", "",
              "| Mode | rows | seconds | rows/s | mean calls |",
              "|---|---|---|---|---|",
              f"| legacy double-run | {r['legacy_double_run']['rows']} | {r['legacy_double_run']['elapsed_s']} | {r['legacy_double_run']['rows_per_s']} | 5 (all) |",
              f"| new exhaustive (reuse) | {r['new_exhaustive_reuse']['rows']} | {r['new_exhaustive_reuse']['elapsed_s']} | {r['new_exhaustive_reuse']['rows_per_s']} | {r['new_exhaustive_reuse'].get('mean_expert_calls')} |",
              f"| new policy | {r['new_policy']['rows']} | {r['new_policy']['elapsed_s']} | {r['new_policy']['rows_per_s']} | {r['new_policy'].get('mean_expert_calls')} |",
              "",
              f"- Exhaustive throughput speedup vs legacy: **{r['throughput_speedup_exhaustive']}x**",
              f"- Policy throughput speedup vs legacy: **{r['throughput_speedup_policy']}x**"]
    q = report["replay"]["new_policy"]
    if "mae_m" in q:
        lines += ["", "## Policy-mode forecast quality", "",
                  f"- MAE: {q['mae_m']} m", f"- Peak-event MAE: {q.get('peak_event_mae_m')} m",
                  f"- Interval coverage: {q.get('interval_coverage')}",
                  f"- Unavailable rate: {q.get('unavailable_rate')}",
                  f"- Fallback rate: {q.get('fallback_rate')}",
                  f"- Escalation rate: {q.get('escalation_rate')}"]
    qc = report.get("forecast_quality_before_after")
    if qc:
        lb = qc.get("legacy_flat_router", {})
        ac = qc.get("adaptive_cascade_cold", {})
        aw = qc.get("adaptive_cascade_warm", {})
        lines += ["", "## Forecast quality before/after (same data, routing differs)", "",
                  "| Metric | legacy flat | adaptive (cold skill) | adaptive (warm skill) |",
                  "|---|---|---|---|",
                  f"| MAE (m) | {lb.get('mae_m')} | {ac.get('mae_m')} | {aw.get('mae_m')} |",
                  f"| Interval coverage | {lb.get('interval_coverage')} | {ac.get('interval_coverage')} | {aw.get('interval_coverage')} |",
                  f"| Unavailable rate | {lb.get('unavailable_rate')} | {ac.get('unavailable_rate')} | {aw.get('unavailable_rate')} |",
                  f"| Fallback rate | {lb.get('fallback_rate')} | {ac.get('fallback_rate')} | {aw.get('fallback_rate')} |"]
    sq = report.get("storm_surge_quality")
    if sq:
        lines += ["", "## Storm-surge fixture (peak-event)", "",
                  f"- MAE: {sq.get('mae_m')} m", f"- Peak-event MAE: {sq.get('peak_event_mae_m')} m",
                  f"- Interval coverage: {sq.get('interval_coverage')}",
                  f"- Escalation rate: {sq.get('escalation_rate')}",
                  f"- Fallback rate: {sq.get('fallback_rate')}"]
    e = report["router_policy_eval"]
    lines += ["", "## Router policy comparison (forward-validated)", "",
              f"- Validation: {e['validation']} (n_test={e['n_test']})",
              f"- Oracle best-expert MAE: {e['oracle_mae']}",
              f"- Rule/cascade MAE: {e['rule_router_mae']}",
              f"- Learned-router MAE: {e['learned_router_mae']}",
              f"- Routing regret vs oracle: {e['routing_regret_m']}"]
    p = report["parallel_escalation"]
    lines += ["", "## Parallel escalation micro-benchmark", "",
              f"- Two slow experts sequential: {p['two_slow_experts_sequential_ms']} ms",
              f"- Two slow experts parallel: {p['two_slow_experts_parallel_ms']} ms",
              f"- Speedup: {p['speedup']}x"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
