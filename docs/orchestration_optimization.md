# Wai Orchestration — Second-Pass Optimization Report

This documents the measured second-pass optimization of Wai's forecast
orchestration: replacing the flat, single-shot expert router with a faster
adaptive cascade, fixing confirmed correctness bugs, and proving the result with
before/after benchmarks. All measurements use checked-in synthetic / mock
fixtures and require no network access.

Reproduce:

```bash
python -m pytest -q
python -m scripts.benchmark_orchestration --iters 300 --replay-periods 2000
```

Artifacts: `reports/orchestration_baseline.json` (genuine pre-change snapshot),
`reports/orchestration_benchmark.json`, `reports/orchestration_benchmark.md`.

## 1. Baseline architecture and bottlenecks

The original `ForecastPipeline.run` routed once with `RuleBasedOrchestrator`,
ran every selected expert through a serial loop, combined, and verified. Profiling
showed:

- **Per-forecast latency was dominated by context construction (~7.0 ms)**, not
  expert execution (~0.03 ms). cProfile attributed **~72%** of context-build time
  to `_residual_trend_m_per_hour`, an `iterrows` loop calling a nearest-record
  search per row (`O(n·m)`), plus repeated `.copy()` / `to_datetime` / `sort` in
  `_canonical_subset`.
- Historical replay re-filtered and re-sorted the full DataFrames at every
  origin, and ran each expert twice (all experts once, then the pipeline re-ran
  the selected experts).
- Baseline replay throughput: **114.6 rows/s**.

### Confirmed correctness issues (all nine suspected issues were still present)

1. Selected experts executed through a serial loop (`_run_selected`).
2. Replay ran all experts then re-ran the selected experts via the pipeline.
3. `model_disagreement_m` was checked in the router *before* any expert ran.
4. `model_disagreement_m` and `recent_model_performance` were never populated by
   the context builder (dead inputs).
5. `RoutingPolicy.performance_weights` was unused and disconnected from
   combination (the pipeline built `ForecastCombiner()` with no weights).
6. `regional_to_local_residual` reported `lag_minutes` in diagnostics but never
   applied it.
7. A combined forecast rejected by the verifier returned unavailable with no
   final safe-fallback attempt.
8. Router training used a random train/test split, not forward-time.
9. Context construction repeatedly copied, filtered, sorted, and scanned full
   DataFrames; the residual trend used an `O(n·m)` `iterrows` loop.

## 2. New orchestration flow

```
prepared/indexed data -> capability gate -> primary ranking -> Stage 1 primary
  -> post-forecast assessment -> early stop OR conditional escalation
  -> optional Stage 2 experts -> skill-aware combination
  -> dependency-aware verification -> safe fallback -> ForecastResult + trace
```

See `docs/forecast_orchestrator.md` for the full description.

## 3. Files added / changed

Added:
- `src/experts/capabilities.py` — `ExpertSpec` capability metadata.
- `src/orchestration/prepared.py` — `PreparedStationData` indexed layer
  (`searchsorted` slicing + vectorized `merge_asof` residual alignment).
- `src/orchestration/cascade.py` — capability gate, `ExecutionBudget`,
  `ForecastPlan`, `PostForecastAssessment`, `ExpertExecutionResult`,
  `ExecutionTrace`, `AdaptiveCascade`.
- `src/orchestration/skill_store.py` — rolling `SkillStore` with hierarchical
  sparse-data fallback.
- `src/orchestration/executor.py` — bounded parallel executor.
- `scripts/benchmark_orchestration.py` — before/after benchmark.
- `tests/test_cascade.py`, `tests/test_prepared_context.py`,
  `tests/test_skill_store.py`, `tests/test_lag_and_replay_modes.py`,
  `tests/test_learned_router_shadow.py`, `tests/test_pipeline_compat.py`.

Changed:
- `src/orchestration/context.py` — delegates to `PreparedStationData`;
  vectorized residual alignment; lag-aware residual lookup; populates
  `recent_model_performance`.
- `src/experts/base.py` + every expert — `spec` metadata, `latency_ms`.
- `src/experts/regional_residual.py` — genuine lag application.
- `src/orchestration/combiner.py` — skill-aware weights; drops `safe_fallback`
  from valid ensembles; disagreement/horizon uncertainty; measured-uncertainty
  interval floor.
- `src/orchestration/verifier.py` — dependency-aware; immutable; recoverable
  rejection.
- `src/forecasting/pipeline.py` — adaptive cascade default;
  `precomputed_forecasts`; execution-trace diagnostics; post-verifier fallback;
  legacy mode for benchmarking/compat.
- `src/evaluation/historical_replay.py` — prepared incremental advance;
  exhaustive vs policy modes; precomputed reuse; optional skill warmup.
- `src/evaluation/router_training.py` — forward-time split; utility/regret policy
  evaluation.
- `src/orchestration/learned_router.py` — shadow mode + fallback conditions.

## 4. Correctness bugs fixed

| # | Issue | Fix |
|---|---|---|
| 1 | Serial expert loop | Bounded executor; cascade runs cheap experts sequentially, escalation optionally parallel |
| 2 | Replay double-runs experts | Exhaustive mode feeds `precomputed_forecasts`; pipeline reuses (cache hits), no second run |
| 3 | Disagreement checked pre-forecast | Real disagreement computed in `PostForecastAssessment` from completed forecasts |
| 4 | Dead context fields | Disagreement is now derived post-hoc; `recent_model_performance` plumbed via `SkillStore` |
| 5 | `performance_weights` disconnected | `SkillStore` weights feed combination directly |
| 6 | Lag never applied | Lagged residual looked up from observed history; `lag_applied` flagged |
| 7 | No post-verifier fallback | Recoverable rejection triggers one safe-fallback attempt, verified independently |
| 8 | Random router split | Forward-time (rolling-origin) split |
| 9 | Slow context build | Indexed prepared layer + vectorized `merge_asof`; `O(n·m)` loop removed |

## 5. Before/after benchmark results

Context construction (single forecast origin):

| | mean ms |
|---|---|
| original | 7.02 |
| new (single-shot, includes indexing) | ~2.7 |
| new (per-origin on prepared data, replay path) | ~0.36 |

Replay throughput (in-process, same data; routing/strategy differs):

| Mode | rows/s | speedup vs legacy double-run |
|---|---|---|
| legacy double-run | ~228 | 1.0x |
| new exhaustive (reuse) | ~1900 | **~8.4x** |
| new policy | ~2065 | **~9.1x** |

Mean expert calls (legacy flat → adaptive cascade):

| Scenario | legacy | adaptive | note |
|---|---|---|---|
| normal_stable_tide | 2 | **1** | early stop (−50%) |
| short_horizon_fresh_local | 1 | 1 | early stop |
| stale_noaa_input | 1 | 1 | early stop |
| local_qc_failure | 2 | **1** | tide path |
| missing_tide_prediction | 2 (unavailable) | **1 (available)** | persistence path |
| large_noaa_residual | 2 | 2 | event escalation |

Forecast quality on the same replay (routing differs only):

| Metric | legacy flat | adaptive (cold skill) | adaptive (warm skill) |
|---|---|---|---|
| MAE (m) | 0.190 | 0.190 | **0.155** |
| Interval coverage | 0.731 | 0.632 | **0.945** |
| Unavailable rate | 0.0 | 0.0 | 0.0 |

Router policy comparison (forward-time validated): oracle MAE ≈ 0.116, rule
cascade ≈ 0.180, learned router ≈ 0.161, regret ≈ 0.044. Parallel escalation
micro-benchmark (two 50 ms experts): 110 ms sequential → 55 ms parallel (~2x).
Storm-surge fixture: peak-event MAE ≈ 0.41 m, escalation rate ≈ 0.18.

## 6. Performance targets

| # | Target | Result |
|---|---|---|
| 1 | ≥40% fewer expert calls on stable normal | **Met** — normal_stable_tide 2→1 (50%) |
| 2 | ≥2x replay throughput | **Met** — ~8.4x exhaustive, ~9.1x policy |
| 3 | Reduce or preserve p95 latency | **Mostly met** — see note below |
| 4 | Preserve/improve MAE and coverage | **Met with measured skill** — warm: MAE 0.155<0.190, coverage 0.945>0.731 |
| 5 | Do not increase unavailable rate | **Met** — unchanged (and `missing_tide` now available) |
| 6 | Preserve datum/QC/leakage/fallback | **Met** — covered by tests |

### Targets not fully reached / honest caveats

- **Target 3 (p95):** End-to-end forecast latency *improves* because context
  construction (7.0 → 2.7 ms) dominates and dwarfs the cascade's fixed
  capability-gate + ranking overhead (~0.03 ms). But in the per-scenario micro
  benchmark, which excludes context build and uses near-free arithmetic experts,
  that fixed overhead makes adaptive p95 slightly higher in absolute terms
  (e.g. 0.006 → 0.040 ms for `missing_tide_prediction`). These are sub-0.15 ms
  differences; on any real expert workload they are negligible.
- **Target 4 (coverage) at cold start:** With no measured skill, the
  single-expert early-stop interval is tighter and under-covers (0.632 vs legacy
  0.731). The measured-uncertainty interval floor restores and exceeds legacy
  coverage (0.945) once ~5 validation samples accrue per regime cell. This is
  reported honestly rather than hidden.

## 7. Test commands and results

```bash
python -m pytest -q          # 292 passed
```

The new suites cover capability exclusion, primary selection, early stop,
escalation (failure, event, post-forecast baseline disagreement), budget and
timeout isolation, deterministic parallel ordering, fallback reservation,
verifier-triggered fallback, dependency-aware staleness, no-duplicate execution,
precomputed reuse, context indexing equivalence, vectorized residual alignment,
real lag application, SkillStore sparse fallback, skill-aware combination,
learned-router shadow mode, forward-time router validation, replay leakage
protection, and `run(context)` backward compatibility.

## 8. Scientific and operational limitations

- No LLM is used in the numerical forecast path; the cascade is deterministic.
- The synthetic mock fixtures intentionally differ between the local and regional
  signals more than a faithful station pair would, so absolute MAE/coverage
  numbers are illustrative, not operational.
- Datum conversion is still not implemented (fails closed on mismatch).
- Residual transfer scale is static; the lag is applied from observed history but
  not learned. Weather-aware, spatial, and learned-local-residual experts remain
  placeholders.
- The learned router is advisory (shadow mode only).
- The cold-start coverage gap above is a genuine precision/coverage tradeoff.

## 9. Next best improvement (evidence-based)

The replay win came almost entirely from the prepared/indexed context layer; the
remaining per-origin cost is now dominated by Python-level per-expert object
construction and JSON serialization of diagnostics, not numerics. The highest-
value next step is to **persist the `SkillStore` across runs and seed it from an
exhaustive forward-replay before serving**, so the cascade starts warm: this
closes the cold-start coverage gap (the only target with a caveat) and lets the
skill-aware route score and interval floor operate from the first forecast,
turning the warm-store quality (MAE 0.155, coverage 0.945) into the default
rather than something earned over the first few dozen origins.
