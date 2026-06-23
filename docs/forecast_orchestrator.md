# Wai Forecast Orchestrator

This document describes the first rule-based foundation for a Fugu-inspired
regional-to-local water-level forecasting orchestrator. It is deterministic:
the router chooses numerical experts, and no LLM generates water-level values.

## Data Flow

1. `src/data/hohonu.py` ingests local Hohonu observations.
2. `src/data/noaa.py` ingests NOAA CO-OPS observations, tide predictions, and
   weather-like products when available.
3. `src/data/canonicalize.py` converts both sources into the canonical schema:
   `timestamp_utc`, `source`, `station_id`, `latitude`, `longitude`,
   `water_level_m`, `datum`, `record_type`, `qc_status`, `qc_flags`,
   `retrieved_at`, and `latency_seconds`.
4. `src/orchestration/context.py` builds a leakage-safe `ForecastContext` at a
   forecast origin.
5. `src/orchestration/router.py` selects one to three experts.
6. `src/forecasting/pipeline.py` runs selected experts, combines successful
   forecasts, verifies the result, and returns a structured `ForecastResult`.

The adapters include chunking, timeouts, retry/backoff, 429 handling, local
JSON caching, and mock fixtures for offline tests.

## Station Pairing

Local-to-regional pairing lives in `src/data/station_mapping.py`. A
`StationPair` stores the target station, paired NOAA station, residual transfer
scale, lag, and expected datum. Defaults currently include `DEMO-HNL` paired to
NOAA `1612340` and `DEMO-SFO` paired to NOAA `9414290`. Production use should
add explicit reviewed mappings rather than relying on nearest-neighbor guesses.

## Datums

The canonical layer standardizes water levels to meters but does not convert
vertical datums. `assert_compatible_datums()` fails closed when Hohonu, NOAA
observations, and NOAA tide predictions are on different datums. This prevents
residual forecasts from silently combining MLLW, MSL, NAVD88, or private local
datums without a verified conversion.

## Experts

Working experts:

- `local_persistence`: latest Hohonu observation plus recent local trend.
- `local_tide`: local tide prediction when available, otherwise paired NOAA
  tide prediction.
- `noaa_residual`: NOAA tide prediction plus recent NOAA observed-minus-tide
  residual.
- `regional_to_local_residual`: transfers the NOAA residual to the local
  station using the configured scale and lag metadata.
- `safe_fallback`: conservative tide-only baseline when live observations fail.

Placeholders that intentionally return unavailable:

- `weather_aware`
- `spatial_neighboring_station`
- `learned_local_residual`

## Adaptive Cascade (default routing)

`ForecastPipeline(adaptive=True)` (the default) runs a staged cascade instead of
the original flat, single-shot selection:

```
prepared/indexed data
  -> capability gate           (exclude impossible experts up front)
  -> primary expert ranking    (cost-aware, skill-aware route score)
  -> Stage 1 cheap forecast    (one primary expert; baseline read from context)
  -> post-forecast assessment  (confidence, interval, real disagreement, risk)
  -> early stop OR escalation   (run 1-2 more experts only when justified)
  -> skill-aware combination
  -> dependency-aware verification
  -> safe fallback (reserved call)
  -> ForecastResult + execution trace
```

The legacy flat router is preserved as `ForecastPipeline(adaptive=False)` and via
`RuleBasedOrchestrator` for benchmarking and backward compatibility.

### Capability metadata and gate

Every expert declares an `ExpertSpec` (`src/experts/capabilities.py`): required
data sources, supported horizon range, whether local/NOAA observations or a tide
prediction are required, whether it is a safe baseline, latency class, compute
cost, thread safety, and cacheability. `CapabilityGate` uses this metadata to
exclude impossible experts *before* routing — the orchestrator no longer
discovers missing dependencies by executing experts and waiting for them to
fail.

### Route score

Eligible experts are ranked by an interpretable, cost-aware score (lower is
better):

```
route_score = predicted_error
            + latency_weight * expected_latency
            + failure_weight * failure_risk
            + safety_penalty (for safe baselines)
            + regime/horizon adjustment
```

`predicted_error` and `failure_risk` come from the `SkillStore` (recent measured
skill), not from self-reported confidence alone. The rule-based capability and
safety gate remains authoritative: the score only orders *safe, eligible*
experts.

### Early stop and escalation

After the primary forecast, `PostForecastAssessment` records success,
confidence, interval width, difference from the tide baseline, **actual**
disagreement among completed forecasts, input quality, recent validation skill,
out-of-distribution and suspicious-jump flags, and the remaining budget.
Disagreement is computed only after forecasts exist — never from a context field
populated before any expert has run. The cascade early-stops when the primary is
confident, well-bounded, agrees with the baseline (for tide-referenced experts),
and is operating in a well-supported regime; otherwise it escalates one or two
additional experts. The default route is capped at three numerical expert calls
plus a reserved safe-fallback call.

### Execution budget

`ExecutionBudget` bounds each forecast: `deadline_ms`, `max_expert_calls`,
`max_parallelism`, `per_expert_timeout_ms`, `reserve_fallback_call`, and
`allow_parallel_escalation`. The safe-fallback call is reserved so it cannot be
starved by optional experts.

### Parallelism

Cheap arithmetic experts run sequentially by default (thread setup costs more
than the work). The bounded executor (`src/orchestration/executor.py`) supports
parallel Stage-2 escalation for independent, thread-safe experts with per-expert
timeouts, deterministic output ordering, and exception isolation. Benchmarks
show parallelism helps only when experts do real I/O or compute; the included
fake-slow-expert tests prove the parallel path works.

### Skill store

`SkillStore` tracks rolling (EWMA) MAE, RMSE, interval coverage, failure rate,
latency, and sample count by `(expert, station, horizon bucket, regime)`, with a
hierarchy that falls back to coarser keys when a cell is sparse
(`station+horizon+regime -> station+horizon -> horizon -> global prior`). A tiny
sample is shrunk toward the prior, so it is never treated as strong evidence.

### Combination and uncertainty

Combination depends on the number and measured skill of successful experts: one
expert is returned directly; two use a skill-weighted average; three or more use
a weighted median. `safe_fallback` is not averaged into an otherwise-valid
ensemble. Final uncertainty combines the experts' own intervals with
between-expert disagreement and a horizon term, and is floored at the measured
historical residual uncertainty so an early-stopped single-expert forecast does
not under-cover.

### Dependency-aware verification and fallback

The verifier only penalises stale or failed sources the forecasts actually use:
a NOAA-only forecast is not downgraded for unused stale local data, and vice
versa. It works on a copy of the combined forecast (no in-place mutation). When
it rejects a recoverable non-fallback result, the pipeline attempts the safe
baseline once and verifies it independently, with no recursive fallback loops.

### Observability

`ForecastResult.diagnostics["trace"]` reports `context_build_ms`,
`capability_gate_ms`, `routing_ms`, `per_expert_ms`, `combination_ms`,
`verification_ms`, `total_ms`, `expert_calls`, `cache_hits`, `stage_1_expert`,
`escalated`, `escalation_reasons`, `early_stop_reason`, `timed_out_experts`,
`route_source`, `fallback_reason`, and `execution_budget_used`.

## Environment Variables

- `HOHONU_API_KEY`: token used by the Hohonu adapter for live requests.

NOAA CO-OPS public water-level and prediction products do not require an API
key. Never commit Hohonu API keys, private station IDs, or customer data.

## Example Forecast

Run a deterministic offline example:

```bash
python -m scripts.run_orchestrated_forecast --horizon-minutes 360
```

Example output shape:

```json
{
  "station_id": "HOHONU_TEST",
  "forecast_time_utc": "2024-01-01 18:00:00+00:00",
  "target_time_utc": "2024-01-02 00:00:00+00:00",
  "horizon_minutes": 360,
  "forecast_m": 0.12,
  "lower_m": -0.08,
  "upper_m": 0.32,
  "confidence": 0.77,
  "regime": "normal_tide_residual",
  "experts_used": ["local_tide", "noaa_residual"],
  "experts_excluded": {},
  "combination_method": "weighted_median",
  "fallback_used": false,
  "warnings": [],
  "diagnostics": {}
}
```

## Historical Replay

Historical replay walks forward through forecast origins without using future
observations as model inputs. It builds one `PreparedStationData` index per
station and advances incrementally with `searchsorted` slices instead of
re-filtering and re-sorting the full dataset at every origin. Two modes are kept
strictly separate:

- **EXHAUSTIVE** runs every available expert exactly once per origin, stores the
  outputs, and feeds them back to the pipeline as `precomputed_forecasts` so no
  expert is ever executed twice at the same origin. This produces the
  router-training / evaluation dataset.
- **POLICY** runs only the experts the adaptive cascade requests, measuring real
  production compute (mean expert calls, latency). Exhaustive compute cost is
  never mixed into the adaptive production cost.

```bash
python -m scripts.run_historical_replay --output reports/routing_replay_mock.csv
```

The replay table includes context features, expert predictions, actual value,
expert errors, forecast horizon, event severity, missing-data conditions,
expert-call counts, cache hits, fallback/escalation flags, approximate compute
cost, and max input timestamps for leakage audits.

## Advisory Learned Router Training

The first supervised router training path lives in
`src/evaluation/router_training.py`. It audits replay rows before training:

- required replay columns must be present
- target time must be after forecast origin
- max Hohonu/NOAA input timestamps must be at or before the origin
- training features must not contain actuals, target-time labels, expert
  predictions, or error fields

Labels are derived after the forecast target is revealed: the best expert is
the successful expert with the smallest absolute error. Features come only from
origin-time `context_features` and `missing_data_conditions`. The first model is
a small `DecisionTreeClassifier` saved as an advisory artifact; it does not
replace the rule-based router.

Train from a replay CSV:

```bash
python -m scripts.train_router \
  --replay reports/routing_replay_mock.csv \
  --model-output reports/router_model.pkl \
  --report-output reports/router_training_report.json
```

Load the artifact for an advisory recommendation:

```python
from src.orchestration.learned_router import LearnedRouter

router = LearnedRouter.load("reports/router_model.pkl")
prediction = router.predict_from_features(
    {"horizon_minutes": 360, "recent_noaa_residual_m": 0.12},
    {"missing_latest_hohonu": False, "missing_tide_prediction": False},
)
print(prediction.recommended_expert)
```

Router training now uses a **forward-time (rolling-origin) split** — never a
random split, which would leak future information into a time series.
`evaluate_router_policies` reports oracle best-expert MAE, rule/cascade MAE,
learned-router MAE, routing regret versus the oracle, average expert calls, and
fallback / unavailable rates, with forward-time validation (or station-held-out
when multiple stations are present).

### Shadow mode

A trained `LearnedRouter` can be attached to the pipeline
(`ForecastPipeline(learned_router=...)`). It runs in **shadow mode**: it records
what it would have selected (`diagnostics["learned_router_shadow"]`) without ever
controlling the live route, which stays `rule_cascade`. It falls back to the rule
router when the model artifact is missing, the feature schema differs, inputs are
out of distribution, sample support is too small, predicted utilities are too
close, or the model fails.

Before using the learned router operationally, train it on reviewed historical
Hohonu/NOAA station pairs and compare it against the rule cascade on a
forward-in-time holdout.

## Benchmarks

`python -m scripts.benchmark_orchestration` compares the legacy flat router
against the adaptive cascade in one process on checked-in synthetic / mock
fixtures (no network), and writes `reports/orchestration_benchmark.json` and
`reports/orchestration_benchmark.md`. It covers the normal, short-horizon,
large-residual, QC-failure, stale-NOAA, missing-tide, expert-exception, and
storm-surge scenarios, plus replay throughput (legacy double-run vs exhaustive
reuse vs policy) and a parallel-escalation micro-benchmark.

## Current Limitations

- Live Hohonu endpoint details may need project-specific URL and payload
  mapping adjustments.
- Datum conversion is not implemented; mismatches fail closed.
- Weather-aware, spatial, and learned local residual experts are placeholders.
- Residual transfer scale is static configuration; the lag is now genuinely
  applied from observed history, but is not yet learned.
- The learned router is advisory and runs in shadow mode; the rule cascade
  remains the default production path. On the synthetic fixtures it does not beat
  the rule cascade (positive regret) — expected for tiny mock data.
- At cold start (no measured skill) the single-expert early-stop interval is
  tighter and under-covers; coverage recovers once a few validation samples
  accrue per regime cell. Benchmark fixtures use synthetic mock data where the
  local and regional signals differ more than a faithful station pair would.
- The verifier uses conservative physical-range and disagreement heuristics,
  not station-specific operational thresholds.
- This remains a research foundation, not an operational warning system.
