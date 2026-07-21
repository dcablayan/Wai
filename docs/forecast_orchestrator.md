# Wai Forecast Orchestrator

Wai exposes three explicit orchestration modes:

- `ForecastPipeline(mode="mini")`: the default latency-oriented adaptive cascade.
- `ForecastPipeline(mode="ultra")`: Wai Ultra, a Fugu-inspired numerical forecast conductor with typed multi-turn coordination.
- `ForecastPipeline(mode="legacy")`: the original flat rule router retained for regression testing.

No LLM generates a numerical water-level forecast. Point forecasts and uncertainty intervals come from numerical experts, statistical synthesis, physical checks, or calibrated offline trajectory search. The conductor may choose, sequence, compare, synthesize, and verify those numerical components, but datum, QC, freshness, leakage, source, horizon, and fallback gates remain outside any learned policy.

## Data Flow

1. Hohonu and NOAA adapters canonicalize observations and predictions.
2. `ForecastContext` is built at a forecast origin using only origin-time data.
3. Mini, Ultra, or Legacy orchestration selects numerical work.
4. Safety and verification gates run before a forecast is user-facing.
5. `ForecastResult` reports forecast values, uncertainty, mode, topology, policy source, and compute accounting.

## Wai Mini

Mini uses the adaptive cascade in `src/orchestration/cascade.py`:

```
prepared/indexed data
  -> capability gate
  -> primary expert ranking
  -> Stage 1 cheap forecast
  -> post-forecast assessment
  -> early stop or conditional escalation
  -> skill-aware combination
  -> dependency-aware verification
  -> reserved safe fallback
```

The capability gate uses expert metadata from `src/experts/capabilities.py` to exclude impossible experts before execution. Ranking is cost-aware and skill-aware, using `SkillStore` rolling MAE, coverage, failure rate, latency, and support counts. Mini keeps the current fast operational path and remains the baseline for latency-sensitive forecasts.

## Wai Ultra

Wai Ultra runs a state-action loop:

1. Encode current context and transcript.
2. Apply authoritative safety and capability masks.
3. Select one feasible expert, role, subtask, and access list.
4. Execute the role-specific component.
5. Append the result to a transcript.
6. Add nodes and access/dependency edges to a workflow graph.
7. Re-encode state.
8. Stop on verifier acceptance, exhausted budget, or verified safe fallback.

Stable short-horizon cases can use `Worker -> Verifier`; elevated residual cases can use `Thinker -> Worker -> Worker -> Synthesis -> Verifier`. The topology is not a fixed primary/Stage-2 template.

## Coordination Protocol

The typed protocol lives in `src/orchestration/protocol.py`.

Roles:

- `THINKER`: numerical diagnostics only; never emits the public forecast.
- `WORKER`: produces or synthesizes numerical forecast candidates.
- `VERIFIER`: accepts, requests evidence, replans, rejects, abstains, or requires fallback.

Every action records turn id, expert id, role, subtask, parameters, access list, parallel group, expected cost, policy score, action probability, and audit rationale. Every message records visible prior turns, status, structured result, latency, and warnings. `WorkflowGraph` serializes nodes, dependency edges, access edges, parallel groups, and the accepted verifier node.

## Access Lists

Ultra does not pass every result to every expert. `UltraExecutor` enforces the action access list before execution.

- `ensemble_synthesis` receives only worker forecasts listed in its access list.
- `physics_datum_verifier` usually receives only the current candidate forecast.
- A verifier-requested worker may receive the verifier diagnosis without receiving unrelated diagnostics.

Access edges are serialized in `ForecastResult.executed_topology`.

## Role Contracts

Thinkers:

- `regime_difficulty_thinker`: estimates regime probabilities, forecast difficulty, event risk, data gaps, OOD score, expected value of more calls, and recommended next subtasks.
- `residual_dynamics_thinker`: analyzes NOAA residual trend, change-point signal, persistence, and regional-versus-local behavior.

Workers:

- Existing numerical experts are wrapped as Ultra workers: `local_persistence`, `local_tide`, `noaa_residual`, `regional_to_local_residual`, and `safe_fallback`.
- `ensemble_synthesis` consumes only allowed worker outputs and emits a point forecast, interval, confidence proxy, weights, disagreement diagnostics, and assumptions.
- `weather_aware`, `spatial_neighboring_station`, and `learned_local_residual` remain explicit unavailable placeholders.

Verifiers:

- `physics_datum_verifier`: units, intervals, plausible range, datum/QC compatibility, suspicious jumps, and source requirements.
- `cross_source_verifier`: local versus NOAA behavior, baseline consistency, and model disagreement.
- `calibration_verifier`: interval width, confidence support, and recent skill proxies.
- `event_risk_verifier`: stricter checks under elevated residuals or rapid changes.

Verifier acceptance is the normal Ultra termination signal.

## Policies

`BootstrapCoordinatorPolicy` is transparent and heuristic. It uses the same typed protocol, transcript, graph, safety masks, access lists, verifier termination, and fallback behavior as learned control, but reports `coordinator_policy_source="bootstrap"` and must not be described as trained.

`LearnedCoordinatorPolicy` loads a small NumPy coordination head. Artifacts include feature schema, action registry, weights, normalization data, training metadata, validation metrics, expert registry version, random seed, and training-data hash. Feature-schema or action-registry mismatches fail closed.

The older `LearnedRouter` remains an advisory best-expert model and is not used as the Ultra conductor.

## State Encoding

`src/orchestration/state_encoder.py` creates a deterministic fixed-size state vector with a versioned schema. It includes origin-time features such as horizon, tide phase, source availability, freshness, QC, local trend/volatility, NOAA residual magnitude/trend, weather proxies, station-pair scale/lag, skill estimates, failure/latency proxies, capabilities, and remaining budget.

Transcript features include roles used, experts used, worker predictions, interval widths, disagreement, thinker diagnoses, verifier verdicts, requested evidence, current candidate, remaining turns, and recursion depth.

It excludes future water level, target-time observations, future NOAA values unavailable at origin, error labels, oracle actions, and future event labels.

## Trajectories And Training

`src/evaluation/coordination_trajectories.py` extends replay rows into transition records with encoded state, action mask, selected action, role, expert, subtask, access list, result summary, immediate cost, terminal flag, final error, coverage, peak-event loss, total calls, latency, and final reward.

`src/evaluation/trajectory_search.py` performs bounded offline oracle search over valid workflows up to the turn limit. It considers single workers, independent worker pairs, synthesis, verification, and fallback using precomputed numerical outputs. Actual future level is revealed only for terminal reward.

`src/evaluation/coordinator_training.py` provides Stage 1 imitation warm start with forward-time splitting, deterministic seeds, event counts, randomized worker-pool conditions, and schema-tagged artifacts. `src/evaluation/cma_es.py` provides optional NumPy-only separable CMA-ES for Stage 2 trajectory-level reward optimization under a strict evaluation budget.

Randomized pool conditions include unavailable Hohonu/NOAA, stale local/regional data, failed QC, missing tide prediction, unavailable weather/spatial experts, disabled numerical model, expert exception, slow expert, and invalid interval.

## Safety And Fallback

Mini and Ultra both keep safety gates outside learned policy. Ultra masks actions in `src/orchestration/action_masks.py` before policy scoring:

- datum/source availability
- QC status
- freshness
- horizon support
- required-source checks
- candidate availability for verifiers
- distinct numerical expert budget
- remaining deadline
- fallback-attempt limits

If no valid action remains, Ultra attempts `safe_fallback` once and independently verifies it. It returns unavailable rather than inventing a forecast.

## Reporting

Every `ForecastResult` now reports:

- `mode`
- `coordinator_policy_source`
- `coordinator_artifact_version`
- `number_of_turns`
- `number_of_unique_experts`
- `role_sequence`
- `executed_topology`
- `termination_reason`
- logical actions, physical expert calls, reused outputs, verifier calls, thinker calls, worker calls, and fallback calls

Mini diagnostics also include cascade trace timing, cache hits, early-stop/escalation reasons, route source, fallback reason, and execution-budget usage.

## Benchmarks

Run the existing model benchmark:

```bash
uv run python -m scripts.run_benchmark
```

Run the upstream orchestration benchmark:

```bash
uv run python -m scripts.benchmark_orchestration
```

Run the local Ultra smoke benchmark:

```bash
uv run python -m scripts.run_ultra_benchmark
```

The Ultra smoke benchmark writes `reports/ultra_benchmark_results.md` and `.json`. It uses bundled mock data and is useful for checking orchestration and compute reporting; it is not real-world validation. Full ablations require a historical trajectory dataset and should compare no-thinker, no-verifier, no-transcript, no-access-control, no-recursive-replanning, fixed-topology, no-randomized-pool-training, Mini, Ultra bootstrap, Ultra learned, and oracle workflow upper bound.

## Current Limitations

- Live Hohonu endpoint details may need project-specific URL and payload mapping.
- Datum conversion is supported only through reviewed, per-station offsets in
  `src.data.datum`; unknown conversions and unresolved mismatches fail closed.
- Weather-aware, spatial, and learned local residual experts are placeholders.
- Residual transfer scale and lag are static station-pair metadata.
- The checked-in learned router is still a single-label advisory model and is not validated for live Ultra control.
- Replay-trained coordination artifacts with non-live feature schemas remain shadow-only until trained and validated against the live `StateEncoder` schema.
- The local Ultra benchmark is mock-data evidence only. Real validation requires reviewed historical Hohonu/NOAA station pairs, forward-time splits, station-held-out evaluation, event stratification, and randomized expert-pool evaluation.
