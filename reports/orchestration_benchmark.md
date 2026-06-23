# Wai Orchestration Benchmark

Legacy flat router vs adaptive cascade on checked-in synthetic fixtures.
No network access required.

## Per-scenario latency and expert calls

| Scenario | Path | p50 ms | p95 ms | mean calls | early-stop | escalation | experts |
|---|---|---|---|---|---|---|---|
| normal_stable_tide | legacy | 0.0381 | 0.0493 | 2 | 0.0 | 0.0 | local_tide,noaa_residual |
| normal_stable_tide | adaptive | 0.0389 | 0.0457 | 1 | 1.0 | 0.0 | noaa_residual |
| short_horizon_fresh_local | legacy | 0.0152 | 0.0194 | 1 | 0.0 | 0.0 | local_persistence |
| short_horizon_fresh_local | adaptive | 0.0393 | 0.042 | 1 | 1.0 | 0.0 | local_persistence |
| large_noaa_residual | legacy | 0.0703 | 0.0842 | 2 | 0.0 | 0.0 | noaa_residual,regional_to_local_residual |
| large_noaa_residual | adaptive | 0.1126 | 0.1321 | 2 | 0.0 | 1.0 | noaa_residual,regional_to_local_residual |
| local_qc_failure | legacy | 0.0205 | 0.0219 | 2 | 0.0 | 0.0 | local_tide |
| local_qc_failure | adaptive | 0.0338 | 0.0407 | 1 | 1.0 | 0.0 | local_tide |
| stale_noaa_input | legacy | 0.0152 | 0.0168 | 1 | 0.0 | 0.0 | local_persistence |
| stale_noaa_input | adaptive | 0.036 | 0.0435 | 1 | 1.0 | 0.0 | local_persistence |
| missing_tide_prediction | legacy | 0.0058 | 0.006 | 2 | 0.0 | 0.0 |  |
| missing_tide_prediction | adaptive | 0.0347 | 0.0398 | 1 | 0.0 | 1.0 | local_persistence |
| expert_exception | adaptive | - | - | - | - | - | ['noaa_residual'] |

## Historical replay throughput

| Mode | rows | seconds | rows/s | mean calls |
|---|---|---|---|---|
| legacy double-run | 182 | 0.7997 | 227.58 | 5 (all) |
| new exhaustive (reuse) | 182 | 0.0956 | 1903.51 | 0.0 |
| new policy | 182 | 0.0881 | 2064.72 | 1.0 |

- Exhaustive throughput speedup vs legacy: **8.36x**
- Policy throughput speedup vs legacy: **9.07x**

## Policy-mode forecast quality

- MAE: 0.1895 m
- Peak-event MAE: None m
- Interval coverage: 0.6319
- Unavailable rate: 0.0
- Fallback rate: 0.0
- Escalation rate: 0.0

## Forecast quality before/after (same data, routing differs)

| Metric | legacy flat | adaptive (cold skill) | adaptive (warm skill) |
|---|---|---|---|
| MAE (m) | 0.1895 | 0.1895 | 0.1546 |
| Interval coverage | 0.7308 | 0.6319 | 0.9451 |
| Unavailable rate | 0.0 | 0.0 | 0.0 |
| Fallback rate | 0.0 | 0.0 | 0.0 |

## Storm-surge fixture (peak-event)

- MAE: 0.1778 m
- Peak-event MAE: 0.4067 m
- Interval coverage: 0.7582
- Escalation rate: 0.1813
- Fallback rate: 0.0

## Router policy comparison (forward-validated)

- Validation: forward_time (n_test=46)
- Oracle best-expert MAE: 0.11626900923706483
- Rule/cascade MAE: 0.18019762436194253
- Learned-router MAE: 0.16059428478081342
- Routing regret vs oracle: 0.044325275543748585

## Parallel escalation micro-benchmark

- Two slow experts sequential: 109.97 ms
- Two slow experts parallel: 55.1 ms
- Speedup: 2.0x
