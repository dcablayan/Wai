# Wai Ultra Local Smoke Benchmark

This benchmark uses bundled mock observations and tide predictions. It checks orchestration behavior and reporting; it is not real-world forecast validation.

## Mode Summary

| mode | overall_mae | interval_coverage | interval_width | unavailable_rate | fallback_rate | p50_latency_ms | p95_latency_ms | logical_actions | physical_expert_calls | unique_experts | average_turns | reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | 0.2809 | 0.3333 | 0.4327 | 0.0000 | 0.0000 | 0.0972 | 0.1380 | 2.6667 | 1.6667 | 2.6667 | 2.6667 | -0.6676 |
| mini | 0.2809 | 0.3333 | 0.4474 | 0.0000 | 0.0000 | 0.1037 | 0.1814 | 2.3333 | 1.3333 | 2.3333 | 2.3333 | -0.6609 |
| ultra_bootstrap_1turn | nan | nan | nan | 1.0000 | 0.0000 | 0.5269 | 0.6368 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | -2.0200 |
| ultra_bootstrap_2turn | 0.3214 | 0.0000 | 0.1781 | 0.6667 | 0.0000 | 0.5821 | 0.5934 | 2.0000 | 1.0000 | 2.0000 | 2.0000 | -1.6472 |
| ultra_bootstrap_3turn | 0.1710 | 0.5000 | 0.2091 | 0.3333 | 0.0000 | 0.7010 | 0.7170 | 2.6667 | 1.3333 | 2.6667 | 2.6667 | -1.0007 |
| ultra_bootstrap_5turn | 0.2542 | 0.3333 | 0.3394 | 0.0000 | 0.0000 | 0.7022 | 1.1443 | 3.3333 | 1.3333 | 3.3333 | 3.3333 | -0.6543 |

## Oracle Workflow Upper Bound

| workflow | loss | reward | calls |
| --- | ---: | ---: | ---: |
| single:noaa_residual | 0.0579 | -0.0579 | 2 |
| single:local_tide | 0.0620 | -0.0620 | 2 |
| single:local_tide | 0.0620 | -0.0620 | 2 |

## Ablations

Ablations are defined in code but not run here because they require a full historical trajectory dataset.
