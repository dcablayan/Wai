# Metrics Interpretation Guide

Wai reports several metric families. They are intentionally separated by data
source because each source answers a different question.

## Synthetic Demo Metrics

Files:

- `reports/model_metrics.json`
- `reports/horizon_metrics.json`
- `reports/event_metrics.json`
- `reports/rolling_origin_metrics.json`
- `reports/conformal_metrics.json`
- `reports/ablation_metrics.json`

These use `data/demo/demo_water_levels.csv`, which is generated from known
tidal components plus synthetic noise and synthetic events. High R2 on this
data shows that the pipeline can recover a known constructed signal. It does
not show operational forecasting skill.

Use `reports/summary.json -> synthetic -> ablation_claims` for ablation
takeaways. Do not hardcode claims such as "harmonics-only reaches 98% R2 on
all stations"; the generated ablation metrics determine whether that is true.

## Tidecast Benchmark Metrics

Files:

- `reports/benchmark_results.md`
- `reports/summary.json -> tidecast`

These use NOAA-derived tidal predictions in `data/demo/tidecast/*.csv`.
They are smooth deterministic harmonic predictions, not raw gauge
observations. RMSE is in the tidecast file units.

The benchmark now includes a real `Persistence (last value)` comparator. Any
claim that `TinyTidePrototype` beats persistence must be checked against the
current `reports/benchmark_results.md` table or the recomputed averages in
`reports/summary.json -> tidecast -> benchmark`. Do not state this as a fixed
property of the model.

Prototype capability wording:

- `WaveGRUPrototype` is smoothing, not a real GRU.
- `SurgeNetPrototype` is a residual heuristic, not meteorological surge
  modeling.
- `TsunamiSentinelPrototype` is an anomaly toy, not a validated tsunami
  detector.

## NOAA Mock Metrics

Files:

- `reports/noaa_mock_metrics.json`
- `reports/noaa_mock_metrics.md`
- `reports/summary.json -> noaa_mock`

These are offline fixtures for CI and reproducibility. They prove that NOAA
merge, baseline, skill-score, and report code runs without network access.
They are not real NOAA performance.

## NOAA Live Metrics

Files:

- `reports/noaa_live_metrics.json`
- `reports/noaa_live_metrics.md`
- `reports/summary.json -> noaa_live`

Live evaluation fetches NOAA observations and NOAA tidal predictions, checks
datum and units, then merges by timestamp. It reports:

- rolling persistence
- NOAA prediction
- NOAA residual persistence
- HarmonicRidge
- GradBoost
- hybrid residual Ridge

Skill scores are error reductions against rolling persistence and NOAA
prediction. Positive skill means the model reduced MAE or RMSE relative to the
baseline; negative skill means it was worse.

Live reports refuse to write if any record has `mock_used=true`. Explicit
mixed fallback runs write `noaa_allow_mock_metrics.*` instead.

## Reading Conformal Coverage

Conformal reports include:

- nominal coverage
- empirical test coverage
- event coverage
- non-event coverage
- qhat
- calibration size
- mean interval width

The nominal value is the requested target. The empirical values are what
actually happened on the future test window and should be cited alongside the
nominal value.
