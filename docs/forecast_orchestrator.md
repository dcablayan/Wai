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

## Routing

The first router is rule based. Example behavior:

- Fresh Hohonu data and a very short horizon selects `local_persistence`.
- Six-to-24-hour forecasts use `local_tide` plus `noaa_residual` when NOAA data
  are fresh.
- Large NOAA residuals select `noaa_residual` and
  `regional_to_local_residual`.
- Failed Hohonu QC excludes `local_persistence` and adds `safe_fallback`.
- Stale NOAA data excludes NOAA residual experts and favors local data plus
  tide.
- Strong model disagreement requests an additional expert and the verifier can
  widen intervals.

The output explanation includes the detected regime, selected experts, excluded
experts and reasons, combination method, fallback status, warnings, and
diagnostics.

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
observations as model inputs. At each origin it runs available experts, stores
expert predictions, then reveals the actual target value and records errors.

```bash
python -m scripts.run_historical_replay --output reports/routing_replay_mock.csv
```

The replay table includes context features, expert predictions, actual value,
expert errors, forecast horizon, event severity, missing-data conditions,
approximate compute cost, and max input timestamps for leakage audits. This is
the intended dataset for training a future learned router.

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

Before using this learned router operationally, train it on reviewed historical
Hohonu/NOAA station pairs and compare it against the rule-based router on a
forward-in-time holdout.

## Current Limitations

- Live Hohonu endpoint details may need project-specific URL and payload
  mapping adjustments.
- Datum conversion is not implemented; mismatches fail closed.
- Weather-aware, spatial, and learned local residual experts are placeholders.
- Residual transfer scale and lag are static configuration, not learned.
- The learned router is advisory and trained from replay rows; the rule-based
  router remains the default production path.
- The verifier uses conservative physical-range and disagreement heuristics,
  not station-specific operational thresholds.
- This remains a research foundation, not an operational warning system.
