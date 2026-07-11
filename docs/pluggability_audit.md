# Pluggability Audit: Plugging Any Tide-Gauge System into Wai

Goal: Wai should accept observations from **any** tide-gauge provider or
arbitrary input data (CSV export, vendor API, met covariates) and produce a
forecast, without editing core orchestration modules.

This audit maps what blocks that goal today and orders the fixes.

## Current state summary

The canonical observation schema (`src/data/canonicalize.py`) is the one clean,
source-agnostic seam in the codebase. Everything on either side of it is
coupled to exactly two providers (Hohonu, NOAA CO-OPS) and two demo stations.

## Findings

### F1. Three competing schemas, no bridge (critical)

- **Schema A** ("canonical", `src/data/canonicalize.py`): `timestamp_utc`,
  `water_level_m`, `latitude`/`longitude`, `record_type`, `qc_status`,
  `qc_flags`, `retrieved_at`, `latency_seconds`. Used by `HohonuAdapter` and
  `NOAACoopsAdapter`.
- **Schema B** ("Wai/legacy", `src/data/loader.py`, `src/data/validation.py`):
  `timestamp`, `water_level`, `lat`/`lon`, `units`. Used by the demo loader,
  feature engineering, tabular models, validation, and the dashboard.
- **Schema C** (`Hohonu-1/`): indexed single-column frames plus a device table.
- `validation.validate()` only understands Schema B; canonical frames cannot be
  validated. Feature engineering only understands Schema B; canonical frames
  cannot be fed to models without ad-hoc renames.

### F2. No data-source abstraction or registry (critical)

- `HohonuAdapter` and `NOAACoopsAdapter` are unrelated classes with different
  method signatures (`start_time/end_time` vs `begin/end`). No ABC/Protocol,
  no registry, no factory, no config-driven selection.
- A new provider means writing a bespoke class and hand-wiring it; nothing in
  the pipeline would discover it.
- There is no generic "load observations from a CSV/DataFrame" source, so
  arbitrary gauge exports cannot enter the system at all.

### F3. Fixed 6-minute cadence assumed silently (critical correctness risk)

- Lag/rolling features operate on **row position**, not time
  (`src/features/engineering.py`): default lags `[1,2,4,10,20,40]` and windows
  `[10,40,240]` are documented as minutes-at-6-min-cadence. A 1-min or 15-min
  gauge silently produces mis-scaled features with no warning.
- No resampling/regularization stage exists anywhere in `src/data/`; adapters
  just concat and sort. Gap handling is drop-only (`dropna()`), shrinking
  training data silently.
- `metrics.compute_episode_metrics` defaults `step_seconds=360`.

### F4. Orchestration hardwired to "hohonu"/"noaa" (high)

- `ForecastContext` fields are provider-named (`latest_hohonu_observation`,
  `noaa_qc_ok`, freshness keyed by literal `"hohonu"`/`"noaa"`).
- The router, adaptive cascade capability gate, verifier dependency map, and
  `ExpertSpec` (`requires_local_obs`/`requires_noaa_obs` booleans) all repeat
  the same two literals. Adding a third source touches ~5 core modules.
- The concepts are really **roles** — a *local* gauge and a *regional
  reference* — accidentally named after the first two vendors.

### F5. No station metadata model or catalog (high)

- Station config is a 2-entry hardcoded dict (`DEFAULT_STATION_PAIRS`) mapping
  local ids to **NOAA-specific** reference ids; unknown stations raise
  `KeyError`. A disconnected 5-entry list lives in
  `scripts/evaluate_noaa_public.py`.
- No catalog with lat/lon, native cadence, datum, units, timezone, or provider;
  no config file or CLI to register a station.

### F6. No datum conversion; mixed datums fail closed (high)

- `assert_compatible_datums` raises on any mix; no vertical conversion exists
  in `src/`. A NAVD88 gauge cannot pair with an MLLW NOAA reference.
- Ironically `Hohonu-1/noaa_datum_converter.py` (legacy silo) implements
  per-station scalar datum offsets — the exact missing feature.

### F7. Quality control is thin (medium)

- Gap detection is advisory-only (validation report); nothing fills or flags
  gaps on the model path. No spike/despike logic in `src/` (the legacy silo
  has a MAD despiker). Range check is a static global `(-15, 30)` m, not
  per-station.

### F8. Experts degrade to persistence-only without NOAA products (medium)

- All experts except `LocalPersistenceExpert` require a tide-prediction
  record; `SafeFallbackExpert` also requires tide. A gauge with no NOAA
  counterpart gets persistence quality only — even though the **internal
  harmonic features** (`add_tidal_harmonics`, 8 constituents, no NOAA
  dependency) plus Ridge/GBM are fully source-agnostic. The bridge expert
  (`LearnedLocalResidualExpert`) is an unimplemented placeholder.

### F9. Uncertainty not wired through (medium)

- Expert intervals are hand-tuned heuristics; split-conformal
  (`src/models/conformal.py`) is never used by the pipeline. Conformal qhat is
  unit/datum sensitive and needs ≥~50 calibration samples.

### F10. Config/docs drift (low)

- `.env.example` documents dead `TIDEFORMER_*` vars and omits the real ones
  (`HOHONU_API_KEY`, `NOAA_OFFLINE`). No live-Hohonu path is exercised by any
  script. No "onboard a new gauge" guide. `Hohonu-1/` duplicates ingestion,
  QC, resampling, and datum logic with zero imports across the boundary.

## Improvement plan (ordered)

| # | Work item | Fixes | Status |
|---|---|---|---|
| 1 | Schema bridge: `to_canonical()`/`to_model_frame()` converters, single vocabulary source of truth | F1 | done |
| 2 | `DataSource` protocol + registry + generic `CSVSource`/`DataFrameSource`; register Hohonu/NOAA adapters | F2 | done |
| 3 | Regularization stage: cadence inference, resample-to-grid, MAD despike, gap policy; cadence-aware lag/window helpers | F3, F7 | done |
| 4 | Station catalog: `StationMetadata` + JSON catalog + generic reference pairing (supersedes `StationPair`, back-compat kept) | F5 | done |
| 5 | Datum conversion: per-station offsets in catalog, `convert_datum()`, relax fail-closed gate when offsets known | F6 | done |
| 6 | Role-based context: `local`/`regional` roles with provider aliases so any source fills the context | F4 | done |
| 7 | End-to-end CLI: forecast from any CSV/registered source (`scripts/run_gauge_forecast.py`) | F2, F5 | done |
| 8 | Harmonic-fallback expert so no-NOAA gauges beat persistence | F8 | done |
| 9 | Conformal intervals wired to the pipeline as an option | F9 | future |
| 10 | Docs: gauge onboarding guide; fix `.env.example` | F10 | done |
