# Wai Architecture

## Pipeline Overview

Two parallel data paths feed into modeling and reporting.

### Path A — Synthetic demo data (dashboard + HTML reports)

```
Raw Data
   │
   ▼
Ingestion (src/data/loader.py)
   │  load_demo_data()       — reads data/demo/demo_water_levels.csv
   │  load_noaa_data()       — fetches from NOAA CO-OPS public API
   ▼
Validation (src/data/validation.py)
   │  validate()             — checks NaNs, gaps, duplicates, out-of-range, timezone
   ▼
Feature Engineering (src/features/engineering.py)
   │  add_tidal_harmonics()  — sin/cos for 8 constituents (M2, S2, K1, O1, N2, M4, M6, Mm)
   │  add_temporal_covariates() — hour-of-day, lunar-phase sin/cos
   │  add_lag_features()     — lagged water-level observations
   │  add_rolling_features() — rolling mean + std at 1hr / 4hr / 24hr windows
   │  build_feature_matrix() — composes X, y dropping NaN rows
   ▼
Forecasting (src/models/baseline.py)
   │  PersistenceModel       — last-value naive baseline
   │  HarmonicRidgeModel     — harmonic regression with Ridge regularisation
   │  WaveGRUModel           — DataFrame adapter for WaveGRUPrototype
   │  compute_metrics()      — MAE, RMSE, R², NSE, Pearson correlation
   │  save_metrics()         — writes reports/model_metrics.json
   ▼
Visualization (app.py)
   │  Streamlit dashboard    — time series, forecast overlay, anomaly markers, map
   ▼
Reporting (src/reporting/report.py)
   │  generate_report()      — produces per-station HTML report
   ▼
Output artifacts
   ├─ reports/model_metrics.json
   └─ reports/report_<station_id>.html
```

### Path B — Tidecast reference data (prototype benchmarks)

```
Tidecast CSVs (data/demo/tidecast/hohonu-*_tidecast.csv)
   │  NOAA-derived harmonic tidal predictions, not raw sensor readings
   ▼
Windowing (src/data/windowing.py)
   │  load_tidecast_series() — parses dt + prediction columns → (times, values)
   │  make_windows()         — sliding lookback/target window dicts
   │  temporal_split()       — train / val / test split by time order
   ▼
Prototype Benchmarks (src/models/prototypes.py)
   │  TinyTidePrototype       — two-layer MLP-style with skip connection
   │  HarmonicNetPrototype    — 8-constituent harmonic projection
   │  WaveGRUPrototype        — bidirectional double-exp smoothing + attention
   │  SurgeNetPrototype       — dual-head tide + surge residual estimator
   │  TsunamiSentinelPrototype — multi-scale anomaly detector
   ▼
Benchmark Script (scripts/run_benchmark.py)
   │  benchmark_station()     — fits + evaluates all models per station
   ▼
Output
   └─ reports/benchmark_results.md
```

## Directory Layout

```
Wai/
├── src/
│   ├── data/
│   │   ├── loader.py        Data ingestion (demo + NOAA API)
│   │   ├── validation.py    Schema and quality checks
│   │   └── windowing.py     Sliding-window builder + tidecast CSV loader
│   ├── features/
│   │   └── engineering.py   8-constituent tidal harmonics, temporal covariates, lags, rolling windows
│   ├── models/
│   │   ├── baseline.py      PersistenceModel + HarmonicRidgeModel + WaveGRUModel adapter
│   │   ├── metrics.py       MAE / RMSE / R² / NSE / corr
│   │   └── prototypes.py    Pure-Python prototype benchmark models
│   └── reporting/
│       └── report.py        HTML report generation
├── scripts/
│   ├── prepare_demo_data.py Generate synthetic demo CSV
│   ├── train_baseline.py    Train + evaluate pipeline models, save metrics
│   ├── generate_report.py   Build HTML reports from data + metrics
│   └── run_benchmark.py     Benchmark prototype models on tidecast data
├── app.py                   Streamlit dashboard entry point
├── data/demo/               Synthetic data + NOAA-derived tidecast predictions
│   └── tidecast/            hohonu-*_tidecast.csv (NOAA harmonic predictions)
├── reports/                 Generated metrics JSON, benchmark markdown, HTML reports
├── tests/                   pytest test suite
└── docs/                    Architecture, modeling, and metrics interpretation docs
```

## Data Schema

All DataFrames in Wai's main pipeline carry these columns:

| Column      | Type                    | Description                            |
|-------------|-------------------------|----------------------------------------|
| timestamp   | datetime64[ns, UTC]     | Observation time, UTC-aware            |
| station_id  | str                     | Station identifier                     |
| water_level | float                   | Water height (meters for demo/NOAA; feet for tidecast) |
| datum       | str                     | Tidal datum (MLLW, NAVD88, MSL, …)    |
| units       | str                     | "m" or "ft"                            |
| lat         | float                   | Station latitude (decimal degrees; NaN for tidecast) |
| lon         | float                   | Station longitude (decimal degrees; NaN for tidecast) |
| source      | str                     | DEMO_SYNTHETIC, NOAA_COOPS, or TIDECAST |

Prototype models consume **window dicts** instead of DataFrames:

```python
window = {
    "values"      : List[float],  # lookback observations
    "times"       : List[float],  # fractional hours from series start
    "target_value": float,        # next-step ground truth
    "target_time" : float,        # fractional hour of prediction target
}
```

Use `src/data/windowing.make_windows()` to build these from a raw series.

## Key Design Decisions

**Single schema contract** — every loader returns the same 8-column DataFrame,
so validation, feature engineering, and reporting never need to branch on source.

**Separation of ingestion and modeling** — `src/data/` handles I/O; `src/models/`
has no file I/O. Scripts in `scripts/` glue them together with a clear
data-flow direction.

**Two model tiers** — pipeline models (baseline.py) fit the DataFrame API and
are evaluated on the synthetic demo data; prototype models (prototypes.py) use
the window-dict API and are evaluated on tidecast data. Keeping them separate
avoids mixing synthetic and NOAA-derived signal in the same evaluation loop.

**Reproducible demo data** — `scripts/prepare_demo_data.py` is deterministic
(fixed seed) and generates a CSV that can be committed safely. No private data,
no API keys.

**Extensibility** — `HarmonicRidgeModel` uses a scikit-learn `Pipeline`, so
swapping the Ridge estimator for an LSTM or Transformer requires only replacing
the estimator object. `build_feature_matrix()` produces the same feature tensor
regardless of downstream model.
