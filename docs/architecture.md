# Wai Architecture

## Pipeline Overview

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
   │  add_tidal_harmonics()  — sin/cos for M2, S2, K1, O1, N2 constituents
   │  add_lag_features()     — lagged water-level observations
   │  add_rolling_features() — rolling mean + std at 1hr / 4hr / 24hr windows
   │  build_feature_matrix() — composes X, y dropping NaN rows
   ▼
Forecasting (src/models/)
   │  PersistenceModel       — last-value naive baseline
   │  HarmonicRidgeModel     — harmonic regression with Ridge regularisation
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

## Directory Layout

```
Wai/
├── src/
│   ├── data/
│   │   ├── loader.py        Data ingestion (demo + NOAA API)
│   │   └── validation.py    Schema and quality checks
│   ├── features/
│   │   └── engineering.py   Tidal harmonics, lags, rolling windows
│   ├── models/
│   │   ├── baseline.py      PersistenceModel + HarmonicRidgeModel
│   │   └── metrics.py       MAE / RMSE / R² / NSE / corr
│   └── reporting/
│       └── report.py        HTML report generation
├── scripts/
│   ├── prepare_demo_data.py Generate synthetic demo CSV
│   ├── train_baseline.py    Train + evaluate models, save metrics
│   └── generate_report.py  Build HTML reports from data + metrics
├── app.py                   Streamlit dashboard entry point
├── data/demo/               Synthetic demo data (safe to commit)
├── reports/                 Generated metrics JSON + HTML reports
├── tests/                   pytest test suite
└── docs/                    Architecture + modeling documentation
```

## Data Schema

All DataFrames in Wai carry these columns:

| Column      | Type                    | Description                            |
|-------------|-------------------------|----------------------------------------|
| timestamp   | datetime64[ns, UTC]     | Observation time, UTC-aware            |
| station_id  | str                     | Station identifier                     |
| water_level | float                   | Water height in meters                 |
| datum       | str                     | Tidal datum (MLLW, NAVD88, MSL, …)    |
| units       | str                     | "m" or "ft"                            |
| lat         | float                   | Station latitude (decimal degrees)     |
| lon         | float                   | Station longitude (decimal degrees)    |
| source      | str                     | DEMO_SYNTHETIC or NOAA_COOPS           |

## Key Design Decisions

**Single schema contract** — every loader returns the same 8-column DataFrame,
so validation, feature engineering, and reporting never need to branch on source.

**Separation of ingestion and modeling** — `src/data/` handles I/O; `src/models/`
has no file I/O. Scripts in `scripts/` glue them together with a clear
data-flow direction.

**Reproducible demo data** — `scripts/prepare_demo_data.py` is deterministic
(fixed seed) and generates a CSV that can be committed safely. No private data,
no API keys.

**Extensibility** — `HarmonicRidgeModel` uses a scikit-learn `Pipeline`, so
swapping the Ridge estimator for an LSTM or Transformer requires only replacing
the estimator object. `build_feature_matrix()` produces the same feature tensor
regardless of downstream model.
