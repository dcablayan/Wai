# Hohonu-1 Tide Forecasting Pipeline

End-to-end tide forecasting for Hohonu nodes with:

- candidate model benchmarking
- single-best / ensemble / meta / mix forecasting strategies
- rolling backtesting and family-level stability scoring
- scheduled/cli execution
- optional API server entrypoints

This repo combines tide observations + NOAA context and evaluates time-series models
(including optional LSTM and PINN families when available).

## Quick project layout

| Path | Purpose |
| --- | --- |
| `hohonu_driver_script.py` | Core pipeline entrypoint for one node (`run_pipeline`). |
| `run_combo_benchmark.py` | CLI tool for benchmarking candidate model families and combos. |
| `tide_ml_engine.py` | Candidate generation, scoring, forecasting, and QA/rolling metrics logic. |
| `pipeline_scheduler.py` | Interval-based batch runner for multiple nodes. |
| `api_server.py` | FastAPI service wrappers for single/batch predictions. |
| `load_hohonu_devices.py` | Builds/updates `data/` station metadata from `nodes*.json`. |
| `noaa_stations.py` | Local NOAA file lookup helpers. |
| `noaa_datum_converter.py` | NOAA datum conversion helpers. |
| `VAR_prediction.py` | VAR-based baseline utilities used in candidate search paths. |
| `DEPLOYMENT.md` | Existing deployment/test command examples. |

## Requirements

Minimum:

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`
- `statsmodels`
- `timezonefinder`

Optional ML extras:

- `tensorflow` (for LSTM candidates)
- `torch` (for PINN candidates)
- `fastapi`, `uvicorn` (for API server)

Core pipeline code has local fallbacks that still run with the core requirements
when these optional packages are missing.

## Installation

```bash
cd /Users/dylancablayan/Hohonu-1
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn statsmodels timezonefinder
```

Optional extras:

```bash
pip install tensorflow torch fastapi uvicorn
```

## Data prerequisites

The code expects local files under `./data` (or `../data`) depending on runtime:

- `nodes.json`
- `nodes_standardized.json`
- `noaa_stations.tsv`
- per-node local observation files (used by fallbacks, e.g. `./data/<node>.csv|.tsv`)
- per-station NOAA data files (as supported by `noaa_stations.py`)

If these files are absent, behavior depends on available optional modules:

- With full `dataproc` stack, remote fallbacks may handle broader workflows.
- With local-only fallback stack, files above are used directly.

## Running the pipeline for one node

```bash
python hohonu_driver_script.py <node_id> --strategy auto --candidate-profile auto --metadata
```

Useful strategies:

- `var` (legacy VAR baseline)
- `auto` / `auto-ml` / `ml` (single best candidate)
- `ensemble` (rank-weighted top-k blend)
- `meta` (meta-stacked blend)
- `mix` (explicit candidate family mix)

Common switches:

- `--include-lstm` / `--include-pinn`
- `--candidate-profile compact|auto|broad`
- `--model-families ridge rf lstm pinn ...`
- `--digital-twin` / `--no-digital-twin`
- `--metadata` (print JSON-like metadata block)

## Benchmarking and model-portfolio exploration

Run full combo benchmark with leaderboard + single/meta/ensemble/mix summaries:

```bash
python run_combo_benchmark.py <node_id> --mode all --candidate-profile broad --json
```

Run with rolling stability analysis + family health:

```bash
python run_combo_benchmark.py <node_id> \
  --mode all \
  --rolling-backtest \
  --rolling-folds 6 \
  --rolling-window month \
  --holdout-steps 720 \
  --family-health-sort-metric leaderboard_score \
  --json
```

This returns:

- `benchmark`: one-shot holdout leaderboard
- `rolling_backtest.rows`: fold-level rows with RMSE/MAE/NSE/QA metrics
- `rolling_backtest.family_health`: family-level stability summaries and ranked composite score

## Scheduled execution

```bash
python pipeline_scheduler.py <node_a> <node_b> \
  --strategy mix \
  --candidate-profile auto \
  --interval-seconds 3600 \
  --once \
  --out-dir ./pipeline_runs
```

Outputs:

- `pipeline_runs/run_<timestamp>_<n>.json` for each cycle
- `pipeline_runs/manifest.csv` with compact per-node metrics

## API server

```bash
python api_server.py
```

Default endpoints:

- `GET /health`
- `POST /predict`
- `POST /batch-predict`

Use the request schema from `api_server.py` (`PredictRequest`) to set strategy,
steps, families, and model preferences.

## QA and stability scoring

See:
- `docs/ROLLING_BACKTEST.md` (rolling backtest + family health)
- `docs/QA_METRICS.md` (QA metric definitions and composite weighting)

## Uploading to GitHub

Before uploading:

1. Verify all sensitive paths/credentials are removed from local run artifacts.
2. Add any external `data/` assets you want excluded via `.gitignore`.
3. Commit:

```bash
git add .
git commit -m "Add rolling backtest health, combo benchmarking, and pipeline docs"
```

Suggested `.gitignore` entries (if not already present):

- `data/`
- `pipeline_runs/`
- `.venv/`
- `__pycache__/`
- `*.pyc`

