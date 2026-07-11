# Hohonu ML Pipeline - Deploy and Test

## 1) Core pipeline entrypoint

```bash
cd /Users/dylancablayan/Hohonu-1
python hohonu_driver_script.py <node_id> --strategy mix --candidate-profile auto --model-families ridge rf lstm pinn --metadata
```

### Strategy options
- `var` — baseline VAR
- `auto`, `auto-ml`, `ml` — best single candidate
- `ensemble` — inverse-RMSE weighted top-k blend
- `meta` — meta-stacked blend
- `mix` — explicit mix-and-match strategy (ridge/lasso/elastic/knn/svr/hgb/mlp/rf/gbr/extra/lstm/pinn families)

### Mix example

```bash
python hohonu_driver_script.py <node_id> \
  --strategy mix \
  --candidate-profile broad \
  --model-families ridge rf gbr lstm pinn \
  --ensemble-size 4 \
  --include-lstm \
  --include-pinn \
  --metadata
```

## 2) Combo benchmark sweep

```bash
python run_combo_benchmark.py <node_id> \
  --mode all \
  --candidate-profile broad \
  --model-families ridge lasso elastic knn svr hgb mlp rf gbr extra \
  --include-lstm \
  --include-pinn \
  --mix-size 4 \
  --json
```

This prints benchmark, single-best, meta, ensemble, and mix outputs.

### Rolling backtest + family health

```bash
python run_combo_benchmark.py <node_id> \
  --mode all \
  --rolling-backtest \
  --rolling-folds 6 \
  --rolling-window month \
  --family-health-sort-metric leaderboard_score \
  --qa-sort-metric qa_score \
  --json
```

This returns the rolling fold rows and family-level health summaries under `rolling_backtest`.

## 3) Interval scheduler for deployment-style runs

```bash
python pipeline_scheduler.py <node_id_1> <node_id_2> \
  --strategy mix \
  --candidate-profile auto \
  --model-families ridge rf gbr hgb \
  --steps 960 \
  --interval-seconds 3600 \
  --max-runs 24 \
  --mix-size 4 \
  --out-dir ./pipeline_runs
```

### Options to note
- `--once` runs one batch immediately and exits.
- `--nodes-file` reads node IDs from a file (one per line, commas accepted).
- `--out-dir` writes timestamped JSON files like `run_YYYYMMDDTHHMMSSZ_0001.json`.
- `manifest.csv` in the same directory is updated with per-node run metadata.

## 5) Documentation

- `README.md`
- `docs/PIPELINE_USAGE.md`
- `docs/ROLLING_BACKTEST.md`
- `docs/QA_METRICS.md`

## 4) API service (FastAPI)

1. Install API deps:

```bash
pip install fastapi uvicorn
```

2. Start server:

```bash
python api_server.py
```

3. Health check:

```bash
curl http://127.0.0.1:8000/health
```

4. Predict one node:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "node_name": "<node_id>",
    "strategy": "mix",
    "candidate_profile": "auto",
    "candidate_model_families": ["ridge","rf","lstm","pinn"],
    "include_lstm": true,
    "include_pinn": true,
    "candidate_mix_max_size": 4
  }'
```

5. Predict multiple nodes in one request:

```bash
curl -X POST http://127.0.0.1:8000/batch-predict \
  -H 'Content-Type: application/json' \
  -d '{
    "node_names": ["<node_id_1>","<node_id_2>"],
    "strategy": "ensemble",
    "candidate_profile": "compact",
    "candidate_model_families": ["ridge","rf","gbr"],
    "ensemble_size": 3
  }'
```
