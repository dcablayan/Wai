# Pipeline usage and runbook

This runbook summarizes what each top-level script does and the most common invocation patterns.

## 1. `hohonu_driver_script.py`

Primary one-node entrypoint.

```bash
uv run python hohonu_driver_script.py <node_id> [flags]
```

Key arguments:

- `--strategy {var,auto,auto-ml,ml,ensemble,meta,mix}`
- `--steps` number of 6-minute steps to forecast (default 960)
- `--candidate-profile {compact,auto,broad}`
- `--model-families` (space/comma separated list)
- `--ensemble-size`
- `--mix-size`
- `--meta-top-k`
- `--meta-holdout-ratio`
- `--include-lstm`
- `--include-pinn`
- `--metadata`

Notes:

- `--digital-twin` and `--no-digital-twin` control NOAA-based forcing behavior.
- `--metadata` prints a JSON-like metadata payload with selected candidate and scores.

## 2. `run_combo_benchmark.py`

Benchmark + strategy sweep and optional rolling stability reporting.

```bash
uv run python run_combo_benchmark.py <node_id> [flags]
```

Core modes:

- `--mode all` runs all strategy lanes (`single`, `ensemble`, `meta`, `mix`)
- `--mode best|ensemble|meta|mix` for targeted runs

Rolling backtest options (new):

- `--rolling-backtest`: enable rolling multi-fold stability benchmark
- `--rolling-folds`: number of fold start points
- `--rolling-window {auto,month,quarter}`
- `--holdout-steps`: fixed holdout size (overrides heuristic)
- `--family-health-sort-metric`: ordering for family-health summary

QA options:

- `--qa-sort-metric` for single-row leaderboard
- `--family-health-sort-metric` for family-health summary
- `--qa-weights "rmse:0.4,mae:0.2,nse:0.2,r2:0.1"` customizes composite scoring
- `--json` emits a full JSON payload for downstream pipelines

Output sections:

- `benchmark`: one-shot holdout metrics per model candidate
- `rolling_backtest.rows`: per-fold candidate rows
- `rolling_backtest.family_health`: per-family roll-up with mean/volatility + leaderboard score

## 3. `pipeline_scheduler.py`

Runs multiple nodes repeatedly on schedule.

```bash
uv run python pipeline_scheduler.py <node_id_1> <node_id_2> ... [flags]
```

Common flags:

- `--once`: single run and exit
- `--nodes-file` load node ids from file
- `--interval-seconds` controls frequency
- `--max-runs` stop after N cycles
- `--out-dir` output artifacts directory
- strategy/profile/family arguments are passed through to the pipeline

Output:

- JSON run artifact: `run_<timestamp>_<counter>.json`
- CSV summary: `manifest.csv`

## 4. `api_server.py`

FastAPI wrapper around `run_pipeline`.

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /batch-predict`

Starts with:

```bash
WAI_API_KEY='replace-with-a-secret' uv run python Hohonu-1/api_server.py
```

Send `X-API-Key` on prediction routes. HTTP requests are bounded to compact
non-neural `var`, `auto`, and `ensemble` strategies; broad searches, `meta`,
`mix`, LSTM, and PINN are offline-only.

## 5. Data utility helpers

### `load_hohonu_devices.py`

Builds/refreshes Hohonu station metadata:

```bash
uv run python load_hohonu_devices.py
```

### `noaa_datum_converter.py`

NOAA datum conversion helpers (`NAVD88`, `MLLW`) used when combining external data.

## 6. Suggested GitHub docs links

Keep these docs linked from README:

- `docs/ROLLING_BACKTEST.md`
- `docs/QA_METRICS.md`
- `DEPLOYMENT.md`
