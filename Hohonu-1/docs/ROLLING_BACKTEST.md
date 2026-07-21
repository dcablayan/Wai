# Rolling backtesting and family health

This project supports time-aware robustness evaluation in addition to one-shot holdout
benchmarking.

## Why rolling backtesting

Single holdout scores can look good by chance. Rolling backtesting evaluates candidate
models across multiple temporal folds:

- progressively later fold starts
- fixed/auto holdout windows
- candidate retraining per fold
- per-fold metric and family-level aggregation

## CLI controls

`run_combo_benchmark.py`:

- `--rolling-backtest`
- `--rolling-folds N` (default 4)
- `--rolling-window {auto,month,quarter}`
- `--holdout-steps K` (optional explicit holdout window size)
- `--family-health-sort-metric {leaderboard_score,avg_qa_score,avg_rmse,...}`

`--rolling-window` behavior:

- `auto`: `max(1, holdout_ratio * len(data))`, clamped to min windows
- `month`: ~30 days worth of data points
- `quarter`: ~90 days worth of data points

All window sizing is based on the inferred timestamp cadence from the input data index.

## Returned structure

When enabled, `run_combo_benchmark.py` includes:

- `rolling_backtest.fold_count`: number of folds generated
- `rolling_backtest.requested_folds`: requested count from CLI
- `rolling_backtest.evaluated_folds`: folds that met minimum train/holdout size
- `rolling_backtest.rows`: per candidate × fold metric rows
- `rolling_backtest.family_health`: family-level summary table

## Family health aggregation

Family health rows combine candidate rows for each model family (e.g. `rf`, `lstm`, `pinn`) and compute:

- `avg_*`, `std_*` for key metrics
- `avg_rank` and `rank_stability` from fold-based ranking
- `avg_qa_score`
- `leaderboard_score` (family composite built from health row means)

This is useful for selecting model families that are not only accurate but also stable.

## Sorting behavior

- Family health default sort is by `leaderboard_score` (higher better).
- `avg_rmse` and most other raw metric orders still apply lower-is-better conventions.

## Example

```bash
uv run python run_combo_benchmark.py NODE123 \
  --mode all \
  --rolling-backtest \
  --rolling-folds 5 \
  --rolling-window month \
  --qa-sort-metric qa_score \
  --family-health-sort-metric leaderboard_score \
  --json
```

If rolling is invalid (not enough history, bad window config), the output includes
an `error` field inside `rolling_backtest`.
