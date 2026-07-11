# QA metrics and scoring

This project reports multiple forecast quality metrics and builds a composite
`selected_qa_score` used for ranking.

## Metric definitions

- `rmse_target`: root-mean-square error on target column
- `mae_target`: mean absolute error on target column
- `mape_target`: mean absolute percentage error
- `r2_target`: R² on target column
- `corr_target`: correlation with observed target
- `mpe_target`: mean percentage error
- `me_target`: mean error (bias)
- `minmax_target`: min-max range score from `forecast_accuracy`
- `nse_target`: Nash–Sutcliffe efficiency

`selected_qa_score` is computed from the above with a weighted blend in
`tide_ml_engine._qa_score_from_metrics(...)`:

- lower-is-better: RMSE, MAE, MAPE, MPE, ME
- higher-is-better: R², CORR, MINMAX, NSE

Default weights:

- rmse: 0.40
- mae: 0.20
- mape: 0.10
- me: 0.07
- mpe: 0.03
- r2: 0.10
- corr: 0.05
- minmax: 0.05
- nse: 0.10

If a metric is missing or non-finite, it is skipped from the weighted score
normalization.

## Composite score input and tuning

Pass a custom score map using `--qa-weights` for the benchmark tool:

```bash
--qa-weights "rmse:0.4,mae:0.2,nse:0.2,r2:0.2"
```

Supported aliases are normalized to internal metric names:

- `rmse` → `rmse_target`
- `mae` → `mae_target`
- `r2` → `r2_target`
- `selected_*` and `qa_score` are accepted synonyms where applicable

## Output schema

`run_combo_benchmark.py` returns:

- `benchmark` rows with raw metrics + `selected_qa_score`
- `rolling_backtest.rows` with same metric columns per fold
- `rolling_backtest.family_health` with family roll-ups:
  - `avg_*` and `std_*` aggregates
  - `avg_rank`, `rank_stability`
  - `avg_qa_score`
  - `leaderboard_score` (family composite)

## Practical interpretation

- Compare `avg_rmse` and `avg_qa_score` for mean quality.
- Compare `std_rmse` / `rank_stability` for robustness across folds.
- Prefer families with strong quality + low instability if runtime allows.

