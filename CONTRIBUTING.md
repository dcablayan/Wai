# Contributing to Wai

## Local setup

```bash
git clone https://github.com/dcablayan/Wai.git
cd Wai
uv sync --locked --all-extras
```

## Running the demo pipeline

```bash
make demo-data   # generate synthetic data
make train       # train pipeline models
make benchmark   # run prototype model benchmarks
make report      # generate HTML reports
make dashboard   # launch Streamlit dashboard
```

Or run the full pipeline in sequence:

```bash
make demo
```

## Running tests

```bash
make test
# or
uv run pytest tests/ -v
```

The test suite covers data loading, validation, windowing, feature engineering,
models, and reporting. All tests run against synthetic data — no external APIs
or private data required.

## Data guidelines

- **Never commit real sensor data, API keys, or private credentials.**
- Demo data in `data/demo/` is synthetic and NOAA-derived — safe to commit.
- The `.gitignore` blocks `data/raw/`, `data/processed/`, and all non-demo
  CSVs. Do not loosen these rules.

## Code style

- Python 3.13+ compatible; CI also runs Python 3.14
- Run `uv run ruff check src scripts tests Hohonu-1 app.py` before review
- Keep learned artifacts as validated JSON; do not introduce pickle/joblib loads
- Type hints on public function signatures
- No external ML frameworks in `src/models/prototypes.py` (stdlib + math only)
- Docstrings on public classes and functions; one line is enough for obvious cases

## Adding a model

1. Add the class to `src/models/baseline.py` (pipeline API) or
   `src/models/prototypes.py` (pure-Python benchmark API).
2. Pipeline models must implement `.fit(df)`, `.predict_on(df)`, and
   `.evaluate(df)` to match `PersistenceModel` and `HarmonicRidgeModel`.
3. Prototype models must accept and return window dicts — see the module
   docstring in `src/models/prototypes.py`.
4. Add tests in `tests/`.
5. Update `docs/modeling.md` with a description of what the model does and
   its honest limitations.
