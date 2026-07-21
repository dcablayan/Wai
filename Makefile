.PHONY: setup demo-data train benchmark report horizons events rolling-origin conformal noaa-stations noaa-eval noaa-eval-offline dashboard test coverage summary freshness visuals scientific-audit forecast replay train-router demo

UV ?= uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest

setup:
	$(UV) sync --locked --all-extras

demo-data:
	$(PYTHON) -m scripts.prepare_demo_data

train:
	$(PYTHON) -m scripts.train_baseline

benchmark:
	$(PYTHON) -m scripts.run_benchmark

report:
	$(PYTHON) -m scripts.generate_report

horizons:
	$(PYTHON) -m scripts.evaluate_horizons

events:
	$(PYTHON) -m scripts.evaluate_events

rolling-origin:
	$(PYTHON) -m scripts.evaluate_rolling_origin

conformal:
	$(PYTHON) -m scripts.evaluate_conformal

noaa-stations:
	$(PYTHON) -m scripts.sync_noaa_stations

noaa-eval:
	$(PYTHON) -m scripts.evaluate_noaa_public

noaa-eval-offline:
	NOAA_OFFLINE=1 $(PYTHON) -m scripts.evaluate_noaa_public

summary:
	$(PYTHON) -m scripts.build_summary

freshness:
	$(PYTHON) -m scripts.check_report_freshness

visuals:
	$(PYTHON) -m scripts.generate_research_visuals

scientific-audit:
	$(PYTHON) -m scripts.audit_scientific_evidence

dashboard:
	$(PYTHON) -m scripts.run_dashboard

forecast:
	$(PYTHON) -m scripts.run_orchestrated_forecast

replay:
	$(PYTHON) -m scripts.run_historical_replay

train-router:
	$(PYTHON) -m scripts.train_router

test:
	$(PYTEST) tests/ -v

coverage:
	$(PYTHON) -m scripts.run_coverage

demo: demo-data train report benchmark horizons events rolling-origin conformal noaa-eval-offline visuals scientific-audit summary
	@echo "Demo pipeline complete. Run 'make dashboard' to view the dashboard."
