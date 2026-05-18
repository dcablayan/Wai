.PHONY: setup demo-data train benchmark report horizons events rolling-origin conformal noaa-eval dashboard test coverage summary visuals demo

setup:
	pip install -r requirements.txt

demo-data:
	python -m scripts.prepare_demo_data

train:
	python -m scripts.train_baseline

benchmark:
	python -m scripts.run_benchmark

report:
	python -m scripts.generate_report

horizons:
	python -m scripts.evaluate_horizons

events:
	python -m scripts.evaluate_events

rolling-origin:
	python -m scripts.evaluate_rolling_origin

conformal:
	python -m scripts.evaluate_conformal

noaa-eval:
	python -m scripts.evaluate_noaa_public

noaa-eval-offline:
	NOAA_OFFLINE=1 python -m scripts.evaluate_noaa_public

summary:
	python -m scripts.build_summary

visuals:
	python -m scripts.generate_research_visuals

dashboard:
	python -m scripts.run_dashboard

test:
	pytest tests/ -v

coverage:
	python -m scripts.run_coverage

demo: demo-data train report benchmark horizons events rolling-origin conformal noaa-eval-offline visuals summary
	@echo "Demo pipeline complete. Run 'make dashboard' to view the dashboard."
