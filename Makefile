.PHONY: setup demo-data train benchmark report horizons noaa-eval dashboard test coverage demo

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

noaa-eval:
	python -m scripts.evaluate_noaa_public

noaa-eval-offline:
	NOAA_OFFLINE=1 python -m scripts.evaluate_noaa_public

dashboard:
	streamlit run app.py

test:
	pytest tests/ -v

coverage:
	pytest tests/ -v --cov=src --cov=scripts --cov-report=term-missing

demo: demo-data train report benchmark horizons noaa-eval-offline
	@echo "Demo pipeline complete. Run 'make dashboard' to view the dashboard."
