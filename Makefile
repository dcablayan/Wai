.PHONY: setup demo-data train benchmark report horizons dashboard test demo

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

dashboard:
	streamlit run app.py

test:
	pytest tests/ -v

demo: demo-data train report benchmark horizons
	@echo "Demo pipeline complete. Run 'make dashboard' to view the dashboard."
