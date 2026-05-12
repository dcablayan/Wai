.PHONY: setup demo-data train benchmark report dashboard test demo

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

dashboard:
	streamlit run app.py

test:
	pytest tests/ -v

demo: demo-data train report benchmark
	@echo "Demo pipeline complete. Run 'make dashboard' to view the dashboard."
