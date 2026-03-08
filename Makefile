.PHONY: install data train pipeline notebook clean

install:
	pip install -r requirements.txt -e .

data:
	python -m anomaly_pipeline.data_generation --output data/sample_pipeline_data.csv --periods=720 --seed=42

train:
	python -m anomaly_pipeline.pipeline --config configs/baseline.yaml

pipeline: data train

notebook:
	jupyter notebook notebooks/anomaly_detection_walkthrough.ipynb

clean:
	rm -rf artifacts/*
	rm -rf __pycache__
	.PHONY: install data train pipeline notebook clean
