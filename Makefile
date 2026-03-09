.PHONY: install data train pipeline notebook clean

ifeq ($(OS),Windows_NT)
RM_CACHE = rd /s /q __pycache__ 2>nul || exit /b 0
RM_ARTIFACTS = del /q artifacts\* 2>nul || exit /b 0
else
RM_CACHE = rm -rf __pycache__
RM_ARTIFACTS = rm -rf artifacts/*
endif

install:
	pip install -r requirements.txt -e .

data:
	python -m anomaly_pipeline.data_generation --output data/sample_pipeline_data.csv --periods=720 --seed=42

train:
	python -m anomaly_pipeline.pipeline --config configs/baseline.yaml

pipeline: data train

notebook:
	jupyter notebook notebooks/anomaly_detection.ipynb

clean:
	$(RM_CACHE)
	$(RM_ARTIFACTS)
