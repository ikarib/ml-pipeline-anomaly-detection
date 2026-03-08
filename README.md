# ML Pipeline Anomaly Detection



This repository is a small anomaly-detection project for pipeline-style operational telemetry.
I built it to look and feel like a real ML portfolio project: there is a clear problem framing, a realistic synthetic dataset, explicit experiment outputs, and a candid discussion of what worked and what did not.


## Why I chose this problem



I wanted an example that sits between textbook tabular ML and real industrial monitoring work.
Pipeline telemetry is a good fit because the data are sequential, the operating regime shifts over time, and the business question is usually not “classify everything perfectly,” but “surface unusual windows early enough for review.”


## What is in the repo

```text
ml-pipeline-anomaly-detection/
├── artifacts/
│   ├── metrics_summary.csv
│   ├── pressure_series.png
├── configs/
│   ├── baseline.yaml
├── data/
│   └── sample_pipeline_data.csv
├── docs/
│   ├── experiment_notes.md
│   ├── limitations.md
├── notebooks/
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── pipeline.py
│   └── reporting.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Data
The sample dataset is synthetically generated and includes:
- daily and weekly seasonality,
- mild regime shifts,
- several distinct anomaly types,
- labels that make it possible to compare unsupervised and supervised approaches.

## Feature engineering
Short rolling windows (5) are used to capture local fluctuations and short-term volatility, while longer windows (12) provide a more stable estimate of the baseline distribution used for Z-score anomaly detection.


## Models and why they are here



### 1. Isolation Forest
I kept Isolation Forest as the unsupervised baseline because in real operations labels are often sparse or delayed. It is useful as a “what looks unusual?” first pass.


### 2. Random Forest



### 3. PyTorch Autoencoder



## Current results

Results were generated from the included sample dataset.

| model | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.727 | 0.842 | 0.780 | 0.988 |

## Quick start

Create virtual environment, activate it and Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
make install
```

Run the full pipeline:
```bash
make pipeline
```

## Key files to inspect first

- `docs/experiment_notes.md` for the modeling story




