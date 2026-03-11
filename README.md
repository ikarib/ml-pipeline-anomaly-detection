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
│   ├── autoencoder_threshold_sweep.csv
│   ├── autoencoder_training_history.csv
│   ├── isolation_forest_scores.png
│   ├── metrics_summary.csv
│   ├── random_forest_feature_importance.csv
│   ├── true_anomalies.png
├── configs/
│   ├── baseline.yaml
├── data/
│   └── sample_pipeline_data.csv
├── docs/
│   ├── experiment_notes.md
│   ├── limitations.md
│   └── what_did_not_work.md
├── notebooks/
│   └── anomaly_detection.ipynb
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── pipeline.py
│   └── reporting.py
├── .gitignore
├── Makefile
├── README.md
├── pyproject.toml
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
I used Isolation Forest as the unsupervised baseline because in real operations labels are often sparse or delayed. It is useful as a “what looks unusual?” first pass.


### 2. Random Forest
This acts as a supervised benchmark on the labeled sample. It is not the most advanced model here, but it gives a good reference point for whether the engineered features are informative.

### 3. PyTorch Autoencoder
I added the autoencoder to show a neural approach based on reconstruction error. In practice, the point was not to “beat everything with deep learning,” but to compare how a compact neural model behaves against classical baselines on structured sensor data.

## Current results

Results were generated from the included sample dataset.

| model | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.727 | 0.842 | 0.780 | 0.988 |
| Random Forest | 1.000 | 0.974 | 0.987 | 1.000 |
| PyTorch Autoencoder | 0.793 | 0.605 | 0.687 | 0.966 |

A few takeaways:
- The **Random Forest** is strongest here because the dataset is labeled and the anomaly patterns are learnable from engineered features.
- **Isolation Forest** is more useful than the autoencoder in this version when recall matters.
- The **autoencoder** is viable, but it is sensitive to threshold choice and loses recall at the selected operating point.
## Why I picked the current autoencoder threshold

I set the default autoencoder threshold using the **96th percentile** of reconstruction error. I chose that because lower thresholds created too many alerts, while higher thresholds suppressed medium-strength anomalies. The full threshold sweep is saved in `artifacts/autoencoder_threshold_sweep.csv`.
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
- `artifacts/metrics_summary.csv` for the headline numbers
- `artifacts/random_forest_feature_importance.csv` for the strongest signals
- `docs/what_did_not_work.md` for the non-polished part of the project

## Next extensions I would do

- train/test splits based on time rather than random holdout,
- drift-aware monitoring,
- event grouping so adjacent anomaly rows are handled as incidents,
- geospatial context if this were tied to an actual asset network,
- a small dashboard for triage.
