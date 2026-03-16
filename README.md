# ML Pipeline Anomaly Detection

This repository is a small anomaly-detection project for pipeline-style operational telemetry. It is designed to feel like a realistic ML portfolio project, with a clear problem framing, a synthetic but structured dataset, reproducible experiment artifacts, and notes on what worked and what did not.

## Why I chose this problem

I wanted an example that sits between textbook tabular ML and real industrial monitoring work. Pipeline telemetry is a good fit because the data are sequential, the operating regime shifts over time, and the business question is usually not "classify everything perfectly," but "surface unusual windows early enough for review."

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
│   └── xgboost_feature_importance.csv
├── configs/
│   └── baseline.yaml
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

Short rolling windows (5) capture local fluctuations and short-term volatility, while longer windows (12) provide a more stable baseline for comparison features such as rolling statistics and z-score style signals.

## Models and why they are here

### 1. Isolation Forest

Isolation Forest is the unsupervised baseline because in real operations labels are often sparse or delayed. It is useful as a first-pass "what looks unusual?" detector.

### 2. Random Forest

Random Forest acts as a supervised benchmark on the labeled sample. It is not the most advanced model here, but it gives a strong reference point for whether the engineered features are informative.

### 3. XGBoost

XGBoost is included as the remaining boosted-tree baseline because it is often one of the strongest default choices for structured tabular data with nonlinear interactions and class imbalance.

### 4. PyTorch Autoencoder

The autoencoder is here to show a neural approach based on reconstruction error. The goal was not to "beat everything with deep learning," but to compare how a compact neural model behaves against classical baselines on structured sensor data.

## Current results

Results below were regenerated from the included sample dataset.

| model | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 0.974 | 0.987 | 1.000 |
| Isolation Forest | 0.727 | 0.842 | 0.780 | 0.986 |
| PyTorch Autoencoder | 0.759 | 0.579 | 0.657 | 0.947 |

A few takeaways:
- XGBoost is now the strongest supervised baseline in the repo, recovering the one anomaly that Random Forest misses.
- Random Forest remains very strong and easier to explain, so it is still a useful reference model.
- Isolation Forest is still more useful than the autoencoder when recall matters and labels are unavailable.
- The autoencoder is viable, but it remains sensitive to threshold choice and loses recall at the selected operating point.

## Why I picked the current autoencoder threshold

The default autoencoder threshold uses the 96th percentile of reconstruction error. Lower thresholds created too many alerts, while higher thresholds suppressed medium-strength anomalies. The full threshold sweep is saved in `artifacts/autoencoder_threshold_sweep.csv`.

## Quick start

Create a virtual environment, activate it, and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
make install
```

Run the full pipeline:

```bash
make pipeline
```

## Key files to inspect first

- `docs/experiment_notes.md` for the modeling story
- `artifacts/metrics_summary.csv` for the headline numbers
- `artifacts/random_forest_feature_importance.csv` for the benchmark tree baseline
- `artifacts/xgboost_feature_importance.csv` for boosted-tree interpretability
- `docs/what_did_not_work.md` for the non-polished part of the project

## Next extensions I would do

- train/test splits based on time rather than random holdout,
- drift-aware monitoring,
- event grouping so adjacent anomaly rows are handled as incidents,
- geospatial context if this were tied to an actual asset network,
- a small dashboard for triage.
