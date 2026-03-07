# ML Pipeline Anomaly Detection



This repository is a small anomaly-detection project for pipeline-style operational telemetry.
I built it to look and feel like a real ML portfolio project: there is a clear problem framing, a realistic synthetic dataset, explicit experiment outputs, and a candid discussion of what worked and what did not.


## Why I chose this problem



I wanted an example that sits between textbook tabular ML and real industrial monitoring work.
Pipeline telemetry is a good fit because the data are sequential, the operating regime shifts over time, and the business question is usually not “classify everything perfectly,” but “surface unusual windows early enough for review.”

The sample dataset is synthetically generated and includes:
- daily and weekly seasonality,
- mild regime shifts,
- several distinct anomaly types,
- labels that make it possible to compare unsupervised and supervised approaches.



## What is in the repo

```text
ml-pipeline-anomaly-detection/
├── artifacts/
├── data/
├── docs/
├── notebooks/
├── src/
│   ├── data_generation.py
├── .gitignore
├── README.md
└── requirements.txt
```



## Models



### 1. Isolation Forest



### 2. Random Forest



### 3. PyTorch Autoencoder



## Current results




## Quick start



```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m src.data_generation
```





