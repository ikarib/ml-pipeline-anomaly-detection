# Experiment Notes

## Problem framing

The goal was to detect unusual operating windows in pipeline-style telemetry with three kinds of signals:
- pressure,
- flow,
- temperature.

I wanted a workflow that answered three practical questions:
1. What can be detected without labels?
2. How much do labels help?
3. Does a compact neural model add anything useful here?

## Dataset choices

The sample data are synthetic, but I made them less toy-like by adding:
- hourly seasonality,
- a weekly temperature pattern,
- small regime shifts,
- multiple anomaly mechanisms rather than one repeated anomaly shape.

That matters because perfectly clean synthetic data can make every model look unrealistically good.

## Feature choices

I intentionally stayed with interpretable engineered features:
- first differences,
- rolling means,
- rolling standard deviations,
- pressure/flow ratio,
- cyclical hour-of-day terms,
- rolling z-scores.

I did not jump straight to sequence models because the first thing I wanted to test was whether simple features already separated anomalous windows.

## Model observations

### Isolation Forest

This worked reasonably well as a label-light baseline. On the chronological holdout it found all anomaly rows, with a small number of false positives.

Observed holdout metrics: precision `0.733`, recall `1.000`, f1 `0.846`, roc_auc `0.991`.

### Random Forest

This stayed very strong on the labeled sample and remained a useful benchmark for interpretable supervised learning. The main lesson was not "Random Forest is always best," but that the engineered features capture enough local structure to make supervised learning highly effective on this dataset.

Observed holdout metrics: precision `1.000`, recall `1.000`, f1 `1.000`, roc_auc `1.000`.

### XGBoost

XGBoost is now the only boosted-tree baseline in the repo, and it remains the strongest supervised model on this sample. Its performance suggests that the anomaly patterns are highly learnable from the engineered tabular features and benefit from boosted-tree fitting.

Observed holdout metrics: precision `1.000`, recall `1.000`, f1 `1.000`, roc_auc `1.000`.

### Autoencoder

The autoencoder trained cleanly and produced a sensible reconstruction-error ranking. Its threshold is now calibrated from training reconstruction errors only, then evaluated on the future holdout. With the current setting it catches the future anomaly window, but at the cost of more false positives than Isolation Forest.

Observed holdout metrics: precision `0.579`, recall `1.000`, f1 `0.733`, roc_auc `0.985`.

## Feature importance notes

The strongest supervised feature stays consistent across the tree baselines that remain: `pressure_flow_ratio` is the top signal for both Random Forest and XGBoost. Temperature features and short rolling flow or pressure statistics also stay near the top. That aligns with how anomalies were injected: several abnormal windows were created by making pressure and flow diverge rather than just moving one variable in isolation.

## Comparison takeaways

XGBoost and Random Forest are tied on the current chronological holdout. Both recover the future anomaly window without false positives.

Random Forest still has a strong role as a simpler supervised benchmark. It is easier to explain, produces stable feature rankings, and lands close enough to XGBoost that the practical gap here is small.

Isolation Forest wins on the label-light story. It does not match the supervised models, but it remains the strongest option in this repo when you assume labels are missing or delayed and you still want useful recall.

The autoencoder trails on this version of the problem because the dataset is relatively small and the tabular feature engineering already makes the anomalies easy for tree models to isolate. Its ranking signal is sensible, but the train-calibrated threshold is noisier than the Isolation Forest operating point.

## What I would do next

For a stronger production-style version, I would:
- convert point anomalies into event-level alerts,
- add calibration logic for alert volume,
- stress-test XGBoost under stronger regime drift,
- compare models across multiple rolling or blocked validation windows.
