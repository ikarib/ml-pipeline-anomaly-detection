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

This worked reasonably well as a label-light baseline. It found many anomalous windows, but the tradeoff was false positives around regime changes.

Observed metrics: precision `0.727`, recall `0.842`, f1 `0.780`, roc_auc `0.986`.

### Random Forest

This stayed very strong on the labeled sample and remained a useful benchmark for interpretable supervised learning. The main lesson was not "Random Forest is always best," but that the engineered features capture enough local structure to make supervised learning highly effective on this dataset.

Observed metrics: precision `1.000`, recall `0.974`, f1 `0.987`, roc_auc `1.000`.

### XGBoost

XGBoost is now the only boosted-tree baseline in the repo, and it remains the strongest supervised model on this sample. Its performance suggests that the anomaly patterns are highly learnable from the engineered tabular features and benefit from boosted-tree fitting.

Observed metrics: precision `1.000`, recall `1.000`, f1 `1.000`, roc_auc `1.000`.

### Autoencoder

The autoencoder trained cleanly and produced a sensible reconstruction-error ranking. However, its practical usefulness still depends heavily on thresholding. With the current setting, precision is acceptable but recall is weaker than Isolation Forest and much weaker than the supervised tree models.

Observed metrics: precision `0.759`, recall `0.579`, f1 `0.657`, roc_auc `0.947`.

## Feature importance notes

The strongest supervised feature stays consistent across the tree baselines that remain: `pressure_flow_ratio` is the top signal for both Random Forest and XGBoost. Temperature features and short rolling flow or pressure statistics also stay near the top. That aligns with how anomalies were injected: several abnormal windows were created by making pressure and flow diverge rather than just moving one variable in isolation.

## Comparison takeaways

XGBoost wins when labels are available and the anomaly patterns are encoded well by the engineered features. On this sample it recovers the last missed supervised anomaly and moves from Random Forest's near-perfect result to perfect precision and recall.

Random Forest still has a strong role as a simpler supervised benchmark. It is easier to explain, produces stable feature rankings, and lands close enough to XGBoost that the practical gap here is small.

Isolation Forest wins on the label-light story. It does not match the supervised models, but it remains the strongest option in this repo when you assume labels are missing or delayed and you still want useful recall.

The autoencoder loses on this version of the problem because the dataset is relatively small and the tabular feature engineering already makes the anomalies easy for tree models to isolate. Its ranking signal is sensible, but the chosen threshold leaves too much recall on the table.

## What I would do next

For a stronger production-style version, I would:
- evaluate on a true future holdout window,
- convert point anomalies into event-level alerts,
- add calibration logic for alert volume,
- stress-test XGBoost under stronger regime drift,
- compare supervised models under time-based validation instead of random holdout.
