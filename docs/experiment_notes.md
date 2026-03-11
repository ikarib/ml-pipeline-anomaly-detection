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
### Random Forest
This became the strongest model on the labeled sample. The main lesson was not “Random Forest is always best,” but that the engineered features capture enough local structure to make supervised learning effective on this dataset.
### Autoencoder
The autoencoder trained cleanly and produced a sensible reconstruction-error ranking. However, its practical usefulness depends heavily on thresholding. With the current setting, precision is acceptable but recall is weaker than Isolation Forest.
## Feature importance notes
The strongest Random Forest feature was `pressure_flow_ratio`, followed by temperature and rolling flow statistics. That aligns with how anomalies were injected: several abnormal windows were created by making pressure and flow diverge rather than just moving one variable in isolation.
## What I would do next

For a stronger production-style version, I would:
- evaluate on a true future holdout window,
- convert point anomalies into event-level alerts,
- add calibration logic for alert volume,
- compare against a gradient-boosted tree baseline,
- test robustness under stronger regime drift.
