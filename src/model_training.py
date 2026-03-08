from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class TrainResult:
    name: str
    predictions: np.ndarray
    scores: np.ndarray
    metrics: dict[str, float]
    artifacts: dict[str, Any]

def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    if y_score is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def train_isolation_forest(
    X: pd.DataFrame,
    y: pd.Series,
    contamination: float = 0.06,
    random_state: int = 42,
    n_estimators: int = 300,
) -> TrainResult:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    scores = -model.score_samples(X)
    preds = (model.predict(X) == -1).astype(int)
    metrics = _metric_dict(y.to_numpy(), preds, scores)
    return TrainResult(
        name="Isolation Forest",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={"model": model, "contamination": contamination},
    )


