from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


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
    options: dict[str, Any] | None = None
) -> TrainResult:
    options = options or {}
    model = IsolationForest(
        n_estimators=options.get("n_estimators", 300),
        contamination=options.get("contamination", 0.06),
        random_state=options.get("random_state", 42),
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
        artifacts={"model": model, "contamination": model.contamination},
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any] | None = None
) -> TrainResult:
    options = options or {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y,
        test_size=options.get("test_size", 0.25),
        random_state=options.get("random_state", 42)
    )
    model = RandomForestClassifier(
        n_estimators=options.get("n_estimators", 350),
        max_depth=options.get("max_depth", 10),
        min_samples_leaf=options.get("min_samples_leaf", 2),
        class_weight=options.get("class_weight", "balanced_subsample"),
        random_state=options.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X)[:, 1]
    preds = (scores >= 0.5).astype(int)
    metrics = _metric_dict(y.to_numpy(), preds, scores)
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return TrainResult(
        name="Random Forest",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "holdout_score": float(model.score(X_test, y_test)),
            "feature_importance": feature_importance.reset_index(drop=True),
        },
    )
