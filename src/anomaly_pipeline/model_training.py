from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    name: str
    predictions: np.ndarray
    scores: np.ndarray
    metrics: dict[str, float]
    artifacts: dict[str, Any]

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden_dim = max(16, input_dim)
        bottleneck_dim = max(4, input_dim // 3)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _feature_importance_frame(feature_names: pd.Index, importance: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


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
    feature_importance = _feature_importance_frame(X.columns, model.feature_importances_)
    return TrainResult(
        name="Random Forest",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "holdout_score": float(model.score(X_test, y_test)),
            "feature_importance": feature_importance,
        },
    )



def train_xgboost(
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
    model = XGBClassifier(
        n_estimators=options.get("n_estimators", 350),
        learning_rate=options.get("learning_rate", 0.05),
        max_depth=options.get("max_depth", 6),
        subsample=options.get("subsample", 0.9),
        colsample_bytree=options.get("colsample_bytree", 0.9),
        reg_lambda=options.get("reg_lambda", 1.0),
        scale_pos_weight=options.get("scale_pos_weight", 8.0),
        random_state=options.get("random_state", 42),
        eval_metric=options.get("eval_metric", "logloss"),
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X)[:, 1]
    preds = (scores >= 0.5).astype(int)
    metrics = _metric_dict(y.to_numpy(), preds, scores)
    feature_importance = _feature_importance_frame(X.columns, model.feature_importances_)
    return TrainResult(
        name="XGBoost",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "holdout_score": float(model.score(X_test, y_test)),
            "feature_importance": feature_importance,
        },
    )


def _train_autoencoder(
    X_scaled: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> tuple[Autoencoder, list[float]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    model = Autoencoder(input_dim=X_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        history.append(epoch_loss / len(X_scaled))
    return model, history


def train_autoencoder_anomaly_detector(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any] | None = None
) -> TrainResult:
    options = options or {}
    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np).astype(np.float32)

    normal_rows = X_scaled[y_np == 0]
    train_matrix = normal_rows if len(normal_rows) else X_scaled

    model, history = _train_autoencoder(
        train_matrix,
        epochs=options.get("epochs", 50),
        batch_size=options.get("batch_size", 32),
        learning_rate=options.get("learning_rate", 0.001),
        random_state=options.get("random_state", 42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        reconstructed = model(full_tensor).cpu().numpy()

    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold_quantile = options.get("threshold_quantile", 0.96)
    threshold = float(np.quantile(reconstruction_error, threshold_quantile))
    preds = (reconstruction_error >= threshold).astype(int)
    metrics = _metric_dict(y_np, preds, reconstruction_error)

    precisions, recalls, thresholds = precision_recall_curve(y_np, reconstruction_error)
    pr_table = pd.DataFrame(
        {
            "precision": precisions[:-1],
            "recall": recalls[:-1],
            "threshold": thresholds,
        }
    )
    return TrainResult(
        name="PyTorch Autoencoder",
        predictions=preds,
        scores=reconstruction_error,
        metrics=metrics,
        artifacts={
            "model": model,
            "scaler": scaler,
            "history": history,
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "pr_table": pr_table,
        },
    )
