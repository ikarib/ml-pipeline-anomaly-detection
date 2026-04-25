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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


@dataclass(frozen=True)
class TimeHoldoutSplit:
    train_positions: np.ndarray
    holdout_positions: np.ndarray
    train_end: Any
    holdout_start: Any
    holdout_end: Any


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


def _coerce_timestamp_series(timestamps: Any, n_samples: int | None = None) -> pd.Series:
    ts = pd.Series(timestamps).reset_index(drop=True)
    if n_samples is not None and len(ts) != n_samples:
        raise ValueError(
            f"Expected {n_samples} timestamps for the time split, got {len(ts)}."
        )
    if ts.empty:
        raise ValueError("Cannot create a time split from an empty timestamp series.")
    if ts.isna().any():
        raise ValueError("Time split timestamps must not contain missing values.")
    if pd.api.types.is_numeric_dtype(ts):
        return ts
    return pd.Series(pd.to_datetime(ts, errors="raise"), name=ts.name)


def _holdout_count(n_samples: int, holdout_size: float | int) -> int:
    if isinstance(holdout_size, bool):
        raise TypeError("holdout_size must be a float fraction or integer row count.")
    if isinstance(holdout_size, float):
        if not 0 < holdout_size < 1:
            raise ValueError("Float holdout_size must be between 0 and 1.")
        return int(np.ceil(n_samples * holdout_size))
    holdout_count = int(holdout_size)
    if holdout_count <= 0:
        raise ValueError("Integer holdout_size must be positive.")
    return holdout_count


def make_time_holdout_split(
    timestamps: Any,
    holdout_size: float | int = 0.25,
    n_samples: int | None = None,
) -> TimeHoldoutSplit:
    """Return train and holdout iloc positions split by time order."""
    ts = _coerce_timestamp_series(timestamps, n_samples=n_samples)
    sample_count = len(ts)
    holdout_count = _holdout_count(sample_count, holdout_size)
    if holdout_count >= sample_count:
        raise ValueError("Holdout split must leave at least one training row.")

    ordered_positions = ts.sort_values(kind="mergesort").index.to_numpy(dtype=int)
    train_positions = ordered_positions[:-holdout_count]
    holdout_positions = ordered_positions[-holdout_count:]

    train_end = ts.iloc[train_positions].max()
    holdout_start = ts.iloc[holdout_positions].min()
    holdout_end = ts.iloc[holdout_positions].max()
    if train_end >= holdout_start:
        raise ValueError(
            "Time split boundary overlaps: train_end must be earlier than holdout_start."
        )

    return TimeHoldoutSplit(
        train_positions=train_positions,
        holdout_positions=holdout_positions,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )


def _resolve_time_split(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any],
    timestamps: Any | None,
    split: TimeHoldoutSplit | None,
) -> TimeHoldoutSplit:
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}.")
    if split is not None:
        return split

    split_timestamps = timestamps if timestamps is not None else np.arange(len(X))
    holdout_size = options.get("holdout_size", options.get("test_size", 0.25))
    return make_time_holdout_split(
        split_timestamps,
        holdout_size=holdout_size,
        n_samples=len(X),
    )


def _split_xy(
    X: pd.DataFrame,
    y: pd.Series,
    split: TimeHoldoutSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return (
        X.iloc[split.train_positions],
        X.iloc[split.holdout_positions],
        y.iloc[split.train_positions],
        y.iloc[split.holdout_positions],
    )


def _split_artifacts(split: TimeHoldoutSplit) -> dict[str, Any]:
    return {
        "train_rows": int(len(split.train_positions)),
        "holdout_rows": int(len(split.holdout_positions)),
        "train_end": split.train_end,
        "holdout_start": split.holdout_start,
        "holdout_end": split.holdout_end,
    }


def _positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def train_isolation_forest(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any] | None = None,
    timestamps: Any | None = None,
    split: TimeHoldoutSplit | None = None,
) -> TrainResult:
    options = options or {}
    split = _resolve_time_split(X, y, options, timestamps, split)
    X_train, X_holdout, _, y_holdout = _split_xy(X, y, split)
    model = IsolationForest(
        n_estimators=options.get("n_estimators", 300),
        contamination=options.get("contamination", 0.06),
        random_state=options.get("random_state", 42),
    )
    model.fit(X_train)
    scores = -model.score_samples(X_holdout)
    preds = (model.predict(X_holdout) == -1).astype(int)
    metrics = _metric_dict(y_holdout.to_numpy(), preds, scores)
    return TrainResult(
        name="Isolation Forest",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "contamination": model.contamination,
            **_split_artifacts(split),
        },
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any] | None = None,
    timestamps: Any | None = None,
    split: TimeHoldoutSplit | None = None,
) -> TrainResult:
    options = options or {}
    split = _resolve_time_split(X, y, options, timestamps, split)
    X_train, X_holdout, y_train, y_holdout = _split_xy(X, y, split)
    model = RandomForestClassifier(
        n_estimators=options.get("n_estimators", 350),
        max_depth=options.get("max_depth", 10),
        min_samples_leaf=options.get("min_samples_leaf", 2),
        class_weight=options.get("class_weight", "balanced_subsample"),
        random_state=options.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    scores = _positive_class_scores(model, X_holdout)
    preds = (scores >= 0.5).astype(int)
    metrics = _metric_dict(y_holdout.to_numpy(), preds, scores)
    feature_importance = _feature_importance_frame(X.columns, model.feature_importances_)
    return TrainResult(
        name="Random Forest",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "holdout_score": float(model.score(X_holdout, y_holdout)),
            "feature_importance": feature_importance,
            **_split_artifacts(split),
        },
    )



def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    options: dict[str, Any] | None = None,
    timestamps: Any | None = None,
    split: TimeHoldoutSplit | None = None,
) -> TrainResult:
    options = options or {}
    split = _resolve_time_split(X, y, options, timestamps, split)
    X_train, X_holdout, y_train, y_holdout = _split_xy(X, y, split)
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
    scores = _positive_class_scores(model, X_holdout)
    preds = (scores >= 0.5).astype(int)
    metrics = _metric_dict(y_holdout.to_numpy(), preds, scores)
    feature_importance = _feature_importance_frame(X.columns, model.feature_importances_)
    return TrainResult(
        name="XGBoost",
        predictions=preds,
        scores=scores,
        metrics=metrics,
        artifacts={
            "model": model,
            "holdout_score": float(model.score(X_holdout, y_holdout)),
            "feature_importance": feature_importance,
            **_split_artifacts(split),
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
    options: dict[str, Any] | None = None,
    timestamps: Any | None = None,
    split: TimeHoldoutSplit | None = None,
) -> TrainResult:
    options = options or {}
    split = _resolve_time_split(X, y, options, timestamps, split)
    X_train, X_holdout, y_train, y_holdout = _split_xy(X, y, split)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_holdout_np = X_holdout.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy()
    y_holdout_np = y_holdout.to_numpy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
    X_holdout_scaled = scaler.transform(X_holdout_np).astype(np.float32)

    normal_rows = X_train_scaled[y_train_np == 0]
    train_matrix = normal_rows if len(normal_rows) else X_train_scaled

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
        train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32).to(device)
        train_reconstructed = model(train_tensor).cpu().numpy()
        holdout_reconstructed = model(holdout_tensor).cpu().numpy()

    train_reconstruction_error = np.mean(
        (X_train_scaled - train_reconstructed) ** 2,
        axis=1,
    )
    reconstruction_error = np.mean(
        (X_holdout_scaled - holdout_reconstructed) ** 2,
        axis=1,
    )
    threshold_quantile = options.get("threshold_quantile", 0.96)
    threshold_errors = train_reconstruction_error[y_train_np == 0]
    if not len(threshold_errors):
        threshold_errors = train_reconstruction_error
    threshold = float(np.quantile(threshold_errors, threshold_quantile))
    preds = (reconstruction_error >= threshold).astype(int)
    metrics = _metric_dict(y_holdout_np, preds, reconstruction_error)

    if len(np.unique(y_holdout_np)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(
            y_holdout_np,
            reconstruction_error,
        )
        pr_table = pd.DataFrame(
            {
                "precision": precisions[:-1],
                "recall": recalls[:-1],
                "threshold": thresholds,
            }
        )
    else:
        pr_table = pd.DataFrame(columns=["precision", "recall", "threshold"])
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
            "threshold_source": "train_normal_reconstruction_error",
            "pr_table": pr_table,
            **_split_artifacts(split),
        },
    )
