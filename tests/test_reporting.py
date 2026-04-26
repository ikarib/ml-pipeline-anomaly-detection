from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anomaly_pipeline.reporting import cv_fold_metrics_table, cv_metrics_summary


def _split(fold: int) -> SimpleNamespace:
    return SimpleNamespace(
        fold_index=fold,
        split_strategy="rolling",
        train_positions=np.arange(10 * fold),
        holdout_positions=np.arange(5),
        train_end=f"2025-01-0{fold} 00:00:00",
        holdout_start=f"2025-01-0{fold} 01:00:00",
        holdout_end=f"2025-01-0{fold} 05:00:00",
    )


def _result(model: str, precision: float, true_positive: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=model,
        metrics={
            "precision": precision,
            "recall": 1.0,
            "f1": precision,
            "true_negative": 3,
            "false_positive": 1,
            "false_negative": 0,
            "true_positive": true_positive,
        },
    )


def test_cv_metrics_summary_aggregates_per_fold_metrics() -> None:
    fold_metrics = cv_fold_metrics_table(
        [
            (_split(1), [_result("Model A", 0.5, 1), _result("Model B", 1.0, 2)]),
            (_split(2), [_result("Model A", 1.0, 3), _result("Model B", 0.5, 4)]),
        ]
    )

    summary = cv_metrics_summary(fold_metrics)
    model_a = summary.loc[summary["model"] == "Model A"].iloc[0]

    assert len(fold_metrics) == 4
    assert model_a["folds"] == 2
    assert model_a["holdout_rows"] == 10
    assert model_a["mean_precision"] == 0.75
    assert model_a["total_true_positive"] == 4
    assert model_a["total_false_positive"] == 2
    assert model_a["micro_precision"] == pytest.approx(4 / 6)
    assert model_a["micro_recall"] == 1.0
    assert model_a["micro_f1"] == pytest.approx(0.8)
