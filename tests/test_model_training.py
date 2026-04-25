from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_pipeline.feature_engineering import engineer_features
from anomaly_pipeline.model_training import (
    make_time_holdout_split,
    train_autoencoder_anomaly_detector,
    train_isolation_forest,
    train_random_forest,
)


def _metric_row_count(metrics: dict[str, float]) -> int:
    return sum(
        int(metrics[key])
        for key in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    )


def _toy_supervised_data(n_rows: int = 40) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=n_rows, freq="h"))
    row_number = np.arange(n_rows)
    X = pd.DataFrame(
        {
            "sensor_a": row_number.astype(float),
            "sensor_b": np.sin(row_number / 3),
        }
    )
    y = pd.Series((row_number % 11 == 0).astype(int))
    return X, y, timestamps


def test_time_holdout_split_uses_latest_rows_even_when_input_is_unsorted() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-04",
                "2025-01-01",
                "2025-01-03",
                "2025-01-02",
                "2025-01-05",
            ]
        )
    )

    split = make_time_holdout_split(timestamps, holdout_size=2)

    assert split.train_positions.tolist() == [1, 3, 2]
    assert split.holdout_positions.tolist() == [0, 4]
    assert timestamps.iloc[split.train_positions].max() < timestamps.iloc[
        split.holdout_positions
    ].min()


def test_time_holdout_split_rejects_overlapping_boundary_timestamps() -> None:
    timestamps = pd.Series(
        pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"])
    )

    with pytest.raises(ValueError, match="boundary overlaps"):
        make_time_holdout_split(timestamps, holdout_size=1)


def test_random_forest_reports_metrics_on_holdout_rows_only() -> None:
    X, y, timestamps = _toy_supervised_data()

    result = train_random_forest(
        X,
        y,
        options={
            "holdout_size": 10,
            "n_estimators": 10,
            "max_depth": 4,
            "random_state": 7,
        },
        timestamps=timestamps,
    )

    assert len(result.predictions) == 10
    assert len(result.scores) == 10
    assert _metric_row_count(result.metrics) == 10
    assert result.artifacts["train_rows"] == 30
    assert result.artifacts["holdout_rows"] == 10
    assert result.artifacts["holdout_start"] == timestamps.iloc[30]


def test_isolation_forest_reports_metrics_on_holdout_rows_only() -> None:
    X, y, timestamps = _toy_supervised_data()

    result = train_isolation_forest(
        X,
        y,
        options={
            "holdout_size": 10,
            "n_estimators": 20,
            "contamination": 0.1,
            "random_state": 7,
        },
        timestamps=timestamps,
    )

    assert len(result.predictions) == 10
    assert len(result.scores) == 10
    assert _metric_row_count(result.metrics) == 10
    assert result.artifacts["train_rows"] == 30
    assert result.artifacts["holdout_rows"] == 10


def test_autoencoder_scaler_is_fit_on_training_rows_only() -> None:
    train_values = np.linspace(0.0, 1.0, 20)
    holdout_values = np.linspace(100.0, 120.0, 5)
    values = np.concatenate([train_values, holdout_values])
    X = pd.DataFrame({"sensor_a": values, "sensor_b": values * 0.1})
    y = pd.Series([0] * 18 + [1] * 2 + [0, 1, 0, 1, 0])
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=len(X), freq="h"))

    result = train_autoencoder_anomaly_detector(
        X,
        y,
        options={
            "holdout_size": 5,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "random_state": 7,
            "threshold_quantile": 0.95,
        },
        timestamps=timestamps,
    )

    np.testing.assert_allclose(
        result.artifacts["scaler"].mean_,
        X.iloc[:20].mean().to_numpy(),
    )
    assert not np.allclose(
        result.artifacts["scaler"].mean_,
        X.mean().to_numpy(),
    )
    assert len(result.predictions) == 5
    assert len(result.scores) == 5
    assert _metric_row_count(result.metrics) == 5
    assert result.artifacts["threshold_source"] == "train_normal_reconstruction_error"


def test_pressure_flow_ratio_fill_does_not_use_future_rows() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "pressure_psi": [10.0, 20.0, 30.0],
            "flow_rate_m3h": [0.0, 2.0, 3.0],
            "temperature_c": [5.0, 6.0, 7.0],
            "is_anomaly": [0, 0, 0],
        }
    )

    bundle = engineer_features(df, short_window=2, long_window=3)

    assert bundle.X.loc[0, "pressure_flow_ratio"] == 0.0
    assert bundle.X.loc[1, "pressure_flow_ratio"] == 10.0
