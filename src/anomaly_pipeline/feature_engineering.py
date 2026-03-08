from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBundle:
    dataframe: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.fillna(0.0)


def engineer_features(df: pd.DataFrame, short_window: int, long_window: int) -> FeatureBundle:
    """Create reproducible time-series features for anomaly detection experiments."""
    if "timestamp" not in df.columns:
        raise ValueError("Input dataframe must contain a 'timestamp' column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_columns = ["pressure_flow_ratio","hour_sin","hour_cos"]
    df["pressure_flow_ratio"] = (
        df["pressure_psi"] / df["flow_rate_m3h"].replace(0, np.nan)
    ).bfill().ffill()

    hour_fraction = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction / 24.0)

    for col, prefix in [
        ("pressure_psi", "pressure"),
        ("flow_rate_m3h", "flow"),
        ("temperature_c", "temperature"),
    ]:
        df[f"{prefix}_diff"] = df[col].diff().fillna(0.0)
        df[f"{prefix}_roll_mean_{short_window}"] = df[col].rolling(window=short_window, min_periods=1).mean()
        df[f"{prefix}_roll_std_{short_window}"] = df[col].rolling(window=short_window, min_periods=2).std().fillna(0.0)
        df[f"{prefix}_zscore_{long_window}"] = _rolling_zscore(df[col], window=long_window)
        feature_columns.append(col)
        feature_columns.append(f"{prefix}_diff")
        feature_columns.append(f"{prefix}_roll_mean_{short_window}")
        feature_columns.append(f"{prefix}_roll_std_{short_window}")
        feature_columns.append(f"{prefix}_zscore_{long_window}")

    X = df[feature_columns].copy()
    y = df["is_anomaly"].astype(int).copy()
    return FeatureBundle(dataframe=df, X=X, y=y)


def load_and_engineer_features(csv_path: str, short_window: int, long_window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    bundle = engineer_features(pd.read_csv(csv_path), short_window, long_window)
    return bundle.dataframe, bundle.X, bundle.y
