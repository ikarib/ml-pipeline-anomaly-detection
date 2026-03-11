from __future__ import annotations

import argparse
import yaml

import matplotlib.pyplot as plt
import pandas as pd

from anomaly_pipeline.feature_engineering import load_and_engineer_features
from anomaly_pipeline.model_training import (
    train_isolation_forest,
    train_random_forest,
    train_autoencoder_anomaly_detector,
)
from anomaly_pipeline.reporting import metrics_table

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    data_path = config["data"]["input_path"]
    short_window = config["features"]["short_rolling_window"]
    long_window = config["features"]["long_rolling_window"]
    true_anomalies_path = config["output"]["true_anomalies_path"]
    isolation_forest_scores_path = config["output"]["isolation_forest_scores_path"]
    metrics_path = config["output"]["metrics_path"]

    df, X, y = load_and_engineer_features(str(data_path), short_window, long_window)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["pressure_psi"], label="Pressure (psi)")
    ax.scatter(
        df.loc[y == 1, "timestamp"],
        df.loc[y == 1, "pressure_psi"],
        marker="x",
        label="True anomaly",
    )
    ax.set_title("Pressure with labeled anomalies")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Pressure (psi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(true_anomalies_path)
    plt.close(fig)
    print(f"Saved pressure figure to {true_anomalies_path}")

    iso = train_isolation_forest(X, y, options=config["model_isolation_forest"])
    rf = train_random_forest(X, y, options=config["model_random_forest"])
    ae = train_autoencoder_anomaly_detector(X, y, options=config["model_autoencoder"])

    summary = metrics_table([iso, rf, ae])
    summary.to_csv(metrics_path, index=False)
    print(f"Saved metrics summary table to {metrics_path}")

    output_path = config['model_random_forest']['feature_importance_path']
    rf.artifacts["feature_importance"].to_csv(output_path, index=False)
    print(f"Saved feature importance table to {output_path}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], iso.scores, label="Isolation Forest anomaly score")
    ax.set_title("Isolation Forest Scores Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Anomaly score")
    fig.tight_layout()
    fig.savefig(isolation_forest_scores_path)
    plt.close(fig)
    print(f"Saved isolation forest scores figure to {isolation_forest_scores_path}")

    output_path = config["model_autoencoder"]["training_history_path"]
    pd.DataFrame(
        {
            "epoch": range(1, len(ae.artifacts["history"]) + 1),
            "train_loss": ae.artifacts["history"],
        }
    ).to_csv(output_path, index=False)
    print(f"Saved training history to {output_path}")

    output_path = config["model_autoencoder"]["threshold_sweep_path"]
    threshold_sweep = ae.artifacts["pr_table"].copy()
    threshold_sweep.to_csv(output_path, index=False)
    print(f"Saved threshold sweep to {output_path}")

if __name__ == "__main__":
    main()
