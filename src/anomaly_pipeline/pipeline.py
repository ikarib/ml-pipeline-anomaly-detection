from __future__ import annotations

import argparse
import yaml

import matplotlib.pyplot as plt
import pandas as pd

from anomaly_pipeline.feature_engineering import load_and_engineer_features
from anomaly_pipeline.model_training import (
    make_time_holdout_split,
    train_isolation_forest,
    train_random_forest,
    train_xgboost,
    train_autoencoder_anomaly_detector,
)
from anomaly_pipeline.reporting import metrics_table

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_feature_importance(result, output_path: str | None) -> None:
    if not output_path or "feature_importance" not in result.artifacts:
        return
    result.artifacts["feature_importance"].to_csv(output_path, index=False)
    print(f"Saved feature importance table to {output_path}")


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
    evaluation_config = config.get("evaluation", {})
    split = make_time_holdout_split(
        df["timestamp"],
        holdout_size=evaluation_config.get("holdout_size", 0.25),
        n_samples=len(X),
    )
    print(
        "Using chronological holdout: "
        f"train rows={len(split.train_positions)}, "
        f"holdout rows={len(split.holdout_positions)}, "
        f"holdout start={split.holdout_start}"
    )

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

    iso = train_isolation_forest(X, y, options=config["model_isolation_forest"], split=split)
    rf = train_random_forest(X, y, options=config["model_random_forest"], split=split)
    xgb = train_xgboost(X, y, options=config["model_xgboost"], split=split)
    ae = train_autoencoder_anomaly_detector(X, y, options=config["model_autoencoder"], split=split)

    summary = metrics_table([iso, rf, xgb, ae])
    summary.to_csv(metrics_path, index=False)
    print(f"Saved metrics summary table to {metrics_path}")

    _write_feature_importance(rf, config["model_random_forest"].get("feature_importance_path"))
    _write_feature_importance(xgb, config["model_xgboost"].get("feature_importance_path"))

    holdout_df = df.iloc[split.holdout_positions]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(holdout_df["timestamp"], iso.scores, label="Isolation Forest anomaly score")
    ax.set_title("Isolation Forest Scores on Holdout Window")
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
