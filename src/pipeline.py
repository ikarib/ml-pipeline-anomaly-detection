from __future__ import annotations

import argparse
import yaml

import matplotlib.pyplot as plt
import pandas as pd

from src.feature_engineering import load_and_engineer_features

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
    figure_path = config["output"]["figure_path"]

    df, X, y = load_and_engineer_features(str(data_path), short_window, long_window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["timestamp"]), df["pressure_psi"], label="Pressure")
    ax.scatter(
        pd.to_datetime(df.loc[y == 1, "timestamp"]),
        df.loc[y == 1, "pressure_psi"],
        marker="x",
        label="True anomalies",
    )
    ax.set_title("Pressure series with labeled anomalies")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Pressure (psi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
