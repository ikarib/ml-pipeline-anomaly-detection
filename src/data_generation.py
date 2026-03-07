from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
from pathlib import Path


def make_sample_pipeline_data(
    periods: int = 720,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=periods, freq="h")
    t = np.arange(periods)

    pressure = 102 + 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.9, periods)
    flow = 60 + 3.5 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 1.2, periods)
    temp = 15 + 6 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 0.7, periods)

    # Mild regime shift to make the series less toy-like.
    pressure[240:420] += 1.4
    flow[420:560] -= 1.1

    is_anomaly = np.zeros(periods, dtype=int)
    anomaly_type = np.array(["normal"] * periods, dtype=object)

    windows = {
        "pressure_spike": slice(110, 118),
        "flow_drop": slice(280, 290),
        "temperature_surge": slice(505, 514),
        "pressure_and_flow_divergence": slice(640, 651),
    }

    pressure[windows["pressure_spike"]] += rng.normal(14, 2, len(range(*windows["pressure_spike"].indices(periods))))
    flow[windows["flow_drop"]] -= rng.normal(16, 2.5, len(range(*windows["flow_drop"].indices(periods))))
    temp[windows["temperature_surge"]] += rng.normal(10, 1.0, len(range(*windows["temperature_surge"].indices(periods))))
    pressure[windows["pressure_and_flow_divergence"]] += rng.normal(9, 1.5, len(range(*windows["pressure_and_flow_divergence"].indices(periods))))
    flow[windows["pressure_and_flow_divergence"]] -= rng.normal(11, 1.7, len(range(*windows["pressure_and_flow_divergence"].indices(periods))))

    for name, slc in windows.items():
        is_anomaly[slc] = 1
        anomaly_type[slc] = name

    return pd.DataFrame(
        {
            "timestamp": ts,
            "pressure_psi": pressure.round(3),
            "flow_rate_m3h": flow.round(3),
            "temperature_c": temp.round(3),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=720, help="Number of hourly periods to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", required=True, help="Path to save the generated dataset CSV")
    args = parser.parse_args()

    df = make_sample_pipeline_data(periods=args.periods, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Saved dataset to {args.output}")

if __name__ == "__main__":
    main()