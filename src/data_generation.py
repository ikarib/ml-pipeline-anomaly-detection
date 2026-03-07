from __future__ import annotations

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

    pressure[windows["pressure_spike"]] += rng.normal(14, 2, len(range(110, 118)))
    flow[windows["flow_drop"]] -= rng.normal(16, 2.5, len(range(280, 290)))
    temp[windows["temperature_surge"]] += rng.normal(10, 1.0, len(range(505, 514)))
    pressure[windows["pressure_and_flow_divergence"]] += rng.normal(9, 1.5, len(range(640, 651)))
    flow[windows["pressure_and_flow_divergence"]] -= rng.normal(11, 1.7, len(range(640, 651)))

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
    df = make_sample_pipeline_data()

    Path("data").mkdir(exist_ok=True)

    output_path = "data/sample_pipeline_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    main()