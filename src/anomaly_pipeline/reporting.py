from __future__ import annotations

import pandas as pd


def metrics_table(results) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
