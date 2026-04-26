from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


_COUNT_COLUMNS = [
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]
_RATE_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]


def metrics_table(results) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def cv_fold_metrics_table(fold_results: Iterable[tuple[object, Sequence[object]]]) -> pd.DataFrame:
    rows = []
    for split, results in fold_results:
        for result in results:
            row = {
                "fold": split.fold_index,
                "split_strategy": split.split_strategy,
                "train_rows": len(split.train_positions),
                "holdout_rows": len(split.holdout_positions),
                "train_end": split.train_end,
                "holdout_start": split.holdout_start,
                "holdout_end": split.holdout_end,
                "model": result.name,
            }
            row.update(result.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def cv_metrics_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return fold_metrics.copy()

    rows = []
    for model_name, group in fold_metrics.groupby("model", sort=False):
        row = {
            "model": model_name,
            "folds": int(group["fold"].nunique()),
            "holdout_rows": int(group["holdout_rows"].sum()),
        }
        for column in _RATE_COLUMNS:
            if column in group.columns:
                row[f"mean_{column}"] = float(group[column].mean())
                row[f"std_{column}"] = float(group[column].std(ddof=0))
        for column in _COUNT_COLUMNS:
            if column in group.columns:
                row[f"total_{column}"] = int(group[column].sum())
        if all(column in group.columns for column in _COUNT_COLUMNS):
            tp = row["total_true_positive"]
            fp = row["total_false_positive"]
            fn = row["total_false_negative"]
            micro_precision = _safe_divide(tp, tp + fp)
            micro_recall = _safe_divide(tp, tp + fn)
            row["micro_precision"] = micro_precision
            row["micro_recall"] = micro_recall
            row["micro_f1"] = _safe_divide(
                2 * micro_precision * micro_recall,
                micro_precision + micro_recall,
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    sort_column = "micro_f1" if "micro_f1" in summary.columns else "mean_f1"
    if sort_column in summary.columns:
        summary = summary.sort_values(sort_column, ascending=False)
    return summary.reset_index(drop=True)
