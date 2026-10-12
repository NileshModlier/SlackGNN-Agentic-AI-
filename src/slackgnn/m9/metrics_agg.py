"""Aggregate fold and seed metrics without requiring the published corpus."""

from __future__ import annotations

import math

METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mae",
    "mse",
    "rmse",
)


def aggregate_metric(rows: list[dict[str, object]], key: str) -> dict[str, float | int | None]:
    """Mean and sample spread of one metric, skipping missing values."""
    values = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        value = float(row[key])
        if math.isnan(value) or math.isinf(value):
            continue
        values.append(value)
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    """Aggregate every standard classification and regression metric."""
    return {key: aggregate_metric(rows, key) for key in METRIC_KEYS}


def total_support(rows: list[dict[str, object]]) -> int:
    """Sum fold support. Support is a count, so it is not averaged."""
    return sum(int(row["support"]) for row in rows if row.get("support") is not None)
