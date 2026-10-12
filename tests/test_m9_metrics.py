"""Unit tests for M9 metric aggregation."""

from __future__ import annotations

import math

from slackgnn.m9.metrics_agg import aggregate_metric, aggregate_rows, total_support


def test_aggregate_metric_skips_missing_values() -> None:
    rows = [{"f1": 1.0}, {"f1": None}, {"f1": float("nan")}, {"f1": 3.0}]

    summary = aggregate_metric(rows, "f1")

    assert summary["count"] == 2
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_sample_spread_of_three_seed_means() -> None:
    rows = [{"f1": 0.5}, {"f1": 0.7}, {"f1": 0.9}]

    summary = aggregate_metric(rows, "f1")

    assert math.isclose(summary["mean"], 0.7)
    assert math.isclose(summary["std"], 0.2)
    assert summary["count"] == 3


def test_empty_metric_has_no_mean() -> None:
    summary = aggregate_metric([{"f1": None}], "f1")

    assert summary["mean"] is None
    assert summary["std"] is None
    assert summary["count"] == 0


def test_aggregate_rows_and_support_sum() -> None:
    rows = [
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0, "mae": 0.2, "mse": 0.04, "rmse": 0.2, "support": 4},
        {"accuracy": 0.5, "precision": None, "recall": 0.0, "f1": None, "roc_auc": None, "mae": 0.4, "mse": 0.16, "rmse": 0.4, "support": 2},
    ]

    summary = aggregate_rows(rows)

    assert math.isclose(summary["accuracy"]["mean"], 0.75)
    assert summary["f1"]["count"] == 1
    assert summary["f1"]["mean"] == 1.0
    assert math.isclose(summary["mae"]["mean"], 0.3)
    assert total_support(rows) == 6
