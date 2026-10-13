"""Tests for scaling fits and corpus projections."""

from __future__ import annotations

from slackgnn.m9.scaling import fit_complexity, project_counts, project_lodo_seconds, protocol_training_seconds


def test_edge_linear_runtime_selects_edge_complexity() -> None:
    rows = [
        {"nodes": 100, "edges": 200, "label_count": 4, "train_seconds": 0.002},
        {"nodes": 1000, "edges": 4000, "label_count": 8, "train_seconds": 0.040},
        {"nodes": 5000, "edges": 20000, "label_count": 12, "train_seconds": 0.200},
    ]

    fit = fit_complexity(rows)

    assert fit["best"] == "O(E)"
    assert abs(fit["slope_vs_edges"] - 1.0) < 1e-6


def test_protocol_time_matches_five_training_designs() -> None:
    seconds = protocol_training_seconds(1.5, 6, heads=1)

    assert seconds == 100 * 1.5


def test_projection_scales_with_design_count() -> None:
    assert project_counts(6, 79856, 12) == 79856 * 2
    projected = project_lodo_seconds(1.5, 6, 6, heads=2)

    assert projected == protocol_training_seconds(1.5, 6, heads=2)
