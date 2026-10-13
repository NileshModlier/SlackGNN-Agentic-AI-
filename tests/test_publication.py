"""Tests for corpus growth and holdout difficulty ranking."""

from __future__ import annotations

from slackgnn.m9.publication import cumulative_growth, rank_holdouts


def test_growth_adds_designs_in_order() -> None:
    designs = [
        {"design": "gcd", "nodes": 10, "edges": 20, "expanded_labels": 2},
        {"design": "aes", "nodes": 5, "edges": 7, "expanded_labels": 1},
    ]

    points = cumulative_growth(designs, ("gcd", "aes"))

    assert points[-1]["nodes"] == 15
    assert points[-1]["edges"] == 27
    assert points[-1]["labels"] == 3


def test_difficulty_uses_defined_f1_only() -> None:
    models = {
        "gcn": {"folds": [{"held_out": "aes", "endpoint": {"f1": 1.0}}, {"held_out": "ibex", "endpoint": {"f1": 0.2}}]},
        "slackgnn": {"folds": [{"held_out": "aes", "endpoint": {"f1": None}}, {"held_out": "ibex", "endpoint": {"f1": 0.4}}]},
    }

    ranking = rank_holdouts(models)

    assert ranking["easiest"] == "aes"
    assert ranking["hardest"] == "ibex"
    assert ranking["easiest_f1"] == 1.0
    assert abs(ranking["hardest_f1"] - 0.3) < 1e-9
