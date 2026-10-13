"""Tests for label thinning, path removal, and robustness ranking."""

from __future__ import annotations

import random

import torch

from slackgnn.m9.robustness import (
    apply_label_noise,
    kept_route_indices,
    rank_models,
    split_one_per_class,
    stratified_subsample,
    supervision_from_routes,
    _run_model,
)
from torch_geometric.data import Data


def test_sparse_sample_keeps_a_balanced_fraction() -> None:
    pairs = [(0, index) for index in range(10)]
    classes = {pair: 0 if pair[1] < 8 else 1 for pair in pairs}
    kept = stratified_subsample(pairs, classes.get, 0.25, random.Random(0))

    assert len([pair for pair in kept if classes[pair] == 0]) == 2
    assert len([pair for pair in kept if classes[pair] == 1]) == 1


def test_split_holds_out_one_pair_per_class() -> None:
    pairs = [(0, index) for index in range(6)]
    classes = {pair: 0 if pair[1] < 3 else 1 for pair in pairs}
    train, validation = split_one_per_class(pairs, classes.get, random.Random(1))

    assert len(validation) == 2
    assert {classes[pair] for pair in validation} == {0, 1}
    assert len(train) + len(validation) == len(pairs)


def test_path_drop_leaves_at_least_one_route() -> None:
    kept = kept_route_indices(4, 0.5, random.Random(0))

    assert len(kept) == 2
    assert len(set(kept)) == 2


def test_dropped_route_removes_its_private_nodes() -> None:
    routes = [
        {"mapped": True, "slack": -0.2, "node_indices": [0, 1]},
        {"mapped": True, "slack": 0.4, "node_indices": [1, 2]},
    ]
    labeled, slack, violation = supervision_from_routes(
        3,
        routes,
        [1],
        torch.tensor([False, False, True]),
        torch.tensor([0.0, 0.0, 0.5]),
    )

    assert labeled.tolist() == [False, True, True]
    assert float(slack[0]) != float(slack[0])
    assert float(slack[2]) == 0.5
    assert int(violation[1].item()) == 0


def test_noise_flips_are_reproducible() -> None:
    data = type("Graph", (), {})()
    data.expanded_violation = torch.tensor([0.0, 1.0, 0.0, 1.0])
    data.expanded_slack = torch.tensor([0.2, -0.3, 0.4, -0.1])
    pairs = [(0, index) for index in range(4)]
    first, sigma = apply_label_noise([data], pairs, 0.5, random.Random(2))
    second, _sigma = apply_label_noise([data], pairs, 0.5, random.Random(2))

    assert sigma > 0
    assert first == second
    assert len(first) == 4


def test_rank_prefers_the_higher_mean_f1() -> None:
    scores, most, least = rank_models(
        {
            "gcn": [{"f1": 0.2, "mae": 1.0}, {"f1": 0.4, "mae": 0.9}],
            "slackgnn": [{"f1": 0.8, "mae": 0.4}, {"f1": 0.7, "mae": 0.5}],
        }
    )

    assert most == "slackgnn"
    assert least == "gcn"
    assert scores["slackgnn"]["mean_f1"] == 0.75


def _toy_graph(name: str, slack: list[float]) -> Data:
    count = len(slack)
    mask = torch.ones(count, dtype=torch.bool)
    values = torch.tensor(slack, dtype=torch.float32)
    violation = (values < 0).to(torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return Data(
        x=torch.eye(4),
        node_features=torch.randn(count, 10),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 10),
        timing_features=torch.randn(count, 3),
        y_slack=values.clone(),
        y_violation=violation.clone(),
        expanded_slack=values.clone(),
        expanded_violation=violation.clone(),
        expanded_mask=mask,
        label_mask=mask,
        design_name=name,
        num_nodes=count,
    )


def test_one_epoch_holdout_returns_endpoint_metrics() -> None:
    graphs = [_toy_graph("a", [0.2, -0.4, 0.1, -0.2]), _toy_graph("b", [0.3, -0.5, 0.2, -0.1])]
    trained = _run_model(
        graphs,
        "slackgnn",
        seed=0,
        prepare=None,
        epochs=1,
        hidden_channels=8,
        patience=1,
        learning_rate=1e-2,
    )

    assert trained["summary"]["endpoint"]["mae"]["count"] == 2
    assert trained["folds"][0]["train_support"] > 0
