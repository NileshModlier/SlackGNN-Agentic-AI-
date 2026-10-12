"""Tests for corpus checks, dataset assembly, model forward passes, and metrics."""

from __future__ import annotations

import json

import pytest
import torch
from torch_geometric.data import Data

from slackgnn.training.dataset_builder import build_training_dataset, load_training_dataset
from slackgnn.training.metrics import classification_metrics, regression_metrics, roc_auc
from slackgnn.training.models import build_model
from slackgnn.training.splits import leave_one_design_out, random_split
from slackgnn.training.validate_corpus import validate_corpus, write_validation_report


def _graph(name: str, slack: list[float]) -> Data:
    nodes = len(slack)
    data = Data(
        x=torch.eye(4)[:nodes],
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long) if nodes > 1 else torch.empty(2, 0, dtype=torch.long),
        y=torch.tensor(slack, dtype=torch.float32),
        label_mask=torch.ones(nodes, dtype=torch.bool),
        positive_slack_mask=torch.tensor([value > 0 for value in slack]),
        negative_slack_mask=torch.tensor([value < 0 for value in slack]),
    )
    data.design_name = name
    data.y_slack = data.y.clone()
    data.y_violation = data.negative_slack_mask.to(torch.float32)
    return data


def test_classification_metrics_score_the_violation_class() -> None:
    scores = classification_metrics([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2])

    assert scores["accuracy"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["roc_auc"] == 1.0


def test_roc_auc_is_undefined_when_one_class_is_missing() -> None:
    assert roc_auc([1, 1], [0.2, 0.8]) is None


def test_regression_metrics_match_absolute_and_squared_error() -> None:
    scores = regression_metrics([1.0, -1.0], [2.0, -1.0])

    assert scores["mae"] == pytest.approx(0.5)
    assert scores["mse"] == pytest.approx(0.5)
    assert scores["rmse"] == pytest.approx(0.5**0.5)


def test_leave_one_design_out_rotates_every_design() -> None:
    graphs = [_graph("gcd", [0.2, -0.4]), _graph("aes", [0.1, -0.2, 0.3])]

    folds = leave_one_design_out(graphs, seed=0)

    assert [fold["held_out"] for fold in folds] == ["gcd", "aes"]
    assert {pair[0] for pair in folds[0]["test"]} == {0}
    assert {pair[0] for pair in folds[0]["train"]} == {1}


def test_random_split_keeps_train_validation_and_test_disjoint() -> None:
    graphs = [_graph("gcd", [0.2, -0.4, 0.1, -0.2, 0.3, -0.1])]

    split = random_split(graphs, seed=0)
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == {(0, node) for node in range(6)}


def test_each_baseline_emits_one_prediction_per_node() -> None:
    data = _graph("gcd", [0.2, -0.4])

    for name in ("gcn", "graphsage", "gin"):
        model = build_model(name, in_channels=4, hidden_channels=8, dropout=0.0)
        prediction = model(data.x, data.edge_index)
        assert prediction.shape == (2,)


def test_validation_report_checks_a_saved_graph(tmp_path) -> None:
    design = tmp_path / "toy" / "pyg"
    design.mkdir(parents=True)
    data = Data(
        x=torch.eye(4),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=torch.tensor([float("nan"), 0.4, -0.2, float("nan")]),
        label_mask=torch.tensor([False, True, True, False]),
        positive_slack_mask=torch.tensor([False, True, False, False]),
        negative_slack_mask=torch.tensor([False, False, True, False]),
    )
    torch.save(data, design / "graph.pt")
    metadata = {
        "node_count": 4,
        "edge_count": 2,
        "label_count": 2,
        "positive_slack_count": 1,
        "negative_slack_count": 1,
        "feature_dimension": 4,
        "register_count": 1,
        "combinational_count": 1,
    }
    (design / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_corpus(tmp_path)

    assert report["passed"] is True
    assert report["designs"] == ["toy"]
    assert report["label_totals"] == 2
    assert report["positive_labels"] == 1
    assert report["negative_labels"] == 1
    destination = write_validation_report(report, tmp_path / "corpus_validation_report.json")
    assert json.loads(destination.read_text(encoding="utf-8"))["node_totals"] == 4


def test_dataset_builder_merges_discovered_designs(tmp_path) -> None:
    for name, slack in (("alpha", [0.5, -0.1]), ("beta", [-0.2])):
        pyg = tmp_path / name / "pyg"
        pyg.mkdir(parents=True)
        nodes = len(slack)
        data = Data(
            x=torch.eye(4)[:nodes],
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            y=torch.tensor(slack, dtype=torch.float32),
            label_mask=torch.ones(nodes, dtype=torch.bool),
            positive_slack_mask=torch.tensor([value > 0 for value in slack]),
            negative_slack_mask=torch.tensor([value < 0 for value in slack]),
        )
        torch.save(data, pyg / "graph.pt")
        (pyg / "metadata.json").write_text(
            json.dumps(
                {
                    "node_count": nodes,
                    "edge_count": 1,
                    "label_count": nodes,
                    "positive_slack_count": sum(value > 0 for value in slack),
                    "negative_slack_count": sum(value < 0 for value in slack),
                    "register_count": 0,
                    "combinational_count": nodes,
                }
            ),
            encoding="utf-8",
        )

    summary, errors = build_training_dataset(
        tmp_path,
        tmp_path / "training_data.pt",
        tmp_path / "training_dataset_summary.json",
    )
    graphs, stored, load_errors = load_training_dataset(tmp_path / "training_data.pt")

    assert errors == ()
    assert load_errors == ()
    assert summary["designs"] == ["alpha", "beta"]
    assert summary["label_count"] == 3
    assert summary["violation_count"] == 2
    assert [graph.design_name for graph in graphs] == ["alpha", "beta"]
    assert stored["met_count"] == 1
