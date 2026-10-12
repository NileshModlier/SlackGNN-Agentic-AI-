"""Tests for SlackGNN features, attention, and explainability."""

from __future__ import annotations

import json
import pickle

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from slackgnn.explainability.rankings import explain_design
from slackgnn.features.feature_builder import enrich_design
from slackgnn.training.models import build_model
from slackgnn.training.models.slackgnn import SlackGNN, grouped_softmax


def _toy_design(root):
    graph = nx.DiGraph()
    graph.add_node("in", node_name="in", node_type="input")
    graph.add_node("gate", node_name="gate", node_type="combinational")
    graph.add_node("ff", node_name="ff", node_type="register")
    graph.add_node("out", node_name="out", node_type="output")
    graph.add_edge("in", "gate")
    graph.add_edge("gate", "ff")
    graph.add_edge("ff", "out")
    design = root / "toy"
    (design / "graphs").mkdir(parents=True)
    (design / "dataset").mkdir()
    with (design / "graphs" / "netlist_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    (design / "dataset" / "path_dataset.json").write_text(
        json.dumps(
            {
                "paths": [
                    {
                        "startpoint": "in",
                        "endpoint": "out",
                        "slack": -0.4,
                        "arrival_time": 1.2,
                        "required_time": 0.8,
                        "path_group": "clk",
                        "path_type": "max",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    data = Data(
        x=torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        y=torch.tensor([float("nan"), float("nan"), float("nan"), -0.4]),
        label_mask=torch.tensor([False, False, False, True]),
        positive_slack_mask=torch.tensor([False, False, False, False]),
        negative_slack_mask=torch.tensor([False, False, False, True]),
    )
    data.design_name = "toy"
    data.y_slack = data.y.clone()
    data.y_violation = data.negative_slack_mask.to(torch.float32)
    return design, data


def test_feature_builder_normalizes_structure_and_withholds_slack(tmp_path) -> None:
    design, data = _toy_design(tmp_path)

    enriched, detail, errors = enrich_design(data, design)

    assert errors == ()
    assert detail["max_logic_depth"] == 1
    assert enriched.node_features.shape == (4, 10)
    assert enriched.edge_attr.shape == (3, 10)
    assert enriched.timing_features.shape == (4, 3)
    assert float(enriched.timing_features[3, 1].item()) == 1.0
    assert float(enriched.arrival_time[3].item()) == pytest.approx(1.2)
    assert not torch.allclose(enriched.timing_features[:, 0], torch.nan_to_num(enriched.y_slack))


def test_grouped_softmax_sums_to_one_for_each_load() -> None:
    scores = torch.tensor([1.0, 3.0, -2.0])
    index = torch.tensor([1, 1, 0])

    attention = grouped_softmax(scores, index, size=3)

    assert torch.allclose(attention[:2].sum(), torch.tensor(1.0))
    assert torch.allclose(attention[2], torch.tensor(1.0))


def test_slackgnn_returns_logits_slack_attention_and_embeddings() -> None:
    model = build_model("slackgnn", in_channels=10, hidden_channels=8, dropout=0.0, edge_channels=10, timing_channels=3)
    assert isinstance(model, SlackGNN)
    logits, slack, attention, embedding = model(
        torch.randn(4, 10),
        torch.tensor([[0, 1, 2], [1, 2, 3]]),
        torch.randn(3, 10),
        torch.rand(4, 3),
    )

    assert logits.shape == (4,)
    assert slack.shape == (4,)
    assert attention.shape == (3,)
    assert embedding.shape == (4, 8)
    assert torch.isfinite(logits).all()


def test_multitask_loss_is_finite_and_backpropagates(tmp_path) -> None:
    _design, data = _toy_design(tmp_path)
    enriched, _detail, errors = enrich_design(data, _design)
    assert errors == ()
    model = SlackGNN(10, 10, 3, hidden_channels=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    index = torch.tensor([3])
    before = None
    for _step in range(5):
        optimizer.zero_grad()
        logits, slack, _attention, _embedding = model(
            enriched.node_features, enriched.edge_index, enriched.edge_attr, enriched.timing_features
        )
        classification = torch.nn.functional.binary_cross_entropy_with_logits(logits[index], enriched.y_violation[index])
        regression = torch.nn.functional.smooth_l1_loss(slack[index], enriched.y_slack[index])
        loss = classification + 0.5 * regression
        before = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert before is not None
    assert before == before


def test_explainability_ranks_the_reported_path(tmp_path) -> None:
    design, data = _toy_design(tmp_path)
    enriched, _detail, errors = enrich_design(data, design)
    assert errors == ()
    attention = torch.tensor([0.2, 0.5, 0.9])

    report = explain_design(enriched, attention, limit=4)

    assert report["influential_paths"][0]["endpoint"] == "out"
    assert report["influential_paths"][0]["hop_count"] == 3
    assert report["attention_heatmap"]["nodes"]
    assert any(row["node"] == "ff" for row in report["node_importance"])
