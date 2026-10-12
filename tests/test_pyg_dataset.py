"""Tests for the PyTorch Geometric graph conversion. No model is trained."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import torch

from slackgnn.dataset.pyg_dataset import (
    build_pyg_data,
    load_pyg_dataset,
    one_hot_node_type,
    save_pyg_dataset,
    summarize_pyg_data,
)
from slackgnn.paths import data_dir


def _graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("a", node_name="a", node_type="input")
    graph.add_node("and1", node_name="and1", node_type="combinational")
    graph.add_node("reg5", node_name="reg5", node_type="register")
    graph.add_node("y", node_name="y", node_type="output")
    graph.add_edge("a", "and1")
    graph.add_edge("and1", "reg5")
    graph.add_edge("reg5", "y")
    return graph


def test_one_hot_node_types() -> None:
    assert one_hot_node_type("input") == [1, 0, 0, 0]
    assert one_hot_node_type("output") == [0, 1, 0, 0]
    assert one_hot_node_type("register") == [0, 0, 1, 0]
    assert one_hot_node_type("combinational") == [0, 0, 0, 1]


def test_pyg_data_uses_directed_edges_and_separate_slack() -> None:
    paths = (
        {"endpoint": "reg5", "startpoint": "a", "slack": -0.75},
        {"endpoint": "y", "startpoint": "reg5", "slack": 0.2},
    )
    data, metadata, errors = build_pyg_data(_graph(), paths)

    assert errors == ()
    assert data is not None
    assert metadata is not None
    assert tuple(data.x.shape) == (4, 4)
    assert torch.equal(data.x[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(data.x[2], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.equal(data.edge_index, torch.tensor([[0, 1, 2], [1, 2, 3]]))
    assert tuple(data.y.shape) == (4,)
    assert torch.isnan(data.y[0])
    assert torch.isnan(data.y[1])
    assert round(float(data.y[2].item()), 2) == -0.75
    assert round(float(data.y[3].item()), 2) == 0.2
    assert data.label_mask.tolist() == [False, False, True, True]
    assert data.negative_slack_mask.tolist() == [False, False, True, False]
    assert data.positive_slack_mask.tolist() == [False, False, False, True]
    assert metadata["node_count"] == 4
    assert metadata["edge_count"] == 3
    assert metadata["feature_dimension"] == 4
    assert metadata["label_count"] == 2
    assert metadata["register_count"] == 1
    assert metadata["combinational_count"] == 1
    assert metadata["positive_slack_count"] == 1
    assert metadata["negative_slack_count"] == 1


def test_duplicate_endpoint_keeps_the_worst_slack() -> None:
    paths = (
        {"endpoint": "reg5", "slack": -0.75},
        {"endpoint": "reg5", "slack": 0.2},
    )
    data, metadata, errors = build_pyg_data(_graph(), paths)
    assert errors == ()
    assert data is not None
    assert metadata is not None
    assert round(float(data.y[2].item()), 2) == -0.75
    assert metadata["label_count"] == 1
    assert metadata["negative_slack_count"] == 1
    assert metadata["positive_slack_count"] == 0


def test_saved_graph_reloads(tmp_path: Path) -> None:
    data, metadata, errors = build_pyg_data(
        _graph(),
        ({"endpoint": "y", "slack": 0.2},),
    )
    assert errors == ()
    assert data is not None
    assert metadata is not None
    assert save_pyg_dataset(data, metadata, tmp_path) == ()
    loaded, loaded_meta, load_errors = load_pyg_dataset(tmp_path)
    assert load_errors == ()
    assert loaded is not None
    assert loaded_meta is not None
    text, summary_errors = summarize_pyg_data(loaded, loaded_meta)
    assert summary_errors == ()
    assert "Graph nodes: 4" in text
    assert "Graph edges: 3" in text
    assert "Feature shape: (4, 4)" in text
    assert "Label shape: (4,)" in text
    assert "graph.pt: loaded" in text
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["label_count"] == 1


def test_gcd_pyg_dataset_matches_the_netlist_and_path_table() -> None:
    from slackgnn.dataset.pyg_dataset import load_networkx_graph, load_path_dataset

    graph, graph_errors = load_networkx_graph(data_dir() / "gcd" / "graphs" / "netlist_graph.pkl")
    paths, path_errors = load_path_dataset(data_dir() / "gcd" / "dataset" / "path_dataset.json")
    data, metadata, errors = build_pyg_data(graph, paths)

    assert graph_errors == ()
    assert path_errors == ()
    assert errors == ()
    assert data is not None
    assert metadata is not None
    assert graph is not None
    assert metadata["node_count"] == 527
    assert metadata["edge_count"] == 928
    assert metadata["feature_dimension"] == 4
    assert metadata["label_count"] == 5
    assert metadata["register_count"] == 35
    assert metadata["combinational_count"] == 438
    assert metadata["positive_slack_count"] == 3
    assert metadata["negative_slack_count"] == 2
    assert int(data.x.sum().item()) == 527
    names = list(graph.nodes)
    endpoint = names.index("dpath.a_reg.out[11]$_DFFE_PP_")
    assert data.label_mask[endpoint].item() is True
    assert round(float(data.y[endpoint].item()), 2) == -1.46
    assert data.negative_slack_mask[endpoint].item() is True
