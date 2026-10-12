"""Unit tests for path-slack propagation."""

from __future__ import annotations

import json
import math
import pickle

import networkx as nx
import torch

from slackgnn.dataset.pyg_dataset import build_pyg_data, save_pyg_dataset
from slackgnn.expansion.label_expansion import (
    expand_design,
    merge_path_records,
    propagate_slack,
    restore_endpoint_slack,
    write_expanded_labels,
)


def test_shared_node_keeps_the_worst_slack() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    paths = [
        {"startpoint": "s", "endpoint": "e", "slack": 0.4, "path_group": "clk", "path_type": "min"},
        {"startpoint": "m", "endpoint": "e", "slack": -2.0, "path_group": "clk", "path_type": "max"},
    ]
    slack, routes = propagate_slack(3, edge_index, paths, {"s": 0, "m": 1, "e": 2})

    assert torch.isfinite(slack).all()
    assert math.isclose(float(slack[0].item()), 0.4, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(slack[1].item()), -2.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(slack[2].item()), -2.0, rel_tol=1e-5, abs_tol=1e-5)
    assert routes[0]["route_length"] == 3
    assert routes[0]["disconnected"] is False


def test_published_endpoint_slack_is_restored() -> None:
    propagated = torch.tensor([0.4, -2.0, -2.0])
    original = torch.tensor([float("nan"), float("nan"), 0.4])
    mask = torch.tensor([False, False, True])

    restored = restore_endpoint_slack(propagated, original, mask)

    assert math.isclose(float(restored[0].item()), 0.4, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(restored[1].item()), -2.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(float(restored[2].item()), 0.4, rel_tol=1e-5, abs_tol=1e-5)


def test_report_only_paths_are_added() -> None:
    dataset = [{"startpoint": "a", "endpoint": "c", "path_group": "clk", "path_type": "max", "slack": -1.0}]
    report = [
        {"startpoint": "a", "endpoint": "c", "path_group": "clk", "path_type": "max", "slack": -1.0},
        {"startpoint": "d", "endpoint": "c", "path_group": "clk", "path_type": "min", "slack": 1.0},
    ]

    merged, report_only = merge_path_records(dataset, report)

    assert report_only == 1
    assert len(merged) == 2
    assert merged[0]["source"] == "report+dataset"
    assert merged[1]["source"] == "report"


def test_expand_design_adds_route_nodes_without_rewriting_graph_pt(tmp_path) -> None:
    design = tmp_path / "toy"
    graph = nx.DiGraph()
    graph.add_node("a", node_type="input")
    graph.add_node("b", node_type="combinational")
    graph.add_node("c", node_type="output")
    graph.add_node("d", node_type="combinational")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("a", "d")
    paths = (
        {
            "startpoint": "a",
            "endpoint": "c",
            "path_group": "clk",
            "path_type": "max",
            "arrival_time": 2.0,
            "required_time": 1.0,
            "slack": -1.0,
        },
    )
    data, metadata, errors = build_pyg_data(graph, paths)
    assert errors == ()
    assert data is not None and metadata is not None
    assert save_pyg_dataset(data, metadata, design / "pyg") == ()
    (design / "graphs").mkdir()
    with (design / "graphs" / "netlist_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    (design / "dataset").mkdir()
    (design / "dataset" / "path_dataset.json").write_text(json.dumps({"paths": list(paths)}), encoding="utf-8")
    (design / "reports").mkdir()
    (design / "reports" / "6_finish.rpt").write_text(
        "\n".join(
            [
                "Startpoint: a",
                "Endpoint: c",
                "Path Group: clk",
                "Path Type: max",
                "                                  2.00   data arrival time",
                "                                  1.00   data required time",
                "                                 -2.00   data arrival time",
                "                                 -1.00   slack (VIOLATED)",
                "",
                "Startpoint: d",
                "Endpoint: c",
                "Path Group: clk",
                "Path Type: min",
                "                                  1.50   data arrival time",
                "                                  0.50   data required time",
                "                                 -1.50   data arrival time",
                "                                  1.00   slack (MET)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    graph_path = design / "pyg" / "graph.pt"
    before = graph_path.read_bytes()

    expanded = expand_design(design)

    assert graph_path.read_bytes() == before
    summary = expanded["summary"]
    assert summary["original_label_count"] == 1
    assert summary["expanded_label_count"] == 4
    assert summary["paths_only_in_report"] == 1
    by_name = {row["node_name"]: row for row in expanded["labeled_nodes"]}
    assert by_name["c"]["slack"] == -1.0
    assert by_name["c"]["original_endpoint"] is True
    assert by_name["d"]["slack"] == 1.0
    assert by_name["d"]["original_endpoint"] is False
    assert by_name["b"]["slack"] == -1.0

    json_path, tensor_path = write_expanded_labels(
        {
            "requested_designs": ["toy"],
            "missing_designs": [],
            "designs": {"toy": expanded},
            "summary": {
                "designs": [summary],
                "missing_designs": [],
                "design_count": 1,
                "original_label_count": 1,
                "expanded_label_count": 4,
                "added_label_count": 3,
                "node_count": 4,
                "edge_count": 3,
                "paths_only_in_report": 1,
                "method": "test",
            },
        },
        tmp_path / "expanded_labels.json",
        tmp_path / "expanded_labels.pt",
    )
    assert graph_path.read_bytes() == before
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["summary"]["expanded_label_count"] == 4
    tensors = torch.load(tensor_path, weights_only=False)
    assert int(tensors["designs"]["toy"]["mask"].sum().item()) == 4
