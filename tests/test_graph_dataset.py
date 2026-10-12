"""Tests for joining timing-path labels onto the signal-flow graph."""

from __future__ import annotations

import pickle
from decimal import Decimal
from pathlib import Path

import networkx as nx

from slackgnn.dataset.graph_dataset import (
    join_timing_labels,
    load_di_graph,
    load_labeled_paths,
    validate_labeled_dataset,
    write_labeled_graph,
    write_labeled_paths,
    write_validation_report,
)
from slackgnn.dataset.timing_report import TimingPath
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


def _path(start: str, end: str, slack: str, path_type: str = "max") -> TimingPath:
    return TimingPath(
        startpoint=start,
        endpoint=end,
        path_group="clk",
        path_type=path_type,
        arrival_time=Decimal("2.00"),
        required_time=Decimal("1.25"),
        slack=Decimal(slack),
    )


def test_join_attaches_slack_to_the_endpoint_only() -> None:
    source = _graph()
    labeled, paths, errors = join_timing_labels(source, [_path("a", "reg5", "-0.75")])

    assert errors == ()
    assert labeled is not None
    assert "timing_labels" not in source.nodes["reg5"]
    assert labeled.nodes["reg5"]["has_timing_label"] is True
    assert labeled.nodes["reg5"]["timing_labels"] == [
        {
            "startpoint": "a",
            "path_group": "clk",
            "path_type": "max",
            "arrival_time": "2.00",
            "required_time": "1.25",
            "slack": "-0.75",
        }
    ]
    assert labeled.nodes["and1"]["has_timing_label"] is False
    assert labeled.nodes["and1"]["timing_labels"] == []
    assert paths[0].endpoint_node_type == "register"
    assert paths[0].path_exists is True
    assert set(labeled.edges) == set(source.edges)


def test_join_rejects_a_missing_endpoint_and_a_disconnected_path() -> None:
    graph = _graph()
    _, _, missing = join_timing_labels(graph, [_path("a", "missing", "-1.00")])
    assert any("endpoint is not in the graph: missing" in message for message in missing)

    disconnected = _graph()
    disconnected.remove_edge("and1", "reg5")
    _, _, unreachable = join_timing_labels(disconnected, [_path("a", "reg5", "-1.00")])
    assert any("No directed path from a to reg5" in message for message in unreachable)


def test_validation_passes_for_a_joined_graph(tmp_path: Path) -> None:
    source = _graph()
    timing = [_path("a", "reg5", "-0.75"), _path("reg5", "y", "0.20", path_type="min")]
    labeled, paths, errors = join_timing_labels(source, timing)
    assert errors == ()
    assert labeled is not None

    report = validate_labeled_dataset(source, timing, labeled, paths)
    assert report.passed
    assert report.labeled_paths == 2
    assert report.labeled_nodes == 2
    assert report.paths_connected == 2
    assert report.timing_violations == 1
    assert report.timing_met_paths == 1

    graph_path = tmp_path / "labeled_graph.pkl"
    paths_path = tmp_path / "labeled_paths.csv"
    report_path = tmp_path / "validation_report.json"
    assert write_labeled_graph(labeled, graph_path) == ()
    assert write_labeled_paths(paths, paths_path) == ()
    assert write_validation_report(report, report_path) == ()
    loaded, load_errors = load_di_graph(graph_path)
    reloaded, row_errors = load_labeled_paths(paths_path)
    assert load_errors == ()
    assert row_errors == ()
    assert loaded is not None
    assert loaded.nodes["reg5"]["timing_labels"][0]["slack"] == "-0.75"
    assert reloaded == paths
    assert '"passed": true' in report_path.read_text(encoding="utf-8")


def test_validation_reports_a_tampered_slack() -> None:
    source = _graph()
    timing = [_path("a", "reg5", "-0.75")]
    labeled, paths, errors = join_timing_labels(source, timing)
    assert errors == ()
    assert labeled is not None
    labeled.nodes["reg5"]["timing_labels"][0]["slack"] = "-9.99"

    report = validate_labeled_dataset(source, timing, labeled, paths)
    assert not report.passed
    assert any("timing_labels do not match" in message for message in report.errors)


def test_gcd_labels_join_onto_the_netlist_graph() -> None:
    graph_path = data_dir() / "gcd" / "graphs" / "netlist_graph.pkl"
    labels_path = data_dir() / "gcd" / "parsed" / "path_labels.csv"
    with graph_path.open("rb") as handle:
        source = pickle.load(handle)
    from slackgnn.dataset.summary import load_path_labels

    paths, errors = load_path_labels(labels_path)
    labeled, joined, join_errors = join_timing_labels(source, paths)

    assert errors == ()
    assert join_errors == ()
    assert labeled is not None
    assert len(joined) == 5
    assert all(path.path_exists for path in joined)
    endpoint = labeled.nodes["dpath.a_reg.out[11]$_DFFE_PP_"]
    assert endpoint["has_timing_label"] is True
    assert endpoint["timing_labels"][0]["slack"] == "-1.46"
    assert endpoint["timing_labels"][0]["startpoint"] == "dpath.b_reg.out[3]$_DFFE_PP_"
    assert labeled.nodes["resp_msg[15]"]["timing_labels"][0]["slack"] == "-1.31"
    labeled_nodes = sum(1 for _, data in labeled.nodes(data=True) if data["has_timing_label"])
    assert labeled_nodes == 5
    report = validate_labeled_dataset(source, paths, labeled, joined)
    assert report.passed
    assert report.timing_violations == 2
    assert report.timing_met_paths == 3
