"""Tests for multi-design corpus assembly. No model is trained."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

from slackgnn.dataset.corpus import (
    build_corpus,
    load_corpus_metadata,
    load_design_manifest,
    render_corpus_summary,
    validate_published_corpus,
)
from slackgnn.dataset.pyg_dataset import build_pyg_data, save_pyg_dataset
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


def _write_manifest(root: Path, names: list[str]) -> Path:
    path = root / "designs.json"
    path.write_text(json.dumps({"designs": names}) + "\n", encoding="utf-8")
    return path


def _write_pyg(design_root: Path) -> None:
    paths = (
        {"endpoint": "reg5", "startpoint": "a", "slack": -0.75},
        {"endpoint": "y", "startpoint": "reg5", "slack": 0.2},
    )
    data, metadata, errors = build_pyg_data(_graph(), paths)
    assert errors == ()
    assert data is not None
    assert metadata is not None
    assert save_pyg_dataset(data, metadata, design_root / "pyg") == ()


def test_manifest_rejects_unsafe_duplicate_and_reserved_names(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, ["gcd", "../aes", "gcd", "corpus", "spi"])
    names, errors = load_design_manifest(manifest)
    assert names == ()
    assert any("not a design name" in message for message in errors)
    assert any("listed more than once" in message for message in errors)
    assert any("reserved" in message for message in errors)


def test_valid_design_is_copied_and_missing_design_is_recorded(tmp_path: Path) -> None:
    _write_pyg(tmp_path / "uart")
    manifest = _write_manifest(tmp_path, ["uart", "aes"])
    corpus = tmp_path / "corpus"
    metadata, errors = build_corpus(tmp_path, manifest, corpus)

    assert errors == ()
    assert metadata is not None
    assert metadata["total_designs"] == 1
    assert metadata["total_nodes"] == 4
    assert metadata["total_edges"] == 3
    assert metadata["total_labels"] == 2
    assert (corpus / "uart" / "graph.pt").is_file()
    assert (corpus / "uart" / "metadata.json").is_file()
    assert not (corpus / "aes").exists()
    failures = metadata["failures"]
    assert isinstance(failures, list)
    assert failures[0]["design_name"] == "aes"
    assert "Missing design directory" in failures[0]["reason"]
    text = render_corpus_summary(metadata)
    assert "Design Count: 1" in text
    assert "Total Nodes: 4" in text
    assert "Total Edges: 3" in text
    assert "Total Labels: 2" in text
    assert "positive_slack_count: 1" in text
    assert "negative_slack_count: 1" in text


def test_graph_and_path_table_are_generated_into_the_corpus(tmp_path: Path) -> None:
    design = tmp_path / "spi"
    (design / "graphs").mkdir(parents=True)
    (design / "dataset").mkdir()
    with (design / "graphs" / "netlist_graph.pkl").open("wb") as handle:
        pickle.dump(_graph(), handle)
    payload = {
        "paths": [
            {"endpoint": "reg5", "slack": -1.0},
            {"endpoint": "y", "slack": 0.5},
        ]
    }
    (design / "dataset" / "path_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = _write_manifest(tmp_path, ["spi"])
    metadata, errors = build_corpus(tmp_path, manifest, tmp_path / "corpus")

    assert errors == ()
    assert metadata is not None
    assert (design / "pyg" / "graph.pt").is_file()
    assert metadata["total_designs"] == 1
    assert metadata["designs"][0]["design_name"] == "spi"
    assert metadata["designs"][0]["label_count"] == 2


def test_zero_node_and_zero_edge_graph_is_a_failure(tmp_path: Path) -> None:
    pyg = tmp_path / "ibex" / "pyg"
    data = Data(
        x=torch.empty((0, 4), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.empty((0,), dtype=torch.float32),
        label_mask=torch.empty((0,), dtype=torch.bool),
        positive_slack_mask=torch.empty((0,), dtype=torch.bool),
        negative_slack_mask=torch.empty((0,), dtype=torch.bool),
    )
    metadata = {
        "node_count": 0,
        "edge_count": 0,
        "feature_dimension": 4,
        "label_count": 0,
        "register_count": 0,
        "combinational_count": 0,
        "positive_slack_count": 0,
        "negative_slack_count": 0,
    }
    assert save_pyg_dataset(data, metadata, pyg) == ()
    manifest = _write_manifest(tmp_path, ["ibex"])
    corpus, errors = build_corpus(tmp_path, manifest, tmp_path / "corpus")

    assert errors == ()
    assert corpus is not None
    assert corpus["total_designs"] == 0
    assert corpus["total_nodes"] == 0
    assert corpus["total_edges"] == 0
    assert not (tmp_path / "corpus" / "ibex").exists()
    assert "node_count" in corpus["failures"][0]["reason"]
    assert "edge_count" in corpus["failures"][0]["reason"]


def test_summary_rejects_a_corpus_whose_files_disappeared(tmp_path: Path) -> None:
    _write_pyg(tmp_path / "gcd")
    manifest = _write_manifest(tmp_path, ["gcd"])
    corpus = tmp_path / "corpus"
    metadata, errors = build_corpus(tmp_path, manifest, corpus)
    assert errors == ()
    assert metadata is not None
    (corpus / "gcd" / "graph.pt").unlink()
    loaded, load_errors = load_corpus_metadata(corpus)
    assert load_errors == ()
    assert loaded is not None
    disk_errors = validate_published_corpus(corpus, loaded)
    assert any("Missing PyG graph" in message for message in disk_errors)


def test_shipped_manifest_publishes_gcd_and_records_absent_designs() -> None:
    metadata, errors = build_corpus()
    assert errors == ()
    assert metadata is not None
    published = {item["design_name"]: item for item in metadata["designs"]}
    assert "gcd" in published
    assert published["gcd"]["node_count"] == 527
    assert published["gcd"]["edge_count"] == 928
    assert published["gcd"]["label_count"] == 5
    assert published["gcd"]["positive_slack_count"] == 3
    assert published["gcd"]["negative_slack_count"] == 2
    assert (data_dir() / "corpus" / "gcd" / "graph.pt").is_file()
    assert (data_dir() / "corpus" / "gcd" / "metadata.json").is_file()
    failed = {item["design_name"] for item in metadata["failures"]}
    for name in ("aes", "ibex", "uart", "spi"):
        if not (data_dir() / name).is_dir():
            assert name in failed
    assert metadata["total_designs"] == len(published)
    assert metadata["total_nodes"] == sum(item["node_count"] for item in published.values())
    assert metadata["total_edges"] == sum(item["edge_count"] for item in published.values())
    assert metadata["total_labels"] == sum(item["label_count"] for item in published.values())
    text = render_corpus_summary(metadata)
    assert "Corpus Overview" in text
    assert "Design Count:" in text
    assert "Total Labels:" in text
