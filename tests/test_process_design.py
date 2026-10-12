"""Tests for single-design dataset generation. No model is trained."""

from __future__ import annotations

import json
from pathlib import Path

from slackgnn.dataset.corpus import load_corpus_metadata, load_design_manifest
from slackgnn.dataset.process_design import process_design

_NETLIST = """
module widget (a, b, clk, y);
  input a;
  input b;
  input clk;
  output y;
  wire n1;
  wire n2;
  sky130_fd_sc_hd__and2_1 and1 (.A(a), .B(b), .X(n1));
  sky130_fd_sc_hd__dfxtp_1 reg5 (.D(n1), .CLK(clk), .Q(n2));
  sky130_fd_sc_hd__buf_1 buf1 (.A(n2), .X(y));
endmodule
"""

_REPORT = """
Startpoint: a (input port clocked by vclk)
Endpoint: reg5
          (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max
                                  2.00   data arrival time
                                  1.25   data required time
                                  1.25   data required time
                                 -2.00   data arrival time
                                 -0.75   slack (VIOLATED)

Startpoint: reg5
            (rising edge-triggered flip-flop clocked by clk)
Endpoint: y (output port clocked by vclk)
Path Group: clk
Path Type: min
                                  1.00   data arrival time
                                  0.40   data required time
                                  0.40   data required time
                                 -1.00   data arrival time
                                  0.60   slack (MET)
"""


def _plant(root: Path, name: str = "widget") -> None:
    design = root / name
    (design / "netlist").mkdir(parents=True)
    (design / "reports").mkdir()
    (design / "netlist" / "6_final.v").write_text(_NETLIST, encoding="utf-8")
    (design / "reports" / "6_finish.rpt").write_text(_REPORT, encoding="utf-8")


def test_new_design_is_processed_and_published(tmp_path: Path) -> None:
    _plant(tmp_path)
    manifest = tmp_path / "designs.json"
    manifest.write_text(json.dumps({"designs": ["other"]}) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus"

    report, code = process_design("widget", tmp_path, manifest, corpus)
    assert code == 0
    assert report["status"] == "success"
    assert (tmp_path / "widget" / "parsed" / "path_labels.csv").is_file()
    assert (tmp_path / "widget" / "parsed" / "dataset_summary.json").is_file()
    assert (tmp_path / "widget" / "graphs" / "netlist_graph.pkl").is_file()
    assert (tmp_path / "widget" / "dataset" / "labeled_graph.pkl").is_file()
    assert (tmp_path / "widget" / "dataset" / "path_dataset.json").is_file()
    assert (tmp_path / "widget" / "pyg" / "graph.pt").is_file()
    assert (tmp_path / "widget" / "pyg" / "metadata.json").is_file()
    assert (tmp_path / "widget" / "process_report.json").is_file()
    assert (corpus / "widget" / "graph.pt").is_file()

    names, errors = load_design_manifest(manifest)
    assert errors == ()
    assert names == ("other", "widget")
    metadata, load_errors = load_corpus_metadata(corpus)
    assert load_errors == ()
    assert metadata is not None
    published = {item["design_name"]: item for item in metadata["designs"]}
    assert published["widget"]["node_count"] == 7
    assert published["widget"]["edge_count"] == 6
    assert published["widget"]["label_count"] == 2
    assert published["widget"]["positive_slack_count"] == 1
    assert published["widget"]["negative_slack_count"] == 1
    assert metadata["total_designs"] == 1
    assert {item["design_name"] for item in metadata["failures"]} == {"other"}

    again, again_code = process_design("widget", tmp_path, manifest, corpus)
    assert again_code == 0
    assert again["status"] == "success"
    names, errors = load_design_manifest(manifest)
    assert errors == ()
    assert names == ("other", "widget")


def test_missing_design_is_recorded_without_creating_it(tmp_path: Path) -> None:
    manifest = tmp_path / "designs.json"
    manifest.write_text(json.dumps({"designs": ["widget"]}) + "\n", encoding="utf-8")
    report, code = process_design("widget", tmp_path, manifest, tmp_path / "corpus")

    assert code == 1
    assert report["status"] == "failed"
    assert "Missing design directory" in str(report["stages"][0]["detail"])
    assert not (tmp_path / "widget").exists()
    metadata, errors = load_corpus_metadata(tmp_path / "corpus")
    assert errors == ()
    assert metadata is not None
    assert metadata["total_designs"] == 0
    assert metadata["failures"][0]["design_name"] == "widget"


def test_unsafe_design_name_is_rejected(tmp_path: Path) -> None:
    report, code = process_design("../aes", tmp_path, tmp_path / "designs.json", tmp_path / "corpus")
    assert code == 1
    assert report["status"] == "failed"
    assert not (tmp_path / "corpus" / "corpus_metadata.json").exists()
