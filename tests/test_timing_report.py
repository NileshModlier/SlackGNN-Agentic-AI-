"""Tests for OpenROAD path-label parsing and the dataset summary."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from slackgnn.dataset.summary import load_path_labels, summarize_paths, write_dataset_summary
from slackgnn.dataset.timing_report import parse_timing_report, parse_timing_report_text, write_path_labels
from slackgnn.paths import data_dir

_MIN_AND_MAX = """
Startpoint: in_a (input port clocked by vclk)
Endpoint: ff1
          (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: min
                                  1.00   data arrival time
                                  0.40   data required time
                                  0.40   data required time
                                 -1.00   data arrival time
                                  0.60   slack (MET)

Startpoint: ff1
            (rising edge-triggered flip-flop clocked by clk)
Endpoint: ff2 (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max
                                  2.00   data arrival time
                                  1.25   data required time
                                  1.25   data required time
                                 -2.00   data arrival time
                                 -0.75   slack (VIOLATED)

Startpoint: ff1
            (rising edge-triggered flip-flop clocked by clk)
Endpoint: ff2 (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max
                                  2.00   data arrival time
                                  1.25   data required time
                                 -0.75   slack (VIOLATED)
"""


def test_parser_reads_min_and_max_paths_and_collapses_reprints() -> None:
    result = parse_timing_report_text(_MIN_AND_MAX)

    assert result.ok
    assert result.path_blocks == 3
    assert len(result.paths) == 2
    hold, setup = result.paths
    assert hold.startpoint == "in_a"
    assert hold.endpoint == "ff1"
    assert hold.path_group == "clk"
    assert hold.path_type == "min"
    assert hold.arrival_time == Decimal("1.00")
    assert hold.required_time == Decimal("0.40")
    assert hold.slack == Decimal("0.60")
    assert setup.slack == Decimal("-0.75")
    assert setup.path_type == "max"


def test_parser_rejects_a_violated_tag_on_positive_slack() -> None:
    text = _MIN_AND_MAX.replace("slack (MET)", "slack (VIOLATED)", 1)
    result = parse_timing_report_text(text)
    assert not result.ok
    assert any("marked VIOLATED" in message for message in result.errors)


def test_parser_keeps_a_multiword_path_group() -> None:
    text = """
Startpoint: latch1
            (negative level-sensitive latch clocked by clk)
Endpoint: gate1 (rising clock gating-check end-point clocked by clk)
Path Group: gated clock
Path Type: min
                                  5.55   data arrival time
                                  4.26   data required time
                                  4.26   data required time
                                 -5.55   data arrival time
                                  1.30   slack (MET)
"""
    result = parse_timing_report_text(text)
    assert result.ok
    assert result.paths[0].path_group == "gated clock"
    assert result.paths[0].startpoint == "latch1"
    assert result.paths[0].endpoint == "gate1"
    assert result.paths[0].slack == Decimal("1.30")


def test_parser_rejects_a_block_without_slack() -> None:
    result = parse_timing_report_text("Startpoint: ff1\nEndpoint: ff2\nPath Group: clk\nPath Type: max\n")
    assert result.paths == ()
    assert any("missing slack" in message for message in result.errors)


def test_parser_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.rpt"
    result = parse_timing_report(missing)
    assert not result.ok
    assert "Cannot read timing report" in result.errors[0]


def test_write_and_summarize_path_labels(tmp_path: Path) -> None:
    parsed = parse_timing_report_text(_MIN_AND_MAX)
    csv_path = tmp_path / "parsed" / "path_labels.csv"
    json_path = tmp_path / "parsed" / "dataset_summary.json"

    assert write_path_labels(parsed.paths, csv_path) == ()
    loaded, errors = load_path_labels(csv_path)
    assert errors == ()
    summary = summarize_paths(loaded)
    assert summary.total_paths == 2
    assert summary.timing_violations == 1
    assert summary.timing_met_paths == 1
    assert summary.worst_slack == Decimal("-0.75")
    assert summary.average_slack == Decimal("-0.0750")
    assert write_dataset_summary(summary, json_path) == ()
    assert '"worst_slack": -0.75' in json_path.read_text(encoding="utf-8")


def test_gcd_finish_report_labels() -> None:
    report = data_dir() / "gcd" / "reports" / "6_finish.rpt"
    result = parse_timing_report(report)

    assert result.ok
    assert result.path_blocks == 8
    assert [(path.startpoint, path.endpoint, path.path_type, path.slack) for path in result.paths] == [
        ("resp_rdy", "ctrl.state.out[0]$_DFF_P_", "min", Decimal("0.53")),
        ("ctrl.state.out[1]$_DFF_P_", "resp_val", "min", Decimal("0.82")),
        ("dpath.b_reg.out[3]$_DFFE_PP_", "dpath.a_reg.out[11]$_DFFE_PP_", "max", Decimal("-1.46")),
        ("dpath.b_reg.out[3]$_DFFE_PP_", "resp_msg[15]", "max", Decimal("-1.31")),
        ("dpath.a_reg.out[0]$_DFFE_PP_", "dpath.b_reg.out[0]$_DFFE_PP_", "min", Decimal("0.63")),
    ]
    summary = summarize_paths(result.paths)
    assert summary.total_paths == 5
    assert summary.timing_violations == 2
    assert summary.timing_met_paths == 3
    assert summary.worst_slack == Decimal("-1.46")
    assert summary.average_slack == Decimal("-0.1580")
