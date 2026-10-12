"""Tests for ORFS design discovery and corpus expansion guards."""

from __future__ import annotations

import json

import pytest

from slackgnn.expansion.discover import discover_designs, select_design, write_available_designs
from slackgnn.expansion.expand import expand_design


def _config(directory, platform: str, design: str, top: str, verilog_name: str = "top.v") -> None:
    folder = directory / "flow" / "designs" / platform / design
    folder.mkdir(parents=True)
    (folder / "config.mk").write_text(
        f"export DESIGN_NAME = {top}\nexport PLATFORM = {platform}\nexport CORE_UTILIZATION = 40\n",
        encoding="utf-8",
    )
    source = directory / "flow" / "designs" / "src" / design
    source.mkdir(parents=True, exist_ok=True)
    (source / verilog_name).write_text("module top; endmodule\n" * 100, encoding="utf-8")


def test_discovery_reports_top_module_platform_and_complexity(tmp_path) -> None:
    _config(tmp_path, "sky130hd", "uart", "uart_top")
    _config(tmp_path, "asap7", "spi", "spi_top")
    _config(tmp_path, "nangate45", "gcd", "gcd")

    report = discover_designs(tmp_path)

    uart = select_design(report, "uart")
    assert uart["top_module"] == "uart_top"
    assert uart["platform"] == "sky130hd"
    assert uart["estimated_complexity"] == "tiny"
    assert uart["finish_artifacts_present"] is False
    assert report["missing_priority"] == ["picorv32", "serv", "tinyRocket"]
    assert select_design(report, "spi", "asap7")["top_module"] == "spi_top"


def test_platform_preference_uses_sky130hd_before_asap7(tmp_path) -> None:
    _config(tmp_path, "asap7", "uart", "uart_asap")
    _config(tmp_path, "sky130hd", "uart", "uart_sky")

    selected = select_design(discover_designs(tmp_path), "uart")

    assert selected["platform"] == "sky130hd"
    assert selected["top_module"] == "uart_sky"


def test_indented_flow_variant_does_not_replace_the_top_module(tmp_path) -> None:
    folder = tmp_path / "flow" / "designs" / "asap7" / "aes"
    folder.mkdir(parents=True)
    (folder / "config.mk").write_text(
        "export DESIGN_NAME = aes_cipher_top\n"
        "ifeq ($(FLOW_VARIANT),blackbox)\n"
        "\texport DESIGN_NAME = aes_key_expand_128\n"
        "endif\n",
        encoding="utf-8",
    )

    selected = select_design(discover_designs(tmp_path), "aes", "asap7")

    assert selected["top_module"] == "aes_cipher_top"


def test_available_designs_report_is_json(tmp_path) -> None:
    _config(tmp_path, "sky130hd", "picorv32", "picorv32")

    path = write_available_designs(discover_designs(tmp_path), tmp_path / "available_designs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["designs"][0]["design_name"] == "picorv32"
    assert payload["designs"][0]["priority_rank"] == 2


def test_expand_refuses_to_replace_the_published_corpus(tmp_path) -> None:
    _config(tmp_path, "sky130hd", "gcd", "gcd")

    status = expand_design("gcd", orfs_root=tmp_path, run_flow=False)

    assert status == 1


def test_expand_imports_existing_finish_files_without_running_orfs(tmp_path, monkeypatch) -> None:
    _config(tmp_path, "asap7", "uart", "uart")
    results = tmp_path / "flow" / "results" / "asap7" / "uart" / "base"
    reports = tmp_path / "flow" / "reports" / "asap7" / "uart" / "base"
    results.mkdir(parents=True)
    reports.mkdir(parents=True)
    for name in ("6_final.v", "6_final.sdc", "6_final.spef"):
        (results / name).write_text(name + "\n", encoding="utf-8")
    (reports / "6_finish.rpt").write_text("finish\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_python(script: str, *args: str) -> int:
        calls.append([script, *args])
        return 0

    monkeypatch.setattr("slackgnn.expansion.expand._run_python", fake_python)
    monkeypatch.setattr("slackgnn.expansion.expand.data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr("slackgnn.expansion.expand.write_available_designs", lambda report, destination=None: tmp_path / "available.json")

    status = expand_design("uart", orfs_root=tmp_path, run_flow=False)

    assert status == 0
    assert (tmp_path / "data" / "uart" / "netlist" / "6_final.v").read_text(encoding="utf-8") == "6_final.v\n"
    assert (tmp_path / "data" / "uart" / "reports" / "6_finish.rpt").is_file()
    assert calls == [["scripts/process_design.py", "uart"], ["scripts/build_corpus.py"]]
