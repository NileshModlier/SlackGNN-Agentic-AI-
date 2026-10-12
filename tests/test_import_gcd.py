"""Tests for the GCD dataset importer. No learning code is involved."""

from __future__ import annotations

from pathlib import Path

from slackgnn.dataset.gcd_import import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_RESULTS_DIR,
    _decode_wsl_listing,
    import_gcd_dataset,
    main,
)

_RESULT_NAMES = ("6_final.v", "6_final.sdc", "6_final.spef")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    reports = root / "reports"
    results.mkdir()
    reports.mkdir()
    for name in _RESULT_NAMES:
        (results / name).write_text(f"payload {name}\n", encoding="utf-8")
    (reports / "6_finish.rpt").write_text("worst slack max -1.46\n", encoding="utf-8")
    (reports / "5_route_drc.rpt-5.rpt").write_text("violation type: Short\n", encoding="utf-8")
    (reports / "final_worst_path.webp.png").write_bytes(b"png")
    (reports / "synth_stat.txt").write_text("synth\n", encoding="utf-8")
    return results, reports


def test_decode_wsl_listing_reads_utf16_distro_names() -> None:
    raw = "\ufeffUbuntu\n".encode("utf-16-le")
    assert _decode_wsl_listing(raw) == ["Ubuntu"]
    assert _decode_wsl_listing(b"") == []


def test_default_sources_match_openroad_gcd_flow() -> None:
    assert DEFAULT_RESULTS_DIR.endswith("/results/sky130hd/gcd/base")
    assert DEFAULT_REPORTS_DIR.endswith("/reports/sky130hd/gcd/base")
    assert DEFAULT_RESULTS_DIR.startswith("/home/administrator/OpenROAD-flow-scripts/flow/")
    assert DEFAULT_REPORTS_DIR.startswith("/home/administrator/OpenROAD-flow-scripts/flow/")


def test_import_copies_netlist_constraints_spef_and_reports(tmp_path: Path) -> None:
    results, reports = _write_fixture(tmp_path)
    destination = tmp_path / "data" / "gcd"

    outcome = import_gcd_dataset(results, reports, destination)

    assert outcome.ok
    assert (destination / "netlist" / "6_final.v").read_text(encoding="utf-8") == "payload 6_final.v\n"
    assert (destination / "sdc" / "6_final.sdc").read_text(encoding="utf-8") == "payload 6_final.sdc\n"
    assert (destination / "spef" / "6_final.spef").read_text(encoding="utf-8") == "payload 6_final.spef\n"
    report_names = sorted(path.name for path in (destination / "reports").iterdir())
    assert report_names == ["5_route_drc.rpt-5.rpt", "6_finish.rpt"]


def test_import_reports_every_missing_source_and_does_not_copy(tmp_path: Path) -> None:
    results, reports = _write_fixture(tmp_path)
    (results / "6_final.sdc").unlink()
    (results / "6_final.spef").unlink()
    destination = tmp_path / "data" / "gcd"

    outcome = import_gcd_dataset(results, reports, destination)

    assert not outcome.ok
    assert any("Missing SDC constraints" in message and "6_final.sdc" in message for message in outcome.errors)
    assert any("Missing SPEF parasitics" in message and "6_final.spef" in message for message in outcome.errors)
    assert not destination.exists()


def test_import_reports_a_missing_reports_directory(tmp_path: Path) -> None:
    results, reports = _write_fixture(tmp_path)
    missing = tmp_path / "absent-reports"

    outcome = import_gcd_dataset(results, missing, tmp_path / "gcd")

    assert [message.startswith("Missing reports directory:") for message in outcome.errors] == [True]
    assert str(missing) in outcome.errors[0]
    assert reports.is_dir()


def test_import_reports_when_reports_contain_no_rpt_files(tmp_path: Path) -> None:
    results, reports = _write_fixture(tmp_path)
    for report in reports.glob("*.rpt"):
        report.unlink()

    outcome = import_gcd_dataset(results, reports, tmp_path / "gcd")

    assert outcome.errors == (f"No .rpt files in reports directory: {reports}",)


def test_repeat_import_overwrites_destination_bytes(tmp_path: Path) -> None:
    results, reports = _write_fixture(tmp_path)
    destination = tmp_path / "gcd"
    import_gcd_dataset(results, reports, destination)
    (results / "6_final.v").write_text("updated netlist\n", encoding="utf-8")

    outcome = import_gcd_dataset(results, reports, destination)

    assert outcome.ok
    assert (destination / "netlist" / "6_final.v").read_text(encoding="utf-8") == "updated netlist\n"


def test_main_prints_success(tmp_path: Path, capsys) -> None:
    results, reports = _write_fixture(tmp_path)
    destination = tmp_path / "gcd"

    code = main(
        [
            "--results-dir",
            str(results),
            "--reports-dir",
            str(reports),
            "--destination",
            str(destination),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == "SUCCESS"
    assert captured.err == ""


def test_main_prints_the_missing_netlist_and_not_success(tmp_path: Path, capsys) -> None:
    results, reports = _write_fixture(tmp_path)
    (results / "6_final.v").unlink()

    code = main(
        [
            "--results-dir",
            str(results),
            "--reports-dir",
            str(reports),
            "--destination",
            str(tmp_path / "gcd"),
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert "Missing Verilog netlist:" in captured.err
    assert "6_final.v" in captured.err
    assert "SUCCESS" not in captured.err
