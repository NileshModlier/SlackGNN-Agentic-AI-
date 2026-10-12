"""Run the OpenROAD dataset pipeline for one design and publish it.

The design name comes from the caller. Paths are derived from
``data/<design>/``. This module does not train a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from slackgnn.dataset.corpus import (
    build_corpus,
    design_name_error,
    load_corpus_metadata,
    record_design_failure,
    register_design,
)
from slackgnn.dataset.graph_dataset import build_main as build_dataset_main
from slackgnn.dataset.netlist_graph import main as extract_graph_main
from slackgnn.dataset.pyg_dataset import build_main as build_pyg_main
from slackgnn.dataset.summary import load_path_labels
from slackgnn.dataset.summary import main as summarize_main
from slackgnn.dataset.timing_report import main as parse_timing_main
from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

_OUTPUT_DIRECTORIES = ("parsed", "graphs", "dataset", "pyg")
_PIPELINE = (
    "parse_timing_report.py",
    "extract_graph.py",
    "build_dataset.py",
    "build_pyg_dataset.py",
)


def process_design(
    design: str,
    data_root: str | Path | None = None,
    manifest: str | Path | None = None,
    corpus_root: str | Path | None = None,
    top: str | None = None,
) -> tuple[dict[str, object], int]:
    """Parse, extract, label, and convert ``design``, then refresh the corpus.

    Return ``(report, exit_code)``. A failed design is recorded in
    ``corpus_metadata.json`` and is left out of the corpus totals.
    """
    root = Path(data_root) if data_root is not None else data_dir()
    manifest_path = Path(manifest) if manifest is not None else root / "designs.json"
    corpus_path = Path(corpus_root) if corpus_root is not None else root / "corpus"
    problem = design_name_error(design)
    if problem:
        logger.error(problem)
        report = _report(design, "failed", [{"script": _PIPELINE[0], "status": "failed", "detail": problem}])
        return report, 1
    if not root.is_dir():
        message = f"Missing data directory: {root}"
        logger.error(message)
        report = _report(design, "failed", [{"script": _PIPELINE[0], "status": "failed", "detail": message}])
        return report, 1

    design_root = root / design
    module = top or design
    logger.info("Processing design %s (top module %s)", design, module)
    if not design_root.is_dir():
        message = f"Missing design directory: {design_root}"
        logger.error(message)
        stages = [_stage(_PIPELINE[0], "failed", message)]
        stages.extend(_stage(script, "skipped", "") for script in _PIPELINE[1:])
        corpus_stage = _publish(False, design, message, manifest_path, root, corpus_path)
        stages.append(corpus_stage)
        report = _report(design, "failed", stages)
        return report, 1

    _create_output_directories(design_root)
    paths = _design_paths(design_root)
    missing = _missing_inputs(paths)
    stages: list[dict[str, str]] = []
    if missing:
        message = "; ".join(missing)
        logger.error(message)
        stages.append(_stage(_PIPELINE[0], "failed", message))
        stages.extend(_stage(script, "skipped", "") for script in _PIPELINE[1:])
        pipeline_ok = False
        failure_reason = message
    else:
        runners = (
            lambda: _run_parse(paths),
            lambda: _run_extract(paths, module),
            lambda: _run_dataset(paths),
            lambda: _run_pyg(paths),
        )
        failed = False
        failure_reason = ""
        for script, runner in zip(_PIPELINE, runners, strict=True):
            if failed:
                stages.append(_stage(script, "skipped", ""))
                continue
            logger.info("Running %s for %s", script, design)
            errors = runner()
            if errors:
                failed = True
                failure_reason = "; ".join(errors)
                logger.error("%s failed for %s: %s", script, design, failure_reason)
                stages.append(_stage(script, "failed", failure_reason))
            else:
                logger.info("%s completed for %s", script, design)
                stages.append(_stage(script, "success", ""))
        pipeline_ok = not failed

    corpus_stage = _publish(pipeline_ok, design, failure_reason, manifest_path, root, corpus_path)
    stages.append(corpus_stage)
    status = "success" if pipeline_ok and corpus_stage["status"] == "success" else "failed"
    report = _report(design, status, stages)
    write_errors = _write_json(report, design_root / "process_report.json")
    if write_errors:
        logger.error(write_errors[0])
        return report, 1
    return report, 0 if status == "success" else 1


def render_process_report(report: dict[str, object]) -> str:
    """Format one design's pipeline report."""
    lines = [
        f"design: {report['design_name']}",
        f"status: {report['status']}",
    ]
    stages = report.get("stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lines.append(f"{stage.get('script', '')}: {stage.get('status', '')}")
            detail = str(stage.get("detail", ""))
            if detail:
                lines.append(f"  detail: {detail}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Process one design named on the command line."""
    parser = argparse.ArgumentParser(
        description="Run timing parsing, graph extraction, dataset construction, and PyG conversion."
    )
    parser.add_argument("design", help="Design name. Files are read from data/<design>/.")
    parser.add_argument("--top", default=None, help="Verilog top module. Defaults to the design name.")
    parser.add_argument("--data-root", type=Path, default=data_dir(), help="Directory that contains design folders")
    parser.add_argument("--manifest", type=Path, default=None, help="Design list JSON. Defaults to <data-root>/designs.json")
    parser.add_argument("--corpus-root", type=Path, default=None, help="Corpus directory. Defaults to <data-root>/corpus")
    args = parser.parse_args(argv)
    report, code = process_design(
        args.design,
        data_root=args.data_root,
        manifest=args.manifest,
        corpus_root=args.corpus_root,
        top=args.top,
    )
    stream = sys.stdout if code == 0 else sys.stderr
    print(render_process_report(report), end="", file=stream)
    return code


def _publish(
    pipeline_ok: bool,
    design: str,
    failure_reason: str,
    manifest_path: Path,
    data_root: Path,
    corpus_path: Path,
) -> dict[str, str]:
    script = "build_corpus.py"
    if pipeline_ok:
        register_errors = register_design(manifest_path, design)
        if register_errors:
            return _stage(script, "failed", "; ".join(register_errors))
        metadata, errors = build_corpus(data_root, manifest_path, corpus_path)
        if errors or metadata is None:
            return _stage(script, "failed", "; ".join(errors))
        published = _published_record(metadata, design)
        if published is None:
            reason = _failure_reason(metadata, design) or f"{design} was not published"
            return _stage(script, "failed", reason)
        detail = (
            f"nodes {published['node_count']}, edges {published['edge_count']}, "
            f"labels {published['label_count']}"
        )
        logger.info("Published %s into %s", design, corpus_path)
        return _stage(script, "success", detail)

    metadata: dict[str, object] | None = None
    if manifest_path.is_file():
        metadata, errors = build_corpus(data_root, manifest_path, corpus_path)
        if errors:
            logger.error("Corpus refresh failed: %s", "; ".join(errors))
            metadata, load_errors = load_corpus_metadata(corpus_path)
            if load_errors:
                metadata = None
    elif (corpus_path / "corpus_metadata.json").is_file():
        metadata, load_errors = load_corpus_metadata(corpus_path)
        if load_errors:
            metadata = None
    reason = failure_reason or f"{design} failed"
    updated, record_errors = record_design_failure(corpus_path, metadata, design, reason)
    if record_errors or updated is None:
        return _stage(script, "failed", "; ".join(record_errors))
    logger.info("Recorded %s as a corpus failure", design)
    return _stage(script, "failed", f"Recorded failure in corpus metadata: {reason}")


def _run_parse(paths: dict[str, Path]) -> tuple[str, ...]:
    code = parse_timing_main(["--report", str(paths["report"]), "--output", str(paths["path_labels"])])
    if code != 0:
        return ("parse_timing_report.py failed",)
    code = summarize_main(["--labels", str(paths["path_labels"]), "--output", str(paths["dataset_summary"])])
    if code != 0:
        return ("dataset summary failed",)
    return ()


def _run_extract(paths: dict[str, Path], top: str) -> tuple[str, ...]:
    code = extract_graph_main(
        ["--netlist", str(paths["netlist"]), "--output", str(paths["graph"]), "--top", top]
    )
    if code != 0:
        return ("extract_graph.py failed",)
    return ()


def _run_dataset(paths: dict[str, Path]) -> tuple[str, ...]:
    code = build_dataset_main(
        [
            "--graph",
            str(paths["graph"]),
            "--labels",
            str(paths["path_labels"]),
            "--graph-output",
            str(paths["labeled_graph"]),
            "--paths-output",
            str(paths["labeled_paths"]),
        ]
    )
    if code != 0:
        return ("build_dataset.py failed",)
    return _write_path_dataset(paths["path_labels"], paths["path_dataset"])


def _run_pyg(paths: dict[str, Path]) -> tuple[str, ...]:
    code = build_pyg_main(
        [
            "--graph",
            str(paths["graph"]),
            "--labels",
            str(paths["path_dataset"]),
            "--output-dir",
            str(paths["pyg"]),
        ]
    )
    if code != 0:
        return ("build_pyg_dataset.py failed",)
    return ()


def _write_path_dataset(labels: Path, destination: Path) -> tuple[str, ...]:
    paths, errors = load_path_labels(labels)
    if errors:
        return errors
    records = [
        {
            "startpoint": path.startpoint,
            "endpoint": path.endpoint,
            "path_group": path.path_group,
            "path_type": path.path_type,
            "arrival_time": float(path.arrival_time),
            "required_time": float(path.required_time),
            "slack": float(path.slack),
        }
        for path in paths
    ]
    return _write_json({"paths": records}, destination)


def _design_paths(design_root: Path) -> dict[str, Path]:
    return {
        "netlist": design_root / "netlist" / "6_final.v",
        "report": design_root / "reports" / "6_finish.rpt",
        "path_labels": design_root / "parsed" / "path_labels.csv",
        "dataset_summary": design_root / "parsed" / "dataset_summary.json",
        "graph": design_root / "graphs" / "netlist_graph.pkl",
        "labeled_graph": design_root / "dataset" / "labeled_graph.pkl",
        "labeled_paths": design_root / "dataset" / "labeled_paths.csv",
        "path_dataset": design_root / "dataset" / "path_dataset.json",
        "pyg": design_root / "pyg",
    }


def _missing_inputs(paths: dict[str, Path]) -> list[str]:
    missing: list[str] = []
    if not paths["netlist"].is_file():
        missing.append(f"Missing netlist: {paths['netlist']}")
    if not paths["report"].is_file():
        missing.append(f"Missing timing report: {paths['report']}")
    return missing


def _create_output_directories(design_root: Path) -> None:
    for name in _OUTPUT_DIRECTORIES:
        (design_root / name).mkdir(parents=True, exist_ok=True)


def _published_record(metadata: dict[str, object], design: str) -> dict[str, object] | None:
    designs = metadata.get("designs")
    if not isinstance(designs, list):
        return None
    for record in designs:
        if isinstance(record, dict) and record.get("design_name") == design:
            return record
    return None


def _failure_reason(metadata: dict[str, object], design: str) -> str:
    failures = metadata.get("failures")
    if not isinstance(failures, list):
        return ""
    for record in failures:
        if isinstance(record, dict) and record.get("design_name") == design:
            return str(record.get("reason", ""))
    return ""


def _stage(script: str, status: str, detail: str) -> dict[str, str]:
    return {"script": script, "status": status, "detail": detail}


def _report(design: str, status: str, stages: list[dict[str, str]]) -> dict[str, object]:
    return {"design_name": design, "status": status, "stages": stages}


def _write_json(payload: dict[str, object], destination: Path) -> tuple[str, ...]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()
