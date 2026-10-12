"""Run ORFS for one benchmark and publish it through the existing pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from slackgnn.expansion.discover import discover_designs, select_design, write_available_designs
from slackgnn.paths import data_dir, repo_root

logger = logging.getLogger(__name__)

PUBLISHED_CORPUS = frozenset({"gcd", "aes", "ibex", "jpeg"})
_IMPORT_FILES = (
    ("6_final.v", "netlist"),
    ("6_final.sdc", "sdc"),
    ("6_final.spef", "spef"),
    ("6_finish.rpt", "reports"),
)


def expand_design(
    design_name: str,
    platform: str | None = None,
    run_flow: bool = True,
    force: bool = False,
    orfs_root: str | Path | None = None,
) -> int:
    """Run the flow when needed, copy finish files, and publish the design."""
    if design_name in PUBLISHED_CORPUS and not force:
        logger.error("%s is already in the M8 corpus. Refusing to replace it.", design_name)
        return 1
    report = discover_designs(orfs_root)
    write_available_designs(report)
    try:
        selected = select_design(report, design_name, platform)
    except LookupError as exc:
        logger.error("%s", exc)
        return 1
    root = Path(str(report["orfs_root"]))
    results = root / "flow" / "results" / str(selected["platform"]) / str(selected["nickname"]) / str(selected["flow_variant"])
    reports = root / "flow" / "reports" / str(selected["platform"]) / str(selected["nickname"]) / str(selected["flow_variant"])
    if not selected["finish_artifacts_present"]:
        if not run_flow:
            logger.error("Finish artifacts are missing for %s and --skip-flow was set.", design_name)
            return 1
        status = _run_orfs(design_name, str(selected["platform"]))
        if status != 0:
            return status
        report = discover_designs(root)
        write_available_designs(report)
        selected = select_design(report, design_name, str(selected["platform"]))
        if not selected["finish_artifacts_present"]:
            logger.error("ORFS finished without the required artifacts for %s.", design_name)
            return 1
    copy_errors = _import_finish_files(design_name, results, reports)
    if copy_errors:
        for message in copy_errors:
            logger.error("%s", message)
        return 1
    if _run_python("scripts/process_design.py", design_name) != 0:
        return 1
    if _run_python("scripts/build_corpus.py") != 0:
        return 1
    logger.info("Published %s from %s", design_name, selected["platform"])
    return 0


def _run_orfs(design_name: str, platform: str) -> int:
    script = repo_root() / "scripts" / "run_orfs_design.sh"
    if not script.is_file():
        logger.error("Missing ORFS launcher: %s", script)
        return 1
    if os.name == "nt":
        command = (
            f"cd {_bash_quote(_windows_to_wsl(repo_root()))} && "
            f"PLATFORM={_bash_quote(platform)} bash ./scripts/run_orfs_design.sh {_bash_quote(design_name)}"
        )
        invocation = ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command]
    else:
        invocation = ["bash", str(script), design_name]
    logger.info("Running ORFS for %s on %s", design_name, platform)
    completed = subprocess.run(invocation, check=False, env={**os.environ, "PLATFORM": platform})
    if completed.returncode != 0:
        logger.error("ORFS failed for %s with status %s", design_name, completed.returncode)
    return completed.returncode


def _import_finish_files(design_name: str, results: Path, reports: Path) -> tuple[str, ...]:
    sources = {
        "6_final.v": results / "6_final.v",
        "6_final.sdc": results / "6_final.sdc",
        "6_final.spef": results / "6_final.spef",
        "6_finish.rpt": reports / "6_finish.rpt",
    }
    errors: list[str] = []
    for filename, folder in _IMPORT_FILES:
        source = sources[filename]
        if not source.is_file() or source.stat().st_size == 0:
            errors.append(f"Missing or empty finish file: {source}")
            continue
        destination_dir = data_dir() / design_name / folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename
        shutil.copy2(source, destination)
        logger.info("Copied %s", destination)
    return tuple(errors)


def _run_python(script: str, *args: str) -> int:
    invocation = [sys.executable, str(repo_root() / script), *args]
    logger.info("Running %s", " ".join(invocation))
    completed = subprocess.run(invocation, check=False, cwd=repo_root())
    if completed.returncode != 0:
        logger.error("%s failed with status %s", script, completed.returncode)
    return completed.returncode


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def _bash_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
