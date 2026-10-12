"""Copy the OpenROAD GCD finish artifacts into ``data/gcd``."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = (
    "/home/administrator/OpenROAD-flow-scripts/flow/results/sky130hd/gcd/base"
)
DEFAULT_REPORTS_DIR = (
    "/home/administrator/OpenROAD-flow-scripts/flow/reports/sky130hd/gcd/base"
)

NETLIST_NAME = "6_final.v"
SDC_NAME = "6_final.sdc"
SPEF_NAME = "6_final.spef"

_RESULT_FILES = (
    (NETLIST_NAME, "netlist", "Verilog netlist"),
    (SDC_NAME, "sdc", "SDC constraints"),
    (SPEF_NAME, "spef", "SPEF parasitics"),
)
_REPORTS_DIRNAME = "reports"


@dataclass(frozen=True)
class ImportResult:
    """Files copied by one import, plus any blocking error messages."""

    copied: tuple[Path, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def resolve_source_path(path: str | Path) -> Path:
    """Return ``path`` when it exists.

    On Windows, a ``/home/...`` path from the OpenROAD flow is opened through
    the WSL filesystem share when the Linux path itself is not present.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate

    raw = str(path).replace("\\", "/")
    if os.name != "nt" or not raw.startswith("/"):
        return candidate

    relative = raw.lstrip("/")
    for resolved in _wsl_path_candidates(relative):
        try:
            found = resolved.exists()
        except OSError:
            logger.warning("Could not stat WSL path %s", resolved)
            continue
        if found:
            logger.info("Resolved %s to %s", path, resolved)
            return resolved
    return candidate


def import_gcd_dataset(
    results_dir: str | Path,
    reports_dir: str | Path,
    destination: str | Path,
) -> ImportResult:
    """Copy the GCD netlist, SDC, SPEF, and ``.rpt`` files.

    Every required source is checked before the first copy. A missing input
    leaves ``destination`` unchanged. The command is safe to repeat: files
    already present are overwritten with the current source bytes.
    """
    results = resolve_source_path(results_dir)
    reports = resolve_source_path(reports_dir)
    dest_root = Path(destination)

    errors: list[str] = []
    sources: list[tuple[Path, Path]] = []

    for filename, folder, role in _RESULT_FILES:
        source = results / filename
        if not source.is_file():
            shown = _checked_path(results_dir, results, filename)
            errors.append(f"Missing {role}: {shown}")
            continue
        sources.append((source, dest_root / folder / filename))

    if not reports.is_dir():
        errors.append(f"Missing reports directory: {_checked_path(reports_dir, reports)}")
    else:
        report_files = _report_files(reports)
        if not report_files:
            errors.append(
                f"No .rpt files in reports directory: {_checked_path(reports_dir, reports)}"
            )
        sources.extend(
            (report, dest_root / _REPORTS_DIRNAME / report.name) for report in report_files
        )

    if errors:
        return ImportResult(copied=(), errors=tuple(errors))

    try:
        for folder in ("netlist", "sdc", "spef", _REPORTS_DIRNAME):
            (dest_root / folder).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = f"Cannot create destination directory under {dest_root}: {exc}"
        logger.error(message)
        return ImportResult(copied=(), errors=(message,))

    copied: list[Path] = []
    copy_errors: list[str] = []
    for source, target in sources:
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            message = f"Failed to copy {source} -> {target}: {exc}"
            logger.error(message)
            copy_errors.append(message)
            continue
        logger.info("Copied %s -> %s", source, target)
        copied.append(target)

    if copy_errors:
        return ImportResult(copied=tuple(copied), errors=tuple(copy_errors))

    logger.info("Copied %d files into %s", len(copied), dest_root)
    return ImportResult(copied=tuple(copied), errors=())


def main(argv: list[str] | None = None) -> int:
    """Import the GCD dataset and print ``SUCCESS`` or the error messages."""
    parser = argparse.ArgumentParser(
        description="Copy the OpenROAD GCD netlist, constraints, SPEF, and reports into data/gcd."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"OpenROAD results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        help=f"OpenROAD reports directory (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Destination dataset root (default: data/gcd in this checkout)",
    )
    args = parser.parse_args(argv)
    destination = args.destination if args.destination is not None else data_dir() / "gcd"

    result = import_gcd_dataset(args.results_dir, args.reports_dir, destination)
    if result.errors:
        for message in result.errors:
            print(message, file=sys.stderr)
        return 1

    print("SUCCESS")
    return 0


def _checked_path(original: str | Path, resolved: Path, child: str = "") -> str:
    """Path string for an error, preserving a source path that did not resolve."""
    if resolved.exists():
        return str(resolved / child) if child else str(resolved)
    text = str(original).rstrip("/\\")
    if not child:
        return text
    separator = "\\" if "\\" in text and "/" not in text else "/"
    return f"{text}{separator}{child}"


def _report_files(reports_dir: Path) -> list[Path]:
    files = [
        path
        for path in reports_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".rpt"
    ]
    return sorted(files, key=lambda path: path.name)


def _wsl_path_candidates(relative: str):
    """Yield UNC paths for a Linux absolute path.

    ``//wsl.localhost`` itself is not a stat-able directory on Windows.
    The distro name is the share, so each candidate is
    ``//wsl.localhost/<distro>/<relative>``. Ubuntu is tried before
    ``wsl.exe -l`` so a normal install does not pay for a distro listing.
    """
    seen: set[str] = set()

    def emit(distro: str):
        for prefix in (f"//wsl.localhost/{distro}", f"//wsl$/{distro}"):
            key = f"{prefix}/{relative}"
            if key in seen:
                continue
            seen.add(key)
            yield Path(prefix) / relative

    yield from emit("Ubuntu")
    for distro in _wsl_distro_names():
        yield from emit(distro)


def _wsl_distro_names() -> list[str]:
    try:
        completed = subprocess.run(
            ["wsl.exe", "-l", "-q"],
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("Could not list WSL distros")
        return []
    return _decode_wsl_listing(completed.stdout)


def _decode_wsl_listing(raw: bytes) -> list[str]:
    if not raw:
        return []
    if b"\x00" in raw[:8]:
        text = raw.decode("utf-16-le", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    names: list[str] = []
    for line in text.replace("\x00", "").splitlines():
        name = line.strip().lstrip("\ufeff")
        if not name or name.lower().startswith("docker-"):
            continue
        if name not in names:
            names.append(name)
    return names
