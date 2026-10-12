"""Inventory OpenROAD-flow-scripts benchmarks without running the flow."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from slackgnn.paths import data_dir, repo_root

PRIORITY_DESIGNS = ("uart", "spi", "picorv32", "serv", "tinyRocket")
_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:?]?=\s*(.*)$")
_VERILOG_SUFFIXES = {".v", ".sv", ".vh"}
_FINISH_FILES = ("6_final.v", "6_final.sdc", "6_final.spef")


def default_orfs_root() -> Path:
    """Return the OpenROAD-flow-scripts checkout visible on this machine."""
    configured = os.environ.get("ORFS_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/home/administrator/OpenROAD-flow-scripts"),
            Path(r"\\wsl.localhost\Ubuntu\home\administrator\OpenROAD-flow-scripts"),
        ]
    )
    for candidate in candidates:
        if (candidate / "flow" / "designs").is_dir():
            return candidate
    raise FileNotFoundError("OpenROAD-flow-scripts checkout not found. Set ORFS_ROOT.")


def discover_designs(orfs_root: str | Path | None = None) -> dict[str, object]:
    """Read every ``config.mk`` and estimate which benchmarks can be imported."""
    root = Path(orfs_root) if orfs_root is not None else default_orfs_root()
    designs_root = root / "flow" / "designs"
    if not designs_root.is_dir():
        raise FileNotFoundError(f"ORFS designs directory not found: {designs_root}")
    published = _published_designs()
    rows: list[dict[str, object]] = []
    for config in sorted(designs_root.glob("*/*/config.mk")):
        if "src" in config.parts:
            continue
        platform = config.parent.parent.name
        design_name = config.parent.name
        if platform in {"src"} or design_name.startswith("."):
            continue
        rows.append(_describe_design(root, config, platform, design_name, published))
    rows.sort(key=lambda row: (int(row["priority_rank"]), str(row["design_name"]), str(row["platform"])))
    found_priority = {str(row["design_name"]) for row in rows}
    return {
        "orfs_root": str(root),
        "design_count": len(rows),
        "priority": list(PRIORITY_DESIGNS),
        "missing_priority": [name for name in PRIORITY_DESIGNS if name not in found_priority],
        "already_published": sorted(published),
        "designs": rows,
    }


def write_available_designs(report: dict[str, object], destination: str | Path | None = None) -> Path:
    """Write ``reports/available_designs.json``."""
    path = Path(destination) if destination is not None else repo_root() / "reports" / "available_designs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def select_design(report: dict[str, object], design_name: str, platform: str | None = None) -> dict[str, object]:
    """Pick one discovered design, preferring sky130hd when the platform is omitted."""
    matches = [row for row in report["designs"] if row["design_name"] == design_name]
    if platform is not None:
        matches = [row for row in matches if row["platform"] == platform]
    if not matches:
        known = ", ".join(sorted({str(row["design_name"]) for row in report["designs"]}))
        target = design_name if platform is None else f"{platform}/{design_name}"
        raise LookupError(f"Design {target} is not in the ORFS checkout. Known designs include: {known}")
    preferred = ("sky130hd", "sky130hs", "asap7", "nangate45", "gf180", "ihp-sg13g2")
    matches.sort(key=lambda row: preferred.index(str(row["platform"])) if str(row["platform"]) in preferred else len(preferred))
    return matches[0]


def _describe_design(
    orfs_root: Path,
    config: Path,
    platform: str,
    design_name: str,
    published: set[str],
) -> dict[str, object]:
    values = _assignments(config.read_text(encoding="utf-8", errors="replace"))
    nickname = _plain(values.get("DESIGN_NICKNAME")) or design_name
    variant = _plain(values.get("FLOW_VARIANT")) or "base"
    top_module = _plain(values.get("DESIGN_NAME")) or design_name
    rtl_bytes = _verilog_bytes(orfs_root / "flow" / "designs" / "src" / nickname)
    rtl_bytes += _verilog_bytes(config.parent)
    results = orfs_root / "flow" / "results" / platform / nickname / variant
    reports = orfs_root / "flow" / "reports" / platform / nickname / variant
    finish_bytes = _file_size(results / "6_final.v")
    complexity_bytes = finish_bytes or rtl_bytes
    return {
        "design_name": design_name,
        "top_module": top_module,
        "platform": platform,
        "nickname": nickname,
        "flow_variant": variant,
        "estimated_complexity": _complexity_class(complexity_bytes),
        "rtl_bytes": rtl_bytes,
        "routed_netlist_bytes": finish_bytes,
        "core_utilization": _plain(values.get("CORE_UTILIZATION")),
        "die_area": _plain(values.get("DIE_AREA")),
        "finish_artifacts_present": _finish_ready(results, reports),
        "in_slackgnn_corpus": design_name in published,
        "priority_rank": PRIORITY_DESIGNS.index(design_name) if design_name in PRIORITY_DESIGNS else len(PRIORITY_DESIGNS),
        "config": str(config),
    }


def _assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        if raw.startswith((" ", "\t")):
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _ASSIGNMENT.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _plain(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().strip('"')
    if not text or "$" in text:
        return None
    return text


def _verilog_bytes(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in _VERILOG_SUFFIXES:
            total += path.stat().st_size
    return total


def _file_size(path: Path) -> int:
    if not path.is_file():
        return 0
    return path.stat().st_size


def _finish_ready(results: Path, reports: Path) -> bool:
    artifacts = [results / name for name in _FINISH_FILES]
    artifacts.append(reports / "6_finish.rpt")
    return all(path.is_file() and path.stat().st_size > 0 for path in artifacts)


def _complexity_class(size: int) -> str:
    if size < 20_000:
        return "tiny"
    if size < 100_000:
        return "small"
    if size < 1_000_000:
        return "medium"
    if size < 5_000_000:
        return "large"
    return "very_large"


def _published_designs() -> set[str]:
    metadata = data_dir() / "corpus" / "corpus_metadata.json"
    if not metadata.is_file():
        return set()
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    designs = payload.get("designs", [])
    if not isinstance(designs, list):
        return set()
    names: set[str] = set()
    for item in designs:
        if isinstance(item, dict) and item.get("design_name"):
            names.add(str(item["design_name"]))
        elif isinstance(item, str):
            names.add(item)
    return names
