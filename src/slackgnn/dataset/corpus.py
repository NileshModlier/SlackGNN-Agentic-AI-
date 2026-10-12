"""Collect per-design PyG graphs into one multi-design corpus.

Design names come from a JSON manifest, so a new benchmark is a new name
plus a ``data/<design>/`` directory. This module copies or builds
``graph.pt`` and ``metadata.json``. It does not train a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from slackgnn.dataset.pyg_dataset import (
    build_pyg_data,
    load_networkx_graph,
    load_path_dataset,
    load_pyg_dataset,
    save_pyg_dataset,
    summarize_pyg_data,
)
from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

_DESIGN_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_RESERVED_NAMES = frozenset({"corpus"})
_PUBLISHED_FILES = ("graph.pt", "metadata.json")
_TOTAL_FIELDS = ("total_designs", "total_nodes", "total_edges", "total_labels")
_DESIGN_FIELDS = (
    "design_name",
    "node_count",
    "edge_count",
    "label_count",
    "positive_slack_count",
    "negative_slack_count",
)


def design_name_error(name: object) -> str | None:
    """Return an error when ``name`` is not a safe design directory name."""
    if not isinstance(name, str) or _DESIGN_NAME.fullmatch(name) is None:
        return f"Design name {name!r} is not a design name"
    if name in _RESERVED_NAMES:
        return f"Design name {name!r} is reserved"
    return None


def register_design(manifest: str | Path, design: str) -> tuple[str, ...]:
    """Append ``design`` to the manifest when it is not already listed."""
    problem = design_name_error(design)
    if problem:
        logger.error(problem)
        return (problem,)
    path = Path(manifest)
    if not path.is_file():
        return _write_json({"designs": [design]}, path)
    names, errors = load_design_manifest(path)
    if errors:
        return errors
    if design in names:
        return ()
    return _write_json({"designs": [*names, design]}, path)


def record_design_failure(
    corpus_root: str | Path,
    metadata: dict[str, object] | None,
    design: str,
    reason: str,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Store ``design`` in the corpus failure list and refresh the totals."""
    problem = design_name_error(design)
    if problem:
        return None, (problem,)
    current = metadata if isinstance(metadata, dict) else {}
    designs = [
        item
        for item in current.get("designs", [])
        if isinstance(item, dict) and item.get("design_name") != design
    ]
    failures = [
        item
        for item in current.get("failures", [])
        if isinstance(item, dict) and item.get("design_name") != design
    ]
    failures.append({"design_name": design, "reason": reason})
    updated: dict[str, object] = {
        "total_designs": len(designs),
        "total_nodes": sum(int(item["node_count"]) for item in designs),
        "total_edges": sum(int(item["edge_count"]) for item in designs),
        "total_labels": sum(int(item["label_count"]) for item in designs),
        "designs": designs,
        "failures": failures,
    }
    errors = _write_json(updated, Path(corpus_root) / "corpus_metadata.json")
    if errors:
        return None, errors
    return updated, ()


def default_manifest_path() -> Path:
    """Return the checked-in design list."""
    return data_dir() / "designs.json"


def default_corpus_dir() -> Path:
    """Return the multi-design corpus directory."""
    return data_dir() / "corpus"


def load_design_manifest(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load design names from a manifest object ``{"designs": [...]}``."""
    source = Path(path)
    if not source.is_file():
        message = f"Missing design manifest: {source}"
        logger.error(message)
        return (), (message,)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"Cannot read design manifest {source}: {exc}"
        logger.error(message)
        return (), (message,)
    except json.JSONDecodeError as exc:
        message = f"Design manifest {source} is not valid JSON: {exc}"
        logger.error(message)
        return (), (message,)
    if not isinstance(payload, dict) or "designs" not in payload:
        return (), (f"Design manifest {source} must be an object with a designs list",)
    names = payload["designs"]
    if not isinstance(names, list) or not names:
        return (), (f"Design manifest {source} does not contain a non-empty designs list",)

    errors: list[str] = []
    accepted: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(names, start=1):
        if not isinstance(name, str) or _DESIGN_NAME.fullmatch(name) is None:
            errors.append(f"Design manifest entry {index} is not a design name: {name!r}")
            continue
        if name in _RESERVED_NAMES:
            errors.append(f"Design name {name!r} is reserved")
            continue
        if name in seen:
            errors.append(f"Design name {name!r} is listed more than once")
            continue
        seen.add(name)
        accepted.append(name)
    if errors:
        for message in errors:
            logger.error(message)
        return (), tuple(errors)
    return tuple(accepted), ()


def build_corpus(
    data_root: str | Path | None = None,
    manifest: str | Path | None = None,
    corpus_root: str | Path | None = None,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Publish one corpus entry per valid design and write ``corpus_metadata.json``.

    A design that fails validation is listed under ``failures`` and is omitted
    from the totals. Other designs are still published.
    """
    root = Path(data_root) if data_root is not None else data_dir()
    manifest_path = Path(manifest) if manifest is not None else default_manifest_path()
    destination_root = Path(corpus_root) if corpus_root is not None else default_corpus_dir()
    if not root.is_dir():
        message = f"Missing data directory: {root}"
        logger.error(message)
        return None, (message,)

    names, manifest_errors = load_design_manifest(manifest_path)
    if manifest_errors:
        return None, manifest_errors

    designs: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for name in names:
        record, reason = _publish_design(root, destination_root, name)
        if record is None:
            failure = {"design_name": name, "reason": reason or "Design failed validation"}
            failures.append(failure)
            logger.warning("Design %s failed: %s", name, failure["reason"])
            continue
        designs.append(record)
        logger.info(
            "Published %s: %s nodes, %s edges, %s labels",
            name,
            record["node_count"],
            record["edge_count"],
            record["label_count"],
        )

    metadata = {
        "total_designs": len(designs),
        "total_nodes": sum(int(item["node_count"]) for item in designs),
        "total_edges": sum(int(item["edge_count"]) for item in designs),
        "total_labels": sum(int(item["label_count"]) for item in designs),
        "designs": designs,
        "failures": failures,
    }
    prune_errors = _prune_corpus(destination_root, {str(item["design_name"]) for item in designs})
    if prune_errors:
        return None, prune_errors
    write_errors = _write_json(metadata, destination_root / "corpus_metadata.json")
    if write_errors:
        return None, write_errors
    return metadata, ()


def load_corpus_metadata(corpus_root: str | Path) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Load and check ``corpus_metadata.json``."""
    path = Path(corpus_root) / "corpus_metadata.json"
    if not path.is_file():
        message = f"Missing corpus metadata: {path}"
        logger.error(message)
        return None, (message,)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"Cannot read corpus metadata {path}: {exc}"
        logger.error(message)
        return None, (message,)
    except json.JSONDecodeError as exc:
        message = f"Corpus metadata {path} is not valid JSON: {exc}"
        logger.error(message)
        return None, (message,)
    errors = _corpus_metadata_errors(payload)
    if errors:
        for message in errors:
            logger.error(message)
        return None, tuple(errors)
    return payload, ()


def validate_published_corpus(corpus_root: str | Path, metadata: dict[str, object]) -> tuple[str, ...]:
    """Check that each successful design still has a non-empty graph and metadata."""
    root = Path(corpus_root)
    if not root.is_dir():
        return (f"Missing corpus directory: {root}",)
    errors: list[str] = []
    designs = metadata["designs"]
    if not isinstance(designs, list):
        return ("corpus metadata designs is not a list",)
    for record in designs:
        if not isinstance(record, dict):
            errors.append("corpus metadata contains a design entry that is not an object")
            continue
        name = str(record["design_name"])
        design_dir = root / name
        graph_path = design_dir / "graph.pt"
        meta_path = design_dir / "metadata.json"
        if not graph_path.is_file():
            errors.append(f"Missing PyG graph: {graph_path}")
        elif graph_path.stat().st_size <= 0:
            errors.append(f"PyG graph is empty: {graph_path}")
        if not meta_path.is_file():
            errors.append(f"Missing metadata: {meta_path}")
            continue
        try:
            published = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read metadata {meta_path}: {exc}")
            continue
        if not isinstance(published, dict):
            errors.append(f"{meta_path} is not a JSON object")
            continue
        for field in ("node_count", "edge_count", "label_count", "positive_slack_count", "negative_slack_count"):
            if field not in published:
                errors.append(f"{meta_path} is missing {field}")
                continue
            try:
                published_value = int(published[field])
                recorded_value = int(record[field])
            except (TypeError, ValueError):
                errors.append(f"{name} {field} is not an integer")
                continue
            if published_value != recorded_value:
                errors.append(
                    f"{name} {field} is {published[field]} in metadata.json "
                    f"but {record[field]} in corpus_metadata.json"
                )
        for field, label in (("node_count", "node_count"), ("edge_count", "edge_count")):
            if field not in published:
                continue
            try:
                count = int(published[field])
            except (TypeError, ValueError):
                continue
            if count <= 0:
                errors.append(f"{name} {label} is {published[field]}; expected a count greater than 0")
    return tuple(errors)


def render_corpus_summary(metadata: dict[str, object]) -> str:
    """Format the corpus overview and per-design metrics."""
    lines = [
        "Corpus Overview",
        "",
        f"Design Count: {metadata['total_designs']}",
        f"Designs: {metadata['total_designs']}",
        f"Total Nodes: {metadata['total_nodes']}",
        f"Total Edges: {metadata['total_edges']}",
        f"Total Labels: {metadata['total_labels']}",
        "",
    ]
    designs = metadata["designs"]
    if isinstance(designs, list):
        for record in designs:
            if not isinstance(record, dict):
                continue
            lines.append(str(record["design_name"]))
            for field in _DESIGN_FIELDS:
                lines.append(f"  {field}: {record[field]}")
            lines.append("")
    failures = metadata.get("failures", [])
    failed = len(failures) if isinstance(failures, list) else 0
    lines.append(f"Failed designs: {failed}")
    if isinstance(failures, list):
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            lines.append(str(failure.get("design_name", "")))
            lines.append(f"  reason: {failure.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def build_main(argv: list[str] | None = None) -> int:
    """Build ``data/corpus/`` from the design manifest."""
    parser = argparse.ArgumentParser(description="Build a multi-design corpus of PyG graphs.")
    parser.add_argument("--data-root", type=Path, default=data_dir(), help="Directory of per-design inputs")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path(), help="JSON design list")
    parser.add_argument("--corpus-root", type=Path, default=default_corpus_dir(), help="Corpus output directory")
    args = parser.parse_args(argv)
    metadata, errors = build_corpus(args.data_root, args.manifest, args.corpus_root)
    if errors or metadata is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    print(render_corpus_summary(metadata), end="")
    print(f"Wrote {args.corpus_root / 'corpus_metadata.json'}")
    return 0


def summary_main(argv: list[str] | None = None) -> int:
    """Print the corpus overview from ``corpus_metadata.json``."""
    parser = argparse.ArgumentParser(description="Summarize the multi-design PyG corpus.")
    parser.add_argument("--corpus-root", type=Path, default=default_corpus_dir(), help="Corpus directory")
    args = parser.parse_args(argv)
    metadata, errors = load_corpus_metadata(args.corpus_root)
    if errors or metadata is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    disk_errors = validate_published_corpus(args.corpus_root, metadata)
    if disk_errors:
        for message in disk_errors:
            print(message, file=sys.stderr)
        return 1
    print(render_corpus_summary(metadata), end="")
    return 0


def _publish_design(
    data_root: Path,
    corpus_root: Path,
    name: str,
) -> tuple[dict[str, object] | None, str | None]:
    design_root = data_root / name
    if not _is_within(data_root, design_root):
        return None, f"Design path escapes the data directory: {design_root}"
    if not design_root.is_dir():
        return None, f"Missing design directory: {design_root}"

    prepare_errors = _ensure_pyg(design_root)
    if prepare_errors:
        _discard_published(corpus_root, name)
        return None, "; ".join(prepare_errors)

    data, metadata, load_errors = load_pyg_dataset(design_root / "pyg")
    if load_errors or data is None or metadata is None:
        _discard_published(corpus_root, name)
        return None, "; ".join(load_errors)

    try:
        _, summary_errors = summarize_pyg_data(data, metadata)
        quality_errors = list(summary_errors)
        if not summary_errors:
            node_count = int(metadata["node_count"])
            edge_count = int(metadata["edge_count"])
            if node_count <= 0:
                quality_errors.append(f"node_count is {node_count}; expected a count greater than 0")
            if edge_count <= 0:
                quality_errors.append(f"edge_count is {edge_count}; expected a count greater than 0")
    except (TypeError, ValueError, RuntimeError) as exc:
        _discard_published(corpus_root, name)
        return None, f"Cannot validate PyG graph for {name}: {exc}"
    if quality_errors:
        _discard_published(corpus_root, name)
        return None, "; ".join(quality_errors)

    publish_errors = _copy_pyg(design_root / "pyg", corpus_root / name)
    if publish_errors:
        _discard_published(corpus_root, name)
        return None, "; ".join(publish_errors)

    return {
        "design_name": name,
        "node_count": int(metadata["node_count"]),
        "edge_count": int(metadata["edge_count"]),
        "label_count": int(metadata["label_count"]),
        "positive_slack_count": int(metadata["positive_slack_count"]),
        "negative_slack_count": int(metadata["negative_slack_count"]),
    }, None


def _ensure_pyg(design_root: Path) -> tuple[str, ...]:
    """Use an existing PyG directory, or build one from the graph and path table."""
    pyg_dir = design_root / "pyg"
    graph_path = pyg_dir / "graph.pt"
    meta_path = pyg_dir / "metadata.json"
    if graph_path.is_file() and meta_path.is_file():
        return ()

    missing: list[str] = []
    if not graph_path.is_file():
        missing.append(f"Missing PyG graph: {graph_path}")
    if not meta_path.is_file():
        missing.append(f"Missing metadata: {meta_path}")

    source_graph = design_root / "graphs" / "netlist_graph.pkl"
    source_paths = design_root / "dataset" / "path_dataset.json"
    if not source_graph.is_file() or not source_paths.is_file():
        if not source_graph.is_file():
            missing.append(f"Missing graph: {source_graph}")
        if not source_paths.is_file():
            missing.append(f"Missing path dataset: {source_paths}")
        return tuple(missing)

    graph, graph_errors = load_networkx_graph(source_graph)
    paths, path_errors = load_path_dataset(source_paths)
    errors = graph_errors + path_errors
    if errors or graph is None:
        return errors
    data, metadata, build_errors = build_pyg_data(graph, paths)
    if build_errors or data is None or metadata is None:
        return build_errors
    write_errors = save_pyg_dataset(data, metadata, pyg_dir)
    if write_errors:
        return write_errors
    logger.info("Generated PyG dataset for %s", design_root.name)
    return ()


def _copy_pyg(source: Path, destination: Path) -> tuple[str, ...]:
    if not _is_within(destination.parent, destination):
        return (f"Refusing to publish outside the corpus directory: {destination}",)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for name in _PUBLISHED_FILES:
            shutil.copy2(source / name, destination / name)
    except OSError as exc:
        message = f"Cannot publish {source} to {destination}: {exc}"
        logger.error(message)
        return (message,)
    for name in _PUBLISHED_FILES:
        published = destination / name
        if not published.is_file() or published.stat().st_size <= 0:
            return (f"Published file is empty: {published}",)
    return ()


def _discard_published(corpus_root: Path, name: str) -> None:
    target = corpus_root / name
    if target.exists():
        errors = _remove_tree(corpus_root, target)
        for message in errors:
            logger.error(message)


def _prune_corpus(corpus_root: Path, keep: set[str]) -> tuple[str, ...]:
    if not corpus_root.exists():
        return ()
    if not corpus_root.is_dir():
        return (f"Corpus path is not a directory: {corpus_root}",)
    errors: list[str] = []
    for child in corpus_root.iterdir():
        if not child.is_dir() or child.name in keep or child.name.startswith("."):
            continue
        errors.extend(_remove_tree(corpus_root, child))
    return tuple(errors)


def _remove_tree(root: Path, child: Path) -> tuple[str, ...]:
    try:
        resolved_root = root.resolve()
        resolved_child = child.resolve()
        resolved_child.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        message = f"Refusing to remove {child} outside {root}: {exc}"
        logger.error(message)
        return (message,)
    if resolved_child == resolved_root:
        return (f"Refusing to remove corpus root {root}",)
    try:
        shutil.rmtree(resolved_child)
    except OSError as exc:
        message = f"Cannot remove {child}: {exc}"
        logger.error(message)
        return (message,)
    return ()


def _corpus_metadata_errors(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["corpus metadata is not a JSON object"]
    errors: list[str] = []
    for field in (*_TOTAL_FIELDS, "designs", "failures"):
        if field not in payload:
            errors.append(f"corpus metadata is missing {field}")
    if errors:
        return errors
    designs = payload["designs"]
    failures = payload["failures"]
    if not isinstance(designs, list):
        errors.append("corpus metadata designs is not a list")
        designs = []
    if not isinstance(failures, list):
        errors.append("corpus metadata failures is not a list")
    for index, record in enumerate(designs, start=1):
        if not isinstance(record, dict):
            errors.append(f"design entry {index} is not an object")
            continue
        for field in _DESIGN_FIELDS:
            if field not in record:
                errors.append(f"design entry {index} is missing {field}")
    if errors:
        return errors
    if int(payload["total_designs"]) != len(designs):
        errors.append(
            f"total_designs is {payload['total_designs']} but {len(designs)} designs are listed"
        )
    sums = {
        "total_nodes": sum(int(record["node_count"]) for record in designs),
        "total_edges": sum(int(record["edge_count"]) for record in designs),
        "total_labels": sum(int(record["label_count"]) for record in designs),
    }
    for field, value in sums.items():
        if int(payload[field]) != value:
            errors.append(f"{field} is {payload[field]} but the designs sum to {value}")
    return errors


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


def _is_within(root: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True
