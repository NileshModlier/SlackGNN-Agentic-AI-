"""Check every discovered design graph against its metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from slackgnn.dataset.pyg_dataset import load_pyg_dataset
from slackgnn.paths import data_dir, repo_root

logger = logging.getLogger(__name__)

_SKIP_DIRECTORIES = frozenset({"corpus"})


def reports_dir() -> Path:
    """Return the training-report directory, creating it when asked to write."""
    return repo_root() / "reports"


def discover_designs(data_root: str | Path | None = None) -> tuple[str, ...]:
    """Return design names that already have ``pyg/graph.pt`` and ``metadata.json``."""
    root = Path(data_root) if data_root is not None else data_dir()
    if not root.is_dir():
        return ()
    names: list[str] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name in _SKIP_DIRECTORIES or child.name.startswith("."):
            continue
        pyg = child / "pyg"
        if (pyg / "graph.pt").is_file() and (pyg / "metadata.json").is_file():
            names.append(child.name)
    return tuple(names)


def validate_corpus(data_root: str | Path | None = None) -> dict[str, object]:
    """Load each discovered graph and compare it with ``metadata.json``."""
    root = Path(data_root) if data_root is not None else data_dir()
    design_reports: list[dict[str, object]] = []
    for name in discover_designs(root):
        design_reports.append(_validate_design(root / name / "pyg", name))
    passed = [item for item in design_reports if item["passed"]]
    report: dict[str, object] = {
        "passed": bool(design_reports) and all(bool(item["passed"]) for item in design_reports),
        "designs": [str(item["design_name"]) for item in passed],
        "design_count": len(passed),
        "node_totals": sum(int(item["node_count"]) for item in passed),
        "edge_totals": sum(int(item["edge_count"]) for item in passed),
        "label_totals": sum(int(item["label_count"]) for item in passed),
        "positive_labels": sum(int(item["positive_slack_count"]) for item in passed),
        "negative_labels": sum(int(item["negative_slack_count"]) for item in passed),
        "design_reports": design_reports,
    }
    return report


def write_validation_report(report: dict[str, object], destination: str | Path | None = None) -> Path:
    """Write ``reports/corpus_validation_report.json``."""
    path = Path(destination) if destination is not None else reports_dir() / "corpus_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    """Validate the corpus and write the JSON report."""
    parser = argparse.ArgumentParser(description="Validate discovered PyG design graphs.")
    parser.add_argument("--data-root", type=Path, default=data_dir())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_corpus(args.data_root)
    path = write_validation_report(report, args.output)
    print(f"Designs: {report['design_count']}")
    print(f"Nodes: {report['node_totals']}")
    print(f"Edges: {report['edge_totals']}")
    print(f"Labels: {report['label_totals']}")
    print(f"Positive labels: {report['positive_labels']}")
    print(f"Negative labels: {report['negative_labels']}")
    print(f"Passed: {report['passed']}")
    print(f"Wrote {path}")
    return 0 if report["passed"] else 1


def _validate_design(pyg_dir: Path, name: str) -> dict[str, object]:
    data, metadata, errors = load_pyg_dataset(pyg_dir)
    record: dict[str, object] = {
        "design_name": name,
        "passed": False,
        "errors": list(errors),
        "node_count": 0,
        "edge_count": 0,
        "label_count": 0,
        "positive_slack_count": 0,
        "negative_slack_count": 0,
    }
    if errors or data is None or metadata is None:
        return record
    record["errors"] = _consistency_errors(data, metadata)
    record["passed"] = not record["errors"]
    record["node_count"] = int(data.num_nodes)
    record["edge_count"] = int(data.edge_index.size(1))
    record["label_count"] = int(data.label_mask.sum().item())
    record["positive_slack_count"] = int(data.positive_slack_mask.sum().item())
    record["negative_slack_count"] = int(data.negative_slack_mask.sum().item())
    if not record["passed"]:
        logger.error("%s failed corpus validation: %s", name, "; ".join(record["errors"]))
    else:
        logger.info(
            "Validated %s: %s nodes, %s edges, %s labels",
            name,
            record["node_count"],
            record["edge_count"],
            record["label_count"],
        )
    return record


def _consistency_errors(data, metadata: dict) -> list[str]:
    errors: list[str] = []
    required = (
        "node_count",
        "edge_count",
        "label_count",
        "positive_slack_count",
        "negative_slack_count",
    )
    for key in required:
        if key not in metadata:
            errors.append(f"metadata.json is missing {key}")
    if data.x is None or data.edge_index is None or data.y is None:
        return ["graph.pt is missing x, edge_index, or y"]
    if errors:
        return errors
    checks = {
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "label_count": int(data.label_mask.sum().item()),
        "positive_slack_count": int(data.positive_slack_mask.sum().item()),
        "negative_slack_count": int(data.negative_slack_mask.sum().item()),
    }
    for key, value in checks.items():
        if int(metadata[key]) != value:
            errors.append(f"metadata {key} is {metadata[key]} but graph.pt has {value}")
    if data.x.dim() != 2 or data.x.size(0) != checks["node_count"]:
        errors.append(f"feature shape is {tuple(data.x.shape)}")
    if "feature_dimension" in metadata and int(metadata["feature_dimension"]) != int(data.x.size(1)):
        errors.append(f"metadata feature_dimension is {metadata['feature_dimension']} but graph.pt has {data.x.size(1)}")
    if tuple(data.edge_index.shape) != (2, checks["edge_count"]):
        errors.append(f"edge_index shape is {tuple(data.edge_index.shape)}")
    if data.y.shape != (checks["node_count"],):
        errors.append(f"slack shape is {tuple(data.y.shape)}")
    labeled = data.label_mask
    slack = data.y
    if torch.isnan(slack[labeled]).any():
        errors.append("a labeled node has NaN slack")
    if (~labeled).any() and not torch.isnan(slack[~labeled]).all():
        errors.append("an unlabeled node has a numeric slack")
    if not torch.equal(data.positive_slack_mask, labeled & (slack > 0)):
        errors.append("positive slack mask does not match slack > 0")
    if not torch.equal(data.negative_slack_mask, labeled & (slack < 0)):
        errors.append("negative slack mask does not match slack < 0")
    if int(data.positive_slack_mask.sum()) + int(data.negative_slack_mask.sum()) > int(labeled.sum()):
        errors.append("positive and negative masks exceed the labeled nodes")
    return errors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
