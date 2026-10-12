"""Merge discovered design graphs into one training collection.

Each graph keeps its own nodes and edges. Labels stay on the endpoint
that already carries slack. A violation label is 1 when slack is negative
and 0 when slack meets the constraint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch_geometric.data import Data

from slackgnn.dataset.pyg_dataset import load_pyg_dataset
from slackgnn.paths import data_dir
from slackgnn.training.validate_corpus import discover_designs, reports_dir, validate_corpus

logger = logging.getLogger(__name__)


def build_training_dataset(
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    """Load passing designs and write ``training_data.pt`` plus a summary."""
    root = Path(data_root) if data_root is not None else data_dir()
    validation = validate_corpus(root)
    if not validation["designs"]:
        return None, ("No design graphs passed corpus validation",)
    graphs: list[Data] = []
    per_design: list[dict[str, object]] = []
    for name in validation["designs"]:
        data, metadata, errors = load_pyg_dataset(root / str(name) / "pyg")
        if errors or data is None or metadata is None:
            return None, errors
        prepared = _attach_task_labels(data, str(name))
        graphs.append(prepared)
        labeled = prepared.label_mask
        per_design.append(
            {
                "design_name": str(name),
                "node_count": int(prepared.num_nodes),
                "edge_count": int(prepared.edge_index.size(1)),
                "feature_dimension": int(prepared.x.size(1)),
                "label_count": int(labeled.sum().item()),
                "violation_count": int(prepared.y_violation[labeled].sum().item()),
                "met_count": int((labeled & (prepared.y_violation == 0)).sum().item()),
                "register_count": int(metadata["register_count"]),
                "combinational_count": int(metadata["combinational_count"]),
            }
        )
    summary = {
        "designs": [item["design_name"] for item in per_design],
        "graph_count": len(graphs),
        "total_nodes": sum(int(item["node_count"]) for item in per_design),
        "total_edges": sum(int(item["edge_count"]) for item in per_design),
        "feature_dimension": int(graphs[0].x.size(1)),
        "label_count": sum(int(item["label_count"]) for item in per_design),
        "violation_count": sum(int(item["violation_count"]) for item in per_design),
        "met_count": sum(int(item["met_count"]) for item in per_design),
        "designs_detail": per_design,
    }
    destination = Path(output) if output is not None else reports_dir() / "training_data.pt"
    summary_path = (
        Path(summary_output) if summary_output is not None else reports_dir() / "training_dataset_summary.json"
    )
    write_errors = _write_dataset(graphs, summary, destination, summary_path)
    if write_errors:
        return None, write_errors
    return summary, ()


def load_training_dataset(path: str | Path) -> tuple[list[Data] | None, dict[str, object] | None, tuple[str, ...]]:
    """Load ``training_data.pt``."""
    source = Path(path)
    if not source.is_file():
        return None, None, (f"Missing training dataset: {source}",)
    try:
        payload = torch.load(source, weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        return None, None, (f"Cannot load training dataset {source}: {exc}",)
    if not isinstance(payload, dict) or "graphs" not in payload or "summary" not in payload:
        return None, None, (f"{source} does not contain graphs and summary",)
    graphs = payload["graphs"]
    summary = payload["summary"]
    if not isinstance(graphs, list) or not graphs:
        return None, None, (f"{source} has no graphs",)
    return graphs, summary, ()


def _attach_task_labels(data: Data, design_name: str) -> Data:
    violation = torch.zeros(data.num_nodes, dtype=torch.float32)
    violation[data.negative_slack_mask] = 1.0
    data.design_name = design_name
    data.y_slack = data.y.clone()
    data.y_violation = violation
    return data


def _write_dataset(
    graphs: list[Data],
    summary: dict[str, object],
    destination: Path,
    summary_path: Path,
) -> tuple[str, ...]:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graphs": graphs, "summary": summary}, destination)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    except OSError as exc:
        message = f"Cannot write training dataset: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    logger.info("Wrote %s", summary_path)
    return ()
