"""Load published graphs and attach auxiliary M9 targets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from slackgnn.dataset.pyg_dataset import load_pyg_dataset
from slackgnn.expansion.label_expansion import inventory, load_expanded_tensors
from slackgnn.features.feature_builder import enrich_design
from slackgnn.paths import data_dir, repo_root

Pair = tuple[int, int]


def default_tensor_path() -> Path:
    return repo_root() / "reports" / "m9" / "expanded_labels.pt"


def load_study_graphs(
    data_root: str | Path | None = None,
    tensor_path: str | Path | None = None,
) -> tuple[list[Data], dict[str, object]]:
    """Enrich every available design and attach expanded slack targets."""
    root = Path(data_root) if data_root is not None else data_dir()
    found = inventory(root)
    source = Path(tensor_path) if tensor_path is not None else default_tensor_path()
    if not source.is_file():
        raise FileNotFoundError(f"Missing expanded labels: {source}. Run scripts/expand_labels.py first.")
    tensors = load_expanded_tensors(source)
    graphs: list[Data] = []
    skipped: list[dict[str, str]] = []
    for name in found["available"]:
        if name not in tensors:
            skipped.append({"design": str(name), "reason": "expanded labels have no tensor for this design"})
            continue
        data, _metadata, errors = load_pyg_dataset(root / str(name) / "pyg")
        if errors or data is None:
            skipped.append({"design": str(name), "reason": "; ".join(errors)})
            continue
        prepared = _attach_task_labels(data, str(name))
        enriched, _detail, enrich_errors = enrich_design(prepared, root / str(name))
        if enrich_errors or enriched is None:
            skipped.append({"design": str(name), "reason": "; ".join(enrich_errors)})
            continue
        item = tensors[str(name)]
        if int(item["slack"].shape[0]) != int(enriched.num_nodes):
            skipped.append({"design": str(name), "reason": "expanded slack length does not match the graph"})
            continue
        enriched.expanded_slack = item["slack"].to(torch.float32).clone()
        enriched.expanded_mask = item["mask"].to(torch.bool).clone()
        violation = torch.zeros(enriched.num_nodes, dtype=torch.float32)
        labeled = enriched.expanded_mask & torch.isfinite(enriched.expanded_slack)
        violation[labeled & (enriched.expanded_slack < 0)] = 1.0
        enriched.expanded_violation = violation
        enriched.expanded_mask = labeled
        graphs.append(enriched)
    _check_feature_widths(graphs)
    found["loaded"] = [str(graph.design_name) for graph in graphs]
    found["skipped"] = skipped
    return graphs, found


def both_endpoint_classes(graph: Data) -> bool:
    """Return True when the published endpoint labels contain a met and a violation."""
    labeled = graph.label_mask
    violations = int((labeled & (graph.y_violation == 1)).sum().item())
    met = int((labeled & (graph.y_violation == 0)).sum().item())
    return violations > 0 and met > 0


def choose_scaling_holdout(graphs: list[Data]) -> tuple[str | None, str]:
    """Prefer jpeg when it has both endpoint classes, otherwise the largest such design."""
    by_name = {str(graph.design_name): graph for graph in graphs}
    if "jpeg" in by_name and both_endpoint_classes(by_name["jpeg"]):
        return "jpeg", "jpeg is present and has both endpoint classes"
    candidates = [graph for graph in graphs if both_endpoint_classes(graph)]
    if not candidates:
        return None, "no available design has both a timing-met endpoint and a timing-violation endpoint"
    chosen = max(candidates, key=lambda graph: int(graph.num_nodes))
    return str(chosen.design_name), f"{chosen.design_name} is the largest available design with both endpoint classes"


def _attach_task_labels(data: Data, design_name: str) -> Data:
    violation = torch.zeros(data.num_nodes, dtype=torch.float32)
    violation[data.negative_slack_mask] = 1.0
    data.design_name = design_name
    data.y_slack = data.y.clone()
    data.y_violation = violation
    return data


def _check_feature_widths(graphs: list[Data]) -> None:
    if not graphs:
        return
    widths = {
        (
            int(graph.node_features.size(1)),
            int(graph.edge_attr.size(1)),
            int(graph.timing_features.size(1)),
        )
        for graph in graphs
    }
    if len(widths) != 1:
        raise ValueError(f"Feature widths differ across designs: {sorted(widths)}")
