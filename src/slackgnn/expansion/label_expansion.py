"""Propagate reported path slack onto signal-flow routes.

The published ``graph.pt`` files keep endpoint slack from the path table.
This module does not rewrite those files and does not change the timing
parser. It rereads each finish report and path table, then stores an
auxiliary target for every node on the shortest driver-to-load route.
A node that lies on more than one route keeps the worst slack. Published
endpoint labels are written back after that propagation so the auxiliary
file still carries the original endpoint measurement.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import deque
from pathlib import Path

import torch

from slackgnn.dataset.pyg_dataset import load_networkx_graph, load_path_dataset, load_pyg_dataset
from slackgnn.dataset.timing_report import parse_timing_report
from slackgnn.paths import data_dir, repo_root
from slackgnn.training.validate_corpus import discover_designs

logger = logging.getLogger(__name__)

REQUESTED_DESIGNS = ("gcd", "aes", "ibex", "jpeg", "riscv32i", "microwatt", "chameleon")
_SLACK_DECIMALS = 4


def inventory(data_root: str | Path | None = None) -> dict[str, object]:
    """List requested designs that have ``pyg/graph.pt`` and those that do not."""
    root = Path(data_root) if data_root is not None else data_dir()
    present = set(discover_designs(root))
    ordered = [name for name in REQUESTED_DESIGNS if name in present]
    ordered.extend(name for name in discover_designs(root) if name not in REQUESTED_DESIGNS)
    missing: list[dict[str, str]] = []
    for name in REQUESTED_DESIGNS:
        if name in present:
            continue
        directory = root / name
        if not directory.is_dir():
            reason = "design directory is missing"
        elif not (directory / "pyg" / "graph.pt").is_file():
            reason = "pyg/graph.pt is missing"
        else:
            reason = "pyg/metadata.json is missing"
        missing.append({"design": name, "reason": reason})
    return {"data_root": str(root), "available": ordered, "missing": missing}


def propagate_slack(
    num_nodes: int,
    edge_index: torch.Tensor,
    paths: list[dict[str, object]],
    name_to_index: dict[str, int],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Assign each route the path slack and keep the minimum on shared nodes.

    Unreachable endpoints stay in the route as the two pins only. The returned
    tensor is NaN where no mapped path touches a node.
    """
    slack = torch.full((num_nodes,), float("nan"))
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    for source, target in zip(sources, targets, strict=True):
        children[int(source)].append(int(target))
    routes: list[dict[str, object]] = []
    for path in paths:
        start_name = str(path["startpoint"])
        end_name = str(path["endpoint"])
        value = float(path["slack"])
        record = {
            "startpoint": start_name,
            "endpoint": end_name,
            "path_group": str(path.get("path_group", "")),
            "path_type": str(path.get("path_type", "")),
            "slack": value,
            "source": str(path.get("source", "")),
            "mapped": False,
            "disconnected": False,
            "route_length": 0,
            "node_indices": [],
        }
        if start_name not in name_to_index or end_name not in name_to_index:
            routes.append(record)
            continue
        start = name_to_index[start_name]
        end = name_to_index[end_name]
        route, disconnected = _shortest_route(children, start, end)
        for node in route:
            current = float(slack[node].item())
            if math.isnan(current) or value < current:
                slack[node] = value
        record["mapped"] = True
        record["disconnected"] = disconnected
        record["route_length"] = len(route)
        record["node_indices"] = route
        routes.append(record)
    return slack, routes


def restore_endpoint_slack(
    propagated: torch.Tensor,
    original_slack: torch.Tensor,
    original_mask: torch.Tensor,
) -> torch.Tensor:
    """Put published endpoint slack back onto labeled endpoints."""
    restored = propagated.clone()
    keep = original_mask.bool() & torch.isfinite(original_slack)
    restored[keep] = original_slack[keep].to(restored.dtype)
    return restored


def merge_path_records(
    dataset_paths: list[dict[str, object]],
    report_paths: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    """Union path records. Report rows already stored in the dataset are not duplicated."""
    merged: list[dict[str, object]] = []
    seen: dict[tuple[object, ...], dict[str, object]] = {}
    for path in dataset_paths:
        item = dict(path)
        item["source"] = "dataset"
        seen[_path_key(item)] = item
        merged.append(item)
    report_only = 0
    for path in report_paths:
        key = _path_key(path)
        if key in seen:
            seen[key]["source"] = "report+dataset"
            continue
        item = dict(path)
        item["source"] = "report"
        seen[key] = item
        merged.append(item)
        report_only += 1
    return merged, report_only


def expand_design(design_dir: str | Path) -> dict[str, object]:
    """Build auxiliary slack targets for one design directory.

    This function only reads the design. It does not write ``graph.pt``.
    """
    directory = Path(design_dir)
    name = directory.name
    data, metadata, load_errors = load_pyg_dataset(directory / "pyg")
    if load_errors or data is None or metadata is None:
        return _failed(name, load_errors or ("Cannot load PyG graph",))
    graph, graph_errors = load_networkx_graph(directory / "graphs" / "netlist_graph.pkl")
    if graph_errors or graph is None:
        return _failed(name, graph_errors or ("Cannot load netlist graph",))
    names = [str(node) for node in graph.nodes]
    if len(names) != int(data.num_nodes):
        return _failed(
            name,
            (f"{name} graph has {len(names)} nodes but graph.pt has {int(data.num_nodes)}",),
        )
    name_to_index = {node_name: position for position, node_name in enumerate(names)}
    alignment = _edge_alignment_error(graph, name_to_index, data.edge_index)
    if alignment:
        return _failed(name, (alignment,))
    dataset_paths, dataset_note = _dataset_paths(directory / "dataset" / "path_dataset.json")
    report_paths, report_note, report_blocks = _report_paths(directory / "reports" / "6_finish.rpt")
    if not dataset_paths and not report_paths:
        notes = tuple(note for note in (dataset_note, report_note) if note)
        return _failed(name, notes or (f"{name} has no timing paths",))
    merged, report_only = merge_path_records(dataset_paths, report_paths)
    original_slack = data.y.detach().to(torch.float32).clone()
    original_mask = data.label_mask.detach().to(torch.bool).clone()
    propagated, routes = propagate_slack(int(data.num_nodes), data.edge_index, merged, name_to_index)
    slack = restore_endpoint_slack(propagated, original_slack, original_mask)
    mask = torch.isfinite(slack)
    labeled_nodes = _labeled_nodes(names, slack, mask, original_mask)
    mapped_lengths = [int(route["route_length"]) for route in routes if route["mapped"]]
    summary = {
        "design_name": name,
        "present": True,
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "original_label_count": int(original_mask.sum().item()),
        "expanded_label_count": int(mask.sum().item()),
        "added_label_count": int((mask & ~original_mask).sum().item()),
        "original_violation_count": _class_count(original_slack, original_mask, negative=True),
        "original_met_count": _class_count(original_slack, original_mask, negative=False),
        "expanded_violation_count": _class_count(slack, mask, negative=True),
        "expanded_met_count": _class_count(slack, mask, negative=False),
        "dataset_path_count": len(dataset_paths),
        "report_path_blocks": report_blocks,
        "report_path_count": len(report_paths),
        "unique_path_count": len(merged),
        "paths_only_in_report": report_only,
        "mapped_path_count": sum(1 for route in routes if route["mapped"]),
        "unmapped_path_count": sum(1 for route in routes if not route["mapped"]),
        "disconnected_path_count": sum(1 for route in routes if route["disconnected"]),
        "endpoints_restored": int((original_mask & torch.isfinite(original_slack)).sum().item()),
        "endpoints_touched_by_worse_path": _worse_endpoint_count(propagated, original_slack, original_mask),
        "route_length_min": min(mapped_lengths) if mapped_lengths else 0,
        "route_length_median": statistics.median(mapped_lengths) if mapped_lengths else 0,
        "route_length_max": max(mapped_lengths) if mapped_lengths else 0,
        "dataset_note": dataset_note,
        "report_note": report_note,
        "errors": [],
    }
    logger.info(
        "%s labels %s -> %s (%s path nodes added, %s report-only paths)",
        name,
        summary["original_label_count"],
        summary["expanded_label_count"],
        summary["added_label_count"],
        summary["paths_only_in_report"],
    )
    return {
        "summary": summary,
        "slack": slack,
        "mask": mask,
        "original_slack": original_slack,
        "original_mask": original_mask,
        "node_names": names,
        "labeled_nodes": labeled_nodes,
        "routes": [{key: value for key, value in route.items() if key != "node_indices"} | {"node_indices": list(route["node_indices"])} for route in routes],
    }


def expand_corpus(data_root: str | Path | None = None) -> dict[str, object]:
    """Expand every available design and record requested designs that are absent."""
    root = Path(data_root) if data_root is not None else data_dir()
    found = inventory(root)
    designs: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    for name in found["available"]:
        expanded = expand_design(root / str(name))
        designs[str(name)] = expanded
        summaries.append(dict(expanded["summary"]))
    payload = {
        "requested_designs": list(REQUESTED_DESIGNS),
        "missing_designs": list(found["missing"]),
        "designs": designs,
        "summary": _corpus_summary(summaries, list(found["missing"])),
    }
    return payload


def write_expanded_labels(
    payload: dict[str, object],
    json_path: str | Path | None = None,
    tensor_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the JSON summary and the tensor file under ``reports/m9``."""
    summary_path = Path(json_path) if json_path is not None else repo_root() / "reports" / "m9" / "expanded_labels.json"
    pt_path = Path(tensor_path) if tensor_path is not None else repo_root() / "reports" / "m9" / "expanded_labels.pt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    document = _json_document(payload)
    summary_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    tensors = {
        "designs": {
            name: {
                "slack": item["slack"],
                "mask": item["mask"],
                "original_slack": item["original_slack"],
                "original_mask": item["original_mask"],
            }
            for name, item in payload["designs"].items()
            if item.get("slack") is not None
        }
    }
    torch.save(tensors, pt_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", pt_path)
    return summary_path, pt_path


def load_expanded_tensors(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load auxiliary slack tensors keyed by design name."""
    loaded = torch.load(Path(path), weights_only=False)
    if not isinstance(loaded, dict) or "designs" not in loaded:
        raise ValueError(f"{path} does not contain expanded-label tensors")
    return loaded["designs"]


def _dataset_paths(path: Path) -> tuple[list[dict[str, object]], str | None]:
    if not path.is_file():
        return [], f"Missing path dataset: {path}"
    records, errors = load_path_dataset(path)
    if errors:
        return [], "; ".join(errors)
    paths: list[dict[str, object]] = []
    for record in records:
        item = _record_from_mapping(record)
        if item is not None:
            paths.append(item)
    return paths, None


def _report_paths(path: Path) -> tuple[list[dict[str, object]], str | None, int]:
    if not path.is_file():
        return [], f"Missing timing report: {path}", 0
    parsed = parse_timing_report(path)
    if parsed.errors:
        return [], "; ".join(parsed.errors), parsed.path_blocks
    paths = [
        {
            "startpoint": item.startpoint,
            "endpoint": item.endpoint,
            "path_group": item.path_group,
            "path_type": item.path_type,
            "slack": float(item.slack),
        }
        for item in parsed.paths
    ]
    return paths, None, parsed.path_blocks


def _record_from_mapping(record: dict[str, object]) -> dict[str, object] | None:
    if "startpoint" not in record or "endpoint" not in record or "slack" not in record:
        return None
    return {
        "startpoint": str(record["startpoint"]),
        "endpoint": str(record["endpoint"]),
        "path_group": str(record.get("path_group", "")),
        "path_type": str(record.get("path_type", "")),
        "slack": float(record["slack"]),
    }


def _path_key(path: dict[str, object]) -> tuple[object, ...]:
    return (
        str(path["startpoint"]),
        str(path["endpoint"]),
        str(path.get("path_group", "")),
        str(path.get("path_type", "")),
        round(float(path["slack"]), _SLACK_DECIMALS),
    )


def _shortest_route(children: list[list[int]], start: int, end: int) -> tuple[list[int], bool]:
    if start == end:
        return [start], False
    parent = {start: -1}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if child in parent:
                continue
            parent[child] = node
            if child == end:
                route = [end]
                while route[-1] != start:
                    route.append(parent[route[-1]])
                route.reverse()
                return route, False
            queue.append(child)
    return [start, end], True


def _edge_alignment_error(graph, index: dict[str, int], edge_index: torch.Tensor) -> str | None:
    published = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    extracted = {(index[str(source)], index[str(target)]) for source, target in graph.edges}
    if published != extracted:
        return "Published edge_index does not match graphs/netlist_graph.pkl"
    return None


def _labeled_nodes(
    names: list[str],
    slack: torch.Tensor,
    mask: torch.Tensor,
    original_mask: torch.Tensor,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in mask.nonzero(as_tuple=False).view(-1).tolist():
        rows.append(
            {
                "node_index": int(node),
                "node_name": names[int(node)],
                "slack": float(slack[int(node)].item()),
                "original_endpoint": bool(original_mask[int(node)].item()),
            }
        )
    return rows


def _class_count(slack: torch.Tensor, mask: torch.Tensor, *, negative: bool) -> int:
    if int(mask.sum().item()) == 0:
        return 0
    values = slack[mask]
    if negative:
        return int((values < 0).sum().item())
    return int((values >= 0).sum().item())


def _worse_endpoint_count(propagated: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> int:
    keep = mask.bool() & torch.isfinite(original) & torch.isfinite(propagated)
    if int(keep.sum().item()) == 0:
        return 0
    return int((propagated[keep] < original[keep] - 1e-4).sum().item())


def _failed(name: str, errors: tuple[str, ...]) -> dict[str, object]:
    logger.error("%s label expansion failed: %s", name, "; ".join(errors))
    summary = {
        "design_name": name,
        "present": False,
        "original_label_count": 0,
        "expanded_label_count": 0,
        "added_label_count": 0,
        "errors": list(errors),
    }
    return {
        "summary": summary,
        "slack": None,
        "mask": None,
        "original_slack": None,
        "original_mask": None,
        "node_names": [],
        "labeled_nodes": [],
        "routes": [],
    }


def _corpus_summary(summaries: list[dict[str, object]], missing: list[dict[str, str]]) -> dict[str, object]:
    usable = [item for item in summaries if item.get("present")]
    return {
        "designs": summaries,
        "missing_designs": missing,
        "design_count": len(usable),
        "original_label_count": sum(int(item["original_label_count"]) for item in usable),
        "expanded_label_count": sum(int(item["expanded_label_count"]) for item in usable),
        "added_label_count": sum(int(item.get("added_label_count", 0)) for item in usable),
        "node_count": sum(int(item.get("node_count", 0)) for item in usable),
        "edge_count": sum(int(item.get("edge_count", 0)) for item in usable),
        "paths_only_in_report": sum(int(item.get("paths_only_in_report", 0)) for item in usable),
        "method": (
            "Each reported startpoint-to-endpoint pair donates its slack to the nodes on the "
            "shortest signal-flow route. Shared nodes keep the minimum slack. Published endpoint "
            "slack from graph.pt is restored afterward. Propagated slack is not a new static-timing measurement."
        ),
    }


def _json_document(payload: dict[str, object]) -> dict[str, object]:
    designs: list[dict[str, object]] = []
    for name, item in payload["designs"].items():
        summary = dict(item["summary"])
        summary["labeled_nodes"] = list(item["labeled_nodes"])
        summary["routes"] = [
            {key: value for key, value in route.items() if key != "node_indices"}
            for route in item["routes"]
        ]
        designs.append(summary)
    summary = dict(payload["summary"])
    summary["designs"] = designs
    return {
        "requested_designs": list(payload["requested_designs"]),
        "missing_designs": list(payload["missing_designs"]),
        "summary": {
            key: value
            for key, value in summary.items()
            if key != "designs"
        },
        "designs": designs,
    }
