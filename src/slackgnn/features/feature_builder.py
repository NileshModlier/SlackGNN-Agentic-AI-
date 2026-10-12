"""Build normalized node, timing, and edge features for each design.

The published graph stores a four-way node type and endpoint slack. This
module adds degrees, logic depth, and reported-path membership. Arrival
time, required time, and slack are extracted for supervision and
explainability. They are not copied into the tensors the encoder reads,
because those values are the prediction targets.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import torch
from torch_geometric.data import Data

from slackgnn.dataset.pyg_dataset import NODE_TYPES, load_networkx_graph, load_pyg_dataset
from slackgnn.paths import data_dir
from slackgnn.training.dataset_builder import _attach_task_labels
from slackgnn.training.validate_corpus import discover_designs, reports_dir

logger = logging.getLogger(__name__)

NODE_FEATURE_NAMES = (
    "gate_input",
    "gate_output",
    "gate_register",
    "gate_combinational",
    "register_flag",
    "fanin",
    "fanout",
    "logic_depth",
    "in_degree",
    "out_degree",
)
TIMING_INPUT_NAMES = (
    "logic_depth",
    "critical_path_membership",
    "source_distance",
)
TIMING_TARGET_NAMES = (
    "arrival_time",
    "required_time",
    "slack",
    "critical_path_membership",
)
EDGE_FEATURE_NAMES = (
    "source_input",
    "source_output",
    "source_register",
    "source_combinational",
    "target_input",
    "target_output",
    "target_register",
    "target_combinational",
    "net_delay",
    "topological_distance",
)
_REGISTER = NODE_TYPES.index("register")


def enrich_corpus(
    data_root: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> tuple[list[Data] | None, dict[str, object] | None, tuple[str, ...]]:
    """Attach SlackGNN features to every discovered design and write the summary."""
    root = Path(data_root) if data_root is not None else data_dir()
    graphs: list[Data] = []
    details: list[dict[str, object]] = []
    for name in discover_designs(root):
        data, metadata, errors = load_pyg_dataset(root / name / "pyg")
        if errors or data is None or metadata is None:
            return None, None, errors
        prepared = _attach_task_labels(data, name)
        enriched, detail, enrich_errors = enrich_design(prepared, root / name)
        if enrich_errors or enriched is None or detail is None:
            return None, None, enrich_errors
        graphs.append(enriched)
        details.append(detail)
    if not graphs:
        return None, None, ("No design graphs were discovered",)
    summary = {
        "designs": [str(item["design_name"]) for item in details],
        "node_features": list(NODE_FEATURE_NAMES),
        "timing_features_extracted": list(TIMING_TARGET_NAMES),
        "timing_features_used_as_input": list(TIMING_INPUT_NAMES),
        "timing_features_withheld_from_input": ["arrival_time", "required_time", "slack"],
        "edge_features": list(EDGE_FEATURE_NAMES),
        "normalization": (
            "Count features use log1p divided by that design's maximum. "
            "Logic depth and topological distance are divided by that design's maximum depth."
        ),
        "net_delay": (
            "Structural proxy: log1p of the driver fanout, normalized per design. "
            "Published edges do not store a SPEF net name, so capacitance is not joined."
        ),
        "gate_type": "Four-way node class from the published graph: input, output, register, combinational.",
        "fanin_fanout": (
            "In this signal-flow graph each edge is one driver-to-load connection, "
            "so fanin equals in-degree and fanout equals out-degree."
        ),
        "total_nodes": sum(int(item["node_count"]) for item in details),
        "total_edges": sum(int(item["edge_count"]) for item in details),
        "designs_detail": details,
    }
    destination = Path(summary_output) if summary_output is not None else reports_dir() / "feature_summary.json"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    except OSError as exc:
        return None, None, (f"Cannot write feature summary: {exc}",)
    logger.info("Wrote %s", destination)
    return graphs, summary, ()


def enrich_design(data: Data, design_dir: str | Path) -> tuple[Data | None, dict[str, object] | None, tuple[str, ...]]:
    """Add feature tensors to one published design graph."""
    directory = Path(design_dir)
    graph, graph_errors = load_networkx_graph(directory / "graphs" / "netlist_graph.pkl")
    if graph_errors or graph is None:
        return None, None, graph_errors
    paths, path_errors = _load_paths(directory / "dataset" / "path_dataset.json")
    if path_errors or paths is None:
        return None, None, path_errors
    names = [str(node) for node in graph.nodes]
    if len(names) != int(data.num_nodes):
        return None, None, (
            f"{directory.name} graph has {len(names)} nodes but graph.pt has {int(data.num_nodes)}",
        )
    index = {name: position for position, name in enumerate(names)}
    edge_error = _edge_alignment_error(graph, index, data.edge_index)
    if edge_error:
        return None, None, (edge_error,)
    node_type = data.x.argmax(dim=1)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    in_degree = torch.bincount(dst, minlength=data.num_nodes).to(torch.float32)
    out_degree = torch.bincount(src, minlength=data.num_nodes).to(torch.float32)
    depth, unordered = _logic_depth(src, dst, node_type, int(data.num_nodes))
    children = _children(src, dst, int(data.num_nodes))
    arrival = torch.full((data.num_nodes,), float("nan"))
    required = torch.full((data.num_nodes,), float("nan"))
    membership = torch.zeros(data.num_nodes, dtype=torch.float32)
    timing_paths: list[dict[str, object]] = []
    endpoint_choice: dict[int, dict[str, object]] = {}
    for record in paths:
        start_name = str(record["startpoint"])
        end_name = str(record["endpoint"])
        if start_name not in index or end_name not in index:
            return None, None, (f"Path endpoint is not in the {directory.name} graph: {start_name} -> {end_name}",)
        start = index[start_name]
        end = index[end_name]
        slack = float(record["slack"])
        previous = endpoint_choice.get(end)
        if previous is None or slack < float(previous["slack"]):
            endpoint_choice[end] = {
                "slack": slack,
                "arrival_time": float(record["arrival_time"]),
                "required_time": float(record["required_time"]),
            }
        route = _shortest_path(children, start, end)
        for node in route:
            membership[node] = 1.0
        timing_paths.append(
            {
                "startpoint": start_name,
                "endpoint": end_name,
                "nodes": route,
                "slack": slack,
                "arrival_time": float(record["arrival_time"]),
                "required_time": float(record["required_time"]),
                "path_group": str(record.get("path_group", "")),
                "path_type": str(record.get("path_type", "")),
            }
        )
    for end, chosen in endpoint_choice.items():
        arrival[end] = float(chosen["arrival_time"])
        required[end] = float(chosen["required_time"])
    depth_tensor = torch.tensor(depth, dtype=torch.float32)
    node_features = _node_features(data.x, node_type, in_degree, out_degree, depth_tensor)
    timing_features = _timing_inputs(depth_tensor, membership)
    edge_attr = _edge_features(node_type, src, dst, out_degree, depth_tensor)
    data.node_features = node_features
    data.timing_features = timing_features
    data.edge_attr = edge_attr
    data.arrival_time = arrival
    data.required_time = required
    data.critical_path = membership
    data.node_names = names
    data.timing_paths = timing_paths
    detail = {
        "design_name": directory.name,
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "node_feature_dimension": int(node_features.size(1)),
        "timing_input_dimension": int(timing_features.size(1)),
        "edge_feature_dimension": int(edge_attr.size(1)),
        "max_logic_depth": int(depth_tensor.max().item()) if data.num_nodes else 0,
        "critical_path_nodes": int(membership.sum().item()),
        "endpoints_with_arrival": int(torch.isfinite(arrival).sum().item()),
        "unordered_cyclic_nodes": unordered,
        "label_count": int(data.label_mask.sum().item()),
        "mean_normalized_depth": float(timing_features[:, 0].mean().item()),
        "mean_fanout": float(out_degree.mean().item()),
    }
    logger.info(
        "Features for %s: depth %s, critical nodes %s",
        directory.name,
        detail["max_logic_depth"],
        detail["critical_path_nodes"],
    )
    return data, detail, ()


def _node_features(
    one_hot: torch.Tensor,
    node_type: torch.Tensor,
    in_degree: torch.Tensor,
    out_degree: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    register = (node_type == _REGISTER).to(torch.float32).unsqueeze(1)
    return torch.cat(
        [
            one_hot.to(torch.float32),
            register,
            _unit_log(in_degree).unsqueeze(1),
            _unit_log(out_degree).unsqueeze(1),
            _unit(depth).unsqueeze(1),
            _unit_log(in_degree).unsqueeze(1),
            _unit_log(out_degree).unsqueeze(1),
        ],
        dim=1,
    )


def _timing_inputs(depth: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Return encoder timing channels that do not contain the slack target."""
    normalized = _unit(depth)
    return torch.stack([normalized, membership, normalized], dim=1)


def _edge_features(
    node_type: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    out_degree: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    source = torch.nn.functional.one_hot(node_type[src], num_classes=len(NODE_TYPES)).to(torch.float32)
    target = torch.nn.functional.one_hot(node_type[dst], num_classes=len(NODE_TYPES)).to(torch.float32)
    delay = _unit_log(out_degree)[src]
    distance = _unit(depth)[dst]
    return torch.cat([source, target, delay.unsqueeze(1), distance.unsqueeze(1)], dim=1)


def _logic_depth(src: torch.Tensor, dst: torch.Tensor, node_type: torch.Tensor, nodes: int) -> tuple[list[int], int]:
    """Longest hop count from an input or register, stopping at the next register."""
    children: list[list[int]] = [[] for _ in range(nodes)]
    indegree = [0] * nodes
    register = node_type.tolist()
    for source, target in zip(src.tolist(), dst.tolist(), strict=True):
        if int(register[target]) == _REGISTER:
            continue
        children[source].append(target)
        indegree[target] += 1
    depth = [0] * nodes
    queue = deque(node for node, count in enumerate(indegree) if count == 0)
    ordered = 0
    while queue:
        node = queue.popleft()
        ordered += 1
        step = 1 if int(register[node]) == _REGISTER else depth[node] + 1
        for child in children[node]:
            if step > depth[child]:
                depth[child] = step
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return depth, nodes - ordered


def _children(src: torch.Tensor, dst: torch.Tensor, nodes: int) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(nodes)]
    for source, target in zip(src.tolist(), dst.tolist(), strict=True):
        children[source].append(target)
    return children


def _shortest_path(children: list[list[int]], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
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
                return route
            queue.append(child)
    return [start, end]


def _edge_alignment_error(graph, index: dict[str, int], edge_index: torch.Tensor) -> str | None:
    published = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    extracted = {(index[str(source)], index[str(target)]) for source, target in graph.edges}
    if published != extracted:
        return "Published edge_index does not match graphs/netlist_graph.pkl"
    return None


def _load_paths(path: Path) -> tuple[list[dict[str, object]] | None, tuple[str, ...]]:
    if not path.is_file():
        return None, (f"Missing path dataset: {path}",)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, (f"Cannot read path dataset {path}: {exc}",)
    records = payload.get("paths") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        return None, (f"{path} does not contain timing paths",)
    required = ("startpoint", "endpoint", "slack", "arrival_time", "required_time")
    for record in records:
        if not isinstance(record, dict) or any(key not in record for key in required):
            return None, (f"{path} has a path record without startpoint, endpoint, slack, arrival, and required time",)
    return records, ()


def _unit_log(values: torch.Tensor) -> torch.Tensor:
    scaled = torch.log1p(values)
    peak = float(scaled.max().item()) if scaled.numel() else 0.0
    if peak <= 0:
        return torch.zeros_like(scaled)
    return scaled / peak


def _unit(values: torch.Tensor) -> torch.Tensor:
    peak = float(values.max().item()) if values.numel() else 0.0
    if peak <= 0:
        return torch.zeros_like(values)
    return values / peak
