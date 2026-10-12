"""Rank gates and reported paths by SlackGNN attention."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def explain_design(data: Data, attention: torch.Tensor, limit: int = 15) -> dict[str, object]:
    """Summarize one design from the incoming-edge attention of the last layer."""
    source = data.edge_index[0]
    target = data.edge_index[1]
    influence = torch.zeros(data.num_nodes, dtype=torch.float32)
    received = torch.zeros(data.num_nodes, dtype=torch.float32)
    influence.index_add_(0, source, attention.detach().cpu())
    received.index_add_(0, target, attention.detach().cpu())
    names = list(data.node_names)
    node_type = data.x.argmax(dim=1).tolist()
    type_names = ("input", "output", "register", "combinational")
    ranking = _rank(names, node_type, type_names, influence, received, data.critical_path, limit)
    gates = [row for row in ranking if row["node_type"] == "combinational"][:limit]
    paths = _paths(data, attention.detach().cpu(), names)
    heatmap = _heatmap(data, attention.detach().cpu(), names, influence)
    return {
        "design_name": str(data.design_name),
        "node_importance": ranking,
        "critical_gate_rankings": gates,
        "influential_paths": paths,
        "attention_heatmap": heatmap,
    }


def _rank(
    names: list[str],
    node_type: list[int],
    type_names: tuple[str, ...],
    influence: torch.Tensor,
    received: torch.Tensor,
    critical: torch.Tensor,
    limit: int,
) -> list[dict[str, object]]:
    order = torch.argsort(influence, descending=True).tolist()
    rows: list[dict[str, object]] = []
    for node in order:
        if len(rows) >= limit:
            break
        if float(influence[node].item()) <= 0 and float(critical[node].item()) <= 0:
            continue
        rows.append(
            {
                "node": names[node],
                "node_type": type_names[int(node_type[node])],
                "influence": float(influence[node].item()),
                "received_attention": float(received[node].item()),
                "on_reported_path": bool(float(critical[node].item()) > 0),
            }
        )
    return rows


def _paths(data: Data, attention: torch.Tensor, names: list[str]) -> list[dict[str, object]]:
    lookup = {
        (int(source), int(target)): float(weight)
        for source, target, weight in zip(
            data.edge_index[0].tolist(),
            data.edge_index[1].tolist(),
            attention.tolist(),
            strict=True,
        )
    }
    rows: list[dict[str, object]] = []
    for record in data.timing_paths:
        nodes = [int(node) for node in record["nodes"]]
        weights = [lookup.get((nodes[index], nodes[index + 1]), 0.0) for index in range(len(nodes) - 1)]
        mean = sum(weights) / len(weights) if weights else 0.0
        rows.append(
            {
                "startpoint": record["startpoint"],
                "endpoint": record["endpoint"],
                "path_group": record["path_group"],
                "path_type": record["path_type"],
                "slack": record["slack"],
                "hop_count": max(len(nodes) - 1, 0),
                "mean_attention": mean,
                "gates": [names[node] for node in nodes[:12]],
            }
        )
    rows.sort(key=lambda row: float(row["mean_attention"]), reverse=True)
    return rows


def _heatmap(data: Data, attention: torch.Tensor, names: list[str], influence: torch.Tensor) -> dict[str, object]:
    critical = data.critical_path.nonzero(as_tuple=False).view(-1).tolist()
    if len(critical) > 24:
        ranked = sorted(critical, key=lambda node: float(influence[node].item()), reverse=True)
        critical = ranked[:24]
    position = {node: index for index, node in enumerate(critical)}
    matrix = [[0.0 for _ in critical] for _ in critical]
    for source, target, weight in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist(), attention.tolist(), strict=True):
        if source in position and target in position:
            matrix[position[source]][position[target]] = float(weight)
    return {"nodes": [names[node] for node in critical], "matrix": matrix}
