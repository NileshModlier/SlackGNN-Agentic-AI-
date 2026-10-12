"""Convert the gate-level signal-flow graph into one PyTorch Geometric graph.

Node features are a one-hot encoding of ``node_type``. Slack from the path
table is stored in ``y``, separate from ``x``, with masks for labeled,
positive, and negative slack. This module does not define a GNN or a
training loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data

from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

NODE_TYPES = ("input", "output", "register", "combinational")
_FEATURE_INDEX = {name: index for index, name in enumerate(NODE_TYPES)}


def one_hot_node_type(node_type: str) -> list[int]:
    """Return the four-way one-hot row for ``node_type``."""
    try:
        index = _FEATURE_INDEX[node_type]
    except KeyError as exc:
        known = ", ".join(NODE_TYPES)
        raise ValueError(f"Unknown node_type {node_type!r}. Expected one of: {known}.") from exc
    row = [0, 0, 0, 0]
    row[index] = 1
    return row


def load_path_dataset(path: str | Path) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    """Load path records from ``path_dataset.json``."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"Cannot read path dataset {source}: {exc}"
        logger.error(message)
        return (), (message,)
    except json.JSONDecodeError as exc:
        message = f"Path dataset {source} is not valid JSON: {exc}"
        logger.error(message)
        return (), (message,)

    records = payload.get("paths") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        return (), (f"Path dataset {source} does not contain a non-empty paths list",)

    parsed: list[dict[str, object]] = []
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        item, error = _parse_path_record(record, index)
        if error or item is None:
            errors.append(f"{source} {error}")
            continue
        parsed.append(item)
    if errors:
        return (), tuple(errors)
    return tuple(parsed), ()


def load_networkx_graph(path: str | Path) -> tuple[nx.DiGraph | None, tuple[str, ...]]:
    """Load the pickled signal-flow graph."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            graph = pickle.load(handle)
    except OSError as exc:
        message = f"Cannot read graph {source}: {exc}"
        logger.error(message)
        return None, (message,)
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError) as exc:
        message = f"Cannot unpickle graph {source}: {exc}"
        logger.error(message)
        return None, (message,)
    if not isinstance(graph, nx.DiGraph):
        return None, (f"{source} is not a networkx.DiGraph",)
    return graph, ()


def build_pyg_data(
    graph: nx.DiGraph,
    paths: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> tuple[Data | None, dict[str, int] | None, tuple[str, ...]]:
    """Build a ``Data`` object and the metadata counts."""
    if graph.number_of_nodes() == 0:
        return None, None, ("Graph has no nodes",)
    if not paths:
        return None, None, ("No timing paths to attach",)

    nodes = list(graph.nodes)
    index_of = {name: position for position, name in enumerate(nodes)}
    errors: list[str] = []
    features: list[list[int]] = []
    counts = {name: 0 for name in NODE_TYPES}
    for name in nodes:
        attributes = graph.nodes[name]
        node_type = attributes.get("node_type")
        if node_type not in _FEATURE_INDEX:
            errors.append(f"Node {name} has node_type {node_type!r}")
            continue
        counts[str(node_type)] += 1
        features.append(one_hot_node_type(str(node_type)))
    if errors:
        return None, None, tuple(errors)

    slack_by_endpoint: dict[str, float] = {}
    for record in paths:
        endpoint = str(record["endpoint"])
        slack = float(record["slack"])
        if endpoint not in index_of:
            errors.append(f"Label endpoint is not in the graph: {endpoint}")
            continue
        startpoint = record.get("startpoint")
        if startpoint is not None and str(startpoint) not in index_of:
            errors.append(f"Label startpoint is not in the graph: {startpoint}")
        if endpoint in slack_by_endpoint:
            previous = slack_by_endpoint[endpoint]
            kept = min(previous, slack)
            logger.info(
                "Endpoint %s has slack labels %s and %s; keeping the worst slack %s",
                endpoint,
                previous,
                slack,
                kept,
            )
            slack_by_endpoint[endpoint] = kept
            continue
        slack_by_endpoint[endpoint] = slack
    if errors:
        return None, None, tuple(errors)

    sources: list[int] = []
    targets: list[int] = []
    for source, target in graph.edges:
        sources.append(index_of[source])
        targets.append(index_of[target])

    y_values = [float("nan")] * len(nodes)
    label_mask = [False] * len(nodes)
    positive_mask = [False] * len(nodes)
    negative_mask = [False] * len(nodes)
    for endpoint, slack in slack_by_endpoint.items():
        position = index_of[endpoint]
        y_values[position] = slack
        label_mask[position] = True
        positive_mask[position] = slack > 0
        negative_mask[position] = slack < 0

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        y=torch.tensor(y_values, dtype=torch.float32),
        label_mask=torch.tensor(label_mask, dtype=torch.bool),
        positive_slack_mask=torch.tensor(positive_mask, dtype=torch.bool),
        negative_slack_mask=torch.tensor(negative_mask, dtype=torch.bool),
    )
    metadata = {
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "feature_dimension": int(data.x.size(1)),
        "label_count": int(data.label_mask.sum().item()),
        "register_count": counts["register"],
        "combinational_count": counts["combinational"],
        "positive_slack_count": int(data.positive_slack_mask.sum().item()),
        "negative_slack_count": int(data.negative_slack_mask.sum().item()),
    }
    logger.info(
        "Built PyG graph: %d nodes, %d edges, %d labels",
        metadata["node_count"],
        metadata["edge_count"],
        metadata["label_count"],
    )
    return data, metadata, ()


def save_pyg_dataset(data: Data, metadata: dict[str, int], output_dir: str | Path) -> tuple[str, ...]:
    """Write ``graph.pt`` and ``metadata.json`` under ``output_dir``."""
    directory = Path(output_dir)
    graph_errors = _write_graph(data, directory / "graph.pt")
    meta_errors = _write_metadata(metadata, directory / "metadata.json")
    return graph_errors + meta_errors


def load_pyg_dataset(dataset_dir: str | Path) -> tuple[Data | None, dict[str, int] | None, tuple[str, ...]]:
    """Load ``graph.pt`` and ``metadata.json`` from ``dataset_dir``."""
    directory = Path(dataset_dir)
    graph_path = directory / "graph.pt"
    meta_path = directory / "metadata.json"
    if not graph_path.is_file():
        return None, None, (f"Missing PyG graph: {graph_path}",)
    if not meta_path.is_file():
        return None, None, (f"Missing metadata: {meta_path}",)
    try:
        loaded = torch.load(graph_path, weights_only=False)
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        message = f"Cannot load PyG graph {graph_path}: {exc}"
        logger.error(message)
        return None, None, (message,)
    if not isinstance(loaded, Data):
        return None, None, (f"{graph_path} is not a torch_geometric.data.Data object",)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"Cannot read metadata {meta_path}: {exc}"
        logger.error(message)
        return None, None, (message,)
    if not isinstance(metadata, dict):
        return None, None, (f"{meta_path} is not a JSON object",)
    return loaded, metadata, ()


def summarize_pyg_data(data: Data, metadata: dict[str, int]) -> tuple[str, tuple[str, ...]]:
    """Return the summary text, or errors when ``graph.pt`` disagrees with metadata."""
    errors = _metadata_errors(data, metadata)
    if errors:
        return "", tuple(errors)
    text = (
        f"Graph nodes: {data.num_nodes}\n"
        f"Graph edges: {data.edge_index.size(1)}\n"
        f"Feature shape: {tuple(data.x.shape)}\n"
        f"Label shape: {tuple(data.y.shape)}\n"
        f"Positive slack: {int(data.positive_slack_mask.sum().item())}\n"
        f"Negative slack: {int(data.negative_slack_mask.sum().item())}\n"
        "graph.pt: loaded\n"
    )
    return text, ()


def build_main(argv: list[str] | None = None) -> int:
    """Build ``data/gcd/pyg/graph.pt`` and ``metadata.json``."""
    parser = argparse.ArgumentParser(description="Convert the netlist graph to a PyTorch Geometric Data object.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=data_dir() / "gcd" / "graphs" / "netlist_graph.pkl",
        help="NetworkX graph (default: data/gcd/graphs/netlist_graph.pkl)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "path_dataset.json",
        help="Path slack records (default: data/gcd/dataset/path_dataset.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir() / "gcd" / "pyg",
        help="Directory for graph.pt and metadata.json (default: data/gcd/pyg)",
    )
    args = parser.parse_args(argv)
    if not args.graph.is_file():
        print(f"Missing graph: {args.graph}", file=sys.stderr)
        return 1
    if not args.labels.is_file():
        print(f"Missing path dataset: {args.labels}", file=sys.stderr)
        return 1
    graph, graph_errors = load_networkx_graph(args.graph)
    paths, path_errors = load_path_dataset(args.labels)
    errors = graph_errors + path_errors
    if errors or graph is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    data, metadata, build_errors = build_pyg_data(graph, paths)
    if build_errors or data is None or metadata is None:
        for message in build_errors:
            print(message, file=sys.stderr)
        return 1
    write_errors = save_pyg_dataset(data, metadata, args.output_dir)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1
    print(f"Wrote {args.output_dir / 'graph.pt'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")
    return 0


def summary_main(argv: list[str] | None = None) -> int:
    """Load ``graph.pt`` and print its shapes."""
    parser = argparse.ArgumentParser(description="Summarize the PyTorch Geometric graph dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=data_dir() / "gcd" / "pyg",
        help="Directory containing graph.pt and metadata.json (default: data/gcd/pyg)",
    )
    args = parser.parse_args(argv)
    data, metadata, errors = load_pyg_dataset(args.dataset_dir)
    if errors or data is None or metadata is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    text, summary_errors = summarize_pyg_data(data, metadata)
    if summary_errors:
        for message in summary_errors:
            print(message, file=sys.stderr)
        return 1
    print(text, end="")
    return 0


def _parse_path_record(record: object, index: int) -> tuple[dict[str, object] | None, str | None]:
    if not isinstance(record, dict):
        return None, f"path {index} is not an object"
    if "endpoint" not in record or "slack" not in record:
        return None, f"path {index} is missing endpoint or slack"
    try:
        slack = float(record["slack"])
    except (TypeError, ValueError):
        return None, f"path {index} has non-numeric slack {record.get('slack')!r}"
    if slack != slack:
        return None, f"path {index} has non-numeric slack"
    parsed = dict(record)
    parsed["endpoint"] = str(record["endpoint"])
    parsed["slack"] = slack
    return parsed, None


def _metadata_errors(data: Data, metadata: dict[str, int]) -> list[str]:
    required = (
        "node_count",
        "edge_count",
        "feature_dimension",
        "label_count",
        "register_count",
        "combinational_count",
        "positive_slack_count",
        "negative_slack_count",
    )
    errors: list[str] = []
    for key in required:
        if key not in metadata:
            errors.append(f"metadata.json is missing {key}")
    if errors:
        return errors
    if data.x is None or data.edge_index is None or data.y is None:
        return ["graph.pt is missing x, edge_index, or y"]
    expected = {
        "node_count": int(data.num_nodes),
        "edge_count": int(data.edge_index.size(1)),
        "feature_dimension": int(data.x.size(1)),
        "label_count": int(data.label_mask.sum().item()),
        "register_count": int(data.x[:, _FEATURE_INDEX["register"]].sum().item()),
        "combinational_count": int(data.x[:, _FEATURE_INDEX["combinational"]].sum().item()),
        "positive_slack_count": int(data.positive_slack_mask.sum().item()),
        "negative_slack_count": int(data.negative_slack_mask.sum().item()),
    }
    for key, value in expected.items():
        if int(metadata[key]) != value:
            errors.append(f"metadata {key} is {metadata[key]} but graph.pt has {value}")
    if tuple(data.x.shape) != (expected["node_count"], 4):
        errors.append(f"Feature shape is {tuple(data.x.shape)}")
    if tuple(data.edge_index.shape) != (2, expected["edge_count"]):
        errors.append(f"edge_index shape is {tuple(data.edge_index.shape)}")
    if tuple(data.y.shape) != (expected["node_count"],):
        errors.append(f"Label shape is {tuple(data.y.shape)}")
    return errors


def _write_graph(data: Data, destination: Path) -> tuple[str, ...]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, temporary)
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()


def _write_metadata(metadata: dict[str, int], destination: Path) -> tuple[str, ...]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()
