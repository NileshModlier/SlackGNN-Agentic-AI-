"""Join OpenROAD path slack onto the gate-level signal-flow graph.

Slack is stored on the timing endpoint, which is the node where the
report defines the check. The module checks that each startpoint can
reach that endpoint in the directed graph. It does not train a model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import networkx as nx

from slackgnn.dataset.summary import load_path_labels
from slackgnn.dataset.timing_report import TimingPath
from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

LABELED_PATH_FIELDS = (
    "startpoint",
    "endpoint",
    "path_group",
    "path_type",
    "arrival_time",
    "required_time",
    "slack",
    "startpoint_node_type",
    "endpoint_node_type",
    "path_exists",
)


@dataclass(frozen=True)
class LabeledPath:
    """One timing check placed on the signal-flow graph."""

    startpoint: str
    endpoint: str
    path_group: str
    path_type: str
    arrival_time: str
    required_time: str
    slack: str
    startpoint_node_type: str
    endpoint_node_type: str
    path_exists: bool

    def as_row(self) -> dict[str, str]:
        return {
            "startpoint": self.startpoint,
            "endpoint": self.endpoint,
            "path_group": self.path_group,
            "path_type": self.path_type,
            "arrival_time": self.arrival_time,
            "required_time": self.required_time,
            "slack": self.slack,
            "startpoint_node_type": self.startpoint_node_type,
            "endpoint_node_type": self.endpoint_node_type,
            "path_exists": "true" if self.path_exists else "false",
        }

    def label_record(self) -> dict[str, str]:
        """Endpoint attribute for this check. The endpoint is the node key."""
        return {
            "startpoint": self.startpoint,
            "path_group": self.path_group,
            "path_type": self.path_type,
            "arrival_time": self.arrival_time,
            "required_time": self.required_time,
            "slack": self.slack,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Result of checking a labeled graph against its source files."""

    passed: bool
    graph_nodes: int
    graph_edges: int
    labeled_paths: int
    labeled_nodes: int
    paths_connected: int
    timing_violations: int
    timing_met_paths: int
    errors: tuple[str, ...]

    def as_json(self) -> str:
        payload = {
            "passed": self.passed,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "labeled_paths": self.labeled_paths,
            "labeled_nodes": self.labeled_nodes,
            "paths_connected": self.paths_connected,
            "timing_violations": self.timing_violations,
            "timing_met_paths": self.timing_met_paths,
            "errors": list(self.errors),
        }
        return json.dumps(payload, indent=2) + "\n"

    def as_text(self) -> str:
        lines = [
            f"passed: {str(self.passed).lower()}",
            f"graph nodes: {self.graph_nodes}",
            f"graph edges: {self.graph_edges}",
            f"labeled paths: {self.labeled_paths}",
            f"labeled nodes: {self.labeled_nodes}",
            f"paths connected: {self.paths_connected}",
            f"timing violations: {self.timing_violations}",
            f"timing-met paths: {self.timing_met_paths}",
        ]
        return "\n".join(lines) + "\n"


def load_di_graph(path: str | Path) -> tuple[nx.DiGraph | None, tuple[str, ...]]:
    """Load a pickled ``networkx.DiGraph``."""
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


def join_timing_labels(
    graph: nx.DiGraph,
    paths: tuple[TimingPath, ...] | list[TimingPath],
) -> tuple[nx.DiGraph | None, tuple[LabeledPath, ...], tuple[str, ...]]:
    """Copy ``graph`` and attach each path slack to its endpoint node.

    The source graph is left unchanged. A missing endpoint, a missing
    startpoint, or a startpoint that cannot reach the endpoint is an
    error, and no labeled graph is returned.
    """
    if not paths:
        return None, (), ("No timing paths to attach",)

    errors: list[str] = []
    labeled_paths: list[LabeledPath] = []
    for path in paths:
        joined, path_errors = _join_path(graph, path)
        errors.extend(path_errors)
        if joined is not None:
            labeled_paths.append(joined)
    if errors:
        return None, (), tuple(errors)

    labeled = copy.deepcopy(graph)
    for node, attributes in labeled.nodes(data=True):
        attributes["has_timing_label"] = False
        attributes["timing_labels"] = []
        if "node_name" not in attributes or "node_type" not in attributes:
            return None, (), (f"Graph node {node} is missing node_name or node_type",)

    by_endpoint: dict[str, list[dict[str, str]]] = {}
    for joined in labeled_paths:
        by_endpoint.setdefault(joined.endpoint, []).append(joined.label_record())
    for endpoint, records in by_endpoint.items():
        labeled.nodes[endpoint]["has_timing_label"] = True
        labeled.nodes[endpoint]["timing_labels"] = records

    logger.info(
        "Attached %d timing paths to %d endpoint nodes",
        len(labeled_paths),
        len(by_endpoint),
    )
    return labeled, tuple(labeled_paths), ()


def validate_labeled_dataset(
    source: nx.DiGraph,
    paths: tuple[TimingPath, ...] | list[TimingPath],
    labeled: nx.DiGraph,
    labeled_paths: tuple[LabeledPath, ...] | list[LabeledPath],
) -> ValidationReport:
    """Check that ``labeled`` is ``source`` plus the endpoint slack records."""
    errors: list[str] = []
    if set(source.nodes) != set(labeled.nodes):
        errors.append("Labeled graph nodes do not match the source graph")
    if set(source.edges) != set(labeled.edges):
        errors.append("Labeled graph edges do not match the source graph")

    for node in source.nodes:
        if node not in labeled:
            continue
        for field in ("node_name", "node_type"):
            if source.nodes[node].get(field) != labeled.nodes[node].get(field):
                errors.append(f"Node {node} changed {field}")
        if labeled.nodes[node].get("node_name") != node:
            errors.append(f"Node {node} node_name does not match its key")

    expected, join_errors = _expected_labels(source, paths)
    errors.extend(join_errors)
    if len(labeled_paths) != len(expected):
        errors.append(
            f"Labeled path table has {len(labeled_paths)} rows; the timing table has {len(expected)}"
        )
    for index, (found, wanted) in enumerate(zip(labeled_paths, expected, strict=False), start=1):
        if found != wanted:
            errors.append(f"Labeled path row {index} does not match the timing table")

    labeled_nodes = 0
    connected = 0
    violations = 0
    met = 0
    for node, attributes in labeled.nodes(data=True):
        records = attributes.get("timing_labels", [])
        has_label = bool(attributes.get("has_timing_label"))
        if has_label != bool(records):
            errors.append(f"Node {node} has_timing_label does not match timing_labels")
        if has_label:
            labeled_nodes += 1
        wanted = [
            path.label_record()
            for path in expected
            if path.endpoint == node and path.path_exists
        ]
        if list(records) != wanted:
            errors.append(f"Node {node} timing_labels do not match the timing table")

    for path in labeled_paths:
        if path.path_exists:
            connected += 1
        if path.slack.startswith("-"):
            violations += 1
        else:
            met += 1

    report = ValidationReport(
        passed=not errors,
        graph_nodes=labeled.number_of_nodes(),
        graph_edges=labeled.number_of_edges(),
        labeled_paths=len(labeled_paths),
        labeled_nodes=labeled_nodes,
        paths_connected=connected,
        timing_violations=violations,
        timing_met_paths=met,
        errors=tuple(errors),
    )
    if report.passed:
        logger.info("Labeled dataset passed validation")
    else:
        for message in report.errors:
            logger.error(message)
    return report


def write_labeled_graph(graph: nx.DiGraph, output: str | Path) -> tuple[str, ...]:
    """Pickle a labeled graph. Return error messages, empty on success."""
    return _write_bytes(Path(output), _pickle_bytes(graph))


def write_labeled_paths(paths: tuple[LabeledPath, ...] | list[LabeledPath], output: str | Path) -> tuple[str, ...]:
    """Write the joined path table. Return error messages, empty on success."""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LABELED_PATH_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(path.as_row() for path in paths)
    return _write_text(Path(output), buffer.getvalue())


def load_labeled_paths(path: str | Path) -> tuple[tuple[LabeledPath, ...], tuple[str, ...]]:
    """Read the joined path table."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Cannot read labeled paths {source}: {exc}"
        logger.error(message)
        return (), (message,)
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != LABELED_PATH_FIELDS:
        found = ",".join(reader.fieldnames or ())
        expected = ",".join(LABELED_PATH_FIELDS)
        return (), (f"Labeled paths {source} header is {found!r}; expected {expected!r}",)
    paths: list[LabeledPath] = []
    for index, row in enumerate(reader, start=2):
        parsed, error = _row_to_labeled_path(row, index)
        if error or parsed is None:
            return (), (f"{source} {error}",)
        paths.append(parsed)
    if not paths:
        return (), (f"Labeled paths {source} contain no paths",)
    return tuple(paths), ()


def write_validation_report(report: ValidationReport, output: str | Path) -> tuple[str, ...]:
    """Write the validation report as JSON."""
    return _write_text(Path(output), report.as_json())


def build_main(argv: list[str] | None = None) -> int:
    """Join path labels onto the netlist graph and write the dataset files."""
    parser = argparse.ArgumentParser(description="Attach OpenROAD path slack to the netlist graph.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=data_dir() / "gcd" / "graphs" / "netlist_graph.pkl",
        help="Signal-flow graph (default: data/gcd/graphs/netlist_graph.pkl)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=data_dir() / "gcd" / "parsed" / "path_labels.csv",
        help="Timing-path labels (default: data/gcd/parsed/path_labels.csv)",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "labeled_graph.pkl",
        help="Labeled graph (default: data/gcd/dataset/labeled_graph.pkl)",
    )
    parser.add_argument(
        "--paths-output",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "labeled_paths.csv",
        help="Joined path table (default: data/gcd/dataset/labeled_paths.csv)",
    )
    args = parser.parse_args(argv)
    return _build_from_files(args.graph, args.labels, args.graph_output, args.paths_output)


def validate_main(argv: list[str] | None = None) -> int:
    """Validate the labeled dataset and write ``validation_report.json``."""
    parser = argparse.ArgumentParser(description="Validate timing labels attached to the netlist graph.")
    parser.add_argument(
        "--graph",
        type=Path,
        default=data_dir() / "gcd" / "graphs" / "netlist_graph.pkl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=data_dir() / "gcd" / "parsed" / "path_labels.csv",
    )
    parser.add_argument(
        "--labeled-graph",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "labeled_graph.pkl",
    )
    parser.add_argument(
        "--labeled-paths",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "labeled_paths.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=data_dir() / "gcd" / "dataset" / "validation_report.json",
    )
    args = parser.parse_args(argv)
    return _validate_from_files(
        args.graph,
        args.labels,
        args.labeled_graph,
        args.labeled_paths,
        args.report,
    )


def _build_from_files(graph_path: Path, labels_path: Path, graph_output: Path, paths_output: Path) -> int:
    if not graph_path.is_file():
        print(f"Missing graph: {graph_path}", file=sys.stderr)
        return 1
    if not labels_path.is_file():
        print(f"Missing path labels: {labels_path}", file=sys.stderr)
        return 1
    graph, graph_errors = load_di_graph(graph_path)
    paths, path_errors = load_path_labels(labels_path)
    errors = graph_errors + path_errors
    if errors or graph is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    labeled, joined, join_errors = join_timing_labels(graph, paths)
    if join_errors or labeled is None:
        for message in join_errors:
            print(message, file=sys.stderr)
        return 1
    write_errors = write_labeled_graph(labeled, graph_output) + write_labeled_paths(joined, paths_output)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1
    endpoints = {path.endpoint for path in joined}
    print(f"Wrote {labeled.number_of_nodes()} nodes and {labeled.number_of_edges()} edges to {graph_output}")
    print(f"Attached {len(joined)} timing paths to {len(endpoints)} endpoint nodes in {paths_output}")
    return 0


def _validate_from_files(
    graph_path: Path,
    labels_path: Path,
    labeled_graph_path: Path,
    labeled_paths_path: Path,
    report_path: Path,
) -> int:
    missing = [
        (graph_path, "graph"),
        (labels_path, "path labels"),
        (labeled_graph_path, "labeled graph"),
        (labeled_paths_path, "labeled paths"),
    ]
    for path, role in missing:
        if not path.is_file():
            print(f"Missing {role}: {path}", file=sys.stderr)
            return 1
    source, source_errors = load_di_graph(graph_path)
    labeled, labeled_errors = load_di_graph(labeled_graph_path)
    timing_paths, timing_errors = load_path_labels(labels_path)
    joined_paths, joined_errors = load_labeled_paths(labeled_paths_path)
    errors = source_errors + labeled_errors + timing_errors + joined_errors
    if errors or source is None or labeled is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    report = validate_labeled_dataset(source, timing_paths, labeled, joined_paths)
    write_errors = write_validation_report(report, report_path)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1
    print(report.as_text(), end="")
    if not report.passed:
        for message in report.errors:
            print(message, file=sys.stderr)
        return 1
    return 0


def _join_path(graph: nx.DiGraph, path: TimingPath) -> tuple[LabeledPath | None, list[str]]:
    errors: list[str] = []
    if path.startpoint not in graph:
        errors.append(f"Label startpoint is not in the graph: {path.startpoint}")
    if path.endpoint not in graph:
        errors.append(f"Label endpoint is not in the graph: {path.endpoint}")
    if errors:
        return None, errors
    exists = nx.has_path(graph, path.startpoint, path.endpoint)
    if not exists:
        errors.append(
            f"No directed path from {path.startpoint} to {path.endpoint} "
            f"({path.path_type} slack {path.slack})"
        )
        return None, errors
    return (
        LabeledPath(
            startpoint=path.startpoint,
            endpoint=path.endpoint,
            path_group=path.path_group,
            path_type=path.path_type,
            arrival_time=format(path.arrival_time, "f"),
            required_time=format(path.required_time, "f"),
            slack=format(path.slack, "f"),
            startpoint_node_type=str(graph.nodes[path.startpoint]["node_type"]),
            endpoint_node_type=str(graph.nodes[path.endpoint]["node_type"]),
            path_exists=True,
        ),
        [],
    )


def _expected_labels(
    graph: nx.DiGraph,
    paths: tuple[TimingPath, ...] | list[TimingPath],
) -> tuple[tuple[LabeledPath, ...], list[str]]:
    expected: list[LabeledPath] = []
    errors: list[str] = []
    for path in paths:
        joined, path_errors = _join_path(graph, path)
        errors.extend(path_errors)
        if joined is not None:
            expected.append(joined)
    return tuple(expected), errors


def _row_to_labeled_path(row: dict[str, str | None], line_number: int) -> tuple[LabeledPath | None, str | None]:
    values: dict[str, str] = {}
    for field in LABELED_PATH_FIELDS:
        raw = row.get(field)
        if raw is None or raw.strip() == "":
            return None, f"row {line_number} is missing {field}"
        values[field] = raw.strip()
    flag = values["path_exists"].lower()
    if flag not in {"true", "false"}:
        return None, f"row {line_number} has path_exists {values['path_exists']!r}"
    return (
        LabeledPath(
            startpoint=values["startpoint"],
            endpoint=values["endpoint"],
            path_group=values["path_group"],
            path_type=values["path_type"],
            arrival_time=values["arrival_time"],
            required_time=values["required_time"],
            slack=values["slack"],
            startpoint_node_type=values["startpoint_node_type"],
            endpoint_node_type=values["endpoint_node_type"],
            path_exists=flag == "true",
        ),
        None,
    )


def _pickle_bytes(graph: nx.DiGraph) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    pickle.dump(graph, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.getvalue()


def _write_text(destination: Path, payload: str) -> tuple[str, ...]:
    return _write_bytes(destination, payload.encode("utf-8"))


def _write_bytes(destination: Path, payload: bytes) -> tuple[str, ...]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(payload)
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()
