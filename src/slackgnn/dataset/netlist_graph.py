"""Build a directed signal-flow graph from a post-route Verilog netlist.

Nodes are primary inputs, primary outputs, flip-flops, and combinational
cells. An edge runs from the driver of a net to each cell or output that
reads it. Hierarchical modules are flattened, so a gate inside an adder
is its own node. This module does not build a tensor or a model.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx

from slackgnn.dataset.verilog import (
    Connection,
    Module,
    VerilogError,
    expand_expr,
    parse_netlist,
)
from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

NODE_TYPES = ("input", "output", "register", "combinational")
_OUTPUT_PINS = frozenset(
    {"X", "Y", "Q", "QN", "Q_N", "HI", "LO", "Z", "ZN", "SUM", "COUT", "Do", "DO"}
)
_POWER_PINS = frozenset(
    {"VGND", "VPWR", "VNB", "VPB", "VDD", "VSS", "VCCD1", "VSSD1", "VPWR_PAD", "VGND_PAD"}
)
_SEQUENTIAL_PREFIXES = (
    "dfxtp",
    "dfrtp",
    "dfstp",
    "dfbbp",
    "dfxbp",
    "edfxtp",
    "edfxbp",
    "dlxtp",
    "dlrtp",
    "dlxtn",
    "sdfxtp",
    "sdfbbp",
    "sdfstp",
)


_AHB_MASTER_OUTPUTS = frozenset({"HADDR", "HTRANS", "HSIZE", "HWRITE", "HWDATA"})


@dataclass(frozen=True)
class _PinUse:
    node: str
    pin: str
    direction: str
    net: str
    cell: str


def cell_node_type(cell: str) -> str:
    """Classify a leaf cell as ``register`` or ``combinational``."""
    leaf = cell.split("__")[-1].lower()
    if leaf.startswith(_SEQUENTIAL_PREFIXES):
        return "register"
    return "combinational"


def extract_netlist_graph(text: str, top: str = "gcd") -> tuple[nx.DiGraph | None, tuple[str, ...]]:
    """Flatten ``top`` and return its signal-flow graph."""
    try:
        modules = parse_netlist(text)
    except VerilogError as exc:
        return None, (str(exc),)
    if top not in modules:
        selected, error = _resolve_top(modules, top)
        if error or selected is None:
            return None, (error or f"Top module {top!r} is not in the netlist.",)
        top = selected
    try:
        graph = _build_graph(modules, top)
    except VerilogError as exc:
        return None, (str(exc),)
    logger.info(
        "Extracted %s: %d nodes, %d edges",
        top,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, ()


def _resolve_top(modules: dict[str, Module], top: str) -> tuple[str | None, str | None]:
    """Pick the finished design when the directory name is not the module name."""
    if len(modules) == 1:
        found = next(iter(modules))
        logger.info("Top module %r is absent. Using the only module %r.", top, found)
        return found, None
    instantiated = {instance.cell for module in modules.values() for instance in module.instances}
    roots = sorted(name for name in modules if name not in instantiated)
    if len(roots) == 1:
        logger.info("Top module %r is absent. Using uninstantiated module %r.", top, roots[0])
        return roots[0], None
    known = ", ".join(sorted(modules))
    return None, f"Top module {top!r} is not in the netlist. Found: {known}."


def extract_netlist_graph_file(
    path: str | Path,
    top: str = "gcd",
) -> tuple[nx.DiGraph | None, tuple[str, ...]]:
    """Read a netlist file and extract its signal-flow graph."""
    netlist = Path(path)
    try:
        text = netlist.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Cannot read netlist {netlist}: {exc}"
        logger.error(message)
        return None, (message,)
    except UnicodeError as exc:
        message = f"Netlist {netlist} is not UTF-8 text: {exc}"
        logger.error(message)
        return None, (message,)
    graph, errors = extract_netlist_graph(text.lstrip("\ufeff"), top=top)
    if errors:
        for message in errors:
            logger.error("%s: %s", netlist, message)
    return graph, errors


def write_netlist_graph(graph: nx.DiGraph, output: str | Path) -> tuple[str, ...]:
    """Pickle ``graph`` to ``output``. Return error messages, empty on success."""
    destination = Path(output)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()


def main(argv: list[str] | None = None) -> int:
    """Extract the GCD netlist graph and pickle it."""
    parser = argparse.ArgumentParser(
        description="Convert a post-route Verilog netlist into a NetworkX signal-flow graph."
    )
    parser.add_argument(
        "--netlist",
        type=Path,
        default=data_dir() / "gcd" / "netlist" / "6_final.v",
        help="Gate-level Verilog netlist (default: data/gcd/netlist/6_final.v)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir() / "gcd" / "graphs" / "netlist_graph.pkl",
        help="Pickled NetworkX DiGraph (default: data/gcd/graphs/netlist_graph.pkl)",
    )
    parser.add_argument("--top", default="gcd", help="Top module name (default: gcd)")
    args = parser.parse_args(argv)

    if not args.netlist.is_file():
        print(f"Missing netlist: {args.netlist}", file=sys.stderr)
        return 1

    graph, errors = extract_netlist_graph_file(args.netlist, top=args.top)
    if errors or graph is None:
        for message in errors:
            print(message, file=sys.stderr)
        return 1

    write_errors = write_netlist_graph(graph, args.output)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1

    counts = {kind: 0 for kind in NODE_TYPES}
    for _, attributes in graph.nodes(data=True):
        counts[attributes["node_type"]] += 1
    print(
        f"Wrote {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {args.output}"
    )
    print(
        "inputs: {input}, outputs: {output}, registers: {register}, combinational: {combinational}".format(
            **counts
        )
    )
    return 0


def _build_graph(modules: dict[str, Module], top: str) -> nx.DiGraph:
    module = modules[top]
    port_map = {bit: bit for port in module.ports.values() for bit in port.bits}
    nodes, pins = _flatten(module, modules, prefix="", port_map=port_map)
    input_nets = {bit for port in module.ports.values() if port.direction == "input" for bit in port.bits}
    pins = _resolve_macro_pin_directions(pins, input_nets)
    graph = nx.DiGraph()
    for name, node_type in nodes:
        graph.add_node(name, node_name=name, node_type=node_type)
    _add_signal_edges(graph, pins)
    return graph


def _flatten(
    module: Module,
    modules: dict[str, Module],
    prefix: str,
    port_map: dict[str, str],
) -> tuple[list[tuple[str, str]], list[_PinUse]]:
    nodes: list[tuple[str, str]] = []
    pins: list[_PinUse] = []
    if prefix == "":
        for port in module.ports.values():
            if port.direction not in {"input", "output"}:
                raise VerilogError(f"Port {port.name} in {module.name} has no direction")
            for bit in port.bits:
                nodes.append((bit, port.direction))

    seen_nodes = {name for name, _ in nodes}
    for instance in module.instances:
        node_name = f"{prefix}/{instance.name}" if prefix else instance.name
        if instance.cell in modules:
            child_nodes, child_pins = _flatten_child(instance, modules, prefix, port_map, module)
            for name, _ in child_nodes:
                if name in seen_nodes:
                    raise VerilogError(f"Flattened node {name} is declared more than once")
                seen_nodes.add(name)
            nodes.extend(child_nodes)
            pins.extend(child_pins)
            continue
        if node_name in seen_nodes:
            raise VerilogError(f"Instance {node_name} collides with an existing node")
        seen_nodes.add(node_name)
        nodes.append((node_name, cell_node_type(instance.cell)))
        pins.extend(_leaf_pins(instance, module, node_name, prefix, port_map))
    return nodes, pins


def _flatten_child(
    instance: Instance,
    modules: dict[str, Module],
    prefix: str,
    port_map: dict[str, str],
    parent: Module,
) -> tuple[list[tuple[str, str]], list[_PinUse]]:
    child = modules[instance.cell]
    child_prefix = f"{prefix}/{instance.name}" if prefix else instance.name
    connected: dict[str, Connection] = {}
    for pin in instance.pins:
        if pin.pin in connected:
            raise VerilogError(
                f"Instance {instance.name} connects pin {pin.pin} more than once (line {pin.line})"
            )
        connected[pin.pin] = pin
    child_map: dict[str, str] = {}
    for port in child.ports.values():
        connection = connected.get(port.name)
        if connection is None:
            raise VerilogError(
                f"Instance {instance.name} of {instance.cell} does not connect port {port.name} "
                f"(line {instance.line})"
            )
        bits = _connected_bits(connection, parent, instance)
        if len(bits) != len(port.bits):
            raise VerilogError(
                f"Instance {instance.name} port {port.name} connects {len(bits)} bits "
                f"to a {len(port.bits)}-bit port (line {connection.line})"
            )
        for local_bit, parent_bit in zip(port.bits, bits, strict=True):
            child_map[local_bit] = _resolve_net(parent_bit, port_map, prefix)
    return _flatten(child, modules, child_prefix, child_map)


def _leaf_pins(
    instance: Instance,
    module: Module,
    node_name: str,
    prefix: str,
    port_map: dict[str, str],
) -> list[_PinUse]:
    uses: list[_PinUse] = []
    for connection in instance.pins:
        if connection.pin in _POWER_PINS:
            continue
        direction = "output" if connection.pin in _OUTPUT_PINS else "input"
        bits = _connected_bits(connection, module, instance)
        if not bits:
            raise VerilogError(
                f"Instance {instance.name} pin {connection.pin} connects no bits "
                f"(line {connection.line})"
            )
        for index, bit in enumerate(bits):
            pin = connection.pin if len(bits) == 1 else f"{connection.pin}[{index}]"
            uses.append(
                _PinUse(
                    node=node_name,
                    pin=pin,
                    direction=direction,
                    net=_resolve_net(bit, port_map, prefix),
                    cell=instance.cell,
                )
            )
    return uses


def _is_macro(cell: str) -> bool:
    """True for a leaf whose pin directions are not fixed by the standard-cell library."""
    return not cell.startswith("sky130_")


def _pin_base(pin: str) -> str:
    return pin.split("[", 1)[0]


def _resolve_macro_pin_directions(pins: list[_PinUse], input_nets: set[str]) -> list[_PinUse]:
    """Give otherwise undriven macro nets a single driver.

    Standard-cell outputs and primary inputs stay as they are. A macro pin
    becomes the driver when it is the only macro pin on an undriven net.
    Shared AHB master pins are driven by ``ibex_wrapper``. Shared IRQ pins
    are driven by the peripheral rather than the core.
    """
    driven = set(input_nets)
    driven.update(use.net for use in pins if use.direction == "output")
    by_net: dict[str, list[_PinUse]] = {}
    for use in pins:
        if _is_macro(use.cell) and use.direction != "output":
            by_net.setdefault(use.net, []).append(use)
    chosen: set[tuple[str, str]] = set()
    for net, group in by_net.items():
        if net in driven:
            continue
        driver = _choose_macro_driver(group)
        if driver is not None:
            chosen.add((driver.node, driver.pin))
            driven.add(net)
    if not chosen:
        return pins
    return [
        replace(use, direction="output") if (use.node, use.pin) in chosen else use
        for use in pins
    ]


def _choose_macro_driver(group: list[_PinUse]) -> _PinUse | None:
    if len(group) == 1:
        return group[0]
    masters = [
        use
        for use in group
        if use.cell == "ibex_wrapper" and _pin_base(use.pin) in _AHB_MASTER_OUTPUTS
    ]
    if len(masters) == 1:
        return masters[0]
    irq_sources = [use for use in group if _pin_base(use.pin) == "IRQ" and use.cell != "ibex_wrapper"]
    if len(irq_sources) == 1:
        return irq_sources[0]
    return None


def _connected_bits(connection: Connection, module: Module, instance: Instance) -> tuple[str, ...]:
    try:
        return expand_expr(connection.expr, module)
    except VerilogError as exc:
        raise VerilogError(f"Instance {instance.name} pin {connection.pin}: {exc}") from exc


def _resolve_net(bit: str, port_map: dict[str, str], prefix: str) -> str:
    if bit in port_map:
        return port_map[bit]
    if prefix:
        return f"{prefix}/{bit}"
    return bit


def _add_signal_edges(graph: nx.DiGraph, pins: list[_PinUse]) -> None:
    drivers: dict[str, str] = {}
    for name, attributes in graph.nodes(data=True):
        if attributes["node_type"] == "input":
            drivers[name] = name

    loads: list[tuple[str, str]] = []
    for use in pins:
        if use.direction == "output":
            previous = drivers.get(use.net)
            if previous is not None and previous != use.node:
                raise VerilogError(f"Net {use.net} has multiple drivers: {previous} and {use.node}")
            drivers[use.net] = use.node
        else:
            loads.append((use.net, use.node))

    for name, attributes in graph.nodes(data=True):
        if attributes["node_type"] == "output":
            loads.append((name, name))

    missing: list[str] = []
    for net, load in loads:
        driver = drivers.get(net)
        if driver is None:
            missing.append(f"Net {net} drives {load} but has no driver")
            continue
        if driver != load:
            graph.add_edge(driver, load)
    if missing:
        preview = missing[:8]
        extra = len(missing) - len(preview)
        if extra > 0:
            preview.append(f"{extra} more undriven nets")
        raise VerilogError("; ".join(preview))
