"""Tests for structural netlist parsing and signal-flow graph extraction."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from slackgnn.dataset.netlist_graph import (
    cell_node_type,
    extract_netlist_graph,
    extract_netlist_graph_file,
    write_netlist_graph,
)
from slackgnn.dataset.verilog import VerilogError, parse_netlist
from slackgnn.paths import data_dir

_DATAFLOW = """
module top (a, b, clk, y);
  input a;
  input b;
  input clk;
  output y;
  wire n1;
  wire n2;
  sky130_fd_sc_hd__and2_1 and1 (.A(a), .B(b), .X(n1));
  sky130_fd_sc_hd__dfxtp_1 reg5 (.D(n1), .CLK(clk), .Q(n2));
  sky130_fd_sc_hd__buf_1 buf1 (.A(n2), .X(y));
endmodule
"""

_HIERARCHY = r"""
module top (a, b, y);
  input a;
  input b;
  output y;
  wire sum;
  \adder_block  u (.A({a, b}), .Y(sum));
  sky130_fd_sc_hd__buf_2 outbuf (.A(sum), .X(y));
endmodule
module \adder_block  (A, Y);
  input [1:0] A;
  output Y;
  sky130_fd_sc_hd__and2_1 g (.A(A[1]), .B(A[0]), .X(Y));
endmodule
"""


def test_cell_node_type_separates_flip_flops_from_delay_cells() -> None:
    assert cell_node_type("sky130_fd_sc_hd__dfxtp_4") == "register"
    assert cell_node_type("sky130_fd_sc_hd__edfxtp_1") == "register"
    assert cell_node_type("sky130_fd_sc_hd__dlygate4sd2_1") == "combinational"
    assert cell_node_type("sky130_fd_sc_hd__and2_1") == "combinational"


def test_signal_flow_edges_follow_drivers_to_loads() -> None:
    graph, errors = extract_netlist_graph(_DATAFLOW, top="top")

    assert errors == ()
    assert isinstance(graph, nx.DiGraph)
    assert graph.nodes["a"]["node_type"] == "input"
    assert graph.nodes["and1"]["node_type"] == "combinational"
    assert graph.nodes["reg5"]["node_type"] == "register"
    assert graph.nodes["y"]["node_type"] == "output"
    assert graph.nodes["and1"]["node_name"] == "and1"
    assert set(graph.edges) == {
        ("a", "and1"),
        ("b", "and1"),
        ("and1", "reg5"),
        ("clk", "reg5"),
        ("reg5", "buf1"),
        ("buf1", "y"),
    }


def test_hierarchy_is_flattened_and_bus_concatenation_is_bit_blasted() -> None:
    graph, errors = extract_netlist_graph(_HIERARCHY, top="top")

    assert errors == ()
    assert graph is not None
    assert "u" not in graph
    assert graph.nodes["u/g"]["node_type"] == "combinational"
    assert set(graph.edges) == {
        ("a", "u/g"),
        ("b", "u/g"),
        ("u/g", "outbuf"),
        ("outbuf", "y"),
    }


def test_half_adder_sum_and_cout_drive_their_nets() -> None:
    text = """
    module top (a, b, y);
      input a;
      input b;
      output y;
      wire c;
      sky130_fd_sc_hd__ha_1 ha (.A(a), .B(b), .COUT(c), .SUM(y));
    endmodule
    """
    graph, errors = extract_netlist_graph(text, top="top")
    assert errors == ()
    assert graph is not None
    assert graph.has_edge("a", "ha")
    assert graph.has_edge("b", "ha")
    assert graph.has_edge("ha", "y")


def test_single_module_netlist_uses_that_module_when_the_requested_top_is_absent() -> None:
    text = """
    module aes_cipher_top (a, y);
      input a;
      output y;
      sky130_fd_sc_hd__buf_1 buf1 (.A(a), .X(y));
    endmodule
    """
    graph, errors = extract_netlist_graph(text, top="aes")
    assert errors == ()
    assert graph is not None
    assert graph.has_edge("buf1", "y")


def test_hierarchy_root_is_used_when_the_requested_top_is_absent() -> None:
    graph, errors = extract_netlist_graph(_HIERARCHY, top="ibex")
    assert errors == ()
    assert graph is not None
    assert graph.has_edge("a", "u/g")


def test_black_box_bus_connects_each_bit() -> None:
    text = """
    module top (a, b, y0, y1);
      input a;
      input b;
      output y0;
      output y1;
      DFFRAM_4K ram (.A({a, b}), .Do({y0, y1}));
    endmodule
    """
    graph, errors = extract_netlist_graph(text, top="top")
    assert errors == ()
    assert graph is not None
    assert graph.has_edge("a", "ram")
    assert graph.has_edge("b", "ram")
    assert graph.has_edge("ram", "y0")
    assert graph.has_edge("ram", "y1")


def test_shared_ahb_master_pins_are_driven_by_the_core() -> None:
    text = """
    module top (clk, y, irq_out);
      input clk;
      output y;
      output irq_out;
      ibex_wrapper core (.HCLK(clk), .HADDR(addr), .IRQ(irq));
      apb_sys_0 periph (.HCLK(clk), .HADDR(addr), .IRQ(irq));
      sky130_fd_sc_hd__buf_1 addr_buf (.A(addr), .X(y));
      sky130_fd_sc_hd__buf_1 irq_buf (.A(irq), .X(irq_out));
    endmodule
    """
    graph, errors = extract_netlist_graph(text, top="top")
    assert errors == ()
    assert graph is not None
    assert graph.has_edge("core", "addr_buf")
    assert graph.has_edge("core", "periph")
    assert graph.has_edge("periph", "irq_buf")
    assert graph.has_edge("periph", "core")
    assert not graph.has_edge("periph", "addr_buf")


def test_width_mismatch_is_reported() -> None:
    text = """
    module top (a, y);
      input a;
      output y;
      sub u (.A(a), .Y(y));
    endmodule
    module sub (A, Y);
      input [1:0] A;
      output Y;
      sky130_fd_sc_hd__buf_1 g (.A(A[0]), .X(Y));
    endmodule
    """
    graph, errors = extract_netlist_graph(text, top="top")
    assert graph is None
    assert any("connects 1 bits" in message for message in errors)


def test_escaped_module_name_parses() -> None:
    modules = parse_netlist(_HIERARCHY)
    assert set(modules) == {"top", "adder_block"}


def test_parser_rejects_an_empty_netlist() -> None:
    try:
        parse_netlist("   ")
    except VerilogError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty netlist was accepted")


def test_missing_netlist_file(tmp_path: Path) -> None:
    graph, errors = extract_netlist_graph_file(tmp_path / "missing.v")
    assert graph is None
    assert "Cannot read netlist" in errors[0]


def test_gcd_netlist_graph_matches_the_critical_path(tmp_path: Path) -> None:
    netlist = data_dir() / "gcd" / "netlist" / "6_final.v"
    graph, errors = extract_netlist_graph_file(netlist, top="gcd")

    assert errors == ()
    assert graph is not None
    assert isinstance(graph, nx.DiGraph)
    kinds = {attributes["node_type"] for _, attributes in graph.nodes(data=True)}
    assert kinds <= set(("input", "output", "register", "combinational"))
    for name, attributes in graph.nodes(data=True):
        assert attributes["node_name"] == name
        assert attributes["node_type"] in {"input", "output", "register", "combinational"}

    assert graph.nodes["req_msg[0]"]["node_type"] == "input"
    assert graph.nodes["resp_msg[15]"]["node_type"] == "output"
    assert graph.nodes["dpath.b_reg.out[3]$_DFFE_PP_"]["node_type"] == "register"
    assert graph.nodes["dpath.a_reg.out[11]$_DFFE_PP_"]["node_type"] == "register"
    assert graph.nodes["_119_/_150_"]["node_type"] == "combinational"
    assert graph.has_edge("req_msg[0]", "input3")
    assert nx.has_path(
        graph,
        "dpath.b_reg.out[3]$_DFFE_PP_",
        "dpath.a_reg.out[11]$_DFFE_PP_",
    )
    registers = [
        name
        for name, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "register"
    ]
    assert len(registers) == 35

    destination = tmp_path / "netlist_graph.pkl"
    assert write_netlist_graph(graph, destination) == ()
    with destination.open("rb") as handle:
        loaded = pickle.load(handle)
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert loaded.number_of_edges() == graph.number_of_edges()
