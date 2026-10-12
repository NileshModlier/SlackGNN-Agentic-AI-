# GCD dataset

## Dataset source

OpenROAD GCD benchmark.

The files come from a completed RTL-to-GDS run of the GCD design on the sky130hd platform, `base` flow configuration. OpenROAD wrote the finish artifacts on 9 October 2026.

| Role | OpenROAD path |
| --- | --- |
| Results | `/home/administrator/OpenROAD-flow-scripts/flow/results/sky130hd/gcd/base/` |
| Reports | `/home/administrator/OpenROAD-flow-scripts/flow/reports/sky130hd/gcd/base/` |

`scripts/import_gcd_dataset.py` copies the files below into this repository. On Windows it reads those Linux paths through the WSL share. Re-running the script overwrites the destination with the current sources.

```powershell
python scripts\import_gcd_dataset.py
```

A successful run prints `SUCCESS`. A missing netlist, constraints file, SPEF file, reports directory, or `.rpt` file prints a specific error and copies nothing.

## Artifacts

- **Netlist** — `data/gcd/netlist/6_final.v`. Gate-level structural Verilog for module `gcd`, mapped to `sky130_fd_sc_hd` cells. This is the circuit connectivity.
- **Constraints** — `data/gcd/sdc/6_final.sdc`. Clock `core_clock` at a 1.1000 ns period, a virtual clock, and 0.2200 ns input and output delays. These define the required times behind every slack number.
- **SPEF** — `data/gcd/spef/6_final.spef`. IEEE 1481 parasitics for the routed design, with time in nanoseconds, capacitance in picofarads, and resistance in ohms.
- **Timing reports** — `data/gcd/reports/*.rpt`. Stage reports from floorplan through finish. `6_finish.rpt` holds total negative slack, worst slack, and the path checks for the finished design. Images and logs from the reports directory are not copied.

What each file contributes to timing-slack prediction is documented in [docs/dataset_structure.md](docs/dataset_structure.md).

## Parsed labels

`scripts/parse_timing_report.py` reads `data/gcd/reports/6_finish.rpt` and writes `data/gcd/parsed/path_labels.csv` with one row per timing path: startpoint, endpoint, path group, path type, arrival time, required time, and slack. `scripts/dataset_summary.py` writes `data/gcd/parsed/dataset_summary.json`.

```powershell
python scripts\parse_timing_report.py
python scripts\dataset_summary.py
```

## Signal-flow graph

`scripts/extract_graph.py` flattens `data/gcd/netlist/6_final.v` into `data/gcd/graphs/netlist_graph.pkl`, a NetworkX directed graph. Nodes are primary inputs, primary outputs, flip-flops, and combinational cells. Each node stores `node_name` and `node_type`. An edge follows a net from its driver to each cell or output that reads it. The Kogge-Stone ALU is flattened, so its gates are nodes such as `_119_/_150_`.

```powershell
python scripts\extract_graph.py
```

## Labeled graph

`scripts/build_dataset.py` attaches each row of `path_labels.csv` to its endpoint in the signal-flow graph. Slack stays on the endpoint because that is where the timing check is defined. The build keeps a path only when the startpoint can reach that endpoint. It writes:

- `data/gcd/dataset/labeled_graph.pkl`
- `data/gcd/dataset/labeled_paths.csv`

`scripts/dataset_validation.py` checks the join and writes `data/gcd/dataset/validation_report.json`.

```powershell
python scripts\build_dataset.py
python scripts\dataset_validation.py
```

## PyTorch Geometric graph

`scripts/build_pyg_dataset.py` converts `netlist_graph.pkl` and `path_dataset.json` into one `torch_geometric.data.Data` object:

- `data/gcd/pyg/graph.pt`
- `data/gcd/pyg/metadata.json`

`x` is a one-hot encoding of `input`, `output`, `register`, and `combinational`. `edge_index` follows the directed nets. `y` holds endpoint slack and is separate from `x`. Unlabeled nodes are NaN. Positive and negative slack are tracked with masks. `scripts/pyg_dataset_summary.py` reloads `graph.pt` and prints the shapes.

```powershell
python scripts\build_pyg_dataset.py
python scripts\pyg_dataset_summary.py
```

## Multi-design corpus

`data/designs.json` lists the benchmarks. Adding a design is a new name in that file and a `data/<design>/` directory with `netlist/`, `reports/`, `graphs/`, `dataset/`, and `pyg/`. The corpus builder does not contain a per-design code path.

`scripts/build_corpus.py` publishes each valid `graph.pt` and `metadata.json` to `data/corpus/<design>/`. When those PyG files are absent and `graphs/netlist_graph.pkl` plus `dataset/path_dataset.json` are present, it builds the PyG graph first. A design that is missing or fails the node and edge checks is recorded under `failures` in `data/corpus/corpus_metadata.json` and is left out of the totals.

```powershell
python scripts\build_corpus.py
python scripts\corpus_summary.py
```

## Process one design

`scripts/process_design.py` takes a design name and runs timing parsing, graph extraction, dataset construction, and PyG conversion for `data/<design>/`. The design directory must already contain `netlist/6_final.v` and `reports/6_finish.rpt`. Output folders `parsed/`, `graphs/`, `dataset/`, and `pyg/` are created by the command. The Verilog top module defaults to the design name; pass `--top` when the module name differs.

A successful run adds the design to `data/designs.json` if it is not already listed and refreshes `data/corpus/corpus_metadata.json`. A failed run records the design under `failures` and leaves it out of the totals. `data/<design>/process_report.json` stores the stage results when the design directory exists.

```powershell
python scripts\process_design.py aes
```

The repository does not yet contain a graph neural network or training code.
