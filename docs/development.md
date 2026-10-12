# Development

SlackGNN is a Python 3.11 research repository for timing-slack analysis on circuit graphs. This revision is the software scaffold: packaging, directory ownership, and tests.

## Environment

Create the environment with Python 3.11 and install from the checkout root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Unix, activate with `source .venv/bin/activate` and use `python3.11` to create the virtual environment.

`requirements.txt` and `pyproject.toml` list the same packages. Runtime dependencies are PyTorch, PyTorch Geometric, and NetworkX. pytest is the test extra declared in `pyproject.toml` and is included in `requirements.txt` so one install produces a working test environment.

PyTorch Geometric 2.8 supports PyTorch 2.9 through 2.12. Install a CUDA build from the PyTorch install selector before `requirements.txt` when the default PyPI wheel is the wrong device build. Accelerated PyG extensions are optional and are not part of this scaffold.

Confirm the interpreter:

```powershell
python scripts\check_python.py
python -m pytest
```

pytest is configured in `pyproject.toml` with `pythonpath = ["src"]`, so the suite runs from a checkout even before the editable install. After `pip install -e .`, imports resolve through the installed package.

## Directory ownership

| Path | Owns |
| --- | --- |
| `src/slackgnn/` | Importable library code. New modules land here and gain tests in `tests/`. |
| `tests/` | pytest modules. Tests import `slackgnn`; they do not modify `sys.path`. |
| `scripts/` | Thin command-line entry points that call the library. |
| `tcl/` | EDA tool scripts for extraction and reporting. |
| `docs/` | Design notes and contributor workflow. |
| `notebooks/` | Exploratory analysis. Checkpoints stay untracked. |
| `data/` | Design and timing inputs. `data/gcd/` is the curated GCD dataset. Other contents stay untracked. |
| `results/` | Local experiment outputs. Contents stay untracked. |

Resolve these locations through `slackgnn.paths` rather than paths relative to the shell's working directory.

## Dependency policy

Add a library only when a design note in `docs/` names the caller and the reason. Update `pyproject.toml` and `requirements.txt` in the same change. Keep Python on 3.11 (`requires-python` is `>=3.11,<3.12`).

## Data and results

`results/` is a local workspace. The git index keeps `results/.gitkeep` and ignores everything else there. `data/gcd/` is the checked-in OpenROAD GCD dataset (netlist, SDC, SPEF, and `.rpt` files). Every other path under `data/` stays untracked, including waveforms and extra flow dumps. How those GCD files are produced is in [dataset_structure.md](dataset_structure.md).

## Tcl

Scripts in `tcl/` drive EDA tools. They read and write files under `data/` and `results/`. They do not embed learning code, and Python does not shell out to an unbounded Tcl string assembled from unsanitized report text. Review Tcl with the same care as `src/`.

## Later revisions

`scripts/parse_timing_report.py` turns `data/gcd/reports/6_finish.rpt` into path labels. `scripts/extract_graph.py` turns `data/gcd/netlist/6_final.v` into a NetworkX signal-flow graph. `scripts/build_dataset.py` attaches those labels to graph endpoints. `scripts/build_pyg_dataset.py` writes one PyTorch Geometric `Data` object. Model training belongs in a later change, with tests and a short design note.
