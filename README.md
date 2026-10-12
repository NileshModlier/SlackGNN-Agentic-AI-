# SlackGNN

SlackGNN is a research repository for timing-slack analysis on circuit graphs. This initial revision is the Python 3.11 project scaffold: package layout, dependency pins, directory ownership, and tests.

The library lives in `src/slackgnn`. PyTorch, PyTorch Geometric, and NetworkX are declared so later graph and learning work installs against a known stack. This revision does not add models or graph neural network modules.

## Requirements

- Python 3.11
- PyTorch 2.9 through 2.12
- PyTorch Geometric 2.8
- NetworkX 3.4 or newer, below 4
- pytest 9

PyTorch Geometric 2.8 is the dependency that sets the PyTorch window. A CUDA wheel, when required, comes from the [PyTorch install selector](https://pytorch.org/get-started/locally/) before the requirements file below.

## Layout

```text
docs/         design notes and contributor workflow
data/         per-design inputs, data/designs.json, and data/corpus/
results/      local experiment outputs (untracked)
notebooks/    exploratory notebooks
scripts/      command-line entry points
src/slackgnn/ importable library
tests/        pytest suite
tcl/          EDA tool scripts
```

`data/gcd/` holds the OpenROAD GCD netlist, constraints, SPEF, and timing reports. `data/designs.json` lists the benchmarks, and `data/corpus/` holds one PyG graph per design that passes validation. Import and file meanings are in [README_dataset.md](README_dataset.md) and [docs/dataset_structure.md](docs/dataset_structure.md). Other design dumps, waveforms, and learned weights stay on the local disk.

## Setup

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts\check_python.py
python -m pytest
```

Unix activation is `source .venv/bin/activate`.

`python -m pytest` uses the `src` layout through the pytest configuration in `pyproject.toml`. Contributor workflow, Tcl boundaries, and the dependency policy are in [docs/development.md](docs/development.md).

## Current library surface

- `slackgnn.__version__` — package version, kept aligned with `pyproject.toml`
- `slackgnn.paths` — checkout directories (`data`, `results`, `tcl`, and the other top-level paths)
- `slackgnn.runtime` — Python 3.11 interpreter check used by `scripts/check_python.py`
