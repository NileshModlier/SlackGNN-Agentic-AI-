"""Build the PyTorch Geometric graph for the GCD design.

From the repository root:

    python scripts/build_pyg_dataset.py

Reads the NetworkX graph and ``path_dataset.json``. Writes
``data/gcd/pyg/graph.pt`` and ``data/gcd/pyg/metadata.json``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.pyg_dataset import build_main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(build_main())
