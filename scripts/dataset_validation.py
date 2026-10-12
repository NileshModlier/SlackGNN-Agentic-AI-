"""Check that timing labels sit on reachable endpoints of the netlist graph.

From the repository root, after the labeled dataset has been built:

    python scripts/dataset_validation.py

Writes ``data/gcd/dataset/validation_report.json``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.graph_dataset import validate_main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(validate_main())
