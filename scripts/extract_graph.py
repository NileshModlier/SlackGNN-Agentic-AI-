"""Extract a signal-flow graph from the GCD gate netlist.

From the repository root:

    python scripts/extract_graph.py

The default input is ``data/gcd/netlist/6_final.v``. The default output is
``data/gcd/graphs/netlist_graph.pkl``, a ``networkx.DiGraph``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.netlist_graph import main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
