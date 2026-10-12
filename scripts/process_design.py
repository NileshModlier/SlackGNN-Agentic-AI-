"""Process one OpenROAD design into the corpus.

From the repository root:

    python scripts/process_design.py aes

Reads ``data/<design>/netlist/6_final.v`` and
``data/<design>/reports/6_finish.rpt``. Runs timing parsing, graph
extraction, dataset construction, and PyG conversion, then refreshes
``data/corpus/corpus_metadata.json``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.process_design import main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
