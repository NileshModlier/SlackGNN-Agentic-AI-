"""Print the multi-design corpus overview.

From the repository root:

    python scripts/corpus_summary.py

Reads ``data/corpus/corpus_metadata.json`` and checks that each published
design still has ``graph.pt`` and ``metadata.json``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.corpus import summary_main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(summary_main())
