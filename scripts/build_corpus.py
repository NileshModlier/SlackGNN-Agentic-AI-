"""Build the multi-design PyG corpus.

From the repository root:

    python scripts/build_corpus.py

Reads design names from ``data/designs.json``. Writes
``data/corpus/<design>/graph.pt``, ``metadata.json``, and
``data/corpus/corpus_metadata.json``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.corpus import build_main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(build_main())
