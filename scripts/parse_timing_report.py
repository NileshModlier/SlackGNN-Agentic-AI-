"""Extract timing-path labels from the GCD finish report.

From the repository root:

    python scripts/parse_timing_report.py

The default input is ``data/gcd/reports/6_finish.rpt``. The default output
is ``data/gcd/parsed/path_labels.csv``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.timing_report import main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
