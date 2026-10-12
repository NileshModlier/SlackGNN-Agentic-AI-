"""Train timing models.

From the repository root:

    python scripts/train.py
    python scripts/train.py --model gcn
    python scripts/train.py --model graphsage
    python scripts/train.py --model gin
    python scripts/train.py --model slackgnn
    python scripts/train.py --model slackgnn --task multitask
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.training.train import main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
