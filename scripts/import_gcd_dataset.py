"""Copy the OpenROAD GCD dataset into ``data/gcd``.

From the repository root:

    python scripts/import_gcd_dataset.py

The default sources are the sky130hd GCD ``base`` flow:

    /home/administrator/OpenROAD-flow-scripts/flow/results/sky130hd/gcd/base
    /home/administrator/OpenROAD-flow-scripts/flow/reports/sky130hd/gcd/base

On Windows those Linux paths are read through the WSL filesystem share.
The process prints ``SUCCESS`` when every required artifact has been copied.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.dataset.gcd_import import main


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
