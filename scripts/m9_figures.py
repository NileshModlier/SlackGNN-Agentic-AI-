"""Write the M9 figures from the JSON reports.

From the repository root:

    python scripts/m9_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.figures import write_m9_figures


def main() -> int:
    written = write_m9_figures()
    for path in written:
        print(path)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
