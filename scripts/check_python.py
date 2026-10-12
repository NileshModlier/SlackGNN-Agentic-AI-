"""Check that the current interpreter is Python 3.11.

Run from a checkout before installing the scientific stack:

    python scripts/check_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
