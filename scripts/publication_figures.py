"""Write the M9.8 publication figures.

From the repository root:

    python scripts/publication_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.publication import load_publication_inputs, write_publication_figures


def main() -> int:
    designs, generalization = load_publication_inputs()
    written = write_publication_figures(designs, generalization)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
