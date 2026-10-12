"""Write the M9 research report and release note from measured JSON.

From the repository root:

    python scripts/write_m9_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.documents import write_m9_documents


def main() -> int:
    report, readme = write_m9_documents()
    print(report)
    print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
