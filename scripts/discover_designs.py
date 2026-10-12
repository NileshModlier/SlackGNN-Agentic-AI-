"""Write the OpenROAD benchmark inventory.

From the repository root:

    python scripts/discover_designs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.expansion.discover import discover_designs, write_available_designs


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    report = discover_designs()
    path = write_available_designs(report)
    priority = [row for row in report["designs"] if int(row["priority_rank"]) < len(report["priority"])]
    print(f"Designs: {report['design_count']}")
    print(f"Missing priority: {', '.join(report['missing_priority']) or 'none'}")
    for row in priority:
        state = "finished" if row["finish_artifacts_present"] else "rtl-only"
        print(
            f"{row['design_name']} {row['platform']} top={row['top_module']} "
            f"complexity={row['estimated_complexity']} {state}"
        )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
