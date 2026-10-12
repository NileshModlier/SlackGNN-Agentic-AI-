"""Measure SlackGNN as the nested training set grows.

From the repository root:

    python scripts/scaling_study.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.reporting import fmt
from slackgnn.m9.studies import run_scaling


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, found = load_study_graphs()
    if not graphs:
        print("No designs with expanded labels were loaded", file=sys.stderr)
        return 1
    payload = run_scaling(graphs, found)
    print(f"Held out: {payload['held_out']}")
    for point in payload["points"]:
        print(
            f"{point['training_design_count']}: F1 {fmt(point['endpoint']['f1'])} "
            f"MAE {fmt(point['endpoint']['mae'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
