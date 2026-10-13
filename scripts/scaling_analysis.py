"""Run the M9.7 scaling measurements.

From the repository root:

    python scripts/scaling_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.scaling import run_scaling_analysis, write_scaling_outputs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, _found = load_study_graphs()
    if not graphs:
        print("No designs were loaded", file=sys.stderr)
        return 1
    payload = run_scaling_analysis(graphs)
    written = write_scaling_outputs(payload)
    bottlenecks = payload["bottleneck_analysis"]
    print(f"Complexity: {payload['complexity_analysis']['best_description']}")
    print(f"Fastest: {bottlenecks['fastest_model']}")
    print(f"Most efficient: {bottlenecks['most_efficient_model']}")
    print(f"Best F1 per second: {bottlenecks['best_accuracy_per_second_model']}")
    for path in written.values():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
