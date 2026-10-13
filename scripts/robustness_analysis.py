"""Run the M9.6 robustness studies.

From the repository root:

    python scripts/robustness_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.robustness import load_design_routes, run_robustness_study, write_robustness_outputs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, _found = load_study_graphs()
    if len(graphs) < 2:
        print("Robustness needs at least two designs", file=sys.stderr)
        return 1
    routes = load_design_routes()
    checkpoint = Path(__file__).resolve().parents[1] / "reports" / "robustness"
    payload = run_robustness_study(graphs, routes, checkpoint_dir=checkpoint)
    written = write_robustness_outputs(payload)
    comparison = payload["comparison"]
    print(f"Most robust: {comparison['most_robust_model']}")
    print(f"Least robust: {comparison['least_robust_model']}")
    print(f"Stable corpus size: {comparison['minimum_stable_corpus_size']}")
    for name, path in written.items():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
