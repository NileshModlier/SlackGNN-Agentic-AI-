"""Leave-one-design-out SlackGNN benchmark on expanded labels.

From the repository root:

    python scripts/cross_design_benchmark.py
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
from slackgnn.m9.studies import run_cross_design


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, found = load_study_graphs()
    if not graphs:
        print("No designs with expanded labels were loaded", file=sys.stderr)
        return 1
    payload = run_cross_design(graphs, found)
    print(f"Endpoint mean F1 {fmt(payload['summary']['endpoint']['f1'])}")
    print(f"Endpoint mean ROC-AUC {fmt(payload['summary']['endpoint']['roc_auc'])}")
    print(f"Endpoint mean MAE {fmt(payload['summary']['endpoint']['mae'])} ns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
