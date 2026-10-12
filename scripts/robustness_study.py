"""Repeat full multitask leave-one-design-out at seeds 0, 1, and 2.

From the repository root:

    python scripts/robustness_study.py
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
from slackgnn.m9.studies import run_robustness


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, found = load_study_graphs()
    if not graphs:
        print("No designs with expanded labels were loaded", file=sys.stderr)
        return 1
    payload = run_robustness(graphs, found)
    across = payload["across_seeds"]["endpoint"]
    print(f"F1 {fmt(across['f1']['mean'])} spread {fmt(across['f1']['std'])}")
    print(f"ROC-AUC {fmt(across['roc_auc']['mean'])} spread {fmt(across['roc_auc']['std'])}")
    print(f"MAE {fmt(across['mae']['mean'])} spread {fmt(across['mae']['std'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
