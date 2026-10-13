"""Run the M9.5 SlackGNN feature ablations.

From the repository root:

    python scripts/feature_ablation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.feature_ablation import (
    run_feature_ablations,
    write_feature_ablation_figure,
    write_feature_ablation_report,
)
from slackgnn.m9.reporting import fmt


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, found = load_study_graphs()
    if len(graphs) < 2:
        print("Ablation needs at least two designs", file=sys.stderr)
        return 1
    payload = run_feature_ablations(graphs, found)
    json_path, markdown_path = write_feature_ablation_report(payload)
    figure = write_feature_ablation_figure(payload)
    comparison = payload["comparison"]
    print(f"Full endpoint F1 {fmt(comparison['full_endpoint_f1'])}")
    print(f"Largest F1 drop: {comparison['largest_f1_drop']}")
    print(f"Largest MAE increase: {comparison['largest_mae_increase']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
