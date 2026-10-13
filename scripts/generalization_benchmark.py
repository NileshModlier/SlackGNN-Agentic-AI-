"""Leave-one-design-out comparison of GCN, GraphSAGE, GIN, and SlackGNN.

From the repository root:

    python scripts/generalization_benchmark.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.generalization import run_generalization, write_generalization_figures, write_generalization_report
from slackgnn.m9.reporting import fmt


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    graphs, found = load_study_graphs()
    if len(graphs) < 2:
        print("Leave-one-design-out needs at least two designs", file=sys.stderr)
        return 1
    payload = run_generalization(graphs, found)
    json_path, markdown_path = write_generalization_report(payload)
    figures = write_generalization_figures(payload)
    ranking = payload["ranking"]
    print(f"Designs: {', '.join(payload['designs'])}")
    print(f"Easiest: {ranking['easiest_design']} F1 {fmt(ranking['easiest_mean_endpoint_f1'])}")
    print(f"Hardest: {ranking['hardest_design']} F1 {fmt(ranking['hardest_mean_endpoint_f1'])}")
    print(f"Most transferable: {ranking['most_transferable_architecture']}")
    for name, score in ranking["architecture_endpoint_f1"].items():
        print(f"{name} endpoint F1 {fmt(score)}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    for path in figures:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
