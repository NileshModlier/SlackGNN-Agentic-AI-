"""Run label expansion and every M9 study in one process.

From the repository root:

    python scripts/run_m9_studies.py

The full multitask seed-0 model is trained once and reused for the ablation
and robustness tables. Stage scripts retrain that model when run alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.expansion.label_expansion import expand_corpus, write_expanded_labels
from slackgnn.m9.corpus import load_study_graphs
from slackgnn.m9.documents import write_m9_documents
from slackgnn.m9.figures import write_m9_figures
from slackgnn.m9.studies import reports_dir, run_ablations, run_cross_design, run_robustness, run_scaling
from slackgnn.paths import repo_root


def main() -> int:
    log_dir = reports_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
        ],
    )
    payload = expand_corpus()
    write_expanded_labels(payload)
    graphs, found = load_study_graphs()
    if not graphs:
        print("No designs with pyg/graph.pt could be prepared", file=sys.stderr)
        return 1
    cross = run_cross_design(graphs, found)
    run_ablations(graphs, found, full_result=cross)
    run_robustness(graphs, found, seed0_result=cross)
    run_scaling(graphs, found)
    figures = write_m9_figures()
    report, readme = write_m9_documents()
    print(f"Figures: {len(figures)}")
    print(report)
    print(readme)
    print(repo_root())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
