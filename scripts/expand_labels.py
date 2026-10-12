"""Propagate reported slack onto signal-flow routes.

From the repository root:

    python scripts/expand_labels.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.expansion.label_expansion import expand_corpus, write_expanded_labels


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = expand_corpus()
    json_path, tensor_path = write_expanded_labels(payload)
    summary = payload["summary"]
    print(f"Designs: {summary['design_count']}")
    print(f"Original labels: {summary['original_label_count']}")
    print(f"Expanded labels: {summary['expanded_label_count']}")
    print(f"Added labels: {summary['added_label_count']}")
    for item in summary["designs"]:
        if not item.get("present"):
            print(f"{item['design_name']}: failed")
            continue
        print(
            f"{item['design_name']}: {item['original_label_count']} -> {item['expanded_label_count']} "
            f"(report-only paths {item['paths_only_in_report']})"
        )
    for item in payload["missing_designs"]:
        print(f"missing {item['design']}: {item['reason']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tensor_path}")
    return 0 if summary["design_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
