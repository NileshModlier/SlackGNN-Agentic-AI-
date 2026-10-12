"""Expand the SlackGNN corpus with one OpenROAD benchmark.

From the repository root:

    python scripts/expand_corpus.py --design uart
    python scripts/expand_corpus.py --design uart --platform asap7
    python scripts/expand_corpus.py --design uart --skip-flow

The command runs ORFS when finish files are absent, copies
``6_final.v``, ``6_final.sdc``, ``6_final.spef``, and ``6_finish.rpt``,
then runs ``process_design.py`` and ``build_corpus.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from slackgnn.expansion.expand import expand_design


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ORFS for one design and publish it.")
    parser.add_argument("--design", required=True, help="ORFS design directory name, such as uart")
    parser.add_argument("--platform", default=None, help="Platform directory. Default prefers sky130hd.")
    parser.add_argument("--skip-flow", action="store_true", help="Import existing finish files and do not run ORFS")
    parser.add_argument("--force", action="store_true", help="Allow replacement of gcd, aes, ibex, or jpeg")
    args = parser.parse_args(argv)
    return expand_design(args.design, args.platform, run_flow=not args.skip_flow, force=args.force)


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
