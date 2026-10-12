"""Summarize a timing-path label table.

Counts and slack statistics come from ``path_labels.csv``. This module
does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from slackgnn.dataset.timing_report import PATH_LABEL_FIELDS, TimingPath
from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

_AVERAGE_PLACES = Decimal("0.0001")


@dataclass(frozen=True)
class DatasetSummary:
    """Aggregate statistics for one path-label table."""

    total_paths: int
    timing_violations: int
    timing_met_paths: int
    worst_slack: Decimal
    average_slack: Decimal

    def as_json(self) -> str:
        return (
            "{\n"
            f'  "total_paths": {self.total_paths},\n'
            f'  "timing_violations": {self.timing_violations},\n'
            f'  "timing_met_paths": {self.timing_met_paths},\n'
            f'  "worst_slack": {format(self.worst_slack, "f")},\n'
            f'  "average_slack": {format(self.average_slack, "f")}\n'
            "}\n"
        )

    def as_text(self) -> str:
        return (
            f"total paths: {self.total_paths}\n"
            f"timing violations: {self.timing_violations}\n"
            f"timing-met paths: {self.timing_met_paths}\n"
            f"worst slack: {format(self.worst_slack, 'f')}\n"
            f"average slack: {format(self.average_slack, 'f')}\n"
        )


def load_path_labels(path: str | Path) -> tuple[tuple[TimingPath, ...], tuple[str, ...]]:
    """Read a path-label CSV. Return ``(paths, errors)``."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Cannot read path labels {source}: {exc}"
        logger.error(message)
        return (), (message,)

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return (), (f"Path labels {source} have no header",)
    header = tuple(reader.fieldnames)
    if header != PATH_LABEL_FIELDS:
        expected = ",".join(PATH_LABEL_FIELDS)
        found = ",".join(header)
        return (), (f"Path labels {source} header is {found!r}; expected {expected!r}",)

    paths: list[TimingPath] = []
    errors: list[str] = []
    for index, row in enumerate(reader, start=2):
        parsed, error = _row_to_path(row, index)
        if error:
            errors.append(f"{source} {error}")
            continue
        if parsed is not None:
            paths.append(parsed)
    if errors:
        return (), tuple(errors)
    if not paths:
        return (), (f"Path labels {source} contain no paths",)
    return tuple(paths), ()


def summarize_paths(paths: tuple[TimingPath, ...] | list[TimingPath]) -> DatasetSummary:
    """Compute violation counts and slack statistics."""
    if not paths:
        raise ValueError("Cannot summarize an empty path table")
    slacks = [path.slack for path in paths]
    violations = sum(1 for slack in slacks if slack < 0)
    average = sum(slacks, start=Decimal(0)) / Decimal(len(slacks))
    average = average.quantize(_AVERAGE_PLACES, rounding=ROUND_HALF_UP)
    return DatasetSummary(
        total_paths=len(paths),
        timing_violations=violations,
        timing_met_paths=len(paths) - violations,
        worst_slack=min(slacks),
        average_slack=average,
    )


def write_dataset_summary(summary: DatasetSummary, output: str | Path) -> tuple[str, ...]:
    """Write ``summary`` as JSON. Return error messages, empty on success."""
    destination = Path(output)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(summary.as_json(), encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()


def main(argv: list[str] | None = None) -> int:
    """Read path labels, print the summary, and write ``dataset_summary.json``."""
    parser = argparse.ArgumentParser(description="Summarize parsed timing-path labels.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=data_dir() / "gcd" / "parsed" / "path_labels.csv",
        help="Path-label CSV (default: data/gcd/parsed/path_labels.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir() / "gcd" / "parsed" / "dataset_summary.json",
        help="Summary JSON (default: data/gcd/parsed/dataset_summary.json)",
    )
    args = parser.parse_args(argv)

    if not args.labels.is_file():
        print(f"Missing path labels: {args.labels}", file=sys.stderr)
        return 1

    paths, errors = load_path_labels(args.labels)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1

    summary = summarize_paths(paths)
    write_errors = write_dataset_summary(summary, args.output)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1

    print(summary.as_text(), end="")
    return 0


def _row_to_path(row: dict[str, str | None], line_number: int) -> tuple[TimingPath | None, str | None]:
    values: dict[str, str] = {}
    for field in PATH_LABEL_FIELDS:
        raw = row.get(field)
        if raw is None or raw.strip() == "":
            return None, f"row {line_number} is missing {field}"
        values[field] = raw.strip()
    try:
        arrival = Decimal(values["arrival_time"])
        required = Decimal(values["required_time"])
        slack = Decimal(values["slack"])
    except InvalidOperation:
        return None, f"row {line_number} has a non-numeric timing field"
    return (
        TimingPath(
            startpoint=values["startpoint"],
            endpoint=values["endpoint"],
            path_group=values["path_group"],
            path_type=values["path_type"],
            arrival_time=arrival,
            required_time=required,
            slack=slack,
        ),
        None,
    )
