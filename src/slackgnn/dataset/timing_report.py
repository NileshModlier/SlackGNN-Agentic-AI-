"""Parse OpenROAD ``report_checks`` paths into timing-slack labels.

The parser reads path blocks from a finish report. It does not build a
graph and it does not train a model. OpenROAD reprints the same check
under later headings (``-unconstrained``, ``reg to reg``). Those reprints
collapse to one label so the table counts each path once.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from slackgnn.paths import data_dir

logger = logging.getLogger(__name__)

PATH_LABEL_FIELDS = (
    "startpoint",
    "endpoint",
    "path_group",
    "path_type",
    "arrival_time",
    "required_time",
    "slack",
)

_BLOCK_SPLIT = re.compile(r"(?=^Startpoint:)", re.MULTILINE)
_STARTPOINT_RE = re.compile(r"^Startpoint:\s+(.+?)\s*$", re.MULTILINE)
_ENDPOINT_RE = re.compile(r"^Endpoint:\s+(.+?)\s*$", re.MULTILINE)
_PATH_GROUP_RE = re.compile(r"^Path Group:\s+(\S(?:.*\S)?)\s*$", re.MULTILINE)
_PATH_TYPE_RE = re.compile(r"^Path Type:\s+(\S+)\s*$", re.MULTILINE)
_ARRIVAL_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+data arrival time\s*$",
    re.MULTILINE,
)
_REQUIRED_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+data required time\s*$",
    re.MULTILINE,
)
_SLACK_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+slack\s+\((MET|VIOLATED)\)\s*$",
    re.MULTILINE,
)
_TRAILING_NOTE_RE = re.compile(r"\s+\([^)]*\)\s*$")
_SLACK_TOLERANCE = Decimal("0.011")


@dataclass(frozen=True)
class TimingPath:
    """One timing check extracted from an OpenROAD path report."""

    startpoint: str
    endpoint: str
    path_group: str
    path_type: str
    arrival_time: Decimal
    required_time: Decimal
    slack: Decimal

    def as_row(self) -> dict[str, str]:
        return {
            "startpoint": self.startpoint,
            "endpoint": self.endpoint,
            "path_group": self.path_group,
            "path_type": self.path_type,
            "arrival_time": format(self.arrival_time, "f"),
            "required_time": format(self.required_time, "f"),
            "slack": format(self.slack, "f"),
        }


@dataclass(frozen=True)
class ParseResult:
    """Unique paths from one report, plus any blocking error messages."""

    paths: tuple[TimingPath, ...]
    errors: tuple[str, ...]
    path_blocks: int

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_timing_report(path: str | Path) -> ParseResult:
    """Parse every ``report_checks`` path in ``path``."""
    report = Path(path)
    try:
        text = report.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"Cannot read timing report {report}: {exc}"
        logger.error(message)
        return ParseResult(paths=(), errors=(message,), path_blocks=0)
    except UnicodeError as exc:
        message = f"Timing report {report} is not UTF-8 text: {exc}"
        logger.error(message)
        return ParseResult(paths=(), errors=(message,), path_blocks=0)

    result = parse_timing_report_text(text.lstrip("\ufeff"))
    if result.errors:
        for message in result.errors:
            logger.error("%s: %s", report, message)
        return result

    logger.info(
        "Parsed %d path blocks from %s into %d unique paths",
        result.path_blocks,
        report,
        len(result.paths),
    )
    return result


def parse_timing_report_text(text: str) -> ParseResult:
    """Parse report text. ``path_blocks`` counts headings before de-duplication."""
    if not text.strip():
        return ParseResult(paths=(), errors=("Timing report is empty",), path_blocks=0)

    blocks = [
        block for block in _BLOCK_SPLIT.split(text) if block.startswith("Startpoint:")
    ]
    if not blocks:
        return ParseResult(
            paths=(),
            errors=("No timing paths found in the report",),
            path_blocks=0,
        )

    errors: list[str] = []
    parsed: list[TimingPath] = []
    for index, block in enumerate(blocks, start=1):
        path, block_errors = _parse_block(block, index)
        errors.extend(block_errors)
        if path is not None:
            parsed.append(path)

    if errors:
        return ParseResult(paths=(), errors=tuple(errors), path_blocks=len(blocks))

    unique = _unique_paths(parsed)
    dropped = len(parsed) - len(unique)
    if dropped:
        logger.info("Collapsed %d reprinted path blocks into existing labels", dropped)
    return ParseResult(paths=tuple(unique), errors=(), path_blocks=len(blocks))


def write_path_labels(paths: tuple[TimingPath, ...] | list[TimingPath], output: str | Path) -> tuple[str, ...]:
    """Write ``paths`` to a CSV file. Return error messages, empty on success."""
    destination = Path(output)
    rows = [path.as_row() for path in paths]
    payload = _render_csv(rows)
    return _write_text(destination, payload)


def main(argv: list[str] | None = None) -> int:
    """Parse the GCD finish report and write ``path_labels.csv``."""
    parser = argparse.ArgumentParser(
        description="Extract timing-path labels from an OpenROAD finish report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=data_dir() / "gcd" / "reports" / "6_finish.rpt",
        help="OpenROAD timing report (default: data/gcd/reports/6_finish.rpt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir() / "gcd" / "parsed" / "path_labels.csv",
        help="CSV path-label table (default: data/gcd/parsed/path_labels.csv)",
    )
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"Missing timing report: {args.report}", file=sys.stderr)
        return 1

    result = parse_timing_report(args.report)
    if result.errors:
        for message in result.errors:
            print(message, file=sys.stderr)
        return 1

    write_errors = write_path_labels(result.paths, args.output)
    if write_errors:
        for message in write_errors:
            print(message, file=sys.stderr)
        return 1

    print(f"Wrote {len(result.paths)} paths to {args.output}")
    return 0


def _parse_block(block: str, index: int) -> tuple[TimingPath | None, list[str]]:
    prefix = f"Path block {index}"
    errors: list[str] = []

    startpoint = _point_name(_STARTPOINT_RE, block)
    endpoint = _point_name(_ENDPOINT_RE, block)
    path_group = _sole_text(_PATH_GROUP_RE, block)
    path_type = _sole_text(_PATH_TYPE_RE, block)
    label = startpoint or "unknown startpoint"

    if startpoint is None:
        errors.append(f"{prefix} is missing a startpoint")
    if endpoint is None:
        errors.append(f"{prefix} ({label}) is missing an endpoint")
    if path_group is None:
        errors.append(f"{prefix} ({label}) is missing a path group")
    if path_type is None:
        errors.append(f"{prefix} ({label}) is missing a path type")

    arrival, arrival_errors = _arrival_time(block, prefix, label)
    required, required_errors = _required_time(block, prefix, label)
    slack, slack_errors = _slack(block, prefix, label)
    errors.extend(arrival_errors)
    errors.extend(required_errors)
    errors.extend(slack_errors)

    if errors or None in (startpoint, endpoint, path_group, path_type, arrival, required, slack):
        return None, errors

    assert startpoint is not None
    assert endpoint is not None
    assert path_group is not None
    assert path_type is not None
    assert arrival is not None
    assert required is not None
    assert slack is not None

    relation_error = _check_slack_equation(path_type, arrival, required, slack, prefix, label)
    if relation_error:
        return None, [relation_error]

    return (
        TimingPath(
            startpoint=startpoint,
            endpoint=endpoint,
            path_group=path_group,
            path_type=path_type,
            arrival_time=arrival,
            required_time=required,
            slack=slack,
        ),
        [],
    )


def _point_name(pattern: re.Pattern[str], block: str) -> str | None:
    found = pattern.findall(block)
    if len(found) != 1:
        return None
    name = _TRAILING_NOTE_RE.sub("", found[0]).strip()
    return name or None


def _sole_text(pattern: re.Pattern[str], block: str) -> str | None:
    found = pattern.findall(block)
    if len(found) != 1:
        return None
    return found[0]


def _arrival_time(block: str, prefix: str, label: str) -> tuple[Decimal | None, list[str]]:
    tokens = _ARRIVAL_RE.findall(block)
    if not tokens:
        return None, [f"{prefix} ({label}) is missing data arrival time"]
    arrival = Decimal(tokens[0])
    if len(tokens) > 1 and Decimal(tokens[-1]) != -arrival:
        return None, [
            f"{prefix} ({label}) has inconsistent data arrival time values: {', '.join(tokens)}"
        ]
    return arrival, []


def _required_time(block: str, prefix: str, label: str) -> tuple[Decimal | None, list[str]]:
    tokens = _REQUIRED_RE.findall(block)
    if not tokens:
        return None, [f"{prefix} ({label}) is missing data required time"]
    required = Decimal(tokens[0])
    if any(Decimal(token) != required for token in tokens[1:]):
        return None, [
            f"{prefix} ({label}) has inconsistent data required time values: {', '.join(tokens)}"
        ]
    return required, []


def _slack(block: str, prefix: str, label: str) -> tuple[Decimal | None, list[str]]:
    found = _SLACK_RE.findall(block)
    if len(found) != 1:
        return None, [f"{prefix} ({label}) is missing slack"]
    value = Decimal(found[0][0])
    status = found[0][1]
    if status == "VIOLATED" and value >= 0:
        return None, [f"{prefix} ({label}) is marked VIOLATED with slack {value}"]
    if status == "MET" and value < 0:
        return None, [f"{prefix} ({label}) is marked MET with slack {value}"]
    return value, []


def _check_slack_equation(
    path_type: str,
    arrival: Decimal,
    required: Decimal,
    slack: Decimal,
    prefix: str,
    label: str,
) -> str | None:
    if path_type == "max":
        expected = required - arrival
    elif path_type == "min":
        expected = arrival - required
    else:
        logger.warning(
            "%s (%s) has path type %s; slack equation was not checked",
            prefix,
            label,
            path_type,
        )
        return None
    if abs(expected - slack) > _SLACK_TOLERANCE:
        return (
            f"{prefix} ({label}) slack {slack} does not match "
            f"{path_type} arrival {arrival} and required {required}"
        )
    return None


def _unique_paths(paths: list[TimingPath]) -> list[TimingPath]:
    unique: list[TimingPath] = []
    seen: set[tuple[str, str, str, str, Decimal, Decimal, Decimal]] = set()
    for path in paths:
        key = (
            path.startpoint,
            path.endpoint,
            path.path_group,
            path.path_type,
            path.arrival_time,
            path.required_time,
            path.slack,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _render_csv(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=PATH_LABEL_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_text(destination: Path, payload: str) -> tuple[str, ...]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    except OSError as exc:
        message = f"Cannot write {destination}: {exc}"
        logger.error(message)
        return (message,)
    logger.info("Wrote %s", destination)
    return ()
