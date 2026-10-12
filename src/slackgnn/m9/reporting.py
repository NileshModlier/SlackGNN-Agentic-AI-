"""Markdown tables and JSON writers for the M9 reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

from slackgnn.m9.metrics_agg import METRIC_KEYS

_HEADERS = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "MAE (ns)",
    "MSE",
    "RMSE (ns)",
    "Support",
)
_KEYS = METRIC_KEYS + ("support",)


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8", newline="\n")
    return destination


def write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


def json_ready(value: object) -> object:
    """Convert study objects to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def fmt(value: object, digits: int = 4) -> str:
    """Format a metric, a mean, or a missing value."""
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return "n/a"
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def metric_cells(row: dict[str, object]) -> list[str]:
    cells = [fmt(row.get(key)) for key in METRIC_KEYS]
    support = row.get("support")
    cells.append("n/a" if support is None else str(int(support)))
    return cells


def markdown_table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    alignments = ["---"] * len(headers)
    for index in range(numeric_from, len(headers)):
        alignments[index] = "---:"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def scope_table(folds: list[dict[str, object]], scope: str) -> str:
    headers = ["Held-out design", *_HEADERS]
    rows = []
    for fold in folds:
        metrics = fold[scope]
        rows.append([str(fold["held_out"]), *metric_cells(metrics)])
    return markdown_table(headers, rows)


def summary_line(summary: dict[str, object], scope: str) -> str:
    block = summary[scope]
    names = _HEADERS[:-1]
    cells = [fmt(block[key]) for key in METRIC_KEYS]
    return "Mean " + scope + ": " + ", ".join(f"{name} {value}" for name, value in zip(names, cells, strict=True))


def missing_lines(missing: list[dict[str, str]]) -> str:
    if not missing:
        return "Every requested design had a published graph."
    return "\n".join(f"- `{item['design']}`: {item['reason']}" for item in missing)
