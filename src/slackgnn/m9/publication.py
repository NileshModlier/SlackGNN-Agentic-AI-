"""Publication figures for the six-design SlackGNN study.

The charts read the saved corpus and generalization reports. They do not
retrain models or change earlier milestone outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from slackgnn.m9.reporting import write_text
from slackgnn.paths import repo_root

DESIGN_ORDER = ("gcd", "aes", "ibex", "jpeg", "riscv32i", "chameleon")
MODEL_ORDER = ("gcn", "graphsage", "gin", "slackgnn")
MODEL_LABELS = {"gcn": "GCN", "graphsage": "GraphSAGE", "gin": "GIN", "slackgnn": "SlackGNN"}
_MODEL_COLORS = {
    "gcn": "#1f4e79",
    "graphsage": "#2f6f4e",
    "gin": "#8c2f39",
    "slackgnn": "#c47b2b",
}
_DESIGN_COLORS = {
    "gcd": "#4c78a8",
    "aes": "#f58518",
    "ibex": "#54a24b",
    "jpeg": "#e45756",
    "riscv32i": "#72b7b2",
    "chameleon": "#b279a2",
}
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def cumulative_growth(designs: list[dict[str, object]], order: tuple[str, ...] = DESIGN_ORDER) -> list[dict[str, object]]:
    """Accumulate nodes, edges, and expanded labels in corpus order."""
    by_name = {str(row["design"]): row for row in designs}
    nodes = edges = labels = 0
    points: list[dict[str, object]] = []
    for name in order:
        row = by_name[name]
        nodes += int(row["nodes"])
        edges += int(row["edges"])
        labels += int(row["expanded_labels"])
        points.append({"design": name, "nodes": nodes, "edges": edges, "labels": labels})
    return points


def rank_holdouts(models: dict[str, dict[str, object]]) -> dict[str, object]:
    """Rank designs by the mean of defined endpoint F1 scores."""
    grouped: dict[str, list[float]] = {}
    for model in models.values():
        for fold in model["folds"]:
            value = _number(fold["endpoint"].get("f1"))
            if value is None:
                continue
            grouped.setdefault(str(fold["held_out"]), []).append(value)
    means = {name: sum(values) / len(values) for name, values in grouped.items() if values}
    easiest = max(means, key=means.get) if means else None
    hardest = min(means, key=means.get) if means else None
    return {
        "means": means,
        "easiest": easiest,
        "hardest": hardest,
        "easiest_f1": means.get(easiest) if easiest else None,
        "hardest_f1": means.get(hardest) if hardest else None,
    }


def write_publication_figures(
    designs: list[dict[str, object]],
    generalization: dict[str, object],
    directory: str | Path | None = None,
) -> list[Path]:
    """Write the dataset, performance, and generalization figures."""
    figure_dir = Path(directory) if directory is not None else repo_root() / "reports" / "figures" / "publication"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranking = rank_holdouts(generalization["models"])
    with plt.rc_context(_STYLE):
        written = [
            _composition(designs, figure_dir / "corpus_composition.png"),
            _growth(cumulative_growth(designs), figure_dir / "corpus_growth.png"),
            _complexity(designs, figure_dir / "design_complexity.png"),
            _model_bars(generalization, "f1", "Endpoint F1", figure_dir / "f1_comparison.png", 0, 1.18),
            _model_bars(generalization, "roc_auc", "Endpoint ROC-AUC", figure_dir / "roc_auc_comparison.png", 0, 1.18),
            _model_bars(generalization, "mae", "Endpoint MAE (ns)", figure_dir / "mae_comparison.png", 0, None),
            _model_bars(generalization, "rmse", "Endpoint RMSE (ns)", figure_dir / "rmse_comparison.png", 0, None),
            _holdout_bars(generalization, "f1", "Leave-one-design-out F1", figure_dir / "lodo_f1.png", 0, 1.08),
            _holdout_bars(generalization, "roc_auc", "Holdout ROC-AUC", figure_dir / "holdout_roc_auc.png", 0, 1.08),
            _holdout_bars(generalization, "mae", "Holdout MAE (ns)", figure_dir / "holdout_mae.png", 0, None),
            _difficulty(ranking, figure_dir / "design_difficulty.png"),
        ]
    write_text(repo_root() / "reports" / "publication_figures.md", render_publication_index(written, ranking, designs))
    return written


def render_publication_index(paths: list[Path], ranking: dict[str, object], designs: list[dict[str, object]]) -> str:
    """List the figure files and the holdout difficulty result."""
    totals = {
        "nodes": sum(int(row["nodes"]) for row in designs),
        "edges": sum(int(row["edges"]) for row in designs),
        "labels": sum(int(row["expanded_labels"]) for row in designs),
    }
    lines = [
        "# Publication figures",
        "",
        "Charts use the six published designs and the M9.4 leave-one-design-out endpoint scores.",
        "",
        f"Corpus: {len(designs)} designs, {totals['nodes']} nodes, {totals['edges']} edges, {totals['labels']} expanded labels.",
        "",
        f"Easiest holdout by mean defined endpoint F1: {ranking['easiest']} ({_fmt(ranking['easiest_f1'])}).",
        f"Hardest holdout by mean defined endpoint F1: {ranking['hardest']} ({_fmt(ranking['hardest_f1'])}).",
        "",
        "Undefined F1 values, from holdouts where a model predicted no violations, are left out of that mean.",
        "",
        "## Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `reports/figures/publication/{path.name}`")
    lines.append("")
    return "\n".join(lines)


def load_publication_inputs(
    scaling_path: str | Path | None = None,
    generalization_path: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load the saved design inventory and generalization report."""
    root = repo_root()
    scaling_file = Path(scaling_path) if scaling_path is not None else root / "reports" / "scaling" / "design_scaling.json"
    generalization_file = Path(generalization_path) if generalization_path is not None else root / "reports" / "generalization_results.json"
    scaling = json.loads(scaling_file.read_text(encoding="utf-8"))
    generalization = json.loads(generalization_file.read_text(encoding="utf-8"))
    return list(scaling["designs"]), generalization


def _composition(designs: list[dict[str, object]], path: Path) -> Path:
    ordered = _ordered(designs)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 4.4), sharey=True)
    specs = (("nodes", "Nodes"), ("edges", "Edges"), ("expanded_labels", "Expanded labels"))
    positions = list(range(len(ordered)))
    labels = [str(row["design"]) for row in ordered]
    for axis, (key, title) in zip(axes, specs, strict=True):
        axis.barh(positions, [int(row[key]) for row in ordered], color=[_DESIGN_COLORS[name] for name in labels])
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
        axis.invert_yaxis()
    axes[0].set_yticks(positions, labels)
    figure.suptitle("Corpus composition", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _growth(points: list[dict[str, object]], path: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 4.2))
    names = [str(point["design"]) for point in points]
    positions = list(range(len(points)))
    specs = (("nodes", "Cumulative nodes"), ("edges", "Cumulative edges"), ("labels", "Cumulative labels"))
    for axis, (key, title) in zip(axes, specs, strict=True):
        axis.plot(positions, [int(point[key]) for point in points], color="#1f4e79", marker="o", linewidth=2.2)
        axis.set_xticks(positions, names, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    figure.suptitle("Corpus growth", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _complexity(designs: list[dict[str, object]], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in _ordered(designs):
        name = str(row["design"])
        axis.scatter(int(row["nodes"]), int(row["edges"]), s=80, color=_DESIGN_COLORS[name], label=name, zorder=3)
    axis.set_xlabel("Nodes")
    axis.set_ylabel("Edges")
    axis.set_title("Design complexity")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _model_bars(payload: dict[str, object], key: str, ylabel: str, path: Path, ymin: float, ymax: float | None) -> Path:
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    values = [_summary_mean(payload, name, key) for name in MODEL_ORDER]
    bars = axis.bar(labels, [_or_zero(value) for value in values], color=[_MODEL_COLORS[name] for name in MODEL_ORDER])
    _label_bars(axis, bars, values)
    axis.set_ylabel(ylabel)
    axis.set_ylim(ymin, ymax if ymax is not None else _headroom(values))
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _holdout_bars(payload: dict[str, object], key: str, title: str, path: Path, ymin: float, ymax: float | None) -> Path:
    designs = [str(fold["held_out"]) for fold in payload["models"]["gcn"]["folds"]]
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    width = 0.18
    positions = list(range(len(designs)))
    flat: list[float] = []
    for offset, name in enumerate(MODEL_ORDER):
        values = [_fold_value(payload, name, design, key) for design in designs]
        flat.extend(value for value in values if value is not None)
        axis.bar(
            [item + (offset - 1.5) * width for item in positions],
            [_or_nan(value) for value in values],
            width=width,
            color=_MODEL_COLORS[name],
            label=MODEL_LABELS[name],
        )
    axis.set_xticks(positions, designs)
    axis.set_ylabel(title)
    axis.set_ylim(ymin, ymax if ymax is not None else _headroom(flat))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False, ncol=4)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _difficulty(ranking: dict[str, object], path: Path) -> Path:
    means = dict(ranking["means"])
    ordered = sorted(means, key=lambda name: float(means[name]))
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = list(range(len(ordered)))
    axis.barh(positions, [float(means[name]) for name in ordered], color=[_DESIGN_COLORS[name] for name in ordered])
    axis.set_yticks(positions, ordered)
    axis.set_xlabel("Mean endpoint F1")
    axis.set_xlim(0, 1.15)
    axis.set_title("Design difficulty")
    axis.grid(axis="x", alpha=0.3)
    for name, label, color in (
        (ranking["hardest"], "hardest", "#8c2f39"),
        (ranking["easiest"], "easiest", "#2f6f4e"),
    ):
        if name not in means:
            continue
        position = ordered.index(str(name))
        axis.text(float(means[str(name)]) + 0.02, position, label, va="center", color=color, fontsize=10)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def _ordered(designs: list[dict[str, object]]) -> list[dict[str, object]]:
    by_name = {str(row["design"]): row for row in designs}
    return [by_name[name] for name in DESIGN_ORDER if name in by_name]


def _summary_mean(payload: dict[str, object], model: str, key: str) -> float | None:
    return _number(payload["models"][model]["summary"]["endpoint"][key])


def _fold_value(payload: dict[str, object], model: str, design: str, key: str) -> float | None:
    fold = next(item for item in payload["models"][model]["folds"] if item["held_out"] == design)
    return _number(fold["endpoint"].get(key))


def _number(value: object) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _label_bars(axis: Axes, bars, values: list[float | None]) -> None:
    for bar, value in zip(bars, values, strict=True):
        if value is None:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _headroom(values: list[float]) -> float:
    if not values:
        return 1.0
    return max(values) * 1.18


def _or_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _or_nan(value: float | None) -> float:
    return float("nan") if value is None else value


def _fmt(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"
