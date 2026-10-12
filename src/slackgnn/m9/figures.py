"""Publication figures for the M9 studies. Existing M7 and M8 figures are left in place."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slackgnn.paths import repo_root
from slackgnn.training.metrics import roc_curve

_ENDPOINT = "#1f4e79"
_EXPANDED = "#c47b2b"
_SEED = "#2f6f4e"


def write_m9_figures(report_dir: str | Path | None = None) -> list[Path]:
    """Read the M9 JSON reports and write the figure set."""
    directory = Path(report_dir) if report_dir is not None else repo_root() / "reports" / "m9"
    figure_dir = directory / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    expansion = _read(directory / "expanded_labels.json")
    cross = _read(directory / "cross_design_results.json")
    ablation = _read(directory / "ablation_results.json")
    robustness = _read(directory / "robustness_results.json")
    scaling = _read(directory / "scaling_results.json")
    if expansion is not None:
        written.append(_label_expansion(expansion, figure_dir / "label_expansion.png"))
    if cross is not None:
        written.append(_cross_design(cross, figure_dir / "cross_design_f1_mae.png"))
        written.append(_roc(cross, figure_dir / "roc_curves.png"))
        written.append(_scatter(cross, figure_dir / "slack_scatter.png"))
    if ablation is not None:
        written.append(_ablation(ablation, figure_dir / "ablation.png"))
    if robustness is not None:
        written.append(_robustness(robustness, figure_dir / "robustness.png"))
    if scaling is not None:
        written.append(_scaling(scaling, figure_dir / "scaling.png"))
    return written


def _label_expansion(payload: dict[str, object], path: Path) -> Path:
    rows = [row for row in payload["designs"] if row.get("present", True) and "original_label_count" in row]
    names = [str(row["design_name"]) for row in rows]
    original = [int(row["original_label_count"]) for row in rows]
    expanded = [int(row["expanded_label_count"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = range(len(names))
    width = 0.36
    axis.bar([item - width / 2 for item in positions], original, width=width, label="Published endpoints", color=_ENDPOINT)
    axis.bar([item + width / 2 for item in positions], expanded, width=width, label="Expanded labels", color=_EXPANDED)
    axis.set_xticks(list(positions), names)
    axis.set_ylabel("Supervised nodes")
    axis.set_xlabel("Design")
    axis.set_title("Label expansion")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def _cross_design(payload: dict[str, object], path: Path) -> Path:
    folds = payload["folds"]
    names = [str(fold["held_out"]) for fold in folds]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _grouped_bars(
        axes[0],
        names,
        [fold["endpoint"]["f1"] for fold in folds],
        [fold["expanded"]["f1"] for fold in folds],
        "F1",
        "F1",
    )
    _grouped_bars(
        axes[1],
        names,
        [fold["endpoint"]["mae"] for fold in folds],
        [fold["expanded"]["mae"] for fold in folds],
        "MAE (ns)",
        "Slack MAE",
    )
    figure.suptitle("Leave-one-design-out")
    return _save(figure, path)


def _ablation(payload: dict[str, object], path: Path) -> Path:
    rows = payload["ablations"]
    names = [str(row["name"]).replace("_", "\n") for row in rows]
    f1 = [row["summary"]["endpoint"]["f1"]["mean"] for row in rows]
    mae = [row["summary"]["endpoint"]["mae"]["mean"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(range(len(names)), [_or_nan(value) for value in f1], color=_ENDPOINT)
    axes[1].bar(range(len(names)), [_or_nan(value) for value in mae], color=_EXPANDED)
    for axis, label in ((axes[0], "Mean endpoint F1"), (axes[1], "Mean endpoint MAE (ns)")):
        axis.set_xticks(range(len(names)), names, fontsize=8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Ablations on published endpoint labels")
    return _save(figure, path)


def _robustness(payload: dict[str, object], path: Path) -> Path:
    across = payload["across_seeds"]["endpoint"]
    seeds = payload["per_seed"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    class_names = ["F1", "ROC-AUC"]
    class_means = [across["f1"]["mean"], across["roc_auc"]["mean"]]
    class_std = [across["f1"]["std"], across["roc_auc"]["std"]]
    axes[0].bar(class_names, [_or_nan(value) for value in class_means], yerr=[_or_zero(value) for value in class_std], color=_ENDPOINT, capsize=4)
    for index, key in enumerate(("f1", "roc_auc")):
        points = [row["summary"]["endpoint"][key]["mean"] for row in seeds]
        axes[0].scatter([index] * len(points), [_or_nan(value) for value in points], color=_SEED, zorder=3, label="Seed mean" if index == 0 else None)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Leave-one-design-out mean")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(["MAE"], [_or_nan(across["mae"]["mean"])], yerr=[_or_zero(across["mae"]["std"])], color=_EXPANDED, capsize=4)
    mae_points = [row["summary"]["endpoint"]["mae"]["mean"] for row in seeds]
    axes[1].scatter([0] * len(mae_points), [_or_nan(value) for value in mae_points], color=_SEED, zorder=3, label="Seed mean")
    axes[1].set_ylabel("MAE (ns)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    figure.suptitle("Robustness across seeds")
    return _save(figure, path)


def _scaling(payload: dict[str, object], path: Path) -> Path:
    points = payload["points"]
    counts = [int(point["training_design_count"]) for point in points]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if counts:
        axes[0].plot(counts, [_or_nan(point["endpoint"]["f1"]) for point in points], marker="o", color=_ENDPOINT, label="Endpoint")
        axes[0].plot(counts, [_or_nan(point["expanded"]["f1"]) for point in points], marker="o", color=_EXPANDED, label="Expanded")
        axes[1].plot(counts, [_or_nan(point["endpoint"]["mae"]) for point in points], marker="o", color=_ENDPOINT, label="Endpoint")
        axes[1].plot(counts, [_or_nan(point["expanded"]["mae"]) for point in points], marker="o", color=_EXPANDED, label="Expanded")
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("MAE (ns)")
    for axis in axes:
        axis.set_xlabel("Training designs")
        axis.grid(alpha=0.3)
        axis.legend()
    held = payload.get("held_out") or "none"
    figure.suptitle(f"Scaling with {held} held out")
    return _save(figure, path)


def _roc(payload: dict[str, object], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1, label="Chance")
    styles = (
        {"color": "#1f4e79", "linestyle": "-", "linewidth": 2.4},
        {"color": "#c47b2b", "linestyle": "--", "linewidth": 2.2},
        {"color": "#2f6f4e", "linestyle": "-.", "linewidth": 2.2},
        {"color": "#8c2f39", "linestyle": ":", "linewidth": 2.6},
    )
    perfect: list[str] = []
    for index, fold in enumerate(payload["folds"]):
        predictions = fold.get("endpoint_predictions") or {}
        labels = [int(value) for value in predictions.get("label", [])]
        scores = [float(value) for value in predictions.get("score", [])]
        if not labels or len(set(labels)) < 2:
            continue
        fpr, tpr = roc_curve(labels, scores)
        auc = fold["endpoint"].get("roc_auc")
        auc_text = "n/a" if auc is None else f"{float(auc):.3f}"
        if auc is not None and float(auc) >= 0.999:
            perfect.append(str(fold["held_out"]))
        style = styles[index % len(styles)]
        axis.plot(fpr, tpr, label=f"{fold['held_out']} (AUC {auc_text})", **style)
    if len(perfect) > 1:
        axis.text(
            0.08,
            0.72,
            "Overlapping perfect ROC:\n" + ", ".join(perfect),
            fontsize=9,
            color="#333333",
        )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Endpoint ROC, full model")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    return _save(figure, path)


def _scatter(payload: dict[str, object], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    actual_all: list[float] = []
    predicted_all: list[float] = []
    for fold in payload["folds"]:
        predictions = fold.get("endpoint_predictions") or {}
        actual = [float(value) for value in predictions.get("slack", [])]
        predicted = [float(value) for value in predictions.get("prediction", [])]
        axis.scatter(actual, predicted, s=36, label=str(fold["held_out"]))
        actual_all.extend(actual)
        predicted_all.extend(predicted)
    if actual_all:
        low = min(actual_all + predicted_all)
        high = max(actual_all + predicted_all)
        pad = max(0.1, 0.05 * (high - low if high > low else 1))
        axis.plot([low - pad, high + pad], [low - pad, high + pad], color="#444444", linestyle="--", linewidth=1, label="Predicted = actual")
    axis.set_xlabel("Actual slack (ns)")
    axis.set_ylabel("Predicted slack (ns)")
    axis.set_title("Endpoint slack, full model")
    axis.legend()
    axis.grid(alpha=0.3)
    return _save(figure, path)


def _grouped_bars(axis, names: list[str], left: list[object], right: list[object], ylabel: str, title: str) -> None:
    positions = range(len(names))
    width = 0.36
    axis.bar([item - width / 2 for item in positions], [_or_nan(value) for value in left], width=width, label="Endpoint", color=_ENDPOINT)
    axis.bar([item + width / 2 for item in positions], [_or_nan(value) for value in right], width=width, label="Expanded", color=_EXPANDED)
    axis.set_xticks(list(positions), names)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Held-out design")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)


def _or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _or_zero(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _read(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
