"""Plots for the corpus labels and the baseline comparison."""

from __future__ import annotations

import logging
from pathlib import Path

from slackgnn.training.metrics import roc_curve

logger = logging.getLogger(__name__)


def write_figures(summary: dict[str, object], payload: dict[str, object], destination: str | Path) -> Path:
    """Write the class, label, comparison, confusion, and ROC figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(destination)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _class_distribution(plt, summary, figure_dir / "class_distribution.png")
    _label_distribution(plt, summary, figure_dir / "label_distribution.png")
    _model_comparison(plt, payload, figure_dir / "model_comparison.png")
    classification = payload["classification"]["leave_one_design_out"]
    for name, result in classification.items():
        _confusion_matrix(plt, result["predictions"], figure_dir / f"confusion_matrix_{name}.png", name)
        _roc_curve(plt, result["predictions"], figure_dir / f"roc_curve_{name}.png", name)
    logger.info("Wrote figures in %s", figure_dir)
    return figure_dir


def _class_distribution(plt, summary: dict[str, object], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Timing met", "Timing violation"], [summary["met_count"], summary["violation_count"]], color=["#4C78A8", "#E45756"])
    axis.set_ylabel("Labeled endpoints")
    axis.set_title("Class distribution")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _label_distribution(plt, summary: dict[str, object], path: Path) -> None:
    details = summary["designs_detail"]
    names = [str(item["design_name"]) for item in details]
    met = [int(item["met_count"]) for item in details]
    violated = [int(item["violation_count"]) for item in details]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, met, label="Timing met", color="#4C78A8")
    axis.bar(names, violated, bottom=met, label="Timing violation", color="#E45756")
    axis.set_ylabel("Labeled endpoints")
    axis.set_title("Label distribution by design")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _model_comparison(plt, payload: dict[str, object], path: Path) -> None:
    classification = payload["classification"]["leave_one_design_out"]
    names = list(classification)
    f1 = [_or_zero(classification[name]["mean"]["f1"]) for name in names]
    auc = [_or_zero(classification[name]["mean"]["roc_auc"]) for name in names]
    positions = range(len(names))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([index - 0.18 for index in positions], f1, width=0.36, label="F1", color="#4C78A8")
    axis.bar([index + 0.18 for index in positions], auc, width=0.36, label="ROC-AUC", color="#F58518")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(names)
    axis.set_ylim(0, 1)
    axis.set_title("Leave-one-design-out model comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _confusion_matrix(plt, predictions: list[dict[str, object]], path: Path, model_name: str) -> None:
    labels = [int(row["label"]) for row in predictions]
    guessed = [1 if float(row["score"]) >= 0.5 else 0 for row in predictions]
    matrix = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, guessed, strict=True):
        matrix[label][prediction] += 1
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Met", "Violation"])
    axis.set_yticks([0, 1], ["Met", "Violation"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{model_name} confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _roc_curve(plt, predictions: list[dict[str, object]], path: Path, model_name: str) -> None:
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    fpr, tpr = roc_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, color="#4C78A8", label=model_name)
    axis.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"{model_name} ROC curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _or_zero(value: object) -> float:
    return 0.0 if value is None else float(value)
