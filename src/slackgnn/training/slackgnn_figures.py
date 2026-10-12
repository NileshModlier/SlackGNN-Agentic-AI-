"""Plots for SlackGNN embeddings, attention, ROC, and slack error."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from slackgnn.training.metrics import roc_curve


def write_slackgnn_figures(
    embedding_rows: list[dict[str, object]],
    explanations: list[dict[str, object]],
    slackgnn_classification: list[dict[str, object]],
    slackgnn_regression: list[dict[str, object]],
    baseline_classification: dict[str, dict[str, object]],
    figure_dir: Path,
    explain_dir: Path,
) -> None:
    """Write the M8 figures under ``reports/figures`` and the attention heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)
    if embedding_rows:
        matrix = torch.tensor([row["embedding"] for row in embedding_rows], dtype=torch.float32)
        labels = [str(row["design"]) for row in embedding_rows]
        _scatter(plt, _pca(matrix), labels, figure_dir / "embedding_pca.png", "PCA of SlackGNN embeddings")
        _scatter(plt, _tsne(matrix), labels, figure_dir / "embedding_tsne.png", "t-SNE of SlackGNN embeddings")
    for explanation in explanations:
        heatmap = explanation["attention_heatmap"]
        if heatmap["nodes"]:
            _heatmap(
                plt,
                heatmap,
                figure_dir / f"attention_heatmap_{explanation['design_name']}.png",
                explain_dir / f"attention_heatmap_{explanation['design_name']}.png",
            )
    _roc(plt, slackgnn_classification, baseline_classification, figure_dir / "roc_comparison.png")
    _scatter_slack(plt, slackgnn_regression, figure_dir / "slack_prediction_scatter.png")
    _error_histogram(plt, slackgnn_regression, figure_dir / "slack_error_distribution.png")


def _pca(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(0) < 2:
        return torch.zeros(matrix.size(0), 2)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    rank = min(2, centered.size(0), centered.size(1))
    _u, _s, vectors = torch.pca_lowrank(centered, q=rank)
    projected = centered @ vectors[:, :rank]
    if rank == 1:
        projected = torch.cat([projected, torch.zeros(projected.size(0), 1)], dim=1)
    return projected


def _tsne(matrix: torch.Tensor, perplexity: float = 12.0, steps: int = 200) -> torch.Tensor:
    count = matrix.size(0)
    if count < 5:
        return _pca(matrix)
    perplexity = min(perplexity, (count - 1) / 3)
    distances = torch.cdist(matrix, matrix, p=2).pow(2)
    affinities = _binary_affinities(distances, perplexity)
    joint = affinities + affinities.T
    joint = joint / joint.sum().clamp_min(1e-12)
    joint.fill_diagonal_(0)
    embedded = _pca(matrix).clone() * 0.01
    for step in range(steps):
        pairwise = torch.cdist(embedded, embedded, p=2).pow(2)
        student = 1.0 / (1.0 + pairwise)
        student.fill_diagonal_(0)
        match = student / student.sum().clamp_min(1e-12)
        match = match.clamp_min(1e-12)
        stiffness = (joint - match) * student
        delta = embedded.unsqueeze(1) - embedded.unsqueeze(0)
        gradient = 4 * (stiffness.unsqueeze(2) * delta).sum(dim=1)
        rate = 80.0 if step < 100 else 40.0
        embedded = embedded - rate * gradient
        embedded = embedded - embedded.mean(dim=0, keepdim=True)
    return embedded


def _binary_affinities(distances: torch.Tensor, perplexity: float) -> torch.Tensor:
    target = math.log(perplexity)
    affinities = torch.zeros_like(distances)
    for row_index in range(distances.size(0)):
        row = distances[row_index].clone()
        row[row_index] = float("inf")
        beta = 1.0
        lower = None
        upper = None
        chosen = torch.zeros_like(row)
        for _ in range(40):
            raw = torch.exp(-row * beta)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0)
            chosen = raw / raw.sum().clamp_min(1e-12)
            entropy = float((-(chosen * (chosen + 1e-12).log()).sum()).item())
            if abs(entropy - target) < 1e-3:
                break
            if entropy > target:
                lower = beta
                beta = beta * 2 if upper is None else 0.5 * (beta + upper)
            else:
                upper = beta
                beta = beta / 2 if lower is None else 0.5 * (beta + lower)
        affinities[row_index] = chosen
        affinities[row_index, row_index] = 0
    return affinities


def _scatter(plt, coords: torch.Tensor, labels: list[str], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = {"aes": "#4C78A8", "gcd": "#F58518", "ibex": "#54A24B", "jpeg": "#E45756"}
    for name in sorted(set(labels)):
        mask = [index for index, label in enumerate(labels) if label == name]
        picked = coords[mask]
        axis.scatter(picked[:, 0].tolist(), picked[:, 1].tolist(), s=18, label=name, color=colors.get(name, "#777777"))
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _heatmap(plt, heatmap: dict[str, object], *paths: Path) -> None:
    matrix = heatmap["matrix"]
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="magma")
    axis.set_title("Attention on reported-path nodes")
    axis.set_xlabel("Load")
    axis.set_ylabel("Driver")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    for path in paths:
        figure.savefig(path, dpi=120)
    plt.close(figure)


def _roc(plt, slackgnn_rows: list[dict[str, object]], baselines: dict[str, dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--")
    series = [("slackgnn", slackgnn_rows)]
    for name, result in baselines.items():
        series.append((name, result.get("predictions", [])))
    colors = {"slackgnn": "#E45756", "gcn": "#4C78A8", "graphsage": "#54A24B", "gin": "#F58518"}
    for name, rows in series:
        if not rows:
            continue
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        fpr, tpr = roc_curve(labels, scores)
        axis.plot(fpr, tpr, label=name, color=colors.get(name))
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Leave-one-design-out ROC comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _scatter_slack(plt, rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    actual = [float(row["slack"]) for row in rows]
    predicted = [float(row["prediction"]) for row in rows]
    axis.scatter(actual, predicted, color="#4C78A8")
    if actual:
        low = min(actual + predicted)
        high = max(actual + predicted)
        axis.plot([low, high], [low, high], color="#bbbbbb", linestyle="--")
    axis.set_xlabel("Actual slack (ns)")
    axis.set_ylabel("Predicted slack (ns)")
    axis.set_title("SlackGNN slack prediction")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _error_histogram(plt, rows: list[dict[str, object]], path: Path) -> None:
    errors = [float(row["prediction"]) - float(row["slack"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6, 4))
    if errors:
        axis.hist(errors, bins=min(8, len(errors)), color="#F58518")
    axis.set_xlabel("Predicted slack minus actual slack (ns)")
    axis.set_ylabel("Labeled endpoints")
    axis.set_title("Slack error distribution")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
