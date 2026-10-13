"""Cross-design leave-one-design-out comparison on expanded labels.

Baselines keep the M7 heads and the published one-hot node features.
SlackGNN keeps the M8 multitask model. Training loss for every model uses
the auxiliary expanded labels. Scores are reported on published endpoints
and on the expanded targets separately. This module does not rewrite
``graph.pt`` or the M9.3 label files.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.m9.metrics_agg import aggregate_rows
from slackgnn.m9.reporting import fmt, json_ready, metric_cells, write_json, write_text
from slackgnn.m9.trainer import expanded_leave_one_design_out, train_lodo
from slackgnn.paths import repo_root
from slackgnn.training.metrics import classification_metrics, regression_metrics, roc_curve
from slackgnn.training.models import MODEL_NAMES, build_model

logger = logging.getLogger(__name__)

ARCHITECTURES = MODEL_NAMES + ("slackgnn",)
_COLORS = {
    "gcn": "#1f4e79",
    "graphsage": "#2f6f4e",
    "gin": "#8c2f39",
    "slackgnn": "#c47b2b",
}
Pair = tuple[int, int]


def run_generalization(
    graphs: list[Data],
    inventory_payload: dict[str, object],
    *,
    seed: int = 0,
    epochs: int = 20,
    hidden_channels: int = 32,
    patience: int = 6,
    learning_rate: float = 1e-2,
) -> dict[str, object]:
    """Train each architecture with every design held out once."""
    if not graphs:
        raise ValueError("No graphs were loaded for the generalization study")
    started = time.perf_counter()
    models: dict[str, dict[str, object]] = {}
    for name in MODEL_NAMES:
        logger.info("Baseline leave-one-design-out for %s", name)
        models[name] = _train_baseline(
            graphs,
            name,
            seed=seed,
            epochs=epochs,
            hidden_channels=hidden_channels,
            patience=patience,
            learning_rate=learning_rate,
        )
    logger.info("SlackGNN leave-one-design-out")
    slack = train_lodo(
        graphs,
        seed=seed,
        task="multitask",
        epochs=epochs,
        hidden_channels=hidden_channels,
        patience=patience,
        learning_rate=learning_rate,
    )
    models["slackgnn"] = {
        "folds": [
            {
                "held_out": fold["held_out"],
                "epochs_ran": fold["epochs_ran"],
                "train_support": fold["train_support"],
                "endpoint": fold["endpoint"],
                "expanded": fold["expanded"],
                "endpoint_predictions": fold["endpoint_predictions"],
            }
            for fold in slack["folds"]
        ],
        "summary": slack["summary"],
    }
    ranking = rank_generalization(models)
    return {
        "protocol": "leave_one_design_out",
        "training_loss": "expanded labels",
        "evaluation": [
            "published endpoints",
            "expanded route labels",
        ],
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(inventory_payload.get("missing", [])),
        "seed": seed,
        "epochs": epochs,
        "hidden_channels": hidden_channels,
        "patience": patience,
        "learning_rate": learning_rate,
        "models": models,
        "ranking": ranking,
        "runtime_seconds": time.perf_counter() - started,
        "note": (
            "Baseline classification and regression are separate M7 heads trained on expanded labels. "
            "SlackGNN is one multitask model. Expanded scores reuse slack copied along each reported path. "
            "Endpoint scores are the published path-table measurements."
        ),
    }


def rank_generalization(models: dict[str, dict[str, object]]) -> dict[str, object]:
    """Name the easiest holdout, the hardest holdout, and the strongest architecture."""
    design_scores: dict[str, list[float]] = {}
    for model in models.values():
        for fold in model["folds"]:
            value = _metric(fold["endpoint"], "f1")
            if value is None:
                continue
            design_scores.setdefault(str(fold["held_out"]), []).append(value)
    means = {
        design: sum(values) / len(values)
        for design, values in design_scores.items()
        if values
    }
    easiest = max(means, key=means.get) if means else None
    hardest = min(means, key=means.get) if means else None
    architecture_scores = {
        name: _metric(model["summary"]["endpoint"]["f1"], "mean")
        for name, model in models.items()
    }
    ranked = {name: score for name, score in architecture_scores.items() if score is not None}
    best = max(ranked, key=ranked.get) if ranked else None
    slack_f1 = architecture_scores.get("slackgnn")
    baseline_f1 = [score for name, score in architecture_scores.items() if name != "slackgnn" and score is not None]
    slack_mae = _metric(models["slackgnn"]["summary"]["endpoint"]["mae"], "mean") if "slackgnn" in models else None
    baseline_mae = [
        _metric(model["summary"]["endpoint"]["mae"], "mean")
        for name, model in models.items()
        if name != "slackgnn"
    ]
    baseline_mae = [value for value in baseline_mae if value is not None]
    return {
        "easiest_design": easiest,
        "easiest_mean_endpoint_f1": means.get(easiest) if easiest else None,
        "hardest_design": hardest,
        "hardest_mean_endpoint_f1": means.get(hardest) if hardest else None,
        "design_mean_endpoint_f1": means,
        "most_transferable_architecture": best,
        "architecture_endpoint_f1": architecture_scores,
        "slackgnn_endpoint_f1_above_every_baseline": bool(
            slack_f1 is not None and baseline_f1 and slack_f1 > max(baseline_f1)
        ),
        "slackgnn_endpoint_mae_below_every_baseline": bool(
            slack_mae is not None and baseline_mae and slack_mae < min(baseline_mae)
        ),
    }


def write_generalization_report(payload: dict[str, object], directory: str | Path | None = None) -> tuple[Path, Path]:
    """Write the JSON and Markdown generalization reports."""
    destination = Path(directory) if directory is not None else repo_root() / "reports"
    json_path = write_json(destination / "generalization_results.json", _public_payload(payload))
    markdown_path = write_text(destination / "generalization_results.md", render_generalization_markdown(payload))
    return json_path, markdown_path


def write_generalization_figures(payload: dict[str, object], directory: str | Path | None = None) -> list[Path]:
    """Write holdout comparison plots next to the report."""
    figure_dir = Path(directory) if directory is not None else repo_root() / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        _grouped_bars(payload, "f1", "Endpoint F1", figure_dir / "generalization_f1.png"),
        _grouped_bars(payload, "mae", "Endpoint MAE (ns)", figure_dir / "generalization_mae.png"),
        _roc(payload, figure_dir / "generalization_roc.png"),
    ]


def render_generalization_markdown(payload: dict[str, object]) -> str:
    """Format the holdout tables and the architecture ranking."""
    ranking = payload["ranking"]
    lines = [
        "# Cross-design generalization",
        "",
        "Each row holds out one design. Training uses the other designs and the expanded route labels. "
        "Endpoint metrics use the published path-table slack. Expanded metrics use the copied route slack.",
        "",
        payload["note"],
        "",
        f"Designs: {', '.join(payload['designs'])}.",
        f"Seed {payload['seed']}, hidden width {payload['hidden_channels']}, epoch cap {payload['epochs']}.",
        "",
        "## Endpoint means",
        "",
        _mean_table(payload, "endpoint"),
        "",
        "## Expanded-label means",
        "",
        _mean_table(payload, "expanded"),
        "",
    ]
    for name in ARCHITECTURES:
        model = payload["models"][name]
        lines.extend(
            [
                f"## {name}",
                "",
                "### Published endpoints",
                "",
                _fold_table(model["folds"], "endpoint"),
                "",
                "### Expanded labels",
                "",
                _fold_table(model["folds"], "expanded"),
                "",
            ]
        )
    lines.extend(
        [
            "## Ranking",
            "",
            f"Easiest holdout by mean endpoint F1: {ranking['easiest_design']} "
            f"({fmt(ranking['easiest_mean_endpoint_f1'])}).",
            f"Hardest holdout by mean endpoint F1: {ranking['hardest_design']} "
            f"({fmt(ranking['hardest_mean_endpoint_f1'])}).",
            f"Most transferable architecture by mean endpoint F1: {ranking['most_transferable_architecture']} "
            f"({fmt(ranking['architecture_endpoint_f1'].get(ranking['most_transferable_architecture']))}).",
            (
                "SlackGNN endpoint F1 is above every baseline, and its endpoint MAE is below every baseline."
                if ranking["slackgnn_endpoint_f1_above_every_baseline"] and ranking["slackgnn_endpoint_mae_below_every_baseline"]
                else "SlackGNN does not lead every baseline on both endpoint F1 and endpoint MAE in this run."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _train_baseline(
    graphs: list[Data],
    model_name: str,
    *,
    seed: int,
    epochs: int,
    hidden_channels: int,
    patience: int,
    learning_rate: float,
) -> dict[str, object]:
    folds = expanded_leave_one_design_out(graphs, seed)
    rows: list[dict[str, object]] = []
    for fold in folds:
        logger.info("Training %s with %s held out", model_name, fold["held_out"])
        classifier = _fit_baseline(
            graphs,
            model_name,
            fold["train"],
            fold["validation"],
            "classification",
            epochs,
            hidden_channels,
            patience,
            learning_rate,
            seed,
        )
        regressor = _fit_baseline(
            graphs,
            model_name,
            fold["train"],
            fold["validation"],
            "regression",
            epochs,
            hidden_channels,
            patience,
            learning_rate,
            seed,
        )
        endpoint = _merged_metrics(
            _baseline_rows(graphs, classifier, fold["test_endpoint"], "classification", "endpoint"),
            _baseline_rows(graphs, regressor, fold["test_endpoint"], "regression", "endpoint"),
        )
        expanded = _merged_metrics(
            _baseline_rows(graphs, classifier, fold["test_expanded"], "classification", "expanded"),
            _baseline_rows(graphs, regressor, fold["test_expanded"], "regression", "expanded"),
        )
        rows.append(
            {
                "held_out": fold["held_out"],
                "train_support": len(fold["train"]),
                "endpoint": endpoint["metrics"],
                "expanded": expanded["metrics"],
                "endpoint_predictions": endpoint["predictions"],
            }
        )
        del classifier, regressor
    return {
        "folds": rows,
        "summary": {
            "endpoint": aggregate_rows([row["endpoint"] for row in rows]),
            "expanded": aggregate_rows([row["expanded"] for row in rows]),
        },
    }


def _fit_baseline(
    graphs: list[Data],
    model_name: str,
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    task: str,
    epochs: int,
    hidden_channels: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    random.seed(seed)
    model = build_model(model_name, int(graphs[0].x.size(1)), hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = _positive_weight(graphs, train_pairs) if task == "classification" else None
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    waited = 0
    for _epoch in range(epochs):
        model.train()
        _baseline_loss(model, graphs, train_pairs, task, optimizer, pos_weight)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            continue
        model.eval()
        with torch.no_grad():
            loss = _baseline_loss(model, graphs, validation_pairs, task, None, pos_weight)
        if loss < best_loss:
            best_loss = loss
            best_state = snapshot
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _baseline_loss(
    model: nn.Module,
    graphs: list[Data],
    pairs: list[Pair],
    task: str,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
) -> float:
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    total = 0.0
    count = 0
    for graph_index, nodes in grouped.items():
        data = graphs[graph_index]
        prediction = model(data.x, data.edge_index)
        index = torch.tensor(nodes, dtype=torch.long)
        if task == "classification":
            loss = nn.functional.binary_cross_entropy_with_logits(
                prediction[index],
                data.expanded_violation[index],
                pos_weight=pos_weight,
            )
        else:
            loss = nn.functional.mse_loss(prediction[index], data.expanded_slack[index])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(nodes)
        count += len(nodes)
    return total / count if count else 0.0


def _baseline_rows(
    graphs: list[Data],
    model: nn.Module,
    pairs: list[Pair],
    task: str,
    scope: str,
) -> list[dict[str, float]]:
    model.eval()
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    cached: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            cached[graph_index] = model(data.x, data.edge_index)
    rows: list[dict[str, float]] = []
    for graph_index, node_index in pairs:
        data = graphs[graph_index]
        raw = float(cached[graph_index][node_index].item())
        if scope == "endpoint":
            label = int(data.y_violation[node_index].item())
            slack = float(data.y_slack[node_index].item())
        else:
            label = int(data.expanded_violation[node_index].item())
            slack = float(data.expanded_slack[node_index].item())
        if task == "classification":
            rows.append({"label": label, "score": float(torch.sigmoid(torch.tensor(raw)).item())})
        else:
            rows.append({"slack": slack, "prediction": raw})
    return rows


def _merged_metrics(
    classification_rows: list[dict[str, float]],
    regression_rows: list[dict[str, float]],
) -> dict[str, object]:
    if not classification_rows:
        empty = {key: None for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "mae", "mse", "rmse")}
        empty["support"] = 0
        return {"metrics": empty, "predictions": {"label": [], "score": [], "slack": [], "prediction": []}}
    classification = classification_metrics(
        [int(row["label"]) for row in classification_rows],
        [float(row["score"]) for row in classification_rows],
    )
    regression = regression_metrics(
        [float(row["slack"]) for row in regression_rows],
        [float(row["prediction"]) for row in regression_rows],
    )
    metrics = {
        "accuracy": classification["accuracy"],
        "precision": classification["precision"],
        "recall": classification["recall"],
        "f1": classification["f1"],
        "roc_auc": classification["roc_auc"],
        "mae": regression["mae"],
        "mse": regression["mse"],
        "rmse": regression["rmse"],
        "support": classification["support"],
    }
    predictions = {
        "label": [int(row["label"]) for row in classification_rows],
        "score": [float(row["score"]) for row in classification_rows],
        "slack": [float(row["slack"]) for row in regression_rows],
        "prediction": [float(row["prediction"]) for row in regression_rows],
    }
    return {"metrics": metrics, "predictions": predictions}


def _positive_weight(graphs: list[Data], pairs: list[Pair]) -> torch.Tensor:
    violations = sum(int(graphs[graph_index].expanded_violation[node_index].item()) for graph_index, node_index in pairs)
    met = len(pairs) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _metric(row: object, key: str) -> float | None:
    if not isinstance(row, dict) or row.get(key) is None:
        return None
    value = float(row[key])
    if value != value:
        return None
    return value


def _public_payload(payload: dict[str, object]) -> dict[str, object]:
    """Drop per-node prediction lists from the written JSON."""
    ready = json_ready(payload)
    if not isinstance(ready, dict):
        return {}
    for model in ready.get("models", {}).values():
        for fold in model.get("folds", []):
            fold.pop("endpoint_predictions", None)
    return ready


def _mean_table(payload: dict[str, object], scope: str) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ARCHITECTURES:
        summary = payload["models"][name]["summary"][scope]
        cells = metric_cells(summary)[:-1]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def _fold_table(folds: list[dict[str, object]], scope: str) -> str:
    lines = [
        "| Held-out | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in folds:
        cells = metric_cells(fold[scope])
        lines.append("| " + " | ".join([str(fold["held_out"]), *cells]) + " |")
    return "\n".join(lines)


def _held_out_names(payload: dict[str, object]) -> list[str]:
    return [str(fold["held_out"]) for fold in payload["models"]["gcn"]["folds"]]


def _grouped_bars(payload: dict[str, object], key: str, ylabel: str, path: Path) -> Path:
    designs = _held_out_names(payload)
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    width = 0.18
    positions = list(range(len(designs)))
    for offset, name in enumerate(ARCHITECTURES):
        values = []
        for design in designs:
            fold = next(item for item in payload["models"][name]["folds"] if item["held_out"] == design)
            value = fold["endpoint"].get(key)
            values.append(float("nan") if value is None else float(value))
        axis.bar(
            [item + (offset - 1.5) * width for item in positions],
            values,
            width=width,
            label=name,
            color=_COLORS[name],
        )
    axis.set_xticks(positions, designs)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Held-out design")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _roc(payload: dict[str, object], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1, label="Chance")
    for name in ARCHITECTURES:
        labels: list[int] = []
        scores: list[float] = []
        for fold in payload["models"][name]["folds"]:
            predictions = fold.get("endpoint_predictions") or {}
            labels.extend(int(value) for value in predictions.get("label", []))
            scores.extend(float(value) for value in predictions.get("score", []))
        if len(set(labels)) < 2:
            continue
        fpr, tpr = roc_curve(labels, scores)
        axis.plot(fpr, tpr, color=_COLORS[name], linewidth=2.2, label=name)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Pooled endpoint ROC")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
