"""Train baseline GNNs and write evaluation reports."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.paths import data_dir, repo_root
from slackgnn.training.dataset_builder import build_training_dataset, load_training_dataset
from slackgnn.training.metrics import classification_metrics, mean_metric, regression_metrics
from slackgnn.training.models import MODEL_NAMES, build_model

SLACKGNN_NAME = "slackgnn"
TRAINABLE_MODELS = MODEL_NAMES + (SLACKGNN_NAME,)
from slackgnn.training.splits import leave_one_design_out, random_split
from slackgnn.training.validate_corpus import reports_dir, write_validation_report, validate_corpus
from slackgnn.training.visualize import write_figures

logger = logging.getLogger(__name__)

Pair = tuple[int, int]


def train_baselines(
    graphs: list[Data],
    models: tuple[str, ...] = MODEL_NAMES,
    epochs: int = 30,
    hidden_channels: int = 32,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> dict[str, object]:
    """Train each requested model under a random split and leave-one-design-out."""
    torch.manual_seed(seed)
    random.seed(seed)
    classification = {
        "protocol": ["random_split", "leave_one_design_out"],
        "label_definition": {"0": "timing met", "1": "timing violation"},
        "random_split": {},
        "leave_one_design_out": {},
    }
    regression = {"protocol": ["random_split", "leave_one_design_out"], "random_split": {}, "leave_one_design_out": {}}
    split = random_split(graphs, seed)
    folds = leave_one_design_out(graphs, seed)
    for name in models:
        logger.info("Training %s classification on the random split", name)
        classification["random_split"][name] = _fit_split(graphs, name, split, "classification", epochs, hidden_channels, learning_rate, seed)
        classification["leave_one_design_out"][name] = _fit_folds(graphs, name, folds, "classification", epochs, hidden_channels, learning_rate, seed)
        logger.info("Training %s regression on the random split", name)
        regression["random_split"][name] = _fit_split(graphs, name, split, "regression", epochs, hidden_channels, learning_rate, seed)
        regression["leave_one_design_out"][name] = _fit_folds(graphs, name, folds, "regression", epochs, hidden_channels, learning_rate, seed)
    return {"classification": classification, "regression": regression}


def run_lodo(
    graphs: list[Data],
    models: tuple[str, ...] = MODEL_NAMES,
    epochs: int = 30,
    hidden_channels: int = 32,
    learning_rate: float = 1e-2,
    seed: int = 0,
    tasks: tuple[str, ...] = ("classification", "regression"),
) -> dict[str, dict[str, object]]:
    """Train each baseline with leave-one-design-out and keep the test predictions."""
    torch.manual_seed(seed)
    random.seed(seed)
    folds = leave_one_design_out(graphs, seed)
    classification: dict[str, object] = {}
    regression: dict[str, object] = {}
    for name in models:
        if "classification" in tasks:
            logger.info("Baseline %s classification leave-one-design-out", name)
            classification[name] = _fit_folds(
                graphs, name, folds, "classification", epochs, hidden_channels, learning_rate, seed
            )
        if "regression" in tasks:
            logger.info("Baseline %s regression leave-one-design-out", name)
            regression[name] = _fit_folds(
                graphs, name, folds, "regression", epochs, hidden_channels, learning_rate, seed
            )
    return {"classification": classification, "regression": regression}


def write_results(payload: dict[str, object], destination: Path | None = None) -> tuple[Path, Path, Path]:
    """Write classification JSON, classification Markdown, and regression JSON."""
    root = reports_dir()
    classification_path = destination or (root / "model_results.json")
    markdown_path = root / "model_results.md"
    regression_path = root / "slack_regression_results.json"
    classification_path.parent.mkdir(parents=True, exist_ok=True)
    classification_path.write_text(
        json.dumps(_without_predictions(payload["classification"]), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(_render_markdown(payload["classification"]), encoding="utf-8", newline="\n")
    regression_path.write_text(
        json.dumps(_without_predictions(payload["regression"]), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return classification_path, markdown_path, regression_path


def write_next_phase(summary: dict[str, object], payload: dict[str, object]) -> Path:
    """Write the architecture note from the measured baseline."""
    path = repo_root() / "docs" / "slackgnn_next_phase.md"
    path.write_text(_next_phase_text(summary, payload), encoding="utf-8", newline="\n")
    logger.info("Wrote %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    """Validate, build the training set, train the requested models, and report."""
    parser = argparse.ArgumentParser(description="Train SlackGNN baseline GNNs.")
    parser.add_argument("--model", choices=TRAINABLE_MODELS, action="append", default=None)
    parser.add_argument("--task", choices=("classification", "regression", "multitask"), default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=data_dir())
    args = parser.parse_args(argv)
    selected = tuple(dict.fromkeys(args.model)) if args.model else MODEL_NAMES
    models = tuple(name for name in selected if name != SLACKGNN_NAME)
    validation = validate_corpus(args.data_root)
    write_validation_report(validation)
    if not validation["passed"]:
        logger.error("Corpus validation failed")
        return 1
    summary, errors = build_training_dataset(args.data_root)
    if errors or summary is None:
        for message in errors:
            logger.error("%s", message)
        return 1
    graphs, _stored, load_errors = load_training_dataset(reports_dir() / "training_data.pt")
    if load_errors or graphs is None:
        for message in load_errors:
            logger.error("%s", message)
        return 1
    if models:
        payload = train_baselines(graphs, models, args.epochs, args.hidden, args.lr, args.seed)
        classification_path, markdown_path, regression_path = write_results(payload)
        figure_dir = write_figures(summary, payload, reports_dir() / "figures")
        document = write_next_phase(summary, payload)
        print(f"Wrote {classification_path}")
        print(f"Wrote {markdown_path}")
        print(f"Wrote {regression_path}")
        print(f"Wrote figures in {figure_dir}")
        print(f"Wrote {document}")
    if SLACKGNN_NAME in selected:
        from slackgnn.training.slackgnn_runner import TASKS, run_slackgnn

        tasks = (args.task,) if args.task else TASKS
        return run_slackgnn(
            args.data_root,
            epochs=args.epochs,
            hidden_channels=args.hidden,
            learning_rate=args.lr,
            seed=args.seed,
            tasks=tasks,
            alpha=args.alpha,
            beta=args.beta,
        )
    return 0


def _fit_split(
    graphs: list[Data],
    model_name: str,
    split: dict[str, list[Pair]],
    task: str,
    epochs: int,
    hidden_channels: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    trained = _train_one(graphs, model_name, split["train"], split["validation"], task, epochs, hidden_channels, learning_rate, seed)
    return {
        "train_labels": len(split["train"]),
        "validation_labels": len(split["validation"]),
        "test_labels": len(split["test"]),
        "test": _evaluate(graphs, trained, split["test"], task),
        "predictions": _prediction_rows(graphs, trained, split["test"], task),
    }


def _fit_folds(
    graphs: list[Data],
    model_name: str,
    folds: list[dict[str, object]],
    task: str,
    epochs: int,
    hidden_channels: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for fold in folds:
        logger.info("Training %s %s with %s held out", model_name, task, fold["held_out"])
        trained = _train_one(
            graphs,
            model_name,
            fold["train"],
            fold["validation"],
            task,
            epochs,
            hidden_channels,
            learning_rate,
            seed,
        )
        score = _evaluate(graphs, trained, fold["test"], task)
        score["held_out"] = fold["held_out"]
        rows.append(score)
        predictions.extend(_prediction_rows(graphs, trained, fold["test"], task))
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc") if task == "classification" else ("mae", "mse", "rmse")
    return {
        "folds": rows,
        "mean": {name: mean_metric(rows, name) for name in metric_names},
        "predictions": predictions,
    }


def _train_one(
    graphs: list[Data],
    model_name: str,
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    task: str,
    epochs: int,
    hidden_channels: int,
    learning_rate: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = build_model(model_name, int(graphs[0].x.size(1)), hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = _positive_weight(graphs, train_pairs) if task == "classification" else None
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    patience = 0
    for _epoch in range(epochs):
        model.train()
        _epoch_loss(model, graphs, train_pairs, task, optimizer, pos_weight)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            continue
        model.eval()
        with torch.no_grad():
            loss = _epoch_loss(model, graphs, validation_pairs, task, None, pos_weight)
        if loss < best_loss:
            best_loss = loss
            best_state = snapshot
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _epoch_loss(
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
                data.y_violation[index],
                pos_weight=pos_weight,
            )
        else:
            loss = nn.functional.mse_loss(prediction[index], data.y_slack[index])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(nodes)
        count += len(nodes)
    return total / count if count else 0.0


def _evaluate(graphs: list[Data], model: nn.Module, pairs: list[Pair], task: str) -> dict[str, object]:
    rows = _prediction_rows(graphs, model, pairs, task)
    if task == "classification":
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        return classification_metrics(labels, scores)
    actual = [float(row["slack"]) for row in rows]
    predicted = [float(row["prediction"]) for row in rows]
    return regression_metrics(actual, predicted)


def _prediction_rows(graphs: list[Data], model: nn.Module, pairs: list[Pair], task: str) -> list[dict[str, object]]:
    model.eval()
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    cached: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            cached[graph_index] = model(data.x, data.edge_index)
    rows: list[dict[str, object]] = []
    for graph_index, node_index in pairs:
        score = float(cached[graph_index][node_index].item())
        data = graphs[graph_index]
        row: dict[str, object] = {
            "design": str(data.design_name),
            "node": node_index,
            "slack": float(data.y_slack[node_index].item()),
        }
        if task == "classification":
            row["label"] = int(data.y_violation[node_index].item())
            row["score"] = float(torch.sigmoid(torch.tensor(score)).item())
            row["logit"] = score
        else:
            row["prediction"] = score
        rows.append(row)
    return rows


def _positive_weight(graphs: list[Data], pairs: list[Pair]) -> torch.Tensor:
    violations = sum(int(graphs[graph_index].y_violation[node_index].item()) for graph_index, node_index in pairs)
    met = len(pairs) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _render_markdown(classification: dict[str, object]) -> str:
    lines = [
        "# Baseline timing-violation results",
        "",
        "Class 0 is a labeled endpoint whose slack meets timing. Class 1 is a timing violation.",
        "Leave-one-design-out trains on every other design and tests the held-out design.",
        "",
        "## Leave-one-design-out mean",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lodo = classification["leave_one_design_out"]
    for name, result in lodo.items():
        mean = result["mean"]
        lines.append(
            "| {name} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} |".format(
                name=name,
                accuracy=_fmt(mean["accuracy"]),
                precision=_fmt(mean["precision"]),
                recall=_fmt(mean["recall"]),
                f1=_fmt(mean["f1"]),
                roc_auc=_fmt(mean["roc_auc"]),
            )
        )
    lines.extend(["", "A mean omits a fold when that metric is undefined. Precision and F1 are undefined when a model predicts no timing violations.", "", "## Leave-one-design-out folds", ""])
    for name, result in lodo.items():
        lines.extend([f"### {name}", "", "| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for fold in result["folds"]:
            lines.append(
                "| {held_out} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {support} |".format(
                    held_out=fold["held_out"],
                    accuracy=_fmt(fold["accuracy"]),
                    precision=_fmt(fold["precision"]),
                    recall=_fmt(fold["recall"]),
                    f1=_fmt(fold["f1"]),
                    roc_auc=_fmt(fold["roc_auc"]),
                    support=fold["support"],
                )
            )
        lines.append("")
    lines.extend(["## Random labeled-endpoint split", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, result in classification["random_split"].items():
        test = result["test"]
        lines.append(
            "| {name} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {support} |".format(
                name=name,
                accuracy=_fmt(test["accuracy"]),
                precision=_fmt(test["precision"]),
                recall=_fmt(test["recall"]),
                f1=_fmt(test["f1"]),
                roc_auc=_fmt(test["roc_auc"]),
                support=test["support"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _without_predictions(payload: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in payload.items():
        if key == "predictions":
            continue
        if isinstance(value, dict):
            cleaned[key] = _without_predictions(value)
        else:
            cleaned[key] = value
    return cleaned


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _next_phase_text(summary: dict[str, object], payload: dict[str, object]) -> str:
    classification = payload["classification"]["leave_one_design_out"]
    regression = payload["regression"]["leave_one_design_out"]
    rows = []
    for name, result in classification.items():
        mean = result["mean"]
        rows.append(f"| {name} | {_fmt(mean['f1'])} | {_fmt(mean['roc_auc'])} | {_fmt(regression[name]['mean']['mae'])} | {_fmt(regression[name]['mean']['rmse'])} |")
    design_rows = []
    for item in summary["designs_detail"]:
        design_rows.append(
            f"| {item['design_name']} | {item['node_count']} | {item['edge_count']} | {item['label_count']} | {item['met_count']} | {item['violation_count']} |"
        )
    return f"""# SlackGNN next phase

## Current corpus statistics

The training set is the published PyG corpus. Each graph is one placed-and-routed design. Node features are the four-way node-type encoding. Slack is the prediction target and is not an input feature.

| Design | Nodes | Edges | Labels | Timing met | Violations |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(design_rows)}

Totals: {summary['graph_count']} designs, {summary['total_nodes']} nodes, {summary['total_edges']} edges, {summary['label_count']} labeled endpoints, {summary['met_count']} timing-met labels, and {summary['violation_count']} timing-violation labels.

## Baseline model performance

GCN, GraphSAGE, and GIN were trained with masked node loss. Message passing uses the full graph. The loss uses only labeled endpoints. The table is the mean of leave-one-design-out folds. Slack error is in nanoseconds.

| Model | F1 | ROC-AUC | Slack MAE | Slack RMSE |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Fold-level numbers are in `reports/model_results.md`. A random split of the same 22 labels is reported beside the design holdout. The design holdout is the result that matters: the test design is absent from training.

## Weaknesses

The label set is 22 endpoints out of {summary['total_nodes']} nodes. OpenROAD's finish report prints a small set of paths, so most endpoints have no target. A model can look accurate by predicting the majority class, timing met, and still miss the few violations.

Each leave-one-design-out test fold has between four and seven labels. Precision, recall, F1, and ROC-AUC move sharply when one endpoint changes. These baselines are a measurement of the current corpus, not a claim that the architecture is ready for timing signoff.

Node features do not include pin capacitance, net parasitics, or path depth. Two nodes of the same type are identical at the input, so the network has to recover timing structure only from connectivity. One IBEX endpoint also carries both a minimum-path slack and a maximum-path slack; the graph stores the worse slack, so the other path is invisible to the loss.

## Proposed SlackGNN architecture

SlackGNN should stay a node model on the full netlist graph, with three additions over these baselines.

1. Edge features from the SPEF: resistance and capacitance on each driver-to-load edge, so delay has a physical input.
2. A timing head with two outputs: violation logit and slack. Train the violation head with a class-weighted loss and the slack head with an error that penalizes optimistic slack.
3. A path readout that pools the nodes on each labeled path, so the loss can see the path rather than only its endpoint.

The message-passing backbone can start from the GraphSAGE baseline, because its neighborhood aggregation is stable on these directed netlist graphs. Design identity should not be an input feature. Leave-one-design-out remains the evaluation protocol.

## Recommended next milestones

1. Expand labels beyond the printed worst paths, so each design has many more endpoints with slack.
2. Attach SPEF edge features without changing the current node-type encoding.
3. Implement the two-head SlackGNN and compare it with these three baselines under the same design holdout.
4. Report confidence intervals over seeds, because a single run on 22 labels is too small to rank architectures.
"""
