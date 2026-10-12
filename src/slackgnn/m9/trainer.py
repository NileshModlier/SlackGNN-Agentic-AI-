"""Leave-one-design-out training on expanded labels."""

from __future__ import annotations

import logging
import random
import time

import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.m9.ablation import AblatedSlackGNN
from slackgnn.m9.metrics_agg import METRIC_KEYS, aggregate_rows, total_support
from slackgnn.training.metrics import classification_metrics, regression_metrics

logger = logging.getLogger(__name__)

Pair = tuple[int, int]
DEFAULT_HIDDEN = 32
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 6
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.5
DEFAULT_SEED = 0

RUNTIME_NOTE = (
    "Message passing runs on each full netlist. The loss uses only supervised nodes. "
    f"Hidden width is {DEFAULT_HIDDEN}, the epoch cap is {DEFAULT_EPOCHS}, and early stopping "
    f"patience is {DEFAULT_PATIENCE}. These limits keep the CPU study finite on the largest graph."
)


def train_lodo(
    graphs: list[Data],
    *,
    seed: int = DEFAULT_SEED,
    task: str = "multitask",
    drop_timing: bool = False,
    drop_edges: bool = False,
    uniform_attention: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    hidden_channels: int = DEFAULT_HIDDEN,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict[str, object]:
    """Train on expanded labels and score both endpoint and expanded targets."""
    folds = expanded_leave_one_design_out(graphs, seed)
    fold_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for fold in folds:
        logger.info("Training %s with %s held out (seed %s)", task, fold["held_out"], seed)
        model, epochs_ran = _train_one(
            graphs,
            fold["train"],
            fold["validation"],
            task=task,
            drop_timing=drop_timing,
            drop_edges=drop_edges,
            uniform_attention=uniform_attention,
            epochs=epochs,
            hidden_channels=hidden_channels,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )
        endpoint, endpoint_predictions = _evaluate(graphs, model, fold["test_endpoint"], "endpoint")
        expanded, expanded_predictions = _evaluate(graphs, model, fold["test_expanded"], "expanded")
        fold_rows.append(
            {
                "held_out": fold["held_out"],
                "epochs_ran": epochs_ran,
                "train_support": len(fold["train"]),
                "validation_support": len(fold["validation"]),
                "endpoint": endpoint,
                "expanded": expanded,
                "endpoint_predictions": endpoint_predictions,
                "expanded_predictions": expanded_predictions,
            }
        )
        del model
    return {
        "folds": fold_rows,
        "summary": {
            "endpoint": aggregate_rows([row["endpoint"] for row in fold_rows]),
            "expanded": aggregate_rows([row["expanded"] for row in fold_rows]),
        },
        "endpoint_support": total_support([row["endpoint"] for row in fold_rows]),
        "expanded_support": total_support([row["expanded"] for row in fold_rows]),
        "runtime_seconds": time.perf_counter() - started,
        "seed": seed,
        "task": task,
        "drop_timing": drop_timing,
        "drop_edges": drop_edges,
        "uniform_attention": uniform_attention,
    }


def train_holdout(
    graphs: list[Data],
    held_out: str,
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    hidden_channels: int = DEFAULT_HIDDEN,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict[str, object]:
    """Train on every supplied design except ``held_out`` and score that design once."""
    subset = [graph for graph in graphs if str(graph.design_name) != held_out]
    target = next(graph for graph in graphs if str(graph.design_name) == held_out)
    ordered = subset + [target]
    fold = next(item for item in expanded_leave_one_design_out(ordered, seed) if item["held_out"] == held_out)
    started = time.perf_counter()
    model, epochs_ran = _train_one(
        ordered,
        fold["train"],
        fold["validation"],
        task="multitask",
        drop_timing=False,
        drop_edges=False,
        uniform_attention=False,
        epochs=epochs,
        hidden_channels=hidden_channels,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
    endpoint, endpoint_predictions = _evaluate(ordered, model, fold["test_endpoint"], "endpoint")
    expanded, expanded_predictions = _evaluate(ordered, model, fold["test_expanded"], "expanded")
    del model
    return {
        "held_out": held_out,
        "training_designs": [str(graph.design_name) for graph in subset],
        "epochs_ran": epochs_ran,
        "train_support": len(fold["train"]),
        "validation_support": len(fold["validation"]),
        "endpoint": endpoint,
        "expanded": expanded,
        "endpoint_predictions": endpoint_predictions,
        "expanded_predictions": expanded_predictions,
        "runtime_seconds": time.perf_counter() - started,
    }


def expanded_leave_one_design_out(graphs: list[Data], seed: int) -> list[dict[str, object]]:
    """Rotate each design into the test set. Validation keeps one expanded label per class."""
    folds: list[dict[str, object]] = []
    for held_out, _graph in enumerate(graphs):
        rng = random.Random(seed + held_out)
        remaining = [pair for pair in _expanded_pairs(graphs) if pair[0] != held_out]
        grouped: dict[int, list[Pair]] = {0: [], 1: []}
        for pair in remaining:
            grouped[_expanded_class(graphs, pair)].append(pair)
        train: list[Pair] = []
        validation: list[Pair] = []
        for items in grouped.values():
            rng.shuffle(items)
            if len(items) <= 1:
                train.extend(items)
                continue
            validation.append(items[0])
            train.extend(items[1:])
        test_endpoint = _endpoint_pairs(graphs, held_out)
        test_expanded = [pair for pair in _expanded_pairs(graphs) if pair[0] == held_out]
        folds.append(
            {
                "held_out": str(graphs[held_out].design_name),
                "train": train,
                "validation": validation,
                "test_endpoint": test_endpoint,
                "test_expanded": test_expanded,
            }
        )
    return folds


def _train_one(
    graphs: list[Data],
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    *,
    task: str,
    drop_timing: bool,
    drop_edges: bool,
    uniform_attention: bool,
    epochs: int,
    hidden_channels: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    alpha: float,
    beta: float,
    seed: int,
) -> tuple[AblatedSlackGNN, int]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = AblatedSlackGNN(
        int(graphs[0].node_features.size(1)),
        int(graphs[0].edge_attr.size(1)),
        int(graphs[0].timing_features.size(1)),
        hidden_channels,
        drop_timing=drop_timing,
        drop_edges=drop_edges,
        uniform_attention=uniform_attention,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = _positive_weight(graphs, train_pairs)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    waited = 0
    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        epoch_started = time.perf_counter()
        model.train()
        train_loss = _epoch_loss(model, graphs, train_pairs, task, optimizer, pos_weight, alpha, beta)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            logger.info("epoch %s/%s train %.4f in %.1fs", epochs_ran, epochs, train_loss, time.perf_counter() - epoch_started)
            continue
        model.eval()
        with torch.no_grad():
            val_loss = _epoch_loss(model, graphs, validation_pairs, task, None, pos_weight, alpha, beta)
        logger.info(
            "epoch %s/%s train %.4f val %.4f in %.1fs",
            epochs_ran,
            epochs,
            train_loss,
            val_loss,
            time.perf_counter() - epoch_started,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = snapshot
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                logger.info("early stop at epoch %s", epochs_ran)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, epochs_ran


def _epoch_loss(
    model: AblatedSlackGNN,
    graphs: list[Data],
    pairs: list[Pair],
    task: str,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
    alpha: float,
    beta: float,
) -> float:
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    total = 0.0
    count = 0
    for graph_index, nodes in grouped.items():
        data = graphs[graph_index]
        logits, slack, _attention, _hidden = model(
            data.node_features, data.edge_index, data.edge_attr, data.timing_features
        )
        index = torch.tensor(nodes, dtype=torch.long)
        loss = _objective(logits, slack, data, index, task, pos_weight, alpha, beta)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(nodes)
        count += len(nodes)
    return total / count if count else 0.0


def _objective(
    logits: torch.Tensor,
    slack: torch.Tensor,
    data: Data,
    index: torch.Tensor,
    task: str,
    pos_weight: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    classification = nn.functional.binary_cross_entropy_with_logits(
        logits[index],
        data.expanded_violation[index],
        pos_weight=pos_weight,
    )
    regression = nn.functional.smooth_l1_loss(slack[index], data.expanded_slack[index])
    if task == "classification":
        return classification
    if task == "regression":
        return regression
    return alpha * classification + beta * regression


def _evaluate(
    graphs: list[Data],
    model: AblatedSlackGNN,
    pairs: list[Pair],
    scope: str,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    rows = _prediction_rows(graphs, model, pairs, scope)
    if not rows:
        empty = {key: None for key in METRIC_KEYS}
        empty["support"] = 0
        return empty, {"label": [], "score": [], "slack": [], "prediction": []}
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    actual = [float(row["slack"]) for row in rows]
    predicted = [float(row["prediction"]) for row in rows]
    classification = classification_metrics(labels, scores)
    regression = regression_metrics(actual, predicted)
    metrics: dict[str, object] = {
        "accuracy": classification["accuracy"],
        "precision": classification["precision"],
        "recall": classification["recall"],
        "f1": classification["f1"],
        "roc_auc": classification["roc_auc"],
        "mae": regression["mae"],
        "mse": regression["mse"],
        "rmse": regression["rmse"],
        "support": classification["support"],
        "true_positive": classification["true_positive"],
        "false_positive": classification["false_positive"],
        "false_negative": classification["false_negative"],
        "true_negative": classification["true_negative"],
    }
    packed = {"label": labels, "score": scores, "slack": actual, "prediction": predicted}
    return metrics, packed


def _prediction_rows(
    graphs: list[Data],
    model: AblatedSlackGNN,
    pairs: list[Pair],
    scope: str,
) -> list[dict[str, float]]:
    model.eval()
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            logits, slack, _attention, _hidden = model(
                data.node_features, data.edge_index, data.edge_attr, data.timing_features
            )
            cached[graph_index] = (logits, slack)
    rows: list[dict[str, float]] = []
    for graph_index, node_index in pairs:
        logits, slack = cached[graph_index]
        data = graphs[graph_index]
        if scope == "endpoint":
            label = int(data.y_violation[node_index].item())
            actual = float(data.y_slack[node_index].item())
        else:
            label = int(data.expanded_violation[node_index].item())
            actual = float(data.expanded_slack[node_index].item())
        rows.append(
            {
                "label": label,
                "score": float(torch.sigmoid(logits[node_index]).item()),
                "slack": actual,
                "prediction": float(slack[node_index].item()),
            }
        )
    return rows


def _positive_weight(graphs: list[Data], pairs: list[Pair]) -> torch.Tensor:
    violations = sum(_expanded_class(graphs, pair) for pair in pairs)
    met = len(pairs) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _expanded_pairs(graphs: list[Data]) -> list[Pair]:
    pairs: list[Pair] = []
    for graph_index, data in enumerate(graphs):
        nodes = data.expanded_mask.nonzero(as_tuple=False).view(-1).tolist()
        pairs.extend((graph_index, int(node)) for node in nodes)
    return pairs


def _endpoint_pairs(graphs: list[Data], graph_index: int) -> list[Pair]:
    data = graphs[graph_index]
    nodes = data.label_mask.nonzero(as_tuple=False).view(-1).tolist()
    return [(graph_index, int(node)) for node in nodes]


def _expanded_class(graphs: list[Data], pair: Pair) -> int:
    graph_index, node_index = pair
    return int(graphs[graph_index].expanded_violation[node_index].item())
