"""Measure how SlackGNN runtime and memory grow with the published corpus.

The study times the existing models on the loaded graphs. It does not change
the M9.4 protocol or rewrite published results. Accuracy figures are read
from the generalization report when that file is present.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.m9.ablation import AblatedSlackGNN
from slackgnn.m9.reporting import fmt, write_json, write_text
from slackgnn.paths import data_dir, repo_root
from slackgnn.training.models import MODEL_NAMES, build_model

logger = logging.getLogger(__name__)

ARCHITECTURES = MODEL_NAMES + ("slackgnn",)
DESIGN_ORDER = ("gcd", "aes", "ibex", "jpeg", "riscv32i", "chameleon")
GROWTH_COUNTS = (10, 20, 50)
PROTOCOL_EPOCHS = 20
TIMING_REPEATS = 3
_COLORS = {
    "gcn": "#1f4e79",
    "graphsage": "#2f6f4e",
    "gin": "#8c2f39",
    "slackgnn": "#c47b2b",
}
_COMPLEXITY = ("O(N)", "O(E)", "O(N log N)", "O(E log N)")


def run_scaling_analysis(
    graphs: list[Data],
    *,
    seed: int = 0,
    repeats: int = TIMING_REPEATS,
    metrics_path: str | Path | None = None,
) -> dict[str, object]:
    """Collect dataset, runtime, memory, inference, and projection results."""
    started = time.perf_counter()
    designs = design_inventory(graphs)
    metrics = load_published_metrics(metrics_path)
    timed = measure_models(graphs, seed=seed, repeats=repeats)
    complexity = analyze_complexity(timed)
    projection = project_corpus(designs, timed)
    inference = inference_summary(timed)
    bottlenecks = analyze_bottlenecks(timed, metrics, complexity)
    return {
        "stage": "M9.7",
        "seed": seed,
        "repeats": repeats,
        "protocol_epochs": PROTOCOL_EPOCHS,
        "design_scaling": {"designs": designs, "totals": _design_totals(designs)},
        "training_time_scaling": timed["training"],
        "memory_scaling": timed["memory"],
        "inference_scaling": inference,
        "corpus_growth_projection": projection,
        "complexity_analysis": complexity,
        "bottleneck_analysis": bottlenecks,
        "published_metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "note": (
            "Epoch time is the median of repeated single-design steps after one warmup. "
            "A published leave-one-design-out run visits every other design once per epoch. "
            "Baseline protocol time counts the separate classification and regression heads. "
            "Accuracy is the published endpoint mean from the generalization report."
        ),
    }


def design_inventory(graphs: list[Data], data_root: str | Path | None = None) -> list[dict[str, object]]:
    """Record graph shape and on-disk size for every loaded design."""
    root = Path(data_root) if data_root is not None else data_dir()
    rows = []
    for graph in graphs:
        name = str(graph.design_name)
        directory = root / name
        rows.append(
            {
                "design": name,
                "nodes": int(graph.num_nodes),
                "edges": int(graph.edge_index.size(1)),
                "endpoint_labels": int(graph.label_mask.sum().item()),
                "expanded_labels": int(graph.expanded_mask.sum().item()),
                "label_count": int(graph.expanded_mask.sum().item()),
                "graph_directory_bytes": _directory_bytes(directory),
                "pyg_graph_bytes": _file_bytes(directory / "pyg" / "graph.pt"),
                "dataset_bytes": _directory_bytes(directory / "dataset"),
                "tensor_bytes": tensor_bytes(graph),
            }
        )
    rows.sort(key=lambda row: DESIGN_ORDER.index(row["design"]) if row["design"] in DESIGN_ORDER else 99)
    return rows


def tensor_bytes(graph: Data) -> int:
    """Byte size of the tensors the training loop keeps for one design."""
    total = 0
    for value in graph.stores[0].values():
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    return total


def measure_models(graphs: list[Data], *, seed: int, repeats: int) -> dict[str, object]:
    """Time and trace one training step, a validation pass, and inference."""
    per_model: dict[str, list[dict[str, object]]] = {}
    for name in ARCHITECTURES:
        logger.info("Scaling measurements for %s", name)
        per_model[name] = [_measure_design(graph, name, seed=seed, repeats=repeats) for graph in graphs]
    training_rows = []
    memory_rows = []
    for name, rows in per_model.items():
        heads = 1 if name == "slackgnn" else 2
        epoch_seconds = sum(float(row["train_seconds"]) for row in rows)
        training_rows.append(
            {
                "model": name,
                "heads": heads,
                "designs": rows,
                "seconds_per_epoch": epoch_seconds,
                "seconds_per_epoch_per_head": epoch_seconds,
                "validation_seconds": sum(float(row["validation_seconds"]) for row in rows),
                "inference_seconds": sum(float(row["inference_seconds"]) for row in rows),
                "forward_seconds": sum(float(row["forward_seconds"]) for row in rows),
                "backward_seconds": sum(float(row["backward_seconds"]) for row in rows),
                "protocol_training_seconds": protocol_training_seconds(epoch_seconds, len(rows), heads),
            }
        )
        memory_rows.append(
            {
                "model": name,
                "parameter_bytes": int(rows[0]["parameter_bytes"]) if rows else 0,
                "dataset_tensor_bytes": sum(int(row["dataset_tensor_bytes"]) for row in rows),
                "peak_traced_bytes": max(int(row["pytorch_step_bytes"]) for row in rows) if rows else 0,
                "process_working_set_bytes": max(int(row["working_set_bytes"] or 0) for row in rows),
                "designs": [
                    {
                        "design": row["design"],
                        "nodes": row["nodes"],
                        "edges": row["edges"],
                        "dataset_tensor_bytes": row["dataset_tensor_bytes"],
                        "peak_traced_bytes": row["pytorch_step_bytes"],
                        "pytorch_step_bytes": row["pytorch_step_bytes"],
                        "working_set_bytes": row["working_set_bytes"],
                    }
                    for row in rows
                ],
            }
        )
    return {"per_model": per_model, "training": {"models": training_rows}, "memory": {"models": memory_rows}}


def protocol_training_seconds(epoch_seconds_all_designs: float, design_count: int, heads: int, epochs: int = PROTOCOL_EPOCHS) -> float:
    """Estimate full leave-one-design-out wall time from one all-design epoch."""
    if design_count <= 1:
        return epochs * heads * epoch_seconds_all_designs
    fold_epoch = epoch_seconds_all_designs * (design_count - 1) / design_count
    return design_count * epochs * heads * fold_epoch


def project_counts(current_designs: int, current_value: float, target_designs: int) -> float:
    """Scale a corpus total by the design-count ratio."""
    if current_designs <= 0:
        return 0.0
    return current_value * target_designs / current_designs


def project_lodo_seconds(epoch_seconds_all_designs: float, current_designs: int, target_designs: int, heads: int) -> float:
    """Project leave-one-design-out time when every new design has the current average size."""
    if current_designs <= 0:
        return 0.0
    average = epoch_seconds_all_designs / current_designs
    fold_epoch = average * max(target_designs - 1, 0)
    return target_designs * PROTOCOL_EPOCHS * heads * fold_epoch


def fit_complexity(rows: list[dict[str, object]], time_key: str = "train_seconds") -> dict[str, object]:
    """Choose the listed complexity family with the smallest relative error."""
    nodes = [float(row["nodes"]) for row in rows]
    edges = [float(row["edges"]) for row in rows]
    labels = [float(row["label_count"]) for row in rows]
    observed = [float(row[time_key]) for row in rows]
    features = {
        "O(N)": nodes,
        "O(E)": edges,
        "O(N log N)": [node * math.log(node) for node in nodes],
        "O(E log N)": [edge * math.log(node) for edge, node in zip(edges, nodes, strict=True)],
    }
    fits = {name: _linear_fit(feature, observed) for name, feature in features.items()}
    best = min(fits, key=lambda name: fits[name]["mean_relative_error"])
    return {
        "best": best,
        "fits": fits,
        "slope_vs_nodes": _log_slope(nodes, observed),
        "slope_vs_edges": _log_slope(edges, observed),
        "slope_vs_labels": _log_slope(labels, observed) if min(labels) > 0 else None,
    }


def analyze_complexity(timed: dict[str, object]) -> dict[str, object]:
    """Fit runtime against nodes, edges, and labels for every model."""
    models = {}
    votes: dict[str, int] = {name: 0 for name in _COMPLEXITY}
    for name, rows in timed["per_model"].items():
        fit = fit_complexity(rows)
        models[name] = fit
        votes[str(fit["best"])] += 1
    chosen = max(votes, key=lambda name: (votes[name], name == "O(E)"))
    return {"models": models, "votes": votes, "best_description": chosen}


def project_corpus(designs: list[dict[str, object]], timed: dict[str, object]) -> dict[str, object]:
    """Project average-design growth to 10, 20, and 50 designs."""
    count = len(designs)
    nodes = sum(int(row["nodes"]) for row in designs)
    edges = sum(int(row["edges"]) for row in designs)
    labels = sum(int(row["label_count"]) for row in designs)
    disk = sum(int(row["graph_directory_bytes"]) for row in designs)
    epoch = {row["model"]: float(row["seconds_per_epoch"]) for row in timed["training"]["models"]}
    heads = {row["model"]: int(row["heads"]) for row in timed["training"]["models"]}
    points = []
    for target in (count, *GROWTH_COUNTS):
        points.append(
            {
                "designs": target,
                "nodes": project_counts(count, nodes, target),
                "edges": project_counts(count, edges, target),
                "labels": project_counts(count, labels, target),
                "graph_directory_bytes": project_counts(count, disk, target),
                "training_seconds": {
                    name: project_lodo_seconds(epoch[name], count, target, heads[name]) for name in epoch
                },
                "resident_tensor_bytes": project_counts(
                    count,
                    sum(int(row["dataset_tensor_bytes"]) for row in timed["memory"]["models"][0]["designs"]),
                    target,
                ),
            }
        )
    return {
        "assumption": "Each added design matches the current mean node, edge, label, and disk size. Training still steps one graph at a time.",
        "current": points[0],
        "projections": points[1:],
    }


def inference_summary(timed: dict[str, object]) -> dict[str, object]:
    """Summarize single-design and sequential batch inference."""
    models = []
    for name, rows in timed["per_model"].items():
        batch_seconds = sum(float(row["inference_seconds"]) for row in rows)
        nodes = sum(int(row["nodes"]) for row in rows)
        labels = sum(int(row["label_count"]) for row in rows)
        models.append(
            {
                "model": name,
                "single_design_seconds": [
                    {"design": row["design"], "nodes": row["nodes"], "edges": row["edges"], "seconds": row["inference_seconds"]}
                    for row in rows
                ],
                "batch_inference_seconds": batch_seconds,
                "nodes_per_second": nodes / batch_seconds if batch_seconds else None,
                "labeled_predictions_per_second": labels / batch_seconds if batch_seconds else None,
            }
        )
    return {"models": models, "batch_definition": "One forward pass on each design, run sequentially."}


def analyze_bottlenecks(
    timed: dict[str, object],
    metrics: dict[str, dict[str, float | None]],
    complexity: dict[str, object],
) -> dict[str, object]:
    """Name the slow graph, the heavy memory graph, and the efficiency leaders."""
    slack_rows = timed["per_model"]["slackgnn"]
    slowest = max(slack_rows, key=lambda row: float(row["train_seconds"]))
    largest_memory = max(slack_rows, key=lambda row: int(row["pytorch_step_bytes"]))
    training = timed["training"]["models"]
    fastest = min(training, key=lambda row: float(row["seconds_per_epoch"]))
    efficient = min(training, key=lambda row: float(row["seconds_per_epoch"]) / _model_edges(timed, str(row["model"])))
    accuracy = []
    for row in training:
        score = metrics.get(str(row["model"]), {})
        f1 = score.get("f1")
        seconds = float(row["protocol_training_seconds"])
        accuracy.append(
            {
                "model": row["model"],
                "f1_per_protocol_second": None if f1 is None or seconds <= 0 else float(f1) / seconds,
            }
        )
    best_accuracy = max(accuracy, key=lambda row: -1.0 if row["f1_per_protocol_second"] is None else row["f1_per_protocol_second"])
    forward = sum(float(row["forward_seconds"]) for row in slack_rows)
    backward = sum(float(row["backward_seconds"]) for row in slack_rows)
    return {
        "largest_runtime_bottleneck": {
            "design": slowest["design"],
            "train_seconds": slowest["train_seconds"],
            "share_of_slackgnn_epoch": float(slowest["train_seconds"]) / sum(float(row["train_seconds"]) for row in slack_rows),
            "dominant_pass": "backward" if backward > forward else "forward",
            "forward_seconds": forward,
            "backward_seconds": backward,
        },
        "largest_memory_bottleneck": {
            "design": largest_memory["design"],
            "pytorch_step_bytes": largest_memory["pytorch_step_bytes"],
            "dataset_tensor_bytes": largest_memory["dataset_tensor_bytes"],
        },
        "fastest_model": fastest["model"],
        "most_efficient_model": efficient["model"],
        "best_accuracy_per_second_model": best_accuracy["model"],
        "accuracy_per_second": accuracy,
        "complexity": complexity["best_description"],
    }


def write_scaling_outputs(payload: dict[str, object], root: str | Path | None = None) -> dict[str, Path]:
    """Write the scaling JSON files, figures, and both reports."""
    base = Path(root) if root is not None else repo_root()
    report_dir = base / "reports" / "scaling"
    figure_dir = base / "reports" / "figures" / "scaling"
    written = {
        "design_scaling": write_json(report_dir / "design_scaling.json", payload["design_scaling"]),
        "training_time_scaling": write_json(report_dir / "training_time_scaling.json", payload["training_time_scaling"]),
        "memory_scaling": write_json(report_dir / "memory_scaling.json", payload["memory_scaling"]),
        "corpus_growth_projection": write_json(report_dir / "corpus_growth_projection.json", payload["corpus_growth_projection"]),
        "complexity_analysis": write_json(report_dir / "complexity_analysis.json", payload["complexity_analysis"]),
        "inference_scaling": write_json(report_dir / "inference_scaling.json", payload["inference_scaling"]),
        "bottleneck_analysis": write_json(report_dir / "bottleneck_analysis.json", payload["bottleneck_analysis"]),
        "results": write_text(base / "reports" / "scaling_results.md", render_scaling_results(payload)),
        "analysis": write_text(base / "docs" / "slackgnn_scaling_analysis.md", render_scaling_analysis(payload)),
    }
    written.update(write_scaling_figures(payload, figure_dir))
    return written


def write_scaling_figures(payload: dict[str, object], directory: str | Path) -> dict[str, Path]:
    """Write the seven scaling charts."""
    figure_dir = Path(directory)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return {
        "runtime_nodes": _runtime_scatter(payload, "nodes", "Nodes", figure_dir / "runtime_vs_nodes.png"),
        "runtime_edges": _runtime_scatter(payload, "edges", "Edges", figure_dir / "runtime_vs_edges.png"),
        "memory_nodes": _memory_scatter(payload, "nodes", "Nodes", figure_dir / "memory_vs_nodes.png"),
        "memory_edges": _memory_scatter(payload, "edges", "Edges", figure_dir / "memory_vs_edges.png"),
        "inference": _inference_scatter(payload, figure_dir / "inference_vs_graph_size.png"),
        "growth": _growth_figure(payload, figure_dir / "corpus_growth_projection.png"),
        "comparison": _comparison_figure(payload, figure_dir / "model_comparison.png"),
    }


def render_scaling_results(payload: dict[str, object]) -> str:
    """Format the scaling tables requested for the phase summary."""
    lines = [
        "# Scaling results",
        "",
        payload["note"],
        "",
        "## Models",
        "",
        _model_table(payload),
        "",
        "## Datasets",
        "",
        _dataset_table(payload),
        "",
        "## Projections",
        "",
        _projection_table(payload),
        "",
        "## Complexity",
        "",
        f"Best description of training time: {payload['complexity_analysis']['best_description']}. "
        f"Mean log-log slope against edges: {fmt(_mean_edge_slope(payload), 2)}.",
        "",
        "## Bottlenecks",
        "",
        _bottleneck_lines(payload),
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in _recommendations(payload))
    lines.append("")
    return "\n".join(lines)


def render_scaling_analysis(payload: dict[str, object]) -> str:
    """Write the longer scaling note."""
    lines = [
        "# SlackGNN scaling analysis",
        "",
        "## Experimental setup",
        "",
        "Each model is timed on one published design at a time. Hidden width is 32. "
        f"Each timing entry is the median of {payload['repeats']} steps after a warmup step. "
        "GCN, GraphSAGE, and GIN use the one-hot node type. SlackGNN uses node, edge, and timing features. "
        "The published accuracy comes from the six-design leave-one-design-out generalization run. "
        "This study does not retrain that comparison.",
        "",
        payload["note"],
        "",
        "## Dataset statistics",
        "",
        _dataset_table(payload),
        "",
        "## Runtime scaling",
        "",
        _model_table(payload),
        "",
        f"Training time follows {payload['complexity_analysis']['best_description']} most closely among O(N), O(E), O(N log N), and O(E log N). "
        f"The mean log-log slope against edges is {fmt(_mean_edge_slope(payload), 2)}.",
        "",
        "## Memory scaling",
        "",
        _memory_table(payload),
        "",
        "Dataset tensor bytes track the netlist. PyTorch step memory is parameters plus forward activations and gradients. "
        "The process working set includes every loaded design.",
        "",
        "## Inference scaling",
        "",
        _inference_table(payload),
        "",
        "## Growth projections",
        "",
        payload["corpus_growth_projection"]["assumption"],
        "",
        _projection_table(payload),
        "",
        "## Bottlenecks",
        "",
        _bottleneck_lines(payload),
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in _recommendations(payload))
    lines.append("")
    return "\n".join(lines)


def load_published_metrics(path: str | Path | None = None) -> dict[str, dict[str, float | None]]:
    """Read endpoint means from the generalization report."""
    source = Path(path) if path is not None else repo_root() / "reports" / "generalization_results.json"
    if not source.is_file():
        return {name: {"f1": None, "roc_auc": None, "mae": None} for name in ARCHITECTURES}
    payload = json.loads(source.read_text(encoding="utf-8"))
    metrics: dict[str, dict[str, float | None]] = {}
    for name in ARCHITECTURES:
        summary = payload.get("models", {}).get(name, {}).get("summary", {}).get("endpoint", {})
        metrics[name] = {
            "f1": _mean(summary.get("f1")),
            "roc_auc": _mean(summary.get("roc_auc")),
            "mae": _mean(summary.get("mae")),
        }
    return metrics


def _measure_design(graph: Data, model_name: str, *, seed: int, repeats: int) -> dict[str, object]:
    torch.manual_seed(seed)
    model = _build(graph, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    _train_step(model, graph, model_name, optimizer)
    train_samples = [_train_step(model, graph, model_name, optimizer) for _ in range(repeats)]
    validation_samples = [_validation_step(model, graph, model_name) for _ in range(repeats)]
    inference_samples = [_inference_step(model, graph, model_name) for _ in range(repeats)]
    train_seconds = _median([sample["seconds"] for sample in train_samples])
    parameter_bytes = _parameter_bytes(model)
    activation_bytes = int(_median([sample["activation_bytes"] for sample in train_samples]))
    gradient_bytes = int(_median([sample["gradient_bytes"] for sample in train_samples]))
    return {
        "design": str(graph.design_name),
        "model": model_name,
        "nodes": int(graph.num_nodes),
        "edges": int(graph.edge_index.size(1)),
        "label_count": int(graph.expanded_mask.sum().item()),
        "train_seconds": train_seconds,
        "forward_seconds": _median([sample["forward_seconds"] for sample in train_samples]),
        "backward_seconds": _median([sample["backward_seconds"] for sample in train_samples]),
        "validation_seconds": _median(validation_samples),
        "inference_seconds": _median(inference_samples),
        "parameter_bytes": parameter_bytes,
        "dataset_tensor_bytes": tensor_bytes(graph),
        "activation_bytes": activation_bytes,
        "gradient_bytes": gradient_bytes,
        "pytorch_step_bytes": parameter_bytes + activation_bytes + gradient_bytes,
        "peak_traced_bytes": max(int(sample["peak_traced_bytes"]) for sample in train_samples),
        "working_set_bytes": working_set_bytes(),
    }


def _build(graph: Data, model_name: str) -> nn.Module:
    if model_name == "slackgnn":
        return AblatedSlackGNN(
            int(graph.node_features.size(1)),
            int(graph.edge_attr.size(1)),
            int(graph.timing_features.size(1)),
            32,
        )
    return build_model(model_name, int(graph.x.size(1)), 32)


def _train_step(model: nn.Module, graph: Data, model_name: str, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tracemalloc.start()
    started = time.perf_counter()
    loss, forward_seconds, activation_bytes = _forward_loss(model, graph, model_name)
    backward_started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - backward_started
    gradient_bytes = _gradient_bytes(model)
    optimizer.step()
    seconds = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_traced_bytes": float(peak),
        "activation_bytes": float(activation_bytes),
        "gradient_bytes": float(gradient_bytes),
    }


def _validation_step(model: nn.Module, graph: Data, model_name: str) -> float:
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        _forward_loss(model, graph, model_name)
    return time.perf_counter() - started


def _inference_step(model: nn.Module, graph: Data, model_name: str) -> float:
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        if model_name == "slackgnn":
            model(graph.node_features, graph.edge_index, graph.edge_attr, graph.timing_features)
        else:
            model(graph.x, graph.edge_index)
    return time.perf_counter() - started


def _forward_loss(model: nn.Module, graph: Data, model_name: str) -> tuple[torch.Tensor, float, int]:
    index = graph.expanded_mask.nonzero(as_tuple=False).view(-1)
    seen: set[int] = set()
    activation = 0

    def remember(output: object) -> None:
        nonlocal activation
        if torch.is_tensor(output):
            storage = output.untyped_storage()
            pointer = int(storage.data_ptr())
            if pointer not in seen:
                seen.add(pointer)
                activation += int(storage.nbytes())
            return
        if isinstance(output, (tuple, list)):
            for item in output:
                remember(item)

    handles = [module.register_forward_hook(lambda _module, _inputs, output: remember(output)) for module in model.modules()]
    started = time.perf_counter()
    try:
        if model_name == "slackgnn":
            logits, slack, _attention, _hidden = model(graph.node_features, graph.edge_index, graph.edge_attr, graph.timing_features)
            pos_weight = _positive_weight(graph.expanded_violation[index])
            classification = nn.functional.binary_cross_entropy_with_logits(logits[index], graph.expanded_violation[index], pos_weight=pos_weight)
            regression = nn.functional.smooth_l1_loss(slack[index], graph.expanded_slack[index])
            loss = classification + 0.5 * regression
        else:
            prediction = model(graph.x, graph.edge_index)
            loss = nn.functional.mse_loss(prediction[index], graph.expanded_slack[index])
        forward_seconds = time.perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    return loss, forward_seconds, activation


def _positive_weight(labels: torch.Tensor) -> torch.Tensor:
    violations = float(labels.sum().item())
    met = float(labels.numel()) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _linear_fit(feature: list[float], observed: list[float]) -> dict[str, float | None]:
    denominator = sum(value * value for value in feature)
    scale = sum(x * y for x, y in zip(feature, observed, strict=True)) / denominator if denominator else 0.0
    predicted = [scale * value for value in feature]
    errors = [abs(y - hat) / y for y, hat in zip(observed, predicted, strict=True) if y]
    mean_y = sum(observed) / len(observed)
    total = sum((value - mean_y) ** 2 for value in observed)
    residual = sum((y - hat) ** 2 for y, hat in zip(observed, predicted, strict=True))
    return {
        "scale": scale,
        "mean_relative_error": sum(errors) / len(errors) if errors else None,
        "r2": None if total == 0 else 1.0 - residual / total,
    }


def _log_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or min(xs) <= 0 or min(ys) <= 0:
        return None
    log_x = [math.log(value) for value in xs]
    log_y = [math.log(value) for value in ys]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True)) / denominator


def _parameter_bytes(model: nn.Module) -> int:
    return sum(int(parameter.numel()) * int(parameter.element_size()) for parameter in model.parameters())


def _gradient_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        storage = parameter.grad.untyped_storage()
        pointer = int(storage.data_ptr())
        if pointer in seen:
            continue
        seen.add(pointer)
        total += int(storage.nbytes())
    return total
    return sum(int(parameter.numel()) * int(parameter.element_size()) for parameter in model.parameters())


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_bytes(path: Path) -> int:
    if not path.is_file():
        return 0
    return int(path.stat().st_size)


def working_set_bytes() -> int | None:
    """Return the process working set on Windows."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
    except OSError:
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    query = psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    query.restype = ctypes.c_int
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def _design_totals(designs: list[dict[str, object]]) -> dict[str, int]:
    return {
        "design_count": len(designs),
        "nodes": sum(int(row["nodes"]) for row in designs),
        "edges": sum(int(row["edges"]) for row in designs),
        "label_count": sum(int(row["label_count"]) for row in designs),
        "graph_directory_bytes": sum(int(row["graph_directory_bytes"]) for row in designs),
        "pyg_graph_bytes": sum(int(row["pyg_graph_bytes"]) for row in designs),
        "dataset_bytes": sum(int(row["dataset_bytes"]) for row in designs),
    }


def _model_edges(timed: dict[str, object], model_name: str) -> float:
    return float(sum(int(row["edges"]) for row in timed["per_model"][model_name]))


def _mean_edge_slope(payload: dict[str, object]) -> float | None:
    slopes = [
        float(row["slope_vs_edges"])
        for row in payload["complexity_analysis"]["models"].values()
        if row.get("slope_vs_edges") is not None
    ]
    if not slopes:
        return None
    return sum(slopes) / len(slopes)
    return float(sum(int(row["edges"]) for row in timed["per_model"][model_name]))


def _mean(value: object) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def _recommendations(payload: dict[str, object]) -> list[str]:
    bottlenecks = payload["bottleneck_analysis"]
    runtime = bottlenecks["largest_runtime_bottleneck"]
    memory = bottlenecks["largest_memory_bottleneck"]
    return [
        f"M9.8 should plot training time against edges and mark {runtime['design']} as the runtime bottleneck. The fitted behavior is {bottlenecks['complexity']}.",
        f"M9.9 should report peak step memory on {memory['design']} separately from corpus disk growth. Accuracy stays the published six-design holdout, and this study does not replace it.",
        f"M9.10 should publish the scaling JSON and figures, keep generated netlists out of the source archive, and document {bottlenecks['fastest_model']} as the fastest model, {bottlenecks['most_efficient_model']} as the lowest seconds per edge, and {bottlenecks['best_accuracy_per_second_model']} as the best published F1 per projected training second.",
    ]


def _model_table(payload: dict[str, object]) -> str:
    metrics = payload["published_metrics"]
    lines = [
        "| Model | Training time (s) | Inference time (s) | PyTorch step (MB) | F1 | ROC-AUC | MAE (ns) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    memory = {row["model"]: row for row in payload["memory_scaling"]["models"]}
    inference = {row["model"]: row for row in payload["inference_scaling"]["models"]}
    for row in payload["training_time_scaling"]["models"]:
        name = str(row["model"])
        score = metrics.get(name, {})
        lines.append(
            "| {model} | {train} | {infer} | {memory} | {f1} | {auc} | {mae} |".format(
                model=name,
                train=fmt(row["protocol_training_seconds"], 2),
                infer=fmt(inference[name]["batch_inference_seconds"], 4),
                memory=fmt(memory[name]["peak_traced_bytes"] / 1e6, 2),
                f1=fmt(score.get("f1")),
                auc=fmt(score.get("roc_auc")),
                mae=fmt(score.get("mae")),
            )
        )
    return "\n".join(lines)


def _dataset_table(payload: dict[str, object]) -> str:
    lines = [
        "| Dataset | Nodes | Edges | Labels | Disk size (MB) | PyG graph (MB) | Dataset (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["design_scaling"]["designs"]:
        lines.append(
            "| {design} | {nodes} | {edges} | {labels} | {disk} | {pyg} | {dataset} |".format(
                design=row["design"],
                nodes=row["nodes"],
                edges=row["edges"],
                labels=row["label_count"],
                disk=fmt(int(row["graph_directory_bytes"]) / 1e6, 2),
                pyg=fmt(int(row["pyg_graph_bytes"]) / 1e6, 2),
                dataset=fmt(int(row["dataset_bytes"]) / 1e6, 2),
            )
        )
    return "\n".join(lines)


def _projection_table(payload: dict[str, object]) -> str:
    lines = [
        "| Designs | Nodes | Edges | Labels | Disk (MB) | SlackGNN training (min) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    current = payload["corpus_growth_projection"]["current"]
    points = [current, *payload["corpus_growth_projection"]["projections"]]
    for point in points:
        lines.append(
            "| {designs} | {nodes} | {edges} | {labels} | {disk} | {minutes} |".format(
                designs=int(point["designs"]),
                nodes=fmt(point["nodes"], 0),
                edges=fmt(point["edges"], 0),
                labels=fmt(point["labels"], 0),
                disk=fmt(float(point["graph_directory_bytes"]) / 1e6, 1),
                minutes=fmt(float(point["training_seconds"]["slackgnn"]) / 60, 1),
            )
        )
    return "\n".join(lines)


def _memory_table(payload: dict[str, object]) -> str:
    lines = [
        "| Model | Parameters (KB) | Dataset tensors (MB) | PyTorch step (MB) | Working set (MB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["memory_scaling"]["models"]:
        lines.append(
            "| {model} | {params} | {data} | {peak} | {rss} |".format(
                model=row["model"],
                params=fmt(row["parameter_bytes"] / 1e3, 1),
                data=fmt(row["dataset_tensor_bytes"] / 1e6, 1),
                peak=fmt(row["peak_traced_bytes"] / 1e6, 2),
                rss=fmt(row["process_working_set_bytes"] / 1e6, 1),
            )
        )
    return "\n".join(lines)


def _inference_table(payload: dict[str, object]) -> str:
    lines = [
        "| Model | Batch inference (s) | Nodes per second | Labeled predictions per second |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["inference_scaling"]["models"]:
        lines.append(
            "| {model} | {seconds} | {nodes} | {labels} |".format(
                model=row["model"],
                seconds=fmt(row["batch_inference_seconds"], 4),
                nodes=fmt(row["nodes_per_second"], 0),
                labels=fmt(row["labeled_predictions_per_second"], 0),
            )
        )
    return "\n".join(lines)


def _bottleneck_lines(payload: dict[str, object]) -> str:
    bottlenecks = payload["bottleneck_analysis"]
    runtime = bottlenecks["largest_runtime_bottleneck"]
    memory = bottlenecks["largest_memory_bottleneck"]
    return "\n".join(
        [
            f"Largest runtime bottleneck: {runtime['design']} ({fmt(100 * float(runtime['share_of_slackgnn_epoch']), 1)}% of a SlackGNN epoch). The heavier pass is {runtime['dominant_pass']}.",
            f"Largest memory bottleneck: {memory['design']} PyTorch step, {fmt(int(memory['pytorch_step_bytes']) / 1e6, 2)} MB.",
            f"Fastest model: {bottlenecks['fastest_model']}.",
            f"Most efficient model by seconds per edge: {bottlenecks['most_efficient_model']}.",
            f"Best published F1 per projected training second: {bottlenecks['best_accuracy_per_second_model']}.",
        ]
    )


def _runtime_scatter(payload: dict[str, object], key: str, xlabel: str, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    per_model = _rows_by_model(payload)
    for name in ARCHITECTURES:
        rows = sorted(per_model[name], key=lambda row: float(row[key]))
        axis.plot(
            [row[key] for row in rows],
            [1000 * float(row["train_seconds"]) for row in rows],
            marker="o",
            linewidth=2.0,
            color=_COLORS[name],
            label=name,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Training step (ms)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _memory_scatter(payload: dict[str, object], key: str, xlabel: str, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    per_model = _rows_by_model(payload)
    for name in ARCHITECTURES:
        rows = sorted(per_model[name], key=lambda row: float(row[key]))
        axis.plot(
            [row[key] for row in rows],
            [float(row["pytorch_step_bytes"]) / 1e6 for row in rows],
            marker="o",
            linewidth=2.0,
            color=_COLORS[name],
            label=name,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("PyTorch step memory (MB)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _inference_scatter(payload: dict[str, object], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for row in payload["inference_scaling"]["models"]:
        name = str(row["model"])
        points = sorted(row["single_design_seconds"], key=lambda point: int(point["nodes"]))
        axis.plot(
            [point["nodes"] for point in points],
            [1000 * float(point["seconds"]) for point in points],
            marker="o",
            linewidth=2.0,
            color=_COLORS[name],
            label=name,
        )
    axis.set_xlabel("Nodes")
    axis.set_ylabel("Inference time (ms)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _growth_figure(payload: dict[str, object], path: Path) -> Path:
    current = payload["corpus_growth_projection"]["current"]
    points = [current, *payload["corpus_growth_projection"]["projections"]]
    counts = [int(point["designs"]) for point in points]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    axes[0].plot(counts, [float(point["training_seconds"]["slackgnn"]) / 60 for point in points], marker="o", color="#c47b2b", linewidth=2.0)
    axes[1].plot(counts, [float(point["graph_directory_bytes"]) / 1e6 for point in points], marker="o", color="#1f4e79", linewidth=2.0)
    axes[0].set_ylabel("SlackGNN training (min)")
    axes[1].set_ylabel("Corpus disk (MB)")
    for axis in axes:
        axis.set_xlabel("Designs")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _comparison_figure(payload: dict[str, object], path: Path) -> Path:
    names = [str(row["model"]) for row in payload["training_time_scaling"]["models"]]
    train = [float(row["seconds_per_epoch"]) for row in payload["training_time_scaling"]["models"]]
    infer = [float(row["batch_inference_seconds"]) for row in payload["inference_scaling"]["models"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    axes[0].bar(names, train, color=[_COLORS[name] for name in names])
    axes[1].bar(names, infer, color=[_COLORS[name] for name in names])
    axes[0].set_ylabel("Epoch over six designs (s)")
    axes[1].set_ylabel("Batch inference (s)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _rows_by_model(payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    return {str(row["model"]): list(row["designs"]) for row in payload["training_time_scaling"]["models"]}
