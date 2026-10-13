"""Robustness of the six-design leave-one-design-out protocol.

Training can see fewer labels, flipped classes, noisy slack, fewer designs,
or fewer reported paths. Held-out endpoint scores always use the published
slack. This module does not change the M9.4 or M9.5 trainers.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.m9.ablation import AblatedSlackGNN
from slackgnn.m9.metrics_agg import METRIC_KEYS, aggregate_rows
from slackgnn.m9.reporting import fmt, write_json, write_text
from slackgnn.m9.trainer import expanded_leave_one_design_out
from slackgnn.paths import data_dir, repo_root
from slackgnn.training.metrics import classification_metrics, regression_metrics
from slackgnn.training.models import MODEL_NAMES, build_model

logger = logging.getLogger(__name__)

ARCHITECTURES = MODEL_NAMES + ("slackgnn",)
CORPUS_ORDER = ("gcd", "aes", "ibex", "jpeg", "riscv32i", "chameleon")
SPARSE_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.1)
NOISE_RATES = (0.05, 0.1, 0.2)
CORPUS_SIZES = (6, 5, 4, 3)
PATH_DROP_FRACTIONS = (0.1, 0.25, 0.5)
REFERENCE_F1 = 0.8076
COLLAPSE_F1_DROP = 0.10
COLLAPSE_AUC = 0.80
COLLAPSE_MAE_INCREASE = 0.15
STABLE_F1_GAP = 0.05
STABLE_MAE_GAP = 0.10
_COLORS = {
    "gcn": "#1f4e79",
    "graphsage": "#2f6f4e",
    "gin": "#8c2f39",
    "slackgnn": "#c47b2b",
}
Pair = tuple[int, int]
Override = dict[Pair, tuple[float, float]]


def dataset_statistics(graphs: list[Data]) -> dict[str, int]:
    """Count nodes, edges, and labels on the loaded study graphs."""
    return {
        "design_count": len(graphs),
        "nodes": sum(int(graph.num_nodes) for graph in graphs),
        "edges": sum(int(graph.edge_index.size(1)) for graph in graphs),
        "expanded_labels": sum(int(graph.expanded_mask.sum().item()) for graph in graphs),
        "endpoint_labels": sum(int(graph.label_mask.sum().item()) for graph in graphs),
    }


def stratified_subsample(pairs: list[Pair], class_of, fraction: float, rng: random.Random) -> list[Pair]:
    """Keep a class-balanced fraction of pairs. A present class keeps at least one pair."""
    if fraction >= 1.0:
        return list(pairs)
    grouped: dict[int, list[Pair]] = {}
    for pair in pairs:
        grouped.setdefault(int(class_of(pair)), []).append(pair)
    kept: list[Pair] = []
    for items in grouped.values():
        rng.shuffle(items)
        count = int(round(len(items) * fraction))
        if count < 1:
            count = 1
        kept.extend(items[: min(count, len(items))])
    return kept


def split_one_per_class(pairs: list[Pair], class_of, rng: random.Random) -> tuple[list[Pair], list[Pair]]:
    """Hold out one pair per class for validation, matching the M9 split rule."""
    grouped: dict[int, list[Pair]] = {0: [], 1: []}
    for pair in pairs:
        grouped.setdefault(int(class_of(pair)), []).append(pair)
    train: list[Pair] = []
    validation: list[Pair] = []
    for items in grouped.values():
        rng.shuffle(items)
        if len(items) <= 1:
            train.extend(items)
            continue
        validation.append(items[0])
        train.extend(items[1:])
    return train, validation


def kept_route_indices(count: int, fraction: float, rng: random.Random) -> list[int]:
    """Drop a fraction of paths, leaving at least one path when any exist."""
    order = list(range(count))
    rng.shuffle(order)
    if count <= 1 or fraction <= 0:
        return order
    drop_count = min(count - 1, math.ceil(count * fraction))
    return order[drop_count:]


def supervision_from_routes(
    num_nodes: int,
    routes: list[dict[str, object]],
    kept_indices: list[int],
    endpoint_mask: torch.Tensor,
    endpoint_slack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Label nodes that remain on a kept route. Covered endpoints keep published slack."""
    slack = torch.full((num_nodes,), float("nan"))
    mapped = [route for route in routes if route.get("mapped") and route.get("node_indices")]
    for index in kept_indices:
        route = mapped[index]
        value = float(route["slack"])
        for node in route["node_indices"]:
            current = float(slack[int(node)].item())
            if math.isnan(current) or value < current:
                slack[int(node)] = value
    labeled = torch.isfinite(slack)
    covered_endpoints = endpoint_mask.bool() & labeled
    slack = slack.clone()
    slack[covered_endpoints] = endpoint_slack[covered_endpoints].to(slack.dtype)
    violation = torch.zeros(num_nodes, dtype=torch.float32)
    violation[labeled & (slack < 0)] = 1.0
    return labeled, slack, violation


def apply_label_noise(
    graphs: list[Data],
    pairs: list[Pair],
    rate: float,
    rng: random.Random,
) -> tuple[Override, float]:
    """Flip training classes and add Gaussian slack noise. Sigma is rate times the slack spread."""
    values = [float(graphs[graph_index].expanded_slack[node].item()) for graph_index, node in pairs]
    sigma = rate * (statistics.pstdev(values) if len(values) > 1 else 1.0)
    overrides: Override = {}
    for graph_index, node in pairs:
        label = float(graphs[graph_index].expanded_violation[node].item())
        if rng.random() < rate:
            label = 1.0 - label
        slack = float(graphs[graph_index].expanded_slack[node].item()) + rng.gauss(0.0, sigma)
        overrides[(graph_index, node)] = (label, slack)
    return overrides, sigma


def run_robustness_study(
    graphs: list[Data],
    routes: dict[str, list[dict[str, object]]],
    *,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run every M9.6 condition for GCN, GraphSAGE, GIN, and SlackGNN."""
    started = time.perf_counter()
    reference = _run_architectures(graphs, seed=seed, tag="full labels")
    sparse = _sparse_study(graphs, reference, seed=seed)
    _checkpoint(checkpoint_dir, "sparse_labels.json", sparse)
    noisy = _noise_study(graphs, seed=seed)
    _checkpoint(checkpoint_dir, "noisy_labels.json", noisy)
    reduced = _corpus_study(graphs, reference, seed=seed)
    _checkpoint(checkpoint_dir, "reduced_corpus.json", reduced)
    missing = _path_study(graphs, routes, seed=seed)
    _checkpoint(checkpoint_dir, "missing_paths.json", missing)
    comparison = compare_robustness(sparse, noisy, reduced, missing, reference)
    return {
        "stage": "M9.6",
        "seed": seed,
        "statistics": dataset_statistics(graphs),
        "designs": [str(graph.design_name) for graph in graphs],
        "sparse_labels": sparse,
        "noisy_labels": noisy,
        "reduced_corpus": reduced,
        "missing_paths": missing,
        "comparison": comparison,
        "runtime_seconds": time.perf_counter() - started,
    }


def compare_robustness(
    sparse: dict[str, object],
    noisy: dict[str, object],
    reduced: dict[str, object],
    missing: dict[str, object],
    reference: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Rank architectures on the degraded conditions and locate SlackGNN's failure points."""
    rows = _condition_rows(sparse, noisy, reduced, missing)
    degraded = [row for row in rows if not row["reference"]]
    by_model: dict[str, list[dict[str, object]]] = {name: [] for name in ARCHITECTURES}
    for row in degraded:
        by_model[str(row["model"])].append(row)
    scores, most, least = rank_models(by_model)
    reference_f1 = _mean_metric(reference["slackgnn"], "f1")
    reference_mae = _mean_metric(reference["slackgnn"], "mae")
    slack_rows = [row for row in rows if row["model"] == "slackgnn"]
    collapses = [row["condition"] for row in slack_rows if _collapsed(row, reference_f1, reference_mae)]
    return {
        "rows": rows,
        "scores": scores,
        "most_robust_model": most,
        "least_robust_model": least,
        "reference_endpoint_f1": reference_f1,
        "reference_endpoint_mae": reference_mae,
        "slackgnn_collapse_conditions": collapses,
        "slackgnn_superior_conditions": [
            row["condition"]
            for row in slack_rows
            if _slackgnn_leads(row["condition"], rows)
        ],
        "minimum_stable_corpus_size": _minimum_stable_corpus(reduced, reference_f1, reference_mae),
        "failure_thresholds": _failure_thresholds(slack_rows, reference_f1, reference_mae),
        "collapse_rule": (
            f"Endpoint F1 falls more than {COLLAPSE_F1_DROP:.2f} below the full-data SlackGNN result, "
            f"ROC-AUC falls below {COLLAPSE_AUC:.2f}, or MAE rises more than {COLLAPSE_MAE_INCREASE:.2f} ns."
        ),
    }


def rank_models(by_model: dict[str, list[dict[str, object]]]) -> tuple[dict[str, dict[str, float | None]], str | None, str | None]:
    """Highest mean endpoint F1 is the most robust. Missing F1 counts as zero."""
    scores: dict[str, dict[str, float | None]] = {}
    for name, rows in by_model.items():
        f1_values = [0.0 if row.get("f1") is None else float(row["f1"]) for row in rows]
        mae_values = [float(row["mae"]) for row in rows if row.get("mae") is not None]
        scores[name] = {
            "mean_f1": sum(f1_values) / len(f1_values) if f1_values else None,
            "worst_f1": min(f1_values) if f1_values else None,
            "mean_mae": sum(mae_values) / len(mae_values) if mae_values else None,
            "condition_count": len(rows),
        }

    def sort_key(name: str) -> tuple[float, float]:
        score = scores[name]
        f1 = -1.0 if score["mean_f1"] is None else float(score["mean_f1"])
        mae = 0.0 if score["mean_mae"] is None else float(score["mean_mae"])
        return (f1, -mae)

    ranked = [name for name in by_model if scores[name]["mean_f1"] is not None]
    if not ranked:
        return scores, None, None
    return scores, max(ranked, key=sort_key), min(ranked, key=sort_key)


def write_robustness_outputs(payload: dict[str, object], root: str | Path | None = None) -> dict[str, Path]:
    """Write the JSON studies, figures, and both research reports."""
    base = Path(root) if root is not None else repo_root()
    report_dir = base / "reports" / "robustness"
    figure_dir = base / "reports" / "figures" / "robustness"
    written = {
        "sparse_labels": write_json(report_dir / "sparse_labels.json", payload["sparse_labels"]),
        "noisy_labels": write_json(report_dir / "noisy_labels.json", payload["noisy_labels"]),
        "reduced_corpus": write_json(report_dir / "reduced_corpus.json", payload["reduced_corpus"]),
        "missing_paths": write_json(report_dir / "missing_paths.json", payload["missing_paths"]),
        "comparison": write_json(report_dir / "model_robustness_comparison.json", payload["comparison"]),
        "results": write_text(base / "reports" / "robustness_results.md", render_results_markdown(payload)),
        "analysis": write_text(base / "docs" / "slackgnn_robustness_analysis.md", render_analysis_markdown(payload)),
    }
    written.update(write_robustness_figures(payload, figure_dir))
    return written


def write_robustness_figures(payload: dict[str, object], directory: str | Path) -> dict[str, Path]:
    """Write the five robustness charts."""
    figure_dir = Path(directory)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sparse = payload["sparse_labels"]["conditions"]
    noisy = payload["noisy_labels"]["conditions"]
    reduced = payload["reduced_corpus"]["conditions"]
    missing = payload["missing_paths"]["conditions"]
    return {
        "sparse_figure": _curve_figure(sparse, "fraction", "Label fraction", figure_dir / "sparse_labels.png"),
        "noise_figure": _curve_figure(noisy, "rate", "Label noise rate", figure_dir / "noisy_labels.png"),
        "corpus_figure": _curve_figure(reduced, "design_count", "Designs in the corpus", figure_dir / "reduced_corpus.png"),
        "paths_figure": _curve_figure(missing, "drop_fraction", "Timing paths removed", figure_dir / "missing_paths.png"),
        "comparison_figure": _comparison_figure(payload["comparison"], figure_dir / "model_robustness.png"),
    }


def render_results_markdown(payload: dict[str, object]) -> str:
    """Summarize every condition with endpoint F1, ROC-AUC, MAE, and RMSE."""
    comparison = payload["comparison"]
    lines = [
        "# Robustness results",
        "",
        "Leave-one-design-out. Training labels change. Endpoint metrics use published slack.",
        "",
        _setup_lines(payload),
        "",
        "## All conditions",
        "",
        _headline_table(comparison["rows"]),
        "",
        "## Failure thresholds",
        "",
        comparison["collapse_rule"],
        "",
    ]
    thresholds = comparison["failure_thresholds"]
    for family, condition in thresholds.items():
        lines.append(f"- {family}: {condition}.")
    lines.extend(
        [
            "",
            f"Most robust model: {comparison['most_robust_model']}.",
            f"Least robust model: {comparison['least_robust_model']}.",
            f"Smallest stable corpus: {comparison['minimum_stable_corpus_size']} designs.",
            "",
            "## Recommendations for scaling",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _recommendations(payload))
    lines.append("")
    return "\n".join(lines)


def render_analysis_markdown(payload: dict[str, object]) -> str:
    """Write the longer robustness note."""
    stats = payload["statistics"]
    comparison = payload["comparison"]
    lines = [
        "# SlackGNN robustness analysis",
        "",
        "## Experimental setup",
        "",
        "Each condition trains GCN, GraphSAGE, GIN, and SlackGNN with one design held out. "
        "The loss uses expanded route labels from the other designs. "
        "Accuracy, precision, recall, F1, and ROC-AUC score the violation class. "
        "MAE and RMSE score slack in nanoseconds. "
        "The held-out endpoints keep their published slack, including when training paths or labels are removed. "
        "Baselines keep separate classification and regression heads on the one-hot node type. "
        "SlackGNN keeps the multitask loss. Hidden width is 32, the epoch cap is 20, and patience is 6. "
        f"The seed is {payload['seed']}.",
        "",
        "Sparse labels keep a class-balanced fraction of the training and validation pool, then hold out one label per class again. "
        "Noise flips that fraction of training classes and adds Gaussian noise whose standard deviation is the same fraction of the training slack spread. "
        "Reduced corpora are prefixes of gcd, aes, ibex, jpeg, riscv32i, chameleon. "
        "Missing paths drop that fraction of mapped routes on the training designs and relabel nodes from the routes that remain. At least one route stays.",
        "",
        "## Dataset statistics",
        "",
        f"Designs: {stats['design_count']}. Nodes: {stats['nodes']}. Edges: {stats['edges']}. "
        f"Expanded labels: {stats['expanded_labels']}. Published endpoints: {stats['endpoint_labels']}.",
        "",
        "## Robustness results",
        "",
        _headline_table(comparison["rows"]),
        "",
        "## Failure thresholds",
        "",
        comparison["collapse_rule"],
        "",
    ]
    for family, condition in comparison["failure_thresholds"].items():
        lines.append(f"- {family}: {condition}.")
    lines.extend(
        [
            "",
            "## Most resilient architecture",
            "",
            f"Most robust model on the degraded conditions: {comparison['most_robust_model']}.",
            f"Least robust model: {comparison['least_robust_model']}.",
            "",
            _score_table(comparison["scores"]),
            "",
            f"SlackGNN stays ahead of every baseline on F1 and MAE for: {', '.join(comparison['slackgnn_superior_conditions']) or 'none'}.",
            "",
            "## Recommended future improvements",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _recommendations(payload))
    lines.append("")
    return "\n".join(lines)


def _checkpoint(directory: str | Path | None, name: str, payload: dict[str, object]) -> None:
    if directory is None:
        return
    path = write_json(Path(directory) / name, payload)
    logger.info("Checkpoint %s", path)


def load_design_routes(data_root: str | Path | None = None) -> dict[str, list[dict[str, object]]]:
    """Reread reported routes. This does not rewrite the expanded-label files."""
    from slackgnn.expansion.label_expansion import expand_design, inventory

    root = Path(data_root) if data_root is not None else data_dir()
    found = inventory(root)
    routes: dict[str, list[dict[str, object]]] = {}
    for name in found["available"]:
        expanded = expand_design(root / str(name))
        errors = list(expanded["summary"].get("errors", []))
        if errors or not expanded["routes"]:
            detail = "; ".join(errors) if errors else "no mapped routes"
            raise RuntimeError(f"{name} routes could not be loaded: {detail}")
        routes[str(name)] = list(expanded["routes"])
    return routes


def _sparse_study(graphs: list[Data], reference: dict[str, dict[str, object]], *, seed: int) -> dict[str, object]:
    conditions = [_reference_condition(reference, "fraction", 1.0, "100% labels")]
    for fraction in SPARSE_FRACTIONS:
        if fraction == 1.0:
            continue
        logger.info("Sparse labels %.0f%%", fraction * 100)
        trained = _run_architectures(
            graphs,
            seed=seed,
            tag=f"sparse {fraction:.0%}",
            prepare=_sparse_prepare(fraction, seed),
        )
        conditions.append(_condition("fraction", fraction, f"{fraction:.0%} labels", trained))
    return _study("M9.6.1", "sparse expanded labels", conditions)


def _noise_study(graphs: list[Data], *, seed: int) -> dict[str, object]:
    conditions = []
    for rate in NOISE_RATES:
        logger.info("Noisy labels %.0f%%", rate * 100)
        trained = _run_architectures(
            graphs,
            seed=seed,
            tag=f"noise {rate:.0%}",
            prepare=_noise_prepare(rate, seed),
        )
        conditions.append(_condition("rate", rate, f"{rate:.0%} noise", trained))
    return _study("M9.6.2", "flipped classes and Gaussian slack noise", conditions)


def _corpus_study(graphs: list[Data], reference: dict[str, dict[str, object]], *, seed: int) -> dict[str, object]:
    by_name = {str(graph.design_name): graph for graph in graphs}
    available = [name for name in CORPUS_ORDER if name in by_name]
    conditions = []
    for count in CORPUS_SIZES:
        names = available[:count]
        if len(names) < count:
            continue
        if count == len(available):
            conditions.append(_reference_condition(reference, "design_count", count, f"{count} designs"))
            continue
        logger.info("Reduced corpus %s", names)
        trained = _run_architectures([by_name[name] for name in names], seed=seed, tag=f"corpus {count}")
        conditions.append(_condition("design_count", count, f"{count} designs", trained, extra={"designs": names}))
    return _study("M9.6.3", "nested design prefixes", conditions)


def _path_study(graphs: list[Data], routes: dict[str, list[dict[str, object]]], *, seed: int) -> dict[str, object]:
    conditions = []
    for fraction in PATH_DROP_FRACTIONS:
        logger.info("Missing paths %.0f%%", fraction * 100)
        plan = _path_plan(graphs, routes, fraction, seed)
        trained = _run_architectures(
            graphs,
            seed=seed,
            tag=f"paths {fraction:.0%}",
            prepare=_path_prepare(plan, seed),
        )
        conditions.append(
            _condition(
                "drop_fraction",
                fraction,
                f"{fraction:.0%} paths removed",
                trained,
                extra={"dropped_paths": plan["dropped"]},
            )
        )
    return _study("M9.6.4", "training routes removed before the loss", conditions)


def _run_architectures(
    graphs: list[Data],
    *,
    seed: int,
    tag: str,
    prepare=None,
    epochs: int = 20,
    hidden_channels: int = 32,
    patience: int = 6,
    learning_rate: float = 1e-2,
) -> dict[str, dict[str, object]]:
    trained: dict[str, dict[str, object]] = {}
    for name in ARCHITECTURES:
        logger.info("Robustness %s / %s", tag, name)
        trained[name] = _run_model(
            graphs,
            name,
            seed=seed,
            prepare=prepare,
            epochs=epochs,
            hidden_channels=hidden_channels,
            patience=patience,
            learning_rate=learning_rate,
        )
    return trained


def _run_model(
    graphs: list[Data],
    model_name: str,
    *,
    seed: int,
    prepare,
    epochs: int,
    hidden_channels: int,
    patience: int,
    learning_rate: float,
) -> dict[str, object]:
    folds = expanded_leave_one_design_out(graphs, seed)
    rows: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        logger.info("Training %s with %s held out", model_name, fold["held_out"])
        train_pairs, validation_pairs, supervision = _prepared_fold(graphs, fold, index, prepare)
        if model_name == "slackgnn":
            model = _fit_slackgnn(
                graphs,
                train_pairs,
                validation_pairs,
                supervision,
                epochs,
                hidden_channels,
                patience,
                learning_rate,
                seed,
            )
            endpoint = _score_slackgnn(graphs, model, fold["test_endpoint"], "endpoint")
            expanded = _score_slackgnn(graphs, model, fold["test_expanded"], "expanded")
            del model
        else:
            classifier = _fit_baseline(
                graphs,
                model_name,
                train_pairs,
                validation_pairs,
                supervision,
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
                train_pairs,
                validation_pairs,
                supervision,
                "regression",
                epochs,
                hidden_channels,
                patience,
                learning_rate,
                seed,
            )
            endpoint = _score_baseline(graphs, classifier, regressor, fold["test_endpoint"], "endpoint")
            expanded = _score_baseline(graphs, classifier, regressor, fold["test_expanded"], "expanded")
            del classifier, regressor
        rows.append(
            {
                "held_out": fold["held_out"],
                "train_support": len(train_pairs),
                "validation_support": len(validation_pairs),
                "endpoint": endpoint,
                "expanded": expanded,
            }
        )
    return {
        "folds": rows,
        "summary": {
            "endpoint": aggregate_rows([row["endpoint"] for row in rows]),
            "expanded": aggregate_rows([row["expanded"] for row in rows]),
        },
    }


def _prepared_fold(graphs: list[Data], fold: dict[str, object], index: int, prepare):
    if prepare is None:
        return list(fold["train"]), list(fold["validation"]), {}
    return prepare(graphs, fold, index)


def _sparse_prepare(fraction: float, seed: int):
    def prepare(graphs: list[Data], fold: dict[str, object], index: int):
        rng = random.Random(seed + 1000 + index)
        pool = list(fold["train"]) + list(fold["validation"])
        class_of = _clean_class(graphs)
        kept = stratified_subsample(pool, class_of, fraction, rng)
        train_pairs, validation_pairs = split_one_per_class(kept, class_of, rng)
        return train_pairs, validation_pairs, {}

    return prepare


def _noise_prepare(rate: float, seed: int):
    def prepare(graphs: list[Data], fold: dict[str, object], index: int):
        rng = random.Random(seed + 2000 + index)
        pool = list(fold["train"]) + list(fold["validation"])
        overrides, _sigma = apply_label_noise(graphs, pool, rate, rng)
        return list(fold["train"]), list(fold["validation"]), overrides

    return prepare


def _path_prepare(plan: dict[str, object], seed: int):
    allowed = plan["allowed"]
    tensors = plan["tensors"]

    def prepare(graphs: list[Data], fold: dict[str, object], index: int):
        rng = random.Random(seed + 4000 + index)
        pool = [
            pair
            for pair in list(fold["train"]) + list(fold["validation"])
            if _pair_survives(graphs, pair, allowed)
        ]
        class_of = _plan_class(graphs, tensors)
        train_pairs, validation_pairs = split_one_per_class(pool, class_of, rng)
        return train_pairs, validation_pairs, _path_overrides(graphs, train_pairs + validation_pairs, tensors)

    return prepare


def _path_plan(graphs: list[Data], routes: dict[str, list[dict[str, object]]], fraction: float, seed: int) -> dict[str, object]:
    allowed: dict[str, set[int]] = {}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    dropped: dict[str, dict[str, int]] = {}
    for index, graph in enumerate(graphs):
        name = str(graph.design_name)
        mapped = [route for route in routes[name] if route.get("mapped") and route.get("node_indices")]
        kept = kept_route_indices(len(mapped), fraction, random.Random(seed + 3000 + index))
        labeled, slack, violation = supervision_from_routes(
            int(graph.num_nodes),
            routes[name],
            kept,
            graph.label_mask,
            graph.y_slack,
        )
        nodes = labeled.nonzero(as_tuple=False).view(-1).tolist()
        allowed[name] = {int(node) for node in nodes}
        tensors[name] = {"slack": slack, "violation": violation}
        dropped[name] = {"mapped_paths": len(mapped), "dropped_paths": len(mapped) - len(kept), "remaining_labels": len(nodes)}
    return {"allowed": allowed, "tensors": tensors, "dropped": dropped}


def _fit_slackgnn(
    graphs: list[Data],
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    supervision: Override,
    epochs: int,
    hidden_channels: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> AblatedSlackGNN:
    torch.manual_seed(seed)
    random.seed(seed)
    model = AblatedSlackGNN(
        int(graphs[0].node_features.size(1)),
        int(graphs[0].edge_attr.size(1)),
        int(graphs[0].timing_features.size(1)),
        hidden_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = _positive_weight(graphs, train_pairs, supervision)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    waited = 0
    for _epoch in range(epochs):
        model.train()
        _slack_loss(model, graphs, train_pairs, supervision, optimizer, pos_weight)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            continue
        model.eval()
        with torch.no_grad():
            loss = _slack_loss(model, graphs, validation_pairs, supervision, None, pos_weight)
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


def _fit_baseline(
    graphs: list[Data],
    model_name: str,
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    supervision: Override,
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
    pos_weight = _positive_weight(graphs, train_pairs, supervision) if task == "classification" else None
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    waited = 0
    for _epoch in range(epochs):
        model.train()
        _baseline_loss(model, graphs, train_pairs, supervision, task, optimizer, pos_weight)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            continue
        model.eval()
        with torch.no_grad():
            loss = _baseline_loss(model, graphs, validation_pairs, supervision, task, None, pos_weight)
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


def _slack_loss(
    model: AblatedSlackGNN,
    graphs: list[Data],
    pairs: list[Pair],
    supervision: Override,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor,
) -> float:
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    total = 0.0
    count = 0
    for graph_index, nodes in grouped.items():
        data = graphs[graph_index]
        logits, slack, _attention, _hidden = model(data.node_features, data.edge_index, data.edge_attr, data.timing_features)
        index = torch.tensor(nodes, dtype=torch.long)
        target_label, target_slack = _target_tensors(graphs, graph_index, nodes, supervision)
        classification = nn.functional.binary_cross_entropy_with_logits(logits[index], target_label, pos_weight=pos_weight)
        regression = nn.functional.smooth_l1_loss(slack[index], target_slack)
        loss = classification + 0.5 * regression
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(nodes)
        count += len(nodes)
    return total / count if count else 0.0


def _baseline_loss(
    model: nn.Module,
    graphs: list[Data],
    pairs: list[Pair],
    supervision: Override,
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
        target_label, target_slack = _target_tensors(graphs, graph_index, nodes, supervision)
        if task == "classification":
            loss = nn.functional.binary_cross_entropy_with_logits(prediction[index], target_label, pos_weight=pos_weight)
        else:
            loss = nn.functional.mse_loss(prediction[index], target_slack)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(nodes)
        count += len(nodes)
    return total / count if count else 0.0


def _score_slackgnn(graphs: list[Data], model: AblatedSlackGNN, pairs: list[Pair], scope: str) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    actual: list[float] = []
    predicted: list[float] = []
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            logits, slack, _attention, _hidden = model(data.node_features, data.edge_index, data.edge_attr, data.timing_features)
            cached[graph_index] = (logits, slack)
        for graph_index, node_index in pairs:
            logits, slack = cached[graph_index]
            data = graphs[graph_index]
            if scope == "endpoint":
                labels.append(int(data.y_violation[node_index].item()))
                actual.append(float(data.y_slack[node_index].item()))
            else:
                labels.append(int(data.expanded_violation[node_index].item()))
                actual.append(float(data.expanded_slack[node_index].item()))
            scores.append(float(torch.sigmoid(logits[node_index]).item()))
            predicted.append(float(slack[node_index].item()))
    return _metrics(labels, scores, actual, predicted)


def _score_baseline(
    graphs: list[Data],
    classifier: nn.Module,
    regressor: nn.Module,
    pairs: list[Pair],
    scope: str,
) -> dict[str, object]:
    classifier.eval()
    regressor.eval()
    labels: list[int] = []
    scores: list[float] = []
    actual: list[float] = []
    predicted: list[float] = []
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    class_cache: dict[int, torch.Tensor] = {}
    slack_cache: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            class_cache[graph_index] = classifier(data.x, data.edge_index)
            slack_cache[graph_index] = regressor(data.x, data.edge_index)
        for graph_index, node_index in pairs:
            data = graphs[graph_index]
            if scope == "endpoint":
                labels.append(int(data.y_violation[node_index].item()))
                actual.append(float(data.y_slack[node_index].item()))
            else:
                labels.append(int(data.expanded_violation[node_index].item()))
                actual.append(float(data.expanded_slack[node_index].item()))
            raw = class_cache[graph_index][node_index]
            scores.append(float(torch.sigmoid(raw).item()))
            predicted.append(float(slack_cache[graph_index][node_index].item()))
    return _metrics(labels, scores, actual, predicted)


def _metrics(labels: list[int], scores: list[float], actual: list[float], predicted: list[float]) -> dict[str, object]:
    if not labels:
        empty = {key: None for key in METRIC_KEYS}
        empty["support"] = 0
        return empty
    classification = classification_metrics(labels, scores)
    regression = regression_metrics(actual, predicted)
    return {
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


def _target_tensors(
    graphs: list[Data],
    graph_index: int,
    nodes: list[int],
    supervision: Override,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = graphs[graph_index]
    labels = []
    slack = []
    for node in nodes:
        if (graph_index, node) in supervision:
            label, value = supervision[(graph_index, node)]
        else:
            label = float(data.expanded_violation[node].item())
            value = float(data.expanded_slack[node].item())
        labels.append(label)
        slack.append(value)
    return torch.tensor(labels, dtype=torch.float32), torch.tensor(slack, dtype=torch.float32)


def _positive_weight(graphs: list[Data], pairs: list[Pair], supervision: Override) -> torch.Tensor:
    violations = 0.0
    for pair in pairs:
        if pair in supervision:
            violations += supervision[pair][0]
        else:
            graph_index, node = pair
            violations += float(graphs[graph_index].expanded_violation[node].item())
    met = len(pairs) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _clean_class(graphs: list[Data]):
    def class_of(pair: Pair) -> int:
        graph_index, node = pair
        return int(graphs[graph_index].expanded_violation[node].item())

    return class_of


def _plan_class(graphs: list[Data], tensors: dict[str, dict[str, torch.Tensor]]):
    def class_of(pair: Pair) -> int:
        graph_index, node = pair
        name = str(graphs[graph_index].design_name)
        return int(tensors[name]["violation"][node].item())

    return class_of


def _pair_survives(graphs: list[Data], pair: Pair, allowed: dict[str, set[int]]) -> bool:
    graph_index, node = pair
    name = str(graphs[graph_index].design_name)
    return node in allowed[name]


def _path_overrides(
    graphs: list[Data],
    pairs: list[Pair],
    tensors: dict[str, dict[str, torch.Tensor]],
) -> Override:
    overrides: Override = {}
    for graph_index, node in pairs:
        name = str(graphs[graph_index].design_name)
        overrides[(graph_index, node)] = (
            float(tensors[name]["violation"][node].item()),
            float(tensors[name]["slack"][node].item()),
        )
    return overrides


def _reference_condition(reference: dict[str, dict[str, object]], key: str, value: float, label: str) -> dict[str, object]:
    return _condition(key, value, label, reference, reference=True)


def _condition(
    key: str,
    value: float,
    label: str,
    trained: dict[str, dict[str, object]],
    *,
    extra: dict[str, object] | None = None,
    reference: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {key: value, "label": label, "reference": reference, "models": trained}
    if extra:
        payload.update(extra)
    return payload


def _study(stage: str, protocol: str, conditions: list[dict[str, object]]) -> dict[str, object]:
    return {"stage": stage, "protocol": protocol, "conditions": conditions}


def _condition_rows(
    sparse: dict[str, object],
    noisy: dict[str, object],
    reduced: dict[str, object],
    missing: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for study, family in (
        (sparse, "sparse_labels"),
        (noisy, "noisy_labels"),
        (reduced, "reduced_corpus"),
        (missing, "missing_paths"),
    ):
        for condition in study["conditions"]:
            for name in ARCHITECTURES:
                summary = condition["models"][name]["summary"]["endpoint"]
                rows.append(
                    {
                        "family": family,
                        "condition": f"{family}: {condition['label']}",
                        "label": condition["label"],
                        "model": name,
                        "reference": bool(condition.get("reference", False)),
                        "f1": _unwrap(summary["f1"]),
                        "roc_auc": _unwrap(summary["roc_auc"]),
                        "mae": _unwrap(summary["mae"]),
                        "rmse": _unwrap(summary["rmse"]),
                        "accuracy": _unwrap(summary["accuracy"]),
                        "precision": _unwrap(summary["precision"]),
                        "recall": _unwrap(summary["recall"]),
                    }
                )
    return rows


def _collapsed(row: dict[str, object], reference_f1: float | None, reference_mae: float | None) -> bool:
    f1 = row.get("f1")
    auc = row.get("roc_auc")
    mae = row.get("mae")
    if f1 is None or (reference_f1 is not None and float(f1) < reference_f1 - COLLAPSE_F1_DROP):
        return True
    if auc is not None and float(auc) < COLLAPSE_AUC:
        return True
    if mae is not None and reference_mae is not None and float(mae) > reference_mae + COLLAPSE_MAE_INCREASE:
        return True
    return False


def _slackgnn_leads(condition: str, rows: list[dict[str, object]]) -> bool:
    grouped = [row for row in rows if row["condition"] == condition]
    slack = next(row for row in grouped if row["model"] == "slackgnn")
    baselines = [row for row in grouped if row["model"] != "slackgnn"]
    if slack.get("f1") is None or slack.get("mae") is None:
        return False
    f1_values = [row.get("f1") for row in baselines]
    mae_values = [row.get("mae") for row in baselines]
    if any(value is None for value in f1_values + mae_values):
        return False
    return float(slack["f1"]) > max(float(value) for value in f1_values) and float(slack["mae"]) < min(
        float(value) for value in mae_values
    )


def _minimum_stable_corpus(reduced: dict[str, object], reference_f1: float | None, reference_mae: float | None) -> int | None:
    if reference_f1 is None or reference_mae is None:
        return None
    stable = []
    for condition in reduced["conditions"]:
        f1 = _mean_metric(condition["models"]["slackgnn"], "f1")
        mae = _mean_metric(condition["models"]["slackgnn"], "mae")
        if f1 is None or mae is None:
            continue
        if abs(f1 - reference_f1) <= STABLE_F1_GAP and abs(mae - reference_mae) <= STABLE_MAE_GAP:
            stable.append(int(condition["design_count"]))
    return min(stable) if stable else None


def _failure_thresholds(rows: list[dict[str, object]], reference_f1: float | None, reference_mae: float | None) -> dict[str, str]:
    families = (
        ("sparse_labels", "Sparse labels"),
        ("noisy_labels", "Noisy labels"),
        ("reduced_corpus", "Reduced corpus"),
        ("missing_paths", "Missing paths"),
    )
    thresholds: dict[str, str] = {}
    for family, title in families:
        failed = [row["label"] for row in rows if row["family"] == family and _collapsed(row, reference_f1, reference_mae)]
        thresholds[title] = failed[0] if failed else "no tested setting crosses the collapse rule"
    return thresholds


def _recommendations(payload: dict[str, object]) -> list[str]:
    comparison = payload["comparison"]
    thresholds = comparison["failure_thresholds"]
    size = comparison["minimum_stable_corpus_size"]
    size_text = "the full six-design corpus" if size is None else f"{size} designs"
    return [
        f"Treat {size_text} as the smallest corpus that stays within {STABLE_F1_GAP:.2f} F1 and {STABLE_MAE_GAP:.2f} ns MAE of the six-design SlackGNN result.",
        f"Sparse-label threshold: {thresholds['Sparse labels']}. Scaling should add reported endpoints before relying on copied route labels alone.",
        f"Noise threshold: {thresholds['Noisy labels']}. A scaling run should keep the published endpoint slack untouched and record label noise separately.",
        f"Missing-path threshold: {thresholds['Missing paths']}. The next scaling design is useful when its finish report contributes new endpoints, not only a larger netlist.",
        f"Compare new designs against {comparison['most_robust_model']} as the current robust reference and against {comparison['least_robust_model']} as the weak reference.",
    ]


def _mean_metric(model: dict[str, object], key: str) -> float | None:
    return _unwrap(model["summary"]["endpoint"][key])


def _unwrap(value: object) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _setup_lines(payload: dict[str, object]) -> str:
    stats = payload["statistics"]
    return (
        f"Designs: {', '.join(payload['designs'])}. "
        f"Nodes {stats['nodes']}, edges {stats['edges']}, expanded labels {stats['expanded_labels']}, "
        f"endpoints {stats['endpoint_labels']}. Seed {payload['seed']}."
    )


def _headline_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Condition | Model | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {model} | {f1} | {roc_auc} | {mae} | {rmse} |".format(
                condition=row["condition"],
                model=row["model"],
                f1=fmt(row["f1"]),
                roc_auc=fmt(row["roc_auc"]),
                mae=fmt(row["mae"]),
                rmse=fmt(row["rmse"]),
            )
        )
    return "\n".join(lines)


def _score_table(scores: dict[str, dict[str, float | None]]) -> str:
    lines = [
        "| Model | Mean degraded F1 | Worst F1 | Mean degraded MAE (ns) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ARCHITECTURES:
        score = scores[name]
        lines.append(
            f"| {name} | {fmt(score['mean_f1'])} | {fmt(score['worst_f1'])} | {fmt(score['mean_mae'])} |"
        )
    return "\n".join(lines)


def _curve_figure(conditions: list[dict[str, object]], key: str, xlabel: str, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for name in ARCHITECTURES:
        xs = [float(condition[key]) for condition in conditions]
        f1 = [_or_nan(_mean_metric(condition["models"][name], "f1")) for condition in conditions]
        mae = [_or_nan(_mean_metric(condition["models"][name], "mae")) for condition in conditions]
        axes[0].plot(xs, f1, marker="o", linewidth=2.0, color=_COLORS[name], label=name)
        axes[1].plot(xs, mae, marker="o", linewidth=2.0, color=_COLORS[name], label=name)
    axes[0].set_ylabel("Endpoint F1")
    axes[1].set_ylabel("Endpoint MAE (ns)")
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _comparison_figure(comparison: dict[str, object], path: Path) -> Path:
    scores = comparison["scores"]
    names = list(ARCHITECTURES)
    f1 = [_or_nan(scores[name]["mean_f1"]) for name in names]
    mae = [_or_nan(scores[name]["mean_mae"]) for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    axes[0].bar(names, f1, color=[_COLORS[name] for name in names])
    axes[1].bar(names, mae, color=[_COLORS[name] for name in names])
    axes[0].set_ylabel("Mean endpoint F1")
    axes[1].set_ylabel("Mean endpoint MAE (ns)")
    axes[0].set_title("Degraded conditions")
    axes[1].set_title("Degraded conditions")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    return value
