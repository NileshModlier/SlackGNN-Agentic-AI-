"""Run the M9 benchmark stages and write their reports."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from torch_geometric.data import Data

from slackgnn.m9.ablation import ABLATIONS
from slackgnn.m9.corpus import choose_scaling_holdout
from slackgnn.m9.metrics_agg import aggregate_metric
from slackgnn.m9.reporting import fmt, markdown_table, missing_lines, scope_table, summary_line, write_json, write_text
from slackgnn.m9.trainer import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    RUNTIME_NOTE,
    train_holdout,
    train_lodo,
)
from slackgnn.paths import repo_root

logger = logging.getLogger(__name__)

_SPREAD_KEYS = ("f1", "roc_auc", "mae")


def reports_dir() -> Path:
    return repo_root() / "reports" / "m9"


def hyper_parameters(seed: int, task: str = "multitask") -> dict[str, object]:
    return {
        "hidden_channels": DEFAULT_HIDDEN,
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "alpha": DEFAULT_ALPHA,
        "beta": DEFAULT_BETA,
        "seed": seed,
        "task": task,
        "runtime_note": RUNTIME_NOTE,
    }


def run_cross_design(
    graphs: list[Data],
    found: dict[str, object],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Multitask leave-one-design-out with expanded-label training loss."""
    started = time.perf_counter()
    trained = train_lodo(graphs, seed=seed, task="multitask", alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)
    payload = {
        "stage": "M9.4",
        "protocol": "leave_one_design_out",
        "training_loss": "expanded labels",
        "evaluation": (
            "Endpoint metrics use the published graph.pt slack. Expanded metrics score the "
            "path-propagated targets, including nodes that inherited slack from a reported path."
        ),
        "hyper_parameters": hyper_parameters(seed),
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(found.get("missing", [])),
        "skipped_designs": list(found.get("skipped", [])),
        "folds": trained["folds"],
        "summary": trained["summary"],
        "endpoint_support": trained["endpoint_support"],
        "expanded_support": trained["expanded_support"],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_cross(payload)
    return payload


def run_ablations(
    graphs: list[Data],
    found: dict[str, object],
    *,
    seed: int = DEFAULT_SEED,
    full_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Repeat leave-one-design-out for each SlackGNN ablation."""
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for spec in ABLATIONS:
        if spec["name"] == "full" and full_result is not None:
            logger.info("Reusing the full multitask leave-one-design-out result")
            trained = {
                "folds": [_public_fold(fold) for fold in full_result["folds"]],
                "summary": full_result["summary"],
            }
        else:
            logger.info("Ablation %s", spec["name"])
            trained = train_lodo(
                graphs,
                seed=seed,
                task=str(spec["task"]),
                drop_timing=bool(spec["drop_timing"]),
                drop_edges=bool(spec["drop_edges"]),
                uniform_attention=bool(spec["uniform_attention"]),
            )
            trained["folds"] = [_public_fold(fold) for fold in trained["folds"]]
        rows.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "task": spec["task"],
                "drop_timing": spec["drop_timing"],
                "drop_edges": spec["drop_edges"],
                "uniform_attention": spec["uniform_attention"],
                "folds": trained["folds"],
                "summary": trained["summary"],
            }
        )
    payload = {
        "stage": "M9.5",
        "protocol": "leave_one_design_out",
        "training_loss": "expanded labels",
        "winner_rule": "Highest mean endpoint F1. Equal F1 is broken by lower endpoint MAE.",
        "hyper_parameters": hyper_parameters(seed),
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(found.get("missing", [])),
        "ablations": rows,
        "winner": _winner(rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_ablation(payload)
    return payload


def run_robustness(
    graphs: list[Data],
    found: dict[str, object],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    seed0_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Repeat the full multitask protocol at each seed."""
    started = time.perf_counter()
    per_seed: list[dict[str, object]] = []
    for seed in seeds:
        if seed == DEFAULT_SEED and seed0_result is not None:
            logger.info("Reusing seed 0 from the cross-design run")
            trained = seed0_result
        else:
            logger.info("Robustness seed %s", seed)
            trained = train_lodo(graphs, seed=seed, task="multitask")
        per_seed.append(
            {
                "seed": seed,
                "summary": trained["summary"],
                "folds": [_public_fold(fold) for fold in trained["folds"]],
            }
        )
    payload = {
        "stage": "M9.6",
        "protocol": "leave_one_design_out",
        "seeds": list(seeds),
        "hyper_parameters": hyper_parameters(DEFAULT_SEED),
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(found.get("missing", [])),
        "per_seed": per_seed,
        "across_seeds": _across_seeds(per_seed),
        "per_fold": _per_fold_spread(per_seed),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_robustness(payload)
    return payload


def run_scaling(graphs: list[Data], found: dict[str, object], *, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Grow a nested training set while one design with both classes stays held out."""
    started = time.perf_counter()
    held_out, reason = choose_scaling_holdout(graphs)
    points: list[dict[str, object]] = []
    if held_out is not None:
        pool = [graph for graph in graphs if str(graph.design_name) != held_out]
        pool.sort(key=lambda graph: (int(graph.num_nodes), str(graph.design_name)))
        held = next(graph for graph in graphs if str(graph.design_name) == held_out)
        for count in range(2, len(pool) + 1):
            subset = pool[:count]
            logger.info("Scaling train %s test %s", [str(graph.design_name) for graph in subset], held_out)
            trained = train_holdout(subset + [held], held_out, seed=seed)
            points.append(
                {
                    "training_design_count": count,
                    "training_designs": [str(graph.design_name) for graph in subset],
                    "train_support": trained["train_support"],
                    "validation_support": trained["validation_support"],
                    "epochs_ran": trained["epochs_ran"],
                    "endpoint": trained["endpoint"],
                    "expanded": trained["expanded"],
                    "runtime_seconds": trained["runtime_seconds"],
                }
            )
    payload = {
        "stage": "M9.7",
        "protocol": "nested training designs, one fixed held-out design",
        "order": "Training designs are added from the smallest graph to the largest.",
        "held_out": held_out,
        "held_out_reason": reason,
        "hyper_parameters": hyper_parameters(seed),
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(found.get("missing", [])),
        "points": points,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_scaling(payload)
    return payload


def _public_fold(fold: dict[str, object]) -> dict[str, object]:
    """Drop per-node predictions from tables that do not plot them."""
    return {
        "held_out": fold["held_out"],
        "epochs_ran": fold.get("epochs_ran"),
        "train_support": fold.get("train_support"),
        "validation_support": fold.get("validation_support"),
        "endpoint": fold["endpoint"],
        "expanded": fold["expanded"],
    }


def _winner(rows: list[dict[str, object]]) -> dict[str, object] | None:
    ranked: list[tuple[float, float, str]] = []
    for row in rows:
        f1 = row["summary"]["endpoint"]["f1"]["mean"]
        mae = row["summary"]["endpoint"]["mae"]["mean"]
        if f1 is None:
            continue
        ranked.append((float(f1), float("inf") if mae is None else float(mae), str(row["name"])))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_f1, best_mae, name = ranked[0]
    return {"name": name, "endpoint_f1": best_f1, "endpoint_mae": None if best_mae == float("inf") else best_mae}


def _across_seeds(per_seed: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for scope in ("endpoint", "expanded"):
        result[scope] = {}
        for key in _SPREAD_KEYS:
            values = [{"value": seed_row["summary"][scope][key]["mean"]} for seed_row in per_seed]
            result[scope][key] = aggregate_metric(values, "value")
    return result


def _per_fold_spread(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    if not per_seed:
        return []
    names = [str(fold["held_out"]) for fold in per_seed[0]["folds"]]
    rows: list[dict[str, object]] = []
    for name in names:
        collected = []
        support_endpoint = None
        support_expanded = None
        for seed_row in per_seed:
            fold = next(item for item in seed_row["folds"] if item["held_out"] == name)
            collected.append(fold)
            support_endpoint = fold["endpoint"].get("support")
            support_expanded = fold["expanded"].get("support")
        endpoint = collected[0]["endpoint"]
        violations = endpoint.get("true_positive", 0)
        false_negatives = endpoint.get("false_negative", 0)
        row: dict[str, object] = {
            "held_out": name,
            "endpoint_support": support_endpoint,
            "expanded_support": support_expanded,
            "endpoint_violation_count": None
            if violations is None or false_negatives is None
            else int(violations) + int(false_negatives),
            "endpoint_met_count": None
            if endpoint.get("true_negative") is None or endpoint.get("false_positive") is None
            else int(endpoint["true_negative"]) + int(endpoint["false_positive"]),
        }
        for key in _SPREAD_KEYS:
            samples = [{"value": fold["endpoint"][key]} for fold in collected]
            row[key] = aggregate_metric(samples, "value")
        rows.append(row)
    return rows


def _write_cross(payload: dict[str, object]) -> None:
    directory = reports_dir()
    write_json(directory / "cross_design_results.json", payload)
    lines = [
        "# Cross-design benchmark",
        "",
        "Leave-one-design-out. Training loss uses expanded labels. "
        "The endpoint table scores published endpoint slack. "
        "The expanded table scores path-propagated slack and is not an independent STA measurement.",
        "",
        f"Designs: {', '.join(payload['designs']) or 'none'}.",
        "",
        "Missing designs:",
        "",
        missing_lines(payload["missing_designs"]),
        "",
        f"Runtime: {fmt(payload['runtime_seconds'], 1)} s. {RUNTIME_NOTE}",
        "",
        "## Endpoint labels",
        "",
        scope_table(payload["folds"], "endpoint"),
        "",
        summary_line(payload["summary"], "endpoint"),
        "",
        "## Expanded labels",
        "",
        scope_table(payload["folds"], "expanded"),
        "",
        summary_line(payload["summary"], "expanded"),
        "",
    ]
    write_text(directory / "cross_design_results.md", "\n".join(lines))


def _write_ablation(payload: dict[str, object]) -> None:
    directory = reports_dir()
    stored = dict(payload)
    stored["ablations"] = [_without_nothing(row) for row in payload["ablations"]]
    write_json(directory / "ablation_results.json", stored)
    headers = ["Ablation", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MAE (ns)", "MSE", "RMSE (ns)"]
    rows = []
    for row in payload["ablations"]:
        block = row["summary"]["endpoint"]
        rows.append([row["name"], *[fmt(block[key]) for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "mae", "mse", "rmse")]])
    winner = payload["winner"]
    winner_text = "No ablation produced a defined endpoint F1."
    if winner is not None:
        winner_text = (
            f"Winner by mean endpoint F1: `{winner['name']}` "
            f"(F1 {fmt(winner['endpoint_f1'])}, MAE {fmt(winner['endpoint_mae'])} ns)."
        )
    lines = [
        "# Ablation study",
        "",
        "Each row is the leave-one-design-out mean on published endpoint labels. "
        "Training still uses expanded labels.",
        "",
        winner_text,
        "",
        markdown_table(headers, rows),
        "",
        "Missing designs:",
        "",
        missing_lines(payload["missing_designs"]),
        "",
        f"Runtime: {fmt(payload['runtime_seconds'], 1)} s.",
        "",
    ]
    for row in payload["ablations"]:
        lines.extend(
            [
                f"## {row['name']}",
                "",
                row["description"],
                "",
                "### Endpoint",
                "",
                scope_table(row["folds"], "endpoint"),
                "",
                summary_line(row["summary"], "endpoint"),
                "",
                "### Expanded",
                "",
                scope_table(row["folds"], "expanded"),
                "",
                summary_line(row["summary"], "expanded"),
                "",
            ]
        )
    write_text(directory / "ablation_results.md", "\n".join(lines))


def _without_nothing(row: dict[str, object]) -> dict[str, object]:
    return row


def _write_robustness(payload: dict[str, object]) -> None:
    directory = reports_dir()
    write_json(directory / "robustness_results.json", payload)
    headers = ["Seed", "F1", "ROC-AUC", "MAE (ns)"]
    rows = []
    for seed_row in payload["per_seed"]:
        block = seed_row["summary"]["endpoint"]
        rows.append([str(seed_row["seed"]), fmt(block["f1"]), fmt(block["roc_auc"]), fmt(block["mae"])])
    across = payload["across_seeds"]["endpoint"]
    lines = [
        "# Robustness",
        "",
        "Full multitask leave-one-design-out at seeds 0, 1, and 2. "
        "The spread is the sample standard deviation of the three leave-one-design-out means.",
        "",
        markdown_table(headers, rows, numeric_from=1),
        "",
        (
            f"Endpoint F1 mean {fmt(across['f1'])}, spread {fmt(across['f1']['std'])}. "
            f"ROC-AUC mean {fmt(across['roc_auc'])}, spread {fmt(across['roc_auc']['std'])}. "
            f"MAE mean {fmt(across['mae'])} ns, spread {fmt(across['mae']['std'])} ns."
        ),
        "",
        "## Per-fold support",
        "",
        markdown_table(
            ["Held-out", "Endpoint support", "Expanded support", "F1 mean", "F1 spread", "ROC-AUC mean", "ROC-AUC spread", "MAE mean (ns)", "MAE spread (ns)"],
            [
                [
                    str(row["held_out"]),
                    "n/a" if row["endpoint_support"] is None else str(int(row["endpoint_support"])),
                    "n/a" if row["expanded_support"] is None else str(int(row["expanded_support"])),
                    fmt(row["f1"]),
                    fmt(row["f1"]["std"]),
                    fmt(row["roc_auc"]),
                    fmt(row["roc_auc"]["std"]),
                    fmt(row["mae"]),
                    fmt(row["mae"]["std"]),
                ]
                for row in payload["per_fold"]
            ],
        ),
        "",
        "Missing designs:",
        "",
        missing_lines(payload["missing_designs"]),
        "",
        f"Runtime: {fmt(payload['runtime_seconds'], 1)} s.",
        "",
    ]
    write_text(directory / "robustness_results.md", "\n".join(lines))


def _write_scaling(payload: dict[str, object]) -> None:
    directory = reports_dir()
    write_json(directory / "scaling_results.json", payload)
    headers = ["Training designs", "Count", "F1", "ROC-AUC", "MAE (ns)", "Expanded F1", "Expanded MAE (ns)", "Endpoint support"]
    rows = []
    for point in payload["points"]:
        rows.append(
            [
                ", ".join(point["training_designs"]),
                str(point["training_design_count"]),
                fmt(point["endpoint"]["f1"]),
                fmt(point["endpoint"]["roc_auc"]),
                fmt(point["endpoint"]["mae"]),
                fmt(point["expanded"]["f1"]),
                fmt(point["expanded"]["mae"]),
                str(int(point["endpoint"]["support"])),
            ]
        )
    lines = [
        "# Scaling",
        "",
        f"Held-out design: `{payload['held_out']}`. {payload['held_out_reason']}",
        "",
        payload["order"],
        "",
        markdown_table(headers, rows, numeric_from=1) if rows else "No scaling points were measured.",
        "",
        "Missing designs:",
        "",
        missing_lines(payload["missing_designs"]),
        "",
        f"Runtime: {fmt(payload['runtime_seconds'], 1)} s.",
        "",
    ]
    write_text(directory / "scaling_results.md", "\n".join(lines))
