"""Feature ablations for the six-design SlackGNN holdout.

Timing input columns are logic depth, reported-path membership, and source
distance. Path membership is column 1. Timing criticality is the two depth
columns. Edge features and attention use the existing ablation switches.
Published graphs and the M9.4 generalization report are not rewritten.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from slackgnn.m9.reporting import fmt, metric_cells, write_json, write_text
from slackgnn.m9.trainer import train_lodo
from slackgnn.paths import repo_root

logger = logging.getLogger(__name__)

FEATURE_ABLATIONS = (
    {
        "name": "full",
        "description": "Learned attention, path membership, timing depth, and edge features.",
        "drop_edges": False,
        "uniform_attention": False,
        "zero_timing_columns": (),
    },
    {
        "name": "no_attention",
        "description": "Incoming driver messages use equal weights.",
        "drop_edges": False,
        "uniform_attention": True,
        "zero_timing_columns": (),
    },
    {
        "name": "no_path_membership",
        "description": "The reported-path membership channel is zero. Depth channels stay.",
        "drop_edges": False,
        "uniform_attention": False,
        "zero_timing_columns": (1,),
    },
    {
        "name": "no_timing_criticality",
        "description": "Logic depth and source distance are zero. Path membership stays.",
        "drop_edges": False,
        "uniform_attention": False,
        "zero_timing_columns": (0, 2),
    },
    {
        "name": "no_edge_features",
        "description": "Edge channels are zero. Attention still uses node and timing state.",
        "drop_edges": True,
        "uniform_attention": False,
        "zero_timing_columns": (),
    },
)


def run_feature_ablations(
    graphs: list[Data],
    inventory_payload: dict[str, object],
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Run each feature ablation with every design held out once."""
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for spec in FEATURE_ABLATIONS:
        logger.info("Feature ablation %s", spec["name"])
        trained = train_lodo(
            graphs,
            seed=seed,
            task="multitask",
            drop_edges=bool(spec["drop_edges"]),
            uniform_attention=bool(spec["uniform_attention"]),
            zero_timing_columns=tuple(spec["zero_timing_columns"]),
        )
        rows.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "drop_edges": spec["drop_edges"],
                "uniform_attention": spec["uniform_attention"],
                "zero_timing_columns": list(spec["zero_timing_columns"]),
                "folds": [
                    {
                        "held_out": fold["held_out"],
                        "endpoint": fold["endpoint"],
                        "expanded": fold["expanded"],
                    }
                    for fold in trained["folds"]
                ],
                "summary": trained["summary"],
            }
        )
    return {
        "stage": "M9.5",
        "protocol": "leave_one_design_out",
        "training_loss": "expanded labels",
        "designs": [str(graph.design_name) for graph in graphs],
        "missing_designs": list(inventory_payload.get("missing", [])),
        "seed": seed,
        "variants": rows,
        "comparison": compare_with_full(rows),
        "runtime_seconds": time.perf_counter() - started,
        "note": (
            "Endpoint metrics use published path-table slack. "
            "Path membership is timing column 1. Timing criticality is logic depth and source distance."
        ),
    }


def compare_with_full(variants: list[dict[str, object]]) -> dict[str, object]:
    """Measure how far each ablation moves from the full model on endpoint F1 and MAE."""
    full = next(row for row in variants if row["name"] == "full")
    full_f1 = _mean(full, "f1")
    full_mae = _mean(full, "mae")
    drops: list[dict[str, object]] = []
    for row in variants:
        if row["name"] == "full":
            continue
        f1 = _mean(row, "f1")
        mae = _mean(row, "mae")
        drops.append(
            {
                "name": row["name"],
                "endpoint_f1": f1,
                "endpoint_mae": mae,
                "f1_change": None if f1 is None or full_f1 is None else f1 - full_f1,
                "mae_change": None if mae is None or full_mae is None else mae - full_mae,
            }
        )
    worst_f1 = min(drops, key=lambda item: item["f1_change"] if item["f1_change"] is not None else 0)
    worst_mae = max(drops, key=lambda item: item["mae_change"] if item["mae_change"] is not None else 0)
    return {
        "full_endpoint_f1": full_f1,
        "full_endpoint_mae": full_mae,
        "variants": drops,
        "largest_f1_drop": worst_f1["name"],
        "largest_mae_increase": worst_mae["name"],
    }


def write_feature_ablation_report(payload: dict[str, object], directory: str | Path | None = None) -> tuple[Path, Path]:
    """Write the M9.5 JSON and Markdown reports."""
    destination = Path(directory) if directory is not None else repo_root() / "reports"
    json_path = write_json(destination / "ablation_results.json", payload)
    markdown_path = write_text(destination / "ablation_results.md", render_feature_ablation_markdown(payload))
    return json_path, markdown_path


def write_feature_ablation_figure(payload: dict[str, object], directory: str | Path | None = None) -> Path:
    """Plot endpoint F1 and MAE for each ablation."""
    figure_dir = Path(directory) if directory is not None else repo_root() / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "ablation_f1_mae.png"
    names = [str(row["name"]) for row in payload["variants"]]
    f1 = [_or_nan(_mean(row, "f1")) for row in payload["variants"]]
    mae = [_or_nan(_mean(row, "mae")) for row in payload["variants"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    axes[0].bar(names, f1, color="#1f4e79")
    axes[1].bar(names, mae, color="#c47b2b")
    axes[0].set_ylabel("Endpoint F1")
    axes[1].set_ylabel("Endpoint MAE (ns)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def render_feature_ablation_markdown(payload: dict[str, object]) -> str:
    """Format the ablation means and the comparison with the full model."""
    comparison = payload["comparison"]
    lines = [
        "# SlackGNN feature ablations",
        "",
        "Leave-one-design-out on the expanded labels. Endpoint metrics use published slack.",
        "",
        payload["note"],
        "",
        f"Designs: {', '.join(payload['designs'])}. Seed {payload['seed']}.",
        "",
        "## Endpoint means",
        "",
        _table(payload, "endpoint"),
        "",
        "## Expanded-label means",
        "",
        _table(payload, "expanded"),
        "",
        "## Change from the full model",
        "",
        "| Variant | Endpoint F1 | F1 change | Endpoint MAE (ns) | MAE change (ns) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["variants"]:
        lines.append(
            "| {name} | {f1} | {f1_change} | {mae} | {mae_change} |".format(
                name=row["name"],
                f1=fmt(row["endpoint_f1"]),
                f1_change=fmt(row["f1_change"]),
                mae=fmt(row["endpoint_mae"]),
                mae_change=fmt(row["mae_change"]),
            )
        )
    lines.extend(
        [
            "",
            f"Largest endpoint F1 drop: {comparison['largest_f1_drop']}.",
            f"Largest endpoint MAE increase: {comparison['largest_mae_increase']}.",
            "",
            "F1 means skip a holdout when that model predicts no violations, so those means can cover different designs. "
            "ROC-AUC and MAE are computed on every holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def _table(payload: dict[str, object], scope: str) -> str:
    lines = [
        "| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variants"]:
        cells = metric_cells(row["summary"][scope])[:-1]
        lines.append("| " + " | ".join([str(row["name"]), *cells]) + " |")
    return "\n".join(lines)


def _mean(row: dict[str, object], key: str) -> float | None:
    summary = row["summary"]["endpoint"][key]
    if isinstance(summary, dict):
        value = summary.get("mean")
    else:
        value = summary
    if value is None:
        return None
    return float(value)


def _or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    return value
