"""Train SlackGNN, compare it with the baselines, and write the M8 reports."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from slackgnn.explainability.rankings import explain_design
from slackgnn.features.feature_builder import enrich_corpus
from slackgnn.paths import data_dir, repo_root
from slackgnn.training.metrics import classification_metrics, mean_metric, regression_metrics
from slackgnn.training.models.slackgnn import SlackGNN
from slackgnn.training.slackgnn_figures import write_slackgnn_figures
from slackgnn.training.splits import leave_one_design_out
from slackgnn.training.train import run_lodo
from slackgnn.training.validate_corpus import reports_dir

logger = logging.getLogger(__name__)

Pair = tuple[int, int]
TASKS = ("classification", "regression", "multitask")


def run_slackgnn(
    data_root: str | Path | None = None,
    epochs: int = 40,
    hidden_channels: int = 48,
    learning_rate: float = 1e-2,
    seed: int = 0,
    tasks: tuple[str, ...] = TASKS,
    alpha: float = 1.0,
    beta: float = 0.5,
    compare_baselines: bool = True,
) -> int:
    """Enrich the corpus, train SlackGNN under leave-one-design-out, and report."""
    graphs, summary, errors = enrich_corpus(data_root or data_dir())
    if errors or graphs is None or summary is None:
        for message in errors:
            logger.error("%s", message)
        return 1
    folds = leave_one_design_out(graphs, seed)
    trained: dict[str, dict[str, object]] = {}
    explanations: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    for task in tasks:
        logger.info("Training SlackGNN task %s", task)
        trained[task] = _fit_task(graphs, folds, task, epochs, hidden_channels, learning_rate, seed, alpha, beta)
        logger.info(
            "SlackGNN %s leave-one-design-out F1 %s ROC-AUC %s MAE %s",
            task,
            trained[task]["classification"]["mean"]["f1"],
            trained[task]["classification"]["mean"]["roc_auc"],
            trained[task]["regression"]["mean"]["mae"],
        )
        if task == "multitask" or "multitask" not in tasks:
            explanations, embedding_rows = _explain_folds(graphs, trained[task]["models"], folds)
    baselines = {"classification": {}, "regression": {}}
    if compare_baselines:
        logger.info("Training baseline leave-one-design-out comparison")
        baselines = run_lodo(graphs, epochs=epochs, hidden_channels=min(hidden_channels, 32), learning_rate=learning_rate, seed=seed)
    payload = _public_payload(summary, trained, baselines, alpha, beta, seed, epochs, hidden_channels)
    _write_reports(payload, explanations)
    classification_rows = _prediction_list(trained, "classification")
    regression_rows = _prediction_list(trained, "regression")
    write_slackgnn_figures(
        embedding_rows,
        explanations,
        classification_rows,
        regression_rows,
        baselines["classification"],
        reports_dir() / "figures",
        reports_dir() / "explainability",
    )
    _write_documents(summary, payload)
    logger.info("SlackGNN evaluation finished. Outperforms a baseline: %s", payload["outperforms_a_baseline"])
    return 0


def _fit_task(
    graphs: list[Data],
    folds: list[dict[str, object]],
    task: str,
    epochs: int,
    hidden_channels: int,
    learning_rate: float,
    seed: int,
    alpha: float,
    beta: float,
) -> dict[str, object]:
    class_rows: list[dict[str, object]] = []
    reg_rows: list[dict[str, object]] = []
    class_predictions: list[dict[str, object]] = []
    reg_predictions: list[dict[str, object]] = []
    models: list[nn.Module] = []
    for fold in folds:
        logger.info("SlackGNN %s with %s held out", task, fold["held_out"])
        model = _train_one(graphs, fold["train"], fold["validation"], task, epochs, hidden_channels, learning_rate, seed, alpha, beta)
        models.append(model)
        class_score, class_rows_fold = _evaluate(graphs, model, fold["test"], "classification")
        reg_score, reg_rows_fold = _evaluate(graphs, model, fold["test"], "regression")
        class_score["held_out"] = fold["held_out"]
        reg_score["held_out"] = fold["held_out"]
        class_rows.append(class_score)
        reg_rows.append(reg_score)
        class_predictions.extend(class_rows_fold)
        reg_predictions.extend(reg_rows_fold)
    return {
        "models": models,
        "classification": {
            "folds": class_rows,
            "mean": {name: mean_metric(class_rows, name) for name in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "predictions": class_predictions,
        },
        "regression": {
            "folds": reg_rows,
            "mean": {name: mean_metric(reg_rows, name) for name in ("mae", "mse", "rmse")},
            "predictions": reg_predictions,
        },
    }


def _train_one(
    graphs: list[Data],
    train_pairs: list[Pair],
    validation_pairs: list[Pair],
    task: str,
    epochs: int,
    hidden_channels: int,
    learning_rate: float,
    seed: int,
    alpha: float,
    beta: float,
) -> SlackGNN:
    torch.manual_seed(seed)
    random.seed(seed)
    model = SlackGNN(
        int(graphs[0].node_features.size(1)),
        int(graphs[0].edge_attr.size(1)),
        int(graphs[0].timing_features.size(1)),
        hidden_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = _positive_weight(graphs, train_pairs)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    patience = 0
    for _epoch in range(epochs):
        model.train()
        _epoch_loss(model, graphs, train_pairs, task, optimizer, pos_weight, alpha, beta)
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if not validation_pairs:
            best_state = snapshot
            continue
        model.eval()
        with torch.no_grad():
            loss = _epoch_loss(model, graphs, validation_pairs, task, None, pos_weight, alpha, beta)
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
    model: SlackGNN,
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
        logits, slack, _attention, _embedding = model(data.node_features, data.edge_index, data.edge_attr, data.timing_features)
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
        data.y_violation[index],
        pos_weight=pos_weight,
    )
    regression = nn.functional.smooth_l1_loss(slack[index], data.y_slack[index])
    if task == "classification":
        return classification
    if task == "regression":
        return regression
    return alpha * classification + beta * regression


def _evaluate(
    graphs: list[Data],
    model: SlackGNN,
    pairs: list[Pair],
    task: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _prediction_rows(graphs, model, pairs, task)
    if task == "classification":
        return classification_metrics([int(row["label"]) for row in rows], [float(row["score"]) for row in rows]), rows
    return regression_metrics([float(row["slack"]) for row in rows], [float(row["prediction"]) for row in rows]), rows


def _prediction_rows(graphs: list[Data], model: SlackGNN, pairs: list[Pair], task: str) -> list[dict[str, object]]:
    model.eval()
    grouped: dict[int, list[int]] = {}
    for graph_index, node_index in pairs:
        grouped.setdefault(graph_index, []).append(node_index)
    cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for graph_index in grouped:
            data = graphs[graph_index]
            logits, slack, _attention, _embedding = model(
                data.node_features, data.edge_index, data.edge_attr, data.timing_features
            )
            cached[graph_index] = (logits, slack)
    rows: list[dict[str, object]] = []
    for graph_index, node_index in pairs:
        logits, slack = cached[graph_index]
        data = graphs[graph_index]
        row: dict[str, object] = {
            "design": str(data.design_name),
            "node": int(node_index),
            "slack": float(data.y_slack[node_index].item()),
        }
        if task == "classification":
            row["label"] = int(data.y_violation[node_index].item())
            row["score"] = float(torch.sigmoid(logits[node_index]).item())
        else:
            row["prediction"] = float(slack[node_index].item())
        rows.append(row)
    return rows


def _explain_folds(
    graphs: list[Data],
    models: list[nn.Module],
    folds: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    explanations: list[dict[str, object]] = []
    embeddings: list[dict[str, object]] = []
    for model, fold in zip(models, folds, strict=True):
        model.eval()
        held_out = str(fold["held_out"])
        graph_index = next(index for index, data in enumerate(graphs) if str(data.design_name) == held_out)
        data = graphs[graph_index]
        with torch.no_grad():
            _logits, _slack, attention, embedding = model(
                data.node_features, data.edge_index, data.edge_attr, data.timing_features
            )
        explanations.append(explain_design(data, attention.cpu()))
        embeddings.extend(_embedding_sample(data, embedding.cpu(), graph_index))
    return explanations, embeddings


def _embedding_sample(data: Data, embedding: torch.Tensor, seed: int) -> list[dict[str, object]]:
    labeled = set(data.label_mask.nonzero(as_tuple=False).view(-1).tolist())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(data.num_nodes, generator=generator).tolist()
    chosen = list(labeled)
    for node in order:
        if len(chosen) >= 80:
            break
        if node not in labeled:
            chosen.append(node)
    rows: list[dict[str, object]] = []
    for node in chosen:
        rows.append(
            {
                "design": str(data.design_name),
                "node": data.node_names[node],
                "labeled": node in labeled,
                "embedding": embedding[node].tolist(),
            }
        )
    return rows


def _positive_weight(graphs: list[Data], pairs: list[Pair]) -> torch.Tensor:
    violations = sum(int(graphs[graph_index].y_violation[node_index].item()) for graph_index, node_index in pairs)
    met = len(pairs) - violations
    weight = met / violations if violations else 1.0
    return torch.tensor(weight, dtype=torch.float32)


def _prediction_list(trained: dict[str, dict[str, object]], kind: str) -> list[dict[str, object]]:
    source = trained["multitask"] if "multitask" in trained else next(iter(trained.values()))
    return list(source[kind]["predictions"])


def _public_payload(
    summary: dict[str, object],
    trained: dict[str, dict[str, object]],
    baselines: dict[str, dict[str, object]],
    alpha: float,
    beta: float,
    seed: int,
    epochs: int,
    hidden_channels: int,
) -> dict[str, object]:
    slackgnn_source = trained["multitask"] if "multitask" in trained else next(iter(trained.values()))
    comparison_class = {name: result["mean"] for name, result in baselines["classification"].items()}
    comparison_reg = {name: result["mean"] for name, result in baselines["regression"].items()}
    comparison_class["slackgnn"] = slackgnn_source["classification"]["mean"]
    comparison_reg["slackgnn"] = slackgnn_source["regression"]["mean"]
    evidence = _evidence(comparison_class, comparison_reg)
    return {
        "protocol": "leave_one_design_out",
        "seed": seed,
        "epochs": epochs,
        "hidden_channels": hidden_channels,
        "alpha": alpha,
        "beta": beta,
        "reported_slackgnn_task": "multitask" if "multitask" in trained else next(iter(trained)),
        "corpus": {
            "designs": summary["designs"],
            "total_nodes": summary["total_nodes"],
            "total_edges": summary["total_edges"],
        },
        "slackgnn": {task: _without_models(result) for task, result in trained.items()},
        "baselines": {
            "classification": {name: _strip_predictions(result) for name, result in baselines["classification"].items()},
            "regression": {name: _strip_predictions(result) for name, result in baselines["regression"].items()},
        },
        "comparison": {"classification": comparison_class, "regression": comparison_reg},
        "outperforms_a_baseline": evidence[0],
        "evidence": evidence[1],
    }


def _evidence(classification: dict[str, object], regression: dict[str, object]) -> tuple[bool, str]:
    slack_f1 = _metric(classification["slackgnn"], "f1")
    slack_auc = _metric(classification["slackgnn"], "roc_auc")
    slack_mae = _metric(regression["slackgnn"], "mae")
    reasons: list[str] = []
    for name in classification:
        if name == "slackgnn":
            continue
        if slack_f1 is not None and _metric(classification[name], "f1") is not None and slack_f1 > float(_metric(classification[name], "f1")):
            reasons.append(f"F1 {slack_f1:.4f} is above {name} F1 {float(_metric(classification[name], 'f1')):.4f}")
        if slack_auc is not None and _metric(classification[name], "roc_auc") is not None and slack_auc > float(_metric(classification[name], "roc_auc")):
            reasons.append(f"ROC-AUC {slack_auc:.4f} is above {name} ROC-AUC {float(_metric(classification[name], 'roc_auc')):.4f}")
        if slack_mae is not None and _metric(regression[name], "mae") is not None and slack_mae < float(_metric(regression[name], "mae")):
            reasons.append(f"slack MAE {slack_mae:.4f} ns is below {name} MAE {float(_metric(regression[name], 'mae')):.4f} ns")
    if not reasons:
        return False, "SlackGNN did not beat a baseline on F1, ROC-AUC, or slack MAE in this run."
    names = [name for name in classification if name != "slackgnn"]
    beats_all_f1 = all(
        slack_f1 is not None and _metric(classification[name], "f1") is not None and slack_f1 > float(_metric(classification[name], "f1"))
        for name in names
    )
    beats_all_auc = all(
        slack_auc is not None and _metric(classification[name], "roc_auc") is not None and slack_auc > float(_metric(classification[name], "roc_auc"))
        for name in names
    )
    beats_all_mae = all(
        slack_mae is not None and _metric(regression[name], "mae") is not None and slack_mae < float(_metric(regression[name], "mae"))
        for name in names
    )
    if beats_all_f1 and beats_all_auc and beats_all_mae:
        return True, (
            f"Multitask SlackGNN is above GCN, GraphSAGE, and GIN on F1 ({slack_f1:.4f}) and ROC-AUC ({slack_auc:.4f}), "
            f"and its slack MAE ({slack_mae:.4f} ns) is below all three."
        )
    return True, "; ".join(reasons)


def _metric(result: object, key: str) -> float | None:
    if not isinstance(result, dict):
        return None
    value = result.get(key)
    if value is None:
        return None
    return float(value)


def _without_models(result: dict[str, object]) -> dict[str, object]:
    return {
        "classification": _strip_predictions(result["classification"]),
        "regression": _strip_predictions(result["regression"]),
    }


def _strip_predictions(result: object) -> object:
    if not isinstance(result, dict):
        return result
    return {key: _strip_predictions(value) if isinstance(value, dict) else value for key, value in result.items() if key != "predictions"}


def _write_reports(payload: dict[str, object], explanations: list[dict[str, object]]) -> None:
    root = reports_dir()
    explain = root / "explainability"
    explain.mkdir(parents=True, exist_ok=True)
    (root / "slackgnn_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    (root / "slackgnn_results.md").write_text(_results_markdown(payload), encoding="utf-8", newline="\n")
    (explain / "node_importance.json").write_text(
        json.dumps([{ "design_name": item["design_name"], "node_importance": item["node_importance"] } for item in explanations], indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (explain / "critical_gate_rankings.json").write_text(
        json.dumps([{ "design_name": item["design_name"], "critical_gate_rankings": item["critical_gate_rankings"] } for item in explanations], indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (explain / "influential_paths.json").write_text(
        json.dumps([{ "design_name": item["design_name"], "influential_paths": item["influential_paths"] } for item in explanations], indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (explain / "attention_heatmaps.json").write_text(
        json.dumps([{ "design_name": item["design_name"], "attention_heatmap": item["attention_heatmap"] } for item in explanations], indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _results_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# SlackGNN evaluation",
        "",
        "Leave-one-design-out mean. SlackGNN numbers are from the multitask model when that task was trained.",
        "Classification class 1 is a timing violation. Slack error is in nanoseconds.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = list(payload["comparison"]["classification"])
    for name in names:
        classification = payload["comparison"]["classification"][name]
        regression = payload["comparison"]["regression"][name]
        lines.append(
            "| {name} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {mae} | {rmse} |".format(
                name=name,
                accuracy=_fmt(classification.get("accuracy")),
                precision=_fmt(classification.get("precision")),
                recall=_fmt(classification.get("recall")),
                f1=_fmt(classification.get("f1")),
                roc_auc=_fmt(classification.get("roc_auc")),
                mae=_fmt(regression.get("mae")),
                rmse=_fmt(regression.get("rmse")),
            )
        )
    lines.extend(["", payload["evidence"], ""])
    slackgnn = payload["slackgnn"]
    if "multitask" in slackgnn:
        lines.extend(["## SlackGNN multitask folds", ""])
        lines.extend(["| Held-out design | Accuracy | F1 | ROC-AUC | MAE | RMSE | Support |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        class_folds = slackgnn["multitask"]["classification"]["folds"]
        reg_folds = slackgnn["multitask"]["regression"]["folds"]
        for class_fold, reg_fold in zip(class_folds, reg_folds, strict=True):
            lines.append(
                "| {held} | {accuracy} | {f1} | {auc} | {mae} | {rmse} | {support} |".format(
                    held=class_fold["held_out"],
                    accuracy=_fmt(class_fold["accuracy"]),
                    f1=_fmt(class_fold["f1"]),
                    auc=_fmt(class_fold["roc_auc"]),
                    mae=_fmt(reg_fold["mae"]),
                    rmse=_fmt(reg_fold["rmse"]),
                    support=class_fold["support"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _write_documents(summary: dict[str, object], payload: dict[str, object]) -> None:
    report = repo_root() / "docs" / "slackgnn_research_report.md"
    readme = repo_root() / "README_M8.md"
    report.write_text(_research_report(summary, payload), encoding="utf-8", newline="\n")
    readme.write_text(_readme(summary, payload), encoding="utf-8", newline="\n")
    logger.info("Wrote %s", report)
    logger.info("Wrote %s", readme)


def _research_report(summary: dict[str, object], payload: dict[str, object]) -> str:
    design_rows = []
    for item in summary["designs_detail"]:
        design_rows.append(
            f"| {item['design_name']} | {item['node_count']} | {item['edge_count']} | {item['label_count']} | {item['max_logic_depth']} | {item['critical_path_nodes']} |"
        )
    table = _results_markdown(payload)
    return f"""# SlackGNN research report

## Corpus statistics

The corpus is the four published OpenROAD designs. Node and edge totals below match the PyG graphs. Logic depth and reported-path membership are computed by the feature builder.

| Design | Nodes | Edges | Labels | Max logic depth | Reported-path nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(design_rows)}

Totals: {summary['total_nodes']} nodes and {summary['total_edges']} edges. The label count remains 22 endpoints.

## Dataset limitations

The finish report prints a handful of paths, so 22 of {summary['total_nodes']} nodes have a slack target. Each leave-one-design-out test fold has four to seven labels. Arrival time and required time exist only at those endpoints. Published edges do not name the SPEF net, so the delay channel is a fanout proxy. One IBEX endpoint keeps the worse of two slack values.

## Architecture design

SlackGNN encodes node structure, edge connection type, and a timing context of logic depth plus reported-path membership. Two GraphSAGE-style layers aggregate drivers into loads with attention. The attention score uses timing criticality, path membership, and a learned slack-magnitude gate. Ground-truth slack, arrival time, and required time stay out of the input. A classification head and a regression head share the embedding. The joint loss is alpha times binary cross-entropy plus beta times smooth L1, with alpha {payload['alpha']} and beta {payload['beta']}.

The architecture diagram and feature definitions are in `docs/slackgnn_architecture.md`.

## Baseline comparison

{table}

## SlackGNN performance

{payload['evidence']}

The per-fold table is in `reports/slackgnn_results.md`. Attention rankings are in `reports/explainability/`.

## Failure modes

A model can rank a violation above timing-met endpoints and still have low precision, because the positive class has six examples. Folds with one violation move F1 and ROC-AUC when that single endpoint changes score. Predicting the training-set mean slack can look competitive on MAE when every design's slack values sit in a similar range. Attention on a shortest signal-flow route is not the cell-by-cell timing path printed by OpenROAD, so a high-attention gate is a structural neighbor of a reported endpoint rather than a proven critical cell.

## Future work

1. Label more endpoints than the printed worst paths.
2. Join SPEF resistance and capacitance once edges carry a net name.
3. Repeat leave-one-design-out over several seeds before ranking architectures.
4. Add a path readout so the loss sees every gate on a reported path, not only the endpoint.
"""


def _readme(summary: dict[str, object], payload: dict[str, object]) -> str:
    return f"""# SlackGNN phase M8

SlackGNN is a timing-aware graph network for endpoint slack and timing-violation prediction. It trains on the published `gcd`, `aes`, `ibex`, and `jpeg` graphs. This phase does not change the dataset-generation pipeline.

## Architecture overview

Node, edge, and timing encoders feed two attention-weighted GraphSAGE layers. One head predicts a violation logit. The other predicts slack in nanoseconds. The timing encoder sees logic depth and reported-path membership. Arrival time, required time, and slack are targets, not inputs.

```text
node features -- node encoder --+
timing context -- timing encoder --+--> GraphSAGE attention layers --> violation logit
edge features -- edge encoder -----+                                 --> slack
```

Details and diagrams: `docs/slackgnn_architecture.md`.

## Training

From the repository root, with the Python 3.11 environment:

```text
python scripts/train.py --model slackgnn
python scripts/train.py --model slackgnn --task multitask
python scripts/train.py --model slackgnn --task classification
python scripts/train.py --model slackgnn --task regression
python scripts/train.py --model slackgnn --alpha 1 --beta 0.5 --hidden 48 --epochs 40
```

Leave-one-design-out rotates the held-out design through `gcd`, `aes`, `ibex`, and `jpeg`. Training on `gcd`, `aes`, and `ibex` with `jpeg` held out is one of those four folds.

## Evaluation

The training command writes:

- `reports/feature_summary.json`
- `reports/slackgnn_results.json`
- `reports/slackgnn_results.md`
- `reports/explainability/`
- `reports/figures/` embeddings, attention heatmaps, ROC comparison, slack scatter, and error histogram

Classification metrics are accuracy, precision, recall, F1, and ROC-AUC. Regression metrics are MAE, MSE, and RMSE.

## Results summary

Corpus: {summary['total_nodes']} nodes, {summary['total_edges']} edges, 22 labels.

{payload['evidence']}

The comparison table is in `reports/slackgnn_results.md`.

## Future roadmap

1. Expand endpoint labels beyond the finish-report paths.
2. Attach SPEF parasitics as edge features.
3. Estimate uncertainty across seeds.
4. Add a path-level readout on top of the node embedding.
"""


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
