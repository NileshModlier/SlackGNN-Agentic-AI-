"""Write the M9 research report and release note from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path

from slackgnn.m9.reporting import fmt
from slackgnn.paths import repo_root


def write_m9_documents(report_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``docs/m9_research_report.md`` and ``README_M9.md``."""
    directory = Path(report_dir) if report_dir is not None else repo_root() / "reports" / "m9"
    expansion = _read(directory / "expanded_labels.json")
    cross = _read(directory / "cross_design_results.json")
    ablation = _read(directory / "ablation_results.json")
    robustness = _read(directory / "robustness_results.json")
    scaling = _read(directory / "scaling_results.json")
    report = _research_report(expansion, cross, ablation, robustness, scaling)
    readme = _readme(expansion, cross, ablation, robustness, scaling)
    docs = repo_root() / "docs" / "m9_research_report.md"
    note = repo_root() / "README_M9.md"
    docs.write_text(report, encoding="utf-8", newline="\n")
    note.write_text(readme, encoding="utf-8", newline="\n")
    return docs, note


def _research_report(
    expansion: dict[str, object] | None,
    cross: dict[str, object] | None,
    ablation: dict[str, object] | None,
    robustness: dict[str, object] | None,
    scaling: dict[str, object] | None,
) -> str:
    sections = [
        "# SlackGNN phase M9 research report",
        "",
        "This report uses the measurements written under `reports/m9/`. "
        "The previously published four-design SlackGNN headline was F1 0.8667, ROC-AUC 0.9000, "
        "and MAE 0.7530 ns on 22 endpoint labels. Those figures are the earlier report, not a result recomputed here. "
        "Every number below is taken from the M9 artifacts.",
        "",
        "## Corpus",
        "",
        _corpus_section(expansion),
        "",
        "## Label expansion",
        "",
        _expansion_section(expansion),
        "",
        "## Cross-design results",
        "",
        _cross_section(cross),
        "",
        "## Ablations",
        "",
        _ablation_section(ablation),
        "",
        "## Robustness",
        "",
        _robustness_section(robustness),
        "",
        "## Scaling",
        "",
        _scaling_section(scaling),
        "",
        "## Failure modes",
        "",
        _failures(expansion, cross, robustness),
        "",
        "## What real label growth requires",
        "",
        "Path-node copying does not create new static-timing measurements. "
        "A larger supervised set requires STA reports that print slack for many more endpoints, "
        "for example a finish report with a higher path count per group or an endpoint slack report that covers the register and output pins. "
        "Those values would then be joined to the graph as endpoint targets. "
        "The three designs named for this corpus that are not in the measurement, when they are missing, also need a finished placement and routing run before they can enter leave-one-design-out. "
        "This phase did not start that flow.",
        "",
    ]
    return "\n".join(sections)


def _readme(
    expansion: dict[str, object] | None,
    cross: dict[str, object] | None,
    ablation: dict[str, object] | None,
    robustness: dict[str, object] | None,
    scaling: dict[str, object] | None,
) -> str:
    endpoint = _mean_phrase(cross, "endpoint") if cross else "not measured"
    winner = "not measured"
    if ablation and ablation.get("winner"):
        winner = (
            f"{ablation['winner']['name']} "
            f"(endpoint F1 {fmt(ablation['winner']['endpoint_f1'])}, "
            f"MAE {fmt(ablation['winner']['endpoint_mae'])} ns)"
        )
    spread = "not measured"
    if robustness:
        across = robustness["across_seeds"]["endpoint"]
        spread = (
            f"F1 {fmt(across['f1']['mean'])} ± {fmt(across['f1']['std'])}, "
            f"ROC-AUC {fmt(across['roc_auc']['mean'])} ± {fmt(across['roc_auc']['std'])}, "
            f"MAE {fmt(across['mae']['mean'])} ± {fmt(across['mae']['std'])} ns"
        )
    return "\n".join(
        [
            "# SlackGNN phase M9",
            "",
            "M9 expands supervision by copying reported slack along signal-flow routes, then measures leave-one-design-out SlackGNN on the designs that already have `pyg/graph.pt`. "
            "It does not modify the dataset builders or the M8 model file, and it does not start an OpenROAD flow.",
            "",
            "## Reproduce",
            "",
            "From the repository root, with the Python 3.11 environment that has the project dependencies:",
            "",
            "```text",
            "python scripts/expand_labels.py",
            "python scripts/cross_design_benchmark.py",
            "python scripts/ablation_study.py",
            "python scripts/robustness_study.py",
            "python scripts/scaling_study.py",
            "python scripts/m9_figures.py",
            "python scripts/write_m9_docs.py",
            "```",
            "",
            "`scripts/run_m9_studies.py` runs the same stages in one process and reuses the full multitask seed-0 model for the ablation and robustness tables. "
            "Running the stage scripts separately retrains that model.",
            "",
            "Focused tests that do not need the corpus:",
            "",
            "```text",
            "python -m pytest tests/test_label_expansion.py tests/test_m9_metrics.py tests/test_m9_ablation.py",
            "```",
            "",
            "Training is CPU-only. The study uses hidden width 32, at most 20 epochs, and early stopping patience 6, because message passing covers the full netlist.",
            "",
            "## Result summary",
            "",
            _corpus_section(expansion),
            "",
            f"Cross-design endpoint means: {endpoint}.",
            "",
            f"Ablation winner: {winner}.",
            "",
            f"Robustness of the endpoint leave-one-design-out means across seeds 0, 1, and 2: {spread}.",
            "",
            _scaling_section(scaling),
            "",
            "This phase does not create a git tag and does not push.",
            "",
            "## File index",
            "",
            "- `src/slackgnn/expansion/label_expansion.py` propagates path slack and restores published endpoint labels.",
            "- `src/slackgnn/m9/ablation.py` wraps SlackGNN for the six ablations.",
            "- `src/slackgnn/m9/trainer.py` trains on expanded labels and scores endpoint and expanded targets separately.",
            "- `src/slackgnn/m9/studies.py` writes the benchmark, ablation, robustness, and scaling reports.",
            "- `src/slackgnn/m9/figures.py` writes `reports/m9/figures/`.",
            "- `scripts/expand_labels.py`",
            "- `scripts/cross_design_benchmark.py`",
            "- `scripts/ablation_study.py`",
            "- `scripts/robustness_study.py`",
            "- `scripts/scaling_study.py`",
            "- `scripts/m9_figures.py`",
            "- `scripts/write_m9_docs.py`",
            "- `scripts/run_m9_studies.py`",
            "- `reports/m9/expanded_labels.json`",
            "- `reports/m9/expanded_labels.pt`",
            "- `reports/m9/cross_design_results.json`",
            "- `reports/m9/cross_design_results.md`",
            "- `reports/m9/ablation_results.json`",
            "- `reports/m9/ablation_results.md`",
            "- `reports/m9/robustness_results.json`",
            "- `reports/m9/robustness_results.md`",
            "- `reports/m9/scaling_results.json`",
            "- `reports/m9/scaling_results.md`",
            "- `reports/m9/figures/label_expansion.png`",
            "- `reports/m9/figures/cross_design_f1_mae.png`",
            "- `reports/m9/figures/ablation.png`",
            "- `reports/m9/figures/robustness.png`",
            "- `reports/m9/figures/scaling.png`",
            "- `reports/m9/figures/roc_curves.png`",
            "- `reports/m9/figures/slack_scatter.png`",
            "- `docs/m9_research_report.md`",
            "- `README_M9.md`",
            "- `tests/test_label_expansion.py`",
            "- `tests/test_m9_metrics.py`",
            "- `tests/test_m9_ablation.py`",
            "",
        ]
    )


def _corpus_section(expansion: dict[str, object] | None) -> str:
    if expansion is None:
        return "Label expansion was not measured."
    summary = expansion["summary"]
    lines = [
        (
            f"Available graphs: {summary['design_count']}. "
            f"Nodes: {summary['node_count']}. Edges: {summary['edge_count']}. "
            f"Published endpoint labels: {summary['original_label_count']}. "
            f"Expanded labels: {summary['expanded_label_count']}."
        ),
        "",
        "| Design | Nodes | Edges | Endpoint labels | Expanded labels | Added | Report-only paths |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in expansion["designs"]:
        if not row.get("present", True):
            lines.append(f"| {row['design_name']} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["design_name"]),
                    str(row.get("node_count", "n/a")),
                    str(row.get("edge_count", "n/a")),
                    str(row["original_label_count"]),
                    str(row["expanded_label_count"]),
                    str(row["added_label_count"]),
                    str(row.get("paths_only_in_report", "n/a")),
                ]
            )
            + " |"
        )
    missing = expansion.get("missing_designs") or []
    if missing:
        lines.extend(["", "Designs requested for this corpus that were not measured:"])
        for item in missing:
            lines.append(f"- `{item['design']}`: {item['reason']}")
    return "\n".join(lines)


def _expansion_section(expansion: dict[str, object] | None) -> str:
    if expansion is None:
        return "Label expansion was not measured."
    summary = expansion["summary"]
    return "\n".join(
        [
            summary["method"],
            "",
            "The limitation is built into the method. A node that is not an endpoint receives the slack of a reported path because it lies on the shortest driver-to-load route from that path's startpoint to its endpoint. "
            "That value was not printed by static timing analysis for the intermediate node. "
            "Where the hop-shortest route is not the pin sequence in the timing report, the auxiliary target follows the hop-shortest route anyway. "
            "A node on several routes keeps the minimum slack. "
            "After propagation, every published endpoint is set back to the slack stored in `graph.pt`. "
            f"Report-only paths across the measured designs: {summary['paths_only_in_report']}. "
            f"Auxiliary nodes added beyond the published endpoints: {summary['added_label_count']}.",
            "",
            "Training may use the auxiliary targets. Test tables report the published endpoints and the expanded targets separately so the effect of the copied slack stays visible.",
        ]
    )


def _cross_section(cross: dict[str, object] | None) -> str:
    if cross is None:
        return "The cross-design benchmark was not measured."
    endpoint = cross["summary"]["endpoint"]
    expanded = cross["summary"]["expanded"]
    lines = [
        f"Designs in the rotation: {', '.join(cross['designs']) or 'none'}.",
        (
            f"Endpoint mean F1 {fmt(endpoint['f1'])} (fold spread {fmt(endpoint['f1']['std'])}), "
            f"ROC-AUC {fmt(endpoint['roc_auc'])} (fold spread {fmt(endpoint['roc_auc']['std'])}), "
            f"MAE {fmt(endpoint['mae'])} ns (fold spread {fmt(endpoint['mae']['std'])} ns)."
        ),
        (
            f"Expanded-label mean F1 {fmt(expanded['f1'])}, "
            f"ROC-AUC {fmt(expanded['roc_auc'])}, "
            f"MAE {fmt(expanded['mae'])} ns. "
            "These expanded scores reuse slack that was copied along each path."
        ),
        "",
        "| Held-out | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) | Endpoint support | Expanded F1 | Expanded MAE (ns) | Expanded support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in cross["folds"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(fold["held_out"]),
                    fmt(fold["endpoint"]["f1"]),
                    fmt(fold["endpoint"]["roc_auc"]),
                    fmt(fold["endpoint"]["mae"]),
                    str(fold["endpoint"]["support"]),
                    fmt(fold["expanded"]["f1"]),
                    fmt(fold["expanded"]["mae"]),
                    str(fold["expanded"]["support"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _ablation_section(ablation: dict[str, object] | None) -> str:
    if ablation is None:
        return "Ablations were not measured."
    lines = [
        ablation["winner_rule"],
        "",
        "| Ablation | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ablation["ablations"]:
        block = row["summary"]["endpoint"]
        lines.append(
            f"| {row['name']} | {fmt(block['f1'])} | {fmt(block['roc_auc'])} | {fmt(block['mae'])} |"
        )
    winner = ablation.get("winner")
    if winner is None:
        lines.append("")
        lines.append("No ablation had a defined endpoint F1.")
    else:
        lines.append("")
        lines.append(
            f"The highest mean endpoint F1 is `{winner['name']}` at {fmt(winner['endpoint_f1'])}, "
            f"with endpoint MAE {fmt(winner['endpoint_mae'])} ns."
        )
    return "\n".join(lines)


def _robustness_section(robustness: dict[str, object] | None) -> str:
    if robustness is None:
        return "Robustness was not measured."
    across = robustness["across_seeds"]["endpoint"]
    lines = [
        (
            f"Across seeds {', '.join(str(seed) for seed in robustness['seeds'])}, "
            f"endpoint F1 mean {fmt(across['f1']['mean'])}, spread {fmt(across['f1']['std'])}, "
            f"range {fmt(across['f1']['min'])} to {fmt(across['f1']['max'])}."
        ),
        (
            f"Endpoint ROC-AUC mean {fmt(across['roc_auc']['mean'])}, spread {fmt(across['roc_auc']['std'])}, "
            f"range {fmt(across['roc_auc']['min'])} to {fmt(across['roc_auc']['max'])}."
        ),
        (
            f"Endpoint MAE mean {fmt(across['mae']['mean'])} ns, spread {fmt(across['mae']['std'])} ns, "
            f"range {fmt(across['mae']['min'])} to {fmt(across['mae']['max'])} ns."
        ),
        "",
        "Per-fold endpoint support is the number of published labels in the held-out design. It does not change with the seed.",
        "",
        "| Held-out | Endpoint support | Expanded support | F1 mean | F1 spread | MAE mean (ns) | MAE spread (ns) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in robustness["per_fold"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["held_out"]),
                    str(row["endpoint_support"]),
                    str(row["expanded_support"]),
                    fmt(row["f1"]),
                    fmt(row["f1"]["std"]),
                    fmt(row["mae"]),
                    fmt(row["mae"]["std"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _scaling_section(scaling: dict[str, object] | None) -> str:
    if scaling is None:
        return "Scaling was not measured."
    lines = [
        f"Held-out design: `{scaling['held_out']}`. {scaling['held_out_reason']}",
        scaling["order"],
        "",
    ]
    points = scaling["points"]
    if not points:
        lines.append("There were fewer than two training designs, so no growth curve was measured.")
        return "\n".join(lines)
    lines.extend(
        [
            "| Training designs | Count | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) | Expanded F1 | Expanded MAE (ns) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in points:
        lines.append(
            "| "
            + " | ".join(
                [
                    ", ".join(point["training_designs"]),
                    str(point["training_design_count"]),
                    fmt(point["endpoint"]["f1"]),
                    fmt(point["endpoint"]["roc_auc"]),
                    fmt(point["endpoint"]["mae"]),
                    fmt(point["expanded"]["f1"]),
                    fmt(point["expanded"]["mae"]),
                ]
            )
            + " |"
        )
    first = points[0]
    last = points[-1]
    lines.extend(
        [
            "",
            (
                f"From {first['training_design_count']} training designs to {last['training_design_count']}, "
                f"endpoint F1 moved from {fmt(first['endpoint']['f1'])} to {fmt(last['endpoint']['f1'])} "
                f"and endpoint MAE moved from {fmt(first['endpoint']['mae'])} ns to {fmt(last['endpoint']['mae'])} ns."
            ),
        ]
    )
    return "\n".join(lines)


def _failures(
    expansion: dict[str, object] | None,
    cross: dict[str, object] | None,
    robustness: dict[str, object] | None,
) -> str:
    lines = [
        "The published endpoint set is still the finish report's worst paths. A model can rank those few pins well and still say nothing about the unlabeled endpoints.",
        "Expanded metrics look at nodes that were given a path's slack by construction. High expanded F1 can appear because neighboring nodes share a copied target. The endpoint table is the measurement that still refers to printed slack.",
        "The hop-shortest route can differ from the timed pin sequence, so some auxiliary targets sit on gates the report never listed.",
        "Leave-one-design-out folds contain only the held-out design's printed endpoints. With support this small, one pin moves F1 and ROC-AUC sharply. Seed spread is the check on that instability.",
    ]
    if expansion:
        for item in expansion.get("missing_designs") or []:
            lines.append(f"`{item['design']}` did not enter the study: {item['reason']}.")
    if cross:
        for fold in cross["folds"]:
            if fold["endpoint"].get("roc_auc") is None:
                lines.append(f"Endpoint ROC-AUC is undefined for held-out `{fold['held_out']}` because one class is absent.")
    if robustness:
        for row in robustness["per_fold"]:
            lines.append(
                f"Held-out `{row['held_out']}` has endpoint support {row['endpoint_support']} and expanded support {row['expanded_support']}."
            )
    return "\n".join(f"- {line}" for line in lines)


def _mean_phrase(cross: dict[str, object], scope: str) -> str:
    block = cross["summary"][scope]
    return (
        f"F1 {fmt(block['f1'])}, ROC-AUC {fmt(block['roc_auc'])}, "
        f"MAE {fmt(block['mae'])} ns, MSE {fmt(block['mse'])}, RMSE {fmt(block['rmse'])} ns"
    )


def _read(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
