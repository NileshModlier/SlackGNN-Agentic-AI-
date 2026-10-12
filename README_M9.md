# SlackGNN phase M9

M9 expands supervision by copying reported slack along signal-flow routes, then measures leave-one-design-out SlackGNN on the designs that already have `pyg/graph.pt`. It does not modify the dataset builders or the M8 model file, and it does not start an OpenROAD flow.

## Reproduce

From the repository root, with the Python 3.11 environment that has the project dependencies:

```text
python scripts/expand_labels.py
python scripts/cross_design_benchmark.py
python scripts/ablation_study.py
python scripts/robustness_study.py
python scripts/scaling_study.py
python scripts/m9_figures.py
python scripts/write_m9_docs.py
```

`scripts/run_m9_studies.py` runs the same stages in one process and reuses the full multitask seed-0 model for the ablation and robustness tables. Running the stage scripts separately retrains that model.

Focused tests that do not need the corpus:

```text
python -m pytest tests/test_label_expansion.py tests/test_m9_metrics.py tests/test_m9_ablation.py
```

Training is CPU-only. The study uses hidden width 32, at most 20 epochs, and early stopping patience 6, because message passing covers the full netlist.

## Result summary

Available graphs: 4. Nodes: 70678. Edges: 166222. Published endpoint labels: 22. Expanded labels: 112.

| Design | Nodes | Edges | Endpoint labels | Expanded labels | Added | Report-only paths |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 527 | 928 | 5 | 27 | 22 | 0 |
| aes | 15868 | 38215 | 4 | 20 | 16 | 0 |
| ibex | 18000 | 44436 | 7 | 42 | 35 | 0 |
| jpeg | 36283 | 82643 | 6 | 23 | 17 | 0 |

Designs requested for this corpus that were not measured:
- `riscv32i`: design directory is missing
- `microwatt`: design directory is missing
- `chameleon`: design directory is missing

Cross-design endpoint means: F1 0.6750, ROC-AUC 0.9500, MAE 0.9890 ns, MSE 1.9571, RMSE 1.1962 ns.

Ablation winner: classification_only (endpoint F1 0.8111, MAE 1.2542 ns).

Robustness of the endpoint leave-one-design-out means across seeds 0, 1, and 2: F1 0.7583 ± 0.0755, ROC-AUC 0.9417 ± 0.0144, MAE 0.9203 ± 0.0746 ns.

Held-out design: `jpeg`. jpeg is present and has both endpoint classes
Training designs are added from the smallest graph to the largest.

| Training designs | Count | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) | Expanded F1 | Expanded MAE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd, aes | 2 | 1.0000 | 1.0000 | 0.9049 | 0.7500 | 0.9654 |
| gcd, aes, ibex | 3 | 0.4000 | 1.0000 | 1.6308 | 0.2222 | 1.2903 |

From 2 training designs to 3, endpoint F1 moved from 1.0000 to 0.4000 and endpoint MAE moved from 0.9049 ns to 1.6308 ns.

This phase does not create a git tag and does not push.

## File index

- `src/slackgnn/expansion/label_expansion.py` propagates path slack and restores published endpoint labels.
- `src/slackgnn/m9/ablation.py` wraps SlackGNN for the six ablations.
- `src/slackgnn/m9/trainer.py` trains on expanded labels and scores endpoint and expanded targets separately.
- `src/slackgnn/m9/studies.py` writes the benchmark, ablation, robustness, and scaling reports.
- `src/slackgnn/m9/figures.py` writes `reports/m9/figures/`.
- `scripts/expand_labels.py`
- `scripts/cross_design_benchmark.py`
- `scripts/ablation_study.py`
- `scripts/robustness_study.py`
- `scripts/scaling_study.py`
- `scripts/m9_figures.py`
- `scripts/write_m9_docs.py`
- `scripts/run_m9_studies.py`
- `reports/m9/expanded_labels.json`
- `reports/m9/expanded_labels.pt`
- `reports/m9/cross_design_results.json`
- `reports/m9/cross_design_results.md`
- `reports/m9/ablation_results.json`
- `reports/m9/ablation_results.md`
- `reports/m9/robustness_results.json`
- `reports/m9/robustness_results.md`
- `reports/m9/scaling_results.json`
- `reports/m9/scaling_results.md`
- `reports/m9/figures/label_expansion.png`
- `reports/m9/figures/cross_design_f1_mae.png`
- `reports/m9/figures/ablation.png`
- `reports/m9/figures/robustness.png`
- `reports/m9/figures/scaling.png`
- `reports/m9/figures/roc_curves.png`
- `reports/m9/figures/slack_scatter.png`
- `docs/m9_research_report.md`
- `README_M9.md`
- `tests/test_label_expansion.py`
- `tests/test_m9_metrics.py`
- `tests/test_m9_ablation.py`
