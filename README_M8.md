# SlackGNN phase M8

<p>
  <img alt="Earlier experiment" src="https://img.shields.io/badge/record-phase%20M8-8c959f?style=flat-square">
  <img alt="Superseded by v3.0" src="https://img.shields.io/badge/superseded%20by-v3.0%20six--design%20study-c47b2b?style=flat-square">
</p>

> [!WARNING]
> **Historical record.** This page documents an earlier stage of SlackGNN and is kept as the record of that run. The published v3.0 result is the six-design study in [docs/slackgnn_final_research_report.md](docs/slackgnn_final_research_report.md): F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, RMSE 0.8667 ns.

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

Corpus: 70678 nodes, 166222 edges, 22 labels.

Multitask SlackGNN is above GCN, GraphSAGE, and GIN on F1 (0.8667) and ROC-AUC (0.9000), and its slack MAE (0.7530 ns) is below all three.

The comparison table is in `reports/slackgnn_results.md`.

## Future roadmap

1. Expand endpoint labels beyond the finish-report paths.
2. Attach SPEF parasitics as edge features.
3. Estimate uncertainty across seeds.
4. Add a path-level readout on top of the node embedding.
