# SlackGNN research report

<p>
  <img alt="Earlier experiment" src="https://img.shields.io/badge/record-four--design%20v2.0-8c959f?style=flat-square">
  <img alt="Superseded by v3.0" src="https://img.shields.io/badge/superseded%20by-v3.0%20six--design%20study-c47b2b?style=flat-square">
</p>

> [!WARNING]
> **Historical record.** This page documents an earlier stage of SlackGNN and is kept as the record of that run. The published v3.0 result is the six-design study in [slackgnn_final_research_report.md](slackgnn_final_research_report.md): F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, RMSE 0.8667 ns.

## Corpus statistics

The corpus is the four published OpenROAD designs. Node and edge totals below match the PyG graphs. Logic depth and reported-path membership are computed by the feature builder.

| Design | Nodes | Edges | Labels | Max logic depth | Reported-path nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| aes | 15868 | 38215 | 4 | 19 | 20 |
| gcd | 527 | 928 | 5 | 14 | 27 |
| ibex | 18000 | 44436 | 7 | 47 | 42 |
| jpeg | 36283 | 82643 | 6 | 27 | 23 |

Totals: 70678 nodes and 166222 edges. The label count remains 22 endpoints.

## Dataset limitations

The finish report prints a handful of paths, so 22 of 70678 nodes have a slack target. Each leave-one-design-out test fold has four to seven labels. Arrival time and required time exist only at those endpoints. Published edges do not name the SPEF net, so the delay channel is a fanout proxy. One IBEX endpoint keeps the worse of two slack values.

## Architecture design

SlackGNN encodes node structure, edge connection type, and a timing context of logic depth plus reported-path membership. Two GraphSAGE-style layers aggregate drivers into loads with attention. The attention score uses timing criticality, path membership, and a learned slack-magnitude gate. Ground-truth slack, arrival time, and required time stay out of the input. A classification head and a regression head share the embedding. The joint loss is alpha times binary cross-entropy plus beta times smooth L1, with alpha 1.0 and beta 0.5.

The architecture diagram and feature definitions are in `docs/slackgnn_architecture.md`.

## Baseline comparison

### SlackGNN evaluation

Leave-one-design-out mean. SlackGNN numbers are from the multitask model when that task was trained.
Classification class 1 is a timing violation. Slack error is in nanoseconds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcn | 0.3649 | 0.3083 | 0.8750 | 0.4435 | 0.6708 | 1.3702 | 1.6721 |
| graphsage | 0.4839 | 0.3375 | 0.8750 | 0.4810 | 0.6875 | 1.2860 | 1.5759 |
| gin | 0.4006 | 0.2841 | 0.7500 | 0.4339 | 0.6833 | 1.2245 | 1.4986 |
| **slackgnn** | **0.9143** | **0.9167** | **0.8750** | **0.8667** | **0.9000** | **0.7530** | **0.9748** |

Multitask SlackGNN is above GCN, GraphSAGE, and GIN on F1 (0.8667) and ROC-AUC (0.9000), and its slack MAE (0.7530 ns) is below all three.

## SlackGNN multitask folds

| Held-out design | Accuracy | F1 | ROC-AUC | MAE | RMSE | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 1.0000 | 1.0000 | 1.0000 | 0.4013 | 0.5224 | 4 |
| gcd | 0.8000 | 0.8000 | 1.0000 | 0.5118 | 0.6794 | 5 |
| ibex | 0.8571 | 0.6667 | 0.6000 | 1.2864 | 1.5622 | 7 |
| jpeg | 1.0000 | 1.0000 | 1.0000 | 0.8127 | 1.1353 | 6 |


## SlackGNN performance

Multitask SlackGNN is above GCN, GraphSAGE, and GIN on F1 (0.8667) and ROC-AUC (0.9000), and its slack MAE (0.7530 ns) is below all three.

The per-fold table is in `reports/slackgnn_results.md`. Attention rankings are in `reports/explainability/`.

## Failure modes

A model can rank a violation above timing-met endpoints and still have low precision, because the positive class has six examples. Folds with one violation move F1 and ROC-AUC when that single endpoint changes score. Predicting the training-set mean slack can look competitive on MAE when every design's slack values sit in a similar range. Attention on a shortest signal-flow route is not the cell-by-cell timing path printed by OpenROAD, so a high-attention gate is a structural neighbor of a reported endpoint rather than a proven critical cell.

## Future work

1. Label more endpoints than the printed worst paths.
2. Join SPEF resistance and capacitance once edges carry a net name.
3. Repeat leave-one-design-out over several seeds before ranking architectures.
4. Add a path readout so the loss sees every gate on a reported path, not only the endpoint.
