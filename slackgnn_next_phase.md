# SlackGNN next phase

<p>
  <img alt="Earlier experiment" src="https://img.shields.io/badge/record-baseline%20trainer%20note-8c959f?style=flat-square">
  <img alt="Superseded by v3.0" src="https://img.shields.io/badge/superseded%20by-v3.0%20six--design%20study-c47b2b?style=flat-square">
</p>

> [!WARNING]
> **Historical record.** This page documents an earlier stage of SlackGNN and is kept as the record of that run. The published v3.0 result is the six-design study in [slackgnn_final_research_report.md](slackgnn_final_research_report.md): F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, RMSE 0.8667 ns.

## Current corpus statistics

The training set is the published PyG corpus. Each graph is one placed-and-routed design. Node features are the four-way node-type encoding. Slack is the prediction target and is not an input feature.

| Design | Nodes | Edges | Labels | Timing met | Violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| aes | 15868 | 38215 | 4 | 3 | 1 |
| gcd | 527 | 928 | 5 | 3 | 2 |
| ibex | 18000 | 44436 | 7 | 5 | 2 |
| jpeg | 36283 | 82643 | 6 | 5 | 1 |

Totals: 4 designs, 70678 nodes, 166222 edges, 22 labeled endpoints, 16 timing-met labels, and 6 timing-violation labels.

## Baseline model performance

GCN, GraphSAGE, and GIN were trained with masked node loss. Message passing uses the full graph. The loss uses only labeled endpoints. The table is the mean of leave-one-design-out folds. Slack error is in nanoseconds.

| Model | F1 | ROC-AUC | Slack MAE | Slack RMSE |
| --- | ---: | ---: | ---: | ---: |
| gcn | 0.4435 | 0.6708 | 1.3702 | 1.6721 |
| graphsage | 0.4810 | 0.6875 | 1.2860 | 1.5759 |
| gin | 0.4339 | 0.6833 | 1.2245 | 1.4986 |

Fold-level numbers are in `reports/model_results.md`. A random split of the same 22 labels is reported beside the design holdout. The design holdout is the result that matters: the test design is absent from training.

## Weaknesses

The label set is 22 endpoints out of 70678 nodes. OpenROAD's finish report prints a small set of paths, so most endpoints have no target. A model can look accurate by predicting the majority class, timing met, and still miss the few violations.

Each leave-one-design-out test fold has between four and seven labels. Precision, recall, F1, and ROC-AUC move sharply when one endpoint changes. These baselines are a measurement of the current corpus, not a claim that the architecture is ready for timing signoff.

Node features do not include pin capacitance, net parasitics, or path depth. Two nodes of the same type are identical at the input, so the network has to recover timing structure only from connectivity. One IBEX endpoint also carries both a minimum-path slack and a maximum-path slack; the graph stores the worse slack, so the other path is invisible to the loss.

## Proposed SlackGNN architecture

> [!NOTE]
> This proposal predates SlackGNN. The model that shipped is described in [slackgnn_architecture.md](slackgnn_architecture.md).

SlackGNN should stay a node model on the full netlist graph, with three additions over these baselines.

1. Edge features from the SPEF: resistance and capacitance on each driver-to-load edge, so delay has a physical input.
2. A timing head with two outputs: violation logit and slack. Train the violation head with a class-weighted loss and the slack head with an error that penalizes optimistic slack.
3. A path readout that pools the nodes on each labeled path, so the loss can see the path rather than only its endpoint.

The message-passing backbone can start from the GraphSAGE baseline, because its neighborhood aggregation is stable on these directed netlist graphs. Design identity should not be an input feature. Leave-one-design-out remains the evaluation protocol.

## Recommended next milestones

1. Expand labels beyond the printed worst paths, so each design has many more endpoints with slack.
2. Attach SPEF edge features without changing the current node-type encoding.
3. Implement the two-head SlackGNN and compare it with these three baselines under the same design holdout.
4. Report confidence intervals over seeds, because a single run on 22 labels is too small to rank architectures.
