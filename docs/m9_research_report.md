# SlackGNN phase M9 research report

This report uses the measurements written under `reports/m9/`. The previously published four-design SlackGNN headline was F1 0.8667, ROC-AUC 0.9000, and MAE 0.7530 ns on 22 endpoint labels. Those figures are the earlier report, not a result recomputed here. Every number below is taken from the M9 artifacts.

## Corpus

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

## Label expansion

Each reported startpoint-to-endpoint pair donates its slack to the nodes on the shortest signal-flow route. Shared nodes keep the minimum slack. Published endpoint slack from graph.pt is restored afterward. Propagated slack is not a new static-timing measurement.

The limitation is built into the method. A node that is not an endpoint receives the slack of a reported path because it lies on the shortest driver-to-load route from that path's startpoint to its endpoint. That value was not printed by static timing analysis for the intermediate node. Where the hop-shortest route is not the pin sequence in the timing report, the auxiliary target follows the hop-shortest route anyway. A node on several routes keeps the minimum slack. After propagation, every published endpoint is set back to the slack stored in `graph.pt`. Report-only paths across the measured designs: 0. Auxiliary nodes added beyond the published endpoints: 90.

Training may use the auxiliary targets. Test tables report the published endpoints and the expanded targets separately so the effect of the copied slack stays visible.

## Cross-design results

Designs in the rotation: gcd, aes, ibex, jpeg.
Endpoint mean F1 0.6750 (fold spread 0.2754), ROC-AUC 0.9500 (fold spread 0.1000), MAE 0.9890 ns (fold spread 0.6676 ns).
Expanded-label mean F1 0.6741, ROC-AUC 0.8117, MAE 0.9180 ns. These expanded scores reuse slack that was copied along each path.

| Held-out | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) | Endpoint support | Expanded F1 | Expanded MAE (ns) | Expanded support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 1.0000 | 0.6342 | 5 | 0.7895 | 0.8734 | 27 |
| aes | 1.0000 | 1.0000 | 0.2298 | 4 | 0.9091 | 0.3555 | 20 |
| ibex | 0.5000 | 0.8000 | 1.4612 | 7 | 0.7755 | 1.1529 | 42 |
| jpeg | 0.4000 | 1.0000 | 1.6308 | 6 | 0.2222 | 1.2903 | 23 |

## Ablations

Highest mean endpoint F1. Equal F1 is broken by lower endpoint MAE.

| Ablation | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) |
| --- | ---: | ---: | ---: |
| full | 0.6750 | 0.9500 | 0.9890 |
| no_timing_context | 0.5333 | 0.9083 | 1.0251 |
| no_edge_features | 0.7167 | 0.9000 | 0.9488 |
| no_attention | 0.7167 | 0.9333 | 0.9936 |
| classification_only | 0.8111 | 1.0000 | 1.2542 |
| regression_only | n/a | 0.0833 | 1.0302 |

The highest mean endpoint F1 is `classification_only` at 0.8111, with endpoint MAE 1.2542 ns.

## Robustness

Across seeds 0, 1, 2, endpoint F1 mean 0.7583, spread 0.0755, range 0.6750 to 0.8222.
Endpoint ROC-AUC mean 0.9417, spread 0.0144, range 0.9250 to 0.9500.
Endpoint MAE mean 0.9203 ns, spread 0.0746 ns, range 0.8409 to 0.9890 ns.

Per-fold endpoint support is the number of published labels in the held-out design. It does not change with the seed.

| Held-out | Endpoint support | Expanded support | F1 mean | F1 spread | MAE mean (ns) | MAE spread (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 5 | 27 | 0.7556 | 0.0770 | 0.6492 | 0.0878 |
| aes | 4 | 20 | 1.0000 | 0.0000 | 0.3655 | 0.1263 |
| ibex | 7 | 42 | 0.5370 | 0.1156 | 1.4547 | 0.1172 |
| jpeg | 6 | 23 | 0.7000 | 0.4243 | 1.2116 | 0.3638 |

## Scaling

Held-out design: `jpeg`. jpeg is present and has both endpoint classes
Training designs are added from the smallest graph to the largest.

| Training designs | Count | Endpoint F1 | Endpoint ROC-AUC | Endpoint MAE (ns) | Expanded F1 | Expanded MAE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd, aes | 2 | 1.0000 | 1.0000 | 0.9049 | 0.7500 | 0.9654 |
| gcd, aes, ibex | 3 | 0.4000 | 1.0000 | 1.6308 | 0.2222 | 1.2903 |

From 2 training designs to 3, endpoint F1 moved from 1.0000 to 0.4000 and endpoint MAE moved from 0.9049 ns to 1.6308 ns.

## Failure modes

- The published endpoint set is still the finish report's worst paths. A model can rank those few pins well and still say nothing about the unlabeled endpoints.
- Expanded metrics look at nodes that were given a path's slack by construction. High expanded F1 can appear because neighboring nodes share a copied target. The endpoint table is the measurement that still refers to printed slack.
- The hop-shortest route can differ from the timed pin sequence, so some auxiliary targets sit on gates the report never listed.
- Leave-one-design-out folds contain only the held-out design's printed endpoints. With support this small, one pin moves F1 and ROC-AUC sharply. Seed spread is the check on that instability.
- `riscv32i` did not enter the study: design directory is missing.
- `microwatt` did not enter the study: design directory is missing.
- `chameleon` did not enter the study: design directory is missing.
- Held-out `gcd` has endpoint support 5 and expanded support 27.
- Held-out `aes` has endpoint support 4 and expanded support 20.
- Held-out `ibex` has endpoint support 7 and expanded support 42.
- Held-out `jpeg` has endpoint support 6 and expanded support 23.

## What real label growth requires

Path-node copying does not create new static-timing measurements. A larger supervised set requires STA reports that print slack for many more endpoints, for example a finish report with a higher path count per group or an endpoint slack report that covers the register and output pins. Those values would then be joined to the graph as endpoint targets. The three designs named for this corpus that are not in the measurement, when they are missing, also need a finished placement and routing run before they can enter leave-one-design-out. This phase did not start that flow.
