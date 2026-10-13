# SlackGNN research highlights

## Problem

Static timing analysis tells a designer whether a placed-and-routed netlist meets its clock. The OpenROAD finish report prints a small set of paths. Across six SkyWater 130 nm designs, that report yields 32 endpoints on 79,856 gates. A model that sees only those endpoints, and that is tested on a design it never trained on, has to transfer timing behavior through circuit structure.

## Method

SlackGNN is a node-level graph network on the driver-to-load netlist. Message passing runs on every gate. The loss uses labeled nodes. The encoder receives structural node features, edge features, logic depth, and reported-path membership. Arrival time, required time, and slack stay out of the input. Two attention layers feed a violation head and a slack head. Training uses 173 route labels obtained by copying each reported slack along the shortest signal-flow route. Evaluation scores the original 32 endpoints. GCN, GraphSAGE, and GIN are trained on the same holdout with one-hot node type only.

## Results

Leave-one-design-out endpoint scores:

| Model | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: |
| GCN | 0.5800 | 0.6243 | 0.9621 | 1.1922 |
| GraphSAGE | 0.3333 | 0.7632 | 0.8704 | 1.1053 |
| GIN | 0.5222 | 0.6951 | 0.9470 | 1.1712 |
| SlackGNN | 0.8076 | 0.9583 | 0.6919 | 0.8667 |

SlackGNN has the best F1, the best ROC-AUC, and the best slack error. It remains ahead of the baselines down to 25% of the training labels, through 20% label noise, and after half of the training paths are removed. It collapses at 10% labels. Path membership and timing depth are the parts that carry the gain. Training time tracks edge count. jpeg dominates the epoch.

## Impact

The study shows that a multitask graph model can rank timing endpoints on an unseen OpenROAD block when it is trained on the reported paths of other blocks. The practical requirement is a corpus of about six designs and more than a tenth of the expanded labels. The scientific boundary is equally clear: 32 printed endpoints are enough to compare these models, and they are too few to claim timing signoff. The next design worth adding is one whose finish report contributes new endpoints.
