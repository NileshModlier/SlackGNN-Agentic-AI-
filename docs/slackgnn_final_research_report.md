# SlackGNN: cross-design timing-slack prediction on OpenROAD netlists

## Abstract

Timing closure depends on static timing analysis of a placed-and-routed netlist. The analysis is exact for the paths it reports, and a finish report from a production flow prints only a small set of those paths. This study asks whether a graph neural network can estimate endpoint slack on a design it has never trained on, using the reported paths of other designs as supervision. The corpus is six OpenROAD designs implemented on the SkyWater 130 nm high-density platform: gcd, aes, ibex, jpeg, riscv32i, and chameleon. The graphs contain 79,856 nodes and 189,028 directed driver-to-load edges. The finish reports contribute 32 published endpoints. Copying each reported slack along the shortest signal-flow route raises the training supervision to 173 nodes. Those copied values are the training target. Endpoint scores still use the path table.

SlackGNN is a two-layer attention model on this graph. It reads structural node features, edge features, logic depth, and reported-path membership. Arrival time, required time, and slack are withheld from the encoder and used only as targets. A shared embedding feeds a violation classifier and a slack regressor. The training loss is a weighted sum of binary cross-entropy and smooth L1. GCN, GraphSAGE, and GIN baselines use the published one-hot node type, with separate classification and regression heads. Every model is trained leave-one-design-out at hidden width 32, epoch cap 20, and seed 0.

SlackGNN records endpoint F1 0.8076, ROC-AUC 0.9583, mean absolute error 0.6919 ns, and root mean square error 0.8667 ns. F1, ROC-AUC, and MAE are better than every baseline mean on the published endpoints. aes is the easiest holdout and ibex is the hardest. Ablations attribute the gain to path membership and to timing depth. Membership is the only removal that lowers both F1 and MAE. Depth removal drops ROC-AUC from 0.9583 to 0.8722 and raises MAE to 0.7360 ns. The model holds through 25% of the expanded labels, through 20% synthetic label noise, and after half of the training paths are removed. It collapses at 10% labels. Six designs are required before slack error stays within 0.10 ns of the full-corpus result. Training time follows the edge count, with jpeg responsible for about 38% of a SlackGNN epoch.

The evidence supports a screening model for unseen blocks that resemble this corpus. Thirty-two endpoints can rank these architectures. Certifying a clock still belongs to the timer.

## 1. Introduction

Digital implementation turns a register-transfer description into a gate-level netlist whose wires have been placed and routed. A clock constraint then becomes a question about path delay. Static timing analysis answers that question for the paths it traces. Designers read worst-path reports because those reports name the endpoints that miss the clock and the endpoints that barely meet it. The same reports are a sparse label set for learning. Most gates never appear as an endpoint in the printed table, so a node-level model is supervised on a thin slice of a large graph.

The practical question is transfer. A model that memorizes the endpoints of gcd, aes, ibex, jpeg, and riscv32i is useful only if it says something true about chameleon, and the same rotation must work for every other held-out block. Random splits of endpoints inside one design answer a different question. They let neighboring gates from the same netlist appear on both sides of the split. Leave-one-design-out closes that leak. The test graph contributes no training labels. Message passing still runs on the full test netlist, because an unlabeled gate can carry structural context to a labeled endpoint, and the loss ignores every node the report did not supervise.

This study builds that experiment on an open flow. OpenROAD-flow-scripts produces a finish netlist, constraints, parasitics, and a timing report for each design. A graph extractor flattens the netlist into a directed signal-flow graph. A PyTorch Geometric dataset stores node type and endpoint slack. An auxiliary expansion copies reported slack onto the shortest route between each reported startpoint and endpoint so the training loss sees more than the single endpoint pin. Evaluation returns to the published endpoints, so a good expanded-label score cannot replace a good endpoint score.

SlackGNN is the model proposed for this graph. It keeps the baselines' discipline about leakage, and it adds the features a timing estimator needs once slack itself is forbidden as an input: how deep a gate sits in the logic, whether it lies on a reported path, how the driver and the load are typed, and a learned attention weight over incoming drivers. The comparison is against GCN, GraphSAGE, and GIN trained on the same splits.

The contributions are empirical and are tied to this corpus.

1. A six-design OpenROAD corpus with a documented path from finish artifacts to PyTorch Geometric graphs, containing 79,856 nodes, 189,028 edges, 32 endpoints, and 173 expanded labels.
2. A multitask graph model that predicts violation and slack without reading arrival time, required time, or slack as inputs.
3. A leave-one-design-out comparison in which SlackGNN reaches endpoint F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, and RMSE 0.8667 ns, ahead of the three baselines on F1, ROC-AUC, and MAE.
4. Ablations, robustness sweeps, and scaling measurements that locate the gain in path membership and timing depth, place the collapse at 10% labels, and describe runtime as closest to linear in edge count.

The remainder of the paper describes the timing problem, the dataset pipeline, the corpus, the architecture, the protocol, and the four result families. Threats to validity and future work state the boundary of the claim.

## 2. Background

### 2.1 Static timing analysis

Static timing analysis computes arrival times and required times on a timing graph. A sequential element or a primary output launches or captures a data transition. Between the launch and the capture, combinational logic and interconnect add delay. For a setup check, slack is the difference between the time the data is required to be stable and the time it arrives. A non-negative slack meets the constraint. A negative slack misses it. The clock period, clock latency, uncertainty, and external input and output delays set the required time. Cell delays and wire delays set the arrival time.

Two properties of this computation matter for learning. First, slack is a path property that the timer reports at an endpoint. Many gates lie on the path and never receive their own slack number in a worst-path report. Second, the numerical slack already contains the answer a predictor would like to emit. A model that receives arrival time, required time, or slack at the endpoint can reconstruct the label by arithmetic. Any honest timing predictor has to build its estimate from structure, from constraints that are not the endpoint slack, and from path context that does not include the slack value.

Worst negative slack and total negative slack summarize a design in one or two numbers. They are useful checks on a flow. They are too coarse to supervise a node model. The path table is the row-level label. Each row names a startpoint, an endpoint, a path group, and a slack. OpenROAD finish reports print a handful of such rows, often the worst paths, and sometimes reprint the same path. After duplicate paths are collapsed, a design in this corpus has between four and seven unique endpoints.

The classification target used here follows the sign of slack. The positive class is a violation, defined as slack strictly below zero. A slack of zero meets the constraint. The regression target is the slack in nanoseconds. Both targets are defined only where the report, or the later route expansion, assigned a value. Every other node participates in message passing and is excluded from the loss.

### 2.2 OpenROAD

OpenROAD is an open implementation flow for digital layout. OpenROAD-flow-scripts wraps synthesis, floorplanning, placement, clock-tree synthesis, routing, and finish for a chosen platform and a chosen RTL design. This study uses the SkyWater 130 nm high-density platform and the base flow variant. The flow is a means of producing consistent artifacts. The learned model does not replace the timer inside the flow, and the study does not rerun placement or routing to manufacture extra labels.

The artifacts that enter the dataset are the finish netlist, the finish constraints, the finish parasitics, and the finish timing report. The netlist is structural Verilog. Hierarchical modules have been elaborated into standard cells and, where the design requires them, into black boxes such as memories. The constraints file records clocks and external delays in the Synopsys Design Constraints language. The parasitics file is SPEF, with resistance in ohms and capacitance in picofarads on the routed nets. The timing report lists path checks with slack marked met or violated.

These four files answer different parts of the timing question. The netlist supplies connectivity. The constraints supply the clock budget. The parasitics supply the routed interconnect. The report supplies the slack a static timing engine computed from the other three. A graph built from the netlist alone knows who drives whom. It does not know the clock period unless the constraints are read, and it does not know the slack unless the report is read. This study reads the report for labels and reads the netlist for the graph. The parasitic resistance and capacitance remain in the flow directory and in the imported SPEF. They are not joined onto edges in the published feature tensor. The edge delay channel used by SlackGNN is a structural proxy, the logarithm of driver fanout, normalized inside each design. That choice is a limitation and is stated again when the features are defined.

The designs themselves span a wide size range and a wide range of RTL intent. gcd is a small arithmetic block. aes is a cryptographic datapath. ibex is a RISC-V core. jpeg is an image codec and the largest graph. riscv32i is a smaller processor. chameleon is a system-on-chip whose elaborated top module is soc_core, with memories and an interconnect around a core. All six were finished under the same platform family so that node-type vocabularies and library timing would not be mixed with a second process technology. A seventh candidate, microwatt, does not appear in the published totals because its design directory is absent. The six-design numbers in this paper are computed on the designs that have a published graph.

### 2.3 Graph neural networks

A graph neural network computes a vector at each node by passing messages along edges. Early spectral and spatial convolutional models showed that a small stack of neighborhood aggregations can classify nodes from a handful of labels. The baselines in this study are three standard instantiations of that idea.

A graph convolutional network aggregates a normalized sum of neighbor states. The normalization depends on degree, so a high-fanout driver does not automatically dominate its loads. A GraphSAGE layer samples or, in the full-batch form used here, aggregates the neighborhood with a learned linear map and combines it with the node's own state. A graph isomorphism network uses a sum aggregator with a multilayer perceptron, motivated by the ability of an injective aggregator to distinguish structures that a mean aggregator can confuse. All three are permutation-aware only through the graph they are given. They have no separate notion of a clock, a path, or a slack. If those signals are absent from the node features, the network must recover them from connectivity.

Netlists differ from the citation graphs on which these models were first demonstrated. The graph is directed, from driver to load, and the reverse edge is not added. Direction matches signal flow. A load's arrival time depends on its drivers, so the messages that matter for slack run forward along that direction. jpeg alone has 36,283 nodes and six published endpoints. The architectures here are node-level. They emit a prediction at every node and score only the labeled ones.

### 2.4 Timing prediction

Learned timing prediction sits beside the timer, not inside it. The timer remains the authority for signoff. A learned model is a proposal about which endpoints are likely to be late, trained from endpoints the timer has already classified in other designs. The hope is transfer: cones of logic that were late in one block share structural shape with cones that are late in another block, even when the instance names and the path names differ.

Several traps follow from the problem definition. The first is leakage. Any feature that is a deterministic function of the label, including the label itself, inflates holdout scores and teaches nothing about structure. Arrival time and required time are in that family because slack is their difference, up to the sign convention of the report. This study stores those fields for evaluation and keeps them out of every encoder.

The second trap is the split. Endpoints on the same path, or gates that share a cone inside one design, are dependent. A random split reports optimism. Holding out an entire design is the stricter and more relevant test, because a new block is what an implementation team actually adds.

The third trap is label poverty. With four to seven endpoints per design, accuracy, precision, recall, and F1 move when one endpoint flips. ROC-AUC and mean absolute error use every endpoint, including folds where the classifier predicts no violations and F1 is undefined. This paper reports F1 with that caveat attached, and it treats ROC-AUC and MAE as the scores that always have a defined mean across the six holdouts.

The fourth trap is copied labels. Expanding a path slack onto every gate of the route creates more training rows, and those rows are duplicates of one measurement. A model can look strong on the expanded set by smoothing a constant along the route. Endpoint metrics remain the published numbers. Expanded metrics are reported because they describe the training target, and they are interpreted as inherited slack.

## 3. Dataset generation pipeline

### 3.1 RTL

Each design enters the flow as register-transfer Verilog from the OpenROAD design catalog, together with a platform configuration and a constraints script. The RTL describes behavior and module hierarchy. It is not the graph the model sees. Synthesis replaces behavioral statements with standard cells from the SkyWater high-density library. Later stages of the flow may resize cells, insert clock buffers, and add diodes during antenna repair. The graph extractor therefore waits for the finish netlist, when the cell list matches the netlist the timer analyzed.

The six designs are mixed in size and function so that a holdout is a genuine shift. gcd is two orders of magnitude smaller than jpeg. The cumulative plots follow the publication order: gcd, aes, ibex, jpeg, riscv32i, chameleon. That order is the order the corpus was published. It does not mean each design contains the previous one.

### 3.2 ORFS and OpenROAD

OpenROAD-flow-scripts runs the platform flow inside the project container. The platform is sky130hd and the flow variant is base. The stages that matter for this dataset are those that produce a routed netlist and a finish timing report. Synthesis builds the gate-level netlist. Floorplanning and placement fix locations. Clock-tree synthesis inserts the clock network. Global and detailed routing assign wires and write parasitics. Finish emits the final Verilog, the final constraints, the SPEF, and the timing report.

The study does not use intermediate stage reports as extra labels. Floorplan slack, placement slack, and post-route slack are different numbers for the same RTL. Mixing them would train the model on several physical realizations at once. The label is the finish report only. Timing that is still negative after routing is left negative. The flow is not iterated to force a met design, because the violation class would then disappear from the corpus.

Import copies the finish files into a fixed directory layout for each design: the netlist, the constraints, the parasitics, and the reports. Logs and implementation databases stay in the flow workspace. A design is eligible for the corpus when those finish files exist and the later extraction stages succeed. uart and spi are recorded as names whose directories are missing. They are outside the published totals.

### 3.3 Timing reports

The parser reads path checks from the finish report. A path check names the startpoint, the endpoint, the path group, and the slack. Groups in this corpus include functional clocks, gated clocks, asynchronous checks, and virtual clocks used for external delays. Exact duplicate path blocks, which the report sometimes reprints, are collapsed. When two checks name the same endpoint, the stored slack is the worse of the two, which is the minimum slack. That choice matches a setup-oriented view: the endpoint is as late as its worst reported path.

The binary label is derived from the sign. Slack below zero is a violation. Slack at or above zero meets the constraint. Unlabeled nodes receive a missing slack and are masked out of every loss. The parser does not invent endpoints that the report omitted. Across the six finish reports there are no additional unique endpoints hiding behind duplicate printouts. The path counts that later appear as four, five, six, or seven labels per design are the unique reported endpoints after this collapse.

### 3.4 Graph extraction

The extractor builds a directed graph from the finish netlist. A node is a primary input, a primary output, a register, or a combinational gate. An edge runs from the driver of a net to each load of that net. Reverse edges are not added. Hierarchical modules are flattened. Unknown cells become leaves. Output pins of standard cells include the usual SkyWater pin names and the sum and carry pins of arithmetic cells. Power pins are skipped.

Two extractor behaviors matter for the larger designs. Multi-bit leaf pins are expanded to one pin use per bit, so a memory address bus becomes a set of single-bit connections. Macro pins whose direction is not obvious from the cell name are resolved from the surrounding net, including the convention that an Ibex core drives the shared AHB address and write-data pins and that a peripheral drives its response and interrupt pins. These rules let chameleon, whose top module is soc_core and whose memories are black boxes, enter the corpus as a connected signal-flow graph. Width mismatches on a defined child port remain errors. The one-bit standard-cell designs that were already published keep the connectivity they had.

Logic depth is computed on this graph as the longest hop count from an input or a register, stopping at the next register. Fanin equals in-degree and fanout equals out-degree, because each edge is one driver-to-load connection. The published node does not store a standard-cell name, so the model cannot one-hot a library cell. Gate type is the four-way class.

### 3.5 PyTorch Geometric dataset

Each design is stored as a PyTorch Geometric `Data` object. The base node matrix is a four-column one-hot encoding in the order input, output, register, combinational. The edge index is a long tensor of shape two by the number of edges, oriented driver to load. The slack target is a float tensor with missing values on unlabeled nodes. Masks mark labeled nodes, positive slack, and negative slack. The dataset also stores node features, edge attributes, and timing features for SlackGNN. Baselines read the one-hot matrix. SlackGNN reads the richer tensors. Both read the same masks and the same slack.

The corpus builder copies each successful design into a corpus directory and writes a metadata file. Headline totals count successful designs only. Failed names stay in the manifest as failures and do not add nodes, edges, or labels. Training loads the published graph, attaches the expanded-label tensor from the label-expansion stage, and enriches features. The published `graph.pt` slack is not overwritten by expansion. Endpoint evaluation can therefore return to the original path-table values after a model has trained on the auxiliary routes.

## 4. Corpus statistics

Figure `reports/figures/publication/corpus_composition.png` shows nodes, edges, and expanded labels side by side. Figure `reports/figures/publication/corpus_growth.png` accumulates those quantities in publication order. Figure `reports/figures/publication/design_complexity.png` plots nodes against edges.

| Design | Nodes | Edges | Endpoints | Met | Violations | Expanded labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 527 | 928 | 5 | 3 | 2 | 27 |
| aes | 15,868 | 38,215 | 4 | 3 | 1 | 20 |
| ibex | 18,000 | 44,436 | 7 | 5 | 2 | 42 |
| jpeg | 36,283 | 82,643 | 6 | 5 | 1 | 23 |
| riscv32i | 5,997 | 15,884 | 4 | 2 | 2 | 33 |
| chameleon | 3,181 | 6,922 | 6 | 4 | 2 | 28 |
| Total | 79,856 | 189,028 | 32 | 22 | 10 | 173 |

The size distribution is skewed. jpeg holds 36,283 of the 79,856 nodes and 82,643 of the 189,028 edges. gcd holds 527 nodes. Edges track nodes closely, which is what a gate-level signal-flow graph should do: each gate has a small, bounded number of loads, and the scatter of nodes against edges is nearly linear across two orders of magnitude. On disk, the six design directories occupy about 137 MB, dominated by netlists and parasitics. The PyTorch Geometric graphs themselves are much smaller, about 4.9 MB in total, because they store typed nodes and edges rather than SPEF bodies.

Labels do not follow the same curve. jpeg is the largest graph and has 23 expanded labels. ibex, at half the node count, has 42, the most in the corpus. Expansion adds 141 labels to the original 32, a gain of about 4.4 route nodes per published endpoint on average. The gain is uneven because route length varies. gcd grows from 5 endpoints to 27 route nodes. riscv32i grows from 4 to 33. aes grows from 4 to 20. Shared gates keep the minimum slack of the routes that touch them, and the published endpoint slack is restored afterward so the endpoint row still matches the path table.

The class balance of the 32 endpoints is 22 met and 10 violated. The expanded set is 66 met and 107 violated, because several of the longer violated routes contribute more gates than the met routes. Training therefore sees more violation rows than the endpoint test set contains. The classification loss uses a positive weight equal to the number of met training labels divided by the number of violation labels, computed on the training split of each fold, so the weight tracks the split actually used.

The supervision rate on the raw graph is 32 labels out of 79,856 nodes, about 0.04%. After expansion it is 173 out of 79,856, about 0.22%. Message passing is dense. Supervision is sparse. That ratio is the central experimental fact of the corpus, and it is why endpoint metrics on four to seven test labels must be read beside ROC-AUC and MAE.

## 5. SlackGNN architecture

### 5.1 Feature engineering

SlackGNN consumes three tensors that the baselines do not.

The node tensor has ten channels. The first four are the one-hot gate class: input, output, register, or combinational. A register flag repeats the sequential bit. Fanin, fanout, in-degree, and out-degree are log-normalized. On this graph fanin equals in-degree and fanout equals out-degree, because each edge is one driver-to-load link. Logic depth is the hop count from the nearest input or register, scaled inside the design. Cell names are absent, so two gates of different drive strengths share a class.

The edge tensor has ten channels. Four one-hot bits name the source class and four name the target class, so the attention scorer can tell a register-to-gate edge from a gate-to-gate edge. One channel is a delay proxy, the log of one plus the driver fanout, normalized per design. One channel is the normalized logic depth of the load, a topological distance along the same definition used for nodes. SPEF resistance and capacitance are not in this tensor. Two nets with equal fanout and equal load depth receive the same edge features even when one is a short local route and the other spans the chip. The ablation that removes edge features is, for that reason, a test of the fanout proxy and the endpoint types, not a test of extracted parasitics.

The timing tensor has three channels. The first is normalized logic depth, used as a structural stand-in for how critical a gate is likely to be. The second is reported-path membership, equal to one for every node that the expansion marked as lying on a reported route, including every published endpoint. The third is source distance, implemented as the same normalized depth measured from an input or register. Membership is a path feature. Depth and source distance are criticality features. Arrival time, required time, and slack are stored on the dataset for the loss and for later inspection. They are not channels of this tensor.

The leakage rule is mechanical. If a channel is a function of the endpoint slack, it is excluded. Membership says that a node appears in the report. It does not say the sign or the magnitude of the slack. Depth says where the node sits in the cone. A late path is often a deep path, and a deep path is often a short path, so depth is a prior, not a label.

### 5.2 Message passing

The model encodes the three tensors separately and then runs two timing-aware GraphSAGE layers. The node encoder maps the ten node channels to the hidden width. The edge encoder maps the ten edge channels to an edge state used by the scorer. The timing encoder maps the three timing channels to a timing state that is added into the node state and also shown to the scorer. Hidden width is 32 in every experiment in this paper. Dropout on SlackGNN is 0.25. The baselines use dropout 0.5, which is their published default.

Each layer updates loads from drivers. For a load, the layer scores every incoming edge, normalizes those scores, and writes a weighted sum of driver messages back onto the load, combined with a linear transform of the load's own state. Two layers give each endpoint a two-hop driver cone. Logic depth on these designs can reach several dozen hops, so the stack sees a local neighborhood and the depth feature summarizes the rest of the cone. Unlabeled gates are updated by the same layers. The loss never asks them to match a slack. They shape the endpoint embedding.

### 5.3 Attention

The score at each incoming edge is a learned function of the driver embedding, the load embedding, the edge encoding, the timing encodings of both ends, and a slack-magnitude gate. The magnitude gate is computed from the node embedding. It is not the ground-truth slack and it is not the absolute value of the label. Softmax is taken over the drivers of one load, so the weights on the incoming edges of a gate sum to one. A gate with a single driver assigns that driver weight one. A gate with many drivers can concentrate on one of them.

Uniform attention, used in the ablation, replaces this softmax with equal weights on the incoming drivers. The rest of the layer stays in place. The comparison therefore asks whether the learned weights carry information beyond the mean of the driver messages. On this corpus they carry ranking information. Removing them lowers endpoint ROC-AUC from 0.9583 to 0.9139. Endpoint F1 and MAE do not get worse under that removal, which means the equal-weight aggregator is already enough for the coarse violation call on these few endpoints, and the learned weights are doing finer ordering.

### 5.4 Multitask heads

The updated embedding splits into two heads. The classification head emits one logit. The probability of a violation is the sigmoid of that logit. The training criterion is binary cross-entropy with logits, with the positive weight described above. The regression head emits a slack in nanoseconds. Its criterion is smooth L1, which behaves like squared error near zero and like absolute error for large residuals, so a single badly missed endpoint does not dominate the gradient as strongly as a pure square would. The joint objective is the classification loss plus half the regression loss. The coefficient on classification is 1 and the coefficient on regression is 0.5.

The heads share the embedding, so a feature that helps one task can help the other, and a feature that helps one can also interfere. The ablations are the evidence on that trade. Removing timing depth raises F1 and hurts MAE and ROC-AUC at the same time. The classifier can look better at the decision threshold while the ranking and the nanosecond error get worse. That is why the paper does not crown an ablation by F1 alone.

At inference the model runs on the full graph and the reported metrics read the published endpoints. A deployment that wanted a path list would still ask the timer for the paths. SlackGNN would be asked which of those endpoints look late, or how late, given the cones around them.

## 6. Experimental setup

The protocol is leave-one-design-out on the six graphs. For each held-out design, the training pool is the expanded labels of the other five. Validation holds out one expanded label of each class from that pool when both classes are present, matching the split used throughout the later milestones. The test set for the headline numbers is the published endpoints of the held-out design. A second score is computed on the expanded labels of the held-out design so the inherited-slack target can be inspected separately. The held-out design contributes no training rows.

Optimization is Adam with learning rate 0.01 and weight decay 0.0001. The epoch cap is 20. Patience is 6 epochs of validation loss. The seed is 0. Hidden width is 32. These limits keep the CPU study finite on jpeg. They are also the reason a comparison across milestones is valid: generalization, ablation, robustness, and the accuracy columns of the scaling table all refer to this protocol. Scaling measures wall-clock on the same models and then extrapolates the protocol. It does not retune the hyperparameters.

Metrics for classification are accuracy, precision, recall, F1, and ROC-AUC, with the positive class equal to a timing violation. Metrics for regression are mean absolute error, mean squared error, and root mean square error, in nanoseconds. F1 is undefined when a model predicts no violations, because precision and recall are then undefined or the recall is zero in a way that leaves no positive predictions to score. Means skip those undefined folds. ROC-AUC and MAE are computed on every holdout that contains both classes or, for MAE, on every holdout. Where a mean F1 covers fewer than six designs, the paper says so.

The implementation is PyTorch with PyTorch Geometric. Message passing runs on CPU. There is no claim about GPU throughput in these measurements. Graphs are stepped one at a time. An epoch visits each training design once. Peak step memory therefore tracks the largest graph in the fold, which is jpeg whenever jpeg is in the training set.

## 7. Baseline models

The baselines are GCN, GraphSAGE, and GIN. Each is a two-stage stack from the one-hot node type to a single scalar per node, with hidden width 32 and dropout 0.5. They do not read edge attributes or timing channels. Classification and regression are separate models. Classification uses binary cross-entropy with a positive weight. Regression uses mean squared error on slack. Evaluation merges the two scores into one metric row. A published baseline run therefore does about twice the step work of a single head, which the scaling section counts as protocol time. GraphSAGE is the closest relative of SlackGNN, because the timing layer is a SAGE-style update with a learned scorer. GCN tests a degree-normalized aggregator. GIN tests a sum aggregator. All three see the same splits and the same expanded training labels.

## 8. Generalization results

Figure `reports/figures/publication/f1_comparison.png`, `roc_auc_comparison.png`, `mae_comparison.png`, and `rmse_comparison.png` show the endpoint means. Figure `lodo_f1.png`, `holdout_roc_auc.png`, and `holdout_mae.png` break the same scores out by held-out design. Figure `design_difficulty.png` ranks the designs.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | 0.6381 | 0.5167 | 0.5833 | 0.5800 | 0.6243 | 0.9621 | 1.1922 |
| GraphSAGE | 0.5980 | 0.1250 | 0.0833 | 0.3333 | 0.7632 | 0.8704 | 1.1053 |
| GIN | 0.6202 | 0.4444 | 0.6667 | 0.5222 | 0.6951 | 0.9470 | 1.1712 |
| SlackGNN | 0.8179 | 0.8133 | 0.7500 | 0.8076 | 0.9583 | 0.6919 | 0.8667 |

SlackGNN has the best endpoint F1, the best ROC-AUC, and the best MAE and RMSE. The F1 gap from the next defined baseline mean, GCN at 0.5800, is 0.2276. The ROC-AUC gap from the next baseline, GraphSAGE at 0.7632, is 0.1951. The MAE gap from GraphSAGE, the best baseline on error, is 0.1785 ns. RMSE moves with MAE, from 1.1053 ns for GraphSAGE to 0.8667 ns for SlackGNN.

GraphSAGE's endpoint F1 of 0.3333 is the chameleon fold only. On gcd, aes, ibex, jpeg, and riscv32i, GraphSAGE predicts no violations, so F1 is undefined and the mean omits those folds. Its ROC-AUC of 0.7632 and MAE of 0.8704 do use all six holdouts. The F1 comparison should be read with that omission visible. Even if one looks only at ROC-AUC and MAE, which are defined everywhere, SlackGNN leads.

The per-design SlackGNN endpoint scores show where the mean comes from.

| Held-out | Support | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gcd | 5 | 0.8000 | 1.0000 | 0.7045 | 0.8072 |
| aes | 4 | n/a | 1.0000 | 0.2578 | 0.2832 |
| ibex | 7 | 0.6667 | 1.0000 | 1.3993 | 1.8235 |
| jpeg | 6 | 1.0000 | 1.0000 | 0.9296 | 1.2074 |
| riscv32i | 4 | 1.0000 | 1.0000 | 0.2785 | 0.3401 |
| chameleon | 6 | 0.5714 | 0.7500 | 0.5815 | 0.7392 |

Five of the six holdouts have ROC-AUC 1.0. The exception is chameleon at 0.7500. jpeg and riscv32i are perfect on the violation call among their published endpoints. aes has ROC-AUC 1.0 and the smallest MAE, 0.2578 ns, and an undefined F1 because the classifier predicted no violations on a design whose single violation was not called at the decision threshold. Ranking can be perfect while the thresholded F1 is undefined. That is the aes fold, and it is why the paper quotes ROC-AUC next to F1.

ibex is the expensive holdout for slack error. Support is 7, F1 is 0.6667, ROC-AUC is 1.0, and MAE is 1.3993 ns. The model orders the ibex endpoints and still misses their magnitudes by more than a nanosecond on average. ibex is also the hardest design when difficulty is the mean of defined endpoint F1 across all four models, at 0.4667. aes is the easiest by that definition, at 0.8333, which averages the defined scores only. GCN scores F1 1.0 on aes and GIN scores 0.6667. GraphSAGE and SlackGNN contribute no F1 on aes. The difficulty ranking is therefore a statement about the scores that exist, and the figure labels aes as easiest and ibex as hardest on that basis.

Expanded-label means tell a related story. SlackGNN reaches expanded F1 0.7955, ROC-AUC 0.9027, and MAE 0.6848 ns. GraphSAGE reaches expanded F1 0.8094, slightly higher, with ROC-AUC 0.8082 and MAE 0.7431 ns. GCN and GIN trail on expanded ROC-AUC and MAE. The expanded F1 edge for GraphSAGE is consistent with copied labels being a smoother target that a plain neighborhood aggregator can fit. The endpoint metrics, which are the published slack, still favor SlackGNN on F1, ROC-AUC, and MAE. This paper treats the endpoint table as the result and the expanded table as a description of the training target.

## 9. Ablation results

The ablation retrains the full SlackGNN protocol and four controlled removals. The full model reproduces the generalization numbers exactly: F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, RMSE 0.8667 ns. The comparison is therefore internal to one training procedure.

| Variant | F1 | ROC-AUC | MAE (ns) | RMSE (ns) | F1 change | MAE change (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full SlackGNN | 0.8076 | 0.9583 | 0.6919 | 0.8667 |  |  |
| No attention | 0.8343 | 0.9139 | 0.6464 | 0.8384 | +0.0267 | −0.0455 |
| No path membership | 0.7886 | 0.9514 | 0.7080 | 0.8773 | −0.0190 | +0.0162 |
| No timing criticality | 0.8933 | 0.8722 | 0.7360 | 0.8925 | +0.0857 | +0.0441 |
| No edge features | 0.8952 | 0.9583 | 0.6547 | 0.8379 | +0.0876 | −0.0372 |

Path membership is the only removal that lowers both endpoint F1 and endpoint MAE. The F1 drop is 0.019 and the MAE rise is 0.016 ns. The absolute changes are small next to the gap versus the baselines, and they are the only changes that point the same direction on both scores. Membership is also the channel that says "this gate was on a reported route." Removing it asks the model to find late endpoints from depth, type, and connectivity alone. It still works, and it works a little less well.

Timing criticality is logic depth and source distance, with membership left in place. Removing it raises the defined-fold F1 to 0.8933 and hurts the scores that use every holdout. ROC-AUC falls from 0.9583 to 0.8722. MAE rises from 0.6919 ns to 0.7360 ns, the largest MAE increase in the table. Depth is doing ranking and calibration. Without it, a thresholded F1 can improve because the decision boundary on a few endpoints happens to land well, while the ordering of scores and the nanosecond error degrade. The paper reads that pattern as evidence that criticality is load-bearing for the slack task, and that F1 on undefined-prone folds is a brittle summary.

Attention removal lowers ROC-AUC to 0.9139 and improves MAE to 0.6464 ns. The learned weights help ranking. The equal-weight mean of drivers is a strong regressor on this corpus. Edge-feature removal leaves ROC-AUC at 0.9583 and improves MAE to 0.6547 ns. The fanout proxy and the endpoint-type bits are not the source of the endpoint ROC-AUC. A future edge channel built from SPEF resistance and capacitance is a different experiment. This ablation does not speak for that channel.

F1 changes in the ablation table inherit the undefined-fold issue. The full model omits aes. The membership ablation omits jpeg. The criticality ablation omits chameleon. The attention ablation omits jpeg. The edge ablation has a defined F1 on all six holdouts. Part of an F1 increase is a change in which designs enter the mean. ROC-AUC and MAE do not have that ambiguity, and they are the basis for the criticality and attention conclusions above.

On the expanded labels, removing criticality drops F1 from 0.7955 to 0.7317 while ROC-AUC stays near 0.9024. The expanded target is itself organized by depth, because a route is a chain of increasing hops. Taking depth away hurts the fit to that chain. Removing membership raises expanded F1 to 0.8120 and raises expanded MAE to 0.6962 ns. The two tasks again disagree in sign, and the endpoint pair of changes remains the headline.

## 10. Robustness results

Robustness repeats the holdout under four degradations. Sparse labels keep a class-balanced fraction of the training and validation pool and then hold out one label per class again. The fractions are 100%, 75%, 50%, 25%, and 10%. Noise flips that fraction of training classes and adds Gaussian noise to training slack, with standard deviation equal to the same fraction of the training slack spread. The rates are 5%, 10%, and 20%. Evaluation uses the clean published endpoints. Reduced corpora are prefixes of the publication order: all six designs, then the first five, four, and three. Missing paths drop that fraction of mapped training routes, at least one route remains, and nodes are relabeled from the routes that stay. Endpoint slack on the held-out design stays the published value.

A SlackGNN condition is marked as a collapse when endpoint F1 falls more than 0.10 below 0.8076, or ROC-AUC falls below 0.80, or MAE rises more than 0.15 ns above 0.6919. The stable-corpus rule is tighter: F1 within 0.05 and MAE within 0.10 ns of the six-design result.

| Condition | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: |
| 100% labels | 0.8076 | 0.9583 | 0.6919 | 0.8667 |
| 75% labels | 0.8267 | 0.8847 | 0.8221 | 0.9959 |
| 50% labels | 0.8778 | 0.9181 | 0.7771 | 0.9514 |
| 25% labels | 0.8250 | 0.8097 | 0.7930 | 0.9557 |
| 10% labels | 0.6143 | 0.7264 | 0.9282 | 1.1551 |
| 5% noise | 0.8833 | 0.9097 | 0.6778 | 0.8616 |
| 10% noise | 0.7533 | 0.9097 | 0.7616 | 0.9558 |
| 20% noise | 0.8200 | 0.9514 | 0.7497 | 0.9517 |
| 6 designs | 0.8076 | 0.9583 | 0.6919 | 0.8667 |
| 5 designs | 0.8667 | 1.0000 | 0.8377 | 1.0083 |
| 4 designs | 0.6750 | 0.9500 | 0.9890 | 1.1962 |
| 3 designs | 0.7333 | 0.9667 | 0.9011 | 1.0263 |
| 10% paths removed | 0.8133 | 0.9514 | 0.7272 | 0.8737 |
| 25% paths removed | 0.8400 | 0.9722 | 0.7094 | 0.8806 |
| 50% paths removed | 0.9000 | 0.9236 | 0.7210 | 0.9085 |

Sparse labels hold through 25%. At 25%, F1 is 0.8250, ROC-AUC is 0.8097, and MAE is 0.7930 ns. The ROC-AUC is just above the 0.80 line and the MAE increase is 0.101 ns, inside the 0.15 ns collapse band. At 10%, F1 is 0.6143, ROC-AUC is 0.7264, and MAE is 0.9282 ns. All three collapse rules fire. The failure threshold for label quantity is between 25% and 10% of the expanded pool. In absolute counts, 25% of roughly 140 training labels is still a few dozen supervised nodes. Ten percent is a handful. The model is using the route labels, and it runs out of them quickly once the pool shrinks past a few dozen.

Noise through 20% does not collapse the model. The worst noise row is 10%, with F1 0.7533, ROC-AUC 0.9097, and MAE 0.7616 ns. The 20% row recovers to F1 0.8200 and ROC-AUC 0.9514. The curve is not monotone. What the sweep supports is a negative statement about fragility: label noise at these rates does not erase the holdout. What it does not support is a claim that more noise is harmless at every rate between the samples.

Path deletion is the surprising sweep. Removing 10%, 25%, or 50% of training routes leaves endpoint F1 at or above 0.8133, ROC-AUC at or above 0.9236, and MAE at or below 0.7272 ns. None of these rows collapse. Reported paths are partly redundant for endpoint transfer. Many route nodes repeat one slack, so deleting some paths leaves other paths that still teach the same structural lesson. The practical consequence is sharp. A new design helps the corpus when its finish report adds endpoints. A larger netlist whose report reprints the same few paths adds edges and adds little supervision. The scaling projection in the next section should be read with that fact in mind.

Corpus size is the sweep that sets the minimum stable collection. Six designs match the reference by construction. Five designs, dropping chameleon, keep a strong ranking: F1 0.8667 and ROC-AUC 1.0. MAE rises to 0.8377 ns, an increase of 0.146 ns. That increase is just inside the 0.15 ns collapse rule and outside the 0.10 ns stability band. Four designs collapse: F1 0.6750 and MAE 0.9890 ns. Three designs recover F1 to 0.7333 and keep ROC-AUC at 0.9667, while MAE stays at 0.9011 ns, still past the error rule. The smallest corpus that satisfies the stability band is six designs. The ranking can look healthy on five. The slack error does not.

Across the degraded conditions, mean endpoint F1 is 0.8022 for SlackGNN, 0.5046 for GIN, 0.4942 for GCN, and 0.4245 for GraphSAGE. SlackGNN is the most robust of the four under that average. GraphSAGE is the least robust, and its worst entry counts an undefined F1 as zero. On the conditions where every baseline F1 is defined, SlackGNN still has the highest F1 and the lowest MAE. On the two conditions where GraphSAGE's F1 is undefined, SlackGNN still has the best MAE and the best defined F1.

## 11. Scaling results

Scaling measures the six graphs directly and then projects larger corpora. Each timing entry is the median of three steps after a warmup. Hidden width is 32. Accuracy columns are the published endpoint means, not a new training run. Protocol training time counts a 20-epoch, six-fold holdout. Baseline protocol time counts two heads. Inference time is one forward pass on each design, run sequentially. PyTorch step memory is parameters plus forward activations and gradients. The working set is the process, which holds every loaded design.

Figure `reports/figures/publication/design_complexity.png` is the right companion to the runtime plots in `reports/figures/scaling/`, because edges and nodes move together and the runtime fit prefers edges.

| Model | Training time (s) | Inference time (s) | PyTorch step (MB) | F1 | ROC-AUC | MAE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | 29.39 | 0.0428 | 14.37 | 0.5800 | 0.6243 | 0.9621 |
| GraphSAGE | 26.21 | 0.0253 | 19.60 | 0.3333 | 0.7632 | 0.8704 |
| GIN | 43.84 | 0.0306 | 19.33 | 0.5222 | 0.6951 | 0.9470 |
| SlackGNN | 67.61 | 0.1647 | 50.97 | 0.8076 | 0.9583 | 0.6919 |

SlackGNN's projected holdout is 67.61 seconds, about 1.1 minutes, on the measured CPU. Its batch inference is 0.165 seconds for all six designs, about 485,000 nodes per second, and about 1,050 labeled predictions per second given 173 expanded labels. The step memory on the largest graph is 50.97 MB. The process working set is about 710 MB once the features and the graphs are resident. The baselines sit near 468 MB because they store the one-hot matrix and a smaller activation. Parameter storage is negligible next to activations: SlackGNN's parameters are about 22 KB.

Among O(N), O(E), O(N log N), and O(E log N), the lowest relative error is O(E) for GCN, GraphSAGE, and SlackGNN. GIN is essentially tied between O(N) and O(E), which follows when edges are nearly proportional to nodes. The mean log-log slope of training time against edges is 0.45. A pure edge-linear model would have slope 1. The shallower slope means a fixed per-step cost is still a large fraction of the gcd step and a smaller fraction of the jpeg step. Projections scale with design count using the current average design. They describe a corpus of average-sized additions. A future jpeg-sized design costs more than that average. A future gcd-sized design costs less.

jpeg is the runtime bottleneck. It accounts for 38.3% of a SlackGNN epoch. The backward pass is the heavier of the forward and backward pair. jpeg is also the memory bottleneck, with a 50.97 MB PyTorch step. GraphSAGE is the fastest model and the cheapest per edge. GCN has the best published F1 per projected training second, because its protocol time is 29.39 seconds and its F1 is 0.5800. SlackGNN spends 67.61 seconds to reach F1 0.8076. The accuracy per second favors GCN. The accuracy favors SlackGNN. Which one a user wants depends on whether the extra 0.23 of F1 and the 0.18 ns of MAE are worth roughly twice the baseline protocol time. For the research question posed in the introduction, they are the quantity the study set out to buy.

Corpus growth, assuming each added design matches the current mean size and that every fold still runs 20 epochs, is quadratic in the number of designs. There are as many folds as designs, and each fold trains on all but one of them.

| Designs | Nodes | Edges | Labels | Disk (MB) | SlackGNN training (min) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 79,856 | 189,028 | 173 | 136.8 | 1.1 |
| 10 | 133,093 | 315,047 | 288 | 228.1 | 3.4 |
| 20 | 266,187 | 630,093 | 577 | 456.1 | 14.3 |
| 50 | 665,467 | 1,575,233 | 1,442 | 1,140.3 | 92.0 |

Fifty average designs are about a 92 minute holdout under this protocol. The label column grows only if new designs contribute endpoints at the current rate, about 29 expanded labels per design. Disk grows because finish parasitics dominate the directory size. The graphs the trainer holds are a small fraction of that disk.

## 12. Discussion

The endpoint table is the result of the paper. A multitask model with depth, path membership, and attention, trained on inherited route slack, estimates published endpoint slack on a held-out OpenROAD design more accurately than GCN, GraphSAGE, or GIN trained on node type alone. The best F1 is 0.8076. The best ROC-AUC is 0.9583. The best MAE is 0.6919 ns. The best RMSE is 0.8667 ns. All four belong to SlackGNN. The ordering is stable enough to survive label thinning down to 25%, label noise up to 20%, and deletion of half the training paths.

The mechanism is narrower than "the whole architecture." Path membership is the only ablation that hurts both F1 and MAE. Timing depth is the ablation that hurts ROC-AUC and MAE. Attention hurts ROC-AUC and does not hurt MAE. The fanout edge features are not required for the endpoint ROC-AUC. A reader who wants one sentence can say that SlackGNN's gain is a timing context on top of a SAGE-style backbone, and that the context which matters is "is this gate on a reported path" together with "how deep is it."

Two results limit how far that sentence travels. The first is aes. Perfect ROC-AUC and a very small MAE coexist with an undefined F1. Thresholds and ranking are different claims. A screening tool that sorts endpoints will look stronger on this corpus than a tool that must emit a hard violation bit. The second is ibex. ROC-AUC is 1.0 and MAE is 1.4 ns. Ordering the endpoints of the hardest design is not the same as predicting their slack to a tenth of a nanosecond. Signoff budgets live in that tenth. This model lives in the half-nanosecond to one-nanosecond band on the designs where it is stressed.

Expanded labels are the reason the training loss has more than 32 rows, and they are also the reason expanded F1 can flatter a baseline. GraphSAGE's expanded F1 slightly exceeds SlackGNN's. The endpoint metrics do not. Any follow-on paper that quotes only expanded scores will tell a different story from this one. The figures in `reports/figures/publication/` use the endpoint scores for that reason.

The robustness and scaling sections agree on a deployment shape. About six diverse finish reports, with at least a quarter of the expanded labels retained, keep slack error near the published result. The run is dominated by the largest netlist's backward pass. A new design helps when its report adds paths. Ten percent of the current label pool is past the collapse line.

The numbers in this paper are the six-design protocol. An earlier measurement of gcd, aes, ibex, and jpeg used 22 endpoints and a different training budget. Quote this paper's tables for the six-design corpus.

## 13. Threats to validity

The endpoint support is the largest threat. Thirty-two test labels, split into folds of four to seven, make F1 a high-variance statistic. One flipped endpoint on chameleon, whose support is six, moves F1 by a large step. The study mitigates this by reporting ROC-AUC and MAE on every fold, by publishing the per-design table, and by refusing to treat an undefined F1 as a zero in the generalization means. It does not create more endpoints. Only a richer timing report can do that. Until then, the confidence interval on F1 is wide even though the point estimate is stable across several robustness sweeps.

The second threat is dependence inside a path. Expanded labels on one route share a slack by construction. Training rows are not independent. The holdout reduces one kind of leak, the leak across designs, and leaves the dependence inside the training designs in place. A model can memorize "this cone shape was late" from a single violated path that contributed twenty gates. That is a legal use of the report, and it is a narrow use. If the next design's late cone looks different, the memorized shape will not transfer. ibex and chameleon, the two hardest SlackGNN F1 folds, are the warning.

The third threat is the feature gap on interconnect. SPEF is imported and unused by the encoder. Wire delay is a large fraction of path delay after routing. A model that ignores extracted RC can still rank endpoints if late paths are systematically the deep ones, which is often true and is not always true. The edge ablation shows that the current edge channels are not carrying the ROC-AUC. It does not show that a parasitic channel would be useless. The threat is an omitted variable, not a failed test of that variable.

The fourth threat is the single seed and the twenty-epoch cap. The six-design robustness section varies the data. A release that quotes 0.8076 should keep that protocol beside the number.

The fifth threat is platform scope. Every design is sky130hd. A second library, a second metal stack, or a commercial timer with a different path printer could move both the features and the label distribution. The transfer result is cross-design within one platform. It is cross-technology only as a hypothesis.

The sixth threat is selection. The six designs are the ones whose flows finished and whose netlists extracted. microwatt is missing. uart and spi were not imported. The corpus is a sample of successful open-flow finishes, and the sample is small. The collapse thresholds of 0.10 F1, 0.80 ROC-AUC, and 0.15 ns MAE are analysis choices. The 10% label row fails a wide range of nearby bands. The 50% path-deletion row passes them. The raw rows are published so a reader can draw the line elsewhere.

## 14. Future work

The highest-value next measurement is a finish report with more unique endpoints. The robustness sweep says the model collapses when the expanded pool falls to 10%, and that it hardly notices when half the existing paths are deleted. New paths change the experiment. More copies of the current paths do not. A design worth adding is one that brings a new endpoint list, whether or not its graph is larger than jpeg.

The highest-value feature is extracted resistance and capacitance on the driver-to-load edge, joined without putting slack into the encoder. The SPEF files are already in the design directories. The ablation of the fanout proxy is not a substitute for that join. If parasitic channels move MAE on ibex, where the error is 1.4 ns, they will have earned their place. If they do not, the depth feature is doing the work the wire delay would have done, and the paper's mechanism story gets stronger.

A seed sweep on the six-design protocol would put an error bar on 0.8076. The data sweeps in this paper are not that bar. Ten seeds at the current budget are a few hours of CPU, which the scaling section says is affordable.

Threshold calibration is a separate project from ranking. The aes fold is the example. A precision-recall curve per holdout, or a slack threshold chosen on the validation labels, would turn the undefined F1 cases into a reported operating point. The ROC-AUC already says the ranking is there.

Finally, the timer should stay in the loop. A credible integration is to run SlackGNN on a new block, list the endpoints it marks late, and ask OpenROAD to report those paths. Agreement with the timer is the external test this corpus cannot run on its own holdout, because the holdout labels are already the timer's answers. A prospective block, labeled only after the model has ranked it, is the clean version of that test.

## 15. Conclusion

SlackGNN estimates timing violations and slack on held-out OpenROAD netlists. On six SkyWater 130 nm designs, with 79,856 nodes, 189,028 edges, 32 published endpoints, and 173 expanded training labels, the leave-one-design-out endpoint scores are F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns, and RMSE 0.8667 ns. Those scores lead GCN, GraphSAGE, and GIN on F1, ROC-AUC, and MAE. The gain is carried by reported-path membership and by logic depth. Attention supports ranking. The current edge features do not.

The operating range is concrete. The model is stable down to 25% of the expanded labels, handles 20% label noise, and handles deletion of half the training paths. It collapses at 10% labels. It wants about six designs before slack error stays inside 0.10 ns of the full result. Runtime scales approximately with edge count, and jpeg dominates the epoch. The projection to larger corpora stays modest on CPU as long as new designs resemble the current average and the epoch cap stays at 20.

The claim stops at the evidence. This is a cross-design screening result on a small endpoint set from one open platform. It is a measured reason to keep a graph model in the timing-analysis loop, and it is a measured reason to ask the timer for more paths before asking the model for signoff.

## References

M. Fey and J. E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," Advances in Neural Information Processing Systems, 2017.

T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," International Conference on Learning Representations, 2017.

The OpenROAD Project, OpenROAD and OpenROAD-flow-scripts, https://github.com/The-OpenROAD-Project.

SkyWater Technology and Google, SkyWater SKY130 Process Design Kit.

K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How Powerful are Graph Neural Networks?," International Conference on Learning Representations, 2019.

## Figure index

The publication figures live in `reports/figures/publication/`.

| Figure | File | What it shows |
| --- | --- | --- |
| Corpus composition | `corpus_composition.png` | Nodes, edges, and expanded labels for each design |
| Corpus growth | `corpus_growth.png` | Cumulative nodes, edges, and labels from gcd through chameleon |
| Design complexity | `design_complexity.png` | Node count against edge count |
| F1 comparison | `f1_comparison.png` | Endpoint F1 for GCN, GraphSAGE, GIN, and SlackGNN |
| ROC-AUC comparison | `roc_auc_comparison.png` | Endpoint ROC-AUC |
| MAE comparison | `mae_comparison.png` | Endpoint MAE in nanoseconds |
| RMSE comparison | `rmse_comparison.png` | Endpoint RMSE in nanoseconds |
| Leave-one-design-out F1 | `lodo_f1.png` | Endpoint F1 by held-out design |
| Holdout ROC-AUC | `holdout_roc_auc.png` | Endpoint ROC-AUC by held-out design |
| Holdout MAE | `holdout_mae.png` | Endpoint MAE by held-out design |
| Design difficulty | `design_difficulty.png` | Mean defined endpoint F1, with aes marked easiest and ibex marked hardest |

Supporting plots from the robustness and scaling studies are in `reports/figures/robustness/` and `reports/figures/scaling/`. The tables in this paper are the numerical source for those charts.
