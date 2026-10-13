<p align="center">
  <img src="docs/assets/banner.png" alt="SlackGNN: timing-aware graph neural networks for OpenROAD timing prediction. Endpoint F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns on six sky130hd designs." width="100%">
</p>

<p align="center">
  <b>Timing-aware graph neural networks for OpenROAD timing prediction.</b><br>
  Predict endpoint timing violations and slack on a placed-and-routed netlist the model has never seen.
</p>

<p align="center">
  <a href="docs/RELEASE_v3.0.md"><img alt="Release v3.0 research" src="https://img.shields.io/badge/release-v3.0%20research-c47b2b?style=flat-square"></a>
  <a href=".python-version"><img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-3776ab?style=flat-square&logo=python&logoColor=white"></a>
  <a href="requirements.txt"><img alt="PyTorch 2.9 to 2.12" src="https://img.shields.io/badge/PyTorch-2.9%E2%80%932.12-ee4c2c?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="requirements.txt"><img alt="PyTorch Geometric 2.8" src="https://img.shields.io/badge/PyG-2.8-3c2179?style=flat-square"></a>
  <a href="https://github.com/The-OpenROAD-Project"><img alt="OpenROAD-flow-scripts" src="https://img.shields.io/badge/flow-OpenROAD--flow--scripts-1f4e79?style=flat-square"></a>
  <a href="docs/dataset_structure.md"><img alt="PDK sky130hd" src="https://img.shields.io/badge/PDK-sky130hd-2f6f4e?style=flat-square"></a>
  <a href="CONTRIBUTING.md#testing"><img alt="Tests: pytest" src="https://img.shields.io/badge/tests-pytest-0a9edc?style=flat-square&logo=pytest&logoColor=white"></a>
  <a href="#license"><img alt="License not yet chosen" src="https://img.shields.io/badge/license-not%20yet%20chosen-lightgrey?style=flat-square"></a>
</p>

<p align="center">
  <a href="#results"><img alt="Endpoint F1 0.8076" src="https://img.shields.io/badge/endpoint%20F1-0.8076-c47b2b?style=for-the-badge"></a>
  <a href="#results"><img alt="ROC-AUC 0.9583" src="https://img.shields.io/badge/ROC--AUC-0.9583-c47b2b?style=for-the-badge"></a>
  <a href="#results"><img alt="MAE 0.6919 ns" src="https://img.shields.io/badge/MAE-0.6919%20ns-c47b2b?style=for-the-badge"></a>
  <a href="#results"><img alt="RMSE 0.8667 ns" src="https://img.shields.io/badge/RMSE-0.8667%20ns-c47b2b?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="#the-problem">Problem</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#how-it-is-tested">Protocol</a> ·
  <a href="#results">Results</a> ·
  <a href="#what-carries-the-gain">Ablations</a> ·
  <a href="#how-far-the-result-holds">Robustness &amp; scaling</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#documentation">Docs</a>
</p>

---

## At a glance

SlackGNN estimates timing slack on a placed-and-routed netlist it has never trained on. It reads a directed gate-level graph from an OpenROAD finish, then predicts two values at each published endpoint: whether the endpoint violates its clock, and the slack in nanoseconds.

| Aspect | Detail |
| --- | --- |
| **Corpus** | 6 OpenROAD designs on SkyWater 130 nm high-density (`sky130hd`): gcd, aes, ibex, jpeg, riscv32i, chameleon |
| **Graph** | 79,856 nodes · 189,028 driver-to-load edges |
| **Supervision** | 32 published timing endpoints · 173 expanded route labels |
| **Protocol** | Leave-one-design-out · seed 0 · hidden width 32 · epoch cap 20 · patience 6 · CPU |
| **Headline** | SlackGNN **F1 0.8076** · **ROC-AUC 0.9583** · **MAE 0.6919 ns** · **RMSE 0.8667 ns** |
| **Compared with** | GCN, GraphSAGE, GIN on the same splits. SlackGNN has the best endpoint F1, ROC-AUC, and MAE |

> [!IMPORTANT]
> These 32 endpoints can rank the four architectures. Certifying a clock remains the job of the timer. SlackGNN is a screening model that sits beside static timing analysis, not inside signoff.

**Read next:** the full paper is [docs/slackgnn_final_research_report.md](docs/slackgnn_final_research_report.md). A short brief for engineers and hiring managers is [docs/executive_summary.md](docs/executive_summary.md).

---

## The problem

Static timing analysis answers the clock question for the paths it reports. A finish report from a production flow prints a small set of those paths. Across this corpus that set is **32 endpoints on 79,856 gates**, a supervision rate of about 0.04%.

A model that sees only those endpoints, and that is tested on a design it never trained on, has to transfer timing behavior through circuit structure. Three traps shape every design decision in this repository:

| Trap | What goes wrong | What SlackGNN does |
| --- | --- | --- |
| **Leakage** | Arrival time and required time determine slack, so a model that reads them can reconstruct the label by arithmetic | Arrival time, required time, and slack are supervision targets. They stay out of the encoder |
| **Split** | Endpoints from one design on both sides of a random split share cones of logic | Leave-one-design-out: the held-out design contributes no training labels |
| **Label poverty** | Four to seven endpoints per design make F1 jump when one endpoint flips | Report ROC-AUC and MAE beside F1, plus per-design tables |

---

## How it works

### From RTL to a graph dataset

Each design runs through OpenROAD-flow-scripts. The finish netlist becomes the graph. The finish timing report becomes the labels. The constraints and parasitics are kept with each design.

```mermaid
flowchart TB
    rtl["RTL · OpenROAD design catalog"] --> orfs["OpenROAD-flow-scripts · sky130hd · base flow"]
    orfs --> finish

    subgraph finish["Finish artifacts per design"]
        direction LR
        v["6_final.v<br/>gate-level netlist"]
        rpt["6_finish.rpt<br/>path table"]
        sdc["6_final.sdc<br/>clock and I/O delays"]
        spef["6_final.spef<br/>routed R and C"]
    end

    subgraph dataset["Graph dataset"]
        direction LR
        g["Signal-flow graph<br/>driver → load"]
        lab["Endpoint slack labels<br/>32 endpoints"]
        pyg["PyTorch Geometric Data<br/>data/corpus/‹design›/graph.pt"]
        exp["Route-label expansion<br/>32 → 173 training labels"]
        g --> pyg
        lab --> pyg
        pyg --> exp
    end

    v --> g
    rpt --> lab
    exp --> model["SlackGNN · leave-one-design-out"]
    model --> out["Endpoint violation and slack in ns"]

    classDef art fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    classDef kept fill:#f6f8fa,stroke:#8c959f,stroke-dasharray:4 3,color:#59636e
    classDef core fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef result fill:#c47b2b,stroke:#965616,color:#ffffff
    class rtl,orfs,v,rpt,g,lab,pyg art
    class sdc,spef kept
    class exp,model core
    class out result
```

<sub>Dashed boxes are stored with each design and are not read by the v3.0 encoder.</sub>

| Step | Command | Output |
| --- | --- | --- |
| Run one design through the flow | `scripts/run_orfs_design.sh` | OpenROAD-flow-scripts design in Docker |
| Parse, extract, label, convert | `scripts/process_design.py` | `data/<design>/pyg/graph.pt` from `netlist/6_final.v` and `reports/6_finish.rpt` |
| Publish the corpus | `scripts/build_corpus.py` | `data/corpus/<design>/` |
| Expand route labels | `scripts/expand_labels.py` | `reports/m9/expanded_labels.pt` |

### Growing supervision without inventing measurements

The finish report prints slack at endpoints only. Route-label expansion copies each reported slack along the shortest signal-flow route from the reported startpoint to its endpoint, so the training loss sees more than the single endpoint pin.

```mermaid
flowchart LR
    sa(["startpoint A"]) --> a1(("gate")) --> sh(("shared gate<br/>min of A and B"))
    sb(["startpoint B"]) --> b1(("gate")) --> sh
    sh --> a2(("gate")) --> ea(["endpoint A<br/>published slack restored"])
    sh --> b2(("gate")) --> eb(["endpoint B<br/>published slack restored"])

    classDef route fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef shared fill:#fcefed,stroke:#b4413c,color:#1f2328
    classDef ep fill:#c47b2b,stroke:#965616,color:#ffffff
    class sa,sb,a1,b1,a2,b2 route
    class sh shared
    class ea,eb ep
```

- Shared nodes keep the minimum slack of the routes that touch them.
- The published endpoint slack is restored afterward, so the endpoint row still matches the path table.
- Expansion lifts training labels from 32 to 173. **Evaluation returns to the 32 published endpoints**, so a good expanded-label score cannot replace a good endpoint score.

> [!NOTE]
> Copied slack is not a new static-timing measurement. Expanded metrics describe the training target. The endpoint table is the result.

### The model

SlackGNN is a node-level network on the driver-to-load graph. Message passing runs on every gate. The loss uses labeled nodes.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.png">
    <img src="docs/assets/architecture.png" alt="SlackGNN architecture: node, edge and timing features feed separate encoders, two timing-aware GraphSAGE attention layers, a shared embedding, a violation head and a slack head. Arrival, required and slack are loss targets only." width="100%">
  </picture>
</p>

- **Inputs.** Structural node features, edge features, logic depth, and reported-path membership. The edge delay channel is `log1p(driver fanout)`, normalized inside the design. SPEF resistance and capacitance are stored with the design and are withheld from this encoder.
- **Message passing.** Two attention-weighted message-passing layers aggregate drivers into each load. Scores are softmax-normalized over the drivers of each load.
- **Heads.** A shared embedding feeds a violation head and a slack head. The training loss is binary cross-entropy plus smooth L1, with weights 1.0 and 0.5.
- **Baselines.** GCN, GraphSAGE, and GIN use the published one-hot node type, with separate classification and regression heads.

<details>
<summary><b>Text-native version of the architecture (Mermaid)</b></summary>

```mermaid
flowchart TB
    subgraph inputs["Inputs"]
        direction LR
        nf["Node features · 10 ch<br/>gate type, register flag,<br/>fanin, fanout, degree, depth"]
        ef["Edge features · 10 ch<br/>source and target class,<br/>log1p fanout, load depth"]
        tf["Timing context · 3 ch<br/>logic depth, path membership,<br/>source distance"]
    end
    rpt["Path table<br/>arrival · required · slack"]

    nf --> ne["Node encoder"]
    ef --> ee["Edge encoder"]
    tf --> te["Timing encoder"]
    ne --> h0["h⁰ = node encoding + timing encoding"]
    te --> h0
    h0 --> l1["Timing-aware GraphSAGE attention · layer 1"]
    l1 --> l2["Timing-aware GraphSAGE attention · layer 2"]
    ee -- "edge state" --> l1
    te -- "timing state" --> l1
    l2 --> emb["Shared embedding"]
    emb --> cls["Violation head<br/>BCE with logits"]
    emb --> reg["Slack head<br/>Smooth L1"]
    cls --> loss["total = 1.0 · BCE + 0.5 · SmoothL1"]
    reg --> loss

    rpt -. "membership only" .-> tf
    rpt -. "targets only" .-> loss

    classDef inp fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    classDef core fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef guard fill:#fcefed,stroke:#b4413c,color:#1f2328
    class nf,ef,ne,ee inp
    class tf,te,h0,l1,l2,emb,cls,reg,loss core
    class rpt guard
```

<sub>The edge and timing states feed the attention scorer of both layers.</sub>

</details>

Feature tables, the attention scorer, and the loss are in [docs/slackgnn_architecture.md](docs/slackgnn_architecture.md).

---

## How it is tested

Leave-one-design-out is the evaluation. Each design is held out once. Training uses the other designs. Test scores use the published path-table slack of the held-out design.

```mermaid
flowchart LR
    subgraph tr["Train · 5 designs"]
        direction TB
        tl["Expanded route labels"]
        vl["Validation<br/>one label per class"]
    end
    m["Model<br/>Adam · lr 0.01<br/>≤ 20 epochs · patience 6"]
    subgraph ho["Held out · 1 design"]
        direction TB
        net["Full netlist<br/>no labels"]
        ep["Published endpoints<br/>path-table slack"]
    end
    tl --> m
    vl --> m
    net -. "structure only" .-> m
    m ==> ep
    ep --> mean["Mean over<br/>6 folds"]

    classDef inp fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    classDef core fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef result fill:#c47b2b,stroke:#965616,color:#ffffff
    class tl,vl,net inp
    class m core
    class ep,mean result
```

- The held-out design contributes no training rows. Message passing still runs on its full netlist, because an unlabeled gate can carry structural context to a labeled endpoint.
- Classification class 1 is a timing violation (slack strictly below zero). Regression scores slack in nanoseconds.
- F1 means skip a holdout when the model predicts no violations. ROC-AUC and MAE are defined on every holdout.

---

## Results

Leave-one-design-out endpoint means. Seed 0, hidden width 32, epoch cap 20, patience 6. F1 means skip a holdout when the model predicts no violations.

| Model | F1 ↑ | ROC-AUC ↑ | MAE (ns) ↓ | RMSE (ns) ↓ |
| --- | ---: | ---: | ---: | ---: |
| GCN | 0.5800 | 0.6243 | 0.9621 | 1.1922 |
| GraphSAGE | 0.3333 | 0.7632 | 0.8704 | 1.1053 |
| GIN | 0.5222 | 0.6951 | 0.9470 | 1.1712 |
| **SlackGNN** | **0.8076** | **0.9583** | **0.6919** | **0.8667** |

<p align="center">
  <img src="reports/figures/publication/f1_comparison.png" alt="Endpoint F1 for GCN, GraphSAGE, GIN, and SlackGNN" width="49%">
  <img src="reports/figures/publication/mae_comparison.png" alt="Endpoint MAE in nanoseconds for GCN, GraphSAGE, GIN, and SlackGNN" width="49%">
</p>

SlackGNN has the best endpoint F1, the best ROC-AUC, and the best MAE. The F1 gap from the next defined baseline mean, GCN at 0.5800, is 0.2276. The ROC-AUC gap from the next baseline, GraphSAGE at 0.7632, is 0.1951. The MAE gap from GraphSAGE, the best baseline on error, is 0.1785 ns.

> [!NOTE]
> **Reading the F1 column.** GraphSAGE's F1 of 0.3333 is the chameleon holdout. The other GraphSAGE folds predict no violations, so their F1 is undefined and is omitted from that mean. SlackGNN's endpoint F1 is undefined on aes for the same reason. On aes, SlackGNN ROC-AUC is 1.0 and MAE is 0.2578 ns. Even looking only at ROC-AUC and MAE, which are defined everywhere, SlackGNN leads.

### Per held-out design

| Held-out | Support | F1 | ROC-AUC | MAE (ns) | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gcd | 5 | 0.8000 | 1.0000 | 0.7045 | 0.8072 |
| aes | 4 | n/a | 1.0000 | 0.2578 | 0.2832 |
| ibex | 7 | 0.6667 | 1.0000 | 1.3993 | 1.8235 |
| jpeg | 6 | 1.0000 | 1.0000 | 0.9296 | 1.2074 |
| riscv32i | 4 | 1.0000 | 1.0000 | 0.2785 | 0.3401 |
| chameleon | 6 | 0.5714 | 0.7500 | 0.5815 | 0.7392 |

Five of the six holdouts have ROC-AUC 1.0. ibex is the expensive holdout for slack error: the model orders its endpoints and still misses their magnitudes by more than a nanosecond on average.

aes is the easiest holdout by mean defined endpoint F1 (0.8333). ibex is the hardest (0.4667).

<p align="center">
  <img src="reports/figures/publication/design_difficulty.png" alt="Holdout difficulty, easiest at the top" width="70%">
</p>

<details>
<summary><b>Expanded-route labels</b></summary>

On expanded-route labels, GraphSAGE F1 is 0.8094 and SlackGNN F1 is 0.7955. SlackGNN reaches expanded ROC-AUC 0.9027 and MAE 0.6848 ns. GraphSAGE reaches ROC-AUC 0.8082 and MAE 0.7431 ns. The expanded F1 edge for GraphSAGE is consistent with copied labels being a smoother target that a plain neighborhood aggregator can fit. Endpoint ranking and slack error still favor SlackGNN.

</details>

<details>
<summary><b>All publication figures</b></summary>

| | |
| :---: | :---: |
| <img src="reports/figures/publication/roc_auc_comparison.png" alt="Endpoint ROC-AUC by model" width="100%"> | <img src="reports/figures/publication/rmse_comparison.png" alt="Endpoint RMSE by model" width="100%"> |
| Endpoint ROC-AUC | Endpoint RMSE |
| <img src="reports/figures/publication/lodo_f1.png" alt="Leave-one-design-out F1 by held-out design" width="100%"> | <img src="reports/figures/publication/holdout_roc_auc.png" alt="Holdout ROC-AUC by held-out design" width="100%"> |
| Leave-one-design-out F1 | Holdout ROC-AUC |
| <img src="reports/figures/publication/holdout_mae.png" alt="Holdout MAE by held-out design" width="100%"> | <img src="reports/figures/publication/design_complexity.png" alt="Node count against edge count per design" width="100%"> |
| Holdout MAE | Design complexity |

Index: [reports/publication_figures.md](reports/publication_figures.md). Numerical source: [reports/generalization_results.md](reports/generalization_results.md).

</details>

---

## What carries the gain

The ablation retrains the full SlackGNN protocol and four controlled removals. The full model reproduces the generalization numbers exactly.

| Variant | F1 | ROC-AUC | MAE (ns) | RMSE (ns) | F1 change | MAE change (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full SlackGNN | 0.8076 | 0.9583 | 0.6919 | 0.8667 |  |  |
| No attention | 0.8343 | 0.9139 | 0.6464 | 0.8384 | +0.0267 | −0.0455 |
| No path membership | 0.7886 | 0.9514 | 0.7080 | 0.8773 | −0.0190 | +0.0162 |
| No timing criticality | 0.8933 | 0.8722 | 0.7360 | 0.8925 | +0.0857 | +0.0441 |
| No edge features | 0.8952 | 0.9583 | 0.6547 | 0.8379 | +0.0876 | −0.0372 |

```mermaid
flowchart TB
    full["Full SlackGNN<br/>F1 0.8076 · ROC-AUC 0.9583 · MAE 0.6919 ns"]
    full --> pm["Path membership<br/>only removal that lowers<br/>both F1 and MAE"]
    full --> tc["Timing criticality<br/>ROC-AUC → 0.8722<br/>MAE → 0.7360 ns"]
    full --> at["Attention<br/>ROC-AUC → 0.9139<br/>supports ranking"]
    full --> ed["Edge features<br/>ROC-AUC unchanged<br/>fanout proxy only"]

    classDef core fill:#c47b2b,stroke:#965616,color:#ffffff
    classDef key fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef minor fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    class full core
    class pm,tc key
    class at,ed minor
```

Path membership is the only removal that lowers both endpoint F1 and endpoint MAE. Timing criticality (logic depth and source distance) is doing ranking and calibration. Attention helps ranking. The current edge features are not the source of the endpoint ROC-AUC.

> [!TIP]
> F1 changes in the ablation table inherit the undefined-fold issue: part of an F1 increase is a change in which designs enter the mean. ROC-AUC and MAE do not have that ambiguity, and they are the basis for the criticality and attention conclusions. Details: [docs/slackgnn_final_research_report.md §9](docs/slackgnn_final_research_report.md#9-ablation-results).

---

## How far the result holds

### Robustness

SlackGNN remains ahead of GCN, GraphSAGE, and GIN when labels are thinned, noised, or drawn from fewer reported paths.

| Condition | Holds through | Collapses at |
| --- | --- | --- |
| Sparse labels | 25% of the expanded labels (F1 0.8250, ROC-AUC 0.8097, MAE 0.7930 ns) | 10% labels (F1 0.6143, ROC-AUC 0.7264, MAE 0.9282 ns) |
| Label noise | 20% noise (F1 0.8200, ROC-AUC 0.9514, MAE 0.7497 ns) | no tested setting |
| Missing paths | 50% of training paths removed (F1 0.9000, ROC-AUC 0.9236, MAE 0.7210 ns) | no tested setting |
| Corpus size | 6 designs are the smallest stable corpus | 4 designs (F1 0.6750, MAE 0.9890 ns) |

Across the degraded conditions, mean endpoint F1 is 0.8022 for SlackGNN, 0.5046 for GIN, 0.4942 for GCN, and 0.4245 for GraphSAGE.

<p align="center">
  <img src="reports/figures/robustness/model_robustness.png" alt="Mean endpoint F1 and MAE under degraded conditions for each model" width="85%">
</p>

> [!TIP]
> Path deletion barely moves the model. A new design helps the corpus when its finish report adds endpoints. A larger netlist whose report reprints the same few paths adds edges and adds little supervision.

### Scaling

| Model | Training time (s) | Inference time (s) | PyTorch step (MB) |
| --- | ---: | ---: | ---: |
| GCN | 29.39 | 0.0428 | 14.37 |
| GraphSAGE | 26.21 | 0.0253 | 19.60 |
| GIN | 43.84 | 0.0306 | 19.33 |
| SlackGNN | 67.61 | 0.1647 | 50.97 |

Training time follows edge count more closely than node count. The mean log-log slope against edges is 0.45. jpeg accounts for about 38% of a SlackGNN epoch, and the backward pass is the heavier part. A full six-design holdout is about 1.1 minutes on the measured CPU. A linear projection to 50 average-sized designs is about 92 minutes if every fold still runs 20 epochs.

Full analyses: [robustness](docs/slackgnn_robustness_analysis.md) · [scaling](docs/slackgnn_scaling_analysis.md).

---

## Corpus

| Design | Nodes | Edges | Endpoints | Met | Violations | Expanded labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 527 | 928 | 5 | 3 | 2 | 27 |
| aes | 15,868 | 38,215 | 4 | 3 | 1 | 20 |
| ibex | 18,000 | 44,436 | 7 | 5 | 2 | 42 |
| jpeg | 36,283 | 82,643 | 6 | 5 | 1 | 23 |
| riscv32i | 5,997 | 15,884 | 4 | 2 | 2 | 33 |
| chameleon | 3,181 | 6,922 | 6 | 4 | 2 | 28 |
| **Total** | **79,856** | **189,028** | **32** | **22** | **10** | **173** |

`data/designs.json` also names uart and spi. Those two designs are absent from the measured corpus.

![Nodes, edges, and expanded labels by design](reports/figures/publication/corpus_composition.png)

---

## Quick start

### Installation

Python 3.11. PyTorch Geometric 2.8 sets the PyTorch window at 2.9 through 2.12. The published measurements used a CPU wheel.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts\check_python.py
```

Unix activation is `source .venv/bin/activate`. Create that environment with `python3.11 -m venv .venv`.

A CUDA wheel, when required, comes from the [PyTorch install selector](https://pytorch.org/get-started/locally/) before `requirements.txt`. Contributor workflow is in [docs/development.md](docs/development.md).

### Smoke test

```powershell
python scripts\check_python.py
python -m pytest
python scripts\corpus_summary.py
```

`corpus_summary.py` reads `data/corpus/corpus_metadata.json` and checks that each published design still has `graph.pt`. The pytest suite uses `pythonpath = ["src"]` from `pyproject.toml`.

### Training

The published leave-one-design-out comparison trains GCN, GraphSAGE, GIN, and SlackGNN at seed 0, hidden width 32, and an epoch cap of 20:

```powershell
python scripts\expand_labels.py
python scripts\generalization_benchmark.py
```

`expand_labels.py` is required once. The benchmark reads `reports/m9/expanded_labels.pt` and writes `reports/generalization_results.json` and `reports/generalization_results.md`. On this corpus the protocol training time is on the order of a few minutes on CPU.

`scripts/train.py` is the single-model entry point. Its default epoch cap is 30.

```powershell
python scripts\train.py --model slackgnn --task multitask --epochs 20 --hidden 32 --seed 0
python scripts\train.py --model gcn --epochs 20 --hidden 32 --seed 0
```

`--task` accepts `classification`, `regression`, or `multitask`. `--alpha` and `--beta` default to 1.0 and 0.5.

### Evaluation

Published metrics already live under `reports/`. Recomputing them retrains the same protocol:

```powershell
python scripts\generalization_benchmark.py
python scripts\feature_ablation.py
python scripts\robustness_analysis.py
python scripts\scaling_analysis.py
python scripts\publication_figures.py
```

| Command | Writes |
| --- | --- |
| `generalization_benchmark.py` | `reports/generalization_results.md` |
| `feature_ablation.py` | `reports/ablation_results.md` |
| `robustness_analysis.py` | `reports/robustness_results.md` |
| `scaling_analysis.py` | `reports/scaling_results.md` |
| `publication_figures.py` | `reports/figures/publication/` |

---

## Who this is for

| Reader | What to take from it | Start with |
| --- | --- | --- |
| **EDA engineers** | A fast screen for which unseen blocks look timing-critical before a full path report is read | [Executive summary](docs/executive_summary.md) |
| **ML engineers** | A reproducible graph-learning benchmark: directed netlists, a leakage rule, a design holdout, and published tables for generalization, ablation, robustness, and scaling | [Final research report](docs/slackgnn_final_research_report.md) |
| **Hiring managers** | An end-to-end ML-for-EDA project: an OpenROAD corpus, a graph dataset, a multitask model, controlled ablations, and an explicit statement of what the evidence does not support | [Research highlights](docs/research_highlights.md) |

---

## Documentation

| Document | Role |
| --- | --- |
| [docs/slackgnn_final_research_report.md](docs/slackgnn_final_research_report.md) | Full paper |
| [docs/executive_summary.md](docs/executive_summary.md) | One-to-two page brief |
| [docs/research_highlights.md](docs/research_highlights.md) | One-page highlights |
| [docs/slackgnn_architecture.md](docs/slackgnn_architecture.md) | Feature channels, attention, heads, leakage rule |
| [docs/RELEASE_v3.0.md](docs/RELEASE_v3.0.md) | Release notes |
| [docs/INDEX.md](docs/INDEX.md) | Documentation index |

`README_M9.md` and `docs/m9_research_report.md` record the earlier four-design experiment (gcd, aes, ibex, jpeg). The six-design table in this README is the published result.

## Repository structure

```text
src/slackgnn/     library: dataset, features, models, studies
scripts/          command-line entry points
tests/            pytest suite
data/             designs, finish artifacts, published graphs
reports/          metrics, JSON, and figures
docs/             papers, architecture, and release notes
tcl/              EDA helper scripts
notebooks/        local exploration
results/          local experiment outputs
```

Folder ownership is in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md). Files that stay off GitHub are listed in [docs/repository_cleanup_report.md](docs/repository_cleanup_report.md).

---

## What the evidence does not support

> [!CAUTION]
> - **Signoff.** Thirty-two endpoints can rank these models. Certifying a clock remains the job of static timing analysis.
> - **Seed variance.** One seed. A second seed is future work.
> - **Other technologies.** One PDK: sky130hd. Cross-technology transfer is a hypothesis.
> - **Parasitics.** SPEF is stored with each design and is withheld from the edge encoder. The delay channel is normalized `log1p(driver fanout)`.
> - **Label floor.** The model collapses at 10% of the expanded labels. Six designs are the smallest corpus that keeps slack error within 0.10 ns of the full result under the study's stability rule.

## Roadmap

- More designs, including the uart and spi names already listed in `data/designs.json`
- More printed endpoints, so the label set can grow past 173
- A second PDK beside sky130hd
- Richer edge features
- A SPEF-aware encoder that joins resistance and capacitance into the edge channels

The milestone list is [ROADMAP.md](ROADMAP.md). Release history is [CHANGELOG.md](CHANGELOG.md). Contribution rules are [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

The Python package version in `pyproject.toml` is 0.1.0. This research release is labeled v3.0.

```bibtex
@software{slackgnn2026,
  title        = {SlackGNN: Timing-Aware Graph Neural Networks for OpenROAD Timing Prediction},
  year         = {2026},
  version      = {3.0},
  howpublished = {Research release}
}
```

## License

This repository has no `LICENSE` file yet. Until a maintainer adds one, GitHub does not grant a license to reuse the code. Choose a license and add it before the first public push.
