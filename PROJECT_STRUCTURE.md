# Project structure

SlackGNN is a Python 3.11 source tree. Importable code lives in `src/slackgnn`. Scripts are thin entry points. Design data, metrics, and prose each have their own top-level directory.

```text
SlackGNN Agentic AI/
├── src/slackgnn/     library
├── scripts/          command-line entry points
├── tests/            pytest modules
├── data/             finish artifacts and published graphs
├── reports/          metrics, JSON, and figures
├── docs/             papers and contributor notes
├── tcl/              EDA helper scripts
├── notebooks/        local exploration
├── results/          local experiment outputs
├── pyproject.toml    package metadata and pytest config
└── requirements.txt  pinned development environment
```

Resolve checkout paths through `slackgnn.paths` inside the library. Scripts add `src` to `sys.path` so they run from a fresh checkout.

## How the pieces connect

```mermaid
flowchart TB
    finish[("data/‹design›/<br/>finish netlist and report")] --> ds["dataset/<br/>reader · parser · graph · PyG"]
    ds --> corpus[("data/corpus/<br/>graph.pt per design")]
    corpus --> ex["expansion/<br/>route-label expansion"]
    ex --> lbl[("reports/m9/<br/>expanded_labels.pt")]
    corpus --> ft["features/<br/>node · edge · timing tensors"]
    finish -. "path membership" .-> ft
    ft --> tr["training/<br/>GCN · GraphSAGE · GIN · SlackGNN"]
    lbl --> tr
    tr --> st["m9/<br/>six-design studies"]
    tr --> xp["explainability/<br/>critical-gate rankings"]
    st --> rep[("reports/<br/>tables · JSON · figures")]
    xp --> rep

    classDef pkg fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef store fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    class ds,ex,ft,tr,st,xp pkg
    class finish,corpus,lbl,rep store
```

<sub>Orange boxes are packages under `src/slackgnn/`. Grey cylinders are files on disk.</sub>

## `src/`

`src/slackgnn/` is the installable package (`pip install -e .`). The package version in `src/slackgnn/__init__.py` matches `pyproject.toml` and is currently 0.1.0.

| Package | Responsibility |
| --- | --- |
| `dataset/` | Verilog netlist reader, timing-report parser, signal-flow graph, PyG `Data` builder, corpus publisher, per-design processing |
| `expansion/` | Design discovery and route-label expansion. Copies reported slack onto the shortest signal-flow route |
| `features/` | Node, edge, and timing tensors. Arrival, required, and slack stay out of the encoder |
| `training/` | Baseline and SlackGNN training, splits, metrics, figures. `training/models/` holds GCN, GraphSAGE, GIN, and SlackGNN |
| `explainability/` | Critical-gate rankings from a trained model |
| `m9/` | Six-design studies: generalization, feature ablation, robustness, scaling, publication figures, and the study trainer |
| `paths.py` | Repository directory locations |
| `runtime.py` | Python 3.11 interpreter check used by `scripts/check_python.py` |

> [!NOTE]
> `m9/studies.py` is the earlier four-design study runner. The six-design commands call `generalization.py`, `feature_ablation.py`, `robustness.py`, `scaling.py`, and `publication.py`.

## `scripts/`

Each script is a command-line wrapper around one library function.

| Group | Scripts |
| --- | --- |
| Flow and import | `run_orfs_design.sh`, `import_gcd_dataset.py`, `discover_designs.py`, `process_design.py` |
| Graph dataset | `parse_timing_report.py`, `extract_graph.py`, `build_dataset.py`, `build_pyg_dataset.py`, `build_corpus.py`, `dataset_summary.py`, `dataset_validation.py`, `validate_corpus.py`, `corpus_summary.py`, `pyg_dataset_summary.py` |
| Labels | `expand_labels.py`, `expand_corpus.py` |
| Models | `train.py` |
| Six-design studies | `generalization_benchmark.py`, `feature_ablation.py`, `robustness_analysis.py`, `scaling_analysis.py`, `publication_figures.py` |
| Earlier four-design studies | `cross_design_benchmark.py`, `ablation_study.py`, `robustness_study.py`, `scaling_study.py`, `m9_figures.py`, `run_m9_studies.py`, `write_m9_docs.py` |
| Environment | `check_python.py` |

## `data/`

`data/designs.json` is the discovery list. The measured corpus is the six directories that contain a published graph.

```text
data/
├── designs.json
├── gcd/ | aes/ | ibex/ | jpeg/ | riscv32i/ | chameleon/
│   ├── netlist/6_final.v
│   ├── sdc/6_final.sdc
│   ├── spef/6_final.spef
│   ├── reports/6_finish.rpt
│   ├── parsed/          path labels
│   ├── graphs/          NetworkX pickle
│   ├── dataset/         labeled graph pickle
│   └── pyg/graph.pt     per-design PyG object
└── corpus/
    ├── corpus_metadata.json
    └── <design>/graph.pt + metadata.json
```

`data/corpus/<design>/graph.pt` is the graph the studies load. The per-design `pyg/graph.pt` is the same object kept next to the finish files. GCD is the small curated example and is described in [dataset_structure.md](dataset_structure.md). Which of these files belong in git is in [repository_cleanup_report.md](repository_cleanup_report.md).

## `reports/`

Metrics and figures. The six-design tables sit at the top of this directory. `reports/m9/` holds the earlier four-design study, plus `expanded_labels.pt`, which the six-design trainer still loads.

| Path | Contents |
| --- | --- |
| `generalization_results.md` | Published endpoint table |
| `ablation_results.md` | Full model against four removals |
| `robustness_results.md` | Sparse labels, noise, reduced corpus, missing paths |
| `robustness/` | JSON for each robustness condition |
| `scaling_results.md` | Runtime, memory, complexity, projections |
| `scaling/` | JSON for each scaling measurement |
| `publication_figures.md` | Index of the publication PNGs |
| `figures/publication/` | Corpus, model, and holdout figures at 300 dpi |
| `figures/robustness/` | Robustness curves |
| `figures/scaling/` | Runtime and memory scatters |
| `figures/ablation_f1_mae.png` | Ablation bars |
| `m9/` | Four-design study outputs and `expanded_labels.pt` |
| `explainability/` | Attention and ranking JSON from the model trainer |

## `docs/`

Prose for readers and contributors. [INDEX.md](INDEX.md) is the map. The final paper, executive summary, research highlights, architecture note, robustness analysis, and scaling analysis all live here. `docs/assets/` holds the banner and the architecture figure, each with its SVG source.

## Other top-level folders

| Path | Role |
| --- | --- |
| `tests/` | pytest modules. They import `slackgnn` and rely on `pythonpath = ["src"]`. |
| `tcl/` | EDA tool scripts. They read and write files under `data/` and `results/`. |
| `notebooks/` | Exploratory notebooks. Checkpoints stay untracked. |
| `results/` | Local workspace. Git keeps `results/.gitkeep` and ignores the rest. |

## Files the studies depend on

> [!IMPORTANT]
> A checkout that retrains the published comparison needs:
>
> - `data/corpus/<design>/graph.pt` for the six designs
> - `data/<design>/` finish reports and graphs, so feature enrichment can rebuild path membership
> - `reports/m9/expanded_labels.pt`

Generating those inputs is the dataset pipeline in the README. The cleanup report says which of the bulky finish files stay on the local disk.
