<p align="center">
  <img src="assets/banner.png" alt="SlackGNN v3.0 research release banner with the six designs and the headline F1, ROC-AUC and MAE" width="100%">
</p>

# SlackGNN v3.0

**Timing-Aware Graph Neural Networks for OpenROAD Timing Prediction**

<p>
  <img alt="Release v3.0 research" src="https://img.shields.io/badge/release-v3.0%20research-c47b2b?style=flat-square">
  <img alt="Package 0.1.0" src="https://img.shields.io/badge/package-0.1.0-8c959f?style=flat-square">
  <img alt="Corpus: 6 sky130hd designs" src="https://img.shields.io/badge/corpus-6%20sky130hd%20designs-2f6f4e?style=flat-square">
  <img alt="Protocol: leave-one-design-out" src="https://img.shields.io/badge/protocol-leave--one--design--out-1f4e79?style=flat-square">
</p>

v3.0 is the research release of the six-design study. The installable package version in `pyproject.toml` remains 0.1.0. This document is the release label for the results, the corpus, and the figures.

## Highlights

- SlackGNN predicts endpoint violation and slack on an OpenROAD design held out of training.
- The corpus is six sky130hd designs: gcd, aes, ibex, jpeg, riscv32i, and chameleon.
- SlackGNN leads GCN, GraphSAGE, and GIN on endpoint F1, ROC-AUC, and MAE.
- Path membership and timing depth carry the gain. The model holds through 25% of the expanded labels and collapses at 10%.
- Training time follows edge count. jpeg accounts for about 38% of a SlackGNN epoch.
- The paper, the executive brief, and the one-page highlights are in `docs/`.

```mermaid
flowchart LR
    c["Corpus<br/>6 designs · 32 endpoints<br/>173 route labels"] --> g["Generalization<br/>leave-one-design-out"]
    g --> a["Ablation<br/>4 removals"]
    g --> rb["Robustness<br/>labels · noise · corpus · paths"]
    g --> sc["Scaling<br/>runtime · memory · projections"]
    a & rb & sc --> doc["Paper · brief · highlights<br/>publication figures"]

    classDef data fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    classDef study fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef out fill:#c47b2b,stroke:#965616,color:#ffffff
    class c data
    class g,a,rb,sc study
    class doc out
```

## Results

Protocol: leave-one-design-out, seed 0, hidden width 32, epoch cap 20, patience 6, CPU. F1 means skip a holdout when the model predicts no violations.

| Model | F1 ↑ | ROC-AUC ↑ | MAE (ns) ↓ | RMSE (ns) ↓ |
| --- | ---: | ---: | ---: | ---: |
| GCN | 0.5800 | 0.6243 | 0.9621 | 1.1922 |
| GraphSAGE | 0.3333 | 0.7632 | 0.8704 | 1.1053 |
| GIN | 0.5222 | 0.6951 | 0.9470 | 1.1712 |
| **SlackGNN** | **0.8076** | **0.9583** | **0.6919** | **0.8667** |

> [!NOTE]
> GraphSAGE’s F1 is the chameleon holdout. The other GraphSAGE folds predict no violations. SlackGNN’s endpoint F1 is undefined on aes. On that fold ROC-AUC is 1.0 and MAE is 0.2578 ns. On expanded-route labels, GraphSAGE F1 is 0.8094 and SlackGNN F1 is 0.7955.

aes is the easiest holdout (mean defined endpoint F1 0.8333). ibex is the hardest (0.4667).

Source: `reports/generalization_results.md`. Narrative: `docs/slackgnn_final_research_report.md`.

## Corpus

| Design | Nodes | Edges | Endpoints | Expanded labels |
| --- | ---: | ---: | ---: | ---: |
| gcd | 527 | 928 | 5 | 27 |
| aes | 15,868 | 38,215 | 4 | 20 |
| ibex | 18,000 | 44,436 | 7 | 42 |
| jpeg | 36,283 | 82,643 | 6 | 23 |
| riscv32i | 5,997 | 15,884 | 4 | 33 |
| chameleon | 3,181 | 6,922 | 6 | 28 |
| **Total** | **79,856** | **189,028** | **32** | **173** |

Platform: SkyWater 130 nm high-density, OpenROAD-flow-scripts base flow. Expanded labels copy each reported slack along the shortest signal-flow route. Endpoint scores use the path table.

![Cumulative nodes, edges, and labels from gcd through chameleon](../reports/figures/publication/corpus_growth.png)

## Figures

Publication PNGs, 300 dpi, live in `reports/figures/publication/`.

| Figure | File |
| --- | --- |
| Corpus composition | `corpus_composition.png` |
| Corpus growth | `corpus_growth.png` |
| Design complexity | `design_complexity.png` |
| F1 comparison | `f1_comparison.png` |
| ROC-AUC comparison | `roc_auc_comparison.png` |
| MAE comparison | `mae_comparison.png` |
| RMSE comparison | `rmse_comparison.png` |
| Leave-one-design-out F1 | `lodo_f1.png` |
| Holdout ROC-AUC | `holdout_roc_auc.png` |
| Holdout MAE | `holdout_mae.png` |
| Design difficulty | `design_difficulty.png` |

Supporting curves are in `reports/figures/robustness/` and `reports/figures/scaling/`. The index is `reports/publication_figures.md`.

## GitHub assets

The banner and the standalone architecture figure are now drawn and live in `docs/assets/`, each with an editable SVG source.

| Asset | File | Use |
| --- | --- | --- |
| Banner | `docs/assets/banner.png` (1280×640), `banner.svg` | Title, the six design names, and the three headline numbers (F1 0.8076, ROC-AUC 0.9583, MAE 0.6919 ns). Set it as the GitHub social preview under *Settings → General → Social preview*. The README opens with it |
| Architecture figure | `docs/assets/architecture.png`, `architecture-dark.png`, `architecture.svg` | Light and dark versions of the same diagram. The README and `docs/slackgnn_architecture.md` switch between them with `<picture>`. Use the light PNG in a deck |
| Mermaid diagrams | `README.md`, `docs/slackgnn_architecture.md` | Text-native pipeline, protocol, label-expansion, and architecture diagrams that GitHub renders in place |
| README figures | `corpus_composition.png`, `f1_comparison.png`, `mae_comparison.png`, and `design_difficulty.png` | Embedded in the README. The other publication figures sit in a collapsible gallery |
| Performance figure | `f1_comparison.png` | The primary chart. Pair it with `mae_comparison.png` when the slide has room for slack error |

## Known limitations

> [!CAUTION]
> - Thirty-two endpoints can rank these models. Certifying a clock remains the job of static timing analysis.
> - One seed. A second seed is future work.
> - One PDK: sky130hd.
> - SPEF is stored with each design and is withheld from the edge encoder. The delay channel is normalized `log1p(driver fanout)`.
> - The model collapses at 10% of the expanded labels. Six designs are the smallest corpus that keeps slack error within 0.10 ns of the full result under the study’s stability rule.
> - uart and spi are named in `data/designs.json` and are absent from the measured corpus.
> - No `LICENSE` file is present. Add one before a public push.
> - The Python package version is 0.1.0. v3.0 is the research release label.
