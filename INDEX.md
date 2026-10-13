# Documentation index

<p>
  <img alt="Published study: six designs" src="https://img.shields.io/badge/published%20study-six%20designs-c47b2b?style=flat-square">
  <img alt="Earlier record: four designs" src="https://img.shields.io/badge/earlier%20record-four%20designs-8c959f?style=flat-square">
</p>

The published evaluation is the six-design study. Start with the final report. The earlier four-design write-ups remain in the tree as the record of that experiment.

## Where to start

```mermaid
flowchart LR
    you(["Your question"]) --> q1(["What is the result?"])
    you --> q2(["How does the model work?"])
    you --> q3(["Can I reproduce or extend it?"])

    q1 --> hl["research_highlights.md<br/>one page"]
    hl --> es["executive_summary.md<br/>brief"]
    es --> fr["slackgnn_final_research_report.md<br/>full paper"]

    q2 --> arch["slackgnn_architecture.md<br/>features · attention · heads"]
    arch --> ds["dataset_structure.md<br/>netlist · SDC · SPEF · reports"]

    q3 --> dev["development.md<br/>Python 3.11 environment"]
    dev --> ps["PROJECT_STRUCTURE.md<br/>folder ownership"]
    ps --> contrib["../CONTRIBUTING.md<br/>tests and pull requests"]

    classDef start fill:#c47b2b,stroke:#965616,color:#ffffff
    classDef ask fill:#f8ead8,stroke:#c47b2b,color:#1f2328
    classDef doc fill:#f6f8fa,stroke:#8c959f,color:#1f2328
    class you start
    class q1,q2,q3 ask
    class hl,es,fr,arch,ds,dev,ps,contrib doc
```

## Release documents

| Document | What it covers |
| --- | --- |
| [slackgnn_final_research_report.md](slackgnn_final_research_report.md) | Final research report: dataset, model, generalization, ablation, robustness, scaling |
| [executive_summary.md](executive_summary.md) | Brief for hiring managers, EDA engineers, and ML engineers |
| [research_highlights.md](research_highlights.md) | One page: problem, method, results, impact |
| [RELEASE_v3.0.md](RELEASE_v3.0.md) | v3.0 highlights, corpus, figures, limitations, GitHub assets |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Folder ownership for `src/`, `scripts/`, `data/`, `reports/`, and `docs/` |
| [repository_cleanup_report.md](repository_cleanup_report.md) | Large artifacts and what stays outside the GitHub source tree |

Repository root companions: [../README.md](../README.md), [../CHANGELOG.md](../CHANGELOG.md), [../ROADMAP.md](../ROADMAP.md), [../CONTRIBUTING.md](../CONTRIBUTING.md).

## Architecture

| Document | What it covers |
| --- | --- |
| [slackgnn_architecture.md](slackgnn_architecture.md) | Feature channels, attention message passing, multitask heads, leakage rule |
| [dataset_structure.md](dataset_structure.md) | GCD finish layout: netlist, SDC, SPEF, timing reports |
| [development.md](development.md) | Python 3.11 environment, directory ownership, dependency policy |

> [!NOTE]
> `slackgnn_architecture.md` describes the model. Its leave-one-design-out sentence names the earlier four-design rotation. The six-design protocol and scores are in the final research report.

Architecture figures for decks and the repository page live in [assets/](assets/): `architecture.png`, `architecture-dark.png`, and the social-preview `banner.png`, each with its SVG source.

## Study reports

| Study | Narrative | Tables |
| --- | --- | --- |
| Generalization | Final report, section 8 | [../reports/generalization_results.md](../reports/generalization_results.md) |
| Ablation | Final report, section 9 | [../reports/ablation_results.md](../reports/ablation_results.md) |
| Robustness | [slackgnn_robustness_analysis.md](slackgnn_robustness_analysis.md) | [../reports/robustness_results.md](../reports/robustness_results.md) |
| Scaling | [slackgnn_scaling_analysis.md](slackgnn_scaling_analysis.md) | [../reports/scaling_results.md](../reports/scaling_results.md) |
| Publication figures | [../reports/publication_figures.md](../reports/publication_figures.md) | `reports/figures/publication/` |

## Earlier experiment

> [!WARNING]
> These files record the four-design run on gcd, aes, ibex, and jpeg. They are the history of that run. The six-design numbers live in the final report and in `reports/generalization_results.md`.

| Document | What it covers |
| --- | --- |
| [m9_research_report.md](m9_research_report.md) | Four-design M9 narrative |
| [slackgnn_research_report.md](slackgnn_research_report.md) | Earlier research note |
| [slackgnn_next_phase.md](slackgnn_next_phase.md) | Note written by the baseline trainer |
| [../README_M8.md](../README_M8.md) | SlackGNN architecture phase on four designs |
| [../README_M9.md](../README_M9.md) | Four-design label expansion and studies |
| [../README_dataset.md](../README_dataset.md) | How the curated GCD finish was imported |

```mermaid
flowchart LR
    v1["v1.0 · Dataset pipeline<br/>README_dataset.md"] --> v2["v2.0 · SlackGNN on four designs<br/>README_M8.md · slackgnn_research_report.md"]
    v2 --> m9["M9 · four-design studies<br/>README_M9.md · m9_research_report.md"]
    m9 --> v3["v3.0 · six-design study<br/>slackgnn_final_research_report.md"]

    classDef old fill:#f6f8fa,stroke:#8c959f,color:#59636e
    classDef now fill:#c47b2b,stroke:#965616,color:#ffffff
    class v1,v2,m9 old
    class v3 now
```
