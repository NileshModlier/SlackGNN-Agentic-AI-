# SlackGNN feature ablations

Leave-one-design-out on the expanded labels. Endpoint metrics use published slack.

Endpoint metrics use published path-table slack. Path membership is timing column 1. Timing criticality is logic depth and source distance.

Designs: gcd, aes, ibex, jpeg, riscv32i, chameleon. Seed 0.

## Endpoint means

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.8179 | 0.8133 | 0.7500 | 0.8076 | 0.9583 | 0.6919 | 1.0294 | 0.8667 |
| no_attention | 0.8317 | 0.7467 | 0.8333 | 0.8343 | 0.9139 | 0.6464 | 1.0486 | 0.8384 |
| no_path_membership | 0.7841 | 0.6933 | 0.8333 | 0.7886 | 0.9514 | 0.7080 | 1.1115 | 0.8773 |
| no_timing_criticality | 0.8873 | 0.9333 | 0.7500 | 0.8933 | 0.8722 | 0.7360 | 1.1092 | 0.8925 |
| no_edge_features | 0.8833 | 0.8444 | 1.0000 | 0.8952 | 0.9583 | 0.6547 | 0.9976 | 0.8379 |

## Expanded-label means

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.7628 | 0.8338 | 0.7877 | 0.7955 | 0.9027 | 0.6848 | 0.9907 | 0.8989 |
| no_attention | 0.7440 | 0.8265 | 0.7595 | 0.7708 | 0.8861 | 0.6692 | 0.9752 | 0.8839 |
| no_path_membership | 0.7858 | 0.8318 | 0.8226 | 0.8120 | 0.8896 | 0.6962 | 0.9590 | 0.8875 |
| no_timing_criticality | 0.7593 | 0.9073 | 0.7170 | 0.7317 | 0.9024 | 0.6924 | 1.0082 | 0.9053 |
| no_edge_features | 0.7823 | 0.8407 | 0.8263 | 0.8202 | 0.8876 | 0.6854 | 0.9939 | 0.9012 |

## Change from the full model

| Variant | Endpoint F1 | F1 change | Endpoint MAE (ns) | MAE change (ns) |
| --- | ---: | ---: | ---: | ---: |
| no_attention | 0.8343 | 0.0267 | 0.6464 | -0.0455 |
| no_path_membership | 0.7886 | -0.0190 | 0.7080 | 0.0162 |
| no_timing_criticality | 0.8933 | 0.0857 | 0.7360 | 0.0441 |
| no_edge_features | 0.8952 | 0.0876 | 0.6547 | -0.0372 |

Largest endpoint F1 drop: no_path_membership.
Largest endpoint MAE increase: no_timing_criticality.

F1 means skip a holdout when that model predicts no violations, so those means can cover different designs. ROC-AUC and MAE are computed on every holdout.

Removing path membership is the only change that lowers both endpoint F1 and MAE. Removing timing criticality raises MAE the most and lowers ROC-AUC from 0.9583 to 0.8722. Removing attention lowers ROC-AUC to 0.9139. Removing edge features leaves endpoint ROC-AUC at 0.9583.
