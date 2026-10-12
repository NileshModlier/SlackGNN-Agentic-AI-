# Ablation study

Each row is the leave-one-design-out mean on published endpoint labels. Training still uses expanded labels.

Winner by mean endpoint F1: `classification_only` (F1 0.8111, MAE 1.2542 ns).

| Ablation | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.7536 | 0.6042 | 0.8750 | 0.6750 | 0.9500 | 0.9890 | 1.9571 | 1.1962 |
| no_timing_context | 0.6554 | 0.4167 | 0.6250 | 0.5333 | 0.9083 | 1.0251 | 2.0605 | 1.2393 |
| no_edge_features | 0.7893 | 0.7292 | 0.8750 | 0.7167 | 0.9000 | 0.9488 | 1.7644 | 1.1205 |
| no_attention | 0.7893 | 0.7292 | 0.8750 | 0.7167 | 0.9333 | 0.9936 | 2.0329 | 1.2350 |
| classification_only | 0.7714 | 0.7381 | 1.0000 | 0.8111 | 1.0000 | 1.2542 | 2.5411 | 1.4928 |
| regression_only | 0.3583 | 0.0000 | 0.0000 | n/a | 0.0833 | 1.0302 | 2.1332 | 1.2616 |

Missing designs:

- `riscv32i`: design directory is missing
- `microwatt`: design directory is missing
- `chameleon`: design directory is missing

Runtime: 273.3 s.

## full

Timing context, edge features, learned attention, and both losses.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 1.0000 | 0.6342 | 0.4813 | 0.6938 | 5 |
| aes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2298 | 0.0785 | 0.2801 | 4 |
| ibex | 0.7143 | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 1.4612 | 3.3940 | 1.8423 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 | 1.6308 | 3.8748 | 1.9684 | 6 |

Mean endpoint: Accuracy 0.7536, Precision 0.6042, Recall 0.8750, F1 0.6750, ROC-AUC 0.9500, MAE (ns) 0.9890, MSE 1.9571, RMSE (ns) 1.1962

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8722 | 0.8734 | 0.8475 | 0.9206 | 27 |
| aes | 0.9000 | 0.9091 | 0.9091 | 0.9091 | 0.9596 | 0.3555 | 0.2453 | 0.4952 | 20 |
| ibex | 0.7381 | 0.7600 | 0.7917 | 0.7755 | 0.8611 | 1.1529 | 2.7579 | 1.6607 | 42 |
| jpeg | 0.3913 | 0.2500 | 0.2000 | 0.2222 | 0.5538 | 1.2903 | 2.7236 | 1.6503 | 23 |

Mean expanded: Accuracy 0.6833, Precision 0.6428, Recall 0.7252, F1 0.6741, ROC-AUC 0.8117, MAE (ns) 0.9180, MSE 1.6436, RMSE (ns) 1.1817

## no_timing_context

Timing-context channels are zeroed before message passing.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 0.8333 | 0.7424 | 0.6928 | 0.8324 | 5 |
| aes | 0.7500 | n/a | 0.0000 | n/a | 1.0000 | 0.2244 | 0.0668 | 0.2584 | 4 |
| ibex | 0.5714 | 0.3333 | 0.5000 | 0.4000 | 0.8000 | 1.4778 | 3.4959 | 1.8697 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 | 1.6558 | 3.9864 | 1.9966 | 6 |

Mean endpoint: Accuracy 0.6554, Precision 0.4167, Recall 0.6250, F1 0.5333, ROC-AUC 0.9083, MAE (ns) 1.0251, MSE 2.0605, RMSE (ns) 1.2393

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8611 | 0.9037 | 0.9061 | 0.9519 | 27 |
| aes | 0.8500 | 0.9000 | 0.8182 | 0.8571 | 0.9495 | 0.4513 | 0.3214 | 0.5669 | 20 |
| ibex | 0.7143 | 0.7143 | 0.8333 | 0.7692 | 0.8449 | 1.3084 | 3.2019 | 1.7894 | 42 |
| jpeg | 0.6522 | 0.5556 | 1.0000 | 0.7143 | 0.7154 | 1.2882 | 2.7736 | 1.6654 | 23 |

Mean expanded: Accuracy 0.7300, Precision 0.7055, Recall 0.9129, F1 0.7825, ROC-AUC 0.8427, MAE (ns) 0.9879, MSE 1.8007, RMSE (ns) 1.2434

## no_edge_features

Edge channels are zeroed. Attention still uses node and timing state.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 1.0000 | 0.5863 | 0.3928 | 0.6267 | 5 |
| aes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2043 | 0.0538 | 0.2319 | 4 |
| ibex | 0.8571 | 1.0000 | 0.5000 | 0.6667 | 0.6000 | 1.3777 | 2.7525 | 1.6591 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 | 1.6269 | 3.8584 | 1.9643 | 6 |

Mean endpoint: Accuracy 0.7893, Precision 0.7292, Recall 0.8750, F1 0.7167, ROC-AUC 0.9000, MAE (ns) 0.9488, MSE 1.7644, RMSE (ns) 1.1205

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8833 | 0.8532 | 0.8268 | 0.9093 | 27 |
| aes | 0.9000 | 0.9091 | 0.9091 | 0.9091 | 0.9394 | 0.3695 | 0.2802 | 0.5293 | 20 |
| ibex | 0.7857 | 0.8000 | 0.8333 | 0.8163 | 0.8727 | 1.1392 | 2.6007 | 1.6127 | 42 |
| jpeg | 0.3913 | 0.2500 | 0.2000 | 0.2222 | 0.5538 | 1.2894 | 2.7139 | 1.6474 | 23 |

Mean expanded: Accuracy 0.6952, Precision 0.6528, Recall 0.7356, F1 0.6843, ROC-AUC 0.8123, MAE (ns) 0.9128, MSE 1.6054, RMSE (ns) 1.1747

## no_attention

Incoming drivers are averaged with equal weight.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 0.8333 | 0.7037 | 0.8078 | 0.8988 | 5 |
| aes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1859 | 0.0527 | 0.2296 | 4 |
| ibex | 0.8571 | 1.0000 | 0.5000 | 0.6667 | 0.9000 | 1.4589 | 3.4195 | 1.8492 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 | 1.6260 | 3.8515 | 1.9625 | 6 |

Mean endpoint: Accuracy 0.7893, Precision 0.7292, Recall 0.8750, F1 0.7167, ROC-AUC 0.9333, MAE (ns) 0.9936, MSE 2.0329, RMSE (ns) 1.2350

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8722 | 0.9482 | 1.0650 | 1.0320 | 27 |
| aes | 0.8500 | 0.9000 | 0.8182 | 0.8571 | 0.9192 | 0.3754 | 0.2858 | 0.5346 | 20 |
| ibex | 0.7143 | 0.7500 | 0.7500 | 0.7500 | 0.8333 | 1.1551 | 2.7836 | 1.6684 | 42 |
| jpeg | 0.6087 | 0.5385 | 0.7000 | 0.6087 | 0.5308 | 1.2885 | 2.7080 | 1.6456 | 23 |

Mean expanded: Accuracy 0.7192, Precision 0.7102, Recall 0.8170, F1 0.7513, ROC-AUC 0.7889, MAE (ns) 0.9418, MSE 1.7106, RMSE (ns) 1.2201

## classification_only

Only the violation loss is trained. The slack head is not supervised.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 1.0000 | 0.7207 | 0.5474 | 0.7398 | 5 |
| aes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9720 | 1.3670 | 1.1692 | 4 |
| ibex | 0.2857 | 0.2857 | 1.0000 | 0.4444 | 1.0000 | 1.6453 | 4.1630 | 2.0404 | 7 |
| jpeg | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.6788 | 4.0869 | 2.0216 | 6 |

Mean endpoint: Accuracy 0.7714, Precision 0.7381, Recall 1.0000, F1 0.8111, ROC-AUC 1.0000, MAE (ns) 1.2542, MSE 2.5411, RMSE (ns) 1.4928

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8889 | 0.9161 | 0.9142 | 0.9562 | 27 |
| aes | 0.9500 | 0.9167 | 1.0000 | 0.9565 | 0.9596 | 0.6196 | 0.8698 | 0.9326 | 20 |
| ibex | 0.6190 | 0.6000 | 1.0000 | 0.7500 | 0.8102 | 1.3556 | 3.2865 | 1.8129 | 42 |
| jpeg | 0.6957 | 0.6667 | 0.6000 | 0.6316 | 0.7846 | 1.3120 | 2.8513 | 1.6886 | 23 |

Mean expanded: Accuracy 0.7421, Precision 0.7089, Recall 0.9000, F1 0.7819, ROC-AUC 0.8608, MAE (ns) 1.0508, MSE 1.9805, RMSE (ns) 1.3476

## regression_only

Only the slack loss is trained. The violation head is not supervised.

### Endpoint

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.6000 | n/a | 0.0000 | n/a | 0.3333 | 0.8017 | 0.8944 | 0.9457 | 5 |
| aes | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.1873 | 0.0415 | 0.2036 | 4 |
| ibex | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1.4800 | 3.6284 | 1.9048 | 7 |
| jpeg | 0.8333 | n/a | 0.0000 | n/a | 0.0000 | 1.6517 | 3.9685 | 1.9921 | 6 |

Mean endpoint: Accuracy 0.3583, Precision 0.0000, Recall 0.0000, F1 n/a, ROC-AUC 0.0833, MAE (ns) 1.0302, MSE 2.1332, RMSE (ns) 1.2616

### Expanded

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.4444 | 0.5000 | 0.1333 | 0.2105 | 0.5167 | 0.8781 | 1.0923 | 1.0451 | 27 |
| aes | 0.0500 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.4194 | 0.2952 | 0.5433 | 20 |
| ibex | 0.3095 | 0.4074 | 0.4583 | 0.4314 | 0.2523 | 1.1642 | 2.8491 | 1.6879 | 42 |
| jpeg | 0.5652 | n/a | 0.0000 | n/a | 0.2077 | 1.2916 | 2.7659 | 1.6631 | 23 |

Mean expanded: Accuracy 0.3423, Precision 0.3025, Recall 0.1479, F1 0.3209, ROC-AUC 0.2442, MAE (ns) 0.9383, MSE 1.7506, RMSE (ns) 1.2349
