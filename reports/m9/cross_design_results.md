# Cross-design benchmark

Leave-one-design-out. Training loss uses expanded labels. The endpoint table scores published endpoint slack. The expanded table scores path-propagated slack and is not an independent STA measurement.

Designs: gcd, aes, ibex, jpeg.

Missing designs:

- `riscv32i`: design directory is missing
- `microwatt`: design directory is missing
- `chameleon`: design directory is missing

Runtime: 72.2 s. Message passing runs on each full netlist. The loss uses only supervised nodes. Hidden width is 32, the epoch cap is 20, and early stopping patience is 6. These limits keep the CPU study finite on the largest graph.

## Endpoint labels

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.8000 | 0.6667 | 1.0000 | 0.8000 | 1.0000 | 0.6342 | 0.4813 | 0.6938 | 5 |
| aes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2298 | 0.0785 | 0.2801 | 4 |
| ibex | 0.7143 | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 1.4612 | 3.3940 | 1.8423 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 1.0000 | 1.6308 | 3.8748 | 1.9684 | 6 |

Mean endpoint: Accuracy 0.7536, Precision 0.6042, Recall 0.8750, F1 0.6750, ROC-AUC 0.9500, MAE (ns) 0.9890, MSE 1.9571, RMSE (ns) 1.1962

## Expanded labels

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE (ns) | MSE | RMSE (ns) | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 0.7037 | 0.6522 | 1.0000 | 0.7895 | 0.8722 | 0.8734 | 0.8475 | 0.9206 | 27 |
| aes | 0.9000 | 0.9091 | 0.9091 | 0.9091 | 0.9596 | 0.3555 | 0.2453 | 0.4952 | 20 |
| ibex | 0.7381 | 0.7600 | 0.7917 | 0.7755 | 0.8611 | 1.1529 | 2.7579 | 1.6607 | 42 |
| jpeg | 0.3913 | 0.2500 | 0.2000 | 0.2222 | 0.5538 | 1.2903 | 2.7236 | 1.6503 | 23 |

Mean expanded: Accuracy 0.6833, Precision 0.6428, Recall 0.7252, F1 0.6741, ROC-AUC 0.8117, MAE (ns) 0.9180, MSE 1.6436, RMSE (ns) 1.1817
