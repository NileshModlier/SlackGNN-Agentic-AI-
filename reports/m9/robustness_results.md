# Robustness

Full multitask leave-one-design-out at seeds 0, 1, and 2. The spread is the sample standard deviation of the three leave-one-design-out means.

| Seed | F1 | ROC-AUC | MAE (ns) |
| --- | ---: | ---: | ---: |
| 0 | 0.6750 | 0.9500 | 0.9890 |
| 1 | 0.7778 | 0.9500 | 0.9308 |
| 2 | 0.8222 | 0.9250 | 0.8409 |

Endpoint F1 mean 0.7583, spread 0.0755. ROC-AUC mean 0.9417, spread 0.0144. MAE mean 0.9203 ns, spread 0.0746 ns.

## Per-fold support

| Held-out | Endpoint support | Expanded support | F1 mean | F1 spread | ROC-AUC mean | ROC-AUC spread | MAE mean (ns) | MAE spread (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 5 | 27 | 0.7556 | 0.0770 | 1.0000 | 0.0000 | 0.6492 | 0.0878 |
| aes | 4 | 20 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.3655 | 0.1263 |
| ibex | 7 | 42 | 0.5370 | 0.1156 | 0.7667 | 0.0577 | 1.4547 | 0.1172 |
| jpeg | 6 | 23 | 0.7000 | 0.4243 | 1.0000 | 0.0000 | 1.2116 | 0.3638 |

Missing designs:

- `riscv32i`: design directory is missing
- `microwatt`: design directory is missing
- `chameleon`: design directory is missing

Runtime: 147.9 s.
