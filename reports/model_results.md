# Baseline timing-violation results

Class 0 is a labeled endpoint whose slack meets timing. Class 1 is a timing violation.
Leave-one-design-out trains on every other design and tests the held-out design.

## Leave-one-design-out mean

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| gcn | 0.3649 | 0.3083 | 0.8750 | 0.4435 | 0.6708 |
| graphsage | 0.4839 | 0.3375 | 0.8750 | 0.4810 | 0.6875 |
| gin | 0.4006 | 0.2841 | 0.7500 | 0.4339 | 0.6833 |

A mean omits a fold when that metric is undefined. Precision and F1 are undefined when a model predicts no timing violations.

## Leave-one-design-out folds

### gcn

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 0.7500 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 4 |
| gcd | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.5833 | 5 |
| ibex | 0.1429 | 0.1667 | 0.5000 | 0.2500 | 0.3000 | 7 |
| jpeg | 0.1667 | 0.1667 | 1.0000 | 0.2857 | 0.8000 | 6 |

### graphsage

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 0.7500 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 4 |
| gcd | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.5000 | 5 |
| ibex | 0.2857 | 0.2000 | 0.5000 | 0.2857 | 0.3500 | 7 |
| jpeg | 0.5000 | 0.2500 | 1.0000 | 0.4000 | 0.9000 | 6 |

### gin

| Held-out design | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 0.7500 | n/a | 0.0000 | n/a | 1.0000 | 4 |
| gcd | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.5833 | 5 |
| ibex | 0.2857 | 0.2857 | 1.0000 | 0.4444 | 0.3500 | 7 |
| jpeg | 0.1667 | 0.1667 | 1.0000 | 0.2857 | 0.8000 | 6 |

## Random labeled-endpoint split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gcn | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0.0000 | 3 |
| graphsage | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0.0000 | 3 |
| gin | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0.0000 | 3 |
