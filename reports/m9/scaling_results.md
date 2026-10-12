# Scaling

Held-out design: `jpeg`. jpeg is present and has both endpoint classes

Training designs are added from the smallest graph to the largest.

| Training designs | Count | F1 | ROC-AUC | MAE (ns) | Expanded F1 | Expanded MAE (ns) | Endpoint support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd, aes | 2 | 1.0000 | 1.0000 | 0.9049 | 0.7500 | 0.9654 | 6 |
| gcd, aes, ibex | 3 | 0.4000 | 1.0000 | 1.6308 | 0.2222 | 1.2903 | 6 |

Missing designs:

- `riscv32i`: design directory is missing
- `microwatt`: design directory is missing
- `chameleon`: design directory is missing

Runtime: 19.6 s.
