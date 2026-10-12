# SlackGNN evaluation

Leave-one-design-out mean. SlackGNN numbers are from the multitask model when that task was trained.
Classification class 1 is a timing violation. Slack error is in nanoseconds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcn | 0.3649 | 0.3083 | 0.8750 | 0.4435 | 0.6708 | 1.3702 | 1.6721 |
| graphsage | 0.4839 | 0.3375 | 0.8750 | 0.4810 | 0.6875 | 1.2860 | 1.5759 |
| gin | 0.4006 | 0.2841 | 0.7500 | 0.4339 | 0.6833 | 1.2245 | 1.4986 |
| slackgnn | 0.9143 | 0.9167 | 0.8750 | 0.8667 | 0.9000 | 0.7530 | 0.9748 |

Multitask SlackGNN is above GCN, GraphSAGE, and GIN on F1 (0.8667) and ROC-AUC (0.9000), and its slack MAE (0.7530 ns) is below all three.

## SlackGNN multitask folds

| Held-out design | Accuracy | F1 | ROC-AUC | MAE | RMSE | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | 1.0000 | 1.0000 | 1.0000 | 0.4013 | 0.5224 | 4 |
| gcd | 0.8000 | 0.8000 | 1.0000 | 0.5118 | 0.6794 | 5 |
| ibex | 0.8571 | 0.6667 | 0.6000 | 1.2864 | 1.5622 | 7 |
| jpeg | 1.0000 | 1.0000 | 1.0000 | 0.8127 | 1.1353 | 6 |
