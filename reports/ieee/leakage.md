# Leakage experiment

Honest baseline = temporal split + leakage-safe features. Inflation is relative to that baseline.

Random-split cells averaged over 1 seeds. Alert rate for precision/recall: top 1% of test transactions.

## Mean (std) by cell

|                            |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('roc_auc', 'mean') |   ('roc_auc', 'std') |   ('precision_at_rate', 'mean') |   ('precision_at_rate', 'std') |   ('recall_at_rate', 'mean') |   ('recall_at_rate', 'std') |
|:---------------------------|---------------------:|--------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-----------------------------:|----------------------------:|
| ('random', 'past_only')    |               0.3547 |                 nan |                0.856  |                  nan |                          0.6486 |                            nan |                       0.1878 |                         nan |
| ('random', 'with_leaky')   |               0.4861 |                 nan |                0.8993 |                  nan |                          0.8281 |                            nan |                       0.2398 |                         nan |
| ('temporal', 'past_only')  |               0.1531 |                 nan |                0.8065 |                  nan |                          0.2725 |                            nan |                       0.08   |                         nan |
| ('temporal', 'with_leaky') |               0.2172 |                 nan |                0.8361 |                  nan |                          0.3996 |                            nan |                       0.1173 |                         nan |


## Inflation vs honest baseline (%)

|                                                |   pr_auc |   roc_auc |   precision_at_rate |   recall_at_rate |
|:-----------------------------------------------|---------:|----------:|--------------------:|-----------------:|
| random split only (same leakage-safe features) |    131.7 |       6.1 |               138   |            134.8 |
| leaky features only (temporal split)           |     41.9 |       3.7 |                46.7 |             46.7 |
| both mistakes (random split + leaky features)  |    217.6 |      11.5 |               203.9 |            199.7 |

