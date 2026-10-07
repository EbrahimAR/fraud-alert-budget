# Leakage experiment

Honest baseline = temporal split + leakage-safe features. Inflation is relative to that baseline.

Random-split cells averaged over 3 seeds. Alert rate for precision/recall: top 1% of test transactions.

## Mean (std) by cell

|                            |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('roc_auc', 'mean') |   ('roc_auc', 'std') |   ('precision_at_rate', 'mean') |   ('precision_at_rate', 'std') |   ('recall_at_rate', 'mean') |   ('recall_at_rate', 'std') |
|:---------------------------|---------------------:|--------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-----------------------------:|----------------------------:|
| ('random', 'past_only')    |               0.9639 |              0.0055 |                0.9991 |               0.0002 |                          0.4262 |                         0.0203 |                       0.9734 |                      0.0067 |
| ('random', 'with_leaky')   |               0.982  |              0.0038 |                0.9994 |               0.0004 |                          0.4324 |                         0.0181 |                       0.9877 |                      0.0041 |
| ('temporal', 'past_only')  |               0.6161 |            nan      |                0.971  |             nan      |                          0.2419 |                       nan      |                       0.6562 |                    nan      |
| ('temporal', 'with_leaky') |               0.6522 |            nan      |                0.9738 |             nan      |                          0.2512 |                       nan      |                       0.6812 |                    nan      |


## Inflation vs honest baseline (%)

|                                                |   pr_auc |   roc_auc |   precision_at_rate |   recall_at_rate |
|:-----------------------------------------------|---------:|----------:|--------------------:|-----------------:|
| random split only (same leakage-safe features) |     56.5 |       2.9 |                76.2 |             48.3 |
| leaky features only (temporal split)           |      5.9 |       0.3 |                 3.8 |              3.8 |
| both mistakes (random split + leaky features)  |     59.4 |       2.9 |                78.7 |             50.5 |

