# Leakage experiment

Honest baseline = temporal split + leakage-safe features. Inflation is relative to that baseline.

Random-split cells averaged over 3 seeds. Alert rate for precision/recall: top 1% of test transactions.

## Mean (std) by cell

|                            |   ('pr_auc', 'mean') |   ('pr_auc', 'std') |   ('roc_auc', 'mean') |   ('roc_auc', 'std') |   ('precision_at_rate', 'mean') |   ('precision_at_rate', 'std') |   ('recall_at_rate', 'mean') |   ('recall_at_rate', 'std') |
|:---------------------------|---------------------:|--------------------:|----------------------:|---------------------:|--------------------------------:|-------------------------------:|-----------------------------:|----------------------------:|
| ('random', 'past_only')    |               0.3603 |              0.0053 |                0.8577 |               0.0018 |                          0.6554 |                         0.0078 |                       0.1883 |                      0.0006 |
| ('random', 'with_leaky')   |               0.4915 |              0.005  |                0.9019 |               0.0023 |                          0.8307 |                         0.0044 |                       0.2387 |                      0.001  |
| ('temporal', 'past_only')  |               0.1531 |            nan      |                0.8065 |             nan      |                          0.2725 |                       nan      |                       0.08   |                    nan      |
| ('temporal', 'with_leaky') |               0.2172 |            nan      |                0.8361 |             nan      |                          0.3996 |                       nan      |                       0.1173 |                    nan      |


## Inflation vs honest baseline (%)

|                                                |   pr_auc |   roc_auc |   precision_at_rate |   recall_at_rate |
|:-----------------------------------------------|---------:|----------:|--------------------:|-----------------:|
| random split only (same leakage-safe features) |    135.4 |       6.3 |               140.5 |            135.3 |
| leaky features only (temporal split)           |     41.9 |       3.7 |                46.7 |             46.7 |
| both mistakes (random split + leaky features)  |    221.1 |      11.8 |               204.9 |            198.3 |

