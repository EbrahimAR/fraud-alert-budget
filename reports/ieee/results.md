# Results (ieee data)

Test period: 2018-04-26 to 2018-06-01, 104,637 transactions, 3562 frauds.

Cost model: £4 per review, £2 friction per false positive.

## Headline: 25 alerts/day

|                           |   alerts_per_day |   queue_utilisation |   frauds_caught |   frauds_total |   precision_at_budget |   recall_at_budget |   value_capture |   fp_per_fraud_caught |   fraud_gbp_caught |   net_benefit_gbp |
|:--------------------------|-----------------:|--------------------:|----------------:|---------------:|----------------------:|-------------------:|----------------:|----------------------:|-------------------:|------------------:|
| ('lightgbm', 'streaming') |           17.514 |               0.701 |             196 |           3562 |                 0.302 |              0.055 |           0.083 |                 2.306 |           44026.5  |         40530.5   |
| ('lightgbm', 'topk')      |           25     |               1     |             244 |           3562 |                 0.264 |              0.069 |           0.092 |                 2.791 |           48960.7  |         43898.7   |
| ('rules', 'streaming')    |           19.216 |               0.769 |              35 |           3562 |                 0.049 |              0.01  |           0.01  |                19.314 |            5148.67 |           952.671 |
| ('rules', 'topk')         |           25     |               1     |              49 |           3562 |                 0.053 |              0.014 |           0.044 |                17.878 |           23447.6  |         17995.6   |


## Ranking metrics on test (reference only)

|          |   roc_auc |   pr_auc |
|:---------|----------:|---------:|
| lightgbm |    0.8065 |   0.1531 |
| rules    |    0.4987 |   0.0363 |


## Budget sweep (streaming mode)

|   k_per_day |   ('net_benefit_gbp', 'lightgbm') |   ('net_benefit_gbp', 'rules') |   ('fp_per_fraud_caught', 'lightgbm') |   ('fp_per_fraud_caught', 'rules') |   ('recall_at_budget', 'lightgbm') |   ('recall_at_budget', 'rules') |
|------------:|----------------------------------:|-------------------------------:|--------------------------------------:|-----------------------------------:|-----------------------------------:|--------------------------------:|
|           5 |                           11654.7 |                        5737.56 |                                  1.44 |                              17.62 |                               0.01 |                            0    |
|          10 |                           19588.3 |                        5428.67 |                                  1.86 |                              20.07 |                               0.02 |                            0    |
|          25 |                           40530.5 |                         952.67 |                                  2.31 |                              19.31 |                               0.06 |                            0.01 |
|          50 |                           57918   |                       30948.7  |                                  2.99 |                              16.98 |                               0.1  |                            0.02 |
|         100 |                           87267.8 |                       43036.8  |                                  3.57 |                              16.9  |                               0.18 |                            0.05 |
|         200 |                          122964   |                       33303.4  |                                  4.58 |                              22.59 |                               0.32 |                            0.08 |


## Top features (gain share)

| feature             |     gain |   share |
|:--------------------|---------:|--------:|
| category            | 105490   |   0.213 |
| card_bin            |  89018   |   0.18  |
| log_amount          |  66399.3 |   0.134 |
| secs_since_last     |  36314.4 |   0.073 |
| sum_7d              |  31947.9 |   0.065 |
| card_mean_amt_prior |  30315.8 |   0.061 |
| card_std_amt_prior  |  27527.3 |   0.056 |
| hour                |  23393.6 |   0.047 |
| card_n_prior        |  15253   |   0.031 |
| amt_to_prior_mean   |  14093   |   0.028 |


![budget curves](budget_curves.png)
