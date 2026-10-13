# Results (ieee data)

Test period: 2018-04-26 to 2018-06-01, 104,637 transactions, 3562 frauds.

Cost model: \$4 per review, \$2 friction per false positive.

## Headline: 500 alerts/day

|                           |   alerts_per_day |   queue_utilisation |   frauds_caught |   frauds_total |   precision_at_budget |   recall_at_budget |   value_capture |   fp_per_fraud_caught |   fraud_amount_caught (\$) |   net_benefit (\$) |
|:--------------------------|-----------------:|--------------------:|----------------:|---------------:|----------------------:|-------------------:|----------------:|----------------------:|---------------------------:|-------------------:|
| ('lightgbm', 'streaming') |          460.838 |               0.922 |            2075 |           3562 |                 0.122 |              0.583 |           0.534 |                 7.217 |                     284729 |             186573 |
| ('lightgbm', 'topk')      |          500     |               1     |            2175 |           3562 |                 0.118 |              0.611 |           0.557 |                 7.506 |                     297185 |             190535 |


## Ranking metrics on test (reference only)

|          |   roc_auc |   pr_auc |
|:---------|----------:|---------:|
| lightgbm |    0.8065 |   0.1531 |


## Budget sweep (streaming mode)

|   k_per_day |   net_benefit (\$), lightgbm |   fp_per_fraud_caught, lightgbm |   recall_at_budget, lightgbm |
|------------:|-----------------------------:|--------------------------------:|-----------------------------:|
|          25 |                      40530.5 |                            2.31 |                         0.06 |
|          50 |                      57918   |                            2.99 |                         0.1  |
|         100 |                      87267.8 |                            3.57 |                         0.18 |
|         200 |                     122964   |                            4.58 |                         0.32 |
|         500 |                     186573   |                            7.22 |                         0.58 |
|        1000 |                     190162   |                           11.92 |                         0.76 |


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
