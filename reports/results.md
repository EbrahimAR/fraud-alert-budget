# Results (synthetic data)

Test period: 2025-05-01 to 2025-05-30, 86,823 transactions, 320 frauds.

Cost model: £4 per review, £2 friction per false positive.

## Headline: 25 alerts/day

|                           |   alerts_per_day |   queue_utilisation |   frauds_caught |   frauds_total |   precision_at_budget |   recall_at_budget |   value_capture |   fp_per_fraud_caught |   fraud_amount_caught (£) |   net_benefit (£) |
|:--------------------------|-----------------:|--------------------:|----------------:|---------------:|----------------------:|-------------------:|----------------:|----------------------:|--------------------------:|------------------:|
| ('lightgbm', 'streaming') |           17.767 |               0.711 |             187 |            320 |                 0.351 |              0.584 |           0.68  |                 1.85  |                   26916.3 |           24092.3 |
| ('lightgbm', 'topk')      |           25     |               1     |             197 |            320 |                 0.263 |              0.616 |           0.761 |                 2.807 |                   30090.1 |           25984.1 |
| ('rules', 'streaming')    |           19.9   |               0.796 |              90 |            320 |                 0.151 |              0.281 |           0.548 |                 5.633 |                   21661.7 |           18259.7 |
| ('rules', 'topk')         |           25     |               1     |              95 |            320 |                 0.127 |              0.297 |           0.58  |                 6.895 |                   22931.5 |           18621.5 |


## Ranking metrics on test (reference only)

|          |   roc_auc |   pr_auc |
|:---------|----------:|---------:|
| lightgbm |    0.971  |   0.6161 |
| rules    |    0.9312 |   0.2805 |


## Budget sweep (streaming mode)

|   k_per_day |   net_benefit (£), lightgbm |   net_benefit (£), rules |   fp_per_fraud_caught, lightgbm |   fp_per_fraud_caught, rules |   recall_at_budget, lightgbm |   recall_at_budget, rules |
|------------:|----------------------------:|-------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|--------------------------:|
|           5 |                     5685.77 |                 10952.8  |                            0    |                         0.18 |                         0.08 |                      0.14 |
|          10 |                    16990    |                 15946.9  |                            0    |                         1.18 |                         0.33 |                      0.21 |
|          25 |                    24092.3  |                 18259.7  |                            1.85 |                         5.63 |                         0.58 |                      0.28 |
|          50 |                    25048.1  |                 20036    |                            4.65 |                         7.46 |                         0.68 |                      0.44 |
|         100 |                    19643.8  |                 16576.6  |                            9.78 |                        12.4  |                         0.76 |                      0.58 |
|         200 |                     5009.62 |                  2911.17 |                           19.2  |                        22    |                         0.82 |                      0.74 |


## Top features (gain share)

| feature           |     gain |   share |
|:------------------|---------:|--------:|
| n_small_1h        | 42839.1  |   0.23  |
| hour              | 33017.5  |   0.177 |
| log_amount        | 22804.4  |   0.122 |
| sum_1h            | 19197.9  |   0.103 |
| category          | 18811.6  |   0.101 |
| cnt_1h            | 15592.9  |   0.084 |
| secs_since_last   |  9748.75 |   0.052 |
| is_online         |  8449.05 |   0.045 |
| amt_to_prior_mean |  7018.76 |   0.038 |
| sum_24h           |  2692.13 |   0.014 |


## Recall by fraud pattern at headline budget (synthetic only)

| fraud_mo   |   frauds |   caught |   recall |
|:-----------|---------:|---------:|---------:|
| A          |      124 |      115 |    0.927 |
| B          |      196 |       72 |    0.367 |


![budget curves](budget_curves.png)
