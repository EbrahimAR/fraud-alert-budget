# Fraud Detection under an Alert Budget

A transaction fraud classifier evaluated the way a fraud operations team experiences it: a fixed number of analysts can review **K alerts per day**, and what matters is how much fraud money those K reviews stop, how many innocent customers get bothered along the way, and whether the whole thing pays for itself.

Built with Python, LightGBM, pandas and scikit-learn.

## Why this project exists

Most fraud-detection notebooks share three problems:

1. **Random train/test splits.** The model gets tested on transactions that happened *before* some of the ones it learned from, and frauds from the same compromised card land on both sides of the split. Scores look excellent and then fall apart in production.
2. **Features that peek at the future.** A common pattern is `df.groupby('card_id')['amount'].transform('mean')` run on the full dataset before splitting. That average includes transactions that hadn't happened yet.
3. **Metrics nobody in operations uses.** ROC-AUC says little to a team that can only review 25 alerts a day.

This repo fixes each one and measures how much the first two mistakes inflate results.

## Method

**Temporal split.** Train on the first 60% of the timeline, early-stop on the next 20%, test on the final 20%. Every test transaction is later than every training transaction. `--gap-days` can drop the tail of each earlier period to simulate label maturity, since chargebacks take weeks to arrive.

**Leakage-safe features** (`features.py`). All 21 features use only information available when the transaction arrives:

- transaction-level fields: amount, hour, day of week, online flag, merchant category, card BIN
- the card's history *strictly before* this transaction: count, mean and standard deviation of past amounts, this amount relative to the card's normal spend, seconds since the last transaction, share of past online use, first-ever use of this merchant category
- velocity over the windows `[t − 1h, t)`, `[t − 24h, t)` and `[t − 7d, t)`: transaction count, total amount, and number of sub-£5 transactions in the last hour (a card-testing signal)

The windows are computed with `searchsorted` over a composite (card, time) key. That makes them exact and vectorised (about 1 second for 400k rows), and makes the "strictly before" boundary explicit in the code. Labels are never used to build features.

**Model.** LightGBM with early stopping on validation **PR-AUC**. With roughly 0.4% fraud, ROC-AUC is dominated by easy negatives, while PR-AUC tracks the top of the ranking, which is the only part an analyst ever sees.

**Evaluation harness** (`evaluate.py`). There are two alerting modes:

- **streaming** (realistic): a score threshold is calibrated on the validation month to produce about K alerts per day. On the test period, transactions arrive in time order, and each one above threshold is alerted until that day's K slots are full.
- **topk** (hindsight upper bound): the K highest-scoring transactions each day. This isn't achievable in real time but is useful as a ceiling.

Reported per budget: precision and recall at budget, £ caught, value capture, false positives per fraud caught, queue utilisation, and **net £ benefit**:

```
net benefit = fraud £ caught × prevention rate
            − alerts × review cost
            − false positives × customer friction cost
```

**Baseline.** A hand-written rules score (`rules.py`) covering card testing, velocity bursts, unusually large amounts, new merchant category, night-time and online. A model has to beat this, not just a random ranking.

## Results (synthetic data, seed 42)

> These numbers come from the bundled synthetic generator, whose properties I designed (see `synth.py`). They demonstrate the methodology. Run on IEEE-CIS for numbers that mean something.

Test period: May 2025, 86,823 transactions, 320 frauds. Assumed costs: £4 per review, £2 friction per false positive.

**At 25 alerts per day (streaming):**

| | LightGBM | Rules |
|---|---|---|
| Frauds caught | 187 / 320 | 90 / 320 |
| Value capture (£ of fraud stopped) | 68% | 55% |
| False positives per fraud caught | 1.85 | 5.63 |
| Net benefit over the month | **£24.1k** | £18.3k |
| PR-AUC (reference) | 0.616 | 0.280 |

**Leakage experiment** (`scripts/leakage_experiment.py`). This is a 2×2 design varying split type and feature type, holding everything else fixed. Random-split cells are averaged over 3 seeds. Inflation is measured against the honest cell (temporal split, leakage-safe features):

| Mistake | PR-AUC inflation | Recall at top 1% | ROC-AUC |
|---|---|---|---|
| Random split only (same safe features) | **+56.5%** | +48.3% | +2.9% |
| Leaky features only (temporal split) | +5.9% | +3.8% | +0.3% |
| Both mistakes | +59.4% | +50.5% | +2.9% |

The precision and recall comparisons use the top 1% of each test set, so test sets of different sizes are compared on equal terms.

### What the results show

- **ROC-AUC hides the leakage almost completely.** It moves 2.9% while PR-AUC moves 56%. A team watching ROC-AUC would not notice the problem.
- **The split matters more than the features here.** Random splitting lets sibling frauds from the same burst sit in both train and test. It also lets the model train on the fraud pattern that only appears in the later months.
- **Drift is visible.** The model catches 93% of the long-standing card-testing pattern but only 37% of the account-takeover pattern that emerges late in the timeline.
- **A static threshold can waste capacity.** At 5 alerts per day, the streaming model *lost* to the rules baseline. Its April-calibrated threshold was too strict for May's shifted scores, so it filled only 17% of the queue: every alert was fraud, but most of the slots sat empty. Recalibrating the threshold on a rolling recent window would address this (see Limitations).
- **More alerts isn't better.** Net benefit peaks at around 25–50 alerts per day. Beyond that, review costs exceed the extra fraud caught (see `reports/budget_curves.png`).

![budget curves](reports/budget_curves.png)

## Running it

```bash
pip install -e ".[dev]"
pytest -q                                   # 15 tests
python scripts/run_pipeline.py              # ~10 s on a laptop
python scripts/leakage_experiment.py        # 8 model fits; resumable (cached per cell)
```

Outputs go to `reports/`: `results.md`, `leakage.md`, `budget_sweep.csv`, `feature_importance.csv` and `budget_curves.png`.

**On real data (IEEE-CIS).** Download `train_transaction.csv` from the [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data) into `data/ieee-fraud-detection/`, then run:

```bash
python scripts/run_pipeline.py --source ieee --data-dir data/ieee-fraud-detection --out reports/ieee
python scripts/leakage_experiment.py --source ieee --data-dir data/ieee-fraud-detection --out reports/ieee
```

IEEE-CIS has no card identifier, so the loader builds the community-standard proxy `card1 + addr1 + (day − D1)`. Amounts are USD but are reported as £ for consistency. Note that in this dataset, transactions on an account after a confirmed fraud are often also labelled fraud, which makes card-history features unusually strong.

## Tests

The most important tests prove the leakage-safety claim directly:

- **Truncating** the dataset at a cutoff leaves every earlier row's features **bit-for-bit identical**.
- **Perturbing** future rows (amounts, categories and labels) changes no earlier row's features.
- A transaction never sees itself or same-timestamp siblings, and window boundaries are exact.
- Velocity features match a brute-force recomputation.
- The deliberately leaky features really do leak, so the experiment's leaky arm is genuine.

The truncation test caught a real bug during development. Window sums originally used one global cumulative sum across all cards, so floating-point rounding from *other* cards' transactions slightly changed a card's standard-deviation feature. Sums now restart per card, and variance is computed on shifted values to avoid cancellation error.

The evaluation tests check that daily budgets are never exceeded, that streaming mode fills slots in arrival order, that threshold calibration hits its target volume, and that the net-benefit arithmetic is correct.

## Assumptions and limitations

- **Cost model.** £4 per review assumes about 10 minutes of analyst time at roughly £24/h fully loaded. £2 friction per false positive is a guess. Both are CLI flags, and conclusions should be checked across a range of values.
- **No card blocking.** Catching the first fraud in a burst would, in reality, block the card and prevent the rest of the burst. Each transaction is scored independently here, which understates the value of early detection.
- **Static threshold.** It is calibrated once on validation. Production systems recalibrate on a rolling recent window, and the 5-per-day result shows why.
- **Single temporal fold.** The honest estimate comes from one test month. Rolling-origin backtesting over several months would give a variance estimate for the honest cell, as the random cells already have.
- **Synthetic data.** It encodes my own assumptions about fraud behaviour. The headline numbers should come from IEEE-CIS.

## Layout

```
src/fraud_alert_budget/
  data.py       schema, validation, synthetic + IEEE-CIS loaders
  synth.py      synthetic generator (bursts, two fraud patterns, drift)
  features.py   leakage-safe features + deliberately leaky ones for the experiment
  split.py      temporal (with optional label gap) and random splits
  model.py      LightGBM with PR-AUC early stopping
  rules.py      rules baseline
  evaluate.py   alert-budget harness and business metrics
  pipeline.py   shared glue
scripts/        run_pipeline.py, leakage_experiment.py
tests/          15 tests
reports/        generated outputs
```
