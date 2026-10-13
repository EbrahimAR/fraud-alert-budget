"""Evaluation under a fixed daily alert budget.

A fraud operations team does not consume a ROC curve. It has a fixed number of
analysts who can review K alerts per day. The questions that matter are:

* Of the K alerts we review, how many are fraud?          (precision at budget)
* How much fraud money do we stop?                         (amount caught, value capture)
* How many innocent customers do we bother per catch?      (false positives per fraud caught)
* After paying for reviews and customer friction, are we
  better off than doing nothing?                           (net benefit)

Two alerting modes
------------------
streaming (default, realistic)
    A score threshold is calibrated on the VALIDATION period so that it would
    have produced ~K alerts/day there. On the test period, transactions arrive
    in time order; each one above threshold becomes an alert until that day's
    K slots are used up. Later transactions that day go unreviewed, even if
    they score higher. If the score distribution drifts, the queue may be
    under- or over-filled, which is reported as utilisation.

topk (hindsight upper bound)
    The K highest-scoring transactions of each day are alerted. Not achievable
    in real time (it needs the whole day's scores), but useful as a ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class CostModel:
    """Money assumptions, in currency units (the same currency as the
    transaction amounts). All configurable; defaults are stated assumptions,
    not industry facts.

    review_cost       analyst cost per alert (~10 min at a loaded rate of
                      ~24 currency units per hour)
    fp_friction_cost  cost of a false alarm to the customer relationship
                      (blocked card, call to the customer, possible churn)
    prevention_rate   share of a caught fraud's amount actually saved
    """
    review_cost: float = 4.0
    fp_friction_cost: float = 2.0
    prevention_rate: float = 1.0


def calibrate_threshold(val_scores: np.ndarray, n_val_days: int, k_per_day: int) -> float:
    """Score threshold that would have produced ~k_per_day alerts on validation."""
    n_target = int(round(k_per_day * n_val_days))
    if n_target <= 0:
        return np.inf
    if n_target >= len(val_scores):
        return -np.inf
    return float(np.sort(val_scores)[::-1][n_target - 1])


def select_alerts(timestamps: pd.Series, scores: np.ndarray, k_per_day: int,
                  mode: str = "streaming", threshold: float | None = None,
                  seed: int = 0) -> np.ndarray:
    """Boolean mask of which transactions get alerted, never more than
    k_per_day on any calendar day."""
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    scores = np.asarray(scores, float)
    day = ts.dt.floor("D").to_numpy().astype("datetime64[D]").astype(np.int64)
    n = len(scores)
    alert = np.zeros(n, bool)

    if mode == "topk":
        tiebreak = np.random.default_rng(seed).random(n)
        order = np.lexsort((tiebreak, -scores, day))
        d_sorted = day[order]
        first_of_day = np.r_[True, d_sorted[1:] != d_sorted[:-1]]
        pos = np.arange(n) - np.maximum.accumulate(np.where(first_of_day, np.arange(n), 0))
        alert[order[pos < k_per_day]] = True
        return alert

    if mode == "streaming":
        if threshold is None:
            raise ValueError("streaming mode needs a threshold calibrated on validation")
        order = np.lexsort((np.arange(n), ts.to_numpy().astype(np.int64), day))
        cand = scores[order] >= threshold
        d_sorted = day[order]
        frame = pd.DataFrame({"d": d_sorted, "c": cand.astype(int)})
        used = frame.groupby("d")["c"].cumsum().to_numpy()
        alert[order[cand & (used <= k_per_day)]] = True
        return alert

    raise ValueError(f"unknown mode {mode!r}")


def budget_report(y: np.ndarray, amount: np.ndarray, alert: np.ndarray,
                  timestamps: pd.Series, k_per_day: int,
                  cost: CostModel = CostModel()) -> dict:
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, float)
    alert = np.asarray(alert, bool)
    n_days = int(pd.Series(pd.to_datetime(timestamps)).dt.floor("D").nunique())

    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    n_alerts = tp + fp
    fraud_total = float(amount[y].sum())
    caught = float(amount[alert & y].sum())
    saved = cost.prevention_rate * caught
    review = cost.review_cost * n_alerts
    friction = cost.fp_friction_cost * fp

    return {
        "k_per_day": k_per_day,
        "days": n_days,
        "alerts": n_alerts,
        "alerts_per_day": n_alerts / max(n_days, 1),
        "queue_utilisation": n_alerts / max(k_per_day * n_days, 1),
        "frauds_total": int(y.sum()),
        "frauds_caught": tp,
        "false_positives": fp,
        "precision_at_budget": tp / n_alerts if n_alerts else 0.0,
        "recall_at_budget": tp / y.sum() if y.sum() else 0.0,
        "fraud_amount_total": fraud_total,
        "fraud_amount_caught": caught,
        "value_capture": caught / fraud_total if fraud_total else 0.0,
        "fp_per_fraud_caught": fp / tp if tp else float("inf"),
        "review_cost": review,
        "friction_cost": friction,
        "net_benefit": saved - review - friction,
        **{f"cost_{k}": v for k, v in asdict(cost).items()},
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Standard ML metrics, reported for reference only."""
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def recall_at_alert_rate(y: np.ndarray, scores: np.ndarray, rate: float = 0.01) -> dict:
    """Precision/recall when alerting the top `rate` share of transactions.
    Rate-based (not per-day) so that test sets of different sizes, e.g. a
    random split vs a temporal one, can be compared fairly."""
    y = np.asarray(y).astype(bool)
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-np.asarray(scores, float), kind="mergesort")[:k]
    tp = int(y[top].sum())
    return {"alert_rate": rate, "precision_at_rate": tp / k,
            "recall_at_rate": tp / y.sum() if y.sum() else 0.0}
