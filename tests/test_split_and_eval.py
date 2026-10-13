import numpy as np
import pandas as pd

from fraud_alert_budget.evaluate import (CostModel, budget_report, calibrate_threshold,
                                         select_alerts)
from fraud_alert_budget.split import random_split, temporal_split


def test_temporal_split_is_ordered_and_disjoint(small_df):
    s = temporal_split(small_df)
    t = small_df["timestamp"]
    assert t[s.train].max() < t[s.val].min()
    assert t[s.val].max() < t[s.test].min()
    assert not (s.train & s.val).any() and not (s.val & s.test).any()
    assert (s.train | s.val | s.test).all()


def test_gap_removes_tail_of_earlier_periods(small_df):
    s = temporal_split(small_df, gap_days=3)
    t = small_df["timestamp"]
    assert t[s.val].min() - t[s.train].max() >= pd.Timedelta(days=3)


def test_random_split_partitions(small_df):
    s = random_split(small_df, seed=1)
    assert ((s.train.astype(int) + s.val + s.test) == 1).all()


def _toy(n_days=5, per_day=40, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Series(pd.Timestamp("2025-01-01")
                   + pd.to_timedelta(np.sort(rng.uniform(0, n_days * 86400, n_days * per_day)), unit="s"))
    return ts, rng.random(len(ts))


def test_topk_respects_daily_budget():
    ts, s = _toy()
    a = select_alerts(ts, s, k_per_day=7, mode="topk")
    assert pd.Series(a).groupby(ts.dt.floor("D").to_numpy()).sum().eq(7).all()


def test_streaming_respects_budget_and_threshold():
    ts, s = _toy()
    a = select_alerts(ts, s, k_per_day=5, mode="streaming", threshold=0.5)
    per_day = pd.Series(a).groupby(ts.dt.floor("D").to_numpy()).sum()
    assert per_day.le(5).all()
    assert (s[a] >= 0.5).all()


def test_streaming_takes_earliest_candidates_first():
    ts = pd.Series(pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"]))
    s = np.array([0.6, 0.7, 0.99])
    a = select_alerts(ts, s, k_per_day=2, mode="streaming", threshold=0.5)
    assert a.tolist() == [True, True, False]   # the 0.99 arrived after the queue filled


def test_calibrated_threshold_hits_target_volume():
    s = np.random.default_rng(3).random(1000)
    thr = calibrate_threshold(s, n_val_days=10, k_per_day=12)
    assert (s >= thr).sum() == 120


def test_budget_report_arithmetic():
    ts = pd.Series(pd.to_datetime(["2025-01-01"] * 4))
    y = np.array([1, 0, 1, 0]); amt = np.array([100.0, 50.0, 30.0, 10.0])
    alert = np.array([True, True, False, False])
    r = budget_report(y, amt, alert, ts, k_per_day=2,
                      cost=CostModel(review_cost=4, fp_friction_cost=2))
    assert r["frauds_caught"] == 1 and r["false_positives"] == 1
    assert r["fp_per_fraud_caught"] == 1.0
    assert r["value_capture"] == 100 / 130
    assert r["net_benefit"] == 100 - 2 * 4 - 1 * 2
