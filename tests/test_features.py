"""The core guarantee: features for a transaction depend only on the past."""
import numpy as np
import pandas as pd
import pytest

from fraud_alert_budget.features import PAST_FEATURES, add_leaky_features, add_past_features


def _feat(df, cats):
    return add_past_features(df, cats).set_index("txn_id")[PAST_FEATURES]


def _assert_equal(a, b):
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_exact=True)


def test_truncating_the_future_changes_nothing(small_df, categories):
    full = _feat(small_df, categories)
    cutoff = small_df["timestamp"].quantile(0.5)
    past = small_df[small_df["timestamp"] <= cutoff]
    _assert_equal(full.loc[past["txn_id"]], _feat(past, categories))


def test_perturbing_the_future_changes_nothing(small_df, categories):
    cutoff = small_df["timestamp"].quantile(0.5)
    future = small_df["timestamp"] > cutoff
    mutated = small_df.copy()
    rng = np.random.default_rng(0)
    mutated.loc[future, "amount"] = rng.uniform(1, 5000, future.sum()).round(2)
    mutated.loc[future, "category"] = "crypto"
    mutated.loc[future, "is_fraud"] = 1 - mutated.loc[future, "is_fraud"]   # labels too
    ids = small_df.loc[~future, "txn_id"]
    _assert_equal(_feat(small_df, categories).loc[ids], _feat(mutated, categories).loc[ids])


def test_current_transaction_excluded_from_its_own_history(categories):
    ts = pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:30:00", "2025-01-01 10:30:00"])
    df = pd.DataFrame({"txn_id": [0, 1, 2], "timestamp": ts, "card_id": [1, 1, 1],
                       "card_bin": 0, "amount": [10.0, 20.0, 40.0], "category": "grocery",
                       "is_online": 0, "is_fraud": 0})
    f = add_past_features(df, categories).set_index("txn_id")
    assert f.loc[0, "cnt_1h"] == 0 and f.loc[0, "card_n_prior"] == 0
    # rows 1 and 2 share a timestamp: neither may see the other, both see row 0
    for i in (1, 2):
        assert f.loc[i, "cnt_1h"] == 1
        assert f.loc[i, "sum_1h"] == 10.0
        assert f.loc[i, "secs_since_last"] == 1800
    assert f.loc[0, "is_new_category"] == 1 and f.loc[1, "is_new_category"] == 0


def test_window_boundaries(categories):
    ts = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:59:59", "2025-01-01 01:00:00",
                         "2025-01-01 01:00:01"])
    df = pd.DataFrame({"txn_id": range(4), "timestamp": ts, "card_id": 5, "card_bin": 0,
                       "amount": 1.0, "category": "grocery", "is_online": 1, "is_fraud": 0})
    f = add_past_features(df, categories).set_index("txn_id")
    assert f.loc[2, "cnt_1h"] == 2          # [00:00, 01:00) includes the first txn
    assert f.loc[3, "cnt_1h"] == 2          # [00:00:01, 01:00:01) drops the first
    assert f.loc[3, "n_small_1h"] == 2


def test_cards_do_not_mix(categories):
    ts = pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:01:00"])
    df = pd.DataFrame({"txn_id": [0, 1], "timestamp": ts, "card_id": [1, 2], "card_bin": 0,
                       "amount": 50.0, "category": "fuel", "is_online": 0, "is_fraud": 0})
    f = add_past_features(df, categories).set_index("txn_id")
    assert f.loc[1, "cnt_1h"] == 0 and np.isnan(f.loc[1, "secs_since_last"])


def test_velocity_matches_bruteforce(small_df, categories):
    f = add_past_features(small_df, categories)
    sample = f.sample(300, random_state=1)
    for _, row in sample.iterrows():
        prior = f[(f["card_id"] == row["card_id"]) & (f["timestamp"] < row["timestamp"])
                  & (f["timestamp"] >= row["timestamp"] - pd.Timedelta(hours=24))]
        assert row["cnt_24h"] == len(prior)
        assert row["sum_24h"] == pytest.approx(prior["amount"].sum())


def test_leaky_features_really_do_leak(small_df, categories):
    """Sanity check that the experiment's 'leaky' arm actually leaks."""
    base = add_past_features(small_df, categories)
    cutoff = base["timestamp"].quantile(0.5)
    early = base[base["timestamp"] <= cutoff]
    full = add_leaky_features(base).set_index("txn_id")
    trunc = add_leaky_features(early).set_index("txn_id")
    diff = (full.loc[trunc.index, "card_total_txns_all"] != trunc["card_total_txns_all"])
    assert diff.mean() > 0.5
