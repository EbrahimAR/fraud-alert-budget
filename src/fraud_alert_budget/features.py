"""Feature engineering.

The rule for every behavioural feature in `add_past_features`:

    A transaction's features may only use transactions on the same card that
    happened STRICTLY BEFORE it (window [t - w, t)).

This is what a real-time scoring system can actually know when the
transaction arrives. Labels are never used to build features, so there is no
label leakage either (chargeback labels arrive weeks later in practice).

Implementation: rows are sorted by (card, time) and encoded as a single
monotonic integer key, so each window is two `searchsorted` calls plus a
cumulative-sum difference. This is exact, vectorised, and makes the
"strictly before" boundary explicit (`side="left"` on the current time).

`add_leaky_features` deliberately does the common mistake: per-card
aggregates computed over the WHOLE dataset (including the future), plus a
"time until next transaction" feature. These exist only for the leakage
experiment and must never be used in the real pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMALL_AMOUNT = 5.0
WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
_KEY_SCALE = np.int64(10**10)  # card stride in the composite key (seconds)

PAST_FEATURES = [
    # transaction-level (known at the moment of the transaction)
    "log_amount", "hour", "dow", "is_online", "category", "card_bin",
    # card history, strictly before t
    "card_n_prior", "card_mean_amt_prior", "card_std_amt_prior",
    "amt_to_prior_mean", "amt_zscore_prior", "secs_since_last",
    "card_online_share_prior", "is_new_category",
    # velocity windows, strictly before t
    "cnt_1h", "cnt_24h", "cnt_7d", "sum_1h", "sum_24h", "sum_7d", "n_small_1h",
]

LEAKY_FEATURES = [
    "card_total_txns_all", "card_mean_amt_all", "card_std_amt_all",
    "amt_to_card_mean_all", "secs_to_next_txn",
]


class _PriorIndex:
    """Fast 'strictly-before' lookups for one grouping (e.g. per card)."""

    def __init__(self, group_codes: np.ndarray, t_sec: np.ndarray):
        if t_sec.max() + max(WINDOWS.values()) >= _KEY_SCALE:
            raise ValueError("time span too large for composite key")
        self.order = np.lexsort((t_sec, group_codes))
        g = group_codes[self.order].astype(np.int64)
        t = t_sec[self.order].astype(np.int64)
        self.key = g * _KEY_SCALE + t
        self.t = t
        self.g = g
        self.group_start = np.searchsorted(self.key, g * _KEY_SCALE, side="left")
        # first row of this group at the SAME time -> excludes itself and ties
        self.before_now = np.searchsorted(self.key, self.key, side="left")

    def window_start(self, w: int) -> np.ndarray:
        return np.searchsorted(self.key, self.key - w, side="left")

    def sorted_values(self, v: np.ndarray) -> np.ndarray:
        return v[self.order]

    def range_sum(self, v_sorted: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Sum of v over sorted positions [lo, hi), where lo and hi-1 lie in
        the same group.

        The cumulative sum restarts at every group, so a row's result depends
        only on its own card's earlier rows, bit for bit. (A single global
        cumsum would let floating-point rounding from other cards leak in.)
        """
        v = np.asarray(v_sorted, float)
        incl = pd.Series(v).groupby(self.g).cumsum().to_numpy()
        excl = incl - v
        return np.where(hi > lo, incl[hi - 1] - excl[lo], 0.0)

    def unsort(self, x_sorted: np.ndarray) -> np.ndarray:
        out = np.empty_like(x_sorted)
        out[self.order] = x_sorted
        return out


def _prepare(df: pd.DataFrame, categories: list[str] | None) -> pd.DataFrame:
    df = df.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True).copy()
    cats = categories if categories is not None else sorted(df["category"].astype(str).unique())
    df["category"] = pd.Categorical(df["category"].astype(str), categories=cats)
    return df


def add_past_features(df: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of `df`, sorted by time, with leakage-safe features added.

    `categories` fixes the category vocabulary (a vocabulary is not leakage,
    but fixing it keeps encodings identical across runs and subsets).
    """
    df = _prepare(df, categories)
    t_sec = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().to_numpy().astype(np.int64)
    amount = df["amount"].to_numpy(float)
    card = pd.factorize(df["card_id"])[0]

    # --- transaction-level ---
    df["log_amount"] = np.log1p(amount)
    df["hour"] = df["timestamp"].dt.hour.astype(np.int16)
    df["dow"] = df["timestamp"].dt.dayofweek.astype(np.int16)

    # --- card history strictly before t ---
    ix = _PriorIndex(card, t_sec)
    lo_all, hi = ix.group_start, ix.before_now
    a_s = ix.sorted_values(amount)
    n_prior = (hi - lo_all).astype(float)
    # shift by each card's first amount before summing squares: variance is
    # shift-invariant and this avoids catastrophic cancellation
    ref = a_s[lo_all]
    d = a_s - ref
    s1 = ix.range_sum(d, lo_all, hi)
    s2 = ix.range_sum(d ** 2, lo_all, hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d = np.where(n_prior > 0, s1 / n_prior, np.nan)
        mean = mean_d + ref
        var = np.where(n_prior > 1, (s2 - n_prior * mean_d ** 2) / (n_prior - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        online_share = np.where(
            n_prior > 0,
            ix.range_sum(ix.sorted_values(df["is_online"].to_numpy(float)), lo_all, hi) / n_prior,
            np.nan,
        )
    has_prev = hi > lo_all
    prev_idx = np.where(has_prev, hi - 1, 0)
    since_last = np.where(has_prev, ix.t - ix.t[prev_idx], np.nan)

    df["card_n_prior"] = ix.unsort(n_prior)
    df["card_mean_amt_prior"] = ix.unsort(mean)
    df["card_std_amt_prior"] = ix.unsort(std)
    df["secs_since_last"] = ix.unsort(since_last.astype(float))
    df["card_online_share_prior"] = ix.unsort(online_share)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["amt_to_prior_mean"] = amount / df["card_mean_amt_prior"].to_numpy()
        df["amt_zscore_prior"] = (amount - df["card_mean_amt_prior"].to_numpy()) / (
            df["card_std_amt_prior"].to_numpy() + 1.0)

    # --- velocity windows [t - w, t) ---
    small_s = ix.sorted_values((amount < SMALL_AMOUNT).astype(float))
    for name, w in WINDOWS.items():
        lo = ix.window_start(w)
        df[f"cnt_{name}"] = ix.unsort((hi - lo).astype(float))
        df[f"sum_{name}"] = ix.unsort(ix.range_sum(a_s, lo, hi))
        if name == "1h":
            df["n_small_1h"] = ix.unsort(ix.range_sum(small_s, lo, hi))

    # --- first time this card is seen at this category (strictly before t) ---
    card_cat = pd.factorize(pd.Series(card).astype(str) + "|" + df["category"].astype(str))[0]
    ix_cc = _PriorIndex(card_cat, t_sec)
    df["is_new_category"] = ix_cc.unsort(
        (ix_cc.before_now == ix_cc.group_start).astype(np.int8))

    return df


def add_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """DELIBERATELY LEAKY. Whole-history aggregates, as produced by the
    common `df.groupby('card_id').transform(...)` before splitting."""
    df = df.copy()
    g = df.groupby("card_id")["amount"]
    df["card_total_txns_all"] = g.transform("size").astype(float)
    df["card_mean_amt_all"] = g.transform("mean")
    df["card_std_amt_all"] = g.transform("std")
    df["amt_to_card_mean_all"] = df["amount"] / df["card_mean_amt_all"]
    nxt = df.groupby("card_id")["timestamp"].shift(-1)          # the future!
    df["secs_to_next_txn"] = (nxt - df["timestamp"]).dt.total_seconds()
    return df
