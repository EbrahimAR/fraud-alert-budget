"""Synthetic card-transaction generator.

Lets the whole pipeline run without downloading IEEE-CIS. The generator is
deliberately built to contain the properties that make fraud evaluation hard:

* Fraud is rare (~0.5% of transactions) and arrives in *bursts* on compromised
  cards, so neighbouring fraud rows look almost identical. A random split puts
  siblings from the same burst in train and test, which flatters the model.
* Two fraud modus operandi (MO):
    - MO "A" (card testing -> cash-out): a few tiny online transactions within
      minutes, then large purchases in high-risk categories, often at night.
    - MO "B" (account-takeover mimicry): moderately inflated amounts, spread
      over a day or two, in normal hours. Rare early on, then it *emerges* in
      the later months. This is concept drift: a model tested on the future
      meets a pattern it has barely seen.
* Genuine customers occasionally make large, unusual purchases, which creates
  realistic false positives.

None of this is real data; numbers produced on it are for demonstrating the
methodology, not for claiming real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery", "fuel", "restaurant", "retail", "online_retail",
    "travel", "utilities", "electronics", "gift_cards", "crypto",
]
_GENUINE_CAT_P = np.array([0.25, 0.12, 0.15, 0.15, 0.15, 0.04, 0.08, 0.04, 0.015, 0.005])
_GENUINE_CAT_P = _GENUINE_CAT_P / _GENUINE_CAT_P.sum()
_CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def _mo_b_probability(day: float) -> float:
    """Share of compromises that use MO B, rising after day 100 (drift)."""
    if day < 100:
        return 0.03
    return float(min(0.6, 0.03 + (day - 100) / 30 * 0.57))


def generate(
    n_cards: int = 5000,
    n_days: int = 150,
    seed: int = 42,
    start: str = "2025-01-01",
    compromise_rate: float = 0.07,
) -> pd.DataFrame:
    """Generate a transaction table in the project's common schema."""
    rng = np.random.default_rng(seed)
    horizon = n_days * 86400

    card_ids, ts, amt, cat, online, fraud, mo = [], [], [], [], [], [], []

    def emit(c, t, a, k, o, f, m):
        card_ids.append(c); ts.append(t); amt.append(a)
        cat.append(k); online.append(o); fraud.append(f); mo.append(m)

    n_bins = 50
    bin_risk = rng.lognormal(0.0, 0.6, n_bins)
    bin_risk /= bin_risk.mean()
    card_bin = rng.integers(0, n_bins, n_cards)

    for c in range(n_cards):
        # --- card profile ---
        rate = rng.lognormal(np.log(0.45), 0.7)            # genuine txns per day
        mu_amt = rng.normal(3.2, 0.5)                       # log-£ typical spend (~£25)
        peak_hour = rng.normal(14, 3)
        cat_pref = rng.dirichlet(_GENUINE_CAT_P * 30)
        p_online = rng.beta(2, 4)
        start_day = rng.uniform(0, n_days * 0.5) if rng.random() < 0.3 else 0.0
        active_days = n_days - start_day

        # --- genuine transactions ---
        n = rng.poisson(rate * active_days)
        if n > 0:
            days = rng.uniform(start_day, n_days, n).astype(int)
            hours = np.clip(rng.normal(peak_hour, 3.5, n), 6, 23.9)
            t = days * 86400 + hours * 3600 + rng.uniform(0, 60, n)
            a = rng.lognormal(mu_amt, 0.7, n)
            k = rng.choice(len(CATEGORIES), n, p=cat_pref)
            # occasional genuine big-ticket purchase -> realistic false positives
            big = rng.random(n) < 0.01
            a[big] *= rng.lognormal(np.log(8), 0.5, big.sum())
            k[big] = rng.choice([_CAT_IDX["electronics"], _CAT_IDX["travel"]], big.sum())
            o = rng.random(n) < p_online
            for i in range(n):
                emit(c, t[i], a[i], k[i], o[i], 0, "")

        # --- fraud events ---
        n_events = rng.poisson(compromise_rate * bin_risk[card_bin[c]] * active_days / n_days)
        for _ in range(n_events):
            day = rng.uniform(start_day, n_days - 2)
            if rng.random() < _mo_b_probability(day):
                # MO B: account-takeover mimicry
                t0 = int(day) * 86400 + np.clip(rng.normal(peak_hour, 3), 7, 23) * 3600
                m = rng.integers(2, 8)
                gaps = rng.exponential(6 * 3600, m)
                for g in np.cumsum(gaps):
                    a = np.exp(mu_amt) * rng.lognormal(np.log(3.0), 0.4)
                    k = rng.choice([_CAT_IDX["online_retail"], _CAT_IDX["travel"],
                                    _CAT_IDX["electronics"], _CAT_IDX["retail"]],
                                   p=[0.4, 0.25, 0.2, 0.15])
                    emit(c, t0 + g, a, k, rng.random() < 0.8, 1, "B")
            else:
                # MO A: card testing then cash-out
                hour = rng.uniform(0, 6) if rng.random() < 0.5 else rng.uniform(0, 24)
                t = int(day) * 86400 + hour * 3600
                if rng.random() < 0.7:
                    for _ in range(rng.integers(1, 5)):
                        t += rng.exponential(90)
                        emit(c, t, rng.uniform(0.5, 3.0), _CAT_IDX["online_retail"], True, 1, "A")
                for _ in range(rng.integers(1, 7)):
                    t += rng.exponential(20 * 60)
                    k = rng.choice([_CAT_IDX["electronics"], _CAT_IDX["gift_cards"],
                                    _CAT_IDX["crypto"]], p=[0.5, 0.35, 0.15])
                    emit(c, t, rng.lognormal(np.log(250), 0.6), k, rng.random() < 0.9, 1, "A")

    df = pd.DataFrame({
        "card_id": np.array(card_ids, dtype=np.int64),
        "t": np.array(ts, dtype=float),
        "amount": np.round(np.array(amt, dtype=float), 2),
        "category": np.array(CATEGORIES, dtype=object)[np.array(cat, dtype=int)],
        "is_online": np.array(online, dtype=np.int8),
        "is_fraud": np.array(fraud, dtype=np.int8),
        "fraud_mo": np.array(mo, dtype=object),
    })
    df = df[(df["t"] >= 0) & (df["t"] < horizon)]
    df["timestamp"] = pd.Timestamp(start) + pd.to_timedelta(df["t"].round(), unit="s")
    df["card_bin"] = card_bin[df["card_id"].to_numpy()]
    df = df.drop(columns="t").sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df), dtype=np.int64))
    return df
