"""Train/validation/test splitting.

`temporal_split` is the honest split: every validation row is later than every
training row, and every test row is later than every validation row. The
optional `gap_days` drops the tail of each earlier period, which simulates
label maturity: chargebacks take weeks to arrive, so the most recent
transactions at training time do not yet have trustworthy labels.

`random_split` exists only for the leakage experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: np.ndarray  # boolean masks aligned to the dataframe
    val: np.ndarray
    test: np.ndarray

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            m = getattr(self, name)
            part = df.loc[m]
            rows.append({
                "split": name,
                "rows": int(m.sum()),
                "fraud_rate": float(part["is_fraud"].mean()),
                "start": part["timestamp"].min(),
                "end": part["timestamp"].max(),
            })
        return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2,
                   gap_days: float = 0.0) -> Split:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must be in (0, 1) and sum to < 1")
    t = df["timestamp"]
    t0, t1 = t.min(), t.max()
    span = t1 - t0
    c1 = t0 + span * train_frac
    c2 = t0 + span * (train_frac + val_frac)
    gap = pd.Timedelta(days=gap_days)
    train = (t < c1 - gap).to_numpy()
    val = ((t >= c1) & (t < c2 - gap)).to_numpy()
    test = (t >= c2).to_numpy()
    return Split(train, val, test)


def random_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2,
                 seed: int = 0) -> Split:
    n = len(df)
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(n * train_frac), int(n * val_frac)
    train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
    train[perm[:n_tr]] = True
    val[perm[n_tr:n_tr + n_va]] = True
    test[perm[n_tr + n_va:]] = True
    return Split(train, val, test)
