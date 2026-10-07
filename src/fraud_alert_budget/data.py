"""Data loading. Every source is mapped to one common schema so the rest of
the pipeline never needs to know where the data came from.

Schema
------
txn_id     int        unique, increasing with time
timestamp  datetime   when the transaction happened
card_id    int        entity key used for behavioural features
card_bin   int        coarse card attribute (issuer-like), safe to use as a feature
amount     float      transaction amount
category   str        merchant category / product code
is_online  int8       card-not-present flag
is_fraud   int8       label
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["txn_id", "timestamp", "card_id", "card_bin", "amount",
                    "category", "is_online", "is_fraud"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must be a datetime column")
    if df["txn_id"].duplicated().any():
        raise ValueError("txn_id must be unique")
    if df["amount"].lt(0).any():
        raise ValueError("negative amounts found")
    return df


def load_synthetic(**kwargs) -> pd.DataFrame:
    from .synth import generate
    return validate(generate(**kwargs))


def load_ieee_cis(data_dir: str | Path) -> pd.DataFrame:
    """Load Kaggle IEEE-CIS `train_transaction.csv` into the common schema.

    IEEE-CIS has no real card key, so we build the community-standard proxy:
    card1 + addr1 + (transaction day - D1), where D1 is "days since the card
    was first seen". This approximates a single customer's card.

    The dataset's `TransactionDT` is a seconds offset from an unknown start;
    we anchor it at 2017-12-01, as is conventional. Amounts are USD, but the
    evaluation treats them as the currency unit (reported as £ for
    consistency with the rest of the project).
    """
    path = Path(data_dir) / "train_transaction.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from "
            f"https://www.kaggle.com/c/ieee-fraud-detection/data and unzip into {data_dir}."
        )
    cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
            "ProductCD", "card1", "addr1", "D1"]
    raw = pd.read_csv(path, usecols=cols)

    day = raw["TransactionDT"] // 86400
    d1 = raw["D1"].fillna(0).round().astype(np.int64)
    addr = raw["addr1"].fillna(-1).astype(np.int64)
    uid = raw["card1"].astype(str) + "_" + addr.astype(str) + "_" + (day - d1).astype(str)

    df = pd.DataFrame({
        "txn_id": raw["TransactionID"].astype(np.int64),
        "timestamp": pd.Timestamp("2017-12-01") + pd.to_timedelta(raw["TransactionDT"], unit="s"),
        "card_id": pd.factorize(uid)[0].astype(np.int64),
        "card_bin": raw["card1"].astype(np.int64),
        "amount": raw["TransactionAmt"].astype(float),
        "category": raw["ProductCD"].astype(str),
        "is_online": np.ones(len(raw), dtype=np.int8),   # IEEE-CIS is all card-not-present
        "is_fraud": raw["isFraud"].astype(np.int8),
    })
    df = df.sort_values(["timestamp", "txn_id"], kind="mergesort").reset_index(drop=True)
    return validate(df)


def load(source: str = "synthetic", data_dir: str | Path | None = None, **kwargs) -> pd.DataFrame:
    if source == "synthetic":
        return load_synthetic(**kwargs)
    if source == "ieee":
        if data_dir is None:
            raise ValueError("--data-dir is required for the IEEE-CIS source")
        return load_ieee_cis(data_dir)
    raise ValueError(f"unknown source {source!r}")
