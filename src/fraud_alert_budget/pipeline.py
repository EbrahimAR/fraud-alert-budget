"""Shared glue used by the scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load
from .features import PAST_FEATURES, add_past_features


def prepare(source: str = "synthetic", data_dir: str | None = None, seed: int = 42) -> pd.DataFrame:
    kwargs = {"seed": seed} if source == "synthetic" else {}
    df = load(source, data_dir=data_dir, **kwargs)
    return add_past_features(df)


def n_days(ts: pd.Series) -> int:
    return int(pd.to_datetime(ts).dt.floor("D").nunique())


def xy(df: pd.DataFrame, mask: np.ndarray, features: list[str] = PAST_FEATURES):
    part = df.loc[mask]
    return part[features], part["is_fraud"].to_numpy()
