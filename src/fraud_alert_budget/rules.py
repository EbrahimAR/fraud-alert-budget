"""A hand-written rules baseline, the kind of thing many fraud teams start
with. Any model has to beat this, not just a random ranking, to justify
itself. It uses only the leakage-safe features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rule_score(df: pd.DataFrame) -> np.ndarray:
    ratio = df["amt_to_prior_mean"].fillna(1.0)
    score = (
        3.0 * (df["n_small_1h"] >= 2)                     # card testing
        + 2.0 * (df["cnt_1h"] >= 3)                       # velocity burst
        + 2.0 * (ratio > 5)                               # amount far above habit
        + 1.0 * (df["is_new_category"] == 1)
        + 1.0 * df["hour"].between(0, 5)                  # night-time
        + 1.0 * (df["is_online"] == 1)
    )
    # tiny tie-breaker on amount so equal rule scores rank bigger exposures first
    return score.to_numpy(float) + 1e-6 * np.log1p(df["amount"].to_numpy(float))
