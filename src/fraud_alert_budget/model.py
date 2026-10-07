"""LightGBM training with early stopping on the validation period."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def train_lgbm(X_train: pd.DataFrame, y_train: np.ndarray,
               X_val: pd.DataFrame, y_val: np.ndarray,
               seed: int = 0, params: dict | None = None) -> lgb.LGBMClassifier:
    """Fit LightGBM; stop when validation average precision stops improving.

    Average precision (PR-AUC) is used for early stopping rather than
    log-loss or ROC-AUC because with ~0.5% positives ROC-AUC is dominated by
    the easy negatives, while PR-AUC tracks the top of the ranking, which is
    the only part an alert-budgeted team ever sees.
    """
    p = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
    model = lgb.LGBMClassifier(**p)
    fit_kw = dict(eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    try:  # LightGBM >= 4.6 API
        model.fit(X_train, y_train, eval_X=(X_val,), eval_y=(y_val,), **fit_kw)
    except TypeError:  # older 4.x
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kw)
    return model


def feature_importance(model: lgb.LGBMClassifier, features: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": features, "gain": gain})
            .assign(share=lambda d: d["gain"] / d["gain"].sum())
            .sort_values("gain", ascending=False)
            .reset_index(drop=True))
