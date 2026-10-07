"""Controlled leakage experiment.

Two factors, everything else held fixed (same data, same model, same
hyperparameters, same 60/20/20 proportions):

    split     temporal  vs  random
    features  past-only vs  past-only + whole-history card aggregates

The temporal + past-only cell is the honest estimate of how the model will do
on future traffic. Every other cell is what you would *report* if you made that
mistake. Inflation = (leaky metric - honest metric) / honest metric.

Random splits are repeated over several seeds; the mean is reported.

Metric choice: PR-AUC and recall when alerting the top 1% of transactions.
These are rate-based, so they are comparable between a random test set (spread
over all months) and a temporal one (the last month only). A per-day alert
budget would not be comparable here, because the two test sets have very
different numbers of transactions per day.

    python scripts/leakage_experiment.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_alert_budget.evaluate import ranking_metrics, recall_at_alert_rate
from fraud_alert_budget.features import LEAKY_FEATURES, PAST_FEATURES, add_leaky_features
from fraud_alert_budget.model import train_lgbm
from fraud_alert_budget.pipeline import prepare, xy
from fraud_alert_budget.split import random_split, temporal_split


def run_cell(df, split, features, seed, rate, cache: Path | None = None):
    """Train + evaluate one cell. Results are cached so the experiment can be
    resumed if interrupted (each cell is a full LightGBM fit)."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    X_tr, y_tr = xy(df, split.train, features)
    X_va, y_va = xy(df, split.val, features)
    X_te, y_te = xy(df, split.test, features)
    m = train_lgbm(X_tr, y_tr, X_va, y_va, seed=seed)
    s = m.predict_proba(X_te)[:, 1]
    res = {**ranking_metrics(y_te, s), **recall_at_alert_rate(y_te, s, rate),
           "best_iteration": int(m.best_iteration_)}
    res = {k: float(v) for k, v in res.items()}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "ieee"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seeds", type=int, default=3, help="random-split repeats")
    ap.add_argument("--alert-rate", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "leakage_cache" / f"{args.source}_seed{args.seed}"

    df = add_leaky_features(prepare(args.source, args.data_dir, args.seed))
    feature_sets = {"past_only": PAST_FEATURES, "with_leaky": PAST_FEATURES + LEAKY_FEATURES}

    rows = []
    for fname, feats in feature_sets.items():
        r = run_cell(df, temporal_split(df), feats, args.seed, args.alert_rate,
                     cache_dir / f"temporal_{fname}.json")
        rows.append({"split": "temporal", "features": fname, "rep": 0, **r})
        print("temporal", fname, {k: round(v, 4) for k, v in r.items()})
        for rep in range(args.seeds):
            r = run_cell(df, random_split(df, seed=rep), feats, args.seed, args.alert_rate,
                         cache_dir / f"random_{fname}_{rep}.json")
            rows.append({"split": "random", "features": fname, "rep": rep, **r})
            print("random  ", fname, rep, {k: round(v, 4) for k, v in r.items()})

    raw = pd.DataFrame(rows)
    metrics = ["pr_auc", "roc_auc", "precision_at_rate", "recall_at_rate"]
    table = raw.groupby(["split", "features"])[metrics].agg(["mean", "std"])
    means = raw.groupby(["split", "features"])[metrics].mean()

    honest = means.loc[("temporal", "past_only")]
    def inflation(cell):
        return ((means.loc[cell] - honest) / honest * 100).round(1).to_dict()

    comparisons = {
        "random split only (same leakage-safe features)": inflation(("random", "past_only")),
        "leaky features only (temporal split)": inflation(("temporal", "with_leaky")),
        "both mistakes (random split + leaky features)": inflation(("random", "with_leaky")),
    }

    raw.to_csv(out / "leakage_raw.csv", index=False)
    (out / "leakage.json").write_text(json.dumps({
        "honest_baseline": honest.to_dict(),
        "inflation_pct_vs_honest": comparisons,
    }, indent=2), encoding="utf-8")

    md = ["# Leakage experiment\n",
          "Honest baseline = temporal split + leakage-safe features. "
          "Inflation is relative to that baseline.\n",
          f"Random-split cells averaged over {args.seeds} seeds. "
          f"Alert rate for precision/recall: top {args.alert_rate:.0%} of test transactions.\n",
          "## Mean (std) by cell\n", table.round(4).to_markdown(), "\n",
          "## Inflation vs honest baseline (%)\n",
          pd.DataFrame(comparisons).T.to_markdown(), "\n"]
    (out / "leakage.md").write_text("\n".join(md), encoding="utf-8")

    print("\n", pd.DataFrame(comparisons).T.to_string())
    print(f"\nwrote {out}/leakage.md")


if __name__ == "__main__":
    main()
