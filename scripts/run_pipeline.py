"""Train on the past, evaluate on the future, under a daily alert budget.

    python scripts/run_pipeline.py                         # synthetic data
    python scripts/run_pipeline.py --source ieee --data-dir data/ieee-fraud-detection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_alert_budget.evaluate import (CostModel, budget_report, calibrate_threshold,
                                         ranking_metrics, select_alerts)
from fraud_alert_budget.features import PAST_FEATURES
from fraud_alert_budget.model import feature_importance, train_lgbm
from fraud_alert_budget.pipeline import n_days, prepare, xy
from fraud_alert_budget.rules import rule_score
from fraud_alert_budget.split import temporal_split

# headline budget, sweep and currency symbol per source (IEEE-CIS amounts are USD)
SOURCE_DEFAULTS = {
    "synthetic": {"budget": 25, "sweep": "5,10,25,50,100,200", "currency": "£"},
    "ieee": {"budget": 500, "sweep": "25,50,100,200,500,1000", "currency": "$"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "ieee"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="headline alerts per day (default: 25 synthetic, 500 ieee)")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated alert budgets (default depends on --source)")
    ap.add_argument("--gap-days", type=float, default=0.0, help="label-maturity gap")
    ap.add_argument("--review-cost", type=float, default=4.0)
    ap.add_argument("--fp-cost", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    defaults = SOURCE_DEFAULTS[args.source]
    if args.budget is None:
        args.budget = defaults["budget"]
    if args.sweep is None:
        args.sweep = defaults["sweep"]
    cur = defaults["currency"]
    # "$" would start mathtext in matplotlib and maths in GitHub markdown
    cur_esc = cur.replace("$", r"\$")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cost = CostModel(review_cost=args.review_cost, fp_friction_cost=args.fp_cost)

    df = prepare(args.source, args.data_dir, args.seed)
    split = temporal_split(df, gap_days=args.gap_days)
    print(split.describe(df).to_string(index=False), "\n")

    X_tr, y_tr = xy(df, split.train)
    X_va, y_va = xy(df, split.val)
    X_te, y_te = xy(df, split.test)
    model = train_lgbm(X_tr, y_tr, X_va, y_va, seed=args.seed)
    print(f"best iteration: {model.best_iteration_}")

    va, te = df.loc[split.val], df.loc[split.test].reset_index(drop=True)
    scores = {
        "lightgbm": (model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]),
    }
    if args.source == "synthetic":
        # the rules target the synthetic fraud patterns; on other data they
        # rank at random level, so they are not a meaningful baseline there
        scores["rules"] = (rule_score(va), rule_score(te))
    val_days = n_days(va["timestamp"])

    ranking = {name: ranking_metrics(y_te, s[1]) for name, s in scores.items()}

    rows = []
    budgets = sorted({*map(int, args.sweep.split(",")), args.budget})
    for k in budgets:
        for name, (s_va, s_te) in scores.items():
            thr = calibrate_threshold(s_va, val_days, k)
            for mode in ("streaming", "topk"):
                alert = select_alerts(te["timestamp"], s_te, k, mode=mode, threshold=thr)
                r = budget_report(y_te, te["amount"].to_numpy(), alert, te["timestamp"], k, cost)
                rows.append({"scorer": name, "mode": mode, **r})
    sweep = pd.DataFrame(rows)

    # per fraud-pattern recall at the headline budget (synthetic data only)
    by_mo = None
    if "fraud_mo" in te.columns:
        s_va, s_te = scores["lightgbm"]
        alert = select_alerts(te["timestamp"], s_te, args.budget, "streaming",
                              calibrate_threshold(s_va, val_days, args.budget))
        f = te.assign(alert=alert)[te["is_fraud"] == 1]
        by_mo = (f.groupby("fraud_mo")
                   .agg(frauds=("alert", "size"), caught=("alert", "sum"))
                   .assign(recall=lambda d: d["caught"] / d["frauds"]))

    imp = feature_importance(model, PAST_FEATURES)
    head = sweep[(sweep["k_per_day"] == args.budget)].set_index(["scorer", "mode"])

    # ---------- write outputs ----------
    sweep.to_csv(out / "budget_sweep.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    (out / "results.json").write_text(json.dumps({
        "source": args.source, "budget": args.budget, "gap_days": args.gap_days,
        "best_iteration": int(model.best_iteration_),
        "split": split.describe(df).astype(str).to_dict("records"),
        "ranking_metrics_test": ranking,
        "headline": head.reset_index().to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (name, mode), g in sweep.groupby(["scorer", "mode"]):
        style = "-" if mode == "streaming" else ":"
        ax[0].plot(g["k_per_day"], g["net_benefit"] / 1000, style, marker="o", label=f"{name} ({mode})")
        ax[1].plot(g["k_per_day"], g["fp_per_fraud_caught"], style, marker="o", label=f"{name} ({mode})")
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set(xscale="log", xlabel="alert budget (alerts/day)", ylabel=f"net benefit on test ({cur_esc}k)",
              title=f"Net {cur_esc} benefit vs alert capacity")
    ax[1].set(xscale="log", xlabel="alert budget (alerts/day)", ylabel="false positives per fraud caught",
              title="Analyst workload per catch")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "budget_curves.png", dpi=150)

    cols = ["alerts_per_day", "queue_utilisation", "frauds_caught", "frauds_total",
            "precision_at_budget", "recall_at_budget", "value_capture",
            "fp_per_fraud_caught", "fraud_amount_caught", "net_benefit"]
    money = {c: f"{c} ({cur_esc})" for c in ("fraud_amount_caught", "net_benefit")}
    sweep_table = (sweep[sweep["mode"] == "streaming"]
                   .pivot(index="k_per_day", columns="scorer",
                          values=["net_benefit", "fp_per_fraud_caught", "recall_at_budget"]))
    sweep_table.columns = [f"{money.get(metric, metric)}, {scorer}"
                           for metric, scorer in sweep_table.columns]
    md = [f"# Results ({args.source} data)\n",
          f"Test period: {te['timestamp'].min():%Y-%m-%d} to {te['timestamp'].max():%Y-%m-%d}, "
          f"{len(te):,} transactions, {int(y_te.sum())} frauds.\n",
          f"Cost model: {cur_esc}{cost.review_cost:g} per review, "
          f"{cur_esc}{cost.fp_friction_cost:g} friction per false positive.\n",
          f"## Headline: {args.budget} alerts/day\n",
          head[cols].rename(columns=money).round(3).to_markdown(), "\n",
          "## Ranking metrics on test (reference only)\n",
          pd.DataFrame(ranking).T.round(4).to_markdown(), "\n",
          "## Budget sweep (streaming mode)\n",
          sweep_table.round(2).to_markdown(), "\n",
          "## Top features (gain share)\n",
          imp.head(10).round(3).to_markdown(index=False), "\n"]
    if by_mo is not None:
        md += ["## Recall by fraud pattern at headline budget (synthetic only)\n",
               by_mo.round(3).to_markdown(), "\n"]
    md.append("![budget curves](budget_curves.png)\n")
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")

    print(head[cols].round(3).to_string())
    print("\nranking:", json.dumps(ranking, indent=1))
    if by_mo is not None:
        print(by_mo)
    print(f"\nwrote {out}/results.md")


if __name__ == "__main__":
    main()
