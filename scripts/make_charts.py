"""Summary charts for the README, drawn only from the generated report files.

    python scripts/make_charts.py                                          # synthetic
    python scripts/make_charts.py --reports-dir reports/ieee --currency "$"

Reads budget_sweep.csv, results.json and leakage.json from the reports
directory and writes budget_summary.png and leakage_summary.png next to them.
Every number on the charts comes from those files; nothing is hard-coded.
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
from matplotlib.ticker import ScalarFormatter

BLUE, RED = "#2a78d6", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FIGSIZE, DPI = (8, 8), 150          # 1200 x 1200 px
RANDOM_ONLY = "random split only (same leakage-safe features)"
LEAKAGE_METRICS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def budget_summary(reports: Path, cur: str) -> None:
    sweep = pd.read_csv(reports / "budget_sweep.csv")
    budget = json.loads((reports / "results.json").read_text(encoding="utf-8"))["budget"]
    g = (sweep[(sweep["scorer"] == "lightgbm") & (sweep["mode"] == "streaming")]
         .sort_values("k_per_day").reset_index(drop=True))
    if budget not in set(g["k_per_day"]):
        raise SystemExit(f"headline budget {budget} is not in {reports / 'budget_sweep.csv'}")
    head = g[g["k_per_day"] == budget].iloc[0]
    k = g["k_per_day"].to_numpy()
    net_k = g["net_benefit"].to_numpy() / 1000
    fp = g["fp_per_fraud_caught"].replace(np.inf, np.nan).to_numpy()

    fig, (top, bot) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, facecolor="white",
                                   layout="constrained")
    for ax, y, label in ((top, net_k, f"Net benefit ({cur}k)"),
                         (bot, fp, "False positives per fraud caught")):
        _style(ax)
        ax.plot(k, y, color=BLUE, lw=2, marker="o", ms=7, mec="white", mew=1.5)
        ax.axvline(budget, color=MUTED, ls="--", lw=1)
        ax.set_ylabel(label, color=INK, fontsize=11)
    top.axhline(0, color=MUTED, lw=0.8)
    top.set_title("LightGBM under a daily alert budget (streaming, test period)",
                  color=INK, fontsize=13, loc="left", pad=12)

    # ring the headline point and put its note beside the dashed line, at
    # whichever end of the panel the curve is not (so it never covers the line)
    head_net_k = head["net_benefit"] / 1000
    top.plot([budget], [head_net_k], "o", ms=13, mfc="none", mec=BLUE, mew=1.5)
    lo, hi = top.get_ylim()
    point_high = head_net_k - lo > (hi - lo) / 2
    on_right = int(np.flatnonzero(k == budget)[0]) >= len(k) / 2
    top.annotate(
        f"{budget:,} alerts/day\n"
        f"{cur}{head_net_k:,.1f}k net benefit\n"
        f"{head['recall_at_budget']:.0%} of frauds caught",
        xy=(budget, 0.06 if point_high else 0.94), xycoords=("data", "axes fraction"),
        xytext=(-10 if on_right else 10, 0), textcoords="offset points",
        ha="right" if on_right else "left", va="bottom" if point_high else "top",
        color=INK, fontsize=10.5, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    bot.set_xscale("log")
    bot.set_xticks(k)
    bot.xaxis.set_major_formatter(ScalarFormatter())
    bot.minorticks_off()
    bot.set_xlabel("Alert budget (alerts per day, log scale)", color=INK, fontsize=11)
    _save(fig, reports / "budget_summary.png")


def _split_means(leak: dict) -> tuple[dict, dict]:
    """(temporal, random) metrics for the leakage-safe features. Reports
    written before `cell_means` existed only hold the baseline and the
    rounded inflation, so the random cell is rebuilt from those."""
    if "cell_means" in leak:
        return leak["cell_means"]["temporal/past_only"], leak["cell_means"]["random/past_only"]
    base = leak["honest_baseline"]
    infl = leak["inflation_pct_vs_honest"][RANDOM_ONLY]
    return base, {m: base[m] * (1 + infl[m] / 100) for m in LEAKAGE_METRICS}


def leakage_summary(reports: Path) -> None:
    temporal, random = _split_means(json.loads((reports / "leakage.json").read_text(encoding="utf-8")))
    x = np.arange(len(LEAKAGE_METRICS))
    width = 0.34

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white", layout="constrained")
    _style(ax)
    series = ((temporal, BLUE, "Time-based split", -1), (random, RED, "Random split", 1))
    for values, color, label, side in series:
        y = [values[m] for m in LEAKAGE_METRICS]
        bars = ax.bar(x + side * (width / 2 + 0.01), y, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, color=INK, fontsize=10.5)
    for i, m in enumerate(LEAKAGE_METRICS):
        inflation = (random[m] - temporal[m]) / temporal[m] * 100
        ax.text(i, max(temporal[m], random[m]) + 0.075, f"{inflation:+.1f}%",
                ha="center", va="bottom", color=INK, fontsize=15, fontweight="bold")

    ax.set_xticks(x, LEAKAGE_METRICS.values(), fontsize=12, color=INK)
    ax.set_ylim(0, max(1.0, max(random[m] for m in LEAKAGE_METRICS)) + 0.2)
    ax.set_ylabel("Score on the test set", color=INK, fontsize=11)
    ax.set_title("Same model and features: what a random split reports\n"
                 "vs the time-based (honest) split",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5, labelcolor=INK)
    _save(fig, reports / "leakage_summary.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--currency", default="£")
    args = ap.parse_args()
    reports = Path(args.reports_dir)
    cur = args.currency.replace("$", r"\$")   # a bare "$" would start mathtext
    budget_summary(reports, cur)
    leakage_summary(reports)


if __name__ == "__main__":
    main()
