"""
Charting functions shared by the batch script (analysis.py) and the notebook.

Every function saves a PNG to outputs/ **and returns the Figure**, so the same
call renders inline in the notebook and writes a committed artifact headlessly.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as U

PAL = U.class_palette()


def _bar(ax, idx, vals, colors, title, ylab, fmt="{:.0f}", yerr=None):
    bars = ax.bar(range(len(idx)), vals, color=[colors[i] for i in idx],
                  yerr=yerr, capsize=4, edgecolor="#333", linewidth=0.6)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylab)
    for i, (b, v) in enumerate(zip(bars, vals)):
        top = v + (yerr[1][i] if yerr is not None else 0)
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, top),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    xytext=(0, 5), textcoords="offset points")
    return bars


def timeline(sent, merged):
    daily_vol = merged.groupby("date")["Size USD"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(sent["date"], sent["value"], color="#cccccc", alpha=0.5, zorder=1)
    ax.scatter(sent["date"], sent["value"], c=sent["classification"].map(U.SENT_COLORS),
               s=7, zorder=2)
    ax.set_ylabel("Fear & Greed value (0-100)")
    ax.set_title("Market sentiment over time, with trader activity")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.bar(daily_vol.index, daily_vol.values, width=1.0, color="#3d6fb4", alpha=0.35)
    ax2.set_ylabel("Daily traded volume ($M)", color="#3d6fb4")
    ax2.grid(False)
    ax.set_xlim(merged["date"].min(), merged["date"].max())
    U.savefig(fig, "01_sentiment_timeline")
    return fig


def pnl_distribution(merged):
    r = merged.loc[merged["is_realized"], "net_pnl"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(r, bins=np.linspace(-2000, 2000, 120), color="#4477aa", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#333", lw=1)
    ax.axvline(r.median(), color="#cc3311", lw=2, ls="--", label=f"median = ${r.median():,.2f}")
    ax.set_title("Net realized PnL per closing trade is violently fat-tailed")
    ax.set_xlabel("Net PnL per realized trade ($, clipped to +/-2000 for display)")
    ax.set_ylabel("trades")
    ax.legend(fontsize=11)
    ax.text(0.99, 0.95,
            f"n={len(r):,}\nmean=${r.mean():,.0f}\nstd=${r.std():,.0f}\n"
            f"min=${r.min():,.0f}\nmax=${r.max():,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    U.savefig(fig, "02_pnl_distribution")
    return fig


def pnl_by_regime(summ):
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = list(summ.index)
    err = np.array([summ["median_net_pnl"] - summ["ci_lo"],
                    summ["ci_hi"] - summ["median_net_pnl"]])
    _bar(ax, idx, summ["median_net_pnl"].values, PAL,
         "Median net PnL per account-day by sentiment regime",
         "Median net PnL per account-day ($)", fmt="${:,.0f}", yerr=err)
    ax.axhline(0, color="#333", lw=1)
    ax.annotate("error bars = bootstrap 95% CI", (0.02, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=9, color="#666", style="italic")
    U.savefig(fig, "03_net_pnl_by_regime")
    return fig


def winrate_by_regime(summ):
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = list(summ.index)
    _bar(ax, idx, (summ["win_rate"] * 100).values, PAL,
         "Profitable-day rate by sentiment regime",
         "% of account-days with net PnL > 0", fmt="{:.1f}%")
    ax.axhline(50, color="#333", lw=1, ls=":")
    U.savefig(fig, "04_winrate_by_regime")
    return fig


def behaviour(beh):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    idx = list(beh.index)
    _bar(axes[0, 0], idx, beh["avg_size_usd"].values, PAL,
         "Average trade size", "Size USD ($)", fmt="${:,.0f}")
    _bar(axes[0, 1], idx, beh["median_size_usd"].values, PAL,
         "Median trade size", "Size USD ($)", fmt="${:,.0f}")
    _bar(axes[1, 0], idx, (beh["long_frac"] * 100).values, PAL,
         "Long-position share", "% of fills on Long side", fmt="{:.1f}%")
    axes[1, 0].axhline(50, color="#333", lw=1, ls=":")
    _bar(axes[1, 1], idx, (beh["open_frac"] * 100).values, PAL,
         "New-position (open) share", "% of fills opening a position", fmt="{:.1f}%")
    fig.suptitle("Trader behaviour shifts with sentiment", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    U.savefig(fig, "05_behaviour_by_regime")
    return fig


def per_account(piv):
    fig, ax = plt.subplots(figsize=(8, 8))
    lim = np.nanpercentile(np.abs(piv.values), 98)
    ax.axline((0, 0), slope=1, color="#888", ls="--", zorder=1)
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)
    colors = np.where(piv["Fear"] > piv["Greed"], U.SENT_COLORS["Fear_bin"], U.SENT_COLORS["Greed_bin"])
    ax.scatter(piv["Greed"], piv["Fear"], c=colors, s=70, edgecolor="#222", zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Median net PnL per Greed-day ($)")
    ax.set_ylabel("Median net PnL per Fear-day ($)")
    ax.set_title("Per-account performance: Fear vs Greed days\n(points above the line = better in Fear)")
    U.savefig(fig, "06_per_account_fear_vs_greed")
    return fig


def cohorts(ad, top, bottom):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(U.CLASS_ORDER)); w = 0.38
    for off, grp, lab, col in [(-w / 2, top, "Top cohort (most profitable)", "#2a9d6f"),
                               (w / 2, bottom, "Bottom cohort", "#c0392b")]:
        med = [ad[(ad["Account"].isin(grp)) & (ad["classification"] == c)]["net_pnl"].median()
               for c in U.CLASS_ORDER]
        ax.bar(x + off, med, w, label=lab, color=col, edgecolor="#222", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(U.CLASS_ORDER, rotation=20, ha="right")
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("Median net PnL per account-day ($)")
    ax.set_title("Do winners and losers behave differently across regimes?")
    ax.legend()
    U.savefig(fig, "07_cohorts_by_regime")
    return fig
