"""
Batch analysis: performance vs sentiment + differentiators. Prints a full numeric
report, regenerates every chart in outputs/, and writes outputs/stats_summary.json.

Charts live in plots.py so the notebook reuses the exact same code.
Run:  python src/analysis.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless batch rendering (notebook uses inline instead)

import utils as U
import plots as P

U.set_style()


def main() -> int:
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    merged = U.build_merged()
    ad = U.account_day(merged)
    summ = U.regime_summary(ad)
    beh = U.behaviour_summary(merged)

    print("=" * 78); print("REGIME SUMMARY  (unit = account-day net PnL)"); print("=" * 78)
    print(summ.to_string())
    print("\n" + "=" * 78); print("BEHAVIOUR SUMMARY  (unit = fill)"); print("=" * 78)
    print(beh.to_string())

    # significance -------------------------------------------------------------
    fear = ad.loc[ad["sentiment"] == "Fear", "net_pnl"]
    greed = ad.loc[ad["sentiment"] == "Greed", "net_pnl"]
    mwu = U.mann_whitney(fear, greed)
    kw = U.kruskal([ad.loc[ad["classification"] == c, "net_pnl"] for c in U.CLASS_ORDER])
    print("\n" + "=" * 78); print("SIGNIFICANCE TESTS"); print("=" * 78)
    print(f"Pooled account-day Fear vs Greed -> Mann-Whitney U={mwu['U']:,.0f} "
          f"p={mwu['p']:.3g} {U.stars(mwu['p'])} Cliff_delta={mwu['delta']:.3f} "
          f"(n_fear={mwu['n1']}, n_greed={mwu['n2']})")
    print(f"  median net PnL/day: Fear=${fear.median():,.2f}  Greed=${greed.median():,.2f}")
    print(f"5-class account-day -> Kruskal-Wallis H={kw['H']:.1f} p={kw['p']:.3g} {U.stars(kw['p'])}")

    piv = (ad[ad["sentiment"].isin(["Fear", "Greed"])]
           .groupby(["Account", "sentiment"])["net_pnl"].median().unstack().dropna())
    wil = stats.wilcoxon(piv["Fear"], piv["Greed"])
    better_fear = int((piv["Fear"] > piv["Greed"]).sum())
    print(f"\nPer-account PAIRED (n={len(piv)} accounts trading both regimes):")
    print(f"  more profitable in Fear: {better_fear}/{len(piv)} ({better_fear/len(piv):.1%})")
    print(f"  Wilcoxon W={wil.statistic:.1f} p={wil.pvalue:.3g} {U.stars(wil.pvalue)}")

    # cohorts ------------------------------------------------------------------
    acct_tot = ad.groupby("Account")["net_pnl"].sum().sort_values()
    k = max(3, len(acct_tot) // 3)
    bottom, top = list(acct_tot.index[:k]), list(acct_tot.index[-k:])
    print(f"\nCohorts: top {len(top)} vs bottom {len(bottom)} accounts by total net PnL")

    # liquidations -------------------------------------------------------------
    liq = merged.groupby("classification")["is_liquidation"].mean().reindex(U.CLASS_ORDER) * 10000
    print("\nLiquidation fills per 10,000 by regime:"); print(liq.round(2).to_string())

    # PnL concentration at extremes (for the implications section) -------------
    ad["extreme"] = ad["classification"].isin(["Extreme Fear", "Extreme Greed"])
    pnl_extreme = ad.loc[ad["extreme"], "net_pnl"].sum()
    share_pnl = pnl_extreme / ad["net_pnl"].sum()
    share_days = ad["extreme"].mean()
    print(f"\nExtremes earn {share_pnl:.1%} of net PnL on {share_days:.1%} of account-days "
          f"(concentration ratio {share_pnl/share_days:.2f}x)")

    # charts -------------------------------------------------------------------
    sent = U.load_sentiment()
    sent = sent[(sent["date"] >= merged["date"].min()) & (sent["date"] <= merged["date"].max())]
    P.timeline(sent, merged); P.pnl_distribution(merged)
    P.pnl_by_regime(summ); P.winrate_by_regime(summ); P.behaviour(beh)
    P.per_account(piv); P.cohorts(ad, top, bottom)
    print("\nsaved 7 charts to", U.OUT_DIR)

    # persist key numbers ------------------------------------------------------
    results = dict(
        regime_summary=summ.reset_index().round(2).to_dict(orient="records"),
        behaviour=beh.reset_index().round(4).to_dict(orient="records"),
        fear_vs_greed_mwu=mwu, kruskal_5class=kw,
        per_account_paired=dict(n=int(len(piv)), better_in_fear=better_fear,
                                wilcoxon_W=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
                                median_fear=float(fear.median()), median_greed=float(greed.median())),
        liquidation_per_10k=liq.round(3).to_dict(),
        extremes=dict(share_pnl=round(float(share_pnl), 4), share_days=round(float(share_days), 4)),
        net_pnl_total=float(merged["net_pnl"].sum()),
    )
    (U.OUT_DIR / "stats_summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("saved outputs/stats_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
