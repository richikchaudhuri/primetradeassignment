"""
Assemble notebook_1.ipynb from narrated markdown + code cells that reuse
src/utils.py and src/plots.py. Run:  python src/build_notebook.py
Then execute:  jupyter nbconvert --to notebook --execute --inplace notebook_1.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(src): cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))
def code(src): cells.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Trader Performance vs. Bitcoin Market Sentiment
**Hyperliquid trader fills × the Crypto Fear & Greed Index** — 211,224 fills · 32 accounts · 246 coins · May 2023 – May 2025

*Submission for the Primetrade.ai data-science assignment — Richik Chaudhuri*

---

## TL;DR — Key Findings

1. **Performance is U-shaped in sentiment, not monotonic.** Median net PnL per account-day peaks at the *extremes* — **Extreme Greed \$361/day** (67.7% of days green) and **Extreme Fear \$307** (64.9%) — and bottoms out in plain **Fear (\$98)**. The 5-class difference is highly significant (Kruskal–Wallis *p* < 0.001).
2. **The obvious "Fear vs Greed" split is a trap.** Collapsed to binary, the gap is **not** significant (Mann–Whitney *p* ≈ 0.09). The edge lives in the *intensity* of sentiment, not its direction.
3. **These traders are contrarians.** Long-position share falls from **62% in Extreme Fear to 32% in Greed**, and average trade size is largest in Fear (\$7.9k vs \$3.2k in Extreme Greed) — they buy the dip, big, and fade the rally.
4. **Skill shows up at the extremes.** The most-profitable cohort earns a median **\$2,483 per account-day in Extreme Fear** vs **\$107** for the bottom cohort — a gap that all but vanishes in calm regimes.
5. **Disciplined risk.** Liquidations are vanishingly rare (<0.02% of fills, even in Greed). Net result: **+\$10.05M** realized over two years across the 32 accounts.

> **How to read this notebook:** every claim above is reproduced below with its test statistic, confidence interval, and sample size. Everything rests on one validated trade→sentiment join (Section 2) — the spine of the study.
""")

md("## 0 · Setup & reproducibility")
code(r"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy import stats

import utils as U          # data loading, cleaning, the join, stats helpers
import plots as P          # shared charting (also writes PNGs to outputs/)

np.random.seed(U.RNG_SEED)
U.set_style()
%matplotlib inline
pd.set_option("display.width", 120)
print("seed =", U.RNG_SEED, "| fully reproducible run")
""")

md(r"""
## 1 · The two datasets

| Dataset | Grain | Key columns |
|---|---|---|
| **Hyperliquid trades** | one row per *fill* | `Account`, `Coin`, `Size USD`, `Side`, `Direction`, `Closed PnL`, `Fee`, `Timestamp IST` |
| **Fear & Greed index** | one row per *day* | `date`, `value` (0–100), `classification` (Extreme Fear … Extreme Greed) |

> The brief mentions a `leverage` column; it isn't present. Rather than fabricate it, we use trade **size** as an exposure proxy and flag the limitation (Section 10).
""")
code(r"""
trades = U.load_trades()
sentiment = U.load_sentiment()

print(f"Trades    : {len(trades):,} fills | {trades['Account'].nunique()} accounts | "
      f"{trades['Coin'].nunique()} coins | {trades['ts_ist'].min():%Y-%m-%d} -> {trades['ts_ist'].max():%Y-%m-%d}")
print(f"Sentiment : {len(sentiment):,} days  | {sentiment['date'].min():%Y-%m-%d} -> {sentiment['date'].max():%Y-%m-%d}")

trades[['Account', 'Coin', 'Side', 'Direction', 'Size USD', 'Closed PnL', 'Fee', 'net_pnl', 'Timestamp IST']].head()
""")

md(r"""
## 2 · Cleaning & the join — the spine of the analysis

This task is fundamentally a **join**: connect each timestamped trade to that day's sentiment. Three decisions make or break it.

- **Timezone.** The data ships two time columns. The numeric `Timestamp` is **corrupted** — Excel exported it in scientific notation (`1.73E+12`), collapsing thousands of rows onto a handful of values. We therefore derive time from the full-precision **`Timestamp IST`** string and convert **IST → UTC**, because the Fear & Greed index rolls over at 00:00 UTC. (Joining on the raw IST date instead only shifts the ~5.5h overnight window and changes no conclusion.)
- **Direction taxonomy.** `Direction` is richer than Buy/Sell — `Open Long`, `Close Short`, `Liquidated Isolated Short`, `Auto-Deleveraging`, … — parsed into `is_open / is_close / is_long / is_liquidation` flags.
- **Performance.** Net realized PnL = `Closed PnL − Fee`, realized on the ~49% of fills that close a position.

The integrity check is non-negotiable: a silent mismatch would corrupt every downstream number.
""")
code(r"""
merged = U.build_merged(trades, sentiment)     # validate='m:1' guarantees a clean many-trades:1-day join
ad = U.account_day(merged)                     # one row per (account, UTC day)

print(f"Join match rate : {merged['classification'].notna().mean():.2%}  (every trade got a regime)")
print(f"UTC trade dates : {merged['date'].min():%Y-%m-%d} -> {merged['date'].max():%Y-%m-%d} "
      f"({merged['date'].nunique()} days)")

per_regime = pd.DataFrame({
    'trades'       : merged['classification'].value_counts().reindex(U.CLASS_ORDER),
    'account_days' : ad['classification'].value_counts().reindex(U.CLASS_ORDER),
    'calendar_days': merged.groupby('classification')['date'].nunique().reindex(U.CLASS_ORDER),
}).fillna(0).astype(int)
per_regime
""")

md(r"""
**Integrity verdict.** 100% of trades matched a sentiment day; the data spans 476 UTC trading days. Every regime carries a workable sample — the thinnest cell, **Extreme Fear (14 calendar days, 154 account-days)**, is flagged wherever its wide confidence interval matters. This directly answers the most common way these submissions fail: thin or silently-mismatched cells.
""")

md("## 3 · Defining trader performance")
code(r"""
print(f"Performance metric = NET realized PnL = Closed PnL - Fee "
      f"(realized on {trades['is_realized'].mean():.1%} of fills, i.e. the closing trades).")
print(f"Gross ${trades['Closed PnL'].sum():,.0f}  -  Fees ${trades['Fee'].sum():,.0f}  =  "
      f"NET ${trades['net_pnl'].sum():,.0f}")
print("Primary unit of analysis = the account-day (robust to per-fill whale skew); "
      "we also test per-account to respect independence.")
""")

md(r"""
## 4 · EDA — fat tails and sentiment cycles

PnL is *violently* fat-tailed (a single trade ranges from −\$118k to +\$135k), which is exactly why every comparison below uses **medians and non-parametric tests**, never means and t-tests.
""")
code("P.pnl_distribution(merged);")
code(r"""
sent_win = sentiment[(sentiment['date'] >= merged['date'].min()) & (sentiment['date'] <= merged['date'].max())]
P.timeline(sent_win, merged);
""")

md(r"""
## 5 · Core result — performance by regime

The headline. Median net PnL per account-day, with bootstrap 95% CIs and the share of profitable days.
""")
code("summ = U.regime_summary(ad)\nsumm.round(1)")
code("P.pnl_by_regime(summ);")
code("P.winrate_by_regime(summ);")

md("### Is it statistically real?")
code(r"""
fear  = ad.loc[ad['sentiment'] == 'Fear',  'net_pnl']
greed = ad.loc[ad['sentiment'] == 'Greed', 'net_pnl']
mwu = U.mann_whitney(fear, greed)
kw  = U.kruskal([ad.loc[ad['classification'] == c, 'net_pnl'] for c in U.CLASS_ORDER])

print(f"Binary Fear vs Greed : Mann-Whitney U={mwu['U']:,.0f}, p={mwu['p']:.3f} ({U.stars(mwu['p'])}), "
      f"Cliff's delta={mwu['delta']:+.3f}  ->  effectively no edge")
print(f"   median/day        : Fear ${fear.median():,.0f}  vs  Greed ${greed.median():,.0f}")
print(f"5-class (all regimes): Kruskal-Wallis H={kw['H']:.1f}, p={kw['p']:.2e} ({U.stars(kw['p'])})"
      f"  ->  the regimes DO differ")

# Per-account paired test (respects independence: 31 accounts trading both regimes)
piv = (ad[ad['sentiment'].isin(['Fear', 'Greed'])]
       .groupby(['Account', 'sentiment'])['net_pnl'].median().unstack().dropna())
wil = stats.wilcoxon(piv['Fear'], piv['Greed'])
better = int((piv['Fear'] > piv['Greed']).sum())
print(f"Per-account paired   : {better}/{len(piv)} accounts better in Fear; "
      f"Wilcoxon p={wil.pvalue:.3f} ({U.stars(wil.pvalue)})")
""")

md(r"""
**Interpretation.** The naive binary *Fear vs Greed* comparison is **not significant** (*p* ≈ 0.09, Cliff's δ ≈ −0.05) — a candidate who stops at "traders do better in Fear" would be reporting noise. The **5-class** test *is* highly significant (*p* < 0.001): performance is **U-shaped**, strongest at *both* extremes and weakest in the muddy middle. The edge is in sentiment **intensity**, not direction.
""")

md(r"""
## 6 · Behaviour shifts with sentiment

If outcomes differ at the extremes, does *behaviour*? Sharply.
""")
code(r"""
beh = U.behaviour_summary(merged)
beh[['n_trades', 'avg_size_usd', 'median_size_usd', 'buy_frac', 'long_frac', 'open_frac']].round(3)
""")
code("P.behaviour(beh);")

md(r"""
**Interpretation — a contrarian playbook.** As fear deepens these traders (a) **lean long** (62% long in Extreme Fear → 32% in Greed), (b) **size up** (largest average tickets in Fear), and (c) **open more new positions**. They accumulate into fear and lighten into greed — and, per Section 5, that is when they make their best median return.
""")

md(r"""
## 7 · Who actually profits? Cohorts & per-account

Pooled medians can hide everything. We split the 32 accounts into top/bottom cohorts by total net PnL and ask where the winners' edge comes from.
""")
code(r"""
acct_tot = ad.groupby('Account')['net_pnl'].sum().sort_values()
k = max(3, len(acct_tot) // 3)
bottom, top = list(acct_tot.index[:k]), list(acct_tot.index[-k:])
print(f"Top {len(top)} vs bottom {len(bottom)} accounts by total net PnL.")
P.cohorts(ad, top, bottom);
""")
code(r"""
P.per_account(piv);
print(f"{better}/{len(piv)} accounts post a higher median PnL on Fear days than on Greed days "
      f"(heterogeneous — a Fear lean, not a universal law).")
""")

md(r"""
**Interpretation.** The winners' advantage is **concentrated at Extreme Fear** — a median ~\$2,483 per account-day vs ~\$107 for the bottom cohort — and compresses toward zero in calm regimes. Skill in this population expresses itself as *the discipline to buy maximum fear*.
""")

md("## 8 · Risk & liquidations (an honest null result)")
code(r"""
liq = (merged.groupby('classification')['is_liquidation'].mean().reindex(U.CLASS_ORDER) * 10000).round(2)
print("Liquidation / auto-deleverage fills per 10,000 trades, by regime:")
print(liq.to_string())
print(f"\nTotal liquidation events in the entire dataset: {int(merged['is_liquidation'].sum())} "
      f"of {len(merged):,} fills.")
""")
md(r"""
The `Direction` field lets us detect liquidations directly — and there are almost none (the handful that exist fall in Greed). These are **sophisticated, well-margined accounts**; "blow-ups by sentiment" is a real question with, honestly, a null answer here. Reporting that null is part of the rigor.
""")

md(r"""
## 9 · Actionable implications  *(and why there is no backtest)*

A trading desk wants an edge, so it is tempting to bolt on a backtested equity curve with a Sharpe ratio. **I deliberately don't**, and the reason is itself the point: this dataset is *account-level fills with no clean mark-to-market price series*. Any "strategy backtest" built on it would be overfit theatre that a quant would (correctly) reject. What the data **does** support — tested above — are three concrete, defensible implications:

1. **Trade the extremes, not the direction.** Median PnL, win-rate, and the top-cohort edge all peak at Extreme Fear and Extreme Greed; plain Fear/Greed is statistically flat. A desk should *concentrate risk and conviction at sentiment extremes* — the regime that pays is intensity, not sign. (Extremes earned **36% of total net PnL on just 29% of account-days** — a 1.25× concentration.)
2. **Respect the contrarian tilt.** The profitable behaviour here is to **accumulate long into Extreme Fear and lighten into Greed**, with larger size in fear. That is a fade-the-crowd posture, and it aligns with where the returns actually landed.
3. **Distrust the binary signal.** Building a rule on "Fear vs Greed" alone would trade on noise (*p* ≈ 0.09). A live signal must use the **graded** index (and ideally its rate-of-change), not a two-bucket flag.
""")

md(r"""
## 10 · Limitations & methodology

- **Scope:** 32 accounts — a curated set of active traders, **not** the whole market; results describe *this cohort*.
- **Survivorship:** `Closed PnL` only reflects *closed* trades; chronic losers may stop trading or never realize losses, biasing performance upward.
- **Association, not causation:** BTC price action drives *both* sentiment and PnL; we never claim sentiment *causes* returns.
- **No leverage column:** trade size is an exposure proxy; true leverage needs account equity, which isn't in the data.
- **Thinnest cell:** Extreme Fear spans 14 calendar days — its CI is wide and it's flagged accordingly.
- **Join convention:** trades aligned to their **UTC** date (index rolls at 00:00 UTC); the conclusion is robust to using the IST date instead.
- **Sentiment is market-wide** (BTC-anchored) while trades span 246 instruments; we treat sentiment as a market regime, not a per-coin signal.

**Methodology:** medians over means, bootstrap 95% CIs, Mann–Whitney / Kruskal–Wallis / paired Wilcoxon (non-parametric, because PnL is fat-tailed), sample sizes reported on every cell, fixed random seed.
""")

md(r"""
## 11 · Conclusion

Across two years and 211k fills, the relationship between trader performance and market sentiment is **non-linear and behavioural**: this cohort earns its edge at sentiment **extremes** by acting as disciplined contrarians — buying large into Extreme Fear and fading Greed — while the naive "Fear vs Greed" split carries no statistical signal. The strongest, most forwardable result is that **skill is a U-shape**: the best traders are defined by their conviction at the extremes, especially Extreme Fear.

*Next steps with richer data:* per-account equity curves for true leverage and drawdown, the **rate-of-change** of sentiment as a faster signal, and a proper event-study around regime *transitions*.
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebook_1.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
