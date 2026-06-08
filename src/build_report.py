"""
Compose a polished 3-page research-note PDF (report/ds_report.pdf) from the
committed charts + narrative. Pure matplotlib (no extra dependencies).

Run:  python src/build_report.py   (run analysis.py first so outputs/ exists)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["text.parse_math"] = False   # treat '$' as a literal dollar sign
plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INK, GREY, PAPER, ACCENT = "#222222", "#7B7B7B", "#F8F8F8", "#8B0000"
A4 = (8.27, 11.69)


def header(fig, title, kicker):
    ax = fig.add_axes([0, 0.9, 1, 0.1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=INK))
    ax.add_patch(plt.Rectangle((0, 0), 0.015, 1, color=ACCENT))
    ax.text(0.06, 0.60, title, color="white", fontsize=17, fontweight="bold", va="center")
    ax.text(0.06, 0.25, kicker, color="#BBBBBB", fontsize=9.5, va="center")


def footer(fig, page):
    ax = fig.add_axes([0, 0, 1, 0.04]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.5, "Trader Performance vs. Market Sentiment  ·  Richik Chaudhuri",
            color=GREY, fontsize=8, va="center")
    ax.text(0.94, 0.5, f"{page}", color=GREY, fontsize=8, va="center", ha="right")


def image(fig, rect, name, caption=None):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.imshow(mpimg.imread(OUT / name))
    if caption:
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color=GREY, wrap=True)


def paragraph(fig, x, y, text, width=96, size=9.5, color=INK, leading=0.022, bold_lead=None):
    for i, line in enumerate(textwrap.wrap(text, width)):
        fig.text(x, y - i * leading, line, fontsize=size, color=color, va="top")
    return y - (len(textwrap.wrap(text, width))) * leading


def main() -> int:
    pdf_path = ROOT / "report" / "ds_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # ---------- Page 1: findings ----------
        fig = plt.figure(figsize=A4)
        header(fig, "Trader Performance vs. Market Sentiment",
               "Hyperliquid trades x Crypto Fear & Greed Index  ·  211,224 fills · 32 accounts · 2023–2025")
        fig.text(0.06, 0.86, "KEY FINDINGS", fontsize=12, fontweight="bold", color=ACCENT, va="top")
        findings = [
            "1.  Performance is U-shaped in sentiment, not monotonic. Median net PnL per account-day peaks "
            "at the extremes — Extreme Greed $361/day (67.7% of days profitable) and Extreme Fear $307 "
            "(64.9%) — and troughs at plain Fear ($98). The 5-class difference is highly significant "
            "(Kruskal-Wallis p < 0.001).",
            "2.  The obvious 'Fear vs Greed' split is a trap. Collapsed to binary it is NOT significant "
            "(Mann-Whitney p = 0.09): the edge is the intensity of sentiment, not its direction.",
            "3.  These traders are contrarians. Long-position share falls from 62% in Extreme Fear to 32% "
            "in Greed, and average trade size is largest in Fear ($7.9k vs $3.2k in Extreme Greed) — they "
            "buy the dip, big, and fade the rally.",
            "4.  Skill shows up at the extremes. The most-profitable cohort earns a median $2,483 per "
            "account-day in Extreme Fear versus $107 for the bottom cohort — a gap that all but vanishes "
            "in calm regimes.",
            "5.  Disciplined risk. Liquidations are vanishingly rare (<0.02% of fills, even in Greed). "
            "Net result: +$10.05M realized over two years across the 32 accounts.",
        ]
        y = 0.825
        for f in findings:
            y = paragraph(fig, 0.06, y, f, width=104, size=9.3, leading=0.0205) - 0.012
        image(fig, [0.08, 0.05, 0.84, 0.30], "03_net_pnl_by_regime.png")
        footer(fig, 1)
        pdf.savefig(fig); plt.close(fig)

        # ---------- Page 2: evidence ----------
        fig = plt.figure(figsize=A4)
        header(fig, "The evidence", "Behaviour, win-rate, and who actually profits")
        image(fig, [0.05, 0.50, 0.43, 0.36], "04_winrate_by_regime.png",
              "Profitable-day rate — highest at both extremes.")
        image(fig, [0.52, 0.50, 0.43, 0.36], "07_cohorts_by_regime.png",
              "Winners' edge is concentrated at Extreme Fear.")
        image(fig, [0.05, 0.09, 0.43, 0.36], "05_behaviour_by_regime.png",
              "Bigger, longer, more-opening trades as fear deepens.")
        image(fig, [0.52, 0.09, 0.43, 0.36], "06_per_account_fear_vs_greed.png",
              "Per-account: a Fear lean (17/31), not a universal law.")
        footer(fig, 2)
        pdf.savefig(fig); plt.close(fig)

        # ---------- Page 3: implications, method, limits ----------
        fig = plt.figure(figsize=A4)
        header(fig, "Implications, method & limitations", "What a desk can act on — and what it can't")
        y = 0.85
        fig.text(0.06, y, "ACTIONABLE IMPLICATIONS", fontsize=11, fontweight="bold", color=ACCENT); y -= 0.03
        for t in [
            "Trade the extremes, not the direction. Median PnL, win-rate and the top-cohort edge all peak "
            "at Extreme Fear and Extreme Greed; plain Fear/Greed is statistically flat. Extremes earned 36% "
            "of total net PnL on just 29% of account-days (1.25x concentration).",
            "Respect the contrarian tilt: accumulate long into Extreme Fear and lighten into Greed, with "
            "larger size in fear — the posture that actually captured the returns here.",
            "Distrust the binary signal: a rule on 'Fear vs Greed' alone trades on noise (p = 0.09). A live "
            "signal must use the graded index (and its rate-of-change), not a two-bucket flag.",
        ]:
            y = paragraph(fig, 0.06, y, "•  " + t, width=104, size=9.3, leading=0.0205) - 0.012

        y -= 0.01
        fig.text(0.06, y, "WHY THERE IS NO BACKTEST", fontsize=11, fontweight="bold", color=ACCENT); y -= 0.03
        y = paragraph(fig, 0.06, y,
                      "The data is account-level fills with no clean mark-to-market price series, so a "
                      "backtested equity curve / Sharpe ratio would be overfit theatre a quant would reject. "
                      "The implications above are what the data genuinely supports — and saying so is the rigor.",
                      width=104, size=9.3, leading=0.0205) - 0.02

        fig.text(0.06, y, "METHOD", fontsize=11, fontweight="bold", color=ACCENT); y -= 0.03
        y = paragraph(fig, 0.06, y,
                      "Join: each trade -> that day's regime, aligned on UTC date (the numeric Timestamp is "
                      "Excel-corrupted, so time comes from Timestamp IST). Performance = net realized PnL = "
                      "Closed PnL - Fee. PnL is fat-tailed, so medians + non-parametric tests (Mann-Whitney, "
                      "Kruskal-Wallis, paired Wilcoxon) with bootstrap 95% CIs and sample sizes on every cell. "
                      "Framed as association, not causation.",
                      width=104, size=9.3, leading=0.0205) - 0.02

        fig.text(0.06, y, "LIMITATIONS", fontsize=11, fontweight="bold", color=ACCENT); y -= 0.03
        paragraph(fig, 0.06, y,
                  "32 accounts (a cohort, not the market) · closed-PnL survivorship · no true leverage "
                  "(size proxy) · Extreme Fear is the thinnest cell (14 calendar days) · sentiment is "
                  "market-wide while trades span 246 coins.",
                  width=104, size=9.3, leading=0.0205)
        footer(fig, 3)
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
