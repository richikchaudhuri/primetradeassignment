"""
Reusable helpers for the *Trader Performance vs. Market Sentiment* analysis.

Design notes
------------
* Join key = the trade's **UTC calendar date**, derived from the machine-precise
  unix ``Timestamp`` (ms). The Crypto Fear & Greed Index rolls over at 00:00 UTC,
  so aligning trades to their UTC day is the correct, defensible convention.
  (``Timestamp IST`` is kept for display and a robustness check.)
* Performance = **net realized PnL = ``Closed PnL`` - ``Fee``**.
* PnL is violently fat-tailed -> we report **medians** and use **non-parametric**
  tests (Mann-Whitney, Kruskal-Wallis) with **bootstrap** CIs everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
RNG_SEED = 42
ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv_files"
OUT_DIR = ROOT / "outputs"

NUMERIC_COLS = ["Execution Price", "Size Tokens", "Size USD",
                "Start Position", "Closed PnL", "Fee"]

# Ordered sentiment regimes (cold -> hot)
CLASS_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
# Collapse 5-class -> binary (Neutral kept separate, excluded from Fear-vs-Greed tests)
BINARY_MAP = {"Extreme Fear": "Fear", "Fear": "Fear", "Neutral": "Neutral",
              "Greed": "Greed", "Extreme Greed": "Greed"}

# Diverging palette: fear = red, greed = green
SENT_COLORS = {"Extreme Fear": "#8B0000", "Fear": "#E06666", "Neutral": "#B0B0B0",
               "Greed": "#93C47D", "Extreme Greed": "#274E13",
               "Fear_bin": "#E06666", "Greed_bin": "#6AA84F"}


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_trades(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the Hyperliquid trade-fill data."""
    path = Path(path) if path else CSV_DIR / "historical_data.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- time ---
    # The numeric `Timestamp` column is CORRUPTED in the source CSV: Excel exported
    # it in scientific notation (e.g. "1.73E+12"), destroying precision so it
    # collapses to a few distinct values. We therefore derive all time from the
    # full-precision `Timestamp IST` string and convert IST -> UTC for the join
    # (the Fear & Greed index rolls over at 00:00 UTC).
    df["ts_ist"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["ts_utc"] = (df["ts_ist"]
                    .dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
                    .dt.tz_convert("UTC"))
    df["date"] = df["ts_utc"].dt.tz_localize(None).dt.normalize()   # UTC calendar day (join key)
    df["date_ist"] = df["ts_ist"].dt.normalize()                    # IST day (robustness check)

    # --- performance ---
    df["net_pnl"] = df["Closed PnL"] - df["Fee"]

    # --- behavioural flags parsed from Direction ---
    d = df["Direction"].astype(str)
    df["is_open"] = d.str.contains("Open", case=False, na=False)
    df["is_close"] = d.str.contains("Close", case=False, na=False)
    df["is_flip"] = d.str.contains(">", na=False)
    df["is_liquidation"] = d.str.contains(r"Liquidat|Deleverag", case=False, regex=True, na=False)
    df["is_realized"] = df["Closed PnL"].abs() > 0   # PnL actually realised on this fill
    df["is_buy"] = df["Side"].astype(str).str.upper().eq("BUY")
    df["pos_side"] = np.select(
        [d.str.contains("Long", case=False, na=False), d.str.contains("Short", case=False, na=False)],
        ["Long", "Short"], default="Other")
    return df


def load_sentiment(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Fear & Greed index with ordered 5-class + binary collapse."""
    path = Path(path) if path else CSV_DIR / "fear_greed_index.csv"
    s = pd.read_csv(path, parse_dates=["date"])
    s["date"] = s["date"].dt.normalize()
    s["classification"] = s["classification"].astype(str).str.strip()
    s["class5"] = pd.Categorical(s["classification"], categories=CLASS_ORDER, ordered=True)
    s["sentiment"] = s["classification"].map(BINARY_MAP)
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s


def build_merged(trades: pd.DataFrame | None = None,
                 sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """LEFT-join trades -> sentiment on the UTC date (many trades : 1 sentiment day)."""
    trades = load_trades() if trades is None else trades
    sentiment = load_sentiment() if sentiment is None else sentiment
    merged = trades.merge(
        sentiment[["date", "value", "classification", "class5", "sentiment"]],
        on="date", how="left", validate="m:1")
    return merged


def account_day(merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fills to one row per (account, UTC day) with that day's regime."""
    g = (merged.groupby(["Account", "date"], observed=True)
         .agg(net_pnl=("net_pnl", "sum"),
              gross_pnl=("Closed PnL", "sum"),
              fees=("Fee", "sum"),
              volume_usd=("Size USD", "sum"),
              n_fills=("net_pnl", "size"),
              n_realized=("is_realized", "sum"),
              n_liq=("is_liquidation", "sum"),
              buy_frac=("is_buy", "mean"),
              classification=("classification", "first"),
              value=("value", "first"))
         .reset_index())
    g["class5"] = pd.Categorical(g["classification"], categories=CLASS_ORDER, ordered=True)
    g["sentiment"] = g["classification"].map(BINARY_MAP)
    g["won_day"] = g["net_pnl"] > 0
    return g


# --------------------------------------------------------------------------- #
# Statistics (non-parametric, bootstrap)
# --------------------------------------------------------------------------- #
def bootstrap_ci(x, func=np.median, n=2000, alpha=0.05, seed=RNG_SEED):
    """Percentile bootstrap CI for a statistic. Returns (point, lo, hi)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = x.size
    boot = np.empty(n)
    for i in range(n):
        boot[i] = func(x[rng.integers(0, m, m)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(func(x)), float(lo), float(hi)


def mann_whitney(a, b):
    """Mann-Whitney U (two-sided) + Cliff's delta effect size (derived from U)."""
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return dict(U=np.nan, p=np.nan, delta=np.nan, n1=len(a), n2=len(b))
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2.0 * U / (len(a) * len(b)) - 1.0          # Cliff's delta in [-1, 1]
    return dict(U=float(U), p=float(p), delta=float(delta), n1=len(a), n2=len(b))


def kruskal(groups):
    """Kruskal-Wallis across >=2 non-trivial groups."""
    groups = [pd.Series(g).dropna().to_numpy() for g in groups]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return dict(H=np.nan, p=np.nan, k=len(groups))
    H, p = stats.kruskal(*groups)
    return dict(H=float(H), p=float(p), k=len(groups))


def win_rate(pnl) -> float:
    pnl = pd.Series(pnl).dropna()
    return float((pnl > 0).mean()) if len(pnl) else np.nan


def stars(p) -> str:
    """Significance markers for tables."""
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def set_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 15, "axes.labelsize": 12,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "font.family": "DejaVu Sans", "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def savefig(fig, name: str):
    """Save a figure to outputs/ as PNG and return its path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(p)
    return p


def class_palette(binary: bool = False) -> dict:
    if binary:
        return {"Fear": SENT_COLORS["Fear_bin"], "Greed": SENT_COLORS["Greed_bin"],
                "Neutral": SENT_COLORS["Neutral"]}
    return {k: SENT_COLORS[k] for k in CLASS_ORDER}


# --------------------------------------------------------------------------- #
# Regime / behaviour summaries (shared by the batch script and the notebook)
# --------------------------------------------------------------------------- #
def regime_summary(ad: pd.DataFrame) -> pd.DataFrame:
    """Per-regime account-day net-PnL: median + bootstrap CI, mean, win-rate."""
    rows = []
    for cls in CLASS_ORDER:
        g = ad.loc[ad["classification"] == cls, "net_pnl"]
        med, lo, hi = bootstrap_ci(g, np.median)
        rows.append(dict(regime=cls, n_acct_days=int(len(g)),
                         median_net_pnl=med, ci_lo=lo, ci_hi=hi,
                         mean_net_pnl=float(g.mean()) if len(g) else np.nan,
                         win_rate=win_rate(g)))
    return pd.DataFrame(rows).set_index("regime")


def behaviour_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-regime fill-level behaviour: trade size, buy/long/open shares, liq rate."""
    rows = []
    for cls in CLASS_ORDER:
        m = merged[merged["classification"] == cls]
        rows.append(dict(regime=cls, n_trades=int(len(m)),
                         avg_size_usd=float(m["Size USD"].mean()),
                         median_size_usd=float(m["Size USD"].median()),
                         buy_frac=float(m["is_buy"].mean()),
                         long_frac=float((m["pos_side"] == "Long").mean()),
                         open_frac=float(m["is_open"].mean()),
                         liq_rate=float(m["is_liquidation"].mean())))
    return pd.DataFrame(rows).set_index("regime")
