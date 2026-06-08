"""
Build & validate the trade <-> sentiment join, then emit an integrity report.

This is the spine of the project: if the join is wrong, every downstream number
is wrong. Run:  python src/prep_and_validate.py
"""
from __future__ import annotations

import json

import pandas as pd

import utils as U


def main() -> int:
    pd.set_option("display.width", 120)
    trades = U.load_trades()
    sent = U.load_sentiment()
    merged = U.build_merged(trades, sent)
    ad = U.account_day(merged)

    print("=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)
    print(f"trades rows ................ {len(trades):,}")
    print(f"unique accounts ............ {trades['Account'].nunique()}")
    print(f"unique coins ............... {trades['Coin'].nunique()}")
    print(f"ts_utc parsed (unix) ....... {trades['ts_utc'].notna().mean():.4%}")
    off = (trades["ts_ist"] - trades["ts_utc"].dt.tz_localize(None)).dropna()
    print(f"median(IST - UTC) offset ... {off.median()}  (expect 0 days 05:30:00)")
    print(f"Fee >= 0 fraction .......... {(trades['Fee'] >= 0).mean():.4%}")
    print(f"realized-PnL fill fraction . {trades['is_realized'].mean():.4%}")
    print(f"gross PnL .................. {trades['Closed PnL'].sum():,.2f}")
    print(f"fees ....................... {trades['Fee'].sum():,.2f}")
    print(f"NET PnL (gross - fees) ..... {trades['net_pnl'].sum():,.2f}")

    print("\n" + "=" * 72)
    print("JOIN INTEGRITY")
    print("=" * 72)
    n = len(merged)
    matched = int(merged["classification"].notna().sum())
    print(f"rows ....................... {n:,}")
    print(f"matched to a sentiment day . {matched:,}  ({matched / n:.4%})")
    print(f"unmatched .................. {n - matched:,}")
    print(f"trade dates (UTC) .......... {merged['date'].min().date()} -> "
          f"{merged['date'].max().date()}  ({merged['date'].nunique()} days)")
    print(f"sentiment dates ............ {sent['date'].min().date()} -> {sent['date'].max().date()}")
    if n - matched:
        bad = sorted(merged.loc[merged["classification"].isna(), "date"].dt.date.unique())
        print(f"unmatched dates (first 10) . {bad[:10]}")

    print("\n" + "=" * 72)
    print("PER-REGIME SAMPLE SIZES  (the council's make-or-break check)")
    print("=" * 72)
    by = pd.DataFrame({
        "trades": merged["classification"].value_counts().reindex(U.CLASS_ORDER),
        "account_days": ad["classification"].value_counts().reindex(U.CLASS_ORDER),
        "calendar_days": merged.groupby("classification")["date"].nunique().reindex(U.CLASS_ORDER),
    }).fillna(0).astype(int)
    print(by.to_string())
    print(f"\nbinary account-days: {ad['sentiment'].value_counts().to_dict()}")

    # ---- persist ----
    ad.to_csv(U.CSV_DIR / "account_day.csv", index=False)
    slim = ["Account", "Coin", "date", "date_ist", "Side", "Direction", "Size USD",
            "Start Position", "Closed PnL", "Fee", "net_pnl", "is_open", "is_close",
            "is_liquidation", "is_realized", "is_buy", "pos_side",
            "value", "classification", "sentiment"]
    merged[slim].to_csv(U.CSV_DIR / "merged_trades_sentiment.csv", index=False)

    report = {
        "trades": int(n),
        "accounts": int(trades["Account"].nunique()),
        "coins": int(trades["Coin"].nunique()),
        "match_rate": round(matched / n, 6),
        "date_min": str(merged["date"].min().date()),
        "date_max": str(merged["date"].max().date()),
        "calendar_days": int(merged["date"].nunique()),
        "net_pnl_total": round(float(trades["net_pnl"].sum()), 2),
        "per_regime_trades": by["trades"].to_dict(),
        "per_regime_account_days": by["account_days"].to_dict(),
    }
    (U.CSV_DIR / "integrity_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved: account_day.csv ({len(ad):,} rows), merged_trades_sentiment.csv "
          f"({len(merged):,} rows), integrity_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
