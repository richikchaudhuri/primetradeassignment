"""
Fetch the Crypto Fear & Greed Index -> csv_files/fear_greed_index.csv

Source priority:
  1. The official assignment file on Google Drive (exact-source match), if reachable.
  2. The alternative.me public API -- the canonical Fear & Greed Index (2018-present).

Both are normalized to a single schema:
    date           (datetime, normalized to midnight)
    value          (int 0-100; NaN if a source omits it)
    classification (str: Extreme Fear / Fear / Neutral / Greed / Extreme Greed)

Run:  python src/fetch_sentiment.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "csv_files" / "fear_greed_index.csv"
DRIVE_ID = "1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fng-fetch/1.0)"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse a column that may hold unix seconds or date strings (either day order)."""
    s = s.astype(str).str.strip()
    if s.str.fullmatch(r"\d{9,13}").all():
        return pd.to_datetime(s.astype("int64"), unit="s")
    d = pd.to_datetime(s, errors="coerce")
    if d.isna().mean() > 0.3:  # likely day-first (e.g. DD-MM-YYYY)
        d2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if d2.isna().mean() < d.isna().mean():
            d = d2
    return d


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary sentiment table onto the canonical schema."""
    lower = {c.lower().strip(): c for c in df.columns}
    date_col = next((lower[k] for k in lower if "date" in k or "time" in k), None)
    cls_col = next((lower[k] for k in lower if any(t in k for t in ("class", "label", "sentiment"))), None)
    val_col = next((lower[k] for k in lower if k in ("value", "index", "fng", "score")), None)
    if date_col is None or cls_col is None:
        raise ValueError(f"Unexpected sentiment columns: {list(df.columns)}")
    out = pd.DataFrame()
    out["date"] = _parse_dates(df[date_col]).dt.normalize()
    out["value"] = pd.to_numeric(df[val_col], errors="coerce") if val_col else pd.NA
    out["classification"] = df[cls_col].astype(str).str.strip()
    out = (out.dropna(subset=["date"])
              .drop_duplicates("date")
              .sort_values("date")
              .reset_index(drop=True))
    return out


def from_drive() -> pd.DataFrame:
    text = _get(DRIVE_URL).decode("utf-8", "replace")
    if "<html" in text[:1000].lower():
        raise ValueError("Drive returned an HTML page, not a CSV (restricted or needs a confirm token).")
    return _normalize(pd.read_csv(io.StringIO(text)))


def from_api() -> pd.DataFrame:
    data = json.loads(_get(API_URL))["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("Int64")
    df["classification"] = df["value_classification"].astype(str).str.strip()
    return (df[["date", "value", "classification"]]
              .drop_duplicates("date")
              .sort_values("date")
              .reset_index(drop=True))


def main() -> int:
    try:
        df, source = from_drive(), "official Google Drive file"
    except Exception as e:  # noqa: BLE001 - any failure should fall back gracefully
        print(f"[fetch] Drive source unavailable ({e});\n        falling back to alternative.me API.", file=sys.stderr)
        df, source = from_api(), "alternative.me API"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows from {source}")
    print(f"  -> {OUT}")
    print(f"  date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  classes    : {df['classification'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
