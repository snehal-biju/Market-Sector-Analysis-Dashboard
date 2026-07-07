"""Constituent-based sector fundamentals.

Sector ETFs don't expose clean aggregate valuation, so we pull key fundamentals
for each sector's representative holdings (from universe.py) via yfinance and
aggregate them (median) to a sector view. Results are cached per day — the first
load is slow (one Yahoo call per holding); later loads are instant.

yfinance's ``.info`` is best-effort and occasionally sparse; missing values come
back as NaN rather than failing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .data import CACHE_DIR
from .universe import US_SECTORS, all_constituents


def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _one(ticker: str) -> dict | None:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None

    # `trailingAnnualDividendYield` is consistently a fraction (0.019 = 1.9%); the
    # plain `dividendYield` field is returned on mixed scales across tickers, so
    # prefer the former and only fall back (with normalization) when it's absent.
    dy = _num(info.get("trailingAnnualDividendYield"))
    if not np.isfinite(dy):
        dy = _num(info.get("dividendYield"))
        if np.isfinite(dy) and dy > 1.0:
            dy /= 100.0
    if np.isfinite(dy) and dy > 0.25:     # a >25% yield is almost certainly bad data
        dy = np.nan

    return {
        "ticker": ticker,
        "trailingPE": _num(info.get("trailingPE")),
        "forwardPE": _num(info.get("forwardPE")),
        "priceToBook": _num(info.get("priceToBook")),
        "profitMargin": _num(info.get("profitMargins")),
        "dividendYield": dy,
    }


def constituent_fundamentals(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    """Per-holding fundamentals, cached to ``data/fundamentals_<date>.parquet``."""
    today = date.today().isoformat()
    path = CACHE_DIR / f"fundamentals_{today}.parquet"

    if use_cache and path.exists():
        try:
            cached = pd.read_parquet(path)
            if set(tickers).issubset(set(cached["ticker"])):
                return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
        except Exception:
            pass

    rows = [r for r in (_one(t) for t in tickers) if r]
    df = pd.DataFrame(rows)
    if use_cache and not df.empty:
        try:
            df.to_parquet(path)
        except Exception:
            pass
    return df


def sector_fundamentals(use_cache: bool = True) -> pd.DataFrame:
    """Median valuation per US sector, aggregated over its representative holdings."""
    funds = constituent_fundamentals(all_constituents(), use_cache)
    if funds.empty:
        return pd.DataFrame()

    rows = []
    for e in US_SECTORS:
        sub = funds[funds["ticker"].isin(e.constituents)]
        if sub.empty:
            continue
        rows.append({
            "ticker": e.ticker,
            "sector": e.name,
            "holdings_with_data": int(sub["trailingPE"].notna().sum()),
            "trailingPE": float(sub["trailingPE"].median()),
            "forwardPE": float(sub["forwardPE"].median()),
            "priceToBook": float(sub["priceToBook"].median()),
            "profitMargin": float(sub["profitMargin"].median()),
            "dividendYield": float(sub["dividendYield"].median()),
            "holdings": ", ".join(e.constituents),
        })
    return pd.DataFrame(rows)
