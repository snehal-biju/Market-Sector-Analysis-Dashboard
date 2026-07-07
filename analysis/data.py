"""End-of-day price data from Yahoo Finance, with a per-day on-disk cache.

Cache key is (sorted tickers, period, today's date), so within a day we hit the
network once and reuse a local parquet file afterwards. Delete the ``data/``
folder (or use the app's refresh button) to force a re-pull. This is strictly
end-of-day data — there is no intraday or real-time path by design.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(tickers: list[str], period: str, today: str) -> Path:
    key = f"{period}|{today}|{','.join(sorted(tickers))}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"prices_{period}_{today}_{digest}.parquet"


def _split_panel(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Turn a yfinance download frame into {ticker: OHLCV DataFrame}.

    Robust to the ticker sitting on either level of a MultiIndex, or to a flat
    single-ticker frame.
    """
    out: dict[str, pd.DataFrame] = {}
    if not isinstance(raw.columns, pd.MultiIndex):
        if all(c in raw.columns for c in OHLCV) and len(tickers) == 1:
            df = raw[OHLCV].dropna(how="all")
            if len(df) > 60:
                out[tickers[0]] = df
        return out

    lvl0 = set(raw.columns.get_level_values(0))
    field_level = 0 if {"Open", "High", "Low", "Close"}.issubset(lvl0) else 1
    ticker_level = 1 - field_level
    available = set(raw.columns.get_level_values(ticker_level))
    for ticker in tickers:
        if ticker not in available:
            continue
        sub = raw.xs(ticker, axis=1, level=ticker_level)
        if not all(c in sub.columns for c in OHLCV):
            continue
        df = sub[OHLCV].dropna(how="all")
        if len(df) > 60:
            df.index.name = "Date"
            out[ticker] = df
    return out


def _dict_to_long(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, df in data.items():
        f = df.copy()
        f["ticker"] = ticker
        frames.append(f.reset_index().rename(columns={f.index.name or "index": "Date"}))
    return pd.concat(frames, ignore_index=True)


def _long_to_dict(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    date_col = "Date" if "Date" in long_df.columns else long_df.columns[0]
    out: dict[str, pd.DataFrame] = {}
    for ticker, grp in long_df.groupby("ticker"):
        g = grp.set_index(date_col).sort_index()
        out[str(ticker)] = g[OHLCV]
        out[str(ticker)].index.name = "Date"
    return out


def fetch_prices(tickers: list[str], period: str = "2y", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for ``tickers``; returns only those with usable history."""
    today = date.today().isoformat()
    path = _cache_path(tickers, period, today)

    if use_cache and path.exists():
        try:
            return _long_to_dict(pd.read_parquet(path))
        except Exception:
            pass  # corrupt cache -> refetch

    import yfinance as yf

    raw = yf.download(
        tickers=tickers, period=period, interval="1d",
        auto_adjust=True, group_by="ticker", threads=True, progress=False,
    )
    if raw is None or raw.empty:
        return {}

    data = _split_panel(raw, tickers)
    if use_cache and data:
        try:
            _dict_to_long(data).to_parquet(path)
        except Exception:
            pass
    return data


def clear_cache() -> int:
    """Delete cached parquet files. Returns the number removed."""
    removed = 0
    for f in CACHE_DIR.glob("prices_*.parquet"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed
