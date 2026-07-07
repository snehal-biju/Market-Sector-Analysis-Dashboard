"""Descriptive technical analysis per instrument.

Everything here is a factual description of price state (returns, moving-average
posture, RSI regime, MACD state) — not a trade signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading-day windows for trailing returns.
WINDOWS = {"ret_1w": 5, "ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _rsi_regime(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if v >= 70:
        return "Overbought"
    if v <= 30:
        return "Oversold"
    if v >= 55:
        return "Bullish"
    if v <= 45:
        return "Bearish"
    return "Neutral"


def _ytd_return(close: pd.Series) -> float:
    last_ts = close.index[-1]
    year_start = close[close.index >= pd.Timestamp(last_ts.year, 1, 1)]
    if len(year_start) < 2:
        return float("nan")
    return float(close.iloc[-1] / year_start.iloc[0] - 1.0)


def technical_snapshot(df: pd.DataFrame) -> dict:
    """Latest-bar technical state for one instrument (OHLCV DataFrame)."""
    close = df["Close"].astype(float)
    snap: dict = {
        "last_close": float(close.iloc[-1]),
        "prev_close": float(close.iloc[-2]) if len(close) > 1 else float("nan"),
    }
    snap["chg_1d"] = snap["last_close"] / snap["prev_close"] - 1.0 if len(close) > 1 else float("nan")

    for name, w in WINDOWS.items():
        snap[name] = float(close.iloc[-1] / close.iloc[-1 - w] - 1.0) if len(close) > w else float("nan")
    snap["ret_ytd"] = _ytd_return(close)

    for w in (20, 50, 200):
        sma = close.rolling(w).mean().iloc[-1]
        snap[f"sma{w}"] = float(sma)
        snap[f"above_sma{w}"] = bool(close.iloc[-1] > sma) if np.isfinite(sma) else False
        snap[f"px_sma{w}"] = float(close.iloc[-1] / sma - 1.0) if np.isfinite(sma) else float("nan")

    r = rsi(close, 14).iloc[-1]
    snap["rsi14"] = float(r)
    snap["rsi_regime"] = _rsi_regime(r)

    line, sig, hist = macd(close)
    snap["macd_hist"] = float(hist.iloc[-1])
    snap["macd_state"] = "Bullish" if hist.iloc[-1] > 0 else "Bearish"

    hi_52 = float(close.tail(252).max())
    lo_52 = float(close.tail(252).min())
    snap["dist_52w_high"] = float(close.iloc[-1] / hi_52 - 1.0) if hi_52 else float("nan")
    snap["dist_52w_low"] = float(close.iloc[-1] / lo_52 - 1.0) if lo_52 else float("nan")

    # Trend posture from the moving-average stack.
    c, s50, s200 = close.iloc[-1], snap["sma50"], snap["sma200"]
    if np.isfinite(s50) and np.isfinite(s200):
        if c > s50 > s200:
            snap["trend"] = "Uptrend"
        elif c < s50 < s200:
            snap["trend"] = "Downtrend"
        else:
            snap["trend"] = "Mixed"
        snap["golden_cross"] = bool(s50 > s200)  # 50-day above 200-day
    else:
        snap["trend"] = "n/a"
        snap["golden_cross"] = False

    return snap
