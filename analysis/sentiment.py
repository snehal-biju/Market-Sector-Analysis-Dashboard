"""Top-down market sentiment: a composite risk-appetite read.

Blends four end-of-day signals into a 0-100 score with a descriptive label:
  * Trend    — S&P 500 vs its 50- and 200-day moving averages
  * Volatility — where VIX sits within its trailing 1-year range (low = calm)
  * Breadth  — share of US sectors trading above their 50-day average
  * Rotation — cyclical vs defensive sectors' 3-month performance spread

It describes the market regime; it does not tell you what to do about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .technicals import technical_snapshot
from .universe import BENCHMARK, CYCLICAL, DEFENSIVE, US_SECTORS, VIX_TICKER


def _label(score: float) -> str:
    if score >= 70:
        return "Risk-on"
    if score >= 58:
        return "Constructive"
    if score >= 42:
        return "Neutral"
    if score >= 30:
        return "Cautious"
    return "Risk-off"


def _avg_ret3m(price_data: dict[str, pd.DataFrame], tickers: list[str]) -> float:
    vals = []
    for t in tickers:
        df = price_data.get(t)
        if df is not None and not df.empty:
            vals.append(technical_snapshot(df)["ret_3m"])
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def market_sentiment(price_data: dict[str, pd.DataFrame]) -> dict:
    out: dict = {}
    comps: dict[str, float] = {}

    # --- Trend (S&P 500 vs MAs) ---
    spy = price_data.get(BENCHMARK)
    if spy is not None and not spy.empty:
        s = technical_snapshot(spy)
        out.update({
            "spy_trend": s["trend"], "spy_above_50": s["above_sma50"],
            "spy_above_200": s["above_sma200"], "spy_ret_1m": s["ret_1m"],
            "spy_ret_3m": s["ret_3m"], "spy_last": s["last_close"], "spy_chg_1d": s["chg_1d"],
        })
        comps["trend"] = 0.5 * float(s["above_sma200"]) + 0.5 * float(s["above_sma50"])

    # --- Volatility (VIX percentile within trailing year; low = calm) ---
    vix = price_data.get(VIX_TICKER)
    if vix is not None and not vix.empty:
        v = vix["Close"].astype(float)
        last = float(v.iloc[-1])
        pct = float((v.tail(252) < last).mean())
        out.update({"vix": last, "vix_percentile": pct})
        comps["volatility"] = 1.0 - pct

    # --- Breadth (US sectors above 50-/200-day MA) ---
    above50, above200 = [], []
    for e in US_SECTORS:
        df = price_data.get(e.ticker)
        if df is not None and not df.empty:
            snap = technical_snapshot(df)
            above50.append(snap["above_sma50"])
            above200.append(snap["above_sma200"])
    if above50:
        out["breadth_50"] = float(np.mean(above50))
        out["breadth_200"] = float(np.mean(above200))
        comps["breadth"] = out["breadth_50"]

    # --- Rotation (cyclicals vs defensives, 3-month) ---
    cyc = _avg_ret3m(price_data, CYCLICAL)
    dfn = _avg_ret3m(price_data, DEFENSIVE)
    if np.isfinite(cyc) and np.isfinite(dfn):
        out["cyc_def_spread"] = cyc - dfn
        comps["rotation"] = float(np.clip(0.5 + (cyc - dfn) * 3.0, 0.0, 1.0))

    if comps:
        score = float(np.mean(list(comps.values()))) * 100.0
        out["score"] = round(score, 1)
        out["label"] = _label(score)
        out["components"] = {k: round(v, 3) for k, v in comps.items()}

    return out
