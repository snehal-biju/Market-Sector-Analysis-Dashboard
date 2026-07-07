"""Sector momentum scoring and classification.

Builds a recent-weighted, annualized momentum score per instrument and labels it
descriptively (Strong +ve ... Strong -ve). Relative strength vs the benchmark is
reported alongside as context. Nothing here is a trade recommendation — it's a
description of momentum state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .technicals import technical_snapshot
from .universe import BENCHMARK, get_etf

# Recent-weighted blend of annualized trailing returns.
_HORIZON_WEIGHTS = {"ret_1m": (0.5, 12), "ret_3m": (0.3, 4), "ret_6m": (0.2, 2)}


def _annualize(ret: float, periods_per_year: int) -> float:
    if not np.isfinite(ret):
        return np.nan
    return (1.0 + ret) ** periods_per_year - 1.0


def _blended_momentum(snap: dict) -> float:
    total, wsum = 0.0, 0.0
    for key, (w, ppy) in _HORIZON_WEIGHTS.items():
        a = _annualize(snap.get(key, np.nan), ppy)
        if np.isfinite(a):
            total += w * a
            wsum += w
    return total / wsum if wsum else np.nan


def classify(score: float, trend: str) -> str:
    """Descriptive momentum label from the blended annualized score + trend."""
    if not np.isfinite(score):
        return "n/a"
    if score >= 0.25 and trend != "Downtrend":
        return "Strong +ve"
    if score >= 0.08:
        return "+ve"
    if score <= -0.25 and trend != "Uptrend":
        return "Strong -ve"
    if score <= -0.08:
        return "-ve"
    return "Neutral"


def momentum_table(price_data: dict[str, pd.DataFrame], tickers: list[str],
                   benchmark: str = BENCHMARK) -> pd.DataFrame:
    """One row per instrument: returns, blended momentum, relative strength,
    RSI, trend posture, and a +ve/-ve classification, ranked strongest first."""
    bench = price_data.get(benchmark)
    bench_snap = technical_snapshot(bench) if bench is not None and not bench.empty else {}
    bench_3m = bench_snap.get("ret_3m", np.nan)

    rows = []
    for t in tickers:
        df = price_data.get(t)
        if df is None or df.empty:
            continue
        snap = technical_snapshot(df)
        score = _blended_momentum(snap)
        rs = snap["ret_3m"] - bench_3m if np.isfinite(snap["ret_3m"]) and np.isfinite(bench_3m) else np.nan
        etf = get_etf(t)
        rows.append({
            "ticker": t,
            "name": etf.name if etf else t,
            "group": etf.group if etf else "",
            "last_close": snap["last_close"],
            "chg_1d": snap["chg_1d"],
            "ret_1m": snap["ret_1m"],
            "ret_3m": snap["ret_3m"],
            "ret_6m": snap["ret_6m"],
            "ann_momentum": score,
            "rel_strength": rs,
            "rsi14": snap["rsi14"],
            "trend": snap["trend"],
            "above_sma200": snap["above_sma200"],
            "classification": classify(score, snap["trend"]),
        })
    if not rows:
        return pd.DataFrame()
    tab = pd.DataFrame(rows).sort_values("ann_momentum", ascending=False).reset_index(drop=True)
    tab.insert(0, "rank", tab.index + 1)
    return tab
