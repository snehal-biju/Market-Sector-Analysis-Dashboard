"""Unsupervised market-regime detection (the ML layer).

Clusters the daily market state — trend, volatility, VIX, breadth, and drawdown —
into ``k`` data-driven regimes with KMeans, then assigns each cluster an
interpretable, stress-ordered label (Calm Uptrend ... Risk-off / Drawdown).

This is descriptive analysis: it labels *what kind of market we're in*, learned
from the data. It makes no price forecast and no trade recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .universe import BENCHMARK, US_SECTORS, VIX_TICKER

REGIME_FEATURES = ["ret_21", "vol_21", "vix", "breadth", "drawdown"]

# Stress-ordered label templates (calmest first). Chosen by cluster count.
_LABEL_TEMPLATES = {
    2: ["Risk-on / Calm", "Risk-off / Stress"],
    3: ["Calm Uptrend", "Choppy / Transition", "Risk-off / Drawdown"],
    4: ["Calm Uptrend", "Steady / Mild", "Choppy / Transition", "Risk-off / Drawdown"],
    5: ["Calm Uptrend", "Steady", "Choppy / Transition", "Volatile", "Risk-off / Drawdown"],
    6: ["Calm Uptrend", "Steady", "Mild Chop", "Choppy / Transition", "Volatile", "Risk-off / Drawdown"],
}

REGIME_COLORS = {
    "Risk-on / Calm": "#2e7d32", "Calm Uptrend": "#2e7d32", "Steady / Mild": "#66bb6a",
    "Steady": "#66bb6a", "Mild Chop": "#c0ca33", "Choppy / Transition": "#ffa726",
    "Volatile": "#fb8c00", "Risk-off / Drawdown": "#c62828", "Risk-off / Stress": "#c62828",
}


def build_regime_features(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily market-state features from SPY, VIX, and sector breadth."""
    spy = price_data.get(BENCHMARK)
    if spy is None or spy.empty:
        return pd.DataFrame()

    close = spy["Close"].astype(float)
    daily = close.pct_change()
    feats = pd.DataFrame(index=close.index)
    feats["ret_21"] = close.pct_change(21)
    feats["vol_21"] = daily.rolling(21).std() * np.sqrt(252)
    feats["drawdown"] = close / close.rolling(252, min_periods=20).max() - 1.0

    vix = price_data.get(VIX_TICKER)
    feats["vix"] = (vix["Close"].astype(float).reindex(feats.index)
                    if vix is not None and not vix.empty else np.nan)

    above = []
    for e in US_SECTORS:
        df = price_data.get(e.ticker)
        if df is not None and not df.empty:
            c = df["Close"].astype(float)
            above.append((c > c.rolling(50).mean()).reindex(feats.index))
    feats["breadth"] = pd.concat(above, axis=1).mean(axis=1) if above else np.nan
    return feats


def _name_clusters(profile: pd.DataFrame, k: int) -> dict[int, str]:
    """Order clusters from calmest to most stressed and map to readable names."""
    std = profile.std(ddof=0).replace(0, 1.0)
    z = (profile - profile.mean()) / std
    stress = (z.get("vol_21", 0) + z.get("vix", 0)
              - z.get("ret_21", 0) - z.get("breadth", 0) - z.get("drawdown", 0))
    order = stress.sort_values().index.tolist()  # calmest first
    labels = _LABEL_TEMPLATES.get(k, [f"Regime {i + 1}" for i in range(k)])
    return {cl: labels[i] for i, cl in enumerate(order)}


def detect_regimes(price_data: dict[str, pd.DataFrame], n_regimes: int = 4,
                   random_state: int = 42) -> dict:
    """Fit KMeans regimes and return a timeline, per-regime profile, and current state."""
    feats = build_regime_features(price_data)
    # Keep only features that actually have data (e.g. drop VIX if it didn't fetch).
    cols = [c for c in REGIME_FEATURES if c in feats.columns and feats[c].notna().any()]
    data = feats[cols].dropna()
    if len(data) < 60 or len(cols) < 2:
        return {}

    X = StandardScaler().fit_transform(data)
    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    clusters = km.fit_predict(X)

    timeline = data.copy()
    timeline["cluster"] = clusters
    profile = timeline.groupby("cluster")[cols].mean()
    names = _name_clusters(profile, n_regimes)

    timeline["regime"] = timeline["cluster"].map(names)
    profile = profile.copy()
    profile["days"] = timeline.groupby("cluster").size()
    profile["pct_time"] = profile["days"] / profile["days"].sum()
    profile["regime"] = [names[c] for c in profile.index]

    # Consecutive days in the current regime.
    current = str(timeline["regime"].iloc[-1])
    run = 0
    for r in reversed(timeline["regime"].tolist()):
        if r == current:
            run += 1
        else:
            break

    return {
        "timeline": timeline,
        "profile": profile.reset_index(drop=True),
        "current": current,
        "days_in_regime": run,
        "n_regimes": n_regimes,
    }
