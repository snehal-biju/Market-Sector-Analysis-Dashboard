"""Market & Sector Analysis Dashboard.

End-of-day, descriptive analysis of US sectors and global markets — sentiment,
momentum, technicals, and fundamentals. It describes market state; it does not
recommend trades.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.data import clear_cache, fetch_prices
from analysis.fundamentals import sector_fundamentals
from analysis.regime import REGIME_COLORS, detect_regimes
from analysis.sectors import momentum_table
from analysis.sentiment import market_sentiment
from analysis.technicals import macd, rsi, technical_snapshot
from analysis.universe import ALL_TICKERS, BENCHMARK, VIX_TICKER, get_etf, tickers_in

st.set_page_config(page_title="Market & Sector Analysis", page_icon="📊", layout="wide")

CLASS_COLORS = {
    "Strong +ve": "#1b7d3c", "+ve": "#66bb6a", "Neutral": "#9e9e9e",
    "-ve": "#ef5350", "Strong -ve": "#b71c1c", "n/a": "#cccccc",
}

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Fetching end-of-day prices…", ttl=60 * 60 * 6)
def load_prices(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    return fetch_prices(list(tickers), period=period)


@st.cache_data(show_spinner="Loading sector fundamentals…", ttl=60 * 60 * 12)
def load_fundamentals() -> pd.DataFrame:
    return sector_fundamentals(use_cache=True)


@st.cache_data(show_spinner="Detecting market regimes…", ttl=60 * 60 * 6)
def load_regimes(tickers: tuple[str, ...], period: str, k: int) -> dict:
    return detect_regimes(load_prices(tickers, period), n_regimes=k)


def snapshot_table(price_data: dict, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        df = price_data.get(t)
        if df is None or df.empty:
            continue
        s = technical_snapshot(df)
        etf = get_etf(t)
        rows.append({
            "Ticker": t, "Name": etf.name if etf else t,
            "Close": s["last_close"], "1d %": s["chg_1d"] * 100,
            "1m %": s["ret_1m"] * 100, "3m %": s["ret_3m"] * 100,
            "RSI": s["rsi14"], "Trend": s["trend"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("📊 Controls")
period = st.sidebar.selectbox(
    "History window", ["1y", "2y", "3y", "5y"], index=1,
    help="How much daily history to pull (needs ~1y for 200-day averages).",
)
n_regimes = st.sidebar.slider(
    "Market regimes (k)", 2, 6, 4,
    help="Number of unsupervised (KMeans) market-state clusters on the Market Regime tab.",
)
if st.sidebar.button("🔄 Refresh data (clear cache)"):
    n = clear_cache()
    load_prices.clear()
    load_fundamentals.clear()
    st.sidebar.success(f"Cleared {n} cached file(s). Reloading…")
    st.rerun()

st.sidebar.markdown(
    "---\nEnd-of-day data from Yahoo Finance. **Descriptive analysis only — "
    "not investment advice.**"
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("📊 Market & Sector Analysis Dashboard")
st.caption(
    "Top-down, end-of-day view of US sectors and global markets — sentiment, "
    "momentum, technicals, and fundamentals."
)
st.info(
    "**Descriptive analysis, not investment advice.** This dashboard summarizes "
    "market *state* (momentum, trend, valuation, sentiment). It does **not** "
    "predict prices or recommend buying or selling any security. Data is "
    "end-of-day and may be delayed.",
    icon="ℹ️",
)

fetch_tickers = tuple(ALL_TICKERS + [VIX_TICKER])
price_data = load_prices(fetch_tickers, period)
if not price_data:
    st.error("No price data returned — Yahoo Finance may be unreachable. Try **Refresh data**.")
    st.stop()

sent = market_sentiment(price_data)
last_session = max(
    (df.index.max() for df in price_data.values() if not df.empty), default=None
)
if last_session is not None:
    st.success(f"Loaded **{len(price_data)}** instruments · latest close **{last_session.date()}**")

tab_overview, tab_regime, tab_sectors, tab_global, tab_fund, tab_detail = st.tabs(
    ["🧭 Market Overview", "🧠 Market Regime", "🔄 Sector Momentum",
     "🌍 Global & Regional", "💰 Fundamentals", "🔎 Detail"]
)

# --------------------------------------------------------------------------- #
# Market Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    if "score" in sent:
        c1, c2 = st.columns([2, 3])
        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=sent["score"],
                title={"text": f"Risk appetite — <b>{sent['label']}</b>"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#333333"},
                    "steps": [
                        {"range": [0, 30], "color": "#ef9a9a"},
                        {"range": [30, 42], "color": "#ffcc80"},
                        {"range": [42, 58], "color": "#eeeeee"},
                        {"range": [58, 70], "color": "#a5d6a7"},
                        {"range": [70, 100], "color": "#66bb6a"},
                    ],
                }))
            gauge.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(gauge, width="stretch")
        with c2:
            st.markdown("**What's behind the read**")
            comp = sent.get("components", {})
            g1, g2 = st.columns(2)
            g1.metric("S&P 500 trend", sent.get("spy_trend", "—"),
                      delta=f"{sent.get('spy_chg_1d', 0) * 100:+.2f}% last close")
            g2.metric("VIX", f"{sent.get('vix', float('nan')):.1f}",
                      delta=f"{sent.get('vix_percentile', float('nan')) * 100:.0f}%ile (1y)",
                      delta_color="off")
            g3, g4 = st.columns(2)
            g3.metric("Sectors > 50-day MA", f"{sent.get('breadth_50', float('nan')) * 100:.0f}%")
            g4.metric("Cyclical − Defensive (3m)", f"{sent.get('cyc_def_spread', float('nan')) * 100:+.1f}%")
            st.caption(
                "Composite blends four EOD signals (each 0–1): "
                + " · ".join(f"{k} {v:.2f}" for k, v in comp.items())
                + ". Higher = more risk-on. Descriptive only."
            )
    else:
        st.warning("Sentiment unavailable (missing SPY/VIX data).")

    st.divider()
    st.subheader("Previous close — broad indices")
    bt = snapshot_table(price_data, tickers_in("US Broad"))
    st.dataframe(
        bt, hide_index=True, width="stretch",
        column_config={
            "Close": st.column_config.NumberColumn(format="$%.2f"),
            "1d %": st.column_config.NumberColumn(format="%.2f%%"),
            "1m %": st.column_config.NumberColumn(format="%.1f%%"),
            "3m %": st.column_config.NumberColumn(format="%.1f%%"),
            "RSI": st.column_config.NumberColumn(format="%.0f"),
        },
    )

# --------------------------------------------------------------------------- #
# Market Regime (unsupervised ML)
# --------------------------------------------------------------------------- #
with tab_regime:
    st.subheader("Market regime — unsupervised ML")
    st.caption(
        "KMeans clusters the daily market state (S&P 500 trend & volatility, VIX, "
        "sector breadth, drawdown) into data-driven regimes, then labels them by "
        "their stress profile. Descriptive — it names the market environment, it "
        "does not forecast prices."
    )
    reg = load_regimes(fetch_tickers, period, n_regimes)
    if not reg:
        st.info("Not enough history to fit regimes — try a longer history window.")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Current regime", reg["current"])
        r2.metric("Days in current regime", f"{reg['days_in_regime']}")
        r3.metric("Clusters (k)", reg["n_regimes"])

        tl = reg["timeline"]
        spy_close = price_data[BENCHMARK]["Close"].astype(float).reindex(tl.index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tl.index, y=spy_close, mode="lines",
                                 line=dict(color="#cccccc", width=1),
                                 name="S&P 500", showlegend=False))
        for name, grp in tl.groupby("regime"):
            fig.add_trace(go.Scatter(
                x=grp.index, y=spy_close.reindex(grp.index), mode="markers",
                marker=dict(size=5, color=REGIME_COLORS.get(name, "#888888")), name=name))
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="S&P 500 (SPY)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is a trading day, colored by its detected regime.")

        prof = reg["profile"]
        pdisp = pd.DataFrame({"Regime": prof["regime"]})
        if "ret_21" in prof:
            pdisp["Avg 21d ret %"] = (prof["ret_21"] * 100).round(1)
        if "vol_21" in prof:
            pdisp["Avg ann. vol %"] = (prof["vol_21"] * 100).round(1)
        if "vix" in prof:
            pdisp["Avg VIX"] = prof["vix"].round(1)
        if "breadth" in prof:
            pdisp["Avg breadth %"] = (prof["breadth"] * 100).round(0)
        if "drawdown" in prof:
            pdisp["Avg drawdown %"] = (prof["drawdown"] * 100).round(1)
        pdisp["Days"] = prof["days"].astype(int)
        pdisp["% of time"] = (prof["pct_time"] * 100).round(1)
        st.dataframe(pdisp, hide_index=True, width="stretch",
                     column_config={"% of time": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption(
            "Clusters are unsupervised, then ordered/named by a stress score "
            "(volatility, VIX, negative return, weak breadth, drawdown). Labels "
            "describe historical market state — not a prediction."
        )

# --------------------------------------------------------------------------- #
# Sector Momentum
# --------------------------------------------------------------------------- #
with tab_sectors:
    st.subheader("US sector momentum")
    st.caption(
        "Blended, recent-weighted annualized momentum (1m/3m/6m) with a "
        "descriptive **+ve / −ve** label; relative strength is vs the S&P 500 (3m)."
    )
    mt = momentum_table(price_data, tickers_in("US Sector"))
    if mt.empty:
        st.info("No sector data available.")
    else:
        colors = ["#1b7d3c" if v >= 0 else "#c62828" for v in mt["ann_momentum"]]
        fig = go.Figure(go.Bar(
            x=mt["ann_momentum"] * 100, y=mt["name"], orientation="h",
            marker_color=colors,
            text=[f"{c}" for c in mt["classification"]], textposition="auto"))
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title="Blended annualized momentum %")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        disp = mt[["rank", "ticker", "name", "classification", "last_close", "chg_1d",
                   "ret_1m", "ret_3m", "ret_6m", "rel_strength", "rsi14", "trend"]].copy()
        for c in ("chg_1d", "ret_1m", "ret_3m", "ret_6m", "rel_strength"):
            disp[c] = disp[c] * 100
        disp = disp.rename(columns={
            "rank": "#", "ticker": "ETF", "name": "Sector", "classification": "Momentum",
            "last_close": "Close", "chg_1d": "1d %", "ret_1m": "1m %", "ret_3m": "3m %",
            "ret_6m": "6m %", "rel_strength": "RS vs SPY", "rsi14": "RSI", "trend": "Trend",
        })
        st.dataframe(
            disp, hide_index=True, width="stretch", height=430,
            column_config={
                "Close": st.column_config.NumberColumn(format="$%.2f"),
                "1d %": st.column_config.NumberColumn(format="%.2f%%"),
                "1m %": st.column_config.NumberColumn(format="%.1f%%"),
                "3m %": st.column_config.NumberColumn(format="%.1f%%"),
                "6m %": st.column_config.NumberColumn(format="%.1f%%"),
                "RS vs SPY": st.column_config.NumberColumn(format="%+.1f%%"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption("Momentum labels describe trailing price behavior — they are not forecasts.")

# --------------------------------------------------------------------------- #
# Global & Regional
# --------------------------------------------------------------------------- #
with tab_global:
    st.subheader("Global & regional momentum")
    gt = momentum_table(price_data, tickers_in("Global"))
    if gt.empty:
        st.info("No global data available.")
    else:
        colors = ["#1b7d3c" if v >= 0 else "#c62828" for v in gt["ann_momentum"]]
        fig = go.Figure(go.Bar(
            x=gt["ann_momentum"] * 100, y=gt["name"], orientation="h",
            marker_color=colors, text=gt["classification"], textposition="auto"))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis_title="Blended annualized momentum %")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

        gdisp = gt[["rank", "ticker", "name", "classification", "last_close", "chg_1d",
                    "ret_1m", "ret_3m", "rel_strength", "rsi14", "trend"]].copy()
        for c in ("chg_1d", "ret_1m", "ret_3m", "rel_strength"):
            gdisp[c] = gdisp[c] * 100
        gdisp = gdisp.rename(columns={
            "rank": "#", "ticker": "ETF", "name": "Region", "classification": "Momentum",
            "last_close": "Close", "chg_1d": "1d %", "ret_1m": "1m %", "ret_3m": "3m %",
            "rel_strength": "RS vs SPY", "rsi14": "RSI", "trend": "Trend",
        })
        st.dataframe(
            gdisp, hide_index=True, width="stretch",
            column_config={
                "Close": st.column_config.NumberColumn(format="$%.2f"),
                "1d %": st.column_config.NumberColumn(format="%.2f%%"),
                "1m %": st.column_config.NumberColumn(format="%.1f%%"),
                "3m %": st.column_config.NumberColumn(format="%.1f%%"),
                "RS vs SPY": st.column_config.NumberColumn(format="%+.1f%%"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
            },
        )

# --------------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------------- #
with tab_fund:
    st.subheader("Sector fundamentals (constituent-based)")
    st.caption(
        "Median of each sector's representative large-cap holdings. First load "
        "fetches one quote per holding from Yahoo (then cached for the day)."
    )
    fund = load_fundamentals()
    if fund.empty:
        st.warning("Fundamentals unavailable right now (Yahoo `.info` throttling). Try again shortly.")
    else:
        fdisp = fund[["sector", "trailingPE", "forwardPE", "priceToBook",
                      "profitMargin", "dividendYield", "holdings_with_data", "holdings"]].copy()
        fdisp["profitMargin"] = fdisp["profitMargin"] * 100
        fdisp["dividendYield"] = fdisp["dividendYield"] * 100
        fdisp = fdisp.rename(columns={
            "sector": "Sector", "trailingPE": "P/E (ttm)", "forwardPE": "P/E (fwd)",
            "priceToBook": "P/B", "profitMargin": "Net margin", "dividendYield": "Div yield",
            "holdings_with_data": "n", "holdings": "Representative holdings",
        })
        st.dataframe(
            fdisp, hide_index=True, width="stretch", height=430,
            column_config={
                "P/E (ttm)": st.column_config.NumberColumn(format="%.1f"),
                "P/E (fwd)": st.column_config.NumberColumn(format="%.1f"),
                "P/B": st.column_config.NumberColumn(format="%.1f"),
                "Net margin": st.column_config.NumberColumn(format="%.1f%%"),
                "Div yield": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "Valuation is descriptive, not a fair-value estimate. Medians reduce the "
            "pull of outliers but a handful of mega-caps still dominate each sector ETF."
        )

# --------------------------------------------------------------------------- #
# Detail
# --------------------------------------------------------------------------- #
with tab_detail:
    avail = [t for t in ALL_TICKERS if t in price_data]
    ticker = st.selectbox("Instrument", avail, index=0)
    etf = get_etf(ticker)
    df = price_data[ticker].tail(260)
    snap = technical_snapshot(price_data[ticker])

    st.caption(f"**{etf.name}** · {etf.group}")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Last close", f"${snap['last_close']:,.2f}", delta=f"{snap['chg_1d'] * 100:+.2f}%")
    d2.metric("1m / 3m", f"{snap['ret_1m'] * 100:+.1f}% / {snap['ret_3m'] * 100:+.1f}%")
    d3.metric("RSI(14)", f"{snap['rsi14']:.0f}", delta=snap["rsi_regime"], delta_color="off")
    d4.metric("Trend", snap["trend"])
    d5.metric("vs 200-day MA", f"{snap['px_sma200'] * 100:+.1f}%")

    close = df["Close"].astype(float)
    sma20, sma50, sma200 = (close.rolling(w).mean() for w in (20, 50, 200))
    r = rsi(close, 14)
    line, sig, hist = macd(close)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03, subplot_titles=("Price + SMAs", "RSI(14)", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name="Price", showlegend=False), row=1, col=1)
    for s, nm in ((sma20, "SMA20"), (sma50, "SMA50"), (sma200, "SMA200")):
        fig.add_trace(go.Scatter(x=df.index, y=s, name=nm, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=r, name="RSI", line=dict(width=1), showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=line, name="MACD", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sig, name="Signal", line=dict(width=1)), row=3, col=1)
    fig.update_layout(height=680, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")

    if etf.constituents:
        st.markdown(f"**Representative holdings:** {', '.join(etf.constituents)}")

st.divider()
st.caption(
    "Data: Yahoo Finance (yfinance), auto-adjusted end-of-day closes · "
    "Descriptive analysis for education/research only — not investment advice."
)
