"""End-to-end smoke test (no Streamlit server).

Fetches the universe, computes market sentiment, sector/global momentum, and
constituent fundamentals, and prints a summary. Run:  python smoke_test.py
"""

import pandas as pd

from analysis.data import fetch_prices
from analysis.fundamentals import sector_fundamentals
from analysis.regime import detect_regimes
from analysis.sectors import momentum_table
from analysis.sentiment import market_sentiment
from analysis.universe import ALL_TICKERS, VIX_TICKER, tickers_in

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main() -> int:
    tickers = ALL_TICKERS + [VIX_TICKER]
    print(f"Fetching {len(tickers)} instruments (2y)...")
    data = fetch_prices(tickers, period="2y", use_cache=False)
    print(f"  usable: {len(data)} -> {sorted(data.keys())}")
    if not data:
        print("  !! No data returned (network/Yahoo issue).")
        return 1

    sent = market_sentiment(data)
    vix = sent.get("vix")
    print(f"\nMarket sentiment: score={sent.get('score')} ({sent.get('label')})")
    print(f"  components: {sent.get('components')}")
    print(f"  SPY trend={sent.get('spy_trend')} | VIX={vix:.1f} " if vix else "  VIX n/a")
    print(f"  breadth>50MA={sent.get('breadth_50')} | cyc-def(3m)={sent.get('cyc_def_spread')}")

    reg = detect_regimes(data, n_regimes=4)
    if reg:
        print(f"\nMarket regime (KMeans k=4): current = {reg['current']} "
              f"({reg['days_in_regime']} days in regime)")
        prof = reg["profile"][["regime", "ret_21", "vol_21", "vix", "days", "pct_time"]].copy()
        prof["ret_21"] = (prof["ret_21"] * 100).round(1)
        prof["vol_21"] = (prof["vol_21"] * 100).round(1)
        print(prof.round(2).to_string(index=False))

    mt = momentum_table(data, tickers_in("US Sector"))
    print("\nUS sector momentum (strongest first):")
    show = mt[["rank", "ticker", "name", "classification", "ret_3m", "rel_strength", "rsi14", "trend"]].copy()
    show["ret_3m"] = (show["ret_3m"] * 100).round(1)
    show["rel_strength"] = (show["rel_strength"] * 100).round(1)
    print(show.to_string(index=False))

    gt = momentum_table(data, tickers_in("Global"))
    print("\nGlobal/regional momentum:")
    gshow = gt[["rank", "ticker", "name", "classification", "ret_3m"]].copy()
    gshow["ret_3m"] = (gshow["ret_3m"] * 100).round(1)
    print(gshow.to_string(index=False))

    print("\nFundamentals (one Yahoo .info call per holding; first run ~30s)...")
    fund = sector_fundamentals(use_cache=True)
    if not fund.empty:
        f = fund[["sector", "trailingPE", "forwardPE", "priceToBook", "dividendYield", "holdings_with_data"]].copy()
        f["dividendYield"] = (f["dividendYield"] * 100).round(2)
        print(f.round(2).to_string(index=False))
    else:
        print("  (fundamentals unavailable right now)")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
