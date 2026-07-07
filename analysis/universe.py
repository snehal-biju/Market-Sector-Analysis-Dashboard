"""The analysis universe: US sector ETFs, broad indices, and global/regional ETFs.

Sector ETFs are the classic SPDR select-sector funds. Each US sector carries a
handful of representative large-cap holdings, used to build a constituent-based
fundamentals view (the ETFs themselves don't expose clean aggregate valuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ETF:
    ticker: str
    name: str
    group: str                      # "US Sector" | "US Broad" | "Global"
    kind: str = "sector"            # sector | index | region
    constituents: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# US sectors (SPDR select-sector ETFs) with representative top holdings
# --------------------------------------------------------------------------- #
US_SECTORS: list[ETF] = [
    ETF("XLK", "Technology", "US Sector", "sector", ("AAPL", "MSFT", "NVDA", "AVGO", "CRM")),
    ETF("XLF", "Financials", "US Sector", "sector", ("BRK-B", "JPM", "V", "MA", "BAC")),
    ETF("XLV", "Health Care", "US Sector", "sector", ("LLY", "UNH", "JNJ", "ABBV", "MRK")),
    ETF("XLE", "Energy", "US Sector", "sector", ("XOM", "CVX", "COP", "SLB", "EOG")),
    ETF("XLI", "Industrials", "US Sector", "sector", ("GE", "CAT", "RTX", "HON", "UNP")),
    ETF("XLY", "Consumer Discretionary", "US Sector", "sector", ("AMZN", "TSLA", "HD", "MCD", "BKNG")),
    ETF("XLP", "Consumer Staples", "US Sector", "sector", ("PG", "COST", "KO", "WMT", "PEP")),
    ETF("XLU", "Utilities", "US Sector", "sector", ("NEE", "SO", "DUK", "CEG", "AEP")),
    ETF("XLB", "Materials", "US Sector", "sector", ("LIN", "SHW", "FCX", "APD", "ECL")),
    ETF("XLRE", "Real Estate", "US Sector", "sector", ("PLD", "AMT", "EQIX", "WELL", "SPG")),
    ETF("XLC", "Communication Services", "US Sector", "sector", ("META", "GOOGL", "NFLX", "DIS", "TMUS")),
]

# --------------------------------------------------------------------------- #
# Broad US indices
# --------------------------------------------------------------------------- #
US_BROAD: list[ETF] = [
    ETF("SPY", "S&P 500", "US Broad", "index"),
    ETF("QQQ", "Nasdaq 100", "US Broad", "index"),
    ETF("IWM", "Russell 2000 (Small Cap)", "US Broad", "index"),
    ETF("DIA", "Dow Jones 30", "US Broad", "index"),
]

# --------------------------------------------------------------------------- #
# Global / regional
# --------------------------------------------------------------------------- #
GLOBAL: list[ETF] = [
    ETF("EEM", "Emerging Markets", "Global", "region"),
    ETF("VGK", "Europe", "Global", "region"),
    ETF("FXI", "China (Large Cap)", "Global", "region"),
    ETF("EWJ", "Japan", "Global", "region"),
    ETF("EWZ", "Brazil", "Global", "region"),
    ETF("INDA", "India", "Global", "region"),
    ETF("EWY", "South Korea", "Global", "region"),
    ETF("EWU", "United Kingdom", "Global", "region"),
]

VIX_TICKER = "^VIX"
BENCHMARK = "SPY"  # relative-strength reference

ALL_ETFS: list[ETF] = US_SECTORS + US_BROAD + GLOBAL
ETF_BY_TICKER: dict[str, ETF] = {e.ticker: e for e in ALL_ETFS}
ALL_TICKERS: list[str] = [e.ticker for e in ALL_ETFS]
GROUPS: list[str] = ["US Sector", "US Broad", "Global"]

# Risk-on (cyclical) vs risk-off (defensive) sectors for the sentiment read.
CYCLICAL = ["XLK", "XLY", "XLF", "XLI", "XLB", "XLE", "XLC"]
DEFENSIVE = ["XLP", "XLU", "XLV", "XLRE"]


def get_etf(ticker: str) -> ETF | None:
    return ETF_BY_TICKER.get(ticker.upper())


def tickers_in(group: str) -> list[str]:
    return [e.ticker for e in ALL_ETFS if e.group == group]


def all_constituents() -> list[str]:
    """Flat, de-duplicated list of every representative holding (US sectors)."""
    seen: dict[str, None] = {}
    for e in US_SECTORS:
        for c in e.constituents:
            seen.setdefault(c, None)
    return list(seen)
