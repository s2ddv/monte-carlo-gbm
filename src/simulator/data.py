"""Historical market data download."""

import pandas as pd
import  yfinance as yf

def download_prices(ticker: str, period: str = "3y") -> pd.Series:
    """Download adjusted daily closing prices for a ticker.

        Args:
            ticker: Asset symbol (e.g. "PETR4.SA" for Brazilian stocks).
            period: How far back to look (e.g. "1y", "3y", "5y").

        Returns:
            Series of daily closing prices indexed by date.
    """

    data = yf.download(ticker, period = period, auto_adjust = True, progress = False)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    prices = data["Close"].squeeze()

    prices = prices.dropna()

    return prices