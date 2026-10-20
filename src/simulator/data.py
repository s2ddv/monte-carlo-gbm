"""Validated adjusted daily prices from yfinance."""

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(ticker: str, period: str = "3y") -> pd.Series:
    """Return one chronological adjusted-close series for a single ticker."""
    data = yf.download(ticker, period=period, auto_adjust=True,
                       progress=False, group_by="column")
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Unexpected price response for {ticker}: expected a table.")
    if data.empty:
        raise ValueError(f"No price history returned for {ticker} ({period}).")
    columns = data.columns
    if isinstance(columns, pd.MultiIndex):
        if columns.nlevels != 2:
            raise ValueError(f"Unexpected price columns for {ticker}: require two levels.")
        matches = [column for column in columns if column[0] == "Close"]
        if len(matches) == 1 and matches[0][1] != ticker:
            matches = []
    else:
        matches = [column for column in columns if column == "Close"]
    if len(matches) != 1 or columns.tolist().count(matches[0]) != 1:
        raise ValueError(f"Unexpected price columns for {ticker}: require one adjusted Close series.")
    prices = data[matches[0]]
    if not isinstance(prices, pd.Series):
        raise ValueError(f"Unexpected price shape for {ticker}: require one series.")
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or not prices.index.is_unique or not prices.index.is_monotonic_increasing):
        raise ValueError(f"Invalid price dates for {ticker}: require unique ordered dates.")
    if len(prices) < 3:
        raise ValueError(f"Insufficient price history for {ticker}: require at least three prices.")
    try:
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prices for {ticker}: require positive finite numbers.") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid prices for {ticker}: require positive finite numbers.")
    return prices.astype(float)
