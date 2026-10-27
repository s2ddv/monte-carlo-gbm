"""Validated adjusted daily prices from yfinance."""

import pandas as pd
import yfinance as yf

from calibration.prices import validate_prices


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
    try:
        return validate_prices(prices, min_observations=3, require_dates=True)
    except ValueError as exc:
        raise ValueError(f"Invalid price history for {ticker}: {exc}") from exc
