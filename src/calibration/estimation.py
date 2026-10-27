"""Estimation of GBM parameters from historical prices."""

import numpy as np
import pandas as pd

from calibration.prices import validate_prices

TRADING_DAYS = 252

def log_returns(prices: pd.Series) -> pd.Series:
    """Compute ln(S_t) - ln(S_{t-1}) for valid ordered positive prices.

    Only the first difference is omitted; missing observations are rejected.
    """
    prices = validate_prices(prices)
    return np.log(prices).diff().iloc[1:]

def estimate_gbm_params(prices: pd.Series) -> tuple[float, float]:
    """Estimate price drift (year^-1) and volatility (year^-1/2).

    Require at least three closes for two returns and sample std (ddof=1).
    Annualize with 252 trading days; mu = 252*mean(log return) + sigma^2/2.
    """
    prices = validate_prices(prices, min_observations=3)
    returns = log_returns(prices)

    sigma = returns.std() * np.sqrt(TRADING_DAYS)

    mu = returns.mean() * TRADING_DAYS + 0.5 * sigma**2

    return float(mu), float(sigma)
