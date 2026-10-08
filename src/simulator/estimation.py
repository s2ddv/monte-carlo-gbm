"""Estimation of GBM parameters from historical prices."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns: ln(S_t / S_{t-1})."""

    return np.log(prices / prices.shift(1)).dropna()

def estimate_gbm_params(prices: pd.Series) -> tuple[float, float]:
    returns = log_returns(prices)

    sigma = returns.std() * np.sqrt(TRADING_DAYS)

    mu = returns.mean() * TRADING_DAYS + 0.5 * sigma**2

    return float(mu), float(sigma)

