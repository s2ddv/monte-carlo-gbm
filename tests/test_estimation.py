"""Regression checks for calibration with prices spanning extreme magnitudes."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from calibration.estimation import estimate_gbm_params, log_returns


def test_extreme_positive_prices_keep_finite_log_returns_and_parameters():
    prices = pd.Series([1e-300, 1e300, 1e-300])

    with localcontext() as context:
        context.prec = 80
        up = Decimal("1e300").ln() - Decimal("1e-300").ln()
        down = -up
        annual_sigma = ((up * up + down * down) * Decimal(252)).sqrt()
        annual_mu = annual_sigma * annual_sigma / 2

    observed_returns = log_returns(prices).to_numpy()
    np.testing.assert_allclose(observed_returns, [float(up), float(down)])
    assert np.isfinite(observed_returns).all()

    mu, sigma = estimate_gbm_params(prices)
    np.testing.assert_allclose([mu, sigma],
                               [float(annual_mu), float(annual_sigma)], rtol=1e-12)
    assert np.isfinite([mu, sigma]).all()
