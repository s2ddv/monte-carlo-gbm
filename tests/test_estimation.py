"""Regression checks for calibration with prices spanning extreme magnitudes."""

from decimal import Decimal, localcontext
from datetime import date

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("function", [log_returns, estimate_gbm_params])
@pytest.mark.parametrize("values", [
    [100., 101., np.nan, 103., 104.],
    [np.nan, 101., 102., 103., 104.],
    [100., 101., 102., 103., np.nan],
    [100., 0., 102.], [100., -1., 102.],
    [100., np.inf, 102.], [100., -np.inf, 102.],
    [100 + 1j, 101., 102.],
])
def test_calibration_rejects_missing_nonfinite_and_nonpositive_prices(function, values):
    prices = pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values)))
    with pytest.raises(ValueError, match="positive finite"):
        function(prices)


@pytest.mark.parametrize("function", [log_returns, estimate_gbm_params])
@pytest.mark.parametrize("index", [
    pd.date_range("2026-01-01", periods=4)[::-1],
    pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-04"]),
    pd.to_datetime(["2026-01-01", None, "2026-01-03", "2026-01-04"]),
    pd.Index(pd.date_range("2026-01-01", periods=4)[::-1].date),
    pd.period_range("2026-01-01", periods=4, freq="D")[::-1],
])
def test_calibration_rejects_unordered_duplicate_or_missing_dates(function, index):
    with pytest.raises(ValueError, match="dates"):
        function(pd.Series([100., 101., 102., 103.], index=index))


@pytest.mark.parametrize("function", [log_returns, estimate_gbm_params])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NaT])
def test_calibration_rejects_missing_dates_in_mixed_object_index(function, missing):
    index = pd.Index([date(2026, 1, 3), missing, date(2026, 1, 1)], dtype=object)
    prices = pd.Series([100., 101., 102.], index=index)
    with pytest.raises(ValueError, match="dates"):
        function(prices)


def test_valid_prices_preserve_return_index_name_and_annualized_parameters():
    prices = pd.Series([100., 102., 101., 105.], name="Close",
                       index=pd.date_range("2026-01-01", periods=4, tz="UTC"))
    before = prices.copy(deep=True)
    expected = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    returns = log_returns(prices)
    pd.testing.assert_index_equal(returns.index, prices.index[1:])
    assert returns.name == "Close"
    np.testing.assert_allclose(returns, expected, rtol=1e-12)
    expected_sigma = np.std(expected, ddof=1) * np.sqrt(252)
    expected_mu = np.mean(expected) * 252 + expected_sigma**2 / 2
    np.testing.assert_allclose(estimate_gbm_params(prices), [expected_mu, expected_sigma])
    pd.testing.assert_series_equal(prices, before)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_estimation_requires_two_returns_for_sample_volatility(count):
    with pytest.raises(ValueError, match="three prices"):
        estimate_gbm_params(pd.Series([100.] * count, dtype=float))


def test_undated_prices_remain_supported():
    prices = pd.Series([100., 100., 100.])
    assert log_returns(prices).tolist() == [0., 0.]
    assert estimate_gbm_params(prices) == (0., 0.)
