"""Chronological, fixed-calibration historical GBM diagnostics."""

from statistics import NormalDist
from numbers import Real

import numpy as np
import pandas as pd

from calibration.estimation import TRADING_DAYS, estimate_gbm_params, log_returns
from calibration.prices import validate_prices


def _violation_rate_interval(violations: int, forecasts: int) -> dict:
    """Two-sided 95% Wilson score interval for a binomial proportion.

    With p_hat=k/n, z=Phi^-1(0.975), d=1+z^2/n, the endpoints are
    (p_hat+z^2/(2*n) +/- z*sqrt(p_hat*(1-p_hat)/n+z^2/(4*n^2)))/d.
    Counts come from the nonempty forecast rows; bounds are dimensionless.
    """
    confidence_level = 0.95
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    rate = violations / forecasts
    denominator = 1 + z**2 / forecasts
    center = (rate + z**2 / (2 * forecasts)) / denominator
    half_width = (z * np.sqrt(rate * (1 - rate) / forecasts
                             + z**2 / (4 * forecasts**2)) / denominator)
    return {
        "method": "wilson_score",
        "confidence_level": confidence_level,
        "lower": 0.0 if violations == 0 else float(max(0.0, center - half_width)),
        "upper": 1.0 if violations == forecasts else float(min(1.0, center + half_width)),
        "assumptions": "Independent Bernoulli violations with a constant probability, conditional on the fitted model.",
        "limitations": (
            "Small samples give wide intervals and approximate coverage; zero violations do not establish zero risk. "
            "Nonoverlapping horizons do not guarantee independence in market data. "
            "This estimates uncertainty in the observed violation rate, not the nominal rate 1-alpha "
            "or uncertainty in GBM parameters or model specification."
        ),
    }


def evaluate_out_of_sample(prices: pd.Series, *, calibration_observations: int,
                           horizon_days: int = 1, alpha: float = 0.95) -> dict:
    """Score disjoint historical horizons using one fit from the initial window.

    ``alpha`` sets the VaR quantile. The observed violation-rate interval uses
    a separate fixed 95% confidence level and a binomial Wilson approximation.
    Forecast counts, rates and interval bounds are dimensionless.
    """
    prices = validate_prices(prices, require_dates=True)
    values = prices.to_numpy()
    if type(calibration_observations) is not int or calibration_observations < 3:
        raise ValueError("calibration_observations must be at least 3 prices (2 returns)")
    if type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 < alpha < 1 or 1 - alpha == 0):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if len(values) - calibration_observations < horizon_days:
        raise ValueError("insufficient evaluation prices for one complete horizon")

    training = prices.iloc[:calibration_observations]
    returns = log_returns(training)
    mu, sigma = estimate_gbm_params(training)
    daily_mean = float(returns.mean())
    daily_std = float(returns.std(ddof=1))
    if not np.isfinite([mu, sigma, daily_mean, daily_std]).all():
        raise ValueError("calibration exceeds finite precision")
    duration = horizon_days / TRADING_DAYS
    # Normal symmetry avoids rounding 1-alpha to 1 for tiny valid alpha.
    z = -NormalDist().inv_cdf(alpha)
    rows = []
    for origin in range(calibration_observations - 1,
                        len(values) - horizon_days, horizon_days):
        target = origin + horizon_days
        start = float(values[origin])
        observed = float(values[target])
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean = float(start * np.exp(mu * duration))
                median = float(start * np.exp((mu - sigma**2 / 2) * duration))
                lower = float(start * np.exp((mu - sigma**2 / 2) * duration
                                             + sigma * np.sqrt(duration) * z))
                error = observed - mean
                var = start - lower
                realized_loss = start - observed
                percentage_error = 100 * abs(error) / start
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("forecast exceeds finite precision") from exc
        if not np.isfinite([mean, median, lower, error, var, realized_loss,
                            percentage_error]).all():
            raise ValueError("forecast exceeds finite precision")
        rows.append({"origin_date": prices.index[origin].isoformat(),
                     "origin_price": start, "target_date": prices.index[target].isoformat(),
                     "observed_price": observed, "predicted_mean": mean,
                     "predicted_median": median, "lower_quantile": lower,
                     "var": var, "realized_loss": realized_loss,
                     "var_violation": bool(realized_loss > var), "error": error,
                     "absolute_percentage_error": percentage_error})

    errors = np.array([row["error"] for row in rows])
    violations = sum(row["var_violation"] for row in rows)
    metrics = {"forecast_count": len(rows), "violation_count": violations,
               "observed_violation_rate": violations / len(rows),
               "nominal_violation_rate": 1 - alpha,
               "mean_bias": float(np.mean(errors)),
               "mae": float(np.mean(np.abs(errors))),
               "rmse": float(np.sqrt(np.mean(np.square(errors)))),
               "mae_percent": float(np.mean([row["absolute_percentage_error"] for row in rows]))}
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("aggregate metrics exceed finite precision")
    warnings = []
    if len(rows) < 30:
        warnings.append("Fewer than 30 nonoverlapping forecasts: observed coverage is imprecise.")
    if sigma == 0:
        warnings.append("Zero volatility is a point mass: strict model violation probability is 0, not 1-alpha.")
    return {
        "method": "fixed_initial_calibration_nonoverlapping_historical_forecasts",
        "calibration": {"first_date": training.index[0].isoformat(),
                        "last_date": training.index[-1].isoformat(),
                        "price_count": len(training), "return_count": len(returns),
                        "mean_daily_log_return": daily_mean,
                        "std_daily_log_return": daily_std,
                        "mu": mu, "sigma": sigma,
                        "trading_days_per_year": TRADING_DAYS, "log_return_std_ddof": 1},
        "evaluation": {"first_origin_date": rows[0]["origin_date"],
                       "last_target_date": rows[-1]["target_date"],
                       "horizon_trading_days": horizon_days,
                       "horizon_years": duration,
                       "unscored_tail_observations": (len(values) - calibration_observations) % horizon_days,
                       "rows": rows, "metrics": metrics, "warnings": warnings,
                       "violation_rate_interval": _violation_rate_interval(violations, len(rows))},
        "protocol": {"calibration_observations": calibration_observations,
                     "horizon_days": horizon_days, "alpha": alpha,
                     "origin_policy": "last calibration close, then advance by horizon_days",
                     "var_violation": "realized_loss > var (strict)",
                     "point_error": "observed_price - predicted_mean",
                     "mae_percent": "mean(100 * abs(error) / origin_price)",
                     "coverage_interpretation": "historical observed coverage only"},
    }
