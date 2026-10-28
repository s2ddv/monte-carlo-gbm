"""Chronological GBM diagnostics with fixed, rolling or expanding calibration."""

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


def _calibration_record(training: pd.Series) -> dict:
    """Fit daily log returns; record price drift/year and volatility/sqrt(year)."""
    returns = log_returns(training)
    mu, sigma = estimate_gbm_params(training)
    daily_mean = float(returns.mean())
    daily_std = float(returns.std(ddof=1))
    if not np.isfinite([mu, sigma, daily_mean, daily_std]).all():
        raise ValueError("calibration exceeds finite precision")
    return {"first_date": training.index[0].isoformat(),
            "last_date": training.index[-1].isoformat(),
            "price_count": len(training), "return_count": len(returns),
            "mean_daily_log_return": daily_mean,
            "std_daily_log_return": daily_std, "mu": mu, "sigma": sigma,
            "trading_days_per_year": TRADING_DAYS, "log_return_std_ddof": 1}


def evaluate_out_of_sample(prices: pd.Series, *, calibration_observations: int,
                           horizon_days: int = 1, alpha: float = 0.95,
                           calibration_mode: str = "fixed",
                           recalibration_frequency: int = 1) -> dict:
    """Score disjoint historical horizons with causal analytical GBM forecasts.

    With N calibration prices and horizon h observed trading sessions, origins
    are N-1+k*h and targets origin+h. An incomplete final horizon is unscored.
    ``fixed`` fits the first N prices once. ``rolling`` fits the latest N
    prices through the origin, while ``expanding`` fits every price through
    the origin. Refit at forecast indices 0, frequency, 2*frequency, ...;
    intervening forecasts reuse parameters but use the current origin price.
    Frequency is measured in forecast origins, not calendar or trading days.
    Fixed calibration requires frequency=1 and preserves the legacy output.

    ``alpha`` sets the VaR quantile. The observed violation-rate interval uses
    a separate fixed 95% confidence level and a binomial Wilson approximation.
    Forecast counts, rates and interval bounds are dimensionless. Prices,
    errors and VaR are in the asset currency; time is h/252 years. No RNG or
    trajectory matrix is needed; working arrays scale with input history and
    forecast count, with one calibration window processed at a time.
    """
    prices = validate_prices(prices, require_dates=True)
    values = prices.to_numpy()
    if type(calibration_observations) is not int or calibration_observations < 3:
        raise ValueError("calibration_observations must be at least 3 prices (2 returns)")
    if type(horizon_days) is not int or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer")
    if not isinstance(calibration_mode, str) or calibration_mode not in (
            "fixed", "rolling", "expanding"):
        raise ValueError("calibration_mode must be fixed, rolling or expanding")
    if type(recalibration_frequency) is not int or recalibration_frequency < 1:
        raise ValueError("recalibration_frequency must be a positive integer")
    if calibration_mode == "fixed" and recalibration_frequency != 1:
        raise ValueError("fixed calibration requires recalibration_frequency=1")
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 < alpha < 1 or 1 - alpha == 0):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    if len(values) - calibration_observations < horizon_days:
        raise ValueError("insufficient evaluation prices for one complete horizon")

    initial_calibration = _calibration_record(prices.iloc[:calibration_observations])
    calibration = initial_calibration
    duration = horizon_days / TRADING_DAYS
    # Normal symmetry avoids rounding 1-alpha to 1 for tiny valid alpha.
    z = -NormalDist().inv_cdf(alpha)
    rows = []
    zero_volatility = False
    origins = range(calibration_observations - 1,
                    len(values) - horizon_days, horizon_days)
    for forecast_index, origin in enumerate(origins):
        recalibrated = forecast_index == 0
        if (calibration_mode != "fixed" and forecast_index > 0
                and forecast_index % recalibration_frequency == 0):
            first = origin + 1 - calibration_observations if calibration_mode == "rolling" else 0
            calibration = _calibration_record(prices.iloc[first:origin + 1])
            recalibrated = True
        mu, sigma = calibration["mu"], calibration["sigma"]
        zero_volatility |= sigma == 0
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
        if calibration_mode != "fixed":
            rows[-1]["calibration"] = dict(calibration)
            rows[-1]["calibration_reused"] = not recalibrated

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
    if zero_volatility:
        warnings.append("Zero volatility is a point mass: strict model violation probability is 0, not 1-alpha.")
    if calibration_mode != "fixed":
        warnings.append(
            "Recalibration windows share observations; nonoverlapping forecast horizons do not "
            "guarantee independent violations or a constant violation probability. The Wilson "
            "interval is descriptive under its stated binomial assumptions.")
    result = {
        "method": "fixed_initial_calibration_nonoverlapping_historical_forecasts",
        "calibration": initial_calibration,
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
    if calibration_mode != "fixed":
        result["method"] = f"{calibration_mode}_calibration_nonoverlapping_historical_forecasts"
        result["protocol"].update({
            "calibration_mode": calibration_mode,
            "recalibration_frequency": recalibration_frequency,
            "recalibration_unit": "forecast_origins",
            "calibration_window": ("latest calibration_observations prices through refit origin"
                                   if calibration_mode == "rolling" else
                                   "all prices from history start through refit origin"),
            "initial_calibration": "top-level calibration describes the initial fit",
            "incomplete_horizon_policy": "count remaining observations without scoring"})
    return result
