"""Analytical terminal GBM benchmarks and nested-sample convergence."""

from statistics import NormalDist

import numpy as np

QUANTILES = (0.05, 0.5, 0.95)


def terminal_theory(s0, mu, sigma, horizon):
    """Moments and quantiles of S(T), including the zero-volatility limit."""
    if (not np.isfinite([s0, mu, sigma, horizon]).all()
            or s0 <= 0 or sigma < 0 or horizon < 0):
        raise ValueError("Require finite parameters, s0 > 0, sigma/horizon >= 0.")
    with np.errstate(over="raise", invalid="raise"):
        mean = float(s0 * np.exp(mu * horizon))
        variance = float(mean**2 * np.expm1(sigma**2 * horizon))
        quantiles = {
            str(p): float(s0 * np.exp((mu - sigma**2 / 2) * horizon
                         + sigma * np.sqrt(horizon) * NormalDist().inv_cdf(p)))
            for p in QUANTILES
        }
    result = {"mean": mean, "variance": variance, "quantiles": quantiles}
    if not np.isfinite([mean, variance, *quantiles.values()]).all():
        raise ValueError("Theoretical terminal statistics exceed finite precision.")
    return result


def _comparison(estimate, theoretical):
    error = None if estimate is None else abs(estimate - theoretical)
    return {"estimate": estimate, "theoretical": theoretical,
            "absolute_error": error,
            "relative_error": (error / abs(theoretical)
                               if error is not None and theoretical != 0 else None)}


def compare_terminal(terminal, theory):
    """Use unbiased sample variance (ddof=1) and linear empirical quantiles."""
    terminal = np.asarray(terminal, dtype=float)
    if (terminal.ndim != 1 or terminal.size == 0
            or not np.isfinite(terminal).all() or np.any(terminal <= 0)):
        raise ValueError("terminal must be a nonempty vector of finite positive prices.")
    return {
        "mean": _comparison(float(terminal.mean()), theory["mean"]),
        "variance": _comparison(float(terminal.var(ddof=1))
                                if terminal.size > 1 else None, theory["variance"]),
        "quantiles": {
            str(p): _comparison(float(np.quantile(terminal, p, method="linear")),
                                theory["quantiles"][str(p)]) for p in QUANTILES
        },
    }


def convergence_analysis(terminal, theory):
    """Compare increasing prefixes of ONE run, not separately reseeded runs.

    The reference SE is the exact RMSE of the sample mean under fixed GBM
    parameters, not an envelope or a guarantee of monotonic observed error.
    """
    terminal = np.asarray(terminal, dtype=float)
    compare_terminal(terminal, theory)  # Validate before constructing checkpoints.
    counts = np.unique(np.geomspace(min(2, terminal.size), terminal.size,
                                   num=min(12, terminal.size)).astype(int))
    counts = np.unique(np.append(counts, terminal.size))
    return [{"n_sims": int(n), **compare_terminal(terminal[:n], theory),
             "mean_standard_error": float(np.sqrt(theory["variance"] / n))}
            for n in counts]
