"""Risk metrics calculated from simulated terminal prices."""

import numpy as np


def _finite_mean(values):
    """Average finite values without overflowing their intermediate sum."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
    if np.isfinite(mean):
        return mean
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return float(np.mean(values / scale) * scale)


def _linear_percentile(values, confidence):
    """NumPy linear percentile with safe interpolation across opposite signs."""
    rank = (values.size - 1) * (confidence * 100 / 100)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    selected = np.partition(values, (lower, upper))
    left, right = float(selected[lower]), float(selected[upper])
    weight = rank - lower
    difference = right - left
    if not np.isfinite(difference):
        return (1 - weight) * left + weight * right
    # Match NumPy's interpolation direction, including threshold ties.
    if weight >= 0.5:
        return right - difference * (1 - weight)
    return left + difference * weight


def risk_summary(paths: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Summarize terminal prices and losses relative to the initial price.

    Loss is initial minus terminal price: negative means a gain. CVaR is
    the mean of losses greater than or equal to VaR; neither is clipped.

    Args:
        paths: Finite prices shaped (steps + 1, n_sims), with a common s0.
        confidence: Loss percentile level, strictly between 0 and 1.

    Returns:
        Expected terminal price, probability of loss, VaR and CVaR.

    Raises:
        ValueError: If paths/confidence are invalid, or a loss or statistic
            cannot be represented as a finite float64 value.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] == 0:
        raise ValueError(
            "paths must be 2D with at least 2 rows and 1 simulation."
        )
    if not np.isfinite(paths).all():
        raise ValueError("paths must contain only finite prices.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1, exclusively.")

    s0 = paths[0, 0]
    if not np.all(paths[0] == s0):
        raise ValueError("All simulations must share the same initial price.")

    final_prices = paths[-1]
    with np.errstate(over="ignore", invalid="ignore"):
        losses = s0 - final_prices
    if not np.isfinite(losses).all():
        raise ValueError("Terminal losses exceed finite precision.")
    var = _linear_percentile(losses, confidence)
    tail_losses = losses[losses >= var]
    cvar = _finite_mean(tail_losses)

    result = {
        "expected_price": _finite_mean(final_prices),
        "prob_loss": float(np.mean(final_prices < s0)),
        "var": var,
        "cvar": cvar,
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Risk statistics exceed finite precision.")
    return result
