"""Risk metrics calculated from simulated terminal prices."""

import numpy as np


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
        ValueError: If paths or confidence are invalid.
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
    losses = s0 - final_prices
    var = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean())

    return {
        "expected_price": float(final_prices.mean()),
        "prob_loss": float(np.mean(final_prices < s0)),
        "var": var,
        "cvar": cvar,
    }
