"""Risk metrics calculated from simulated terminal prices."""

import numpy as np


def risk_summary(paths: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Summarize paths shaped (steps + 1, n_sims), including initial prices.

    Loss is initial minus terminal price, in price units: negative means a
    gain. VaR is the loss percentile at ``confidence``; CVaR is the mean of
    losses strictly above VaR. If that tail is empty (e.g. identical outcomes),
    CVaR equals VaR. Neither metric is clipped at zero.

    The expected price estimates s0 * exp(mu * horizon), subject to sampling
    error. Drift and horizon cannot be inferred from paths alone.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] == 0:
        raise ValueError("paths must contain an initial row and terminal prices.")
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
    tail_losses = losses[losses > var]
    cvar = float(tail_losses.mean()) if tail_losses.size else var

    summary = {
        "expected_price": float(final_prices.mean()),
        "prob_loss": float(np.mean(final_prices < s0)),
        "var": var,
        "cvar": cvar,
    }
    assert summary["cvar"] >= summary["var"]
    assert 0 <= summary["prob_loss"] <= 1
    return summary
