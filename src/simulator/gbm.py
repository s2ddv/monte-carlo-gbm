"""Geometric Brownian Motion (GBM) price path simulation."""

import numpy as np


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    horizon: float,
    steps: int,
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate asset price paths under Geometric Brownian Motion.

    Returns an array of shape (steps + 1, n_sims). Row 0 holds the
    initial price; each column is one simulated path.
    """
    dt = horizon / steps

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((steps, n_sims))

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    cumulative = np.cumsum(log_returns, axis=0)

    prices = s0 * np.exp(cumulative)
    paths = np.vstack([np.full((1, n_sims), s0), prices])

    return paths