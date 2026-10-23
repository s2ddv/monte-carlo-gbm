"""Small checks for the public GBM path contract."""

import numpy as np

from simulation.gbm import simulate_gbm


def test_zero_volatility_follows_exact_deterministic_solution():
    paths = simulate_gbm(100, 0.08, 0, 1, 4, 3, seed=42)
    expected = 100 * np.exp(0.08 * np.arange(5) / 4)
    assert paths.shape == (5, 3)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape))


def test_same_seed_reproduces_paths():
    first = simulate_gbm(100, 0.08, 0.2, 1, 4, 5, seed=42)
    second = simulate_gbm(100, 0.08, 0.2, 1, 4, 5, seed=42)
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)
