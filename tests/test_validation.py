"""Independent analytical benchmarks and deterministic convergence checks."""

import numpy as np
import pytest
from scipy.stats import lognorm

from simulation.gbm import simulate_gbm
from validation.statistics import (QUANTILES, compare_terminal,
                                  convergence_analysis, terminal_theory)


@pytest.mark.parametrize("s0,mu,sigma,horizon", [
    (100, 0.08, 0.2, 1), (42, -0.1, 0.6, 2.5), (1, 0, 0.01, 0.25),
])
def test_theory_matches_independent_scipy_distribution(s0, mu, sigma, horizon):
    distribution = lognorm(s=sigma * np.sqrt(horizon),
                           scale=s0 * np.exp((mu - sigma**2 / 2) * horizon))
    theory = terminal_theory(s0, mu, sigma, horizon)
    assert theory["mean"] == pytest.approx(distribution.mean())
    assert theory["variance"] == pytest.approx(distribution.var())
    for p in QUANTILES:
        assert theory["quantiles"][str(p)] == pytest.approx(distribution.ppf(p))


@pytest.mark.parametrize("sigma,horizon", [(0, 2), (0.3, 0)])
def test_degenerate_distribution(sigma, horizon):
    theory = terminal_theory(100, 0.04, sigma, horizon)
    expected = 100 * np.exp(0.04 * horizon)
    assert theory["mean"] == pytest.approx(expected)
    assert theory["variance"] == 0
    assert list(theory["quantiles"].values()) == pytest.approx([expected] * 3)
    comparison = compare_terminal([expected] * 5, theory)
    assert comparison["variance"]["absolute_error"] == pytest.approx(0, abs=1e-20)
    assert comparison["variance"]["relative_error"] is None


def test_explicit_sample_moments_quantiles_and_errors():
    theory = terminal_theory(3, 0, 0, 1)
    result = compare_terminal([1, 2, 3, 4, 5], theory)
    assert result["mean"]["estimate"] == 3
    assert result["variance"]["estimate"] == 2.5
    assert result["variance"]["absolute_error"] == 2.5
    assert result["quantiles"]["0.05"]["estimate"] == pytest.approx(1.2)
    assert result["quantiles"]["0.5"]["estimate"] == 3
    assert result["quantiles"]["0.95"]["estimate"] == pytest.approx(4.8)
    assert result["quantiles"]["0.95"]["relative_error"] == pytest.approx(0.6)


def test_nested_convergence_and_theoretical_standard_error():
    terminal = np.arange(1, 101, dtype=float)
    theory = terminal_theory(50, 0, 0.3, 1)
    rows = convergence_analysis(terminal, theory)
    counts = [row["n_sims"] for row in rows]
    assert counts == sorted(set(counts))
    assert counts[0] == 2
    assert counts[-1] == 100
    for row in rows:
        n = row["n_sims"]
        assert row["mean"]["estimate"] == pytest.approx((n + 1) / 2)
        assert row["variance"]["estimate"] == pytest.approx(n * (n + 1) / 12)
        assert row["mean_standard_error"] == pytest.approx(np.sqrt(theory["variance"] / n))
    assert rows[-1]["mean_standard_error"] / rows[0]["mean_standard_error"] == pytest.approx(np.sqrt(2/100))
    assert rows[-1]["quantiles"] == compare_terminal(terminal, theory)["quantiles"]


def test_single_simulation_has_undefined_sample_variance():
    theory = terminal_theory(100, 0, 0, 1)
    rows = convergence_analysis([100], theory)
    assert len(rows) == 1
    assert rows[0]["n_sims"] == 1
    assert rows[0]["variance"]["estimate"] is None
    assert rows[0]["variance"]["absolute_error"] is None
    assert rows[0]["mean_standard_error"] == 0


def test_seeded_simulator_agrees_with_terminal_distribution():
    theory = terminal_theory(100, 0.08, 0.2, 1)
    paths = simulate_gbm(100, 0.08, 0.2, 1, 4, 100_000, seed=73)
    result = compare_terminal(paths[-1], theory)
    assert result["mean"]["absolute_error"] < 4 * np.sqrt(theory["variance"] / 100_000)
    assert result["variance"]["relative_error"] < 0.03
    assert all(value["relative_error"] < 0.01 for value in result["quantiles"].values())


@pytest.mark.parametrize("terminal", [[], [[1]], [0], [-1], [np.nan], [np.inf]])
def test_invalid_terminal_samples(terminal):
    with pytest.raises(ValueError):
        convergence_analysis(terminal, terminal_theory(100, 0, 0.2, 1))


@pytest.mark.parametrize("parameters", [
    (0, 0, 0.2, 1), (100, np.nan, 0.2, 1),
    (100, 0, -0.2, 1), (100, 0, 0.2, -1), (100, 0, np.inf, 1),
])
def test_invalid_theoretical_parameters(parameters):
    with pytest.raises(ValueError):
        terminal_theory(*parameters)
