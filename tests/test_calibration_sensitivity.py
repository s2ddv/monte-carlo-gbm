import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from validation.calibration_sensitivity import calibration_sensitivity


def test_scenario_grid_matches_independent_normal_reference():
    result = calibration_sensitivity(123.0, -0.03, 0.28, 1.75, alpha=0.975,
                                     mu_delta=0.04, sigma_relative_delta=0.25)
    assert len(result["scenarios"]) == 9
    assert result["baseline"] == result["scenarios"][4]
    assert result["baseline"]["mu"] == -0.03
    assert result["baseline"]["sigma"] == 0.28
    assert sorted({row["mu"] for row in result["scenarios"]}) == pytest.approx(
        [-0.07, -0.03, 0.01])
    assert sorted({row["sigma"] for row in result["scenarios"]}) == pytest.approx(
        [0.21, 0.28, 0.35])
    for row in result["scenarios"]:
        location = (row["mu"] - row["sigma"] ** 2 / 2) * 1.75
        scale = row["sigma"] * math.sqrt(1.75)
        expected = 123.0 * math.exp(row["mu"] * 1.75)
        assert row["mean"] == pytest.approx(expected, rel=2e-14)
        for p in (0.05, 0.5, 0.95):
            oracle = 123.0 * math.exp(norm.ppf(p, loc=location, scale=scale))
            assert row["quantiles"][str(p)] == pytest.approx(oracle, rel=2e-14)
        oracle_lower = 123.0 * math.exp(norm.ppf(0.025, loc=location, scale=scale))
        assert row["lower_quantile"] == pytest.approx(oracle_lower, rel=2e-14)
        assert row["var"] == pytest.approx(123.0 - oracle_lower, abs=2e-12)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_drift_monotonicity_and_volatility_mean_invariance():
    result = calibration_sensitivity(100, 0.08, 0.2, 1)
    rows = result["scenarios"]
    for volatility_index in range(3):
        column = rows[volatility_index::3]
        assert column[0]["mean"] < column[1]["mean"] < column[2]["mean"]
        assert column[0]["var"] > column[1]["var"] > column[2]["var"]
        for p in ("0.05", "0.5", "0.95"):
            assert column[0]["quantiles"][p] < column[1]["quantiles"][p] < column[2]["quantiles"][p]
    for start in (0, 3, 6):
        triple = rows[start:start + 3]
        assert triple[0]["mean"] == triple[1]["mean"] == triple[2]["mean"]
        # At alpha >= 0.5, lower tail falls and loss VaR rises with sigma.
        assert triple[0]["var"] < triple[1]["var"] < triple[2]["var"]
    for metric in ("mean", "var"):
        assert result["ranges"][metric] == {
            "min": min(row[metric] for row in rows),
            "max": max(row[metric] for row in rows),
        }


@pytest.mark.parametrize("sigma,horizon", [(0, 1), (0.2, 0)])
def test_point_mass_and_negative_var(sigma, horizon):
    result = calibration_sensitivity(100, 0.1, sigma, horizon)
    for row in result["scenarios"]:
        expected = 100 * math.exp(row["mu"] * horizon)
        assert row["mean"] == pytest.approx(expected)
        assert list(row["quantiles"].values()) == pytest.approx([expected] * 3)
        assert row["var"] == pytest.approx(100 - expected)
        assert row["var"] <= 0


def test_zero_spreads_and_maximum_volatility_spread():
    fixed = calibration_sensitivity(100, 0.05, 0.2, 1, mu_delta=0,
                                    sigma_relative_delta=0)
    assert all(row == fixed["baseline"] for row in fixed["scenarios"])
    extreme = calibration_sensitivity(100, 0.05, 0.2, 1, sigma_relative_delta=1)
    assert extreme["scenarios"][0]["sigma"] == 0
    assert extreme["scenarios"][2]["sigma"] == 0.4


@pytest.mark.parametrize("name,value", [
    ("s0", 0), ("s0", -1), ("s0", True), ("s0", "100"),
    ("mu", float("nan")), ("mu", float("inf")), ("mu", None),
    ("sigma", -0.1), ("sigma", float("inf")), ("horizon", -1),
    ("horizon", False), ("alpha", 0), ("alpha", 1), ("alpha", True),
    ("mu_delta", -0.01), ("mu_delta", float("nan")),
    ("sigma_relative_delta", -0.1), ("sigma_relative_delta", 1.1),
    ("sigma_relative_delta", float("inf")), ("sigma", np.array([0.2])),
])
def test_rejects_invalid_parameters(name, value):
    arguments = dict(s0=100, mu=0.08, sigma=0.2, horizon=1)
    arguments[name] = value
    with pytest.raises(ValueError):
        calibration_sensitivity(**arguments)


@pytest.mark.parametrize("mu", [1000, -1000])
def test_rejects_price_overflow_and_underflow(mu):
    with pytest.raises(ValueError, match="finite precision"):
        calibration_sensitivity(100, mu, 0.2, 1)


def test_log_space_avoids_unnecessary_intermediate_overflow():
    result = calibration_sensitivity(1e-300, 710, 0, 1, mu_delta=0,
                                     sigma_relative_delta=0)
    assert result["baseline"]["mean"] == pytest.approx(math.exp(math.log(1e-300) + 710))
    # A large price can have unrepresentable variance while the requested
    # mean, quantiles and VaR are finite; no variance is needed here.
    large = calibration_sensitivity(1e200, 0, 0.2, 1)
    json.dumps(large, allow_nan=False)


def test_tiny_alpha_uses_normal_symmetry():
    result = calibration_sensitivity(100, 0.02, 0.1, 1, alpha=1e-20,
                                     mu_delta=0, sigma_relative_delta=0)
    oracle = 100 * math.exp(0.02 - 0.1 ** 2 / 2 + 0.1 * norm.isf(1e-20))
    assert result["baseline"]["lower_quantile"] == pytest.approx(oracle, rel=2e-14)
