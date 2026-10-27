"""Offline terminal-only GBM contracts, including Monte Carlo uncertainty."""

from decimal import Decimal, localcontext
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from risk.statistics import risk_summary
from simulation.gbm import GbmSimulator, simulate_terminal_batched


class TerminalGbmTests(unittest.TestCase):
    def test_shape_seed_batches_and_local_rng(self):
        options = dict(s0=100, mu=0.07, sigma=0.2, horizon=1.2,
                       n_sims=23, seed=19, batch_size=7)
        global_before = np.random.get_state()
        first = simulate_terminal_batched(**options)
        second = simulate_terminal_batched(**options)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (23,))
        self.assertEqual(first.dtype, np.dtype("float64"))
        self.assertTrue(np.all(first > 0))
        global_after = np.random.get_state()
        for before, after in zip(global_before, global_after):
            np.testing.assert_equal(before, after)

        # One terminal draw per price, including the final partial batch.
        z = np.random.default_rng(19).standard_normal(23)
        expected = 100 * np.exp((0.07 - 0.2**2 / 2) * 1.2
                                + 0.2 * np.sqrt(1.2) * z)
        np.testing.assert_allclose(first, expected, rtol=2e-15)

    def test_instance_rng_advances_and_grid_is_irrelevant(self):
        model = GbmSimulator(100, 1, 252, 21, seed=13, mu=0.04, sigma=0.2)
        replica = GbmSimulator(100, 1, 1, 21, seed=13, mu=0.04, sigma=0.2)
        first = model.generate_terminal(batch_size=8)
        second = model.generate_terminal(batch_size=8)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, replica.generate_terminal(batch_size=8))
        np.testing.assert_array_equal(second, replica.generate_terminal(batch_size=8))

    def test_array_budget_rejects_before_allocation_or_rng_and_accepts_boundary(self):
        model = GbmSimulator(100, 1, 252, 13, seed=13, mu=0.04, sigma=0.2)
        before = model.rng.bit_generator.state
        with patch("simulation.gbm.np.empty", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "requires 104 bytes"):
                model.generate_terminal(batch_size=5, max_working_bytes=103)
        self.assertEqual(before, model.rng.bit_generator.state)
        # The output is the only large array; a full paths matrix would not fit.
        result = model.generate_terminal(batch_size=5, max_working_bytes=104)
        self.assertEqual(result.nbytes, 104)
        singleton = simulate_terminal_batched(100, 0, 0, 1, 1,
                                              batch_size=100, max_working_bytes=8)
        np.testing.assert_array_equal(singleton, [100.0])

    def test_invalid_controls_and_uncalibrated_model_leave_rng_untouched(self):
        model = GbmSimulator(100, 1, 2, 3, seed=19, mu=0.04, sigma=0.2)
        before = model.rng.bit_generator.state
        for name in ("batch_size", "max_working_bytes"):
            for value in (0, -1, True, 1.5, np.nan, np.inf, "10"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        model.generate_terminal(**{name: value})
                    self.assertEqual(before, model.rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError, "calibrate or supply"):
            GbmSimulator(100, 1, 2, 3).generate_terminal()

    def test_public_parameters_are_validated(self):
        good = dict(s0=100, mu=0.04, sigma=0.2, horizon=1, n_sims=3, seed=1)
        for name, value in (("s0", 0), ("s0", np.inf), ("mu", np.nan),
                            ("sigma", -1), ("sigma", np.inf), ("horizon", 0),
                            ("n_sims", True), ("n_sims", 1.5), ("seed", -1)):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    simulate_terminal_batched(**(good | {name: value}))

    def test_mean_variance_and_quantiles_against_analytic_sampling_error(self):
        s0, mu, sigma, horizon, n = 100, 0.07, 0.25, 1.2, 80_000
        terminal = simulate_terminal_batched(s0, mu, sigma, horizon, n,
                                             seed=2026, batch_size=4093)
        # 640,000 output bytes. Tolerances are six analytic sampling SEs,
        # deliberately generous for fixed-seed, finite Monte Carlo checks.
        raw = [s0**k * np.exp(k * mu * horizon
                              + k * (k - 1) / 2 * sigma**2 * horizon)
               for k in range(1, 5)]
        mean = raw[0]
        variance = raw[1] - mean**2
        central_fourth = raw[3] - 4 * mean * raw[2] + 6 * mean**2 * raw[1] - 3 * mean**4
        mean_se = np.sqrt(variance / n)
        variance_se = np.sqrt((central_fourth - (n - 3) / (n - 1) * variance**2) / n)
        self.assertLess(abs(terminal.mean() - mean), 6 * mean_se)
        self.assertLess(abs(terminal.var(ddof=1) - variance), 6 * variance_se)

        probabilities = np.array([0.05, 0.5, 0.95])
        z = norm.ppf(probabilities)
        quantiles = s0 * np.exp((mu - sigma**2 / 2) * horizon
                                + sigma * np.sqrt(horizon) * z)
        density = norm.pdf(z) / (quantiles * sigma * np.sqrt(horizon))
        quantile_se = np.sqrt(probabilities * (1 - probabilities) / n) / density
        self.assertTrue(np.all(np.abs(np.quantile(terminal, probabilities) - quantiles)
                               < 6 * quantile_se))

    def test_deterministic_prices_preserve_loss_var_and_cvar_conventions(self):
        terminal = simulate_terminal_batched(100, 0.1, 0, 2, 11, seed=19, batch_size=3)
        expected = 100 * np.exp(0.2)
        np.testing.assert_allclose(terminal, expected, rtol=1e-15)
        summary = risk_summary(np.vstack((np.full(11, 100), terminal)))
        self.assertAlmostEqual(summary["var"], 100 - expected)
        self.assertAlmostEqual(summary["cvar"], 100 - expected)
        self.assertLess(summary["var"], 0)
        self.assertEqual(summary["prob_loss"], 0)

    def test_extreme_representable_prices_and_scaled_volatility(self):
        with localcontext() as context:
            context.prec = 80
            for initial, drift in ((1e-300, 800), (1e300, -800)):
                expected = float(Decimal(str(initial)) * Decimal(drift).exp())
                terminal = simulate_terminal_batched(initial, drift, 0, 1, 3, seed=0)
                np.testing.assert_allclose(terminal, expected, rtol=1e-12)
        with np.errstate(all="raise"):
            terminal = simulate_terminal_batched(100, 0, 1e160, 1e-320, 5, seed=19)
        self.assertTrue(np.isfinite(terminal).all() and np.all(terminal > 0))

    def test_unrepresentable_prices_and_increments_raise(self):
        for drift in (-1000, 1000):
            with self.subTest(drift=drift):
                with self.assertRaisesRegex(ValueError, "finite positive precision"):
                    simulate_terminal_batched(100, drift, 0, 1, 3, seed=0)
        model = GbmSimulator(100, 1, 1, 3, seed=0, mu=0, sigma=1e200)
        before = model.rng.bit_generator.state
        with self.assertRaisesRegex(ValueError, "log returns exceed finite precision"):
            model.generate_terminal()
        self.assertEqual(before, model.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
