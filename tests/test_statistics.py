"""Deterministic examples and analytical checks for terminal risk metrics."""

import unittest

import numpy as np

from simulation.gbm import simulate_gbm
from risk.statistics import risk_summary


class RiskSummaryTests(unittest.TestCase):
    def test_uses_terminal_row_and_strict_loss_threshold(self):
        paths = np.array([[100] * 5, [500] * 5, [80, 90, 100, 110, 120]])
        summary = risk_summary(paths)
        self.assertEqual(summary["expected_price"], 100)
        self.assertEqual(summary["prob_loss"], 0.4)
        self.assertAlmostEqual(summary["var"], 18)
        self.assertEqual(summary["cvar"], 20)

    def test_custom_confidence_includes_losses_equal_to_var(self):
        paths = np.array([[100] * 5, [80, 90, 100, 110, 120]])
        summary = risk_summary(paths, confidence=0.5)
        self.assertEqual(summary["var"], 0)
        self.assertEqual(summary["cvar"], 10)

    def test_cvar_includes_all_ties_at_var(self) -> None:
        paths = np.array([[100] * 5, [80, 90, 90, 90, 110]])
        summary = risk_summary(paths, confidence=0.5)
        self.assertEqual(summary["var"], 10)
        self.assertEqual(summary["cvar"], 12.5)

    def test_negative_var_is_preserved(self):
        summary = risk_summary([[100] * 5, [110, 120, 130, 140, 150]])
        self.assertAlmostEqual(summary["var"], -12)
        self.assertEqual(summary["cvar"], -10)
        self.assertEqual(summary["prob_loss"], 0)

    def test_identical_outcomes_and_single_simulation(self):
        for final_price in (90, 100, 110):
            for n_sims in (1, 1000):
                with self.subTest(final_price=final_price, n_sims=n_sims):
                    summary = risk_summary([[100] * n_sims, [final_price] * n_sims])
                    self.assertEqual(summary["var"], 100 - final_price)
                    self.assertEqual(summary["cvar"], summary["var"])

    def test_gbm_mean_and_invariants_for_1000_paths(self):
        s0, mu, sigma = 100, 0.04, 0.2
        for horizon in (0.5, 1):
            with self.subTest(horizon=horizon):
                paths = simulate_gbm(s0, mu, sigma, horizon, 252, 1000, seed=42)
                summary = risk_summary(paths)
                theoretical_mean = s0 * np.exp(mu * horizon)
                standard_error = theoretical_mean * np.sqrt(
                    np.expm1(sigma**2 * horizon) / 1000
                )
                self.assertLess(abs(summary["expected_price"] - theoretical_mean),
                                4 * standard_error)
                self.assertGreaterEqual(summary["cvar"], summary["var"])
                self.assertTrue(0 <= summary["prob_loss"] <= 1)
                self.assertTrue(0.4 <= summary["prob_loss"] <= 0.5)

    def test_invalid_confidence(self):
        for confidence in (-0.1, 0, 1, 1.1, np.nan):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                risk_summary([[100], [90]], confidence)

    def test_invalid_paths(self):
        for paths in ([], [100, 90], [[100]], [[], []],
                      [[100, 101], [90, 90]], [[100], [np.nan]],
                      [[100], [np.inf]]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                risk_summary(paths)


if __name__ == "__main__":
    unittest.main()
