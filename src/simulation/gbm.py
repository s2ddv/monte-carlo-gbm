"""Normal GBM paths and a heavy-tailed log-return scenario."""

from typing import Literal
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from calibration.estimation import TRADING_DAYS
from calibration.prices import validate_prices
from simulation.base import BaseStochasticSimulator

DistributionType = Literal["normal", "t_student"]


class GbmSimulator(BaseStochasticSimulator):
    """Simulate log-price increments with normal or standardized t shocks.

    ``mu`` and ``sigma`` are annualized. The normal mode is exact GBM.
    The t mode substitutes independent standardized t shocks at each grid
    step; its distribution depends on the step size and is not exact GBM.
    """

    def __init__(self, s0: float, horizon: float, steps: int, n_sims: int,
                 seed: int | None = None, *, mu: float | None = None,
                 sigma: float | None = None) -> None:
        super().__init__(s0, horizon, steps, n_sims, seed)
        self.mu: float | None = None
        self.sigma: float | None = None
        self.calibrated_df: float | None = None
        if (mu is None) != (sigma is None):
            raise ValueError("mu and sigma must be supplied together")
        if mu is not None and sigma is not None:
            if (isinstance(mu, bool) or isinstance(sigma, bool)
                    or not isinstance(mu, Real) or not isinstance(sigma, Real)
                    or not np.isfinite(mu) or not np.isfinite(sigma) or sigma < 0):
                raise ValueError("mu and sigma must be finite; sigma must be nonnegative")
            self.mu, self.sigma = float(mu), float(sigma)

    def calibrate(self, historical_prices: pd.Series) -> None:
        """Fit annualized normal moments and t degrees of freedom to daily logs.

        The t fit estimates location and scale as nuisance parameters. The
        simulation retains the classical sample-moment ``mu`` and ``sigma``.
        """
        historical_prices = validate_prices(historical_prices, min_observations=3)
        values = historical_prices.to_numpy()

        returns = np.diff(np.log(values))
        if not np.isfinite(returns).all():
            raise ValueError("log returns exceed finite precision")
        daily_sigma = float(np.std(returns, ddof=1))
        sigma = float(np.sqrt(TRADING_DAYS) * daily_sigma)
        mu = float(TRADING_DAYS * np.mean(returns) + 0.5 * sigma**2)
        if not np.isfinite([mu, sigma]).all():
            raise ValueError("calibrated moments exceed finite precision")
        if daily_sigma == 0:
            # A point mass has no identifiable t degrees of freedom.
            self.mu, self.sigma, self.calibrated_df = mu, sigma, None
            return
        df, _, _ = student_t.fit(returns / daily_sigma)
        if not np.isfinite(df) or df <= 0:
            raise ValueError("t fit returned invalid degrees of freedom")
        self.mu, self.sigma, self.calibrated_df = mu, sigma, float(df)

    def generate_paths(self, dist_type: DistributionType = "normal") -> np.ndarray:
        """Return ``(steps + 1, n_sims)`` positive prices.

        The output holds the working shocks and cumulative log returns in
        place. The t mode requires a calibrated ``df > 2`` and rescales the
        shocks to unit variance before applying the annualized volatility.
        Unrepresentable increments or prices raise ``ValueError``. In t
        mode ``mu`` controls the log-return location through the normal-mode
        convention; it is not a finite expected-price growth rate.
        """
        if dist_type not in ("normal", "t_student"):
            raise ValueError("dist_type must be 'normal' or 't_student'")
        if self.mu is None or self.sigma is None:
            raise ValueError("calibrate or supply mu and sigma before generating paths")
        if (dist_type == "t_student" and
                (isinstance(self.calibrated_df, bool)
                 or not isinstance(self.calibrated_df, Real)
                 or not np.isfinite(self.calibrated_df)
                 or self.calibrated_df <= 2)):
            raise ValueError("t_student requires calibrated degrees of freedom above 2")

        # Preserve the normal mode's arithmetic order for existing replays.
        # For extreme annual parameters, sigma**2 can overflow even when
        # sigma**2 * dt is representable. Retry using the per-step scale.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scale = float(self.sigma * np.sqrt(self.dt))
            try:
                drift = (self.mu - 0.5 * self.sigma**2) * self.dt
            except OverflowError:
                drift = float("nan")
            if not np.isfinite(drift):
                drift = float(self.mu * self.dt - 0.5 * np.float64(scale) ** 2)
            if dist_type == "t_student":
                df = self.calibrated_df
                assert df is not None
                # Form the bounded variance factor first: dt * (df - 2)
                # may overflow even though the final t scale is finite.
                scale *= float(np.sqrt((df - 2) / df))
        if not np.isfinite(drift) or not np.isfinite(scale):
            raise ValueError("simulation increments exceed finite precision")

        # NumPy's t sampler returns a separate full shock matrix; the normal
        # sampler writes directly into the output. Check before either allocates.
        self._check_memory(temporary_arrays=int(dist_type == "t_student"))

        paths = np.empty((self.steps + 1, self.n_sims), dtype=np.float64)
        paths[0].fill(self.s0)
        increments = paths[1:]
        if dist_type == "normal":
            self.rng.standard_normal(size=increments.shape, out=increments)
        else:
            df = self.calibrated_df
            assert df is not None
            increments[:] = self.rng.standard_t(df=df, size=increments.shape)

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            np.multiply(increments, scale, out=increments)
            increments += drift
            np.cumsum(increments, axis=0, out=increments)
            # Only move the initial price into log space when exp alone would
            # overflow or enter the subnormal range before multiplication.
            # Keep the usual arithmetic order for normal-sized replays.
            if (np.max(increments) >= np.log(np.finfo(np.float64).max)
                    or np.min(increments) <= np.log(np.finfo(np.float64).tiny)):
                increments += np.log(self.s0)
                np.exp(increments, out=increments)
            else:
                np.exp(increments, out=increments)
                increments *= self.s0
        # Scalar reductions avoid another full-size boolean mask after the
        # output has been filled. NaN and either infinity fail these checks.
        min_price, max_price = np.min(paths), np.max(paths)
        if (not np.isfinite(min_price) or not np.isfinite(max_price)
                or min_price <= 0):
            raise ValueError("simulated prices exceed finite positive precision")
        return paths


def simulate_gbm(s0: float, mu: float, sigma: float, horizon: float,
                 steps: int, n_sims: int, seed: int | None = None) -> np.ndarray:
    """Compatibility entry point for the existing normal GBM pipeline."""
    return GbmSimulator(s0, horizon, steps, n_sims, seed, mu=mu,
                        sigma=sigma).generate_paths()
