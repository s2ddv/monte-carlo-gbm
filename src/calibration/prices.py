"""Shared validation of historical prices, independent of market providers."""

from datetime import date

import numpy as np
import pandas as pd


def validate_prices(prices: pd.Series, *, min_observations: int = 1,
                    require_dates: bool = False) -> pd.Series:
    """Return float prices without sorting, imputing or dropping observations.

    Prices must be positive finite real numbers. Datetime/period/date indices
    must be strictly increasing, unique and nonmissing. Undated Series remain
    supported by numerical calibration; provider and snapshot boundaries may
    require a DatetimeIndex. The index and Series name are preserved.
    """
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series")
    index = prices.index
    dated = (isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex))
             or index.inferred_type in {"date", "datetime", "datetime64"}
             # Missing values can make an object date index infer as "mixed".
             or (index.inferred_type.startswith("mixed")
                 and any(isinstance(value, (date, np.datetime64, pd.Period))
                         for value in index)))
    if ((require_dates and not isinstance(index, pd.DatetimeIndex))
            or (dated and (index.hasnans or not index.is_unique
                          or not index.is_monotonic_increasing))):
        raise ValueError("Invalid price dates: require unique increasing ordered dates.")
    if len(prices) < min_observations:
        minimum = "three" if min_observations == 3 else str(min_observations)
        raise ValueError(f"Insufficient price history: require at least {minimum} prices.")
    try:
        # Casting complex values to float would silently discard the imaginary part.
        if np.iscomplexobj(prices.to_numpy()):
            raise ValueError("complex prices")
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid prices: require positive finite real numbers.") from exc
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid prices: require positive finite real numbers.")
    return pd.Series(values, index=index, name=prices.name)
