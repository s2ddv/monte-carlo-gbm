"""JSON simulation records with historical snapshots for offline replay."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from calibration.estimation import TRADING_DAYS, estimate_gbm_params
from simulation.gbm import simulate_gbm
from risk.statistics import risk_summary
from validation.statistics import compare_terminal, convergence_analysis, terminal_theory


@dataclass(frozen=True)
class SimulationConfig:
    horizon: float = 1.0
    steps: int = 252
    n_sims: int = 1000
    seed: int = 42
    confidence: float = 0.95

    def __post_init__(self):
        if not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("horizon must be positive and finite.")
        for name in ("steps", "n_sims", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid range.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be strictly between zero and one.")


def _digest(observations):
    return sha256(json.dumps(observations, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def historical_snapshot(prices, *, ticker, period, retrieved_at, source="yfinance"):
    """Record the exact adjusted closes, observation dates and retrieval time."""
    if (not isinstance(prices, pd.Series) or len(prices) < 3
            or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.hasnans or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy(dtype=float)).all()
            or (prices <= 0).any()):
        raise ValueError("Require at least 3 positive finite prices with unique ordered dates.")
    timestamp = pd.Timestamp(retrieved_at)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone.")
    observations = [{"date": date.isoformat(), "close": float(price)}
                    for date, price in prices.items()]
    return {"ticker": ticker, "requested_period": period, "source": source,
            "price_field": "adjusted_close", "interval": "1d",
            "retrieved_at": timestamp.isoformat(),
            "first_date": observations[0]["date"],
            "last_date": observations[-1]["date"], "count": len(prices),
            "sha256": _digest(observations), "observations": observations}


def prices_from_snapshot(snapshot):
    observations = snapshot["observations"]
    if _digest(observations) != snapshot["sha256"]:
        raise ValueError("Historical snapshot checksum mismatch.")
    prices = pd.Series([row["close"] for row in observations],
                       index=pd.DatetimeIndex([row["date"] for row in observations]))
    rebuilt = historical_snapshot(prices, ticker=snapshot["ticker"],
                                  period=snapshot["requested_period"],
                                  retrieved_at=snapshot["retrieved_at"],
                                  source=snapshot["source"])
    if rebuilt != snapshot:
        raise ValueError("Historical snapshot metadata mismatch.")
    return prices


def run_simulation(snapshot, config):
    """Return a deterministic, JSON-safe report and its simulated paths."""
    prices = prices_from_snapshot(snapshot)
    mu, sigma = estimate_gbm_params(prices)
    s0 = float(prices.iloc[-1])
    theory = terminal_theory(s0, mu, sigma, config.horizon)
    paths = simulate_gbm(s0, mu, sigma, config.horizon, config.steps,
                         config.n_sims, config.seed)
    comparison = compare_terminal(paths[-1], theory)
    report = {
        "schema_version": 1, "model": "single_asset_gbm",
        "historical_data": snapshot, "last_price": s0,
        "parameters": {"mu": mu, "sigma": sigma,
                       "trading_days_per_year": TRADING_DAYS,
                       "log_return_std_ddof": 1},
        "simulation": asdict(config),
        "conventions": {"horizon_unit": "years", "terminal_variance_ddof": 1,
                        "quantile_method": "linear", "rng": "NumPy default_rng/PCG64",
                        "convergence": "nested prefixes of the same terminal sample",
                        "undefined_values": "null (N=1 variance or zero-reference relative error)"},
        "results": {"terminal": comparison,
                    "risk": risk_summary(paths, config.confidence)},
        "convergence": convergence_analysis(paths[-1], theory),
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in
                           ("numpy", "pandas", "matplotlib", "yfinance")}},
    }
    # Fail on nonfinite computations instead of writing nonstandard JSON.
    json.dumps(report, allow_nan=False)
    return report, paths


def write_report(report, path):
    path = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_report(path):
    """Read a validated report, including any saved company snapshot."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report["schema_version"] != 1 or report["model"] != "single_asset_gbm":
        raise ValueError("Unsupported report schema/model.")
    snapshot = report["historical_data"]
    prices_from_snapshot(snapshot)
    SimulationConfig(**report["simulation"])
    return report


def read_replay(path):
    report = read_report(path)
    snapshot = report["historical_data"]
    return snapshot, SimulationConfig(**report["simulation"])
