"""Offline checks of chronological GBM forecasts against dated synthetic closes."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import main as app
from calibration.fundamentals import company_snapshot
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                               run_simulation, write_report)
from validation.out_of_sample import evaluate_out_of_sample


def dated_prices(n=70):
    daily_logs = np.resize([0.01, -0.006, 0.004, -0.002], n - 1)
    return pd.Series(100 * np.exp(np.r_[0, np.cumsum(daily_logs)]),
                     index=pd.date_range("2025-01-02", periods=n, freq="B"))


def test_frozen_calibration_and_nonoverlapping_forecasts():
    prices = dated_prices()
    result = evaluate_out_of_sample(prices, calibration_observations=40,
                                    horizon_days=7, alpha=0.9)
    changed = prices.copy()
    changed.iloc[40:] *= 3
    altered = evaluate_out_of_sample(changed, calibration_observations=40,
                                     horizon_days=7, alpha=0.9)
    assert result["calibration"] == altered["calibration"]
    rows = result["evaluation"]["rows"]
    assert len(rows) == 4
    assert result["evaluation"]["unscored_tail_observations"] == 2
    assert rows[0]["origin_date"] == prices.index[39].isoformat()
    assert rows[0]["target_date"] == prices.index[46].isoformat()
    assert rows[-1]["target_date"] == prices.index[67].isoformat()
    for current, following in zip(rows, rows[1:]):
        assert current["origin_date"] < current["target_date"]
        assert current["target_date"] == following["origin_date"]
    assert rows[-1]["origin_date"] < rows[-1]["target_date"]
    assert rows[0]["predicted_mean"] == pytest.approx(
        altered["evaluation"]["rows"][0]["predicted_mean"])
    assert rows[0]["var"] == pytest.approx(
        altered["evaluation"]["rows"][0]["var"])


def test_manual_losses_var_and_point_errors():
    # The training closes are constant, so sigma=mu=0. Four disjoint
    # one-session forecasts permit exact hand calculations without simulation.
    prices = pd.Series([100.0] * 31 + [90.0, 110.0, 100.0, 80.0],
                       index=pd.date_range("2025-01-02", periods=35, freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=31,
                                    horizon_days=1, alpha=0.9)
    rows = result["evaluation"]["rows"]
    assert result["calibration"]["sigma"] == 0
    assert result["calibration"]["mu"] == 0
    expected_origins = [100.0, 90.0, 110.0, 100.0]
    expected_targets = [90.0, 110.0, 100.0, 80.0]
    expected_errors = [-10.0, 20.0, -10.0, -20.0]
    for row, origin, target, error in zip(rows, expected_origins,
                                           expected_targets, expected_errors):
        assert row["origin_price"] == origin
        assert row["observed_price"] == target
        assert row["predicted_mean"] == origin
        assert row["predicted_median"] == origin
        assert row["lower_quantile"] == origin
        assert row["var"] == 0
        assert row["realized_loss"] == origin - target
        assert row["error"] == error
        assert row["var_violation"] is (error < 0)
    evaluation = result["evaluation"]["metrics"]
    assert evaluation["forecast_count"] == 4
    assert evaluation["violation_count"] == 3
    assert evaluation["observed_violation_rate"] == pytest.approx(0.75)
    assert evaluation["nominal_violation_rate"] == pytest.approx(0.1)
    assert evaluation["mean_bias"] == pytest.approx(-5.0)
    assert evaluation["mae"] == pytest.approx(15.0)
    assert evaluation["rmse"] == pytest.approx(np.sqrt(250.0))
    assert evaluation["mae_percent"] == pytest.approx(100 * np.mean(
        [10 / 100, 20 / 90, 10 / 110, 20 / 100]))
    json.dumps(result, allow_nan=False)


def test_nonzero_volatility_quantile_is_independent_analytic_reference():
    prices = dated_prices(50)
    result = evaluate_out_of_sample(prices, calibration_observations=40,
                                    horizon_days=5, alpha=0.95)
    calibration = result["calibration"]
    row = result["evaluation"]["rows"][0]
    training_logs = np.diff(np.log(prices.iloc[:40].to_numpy()))
    annual_sigma = np.std(training_logs, ddof=1) * np.sqrt(252)
    annual_mu = 252 * np.mean(training_logs) + annual_sigma**2 / 2
    t = 5 / 252
    s0 = float(prices.iloc[39])
    expected_mean = s0 * np.exp(annual_mu * t)
    expected_median = s0 * np.exp((annual_mu - annual_sigma**2 / 2) * t)
    expected_lower = s0 * np.exp((annual_mu - annual_sigma**2 / 2) * t
                                 + annual_sigma * np.sqrt(t) * norm.ppf(0.05))
    assert calibration["sigma"] == pytest.approx(annual_sigma)
    assert calibration["mu"] == pytest.approx(annual_mu)
    assert row["predicted_mean"] == pytest.approx(expected_mean)
    assert row["predicted_median"] == pytest.approx(expected_median)
    assert row["lower_quantile"] == pytest.approx(expected_lower)
    assert row["var"] == pytest.approx(s0 - expected_lower)
    assert row["var_violation"] is (row["realized_loss"] > row["var"])


@pytest.mark.parametrize("change", [
    lambda p: p.set_axis(pd.RangeIndex(len(p))),
    lambda p: p.set_axis(p.index[::-1]),
    lambda p: p.set_axis(pd.DatetimeIndex([p.index[0]] * len(p))),
    lambda p: p.mask(np.arange(len(p)) == 2, np.nan),
    lambda p: p.mask(np.arange(len(p)) == 2, 0),
    lambda p: p.mask(np.arange(len(p)) == 2, np.inf),
])
def test_invalid_history_rejected(change):
    with pytest.raises(ValueError):
        evaluate_out_of_sample(change(dated_prices()), calibration_observations=40,
                               horizon_days=7, alpha=0.9)


@pytest.mark.parametrize("calibration,horizon,alpha,n", [
    (2, 1, 0.9, 70), (31, 1, 0.9, 31), (40, 31, 0.9, 70),
    (40, 0, 0.9, 70), (40, 1.5, 0.9, 70), (True, 1, 0.9, 70),
    (40, 1, 0, 70), (40, 1, 1, 70), (40, 1, np.nan, 70),
])
def test_invalid_parameters_or_insufficient_history(calibration, horizon, alpha, n):
    with pytest.raises(ValueError):
        evaluate_out_of_sample(dated_prices(n), calibration_observations=calibration,
                               horizon_days=horizon, alpha=alpha)


def test_no_violations_and_zero_volatility_are_json_finite():
    prices = pd.Series([100.0] * 35,
                       index=pd.date_range("2025-01-02", periods=35, freq="B"))
    result = evaluate_out_of_sample(prices, calibration_observations=31,
                                    horizon_days=1, alpha=0.95)
    assert result["evaluation"]["metrics"]["violation_count"] == 0
    assert result["evaluation"]["metrics"]["observed_violation_rate"] == 0
    assert all(row["var_violation"] is False for row in result["evaluation"]["rows"])
    json.dumps(result, allow_nan=False)


def test_tiny_valid_alpha_keeps_lower_tail_quantile_finite():
    result = evaluate_out_of_sample(dated_prices(50),
                                    calibration_observations=40,
                                    horizon_days=5, alpha=1e-20)
    row = result["evaluation"]["rows"][0]
    assert np.isfinite(row["lower_quantile"])
    assert row["lower_quantile"] > 0
    assert row["lower_quantile"] > row["predicted_median"]
    json.dumps(result, allow_nan=False)


def test_saved_diagnostic_detects_changed_snapshot_or_metrics(tmp_path):
    prices = dated_prices()
    snapshot = historical_snapshot(prices, ticker="SYN", period="fixture",
                                   retrieved_at="2025-05-01T00:00:00Z", source="synthetic")
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10),
                               {"calibration_observations": 40,
                                "horizon_days": 7, "alpha": 0.9})
    path = tmp_path / "valid.json"
    write_report(report, path)
    assert read_report(path) == report
    changed = deepcopy(report)
    changed["out_of_sample"]["evaluation"]["metrics"]["violation_count"] += 1
    write_report(changed, tmp_path / "changed-metrics.json")
    with pytest.raises(ValueError):
        read_report(tmp_path / "changed-metrics.json")
    changed = deepcopy(report)
    changed["historical_data"]["observations"][0]["close"] += 1
    write_report(changed, tmp_path / "changed-snapshot.json")
    with pytest.raises(ValueError):
        read_report(tmp_path / "changed-snapshot.json")


def test_cli_opt_in_replay_is_offline_and_legacy_stays_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prices = dated_prices()
    monkeypatch.setattr(app, "download_prices", lambda *a, **k: prices)
    monkeypatch.setattr(app, "download_company", lambda ticker: company_snapshot(
        ticker, {"longName": "Synthetic", "currency": "USD"},
        retrieved_at="2025-05-01T00:00:00Z"))
    first = tmp_path / "first.json"
    app.main(["--ticker", "SYN", "--period", "fixture", "--out-of-sample",
              "--calibration-observations", "40", "--evaluation-horizon-days", "7",
              "--evaluation-confidence", "0.9", "--steps", "2", "--n-sims", "10",
              "--report", str(first), "--no-plot"])
    first_report = json.loads((tmp_path / "first" / "report.json").read_text())
    assert first_report["out_of_sample"]["evaluation"]["metrics"]["forecast_count"] == 4
    assert first_report["out_of_sample"]["historical_snapshot"]["sha256"] == (
        first_report["historical_data"]["sha256"])
    monkeypatch.setattr(app, "download_prices", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(app, "download_company", lambda *a, **k: pytest.fail("network"))
    second = tmp_path / "second.json"
    app.main(["--replay", str(tmp_path / "first" / "report.json"),
              "--report", str(second), "--no-plot"])
    second_report = json.loads((tmp_path / "second" / "report.json").read_text())
    assert second_report["out_of_sample"] == first_report["out_of_sample"]
    assert second_report == first_report
    snapshot = historical_snapshot(prices, ticker="SYN", period="fixture",
                                   retrieved_at="2025-05-01T00:00:00Z", source="synthetic")
    baseline, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10))
    assert "out_of_sample" not in baseline
    json.dumps(second_report, allow_nan=False)
