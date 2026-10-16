"""Reproducible offline report generation, serialization and replay."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import main as app
from simulator.plotting import plot_convergence
from simulator.fundamentals import company_snapshot
from simulator.report import (SimulationConfig, historical_snapshot,
                              prices_from_snapshot, read_replay,
                              run_simulation, write_report)


@pytest.fixture
def snapshot():
    prices = pd.Series(100 * np.exp([0, 0.01, -0.01, 0.02]),
                       index=pd.date_range("2026-01-05", periods=4, freq="B"))
    return historical_snapshot(prices, ticker="SYNTHETIC", period="fixture",
                               retrieved_at="2026-01-09T12:00:00+00:00",
                               source="deterministic_test_fixture")


def test_report_records_inputs_calibration_results_and_replays(snapshot, tmp_path):
    config = SimulationConfig(horizon=0.5, steps=6, n_sims=100, seed=123, confidence=0.9)
    report, paths = run_simulation(snapshot, config)
    second, second_paths = run_simulation(snapshot, config)
    assert report == second
    np.testing.assert_array_equal(paths, second_paths)
    assert report["historical_data"] == snapshot
    assert snapshot["first_date"] == "2026-01-05T00:00:00"
    assert snapshot["last_date"] == "2026-01-08T00:00:00"
    assert snapshot["count"] == 4
    assert report["last_price"] == pytest.approx(100 * np.exp(0.02))
    returns = np.array([0.01, -0.02, 0.03])
    variance = np.var(returns, ddof=1) * 252
    assert report["parameters"]["sigma"] == pytest.approx(np.sqrt(variance))
    assert report["parameters"]["mu"] == pytest.approx(252 * returns.mean() + variance / 2)
    assert report["simulation"] == {"horizon": 0.5, "steps": 6, "n_sims": 100,
                                     "seed": 123, "confidence": 0.9}
    terminal = report["results"]["terminal"]
    assert terminal["mean"]["estimate"] == paths[-1].mean()
    assert terminal["variance"]["estimate"] == paths[-1].var(ddof=1)
    assert report["convergence"][-1]["mean"] == terminal["mean"]
    assert report["environment"]["numpy"] == np.__version__
    path = tmp_path / "nested" / "run.json"
    write_report(report, path)
    assert json.loads(path.read_text()) == report
    first_bytes = path.read_bytes()
    write_report(second, path)
    assert path.read_bytes() == first_bytes
    replay_snapshot, replay_config = read_replay(path)
    replay, _ = run_simulation(replay_snapshot, replay_config)
    assert replay == report


@pytest.mark.parametrize("with_company", [False, True])
def test_cli_replay_never_downloads(snapshot, tmp_path, monkeypatch, with_company):
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=20))
    if with_company:
        report["company_analysis"] = company_snapshot(
            "SYNTHETIC", {"totalRevenue": 12345}, retrieved_at="2026-01-09T12:00:00Z")
    source = tmp_path / "source.json"
    destination = tmp_path / "replay.json"
    write_report(report, source)

    def forbidden_download(*args, **kwargs):
        pytest.fail("Replay must not access the data provider")

    monkeypatch.setattr(app, "download_prices", forbidden_download)
    monkeypatch.setattr(app, "download_company", forbidden_download)
    monkeypatch.setattr("builtins.input", forbidden_download)
    app.main(["--replay", str(source), "--report", str(destination), "--no-plot"])
    assert source.read_bytes() == destination.read_bytes()
    assert destination.with_name("replay-convergence.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("change", ["price", "date", "metadata"])
def test_corrupt_snapshot_is_rejected(snapshot, change):
    damaged = deepcopy(snapshot)
    if change == "price":
        damaged["observations"][0]["close"] += 1
    elif change == "date":
        damaged["observations"][0]["date"] = "2000-01-01"
    else:
        damaged["count"] = 999
    with pytest.raises(ValueError, match="mismatch"):
        prices_from_snapshot(damaged)


@pytest.mark.parametrize("prices", [
    pd.Series([1, 2, 3]),
    pd.Series([1, 2], index=pd.date_range("2026-01-01", periods=2)),
    pd.Series([1, 0, 3], index=pd.date_range("2026-01-01", periods=3)),
    pd.Series([1, np.nan, 3], index=pd.date_range("2026-01-01", periods=3)),
    pd.Series([1, 2, 3], index=pd.to_datetime(["2026-01-01"] * 3)),
    pd.Series([1, 2, 3], index=pd.date_range("2026-01-01", periods=3)[::-1]),
])
def test_invalid_history_is_rejected(prices):
    with pytest.raises(ValueError):
        historical_snapshot(prices, ticker="X", period="1y",
                            retrieved_at="2026-01-09T12:00:00+00:00")


@pytest.mark.parametrize("kwargs", [
    {"horizon": np.nan}, {"horizon": 0}, {"steps": 0}, {"steps": 2.5},
    {"n_sims": 0}, {"seed": -1}, {"seed": None}, {"confidence": 1},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_zero_volatility_and_one_path_are_serializable_and_plottable(tmp_path):
    snapshot = historical_snapshot(
        pd.Series([100.0] * 4, index=pd.date_range("2026-01-01", periods=4)),
        ticker="CONSTANT", period="fixture", retrieved_at="2026-01-09T00:00:00Z",
        source="deterministic_test_fixture")
    report, _ = run_simulation(snapshot, SimulationConfig(steps=1, n_sims=1))
    path = tmp_path / "single.json"
    write_report(report, path)
    assert "NaN" not in path.read_text()
    assert report["results"]["terminal"]["variance"]["estimate"] is None
    figure = plot_convergence(report["convergence"], tmp_path / "single.png")
    assert len(figure.axes) == 3
    assert figure.axes[0].get_xscale() == "log"
