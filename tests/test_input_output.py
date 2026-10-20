"""Provider boundary and complete-run publication tests."""

import pandas as pd
import pytest

import main as app
from simulator.data import download_prices
from simulator.report import historical_snapshot, SimulationConfig, run_simulation, write_report


@pytest.fixture
def dates():
    return pd.date_range("2026-01-01", periods=3)


def test_provider_single_adjusted_series(monkeypatch, dates):
    def fake(*args, **kwargs):
        assert kwargs["auto_adjust"] is True
        assert kwargs["group_by"] == "column"
        return pd.DataFrame({("Close", "AAPL"): [100., 101., 102.],
                             ("Open", "AAPL"): [99., 100., 101.]}, index=dates)
    monkeypatch.setattr("simulator.data.yf.download", fake)
    prices = download_prices("AAPL")
    assert prices.tolist() == [100., 101., 102.]
    snapshot = historical_snapshot(prices, ticker="AAPL", period="3y",
                                   retrieved_at="2026-01-04T00:00:00Z")
    assert (snapshot["source"], snapshot["price_field"], snapshot["interval"]) == (
        "yfinance", "adjusted_close", "1d")


@pytest.mark.parametrize("frame, message", [
    (pd.DataFrame(), "No price history"),
    (pd.DataFrame({"Open": [1., 2., 3.]}, index=pd.date_range("2026-01-01", periods=3)), "columns"),
    (pd.DataFrame({"Close": [1.]}, index=pd.date_range("2026-01-01", periods=1)), "three"),
    (pd.DataFrame({"Close": [1., 2., 3.]}, index=["bad", "dates", "here"]), "dates"),
    (pd.DataFrame({"Close": [1., 2., 3.]}, index=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"])), "dates"),
    (pd.DataFrame({"Close": [1., float("nan"), 3.]}, index=pd.date_range("2026-01-01", periods=3)), "finite"),
])
def test_provider_rejects_invalid_response(monkeypatch, frame, message):
    monkeypatch.setattr("simulator.data.yf.download", lambda *a, **kw: frame)
    with pytest.raises(ValueError, match=message):
        download_prices("AAPL")


def test_provider_rejects_multiple_tickers(monkeypatch, dates):
    frame = pd.DataFrame({("Close", "AAPL"): [1., 2., 3.],
                          ("Close", "MSFT"): [1., 2., 3.]}, index=dates)
    monkeypatch.setattr("simulator.data.yf.download", lambda *a, **kw: frame)
    with pytest.raises(ValueError, match="one adjusted Close"):
        download_prices("AAPL")


def test_existing_name_and_figure_failure_leave_no_report(monkeypatch, tmp_path, dates):
    prices = pd.Series([100., 101., 102.], index=dates)
    snapshot = historical_snapshot(prices, ticker="AAPL", period="fixture",
                                   retrieved_at="2026-01-04T00:00:00Z")
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10))
    source = tmp_path / "source.json"
    write_report(report, source)
    args = ["--replay", str(source), "--no-plot", "--report", str(tmp_path / "run.json")]
    monkeypatch.setattr(app, "plot_paths", lambda *a, **kw: (_ for _ in ()).throw(OSError("figure failed")))
    with pytest.raises(OSError, match="figure failed"):
        app.main(args)
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob(".run-*"))
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError, match="Choose another"):
        app.main(args)
    assert list((tmp_path / "run").iterdir()) == []


def test_report_write_failure_leaves_no_run(monkeypatch, tmp_path, dates):
    prices = pd.Series([100., 101., 102.], index=dates)
    snapshot = historical_snapshot(prices, ticker="AAPL", period="fixture",
                                   retrieved_at="2026-01-04T00:00:00Z")
    report, _ = run_simulation(snapshot, SimulationConfig(steps=2, n_sims=10))
    source = tmp_path / "source.json"
    write_report(report, source)
    monkeypatch.setattr(app, "write_report", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        app.main(["--replay", str(source), "--no-plot", "--report", str(tmp_path / "run.json")])
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob(".run-*"))
