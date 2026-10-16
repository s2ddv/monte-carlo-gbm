"""Command-line parsing and an offline entry-point smoke test."""

import pandas as pd
import pytest

import main as app
from simulator.fundamentals import company_snapshot


@pytest.fixture(autouse=True)
def offline_company(monkeypatch):
    monkeypatch.setattr(app, "download_company", lambda ticker: company_snapshot(
        ticker, {"longName": "Test Company", "currency": "USD", "financialCurrency": "USD",
                 "totalRevenue": 123456789, "trailingPE": 15, "profitMargins": 0.125},
        retrieved_at="2026-01-09T12:00:00+00:00"))


@pytest.mark.parametrize("arguments", [
    ["--steps", "0"], ["--n-sims", "-1"],
    ["--horizon", "nan"], ["--horizon", "inf"],
    ["--confidence", "0"], ["--confidence", "1"], ["--seed", "-1"],
])
def test_argparse_rejects_invalid_values(arguments):
    with pytest.raises(SystemExit) as error:
        app.build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_main_runs_with_arguments_without_plot_or_network(monkeypatch, capsys, tmp_path):
    prices = pd.Series([100.0, 101.0, 102.0, 103.0],
                       index=pd.date_range("2026-01-01", periods=4))

    def fake_download(ticker, period):
        assert (ticker, period) == ("TEST", "1y")
        return prices

    monkeypatch.setattr(app, "download_prices", fake_download)
    app.main(["--ticker", "TEST", "--period", "1y", "--horizon", "0.5",
              "--steps", "10", "--n-sims", "1000", "--seed", "42",
              "--confidence", "0.9", "--no-plot", "--report", str(tmp_path / "run.json")])

    output = capsys.readouterr().out
    assert "'expected_price':" in output
    assert "'prob_loss':" in output
    assert "'var':" in output
    assert "'cvar':" in output
    assert "Preço esperado teórico (0.5 ano(s))" in output
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "run-convergence.png").read_bytes().startswith(b"\x89PNG")
    assert "Test Company (TEST)" in output
    assert "P/L (TTM): 15.00x" in output
    assert "Receita (revenue, TTM): USD 123,456,789.00" in output


def test_prompt_retries_and_normalizes_brazilian_ticker(monkeypatch, capsys, tmp_path):
    answers = iter(["", "two tickers", " petr4 "])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    received = []

    def prices(ticker, period):
        received.append(ticker)
        return pd.Series([100., 101., 102.], index=pd.date_range("2026-01-01", periods=3))

    monkeypatch.setattr(app, "download_prices", prices)
    app.main(["--no-plot", "--steps", "2", "--n-sims", "10",
              "--report", str(tmp_path / "prompt.json")])
    assert received == ["PETR4.SA"]
    assert capsys.readouterr().out.count("Informe apenas um ticker válido") == 2


def test_explicit_ticker_does_not_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("Unexpected prompt"))
    monkeypatch.setattr(app, "download_prices", lambda *a, **kw: pd.Series(
        [100., 101., 102.], index=pd.date_range("2026-01-01", periods=3)))
    app.main(["--ticker", "aapl", "--no-plot", "--steps", "2", "--n-sims", "10",
              "--report", str(tmp_path / "explicit.json")])


def test_prompt_eof_explains_cli_option(monkeypatch, capsys):
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(SystemExit) as exc:
        app.main([])
    assert exc.value.code == 2
    assert "--ticker" in capsys.readouterr().err


def test_unavailable_price_history_has_actionable_error(monkeypatch, capsys, tmp_path):
    def unavailable(*args, **kwargs):
        raise ValueError("No data")
    monkeypatch.setattr(app, "download_prices", unavailable)
    destination = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as exc:
        app.main(["--ticker", "INVALID", "--report", str(destination)])
    assert exc.value.code == 1
    assert "Confira o ticker, o período e a conexão" in capsys.readouterr().err
    assert not destination.exists()
