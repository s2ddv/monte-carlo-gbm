"""Command-line parsing and an offline entry-point smoke test."""

import pandas as pd
import pytest

import main as app


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
