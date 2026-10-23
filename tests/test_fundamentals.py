"""Financial display semantics and provider failures, without network access."""

import json

import pytest

from calibration import fundamentals as company


def snapshot(info):
    return company.company_snapshot("TEST", info, retrieved_at="2026-01-09T12:00:00Z")


@pytest.mark.parametrize("typed,expected", [
    (" aapl ", "AAPL"), ("petr4", "PETR4.SA"), ("vale3.sa", "VALE3.SA"),
    ("BRK-B", "BRK-B"), ("0700.HK", "0700.HK"), ("SAP.DE", "SAP.DE"),
])
def test_ticker_normalization(typed, expected):
    assert company.normalize_ticker(typed) == expected


@pytest.mark.parametrize("typed", ["", " ", "AAPL MSFT", "AAPL,MSFT", "../../file", "x" * 31])
def test_invalid_ticker(typed):
    with pytest.raises(ValueError):
        company.normalize_ticker(typed)


def test_units_zero_negative_and_missing_values():
    data = snapshot({"currency": "USD", "financialCurrency": "BRL", "marketCap": 2e9,
                     "totalRevenue": 9e9, "profitMargins": -0.125, "totalDebt": 0,
                     "revenueGrowth": 0.05, "trailingAnnualDividendYield": 0.03,
                     "dividendYield": 3, "beta": float("nan"), "forwardPE": float("inf"),
                     "currentRatio": True, "freeCashflow": -500, "quickRatio": "bad"})
    output = company.format_company(data)
    assert "Valor de mercado: USD 2,000,000,000.00" in output
    assert "Receita (revenue, TTM): BRL 9,000,000,000.00" in output
    assert "Margem líquida (TTM): -12.50%" in output
    assert "Dívida total (último trimestre): BRL 0.00" in output
    assert "Dividend yield histórico anual: 3.00%" in output
    assert "Fluxo de caixa livre (TTM): BRL -500.00" in output
    assert "Beta (fornecedor): N/D" in output
    assert "Liquidez corrente (último trimestre): N/D" in output
    json.dumps(data, allow_nan=False)


def test_loss_making_and_negative_equity_multiples_are_not_bargains():
    data = snapshot({"trailingPE": 5, "trailingEps": -2, "forwardPE": -3,
                     "priceToBook": 4, "bookValue": -10,
                     "enterpriseToEbitda": 8, "ebitda": 0})
    for metric in data["metrics"]:
        if metric["field"] in {"trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda"}:
            assert metric["value"] is None
            assert metric["unavailable_reason"].startswith("nonpositive")


def test_metadata_dates_and_unknown_currency_are_explicit():
    data = snapshot({"longName": "Example Company", "marketCap": 500,
                     "regularMarketTime": 1767225600, "mostRecentQuarter": 1759190400,
                     "lastFiscalYearEnd": 1e100})
    assert data["market_time"] == "2026-01-01T00:00:00+00:00"
    assert data["most_recent_quarter"] == "2025-09-30T00:00:00+00:00"
    assert data["last_fiscal_year_end"] is None
    assert "moeda não informada 500.00" in company.format_company(data)
    assert "finance.yahoo.com/quote/TEST/" in data["source_url"]


@pytest.mark.parametrize("info", [None, {}, {"quoteType": "ETF"}])
def test_empty_or_non_equity_data(info):
    data = snapshot(info)
    assert all(metric["value"] is None for metric in data["metrics"])
    assert data["warnings"]
    assert "Indicadores indisponíveis" in company.format_company(data)


def test_provider_failure_is_nonfatal(monkeypatch):
    class UnavailableTicker:
        def __init__(self, ticker):
            assert ticker == "TEST"

        def get_info(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(company.yf, "Ticker", UnavailableTicker)
    data = company.download_company("TEST")
    assert "Falha ao consultar fundamentos" in company.format_company(data)
    assert data["retrieved_at"].endswith("+00:00")


def test_provider_response_is_normalized(monkeypatch):
    class ProviderTicker:
        def __init__(self, ticker):
            assert ticker == "AAPL"

        def get_info(self):
            return {"longName": "Apple Inc.", "trailingPE": 25}

    monkeypatch.setattr(company.yf, "Ticker", ProviderTicker)
    data = company.download_company("AAPL")
    assert data["name"] == "Apple Inc."
    assert "P/L (TTM): 25.00x" in company.format_company(data)
