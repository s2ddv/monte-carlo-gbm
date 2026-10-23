"""Provider-reported company fundamentals, kept separate from GBM forecasts."""

from datetime import datetime, timezone
import math
from numbers import Real
import re
from urllib.parse import quote

import yfinance as yf


# Monetary fields explicitly distinguish trading and financial-statement units.
# Percent fields below are fractions in Yahoo's quoteSummary response.
METRICS = (
    ("Mercado e valuation", "currentPrice", "Cotação", "quote", "latest quote"),
    ("Mercado e valuation", "marketCap", "Valor de mercado", "quote", "latest available"),
    ("Mercado e valuation", "enterpriseValue", "Enterprise value (EV)", "quote", "latest available"),
    ("Mercado e valuation", "trailingPE", "P/L (TTM)", "multiple", "TTM"),
    ("Mercado e valuation", "forwardPE", "P/L projetado", "multiple", "forward estimate"),
    ("Mercado e valuation", "priceToBook", "P/VP", "multiple", "latest available"),
    ("Mercado e valuation", "priceToSalesTrailing12Months", "Preço/receita (TTM)", "multiple", "TTM"),
    ("Mercado e valuation", "enterpriseToRevenue", "EV/receita", "multiple", "provider reported"),
    ("Mercado e valuation", "enterpriseToEbitda", "EV/EBITDA", "multiple", "provider reported"),
    ("Resultados e crescimento", "totalRevenue", "Receita (revenue, TTM)", "financial", "TTM"),
    ("Resultados e crescimento", "netIncomeToCommon", "Lucro líquido atribuível (TTM)", "financial", "TTM"),
    ("Resultados e crescimento", "ebitda", "EBITDA (TTM)", "financial", "TTM"),
    ("Resultados e crescimento", "revenueGrowth", "Crescimento da receita (trimestre vs. ano anterior)", "percent", "quarterly YoY"),
    ("Resultados e crescimento", "earningsQuarterlyGrowth", "Crescimento do lucro (trimestre vs. ano anterior)", "percent", "quarterly YoY"),
    ("Margens e rentabilidade", "grossMargins", "Margem bruta (TTM)", "percent", "TTM"),
    ("Margens e rentabilidade", "operatingMargins", "Margem operacional (último trimestre)", "percent", "latest quarter"),
    ("Margens e rentabilidade", "profitMargins", "Margem líquida (TTM)", "percent", "TTM"),
    ("Margens e rentabilidade", "returnOnEquity", "ROE (TTM)", "percent", "TTM"),
    ("Margens e rentabilidade", "returnOnAssets", "ROA (TTM)", "percent", "TTM"),
    ("Caixa, dívida e liquidez", "operatingCashflow", "Fluxo de caixa operacional (TTM)", "financial", "TTM"),
    ("Caixa, dívida e liquidez", "freeCashflow", "Fluxo de caixa livre (TTM)", "financial", "TTM"),
    ("Caixa, dívida e liquidez", "totalCash", "Caixa total (último trimestre)", "financial", "latest quarter"),
    ("Caixa, dívida e liquidez", "totalDebt", "Dívida total (último trimestre)", "financial", "latest quarter"),
    ("Caixa, dívida e liquidez", "currentRatio", "Liquidez corrente (último trimestre)", "multiple", "latest quarter"),
    ("Caixa, dívida e liquidez", "quickRatio", "Liquidez seca (último trimestre)", "multiple", "latest quarter"),
    ("Dividendos e risco de mercado", "trailingAnnualDividendYield", "Dividend yield histórico anual", "percent", "trailing annual"),
    ("Dividendos e risco de mercado", "payoutRatio", "Payout (TTM)", "percent", "TTM"),
    ("Dividendos e risco de mercado", "beta", "Beta (fornecedor)", "number", "provider reported"),
    ("Dividendos e risco de mercado", "fiftyTwoWeekLow", "Mínima de 52 semanas", "quote", "52 weeks"),
    ("Dividendos e risco de mercado", "fiftyTwoWeekHigh", "Máxima de 52 semanas", "quote", "52 weeks"),
)


def normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]{0,29}", ticker):
        raise ValueError("Informe apenas um ticker válido, como PETR4.SA ou AAPL.")
    # B3's familiar local shorthand; other exchanges keep their explicit suffix.
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", ticker):
        ticker += ".SA"
    return ticker


def _number(value):
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


def _date(value):
    value = _number(value)
    if value is None or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def company_snapshot(ticker, info, *, retrieved_at, warning=None):
    """Normalize missing values without inventing financial data or FX rates."""
    info = info if isinstance(info, dict) else {}
    quote_currency = _text(info.get("currency"))
    financial_currency = _text(info.get("financialCurrency"))
    quote_type = _text(info.get("quoteType"))
    warnings = [warning] if warning else []
    if quote_type and quote_type != "EQUITY":
        warnings.append("O ativo não é uma ação de empresa; indicadores corporativos podem não se aplicar.")
    metrics = []
    for group, field, label, kind, period in METRICS:
        value = _number(info.get(field))
        reason = None if value is not None else "not supplied or nonfinite"
        if field in ("trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda"):
            denominator = {"trailingPE": "trailingEps", "forwardPE": "forwardEps",
                           "priceToBook": "bookValue", "enterpriseToEbitda": "ebitda"}[field]
            base = _number(info.get(denominator))
            if (base is not None and base <= 0) or (value is not None and value <= 0):
                value, reason = None, "nonpositive earnings, book value or multiple"
        metrics.append({"field": field, "label": label, "group": group,
                        "value": value, "kind": kind, "period": period,
                        "currency": quote_currency if kind == "quote" else
                        financial_currency if kind == "financial" else None,
                        "unavailable_reason": reason})
    if not any(metric["value"] is not None for metric in metrics):
        warnings.append("Indicadores indisponíveis: confira o ticker ou tente novamente mais tarde.")
    return {"ticker": ticker, "name": _text(info.get("longName")) or _text(info.get("shortName")),
            "sector": _text(info.get("sector")), "industry": _text(info.get("industry")),
            "country": _text(info.get("country")), "quote_type": quote_type,
            "quote_currency": quote_currency, "financial_currency": financial_currency,
            "source": "Yahoo Finance via yfinance.Ticker.get_info",
            "source_url": f"https://finance.yahoo.com/quote/{quote(ticker, safe='')}/key-statistics/",
            "retrieved_at": retrieved_at,
            "market_time": _date(info.get("regularMarketTime")),
            "most_recent_quarter": _date(info.get("mostRecentQuarter")),
            "last_fiscal_year_end": _date(info.get("lastFiscalYearEnd")),
            "metrics": metrics, "warnings": warnings}


def download_company(ticker: str):
    warning = None
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
        warning = "Falha ao consultar fundamentos. A simulação continuará se houver preços históricos."
    return company_snapshot(ticker, info, retrieved_at=datetime.now(timezone.utc).isoformat(),
                            warning=warning)


def _format_metric(metric):
    value = metric["value"]
    if value is None:
        if metric["unavailable_reason"] == "nonpositive earnings, book value or multiple":
            return "N/D (base ou múltiplo não positivo)"
        return "N/D"
    if metric["kind"] == "percent":
        return f"{value:.2%}"
    if metric["kind"] == "multiple":
        return f"{value:.2f}x"
    if metric["kind"] in ("quote", "financial"):
        currency = metric["currency"] or "moeda não informada"
        return f"{currency} {value:,.2f}"
    return f"{value:,.2f}"


def format_company(snapshot):
    """Render the saved snapshot, so offline replay cannot refresh fundamentals."""
    lines = [f"\nAnálise da empresa: {snapshot['name'] or snapshot['ticker']} ({snapshot['ticker']})",
             f"Setor: {snapshot['sector'] or 'N/D'} | Indústria: {snapshot['industry'] or 'N/D'}",
             f"Fonte: {snapshot['source']} | Consulta UTC: {snapshot['retrieved_at']}",
             f"Cotação em: {snapshot['market_time'] or 'N/D'} | Último trimestre encerrado: "
             f"{snapshot['most_recent_quarter'] or 'N/D'}",
             f"Moeda de negociação: {snapshot['quote_currency'] or 'N/D'} | "
             f"Moeda dos demonstrativos: {snapshot['financial_currency'] or 'N/D'}"]
    lines.extend(f"Aviso: {warning}" for warning in snapshot["warnings"])
    last_group = None
    for metric in snapshot["metrics"]:
        if metric["group"] != last_group:
            lines.append(f"\n{metric['group']}")
            last_group = metric["group"]
        lines.append(f"  {metric['label']}: {_format_metric(metric)}")
    lines.extend(["\nTTM = últimos 12 meses. Projetado = estimativa do fornecedor.",
                  "N/D = indisponível ou não aplicável. Valores monetários sem abreviação; "
                  "vírgula separa milhares e ponto separa decimais.",
                  "Indicadores reportados pelo fornecedor; datas e períodos podem diferir. "
                  "Não há conversão de moedas nem recálculo de múltiplos.",
                  "Compare com empresas do mesmo setor e seus demonstrativos. "
                  "Valor de mercado e EV não são uma estimativa de valor justo; "
                  "a simulação GBM não calcula valuation fundamentalista."])
    return "\n".join(lines)
