"""Analyze a selected company and simulate its price with GBM."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import sys

# Allow `python main.py` from the repository without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.data import download_prices
from calibration.fundamentals import download_company, format_company, normalize_ticker
from simulation.plotting import (plot_convergence, plot_final_distribution,
                                plot_paths)
from simulation.report import (SimulationConfig, historical_snapshot, read_report,
                              run_simulation, write_report)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def confidence_level(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be between 0 and 1") from exc
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", type=normalize_ticker,
                        help="market ticker (asks when omitted; e.g. PETR4.SA, AAPL)")
    parser.add_argument("--period", default="3y", help="price history period")
    parser.add_argument("--horizon", type=positive_float, default=1.0,
                        help="simulation horizon in years (default: 1)")
    parser.add_argument("--steps", type=positive_int, default=252,
                        help="steps per simulation (default: 252)")
    parser.add_argument("--n-sims", type=positive_int, default=1000,
                        help="number of paths (default: 1000)")
    parser.add_argument("--seed", type=nonnegative_int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--confidence", type=confidence_level, default=0.95,
                        help="VaR confidence level (default: 0.95)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the interactive chart")
    parser.add_argument("--report", type=Path, default=Path("reports/simulation.json"),
                        help="output JSON (default: reports/simulation.json)")
    parser.add_argument("--replay", type=Path,
                        help="replay a report offline; uses its data and simulation settings")
    parser.add_argument("--out-of-sample", action="store_true",
                        help="add fixed-calibration chronological historical diagnostics")
    parser.add_argument("--calibration-observations", type=positive_int,
                        help="number of initial closes used only for historical calibration")
    parser.add_argument("--evaluation-horizon-days", type=positive_int, default=1,
                        help="nonoverlapping historical horizon in trading observations (default: 1)")
    parser.add_argument("--evaluation-confidence", type=confidence_level, default=0.95,
                        help="historical VaR confidence (default: 0.95)")
    return parser


def check_destination(report_path: Path) -> Path:
    """Reject an occupied run name before acquiring data or simulating."""
    destination = report_path.with_suffix("")
    if destination.exists() or report_path.exists():
        raise FileExistsError(
            f"Execution already exists: {destination} or {report_path}. Choose another --report name.")
    return destination


def format_simulation_summary(report, config, terminal_mean, terminal_std) -> str:
    """Build the terminal summary without calculation or I/O side effects."""
    snapshot = report["historical_data"]
    s0 = report["last_price"]
    mu, sigma = report["parameters"]["mu"], report["parameters"]["sigma"]
    lines = [f"\nSimulação GBM: {snapshot['ticker']} (preços históricos ajustados)",
             f"Último preço: {s0:.2f}", f"mu = {mu:.2%} | sigma = {sigma:.2%}",
             str(report["results"]["risk"]),
             f"Preço esperado teórico ({config.horizon:g} ano(s)): "
             f"{report['results']['terminal']['mean']['theoretical']:.2f}",
             f"Média dos preços finais:  {terminal_mean}",
             f"Desvio padrão dos preços finais:  {terminal_std}",
             "Validação analítica (estimativa, teoria e erros):",
             str(report["results"]["terminal"])]
    if "out_of_sample" in report:
        evaluation = report["out_of_sample"]["evaluation"]
        metrics = evaluation["metrics"]
        lines.append("Cobertura histórica observada (janelas sem sobreposição): "
                     f"{metrics['violation_count']}/{metrics['forecast_count']} violações; "
                     f"taxa observada {metrics['observed_violation_rate']:.2%}, "
                     f"referência nominal {metrics['nominal_violation_rate']:.2%}.")
        interval = evaluation.get("violation_rate_interval")
        if interval:
            lines.append(f"IC {interval['confidence_level']:.0%} Wilson da taxa observada "
                         f"({metrics['forecast_count']} previsões): "
                         f"[{interval['lower']:.2%}, {interval['upper']:.2%}].")
        lines.extend(f"Aviso: {warning}" for warning in evaluation["warnings"])
    return "\n".join(lines)


def publish_run(report, paths, config, report_path, no_plot):
    """Write a complete bundle to staging, then publish it atomically."""
    destination = report_path.with_suffix("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    figures = []
    plt = None
    published = False
    try:
        if no_plot:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ticker = report["historical_data"]["ticker"]
        figures.append(plot_convergence(report["convergence"], staging / "convergence.png"))
        figures.append(plot_paths(paths, title=f"GBM price paths: {ticker}",
                                  save_path=staging / "paths.png"))
        figures.append(plot_final_distribution(
            paths, var=report["results"]["risk"]["var"],
            title=f"Final prices: {ticker} (VaR confidence: {config.confidence:.1%})",
            save_path=staging / "final-distribution.png"))
        write_report(report, staging / "report.json")
        if destination.exists() or report_path.exists():
            raise FileExistsError(f"Execution already exists: {destination} or {report_path}.")
        staging.rename(destination)
        published = True
    except Exception:
        shutil.rmtree(staging)
        raise
    finally:
        if plt is not None:
            try:
                if published and not no_plot:
                    plt.show()
            finally:
                for figure in figures:
                    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_destination(args.report)
    company = None
    if args.replay:
        saved = read_report(args.replay)
        snapshot = saved["historical_data"]
        config = SimulationConfig(**saved["simulation"])
        company = saved.get("company_analysis")
        historical_config = None
        if "out_of_sample" in saved:
            protocol = saved["out_of_sample"]["protocol"]
            historical_config = {
                "calibration_observations": protocol["calibration_observations"],
                "horizon_days": protocol["horizon_days"], "alpha": protocol["alpha"]}
        if args.out_of_sample:
            if args.calibration_observations is None:
                parser.error("--out-of-sample requires --calibration-observations")
            historical_config = {
                "calibration_observations": args.calibration_observations,
                "horizon_days": args.evaluation_horizon_days,
                "alpha": args.evaluation_confidence}
    else:
        if args.out_of_sample and args.calibration_observations is None:
            parser.error("--out-of-sample requires --calibration-observations")
        if not args.out_of_sample and args.calibration_observations is not None:
            parser.error("--calibration-observations requires --out-of-sample")
        historical_config = ({"calibration_observations": args.calibration_observations,
                              "horizon_days": args.evaluation_horizon_days,
                              "alpha": args.evaluation_confidence}
                             if args.out_of_sample else None)
        if args.ticker is None:
            while True:
                try:
                    args.ticker = normalize_ticker(input(
                        "Digite o ticker da ação (ex.: PETR4.SA, VALE3.SA, AAPL): "))
                    break
                except ValueError as exc:
                    print(exc)
                except (EOFError, KeyboardInterrupt):
                    parser.error("Informe uma ação com --ticker (ex.: --ticker PETR4.SA).")
        print(f"Consultando {args.ticker}...")
        company = download_company(args.ticker)
        print(format_company(company))
        try:
            prices = download_prices(args.ticker, period=args.period)
        except Exception as exc:
            parser.exit(1, f"Não foi possível obter preços de {args.ticker}: {exc}. "
                        "Confira o ticker, o período e a conexão e tente novamente.\n")
        snapshot = historical_snapshot(prices, ticker=args.ticker, period=args.period,
                                       retrieved_at=datetime.now(timezone.utc).isoformat())
        config = SimulationConfig(args.horizon, args.steps, args.n_sims,
                                  args.seed, args.confidence)
    report, paths = run_simulation(snapshot, config, historical_config)
    if company is not None:
        report["company_analysis"] = company
        if args.replay:
            print(format_company(company))
    elif args.replay:
        print("Este relatório antigo não contém indicadores da empresa.")
    terminal_mean = paths[-1].mean()
    terminal_std = paths[-1].std()
    print(format_simulation_summary(report, config, terminal_mean, terminal_std))
    destination = publish_run(report, paths, config, args.report, args.no_plot)
    print(f"Relatório: {destination / 'report.json'}")
    for name in ("convergence.png", "paths.png", "final-distribution.png"):
        print(f"Figura: {destination / name}")


if __name__ == "__main__":
    main()
