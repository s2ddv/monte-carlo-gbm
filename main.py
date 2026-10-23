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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    company = None
    if args.replay:
        saved = read_report(args.replay)
        snapshot = saved["historical_data"]
        config = SimulationConfig(**saved["simulation"])
        company = saved.get("company_analysis")
    else:
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
    report, paths = run_simulation(snapshot, config)
    if company is not None:
        report["company_analysis"] = company
        if args.replay:
            print(format_company(company))
    elif args.replay:
        print("Este relatório antigo não contém indicadores da empresa.")
    print(f"\nSimulação GBM: {snapshot['ticker']} (preços históricos ajustados)")
    s0 = report["last_price"]
    mu, sigma = report["parameters"]["mu"], report["parameters"]["sigma"]
    print(f"Último preço: {s0:.2f}")
    print(f"mu = {mu:.2%} | sigma = {sigma:.2%}")
    # run_simulation computes risk_summary using the configured confidence.
    risk = report["results"]["risk"]
    print(risk)
    print(f"Preço esperado teórico ({config.horizon:g} ano(s)): "
          f"{report['results']['terminal']['mean']['theoretical']:.2f}")
    print("Média dos preços finais: ", paths[-1].mean())
    print("Desvio padrão dos preços finais: ", paths[-1].std())
    print("Validação analítica (estimativa, teoria e erros):")
    print(report["results"]["terminal"])
    # --report names a run; all artifacts live in its sibling directory.
    destination = args.report.with_suffix("")
    if destination.exists() or args.report.exists():
        raise FileExistsError(f"Execution already exists: {destination} or {args.report}. Choose another --report name.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    figures = []
    plt = None
    published = False
    try:
        if args.no_plot:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        convergence = staging / "convergence.png"
        figures.append(plot_convergence(report["convergence"], convergence))
        paths_file = staging / "paths.png"
        figures.append(plot_paths(paths, title=f"GBM price paths: {snapshot['ticker']}",
                                  save_path=paths_file))
        distribution = staging / "final-distribution.png"
        figures.append(plot_final_distribution(
            paths, var=risk["var"],
            title=f"Final prices: {snapshot['ticker']} (VaR confidence: {config.confidence:.1%})",
            save_path=distribution))
        write_report(report, staging / "report.json")
        # Rename the complete bundle only after every artifact is written.
        if destination.exists():
            raise FileExistsError(f"Execution already exists: {destination}.")
        staging.rename(destination)
        published = True
    except Exception:
        shutil.rmtree(staging)
        raise
    finally:
        if plt is not None:
            try:
                if published and not args.no_plot:
                    plt.show()
            finally:
                for figure in figures:
                    plt.close(figure)
    print(f"Relatório: {destination / 'report.json'}")
    for name in ("convergence.png", "paths.png", "final-distribution.png"):
        print(f"Figura: {destination / name}")


if __name__ == "__main__":
    main()
