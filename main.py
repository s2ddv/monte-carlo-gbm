"""Run a GBM simulation and summarize terminal price risk."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow `python main.py` from the repository without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from simulator.data import download_prices
from simulator.plotting import plot_convergence
from simulator.report import (SimulationConfig, historical_snapshot, read_replay,
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
    parser.add_argument("--ticker", default="PETR4.SA", help="market ticker")
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
    args = build_parser().parse_args(argv)
    if args.replay:
        snapshot, config = read_replay(args.replay)
    else:
        prices = download_prices(args.ticker, period=args.period)
        snapshot = historical_snapshot(prices, ticker=args.ticker, period=args.period,
                                       retrieved_at=datetime.now(timezone.utc).isoformat())
        config = SimulationConfig(args.horizon, args.steps, args.n_sims,
                                  args.seed, args.confidence)
    report, paths = run_simulation(snapshot, config)
    s0 = report["last_price"]
    mu, sigma = report["parameters"]["mu"], report["parameters"]["sigma"]
    print(f"Último preço: {s0:.2f}")
    print(f"mu = {mu:.2%} | sigma = {sigma:.2%}")
    print(report["results"]["risk"])
    print(f"Preço esperado teórico ({config.horizon:g} ano(s)): "
          f"{report['results']['terminal']['mean']['theoretical']:.2f}")
    print("Média dos preços finais: ", paths[-1].mean())
    print("Desvio padrão dos preços finais: ", paths[-1].std())
    print("Validação analítica (estimativa, teoria e erros):")
    print(report["results"]["terminal"])
    write_report(report, args.report)
    chart_path = args.report.with_name(args.report.stem + "-convergence.png")
    plot_convergence(report["convergence"], chart_path)
    print(f"Relatório: {args.report}\nConvergência: {chart_path}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        import numpy as np

        plt.plot(np.linspace(0, config.horizon, config.steps + 1),
                 paths[:, :50], alpha=0.5)
        plt.title(f"Monte Carlo: {snapshot['ticker']}")
        plt.xlabel("Anos")
        plt.ylabel("Preço")
        plt.show()


if __name__ == "__main__":
    main()
