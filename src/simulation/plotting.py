"""Price paths, terminal distributions and convergence figures."""

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np


def _validate_paths(paths: np.ndarray) -> np.ndarray:
    """Return a finite price matrix with a shared initial price."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] == 0:
        raise ValueError(
            "paths must be 2D with at least 2 rows and 1 simulation."
        )
    if not np.isfinite(paths).all():
        raise ValueError("paths must contain only finite prices.")
    if not np.all(paths[0] == paths[0, 0]):
        raise ValueError("All simulations must share the same initial price.")
    return paths


def _save_figure(save_path: str | Path | None) -> None:
    """Save the current figure, creating parent directories when needed."""
    if save_path is not None:
        import matplotlib.pyplot as plt

        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(destination, dpi=150)


def plot_paths(
    paths: np.ndarray,
    n_display: int = 50,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the first n_display paths without opening a window.

    Args:
        paths: Prices shaped (steps + 1, n_sims), including initial prices.
        n_display: Positive maximum number of paths to display.
        title: Optional chart title.
        save_path: Optional output file; parent directories are created.

    Returns:
        The figure, left open for the caller to display or close.

    Raises:
        ValueError: If paths or n_display are invalid.
    """
    paths = _validate_paths(paths)
    if (isinstance(n_display, (bool, np.bool_))
            or not isinstance(n_display, (int, np.integer))
            or n_display <= 0):
        raise ValueError("n_display must be a positive integer.")

    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.plot(paths[:, :n_display], color="steelblue", alpha=0.3, linewidth=0.8)
    ax.set_title(title if title is not None else "GBM simulated price paths")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    _save_figure(save_path)
    return figure


def plot_final_distribution(
    paths: np.ndarray,
    var: float | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot terminal prices and reference lines without opening a window.

    Args:
        paths: Prices shaped (steps + 1, n_sims), including initial prices.
        var: Optional loss VaR in price units; its price line is s0 - var.
        title: Optional chart title.
        save_path: Optional output file; parent directories are created.

    Returns:
        The figure, left open for the caller to display or close.

    Raises:
        ValueError: If paths are invalid or var is nonfinite.
    """
    paths = _validate_paths(paths)
    if var is not None and not np.isfinite(var):
        raise ValueError("var must be finite when provided.")

    import matplotlib.pyplot as plt

    s0 = paths[0, 0]
    final_prices = paths[-1]
    figure, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.hist(final_prices, bins=50, color="steelblue", alpha=0.75)
    ax.axvline(s0, color="black", linestyle="--", label=f"Initial: {s0:.2f}")
    mean = final_prices.mean()
    ax.axvline(mean, color="darkgreen", label=f"Mean: {mean:.2f}")
    if var is not None:
        var_price = s0 - var
        ax.axvline(var_price, color="firebrick", linestyle=":",
                   label=f"Price at VaR: {var_price:.2f}")
    ax.set_title(title if title is not None else "GBM final price distribution")
    ax.set_xlabel("Final price")
    ax.set_ylabel("Number of simulations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(save_path)
    return figure


def plot_convergence(rows, path):
    # Figure + Agg avoids requiring a desktop backend for --no-plot runs.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    counts = [row["n_sims"] for row in rows]
    for ax, metric in zip(axes[:2], ("mean", "variance")):
        ax.plot(counts, [row[metric]["absolute_error"] for row in rows],
                "o-", label="Observed absolute error")
        ax.set_title(f"Terminal {metric}")
    axes[0].plot(counts, [row["mean_standard_error"] for row in rows],
                 "--", label="Theoretical mean SE (1/sqrt(N))")
    for p in rows[0]["quantiles"]:
        axes[2].plot(counts, [row["quantiles"][p]["absolute_error"] for row in rows],
                     "o-", label=f"p={p}")
    axes[2].set_title("Terminal quantiles")
    for index, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlabel("Number of simulations (nested samples)")
        ax.set_ylabel("Absolute error (price²)" if index == 1 else "Absolute error (price)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return figure
