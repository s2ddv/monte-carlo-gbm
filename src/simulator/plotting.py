"""Convergence figures; saving works without an interactive display."""

from pathlib import Path


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
