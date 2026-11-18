"""Style commun des figures (palette catégorielle validée, ordre fixe)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10.5, "axes.titlesize": 12,
        "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
    })


def errorbar_ci(ax, x, results, color, label, key="score", marker="o"):
    """Trace le score (ou taux) avec l'intervalle de confiance de Wilson à 95 %."""
    y = [r[key] for r in results]
    lo = [max(0.0, r[key] - r["ci95"][0]) for r in results]
    hi = [max(0.0, r["ci95"][1] - r[key]) for r in results]
    ax.errorbar(x, y, yerr=[lo, hi], color=color, label=label, marker=marker, capsize=3,
                elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
