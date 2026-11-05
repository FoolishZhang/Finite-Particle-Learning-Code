# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Drawing function projected from the manuscript asset builder; saved points only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
STRUCTURES = ("L", "A", "B", "JNT")
NAMES_EN = ("A − L, all", "B − A, within", "B − A, cross", "JNT − B, cross")


def figure(result, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.13, 1], hspace=.58, wspace=.43,
                           left=.055, right=.985, bottom=.13, top=.91)
    blue, orange = "#2065a7", "#c6681a"
    fig.text(.055, .975, "(a) Input access and joint outputs", weight="bold", va="top", fontsize=9)
    for k, q in enumerate(STRUCTURES):
        ax = fig.add_subplot(grid[0, k])
        ax.set(xlim=(-.65, 7.65), ylim=(-.55, 2.45))
        ax.axis("off")
        ax.text(3.5, 2.30, q, ha="center", weight="bold", fontsize=10)
        ax.text(3.5, 1.94, "Own input" if q == "L" else "All inputs", ha="center")
        for i in range(8):
            ax.add_patch(Circle((i, 1.50), .12, facecolor="#b9cadb", edgecolor="white", lw=.4))
        if q == "L":
            for i in range(8):
                ax.annotate("", (i, .50), (i, 1.33), arrowprops={"arrowstyle": "->", "lw": .7, "color": "#53697b"})
        else:
            ax.plot([0, 7], [1.02, 1.02], color="#53697b", lw=.8)
            for i in range(8):
                ax.plot([i, i], [1.36, 1.02], color="#53697b", lw=.65)
                ax.annotate("", (i, .50), (i, 1.02), arrowprops={"arrowstyle": "->", "lw": .65, "color": "#53697b"})
        groups = [[i] for i in range(8)] if q in ("L", "A") else [[i, i + 1] for i in range(0, 8, 2)] if q == "B" else [list(range(8))]
        for block in groups:
            ax.add_patch(FancyBboxPatch((min(block) - .34, .02), max(block) - min(block) + .68, .49,
                                       boxstyle="round,pad=0.025,rounding_size=0.12",
                                       edgecolor=blue, facecolor="#e9f2f9", lw=.9))
        for i in range(8):
            ax.text(i, .265, str(i + 1), ha="center", va="center", fontsize=6.5, color=blue)
        ax.text(3.5, -.33, "Product output" if q in ("L", "A") else "Four pair blocks" if q == "B" else "Joint output",
                ha="center", fontsize=8)
    fig.text(.055, .565, "(b) Selective response increments", weight="bold", fontsize=9)
    handles = [Line2D([], [], marker="D", linestyle="none", color=orange, markersize=4, label="Leading prediction"),
               Line2D([], [], marker="o", color=blue, markersize=4, label="Mean + Bonferroni–Student CI")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .54), frameon=False, ncol=2,
               columnspacing=1.3, handletextpad=.5, fontsize=7.4)
    for k, (key, st) in enumerate(result["statistics"].items()):
        ax = fig.add_subplot(grid[1, k])
        cross = st["kind"] == "equivalence"
        scale = 1000 if cross else 1
        values = np.array(list(st["values_by_data_id"].values())) * scale
        mean, pred = st["mean"] * scale, st["prediction"] * scale
        lo, hi = np.array(st["simultaneous_95_CI"]) * scale
        span = max(hi - lo, values.max() - values.min())
        center = (max(values.max(), hi, pred) + min(values.min(), lo, pred)) / 2
        ax.set_xlim(center - .72 * span, center + .72 * span)
        ax.scatter(values, np.linspace(.06, .26, len(values)), s=8, color="#8998a4", alpha=.8, zorder=2)
        ax.errorbar(mean, .63, xerr=[[mean - lo], [hi - mean]], color=blue, marker="o", ms=4.5,
                    capsize=3, lw=1.5, zorder=4)
        ax.scatter([pred], [.90], marker="D", s=23, color=orange, zorder=5)
        if cross:
            ax.axvline(0, color="#777777", ls=":", lw=.7)
        ax.set_ylim(-.08, 1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(NAMES_EN[k], fontsize=8, pad=6)
        ax.set_xlabel(r"$\Delta\rho\; (\times 10^{-3})$" if cross else r"$\Delta\rho$", fontsize=8, labelpad=3)
        ax.locator_params(axis="x", nbins=3)
        ax.tick_params(axis="x", labelsize=7, pad=2)
        ax.grid(axis="x", alpha=.12, lw=.5)
    fig.text(.055, .023, "12 independent paired datasets; each panel has its own horizontal scale.", fontsize=7.5)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / "neural_structure.png", dpi=320)
    fig.savefig(OUT / "neural_structure.svg", dpi=320)
    plt.close(fig)
