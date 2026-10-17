"""Plot all 24 saved paired cases; no model evaluation or statistical selection."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(analysis, out):
    source = analysis / "RISK_ANALYSIS.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["paired_case_rows"]
    if [r["case_id"] for r in rows] != list(range(2000, 2024)):
        raise ValueError("Exactly all original cases in fixed ID order required")
    if any(data["primary"][q]["simultaneous_interval"] is None for q in ("D_H", "G")):
        raise ValueError("Unresolved primary analysis: do not plot as qualified")
    d0 = np.array([r["D_P0"] for r in rows], dtype=float)
    dh = np.array([r["D_H"] for r in rows], dtype=float)
    g = np.array([r["G"] for r in rows], dtype=float)
    if not np.isfinite(np.r_[d0, dh, g]).all():
        raise ValueError("Incomplete paired values")
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "budget-sensitivity-v1"})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    lo, hi = min(d0.min(), dh.min()), max(d0.max(), dh.max())
    margin = max(.08 * (hi - lo), .01)
    limits = (lo - margin, hi + margin)
    ax[0].plot(limits, limits, color="#8b949e", lw=1, ls="--", label="No change")
    ax[0].scatter(d0, dh, s=34, color="#1769aa", alpha=.85, zorder=3)
    ax[0].set(xlim=limits, ylim=limits,
              xlabel=r"Original $D^{P0}$  ($N\times$ joint KL difference)",
              ylabel=r"Higher budget $D^H$", title="A. Same 24 datasets, same K")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].text(.04, .81, f"Mean original: {d0.mean():.5f}\nMean higher budget: {dh.mean():.5f}",
               transform=ax[0].transAxes, va="top", fontsize=9)
    ci = data["primary"]["G"]["simultaneous_interval"]
    ax[1].axhspan(ci["lower"], ci["upper"], color="#f0b36a", alpha=.3,
                 label="Mean-change interval (family 2)")
    ax[1].axhline(ci["mean"], color="#ba5a16", lw=1.3, label="Mean change")
    ax[1].axhline(0, color="#8b949e", lw=1, ls="--")
    ax[1].scatter(np.arange(24), g, s=32, color="#1769aa", zorder=3, label="Individual dataset")
    ax[1].set(xlim=(-1, 24), xlabel="Original dataset ID",
              ylabel=r"Paired change $G=D^H-D^{P0}$", title="B. Changes retain every output")
    ax[1].set_xticks(np.arange(0, 24, 3), [str(x) for x in range(2000, 2024, 3)])
    ax[1].legend(frameon=False, fontsize=8, loc="best")
    for a in ax:
        a.grid(alpha=.15)
        a.set_axisbelow(True)
    fig.suptitle("Retrospective numerical-budget sensitivity | strong signal, N=2048, K=8", fontsize=12)
    fig.savefig(out / "paired_budget.png", dpi=220)
    fig.savefig(out / "paired_budget.svg", metadata={"Date": None})
    plt.close(fig)
    receipt = {"source_sha256": sha(Path(__file__)), "risk_analysis_sha256": sha(source),
               "case_ids": [r["case_id"] for r in rows], "outputs_filtered": False,
               "risk_unit": "N times joint KL; mean two start risks before dataset comparison",
               "interval_scope": "Approximate family-2 simultaneous 95% t interval, df23; not equivalence",
               "figure_scope": "Descriptive saved statistics; no new scientific evaluation",
               "files": {name: sha(out / name) for name in ("paired_budget.png", "paired_budget.svg")}}
    (out / "FIGURE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.analysis, args.out)
