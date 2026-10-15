"""Render h3.2 presentation figures from saved aggregates and theory constants.

No datasets, posterior samples, model integrals, fits, subset analyses, sample
means, sample variances or confidence intervals are computed here. Existing
means, standard errors and interval endpoints are copied from the named inputs.
The only numerical evaluation is the already proved local-risk formula on a
display grid and its two analytical crossings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/humanlike_h3_2_20261015/figure_revision"
FIG = ROOT / "docs/07_papers_论文工作/ParticleJointSupervision/humanlike_h3_2_20261015/figures"
J1 = Path("artifacts/gaussian_remainder_v1_20261014/j1_attempt1")
LOCAL = Path("artifacts/local_shrinkage_v1_20261015/analysis_v1")
SELECTION = Path("artifacts/selection_limit_v1_20261014")

INPUT_ROLES = {
    J1 / "RESULTS.json": "four saved population-objective means and simultaneous interval endpoints",
    J1 / "FREEZE.json": "paired comparison parameters frozen before its new observations",
    J1 / "OLD_RISK_BINDING.json": "earlier same-model joint-KL scale estimates, not new paired-comparison risk samples",
    LOCAL / "PLOT_POINTS.csv": "16 saved method-by-condition scaled risk means and dataset-level SEs",
    LOCAL / "SUMMARY.json": "saved channel constants v, j and r_K8; no per-case analysis",
    SELECTION / "h1_attempt1/PARAMETERS.json": "previously computed endpoint v1 used to choose gamma",
    SELECTION / "h3_attempt1/FREEZE.json": "fixed t=16 and gamma after development, before finite-M data",
    SELECTION / "analysis_v1/H1_TABLE.md": "complete development candidate contrasts; read for design lineage only",
    Path("docs/08_research_研究工作/SelectionLimit_v1_20261014/PLAN_H1.md"): "pre-development candidate grid and development role",
    Path("docs/08_research_研究工作/SelectionLimit_v1_20261014/PLAN_H2_H3.md"): "post-development selection rule and pre-validation freeze",
    Path("docs/08_research_研究工作/GaussianRemainder_v1_20261014/PLAN.md"): "pre-J1 plan retaining the earlier development choice",
    Path("docs/08_research_研究工作/LocalShrinkage_v1_20261015/DESIGN_POWER_CHECK.md"): "pre-data signal design and distinction from exact finite-N crossings",
    Path("docs/08_research_研究工作/LocalShrinkage_v1_20261015/PROTOCOL_SELECTION.md"): "fixed numerical protocol, two-start risk means and non-nested conditions",
    Path("docs/08_research_研究工作/LocalShrinkage_v1_20261015/LOCAL_RISK_THEORY.md"): "analytical risk curves and crossing formulas, not new theoretical claims",
}

COLORS = {"group8": "#007F78", "ridge": "#C58A13", "integrated": "#315F9A", "I": "#6E607B"}
LABELS = {"group8": "Group posterior (K = 8)", "ridge": "Matched ridge",
          "integrated": "Integrated ML", "I": "Independence"}
MARKERS = {"group8": "^", "ridge": "D", "integrated": "o", "I": "X"}


def read(relative: Path):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 16, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .9, "savefig.facecolor": "white",
        "figure.facecolor": "white", "mathtext.fontset": "dejavusans",
    })


def figure1(rows: list[dict], out: Path):
    fig = plt.figure(figsize=(9.4, 5.6))
    ax = fig.add_axes([.14, .18, .83, .61])
    ylo, yhi = -.0047, .0071
    ax.axhspan(0, yhi, color="#DCE9F3", alpha=.40, zorder=0)
    ax.axhspan(ylo, 0, color="#F7E5D8", alpha=.45, zorder=0)
    ax.axhline(0, color="#5D6268", lw=1.2, zorder=1)
    handles = []
    for arm, offset, color, marker, label in [
        ("beta1", -.075, COLORS["integrated"], "o", r"Proposal with $\beta=1$"),
        ("critical", .075, "#C56632", "s", "Tempered proposal"),
    ]:
        chosen = [r for r in rows if r["arm"] == arm]
        x = np.array([0 if r["M"] == 32 else 1 for r in chosen], dtype=float) + offset
        means = np.array([r["mean"] for r in chosen])
        intervals = np.array([r["interval"] for r in chosen])
        ax.errorbar(x, means, yerr=np.stack([means-intervals[:, 0], intervals[:, 1]-means]),
                    fmt=marker, markersize=8.5, capsize=6, elinewidth=1.8,
                    markeredgewidth=1.4, color=color, zorder=4)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", markersize=8.5, label=label))
    ax.text(.5, .0035, "Positive: favors independent truth", ha="center", va="center",
            fontsize=12.5, color="#244F73", fontweight="normal")
    ax.text(.5, -.0013, "Negative: favors correlated candidate", ha="center", va="center",
            fontsize=12.5, color="#9C4F26", fontweight="normal")
    ax.set(xlim=(-.24, 1.24), ylim=(ylo, yhi), xticks=[0, 1], xticklabels=[r"$M=32$", r"$M=128$"],
           xlabel="Private observations per group",
           ylabel="Objective(correlated) − objective(true)\n(nats per complete observation)")
    ax.set_yticks([-.004, -.002, 0, .002, .004, .006])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Changing only the proposal reverses the preference", y=.97, fontsize=16.5)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .91), ncol=2,
               frameon=False, columnspacing=2.2, fontsize=13)
    fig.text(.55, .035, "Bars: 95% approximate simultaneous intervals", ha="center", fontsize=11.5)
    fig.savefig(out / "proposal_reversal.png", dpi=240)
    plt.close(fig)


def method_handles():
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="none",
                   markerfacecolor="white" if m == "ridge" else COLORS[m],
                   markeredgewidth=1.5, markersize=8.5, label=LABELS[m])
            for m in ("group8", "ridge", "integrated", "I")]


def theory_handles():
    return [Line2D([], [], color=COLORS[m], linestyle="--", linewidth=1.8, label=label)
            for m, label in (("group8", "Group posterior + matched ridge"),
                             ("integrated", "Integrated ML"), ("I", "Independence"))]


def plot_theory(ax, grid, theory, limits):
    ax.axvspan(limits[0], limits[1], color="#65A799", alpha=.14, zorder=0)
    for x in limits:
        ax.axvline(x, color="#668C82", linewidth=1.1, linestyle=(0, (2.5, 3)), zorder=1)
    for key in ("I", "integrated", "group8"):
        ax.plot(grid, theory[key], linestyle="--", linewidth=1.8, color=COLORS[key], zorder=2)


def plot_programs(ax, rows):
    for m in ("I", "integrated", "ridge", "group8"):
        chosen = [r for r in rows if r["method"] == m]
        ax.errorbar([r["S"] for r in chosen], [r["scaled_mean"] for r in chosen],
                    yerr=[r["scaled_SE"] for r in chosen], fmt=MARKERS[m],
                    color=COLORS[m], markerfacecolor="white" if m == "ridge" else COLORS[m],
                    markeredgewidth=1.4, markersize=8.5 if m != "group8" else 7.3,
                    capsize=5.5 if m == "ridge" else 3.6,
                    elinewidth=1.3, zorder=5 if m == "group8" else 4)


def figure2(rows, grid, theory, limits, out: Path):
    fig = plt.figure(figsize=(10.0, 6.7))
    ax = fig.add_axes([.095, .21, .88, .57])
    plot_theory(ax, grid, theory, limits)
    plot_programs(ax, [r for r in rows if r["N"] == 2048])
    ax.text(3.2, 5.15, "Local theory:\ngroup + ridge below both\nML and independence",
            ha="center", va="center", fontsize=12, color="#245E52")
    ax.text(limits[0]+.08, 6.42, r"$S_I=0.5717$", fontsize=11.5, color="#245E52", ha="left")
    ax.text(limits[1]+.12, 6.42, r"$S_{\rm ML}=5.8251$", fontsize=11.5, color="#245E52", ha="left")
    ax.set(xlim=(-.28, 12.45), ylim=(-.16, 6.7), xticks=[0, 2, 4, 6, 8, 10, 12],
           yticks=[0, 1, 2, 3, 4, 5, 6], xlabel=r"Local signal $S$", ylabel=r"$N\,\times$ joint KL risk")
    ax.grid(axis="y", color="#B7BDC3", alpha=.30, linewidth=.6)
    fig.suptitle(r"Risk across three prespecified signal conditions ($N=2048$)", y=.982, fontsize=16)
    fig.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.535, .935), ncol=4,
               title="Fixed programs: mean ± 1 SE, 24 independent datasets", title_fontsize=12,
               fontsize=11.5, frameon=False, handletextpad=.45, columnspacing=1.2)
    fig.legend(handles=theory_handles(), loc="center", bbox_to_anchor=(.54, .077), ncol=3,
               title="Local asymptotic expected-risk limits (dashed)", title_fontsize=11.5,
               fontsize=11, frameon=False, columnspacing=1.7, handlelength=2.7)
    fig.text(.54, .013, "Thresholds and shading describe local theory, not measured finite-N crossings.",
             ha="center", fontsize=10.5, color="#4E555C")
    fig.savefig(out / "local_risk.png", dpi=240)
    plt.close(fig)


def figure3(rows, grid, theory, limits, out: Path):
    # This descriptive condition has a different teacher; do not join it to N2048.
    fig = plt.figure(figsize=(9.5, 5.7))
    ax = fig.add_axes([.105, .22, .865, .52])
    plot_theory(ax, grid, theory, limits)
    plot_programs(ax, [r for r in rows if r["N"] == 256])
    ax.set(xlim=(-.25, 6.4), ylim=(-.12, 3.5), xticks=[0, 1, 2, 3, 4, 5, 6],
           xlabel=r"Local signal $S$", ylabel=r"$N\,\times$ joint KL risk")
    ax.grid(axis="y", color="#B7BDC3", alpha=.30, linewidth=.6)
    ax.text(4.5, .45, "One descriptive condition\nat S = 2", ha="center", fontsize=12, color="#4E555C")
    fig.suptitle(r"Descriptive condition ($N=256$): a different teacher", y=.982, fontsize=16)
    fig.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.535, .925), ncol=4,
               title="Fixed programs: mean ± 1 SE, 24 independent datasets", title_fontsize=12,
               fontsize=11, frameon=False, handletextpad=.35, columnspacing=1.05)
    fig.legend(handles=theory_handles(), loc="center", bbox_to_anchor=(.54, .084), ncol=3,
               title="Local asymptotic limits (dashed); shaded region defined by local theory",
               title_fontsize=11, fontsize=10.5, frameon=False, columnspacing=1.3)
    fig.text(.54, .015, "Independent data; not a nested sample-size comparison with N = 2048.",
             ha="center", fontsize=10.5, color="#4E555C")
    fig.savefig(out / "local_risk_n256.png", dpi=240)
    plt.close(fig)


def run(run_name: str, publish: bool):
    out = ART / run_name
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one directory name")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "h32_figures_source.py")
    j1 = read(J1 / "RESULTS.json")
    j1_freeze = read(J1 / "FREEZE.json")
    old_risk = read(J1 / "OLD_RISK_BINDING.json")
    local = read(LOCAL / "SUMMARY.json")
    parameters = read(SELECTION / "h1_attempt1/PARAMETERS.json")
    h3 = read(SELECTION / "h3_attempt1/FREEZE.json")
    figure1_rows = [{"M": row["M"], "arm": row["arm"],
                     "mean": row["statistics"]["mean"], "SE": row["statistics"]["se"],
                     "interval": row["interval"], "sample_count": row["statistics"]["banks"],
                     "beta": next(c for c in j1["contexts"] if c["M"] == row["M"])["arms"][row["arm"]]["parameters"]["beta"]}
                    for row in j1["primary"]]
    with (ROOT / LOCAL / "PLOT_POINTS.csv").open(encoding="utf-8", newline="") as stream:
        figure2_rows = [{"cell_id": row["cell_id"], "N": int(row["N"]), "S": float(row["S"]),
                         "method": row["method"], "scaled_mean": float(row["scaled_mean"]),
                         "scaled_SE": float(row["scaled_SE"])} for row in csv.DictReader(stream)]
    r = local["theory_coefficients"]["r_K8"]
    limits = [3*r/(2-r), 3*(1+r)/(1-r)]
    grid = np.linspace(0, 12.5, 501)
    theory = {"group8": .5*((1-r)**2*grid+3*r*r), "integrated": np.full_like(grid, 1.5), "I": .5*grid}
    input_binding = {str(p).replace("\\", "/"): {"sha256": digest(ROOT / p), "bytes": (ROOT / p).stat().st_size, "role": role}
                     for p, role in INPUT_ROLES.items()}
    # Bind the actual earlier risk sources as well as the old binding record.
    for entry in old_risk.values():
        p = ROOT / entry["source_path"]
        if digest(p) != entry["sha256"]:
            raise ValueError("Earlier risk source no longer matches its archived binding")
        input_binding[entry["source_path"]] = {"sha256": digest(p), "bytes": p.stat().st_size,
                                               "role": "earlier complete-joint-risk result, cited for scale only"}
    # Only identity checks here; no recomputation of scientific statistics.
    if len(figure1_rows) != 4 or len(figure2_rows) != 16:
        raise ValueError("Unexpected saved aggregate layout")
    if j1_freeze["gamma"] != h3["gamma"]:
        raise ValueError("Design lineage gamma differs")
    if parameters["selected"]["v1"] != h3["v1_from_H1"]:
        raise ValueError("Design lineage v1 differs")
    write(out / "INPUTS_USED.json", {"sources": input_binding, "figure1": figure1_rows,
          "figure2_and_appendix": figure2_rows, "theory_coefficients": local["theory_coefficients"],
          "analytical_crossings": {"S_I": limits[0], "S_ML": limits[1]},
          "old_risk_scale": old_risk})
    write(out / "DESIGN_LINEAGE.json", {
        "gamma": h3["gamma"], "design_t": h3["t"], "v1": h3["v1_from_H1"],
        "formula": "gamma=sqrt(t/v1); arithmetic correspondence only, no new parameter selection",
        "H1_pre_data_grid": {"t": [0, 1, 4, 16, 64], "eta_multiplier": [0, 1, 4],
                             "teachers": ["independent", "interior"], "candidates": ["equal", "asymmetric"]},
        "choice_timing": "H1 was development. After viewing H1 results, PLAN_H2_H3 recorded a selection rule and froze H2/H3 before their first random draws. Do not call this rule preregistered before H1.",
        "recorded_selection": "For independent teacher, equal(.45) candidate and natural coupled noise eta0, choose the smallest t in {1,4,16,64} with development controlled mean+3SE<0; selected t=16.",
        "J1_timing": "Later GaussianRemainder J1 fixed the earlier gamma before new paired two-proposal observations at M32/M128. Both proposal results in Figure 1 use those new J1 observations.",
        "not_claimed": ["constant supplied quantitatively by the existence theorem", "selection before any development outcomes", "newly selected from plotted J1 results", "finite-M beta changes only a variance", "risk reference newly sampled for J1"],
        "caption_suggestion_en": "The tempering level was chosen in a separate development study and fixed before this paired comparison.",
        "caption_suggestion_zh": "温度水平由另一开发研究选定，并在本次配对比较前固定。",
    })
    style()
    figure1(figure1_rows, out)
    figure2(figure2_rows, grid, theory, limits, out)
    figure3(figure2_rows, grid, theory, limits, out)
    with (out / "THEORY_DISPLAY_GRID.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["S", "group_and_ridge_local_limit", "integrated_local_limit", "independence_local_limit"])
        writer.writerows(zip(grid, theory["group8"], theory["integrated"], theory["I"]))
    outputs = ["proposal_reversal.png", "local_risk.png", "local_risk_n256.png"]
    if publish:
        FIG.mkdir(parents=True, exist_ok=True)
        for name in outputs:
            shutil.copyfile(out / name, FIG / name)
    manifest = {
        "schema": "h32_saved_aggregate_figure_render_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": run_name, "producer": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "producer_sha256": digest(Path(__file__)), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "input_sources": input_binding,
        "scientific_operations": {"new_observations": 0, "RNG_calls": 0, "model_integrations": 0,
                                  "fits": 0, "subset_statistics": 0, "sample_statistics_recomputed": 0,
                                  "confidence_intervals_recomputed": 0},
        "permitted_arithmetic": "Render saved means/SEs/CI endpoints; subtract endpoints from stored mean for plotting error-bar lengths; evaluate fixed analytical local-risk curves and crossings.",
        "figure_scopes": {
            "proposal_reversal.png": "four saved approximate normal-Bonferroni simultaneous intervals; objective differences, not risk changes",
            "local_risk.png": "all twelve saved N2048 method/condition mean+SE records; local theoretical crossings and shared group/ridge region; no universal optimality claim",
            "local_risk_n256.png": "all four saved N256 mean+SE records in a separate descriptive figure; different teacher and non-nested data"
        },
        "visual_review": "pending direct image inspection; successful rendering alone is not visual acceptance",
        "published_to": str(FIG.relative_to(ROOT)).replace("\\", "/") if publish else None,
        "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()},
    }
    write(out / "MANIFEST.json", manifest)
    print(json.dumps({"run": str(out.relative_to(ROOT)), "outputs": outputs, "published": publish,
                      "crossings": limits, "new_scientific_execution": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    run(args.run_name, args.publish)
