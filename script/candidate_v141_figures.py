"""Plot existing finite-instance summaries; no sampling, scoring or inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_4_1_20261006"
OUT = ROOT / "artifacts/submission_candidate_v1_4_1_20261006"
OLD = DOC.parent / "submission_candidate_v1_4_20261006"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (OUT / "VERSION_LOCK.json").exists()
    source_paths = [
        ROOT / "artifacts/nuisance_contrast_v1_20261006/POPULATION.json",
        ROOT / "artifacts/local_geometry_v1_20261006/STATISTICS.json",
        ROOT / "artifacts/nuisance_contrast_v1_20261006/RESULTS.json",
    ]
    gap, curvature, result = [json.loads(p.read_text("utf-8")) for p in source_paths]
    rows = [gap["results"][:2], curvature["results"][:2]]
    assert [r["contrast"] for r in rows[0]] == ["g0", "g256"]
    assert [r["contrast"] for r in rows[1]] == ["h0", "h256"]
    assert gap["alpha"] == curvature["alpha"] == .01
    assert gap["banks"] == curvature["banks"] == 262144
    assert result["selected"][0]["candidate"] == [0, 0, 0]
    assert result["selected"][1]["candidate"] == [9, 9, 9]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#83919c",
        "axes.labelcolor": "#233340", "text.color": "#233340",
        "xtick.color": "#445462", "ytick.color": "#233340",
        "savefig.facecolor": "white", "svg.hashsalt": "finite-instance-v141",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.15))
    colors = ["#2d6d96", "#bd583a"]
    for ax, panel in zip(axes, rows):
        for i, row in enumerate(panel):
            mean = row["mean"]; lo, hi = row["interval"]
            ax.errorbar(mean, 1-i, xerr=np.array([[mean-lo], [hi-mean]]),
                        fmt="o", markersize=7, color=colors[i], ecolor=colors[i],
                        elinewidth=2, capsize=4, capthick=1.5, zorder=3)
        ax.axvline(0, color="#66737e", linewidth=1, linestyle="--", zorder=1)
        ax.set_yticks([1, 0], ["$n=0$", "$n=256$"])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_ylim(-.55, 1.55)
        ax.grid(axis="x", color="#e4e9ed", linewidth=.7)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_title("(a) Population anchor gap", loc="left", fontsize=11, pad=12)
    axes[0].set_xlabel(r"$g_n=F_n(I)-F_n(C_b)$", labelpad=8)
    axes[0].set_xlim(-.036, .013)
    axes[0].set_xticks([-.03, -.02, -.01, 0, .01])
    axes[0].text(.5, 1.02, "positive: correlated candidate preferred",
                 transform=axes[0].transAxes, ha="center", va="bottom", fontsize=8.2)
    axes[1].set_title("(b) Curvature at the independent truth", loc="left", fontsize=11, pad=12)
    axes[1].set_xlabel(r"$h_n$ in $\nabla^2 F_n(I)=h_n I_3$", labelpad=8)
    axes[1].set_xlim(-.008, .033)
    axes[1].set_xticks([0, .01, .02, .03])
    axes[1].text(.5, 1.02, "negative: local maximum; positive: local minimum",
                 transform=axes[1].transAxes, ha="center", va="bottom", fontsize=8.2)
    fig.text(.5, .065,
             r"Saved matched selection: $n=0 \rightarrow I$, KL $=0$"
             r"    |    $n=256 \rightarrow C_b$, KL $=.03110309$",
             ha="center", va="center", fontsize=9.5)
    fig.text(.5, .005,
             "Each panel: its own 99% simultaneous family; across panels: at least 98% (numerically qualified).",
             ha="center", va="bottom", fontsize=8.2)
    fig.subplots_adjust(left=.08, right=.985, top=.77, bottom=.28, wspace=.34)
    figure_dir = DOC / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "finite_instance.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "finite_instance.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    old_outputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in figure_dir.iterdir()
                   if p.is_file() and not p.name.startswith("finite_instance.")}
    for name, digest in old_outputs.items():
        assert digest == sha(OLD / "figures" / Path(name).name)
    outputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(figure_dir.iterdir()) if p.is_file()}
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), sources={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        outputs=outputs, main_figures=["finite_instance"],
        supplementary_figures=["representation_structure", "overlap_coupling",
                               "preference_geometry", "learning_risk", "readout_and_selection"],
        data={"anchor_gap": rows[0], "curvature": rows[1]},
        per_panel_family_coverage=.99, cross_panel_family_lower_coverage=.98,
        within_panel_third_prespecified_contrast_omitted_from_plot=True,
        selection_risk_annotations_are_saved_program_outcomes=True,
        numerical_qualification_retained=True,
        new_scientific_computation=False, new_samples=0, new_scores=0, new_integrals=0,
        old_figure_bytes_preserved=True,
    )
    (OUT / "FIGURES.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"new_figure": "finite_instance", "panels": record["data"],
                      "new_scientific_computation": False}, ensure_ascii=True))


if __name__ == "__main__":
    main()
