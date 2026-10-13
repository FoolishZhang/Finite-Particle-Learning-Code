"""Prespecified SCBM API figures from complete frozen outputs, without new science.

The original four-gate intervals are reused, never recomputed as a plot family.
Auxiliary K comparisons use the same first 1024 observations. Numerical controls
remain a qualification supplement, including failures; they are not learned fixes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/scbm_invariance"
DEST = OUT / "figures"
SOURCE = Path(__file__).resolve()
PADDINGS = [0, 64, 256]
SCOPES = ["K100_all", "K100_prefix1024", "K400_prefix1024"]
NAMES = ["K100_base_high_minus_truth", "K100_pad256_high_minus_truth",
         "K100_padding_correlation_gradient_shift",
         "K100_pad256_nuisance_scale_gradient_per_coordinate"]
COLORS = dict(raw="#BA4855", factorized="#168472", aux="#386DA5", exact="#54545A")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def stats(values):
    a = np.asarray(values, float)
    assert a.shape == (16,) and np.all(np.isfinite(a))
    return dict(stream_values=a.tolist(), mean=float(a.mean()),
                SE=float(a.std(ddof=1) / math.sqrt(16)))


def close(a, b):
    assert np.allclose(a, b, rtol=2e-12, atol=2e-14), (a, b)


def load_complete():
    # RESULTS is only written after all sixteen streams have completed.
    paths = [OUT / "RUN/RESULTS.json", OUT / "RUN/FREEZE.json",
             OUT / "REFERENCE/MODEL.json"]
    assert not (OUT / "VERSION_LOCK.json").exists(), "Package already sealed"
    assert not (OUT / "RUN/ERROR.json").exists(), "Do not plot an interrupted run"
    data, freeze, reference = [read(p) for p in paths]
    assert data["freeze_sha256"] == sha(paths[1])
    assert data["reference_sha256"] == freeze["reference_sha256"] == sha(paths[2])
    assert data["source_sha256"] == freeze["source_sha256"] == sha(ROOT / "script/scbminv_run.py")
    assert data["kernel_sha256"] == freeze["kernel_sha256"] == sha(ROOT / "script/scbminv_kernel.py")
    assert data["config"] == freeze["config"] and data["fixture"] == freeze["fixture"]
    assert data["config"]["paddings"] == PADDINGS
    assert data["config"]["candidate_t"] == [-.75, 0., .75]
    assert data["config"]["observations"] == 4096 and data["config"]["aux_observations"] == 1024
    assert data["config"]["K_main"] == 100 and data["config"]["K_aux"] == 400
    assert [r["stream"] for r in data["records"]] == list(range(16))
    assert len(data["primary_family"]) == data["family_size"] == 4
    assert data["alpha"] == .05 and data["df"] == 15
    assert [r["name"] for r in data["primary_family"]] == NAMES
    assert len(data["descriptive_score_tables"]) == 27
    assert len(data["descriptive_gradient_tables"]) == 9
    assert len(data["numerical64"]) == 288 and len(data["scale_controls"]) == 384
    assert reference["qualification_passed"]
    paths += [ROOT / freeze["design_path"], ROOT / "script/scbminv_run.py",
              ROOT / "script/scbminv_kernel.py"]
    assert sha(paths[3]) == freeze["design_sha256"]
    bindings = {rel(p): sha(p) for p in paths}
    return data, reference, bindings


def collect_points(data, reference):
    records = data["records"]
    primary = {r["name"]: r for r in data["primary_family"]}
    gap_rows = []
    for scope in SCOPES:
        summaries = [r["summaries"][scope] for r in records]
        expected_rows = 4096 if scope == "K100_all" else 1024
        assert all(s["rows"] == expected_rows for s in summaries)
        for ni, n in enumerate(PADDINGS):
            row = dict(scope=scope, padding=n, rows_per_stream=expected_rows,
                       raw=stats([s["raw_high_minus_truth"][ni] for s in summaries]),
                       factorized=stats([s["factorized_high_minus_truth"] for s in summaries]),
                       displayed_interval=None)
            table = next(r for r in data["descriptive_score_tables"]
                         if r["scope"] == scope and r["padding"] == n and r["t"] == .75)
            for method in ("raw", "factorized"):
                for key in ("mean", "SE"):
                    close(row[method][key], table[method + "_contrast_to_truth"][key])
            if scope == "K100_all" and n in (0, 256):
                gate = primary[NAMES[0 if n == 0 else 1]]
                close(row["raw"]["stream_values"], gate["stream_values"])
                row["displayed_interval"] = gate.copy()
            gap_rows.append(row)
    gradients = []
    s = [r["summaries"]["K100_all"] for r in records]
    for ni, n in enumerate(PADDINGS):
        raw_direction = stats([r["raw_correlation_direction"][ni] for r in s])
        base_direction = stats([r["raw_correlation_direction"][0] for r in s])
        shift = stats([r["raw_correlation_direction"][ni] - r["raw_correlation_direction"][0] for r in s])
        # The source uses zero as an n=0 array placeholder; normalized derivative
        # is undefined there, and is exported as JSON null rather than as zero.
        scale = stats([r["raw_gradient"][ni][-1] / n for r in s]) if n else None
        row = dict(scope="K100_all", padding=n, raw_correlation_direction=raw_direction,
                   factorized_correlation_direction=base_direction,
                   correlation_gradient_shift=shift,
                   nuisance_scale_gradient_per_coordinate=scale,
                   normalized_scale_display="NA" if n == 0 else "value",
                   correlation_interval=None, scale_interval=None)
        if n == 256:
            close(shift["stream_values"], primary[NAMES[2]]["stream_values"])
            close(scale["stream_values"], primary[NAMES[3]]["stream_values"])
            row["correlation_interval"] = primary[NAMES[2]].copy()
            row["scale_interval"] = primary[NAMES[3]].copy()
        gradients.append(row)
    high = next(r for r in reference["records"] if r["t"] == .75)
    truth = next(r for r in reference["records"] if r["t"] == 0.)
    p, q = np.asarray(truth["main_pmf"]), np.asarray(high["main_pmf"])
    known_kl = float(np.sum(p * np.log(p / q)))
    close(known_kl, high["reference_joint_KL"])
    # Check displayed intervals against their stored stream means. This is a
    # consistency check of existing intervals, not an added inference family.
    for gate in primary.values():
        st = stats(gate["stream_values"])
        close(st["mean"], gate["mean"])
        close(st["SE"], gate["SE"])
        close(gate["lower"], st["mean"] - data["critical_t"] * st["SE"])
        close(gate["upper"], st["mean"] + data["critical_t"] * st["SE"])
    return dict(candidate_contrast=dict(high_t=.75, truth_t=0., lower_is_better=True),
                gap_points=gap_rows, gradient_points=gradients,
                numerical_exact_task_reference=dict(direction="D(Q_truth || Q_high)",
                    value=known_kl, padding_invariant=True,
                    qualification="Frozen PMF quadrature reference, not a rigorous integration certificate"),
                original_primary_family=data["primary_family"], critical_t=data["critical_t"],
                interval_scope="Original four-gate Bonferroni two-sided t development approximation, 16 independent streams; no new family or CI",
                auxiliary_scope="Paired first 1024 observations per stream, K100 nested in K400; means only",
                scope="Fixed-point official concept-loss API candidate ranking and feasible gradient directions; no optimizer or neural-network training")


def style():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "grid.alpha": .18,
        "svg.hashsalt": "scbminv-fixed-api-panels", "savefig.facecolor": "white"})


def export(fig, name):
    outputs = {}
    for suffix in ("png", "svg"):
        p = DEST / f"{name}.{suffix}"
        assert not p.exists(), "Preserve completed figures; any layout revision needs an explicit history snapshot"
        fig.savefig(p, dpi=235, metadata={"Date": None} if suffix == "svg" else None)
        outputs[rel(p)] = sha(p)
    plt.close(fig)
    return outputs


def panel_axes(ax, title, ylabel):
    ax.set(title=title, ylabel=ylabel, xlabel="Independent nuisance concepts, n",
           xticks=PADDINGS, xticklabels=[str(n) for n in PADDINGS], xlim=(-23, 279))
    ax.axhline(0, color="#777777", linewidth=.65, zorder=1)
    ax.grid(axis="y")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)


def interval(ax, x, gate, color):
    y = gate["mean"]
    ax.errorbar(x, y, yerr=[[y - gate["lower"]], [gate["upper"] - y]],
                fmt="none", ecolor=color, elinewidth=1.5, capsize=4, zorder=4)


def candidate_figure(points):
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.25))
    fig.subplots_adjust(left=.087, right=.985, top=.845, bottom=.32, wspace=.31)
    primary = [r for r in points["gap_points"] if r["scope"] == "K100_all"]
    xs = np.asarray(PADDINGS)
    axes[0].plot(xs, [r["raw"]["mean"] for r in primary], color=COLORS["raw"],
                 marker="o", markersize=5, linewidth=1.25, label="Raw API, K=100")
    for i, row in enumerate(primary):
        if row["displayed_interval"] is not None:
            interval(axes[0], row["padding"], row["displayed_interval"], COLORS["raw"])
    axes[0].axhline(primary[0]["factorized"]["mean"], color=COLORS["factorized"],
                    linewidth=1.3, linestyle="--", label="Factorized API, K=100")
    exact = points["numerical_exact_task_reference"]["value"]
    axes[0].axhline(exact, color=COLORS["exact"], linewidth=1., linestyle=":",
                    label="Exact-task KL (PMF reference)")
    panel_axes(axes[0], "(a) Main: 4096 observations per stream", r"Score gap: $F(t=0.75)-F(t=0)$")
    for scope, color, marker, label in (("K100_prefix1024", COLORS["raw"], "o", "Raw API, K=100"),
                                       ("K400_prefix1024", COLORS["aux"], "s", "Raw API, K=400")):
        rows = [r for r in points["gap_points"] if r["scope"] == scope]
        axes[1].plot(xs, [r["raw"]["mean"] for r in rows], color=color,
                     marker=marker, markersize=5, linewidth=1.25, label=label)
    panel_axes(axes[1], "(b) Auxiliary: paired first 1024", r"Score gap: $F(t=0.75)-F(t=0)$")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False, fontsize=7.7)
    fig.suptitle("An unchanged observation model, a changed finite-particle ranking", y=.985, fontsize=11)
    fig.text(.5, .018, "Negative gap favors the high-correlation candidate. Bars reuse the original four-gate t intervals; other markers are means.",
             ha="center", va="bottom", fontsize=7.5)
    return export(fig, "api_candidate_ranking")


def gradient_figure(points):
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.05))
    fig.subplots_adjust(left=.087, right=.985, top=.84, bottom=.25, wspace=.31)
    rows = points["gradient_points"]
    axes[0].plot(PADDINGS, [r["correlation_gradient_shift"]["mean"] for r in rows],
                 color=COLORS["raw"], marker="o", markersize=5, linewidth=1.25)
    interval(axes[0], 256, rows[2]["correlation_interval"], COLORS["raw"])
    panel_axes(axes[0], "(a) Correlation-direction gradient shift",
               r"$\partial_t F_n-\partial_t F_0$ at truth")
    axes[1].plot(PADDINGS[1:], [r["nuisance_scale_gradient_per_coordinate"]["mean"] for r in rows[1:]],
                 color=COLORS["raw"], marker="o", markersize=5, linewidth=1.25)
    interval(axes[1], 256, rows[2]["scale_interval"], COLORS["raw"])
    panel_axes(axes[1], "(b) Nuisance-scale gradient per concept",
               r"$n^{-1}\partial_a F_n$ at $a=1$")
    axes[1].text(0, .08, "NA", transform=axes[1].get_xaxis_transform(),
                 ha="center", va="bottom", color="#555555", fontsize=9)
    for ax in axes:
        ax.axhline(0, color=COLORS["factorized"], linestyle="--", linewidth=1.3)
    handles = [Line2D([0], [0], color=COLORS["raw"], marker="o", label="Raw official API"),
               Line2D([0], [0], color=COLORS["factorized"], linestyle="--", label="Factorized reference: zero")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .082), ncol=2, frameon=False)
    fig.suptitle("Feasible local directions at the true parameters", y=.985, fontsize=11)
    fig.text(.5, .018, "K=100; 4096 observations per stream. Bars reuse original intervals. The n=0 normalized scale derivative is undefined.",
             ha="center", va="bottom", fontsize=7.5)
    return export(fig, "api_gradient_directions")


def controls_supplement(data):
    # Preserve all numerical failures and original thresholds. Never infer the
    # source of an error from the difference between two compound expressions.
    gradient_tables = []
    for r in data["descriptive_gradient_tables"]:
        row = r.copy()
        if row["padding"] == 0:
            row["nuisance_scale_per_coordinate"] = None
            row["normalized_scale_display"] = "NA; production array retains its zero placeholder"
        gradient_tables.append(row)
    return dict(numerical64=data["numerical64"], scale_controls=data["scale_controls"],
                numerical_qualification=data["numerical_qualification"],
                scale_control_qualification=data["scale_control_qualification"],
                descriptive_gradient_tables=gradient_tables,
                probability_diagnostics=[dict(stream=r["stream"], **r["probability_diagnostics"]) for r in data["records"]],
                thresholds=dict(score=data["config"]["numerical_score_tolerance"],
                                gradient=data["config"]["numerical_gradient_tolerance"]),
                interpretation="Qualification supplement only, not a learned repair or a new success family. Stable64 differences include input/parameter precision and mathematical expression changes, not only BCE. Sigmoid counters count evaluations, not independent samples. Zero scale is an algebraic boundary, not an attained original softplus+floor parameter.")


def run():
    source_hash = sha(SOURCE)
    assert not (OUT / "FIGURES.json").exists(), "Figures already produced"
    data, reference, bindings = load_complete()
    points = collect_points(data, reference)
    style()
    point_path = OUT / "FIGURE_POINTS.json"
    save(point_path, points)
    control_path = OUT / "FIGURE_NUMERICAL_SUPPLEMENT.json"
    save(control_path, controls_supplement(data))
    outputs = candidate_figure(points)
    outputs.update(gradient_figure(points))
    assert sha(SOURCE) == source_hash
    for path, digest in bindings.items():
        assert sha(ROOT / path) == digest, path
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source=rel(SOURCE), source_sha256=source_hash, input_sha256=bindings,
        points={rel(point_path): sha(point_path)}, numerical_supplement={rel(control_path): sha(control_path)},
        figures=outputs, figure_count=2, new_samples=0, new_model_integrations=0,
        new_parameter_fits=0, new_significance_families=0,
        visual_QA="Pending actual viewing of both PNG files; this metadata does not assert visual approval",
        scope=points["scope"], interval_scope=points["interval_scope"],
        auxiliary_scope=points["auxiliary_scope"])
    save(OUT / "FIGURES.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["run"])
    parser.parse_args()
    run()
