"""Display frozen weak-emission results. No integration, RNG or optimization."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/weak_emission"
NUM = BASE/"NUMERIC"
OUT = BASE/"figures"
DOC = ROOT/"docs/08_research_研究工作/弱发射展开与总体偏好_20261001"
FIGURES = ("rho_location_vs_a", "observation_risk")
COLORS = {2: "#ba5c39", 3: "#347fad", 4: "#8b66a2", 8: "#248e78"}
MARKERS = {2: "s", 3: "o", 4: "^", 8: "D"}
GUIDE_A8_COEFFICIENT = 1e-5  # Fixed display scale, neither a fitted rate nor a bound.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_file(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()+"\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def write(path, data):
    text_file(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def verify(files):
    for name, digest in files.items():
        assert sha(ROOT/name) == digest, name


def load():
    assert read(NUM/"STATE.json")["state"] == "complete"
    result_seal = read(NUM/"RESULTS_FREEZE.json")
    for name, digest in result_seal["files"].items():
        assert sha(NUM/name) == digest, name
    freeze = read(NUM/"FREEZE.json")
    paths = [Path(__file__).resolve(), NUM/"RESULTS_FREEZE.json", NUM/"UNIT_CHECK.json"]
    paths += [NUM/name for name in result_seal["files"]]
    for entry in freeze["inputs"]:
        path = ROOT/entry["snapshot"]
        assert sha(path) == entry["sha256"]
        paths.append(path)
    theory = BASE/"T/RECEIPT.json"
    verify(read(theory)["files"])
    paths += [theory]+[ROOT/name for name in read(theory)["files"]]
    results = read(NUM/"RESULTS.json")
    corrections = read(NUM/"CORRECTION.json")["rows"]
    assert results["numerical_qualified"]
    assert results["prediction_tests_passed"] == results["prediction_tests_total"] == 8
    assert len(results["primary_locations"]) == 20 and len(corrections) == 15
    assert all(x["feasible"] and x["descriptive_only_no_prediction_gate"] for x in corrections)
    audit_path = BASE/"INDEPENDENT/RECEIPT.json"
    audit_passed = False
    if audit_path.exists():
        audit = read(audit_path)
        audit_passed = bool(audit["passed"] and not audit["failures"])
        assert audit_passed, "Retained audit failure: do not finalize plots or statements"
        start = BASE/"INDEPENDENT/START.json"
        assert sha(start) == audit["start_sha256"]
        verify(audit["files"])
        paths += [audit_path, start]+[ROOT/name for name in audit["files"]]
    return {"results": results, "locations": results["primary_locations"], "corrections": corrections,
            "config": read(NUM/"CONFIG.json"), "audit_passed": audit_passed,
            "sources": {rel(p): sha(p) for p in sorted(set(paths))}}


def save_figure(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=200, facecolor="white")
    fig.savefig(OUT/(name+".svg"), facecolor="white", metadata={"Date": None})
    path = OUT/(name+".svg")
    text_file(path, path.read_text(encoding="utf-8"))
    plt.close(fig)


def style_axis(ax, emissions):
    ax.set_xscale("log")
    ax.set_xticks(emissions, labels=[f"{x:g}" for x in emissions])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(emissions[0]*.80, emissions[-1]*1.2)
    ax.set_xlabel("Emission scale a (log scale)")
    ax.grid(alpha=.17)


def location_figure(data):
    emissions = data["config"]["emissions"]
    theta = data["config"]["theta"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7))
    fig.subplots_adjust(left=.075, right=.98, bottom=.245, top=.77, wspace=.29)
    fig.suptitle("Finite-K covariance inflation in a correctly specified d=2 model", y=.975, fontsize=14.5)
    for k in (2, 3, 4, 8):
        rows = sorted([r for r in data["locations"] if r["K"] == k], key=lambda r: r["a"])
        axes[0].plot([r["a"] for r in rows], [r["rho"] for r in rows],
                     marker=MARKERS[k], color=COLORS[k], label=f"K={k}", lw=1.8, ms=6)
        target = .75 if k == 2 else k*theta/(k-2)
        axes[0].axhline(target, color=COLORS[k], ls=":", lw=1, alpha=.75)
    for k in (3, 4, 8):
        rows = sorted([r for r in data["corrections"] if r["K"] == k], key=lambda r: r["a"])
        axes[1].plot([r["a"] for r in rows], [r["rho_corrected"] for r in rows],
                     marker=MARKERS[k], color=COLORS[k], lw=1.8, ms=6)
    for ax in axes:
        ax.axhline(theta, color="#333333", ls="--", lw=1.3)
        style_axis(ax, emissions)
    axes[0].set_ylim(.115, .79)
    axes[0].set_yticks([.15, .2, .3, .45, .6, .75])
    axes[0].set_ylabel(r"Latent correlation $\rho$ at raw numerical locator")
    axes[0].set_title("Raw objective: 20 fixed numerical locations")
    axes[0].text(.025, .85, "K=2: upper endpoint in all five numerical runs",
                 transform=axes[0].transAxes, fontsize=8.6, va="top")
    axes[1].set_ylim(.141, .1565)
    axes[1].set_ylabel(r"Corrected latent $\rho$ (zoomed scale)")
    axes[1].set_title(r"Fixed shrinkage: $\rho_{corr}=(K-2)\rho/K$")
    axes[1].text(.025, .035, "K>2 only; no new localization or tuning",
                 transform=axes[1].transAxes, fontsize=9)
    legend = [Line2D([0], [0], color=COLORS[k], marker=MARKERS[k], label=f"K={k}") for k in (2, 3, 4, 8)]
    legend += [Line2D([0], [0], color="#333333", ls="--", label=r"Teacher $\theta=0.15$")]
    fig.legend(handles=legend, ncol=5, loc="upper center", bbox_to_anchor=(.51, .905), frameon=False)
    fig.text(.5, .125, "Dotted levels: weak-emission predictions 0.75, 0.45, 0.30, 0.20; not finite-a equalities.\n"
             "K=1 is exactly flat over the whole correlation interval and has no unique locator.", ha="center", fontsize=9)
    fig.text(.5, .038, "Three predeclared quadrature variants passed; finite quadrature and root search are not strict global certificates.\n"
             "Shrinkage is conditional on unclipped inflation and is not a universal repair.", ha="center", fontsize=9, color="#444444")
    save_figure(fig, FIGURES[0])


def risk_figure(data):
    emissions = data["config"]["emissions"]
    guide_a = np.geomspace(emissions[0], emissions[-1], 201)
    prediction = {r["K"]: r for r in data["results"]["prediction_gates"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))
    fig.subplots_adjust(left=.09, right=.98, bottom=.26, top=.78, wspace=.26)
    fig.suptitle("Observed joint risk: raw locations and conditional correction", y=.98, fontsize=14.5)
    for k in (2, 3, 4, 8):
        rows = sorted([r for r in data["locations"] if r["K"] == k], key=lambda r: r["a"])
        axes[0].plot([r["a"] for r in rows], [r["joint_KL"] for r in rows],
                     marker=MARKERS[k], color=COLORS[k], lw=1.8, ms=6)
        axes[0].plot(guide_a, prediction[k]["leading_KL_over_a4"]*guide_a**4,
                     ls=":", lw=1.1, color=COLORS[k], alpha=.80)
    for k in (3, 4, 8):
        rows = sorted([r for r in data["corrections"] if r["K"] == k], key=lambda r: r["a"])
        axes[1].plot([r["a"] for r in rows], [r["primary_joint_KL"] for r in rows],
                     marker=MARKERS[k], color=COLORS[k], lw=1.8, ms=6)
    axes[1].plot(guide_a, GUIDE_A8_COEFFICIENT*guide_a**8, color="#666666", ls="--", lw=1.3,
                 label=r"$10^{-5}a^8$: slope guide only")
    for ax in axes:
        style_axis(ax, emissions)
        ax.set_yscale("log")
        ax.set_ylabel("Observed joint KL (total nats; log scale)")
    axes[0].set_title(r"Raw: $a^4$ predictions with analytic coefficients")
    axes[0].set_ylim(1e-10, 2e-2)
    axes[1].set_title(r"Corrected: theory gives an $O(a^8)$ upper rate")
    axes[1].set_ylim(1e-22, 1e-4)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    legend = [Line2D([0], [0], color=COLORS[k], marker=MARKERS[k], label=f"K={k}") for k in (2, 3, 4, 8)]
    fig.legend(handles=legend, ncol=4, loc="upper center", bbox_to_anchor=(.51, .91), frameon=False)
    fig.text(.5, .14, "All 20 raw and 15 corrected outputs are shown. The true model has KL=0 (not plotted on log axes).\n"
             "K=2 is not corrected. K=1 has no unique selected parameter.", ha="center", fontsize=9)
    fig.text(.5, .048, "The a⁸ guide has arbitrary vertical scale; it is not fitted and is not an error certificate.\n"
             "K=4 decreases faster here; no sharp a⁸ or a¹² rate is claimed. Numerical qualification is not a strict integral bound.",
             ha="center", fontsize=9, color="#444444")
    save_figure(fig, FIGURES[1])


def export_table(name, rows):
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build():
    assert not (OUT/"MANIFEST.json").exists(), "sealed output cannot be overwritten"
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "weak-emission-scientific-figures"})
    location_figure(data)
    risk_figure(data)
    export_table("RAW_LOCATIONS.csv", [{k: r[k] for k in
                 ("a", "K", "rho", "score_minus_rho0", "joint_KL", "selected_kind", "global_minimum_certificate")}
                 for r in data["locations"]])
    export_table("CORRECTED_LOCATIONS.csv", [{k: r[k] for k in
                 ("a", "K", "rho_raw", "rho_corrected", "primary_joint_KL", "primary_joint_KL_over_a8",
                  "corrected_over_raw_KL", "descriptive_only_no_prediction_gate")}
                 for r in data["corrections"]])
    write(OUT/"SOURCE_HASHES.json", data["sources"])
    files = [p for p in OUT.iterdir() if p.is_file() and p.name not in ("BUILD_RECEIPT.json", "VISUAL_QA.json", "MANIFEST.json")]
    for name in ("MAIN_RESULT_EN.md", "MAIN_RESULT_ZH.md"):
        if (DOC/name).exists():
            files.append(DOC/name)
    write(OUT/"BUILD_RECEIPT.json", {"created": datetime.now(timezone.utc).isoformat(),
          "files": {rel(p): sha(p) for p in files}, "source_sha256": sha(Path(__file__)),
          "raw_points": 20, "corrected_points": 15, "a8_guide_coefficient": GUIDE_A8_COEFFICIENT,
          "a8_guide_is_fit_or_certificate": False, "independent_audit_passed": data["audit_passed"],
          "new_integrals": 0, "new_random_draws": 0, "new_optimizations": 0})
    print(json.dumps({"built": True, "audit_passed": data["audit_passed"], "sealed": False}))


def seal(visual_qa_complete):
    assert visual_qa_complete and not (OUT/"MANIFEST.json").exists()
    current = load()
    assert current["audit_passed"], "wait for the independent numerical audit"
    assert current["sources"] == read(OUT/"SOURCE_HASHES.json"), "rebuild after final audit/source changes"
    build_receipt = read(OUT/"BUILD_RECEIPT.json")
    assert build_receipt["independent_audit_passed"]
    verify(build_receipt["files"])
    for name in ("MAIN_RESULT_EN.md", "MAIN_RESULT_ZH.md"):
        assert rel(DOC/name) in build_receipt["files"], "include final statements before seal"
    write(OUT/"VISUAL_QA.json", {"created": datetime.now(timezone.utc).isoformat(), "complete": True,
          "viewed_pngs": {name: sha(OUT/(name+".png")) for name in FIGURES},
          "checks": ["latent correlation and observed KL units", "all fixed points retained",
                     "no sharp corrected-rate claim", "no clipping/universal-repair claim", "labels and legends clear"]})
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "MANIFEST.json"]
    files += [DOC/"MAIN_RESULT_EN.md", DOC/"MAIN_RESULT_ZH.md", Path(__file__).resolve()]
    write(OUT/"MANIFEST.json", {"created": datetime.now(timezone.utc).isoformat(),
          "files": {rel(p): sha(p) for p in files}, "sources": current["sources"],
          "visual_qa_complete": True, "independent_numerical_audit_passed": True,
          "new_scientific_computation": False})
    print(json.dumps({"sealed": True, "manifest_sha256": sha(OUT/"MANIFEST.json")}))


def check():
    manifest = read(OUT/"MANIFEST.json")
    verify(manifest["files"])
    verify(manifest["sources"])
    assert manifest["visual_qa_complete"] and manifest["independent_numerical_audit_passed"]
    assert not manifest["new_scientific_computation"]
    print(json.dumps({"passed": True, "read_only": True, "scientific_computation_reexecuted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "seal", "check"))
    parser.add_argument("--visual-qa-complete", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "seal":
        seal(args.visual_qa_complete)
    else:
        check()
