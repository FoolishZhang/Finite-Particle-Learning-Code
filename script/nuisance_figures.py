"""Two MC figures from frozen nuisance-dimension results; no scientific runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT/"artifacts/nuisance_dimensions/MC"
AUDIT = ROOT/"artifacts/nuisance_dimensions/MC_AUDIT"
OUT = ROOT/"output/figures/nuisance_dimensions"
FIGURES = ("mc_gradients_and_gap", "finite_n_risk_contract")
COLORS = ["#303030", "#347fad", "#a76139", "#7a5c99", "#a84e6c", "#268774"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_file(path, text):
    text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()+"\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def write(path, value):
    text_file(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def verify(files):
    for name, digest in files.items():
        assert sha(ROOT/name) == digest, name


def load():
    assert read(MC/"STATE.json")["status"] == "complete"
    frozen, seal = read(MC/"FREEZE.json"), read(MC/"RESULTS_FREEZE.json")
    verify(frozen["files"])
    verify({k: v for k, v in seal["files"].items() if not k.endswith(".npz")})
    assert seal["source_freeze_sha256"] == sha(MC/"FREEZE.json")
    audit = read(AUDIT/"RECEIPT.json")
    assert audit["passed"]
    verify(audit["files"])
    verify(audit["audit_artifacts"])
    results = read(MC/"RESULTS.json")
    cfg = frozen["config"]
    assert cfg["K"] == 4 and cfg["theta"] == 0 and cfg["a"] == 1
    assert len(results["records"]) == 18 and len(results["finite_n_risk_certificates"]) == 6
    files = {**frozen["files"], **{k: v for k, v in seal["files"].items() if not k.endswith(".npz")},
             rel(MC/"RESULTS_FREEZE.json"): sha(MC/"RESULTS_FREEZE.json"),
             rel(AUDIT/"RECEIPT.json"): sha(AUDIT/"RECEIPT.json"),
             rel(Path(__file__).resolve()): sha(Path(__file__))}
    return results, cfg, files


def save_plot(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=200, facecolor="white")
    fig.savefig(OUT/(name+".svg"), facecolor="white", metadata={"Date": None})
    svg = OUT/(name+".svg")
    text_file(svg, svg.read_text(encoding="utf-8"))
    plt.close(fig)


def n_axis(ax, ns):
    xx = np.log1p(np.array(ns))
    ax.set_xticks(xx, labels=[str(n) for n in ns])
    ax.set_xlim(-.3, xx[-1]+.3)
    ax.set_xlabel("Nuisance coordinates n (log(1+n) spacing)")
    ax.grid(axis="y", alpha=.18)
    return xx


def plot_measurements(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    fig.subplots_adjust(left=.09, right=.975, top=.77, bottom=.28, wspace=.27)
    fig.suptitle("Independent padding reverses gradient and score-gap signs", y=.965, fontsize=14.2)
    for ax in axes:
        ax.axhline(0, lw=1, color="#555555")
        n_axis(ax, cfg["ns"])
    for rho, color, marker in ((.3, "#347fad", "o"), (.75, "#a76139", "s")):
        rows = [r for r in results["records"] if r["kind"] == "gradient" and r["rho"] == rho]
        rows.sort(key=lambda r: r["n"])
        x = np.log1p([r["n"] for r in rows])
        y = np.array([r["mean"] for r in rows])*1e3
        err = np.array([[r["mean"]-r["low"] for r in rows], [r["high"]-r["mean"] for r in rows]])*1e3
        axes[0].errorbar(x, y, yerr=err, fmt=marker+"-", color=color, capsize=4, ms=5,
                         lw=1.5, label=rf"$\rho={rho:g}$")
    rows = [r for r in results["records"] if r["kind"] == "gap_R_minus_0"]
    rows.sort(key=lambda r: r["n"])
    x = np.log1p([r["n"] for r in rows])
    y = np.array([r["mean"] for r in rows])*1e3
    err = np.array([[r["mean"]-r["low"] for r in rows], [r["high"]-r["mean"] for r in rows]])*1e3
    axes[1].errorbar(x, y, yerr=err, fmt="o-", color="#268774", capsize=4, ms=5, lw=1.5)
    axes[0].set_title("Population gradients at two prespecified correlations")
    axes[0].set_ylabel(r"$F'_n(\rho)$ ($10^{-3}$ nats per correlation unit)")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_title(r"Same fixed models: $F_n(0.75)-F_n(0)$")
    axes[1].set_ylabel(r"Population score difference ($10^{-3}$ total nats)")
    axes[1].text(.98, .94, "n=0: correlated point scores worse\nn=4094: correlated point scores better",
                 ha="right", va="top", transform=axes[1].transAxes, fontsize=9)
    fig.text(.5, .85, "K=4, a=1; independent teacher; 65,536 iid-bank sampling units; no labels or model fitting",
             ha="center", fontsize=10)
    fig.text(.5, .14, "Error bars: simultaneous 95% empirical Bernstein intervals for the original 18 quantities.\n"
             "All n prefixes reuse inputs; they are not independent replications. Lines only connect measured points.",
             ha="center", fontsize=9)
    fig.text(.5, .047, "These are gradients at prespecified correlations and one score contrast, not global-minimum locations.\n"
             "Sampling coverage assumes ideal iid inputs and exact statistics; floating-point and PRNG error are not certified.",
             ha="center", fontsize=9, color="#444444")
    save_plot(fig, FIGURES[0])


def derived_risk(results, cfg):
    a, A = cfg["a"], cfg["price_abs_bound"]
    kappa = a*a/(4*(1+a*a/2))
    coefficient = kappa*kappa/(2*A*A)
    eps = np.linspace(0, 1.05*max(r["eta"] for r in results["finite_n_risk_certificates"]), 201)
    curves = []
    for row in results["finite_n_risk_certificates"]:
        bound = coefficient*np.maximum(0, row["eta"]-eps)**2
        assert abs(bound[0]-row["exact_min_joint_KL_lower"]) < 1e-18
        curves.append({"n": row["n"], "eta": row["eta"], "epsilon": eps.tolist(),
                       "joint_KL_lower": bound.tolist()})
    return {"coefficient": coefficient, "A": A, "kappa_observation": kappa,
            "curves": curves, "extra_confidence_family": False,
            "curve_points_are_display_only_not_scientific_models": True}


def plot_contract(results, cfg, derived):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7))
    fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.285, wspace=.27)
    fig.suptitle("A finite-n risk consequence from a negative score-gap upper bound", y=.97, fontsize=14)
    cert = results["finite_n_risk_certificates"]
    ns = [r["n"] for r in cert]
    x = n_axis(axes[0], ns)
    y = np.array([r["exact_min_joint_KL_lower"] for r in cert])*1e7
    axes[0].plot(x, y, color="#999999", lw=1, zorder=1)
    axes[0].scatter(x, y, c=COLORS, s=35, zorder=2)
    axes[0].set_title(r"For any exact population minimum ($\epsilon=0$)")
    axes[0].set_ylabel(r"Joint-KL lower bound ($10^{-7}$ total nats)")
    axes[0].set_ylim(-.15, 5.1)
    axes[0].text(.02, .91, "n=0 gives only a zero lower bound", transform=axes[0].transAxes, fontsize=9)
    for i, curve in enumerate(derived["curves"]):
        if curve["n"] == 0:
            continue
        axes[1].plot(np.array(curve["epsilon"])*1e3, np.array(curve["joint_KL_lower"])*1e7,
                     lw=1.6, color=COLORS[i], label=f"n={curve['n']}")
    axes[1].set_title(r"For any $\epsilon$-near population minimum")
    axes[1].set_xlabel(r"Population objective allowance $\epsilon$ ($10^{-3}$ nats)")
    axes[1].set_ylabel(r"Joint-KL lower bound ($10^{-7}$ total nats)")
    axes[1].set_ylim(-.15, 5.1)
    axes[1].grid(alpha=.18)
    axes[1].legend(loc="upper right", fontsize=9, frameon=False)
    fig.text(.5, .85, r"$\eta=\max(0,-\mathrm{upper\ bound\ on}\ [F_n(0.75)-F_n(0)])$;  risk bound $=\frac{2}{9}(\eta-\epsilon)_+^2$",
             ha="center", fontsize=10)
    fig.text(.5, .14, "These are conservative lower bounds on integrated-model risk, not estimated risks or fitted endpoints.\n"
             "The same simultaneous sampling event supports all curves; no new confidence family is introduced.",
             ha="center", fontsize=9)
    fig.text(.5, .047, "No claim that these finite n values exceed the unknown asymptotic threshold.\n"
             "The asymptotic endpoint bound 1/128 is not substituted for these much smaller finite-sample consequences.",
             ha="center", fontsize=9, color="#444444")
    save_plot(fig, FIGURES[1])


def build():
    assert not (OUT/"FIGURE_RECEIPT.json").exists(), "recorded figure component is immutable"
    results, cfg, files = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "nuisance-dimensions-MC"})
    derived = derived_risk(results, cfg)
    plot_measurements(results, cfg)
    plot_contract(results, cfg, derived)
    write(OUT/"DERIVED_POINTS.json", {"records": results["records"],
          "risk_certificates": results["finite_n_risk_certificates"], "near_min_display": derived,
          "plot_n_axis": "log(1+n), ticks labelled by original n", "derivation_only": True,
          "new_scientific_samples": 0, "new_integrals": 0, "new_fits": 0})
    write(OUT/"SOURCE_HASHES.json", files)
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in ("BUILD.json", "FIGURE_RECEIPT.json")]
    write(OUT/"BUILD.json", {"created": datetime.now(timezone.utc).isoformat(),
          "files": {rel(p): sha(p) for p in outputs}, "source_sha256": sha(Path(__file__)),
          "figures": list(FIGURES), "raw_MC_quantities": 18, "risk_certificates": 6,
          "new_scientific_samples": 0, "new_integrals": 0, "new_fits": 0})
    print(json.dumps({"built": True, "package_sealed": False}))


def record(visual_qa_complete):
    assert visual_qa_complete and not (OUT/"FIGURE_RECEIPT.json").exists()
    build_info = read(OUT/"BUILD.json")
    verify(build_info["files"])
    verify(read(OUT/"SOURCE_HASHES.json"))
    assert sha(Path(__file__)) == build_info["source_sha256"]
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != "FIGURE_RECEIPT.json"]
    paths.append(Path(__file__).resolve())
    write(OUT/"FIGURE_RECEIPT.json", {"created": datetime.now(timezone.utc).isoformat(),
          "files": {rel(p): sha(p) for p in paths}, "sources": read(OUT/"SOURCE_HASHES.json"),
          "component_only": True, "research_package_sealed": False, "visual_qa_complete": True,
          "viewed_pngs": {name: sha(OUT/(name+".png")) for name in FIGURES},
          "new_scientific_computation": False})
    print(json.dumps({"recorded": True, "component_only": True, "receipt_sha256": sha(OUT/"FIGURE_RECEIPT.json")}))


def check():
    receipt = read(OUT/"FIGURE_RECEIPT.json")
    verify(receipt["files"])
    verify(receipt["sources"])
    assert receipt["visual_qa_complete"] and not receipt["research_package_sealed"]
    print(json.dumps({"passed": True, "component_only": True, "read_only": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "record", "check"))
    parser.add_argument("--visual-qa-complete", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "record":
        record(args.visual_qa_complete)
    else:
        check()
