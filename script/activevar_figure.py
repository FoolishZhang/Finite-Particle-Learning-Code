"""Draw one frozen-data scientific figure; no RNG, fitting, or integration.

The three-panel specification fixes mixed as the representative teacher before
results exist. Only the producer's JSON summaries and reference values are read.
PNG/SVG and their exact data mapping are written under the new package's docs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/active_scale_boundary"
DOC = ROOT / "docs/08_research_研究工作/有效尺度自由度与有限曲率_20261004"
FIG = DOC / "figures"
STEM = "active_scale_boundary"
FILES = [f"{STEM}.png", f"{STEM}.svg", "FIGURE_DATA.json"]
TEACHERS = ["independent", "mixed", "equicorrelation", "signed"]
LABELS = {"independent": "Independent (null)", "mixed": "Mixed",
          "equicorrelation": "Equicorrelation", "signed": "Signed"}
REPRESENTATIVE = "mixed"  # Prespecified before production results exist.
COLORS = {"n0": "#24679D", "n256": "#C84E3A", "risk": "#282B31",
          "mixed": "#DD9132", "equicorrelation": "#34826A", "signed": "#8059A5"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def save(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_inputs():
    inputs = {"summary": OUT / "SUMMARY/RESULTS.json",
              "reference": OUT / "REFERENCE/RESULTS.json",
              "spec": OUT / "SPEC.json", "freeze": OUT / "FREEZE.json"}
    payload = {name: read(path) for name, path in inputs.items()}
    result, reference, spec, freeze = (payload[k] for k in
                                        ("summary", "reference", "spec", "freeze"))
    if not freeze.get("science_authorized") or not reference["qualification_passed"]:
        raise ValueError("Scientific freeze/reference qualification is missing")
    if (result["freeze_sha256"] != sha(inputs["freeze"]) or
            reference["freeze_sha256"] != sha(inputs["freeze"]) or
            result["reference_sha256"] != sha(inputs["reference"])):
        raise ValueError("Frozen reference/summary chain does not match")
    if ([r["name"] for r in reference["records"]] != TEACHERS or
            spec["t_values"] != [0., .1, .2, .4] or spec["primary_t"] != .2 or
            spec["mc"]["nuisance_values"] != [0, 4, 16, 64, 256]):
        raise ValueError("Prespecified figure panel no longer matches the experiment")
    primary = result["raw_primary_family"]["results"]
    curve_rows = primary + result["descriptive_curves"]
    table = {(r["teacher"], float(r["t"]), r["nuisance"]): r for r in curve_rows}
    if len(curve_rows) != 80 or len(table) != 80:
        raise ValueError("The complete fixed 4 x 4 x 5 score panel is required")
    refs = {r["name"]: r for r in reference["records"]}
    collisions = result["collision_family"]["results"]
    panel_a = {"n": [r["nuisance"] for r in collisions],
               "collision": [{k: r[k] for k in ("nuisance", "mean", "lower", "upper", "exact")}
                             for r in collisions],
               "thresholds": [{"teacher": name,
                               "collision_threshold": refs[name]["derived"]["collision_threshold"]}
                              for name in TEACHERS[1:]],
               "source_pointer": "summary.collision_family.results; reference.records[].derived.collision_threshold",
               "interval": "Prespecified q=4, alpha=.025 simultaneous iid-bank Hoeffding intervals; n=0 exact."}
    panel_b = []
    for name in TEACHERS:
        records = [table[(name, .2, n)] for n in (0, 256)]
        ref = next(r for r in refs[name]["candidates"] if r["t"] == .2)
        for row in records:
            if row["reference_KL_contrast"] != ref["KL_minus_product"]:
                raise ValueError("Summary/reference risk mismatch")
        panel_b.append({"teacher": name, "t": .2,
                        "raw": [{k: row[k] for k in ("nuisance", "raw_score_contrast", "lower", "upper",
                                                       "SE_across_16_stream_means", "precision_passed",
                                                       "direction_passed")}
                                for row in records],
                        "reference_KL_contrast": ref["KL_minus_product"]})
    panel_c = {"teacher": REPRESENTATIVE, "nuisance": [0, 256], "rows": []}
    for tvalue in spec["t_values"]:
        ref = next(r for r in refs[REPRESENTATIVE]["candidates"] if r["t"] == tvalue)
        row = {"t": tvalue, "reference_KL_contrast": ref["KL_minus_product"], "raw": []}
        for n in (0, 256):
            score = table[(REPRESENTATIVE, tvalue, n)]
            item = {"nuisance": n, "raw_score_contrast": score["raw_score_contrast"]}
            if tvalue == .2:
                item.update(lower=score["lower"], upper=score["upper"])
            row["raw"].append(item)
        panel_c["rows"].append(row)
    mapping = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "figure_source": {"path": rel(SOURCE), "sha256": sha(SOURCE)},
               "input_files": {rel(path): sha(path) for path in inputs.values()},
               "panel_A": panel_a, "panel_B": panel_b, "panel_C": panel_c,
               "transforms": {"B_and_C": "Multiply all score/risk contrasts and CI endpoints by 1000 for display.",
                              "A": "Equally spaced prespecified n categories; connecting line is a guide, not fitted interpolation.",
                              "C": "Straight line segments through the four preset t values; no fitted model or extra evaluations."},
               "scope": {"new_RNG": 0, "new_integrals": 0, "new_fits": 0,
                         "representative_selected_before_results": REPRESENTATIVE,
                         "raw_intervals": "q=8, alpha=.025 simultaneous development t intervals over 16 independent prior streams; fixed numeric teacher distributions; not training/data uncertainty.",
                         "reference": "Quadrature-qualified integrated risk; no statistical intervals or certified integral remainder.",
                         "C_errorbars": "Only the prespecified t=.2 primary comparisons inherit their original intervals; remaining points are descriptive."}}
    return mapping


def archive_for_revision():
    existing = [FIG / name for name in FILES if (FIG / name).exists()]
    if not existing:
        return
    history = FIG / "build_history"
    history.mkdir(exist_ok=True)
    number = len([p for p in history.iterdir() if p.is_dir()]) + 1
    folder = history / f"revision_{number:02d}"
    folder.mkdir(exist_ok=False)
    records = {}
    for path in existing:
        # Copies only owned, explicit figure files; no recursive move/delete.
        shutil.copy2(path, folder / path.name)
        records[path.name] = sha(path)
    shutil.copy2(SOURCE, folder / "activevar_figure_at_revision.py")
    save(folder / "ARCHIVE.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                  "preserved_files": records,
                                  "note": "Layout-only regeneration from the same frozen science; former assets preserved."})


def draw(mapping):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "svg.fonttype": "none", "svg.hashsalt": "active-scale-boundary-fixed-figure",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 6.0),
                            gridspec_kw={"width_ratios": [1.02, 1.28, 1.]})
    fig.subplots_adjust(left=.047, right=.986, top=.81, bottom=.23, wspace=.36)
    fig.suptitle("Free active scales: local curvature and finite risk changes", x=.047,
                 y=.965, ha="left", fontsize=17, weight="bold")
    fig.text(.047, .906, "Fixed nuisance representation  |  K = 100  |  No training or new observed data",
             fontsize=11, color="#555B62", ha="left")
    for ax in axes:
        ax.grid(axis="y", color="#E5E8EB", linewidth=.7, zorder=0)
        ax.tick_params(colors="#43484D")
        ax.spines["left"].set_color("#93999F")
        ax.spines["bottom"].set_color("#93999F")

    # A: a single collision family, not a per-teacher training uncertainty band.
    ax = axes[0]
    ca = mapping["panel_A"]
    xpos = np.arange(len(ca["n"]))
    mean = np.array([r["mean"] for r in ca["collision"]])
    lower = np.array([r["lower"] for r in ca["collision"]])
    upper = np.array([r["upper"] for r in ca["collision"]])
    ax.errorbar(xpos, mean, yerr=np.vstack([mean - lower, upper - mean]),
                marker="o", markersize=5, capsize=3, linewidth=1.7, color=COLORS["n0"],
                label=r"Nuisance collision $\chi$", zorder=4)
    for item in ca["thresholds"]:
        name = item["teacher"]
        ax.axhline(item["collision_threshold"], color=COLORS[name], linestyle="--", linewidth=1.25,
                   label=f"{LABELS[name]} threshold", zorder=2)
    ax.set_xticks(xpos, [str(n) for n in ca["n"]])
    ax.set_xlim(-.15, 4.2)
    ax.set_ylim(-.015, min(1., max(.75, float(upper.max()) + .06)))
    ax.set_xlabel("Nuisance dimension n")
    ax.set_ylabel(r"Collision $\chi=E\sum_k\pi_k^2$")
    ax.set_title("A  Local stability thresholds", loc="left", pad=16, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=2.0,
              borderaxespad=.5, labelspacing=.55)

    # B: exact same primary comparisons; the null is retained and named.
    ax = axes[1]
    ax.grid(False)
    ax.axvline(0., color="#818890", linewidth=.9, zorder=0)
    ybase = np.arange(4)[::-1]
    offsets = [.19, -.19]
    ax.axhspan(2.62, 3.38, facecolor="#F1F3F5", zorder=-2)
    for base, item in zip(ybase, mapping["panel_B"]):
        for j, raw in enumerate(item["raw"]):
            mean, low, high = [raw[k] * 1000 for k in ("raw_score_contrast", "lower", "upper")]
            ax.errorbar(mean, base + offsets[j], xerr=[[mean - low], [high - mean]],
                        fmt="o" if j == 0 else "s", markersize=5, capsize=3,
                        linewidth=1.35, color=COLORS["n0" if j == 0 else "n256"], zorder=4)
        ax.plot(item["reference_KL_contrast"] * 1000, base, marker="D", markersize=5.3,
                color=COLORS["risk"], linestyle="none", zorder=5)
        ax.axhline(base - .5, color="#E8EBED", linewidth=.6, zorder=-1)
    ax.set_yticks(ybase, [LABELS[r["teacher"]] for r in mapping["panel_B"]])
    ax.set_ylim(-.5, 3.52)
    ax.set_xlabel(r"Change from product baseline ($\times10^{-3}$)")
    ax.set_title("B  Finite activation, t = 0.2", loc="left", pad=16, weight="bold")
    handles = [Line2D([], [], color=COLORS["n0"], marker="o", linestyle="none", label="Raw, n = 0"),
               Line2D([], [], color=COLORS["n256"], marker="s", linestyle="none", label="Raw, n = 256"),
               Line2D([], [], color=COLORS["risk"], marker="D", linestyle="none", label="Integrated risk")]
    ax.legend(handles=handles, frameon=False, fontsize=8.6, loc="lower center", bbox_to_anchor=(.5, -.20),
              ncol=3, columnspacing=.85, handletextpad=.35)

    # C: mixed was fixed in the plotting specification before scientific output.
    ax = axes[2]
    rows = mapping["panel_C"]["rows"]
    ts = np.array([r["t"] for r in rows])
    ax.axhline(0., color="#818890", linewidth=.8, zorder=0)
    for j, n in enumerate([0, 256]):
        vals = np.array([r["raw"][j]["raw_score_contrast"] for r in rows]) * 1000
        ax.plot(ts, vals, marker="o" if n == 0 else "s", markersize=4.5, linewidth=1.5,
                color=COLORS["n0" if n == 0 else "n256"], label=f"Raw, n = {n}")
        primary = next(r["raw"][j] for r in rows if r["t"] == .2)
        estimate, lo, hi = [primary[k] * 1000 for k in ("raw_score_contrast", "lower", "upper")]
        ax.errorbar([.2], [estimate], yerr=[[estimate - lo], [hi - estimate]], fmt="none", capsize=3,
                    color=COLORS["n0" if n == 0 else "n256"], linewidth=1.2)
    risk = np.array([r["reference_KL_contrast"] for r in rows]) * 1000
    ax.plot(ts, risk, color=COLORS["risk"], marker="D", markersize=4.3, linewidth=1.3,
            linestyle="--", label="Integrated risk")
    ax.set_xticks(ts, ["0", ".1", ".2", ".4"])
    ax.set_xlabel(r"Active scale amplitude t ($s=t\,a$)")
    ax.set_ylabel(r"Change from product ($\times10^{-3}$)")
    ax.set_title("C  Mixed teacher (prespecified)", loc="left", pad=16, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=2.0)

    fig.text(.047, .077,
             "A: 97.5% simultaneous Hoeffding bounds over iid prior banks; thresholds condition on numerical teacher moments.",
             fontsize=9, color="#555B62")
    fig.text(.047, .048,
             "B: 97.5% simultaneous development t intervals (16 prior streams); integrated risk uses qualified quadrature, not a confidence interval.",
             fontsize=9, color="#555B62")
    fig.text(.047, .019,
             "C: preset grid joined for description; only t = 0.2 carries its original interval. All intervals exclude observation/training variability and integration error.",
             fontsize=9, color="#555B62")
    fig.savefig(FIG / f"{STEM}.png", dpi=180)
    fig.savefig(FIG / f"{STEM}.svg", metadata={"Date": None,
                                                "Description": "Frozen-data active-scale boundary figure; no new scientific computation."})
    plt.close(fig)


def check():
    data = read(FIG / "FIGURE_DATA.json")
    if data["figure_source"]["sha256"] != sha(SOURCE):
        raise ValueError("Figure script changed")
    for name, digest in {**data["input_files"], **data["output_files"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Figure provenance changed: {name}")
    print(json.dumps({"passed": True, "figure": rel(FIG / f"{STEM}.png"),
                      "new_scientific_computation": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revise", action="store_true", help="Preserve preceding assets before layout regeneration")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if (OUT / "VERSION_LOCK.json").exists():
        raise RuntimeError("Package is sealed; no figure mutation")
    mapping = prepare_inputs()
    FIG.mkdir(parents=True, exist_ok=True)
    existing = any((FIG / name).exists() for name in FILES)
    if existing and not args.revise:
        raise FileExistsError("Owned figure assets exist; explicit --revise preserves them")
    if existing:
        archive_for_revision()
    draw(mapping)
    mapping["output_files"] = {rel(FIG / name): sha(FIG / name) for name in FILES[:2]}
    map_path = FIG / "FIGURE_DATA.json"
    if map_path.exists():
        # Its earlier bytes were explicitly copied above into build_history.
        with map_path.open("w", encoding="utf8", newline="\n") as handle:
            json.dump(mapping, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    else:
        save(map_path, mapping)
    check()


if __name__ == "__main__":
    main()
