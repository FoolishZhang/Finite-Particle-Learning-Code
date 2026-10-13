"""Final, fixed-field readout of the complete and independently audited panel.

This script never samples, scores banks, estimates intervals from arrays, or
chooses a scientific claim. --check rebuilds scalar/readable representations
and checks existing plot hashes without rendering plots again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/asymptotic_bridge_v1_20261008"
N_VALUES = (64, 256, 1024, 4096)
METRICS = ("path_gap", "D12", "overlap", "coupling", "direct_gap_truncated")
OUTPUT_NAMES = ("READABLE_TABLE.md", "AUTHOR_READOUT.json", "bridge_summary.png", "bridge_summary.svg")
PUBLIC_INPUTS = ("FREEZE.json", "REFERENCE_CONTRACT.json", "REFERENCE.json", "RUN_START.json",
                 "RUN_COMPLETE.json", "INDEX.json", "STATISTICS.json", "AUDIT.json")
SCOPE = ("99% conditional joint intervals for 26 fixed targets; ideal iid complete banks and "
         "the stated numerical qualification; frozen numeric references. "
         "131072 same banks across all n, not independent replicates. "
         "Native floating-point error and exact coefficient enclosures are not certified.")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text("utf-8"))


def close(a: float, b: float, label: str, tolerance: float = 2e-12) -> None:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise ValueError("nonfinite scalar: " + label)
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("scalar reconstruction mismatch: " + label)


def summary_fields(value: dict) -> dict:
    """Copy the same fixed scalar fields for every n and target."""
    fields = ("mean", "interval", "halfwidth", "SE", "unbiased_variance", "statistical_halfwidth",
              "numerical_allowance", "tail_allowance", "support", "banks", "delta", "delta_per_side")
    result = {name: value[name] for name in fields}
    for name in ("target_id", "reuses_target", "affine_scale", "affine_shift", "extra_alpha"):
        if name in value:
            result[name] = value[name]
    return result


def check_raw_summary(value: dict, label: str) -> None:
    """Reconstruct declared EB scalar algebra from saved mean/variance/support."""
    if value["banks"] != 131072 or value["unbiased_variance"] < 0:
        raise ValueError("invalid summary bank count/variance: " + label)
    delta = .01 / 26
    close(value["delta"], delta, label + ":delta")
    close(value["delta_per_side"], delta / 2, label + ":side")
    width = value["support"][1] - value["support"][0]
    if width < 0:
        raise ValueError("reversed support: " + label)
    statistical = math.sqrt(2 * value["unbiased_variance"] * math.log(4 / delta) / value["banks"])
    statistical += 7 * width * math.log(4 / delta) / (3 * (value["banks"] - 1))
    half = statistical + value["numerical_allowance"] + value["tail_allowance"]
    for actual, expected, suffix in (
        (value["statistical_halfwidth"], statistical, ":statistical"),
        (value["halfwidth"], half, ":halfwidth"),
        (value["SE"], math.sqrt(value["unbiased_variance"] / value["banks"]), ":SE"),
        (value["interval"][0], value["mean"] - half, ":low"),
        (value["interval"][1], value["mean"] + half, ":high"),
    ):
        close(actual, expected, label + suffix)


def check_affine(raw: dict, derived: dict, scale: float, shift: float, label: str) -> None:
    close(derived["mean"], scale * raw["mean"] + shift, label + ":mean")
    expected = sorted(scale * endpoint + shift for endpoint in raw["interval"])
    for index in (0, 1):
        close(derived["interval"][index], expected[index], label + f":endpoint{index}")
    close(derived["halfwidth"], abs(scale) * raw["halfwidth"], label + ":halfwidth")
    close(derived["unbiased_variance"], scale * scale * raw["unbiased_variance"], label + ":variance")
    close(derived["affine_scale"], scale, label + ":scale")
    close(derived["affine_shift"], shift, label + ":shift")
    if derived["extra_alpha"] != 0:
        raise ValueError("derived quantity allocated additional alpha: " + label)


def load_final_inputs() -> tuple[dict, dict, dict, dict]:
    """Require full completion and passed final audit before reading statistics."""
    for name in PUBLIC_INPUTS:
        if not (OUT / name).is_file():
            raise FileNotFoundError("complete final inputs required; missing " + name)
    audit = read(OUT / "AUDIT.json")
    if audit.get("passed") is not True or audit.get("statistics", {}).get("passed") is not True:
        raise ValueError("final independent arithmetic/statistical audit must pass")
    if audit.get("schema_version") != "asymptotic_bridge_independent_audit_v1":
        raise ValueError("unexpected audit schema")
    complete, index, frozen = (read(OUT / name) for name in ("RUN_COMPLETE.json", "INDEX.json", "FREEZE.json"))
    freeze_hash = sha(OUT / "FREEZE.json")
    reference_hash = sha(OUT / "REFERENCE.json")
    for document in (complete, index):
        if document["freeze_sha256"] != freeze_hash or document["reference_sha256"] != reference_hash:
            raise ValueError("completion provenance mismatch")
    if complete["index_sha256"] != sha(OUT / "INDEX.json"):
        raise ValueError("completed index binding mismatch")
    if index["resources"]["independent_banks"] != 131072 or len(index["records"]) != 2048:
        raise ValueError("only the complete fixed panel may be reported")
    if index["metric_names"] != list(METRICS) or index["n_values"] != list(N_VALUES):
        raise ValueError("panel metric/n schema mismatch")
    if audit["source_hashes"] != frozen["sources"]:
        raise ValueError("audit/freeze source binding mismatch")
    for relative, expected in frozen["sources"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
            raise ValueError("changed frozen source: " + relative)
    for name in PUBLIC_INPUTS:
        if name == "AUDIT.json":
            continue
        relative = (OUT / name).relative_to(ROOT).as_posix()
        if audit["input_hashes"].get(relative) != sha(OUT / name):
            raise ValueError("changed audited public input: " + name)
    # No primitive NPZ is read. This hash verifies the already-audited aggregate.
    aggregate = OUT / "BANK_SCORES.npy"
    aggregate_hash = sha(aggregate)
    if aggregate_hash != index["scores"]["sha256"]:
        raise ValueError("changed aggregate after audit")
    if audit["input_hashes"].get(aggregate.relative_to(ROOT).as_posix()) != aggregate_hash:
        raise ValueError("aggregate/audit binding mismatch")
    stats, reference = read(OUT / "STATISTICS.json"), read(OUT / "REFERENCE.json")
    if stats.get("banks") != 131072 or stats.get("formal_bank_count_matches") is not True:
        raise ValueError("formal complete statistics required")
    if stats.get("family_target_count") != 26 or stats.get("n") != list(N_VALUES):
        raise ValueError("fixed final family schema mismatch")
    if stats.get("metric_order") != list(METRICS) or not stats.get("shared_bank_pairing_preserved"):
        raise ValueError("metric order or bank pairing mismatch")
    if stats["config"]["alpha"] != .01 or stats["config"]["target_count"] != 26:
        raise ValueError("joint interval family changed")
    if stats["raw_sha256"] != aggregate_hash or stats["reference_sha256"] != reference_hash:
        raise ValueError("statistics input binding mismatch")
    if stats["statistics_script_sha256"] != sha(ROOT / "script/asymptotic_bridge_statistics.py"):
        raise ValueError("statistics producer source mismatch")
    if reference.get("qualification", {}).get("passed") is not True:
        raise ValueError("frozen reference qualification must pass")
    hashes = {(OUT / name).relative_to(ROOT).as_posix(): sha(OUT / name) for name in PUBLIC_INPUTS}
    hashes[aggregate.relative_to(ROOT).as_posix()] = aggregate_hash
    return stats, reference, audit, hashes


def extract(stats: dict, reference: dict, audit: dict, input_hashes: dict) -> dict:
    """All four rows and six contrasts, in contract order; no outcome selection."""
    h, gapv = float(reference["m_H45"]), float(reference["gapV"])
    if min(h, gapv) <= 0:
        raise ValueError("positive frozen reference coefficients required")
    rows = []
    if [row["n"] for row in stats["per_n"]] != list(N_VALUES):
        raise ValueError("all n rows required in fixed order")
    aliases = dict(g="path_gap", D12="D12", overlap="overlap", coupling="coupling", secondary_direct_gap="direct_gap_truncated")
    for row in stats["per_n"]:
        n, w = row["n"], float(reference["w_by_n"][str(row["n"])])
        close(row["w"], w, f"n{n}:w")
        close(w, reference["J100"] / math.sqrt(n * reference["v"]), f"n{n}:reference w")
        for metric in METRICS:
            check_raw_summary(row["raw"][metric], f"n{n}:{metric}")
        recipes = dict(A=("path_gap", 1/(w*gapv), 0.), B=("D12", -1/(w*h), 0.),
                       omega_over_w=("overlap", 1/w, 0.), R=("D12", 1/w, h),
                       O=("overlap", -h/w, h), J=("coupling", -1/w, 0.))
        for name, (raw, scale, shift) in recipes.items():
            check_affine(row["raw"][raw], row["derived"][name], scale, shift, f"n{n}:{name}")
        close(row["derived"]["R"]["mean"], row["derived"]["O"]["mean"]+row["derived"]["J"]["mean"], f"n{n}:R=O+J")
        compact = dict(n=n, w=w, leading_gap=w*gapv, leading_D12=-w*h)
        compact.update({name:summary_fields(row["raw"][source]) for name,source in aliases.items()})
        compact.update({("Bgrad" if name=="B" else name):summary_fields(row["derived"][name]) for name in recipes})
        compact["residual_covariance_order"] = row["residual_covariance_order"]
        compact["residual_covariance"] = row["residual_covariance"]
        compact["direct_minus_path_diagnostic"] = row["direct_minus_path_diagnostic"]
        compact["bank_identity_max_absolute_residual"] = row["bank_identity_max_absolute_residual"]
        rows.append(compact)
    paired = []
    for kind, family, derived in (("A", "paired_gap", "A"), ("Bgrad", "paired_gradient", "B")):
        if len(stats[family]) != 3:
            raise ValueError("all six predeclared contrasts required")
        for index, contrast in enumerate(stats[family]):
            previous, following = N_VALUES[index:index+2]
            if contrast["n_previous"] != previous or contrast["n_next"] != following or contrast["direction"] != "next_minus_previous":
                raise ValueError("paired contrast order changed")
            summary = contrast["summary"]
            check_raw_summary(summary, f"{kind}:{following}-{previous}")
            close(summary["mean"], rows[index+1]["Bgrad" if derived=="B" else derived]["mean"]-
                  rows[index]["Bgrad" if derived=="B" else derived]["mean"], f"{kind}:paired mean")
            expected_variance = contrast["independent_variance_not_used"]-2*contrast["next_scale"]*contrast["previous_scale"]*contrast["raw_same_bank_covariance"]
            close(summary["unbiased_variance"], expected_variance, f"{kind}:paired covariance variance")
            paired.append(dict(kind=kind,n_previous=previous,n_next=following,direction="next_minus_previous",
                               summary=summary_fields(summary),raw_same_bank_covariance=contrast["raw_same_bank_covariance"],
                               independent_variance_not_used=contrast["independent_variance_not_used"],
                               covariance_reconstructed_variance=contrast["covariance_reconstructed_variance"],
                               covariance_reconstruction_absolute_error=contrast["covariance_reconstruction_absolute_error"]))
    return dict(schema_version="asymptotic_bridge_author_readout_v1",source_sha256=sha(Path(__file__)),
                input_hashes=input_hashes,banks=131072,n=list(N_VALUES),K=100,
                confidence=dict(coverage=.99,target_count=26,alpha=.01,conditional=True,scope=SCOPE,
                                numerical_qualification_is_not_strict_certificate=True),
                frozen_references={name:reference[name] for name in ("v","J100","gapV","m_H45","w_by_n")},
                definitions=dict(A="g/(w*gapV)",Bgrad="-D12/(m_H45*w)",omega_over_w="overlap/w",
                                 R="D12/w+m_H45",O="m_H45*(1-overlap/w)",J="-coupling/w",identity="R=O+J"),
                per_n=rows,paired_contrasts=paired,
                covariance20_order=stats["covariance20_order"],covariance20=stats["covariance20"],
                covariance26_order=stats["covariance26_order"],covariance26=stats["covariance26"],
                covariance_note="All covariance entries use the same complete banks and ddof=1. Paired contrast variance uses covariance; the independence-only variance is diagnostic and not used for intervals.",
                audit_passed=audit["passed"],audit_scope=audit["scope"],
                author_claims_selected=False,scientific_samples_generated=0,scientific_score_calls=0,new_integrals=0)


def fmt(value: float) -> str:
    return format(float(value), ".9g")


def cell(summary: dict) -> str:
    return f"{fmt(summary['mean'])} [{fmt(summary['interval'][0])}, {fmt(summary['interval'][1])}]"


def readable(readout: dict) -> str:
    rows=readout["per_n"]
    text=["# AsymptoticBridge-v1: fixed full-panel readout", "", SCOPE, "",
          "Every cell below is mean [joint-family interval]. All four n values and all six predeclared paired contrasts are retained. Numerical qualification is not a strict enclosure of native arithmetic or exact reference coefficients.", "",
          "Primary gap and single-coordinate gradient", "",
          "| n | g = F(I)-F(Cb) | A = g/(w gapV) | D12 at rho=.45 | Bgrad = -D12/(m_H w) |",
          "|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['n']} | {cell(row['g'])} | {cell(row['A'])} | {cell(row['D12'])} | {cell(row['Bgrad'])} |")
    text += ["", "Residual decomposition on one fixed absolute scale", "",
             "| n | overlap/w | R = D12/w+m_H | O = m_H(1-overlap/w) | J = -coupling/w |",
             "|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['n']} | {cell(row['omega_over_w'])} | {cell(row['R'])} | {cell(row['O'])} | {cell(row['J'])} |")
    text += ["", "R=O+J holds bankwise up to the declared numerical qualification. Individual R/O/J intervals reuse their raw target events; their endpoints must not be summed as if independent.", "",
             "Secondary direct loss gap and fixed implementation diagnostics", "",
             "| n | direct gap, T8 truncation and tail allowance | direct minus path mean | paired diagnostic SE | max bank identity residual |",
             "|---:|---:|---:|---:|---:|"]
    for row in rows:
        diagnostic=row["direct_minus_path_diagnostic"]
        text.append(f"| {row['n']} | {cell(row['secondary_direct_gap'])} | {fmt(diagnostic['mean'])} | {fmt(diagnostic['paired_SE'])} | {fmt(row['bank_identity_max_absolute_residual'])} |")
    text += ["", "The direct-minus-path mean and SE are diagnostics, with no additional confidence claim or outcome-based pass threshold. Only the direct gap is clipped when any active |Z|>8; primary g and D12 are not clipped.", "",
             "All six adjacent paired contrasts: next n minus previous n", "",
             "| quantity | n transition | difference [joint-family interval] | raw same-bank covariance | paired sample variance | independence-only variance, unused |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in readout["paired_contrasts"]:
        text.append(f"| {row['kind']} | {row['n_previous']} to {row['n_next']} | {cell(row['summary'])} | {fmt(row['raw_same_bank_covariance'])} | {fmt(row['summary']['unbiased_variance'])} | {fmt(row['independent_variance_not_used'])} |")
    ref=readout["frozen_references"]
    text += ["",readout["covariance_note"], "The complete 20-target and 26-target covariance matrices, plus the R/O/J covariance matrices, are saved in AUTHOR_READOUT.json.", "",
             f"Frozen numeric references: v={fmt(ref['v'])}, J100={fmt(ref['J100'])}, gapV={fmt(ref['gapV'])}, m_H45={fmt(ref['m_H45'])}.", "",
             "| n | w_n | leading gap w_n gapV | leading single-coordinate gradient -m_H45 w_n |",
             "|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['n']} | {fmt(row['w'])} | {fmt(row['leading_gap'])} | {fmt(row['leading_D12'])} |")
    text += ["", "This readout extracts fixed fields only and selects no author claim. Signs, intervals spanning reference levels, and nonmonotone sequences remain in the tables and figure.", "",
             "![Fixed full-panel ratios and residuals](bridge_summary.png)", "",
             "Input hashes and producer source hash are recorded in AUTHOR_READOUT.json.", ""]
    return "\n".join(text)


def render(readout: dict, png: Path, svg: Path) -> None:
    """Two static scientific panels, all rows and intervals on shared axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.hashsalt":"asymptotic_bridge_v1"})
    figure,axes=plt.subplots(1,2,figsize=(11.8,5.3))
    rows=readout["per_n"]
    def series(axis,field,label,color,marker,multiplier):
        x=[row["n"]*multiplier for row in rows]
        mean=[row[field]["mean"] for row in rows]
        low=[row[field]["mean"]-row[field]["interval"][0] for row in rows]
        high=[row[field]["interval"][1]-row[field]["mean"] for row in rows]
        axis.errorbar(x,mean,yerr=[low,high],label=label,color=color,marker=marker,
                      linewidth=1.3,markersize=5,capsize=3,elinewidth=1.1)
    series(axes[0],"A","A: gap / leading gap","#2166ac","o",.94)
    series(axes[0],"Bgrad","Bgrad: D12 / leading D12","#b2182b","s",1.06)
    axes[0].axhline(1,color="#555555",linestyle="--",linewidth=1)
    axes[0].set_title("Finite-to-leading-term ratios")
    axes[0].set_ylabel("Ratio to frozen numeric leading term")
    for name,label,color,marker,mult in (("R","R","#222222","o",.90),("O","O","#1b9e77","s",1.),("J","J","#7570b3","^",1.10)):
        series(axes[1],name,label,color,marker,mult)
    axes[1].axhline(0,color="#555555",linestyle="--",linewidth=1)
    axes[1].set_title("R = O + J: fixed residual scale")
    axes[1].set_ylabel("Residual (one common absolute scale)")
    for axis in axes:
        axis.set_xscale("log",base=2)
        axis.set_xticks(N_VALUES,labels=[str(n) for n in N_VALUES])
        axis.set_xlim(48,5500)
        axis.set_xlabel("Nuisance dimension n (log2 axis)")
        axis.grid(axis="y",alpha=.25,linewidth=.6)
        axis.legend(fontsize=9,loc="best",frameon=False)
        axis.margins(y=.12)
    figure.suptitle("AsymptoticBridge-v1 | K100 | all four frozen n values",fontsize=13,y=.975)
    figure.text(.5,.052,"99% conditional joint intervals; frozen numeric references; numerical qualification applies.",ha="center",fontsize=9)
    figure.text(.5,.023,"B = 131072 same complete banks across n; arms and coordinates are not independent replicates.",ha="center",fontsize=9)
    figure.subplots_adjust(left=.075,right=.975,bottom=.18,top=.85,wspace=.30)
    figure.savefig(png,dpi=200,metadata={"Software":"asymptotic_bridge_report.py"})
    figure.savefig(svg,metadata={"Creator":"asymptotic_bridge_report.py","Date":None})
    plt.close(figure)


def run() -> dict:
    if any((OUT/name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("refusing to replace existing final report artifacts")
    stats,reference,audit,hashes=load_final_inputs()
    extracted=extract(stats,reference,audit,hashes)
    table=readable(extracted)
    with tempfile.TemporaryDirectory(prefix="bridge_report_",dir=OUT) as directory:
        staged=Path(directory)
        (staged/"READABLE_TABLE.md").write_text(table,encoding="utf-8",newline="\n")
        render(extracted,staged/"bridge_summary.png",staged/"bridge_summary.svg")
        extracted["artifacts"]={name:dict(sha256=sha(staged/name),bytes=(staged/name).stat().st_size)
                                for name in ("READABLE_TABLE.md","bridge_summary.png","bridge_summary.svg")}
        (staged/"AUTHOR_READOUT.json").write_text(json.dumps(extracted,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
        for name in OUTPUT_NAMES:
            (staged/name).rename(OUT/name)
    return dict(passed=True,output_files=list(OUTPUT_NAMES),rows=4,paired_contrasts=6,
                scientific_samples_generated=0,scientific_score_calls=0,new_integrals=0,
                source_sha256=sha(Path(__file__)))


def check() -> dict:
    stats,reference,audit,hashes=load_final_inputs()
    saved=read(OUT/"AUTHOR_READOUT.json")
    expected=extract(stats,reference,audit,hashes)
    artifact_records=saved.get("artifacts")
    if not isinstance(artifact_records,dict) or set(artifact_records)!={"READABLE_TABLE.md","bridge_summary.png","bridge_summary.svg"}:
        raise ValueError("saved report artifact schema mismatch")
    expected["artifacts"]=artifact_records
    if saved!=expected:
        raise ValueError("saved readout differs from deterministic scalar extraction")
    if (OUT/"READABLE_TABLE.md").read_text("utf-8")!=readable(expected):
        raise ValueError("readable table reconstruction mismatch")
    for name,record in artifact_records.items():
        path=OUT/name
        if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:
            raise ValueError("saved report artifact hash/size mismatch: "+name)
    return dict(passed=True,scope="final input/output hashes and deterministic saved scalar/table reconstruction",
                plots_rerendered=0,scientific_arrays_recomputed=0,new_integrals=0,new_random_samples=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--check",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run() if args.run else check(),ensure_ascii=False,indent=2,allow_nan=False))
