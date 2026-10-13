"""Locked bounded statistics and reference-only scalar quadrature.

analyse is pure: it never samples, integrates, or reads study files. Reference
quadrature is separately frozen before --run-reference. --check reintegrates
nothing; it replays saved scalar algebra and, if present, the new saved scores.
Native arithmetic allowances are qualification conditions, not certificates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/asymptotic_bridge_v1_20261008"
REFERENCE_CONTRACT = ART / "REFERENCE_CONTRACT.json"
REFERENCE = ART / "REFERENCE.json"
STATISTICS = ART / "STATISTICS.json"
RAW = ART / "BANK_SCORES.npy"
N_VALUES = (64, 256, 1024, 4096)
METRIC_NAMES = ("path_gap", "D12", "overlap", "coupling", "direct_gap_truncated")
K = 100
FORMAL_BANK_COUNT = 131072
ALPHA = .01
TARGET_COUNT = 26
DELTA = ALPHA / TARGET_COUNT
NUMERICAL_ALLOWANCE = 1e-9
OLD_RESULT = "artifacts/submission_candidate_v1_7_1_20261008/CALIBRATION_RESULTS.json"
OLD_MANIFEST = "artifacts/submission_candidate_v1_7_1_20261008/CALIBRATION_MANIFEST.json"
OLD_CONTRACT = "artifacts/submission_candidate_v1_7_1_20261008/CALIBRATION_CONTRACT.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def direct_gap_constants():
    """Old K-general Gaussian maximum/union bound, without old data access."""
    a = max(1 - math.sqrt(.1), math.sqrt(2.8) - 1)
    t = 8.
    m = 3 * a * t
    bk = (8 / 3) * (math.log(K) + 3 * math.log(2))
    union = 2 * K * 3 * math.exp(-t*t/2) / (math.sqrt(2*math.pi) * t)
    tail = math.sqrt(3) * a * math.sqrt(bk * union)
    return dict(T=t, A=a, M=m, support=[-m, m], support_width=2*m,
                b_K=bk, Gaussian_union_probability_bound=union, tail_allowance=tail,
                formula="sqrt(3)*A*sqrt(b_K*(2*K*3*exp(-T^2/2)/(sqrt(2*pi)*T)))",
                source="script/nuisance_contrast_compute.py: population T8 audit")


def support_bounds(reference):
    h = float(reference["m_H45"])
    if not math.isfinite(h) or not 0 < h <= .028125:
        raise ValueError("m_H45 must lie in (0,.028125]")
    s = 1 - 1 / K
    return dict(path_gap=(-.675*s, 1.35*s), D12=(-s/2, s/2),
                overlap=(0., s), coupling=(-(1/2+h)*s, (1-h)*s/2),
                direct_gap_truncated=tuple(direct_gap_constants()["support"]))


def config_dump():
    return dict(schema="asymptotic_bridge_statistics_v1", n=list(N_VALUES), K=K,
                metric_names=list(METRIC_NAMES), formal_bank_count=FORMAL_BANK_COUNT,
                alpha=ALPHA, target_count=TARGET_COUNT, formal_look_count=1,
                delta_per_two_sided_target=DELTA, delta_per_side=DELTA/2,
                radius_logarithm="log(4/delta)", empirical_variance_ddof=1,
                numerical_allowance_raw=NUMERICAL_ALLOWANCE,
                numerical_allowance_is_qualification_not_certificate=True,
                source="script/direction_validation_statistics.py:sample_empirical_bernstein_ci",
                theorem_source="https://arxiv.org/pdf/0907.3740",
                target_law="ideal iid complete banks and independent continuous Uniform(0,.9)",
                exact_coefficient_enclosures=False, coefficient_target="frozen numeric references",
                pairing="next n minus previous n; calculate each contrast bankwise",
                extra_looks_or_outcome_based_targets_allowed=False)


def _moments(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("A finite one-dimensional sample of at least 2 banks is required")
    mean = math.fsum(map(float, x)) / len(x)
    variance = math.fsum((float(v) - mean)**2 for v in x) / (len(x)-1)
    return mean, variance


def bounded_summary(samples, lower, upper, numerical_allowance=NUMERICAL_ALLOWANCE,
                    tail_allowance=0., delta=DELTA):
    x = np.asarray(samples, dtype=float)
    if not all(math.isfinite(float(v)) for v in (lower, upper, numerical_allowance, tail_allowance, delta)):
        raise ValueError("Support, allowances and delta must be finite")
    if lower > upper or min(numerical_allowance, tail_allowance) < 0 or not 0 < delta < 1:
        raise ValueError("Invalid support, allowance or delta")
    mean, variance = _moments(x)
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("Scores lie outside the externally fixed global support")
    banks = len(x)
    width = upper-lower
    logterm = math.log(4/delta)
    statistical = math.sqrt(2*variance*logterm/banks) + 7*width*logterm/(3*(banks-1))
    halfwidth = statistical + numerical_allowance + tail_allowance
    return dict(banks=banks, mean=mean, unbiased_variance=variance,
                SE=math.sqrt(variance/banks), support=[lower, upper], support_width=width,
                sample_min=float(x.min()), sample_max=float(x.max()),
                statistical_halfwidth=statistical, numerical_allowance=numerical_allowance,
                tail_allowance=tail_allowance, halfwidth=halfwidth,
                interval=[mean-halfwidth, mean+halfwidth], delta=delta,
                delta_per_side=delta/2, method="bounded_two_sided_empirical_Bernstein",
                variance_source="same-bank samples; ddof=1",
                range_source="frozen analytic global support, never sample extrema",
                numerical_error_certified=False, intervals_clipped_to_support=False)


def affine_summary(summary, scale, shift, target):
    """Reuse the original confidence event; no additional target or alpha."""
    scale, shift = float(scale), float(shift)
    if not math.isfinite(scale) or not math.isfinite(shift):
        raise ValueError("Finite affine coefficients required")
    result = dict(summary)
    result["mean"] = scale*summary["mean"] + shift
    result["unbiased_variance"] = scale*scale*summary["unbiased_variance"]
    for key in ("SE", "support_width", "statistical_halfwidth", "numerical_allowance", "tail_allowance", "halfwidth"):
        result[key] = abs(scale)*summary[key]
    for key in ("interval", "support"):
        result[key] = sorted(scale*v+shift for v in summary[key])
    endpoints = sorted(scale*summary[key]+shift for key in ("sample_min", "sample_max"))
    result["sample_min"], result["sample_max"] = endpoints
    result.update(reuses_target=target, affine_scale=scale, affine_shift=shift,
                  coefficient_target="frozen numeric reference", extra_alpha=0.)
    return result


def _validate_reference(reference):
    for key in ("v", "J100", "gapV", "m_H45"):
        value = float(reference[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be positive and finite")
    support_bounds(reference)
    ws = [float(reference["w_by_n"][str(n)]) for n in N_VALUES]
    expected = [reference["J100"]/math.sqrt(n*reference["v"]) for n in N_VALUES]
    if not np.allclose(ws, expected, rtol=2e-15, atol=0):
        raise ValueError("w_by_n does not agree with frozen v/J100")
    return ws


def analyse(scores, reference):
    """Pure analysis of B x 4 x 5 paired scores, B>=2; no file access."""
    if np.iscomplexobj(scores):
        raise ValueError("Scores must be real")
    x = np.array(scores, dtype=float, copy=True)
    if x.ndim != 3 or x.shape[1:] != (len(N_VALUES), len(METRIC_NAMES)) or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("Scores must be finite B x 4 x 5 with B>=2")
    ws = _validate_reference(reference)
    h, gapv = float(reference["m_H45"]), float(reference["gapV"])
    bounds = support_bounds(reference)
    tail = direct_gap_constants()["tail_allowance"]
    per_n, target_order, target_samples = [], [], []
    for ni, (n, w) in enumerate(zip(N_VALUES, ws)):
        raw = {}
        for mi, metric in enumerate(METRIC_NAMES):
            target = f"n{n}:{metric}"
            lo, hi = bounds[metric]
            raw[metric] = bounded_summary(x[:, ni, mi], lo, hi,
                                         tail_allowance=tail if metric == "direct_gap_truncated" else 0.)
            raw[metric]["target_id"] = target
            target_order.append(target)
            target_samples.append(x[:, ni, mi])
        d, g, c = x[:, ni, 1], x[:, ni, 2], x[:, ni, 3]
        identity = d+h*g+c
        if np.max(np.abs(identity)) > NUMERICAL_ALLOWANCE:
            raise ValueError("Bankwise D12=-m_H45*overlap-coupling identity failed")
        derived = dict(A=affine_summary(raw["path_gap"], 1/(w*gapv), 0., f"n{n}:path_gap"),
                       B=affine_summary(raw["D12"], -1/(w*h), 0., f"n{n}:D12"),
                       omega_over_w=affine_summary(raw["overlap"], 1/w, 0., f"n{n}:overlap"),
                       R=affine_summary(raw["D12"], 1/w, h, f"n{n}:D12"),
                       O=affine_summary(raw["overlap"], -h/w, h, f"n{n}:overlap"),
                       J=affine_summary(raw["coupling"], -1/w, 0., f"n{n}:coupling"))
        residual_samples = np.column_stack((d/w+h, h*(1-g/w), -c/w))
        discrepancy = x[:, ni, 4]-x[:, ni, 0]
        dm, dv = _moments(discrepancy)
        per_n.append(dict(n=n, w=w, raw=raw, derived=derived,
                          bank_identity_max_absolute_residual=float(np.max(np.abs(identity))),
                          residual_covariance_order=["R", "O", "J"],
                          residual_covariance=np.cov(residual_samples, rowvar=False, ddof=1).tolist(),
                          residual_mean_identity_error=derived["R"]["mean"]-derived["O"]["mean"]-derived["J"]["mean"],
                          direct_minus_path_diagnostic=dict(mean=dm, unbiased_variance=dv,
                              paired_SE=math.sqrt(dv/len(x)), diagnostic_only=True,
                              additional_confidence_claim=False, enforced_pass_threshold=None)))
    paired_gap, paired_gradient = [], []
    for kind, mi, denominator, output in (("A", 0, gapv, paired_gap), ("B", 1, h, paired_gradient)):
        sign = 1. if kind == "A" else -1.
        lo, hi = bounds[METRIC_NAMES[mi]]
        for ni in range(3):
            previous_scale, next_scale = sign/(ws[ni]*denominator), sign/(ws[ni+1]*denominator)
            prev_support = sorted((previous_scale*lo, previous_scale*hi))
            next_support = sorted((next_scale*lo, next_scale*hi))
            z = next_scale*x[:, ni+1, mi] - previous_scale*x[:, ni, mi]
            numerical = (abs(previous_scale)+abs(next_scale))*NUMERICAL_ALLOWANCE
            summary = bounded_summary(z, next_support[0]-prev_support[1], next_support[1]-prev_support[0], numerical)
            target = f"{kind}:n{N_VALUES[ni+1]}_minus_n{N_VALUES[ni]}"
            summary["target_id"] = target
            within_cov = float(np.cov(x[:, ni+1, mi], x[:, ni, mi], ddof=1)[0, 1])
            vm = summary["unbiased_variance"]
            independent_v = next_scale**2*per_n[ni+1]["raw"][METRIC_NAMES[mi]]["unbiased_variance"] + previous_scale**2*per_n[ni]["raw"][METRIC_NAMES[mi]]["unbiased_variance"]
            covariance_v = independent_v - 2*next_scale*previous_scale*within_cov
            output.append(dict(n_previous=N_VALUES[ni], n_next=N_VALUES[ni+1], direction="next_minus_previous",
                               previous_scale=previous_scale, next_scale=next_scale, summary=summary,
                               raw_same_bank_covariance=within_cov, independent_variance_not_used=independent_v,
                               covariance_reconstructed_variance=covariance_v,
                               covariance_reconstruction_absolute_error=abs(vm-covariance_v)))
            target_order.append(target)
            target_samples.append(z)
    matrix = np.column_stack(target_samples)
    covariance26 = np.cov(matrix, rowvar=False, ddof=1)
    return dict(schema="asymptotic_bridge_statistics_v1", config=config_dump(), banks=len(x),
                formal_bank_count_matches=(len(x) == FORMAL_BANK_COUNT), n=list(N_VALUES), K=K,
                metric_order=list(METRIC_NAMES), reference_coefficients={k:reference[k] for k in ("v", "J100", "gapV", "m_H45", "w_by_n")},
                direct_gap_T8=direct_gap_constants(), per_n=per_n,
                paired_gap=paired_gap, paired_gradient=paired_gradient,
                target_order=target_order, target_means=[math.fsum(map(float, v))/len(x) for v in target_samples],
                covariance20_order=target_order[:20], covariance20=covariance26[:20, :20].tolist(),
                covariance26_order=target_order, covariance26=covariance26.tolist(),
                covariance_ddof=1, family_target_count=len(target_order), shared_bank_pairing_preserved=True,
                interpretation="One 99% family for 26 fixed targets, conditional on ideal iid and stated numerical qualification; native error is not certified.",
                scientific_samples_generated_by_analysis=0, integrals_evaluated_by_analysis=0)


def freeze_reference():
    """Write the reference-only quadrature rules before evaluating integrands."""
    sources = [OLD_RESULT, OLD_MANIFEST, OLD_CONTRACT, "script/candidate_v171_calibration.py",
               "script/nuisance_contrast_compute.py", "script/direction_validation_statistics.py",
               "docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_7_1_20261008/APPENDIX_THEORY_EN.md",
               Path(__file__).relative_to(ROOT).as_posix()]
    contract = dict(schema="asymptotic_bridge_reference_contract_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes={rel:sha(ROOT/rel) for rel in sources},
                    scope=dict(reference_only=True, rho=.45, mean_mark="kappa(.45)/4", n=list(N_VALUES), K=100,
                               new_banks=0, new_random_samples=0, scientific_NPZ_reads=0, new_training_steps=0),
                    old_coefficients=dict(result=OLD_RESULT, manifest=OLD_MANIFEST,
                                          v_key="derived.v", J100_key="derived.J100", gapV_key="-derived.V_Cb"),
                    rules=[dict(id=f"GH{order}", rule="Gauss-Hermite standard normal", order=order,
                                transform="x=a, y=rho*a+sqrt(1-rho^2)*b") for order in (40, 64, 96)] +
                          [dict(id="GL128", rule="Gauss-Legendre normal-density weighted", order=128,
                                interval=[-12., 12.], renormalize=False,
                                transform="x=sqrt((1+rho)/2)*a+sqrt((1-rho)/2)*b; y=sqrt((1+rho)/2)*a-sqrt((1-rho)/2)*b")],
                    primary_rule="GH96", quadrature_integrals_planned=4,
                    bivariate_nodes_planned=sum(order*order for order in (40,64,96,128)),
                    qualification=dict(max_absolute_m_H45_rule_difference=1e-10,
                                       maximum_normal_mass_error=1e-12,
                                       rigorous_integration_enclosure=False,
                                       sensitivity_gate_is_numerical_only=True,
                                       frozen_numeric_reference_target=True),
                    integration_scope="Four 2D expectations only; no new v/J/V integration, no banks, no NPZ access")
    save_new(REFERENCE_CONTRACT, contract)
    return dict(passed=True, contract_sha256=sha(REFERENCE_CONTRACT), new_integrals=0, new_random_samples=0)


def _reference_sources(contract):
    for rel, digest in contract["source_hashes"].items():
        if sha(ROOT/rel) != digest:
            raise ValueError("Changed reference source: " + rel)
    old = read(ROOT/OLD_RESULT)
    manifest = read(ROOT/OLD_MANIFEST)
    if manifest["results"] != sha(ROOT/OLD_RESULT) or manifest["script"] != sha(ROOT/"script/candidate_v171_calibration.py") or manifest["contract"] != sha(ROOT/OLD_CONTRACT):
        raise ValueError("Old calibration result/producer manifest mismatch")
    return old


def run_reference():
    if REFERENCE.exists():
        raise ValueError("Refusing to overwrite an existing reference receipt")
    import scipy
    from scipy.special import roots_hermitenorm, roots_legendre
    contract = read(REFERENCE_CONTRACT)
    old = _reference_sources(contract)
    rho = contract["scope"]["rho"]
    records = []
    for rule in contract["rules"]:
        order = rule["order"]
        if rule["id"].startswith("GH"):
            nodes, weights = roots_hermitenorm(order)
            weights = weights / math.sqrt(2*math.pi)
            a = nodes[:, None]
            b = rho*a + math.sqrt(1-rho*rho)*nodes[None, :]
        else:
            nodes, weights = roots_legendre(order)
            lo, hi = rule["interval"]
            nodes = (lo+hi)/2 + (hi-lo)/2*nodes
            weights = weights*(hi-lo)/2*np.exp(-nodes*nodes/2)/math.sqrt(2*math.pi)
            a = math.sqrt((1+rho)/2)*nodes[:, None] + math.sqrt((1-rho)/2)*nodes[None, :]
            b = math.sqrt((1+rho)/2)*nodes[:, None] - math.sqrt((1-rho)/2)*nodes[None, :]
        value = float(weights @ (np.tanh(a/2)*np.tanh(b/2)) @ weights)
        records.append(dict(id=rule["id"], rule=rule, rho=rho, kappa45=value, m_H45=value/4,
                            normal_nodes=nodes.tolist(), normal_weights=weights.tolist(),
                            normal_mass=math.fsum(map(float,weights)), integral_count=1,
                            bivariate_node_evaluations=order*order))
    primary = next(r for r in records if r["id"] == contract["primary_rule"])
    difference = max(abs(r["m_H45"]-primary["m_H45"]) for r in records)
    mass_error = max(abs(r["normal_mass"]-1) for r in records)
    gate = contract["qualification"]
    passed = difference <= gate["max_absolute_m_H45_rule_difference"] and mass_error <= gate["maximum_normal_mass_error"]
    if not passed:
        raise ValueError("Frozen reference qualification failed: " + repr((difference,mass_error)))
    v, j, gapv = old["derived"]["v"], old["derived"]["J100"], -old["derived"]["V_Cb"]
    result = dict(schema="asymptotic_bridge_reference_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  contract_sha256=sha(REFERENCE_CONTRACT), source_hashes=contract["source_hashes"],
                  script_sha256=sha(__file__), environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                  v=v, J100=j, gapV=gapv, V_Cb=old["derived"]["V_Cb"], V_I=0.,
                  m_H45=primary["m_H45"], kappa45=primary["kappa45"],
                  w_by_n={str(n):j/math.sqrt(n*v) for n in N_VALUES},
                  old_reference_keys=contract["old_coefficients"], primary_rule=contract["primary_rule"],
                  quadrature_records=records,
                  qualification=dict(passed=passed, max_absolute_m_H45_rule_difference=difference,
                                     maximum_normal_mass_error=mass_error, rigorous_integration_enclosure=False,
                                     frozen_numeric_reference_target=True),
                  actual_work=dict(new_reference_integrals=len(records), bivariate_node_evaluations=sum(r["bivariate_node_evaluations"] for r in records),
                                   new_random_samples=0, new_banks=0, scientific_NPZ_reads=0, new_training_steps=0),
                  direct_gap_T8=direct_gap_constants(), statistical_config=config_dump(),
                  limitations="Rule agreement is numerical qualification, not an enclosure of the exact coefficient. Old v/J100/gapV are reused unchanged.")
    save_new(REFERENCE, result)
    return dict(passed=passed, reference_sha256=sha(REFERENCE), m_H45=result["m_H45"], actual_work=result["actual_work"], qualification=result["qualification"])


def check_reference():
    contract, result = read(REFERENCE_CONTRACT), read(REFERENCE)
    old = _reference_sources(contract)
    if result["contract_sha256"] != sha(REFERENCE_CONTRACT) or result["source_hashes"] != contract["source_hashes"]:
        raise ValueError("Changed reference contract")
    if result["script_sha256"] != sha(__file__):
        raise ValueError("Changed reference script")
    for key, expected in (("v",old["derived"]["v"]), ("J100",old["derived"]["J100"]), ("gapV",-old["derived"]["V_Cb"])):
        if result[key] != expected:
            raise ValueError("Changed original scalar: " + key)
    records = result["quadrature_records"]
    if len(records) != 4:
        raise ValueError("Expected four saved reference integrals")
    for rule, row in zip(contract["rules"], records):
        if row["rule"] != rule or len(row["normal_nodes"]) != rule["order"] or len(row["normal_weights"]) != rule["order"]:
            raise ValueError("Changed reference rule/node ledger")
        if row["m_H45"] != row["kappa45"]/4 or row["normal_mass"] != math.fsum(row["normal_weights"]) or row["bivariate_node_evaluations"] != rule["order"]**2:
            raise ValueError("Saved reference scalar algebra mismatch")
    if result["primary_rule"] != contract["primary_rule"]:
        raise ValueError("Changed primary reference rule")
    primary = next(r for r in records if r["id"] == result["primary_rule"])
    difference = max(abs(r["m_H45"]-primary["m_H45"]) for r in records)
    mass_error = max(abs(r["normal_mass"]-1) for r in records)
    if result["m_H45"] != primary["m_H45"] or result["kappa45"] != primary["kappa45"] or result["qualification"]["max_absolute_m_H45_rule_difference"] != difference or result["qualification"]["maximum_normal_mass_error"] != mass_error:
        raise ValueError("Saved reference qualification mismatch")
    if difference > contract["qualification"]["max_absolute_m_H45_rule_difference"] or mass_error > contract["qualification"]["maximum_normal_mass_error"]:
        raise ValueError("Reference qualification gate failed")
    _validate_reference(result)
    expected_work = dict(new_reference_integrals=4, bivariate_node_evaluations=contract["bivariate_nodes_planned"],
                         new_random_samples=0, new_banks=0, scientific_NPZ_reads=0, new_training_steps=0)
    if result["actual_work"] != expected_work or result["statistical_config"] != config_dump() or not result["qualification"]["passed"]:
        raise ValueError("Changed reference work/configuration ledger")
    if result["direct_gap_T8"] != direct_gap_constants():
        raise ValueError("Saved tail algebra mismatch")
    return dict(passed=True, reference_sha256=sha(REFERENCE), new_integrals=0, new_random_samples=0, scientific_NPZ_reads=0,
                original_reference_integrals=result["actual_work"]["new_reference_integrals"])


def analyze_file(raw_path=RAW):
    check_reference()
    scores = np.load(raw_path, allow_pickle=False)
    if scores.shape != (FORMAL_BANK_COUNT,4,5):
        raise ValueError("Formal analysis requires the frozen 131072-bank panel")
    result = analyse(scores, read(REFERENCE))
    result.update(raw_path=Path(raw_path).resolve().relative_to(ROOT).as_posix(), raw_sha256=sha(raw_path),
                  reference_sha256=sha(REFERENCE), statistics_script_sha256=sha(__file__))
    save_new(STATISTICS, result)
    return dict(passed=True, banks=result["banks"], family_target_count=result["family_target_count"], statistics_sha256=sha(STATISTICS))


def check_saved_statistics():
    check_reference()
    saved = read(STATISTICS)
    raw_path = ROOT/saved["raw_path"]
    if sha(raw_path) != saved["raw_sha256"] or sha(REFERENCE) != saved["reference_sha256"] or sha(__file__) != saved["statistics_script_sha256"]:
        raise ValueError("Changed saved statistics input/source")
    result = analyse(np.load(raw_path, allow_pickle=False), read(REFERENCE))
    expected = {k:v for k,v in saved.items() if k not in ("raw_path", "raw_sha256", "reference_sha256", "statistics_script_sha256")}
    if result != expected:
        raise ValueError("Saved statistics replay mismatch")
    return dict(passed=True, banks=result["banks"], family_target_count=result["family_target_count"],
                new_integrals=0, new_random_samples=0, old_scientific_array_reads=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze-reference", "run-reference", "check-reference", "analyze", "check"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    if args.freeze_reference:
        answer = freeze_reference()
    elif args.run_reference:
        answer = run_reference()
    elif args.check_reference:
        answer = check_reference()
    elif args.analyze:
        answer = analyze_file(args.raw)
    else:
        answer = check_saved_statistics() if STATISTICS.exists() else check_reference()
    print(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False))
