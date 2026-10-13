"""Independent fixed-order REF reconstruction. Never imports the producer."""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_learning/REF"
DOC = ROOT / "docs/08_research_研究工作/有限样本学习与均值自由度_20261001"
RECEIPT = OUT / "INDEPENDENT_AUDIT.json"
PLAN = DOC / "REF_AUDIT.md"
MARKER = "<!-- REF_AUDIT_PLAN_END -->"
AXIS = (1., 1.5, 2., 3., 4., 6., 8.)
PAIRS = list(itertools.product(AXIS, AXIS))
TRUTH = PAIRS.index((6., 2.))
AUDIT_OUTER = 384
AUDIT_PRIVATE = 2048
GH_CACHE = {}
GH_META = {}
WORK = {"lookup_private_node_pairs": 0, "pmf_private_node_pairs": 0,
        "quad_location_private_node_pairs": 0, "private_scalar_expectations": 0,
        "private_integrand_contributions": 0, "expit_scalar_evaluations": 0,
        "independent_PMF_matrix_integrals": 0, "outer_scalar_integrand_evaluations": 0,
        "saved_variant_PMFs_reanalyzed": 0, "new_scientific_models": 0,
        "new_random_samples": 0, "new_scientific_fits": 0, "RNG_calls": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_hash():
    text = PLAN.read_text(encoding="utf-8")
    if text.count(MARKER) != 1:
        raise RuntimeError("Missing or repeated fixed audit-plan marker")
    prefix = text.split(MARKER)[0] + MARKER + "\n"
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def save(obj, initial=False):
    with RECEIPT.open("x" if initial else "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_bundle(workspace):
    freeze = read(OUT / "FREEZE.json")
    result = read(OUT / "RESULTS_FREEZE.json")
    require(result["source_freeze_sha256"] == sha(OUT / "FREEZE.json"), "REF chain")
    files = dict(freeze["files"])
    files.update(result["files"])
    files.update({str((OUT / name).relative_to(ROOT)).replace("\\", "/"): sha(OUT / name)
                  for name in ("FREEZE.json", "RESULTS_FREEZE.json", "STATE.json")})
    for name, digest in files.items():
        require(sha(ROOT / name) == digest, "Public source/result changed: " + name)
    if workspace:
        for name, digest in result["local_arrays"].items():
            require(sha(ROOT / name) == digest, "Private lookup changed: " + name)
    require(read(OUT / "STATE.json")["status"] == "complete", "REF not complete")
    require(read(OUT / "QUALIFICATION.json")["all_passed"], "Original REF not qualified")
    return files, dict(result["local_arrays"])


def normal_rule(order):
    if order not in GH_CACHE:
        z, vectors = eigh_tridiagonal(np.zeros(order), np.sqrt(np.arange(1, order)),
                                     lapack_driver="stemr", check_finite=True)
        w = vectors[0, :] ** 2
        total = float(w.sum())
        require(abs(total - 1) <= 1e-12 and np.all(w >= 0), "Jacobi GH weights")
        w = w / total
        GH_META[str(order)] = {
            "raw_weight_mass": total, "normalized_weight_mass": float(w.sum()),
            "zero_weights": int((w == 0).sum()),
            "node_symmetry_error": float(abs(z + z[::-1]).max()),
            "weight_symmetry_error": float(abs(w - w[::-1]).max()),
            "method": "eigh_tridiagonal standard-normal Jacobi matrix, stemr"}
        GH_CACHE[order] = (z, w)
    return GH_CACHE[order]


def direct(z, d, scale, order, category, derivative=True):
    """Separate h/q sums, no complement subtraction, logsumexp, or clipping."""
    z = np.asarray(z, dtype=float)
    e, w = normal_rule(order)
    rho = .9 / (d - 1)
    slope = scale * math.sqrt(rho)
    private = scale * math.sqrt(1 - rho)
    h = np.empty(z.size)
    q = np.empty(z.size)
    hp = np.empty(z.size) if derivative else None
    for start in range(0, z.size, 64):
        zz = z[start:start + 64]
        eta = slope * zz[:, None] + private * e[None, :]
        positive = expit(eta)
        negative = expit(-eta)
        h[start:start + zz.size] = positive @ w
        q[start:start + zz.size] = negative @ w
        if derivative:
            hp[start:start + zz.size] = slope * ((positive * negative) @ w)
    pairs = int(z.size * order)
    WORK[category + "_private_node_pairs"] += pairs
    WORK["private_scalar_expectations"] += int(z.size * (3 if derivative else 2))
    WORK["private_integrand_contributions"] += pairs * (3 if derivative else 2)
    WORK["expit_scalar_evaluations"] += 2 * pairs
    require(np.all(h > 0) and np.all(q > 0) and np.isfinite(h + q).all(),
            "Independent kernel is nonpositive/nonfinite")
    return h, q, hp


def binary_kl(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    require(np.all((p > 0) & (p < 1)) and np.all((q > 0) & (q < 1)), "Event positivity")
    return p * (np.log(p) - np.log(q)) + (1 - p) * (np.log1p(-p) - np.log1p(-q))


def counts_and_events(matrices, d):
    m = d // 2
    a, b = np.indices((m + 1, m + 1))
    spin = 2 * (a + b) - d
    masks = [spin * spin <= d, (spin > 0) & (spin * spin > d),
             (spin < 0) & (spin * spin > d)]
    return np.stack([matrices[:, mask].sum(axis=1) for mask in masks], axis=1)


def risk_family(variants, d):
    events = np.asarray([counts_and_events(v, d) for v in variants])
    joint, briers, logs = [], [], []
    for teacher in range(len(variants)):
        p = variants[teacher, TRUTH]
        for candidate in range(len(variants)):
            q = variants[candidate]
            joint.append(np.sum(p * (np.log(p) - np.log(q)), axis=(1, 2)))
            pe = events[teacher, TRUTH]
            qe = events[candidate]
            briers.append((pe - qe) ** 2)
            logs.append(binary_kl(pe, qe))
    return np.asarray(joint), events, np.asarray(briers), np.asarray(logs)


def lookup_audit(d, cfg):
    meta = read(OUT / f"LOOKUP_d{d}.json")
    with np.load(ROOT / meta["array"], allow_pickle=False) as f:
        z, saved_h, saved_hp = (f[n].copy() for n in ("z", "h", "derivative"))
    require(z.shape == (4097,) and saved_h.shape == (2, 7, 4097)
            and saved_hp.shape == saved_h.shape, "Lookup shapes")
    step = 24 / 4096
    grid_error = float(abs(z - np.linspace(-12, 12, 4097)).max())
    slope = np.diff(saved_h, axis=2) / step
    middle = 3 * slope - saved_hp[:, :, :-1] - saved_hp[:, :, 1:]
    minimum = min(float(saved_h.min()), float((1 - saved_h).min()))
    eh = float(abs(saved_h[1] - saved_h[0]).max())
    ed = float(abs(saved_hp[1] - saved_hp[0]).max())
    perturbation = eh + step * ed / 4
    stability = d * perturbation / minimum
    rho = .9 / (d - 1)
    factor = math.sqrt(rho / (1 - rho))
    fourth = math.sqrt(24) * factor ** 4
    interpolation = step ** 4 * fourth / 384
    exact_minimum = .25 * math.erfc(12 * factor / math.sqrt(2))
    exact_score = d * interpolation / (exact_minimum - interpolation)
    symmetry = max(float(abs(saved_h + saved_h[:, :, ::-1] - 1).max()),
                   float(abs(saved_hp - saved_hp[:, :, ::-1]).max()))
    comparisons = []
    normalization = 0.
    for oi, order in enumerate((1024, 2048)):
        for si, scale in enumerate(AXIS):
            h, q, hp = direct(z, d, scale, order, "lookup")
            normalization = max(normalization, float(abs(np.log(h + q)).max()))
            dh = abs(h - saved_h[oi, si])
            dq = abs(q - (1 - saved_h[oi, si]))
            dd = abs(hp - saved_hp[oi, si])
            comparisons.append({
                "order": order, "scale": scale, "nodes_compared": z.size,
                "h_max_abs_difference": float(dh.max()),
                "q_vs_saved_complement_max_abs_difference": float(dq.max()),
                "derivative_max_abs_difference": float(dd.max()),
                "maximum_h_error_node": int(dh.argmax()),
                "h_nodes_over_original_gate": int((dh > cfg["h_order_gate"]).sum()),
                "q_nodes_over_original_gate": int((dq > cfg["h_order_gate"]).sum()),
                "derivative_nodes_over_original_gate": int((dd > cfg["derivative_order_gate"]).sum()),
            })
    quad_rows = read(OUT / f"QUAD_d{d}.json")
    quad_comparisons = []
    for scale in AXIS:
        rows = [r for r in quad_rows if r["scale"] == scale]
        require([r["z"] for r in rows] == cfg["quad_z"], "Quad order")
        h, q, hp = direct(cfg["quad_z"], d, scale, 2048, "quad_location")
        for i, row in enumerate(rows):
            values = row["quad_values_h_q_derivative"]
            log_error = max(abs(math.log(h[i]) - math.log(values[0])),
                            abs(math.log(q[i]) - math.log(values[1])))
            derivative_error = abs(float(hp[i]) - values[2])
            quad_comparisons.append({
                "scale": scale, "z": row["z"], "log_probability_difference": log_error,
                "derivative_difference": derivative_error,
                "original_record_passed": row["passed"],
                "passed_original_gates": log_error <= cfg["quad_log_probability_gate"]
                and derivative_error <= cfg["quad_derivative_gate"]})
    checks = {
        "grid_exact": grid_error == 0,
        "all_nodes_strictly_positive": minimum > 0,
        "all_interval_secants_positive": bool(np.all(slope > 0)),
        "all_endpoint_derivatives_nonnegative": bool(np.all(saved_hp >= 0)),
        "all_derivative_Bernstein_middle_coefficients_nonnegative": bool(np.all(middle >= 0)),
        "original_h_order_gate": eh <= cfg["h_order_gate"],
        "original_derivative_order_gate": ed <= cfg["derivative_order_gate"],
        "saved_symmetry_gate": symmetry <= cfg["identity_gate"],
        "independent_normalization_gate": normalization <= cfg["identity_gate"],
        "original_reported_normalization_gate": meta["normalization_log_error"] <= cfg["identity_gate"],
        "interpolator_stability_gate": stability <= cfg["score_order_bound_gate"],
        "exact_node_interpolation_gate": exact_score <= cfg["exact_node_interpolation_score_bound_gate"],
        "node_reconstruction_h_q": all(x["h_nodes_over_original_gate"] == 0
                                       and x["q_nodes_over_original_gate"] == 0 for x in comparisons),
        "node_reconstruction_derivative": all(x["derivative_nodes_over_original_gate"] == 0
                                             for x in comparisons),
        "independent_GH_against_saved_quad": all(x["passed_original_gates"] for x in quad_comparisons),
        "original_quad_flags": all(x["original_record_passed"] for x in quad_comparisons),
        "metadata_stability_matches": abs(stability - meta["vector_logscore_order_stability_bound"]) <= 1e-15,
        "metadata_exact_interpolation_matches": abs(exact_score - meta["analytic_interpolation"]
                                                   ["exact_node_vector_logscore_error_bound"]) <= 1e-15,
    }
    return {"d": d, "passed": all(checks.values()), "checks": checks,
            "node_comparisons": comparisons, "quad_location_comparisons": quad_comparisons,
            "grid_max_error": grid_error, "node_h_order_difference": eh,
            "node_derivative_order_difference": ed, "minimum_node_probability": minimum,
            "minimum_Bernstein_middle_coefficient": float(middle.min()),
            "independent_normalization_log_error": normalization, "saved_symmetry_error": symmetry,
            "interpolator_value_difference_bound": perturbation,
            "interpolator_logscore_stability_bound": stability,
            "exact_node_probability_interpolation_bound": interpolation,
            "exact_node_logscore_interpolation_bound": exact_score,
            "GH_error_not_included_in_analytic_interpolation_bound": True,
            "saved_lookup_log_normalization_statistic_not_reconstructible_exactly":
                "Original lp/lq arrays were not saved; its gate is checked, and fresh h/q normalization is independently checked."}


def binomial_recurrence(h, q, m):
    """Polynomial multiplication by q+h*x; stable without h/q ratios."""
    arr = np.ones((h.size, 1))
    for _ in range(m):
        new = np.zeros((h.size, arr.shape[1] + 1))
        new[:, :-1] += arr * q[:, None]
        new[:, 1:] += arr * h[:, None]
        arr = new
    return arr


def reference_audit(d, cfg):
    record = read(OUT / f"REFERENCE_d{d}.json")
    m = d // 2
    variants = np.asarray(record["variants_pmf"])
    primary = np.asarray(record["pmf"])
    masses = np.asarray(record["raw_variant_masses"])
    require(variants.shape == (3, 49, m + 1, m + 1), "Reference shapes")
    require(np.all(variants > 0) and np.isfinite(variants).all(), "Saved variant positivity")
    WORK["saved_variant_PMFs_reanalyzed"] += 147
    l1 = max(float(abs(variants[a] - variants[b]).sum(axis=(1, 2)).max())
             for a, b in itertools.combinations(range(3), 2))
    logdiff = max(float(abs(np.log(variants[a]) - np.log(variants[b])).max())
                  for a, b in itertools.combinations(range(3), 2))
    mirror = max(float(abs(variants[v, 7*i+j] - variants[v, 7*j+i].T).max())
                 for v in range(3) for i in range(7) for j in range(7))
    complement = float(abs(variants - variants[:, :, ::-1, ::-1]).max())
    logchoose = np.array([math.log(math.comb(m, i)) for i in range(m + 1)])
    pattern = np.log(primary) - logchoose[:, None] - logchoose[None, :]
    risks, events, briers, logs = risk_family(variants, d)
    qualified = {
        "raw_PMF_mass_max_error": float(abs(masses - 1).max()),
        "PMF_L1_variant_difference": l1, "log_PMF_variant_difference": logdiff,
        "mirror_error": mirror, "global_bit_complement_error": complement,
        "teacher_candidate_joint_KL_sensitivity": float(np.ptp(risks, axis=0).max()),
        "event_probability_sensitivity": float(np.ptp(events, axis=0).max()),
        "event_risk_sensitivity": max(float(np.ptp(briers, axis=0).max()),
                                      float(np.ptp(logs, axis=0).max()))}
    outer, weights = normal_rule(AUDIT_OUTER)
    components = []
    kernel_norm = 0.
    for scale in AXIS:
        h, q, _ = direct(outer, d, scale, AUDIT_PRIVATE, "pmf", derivative=False)
        kernel_norm = max(kernel_norm, float(abs(h + q - 1).max()))
        components.append(binomial_recurrence(h, q, m))
    independent = []
    raw_masses = []
    for i, j in itertools.product(range(7), repeat=2):
        raw = np.einsum("k,ka,kb->ab", weights, components[i], components[j], optimize=False)
        mass = float(raw.sum())
        require(np.all(raw > 0) and np.isfinite(raw).all() and mass > 0, "Independent PMF positivity")
        raw_masses.append(mass)
        independent.append(raw / mass)
    independent = np.asarray(independent)
    WORK["independent_PMF_matrix_integrals"] += 49
    WORK["outer_scalar_integrand_evaluations"] += 49 * (m + 1) ** 2 * AUDIT_OUTER
    independent_risks, independent_events, independent_brier, independent_logs = risk_family(
        independent[None, :], d)
    per_candidate = []
    for i, pair in enumerate(PAIRS):
        dl1 = [float(abs(independent[i] - variants[v, i]).sum()) for v in range(3)]
        dlog = [float(abs(np.log(independent[i]) - np.log(variants[v, i])).max()) for v in range(3)]
        per_candidate.append({
            "index": i, "scales": list(pair), "raw_mass": raw_masses[i],
            "min_independent_cell": float(independent[i].min()),
            "L1_against_each_saved_variant": dl1, "max_log_cell_against_each_saved_variant": dlog,
            "joint_KL_difference": float(abs(independent_risks[0, i] - risks[8, i])),
            "event_probability_max_difference": float(abs(independent_events[0, i] - events[2, i]).max()),
            "event_Brier_max_difference": float(abs(independent_brier[0, i] - briers[8, i]).max()),
            "event_logloss_max_difference": float(abs(independent_logs[0, i] - logs[8, i]).max()),
        })
    stored_risk_error = float(abs(risks[8] - np.asarray(record["joint_KL_to_teacher"])).max())
    stored_event_error = float(abs(events[2] - np.asarray(record["event_probability"])).max())
    stored_brier_error = float(abs(briers[8] - np.asarray(record["event_brier_excess"])).max())
    stored_log_error = float(abs(logs[8] - np.asarray(record["event_logloss_excess"])).max())
    checks = {
        "candidate_order": record["parameters"] == [list(p) for p in PAIRS] and record["truth_index"] == TRUTH,
        "primary_is_variant_2": np.array_equal(primary, variants[2]),
        "variant_normalized_mass": float(abs(variants.sum(axis=(2, 3)) - 1).max()) <= cfg["identity_gate"],
        "original_raw_mass_gate": qualified["raw_PMF_mass_max_error"] <= cfg["identity_gate"],
        "original_L1_gate": l1 <= cfg["PMF_L1_gate"],
        "original_log_PMF_gate": logdiff <= cfg["log_PMF_gate"],
        "original_symmetry_gate": max(mirror, complement) <= cfg["identity_gate"],
        "original_sensitivity_gates": max(qualified[k] for k in (
            "teacher_candidate_joint_KL_sensitivity", "event_probability_sensitivity",
            "event_risk_sensitivity")) <= cfg["risk_sensitivity_gate"],
        "log_pattern_combinatorics": float(abs(pattern - np.asarray(record["log_pattern"])).max()) <= 1e-12,
        "stored_risk_and_events_match": max(stored_risk_error, stored_event_error,
                                           stored_brier_error, stored_log_error) <= 1e-12,
        "events_partition_unit_mass": float(abs(events.sum(axis=2) - 1).max()) <= cfg["identity_gate"],
        "independent_kernel_normalization": kernel_norm <= cfg["identity_gate"],
        "independent_raw_mass": max(abs(v - 1) for v in raw_masses) <= cfg["identity_gate"],
        "all_49_independent_L1": all(max(r["L1_against_each_saved_variant"]) <= cfg["PMF_L1_gate"]
                                     for r in per_candidate),
        "all_49_independent_log_cells": all(max(r["max_log_cell_against_each_saved_variant"])
                                            <= cfg["log_PMF_gate"] for r in per_candidate),
        "all_49_independent_risk_events": all(max(r[k] for k in (
            "joint_KL_difference", "event_probability_max_difference", "event_Brier_max_difference",
            "event_logloss_max_difference")) <= cfg["risk_sensitivity_gate"] for r in per_candidate),
    }
    return {"d": d, "passed": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()},
            "recomputed_original_qualification": qualified, "candidates": per_candidate,
            "saved_log_pattern_max_difference": float(abs(pattern - np.asarray(record["log_pattern"])).max()),
            "saved_risk_max_difference": stored_risk_error, "saved_event_max_difference": stored_event_error,
            "saved_Brier_max_difference": stored_brier_error, "saved_logloss_max_difference": stored_log_error,
            "independent_kernel_normalization_error": kernel_norm,
            "independent_raw_masses": raw_masses, "independent_pmf": independent.tolist(),
            "independent_joint_KL": independent_risks[0].tolist(),
            "independent_event_probabilities": independent_events[0].tolist(),
            "independent_event_Brier": independent_brier[0].tolist(),
            "independent_event_logloss": independent_logs[0].tolist(),
            "raw_saved_masses_scope": "Saved raw masses checked; old 256/512 integrations not re-executed. Independent 384 raw masses reconstructed."}


def write_audit():
    require(not RECEIPT.exists(), "Audit receipt exists; no silent rerun")
    files, local = source_bundle(True)
    cfg = read(OUT / "CONFIG.json")
    require(cfg["d"] == [16, 64] and cfg["axis"] == list(AXIS)
            and cfg["reference_variants"] == [[1024, 512], [2048, 256], [2048, 512]],
            "Unexpected producer configuration")
    receipt = {"status": "running", "started_utc": now(), "passed": False,
               "audit_source_sha256": sha(Path(__file__)), "audit_plan_prefix_sha256": plan_hash(),
               "plan_hash_convention": "UTF8/LF prefix through REF_AUDIT_PLAN_END plus newline",
               "frozen_public_files": files, "frozen_local_arrays": local,
               "private_arrays_audited": True, "outer_GH": AUDIT_OUTER, "private_GH": AUDIT_PRIVATE,
               "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                               "platform": platform.platform(), "threads": 1},
               "lookup_audits": [], "reference_audits": [], "work": dict(WORK)}
    save(receipt, initial=True)
    started = time.perf_counter()
    try:
        toy = binomial_recurrence(np.array([.25]), np.array([.75]), 2)[0]
        require(float(abs(toy - [.5625, .375, .0625]).max()) == 0, "Independent recurrence QA")
        for d in (16, 64):
            receipt["lookup_audits"].append(lookup_audit(d, cfg))
            receipt["work"] = dict(WORK)
            save(receipt)
            print("lookup", d, receipt["lookup_audits"][-1]["passed"], flush=True)
            receipt["reference_audits"].append(reference_audit(d, cfg))
            receipt["work"] = dict(WORK)
            save(receipt)
            print("reference", d, receipt["reference_audits"][-1]["passed"], flush=True)
        receipt["passed"] = all(row["passed"] for group in ("lookup_audits", "reference_audits")
                                for row in receipt[group])
        receipt["status"] = "complete" if receipt["passed"] else "completed_with_retained_failures"
    except Exception as exc:
        receipt["status"] = "stopped_with_retained_failure"
        receipt["failure"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    receipt["finished_utc"] = now()
    receipt["elapsed_seconds"] = time.perf_counter() - started
    receipt["work"] = dict(WORK)
    receipt["work"]["total_private_node_pairs"] = sum(WORK[k] for k in (
        "lookup_private_node_pairs", "pmf_private_node_pairs", "quad_location_private_node_pairs"))
    receipt["GH_rule_metadata"] = GH_META
    receipt["strict_integral_error_certificate"] = False
    receipt["scope"] = ("All lookup nodes freshly reconstructed; 294 stored variants reanalyzed; "
                        "98 full PMFs independently integrated with 384/2048 orders. "
                        "No production numerical/statistical function imported. Audit integrations are additional work.")
    save(receipt)
    print(json.dumps({"passed": receipt["passed"], "status": receipt["status"],
                      "elapsed_seconds": receipt["elapsed_seconds"], "work": receipt["work"]}), flush=True)
    if not receipt["passed"]:
        raise SystemExit(1)


def check(workspace):
    audit = read(RECEIPT)
    require(audit["status"] == "complete" and audit["passed"], "Audit did not pass")
    require(audit["audit_source_sha256"] == sha(Path(__file__)), "Audit source changed")
    require(audit["audit_plan_prefix_sha256"] == plan_hash(), "Audit plan changed")
    for name, digest in audit["frozen_public_files"].items():
        require(sha(ROOT / name) == digest, "Audited public source/result changed: " + name)
    if workspace:
        for name, digest in audit["frozen_local_arrays"].items():
            require(sha(ROOT / name) == digest, "Audited local array changed: " + name)
    print(json.dumps({"passed": True, "workspace": workspace, "integrals_reexecuted": False,
                      "private_arrays_were_audited_in_write": audit["private_arrays_audited"],
                      "receipt_sha256": sha(RECEIPT)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("write", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    write_audit() if args.phase == "write" else check(args.workspace)
