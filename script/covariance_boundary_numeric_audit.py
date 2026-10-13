"""Independent full-probability audit of frozen covariance-boundary NUMERIC."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
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
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/covariance_boundary/NUMERIC"
DOC = ROOT / "docs/08_research_研究工作/相关结构补偿与发射噪声_20261001"
PLAN = DOC / "NUMERIC_AUDIT.md"
MARKER = "<!-- NUMERIC_AUDIT_PLAN_END -->"
RECEIPT = OUT / "INDEPENDENT_AUDIT.json"
NAMES = ("teacher", "fixedRfloor", "comp", "half", "over", "independent")
OLD_N = ROOT / "artifacts/objective_strengthening/N"
OLD_REF = ROOT / "artifacts/finite_learning/REF"
OUTER, INNER = 768, 2048
RULES, RULE_META = {}, {}
WORK = {"private_node_pairs_pmf": 0, "private_node_pairs_quad": 0,
        "private_scalar_expectations": 0, "expit_scalar_evaluations": 0,
        "independent_PMF_matrix_integrals": 0, "outer_scalar_integrand_evaluations": 0,
        "saved_variant_PMFS_reanalyzed": 0, "Decimal80_KL_evaluations": 0,
        "Decimal80_cells": 0, "new_alpha_integrals": 0, "new_scientific_models": 0,
        "new_random_samples": 0, "new_scientific_fits": 0, "RNG_calls": 0}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def plan_hash():
    text = PLAN.read_text(encoding="utf-8")
    require(text.count(MARKER) == 1, "Audit plan marker")
    prefix = text.split(MARKER)[0] + MARKER + "\n"
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


def save(data, initial=False):
    def serial(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    with RECEIPT.open("x" if initial else "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False, default=serial)
        f.write("\n")


def sources(workspace):
    freeze = read(OUT / "FREEZE.json")
    result = read(OUT / "RESULTS_FREEZE.json")
    require(result["source_freeze_sha256"] == sha(OUT / "FREEZE.json"), "NUMERIC chain")
    files = dict(freeze["files"])
    files.update(result["files"])
    files.update(read(OUT / "INPUT_HASHES.json"))
    for p in (OUT / "FREEZE.json", OUT / "RESULTS_FREEZE.json", OUT / "STATE.json", OLD_REF / "FREEZE.json"):
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    for name, digest in files.items():
        require(sha(ROOT / name) == digest, "Public input/result hash: " + name)
    nseal, rseal = read(OLD_N / "RESULTS_FREEZE.json"), read(OLD_REF / "RESULTS_FREEZE.json")
    require(nseal["source_freeze_sha256"] == sha(OLD_N / "FREEZE.json"), "Old N chain")
    require(rseal["source_freeze_sha256"] == sha(OLD_REF / "FREEZE.json"), "Old REF chain")
    for p, seal in [(OLD_N / "MOMENTS.json", nseal), (OLD_N / "LIMITS.json", nseal),
                    (OLD_REF / "REFERENCE_d16.json", rseal), (OLD_REF / "REFERENCE_d64.json", rseal)]:
        require(sha(p) == seal["files"][p.relative_to(ROOT).as_posix()], "Old sealed numerical input")
    if workspace:
        for name, digest in result["local_arrays"].items():
            require(sha(ROOT / name) == digest, "Local variants hash: " + name)
    require(read(OUT / "STATE.json")["status"] == "complete", "Producer incomplete")
    require(read(OUT / "SUMMARY.json")["all_qualified"], "Producer unqualified")
    return files, dict(result["local_arrays"])


def normal_rule(n):
    if n not in RULES:
        x, vectors = eigh_tridiagonal(np.zeros(n), np.sqrt(np.arange(1, n)),
                                     lapack_driver="stemr", check_finite=True)
        w = vectors[0] ** 2
        mass = float(w.sum())
        require(abs(mass - 1) <= 1e-12 and np.all(w >= 0), "Jacobi weights")
        w = w / mass
        RULES[n] = x, w
        RULE_META[str(n)] = {"method": "standard-normal Jacobi eigendecomposition",
                            "raw_weight_mass": mass, "normalized_weight_mass": float(w.sum()),
                            "node_symmetry_error": float(abs(x + x[::-1]).max()),
                            "weight_symmetry_error": float(abs(w - w[::-1]).max()),
                            "zero_weights": int((w == 0).sum())}
    return RULES[n]


def kernel(z, scale, loading, category):
    z = np.asarray(z, dtype=float)
    e, w = normal_rule(INNER)
    h, q = np.empty(z.size), np.empty(z.size)
    for start in range(0, z.size, 64):
        zz = z[start:start + 64]
        eta = scale * (loading * zz[:, None] + math.sqrt(1 - loading ** 2) * e)
        h[start:start + zz.size] = expit(eta) @ w
        q[start:start + zz.size] = expit(-eta) @ w
    require(np.all(h > 0) and np.all(q > 0) and np.isfinite(h + q).all(), "Kernel positivity")
    nodes = int(z.size * INNER)
    WORK["private_node_pairs_" + category] += nodes
    WORK["private_scalar_expectations"] += 2 * int(z.size)
    WORK["expit_scalar_evaluations"] += 2 * nodes
    return h, q


def binomial(h, q, m):
    a = np.ones((h.size, 1))
    for _ in range(m):
        b = np.zeros((h.size, a.shape[1] + 1))
        b[:, :-1] += a * q[:, None]
        b[:, 1:] += a * h[:, None]
        a = b
    return a


def normalized(x):
    x = np.asarray(x, dtype=float).ravel()
    require(np.all(x > 0) and np.isfinite(x).all(), "Full support required")
    return x / math.fsum(x.tolist())


def stable_kl(p, q):
    """Independent symmetric-atanh expansion, not producer delta-polynomial."""
    p, q = normalized(p), normalized(q)
    u = (p - q) / (p + q)
    small = abs(u) <= .05
    terms = np.empty_like(p)
    us = u[small]
    power, correction = us ** 3, np.zeros(us.size)
    for k in range(1, 17):
        correction += power / (2 * k + 1)
        power *= us * us
    terms[small] = (p[small] + q[small]) * (us * us + (1 + us) * correction)
    # Never evaluate atanh/log1p at rounded u=+/-1 in far-tail cells.
    terms[~small] = p[~small] * (np.log(p[~small]) - np.log(q[~small])) - p[~small] + q[~small]
    require(np.isfinite(terms).all() and np.all(terms >= 0), "Generalized KL term")
    return math.fsum(terms.tolist())


def decimal_direct_kl(p, q):
    """Direct normalized sum p*ln(p/q), distinct from generalized-KL summation."""
    with localcontext() as context:
        context.prec = 80
        pp = [Decimal.from_float(float(v)) for v in np.asarray(p).ravel()]
        qq = [Decimal.from_float(float(v)) for v in np.asarray(q).ravel()]
        sp, sq = sum(pp), sum(qq)
        value = sum((a / sp) * ((a / sp) / (b / sq)).ln() for a, b in zip(pp, qq))
    WORK["Decimal80_KL_evaluations"] += 1
    WORK["Decimal80_cells"] += len(pp)
    return str(value), float(value)


def alpha_sources():
    m = read(OLD_N / "MOMENTS.json")
    require(m["qualified"] and not m["interval_certified"], "Old alpha scope")
    require(m["gh"][-1]["order"] == 2048, "Alpha primary order")
    a = {}
    for name, order in (("primary", 2048), ("previous", 1024)):
        values = next(x["values"] for x in m["gh"] if x["order"] == order)
        a[name] = {s: values[str(float(s))]["alpha"] for s in (1, 2, 6)}
    a["quad"] = {s: m["quad"][str(float(s))]["alpha"]["value"] for s in (1, 2, 6)}
    stored = read(OUT / "NOISE_CONSTANTS.json")
    checks = []
    for name, values in a.items():
        tau = 2 * values[1]
        variance = 1 - .9 * ((values[6] + values[2]) / 2) ** 2
        row = next(r for r in stored["rows"] if r["source"] == name)
        differences = {
            "alpha": max(abs(values[s] - row["alpha"][str(s)]) for s in (1, 2, 6)),
            "tau2": abs(tau - row["emission_noise_variance_tau2"]),
            "negative_variance": abs(variance - row["negative_teacher_limit_variance"]),
            "gap": abs(tau - variance - row["tau2_minus_teacher_variance"])}
        checks.append({"source": name, "differences": differences,
                       "tau2": tau, "negative_teacher_variance": variance,
                       "passed": max(differences.values()) <= 1e-14
                       and row["negative_teacher_below_emission_noise"] == (variance < tau)})
    return a["primary"], {"rows": checks, "passed": all(r["passed"] for r in checks),
                          "new_alpha_integrals": 0, "strict_interval_certificate": False}


def params(d, alpha):
    base = math.sqrt(.9 / (d - 1))
    matched = [base * alpha[s] / alpha[1] for s in (6, 2)]
    return [
        {"name": "teacher", "scales": [6., 2.], "lambda": [base, base]},
        {"name": "fixedRfloor", "scales": [1., 1.], "lambda": [base, base]},
        {"name": "comp", "scales": [1., 1.], "lambda": matched},
        {"name": "half", "scales": [1., 1.], "lambda": [v / 2 for v in matched]},
        {"name": "over", "scales": [1., 1.], "lambda": [3 * v / 2 for v in matched]},
        {"name": "independent", "scales": [1., 1.], "lambda": [0., 0.]}]


def spectrum_check(d, row, original, cfg):
    m = d // 2
    s, l = np.asarray(row["scales"]), np.asarray(row["lambda"])
    residual = s * s * (1 - l * l)
    factor = s * l
    vector = np.repeat(factor, m)
    matrix = np.diag(np.repeat(residual, m)) + vector[:, None] * vector[None, :]
    exact = eigh(matrix, eigvals_only=True, driver="evr")
    aa, bb = residual + m * factor * factor
    cross = m * factor[0] * factor[1]
    radius = math.hypot(aa - bb, 2 * cross)
    roots = [(aa + bb - radius) / 2, (aa + bb + radius) / 2]
    formula = np.sort(np.r_[np.repeat(residual, m - 1), roots])
    formula_error = float(abs(exact - formula).max())
    stored_error = float(abs(exact - np.asarray(original["eigenvalues"])).max())
    diagonal_error = float(abs(np.diag(matrix) - np.repeat(s * s, m)).max())
    return {"name": row["name"], "eigenvalues": exact.tolist(),
            "minimum": float(exact[0]), "maximum": float(exact[-1]),
            "trace": float(np.trace(matrix)), "formula_max_error": formula_error,
            "stored_eigenvalue_max_error": stored_error, "diagonal_max_error": diagonal_error,
            "passed": bool(formula_error <= cfg["spectrum_gate"] and stored_error <= cfg["spectrum_gate"]
            and diagonal_error <= cfg["symmetry_gate"]
            and exact[0] >= cfg["spectral_lower"] and exact[-1] <= cfg["spectral_upper"])}


def events(pmf, d):
    n, k = np.indices((d // 2 + 1, d // 2 + 1))
    spin = 2 * (n + k) - d
    masks = [spin * spin <= d, (spin > 0) & (spin * spin > d),
             (spin < 0) & (spin * spin > d)]
    return np.array([[math.fsum(p[mask].tolist()) for mask in masks] for p in pmf])


def cross_metrics(variants, d):
    ev = np.asarray([events(p, d) for p in variants])
    count = len(variants)
    risk = np.empty((count, count, 6))
    br, kl = np.empty((count, count, 6, 3)), np.empty((count, count, 6, 3))
    for a in range(count):
        for b in range(count):
            for j in range(6):
                risk[a, b, j] = stable_kl(variants[a, 0], variants[b, j])
                for e in range(3):
                    p, q = float(ev[a, 0, e]), float(ev[b, j, e])
                    br[a, b, j, e] = (p - q) ** 2
                    kl[a, b, j, e] = stable_kl([p, 1 - p], [q, 1 - q])
    return risk, ev, br, kl


def dimension(d, cfg, alpha):
    record = read(OUT / f"REFERENCE_d{d}.json")
    pp = params(d, alpha)
    require(record["names"] == list(NAMES), "Model order")
    parameter_error = 0.
    for a, b in zip(pp, record["params"]):
        require(a["name"] == b["name"] and b["mean"] == [0., 0.], "Model mean/name")
        for key in ("scales", "lambda"):
            parameter_error = max(parameter_error, float(abs(np.array(a[key]) - b[key]).max()))
    require(parameter_error <= 1e-15, "Old-alpha parameter reconstruction")
    arrays = ROOT / record["variant_array"]
    require(sha(arrays) == record["variant_array_sha256"], "Variant record hash")
    with np.load(arrays, allow_pickle=False) as f:
        variants, masses = f["pmf"].copy(), f["raw_masses"].copy()
    m = d // 2
    require(variants.shape == (4, 6, m + 1, m + 1) and masses.shape == (4, 6), "Variant shapes")
    require(np.all(variants > 0) and np.isfinite(variants).all(), "Variant full support")
    WORK["saved_variant_PMFS_reanalyzed"] += 24
    spectra = [spectrum_check(d, r, old, cfg) for r, old in zip(pp, record["spectra"])]
    cross, ev, br, binary = cross_metrics(variants, d)
    primary = cross[2, 2]
    spans = np.ptp(cross, axis=(0, 1))
    original_checks = {
        "primary_matches_public": np.array_equal(variants[2], np.asarray(record["pmf"])),
        "raw_mass": float(abs(masses - 1).max()) <= cfg["mass_gate"],
        "normalized_mass": max(abs(math.fsum(p.ravel().tolist()) - 1) for v in variants for p in v)
                           <= cfg["mass_gate"],
        "complement": float(abs(variants - variants[:, :, ::-1, ::-1]).max()) <= cfg["symmetry_gate"],
        "all_spectra": all(x["passed"] for x in spectra),
        "general_KL_span": float(spans[1:].max()) <= cfg["general_KL_span_gate"],
        "comp_positive_and_span": primary[2] > 0 and spans[2] <= cfg["comp_KL_span_gate"]
                                  and spans[2] / primary[2] <= cfg["comp_KL_relative_span_gate"],
        "event_probability_span": float(np.ptp(ev, axis=0).max()) <= cfg["event_sensitivity_gate"],
        "event_risk_span": max(float(np.ptp(br, axis=(0, 1)).max()),
                               float(np.ptp(binary, axis=(0, 1)).max())) <= cfg["event_sensitivity_gate"]}
    l1 = max(float(abs(variants[a] - variants[b]).sum(axis=(1, 2)).max())
             for a, b in itertools.combinations(range(4), 2))
    logerr = max(float(abs(np.log(variants[a]) - np.log(variants[b])).max())
                 for a, b in itertools.combinations(range(4), 2))
    original_checks["variant_L1"] = l1 <= cfg["PMF_L1_gate"]
    original_checks["variant_logcell"] = logerr <= cfg["log_PMF_gate"]
    choose = np.array([math.log(math.comb(m, j)) for j in range(m + 1)])
    patterns = np.log(variants[2]) - choose[:, None] - choose[None, :]
    original_checks["log_pattern"] = float(abs(patterns - np.asarray(record["log_pattern"])).max()) <= 1e-12
    saved_errors = {
        "risk_primary": float(abs(primary - record["risk_to_teacher"]).max()),
        "risk_cross": float(abs(cross - np.asarray(record["risk_cross_variants"])).max()),
        "event_probability": float(abs(ev[2] - np.asarray(record["event_probability"])).max()),
        "event_Brier": float(abs(br[2, 2] - np.asarray(record["event_brier"])).max()),
        "event_KL": float(abs(binary[2, 2] - np.asarray(record["event_logloss"])).max())}
    original_checks["saved_metric_values"] = max(saved_errors.values()) <= 1e-12
    original_checks["saved_comp_metric_relative"] = abs(primary[2] - record["risk_to_teacher"][2]) / primary[2] <= 1e-6
    decimal_rows = []
    for j in range(1, 6):
        digits, value = decimal_direct_kl(variants[2, 0], variants[2, j])
        difference = abs(value - primary[j])
        original_decimal = next(x for x in record["precision"]["Decimal80_audits"] if x["name"] == NAMES[j])
        decimal_rows.append({"name": NAMES[j], "Decimal80_direct_normalized": digits,
                             "difference_from_symmetric_series": difference,
                             "relative_difference": difference / value,
                             "difference_from_saved_Decimal80": abs(value - original_decimal["float_value"]),
                             "passed": difference <= cfg["decimal_KL_absolute_gate"]
                             and (j != 2 or difference / value <= cfg["decimal_comp_relative_gate"])})
    original_checks["Decimal80"] = all(x["passed"] for x in decimal_rows)
    outer, weights = normal_rule(OUTER)
    unique = sorted({(s, l) for row in pp for s, l in zip(row["scales"], row["lambda"])})
    component, kernel_errors = {}, []
    quad_rows = read(OUT / f"QUAD_d{d}.json")
    quad_checks = []
    for scale, loading in unique:
        h, q = kernel(outer, scale, loading, "pmf")
        component[(scale, loading)] = binomial(h, q, m)
        kernel_errors.append(float(abs(h + q - 1).max()))
        rows = [x for x in quad_rows if x["scale"] == scale and abs(x["loading"] - loading) <= 1e-15]
        require(len(rows) == 9 and [x["common_z"] for x in rows] == cfg["quad_z"], "Quad rows")
        hh, qq = kernel(cfg["quad_z"], scale, loading, "quad")
        for i, old in enumerate(rows):
            error = max(abs(math.log(hh[i]) - math.log(old["probabilities"][0])),
                        abs(math.log(qq[i]) - math.log(old["probabilities"][1])))
            quad_checks.append({"scale": scale, "loading": loading, "z": old["common_z"],
                                "independent_log_difference": error, "original_passed": old["passed"],
                                "passed": error <= cfg["quad_log_gate"] and old["passed"]})
    independent, raw_masses = [], []
    for row in pp:
        a, b = (component[k] for k in zip(row["scales"], row["lambda"]))
        raw = np.einsum("k,ka,kb->ab", weights, a, b, optimize=False)
        require(np.all(raw > 0) and np.isfinite(raw).all(), "Independent complete PMF")
        mass = math.fsum(raw.ravel().tolist())
        independent.append(raw / mass)
        raw_masses.append(mass)
    independent = np.asarray(independent)
    WORK["independent_PMF_matrix_integrals"] += 6
    WORK["outer_scalar_integrand_evaluations"] += 6 * (m + 1) ** 2 * OUTER
    rr, ee, bb, kk = cross_metrics(independent[None], d)
    fresh = rr[0, 0]
    fresh_decimal, decimal_value = decimal_direct_kl(independent[0], independent[2])
    primary_comp_difference = abs(decimal_value - primary[2])
    candidate_comparisons = []
    for j in range(6):
        candidate_comparisons.append({
            "name": NAMES[j], "raw_mass": raw_masses[j],
            "minimum_cell": float(independent[j].min()),
            "L1_against_saved_variants": [float(abs(independent[j] - variants[v, j]).sum()) for v in range(4)],
            "max_logcell_against_saved_variants": [float(abs(np.log(independent[j]) - np.log(variants[v, j])).max()) for v in range(4)],
            "teacher_to_candidate_KL": float(fresh[j]),
            "KL_difference_from_primary": abs(float(fresh[j]) - float(primary[j])),
            "event_probability_difference": float(abs(ee[0, j] - ev[2, j]).max()),
            "event_Brier_difference": float(abs(bb[0, 0, j] - br[2, 2, j]).max()),
            "event_KL_difference": float(abs(kk[0, 0, j] - binary[2, 2, j]).max())})
    fresh_checks = {
        "private_probability_normalization": max(kernel_errors) <= cfg["symmetry_gate"],
        "all_quad_locations": all(x["passed"] for x in quad_checks),
        "raw_masses": max(abs(x - 1) for x in raw_masses) <= cfg["mass_gate"],
        "all_PMF_L1": max(max(x["L1_against_saved_variants"]) for x in candidate_comparisons) <= cfg["PMF_L1_gate"],
        "all_logcells": max(max(x["max_logcell_against_saved_variants"]) for x in candidate_comparisons) <= cfg["log_PMF_gate"],
        "all_non_teacher_KL": max(x["KL_difference_from_primary"] for x in candidate_comparisons[1:]) <= cfg["general_KL_span_gate"],
        "comp_tiny_KL": decimal_value > 0 and primary_comp_difference <= cfg["comp_KL_span_gate"]
                       and primary_comp_difference / decimal_value <= cfg["comp_KL_relative_span_gate"],
        "fresh_comp_Decimal80": abs(decimal_value - fresh[2]) <= cfg["decimal_KL_absolute_gate"]
                              and abs(decimal_value - fresh[2]) / decimal_value <= cfg["decimal_comp_relative_gate"],
        "all_events": max(x[k] for x in candidate_comparisons for k in (
            "event_probability_difference", "event_Brier_difference", "event_KL_difference")) <= cfg["event_sensitivity_gate"]}
    old_anchor = None
    if d in (16, 64):
        old = read(OLD_REF / f"REFERENCE_d{d}.json")
        teacher = old["parameters"].index([6., 2.])
        p = np.asarray(old["pmf"])
        oldrisk = stable_kl(p[teacher], p[0])
        old_anchor = {"teacher_L1": float(abs(independent[0] - p[teacher]).sum()),
                      "fixed_floor_L1": float(abs(independent[1] - p[0]).sum()),
                      "floor_KL_difference": abs(float(fresh[1]) - oldrisk)}
        fresh_checks["old_anchor"] = max(old_anchor["teacher_L1"], old_anchor["fixed_floor_L1"]) <= cfg["old_reference_L1_gate"] and old_anchor["floor_KL_difference"] <= cfg["old_floor_KL_gate"]
    return {"d": d, "passed": bool(all(original_checks.values()) and all(fresh_checks.values())),
            "original_checks": {k: bool(v) for k, v in original_checks.items()},
            "independent_checks": {k: bool(v) for k, v in fresh_checks.items()},
            "parameter_max_error": parameter_error, "spectra": spectra,
            "saved_metric_max_errors": saved_errors, "original_Decimal80_rechecks": decimal_rows,
            "original_variant_L1": l1, "original_variant_logcell": logerr,
            "original_KL_cross_variant_spans": spans.tolist(),
            "quad_location_checks": quad_checks, "candidate_comparisons": candidate_comparisons,
            "fresh_comp_Decimal80_normalized_KL": fresh_decimal,
            "fresh_comp_KL_relative_difference_from_primary": primary_comp_difference / decimal_value,
            "fresh_comp_Decimal80_relative_function_error": abs(decimal_value - fresh[2]) / decimal_value,
            "old_anchor": old_anchor, "independent_raw_masses": raw_masses,
            "independent_pmf": independent.tolist(), "independent_event_probabilities": ee[0].tolist(),
            "independent_event_Brier": bb[0, 0].tolist(), "independent_event_KL": kk[0, 0].tolist(),
            "KL_direction": "D(P_teacher || Q_candidate)",
            "scope": "72 stored variants reanalyzed across dimensions; one 768/2048 reconstruction per scientific candidate, not rerunning old four quadrature variants"}


def write_audit():
    require(not RECEIPT.exists(), "No silent audit rerun")
    public, local = sources(True)
    cfg = read(OUT / "CONFIG.json")
    require(cfg["dimensions"] == [16, 64, 256] and cfg["names"] == list(NAMES), "Configuration")
    data = {"status": "running", "passed": False, "started_utc": now(),
            "audit_source_sha256": sha(Path(__file__)), "audit_plan_prefix_sha256": plan_hash(),
            "plan_hash_convention": "UTF8/LF prefix through NUMERIC_AUDIT_PLAN_END plus newline",
            "frozen_public_files": public, "frozen_local_arrays": local,
            "private_arrays_audited": True, "outer_GH": OUTER, "private_GH": INNER,
            "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                            "platform": platform.platform(), "threads": 1},
            "dimensions": [], "work": dict(WORK)}
    save(data, initial=True)
    started = time.perf_counter()
    try:
        alpha, constants = alpha_sources()
        data["old_alpha_and_noise_constants"] = constants
        require(constants["passed"], "Noise constant arithmetic")
        # Deterministic algebra check, not a new model or stochastic calculation.
        require(stable_kl([.5, .5], [.5, .5]) == 0, "Identical PMF KL")
        for d in (16, 64, 256):
            row = dimension(d, cfg, alpha)
            data["dimensions"].append(row)
            data["work"] = dict(WORK)
            save(data)
            print("NUMERIC AUDIT", d, row["passed"], row["fresh_comp_Decimal80_normalized_KL"], flush=True)
        data["passed"] = all(x["passed"] for x in data["dimensions"]) and constants["passed"]
        data["status"] = "complete" if data["passed"] else "completed_with_retained_failures"
    except Exception as exc:
        data["status"] = "stopped_with_retained_failure"
        data["failure"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    data["finished_utc"] = now()
    data["elapsed_seconds"] = time.perf_counter() - started
    data["GH_rules"] = RULE_META
    data["work"] = dict(WORK)
    data["work"]["total_private_node_pairs"] = WORK["private_node_pairs_pmf"] + WORK["private_node_pairs_quad"]
    data["strict_integration_error_certificate"] = False
    data["stable_KL_method"] = "u=(p-q)/(p+q), atanh correction through u^33 for |u|<=.05; direct generalized KL outside; math.fsum; Decimal80 direct KL after exact-decimal normalization"
    save(data)
    print(json.dumps({"passed": data["passed"], "status": data["status"],
                      "elapsed_seconds": data["elapsed_seconds"], "work": data["work"]}), flush=True)
    if not data["passed"]:
        raise SystemExit(1)


def check(workspace):
    audit = read(RECEIPT)
    require(audit["passed"] and audit["status"] == "complete", "Audit did not pass")
    require(audit["audit_source_sha256"] == sha(Path(__file__)), "Audit source changed")
    require(audit["audit_plan_prefix_sha256"] == plan_hash(), "Audit plan changed")
    for name, digest in audit["frozen_public_files"].items():
        require(sha(ROOT / name) == digest, "Public source/result changed: " + name)
    if workspace:
        for name, digest in audit["frozen_local_arrays"].items():
            require(sha(ROOT / name) == digest, "Local variants changed: " + name)
    print(json.dumps({"passed": True, "workspace": workspace, "integrals_reexecuted": False,
                      "local_arrays_were_audited_in_write": audit["private_arrays_audited"],
                      "receipt_sha256": sha(RECEIPT)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("write", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    write_audit() if args.phase == "write" else check(args.workspace)
