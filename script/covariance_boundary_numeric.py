"""Deterministic covariance-compensation PMFs and emission-noise constants.

No RNG and no fitting. All selected Gaussian integrations run only after freeze.
External sandbox code/results are not used as numerical inputs.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from functools import lru_cache
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
from scipy.integrate import quad
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, roots_legendre, ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/covariance_boundary/NUMERIC"
DOC = ROOT/"docs/08_research_研究工作/相关结构补偿与发射噪声_20261001"
OLD_N = ROOT/"artifacts/objective_strengthening/N"
OLD_REF = ROOT/"artifacts/finite_learning/REF"
BASE = "6fcb333f7a0be02f568bbaccf38d28af2de407fd"
HANDOFF = "ebd22608c6c192bef4596e5cab5fd3ef9aa62f6120d39e9d1f83c9575291d8ee"
NAMES = ("teacher", "fixedRfloor", "comp", "half", "over", "independent")
VARIANTS = (
    {"private_order": 512, "outer_kind": "GH", "outer_order": 256},
    {"private_order": 1024, "outer_kind": "GH", "outer_order": 512},
    {"private_order": 2048, "outer_kind": "GH", "outer_order": 1024},
    {"private_order": 2048, "outer_kind": "GL", "outer_order": 1024, "outer_radius": 16.},
)
CFG = {
    "dimensions": [16, 64, 256], "c": .9, "truth": [6., 2.], "a": 1.,
    "names": list(NAMES), "variants": list(VARIANTS), "primary_variant": 2,
    "spectral_lower": .1, "spectral_upper": 121.6,
    "mass_gate": 1e-12, "PMF_L1_gate": 1e-8, "log_PMF_gate": 1e-6,
    "symmetry_gate": 1e-12, "spectrum_gate": 1e-10,
    "general_KL_span_gate": 1e-10, "comp_KL_span_gate": 1e-12,
    "comp_KL_relative_span_gate": 1e-3,
    "event_sensitivity_gate": 1e-10,
    "decimal_digits": 80, "decimal_KL_absolute_gate": 1e-14,
    "decimal_comp_relative_gate": 1e-6,
    "private_chunk": 128, "log1p_series_threshold": .01, "log1p_series_degree": 18,
    "quad_z": [-12., -8., -4., -1., 0., 1., 4., 8., 12.],
    "quad_interval": [-12., 12.], "quad_epsabs": 1e-13,
    "quad_epsrel": 1e-13, "quad_limit": 400, "quad_log_gate": 1e-9,
    "old_reference_L1_gate": 1e-10, "old_floor_KL_gate": 1e-10,
    "alpha_source": "old N qualified GH2048 primary; old GH1024/quad sensitivity retained",
    "new_observations": 0, "new_fits": 0, "external_sandbox_files_used": False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, default=serial, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT/name) == digest, "input hash mismatch: "+name)


def input_paths():
    return [
        OLD_N/"FREEZE.json", OLD_N/"RESULTS_FREEZE.json", OLD_N/"MOMENTS.json",
        OLD_N/"LIMITS.json", OLD_REF/"RESULTS_FREEZE.json",
        OLD_REF/"REFERENCE_d16.json", OLD_REF/"REFERENCE_d64.json",
    ]


def old_inputs():
    nseal, refseal = read(OLD_N/"RESULTS_FREEZE.json"), read(OLD_REF/"RESULTS_FREEZE.json")
    require(nseal["source_freeze_sha256"] == sha(OLD_N/"FREEZE.json"), "old N seal chain")
    for path in (OLD_N/"MOMENTS.json", OLD_N/"LIMITS.json"):
        require(sha(path) == nseal["files"][rel(path)], "old N input byte identity")
    for d in (16, 64):
        path = OLD_REF/f"REFERENCE_d{d}.json"
        require(sha(path) == refseal["files"][rel(path)], "old REF input byte identity")
    moments = read(OLD_N/"MOMENTS.json")
    require(moments["qualified"] and not moments["interval_certified"], "old moment qualification")
    require(moments["gh"][-1]["order"] == 2048, "old primary alpha order")
    return moments


def alpha_values(moments, mode="primary"):
    if mode == "primary":
        row = moments["gh"][-1]["values"]
        return {s: float(row[str(float(s))]["alpha"]) for s in (1, 2, 6)}
    if mode == "previous":
        row = next(r["values"] for r in moments["gh"] if r["order"] == 1024)
        return {s: float(row[str(float(s))]["alpha"]) for s in (1, 2, 6)}
    return {s: float(moments["quad"][str(float(s))]["alpha"]["value"]) for s in (1, 2, 6)}


def parameters(d, alpha):
    rho = CFG["c"]/(d-1)
    shared = math.sqrt(rho)
    matched = [shared*alpha[s]/alpha[1] for s in (6, 2)]
    models = [
        {"name": "teacher", "scales": [6., 2.], "lambda": [shared, shared]},
        {"name": "fixedRfloor", "scales": [1., 1.], "lambda": [shared, shared]},
        {"name": "comp", "scales": [1., 1.], "lambda": matched},
        {"name": "half", "scales": [1., 1.], "lambda": [x*.5 for x in matched]},
        {"name": "over", "scales": [1., 1.], "lambda": [x*1.5 for x in matched]},
        {"name": "independent", "scales": [1., 1.], "lambda": [0., 0.]},
    ]
    for row in models:
        require(all(0 <= value < 1 for value in row["lambda"]), "invalid factor loading")
        row["mean"] = [0., 0.]
        row["single_Gaussian"] = True
        row["loading_uses_teacher_information"] = row["name"] in ("comp", "half", "over")
    return models


def spectrum(d, row):
    m = d//2
    scales, loadings = np.asarray(row["scales"]), np.asarray(row["lambda"])
    variances = scales**2*(1-loadings**2)
    factors = scales*loadings
    diagonal = np.repeat(variances, m)
    vector = np.repeat(factors, m)
    covariance = np.diag(diagonal)+np.outer(vector, vector)
    actual = np.linalg.eigvalsh(covariance)
    quotient = np.diag(variances)+m*np.outer(factors, factors)
    analytic = np.sort(np.r_[np.repeat(variances[0], m-1),
                             np.repeat(variances[1], m-1), np.linalg.eigvalsh(quotient)])
    error = float(abs(actual-analytic).max())
    diagonal_error = float(abs(np.diag(covariance)-np.repeat(scales**2, m)).max())
    require(error <= CFG["spectrum_gate"] and diagonal_error <= CFG["symmetry_gate"], "covariance algebra gate")
    require(actual[0] >= CFG["spectral_lower"] and actual[-1] <= CFG["spectral_upper"], "declared spectral class")
    return {
        "eigenvalues": actual.tolist(), "minimum": float(actual[0]), "maximum": float(actual[-1]),
        "condition_number": float(actual[-1]/actual[0]), "trace": float(np.trace(covariance)),
        "group_perpendicular_eigenvalues": variances.tolist(),
        "group_perpendicular_multiplicities": [m-1, m-1], "two_group_quotient": quotient.tolist(),
        "analytic_eigenvalues": analytic.tolist(), "spectral_formula_max_error": error,
        "marginal_variance_max_error": diagonal_error,
        "within_nu_Lambda_class": True,
    }


@lru_cache(None)
def gh(order):
    z, w = roots_hermitenorm(order)
    w = w/math.sqrt(2*math.pi)
    require(abs(float(w.sum())-1) <= 1e-12 and np.all(w >= 0), "GH probability weights")
    logw = np.full_like(w, -np.inf)
    positive = w > 0
    logw[positive] = np.log(w[positive])
    return z, w, logw


def outer_rule(variant):
    if variant["outer_kind"] == "GH":
        z, w, _ = gh(variant["outer_order"])
        return z, w
    z, w = roots_legendre(variant["outer_order"])
    radius = variant["outer_radius"]
    z = radius*z
    w = radius*w*np.exp(-z*z/2)/math.sqrt(2*math.pi)
    return z, w


def kernel_logs(common, scale, loading, order):
    common = np.asarray(common, dtype=float).ravel()
    if loading == 0:
        return np.full(len(common), -math.log(2)), np.full(len(common), -math.log(2))
    nodes, _, logw = gh(order)
    a, b = scale*loading, scale*math.sqrt(1-loading*loading)
    positive, negative = np.empty(len(common)), np.empty(len(common))
    for start in range(0, len(common), CFG["private_chunk"]):
        z = common[start:start+CFG["private_chunk"]]
        eta = a*z[:, None]+b*nodes
        positive[start:start+len(z)] = logsumexp(log_expit(eta)+logw, axis=1)
        negative[start:start+len(z)] = logsumexp(log_expit(-eta)+logw, axis=1)
    require(np.isfinite(positive).all() and np.isfinite(negative).all(), "nonfinite private log probability")
    require(float(abs(np.logaddexp(positive, negative)).max()) <= CFG["symmetry_gate"], "private normalization")
    return positive, negative


def count_geometry(d):
    m = d//2
    n = np.arange(m+1)
    logchoose = np.asarray([math.log(math.comb(m, int(k))) for k in n])
    a, b = np.meshgrid(n, n, indexing="ij")
    return a, b, logchoose[:, None]+logchoose[None, :], logchoose


def pmf_variant(d, params, variant):
    m = d//2
    z, weights = outer_rule(variant)
    logchoose = count_geometry(d)[3][:, None]
    counts = np.arange(m+1)[:, None]
    cache = {}
    pmfs, masses = [], []
    for row in params:
        pieces = []
        for scale, loading in zip(row["scales"], row["lambda"]):
            key = (scale, loading)
            if key not in cache:
                lp, lq = kernel_logs(z, scale, loading, variant["private_order"])
                cache[key] = np.exp(logchoose+counts*lp+(m-counts)*lq)
            pieces.append(cache[key])
        raw = (pieces[0]*weights[None, :]) @ pieces[1].T
        require(np.isfinite(raw).all() and np.all(raw > 0), "complete PMF positivity")
        mass = float(np.sum(raw, dtype=np.longdouble))
        pmfs.append(np.asarray(np.asarray(raw, dtype=np.longdouble)/np.longdouble(mass), dtype=np.float64))
        masses.append(mass)
    return np.asarray(pmfs), masses


def stable_generalized_kl(p, q):
    """Normalized-PMF generalized KL with a stable small-relative-error series."""
    p = np.asarray(p, dtype=np.longdouble).ravel()
    q = np.asarray(q, dtype=np.longdouble).ravel()
    require(np.all(p > 0) and np.all(q > 0), "KL requires full-support vectors")
    p = p/np.sum(p, dtype=np.longdouble)
    q = q/np.sum(q, dtype=np.longdouble)
    delta = (p-q)/q
    small = np.abs(delta) <= CFG["log1p_series_threshold"]
    terms = np.empty_like(p)
    x = delta[small]
    poly = np.full(len(x), ((-1)**CFG["log1p_series_degree"])
                   / (CFG["log1p_series_degree"]*(CFG["log1p_series_degree"]-1)), dtype=np.longdouble)
    for degree in range(CFG["log1p_series_degree"]-1, 1, -1):
        poly = np.longdouble((-1)**degree)/np.longdouble(degree*(degree-1))+x*poly
    terms[small] = q[small]*x*x*poly
    # Away from zero no dangerous near-equal subtraction; avoids log1p(-1)
    # when p/q is positive but delta rounds to -1 in a far-tail cell.
    terms[~small] = p[~small]*(np.log(p[~small])-np.log(q[~small]))-p[~small]+q[~small]
    require(np.isfinite(terms).all() and np.all(terms >= 0), "negative generalized-KL term")
    return float(np.sum(terms, dtype=np.longdouble))


def decimal_kl(p, q):
    """80-digit evaluation of the existing floating PMFs, not better quadrature."""
    with localcontext() as context:
        context.prec = CFG["decimal_digits"]
        pp = [Decimal.from_float(float(x)) for x in np.asarray(p).ravel()]
        qq = [Decimal.from_float(float(x)) for x in np.asarray(q).ravel()]
        ps, qs = sum(pp), sum(qq)
        total = Decimal(0)
        for a, b in zip(pp, qq):
            a, b = a/ps, b/qs
            total += a*(a/b).ln()-a+b
        return str(total), float(total)


def event_probabilities(pmf, d):
    a, b, _, _ = count_geometry(d)
    spin = 2*(a+b)-d
    masks = (spin*spin <= d, spin > math.sqrt(d), spin < -math.sqrt(d))
    return np.asarray([[float(np.sum(p[mask], dtype=np.longdouble)) for mask in masks] for p in pmf])


def scalar_quad_audit(d, params):
    rows = []
    unique = sorted({(s, l) for row in params for s, l in zip(row["scales"], row["lambda"])})
    for scale, loading in unique:
        lp, lq = kernel_logs(CFG["quad_z"], scale, loading, 2048)
        a, b = scale*loading, scale*math.sqrt(1-loading*loading)
        for i, z in enumerate(CFG["quad_z"]):
            vals, errors, evals = [], [], []
            for sign in (1, -1):
                def integrand(e):
                    return (math.exp(-e*e/2)/math.sqrt(2*math.pi)
                            *float(expit(sign*(a*z+b*e))))
                answer = quad(integrand, *CFG["quad_interval"], epsabs=CFG["quad_epsabs"],
                              epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"], full_output=1)
                require(len(answer) == 3, "alternative private quad convergence")
                vals.append(float(answer[0])); errors.append(float(answer[1]))
                evals.append(int(answer[2]["neval"]))
            require(min(vals) > 0, "alternative private probabilities")
            error = max(abs(math.log(vals[0])-lp[i]), abs(math.log(vals[1])-lq[i]))
            rows.append({"d": d, "scale": scale, "loading": loading, "common_z": z,
                         "probabilities": vals, "reported_errors": errors, "evaluations": evals,
                         "log_difference": error, "passed": error <= CFG["quad_log_gate"],
                         "inner_normal_tail_mass_bound": 2*float(ndtr(-12.))})
    return rows


def noise_constants(moments):
    rows = []
    for name in ("primary", "previous", "quad"):
        alpha = alpha_values(moments, name)
        average = (alpha[6]+alpha[2])/2
        tau2 = 2*alpha[1]/CFG["a"]
        teacher = 1-CFG["c"]*average**2
        rows.append({"source": name, "alpha": {str(k): v for k, v in alpha.items()},
                     "emission_noise_variance_tau2": tau2,
                     "negative_teacher_limit_variance": teacher,
                     "tau2_minus_teacher_variance": tau2-teacher,
                     "negative_teacher_below_emission_noise": teacher < tau2})
    return {
        "rows": rows, "primary": rows[0],
        "old_moments_qualified": moments["qualified"],
        "strict_interval_certificate": False, "new_scalar_alpha_integrals": 0,
        "not_a_general_covariance_theorem": True,
        "tau2_identity": "E sech^2(aZ/2) = 2*alpha(a)/a; do not square this value again",
    }


def one_dimension(d, alpha):
    started = time.perf_counter()
    params = parameters(d, alpha)
    specs = [spectrum(d, row) for row in params]
    variants, masses = [], []
    for variant in VARIANTS:
        values, raw = pmf_variant(d, params, variant)
        variants.append(values); masses.append(raw)
    variants = np.asarray(variants)
    main = variants[CFG["primary_variant"]]
    array = OUT/"local"/f"PMF_variants_d{d}.npz"
    np.savez_compressed(array, pmf=variants, raw_masses=np.asarray(masses))
    quad_rows = scalar_quad_audit(d, params)
    write(OUT/f"QUAD_d{d}.json", quad_rows)
    cross_risks = np.empty((4, 4, 6))
    eventp = np.asarray([event_probabilities(p, d) for p in variants])
    cross_brier = np.empty((4, 4, 6, 3))
    cross_binary_kl = np.empty_like(cross_brier)
    for a in range(4):
        for b in range(4):
            for model in range(6):
                cross_risks[a, b, model] = stable_generalized_kl(variants[a, 0], variants[b, model])
                for event in range(3):
                    p, q = eventp[a, 0, event], eventp[b, model, event]
                    cross_brier[a, b, model, event] = (p-q)**2
                    cross_binary_kl[a, b, model, event] = stable_generalized_kl([p, 1-p], [q, 1-q])
    primary = cross_risks[2, 2]
    spans = np.ptp(cross_risks, axis=(0, 1))
    require(primary[2] > 0, "compensated KL numerically unresolved")
    relative_span = float(spans[2]/primary[2])
    decimal_rows = []
    for model in range(1, 6):
        digits, value = decimal_kl(main[0], main[model])
        error = abs(value-primary[model])
        decimal_rows.append({
            "name": NAMES[model], "Decimal80_normalized_KL": digits,
            "float_value": value, "stable_primary_KL": float(primary[model]),
            "absolute_difference": error,
            "relative_difference": error/primary[model] if primary[model] else 0.0,
            "passed": bool(error <= CFG["decimal_KL_absolute_gate"]
                           and (model != 2 or error/primary[model] <= CFG["decimal_comp_relative_gate"])),
        })
    mass_error = float(np.max(abs(np.asarray(masses)-1)))
    l1 = max(float(abs(variants[a]-variants[b]).sum(axis=(1, 2)).max())
             for a, b in itertools.combinations(range(4), 2))
    log_error = max(float(abs(np.log(variants[a])-np.log(variants[b])).max())
                    for a, b in itertools.combinations(range(4), 2))
    complement = float(abs(variants-variants[:, :, ::-1, ::-1]).max())
    probability_span = float(np.ptp(eventp, axis=0).max())
    event_risk_span = max(float(np.ptp(cross_brier, axis=(0, 1)).max()),
                          float(np.ptp(cross_binary_kl, axis=(0, 1)).max()))
    old_anchor = None
    if d in (16, 64):
        old = read(OLD_REF/f"REFERENCE_d{d}.json")
        ii = old["parameters"].index([6., 2.])
        oldp = np.asarray(old["pmf"])
        old_kl = stable_generalized_kl(oldp[ii], oldp[0])
        old_anchor = {"teacher_L1": float(abs(main[0]-oldp[ii]).sum()),
                      "fixed_floor_L1": float(abs(main[1]-oldp[0]).sum()),
                      "old_floor_stable_KL": old_kl,
                      "floor_KL_difference": abs(old_kl-primary[1])}
    qualified = (
        mass_error <= CFG["mass_gate"] and l1 <= CFG["PMF_L1_gate"]
        and log_error <= CFG["log_PMF_gate"] and complement <= CFG["symmetry_gate"]
        and float(spans[1:].max()) <= CFG["general_KL_span_gate"]
        and spans[2] <= CFG["comp_KL_span_gate"]
        and relative_span <= CFG["comp_KL_relative_span_gate"]
        and max(probability_span, event_risk_span) <= CFG["event_sensitivity_gate"]
        and all(r["passed"] for r in decimal_rows) and all(r["passed"] for r in quad_rows)
    )
    if old_anchor:
        qualified = qualified and (
            max(old_anchor["teacher_L1"], old_anchor["fixed_floor_L1"]) <= CFG["old_reference_L1_gate"]
            and old_anchor["floor_KL_difference"] <= CFG["old_floor_KL_gate"])
    precision = {
        "mass_error": mass_error, "PMF_L1_variant_difference": l1,
        "log_PMF_variant_difference": log_error, "complement_symmetry_error": complement,
        "KL_cross_variant_spans": spans.tolist(), "comp_KL_relative_span": relative_span,
        "event_probability_span": probability_span, "event_risk_span": event_risk_span,
        "quad_max_log_difference": max(r["log_difference"] for r in quad_rows),
        "Decimal80_audits": decimal_rows, "old_fixed_model_anchor": old_anchor,
        "outer_GL_truncation_mass_bound": 2*float(ndtr(-16.)),
        "qualified": bool(qualified), "strict_integration_error_certificate": False,
    }
    record = {
        "d": d, "m": d//2, "names": list(NAMES), "params": params,
        "pmf": main, "log_pattern": np.log(main)-count_geometry(d)[2],
        "count_order": "row nA, column nB; full pattern probability is pmf divided by both binomial multiplicities",
        "risk_to_teacher": primary, "risk_cross_variants": cross_risks,
        "spectra": specs, "event_names": ["central", "upper_tail", "lower_tail"],
        "event_probability": eventp[2], "event_brier": cross_brier[2, 2],
        "event_logloss": cross_binary_kl[2, 2], "precision": precision,
        "variant_array": rel(array), "variant_array_sha256": sha(array),
        "elapsed_seconds": time.perf_counter()-started,
        "renormalization": "record raw mass, normalize positive PMF; no clipping",
        "oracle_compensation_not_a_trained_model": True,
    }
    write(OUT/f"REFERENCE_d{d}.json", record)
    require(qualified, "numeric qualification failed at d="+str(d))
    print("NUMERIC", d, "comp_KL", float(primary[2]), "relative_span", relative_span, flush=True)
    return record


def algebra_check():
    rows = []
    for epsilon in (0., 1e-4, 1e-7):
        p, q = [.5+epsilon, .5-epsilon], [.5, .5]
        stable = stable_generalized_kl(p, q)
        text, exact = decimal_kl(p, q)
        error = abs(stable-exact)
        require(error <= 1e-18, "small-KL algebra fixture")
        rows.append({"epsilon": epsilon, "stable": stable, "Decimal80": text, "difference": error})
    row = {"scales": [2., 1.], "lambda": [.3, .2]}
    specs = spectrum(4, row)
    return {"passed": True, "KL_fixtures": rows, "d4_spectrum_error": specs["spectral_formula_max_error"],
            "scientific_integrals": 0, "RNG_calls": 0, "new_fits": 0}


def freeze():
    require(not OUT.exists(), "NUMERIC output already exists")
    require(sha(ROOT/"HANDOFF.md") == HANDOFF, "HANDOFF version changed")
    old_inputs()
    OUT.mkdir(parents=True)
    (OUT/"local").mkdir()
    (OUT/".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    snapshot = OUT/"HANDOFF_INPUT.md"
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json", CFG)
    write(OUT/"INPUT_HASHES.json", {rel(p): sha(p) for p in input_paths()})
    write(OUT/"ENVIRONMENT.json", {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "longdouble_mantissa_bits": np.finfo(np.longdouble).nmant,
        "decimal_digits": CFG["decimal_digits"], "threads": 2})
    paths = [Path(__file__).resolve(), DOC/"PLAN_NUMERIC.md", snapshot,
             OUT/"CONFIG.json", OUT/"INPUT_HASHES.json", OUT/"ENVIRONMENT.json"]
    write(OUT/"FREEZE.json", {
        "utc": now(), "baseline_commit": BASE, "handoff_raw_sha256": HANDOFF,
        "files": {rel(p): sha(p) for p in paths}, "before_scientific_integrals": True,
        "external_sandbox_files_used": False})
    write(OUT/"STATE.json", {"status": "frozen", "utc": now(), "active_jobs": []})
    print(json.dumps({"frozen": True, "FREEZE_sha256": sha(OUT/"FREEZE.json")}), flush=True)


def verify_freeze():
    verify(read(OUT/"FREEZE.json")["files"])
    verify(read(OUT/"INPUT_HASHES.json"))
    require(read(OUT/"CONFIG.json") == CFG, "config changed")


def run():
    verify_freeze()
    require(read(OUT/"STATE.json")["status"] == "frozen", "no silent numerical restart")
    write(OUT/"ALGEBRA_QA.json", algebra_check())
    write(OUT/"START.json", {"utc": now(), "before_selected_integrals": True})
    moments = old_inputs()
    write(OUT/"NOISE_CONSTANTS.json", noise_constants(moments))
    alpha = alpha_values(moments)
    summaries = []
    for d in CFG["dimensions"]:
        write(OUT/"STATE.json", {"status": "integrating", "d": d, "pid": os.getpid(), "utc": now()}, True)
        result = one_dimension(d, alpha)
        summaries.append({"d": d, "params": result["params"],
                          "risks": result["risk_to_teacher"].tolist(),
                          "minimum_eigenvalues": [s["minimum"] for s in result["spectra"]],
                          "maximum_eigenvalues": [s["maximum"] for s in result["spectra"]],
                          "precision": result["precision"], "elapsed_seconds": result["elapsed_seconds"]})
    write(OUT/"SUMMARY.json", {
        "all_qualified": all(r["precision"]["qualified"] for r in summaries),
        "rows": summaries, "primary_PMFS": 18, "variant_PMFS": 72,
        "new_observations": 0, "new_fits": 0,
        "new_alpha_integrals": 0, "external_sandbox_files_used": False,
        "no_global_optimization_or_learning_claim": True,
        "strict_integration_error_certificate": False})
    files = [p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts
             and p.name not in ("STATE.json", "RESULTS_FREEZE.json")]
    write(OUT/"RESULTS_FREEZE.json", {
        "utc": now(), "source_freeze_sha256": sha(OUT/"FREEZE.json"),
        "files": {rel(p): sha(p) for p in files},
        "local_arrays": {rel(p): sha(p) for p in (OUT/"local").glob("*.npz")}})
    write(OUT/"STATE.json", {"status": "complete", "utc": now(), "active_jobs": []}, True)
    print("NUMERIC complete; all frozen numerical gates passed.", flush=True)


@lru_cache(None)
def _reference(d):
    require(read(OUT/"STATE.json")["status"] == "complete", "NUMERIC not complete")
    require(read(OUT/"SUMMARY.json")["all_qualified"], "NUMERIC not qualified")
    path = OUT/f"REFERENCE_d{d}.json"
    seal = read(OUT/"RESULTS_FREEZE.json")
    require(sha(path) == seal["files"][rel(path)], "reference bytes changed")
    result = read(path)
    for key in ("pmf", "log_pattern", "risk_to_teacher", "event_probability", "event_brier", "event_logloss"):
        array = np.asarray(result[key], dtype=np.float64)
        array.setflags(write=False)
        result[key] = array
    return result


def reference(d):
    require(d in CFG["dimensions"], "unsupported reference dimension")
    return dict(_reference(d))


def check(workspace=False):
    verify_freeze()
    require(read(OUT/"STATE.json")["status"] == "complete", "package not complete")
    result = read(OUT/"RESULTS_FREEZE.json")
    require(result["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "seal chain")
    verify(result["files"])
    if workspace:
        verify(result["local_arrays"])
    require(read(OUT/"SUMMARY.json")["all_qualified"], "not numerically qualified")
    print(json.dumps({"passed": True, "workspace": workspace, "integrals_not_reexecuted": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("algebra-check", "freeze", "run", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "algebra-check":
            print(json.dumps(algebra_check(), indent=2))
        elif args.phase == "freeze":
            freeze()
        elif args.phase == "run":
            run()
        else:
            check(args.workspace)
    except Exception as exc:
        if args.phase == "run" and (OUT/"FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",
                  {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            write(OUT/"STATE.json", {"status": "stopped_numeric_failure", "utc": now(), "active_jobs": []}, True)
        raise
