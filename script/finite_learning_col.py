"""Positive two-scale private-noise-collapsed numerical core and frozen REF.

Only freeze/build create artifacts. Public log_h/reference APIs read a complete
sealed REF; no sampling, quadrature, or writes occur through those APIs.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
from datetime import datetime, timezone
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
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, gammaln, ndtr, rel_entr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/finite_learning/REF"
DOC = ROOT/"docs/08_research_研究工作/有限样本学习与均值自由度_20261001"
BASE = "994765369015e1ce2dd925ec2a28dc2fa7a875cc"
AXIS = (1., 1.5, 2., 3., 4., 6., 8.)
VARIANTS = ((1024, 512), (2048, 256), (2048, 512))
CFG = {
    "d": [16, 64], "c": .9, "axis": list(AXIS), "truth": [6., 2.],
    "domain": [-12., 12.], "lookup_nodes": 4097,
    "private_orders": [1024, 2048], "primary_private_order": 2048,
    "reference_variants": [list(v) for v in VARIANTS], "primary_variant": 2,
    "GH_chunk": 128, "h_order_gate": 1e-9, "derivative_order_gate": 1e-8,
    "identity_gate": 1e-12, "score_order_bound_gate": 1e-6,
    "exact_node_interpolation_score_bound_gate": 1e-7,
    "PMF_L1_gate": 1e-8, "log_PMF_gate": 1e-6, "risk_sensitivity_gate": 1e-7,
    "quad_z": [-12., -8., -4., -1., 0., 1., 4., 8., 12.],
    "quad_interval": [-12., 12.], "quad_epsabs": 1e-12,
    "quad_epsrel": 1e-12, "quad_limit": 400,
    "quad_log_probability_gate": 1e-9, "quad_derivative_gate": 1e-10,
    "new_observations": 0, "new_fits": 0, "numerical_qualification_not_interval_certificate": True,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, obj, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, default=serial, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pairs():
    return list(itertools.product(AXIS, AXIS))


def geometry(d):
    require(d in CFG["d"] or d == 4, "unsupported geometry")
    m = d//2
    n = np.arange(m+1)
    logchoose = gammaln(m+1)-gammaln(n+1)-gammaln(m-n+1)
    a, b = np.meshgrid(n, n, indexing="ij")
    return a, b, logchoose[:, None]+logchoose[None, :]


@lru_cache(None)
def gh(order):
    z, w = roots_hermitenorm(order)
    w = w/math.sqrt(2*math.pi)
    require(np.all(w >= 0) and abs(float(w.sum())-1) <= CFG["identity_gate"], "GH weights")
    require(float(np.max(abs(z+z[::-1]))) <= CFG["identity_gate"], "GH node symmetry")
    require(float(np.max(abs(w-w[::-1]))) <= CFG["identity_gate"], "GH weight symmetry")
    lw = np.full_like(w, -np.inf)
    positive = w > 0
    lw[positive] = np.log(w[positive])
    return z, w, lw


def direct_kernel(common, d, scale, order, derivatives=False):
    """Direct quadrature; called only in REF build, never public runtime log_h."""
    z = np.asarray(common, dtype=np.float64).ravel()
    nodes, weights, logweights = gh(order)
    rho = CFG["c"]/(d-1)
    a, b = scale*math.sqrt(rho), scale*math.sqrt(1-rho)
    lp, lq = np.empty(len(z)), np.empty(len(z))
    derivative = np.empty(len(z)) if derivatives else None
    for begin in range(0, len(z), CFG["GH_chunk"]):
        zz = z[begin:begin+CFG["GH_chunk"]]
        eta = a*zz[:, None]+b*nodes
        lp[begin:begin+len(zz)] = logsumexp(log_expit(eta)+logweights, axis=1)
        lq[begin:begin+len(zz)] = logsumexp(log_expit(-eta)+logweights, axis=1)
        if derivatives:
            derivative[begin:begin+len(zz)] = a*((expit(eta)*expit(-eta)) @ weights)
    require(np.isfinite(lp).all() and np.isfinite(lq).all(), "nonfinite direct log probabilities")
    return lp, lq, derivative


def hermite_values(grid, values, derivatives, query):
    original = np.asarray(query, dtype=np.float64)
    flat = original.ravel()
    require(np.isfinite(flat).all() and np.all(flat >= grid[0]) and np.all(flat <= grid[-1]),
            "common Gaussian outside frozen interpolation domain; no clipping")
    delta = float(grid[1]-grid[0])
    position = np.floor((flat-grid[0])/delta).astype(np.int64)
    position = np.minimum(position, len(grid)-2)  # Endpoint interval, not probability clipping.
    t = (flat-grid[position])/delta
    h00, h10 = 2*t**3-3*t**2+1, t**3-2*t**2+t
    h01, h11 = -2*t**3+3*t**2, t**3-t**2
    answer = (values[:, position]*h00+delta*derivatives[:, position]*h10
              +values[:, position+1]*h01+delta*derivatives[:, position+1]*h11)
    return answer.reshape((len(values),)+original.shape)


def interpolation_bound(d):
    rho = CFG["c"]/(d-1)
    q = math.sqrt(rho/(1-rho))
    delta = (CFG["domain"][1]-CFG["domain"][0])/(CFG["lookup_nodes"]-1)
    fourth = math.sqrt(24)*q**4
    error = delta**4*fourth/384
    minimum = .5*float(ndtr(-12*q))
    require(error < minimum, "interpolation analytic positivity margin")
    score = d*error/(minimum-error)
    return {
        "grid_spacing": delta, "fourth_derivative_bound": fourth,
        "exact_node_uniform_h_error": error, "exact_h_and_complement_lower": minimum,
        "exact_node_vector_logscore_error_bound": score,
        "analytic_formulas_evaluated_in_float_not_interval_arithmetic": True,
        "GH_node_error_not_included_in_analytic_bound": True,
    }


def quad_checks(d):
    rows = []
    root2pi = math.sqrt(2*math.pi)
    rho = CFG["c"]/(d-1)
    for scale in AXIS:
        lp, lq, derivative = direct_kernel(CFG["quad_z"], d, scale, 2048, True)
        a, b = scale*math.sqrt(rho), scale*math.sqrt(1-rho)
        for i, z in enumerate(CFG["quad_z"]):
            values, estimates, calls = [], [], []
            for kind in ("h", "q", "derivative"):
                def integrand(e):
                    eta = a*z+b*e
                    density = math.exp(-e*e/2)/root2pi
                    if kind == "h":
                        return density*float(expit(eta))
                    if kind == "q":
                        return density*float(expit(-eta))
                    return density*a*float(expit(eta)*expit(-eta))
                result = quad(integrand, *CFG["quad_interval"], epsabs=CFG["quad_epsabs"],
                              epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"], full_output=1)
                require(len(result) == 3, "independent quad failed")
                values.append(float(result[0]))
                estimates.append(float(result[1]))
                calls.append(int(result[2]["neval"]))
            require(values[0] > 0 and values[1] > 0, "quad positive probabilities")
            log_error = max(abs(math.log(values[0])-lp[i]), abs(math.log(values[1])-lq[i]))
            derivative_error = abs(values[2]-derivative[i])
            rows.append({
                "d": d, "scale": scale, "z": z, "quad_values_h_q_derivative": values,
                "quad_reported_errors": estimates, "quad_evaluations": calls,
                "log_probability_difference": log_error, "derivative_difference": derivative_error,
                "Gaussian_truncation_mass_bound": 2*float(ndtr(-12.)),
                "passed": bool(log_error <= CFG["quad_log_probability_gate"]
                               and derivative_error <= CFG["quad_derivative_gate"]),
            })
    return rows


def build_lookup(d):
    started = time.perf_counter()
    grid = np.linspace(*CFG["domain"], CFG["lookup_nodes"])
    hs, derivatives = [], []
    normalization = symmetry = 0.0
    for order in CFG["private_orders"]:
        hrows, drows = [], []
        for scale in AXIS:
            lp, lq, prime = direct_kernel(grid, d, scale, order, True)
            h = np.exp(lp)
            normalization = max(normalization, float(abs(np.logaddexp(lp, lq)).max()))
            symmetry = max(symmetry, float(abs(h+h[::-1]-1).max()),
                           float(abs(prime-prime[::-1]).max()))
            hrows.append(h)
            drows.append(prime)
        hs.append(np.asarray(hrows))
        derivatives.append(np.asarray(drows))
    hs, derivatives = np.asarray(hs), np.asarray(derivatives)
    delta_h = float(abs(hs[1]-hs[0]).max())
    delta_prime = float(abs(derivatives[1]-derivatives[0]).max())
    analytic = interpolation_bound(d)
    spacing = analytic["grid_spacing"]
    minimum = min(float(hs.min()), float((1-hs).min()))
    positive = bool(minimum > 0 and np.isfinite(derivatives).all())
    secant = np.diff(hs, axis=2)/spacing
    monotone = bool(np.all(secant > 0) and np.all(derivatives >= 0)
                    and np.all(derivatives[:, :, :-1]+derivatives[:, :, 1:] <= 3*secant))
    perturbation = delta_h+spacing*delta_prime/4
    score_order_bound = d*perturbation/minimum
    record = {
        "d": d, "orders": CFG["private_orders"], "node_h_order_difference": delta_h,
        "node_derivative_order_difference": delta_prime,
        "normalization_log_error": normalization, "symmetry_error": symmetry,
        "minimum_node_probability_or_complement": minimum,
        "lookup_positive": positive, "monotonicity_sufficient_condition": monotone,
        "interpolator_order_uniform_value_bound": perturbation,
        "vector_logscore_order_stability_bound": score_order_bound,
        "analytic_interpolation": analytic,
        "qualified": bool(positive and monotone and delta_h <= CFG["h_order_gate"]
                          and delta_prime <= CFG["derivative_order_gate"]
                          and normalization <= CFG["identity_gate"] and symmetry <= CFG["identity_gate"]
                          and score_order_bound <= CFG["score_order_bound_gate"]
                          and analytic["exact_node_vector_logscore_error_bound"]
                              <= CFG["exact_node_interpolation_score_bound_gate"]),
        "elapsed_seconds": time.perf_counter()-started,
    }
    path = OUT/"local"/f"lookup_d{d}.npz"
    np.savez_compressed(path, z=grid, h=hs, derivative=derivatives)
    record["array"] = relative(path)
    record["array_sha256"] = sha(path)
    write(OUT/f"LOOKUP_d{d}.json", record)
    require(record["qualified"], "lookup qualification failed at d="+str(d))
    independent = quad_checks(d)
    write(OUT/f"QUAD_d{d}.json", independent)
    require(all(row["passed"] for row in independent), "quad qualification failed at d="+str(d))
    return record, independent


def pmf_from_logs(lp_a, lq_a, lp_b, lq_b, weights, m):
    counts = np.arange(m+1)[:, None]
    combinatorial = (gammaln(m+1)-gammaln(counts+1)-gammaln(m-counts+1))
    aa = np.exp(combinatorial+counts*lp_a+(m-counts)*lq_a)
    bb = np.exp(combinatorial+counts*lp_b+(m-counts)*lq_b)
    return (aa*weights[None, :]) @ bb.T


def event_arrays(pmf, d, teacher_index):
    na, nb, _ = geometry(d)
    spin = 2*(na+nb)-d
    masks = [(spin*spin <= d), (spin > math.sqrt(d)), (spin < -math.sqrt(d))]
    probability = np.stack([pmf[:, mask].sum(axis=1) for mask in masks], axis=1)
    p = probability[teacher_index]
    brier = (probability-p)**2
    logloss = rel_entr(p, probability)+rel_entr(1-p, 1-probability)
    return probability, brier, logloss


def build_reference(d):
    started = time.perf_counter()
    m = d//2
    all_pmfs, raw_masses = [], []
    direct_cache = {}
    for inner, outer in VARIANTS:
        z, weights, _ = gh(outer)
        lp = np.empty((7, outer))
        lq = np.empty_like(lp)
        for si, scale in enumerate(AXIS):
            key = (inner, outer, si)
            if key not in direct_cache:
                direct_cache[key] = direct_kernel(z, d, scale, inner)[:2]
            lp[si], lq[si] = direct_cache[key]
        matrices, masses = [], []
        for ui, vi in itertools.product(range(7), repeat=2):
            raw = pmf_from_logs(lp[ui], lq[ui], lp[vi], lq[vi], weights, m)
            mass = float(raw.sum())
            require(np.isfinite(raw).all() and np.all(raw > 0), "nonpositive reference PMF")
            matrices.append(raw/mass)
            masses.append(mass)
        all_pmfs.append(matrices)
        raw_masses.append(masses)
    variants = np.asarray(all_pmfs)
    raw_masses = np.asarray(raw_masses)
    primary = variants[2]
    log_mult = geometry(d)[2]
    log_pattern = np.log(primary)-log_mult
    full_mass_error = float(abs(raw_masses-1).max())
    l1 = max(float(abs(variants[a]-variants[b]).sum(axis=(1, 2)).max())
             for a, b in itertools.combinations(range(3), 2))
    log_diff = max(float(abs(np.log(variants[a])-np.log(variants[b])).max())
                   for a, b in itertools.combinations(range(3), 2))
    mirror_error = complement_error = 0.0
    for vi in range(3):
        for ui, wi in itertools.product(range(7), repeat=2):
            matrix = variants[vi, 7*ui+wi]
            mirror_error = max(mirror_error, float(abs(matrix-variants[vi, 7*wi+ui].T).max()))
            complement_error = max(complement_error, float(abs(matrix-matrix[::-1, ::-1]).max()))
    truth_index = pairs().index(tuple(CFG["truth"]))
    risks = []
    events, briers, binary_logs = [], [], []
    for a in range(3):
        p = variants[a, truth_index]
        ep, eb, el = event_arrays(variants[a], d, truth_index)
        events.append(ep)
        briers.append(eb)
        binary_logs.append(el)
        for b in range(3):
            risks.append(np.sum(p*(np.log(p)-np.log(variants[b])), axis=(1, 2)))
    risks = np.asarray(risks)
    events = np.asarray(events)
    teacher_risk_sensitivity = float(np.ptp(risks, axis=0).max())
    event_sensitivity = float(np.ptp(events, axis=0).max())
    cross_brier, cross_log = [], []
    for a in range(3):
        p = events[a, truth_index]
        for b in range(3):
            q = events[b]
            cross_brier.append((p-q)**2)
            cross_log.append(rel_entr(p, q)+rel_entr(1-p, 1-q))
    event_risk_sensitivity = max(float(np.ptp(cross_brier, axis=0).max()),
                                 float(np.ptp(cross_log, axis=0).max()))
    qualification = {
        "raw_PMF_mass_max_error": full_mass_error, "PMF_L1_variant_difference": l1,
        "log_PMF_variant_difference": log_diff, "mirror_error": mirror_error,
        "global_bit_complement_error": complement_error,
        "teacher_candidate_joint_KL_sensitivity": teacher_risk_sensitivity,
        "event_probability_sensitivity": event_sensitivity,
        "event_risk_sensitivity": event_risk_sensitivity,
        "all_probabilities_positive": True,
        "all_passed": bool(full_mass_error <= CFG["identity_gate"] and l1 <= CFG["PMF_L1_gate"]
                           and log_diff <= CFG["log_PMF_gate"]
                           and max(mirror_error, complement_error) <= CFG["identity_gate"]
                           and max(teacher_risk_sensitivity, event_sensitivity, event_risk_sensitivity)
                               <= CFG["risk_sensitivity_gate"]),
        "strict_integral_error_certificate": False,
    }
    record = {
        "d": d, "m": m, "c": CFG["c"], "axis": list(AXIS), "parameters": pairs(),
        "count_order": "row nA, column nB; nA,nB=0..m", "truth_index": truth_index,
        "reference_variants": [list(v) for v in VARIANTS], "primary_variant": 2,
        "pmf": primary, "log_pattern": log_pattern, "variants_pmf": variants,
        "raw_variant_masses": raw_masses, "renormalization": "divide by recorded raw positive PMF mass",
        "joint_KL_to_teacher": risks[8],
        "event_names": ["central", "upper_tail", "lower_tail"],
        "event_definitions": ["(2*(nA+nB)-d)^2<=d", "2*(nA+nB)-d>sqrt(d)", "2*(nA+nB)-d<-sqrt(d)"],
        "event_probability": events[2], "event_brier_excess": np.asarray(briers)[2],
        "event_logloss_excess": np.asarray(binary_logs)[2],
        "qualification": qualification, "elapsed_seconds": time.perf_counter()-started,
        "outer_nodes_outside_lookup_domain_use_direct_log_GH": True,
    }
    write(OUT/f"REFERENCE_d{d}.json", record)
    require(qualification["all_passed"], "reference qualification failed at d="+str(d))
    return record


def algebra_check():
    grid = np.linspace(-1., 1., 9)
    f = lambda x: .5+.02*x+.003*x*x-.001*x*x*x
    fp = lambda x: .02+.006*x-.003*x*x
    query = np.linspace(-1., 1., 37)
    error = float(abs(hermite_values(grid, f(grid)[None, :], fp(grid)[None, :], query)[0]-f(query)).max())
    require(error <= 1e-14, "fixed cubic interpolation identity")
    pa, pb = np.asarray([.25, .75]), np.asarray([.4, .6])
    weights = np.asarray([.3, .7])
    matrix = pmf_from_logs(np.log(pa), np.log1p(-pa), np.log(pb), np.log1p(-pb), weights, 2)
    brute = np.zeros((3, 3))
    for c in itertools.product((0, 1), repeat=4):
        probability = sum(w*np.prod([p if bit else 1-p for bit in c[:2]])
                          *np.prod([q if bit else 1-q for bit in c[2:]])
                          for w, p, q in zip(weights, pa, pb))
        brute[sum(c[:2]), sum(c[2:])] += probability
    mass_error = abs(float(matrix.sum())-1)
    identity = float(abs(matrix-brute).max())
    require(mass_error <= 1e-14 and identity <= 1e-14, "d4 product/count bookkeeping")
    return {"passed": True, "cubic_error": error, "d4_mass_error": mass_error,
            "d4_pattern_count_error": identity, "scientific_integrals": 0,
            "scientific_observations": 0, "RNG_calls": 0}


def freeze():
    require(not OUT.exists(), "REF directory already exists")
    OUT.mkdir(parents=True)
    (OUT/"local").mkdir()
    (OUT/".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    snapshot = OUT/"HANDOFF_INPUT.md"
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json", CFG)
    write(OUT/"ENVIRONMENT.json", {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "threads": 2})
    paths = [Path(__file__).resolve(), DOC/"PLAN_REF.md", OUT/"CONFIG.json",
             OUT/"ENVIRONMENT.json", snapshot]
    write(OUT/"FREEZE.json", {
        "utc": now(), "baseline_commit": BASE, "before_scientific_integrals": True,
        "files": {relative(p): sha(p) for p in paths},
        "handoff_raw_sha256": sha(snapshot), "new_samples": 0, "new_fits": 0})
    write(OUT/"STATE.json", {"status": "frozen", "utc": now(), "active_jobs": []})
    print(json.dumps({"frozen": True, "FREEZE_sha256": sha(OUT/"FREEZE.json")}), flush=True)


def verify_freeze():
    for name, digest in read(OUT/"FREEZE.json")["files"].items():
        require(sha(ROOT/name) == digest, "REF frozen input mismatch: "+name)
    require(read(OUT/"CONFIG.json") == CFG, "REF config changed")


def build():
    verify_freeze()
    require(read(OUT/"STATE.json")["status"] == "frozen", "no silent REF restart")
    write(OUT/"START.json", {"utc": now(), "first_scientific_integral_after_freeze": True})
    write(OUT/"ALGEBRA_QA.json", algebra_check())
    results = []
    for d in CFG["d"]:
        write(OUT/"STATE.json", {"status": "building", "d": d, "pid": os.getpid(), "utc": now()}, True)
        lookup, quad_rows = build_lookup(d)
        reference_record = build_reference(d)
        results.append({"d": d, "lookup_qualified": lookup["qualified"],
                        "quad_qualified": all(row["passed"] for row in quad_rows),
                        "reference_qualified": reference_record["qualification"]["all_passed"]})
        print("REF", d, "qualified", flush=True)
    qualification = {"all_passed": all(all(row[k] for k in ("lookup_qualified", "quad_qualified", "reference_qualified")) for row in results),
                     "dimensions": results, "reference_PMFS": 98, "variant_PMF_evaluations": 294,
                     "new_samples": 0, "new_fits": 0,
                     "strict_integral_error_certificate": False}
    write(OUT/"QUALIFICATION.json", qualification)
    public = [p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts
              and p.name not in ("STATE.json", "RESULTS_FREEZE.json")]
    write(OUT/"RESULTS_FREEZE.json", {
        "utc": now(), "source_freeze_sha256": sha(OUT/"FREEZE.json"),
        "files": {relative(p): sha(p) for p in public},
        "local_arrays": {relative(p): sha(p) for p in (OUT/"local").glob("*.npz")}})
    write(OUT/"STATE.json", {"status": "complete", "utc": now(), "active_jobs": []}, True)
    print(json.dumps(qualification), flush=True)


def ready():
    require(read(OUT/"STATE.json")["status"] == "complete", "REF is not complete")
    require(read(OUT/"QUALIFICATION.json")["all_passed"], "REF is not qualified")


@lru_cache(None)
def _lookup(d):
    ready()
    require(d in CFG["d"], "unsupported REF dimension")
    meta = read(OUT/f"LOOKUP_d{d}.json")
    require(meta["qualified"], "lookup not qualified")
    path = ROOT/meta["array"]
    require(sha(path) == meta["array_sha256"], "lookup bytes changed")
    with np.load(path, allow_pickle=False) as f:
        arrays = [f["z"].copy(), f["h"].copy(), f["derivative"].copy()]
    for array in arrays:
        array.setflags(write=False)
    return tuple(arrays)


def log_h(common, d, order=2048):
    """Return lp,lq with shape (7,*common.shape), no integration or writes."""
    require(order in CFG["private_orders"], "only frozen lookup orders may be requested")
    grid, values, primes = _lookup(int(d))
    oi = CFG["private_orders"].index(order)
    h = hermite_values(grid, values[oi], primes[oi], common)
    require(np.all(h > 0) and np.all(h < 1) and np.isfinite(h).all(), "interpolated probability invalid")
    return np.log(h), np.log1p(-h)


@lru_cache(None)
def _reference(d):
    ready()
    path = OUT/f"REFERENCE_d{d}.json"
    seal = read(OUT/"RESULTS_FREEZE.json")
    require(sha(path) == seal["files"][relative(path)], "reference bytes changed")
    record = read(path)
    for key in ("pmf", "log_pattern", "variants_pmf", "raw_variant_masses", "joint_KL_to_teacher",
                "event_probability", "event_brier_excess", "event_logloss_excess"):
        array = np.asarray(record[key], dtype=np.float64)
        array.setflags(write=False)
        record[key] = array
    return record


def reference(d):
    """49 lex-axis Cartesian PMFs/log-pattern tables, and three qualified variants."""
    require(int(d) in CFG["d"], "unsupported REF dimension")
    return dict(_reference(int(d)))


def check(workspace=False):
    verify_freeze()
    ready()
    seal = read(OUT/"RESULTS_FREEZE.json")
    require(seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "REF seal chain")
    for name, digest in seal["files"].items():
        require(sha(ROOT/name) == digest, "REF public result hash")
    if workspace:
        for name, digest in seal["local_arrays"].items():
            require(sha(ROOT/name) == digest, "REF local lookup hash")
    print(json.dumps({"passed": True, "workspace": workspace, "integrals_not_reexecuted": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("algebra-check", "freeze", "build", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "algebra-check":
            print(json.dumps(algebra_check()))
        elif args.phase == "freeze":
            freeze()
        elif args.phase == "build":
            build()
        else:
            check(args.workspace)
    except Exception as exc:
        if args.phase == "build" and (OUT/"FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",
                  {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            write(OUT/"STATE.json", {"status": "stopped_numerical_failure", "utc": now(), "active_jobs": []}, True)
        raise
