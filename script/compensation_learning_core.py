"""Continuous two-loading compensated-model core and frozen numerical REF.

Candidate functions use no teacher parameters. Teacher tables are exposed by a
separate evaluation-only API. No RNG, fitting, or scientific quadrature before
the REF freeze; algebra-check uses an explicit finite d4 arithmetic fixture.
"""
from __future__ import annotations
import os
for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_key] = "2"
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
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/compensation_learning/REF"
DOC = ROOT/"docs/08_research_研究工作/补偿学习与连续目标_20261001"
OLD = ROOT/"artifacts/covariance_boundary/NUMERIC"
BASE = "eca7733c5085645245fb297502b50a69b14d2f22"
HANDOFF_SHA = "ebd22608c6c192bef4596e5cab5fd3ef9aa62f6120d39e9d1f83c9575291d8ee"
VARIANTS = ((64, 128), (64, 256), (128, 512))
CFG = {
    "dimensions": [16, 64], "w_bounds": [.5, 2.5], "scale": 1., "mean": 0.,
    "reference_axis": list(np.linspace(.5, 2.5, 9)),
    "kernel_w_points": 65, "kernel_z_points": 97, "runtime_common_domain": [-12., 12.],
    "private_primary": 64, "private_check": 128,
    "reference_variants": [list(v) for v in VARIANTS], "primary_variant": 1,
    "kernel_chunk": 2048, "kernel_log_gate": 1e-10, "kernel_log_gradient_gate": 1e-9,
    "normalization_gate": 1e-12, "PMF_mass_gate": 1e-11, "PMF_gradient_mass_gate": 1e-10,
    "PMF_L1_gate": 1e-9, "PMF_log_gate": 1e-7,
    "PMF_log_gradient_gate": 5e-7, "reference_risk_gate": 1e-9,
    "old_comp_KL_gate": 1e-12, "old_anchor_L1_gate": 1e-9,
    "finite_difference_axis": [.6, 1.5, 2.4], "finite_difference_step": 1e-4,
    "finite_difference_log_gradient_gate": 5e-8,
    "quad_w": [.5, 1., 1.5, 2., 2.5],
    "quad_z": [-12., -8., -4., -1., 0., 1., 4., 8., 12.],
    "quad_interval": [-12., 12.], "quad_epsabs": 1e-13, "quad_epsrel": 1e-13,
    "quad_limit": 400, "quad_log_gate": 1e-10, "quad_log_gradient_gate": 1e-9,
    "new_labels": 0, "new_RNG_calls": 0, "new_fits": 0,
    "finite_validation_grid_not_uniform_continuous_certificate": True,
    "runtime_is_direct_quadrature_not_interpolation": True,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=serial)
        handle.write("\n")


def verify(files):
    for name, digest in files.items():
        require(sha(ROOT/name) == digest, "hash changed: "+name)


def validate(d, w):
    require(d in CFG["dimensions"], "unsupported continuous model dimension")
    w = np.asarray(w, dtype=np.float64)
    require(w.shape == (2,) and np.isfinite(w).all(), "two finite loadings required")
    require(np.all(w >= CFG["w_bounds"][0]) and np.all(w <= CFG["w_bounds"][1]), "w outside frozen box")
    require(np.all(w*w < d-1), "nonpositive private variance")
    return w


@lru_cache(None)
def _gh(order):
    nodes, weights = roots_hermitenorm(order)
    weights = weights/math.sqrt(2*math.pi)
    require(np.all(weights >= 0) and abs(float(weights.sum())-1) <= CFG["normalization_gate"], "GH weights")
    return nodes, weights


def _kernel_rule(common, d, w_scalar, nodes, weights):
    """Exact derivative of this finite sum; no integration-by-parts substitution."""
    original = np.asarray(common, dtype=np.float64)
    flat = original.ravel()
    require(np.isfinite(flat).all(), "nonfinite common factors")
    lam = float(w_scalar)/math.sqrt(d-1)
    require(abs(lam) < 1, "invalid scalar loading")
    private = math.sqrt(1-lam*lam)
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    logweights = np.full_like(weights, -np.inf, dtype=float)
    positive = weights > 0
    logweights[positive] = np.log(weights[positive])
    lp, lq, dlp, dlq = (np.empty(len(flat)) for _ in range(4))
    for start in range(0, len(flat), CFG["kernel_chunk"]):
        z = flat[start:start+CFG["kernel_chunk"]]
        eta = lam*z[:, None]+private*nodes
        path_gradient = (z[:, None]-lam/private*nodes)/math.sqrt(d-1)
        logplus = log_expit(eta)+logweights
        logminus = log_expit(-eta)+logweights
        p = logsumexp(logplus, axis=1)
        q = logsumexp(logminus, axis=1)
        # Posterior normalized weights avoid dividing by tiny tail probabilities.
        gp = np.sum(np.exp(logplus-p[:, None])*expit(-eta)*path_gradient, axis=1)
        gq = -np.sum(np.exp(logminus-q[:, None])*expit(eta)*path_gradient, axis=1)
        target = slice(start, start+len(z))
        lp[target], lq[target], dlp[target], dlq[target] = p, q, gp, gq
    require(all(np.isfinite(x).all() for x in (lp, lq, dlp, dlq)), "kernel nonfinite")
    return tuple(x.reshape(original.shape) for x in (lp, lq, dlp, dlq))


def _kernel(common, d, w_scalar, order):
    return _kernel_rule(common, d, w_scalar, *_gh(order))


def _log_h(common, d, w, order):
    rows = [_kernel(common, d, g, order) for g in w]
    return tuple(np.stack([row[j] for row in rows]) for j in range(4))


@lru_cache(None)
def geometry(d):
    require(d > 0 and d % 2 == 0, "even dimension required")
    m = d//2
    counts = np.arange(m+1, dtype=float)
    choose = np.array([math.log(math.comb(m, int(k))) for k in counts])
    na, nb = np.meshgrid(counts, counts, indexing="ij")
    return na, nb, choose[:, None]+choose[None, :], choose


def _reference_rule(d, w, private_rule, outer_rule):
    """Positive PMF and exact derivative of the normalized finite quadrature."""
    m = d//2
    z, weights = outer_rule
    counts = np.arange(m+1, dtype=float)[:, None]
    choose = geometry(d)[3][:, None]
    pieces, derivatives = [], []
    for wi in w:
        lp, lq, dlp, dlq = _kernel_rule(z, d, wi, *private_rule)
        p = np.exp(choose+counts*lp+(m-counts)*lq)
        pieces.append(p)
        derivatives.append(p*(counts*dlp+(m-counts)*dlq))
    raw = (pieces[0]*weights) @ pieces[1].T
    draw = np.stack(((derivatives[0]*weights) @ pieces[1].T,
                     (pieces[0]*weights) @ derivatives[1].T), axis=-1)
    mass = float(raw.sum())
    grad_mass = draw.sum(axis=(0, 1))
    require(np.all(raw > 0) and np.isfinite(raw).all() and np.isfinite(draw).all(), "positive finite PMF")
    pmf = raw/mass
    grad_pmf = (draw-pmf[:, :, None]*grad_mass)/mass
    log_pattern = np.log(pmf)-geometry(d)[2]
    grad_log = grad_pmf/pmf[:, :, None]
    return {"d": d, "w": np.asarray(w).tolist(), "lambda": (np.asarray(w)/math.sqrt(d-1)).tolist(),
            "pmf": pmf, "log_pattern": log_pattern, "grad_pmf": grad_pmf,
            "grad_log_pattern": grad_log, "raw_mass": mass, "raw_gradient_mass": grad_mass,
            "normalization_gradient_included": True}


def _reference(d, w, variant):
    inner, outer = VARIANTS[variant]
    row = _reference_rule(d, w, _gh(inner), _gh(outer))
    row.update({"variant": variant, "private_order": inner, "outer_order": outer})
    return row


def stable_kl(p, q):
    p, q = np.asarray(p, dtype=np.longdouble), np.asarray(q, dtype=np.longdouble)
    require(np.all(p > 0) and np.all(q > 0), "full support required")
    p, q = p/p.sum(), q/q.sum()
    delta = (p-q)/q
    small = np.abs(delta) <= .01
    x = delta[small]
    poly = np.full(len(x), np.longdouble(1)/np.longdouble(18*17))
    for degree in range(17, 1, -1):
        poly = np.longdouble((-1)**degree)/np.longdouble(degree*(degree-1))+x*poly
    values = np.empty_like(p)
    values[small] = q[small]*x*x*poly
    values[~small] = p[~small]*(np.log(p[~small])-np.log(q[~small]))-p[~small]+q[~small]
    require(np.isfinite(values).all() and np.all(values >= 0), "stable KL term failure")
    return float(values.sum())


def _point_qualification(d, w, teacher=None):
    rows = [_reference(d, w, variant) for variant in range(3)]
    main = rows[CFG["primary_variant"]]
    metric = {
        "PMF_mass_error": max(abs(r["raw_mass"]-1) for r in rows),
        "PMF_gradient_mass_error": max(float(abs(r["raw_gradient_mass"]).max()) for r in rows),
        "PMF_L1_difference": max(float(abs(a["pmf"]-b["pmf"]).sum()) for a, b in itertools.combinations(rows, 2)),
        "PMF_log_difference": max(float(abs(a["log_pattern"]-b["log_pattern"]).max()) for a, b in itertools.combinations(rows, 2)),
        "PMF_log_gradient_difference": max(float(abs(a["grad_log_pattern"]-b["grad_log_pattern"]).max()) for a, b in itertools.combinations(rows, 2)),
        "complement_error": max(float(abs(r["pmf"]-r["pmf"][::-1, ::-1]).max()) for r in rows),
    }
    if teacher is not None:
        risks = [stable_kl(teacher, r["pmf"]) for r in rows]
        metric["reference_risks"] = risks
        metric["reference_risk_span"] = max(risks)-min(risks)
    checks = {
        "mass": metric["PMF_mass_error"] <= CFG["PMF_mass_gate"],
        "gradient_mass": metric["PMF_gradient_mass_error"] <= CFG["PMF_gradient_mass_gate"],
        "L1": metric["PMF_L1_difference"] <= CFG["PMF_L1_gate"],
        "log": metric["PMF_log_difference"] <= CFG["PMF_log_gate"],
        "log_gradient": metric["PMF_log_gradient_difference"] <= CFG["PMF_log_gradient_gate"],
        "complement": metric["complement_error"] <= CFG["normalization_gate"],
        "risk": teacher is None or metric["reference_risk_span"] <= CFG["reference_risk_gate"],
    }
    return main, {"d": d, "w": list(w), "metrics": metric, "checks": checks,
                  "qualified": all(checks.values()), "strict_error_certificate": False}, rows


def old_inputs():
    seal = read(OLD/"RESULTS_FREEZE.json")
    require(seal["source_freeze_sha256"] == sha(OLD/"FREEZE.json"), "old NUMERIC source seal")
    for d in CFG["dimensions"]:
        path = OLD/f"REFERENCE_d{d}.json"
        require(seal["files"][rel(path)] == sha(path), "old teacher input hash")
        require(read(path)["precision"]["qualified"], "old input unqualified")
    require(read(OLD/"INDEPENDENT_AUDIT.json")["passed"], "old NUMERIC audit")
    return seal


def _old_teacher(d):
    old = read(OLD/f"REFERENCE_d{d}.json")
    return np.asarray(old["pmf"][0], dtype=float)


def _kernel_qualification(d):
    z = np.linspace(*CFG["runtime_common_domain"], CFG["kernel_z_points"])
    ws = np.linspace(*CFG["w_bounds"], CFG["kernel_w_points"])
    saved = []
    for w in ws:
        low = _kernel(z, d, w, CFG["private_primary"])
        high = _kernel(z, d, w, CFG["private_check"])
        probability = max(float(abs(low[j]-high[j]).max()) for j in (0, 1))
        gradient = max(float(abs(low[j]-high[j]).max()) for j in (2, 3))
        normalization = max(float(abs(np.logaddexp(x[0], x[1])).max()) for x in (low, high))
        derivative_normalization = max(float(abs(np.exp(x[0])*x[2]+np.exp(x[1])*x[3]).max()) for x in (low, high))
        symmetry = max(float(abs(x[0]-x[1][::-1]).max()) for x in (low, high))
        gradient_symmetry = max(float(abs(x[2]-x[3][::-1]).max()) for x in (low, high))
        passed = (probability <= CFG["kernel_log_gate"] and gradient <= CFG["kernel_log_gradient_gate"]
                  and max(normalization, derivative_normalization, symmetry, gradient_symmetry) <= CFG["normalization_gate"])
        saved.append({"d": d, "w": float(w), "log_difference": probability,
                      "log_gradient_difference": gradient, "normalization": normalization,
                      "gradient_normalization": derivative_normalization, "symmetry": symmetry,
                      "gradient_symmetry": gradient_symmetry, "qualified": passed})
    write(OUT/f"KERNEL_d{d}.json", saved)
    require(all(r["qualified"] for r in saved), "kernel convergence gate failed")
    return saved


def _quad_qualification(d):
    rows = []
    for w in CFG["quad_w"]:
        lam = w/math.sqrt(d-1)
        private = math.sqrt(1-lam*lam)
        direct = _kernel(CFG["quad_z"], d, w, CFG["private_primary"])
        for i, z in enumerate(CFG["quad_z"]):
            values, errors, evaluations = [], [], []
            for kind in ("positive", "negative", "gradient"):
                def integrand(e):
                    eta = lam*z+private*e
                    density = math.exp(-e*e/2)/math.sqrt(2*math.pi)
                    if kind == "positive":
                        return density*float(expit(eta))
                    if kind == "negative":
                        return density*float(expit(-eta))
                    # Independent Stein form, not the finite-GH path derivative.
                    p, q = float(expit(eta)), float(expit(-eta))
                    return density*p*q*(z-lam*(1-2*p))/math.sqrt(d-1)
                answer = quad(integrand, *CFG["quad_interval"], epsabs=CFG["quad_epsabs"],
                              epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"], full_output=1)
                require(len(answer) == 3, "independent quad failed")
                values.append(float(answer[0])); errors.append(float(answer[1]))
                evaluations.append(int(answer[2]["neval"]))
            lp, lq = math.log(values[0]), math.log(values[1])
            gp, gq = values[2]/values[0], -values[2]/values[1]
            log_error = max(abs(lp-direct[0][i]), abs(lq-direct[1][i]))
            grad_error = max(abs(gp-direct[2][i]), abs(gq-direct[3][i]))
            rows.append({"d": d, "w": w, "z": z, "values": values, "reported_errors": errors,
                         "evaluations": evaluations, "log_difference": float(log_error),
                         "log_gradient_difference": float(grad_error),
                         "Gaussian_tail_mass_bound": 2*float(ndtr(-12.)),
                         "derivative_tail_bound": ((abs(z)+lam)/math.sqrt(d-1))*.25*2*float(ndtr(-12.)),
                         "qualified": bool(log_error <= CFG["quad_log_gate"] and grad_error <= CFG["quad_log_gradient_gate"])})
    write(OUT/f"QUAD_d{d}.json", rows)
    require(all(r["qualified"] for r in rows), "independent kernel quad gate failed")
    return rows


def _finite_difference(d):
    rows = []
    step = CFG["finite_difference_step"]
    for w in itertools.product(CFG["finite_difference_axis"], repeat=2):
        main = _reference(d, w, CFG["primary_variant"])
        for g in (0, 1):
            variants = []
            for offset in (-2, -1, 1, 2):
                point = np.asarray(w, dtype=float)
                point[g] += offset*step
                variants.append(_reference(d, point, CFG["primary_variant"])["log_pattern"])
            finite = (variants[0]-8*variants[1]+8*variants[2]-variants[3])/(12*step)
            error = float(abs(finite-main["grad_log_pattern"][:, :, g]).max())
            rows.append({"d": d, "w": list(w), "parameter": g, "step": step,
                         "max_log_gradient_difference": error,
                         "qualified": error <= CFG["finite_difference_log_gradient_gate"]})
    write(OUT/f"FINITE_DIFFERENCE_d{d}.json", rows)
    require(all(r["qualified"] for r in rows), "finite-difference PMF gradient gate failed")
    return rows


def _anchors(d):
    old = read(OLD/f"REFERENCE_d{d}.json")
    rows = []
    for name in ("fixedRfloor", "comp"):
        j = old["names"].index(name)
        w = np.asarray(old["params"][j]["lambda"])*math.sqrt(d-1)
        result = _reference(d, w, CFG["primary_variant"])
        l1 = float(abs(result["pmf"]-np.asarray(old["pmf"][j])).sum())
        risk = stable_kl(np.asarray(old["pmf"][0]), result["pmf"])
        delta = abs(risk-old["risk_to_teacher"][j])
        rows.append({"d": d, "name": name, "w": w, "PMF_L1": l1, "joint_KL": risk,
                     "old_KL": old["risk_to_teacher"][j], "KL_difference": delta,
                     "qualified": l1 <= CFG["old_anchor_L1_gate"] and delta <= CFG["old_comp_KL_gate"]})
    write(OUT/f"ANCHORS_d{d}.json", rows)
    require(all(r["qualified"] for r in rows), "old fixed-model identity gate failed")
    return rows


def algebra_check():
    # Finite toy distribution only: these are not Gaussian quadrature nodes.
    d, w = 4, np.array([.6, .8])
    private_rule = (np.array([-.7, .7]), np.array([.5, .5]))
    outer_rule = (np.array([-1., 0., 1.]), np.array([.25, .5, .25]))
    result = _reference_rule(d, w, private_rule, outer_rule)
    states = np.array(list(itertools.product((0, 1), repeat=d)))
    z, weights = outer_rule
    positive = [np.exp(_kernel_rule(z, d, g, *private_rule)[0]) for g in w]
    by_pattern = []
    for state in states:
        probabilities = np.stack([positive[0], positive[0], positive[1], positive[1]])
        terms = np.prod(np.where(state[:, None] == 1, probabilities, 1-probabilities), axis=0)
        by_pattern.append(float(terms @ weights))
    counts = np.zeros((3, 3))
    for state, prob in zip(states, by_pattern):
        counts[state[:2].sum(), state[2:].sum()] += prob
    enum_error = float(abs(counts-result["pmf"]).max())
    max_gradient = 0.
    step = 1e-5
    for g in (0, 1):
        a, b = w.copy(), w.copy()
        a[g] += step; b[g] -= step
        plus = _reference_rule(d, a, private_rule, outer_rule)
        minus = _reference_rule(d, b, private_rule, outer_rule)
        finite = (plus["log_pattern"]-minus["log_pattern"])/(2*step)
        max_gradient = max(max_gradient, float(abs(finite-result["grad_log_pattern"][:, :, g]).max()))
    gradient_mass = float(abs(result["grad_pmf"].sum(axis=(0, 1))).max())
    kl0 = stable_kl(result["pmf"], result["pmf"])
    require(enum_error <= 1e-12 and max_gradient <= 1e-8 and gradient_mass <= 1e-12 and kl0 == 0, "finite d4 algebra fixture")
    return {"passed": True, "d4_count_vs_16_pattern_max_error": enum_error,
            "d4_gradient_max_error": max_gradient, "gradient_mass": gradient_mass,
            "self_KL": kl0, "scientific_integrals": 0, "RNG_calls": 0, "fits": 0,
            "explicit_non_Gaussian_finite_fixture": True}


def freeze():
    require(not OUT.exists(), "REF already exists; no overwrite")
    require(sha(ROOT/"HANDOFF.md") == HANDOFF_SHA, "HANDOFF changed")
    old_inputs()
    OUT.mkdir(parents=True)
    (OUT/"local").mkdir()
    (OUT/".gitignore").write_text("/local/\n", encoding="utf-8", newline="\n")
    (OUT/"HANDOFF_INPUT.md").write_bytes((ROOT/"HANDOFF.md").read_bytes())
    inputs = [OLD/"FREEZE.json", OLD/"RESULTS_FREEZE.json", OLD/"INDEPENDENT_AUDIT.json"]
    inputs += [OLD/f"REFERENCE_d{d}.json" for d in CFG["dimensions"]]
    write(OUT/"INPUT_HASHES.json", {rel(p): sha(p) for p in inputs})
    write(OUT/"CONFIG.json", CFG)
    write(OUT/"ENVIRONMENT.json", {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
          "platform": platform.platform(), "threads": 2, "longdouble_mantissa_bits": np.finfo(np.longdouble).nmant})
    files = [Path(__file__).resolve(), DOC/"PLAN_REF.md", OUT/"HANDOFF_INPUT.md",
             OUT/"INPUT_HASHES.json", OUT/"CONFIG.json", OUT/"ENVIRONMENT.json"]
    write(OUT/"FREEZE.json", {"utc": now(), "baseline_commit": BASE, "handoff_raw_sha256": HANDOFF_SHA,
          "files": {rel(p): sha(p) for p in files}, "before_scientific_integrals": True,
          "before_scientific_RNG": True})
    write(OUT/"STATE.json", {"status": "frozen", "utc": now()})
    print(json.dumps({"frozen": True, "FREEZE_sha256": sha(OUT/"FREEZE.json")}))


def verify_freeze():
    verify(read(OUT/"FREEZE.json")["files"])
    verify(read(OUT/"INPUT_HASHES.json"))
    require(read(OUT/"CONFIG.json") == CFG, "config changed")


def run():
    verify_freeze()
    require(read(OUT/"STATE.json")["status"] == "frozen", "no silent REF restart")
    started = time.perf_counter()
    write(OUT/"START.json", {"utc": now(), "pid": os.getpid()})
    write(OUT/"ALGEBRA_QA.json", algebra_check())
    summary = []
    for d in CFG["dimensions"]:
        write(OUT/"STATE.json", {"status": "integrating", "d": d, "utc": now()}, True)
        kernel = _kernel_qualification(d)
        independent = _quad_qualification(d)
        teacher = _old_teacher(d)
        qualifiers, pmfs, gradients = [], [], []
        grid = list(itertools.product(CFG["reference_axis"], repeat=2))
        for w in grid:
            main, qualification, variants = _point_qualification(d, w, teacher)
            qualifiers.append(qualification)
            pmfs.append(np.stack([r["pmf"] for r in variants]))
            gradients.append(np.stack([r["grad_log_pattern"] for r in variants]))
        path = OUT/"local"/f"reference_grid_d{d}.npz"
        np.savez_compressed(path, w=np.asarray(grid), pmf=np.asarray(pmfs), grad_log_pattern=np.asarray(gradients))
        write(OUT/f"REFERENCE_GRID_d{d}.json", {"d": d, "points": grid, "qualification": qualifiers,
              "array": rel(path), "array_sha256": sha(path), "all_qualified": all(r["qualified"] for r in qualifiers),
              "continuous_optimizer_certificate": False})
        require(all(r["qualified"] for r in qualifiers), "reference-grid qualification failed")
        finite = _finite_difference(d)
        anchors = _anchors(d)
        summary.append({"d": d, "kernel_points": len(kernel)*CFG["kernel_z_points"],
                        "quad_locations": len(independent), "reference_points": len(grid),
                        "reference_variant_PMFS": len(grid)*3, "finite_difference_parameters": len(finite),
                        "anchors": len(anchors), "qualified": True})
        print("REF", d, "qualified", flush=True)
    write(OUT/"QUALIFICATION.json", {"all_qualified": True, "rows": summary,
          "reference_grid_PMFS": 486, "finite_difference_primary_PMFS": 162,
          "old_anchor_PMFS": 4, "kernel_pairs_per_order": 12610,
          "adaptive_quad_scalar_calls": 270, "new_labels": 0, "new_RNG_calls": 0, "new_fits": 0,
          "elapsed_seconds": time.perf_counter()-started,
          "qualification_is_finite_not_uniform_certificate": True,
          "runtime_endpoint_three_variant_checks_required": True})
    files = [p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts
             and p.name not in ("STATE.json", "RESULTS_FREEZE.json")]
    write(OUT/"RESULTS_FREEZE.json", {"utc": now(), "source_freeze_sha256": sha(OUT/"FREEZE.json"),
          "files": {rel(p): sha(p) for p in files},
          "local_arrays": {rel(p): sha(p) for p in (OUT/"local").glob("*.npz")}})
    write(OUT/"STATE.json", {"status": "complete", "utc": now(), "active_jobs": []}, True)
    print("REF complete", flush=True)


@lru_cache(None)
def _ready():
    verify_freeze()
    require(read(OUT/"STATE.json")["status"] == "complete", "REF incomplete")
    seal = read(OUT/"RESULTS_FREEZE.json")
    require(seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "REF seal chain")
    path = OUT/"QUALIFICATION.json"
    require(sha(path) == seal["files"][rel(path)] and read(path)["all_qualified"], "REF unqualified")
    return True


def log_h(common, d, w):
    """lp,lq,dlp,dlq, each [2,*common.shape]; gradient row g is d/dw_g."""
    _ready()
    w = validate(d, w)
    common = np.asarray(common, dtype=float)
    require(np.all(common >= CFG["runtime_common_domain"][0]) and np.all(common <= CFG["runtime_common_domain"][1]),
            "common outside prespecified validation domain; stop, no clipping")
    return _log_h(common, d, w, CFG["private_primary"])


def reference(d, w, variant="primary"):
    """Numerical full Q and gradients; teacher-independent; no optimizer call."""
    _ready()
    w = validate(d, w)
    index = CFG["primary_variant"] if variant == "primary" else int(variant)
    require(index in (0, 1, 2), "unknown reference variant")
    return _reference(d, w, index)


def qualify_reference(d, w, teacher_pmf=None):
    """Endpoint audit, optional teacher only for evaluation risk sensitivity."""
    _ready()
    w = validate(d, w)
    main, receipt, _ = _point_qualification(d, w, teacher_pmf)
    return main, receipt


def qualify_log_h(common, d, w):
    """Endpoint stability of the runtime bank values and gradients."""
    low = log_h(common, d, w)
    high = _log_h(common, d, w, CFG["private_check"])
    value = max(float(abs(low[j]-high[j]).max()) for j in (0, 1))
    gradient = max(float(abs(low[j]-high[j]).max()) for j in (2, 3))
    return {"log_difference": value, "log_gradient_difference": gradient,
            "qualified": value <= CFG["kernel_log_gate"] and gradient <= CFG["kernel_log_gradient_gate"],
            "strict_error_certificate": False}


def teacher_reference(d):
    """Evaluation/gen only: frozen old positive (6,2),c=.9 teacher."""
    _ready()
    require(d in CFG["dimensions"], "unsupported teacher dimension")
    path = OLD/f"REFERENCE_d{d}.json"
    require(read(OUT/"INPUT_HASHES.json")[rel(path)] == sha(path), "teacher source changed")
    old = read(path)
    return {"d": d, "pmf": np.asarray(old["pmf"][0]), "log_pattern": np.asarray(old["log_pattern"][0]),
            "params": old["params"][0], "source": rel(path), "source_sha256": sha(path),
            "numerically_qualified_not_strict_integration_certificate": True,
            "evaluation_only_not_a_candidate_or_learner_input": True}


def check(workspace=False):
    _ready()
    seal = read(OUT/"RESULTS_FREEZE.json")
    verify(seal["files"])
    if workspace:
        verify(seal["local_arrays"])
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
            write(OUT/f"FAILURE_{time.time_ns()}.json", {"utc": now(), "error": repr(exc),
                                                       "traceback": traceback.format_exc()})
            write(OUT/"STATE.json", {"status": "stopped_numeric_failure", "utc": now(), "active_jobs": []}, True)
        raise
