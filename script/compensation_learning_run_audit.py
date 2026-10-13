"""Independent result audit of the frozen compensation-learning RUN.

Predeclared scope: all selections, all statistics, all independent deployment
PMFs (exact duplicate parameters cached), all input RNG replay; likelihood
replay only first 16 training labels/2 banks and first 2 confirmation banks.
No optimization or new scientific samples. The producer is imported solely to
obtain prefix objective/gradient outputs for comparison with independent code.
"""
from __future__ import annotations
import os
for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_key] = "2"
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, kl_div
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"artifacts/compensation_learning/RUN"
REF = ROOT/"artifacts/compensation_learning/REF"
OUT = ROOT/"artifacts/compensation_learning/RUN_AUDIT"
DOC = ROOT/"docs/08_research_研究工作/补偿学习与连续目标_20261001"
OLD = ROOT/"artifacts/covariance_boundary/NUMERIC"
MOM = ROOT/"artifacts/objective_strengthening/N/MOMENTS.json"
SCOPE = {
    "dimensions": [16, 64], "streams_per_dimension": 8,
    "training_prefix_labels": 16, "training_prefix_banks": 2,
    "confirmation_prefix_banks": 2, "K_retained": "d",
    "independent_private_GH": 96, "independent_outer_GH": 384,
    "value_gradient_tolerance": 1e-8, "risk_event_tolerance": 1e-9,
    "statistics_tolerance": 1e-10, "bookkeeping_tolerance": 1e-11,
    "independent_PMFs": "all unique (d,w) across 128 learned outputs",
    "all_stored_RNG_inputs_replayed": True,
    "reoptimize": False, "new_scientific_samples": 0,
    "all_original_likelihood_terms_replayed": False,
}


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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, default=serial, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


class Audit:
    def __init__(self):
        self.count = 0
        self.maximum = {}

    def test(self, value, message):
        self.count += 1
        if not value:
            raise AssertionError(message)

    def close(self, a, b, category, tolerance=None):
        tolerance = SCOPE["bookkeeping_tolerance"] if tolerance is None else tolerance
        a, b = np.asarray(a), np.asarray(b)
        self.test(a.shape == b.shape, category+" shape")
        self.test(np.isfinite(a).all() and np.isfinite(b).all(), category+" finite")
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.maximum[category] = max(self.maximum.get(category, 0.), error)
        self.test(error <= tolerance, f"{category}: {error} > {tolerance}")
        return error


def verify_hashes(files, audit):
    for name, digest in files.items():
        audit.test(sha(ROOT/name) == digest, "hash mismatch: "+name)


@lru_cache(None)
def gaussian_rule(order):
    z, weight = roots_hermitenorm(order)
    return z, weight/math.sqrt(2*math.pi)


def independent_kernel(common, d, w):
    """GH96 probabilities with independent Stein-gradient calculation."""
    original = np.asarray(common)
    z = original.ravel()
    e, weights = gaussian_rule(SCOPE["independent_private_GH"])
    result = []
    for wi in w:
        loading = wi/math.sqrt(d-1)
        eta = loading*z[:, None]+math.sqrt(1-loading**2)*e
        p, q = expit(eta), expit(-eta)
        h, complement = p@weights, q@weights
        gradient = (p*q*(z[:, None]-loading*(1-2*p)))@weights/math.sqrt(d-1)
        result.append([np.log(h), np.log(complement), gradient/h, -gradient/complement])
    return tuple(np.asarray([r[i] for r in result]).reshape((2,)+original.shape) for i in range(4))


@lru_cache(None)
def independent_reference(d, w):
    m = d//2
    z, weights = gaussian_rule(SCOPE["independent_outer_GH"])
    lp, lq, gp, gq = independent_kernel(z, d, w)
    n = np.arange(m+1)[:, None]
    combinations = np.array([math.comb(m, k) for k in range(m+1)], dtype=float)[:, None]
    prob, gradient = [], []
    for g in range(2):
        row = combinations*np.exp(n*lp[g]+(m-n)*lq[g])
        prob.append(row)
        gradient.append(row*(n*gp[g]+(m-n)*gq[g]))
    raw = np.einsum("ik,k,jk->ij", prob[0], weights, prob[1], optimize=True)
    draw = np.stack([np.einsum("ik,k,jk->ij", gradient[0], weights, prob[1], optimize=True),
                     np.einsum("ik,k,jk->ij", prob[0], weights, gradient[1], optimize=True)], axis=-1)
    mass = float(raw.sum())
    pmf = raw/mass
    dp = draw/mass-pmf[:, :, None]*draw.sum(axis=(0, 1))/mass
    logchoose = np.log(combinations[:, 0])
    log_pattern = np.log(pmf)-logchoose[:, None]-logchoose[None, :]
    return {"pmf": pmf, "log_pattern": log_pattern, "grad_log_pattern": dp/pmf[:, :, None],
            "raw_mass": mass, "raw_gradient_mass": draw.sum(axis=(0, 1))}


def direct_training(method, d, y, z, e, w):
    """Full coordinates; no group-softplus or count-histogram calculation."""
    m, K = d//2, z.shape[1]
    if method == "raw":
        loading = np.repeat(w, m)/math.sqrt(d-1)
        private = np.sqrt(1-loading**2)
        eta = z[:, :, None]*loading+e*private
        spin_eta = y[:, None, None, :]*eta
        log_terms = log_expit(spin_eta)
        particle_log = log_terms.sum(axis=-1)
        path = (z[:, :, None]-loading/private*e)/math.sqrt(d-1)
        derivative = y[:, None, None, :]*expit(-spin_eta)*path
        scores = np.stack([derivative[:, :, :, g*m:(g+1)*m].sum(axis=-1) for g in range(2)], axis=-1)
    elif method == "col":
        lp, lq, gp, gq = independent_kernel(z, d, w)
        positive = np.repeat(lp, m, axis=0).transpose(1, 2, 0)
        negative = np.repeat(lq, m, axis=0).transpose(1, 2, 0)
        terms = np.where(y[:, None, None, :] > 0, positive, negative)
        particle_log = terms.sum(axis=-1)
        scores = np.stack([np.where(y[:, None, None, g*m:(g+1)*m] > 0,
                                   gp[g][None, :, :, None], gq[g][None, :, :, None]).sum(axis=-1)
                           for g in range(2)], axis=-1)
    else:
        q = independent_reference(d, tuple(w))
        na = (y[:, :m] > 0).sum(axis=1)
        nb = (y[:, m:] > 0).sum(axis=1)
        return (-float(q["log_pattern"][na, nb].mean()),
                -q["grad_log_pattern"][na, nb].mean(axis=0))
    total = logsumexp(particle_log, axis=2)
    posterior = np.exp(particle_log-total[:, :, None])
    return float((math.log(K)-total).mean()), -(posterior[:, :, :, None]*scores).sum(axis=2).mean(axis=(0, 1))


@lru_cache(None)
def canonical_bits(d):
    m = d//2
    return np.array([list(np.arange(m) < a)+list(np.arange(m) < b)
                     for a in range(m+1) for b in range(m+1)], dtype=float)


def direct_confirmation(d, z, e, w, teacher, ell):
    """Explicit full-coordinate log-sigmoid sums, then teacher-PMF weighting."""
    bits = canonical_bits(d)
    lam = np.repeat(w, d//2)/math.sqrt(d-1)
    eta = lam*z[:, :, None]+np.sqrt(1-lam*lam)*e
    logp, logq = log_expit(eta), log_expit(-eta)
    raw_log = np.einsum("bki,ci->bkc", logp, bits, optimize=True)
    raw_log += np.einsum("bki,ci->bkc", logq, 1-bits, optimize=True)
    p = teacher.ravel()
    raw = (math.log(z.shape[1])-logsumexp(raw_log, axis=1))@p
    single = (-raw_log.mean(axis=1))@p
    corrected = raw-single+d*ell
    lp, lq, _, _ = independent_kernel(z, d, w)
    logp = np.repeat(lp, d//2, axis=0).transpose(1, 2, 0)
    logq = np.repeat(lq, d//2, axis=0).transpose(1, 2, 0)
    col_log = np.einsum("bki,ci->bkc", logp, bits, optimize=True)
    col_log += np.einsum("bki,ci->bkc", logq, 1-bits, optimize=True)
    col = (math.log(z.shape[1])-logsumexp(col_log, axis=1))@p
    return {"raw_uncontrolled": raw, "single": single, "raw": corrected, "col": col}


def check_saved_qualification(numeric, cfg, audit):
    count = 0
    for row in numeric:
        k = row["kernel"]
        qualified = k["log_difference"] <= cfg["kernel_log_gate"] and k["log_gradient_difference"] <= cfg["kernel_log_gradient_gate"]
        audit.test(bool(k["qualified"]) == bool(qualified), "kernel original gate")
        audit.test(qualified, "unqualified endpoint retained but RUN cannot be complete")
        if "reference" not in row:
            continue
        count += 1
        r = row["reference"]
        m = r["metrics"]
        checks = {
            "mass": m["PMF_mass_error"] <= cfg["PMF_mass_gate"],
            "gradient_mass": m["PMF_gradient_mass_error"] <= cfg["PMF_gradient_mass_gate"],
            "L1": m["PMF_L1_difference"] <= cfg["PMF_L1_gate"],
            "log": m["PMF_log_difference"] <= cfg["PMF_log_gate"],
            "log_gradient": m["PMF_log_gradient_difference"] <= cfg["PMF_log_gradient_gate"],
            "complement": m["complement_error"] <= cfg["normalization_gate"],
            "risk": m["reference_risk_span"] <= cfg["reference_risk_gate"],
        }
        audit.test(checks == r["checks"] and all(checks.values()) == r["qualified"], "reference original gates")
        audit.test(all(checks.values()), "incomplete numeric qualification")
    return count


def rederive_fit(row, cfg, audit):
    starts = row["starts"]
    audit.test(len(starts) == 9, "nine original starts")
    audit.close([s["start"] for s in starts], list(itertools.product(cfg["starts"], repeat=2)), "start_grid", 0)
    minimum = min(s["objective"] for s in starts)
    selected = next(i for i, s in enumerate(starts) if s["objective"] <= minimum+1e-12)
    audit.test(selected == row["selected_start"], "deterministic selected start")
    audit.close(row["w"], starts[selected]["w"], "selected_parameter", 0)
    audit.close(row["objective"], starts[selected]["objective"], "selected_objective")
    audit.close(row["gradient"], starts[selected]["gradient"], "selected_gradient")
    for candidate in starts:
        point = np.asarray(candidate["w"])
        audit.test(np.isfinite(point).all() and np.all(point >= .5-1e-12) and np.all(point <= 2.5+1e-12),
                   "all-start finite box feasibility")
        audit.test(np.isfinite(candidate["objective"]) and np.isfinite(candidate["gradient"]).all(), "all-start finite output")
    w, gradient = np.array(row["w"]), np.array(row["gradient"])
    audit.test(np.all(w >= .5-1e-12) and np.all(w <= 2.5+1e-12), "selected box feasibility")
    pg = float(np.max(np.abs(w-np.clip(w-gradient, .5, 2.5))))
    fd = max(float(np.max(np.abs(np.asarray(f["gradient"])-gradient))) for f in row["fd"])
    audit.close([x["step"] for x in row["fd"]], cfg["fd_steps"], "FD_steps", 0)
    grid = row["diagnostic_grid"]
    audit.close([r["w"] for r in grid], list(itertools.product(cfg["grid"], repeat=2)), "diagnostic_grid", 0)
    gap = max(0., row["objective"]-min(x["objective"] for x in grid))
    audit.close([pg, fd, gap], [row["pg"], row["fd_error"], row["grid_improvement"]], "diagnostics")
    checks = {"projected_gradient": pg <= cfg["pg_gate"], "finite_difference": fd <= cfg["fd_gate"], "grid": gap <= cfg["grid_gate"]}
    audit.test(row["optimization_checks"] == checks, "original optimization checks")
    audit.test(row["optimization_qualified"] == all(checks.values()), "original optimization qualification")
    audit.test(not row["global_optimality_certified"], "no global certificate")
    audit.test(row["requests"] >= row["actual"] > 0, "objective counters")
    return bool(all(checks.values()))


def interval(values, alpha, family, threshold):
    values = np.array(values, dtype=float)
    mean = float(np.sum(values)/len(values))
    sd = math.sqrt(float(np.sum((values-mean)**2)/(len(values)-1)))
    se = sd/math.sqrt(len(values))
    radius = float(t.isf(alpha/(2*family), len(values)-1))*se
    return {"values": values, "mean": mean, "SE": se, "low": mean-radius, "high": mean+radius,
            "precision": se <= threshold, "alpha": alpha, "family": family, "df": len(values)-1}


def rederive_statistics(result, select, risks, confirmations, cfg, audit):
    counts = {"risk": 0, "score": 0}
    decisions = {"risk": [], "score": []}
    expected_risk_pairs = ("raw-MOM", "raw-col", "raw-quasiMLE", "MOM-quasiMLE", "col-quasiMLE", "MOM-col")
    expected_score_pairs = (("raw", "raw-MOM"), ("raw", "raw-quasiMLE"),
                            ("col", "col-MOM"), ("col", "col-quasiMLE"))
    audit.test({(r["d"], r["N"], r["pair"]) for r in result["risk_contrasts"]}
               == set(itertools.product(cfg["dimensions"], cfg["Ns"], expected_risk_pairs)), "complete risk family keys")
    audit.test({(r["d"], r["N"], r["objective"], r["pair"]) for r in result["score_contrasts"]}
               == {(d, N, objective, pair) for d in cfg["dimensions"] for N in cfg["Ns"]
                   for objective, pair in expected_score_pairs}, "complete score family keys")
    for kind, alpha, family, threshold in (("risk", cfg["risk_alpha"], cfg["risk_family"], cfg["risk_se"]),
                                           ("score", cfg["score_alpha"], cfg["score_family"], cfg["score_se"])):
        for row in result[kind+"_contrasts"]:
            d, N = row["d"], row["N"]
            a, b = row["pair"].split("-")
            values = []
            for s in range(cfg["streams"]):
                if kind == "risk":
                    value = risks[(d, s, N, a)]["joint_KL"]-risks[(d, s, N, b)]["joint_KL"]
                else:
                    cc = confirmations[(d, s)]
                    def score(method):
                        w = select[(d, s, N, method)]["w"]
                        i = next(i for i, q in enumerate(cc["w"]) if np.array_equal(q, w))
                        return float(cc[row["objective"]][i].sum()/cfg["confirm_B"])
                    value = score(a)-score(b)
                values.append(value)
            actual = interval(values, alpha, family, threshold)
            for key in ("values", "mean", "SE", "low", "high"):
                audit.close(actual[key], row[key], "statistics_"+key, SCOPE["statistics_tolerance"])
            for key in ("precision", "alpha", "family", "df"):
                audit.test(actual[key] == row[key], "original statistics field "+key)
            audit.test(row["approximate_t"], "approximate t label retained")
            expected_primary = N == 4096 and (
                row["pair"] in ("raw-MOM", "raw-col", "MOM-quasiMLE", "col-quasiMLE") if kind == "risk"
                else row["objective"] == "raw" and row["pair"] == "raw-MOM")
            audit.test(("main_gate" in row) == expected_primary, "prespecified primary membership")
            if "main_gate" in row:
                main = bool(actual["precision"] and (
                    actual["low"] > .002 if kind == "risk" and a == "raw" else
                    actual["high"] < .001 if kind == "risk" else actual["high"] < -.001))
                audit.test(row["main_gate"] == main, "original primary gate")
                decisions[kind].append(main)
            counts[kind] += 1
    audit.test(counts == {"risk": 24, "score": 16}, "all contrast counts")
    audit.test(decisions["risk"] == result["risk_main_gates"], "all risk gates")
    audit.test(decisions["score"] == result["score_main_gates"], "all score gates")
    audit.test(len(decisions["risk"]) == 8 and len(decisions["score"]) == 2, "all ten gates")
    return {"counts": counts, "main_decisions": decisions}


def execute():
    started = time.perf_counter()
    audit = Audit()
    require_complete = read(RUN/"STATE.json")
    audit.test(require_complete["status"] == "complete", "RUN not complete")
    audit.test(not (OUT/"START.json").exists(), "existing audit attempt preserved; no silent restart")
    freeze, seal = read(RUN/"FREEZE.json"), read(RUN/"RESULTS_FREEZE.json")
    cfg = freeze["config"]
    audit.test(cfg["dimensions"] == [16, 64] and cfg["streams"] == 8 and cfg["Ns"] == [256, 4096], "frozen RUN scope")
    audit.test(cfg["B"] == 64 and cfg["confirm_B"] == 512, "frozen bank scope")
    verify_hashes(freeze["files"], audit)
    verify_hashes(seal["files"], audit)
    audit.test(seal["source_freeze_sha256"] == sha(RUN/"FREEZE.json"), "source-result chain")
    selections_seal = read(RUN/"SELECTION_FREEZE.json")
    verify_hashes(selections_seal["files"], audit)
    audit.test(selections_seal["all_128_selected_before_risk"], "selection stage identity")
    audit.test(all(Path(name).name.startswith(("TRAIN", "SELECT")) for name in selections_seal["files"]), "selection-only seal contents")
    files = {**freeze["files"], **{k: v for k, v in seal["files"].items() if not k.endswith(".npz")},
             rel(RUN/"RESULTS_FREEZE.json"): sha(RUN/"RESULTS_FREEZE.json"),
             rel(Path(__file__)): sha(Path(__file__)), rel(REF/"CONFIG.json"): sha(REF/"CONFIG.json")}
    arrays = {k: v for k, v in seal["files"].items() if k.endswith(".npz")}
    write_json(OUT/"START.json", {"utc": now(), "audit_source_sha256": sha(Path(__file__)),
               "scope": SCOPE, "files": files, "local_arrays": arrays,
               "new_scientific_rng": False, "not_an_optimizer_rerun": True})
    try:
        sys.path.insert(0, str(ROOT))
        producer = importlib.import_module("script.compensation_learning_run")
        audit.test(producer.CFG == cfg, "imported producer matches frozen config")
        alpha = read(MOM)["gh"][-1]["values"]["1.0"]["alpha"]
        ell = read(MOM)["gh"][-1]["values"]["1.0"]["ell"]
        ref_cfg = read(REF/"CONFIG.json")
        result = read(RUN/"RESULTS.json")
        all_select, all_risk, confirmations = {}, {}, {}
        stream_receipts = []
        qualified_count = 0
        direct_candidate_count = 0
        reference_count = 0
        original_endpoint_references = 0
        rng_normals = rng_uniforms = 0
        for d in cfg["dimensions"]:
            m = d//2
            old = read(OLD/f"REFERENCE_d{d}.json")
            teacher = np.array(old["pmf"][0])
            oracle = np.array(old["params"][old["names"].index("comp")]["lambda"])*math.sqrt(d-1)
            count_sum = np.add.outer(np.arange(m+1), np.arange(m+1))
            spin = 2*count_sum-d
            masks = [spin*spin <= d, spin < -math.sqrt(d), spin > math.sqrt(d)]
            for s in range(cfg["streams"]):
                path = RUN/f"d{d}_s{s}"
                with np.load(path/"TRAIN.npz") as arr:
                    y, z, e = (arr[key].copy() for key in ("y", "z", "e"))
                rows, risks = read(path/"SELECTIONS.json"), read(path/"RISKS.json")
                evaluation = read(path/"EVALUATION.json")
                audit.test(len(rows) == 8 and len(risks) == 8, "eight outputs per stream")
                audit.test(y.shape == (4096, d) and z.shape == (64, d) and e.shape == (64, d, d), "training input shapes")
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg["seed"], d, s, 0])))
                common_teacher = rng.standard_normal((4096, 1))
                private_teacher = rng.standard_normal((4096, d))
                rho = .9/(d-1)
                eta_teacher = (math.sqrt(rho)*common_teacher+math.sqrt(1-rho)*private_teacher)*np.repeat([6., 2.], m)
                regenerated = (2*(rng.random((4096, d)) < expit(eta_teacher))-1).astype(np.int8)
                audit.close(y, regenerated, "RNG_training_labels", 0)
                audit.close(z, rng.standard_normal((64, d)), "RNG_training_common", 0)
                audit.close(e, rng.standard_normal((64, d, d)), "RNG_training_private", 0)
                rng_normals += 4096*(d+1)+64*d*(d+1)
                rng_uniforms += 4096*d
                prefix = []
                for row in rows:
                    N, method = row["N"], row["method"]
                    key = (d, s, N, method)
                    audit.test(key not in all_select and method in cfg["methods"] and N in cfg["Ns"], "unique selection")
                    all_select[key] = row
                    if method == "MOM":
                        pair = []
                        for g in range(2):
                            labels = y[:N, g*m:(g+1)*m].astype(float)
                            gram = labels.T@labels/N
                            pair.append(float(gram[np.triu_indices(m, 1)].mean()))
                        squared = (d-1)*np.array(pair)/alpha**2
                        estimate = np.sqrt(np.minimum(np.maximum(squared, .25), 6.25))
                        audit.close(pair, row["moment"]["pair"], "MOM_pairs")
                        audit.close(squared, row["moment"]["untruncated_squared_loading"], "MOM_squared")
                        audit.close(estimate, row["w"], "MOM_parameters")
                    else:
                        qualified_count += rederive_fit(row, cfg, audit)
                        saved = read(path/f"SELECT_N{N}_{method}.json")
                        audit.test(saved == row, "separate selection file")
                    yy, zz, ee = y[:16], z[:2], e[:2]
                    errors = {}
                    for objective in ("raw", "col", "quasiMLE"):
                        expected = direct_training(objective, d, yy, zz, ee, row["w"])
                        actual = producer.Objective(objective, d, yy, zz, ee)(row["w"])
                        errors[objective] = [audit.close(expected[j], actual[j], "prefix_"+objective+("_value" if j == 0 else "_gradient"),
                                                        SCOPE["value_gradient_tolerance"]) for j in (0, 1)]
                    prefix.append({"N": N, "method": method, "w": row["w"], "differences": errors})
                for row in risks:
                    key = (d, s, row["N"], row["method"])
                    audit.close(row["w"], all_select[key]["w"], "risk_selected_parameter", 0)
                    independent = independent_reference(d, tuple(row["w"]))
                    q = independent["pmf"]
                    audit.test(np.all(q > 0) and np.isfinite(q).all(), "independent PMF positivity")
                    audit.close(independent["raw_mass"], 1., "independent_PMF_mass", 1e-10)
                    risk = float(kl_div(teacher, q).sum())
                    audit.close(risk, row["joint_KL"], "independent_reference_KL", SCOPE["risk_event_tolerance"])
                    for j, mask in enumerate(masks):
                        audit.close([float(teacher[mask].sum()), float(q[mask].sum())], [row["events"][j]["P"], row["events"][j]["Q"]],
                                    "independent_reference_events", SCOPE["risk_event_tolerance"])
                    audit.test(row["optimization_qualified"] == all_select[key].get("optimization_qualified", True), "risk qualification annotation")
                    all_risk[key] = row
                    reference_count += 1
                nref = check_saved_qualification(evaluation["numerical"], ref_cfg, audit)
                original_endpoint_references += nref
                audit.test(nref == evaluation["unique_learned_references"], "unique endpoint references")
                audit.test(3*nref == evaluation["reference_variant_integrations"], "original reference calls")
                with np.load(path/"CONFIRM_INPUT.npz") as arr:
                    cz, ce = arr["z"].copy(), arr["e"].copy()
                with np.load(path/"CONFIRM.npz") as arr:
                    cc = {k: arr[k].copy() for k in arr.files}
                expected_parameters = list(dict.fromkeys(tuple(row["w"]) for row in rows))
                if tuple(oracle) not in expected_parameters:
                    expected_parameters.append(tuple(oracle))
                audit.close(cc["w"], expected_parameters, "confirmation_parameter_order", 0)
                audit.test(len(cc["w"]) == evaluation["confirmation_candidates"], "confirmation candidate count")
                audit.test(evaluation["no_selection_after_risk"], "no postrisk selection declaration")
                audit.close(evaluation["oracle_comp"], oracle, "oracle_evaluation_anchor", 0)
                crng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg["seed"], d, s, 1])))
                audit.close(cz, crng.standard_normal((512, d)), "RNG_confirmation_common", 0)
                audit.close(ce, crng.standard_normal((512, d, d)), "RNG_confirmation_private", 0)
                rng_normals += 512*d*(d+1)
                cprefix = []
                for i, w in enumerate(cc["w"]):
                    expected = direct_confirmation(d, cz[:2], ce[:2], w, teacher, ell)
                    errors = {objective: audit.close(expected[objective], cc[objective][i, :2], "confirm_prefix_"+objective,
                                                     SCOPE["value_gradient_tolerance"]) for objective in ("raw", "col")}
                    cprefix.append({"w": w, "first_two_raw_uncontrolled": expected["raw_uncontrolled"],
                                    "first_two_single": expected["single"], "differences": errors})
                    direct_candidate_count += 1
                audit.test(cc["raw"].shape == cc["col"].shape == (len(cc["w"]), 512), "all-bank score array dimensions")
                audit.test(np.isfinite(cc["raw"]).all() and np.isfinite(cc["col"]).all(), "all-bank finite scores")
                confirmations[(d, s)] = cc
                stream_receipts.append({"d": d, "stream": s, "training_prefix": prefix,
                                        "confirmation_prefix": cprefix, "all_RNG_inputs_match": True,
                                        "all_original_scores_recomputed": False})
                write_json(OUT/f"d{d}_s{s}.json", stream_receipts[-1])
                print(f"AUDIT d{d} s{s} passed", flush=True)
        audit.test(len(all_select) == 128 and len(all_risk) == 128 and reference_count == 128, "all endpoints")
        audit.test(len(result["risks"]) == 128 and {(r["d"], r["stream"], r["N"], r["method"]) for r in result["risks"]}
                   == set(all_risk), "aggregate risk coverage")
        expected_rows = [all_risk[(r["d"], r["stream"], r["N"], r["method"])] for r in result["risks"]]
        audit.test(expected_rows == result["risks"], "aggregate risks")
        audit.test(result["optimization_qualified"] == qualified_count, "optimization qualification count")
        audit.test(result["continuous_procedures"] == 96 and result["MOM_outputs"] == 32 and result["optimizer_invocations"] == 864,
                   "procedure ledger")
        audit.test(result["endpoint_reference_integrations"] == 3*original_endpoint_references,
                   "producer endpoint integration ledger")
        for key in ("requests", "actual"):
            field = "objective_requests" if key == "requests" else "actual_objective_evaluations"
            audit.test(result[field] == sum(r.get(key, 0) for r in all_select.values()), "objective count "+key)
        stat = rederive_statistics(result, all_select, all_risk, confirmations, cfg, audit)
        ledger = {"new_training_vectors": 2*8*4096, "new_training_bits": 8*4096*(16+64),
                  "training_prior_vectors": 8*64*(16+64), "confirm_prior_vectors": 8*512*(16+64),
                  "total_prior_vectors": 8*(64+512)*(16+64), "independent_streams": 16,
                  "new_deployment_labels": 0, "new_NN": 0}
        for key, value in ledger.items():
            audit.test(result[key] == value, "resource ledger "+key)
        audit.test(result["selection_seal_sha256"] == sha(RUN/"SELECTION_FREEZE.json"), "selection-result chain")
        receipt = {"passed": True, "utc": now(), "scope": SCOPE, "checks": audit.count,
                   "maximum_differences": audit.maximum, "audit_source_sha256": sha(Path(__file__)),
                   "files": files, "local_arrays": arrays,
                   "source_freeze_sha256": sha(RUN/"FREEZE.json"),
                   "results_freeze_sha256": sha(RUN/"RESULTS_FREEZE.json"),
                   "selection_freeze_sha256": sha(RUN/"SELECTION_FREEZE.json"),
                   "all_endpoints_checked": 128, "MOM_outputs": 32, "fit_selection_rules": 96,
                   "all_solver_runs_refitted": False, "optimizer_calls": 0,
                   "sampled_training_parameter_checks": 128,
                   "sampled_training_objective_gradient_checks": 128*3,
                   "all_reference_risks_rebuilt": 128,
                   "producer_endpoint_reference_integrations": 3*original_endpoint_references,
                   "unique_independent_reference_integrals": independent_reference.cache_info().misses,
                   "confirmation_candidate_prefixes": direct_candidate_count,
                   "all_confirmation_banks_aggregated": True, "all_confirmation_scores_recomputed": False,
                   "RNG_replayed_normal_scalars": rng_normals, "RNG_replayed_uniform_scalars": rng_uniforms,
                   "RNG_replay_is_not_new_independent_scientific_sampling": True,
                   "statistics": stat, "ledger": ledger, "elapsed_seconds": time.perf_counter()-started,
                   "optimization_qualified": qualified_count,
                   "saved_numerical_gate_metrics_rederived_not_all_three_variants_reintegrated": True,
                   "strict_integration_certificate": False,
                   "audit_artifacts": {rel(p): sha(p) for p in OUT.glob("*.json")}}
        write_json(OUT/"RECEIPT.json", receipt)
        print(json.dumps({"passed": True, "receipt_sha256": sha(OUT/"RECEIPT.json")}))
    except Exception as exc:
        write_json(OUT/"FAILURE.json", {"utc": now(), "passed": False, "error": repr(exc),
                   "traceback": traceback.format_exc(), "checks_completed": audit.count,
                   "maximum_differences": audit.maximum, "audit_source_sha256": sha(Path(__file__))})
        raise


def check(workspace=False):
    receipt = read(OUT/"RECEIPT.json")
    audit = Audit()
    audit.test(receipt["passed"], "audit was not passed")
    verify_hashes(receipt["files"], audit)
    verify_hashes(receipt["audit_artifacts"], audit)
    audit.test(receipt["audit_source_sha256"] == sha(Path(__file__)), "audit source identity")
    if workspace:
        verify_hashes(receipt["local_arrays"], audit)
    print(json.dumps({"passed": True, "workspace": workspace, "read_only_receipt_check": True,
                      "RNG_or_likelihood_not_replayed_by_check": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.action == "write":
        execute()
    else:
        check(args.workspace)
