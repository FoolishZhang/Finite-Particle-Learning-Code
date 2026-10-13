"""Frozen active-scale boundary: quadrature references and independent prior banks.

This producer never trains, samples observations, modifies an old package, or
calls the native SCBM API. All Monte Carlo scores are stable mathematical
float64. Independent native/API and reference checks belong to the audit.

Scientific commands require SPEC.json and a matching FREEZE.json. Each command
creates an exclusive execution receipt before doing scientific work; an old or
failed receipt must not be removed to retry a scientific unit.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from scipy import __version__ as scipy_version
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/active_scale_boundary"
PATTERNS = ((np.arange(16)[:, None] >> np.array([3, 2, 1, 0])) & 1).astype(np.int8)
PAIR_ORDER = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_mixed = np.eye(4)
for _pair, _rho in zip(PAIR_ORDER, [.25, -.15, .1, .1, -.15, .25]):
    _mixed[_pair] = _mixed[_pair[::-1]] = _rho
_q = np.array([1., 1., -1., -1.])
SPEC = {
    "package": "active_scale_boundary",
    "mean": [-.6, -.2, .2, .6],
    "teachers": [
        {"name": "independent", "covariance": np.eye(4).tolist()},
        {"name": "mixed", "covariance": _mixed.tolist()},
        {"name": "equicorrelation", "covariance": (.6 * np.eye(4) + .4 * np.ones((4, 4))).tolist()},
        {"name": "signed", "covariance": (.15 * np.eye(4) + .85 * np.outer(_q, _q)).tolist()},
    ],
    "delta": .1,
    "t_values": [0., .1, .2, .4],
    "primary_t": .2,
    "patterns": PATTERNS.tolist(),
    "quadrature": {"orders": [24, 32, 40], "main_order": 40, "comparison_order": 32,
                   "chunk": 65536, "max_abs_PMF_change": 2e-8,
                   "max_abs_KL_change": 2e-7, "mass_tolerance": 2e-12},
    "mc": {"K": 100, "nuisance_values": [0, 4, 16, 64, 256],
           "streams": 16, "banks_per_stream": 2048, "chunk": 32,
           "seed_base": 202610044000, "bit_generator": "PCG64DXSM",
           "roles": {"nuisance": 0, "active": 1}, "saved_prefix_banks": 2,
           "canonical_nuisance_label": 1, "dtype": "float64", "BLAS_threads": 1},
    "inference": {
        "collision": {"alpha": .025, "q": 4, "method": "bounded two-sided Hoeffding",
                      "n0": "exact 1/K; not counted in q"},
        "raw": {"alpha": .025, "q": 8, "df": 15, "primary_nuisance_values": [0, 256],
                "max_SE": .00015, "method": "Bonferroni two-sided development Student t",
                "dependent_expected_signs": {"0": "negative", "256": "positive"},
                "null_role": "control only; no directional gate"},
    },
    "scope": {"new_observations": 0, "new_fits": 0, "new_NN_training": 0,
              "native_API_calls": 0,
              "main_target": "expected finite-particle log score, mathematical float64",
              "teacher_weights": "positive order-40 Gaussian-logistic PMF normalized by its raw mass",
              "candidate_rule": "top R_Y eigenvector, free-C sign attainment, nonnegative unit scale direction",
              "null_rule": "analytic R_Y=I, unit top vector (1,1,1,1)/2",
              "stop": "fixed budgets; preserve failures; no automatic retry or added samples"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    record = {"python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy_version, "platform": platform.platform(),
              "thread_environment": {k: os.environ[k] for k in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        from threadpoolctl import threadpool_info
        record["threadpool_info"] = threadpool_info()
    except ImportError:
        record["threadpool_info"] = None
    return record


def contract():
    spec = read(OUT / "SPEC.json")
    frozen = read(OUT / "FREEZE.json")
    if spec != SPEC:
        raise ValueError("SPEC.json does not equal the source-defined pre-science contract")
    if frozen.get("source_sha256") != sha(SOURCE):
        raise ValueError("Producer source does not match FREEZE.json")
    if frozen.get("spec_sha256") != sha(OUT / "SPEC.json"):
        raise ValueError("SPEC.json does not match FREEZE.json")
    if frozen.get("science_authorized") is not True:
        raise ValueError("FREEZE.json has not authorized scientific execution")
    for name, digest in frozen.get("input_sha256", {}).items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return spec, frozen


def begin(unit):
    spec, frozen = contract()
    unit.mkdir(parents=True, exist_ok=True)
    receipt = {"started_utc": utc_now(), "source_sha256": sha(SOURCE),
               "spec_sha256": sha(OUT / "SPEC.json"),
               "freeze_sha256": sha(OUT / "FREEZE.json"), "environment": environment()}
    save(unit / "EXECUTION.json", receipt)
    return spec, frozen, receipt


def fail(unit):
    save(unit / "FAILURE.json", {"failed_utc": utc_now(), "traceback": traceback.format_exc(),
                                 "policy": "No retry, removal, or replacement is authorized by this producer."})


def normalized_probability(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (16,) or not np.all(np.isfinite(raw)) or not np.all(raw > 0.):
        raise ValueError("PMF must contain 16 finite, strictly positive values")
    mass = float(raw.sum())
    return raw / mass, mass


def integrate_pmf(mean, factor, order, chunk=65536):
    """Tensor Gauss-Hermite for mean + epsilon @ factor.T, with no RNG."""
    mean, factor = np.asarray(mean, dtype=np.float64), np.asarray(factor, dtype=np.float64)
    if mean.shape != (4,) or factor.shape != (4, 4):
        raise ValueError("Expected four-dimensional mean and Gaussian factor")
    nodes, weights = roots_hermitenorm(order)
    weights = weights / np.sqrt(2. * np.pi)
    accumulated = np.zeros(16, dtype=np.longdouble)
    for start in range(0, order ** 4, chunk):
        flat = np.arange(start, min(start + chunk, order ** 4))
        ids = np.stack(np.unravel_index(flat, (order,) * 4), axis=1)
        logits = nodes[ids] @ factor.T + mean
        weight = np.prod(weights[ids], axis=1)
        plus, minus = expit(logits), expit(-logits)
        for i, pattern in enumerate(PATTERNS):
            accumulated[i] += np.sum(weight * np.prod(np.where(pattern, plus, minus), axis=1),
                                     dtype=np.longdouble)
    raw = np.asarray(accumulated, dtype=np.float64)
    probability, mass = normalized_probability(raw)
    return {"order": order, "Gaussian_nodes": order ** 4, "raw_pmf": raw.tolist(),
            "raw_mass": mass, "pmf": probability.tolist(),
            "normalization_applied": True, "mass_error": abs(mass - 1.)}


def kl(probability, candidate):
    probability, candidate = np.asarray(probability), np.asarray(candidate)
    return float(np.sum(probability * (np.log(probability) - np.log(candidate))))


def moments_and_direction(probability, name, delta=.1):
    """Direction fixed by the numeric teacher, never selected by risk scores."""
    probability = np.asarray(probability, dtype=np.float64)
    p = probability @ PATTERNS
    centered = PATTERNS - p
    gamma_numerical = centered.T @ (probability[:, None] * centered)
    d = p * (1. - p)
    ry_numerical = gamma_numerical / np.sqrt(d[:, None] * d[None, :])
    if name == "independent":
        gamma, ry = np.diag(d), np.eye(4)
        eigenvalue, vector = 1., np.ones(4) / 2.
    else:
        gamma, ry = gamma_numerical, ry_numerical
        eigenvalues, eigenvectors = np.linalg.eigh(ry)
        eigenvalue, vector = float(eigenvalues[-1]), eigenvectors[:, -1]
        if vector[np.flatnonzero(np.abs(vector) > 1e-15)[0]] < 0.:
            vector = -vector
    sign = np.where(vector >= 0., 1., -1.)
    cdir = delta * np.eye(4) + (1. - delta) * np.outer(sign, sign)
    unnormalized = np.abs(vector) / np.sqrt(d)
    norm_squared = float(unnormalized @ unnormalized)
    direction = unnormalized / np.sqrt(norm_squared)
    b = float(delta + (1. - delta) * eigenvalue)
    mu0 = np.log(p) - np.log1p(-p)
    product = np.prod(np.where(PATTERNS, p, 1. - p), axis=1)
    product /= product.sum()
    return {"p": p.tolist(), "Gamma": gamma.tolist(), "R_Y": ry.tolist(),
            "numerical_Gamma_before_null_identity": gamma_numerical.tolist(),
            "numerical_R_Y_before_null_identity": ry_numerical.tolist(),
            "null_identity_correction_max_abs": float(np.max(np.abs(ry_numerical - ry))),
            "lambda_max_R_Y": eigenvalue, "top_eigenvector": vector.tolist(),
            "B": b, "collision_threshold": float(1. - 1. / b),
            "normalization_squared": norm_squared, "scale_direction": direction.tolist(),
            "direction_sign": sign.tolist(), "direction_correlation": cdir.tolist(),
            "direction_correlation_eigenvalues": np.linalg.eigvalsh(cdir).tolist(),
            "mu0": mu0.tolist(), "product_pmf": product.tolist(),
            "product_KL": kl(probability, product),
            "exact_scale_curvature": float((1. - b) / norm_squared),
            "derivation_scope": "Expected transverse scale curvature at collapsed product; C unidentifiable there."
            }


def run_reference():
    unit = OUT / "REFERENCE"
    _, _, receipt = begin(unit)
    started = time.perf_counter()
    try:
        records, qualifications = [], []
        qcfg = SPEC["quadrature"]
        for teacher in SPEC["teachers"]:
            name = teacher["name"]
            covariance = np.array(teacher["covariance"], dtype=np.float64)
            variants = [integrate_pmf(SPEC["mean"], np.linalg.cholesky(covariance), order, qcfg["chunk"])
                        for order in qcfg["orders"]]
            by_order = {row["order"]: row for row in variants}
            p = np.array(by_order[40]["pmf"])
            p32 = np.array(by_order[32]["pmf"])
            derived = moments_and_direction(p, name, SPEC["delta"])
            teacher_change = float(np.max(np.abs(p - p32)))
            teacher_mass_passed = all(row["mass_error"] <= qcfg["mass_tolerance"] for row in variants)
            qualifications.append({"teacher": name, "object": "teacher", "t": None,
                                   "PMF_change_40_minus_32": teacher_change,
                                   "PMF_change_40_minus_24_descriptive": float(np.max(np.abs(p - by_order[24]["pmf"]))),
                                   "mass_passed": teacher_mass_passed,
                                   "passed": bool(teacher_change <= qcfg["max_abs_PMF_change"] and teacher_mass_passed)})
            factor = np.linalg.cholesky(np.array(derived["direction_correlation"]))
            base = np.array(derived["product_pmf"])
            base_risk = kl(p, base)
            candidates = [{"t": 0., "pmf": base.tolist(), "raw_mass": float(base.sum()),
                           "method": "analytic product at zero scale", "Gaussian_nodes": 0,
                           "joint_KL": base_risk, "KL_minus_product": 0., "variants": []}]
            for tvalue in SPEC["t_values"][1:]:
                scale = tvalue * np.array(derived["scale_direction"])
                candidate_variants = [integrate_pmf(derived["mu0"], scale[:, None] * factor,
                                                     order, qcfg["chunk"]) for order in qcfg["orders"]]
                cb = {row["order"]: row for row in candidate_variants}
                cq40, cq32 = np.array(cb[40]["pmf"]), np.array(cb[32]["pmf"])
                risk = kl(p, cq40)
                diff = risk - base_risk
                # Both candidate-only and teacher+candidate quadrature sensitivity are explicit.
                candidate_risk_change = abs(risk - kl(p, cq32))
                total_risk_change = abs(risk - kl(p32, cq32))
                total_contrast_change = abs(diff - (kl(p32, cq32) - kl(p32, base)))
                pmf_change = float(np.max(np.abs(cq40 - cq32)))
                mass_passed = all(row["mass_error"] <= qcfg["mass_tolerance"] for row in candidate_variants)
                qualified = (pmf_change <= qcfg["max_abs_PMF_change"] and
                             max(candidate_risk_change, total_risk_change, total_contrast_change) <= qcfg["max_abs_KL_change"]
                             and mass_passed)
                qualifications.append({"teacher": name, "object": "candidate", "t": tvalue,
                                       "PMF_change_40_minus_32": pmf_change,
                                       "PMF_change_40_minus_24_descriptive": float(np.max(np.abs(cq40 - cb[24]["pmf"]))),
                                       "KL_change_candidate_only": candidate_risk_change,
                                       "KL_change_teacher_and_candidate": total_risk_change,
                                       "KL_contrast_change_teacher_and_candidate": total_contrast_change,
                                       "mass_passed": mass_passed, "passed": bool(qualified)})
                candidates.append({"t": tvalue, "scale": scale.tolist(), "pmf": cq40.tolist(),
                                   "raw_mass": cb[40]["raw_mass"], "joint_KL": risk,
                                   "KL_minus_product": diff, "variants": candidate_variants})
            records.append({**teacher, "teacher_variants": variants, "teacher_pmf": p.tolist(),
                            "teacher_raw_mass": by_order[40]["raw_mass"],
                            "derived": derived, "candidates": candidates})
        integrations = len(SPEC["teachers"]) * len(SPEC["t_values"]) * len(qcfg["orders"])
        results = {"source_sha256": receipt["source_sha256"], "freeze_sha256": receipt["freeze_sha256"],
                   "patterns": PATTERNS.tolist(), "records": records, "qualifications": qualifications,
                   "qualification_passed": all(q["passed"] for q in qualifications),
                   "accounting": {"PMF_integrations": integrations, "scalar_pattern_integrals": 16 * integrations,
                                  "Gaussian_nodes": len(records) * len(SPEC["t_values"]) * sum(o ** 4 for o in qcfg["orders"]),
                                  "analytic_product_PMFS": len(records), "new_random_draws": 0,
                                  "new_observations": 0, "new_fits": 0},
                   "seconds": time.perf_counter() - started,
                   "scope": "Numerical quadrature qualification, not a certified integration remainder bound. P is the normalized order-40 reference."}
        save(unit / "RESULTS.json", results)
        print(json.dumps({"phase": "reference", "passed": results["qualification_passed"],
                          "seconds": results["seconds"], "accounting": results["accounting"]}))
    except Exception:
        fail(unit)
        raise


def evaluate_block(active_eps, nuisance_eps, reference):
    """Returns contrast[bank,teacher,t,n], collision[bank,n], no random draw.

    The stable score is -E_P log(sum pi exp(log A_t-log A_0)).  This
    analytically cancels the nuisance score and collapsed active score before
    reduction, rather than subtracting two large joint NLLs.
    """
    active_eps = np.asarray(active_eps, dtype=np.float64)
    nuisance_eps = np.asarray(nuisance_eps, dtype=np.float64)
    cfg = SPEC["mc"]
    if (active_eps.ndim != 3 or active_eps.shape[1:] != (cfg["K"], 4) or
            nuisance_eps.shape != (len(active_eps), cfg["K"], max(cfg["nuisance_values"]))):
        raise ValueError("Unexpected active or nuisance shape")
    banks = len(active_eps)
    cumulative = np.cumsum(-np.logaddexp(0., -nuisance_eps), axis=2)
    noise_log = np.stack([np.zeros((banks, cfg["K"])) if n == 0 else cumulative[:, :, n - 1]
                          for n in cfg["nuisance_values"]], axis=1)
    logpi = noise_log - logsumexp(noise_log, axis=2, keepdims=True)
    collision = np.exp(2. * logpi).sum(axis=2)
    collision[:, 0] = 1. / cfg["K"]
    contrasts = np.zeros((banks, len(reference["records"]), len(SPEC["t_values"]), len(cfg["nuisance_values"])))
    for ti, teacher in enumerate(reference["records"]):
        derived = teacher["derived"]
        mu0 = np.array(derived["mu0"])
        direction = np.array(derived["scale_direction"])
        normal = active_eps @ np.linalg.cholesky(np.array(derived["direction_correlation"])).T
        shift = normal * direction
        p = np.array(teacher["teacher_pmf"])
        for si, scale in enumerate(SPEC["t_values"][1:], start=1):
            xshift = scale * shift
            # log A_t-log A_0: no large common terms or nuisance constants.
            ratio = (xshift @ PATTERNS.T -
                     (np.logaddexp(0., mu0 + xshift) - np.logaddexp(0., mu0)).sum(axis=2)[:, :, None])
            for ni in range(len(cfg["nuisance_values"])):
                log_average = logsumexp(logpi[:, ni, :, None] + ratio, axis=1)
                contrasts[:, ti, si, ni] = -log_average @ p
    if not np.isfinite(contrasts).all() or not np.isfinite(collision).all():
        raise FloatingPointError("Nonfinite MC score or collision")
    return contrasts, collision


def stream_rng(stream, role):
    """Two separate, named SeedSequence children; no accidental prefix reuse."""
    sequence = np.random.SeedSequence(SPEC["mc"]["seed_base"] + stream, spawn_key=(role,))
    return np.random.Generator(np.random.PCG64DXSM(sequence))


def run_mc(stream):
    cfg = SPEC["mc"]
    if not 0 <= stream < cfg["streams"]:
        raise ValueError("Stream outside frozen panel")
    unit = OUT / "MC" / f"stream_{stream:02d}"
    _, _, receipt = begin(unit)
    started = time.perf_counter()
    try:
        reference = read(OUT / "REFERENCE/RESULTS.json")
        if reference["source_sha256"] != sha(SOURCE) or reference["freeze_sha256"] != sha(OUT / "FREEZE.json"):
            raise ValueError("Reference is not from this frozen producer")
        if not reference["qualification_passed"]:
            raise ValueError("Reference qualification failed; MC must not proceed")
        banks, k, nmax = cfg["banks_per_stream"], cfg["K"], max(cfg["nuisance_values"])
        nuisance_rng, active_rng = stream_rng(stream, 0), stream_rng(stream, 1)
        nuisance_hash, active_hash = hashlib.sha256(), hashlib.sha256()
        contrasts = np.empty((banks, len(reference["records"]), len(SPEC["t_values"]), len(cfg["nuisance_values"])))
        collisions = np.empty((banks, len(cfg["nuisance_values"])))
        prefix_active, prefix_nuisance = None, None
        maximum_active, maximum_nuisance = 0., 0.
        for start in range(0, banks, cfg["chunk"]):
            count = min(cfg["chunk"], banks - start)
            nuisance = nuisance_rng.standard_normal((count, k, nmax), dtype=np.float64)
            active = active_rng.standard_normal((count, k, 4), dtype=np.float64)
            nuisance_hash.update(nuisance.tobytes(order="C"))
            active_hash.update(active.tobytes(order="C"))
            if start == 0:
                prefix_active = active[:cfg["saved_prefix_banks"]].copy()
                prefix_nuisance = nuisance[:cfg["saved_prefix_banks"]].copy()
            values, collision = evaluate_block(active, nuisance, reference)
            contrasts[start:start + count], collisions[start:start + count] = values, collision
            maximum_active = max(maximum_active, float(np.max(np.abs(active))))
            maximum_nuisance = max(maximum_nuisance, float(np.max(np.abs(nuisance))))
        output = unit / "BANKS.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, contrasts=contrasts, collisions=collisions,
                                prefix_active=prefix_active, prefix_nuisance=prefix_nuisance)
        record = {"stream": stream, "source_sha256": sha(SOURCE), "freeze_sha256": receipt["freeze_sha256"],
                  "reference_sha256": sha(OUT / "REFERENCE/RESULTS.json"),
                  "seed": {"entropy": cfg["seed_base"] + stream, "bit_generator": cfg["bit_generator"],
                           "nuisance_spawn_key": [0], "active_spawn_key": [1]},
                  "input_sha256": {"nuisance_C_order_float64": nuisance_hash.hexdigest(),
                                   "active_C_order_float64": active_hash.hexdigest()},
                  "npz_sha256": sha(output), "array_sha256": {"contrasts": array_sha(contrasts),
                                                               "collisions": array_sha(collisions)},
                  "contrast_axis_order": ["bank", "teacher", "t", "nuisance"],
                  "collision_axis_order": ["bank", "nuisance"],
                  "teacher_order": [r["name"] for r in reference["records"]],
                  "t_values": SPEC["t_values"], "nuisance_values": cfg["nuisance_values"],
                  "mean_contrasts": contrasts.mean(axis=0).tolist(),
                  "mean_collisions": collisions.mean(axis=0).tolist(),
                  "maximum_absolute_active_base_normal": maximum_active,
                  "maximum_absolute_nuisance_normal": maximum_nuisance,
                  "accounting": {"banks": banks, "prior_vectors": banks * k,
                                 "active_Gaussian": banks * k * 4, "nuisance_Gaussian": banks * k * nmax,
                                 "total_Gaussian": banks * k * (4 + nmax),
                                 "canonical_patterns_not_sampled": 16,
                                 "active_likelihood_ratios": banks * k * len(reference["records"]) * 3 * 16,
                                 "weighted_log_score_cells": banks * len(reference["records"]) * 3 * len(cfg["nuisance_values"]) * 16,
                                 "new_observations": 0, "fits": 0, "NN_training": 0, "native_API_calls": 0},
                  "seconds": time.perf_counter() - started,
                  "scope": "One independent prior stream; shared inputs across teachers/t/n counted once. No new observed data or fitted model."}
        save(unit / "RESULTS.json", record)
        print(json.dumps({"phase": "mc", "stream": stream, "seconds": record["seconds"],
                          "npz_sha256": record["npz_sha256"]}))
    except Exception:
        fail(unit)
        raise


def curvature(collision, derived):
    return float((1. - (1. - collision) * derived["B"]) / derived["normalization_squared"])


def summarize():
    unit = OUT / "SUMMARY"
    _, _, receipt = begin(unit)
    started = time.perf_counter()
    try:
        reference = read(OUT / "REFERENCE/RESULTS.json")
        if not reference["qualification_passed"] or reference["source_sha256"] != sha(SOURCE):
            raise ValueError("Reference not qualified under this producer")
        cfg, infer = SPEC["mc"], SPEC["inference"]
        stream_records, all_contrasts, all_collisions = [], [], []
        for stream in range(cfg["streams"]):
            folder = OUT / "MC" / f"stream_{stream:02d}"
            row = read(folder / "RESULTS.json")
            if (row["stream"] != stream or row["source_sha256"] != sha(SOURCE) or
                    row["freeze_sha256"] != receipt["freeze_sha256"] or
                    row["reference_sha256"] != sha(OUT / "REFERENCE/RESULTS.json") or
                    row["npz_sha256"] != sha(folder / "BANKS.npz")):
                raise ValueError(f"Stream {stream} provenance mismatch")
            with np.load(folder / "BANKS.npz", allow_pickle=False) as archive:
                contrast, collision = archive["contrasts"], archive["collisions"]
                if row["array_sha256"] != {"contrasts": array_sha(contrast), "collisions": array_sha(collision)}:
                    raise ValueError("Array hash mismatch")
                if contrast.shape != (cfg["banks_per_stream"], 4, 4, 5) or collision.shape != (cfg["banks_per_stream"], 5):
                    raise ValueError("Incomplete panel")
                all_contrasts.append(contrast.mean(axis=0))
                all_collisions.append(collision)
            stream_records.append({"stream": stream, "metadata_sha256": sha(folder / "RESULTS.json"),
                                   "npz_sha256": row["npz_sha256"]})
        means = np.stack(all_contrasts)
        chi = np.concatenate(all_collisions)
        total_banks = cfg["streams"] * cfg["banks_per_stream"]
        hcfg, rcfg = infer["collision"], infer["raw"]
        half = (1. - 1. / cfg["K"]) * math.sqrt(math.log(2. * hcfg["q"] / hcfg["alpha"]) / (2. * total_banks))
        collision_rows = []
        for ni, n in enumerate(cfg["nuisance_values"]):
            estimate = 1. / cfg["K"] if n == 0 else float(chi[:, ni].mean())
            lower, upper = ((estimate, estimate) if n == 0 else
                            (max(1. / cfg["K"], estimate - half), min(1., estimate + half)))
            collision_rows.append({"nuisance": n, "mean": estimate, "lower": lower, "upper": upper,
                                   "halfwidth_before_clipping": 0. if n == 0 else half,
                                   "exact": n == 0,
                                   "curvatures": [{"teacher": r["name"], "mean": curvature(estimate, r["derived"]),
                                                   "lower": curvature(lower, r["derived"]),
                                                   "upper": curvature(upper, r["derived"]),
                                                   "exact_risk_curvature": r["derived"]["exact_scale_curvature"],
                                                   "raw_uniform_C_strict_min_qualified": bool((1. - lower) * r["derived"]["B"] < 1.),
                                                   "raw_constructed_descent_qualified": bool((1. - upper) * r["derived"]["B"] > 1.)}
                                                  for r in reference["records"]]})
        multiplier = float(student_t.ppf(1. - rcfg["alpha"] / (2. * rcfg["q"]), rcfg["df"]))
        primary, descriptive = [], []
        for ti, teacher in enumerate(reference["records"]):
            for si, tvalue in enumerate(SPEC["t_values"]):
                for ni, n in enumerate(cfg["nuisance_values"]):
                    values = means[:, ti, si, ni]
                    estimate, se = float(values.mean()), float(values.std(ddof=1) / math.sqrt(cfg["streams"]))
                    record = {"teacher": teacher["name"], "t": tvalue, "nuisance": n,
                              "raw_score_contrast": estimate, "SE_across_16_stream_means": se,
                              "stream_means": values.tolist(),
                              "reference_KL_contrast": teacher["candidates"][si]["KL_minus_product"]}
                    if tvalue == SPEC["primary_t"] and n in rcfg["primary_nuisance_values"]:
                        record.update(lower=estimate - multiplier * se, upper=estimate + multiplier * se,
                                      precision_passed=bool(se <= rcfg["max_SE"]), max_SE=rcfg["max_SE"])
                        if teacher["name"] == "independent":
                            record.update(role="null control", direction_gate=None, direction_passed=None)
                        else:
                            expected = rcfg["dependent_expected_signs"][str(n)]
                            passed = record["upper"] < 0. if expected == "negative" else record["lower"] > 0.
                            record.update(role="dependent direction", direction_gate=expected, direction_passed=bool(passed))
                        primary.append(record)
                    else:
                        descriptive.append(record)
        results = {"source_sha256": sha(SOURCE), "freeze_sha256": receipt["freeze_sha256"],
                   "reference_sha256": sha(OUT / "REFERENCE/RESULTS.json"), "streams": stream_records,
                   "collision_family": {**hcfg, "independent_banks": total_banks,
                                        "halfwidth_before_clipping": half, "results": collision_rows,
                                        "scope": "Finite-sample Hoeffding statement for exact real-arithmetic collision of iid banks; numerical controls are separate. Derived curvature holds conditional on numeric teacher moments."},
                   "raw_primary_family": {**rcfg, "critical_t": multiplier, "results": primary,
                                          "direction_passed": sum(r["direction_passed"] is True for r in primary),
                                          "direction_total": 6,
                                          "precision_passed": sum(r["precision_passed"] for r in primary),
                                          "precision_total": 8,
                                          "scope": "Development t approximation over 16 independent prior streams; fixed numerical teachers, no observed-data or training variability; quadrature error excluded."},
                   "descriptive_curves": descriptive,
                   "accounting": {"new_prior_banks": total_banks, "prior_vectors": total_banks * cfg["K"],
                                  "active_Gaussian": total_banks * cfg["K"] * 4,
                                  "nuisance_Gaussian": total_banks * cfg["K"] * max(cfg["nuisance_values"]),
                                  "total_Gaussian": total_banks * cfg["K"] * (4 + max(cfg["nuisance_values"])),
                                  "new_observations": 0, "new_fits": 0, "NN_training": 0, "native_API_calls": 0},
                   "seconds": time.perf_counter() - started}
        save(unit / "RESULTS.json", results)
        print(json.dumps({"phase": "summary", "directions": results["raw_primary_family"]["direction_passed"],
                          "directions_total": 6, "precision": results["raw_primary_family"]["precision_passed"],
                          "precision_total": 8, "accounting": results["accounting"]}))
    except Exception:
        fail(unit)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["schema", "reference", "mc", "summarize"])
    parser.add_argument("--stream", type=int)
    args = parser.parse_args()
    if args.mode == "schema":
        print(json.dumps({"source_sha256": sha(SOURCE), "spec": SPEC}, ensure_ascii=False, indent=2))
    elif args.mode == "reference":
        run_reference()
    elif args.mode == "mc":
        if args.stream is None:
            parser.error("mc requires --stream")
        run_mc(args.stream)
    else:
        summarize()


if __name__ == "__main__":
    main()
