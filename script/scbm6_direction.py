"""Frozen six-coordinate Price/overlap decomposition, independent of training."""
from __future__ import annotations

import os
for _option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_option] = "1"
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import hashlib
import json
import math
import platform
import time
import traceback
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t
import torch
import scbmeval_kernel as legacy

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/scbm_multicorrelation/DIRECTION"
DESIGN = ROOT / "docs/08_research_研究工作/六相关自由学习与方向机制_20261003/EXPERIMENT_DESIGN.md"
TRUTH = ROOT / "artifacts/scbm_invariance/REFERENCE/MODEL.json"
LEGACY_SHA = "9dc59d1436abeb1ce5eaf4c70a098a6f34c4ced1d797492c0c56ff5eee8e64d1"
TRUTH_SHA = "2b5d12858dd6d581f36ded2327fd2ce26f0bc69eed314360783479403b9c9975"
MU = legacy.MU.copy()
CTRUTH = legacy.C0.copy()
PATTERNS = legacy.PATTERNS.copy()
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
MATRICES = ["identity", "truth"]
PROFILES = ["scale1", "scale02", "FACT"]
SCALAR_METRICS = ["score", "centered_score", "overlap", "mean_ess", "mean_inverse_ess"]
VECTOR_METRICS = ["gradient", "mean_mark_term", "coupling_term", "markmc"]
METRICS = SCALAR_METRICS + VECTOR_METRICS
CONFIG = {
    "replicates": 32, "banks_per_replicate": 512, "K": 100, "nuisance": 256,
    "chunk_banks": 64, "saved_prefix_banks": 4, "workers": 2,
    "seed_base": 202610031200, "roles": {"active": 0, "nuisance": 1},
    "random_shape_order": "[bank, particle, coordinate], C-contiguous float64",
    "matrices": MATRICES, "profiles": PROFILES, "scales": [1., .2, 0.],
    "pairs": [list(p) for p in PAIRS], "canonical_nuisance_label": 0,
    "mean": MU.tolist(), "truth_covariance": CTRUTH.tolist(),
    "dtype": "float64", "pattern_weights": "old truth32 PMF without renormalization",
    "family_size": 18, "family_alpha": .05, "df": 31, "SE_threshold": 5e-5,
    "primary_order": ["signed_scale1", "signed_scale1_minus_FACT", "signed_scale02_minus_scale1"],
    "decomposition_tolerance": 1e-10, "normalization_tolerance": 1e-10,
    "new_observations": 0, "new_fits": 0, "new_integrals": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def reference():
    assert sha(TRUTH) == TRUTH_SHA
    model = read(TRUTH)
    assert model["patterns"] == PATTERNS.tolist()
    assert model["config"]["mean"] == MU.tolist()
    assert model["config"]["covariance"] == CTRUTH.tolist()
    p = np.asarray(next(r["main_pmf"] for r in model["records"] if r["t"] == 0.), dtype=float)
    assert np.all(p > 0.) and abs(float(p.sum()) - 1.) < 2e-12
    spins = 2. * PATTERNS - 1.
    mean = p @ spins
    second = np.asarray([p @ (spins[:, i] * spins[:, j]) for i, j in PAIRS])
    covariance = second - np.asarray([mean[i] * mean[j] for i, j in PAIRS])
    centers = np.stack((covariance / 4., covariance / 2.))
    return {
        "weights": p.tolist(), "weight_sum": float(p.sum()),
        "spin_means": mean.tolist(), "spin_pair_moments": second.tolist(),
        "spin_covariances": covariance.tolist(), "mean_mark_reference": centers.tolist(),
        "truth_reference_sha256": sha(TRUTH),
        "scope": "Numerically qualified unconditional centers over labels and candidate prior, not per-pattern conditional means. No normalization or new model integration. Any fixed center gives exact decomposition algebra."}


def provenance():
    assert sha(legacy.__file__) == LEGACY_SHA
    old = legacy.provenance()
    return {
        "source_sha256": sha(SOURCE), "legacy_math_path": rel(legacy.__file__),
        "legacy_math_sha256": LEGACY_SHA, "official_adapter": old["official"],
        "reused_function": "scbmeval_kernel.prepare_noise only in main mathematics",
        "main_api_calls": 0,
        "scope": "Main DIRECTION is mathematical float64; official code provenance is bound, not a claim of main API32 evaluation. Training is a different authorized program."}


def environment():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
        "thread_environment": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "actual_BLAS_threads_measured": False,
    }


def active_terms(eps, covariance):
    eps = np.asarray(eps, dtype=float)
    assert eps.ndim == 3 and eps.shape[2] == 4
    x = eps @ np.linalg.cholesky(covariance).T + MU
    logactive = x @ PATTERNS.T - np.logaddexp(0., x).sum(axis=2)[:, :, None]
    u = PATTERNS[None, None, :, :] - expit(x)[:, :, None, :]
    marks = np.stack([u[..., i] * u[..., j] for i, j in PAIRS], axis=-1)
    return {"logits": x, "logactive": logactive, "marks": marks}


def from_terms(terms, centered_noise, center, n):
    logweight = terms["logactive"] + centered_noise[:, :, None]
    normalizer = logsumexp(logweight, axis=1)
    pi = np.exp(logweight - normalizer[:, None, :])
    gamma = pi * (1. - pi)
    overlap = gamma.sum(axis=1)
    inverse_ess = np.square(pi).sum(axis=1)
    centered = math.log(pi.shape[1]) - normalizer
    marks = terms["marks"]
    gradient = -np.sum(gamma[:, :, :, None] * marks, axis=1)
    mean_term = -overlap[:, :, None] * center[None, None, :]
    coupling = -np.sum(gamma[:, :, :, None] * (marks - center[None, None, None, :]), axis=1)
    result = {
        "score": centered + n * math.log(2.), "centered_score": centered,
        "overlap": overlap, "mean_ess": 1. / inverse_ess, "mean_inverse_ess": inverse_ess,
        "gradient": gradient, "mean_mark_term": mean_term, "coupling_term": coupling,
        "markmc": marks.mean(axis=1),
    }
    diagnostics = {
        "maximum_decomposition_error": float(np.max(abs(gradient - mean_term - coupling))),
        "maximum_weight_sum_error": float(np.max(abs(pi.sum(axis=1) - 1.))),
        "maximum_overlap_identity_error": float(np.max(abs(overlap + inverse_ess - 1.))),
        "minimum_ESS": float(np.min(1. / inverse_ess)), "maximum_ESS": float(np.max(1. / inverse_ess)),
        "maximum_absolute_active_logit": float(np.max(abs(terms["logits"]))),
    }
    assert all(np.isfinite(a).all() for a in result.values())
    assert diagnostics["maximum_decomposition_error"] <= CONFIG["decomposition_tolerance"]
    assert diagnostics["maximum_weight_sum_error"] <= CONFIG["normalization_tolerance"]
    assert diagnostics["maximum_overlap_identity_error"] <= CONFIG["normalization_tolerance"]
    return result, diagnostics


def evaluate_block(active, noise, ref):
    assert active.shape[:2] == noise.shape[:2] and active.shape[2] == 4
    cache = legacy.prepare_noise(noise, np.ones(noise.shape[2]), alphas=(1., .2, 0.))
    centers = np.asarray(ref["mean_mark_reference"])
    collected = {key: [] for key in METRICS}
    diagnostics = []
    for ci, covariance in enumerate([np.eye(4), CTRUTH]):
        terms = active_terms(active, covariance)
        profiles = {key: [] for key in METRICS}
        for pi, scale in enumerate(CONFIG["scales"]):
            values, diagnostic = from_terms(terms, cache["centered_logweights"][scale],
                                            centers[ci], noise.shape[2])
            for key in METRICS:
                profiles[key].append(values[key])
            diagnostics.append({"matrix": MATRICES[ci], "profile": PROFILES[pi], **diagnostic})
        for key in METRICS:
            collected[key].append(np.stack(profiles[key]))
    return {key: np.stack(value) for key, value in collected.items()}, diagnostics


def weight_patterns(pattern_means, weights):
    # Input scalar [2,3,16], vector [2,3,16,6]; weights are not silently normalized.
    p = np.asarray(weights)
    return {key: (np.einsum("cmp,p->cm", value, p) if key in SCALAR_METRICS
                  else np.einsum("cmpq,p->cmq", value, p))
            for key, value in pattern_means.items()}


def make_primary(records):
    gradients = np.asarray([r["means"]["gradient"] for r in records], dtype=float)
    assert gradients.shape == (32, 2, 3, 6)
    sign = np.sign([CTRUTH[i, j] for i, j in PAIRS])
    critical = float(student_t.ppf(1. - .05 / (2 * 18), 31))
    answer = []
    definitions = [
        ("signed_scale1", gradients[:, 1, 0, :] * sign, "negative"),
        ("signed_scale1_minus_FACT", (gradients[:, 1, 0, :] - gradients[:, 1, 2, :]) * sign, "negative"),
        ("signed_scale02_minus_scale1", (gradients[:, 1, 1, :] - gradients[:, 1, 0, :]) * sign, "positive")]
    for name, values, expected in definitions:
        for j, pair in enumerate(PAIRS):
            v = values[:, j]
            mean = float(v.mean())
            se = float(v.std(ddof=1) / math.sqrt(32))
            lower, upper = mean - critical * se, mean + critical * se
            answer.append({
                "name": name, "pair": list(pair), "truth_sign": int(sign[j]),
                "replicate_values": v.tolist(), "mean": mean, "SE": se, "lower": lower, "upper": upper,
                "replicates": 32, "df": 31, "critical_t": critical, "expected": expected,
                "effect_passed": bool(upper < 0 if expected == "negative" else lower > 0),
                "SE_threshold": 5e-5, "precision_passed": bool(se <= 5e-5)})
    assert len(answer) == 18
    return answer


def selftest():
    assert not (OUT / "EXECUTION.json").exists()
    prov = provenance()
    ref = reference()
    B, K, n = 3, 7, 5
    active = (1.1 * np.sin(np.arange(B * K * 4) * .113)).reshape(B, K, 4)
    noise = (.9 * np.cos(np.arange(B * K * n) * .017)).reshape(B, K, n)
    all_values, diagnostics = evaluate_block(active, noise, ref)
    centers = np.asarray(ref["mean_mark_reference"])
    weights = np.asarray(ref["weights"])
    pmean = {key: value.mean(axis=2) for key, value in all_values.items()}
    weighted = weight_patterns(pmean, weights)
    assert np.allclose(weighted["gradient"], weighted["mean_mark_term"] + weighted["coupling_term"], atol=1e-13, rtol=0)
    assert np.allclose(weighted["overlap"] + weighted["mean_inverse_ess"], weights.sum(), atol=1e-13, rtol=0)
    assert np.array_equal(all_values["markmc"][:, 0], all_values["markmc"][:, 1])
    assert np.array_equal(all_values["markmc"][:, 0], all_values["markmc"][:, 2])
    for ci in range(2):
        direct = -logsumexp(active_terms(active, [np.eye(4), CTRUTH][ci])["logactive"], axis=1) + math.log(K)
        assert np.allclose(all_values["centered_score"][ci, 2], direct, atol=1e-13, rtol=0)
    hessian_errors = []
    path_fd_errors = []
    cache = legacy.prepare_noise(noise, np.ones(n), alphas=(1., .2, 0.))
    for ci, covariance in enumerate([np.eye(4), CTRUTH]):
        initial = active_terms(active[:1], covariance)["logits"][0]
        for scale_index, scale in enumerate(CONFIG["scales"]):
            lognu = cache["centered_logweights"][scale][0]
            for pattern_index in [0, 5, 10, 15]:
                y = torch.tensor(PATTERNS[pattern_index], dtype=torch.float64)
                x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
                loss = -torch.logsumexp((x * y - torch.nn.functional.softplus(x)).sum(dim=1)
                                       + torch.tensor(lognu), dim=0) + math.log(K)
                first = torch.autograd.grad(loss, x, create_graph=True)[0]
                for pair_index, (i, j) in enumerate(PAIRS):
                    mixed = []
                    for k in range(K):
                        second = torch.autograd.grad(first[k, i], x, retain_graph=True)[0]
                        mixed.append(float(second[k, j]))
                    expected = all_values["gradient"][ci, scale_index, 0, pattern_index, pair_index]
                    hessian_errors.append(abs(sum(mixed) - expected))
            # Differentiate the fixed-bank Cholesky path and compare with finite differences.
            rho0 = np.asarray([covariance[i, j] for i, j in PAIRS])
            rho = torch.tensor(rho0, dtype=torch.float64, requires_grad=True)
            basis = torch.zeros((6, 4, 4), dtype=torch.float64)
            for q, (i, j) in enumerate(PAIRS):
                basis[q, i, j] = basis[q, j, i] = 1.
            matrix = torch.eye(4, dtype=torch.float64) + torch.einsum("q,qij->ij", rho, basis)
            x = torch.tensor(active, dtype=torch.float64) @ torch.linalg.cholesky(matrix).T + torch.tensor(MU)
            patterns = torch.tensor(PATTERNS, dtype=torch.float64)
            logactive = x @ patterns.T - torch.nn.functional.softplus(x).sum(dim=2)[:, :, None]
            logweight = logactive + torch.tensor(cache["centered_logweights"][scale])[:, :, None]
            loss = ((math.log(K) - torch.logsumexp(logweight, dim=1)) @ torch.tensor(weights)).mean()
            gradient = torch.autograd.grad(loss, rho)[0].detach().numpy()
            for q, (i, j) in enumerate(PAIRS):
                h = 1e-5
                plus, minus = covariance.copy(), covariance.copy()
                plus[i, j] += h; plus[j, i] += h
                minus[i, j] -= h; minus[j, i] -= h
                values = []
                for c in [plus, minus]:
                    term = active_terms(active, c)
                    out, _ = from_terms(term, cache["centered_logweights"][scale], centers[ci], n)
                    values.append(float((out["centered_score"] @ weights).mean()))
                path_fd_errors.append(abs((values[0] - values[1]) / (2 * h) - gradient[q]))
    assert max(hessian_errors) < 5e-12
    assert max(path_fd_errors) < 2e-8
    record = {
        "passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(SOURCE), "design_sha256": sha(DESIGN), "config": CONFIG,
        "provenance": prov, "reference": ref, "environment": environment(),
        "deterministic_shapes": {"banks": B, "K": K, "nuisance": n},
        "hessian_pair_checks": len(hessian_errors), "maximum_hessian_error": max(hessian_errors),
        "path_finite_difference_checks": len(path_fd_errors), "maximum_path_finite_difference_error": max(path_fd_errors),
        "maximum_decomposition_error": max(d["maximum_decomposition_error"] for d in diagnostics),
        "Price_path_bankwise_equality_asserted": False,
        "new_scientific_RNG": 0, "new_model_integrals": 0, "new_fits": 0,
        "scope": "Fixed sin/cos arrays; logit mixed-Hessian identity and separate path finite differences, not a population or convergence pilot."}
    save(OUT / "SELFTEST.json", record)
    print(json.dumps({k: record[k] for k in ["passed", "source_sha256", "hessian_pair_checks",
          "maximum_hessian_error", "path_finite_difference_checks", "maximum_path_finite_difference_error"]}))


def benchmark():
    # Fixed full production shape, no random generator and no scientific target summaries.
    ref = reference()
    B, K, n = 64, 100, 256
    active = np.sin(np.arange(B * K * 4, dtype=float) * .071).reshape(B, K, 4)
    noise = np.cos(np.arange(B * K * n, dtype=float) * .013).reshape(B, K, n)
    start = time.perf_counter()
    result, _ = evaluate_block(active, noise, ref)
    elapsed = time.perf_counter() - start
    save(OUT / "BENCHMARK.json", {
        "source_sha256": sha(SOURCE), "seconds_one_fixed_block": elapsed,
        "scalar_shape": list(result["score"].shape), "vector_shape": list(result["gradient"].shape),
        "environment": environment(), "new_scientific_RNG": 0, "new_model_integrals": 0,
        "scope": "One deterministic full-shape timing; no target scores or convergence pilot."})
    print(json.dumps({"seconds": elapsed, "new_scientific_RNG": 0}))


def freeze():
    test = read(OUT / "SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(SOURCE)
    assert test["config"] == CONFIG
    assert not (OUT / "EXECUTION.json").exists()
    prov = provenance()
    files = [SOURCE, Path(legacy.__file__), Path(legacy.official_adapter.__file__),
             legacy.official_adapter.OFFICIAL_LOSSES, legacy.official_adapter.OFFICIAL_MODELS,
             TRUTH, DESIGN, OUT / "SELFTEST.json"]
    if (OUT / "BENCHMARK.json").exists():
        assert read(OUT / "BENCHMARK.json")["source_sha256"] == sha(SOURCE)
        files.append(OUT / "BENCHMARK.json")
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
        "config": CONFIG, "reference": reference(), "provenance": prov, "environment": environment(),
        "inputs": {rel(p): sha(p) for p in files}, "scientific_run_authorized": True,
        "automatic_retry": False,
    }
    save(OUT / "FREEZE.json", record)
    print(json.dumps({"frozen": True, "source_sha256": sha(SOURCE), "freeze_sha256": sha(OUT / "FREEZE.json")}))


def validate_freeze():
    frozen = read(OUT / "FREEZE.json")
    assert frozen["source_sha256"] == sha(SOURCE) and frozen["config"] == CONFIG
    assert frozen["scientific_run_authorized"] is True
    for path, digest in frozen["inputs"].items():
        assert sha(ROOT / path) == digest, path
    assert frozen["reference"] == reference()
    provenance()
    return frozen


def run_replicate(replicate):
    frozen = validate_freeze()
    record_path = OUT / f"replicate_{replicate:02}.json"
    assert not record_path.exists()
    assert not (OUT / f"ERROR_replicate_{replicate:02}.json").exists()
    assert not (OUT / f"arrays_{replicate:02}.npz").exists()
    ref = frozen["reference"]
    rng = {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [CONFIG["seed_base"], replicate, role]))) for name, role in CONFIG["roles"].items()}
    initial = {name: generator.bit_generator.state for name, generator in rng.items()}
    digests = {name: hashlib.sha256() for name in rng}
    blocks = {key: [] for key in METRICS}
    prefix = {}
    diagnostics = []
    consumed = 0
    start = time.perf_counter()
    try:
        for block in range(8):
            active = rng["active"].standard_normal((64, 100, 4), dtype=np.float64)
            noise = rng["nuisance"].standard_normal((64, 100, 256), dtype=np.float64)
            consumed += 64
            digests["active"].update(active.tobytes(order="C"))
            digests["nuisance"].update(noise.tobytes(order="C"))
            values, diagnostic = evaluate_block(active, noise, ref)
            diagnostics.extend(diagnostic)
            for key in METRICS:
                blocks[key].append(values[key].mean(axis=2))
            if block == 0:
                prefix["active"] = active[:4].copy()
                prefix["nuisance"] = noise[:4].copy()
                for key in METRICS:
                    prefix["pattern_" + key] = values[key][:, :, :4].copy()
        block_arrays = {key: np.stack(value) for key, value in blocks.items()}
        pattern_means = {key: value.mean(axis=0) for key, value in block_arrays.items()}
        means = weight_patterns(pattern_means, ref["weights"])
        raw_mass = ref["weight_sum"]
        overlap_identity = np.max(abs(means["overlap"] + means["mean_inverse_ess"] - raw_mass))
        decomposition_error = np.max(abs(means["gradient"] - means["mean_mark_term"] - means["coupling_term"]))
        assert overlap_identity <= CONFIG["normalization_tolerance"]
        assert decomposition_error <= CONFIG["decomposition_tolerance"]
        for key, value in block_arrays.items():
            prefix["block_" + key] = value
        npz = OUT / f"arrays_{replicate:02}.npz"
        assert not npz.exists()
        np.savez_compressed(npz, **prefix)
        output = {
            "replicate": replicate, "source_sha256": sha(SOURCE), "freeze_sha256": sha(OUT / "FREEZE.json"),
            "arrays_path": rel(npz), "arrays_sha256": sha(npz),
            "role_entropy": {name: [CONFIG["seed_base"], replicate, role] for name, role in CONFIG["roles"].items()},
            "RNG_initial_states": initial, "RNG_final_states": {name: g.bit_generator.state for name, g in rng.items()},
            "role_input_sha256": {name: h.hexdigest() for name, h in digests.items()},
            "prefix_active_sha256": array_sha(prefix["active"]),
            "prefix_nuisance_sha256": array_sha(prefix["nuisance"]),
            "consumed_banks": consumed, "completed_blocks": 8,
            "pattern_means": {key: value.tolist() for key, value in pattern_means.items()},
            "means": {key: value.tolist() for key, value in means.items()},
            "overlap_identities": {
                "raw_pattern_weight_mass": raw_mass,
                "mass_minus_mean_inverse_ESS": (raw_mass - means["mean_inverse_ess"]).tolist(),
                "one_minus_mean_inverse_ESS": (1. - means["mean_inverse_ess"]).tolist(),
                "one_minus_inverse_mean_ESS_different_quantity": (1. - 1. / means["mean_ess"]).tolist(),
                "maximum_actual_identity_error": float(overlap_identity),
                "scope": "Full active+nuisance likelihood weights; EESS and E(1/ESS) differ. Raw numeric label weights retained."},
            "maximum_decomposition_error": float(decomposition_error),
            "maximum_bank_decomposition_error": max(d["maximum_decomposition_error"] for d in diagnostics),
            "maximum_bank_weight_sum_error": max(d["maximum_weight_sum_error"] for d in diagnostics),
            "maximum_bank_overlap_error": max(d["maximum_overlap_identity_error"] for d in diagnostics),
            "maximum_absolute_active_logit": max(d["maximum_absolute_active_logit"] for d in diagnostics),
            "seconds": time.perf_counter() - start,
        }
        validate_freeze()
        save(record_path, output)
        return output
    except BaseException:
        partial = OUT / f"partial_{replicate:02}.npz"
        if blocks["score"] and not partial.exists():
            np.savez_compressed(partial, **{key: np.stack(value) for key, value in blocks.items()})
        save(OUT / f"ERROR_replicate_{replicate:02}.json", {
            "replicate": replicate, "traceback": traceback.format_exc(),
            "consumed_banks": consumed, "completed_blocks": len(blocks["score"]),
            "RNG_initial_states": initial, "RNG_current_states": {name: g.bit_generator.state for name, g in rng.items()},
            "input_hashes_so_far": {name: h.hexdigest() for name, h in digests.items()},
            "source_sha256": sha(SOURCE), "automatic_retry": False})
        raise


def run():
    frozen = validate_freeze()
    assert not (OUT / "EXECUTION.json").exists()
    assert not list(OUT.glob("replicate_*.json")) and not list(OUT.glob("arrays_*.npz"))
    save(OUT / "EXECUTION.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
        "freeze_sha256": sha(OUT / "FREEZE.json"), "automatic_retry": False})
    start = time.perf_counter()
    records = {}
    pool = ProcessPoolExecutor(max_workers=2)
    active = {}
    next_id = 0
    try:
        while next_id < 2:
            active[pool.submit(run_replicate, next_id)] = next_id
            next_id += 1
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            fresh = [future.result() for future in completed]
            for future in completed:
                del active[future]
            for result in fresh:
                records[result["replicate"]] = result
                print(json.dumps({"completed_replicate": result["replicate"], "seconds": result["seconds"]}), flush=True)
            save(OUT / "PROGRESS.json", {"completed": sorted(records), "running": sorted(active.values()),
                 "seconds": time.perf_counter() - start}, False)
            while len(active) < 2 and next_id < 32:
                active[pool.submit(run_replicate, next_id)] = next_id
                next_id += 1
        ordered = [records[i] for i in range(32)]
        primary = make_primary(ordered)
        descriptive = {}
        for key in METRICS:
            value = np.asarray([row["means"][key] for row in ordered], dtype=float)
            descriptive[key] = {"mean": value.mean(axis=0).tolist(), "SE": (value.std(axis=0, ddof=1) / math.sqrt(32)).tolist()}
        gradients = np.asarray([row["means"]["gradient"] for row in ordered])
        residual = gradients[:, 1, 1] - gradients[:, 1, 2]
        result = {
            "source_sha256": sha(SOURCE), "freeze_sha256": sha(OUT / "FREEZE.json"),
            "design_sha256": sha(DESIGN), "config": CONFIG, "reference": frozen["reference"],
            "records": ordered, "primary_family": primary,
            "primary_effects_passed": sum(p["effect_passed"] for p in primary),
            "primary_precision_passed": sum(p["precision_passed"] for p in primary),
            "descriptive": descriptive,
            "truth_scale02_minus_FACT_descriptive": {"mean": residual.mean(axis=0).tolist(),
                 "SE": (residual.std(axis=0, ddof=1) / math.sqrt(32)).tolist(), "new_family": False},
            "ledger": {"independent_MC_groups": 32, "new_base_banks": 16384, "new_prior_vectors": 1638400,
                 "Gaussian_scalars": 425984000, "active_Gaussian_scalars": 6553600,
                 "nuisance_Gaussian_scalars": 419430400, "new_observations": 0,
                 "new_fits": 0, "new_model_integrals": 0, "main_official_API_calls": 0,
                 "saved_prefix_complete_banks": 128, "saved_block_pattern_mean_arrays": 32 * 8},
            "seconds": time.perf_counter() - start,
            "scope": "Independent mathematical float64 mechanism experiment, no learning or labels. Twelve relative and six absolute signed comparisons form one 18-test development-t family, separate from training risks. Small uniform scale .2 is not a learned FREE panel."}
        validate_freeze()
        save(OUT / "RESULTS.json", result)
        print(json.dumps({"completed": True, "effects_passed": result["primary_effects_passed"],
             "precision_passed": result["primary_precision_passed"], "family_size": 18,
             "seconds": result["seconds"]}), flush=True)
    except BaseException:
        for future in active:
            future.cancel()
        save(OUT / "ERROR.json", {"traceback": traceback.format_exc(), "completed_replicates": sorted(records),
             "already_active_replicates": sorted(active.values()), "automatic_retry": False})
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["schema", "check", "bench", "freeze", "run"])
    args = parser.parse_args()
    if args.mode == "schema":
        print(json.dumps({"config": CONFIG, "provenance": provenance(), "reference": reference(),
                          "environment": environment()}, ensure_ascii=False))
    else:
        {"check": selftest, "bench": benchmark, "freeze": freeze, "run": run}[args.mode]()
