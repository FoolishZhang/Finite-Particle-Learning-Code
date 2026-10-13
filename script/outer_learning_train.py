"""True shared-global-bank finite-grid ERM; science requires an explicit freeze."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np
import scipy
from scipy.special import log_expit

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/外层精度与有限样本伪相关_20261001"
OUT = ROOT / "artifacts/nuisance_learning"
BASE = "33f2d1c5378e269563a96cbfdd6eb0b2c32d9a7b"
OLD = ROOT / "artifacts/nuisance_dimensions/NUMERIC"
RHO = [3 * i / 20 for i in range(-5, 6)]
ZERO = 5
CONFIG = {
    "schema": 1, "a": 1.0, "theta": 0.0, "R": 0.75,
    "rho_grid": RHO, "grid_integer_coordinates": list(range(-5, 6)),
    "streams": 16, "n_values": [0, 254], "Ks": [2, 4],
    "Ns": [64, 256, 1024], "Bs": [32, 512, 8192],
    "seed": 2037100101, "roles": {"labels": 0, "prior": 1},
    "bank_chunk": 128, "near_tie_diagnostic_tolerance": 1e-12,
    "selection_tie_rule": "lowest index among exactly equal computed minima; near ties do not alter selection",
    "primary": {"n": 254, "N": 1024, "B": 8192, "K": 4,
                "integer_event": "raw_grid_coordinate^2 - RB_grid_coordinate^2 >= 4",
                "guaranteed_true_KL_gain_exceeds": 0.001,
                "null_success_probability": [1, 2], "alpha": [1, 20]},
    "endpoint_CP": {"Ks": [2, 4], "n": 254, "N": 1024, "B": 8192,
                    "family_alpha": [1, 20], "individual_alpha": [1, 40], "dyadic_bits": 40},
    "concentration_illustration_delta": 0.05,
    "new_reference_integrals": 0, "continuous_optimizations": 0,
}


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


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=serial)
        f.write("\n")


def fraction_record(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "float_display": float(value)}


def from_fraction_record(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def choose(curve):
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (len(RHO),) or not np.all(np.isfinite(curve)):
        raise ArithmeticError("Invalid objective curve.")
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    ties = np.flatnonzero(curve == minimum).tolist()
    near = np.flatnonzero(curve <= minimum + CONFIG["near_tie_diagnostic_tolerance"]).tolist()
    return {"index": index, "grid_coordinate": index - ZERO, "rho": RHO[index],
            "minimum_contrast": minimum, "exact_ties": ties,
            "near_ties_diagnostic": near, "near_ties_used_for_selection": False}


def active_log_table(prior_active, rhos=RHO):
    """[rho, actual binary active pattern, bank, particle], not label-averaged."""
    z1, z2 = prior_active[..., 0], prior_active[..., 1]
    result = []
    for rho in rhos:
        x2 = rho * z1 + math.sqrt(1 - rho * rho) * z2
        lp1, lq1, lp2, lq2 = log_expit(z1), log_expit(-z1), log_expit(x2), log_expit(-x2)
        result.append(np.stack([lq1 + lq2, lq1 + lp2, lp1 + lq2, lp1 + lp2]))
    return np.asarray(result)


def losses_K2_K4(log_values):
    """Stable original negative logmeans; K2 uses first two of each K4 bank."""
    pair01 = np.logaddexp(log_values[..., 0], log_values[..., 1])
    pair23 = np.logaddexp(log_values[..., 2], log_values[..., 3])
    return np.stack([math.log(2) - pair01,
                     math.log(4) - np.logaddexp(pair01, pair23)], axis=-1)


def rb_bank_contrasts(labels, active_table, Ns):
    patterns = 2 * labels[:, 0].astype(np.int64) + labels[:, 1]
    losses = losses_K2_K4(active_table)  # [rho, pattern, bank, Kchoice]
    contrasts = losses - losses[ZERO][None, ...]
    all_contrasts, references, counts = [], [], []
    for N in Ns:
        count = np.bincount(patterns[:N], minlength=4)
        weight = count / N
        all_contrasts.append(np.einsum("p,gpbk->kgb", weight, contrasts))
        references.append(np.einsum("p,pbk->kb", weight, losses[ZERO]))
        counts.append(count)
    return np.asarray(all_contrasts), np.asarray(references), np.asarray(counts)


def raw_bank_contrasts(labels, prior, active_table, Ns, chunk):
    """Use actual Y_noise rows and global full prior banks, without sign absorption."""
    Nmax, Bmax, Kmax, dim = len(labels), len(prior), prior.shape[1], prior.shape[2]
    assert Kmax == 4 and labels.shape[1] == dim
    spin = 2.0 * labels.astype(np.float64) - 1.0
    patterns = 2 * labels[:, 0].astype(np.int64) + labels[:, 1]
    n = dim - 2
    out = np.zeros((len(Ns), 2, len(RHO), Bmax))
    reference = np.empty((len(Ns), 2, Bmax))
    for start in range(0, Bmax, chunk):
        stop = min(start + chunk, Bmax)
        bb = stop - start
        if n:
            noise = prior[start:stop, :, 2:]
            flat = np.ascontiguousarray(noise).reshape(bb * Kmax, n)
            nuisance = (0.5 * spin[:, 2:] @ flat.T).reshape(Nmax, bb, Kmax)
            nuisance -= np.logaddexp(noise / 2, -noise / 2).sum(axis=2)[None, ...]
        else:
            nuisance = np.zeros((Nmax, bb, Kmax))
        base = losses_K2_K4(nuisance + active_table[ZERO, :, start:stop, :][patterns])
        for ni, N in enumerate(Ns):
            reference[ni, :, start:stop] = base[:N].mean(axis=0).T
        for gi in range(len(RHO)):
            if gi == ZERO:
                continue
            actual = losses_K2_K4(nuisance + active_table[gi, :, start:stop, :][patterns])
            # This is a parameter-independent reference subtraction, not a CV.
            difference = actual - base
            for ni, N in enumerate(Ns):
                out[ni, :, gi, start:stop] = difference[:N].mean(axis=0).T
    return out, reference


def table_likelihood_curve(same_count, N, spin_moments):
    p = int(same_count) / int(N)
    r = np.asarray(spin_moments)
    return -p * np.log1p(r) - (1 - p) * np.log1p(-r)


def artificial_checks():
    labels = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 1, 1, 1],
                       [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    prior = (np.arange(3 * 4 * 4).reshape(3, 4, 4) - 23) / 13
    Ns = [1, 3, 5]
    active = active_log_table(prior[:, :, :2])
    raw, ref = raw_bank_contrasts(labels, prior, active, Ns, 2)
    rb, rbref, _ = rb_bank_contrasts(labels, active, Ns)
    raw0, ref0 = raw_bank_contrasts(labels[:, :2], prior[:, :, :2], active, Ns, 2)
    errors, nuisance_row_effects = [], []
    for gi, rho in enumerate(RHO):
        eta = prior.copy()
        eta[..., 1] = rho * prior[..., 0] + math.sqrt(1 - rho * rho) * prior[..., 1]
        lp = log_expit((2 * labels.astype(float) - 1)[:, None, None, :] * eta[None]).sum(-1)
        base_lp = log_expit((2 * labels.astype(float) - 1)[:, None, None, :] * prior[None]).sum(-1)
        for ki, K in enumerate(CONFIG["Ks"]):
            maximum = lp[..., :K].max(axis=-1)
            direct = -maximum - np.log(np.exp(lp[..., :K] - maximum[..., None]).mean(axis=-1))
            bm = base_lp[..., :K].max(axis=-1)
            base = -bm - np.log(np.exp(base_lp[..., :K] - bm[..., None]).mean(axis=-1))
            nuisance_row_effects.append(float(np.max(np.abs((direct - base)[0] - (direct - base)[1]))))
            for ni, N in enumerate(Ns):
                errors.append(float(np.max(np.abs(raw[ni, ki, gi] - (direct[:N] - base[:N]).mean(0)))))
    error = max(errors)
    assert error < 1e-12
    assert np.max(np.abs(raw0 - rb)) < 1e-12
    assert np.max(np.abs(ref0 - rbref)) < 1e-12
    # Same active labels but different nuisance bits genuinely change the row.
    yn = 2 * labels[:, 2:].astype(float) - 1
    scores = 0.5 * yn @ prior[:, :, 2:].reshape(12, 2).T
    assert np.max(abs(scores[0] - scores[1])) > 0.1
    assert max(nuisance_row_effects) > 1e-5
    moment = np.linspace(-.1, .1, len(RHO))
    s = int(np.sum(labels[:, 0] == labels[:, 1]))
    curve = table_likelihood_curve(s, len(labels), moment)
    pattern_sign = (2 * labels[:, 0].astype(float) - 1) * (2 * labels[:, 1].astype(float) - 1)
    direct_curve = -np.log1p(pattern_sign[:, None] * moment[None]).mean(0)
    assert np.max(abs(curve - direct_curve)) < 1e-14
    previous = Fraction(-1)
    for count in range(17):
        lower, _ = cp_lower_exact(count)
        assert previous <= lower < 1
        previous = lower
    for i in range(-5, 6):
        for j in range(-5, 6):
            assert (i * i - j * j >= 4) == (Fraction(i * i - j * j, 3200) > Fraction(1, 1000))
    return {"passed": True, "raw_direct_coordinate_error": error,
            "n0_raw_equals_RB_error": float(np.max(abs(raw0 - rb))),
            "table_likelihood_direct_error": float(np.max(abs(curve - direct_curve))),
            "actual_nuisance_label_axis_checked": True,
            "same_active_different_nuisance_contrast_effect": max(nuisance_row_effects),
            "CP_exact_arithmetic_cases": 17, "integer_improvement_grid_pairs": 121,
            "Gaussian_integrals": 0, "scientific_RNG_calls": 0,
            "scientific_training_fits": 0, "fixture_labels": 5, "fixture_banks": 3}


def derive_reference():
    frozen = read(OLD / "RESULTS_FREEZE.json")
    for name, expected in frozen["files"].items():
        assert sha(OLD / name) == expected, name
    assert read(OLD / "RESULTS.json")["numerically_qualified"]
    grid = read(OLD / "GRID.json")["rows"]
    rows = {r["rho"]: r for r in grid if r["a"] == 1 and r["theta"] == 0 and r["n"] == 0 and r["variant"] == "primary"}
    picked = [rows[rho] for rho in RHO]
    moments = [r["candidate_spin_moment"] for r in picked]
    assert moments[ZERO] == 0 and all(abs(x) < 1 for x in moments)
    return {"rho_grid": RHO, "spin_moments": moments,
            "reference_joint_KL": [r["joint_KL"] for r in picked],
            "a": 1.0, "teacher_theta": 0.0,
            "source_grid_sha256": sha(OLD / "GRID.json"),
            "source_results_freeze_sha256": sha(OLD / "RESULTS_FREEZE.json"),
            "numerically_qualified_not_strict_integral_certificate": True,
            "new_reference_integrals": 0}


def freeze():
    if (OUT / "FREEZE.json").exists():
        raise FileExistsError("Refusing to replace freeze.")
    files = [Path(__file__), DOC / "SHARED_ERM_THEORY.md", DOC / "LEARNING_DESIGN.md",
             OLD / "GRID.json", OLD / "RESULTS.json", OLD / "RESULTS_FREEZE.json",
             ROOT / "docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_THEORY.md",
             ROOT / "docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_REVIEW.md",
             ROOT / "HANDOFF.md"]
    unit = read(OUT / "UNIT_CHECK.json")
    assert unit["passed"] and unit["source_sha256"] == sha(__file__)
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
    save(OUT / "CONFIG.json", CONFIG)
    save(OUT / "REFERENCE.json", derive_reference())
    inputs = []
    for i, p in enumerate(files):
        target = OUT / "source_inputs" / f"{i:02d}_{p.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(p.read_bytes())
        inputs.append({"original": rel(p), "snapshot": rel(target), "sha256": sha(target)})
    save(OUT / "FREEZE.json", {"created": now(), "base_commit": BASE, "inputs": inputs,
                                "config_sha256": sha(OUT / "CONFIG.json"),
                                "reference_sha256": sha(OUT / "REFERENCE.json"),
                                "unit_sha256": sha(OUT / "UNIT_CHECK.json"),
                                "Python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__})


def verify_freeze(live=False):
    receipt = read(OUT / "FREEZE.json")
    assert read(OUT / "CONFIG.json") == CONFIG
    for name, key in [("CONFIG.json", "config_sha256"), ("REFERENCE.json", "reference_sha256"), ("UNIT_CHECK.json", "unit_sha256")]:
        assert sha(OUT / name) == receipt[key]
    for item in receipt["inputs"]:
        assert sha(ROOT / item["snapshot"]) == item["sha256"]
        if live:
            assert sha(ROOT / item["original"]) == item["sha256"]


def train_stream(stream, spin_moments):
    folder = OUT / f"s{stream:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    make_rng = lambda role: np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG["seed"], stream, role])))
    lrng, prng = make_rng(0), make_rng(1)
    initial = {"labels": lrng.bit_generator.state, "prior": prng.bit_generator.state}
    Nmax, Bmax, Kmax, Dmax = max(CONFIG["Ns"]), max(CONFIG["Bs"]), max(CONFIG["Ks"]), max(CONFIG["n_values"]) + 2
    started = time.perf_counter()
    labels = lrng.integers(0, 2, size=(Nmax, Dmax), dtype=np.uint8)
    prior = prng.standard_normal((Bmax, Kmax, Dmax))
    final = {"labels": lrng.bit_generator.state, "prior": prng.bit_generator.state}
    np.savez_compressed(folder / "INPUT.npz", labels=labels, prior=prior)
    active = active_log_table(prior[:, :, :2])
    rb, rb_reference, counts = rb_bank_contrasts(labels, active, CONFIG["Ns"])
    raw, raw_reference = raw_bank_contrasts(labels, prior, active, CONFIG["Ns"], CONFIG["bank_chunk"])
    np.savez_compressed(folder / "BANK_CONTRASTS.npz", raw=raw, RB=rb,
                        raw_rho0=raw_reference, RB_rho0=rb_reference)
    curves = []
    for ni, N in enumerate(CONFIG["Ns"]):
        for ki, K in enumerate(CONFIG["Ks"]):
            for B in CONFIG["Bs"]:
                for method, array, reference in [("raw", raw, raw_reference), ("RB", rb, rb_reference)]:
                    curve = array[ni, ki, :, :B].mean(axis=-1)
                    ref_mean = float(reference[ni, ki, :B].mean())
                    curves.append({"id": f"{method}_N{N}_B{B}_K{K}", "method": method,
                                   "N": N, "B": B, "K": K,
                                   "rho0_objective": ref_mean,
                                   "objective_curve": (curve + ref_mean).tolist(),
                                   "contrast_curve": curve.tolist(),
                                   "objective_scope": "full n254 raw" if method == "raw" else "active score; add n log2 for full RB, identical to n0 raw",
                                   **choose(curve)})
        same = int(np.sum(labels[:N, 0] == labels[:N, 1]))
        curve = table_likelihood_curve(same, N, spin_moments)
        curves.append({"id": f"table_MLE_N{N}", "method": "table_MLE", "N": N,
                       "same_count": same, "contrast_curve": curve.tolist(),
                       "objective_curve": (curve + math.log(4)).tolist(),
                       "objective_scope": "table active likelihood; add n log2 for full observations",
                       **choose(curve)})
    assert len(curves) == 39
    files = {"INPUT.npz": sha(folder / "INPUT.npz"), "BANK_CONTRASTS.npz": sha(folder / "BANK_CONTRASTS.npz")}
    record = {"stream": stream, "active_pattern_counts_by_N": counts.tolist(), "curves": curves,
              "n0_raw_is_exact_RB_reuse": True, "RB_selection_invariant_to_n": True,
              "RNG_initial": initial, "RNG_final": final, "local_arrays": files,
              "wall_seconds": time.perf_counter() - started,
              "ledger": {"raw_RB_grid_optimizations": 36, "table_MLE_training_argmins": 3,
                         "new_label_vectors": Nmax, "new_label_bits": Nmax * Dmax,
                         "global_prior_banks": Bmax, "global_prior_vectors": Bmax * Kmax,
                         "Gaussian_scalars": Bmax * Kmax * Dmax,
                         "nuisance_GEMM_multiply_adds": Nmax * Bmax * Kmax * (Dmax - 2),
                         "nuisance_rows_label_axis_retained": True}}
    save(folder / "TRAINING.json", record)
    print(f"training stream {stream:02d} complete; 39 choices sealed locally", flush=True)
    return record


def integer_binomial_tail(trials, successes, numerator, denominator):
    return sum(math.comb(trials, j) * numerator ** j * (denominator - numerator) ** (trials - j)
               for j in range(successes, trials + 1))


def cp_lower_exact(successes):
    trials, bits, alpha_den = CONFIG["streams"], CONFIG["endpoint_CP"]["dyadic_bits"], 40
    denominator = 1 << bits
    if successes == 0:
        return Fraction(0), {"successes": 0, "trials": trials, "lower_numerator": 0,
                             "dyadic_denominator": denominator, "alpha": [1, alpha_den], "zero_case": True}
    low, high = 0, denominator
    rhs = denominator ** trials
    for _ in range(bits):
        middle = (low + high) // 2
        lhs = alpha_den * integer_binomial_tail(trials, successes, middle, denominator)
        if lhs <= rhs:
            low = middle
        else:
            high = middle
    assert high == low + 1
    lower_lhs = alpha_den * integer_binomial_tail(trials, successes, low, denominator)
    upper_lhs = alpha_den * integer_binomial_tail(trials, successes, high, denominator)
    assert lower_lhs <= rhs < upper_lhs
    return Fraction(low, denominator), {"successes": successes, "trials": trials,
                                       "lower_numerator": low, "upper_numerator": high,
                                       "dyadic_denominator": denominator, "alpha": [1, alpha_den],
                                       "lower_comparison_lhs": str(lower_lhs),
                                       "upper_comparison_lhs": str(upper_lhs), "comparison_rhs": str(rhs),
                                       "bisection_iterations": bits, "zero_case": False}


def enumerate_table_mle(reference):
    moments, risks = np.asarray(reference["spin_moments"]), reference["reference_joint_KL"]
    rows = []
    for N in CONFIG["Ns"]:
        indices = [choose(table_likelihood_curve(s, N, moments))["index"] for s in range(N + 1)]
        square_sum = sum(math.comb(N, s) * (i - ZERO) ** 2 for s, i in enumerate(indices))
        upper = Fraction(9 * square_sum, 12350 * (1 << N))
        mass = [math.comb(N, s) / (1 << N) for s in range(N + 1)]
        rows.append({"N": N, "indices_for_same_count_0_through_N": indices,
                     "exact_binomial_integer_square_sum": str(square_sum),
                     "true_expected_KL_upper": fraction_record(upper),
                     "reference_expected_KL": math.fsum(p * risks[i] for p, i in zip(mass, indices)),
                     "binomial_float_mass_sum": math.fsum(mass),
                     "count_state_argmin_evaluations": N + 1,
                     "not_independent_training_models": True})
    return rows


def contrast_bound(N, B, K, delta=0.05):
    sigma = math.sqrt(2 * (1 - math.sqrt(1 - CONFIG["R"] ** 2)))
    h = sigma * math.sqrt(2 * math.log(3 * B * K / delta))
    L = math.log(12 * len(RHO) / delta)
    tail = K * (h + sigma * sigma / h) * math.exp(-h * h / (2 * sigma * sigma))
    bound = h * (math.sqrt(2 * L / N) + math.sqrt(2 * L / B)) + tail
    return {"N": N, "B": B, "K": K, "delta": delta, "h": h, "L": L,
            "tail_bias_bound": tail, "error_bound_nats": bound,
            "scope": "raw and RB grid jointly for this K; not joint over two K values",
            "diagnostic_only_not_new_gate": True, "finite_precision_not_certified": True}


def collect(training, reference):
    assert (OUT / "SELECTION_FREEZE.json").exists()
    risks = reference["reference_joint_KL"]
    paired = []
    for record in training:
        models = {x["id"]: x for x in record["curves"]}
        for n in CONFIG["n_values"]:
            for N in CONFIG["Ns"]:
                ml = models[f"table_MLE_N{N}"]
                for B in CONFIG["Bs"]:
                    for K in CONFIG["Ks"]:
                        rb = models[f"RB_N{N}_B{B}_K{K}"]
                        raw = rb if n == 0 else models[f"raw_N{N}_B{B}_K{K}"]
                        difference = raw["grid_coordinate"] ** 2 - rb["grid_coordinate"] ** 2
                        paired.append({"stream": record["stream"], "n": n, "N": N, "B": B, "K": K,
                                       "raw_index": raw["index"], "RB_index": rb["index"], "table_MLE_index": ml["index"],
                                       "raw_grid_coordinate": raw["grid_coordinate"], "RB_grid_coordinate": rb["grid_coordinate"],
                                       "raw_reference_KL": risks[raw["index"]], "RB_reference_KL": risks[rb["index"]],
                                       "table_MLE_reference_KL": risks[ml["index"]],
                                       "reference_raw_minus_RB": risks[raw["index"]] - risks[rb["index"]],
                                       "integer_square_gap": difference,
                                       "sufficient_true_KL_gain_gt_001": difference >= 4,
                                       "analytic_gain_lower_if_nonnegative": fraction_record(Fraction(difference, 3200)) if difference >= 0 else None})
    save(OUT / "EVALUATION.json", {"paired_cells": paired, "risk_values_are_qualified_references": True})
    enumeration = enumerate_table_mle(reference)
    save(OUT / "MLE_ENUMERATION.json", {"rows": enumeration, "total_count_states": sum(x["count_state_argmin_evaluations"] for x in enumeration)})
    high = [x for x in paired if all(x[k] == CONFIG["primary"][k] for k in ("n", "N", "B", "K"))]
    assert len(high) == CONFIG["streams"]
    successes = sum(x["sufficient_true_KL_gain_gt_001"] for x in high)
    pvalue = Fraction(sum(math.comb(16, j) for j in range(successes, 17)), 1 << 16)
    primary = {"successes": successes, "trials": 16, "events": [x["sufficient_true_KL_gain_gt_001"] for x in high],
               "integer_square_gaps": [x["integer_square_gap"] for x in high],
               "exact_one_sided_pvalue": fraction_record(pvalue), "alpha": [1, 20],
               "passed": pvalue <= Fraction(1, 20),
               "target": "probability of analytically guaranteed true KL gain >.001 exceeds 1/2; not a mean-gain test"}
    ml_upper = from_fraction_record(next(x for x in enumeration if x["N"] == max(CONFIG["Ns"]))["true_expected_KL_upper"])
    endpoint = []
    for K in CONFIG["endpoint_CP"]["Ks"]:
        rows = [x for x in paired if x["n"] == 254 and x["N"] == 1024 and x["B"] == 8192 and x["K"] == K]
        events = [abs(x["raw_grid_coordinate"]) == 5 for x in rows]
        count = sum(events)
        lower, certificate = cp_lower_exact(count)
        risk_lower = lower / 128
        difference = risk_lower - ml_upper
        endpoint.append({"K": K, "successes": count, "trials": 16, "events": events,
                         "CP_lower": fraction_record(lower), "dyadic_certificate": certificate,
                         "true_raw_expected_KL_lower": fraction_record(risk_lower),
                         "true_table_MLE_expected_KL_upper": fraction_record(ml_upper),
                         "true_expected_raw_minus_table_MLE_lower": fraction_record(difference),
                         "derived_lower_positive": difference > 0,
                         "derived_consequence_not_extra_confidence_family": True,
                         "finite_n_endpoint_theorem_threshold_not_claimed": True})
    save(OUT / "RESULTS.json", {"primary_sign_test": primary, "endpoint_CP_family": endpoint,
                                "families_not_combined_as_overall_95_percent": True,
                                "reference_MLE_expectations": enumeration,
                                "concentration_illustrations": [contrast_bound(1024, 8192, K) for K in CONFIG["Ks"]],
                                "ledger": {"independent_training_streams": 16, "role_RNG_streams": 32,
                                           "raw_RB_training_grid_optimizations": 576, "table_MLE_training_argmins": 48,
                                           "unique_training_model_outputs": 624, "raw_RB_cell_assignments": 1152,
                                           "count_state_analysis_argmins": 1347,
                                           "new_label_vectors": 16384, "new_label_bits": 4194304,
                                           "global_prior_banks": 131072, "prior_vectors": 524288,
                                           "Gaussian_scalars": 134217728,
                                           "new_reference_integrals": 0, "new_neural_models": 0,
                                           "continuous_optimizations": 0}})


def run():
    verify_freeze(live=True)
    if (OUT / "STATE.json").exists():
        raise FileExistsError("Existing run; no automatic restart or added budget.")
    started = time.perf_counter()
    save(OUT / "STATE.json", {"state": "training", "started": now()})
    training = []
    try:
        reference = read(OUT / "REFERENCE.json")
        # Only model probabilities, never deployment risks, are passed to training.
        for stream in range(CONFIG["streams"]):
            training.append(train_stream(stream, np.asarray(reference["spin_moments"])))
        files = {rel(OUT / f"s{s:02d}" / "TRAINING.json"): sha(OUT / f"s{s:02d}" / "TRAINING.json") for s in range(16)}
        save(OUT / "SELECTION_FREEZE.json", {"created": now(), "files": files,
                                            "all_624_outputs_fixed_before_risk_evaluation": True})
        collect(training, reference)
        save(OUT / "STATE.json", {"state": "complete", "finished": now(), "wall_seconds": time.perf_counter() - started}, exclusive=False)
        names = ["CONFIG.json", "REFERENCE.json", "FREEZE.json", "SELECTION_FREEZE.json", "EVALUATION.json",
                 "MLE_ENUMERATION.json", "RESULTS.json", "STATE.json"]
        public = {name: sha(OUT / name) for name in names}
        arrays = {}
        for s in range(16):
            folder = OUT / f"s{s:02d}"
            public[f"s{s:02d}/TRAINING.json"] = sha(folder / "TRAINING.json")
            for name in ("INPUT.npz", "BANK_CONTRASTS.npz"):
                arrays[f"s{s:02d}/{name}"] = sha(folder / name)
        save(OUT / "RESULTS_FREEZE.json", {"created": now(), "files": public, "optional_local_arrays": arrays})
        print(json.dumps({"state": "complete", "primary_pass": read(OUT / "RESULTS.json")["primary_sign_test"]["passed"],
                          "wall_seconds": time.perf_counter() - started}), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"created": now(), "type": type(exc).__name__, "message": str(exc),
                                    "completed_streams": len(training), "automatic_retry": False})
        save(OUT / "STATE.json", {"state": "failed", "finished": now()}, exclusive=False)
        raise


def check(workspace=False):
    verify_freeze()
    receipt = read(OUT / "RESULTS_FREEZE.json")
    for name, expected in receipt["files"].items():
        assert sha(OUT / name) == expected, name
    if workspace:
        for name, expected in receipt["optional_local_arrays"].items():
            assert sha(OUT / name) == expected, name
    total = 0
    for s in range(16):
        record = read(OUT / f"s{s:02d}/TRAINING.json")
        assert len(record["curves"]) == 39
        for row in record["curves"]:
            selected = choose(row["contrast_curve"])
            assert all(row[k] == v for k, v in selected.items())
            total += 1
    assert total == 624
    result = read(OUT / "RESULTS.json")
    test = result["primary_sign_test"]
    x = sum(test["events"])
    p = Fraction(sum(math.comb(16, j) for j in range(x, 17)), 1 << 16)
    assert from_fraction_record(test["exact_one_sided_pvalue"]) == p
    assert test["passed"] == (p <= Fraction(1, 20))
    for row in result["endpoint_CP_family"]:
        lower, certificate = cp_lower_exact(sum(row["events"]))
        assert lower == from_fraction_record(row["CP_lower"])
        assert certificate == row["dyadic_certificate"]
        difference = lower / 128 - from_fraction_record(row["true_table_MLE_expected_KL_upper"])
        assert difference == from_fraction_record(row["true_expected_raw_minus_table_MLE_lower"])
        assert row["derived_lower_positive"] == (difference > 0)
    print(json.dumps({"passed": True, "workspace_arrays_checked": workspace,
                      "scope": "frozen bytes, 624 saved selections, exact sign/CP rational arithmetic; no training replay"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("qa", "freeze", "run", "check"):
        group.add_argument("--" + name, action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.qa:
        if (OUT / "FREEZE.json").exists():
            raise FileExistsError("Frozen QA cannot be replaced.")
        result = artificial_checks()
        result.update({"source_sha256": sha(__file__), "created": now()})
        path = OUT / "UNIT_CHECK.json"
        count = 1
        if path.exists():
            old = read(path)
            count = old.get("completed_QA_executions", 1) + 1
            save(OUT / "QA_HISTORY" / f"unit_{sha(path)}.json", old)
        result["completed_QA_executions"] = count
        save(path, result, exclusive=not path.exists())
        print(json.dumps(result))
    elif args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        check(args.workspace)


if __name__ == "__main__":
    main()
