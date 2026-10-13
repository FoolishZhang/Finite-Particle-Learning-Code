"""Post-hoc table diagnosis of conditional expected-bank active objectives.

No Gaussian integration, labels, banks, training replay, or new confidence
family. --run is allowed only after this separate diagnostic freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/外层精度与有限样本伪相关_20261001"
LEARN = ROOT / "artifacts/nuisance_learning"
OLD = ROOT / "artifacts/weak_emission/NUMERIC"
FOURIER = ROOT / "artifacts/nuisance_dimensions/NUMERIC"
OUT = LEARN / "ACTIVE_TARGET"
RHO = [3 * i / 20 for i in range(-5, 6)]
VARIANTS = ["low", "primary", "high"]
CONFIG = {"schema": 1, "a": 1.0, "old_latent_theta": 0.15, "rho_grid": RHO,
          "Ns": [64, 256, 1024], "Ks": [2, 4], "Bs": [32, 512, 8192],
          "variants": VARIANTS, "primary": "primary", "streams": 16,
          "score_variant_tolerance": 1e-10, "K2_null_cross_tolerance": 1e-10,
          "choice_qualification": "absolute-rho argmin classes equal; all exact signed sets retained",
          "post_hoc_after_observing_K2_RB_results": True,
          "new_Gaussian_integrals": 0, "new_labels": 0, "new_banks": 0,
          "new_training_models": 0, "new_confidence_families": 0}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def frac(value):
    value = Fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "float_display": float(value)}


def from_frac(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def invert_reference(delta, teacher_spin):
    delta = np.asarray(delta, dtype=float)
    if abs(teacher_spin) < 1e-12:
        raise ArithmeticError("Cannot invert an effectively zero teacher pair moment.")
    return 0.5 * (delta + delta[::-1]), 0.5 * (delta - delta[::-1]) / teacher_spin


def argmin_set(curve):
    value = float(np.min(curve))
    indices = np.flatnonzero(curve == value).tolist()
    return {"minimum_contrast": value, "argmin_indices": indices,
            "argmin_rhos": [RHO[i] for i in indices],
            "absolute_grid_coordinates": sorted({abs(i - 5) for i in indices}),
            "exact_signed_mirror_tie": len(indices) > 1 and sorted(indices) == sorted(10 - i for i in indices)}


def artificial_checks():
    # Arbitrary finite losses: no selected Gaussian point or empirical count.
    same = np.array([1.8, 1.5, 1.4, 1.25, 1.1])
    different = same[::-1]
    teacher_spin = .37
    center = same[2]
    old = (1 + teacher_spin) / 2 * same + (1 - teacher_spin) / 2 * different - center
    even, odd = invert_reference(old, teacher_spin)
    errors = []
    for t in (-1.0, -.27, 0.0, .31, 1.0):
        direct = (1 + t) / 2 * same + (1 - t) / 2 * different - center
        errors.append(float(np.max(abs(direct - (even + t * odd)))))
    counts = np.array([1, 7, 2, 5])  # 00,01,10,11; deliberately unequal margins.
    empirical = (counts[0] * same + counts[1] * different + counts[2] * different + counts[3] * same) / counts.sum() - center
    t = 2 * (counts[0] + counts[3]) / counts.sum() - 1
    errors.append(float(np.max(abs(empirical - (even + t * odd)))))
    assert max(errors) < 1e-14
    return {"passed": True, "max_inversion_error": max(errors),
            "empirical_unbalanced_margins_checked": True, "t_endpoints_included": True,
            "Gaussian_integrals": 0, "new_labels": 0, "new_banks": 0,
            "scientific_count_state_argmins": 0}


def verify_original_seals():
    for folder in (OLD, FOURIER, LEARN):
        receipt = read(folder / "RESULTS_FREEZE.json")
        for name, expected in receipt["files"].items():
            assert sha(folder / name) == expected, (folder, name)
    assert read(OLD / "RESULTS.json")["numerical_qualified"]
    assert read(FOURIER / "RESULTS.json")["numerically_qualified"]
    assert read(LEARN / "STATE.json")["state"] == "complete"
    assert read(OLD / "CONFIG.json")["theta"] == CONFIG["old_latent_theta"]


def freeze():
    if (OUT / "FREEZE.json").exists():
        raise FileExistsError("Refusing to replace diagnostic freeze.")
    verify_original_seals()
    unit = read(OUT / "UNIT_CHECK.json")
    assert unit["passed"] and unit["source_sha256"] == sha(__file__)
    files = [Path(__file__), DOC / "ACTIVE_TARGET_DESIGN.md", DOC / "ACTIVE_TARGET_THEORY.md",
             OLD / "GRID.json", OLD / "CONFIG.json", OLD / "RESULTS.json", OLD / "RESULTS_FREEZE.json",
             FOURIER / "GRID.json", FOURIER / "RESULTS.json", FOURIER / "RESULTS_FREEZE.json",
             LEARN / "REFERENCE.json", LEARN / "CONFIG.json", LEARN / "SELECTION_FREEZE.json",
             LEARN / "MLE_ENUMERATION.json", LEARN / "RESULTS_FREEZE.json"]
    files.extend(LEARN / f"s{s:02d}/TRAINING.json" for s in range(16))
    save(OUT / "CONFIG.json", CONFIG)
    rows = []
    for i, path in enumerate(files):
        target = OUT / "source_inputs" / f"{i:02d}_{path.parent.name}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(path.read_bytes())
        rows.append({"original": rel(path), "snapshot": rel(target), "sha256": sha(target)})
    save(OUT / "FREEZE.json", {"created": now(), "inputs": rows,
                                "config_sha256": sha(OUT / "CONFIG.json"),
                                "unit_sha256": sha(OUT / "UNIT_CHECK.json"),
                                "post_hoc_trigger": "Original high-budget K2 RB reference risk remained close to raw; diagnostic designed afterwards.",
                                "diagnostic_count_states_evaluated_before_freeze": 0})


def verify_freeze(live=False):
    record = read(OUT / "FREEZE.json")
    assert read(OUT / "CONFIG.json") == CONFIG
    assert sha(OUT / "CONFIG.json") == record["config_sha256"]
    assert sha(OUT / "UNIT_CHECK.json") == record["unit_sha256"]
    for x in record["inputs"]:
        assert sha(ROOT / x["snapshot"]) == x["sha256"]
        if live:
            assert sha(ROOT / x["original"]) == x["sha256"]


def coefficients():
    grid = read(OLD / "GRID.json")["rows"]
    result = []
    for variant in VARIANTS:
        indexed = {x["rho"]: x for x in grid if x["a"] == 1.0 and x["variant"] == variant}
        teacher = indexed[0.0]["teacher_spin_moment"]
        assert all(indexed[r]["teacher_spin_moment"] == teacher for r in RHO)
        for K in CONFIG["Ks"]:
            delta = np.array([indexed[r]["score_minus_rho0"][indexed[r]["K"].index(K)] for r in RHO])
            even, odd = invert_reference(delta, teacher)
            assert even[5] == 0 and odd[5] == 0
            assert np.array_equal(even, even[::-1]) and np.array_equal(odd, -odd[::-1])
            result.append({"variant": variant, "K": K, "teacher_spin_moment": teacher,
                           "old_latent_theta_not_used_as_divisor": CONFIG["old_latent_theta"],
                           "rho_grid": RHO, "old_delta_F": delta.tolist(),
                           "even": even.tolist(), "odd": odd.tolist()})
    return result


def enumerate_targets(coef):
    coefficients_by_key = {(x["variant"], x["K"]): x for x in coef}
    groups, checks = [], []
    for N in CONFIG["Ns"]:
        for K in CONFIG["Ks"]:
            states = {v: [] for v in VARIANTS}
            for s in range(N + 1):
                t = (2 * s - N) / N
                curves, sets = [], []
                for variant in VARIANTS:
                    c = coefficients_by_key[(variant, K)]
                    curve = np.asarray(c["even"]) + t * np.asarray(c["odd"])
                    state = {"same_count": s, "empirical_pair": t, **argmin_set(curve)}
                    states[variant].append(state)
                    curves.append(curve)
                    sets.append(state["argmin_indices"])
                discrepancy = float(np.max(np.ptp(curves, axis=0)))
                same_sets = all(x == sets[0] for x in sets)
                absolute_sets = [sorted({abs(i - 5) for i in x}) for x in sets]
                same_risk_classes = all(x == absolute_sets[0] for x in absolute_sets)
                checks.append({"N": N, "K": K, "same_count": s,
                               "max_pointwise_curve_difference": discrepancy,
                               "score_limit": CONFIG["score_variant_tolerance"],
                               "score_qualified": discrepancy <= CONFIG["score_variant_tolerance"],
                               "variant_argmin_sets": sets, "variant_absolute_argmin_sets": absolute_sets,
                               "signed_sets_identical": same_sets, "choice_qualified": same_risk_classes,
                               "choice_qualification_scope": "absolute-rho/null-risk class, not forced signed index",
                               "passed": bool(same_risk_classes and discrepancy <= CONFIG["score_variant_tolerance"])})
            for variant in VARIANTS:
                groups.append({"N": N, "K": K, "variant": variant, "states": states[variant]})
    return groups, checks


def cross_null_K2(coef):
    rows = read(FOURIER / "GRID.json")["rows"]
    indexed = {x["rho"]: x for x in rows if x["a"] == 1.0 and x["theta"] == 0.0 and x["n"] == 0 and x["variant"] == "primary"}
    checks = []
    for item in coef:
        if item["K"] != 2:
            continue
        for rho, even in zip(RHO, item["even"]):
            other = indexed[rho]["delta_F"]
            error = abs(even - other)
            checks.append({"variant": item["variant"], "rho": rho, "reconstructed_null": even,
                           "frozen_Fourier_null": other, "error": error,
                           "limit": CONFIG["K2_null_cross_tolerance"],
                           "passed": error <= CONFIG["K2_null_cross_tolerance"]})
    return checks


def expected_risks(groups, reference):
    risks = [Fraction.from_float(float(x)) for x in reference["reference_joint_KL"]]
    original_mle = {x["N"]: x for x in read(LEARN / "MLE_ENUMERATION.json")["rows"]}
    mle = {}
    for N, old in original_mle.items():
        indices = old["indices_for_same_count_0_through_N"]
        weights = [math.comb(N, s) for s in range(N + 1)]
        exact_reference = sum((w * risks[i] for w, i in zip(weights, indices)), Fraction()) / (1 << N)
        square_sum = sum(w * (i - 5) ** 2 for w, i in zip(weights, indices))
        upper = Fraction(9 * square_sum, 12350 * (1 << N))
        assert upper == from_frac(old["true_expected_KL_upper"])
        mle[N] = {"N": N, "original_reference_expected_KL": old["reference_expected_KL"],
                  "exact_reference_expected_KL": frac(exact_reference), "true_expected_KL_upper": frac(upper),
                  "argmin_recomputed": False}
    result = []
    for group in groups:
        N = group["N"]
        lower, upper = Fraction(), Fraction()
        max_square_sum = 0
        multi_magnitude = []
        for s, state in enumerate(group["states"]):
            indices = state["argmin_indices"]
            w = math.comb(N, s)
            lower += w * min(risks[i] for i in indices)
            upper += w * max(risks[i] for i in indices)
            max_square_sum += w * max((i - 5) ** 2 for i in indices)
            if len(state["absolute_grid_coordinates"]) > 1:
                multi_magnitude.append(s)
        lower /= 1 << N
        upper /= 1 << N
        analytic = Fraction(9 * max_square_sum, 12350 * (1 << N))
        ml_ref = from_frac(mle[N]["exact_reference_expected_KL"])
        result.append({"N": N, "K": group["K"], "variant": group["variant"],
                       "exact_reference_expected_KL_min": frac(lower), "exact_reference_expected_KL_max": frac(upper),
                       "true_expected_KL_upper_for_any_argmin_selection": frac(analytic),
                       "exact_binomial_max_square_sum": str(max_square_sum),
                       "counts_with_multiple_magnitude_minima": multi_magnitude,
                       "reference_proxy_minus_table_MLE_min": frac(lower - ml_ref),
                       "reference_proxy_minus_table_MLE_max": frac(upper - ml_ref),
                       "table_MLE": mle[N],
                       "difference_of_analytic_upper_bounds_is_not_a_risk_gap_bound": True})
    return result


def actual_lookups(groups, checks, reference):
    indexed = {(x["N"], x["K"], x["variant"]): x["states"] for x in groups}
    qualified = {(x["N"], x["K"], x["same_count"]): x["passed"] for x in checks}
    risk = reference["reference_joint_KL"]
    targets, comparisons = [], []
    for stream in range(16):
        training = read(LEARN / f"s{stream:02d}/TRAINING.json")
        models = {x["id"]: x for x in training["curves"]}
        for ni, N in enumerate(CONFIG["Ns"]):
            s = models[f"table_MLE_N{N}"]["same_count"]
            counts = training["active_pattern_counts_by_N"][ni]
            assert sum(counts) == N and counts[0] + counts[3] == s
            for K in CONFIG["Ks"]:
                primary = indexed[(N, K, "primary")][s]
                inds = primary["argmin_indices"]
                target = {"stream": stream, "N": N, "K": K, "same_count": s,
                          "empirical_pair": primary["empirical_pair"], "target_indices": inds,
                          "target_rhos": primary["argmin_rhos"],
                          "target_absolute_grid_coordinates": primary["absolute_grid_coordinates"],
                          "variant_target_sets": {v: indexed[(N, K, v)][s]["argmin_indices"] for v in VARIANTS},
                          "count_state_qualified": qualified[(N, K, s)],
                          "target_reference_risk_min": min(risk[i] for i in inds),
                          "target_reference_risk_max": max(risk[i] for i in inds)}
                targets.append(target)
                for B in CONFIG["Bs"]:
                    rb = models[f"RB_N{N}_B{B}_K{K}"]
                    idx = rb["index"]
                    comparisons.append({**target, "B": B, "original_RB_index": idx,
                                        "original_RB_rho": rb["rho"], "original_RB_reference_KL": risk[idx],
                                        "exact_target_set_membership": idx in inds,
                                        "target_absolute_rho_membership": abs(idx - 5) in primary["absolute_grid_coordinates"],
                                        "RB_minus_target_reference_risk_min": risk[idx] - target["target_reference_risk_max"],
                                        "RB_minus_target_reference_risk_max": risk[idx] - target["target_reference_risk_min"],
                                        "descriptive_not_independent_replicate": True})
    assert len(targets) == 96 and len(comparisons) == 288
    return targets, comparisons


def run():
    verify_freeze(live=True)
    verify_original_seals()
    if (OUT / "STATE.json").exists():
        raise FileExistsError("Existing diagnostic; no automatic rerun or new orders.")
    started = time.perf_counter()
    save(OUT / "STATE.json", {"state": "running", "started": now()})
    try:
        coef = coefficients()
        save(OUT / "COEFFICIENTS.json", {"rows": coef})
        groups, checks = enumerate_targets(coef)
        save(OUT / "ENUMERATION.json", {"groups": groups, "count_state_argmins": 8082,
                                         "signed_ties_not_forced_to_one_index": True})
        cross = cross_null_K2(coef)
        save(OUT / "QUALIFICATION.json", {"count_state_rows": checks, "K2_null_cross_rows": cross,
                                           "all_passed": all(x["passed"] for x in checks + cross)})
        reference = read(LEARN / "REFERENCE.json")
        assert reference["rho_grid"] == RHO
        expectation = expected_risks(groups, reference)
        save(OUT / "EXPECTED_RISKS.json", {"rows": expectation,
                                            "exact_arithmetic_conditional_on_fixed_reference": True,
                                            "strict_B_infinity_Gaussian_target_certificate": False})
        targets, comparisons = actual_lookups(groups, checks, reference)
        save(OUT / "ACTUAL_TARGETS.json", {"rows": targets})
        save(OUT / "RB_COMPARISONS.json", {"rows": comparisons})
        qualified = all(x["passed"] for x in checks + cross)
        save(OUT / "RESULTS.json", {"qualified": qualified, "post_hoc": True,
                                    "original_primary_and_CP_unchanged": True,
                                    "not_independent_confirmation": True,
                                    "ledger": {"count_state_argmins": 8082, "unique_count_K_states": 2694,
                                               "primary_actual_count_lookups": 96, "original_RB_pairings": 288,
                                               "old_MLE_state_index_lookups": 1347,
                                               "new_Gaussian_integrals": 0, "new_labels": 0, "new_banks": 0,
                                               "new_training_models": 0, "new_confidence_families": 0,
                                               "wall_seconds": time.perf_counter() - started}})
        save(OUT / "STATE.json", {"state": "complete" if qualified else "qualification_failed", "finished": now()}, exclusive=False)
        verify_original_seals()
        names = ["CONFIG.json", "FREEZE.json", "COEFFICIENTS.json", "ENUMERATION.json", "QUALIFICATION.json",
                 "EXPECTED_RISKS.json", "ACTUAL_TARGETS.json", "RB_COMPARISONS.json", "RESULTS.json", "STATE.json"]
        save(OUT / "RESULTS_FREEZE.json", {"created": now(), "files": {x: sha(OUT / x) for x in names}})
        print(json.dumps({"qualified": qualified, "qualification_failures": sum(not x["passed"] for x in checks + cross),
                          "count_state_argmins": 8082, "actual_lookups": 96, "RB_pairings": 288,
                          "wall_seconds": time.perf_counter() - started}), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"created": now(), "type": type(exc).__name__, "message": str(exc), "automatic_retry": False})
        save(OUT / "STATE.json", {"state": "execution_failed", "finished": now()}, exclusive=False)
        raise


def check():
    verify_freeze()
    verify_original_seals()
    for name, expected in read(OUT / "RESULTS_FREEZE.json")["files"].items():
        assert sha(OUT / name) == expected, name
    groups = read(OUT / "ENUMERATION.json")["groups"]
    assert sum(len(x["states"]) for x in groups) == 8082
    coefs = {(x["variant"], x["K"]): x for x in read(OUT / "COEFFICIENTS.json")["rows"]}
    for group in groups:
        c = coefs[(group["variant"], group["K"])]
        for s, state in enumerate(group["states"]):
            t = (2 * s - group["N"]) / group["N"]
            curve = np.asarray(c["even"]) + t * np.asarray(c["odd"])
            assert state["empirical_pair"] == t and state["same_count"] == s
            assert state["argmin_indices"] == np.flatnonzero(curve == np.min(curve)).tolist()
            assert state["minimum_contrast"] == float(np.min(curve))
    assert len(read(OUT / "ACTUAL_TARGETS.json")["rows"]) == 96
    assert len(read(OUT / "RB_COMPARISONS.json")["rows"]) == 288
    rows = read(OUT / "QUALIFICATION.json")
    passed = all(x["score_qualified"] and x["choice_qualified"] for x in rows["count_state_rows"]) and all(x["passed"] for x in rows["K2_null_cross_rows"])
    assert rows["all_passed"] == passed == read(OUT / "RESULTS.json")["qualified"]
    print(json.dumps({"passed": True, "scope": "frozen bytes and saved finite-table arithmetic; no Gaussian evaluation or new data", "qualified": passed}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("qa", "freeze", "run", "check"):
        group.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.qa:
        if (OUT / "FREEZE.json").exists():
            raise FileExistsError("Frozen artificial QA cannot be replaced.")
        result = artificial_checks()
        result.update({"source_sha256": sha(__file__), "created": now()})
        save(OUT / "UNIT_CHECK.json", result)
        print(json.dumps(result))
    elif args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        check()
