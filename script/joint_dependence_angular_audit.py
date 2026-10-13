"""Independent execution/receipt checks for the frozen angular calculation.

This module does not import joint_dependence_angular or its kernel. It reads all
32 raw files, recombines saved values with saved weights, and independently
evaluates five prescribed banks per file using direct Bernoulli products and a
hand-written path derivative. Existing low-dimensional PMFs and the order-8
finite-difference fixture are reconstructed, not the full billion-score panel.

Passing means implementation/receipt consistency. The two unqualified epsilon
points must remain unqualified; this is not a quadrature remainder certificate
or an external scientific/theorem review. No RNG is used and inputs are read-only.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
FREEZE = OUT / "ANGULAR_FREEZE.json"
RESULTS = OUT / "ANGULAR_RESULTS.json"
AUDIT = OUT / "ANGULAR_AUDIT.json"
PHI = np.arange(-7, 8, 2, dtype=np.float64) / 10.0
Y = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.float64)
DIRECT_TOLERANCE = 5e-11
SUMMARY_TOLERANCE = 1e-11
PMF_REPLAY_TOLERANCE = 2e-12
NODE_TOLERANCE = 2e-13


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def maximum_error(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def check_close(actual, reference, tolerance, label):
    error = maximum_error(actual, reference)
    assert math.isfinite(error) and error <= tolerance, (label, error, tolerance)
    return error


def parameters(epsilon, t):
    angles = (1.0 - t) * PHI
    W = epsilon * np.column_stack((np.cos(angles), np.sin(angles)))
    # d angle_j / dt = -phi_j; the row tangent is (-W_j2,W_j1).
    path_dW = (-PHI[:, None]) * np.column_stack((-W[:, 1], W[:, 0]))
    return W, path_dW


def normal_nodes(order):
    """Different GH constructor from the frozen scipy roots_hermitenorm path."""
    nodes, weights = hermegauss(order)
    return nodes, weights / math.sqrt(2.0 * math.pi)


def direct_banks(banks, W, path_dW, teacher, chunk=128):
    """Direct K2 probability products and d/dt of their average likelihood."""
    assert banks.ndim == 3 and banks.shape[1:] == (2, 2)
    assert teacher.shape == (256,) and np.all(teacher > 0)
    answer = np.empty((len(banks), 2), dtype=np.float64)
    for start in range(0, len(banks), chunk):
        stop = min(start + chunk, len(banks))
        z = banks[start:stop]
        logits = z @ W.T
        probabilities = expit(logits)
        factors = np.where(Y[None, None, :, :] == 1,
                           probabilities[:, :, None, :], 1.0 - probabilities[:, :, None, :])
        likelihood = np.prod(factors, axis=3)  # bank, particle, pattern
        mixture = 0.5 * (likelihood[:, 0, :] + likelihood[:, 1, :])
        assert np.isfinite(mixture).all() and np.all(mixture > 0), "Probability-product fixture underflow"
        eta_dot = z @ path_dW.T
        log_derivative = np.einsum("ym,bkm->bky", Y, eta_dot, optimize=True)
        log_derivative -= np.sum(probabilities * eta_dot, axis=2)[:, :, None]
        likelihood_derivative = likelihood * log_derivative
        mixture_derivative = 0.5 * (likelihood_derivative[:, 0, :] + likelihood_derivative[:, 1, :])
        answer[start:stop, 0] = -np.log(mixture) @ teacher
        answer[start:stop, 1] = -(mixture_derivative / mixture) @ teacher
    return answer


def independent_pmf(epsilon, t, order, angle=0.0):
    one_nodes, one_weights = normal_nodes(order)
    index = np.arange(order * order)
    nodes = np.column_stack((one_nodes[index // order], one_nodes[index % order]))
    weights = one_weights[index // order] * one_weights[index % order]
    if angle:
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        nodes = nodes @ R.T
    W, _ = parameters(epsilon, t)
    pmf = np.zeros(256, dtype=np.float64)
    for start in range(0, len(nodes), 256):
        stop = min(start + 256, len(nodes))
        probability = expit(nodes[start:stop] @ W.T)
        products = np.prod(np.where(Y[None, :, :] == 1,
                          probability[:, None, :], 1.0 - probability[:, None, :]), axis=2)
        pmf += weights[start:stop] @ products
    assert np.isfinite(pmf).all() and np.all(pmf > 0)
    return pmf


def check_frozen_sources(frozen):
    for name, digest in frozen["sources"].items():
        assert sha(ROOT / name) == digest, ("Frozen source drift", name)


def execute_checks():
    frozen, result = read(FREEZE), read(RESULTS)
    check_frozen_sources(frozen)
    source_hashes = {rel(FREEZE): sha(FREEZE), rel(RESULTS): sha(RESULTS),
                     **frozen["sources"]}
    assert result["freeze_sha256"] == source_hashes[rel(FREEZE)]
    config = frozen["config"]
    assert result["config"] == config
    assert config["epsilons"] == [.125, .25, .5, .8] and config["t_values"] == [0., .05]
    assert config["orders"] == [8, 12, 16, 20] and config["pmf_orders"] == [40, 60, 80]
    assert config["rotated_order"] == 72
    assert [row["epsilon"] for row in result["rows"]] == config["epsilons"]
    independent_rule_cache = {order: normal_nodes(order) for order in config["orders"]}
    raw_checks, point_checks = [], []
    fd_inputs = None
    maximum_direct_loss_error = maximum_direct_derivative_error = 0.0
    maximum_weighted_loss_error = maximum_weighted_derivative_error = 0.0
    maximum_pmf_error = maximum_pmf_diagnostic_error = maximum_risk_error = 0.0
    raw_banks = bytes_read = pmf_node_evaluations = pmf_calls = 0
    rotation_loss_error = rotation_derivative_error = 0.0
    qualified_flags = []
    for row in result["rows"]:
        epsilon = row["epsilon"]
        teacher = np.asarray(row["teacher_pmf"], dtype=np.float64)
        assert teacher.shape == (256,) and np.isfinite(teacher).all() and np.all(teacher > 0)
        teacher_mass = float(abs(teacher.sum() - 1.0))
        check_close(teacher_mass, row["teacher_mass_error"], 1e-15, "teacher mass receipt")
        assert teacher_mass <= config["mass_tolerance"]
        assert [entry["t"] for entry in row["evaluations"]] == config["t_values"]
        evaluation_replays = []
        reference_pmf_by_t = {}
        for entry in row["evaluations"]:
            t = entry["t"]
            pmfs = [independent_pmf(epsilon, t, order) for order in config["pmf_orders"]]
            pmfs.append(independent_pmf(epsilon, t, config["rotated_order"], config["rotation"]))
            pmf_calls += 4
            pmf_node_evaluations += sum(order ** 2 for order in config["pmf_orders"]) + config["rotated_order"] ** 2
            reference_pmf_by_t[t] = pmfs
            mass_error = float(max(abs(pmf.sum() - 1.0) for pmf in pmfs))
            discrepancy = float(max(np.max(np.abs(pmf - pmfs[2])) for pmf in pmfs))
            risks = [float(teacher @ (np.log(teacher) - np.log(pmf))) for pmf in pmfs]
            risk_discrepancy = max(abs(risk - risks[2]) for risk in risks)
            maximum_risk_error = max(maximum_risk_error,
                check_close(risks, entry["risk_all_orders"], PMF_REPLAY_TOLERANCE, "same-teacher risk orders"),
                check_close(risks[2], entry["risk"], PMF_REPLAY_TOLERANCE, "risk center"))
            maximum_pmf_diagnostic_error = max(maximum_pmf_diagnostic_error,
                check_close(mass_error, entry["pmf_mass_error"], PMF_REPLAY_TOLERANCE, "PMF mass diagnostic"),
                check_close(discrepancy, entry["pmf_discrepancy"], PMF_REPLAY_TOLERANCE, "PMF order diagnostic"),
                check_close(risk_discrepancy, entry["risk_discrepancy"], PMF_REPLAY_TOLERANCE, "risk order diagnostic"))
            assert [item["order"] for item in entry["finite_K2"]] == config["orders"]
            values_by_order = []
            for item in entry["finite_K2"]:
                order = item["order"]
                path = ROOT / item["raw"]
                assert path.parent.resolve() == OUT.resolve() and path.suffix == ".npz"
                assert sha(path) == item["raw_sha256"], path
                with np.load(path, allow_pickle=False) as raw:
                    assert set(raw.files) == {"values", "weights", "indices", "nodes_1d", "weights_1d", "teacher_pmf", "W", "path_dW"}
                    arrays = {name: raw[name] for name in raw.files}
                values, weights, indices = arrays["values"], arrays["weights"], arrays["indices"]
                x, w = arrays["nodes_1d"], arrays["weights_1d"]
                count = order ** 4
                assert values.shape == (count, 2) and weights.shape == (count,)
                assert indices.shape == (count, 4) and indices.dtype == np.int16
                assert x.shape == w.shape == (order,)
                assert arrays["W"].shape == arrays["path_dW"].shape == (8, 2)
                assert arrays["teacher_pmf"].shape == (256,)
                assert all(np.isfinite(value).all() for value in arrays.values())
                assert all(arrays[name].dtype == np.float64 for name in arrays if name != "indices")
                assert np.array_equal(arrays["teacher_pmf"], teacher), "Both endpoints/orders must use the identical stored t=0 teacher"
                flat = np.arange(count, dtype=np.int64)
                expected_indices = np.column_stack([(flat // order ** power) % order for power in (3, 2, 1, 0)])
                assert np.array_equal(indices, expected_indices)
                other_x, other_w = independent_rule_cache[order]
                node_error = check_close(x, other_x, NODE_TOLERANCE, "standard normal nodes")
                one_weight_error = check_close(w, other_w, NODE_TOLERANCE, "standard normal weights")
                assert np.all(w > 0) and np.all(weights > 0)
                assert abs(w.sum() - 1) <= config["mass_tolerance"]
                assert abs(w @ x) <= config["mass_tolerance"]
                assert abs(w @ (x ** 2) - 1) <= config["mass_tolerance"]
                assert abs(w @ (x ** 4) - 3) <= 1e-11
                assert np.array_equal(weights, np.prod(w[indices], axis=1))
                W, dW = parameters(epsilon, t)
                check_close(arrays["W"], W, 2e-16, "saved W geometry")
                check_close(arrays["path_dW"], dW, 2e-16, "saved angular path derivative")
                check_close(np.linalg.norm(arrays["W"], axis=1), epsilon, 5e-16, "fixed row norms")
                assert item["banks"] == count and item["particle_vectors"] == 2 * count
                totals = weights @ values
                mass = float(weights.sum())
                weighted_loss_error = check_close(totals[0], item["loss"], SUMMARY_TOLERANCE, "weighted saved loss")
                weighted_derivative_error = check_close(totals[1], item["path_derivative"], SUMMARY_TOLERANCE, "weighted saved derivative")
                check_close(mass, item["quadrature_mass"], 1e-15, "four-dimensional GH mass")
                maximum_weighted_loss_error = max(maximum_weighted_loss_error, weighted_loss_error)
                maximum_weighted_derivative_error = max(maximum_weighted_derivative_error, weighted_derivative_error)
                fixed = [0, count // 7, count // 3, count // 2, count - 1]
                banks = x[indices[fixed]].reshape(5, 2, 2)
                direct = direct_banks(banks, W, dW, teacher)
                loss_error = check_close(direct[:, 0], values[fixed, 0], DIRECT_TOLERANCE, "direct probability-product loss")
                derivative_error = check_close(direct[:, 1], values[fixed, 1], DIRECT_TOLERANCE, "hand angular derivative")
                maximum_direct_loss_error = max(maximum_direct_loss_error, loss_error)
                maximum_direct_derivative_error = max(maximum_direct_derivative_error, derivative_error)
                raw_checks.append(dict(epsilon=epsilon, t=t, order=order, path=item["raw"],
                    sha256=item["raw_sha256"], bytes=path.stat().st_size, banks=count,
                    nodes_1d_max_difference_from_numpy_rule=node_error,
                    weights_1d_max_difference_from_numpy_rule=one_weight_error,
                    index_shape_and_lexicographic_order=True, identical_t0_teacher=True,
                    quadrature_mass=mass, weighted_loss=float(totals[0]), weighted_derivative=float(totals[1]),
                    weighted_loss_error=weighted_loss_error, weighted_derivative_error=weighted_derivative_error,
                    selected_indices=fixed, direct_values=direct.tolist(),
                    saved_selected_values=values[fixed].tolist(),
                    direct_loss_max_error=loss_error, direct_derivative_max_error=derivative_error))
                raw_banks += count
                bytes_read += path.stat().st_size
                values_by_order.append([float(totals[0]), float(totals[1])])
                if epsilon == .5 and t == 0.0 and order == 8:
                    fd_inputs = (x.copy(), indices.copy(), weights.copy(), teacher.copy())
                if epsilon == .25 and t == .05 and order == 12:
                    angle = .731
                    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                    rotated = direct_banks(banks @ R.T, W @ R.T, dW @ R.T, teacher)
                    rotation_loss_error = check_close(rotated[:, 0], direct[:, 0], DIRECT_TOLERANCE, "common rotation loss")
                    rotation_derivative_error = check_close(rotated[:, 1], direct[:, 1], DIRECT_TOLERANCE, "common rotation derivative")
            evaluation_replays.append(dict(t=t, pmf_mass_error=mass_error, pmf_discrepancy=discrepancy,
                 risk_all_orders=risks, risk_discrepancy=risk_discrepancy, weighted_values_by_order=values_by_order))
        teacher_pmfs = reference_pmf_by_t[0.0]
        maximum_pmf_error = max(maximum_pmf_error,
            check_close(teacher_pmfs[2], teacher, PMF_REPLAY_TOLERANCE, "teacher GH80 probability products"))
        teacher_discrepancy = float(max(np.max(np.abs(pmf - teacher)) for pmf in (teacher_pmfs[0], teacher_pmfs[1], teacher_pmfs[3])))
        maximum_pmf_diagnostic_error = max(maximum_pmf_diagnostic_error,
            check_close(teacher_discrepancy, row["teacher_pmf_discrepancy"], PMF_REPLAY_TOLERANCE, "teacher cross-order discrepancy"))
        left, right = evaluation_replays
        differences = [b[0] - a[0] for a, b in zip(left["weighted_values_by_order"], right["weighted_values_by_order"])]
        derivatives = [value[1] for value in left["weighted_values_by_order"]]
        gap_discrepancy = max(abs(value - differences[-1]) for value in differences[-2:])
        derivative_discrepancy = max(abs(value - derivatives[-1]) for value in derivatives[-2:])
        for value, reference, label in ((differences, row["K2_difference_all_orders"], "all point contrasts"),
             (differences[-1], row["K2_difference"], "point contrast"),
             (gap_discrepancy, row["K2_difference_discrepancy"], "high-order contrast discrepancy"),
             (derivatives[-1], row["K2_derivative"], "path derivative"),
             (derivative_discrepancy, row["K2_derivative_discrepancy"], "high-order derivative discrepancy"),
             (derivatives[-1] / epsilon ** 4, row["derivative_over_epsilon4"], "scaled derivative")):
            check_close(value, reference, SUMMARY_TOLERANCE, label)
        qualifies = bool(teacher_mass <= config["mass_tolerance"] and teacher_discrepancy <= config["pmf_tolerance"]
            and all(entry["pmf_mass_error"] <= config["mass_tolerance"] and entry["pmf_discrepancy"] <= config["pmf_tolerance"]
                    and entry["risk_discrepancy"] <= config["risk_tolerance"] for entry in evaluation_replays)
            and gap_discrepancy <= config["angular_tolerance"] and derivative_discrepancy <= config["angular_tolerance"])
        assert qualifies == row["qualified"]
        qualified_flags.append(qualifies)
        point_checks.append(dict(epsilon=epsilon, teacher_mass_error=teacher_mass,
              independent_teacher_discrepancy=teacher_discrepancy, evaluations=evaluation_replays,
              point_contrast=differences[-1], point_contrast_all_orders=differences,
              high_order_comparison_orders=config["orders"][-2:], point_contrast_discrepancy=gap_discrepancy,
              derivative_discrepancy=derivative_discrepancy, recomputed_qualified=qualifies,
              recorded_qualified=row["qualified"], teacher_changed_between_t_values=False))
    assert len(raw_checks) == 32 and len({row["path"] for row in raw_checks}) == 32
    assert {rel(path) for path in OUT.glob("angular_e*_t*_q*.npz")} == {row["path"] for row in raw_checks}
    assert qualified_flags == [True, True, False, False]
    assert result["all_qualified"] is False and result["all_qualified"] == all(qualified_flags)
    assert fd_inputs is not None
    x, indices, weights, fixed_teacher = fd_inputs
    step = config["finite_difference_step"]
    banks = x[indices].reshape(-1, 2, 2)
    finite_difference_values = []
    for t in (0.0, step, -step):
        W, dW = parameters(.5, t)
        finite_difference_values.append(weights @ direct_banks(banks, W, dW, fixed_teacher))
    center, plus, minus = finite_difference_values
    numeric = float((plus[0] - minus[0]) / (2 * step))
    fd_error = abs(numeric - float(center[1]))
    original_fd = result["finite_difference"]
    check_close(original_fd["absolute_error"], abs(original_fd["analytic"] - original_fd["numeric"]), 1e-16, "FD receipt arithmetic")
    check_close(center[1], original_fd["analytic"], SUMMARY_TOLERANCE, "independent FD center")
    numeric_receipt_difference = check_close(numeric, original_fd["numeric"], config["finite_difference_tolerance"], "independent FD replay")
    assert original_fd["absolute_error"] <= config["finite_difference_tolerance"]
    assert fd_error <= config["finite_difference_tolerance"]
    pair = np.triu_indices(8, 1)
    delta = PHI[:, None] - PHI[None, :]
    gram_offdiagonal = np.cos(delta)[pair]
    derivative_gram = (delta * np.sin(delta))[pair]
    directional_sum = float(gram_offdiagonal @ derivative_gram)
    geometry = result["geometry"]
    check_close(geometry["phi"], PHI, 0, "geometry phi")
    check_close(geometry["sum_phi"], PHI.sum(), 0, "geometry sum phi")
    check_close(geometry["minimum_unscaled_offdiagonal"], gram_offdiagonal.min(), 1e-15, "geometry minimum")
    check_close(geometry["S_directional_sum"], directional_sum, 1e-14, "geometry scalar arithmetic")
    check_close(geometry["K2_derivative_limit"], -directional_sum / 32, 1e-15, "recorded scalar coefficient arithmetic")
    for row in result["rows"]:
        check_close(row["predicted_limit"], -directional_sum / 32, 1e-15, "point scalar coefficient")
    original_bank_count = raw_banks + 3 * 8 ** 4
    counts = result["counts"]
    assert counts["quadrature_banks"] == original_bank_count == 2015232
    assert counts["two_particle_vectors_at_quadrature_nodes"] == 2 * original_bank_count
    assert counts["conditional_pattern_particle_scores"] == 256 * 2 * original_bank_count
    assert counts["random_draws"] == 0 and counts["full_observation_patterns"] == 256
    assert counts["monte_carlo_confidence_claim"] is False
    assert result["numeric_qualification_is_rigorous_interval_certificate"] is False
    check_frozen_sources(frozen)
    assert sha(FREEZE) == source_hashes[rel(FREEZE)] and sha(RESULTS) == source_hashes[rel(RESULTS)]
    return dict(implementation_receipts_consistent=True, source_hashes=source_hashes,
        all_raw_hashes_checked=True, raw_files=raw_checks, point_checks=point_checks,
        maxima=dict(weighted_saved_loss_error=maximum_weighted_loss_error,
            weighted_saved_derivative_error=maximum_weighted_derivative_error,
            sampled_direct_loss_error=maximum_direct_loss_error,
            sampled_direct_derivative_error=maximum_direct_derivative_error,
            teacher_pmf_replay_error=maximum_pmf_error, pmf_diagnostic_replay_error=maximum_pmf_diagnostic_error,
            same_teacher_risk_replay_error=maximum_risk_error),
        qualification=dict(recorded_all_qualified=False, recomputed_all_qualified=False,
            qualified_epsilons=[.125, .25], unqualified_epsilons=[.5, .8], passed_points=2,
            unqualified_points=2, unqualified_is_not_an_audit_failure=True,
            frozen_tolerances_preserved=True, low_order_results_not_removed=True),
        finite_difference=dict(epsilon=.5, order=8, step=step, same_saved_teacher_for_center_plus_minus=True,
            direct_center_loss=float(center[0]), direct_plus_loss=float(plus[0]), direct_minus_loss=float(minus[0]),
            direct_analytic=float(center[1]), direct_numeric=numeric, direct_absolute_error=fd_error,
            recorded_absolute_error=original_fd["absolute_error"],
            numeric_difference_from_record=numeric_receipt_difference,
            tolerance=config["finite_difference_tolerance"], passed=True),
        geometry_common_rotation=dict(fixture_epsilon=.25, fixture_t=.05, fixture_order=12,
            fixed_banks=5, angle=.731, loss_max_error=rotation_loss_error,
            path_derivative_max_error=rotation_derivative_error, theory_coefficient_arithmetic_only=True),
        scope=dict(raw_files_read=32, compressed_raw_bytes_read=bytes_read,
            all_saved_weighted_rows_read=raw_banks, direct_fixed_banks_per_raw_file=5,
            direct_fixed_banks_total=160, direct_labels_per_bank=256,
            direct_sampled_particle_pattern_products=160 * 2 * 256,
            finite_difference_replay_banks=3 * 8 ** 4,
            finite_difference_particle_pattern_products=3 * 8 ** 4 * 2 * 256,
            independent_PMF_recomputations=pmf_calls, independent_PMF_nodes=pmf_node_evaluations,
            independent_PMF_pattern_products=pmf_node_evaluations * 256,
            common_rotation_extra_bank_evaluations=5, random_draws=0, new_scientific_design_points=0,
            full_billion_score_panel_replayed=False, angular_quadrature_called=False,
            frozen_kernel_called=False, outputs_or_frozen_sources_modified=False),
        tolerances=dict(direct_bank=DIRECT_TOLERANCE, weighted_summary=SUMMARY_TOLERANCE,
            independent_PMF=PMF_REPLAY_TOLERANCE, independent_nodes=NODE_TOLERANCE),
        quadrature_remainder_certified=False, numeric_qualification_is_rigorous_interval_certificate=False,
        external_scientific_review_performed=False, theorem_review_performed=False)


def main(check):
    started = time.perf_counter()
    validation = execute_checks()
    if check:
        record = read(AUDIT)
        assert record["script_sha256"] == sha(__file__)
        assert record["validation"] == validation
        assert record["status"] == "passed_implementation_and_receipt_checks"
    else:
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            script_sha256=sha(__file__), status="passed_implementation_and_receipt_checks",
            wall_seconds=time.perf_counter() - started, validation=validation)
        with AUDIT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(dict(passed=True, checked_existing_receipt=check,
          wall_seconds=time.perf_counter() - started, raw_files=32,
          maxima=validation["maxima"], qualification=validation["qualification"],
          finite_difference_error=validation["finite_difference"]["direct_absolute_error"],
          rigorous_quadrature_certificate=False, scientific_review=False), ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Run once and save the author implementation-audit receipt")
    group.add_argument("--check", action="store_true", help="Recompute the audit and compare with its saved receipt")
    arguments = parser.parse_args()
    main(arguments.check)
