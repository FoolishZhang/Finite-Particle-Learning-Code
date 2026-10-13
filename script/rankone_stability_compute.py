"""Fixed rank-one angular-stability quadratures; no training or random draws.

--self-test uses manufactured fixtures only. --freeze binds the four public
source streams and all 28 anchors before scientific evaluation. --run --rep N
evaluates one frozen stream and never replaces an existing output directory.
The C/D gates are numerical comparisons, not rigorous integration remainder
certificates. Common rotation is removed with a fixed Helmert basis, while
the independent compensated-Hessian Ward residual is retained and checked.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
import rankone_stability_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/RankOneStability_v1_20261007"
OUT = ROOT / "artifacts/rankone_stability_v1_20261007"
SOURCE = ROOT / "artifacts/joint_dependence_v1_20261007"
FREEZE = OUT / "FREEZE.json"
KERNEL_TEST = OUT / "KERNEL_TEST.json"
DRIVER_TEST = Path(__file__).with_name("rankone_stability_compute_test.py")
TARGET_NAMES = ("empirical", "population")
K_VALUES = (8, 64, "inf")
RULES = {
    "A": dict(normal_rule="hermite", normal_order=96, normal_limit=10., logu_min=-24., logu_max=48., logu_step=.25, outer_rule="trapezoid"),
    "B": dict(normal_rule="legendre", normal_order=256, normal_limit=10., logu_min=-24., logu_max=48., logu_step=.25, outer_rule="trapezoid"),
    "C": dict(normal_rule="legendre", normal_order=384, normal_limit=10., logu_min=-24., logu_max=56., logu_step=.125, outer_rule="trapezoid"),
    "D": dict(normal_rule="legendre", normal_order=512, normal_limit=11., logu_min=-28., logu_max=64., logu_step=.125, outer_rule="trapezoid"),
}
THRESHOLDS = dict(matrix_difference=2e-7, eigenvalue_difference=2e-6,
                  ward_residual=2e-7, normal_mass_error=2e-12,
                  d_mass_error=2e-7, c_mass_error=2e-7, sign_resolution=1e-5)
CONFIG = dict(reps=4, coordinates=8, anchors_per_rep=7, targets=list(TARGET_NAMES),
              K_values=list(K_VALUES), rules=RULES, thresholds=THRESHOLDS,
              primary_rule="C", comparison_rule="D", scalar_gate_scope="all_256_labels_per_K",
              target_weights_renormalized=False, compute_loss=False, return_coefficients=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON scalar")
    return value


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(native(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_npz(path, arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def artifact(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def helmert_basis(m):
    """A fixed orthonormal basis of 1-perp, independent of all matrices."""
    if m < 2:
        raise ValueError("at least two coordinates required")
    Q = np.zeros((m, m - 1))
    for j in range(m - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        Q[:j + 1, j] = 1. / scale
        Q[j + 1, j] = -(j + 1) / scale
    return Q


def checked_candidate(candidate, m=8):
    b, W, theta = (np.asarray(candidate[k], dtype=float) for k in ("b", "W", "theta"))
    if b.shape != (m,) or W.shape != (m, 2) or theta.shape != (3 * m,):
        raise ValueError("source candidate shapes must be b[8], W[8,2], theta[24]")
    if not np.isfinite(theta).all() or not np.array_equal(theta, np.r_[b, W.ravel()]):
        raise ValueError("source redundant parameters disagree or are nonfinite")
    a = np.linalg.norm(W, axis=1)
    if np.any(a <= 0):
        raise ValueError("canonical rank-one anchors require positive source row norms")
    return b.copy(), a, W.copy()


def construct_anchors(candidates, m=8):
    """Only b/W enter this construction; no frequency, teacher or risk input."""
    by_name = {c["name"]: c for c in candidates}
    if len(by_name) != len(candidates) or not {"K8", "REF"} <= set(by_name):
        raise ValueError("unique K8 and REF source candidates required")
    b8, s8, _ = checked_candidate(by_name["K8"], m)
    bR, sR, _ = checked_candidate(by_name["REF"], m)
    rows = []
    for name, b, a, bs, lam in (
            ("b8_s8", b8, s8, "K8", 0.), ("b8_sREF", b8, sR, "K8", 1.),
            ("bREF_s8", bR, s8, "REF", 0.), ("bREF_sREF", bR, sR, "REF", 1.)):
        rows.append(dict(name=name, b=b, a=a, bias_source=bs, lambda_value=lam))
    for suffix, lam in (("025", .25), ("050", .5), ("075", .75)):
        rows.append(dict(name="b8_lambda" + suffix, b=b8,
                         a=(1. - lam) * s8 + lam * sR, bias_source="K8", lambda_value=lam))
    for row in rows:
        row["W"] = np.column_stack((row["a"], np.zeros(m)))
        row["canonical_common_line"] = True
        row["original_training_endpoint"] = False
    return native(rows)


def target_vector(values, name):
    q = np.asarray(values, dtype=float)
    if q.shape != (256,) or not np.isfinite(q).all() or np.any(q < 0) or abs(float(q.sum()) - 1.) > 1e-12:
        raise ValueError("invalid fixed " + name + " target weights")
    return q


def load_stream(rep):
    """Read public JSON only; no old arrays are opened or rehashed."""
    folder = SOURCE / f"rep_{rep:02d}"
    paths = {n: folder / n for n in ("CANDIDATES_LOCK.json", "ENDPOINT_QUALIFICATION.json", "RESULTS.json")}
    lock, qualification, result = (read(paths[n]) for n in paths)
    lock_sha = sha(paths["CANDIDATES_LOCK.json"])
    if lock["rep"] != rep or result["rep"] != rep or result["status"] != "completed":
        raise ValueError("source stream is not completed with the requested rep")
    if qualification["candidates_lock_sha256"] != lock_sha or result["candidates_lock_sha256"] != lock_sha:
        raise ValueError("source candidate lock hash mismatch")
    if not qualification["all_qualified"] or not all(e["qualified"] for e in qualification["endpoints"]):
        raise ValueError("source endpoints are not all numerically qualified")
    if not lock["locked_before_teacher_risk_and_independent_cross_evaluation"]:
        raise ValueError("source parameter/teacher boundary was not locked")
    if result["candidates"] != lock["candidates"] or result["endpoints"] != qualification["endpoints"]:
        raise ValueError("source public candidates/qualifications disagree")
    frequency = target_vector(lock["frequency"], "empirical")
    if not np.array_equal(frequency, np.asarray(result["frequency"])):
        raise ValueError("source frequency records disagree")
    # No PMF or reported risk is passed to the anchor construction.
    anchors = construct_anchors(lock["candidates"])
    teacher = target_vector(qualification["teacher_pmf"], "population")
    main = [v for v in qualification["teacher_variants"] if v["order"] == 60 and v["rotation"] == 0]
    if len(main) != 1 or not np.array_equal(teacher, np.asarray(main[0]["pmf"])):
        raise ValueError("source teacher PMF is not its recorded GH60 main rule")
    if qualification["teacher_mass_error"] > 2e-12 or qualification["teacher_pmf_error"] > 2e-7:
        raise ValueError("source teacher PMF numerical qualification failed")
    return native(dict(rep=rep, folder=rel(folder), sources={rel(p): sha(p) for p in paths.values()},
        anchors=anchors, frequency=frequency, teacher_pmf=teacher,
        target_masses=[frequency.sum(), teacher.sum()], targets_renormalized=False,
        source_roles=dict(CANDIDATES_LOCK="fixed K8/REF b and W row norms; empirical frequency",
                          ENDPOINT_QUALIFICATION="fixed original teacher PMF and existing qualification",
                          RESULTS="completed-source identity and public-record consistency only"),
        eligibility=dict(completed=True, all_endpoints_qualified=True,
                         REF_global_MLE_claimed=False, old_arrays_read=False)))


def add_projection(result):
    Q = helmert_basis(result["laplacian"].shape[-1])
    projected = np.einsum("ia,tkij,jb->tkab", Q, result["laplacian"], Q, optimize=True)
    values, vectors = np.linalg.eigh(projected)
    result.update(helmert_basis=Q, projected_hessian=projected,
                  projected_eigenvalues=values, projected_eigenvectors=vectors,
                  angle_eigenvectors=np.einsum("ij,tkjl->tkil", Q, vectors, optimize=True))
    return result


def mass_diagnostics(result, ki):
    d = result["label_d_mass"][ki]
    c = result["label_c_mass"][ki]
    upper = float(result["c_mass_upper"][ki])
    di = int(np.argmax(abs(d - 1.)))
    return dict(normal_mass=float(result["normal_quadrature_mass"]),
        normal_mass_error=abs(float(result["normal_quadrature_mass"]) - 1.),
        d_mass_max_error=float(np.max(abs(d - 1.))), d_mass_worst_label=di,
        d_mass_at_worst_label=float(d[di]), d_mass_min=float(d.min()), d_mass_max=float(d.max()),
        c_mass_min=float(c.min()), c_mass_max=float(c.max()), c_mass_exact_upper=upper,
        c_mass_lower_violation=max(0., float(-c.min())),
        c_mass_upper_violation=max(0., float(c.max() - upper)),
        c_mass_min_label=int(np.argmin(c)), c_mass_max_label=int(np.argmax(c)),
        scope="all labels, including labels with zero empirical frequency")


def comparison(left, right, ti, ki):
    fields = dict(laplacian_max_difference="laplacian", projected_hessian_max_difference="projected_hessian",
                  projected_eigenvalue_max_difference="projected_eigenvalues", Hperp_max_difference="Hperp",
                  ward_angle_hessian_max_difference="ward_angle_hessian",
                  radial_gradient_max_difference="radial_gradient", bias_gradient_max_difference="bias_gradient")
    out = {name: float(np.max(abs(left[key][ti, ki] - right[key][ti, ki]))) for name, key in fields.items()}
    for key in ("label_d_mass", "label_c_mass", "label_M", "label_radial_gradient", "label_bias_gradient"):
        out[key + "_max_difference"] = float(np.max(abs(left[key][ki] - right[key][ki])))
    return out


def intrinsic_gates(result, ti, ki):
    scalar = mass_diagnostics(result, ki)
    gates = dict(independent_Ward=float(result["ward_residual_max"][ti, ki]) <= THRESHOLDS["ward_residual"])
    for label, threshold in (("normal_mass_error", "normal_mass_error"), ("d_mass_max_error", "d_mass_error"),
                             ("c_mass_lower_violation", "c_mass_error"), ("c_mass_upper_violation", "c_mass_error")):
        gates[label] = scalar[label] <= THRESHOLDS[threshold]
    return gates


def classify(minimum, qualified):
    if not qualified:
        return "unresolved_numerical_qualification"
    if minimum > THRESHOLDS["sign_resolution"]:
        return "numerically_qualified_angular_local_minimum"
    if minimum < -THRESHOLDS["sign_resolution"]:
        return "numerically_qualified_negative_angular_curvature"
    return "unresolved_at_fixed_sign_resolution"


def qualify(C, D, ti, ki):
    diff = comparison(C, D, ti, ki)
    gates = dict(matrix_difference=diff["laplacian_max_difference"] <= THRESHOLDS["matrix_difference"],
                 projected_eigenvalue_difference=diff["projected_eigenvalue_max_difference"] <= THRESHOLDS["eigenvalue_difference"])
    scalar = {}
    for name, result in (("C", C), ("D", D)):
        scalar[name] = mass_diagnostics(result, ki)
        gates.update({name + "_" + label: passed for label, passed in intrinsic_gates(result, ti, ki).items()})
    passed = all(gates.values())
    minimum = float(C["projected_eigenvalues"][ti, ki, 0])
    return dict(qualified=passed, gates=gates, failed_gates=[key for key, ok in gates.items() if not ok],
                C_D=diff, scalar_diagnostics=scalar, C_minimum_projected_eigenvalue=minimum,
                D_minimum_projected_eigenvalue=float(D["projected_eigenvalues"][ti, ki, 0]),
                classification=classify(minimum, passed),
                raw_C_sign_classification_ignoring_qualification=classify(minimum, True))


def rule_summary(result, a, seconds):
    rows = []
    for ti, target in enumerate(result["target_names"]):
        for ki, K in enumerate(result["K_values"]):
            ga, gb = result["radial_gradient"][ti, ki], result["bias_gradient"][ti, ki]
            direct = result["ward_angle_hessian"][ti, ki]
            diagnostic_gates = intrinsic_gates(result, ti, ki)
            rows.append(dict(target=target, K=K,
                **{name: result[name][ti, ki] for name in (
                    "laplacian", "edge_weights", "Hperp", "ward_angle_hessian", "ward_residual",
                    "eigenvalues", "projected_hessian", "projected_eigenvalues", "projected_eigenvectors", "angle_eigenvectors")},
                radial_gradient=ga, bias_gradient=gb, radial_gradient_L2=float(np.linalg.norm(ga)),
                radial_gradient_Linf=float(np.max(abs(ga))), bias_gradient_L2=float(np.linalg.norm(gb)),
                bias_gradient_Linf=float(np.max(abs(gb))),
                independent_Ward_max=float(result["ward_residual_max"][ti, ki]),
                direct_common_angle_residual=float(np.max(abs(direct.sum(axis=-1)))),
                Cartesian_Ward_max=float(np.max(abs(result["Hperp"][ti, ki] @ a - ga))),
                structural_laplacian_rowsum_max=float(result["common_angle_residual"][ti, ki]),
                structural_rowsum_is_not_Ward_verification=True,
                target_d_mass=float(result["target_d_mass"][ti, ki]),
                target_c_mass=float(result["target_c_mass"][ti, ki]),
                intrinsic_diagnostic_gates=diagnostic_gates,
                failed_intrinsic_diagnostic_gates=[key for key, passed in diagnostic_gates.items() if not passed],
                scalar_diagnostics=mass_diagnostics(result, ki)))
    return native(dict(config=result["config"], seconds=seconds, work_counts=result["work_counts"],
                       qualifications=result["qualifications"], matrices=rows))


def anchor_summary(anchor, results):
    rows = []
    for ti, target in enumerate(TARGET_NAMES):
        for ki, K in enumerate(K_VALUES):
            rows.append(dict(target=target, K=K, **qualify(results["C"], results["D"], ti, ki),
                all_rule_pair_comparisons={left + "_" + right: comparison(results[left], results[right], ti, ki)
                    for left, right in combinations(RULES, 2)}))
    return native(dict(anchor=anchor, primary_rule="C", comparison_rule="D", targets=rows,
        all_primary_matrices_qualified=all(row["qualified"] for row in rows),
        coarse_rules_A_B_retained_but_not_used_as_primary_gates=True))


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), float_dtype="binary64",
                thread_environment={name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})


def self_test():
    # Importing this module does not read scientific source JSON or arrays.
    import rankone_stability_compute_test as test_module
    return test_module.run_tests()


def freeze():
    if FREEZE.exists():
        raise FileExistsError("FREEZE.json exists and will not be replaced")
    if any((OUT / f"rep_{rep:02d}").exists() for rep in range(4)):
        raise FileExistsError("study output directories already exist; a new pre-execution freeze cannot be claimed")
    source_paths = [Path(__file__), Path(kernel.__file__), DRIVER_TEST,
        Path(kernel.__file__).with_name("rankone_stability_kernel_test.py"),
        DOC / "DESIGN.md", DOC / "PROOF.md", DOC / "ANALYSIS_RULES.md", OUT / "INPUTS_SCOPE.json", KERNEL_TEST]
    # A stale manufactured-kernel receipt is rejected, not silently relabelled.
    receipt = read(KERNEL_TEST)
    test_receipt = json.loads(receipt["stdout"])
    if receipt["source_sha256"] != sha(kernel.__file__) or receipt["test_sha256"] != sha(source_paths[3]):
        raise ValueError("kernel manufactured-test receipt does not bind the current source/test")
    if receipt["exit_code"] != 0 or not receipt["manufactured_only"] or receipt["scientific_data_used"] or not test_receipt["passed"]:
        raise ValueError("kernel manufactured qualification did not pass without scientific data")
    scope = read(OUT / "INPUTS_SCOPE.json")
    if scope["design_sha256"] != sha(DOC / "DESIGN.md"):
        raise ValueError("design differs from the recorded pre-execution scope")
    manufactured = self_test()
    if not manufactured["passed"]:
        raise ValueError("driver manufactured qualification failed")
    streams = [load_stream(rep) for rep in range(4)]
    sources = {rel(path): sha(path) for path in source_paths}
    save(FREEZE, dict(created_utc=now(), config=CONFIG, sources=sources, streams=streams,
        manufactured_qualification=manufactured, environment=environment(),
        all_28_anchors_fixed_before_quadrature=True, old_arrays_read=False,
        new_random_draws=0, optimizer_updates=0, statistical_alpha_spent=0,
        expected_primary_matrices=168, expected_all_rule_matrices=672,
        target_weights_renormalized=False, scientific_run_started=False))
    print(json.dumps(dict(frozen=True, path=rel(FREEZE), sha256=sha(FREEZE))), flush=True)


def verify_freeze():
    frozen = read(FREEZE)
    if frozen["config"] != native(CONFIG):
        raise ValueError("driver configuration differs from the freeze")
    for name, digest in frozen["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("changed frozen source: " + name)
    if [s["rep"] for s in frozen["streams"]] != list(range(4)):
        raise ValueError("freeze must bind all four ordered streams")
    for stream in frozen["streams"]:
        for name, digest in stream["sources"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("changed frozen public source: " + name)
    return frozen


def new_counts():
    return dict(successful_kernel_calls=0, failed_kernel_calls=0, new_random_draws=0,
                new_observations=0, optimizer_updates=0, statistical_alpha_spent=0,
                per_label_K_cells=0, all_rule_target_K_matrices=0,
                high_level_factor_totals={},
                interpretation="kernel high-level scalar factors; not CPU instructions or scipy internal exp calls")


def record_counts(counts, result):
    counts["successful_kernel_calls"] += 1
    counts["per_label_K_cells"] += len(result["patterns"]) * len(result["K_values"])
    counts["all_rule_target_K_matrices"] += len(result["target_names"]) * len(result["K_values"])
    # Sum node counts as visits across kernel calls, not as unique nodes.
    for name, value in result["work_counts"].items():
        if isinstance(value, int):
            counts["high_level_factor_totals"][name] = counts["high_level_factor_totals"].get(name, 0) + value


def run(rep):
    if rep not in range(4):
        raise ValueError("rep must be 0, 1, 2 or 3")
    frozen = verify_freeze()
    stream = frozen["streams"][rep]
    # Detect accidental changes to the construction, despite source hashing.
    if load_stream(rep) != stream:
        raise ValueError("public source interpretation differs from its frozen representation")
    folder = OUT / f"rep_{rep:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    counts, anchor_rows, rule_rows = new_counts(), [], []
    current = None
    try:
        save(folder / "RUN_START.json", dict(created_utc=now(), rep=rep, freeze_sha256=sha(FREEZE),
            environment=environment(), source_stream=stream["folder"], new_random_draws=0))
        targets = {"empirical": np.asarray(stream["frequency"]), "population": np.asarray(stream["teacher_pmf"])}
        save(folder / "ANCHORS_LOCK.json", dict(rep=rep, freeze_sha256=sha(FREEZE),
            anchors=stream["anchors"], targets=targets, target_weights_renormalized=False,
            source_hashes=stream["sources"], locked_before_rankone_quadrature=True))
        for anchor in stream["anchors"]:
            results = {}
            for name, settings in RULES.items():
                current = dict(anchor=anchor["name"], rule=name)
                tick = time.perf_counter()
                try:
                    result = kernel.rankone_stability(anchor["b"], anchor["a"], targets,
                        K_values=K_VALUES, **settings, compute_loss=False, return_coefficients=False)
                except BaseException:
                    counts["failed_kernel_calls"] += 1
                    raise
                seconds = time.perf_counter() - tick
                record_counts(counts, result)
                add_projection(result)
                results[name] = result
                stem = anchor["name"] + "_" + name
                arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
                arrays.update(b=np.asarray(anchor["b"]), a=np.asarray(anchor["a"]),
                    target_weights=np.asarray(list(targets.values())), target_names=np.asarray(TARGET_NAMES),
                    K_names=np.asarray([str(K) for K in K_VALUES]),
                    normal_quadrature_mass=np.asarray(result["normal_quadrature_mass"]))
                npz = folder / (stem + ".npz")
                save_npz(npz, arrays)
                summary = dict(anchor_name=anchor["name"], rule=name, **rule_summary(result, np.asarray(anchor["a"]), seconds),
                               array_artifact=artifact(npz))
                js = folder / (stem + ".json")
                save(js, summary)
                rule_rows.append(dict(anchor_name=anchor["name"], rule=name, seconds=seconds,
                                      summary_artifact=artifact(js), array_artifact=artifact(npz), work_counts=result["work_counts"]))
                print(json.dumps(dict(rep=rep, anchor=anchor["name"], rule=name, seconds=seconds)), flush=True)
            row = anchor_summary(anchor, results)
            save(folder / (anchor["name"] + "_QUALIFICATION.json"), row)
            anchor_rows.append(row)
        verify_freeze()
        qualified = sum(row["qualified"] for anchor in anchor_rows for row in anchor["targets"])
        save(folder / "RESULTS.json", dict(status="completed", rep=rep, freeze_sha256=sha(FREEZE),
            source_hashes=stream["sources"], anchors_lock_sha256=sha(folder / "ANCHORS_LOCK.json"),
            anchors=anchor_rows, rule_results=rule_rows, primary_matrices=42, all_rule_matrices=168,
            qualified_primary_matrices=qualified, unresolved_primary_matrices=42 - qualified,
            all_primary_matrices_qualified=qualified == 42, target_masses=stream["target_masses"],
            target_weights_renormalized=False, resources=counts, seconds=time.perf_counter() - started,
            artifacts={p.name: artifact(p) for p in folder.iterdir() if p.is_file()},
            claims=dict(conditional_on_fixed_old_targets=True, numerical_quadrature_only=True,
                rigorous_Gaussian_or_outer_remainder_certificate=False, all_parameter_stationarity=False,
                full_parameter_local_minimum=False, global_optimum=False,
                optimizer_convergence=False, training_population_probability=False,
                old_scientific_results_changed=False, no_new_statistical_family=True)))
        print(json.dumps(dict(completed=True, rep=rep, primary_matrices=42,
            qualified_primary_matrices=qualified, seconds=time.perf_counter() - started)), flush=True)
    except BaseException as error:
        save(folder / "FAILURE.json", dict(rep=rep, freeze_sha256=sha(FREEZE), current=current,
            reason=str(error), traceback=traceback.format_exc(), resources=counts,
            completed_anchor_count=len(anchor_rows), completed_rule_outputs=rule_rows,
            seconds=time.perf_counter() - started, partial_internal_work_of_failed_kernel_not_counted=True,
            retries_or_rule_changes_started=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--rep", type=int, choices=range(4))
    args = parser.parse_args()
    if args.run != (args.rep is not None):
        parser.error("--run and --rep N must be supplied together")
    if args.self_test:
        outcome = self_test()
        print(json.dumps(outcome, sort_keys=True), flush=True)
        raise SystemExit(0 if outcome["passed"] else 1)
    elif args.freeze:
        freeze()
    else:
        run(args.rep)
