"""Frozen two-stage rank-one F8 solves and subsequent locked evaluations.

SOURCE.json and FREEZE.json are prepared by the coordinating author. This
driver never creates them. --optimize solves all five fixed objects and locks
all endpoints before --evaluate may consume control targets or probe risks.
--self-test uses manufactured fixtures only. No prior samples are generated.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
import rankone_stationary_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/RankOneStationary_v1_20261007"
OUT = ROOT / "artifacts/rankone_stationary_v1_20261007"
SOURCE = OUT / "SOURCE.json"
FREEZE = OUT / "FREEZE.json"
ENDPOINTS = OUT / "ENDPOINTS_LOCK.json"
SOLVE_NAMES = ("E0", "E1", "E2", "E3", "P")
KS = (8, 64, "inf")
RULES = {
    "O": dict(normal_rule="legendre", normal_order=128, normal_limit=10., logu_min=-24., logu_max=48., logu_step=.25, outer_rule="trapezoid"),
    "C": dict(normal_rule="legendre", normal_order=256, normal_limit=10., logu_min=-28., logu_max=56., logu_step=.25, outer_rule="trapezoid"),
    "D": dict(normal_rule="legendre", normal_order=384, normal_limit=11., logu_min=-32., logu_max=64., logu_step=.125, outer_rule="trapezoid"),
}
STAGES = (dict(name="O", max_accepted_steps=12, max_calls=80, gradient_tolerance=1e-9),
          dict(name="C", max_accepted_steps=4, max_calls=40, gradient_tolerance=1e-10))
SOLVER = dict(eigenvalue_floor=1e-4, direction_Linf_cap=.25, boundary_fraction=.99,
              maximum_halvings=12, armijo_constant=1e-4, loss_comparison_slack=5e-13)
GATES = dict(loss=1e-7, gradient=2e-7, matrix=2e-7, eigenvalue=2e-6,
    normal_mass=2e-12, d_relative_mass=2e-7, c_raw_mass=2e-7, Ward=2e-7,
    fit_gradient=1e-8, box_margin=1e-3, positive_eigenvalue=1e-5,
    Cartesian_Ward=2e-7, rotation_residual=2e-7,
    PMF_difference=2e-7, PMF_mass=2e-12, KL_difference=1e-7,
    probe_gradient_norm_floor=1e-10)
PROBE_T = (-.02, -.01, 0., .01, .02)


def config_dump():
    return native(dict(schema="rankone_stationary_config_v1", dimension=8,
        coordinate_order="b_then_a", solve_names=SOLVE_NAMES, fitted_K=8, evaluation_K=KS,
        rules=RULES, stages=STAGES, solver=SOLVER, gates=GATES,
        bias_box=[-2., 2.], amplitude_box=[.05, 2.], probe_t=PROBE_T,
        scalar_gate_scope="all_256_labels", target_weights_renormalized=False,
        eigenvalue_and_Newton_input="symmetric_part_of_saved_raw_Hessian_without_spectral_clipping",
        probe_direction="negative_D_integrated_gradient_divided_by_its_L2_norm"))


def now():
    return datetime.now(timezone.utc).isoformat()


def native(value):
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return native(value.item())
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("nonfinite JSON value")
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(native(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_npz(path, arrays):
    with Path(path).open("xb") as f: np.savez_compressed(f, **arrays)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): h.update(chunk)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def artifact(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), threads={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})


def bounds(m):
    return np.r_[np.full(m, -2.), np.full(m, .05)], np.full(2 * m, 2.)


def box_record(theta):
    theta = np.asarray(theta, dtype=float)
    lower, upper = bounds(len(theta) // 2)
    margins = np.minimum(theta - lower, upper - theta)
    return dict(feasible=bool(np.isfinite(theta).all() and np.all(margins >= 0)),
                minimum_margin=float(margins.min()), coordinate_margins=margins.copy())


def sym(H):
    return .5 * (H + H.swapaxes(-1, -2))


def helmert(m):
    Q = np.zeros((m, m - 1))
    for j in range(m - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        Q[:j + 1, j] = 1. / scale
        Q[j + 1, j] = -(j + 1) / scale
    return Q


def transverse_basis(a):
    """Fixed Householder basis of a-perp, including an explicit sign rule."""
    a = np.asarray(a, dtype=float)
    norm = float(np.linalg.norm(a))
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or norm == 0:
        raise ValueError("nonzero finite amplitude vector with m>=2 required")
    u = a / norm
    v = u.copy(); v[0] += 1. if u[0] >= 0 else -1.
    H = np.eye(len(a)) - 2. * np.outer(v, v) / float(v @ v)
    return H[:, 1:].copy()


def project_result(result):
    a = result["a"]
    V, Q = transverse_basis(a), helmert(len(a))
    result["transverse_basis"] = V.copy(); result["helmert_basis"] = Q.copy()
    for name, H in (("H_ba", result["H_ba"]),
                    ("Hperp_projected", np.einsum("ia,tkij,jb->tkab", V, result["Hperp"], V, optimize=True)),
                    ("Hangle_projected", np.einsum("ia,tkij,jb->tkab", Q, result["Hangle"], Q, optimize=True))):
        if name != "H_ba": result[name] = H.copy()
        result[name + "_symmetry_residual"] = np.max(abs(H - H.swapaxes(-1, -2)), axis=(-2, -1))
        val, vec = np.linalg.eigh(sym(H))
        result[name + "_eigenvalues"] = val.copy()
        result[name + "_eigenvectors"] = vec.copy()
    return result


def scalar_diagnostics(result, ki):
    expected = float(result["raw_bank_quadrature_mass"][ki])
    d, c = result["label_d_mass"][ki], result["label_c_mass"][ki]
    upper = float(result["raw_c_mass_upper"][ki])
    return dict(normal_mass=float(result["normal_quadrature_mass"]),
        normal_mass_error=abs(float(result["normal_quadrature_mass"]) - 1.),
        expected_raw_d_mass=expected,
        d_relative_mass_error=float(np.max(abs(d / expected - 1.))),
        d_raw_min=float(d.min()), d_raw_max=float(d.max()),
        c_raw_lower_violation=max(0., float(-c.min())), c_raw_upper_violation=max(0., float(c.max() - upper)),
        c_raw_min=float(c.min()), c_raw_max=float(c.max()), c_raw_upper=upper,
        target_weights_mass=result["config"]["target_masses"].copy(), scope="all labels")


def qualification(C, D, ti, ki):
    """Prescribed blockwise C/D comparison; no hidden adaptive thresholds."""
    comparisons = {name: float(np.max(abs(C[name][ti, ki] - D[name][ti, ki]))) for name in
        ("loss", "gradient", "H_ba", "Hperp", "Hangle", "H_ba_eigenvalues", "Hperp_projected_eigenvalues", "Hangle_projected_eigenvalues")}
    gates = dict(loss=comparisons["loss"] <= GATES["loss"], gradient=comparisons["gradient"] <= GATES["gradient"])
    for name in ("H_ba", "Hperp", "Hangle"): gates[name] = comparisons[name] <= GATES["matrix"]
    for name in ("H_ba_eigenvalues", "Hperp_projected_eigenvalues", "Hangle_projected_eigenvalues"):
        gates[name] = comparisons[name] <= GATES["eigenvalue"]
    scalars = {}
    for name, result in (("C", C), ("D", D)):
        diag = scalar_diagnostics(result, ki); scalars[name] = diag
        gates[name + "_normal_mass"] = diag["normal_mass_error"] <= GATES["normal_mass"]
        gates[name + "_d_relative_mass"] = diag["d_relative_mass_error"] <= GATES["d_relative_mass"]
        gates[name + "_c_lower"] = diag["c_raw_lower_violation"] <= GATES["c_raw_mass"]
        gates[name + "_c_upper"] = diag["c_raw_upper_violation"] <= GATES["c_raw_mass"]
        gates[name + "_Ward"] = float(result["ward_residual_max"][ti, ki]) <= GATES["Ward"]
    return native(dict(qualified=all(gates.values()), gates=gates,
        failed_gates=[key for key, passed in gates.items() if not passed], comparisons=comparisons,
        scalar_diagnostics=scalars, rigorous_remainder_certificate=False))


def classification(theta, D, ti, ki, numerical_qualification):
    """Only call this for the endpoint's own fitted target at K8."""
    theta = np.asarray(theta)
    m = len(theta) // 2; a = theta[m:]
    g = D["gradient"][ti, ki]
    H = D["Hperp"][ti, ki]
    cart = float(np.max(abs(H @ a - g[m:])))
    rotation = float(np.max(abs(H @ (a / np.linalg.norm(a)))))
    margin = box_record(theta)["minimum_margin"]
    gb = float(np.max(abs(g)))
    bmin = float(D["H_ba_eigenvalues"][ti, ki, 0])
    pmin = float(D["Hperp_projected_eigenvalues"][ti, ki, 0])
    gates = dict(numerical_qualification=bool(numerical_qualification["qualified"]),
        gradient=gb <= GATES["fit_gradient"], box_margin=margin >= GATES["box_margin"],
        H_ba_positive=bmin > GATES["positive_eigenvalue"], Hperp_transverse_positive=pmin > GATES["positive_eigenvalue"],
        Cartesian_Ward=cart <= GATES["Cartesian_Ward"], near_rotation_mode=rotation <= GATES["rotation_residual"])
    passed = all(gates.values())
    return native(dict(qualified_full_local_minimum_signature=passed,
        classification="numerically_qualified_full_nonrotation_local_minimum_signature" if passed else "unresolved_full_local_minimum_signature",
        gates=gates, failed_gates=[key for key, value in gates.items() if not value],
        D_gradient_Linf=gb, minimum_box_margin=margin, D_H_ba_minimum=bmin,
        D_Hperp_transverse_minimum=pmin, D_Cartesian_Ward_Linf=cart, D_near_rotation_residual_Linf=rotation,
        nonrotation_dimensions=3 * m - 1, own_fitted_K8_target_only=True,
        interval_existence_or_uniqueness_certificate=False, global_optimum_claimed=False))


def stationary_solve(theta0, evaluator, stages=STAGES, trace=None):
    """Solve using an injected deterministic evaluator; all tape arrays copy.

    evaluator(theta_copy, phase, compute_hessian, context) returns a kernel
    result for one target and K8. It may persist every evaluation before this
    function tests Armijo acceptance. `trace` remains inspectable on failure.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    if theta.ndim != 1 or len(theta) % 2 or not box_record(theta)["feasible"]:
        raise ValueError("initial theta is not finite and feasible")
    if trace is None: trace = {}
    trace.update(initial_theta=theta.copy(), calls=[], attempted_calls=[], steps=[], stages=[],
                 accepted_states=[dict(state_id=0, phase="initial", call_id=None, theta=theta.copy())])
    for stage in stages:
        phase = stage["name"]
        sr = dict(phase=phase, start_theta=theta.copy(), call_count=0, accepted_step_count=0,
                  max_calls=stage["max_calls"], max_accepted_steps=stage["max_accepted_steps"],
                  gradient_tolerance=stage["gradient_tolerance"], stop_reason=None)
        trace["stages"].append(sr)

        def call(at, with_hessian, role, **metadata):
            if sr["call_count"] >= stage["max_calls"]: raise RuntimeError("internal call-budget violation")
            cid = len(trace["calls"])
            sr["call_count"] += 1
            context = dict(call_id=cid, phase=phase, role=role,
                           phase_call_index=sr["call_count"], **metadata)
            attempt = dict(**context, theta=at.copy(), compute_hessian=with_hessian, status="started")
            trace["attempted_calls"].append(attempt)
            try:
                result = evaluator(at.copy(), phase, with_hessian, copy.deepcopy(context))
                f = float(result["loss"][0, 0]); g = np.asarray(result["gradient"][0, 0]).copy()
                H = None if not with_hessian else np.asarray(result["H_ba"][0, 0]).copy()
                if g.shape != theta.shape or not math.isfinite(f) or not np.isfinite(g).all() or with_hessian and (H.shape != (len(theta), len(theta)) or not np.isfinite(H).all()):
                    raise FloatingPointError("nonfinite or malformed solver loss/gradient/Hessian")
            except BaseException as error:
                attempt.update(status="failed", reason=str(error))
                raise
            attempt["status"] = "completed"
            row = dict(**context, theta=at.copy(), loss=f, gradient=g, H_ba=H,
                       compute_hessian=with_hessian, box=box_record(at))
            trace["calls"].append(row)
            return row

        while sr["stop_reason"] is None:
            if sr["accepted_step_count"] >= stage["max_accepted_steps"]:
                sr["stop_reason"] = "accepted_step_budget"; break
            if sr["call_count"] >= stage["max_calls"]:
                sr["stop_reason"] = "function_call_budget"; break
            current = call(theta, True, "current")
            f, g, raw = current["loss"], current["gradient"].copy(), current["H_ba"].copy()
            if float(np.max(abs(g))) <= stage["gradient_tolerance"]:
                sr["stop_reason"] = "gradient_tolerance"; break
            symmetric = sym(raw)
            eigenvalues = np.linalg.eigvalsh(symmetric)
            mu = max(0., SOLVER["eigenvalue_floor"] - float(eigenvalues[0]))
            raw_d = np.linalg.solve(symmetric + mu * np.eye(len(theta)), -g)
            raw_max = float(np.max(abs(raw_d)))
            if not np.isfinite(raw_d).all() or raw_max == 0: raise FloatingPointError("invalid Newton direction")
            cap_scale = min(1., SOLVER["direction_Linf_cap"] / raw_max)
            d = raw_d * cap_scale
            gd = float(g @ d)
            if not gd < 0: raise FloatingPointError("damped Newton direction is not descent")
            lower, upper = bounds(len(theta) // 2)
            ratios = np.full(len(theta), np.inf)
            positive, negative = d > 0, d < 0
            ratios[positive] = (upper[positive] - theta[positive]) / d[positive]
            ratios[negative] = (lower[negative] - theta[negative]) / d[negative]
            max_alpha = float(ratios.min())
            initial_alpha = min(1., SOLVER["boundary_fraction"] * max_alpha)
            step = dict(step_id=len(trace["steps"]), phase=phase,
                phase_step_index=sr["accepted_step_count"], current_call_id=current["call_id"],
                theta=theta.copy(), loss=f, gradient=g.copy(), raw_H=raw.copy(), sym_H=symmetric.copy(),
                skew_max=float(np.max(abs(raw - raw.T))), eigenvalues=eigenvalues.copy(), mu=mu,
                raw_direction=raw_d.copy(), capped_direction=d.copy(), cap_scale=cap_scale,
                box_max_alpha=max_alpha, initial_alpha=initial_alpha, g_dot_d=gd, trials=[])
            trace["steps"].append(step)
            if initial_alpha <= 0:
                sr["stop_reason"] = "no_positive_feasible_box_step"; break
            accepted = False
            for halves in range(SOLVER["maximum_halvings"] + 1):
                if sr["call_count"] >= stage["max_calls"]:
                    sr["stop_reason"] = "function_call_budget"; break
                alpha = initial_alpha * (2. ** -halves)
                trial_theta = theta + alpha * d
                if not box_record(trial_theta)["feasible"]: raise FloatingPointError("trial left fixed box without projection")
                trial = call(trial_theta, False, "trial", step_id=step["step_id"], alpha=alpha, halvings=halves)
                rhs = f + SOLVER["armijo_constant"] * alpha * gd + SOLVER["loss_comparison_slack"]
                accepted = trial["loss"] <= rhs
                step["trials"].append(dict(call_id=trial["call_id"], alpha=alpha, halvings=halves,
                    theta=trial_theta.copy(), loss=trial["loss"], gradient=trial["gradient"].copy(),
                    armijo_rhs=rhs, accepted=accepted))
                if accepted:
                    theta = trial_theta.copy()
                    sr["accepted_step_count"] += 1
                    trace["accepted_states"].append(dict(state_id=len(trace["accepted_states"]), phase=phase,
                        call_id=trial["call_id"], theta=theta.copy(), loss=trial["loss"],
                        gradient=trial["gradient"].copy(), box=box_record(theta)))
                    if float(np.max(abs(trial["gradient"]))) <= stage["gradient_tolerance"]:
                        sr["stop_reason"] = "gradient_tolerance_at_accepted_FG"
                    break
            if not accepted and sr["stop_reason"] is None: sr["stop_reason"] = "line_search_failed"
        sr["final_theta"] = theta.copy()
        sr["final_box"] = box_record(theta)
    trace.update(status="completed", final_theta=theta.copy(), final_box=box_record(theta),
        total_calls=len(trace["calls"]), accepted_step_count=len(trace["accepted_states"]) - 1,
        constrained_optimizer_convergence_claimed=False)
    return trace


def new_counts():
    return dict(successful_kernel_calls=0, failed_kernel_calls=0, full_Hessian_calls=0, FG_only_calls=0,
                target_K_blocks=0, high_level_factor_totals={}, new_random_draws=0, new_observations=0,
                statistical_alpha_spent=0, accepted_optimizer_steps=0)


def add_counts(counts, result, hessian):
    counts["successful_kernel_calls"] += 1
    counts["full_Hessian_calls" if hessian else "FG_only_calls"] += 1
    counts["target_K_blocks"] += len(result["target_names"]) * len(result["K_values"])
    for key, value in result["work_counts"].items():
        if isinstance(value, int): counts["high_level_factor_totals"][key] = counts["high_level_factor_totals"].get(key, 0) + value


def kernel_call(theta, targets, Ks, rule, hessian, counts):
    m = len(theta) // 2
    tick = time.perf_counter()
    try:
        result = kernel.rankone_stationary(np.asarray(theta[:m]).copy(), np.asarray(theta[m:]).copy(),
            {key: np.asarray(q).copy() for key, q in targets.items()}, K_values=Ks,
            **RULES[rule], compute_hessian=hessian, return_coefficients=False)
    except BaseException:
        counts["failed_kernel_calls"] += 1; raise
    add_counts(counts, result, hessian)
    return result, time.perf_counter() - tick


def kernel_arrays(result, theta):
    arrays = {key: value.copy() for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(theta=np.asarray(theta).copy(), target_names=np.asarray(result["target_names"]),
        K_names=np.asarray([str(K) for K in result["K_values"]]),
        normal_quadrature_mass=np.asarray(result["normal_quadrature_mass"]))
    return arrays


def validate_source(source):
    if source["schema"] != "rankone_stationary_source_v1" or [s["name"] for s in source["solves"]] != list(SOLVE_NAMES):
        raise ValueError("SOURCE must list fixed E0/E1/E2/E3/P in order")
    base = ROOT / "artifacts/joint_dependence_v1_20261007"
    teacher = None
    for rep in range(4):
        lp, qp = (base / f"rep_{rep:02d}" / name for name in ("CANDIDATES_LOCK.json", "ENDPOINT_QUALIFICATION.json"))
        for path in (lp, qp):
            if source["sources"].get(rel(path)) != sha(path): raise ValueError("SOURCE public input binding mismatch")
        lock, qual = read(lp), read(qp)
        if qual["candidates_lock_sha256"] != sha(lp) or not qual["all_qualified"]:
            raise ValueError("source lock/endpoint qualification failed")
        candidates = [c for c in lock["candidates"] if c["name"] == "K8"]
        if len(candidates) != 1: raise ValueError("unique K8 candidate required")
        c = candidates[0]; b, W = np.asarray(c["b"]), np.asarray(c["W"])
        if b.shape != (8,) or W.shape != (8, 2) or not np.array_equal(c["theta"], np.r_[b, W.ravel()]):
            raise ValueError("source K8 parameter fields disagree")
        initial = np.r_[b, np.linalg.norm(W, axis=1)]
        target = np.asarray(lock["frequency"])
        s = source["solves"][rep]
        if s["target_name"] != "empirical" or not np.array_equal(s["theta0"], initial) or not np.array_equal(s["target"], target):
            raise ValueError("empirical source theta/target mismatch")
        current_teacher = np.asarray(qual["teacher_pmf"])
        if teacher is None: teacher = current_teacher.copy()
        elif not np.array_equal(teacher, current_teacher): raise ValueError("four public teacher PMFs differ")
    P = source["solves"][4]
    if P["target_name"] != "population" or not np.array_equal(P["theta0"], source["solves"][0]["theta0"]) or not np.array_equal(P["target"], teacher) or not np.array_equal(source["teacher_pmf"], teacher):
        raise ValueError("fixed population source mismatch")
    for s in source["solves"]:
        q = np.asarray(s["target"])
        if q.shape != (256,) or np.any(q < 0) or not np.isfinite(q).all() or abs(q.sum() - 1.) > 1e-12 or not box_record(s["theta0"])["feasible"]:
            raise ValueError("invalid source target or starting point")
    return copy.deepcopy(source)


def verify_freeze():
    frozen = read(FREEZE)
    if frozen["config"] != config_dump() or frozen["source_sha256"] != sha(SOURCE):
        raise ValueError("configuration/SOURCE differs from FREEZE")
    required = [Path(__file__), Path(__file__).with_name("rankone_stationary_compute_test.py"),
        Path(kernel.__file__), Path(kernel.__file__).with_name("rankone_stationary_kernel_test.py"),
        DOC / "DESIGN.md", DOC / "PROOF.md", OUT / "SCOPE.json", SOURCE]
    for path in required:
        if rel(path) not in frozen["sources"]: raise ValueError("required freeze binding missing: " + rel(path))
    for mapping in (frozen["sources"], read(SOURCE)["sources"]):
        for name, digest in mapping.items():
            if not isinstance(digest, str) or sha(ROOT / name) != digest: raise ValueError("changed frozen source: " + name)
    source = validate_source(read(SOURCE))
    return frozen, source


def tape_arrays(trace):
    n, d = len(trace["calls"]), len(trace["initial_theta"])
    Hessians = np.full((n, d, d), np.nan)
    for i, c in enumerate(trace["calls"]):
        if c["compute_hessian"]: Hessians[i] = c["H_ba"]
    return dict(call_id=np.array([c["call_id"] for c in trace["calls"]], dtype=int),
        call_theta=np.array([c["theta"] for c in trace["calls"]]).reshape(n, d),
        call_loss=np.array([c["loss"] for c in trace["calls"]]),
        call_gradient=np.array([c["gradient"] for c in trace["calls"]]).reshape(n, d),
        call_H_ba=Hessians, compute_hessian=np.array([c["compute_hessian"] for c in trace["calls"]], dtype=bool),
        phase=np.array([c["phase"] for c in trace["calls"]]), role=np.array([c["role"] for c in trace["calls"]]),
        accepted_theta=np.array([s["theta"] for s in trace["accepted_states"]]),
        accepted_call_id=np.array([-1 if s["call_id"] is None else s["call_id"] for s in trace["accepted_states"]]))


def optimize():
    _, source = verify_freeze()
    if ENDPOINTS.exists(): raise FileExistsError("endpoint lock already exists")
    folder = OUT / "optimization"; folder.mkdir(exist_ok=False)
    total = new_counts(); endpoints = []; started = time.perf_counter(); active = None
    try:
        save(folder / "RUN_START.json", dict(created_utc=now(), freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE), environment=environment()))
        for s in source["solves"]:
            active = s["name"]; target_folder = folder / active; target_folder.mkdir()
            trace = {}; counts = new_counts(); call_artifacts = []; tick = time.perf_counter()

            def evaluator(theta, phase, hessian, context):
                result, seconds = kernel_call(theta, {s["target_name"]: s["target"]}, (8,), phase, hessian, counts)
                stem = f"call_{context['call_id']:04d}"
                npz = target_folder / (stem + ".npz")
                save_npz(npz, kernel_arrays(result, theta))
                js = target_folder / (stem + ".json")
                save(js, dict(**context, theta=theta.copy(), loss=float(result["loss"][0, 0]),
                    gradient=result["gradient"][0, 0].copy(),
                    H_ba=result["H_ba"][0, 0].copy() if hessian else None,
                    compute_hessian=hessian, seconds=seconds, config=result["config"],
                    work_counts=result["work_counts"], array_artifact=artifact(npz)))
                call_artifacts.append(dict(call_id=context["call_id"], array_artifact=artifact(npz), summary_artifact=artifact(js)))
                return result

            try:
                stationary_solve(s["theta0"], evaluator, trace=trace)
                counts["accepted_optimizer_steps"] = trace["accepted_step_count"]
                tape = target_folder / "TAPE.npz"; save_npz(tape, tape_arrays(trace))
                verify_freeze()
                result_path = target_folder / "SOLVER.json"
                save(result_path, dict(name=active, fit_target_name=s["target_name"], fit_target=s["target"],
                    **trace, resources=counts, seconds=time.perf_counter() - tick,
                    call_artifacts=call_artifacts, tape_artifact=artifact(tape),
                    hessian_missing_fill_in_tape="NaN with compute_hessian=false; not an evaluated Hessian",
                    freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE)))
            except BaseException as error:
                counts["accepted_optimizer_steps"] = max(0, len(trace.get("accepted_states", [])) - 1)
                save(target_folder / "FAILURE.json", dict(name=active, reason=str(error), traceback=traceback.format_exc(),
                    partial_trace=trace, call_artifacts=call_artifacts, resources=counts, seconds=time.perf_counter() - tick,
                    failed_kernel_internal_partial_work_not_counted=True, automatic_restart=False))
                raise
            endpoints.append(dict(name=active, theta=trace["final_theta"].copy(), fit_target_name=s["target_name"],
                fit_target=np.asarray(s["target"]).copy(), solver_artifact=artifact(result_path), tape_artifact=artifact(tape),
                solver_stage_stop_reasons=[row["stop_reason"] for row in trace["stages"]], box=box_record(trace["final_theta"])))
            for key, value in counts.items():
                if isinstance(value, int): total[key] += value
            for key, value in counts["high_level_factor_totals"].items():
                total["high_level_factor_totals"][key] = total["high_level_factor_totals"].get(key, 0) + value
            print(json.dumps(dict(solve=active, locked_later_with_all_five=True, calls=trace["total_calls"], accepted_steps=trace["accepted_step_count"])), flush=True)
        verify_freeze()
        save(ENDPOINTS, dict(created_utc=now(), schema="rankone_stationary_endpoints_v1", endpoints=endpoints,
            freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE), resources=total,
            optimization_artifacts={rel(p): sha(p) for p in folder.rglob("*") if p.is_file()},
            all_five_locked_before_control_evaluations=True, teacher_risk_used_to_select_endpoints=False,
            seconds=time.perf_counter() - started))
        print(json.dumps(dict(optimized=True, endpoint_lock=artifact(ENDPOINTS), resources=total)), flush=True)
    except BaseException as error:
        save(folder / "FAILURE.json", dict(active_solve=active, reason=str(error), traceback=traceback.format_exc(),
            completed_endpoints=endpoints, completed_solves_resources=total, seconds=time.perf_counter() - started,
            all_five_locked=False, automatic_restart=False))
        raise


def probability_summary(result, teacher):
    q = np.asarray(teacher, dtype=float)
    p = np.exp(result["log_evidence"])
    positive = q > 0
    KL = float(q[positive] @ (np.log(q[positive]) - result["log_evidence"][positive]))
    return dict(pmf=p.copy(), pmf_mass=float(p.sum()), pmf_mass_error=abs(float(p.sum()) - 1.),
        teacher_mass=float(q.sum()), raw_KL=KL, formula="sum_y q_y log(q_y/p_y), original raw target weights",
        model_or_target_renormalized=False)


def evaluation_summary(theta, result, teacher, seconds):
    rows = []
    a = result["a"]
    fields = ("loss", "gradient", "bias_gradient", "radial_gradient", "H_ba", "Hperp", "Hangle", "directWard",
        "ward_residual", "H_ba_eigenvalues", "H_ba_eigenvectors", "Hperp_projected", "Hperp_projected_eigenvalues",
        "Hperp_projected_eigenvectors", "Hangle_projected", "Hangle_projected_eigenvalues", "Hangle_projected_eigenvectors")
    for ti, target in enumerate(result["target_names"]):
        for ki, K in enumerate(result["K_values"]):
            rows.append(dict(target=target, K=K, **{key: result[key][ti, ki].copy() for key in fields},
                gradient_Linf=float(np.max(abs(result["gradient"][ti, ki]))),
                raw_H_ba_symmetry_residual=float(result["H_ba_symmetry_residual"][ti, ki]),
                Ward_Linf=float(result["ward_residual_max"][ti, ki]),
                Cartesian_Ward_Linf=float(np.max(abs(result["Hperp"][ti, ki] @ a - result["radial_gradient"][ti, ki]))),
                near_rotation_residual_Linf=float(np.max(abs(result["Hperp"][ti, ki] @ (a / np.linalg.norm(a))))),
                structural_common_angle_residual=float(result["common_angle_residual"][ti, ki]),
                scalar_diagnostics=scalar_diagnostics(result, ki)))
    return native(dict(theta=np.asarray(theta).copy(), target_names=result["target_names"], K_values=result["K_values"],
        transverse_basis=result["transverse_basis"].copy(), helmert_basis=result["helmert_basis"].copy(),
        projection_definition="Householder u=a/norm(a), v=u+sign(u0)e0 (sign(0)=1), V=(I-2vvT/vTv)[:,1:]",
        raw_matrices_preserved=True, spectra_of_symmetric_parts=True, spectral_clipping=False,
        blocks=rows, probability=probability_summary(result, teacher), config=result["config"],
        work_counts=result["work_counts"], seconds=seconds, numerical_qualification_only=True))


def paired_evaluation(theta, targets, teacher, folder, stem, counts):
    result_by_rule, files = {}, {}
    for rule in ("C", "D"):
        result, seconds = kernel_call(theta, targets, KS, rule, True, counts)
        project_result(result)
        npz = folder / (stem + "_" + rule + ".npz")
        save_npz(npz, kernel_arrays(result, theta))
        js = folder / (stem + "_" + rule + ".json")
        save(js, dict(rule=rule, **evaluation_summary(theta, result, teacher, seconds), array_artifact=artifact(npz)))
        result_by_rule[rule] = result
        files[rule] = dict(array_artifact=artifact(npz), summary_artifact=artifact(js))
    rows = []
    for ti, target in enumerate(targets):
        for ki, K in enumerate(KS):
            rows.append(dict(target=target, K=K, **qualification(result_by_rule["C"], result_by_rule["D"], ti, ki)))
    return result_by_rule, dict(rule_artifacts=files, qualifications=rows)


def load_endpoint_lock(source):
    lock = read(ENDPOINTS)
    if lock["freeze_sha256"] != sha(FREEZE) or lock["source_sha256"] != sha(SOURCE) or not lock["all_five_locked_before_control_evaluations"]:
        raise ValueError("endpoint lock does not bind the current frozen five-solve procedure")
    if [e["name"] for e in lock["endpoints"]] != list(SOLVE_NAMES):
        raise ValueError("all five endpoints must be locked together before evaluation")
    for name, digest in lock["optimization_artifacts"].items():
        if sha(ROOT / name) != digest: raise ValueError("changed saved optimization artifact: " + name)
    for e, s in zip(lock["endpoints"], source["solves"]):
        solver = read(ROOT / e["solver_artifact"]["path"])
        if solver["status"] != "completed" or not box_record(e["theta"])["feasible"] or not np.array_equal(e["theta"], solver["final_theta"]):
            raise ValueError("endpoint is not the finite feasible LAST of its completed procedure")
        if e["fit_target_name"] != s["target_name"] or not np.array_equal(e["fit_target"], s["target"]):
            raise ValueError("endpoint fitted target mismatch")
    return lock


def probe_plan(theta, D_population, fit_classification):
    if not fit_classification["qualified_full_local_minimum_signature"]:
        return dict(status="skipped_P_fit_signature_not_qualified", evaluated=False)
    ki = list(D_population["K_values"]).index("inf")
    g = D_population["gradient"][0, ki].copy()
    norm = float(np.linalg.norm(g))
    if norm <= GATES["probe_gradient_norm_floor"]:
        return dict(status="undefined_integrated_gradient_too_small", integrated_gradient=g,
                    gradient_L2=norm, evaluated=False)
    d = -g / norm
    return dict(status="defined", direction=d.copy(), integrated_gradient=g.copy(), gradient_L2=norm,
        direction_source="D population integrated gradient at locked P endpoint",
        oracle_teacher_diagnostic=True, deployable_learning_algorithm=False,
        points=[dict(index=i, t=t, theta=(np.asarray(theta) + t * d).copy(), box=box_record(np.asarray(theta) + t * d)) for i, t in enumerate(PROBE_T)],
        direction_locked_before_probe_evaluation=True)


def run_probe(theta, D_population, fit_classification, teacher, folder, counts):
    plan = probe_plan(theta, D_population, fit_classification)
    path = folder / "PROBE_PLAN.json"; save(path, plan)
    if plan["status"] != "defined": return dict(**plan, plan_artifact=artifact(path), points=[])
    points = []
    for point in plan["points"]:
        if not point["box"]["feasible"]:
            points.append(dict(**point, status="infeasible_not_evaluated", step_reduced_or_projected=False))
            continue
        results, summary = paired_evaluation(point["theta"], {"population": teacher}, teacher,
            folder, f"point_{point['index']:02d}", counts)
        pc, pd = probability_summary(results["C"], teacher), probability_summary(results["D"], teacher)
        comparisons = dict(PMF_max_difference=float(np.max(abs(pc["pmf"] - pd["pmf"]))),
            C_PMF_mass_error=pc["pmf_mass_error"], D_PMF_mass_error=pd["pmf_mass_error"],
            KL_difference=abs(pc["raw_KL"] - pd["raw_KL"]))
        gates = dict(PMF_difference=comparisons["PMF_max_difference"] <= GATES["PMF_difference"],
            C_PMF_mass=comparisons["C_PMF_mass_error"] <= GATES["PMF_mass"],
            D_PMF_mass=comparisons["D_PMF_mass_error"] <= GATES["PMF_mass"],
            KL_difference=comparisons["KL_difference"] <= GATES["KL_difference"])
        row = dict(**point, status="evaluated", **summary, probability={"C": pc, "D": pd},
            PMF_KL_comparisons=comparisons, PMF_KL_gates=gates,
            PMF_KL_failed_gates=[key for key, passed in gates.items() if not passed],
            qualified=all(gates.values()) and all(b["qualified"] for b in summary["qualifications"]),
            step_reduced_or_projected=False, complete_stationarity_at_probe_point_claimed=False)
        save(folder / f"point_{point['index']:02d}_QUALIFICATION.json", row)
        points.append(row)
    return dict(status="completed_prespecified_probe", plan_artifact=artifact(path), direction=plan["direction"].copy(),
        direction_source=plan["direction_source"], points=points, oracle_teacher_diagnostic=True,
        eligible_point_count=sum(p["box"]["feasible"] for p in points), additional_optimization_steps=0)


def evaluate():
    _, source = verify_freeze(); lock = load_endpoint_lock(source)
    if (OUT / "RESULTS.json").exists(): raise FileExistsError("RESULTS already exists")
    folder = OUT / "evaluation"; folder.mkdir(exist_ok=False)
    counts = new_counts(); endpoint_rows = []; started = time.perf_counter(); active = None
    try:
        save(folder / "RUN_START.json", dict(created_utc=now(), endpoint_lock_sha256=sha(ENDPOINTS),
            freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE), environment=environment()))
        P_results = P_signature = None
        for endpoint in lock["endpoints"]:
            active = endpoint["name"]; theta = np.asarray(endpoint["theta"]).copy()
            targets = {"population": np.asarray(source["teacher_pmf"]).copy()}
            if active != "P": targets = {"empirical": np.asarray(endpoint["fit_target"]).copy(), **targets}
            results, summary = paired_evaluation(theta, targets, source["teacher_pmf"], folder, active, counts)
            fit_ti = list(targets).index(endpoint["fit_target_name"])
            fit_q = next(row for row in summary["qualifications"] if row["target"] == endpoint["fit_target_name"] and row["K"] == 8)
            signature = classification(theta, results["D"], fit_ti, 0, fit_q)
            row = dict(name=active, theta=theta.copy(), fit_target_name=endpoint["fit_target_name"],
                **summary, fitted_signature=signature,
                probability={rule: probability_summary(result, source["teacher_pmf"]) for rule, result in results.items()},
                control_blocks_not_claimed_stationary=True)
            save(folder / (active + "_QUALIFICATION.json"), row)
            endpoint_rows.append(row)
            if active == "P": P_results, P_signature = results, signature
            print(json.dumps(dict(evaluated_endpoint=active, fitted_signature=signature["classification"])), flush=True)
        probe_folder = folder / "probe"; probe_folder.mkdir()
        probe = run_probe(np.asarray(lock["endpoints"][4]["theta"]).copy(), P_results["D"], P_signature,
                          source["teacher_pmf"], probe_folder, counts)
        verify_freeze()
        # Check the endpoint identity again after evaluation without any refit.
        if sha(ENDPOINTS) != read(folder / "RUN_START.json")["endpoint_lock_sha256"]:
            raise ValueError("endpoint lock changed during evaluation")
        block_count = sum(len(e["qualifications"]) for e in endpoint_rows)
        if block_count != 27: raise RuntimeError("the fixed endpoint evaluation did not produce 27 blocks")
        save(OUT / "RESULTS.json", dict(status="completed", created_utc=now(), freeze_sha256=sha(FREEZE),
            source_sha256=sha(SOURCE), endpoint_lock_sha256=sha(ENDPOINTS), endpoints=endpoint_rows, probe=probe,
            endpoint_blocks=block_count, endpoint_rule_blocks=54,
            qualified_endpoint_blocks=sum(q["qualified"] for e in endpoint_rows for q in e["qualifications"]),
            fit_signatures_passed=sum(e["fitted_signature"]["qualified_full_local_minimum_signature"] for e in endpoint_rows),
            evaluation_resources=counts, optimization_resources=lock["resources"], seconds=time.perf_counter() - started,
            artifacts={rel(p): sha(p) for p in folder.rglob("*") if p.is_file()},
            claims=dict(numerical_signature_only=True, interval_stationary_existence_certificate=False,
                unique_stationary_point=False, global_optimum=False, P_is_independent_training_repeat=False,
                no_new_observations_or_prior=True, no_new_statistical_alpha=True, manuscript_changed=False,
                retained_target_mass=float(np.asarray(source["teacher_pmf"]).sum()))))
        print(json.dumps(dict(completed=True, endpoint_blocks=27, probe_status=probe["status"], evaluation_resources=counts)), flush=True)
    except BaseException as error:
        save(folder / "FAILURE.json", dict(active_endpoint=active, reason=str(error), traceback=traceback.format_exc(),
            completed_endpoint_rows=endpoint_rows, resources=counts, seconds=time.perf_counter() - started,
            failed_kernel_internal_partial_work_not_counted=True, automatic_restart=False))
        raise


def self_test():
    import rankone_stationary_compute_test as tests
    return tests.run_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--optimize", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.config: print(json.dumps(config_dump(), sort_keys=True), flush=True)
    elif args.self_test:
        outcome = self_test(); print(json.dumps(outcome, sort_keys=True), flush=True)
        raise SystemExit(0 if outcome["passed"] else 1)
    elif args.optimize: optimize()
    else: evaluate()
