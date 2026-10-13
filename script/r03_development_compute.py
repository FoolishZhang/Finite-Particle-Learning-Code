"""Bounded R03 development computation; no file I/O, RNG or execution on import.

The caller must freeze inputs/configuration before calling ``compute``.  All
returned values (and checkpoint states) are JSON-compatible.  This is float64
development evidence, never a rigorous continuation/root certificate.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import time

import numpy as np

from joint_dependence_kernel import model_pmf, patterns
from rankone_stationary_kernel import rankone_stationary


LAMBDAS = ("0", "1/100", "-1/100", "1/20", "-1/20", "1/10", "-1/10")
RULES = {
    "O": dict(normal_rule="legendre", normal_order=128, normal_limit=10.,
              logu_min=-24., logu_max=48., logu_step=.25),
    "C": dict(normal_rule="legendre", normal_order=256, normal_limit=10.,
              logu_min=-28., logu_max=56., logu_step=.25),
    "D": dict(normal_rule="legendre", normal_order=384, normal_limit=11.,
              logu_min=-32., logu_max=64., logu_step=.125),
}
GATES = dict(full_gradient_difference=1e-7, hessian_operator_difference=1e-5,
             fixed_risk_curvature_difference=1e-7, teacher_l1_difference=1e-10)


def serializable(value):
    """Independent JSON snapshot; reject nonfinite numbers instead of hiding them."""
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite output is not a JSON evidence value")
    return value


def transverse_basis(a):
    """Deterministic Householder basis of a-perp, with sign(0)=+1."""
    a = np.asarray(a, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all() or np.linalg.norm(a) == 0:
        raise ValueError("a must be a nonzero finite length-eight vector")
    v = a / np.linalg.norm(a)
    v[0] += 1. if v[0] >= 0 else -1.
    return (np.eye(8) - 2 * np.outer(v, v) / float(v @ v))[:, 1:]


def _sha(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fraction(value):
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(str(value))


def _sum_counts(total, counts):
    for key, value in counts.items():
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + int(value)


def _summary(result, q, a0, d):
    a = np.asarray(result["a"])
    V = transverse_basis(a)
    records = []
    for ki, K in enumerate(result["K_values"]):
        H = np.asarray(result["H_ba"])[0, ki]
        P = V.T @ np.asarray(result["Hperp"])[0, ki] @ V
        # eigvalsh sees the explicit symmetric part; the raw matrix and its
        # asymmetry are separately retained, and eigenvalues are never clipped.
        records.append(dict(K=K, loss=float(result["loss"][0, ki]),
            full_gradient_infinity=float(np.max(np.abs(result["gradient"][0, ki]))),
            full_gradient_l2=float(np.linalg.norm(result["gradient"][0, ki])),
            H_ba_eigenvalues=np.linalg.eigvalsh((H + H.T) / 2),
            Hperp_projected=P, Hperp_projected_eigenvalues=np.linalg.eigvalsh((P + P.T) / 2),
            full_H_ba_min_eigenvalue=float(np.linalg.eigvalsh((H + H.T) / 2)[0]),
            transverse_seven_min_eigenvalue=float(np.linalg.eigvalsh((P + P.T) / 2)[0]),
            fixed_angle_curvature=float(d @ np.asarray(result["Hangle"])[0, ki] @ d),
            ward_residual_max=float(result["ward_residual_max"][0, ki]),
            H_ba_asymmetry_max=float(np.max(np.abs(H - H.T))),
            common_angle_residual=float(result["common_angle_residual"][0, ki])))
    inf = list(result["K_values"]).index("inf")
    active = q > 0
    return serializable(dict(blocks=records, transverse_basis=V,
        basis_orthogonality_error=float(np.max(np.abs(V.T @ V - np.eye(7)))),
        basis_a_error=float(np.max(np.abs(V.T @ a))),
        a_drift=a-a0, a_drift_l2=float(np.linalg.norm(a-a0)),
        a_reversal_pair_differences=a-a[::-1],
        a_reversal_symmetrybreaking_max=float(np.max(np.abs(a-a[::-1]))),
        risk_KL_raw_quadrature=float(result["loss"][0, inf] + q[active] @ np.log(q[active])),
        candidate_pmf=np.exp(np.asarray(result["log_evidence"])),
        candidate_pmf_mass=float(np.exp(np.asarray(result["log_evidence"])).sum()),
        teacher_probability_mass=float(q.sum())))


def _qualification(C, D, d, gates):
    grad = float(np.max(np.abs(np.asarray(C["gradient"]) - np.asarray(D["gradient"]))))
    matrices = {}
    for name in ("H_ba", "Hperp", "Hangle"):
        matrices[name] = [float(np.linalg.norm(np.asarray(C[name])[0, k] -
                            np.asarray(D[name])[0, k], ord=2)) for k in range(2)]
    curv = abs(float(d @ (np.asarray(C["Hangle"])[0, 1] -
                         np.asarray(D["Hangle"])[0, 1]) @ d))
    checks = dict(full_gradient=grad <= gates["full_gradient_difference"],
                  hessian_operator=max(max(x) for x in matrices.values()) <= gates["hessian_operator_difference"],
                  fixed_risk_curvature=curv <= gates["fixed_risk_curvature_difference"])
    return dict(full_gradient_difference=grad, hessian_operator_differences=matrices,
                fixed_risk_curvature_difference=curv, checks=checks, passed=all(checks.values()))


class _Stop(Exception):
    pass


def _newton(a, evaluate, rule, max_iterations, tolerance, point):
    """Eight radial coordinates, b=0; injected evaluator enables fake-only tests."""
    a = np.asarray(a, dtype=float).copy()
    trace = dict(rule=rule, max_iterations=max_iterations, tolerance=tolerance,
                 iterations=[], status="running", initial_a=a.tolist(), final_a=a.tolist())
    point.setdefault("root_phases", []).append(trace)
    for iteration in range(max_iterations):
        value = evaluate(a, rule, (8,), True, "root_hessian", iteration)
        g = np.asarray(value["gradient"])[0, 0, 8:]
        norm = float(np.max(np.abs(g)))
        row = dict(iteration=iteration, a=a.tolist(), radial_gradient=g.tolist(),
                   full_gradient=np.asarray(value["gradient"])[0, 0].tolist(),
                   radial_gradient_infinity=norm, trials=[])
        trace["iterations"].append(row)
        if norm <= tolerance:
            trace.update(status="converged", final_gradient_infinity=norm)
            return a, True
        try:
            step = np.linalg.solve(np.asarray(value["H_ba"])[0, 0, 8:, 8:], -g)
        except np.linalg.LinAlgError as error:
            trace.update(status="singular_newton_matrix", error=str(error))
            return a, False
        if not np.isfinite(step).all():
            trace.update(status="nonfinite_newton_step")
            return a, False
        row["newton_step"] = step.tolist()
        accepted = False
        for halving in range(9):
            alpha = 2. ** (-halving)
            trial = a + alpha * step
            attempt = dict(halving=halving, alpha=alpha, a=trial.tolist(), accepted=False)
            row["trials"].append(attempt)
            if np.any(trial < .1) or np.any(trial > 1.):
                attempt["reason"] = "outside_closed_amplitude_box"
                continue
            candidate = evaluate(trial, rule, (8,), False, "root_line_search", iteration)
            tg = np.asarray(candidate["gradient"])[0, 0, 8:]
            trial_norm = float(np.max(np.abs(tg)))
            limit = (1 - 1e-4 * alpha) * norm
            attempt.update(radial_gradient=tg.tolist(), radial_gradient_infinity=trial_norm,
                           armijo_limit=limit)
            if trial_norm <= limit:
                attempt.update(accepted=True, reason="gradient_merit_decrease")
                a, accepted = trial, True
                trace.update(final_a=a.tolist(), final_gradient_infinity=trial_norm)
                if trial_norm <= tolerance:
                    trace["status"] = "converged"
                    return a, True
                break
            attempt["reason"] = "gradient_merit_not_decreased"
        if not accepted:
            trace["status"] = "line_search_exhausted"
            return a, False
    trace["status"] = "iteration_budget_exhausted"
    return a, False


def compute(initial_exact, direction, lambdas=LAMBDAS, *, wall_seconds=600.,
            gates=None, stop_requested=None, checkpoint=None,
            _stationary_kernel=None, _teacher_kernel=None, _clock=None):
    """Compute the frozen seven-point panel, returning partial evidence on stop.

    ``initial_exact`` is the saved eight ``a_exact`` rational dictionaries;
    ``direction`` contains ``u_integer`` and ``S_integer``.  ``checkpoint`` may
    persist the supplied JSON-compatible live state but must not mutate it.
    Its duration is excluded from this compute budget.  ``stop_requested`` is
    a no-argument callback.  Time/stop checks are cooperative at call boundaries;
    the caller may impose a process deadline.  Private injected kernels/clock
    are solely for manufactured implementation tests.
    """
    clock = time.perf_counter if _clock is None else _clock
    stationary = rankone_stationary if _stationary_kernel is None else _stationary_kernel
    teacher = model_pmf if _teacher_kernel is None else _teacher_kernel
    start, paused = clock(), 0.
    budget = float(wall_seconds)
    if not math.isfinite(budget) or budget <= 0 or budget > 600:
        raise ValueError("wall_seconds must be positive and at most 600")
    lam = tuple(_fraction(x) for x in lambdas)
    if lam != tuple(Fraction(x) for x in LAMBDAS):
        raise ValueError("this driver is frozen to the seven ordered development lambdas")
    exact = tuple(_fraction(x) for x in initial_exact)
    a0 = np.asarray([float(x) for x in exact])
    if a0.shape != (8,) or not np.isfinite(a0).all() or np.any(a0 < .1) or np.any(a0 > 1):
        raise ValueError("initial a must be eight exact values in [.1,1]")
    u = tuple(int(x) for x in direction["u_integer"])
    S = int(direction["S_integer"])
    if len(u) != 8 or sum(u) != 0 or sum(x*x for x in u) != S or S <= 0:
        raise ValueError("direction must satisfy length=8, sum(u)=0, S=sum(u^2)>0")
    d = np.asarray(u, dtype=float) / math.sqrt(S)
    selected_gates = dict(GATES)
    if gates is not None:
        if set(gates) != set(GATES) or any(not math.isfinite(float(x)) or x <= 0 for x in gates.values()):
            raise ValueError("gates must supply every declared positive finite gate")
        selected_gates = {k: float(v) for k, v in gates.items()}
    Y = patterns(8)
    spins = 2*Y.astype(float)-1
    reverse_index = Y[:, ::-1] @ (2 ** np.arange(7, -1, -1))
    output = dict(schema="r03_development_v1", status="running", config=dict(
        lambdas=[str(x) for x in lam], rules=RULES, gates=selected_gates,
        newton_iterations={"O": 10, "C": 6}, gradient_tolerance={"O": 1e-8, "C": 1e-11},
        armijo_gradient_merit=1e-4, halvings=list(range(9)), a_box=[.1, 1.],
        teacher_orders=[60, 80, 80], teacher_rotations=[0., 0., .271],
        wall_seconds=budget, timing="cooperative call boundaries; checkpoint time excluded",
        target_patterns="all 256 lexicographic labels; no class compression",
        positive_negative_branches="previous successfully qualified same-side point; baseline seeds both"),
        initial_a_exact=[dict(numerator=str(x.numerator), denominator=str(x.denominator)) for x in exact],
        direction=dict(u_integer=list(u), S_integer=S, definition="u/sqrt(S)", float64=d.tolist()),
        points=[], call_log=[], counts=dict(root_calls_started=0, root_calls_completed=0,
            final_calls_started=0, final_calls_completed=0, teacher_calls_started=0,
            teacher_calls_completed=0, successful_target_work={}, successful_teacher_work={}),
        qualifications=dict(rigorous_root=False, rigorous_continuous_range=False,
            new_random_samples=0, renormalization=False, silent_symmetry_average=False,
            float64_development_only=True))

    def elapsed():
        return clock()-start-paused

    def emit():
        nonlocal paused
        output["elapsed_compute_seconds"] = elapsed()
        if checkpoint is not None:
            before = clock()
            checkpoint(output)
            paused += clock()-before

    def guard():
        if stop_requested is not None and stop_requested():
            raise _Stop("caller_stop_requested")
        if elapsed() >= budget:
            raise _Stop("wall_clock_budget")

    def call_kernel(point, q, a, rule, Ks, hessian, purpose, iteration):
        guard()
        group = "root" if purpose.startswith("root_") else "final"
        output["counts"][group+"_calls_started"] += 1
        record = dict(call_id=len(output["call_log"]), point_index=point["index"],
            lambda_exact=point["lambda_exact"], q_sha256=_sha(q), b=[0.]*8,
            a=a.tolist(), rule=rule, rule_parameters=RULES[rule], K_values=list(Ks),
            compute_hessian=hessian, purpose=purpose, iteration=iteration,
            status="started", start_compute_seconds=elapsed())
        output["call_log"].append(record)
        emit()
        try:
            value = stationary(np.zeros(8), a, {"teacher": q}, Ks,
                               compute_hessian=hessian, return_coefficients=False, **RULES[rule])
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error),
                          uncompleted_work_count="unknown; not included in successful counts")
            emit()
            raise
        record.update(status="completed", end_compute_seconds=elapsed(),
            work_counts=serializable(value["work_counts"]),
            gradient=serializable(value["gradient"]), loss=serializable(value["loss"]))
        output["counts"][group+"_calls_completed"] += 1
        _sum_counts(output["counts"]["successful_target_work"], value["work_counts"])
        emit()
        # The completed value is always retained by the caller before its next
        # guard; an over-budget final call can therefore remain inspectable.
        return value

    seeds, blocked, base_q = {}, set(), None
    stopped = None
    for index, value in enumerate(lam):
        branch = 0 if value == 0 else (1 if value > 0 else -1)
        point = dict(index=index, lambda_exact=str(value), lambda_float=float(value),
                     branch=branch, status="pending", teacher={})
        output["points"].append(point)
        if stopped is not None:
            point.update(status="not_run", reason=stopped)
            continue
        if branch in blocked or branch != 0 and branch not in seeds:
            point.update(status="blocked_by_previous_point", reason="same-side qualified seed unavailable")
            continue
        try:
            guard()
            phi = [Fraction(-7+2*j, 10) + (value if j == 0 else 0) for j in range(8)]
            angles = np.asarray([float(x) for x in phi])
            W = .8 * np.column_stack((np.cos(angles), np.sin(angles)))
            point["teacher"].update(phi_exact=[str(x) for x in phi], rowlength_exact="4/5",
                                    W_float64=W.tolist(), bias=[0.]*8, grids=[])
            qs = []
            for order, angle in ((60, 0.), (80, 0.), (80, .271)):
                guard()
                output["counts"]["teacher_calls_started"] += 1
                grid = dict(order=order, rotation=angle, status="started")
                point["teacher"]["grids"].append(grid)
                emit()
                try:
                    q = np.asarray(teacher(np.zeros(8), W, order, angle), dtype=float)
                    if q.shape != (256,) or not np.isfinite(q).all() or np.any(q < 0):
                        raise ValueError("teacher kernel must return 256 finite nonnegative raw probabilities")
                except Exception as error:
                    grid.update(status="failed", error_type=type(error).__name__, error=str(error))
                    raise
                work = dict(gaussian_nodes=order**2, conditional_likelihood_values=256*order**2,
                            bernoulli_logit_evaluations=8*order**2, label_integrals=256)
                grid.update(status="completed", q=q.tolist(), q_sha256=_sha(q), work_counts=work,
                    probability_mass=float(q.sum()), complement_max=float(np.max(np.abs(q-q[::-1]))),
                    complement_l1=float(np.sum(np.abs(q-q[::-1]))))
                output["counts"]["teacher_calls_completed"] += 1
                _sum_counts(output["counts"]["successful_teacher_work"], work)
                qs.append(q)
                emit()
            q = qs[0]
            if base_q is None:
                base_q = q.copy()
            pair = spins.T @ (q[:, None]*spins)
            base_pair = spins.T @ (base_q[:, None]*spins)
            comparisons = dict(GH60_GH80_l1=float(np.sum(np.abs(qs[0]-qs[1]))),
                GH80_rotated_GH80_l1=float(np.sum(np.abs(qs[1]-qs[2]))),
                GH60_rotated_GH80_l1=float(np.sum(np.abs(qs[0]-qs[2]))))
            teacher_pass = max(comparisons.values()) <= selected_gates["teacher_l1_difference"]
            point["teacher"].update(order_checks=comparisons, qualified=teacher_pass,
                main_q_sha256=_sha(q), observed_marginals=(q @ Y).tolist(),
                observed_marginal_half_error=float(np.max(np.abs(q @ Y-.5))),
                q_l1_from_baseline=float(np.sum(np.abs(q-base_q))), spin_pair_matrix=pair.tolist(),
                spin_pair_change_from_baseline=(pair-base_pair).tolist(),
                q_reversal_l1=float(np.sum(np.abs(q-q[reverse_index]))),
                q_reversal_max=float(np.max(np.abs(q-q[reverse_index]))))
            if not teacher_pass:
                point["status"] = "teacher_qualification_failed"
                blocked.update((1, -1) if branch == 0 else (branch,))
                emit()
                continue
            a = a0.copy() if branch == 0 else seeds[branch][1].copy()
            point.update(seed_point_index=None if branch == 0 else seeds[branch][0], initial_a=a.tolist())
            evaluate = lambda a, r, k, h, p, i: call_kernel(point, q, a, r, k, h, p, i)
            a, ok = _newton(a, evaluate, "O", 10, 1e-8, point)
            if ok:
                a, ok = _newton(a, evaluate, "C", 6, 1e-11, point)
            point.update(final_a=a.tolist(), root_converged=ok)
            if not ok:
                point["status"] = "root_failed"
                blocked.update((1, -1) if branch == 0 else (branch,))
                emit()
                continue
            finals = {}
            point["evaluations"] = {}
            for rule in ("C", "D"):
                result = evaluate(a, rule, (8, "inf"), True, "final_evaluation", None)
                finals[rule] = result
                point["evaluations"][rule] = dict(result=serializable(result),
                    summary=_summary(result, q, a0, d))
                emit()
            point["C_D_qualification"] = _qualification(finals["C"], finals["D"], d, selected_gates)
            qualified = point["C_D_qualification"]["passed"]
            point["status"] = "qualified_development_point" if qualified else "C_D_qualification_failed"
            if qualified:
                for side in ((1, -1) if branch == 0 else (branch,)):
                    seeds[side] = index, a.copy()
            else:
                blocked.update((1, -1) if branch == 0 else (branch,))
            emit()
            guard()
        except _Stop as error:
            stopped = str(error)
            if point["status"] in ("pending", "running"):
                point["status"] = "partial_stopped"
            point["stop_reason"] = stopped
            emit()
        except Exception as error:
            point.update(status="failed_exception", error_type=type(error).__name__, error=str(error))
            blocked.update((1, -1) if branch == 0 else (branch,))
            emit()
    output["status"] = "partial_stopped" if stopped else "completed_bounded_panel"
    output["stop_reason"] = stopped
    output["all_points_qualified"] = all(p["status"] == "qualified_development_point" for p in output["points"])
    output["elapsed_compute_seconds"] = elapsed()
    output["wall_budget_overshoot_seconds"] = max(0., output["elapsed_compute_seconds"]-budget)
    emit()
    return serializable(output)


def manufactured_checks():
    """No real target/teacher calls: algorithm, bookkeeping and stop fixtures."""
    checks = {}
    point = {}
    calls = []

    def fake(a, rule, Ks, hessian, purpose, iteration):
        calls.append(purpose)
        gradient = np.r_[np.zeros(8), a-.5][None, None, :]
        return dict(gradient=gradient, H_ba=np.eye(16)[None, None, :, :])

    a, ok = _newton(np.full(8, .6), fake, "O", 10, 1e-8, point)
    checks["affine_root_one_newton_step"] = bool(ok and np.array_equal(a, np.full(8, .5)))
    checks["line_search_uses_gradient_only"] = calls == ["root_hessian", "root_line_search"]
    point = {}

    def no_decrease(a, rule, Ks, hessian, purpose, iteration):
        return dict(gradient=np.r_[np.zeros(8), np.full(8, .01)][None, None, :],
                    H_ba=np.eye(16)[None, None, :, :])

    _, ok = _newton(np.full(8, .6), no_decrease, "O", 10, 1e-8, point)
    checks["nine_trials_then_failure"] = not ok and len(point["root_phases"][0]["iterations"][0]["trials"]) == 9
    V = transverse_basis(np.arange(1, 9, dtype=float))
    checks["householder_basis"] = bool(np.max(np.abs(V.T @ V-np.eye(7))) < 1e-14 and
                                       np.max(np.abs(V.T @ np.arange(1, 9))) < 1e-14)
    calls_before = 0

    def forbidden(*args, **kwargs):
        nonlocal calls_before
        calls_before += 1
        raise AssertionError("a stopped fixture must not execute a kernel")

    result = compute(["1/2"]*8, dict(u_integer=[-1, 1, 0, 0, 0, 0, 0, 0], S_integer=2),
                     stop_requested=lambda: True, _stationary_kernel=forbidden, _teacher_kernel=forbidden)
    checks["stop_before_any_kernel"] = result["status"] == "partial_stopped" and calls_before == 0
    checks["seven_points_retained_on_stop"] = len(result["points"]) == 7
    json.dumps(result, allow_nan=False)
    checks["partial_is_strict_json"] = True

    def fake_teacher(b, W, order, angle):
        q = np.full(256, 1/256)
        delta = (math.atan2(W[0, 1], W[0, 0]) + .7)*.005
        q[[0, 255]] += delta
        q[[1, 254]] -= delta
        return q

    def full_fake(b, a, targets, Ks, *, compute_hessian, **kwargs):
        q = np.asarray(targets["teacher"])
        lam = (q[0]-1/256)/.005
        goal = .5 + lam*np.arange(1, 9)/80
        grad = np.tile(np.r_[np.zeros(8), a-goal], (1, len(Ks), 1))
        H = np.tile(np.eye(16), (1, len(Ks), 1, 1))
        P = np.tile(np.eye(8), (1, len(Ks), 1, 1))
        angle = np.tile(np.eye(8)-np.ones((8, 8))/8, (1, len(Ks), 1, 1))
        return dict(a=a, K_values=Ks, gradient=grad, H_ba=H, Hperp=P, Hangle=angle,
            loss=np.full((1, len(Ks)), 6.), ward_residual_max=np.zeros((1, len(Ks))),
            common_angle_residual=np.zeros((1, len(Ks))), log_evidence=np.full(256, -math.log(256)),
            work_counts=dict(manufactured_calls=1))

    full = compute(["1/2"]*8, dict(u_integer=[-1, 1, 0, 0, 0, 0, 0, 0], S_integer=2),
                   _stationary_kernel=full_fake, _teacher_kernel=fake_teacher)
    checks["complete_fake_panel"] = full["all_points_qualified"]
    checks["separate_same_side_seeds"] = [p["seed_point_index"] for p in full["points"]] == [None, 0, 0, 1, 2, 3, 4]
    checks["exact_successful_call_ledger"] = (
        full["counts"]["root_calls_completed"] + full["counts"]["final_calls_completed"] ==
        len(full["call_log"]) == full["counts"]["successful_target_work"]["manufactured_calls"])
    checks["teacher_all_three_grids_counted"] = (full["counts"]["teacher_calls_completed"] == 21 and
        full["counts"]["successful_teacher_work"]["gaussian_nodes"] == 7*(60**2+2*80**2))
    checks["raw_q_saved_unchanged"] = full["points"][1]["teacher"]["grids"][0]["q"] == fake_teacher(
        None, np.array(full["points"][1]["teacher"]["W_float64"]), 60, 0.).tolist()

    def failed_CD(*args, **kwargs):
        answer = full_fake(*args, **kwargs)
        if kwargs["normal_order"] == 384:
            answer["gradient"][0, 0, 0] += 2e-7
        return answer

    failed = compute(["1/2"]*8, dict(u_integer=[-1, 1, 0, 0, 0, 0, 0, 0], S_integer=2),
                     _stationary_kernel=failed_CD, _teacher_kernel=fake_teacher)
    checks["qualification_failure_stops_both_baseline_branches"] = (
        failed["points"][0]["status"] == "C_D_qualification_failed" and
        all(p["status"] == "blocked_by_previous_point" for p in failed["points"][1:]) and
        failed["counts"]["teacher_calls_completed"] == 3)
    json.dumps(full, allow_nan=False)
    checks["complete_is_strict_json"] = True
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(schema="r03_manufactured_implementation_checks_v1", checks=checks,
                all_passed=True, real_teacher_calls=0, real_target_calls=0)


if __name__ == "__main__":
    # There is deliberately no command-line route to scientific execution.
    print(json.dumps(manufactured_checks(), indent=2, allow_nan=False))
