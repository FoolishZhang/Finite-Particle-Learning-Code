"""Frozen one-dimensional K continuation, bounded refinement and 2x2 algebra.

The old stationary kernel and K-independent solver are imported unchanged.
--run is the only scientific entry point. --self-test uses manufactured data.
No noncollinear function evaluation, new observation or random prior is used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import rankone_stationary_compute as base
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/RankOneContinuation_v1_20261007"
OUT = ROOT / "artifacts/rankone_continuation_v1_20261007"
SOURCE = OUT / "SOURCE.json"
FREEZE = OUT / "FREEZE.json"
COARSE_K = (9, 10, 12, 16, 24, 32, 64, 128, "inf")
MAX_REFINEMENTS = 6
native, read, save, save_npz, sha, now = base.native, base.read, base.save, base.save_npz, base.sha, base.now
# Local aliases permit manufactured injection without modifying old module globals.
kernel_call = base.kernel_call


def config_dump():
    return native(dict(schema="rankone_continuation_config_v1", dimension=8,
        coarse_K=COARSE_K, initial_K=8, maximum_refinements=MAX_REFINEMENTS,
        midpoint="floor((lo+hi)/2)", midpoint_start="linear_interpolation_in_K_between_current_endpoint_theta",
        bracket="first adjacent finite positive-to-negative pair in complete coarse table including K8",
        unresolved_midpoint="retain_previous_pair_and_stop", stages=base.STAGES,
        rules=base.RULES, solver=base.SOLVER, gates=base.GATES,
        bias_box=[-2., 2.], amplitude_box=[.05, 2.], coordinate_order="b_then_a",
        target_weights_renormalized=False, new_random_draws=0,
        cross_direction="negative_endpoint_D_minimum_Helmert_angle_eigenvector; largest_absolute_component_positive; first_tie",
        cross_new_target_blocks=2, cross_new_kernel_calls=4, noncollinear_function_evaluations=0))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def artifact(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def k_token(K):
    return "Kinf" if K == "inf" else f"K{K:03d}"


def endpoint_Ks(K):
    return ("inf",) if K == "inf" else (K, "inf")


def total_counts(items):
    total = base.new_counts()
    for counts in items:
        for key, value in counts.items():
            if isinstance(value, int): total[key] += value
        for key, value in counts["high_level_factor_totals"].items():
            total["high_level_factor_totals"][key] = total["high_level_factor_totals"].get(key, 0) + value
    return total


def own_classification(theta, D_block, numerical_qualification):
    """Self-K radial eligibility is separate from positive/negative transverse type."""
    theta = np.asarray(theta, dtype=float); m = len(theta) // 2; a = theta[m:]
    g, H = np.asarray(D_block["gradient"]), np.asarray(D_block["Hperp"])
    gradient = float(np.max(abs(g)))
    margin = base.box_record(theta)["minimum_margin"]
    radial_minimum = float(D_block["H_ba_eigenvalues"][0])
    transverse_minimum = float(D_block["Hperp_projected_eigenvalues"][0])
    cart = float(np.max(abs(H @ a - g[m:])))
    rotation = float(np.max(abs(H @ (a / np.linalg.norm(a)))))
    gates = dict(numerical_qualification=bool(numerical_qualification["qualified"]),
        gradient=gradient <= base.GATES["fit_gradient"], box_margin=margin >= base.GATES["box_margin"],
        H_ba_positive=radial_minimum > base.GATES["positive_eigenvalue"],
        Cartesian_Ward=cart <= base.GATES["Cartesian_Ward"], near_rotation_mode=rotation <= base.GATES["rotation_residual"])
    eligible = all(gates.values())
    raw_sign = "positive" if transverse_minimum > base.GATES["positive_eigenvalue"] else (
        "negative" if transverse_minimum < -base.GATES["positive_eigenvalue"] else "unresolved")
    return native(dict(radial_qualification=dict(qualified=eligible, gates=gates,
            failed_gates=[key for key, passed in gates.items() if not passed]),
        transverse_label=raw_sign if eligible else "unresolved",
        raw_transverse_label_without_radial_qualification=raw_sign,
        D_gradient_Linf=gradient, minimum_box_margin=margin, D_H_ba_minimum=radial_minimum,
        D_transverse_minimum=transverse_minimum, D_Cartesian_Ward_Linf=cart,
        D_near_rotation_residual_Linf=rotation,
        scope="numerical own-K stationary signature, not interval existence or global optimality"))


def select_first_bracket(coarse):
    # Keep adjacency in the original table; never remove unqualified rows first.
    for left, right in zip(coarse, coarse[1:]):
        if left["K"] == "inf" or right["K"] == "inf": continue
        if left["radial_qualification"]["qualified"] and right["radial_qualification"]["qualified"] and left["transverse_label"] == "positive" and right["transverse_label"] == "negative":
            return left, right
    return None


def midpoint_plan(left, right, index):
    lo, hi = left["K"], right["K"]
    if not isinstance(lo, int) or not isinstance(hi, int) or hi - lo <= 1:
        raise ValueError("midpoint requires two nonadjacent finite integer endpoints")
    mid = (lo + hi) // 2; fraction = (mid - lo) / (hi - lo)
    theta = (1. - fraction) * np.asarray(left["theta"]) + fraction * np.asarray(right["theta"])
    return native(dict(index=index, point_id=f"refine_{index:02d}_" + k_token(mid), K=mid,
        lower_point_id=left["point_id"], upper_point_id=right["point_id"], lower_K=lo, upper_K=hi,
        lower_theta=np.asarray(left["theta"]).copy(), upper_theta=np.asarray(right["theta"]).copy(),
        interpolation_fraction=fraction, theta0=theta.copy(), interpolation_coordinate="K"))


def refinement_action(point):
    if not point["radial_qualification"]["qualified"]: return "stop_radial_or_numerical_unresolved"
    if point["transverse_label"] == "positive": return "replace_lower"
    if point["transverse_label"] == "negative": return "replace_upper"
    return "stop_transverse_unresolved"


def same_angle_direction(D_block, a):
    Q = base.helmert(len(a))
    values = np.asarray(D_block["Hangle_projected_eigenvalues"])
    vector = np.asarray(D_block["Hangle_projected_eigenvectors"])[:, 0]
    direction = Q @ vector
    index = int(np.argmax(abs(direction)))
    sign = 1. if direction[index] >= 0 else -1.
    direction = sign * direction
    return native(dict(direction=direction.copy(), helmert_eigenvector=(sign * vector).copy(),
        sign_coordinate=index, D_minimum_angle_eigenvalue=float(values[0]),
        D_angle_eigengap=float(values[1] - values[0]) if len(values) > 1 else None,
        direction_L2=float(np.linalg.norm(direction)), direction_sum=float(direction.sum()),
        numerical_eigenbasis_dependence="A repeated or near-repeated minimum can give another basis direction; no uniqueness is claimed."))


def cross_algebra(values):
    LL, LH, HL, HH = (float(values[key]) for key in ("LL", "LH", "HL", "HH"))
    # First index is theta endpoint, second index is K endpoint.
    total = HH - LL
    k_first = dict(K_change=LH - LL, theta_change=HH - LH)
    theta_first = dict(theta_change=HL - LL, K_change=HH - HL)
    return dict(total=total, K_first=k_first, theta_first=theta_first,
        K_first_sum=sum(k_first.values()), theta_first_sum=sum(theta_first.values()),
        K_first_roundoff_residual=sum(k_first.values()) - total,
        theta_first_roundoff_residual=sum(theta_first.values()) - total,
        interaction=HH - LH - HL + LL,
        descriptive_not_causal=True, unique_mechanism_attribution=False)


def verify_freeze():
    frozen, source = read(FREEZE), read(SOURCE)
    if frozen["config"] != config_dump() or frozen["source_sha256"] != sha(SOURCE):
        raise ValueError("continuation configuration/SOURCE differs from FREEZE")
    required = [Path(__file__), Path(__file__).with_name("rankone_continuation_compute_test.py"),
        Path(base.__file__), Path(base.kernel.__file__), DOC / "DESIGN.md", DOC / "INTERPRETATION.md", SOURCE, OUT / "SCOPE.json"]
    for path in required:
        if rel(path) not in frozen["sources"]: raise ValueError("required freeze source missing: " + rel(path))
    for mapping in (frozen["sources"], source["sources"]):
        for name, digest in mapping.items():
            if not isinstance(digest, str) or sha(ROOT / name) != digest: raise ValueError("changed frozen source: " + name)
    if source["schema"] != "rankone_continuation_source_v1": raise ValueError("invalid SOURCE schema")
    q = np.asarray(source["teacher_pmf"], dtype=float)
    theta = np.asarray(source["initial_theta"], dtype=float)
    if q.shape != (256,) or not np.isfinite(q).all() or np.any(q < 0) or abs(q.sum() - 1.) > 1e-12:
        raise ValueError("invalid unnormalized saved teacher weights")
    if theta.shape != (16,) or not base.box_record(theta)["feasible"]: raise ValueError("invalid old P starting theta")
    for name in ("C", "D", "qualification", "endpoint_lock"):
        path = source["initial_K8"][name]
        if source["sources"].get(path) != sha(ROOT / path): raise ValueError("old public anchor binding missing")
    old_lock = read(ROOT / source["initial_K8"]["endpoint_lock"])
    P = [e for e in old_lock["endpoints"] if e["name"] == "P"]
    if len(P) != 1 or not np.array_equal(theta, P[0]["theta"]) or not np.array_equal(q, P[0]["fit_target"]):
        raise ValueError("starting theta/target differs from locked old P")
    return frozen, copy.deepcopy(source)


def find_block(summary, K):
    found = [b for b in summary["blocks"] if b["target"] == "population" and b["K"] == K]
    if len(found) != 1: raise ValueError("expected one population block for requested K")
    return copy.deepcopy(found[0])


def initial_point(source):
    summaries = {rule: read(ROOT / source["initial_K8"][rule]) for rule in ("C", "D")}
    old_q = read(ROOT / source["initial_K8"]["qualification"])
    own_q = [q for q in old_q["qualifications"] if q["target"] == "population" and q["K"] == 8]
    if len(own_q) != 1: raise ValueError("missing old P K8 qualification")
    for summary in summaries.values():
        if not np.array_equal(summary["theta"], source["initial_theta"]): raise ValueError("old P public parameter mismatch")
    blocks = {rule: find_block(summary, 8) for rule, summary in summaries.items()}
    return native(dict(point_id="initial_K8", kind="reused_initial", K=8, theta=source["initial_theta"],
        own_blocks=blocks, own_qualification=own_q[0],
        **own_classification(source["initial_theta"], blocks["D"], own_q[0]),
        probability={rule: summary["probability"] for rule, summary in summaries.items()},
        old_public_artifacts={key: artifact(ROOT / path) for key, path in source["initial_K8"].items() if key in ("C", "D", "qualification", "endpoint_lock")},
        old_arrays_read=False, new_kernel_calls=0))


def evaluate_pair(theta, K_values, q, folder, stem, counts):
    results, summaries, files = {}, {}, {}
    for rule in ("C", "D"):
        result, seconds = kernel_call(np.asarray(theta).copy(), {"population": np.asarray(q).copy()},
                                          K_values, rule, True, counts)
        base.project_result(result)
        npz = folder / (stem + "_" + rule + ".npz")
        save_npz(npz, base.kernel_arrays(result, theta))
        summary = base.evaluation_summary(theta, result, q, seconds)
        js = folder / (stem + "_" + rule + ".json")
        save(js, dict(rule=rule, **summary, array_artifact=artifact(npz)))
        files[rule] = dict(array_artifact=artifact(npz), summary_artifact=artifact(js))
        results[rule], summaries[rule] = result, summary
    qualifications = [dict(target="population", K=K, **base.qualification(results["C"], results["D"], 0, ki)) for ki, K in enumerate(K_values)]
    return summaries, dict(rule_artifacts=files, qualifications=qualifications)


def solve_point(point_id, K, theta0, q, kind, start_context):
    folder = OUT / "points" / point_id; folder.mkdir(parents=True, exist_ok=False)
    opt = folder / "optimization"; opt.mkdir()
    started = time.perf_counter(); trace = {}; counts = base.new_counts(); evaluation_counts = base.new_counts(); artifacts = []
    try:
        verify_freeze()
        save(folder / "START_LOCK.json", dict(created_utc=now(), point_id=point_id, K=K, kind=kind,
            theta0=np.asarray(theta0).copy(), target=np.asarray(q).copy(), start_context=copy.deepcopy(start_context),
            freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE), locked_before_optimization=True))

        def evaluator(theta, phase, hessian, context):
            result, seconds = kernel_call(theta.copy(), {"population": np.asarray(q).copy()}, (K,), phase, hessian, counts)
            stem = f"call_{context['call_id']:04d}"
            npz = opt / (stem + ".npz"); save_npz(npz, base.kernel_arrays(result, theta))
            js = opt / (stem + ".json")
            save(js, dict(**context, K=K, theta=theta.copy(), loss=float(result["loss"][0, 0]),
                gradient=result["gradient"][0, 0].copy(), H_ba=result["H_ba"][0, 0].copy() if hessian else None,
                compute_hessian=hessian, seconds=seconds, config=result["config"], work_counts=result["work_counts"],
                array_artifact=artifact(npz)))
            artifacts.append(dict(call_id=context["call_id"], array_artifact=artifact(npz), summary_artifact=artifact(js)))
            return result

        base.stationary_solve(np.asarray(theta0).copy(), evaluator, trace=trace)
        counts["accepted_optimizer_steps"] = trace["accepted_step_count"]
        tape = opt / "TAPE.npz"; save_npz(tape, base.tape_arrays(trace))
        solver_path = opt / "SOLVER.json"
        save(solver_path, dict(point_id=point_id, K=K, fit_target_name="population", fit_target=np.asarray(q).copy(),
            **trace, resources=counts, call_artifacts=artifacts, tape_artifact=artifact(tape),
            hessian_missing_fill_in_tape="NaN with compute_hessian=false", freeze_sha256=sha(FREEZE)))
        verify_freeze()
        theta = trace["final_theta"].copy()
        if not base.box_record(theta)["feasible"]: raise FloatingPointError("LAST is not finite and feasible")
        lock = folder / "PARAMETERS_LOCK.json"
        save(lock, dict(created_utc=now(), point_id=point_id, K=K, theta=theta.copy(),
            start_lock_artifact=artifact(folder / "START_LOCK.json"), solver_artifact=artifact(solver_path),
            tape_artifact=artifact(tape), optimization_artifacts={rel(p): sha(p) for p in opt.iterdir() if p.is_file()},
            locked_before_own_K_and_integrated_evaluation=True, selected_state="finite_feasible_LAST",
            parameter_selection_by_risk_or_curvature=False))
        summaries, evaluation = evaluate_pair(theta, endpoint_Ks(K), q, folder, "evaluation", evaluation_counts)
        verify_freeze()
        blocks = {rule: find_block(summary, K) for rule, summary in summaries.items()}
        own_q = evaluation["qualifications"][0]
        point = native(dict(status="completed", point_id=point_id, kind=kind, K=K, theta=theta.copy(),
            theta0=np.asarray(theta0).copy(), start_context=start_context, parameter_lock_artifact=artifact(lock),
            solver_artifact=artifact(solver_path), own_blocks=blocks, own_qualification=own_q,
            **own_classification(theta, blocks["D"], own_q), evaluation=evaluation,
            probability={rule: summary["probability"] for rule, summary in summaries.items()},
            stage_stop_reasons=[stage["stop_reason"] for stage in trace["stages"]],
            optimization_resources=counts, evaluation_resources=evaluation_counts,
            resources=total_counts([counts, evaluation_counts]), seconds=time.perf_counter() - started,
            integrated_control_not_claimed_stationary=K != "inf"))
        save(folder / "RESULTS.json", point)
        point["result_artifact"] = artifact(folder / "RESULTS.json")
        print(json.dumps(dict(point_id=point_id, K=K, radial_qualified=point["radial_qualification"]["qualified"],
            transverse_label=point["transverse_label"], solver_calls=trace["total_calls"])), flush=True)
        return point
    except BaseException as error:
        counts["accepted_optimizer_steps"] = max(0, len(trace.get("accepted_states", [])) - 1)
        save(folder / "FAILURE.json", dict(point_id=point_id, K=K, kind=kind, reason=str(error),
            traceback=traceback.format_exc(), partial_trace=trace, call_artifacts=artifacts,
            optimization_resources=counts, evaluation_resources=evaluation_counts,
            seconds=time.perf_counter() - started, failed_kernel_internal_partial_work_not_counted=True,
            automatic_retry=False, dependent_chain_must_stop=True))
        raise


def coarse_chain(initial, solve):
    """No sign/risk-based branch can shorten the prescribed coarse sequence."""
    rows = [copy.deepcopy(initial)]
    for index, K in enumerate(COARSE_K):
        previous = rows[-1]
        context = dict(coarse_index=index, predecessor_point_id=previous["point_id"], predecessor_K=previous["K"],
            predecessor_theta=copy.deepcopy(previous["theta"]), start_rule="preceding_finite_feasible_LAST",
            predecessor_result_artifact=copy.deepcopy(previous.get("result_artifact")))
        point = solve("coarse_" + k_token(K), K, np.asarray(previous["theta"]).copy(), "coarse", context)
        rows.append(copy.deepcopy(point))
    return rows


def refine(coarse, solve, plan_writer=lambda name, value: None, decision_writer=lambda index, value: None):
    pair = select_first_bracket(coarse)
    initial_plan = dict(maximum_refinements=MAX_REFINEMENTS,
        coarse_order=[dict(point_id=p["point_id"], K=p["K"], radial_qualified=p["radial_qualification"]["qualified"],
                           transverse_label=p["transverse_label"]) for p in coarse],
        selected_first_adjacent_pair=None if pair is None else [p["point_id"] for p in pair],
        selected_K_pair=None if pair is None else [p["K"] for p in pair])
    plan_writer("initial", copy.deepcopy(initial_plan))
    if pair is None:
        return dict(status="no_usable_finite_bracket", plan=initial_plan, points=[], decisions=[], final_pair=None)
    left, right = copy.deepcopy(pair[0]), copy.deepcopy(pair[1])
    points, decisions = [], []
    status = "maximum_refinements_reached"
    for index in range(MAX_REFINEMENTS):
        if right["K"] - left["K"] <= 1:
            status = "adjacent_integers"; break
        plan = midpoint_plan(left, right, index)
        plan_writer(f"midpoint_{index:02d}", copy.deepcopy(plan))
        middle = solve(plan["point_id"], plan["K"], np.asarray(plan["theta0"]).copy(), "refinement", copy.deepcopy(plan))
        points.append(copy.deepcopy(middle)); action = refinement_action(middle)
        before = dict(lower_point_id=left["point_id"], upper_point_id=right["point_id"], lower_K=left["K"], upper_K=right["K"])
        if action == "replace_lower": left = copy.deepcopy(middle)
        elif action == "replace_upper": right = copy.deepcopy(middle)
        decision = dict(index=index, plan=plan, midpoint_point_id=middle["point_id"], midpoint_K=middle["K"],
            midpoint_result_artifact=copy.deepcopy(middle.get("result_artifact")),
            midpoint_radial_qualification=copy.deepcopy(middle["radial_qualification"]),
            midpoint_transverse_label=middle["transverse_label"], action=action, before=before,
            after=dict(lower_point_id=left["point_id"], upper_point_id=right["point_id"], lower_K=left["K"], upper_K=right["K"]))
        decisions.append(decision); decision_writer(index, copy.deepcopy(decision))
        if action.startswith("stop_"):
            status = action; break
        if right["K"] - left["K"] <= 1:
            status = "adjacent_integers"; break
    return dict(status=status, plan=initial_plan, points=points, decisions=decisions,
        final_pair=dict(lower=left, upper=right), iterations=len(points),
        monotonic_or_unique_transition_claimed=False, continuous_stationary_branch_certified=False)


def final_cross(pair, q, endpoint_lock_sha256):
    if pair is None:
        return dict(status="skipped_no_usable_finite_bracket", resources=base.new_counts(), new_target_blocks=0)
    left, right = pair["lower"], pair["upper"]
    folder = OUT / "cross"; folder.mkdir(exist_ok=False)
    counts = base.new_counts(); tick = time.perf_counter()
    a = np.asarray(right["theta"])[len(right["theta"]) // 2:]
    direction = same_angle_direction(right["own_blocks"]["D"], a)
    plan = dict(created_utc=now(), lower_point_id=left["point_id"], upper_point_id=right["point_id"],
        lower_K=left["K"], upper_K=right["K"], theta_L=copy.deepcopy(left["theta"]), theta_H=copy.deepcopy(right["theta"]),
        **direction, endpoint_lock_sha256=endpoint_lock_sha256,
        cell_index_convention="first index theta endpoint, second index K endpoint",
        cells=dict(LL=dict(theta_source="lower", K=left["K"], reused_own_block=True),
                   LH=dict(theta_source="lower", K=right["K"], reused_own_block=False),
                   HL=dict(theta_source="upper", K=left["K"], reused_own_block=False),
                   HH=dict(theta_source="upper", K=right["K"], reused_own_block=True)),
        direction_locked_before_cross_evaluation=True, noncollinear_function_evaluations=0)
    save(folder / "PLAN.json", plan)
    try:
        LH, LH_q = evaluate_pair(np.asarray(left["theta"]).copy(), (right["K"],), q, folder, "LH", counts)
        HL, HL_q = evaluate_pair(np.asarray(right["theta"]).copy(), (left["K"],), q, folder, "HL", counts)
        own = {"LL": left["own_blocks"], "HH": right["own_blocks"],
               "LH": {rule: find_block(s, right["K"]) for rule, s in LH.items()},
               "HL": {rule: find_block(s, left["K"]) for rule, s in HL.items()}}
        quals = {"LL": left["own_qualification"], "HH": right["own_qualification"],
                 "LH": LH_q["qualifications"][0], "HL": HL_q["qualifications"][0]}
        psi = np.asarray(direction["direction"])
        cells = {}
        for name, blocks in own.items():
            values = {rule: float(psi @ np.asarray(block["Hangle"]) @ psi) for rule, block in blocks.items()}
            cells[name] = dict(**plan["cells"][name], C_curvature=values["C"], D_curvature=values["D"],
                curvature_rule_difference=abs(values["C"] - values["D"]), numerical_qualification=copy.deepcopy(quals[name]),
                Hangle={rule: copy.deepcopy(block["Hangle"]) for rule, block in blocks.items()},
                complete_stationarity_at_cross_cell_claimed=False if name in ("LH", "HL") else None)
        result = native(dict(status="completed", plan_artifact=artifact(folder / "PLAN.json"),
            lower_point_id=left["point_id"], upper_point_id=right["point_id"], lower_K=left["K"], upper_K=right["K"],
            direction=direction, cells=cells, cross_evaluations={"LH": LH_q, "HL": HL_q},
            algebra={rule: cross_algebra({name: cell[rule + "_curvature"] for name, cell in cells.items()}) for rule in ("C", "D")},
            all_four_cells_numerically_qualified=all(qr["qualified"] for qr in quals.values()),
            resources=counts, new_target_blocks=2, new_rule_blocks=4, new_optimizer_steps=0,
            noncollinear_function_evaluations=0, seconds=time.perf_counter() - tick,
            descriptive_not_causal=True, unique_mechanism_attribution=False))
        save(folder / "RESULTS.json", result)
        result["result_artifact"] = artifact(folder / "RESULTS.json")
        return result
    except BaseException as error:
        save(folder / "FAILURE.json", dict(reason=str(error), traceback=traceback.format_exc(), resources=counts,
            plan_artifact=artifact(folder / "PLAN.json"), seconds=time.perf_counter() - tick,
            failed_kernel_internal_partial_work_not_counted=True, automatic_retry=False))
        raise


def run():
    if any((OUT / name).exists() for name in ("RUN_START.json", "RESULTS.json", "points", "ENDPOINTS_LOCK.json")):
        raise FileExistsError("continuation outputs already exist and will not be replaced")
    try:
        _, source = verify_freeze()
    except BaseException as error:
        rejected = OUT / "START_REJECTED.json"
        if not rejected.exists():
            save(rejected, dict(created_utc=now(), status="rejected_before_scientific_evaluation", reason=str(error),
                traceback=traceback.format_exc(), new_kernel_calls=0, automatic_retry=False))
        raise
    tick = time.perf_counter(); points = []; current = None
    save(OUT / "RUN_START.json", dict(created_utc=now(), freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE), environment=base.environment()))
    try:
        initial = initial_point(source)
        save(OUT / "INITIAL_K8.json", initial)
        def solve(point_id, K, theta0, kind, context):
            nonlocal current
            current = dict(point_id=point_id, K=K, kind=kind)
            point = solve_point(point_id, K, theta0.copy(), np.asarray(source["teacher_pmf"]).copy(), kind, copy.deepcopy(context))
            points.append(copy.deepcopy(point))
            return point
        coarse = coarse_chain(initial, solve)
        save(OUT / "COARSE_LOCK.json", dict(created_utc=now(), rows=coarse, full_sequence_completed=True,
            coarse_solve_count=len(COARSE_K), curve_or_risk_based_early_stop=False,
            result_artifacts=[p["result_artifact"] for p in coarse[1:]]))
        refinement_folder = OUT / "refinement"; refinement_folder.mkdir()
        def plan_writer(name, value):
            path = OUT / "REFINEMENT_PLAN.json" if name == "initial" else refinement_folder / (name + "_PLAN.json")
            save(path, value)
        def decision_writer(index, value): save(refinement_folder / f"decision_{index:02d}.json", value)
        refinement = refine(coarse, solve, plan_writer, decision_writer)
        save(OUT / "BRACKET_FINAL.json", refinement)
        verify_freeze()
        lock_path = OUT / "ENDPOINTS_LOCK.json"
        save(lock_path, dict(created_utc=now(), freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE),
            new_endpoint_count=len(points), endpoints=[dict(point_id=p["point_id"], K=p["K"], theta=p["theta"],
                parameter_lock_artifact=p["parameter_lock_artifact"], result_artifact=p["result_artifact"]) for p in points],
            old_K8_artifact=artifact(OUT / "INITIAL_K8.json"), coarse_lock_artifact=artifact(OUT / "COARSE_LOCK.json"),
            bracket_final_artifact=artifact(OUT / "BRACKET_FINAL.json"), all_endpoints_locked_before_cross=True))
        cross = final_cross(refinement["final_pair"], np.asarray(source["teacher_pmf"]).copy(), sha(lock_path))
        verify_freeze()
        total = total_counts([p["resources"] for p in points] + [cross["resources"]])
        save(OUT / "RESULTS.json", dict(status="completed", created_utc=now(), freeze_sha256=sha(FREEZE), source_sha256=sha(SOURCE),
            initial=initial, coarse=coarse, refinement=refinement, points=points, cross=cross,
            coarse_solve_count=len(COARSE_K), refinement_solve_count=len(refinement["points"]), new_endpoint_count=len(points),
            endpoint_lock_artifact=artifact(lock_path), resources=total, seconds=time.perf_counter() - tick,
            artifacts={rel(p): sha(p) for p in OUT.rglob("*") if p.is_file() and p != OUT / "RESULTS.json"},
            claims=dict(negative_curvature_is_not_execution_failure=True, numerical_signatures_only=True,
                exact_stationary_point_existence_certified=False, unique_or_monotone_branch_claimed=False,
                causal_training_decomposition=False, independent_training_repeats=0,
                new_observations=0, new_random_prior_vectors=0, statistical_alpha_spent=0,
                noncollinear_function_evaluations=0, manuscript_changed=False)))
        print(json.dumps(dict(completed=True, coarse_solve_count=9, refinement_solve_count=len(refinement["points"]),
            bracket_status=refinement["status"], cross_status=cross["status"], resources=total)), flush=True)
    except BaseException as error:
        save(OUT / "FAILURE.json", dict(current=current, reason=str(error), traceback=traceback.format_exc(),
            completed_points=points, completed_points_resources=total_counts([p["resources"] for p in points]),
            seconds=time.perf_counter() - tick, dependent_chain_aborted=True, automatic_retry=False))
        raise


def self_test():
    import rankone_continuation_compute_test as tests
    return tests.run_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.config: print(json.dumps(config_dump(), sort_keys=True), flush=True)
    elif args.self_test:
        outcome = self_test(); print(json.dumps(outcome, sort_keys=True), flush=True)
        raise SystemExit(0 if outcome["passed"] else 1)
    else: run()
