"""Data-only, no-I/O building blocks for a prospective diagnostic development.

No function generates random numbers, reads a teacher, selects a checkpoint by
risk, or writes a file. Old frozen solver/kernel functions are imported without
changing their source or configuration. Their import fixes BLAS thread settings
as in the prior implementation. Numerical qualifications are not data confidence
bounds, stationary existence certificates, or global optimality guarantees.

Production defaults are fixed. Explicit manufactured_config overrides are only
accepted for m=2 or 3 fixtures and marked in outputs; production callers must not
pass them. Every stored parameter/array is a snapshot, never a live view.
"""
from __future__ import annotations

import copy
import math
import operator
import time

import rankone_stationary_compute as stationary
import rankone_stationary_kernel as rankone
import joint_dependence_kernel as joint
import numpy as np
from scipy.optimize import minimize


SIGN_VARIANCE_FLOOR = 1e-8
SIGN_GAP_FLOOR = 1e-8
SIGN_COMPONENT_FLOOR = 1e-6
CURVATURE_FLOOR = 5e-4
DISAGREEMENT_MULTIPLIER = 10.


def snapshot(value):
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def _real(value, name):
    if np.iscomplexobj(value): raise ValueError(f"{name} must be real")
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all(): raise ValueError(f"{name} must be finite")
    return out


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)): raise ValueError(f"{name} must be an integer")
    try: value = operator.index(value)
    except TypeError as error: raise ValueError(f"{name} must be an integer") from error
    if value < 1: raise ValueError(f"{name} must be positive")
    return value


def _target(q):
    q = _real(q, "frequency")
    if q.ndim != 1 or len(q) < 4 or len(q) & (len(q) - 1) or np.any(q < 0):
        raise ValueError("frequency must be nonnegative, length 2**m with m>=2")
    if not math.isclose(math.fsum(float(v) for v in q), 1., rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("original frequency must sum to one; no renormalization")
    return q.copy(), (len(q).bit_length() - 1)


def _configuration(m, manufactured_config):
    config = dict(rules=snapshot(stationary.RULES), stages=snapshot(stationary.STAGES),
                  arm_updates=600, arm_switch_step=400, checkpoint_every=100,
                  reference_order=32,
                  reference_options=dict(maxiter=120, maxfun=180, maxls=25, gtol=1e-7, ftol=1e-12),
                  manufactured_only=False)
    if manufactured_config is not None:
        if m not in (2, 3): raise ValueError("manufactured overrides are restricted to m2/m3")
        allowed = {"rules", "stages", "arm_updates", "arm_switch_step", "checkpoint_every", "reference_order", "reference_options"}
        if set(manufactured_config) - allowed: raise ValueError("unknown manufactured override")
        config.update(snapshot(manufactured_config)); config["manufactured_only"] = True
    for key in ("arm_updates", "arm_switch_step", "checkpoint_every", "reference_order"):
        config[key] = _positive_integer(config[key], key)
    if set(config["rules"]) != {"O", "C", "D"}: raise ValueError("O/C/D rules required")
    if [x["name"] for x in config["stages"]] != ["O", "C"]: raise ValueError("O then C solver stages required")
    return config


def configuration():
    """Public production settings; this does not run any objective or fit."""
    return snapshot(dict(schema="prospective_diagnostic_core_v1", **_configuration(8, None),
                         low_K=8, high_K=64, sign_variance_floor=SIGN_VARIANCE_FLOOR,
                         sign_gap_floor=SIGN_GAP_FLOOR, sign_component_floor=SIGN_COMPONENT_FLOOR,
                         curvature_floor=CURVATURE_FLOOR, disagreement_multiplier=DISAGREEMENT_MULTIPLIER,
                         direction_rule="D", reference_objective_hard_call_cap=180,
                         neutral_kick_amplitude=.15, no_RNG_or_application_file_IO=True))


def recode_frequency(q, signs):
    """Bijective bit complementation, preserving original masses exactly.

    Returns q_recoded and old_to_new. A second application is the inverse.
    """
    q, m = _target(q); signs = np.asarray(signs)
    if signs.shape != (m,) or not np.all((signs == -1) | (signs == 1)):
        raise ValueError("signs must be an m-vector in {-1,+1}")
    labels = joint.patterns(m)
    recoded = np.where(signs[None, :] > 0, labels, 1 - labels)
    powers = 2 ** np.arange(m - 1, -1, -1, dtype=np.int64)
    index = recoded.astype(np.int64) @ powers
    out = np.empty_like(q); out[index] = q
    return out.copy(), index.copy()


def sign_initialization(q, N):
    q, m = _target(q); N = _positive_integer(N, "N")
    labels = joint.patterns(m).astype(float)
    p = q @ labels
    centered = labels - p
    covariance = centered.T @ (q[:, None] * centered)
    variance = p * (1. - p)
    scaled = np.maximum(variance, SIGN_VARIANCE_FLOOR)
    correlation = covariance / np.sqrt(scaled[:, None] * scaled[None, :])
    values, vectors = np.linalg.eigh(.5 * (correlation + correlation.T))
    direction = vectors[:, -1].copy()
    canonical_index = int(np.argmax(abs(direction)))
    if direction[canonical_index] < 0: direction *= -1.
    gap = float(values[-1] - values[-2])
    reasons = []
    if not gap > SIGN_GAP_FLOOR: reasons.append("leading_eigenvalue_gap_unresolved")
    if not np.all(abs(direction) > SIGN_COMPONENT_FLOOR): reasons.append("leading_eigenvector_component_unresolved")
    resolved = not reasons
    signs = np.where(direction >= 0., 1, -1).astype(np.int8) if resolved else np.ones(m, dtype=np.int8)
    qr, permutation = recode_frequency(q, signs)
    smooth = (p * N + .5) / (N + 1.)
    b = np.clip(np.log(smooth) - np.log1p(-smooth), -2., 2.)
    initial = np.r_[signs * b, np.full(m, .5)]
    return snapshot(dict(signs=signs, sign_resolved=resolved, reasons=reasons,
                         fallback_all_positive=not resolved, marginals=p, covariance=covariance,
                         bernoulli_variances=variance, floored_variances=scaled, correlation=correlation,
                         eigenvalues=values, canonical_eigenvector=direction, canonical_index=canonical_index,
                         eigenvalue_gap=gap, variance_floor=SIGN_VARIANCE_FLOOR,
                         covariance_definition="frequency centered second moment; no forced unit correlation diagonal",
                         q_original=q, q_recoded=qr, old_to_new_pattern_index=permutation,
                         target_mass=float(q.sum()), recoded_mass=float(qr.sum()),
                         smoothed_initial_marginals=smooth, initial_bias_original=b,
                         theta0_recoded=initial, theta0_signed=np.r_[b, signs * .5],
                         target_smoothed=False, target_renormalized=False, N=N))


def _new_cost():
    return dict(wall_seconds=0., kernel_calls=0, objective_calls=0, successful_kernel_calls=0,
                failed_kernel_calls=0, full_hessian_calls=0, loss_gradient_calls=0,
                optimizer_updates=0, processed_prior_vectors=0, label_node_products=0,
                high_level_factor_totals={}, generated_random_vectors=0)


def _count_raw(cost, raw, hessian):
    cost["successful_kernel_calls"] += 1
    cost["full_hessian_calls" if hessian else "loss_gradient_calls"] += 1
    for key, value in raw["work_counts"].items():
        if isinstance(value, int):
            cost["high_level_factor_totals"][key] = cost["high_level_factor_totals"].get(key, 0) + value
    cost["label_node_products"] += raw["work_counts"]["conditional_loglikelihood_values"]


def _rankone_call(theta, q, Ks, rule, config, cost, calls, context, hessian=True):
    m = len(theta) // 2
    tick = time.perf_counter()
    record = dict(context=snapshot(context), theta_recoded=np.asarray(theta).copy(), rule=rule,
                  K_values=tuple(Ks), compute_hessian=bool(hessian), status="started")
    calls.append(record); cost["kernel_calls"] += 1; cost["objective_calls"] += 1
    try:
        raw = rankone.rankone_stationary(theta[:m], theta[m:], q, K_values=Ks,
                                        **config["rules"][rule], compute_hessian=hessian)
        _count_raw(cost, raw, hessian)
        record.update(status="completed", raw=snapshot(raw))
        return raw
    except FloatingPointError as error:
        cost["failed_kernel_calls"] += 1
        record.update(status="nonfinite_failure", failure=str(error), partial_internal_work_unmeasured=True)
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - tick


def fit_anchor(q, N, *, manufactured_config=None):
    """Fit positive amplitudes after an exact, training-only label recoding."""
    tick = time.perf_counter(); q, m = _target(q)
    cfg = _configuration(m, manufactured_config)
    initialization = sign_initialization(q, N)
    qr, signs = initialization["q_recoded"], initialization["signs"]
    trace = {}; calls = []; cost = _new_cost(); failure = None
    def evaluator(theta, phase, hessian, context):
        return _rankone_call(theta, qr, (8,), phase, cfg, cost, calls, context, hessian)
    try:
        stationary.stationary_solve(initialization["theta0_recoded"], evaluator,
                                    stages=cfg["stages"], trace=trace)
        theta = trace["final_theta"].copy()
        stops = [row["stop_reason"] for row in trace["stages"]]
        if "line_search_failed" in stops: status = "completed_with_line_search_stop"
        elif "no_positive_feasible_box_step" in stops: status = "completed_with_boundary_stop"
        elif any(stop in ("accepted_step_budget", "function_call_budget") for stop in stops): status = "completed_with_budget_stop"
        else: status = "completed_gradient_stops"
        fallback = False
    except FloatingPointError as error:
        failure = dict(kind="nonfinite_anchor_or_solver", message=str(error))
        feasible = [x["theta"] for x in trace.get("accepted_states", [])
                    if stationary.box_record(x["theta"])["feasible"]
                    and any(call["status"] == "completed" and np.array_equal(call["theta_recoded"], x["theta"]) for call in calls)]
        theta = np.asarray(feasible[-1] if feasible else initialization["theta0_recoded"]).copy()
        fallback = not bool(feasible)
        status = "nonfinite_stopped_last_finite"
    if not stationary.box_record(theta)["feasible"]:
        theta = initialization["theta0_recoded"].copy(); fallback = True
        status = "fallback_initialization"
    original_b, amplitudes = signs * theta[:m], theta[m:].copy()
    W = np.column_stack((signs * amplitudes, np.zeros(m)))
    cost["optimizer_updates"] = max(0, len(trace.get("accepted_states", [])) - 1)
    cost["wall_seconds"] = time.perf_counter() - tick
    return snapshot(dict(schema="prospective_anchor_v1", status=status, b=original_b, W=W,
                         signs=signs, sign_resolved=initialization["sign_resolved"],
                         q_original=q, q_recoded=qr, N=N, theta_recoded=theta,
                         theta_signed=np.r_[original_b, signs * amplitudes], amplitudes=amplitudes,
                         initialization=initialization, trace=trace, calls=calls, cost=cost, failure=failure,
                         finite_feasible=stationary.box_record(theta)["feasible"], used_initialization_fallback=fallback,
                         fallback_reason="no_finite_evaluated_accepted_state" if fallback else None,
                         solver_execution_completed=trace.get("status") == "completed" and failure is None,
                         solver_stop_reasons=[row["stop_reason"] for row in trace.get("stages", [])],
                         box=stationary.box_record(theta), config=cfg, global_signed_optimum_claimed=False,
                         exact_stationary_existence_claimed=False, teacher_access=False))


def _sign_pair(values, threshold):
    if threshold is None: return "unavailable"
    values = np.asarray(values, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all(): return "unavailable"
    if np.all(values > threshold): return "positive"
    if np.all(values < -threshold): return "negative"
    return "unresolved"


def classify_diagnostic(*, low_min, reference_min, high_curvature, low_disagreement,
                        reference_disagreement, high_disagreement, reference_spectrum_disagreement,
                        reference_gaps, full_prerequisites, reference_prerequisites):
    """Pure ordered A/D0-D3 and B rule, with no integration or teacher access."""
    def threshold(disagreement):
        if disagreement is None: return None
        if not math.isfinite(disagreement) or disagreement < 0: raise ValueError("numerical disagreement must be finite and nonnegative")
        return max(CURVATURE_FLOOR, DISAGREEMENT_MULTIPLIER * disagreement)
    thresholds = dict(low=threshold(low_disagreement), reference=threshold(reference_disagreement), high=threshold(high_disagreement))
    signs = dict(low=_sign_pair(low_min, thresholds["low"]), reference=_sign_pair(reference_min, thresholds["reference"]),
                 high=_sign_pair(high_curvature, thresholds["high"]))
    ref_failed = [key for key, value in reference_prerequisites.items() if not value]
    ref_raise = not ref_failed and signs["reference"] == "negative"
    ref_reasons = ref_failed.copy()
    if not ref_failed and not ref_raise: ref_reasons.append("reference_" + signs["reference"])
    all_prerequisites = {"reference." + k: v for k, v in reference_prerequisites.items()}
    all_prerequisites.update(full_prerequisites)
    failed = [key for key, value in all_prerequisites.items() if not value]
    computation = [x for x in failed if not x.endswith("sign_resolved")]
    resolution = [x for x in failed if x.endswith("sign_resolved")]
    gap_threshold = None if reference_spectrum_disagreement is None else DISAGREEMENT_MULTIPLIER * reference_spectrum_disagreement
    gap_resolved = reference_gaps is not None and gap_threshold is not None and all(gap is None or gap > gap_threshold for gap in reference_gaps)
    level, reasons, suppression = "D0", failed.copy(), False
    if not failed:
        if signs["low"] == "negative": level, reasons = "D1", ["low_already_negative"]
        elif signs["low"] != "positive": reasons = ["low_resolution_band"]; resolution.extend(reasons)
        elif signs["reference"] == "positive": level, reasons = "D1", ["reference_positive"]
        elif signs["reference"] != "negative": reasons = ["reference_resolution_band"]; resolution.extend(reasons)
        else:
            suppression = True
            if not gap_resolved: reasons = ["reference_direction_gap_unresolved"]; resolution.extend(reasons)
            elif signs["high"] == "negative": level, reasons = "D3", ["suppression_and_release"]
            elif signs["high"] == "positive": level, reasons = "D2", ["suppression_high_direction_positive"]
            else: reasons = ["suppression_confirmed_release_unresolved"]; resolution.extend(reasons)
    full_raise = level == "D3"
    if full_raise and not ref_raise: raise AssertionError("full rule must imply reference-only rule")
    return dict(full_level=level, full_raise=full_raise, ref_raise=bool(ref_raise), full_reasons=reasons,
                ref_reasons=ref_reasons, reasons=dict(full=reasons, reference_only=ref_reasons),
                D0_reason_categories=dict(computation_or_solver=computation, resolution=resolution),
                full_prerequisites=all_prerequisites, reference_prerequisites=reference_prerequisites,
                thresholds=thresholds, signs=signs, reference_gap_threshold=gap_threshold,
                reference_direction_resolved=gap_resolved, suppression_confirmed=suppression,
                full_implies_reference_only=True, statistical_confidence_claimed=False)


def _reference_qualification(C, D):
    """Only the reference angle matrix, its spectrum, masses and Ward check."""
    comp = dict(Hangle=float(np.max(abs(C["Hangle"] - D["Hangle"]))),
                spectrum=float(np.max(abs(C["Hangle_projected_eigenvalues"] - D["Hangle_projected_eigenvalues"]))))
    gates = dict(Hangle=comp["Hangle"] <= stationary.GATES["matrix"], spectrum=comp["spectrum"] <= stationary.GATES["eigenvalue"])
    scalars = {}
    for rule, raw in (("C", C), ("D", D)):
        s = stationary.scalar_diagnostics(raw, 0); scalars[rule] = s
        for field, bound in (("normal_mass_error", "normal_mass"), ("d_relative_mass_error", "d_relative_mass"),
                             ("c_raw_lower_violation", "c_raw_mass"), ("c_raw_upper_violation", "c_raw_mass")):
            gates[rule + "_" + field] = s[field] <= stationary.GATES[bound]
        gates[rule + "_Ward"] = float(raw["ward_residual_max"][0, 0]) <= stationary.GATES["Ward"]
    return dict(qualified=all(gates.values()), gates=gates, failed_gates=[k for k, v in gates.items() if not v],
                comparisons=comp, scalar_diagnostics=scalars, requires_radial_stationarity=False,
                requires_low_or_high_K=False, requires_reference_eigengap=False)


def diagnose(anchor, q, *, manufactured_config=None):
    """Independent ref-only timing, then additional finite-K diagnostics."""
    tick = time.perf_counter(); q, m = _target(q); cfg = _configuration(m, manufactured_config)
    if not np.array_equal(q, anchor["q_original"]): raise ValueError("diagnosis target differs from anchor data")
    qr, _ = recode_frequency(q, anchor["signs"])
    if not np.array_equal(qr, anchor["q_recoded"]): raise ValueError("anchor recoding mismatch")
    theta = _real(anchor["theta_recoded"], "anchor theta").copy()
    if theta.shape != (2 * m,): raise ValueError("malformed anchor theta")
    phases = {}
    for phase, Ks in (("reference_only", ("inf",)), ("finite_extra", (8, 64))):
        started = time.perf_counter(); cost = _new_cost(); calls = []; raw = {}; failures = {}
        for rule in ("C", "D"):
            try:
                value = _rankone_call(theta, qr, Ks, rule, cfg, cost, calls, dict(phase=phase, rule=rule))
                stationary.project_result(value)
                raw[rule] = value
            except FloatingPointError as error:
                failures[rule] = str(error)
        cost["wall_seconds"] = time.perf_counter() - started
        phases[phase] = dict(raw=raw, calls=calls, failures=failures, cost=cost)
    ref, finite = phases["reference_only"]["raw"], phases["finite_extra"]["raw"]
    box = stationary.box_record(theta)
    ref_ok = len(ref) == 2
    ref_scalar_started = time.perf_counter()
    ref_qual = _reference_qualification(ref["C"], ref["D"]) if ref_ok else dict(qualified=False, failures=phases["reference_only"]["failures"])
    ref_prereq = dict(anchor_finite_feasible=bool(anchor["finite_feasible"] and box["feasible"]),
                      sign_resolved=bool(anchor["sign_resolved"]), box_margin=box["minimum_margin"] >= stationary.GATES["box_margin"],
                      reference_matrix_qualified=ref_qual["qualified"])
    if ref_ok:
        ref_mins_for_B = [float(ref[rule]["Hangle_projected_eigenvalues"][0, 0, 0]) for rule in ("C", "D")]
        ref_threshold_for_B = max(CURVATURE_FLOOR, DISAGREEMENT_MULTIPLIER * abs(ref_mins_for_B[0] - ref_mins_for_B[1]))
        independently_timed_B = all(ref_prereq.values()) and _sign_pair(ref_mins_for_B, ref_threshold_for_B) == "negative"
    else:
        independently_timed_B = False
    ref_scalar_seconds = time.perf_counter() - ref_scalar_started
    nums = dict(reference_qualification=ref_qual, finite_qualifications={}, coordinate_order="recoded_positive_amplitude",
                angle_Hessian_equal_to_signed_original_coordinates=True, direction_rule="D")
    low_values = ref_values = high_values = [None, None]
    low_diff = ref_diff = high_diff = ref_spectrum_diff = None
    gaps = None; direction = None
    full_prereq = dict(anchor_solver_completed=bool(anchor["solver_execution_completed"]), finite_matrices_available=len(finite) == 2,
                       all_full_matrix_qualifications=False, radial_gradient=False, radial_positive=False,
                       Cartesian_Ward=False, near_rotation_mode=False)
    if ref_ok:
        spectra = {rule: ref[rule]["Hangle_projected_eigenvalues"][0, 0].copy() for rule in ("C", "D")}
        ref_values = [float(spectra[rule][0]) for rule in ("C", "D")]
        ref_diff = abs(ref_values[0] - ref_values[1])
        ref_spectrum_diff = float(np.max(abs(spectra["C"] - spectra["D"])))
        gaps = [None if len(spectra[rule]) == 1 else float(spectra[rule][1] - spectra[rule][0]) for rule in ("C", "D")]
        direction = ref["D"]["helmert_basis"] @ ref["D"]["Hangle_projected_eigenvectors"][0, 0, :, 0]
        canonical_index = int(np.argmax(abs(direction)))
        if direction[canonical_index] < 0: direction = -direction
        nums.update(reference_spectra=spectra, direction=direction.copy(), direction_sign_coordinate=canonical_index)
    if len(finite) == 2:
        for ki, K in enumerate((8, 64)):
            nums["finite_qualifications"][str(K)] = stationary.qualification(finite["C"], finite["D"], 0, ki)
        low_values = [float(finite[rule]["Hangle_projected_eigenvalues"][0, 0, 0]) for rule in ("C", "D")]
        low_diff = abs(low_values[0] - low_values[1])
        if direction is not None:
            high_values = [float(direction @ finite[rule]["Hangle"][0, 1] @ direction) for rule in ("C", "D")]
            high_diff = abs(high_values[0] - high_values[1])
        low = finite["D"]; g = low["gradient"][0, 0]; Hperp = low["Hperp"][0, 0]; a = theta[m:]
        nums.update(anchor_gradient_Linf=float(np.max(abs(g))), anchor_Hba_min=float(low["H_ba_eigenvalues"][0, 0, 0]),
                    Cartesian_Ward_Linf=float(np.max(abs(Hperp @ a - g[m:]))),
                    rotation_residual_Linf=float(np.max(abs(Hperp @ (a / np.linalg.norm(a))))))
        full_prereq.update(radial_gradient=nums["anchor_gradient_Linf"] <= stationary.GATES["fit_gradient"],
                           radial_positive=nums["anchor_Hba_min"] > stationary.GATES["positive_eigenvalue"],
                           Cartesian_Ward=nums["Cartesian_Ward_Linf"] <= stationary.GATES["Cartesian_Ward"],
                           near_rotation_mode=nums["rotation_residual_Linf"] <= stationary.GATES["rotation_residual"])
    if ref_ok:
        nums["full_reference_qualification"] = stationary.qualification(ref["C"], ref["D"], 0, 0)
        full_prereq["all_full_matrix_qualifications"] = (len(finite) == 2 and nums["full_reference_qualification"]["qualified"]
                                                        and all(v["qualified"] for v in nums["finite_qualifications"].values()))
    decision = classify_diagnostic(low_min=low_values, reference_min=ref_values, high_curvature=high_values,
                                   low_disagreement=low_diff, reference_disagreement=ref_diff, high_disagreement=high_diff,
                                   reference_spectrum_disagreement=ref_spectrum_diff, reference_gaps=gaps,
                                   full_prerequisites=full_prereq, reference_prerequisites=ref_prereq)
    if decision["ref_raise"] != independently_timed_B: raise AssertionError("independent B readout changed in full classification")
    nums.update(low_min=low_values, reference_min=ref_values, high_curvature=high_values,
                low_disagreement=low_diff, reference_disagreement=ref_diff, high_disagreement=high_diff,
                reference_spectrum_disagreement=ref_spectrum_diff, reference_gaps=gaps, box=box,
                reference_baseline_evaluated_at_initialization_fallback=bool(anchor["used_initialization_fallback"]),
                reference_baseline_evaluated_after_anchor_numerical_failure=anchor["failure"] is not None)
    costs = {name: snapshot(value["cost"]) for name, value in phases.items()}
    costs["wall_seconds"] = time.perf_counter() - tick
    costs["reference_only"]["kernel_and_projection_seconds"] = costs["reference_only"]["wall_seconds"]
    costs["reference_only"]["scalar_readout_seconds"] = ref_scalar_seconds
    costs["reference_only"]["wall_seconds"] += ref_scalar_seconds
    costs["finite_extra"]["kernel_and_projection_seconds"] = costs["finite_extra"]["wall_seconds"]
    costs["finite_extra"]["wall_seconds"] = max(0., costs["wall_seconds"] - costs["reference_only"]["wall_seconds"])
    # Cost phases partition the full call. The latter includes full-only scalar
    # decisions/validation; B is measured independently and never charged those.
    phases["reference_only"]["cost"] = snapshot(costs["reference_only"])
    phases["finite_extra"]["cost"] = snapshot(costs["finite_extra"])
    costs["full_policy_diagnostic_seconds"] = costs["wall_seconds"]
    costs["reference_only_policy_diagnostic_seconds"] = costs["reference_only"]["wall_seconds"]
    costs["always_policy_diagnostic_seconds"] = 0.
    return snapshot(dict(schema="prospective_diagnosis_v1", status="completed_with_recorded_qualifications", **decision,
                         numerics=nums, reference_only=phases["reference_only"], finite_extra=phases["finite_extra"],
                         cost=costs, config=cfg, teacher_access=False))


def neutral_kick(anchor, xi, amplitude=.15):
    """Use the caller's one fixed xi draw; never redraw for poor alignment."""
    tick = time.perf_counter()
    theta = _real(anchor["theta_recoded"], "theta_recoded")
    signs = np.asarray(anchor["signs"]); m = len(signs)
    xi = _real(xi, "xi")
    if theta.shape != (2 * m,) or xi.shape != (m,): raise ValueError("kick shape mismatch")
    if not np.all((signs == 1) | (signs == -1)): raise ValueError("invalid anchor signs")
    amplitude = float(amplitude)
    if not math.isfinite(amplitude) or amplitude != .15: raise ValueError("configured kick amplitude is exactly .15")
    centered = xi - xi.mean()
    if not np.isfinite(centered).all(): raise FloatingPointError("nonfinite centered xi")
    scale = float(np.max(abs(centered)))
    if scale == 0: raise ValueError("degenerate xi; no redraw or replacement is performed")
    scaled = centered / scale
    scaled_norm = float(np.sqrt(np.dot(scaled, scaled)))
    norm = scale * scaled_norm
    if not math.isfinite(norm): raise FloatingPointError("centered xi norm is not representable")
    # Normalize through scaled coordinates, so small nonzero xi does not lose
    # its norm when its squared components underflow in a direct dot product.
    d = scaled / scaled_norm; a = signs * theta[m:]
    W = a[:, None] * np.column_stack((np.cos(amplitude * d), np.sin(amplitude * d)))
    return snapshot(dict(b=signs * theta[:m], W=W, status="prepared_shared_neutral_kick", xi=xi, centered_xi=centered,
                         direction=d, amplitude=amplitude, signs=signs, theta_recoded=theta,
                         centered_norm=norm, normalization_maxabs=scale, scaled_L2_norm=scaled_norm,
                         normalization_method="maxabs_scaled_L2",
                         generated_random_vectors=0, resampled=False, selected_by_alignment=False,
                         cost=dict(wall_seconds=time.perf_counter() - tick, generated_random_vectors=0,
                                   rng_generation_time_charged_here=False)))


def _pack(b, W):
    return np.r_[b, np.asarray(W).ravel()]


def _unpack(theta, m):
    return theta[:m].copy(), theta[m:].reshape(m, 2).copy()


def _start(start, m):
    if not isinstance(start, dict) or "b" not in start or "W" not in start:
        raise ValueError("start requires b and W")
    b, W = _real(start["b"], "start b"), _real(start["W"], "start W")
    if b.shape != (m,) or W.shape != (m, 2) or np.any(abs(b) > 2.) or np.any(abs(W) > 2.):
        raise ValueError("start must be finite, correctly shaped and in [-2,2]")
    return _pack(b, W)


def _checkpoint(theta, m, step):
    b, W = _unpack(theta, m)
    return dict(step=int(step), theta=theta.copy(), b=b, W=W,
                state_only=True, additional_objective_evaluation=False)


def train_arm(q, start, prior, K, *, manufactured_config=None):
    """Matched finite training using a caller-owned [updates,4,64,2] pool.

    Loss/gradient records are at theta_before_update. Checkpoints are states,
    with no additional scoring. A numerical nonfinite stops at finite LAST;
    input/schema errors propagate to the coordinating driver.
    """
    tick = time.perf_counter(); q, m = _target(q); cfg = _configuration(m, manufactured_config)
    K = _positive_integer(K, "K")
    if K not in (8, 64): raise ValueError("arm K must be 8 or 64")
    pool = _real(prior, "prior")
    updates = cfg["arm_updates"]
    if pool.shape != (updates, 4, 64, 2): raise ValueError("prior must have shape [updates,4,64,2]")
    theta = _start(start, m); initial = theta.copy()
    first = np.zeros_like(theta); second = np.zeros_like(theta)
    parameters = [theta.copy()]; loss_records = []; gradients = []; proposals = []
    first_records = []; second_records = []; masks = []; rates = []; evaluation_theta = []
    checkpoints = [_checkpoint(theta, m, 0)]
    cost = _new_cost(); active = int(np.count_nonzero(q)); failure = None; completed = 0
    status = "completed_fixed_budget"
    for step in range(updates):
        before = theta.copy(); b, W = _unpack(before, m)
        cost["kernel_calls"] += 1; cost["objective_calls"] += 1
        cost["processed_prior_vectors"] += 4 * K
        cost["label_node_products"] += 4 * K * active
        try:
            loss, gb, gW = joint.mc_loss_grad(b, W, q, pool[step, :, :K, :])
            grad = _pack(gb, gW)
            if not math.isfinite(loss) or not np.isfinite(grad).all():
                raise FloatingPointError("nonfinite MC loss/gradient")
            cost["successful_kernel_calls"] += 1; cost["loss_gradient_calls"] += 1
        except FloatingPointError as error:
            cost["failed_kernel_calls"] += 1
            failure = dict(stage="objective", step=step + 1, theta=before.copy(), message=str(error),
                           partial_internal_work_unmeasured=True)
            status = "nonfinite_objective_stopped_last_finite"; break
        # These exact pairs are retained even if a subsequent update fails.
        evaluation_theta.append(before.copy()); loss_records.append(float(loss)); gradients.append(grad.copy())
        lr = .01 if step < cfg["arm_switch_step"] else .002
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                new_first = .9 * first + .1 * grad
                new_second = .999 * second + .001 * grad * grad
                direction = (new_first / (1. - .9 ** (step + 1))) / (np.sqrt(new_second / (1. - .999 ** (step + 1))) + 1e-8)
                proposal = before - lr * direction
            if not np.isfinite(proposal).all(): raise FloatingPointError("nonfinite Adam proposal")
            candidate = np.clip(proposal, -2., 2.)
        except FloatingPointError as error:
            failure = dict(stage="update", step=step + 1, theta=before.copy(), message=str(error))
            status = "nonfinite_update_stopped_last_finite"; break
        theta, first, second = candidate.copy(), new_first.copy(), new_second.copy()
        proposals.append(proposal.copy()); masks.append(proposal != candidate); rates.append(lr)
        first_records.append(first.copy()); second_records.append(second.copy()); parameters.append(theta.copy())
        completed += 1; cost["optimizer_updates"] += 1
        if completed % cfg["checkpoint_every"] == 0:
            checkpoints.append(_checkpoint(theta, m, completed))
    if checkpoints[-1]["step"] != completed: checkpoints.append(_checkpoint(theta, m, completed))
    d = 3 * m
    as_rows = lambda x: np.asarray(x, dtype=float).reshape(-1, d)
    trace = dict(theta=np.asarray(parameters), theta_before_evaluation=as_rows(evaluation_theta),
                 loss_before_update=np.asarray(loss_records), gradient_before_update=as_rows(gradients),
                 proposal=as_rows(proposals), projected_mask=np.asarray(masks, dtype=bool).reshape(-1, d),
                 learning_rate=np.asarray(rates), adam_first_after_update=as_rows(first_records),
                 adam_second_after_update=as_rows(second_records),
                 loss_gradient_pairing="theta_before_evaluation; never label these as endpoint evaluations")
    b, W = _unpack(theta, m); cost["wall_seconds"] = time.perf_counter() - tick
    cost.update(active_label_count=active, prior_vectors_per_completed_update=4 * K,
                requested_updates=updates, completed_updates=completed,
                generated_shared_prior_vectors_charged_here=0)
    return snapshot(dict(schema="prospective_finite_arm_v1", b=b, W=W, status=status, K=int(K),
                         initial_theta=initial, theta=theta, last_finite_theta=theta,
                         completed_updates=completed, trace=trace, checkpoints=checkpoints, failure=failure,
                         cost=cost, config=cfg, prior_input_shape=pool.shape,
                         prior_use="first K particles of each of the caller's same four banks",
                         endpoint_objective_reevaluated=False, readout="LAST", teacher_access=False,
                         checkpoint_selection_by_risk=False, restarted=False))


class _ReferenceBudgetStop(Exception):
    pass


def fit_reference(q, start, *, manufactured_config=None):
    """GH L-BFGS-B reference, with a hard actual-objective-call budget.

    Normal completion uses the optimizer's final accepted x. On a hard budget
    request or nonfinite evaluation, return the most recent finite callback
    accepted state, never the last line-search trial. With no callback use the
    shared initial state. No new scoring is done to choose a fallback.
    """
    tick = time.perf_counter(); q, m = _target(q); cfg = _configuration(m, manufactured_config)
    initial = _start(start, m); theta = initial.copy()
    options = cfg["reference_options"].copy()
    for key in ("maxiter", "maxfun", "maxls"): options[key] = _positive_integer(options[key], key)
    cap = options["maxfun"]; order = cfg["reference_order"]
    evaluations, accepted = [], [_checkpoint(initial, m, 0)]
    attempts = actual = 0; failure = None; optimizer = None; cost = _new_cost()
    active = int(np.count_nonzero(q))
    def fun(at):
        nonlocal attempts, actual
        attempts += 1
        if actual >= cap: raise _ReferenceBudgetStop("actual GH objective call cap reached")
        actual += 1; cost["kernel_calls"] += 1; cost["objective_calls"] += 1
        cost["label_node_products"] += order * order * active
        record = dict(evaluation_id=actual - 1, theta=np.asarray(at).copy(), status="started")
        evaluations.append(record)
        b, W = _unpack(np.asarray(at), m)
        try:
            loss, gb, gW = joint.exact_loss_grad(b, W, q, order=order)
            gradient = _pack(gb, gW)
            if not math.isfinite(loss) or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite reference loss/gradient")
            cost["successful_kernel_calls"] += 1; cost["loss_gradient_calls"] += 1
            record.update(status="completed", loss=float(loss), gradient=gradient.copy())
            return float(loss), gradient
        except FloatingPointError as error:
            cost["failed_kernel_calls"] += 1
            record.update(status="nonfinite_failure", message=str(error), partial_internal_work_unmeasured=True)
            raise
    def callback(at):
        at = np.asarray(at, dtype=float)
        if not np.isfinite(at).all() or np.any(abs(at) > 2.):
            raise FloatingPointError("nonfinite or infeasible accepted reference state")
        accepted.append(_checkpoint(at, m, len(accepted)))
    try:
        result = minimize(fun, initial.copy(), method="L-BFGS-B", jac=True,
                          bounds=[(-2., 2.)] * len(initial), callback=callback, options=options)
        optimizer = dict(success=bool(result.success), status_code=int(result.status), message=str(result.message),
                         iterations=int(result.nit), reported_evaluations=int(result.nfev))
        if not np.isfinite(result.x).all() or np.any(abs(result.x) > 2.) or not math.isfinite(float(result.fun)):
            raise FloatingPointError("optimizer returned nonfinite or infeasible reference LAST")
        theta = np.asarray(result.x).copy()
        if result.success:
            status = "converged_gradient" if np.max(abs(result.jac)) <= options["gtol"] else "converged_function_tolerance"
        elif int(result.nit) >= options["maxiter"]: status = "iteration_budget_stop"
        elif actual >= cap: status = "objective_budget_stop"
        elif int(result.status) == 2: status = "line_search_stop"
        else: status = "optimizer_stopped_without_convergence"
    except _ReferenceBudgetStop as error:
        theta = accepted[-1]["theta"].copy(); status = "objective_budget_stop"
        failure = dict(kind="hard_objective_budget", message=str(error), next_request_not_evaluated=True)
    except FloatingPointError as error:
        theta = accepted[-1]["theta"].copy(); status = "nonfinite_stopped_last_accepted"
        failure = dict(kind="nonfinite_reference", message=str(error))
    final_pair = next((r for r in reversed(evaluations) if r["status"] == "completed" and np.array_equal(r["theta"], theta)), None)
    b, W = _unpack(theta, m)
    cost["optimizer_updates"] = max(0, len(accepted) - 1)
    cost["wall_seconds"] = time.perf_counter() - tick
    cost.update(attempted_calls=attempts, actual_evaluations=actual, active_label_count=active,
                normal_nodes_per_evaluation=order * order, hard_objective_cap=cap)
    return snapshot(dict(schema="prospective_reference_v1", b=b, W=W, theta=theta, last_finite_theta=theta,
                         status=status, initial_theta=initial, trace=dict(evaluations=evaluations, accepted_states=accepted),
                         checkpoints=accepted, optimizer=optimizer, failure=failure, cost=cost, config=cfg,
                         endpoint_loss=None if final_pair is None else final_pair["loss"],
                         endpoint_gradient=None if final_pair is None else final_pair["gradient"],
                         output_is_last_accepted_or_initial=True, global_MLE_claimed=False,
                         teacher_access=False, restarted=False))


def offline_policy_algebra(delta_R, A, B, *, anchor_seconds=0., reference_seconds=0.,
                           finite_extra_seconds=0., low_seconds=0., high_seconds=0.):
    """Post-lock scalar/vector identities; this routine does not evaluate risk."""
    gain = _real(delta_R, "delta_R")
    A, B = np.asarray(A), np.asarray(B)
    if not np.all((A == 0) | (A == 1)) or not np.all((B == 0) | (B == 1)):
        raise ValueError("policy indicators must be Boolean")
    gain, A, B = np.broadcast_arrays(gain, A.astype(float), B.astype(float))
    values = [_real(x, "cost") for x in (anchor_seconds, reference_seconds, finite_extra_seconds, low_seconds, high_seconds)]
    if any(np.any(x < 0) for x in values): raise ValueError("costs must be nonnegative")
    anchor_s, ref_s, extra_s, low_s, high_s = values
    full_cost = anchor_s + ref_s + extra_s + (1 - A) * low_s + A * high_s
    ref_cost = anchor_s + ref_s + (1 - B) * low_s + B * high_s
    always_low = anchor_s + low_s; always_high = anchor_s + high_s
    return snapshot(dict(full_minus_low_risk=-A * gain, full_minus_high_risk=(1 - A) * gain,
                         reference_minus_low_risk=-B * gain, reference_minus_high_risk=(1 - B) * gain,
                         reference_minus_full_risk=(A - B) * gain,
                         full_policy_cost=full_cost, reference_policy_cost=ref_cost,
                         always_low_cost=always_low, always_high_cost=always_high,
                         full_minus_high_cost=ref_s + extra_s - (1 - A) * (high_s - low_s),
                         reference_minus_high_cost=ref_s - (1 - B) * (high_s - low_s),
                         fewer_raises_alone_is_not_risk_improvement=True))
