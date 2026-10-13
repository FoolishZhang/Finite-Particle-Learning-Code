"""Post-EVALUATION_LOCK execution audit for the fixed development panel.

This independently reconstructs saved labels, recoding, diagnostic decisions,
all saved Adam updates and scalar policy identities. Selected finite gradients
are recalculated from saved priors. It does not retrain any arm or independently
repeat all rank-one or observation-PMF integrals, and is not external review.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import traceback

import prospective_diagnostic_analyze as analysis
import prospective_diagnostic_dev as dev
import numpy as np
from scipy.special import expit, logsumexp

ROOT, DOC, OUT = dev.ROOT, dev.DOC, dev.OUT
SELECTED_FINITE_GRADIENT_STEPS = (0, 400, 599)
COUNTER_KEYS = analysis.COST_KEYS


def safe(value):
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return dict(nonfinite=repr(value))
    return value


class Checks:
    def __init__(self): self.count = 0; self.failures = []; self.max_errors = {}; self.elements = 0
    def require(self, condition, label, detail=None):
        self.count += 1
        if not bool(condition): self.failures.append(dict(check=label, detail=safe(detail)))
    def exact(self, actual, expected, label): self.require(safe(actual) == safe(expected), label)
    def close(self, actual, expected, label, atol=2e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        okay = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(abs(a - b), initial=0.)) if okay else None
        self.elements += a.size
        family = label.split(":")[-1]
        if error is not None and error > self.max_errors.get(family, {}).get("error", -1.):
            self.max_errors[family] = dict(error=error, context=label)
        self.require(okay and error <= atol, label, dict(error=error, atol=atol))


def patterns(m): return ((np.arange(1 << m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(float)
def packed(b, W): return np.r_[b, np.asarray(W).ravel()]
def symmetric(H): return (H + H.swapaxes(-1, -2)) / 2
def helmert(m):
    Q = np.zeros((m, m - 1))
    for j in range(1, m): Q[:j, j - 1] = 1. / math.sqrt(j * (j + 1)); Q[j, j - 1] = -j / math.sqrt(j * (j + 1))
    return Q
def feasible(theta, rankone=False):
    theta = np.asarray(theta)
    lower = np.r_[np.full(len(theta)//2, -2.), np.full(len(theta)//2, .05)] if rankone else np.full(len(theta), -2.)
    return bool(np.isfinite(theta).all() and np.all(theta >= lower) and np.all(theta <= 2.))


def restored(stem, checks):
    envelope = dev.read(Path(stem).with_suffix(".json")); expected_raw = Path(stem).with_suffix(".npz")
    checks.exact(envelope["raw"]["path"], dev.rel(expected_raw), str(stem) + ":raw_path")
    references = []
    def walk(value):
        if isinstance(value, dict):
            if "__ndarray__" in value: references.append(value["__ndarray__"])
            else:
                for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(envelope["tree"])
    with np.load(expected_raw, allow_pickle=False) as archive:
        checks.require(len(references) == len(set(references)) and set(references) == set(archive.files), str(stem) + ":raw_array_inventory")
    return dev.restore(stem)


def check_generated(unit, data, teacher, checks):
    name = unit["id"]; N = unit["N"]; z = data["latent"]; u = data["uniforms"]
    checks.exact(data["unit"], unit, name + ":unit_identity")
    checks.exact(z.shape, (N, 2), name + ":latent_shape"); checks.exact(u.shape, (N, 8), name + ":uniform_shape")
    checks.require(np.isfinite(z).all() and np.isfinite(u).all() and np.all((u >= 0) & (u < 1)), name + ":saved_inputs_valid")
    W = np.asarray(teacher["a"])[:, None] * np.c_[np.cos(teacher["phi"]), np.sin(teacher["phi"])]
    y = (u < expit(np.asarray(teacher["b"]) + z @ W.T)).astype(np.int8)
    index = y @ 2 ** np.arange(7, -1, -1)
    count = np.bincount(index, minlength=256)
    checks.close(data["labels"], y, name + ":labels_from_saved_Z_uniform", 0.)
    checks.close(data["counts"], count, name + ":label_counts", 0.)
    checks.close(data["q"], count / N, name + ":frequency", 0.)
    checks.exact(data["prior"].shape, (600, 4, 64, 2), name + ":shared_prior_shape")
    checks.require(np.isfinite(data["prior"]).all(), name + ":shared_prior_finite")
    checks.exact(data["prior_vectors_generated"], 153600, name + ":prior_generated_once")
    checks.exact(data["Gaussian_scalars"], 2 * N + 8 + 600 * 4 * 64 * 2, name + ":Gaussian_scalar_count")
    checks.exact(data["uniform_scalars"], 8 * N, name + ":uniform_scalar_count")
    for key in ("seconds", "data_generation_seconds", "kick_generation_seconds", "prior_generation_seconds"):
        analysis.required_number(data, key)


def check_initialization(data, anchor, checks, name):
    q = data["q"]; m = int(round(math.log2(len(q)))); Y = patterns(m)
    p = np.sum(q[:, None] * Y, axis=0)
    covariance = np.einsum("p,pi,pj->ij", q, Y - p, Y - p)
    var = p * (1 - p); scale = np.maximum(var, 1e-8)
    corr = covariance / np.sqrt(scale[:, None] * scale[None, :])
    values, vectors = np.linalg.eigh(symmetric(corr)); v = vectors[:, -1].copy(); pivot = int(np.argmax(abs(v)))
    if v[pivot] < 0: v *= -1
    gap = float(values[-1] - values[-2]); resolved = gap > 1e-8 and bool(np.all(abs(v) > 1e-6))
    signs = np.where(v >= 0, 1, -1).astype(np.int8) if resolved else np.ones(m, dtype=np.int8)
    init = anchor["initialization"]
    for key, expected in (("marginals", p), ("covariance", covariance), ("correlation", corr), ("eigenvalues", values)):
        checks.close(init[key], expected, name + ":sign_init_" + key)
    checks.exact(init["sign_resolved"], resolved, name + ":sign_resolved")
    checks.close(anchor["signs"], signs, name + ":chosen_signs", 0.)
    mask = int(np.sum((signs < 0) * 2 ** np.arange(m - 1, -1, -1)))
    permutation = np.arange(len(q)) ^ mask; qr = q[permutation]
    checks.close(init["old_to_new_pattern_index"], permutation, name + ":bit_XOR_permutation", 0.)
    checks.close(anchor["q_recoded"], qr, name + ":recoded_target", 0.)
    checks.close(anchor["q_original"], q, name + ":original_target", 0.)
    smooth = (data["unit"]["N"] * p + .5) / (data["unit"]["N"] + 1)
    b = np.clip(np.log(smooth) - np.log1p(-smooth), -2., 2.)
    theta0 = np.r_[signs * b, np.full(m, .5)]
    checks.close(init["theta0_recoded"], theta0, name + ":initial_theta_recoded")
    checks.exact(init["target_smoothed"], False, name + ":no_target_smoothing")
    checks.exact(init["target_renormalized"], False, name + ":no_target_renormalization")
    theta = anchor["theta_recoded"]
    checks.close(anchor["b"], signs * theta[:m], name + ":original_bias_map", 0.)
    checks.close(anchor["W"], np.c_[signs * theta[m:], np.zeros(m)], name + ":signed_loading_map", 0.)
    checks.close(anchor["theta_signed"], np.r_[anchor["b"], signs * theta[m:]], name + ":signed_theta", 0.)
    checks.require(feasible(theta, True), name + ":anchor_box")
    return qr


def check_kick(data, anchor, start, checks, name):
    xi = np.asarray(data["xi"]); centered = xi - xi.mean(); norm = math.hypot(*centered)
    checks.require(norm > 0 and math.isfinite(norm), name + ":neutral_norm_nonzero")
    direction = centered / norm; m = len(xi); signed_a = anchor["signs"] * anchor["theta_recoded"][m:]
    W = signed_a[:, None] * np.c_[np.cos(.15 * direction), np.sin(.15 * direction)]
    checks.close(start["xi"], xi, name + ":same_raw_xi", 0.)
    checks.close(start["centered_xi"], centered, name + ":centered_xi")
    checks.close(start["centered_norm"] / norm, 1., name + ":stable_centered_norm_relative", 2e-14)
    checks.close(start["direction"], direction, name + ":neutral_direction")
    checks.close(float(np.sum(start["direction"])), 0., name + ":neutral_direction_centering")
    checks.close(np.linalg.norm(start["direction"]), 1., name + ":neutral_direction_norm")
    checks.close(start["W"], W, name + ":signed_neutral_kick")
    checks.close(start["b"], anchor["b"], name + ":kick_bias_unchanged", 0.)
    checks.close(np.linalg.norm(start["W"], axis=1), abs(signed_a), name + ":kick_row_norms")
    checks.exact(start["resampled"], False, name + ":no_kick_redraw")
    checks.exact(start["selected_by_alignment"], False, name + ":no_kick_selection")
    checks.exact(start["amplitude"], .15, name + ":kick_amplitude")
    analysis.required_number(start["cost"], "wall_seconds")


def reconstruct_raw(raw, checks, name):
    q, a = np.asarray(raw["targets"]), np.asarray(raw["a"]); m = len(a)
    combine = lambda array: np.stack([np.sum(array * weights.reshape((1, len(weights)) + (1,) * (array.ndim - 2)), axis=1) for weights in q])
    loss, grad = combine(raw["label_loss"]), combine(raw["label_gradient"])
    checks.close(raw["loss"], loss, name + ":weighted_loss")
    checks.close(raw["gradient"], grad, name + ":weighted_gradient")
    checks.close(raw["target_d_mass"], combine(raw["label_d_mass"]), name + ":weighted_d_mass")
    checks.close(raw["normal_quadrature_mass"], math.fsum(map(float, raw["normal_weights"])), name + ":normal_mass", 0.)
    mass = float(raw["normal_quadrature_mass"])
    expected = np.array([1. if K == "inf" else math.exp(K * math.log(mass)) for K in raw["K_values"]])
    checks.close(raw["raw_bank_quadrature_mass"], expected, name + ":raw_bank_mass", 0.)
    if raw["H_ba"] is None: return
    label_H = raw["label_B"] - raw["label_same_particle_term"] + raw["label_cross_particle_term"]
    checks.close(raw["label_H_ba"], label_H, name + ":Hba_label_decomposition", 0.)
    Hba = combine(label_H); M = combine(raw["label_M"]); sp = combine(raw["label_sigma_prime_average"])
    Hp = -M.copy(); ix = np.arange(m); Hp[..., ix, ix] += sp
    edges = M * a[None, None, :, None] * a[None, None, None, :]; edges[..., ix, ix] = 0.
    Hangle = -edges.copy(); Hangle[..., ix, ix] = edges.sum(axis=-1)
    ward = a[:, None] * Hp * a[None, :]; ward[..., ix, ix] -= a * grad[..., m:]
    for key, value in (("H_ba", Hba), ("Hperp", Hp), ("Hangle", Hangle), ("directWard", ward), ("ward_residual", ward - Hangle)):
        checks.close(raw[key], value, name + ":rebuilt_" + key)
    checks.close(raw["ward_residual_max"], np.max(abs(ward - Hangle), axis=(-1, -2)), name + ":Ward_max")
    if "Hangle_projected" in raw:
        Q = helmert(m); V = np.asarray(raw["transverse_basis"])
        checks.close(raw["helmert_basis"], Q, name + ":fixed_Helmert")
        checks.close(V.T @ V, np.eye(m - 1), name + ":transverse_orthonormal")
        checks.close(V.T @ a, np.zeros(m - 1), name + ":transverse_complement")
        for key, matrix in (("H_ba", Hba), ("Hangle_projected", Q.T @ Hangle @ Q), ("Hperp_projected", V.T @ Hp @ V)):
            checks.close(raw[key], matrix, name + ":projected_" + key)
            vals, vecs = raw[key + "_eigenvalues"], raw[key + "_eigenvectors"]
            checks.close(vals, np.linalg.eigvalsh(symmetric(matrix)), name + ":eigenvalues_" + key)
            checks.close(vecs.swapaxes(-1, -2) @ vecs, np.broadcast_to(np.eye(vecs.shape[-1]), vecs.shape), name + ":eigenbasis_orthonormal_" + key)
            checks.close(vecs @ (vals[..., None] * vecs.swapaxes(-1, -2)), symmetric(matrix), name + ":eigendecomposition_" + key)


def scalar_gate(raw, ki):
    mass = float(raw["normal_quadrature_mass"]); expected = float(raw["raw_bank_quadrature_mass"][ki])
    c = raw["label_c_mass"][ki]; upper = float(raw["raw_c_mass_upper"][ki])
    return dict(normal=abs(mass - 1.) <= 2e-12, d=bool(np.max(abs(raw["label_d_mass"][ki] / expected - 1.)) <= 2e-7),
        c_lower=bool(np.min(c) >= -2e-7), c_upper=bool(np.max(c) - upper <= 2e-7),
        Ward=bool(raw["ward_residual_max"][0, ki] <= 2e-7))


def numerical_quality(C, D, ki, reference_only=False):
    gates = {}
    if reference_only:
        # B intentionally does not require Hperp, Hba, radial stationarity,
        # reference eigengap or any finite-K qualification.
        gates["Hangle"] = float(np.max(abs(C["Hangle"] - D["Hangle"]))) <= 2e-7
        gates["spectrum"] = float(np.max(abs(C["Hangle_projected_eigenvalues"] - D["Hangle_projected_eigenvalues"]))) <= 2e-6
    else:
        for key, tol in (("loss", 1e-7), ("gradient", 2e-7), ("H_ba", 2e-7), ("Hperp", 2e-7), ("Hangle", 2e-7),
                         ("H_ba_eigenvalues", 2e-6), ("Hperp_projected_eigenvalues", 2e-6), ("Hangle_projected_eigenvalues", 2e-6)):
            gates[key] = float(np.max(abs(C[key][0, ki] - D[key][0, ki]))) <= tol
    for rule, raw in (("C", C), ("D", D)):
        gates.update({rule + "_" + key: value for key, value in scalar_gate(raw, ki).items()})
    return all(gates.values())


def independent_decision(low, reference, high, differences, gaps, full_prerequisites, ref_prerequisites):
    thresholds = {key: None if value is None else max(5e-4, 10. * value) for key, value in differences.items() if key in ("low", "reference", "high")}
    def sign(values, threshold):
        if threshold is None or values is None or any(v is None or not math.isfinite(v) for v in values): return "unavailable"
        if min(values) > threshold: return "positive"
        if max(values) < -threshold: return "negative"
        return "unresolved"
    signs = {name: sign(values, thresholds[name]) for name, values in (("low", low), ("reference", reference), ("high", high))}
    B = all(ref_prerequisites.values()) and signs["reference"] == "negative"
    prerequisites = {"reference." + k: v for k, v in ref_prerequisites.items()}; prerequisites.update(full_prerequisites)
    failed = [key for key, value in prerequisites.items() if not value]
    computation = [key for key in failed if not key.endswith("sign_resolved")]
    resolution = [key for key in failed if key.endswith("sign_resolved")]
    gap_limit = None if differences["reference_spectrum"] is None else 10. * differences["reference_spectrum"]
    gap_ok = gaps is not None and gap_limit is not None and all(value is None or value > gap_limit for value in gaps)
    level, reasons, suppression = "D0", failed.copy(), False
    if not failed:
        if signs["low"] == "negative": level, reasons = "D1", ["low_already_negative"]
        elif signs["low"] != "positive": reasons = ["low_resolution_band"]; resolution += reasons
        elif signs["reference"] == "positive": level, reasons = "D1", ["reference_positive"]
        elif signs["reference"] != "negative": reasons = ["reference_resolution_band"]; resolution += reasons
        else:
            suppression = True
            if not gap_ok: reasons = ["reference_direction_gap_unresolved"]; resolution += reasons
            elif signs["high"] == "negative": level, reasons = "D3", ["suppression_and_release"]
            elif signs["high"] == "positive": level, reasons = "D2", ["suppression_high_direction_positive"]
            else: reasons = ["suppression_confirmed_release_unresolved"]; resolution += reasons
    ref_reasons = [key for key, value in ref_prerequisites.items() if not value]
    if not ref_reasons and not B: ref_reasons = ["reference_" + signs["reference"]]
    return dict(full_raise=level == "D3", ref_raise=bool(B), full_level=level, full_reasons=reasons, ref_reasons=ref_reasons,
        thresholds=thresholds, signs=signs, reference_gap_threshold=gap_limit, reference_direction_resolved=gap_ok,
        suppression_confirmed=suppression, full_prerequisites=prerequisites,
        D0_reason_categories=dict(computation_or_solver=computation, resolution=resolution))


def check_diagnosis(anchor, diag, checks, name):
    m = len(anchor["signs"]); theta = anchor["theta_recoded"]
    ref, finite = diag["reference_only"]["raw"], diag["finite_extra"]["raw"]
    for phase, Ks in (("reference_only", ["inf"]), ("finite_extra", [8, 64])):
        part = diag[phase]; checks.exact(len(part["calls"]), 2, name + ":fixed_" + phase + "_calls")
        for call in part["calls"]:
            checks.close(call["theta_recoded"], theta, name + ":diagnosis_same_anchor", 0.)
            checks.exact(call["K_values"], Ks, name + ":diagnostic_K")
            analysis.required_number(call, "wall_seconds")
            if call["status"] == "completed":
                checks.close(call["raw"]["targets"][0], anchor["q_recoded"], name + ":diagnostic_q", 0.)
                reconstruct_raw(call["raw"], checks, name + ":diagnostic_raw_" + phase + call["rule"])
        for rule, raw in part["raw"].items(): reconstruct_raw(raw, checks, name + ":projected_" + phase + rule)
        checks.exact(part["cost"]["kernel_calls"], len(part["calls"]), name + ":diagnostic_call_count")
    low = refmins = high = [None, None]; gaps = None; direction = None
    differences = dict(low=None, reference=None, high=None, reference_spectrum=None)
    margin = float(np.min(np.minimum(theta - np.r_[np.full(m, -2.), np.full(m, .05)], 2. - theta)))
    refq = len(ref) == 2 and numerical_quality(ref["C"], ref["D"], 0, True)
    refpre = dict(anchor_finite_feasible=bool(anchor["finite_feasible"] and feasible(theta, True)), sign_resolved=bool(anchor["sign_resolved"]),
                  box_margin=margin >= 1e-3, reference_matrix_qualified=refq)
    fullpre = dict(anchor_solver_completed=bool(anchor["solver_execution_completed"]), finite_matrices_available=len(finite) == 2,
        all_full_matrix_qualifications=False, radial_gradient=False, radial_positive=False, Cartesian_Ward=False, near_rotation_mode=False)
    if len(ref) == 2:
        spectra = [np.linalg.eigvalsh(symmetric(helmert(m).T @ ref[rule]["Hangle"][0, 0] @ helmert(m))) for rule in ("C", "D")]
        refmins = [float(s[0]) for s in spectra]; differences["reference"] = abs(refmins[0] - refmins[1])
        differences["reference_spectrum"] = float(np.max(abs(spectra[0] - spectra[1])))
        gaps = [None if m == 2 else float(s[1] - s[0]) for s in spectra]
        # Verify and use the saved prescribed D eigenbasis rather than selecting
        # a different basis in a multiple/near-multiple minimum eigenspace.
        saved_vector = ref["D"]["Hangle_projected_eigenvectors"][0, 0, :, 0]
        matrix = symmetric(helmert(m).T @ ref["D"]["Hangle"][0, 0] @ helmert(m))
        checks.close(matrix @ saved_vector, spectra[1][0] * saved_vector, name + ":saved_minimum_eigenvector")
        direction = helmert(m) @ saved_vector
        if direction[np.argmax(abs(direction))] < 0: direction *= -1
        checks.close(diag["numerics"]["direction"], direction, name + ":reference_direction", 2e-8)
        checks.exact(diag["numerics"]["reference_qualification"]["qualified"], refq, name + ":reference_quality")
    if len(finite) == 2:
        low = [float(np.linalg.eigvalsh(symmetric(helmert(m).T @ finite[rule]["Hangle"][0, 0] @ helmert(m)))[0]) for rule in ("C", "D")]
        differences["low"] = abs(low[0] - low[1])
        if direction is not None:
            high = [float(direction @ finite[rule]["Hangle"][0, 1] @ direction) for rule in ("C", "D")]
            differences["high"] = abs(high[0] - high[1])
        D = finite["D"]; g, Hp, a = D["gradient"][0, 0], D["Hperp"][0, 0], theta[m:]
        fullpre.update(radial_gradient=bool(np.max(abs(g)) <= 1e-8), radial_positive=bool(np.linalg.eigvalsh(symmetric(D["H_ba"][0, 0]))[0] > 1e-5),
            Cartesian_Ward=bool(np.max(abs(Hp @ a - g[m:])) <= 2e-7), near_rotation_mode=bool(np.max(abs(Hp @ (a / np.linalg.norm(a)))) <= 2e-7))
        for ki, K in enumerate((8, 64)):
            checks.exact(diag["numerics"]["finite_qualifications"][str(K)]["qualified"], numerical_quality(finite["C"], finite["D"], ki), name + ":finite_quality_" + str(K))
        if len(ref) == 2:
            fullpre["all_full_matrix_qualifications"] = numerical_quality(ref["C"], ref["D"], 0) and all(numerical_quality(finite["C"], finite["D"], ki) for ki in (0, 1))
    decision = independent_decision(low, refmins, high, differences, gaps, fullpre, refpre)
    for key in ("full_raise", "ref_raise", "full_level", "full_reasons", "ref_reasons", "signs", "D0_reason_categories", "full_prerequisites", "suppression_confirmed"):
        checks.exact(diag[key], decision[key], name + ":independent_" + key)
    checks.exact(diag["reference_prerequisites"], refpre, name + ":independent_reference_prerequisites")
    checks.exact(diag["reference_direction_resolved"], decision["reference_direction_resolved"], name + ":reference_gap_gate")
    for key, value in decision["thresholds"].items():
        if value is None: checks.exact(diag["thresholds"][key], None, name + ":unavailable_threshold_" + key)
        else: checks.close(diag["thresholds"][key], value, name + ":threshold_" + key)
    checks.require(not diag["full_raise"] or diag["ref_raise"], name + ":A_implies_B")
    checks.close(diag["cost"]["wall_seconds"], diag["cost"]["reference_only"]["wall_seconds"] + diag["cost"]["finite_extra"]["wall_seconds"], name + ":diagnostic_cost_partition", 1e-9)
    checks.require(diag["teacher_access"] is False, name + ":data_only_diagnosis")


def check_newton_trace(anchor, checks, name):
    trace = anchor["trace"]; calls = trace.get("calls", []); m = len(anchor["signs"])
    for rawcall in anchor["calls"]:
        analysis.required_number(rawcall, "wall_seconds")
        if rawcall["status"] != "completed": continue
        raw = rawcall["raw"]; reconstruct_raw(raw, checks, name + ":anchor_kernel")
        i = rawcall["context"]["call_id"]
        checks.close(calls[i]["theta"], rawcall["theta_recoded"], name + ":solver_call_theta", 0.)
        checks.close(calls[i]["loss"], raw["loss"][0, 0], name + ":solver_call_loss", 0.)
        checks.close(calls[i]["gradient"], raw["gradient"][0, 0], name + ":solver_call_gradient", 0.)
        if calls[i]["compute_hessian"]: checks.close(calls[i]["H_ba"], raw["H_ba"][0, 0], name + ":solver_call_H", 0.)
    for step in trace.get("steps", []):
        current = calls[step["current_call_id"]]; theta, g, H = current["theta"], current["gradient"], symmetric(current["H_ba"])
        eig = np.linalg.eigvalsh(H); mu = max(0., 1e-4 - float(eig[0])); rawd = np.linalg.solve(H + mu * np.eye(2 * m), -g)
        factor = min(1., .25 / np.max(abs(rawd))); d = factor * rawd
        lower = np.r_[np.full(m, -2.), np.full(m, .05)]; bounds = []
        for i, value in enumerate(d):
            if value > 0: bounds.append((2. - theta[i]) / value)
            elif value < 0: bounds.append((lower[i] - theta[i]) / value)
        alpha0 = min(1., .99 * min(bounds))
        for key, value in (("mu", mu), ("raw_direction", rawd), ("capped_direction", d), ("initial_alpha", alpha0), ("sym_H", H)):
            checks.close(step[key], value, name + ":Newton_" + key, 2e-10)
        checks.require(len(step["trials"]) <= 13, name + ":line_search_budget")
        for j, trial in enumerate(step["trials"]):
            alpha = alpha0 * 2. ** -j; proposal = theta + alpha * d; call = calls[trial["call_id"]]
            rhs = current["loss"] + 1e-4 * alpha * float(g @ d) + 5e-13
            checks.close(trial["theta"], proposal, name + ":Newton_trial_theta")
            checks.close(call["theta"], proposal, name + ":Newton_call_trial")
            checks.close(trial["armijo_rhs"], rhs, name + ":Armijo_rhs")
            checks.exact(trial["accepted"], bool(call["loss"] <= rhs), name + ":Armijo_decision")
            checks.exact(trial["halvings"], j, name + ":halving_count")
            if trial["accepted"]: checks.exact(j, len(step["trials"]) - 1, name + ":accept_stops_line_search")
    for stage in trace.get("stages", []):
        max_steps, max_calls = (12, 80) if stage["phase"] == "O" else (4, 40)
        checks.require(stage["accepted_step_count"] <= max_steps and stage["call_count"] <= max_calls, name + ":stage_budget")
    accepted = trace.get("accepted_states", [])
    if anchor["solver_execution_completed"]:
        checks.exact([s["phase"] for s in trace["stages"]], ["O", "C"], name + ":both_solver_stages")
        checks.close(trace["stages"][1]["start_theta"], trace["stages"][0]["final_theta"], name + ":C_from_O_LAST", 0.)
        checks.close(anchor["theta_recoded"], trace["final_theta"], name + ":completed_LAST", 0.)
    if accepted and not anchor["used_initialization_fallback"]:
        checks.close(anchor["theta_recoded"], accepted[-1]["theta"], name + ":last_accepted_anchor", 0.)
    elif anchor["used_initialization_fallback"]:
        checks.close(anchor["theta_recoded"], anchor["initialization"]["theta0_recoded"], name + ":explicit_initial_fallback", 0.)
    checks.exact(anchor["cost"]["kernel_calls"], len(anchor["calls"]), name + ":anchor_call_count")


def independent_mc(b, W, q, banks):
    Y = patterns(len(b)); x = b + banks @ W.T
    log_likelihood = np.einsum("bkm,pm->bkp", -np.logaddexp(0., -x), Y) + np.einsum("bkm,pm->bkp", -np.logaddexp(0., x), 1 - Y)
    normalization = logsumexp(log_likelihood, axis=1)
    posterior = np.exp(log_likelihood - normalization[:, None, :])
    weights = posterior * q[None, None, :]
    residual = expit(x) * weights.sum(axis=2)[:, :, None] - np.einsum("bkp,pm->bkm", weights, Y)
    gb = residual.sum(axis=(0, 1)) / len(banks)
    gW = np.einsum("bkm,bkd->md", residual, banks) / len(banks)
    loss = -float(np.mean((normalization - math.log(banks.shape[1])) @ q))
    return loss, packed(gb, gW)


def check_arm(data, start, fit, checks, name, targeted=True):
    trace = fit["trace"]; theta = trace["theta"]; gradients = trace["gradient_before_update"]
    m = len(start["b"]); initial = packed(start["b"], start["W"]); completed = fit["completed_updates"]; K = fit["K"]
    checks.close(fit["initial_theta"], initial, name + ":shared_initial", 0.)
    checks.close(theta[0], initial, name + ":trace_initial", 0.)
    checks.exact(theta.shape, (completed + 1, 3 * m), name + ":state_snapshot_count")
    checks.exact(len(gradients), fit["cost"]["successful_kernel_calls"], name + ":saved_gradient_count")
    first = np.zeros(3 * m); second = first.copy()
    for i in range(completed):
        g = gradients[i]; first = .9 * first + .1 * g; second = .999 * second + .001 * g * g
        lr = .01 if i < fit["config"]["arm_switch_step"] else .002
        proposal = theta[i] - lr * (first / (1 - .9 ** (i + 1))) / (np.sqrt(second / (1 - .999 ** (i + 1))) + 1e-8)
        for key, expected in (("theta_before_evaluation", theta[i]), ("adam_first_after_update", first), ("adam_second_after_update", second), ("proposal", proposal)):
            checks.close(trace[key][i], expected, name + ":Adam_" + key)
        checks.close(theta[i + 1], np.clip(proposal, -2., 2.), name + ":Adam_projected_state")
        checks.close(trace["learning_rate"][i], lr, name + ":learning_rate", 0.)
        checks.exact(trace["projected_mask"][i], proposal != np.clip(proposal, -2., 2.), name + ":projection_mask")
    checks.close(fit["theta"], theta[-1], name + ":finite_LAST", 0.)
    checks.close(packed(fit["b"], fit["W"]), theta[-1], name + ":packed_endpoint", 0.)
    checks.require(all(feasible(t) for t in theta), name + ":all_saved_states_in_box")
    checks.close(fit["checkpoints"][-1]["theta"], theta[-1], name + ":last_checkpoint", 0.)
    for checkpoint in fit["checkpoints"]: checks.close(checkpoint["theta"], theta[checkpoint["step"]], name + ":checkpoint_snapshot", 0.)
    calls = fit["cost"]["kernel_calls"]
    checks.exact(fit["cost"]["processed_prior_vectors"], calls * 4 * K, name + ":actual_prior_processing")
    checks.exact(fit["cost"]["label_node_products"], calls * 4 * K * np.count_nonzero(data["q"]), name + ":label_node_count")
    checks.exact(fit["prior_input_shape"], data["prior"].shape, name + ":shared_pool_shape")
    checks.exact(fit["teacher_access"], False, name + ":no_teacher_in_training")
    if completed == 600: checks.exact(fit["cost"]["processed_prior_vectors"], 19200 if K == 8 else 153600, name + ":full_budget_processing")
    recomputed = []
    if targeted:
        for i in SELECTED_FINITE_GRADIENT_STEPS:
            if i >= len(gradients): continue
            at = trace["theta_before_evaluation"][i]
            loss, grad = independent_mc(at[:m], at[m:].reshape(m, 2), data["q"], data["prior"][i, :, :K, :])
            checks.close(trace["loss_before_update"][i], loss, name + ":independent_MC_loss", 2e-10)
            checks.close(gradients[i], grad, name + ":independent_MC_gradient", 2e-10)
            recomputed.append(i)
    return dict(completed_updates=completed, saved_updates_replayed=completed, independently_recomputed_gradient_steps_zero_based=recomputed,
                attempted_calls=calls, processed_prior_vectors=fit["cost"]["processed_prior_vectors"])


def check_reference(start, fit, checks, name):
    calls = fit["trace"]["evaluations"]; states = fit["trace"]["accepted_states"]
    checks.close(states[0]["theta"], packed(start["b"], start["W"]), name + ":reference_shared_start", 0.)
    checks.close(fit["theta"], states[-1]["theta"], name + ":reference_last_accepted", 0.)
    checks.close(packed(fit["b"], fit["W"]), fit["theta"], name + ":reference_packed_endpoint", 0.)
    checks.require(len(calls) <= 180, name + ":reference_hard_call_cap")
    checks.exact(fit["cost"]["actual_evaluations"], len(calls), name + ":reference_actual_evaluations")
    checks.exact(fit["cost"]["kernel_calls"], len(calls), name + ":reference_kernel_calls")
    checks.exact(fit["cost"]["optimizer_updates"], len(states) - 1, name + ":reference_accepted_updates")
    checks.require(all(feasible(s["theta"]) for s in states), name + ":reference_feasible_states")
    for state in states[1:]:
        checks.require(any(r["status"] == "completed" and np.array_equal(r["theta"], state["theta"]) for r in calls), name + ":accepted_state_was_evaluated")
    checks.exact(fit["output_is_last_accepted_or_initial"], True, name + ":reference_LAST_semantics")
    checks.exact(fit["teacher_access"], False, name + ":reference_no_teacher")


def check_risk(data, target, risk, checks, name):
    checks.exact(risk["main_index"], 1, name + ":GH60_main_index")
    checks.exact(risk["rules"], [[40, 0.], [60, 0.], [80, 0.], [72, math.pi / 7]], name + ":fixed_risk_rules")
    KL = {}; qualified = {}
    for arm, P in risk["pmfs"].items():
        checks.require(np.isfinite(P).all() and np.all(P > 0) and np.all(target > 0), name + ":positive_raw_PMF_" + arm)
        KL[arm] = np.sum(target * (np.log(target) - np.log(P)), axis=1)
        checks.close(risk["KL"][arm], KL[arm], name + ":raw_KL_" + arm, 0.)
        checks.close(risk["empirical_NLL"][arm], -np.log(P) @ data["q"], name + ":empirical_NLL_" + arm, 0.)
        mass = max(float(np.max(abs(P.sum(axis=1) - 1))), float(np.max(abs(target.sum(axis=1) - 1))))
        discrepancy = max(float(np.max(abs(P - P[1]))), float(np.max(abs(target - target[1]))))
        qualified[arm] = mass <= 2e-12 and discrepancy <= 2e-8
        checks.exact(risk["qualified_by_arm"][arm], qualified[arm], name + ":risk_quality_" + arm)
        checks.close(risk["errors"][arm]["mass_error"], mass, name + ":mass_error_" + arm, 0.)
        checks.close(risk["errors"][arm]["pmf_max_difference"], discrepancy, name + ":PMF_disagreement_" + arm, 0.)
    gain = np.sum(target * (np.log(risk["pmfs"]["HIGH"]) - np.log(risk["pmfs"]["LOW"])), axis=1)
    checks.close(risk["paired_gain"], gain, name + ":direct_paired_gain", 0.)
    checks.close(gain, KL["LOW"] - KL["HIGH"], name + ":paired_KL_identity")
    discrepancy = float(np.max(abs(gain - gain[1])))
    checks.exact(risk["LOW_HIGH_risk_qualified"], qualified["LOW"] and qualified["HIGH"] and discrepancy <= 2e-6, name + ":main_risk_qualification")


def check_policies(row, checks):
    name = row["unit"]; A, B = int(row["A"]), int(row["B"]); L, H = row["risk_KL"]["LOW"], row["risk_KL"]["HIGH"]
    c = row["cost_seconds"]; gain = row["delta_R"]; saved = row["policy_algebra"]["policies"]
    full = H if A else L; ref = H if B else L
    checks.close(saved["FULL"]["risk"], full, name + ":policy_FULL_risk", 0.)
    checks.close(saved["reference_only"]["risk"], ref, name + ":policy_B_risk", 0.)
    for actual, expected, label in ((full - L, -A * gain, "FULL_low"), (full - H, (1 - A) * gain, "FULL_high"),
                                  (ref - full, (A - B) * gain, "increment")):
        checks.close(actual, expected, name + ":risk_identity_" + label)
    fullcost = c["common_C0"] + c["full_diagnostic"] + c["HIGH" if A else "LOW"]
    refcost = c["common_C0"] + c["reference_diagnostic"] + c["HIGH" if B else "LOW"]
    checks.close(saved["FULL"]["wall_seconds"], fullcost, name + ":FULL_policy_cost", 1e-9)
    checks.close(saved["reference_only"]["wall_seconds"], refcost, name + ":B_policy_cost", 1e-9)
    checks.close(fullcost - (c["common_C0"] + c["HIGH"]), c["full_diagnostic"] - (1 - A) * (c["HIGH"] - c["LOW"]), name + ":FULL_cost_identity", 1e-9)
    checks.close(refcost - (c["common_C0"] + c["HIGH"]), c["reference_diagnostic"] - (1 - B) * (c["HIGH"] - c["LOW"]), name + ":B_cost_identity", 1e-9)


def lock_order(checks):
    times = {}
    previous = None
    for name in ("INPUTS_LOCK", "DIAGNOSIS_LOCK", "ENDPOINTS_LOCK", "EVALUATION_LOCK"):
        lock = dev.read(OUT / (name + ".json")); tick = datetime.fromisoformat(lock["created_utc"])
        times[name] = tick
        if previous is not None: checks.require(tick >= previous, name + ":global_lock_order")
        previous = tick
        for filename in lock["files"]:
            if filename.endswith(".json"):
                created = datetime.fromisoformat(dev.read(ROOT / filename)["created_utc"])
                checks.require(created <= tick, filename + ":saved_before_own_global_lock")
                if name != "INPUTS_LOCK":
                    prior = {"DIAGNOSIS_LOCK":"INPUTS_LOCK", "ENDPOINTS_LOCK":"DIAGNOSIS_LOCK", "EVALUATION_LOCK":"ENDPOINTS_LOCK"}[name]
                    checks.require(created >= times[prior], filename + ":saved_after_preceding_global_lock")


def audit():
    config, hashes = analysis.locked_inputs()
    checks = Checks(); lock_order(checks)
    checks.exact(config["seed"], 202610071601, "development_seed")
    checks.exact(config["formal_units_generated"], 0, "no_formal_units")
    checks.exact(sorted(p.name for p in (OUT / "units").iterdir() if p.is_dir()), sorted(u["id"] for u in dev.units()), "exact_twelve_unit_directories")
    teachers = [restored(OUT / f"teacher_D{i}", checks)["pmfs"] for i in (1, 2, 3)]
    rows = []; unit_receipts = []
    for unit in dev.units():
        name = unit["id"]; folder = OUT / "units" / name
        data, anchor, diag, start, risk = [restored(folder / key, checks) for key in ("inputs", "anchor", "diagnosis", "start", "risk")]
        fits = {arm: restored(folder / arm, checks) for arm in ("LOW", "HIGH", "REF")}
        check_generated(unit, data, config["teachers"][unit["teacher"]], checks)
        check_initialization(data, anchor, checks, name); check_newton_trace(anchor, checks, name)
        check_kick(data, anchor, start, checks, name); check_diagnosis(anchor, diag, checks, name)
        arms = {arm: check_arm(data, start, fits[arm], checks, name + ":" + arm) for arm in ("LOW", "HIGH")}
        check_reference(start, fits["REF"], checks, name + ":REF")
        check_risk(data, teachers[unit["teacher"]], risk, checks, name)
        row = analysis.unit_row(unit, data, anchor, diag, start, fits, risk); check_policies(row, checks); rows.append(row)
        unit_receipts.append(dict(unit=name, finite_arms=arms, all_recorded_anchor_calls_reweighted=True,
            independent_diagnostic_decision=True, risk_PMFS_reintegrated=False, reference_optimizer_rerun=False))
    summary_checked = (OUT / "SUMMARY.json").is_file()
    if summary_checked:
        summary = analysis.read(OUT / "SUMMARY.json")
        checks.exact(summary["rows"], rows, "published_analysis_unit_rows")
        checks.exact(summary["source_sha256"], analysis.sha(analysis.__file__), "analysis_source_binding")
        hashes[analysis.rel(OUT / "SUMMARY.json")] = analysis.sha(OUT / "SUMMARY.json")
    _, after = analysis.locked_inputs()
    checks.require(all(after[key] == value for key, value in hashes.items() if key in after), "inputs_unchanged_during_audit")
    return dict(schema="prospective_dev_audit_v1", passed=not checks.failures, assertions=checks.count,
        compared_array_elements=checks.elements, failures=checks.failures, maxima=checks.max_errors,
        source_sha256=analysis.sha(__file__), analyze_source_sha256=analysis.sha(analysis.__file__), driver_sha256=analysis.sha(dev.__file__),
        freeze_sha256=analysis.sha(OUT / "FREEZE.json"), evaluation_lock_sha256=analysis.sha(OUT / "EVALUATION_LOCK.json"),
        input_hashes=hashes, unit_receipts=unit_receipts, unit_count=12, analysis_summary_checked=summary_checked,
        scope=dict(saved_Z_uniform_labels_rebuilt=True, saved_sign_recoding_kicks_rebuilt=True,
            all_saved_Adam_updates_replayed=True, all_saved_anchor_call_aggregates_reweighted=True,
            independent_finite_gradient_steps_zero_based=SELECTED_FINITE_GRADIENT_STEPS,
            all_training_gradients_independently_recomputed=False, all_integrals_independently_recomputed=False,
            RNG_streams_regenerated=False, original_in_memory_aliasing_before_persistence_directly_observable=False,
            saved_snapshot_arithmetic_consistency_checked=True, teacher_PMFS_recomputed=False,
            saved_eigenbasis_verified_not_reselected_when_degenerate=True,
            optimization_restarted=False, external_scientific_review=False))


def main(mode):
    output = OUT / "AUDIT.json"
    if mode == "run" and output.exists(): raise FileExistsError("AUDIT already exists")
    # The final global lock is checked before any real audit output is written.
    analysis.locked_inputs(); started = time.perf_counter()
    try: record = audit()
    except BaseException as error:
        if mode == "run":
            analysis.save(OUT / "AUDIT_EXCEPTION.json", dict(created_utc=analysis.now(), error=str(error), traceback=traceback.format_exc(),
                source_sha256=analysis.sha(__file__), audit_completed=False, no_retraining=True))
        raise
    if mode == "check":
        saved = analysis.read(output)
        assert {k:v for k,v in saved.items() if k not in ("created_utc", "audit_seconds")} == safe(record)
    else:
        record.update(created_utc=analysis.now(), audit_seconds=time.perf_counter() - started)
        analysis.save(output, safe(record))
        scope = "# Development execution-audit scope\n\n" + json.dumps(safe(record["scope"]), ensure_ascii=False, indent=2) + "\n\n"
        scope += f"Assertions: {record['assertions']}; failures: {len(record['failures'])}; fixed units: 12.\n"
        scope += "This is an internal numerical/execution check, not external scientific review or a complete independent optimization/integration replay.\n"
        analysis.text_save(OUT / "AUDIT_SCOPE.md", scope)
    print(json.dumps(dict(passed=record["passed"], assertions=record["assertions"], failures=len(record["failures"]), units=12)))
    if not record["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True); modes.add_argument("--run", action="store_true"); modes.add_argument("--check", action="store_true")
    args = parser.parse_args(); main("run" if args.run else "check")
