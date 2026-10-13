"""Independent execution audit of four frozen training streams, not peer review.

All Adam updates are replayed from saved gradients. Only the prespecified five
training steps per arm are independently differentiated; this is not a complete
gradient replay or retraining. Low-dimensional reference checks and five fixed
evaluation banks use NumPy Hermite rules and direct log-Bernoulli formulas,
without calling the frozen driver objective or core kernel.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import ast
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
AUDIT = OUT / "TRAINING_AUDIT.json"
DRIVER = ROOT / "script/joint_dependence_train.py"
ARMS = ("K8", "K64", "GH")
NAMES = (*ARMS, "REF", "IND")
Y = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.float64)
PHI = np.arange(-7, 8, 2, dtype=np.float64) / 10
STEPS = [0, 299, 599, 899, 1199]
VALUE_TOL = 2e-10
GRAD_TOL = 2e-10
STATE_TOL = 2e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def close(actual, expected, label, tolerance=VALUE_TOL):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert math.isfinite(error) and error <= tolerance, (label, error, tolerance)
    return error


@lru_cache(maxsize=16)
def quadrature(order, angle):
    x, w = hermegauss(order)
    w = w / math.sqrt(2 * math.pi)
    i = np.arange(order ** 2)
    nodes = np.column_stack((x[i // order], x[i % order]))
    weights = w[i // order] * w[i % order]
    if angle:
        c, s = math.cos(angle), math.sin(angle)
        nodes = nodes @ np.array([[c, -s], [s, c]]).T
    assert abs(weights.sum() - 1) < 2e-12
    return nodes, np.log(weights)


def independent_gh(theta, target, order, angle=0.0):
    b, W = np.asarray(theta)[:8], np.asarray(theta)[8:].reshape(8, 2)
    nodes, logweight = quadrature(order, angle)
    eta = b + nodes @ W.T
    logyes, logno = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
    sigmoid = expit(eta)
    pmf = np.empty(256)
    loss, gb, gw = 0.0, np.zeros(8), np.zeros((8, 2))
    for lo in range(0, 256, 16):
        y = Y[lo:lo + 16]
        # Directly select each of the eight Bernoulli log factors; no likelihood
        # mass normalization or source-kernel matrix rearrangement is used.
        logs = np.where(y[None, :, :] == 1, logyes[:, None, :], logno[:, None, :]).sum(axis=2)
        joint = logs + logweight[:, None]
        logmass = logsumexp(joint, axis=0)
        pmf[lo:lo + 16] = np.exp(logmass)
        loss -= float(target[lo:lo + 16] @ logmass)
        posterior = np.exp(joint - logmass) * target[None, lo:lo + 16]
        residue = np.sum(posterior[:, :, None] * (sigmoid[:, None, :] - y[None, :, :]), axis=1)
        gb += residue.sum(axis=0)
        gw += residue.T @ nodes
    return loss, np.concatenate((gb, gw.ravel())), pmf


def independent_mc(theta, target, banks, gradient=True):
    b, W = np.asarray(theta)[:8], np.asarray(theta)[8:].reshape(8, 2)
    eta = b + banks @ W.T
    yes, no = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
    sig = expit(eta)
    logs_by_pattern = np.empty((len(banks), 256))
    gb, gw = np.zeros(8), np.zeros((8, 2))
    for lo in range(0, 256, 16):
        y = Y[lo:lo + 16]
        loglike = np.where(y[None, None, :, :] == 1, yes[:, :, None, :], no[:, :, None, :]).sum(axis=3)
        normalizer = logsumexp(loglike, axis=1)
        logs_by_pattern[:, lo:lo + 16] = normalizer - math.log(banks.shape[1])
        if gradient:
            posterior = np.exp(loglike - normalizer[:, None, :]) * target[None, None, lo:lo + 16]
            residue = np.sum(posterior[:, :, :, None] * (sig[:, :, None, :] - y[None, None, :, :]), axis=2)
            gb += residue.sum(axis=(0, 1))
            gw += np.einsum("bkm,bkq->mq", residue, banks, optimize=True)
    losses = -logs_by_pattern @ target
    return losses, np.concatenate((gb, gw.ravel())) / len(banks), logs_by_pattern


def source_interface_check():
    tree = ast.parse(DRIVER.read_text("utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("train_joint", "fit_reference"):
        func = functions[name]
        parameters = [arg.arg for arg in func.args.args + func.args.kwonlyargs]
        assert not any("teacher" in key for key in parameters)
        assert not any("teacher" in node.id for node in ast.walk(func) if isinstance(node, ast.Name))
    run = functions["run"]
    lock_line = next(node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "save" and node.args
                     and isinstance(node.args[0], ast.Name) and node.args[0].id == "lock")
    teacher_line = next(node.lineno for node in ast.walk(run) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "teacher_variants" for target in node.targets))
    assert lock_line < teacher_line
    return dict(learner_signatures_do_not_accept_teacher=True, learner_bodies_do_not_reference_teacher=True,
                lock_save_line=lock_line, teacher_probability_evaluation_line=teacher_line,
                source_call_order_supports_locked_evaluation=True,
                data_generation_teacher_use_is_not_training_or_selection_input=True)


def geometry_check(theta, pmf, teacher, record):
    b, W = theta[:8], theta[8:].reshape(8, 2)
    lengths = np.sqrt(np.sum(W * W, axis=1))
    gram = W @ W.T
    denom = lengths[:, None] * lengths[None, :]
    normalized = np.full((8, 8), np.nan)
    np.divide(gram, denom, out=normalized, where=denom > 0)
    spin = 2 * Y - 1
    moments = np.einsum("yi,yj,y->ij", spin, spin, pmf, optimize=True)
    truth = np.einsum("yi,yj,y->ij", spin, spin, teacher, optimize=True)
    pair_error = (moments - truth)[np.triu_indices(8, 1)]
    for actual, key in ((b, "bias"), (lengths, "row_norms"), (gram, "loading_gram"),
                        (moments, "pair_spin_moments"), (pair_error, "pair_moment_error"),
                        (np.max(np.abs(pair_error)), "pair_moment_max_error"),
                        (np.sqrt(np.mean(pair_error ** 2)), "pair_moment_rmse")):
        close(actual, record[key], "geometry " + key)
    stored = np.asarray(record["normalized_gram"], dtype=np.float64)
    assert np.array_equal(np.isnan(stored), np.isnan(normalized))
    if np.isfinite(normalized).any():
        close(normalized[np.isfinite(normalized)], stored[np.isfinite(stored)], "normalized Gram")
    assert record["normalized_gram_undefined_rows"] == np.flatnonzero(lengths == 0).tolist()
    assert record["coordinate_W_error_used_as_risk"] is False


def scalar_summary(values):
    mean = float(np.mean(values))
    sd = float(np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1)))
    return dict(mean=mean, sample_sd=sd, MCSE=sd / math.sqrt(len(values)), banks=len(values))


def check_qualifications(theta, target, teacher_variants, endpoint, config, baseline_p=None):
    main = config["endpoint_orders"].index(config["main_reference_order"])
    teacher = teacher_variants[main]
    if baseline_p is None:
        specifications = [(order, 0.0) for order in config["endpoint_orders"]] + [(config["rotated_order"], config["rotation"])]
        replay = [independent_gh(theta, target, order, angle) for order, angle in specifications]
    else:
        specifications = [(0, 0.0)]
        logpmf = (Y * np.log(baseline_p) + (1 - Y) * np.log1p(-baseline_p)).sum(axis=1)
        grad = np.concatenate((baseline_p - target @ Y, np.zeros(16)))
        replay = [(float(-target @ logpmf), grad, np.exp(logpmf))]
        main = 0
    assert len(endpoint["variants"]) == len(replay)
    loss_error = grad_error = pmf_error = risk_error = 0.0
    risks = []
    for (order, angle), (loss, gradient, pmf), old in zip(specifications, replay, endpoint["variants"]):
        assert old["order"] == order and old["rotation"] == angle
        loss_error = max(loss_error, close(loss, old["empirical_NLL"], "terminal loss"))
        grad_error = max(grad_error, close(gradient, old["gradient"], "terminal gradient", GRAD_TOL))
        pmf_error = max(pmf_error, close(pmf, old["pmf"], "terminal PMF"))
        risk = float(teacher @ (np.log(teacher) - np.log(pmf)))
        risk_error = max(risk_error, close(risk, old["risk_KL"], "terminal risk"))
        risks.append(risk)
    main_loss, main_grad, main_pmf = replay[main]
    mass = max(abs(pmf.sum() - 1) for _, _, pmf in replay)
    pd = max(float(np.max(np.abs(pmf - main_pmf))) for _, _, pmf in replay)
    rd = max(abs(value - risks[main]) for value in risks)
    td = max(abs(float(p @ (np.log(p) - np.log(main_pmf))) - risks[main]) for p in teacher_variants)
    ld = max(abs(loss - main_loss) for loss, _, _ in replay)
    gd = max(float(np.max(np.abs(gradient - main_grad))) for _, gradient, _ in replay)
    for value, key in ((mass, "mass_max_error"), (pd, "pmf_max_discrepancy"), (max(rd, td), "risk_max_discrepancy"),
                       (ld, "NLL_max_discrepancy"), (gd, "gradient_max_discrepancy"),
                       (main_loss, "empirical_NLL"), (risks[main], "risk_KL")):
        close(float(value), endpoint[key], "terminal qualification " + key)
    close(main_grad, endpoint["gradient"], "terminal main gradient", GRAD_TOL)
    close(main_pmf, endpoint["pmf"], "terminal main PMF")
    qualifies = bool(mass <= config["mass_tolerance"] and pd <= config["pmf_tolerance"] and max(rd, td) <= config["risk_tolerance"]
                     and ld <= config["path_loss_tolerance"] and gd <= config["path_gradient_tolerance"])
    assert qualifies == endpoint["qualified"]
    geometry_check(theta, np.asarray(endpoint["pmf"]), teacher, endpoint["geometry"])
    return dict(loss_error=loss_error, gradient_error=grad_error, pmf_error=pmf_error,
                risk_error=risk_error, recomputed_qualified=qualifies, recorded_qualified=endpoint["qualified"])


def expected_resources(config, active_patterns, ref_calls, path_calls):
    steps, priors, n, B = config["steps"], config["step_prior"], config["observations"], config["evaluation_banks"]
    node_sum = sum(order ** 2 for order in config["endpoint_orders"]) + config["rotated_order"] ** 2
    return dict(new_observations=n, data_Gaussian_scalars=n * 2, data_uniform_scalars=n * 8,
        initialization_Gaussian_scalars=16, training_prior_vectors=steps * priors, training_Gaussian_scalars=steps * priors * 2,
        independent_evaluation_prior_vectors=B * 64, evaluation_Gaussian_scalars=B * 64 * 2,
        Adam_updates=3 * steps, Adam_objective_gradient_calls={name: steps for name in ARMS},
        GH_Adam_nodes=steps * config["gh_order"] ** 2,
        GH_path_nodes=path_calls * (config["gh_order"] ** 2 + config["path_reference_order"] ** 2),
        REF_objective_gradient_calls=ref_calls, REF_nodes=ref_calls * config["ref_order"] ** 2,
        terminal_GH_PMF_nodes=5 * node_sum, terminal_GH_loss_gradient_nodes=4 * node_sum,
        MC_evaluation_bank_candidate_K_target_cells=B * 5 * 2 * 2, MC_computed_nonindependent_cells=B * 4 * 2 * 2,
        MC_evaluation_particle_pattern_products=B * 4 * (8 + 64) * (active_patterns + 256),
        training_MC_particle_pattern_products=steps * 2 * priors * active_patterns,
        training_GH_node_pattern_products=steps * config["gh_order"] ** 2 * active_patterns,
        active_empirical_patterns=active_patterns, observed_patterns=256, independent_training_streams=1,
        reference_extra_cost_separate=True)


def audit_stream(rep, attempt, frozen, freeze_hash, teacher_replays, erratum):
    config = frozen["config"]
    folder = OUT / f"rep_{rep:02}"
    if attempt != 1:
        folder /= f"attempt_{attempt:03}"
    result = read(folder / "RESULTS.json")
    assert result["status"] == "completed" and result["rep"] == rep and result["attempt"] == attempt
    files = {rel(path): dict(sha256=sha(path), bytes=path.stat().st_size) for path in folder.iterdir() if path.is_file()}
    for name, evidence in result["artifacts"].items():
        assert Path(evidence["path"]).name == name
        assert files[evidence["path"]] == {key: evidence[key] for key in ("sha256", "bytes")}
    params, lock = read(folder / "PARAMS.json"), read(folder / "CANDIDATES_LOCK.json")
    endpoint_qualification = read(folder / "ENDPOINT_QUALIFICATION.json")
    inputs, tape = arrays(folder / "INPUTS.npz"), arrays(folder / "TRAINING.npz")
    assert result["params_sha256"] == lock["params_sha256"] == sha(folder / "PARAMS.json")
    assert result["candidates_lock_sha256"] == endpoint_qualification["candidates_lock_sha256"] == sha(folder / "CANDIDATES_LOCK.json")
    assert result["freeze_sha256"] == params["freeze_sha256"] == freeze_hash
    assert params["seed"] == [config["seed"], rep]
    assert params["roles"] == {"latent": 0, "uniform": 1, "initialization": 2, "training_prior": 3, "evaluation_prior": 4}
    assert params["source_sha256"] == frozen["sources"][rel(DRIVER)] and params["config"] == config
    assert params["kernel_sha256"] == frozen["sources"]["script/joint_dependence_kernel.py"]
    assert params["design_sha256"] == frozen["sources"]["docs/08_research_研究工作/JointDependence_v1_20261007/DESIGN.md"]
    assert params["inputs_sha256"] == sha(ROOT / params["inputs"]) == sha(folder / "INPUTS.npz")
    labels, prior = inputs["labels"], inputs["training_prior"]
    assert labels.shape == (4096, 8) and np.isin(labels, (0, 1)).all()
    assert inputs["latent"].shape == (4096, 2) and inputs["uniform"].shape == (4096, 8)
    assert np.all((inputs["uniform"] >= 0) & (inputs["uniform"] < 1))
    teacher_W = .8 * np.column_stack((np.cos(PHI), np.sin(PHI)))
    assert np.array_equal(labels, (inputs["uniform"] < expit(inputs["latent"] @ teacher_W.T)).astype(np.uint8))
    codes = np.packbits(labels.astype(np.uint8), axis=1, bitorder="big")[:, 0]
    freq = np.bincount(codes, minlength=256) / len(labels)
    for recorded in (inputs["frequency"], params["frequency"], lock["frequency"], result["frequency"]):
        assert np.array_equal(freq, recorded)
    p = (labels.sum(axis=0) + .5) / (len(labels) + 1)
    raw_initial = np.concatenate((np.log(p) - np.log1p(-p), inputs["initial_loading"].ravel()))
    initial = np.clip(raw_initial, -2, 2)
    assert np.array_equal(inputs["raw_initial"], raw_initial) and np.array_equal(inputs["initial"], initial)
    assert np.array_equal(params["initial"], initial) and np.array_equal(params["raw_initial"], raw_initial)
    assert params["initial_projection_count"] == int(np.sum(initial != raw_initial))
    assert ah(labels) == params["data_sha256"] and ah(prior) == params["training_prior_sha256"]
    assert prior.shape == (1200, 128, 2) and np.array_equal(prior, tape["training_prior"])
    assert tape["theta"].shape == (1201, 3, 24)
    for key in ("gradient_before_update", "proposal", "projection_mask"):
        assert tape[key].shape == (1200, 3, 24)
    assert tape["loss_before_update"].shape == (1200, 3)
    assert all(np.isfinite(value).all() for value in tape.values())
    assert np.array_equal(tape["theta"][0], np.repeat(initial[None, :], 3, axis=0))
    assert np.all(np.abs(tape["theta"]) <= 2)
    first, second, state = np.zeros((3, 24)), np.zeros((3, 24)), tape["theta"][0].copy()
    max_state = max_proposal = max_moment = 0.0
    for index in range(1200):
        gradient = tape["gradient_before_update"][index]
        step = index + 1
        first = config["beta1"] * first + (1 - config["beta1"]) * gradient
        second = config["beta2"] * second + (1 - config["beta2"]) * gradient ** 2
        lr = config["learning_rates"][0 if step <= config["switch_step"] else 1]
        proposal = state - lr * (first / (1 - config["beta1"] ** step)) / (np.sqrt(second / (1 - config["beta2"] ** step)) + config["adam_epsilon"])
        state = np.clip(proposal, -2, 2)
        max_proposal = max(max_proposal, close(proposal, tape["proposal"][index], "Adam proposal", STATE_TOL))
        max_state = max(max_state, close(state, tape["theta"][step], "Adam state", STATE_TOL))
        assert np.array_equal(proposal != state, tape["projection_mask"][index])
        if step % 300 == 0:
            checkpoint = arrays(folder / f"CHECKPOINT_{step:04}.npz")
            for actual, key in ((state, "theta"), (proposal, "proposal"), (first, "adam_first"), (second, "adam_second")):
                max_moment = max(max_moment, close(actual, checkpoint[key], "checkpoint " + key, STATE_TOL))
            assert np.array_equal(checkpoint["gradient_before_update"], gradient)
            assert np.array_equal(checkpoint["loss_before_update"], tape["loss_before_update"][index])
            assert np.array_equal(checkpoint["projection"], tape["projection_mask"][index])
    assert result["training"]["completed_steps"] == 1200
    assert result["training"]["arms"] == list(ARMS) and result["training"]["objective_calls"] == [1200] * 3
    assert np.array_equal(tape["projection_mask"].sum(axis=(0, 2)), result["training"]["projection_counts"])
    assert params["training_readout"] == "LAST" and not params["teacher_used_by_training_function"]
    assert not result["training"]["teacher_passed_to_learner"]
    gradient_checks = []
    for index in STEPS:
        for arm, name in enumerate(ARMS):
            theta = tape["theta"][index, arm]
            if name == "GH":
                loss, gradient, _ = independent_gh(theta, freq, config["gh_order"])
            else:
                K = 8 if name == "K8" else 64
                losses, gradient, _ = independent_mc(theta, freq, prior[index].reshape(128 // K, K, 2))
                loss = float(losses.mean())
            gradient_checks.append(dict(step_index=index, arm=name,
                loss_error=close(loss, tape["loss_before_update"][index, arm], "fixed-step loss"),
                gradient_error=close(gradient, tape["gradient_before_update"][index, arm], "fixed-step gradient", GRAD_TOL)))
    path_checks = []
    assert [row["step"] for row in result["training"]["path_checks"]] == [0, 300, 600, 900, 1200]
    correction_path = folder / "PATH_METADATA_CORRECTED.json"
    assert sha(correction_path) == erratum["correction"][rel(correction_path)]
    correction = read(correction_path)
    assert correction["rep"] == rep and correction["affected_summary_steps"] == [0, 300, 600, 900]
    assert correction["only_field_changed"] == "theta" and correction["original_records_preserved"]
    assert correction["scientific_outputs_or_qualifications_changed"] is False
    for name, digest in correction["sources"].items():
        assert sha(ROOT / name) == digest, ("Corrected path source drift", name)
    assert len(correction["corrected_path_checks"]) == 5
    for original, recorded in zip(result["training"]["path_checks"], correction["corrected_path_checks"]):
        step = recorded["step"]
        assert recorded == read(folder / f"PATH_{step:04}.json")
        assert set(original) == set(recorded)
        changed_keys = [key for key in original if original[key] != recorded[key]]
        assert changed_keys == (["theta"] if step < 1200 else []), ("Unexplained path metadata change", rep, step, changed_keys)
        assert np.array_equal(original["theta"], tape["theta"][-1, 2]), "The retained defect must exactly match the documented LAST alias"
        theta = tape["theta"][step, 2]
        assert np.array_equal(theta, recorded["theta"])
        lo, go, _ = independent_gh(theta, freq, config["gh_order"])
        lr, gr, _ = independent_gh(theta, freq, config["path_reference_order"])
        for a, b, label in ((lo, recorded["loss"], "path loss"), (lr, recorded["reference_loss"], "path reference loss"),
                             (go, recorded["gradient"], "path gradient"), (gr, recorded["reference_gradient"], "path reference gradient")):
            close(a, b, label)
        le, ge = abs(lo - lr), float(np.max(np.abs(go - gr)))
        close(le, recorded["loss_difference"], "path loss discrepancy")
        close(ge, recorded["gradient_max_difference"], "path gradient discrepancy")
        qualified = le <= config["path_loss_tolerance"] and ge <= config["path_gradient_tolerance"]
        assert qualified == recorded["qualified"]
        path_checks.append(dict(step=step, recomputed_qualified=qualified, recorded_qualified=recorded["qualified"],
             loss_discrepancy=le, gradient_discrepancy=ge, original_summary_fields_differing=changed_keys,
             original_summary_theta_is_LAST=True, authoritative_PATH_theta_equals_training_tape=True,
             qualification_and_numeric_result_fields_unchanged=True))
    candidates = lock["candidates"]
    assert candidates == result["candidates"] and [c["name"] for c in candidates] == list(NAMES)
    assert lock["training_sha256"] == sha(folder / "TRAINING.npz")
    assert lock["reference_training_sha256"] == sha(folder / "REFERENCE_TRAINING.npz")
    assert lock["locked_before_teacher_risk_and_independent_cross_evaluation"]
    for index in range(3):
        assert np.array_equal(candidates[index]["theta"], tape["theta"][-1, index])
    for candidate in candidates:
        theta = np.asarray(candidate["theta"])
        assert theta.shape == (24,) and np.isfinite(theta).all()
        assert np.array_equal(candidate["b"], theta[:8]) and np.array_equal(candidate["W"], theta[8:].reshape(8, 2))
    independent_theta = np.concatenate((np.log(p) - np.log1p(-p), np.zeros(16)))
    assert np.array_equal(candidates[4]["p"], p) and np.array_equal(candidates[4]["theta"], independent_theta)
    ref_trace = arrays(folder / "REFERENCE_TRAINING.npz")
    ref = result["reference_optimizer"]
    assert ref == lock["reference_optimizer"]
    calls = ref["objective_calls"]
    assert ref_trace["theta"].shape == ref_trace["gradient"].shape == (calls, 24)
    assert ref_trace["loss"].shape == (calls,) and np.array_equal(ref_trace["theta"][0], tape["theta"][-1, 2])
    assert all(np.isfinite(value).all() for value in ref_trace.values())
    assert np.all(np.abs(ref_trace["theta"]) <= 2)
    assert np.array_equal(ref["theta"], candidates[3]["theta"])
    matches = np.flatnonzero(np.all(ref_trace["theta"] == ref["theta"], axis=1))
    assert len(matches) and any(abs(ref_trace["loss"][i] - ref["loss"]) <= VALUE_TOL and np.max(np.abs(ref_trace["gradient"][i] - ref["gradient"])) <= GRAD_TOL for i in matches)
    assert ref["convergence_unconfirmed"] == (not ref["success"]) and not ref["restarted"] and not ref["global_MLE_claimed"]
    ref_selected = sorted(set([0, calls // 4, calls // 2, (3 * calls) // 4, calls - 1]))
    ref_checks = []
    for index in ref_selected:
        loss, gradient, _ = independent_gh(ref_trace["theta"][index], freq, config["ref_order"])
        ref_checks.append(dict(call_index=index, loss_error=close(loss, ref_trace["loss"][index], "REF trace loss"),
                               gradient_error=close(gradient, ref_trace["gradient"][index], "REF trace gradient", GRAD_TOL)))
    if ref["success"] and ref["iterations"] == 0:
        gradient = np.asarray(ref["gradient"])
        theta = np.asarray(ref["theta"])
        projected = np.where(((theta <= -2) & (gradient > 0)) | ((theta >= 2) & (gradient < 0)), 0, gradient)
        assert np.max(np.abs(projected)) <= config["ref_optimizer"]["gtol"]
    specs = [(order, 0.0) for order in config["endpoint_orders"]] + [(config["rotated_order"], config["rotation"])]
    assert [(row["order"], row["rotation"]) for row in endpoint_qualification["teacher_variants"]] == specs
    for pmf, row in zip(teacher_replays, endpoint_qualification["teacher_variants"]):
        close(pmf, row["pmf"], "locked-stage teacher PMF")
    teacher_main_index = config["endpoint_orders"].index(config["main_reference_order"])
    teacher = np.asarray(endpoint_qualification["teacher_pmf"])
    assert np.array_equal(teacher, endpoint_qualification["teacher_variants"][teacher_main_index]["pmf"])
    close(max(float(np.max(np.abs(p - teacher_replays[teacher_main_index]))) for p in teacher_replays), endpoint_qualification["teacher_pmf_error"], "teacher PMF discrepancy")
    close(max(abs(float(p.sum()) - 1) for p in teacher_replays), endpoint_qualification["teacher_mass_error"], "teacher mass discrepancy")
    assert endpoint_qualification["endpoints"] == result["endpoints"]
    endpoint_checks = []
    for candidate, endpoint in zip(candidates, result["endpoints"]):
        assert endpoint["name"] == candidate["name"]
        endpoint_checks.append(dict(name=candidate["name"], **check_qualifications(np.asarray(candidate["theta"]), freq, teacher_replays, endpoint, config,
            baseline_p=p if candidate["name"] == "IND" else None)))
    assert endpoint_qualification["all_qualified"] == all(row["recomputed_qualified"] for row in endpoint_checks)
    evaluation = arrays(folder / "EVALUATION.npz")
    evaluation_prior = arrays(folder / "EVALUATION_PRIOR.npz")["prior"]
    loss = evaluation["loss"]
    assert loss.shape == (4096, 5, 2, 2) and np.isfinite(loss).all()
    assert evaluation_prior.shape == (4096, 64, 2) and np.array_equal(evaluation["prior"], evaluation_prior)
    assert evaluation["candidate_names"].tolist() == list(NAMES) and evaluation["particles"].tolist() == [8, 64]
    assert evaluation["target_names"].tolist() == ["empirical", "population"]
    assert np.array_equal(evaluation["frequency"], freq) and np.array_equal(evaluation["teacher_pmf"], teacher)
    cross = result["cross_evaluation"]
    assert len(cross["means"]) == 20 and len(cross["paired_differences"]) == 40
    summary_error = 0.0
    for row in cross["means"] + cross["paired_differences"]:
        ki, ti = [8, 64].index(row["K"]), ["empirical", "population"].index(row["target"])
        if "candidate" in row:
            values = loss[:, NAMES.index(row["candidate"]), ki, ti]
        else:
            values = loss[:, NAMES.index(row["left"]), ki, ti] - loss[:, NAMES.index(row["right"]), ki, ti]
        summary = scalar_summary(values)
        for key, value in summary.items():
            summary_error = max(summary_error, close(value, row[key], "paired summary " + key))
        assert row["scope"] == "conditional on locked endpoints; not independent training repeats"
    for ki, K in enumerate([8, 64]):
        for ti, target in enumerate(["empirical", "population"]):
            means = loss[:, :, ki, ti].mean(axis=0)
            ranking = [NAMES[i] for i in sorted(range(5), key=lambda i: (means[i], i))]
            assert cross["rankings"][f"K{K}_{target}"] == ranking
    assert cross["statistical_family_claimed"] is False and cross["training_streams_in_this_record"] == 1
    bank_indices = [0, 4096 // 7, 4096 // 3, 4096 // 2, 4095]
    evaluation_checks = []
    for ci, candidate in enumerate(candidates[:4]):
        for ki, K in enumerate([8, 64]):
            _, _, logmass = independent_mc(np.asarray(candidate["theta"]), freq, evaluation_prior[bank_indices, :K], gradient=False)
            replay = np.column_stack((-logmass @ freq, -logmass @ teacher))
            evaluation_checks.append(dict(candidate=candidate["name"], K=K, bank_indices=bank_indices,
                empirical_max_error=close(replay[:, 0], loss[bank_indices, ci, ki, 0], "fixed evaluation empirical loss"),
                population_max_error=close(replay[:, 1], loss[bank_indices, ci, ki, 1], "fixed evaluation population loss")))
    ind_logq = (Y * np.log(p) + (1 - Y) * np.log1p(-p)).sum(axis=1)
    for ti, target in enumerate((freq, teacher)):
        close(loss[:, 4, :, ti], np.full((4096, 2), -target @ ind_logq), "all IND evaluation cells")
    expected = expected_resources(config, int(np.count_nonzero(freq)), calls, len(path_checks))
    assert result["resources"] == expected, (rep, result["resources"], expected)
    for key, value in result["claims"].items():
        if key in ("exploratory_training_panel", "MCSE_conditions_on_fixed_endpoints"):
            assert value is True
        else:
            assert value is False
    for name, row in files.items():
        assert sha(ROOT / name) == row["sha256"], ("Stream file changed during audit", name)
    return dict(rep=rep, files=files, observed_frequencies_and_Jeffreys_recomputed=True,
        original_generated_labels_checked_from_saved_latent_and_uniform=True,
        Adam=dict(updates_replayed=3600, saved_gradients_used=True, all_training_gradients_recomputed=False,
                  max_state_error=max_state, max_proposal_error=max_proposal, max_checkpoint_moment_error=max_moment,
                  LAST_matches_locked_candidates=True), fixed_step_gradient_checks=gradient_checks,
        path_checks=path_checks, REF=dict(success=ref["success"], iterations=ref["iterations"],
            objective_calls=calls, convergence_unconfirmed=ref["convergence_unconfirmed"],
            no_restart=True, global_MLE_claimed=False, selected_trace_checks=ref_checks),
        endpoint_checks=endpoint_checks, all_endpoint_qualifications_preserved=True,
        paired_summary_max_error=summary_error, paired_means_checked=20, paired_differences_checked=40,
        rankings= cross["rankings"], fixed_evaluation_bank_checks=evaluation_checks,
        geometry_is_observation_or_Gram_based=True, teacher_PMF_used_only_in_locked_evaluation_source_interface=True,
        source_call_order_is_not_a_separate_runtime_information_flow_proof=True, resources_checked=expected,
        path_record_erratum=dict(path=rel(correction_path), sha256=sha(correction_path),
            affected_original_summary_theta_fields=4, original_RESULT_bytes_retained=True,
            correct_original_PATH_and_training_tape_used=True, unexplained_record_differences=0))


def execute(attempt):
    freeze_path = OUT / ("TRAIN_FREEZE.json" if attempt == 1 else f"TRAIN_FREEZE_ATTEMPT_{attempt:03}.json")
    frozen = read(freeze_path)
    assert frozen["attempt"] == attempt
    config = frozen["config"]
    assert (config["reps"], config["steps"], config["observations"], config["dimension"], config["latent"]) == (4, 1200, 4096, 8, 2)
    assert config["particles"] == [8, 64] and config["seed"] == 202610070100
    assert config["bias_bounds"] == config["loading_bounds"] == [-2., 2.]
    for name, digest in frozen["sources"].items():
        assert sha(ROOT / name) == digest, ("Frozen source changed", name)
    snapshot = ROOT / frozen["source_snapshot"]
    for name in ("joint_dependence_train.py", "joint_dependence_kernel.py", "DESIGN.md"):
        original = next(key for key in frozen["sources"] if Path(key).name == name)
        assert sha(snapshot / name) == frozen["sources"][original]
    interface = source_interface_check()
    erratum_path = OUT / "PATH_RECORDS_ERRATUM.json"
    erratum = read(erratum_path)
    assert erratum["source_sha256"] == sha(ROOT / "script/joint_dependence_path_records.py")
    assert erratum["status"] == "corrected_by_original_immediate_records_and_tape"
    assert erratum["affected_streams"] == 4 and erratum["affected_theta_summary_fields"] == 16
    assert len(erratum["correction"]) == 4
    assert all(erratum[key] is False for key in ("loss_gradient_gate_or_training_changed", "new_scientific_execution", "frozen_driver_modified", "original_RESULTS_modified"))
    assert erratum["forward_copy_regression_fixture_passed"]
    teacher_theta = np.concatenate((np.zeros(8), (.8 * np.column_stack((np.cos(PHI), np.sin(PHI)))).ravel()))
    specs = [(order, 0.0) for order in config["endpoint_orders"]] + [(config["rotated_order"], config["rotation"])]
    teacher_replays = [independent_gh(teacher_theta, np.ones(256) / 256, order, angle)[2] for order, angle in specs]
    streams = []
    for rep in range(4):
        streams.append(audit_stream(rep, attempt, frozen, sha(freeze_path), teacher_replays, erratum))
        print(json.dumps(dict(stage="training_audit", rep=rep, passed=True)), flush=True)
    failures = {rel(path): sha(path) for path in OUT.rglob("*FAILURE*.json")}
    history_tests = {rel(path): dict(sha256=sha(path), passed=read(path).get("passed"),
                     source_sha256=read(path).get("source_sha256")) for path in OUT.glob("TRAIN_TEST_ATTEMPT_*.json")}
    for name, digest in frozen["sources"].items():
        assert sha(ROOT / name) == digest
    profile_path = ROOT / "script/joint_dependence_profile.py"
    profile_freeze_path = OUT / "PROFILE_FREEZE.json"
    profile_source = profile_path.read_text("utf-8")
    profile_freeze = read(profile_freeze_path)
    assert profile_freeze["sources"][rel(profile_path)] == sha(profile_path)
    assert profile_source.count('["path_checks"]') == 1
    assert 'all(x["qualified"] for x in r["training"]["path_checks"])' in profile_source
    assert 'construct_profiles(l["candidates"])' in profile_source
    assert 'construct_profiles(source_lock["candidates"])' in profile_source
    return dict(attempt=attempt, freeze_path=rel(freeze_path), freeze_sha256=sha(freeze_path),
        frozen_sources=frozen["sources"], source_interface=interface, streams=streams,
        execution_and_authoritative_corrected_records_consistent=True,
        original_metadata_theta_alias_defect_disclosed=True,
        path_records_erratum=dict(path=rel(erratum_path), sha256=sha(erratum_path), affected_fields=16,
            qualification_flags_and_locked_endpoints_unchanged=True, no_retraining_or_result_rewrite=True),
        profile_impact_source_check=dict(source=rel(profile_path), source_sha256=sha(profile_path),
            profile_freeze_sha256=sha(profile_freeze_path),
            only_path_check_field_consumed_is_qualified=True,
            profile_geometry_comes_from_locked_candidates=True,
            verified_summary_alias_does_not_change_profile_inputs_or_eligibility=True,
            profile_scientific_results_not_audited_here=True),
        old_failure_records_retained=failures,
        historical_manufactured_test_attempts_retained=history_tests,
        scope=dict(training_streams=4, projected_Adam_updates_replayed=14400,
            selected_step_indices=STEPS, training_gradients_independently_recomputed=60,
            all_training_gradients_recomputed=False, full_retraining=False,
            fixed_path_checkpoints=20, locked_endpoints=20, all_evaluation_scalar_cells_read=4 * 4096 * 5 * 2 * 2,
            fixed_joint_evaluation_loss_cells_checked=4 * 5 * 4 * 2 * 2,
            independent_RNG_draws=0, optimizers_run_by_auditor=0,
            observations_regenerated=False, scientific_design_changed=False),
        numerical_qualification_is_rigorous_quadrature_remainder_certificate=False,
        exploratory_training_panel_not_confirmatory=True, external_scientific_review_performed=False,
        theory_or_mechanism_validity_reviewed=False)


def main(check, attempt):
    started = time.perf_counter()
    try:
        validation = execute(attempt)
    except BaseException:
        # The auditor owns this receipt only; retain an unsuccessful attempt
        # within it instead of changing any scientific source or original result.
        if not check:
            prior = read(AUDIT) if AUDIT.exists() else {}
            attempts = prior.get("failed_audit_attempts", [])
            attempts.append(dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
                                 traceback=traceback.format_exc(), attempt=attempt))
            prior.update(status="failed_execution_audit", failed_audit_attempts=attempts)
            AUDIT.write_text(json.dumps(prior, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    if check:
        old = read(AUDIT)
        assert old["status"] == "passed_execution_audit" and old["script_sha256"] == sha(__file__)
        assert old["validation"] == validation
    else:
        old = read(AUDIT) if AUDIT.exists() else {}
        assert old.get("status") != "passed_execution_audit", "Do not overwrite a completed audit"
        record = dict(created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
                      status="passed_execution_audit", wall_seconds=time.perf_counter() - started,
                      failed_audit_attempts=old.get("failed_audit_attempts", []), validation=validation)
        AUDIT.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, checked_existing_receipt=check, seconds=time.perf_counter() - started,
          Adam_updates_replayed=14400, independent_training_gradient_checks=60,
          all_training_gradients_recomputed=False, strict_quadrature_certificate=False), ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    main(args.check, args.attempt)
