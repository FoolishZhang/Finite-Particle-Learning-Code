"""Independent replay of joint-scale learning; no producer evaluator import.

Scientific execution is gated by a separately saved audit protocol.  NumPy
provides the mathematical score, Cholesky directional derivatives and Adam
arithmetic; the original frozen SCBM loss is loaded only for limited API replay.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import time
import traceback

import numpy as np
import torch
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
BASE = ROOT / "artifacts/joint_scale_learning"
TRAIN = BASE / "TRAIN"
OUT = BASE / "AUDIT"
DOC = ROOT / "docs/08_research_研究工作/两组尺度共同学习与恢复边界_20261004"
OFF = ROOT / "reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py"
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
METHODS = ("FIXED", "FREE", "ZERO")
STARTS = ("NEAR", "UNIT")
ARMS = tuple(f"{start}_{method}" for start in STARTS for method in METHODS)
STEPS = (0, 31, 127, 511)
PATTERNS = np.array(list(itertools.product((0, 1), repeat=4)))
TOL = dict(api_loss=6e-5, api_gradient=6e-6, mathematical=3e-10,
           arithmetic=3e-11, one_step=4e-6, moment_abs=3e-6,
           moment_rel=3e-5, repair=2e-12, repair_parameter=3e-7,
           maxlogit=4e-6, reference_pmf=2e-7, reference_KL=2e-6)
_LOSS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(left, right, tolerance=TOL["arithmetic"], message="numeric"):
    left, right = np.asarray(left, float), np.asarray(right, float)
    require(left.shape == right.shape, message + " shape")
    error = float(np.max(abs(left - right))) if left.size else 0.
    require(np.isfinite(left).all() and np.isfinite(right).all()
            and np.all(abs(left - right) <= tolerance), message + f" max error {error}")
    return error


def now():
    return datetime.now(timezone.utc).isoformat()


def correlation(rho):
    value = np.eye(4)
    for r, (i, j) in zip(rho, PAIRS):
        value[i, j] = value[j, i] = float(r)
    return value


def covariance(vector):
    vector = np.asarray(vector, float)
    return vector[4:8, None] * correlation(vector[8:14]) * vector[None, 4:8]


def repair(proposal, dtype=np.float32):
    value = np.asarray(proposal, dtype=dtype).copy()
    value[:4] = np.clip(value[:4], -2., 2.)
    value[4:8] = np.clip(value[4:8], .001, 2.)
    value[14] = np.clip(value[14], 0., 2.)
    before = float(np.linalg.eigvalsh(correlation(value[8:14]))[0])
    alpha = (1. - .1001) / (1. - before) if before < .1 else 1.
    value[8:14] = (np.asarray(value[8:14], float) * alpha).astype(dtype)
    after = float(np.linalg.eigvalsh(correlation(value[8:14]))[0])
    require(after >= .1, "independent repair feasibility")
    return value, alpha, before, after


def official(K):
    if K not in _LOSS:
        spec = importlib.util.spec_from_file_location("jointscale_independent_original", OFF)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOSS[K] = module.SCBLoss(alpha=1., config=SimpleNamespace(
            reg_precision="l1", reg_weight=1., num_monte_carlo=K, training_mode="joint"))
    return _LOSS[K]


def native(method, vector, active, noise, labels, noise_labels):
    """Independent physical transform and original float32 API, one state."""
    v = np.asarray(vector, float)
    mu = torch.tensor(v[:4], dtype=torch.float32, requires_grad=True)
    scales = torch.tensor(v[4:8], dtype=torch.float32, requires_grad=True)
    rho = torch.tensor(v[8:14], dtype=torch.float32, requires_grad=True)
    tau = torch.tensor(v[14], dtype=torch.float32, requires_grad=method == "FREE")
    rows = []
    for i in range(4):
        rows.append(torch.stack([torch.tensor(1., dtype=torch.float32) if i == j
            else rho[PAIRS.index(tuple(sorted((i, j))))] for j in range(4)]))
    lower = torch.linalg.cholesky(torch.stack(rows))
    e = torch.tensor(active, dtype=torch.float32)
    x = (lower @ e.unsqueeze(-1)).squeeze(-1) * scales + mu
    target = labels
    if method != "ZERO":
        x = torch.cat((x, torch.tensor(noise, dtype=torch.float32) * tau), dim=-1)
        target = np.concatenate((labels, noise_labels), axis=1)
    probability = torch.sigmoid(x.movedim(0, -1))
    loss = official(len(active)).compute_concept_loss(probability, torch.tensor(target, dtype=torch.int64))
    if method == "ZERO":
        loss = loss + noise.shape[-1] * math.log(2.)
    pars = (mu, scales, rho, tau) if method == "FREE" else (mu, scales, rho)
    values = torch.autograd.grad(loss, pars)
    gradient = np.r_[values[0].detach().numpy(), values[1].detach().numpy(),
                     values[2].detach().numpy(), float(values[3]) if method == "FREE" else 0.]
    return dict(loss=float(loss.detach()), gradient=gradient,
                sigmoid_zero_count=int((probability.detach() == 0).sum()),
                sigmoid_one_count=int((probability.detach() == 1).sum()),
                probability_evaluations=probability.numel(),
                maximum_absolute_logit=float(x.detach().abs().max()))


def mathematical(method, vector, active, noise, labels, noise_labels):
    """Independent NumPy score and analytic Cholesky path derivatives."""
    v = np.asarray(vector, float)
    mu, scales, rho, tau = v[:4], v[4:8], v[8:14], v[14]
    lower = np.linalg.cholesky(correlation(rho))
    z = active @ lower.T
    x = mu + z * scales
    sign = (2. * labels - 1.)[None]
    signed = x * sign
    logs = -np.logaddexp(0., -signed).sum(axis=-1)
    sn = None
    if method != "ZERO":
        sn = noise * (2. * noise_labels - 1.)[None]
        logs += -np.logaddexp(0., -sn * tau).sum(axis=-1)
    norm = logsumexp(logs, axis=0)
    weights = np.exp(logs - norm[None])
    loss = float(np.mean(math.log(len(active)) - norm))
    if method == "ZERO":
        loss += noise.shape[-1] * math.log(2.)
    residual = -weights[:, :, None] * sign * expit(-signed) / len(labels)
    gradient = np.zeros(15)
    gradient[:4] = residual.sum(axis=(0, 1))
    gradient[4:8] = (residual * z).sum(axis=(0, 1))
    for q, (i, j) in enumerate(PAIRS):
        change = np.zeros((4, 4)); change[i, j] = change[j, i] = 1.
        middle = np.linalg.solve(lower, np.linalg.solve(lower, change).T).T
        phi = np.tril(middle); phi[np.diag_indices(4)] *= .5
        dx = active @ (lower @ phi).T
        gradient[8 + q] = np.sum(residual * scales * dx)
    if method == "FREE":
        gradient[14] = -np.sum(weights[:, :, None] * sn * expit(-sn * tau)) / len(labels)
    return dict(loss=loss, gradient=gradient)


def adam(moment, square, gradient, step, dtype=np.float32):
    """Elementwise arithmetic; no call to the production or torch optimizer."""
    g = np.asarray(gradient, dtype=dtype)
    m = (moment + dtype(.1) * (g - moment)).astype(dtype)
    v = (square * dtype(.999) + dtype(.001) * g * g).astype(dtype)
    denominator = (np.sqrt(v) / dtype(math.sqrt(1. - .999 ** step)) + dtype(1e-8)).astype(dtype)
    delta = (dtype(-.02 / (1. - .9 ** step)) * m / denominator).astype(dtype)
    return m, v, delta


def check():
    """Deterministic qualification; no science inputs, fits or model integrals."""
    active = np.sin(np.arange(100 * 3 * 4).reshape(100, 3, 4) * .13)
    noise = np.cos(np.arange(100 * 3 * 5).reshape(100, 3, 5) * .017)
    labels = (np.arange(12).reshape(3, 4) % 3 == 0).astype(np.int8)
    yn = (np.arange(15).reshape(3, 5) % 5 < 2).astype(np.int8)
    v = np.array([-.3, -.1, .2, .4, .4, .7, .9, 1.1, .11, -.07, .05, .09, -.06, .13, .8])
    fd_errors, native_errors = [], []
    for method in METHODS:
        vv = v.copy()
        if method == "FIXED": vv[14] = 1.
        if method == "ZERO": vv[14] = 0.
        value = mathematical(method, vv, active, noise, labels, yn)
        for q in range(15 if method == "FREE" else 14):
            h = 1e-5; plus, minus = vv.copy(), vv.copy(); plus[q] += h; minus[q] -= h
            numeric = (mathematical(method, plus, active, noise, labels, yn)["loss"]
                       - mathematical(method, minus, active, noise, labels, yn)["loss"]) / (2. * h)
            fd_errors.append(near(numeric, value["gradient"][q], 3e-8, "deterministic FD"))
        api = native(method, vv, active, noise, labels, yn)
        native_errors.append(near(api["loss"], value["loss"], 1e-4, "deterministic API value"))
        near(api["gradient"], value["gradient"], 5e-5, "deterministic API gradient")
    parameter = torch.nn.Parameter(torch.zeros(15, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=.02, betas=(.9, .999), eps=1e-8,
                                 foreach=False, fused=False)
    m = np.zeros(15, np.float32); sq = m.copy(); maximum = 0.
    for step in range(512):
        g = (.05 * np.sin((np.arange(15) + 1) * (step + 1) * .013)).astype(np.float32)
        before = parameter.detach().numpy().copy()
        parameter.grad = torch.tensor(g); optimizer.step(); m, sq, delta = adam(m, sq, g, step + 1)
        maximum = max(maximum, near(before + delta, parameter.detach().numpy(),
                                   TOL["one_step"], "deterministic Adam"))
    for value, key in ((m, "exp_avg"), (sq, "exp_avg_sq")):
        target = optimizer.state[parameter][key].numpy()
        near(value, target, TOL["moment_abs"] + TOL["moment_rel"] * abs(target), "deterministic moments")
    for signs in itertools.product((-1., 1.), repeat=6):
        proposed = np.r_[[-3., -1., 1., 3.], [-.1, .01, 1., 3.], .95 * np.array(signs), -.1]
        out, alpha, before, after = repair(proposed)
        require(after >= .1 and before < .1 and 0. < alpha < 1., "deterministic repair")
        near(out[:4], [-2., -1., 1., 2.], 0., "mean clipping")
        require(out[14] == 0., "tau clipping")
    full_dimension = shared_scale_numerical_diagnostic()
    return dict(passed=True, structural_and_independent_formula_checks_passed=True,
                source_sha256=sha(SOURCE), maximum_FD_error=max(fd_errors),
                maximum_native_value_error=max(native_errors), Adam_one_step_error=maximum,
                deterministic_repair_vertices=64, new_scientific_RNG=0,
                new_model_integrals=0, new_scientific_fits=0,
                full_dimension_native64_qualification=full_dimension,
                qualification_scope="passed refers to independent formula/update checks; the explicitly retained full-dimensional 4e-5 native/math gradient gate is separate")


def shared_scale_numerical_diagnostic():
    """Retain the pre-freeze B7/n256 absolute gradient qualification failure."""
    active = 1.1 * np.sin(np.arange(100 * 7 * 4, dtype=float) * .113).reshape(100, 7, 4)
    noise = .9 * np.cos(np.arange(100 * 7 * 256, dtype=float) * .017).reshape(100, 7, 256)
    yi = np.arange(7 * 4).reshape(7, 4)
    ni = np.arange(7 * 256).reshape(7, 256)
    labels = ((3 * yi + yi // 4) % 5 < 2).astype(np.int8)
    yn = ((7 * ni + ni // 256) % 11 < 5).astype(np.int8)
    vector = np.r_[[-.35, -.1, .1, .35], np.full(4, .1), np.zeros(6), 1.].astype(np.float32).astype(float)
    mat = mathematical("FREE", vector, active, noise, labels, yn)
    api = native("FREE", vector, active, noise, labels, yn)
    finite = []
    for step in (1e-3, 1e-4, 1e-5):
        plus, minus = vector.copy(), vector.copy(); plus[-1] += step; minus[-1] -= step
        fd = (mathematical("FREE", plus, active, noise, labels, yn)["loss"]
              - mathematical("FREE", minus, active, noise, labels, yn)["loss"]) / (2. * step)
        finite.append(dict(step=step, finite_difference=fd, difference=fd - mat["gradient"][-1]))
    require(abs(finite[-1]["difference"]) < 3e-8, "full-dimensional independent tau derivative")
    difference = float(api["gradient"][-1] - mat["gradient"][-1])
    return dict(math64_tau=float(mat["gradient"][-1]), native32_tau=float(api["gradient"][-1]),
                tau_difference=difference, tau_relative_error=abs(difference) / abs(float(mat["gradient"][-1])),
                other_coordinate_max_error=float(np.max(abs(api["gradient"][:-1] - mat["gradient"][:-1]))),
                loss_difference=float(api["loss"] - mat["loss"]), finite_differences=finite,
                original_absolute_gradient_tolerance=4e-5, passed=bool(abs(difference) <= 4e-5),
                scope="No implementation error established; original absolute native/math gate retained, no rounding-operation attribution or whole-trajectory equivalence")


def quadrature44(mu, sigma):
    """Separate four-dimensional GH implementation with symmetric covariance root."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    require(np.all(eigenvalues > 0.), "independent covariance positive definite")
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    nodes, weights = np.polynomial.hermite.hermgauss(44)
    nodes *= math.sqrt(2.); weights /= math.sqrt(math.pi)
    total = 44 ** 4; pmf = np.zeros(16)
    for begin in range(0, total, 65536):
        index = np.arange(begin, min(begin + 65536, total))
        digit = np.array(np.unravel_index(index, (44, 44, 44, 44))).T
        gaussian = nodes[digit]
        weight = np.prod(weights[digit], axis=1)
        logits = gaussian @ root.T + mu
        probability = expit(logits)
        for j, pattern in enumerate(PATTERNS):
            pmf[j] += np.sum(weight * np.prod(np.where(pattern, probability, 1. - probability), axis=1))
    mass = float(pmf.sum())
    return dict(raw_pmf=pmf.tolist(), raw_mass=mass, pmf=(pmf / mass).tolist(), order=44, nodes=total,
                minimum_covariance_eigenvalue=float(eigenvalues[0]),
                covariance_root="symmetric eigendecomposition", normalization_error=abs(mass - 1.),
                normalization_applied=True)


def risk(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    require(p.shape == q.shape == (16,) and (p > 0.).all() and (q > 0.).all(), "risk PMF support")
    return math.fsum(float(a) * math.log(float(a / b)) for a, b in zip(p, q))


def product_reference(p):
    p = np.asarray(p, float)
    marginal = p @ PATTERNS
    q = np.prod(np.where(PATTERNS, marginal, 1. - marginal), axis=1)
    return dict(marginals=marginal.tolist(), pmf=q.tolist(), reference_joint_KL=risk(p, q))


def family(risks, product_risk):
    """Six predetermined paired contrasts; no new statistical choices."""
    risks = np.asarray(risks, float)
    require(risks.shape == (16, 6), "primary risk matrix shape")
    critical = float(student_t.ppf(1. - .05 / 12., 15))
    rows = []
    for start_index, start in enumerate(STARTS):
        fixed, free, zero = risks[:, 3 * start_index:3 * start_index + 3].T
        for name, values in (("FREE_minus_FIXED", free - fixed),
                             ("FREE_minus_PRODUCT", free - product_risk),
                             ("ZERO_minus_PRODUCT", zero - product_risk)):
            mean = math.fsum(values) / 16
            se = math.sqrt(math.fsum((values - mean) ** 2) / 15 / 16)
            lower, upper = mean - critical * se, mean + critical * se
            rows.append(dict(start=start, name=name, stream_values=values.tolist(),
                             mean=mean, SE=se, lower=lower, upper=upper,
                             effect_passed=upper < 0., precision_passed=se <= .0002))
    return rows


def trajectory(tape, initial_mu):
    """Check every saved gradient-driven update without treating it as fresh training."""
    parameters = np.asarray(tape["params"])
    gradient = np.asarray(tape["gradient"])
    proposals = np.asarray(tape["proposal"])
    mm, vv = np.asarray(tape["adam_m"]), np.asarray(tape["adam_v"])
    require(parameters.shape == mm.shape == vv.shape == (6, 513, 15), "state/moment dimensions")
    require(gradient.shape == proposals.shape == (6, 512, 15), "gradient/proposal dimensions")
    require(parameters.dtype == np.float32 and gradient.dtype == np.float32, "main float32 dtype")
    require(all(np.isfinite(x).all() for x in (parameters, gradient, proposals, mm, vv)), "finite tapes")
    require(np.array_equal(tape["batch_start"], np.arange(512) * 64 % 4096), "cyclic batch order")
    require(np.count_nonzero(mm[:, 0]) == np.count_nonzero(vv[:, 0]) == 0, "zero initial Adam moments")
    maxima = dict(proposal=0., moment=0., variance=0., repair=0., eigenvalue=0.)
    tails = []
    for arm, name in enumerate(ARMS):
        start, method = name.split("_")
        initial = np.r_[np.asarray(initial_mu, np.float32),
                        np.full(4, .1 if start == "NEAR" else 1., np.float32),
                        np.zeros(6, np.float32), 0. if method == "ZERO" else 1.].astype(np.float32)
        near(parameters[arm, 0], initial, 0., "all-arm initialization")
        m = np.zeros(15, np.float32); sq = m.copy()
        if method != "FREE":
            require(np.count_nonzero(gradient[arm, :, 14]) == 0
                    and np.count_nonzero(mm[arm, :, 14]) == 0
                    and np.count_nonzero(vv[arm, :, 14]) == 0, "fixed tau has no Adam state")
            require(np.all(parameters[arm, :, 14] == initial[14]), "fixed tau invariant")
        for step in range(512):
            m, sq, delta = adam(m, sq, gradient[arm, step], step + 1)
            maxima["moment"] = max(maxima["moment"], near(m, mm[arm, step + 1],
                TOL["moment_abs"] + TOL["moment_rel"] * abs(mm[arm, step + 1]), "all Adam first moments"))
            maxima["variance"] = max(maxima["variance"], near(sq, vv[arm, step + 1],
                TOL["moment_abs"] + TOL["moment_rel"] * abs(vv[arm, step + 1]), "all Adam second moments"))
            expected = parameters[arm, step] + delta
            maxima["proposal"] = max(maxima["proposal"], near(expected, proposals[arm, step],
                TOL["one_step"], "all pre-repair Adam proposals"))
            repaired, alpha, eig0, eig1 = repair(proposals[arm, step])
            maxima["repair"] = max(maxima["repair"], near(repaired, parameters[arm, step + 1],
                TOL["repair_parameter"], "all physical repairs"))
            near(alpha, tape["projection_alpha"][arm, step], TOL["repair"], "all correlation shrinkage")
            near(eig0, tape["proposal_eigmin"][arm, step], TOL["repair"], "all proposed eigenvalues")
            maxima["eigenvalue"] = max(maxima["eigenvalue"], near(eig1, tape["eigmin"][arm, step + 1],
                TOL["repair"], "all postupdate eigenvalues"))
            require(eig1 >= .1, "correlation floor")
            groups = (slice(0, 4), slice(4, 8), slice(8, 14), slice(14, 15))
            projected = [bool(np.any(proposals[arm, step, group] != parameters[arm, step + 1, group]))
                         for group in groups]
            require(np.array_equal(projected, tape["projected"][arm, step]), "all repair flags")
            np.linalg.cholesky(correlation(parameters[arm, step + 1, 8:14]))
        states = parameters[arm, 449:].astype(float)
        mu = states[:, :4].mean(axis=0)
        sigma = np.mean([covariance(row) for row in states], axis=0)
        scales = np.sqrt(np.diag(sigma))
        corr = sigma / scales[:, None] / scales[None, :]
        require(np.linalg.eigvalsh(corr)[0] >= .1 - 1e-12, "TAIL correlation feasibility")
        require(np.linalg.eigvalsh(sigma - .1 * np.diag(np.diag(sigma)))[0] >= -1e-12,
                "TAIL covariance cone feasibility")
        tails.append(dict(arm=name, mu=mu.tolist(), covariance=sigma.tolist(), s=scales.tolist(),
                          rho=[float(corr[i, j]) for i, j in PAIRS], tau=float(states[:, 14].mean())))
        require(np.all(tape["probability_evaluations"][arm] == 100 * 64 * (4 if method == "ZERO" else 260)),
                "all probability evaluation counts")
    return dict(passed=True, checked_program_steps=49152 // 16, maxima=maxima, tails=tails)


def replay_inputs(seed_base, stream, teacher_mu, teacher_correlation):
    """Replay the original data/prior streams, not new independent science draws."""
    names = ("data_gaussian", "data_active_uniform", "data_nuisance_uniform",
             "prior_active_gaussian", "prior_nuisance_gaussian")
    rng = {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(seed_base), int(stream), role]))) for role, name in enumerate(names)}
    initial = {name: gen.bit_generator.state for name, gen in rng.items()}
    gaussian = rng[names[0]].standard_normal((4096, 4))
    uniform = rng[names[1]].random((4096, 4))
    noise_uniform = rng[names[2]].random((4096, 256))
    logits = gaussian @ np.linalg.cholesky(np.asarray(teacher_correlation, float)).T + teacher_mu
    y = (uniform < expit(logits)).astype(np.int8)
    yn = (noise_uniform < .5).astype(np.int8)
    hashes = {names[0]: ah(gaussian), names[1]: ah(uniform), names[2]: ah(noise_uniform)}
    streams = {names[3]: hashlib.sha256(), names[4]: hashlib.sha256()}
    chosen, first_hashes = {}, {}
    for step in range(512):
        active = rng[names[3]].standard_normal((100, 64, 4))
        nuisance = rng[names[4]].standard_normal((100, 64, 256))
        for name, array in ((names[3], active), (names[4], nuisance)):
            streams[name].update(array.tobytes(order="C"))
            if step == 0:
                first_hashes[name] = ah(array)
        if step in STEPS:
            chosen[step] = active.copy(), nuisance.copy()
    hashes.update({name: state.hexdigest() for name, state in streams.items()})
    inputs = dict(active_labels=y, nuisance_labels=yn, data_active_gaussian=gaussian,
                  data_active_uniform=uniform, first_prior_active=chosen[0][0],
                  first_prior_nuisance=chosen[0][1])
    p = (y.astype(float).sum(axis=0) + .5) / 4097.
    initial_mu = np.clip(np.log(p / (1. - p)), -2., 2.)
    return dict(input_arrays=inputs, chosen=chosen, role_input_sha256=hashes,
                first_prior_sha256=first_hashes, RNG_initial_states=initial,
                RNG_final_states={name: gen.bit_generator.state for name, gen in rng.items()},
                initial_mu=initial_mu)


def check_prefix(replay, tape):
    maxima = dict(API_loss=0., API_gradient=0., maximum_logit=0.,
                  math_API_loss=0., math_API_gradient=0.)
    rows = []
    for step in STEPS:
        begin = step * 64 % 4096
        labels = replay["input_arrays"]["active_labels"][begin:begin + 64]
        yn = replay["input_arrays"]["nuisance_labels"][begin:begin + 64]
        active, noise = replay["chosen"][step]
        for ai, arm in enumerate(ARMS):
            method = arm.split("_")[1]
            vector = tape["params"][ai, step]
            api = native(method, vector, active, noise, labels, yn)
            mat = mathematical(method, vector, active, noise, labels, yn)
            maxima["API_loss"] = max(maxima["API_loss"], near(api["loss"], tape["loss"][ai, step],
                                                            TOL["api_loss"], "limited original API loss"))
            maxima["API_gradient"] = max(maxima["API_gradient"], near(api["gradient"], tape["gradient"][ai, step],
                                                            TOL["api_gradient"], "limited original API gradient"))
            for name in ("sigmoid_zero_count", "sigmoid_one_count", "probability_evaluations"):
                require(api[name] == tape[name][ai, step], "limited original API " + name)
            maxima["maximum_logit"] = max(maxima["maximum_logit"], near(api["maximum_absolute_logit"],
                tape["maximum_absolute_logit"][ai, step], TOL["maxlogit"], "limited maximum logit"))
            maxima["math_API_loss"] = max(maxima["math_API_loss"], abs(api["loss"] - mat["loss"]))
            maxima["math_API_gradient"] = max(maxima["math_API_gradient"],
                                                float(np.max(abs(api["gradient"] - mat["gradient"]))))
            tau = float(vector[14])
            if tau == 0.:
                chi = .01
            else:
                lw = -np.logaddexp(0., -noise * tau * (2. * yn[None] - 1.)).sum(axis=-1)
                lp = lw - logsumexp(lw, axis=0, keepdims=True)
                chi = float(np.mean(np.sum(np.exp(2. * lp), axis=0)))
            near(chi, tape["nuisance_chi"][ai, step], TOL["mathematical"], "limited training-bank chi")
            rows.append(dict(preupdate_step=step, arm=arm,
                             independent_native_loss=api["loss"], independent_native_gradient=api["gradient"].tolist(),
                             independent_math64_loss=mat["loss"], independent_math64_gradient=mat["gradient"].tolist()))
    return dict(passed=True, maxima=maxima, records=rows, original_API_calls=24, mathematical_scores=24)


def freeze():
    root_freeze = read(BASE / "FREEZE.json")
    require(root_freeze["run_authorized"] is True, "main experiment authorization")
    paths = [SOURCE, DOC / "PROTOCOL.md", DOC / "INDEPENDENT_REVIEW_DRAFT.md",
             BASE / "FREEZE.json", TRAIN / "RESULTS.json", ROOT / "script/jointscale_compute.py",
             ROOT / "script/jointscale_kernel.py", OFF,
             ROOT / root_freeze["schema"]["teacher_reference_path"]]
    if (BASE / "INDEPENDENT_PREQUALIFICATION.json").exists():
        paths.append(BASE / "INDEPENDENT_PREQUALIFICATION.json")
    if (BASE / "REFERENCE/RESULTS.json").exists():
        paths.append(BASE / "REFERENCE/RESULTS.json")
    result = dict(created_utc=now(), source_sha256=sha(SOURCE),
                  inputs={rel(path): sha(path) for path in paths}, tolerances=TOL,
                  deterministic_check=check(), data_streams=list(range(16)),
                  bottom_level_steps=list(STEPS), arms=list(ARMS),
                  independent_reference=dict(order=44, root="symmetric", models=[
                      dict(stream=s, arm=arm, kind="TAIL") for s in range(16)
                      for arm in ("NEAR_FREE", "UNIT_ZERO")],
                      max_pmf_error=2e-7, max_KL_error=2e-6, mass_tolerance=2e-12),
                  scope="All original RNG bytes and saved updates; bottom-level only 384 training states, not independent full retraining; 32 independent TAIL PMFs",
                  new_independent_scientific_draws=0, automatic_retry=False,
                  numpy_version=np.__version__, torch_version=torch.__version__)
    save(OUT / "PROTOCOL.json", result)
    print(json.dumps(dict(audit_frozen=True, protocol_sha256=sha(OUT / "PROTOCOL.json"))), flush=True)


def bind():
    protocol = read(OUT / "PROTOCOL.json")
    require(protocol["source_sha256"] == sha(SOURCE), "audit source changed")
    for path, digest in protocol["inputs"].items():
        require(sha(ROOT / path) == digest, "audit frozen input changed: " + path)
    main = read(BASE / "FREEZE.json")
    require(sha(ROOT / "script/jointscale_compute.py") == main["schema"]["source_sha256"],
            "main frozen producer source")
    require(sha(ROOT / "script/jointscale_kernel.py") == main["schema"]["kernel_sha256"],
            "main frozen kernel source")
    for path, digest in main.get("inputs", {}).items():
        require(sha(ROOT / path) == digest, "main frozen input changed: " + path)
    return protocol, main


def compare_outputs(tape, record):
    for ai, arm in enumerate(ARMS):
        selections = [("TAIL", tape["params"][ai, -64:]), ("LAST", tape["params"][ai, -1:])]
        selections += [(str(step), tape["params"][ai, step:step + 1]) for step in (0, 32, 128, 512)]
        for kind, selected in selections:
            values = selected.astype(float)
            mu = values[:, :4].mean(axis=0)
            sigma = np.mean([covariance(v) for v in values], axis=0)
            scales = np.sqrt(np.diag(sigma)); corr = sigma / scales[:, None] / scales[None, :]
            expected = dict(mu=mu, covariance=sigma, s=scales, correlation=corr,
                            rho=np.array([corr[i, j] for i, j in PAIRS]), tau=values[:, 14].mean(),
                            eigmin=np.linalg.eigvalsh(corr)[0])
            saved = record["outputs"][arm][kind] if kind in ("TAIL", "LAST") else record["outputs"][arm]["checkpoints"][kind]
            for key, value in expected.items():
                near(value, saved[key], TOL["arithmetic"], "all saved output " + key)
        d = record["diagnostics"][arm]
        require(np.array_equal(d["repair_counts"], tape["projected"][ai].sum(axis=0)), "repair count")
        for key in ("sigmoid_zero_count", "sigmoid_one_count", "probability_evaluations"):
            require(d[key] == int(tape[key][ai].sum()), "diagnostic " + key)
        near(d["minimum_eigenvalue"], tape["eigmin"][ai].min(), 0., "diagnostic min eig")
        near(d["maximum_absolute_logit"], tape["maximum_absolute_logit"][ai].max(), 0., "diagnostic max logit")


def compare_controls(record, tape, prefix):
    saved = record["numerical_controls"]
    require({(r["preupdate_step"], r["arm"]) for r in saved} == {(s, a) for s in STEPS for a in ARMS},
            "numerical-control scope")
    independently = {(r["preupdate_step"], r["arm"]): r for r in prefix["records"]}
    maximum = 0.
    for row in saved:
        step, arm = row["preupdate_step"], row["arm"]
        ai = ARMS.index(arm); independent = independently[step, arm]
        near(row["official_loss"], tape["loss"][ai, step], 0., "control main loss binding")
        near(row["official_gradient"], tape["gradient"][ai, step], 0., "control main gradient binding")
        maximum = max(maximum, near(row["math64_loss"], independent["independent_math64_loss"],
                                    TOL["mathematical"], "independent stable value"))
        maximum = max(maximum, near(row["math64_gradient"], independent["independent_math64_gradient"],
                                    TOL["mathematical"], "independent stable all gradients"))
        difference = float(row["official_loss"]) - float(row["math64_loss"])
        gradient_difference = np.array(row["official_gradient"]) - np.array(row["math64_gradient"])
        near(difference, row["loss_difference"], message="saved loss difference")
        near(gradient_difference, row["gradient_difference"], message="saved gradient difference")
        require(row["passed"] == (abs(difference) <= 1e-4 and np.max(abs(gradient_difference)) <= 4e-5),
                "numerical qualification flag")
    summary = record["numerical_qualification"]
    require(summary["checks"] == 24 and summary["passed"] == sum(r["passed"] for r in saved), "control counts")
    near(summary["maximum_loss_difference"], max(abs(r["loss_difference"]) for r in saved), message="maximum loss error")
    near(summary["maximum_gradient_difference"], max(max(abs(v) for v in r["gradient_difference"]) for r in saved),
         message="maximum gradient error")
    return maximum


def run():
    protocol, main = bind(); schema = main["schema"]; config = schema["config"]
    production = read(TRAIN / "RESULTS.json")
    require(production["config"] == config, "training config")
    require(config["seed_base"] == 202610041500 and config["arms"] == list(ARMS), "prespecified seed/arms")
    save(OUT / "TRAIN_EXECUTION.json", dict(started_utc=now(), source_sha256=sha(SOURCE),
        protocol_sha256=sha(OUT / "PROTOCOL.json"), automatic_retry=False))
    tick = time.perf_counter(); records = []
    require([r["stream"] for r in production["records"]] == list(range(16)), "training stream order")
    for stream, record in enumerate(production["records"]):
        begin = time.perf_counter()
        require(record == read(TRAIN / f"stream_{stream:02}.json"), "stream record aggregation")
        require(record["source_sha256"] == sha(ROOT / "script/jointscale_compute.py")
                and record["freeze_sha256"] == sha(BASE / "FREEZE.json") and record["completed_steps"] == 512,
                "training record binding")
        replay = replay_inputs(config["seed_base"], stream, schema["data_model"]["mu"], schema["data_model"]["covariance"])
        for key in ("role_input_sha256", "first_prior_sha256", "RNG_initial_states", "RNG_final_states"):
            require(replay[key] == record[key], "complete original input replay " + key)
        require(record["role_seed_sequences"] == {name: [config["seed_base"], stream, role]
                for name, role in config["roles"].items()}, "role seed assignments")
        near(replay["initial_mu"], record["initial_mu"], TOL["arithmetic"], "empirical initial mean")
        require(np.array_equal(record["data_counts"], replay["input_arrays"]["active_labels"].sum(axis=0)), "data counts")
        for name in ("active_labels", "nuisance_labels"):
            require(ah(replay["input_arrays"][name]) == record["label_sha256"][name], "label hash")
        require(sha(ROOT / record["input_file"]) == record["input_sha256"], "input NPZ binding")
        require(sha(ROOT / record["output_file"]) == record["output_sha256"], "tape NPZ binding")
        with np.load(ROOT / record["input_file"]) as saved:
            require(set(saved.files) == set(replay["input_arrays"]), "input NPZ fields")
            for name, values in replay["input_arrays"].items():
                require(np.array_equal(saved[name], values), "all saved input bytes " + name)
        with np.load(ROOT / record["output_file"]) as saved:
            tape = {name: saved[name].copy() for name in saved.files}
        updates = trajectory(tape, replay["initial_mu"])
        compare_outputs(tape, record)
        prefix = check_prefix(replay, tape)
        stable_error = compare_controls(record, tape, prefix)
        result = dict(stream=stream, passed=True, complete_RNG_hashes_and_states=True,
                      complete_saved_updates=updates, prefix=prefix, stable64_independent_error=stable_error,
                      seconds=time.perf_counter() - begin)
        save(OUT / f"stream_{stream:02}.json", result); records.append(result)
        print(json.dumps(dict(audit_stream=stream, passed=True, seconds=result["seconds"],
                              stable64_error=stable_error)), flush=True)
    ledger = dict(new_observations=65536, optimizations=96, main_API_calls=49152, numerical_API_calls=384,
                  prior_vectors=52428800, prior_Gaussian=13631488000, data_Gaussian=262144,
                  data_Uniform=17039360, new_NN=0)
    require(production["accounting"] == ledger, "training accounting")
    qualified = all(r["numerical_qualification"]["passed"] == 24 for r in production["records"])
    require(production["numerical_qualification_passed"] == qualified, "training numerical summary")
    save(OUT / "TRAIN_REPORT.json", dict(passed=True, records=records, seconds=time.perf_counter() - tick,
        numerical_qualification_passed=qualified, source_sha256=sha(SOURCE),
        protocol_sha256=sha(OUT / "PROTOCOL.json"), training_results_sha256=sha(TRAIN / "RESULTS.json"),
        accounting=dict(replayed_data_Gaussian=262144, replayed_data_Uniform=17039360,
                        replayed_prior_Gaussian=13631488000, new_independent_scientific_draws=0,
                        original_API_calls=384, mathematical_scores=384, saved_program_updates=49152,
                        new_model_integrals=0),
        scope="All original input hashes/states and all saved update arithmetic; bottom-level only 384 states. Not full independent retraining."))


def teacher(main):
    path = ROOT / main["schema"]["teacher_reference_path"]
    require(sha(path) == main["schema"]["teacher_reference_sha256"], "fixed teacher binding")
    row = next(r for r in read(path)["records"] if r["name"] == "mixed")
    p = np.asarray(row["teacher_pmf"], float)
    require(p.shape == (16,) and (p > 0.).all() and abs(p.sum() - 1.) <= 2e-12, "teacher PMF")
    near(p @ PATTERNS, row["derived"]["p"], TOL["arithmetic"], "teacher moments")
    return p


def statistics():
    _, main = bind(); p = teacher(main)
    ref = read(BASE / "REFERENCE/RESULTS.json"); train = read(TRAIN / "RESULTS.json")
    config = main["schema"]["config"]
    require(ref["config"] == config, "reference config")
    require(ref["teacher_reference_path"] == main["schema"]["teacher_reference_path"]
            and ref["teacher_reference_sha256"] == main["schema"]["teacher_reference_sha256"], "reference teacher")
    require(len(ref["records"]) == 96, "TAIL-only reference count")
    require([(r["stream"], r["arm"], r["kind"]) for r in ref["records"]]
            == [(s, arm, "TAIL") for s in range(16) for arm in ARMS], "TAIL-only reference order")
    save(OUT / "STATISTICS_EXECUTION.json", dict(started_utc=now(), source_sha256=sha(SOURCE),
        reference_results_sha256=sha(BASE / "REFERENCE/RESULTS.json"), new_integrals=0))
    risks = np.zeros((16, 6)); maxima = dict(saved_KL=0., normalization=0., qualification=0.)
    for row in ref["records"]:
        stream, arm = row["stream"], row["arm"]
        require(row["model"] == train["records"][stream]["outputs"][arm]["TAIL"], "reference model identity")
        require([v["order"] for v in row["variants"]] == [32, 40, 48], "fixed quadrature orders")
        main_q = np.asarray(row["main_pmf"], float)
        near(main_q, row["variants"][1]["pmf"], 0., "main order identity")
        main_risk = risk(p, main_q)
        maxima["saved_KL"] = max(maxima["saved_KL"], near(main_risk, row["reference_joint_KL"],
                                                        TOL["arithmetic"], "all saved PMF risks"))
        risks[stream, ARMS.index(arm)] = main_risk
        qualified = []
        for variant, qual in zip(row["variants"], row["qualifications"]):
            raw = np.asarray(variant["raw_pmf"], float); mass = float(raw.sum()); q = raw / mass
            require(variant["root"] == "cholesky" and variant["normalization_applied"] is True,
                    "reference root/normalization")
            near(mass, variant["raw_mass"], message="raw mass")
            near(q, variant["pmf"], message="normalized PMF")
            mass_error = abs(mass - 1.); change = float(np.max(abs(q - main_q))); delta = risk(p, q) - main_risk
            positive = bool((q > 0.).all())
            passed = change <= 2e-7 and abs(delta) <= 2e-6 and mass_error <= 2e-12 and positive
            for key, value in (("mass_error", mass_error), ("max_pmf_change", change), ("KL_change", delta)):
                near(value, qual[key], TOL["arithmetic"], "all qualification " + key)
            near(mass_error, variant["mass_error"], message="raw mass error")
            require(qual["order"] == variant["order"] and qual["positive"] == positive
                    and qual["passed"] == passed and variant["Gaussian_nodes"] == variant["order"] ** 4,
                    "all quadrature qualifications")
            qualified.append(passed)
        require(row["qualification_passed"] == all(qualified), "per-model qualification")
    for stream in range(16):
        saved = read(BASE / "REFERENCE" / f"stream_{stream:02}.json")
        require(saved["stream"] == stream and saved["rows"] == ref["records"][6 * stream:6 * (stream + 1)],
                "reference stream aggregation")
        require(saved["qualification_passed"] == all(r["qualification_passed"] for r in saved["rows"]),
                "reference stream qualification")
    product = product_reference(p)
    near(p, ref["teacher_pmf"], 0., "reference teacher PMF")
    near(product["marginals"], ref["teacher_marginals"], message="teacher marginals")
    near(product["pmf"], ref["truth_product_pmf"], message="product reference")
    near(product["reference_joint_KL"], ref["truth_product_risk"], message="total correlation")
    primary = family(risks, product["reference_joint_KL"])
    near(float(student_t.ppf(1. - .05 / 12., 15)), ref["critical_t"], message="family critical t")
    require(len(ref["primary"]) == 6, "primary family count")
    for expected, saved in zip(primary, ref["primary"]):
        require(saved["name"] == expected["start"] + "_" + expected["name"]
                and saved["start"] == expected["start"] and saved["expected"] == "negative", "contrast identity")
        for key in ("stream_values", "mean", "SE", "lower", "upper"):
            near(expected[key], saved[key], message="primary " + key)
        require(expected["effect_passed"] == saved["effect_passed"]
                and expected["precision_passed"] == saved["precision_passed"], "primary gates")
    for ai, arm in enumerate(ARMS):
        near(risks[:, ai], ref["descriptive"][arm]["risks"], message="all arm risks")
        near(risks[:, ai].mean(), ref["descriptive"][arm]["mean_joint_KL"], message="arm risk mean")
    effects = sum(r["effect_passed"] for r in primary); precision = sum(r["precision_passed"] for r in primary)
    require(ref["primary_effects_passed"] == effects and ref["primary_precision_passed"] == precision, "primary gate counts")
    qualified = all(r["qualification_passed"] for r in ref["records"])
    require(ref["qualification_passed"] == qualified, "all reference qualification")
    ledger = dict(new_PMF_integrations=288, Gaussian_nodes=856031232, new_observations=0, new_prior=0, new_fits=0)
    require(ref["accounting"] == ledger, "reference accounting")
    save(OUT / "STATISTICS_REPORT.json", dict(passed=True, source_sha256=sha(SOURCE),
        protocol_sha256=sha(OUT / "PROTOCOL.json"), reference_results_sha256=sha(BASE / "REFERENCE/RESULTS.json"),
        checked_model_outputs=96, checked_saved_PMFs=288, reference_qualified=qualified,
        science_gate_counts=dict(effect=effects, precision=precision,
            combined=sum(r["effect_passed"] and r["precision_passed"] for r in primary), total=6),
        maxima=maxima, new_model_integrals=0, scope="All saved PMF normalization, qualifications, risk and family arithmetic; independent integrations are a separate phase."))


def reference():
    protocol, main = bind(); p = teacher(main)
    ref = read(BASE / "REFERENCE/RESULTS.json")
    lookup = {(r["stream"], r["arm"]): r for r in ref["records"]}
    save(OUT / "REFERENCE_EXECUTION.json", dict(started_utc=now(), source_sha256=sha(SOURCE),
        protocol_sha256=sha(OUT / "PROTOCOL.json"), reference_results_sha256=sha(BASE / "REFERENCE/RESULTS.json"),
        planned_PMFs=32, planned_Gaussian_nodes=119939072, automatic_retry=False))
    records = []; tick = time.perf_counter()
    for selection in protocol["independent_reference"]["models"]:
        stream, arm = selection["stream"], selection["arm"]
        row = lookup[stream, arm]; model = row["model"]
        value = quadrature44(model["mu"], model["covariance"])
        q = np.asarray(value["pmf"]); kl = risk(p, q)
        error = float(np.max(abs(q - np.asarray(row["main_pmf"]))))
        delta = kl - float(row["reference_joint_KL"])
        positive = bool((q > 0.).all())
        passed = error <= 2e-7 and abs(delta) <= 2e-6 and value["normalization_error"] <= 2e-12 and positive
        result = dict(stream=stream, arm=arm, kind="TAIL", passed=passed, model=model, **value,
                      independent_reference_joint_KL=kl, maximum_pmf_difference=error,
                      KL_difference=delta, positive=positive)
        records.append(result); save(OUT / f"reference_{stream:02}_{arm}.json", result)
        print(json.dumps(dict(reference_audit_stream=stream, arm=arm, passed=passed,
                              maximum_pmf_error=error, KL_error=delta)), flush=True)
    save(OUT / "REFERENCE_REPORT.json", dict(passed=all(r["passed"] for r in records),
        records=records, source_sha256=sha(SOURCE), protocol_sha256=sha(OUT / "PROTOCOL.json"),
        reference_results_sha256=sha(BASE / "REFERENCE/RESULTS.json"), seconds=time.perf_counter() - tick,
        accounting=dict(new_independent_scientific_draws=0, new_observations=0, new_fits=0,
                        independent_PMF_integrations=32, independent_Gaussian_nodes=119939072),
        scope="Prespecified 32 TAIL models only, GH44 symmetric square root. Numerical qualification, not a rigorous integration-error certificate."))


def report():
    bind()
    names = ("TRAIN_REPORT.json", "STATISTICS_REPORT.json", "REFERENCE_REPORT.json")
    pieces = {name: read(OUT / name) for name in names}
    result = dict(passed=all(row["passed"] for row in pieces.values()), source_sha256=sha(SOURCE),
                  protocol_sha256=sha(OUT / "PROTOCOL.json"), phase_receipts={name: sha(OUT / name) for name in names},
                  science_gate_counts=pieces["STATISTICS_REPORT.json"]["science_gate_counts"],
                  main_reference_qualified=pieces["STATISTICS_REPORT.json"]["reference_qualified"],
                  native32_control_qualified=pieces["TRAIN_REPORT.json"]["numerical_qualification_passed"],
                  accounting=dict(new_independent_scientific_draws=0, new_observations=0, new_fits=0,
                                  independent_PMF_integrations=32, original_API_calls=384,
                                  mathematical_scores=384, saved_Adam_updates=49152),
                  scope="All original inputs and saved arithmetic; 384 states re-evaluated and 32 independent PMFs; not full bottom-level retraining.")
    save(OUT / "REPORT.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("check", "freeze", "run", "statistics", "reference", "report"))
    args = parser.parse_args()
    if args.phase == "check":
        print(json.dumps(check(), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        try:
            globals()[args.phase]()
        except BaseException:
            failure = OUT / (args.phase.upper() + "_FAILURE.json")
            if not failure.exists():
                save(failure, dict(utc=now(), traceback=traceback.format_exc(), automatic_retry=False))
            raise
