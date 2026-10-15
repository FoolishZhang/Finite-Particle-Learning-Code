"""Replayable full-private finite-M kernels for critical-path checks.

Any positive integer M is accepted; no full observation alphabet is built.
Imports never sample, choose a seed/budget/candidate, or write files. Raw RP
differentiates a saved bank loss, CV differentiates its controlled scalar, and
Price is a population identity rather than a saved-bank RP equality.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, log_expit, logsumexp
import signal_proposal_core as legacy
from competition_core import h_probabilities, proposal_log_normalizer, sample_oriented_qbeta

SCHEMA = "selection_finite_core_v1"
ACTIVE_DIMENSION = 3


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _parameters(epsilon, tau):
    epsilon = np.asarray(epsilon, dtype=float)
    if epsilon.shape != (3,) or not np.isfinite(epsilon).all():
        raise ValueError("epsilon must contain three finite group couplings")
    if isinstance(tau, bool) or not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite nonnegative scalar")
    return epsilon, float(tau)


def _beta(beta):
    if isinstance(beta, bool) or not np.isscalar(beta) or not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be a finite scalar in [0,1]")
    return float(beta)


def _counts(counts, B, M):
    counts = np.asarray(counts)
    if counts.shape != (B, 3) or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > M):
        raise ValueError(f"counts must be integer (B,3) with entries 0..{M}")
    return counts.astype(np.int64, copy=False)


def _labels(label_probs, B):
    labels = legacy.spin_labels(3)
    q = np.full(8, 1 / 8) if label_probs is None else np.asarray(label_probs, dtype=float)
    if q.shape == (8,):
        q = np.broadcast_to(q, (B, 8))
    if q.shape != (B, 8) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("label_probs must be finite nonnegative (8,) or (B,8)")
    if not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-13):
        raise ValueError("label_probs must sum to one; no renormalization")
    return labels, q


def critical_parameters(M, *, a, tau, gamma):
    """Caller-fixed epsilon=a/sqrt(M), beta=1-gamma/sqrt(3M).

    a may be a scalar (same coupling in every group) or three group values.
    A requested point outside beta in [0,1] is rejected, never clipped.
    """
    M = _integer(M, "M")
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = np.repeat(a, 3)
    a, tau = _parameters(a, tau)
    if isinstance(gamma, bool) or not np.isscalar(gamma) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative scalar")
    beta = _beta(1 - float(gamma) / math.sqrt(3 * M))
    return {"M": M, "epsilon": a / math.sqrt(M), "tau": tau, "beta": beta,
            "a": a.copy(), "gamma": float(gamma), "path": "fixed_a_tau_gamma_critical"}


def canonical_spins(counts, M):
    """u[B,3,M]: first counts entries +1, the remaining entries -1."""
    M = _integer(M, "M")
    counts = np.asarray(counts)
    counts = _counts(counts, len(counts), M)
    return np.where(np.arange(M)[None, None, :] < counts[:, :, None], 1, -1).astype(np.int8)


def count_log_combinations(counts, M):
    """Multiplicity only for supplied counts; no (M+1)^3 alphabet allocation."""
    M = _integer(M, "M")
    counts = np.asarray(counts)
    counts = _counts(counts, len(counts), M)
    unique, inverse = np.unique(counts, return_inverse=True)
    logs = np.asarray([math.log(math.comb(M, int(c))) for c in unique])
    return logs[inverse].reshape(counts.shape).sum(axis=1)


def sample_teacher_observations(rng, Xstar, *, M, epsilon, tau, inner_order,
                                h_chunk_size=4096):
    """Sample counts | Xstar ~ product Binomial(M,h_tau(epsilon*Xstar)).

    h_tau(s)=E_phi sigma(tau*G+s), evaluated by caller-selected ordinary GH;
    it is NOT sigma(s) and is not a certified integral. label_probs enumerates
    Y|Xstar. Averaging these banks is a population teacher evaluation with
    labels integrated conditionally; neither Xstar nor these integration
    weights is a learning-visible observation or proposal input. Candidate
    particles must be drawn independently of teacher banks by the caller.
    """
    M = _integer(M, "M")
    epsilon, tau = _parameters(epsilon, tau)
    Xstar = np.asarray(Xstar, dtype=float)
    if Xstar.ndim != 2 or Xstar.shape[0] < 1 or Xstar.shape[1] != 3 or not np.isfinite(Xstar).all():
        raise ValueError("Xstar must be finite nonempty (B,3)")
    if rng is None or not hasattr(rng, "binomial"):
        raise ValueError("An explicit caller-owned RNG with binomial is required")
    h = h_probabilities(Xstar * epsilon, tau=tau, inner_order=inner_order,
                        chunk_size=h_chunk_size)
    if np.any(h["h"] < 0) or np.any(h["h"] > 1):
        raise FloatingPointError("GH count probabilities are outside [0,1]; no clipping")
    counts = rng.binomial(M, h["h"])
    labels = legacy.spin_labels(3)
    label_probs = expit(Xstar[:, None, :] * labels[None, :, :]).prod(axis=2)
    return {"counts": counts, "label_probs": label_probs, "labels": labels,
            "count_success_probability": h["h"], "count_failure_probability": h["complement"],
            "sampling": {"teacher_banks": len(Xstar), "binomial_variates": 3 * len(Xstar),
                         "private_observations_represented": 3 * M * len(Xstar),
                         "sampled_labels": 0, "labels_conditionally_enumerated": 8,
                         "rng_ownership": "caller", "group_size": M},
            "label_conditioning": "Y_given_Xstar_population_evaluation_only",
            "teacher_latent_is_learning_visible": False, "M": M,
            "epsilon": epsilon.copy(), "tau": tau, "inner_order": int(inner_order),
            "qualification": h["qualification"]}


def _score_banks(x, dx, labels, q, coupling, coupling_score, coupling_diag,
                 baseline, nuisance_log_constant, return_per_label):
    B, K, _ = x.shape
    constant = np.broadcast_to(np.asarray(nuisance_log_constant, dtype=float), (B,))
    if not np.isfinite(constant).all():
        raise ValueError("nuisance_log_constant must be finite scalar or (B,)")
    shift = baseline.max(axis=1)
    centered = baseline - shift[:, None]
    nu, _, _ = legacy.posterior_weights(centered)
    winner = np.argmax(nu, axis=1)
    out = {key: np.zeros((B, 3)) for key in ("raw_rp", "cv_rp", "price")}
    out.update({key: np.zeros(B) for key in ("shifted_loss", "shifted_controlled_loss", "overlap", "ess")})
    details = []
    for index, y in enumerate(labels):
        signed = x * y
        active = log_expit(signed).sum(axis=2) + coupling
        score = y * expit(-signed) + coupling_score
        diagonal = -expit(signed) * expit(-signed) + coupling_diag
        logits = centered + active
        pi, _, gamma = legacy.posterior_weights(logits)
        derivative = np.einsum("bki,bkip->bkp", score, dx, optimize=True)
        raw = -np.sum(pi[:, :, None] * derivative, axis=1)
        difference = nu - pi
        # Recover the dominant component from sum(nu-pi)=0, including when
        # both dominant probabilities round to one in a high-noise bank.
        difference[np.arange(B), winner] = 0
        difference[np.arange(B), winner] = -difference.sum(axis=1)
        cv = np.sum(difference[:, :, None] * derivative, axis=1)
        price = np.stack([-np.sum(gamma * score[:, :, i] * score[:, :, j], axis=1)
                          for i, j in legacy.pair_order(3)], axis=1)
        loss = math.log(K) - logsumexp(logits, axis=1)
        controlled = loss + np.sum(nu * active, axis=1)
        probability = q[:, index]
        for key, value in (("raw_rp", raw), ("cv_rp", cv), ("price", price)):
            out[key] += probability[:, None] * value
        for key, value in (("shifted_loss", loss), ("shifted_controlled_loss", controlled),
                           ("overlap", gamma.sum(axis=1)), ("ess", 1 / np.sum(pi**2, axis=1))):
            out[key] += probability * value
        if return_per_label:
            details.append({"raw_rp": raw, "cv_rp": cv, "price": price,
                            "shifted_loss": loss, "shifted_controlled_loss": controlled,
                            "score": score, "diag_hessian": diagonal,
                            "pi": pi, "active_log_likelihood": active,
                            "log_weights": baseline + active + constant[:, None]})
    out["loss"] = out["shifted_loss"] - shift - constant
    out["controlled_loss"] = out["shifted_controlled_loss"] - shift - constant
    out["rp"] = out["raw_rp"]
    out.update({"S_base": baseline, "noise_pi": nu, "bank_shift": shift,
                "nuisance_log_constant": constant.copy(), "labels": labels,
                "pair_order": legacy.pair_order(3),
                "target": "full_private_variables_fixed_epsilon0_qbeta_proposal",
                "combination_constants_in_loss": False})
    if return_per_label:
        per = {key: np.stack([r[key] for r in details], axis=1) for key in details[0]}
        per["loss"] = per["shifted_loss"] - shift[:, None] - constant[:, None]
        per["controlled_loss"] = per["shifted_controlled_loss"] - shift[:, None] - constant[:, None]
        per["rp"] = per["raw_rp"]
        out["per_label"] = per
    return out


def full_banks(z, T, counts, C, *, M, epsilon, tau, beta, label_probs=None,
               q=None, nuisance_log_constant=0., return_per_label=False,
               private_chunk_size=None):
    """Full importance scores with T[B,K,3,M] oriented as u*G under q_beta,0.

    The private proposal is phi(T)*sigma(tau*T)**beta/Z_beta,tau; it is fixed
    independently of candidate C. The absolute ordered-observation weight is
    product sigma(y_i*x_i) * product sigma(tau*T+epsilon_i*u*x_i) /
    sigma(tau*T)**beta * Z_beta,tau**(3*M). Supply 3*M*log(Z_beta,tau)
    through nuisance_log_constant for that normalized loss. Default zero
    omits this candidate-independent constant, as in the frozen old kernel.
    loss omits binomial multiplicity; count_loss includes it. Stable candidate
    contrasts use shifted_loss. q aliases label_probs, not an inferred label
    conditional law. The caller supplies valid weights or defaults to uniform.

    Chunking bounds temporary private arrays and consumes no RNG. Given the
    same T, chunks are numerically equivalent, not promised bitwise identical
    across chunk sizes because floating summation order changes. T remains a
    complete replay input (also accepted as a memory-mapped float64 array).
    """
    M = _integer(M, "M")
    epsilon, tau = _parameters(epsilon, tau)
    beta = _beta(beta)
    if q is not None:
        if label_probs is not None:
            raise ValueError("Provide label_probs or its alias q, not both")
        label_probs = q
    z = np.asarray(z, dtype=float)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or z.shape[2] != 3 or not np.isfinite(z).all():
        raise ValueError("z must be finite nonempty (B,K,3)")
    if np.asarray(C).shape != (3, 3):
        raise ValueError("C must be a positive definite m3 correlation matrix")
    x, dx = legacy.candidate_inputs(z, C)
    counts = _counts(counts, len(z), M)
    labels, probabilities = _labels(label_probs, len(z))
    T = np.asarray(T, dtype=float)
    if T.shape != x.shape[:2] + (3, M):
        raise ValueError("T must be oriented private draws (B,K,3,M)")
    chunk = M if private_chunk_size is None else _integer(private_chunk_size, "private_chunk_size")
    baseline = np.zeros(x.shape[:2])
    coupling = np.zeros_like(baseline)
    coupling_score = np.zeros_like(x)
    coupling_diag = np.zeros_like(x)
    for start in range(0, M, chunk):
        stop = min(start + chunk, M)
        block = T[..., start:stop]
        if not np.isfinite(block).all():
            raise ValueError("T must be finite")
        spins = np.where(np.arange(start, stop)[None, None, None, :] < counts[:, None, :, None], 1, -1)
        base_logits = tau * block
        shifted = base_logits + epsilon[None, None, :, None] * spins * x[:, :, :, None]
        if not np.isfinite(shifted).all() or not np.isfinite(base_logits).all():
            raise FloatingPointError("Full-private logit overflow")
        base_loglik = log_expit(base_logits)
        baseline += (1 - beta) * base_loglik.sum(axis=(2, 3))
        coupling += (log_expit(shifted) - base_loglik).sum(axis=(2, 3))
        coupling_score += (epsilon[None, None, :, None] * spins * expit(-shifted)).sum(axis=3)
        coupling_diag -= (epsilon[None, None, :, None]**2 * expit(shifted) * expit(-shifted)).sum(axis=3)
    result = _score_banks(x, dx, labels, probabilities, coupling, coupling_score,
                         coupling_diag, baseline, nuisance_log_constant, return_per_label)
    combinations = count_log_combinations(counts, M)
    result.update({"schema": SCHEMA, "M": M, "group_size": M, "beta": beta, "tau": tau,
                   "epsilon": epsilon.copy(), "private_variables_per_particle": 3 * M,
                   "private_chunk_size": min(chunk, M), "count_log_combination": combinations,
                   "count_loss": result["loss"] - combinations,
                   "normalizer_policy": "caller_supplied_3M_logZ; default_omits_constant",
                   "proposal_is_true_new_conditional_posterior": tau == 0 or (not np.any(epsilon) and beta == 1)})
    if return_per_label:
        result["per_label"]["count_loss"] = result["per_label"]["loss"] - combinations[:, None]
    return result
