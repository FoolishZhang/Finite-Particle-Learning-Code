"""Independent three-dimensional critical-proposal Gaussian limit kernel.

Imports never draw random inputs or write files. Callers supply independent
teacher/data, prior z, connection W and baseline V banks. Across candidates
they must reuse those same inputs. label_probs[B,8] is mandatory and uses
lexicographic {-1,+1} labels; under the model it is P(Y|Xstar), not P(Y|Z).

Only physical Cholesky derivatives, label order and stable posterior weights
are reused from the old kernel. The new likelihood and scores are assembled
here. RP differentiates each saved-bank scalar; Price is a Gaussian expectation
identity. Raw RP, baseline-CV RP and Price agree in expectation, not bankwise.
Noise derivatives below also denote expectation identities for the raw loss,
not derivatives of a particular realization with V and W held fixed.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm
from signal_proposal_core import candidate_inputs, posterior_weights, spin_labels

SCHEMA = "selection_limit_core_v1"
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))
LABELS = spin_labels(3)
LABELS.setflags(write=False)


def _scalar(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _labels(label_probs, B):
    q = _finite_array(label_probs, (B, 8), "label_probs")
    if np.any(q < 0) or not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-13):
        raise ValueError("label_probs must be nonnegative and sum to one; no renormalization")
    return q


def data_from_inputs(teacher_z, observation_noise, Cstar, *, b):
    """Transform independent N(0,I3) inputs into Xstar, Z and P(Y|Xstar).

    This function performs no sampling. Cstar is strictly positive definite
    with unit diagonal. Conditional on Xstar, the eight label probabilities
    factor into logistic Bernoulli probabilities. E is independent of Xstar.
    """
    b = _scalar(b, "b")
    teacher_z = np.asarray(teacher_z, dtype=np.float64)
    if teacher_z.ndim != 2 or teacher_z.shape[0] < 1 or teacher_z.shape[1] != 3:
        raise ValueError("teacher_z must be finite nonempty (B,3)")
    teacher_z = _finite_array(teacher_z, teacher_z.shape, "teacher_z")
    E = _finite_array(observation_noise, teacher_z.shape, "observation_noise")
    if np.asarray(Cstar).shape != (3, 3):
        raise ValueError("Cstar must be a positive definite m3 correlation matrix")
    Xstar, _ = candidate_inputs(teacher_z[:, None, :], Cstar)
    Xstar = Xstar[:, 0, :]
    Z = b * Xstar + E
    q = np.exp(log_expit(Xstar[:, None, :] * LABELS[None, :, :]).sum(axis=2))
    if not np.isfinite(Z).all():
        raise FloatingPointError("Observation transformation overflowed")
    _labels(q, len(q))
    return {"Xstar": Xstar, "Z": Z, "label_probs": q,
            "labels": LABELS.copy(), "label_mode": "conditional_on_Xstar_enumerated"}


def control_mean_constant(*, b, eta, order=64):
    """Ordinary GH value of the C-independent mean of the added CV control.

    E[sum(nu*active)|Y,Z,V] = 3 E log sigma(G) - 3(b*b+eta)/2,
    after integrating independent z and W. The W.X and Z.X terms have zero
    mean. This value is diagnostic quadrature, not an interval certificate.
    """
    b, eta = _scalar(b, "b"), _scalar(eta, "eta", nonnegative=True)
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("order must be an integer >=2")
    nodes, weights = roots_hermitenorm(int(order))
    return float(3 * (weights @ log_expit(nodes)) / math.sqrt(2 * math.pi)
                 - 1.5 * (b * b + eta))


def evaluate_banks(z, W, V, Z, C, *, b, t, eta, label_probs, return_per_label=False):
    """Score caller-provided shared banks, with independently variable t,eta>=0.

    z,W[B,K,3], V[B,K], Z[B,3], label_probs[B,8]. Candidate C has unit
    diagonal and must be positive definite. Parameters and the data channel b
    are fixed during physical correlation differentiation. The raw scalar is
    -log(mean(exp(logweight))) without the candidate-independent Gaussian
    observation density. Its candidate contrasts are therefore normalized.

    The legal CV adds sum_k nu_k active_k with nu=softmax(sqrt(t)*V), which
    depends only on baseline noise. Its expectation is C-independent. It is
    not a zero-mean absolute loss: controlled_loss differs by that known mean.
    Noise derivatives refer to loss, not controlled_loss; the latter's eta
    derivative has the additional constant -3/2. Candidate contrasts cancel
    the CV mean. Enumerated labels/particles are not independent repetitions.
    """
    b = _scalar(b, "b")
    t, eta = (_scalar(value, name, nonnegative=True) for value, name in ((t, "t"), (eta, "eta")))
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or z.shape[2] != 3:
        raise ValueError("z must be finite nonempty (B,K,3)")
    B, K, _ = z.shape
    z, W = _finite_array(z, z.shape, "z"), _finite_array(W, z.shape, "W")
    V, Z = _finite_array(V, (B, K), "V"), _finite_array(Z, (B, 3), "Z")
    q = _labels(label_probs, B)
    if np.asarray(C).shape != (3, 3):
        raise ValueError("C must be a positive definite m3 correlation matrix")
    x, dx = candidate_inputs(z, C)
    norm2 = np.sum(x * x, axis=2)
    coupling = (b * np.sum(Z[:, None, :] * x, axis=2)
                + math.sqrt(eta) * np.sum(W * x, axis=2)
                - .5 * (b * b + eta) * norm2)
    coupling_score = b * Z[:, None, :] + math.sqrt(eta) * W - (b * b + eta) * x
    baseline = math.sqrt(t) * V - .5 * t
    if not all(np.isfinite(a).all() for a in (x, dx, norm2, coupling, coupling_score, baseline)):
        raise FloatingPointError("Limit logweights or path derivatives overflowed")
    shift = baseline.max(axis=1)
    centered = baseline - shift[:, None]
    nu, _, _ = posterior_weights(centered)
    winner = np.argmax(nu, axis=1)
    scalar_keys = ("shifted_loss", "shifted_controlled_loss", "overlap", "ess",
                   "noise_t_derivative", "noise_eta_derivative")
    out = {key: np.zeros(B) for key in scalar_keys}
    out.update({key: np.zeros((B, 3)) for key in ("raw_rp", "cv_rp", "price")})
    details = []
    for label_index, y in enumerate(LABELS):
        if not return_per_label and not np.any(q[:, label_index]):
            continue
        signed = x * y
        active = log_expit(signed).sum(axis=2) + coupling
        score = y * expit(-signed) + coupling_score
        logits = centered + active
        pi, _, pi_complement = posterior_weights(logits)
        path_score = np.einsum("bki,bkip->bkp", score, dx, optimize=True)
        raw = -np.sum(pi[:, :, None] * path_score, axis=1)
        difference = nu - pi
        # Reconstruct one coordinate from the exact zero-sum constraint.
        difference[np.arange(B), winner] = 0
        difference[np.arange(B), winner] = -difference.sum(axis=1)
        cv = np.sum(difference[:, :, None] * path_score, axis=1)
        price = np.stack([-np.sum(pi_complement * score[:, :, i] * score[:, :, j], axis=1)
                          for i, j in PAIR_ORDER], axis=1)
        loss = math.log(K) - logsumexp(logits, axis=1)
        values = {"shifted_loss": loss,
                  "shifted_controlled_loss": loss + np.sum(nu * active, axis=1),
                  "raw_rp": raw, "cv_rp": cv, "price": price,
                  "overlap": pi_complement.sum(axis=1), "ess": 1 / np.sum(pi * pi, axis=1),
                  "noise_t_derivative": .5 * np.sum(pi * pi, axis=1),
                  "noise_eta_derivative": .5 * np.sum(pi * pi * norm2, axis=1)}
        probability = q[:, label_index]
        for key, value in values.items():
            out[key] += probability.reshape((B,) + (1,) * (value.ndim - 1)) * value
        if return_per_label:
            details.append(dict(values, pi=pi, score=score, active_log_likelihood=active,
                                logweight=logits + shift[:, None],
                                diag_hessian=-expit(signed) * expit(-signed) - (b * b + eta)))
    out["loss"] = out["shifted_loss"] - shift
    out["controlled_loss"] = out["shifted_controlled_loss"] - shift
    out.update(rp=out["raw_rp"], S_base=baseline, noise_pi=nu, bank_shift=shift,
               labels=LABELS.copy(), pair_order=PAIR_ORDER, b=b, t=t, eta=eta,
               target="critical_proposal_Gaussian_limit", label_mode="explicit_enumerated_labels",
               gradients_equal_bankwise=False, noise_derivatives_are_expectation_identities=True)
    if return_per_label:
        per = {key: np.stack([record[key] for record in details], axis=1) for key in details[0]}
        per["loss"] = per["shifted_loss"] - shift[:, None]
        per["controlled_loss"] = per["shifted_controlled_loss"] - shift[:, None]
        per["rp"] = per["raw_rp"]
        out["per_label"] = per
    return out


def contrast_banks(z, W, V, Z, C, reference_C, **kwargs):
    """Paired raw/CV candidate loss differences; common baseline shift cancels."""
    left = evaluate_banks(z, W, V, Z, C, **kwargs)
    right = evaluate_banks(z, W, V, Z, reference_C, **kwargs)
    return {"loss": left["shifted_loss"] - right["shifted_loss"],
            "controlled_loss": left["shifted_controlled_loss"] - right["shifted_controlled_loss"]}
