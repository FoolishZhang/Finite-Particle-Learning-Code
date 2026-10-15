"""Replay-only m3 endpoint kernels for teacher-level finite-model comparisons.

The private-observation size M is any positive integer. The full endpoint uses
the old, candidate-independent beta=1, lambda=0 proposal; the collapsed endpoint
is the exact private conditional-integration formula at beta=lambda=1, evaluated
with ordinary 1D GH. Active prior particles are still finite K in both cases.
Imports never sample, choose a scientific configuration, or write files.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit
import competition_core as competition
import selection_finite_core as finite
import signal_proposal_core as legacy
from informative_projection import project_coordinates

SCHEMA = "teacher_endpoint_core_v1"
ENDPOINTS = ("full", "collapsed")
# The existing exact rejection sampler remains the only private sampler.
sample_oriented_qbeta = competition.sample_oriented_qbeta


def observed_label_probabilities(observed_labels):
    """One-hot observed Y[B,3] in the established lexicographic spin order.

    This is an observed-data interface, not Y|Xstar conditional enumeration.
    No teacher latent, teacher parameter, or inferred label law is accepted.
    """
    y = np.asarray(observed_labels)
    if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] != 3 or not np.isin(y, (-1, 1)).all():
        raise ValueError("observed_labels must be nonempty spins (B,3)")
    index = ((y[:, 0] + 1) // 2 * 4 + (y[:, 1] + 1) // 2 * 2 + (y[:, 2] + 1) // 2).astype(int)
    q = np.zeros((len(y), 8))
    q[np.arange(len(y)), index] = 1.
    return q


def teacher_label_probabilities(Xstar):
    """Enumerated Y|Xstar probabilities, for offline population evaluation only."""
    Xstar = np.asarray(Xstar, dtype=float)
    if Xstar.ndim != 2 or Xstar.shape[0] < 1 or Xstar.shape[1] != 3 or not np.isfinite(Xstar).all():
        raise ValueError("Xstar must be finite nonempty (B,3)")
    return expit(Xstar[:, None, :] * legacy.spin_labels(3)[None, :, :]).prod(axis=2)


def observed_training_batch(counts, observed_labels, indices, *, M):
    """Select finite-N observations with caller-owned integer training indices.

    The signature deliberately excludes Xstar and conditional label weights.
    Repeated indices are legal: they represent empirical-data resampling.
    """
    M = finite._integer(M, "M")
    y = np.asarray(observed_labels)
    observed_label_probabilities(y)
    counts = finite._counts(counts, len(y), M)
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(y)):
        raise ValueError("indices must be a nonempty integer vector in 0..N-1")
    return {"counts": counts[indices].copy(), "observed_labels": y[indices].copy(),
            "data_indices": indices.copy(), "N": len(y),
            "teacher_latent_is_learning_visible": False}


def _probabilities(label_probs, observed_labels, B):
    if observed_labels is not None:
        if label_probs is not None:
            raise ValueError("Choose label_probs or observed_labels, not both")
        label_probs = observed_label_probabilities(observed_labels)
    labels, probabilities = finite._labels(label_probs, B)
    mode = "observed_Y" if observed_labels is not None else "caller_supplied_label_probabilities"
    return labels, probabilities, mode


def full_banks(z, T, counts, C, *, M, epsilon, tau, label_probs=None,
               observed_labels=None, return_per_label=False, private_chunk_size=None):
    """Normalized beta1/lambda0 losses, raw RP, CV RP and legal Price scores.

    T[B,K,3,M] must be oriented draws from sample_oriented_qbeta(beta=1).
    T is held fixed in RP and the proposal is independent of x and C. Analytic
    3M log(1/2) is always included; binomial multiplicity is in count_loss.
    Each endpoint keeps its own legal CV control: controlled_loss absolute
    constants may differ across endpoints, unlike their normalized raw loss.
    """
    M = finite._integer(M, "M")
    _, probabilities, mode = _probabilities(label_probs, observed_labels, len(z))
    result = finite.full_banks(z, T, counts, C, M=M, epsilon=epsilon, tau=tau,
                               beta=1., label_probs=probabilities,
                               nuisance_log_constant=-3*M*math.log(2),
                               return_per_label=return_per_label,
                               private_chunk_size=private_chunk_size)
    result.update({"schema": SCHEMA, "endpoint": "full", "lambda": 0.,
                   "label_mode": mode, "proposal_depends_on_candidate_x": False})
    return result


def collapsed_banks(z, counts, C, *, M, epsilon, tau, inner_order,
                    label_probs=None, observed_labels=None, return_per_label=False,
                    h_chunk_size=4096):
    """Exact private-integration formula for any M; ordinary GH h moments.

    The uniform-particle CV controls a sum of one-coordinate terms. Its
    expectation is independent of C for any fixed observed-data law, including
    correlated teachers. Neither its raw nor Price gradient at I is set to zero.
    This changes the finite-K objective, rather than just its estimator variance.
    """
    M = finite._integer(M, "M")
    epsilon, tau = finite._parameters(epsilon, tau)
    z = np.asarray(z, dtype=float)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or z.shape[2] != 3 or not np.isfinite(z).all():
        raise ValueError("z must be finite nonempty (B,K,3)")
    if np.asarray(C).shape != (3, 3):
        raise ValueError("C must be a positive definite m3 correlation matrix")
    x, dx = legacy.candidate_inputs(z, C)
    counts = finite._counts(counts, len(z), M)
    labels, probabilities, mode = _probabilities(label_probs, observed_labels, len(z))
    h = competition.h_moments(x * epsilon, tau=tau, inner_order=inner_order,
                               chunk_size=h_chunk_size)
    c, b = counts[:, None, :], M - counts[:, None, :]
    coupling = (c*h["log_h"] + b*h["log_complement"]).sum(axis=2)
    score = epsilon*(c*h["dlog_h"] + b*h["dlog_complement"])
    diagonal = epsilon**2*(c*h["d2log_h"] + b*h["d2log_complement"])
    # Reuse the established scorer so all bank/label/coordinate conventions
    # match selection_finite_core.full_banks exactly.
    result = finite._score_banks(x, dx, labels, probabilities, coupling, score,
                                 diagonal, np.zeros(x.shape[:2]), 0., return_per_label)
    combinations = finite.count_log_combinations(counts, M)
    result.update({"schema": SCHEMA, "endpoint": "collapsed", "beta": 1., "lambda": 1.,
                   "M": M, "group_size": M, "epsilon": epsilon.copy(), "tau": tau,
                   "inner_order": int(inner_order), "count_log_combination": combinations,
                   "count_loss": result["loss"] - combinations,
                   "target": "private_integrated_active_prior_finite_K",
                   "private_variables_per_particle": 0, "label_mode": mode,
                   "proposal_depends_on_candidate_x": True,
                   "qualification": "exact_private_integration_formula_with_ordinary_1D_GH; not_interval_certificate"})
    if return_per_label:
        result["per_label"]["count_loss"] = result["per_label"]["loss"] - combinations[:, None]
    return result


def endpoint_banks(endpoint, z, counts, C, *, T=None, M, epsilon, tau,
                   inner_order, label_probs=None, observed_labels=None,
                   return_per_label=False, private_chunk_size=None):
    """Dispatch with shared prior z and each endpoint's own legal private input."""
    kw = dict(M=M, epsilon=epsilon, tau=tau, label_probs=label_probs,
              observed_labels=observed_labels, return_per_label=return_per_label)
    if endpoint == "full":
        if T is None:
            raise ValueError("full endpoint requires oriented beta1 private draws T")
        return full_banks(z, T, counts, C, private_chunk_size=private_chunk_size, **kw)
    if endpoint == "collapsed":
        return collapsed_banks(z, counts, C, inner_order=inner_order, **kw)
    raise ValueError("endpoint must be full or collapsed")


def path_banks(endpoint, z, counts, C_from, C_to, *, points, T=None, M,
               epsilon, tau, inner_order, label_probs=None, observed_labels=None,
               private_chunk_size=None, return_point_gradients=False):
    """Midpoint integration over any PD3 straight correlation-matrix path.

    Per-bank raw path integrals approach the saved-bank direct loss contrast.
    Price path integrals estimate the population contrast; they need not match
    one saved-bank RP contrast. The data law stays fixed along the path. No
    independent-teacher control, or vanishing gradient at I, is assumed.
    """
    points = finite._integer(points, "points")
    start = legacy._matrix(C_from)
    stop = legacy._matrix(C_to)
    if start.shape != (3, 3) or stop.shape != (3, 3):
        raise ValueError("Both path endpoints must be positive definite m3 matrices")
    delta = legacy.correlation_coordinates(stop) - legacy.correlation_coordinates(start)
    kw = dict(T=T, M=M, epsilon=epsilon, tau=tau, inner_order=inner_order,
              label_probs=label_probs, observed_labels=observed_labels,
              private_chunk_size=private_chunk_size)
    left = endpoint_banks(endpoint, z, counts, start, **kw)
    right = endpoint_banks(endpoint, z, counts, stop, **kw)
    keys = ("raw_rp", "cv_rp", "price")
    integrals = {key: np.zeros(len(z)) for key in keys}
    records = {key: [] for key in keys} if return_point_gradients else None
    nodes = (np.arange(points) + .5) / points
    for t in nodes:
        scores = endpoint_banks(endpoint, z, counts, start+t*(stop-start), **kw)
        for key in keys:
            integrals[key] += scores[key] @ delta / points
            if records is not None:
                records[key].append(scores[key])
    result = {"endpoint": endpoint, "points": points, "midpoint_nodes": nodes,
              "coordinate_difference": delta, "pair_order": legacy.pair_order(3),
              "direct_loss_gap": right["shifted_loss"]-left["shifted_loss"],
              "direct_controlled_loss_gap": right["shifted_controlled_loss"]-left["shifted_controlled_loss"],
              "path_raw_rp_gap": integrals["raw_rp"], "path_cv_rp_gap": integrals["cv_rp"],
              "path_price_gap": integrals["price"],
              "start_gradients": {key: left[key] for key in keys},
              "end_gradients": {key: right[key] for key in keys},
              "qualification": "ordinary_midpoint_path_diagnostic; Price_is_population_identity_not_saved_bank_RP"}
    if records is not None:
        result["point_gradients"] = {key: np.stack(value, axis=1) for key, value in records.items()}
    return result
