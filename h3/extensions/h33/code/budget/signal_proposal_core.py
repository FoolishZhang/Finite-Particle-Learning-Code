"""Shared S1/S2 numerical kernels; import performs no sampling or file writes.

Models have m=2 or 3 centered Gaussian logits, unit marginal variances, and
spins {-1,+1}. Correlation derivatives use *physical* off-diagonal coordinates
(01), (02), (12), not unconstrained Cholesky parameters. A bank is the sampling
unit: particles, enumerated labels, candidates and nested n arms are paired.

Price is an expectation identity, not the derivative of a particular saved
sample-average objective. RP differentiates that saved objective. Proposals
below concern only independent nuisance coordinates and never depend on C.
All GH quantities are floating precision diagnostics, not interval certificates.
Actual scientific runs require the separately frozen experiment contract.
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm

SCHEMA = "signal_proposal_core_v1"


def pair_order(m):
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m not in (2, 3):
        raise ValueError("Only m=2 and m=3 are supported")
    return tuple(itertools.combinations(range(m), 2))


def spin_labels(m):
    pair_order(m)
    return np.asarray(list(itertools.product((-1, 1), repeat=m)), dtype=np.int8)


def _matrix(C, *, positive=True):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square m2/m3 matrix")
    pair_order(len(C))
    if not np.isfinite(C).all() or not np.allclose(C, C.T, rtol=0, atol=1e-14):
        raise ValueError("C must be finite and symmetric")
    if not np.allclose(np.diag(C), 1, rtol=0, atol=1e-14):
        raise ValueError("C must have unit diagonal")
    if positive and np.linalg.eigvalsh(C)[0] <= 0:
        raise ValueError("C must be positive definite for pathwise derivatives")
    return C


def correlation_matrix(coordinates, m=3):
    pairs = pair_order(m)
    c = np.asarray(coordinates, dtype=np.float64)
    if c.shape != (len(pairs),) or not np.isfinite(c).all():
        raise ValueError("Physical coordinates have the wrong shape or are nonfinite")
    C = np.eye(m)
    for value, (i, j) in zip(c, pairs):
        C[i, j] = C[j, i] = value
    return C  # May be infeasible: radial_retract is a separate operation.


def correlation_coordinates(C):
    C = _matrix(C, positive=False)
    return np.asarray([C[i, j] for i, j in pair_order(len(C))])


def cholesky_with_derivatives(C):
    """Return L and dL[P,m,m] for a unit perturbation of each symmetric pair."""
    C = _matrix(C)
    L = np.linalg.cholesky(C)
    derivatives = []
    for i, j in pair_order(len(C)):
        E = np.zeros_like(C)
        E[i, j] = E[j, i] = 1
        A = solve_triangular(L, E, lower=True, check_finite=False)
        A = solve_triangular(L, A.T, lower=True, check_finite=False).T
        phi = np.tril(A)
        phi[np.diag_indices_from(phi)] *= .5
        derivatives.append(L @ phi)
    return L, np.asarray(derivatives)


def candidate_inputs(z, C):
    """Reusable x[B,K,m], dx[B,K,m,P], with common independent prior z."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or not np.isfinite(z).all():
        raise ValueError("z must be finite (B,K,m), B,K>=1")
    L, dL = cholesky_with_derivatives(C)
    if z.shape[-1] != len(L):
        raise ValueError("z and C dimensions differ")
    return z @ L.T, np.einsum("bkj,pij->bkip", z, dL, optimize=True)


def posterior_weights(logits):
    """Stable complete pi, 1-pi and pi(1-pi), matching the old Price bridge.

    The dominant complement is summed from the other weights, rather than
    subtracting a probability already rounded to one. Logits may have any
    leading axes; the last axis is particles. Native underflow is not certified.
    """
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim < 1 or not a.shape[-1] or not np.isfinite(a).all():
        raise ValueError("Finite logits with a nonempty particle axis required")
    flat = a.reshape(-1, a.shape[-1])
    maximum = np.argmax(flat, axis=1)
    w = np.exp(flat - flat[np.arange(len(flat)), maximum, None])
    denominator = np.sum(w, axis=1, keepdims=True)
    pi = w / denominator
    complement = 1 - pi
    other = w.copy()
    other[np.arange(len(flat)), maximum] = 0
    complement[np.arange(len(flat)), maximum] = np.sum(other, axis=1) / denominator[:, 0]
    return tuple(v.reshape(a.shape) for v in (pi, complement, pi * complement))


def _labels_and_probabilities(B, m, label_probs, sampled_labels):
    labels = spin_labels(m)
    if sampled_labels is not None:
        if label_probs is not None:
            raise ValueError("Choose enumerated probabilities or sampled labels, not both")
        y = np.asarray(sampled_labels)
        if y.shape != (B, m) or not np.isin(y, (-1, 1)).all():
            raise ValueError("sampled_labels must be spins with shape (B,m)")
        return [(y[:, None, :], np.ones(B))], "one_sampled_label_per_bank"
    q = np.full(len(labels), 1 / len(labels)) if label_probs is None else np.asarray(label_probs, dtype=float)
    if q.shape == (len(labels),):
        q = np.broadcast_to(q, (B, len(labels)))
    if q.shape != (B, len(labels)) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("label_probs must be finite nonnegative (2**m,) or (B,2**m)")
    if not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-13):
        raise ValueError("Label probabilities must sum to one; they are not renormalized")
    return [(y[None, None, :], q[:, i]) for i, y in enumerate(labels)], "enumerated_labels"


def score_correlated_banks(x, dx, s, *, label_probs=None, sampled_labels=None,
                           nuisance_log_constant=0., return_label_details=False,
                           return_per_label=False):
    """Loss and both gradient estimators for already transformed prior banks.

    s[B,K] contains nuisance logweights *without* their common normalizer.
    Supply that normalizer as nuisance_log_constant for an absolute loss.
    shifted_loss removes max(s) bankwise and is for stable paired contrasts,
    not a separately normalized population objective. Labels must be generated
    independently of candidate prior particles. No statistic treats K or L as
    independent repetitions.

    controlled_loss adds sum(noise_pi*A_C), where A_C is active log-likelihood
    and noise_pi uses nuisance weights alone. Its derivative is cv_rp; it is NOT
    the original raw SAA scalar. The added term has C-independent expectation
    m*E_phi log sigma(G) for each label because active marginal variances are
    fixed. raw_rp/rp, cv_rp and Price therefore share a population derivative.
    """
    x, dx, s = (np.asarray(a, dtype=np.float64) for a in (x, dx, s))
    if x.ndim != 3 or min(x.shape[:2]) < 1:
        raise ValueError("x must be (B,K,m)")
    B, K, m = x.shape
    pairs = pair_order(m)
    if dx.shape != (B, K, m, len(pairs)) or s.shape != (B, K):
        raise ValueError("dx/s shapes differ from x")
    if not all(np.isfinite(a).all() for a in (x, dx, s)):
        raise ValueError("Nonfinite bank input")
    constant = np.broadcast_to(np.asarray(nuisance_log_constant, dtype=float), (B,))
    if not np.isfinite(constant).all():
        raise ValueError("Nonfinite common log normalizer")
    labels, mode = _labels_and_probabilities(B, m, label_probs, sampled_labels)
    shift = np.max(s, axis=1)
    centered = s - shift[:, None]
    shifted_loss = np.zeros(B)
    rp, price, cv_rp = (np.zeros((B, len(pairs))) for _ in range(3))
    shifted_controlled_loss = np.zeros(B)
    noise_pi, _, _ = posterior_weights(centered)
    noise_maximum = np.argmax(noise_pi, axis=1)
    overlap, ess = np.zeros(B), np.zeros(B)
    details = []
    for y, q in labels:
        signed = x * y
        active = log_expit(signed).sum(axis=2)
        logits = centered + active
        pi, _, gamma = posterior_weights(logits)
        residual = y * expit(-signed)
        loss_y = math.log(K) - logsumexp(logits, axis=1)
        rp_y = -np.einsum("bk,bkm,bkmp->bp", pi, residual, dx, optimize=True)
        difference = noise_pi - pi
        # Both probabilities can round to one in a dominant-weight bank.
        # Reconstruct one component from the exact zero-sum identity instead.
        difference[np.arange(B), noise_maximum] = 0.
        difference[np.arange(B), noise_maximum] = -difference.sum(axis=1)
        cv_y = np.einsum("bk,bkm,bkmp->bp", difference, residual, dx, optimize=True)
        controlled_y = loss_y + np.sum(noise_pi * active, axis=1)
        price_y = np.stack([-np.sum(gamma * residual[:, :, i] * residual[:, :, j], axis=1)
                            for i, j in pairs], axis=-1)
        shifted_loss += q * loss_y
        rp += q[:, None] * rp_y
        cv_rp += q[:, None] * cv_y
        shifted_controlled_loss += q * controlled_y
        price += q[:, None] * price_y
        overlap += q * gamma.sum(axis=1)
        ess += q / np.sum(pi * pi, axis=1)
        if return_label_details or return_per_label:
            details.append({"shifted_loss": loss_y, "shifted_controlled_loss": controlled_y,
                            "rp": rp_y, "cv_rp": cv_y, "price": price_y,
                            "overlap": gamma.sum(axis=1), "pi": pi})
    result = {"loss": shifted_loss - shift - constant, "shifted_loss": shifted_loss,
              "rp": rp, "raw_rp": rp, "cv_rp": cv_rp, "price": price,
              "controlled_loss": shifted_controlled_loss - shift - constant,
              "shifted_controlled_loss": shifted_controlled_loss,
              "overlap": overlap, "ess": ess,
              "bank_shift": shift, "nuisance_log_constant": constant.copy(),
              "label_mode": mode, "pair_order": pairs}
    if return_label_details:
        result["label_details"] = details
    if return_per_label:
        result["per_label"] = {key: np.stack([d[key] for d in details], axis=1)
                               for key in ("shifted_loss", "shifted_controlled_loss", "rp", "cv_rp", "price", "overlap")}
        result["per_label"]["raw_rp"] = result["per_label"]["rp"]
        result["per_label"]["loss"] = result["per_label"]["shifted_loss"] - shift[:, None] - constant[:, None]
        result["per_label"]["controlled_loss"] = result["per_label"]["shifted_controlled_loss"] - shift[:, None] - constant[:, None]
    return result


def evaluate_banks(z, s, C, **kwargs):
    x, dx = candidate_inputs(z, C)
    return score_correlated_banks(x, dx, s, **kwargs)


def evaluate_panel(z, s_prefix, candidates, *, label_probs=None, sampled_labels=None,
                   nuisance_log_constant=0.):
    """Return arrays with axes (bank,candidate,n[,physical_coordinate]).

    Prior z and each nuisance arm are reused across every candidate and label.
    Each candidate's Cholesky and derivatives are constructed once. Common
    normalizers may be scalar, (N,), or (B,N). Input arms are not regenerated.
    """
    z, s = np.asarray(z, dtype=float), np.asarray(s_prefix, dtype=float)
    if z.ndim != 3 or s.ndim != 3 or s.shape[0] != z.shape[0] or s.shape[2] != z.shape[1]:
        raise ValueError("Need z[B,K,m], s_prefix[B,N,K]")
    Cs = np.asarray(candidates, dtype=float)
    if Cs.ndim != 3 or len(Cs) < 1 or Cs.shape[1:] != (z.shape[2], z.shape[2]) or s.shape[1] < 1:
        raise ValueError("Nonempty candidates[M,m,m] and n arms required")
    constants = np.broadcast_to(np.asarray(nuisance_log_constant, dtype=float), s.shape[:2])
    output = {}
    for ci, C in enumerate(Cs):
        x, dx = candidate_inputs(z, C)
        for ni in range(s.shape[1]):
            result = score_correlated_banks(x, dx, s[:, ni, :], label_probs=label_probs,
                                            sampled_labels=sampled_labels,
                                            nuisance_log_constant=constants[:, ni])
            for key in ("loss", "shifted_loss", "controlled_loss", "shifted_controlled_loss",
                        "rp", "raw_rp", "cv_rp", "price", "overlap", "ess"):
                value = result[key]
                if key not in output:
                    output[key] = np.empty((len(z), len(Cs), s.shape[1]) + value.shape[1:])
                output[key][:, ci, ni] = value
    return output


def contrast_banks(z, s, C, reference_C=None, **kwargs):
    """Per-bank loss(C)-loss(reference), using paired centered log-sum-exp."""
    C = _matrix(C)
    reference_C = np.eye(len(C)) if reference_C is None else reference_C
    left, right = evaluate_banks(z, s, C, **kwargs), evaluate_banks(z, s, reference_C, **kwargs)
    return left["shifted_loss"] - right["shifted_loss"]


def price_path_gap(z, s, C, q, t):
    """Per-bank Price radial integral estimating F(C)-F(I).

    Caller supplies independently uniform t[B] in [0,1] and fixed label
    probabilities q. Only Price is computed. Batched Cholesky couples all
    points to z. Uniform-t expectation supplies the path integral; this is not
    a pathwise derivative or direct SAA gap. Common nuisance normalization and
    bankwise logweight shifts cancel, including any near-posterior drift.
    """
    C = _matrix(C)
    z, s, t = np.asarray(z, dtype=float), np.asarray(s, dtype=float), np.asarray(t, dtype=float)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or z.shape[2] != len(C):
        raise ValueError("z must be nonempty (B,K,m) matching C")
    if s.shape != z.shape[:2] or t.shape != (len(z),):
        raise ValueError("Need s[B,K], t[B]")
    if not all(np.isfinite(a).all() for a in (z, s, t)) or np.any(t < 0) or np.any(t > 1):
        raise ValueError("Finite inputs and t in [0,1] required")
    identity = np.eye(len(C))
    path = identity + t[:, None, None] * (C - identity)
    L = np.linalg.cholesky(path)
    x = np.einsum("bij,bkj->bki", L, z, optimize=True)
    centered = s - s.max(axis=1, keepdims=True)
    coefficients = correlation_coordinates(C)
    out = np.zeros(len(z))
    labels, _ = _labels_and_probabilities(len(z), len(C), q, None)
    for y, probability in labels:
        signed = x * y
        _, _, gamma = posterior_weights(centered + log_expit(signed).sum(axis=2))
        residual = y * expit(-signed)
        mark = np.zeros_like(gamma)
        for c, (i, j) in zip(coefficients, pair_order(len(C))):
            mark += c * residual[:, :, i] * residual[:, :, j]
        out -= probability * np.sum(gamma * mark, axis=1)
    return out


@lru_cache(maxsize=32)
def _gaussian_rule(order):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("GH order must be an integer >=2")
    x, w = roots_hermitenorm(int(order))
    w /= np.sqrt(2 * np.pi)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def _beta(beta):
    beta = float(beta)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")
    return beta


def proposal_moments(beta, order=80, max_moment=6):
    """GH diagnostics for oriented T=uG under q_beta, not raw Gaussian G.

    logweights are log Z_beta + (1-beta)*log sigma(T). Endpoint normalizers
    Z0=1 and Z1=1/2 are imposed analytically. v_beta is variance of log sigma,
    not variance of the importance logweight, which is (1-beta)**2*v_beta.
    """
    beta = _beta(beta)
    if isinstance(max_moment, bool) or not isinstance(max_moment, (int, np.integer)) or not 2 <= max_moment <= 8:
        raise ValueError("max_moment must be an integer from 2 through 8")
    x, w = _gaussian_rule(order)
    ell = log_expit(x)
    unnormalized = w * np.exp(beta * ell)
    computed_Z = unnormalized.sum()
    Z = 1. if beta == 0 else .5 if beta == 1 else float(computed_Z)
    qw = unnormalized / Z
    mu = float(qw @ ell)
    centered = ell - mu
    raw = [float(qw @ ell**j) for j in range(1, max_moment + 1)]
    central = [float(qw @ centered**j) for j in range(1, max_moment + 1)]
    variance = float(qw @ (centered * centered))
    return {"beta": beta, "order": int(order), "Z_beta": Z, "log_Z_beta": math.log(Z),
            "mu_beta": mu, "v_beta": variance, "raw_log_moments": raw,
            "central_log_moments": central,
            "residual_variance_per_coordinate": (1 - beta)**2 * variance,
            "absolute_logweight_mean_per_coordinate": math.log(Z) + (1 - beta) * mu,
            "expected_importance_weight": float(qw @ (Z * np.exp((1 - beta) * ell))),
            "GH_normalizer_endpoint_residual": float(computed_Z - Z),
            "qualification": "floating_GH_precision_diagnostic_not_certificate"}


def proposal_diagnostics(beta, orders=(48, 80), max_moment=6):
    if len(orders) < 2 or len(set(orders)) != len(orders):
        raise ValueError("At least two distinct GH orders required")
    values = [proposal_moments(beta, order, max_moment) for order in orders]
    discrepancies = {}
    for key in ("Z_beta", "mu_beta", "v_beta", "expected_importance_weight"):
        discrepancies[key] = max(abs(v[key] - values[-1][key]) for v in values[:-1])
    for key in ("raw_log_moments", "central_log_moments"):
        discrepancies[key] = np.max(np.abs(np.asarray([v[key] for v in values[:-1]]) - values[-1][key]), axis=0).tolist()
    return {"values": values, "order_discrepancies": discrepancies,
            "qualification": "order_agreement_is_not_a_strict_quadrature_error_bound"}


def sample_qbeta(rng, size, beta, *, u=1, proposal_batch_size=1_000_000, max_proposals=None):
    """Exact rejection from phi with acceptance sigma(uG)**beta.

    accepted counts *all* accepted proposals, including a final unused surplus;
    used counts returned samples. proposed includes all generated proposals.
    At beta=0 there is no rejection and no uniform draw. Endpoints are exact
    distributions subject to the RNG/native arithmetic qualification.
    """
    beta = _beta(beta)
    if isinstance(size, bool):
        raise ValueError("size must contain nonnegative integers")
    try:
        shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(size)
    except TypeError as exc:
        raise ValueError("size must contain nonnegative integers") from exc
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in shape):
        raise ValueError("size must contain nonnegative integers")
    count = math.prod(shape)
    if isinstance(proposal_batch_size, bool) or not isinstance(proposal_batch_size, (int, np.integer)) or proposal_batch_size < 1:
        raise ValueError("proposal_batch_size must be a positive integer")
    signs = np.broadcast_to(np.asarray(u), shape)
    if not np.isin(signs, (-1, 1)).all():
        raise ValueError("u must contain spins")
    if max_proposals is not None and (isinstance(max_proposals, bool) or not isinstance(max_proposals, (int, np.integer)) or max_proposals < count):
        raise ValueError("max_proposals must be an integer >= requested count")
    if beta == 0:
        values = rng.standard_normal(shape)
        return values * signs, {"proposed": count, "accepted": count, "used": count,
                               "discarded_accepted": 0, "acceptance_fraction": 1. if count else None}
    values = np.empty(count)
    used = proposed = accepted = 0
    while used < count:
        batch = min(int(proposal_batch_size), max(32, 2 * (count - used)))
        if max_proposals is not None:
            batch = min(batch, max_proposals - proposed)
            if batch <= 0:
                raise RuntimeError(f"Rejection budget exhausted: proposed={proposed}, accepted={accepted}, used={used}")
        g = rng.standard_normal(batch)
        uniforms = rng.random(batch)
        with np.errstate(divide="ignore"):
            log_uniforms = np.log(uniforms)
        mask = log_uniforms < beta * log_expit(g)
        good = g[mask]
        take = min(len(good), count - used)
        values[used:used + take] = good[:take]
        used += take
        proposed += batch
        accepted += len(good)
    return values.reshape(shape) * signs, {"proposed": proposed, "accepted": accepted, "used": used,
                                           "discarded_accepted": accepted - used,
                                           "acceptance_fraction": accepted / proposed if proposed else None}


def _n_values(n_values):
    values = tuple(n_values)
    if not values or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 for n in values):
        raise ValueError("n_values must be nonempty nonnegative integer checkpoints")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("n_values must be strictly increasing")
    return values


def nuisance_sums(rng, banks, particles, n_values, beta, *, block_coords=64,
                   log_Z_beta=None, mu_beta=None, proposal_batch_size=1_000_000):
    """Block-streamed nested nuisance logweights and separate normalizers.

    reduced_logweights[B,N,K]=(1-beta)*sum log sigma(T); common_log_constant[N]
    is n*log Z_beta. Their sum is the absolute importance logweight. beta=1
    returns bitwise zeros and n*log(1/2), with no RNG draw or rejection at all.
    Nested means prefixes *within this invocation*. Changing checkpoints or
    block sizes may change the RNG stream/layout and is not bitwise equivalent.
    """
    beta, ns = _beta(beta), _n_values(n_values)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (banks, particles, block_coords)):
        raise ValueError("banks, particles and block_coords must be positive integers")
    if beta in (0, 1):
        exact = 0. if beta == 0 else -math.log(2)
        if log_Z_beta is not None and abs(float(log_Z_beta) - exact) > 1e-14:
            raise ValueError("Endpoint log normalizer must match the analytic value")
        log_Z_beta = exact
    elif log_Z_beta is None:
        raise ValueError("Interior beta requires an explicitly chosen GH log_Z_beta")
    if not np.isfinite(log_Z_beta) or not -math.log(2) <= float(log_Z_beta) <= 0:
        raise ValueError("Invalid proposal log normalizer")
    result = np.zeros((banks, len(ns), particles))
    ledger = {"proposed": 0, "accepted": 0, "used": 0, "discarded_accepted": 0}
    running = np.zeros((banks, particles))
    position = 0
    if beta != 1:
        for ni, n in enumerate(ns):
            while position < n:
                chunk = min(block_coords, n - position)
                t, counts = sample_qbeta(rng, (banks, particles, chunk), beta,
                                        proposal_batch_size=proposal_batch_size)
                running += (1 - beta) * log_expit(t).sum(axis=-1)
                for key in ledger:
                    ledger[key] += counts[key]
                position += chunk
            result[:, ni, :] = running
    ledger["acceptance_fraction"] = ledger["accepted"] / ledger["proposed"] if ledger["proposed"] else None
    if mu_beta is None:
        mu_beta = proposal_moments(beta)["mu_beta"]
    if not np.isfinite(mu_beta):
        raise ValueError("Finite mu_beta required for noise centering")
    mean = np.asarray(ns, dtype=float) * ((1 - beta) * float(mu_beta))
    normalizer = np.asarray(ns, dtype=float) * float(log_Z_beta)
    return {"reduced_logweights": result,
            "common_log_constant": normalizer,
            "centered_logweights": result - mean[None, :, None],
            "centered_log_constant": normalizer + mean,
            "centering_mean": mean, "mu_beta": float(mu_beta),
            "n_values": ns, "beta": beta, "sampling": ledger,
            "oriented_u_g": True, "proposal": "exact_rejection_phi_sigma_power"}


def gaussian_bridge_sums(rng, banks, particles, n_values, variance_per_coordinate,
                         *, mean_per_coordinate=0., log_normalizer_per_coordinate=0.):
    """Independent Gaussian increments, paired at the requested nested n arms.

    This is a moment-matched residual-noise surrogate, NOT a normalized
    importance proposal. Constants permit a matched absolute logweight mean,
    but do not imply E(weight)=1/2. Use centered contrasts for bridge comparison.
    """
    ns = _n_values(n_values)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (banks, particles)):
        raise ValueError("banks and particles must be positive integers")
    v, mean, normalizer = map(float, (variance_per_coordinate, mean_per_coordinate, log_normalizer_per_coordinate))
    if not np.isfinite([v, mean, normalizer]).all() or v < 0:
        raise ValueError("Finite moments with nonnegative variance required")
    result, running = np.zeros((banks, len(ns), particles)), np.zeros((banks, particles))
    old = 0
    for ni, n in enumerate(ns):
        if n > old and v > 0:
            running += math.sqrt((n - old) * v) * rng.standard_normal((banks, particles))
        result[:, ni, :] = running
        old = n
    return {"reduced_logweights": result,
            "centered_logweights": result,
            "common_log_constant": np.asarray(ns) * (mean + normalizer),
            "centered_log_constant": np.asarray(ns) * (mean + normalizer),
            "n_values": ns, "variance_per_coordinate": v,
            "qualification": "Gaussian_surrogate_not_exact_importance_proposal"}


def pairwise_kappa(rho, order=64, *, derivative=False):
    """E[tanh(X1/2)tanh(X2/2)]; derivative by Gaussian Price identity."""
    rho = float(rho)
    if not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("Pair correlation must lie in [-1,1]")
    g, w = _gaussian_rule(order)
    x = g[:, None]
    y = rho * x + math.sqrt(max(0., 1 - rho * rho)) * g[None, :]
    weights = w[:, None] * w[None, :]
    if derivative:
        integrand = 4 * expit(x) * expit(-x) * expit(y) * expit(-y)
        return float(np.sum(weights * integrand))
    if rho == 0:
        return 0.
    return float(np.sum(weights * np.tanh(x / 2) * np.tanh(y / 2)))


def observation_law(C, order=64):
    """Exact spin-algebra form for m2/m3, with GH2 approximated pair moments.

    Centered joint-Gaussian reflection kills odd spin moments. Thus for m=3
    Q(y)=2^-3*(1+sum kappa_ij*y_i*y_j); no third-order moment is discarded.
    This law is independently integrated, not evaluated from a particle bank.
    """
    C = _matrix(C, positive=False)
    if np.linalg.eigvalsh(C)[0] < -1e-12:
        raise ValueError("Observation law requires a positive semidefinite C")
    labels, pairs = spin_labels(len(C)), pair_order(len(C))
    kappas = np.asarray([pairwise_kappa(C[i, j], order) for i, j in pairs])
    characters = np.asarray([labels[:, i] * labels[:, j] for i, j in pairs]).T
    q = (1 + characters @ kappas) / len(labels)
    if np.any(q <= 0) or not np.isfinite(q).all():
        raise FloatingPointError("GH-derived observation probabilities must stay positive")
    return {"labels": labels, "probabilities": q, "kappa": kappas,
            "pair_order": pairs, "order": int(order),
            "normalization_residual": float(q.sum() - 1),
            "qualification": "exact_spin_identity_with_floating_GH_pair_moments"}


def observation_risk(C, teacher_probabilities, order=64):
    """Independent KL and physical gradient; teacher is used for evaluation only."""
    law = observation_law(C, order)
    q, y = law["probabilities"], law["labels"]
    p = np.asarray(teacher_probabilities, dtype=float)
    if p.shape != q.shape or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum() - 1) > 1e-13:
        raise ValueError("Teacher must be a normalized probability vector in lexicographic spin order")
    mask = p > 0
    risk = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
    gradient = np.asarray([-np.sum((p / q) * y[:, i] * y[:, j]) / len(y) *
                           pairwise_kappa(C[i, j], order, derivative=True)
                           for i, j in law["pair_order"]])
    return {"kl": risk, "gradient": gradient, "candidate_probabilities": q,
            "qualification": law["qualification"]}


def observation_kl(teacher_C, candidate_C, orders=(48, 80)):
    if len(orders) < 2 or len(set(orders)) != len(orders):
        raise ValueError("Independent risk precision diagnostic needs distinct GH orders")
    records = []
    for order in orders:
        teacher = observation_law(teacher_C, order)
        value = observation_risk(candidate_C, teacher["probabilities"], order)
        records.append({"order": int(order), **value})
    return {"values": records,
            "kl_order_discrepancy": max(abs(v["kl"] - records[-1]["kl"]) for v in records[:-1]),
            "qualification": "independent_risk_evaluation_floating_GH_not_strict_certificate"}


def radial_retract(C, delta):
    """Retract along I+t(C-I) to lambda_min>=delta; NOT Euclidean projection.

    It preserves unit diagonal/symmetry and the ray, and changes every physical
    coordinate by the same t. A few nextafter steps protect the computed floor
    against rounding. delta>0 is needed by Cholesky-based subsequent scoring.
    """
    C = _matrix(C, positive=False)
    delta = float(delta)
    if not np.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError("delta must lie in [0,1)")
    smallest = float(np.linalg.eigvalsh(C)[0])
    t = 1. if smallest >= delta else (1 - delta) / (1 - smallest)
    result = np.eye(len(C)) + t * (C - np.eye(len(C)))
    for _ in range(32):
        if np.linalg.eigvalsh(result)[0] >= delta:
            return {"matrix": result, "t": t, "lambda_min": float(np.linalg.eigvalsh(result)[0]),
                    "method": "identity_ray_retraction_not_Euclidean_projection"}
        t = float(np.nextafter(t, 0.))
        result = np.eye(len(C)) + t * (C - np.eye(len(C)))
    raise FloatingPointError("Could not protect radial spectral floor against rounding")
