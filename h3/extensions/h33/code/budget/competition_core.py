"""Manufactured-testable competition observation kernels (m=3, M=4/8/16).

Every private observation retains its own latent variable. Counts are in
lexicographic product order; active labels use the legacy (-1,+1) spin order.
Imports never sample, choose seeds, write files or launch a scientific panel.
GH results are ordinary floating diagnostics, not interval certificates.
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import math
import numpy as np
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm
import signal_proposal_core as legacy

SCHEMA = "competition_core_v1"
ACTIVE_DIMENSION = 3
ALLOWED_GROUP_SIZES = (4, 8, 16)


def _group_size(M):
    M = _integer(M, "M")
    if M not in ALLOWED_GROUP_SIZES:
        raise ValueError("M must be 4, 8, or 16")
    return M


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _parameters(epsilon, tau):
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if epsilon.shape != (3,) or not np.isfinite(epsilon).all():
        raise ValueError("epsilon must be three finite fixed group couplings")
    if isinstance(tau, bool) or not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite nonnegative fixed parameter")
    return epsilon, float(tau)


def _beta(beta):
    if isinstance(beta, bool) or not np.isscalar(beta) or not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    return float(beta)


def _counts(counts, B, M):
    counts = np.asarray(counts)
    if counts.shape != (B, 3) or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > M):
        raise ValueError(f"counts must be an integer array (B,3) with entries 0..{M}")
    return counts.astype(np.int64, copy=False)


def _labels(q, B):
    labels = legacy.spin_labels(3)
    q = np.full(8, 1 / 8) if q is None else np.asarray(q, dtype=float)
    if q.shape == (8,):
        q = np.broadcast_to(q, (B, 8))
    if q.shape != (B, 8) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("q_y_given_counts must be nonnegative finite (8,) or (B,8)")
    if not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-13):
        raise ValueError("Conditional label probabilities must sum to one; no renormalization")
    return labels, q


def _active(z, counts, C, q, M):
    z = np.asarray(z, dtype=float)
    if z.ndim != 3 or min(z.shape[:2]) < 1 or z.shape[2] != 3 or not np.isfinite(z).all():
        raise ValueError("z must be finite nonempty (B,K,3)")
    if np.asarray(C).shape != (3, 3):
        raise ValueError("C must be a positive definite unit-diagonal m3 correlation matrix")
    counts = _counts(counts, len(z), M)
    labels, q = _labels(q, len(z))
    x, dx = legacy.candidate_inputs(z, C)
    return x, dx, counts, labels, q


def canonical_spins(counts, M):
    """u[B,3,M]: first c entries are +1, remaining entries -1."""
    M = _group_size(M)
    counts = np.asarray(counts)
    counts = _counts(counts, len(counts), M)
    return np.where(np.arange(M)[None, None, :] < counts[:, :, None], 1, -1).astype(np.int8)


def _normal_rule(order):
    order = _integer(order, "GH order", 2)
    return _normal_rule_cached(order)


@lru_cache(maxsize=32)
def _normal_rule_cached(order):
    nodes, weights = roots_hermitenorm(order)
    weights = weights / math.sqrt(2 * math.pi)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def h_moments(a, *, tau, inner_order, chunk_size=4096):
    """h(a)=E_phi sigma(tau*G+a), its derivatives and stable log derivatives.

    Positive and negative probabilities are integrated separately, rather
    than obtaining a tiny complement by subtraction. Neither is normalized.
    All derivative fields are with respect to a, not x or tau.
    """
    _, tau = _parameters([0, 0, 0], tau)
    chunk_size = _integer(chunk_size, "chunk_size")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("h arguments must be finite")
    g, w = _normal_rule(inner_order)
    names = ("h", "complement", "first", "second", "log_h", "log_complement",
             "dlog_h", "dlog_complement", "d2log_h", "d2log_complement")
    out = {name: np.empty(a.size) for name in names}
    logw = np.log(w)[None, :]
    flat = a.reshape(-1)
    for start in range(0, a.size, chunk_size):
        stop = min(start + chunk_size, a.size)
        logits = flat[start:stop, None] + tau * g[None, :]
        if not np.isfinite(logits).all():
            raise FloatingPointError("h quadrature logits overflowed")
        positive, negative = expit(logits), expit(-logits)
        lp = logw + log_expit(logits)
        lm = logw + log_expit(-logits)
        logh, logm = logsumexp(lp, axis=1), logsumexp(lm, axis=1)
        tilted_p, tilted_m = np.exp(lp - logh[:, None]), np.exp(lm - logm[:, None])
        dlogh = np.sum(tilted_p * negative, axis=1)
        dlogm = -np.sum(tilted_m * positive, axis=1)
        curvature = positive * negative
        values = {
            "h": np.exp(logh), "complement": np.exp(logm),
            "first": curvature @ w,
            "second": (curvature * (1 - 2 * positive)) @ w,
            "log_h": logh, "log_complement": logm,
            "dlog_h": dlogh, "dlog_complement": dlogm,
            "d2log_h": np.sum(tilted_p * (negative**2 - curvature), axis=1) - dlogh**2,
            "d2log_complement": np.sum(tilted_m * (positive**2 - curvature), axis=1) - dlogm**2,
        }
        for name in names:
            out[name][start:stop] = values[name]
    return {name: value.reshape(a.shape) for name, value in out.items()}


def _bank_scores(x, dx, labels, q, coupling, coupling_score, coupling_diag,
                 baseline, nuisance_log_constant, return_per_label, target):
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
        # Avoid loss of a dominant component through 1-1 cancellation.
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
                            "pi": pi, "active_log_likelihood": active})
    out["loss"] = out["shifted_loss"] - shift - constant
    out["controlled_loss"] = out["shifted_controlled_loss"] - shift - constant
    out["rp"] = out["raw_rp"]
    out.update({"S_base": baseline, "noise_pi": nu, "bank_shift": shift,
                "nuisance_log_constant": constant.copy(), "labels": labels,
                "pair_order": legacy.pair_order(3), "target": target,
                "combination_constants_in_loss": False})
    if return_per_label:
        per = {key: np.stack([record[key] for record in details], axis=1) for key in details[0]}
        per["loss"] = per["shifted_loss"] - shift[:, None] - constant[:, None]
        per["controlled_loss"] = per["shifted_controlled_loss"] - shift[:, None] - constant[:, None]
        per["rp"] = per["raw_rp"]
        out["per_label"] = per
    return out


def full_banks(z, T, counts, C, *, M, epsilon, tau, beta,
               q_y_given_counts=None, nuisance_log_constant=0., return_per_label=False):
    """Full-private-variable importance objective and raw/CV/Price gradients.

    T[B,K,3,M] is oriented u*G under q_beta,0. Draws remain independent per
    observation, even when observations are stored as counts. The caller
    supplies 3*M*log(Z_beta,tau) separately as nuisance_log_constant for an
    absolute normalized loss; omitting it does not change candidate contrasts.
    """
    M = _group_size(M)
    epsilon, tau = _parameters(epsilon, tau)
    beta = _beta(beta)
    x, dx, counts, labels, q = _active(z, counts, C, q_y_given_counts, M)
    T = np.asarray(T, dtype=float)
    if T.shape != x.shape[:2] + (3, M) or not np.isfinite(T).all():
        raise ValueError("T must be finite oriented private draws (B,K,3,M)")
    spins = canonical_spins(counts, M)[:, None, :, :]
    base_logits = tau * T
    shifted = base_logits + epsilon[None, None, :, None] * spins * x[:, :, :, None]
    if not np.isfinite(shifted).all():
        raise FloatingPointError("Full-private logit overflow")
    base_loglik = log_expit(base_logits)
    baseline = (1 - beta) * base_loglik.sum(axis=(2, 3))
    coupling = (log_expit(shifted) - base_loglik).sum(axis=(2, 3))
    coupling_score = (epsilon[None, None, :, None] * spins * expit(-shifted)).sum(axis=3)
    coupling_diag = -(epsilon[None, None, :, None]**2 * expit(shifted) * expit(-shifted)).sum(axis=3)
    result = _bank_scores(x, dx, labels, q, coupling, coupling_score, coupling_diag,
                          baseline, nuisance_log_constant, return_per_label,
                          "full_private_variables_fixed_epsilon0_qbeta_proposal")
    result.update({"beta": beta, "tau": tau, "epsilon": epsilon.copy(), "private_variables_per_particle": 3 * M, "group_size": M,
                   "proposal_is_true_new_conditional_posterior": tau == 0 or (not np.any(epsilon) and beta == 1)})
    result["count_log_combination"] = count_log_combinations(counts, M)
    return result


def collapsed_banks(z, counts, C, *, M, epsilon, tau, inner_order,
                    q_y_given_counts=None, return_per_label=False):
    """Private variables integrated out, preserving counts information.

    The finite-K objective changes; this is not the same full objective with
    an unbiased lower-variance gradient. Uniform nu is a valid separate CV.
    Count binomial-combination constants are intentionally omitted from loss.
    """
    M = _group_size(M)
    epsilon, tau = _parameters(epsilon, tau)
    x, dx, counts, labels, q = _active(z, counts, C, q_y_given_counts, M)
    h = h_moments(x * epsilon, tau=tau, inner_order=inner_order)
    c = counts[:, None, :]
    b = M - c
    coupling = (c * h["log_h"] + b * h["log_complement"]).sum(axis=2)
    score = epsilon * (c * h["dlog_h"] + b * h["dlog_complement"])
    diagonal = epsilon**2 * (c * h["d2log_h"] + b * h["d2log_complement"])
    result = _bank_scores(x, dx, labels, q, coupling, score, diagonal,
                          np.zeros(x.shape[:2]), 0., return_per_label,
                          "private_integrated_active_prior_finite_K")
    result.update({"group_size": M, "inner_order": int(inner_order), "tau": tau, "epsilon": epsilon.copy(),
                   "qualification": "ordinary_1D_GH_h_and_derivatives_not_interval_certificate"})
    result["count_log_combination"] = count_log_combinations(counts, M)
    return result


def count_log_combinations(counts, M):
    """Candidate-independent count multiplicity, omitted from bank loss."""
    M = _group_size(M)
    counts = _counts(counts, len(counts), M)
    values = np.log(np.asarray([math.comb(M, c) for c in range(M + 1)], dtype=float))
    return values[counts].sum(axis=1)


def h_probabilities(a, *, tau, inner_order, chunk_size=4096):
    """Positive GH sums of sigma(logit) and sigma(-logit), separately.

    This probability-only path avoids log derivative work. Complements are
    never formed by subtraction. Floating underflow is retained, not clipped.
    Scores continue to use the stable tilted-log-GH h_moments implementation.
    """
    _, tau = _parameters([0, 0, 0], tau)
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("a must be finite")
    chunk_size = _integer(chunk_size, "chunk_size")
    g, w = _normal_rule(inner_order)
    flat = a.ravel()
    h, hc = np.empty(len(flat)), np.empty(len(flat))
    for start in range(0, len(flat), chunk_size):
        logits = flat[start:start + chunk_size, None] + tau * g
        h[start:start + chunk_size] = expit(logits) @ w
        hc[start:start + chunk_size] = expit(-logits) @ w
    return {"h": h.reshape(a.shape), "complement": hc.reshape(a.shape),
            "qualification": "ordinary_positive_GH_separate_probability_and_complement_sums"}


def state_mapping(M):
    """Lex counts, spin labels, and their [2(M+1)]^3 cell indices."""
    M = _group_size(M)
    counts = np.asarray(list(itertools.product(range(M + 1), repeat=3)), dtype=np.int64)
    labels = legacy.spin_labels(3)
    group_states = np.asarray(list(itertools.product((-1, 1), range(M + 1))), dtype=np.int64)
    indices = counts[:, None, :] + ((labels[None, :, :] + 1) // 2) * (M + 1)
    return {"counts_states": counts, "labels": labels, "group_states": group_states,
            "state_indices": indices,
            "ordering": f"counts product(0..{M})^3; labels product(-1,+1)^3; per-axis (-1,c),(+1,c)"}


def _count_y_table(pmf, M):
    mapping = state_mapping(M)
    indices = mapping["state_indices"]
    return pmf[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]], mapping


def _entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def _binomial(h, hc, M):
    c = np.arange(M + 1)
    choose = np.asarray([math.comb(M, int(v)) for v in c], dtype=float)
    return choose * h[..., None]**c * hc[..., None]**(M - c)


def _group_factors(x, epsilon, tau, M, inner_order, h_chunk):
    h = h_probabilities(epsilon * x, tau=tau, inner_order=inner_order, chunk_size=h_chunk)
    bins = _binomial(h["h"], h["complement"], M)
    factors = np.concatenate((expit(-x)[..., None] * bins,
                              expit(x)[..., None] * bins), axis=-1)
    return factors, bins


def joint_observation_law(C, *, M, epsilon, tau, outer_order, inner_order,
                          pair_chunk=64, h_chunk=4096):
    """Raw full-alphabet PMF, 3D active GH and 1D private GH.

    Cholesky triangular conditioning integrates group 3 over z3 first.
    The final positive contractions use BLAS over only the z1,z2 pairs.
    No PMF normalization, negative-value clipping or MI clipping is applied.
    Group marginal invariance is analytic; finite-GH discrepancies are exposed.
    """
    M = _group_size(M)
    epsilon, tau = _parameters(epsilon, tau)
    if np.asarray(C).shape != (3, 3):
        raise ValueError("Joint observation law requires m3")
    L, _ = legacy.cholesky_with_derivatives(C)
    pair_chunk = _integer(pair_chunk, "pair_chunk")
    h_chunk = _integer(h_chunk, "h_chunk")
    g, w = _normal_rule(outer_order)
    N, S = len(g), 2 * (M + 1)
    marginals = []
    expected_conditional_entropy = 0.
    local_mass_error = 0.
    for e in epsilon:
        factors, bins = _group_factors(g, e, tau, M, inner_order, h_chunk)
        marginals.append(w @ factors)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(np.where(bins > 0, bins * np.log(bins), 0.), axis=1)
        expected_conditional_entropy += float(w @ entropy)
        local_mass_error = max(local_mass_error, float(np.max(np.abs(factors.sum(axis=1) - 1))))
    marginals = np.asarray(marginals)
    independent = np.array_equal(np.asarray(C), np.eye(3))
    if independent:
        pmf = np.einsum("a,b,c->abc", *marginals)
        argument_count = 3 * N
    else:
        i1 = np.repeat(np.arange(N), N)
        i2 = np.tile(np.arange(N), N)
        pair_weights = w[i1] * w[i2]
        f1, _ = _group_factors(L[0, 0] * g, epsilon[0], tau, M, inner_order, h_chunk)
        f2, _ = _group_factors(L[1, 0] * g[i1] + L[1, 1] * g[i2],
                               epsilon[1], tau, M, inner_order, h_chunk)
        f3_integrated = np.empty((N * N, S))
        for start in range(0, N * N, pair_chunk):
            stop = min(start + pair_chunk, N * N)
            x3 = (L[2, 0] * g[i1[start:stop]] + L[2, 1] * g[i2[start:stop]])[:, None] + L[2, 2] * g
            factors, _ = _group_factors(x3, epsilon[2], tau, M, inner_order, h_chunk)
            f3_integrated[start:stop] = np.einsum("bns,n->bs", factors, w)
        pmf = np.empty((S, S, S))
        for a in range(S):
            pmf[a] = f2.T @ ((pair_weights * f1[i1, a])[:, None] * f3_integrated)
        argument_count = 3 * N + N + N * N + N**3
    if not np.isfinite(pmf).all() or np.any(pmf < 0):
        raise FloatingPointError("Invalid GH PMF; no clipping or normalization is performed")
    count_y, mapping = _count_y_table(pmf, M)
    counts_probs = count_y.sum(axis=1)
    if np.any(counts_probs <= 0):
        raise FloatingPointError("Underflowed count marginal; conditional Y law is undefined")
    q = count_y / counts_probs[:, None]
    y_probs = count_y.sum(axis=0)
    joint_entropy, y_entropy = _entropy(pmf), _entropy(y_probs)
    computed_marginals = np.asarray([pmf.sum(axis=(1, 2)), pmf.sum(axis=(0, 2)), pmf.sum(axis=(0, 1))])
    result = {"schema": SCHEMA, "joint_pmf": pmf, "mass": float(pmf.sum()),
              "mass_error": float(pmf.sum() - 1), "minprob": float(pmf.min()),
              "counts_probs": counts_probs, "q_y_given_counts": q,
              "count_y_pmf": count_y, "y_marginal_probs": y_probs,
              "joint_entropy": joint_entropy, "y_entropy": y_entropy,
              "conditional_binomial_entropy_sum": expected_conditional_entropy,
              "conditional_mutual_information": joint_entropy - y_entropy - expected_conditional_entropy,
              "outer_order": int(outer_order), "inner_order": int(inner_order),
              "tau": tau, "epsilon": epsilon.copy(), "group_size": M,
              "private_observation_count": 3 * M, "combination_probabilities_included": True,
              "automatic_PMF_normalization": False, "negative_probability_or_MI_clipping": False,
              "group_marginal_probs": computed_marginals, "group_marginal_reference_1D": marginals,
              "group_marginal_GH_discrepancy": float(np.max(np.abs(computed_marginals - marginals))),
              "max_node_group_mass_error": local_mass_error,
              "PMF_backend": "separate_positive_h_GH_triangular_pair_BLAS",
              "independent_product_shortcut": independent, "private_h_argument_count": argument_count,
              "pair_chunk": pair_chunk, "h_chunk": h_chunk,
              "qualification": "finite_alphabet_exact_formula_with_ordinary_3D_outer_and_1D_inner_GH; no certified_error_bound"}
    result.update(mapping)
    return result


def law_precision_diagnostics(C, *, M, epsilon, tau, orders, pair_chunk=64, h_chunk=4096):
    """Compare only caller-requested (outer,inner) orders; not a certificate."""
    orders = tuple(tuple(pair) for pair in orders)
    if len(orders) < 2 or len(set(orders)) != len(orders) or any(len(p) != 2 for p in orders):
        raise ValueError("Provide at least two distinct (outer,inner) orders")
    laws = [joint_observation_law(C, M=M, epsilon=epsilon, tau=tau, outer_order=o,
                                 inner_order=i, pair_chunk=pair_chunk, h_chunk=h_chunk) for o, i in orders]
    return {"orders": orders, "laws": laws,
            "max_PMF_difference_from_last": max(float(np.max(np.abs(l["joint_pmf"] - laws[-1]["joint_pmf"]))) for l in laws[:-1]),
            "max_CMI_difference_from_last": max(abs(l["conditional_mutual_information"] - laws[-1]["conditional_mutual_information"]) for l in laws[:-1]),
            "qualification": "ordinary_GH_order_comparison_not_certified_error_bound"}


def joint_kl(teacher_pmf, candidate_pmf, *, M=None):
    """Raw PMF sums and Y marginal KL; no normalization or clipping."""
    teacher, candidate = (np.asarray(p, dtype=float) for p in (teacher_pmf, candidate_pmf))
    if M is None:
        if teacher.ndim != 3 or len(set(teacher.shape)) != 1 or teacher.shape[0] % 2:
            raise ValueError("A cubic full-alphabet PMF is required to infer M")
        M = teacher.shape[0] // 2 - 1
    M = _group_size(M)
    S = 2 * (M + 1)
    if any(p.shape != (S, S, S) or not np.isfinite(p).all() or np.any(p < 0) for p in (teacher, candidate)):
        raise ValueError(f"PMFs must be finite nonnegative ({S},{S},{S})")
    def kl(p, q):
        mask = p > 0
        if np.any(q[mask] <= 0):
            return math.inf
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
    teacher_y = _count_y_table(teacher, M)[0].sum(axis=0)
    candidate_y = _count_y_table(candidate, M)[0].sum(axis=0)
    mt, mc = float(teacher.sum()), float(candidate.sum())
    return {"joint_kl": kl(teacher, candidate), "y_marginal_kl": kl(teacher_y, candidate_y),
            "teacher_mass": mt, "candidate_mass": mc, "mass_difference": mt - mc,
            "teacher_mass_error": mt - 1, "candidate_mass_error": mc - 1,
            "automatic_normalization": False, "negative_KL_clipping": False,
            "qualification": "ordinary_GH_PMF_log_ratio_sums; mass_errors_retained_not_interval_certificate"}


def proposal_log_normalizer(beta, *, tau, order):
    """Explicit GH Z diagnostic, with exact beta0/1 endpoint normalizers."""
    beta = _beta(beta)
    _, tau = _parameters([0, 0, 0], tau)
    g, w = _normal_rule(order)
    computed = float(w @ np.exp(beta * log_expit(tau * g)))
    Z = 1. if beta == 0 else .5 if beta == 1 else computed
    return {"Z_beta": Z, "log_Z_beta": math.log(Z), "computed_GH_Z": computed,
            "GH_endpoint_residual": computed - Z, "beta": beta, "tau": tau, "order": int(order),
            "qualification": "ordinary_1D_GH_normalizer_diagnostic; endpoints_analytic"}


def sample_oriented_qbeta(rng, size, *, tau, beta, proposal_batch_size=1_000_000, max_proposals=None):
    """Caller-owned RNG, exact rejection acceptance sigma(tau*T)**beta.

    Ledger counts all proposals and accepted surplus, not only returned draws.
    beta0 consumes no uniforms. No seed or scientific budget is chosen here.
    """
    beta = _beta(beta)
    _, tau = _parameters([0, 0, 0], tau)
    if isinstance(size, bool):
        raise ValueError("size must be a shape of nonnegative integers")
    try:
        shape = (int(size),) if isinstance(size, (int, np.integer)) else tuple(size)
    except TypeError as exc:
        raise ValueError("size must be a shape of nonnegative integers") from exc
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in shape):
        raise ValueError("size must be a shape of nonnegative integers")
    batch_size = _integer(proposal_batch_size, "proposal_batch_size")
    requested = math.prod(shape)
    if max_proposals is not None:
        max_proposals = _integer(max_proposals, "max_proposals", 0)
        if max_proposals < requested:
            raise ValueError("max_proposals must be >= requested draws")
    if rng is None or not hasattr(rng, "standard_normal") or not hasattr(rng, "random"):
        raise ValueError("An explicit caller-owned RNG is required")
    if beta == 0:
        result = rng.standard_normal(shape)
        return result, {"proposed": requested, "accepted": requested, "used": requested,
                        "discarded_accepted": 0, "uniforms_generated": 0,
                        "acceptance_fraction": 1. if requested else None}
    result = np.empty(requested)
    used = proposed = accepted = 0
    while used < requested:
        batch = min(batch_size, max(32, 2 * (requested - used)))
        if max_proposals is not None:
            batch = min(batch, max_proposals - proposed)
            if batch <= 0:
                raise RuntimeError(f"Rejection budget exhausted: proposed={proposed}, accepted={accepted}, used={used}")
        normals = rng.standard_normal(batch)
        uniforms = rng.random(batch)
        with np.errstate(divide="ignore"):
            good = normals[np.log(uniforms) < beta * log_expit(tau * normals)]
        take = min(len(good), requested - used)
        result[used:used + take] = good[:take]
        used += take
        proposed += batch
        accepted += len(good)
    return result.reshape(shape), {"proposed": proposed, "accepted": accepted, "used": used,
                                   "discarded_accepted": accepted - used, "uniforms_generated": proposed,
                                   "acceptance_fraction": accepted / proposed if proposed else None}
