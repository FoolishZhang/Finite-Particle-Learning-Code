"""Numerical angular Hessians at b_j + a_j z (two latent coordinates).

This is configurable float64 quadrature, not a rigorous Gaussian or Laplace-tail
certificate. Hermite uses the standard-normal rule; normal Legendre integrates
phi(z) on [-normal_limit,normal_limit] without renormalizing its mass. The outer
Laplace integral uses v=log(u) on an explicit finite interval, by a uniform
trapezoidal rule by default or by Legendre as a separate numerical comparison.

No sign, coefficient mass, Hessian eigenvalue, or Ward residual is clipped.
Finite-interval omissions remain visible in the d-mass and Ward diagnostics.
K=1 has c=0 and the exact zero angular Laplacian; its directly evaluated Ward
expression can still show quadrature error. K=infinity uses the one-dimensional
posterior rule. A positive likelihood_scale merely sets x=A/scale for finite K;
it need not be the true marginal probability and it does not alter target q.

All likelihood and Laplace factors are assembled in log space. Exponentials
below float64 range may underflow to zero; unrepresentable outputs raise an
error rather than being replaced. Pattern count grows as 2**m. There is no RNG,
file I/O, study-specific teacher, or dependency on a frozen research module.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import operator
import numpy as np
from scipy.special import expit, logsumexp, roots_hermitenorm, roots_legendre


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def patterns(m):
    """Lexicographic binary patterns, first coordinate most significant."""
    m = _integer(m, "m")
    if m >= np.iinfo(np.intp).bits - 1:
        raise ValueError("pattern enumeration is not indexable on this platform")
    index = np.arange(1 << m, dtype=np.uint64)
    shifts = np.arange(m - 1, -1, -1, dtype=np.uint64)
    return ((index[:, None] >> shifts) & 1).astype(np.int8)


def normal_nodes(order, *, rule="hermite", limit=10.0):
    """One-dimensional standard-normal quadrature; no weight mass correction."""
    order = _integer(order, "normal order")
    if rule == "hermite":
        z, weights = roots_hermitenorm(order)
        weights = weights / math.sqrt(2 * math.pi)
    elif rule == "legendre":
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("normal_limit must be finite and positive")
        nodes, weights = roots_legendre(order)
        z = limit * nodes
        weights = limit * weights * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    else:
        raise ValueError("normal_rule must be 'hermite' or 'legendre'")
    if not np.isfinite(z).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise FloatingPointError("invalid normal quadrature nodes or weights")
    return np.asarray(z, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def logu_nodes(logu_min=-24.0, logu_max=56.0, logu_step=.25, *, rule="trapezoid", order=256):
    """Nodes and integration weights for dv on the stated finite log-u interval."""
    lo, hi = float(logu_min), float(logu_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("finite logu_min < logu_max is required")
    if rule == "trapezoid":
        step = float(logu_step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("logu_step must be finite and positive")
        intervals = round((hi - lo) / step)
        if intervals < 1 or not math.isclose(intervals * step, hi - lo, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("log-u interval width must be an integer multiple of logu_step")
        nodes = np.linspace(lo, hi, intervals + 1)
        weights = np.full(intervals + 1, (hi - lo) / intervals)
        weights[[0, -1]] *= .5
    elif rule == "legendre":
        nodes, weights = roots_legendre(_integer(order, "outer_order"))
        nodes = (hi + lo) / 2 + (hi - lo) / 2 * nodes
        weights = (hi - lo) / 2 * weights
    else:
        raise ValueError("outer_rule must be 'trapezoid' or 'legendre'")
    return nodes, weights


def _counts(values):
    counts = []
    for value in values:
        if value is None or (isinstance(value, str) and value.lower() in ("inf", "infinity")):
            count = "inf"
        else:
            number = float(value)
            if number == math.inf:
                count = "inf"
            elif math.isfinite(number) and number.is_integer() and 1 <= number <= 2 ** 53:
                count = int(number)
            else:
                raise ValueError("K_values must be positive exact integers <=2**53 or infinity")
        if count in counts:
            raise ValueError("K_values must not contain duplicates")
        counts.append(count)
    if not counts:
        raise ValueError("at least one K is required")
    return tuple(counts)


def _targets(targets, count):
    if isinstance(targets, Mapping):
        names = tuple(str(name) for name in targets)
        target = _real(list(targets.values()), "targets")
    else:
        target = _real(targets, "targets")
        if target.ndim == 1:
            target = target[None, :]
        names = tuple(f"target_{i}" for i in range(len(target))) if target.ndim else ()
    if target.ndim != 2 or target.shape[1] != count or not len(target) or np.any(target < 0):
        raise ValueError("targets must be one or more nonnegative complete pattern-probability vectors")
    mass = target.sum(axis=1)
    if not np.allclose(mass, 1.0, atol=1e-12, rtol=1e-10):
        raise ValueError("each target must sum to one; targets are not automatically renormalized")
    return names, target


def _exp_difference(loga, logb):
    """exp(loga)-exp(logb), preserving its sign and avoiding 0*infinity."""
    out = np.zeros_like(loga)
    different = loga != logb
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = loga[different] - logb[different]
        top = np.maximum(loga[different], logb[different])
        out[different] = np.sign(delta) * np.exp(top) * (-np.expm1(-np.abs(delta)))
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite Laplace loss integrand")
    return out


def rankone_stability(b, a, targets, K_values=(1, 2, np.inf), *, gh_order=64,
                      normal_rule="hermite", normal_order=None, normal_limit=10.0,
                      logu_min=-24.0, logu_max=56.0, logu_step=.25,
                      outer_rule="trapezoid", outer_order=256, logu_limit=None,
                      likelihood_scale=None, compute_loss=False, return_coefficients=False):
    """Return angular stability matrices with axes target, K, coordinate(s).

    ``targets`` accepts [2**m], [T,2**m], or a mapping of names to probability
    vectors. ``K_values`` accepts positive integers and np.inf/'inf'/None.
    ``normal_order`` overrides gh_order; the latter remains a convenient alias.
    ``logu_limit=L`` requests a symmetric [-L,L] interval instead of min/max.

    Main outputs: laplacian / edge_weights / Hperp / ward_angle_hessian /
    ward_residual [T,K,m,m]; bias_gradient / radial_gradient [T,K,m];
    eigenvalues [T,K,m].
    Per-label matrices/gradients and d/c masses have axes [K,pattern,...].
    Optional c/d coefficient arrays are [K,pattern,normal_node]. Losses, when
    requested, use the same truncated log-u interval and are not certified.
    ``laplacian`` and its eigenvalues are unprojected; no common-angle mode is
    deleted. ``work_counts`` counts high-level scalar factors, not hardware
    exponential instructions or the internal operations of scipy.logsumexp.
    """
    b, a = _real(b, "b"), _real(a, "a")
    if b.ndim != 1 or not len(b) or a.shape != b.shape:
        raise ValueError("b and a must be matching nonempty vectors")
    m = len(b)
    labels = patterns(m).astype(np.float64)
    names, target = _targets(targets, len(labels))
    counts = _counts(K_values)
    order = _integer(gh_order if normal_order is None else normal_order, "normal_order")
    if logu_limit is not None:
        limit = float(logu_limit)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("logu_limit must be finite and positive")
        logu_min, logu_max = -limit, limit
    z, normal_weights = normal_nodes(order, rule=normal_rule, limit=normal_limit)
    v, outer_weights = logu_nodes(logu_min, logu_max, logu_step, rule=outer_rule, order=outer_order)
    with np.errstate(divide="ignore"):
        logw = np.log(normal_weights)
    logvweight = np.log(outer_weights)
    with np.errstate(over="raise", invalid="raise"):
        logits = b[None, :] + z[:, None] * a[None, :]
        logyes, logno = -np.logaddexp(0, -logits), -np.logaddexp(0, logits)
        logA = logyes @ labels.T + logno @ (1 - labels).T
    if not np.isfinite(logA).all():
        raise FloatingPointError("nonfinite conditional log likelihood")
    logEA = logsumexp(logw[:, None] + logA, axis=0)
    if likelihood_scale is None:
        logscale = logEA.copy()
    else:
        scale = _real(likelihood_scale, "likelihood_scale")
        if scale.ndim == 0:
            scale = np.full(len(labels), float(scale))
        if scale.shape != (len(labels),) or np.any(scale <= 0):
            raise ValueError("likelihood_scale must be positive, scalar or length 2**m")
        logscale = np.log(scale)
    sigma = expit(logits)
    sigma_prime = sigma * (1.0 - sigma)
    shape = (len(counts), len(labels))
    M = np.zeros(shape + (m, m))
    bias = np.zeros(shape + (m,))
    radial = np.zeros(shape + (m,))
    diag_sigma = np.zeros(shape + (m,))
    mass_c, mass_d = np.zeros(shape), np.zeros(shape)
    label_loss = np.empty(shape) if compute_loss else None
    all_c = np.empty(shape + (len(z),)) if return_coefficients else None
    all_d = np.empty(shape + (len(z),)) if return_coefficients else None
    finite_needed = any(K != "inf" and K > 1 for K in counts)
    for yi, y in enumerate(labels):
        r = y[None, :] - sigma
        posterior = np.exp(logw + logA[:, yi] - logEA[yi])
        if finite_needed:
            logx = logA[:, yi] - logscale[yi]
            # -exp(log(u*x)) is the log of exp(-u*x). Overflow of u*x gives
            # the correctly underflowed zero Laplace factor, not a coefficient clamp.
            with np.errstate(over="ignore", under="ignore"):
                negative_ux = -np.exp(v[:, None] + logx[None, :])
            logPsi = logsumexp(logw[None, :] + negative_ux, axis=1)
            logB0 = logsumexp(logw[None, :] + logx[None, :] + negative_ux, axis=1)
        for ki, K in enumerate(counts):
            if K == 1:
                c = np.zeros_like(z)
                d = normal_weights.copy()
                if compute_loss:
                    label_loss[ki, yi] = -float(normal_weights @ logA[:, yi])
            elif K == "inf":
                c = posterior.copy()
                d = posterior.copy()
                if compute_loss:
                    label_loss[ki, yi] = -logEA[yi]
            else:
                with np.errstate(over="ignore", under="ignore", invalid="raise"):
                    power_c = np.zeros_like(logPsi) if K == 2 else (K - 2) * logPsi
                    outer_c = logvweight + math.log(K) + math.log(K - 1) + 2 * v + logB0 + power_c
                    outer_d = logvweight + math.log(K) + v + (K - 1) * logPsi
                    c = np.exp(logw + logx + logsumexp(outer_c[:, None] + negative_ux, axis=0))
                    d = np.exp(logw + logx + logsumexp(outer_d[:, None] + negative_ux, axis=0))
                if compute_loss:
                    with np.errstate(over="ignore", under="ignore"):
                        other = -K * np.exp(v)
                    integrand = _exp_difference(K * logPsi, other)
                    label_loss[ki, yi] = -logscale[yi] + float(outer_weights @ integrand)
            if not np.isfinite(c).all() or not np.isfinite(d).all():
                raise FloatingPointError("nonfinite Laplace coefficients; inspect the quadrature configuration")
            # Use one dot product for each symmetric pair. This enforces only
            # the mathematical symmetry, without changing an edge sign/eigenvalue.
            for i in range(m):
                for j in range(i, m):
                    value = float(c @ (r[:, i] * r[:, j]))
                    M[ki, yi, i, j] = M[ki, yi, j, i] = value
            bias[ki, yi] = -(d[:, None] * r).sum(axis=0)
            radial[ki, yi] = -(d[:, None] * r * z[:, None]).sum(axis=0)
            diag_sigma[ki, yi] = (d[:, None] * sigma_prime).sum(axis=0)
            mass_c[ki, yi], mass_d[ki, yi] = c.sum(), d.sum()
            if return_coefficients:
                all_c[ki, yi], all_d[ki, yi] = c, d
    target_M = np.einsum("tp,kpij->tkij", target, M, optimize=True)
    gb = np.einsum("tp,kpi->tki", target, bias, optimize=True)
    g = np.einsum("tp,kpi->tki", target, radial, optimize=True)
    diagonals = np.einsum("tp,kpi->tki", target, diag_sigma, optimize=True)
    edges = target_M * a[None, None, :, None] * a[None, None, None, :]
    diagonal_index = np.arange(m)
    edges[..., diagonal_index, diagonal_index] = 0.0  # Edges are defined only for i != j.
    laplacian = -edges.copy()
    laplacian[..., diagonal_index, diagonal_index] = edges.sum(axis=-1)
    Hperp = -target_M.copy()
    Hperp[..., diagonal_index, diagonal_index] += diagonals
    ward = Hperp * a[None, None, :, None] * a[None, None, None, :]
    ward[..., diagonal_index, diagonal_index] -= a[None, None, :] * g
    residual = ward - laplacian
    upper = np.array([1.0 if K == "inf" else 1.0 - 1.0 / K for K in counts])
    result = dict(target_names=names, K_values=counts, patterns=labels.astype(np.int8),
        laplacian=laplacian, edge_weights=edges, bias_gradient=gb, radial_gradient=g, Hperp=Hperp,
        ward_angle_hessian=ward, ward_residual=residual,
        ward_residual_max=np.max(np.abs(residual), axis=(-2, -1)),
        common_angle_residual=np.max(np.abs(laplacian.sum(axis=-1)), axis=-1),
        eigenvalues=np.linalg.eigvalsh(laplacian),
        label_M=M, label_bias_gradient=bias, label_radial_gradient=radial, label_sigma_prime_average=diag_sigma,
        label_d_mass=mass_d, label_d_mass_error=mass_d - 1.0,
        label_c_mass=mass_c, c_mass_upper=upper,
        label_c_mass_upper_excess=mass_c - upper[:, None],
        target_d_mass=np.einsum("tp,kp->tk", target, mass_d),
        target_c_mass=np.einsum("tp,kp->tk", target, mass_c),
        log_evidence=logEA, log_likelihood_scale=logscale,
        normal_nodes=z, normal_weights=normal_weights, normal_quadrature_mass=float(normal_weights.sum()),
        outer_logu_nodes=v, outer_weights=outer_weights,
        loss=None if label_loss is None else np.einsum("tp,kp->tk", target, label_loss),
        label_loss=label_loss,
        config=dict(normal_rule=normal_rule, normal_order=order, normal_limit=None if normal_rule == "hermite" else float(normal_limit),
                    outer_rule=outer_rule, logu_min=float(logu_min), logu_max=float(logu_max),
                    logu_step=float(logu_step) if outer_rule == "trapezoid" else None,
                    outer_order=len(v) if outer_rule == "legendre" else None,
                    likelihood_scale_source="inner_quadrature_evidence" if likelihood_scale is None else "supplied_positive_scale",
                    target_masses=target.sum(axis=1)),
        work_counts=dict(targets=len(target), labels=len(labels), normal_nodes=len(z), outer_nodes=len(v),
                         finite_K_greater_than_one=sum(K != "inf" and K > 1 for K in counts),
                         bernoulli_logit_evaluations=len(z) * m,
                         conditional_loglikelihood_values=len(labels) * len(z),
                         posterior_coefficients=len(labels) * len(z),
                         laplace_exp_minus_ux_factors=len(labels) * len(v) * len(z) if finite_needed else 0,
                         laplace_logPsi_values=len(labels) * len(v) if finite_needed else 0,
                         laplace_logB0_values=len(labels) * len(v) if finite_needed else 0,
                         finite_c_d_coefficient_pairs=len(labels) * len(z) * sum(K != "inf" and K > 1 for K in counts),
                         interpretation="high_level_scalar_factor_counts_not_CPU_instruction_counts"),
        qualifications=dict(rigorous_Gaussian_remainder_certificate=False, rigorous_outer_truncation_certificate=False,
                            signs_or_eigenvalues_clipped=False, coefficient_masses_renormalized=False))
    if return_coefficients:
        result.update(c_coefficients=all_c, d_coefficients=all_d)
    for key in ("laplacian", "bias_gradient", "radial_gradient", "Hperp", "ward_angle_hessian", "ward_residual", "eigenvalues"):
        if not np.isfinite(result[key]).all():
            raise FloatingPointError(f"nonfinite output {key}")
    if compute_loss and not np.isfinite(result["loss"]).all():
        raise FloatingPointError("nonfinite loss output")
    return result
