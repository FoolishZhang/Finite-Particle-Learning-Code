"""Float64 loss, full (b,a) derivatives and transverse blocks at rank one.

Coordinates are theta=(b[0:m], a[0:m]) with logits b_j+a_j*z.  Angles use
W_j=a_j*(cos(phi_j),sin(phi_j)), including signed a_j.  Targets are fixed
complete binary-pattern probabilities; they are never model-renormalized.

The finite-K Laplace identity is integrated on a stated finite log-u interval.
Normal Legendre weights include the standard-normal density and are NOT mass
renormalized. With normal-rule mass M, finite-K loss uses the raw bank measure
of mass M**K, including that multiplier in both Frullani subtraction terms.
These are numerical approximations, not certified Gaussian or
Laplace remainder bounds.  Mass, Ward and symmetry errors remain inspectable;
no gradient, edge, eigenvalue or coefficient-mass clipping is performed.

There is no RNG, file I/O or dependency on previous frozen research kernels.
Every returned ndarray is an independent snapshot, including aliases/views.
Underflowed positive factors may become zero.  Nonfinite requested outputs
raise FloatingPointError rather than receiving an invented finite replacement.
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
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def patterns(m):
    """Lexicographic 0/1 rows; the first coordinate is most significant."""
    m = _integer(m, "m")
    if m >= np.iinfo(np.intp).bits - 1:
        raise ValueError("pattern enumeration is not indexable")
    index = np.arange(1 << m, dtype=np.uint64)
    shift = np.arange(m - 1, -1, -1, dtype=np.uint64)
    return ((index[:, None] >> shift) & 1).astype(np.int8)


def normal_nodes(order, *, rule="hermite", limit=10.0):
    """Standard-normal Hermite or density-weighted truncated Legendre rule."""
    order = _integer(order, "normal_order")
    if rule == "hermite":
        z, w = roots_hermitenorm(order)
        w = w / math.sqrt(2 * math.pi)
    elif rule == "legendre":
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("normal_limit must be positive and finite")
        x, w = roots_legendre(order)
        z = limit * x
        w = limit * w * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi)
    else:
        raise ValueError("normal_rule must be 'hermite' or 'legendre'")
    if not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise FloatingPointError("invalid normal rule")
    return z.copy(), w.copy()


def logu_nodes(logu_min=-24., logu_max=56., logu_step=.25, *, rule="trapezoid", order=256):
    lo, hi = float(logu_min), float(logu_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("finite logu_min < logu_max is required")
    if rule == "trapezoid":
        step = float(logu_step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("logu_step must be positive and finite")
        n = round((hi - lo) / step)
        if n < 1 or not math.isclose(n * step, hi - lo, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("log-u width must be an integer multiple of logu_step")
        v, w = np.linspace(lo, hi, n + 1), np.full(n + 1, (hi - lo) / n)
        w[[0, -1]] *= .5
    elif rule == "legendre":
        v, w = roots_legendre(_integer(order, "outer_order"))
        v, w = (hi + lo) / 2 + (hi - lo) / 2 * v, (hi - lo) / 2 * w
    else:
        raise ValueError("outer_rule must be 'trapezoid' or 'legendre'")
    return v, w


def _counts(values):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("K must be an integer or infinity")
        if value is None or isinstance(value, str) and value.lower() in ("inf", "infinity"):
            K = "inf"
        else:
            number = float(value)
            if number == math.inf:
                K = "inf"
            elif math.isfinite(number) and number.is_integer() and 1 <= number <= 2 ** 53:
                K = int(number)
            else:
                raise ValueError("K must be an exact integer in [1,2**53] or infinity")
        if K in result:
            raise ValueError("duplicate K")
        result.append(K)
    if not result:
        raise ValueError("at least one K is required")
    return tuple(result)


def _targets(value, count):
    if isinstance(value, Mapping):
        names, q = tuple(str(k) for k in value), _real(list(value.values()), "targets")
    else:
        q = _real(value, "targets")
        if q.ndim == 1:
            q = q[None, :]
        names = tuple(f"target_{i}" for i in range(len(q))) if q.ndim else ()
    if q.ndim != 2 or q.shape[1] != count or not len(q) or np.any(q < 0):
        raise ValueError("targets must be complete nonnegative pattern-probability vectors")
    if not np.allclose(q.sum(axis=1), 1, atol=1e-12, rtol=1e-10):
        raise ValueError("target probability mass must be one; it is not renormalized")
    return names, q


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return value


def _logpsi(negative_ux, weights, logweights):
    """Use log1p/expm1 near Psi=1, including the uncorrected normal mass."""
    answer = logsumexp(logweights[None, :] + negative_ux, axis=1)
    # For small u, exp(-u*x) would round to one. expm1 preserves its deficit.
    deficit = np.expm1(negative_ux) @ weights
    mass = math.fsum(float(w) for w in weights)
    offset = deficit / mass
    small = np.abs(offset) < .25
    answer[small] = math.log(mass) + np.log1p(offset[small])
    return answer


def _exp_difference(loga, logb):
    result = np.zeros_like(loga)
    different = loga != logb
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = loga[different] - logb[different]
        result[different] = (np.sign(delta) * np.exp(np.maximum(loga[different], logb[different]))
                             * -np.expm1(-np.abs(delta)))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite Laplace loss integrand")
    return result


def rankone_stationary(b, a, targets, K_values=(1, 2, np.inf), *,
                       normal_rule="hermite", normal_order=None, normal_limit=10., gh_order=64,
                       logu_min=-24., logu_max=56., logu_step=.25,
                       outer_rule="trapezoid", outer_order=256,
                       likelihood_scale=None, compute_hessian=True, return_coefficients=False):
    """Compute fixed-target rank-one losses and derivatives.

    Ordering is [b_0,...,b_(m-1),a_0,...,a_(m-1)]. Targets accept [2**m],
    [T,2**m], or name->vector. K accepts positive integers, np.inf/'inf'/None.
    Main shapes: loss[T,K], gradient[T,K,2m], H_ba[T,K,2m,2m], and
    Hperp/Hangle/directWard[T,K,m,m]. Per-label counterparts start [K,2**m].
    Hangle is the signed Laplacian; directWard independently contracts Hperp
    and the radial gradient. Neither is projected or spectrally corrected.
    If normal mass is M, finite-K d masses approximate M**K and the raw
    collision-mass bound is M**K*(1-1/K). Both raw and unit-mass diagnostics
    are returned. K=infinity uses posterior ratios and d mass one.
    With compute_hessian=False all second-order/c-coefficient fields are None;
    loss, first derivatives and d-mass diagnostics retain the same definitions.
    Optional c/d arrays have shape [K,2**m,normal_node]. A supplied positive
    likelihood_scale affects only Laplace coordinates, never the target.
    """
    b, a = _real(b, "b"), _real(a, "a")
    if b.ndim != 1 or not len(b) or b.shape != a.shape:
        raise ValueError("b and a must be matching nonempty vectors")
    m = len(b)
    Y = patterns(m)
    names, q = _targets(targets, len(Y))
    Ks = _counts(K_values)
    order = _integer(gh_order if normal_order is None else normal_order, "normal_order")
    z, w = normal_nodes(order, rule=normal_rule, limit=normal_limit)
    normal_mass = math.fsum(float(t) for t in w)
    logmass = math.log(normal_mass)
    bank_mass = np.array([1. if K == "inf" else math.exp(K * logmass) for K in Ks])
    v, wv = logu_nodes(logu_min, logu_max, logu_step, rule=outer_rule, order=outer_order)
    with np.errstate(divide="ignore"):
        logw = np.log(w)
    logwv = np.log(wv)
    with np.errstate(over="raise", invalid="raise"):
        logits = b + z[:, None] * a
        logA = -np.logaddexp(0, -logits) @ Y.T - np.logaddexp(0, logits) @ (1 - Y).T
    if not np.isfinite(logA).all():
        raise FloatingPointError("nonfinite conditional log likelihood")
    logEA = logsumexp(logw[:, None] + logA, axis=0)
    if likelihood_scale is None:
        logscale = logEA - logmass
    else:
        scale = _real(likelihood_scale, "likelihood_scale")
        if scale.ndim == 0:
            scale = np.full(len(Y), float(scale))
        if scale.shape != (len(Y),) or np.any(scale <= 0):
            raise ValueError("likelihood_scale must be positive scalar or pattern vector")
        logscale = np.log(scale)
    sigma = expit(logits)
    sp = expit(logits) * expit(-logits)
    shape = (len(Ks), len(Y))
    loss_y, mass_d = np.empty(shape), np.empty(shape)
    grad_y = np.empty(shape + (2 * m,))
    H_y = np.empty(shape + (2 * m, 2 * m)) if compute_hessian else None
    B_y = np.empty_like(H_y) if compute_hessian else None
    same_y = np.empty_like(H_y) if compute_hessian else None
    cross_y = np.empty_like(H_y) if compute_hessian else None
    M_y = np.empty(shape + (m, m)) if compute_hessian else None
    sigma_y = np.empty(shape + (m,)) if compute_hessian else None
    mass_c = np.empty(shape) if compute_hessian else None
    all_d = np.empty(shape + (len(z),)) if return_coefficients else None
    all_c = np.empty(shape + (len(z),)) if return_coefficients and compute_hessian else None
    finite_Ks = sum(K != "inf" and K > 1 for K in Ks)
    ix = np.arange(m)
    for yi, y in enumerate(Y):
        r = y - sigma
        G = np.concatenate((r, z[:, None] * r), axis=1)
        if finite_Ks:
            logx = logA[:, yi] - logscale[yi]
            with np.errstate(over="ignore", under="ignore"):
                negative_ux = -np.exp(v[:, None] + logx[None, :])
            lp = _logpsi(negative_ux, w, logw)
            if compute_hessian:
                logt = logw[None, :] + logx[None, :] + negative_ux
                lb0 = logsumexp(logt, axis=1)
                # M_G/B0 is a bounded weighted average of G. Rows whose
                # B0 underflows in log space carry exactly zero outer weight.
                avgG = np.zeros((len(v), 2 * m))
                nonzero = np.isfinite(lb0)
                avgG[nonzero] = np.exp(logt[nonzero] - lb0[nonzero, None]) @ G
        for ki, K in enumerate(Ks):
            if K == 1:
                d = w.copy()
                c = np.zeros_like(w) if compute_hessian else None
                cross = np.zeros((2 * m, 2 * m)) if compute_hessian else None
                loss_y[ki, yi] = -w @ logA[:, yi]
            elif K == "inf":
                d = np.exp(logw + logA[:, yi] - logEA[yi])
                c = d.copy() if compute_hessian else None
                if compute_hessian:
                    meanG = d @ G
                    cross = np.outer(meanG, meanG)
                loss_y[ki, yi] = -logEA[yi]
            else:
                with np.errstate(over="ignore", under="ignore", invalid="raise"):
                    ld = logwv + math.log(K) + v + (K - 1) * lp
                    d = np.exp(logw + logx + logsumexp(ld[:, None] + negative_ux, axis=0))
                    if compute_hessian:
                        power = np.zeros_like(lp) if K == 2 else (K - 2) * lp
                        outer = logwv + math.log(K) + math.log(K - 1) + 2 * v + power
                        c = np.exp(logw + logx + logsumexp((outer + lb0)[:, None] + negative_ux, axis=0))
                        crossweight = np.exp(outer + 2 * lb0)
                        cross = avgG.T @ (crossweight[:, None] * avgG)
                    other = K * logmass - K * np.exp(v)
                loss_y[ki, yi] = -bank_mass[ki] * logscale[yi] + wv @ _exp_difference(K * lp, other)
            if not np.isfinite(d).all() or compute_hessian and not np.isfinite(c).all():
                raise FloatingPointError("nonfinite Laplace coefficients")
            grad_y[ki, yi] = -d @ G
            mass_d[ki, yi] = d.sum()
            if return_coefficients:
                all_d[ki, yi] = d
            if compute_hessian:
                ds = d[:, None] * sp
                B = np.zeros((2 * m, 2 * m))
                B[ix, ix] = ds.sum(axis=0)
                B[ix, m + ix] = B[m + ix, ix] = z @ ds
                B[m + ix, m + ix] = (z * z) @ ds
                same = G.T @ (c[:, None] * G)
                H_y[ki, yi] = B - same + cross
                B_y[ki, yi], same_y[ki, yi], cross_y[ki, yi] = B, same, cross
                M_y[ki, yi] = r.T @ (c[:, None] * r)
                sigma_y[ki, yi] = ds.sum(axis=0)
                mass_c[ki, yi] = c.sum()
                if return_coefficients:
                    all_c[ki, yi] = c

    def combine(array):
        return None if array is None else np.einsum("tp,kp...->tk...", q, array, optimize=True)

    loss, grad, H = combine(loss_y), combine(grad_y), combine(H_y)
    Hperp = Hangle = edges = ward = ward_residual = None
    label_Hperp = label_Hangle = label_ward = None
    if compute_hessian:
        M = combine(M_y)
        Hperp = -M.copy()
        Hperp[..., ix, ix] += combine(sigma_y)
        edges = M * a[None, None, :, None] * a[None, None, None, :]
        edges[..., ix, ix] = 0
        Hangle = -edges.copy()
        Hangle[..., ix, ix] = edges.sum(axis=-1)
        ward = Hperp * a[None, None, :, None] * a[None, None, None, :]
        ward[..., ix, ix] -= a * grad[..., m:]
        ward_residual = ward - Hangle
        label_Hperp = -M_y.copy()
        label_Hperp[..., ix, ix] += sigma_y
        label_edges = M_y * a[None, None, :, None] * a[None, None, None, :]
        label_edges[..., ix, ix] = 0
        label_Hangle = -label_edges.copy()
        label_Hangle[..., ix, ix] = label_edges.sum(axis=-1)
        label_ward = label_Hperp * a[None, None, :, None] * a[None, None, None, :]
        label_ward[..., ix, ix] -= a * grad_y[..., m:]
    upper = np.array([1. if K == "inf" else 1. - 1. / K for K in Ks])
    result = dict(target_names=names, K_values=Ks, patterns=Y,
                  b=b, a=a, targets=q, loss=loss, gradient=grad,
                  bias_gradient=grad[..., :m], radial_gradient=grad[..., m:], H_ba=H,
                  Hperp=Hperp, Hangle=Hangle, directWard=ward, edge_weights=edges,
                  ward_residual=ward_residual,
                  ward_residual_max=None if ward is None else np.max(np.abs(ward_residual), axis=(-2, -1)),
                  H_ba_symmetry_residual=None if H is None else np.max(np.abs(H - H.swapaxes(-1, -2)), axis=(-2, -1)),
                  common_angle_residual=None if Hangle is None else np.max(np.abs(Hangle.sum(axis=-1)), axis=-1),
                  label_loss=loss_y, label_gradient=grad_y,
                  label_bias_gradient=grad_y[..., :m], label_radial_gradient=grad_y[..., m:],
                  label_H_ba=H_y, label_B=B_y, label_same_particle_term=same_y,
                  label_cross_particle_term=cross_y, label_M=M_y, label_sigma_prime_average=sigma_y,
                  label_Hperp=label_Hperp, label_Hangle=label_Hangle, label_directWard=label_ward,
                  label_d_mass=mass_d, label_d_mass_error=mass_d - 1,
                  label_d_mass_expected=np.broadcast_to(bank_mass[:, None], shape),
                  label_d_mass_quadrature_error=mass_d - bank_mass[:, None],
                  label_c_mass=mass_c, c_mass_upper=upper,
                  label_c_mass_upper_excess=None if mass_c is None else mass_c - upper[:, None],
                  raw_c_mass_upper=upper * bank_mass,
                  label_raw_c_mass_upper_excess=None if mass_c is None else mass_c - (upper * bank_mass)[:, None],
                  target_d_mass=combine(mass_d), target_c_mass=combine(mass_c),
                  log_evidence=logEA, log_likelihood_scale=logscale,
                  normal_nodes=z, normal_weights=w, normal_quadrature_mass=normal_mass,
                  raw_bank_quadrature_mass=bank_mass,
                  outer_logu_nodes=v, outer_weights=wv,
                  d_coefficients=all_d, c_coefficients=all_c,
                  config=dict(coordinate_order="b_then_a", compute_hessian=bool(compute_hessian),
                              normal_rule=normal_rule, normal_order=order,
                              normal_limit=float(normal_limit) if normal_rule == "legendre" else None,
                              outer_rule=outer_rule, logu_min=float(logu_min), logu_max=float(logu_max),
                              logu_step=float(logu_step) if outer_rule == "trapezoid" else None,
                              outer_order=len(v) if outer_rule == "legendre" else None,
                              target_masses=q.sum(axis=1),
                              likelihood_scale_source="inner_quadrature_evidence_divided_by_normal_mass" if likelihood_scale is None else "supplied_positive_scale",
                              finite_loss_mass_correction="minus_M_power_K_log_scale_and_M_power_K_Frullani_subtraction"),
                  work_counts=dict(targets=len(q), labels=len(Y), normal_nodes=len(z), outer_nodes=len(v),
                                   finite_K_greater_than_one=finite_Ks,
                                   bernoulli_logit_evaluations=len(z) * m,
                                   conditional_loglikelihood_values=len(Y) * len(z),
                                   laplace_exp_minus_ux_factors=len(Y) * len(v) * len(z) if finite_Ks else 0,
                                   finite_d_coefficients=len(Y) * len(z) * finite_Ks,
                                   finite_c_coefficients=len(Y) * len(z) * finite_Ks if compute_hessian else 0,
                                   finite_cross_particle_outer_products=len(Y) * len(v) * finite_Ks if compute_hessian else 0,
                                   interpretation="high_level_scalar_factor_counts_not_CPU_instruction_counts"),
                  qualifications=dict(rigorous_Gaussian_remainder_certificate=False,
                                      rigorous_outer_truncation_certificate=False,
                                      signs_or_eigenvalues_clipped=False, coefficient_masses_renormalized=False))
    for key in ("loss", "gradient", "H_ba", "Hperp", "Hangle", "directWard"):
        if result[key] is not None and not np.isfinite(result[key]).all():
            raise FloatingPointError(f"nonfinite {key}")
    return _snapshot(result)
