"""Paired finite-K losses for binary observations with a 2D Gaussian latent.

The primary finite-K difference integrates Psi_new**K - Psi_base**K directly.
It is not obtained by subtracting two absolute losses. Both models share the
baseline scale c_y=raw_evidence_base_y/raw_normal_mass (or a supplied positive
scale) and exactly the same normal/log-u rule. Normal weights and targets are
never renormalized. Absolute finite-K losses use the raw bank mass M**K.

Normal axes can have different Hermite / density-weighted Legendre rules.
This is configurable float64 numerical quadrature, not a Gaussian or Laplace
tail certificate. Finite log-u bounds, floating-point underflow and quadrature
error remain numerical qualifications. No RNG, optimization, file I/O, global
thread-setting changes or dependency on frozen research modules is present.
Callers/entry points may set BLAS thread counts before importing this module.
Every returned ndarray is an independent snapshot, not an alias to an input.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import operator
import numpy as np
from scipy.special import logsumexp, roots_hermitenorm, roots_legendre


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
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) for v in value)
    return value


def patterns(m):
    m = _integer(m, "m")
    if m >= np.iinfo(np.intp).bits - 1:
        raise ValueError("pattern enumeration is not indexable")
    ids = np.arange(1 << m, dtype=np.uint64)
    shifts = np.arange(m - 1, -1, -1, dtype=np.uint64)
    return ((ids[:, None] >> shifts) & 1).astype(np.int8)


def _counts(values):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("K must be a positive integer or infinity")
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
    if not np.allclose(q.sum(axis=1), 1., atol=1e-12, rtol=1e-10):
        raise ValueError("target probability mass must be one; no renormalization is applied")
    return names, q


def normal_tensor(normal_specs):
    """Return separate axis rules and their tensor product, axis 1 fastest."""
    if len(normal_specs) != 2:
        raise ValueError("normal_specs must contain exactly two axis rules")
    axes, specs = [], []
    for given in normal_specs:
        if not isinstance(given, Mapping):
            raise ValueError("each normal spec must be a mapping")
        if set(given) - {"rule", "order", "limit"}:
            raise ValueError("unknown normal spec key")
        order = _integer(given["order"], "normal order")
        rule = given["rule"]
        if rule == "hermite":
            z, w = roots_hermitenorm(order)
            w = w / math.sqrt(2 * math.pi)
            spec = dict(rule=rule, order=order, limit=None)
        elif rule == "legendre":
            limit = float(given["limit"])
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("Legendre normal limit must be finite and positive")
            z, w = roots_legendre(order)
            z = limit * z
            w = limit * w * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi)
            spec = dict(rule=rule, order=order, limit=limit)
        else:
            raise ValueError("normal rule must be 'hermite' or 'legendre'")
        if not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise FloatingPointError("invalid axis quadrature")
        axes.append((z, w)); specs.append(spec)
    z0, z1 = np.meshgrid(axes[0][0], axes[1][0], indexing="ij")
    nodes = np.column_stack((z0.ravel(), z1.ravel()))
    weights = (axes[0][1][:, None] * axes[1][1][None, :]).ravel()
    return dict(normal_specs=specs, normal_nodes=nodes, normal_weights=weights,
                normal_axis_0_nodes=axes[0][0], normal_axis_0_weights=axes[0][1],
                normal_axis_1_nodes=axes[1][0], normal_axis_1_weights=axes[1][1],
                normal_axis_masses=np.array([math.fsum(float(t) for t in w) for _, w in axes]),
                normal_quadrature_mass=math.fsum(float(t) for t in weights))


def logu_nodes(logu_min, logu_max, logu_step):
    lo, hi, step = float(logu_min), float(logu_max), float(logu_step)
    if not all(math.isfinite(x) for x in (lo, hi, step)) or hi <= lo or step <= 0:
        raise ValueError("finite logu_min < logu_max and positive logu_step are required")
    n = round((hi - lo) / step)
    if n < 1 or not math.isclose(n * step, hi - lo, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("log-u width must be an integer multiple of its step")
    nodes = np.linspace(lo, hi, n + 1)
    weights = np.full(n + 1, (hi - lo) / n)
    weights[[0, -1]] *= .5
    return nodes, weights


def _exp_difference(left, right):
    """exp(left)-exp(right), including two equal negative infinities -> zero."""
    left, right = np.broadcast_arrays(left, right)
    result = np.zeros(left.shape, dtype=np.float64)
    different = left != right
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = left[different] - right[different]
        result[different] = (np.sign(delta) * np.exp(np.maximum(left[different], right[different]))
                             * -np.expm1(-np.abs(delta)))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite exponential difference")
    return result


def _logpsi_reference(logx, nodes_v, weights, logweights, mass, outer_chunk):
    """All-logsumexp reference path retained for manufactured comparisons."""
    answer = np.empty(len(nodes_v))
    for start in range(0, len(nodes_v), outer_chunk):
        stop = min(start + outer_chunk, len(nodes_v))
        with np.errstate(over="ignore", under="ignore"):
            negative_ux = -np.exp(nodes_v[start:stop, None] + logx[None, :])
        values = logsumexp(logweights[None, :] + negative_ux, axis=1)
        # Preserve the small-u deficit instead of first rounding exp(-ux) to 1.
        offset = (np.expm1(negative_ux) @ weights) / mass
        small = np.abs(offset) < .25
        values[small] = math.log(mass) + np.log1p(offset[small])
        answer[start:stop] = values
    if np.isnan(answer).any() or np.isposinf(answer).any():
        raise FloatingPointError("invalid log Laplace transform")
    return answer


def _logpsi(logx, nodes_v, weights, logweights, mass, outer_chunk):
    """Positive weighted sum, with small-u and very-small-sum safeguards.

    The same original weights are used in every branch. The logsumexp fallback
    preserves logarithms even when positive-space summation would underflow.
    Only near-mass rows need expm1 to retain their small departure from M.
    """
    answer = np.empty(len(nodes_v))
    with np.errstate(over="ignore", under="ignore"):
        x = np.exp(logx)
        u = np.exp(nodes_v)
    ordinary_scale = np.isfinite(x).all() and np.all(x > 0)
    for start in range(0, len(nodes_v), outer_chunk):
        stop = min(start + outer_chunk, len(nodes_v))
        with np.errstate(over="ignore", under="ignore"):
            # Avoid a full matrix exponential just to form u*x when each x is
            # representable. Extreme supplied scales retain the log-domain path.
            negative_ux = (-u[start:stop, None] * x[None, :] if ordinary_scale
                           else -np.exp(nodes_v[start:stop, None] + logx[None, :]))
            psi = np.exp(negative_ux) @ weights
        values = np.empty(len(psi))
        tail = psi < 1e-250
        ordinary = ~tail
        values[ordinary] = np.log(psi[ordinary])
        if np.any(tail):
            # Preserve the reference log-domain tail values as well as the
            # resulting underflowed integrands; exp(v)*exp(logx) can differ by
            # large absolute ulps when the true log Psi is a huge negative value.
            with np.errstate(over="ignore", under="ignore"):
                tail_negative_ux = -np.exp(nodes_v[start:stop][tail, None] + logx[None, :])
            values[tail] = logsumexp(logweights[None, :] + tail_negative_ux, axis=1)
        small = psi > .75 * mass
        if np.any(small):
            offset = (np.expm1(negative_ux[small]) @ weights) / mass
            values[small] = math.log(mass) + np.log1p(offset)
        answer[start:stop] = values
    if np.isnan(answer).any() or np.isposinf(answer).any():
        raise FloatingPointError("invalid log Laplace transform")
    return answer


def _model(b, W, q, Y, Ks, nodes, weights, v, wv, logscale, outer_chunk, label_chunk, keep_logA):
    mass = math.fsum(float(t) for t in weights)
    with np.errstate(divide="ignore"):
        logw = np.log(weights)
    with np.errstate(over="raise", invalid="raise"):
        logits = nodes @ W.T + b
        logyes, logno = -np.logaddexp(0., -logits), -np.logaddexp(0., logits)
    evidence, scale_logs = np.empty(len(Y)), np.empty(len(Y))
    logPsi = np.empty((len(Y), len(v)))
    stored_logA = np.empty((len(nodes), len(Y))) if keep_logA else None
    for start in range(0, len(Y), label_chunk):
        stop = min(start + label_chunk, len(Y))
        with np.errstate(over="raise", invalid="raise"):
            block = logyes @ Y[start:stop].T + logno @ (1 - Y[start:stop]).T
        if not np.isfinite(block).all():
            raise FloatingPointError("nonfinite conditional log likelihood")
        if keep_logA:
            stored_logA[:, start:stop] = block
        for local, yi in enumerate(range(start, stop)):
            logs = block[:, local]
            evidence[yi] = logsumexp(logw + logs)
            scale_logs[yi] = evidence[yi] - math.log(mass) if logscale is None else logscale[yi]
            logPsi[yi] = _logpsi(logs - scale_logs[yi], v, weights, logw, mass, outer_chunk)
    finite = tuple(K for K in Ks if K != "inf")
    label_loss = np.empty((len(Ks), len(Y)))
    integrands = np.empty((len(finite), len(Y), len(v)))
    bankmass = np.empty(len(Ks))
    fi = 0
    for ki, K in enumerate(Ks):
        if K == "inf":
            bankmass[ki] = 1.
            label_loss[ki] = -evidence
        else:
            bankmass[ki] = math.exp(K * math.log(mass))
            with np.errstate(over="ignore", under="ignore"):
                reference = K * math.log(mass) - K * np.exp(v)
            integrands[fi] = _exp_difference(K * logPsi, reference[None, :])
            label_loss[ki] = -bankmass[ki] * scale_logs + integrands[fi] @ wv
            fi += 1
    pmf = np.exp(evidence)
    qlogq = np.zeros_like(q)
    nonzero = q > 0
    qlogq[nonzero] = q[nonzero] * np.log(q[nonzero])
    rawKL = qlogq.sum(axis=1) - q @ evidence
    result = dict(conditional_logA=stored_logA, log_evidence=evidence, log_likelihood_scale=scale_logs,
                  logPsi=logPsi, label_loss=label_loss, absolute_loss=np.einsum('tp,kp->tk', q, label_loss),
                  absolute_integrand=integrands, raw_bank_quadrature_mass=bankmass,
                  pmf=pmf, marginals=pmf @ Y, model_mass=float(pmf.sum()), raw_KL=rawKL,
                  q_mass=q.sum(axis=1))
    for key in ("log_evidence", "log_likelihood_scale", "label_loss", "absolute_loss", "raw_KL"):
        if not np.isfinite(result[key]).all():
            raise FloatingPointError(f"nonfinite {key}")
    return result


def _work(N, P, V, m, finite_K_count, baseline):
    return dict(kernel_evaluations=1, baseline_preparations=int(baseline), perturbed_model_evaluations=int(not baseline),
                tensor_normal_rules_constructed=int(baseline), bernoulli_logit_values=N * m,
                conditional_loglikelihood_values=N * P, laplace_exp_minus_ux_factors=P * V * N,
                laplace_logPsi_values=P * V, finite_absolute_integrand_values=finite_K_count * P * V,
                paired_integrand_values=0 if baseline else finite_K_count * P * V,
                new_random_draws=0, optimizer_updates=0,
                interpretation="high_level_scalar_factor_counts_not_CPU_instruction_counts")


def prepare_baseline(b, W0, targets, K_values=(11, 12, "inf"), *, normal_specs=None,
                     logu_min=-28., logu_max=56., logu_step=.25, outer_chunk=32, label_chunk=8,
                     likelihood_scale=None):
    """Prepare one baseline/rule for arbitrarily many separately evaluated W1.

    Shapes: conditional_logA[N,P], logPsi[P,V], label_loss[K,P],
    absolute_loss[T,K], pmf[P], marginals[m], raw_KL[T]. Normal node order is
    first axis outer / second axis inner. finite_K_values indexes integrands.
    The full conditional_logA is returned only here for baseline preservation.
    """
    b, W0 = _real(b, "b"), _real(W0, "W0")
    if b.ndim != 1 or not len(b) or W0.shape != (len(b), 2):
        raise ValueError("b must have shape [m] and W0 shape [m,2]")
    Y = patterns(len(b))
    names, q = _targets(targets, len(Y))
    Ks = _counts(K_values)
    outer_chunk, label_chunk = _integer(outer_chunk, "outer_chunk"), _integer(label_chunk, "label_chunk")
    if normal_specs is None:
        normal_specs = ({"rule": "hermite", "order": 24}, {"rule": "hermite", "order": 24})
    normal = normal_tensor(normal_specs)
    v, wv = logu_nodes(logu_min, logu_max, logu_step)
    logs = None
    if likelihood_scale is not None:
        scale = _real(likelihood_scale, "likelihood_scale")
        if scale.ndim == 0:
            scale = np.full(len(Y), float(scale))
        if scale.shape != (len(Y),) or np.any(scale <= 0):
            raise ValueError("likelihood_scale must be positive scalar or pattern vector")
        logs = np.log(scale)
    result = _model(b, W0, q, Y, Ks, normal["normal_nodes"], normal["normal_weights"], v, wv,
                    logs, outer_chunk, label_chunk, True)
    finite = tuple(K for K in Ks if K != "inf")
    result.update(schema="noncollinear_probe_prepared_v1", b=b, W0=W0, patterns=Y, targets=q,
                  target_names=names, K_values=Ks, finite_K_values=finite,
                  **normal, outer_logu_nodes=v, outer_weights=wv,
                  config=dict(normal_specs=normal["normal_specs"], logu_min=float(logu_min), logu_max=float(logu_max),
                              logu_step=float(logu_step), outer_rule="trapezoid", outer_chunk=outer_chunk,
                              label_chunk=label_chunk, tensor_order="axis_1_fastest", baseline_folded=False,
                              scale_source="raw_baseline_evidence_divided_by_normal_mass" if logs is None else "supplied_positive_scale",
                              laplace_sum_method="positive_u_times_x_weighted_sum_with_log_domain_extreme_scale_fallback_logsumexp_below_1e-250_log1p_above_0.75M",
                              finite_loss_mass_correction=True, target_weights_renormalized=False,
                              normal_weights_renormalized=False),
                  work_counts=_work(len(normal["normal_nodes"]), len(Y), len(v), len(b), len(finite), True))
    return _snapshot(result)


def evaluate_prepared(prepared, W1):
    """Evaluate W1 without changing the prepared baseline or any input arrays.

    Shapes: absolute_loss[T,K,2], paired_delta[T,K], label_loss[K,P,2],
    label_delta[K,P], logPsi[2,P,V], paired_integrand[F,P,V],
    absolute_integrand[F,P,V,2], pmf/log_evidence[2,P], marginals[2,m],
    raw_KL[T,2], paired_risk_delta[T]. Axis 2 of absolute quantities is
    baseline/new. F follows finite_K_values and excludes infinity.
    work_counts counts only this new-model call, not baseline preparation.
    """
    if prepared.get("schema") != "noncollinear_probe_prepared_v1":
        raise ValueError("unrecognized prepared baseline")
    b, W0 = _real(prepared["b"], "prepared b"), _real(prepared["W0"], "prepared W0")
    W1 = _real(W1, "W1")
    if W1.shape != W0.shape or W0.shape != (len(b), 2):
        raise ValueError("W1 must match the baseline [m,2] shape")
    Y, q = prepared["patterns"], prepared["targets"]
    Ks = _counts(prepared["K_values"])
    v, wv = prepared["outer_logu_nodes"], prepared["outer_weights"]
    cfg = prepared["config"]
    current = _model(b, W1, q, Y, Ks, prepared["normal_nodes"], prepared["normal_weights"], v, wv,
                     prepared["log_likelihood_scale"], cfg["outer_chunk"], cfg["label_chunk"], False)
    finite = tuple(K for K in Ks if K != "inf")
    integrands = np.empty((len(finite), len(Y), len(v)))
    label_delta = np.empty((len(Ks), len(Y)))
    fi = 0
    for ki, K in enumerate(Ks):
        if K == "inf":
            label_delta[ki] = prepared["log_evidence"] - current["log_evidence"]
        else:
            integrands[fi] = _exp_difference(K * current["logPsi"], K * prepared["logPsi"])
            label_delta[ki] = integrands[fi] @ wv
            fi += 1
    paired = np.einsum('tp,kp->tk', q, label_delta)
    absolute = np.stack((prepared["absolute_loss"], current["absolute_loss"]), axis=-1)
    absolute_difference = absolute[..., 1] - absolute[..., 0]
    paired_risk = -(q @ (current["log_evidence"] - prepared["log_evidence"]))
    risk = np.stack((prepared["raw_KL"], current["raw_KL"]), axis=-1)
    result = dict(schema="noncollinear_probe_paired_v1", b=b, W0=W0, W1=W1, targets=q,
                  patterns=Y, target_names=prepared["target_names"], K_values=Ks, finite_K_values=finite,
                  absolute_loss=absolute, paired_delta=paired, label_delta=label_delta,
                  label_loss=np.stack((prepared["label_loss"], current["label_loss"]), axis=-1),
                  absolute_difference=absolute_difference, delta_consistency_error=paired - absolute_difference,
                  paired_integrand=integrands,
                  absolute_integrand=np.stack((prepared["absolute_integrand"], current["absolute_integrand"]), axis=-1),
                  logPsi=np.stack((prepared["logPsi"], current["logPsi"])),
                  log_evidence=np.stack((prepared["log_evidence"], current["log_evidence"])),
                  log_likelihood_scale=prepared["log_likelihood_scale"],
                  pmf=np.stack((prepared["pmf"], current["pmf"])),
                  marginals=np.stack((prepared["marginals"], current["marginals"])),
                  model_mass=np.array([prepared["model_mass"], current["model_mass"]]),
                  raw_KL=risk, paired_risk_delta=paired_risk,
                  risk_consistency_error=paired_risk - (risk[:, 1] - risk[:, 0]),
                  q_mass=prepared["q_mass"], normal_quadrature_mass=prepared["normal_quadrature_mass"],
                  raw_bank_quadrature_mass=prepared["raw_bank_quadrature_mass"], config=cfg,
                  work_counts=_work(len(prepared["normal_nodes"]), len(Y), len(v), len(b), len(finite), False),
                  qualifications=dict(rigorous_quadrature_remainder_certificate=False,
                                      paired_delta_from_absolute_subtraction=False,
                                      symmetry_used_to_skip_evaluation=False,
                                      target_or_normal_mass_renormalized=False))
    for key in ("normal_nodes", "normal_weights", "normal_axis_0_nodes", "normal_axis_0_weights",
                "normal_axis_1_nodes", "normal_axis_1_weights", "normal_axis_masses", "outer_logu_nodes", "outer_weights"):
        result[key] = prepared[key]
    for key in ("paired_delta", "label_delta", "paired_risk_delta"):
        if not np.isfinite(result[key]).all():
            raise FloatingPointError(f"nonfinite {key}")
    return _snapshot(result)


def paired_losses(b, W0, W1, targets, K_values=(11, 12, "inf"), **kwargs):
    """Convenience one-off call; work_counts includes preparation and evaluation."""
    baseline = prepare_baseline(b, W0, targets, K_values, **kwargs)
    result = evaluate_prepared(baseline, W1)
    result["preparation_work_counts"] = _snapshot(baseline["work_counts"])
    result["new_model_work_counts"] = _snapshot(result["work_counts"])
    for key, value in baseline["work_counts"].items():
        if isinstance(value, int):
            result["work_counts"][key] += value
    return result


def path_W(a, d, t):
    """W_j(t)=a_j(cos(t*d_j),sin(t*d_j)); no direction normalization."""
    a, d = _real(a, "a"), _real(d, "d")
    t_array = _real(t, "t")
    if a.ndim != 1 or not len(a) or d.shape != a.shape or t_array.ndim != 0:
        raise ValueError("a and d must be matching vectors and t a scalar")
    with np.errstate(over="raise", invalid="raise"):
        angle = float(t_array) * d
        W = a[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    return W.copy()


def angular_path_losses(b, a, d, t, targets, K_values=(11, 12, "inf"), **kwargs):
    result = paired_losses(b, path_W(a, d, 0.), path_W(a, d, t), targets, K_values, **kwargs)
    result.update(a=_real(a, "a").copy(), direction=_real(d, "d").copy(), t=float(t))
    return result
