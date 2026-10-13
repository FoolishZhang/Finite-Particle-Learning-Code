"""Float64 normalized binary models, convex box fits, and exact partial queries."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import itertools
import json
import math
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

FAMILIES = ("product", "cardinality", "ising", "linear_ar", "count_ar")
GAP_TOL = 1e-8


class OptimizationFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


@lru_cache(maxsize=12)
def _states(d):
    if not isinstance(d, (int, np.integer)) or not 1 <= d <= 12:
        raise ValueError("exact-state core supports integer d in 1..12")
    result = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(np.int8)
    result.setflags(write=False)
    return result


def states(d):
    """Read-only [2**d,d] binary states; bit i has integer weight 2**i."""
    return _states(int(d)) if int(d) == d else _states(d)


def _probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or len(p) & (len(p)-1):
        raise ValueError("p must be a vector with power-of-two length")
    d = len(p).bit_length()-1
    _states(d)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("p must be finite and strictly positive; caller supplies smoothing")
    mass = float(p.sum())
    if abs(mass-1) > 1e-12:
        raise ValueError("p mass outside the roundoff normalization gate")
    return p/mass, d, mass


@lru_cache(maxsize=60)
def _design(d, family):
    if family not in FAMILIES:
        raise ValueError("unknown family "+str(family))
    x = _states(d)
    spin = 2*x.astype(np.float64)-1
    if family == "product":
        features = spin
        bound = np.full(d, 8.)
    elif family == "cardinality":
        count = x.sum(axis=1)
        features = np.column_stack([spin, (count[:, None] == np.arange(2, d+1)).astype(float)])
        bound = np.r_[np.full(d, 8.), np.full(d-1, 64.)]
    elif family == "ising":
        pair = [spin[:, i]*spin[:, j] for i in range(d) for j in range(i+1, d)]
        features = np.column_stack([spin]+pair)
        bound = np.full(d*(d+1)//2, 8.)
    else:
        features = None
        bound = np.full(d*(d+1)//2, 8.)
    if features is not None:
        features = np.ascontiguousarray(features, dtype=np.float64)
        features.setflags(write=False)
    bound.setflags(write=False)
    return features, bound


@lru_cache(maxsize=12)
def _ar_contexts(d):
    tables = []
    for i in range(d):
        codes = np.arange(1 << i)
        previous = ((codes[:, None] >> np.arange(i)) & 1).astype(float)
        design = np.column_stack([np.ones(len(codes)), 2*previous-1])
        design.setflags(write=False)
        prefix_count = previous.sum(axis=1).astype(int)
        prefix_count.setflags(write=False)
        tables.append((design, prefix_count))
    return tuple(tables)


def _objective(p, d, family):
    """Return a target-specific convex value/gradient/Hessian function."""
    features, bound = _design(d, family)
    if features is not None:
        target = p @ features

        def evaluate(theta, hessian=False):
            score = features @ theta
            logz = logsumexp(score)
            q = np.exp(score-logz)
            mean = q @ features
            value = float(logz-target @ theta)
            gradient = mean-target
            if not hessian:
                return value, gradient
            centered = features-mean
            H = centered.T @ (q[:, None]*centered)
            return value, gradient, .5*(H+H.T)
        return evaluate, bound

    x = _states(d)
    ids = np.arange(len(p))
    blocks = []
    offset = 0
    for i, (design, prefix_count) in enumerate(_ar_contexts(d)):
        prefix = ids & ((1 << i)-1)
        joint = np.bincount(2*prefix+x[:, i], weights=p, minlength=2*(1 << i)).reshape(-1, 2)
        if family == "linear_ar":
            mass = joint.sum(axis=1)
            positive = joint[:, 1]
            X = design
        else:
            mass = np.bincount(prefix_count, weights=joint.sum(axis=1), minlength=i+1)
            positive = np.bincount(prefix_count, weights=joint[:, 1], minlength=i+1)
            X = None
        blocks.append((offset, i+1, X, mass, positive))
        offset += i+1

    def evaluate(theta, hessian=False):
        value = 0.
        gradient = np.zeros_like(theta)
        H = np.zeros((len(theta), len(theta))) if hessian else None
        for start, size, X, mass, positive in blocks:
            block = theta[start:start+size]
            eta = block if X is None else X @ block
            prob = expit(eta)
            value += float(mass @ np.logaddexp(0., eta)-positive @ eta)
            residual = mass*prob-positive
            gradient[start:start+size] = residual if X is None else X.T @ residual
            if hessian:
                curvature = mass*expit(eta)*expit(-eta)
                hb = np.diag(curvature) if X is None else X.T @ (curvature[:, None]*X)
                H[start:start+size, start:start+size] = hb
        return (value, gradient, H) if hessian else (value, gradient)
    return evaluate, bound


def _gap(theta, gradient, bound):
    if np.any(theta < -bound) or np.any(theta > bound):
        raise ValueError("gap requires a feasible parameter")
    return float(np.sum(np.where(gradient >= 0,
        gradient*(theta+bound), -gradient*(bound-theta))))


def _pack(theta, d, family):
    theta = np.asarray(theta, dtype=float)
    params = dict(family=family, d=d, theta=theta.tolist())
    if family in ("product", "cardinality", "ising"):
        params["spin_unary"] = theta[:d].tolist()
    if family == "cardinality":
        params["count_potential"] = [0., 0.]+theta[d:].tolist()
        params["gauge"] = "v0=v1=0"
    elif family == "ising":
        params["spin_pairs"] = theta[d:].tolist()
        params["pair_indices"] = [[i, j] for i in range(d) for j in range(i+1, d)]
    elif family in ("linear_ar", "count_ar"):
        params["order"] = list(range(d))
        params["logits"] = []
        start = 0
        for i in range(d):
            params["logits"].append(theta[start:start+i+1].tolist())
            start += i+1
    return params


def _unpack(params):
    family, d = params["family"], int(params["d"])
    _, bound = _design(d, family)
    theta = np.asarray(params["theta"], dtype=np.float64)
    if theta.shape != bound.shape or not np.isfinite(theta).all():
        raise ValueError("invalid saved parameter shape or values")
    if np.any(abs(theta) > bound):
        raise ValueError("saved parameter outside the declared box")
    canonical = _pack(theta, d, family)
    for key, expected in canonical.items():
        if key in params and params[key] != expected:
            raise ValueError("inconsistent saved parameter field "+key)
    return theta, d, family


def log_pmf_from_params(params):
    theta, d, family = _unpack(params)
    features, _ = _design(d, family)
    if features is not None:
        score = features @ theta
        return score-logsumexp(score)
    x = _states(d)
    spin = 2*x.astype(float)-1
    logq = np.zeros(len(x))
    start = 0
    for i in range(d):
        block = theta[start:start+i+1]
        eta = block[0]+spin[:, :i] @ block[1:] if family == "linear_ar" else block[x[:, :i].sum(axis=1)]
        logq += x[:, i]*eta-np.logaddexp(0., eta)
        start += i+1
    if abs(float(logsumexp(logq))) > 1e-11:
        raise OptimizationFailure("AR normalization identity failed", dict(family=family, d=d))
    return logq


def pmf_from_params(params):
    """Re-evaluate a saved model independently of any training target."""
    logq = log_pmf_from_params(params)
    q = np.exp(logq)
    if not np.isfinite(q).all() or np.any(q <= 0) or abs(float(q.sum())-1) > 1e-11:
        raise OptimizationFailure("nonfinite, underflowed, or unnormalized model PMF",
                                  dict(family=params["family"], d=params["d"]))
    return q


def fit(p, family):
    p, d, raw_mass = _probability(p)
    fun, bound = _objective(p, d, family)
    started = time.perf_counter()
    objective_calls = 0
    hessian_calls = 0

    def evaluate(theta, hessian=False):
        nonlocal objective_calls, hessian_calls
        objective_calls += 1
        hessian_calls += int(hessian)
        out = fun(theta, hessian)
        if not all(np.isfinite(v).all() for v in out):
            raise OptimizationFailure("nonfinite convex objective", dict(family=family, d=d))
        return out

    trace = []
    if family == "product":
        x = _states(d)
        p1 = p @ x
        p0 = p @ (1-x)
        theta = np.clip(.5*(np.log(p1)-np.log(p0)), -bound, bound)
        method = "closed_product_spin_MLE"
        lbfgs = None
    elif family == "count_ar":
        x = _states(d)
        theta = []
        for i in range(d):
            prefix_count = x[:, :i].sum(axis=1)
            negative = np.bincount(prefix_count, weights=p*(1-x[:, i]), minlength=i+1)
            positive = np.bincount(prefix_count, weights=p*x[:, i], minlength=i+1)
            theta.extend(np.log(positive)-np.log(negative))
        theta = np.clip(np.asarray(theta), -bound, bound)
        method = "closed_prefix_count_Bernoulli_MLE"
        lbfgs = None
    else:
        theta0 = np.zeros(len(bound))
        result = minimize(lambda z: evaluate(z), theta0, jac=True,
            bounds=list(zip(-bound, bound)), method="L-BFGS-B",
            options=dict(maxiter=3000, ftol=1e-15, gtol=1e-12, maxls=50))
        if not np.isfinite(result.x).all():
            raise OptimizationFailure("nonfinite L-BFGS-B endpoint", dict(family=family, d=d, message=str(result.message)))
        theta = np.clip(result.x, -bound, bound)
        lbfgs = dict(success=bool(result.success), status=int(result.status),
                     iterations=int(result.nit), message=str(result.message))
        method = "one_L-BFGS-B_then_damped_box_Newton"
        f, g = evaluate(theta)
        trace.append(dict(stage="L-BFGS-B", objective=f, gap=_gap(theta, g, bound)))
        for iteration in range(100):
            f, g, H = evaluate(theta, True)
            gap = _gap(theta, g, bound)
            if gap <= GAP_TOL:
                break
            active = ((theta <= -bound+1e-12) & (g >= 0)) | ((theta >= bound-1e-12) & (g <= 0))
            free = np.flatnonzero(~active)
            if not len(free):
                trace.append(dict(stage="Newton", iteration=iteration, stopped="no_free_coordinates", gap=gap))
                break
            hf = H[np.ix_(free, free)]
            least = float(np.linalg.eigvalsh(hf)[0])
            scale = max(1., float(np.max(np.diag(hf))))
            damping = max(0., 1e-12*scale-least)
            direction = np.zeros_like(theta)
            direction_kind = "damped_Newton"
            try:
                direction[free] = np.linalg.solve(hf+damping*np.eye(len(free)), -g[free])
            except np.linalg.LinAlgError:
                direction = np.clip(theta-g, -bound, bound)-theta
                direction_kind = "projected_gradient_fallback"
            if not np.isfinite(direction).all() or g @ direction >= 0:
                direction = np.clip(theta-g, -bound, bound)-theta
                direction_kind = "projected_gradient_fallback"
            alpha = 1.
            for j, value in enumerate(direction):
                if value > 0:
                    alpha = min(alpha, (bound[j]-theta[j])/value)
                elif value < 0:
                    alpha = min(alpha, (-bound[j]-theta[j])/value)
            if alpha <= 1e-15:
                direction = np.clip(theta-g, -bound, bound)-theta
                direction_kind = "projected_gradient_fallback"
                alpha = 1.
            accepted = False
            for backtrack in range(60):
                candidate = np.clip(theta+alpha*direction, -bound, bound)
                fc, _ = evaluate(candidate)
                if fc <= f+1e-4*float(g @ (candidate-theta)):
                    accepted = True
                    break
                alpha *= .5
            trace.append(dict(stage="Newton", iteration=iteration, objective=f, gap=gap,
                              damping=damping, direction=direction_kind,
                              step=alpha, backtracks=backtrack, accepted=accepted))
            if not accepted or np.array_equal(candidate, theta):
                break
            theta = candidate
    f, gradient = evaluate(theta)
    gap = _gap(theta, gradient, bound)
    params = _pack(theta, d, family)
    q = pmf_from_params(params)
    direct = -float(p @ log_pmf_from_params(params))
    if abs(direct-f) > 1e-10:
        raise OptimizationFailure("objective and normalized-PMF NLL disagree",
                                  dict(family=family, objective=f, PMF_objective=direct))
    converged = gap <= GAP_TOL
    receipt = dict(method=method, status="certified" if converged else "gap_not_met",
        converged=converged, L_BFGS_B=lbfgs, trace=trace,
        objective_calls=objective_calls, hessian_calls=hessian_calls,
        elapsed_seconds=time.perf_counter()-started,
        max_L_BFGS_iterations=3000, max_Newton_iterations=100,
        max_backtracks_per_Newton=60, hidden_restarts=0,
        gradient=gradient.tolist(), lower_bounds=(-bound).tolist(),
        upper_bounds=bound.tolist(), target_raw_mass=raw_mass,
        certificate_scope="convex NLL in the declared finite parameter box; float64 arithmetic")
    return dict(q=q, params=params, objective=f, gap=gap, lower_bound=f-gap,
                upper_bound=f, converged=converged, family=family, d=d,
                parameter_count=len(theta), optimizer_receipt=receipt)


@lru_cache(maxsize=144)
def _query_partitions(d, k):
    if not 0 <= k < d:
        raise ValueError("evidence size must satisfy 0 <= k < d")
    x = _states(d)
    result = []
    for observed in itertools.combinations(range(d), k):
        code = np.zeros(len(x), dtype=int)
        for bit, j in enumerate(observed):
            code |= x[:, j].astype(int) << bit
        for target in range(d):
            if target not in observed:
                labels = 2*code+x[:, target]
                labels.setflags(write=False)
                result.append(labels)
    return tuple(result)


def query_metrics(p, q, k):
    p, d, _ = _probability(p)
    q, dq, _ = _probability(q)
    if dq != d:
        raise ValueError("p and q dimensions differ")
    partitions = _query_partitions(d, int(k))
    KL, raw_brier, excess_brier, bayes_brier = 0., 0., 0., 0.
    for labels in partitions:
        pjoint = np.bincount(labels, weights=p, minlength=2*(1 << k)).reshape(-1, 2)
        qjoint = np.bincount(labels, weights=q, minlength=2*(1 << k)).reshape(-1, 2)
        pe, qe = pjoint.sum(axis=1), qjoint.sum(axis=1)
        logp = np.log(pjoint)-np.log(pe[:, None])
        logq = np.log(qjoint)-np.log(qe[:, None])
        KL += float(np.sum(pjoint*(logp-logq)))
        pc, qc = pjoint/pe[:, None], qjoint/qe[:, None]
        raw_brier += float(np.sum(pjoint[:, 0]*qc[:, 1]**2+pjoint[:, 1]*qc[:, 0]**2))
        excess_brier += float(pe @ ((pc[:, 1]-qc[:, 1])**2))
        bayes_brier += float(pe @ (pc[:, 0]*pc[:, 1]))
    count = len(partitions)
    return dict(conditional_excess_KL=KL/count, brier=raw_brier/count,
                excess_Brier=excess_brier/count, excess_brier=excess_brier/count,
                bayes_brier=bayes_brier/count, evidence_size=int(k),
                query_configurations=count)


def _evidence(evidence, target, d):
    if not isinstance(target, (int, np.integer)) or not 0 <= target < d:
        raise ValueError("invalid target")
    result = {}
    for key, value in evidence.items():
        if not isinstance(key, (int, np.integer)) or not 0 <= key < d or value not in (0, 1):
            raise ValueError("evidence requires valid coordinate indices and binary values")
        result[int(key)] = int(value)
    return result


def conditional_query(q, evidence, target):
    q, d, _ = _probability(q)
    evidence = _evidence(evidence, target, d)
    if target in evidence:
        return float(evidence[target])
    x = _states(d)
    compatible = np.ones(len(q), dtype=bool)
    for i, value in evidence.items():
        compatible &= x[:, i] == value
    zero = float(q[compatible & (x[:, target] == 0)].sum())
    one = float(q[compatible & (x[:, target] == 1)].sum())
    return one/(zero+one)


def _count_ar_logmass(logits, evidence):
    d = len(logits)
    current = np.full(d+1, -np.inf)
    current[0] = 0.
    for i in range(d):
        row = np.asarray(logits[i], dtype=float)
        if row.shape != (i+1,) or not np.isfinite(row).all():
            raise ValueError("count_ar logits row i must have length i+1")
        nxt = np.full(d+1, -np.inf)
        choices = (evidence[i],) if i in evidence else (0, 1)
        for k in range(i+1):
            for bit in choices:
                logprob = -np.logaddexp(0., -row[k] if bit else row[k])
                nxt[k+bit] = np.logaddexp(nxt[k+bit], current[k]+logprob)
        current = nxt
    return float(logsumexp(current))


def count_ar_query(logits, evidence, target):
    d = len(logits)
    evidence = _evidence(evidence, target, d)
    if target in evidence:
        return float(evidence[target])
    logmass = [_count_ar_logmass(logits, evidence | {int(target): bit}) for bit in (0, 1)]
    return float(np.exp(logmass[1]-logsumexp(logmass)))


def _card_logmass(unary, potential, evidence):
    d = len(unary)
    observed_ones = sum(evidence.values())
    unknown = [i for i in range(d) if i not in evidence]
    constant = sum(unary[i]*(2*bit-1) for i, bit in evidence.items())-sum(unary[i] for i in unknown)
    dp = np.full(len(unknown)+1, -np.inf)
    dp[0] = 0.
    used = 0
    for i in unknown:
        nxt = dp.copy()
        for k in range(used+1):
            nxt[k+1] = np.logaddexp(dp[k+1], 2*unary[i]+dp[k])
        dp = nxt
        used += 1
    return float(constant+logsumexp(dp+potential[observed_ones:observed_ones+len(unknown)+1]))


def cardinality_query(params, evidence, target):
    theta, d, family = _unpack(params)
    if family != "cardinality":
        raise ValueError("cardinality_query requires cardinality parameters")
    evidence = _evidence(evidence, target, d)
    if target in evidence:
        return float(evidence[target])
    unary, potential = theta[:d], np.r_[0., 0., theta[d:]]
    logmass = [_card_logmass(unary, potential, evidence | {int(target): bit}) for bit in (0, 1)]
    return float(np.exp(logmass[1]-logsumexp(logmass)))


def pushforward_flip(q, mask):
    q, d, _ = _probability(q)
    if not isinstance(mask, (int, np.integer)) or not 0 <= mask < (1 << d):
        raise ValueError("invalid XOR mask")
    return q[np.arange(len(q)) ^ int(mask)]


def _flip_spin_params(params, mask):
    theta, d, family = _unpack(params)
    signs = 1-2*((int(mask) >> np.arange(d)) & 1)
    theta = theta.copy()
    if family == "product":
        theta *= signs
    elif family == "ising":
        theta[:d] *= signs
        for offset, (i, j) in enumerate(itertools.combinations(range(d), 2), start=d):
            theta[offset] *= signs[i]*signs[j]
    elif family == "linear_ar":
        offset = 0
        for i in range(d):
            theta[offset] *= signs[i]
            theta[offset+1:offset+i+1] *= signs[i]*signs[:i]
            offset += i+1
    else:
        raise ValueError("no finite-box flip-closure claim for this family")
    return _pack(theta, d, family)


def synthetic_checks():
    """Deterministic implementation fixtures only; no scientific observations."""
    d = 4
    x = _states(d)
    y = 2*x.astype(float)-1
    p = np.exp(.13*y[:, 0]-.09*y[:, 1]+.11*y[:, 1]*y[:, 2])
    p /= p.sum()
    maximum_gradient, maximum_hessian, maximum_DP, maximum_flip = 0., 0., 0., 0.
    fits = []
    for family in FAMILIES:
        fun, bounds = _objective(p, d, family)
        theta = .17*np.sin(np.arange(len(bounds))+1)
        value, gradient, H = fun(theta, True)
        h = 1e-6
        for j in range(len(theta)):
            step = np.zeros_like(theta)
            step[j] = h
            numerical = (fun(theta+step)[0]-fun(theta-step)[0])/(2*h)
            maximum_gradient = max(maximum_gradient, abs(numerical-gradient[j]))
        direction = np.cos(np.arange(len(theta))+1)
        direction /= np.linalg.norm(direction)
        numerical_Hv = (fun(theta+h*direction)[1]-fun(theta-h*direction)[1])/(2*h)
        maximum_hessian = max(maximum_hessian, float(np.max(abs(numerical_Hv-H @ direction))))
        params = _pack(theta, d, family)
        q = pmf_from_params(params)
        assert abs(q.sum()-1) < 1e-12
        if family in ("product", "ising", "linear_ar"):
            transformed = pmf_from_params(_flip_spin_params(params, 5))
            maximum_flip = max(maximum_flip, float(np.max(abs(transformed-pushforward_flip(q, 5)))))
        if family in ("cardinality", "count_ar"):
            for target in range(d):
                others = [i for i in range(d) if i != target]
                for ternary in itertools.product((-1, 0, 1), repeat=d-1):
                    evidence = {i: bit for i, bit in zip(others, ternary) if bit >= 0}
                    actual = (cardinality_query(params, evidence, target) if family == "cardinality"
                              else count_ar_query(params["logits"], evidence, target))
                    maximum_DP = max(maximum_DP, abs(actual-conditional_query(q, evidence, target)))
        fit_result = fit(p, family)
        assert fit_result["converged"] and fit_result["gap"] <= GAP_TOL
        assert np.array_equal(pmf_from_params(fit_result["params"]), fit_result["q"])
        fits.append(dict(family=family, gap=fit_result["gap"], objective=fit_result["objective"]))
        metric = query_metrics(p, fit_result["q"], 2)
        assert abs(metric["brier"]-metric["bayes_brier"]-metric["excess_Brier"]) < 1e-12
    assert maximum_gradient < 2e-7 and maximum_hessian < 2e-6
    assert maximum_DP < 1e-12 and maximum_flip < 1e-12
    return dict(passed=True, synthetic_optimizer_checks=5, d=d,
        maximum_gradient_finite_difference_error=maximum_gradient,
        maximum_Hessian_vector_difference=maximum_hessian,
        maximum_DP_vs_enumeration_error=maximum_DP,
        maximum_closed_spin_box_flip_error=maximum_flip, fits=fits,
        cardinality_box_flip_closure_not_claimed=True,
        count_ar_partial_flip_closure_not_claimed=True,
        scientific_observations=0, scientific_benchmark_fits=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(synthetic_checks(), ensure_ascii=False, indent=2))
