"""Standalone convex laminar-cardinality and classical structure-learning core."""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import hashlib
import itertools
import json
import math
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from v3_query_core import states, query_metrics, conditional_query, pushforward_flip

GAP_TOL = 1e-8
TIE_TOL = 1e-12
FLOAT_EPS = np.finfo(np.float64).eps


def _probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or len(p) & (len(p)-1):
        raise ValueError("PMF length must be a power of two")
    d = len(p).bit_length()-1
    if not 1 <= d <= 8 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("pilot requires d<=8 and a finite strictly positive PMF")
    mass = float(p.sum())
    if abs(mass-1) > 1e-12:
        raise ValueError("PMF mass outside roundoff gate")
    return p/mass, d


def _nodes(d, nodes):
    result = [tuple(sorted(int(i) for i in node)) for node in nodes]
    if len(set(result)) != len(result):
        raise ValueError("duplicate count node")
    for node in result:
        if not 2 <= len(node) <= d or len(set(node)) != len(node) or min(node) < 0 or max(node) >= d:
            raise ValueError("internal nodes must be distinct subsets of size 2..d")
    if tuple(range(d)) not in result:
        raise ValueError("the full root must be included")
    sets = [set(n) for n in result]
    for i, left in enumerate(sets):
        for right in sets[i+1:]:
            if left & right and not (left <= right or right <= left):
                raise ValueError("count nodes must be laminar")
    return tuple(sorted(result, key=lambda n: (len(n), n)))


@lru_cache(maxsize=512)
def _rc_design(d, nodes):
    x = states(d)
    pieces = [2*x.astype(float)-1]
    bounds = [8.]*d
    for node in nodes:
        count = x[:, node].sum(axis=1)
        pieces.append((count[:, None] == np.arange(2, len(node)+1)).astype(float))
        bounds.extend([64.]*(len(node)-1))
    features = np.ascontiguousarray(np.column_stack(pieces))
    bound = np.array(bounds)
    features.setflags(write=False)
    bound.setflags(write=False)
    return features, bound


@lru_cache(maxsize=8)
def _ising_design(d):
    y = 2*states(d).astype(float)-1
    pairs = list(itertools.combinations(range(d), 2))
    features = np.ascontiguousarray(np.column_stack([y]+[y[:, i]*y[:, j] for i, j in pairs]))
    features.setflags(write=False)
    return features


def _bounds(bounds, n):
    value = np.asarray(bounds, dtype=float)
    if value.ndim == 0:
        value = np.full(n, float(value))
    if value.shape == (n,):
        if np.any(value <= 0):
            raise ValueError("symmetric bound magnitudes must be positive")
        lower, upper = -value, value
    elif value.shape == (n, 2):
        lower, upper = value[:, 0], value[:, 1]
    else:
        raise ValueError("bounds must be scalar, positive [p], or [p,2]")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError("invalid finite box")
    if np.any(lower > 0) or np.any(upper < 0):
        raise ValueError("the predefined zero initialization must be feasible")
    return lower, upper


def _box_gap(theta, gradient, lower, upper):
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("infeasible certificate point")
    return float(np.sum(np.where(gradient >= 0, gradient*(theta-lower),
                                -gradient*(upper-theta))))


def _exp_objective(p, features):
    target = p @ features
    def f(theta, hessian=False):
        score = features @ theta
        logq = score-logsumexp(score)
        q = np.exp(logq)
        value = -float(p @ logq)  # avoids cancellation of two large energy terms
        average = q @ features
        gradient = average-target
        if not hessian:
            return value, gradient
        centered = features-average
        H = centered.T @ (q[:, None]*centered)
        return value, gradient, .5*(H+H.T)
    return f


def fit_exp(p, features, bounds, metadata=None):
    p, d = _probability(p)
    F = np.ascontiguousarray(features, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] != len(p) or not np.isfinite(F).all() or F.shape[1] == 0:
        raise ValueError("invalid feature matrix")
    lower, upper = _bounds(bounds, F.shape[1])
    fun = _exp_objective(p, F)
    calls = 0
    hessians = 0
    started = time.perf_counter()
    def evaluate(theta, hessian=False):
        nonlocal calls, hessians
        calls += 1
        hessians += int(hessian)
        output = fun(theta, hessian)
        if not all(np.isfinite(v).all() for v in output):
            raise FloatingPointError("nonfinite exponential-family objective")
        return output
    result = minimize(lambda z: evaluate(z), np.zeros(F.shape[1]), jac=True,
        bounds=list(zip(lower, upper)), method="L-BFGS-B",
        options=dict(maxiter=3000, ftol=1e-15, gtol=1e-12, maxls=50))
    if not np.isfinite(result.x).all():
        raise FloatingPointError("nonfinite L-BFGS endpoint")
    theta = np.clip(result.x, lower, upper)
    trace = []
    relaxed_steps = 0
    for iteration in range(100):
        value, gradient = evaluate(theta)
        gap = _box_gap(theta, gradient, lower, upper)
        if gap <= GAP_TOL:
            break
        _, _, H = evaluate(theta, True)
        active = ((theta <= lower+1e-12) & (gradient >= 0)) | ((theta >= upper-1e-12) & (gradient <= 0))
        free = np.flatnonzero(~active)
        if not len(free):
            trace.append(dict(iteration=iteration, stopped="no free coordinates", gap=gap))
            break
        hf = H[np.ix_(free, free)]
        damping = max(0., 1e-12*max(1., float(np.max(np.diag(hf))))-float(np.linalg.eigvalsh(hf)[0]))
        direction = np.zeros_like(theta)
        kind = "damped_Newton"
        try:
            direction[free] = np.linalg.solve(hf+damping*np.eye(len(free)), -gradient[free])
        except np.linalg.LinAlgError:
            direction = np.clip(theta-gradient, lower, upper)-theta
            kind = "projected_gradient"
        if not np.isfinite(direction).all() or gradient @ direction >= 0:
            direction = np.clip(theta-gradient, lower, upper)-theta
            kind = "projected_gradient"
        step = 1.
        for i, value_i in enumerate(direction):
            if value_i > 0:
                step = min(step, (upper[i]-theta[i])/value_i)
            elif value_i < 0:
                step = min(step, (lower[i]-theta[i])/value_i)
        if step <= 1e-15:
            direction = np.clip(theta-gradient, lower, upper)-theta
            kind, step = "projected_gradient", 1.
        accepted = False
        relaxed = False
        for backtrack in range(60):
            candidate = np.clip(theta+step*direction, lower, upper)
            candidate_f, candidate_g = evaluate(candidate)
            candidate_gap = _box_gap(candidate, candidate_g, lower, upper)
            armijo = value+1e-4*float(gradient @ (candidate-theta))
            slack = 16*FLOAT_EPS*max(1., abs(value), abs(candidate_f))
            strict = candidate_f <= armijo
            relaxed = (not strict and candidate_f <= armijo+slack and candidate_gap < gap)
            if strict or relaxed:
                accepted = True
                break
            step *= .5
        trace.append(dict(iteration=iteration, objective=value, gap=gap, damping=damping,
                          direction=kind, step=step, backtracks=backtrack,
                          accepted=accepted, roundoff_relaxed=relaxed,
                          roundoff_slack=slack, candidate_gap=candidate_gap))
        relaxed_steps += int(accepted and relaxed)
        if not accepted or np.array_equal(candidate, theta):
            break
        theta = candidate
    objective, gradient = evaluate(theta)
    gap = _box_gap(theta, gradient, lower, upper)
    score = F @ theta
    logq = score-logsumexp(score)
    q = np.exp(logq)
    positive = bool(np.isfinite(q).all() and np.all(q > 0) and abs(float(q.sum())-1) < 1e-11)
    qualified = gap <= GAP_TOL and positive
    params = dict(metadata or {"family": "feature_table"})
    params.update(d=d, theta=theta.tolist(), feature_count=F.shape[1],
                  feature_sha256=hashlib.sha256(F.tobytes()).hexdigest(),
                  lower_bounds=lower.tolist(), upper_bounds=upper.tolist())
    receipt = dict(status="certified" if qualified else "gap_not_met" if positive else "nonpositive_float_PMF",
        qualified=qualified, method="one L-BFGS-B then at most 100 damped Newton steps",
        lbfgs_iterations=int(result.nit), lbfgs_success=bool(result.success), lbfgs_message=str(result.message),
        max_lbfgs_iterations=3000, max_Newton_iterations=100, max_backtracks=60,
        objective_calls=calls, hessian_calls=hessians, relaxed_steps=relaxed_steps,
        roundoff_policy="16 eps max(1,abs(f),abs(fc)); relaxed step additionally requires smaller box gap",
        hidden_restarts=0, gradient=gradient.tolist(), trace=trace,
        elapsed_seconds=time.perf_counter()-started, float_PMF_positive=positive,
        certificate_scope="fixed finite feature/parameter box, float64 supporting plane")
    return dict(q=q, logq=logq, params=params, theta=theta.copy(), objective=objective,
                gap=gap, lower_bound=objective-gap, upper_bound=objective,
                qualified=qualified, converged=qualified, parameter_count=F.shape[1],
                receipt=receipt, optimizer_receipt=receipt)


def fit_rc(p, nodes):
    p, d = _probability(p)
    nodes = _nodes(d, nodes)
    F, bound = _rc_design(d, nodes)
    result = fit_exp(p, F, bound, metadata=dict(family="rc", nodes=[list(n) for n in nodes],
        gauge="each internal count potential v0=v1=0; leaf spin unaries"))
    params = result["params"]
    params["spin_unary"] = params["theta"][:d]
    offset = d
    potentials = []
    for node in nodes:
        size = len(node)-1
        potentials.append([0., 0.]+params["theta"][offset:offset+size])
        offset += size
    params["node_potentials"] = potentials
    return result


def fit_cardinality(p):
    _, d = _probability(p)
    result = fit_rc(p, [tuple(range(d))])
    result["params"]["family"] = "cardinality"
    return result


def fit_ising(p):
    _, d = _probability(p)
    return fit_exp(p, _ising_design(d), 8., metadata=dict(family="ising",
        pair_indices=[list(x) for x in itertools.combinations(range(d), 2)]))


def fit_product(p):
    p, d = _probability(p)
    x = states(d)
    theta = np.clip(.5*(np.log(p @ x)-np.log(p @ (1-x))), -8, 8)
    F = 2*x.astype(float)-1
    f, g = _exp_objective(p, F)(theta)
    bound = np.full(d, 8.)
    gap = _box_gap(theta, g, -bound, bound)
    params = dict(family="product", d=d, theta=theta.tolist(), spin_unary=theta.tolist())
    q = pmf_from_params(params)
    receipt = dict(status="certified" if gap <= GAP_TOL else "gap_not_met",
                   method="closed clipped spin Bernoulli MLE", hidden_restarts=0,
                   certificate_scope="spin unary box [-8,8]", gradient=g.tolist())
    return dict(q=q, params=params, objective=f, gap=gap, lower_bound=f-gap, upper_bound=f,
                qualified=gap <= GAP_TOL, converged=gap <= GAP_TOL, parameter_count=d,
                receipt=receipt, optimizer_receipt=receipt)


def fit_histogram(p):
    p, d = _probability(p)
    objective = -float(p @ np.log(p))
    receipt = dict(status="closed_form", method="supplied smoothed target PMF; no extra pseudocount",
                   certificate_scope="unconstrained positive full-state distribution")
    return dict(q=p.copy(), params=dict(family="histogram", d=d, q=p.tolist()),
                objective=objective, gap=0., lower_bound=objective, upper_bound=objective,
                qualified=True, converged=True, parameter_count=len(p)-1,
                receipt=receipt, optimizer_receipt=receipt)


def _marginals(p):
    p, d = _probability(p)
    x = states(d)
    single = np.array([[float(p[x[:, i] == bit].sum()) for bit in (0, 1)] for i in range(d)])
    pair = {}
    for i, j in itertools.combinations(range(d), 2):
        pair[(i, j)] = np.bincount(2*x[:, i]+x[:, j], weights=p, minlength=4).reshape(2, 2)
    return single, pair


def edge_scores(p, kind, q=None):
    p, d = _probability(p)
    single, pair = _marginals(p)
    model_pair = _marginals(q)[1] if q is not None else None
    aliases = {"MI": "mi", "absolute_residual": "residual", "bernoulli_gain": "gain"}
    kind = aliases.get(kind, kind)
    if kind not in ("mi", "residual", "gain", "agreement"):
        raise ValueError("edge kind must be mi, residual, gain, or agreement")
    if kind in ("residual", "gain") and model_pair is None:
        raise ValueError("residual/gain need a training-fitted baseline PMF q")
    weights = np.zeros((d, d))
    for (i, j), joint in pair.items():
        event = np.array([joint[0, 0]+joint[1, 1], joint[0, 1]+joint[1, 0]])
        if kind == "mi":
            value = float(np.sum(joint*np.log(joint/(single[i, :, None]*single[j, None, :]))))
        elif kind == "agreement":
            value = float(event[0])
        else:
            base = model_pair[(i, j)]
            event_q = np.array([base[0, 0]+base[1, 1], base[0, 1]+base[1, 0]])
            value = (abs((event[0]-event[1])-(event_q[0]-event_q[1])) if kind == "residual"
                     else float(event @ np.log(event/event_q)))
        weights[i, j] = weights[j, i] = value
    return weights


@lru_cache(maxsize=32)
def _matchings(d, b):
    if not 0 <= b <= d//2:
        raise ValueError("invalid matching cardinality")
    def visit(available, remaining):
        if remaining == 0:
            yield ()
            return
        if len(available) < 2*remaining:
            return
        first = available[0]
        for pos in range(1, len(available)):
            second = available[pos]
            rest = available[1:pos]+available[pos+1:]
            for tail in visit(rest, remaining-1):
                yield ((first, second),)+tail
        if len(available) > 2*remaining:
            yield from visit(available[1:], remaining)
    result = tuple(sorted(set(visit(tuple(range(d)), b))))
    if d == 8 and b == 4:
        assert len(result) == 105
    return result


def matching(d, weights, b=4, return_details=False):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (d, d) or not np.isfinite(weights).all() or np.max(abs(weights-weights.T)) > 1e-12:
        raise ValueError("matching needs a finite symmetric weight matrix")
    candidates = _matchings(d, b)
    values = [math.fsum(float(weights[i, j]) for i, j in edges) for edges in candidates]
    best = max(values)
    chosen = next(i for i, v in enumerate(values) if v >= best-TIE_TOL)
    pairs = [list(edge) for edge in candidates[chosen]]
    if not return_details:
        return pairs
    return dict(pairs=pairs, score=values[chosen], maximum_score=best,
                score_gap=best-values[chosen], candidates=len(candidates),
                selected_index=chosen, tie_tolerance=TIE_TOL, tie_break="lexicographic edge tuples",
                not_a_global_RC_likelihood_structure_certificate=True)


def agglomerative_tree(p, return_details=False):
    p, d = _probability(p)
    similarity = edge_scores(p, "agreement")
    active = [(i,) for i in range(d)]
    nodes, steps = [], []
    while len(active) > 1:
        options = []
        for left, right in itertools.combinations(sorted(active), 2):
            score = math.fsum(float(similarity[i, j]) for i in left for j in right)/(len(left)*len(right))
            options.append((left, right, score))
        best = max(item[2] for item in options)
        left, right, score = next(item for item in options if item[2] >= best-TIE_TOL)
        merged = tuple(sorted(left+right))
        active.remove(left)
        active.remove(right)
        active.append(merged)
        nodes.append(merged)
        steps.append(dict(left=list(left), right=list(right), merged=list(merged),
                          average_agreement=score, maximum_available_score=best))
    canonical = [list(node) for node in _nodes(d, nodes)]
    if not return_details:
        return canonical
    return dict(nodes=canonical, steps=steps, similarity="empirical P(Ci=Cj)",
                linkage="average", tie_tolerance=TIE_TOL,
                description="classical agreement average-link; Tarlow-inspired, not exact reproduction",
                not_a_global_RC_likelihood_structure_certificate=True)


def fit_chow_liu(p):
    p, d = _probability(p)
    single, pairs = _marginals(p)
    MI = edge_scores(p, "mi")
    edges = sorted(pairs, key=lambda e: (-float(MI[e]), e))
    parent = list(range(d))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    selected = []
    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            selected.append((i, j))
        if len(selected) == d-1:
            break
    selected.sort()
    params = dict(family="chow_liu", d=d, edges=[list(e) for e in selected],
                  node_marginals=single.tolist(), edge_marginals=[pairs[e].tolist() for e in selected])
    q = pmf_from_params(params)
    objective = -float(p @ np.log(q))
    entropy_sum = -float(np.sum(single*np.log(single)))
    max_weight = math.fsum(float(MI[e]) for e in selected)
    identity_optimum = entropy_sum-max_weight
    error = abs(objective-identity_optimum)
    gap = error
    receipt = dict(status="certified" if error <= GAP_TOL else "identity_not_met",
        method="exact computed-MI maximum spanning tree plus empirical pair marginals",
        structural_weight=max_weight, likelihood_identity_error=error,
        unrounded_identity_optimum=identity_optimum,
        certificate_scope="unconstrained positive binary tree MLE; not a finite-logit-box certificate",
        tie_break="descending computed MI, exact ties lexicographic edges", new_iterative_optimization=False)
    return dict(q=q, params=params, objective=objective, gap=gap,
                lower_bound=objective-gap, upper_bound=objective, qualified=gap <= GAP_TOL,
                converged=gap <= GAP_TOL, parameter_count=2*d-1,
                receipt=receipt, optimizer_receipt=receipt)


def pmf_from_params(params, features=None):
    family, d = params["family"], int(params["d"])
    x = states(d)
    if family == "histogram":
        return _probability(params["q"])[0]
    if family == "chow_liu":
        single = np.asarray(params["node_marginals"])
        degree = np.zeros(d, dtype=int)
        logq = np.zeros(len(x))
        for edge, marginal in zip(params["edges"], params["edge_marginals"]):
            i, j = edge
            degree[[i, j]] += 1
            marginal = np.asarray(marginal)
            if np.any(marginal <= 0):
                raise ValueError("tree marginal must be positive")
            logq += np.log(marginal[x[:, i], x[:, j]])
        for i in range(d):
            logq += (1-degree[i])*np.log(single[i, x[:, i]])
        if abs(float(logsumexp(logq))) > 1e-10:
            raise ValueError("tree factorization is not normalized")
    else:
        theta = np.asarray(params["theta"], dtype=float)
        if family in ("rc", "cardinality"):
            nodes = _nodes(d, params["nodes"])
            F, _ = _rc_design(d, nodes)
        elif family == "ising":
            F = _ising_design(d)
        elif family == "product":
            F = 2*x.astype(float)-1
        elif family == "feature_table" and features is not None:
            F = np.ascontiguousarray(features, dtype=np.float64)
            if hashlib.sha256(F.tobytes()).hexdigest() != params["feature_sha256"]:
                raise ValueError("generic feature table hash differs")
        else:
            raise ValueError("unknown family or missing generic feature table")
        score = F @ theta
        logq = score-logsumexp(score)
    q = np.exp(logq)
    if not np.isfinite(q).all() or np.any(q <= 0) or abs(float(q.sum())-1) > 1e-10:
        raise FloatingPointError("invalid/underflowed probability; no floor applied")
    return q/q.sum()


def _evidence(evidence, target, d):
    if not isinstance(target, (int, np.integer)) or not 0 <= target < d:
        raise ValueError("invalid target")
    result = {}
    for i, bit in evidence.items():
        if not isinstance(i, (int, np.integer)) or not 0 <= i < d or bit not in (0, 1):
            raise ValueError("evidence must map valid coordinates to binary values")
        result[int(i)] = int(bit)
    return result


def _convolve(left, right):
    out = np.full(len(left)+len(right)-1, -np.inf)
    for i, value in enumerate(left):
        out[i:i+len(right)] = np.logaddexp(out[i:i+len(right)], value+right)
    return out


@lru_cache(maxsize=512)
def _rc_topology(d, nodes):
    # Internal nodes are sorted by size, so the first strict superset is
    # the parent. Laminarity makes that parent unique. This preparation
    # is separate from the quadratic convolution arithmetic per query.
    masks = {node: sum(1 << i for i in node) for node in nodes}
    children = {node: [] for node in nodes}
    for node in nodes:
        parent = next((other for other in nodes if len(other) > len(node)
                       and masks[node] & masks[other] == masks[node]), None)
        if parent is not None:
            children[parent].append(node)
    result = {}
    for node in nodes:
        covered = set().union(*children[node]) if children[node] else set()
        result[node] = (tuple(children[node]), tuple(i for i in node if i not in covered))
    return result


def _rc_logmass(params, evidence):
    d = int(params["d"])
    nodes = _nodes(d, params["nodes"])
    theta = np.asarray(params["theta"], dtype=float)
    unary = theta[:d]
    potentials, offset = {}, d
    for node in nodes:
        potentials[node] = np.r_[0., 0., theta[offset:offset+len(node)-1]]
        offset += len(node)-1
    if offset != len(theta):
        raise ValueError("RC parameter shape differs")
    topology = _rc_topology(d, nodes)
    def leaf(i):
        result = np.array([-unary[i], unary[i]])
        if i in evidence:
            result[1-evidence[i]] = -np.inf
        return result
    def message(node):
        children, leaves = topology[node]
        pieces = [message(child) for child in children]+[leaf(i) for i in leaves]
        result = np.array([0.])
        for piece in pieces:
            result = _convolve(result, piece)
        assert len(result) == len(node)+1
        return result+potentials[node]
    return float(logsumexp(message(tuple(range(d)))))


def dp_query(params, evidence, target):
    if params["family"] not in ("rc", "cardinality"):
        raise ValueError("RC DP requires RC/cardinality parameters")
    d = int(params["d"])
    evidence = _evidence(evidence, target, d)
    if target in evidence:
        return float(evidence[target])
    masses = [_rc_logmass(params, evidence | {int(target): bit}) for bit in (0, 1)]
    return float(np.exp(masses[1]-logsumexp(masses)))


def tree_query(params, evidence, target):
    if params["family"] != "chow_liu":
        raise ValueError("tree_query requires Chow-Liu parameters")
    d = int(params["d"])
    evidence = _evidence(evidence, target, d)
    if target in evidence:
        return float(evidence[target])
    single = np.asarray(params["node_marginals"])
    adjacency = [[] for _ in range(d)]
    potentials = {}
    for edge, value in zip(params["edges"], params["edge_marginals"]):
        i, j = edge
        adjacency[i].append(j)
        adjacency[j].append(i)
        potentials[(i, j)] = np.log(np.asarray(value))
        potentials[(j, i)] = potentials[(i, j)].T
    local = [(1-len(adjacency[i]))*np.log(single[i]) for i in range(d)]
    for i, bit in evidence.items():
        local[i] = local[i].copy()
        local[i][1-bit] = -np.inf
    def belief(i, parent):
        value = local[i].copy()
        for child in adjacency[i]:
            if child != parent:
                value += logsumexp(belief(child, i)[:, None]+potentials[(child, i)], axis=0)
        return value
    posterior = belief(int(target), -1)
    return float(np.exp(posterior[1]-logsumexp(posterior)))


def synthetic_checks():
    d = 4
    x = states(d)
    y = 2*x.astype(float)-1
    p = np.exp(.21*y[:, 0]*y[:, 1]-.17*y[:, 2]*y[:, 3]+.10*y[:, 0])
    p /= p.sum()
    root = tuple(range(d))
    matching_nodes = [root, (0, 1), (2, 3)]
    nodes = _nodes(d, matching_nodes)
    F, bound = _rc_design(d, nodes)
    assert np.linalg.matrix_rank(F-F[:1]) == F.shape[1]
    fun = _exp_objective(p, F)
    theta = .09*np.cos(np.arange(len(bound))+1)
    _, g, H = fun(theta, True)
    gradient_error = 0.
    h = 1e-6
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        gradient_error = max(gradient_error, abs((fun(theta+e)[0]-fun(theta-e)[0])/(2*h)-g[i]))
    direction = np.sin(np.arange(len(theta))+1)
    direction /= np.linalg.norm(direction)
    hessian_error = float(np.max(abs((fun(theta+h*direction)[1]-fun(theta-h*direction)[1])/(2*h)-H @ direction)))
    assert gradient_error < 2e-7 and hessian_error < 2e-6
    fitters = [("product", lambda: fit_product(p)), ("cardinality", lambda: fit_cardinality(p)),
               ("ising", lambda: fit_ising(p)), ("rc_matching", lambda: fit_rc(p, matching_nodes)),
               ("rc_agreement", lambda: fit_rc(p, agglomerative_tree(p))),
               ("chow_liu", lambda: fit_chow_liu(p)), ("histogram", lambda: fit_histogram(p))]
    fits, dp_error = [], 0.
    for name, fitter in fitters:
        result = fitter()
        assert result["qualified"] and result["gap"] <= GAP_TOL
        assert np.max(abs(result["q"]-pmf_from_params(result["params"]))) < 1e-12
        fits.append(dict(family=name, gap=result["gap"], parameter_count=result["parameter_count"]))
        if result["params"]["family"] in ("rc", "cardinality", "chow_liu"):
            for target in range(d):
                remaining = [i for i in range(d) if i != target]
                for code in itertools.product((-1, 0, 1), repeat=d-1):
                    evidence = {i: v for i, v in zip(remaining, code) if v >= 0}
                    value = (tree_query(result["params"], evidence, target)
                             if name == "chow_liu" else dp_query(result["params"], evidence, target))
                    dp_error = max(dp_error, abs(value-conditional_query(result["q"], evidence, target)))
    assert dp_error < 1e-11
    mi = matching(d, edge_scores(p, "mi"), b=2, return_details=True)
    assert mi["pairs"] == [[0, 1], [2, 3]] and mi["candidates"] == 3
    assert len(_matchings(8, 4)) == 105
    assert len(agglomerative_tree(p)) == 3
    return dict(passed=True, deterministic_unit_fit_calls=7, iterative_unit_fit_calls=4,
                closed_unit_fit_calls=3, gradient_error=gradient_error,
                Hessian_vector_error=hessian_error, DP_enumeration_error=dp_error,
                perfect_matchings_d8=105, fits=fits,
                new_scientific_observations=0, scientific_benchmark_fits=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(synthetic_checks(), ensure_ascii=False, indent=2))
