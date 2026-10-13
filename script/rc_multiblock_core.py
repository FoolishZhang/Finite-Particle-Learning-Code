"""Exact six-bit-template learning and post-fit multiblock evaluation.

No import-time RNG or scientific execution. Local state_id is little endian:
state_id = sum(bits[position] * 2**position). Only sample_teacher consumes RNG.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from functools import lru_cache

import numpy as np

EPS_MIN = 0.7
AMPLITUDE_MAX = 0.95
MASS_TOL = 1e-12
QA_TOL = 2e-12
METHODS = ("MOM", "MLE", "HIST", "PRODUCT")


@lru_cache(maxsize=20)
def states(k: int) -> np.ndarray:
    k = int(k)
    if not 1 <= k <= 16:
        raise ValueError("states supports 1..16 coordinates")
    result = ((np.arange(1 << k, dtype=np.uint64)[:, None]
               >> np.arange(k, dtype=np.uint64)) & 1).astype(np.uint8)
    result.setflags(write=False)
    return result


def _binary_data(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("data must be nonempty [n,D]")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("data must contain only zero and one")
    return x.astype(np.uint8, copy=False)


def canonical_block(block) -> tuple[int, ...]:
    values = tuple(int(v) for v in block)
    if len(values) != 6 or len(set(values)) != 6:
        raise ValueError("each template block has six different coordinates")
    pairs = sorted(tuple(sorted(values[i:i+2])) for i in (0, 2, 4))
    return tuple(v for pair in pairs for v in pair)


def canonical_blocks(blocks, D: int) -> list[tuple[int, ...]]:
    """Canonicalize within blocks, preserving block order and epsilon alignment.

    Input adjacent positions define pairs. Epsilon is defined relative to the
    resulting canonical orientation; this function does not transform its sign.
    """
    D = int(D)
    result = [canonical_block(block) for block in blocks]
    flattened = [v for block in result for v in block]
    if D < 6 or D % 6 or sorted(flattened) != list(range(D)):
        raise ValueError("blocks must partition all D coordinates into sixes")
    return result


def _teacher(blocks, eps, D):
    blocks = canonical_blocks(blocks, D)
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape != (len(blocks),) or not np.isfinite(eps).all():
        raise ValueError("one finite amplitude per teacher block required")
    if np.any(np.abs(eps) >= 1):
        raise ValueError("teacher amplitudes must have magnitude below one")
    return blocks, eps


def template_h(x: np.ndarray) -> np.ndarray:
    x = _binary_data(x)
    if x.shape[1] != 6:
        raise ValueError("h needs six locally ordered coordinates")
    y = x.astype(np.int8)
    return ((y[:, 0]-y[:, 1]) * (y[:, 2]-y[:, 3])
            * (y[:, 4]-y[:, 5]))


def template_pmf(epsilon: float) -> np.ndarray:
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or abs(epsilon) >= 1:
        raise ValueError("template must have finite full-support amplitude")
    return (1.0+epsilon*template_h(states(6)).astype(float))/64.0


def sample_teacher(rng, n: int, blocks, eps, D: int) -> np.ndarray:
    """Draw n iid complete vectors, using one rng.choice call per teacher block."""
    n = int(n)
    D = int(D)
    if n < 1:
        raise ValueError("n must be positive")
    blocks, eps = _teacher(blocks, eps, D)
    x = np.empty((n, D), dtype=np.uint8)
    for block, amplitude in zip(blocks, eps):
        ids = rng.choice(64, size=n, p=template_pmf(float(amplitude)))
        x[:, list(block)] = states(6)[ids]
    return x


def _bit_columns(x):
    packed = np.packbits(x, axis=0, bitorder="little")
    return [int.from_bytes(packed[:, j].tobytes(), "little")
            for j in range(x.shape[1])]


def _triple_numerators(x):
    """Yield exact n*empirical E[YiYjYk], with Yi=2Ci-1."""
    n, D = x.shape
    columns = _bit_columns(x)
    for i in range(D-2):
        left = columns[i]
        for j in range(i+1, D-1):
            pair_xor = left ^ columns[j]
            for k in range(j+1, D):
                yield (i, j, k), 2*(pair_xor ^ columns[k]).bit_count()-n


def discover(X) -> dict:
    """Train-data-only exact-threshold three-moment structure discovery."""
    x = _binary_data(X)
    n, D = x.shape
    if D < 6 or D % 6:
        raise ValueError("D must be a positive multiple of six")
    triples, numerators = [], []
    adjacency = [set() for _ in range(D)]
    for triple, numerator in _triple_numerators(x):
        # Strict |moment| > 0.7/16 = 7/160, without floating decisions.
        if abs(numerator)*160 > 7*n:
            triples.append(triple)
            numerators.append(numerator)
            for i, j in itertools.combinations(triple, 2):
                adjacency[i].add(j)
                adjacency[j].add(i)
    components = []
    unseen = set(range(D))
    while unseen:
        first = min(unseen)
        stack = [first]
        unseen.remove(first)
        component = []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in sorted(adjacency[i], reverse=True):
                if j in unseen:
                    unseen.remove(j)
                    stack.append(j)
        components.append(tuple(sorted(component)))
    components.sort()
    errors, recovered = [], []
    if len(components) != D//6:
        errors.append("component_count")
    if any(len(comp) != 6 for comp in components):
        errors.append("component_size")
    if not errors:
        expected = set()
        for comp in components:
            missing = [tuple(pair) for pair in itertools.combinations(comp, 2)
                       if pair[1] not in adjacency[pair[0]]]
            degree = {i: 0 for i in comp}
            for i, j in missing:
                degree[i] += 1
                degree[j] += 1
            if len(missing) != 3 or any(value != 1 for value in degree.values()):
                errors.append("missing_edges_not_matching")
                continue
            block = canonical_block([v for pair in sorted(missing) for v in pair])
            recovered.append(block)
            pairs = [block[0:2], block[2:4], block[4:6]]
            expected.update(tuple(sorted(t)) for t in itertools.product(*pairs))
        if set(triples) != expected:
            errors.append("not_exact_transversal_hypergraph")
    valid = not errors
    return {
        "D": D, "n": n, "threshold": "strict abs(moment)>7/160",
        "integer_rule": "abs(2*popcount(xor)-n)*160 > 7*n",
        "selected_triples": [list(t) for t in triples],
        "selected_numerators": numerators,
        "selected_count": len(triples),
        "components": [list(c) for c in components],
        "valid": valid,
        "recovered_blocks": [list(b) for b in sorted(recovered)] if valid else [],
        "validation_errors": sorted(set(errors)),
        "teacher_access": False,
    }


def _factor(coords, pmf, **metadata):
    q = np.asarray(pmf, dtype=np.float64)
    coords = [int(v) for v in coords]
    if q.shape != (1 << len(coords),) or len(set(coords)) != len(coords):
        raise ValueError("factor dimensions do not match")
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("factors must have finite full support")
    if abs(float(q.sum())-1) > MASS_TOL:
        raise ValueError("factor mass gate failed")
    return {"coords": coords, "pmf": q.tolist(), **metadata}


def uniform_model(D, method="UNIFORM", fallback=True):
    return {
        "method": str(method), "D": int(D), "fallback": bool(fallback),
        "discovery_valid": False,
        "factors": [_factor([j], [0.5, 0.5]) for j in range(int(D))],
    }


def _local_ids(x):
    return x.astype(np.int64) @ (1 << np.arange(x.shape[1], dtype=np.int64))


def fit_models(X1, X2, discovery: dict) -> dict:
    """Closed-form fitting only; receives no teacher object or evaluation labels."""
    x1, x2 = _binary_data(X1), _binary_data(X2)
    n1, D = x1.shape
    n2, D2 = x2.shape
    if D != D2 or int(discovery["D"]) != D or int(discovery["n"]) != n1:
        raise ValueError("training/discovery dimensions do not agree")
    valid = bool(discovery["valid"])
    models = {}
    if valid:
        blocks = canonical_blocks(discovery["recovered_blocks"], D)
        for method in ("MOM", "MLE", "HIST"):
            models[method] = {
                "method": method, "D": D, "fallback": False,
                "discovery_valid": True, "n1": n1, "n2": n2,
                "fitting_sample": "X2_only", "factors": [],
            }
        for block in blocks:
            local = x2[:, list(block)]
            h = template_h(local)
            nplus = int(np.count_nonzero(h == 1))
            nminus = int(np.count_nonzero(h == -1))
            nactive = nplus+nminus
            raw_mom = 8.0*(nplus-nminus)/n2
            raw_mle = (nplus-nminus)/nactive if nactive else 0.0
            for method, raw in (("MOM", raw_mom), ("MLE", raw_mle)):
                amplitude = float(np.clip(raw, -AMPLITUDE_MAX, AMPLITUDE_MAX))
                models[method]["factors"].append(_factor(
                    block, template_pmf(amplitude), epsilon=amplitude,
                    raw_epsilon=raw, nplus=nplus, nminus=nminus,
                    nactive=nactive, mle_zero_active_rule="epsilon=0"))
            counts = np.bincount(_local_ids(local), minlength=64)
            histogram = (counts.astype(float)+1.0/64.0)/(n2+1.0)
            models["HIST"]["factors"].append(_factor(
                block, histogram, counts=counts.tolist(),
                smoothing="Dirichlet total mass 1: per state 1/64"))
    else:
        for method in ("MOM", "MLE", "HIST"):
            models[method] = uniform_model(D, method=method, fallback=True)
            models[method].update(n1=n1, n2=n2,
                                  fitting_sample="invalid_discovery_global_uniform")
    successes = x1.sum(axis=0, dtype=np.int64)+x2.sum(axis=0, dtype=np.int64)
    probabilities = (successes.astype(float)+0.5)/(n1+n2+1.0)
    models["PRODUCT"] = {
        "method": "PRODUCT", "D": D, "fallback": False,
        "discovery_valid": valid, "n1": n1, "n2": n2,
        "fitting_sample": "X1_and_X2_pooled",
        "smoothing": "independent Beta(1/2,1/2)",
        "factors": [_factor([j], [1-p, p], successes=int(successes[j]))
                    for j, p in enumerate(probabilities)],
    }
    return models


def kappa(epsilon):
    e = float(epsilon)
    if not math.isfinite(e) or abs(e) >= 1:
        raise ValueError("kappa amplitude must have magnitude below one")
    return 0.5*((1+e)*math.log1p(e)+(1-e)*math.log1p(-e))


def teacher_marginal(coords, teacherblocks, eps, D):
    """Exact factorization formula for an arbitrary subset of at most six bits."""
    coords = tuple(int(v) for v in coords)
    if not 1 <= len(coords) <= 6 or len(set(coords)) != len(coords):
        raise ValueError("requested marginal needs 1..6 distinct coordinates")
    if min(coords) < 0 or max(coords) >= int(D):
        raise ValueError("marginal coordinates out of range")
    blocks, amplitudes = _teacher(teacherblocks, eps, D)
    x = states(len(coords))
    position = {v: i for i, v in enumerate(coords)}
    result = np.full(len(x), 2.0**(-len(coords)))
    for block, amplitude in zip(blocks, amplitudes):
        conditional_h = np.ones(len(x))
        for i, j in zip(block[0::2], block[1::2]):
            mi = x[:, position[i]] if i in position else 0.5
            mj = x[:, position[j]] if j in position else 0.5
            # Explicit float conversion prevents uint8 subtraction wraparound.
            conditional_h *= np.asarray(mi, dtype=float)-np.asarray(mj, dtype=float)
        result *= 1+float(amplitude)*conditional_h
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("teacher marginal positivity failed")
    if abs(float(result.sum())-1) > MASS_TOL:
        raise ValueError("teacher marginal normalization failed")
    return result


def _validated_factors(model, D):
    if int(model["D"]) != int(D):
        raise ValueError("model dimension mismatch")
    factors, flattened = [], []
    for factor in model["factors"]:
        coords = tuple(int(v) for v in factor["coords"])
        if not 1 <= len(coords) <= 6 or len(set(coords)) != len(coords):
            raise ValueError("model factors must have 1..6 distinct coordinates")
        q = np.asarray(factor["pmf"], dtype=float)
        _factor(coords, q)
        factors.append((coords, q))
        flattened.extend(coords)
    if sorted(flattened) != list(range(int(D))):
        raise ValueError("model factors must be a disjoint partition of all D")
    return factors


def eval_risk(model, teacherblocks, eps, D):
    """Post-fit exact teacher expectation, including every wrong valid partition.

    No RNG, test draws, teacher-dependent fitting, or hidden true total count.
    """
    D = int(D)
    blocks, amplitudes = _teacher(teacherblocks, eps, D)
    factors = _validated_factors(model, D)
    log2 = math.log(2.0)
    teacher_entropy = sum(6*log2-kappa(e)/8 for e in amplitudes)
    true_conditional_entropy = np.empty(D)
    for block, e in zip(blocks, amplitudes):
        entropy = log2-(kappa(e)-2*kappa(e/2))/8
        true_conditional_entropy[list(block)] = entropy
    query_cross_entropy = np.empty(D)
    factor_rows = []
    joint_cross_entropy = 0.0
    marginal_mass_error = 0.0
    model_mass_error = 0.0
    minimum_model_probability = 1.0
    for coords, q in factors:
        p = teacher_marginal(coords, blocks, amplitudes, D)
        logq = np.log(q)
        ce = -float(p @ logq)
        joint_cross_entropy += ce
        model_mass_error = max(model_mass_error, abs(float(q.sum())-1))
        marginal_mass_error = max(marginal_mass_error, abs(float(p.sum())-1))
        minimum_model_probability = min(minimum_model_probability, float(q.min()))
        ids = np.arange(len(q))
        for position, target in enumerate(coords):
            denominator = q+q[ids ^ (1 << position)]
            logconditional = logq-np.log(denominator)
            query_cross_entropy[target] = -float(p @ logconditional)
        factor_rows.append({"coords": list(coords), "cross_entropy": ce,
                            "teacher_marginal_mass": float(p.sum())})
    per_target = query_cross_entropy-true_conditional_entropy
    return {
        "D": D, "method": str(model["method"]),
        "joint_kl": joint_cross_entropy-teacher_entropy,
        "joint_cross_entropy": joint_cross_entropy,
        "teacher_entropy": teacher_entropy,
        "query_kl": float(per_target.mean()),
        "query_cross_entropy": float(query_cross_entropy.mean()),
        "teacher_query_entropy": float(true_conditional_entropy.mean()),
        "per_target_query_kl": per_target.tolist(),
        "per_target_query_cross_entropy": query_cross_entropy.tolist(),
        "per_target_teacher_query_entropy": true_conditional_entropy.tolist(),
        "factor_cross_entropies": factor_rows,
        "normalization_audit": {
            "model_factor_max_mass_error": model_mass_error,
            "teacher_marginal_max_mass_error": marginal_mass_error,
            "minimum_model_factor_probability": minimum_model_probability,
            "finite": bool(np.isfinite(per_target).all()
                           and math.isfinite(joint_cross_entropy)),
        },
        "evaluation": "exact_teacher_marginals_at_most_six_bits_no_test_draws",
        "negative_roundoff_clipped": False,
    }


def _explicit_model(blocks, eps, D, method):
    blocks, eps = _teacher(blocks, eps, D)
    return {"method": method, "D": D, "fallback": False,
            "factors": [_factor(b, template_pmf(e))
                        for b, e in zip(blocks, eps)]}


def _dense_pmf(model, D):
    x = states(D)
    q = np.ones(len(x))
    for coords, factor in _validated_factors(model, D):
        q *= factor[_local_ids(x[:, list(coords)])]
    return q


def deterministic_check():
    """Small deterministic unit fixtures; no RNG or population optimization."""
    checks = {}
    toy = (((np.arange(37)[:, None]+3)
            * (np.arange(7)[None, :]+5)
            + np.arange(37)[:, None]**2) >> 2) & 1
    y = 2*toy.astype(np.int64)-1
    parity_error = 0
    for triple, numerator in _triple_numerators(toy.astype(np.uint8)):
        naive = int(np.prod(y[:, triple], axis=1).sum())
        parity_error = max(parity_error, abs(numerator-naive))
    checks["bitset_vs_naive_max_integer_error"] = parity_error

    # Exact deterministic rational fixture, not a teacher random sample.
    six = states(6)
    fixture_weights = 5+4*template_h(six).astype(int)
    fixture = np.repeat(six, fixture_weights, axis=0)
    discovery = discover(fixture)
    expected = {tuple(sorted(t)) for t in
                itertools.product((0, 1), (2, 3), (4, 5))}
    assert discovery["valid"]
    assert {tuple(t) for t in discovery["selected_triples"]} == expected
    assert discovery["recovered_blocks"] == [[0, 1, 2, 3, 4, 5]]
    assert not discover(six)["valid"]
    checks["exact_hypergraph_fixture"] = True

    # Closed-form fitting unit fixture; no optimizer or scientific fit.
    fitted = fit_models(fixture, fixture, discovery)
    assert tuple(fitted) == METHODS
    assert abs(fitted["MOM"]["factors"][0]["epsilon"]-0.8) < QA_TOL
    assert abs(fitted["MLE"]["factors"][0]["epsilon"]-0.8) < QA_TOL
    assert not any(model["fallback"] for model in fitted.values())
    fallback = fit_models(six, six, discover(six))
    assert all(fallback[key]["fallback"] for key in ("MOM", "MLE", "HIST"))
    checks["closed_form_and_fallback_fixtures"] = True

    D = 12
    teacherblocks = [(0, 1, 4, 7, 9, 11), (2, 3, 5, 6, 8, 10)]
    eps = [0.35, -0.55]
    teacher = _explicit_model(teacherblocks, eps, D, "BAYES_UNIT")
    dense_p = _dense_pmf(teacher, D)
    misblocks = [(0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)]
    models = [
        teacher,
        _explicit_model(misblocks, [0.2, -0.4], D, "MISPARTITION_UNIT"),
        uniform_model(D, method="UNIFORM_UNIT"),
        {"method": "HIST_FIELD_UNIT", "D": D, "factors": [
            _factor(block, (np.arange(64)+1.0)/2080.0) for block in misblocks]},
        {"method": "PRODUCT_FIELD_UNIT", "D": D, "factors": [
            _factor([j], [1-p, p]) for j, p in enumerate(np.linspace(.25, .7, D))]},
    ]
    dense_h = -float(dense_p @ np.log(dense_p))
    all_ids = np.arange(1 << D)
    dense_true_cond_entropy = np.array([
        -float(dense_p @ np.log(dense_p/(dense_p+dense_p[all_ids ^ (1 << j)])))
        for j in range(D)])
    max_joint = max_query = max_entropy = max_per_target = 0.0
    for model in models:
        result = eval_risk(model, teacherblocks, eps, D)
        dense_q = _dense_pmf(model, D)
        direct_joint = float(dense_p @ (np.log(dense_p)-np.log(dense_q)))
        direct_ce = np.array([
            -float(dense_p @ np.log(dense_q/(dense_q+dense_q[all_ids ^ (1 << j)])))
            for j in range(D)])
        direct_per_target = direct_ce-dense_true_cond_entropy
        max_joint = max(max_joint, abs(result["joint_kl"]-direct_joint))
        max_query = max(max_query, abs(result["query_kl"]-float(direct_per_target.mean())))
        max_entropy = max(max_entropy, abs(result["teacher_entropy"]-dense_h),
                          float(np.max(np.abs(
                              np.array(result["per_target_teacher_query_entropy"])
                              -dense_true_cond_entropy))))
        max_per_target = max(max_per_target, float(np.max(np.abs(
            np.array(result["per_target_query_kl"])-direct_per_target))))
    checks.update(
        dense_D12_models=len(models),
        dense_joint_max_error=max_joint, dense_query_max_error=max_query,
        dense_entropy_max_error=max_entropy,
        dense_per_target_max_error=max_per_target,
    )
    x = states(D)
    marginal_error = 0.0
    for coords in ((0,), (1, 4, 9), (0, 2, 4, 6, 8, 10), (9, 11, 4, 7, 0, 1)):
        marginal = teacher_marginal(coords, teacherblocks, eps, D)
        direct = np.bincount(_local_ids(x[:, list(coords)]), weights=dense_p,
                             minlength=1 << len(coords))
        marginal_error = max(marginal_error, float(np.max(np.abs(marginal-direct))))
    checks["dense_marginal_max_error"] = marginal_error
    numerical_errors = [max_joint, max_query, max_entropy, max_per_target, marginal_error]
    assert parity_error == 0
    assert max(numerical_errors) <= QA_TOL, checks
    checks.update(passed=True, tolerance=QA_TOL, rng_calls=0,
                  scientific_samples=0, scientific_fits=0,
                  unit_closed_form_fit_calls=2, iterative_optimizations=0)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true",
                        help="deterministic small unit fixtures only")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("no scientific runner is provided; use --self-check")
    print(json.dumps(deterministic_check(), indent=2, sort_keys=True))
