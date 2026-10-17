"""Fixed-bank quadratic controls and Taylor remainders for actual product q.

Imports do not sample, integrate, optimize, or write. Callers bind bank, proposal,
teacher-rule and saved-output identities in a frozen execution contract. The
teacher derivative is the derivative of the supplied finite teacher PMF rule;
the old product-posterior/Stein derivative is never substituted silently.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
from scipy.special import logsumexp

from group_posterior_objective import PreparedGroupBanks
from signal_proposal_core import correlation_matrix
from learned_local_objective import PAIRS, SHARED_OTHER_PAIR, ALPHA, V, stats

SCHEMA = "learned_precision_core_v1"
CONTROL = "uniform_logratio_plus_actual_product_q_centered_score_covariance"


def require(test, message):
    if not test:
        raise ValueError(message)


def add(budget, name, amount=1):
    if budget is not None:
        budget.add(name, int(amount))


def check(budget):
    if budget is not None:
        budget.check()


def array_sha(value):
    arr = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(str(arr.shape).encode("ascii") + arr.tobytes()).hexdigest()


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def teacher_reference_from_law(law, *, identity, origin_coordinates=(0., 0., 0.)):
    """Bind p0 and dp0 from the SAME finite-GH Cholesky teacher family.

    ``law`` is ActiveReferenceLikelihood.law(I). Manufactured tests may declare
    gradient_method='manufactured_analytic_derivative_of_same_teacher_family'.
    Coordinates are explicit caller provenance, not inferred from PMF values.
    No PMF repair, normalization, or channel/model evaluation occurs here.
    """
    require(isinstance(identity, str) and identity, "explicit teacher reference identity required")
    require(np.array_equal(np.asarray(origin_coordinates, float), np.zeros(3)), "teacher reference must be at origin")
    method = law.get("gradient_method")
    require(method in ("same_finite_GH_Cholesky_chain_rule",
                       "manufactured_analytic_derivative_of_same_teacher_family"),
            "same-teacher-rule derivative definition required; no product-moment substitution")
    p, dp = np.asarray(law["joint_pmf"], float), np.asarray(law["pmf_gradient"], float)
    require(p.ndim == 3 and len(set(p.shape)) == 1 and p.shape[0] > 0,
            "complete cubic observation PMF required")
    require(dp.shape == (3,) + p.shape and np.isfinite(p).all() and np.isfinite(dp).all()
            and np.all(p > 0), "positive finite p0 and finite candidate-ordered dp0 required")
    require(abs(float(p.sum()) - 1) <= 1e-10 and np.max(abs(dp.sum(axis=(1, 2, 3)))) <= 1e-10,
            "raw teacher mass/derivative mass fails; no repair")
    if "pair_order" in law:
        require(tuple(map(tuple, law["pair_order"])) == PAIRS, "teacher derivative pair order differs")
    if method == "same_finite_GH_Cholesky_chain_rule":
        require(all(isinstance(law.get(k), (int, np.integer)) and law[k] >= 2
                    for k in ("outer_order", "inner_order")), "explicit finite teacher rules required")
    states = np.indices(p.shape).reshape(3, -1).T
    weights, derivative = p.ravel(), dp.reshape(3, -1).T
    return {"identity": identity, "state_indices": _readonly(states, np.int64),
            "weights": _readonly(weights), "pmf_gradient": _readonly(derivative),
            "teacher_score": _readonly(derivative / weights[:, None]),
            "p0_sha256": array_sha(weights), "dp0_sha256": array_sha(derivative),
            "gradient_method": method, "origin_coordinates": [0., 0., 0.],
            "outer_order": law.get("outer_order"), "inner_order": law.get("inner_order"),
            "qualification": "derivatives_of_supplied_teacher_family; ordinary_GH_not_certified_error_bound"}


def _moments(states, mean, second, nstates):
    mean, second = np.asarray(mean, float), np.asarray(second, float)
    require(mean.shape == second.shape == (3, nstates) and np.isfinite(mean).all()
            and np.isfinite(second).all() and np.all(second >= mean*mean - 1e-13),
            "consistent finite moments of actual product proposal required")
    mu = np.stack([mean[g, states[:, g]] for g in range(3)], axis=-1)
    nu = np.stack([second[g, states[:, g]] for g in range(3)], axis=-1)
    return mu, nu


def product_score_covariance(states, proposal_mean, proposal_second):
    """Cov_q((X1X2,X1X3,X2X3)) using only actual-q first/second moments."""
    states = np.asarray(states)
    require(states.ndim == 2 and states.shape[1] == 3 and states.dtype.kind in "iu",
            "integer pattern indices required")
    nstates = np.asarray(proposal_mean).shape[-1]
    require(np.all((states >= 0) & (states < nstates)), "state outside moment tables")
    mu, nu = _moments(states, proposal_mean, proposal_second, nstates)
    cov = np.empty((len(states), 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        cov[:, a, a] = nu[:, i]*nu[:, j] - mu[:, i]**2*mu[:, j]**2
        for b in range(a+1, 3):
            u, v = SHARED_OTHER_PAIR[a, b]
            shared = 3-u-v
            cov[:, a, b] = cov[:, b, a] = (nu[:, shared]-mu[:, shared]**2)*mu[:, u]*mu[:, v]
    require(np.isfinite(cov).all(), "nonfinite actual-q score covariance")
    return cov


def origin_coefficients(x, local, reference, proposal_mean, proposal_second, *,
                        teacher_weights=None, bank_id, pattern_chunk=8, budget=None):
    """One pass yields original raw/CV coefficients, new CV2, C0 and every Ct.

    B is evaluated directly as -sum_O g_bank(O) dp0(O)^T. The control uses
    C(O)=mean(ss')-sbar*sbar'-(1-1/K)Cov_q(s|O). Its origin additions are
    dg=dB=0, dH=C0. At a finite teacher the scalar control requires Ct, NOT C0.
    All matrices and prediction inputs retain [R,B,...], without projection.
    """
    x, local = np.asarray(x, float), np.asarray(local, float)
    require(x.shape == local.shape and x.ndim == 5 and x.shape[3] == 3
            and min(x.shape) > 0 and x.shape[0] >= 2 and np.isfinite(x).all()
            and np.isfinite(local).all(), "finite x/local[R>=2,B,K,3,S] required")
    require(isinstance(bank_id, str) and bank_id, "explicit fixed bank identity required")
    require(isinstance(pattern_chunk, (int, np.integer)) and not isinstance(pattern_chunk, bool)
            and pattern_chunk >= 1, "positive integer pattern chunk required")
    R, B, K, _, S = x.shape
    states = np.asarray(reference["state_indices"])
    p0, dp0 = np.asarray(reference["weights"], float), np.asarray(reference["pmf_gradient"], float)
    expected = np.indices((S, S, S)).reshape(3, -1).T
    require(np.array_equal(states, expected) and p0.shape == (len(states),)
            and dp0.shape == (len(states), 3), "complete matching teacher state enumeration required")
    require(np.isfinite(p0).all() and np.isfinite(dp0).all() and np.all(p0 > 0)
            and abs(float(p0.sum())-1) <= 1e-10 and np.max(abs(dp0.sum(axis=0))) <= 1e-10,
            "positive raw p0 and finite mass-preserving dp0 required")
    require(array_sha(p0) == reference["p0_sha256"] and array_sha(dp0) == reference["dp0_sha256"],
            "teacher reference arrays changed after definition")
    require(reference["gradient_method"] in ("same_finite_GH_Cholesky_chain_rule",
            "manufactured_analytic_derivative_of_same_teacher_family"), "unsupported teacher derivative")
    _moments(states[:1], proposal_mean, proposal_second, S)
    qm, qs = np.asarray(proposal_mean, float), np.asarray(proposal_second, float)
    teachers = {} if teacher_weights is None else dict(teacher_weights)
    require(all(isinstance(key, str) and key and key != "origin" for key in teachers),
            "nonempty teacher keys, excluding reserved origin, required")
    names = ["origin", *teachers]
    weight_arrays = [p0]
    for name in teachers:
        weight = np.asarray(teachers[name], float).reshape(-1)
        require(weight.shape == p0.shape and np.isfinite(weight).all() and np.all(weight >= 0)
                and abs(float(weight.sum())-1) <= 1e-10, "invalid raw teacher weights: " + name)
        weight_arrays.append(weight)
    weights_all = np.stack(weight_arrays)
    add(budget, "anchor_calls")
    add(budget, "anchor_particle_pattern_terms", R*B*K*len(states))
    # Six symmetric matrix contractions per pattern, bank and teacher law.
    add(budget, "q2_control_matrix_pattern_terms", 6*R*B*len(states)*len(names))
    output = {name: np.zeros((R, B, 3) if name.endswith("g") else (R, B, 3, 3))
              for name in ("raw_g", "cv_g", "raw_H", "cv_H", "raw_B", "cv_B")}
    controls = np.zeros((len(names), R, B, 3, 3))
    min_ess, max_pi = math.inf, 0.
    for start in range(0, len(states), pattern_chunk):
        stop = min(start+pattern_chunk, len(states))
        ids, p, dp = states[start:stop], p0[start:stop], dp0[start:stop]
        xx = [np.moveaxis(x[..., j, ids[:, j]], 2, 3) for j in range(3)]
        ll = sum(np.moveaxis(local[..., j, ids[:, j]], 2, 3) for j in range(3))
        pi = np.exp(ll-logsumexp(ll, axis=-1)[..., None])
        scores = [xx[i]*xx[j] for i, j in PAIRS]
        U = np.stack([np.sum(pi*s, axis=-1) for s in scores], axis=-1)
        mean_s = np.stack([s.mean(axis=-1) for s in scores], axis=-1)
        mu, nu = _moments(ids, qm, qs, S)
        expect_s = np.stack([mu[:, i]*mu[:, j] for i, j in PAIRS], axis=-1)
        raw_g, cv_g = -U, -U+mean_s-expect_s
        for tag, grad in (("raw", raw_g), ("cv", cv_g)):
            output[tag+"_g"] += np.einsum("rbpi,p->rbi", grad, p)
            output[tag+"_B"] -= np.einsum("rbpi,pj->rbij", grad, dp)
        covariance = product_score_covariance(ids, qm, qs)
        for a, (i, j) in enumerate(PAIRS):
            for b in range(a, 3):
                if a == b:
                    T = 1-xx[i]**2-xx[j]**2
                    expected_T = 1-nu[:, i]-nu[:, j]
                else:
                    u, v = SHARED_OTHER_PAIR[a, b]
                    T = -xx[u]*xx[v]
                    expected_T = -mu[:, u]*mu[:, v]
                raw_h = -np.sum(pi*(T+scores[a]*scores[b]), axis=-1)+U[..., a]*U[..., b]
                cv_h = raw_h+T.mean(axis=-1)-expected_T
                for name, value in (("raw_H", raw_h), ("cv_H", cv_h)):
                    contraction = np.einsum("rbp,p->rb", value, p)
                    output[name][..., a, b] += contraction
                    if a != b:
                        output[name][..., b, a] += contraction
                if K > 1:
                    centered = ((scores[a]*scores[b]).mean(axis=-1)-mean_s[..., a]*mean_s[..., b]
                                -(1-1/K)*covariance[:, a, b])
                    contraction = np.einsum("rbp,np->nrb", centered, weights_all[:, start:stop])
                    controls[..., a, b] += contraction
                    if a != b:
                        controls[..., b, a] += contraction
                # K=1 is analytically zero; avoid a subtractive fake residual.
        min_ess = min(min_ess, float(np.min(1/np.sum(pi*pi, axis=-1))))
        max_pi = max(max_pi, float(pi.max()))
        check(budget)
    output.update(q2_g=output["cv_g"].copy(), q2_H=output["cv_H"]+controls[0], q2_B=output["cv_B"].copy())
    for value in [*output.values(), controls]:
        require(np.isfinite(value).all(), "nonfinite fixed-bank coefficient/control")
        value.setflags(write=False)
    return {"schema": SCHEMA, "bank_id": bank_id, "reference_identity": reference["identity"],
            "reference_p0_sha256": reference["p0_sha256"], "reference_dp0_sha256": reference["dp0_sha256"],
            "reference_derivative": reference["gradient_method"], "outer_order": reference["outer_order"],
            "inner_order": reference["inner_order"], "banks": output,
            "statistics": {key: stats(value) for key, value in output.items()},
            "C_origin": controls[0], "C_teachers": {name: controls[i+1] for i, name in enumerate(teachers)},
            "teacher_weight_sha256": {name: array_sha(value) for name, value in zip(names, weight_arrays)},
            "control_statistics": {name: stats(controls[i]) for i, name in enumerate(names)},
            "control": CONTROL, "min_sample_ESS": min_ess, "max_normalized_particle_weight": max_pi,
            "work": {"anchor_calls": 1, "anchor_particle_pattern_terms": R*B*K*len(states),
                     "q2_control_matrix_pattern_terms": 6*R*B*len(states)*len(names)},
            "work_scope": "one full-softmax particle-pattern pass; six symmetric control contractions per teacher; not CPU operations",
            "no_group_permutation_projection": True}


def quadratic_increment(control, coordinates):
    """Scalar, gradient, Hessian added to a fixed bank: +.5 theta'C theta."""
    c, theta = np.asarray(control, float), np.asarray(coordinates, float)
    require(c.ndim >= 2 and c.shape[-2:] == (3, 3) and np.isfinite(c).all()
            and np.allclose(c, np.swapaxes(c, -1, -2), rtol=0, atol=1e-13), "finite symmetric C required")
    require(theta.shape == (3,) and np.isfinite(theta).all(), "finite three candidate coordinates required")
    return {"value": .5*np.einsum("i,...ij,j->...", theta, c, theta),
            "gradient": np.einsum("...ij,j->...i", c, theta), "hessian": c}


def prediction_components(g, H, B, *, t, alpha=ALPHA, direction=V):
    """Retain per-bank linear/H/B contributions and all their covariance inputs."""
    g, H, B, v = map(lambda value: np.asarray(value, float), (g, H, B, direction))
    require(g.shape[-1:] == (3,) and H.shape == B.shape == g.shape[:-1]+(3, 3)
            and v.shape == (3,) and all(np.isfinite(a).all() for a in (g, H, B, v))
            and np.isfinite(t) and np.isfinite(alpha), "matching finite full prediction arrays required")
    linear = (alpha-1)*np.einsum("...i,i->...", g, v)
    hpart = .5*(alpha*alpha-1)*np.einsum("i,...ij,j->...", v, H, v)
    bpart = -(alpha-1)*np.einsum("i,...ij,j->...", v, B, v)
    return {"linear_coefficient": linear, "H_quadratic_coefficient": hpart,
            "B_quadratic_coefficient": bpart, "quadratic_coefficient": hpart+bpart,
            "prediction": t*linear+t*t*(hpart+bpart)}


def apply_candidate_control(base_difference, coefficients, *, teacher_key, teacher_weights,
                            t, alpha=ALPHA, direction=V, bank_id):
    """Reuse one saved/evaluated base D; no extra candidate kernel is called.

    The driver must bind this saved D to the declared bank, local weights and
    finite teacher PMF. Array shapes alone cannot establish that provenance.
    """
    require(bank_id == coefficients["bank_id"], "candidate/coefficient bank identities differ")
    require(teacher_key in coefficients["C_teachers"], "teacher control was not accumulated")
    require(array_sha(np.asarray(teacher_weights, float).reshape(-1)) == coefficients["teacher_weight_sha256"][teacher_key],
            "candidate teacher weights differ from control weights")
    d, v = np.asarray(base_difference, float), np.asarray(direction, float)
    require(d.shape == coefficients["C_origin"].shape[:2] and np.isfinite(d).all()
            and v.shape == (3,) and np.isfinite(v).all() and np.isfinite(t) and np.isfinite(alpha),
            "finite same-bank D and candidate definition required")
    Ct = coefficients["C_teachers"][teacher_key]
    increment = .5*(alpha*alpha-1)*t*t*np.einsum("i,rbij,j->rb", v, Ct, v)
    banks = {"cv_D": d.copy(), "q2_D": d+increment, "D_control_increment": increment}
    require(all(np.isfinite(value).all() for value in banks.values()), "nonfinite controlled candidate difference")
    return {"bank_id": bank_id, "reference_identity": coefficients["reference_identity"],
            "teacher_key": teacher_key, "teacher_weight_sha256": coefficients["teacher_weight_sha256"][teacher_key],
            "t": float(t), "alpha": float(alpha), "direction": v.copy(), "banks": banks,
            "statistics": {key: stats(value) for key, value in banks.items()}, "additional_candidate_calls": 0}


def candidate_difference(x, local, reference, proposal_mean, proposal_second, coefficients, *,
                         teacher_key, teacher_weights, t, alpha=ALPHA, direction=V,
                         bank_id, pattern_chunk=8, budget=None):
    """Evaluate the two base candidates once, then obtain both control policies."""
    require(reference["identity"] == coefficients["reference_identity"]
            and reference["p0_sha256"] == coefficients["reference_p0_sha256"]
            and reference["dp0_sha256"] == coefficients["reference_dp0_sha256"], "teacher reference differs")
    p = np.asarray(teacher_weights, float).reshape(-1)
    require(bank_id == coefficients["bank_id"], "candidate/coefficient bank identities differ")
    require(teacher_key in coefficients["C_teachers"]
            and array_sha(p) == coefficients["teacher_weight_sha256"][teacher_key],
            "candidate teacher control binding differs before model evaluation")
    bank = PreparedGroupBanks(x, local, reference["state_indices"], p,
                              proposal_mean=proposal_mean, proposal_second_moment=proposal_second)
    values, raw_values = [], []
    for scale in (alpha, 1.):
        add(budget, "candidate_calls")
        add(budget, "candidate_particle_pattern_terms", int(np.prod(x.shape[:3]))*len(p))
        result = bank.evaluate(correlation_matrix(scale*t*np.asarray(direction)), control="uniform_logratio",
                               pattern_chunk=pattern_chunk)
        values.append(result["bank_values"])
        raw_values.append(result["raw_bank_values"])
        check(budget)
    result = apply_candidate_control(values[0]-values[1], coefficients, teacher_key=teacher_key,
                                     teacher_weights=p, t=t, alpha=alpha, direction=direction, bank_id=bank_id)
    result["banks"]["raw_D"] = raw_values[0]-raw_values[1]
    result["statistics"]["raw_D"] = stats(result["banks"]["raw_D"])
    result["candidate_calls"] = 2
    return result


def same_bank_remainder(candidate, coefficients):
    """D-P from exactly the same fixed bank and finite teacher rule.

    This measures the Taylor remainder of that numerical scalar, not success
    of an independently frozen prediction. No previous precision count changes.
    """
    require(candidate["bank_id"] == coefficients["bank_id"]
            and candidate["reference_identity"] == coefficients["reference_identity"],
            "same bank and teacher derivative reference required")
    key = candidate["teacher_key"]
    require(candidate["teacher_weight_sha256"] == coefficients["teacher_weight_sha256"][key],
            "same teacher PMF required")
    t, alpha, v = candidate["t"], candidate["alpha"], candidate["direction"]
    require(t != 0, "nonzero amplitude required for normalized remainder")
    output = {}
    for tag in ("cv", "q2"):
        pieces = prediction_components(*(coefficients["banks"][tag+"_"+name] for name in ("g", "H", "B")),
                                       t=t, alpha=alpha, direction=v)
        output.update({tag+"_"+key: value for key, value in pieces.items()})
        output[tag+"_D"] = candidate["banks"][tag+"_D"]
        output[tag+"_remainder"] = output[tag+"_D"]-pieces["prediction"]
        output[tag+"_normalized_remainder"] = output[tag+"_remainder"]/(t*t)
    expected = .5*(alpha*alpha-1)*t*t*np.einsum("i,rbij,j->rb", v,
                  coefficients["C_teachers"][key]-coefficients["C_origin"], v)
    observed = output["q2_remainder"]-output["cv_remainder"]
    error = float(np.max(abs(expected-observed)))
    require(np.allclose(expected, observed, rtol=2e-11, atol=2e-13), "Ct-C0 remainder control identity failed")
    return {"bank_id": coefficients["bank_id"], "reference_identity": coefficients["reference_identity"],
            "teacher_key": key, "t": t, "alpha": alpha, "direction": np.asarray(v).copy(),
            "banks": output, "statistics": {key: stats(value) for key, value in output.items()},
            "control_remainder_increment_identity_max_error": error,
            "scope": "same_bank_Taylor_remainder_of_declared_finite_teacher_scalar; not_independent_prediction_validation"}
