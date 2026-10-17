"""Fixed learned-proposal target kernels; callers own science contracts and RNG.

Local weights are log(phi*ell/q), never an exact-posterior constant. The only
variance control is the uniform log Gaussian-density ratio with analytic
moments of the actual Gaussian-mixture proposal. No work runs on import.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import log_expit, logsumexp, xlogy
from competition_core import h_probabilities
from active_reference_likelihood import normal_rule, ActiveReferenceLikelihood
from group_posterior_objective import PreparedGroupBanks
from signal_proposal_core import correlation_matrix

PAIRS = ((0, 1), (0, 2), (1, 2))
SHARED_OTHER_PAIR = {(0, 1): (1, 2), (0, 2): (0, 2), (1, 2): (0, 1)}
M, S, K = 8, 18, 8
ALPHA = .3201223434446316
V = np.array([1., 1., 0.]) / math.sqrt(2)


def require(test, message):
    if not test:
        raise ValueError(message)


def add(budget, name, amount):
    if budget is not None:
        budget.add(name, int(amount))


def check(budget):
    if budget is not None:
        budget.check()


def channel_loglike(x, state, *, inner_order=256, budget=None):
    """Known homogeneous local channel, count multiplicities retained."""
    x, state = np.broadcast_arrays(np.asarray(x, float), np.asarray(state))
    require(state.dtype.kind in "iu" and np.all((state >= 0) & (state < S)), "integer state0..17 required")
    require(np.isfinite(x).all(), "finite local arguments required")
    add(budget, "local_likelihood_x", x.size)
    add(budget, "local_private_node_terms", x.size * inner_order)
    h = h_probabilities((1.4 / math.sqrt(8)) * x, tau=3., inner_order=inner_order)
    counts, labels = state % 9, np.where(state < 9, -1, 1)
    choose = np.array([math.log(math.comb(M, j)) for j in range(9)])
    value = log_expit(labels*x) + choose[counts] + xlogy(counts, h["h"]) + xlogy(M-counts, h["complement"])
    require(np.isfinite(value).all(), "nonfinite local likelihood; no clipping")
    check(budget)
    return value


def channel_level(outer_order, *, inner_order=256, budget=None):
    """One shared homogeneous channel. Raw finite-rule mass, no normalization."""
    x, w = normal_rule(outer_order)
    ell = channel_loglike(x[:, None], np.arange(S)[None, :], inner_order=inner_order, budget=budget)
    factors = np.exp(ell)
    z = w @ factors
    mean = (w*x) @ factors / z
    m2 = (w*x*x) @ factors / z
    require(np.all(z > 0) and np.isfinite(mean).all() and np.all(m2 >= mean*mean), "invalid channel level")
    return {"outer_order": outer_order, "inner_order": inner_order, "normalizer": z,
            "mean": mean, "second_moment": m2, "mass": float(z.sum()),
            "reflection_mass_error": float(np.max(abs(z-z[::-1]))),
            "reflection_mean_error": float(np.max(abs(mean+mean[::-1]))),
            "J_pair": float((z @ (mean*mean))**2)}


def population_origin(level):
    states = np.indices((S, S, S)).reshape(3, -1).T
    z, mu = np.asarray(level["normalizer"]), np.asarray(level["mean"])
    weights = np.prod(z[states], axis=1)
    score = np.stack([mu[states[:, i]]*mu[states[:, j]] for i, j in PAIRS], axis=1)
    return {"state_indices": states, "weights": weights, "teacher_score": score}


def density_metrics(proposal, reference_normalizer, *, orders=(256,384), budget=None):
    """All-state ordinary GH diagnostics against one fixed numerical q0.

    This does not choose a trained proposal or certify a true integral. The
    Gaussian-mixture proposal itself has an analytic normalizer of one.
    """
    z=np.asarray(reference_normalizer,float)
    require(z.shape==(S,) and np.all(z>0), "fixed shared-channel normalizer required")
    levels=[]
    for order in orders:
        add(budget,"density_diagnostic_rules",1)
        x,w=normal_rule(order)
        ll=channel_loglike(x[:,None],np.arange(S)[None,:],inner_order=256,budget=budget)
        logq=proposal.logpdf(x)
        add(budget,"proposal_density_values",order*S)
        logphi=-.5*x*x-.5*math.log(2*math.pi)
        logq_ratio=logq-logphi[:,None]
        logq0_ratio=ll-np.log(z)
        q0_mass=w@np.exp(logq0_ratio)
        kl=np.sum(w[:,None]*np.exp(logq0_ratio)*(logq0_ratio-logq_ratio),axis=0)
        with np.errstate(divide="ignore"):
            logw=np.log(w)[:,None]
        chi=np.exp(logsumexp(logw+2*logq0_ratio-logq_ratio,axis=0))
        qmass=np.exp(logsumexp(logw+logq_ratio,axis=0))
        for arr in (q0_mass,kl,chi,qmass): require(np.isfinite(arr).all(),"nonfinite density metric")
        levels.append({"outer_order":order,"private_inner_order":256,
                       "KL_q0_to_q":kl,"second_ratio_moment":chi,
                       "q_mass":qmass,"q0_mass":q0_mass,
                       "natural_frequency_weighted_KL":float(z@kl),
                       "maximum_state_KL":float(kl.max()),
                       "natural_frequency_weighted_second_ratio_moment":float(z@chi),
                       "maximum_state_second_ratio_moment":float(chi.max())})
        check(budget)
    low,high=levels
    diffs={key:float(np.max(abs(high[key]-low[key]))) for key in ("KL_q0_to_q","second_ratio_moment")}
    errors={key:float(np.max(abs(high[key]-1))) for key in ("q_mass","q0_mass")}
    qualified=diffs["KL_q0_to_q"]<=1e-4 and diffs["second_ratio_moment"]<=1e-3 and errors["q_mass"]<=1e-6 and errors["q0_mass"]<=1e-10
    return {"levels":levels,"outer_order_max_differences":diffs,"high_rule_mass_errors":errors,
            "ordinary_diagnostic_qualified":bool(qualified),"never_output_inclusion_rule":True,
            "q0_normalizer_held_fixed_across_orders":True,
            "qualification":"ordinary_GH_order_diagnostic_not_strict_error_certificate"}


def local_weights(proposal, x, *, inner_order=256, budget=None):
    require(x.ndim == 5 and x.shape[3:] == (3, S), "x[R,B,K,3,18] required")
    state = np.broadcast_to(np.arange(S), x.shape)
    logq = proposal.logpdf(x, state)
    add(budget, "proposal_density_values", x.size)
    ell = channel_loglike(x, state, inner_order=inner_order, budget=budget)
    local = -.5*x*x - .5*math.log(2*math.pi) + ell - logq
    require(np.isfinite(local).all(), "full local logweight must remain finite")
    return local


def stats(banks):
    """Average B within each complete randomization; SE across R."""
    banks = np.asarray(banks, float)
    require(banks.ndim >= 2 and banks.shape[0] >= 2 and np.isfinite(banks).all(), "finite banks with R>=2")
    reps = banks.mean(axis=1)
    return {"replicates": reps, "mean": reps.mean(axis=0),
            "SE": reps.std(axis=0, ddof=1)/math.sqrt(len(reps)), "R": len(reps), "B": banks.shape[1]}


def origin_coefficients(x, local, states, weights, teacher_score, proposal_mean, proposal_second,
                        *, pattern_chunk=8, budget=None):
    """Raw and controlled g/H/B, with full normalized particle weights.

    H uses -E_pi(T)-Cov_pi(s); B row=candidate, column=teacher.
    Uniform-logratio adds sbar-Eq(s) and Tbar-Eq(T). B is minus
    the controlled conditional gradient times the teacher observation score.
    """
    require(x.shape == local.shape and x.ndim == 5 and x.shape[3] == 3, "matching full local weights required")
    R, B, k, _, nstates = x.shape
    states, weights, teacher_score = map(np.asarray, (states, weights, teacher_score))
    require(states.shape == (len(weights), 3) and teacher_score.shape == (len(weights), 3), "pattern/score shapes")
    require(abs(weights.sum()-1) < 1e-10 and np.all(weights >= 0), "raw population mass required")
    qm, qs = np.asarray(proposal_mean), np.asarray(proposal_second)
    require(qm.shape == qs.shape == (3, nstates), "actual proposal moment tables required")
    add(budget, "anchor_calls", 1)
    add(budget, "anchor_particle_pattern_terms", R*B*k*len(weights))
    output = {name: np.zeros((R, B, 3) if name.endswith("g") else (R, B, 3, 3))
              for name in ("raw_g", "cv_g", "raw_H", "cv_H", "raw_B", "cv_B")}
    min_ess, max_pi = math.inf, 0.
    for start in range(0, len(weights), pattern_chunk):
        stop = min(start+pattern_chunk, len(weights)); ids=states[start:stop]; p=weights[start:stop]
        xx = [np.moveaxis(x[..., j, ids[:, j]], 2, 3) for j in range(3)]
        ll = sum(np.moveaxis(local[..., j, ids[:, j]], 2, 3) for j in range(3))
        pi = np.exp(ll-logsumexp(ll, axis=-1)[..., None])
        scores = [xx[i]*xx[j] for i, j in PAIRS]
        U = np.stack([np.sum(pi*s, axis=-1) for s in scores], axis=-1)
        mean_s = np.stack([s.mean(axis=-1) for s in scores], axis=-1)
        expect_s = np.stack([qm[i, ids[:, i]]*qm[j, ids[:, j]] for i,j in PAIRS],axis=-1)
        raw_g, cv_g = -U, -U+mean_s-expect_s
        output["raw_g"] += np.einsum("rbpd,p->rbd", raw_g,p)
        output["cv_g"] += np.einsum("rbpd,p->rbd", cv_g,p)
        m=teacher_score[start:stop]
        output["raw_B"] -= np.einsum("rspa,pb,p->rsab",raw_g,m,p)
        output["cv_B"] -= np.einsum("rspa,pb,p->rsab",cv_g,m,p)
        for a, (i,j) in enumerate(PAIRS):
            for b in range(a,3):
                if a==b:
                    T=1-xx[i]**2-xx[j]**2
                    expect_T=1-qs[i,ids[:,i]]-qs[j,ids[:,j]]
                else:
                    u,v=SHARED_OTHER_PAIR[(a,b)]
                    T=-xx[u]*xx[v]
                    expect_T=-qm[u,ids[:,u]]*qm[v,ids[:,v]]
                raw_h=-np.sum(pi*(T+scores[a]*scores[b]),axis=-1)+U[...,a]*U[...,b]
                cv_h=raw_h+T.mean(axis=-1)-expect_T
                for name, arr in (("raw_H",raw_h),("cv_H",cv_h)):
                    value=np.einsum("rbp,p->rb",arr,p)
                    output[name][...,a,b]+=value
                    if b!=a: output[name][...,b,a]+=value
        min_ess=min(min_ess,float(np.min(1/np.sum(pi*pi,axis=-1))))
        max_pi=max(max_pi,float(pi.max()))
        check(budget)
    for arr in output.values(): require(np.isfinite(arr).all(), "nonfinite coefficient")
    return {"banks":output,"statistics":{key:stats(arr) for key,arr in output.items()},
            "min_sample_ESS":min_ess,"max_normalized_particle_weight":max_pi,
            "coefficient_target":"fixed_q_g_H_and_B_negative_teacher_cross_derivative",
            "control":"uniform_logratio_actual_mixture_analytic_moments",
            "variance_units":"R independent complete randomizations, not B/K/patterns"}


def teacher_law(t, *, outer_order, budget=None):
    engine=ActiveReferenceLikelihood(M=8,epsilon=np.full(3,1.4/math.sqrt(8)),tau=3.,
                                    outer_order=outer_order,inner_order=256)
    add(budget,"pmf_law_calls",1)
    law=engine.law(correlation_matrix(float(t)*V))
    add(budget,"pmf_private_node_terms",law["private_node_evaluations_this_call"])
    check(budget)
    return law


def candidate_difference(x,local,states,teacher_weights,proposal_mean,proposal_second,
                         *,t,alpha=ALPHA,pattern_chunk=8,budget=None):
    """Two fixed candidates on the same accepted bank and observation weights."""
    bank=PreparedGroupBanks(x,local,states,teacher_weights,proposal_kind="frozen_learned_local",
                            proposal_mean=proposal_mean,proposal_second_moment=proposal_second)
    values=[]
    for multiplier in (alpha,1.):
        add(budget,"candidate_calls",1)
        add(budget,"candidate_particle_pattern_terms",int(np.prod(x.shape[:3]))*len(states))
        result=bank.evaluate(correlation_matrix(multiplier*float(t)*V),control="uniform_logratio",pattern_chunk=pattern_chunk)
        values.append(result)
        check(budget)
    difference=values[0]["bank_values"]-values[1]["bank_values"]
    raw=values[0]["raw_bank_values"]-values[1]["raw_bank_values"]
    return {"t":float(t),"alpha":float(alpha),"banks":difference,"raw_banks":raw,
            "statistics":stats(difference),"raw_statistics":stats(raw),
            "pairing":"same_q_particles_and_same_teacher_PMF; shrink_candidate_minus_teacher_candidate",
            "candidate_coordinates":[(alpha*float(t)*V).tolist(),(float(t)*V).tolist()]}


def prediction(g,H,B,*,t,alpha=ALPHA):
    return ((alpha-1)*t*np.einsum("...i,i->...",g,V)
            +t*t*(.5*(alpha*alpha-1)*np.einsum("i,...ij,j->...",V,H,V)
                   -(alpha-1)*np.einsum("i,...ij,j->...",V,B,V)))
