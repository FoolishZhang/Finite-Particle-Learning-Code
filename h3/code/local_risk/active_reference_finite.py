"""Smooth fixed-bank collapsed finite-K objectives, with exact SAA RP gradients.

Caller supplies z[replicate,B,K,3]. A Sobol bank must be one point of dimension
3*K; consecutive points of a dimension-3 net do not represent iid-K banks.
No RNG, optimizer, teacher generation, source edits, or file writes here.
Scramble replicates, rather than their dependent B rows, define QMC error units.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from scipy.special import expit, log_expit, logsumexp

import competition_core as competition
import signal_proposal_core as legacy

SCHEMA = "active_reference_finite_v1"
MIN_H = 1e-250


def h_first(shifts, *, tau, inner_order, chunk_size=4096):
    """Only h/hc and first derivatives: positive GH, separately computed tails.

    The rule/nodes match the legacy kernel. Underflow is a qualification failure,
    never clipping or an adaptive change of numerical target.
    """
    nodes, weights = competition._normal_rule(inner_order)
    shifts = np.asarray(shifts,dtype=float)
    if not np.isfinite(shifts).all():raise ValueError("finite h shifts required")
    output={name:np.empty(shifts.size) for name in ("log_h","log_complement","dlog_h","dlog_complement")}
    flat=shifts.ravel()
    for start in range(0,len(flat),chunk_size):
        stop=min(start+chunk_size,len(flat));logits=flat[start:stop,None]+tau*nodes
        if not np.isfinite(logits).all():raise FloatingPointError("h logits overflow")
        p,f=expit(logits),expit(-logits)
        hp=np.sum((p*f)*weights,axis=1)
        h=np.sum(p*weights,axis=1);hc=np.sum(f*weights,axis=1)
        if np.any(h<=MIN_H) or np.any(hc<=MIN_H) or not all(np.isfinite(a).all() for a in (h,hc,hp)):
            raise FloatingPointError("nonfinite/underflow positive GH probability; no clipping")
        values=(np.log(h),np.log(hc),hp/h,-hp/hc)
        for name,value in zip(output,values):output[name][start:stop]=value
    return {name:value.reshape(shifts.shape) for name,value in output.items()}


@lru_cache(maxsize=32)
def _prior_expectations(epsilon,tau,inner_order,marginal_order):
    nodes,weights=competition._normal_rule(marginal_order)
    h=h_first(nodes[:,None]*np.asarray(epsilon),tau=tau,inner_order=inner_order)
    return (weights @ h["log_h"],weights @ h["log_complement"],
            float(weights @ log_expit(nodes)),float(weights @ log_expit(-nodes)))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _patterns(counts, labels, weights, M):
    counts, labels = np.asarray(counts), np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    if (counts.ndim != 2 or counts.shape[1] != 3 or len(counts) < 1
            or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > M)):
        raise ValueError("counts must be nonempty integer (P,3) in 0..M")
    if labels.shape != counts.shape or not np.isin(labels, (-1, 1)).all():
        raise ValueError("labels must be observed spins (P,3)")
    if (weights.shape != (len(counts),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("weights must be finite nonnegative (P,) of mass one; no renormalization")
    indices = counts + ((labels + 1)//2)*(M+1)
    return counts, labels, weights, indices.astype(np.int64)


def empirical_patterns(counts, observed_labels, *, M):
    """Compress actual observations only; the interface has no teacher argument."""
    M = _integer(M, "M")
    counts, observed_labels = np.asarray(counts), np.asarray(observed_labels)
    if counts.ndim != 2 or len(counts) < 1:
        raise ValueError("a nonempty observation matrix is required")
    _, _, _, indices = _patterns(counts, observed_labels, np.full(len(counts), 1/len(counts)), M)
    unique, multiplicity = np.unique(indices, axis=0, return_counts=True)
    return {"counts": unique % (M+1), "labels": 2*(unique//(M+1))-1,
            "weights": multiplicity/len(counts), "state_indices": unique,
            "sample_size": len(counts), "data_role": "empirical_actual_counts_Y"}


def population_patterns(joint_pmf, *, M):
    """Raw full PMF as offline scientific weights, with no clipping/normalization."""
    M = _integer(M, "M")
    pmf = np.asarray(joint_pmf, dtype=float)
    size = 2*(M+1)
    if pmf.shape != (size,)*3:
        raise ValueError("full joint PMF must have shape (2*(M+1),)**3")
    indices = np.indices(pmf.shape).reshape(3, -1).T
    counts, labels = indices % (M+1), 2*(indices//(M+1))-1
    _patterns(counts, labels, pmf.ravel(), M)
    return {"counts": counts, "labels": labels, "weights": pmf.ravel(),
            "state_indices": indices, "data_role": "population_offline_PMF_not_learning_visible",
            "raw_mass": float(pmf.sum()), "automatic_normalization": False}


def prior_logell_constant(counts, labels, *, M, epsilon, tau, inner_order, marginal_order):
    """C-independent prior E log ell via 1D GH; ordinary numerical restoration.

    For unit-diagonal C, log ell is additive in its three standard-normal
    marginals. Same-rule moments approximate this constant, not a certified
    continuous integral. A constant error changes levels but no C gradient.
    """
    M = _integer(M, "M")
    counts, labels = np.asarray(counts), np.asarray(labels)
    _patterns(counts, labels, np.full(len(counts), 1/len(counts)), M)
    epsilon, tau = competition._parameters(epsilon, tau)
    positive,negative,active_positive,active_negative=_prior_expectations(tuple(epsilon),tau,int(inner_order),int(marginal_order))
    active = np.where(labels > 0, active_positive, active_negative).sum(axis=1)
    choose = np.asarray([math.log(math.comb(M, c)) for c in range(M+1)])
    return active + np.sum(choose[counts]+counts*positive+(M-counts)*negative, axis=1)


def evaluate(z, C, counts, labels, weights, *, M, epsilon, tau, inner_order,
             control="raw", marginal_order=128, pattern_chunk=64,
             return_pattern_details=False):
    """Fixed differentiable scalar and its exact physical-coordinate RP gradient.

    raw is the fixed SAA of E log(K^-1 sum ell). uniform_log evaluates a
    *different fixed surrogate* f+mean(log ell)-E_prior(log ell), with the same
    population target under true prior sampling. Its own RP gradient is exact;
    Price is not substituted. Both count multiplicities are retained.
    """
    M, pattern_chunk = _integer(M, "M"), _integer(pattern_chunk, "pattern_chunk")
    epsilon, tau = competition._parameters(epsilon, tau)
    counts, labels, weights, states = _patterns(counts, labels, weights, M)
    if control not in ("raw", "uniform_log"):
        raise ValueError("control must be raw or uniform_log")
    z = np.asarray(z, dtype=float)
    if z.ndim != 4 or min(z.shape) < 1 or z.shape[-1] != 3 or not np.isfinite(z).all():
        raise ValueError("caller-owned finite nonempty z[replicate,B,K,3] required")
    R, B, K, _ = z.shape
    x, dx = legacy.candidate_inputs(z.reshape(R*B, K, 3), C)
    h = h_first(x*epsilon, tau=tau, inner_order=inner_order)
    S = 2*(M+1)
    count_state = np.arange(S) % (M+1)
    label_state = np.where(np.arange(S) < M+1, -1, 1)
    choose = np.asarray([math.log(math.comb(M, c)) for c in range(M+1)])
    signed = x[..., None]*label_state
    logell = (log_expit(signed)+choose[count_state]
              +count_state*h["log_h"][..., None]+(M-count_state)*h["log_complement"][..., None])
    score = (label_state*expit(-signed)+epsilon[None,None,:,None]*(
        count_state*h["dlog_h"][..., None]+(M-count_state)*h["dlog_complement"][..., None]))
    use_cv = control == "uniform_log"
    constant = (prior_logell_constant(counts, labels, M=M, epsilon=epsilon, tau=tau,
                                      inner_order=inner_order, marginal_order=marginal_order)
                if use_cv else None)
    bank_value = np.zeros(R*B); bank_gradient = np.zeros((R*B, 3))
    raw_value = np.zeros(R*B); raw_gradient = np.zeros((R*B, 3))
    per_value, per_gradient = [], []
    for start in range(0, len(counts), pattern_chunk):
        stop = min(start+pattern_chunk, len(counts)); block = states[start:stop]
        logweight = sum(logell[:,:,i,block[:,i]] for i in range(3))
        total = logsumexp(logweight, axis=1)
        pi = np.exp(logweight-total[:,None,:])
        raw = math.log(K)-total
        scalar = raw+logweight.mean(axis=1)-constant[None,start:stop] if use_cv else raw
        pattern_weight = weights[start:stop]
        raw_value += raw @ pattern_weight; bank_value += scalar @ pattern_weight
        gradient = np.empty((R*B, stop-start, 3)) if return_pattern_details else None
        weighted_pi=pi*pattern_weight[None,None,:]
        # x1=z1 has no C derivative; x2 depends only on c12; x3 on all three.
        # Aggregate over patterns before contracting dx, avoiding nine large
        # particle-pattern direction arrays while preserving the same RP chain.
        group_derivatives={}
        for i in (1,2):
            state_score=score[:,:,i,block[:,i]]
            raw_i=-np.einsum("fkp,fkp->fk",weighted_pi,state_score)
            selected_i=raw_i+(state_score @ pattern_weight)/K if use_cv else raw_i
            group_derivatives[i]=(raw_i,selected_i,state_score)
        for physical in range(3):
            indices=(1,2) if physical==0 else (2,)
            for i in indices:
                raw_i,selected_i,state_score=group_derivatives[i]
                raw_gradient[:,physical]+=np.sum(raw_i*dx[:,:,i,physical],axis=1)
                bank_gradient[:,physical]+=np.sum(selected_i*dx[:,:,i,physical],axis=1)
            if return_pattern_details:
                directional=sum(group_derivatives[i][2]*dx[:,:,i,physical,None] for i in indices)
                derivative=-np.sum(pi*directional,axis=1)
                if use_cv:derivative+=directional.mean(axis=1)
                gradient[...,physical]=derivative
        if return_pattern_details:
            per_value.append(scalar);per_gradient.append(gradient)
    bank_value = bank_value.reshape(R,B);bank_gradient = bank_gradient.reshape(R,B,3)
    replicate_value = bank_value.mean(axis=1);replicate_gradient = bank_gradient.mean(axis=1)
    if not all(np.isfinite(value).all() for value in (bank_value,bank_gradient,raw_value,raw_gradient)):
        raise FloatingPointError("nonfinite finite-K scalar/gradient; no clipping")
    output = {"schema":SCHEMA,"value":float(replicate_value.mean()),"gradient":replicate_gradient.mean(axis=0),
        "replicate_values":replicate_value,"replicate_gradients":replicate_gradient,
        "bank_values":bank_value,"bank_gradients":bank_gradient,
        "raw_value":float(raw_value.mean()),"raw_gradient":raw_gradient.mean(axis=0),
        "raw_replicate_values":raw_value.reshape(R,B).mean(axis=1),
        "raw_replicate_gradients":raw_gradient.reshape(R,B,3).mean(axis=1),
        "control":control,"prior_logell_constant":constant,"M":M,"K":K,"replicates":R,"banks_per_replicate":B,
        "patterns":len(counts),"inner_order":int(inner_order),"marginal_order":int(marginal_order) if use_cv else None,
        "pattern_mass":float(weights.sum()),"count_multiplicities_included":True,
        "bank_pattern_pairs":R*B*len(counts),"particle_pattern_terms":R*B*K*len(counts),
        "h_argument_locations":R*B*K*3,"private_GH_nodes":R*B*K*3*int(inner_order),
        "max_temporary_particle_pattern_terms":R*B*K*min(pattern_chunk,len(counts)),
        "gradient_qualification":"exact_RP_derivative_of_selected_fixed_scalar_not_Price",
        "uncertainty_unit":"independent_scramble_replicate; no_iid_bank_SE_provided",
        "numerical_qualification":"ordinary_positive_1D_GH; no_continuous_integral_error_certificate",
        "implementation":"same_positive_GH_first_only_sparse_Cholesky_RP",
        "positive_GH_minimum_h_failure_threshold":MIN_H}
    if R > 1:
        output["replicate_se"] = float(replicate_value.std(ddof=1)/math.sqrt(R))
        output["replicate_gradient_se"] = replicate_gradient.std(axis=0,ddof=1)/math.sqrt(R)
        output["raw_replicate_se"]=float(output["raw_replicate_values"].std(ddof=1)/math.sqrt(R))
        output["raw_replicate_gradient_se"]=output["raw_replicate_gradients"].std(axis=0,ddof=1)/math.sqrt(R)
    else:
        output["replicate_se"] = None;output["replicate_gradient_se"] = None
        output["raw_replicate_se"]=None;output["raw_replicate_gradient_se"]=None
    if return_pattern_details:
        output["pattern_values"] = np.concatenate(per_value,axis=1).reshape(R,B,len(counts))
        output["pattern_gradients"] = np.concatenate(per_gradient,axis=1).reshape(R,B,len(counts),3)
    return output


def empirical_objective(z, C, counts, observed_labels, **kwargs):
    patterns = empirical_patterns(counts, observed_labels, M=kwargs["M"])
    return evaluate(z,C,patterns["counts"],patterns["labels"],patterns["weights"],**kwargs)


def population_objective(z, C, joint_pmf, **kwargs):
    """Offline population reference; never a training-visible teacher input."""
    patterns = population_patterns(joint_pmf,M=kwargs["M"])
    output = evaluate(z,C,patterns["counts"],patterns["labels"],patterns["weights"],**kwargs)
    output["data_role"] = patterns["data_role"]
    return output
