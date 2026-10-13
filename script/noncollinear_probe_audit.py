"""Independent numerical/execution checks for NoncollinearProbe-v1.

Saved paired Laplace differences, rather than subtraction of rounded absolute
losses, are the primary audit object. Independent selected-label integration
uses the prescribed C inner rule and adaptive outer quadrature. None of these
comparisons certifies the Gaussian quadrature remainder.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.integrate import quad_vec
import rankone_stationary_audit as audit_base

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/noncollinear_probe_v1_20261007"
RECEIPT=ART/"AUDIT.json"
read,sha,rel,native=audit_base.read,audit_base.sha,audit_base.rel,audit_base.native
INDEPENDENT_LABELS=(0,85,255)
INDEPENDENT_T=(.02,.10)
POINTS=(("zero",0.),("m010",-.10),("m005",-.05),("m002",-.02),("p002",.02),("p005",.05),("p010",.10))
INDEPENDENT_SPEC=dict(inner_rule="numpy leggauss384 on [-11,11] times hermegauss40",
    logu_interval=[-36.,72.],breakpoints=[-24.,-16.,-10.,-5.,0.,5.,10.,20.,32.,48.],
    epsabs=2e-12,epsrel=1e-10,norm="max",absolute_delta_comparison_tolerance=2e-10)


def independent_nodes(order1=384,limit1=11.,order2=40):
    x,w1=np.polynomial.legendre.leggauss(order1);x=limit1*x
    w1=limit1*w1*np.exp(-x*x/2)/math.sqrt(2*math.pi)
    y,w2=np.polynomial.hermite_e.hermegauss(order2);w2=w2/math.sqrt(2*math.pi)
    return np.column_stack([np.repeat(x,order2),np.tile(y,order1)]),np.repeat(w1,order2)*np.tile(w2,order1)


def log_likelihood(b,W,nodes,labels):
    logits=np.asarray(b)[None,:]+nodes@np.asarray(W).T
    logp=-np.logaddexp(0.,-logits);logno=-np.logaddexp(0.,logits)
    return logp@np.asarray(labels).T+logno@(1-np.asarray(labels)).T


def independent_pmf(b,W,nodes,weights):
    m=len(b);Y=((np.arange(1<<m)[:,None]>>np.arange(m-1,-1,-1))&1)
    result=np.empty(len(Y))
    for first in range(0,len(Y),32):
        block=Y[first:first+32]
        result[first:first+len(block)]=weights@np.exp(log_likelihood(b,W,nodes,block))
    return result


def stable_pair_at_u(u,x0,x1,weights,K):
    """Power difference through its telescoping polynomial, preserving dPsi.

    dPsi uses a nodewise stable exponential difference before integration.
    No power-difference subtraction or production log-Psi routine is used.
    """
    a0=-u*x0;a1=-u*x1;delta=-u*(x1-x0)
    difference=np.sign(delta)*np.exp(np.maximum(a0,a1))*(-np.expm1(-abs(delta)))
    dpsi=float(weights@difference)
    psi0=float(weights@np.exp(a0));psi1=float(weights@np.exp(a1))
    polynomial=math.fsum(psi1**j*psi0**(K-1-j) for j in range(K))
    return dpsi*polynomial


def adaptive_pair(x0,x1,weights,K,lo=-36.,hi=72.):
    values,error,info=quad_vec(lambda v:stable_pair_at_u(math.exp(float(v)),x0,x1,weights,K),lo,hi,
        points=[v for v in INDEPENDENT_SPEC["breakpoints"] if lo<v<hi],
        epsabs=INDEPENDENT_SPEC["epsabs"],epsrel=INDEPENDENT_SPEC["epsrel"],norm="max",full_output=True)
    return float(values),dict(quad_success=bool(info.success),quad_status=int(info.status),estimated_error=float(error),
        integrand_evaluations=int(info.neval),adaptive_intervals=len(info.intervals),quad_message=str(info.message))


def independent_label(b,W0,W1,label,K,nodes,weights,shared_logscale=None):
    m=len(b);y=np.array([[(label>>j)&1 for j in range(m-1,-1,-1)]])
    logA0=log_likelihood(b,W0,nodes,y)[:,0];logA1=log_likelihood(b,W1,nodes,y)[:,0]
    if shared_logscale is None:
        shared_logscale=math.log(float(weights@np.exp(logA0))/math.fsum(float(w) for w in weights))
    x0=np.exp(logA0-shared_logscale);x1=np.exp(logA1-shared_logscale)
    value,info=adaptive_pair(x0,x1,weights,int(K))
    p0,p1=float(weights@np.exp(logA0)),float(weights@np.exp(logA1))
    info.update(label_index=label,K=int(K),shared_logscale=float(shared_logscale),pmf0=p0,pmf1=p1,
        paired_delta=value,direct_log_evidence_delta=math.log(p0)-math.log(p1))
    return value,info


def exp_difference(left,right):
    x,y=np.broadcast_arrays(left,right);out=np.zeros(x.shape,dtype=float)
    positive=x>y;negative=x<y
    out[positive]=np.exp(x[positive])*(-np.expm1(y[positive]-x[positive]))
    out[negative]=np.exp(y[negative])*np.expm1(x[negative]-y[negative])
    return out


def reconstruct(raw,baseline,checks,context):
    """Check every saved transform/integrand and rebuild all aggregations.

    The complete transforms are inspected and used algebraically; the 12-cell
    check, not this routine, independently recomputes the outer integral.
    """
    isbase=baseline is None
    q=np.asarray(raw["targets"]);Y=raw["patterns"];v=raw["outer_logu_nodes"];wv=raw["outer_weights"]
    Ks=list(raw["K_names"]);finite=[(ki,int(k)) for ki,k in enumerate(Ks) if k!="inf"]
    M=float(raw["normal_quadrature_mass"]);logM=math.log(M)
    for key,value in raw.items():
        a=np.asarray(value)
        if a.dtype.kind in "fc":
            legal=(~np.isnan(a)&~np.isposinf(a)).all() if key=="logPsi" else np.isfinite(a).all()
            checks.require(legal,context+":finite_or_legal_logzero_"+key)
    checks.require(np.all(raw["pmf"]>=0),context+":nonnegative_PMF")
    checks.require(np.max(raw["logPsi"]-logM)<=2e-12,context+":Laplace_mass_upper_bound")
    prev,next_=raw["logPsi"][...,:-1],raw["logPsi"][...,1:]
    with np.errstate(invalid="ignore"):
        differences=np.where(np.isneginf(prev)&np.isneginf(next_),0.,next_-prev)
    checks.require(np.max(differences)<=2e-12,context+":Laplace_monotonicity")
    checks.close(raw["q_mass"],q.sum(axis=1),"raw_q_mass",context,0.)
    checks.close(M,math.fsum(float(w) for w in raw["normal_weights"]),"raw_normal_mass",context,0.)
    bank=np.array([1. if k=="inf" else math.exp(int(k)*logM) for k in Ks])
    checks.close(raw["raw_bank_quadrature_mass"],bank,"raw_bank_mass",context,0.)
    checks.close(raw["pmf"],np.exp(raw["log_evidence"]),"evidence_to_PMF",context,0.)
    checks.close(raw["marginals"],raw["pmf"]@Y,"marginals_from_PMF",context)
    checks.close(raw["model_mass"],np.sum(raw["pmf"],axis=-1),"model_mass",context,0.)
    entropy=np.sum(np.where(q>0,q*np.log(np.where(q>0,q,1.)),0.),axis=1)
    if isbase:
        checks.close(raw["log_likelihood_scale"],raw["log_evidence"]-logM,"baseline_shared_scale",context,0.)
        label=np.empty_like(raw["label_loss"])
        for fi,(ki,K) in enumerate(finite):
            reference=K*logM-K*np.exp(v)
            integ=exp_difference(K*raw["logPsi"],reference[None,:])
            checks.close(raw["absolute_integrand"][fi],integ,"absolute_integrand",context,2e-14)
            label[ki]=-bank[ki]*raw["log_likelihood_scale"]+integ@wv
        for ki,k in enumerate(Ks):
            if k=="inf":label[ki]=-raw["log_evidence"]
        loss=q@label.T
        kl=entropy-q@raw["log_evidence"]
        checks.close(raw["label_loss"],label,"label_loss",context)
        checks.close(raw["absolute_loss"],loss,"absolute_loss",context)
        checks.close(raw["raw_KL"],kl,"raw_KL",context)
        # Conditional log likelihoods are small enough to reconstruct in label
        # chunks; this is not a replay of the full inner-by-outer Laplace grid.
        pmf=np.empty(len(Y))
        for first in range(0,len(Y),32):
            logA=log_likelihood(raw["b"],raw["W0"],raw["normal_nodes"],Y[first:first+32])
            checks.close(raw["conditional_logA"][:,first:first+32],logA,"baseline_conditional_logA",context)
            pmf[first:first+len(logA.T)]=raw["normal_weights"]@np.exp(logA)
        checks.close(raw["pmf"],pmf,"baseline_direct_PMF",context)
        return dict(absolute_loss=loss,raw_KL=kl,pmf=raw["pmf"],marginals=raw["marginals"])
    for key in ("normal_nodes","normal_weights","normal_axis_0_nodes","normal_axis_0_weights","normal_axis_1_nodes",
        "normal_axis_1_weights","normal_axis_masses","outer_logu_nodes","outer_weights","log_likelihood_scale","b","W0","targets"):
        checks.exact(raw[key],baseline[key],context+":shared_"+key)
    for key in ("logPsi","log_evidence","pmf","marginals"):
        checks.exact(raw[key][0],baseline[key],context+":baseline_copy_"+key)
    for key in ("absolute_loss","label_loss","absolute_integrand","raw_KL"):
        checks.exact(raw[key][...,0],baseline[key],context+":baseline_copy_"+key)
    label=np.empty_like(raw["label_loss"]);delta=np.empty_like(raw["label_delta"])
    for fi,(ki,K) in enumerate(finite):
        paired=exp_difference(K*raw["logPsi"][1],K*raw["logPsi"][0])
        checks.close(raw["paired_integrand"][fi],paired,"paired_integrand",context,2e-14)
        delta[ki]=paired@wv
        reference=K*logM-K*np.exp(v)
        for side in range(2):
            integ=exp_difference(K*raw["logPsi"][side],reference[None,:])
            checks.close(raw["absolute_integrand"][fi,:,:,side],integ,"absolute_integrand",context,2e-14)
            label[ki,:,side]=-bank[ki]*raw["log_likelihood_scale"]+integ@wv
    for ki,k in enumerate(Ks):
        if k=="inf":
            label[ki]=-raw["log_evidence"].T
            delta[ki]=raw["log_evidence"][0]-raw["log_evidence"][1]
    loss=np.stack([q@label[...,side].T for side in range(2)],axis=-1)
    paired=q@delta.T
    risk=-(q@(raw["log_evidence"][1]-raw["log_evidence"][0]))
    kl=entropy[:,None]-q@raw["log_evidence"].T
    for key,value in dict(label_loss=label,label_delta=delta,absolute_loss=loss,paired_delta=paired,paired_risk_delta=risk,
        raw_KL=kl,absolute_difference=loss[...,1]-loss[...,0],delta_consistency_error=paired-(loss[...,1]-loss[...,0]),
        risk_consistency_error=risk-(kl[:,1]-kl[:,0])).items():
        checks.close(raw[key],value,"reconstructed."+key,context)
    return dict(absolute_loss=loss,paired_delta=paired,paired_risk_delta=risk,raw_KL=kl,pmf=raw["pmf"],marginals=raw["marginals"])


def prepared_fingerprint(raw,summary):
    """Reconstruct the immutable prepared-state fingerprint from saved fields."""
    prepared={k:v for k,v in raw.items() if k not in ("theta","direction","t","K_names","finite_K_values","target_names")}
    prepared["normal_quadrature_mass"]=float(raw["normal_quadrature_mass"])
    prepared["model_mass"]=float(raw["model_mass"])
    prepared.update(schema="noncollinear_probe_prepared_v1",K_values=tuple("inf" if k=="inf" else int(k) for k in raw["K_names"]),
        finite_K_values=tuple(int(k) for k in raw["finite_K_values"]),target_names=tuple(str(k) for k in raw["target_names"]),
        config=summary["config"],normal_specs=summary["config"]["normal_specs"],work_counts=summary["work_counts"])
    h=hashlib.sha256()
    def update(x):
        if isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(memoryview(np.ascontiguousarray(x)).cast("B"))
        elif isinstance(x,dict):
            for key in sorted(x):h.update(str(key).encode());update(x[key])
        elif isinstance(x,(list,tuple)):
            for item in x:update(item)
        else:h.update(json.dumps(native(x),sort_keys=True).encode())
    update(prepared)
    return h.hexdigest()


def check_summary(saved,raw,anchor,d,t,checks,context):
    isbase=t==0.;theta=np.asarray(anchor["theta"]);a=theta[8:]
    expected_W=a[:,None]*np.column_stack([np.cos(t*d),np.sin(t*d)])
    W=raw["W0"] if isbase else raw["W1"]
    checks.exact(raw["b"],theta[:8],context+":fixed_bias")
    checks.exact(raw["theta"],theta,context+":locked_theta")
    checks.exact(raw["direction"],d,context+":fixed_direction")
    checks.exact(float(raw["t"]),t,context+":fixed_t")
    checks.exact(W,expected_W,context+":fixed_angular_path")
    facts=dict(t=t,K_values=[int(k) if k!="inf" else "inf" for k in raw["K_names"]],
        normal_mass=float(raw["normal_quadrature_mass"]),target_mass=float(raw["q_mass"][0]),
        all_outputs_finite=all((~np.isnan(v)&~np.isposinf(v)).all() if k=="logPsi" else np.isfinite(v).all()
            for k,v in raw.items() if np.asarray(v).dtype.kind in "fc"),
        PMF_nonnegative=bool(np.all(raw["pmf"]>=0)),logPsi_zero_count=int(np.isneginf(raw["logPsi"]).sum()))
    norms=np.linalg.norm(W,axis=1)
    geom=dict(bias_exactly_equal=bool(np.array_equal(raw["b"],theta[:8])),row_norm_max_error=float(np.max(abs(norms-a))),
        row_norms=norms,singular_values=np.linalg.svd(W,compute_uv=False),path_W_max_error=float(np.max(abs(W-expected_W))))
    checks.exact(saved["geometry"],geom,context+":geometry")
    if isbase:
        facts.update(absolute_loss=raw["absolute_loss"][0],raw_KL=float(raw["raw_KL"][0]),pmf=raw["pmf"],
            marginals=raw["marginals"],model_mass=float(raw["model_mass"]),paired_F=0.,paired_R=0.,paired_infinite=0.,
            reused_zero=True,no_additional_zero_evaluation=True,quadratic_F=None,quadratic_R=None)
    else:
        own=list(raw["K_names"]).index(str(anchor["K"]));inf=list(raw["K_names"]).index("inf")
        F=float(raw["paired_delta"][0,own]);R=float(raw["paired_risk_delta"][0])
        facts.update(paired_F=F,paired_R=R,paired_infinite=float(raw["paired_delta"][0,inf]),paired_delta=raw["paired_delta"][0],
            absolute_loss=raw["absolute_loss"][0,:,1],baseline_absolute_loss=raw["absolute_loss"][0,:,0],
            absolute_difference=raw["absolute_difference"][0],raw_KL=float(raw["raw_KL"][0,1]),baseline_raw_KL=float(raw["raw_KL"][0,0]),
            delta_consistency_error=raw["delta_consistency_error"][0],risk_consistency_error=float(raw["risk_consistency_error"][0]),
            infinite_risk_consistency_error=float(raw["paired_delta"][0,inf]-raw["paired_risk_delta"][0]),
            pmf=raw["pmf"][1],baseline_pmf=raw["pmf"][0],marginals=raw["marginals"][1],baseline_marginals=raw["marginals"][0],
            marginal_max_difference=float(np.max(abs(raw["marginals"][1]-raw["marginals"][0]))),model_mass=float(raw["model_mass"][1]),
            quadratic_F=2*F/(t*t),quadratic_R=2*R/(t*t))
        checks.exact(saved["old_curvature"],anchor["curvature"],context+":old_F_curvature")
        checks.exact(saved["old_risk_curvature"],anchor["risk_curvature"],context+":old_R_curvature")
        checks.exact(saved["quadratic_F_minus_old"],{r:facts["quadratic_F"]-value for r,value in anchor["curvature"].items()},context+":quadratic_F_comparison")
        checks.exact(saved["quadratic_R_minus_old"],{r:facts["quadratic_R"]-value for r,value in anchor["risk_curvature"].items()},context+":quadratic_R_comparison")
    for key,value in facts.items():checks.exact(saved[key],value,context+":summary_"+key)
    return native(dict(**facts,geometry=geom))


def baseline_gate(record,old):
    expected=[next(b["loss"] for b in old["blocks"] if b["target"]=="population" and b["K"]==k) for k in record["K_values"]]
    comparisons=dict(old_1D_D_loss_max_difference=float(np.max(abs(np.asarray(record["absolute_loss"])-expected))),
        old_1D_D_PMF_max_difference=float(np.max(abs(np.asarray(record["pmf"])-old["probability"]["pmf"]))))
    gates=dict(outputs_finite=record["all_outputs_finite"],PMF_nonnegative=record["PMF_nonnegative"],
        normal_mass=abs(record["normal_mass"]-1)<=2e-12,model_mass=abs(record["model_mass"]-1)<=2e-12,
        bias_unchanged=record["geometry"]["bias_exactly_equal"],row_norm=record["geometry"]["row_norm_max_error"]<=5e-14,
        old_1D_loss=comparisons["old_1D_D_loss_max_difference"]<=1e-7,old_1D_PMF=comparisons["old_1D_D_PMF_max_difference"]<=2e-10)
    return dict(qualified=all(gates.values()),gates=gates,failed_gates=[k for k,v in gates.items() if not v],comparisons=comparisons)


def even_gate(minus,plus):
    comparisons=dict(F_difference=abs(minus["paired_F"]-plus["paired_F"]),R_difference=abs(minus["paired_R"]-plus["paired_R"]),
        PMF_max_difference=float(np.max(abs(np.asarray(minus["pmf"])-plus["pmf"]))))
    gates=dict(F=comparisons["F_difference"]<=1e-11,R=comparisons["R_difference"]<=1e-11,PMF=comparisons["PMF_max_difference"]<=2e-12)
    return dict(qualified=all(gates.values()),gates=gates,failed_gates=[k for k,v in gates.items() if not v],comparisons=comparisons)


def precision_gate(left,right,effect,pair):
    diff=abs(left-right);relative=(.02 if pair=="BC" else .10)*max(abs(left),abs(right),1e-12)
    absolute={"FBC":1e-10,"FAC":1e-9,"RBC":1e-11,"RAC":1e-10}[effect+pair]
    gates=dict(absolute=diff<=absolute,relative=diff<=relative)
    return dict(qualified=all(gates.values()),gates=gates,difference=diff,absolute_limit=absolute,relative_limit=relative)


def sign_gate(B,C):
    difference=abs(B-C);same=B*C>0;floor=max(1e-9,20*difference)
    resolved=same and min(abs(B),abs(C))>floor
    return dict(resolved=resolved,sign=("positive" if B>0 else "negative") if resolved else "unresolved",
        B=B,C=C,B_C_difference=difference,same_nonzero_sign=same,required_minimum_magnitude=floor,
        probabilistic_or_rigorous_error_bound=False)


def point_gate(records,baselines,even):
    shared={};comparisons={}
    for rule,r in records.items():
        shared.update({rule+"_baseline":baselines[rule]["qualification"]["qualified"],rule+"_finite":r["all_outputs_finite"],
            rule+"_PMF_nonnegative":r["PMF_nonnegative"],rule+"_normal_mass":abs(r["normal_mass"]-1)<=2e-12,
            rule+"_model_mass":abs(r["model_mass"]-1)<=2e-12,rule+"_bias":r["geometry"]["bias_exactly_equal"],
            rule+"_row_norm":r["geometry"]["row_norm_max_error"]<=5e-14,rule+"_marginal":r["marginal_max_difference"]<=2e-10,
            rule+"_delta_interface":np.max(abs(np.asarray(r["delta_consistency_error"])))<=5e-12,
            rule+"_risk_interface":abs(r["risk_consistency_error"])<=5e-12,
            rule+"_infinite_risk_interface":abs(r["infinite_risk_consistency_error"])<=5e-12,
            rule+"_even_PMF":even[rule]["gates"]["PMF"]})
    for rule in ("A","B"):
        comparisons[rule+"_C"]=dict(loss_max_difference=float(np.max(abs(np.asarray(records[rule]["absolute_loss"])-records["C"]["absolute_loss"]))),
            raw_KL_difference=abs(records[rule]["raw_KL"]-records["C"]["raw_KL"]),
            PMF_max_difference=float(np.max(abs(np.asarray(records[rule]["pmf"])-records["C"]["pmf"]))))
        c=comparisons[rule+"_C"]
        shared.update({rule+"_C_absolute_loss":c["loss_max_difference"]<=1e-7,rule+"_C_raw_KL":c["raw_KL_difference"]<=1e-10,
                       rule+"_C_PMF":c["PMF_max_difference"]<=2e-10})
    effects={}
    for effect in ("F","R"):
        precision={pair:precision_gate(records[pair[0]]["paired_"+effect],records["C"]["paired_"+effect],effect,pair) for pair in ("BC","AC")}
        gates={"precision_"+pair:p["qualified"] for pair,p in precision.items()}
        gates.update({rule+"_evenness":e["gates"][effect] for rule,e in even.items()})
        effects[effect]=dict(qualified=all(shared.values()) and all(gates.values()),gates=gates,failed_gates=[k for k,v in gates.items() if not v],
            precision=precision,sign=sign_gate(records["B"]["paired_"+effect],records["C"]["paired_"+effect]))
    return native(dict(qualified=all(e["qualified"] for e in effects.values()),qualified_F=effects["F"]["qualified"],qualified_R=effects["R"]["qualified"],
        shared_gates=shared,shared_failed_gates=[k for k,v in shared.items() if not v],absolute_comparisons=comparisons,
        evenness=even,effects=effects,strict_numeric_error_bound_claimed=False,statistical_confidence_claimed=False))


def bind_artifact(item,checks):
    checks.bind(ROOT/item["path"],item["sha256"],item.get("bytes"))


def load_raw(record,q,rule,checks,resources,isbase):
    bind_artifact(record["array_artifact"],checks);bind_artifact(record["summary_artifact"],checks)
    checks.exact(read(ROOT/record["summary_artifact"]["path"]),{k:v for k,v in record.items() if k!="summary_artifact"},"summary_snapshot")
    with np.load(ROOT/record["array_artifact"]["path"],allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    context=record["array_artifact"]["path"]
    checks.exact(raw["targets"],[q],context+":target_not_renormalized")
    checks.exact(raw["target_names"],["population"],context+":target_name")
    checks.exact(raw["K_names"],[str(record["K"]),"inf"],context+":own_K_and_inf")
    checks.exact(raw["finite_K_values"],[record["K"]],context+":finite_K")
    labels=((np.arange(256)[:,None]>>np.arange(7,-1,-1))&1).astype(np.int8)
    checks.exact(raw["patterns"],labels,context+":binary_labels")
    n0,n1,lo,hi,step={"A":(96,24,-28.,56.,.25),"B":(144,32,-32.,64.,.125),"C":(384,40,-36.,72.,.125)}[rule]
    z0,z1=raw["normal_axis_0_nodes"],raw["normal_axis_1_nodes"]
    w0,w1=raw["normal_axis_0_weights"],raw["normal_axis_1_weights"]
    checks.exact((len(z0),len(z1)),(n0,n1),context+":axis_orders")
    checks.exact(raw["normal_nodes"],np.column_stack([np.repeat(z0,n1),np.tile(z1,n0)]),context+":tensor_nodes")
    checks.exact(raw["normal_weights"],(w0[:,None]*w1[None,:]).ravel(),context+":tensor_weights")
    checks.close(raw["normal_axis_masses"],[math.fsum(float(x) for x in w0),math.fsum(float(x) for x in w1)],"axis_masses",context,0.)
    v=np.linspace(lo,hi,round((hi-lo)/step)+1);wv=np.full(len(v),step);wv[[0,-1]]/=2
    checks.exact(raw["outer_logu_nodes"],v,context+":outer_nodes")
    checks.exact(raw["outer_weights"],wv,context+":outer_weights")
    N=n0*n1;P=256;V=len(v)
    work=dict(kernel_evaluations=1,baseline_preparations=int(isbase),perturbed_model_evaluations=int(not isbase),
        tensor_normal_rules_constructed=int(isbase),bernoulli_logit_values=N*8,conditional_loglikelihood_values=N*P,
        laplace_exp_minus_ux_factors=P*V*N,laplace_logPsi_values=P*V,finite_absolute_integrand_values=P*V,
        paired_integrand_values=0 if isbase else P*V,new_random_draws=0,optimizer_updates=0)
    for key,value in work.items():checks.exact(record["work_counts"][key],value,context+":work_"+key)
    resources.append(work)
    return raw


def outcome(anchors):
    amplitudes={}
    for amp in (.02,.05,.10):
        rows=[]
        for anchor in anchors:
            for side in (-1.,1.):
                p=next(p for p in anchor["points"] if p["t"]==side*amp);q=p["qualification"]
                expected="positive" if anchor["K"]==11 else "negative"
                fs=q["effects"]["F"]["sign"]["sign"];rs=q["effects"]["R"]["sign"]["sign"]
                support=q["qualified"] and fs==expected and rs=="negative"
                rows.append(dict(anchor_name=anchor["name"],K=anchor["K"],t=p["t"],qualified=q["qualified"],
                    F_sign=fs,R_sign=rs,expected_F_sign=expected,expected_R_sign="negative",supports_expected_combination=support))
        resolved=all(r["qualified"] and r["F_sign"]!="unresolved" and r["R_sign"]!="unresolved" for r in rows)
        support=all(r["supports_expected_combination"] for r in rows)
        status="supported" if support else "resolved_different_combination" if resolved else "unresolved_or_unqualified"
        amplitudes[str(amp)]=dict(amplitude=amp,status=status,cells=rows,supported=support)
    return dict(primary_amplitude=.02,primary=amplitudes["0.02"],sensitivities=[amplitudes["0.05"],amplitudes["0.1"]],
        larger_amplitude_does_not_replace_primary=True,fixed_panel_complete_is_stopping_rule=True,
        further_rules_amplitudes_K_or_optimization_added=False)


def perform():
    tick=time.perf_counter();checks=audit_base.Checks();resources=[]
    if not (ART/"RESULTS.json").exists() or read(ART/"RESULTS.json").get("status")!="completed":
        raise RuntimeError("The frozen full panel must complete before its audit")
    fh=checks.bind(ART/"FREEZE.json");sh=checks.bind(ART/"SOURCE.json");afh=checks.bind(ART/"AUDIT_FREEZE.json")
    frozen=read(ART/"FREEZE.json");source=read(ART/"SOURCE.json");af=read(ART/"AUDIT_FREEZE.json")
    checks.exact(af["scientific_freeze_sha256"],fh,"audit_freeze:scientific_source")
    checks.exact(af["sources"].get(rel(__file__)),sha(__file__),"audit_freeze:new_audit")
    checks.exact(af["sources"].get(rel(audit_base.__file__)),sha(audit_base.__file__),"audit_freeze:reused_checks")
    checks.exact(frozen["source_sha256"],sh,"freeze:source")
    for mapping in (frozen["sources"],source["sources"],af["sources"]):
        for path,digest in mapping.items():checks.bind(ROOT/path,digest)
    config=frozen["config"]
    checks.exact(config["points"],[dict(point_id=p,t=t) for p,t in POINTS],"fixed_seven_point_panel")
    checks.exact(config["primary_amplitude"],.02,"fixed_primary_amplitude")
    result=read(ART/"RESULTS.json");checks.bind(ART/"RESULTS.json")
    checks.exact(result["source_sha256"],sh,"result:source")
    checks.exact(result["freeze_sha256"],fh,"result:freeze")
    for path,digest in result["artifacts"].items():checks.bind(ROOT/path,digest)
    bind_artifact(result["panel_lock_artifact"],checks);panel=read(ART/"PANEL_LOCK.json")
    checks.exact(panel["locked_before_any_2D_evaluation"],True,"panel:prelocked")
    checks.exact(panel["source_sha256"],sh,"panel:source")
    checks.exact(panel["freeze_sha256"],fh,"panel:freeze")
    q=np.asarray(source["teacher_pmf"]);d=np.asarray(source["direction"])
    oldplan=read(ROOT/source["cross_plan"]);oldcross=read(ROOT/source["cross_results"])
    checks.exact(d,oldplan["direction"],"source:unchanged_direction")
    checks.exact(panel["direction"],d,"panel:direction")
    checks.close(d@d,1.,"direction_norm","source")
    checks.close(d.sum(),0.,"direction_common_rotation","source")
    checks.exact([a["K"] for a in source["anchors"]],[11,12],"source:fixed_anchor_K")
    checks.exact(len(result["anchors"]),2,"two_anchor_results")
    Cnodes,Cweights=independent_nodes();independent=[];pmf_checks=[];anchor_checks=[]
    for ai,(anchor,summary,locked) in enumerate(zip(source["anchors"],result["anchors"],panel["anchors"])):
        name=anchor["name"];theta=np.asarray(anchor["theta"]);a=theta[8:];W0=np.column_stack([a,np.zeros(8)])
        checks.exact(theta,oldplan["theta_L" if ai==0 else "theta_H"],name+":old_locked_anchor")
        checks.exact(locked["theta"],theta,name+":panel_theta")
        checks.exact(locked["K"],anchor["K"],name+":panel_K")
        checks.exact(summary["theta"],theta,name+":summary_theta")
        bind_artifact(summary["result_artifact"],checks)
        checks.exact(read(ART/"anchors"/name/"RESULTS.json"),{k:v for k,v in summary.items() if k!="result_artifact"},name+":result_snapshot")
        old_summaries={r:read(ROOT/anchor["one_d"][r]) for r in ("C","D")}
        for r,old in old_summaries.items():
            checks.exact(old["theta"],theta,name+":old_theta")
            checks.exact(anchor["curvature"][r],oldcross["cells"]["LL" if ai==0 else "HH"][r+"_curvature"],name+":old_F_curvature")
            H=np.asarray(next(b for b in old["blocks"] if b["target"]=="population" and b["K"]=="inf")["Hangle"])
            checks.close(anchor["risk_curvature"][r],d@H@d,"old_R_curvature",name)
        checks.exact([(p["point_id"],p["t"]) for p in summary["points"]],POINTS,name+":all_points")
        for lockpoint,(pid,t) in zip(locked["points"],POINTS):
            W=a[:,None]*np.column_stack([np.cos(t*d),np.sin(t*d)])
            checks.exact(lockpoint,dict(point_id=pid,t=t,b=theta[:8],W=W),name+":frozen_path_point")
        public={p["point_id"]:p for p in summary["points"]};records={};baseline_records={}
        for rule in ("A","B","C"):
            folder=ART/"anchors"/name/rule;baseline_record=summary["baselines"][rule]
            baseline=load_raw(baseline_record,q,rule,checks,resources,True)
            reconstruct(baseline,None,checks,name+"/"+rule+"/zero")
            check_summary(baseline_record,baseline,anchor,d,0.,checks,name+"/"+rule+"/zero")
            gate=baseline_gate(baseline_record,old_summaries["D"])
            checks.exact(baseline_record["qualification"],gate,name+":baseline_gate")
            fingerprint=prepared_fingerprint(baseline,baseline_record)
            checks.exact(baseline_record["prepared_fingerprint"],fingerprint,name+":prepared_fingerprint")
            checks.exact(public["zero"]["rules"][rule],baseline_record,name+":zero_reference_only")
            baseline_records[rule]=baseline_record;records[rule]={"zero":baseline_record}
            if rule=="C":
                pmf=independent_pmf(theta[:8],W0,Cnodes,Cweights)
                err=checks.close(pmf,baseline["pmf"],"independent_full_PMF",name+"/zero",2e-10)
                pmf_checks.append(dict(anchor=name,t=0.,max_abs_error=err))
            for pid,t in POINTS[1:]:
                record=public[pid]["rules"][rule];records[rule][pid]=record
                raw=load_raw(record,q,rule,checks,resources,False)
                reconstruct(raw,baseline,checks,name+"/"+rule+"/"+pid)
                check_summary(record,raw,anchor,d,t,checks,name+"/"+rule+"/"+pid)
                checks.exact(record["independently_evaluated"],True,name+":nonzero_evaluation")
                checks.exact(record["prepared_fingerprint"],fingerprint,name+":immutable_prepared_reference")
                checks.exact(record["baseline_array_artifact"],baseline_record["array_artifact"],name+":same_baseline_array")
                if rule=="C":
                    pmf=independent_pmf(theta[:8],raw["W1"],Cnodes,Cweights)
                    err=checks.close(pmf,raw["pmf"][1],"independent_full_PMF",name+"/"+pid,2e-10)
                    pmf_checks.append(dict(anchor=name,t=t,max_abs_error=err))
                    if t in INDEPENDENT_T:
                        for label in INDEPENDENT_LABELS:
                            value,info=independent_label(theta[:8],W0,raw["W1"],label,anchor["K"],Cnodes,Cweights)
                            checks.require(info["quad_success"],name+"/"+pid+f"/{label}:adaptive_integral")
                            error=checks.close(value,raw["label_delta"][0,label],"independent_paired_label",name+"/"+pid+f"/{label}",2e-10)
                            checks.close(info["shared_logscale"],baseline["log_likelihood_scale"][label],"independent_shared_scale",name+"/"+pid+f"/{label}",2e-12)
                            independent.append(dict(anchor=name,t=t,label=label,saved_delta=float(raw["label_delta"][0,label]),error=error,diagnostics=info))
            receipt=read(folder/"RULE_COMPLETE.json")
            for key,value in dict(prepared_unchanged=True,baseline_fingerprint=fingerprint,baseline_preparations=1,
                independently_evaluated_nonzero_points=6,zero_additional_evaluations=0).items():checks.exact(receipt[key],value,name+":rule_"+key)
            checks.exact(receipt["point_artifacts"],{p:record["summary_artifact"] for p,record in records[rule].items()},name+":seven_public_rule_records")
        even={}
        for suffix in ("002","005","010"):
            even[suffix]={r:even_gate(records[r]["m"+suffix],records[r]["p"+suffix]) for r in records}
        checks.exact(summary["evenness"],even,name+":all_evenness_gates")
        qualified=[]
        for pid,t in POINTS[1:]:
            gate=point_gate({r:records[r][pid] for r in records},baseline_records,even[pid[1:]])
            checks.exact(public[pid]["qualification"],gate,name+":point_qualification")
            qualified.append(dict(point_id=pid,t=t,qualified=gate["qualified"],F_sign=gate["effects"]["F"]["sign"]["sign"],
                                  R_sign=gate["effects"]["R"]["sign"]["sign"],qualification=gate))
        checks.exact(summary["all_nonzero_points_qualified"],all(p["qualified"] for p in qualified),name+":all_point_gates")
        checks.exact(public["zero"]["qualified"],all(b["qualification"]["qualified"] for b in baseline_records.values()),name+":zero_qualification")
        anchor_checks.append(dict(name=name,K=anchor["K"],points=qualified))
        print(json.dumps(dict(audited_anchor=name,independent_C_pmfs=7,independent_paired_labels=6)),flush=True)
    checks.exact(result["outcome"],outcome(result["anchors"]),"fixed_primary_and_sensitivity_outcome")
    factors={}
    for work in resources:
        for key,value in work.items():factors[key]=factors.get(key,0)+value
    counts=dict(baseline_preparations=6,nonzero_evaluations=36,reused_zero_records=6,failed_kernel_calls=0,
        high_level_factor_totals=factors,new_observations=0,new_random_draws=0,optimizer_updates=0,statistical_alpha_spent=0)
    checks.exact(result["resources"],counts,"fixed_actual_work_counts")
    checks.exact(len(resources),42,"six_plus_thirty_six_actual_calls")
    checks.exact(len(independent),12,"twelve_fixed_independent_cells")
    checks.exact(len(pmf_checks),14,"fourteen_C_full_PMF_checks")
    for path,digest in list(checks.hashes.items()):checks.bind(ROOT/path,digest)
    return dict(passed=not checks.failures,source_sha256=sha(__file__),reused_checks_sha256=sha(audit_base.__file__),
        freeze_sha256=fh,audit_freeze_sha256=afh,input_hashes=checks.hashes,failures=checks.failures,checks=checks.count,
        numeric_elements_compared=checks.elements,numerical_maxima=checks.maxima,anchor_checks=anchor_checks,
        independent_pair_checks=independent,independent_full_PMF_checks=pmf_checks,outcome=result["outcome"],
        scope=dict(internal_numerical_execution_audit=True,external_scientific_review=False,
            saved_transform_integrand_and_aggregation_algebra_checked_for_all_42_calls=True,
            full_independent_inner_outer_Psi_replay=False,independent_paired_label_cells=12,
            independent_negative_side_outer_integration=False,all_negative_outputs_checked_by_algebra_and_evenness=True,
            independent_C_PMF_points=14,independent_spec=INDEPENDENT_SPEC,
            Gaussian_or_outer_remainder_certificate=False,new_optimizer_updates=0,new_random_draws=0,new_statistical_alpha=0,
            unresolved_or_reversed_scientific_findings_are_not_audit_failures=True),
        seconds=time.perf_counter()-tick,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))


def self_test():
    rows=[]
    b=np.array([.2,-.3]);a=np.array([.45,.6]);d=np.array([1.,-1.])/math.sqrt(2)
    W0=np.column_stack([a,np.zeros(2)]);W1=np.column_stack([a*np.cos(.05*d),a*np.sin(.05*d)])
    for limit in (7.,1.2):
        nodes,w=independent_nodes(6,limit,4)
        errors=[]
        for label in (0,1,3):
            y=np.array([[(label>>1)&1,label&1]])
            A0=np.exp(log_likelihood(b,W0,nodes,y)[:,0]);A1=np.exp(log_likelihood(b,W1,nodes,y)[:,0])
            s0=A0[:,None]+A0[None,:];s1=A1[:,None]+A1[None,:]
            # Each bank has two 2D particles: this directly sums a 4D rule.
            direct=float(np.sum(w[:,None]*w[None,:]*(-np.log1p((s1-s0)/s0))))
            delta,diag=independent_label(b,W0,W1,label,2,nodes,w)
            errors.append(abs(delta-direct))
            if not diag["quad_success"]:errors.append(math.inf)
        rows.append(dict(kind="direct_K2_four_dimensional_discrete_banks",normal_limit=limit,normal_mass=float(w.sum()),
                         maximum_delta_error=max(errors),passed=max(errors)<2e-10))
    for mass in (1.,.8):
        x0=np.array([1.]);x1=np.array([1.+1e-12]);weights=np.array([mass])
        delta,diag=adaptive_pair(x0,x1,weights,12)
        expected=-(mass**12)*math.log1p(float(x1[0]-x0[0]))
        error=abs(delta-expected)
        rows.append(dict(kind="small_paired_difference_and_raw_mass",normal_mass=mass,delta=delta,expected=expected,
                         absolute_error=error,passed=diag["quad_success"] and error<1e-19))
    # Production helpers are invoked only to manufacture small schema fixtures;
    # the scientific perform() path never imports or calls them.
    import noncollinear_probe_compute as manufactured
    b=np.linspace(-.2,.2,8);a=np.linspace(.3,.5,8);d=np.arange(8.)-3.5;d/=np.linalg.norm(d)
    anchor=dict(theta=np.r_[b,a],K=11,curvature={"C":.1,"D":.1},risk_curvature={"C":-.1,"D":-.1})
    q=np.arange(1.,257.);q/=q.sum();W0=np.column_stack([a,np.zeros(8)])
    prepared=manufactured.kernel.prepare_baseline(b,W0,{"population":q},K_values=(11,"inf"),
        normal_specs=[dict(rule="legendre",order=3,limit=1.2),dict(rule="hermite",order=4)],
        logu_min=-12.,logu_max=12.,logu_step=1.)
    checks=audit_base.Checks();raw0=manufactured.arrays_for_save(prepared,anchor,d,0.)
    base_record=manufactured.baseline_summary(prepared,anchor,d)
    reconstruct(raw0,None,checks,"manufactured_baseline")
    check_summary(base_record,raw0,anchor,d,0.,checks,"manufactured_baseline")
    checks.exact(prepared_fingerprint(raw0,base_record),manufactured.fingerprint(prepared),"manufactured_prepared_fingerprint")
    points={}
    for t in (-.02,.02):
        result=manufactured.kernel.evaluate_prepared(prepared,a[:,None]*np.column_stack([np.cos(t*d),np.sin(t*d)]))
        raw=manufactured.arrays_for_save(result,anchor,d,t);record=manufactured.point_summary(result,anchor,d,t)
        reconstruct(raw,raw0,checks,"manufactured_pair")
        check_summary(record,raw,anchor,d,t,checks,"manufactured_pair")
        points[t]=record
    even=even_gate(points[-.02],points[.02])
    checks.exact(even,manufactured.evenness(points[-.02],points[.02]),"manufactured_evenness_schema")
    old=dict(blocks=[dict(target="population",K=11,loss=float(raw0["absolute_loss"][0,0])),
                     dict(target="population",K="inf",loss=float(raw0["absolute_loss"][0,1]))],probability=dict(pmf=raw0["pmf"]))
    qualified=baseline_gate(base_record,old)
    checks.exact(qualified,manufactured.qualify_baseline(base_record,old),"manufactured_baseline_gate")
    baseline_records={r:dict(**base_record,qualification=qualified) for r in ("A","B","C")}
    point_records={r:points[.02] for r in ("A","B","C")};evens={r:even for r in ("A","B","C")}
    gate=point_gate(point_records,baseline_records,evens)
    checks.exact(gate,manufactured.qualify_point(point_records,baseline_records,evens),"manufactured_all_gate_schema")
    checks.require(not gate["qualified"],"manufactured_nonunit_mass_failure_retained")
    fake_anchors=[dict(name=f"fixture_{K}",K=K,points=[dict(t=t,qualification=gate) for _,t in POINTS[1:]]) for K in (11,12)]
    checks.exact(outcome(fake_anchors),manufactured.outcome_summary(fake_anchors),"manufactured_primary_sensitivity_reporting")
    checks.close(exp_difference(np.array([-np.inf]),np.array([-np.inf])),np.array([0.]),"legal_log_zero_difference","manufactured",0.)
    return dict(passed=all(r["passed"] for r in rows) and not checks.failures,manufactured_only=True,scientific_inputs_read=False,
                new_random_draws=0,reused_checks_sha256=sha(audit_base.__file__),cases=rows,
                execution_assertions=checks.count,execution_failures=checks.failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test","--selftest",action="store_true")
    mode.add_argument("--run",action="store_true");mode.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_test();print(json.dumps(native(result),indent=2,allow_nan=False));return 0 if result["passed"] else 1
    if args.run and RECEIPT.exists():raise FileExistsError("AUDIT.json exists and will not be replaced")
    old=read(RECEIPT) if args.check else None
    if old is not None and old["source_sha256"]!=sha(__file__):raise ValueError("audit source changed after the saved receipt")
    try:result=perform()
    except BaseException as error:
        result=dict(passed=False,source_sha256=sha(__file__),failures=[dict(exception=str(error),traceback=traceback.format_exc())])
    if args.run:
        result["created_utc"]=datetime.now(timezone.utc).isoformat()
        with RECEIPT.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(native(result),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    elif result["passed"]:
        for key in ("passed","source_sha256","reused_checks_sha256","freeze_sha256","audit_freeze_sha256","input_hashes","failures",
            "checks","numeric_elements_compared","numerical_maxima","anchor_checks","independent_pair_checks","independent_full_PMF_checks","outcome","scope"):
            if native(result[key])!=old[key]:raise ValueError("stable audit field changed on read-only repeat: "+key)
    print(json.dumps(native({k:result[k] for k in ("passed","checks","seconds","failures") if k in result}),allow_nan=False),flush=True)
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
