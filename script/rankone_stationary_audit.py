"""Independent numerical/execution checks for RankOneStationary-v1.

No frozen compute or kernel code is used by the scientific audit.  Its label
checks reconstruct the normal rule with numpy leggauss and integrate log-u by
adaptive quad_vec.  This is a numerical comparison, not an integration bound.
"""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.integrate import quad_vec

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/rankone_stationary_v1_20261007"
RECEIPT = ART / "AUDIT.json"
SOLVE_NAMES = ("E0", "E1", "E2", "E3", "P")
STAGES = (("O",12,80,1e-9),("C",4,40,1e-10))
RULES = {"O":(128,10.,-24.,48.,.25),"C":(256,10.,-28.,56.,.25),"D":(384,11.,-32.,64.,.125)}
LABEL_INDICES = (0, 1, 85, 255)
QUAD_SPEC = dict(normal_rule="numpy.polynomial.legendre.leggauss", normal_order=384,
    normal_limit=11., logu_min=-32., logu_max=64.,
    breakpoints=[-16., -10., -5., 0., 5., 10., 20., 32., 48.],
    epsabs=1e-10, epsrel=1e-9, norm="max", absolute_comparison_tolerance=5e-9)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def native(x):
    if isinstance(x, np.ndarray):
        return native(x.tolist())
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    return x


class Checks:
    def __init__(self):
        self.count=0; self.failures=[]; self.maxima={}; self.hashes={}; self.elements=0

    def require(self, condition, name, detail=None):
        self.count+=1
        if not bool(condition):
            self.failures.append(dict(check=name,detail=native(detail)))

    def exact(self, actual, expected, name):
        self.require(native(actual)==native(expected),name)

    def close(self, actual, expected, field, context, atol=2e-12):
        a,b=np.asarray(actual),np.asarray(expected)
        okay=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error=float(np.max(abs(a-b),initial=0.)) if okay else None
        self.elements+=a.size
        if error is not None and error>self.maxima.get(field,{}).get("error",-1.):
            self.maxima[field]=dict(error=error,context=context)
        self.require(okay and error<=atol,context+":"+field,dict(error=error,atol=atol))
        return error

    def bind(self,path,expected=None,size=None):
        path=Path(path); name=rel(path); digest=sha(path)
        if name in self.hashes:
            self.exact(digest,self.hashes[name],name+":unchanged")
        self.hashes[name]=digest
        if expected is not None:self.exact(digest,expected,name+":sha256")
        if size is not None:self.exact(path.stat().st_size,size,name+":bytes")
        return digest


def complement(vector):
    """Independent Householder orthonormal basis of vector-perp."""
    u=np.asarray(vector)/np.linalg.norm(vector)
    v=u.copy(); v[0]-=1.
    if np.linalg.norm(v)<1e-14:
        return np.eye(len(v))[:,1:]
    return (np.eye(len(v))-2*np.outer(v,v)/(v@v))[:,1:]


def helmert(m):
    Q=np.zeros((m,m-1))
    for k in range(1,m):
        Q[:k,k-1]=1/math.sqrt(k*(k+1)); Q[k,k-1]=-k/math.sqrt(k*(k+1))
    return Q


def reconstruct(raw, checks, context):
    """Rebuild all aggregations from saved per-label primitive arrays."""
    q,a=np.asarray(raw["targets"]),np.asarray(raw["a"]); m=len(a)
    for key,value in raw.items():
        if np.asarray(value).dtype.kind in "fc":
            checks.require(np.isfinite(value).all(),context+":finite_"+key)
    def mix(key):
        arr=raw[key]
        return np.stack([np.sum(arr*weights.reshape((1,len(weights))+(1,)*(arr.ndim-2)),axis=1) for weights in q])
    out=dict(loss=mix("label_loss"),gradient=mix("label_gradient"),
             target_d_mass=mix("label_d_mass"))
    checks.close(raw["label_bias_gradient"],raw["label_gradient"][...,:m],"label_bias_gradient",context,0.)
    checks.close(raw["label_radial_gradient"],raw["label_gradient"][...,m:],"label_radial_gradient",context,0.)
    out["bias_gradient"]=out["gradient"][...,:m];out["radial_gradient"]=out["gradient"][...,m:]
    hessian="label_H_ba" in raw
    if hessian:
        labelH=raw["label_B"]-raw["label_same_particle_term"]+raw["label_cross_particle_term"]
        checks.close(raw["label_H_ba"],labelH,"label_H_ba_decomposition",context,0.)
        out["H_ba"]=mix("label_H_ba")
        M,sp=mix("label_M"),mix("label_sigma_prime_average")
        Hp=-M.copy();ix=np.arange(m);Hp[...,ix,ix]+=sp
        edges=M*a[None,None,:,None]*a[None,None,None,:]
        edges[...,ix,ix]=0.
        Ha=-edges.copy();Ha[...,ix,ix]=edges.sum(axis=-1)
        direct=np.diag(a)@Hp@np.diag(a)
        direct[...,ix,ix]-=a*out["radial_gradient"]
        out.update(Hperp=Hp,Hangle=Ha,edge_weights=edges,directWard=direct,ward_residual=direct-Ha,
                   ward_residual_max=np.max(abs(direct-Ha),axis=(-2,-1)),
                   H_ba_symmetry_residual=np.max(abs(out["H_ba"]-out["H_ba"].swapaxes(-1,-2)),axis=(-2,-1)),
                   common_angle_residual=np.max(abs(Ha.sum(axis=-1)),axis=-1),target_c_mass=mix("label_c_mass"))
        checks.close(raw["label_M"],raw["label_same_particle_term"][...,:m,:m],"same_particle_M_block",context)
        labHp=-raw["label_M"].copy();labHp[...,ix,ix]+=raw["label_sigma_prime_average"]
        labE=raw["label_M"]*a[None,None,:,None]*a[None,None,None,:];labE[...,ix,ix]=0.
        labHa=-labE.copy();labHa[...,ix,ix]=labE.sum(axis=-1)
        labW=np.diag(a)@labHp@np.diag(a);labW[...,ix,ix]-=a*raw["label_gradient"][...,m:]
        for key,value in dict(label_Hperp=labHp,label_Hangle=labHa,label_directWard=labW).items():
            checks.close(raw[key],value,key,context)
        Q=helmert(m); V=complement(a)
        symmetric=lambda H:(H+H.swapaxes(-1,-2))/2
        out["audit_H_ba_eigenvalues"]=np.linalg.eigvalsh(symmetric(out["H_ba"]))
        out["audit_Hangle_eigenvalues"]=np.linalg.eigvalsh(symmetric(Q.T@Ha@Q))
        out["audit_Hperp_eigenvalues"]=np.linalg.eigvalsh(symmetric(V.T@Hp@V))
        out["audit_Cartesian_Ward"]=np.max(abs(Hp@a-out["radial_gradient"]),axis=-1)
        out["audit_normalized_rotation_residual"]=np.max(abs(Hp@(a/np.linalg.norm(a))),axis=-1)
        out["audit_transverse_rotation_rayleigh"]=np.einsum("i,...ij,j->...",a,Hp,a)/(a@a)
    for key,value in out.items():
        if not key.startswith("audit_"):
            checks.close(raw[key],value,"aggregate."+key,context)
    mass=math.fsum(float(v) for v in raw["normal_weights"])
    checks.close(raw["normal_quadrature_mass"],mass,"normal_mass",context,0.)
    checks.close(raw["label_d_mass_error"],raw["label_d_mass"]-1,"unit_d_mass_error",context,0.)
    checks.close(raw["label_d_mass_quadrature_error"],raw["label_d_mass"]-raw["label_d_mass_expected"],"raw_d_mass_error",context,0.)
    if hessian:
        checks.close(raw["label_raw_c_mass_upper_excess"],raw["label_c_mass"]-raw["raw_c_mass_upper"][:,None],"raw_c_mass_error",context,0.)
    return out


def check_newton_step(step,calls,checks,context):
    """Replay arithmetic/acceptance from immutable calls, without optimizing."""
    current=calls[step["current_call_id"]]
    theta=np.asarray(current["theta"]); g=np.asarray(current["gradient"])
    H=np.asarray(current["H_ba"]); sym=(H+H.T)/2
    eig=np.linalg.eigvalsh(sym); mu=max(0.,1e-4-float(eig[0]))
    direction=-np.linalg.solve(sym+mu*np.eye(len(g)),g)
    cap=min(1.,.25/float(np.max(abs(direction)))) if np.any(direction) else 1.
    d=cap*direction
    lo=np.r_[np.full(8,-2.),np.full(8,.05)]
    hi=np.full(16,2.)
    bounds=[]
    for k,v in enumerate(d):
        if v>0:bounds.append((hi[k]-theta[k])/v)
        elif v<0:bounds.append((lo[k]-theta[k])/v)
    maxalpha=min(bounds) if bounds else math.inf
    initial=min(1.,.99*maxalpha)
    expected=dict(raw_H=H,sym_H=sym,skew_max=np.max(abs(H-H.T)),eigenvalues=eig,mu=mu,
        raw_direction=direction,capped_direction=d,cap_scale=cap,box_max_alpha=maxalpha,
        initial_alpha=initial,g_dot_d=g@d)
    for key,value in expected.items():
        checks.close(step[key],value,"solver."+key,context,2e-10 if key in ("raw_direction","box_max_alpha") else 2e-12)
    checks.require(0<=len(step["trials"])<=13,context+":fixed_backtracking_budget")
    accepted=None
    for trial_number,trial in enumerate(step["trials"]):
        call=calls[trial["call_id"]]
        alpha=initial/(2**trial_number)
        trialtheta=theta+alpha*d
        rhs=float(current["loss"])+1e-4*alpha*float(g@d)+5e-13
        for key,value in dict(alpha=alpha,theta=trialtheta,loss=call["loss"],armijo_rhs=rhs).items():
            checks.close(trial[key],value,"solver.trial_"+key,context)
        checks.exact(trial["halvings"],trial_number,context+":halving_sequence")
        checks.close(call["theta"],trialtheta,"solver.trial_call_theta",context)
        checks.exact(call["role"],"trial",context+":trial_role")
        checks.exact(call["compute_hessian"],False,context+":trial_FG_only")
        checks.require(np.all(trialtheta>=lo) and np.all(trialtheta<=hi),context+":trial_feasible")
        isaccepted=float(call["loss"])<=rhs
        checks.exact(trial["accepted"],isaccepted,context+":armijo_acceptance")
        if isaccepted:
            checks.exact(trial_number,len(step["trials"])-1,context+":no_trial_after_acceptance")
            accepted=call
    return dict(current_call_id=step["current_call_id"],accepted_call_id=None if accepted is None else accepted["call_id"],
                trial_count=len(step["trials"]),mu=mu,cap_scale=cap)


def box(theta):
    x=np.asarray(theta); lo=np.r_[np.full(8,-2.),np.full(8,.05)];hi=np.full(16,2.)
    margin=np.minimum(x-lo,hi-x)
    return dict(feasible=bool(np.isfinite(x).all() and np.all(margin>=0)),
                minimum_margin=float(margin.min()),coordinate_margins=margin)


def load_raw(path,checks,theta,target_names,targets,rule,resources):
    checks.bind(path)
    with np.load(path,allow_pickle=False) as f:raw={key:f[key] for key in f.files}
    context=rel(path);theta=np.asarray(theta);hess="H_ba" in raw
    for key,value in dict(theta=theta,b=theta[:8],a=theta[8:],targets=targets,target_names=target_names).items():
        checks.exact(raw[key],value,context+":input_"+key)
    labels=np.array(list(itertools.product((0,1),repeat=8)),dtype=np.int8)
    checks.exact(raw["patterns"],labels,context+":patterns")
    order,limit,lo,hi,step=RULES[rule]
    checks.exact(len(raw["normal_nodes"]),order,context+":normal_order")
    checks.require(np.all(abs(raw["normal_nodes"])<=limit) and np.all(raw["normal_weights"]>=0),context+":normal_rule_support")
    nodes=np.linspace(lo,hi,round((hi-lo)/step)+1);weights=np.full(len(nodes),step);weights[[0,-1]]/=2
    checks.close(raw["outer_logu_nodes"],nodes,"outer_nodes",context,0.)
    checks.close(raw["outer_weights"],weights,"outer_weights",context,0.)
    M=math.fsum(float(v) for v in raw["normal_weights"])
    mass=np.array([1. if k=="inf" else math.exp(int(k)*math.log(M)) for k in raw["K_names"]])
    checks.close(raw["raw_bank_quadrature_mass"],mass,"bank_mass",context,0.)
    checks.close(raw["label_d_mass_expected"],np.broadcast_to(mass[:,None],raw["label_loss"].shape),"expected_d_mass",context,0.)
    checks.close(raw["log_likelihood_scale"],raw["log_evidence"]-math.log(M),"likelihood_scale",context,0.)
    if hess:
        upper=np.array([1. if k=="inf" else 1-1/int(k) for k in raw["K_names"]])
        checks.close(raw["c_mass_upper"],upper,"unit_c_bound",context,0.)
        checks.close(raw["raw_c_mass_upper"],upper*mass,"raw_c_bound",context,0.)
    rebuilt=reconstruct(raw,checks,context)
    resources.append(dict(path=context,full_Hessian=hess,targets=len(target_names),K_count=len(raw["K_names"]),
                          normal_nodes=order,outer_nodes=len(nodes),finite_K=sum(k!="inf" and int(k)>1 for k in raw["K_names"])))
    return raw,rebuilt


def check_solver(source,endpoint,checks,resources):
    name=source["name"];folder=ART/"optimization"/name
    path=folder/"SOLVER.json";checks.bind(path,endpoint["solver_artifact"]["sha256"])
    data=read(path);context="solver/"+name
    checks.exact(data["status"],"completed",context+":status")
    checks.exact(data["initial_theta"],source["theta0"],context+":initial")
    checks.exact(data["fit_target"],source["target"],context+":target")
    calls={c["call_id"]:c for c in data["calls"]}
    checks.exact(list(calls),list(range(len(calls))),context+":call_ids")
    checks.exact(len(data["attempted_calls"]),len(calls),context+":attempted_count")
    for attempt,c in zip(data["attempted_calls"],data["calls"]):
        checks.exact(attempt["status"],"completed",context+":attempt_status")
        for field in ("call_id","phase","role","phase_call_index","theta","compute_hessian"):
            checks.exact(attempt[field],c[field],context+":attempt_"+field)
    for item,c in zip(data["call_artifacts"],data["calls"]):
        checks.exact(item["call_id"],c["call_id"],context+":artifact_call_id")
        for field in ("array_artifact","summary_artifact"):
            a=item[field];checks.bind(ROOT/a["path"],a["sha256"],a["bytes"])
        summary=read(ROOT/item["summary_artifact"]["path"])
        for field in ("call_id","phase","role","phase_call_index","theta","loss","gradient","H_ba","compute_hessian"):
            checks.exact(summary[field],c[field],context+":independent_snapshot_"+field)
        raw,_=load_raw(ROOT/item["array_artifact"]["path"],checks,c["theta"],[source["target_name"]],
                       [source["target"]],c["phase"],resources)
        checks.exact(raw["K_names"],["8"],context+":fitted_K")
        for field in ("loss","gradient"):
            checks.exact(c[field],raw[field][0,0],context+":call_raw_"+field)
        checks.exact(c["H_ba"],raw["H_ba"][0,0] if c["compute_hessian"] else None,context+":call_H")
        checks.exact(c["box"],box(c["theta"]),context+":call_box")
    checks.exact(len(data["call_artifacts"]),len(calls),context+":artifact_count")
    steps={s["current_call_id"]:s for s in data["steps"]}
    checks.exact(len(steps),len(data["steps"]),context+":unique_step_current")
    checks.exact([s["step_id"] for s in data["steps"]],list(range(len(steps))),context+":step_ids")
    step_records=[check_newton_step(s,calls,checks,context+f"/step{s['step_id']}") for s in data["steps"]]
    last=np.asarray(source["theta0"]);accepted_ids=[];stop_records=[]
    for stage,(phase,maxsteps,maxcalls,gtol) in zip(data["stages"],STAGES):
        for field,value in dict(phase=phase,max_calls=maxcalls,max_accepted_steps=maxsteps,gradient_tolerance=gtol).items():
            checks.exact(stage[field],value,context+":stage_"+field)
        checks.exact(stage["start_theta"],last,context+":stage_start")
        phasecalls=[c for c in data["calls"] if c["phase"]==phase]
        checks.exact([c["phase_call_index"] for c in phasecalls],list(range(1,len(phasecalls)+1)),context+":phase_call_sequence")
        checks.require(len(phasecalls)<=maxcalls,context+":call_budget")
        pos=0;naccepted=0;reason=None
        while reason is None:
            if naccepted>=maxsteps:reason="accepted_step_budget";break
            if pos>=maxcalls:reason="function_call_budget";break
            checks.require(pos<len(phasecalls),context+":expected_current_call")
            if pos>=len(phasecalls):reason="missing_current_call";break
            c=phasecalls[pos];pos+=1
            checks.exact(c["role"],"current",context+":current_role")
            checks.exact(c["compute_hessian"],True,context+":current_H")
            checks.exact(c["theta"],last,context+":current_LAST")
            if np.max(abs(np.asarray(c["gradient"])))<=gtol:
                reason="gradient_tolerance";break
            step=steps[c["call_id"]]
            checks.exact(step["phase"],phase,context+":step_phase")
            checks.exact(step["phase_step_index"],naccepted,context+":phase_step_index")
            for field in ("theta","loss","gradient"):
                checks.exact(step[field],c[field],context+":step_snapshot_"+field)
            if step["initial_alpha"]<=0:reason="no_positive_feasible_box_step";break
            accepted=False
            for trial in step["trials"]:
                checks.require(pos<len(phasecalls),context+":expected_trial_call")
                if pos>=len(phasecalls):break
                tc=phasecalls[pos];pos+=1
                checks.exact(tc["call_id"],trial["call_id"],context+":trial_order")
                for field in ("theta","loss","gradient","alpha","halvings"):
                    checks.exact(trial[field],tc[field],context+":trial_snapshot_"+field)
                if trial["accepted"]:
                    accepted=True;naccepted+=1;last=np.asarray(tc["theta"]);accepted_ids.append(tc["call_id"])
                    if np.max(abs(np.asarray(tc["gradient"])))<=gtol:reason="gradient_tolerance_at_accepted_FG"
                    break
            if not accepted:
                reason="function_call_budget" if pos>=maxcalls else "line_search_failed"
                if reason=="line_search_failed":checks.exact(len(step["trials"]),13,context+":exhausted_backtracking")
        checks.exact(pos,len(phasecalls),context+":all_phase_calls_accounted")
        checks.exact(stage["stop_reason"],reason,context+":stop_reason")
        checks.exact(stage["call_count"],len(phasecalls),context+":phase_count")
        checks.exact(stage["accepted_step_count"],naccepted,context+":accepted_count")
        checks.exact(stage["final_theta"],last,context+":stage_final")
        checks.exact(stage["final_box"],box(last),context+":stage_final_box")
        stop_records.append(dict(phase=phase,stop_reason=reason,calls=pos,accepted_steps=naccepted))
    checks.exact(len(data["stages"]),2,context+":two_stages")
    checks.exact(data["final_theta"],last,context+":LAST")
    checks.exact(endpoint["theta"],last,context+":endpoint_lock_LAST")
    checks.exact(data["total_calls"],len(calls),context+":total_calls")
    checks.exact(data["accepted_step_count"],len(accepted_ids),context+":total_accepted")
    checks.exact([s["call_id"] for s in data["accepted_states"]],[None]+accepted_ids,context+":accepted_call_ids")
    for state_id,state in enumerate(data["accepted_states"]):
        checks.exact(state["state_id"],state_id,context+":accepted_state_id")
        expected=source["theta0"] if state_id==0 else calls[state["call_id"]]["theta"]
        checks.exact(state["theta"],expected,context+":accepted_snapshot")
        if state_id:
            c=calls[state["call_id"]]
            for field in ("phase","loss","gradient","box"):
                checks.exact(state[field],c[field],context+":accepted_snapshot_"+field)
    tp=ROOT/data["tape_artifact"]["path"];checks.bind(tp,data["tape_artifact"]["sha256"],data["tape_artifact"]["bytes"])
    with np.load(tp,allow_pickle=False) as f:tape={k:f[k] for k in f.files}
    for field,sourcekey in dict(call_id="call_id",call_theta="theta",call_loss="loss",call_gradient="gradient",
        compute_hessian="compute_hessian",phase="phase",role="role").items():
        checks.exact(tape[field],[c[sourcekey] for c in data["calls"]],context+":tape_"+field)
    for index,c in enumerate(data["calls"]):
        if c["compute_hessian"]:checks.exact(tape["call_H_ba"][index],c["H_ba"],context+":tape_H")
        else:checks.require(np.isnan(tape["call_H_ba"][index]).all(),context+":missing_H_mask")
    checks.exact(tape["accepted_theta"],[s["theta"] for s in data["accepted_states"]],context+":tape_accepted_theta")
    checks.exact(tape["accepted_call_id"],[-1]+accepted_ids,context+":tape_accepted_call_id")
    checks.exact(endpoint["solver_stage_stop_reasons"],[s["stop_reason"] for s in data["stages"]],context+":locked_stops")
    return dict(name=name,stages=stop_records,calls=len(calls),accepted_steps=len(accepted_ids),steps=step_records)


def scalar_diagnostics(raw,ki):
    expected=float(raw["raw_bank_quadrature_mass"][ki]);d=raw["label_d_mass"][ki];c=raw["label_c_mass"][ki]
    upper=float(raw["raw_c_mass_upper"][ki]);M=float(raw["normal_quadrature_mass"])
    return dict(normal_mass=M,normal_mass_error=abs(M-1.),expected_raw_d_mass=expected,
        d_relative_mass_error=float(np.max(abs(d/expected-1))),d_raw_min=float(d.min()),d_raw_max=float(d.max()),
        c_raw_lower_violation=max(0.,float(-c.min())),c_raw_upper_violation=max(0.,float(c.max()-upper)),
        c_raw_min=float(c.min()),c_raw_max=float(c.max()),c_raw_upper=upper,
        target_weights_mass=raw["targets"].sum(axis=1),scope="all labels")


def check_probability(saved,raw,teacher,checks,context):
    pmf=np.exp(raw["log_evidence"]);q=np.asarray(teacher);positive=q>0
    kl=float(q[positive]@(np.log(q[positive])-raw["log_evidence"][positive]))
    expected=dict(pmf=pmf,pmf_mass=pmf.sum(),pmf_mass_error=abs(pmf.sum()-1),teacher_mass=q.sum(),raw_KL=kl)
    for key,value in expected.items():checks.close(saved[key],value,"probability."+key,context)
    checks.exact(saved["model_or_target_renormalized"],False,context+":unmodified_probability_mass")
    return expected


def check_final_summary(summary,raw,rebuilt,teacher,checks,context):
    a=raw["a"];V=np.asarray(raw["transverse_basis"]);Q=helmert(8)
    u=a/np.linalg.norm(a);v=u.copy();v[0]+=1 if u[0]>=0 else -1
    fixedV=(np.eye(8)-2*np.outer(v,v)/(v@v))[:,1:]
    checks.close(V,fixedV,"fixed_transverse_basis",context,0.)
    checks.close(V.T@V,np.eye(7),"transverse_basis_orthonormal",context)
    checks.close(V.T@a,np.zeros(7),"transverse_basis_orthogonal_to_a",context)
    checks.close(raw["helmert_basis"],Q,"helmert_basis",context,0.)
    for name,H in (("H_ba",rebuilt["H_ba"]),("Hperp_projected",V.T@rebuilt["Hperp"]@V),
                   ("Hangle_projected",Q.T@rebuilt["Hangle"]@Q)):
        checks.close(raw[name],H,"projection."+name,context)
        S=(H+H.swapaxes(-1,-2))/2
        vals=np.linalg.eigvalsh(S);vec=raw[name+"_eigenvectors"]
        checks.close(raw[name+"_eigenvalues"],vals,"spectrum."+name,context)
        checks.close(S@vec,vec*raw[name+"_eigenvalues"][...,None,:],"eigenpair."+name,context)
        checks.close(vec.swapaxes(-1,-2)@vec,np.broadcast_to(np.eye(H.shape[-1]),vec.shape),"orthonormal."+name,context)
    checks.close(raw["Hperp_projected_eigenvalues"],rebuilt["audit_Hperp_eigenvalues"],"independent_basis_spectrum",context)
    for field in ("theta","target_names","transverse_basis","helmert_basis"):
        checks.exact(summary[field],raw[field],context+":summary_"+field)
    checks.exact([str(x) for x in summary["K_values"]],raw["K_names"],context+":K_values")
    checks.exact(len(summary["blocks"]),len(raw["targets"])*3,context+":block_count")
    for n,row in enumerate(summary["blocks"]):
        ti,ki=divmod(n,3)
        checks.exact((row["target"],str(row["K"])),(raw["target_names"][ti],raw["K_names"][ki]),context+":block_identity")
        for key in ("loss","gradient","bias_gradient","radial_gradient","H_ba","Hperp","Hangle","directWard",
            "ward_residual","H_ba_eigenvalues","H_ba_eigenvectors","Hperp_projected","Hperp_projected_eigenvalues",
            "Hperp_projected_eigenvectors","Hangle_projected","Hangle_projected_eigenvalues","Hangle_projected_eigenvectors"):
            checks.exact(row[key],raw[key][ti,ki],context+":json_npz_"+key)
        for key,value in dict(gradient_Linf=np.max(abs(rebuilt["gradient"][ti,ki])),
            raw_H_ba_symmetry_residual=rebuilt["H_ba_symmetry_residual"][ti,ki],Ward_Linf=rebuilt["ward_residual_max"][ti,ki],
            Cartesian_Ward_Linf=rebuilt["audit_Cartesian_Ward"][ti,ki],
            near_rotation_residual_Linf=rebuilt["audit_normalized_rotation_residual"][ti,ki],
            structural_common_angle_residual=rebuilt["common_angle_residual"][ti,ki]).items():
            checks.close(row[key],value,"summary."+key,context)
        checks.exact(row["scalar_diagnostics"],scalar_diagnostics(raw,ki),context+":scalar_diagnostics")
    return check_probability(summary["probability"],raw,teacher,checks,context)


def check_qualifications(saved,C,D,checks,context):
    rows=[]
    checks.exact(len(saved),len(C["targets"])*3,context+":qualification_count")
    for idx,row in enumerate(saved):
        ti,ki=divmod(idx,3)
        checks.exact((row["target"],str(row["K"])),(C["target_names"][ti],C["K_names"][ki]),context+":qualification_identity")
        comp={key:float(np.max(abs(C[key][ti,ki]-D[key][ti,ki]))) for key in
            ("loss","gradient","H_ba","Hperp","Hangle","H_ba_eigenvalues","Hperp_projected_eigenvalues","Hangle_projected_eigenvalues")}
        gates={"loss":comp["loss"]<=1e-7,"gradient":comp["gradient"]<=2e-7}
        for key in ("H_ba","Hperp","Hangle"):gates[key]=comp[key]<=2e-7
        for key in ("H_ba_eigenvalues","Hperp_projected_eigenvalues","Hangle_projected_eigenvalues"):gates[key]=comp[key]<=2e-6
        for rule,r in (("C",C),("D",D)):
            diag=scalar_diagnostics(r,ki)
            gates.update({rule+"_normal_mass":diag["normal_mass_error"]<=2e-12,
                rule+"_d_relative_mass":diag["d_relative_mass_error"]<=2e-7,
                rule+"_c_lower":diag["c_raw_lower_violation"]<=2e-7,
                rule+"_c_upper":diag["c_raw_upper_violation"]<=2e-7,
                rule+"_Ward":r["ward_residual_max"][ti,ki]<=2e-7})
            checks.exact(row["scalar_diagnostics"][rule],diag,context+":qualification_scalars")
        checks.exact(row["comparisons"],comp,context+":qualification_comparisons")
        checks.exact(row["gates"],gates,context+":qualification_gates")
        checks.exact(row["qualified"],all(gates.values()),context+":qualification_passed")
        checks.exact(row["failed_gates"],[k for k,v in gates.items() if not v],context+":qualification_failed_gates")
        rows.append(dict(target=row["target"],K=row["K"],qualified=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v]))
    return rows


def check_signature(saved,theta,D,numerical,ti,checks,context):
    a=np.asarray(theta)[8:];g=D["gradient"][ti,0];Hp=D["Hperp"][ti,0]
    values=dict(D_gradient_Linf=float(np.max(abs(g))),minimum_box_margin=box(theta)["minimum_margin"],
        D_H_ba_minimum=float(D["H_ba_eigenvalues"][ti,0,0]),
        D_Hperp_transverse_minimum=float(D["Hperp_projected_eigenvalues"][ti,0,0]),
        D_Cartesian_Ward_Linf=float(np.max(abs(Hp@a-g[8:]))),
        D_near_rotation_residual_Linf=float(np.max(abs(Hp@(a/np.linalg.norm(a))))))
    gates=dict(numerical_qualification=numerical["qualified"],gradient=values["D_gradient_Linf"]<=1e-8,
        box_margin=values["minimum_box_margin"]>=1e-3,H_ba_positive=values["D_H_ba_minimum"]>1e-5,
        Hperp_transverse_positive=values["D_Hperp_transverse_minimum"]>1e-5,
        Cartesian_Ward=values["D_Cartesian_Ward_Linf"]<=2e-7,near_rotation_mode=values["D_near_rotation_residual_Linf"]<=2e-7)
    for key,value in values.items():checks.close(saved[key],value,"signature."+key,context)
    checks.exact(saved["gates"],gates,context+":signature_gates")
    passed=all(gates.values())
    checks.exact(saved["qualified_full_local_minimum_signature"],passed,context+":signature_passed")
    expected="numerically_qualified_full_nonrotation_local_minimum_signature" if passed else "unresolved_full_local_minimum_signature"
    checks.exact(saved["classification"],expected,context+":signature_classification")
    checks.exact(saved["failed_gates"],[k for k,v in gates.items() if not v],context+":signature_failed_gates")
    checks.exact(saved["nonrotation_dimensions"],23,context+":dimensions")
    return dict(passed=passed,classification=expected,values=values,failed_gates=[k for k,v in gates.items() if not v])


def check_pair(folder,stem,theta,targets,teacher,row,checks,resources):
    raws={};prob={};target_names=list(targets)
    for rule in ("C","D"):
        npz=folder/(stem+"_"+rule+".npz");js=npz.with_suffix(".json")
        for key,path in (("array_artifact",npz),("summary_artifact",js)):
            artifact=row["rule_artifacts"][rule][key]
            checks.exact(artifact["path"],rel(path),stem+":artifact_path")
            checks.bind(path,artifact["sha256"],artifact["bytes"])
        raw,rebuilt=load_raw(npz,checks,theta,target_names,list(targets.values()),rule,resources)
        checks.exact(raw["K_names"],["8","64","inf"],stem+":control_K")
        summary=read(js)
        checks.exact(summary["array_artifact"],row["rule_artifacts"][rule]["array_artifact"],stem+":array_binding")
        prob[rule]=check_final_summary(summary,raw,rebuilt,teacher,checks,stem+"/"+rule)
        check_probability(row["probability"][rule],raw,teacher,checks,stem+"/"+rule)
        raws[rule]=raw
    qual=check_qualifications(row["qualifications"],raws["C"],raws["D"],checks,stem)
    return raws,qual,prob


def check_resources(items,saved,accepted,checks,context):
    counts=dict(successful_kernel_calls=len(items),failed_kernel_calls=0,
        full_Hessian_calls=sum(x["full_Hessian"] for x in items),FG_only_calls=sum(not x["full_Hessian"] for x in items),
        target_K_blocks=sum(x["targets"]*x["K_count"] for x in items),new_random_draws=0,new_observations=0,
        statistical_alpha_spent=0,accepted_optimizer_steps=accepted)
    factors={}
    for item in items:
        n,v,f,h=item["normal_nodes"],item["outer_nodes"],item["finite_K"],item["full_Hessian"]
        work=dict(targets=item["targets"],labels=256,normal_nodes=n,outer_nodes=v,
            finite_K_greater_than_one=f,bernoulli_logit_evaluations=n*8,conditional_loglikelihood_values=256*n,
            laplace_exp_minus_ux_factors=256*v*n if f else 0,finite_d_coefficients=256*n*f,
            finite_c_coefficients=256*n*f if h else 0,finite_cross_particle_outer_products=256*v*f if h else 0)
        summary=read((ROOT/item["path"]).with_suffix(".json"))
        for key,value in work.items():
            checks.exact(summary["work_counts"][key],value,context+":work_count_"+key)
            factors[key]=factors.get(key,0)+value
    counts["high_level_factor_totals"]=factors
    checks.exact(saved,counts,context+":resources")
    return counts


def perform():
    started=time.perf_counter();checks=Checks();resources=[]
    resultpath=ART/"RESULTS.json"
    if not resultpath.exists() or read(resultpath).get("status")!="completed":
        raise RuntimeError("The five solves and locked evaluation must finish before this audit")
    frozenpath=ART/"FREEZE.json";sourcepath=ART/"SOURCE.json";afpath=ART/"AUDIT_FREEZE.json"
    fh=checks.bind(frozenpath);sh=checks.bind(sourcepath);afh=checks.bind(afpath)
    frozen,source,af=read(frozenpath),read(sourcepath),read(afpath)
    checks.exact(af["scientific_freeze_sha256"],fh,"audit_freeze:scientific_freeze")
    checks.exact(af["sources"].get(rel(__file__)),sha(__file__),"audit_freeze:source")
    checks.exact(frozen["source_sha256"],sh,"scientific_freeze:source")
    for mapping in (af["sources"],frozen["sources"],source["sources"]):
        for path,digest in mapping.items():checks.bind(ROOT/path,digest)
    checks.exact([s["name"] for s in source["solves"]],SOLVE_NAMES,"source:five_ordered_solves")
    for rep,s in enumerate(source["solves"][:4]):
        folder=ROOT/"artifacts/joint_dependence_v1_20261007"/f"rep_{rep:02d}"
        old=read(folder/"CANDIDATES_LOCK.json");qualified=read(folder/"ENDPOINT_QUALIFICATION.json")
        candidate=next(c for c in old["candidates"] if c["name"]=="K8")
        b,W=np.asarray(candidate["b"]),np.asarray(candidate["W"])
        checks.exact(candidate["theta"],np.r_[b,W.ravel()],f"source/{rep}:redundant_theta")
        checks.exact(s["theta0"],np.r_[b,np.sqrt(np.sum(W*W,axis=1))],f"source/{rep}:norm_anchor")
        checks.exact(s["target"],old["frequency"],f"source/{rep}:empirical_original")
        checks.exact(s["target_name"],"empirical",f"source/{rep}:target_name")
        checks.exact(source["teacher_pmf"],qualified["teacher_pmf"],f"source/{rep}:teacher_original")
    checks.exact(source["solves"][4]["theta0"],source["solves"][0]["theta0"],"source:P_start")
    checks.exact(source["solves"][4]["target"],source["teacher_pmf"],"source:P_target")
    checks.exact(source["solves"][4]["target_name"],"population","source:P_target_name")
    lockpath=ART/"ENDPOINTS_LOCK.json";lh=checks.bind(lockpath);lock=read(lockpath)
    result=read(resultpath);checks.bind(resultpath)
    checks.exact(lock["all_five_locked_before_control_evaluations"],True,"lock:before_controls")
    checks.exact(lock["teacher_risk_used_to_select_endpoints"],False,"lock:no_risk_selection")
    checks.exact([e["name"] for e in lock["endpoints"]],SOLVE_NAMES,"lock:all_five")
    for obj in (lock,result):
        checks.exact(obj["freeze_sha256"],fh,"saved:freeze_hash")
        checks.exact(obj["source_sha256"],sh,"saved:source_hash")
    checks.exact(result["endpoint_lock_sha256"],lh,"saved:endpoint_lock_hash")
    for mapping in (lock["optimization_artifacts"],result["artifacts"]):
        for path,digest in mapping.items():checks.bind(ROOT/path,digest)
    runstart=read(ART/"evaluation/RUN_START.json")
    checks.exact(runstart["endpoint_lock_sha256"],lh,"evaluation:locked_first")
    solver_records=[]
    for s,e in zip(source["solves"],lock["endpoints"]):
        checks.exact(e["fit_target_name"],s["target_name"],s["name"]+":locked_target_name")
        checks.exact(e["fit_target"],s["target"],s["name"]+":locked_target")
        begin=len(resources)
        record=check_solver(s,e,checks,resources);solver_records.append(record)
        solver=read(ROOT/e["solver_artifact"]["path"])
        checks.exact(solver["freeze_sha256"],fh,s["name"]+":solver_freeze")
        checks.exact(solver["source_sha256"],sh,s["name"]+":solver_source")
        check_resources(resources[begin:],solver["resources"],record["accepted_steps"],checks,s["name"])
    optimization_count=len(resources)
    check_resources(resources,lock["resources"],sum(r["accepted_steps"] for r in solver_records),checks,"optimization_total")
    checks.exact(result["optimization_resources"],lock["resources"],"result:optimization_resources")
    endpoint_records=[];independent=[];Praw=None;Psignature=None
    for endpoint,row in zip(lock["endpoints"],result["endpoints"]):
        name=endpoint["name"];theta=np.asarray(endpoint["theta"])
        checks.exact(row["name"],name,"endpoint:order")
        checks.exact(row["theta"],theta,name+":unchanged_endpoint")
        checks.exact(row,read(ART/"evaluation"/(name+"_QUALIFICATION.json")),name+":qualification_snapshot")
        targets={"population":source["teacher_pmf"]} if name=="P" else {"empirical":endpoint["fit_target"],"population":source["teacher_pmf"]}
        raws,qual,_=check_pair(ART/"evaluation",name,theta,targets,source["teacher_pmf"],row,checks,resources)
        fit_ti=list(targets).index(endpoint["fit_target_name"])
        signature=check_signature(row["fitted_signature"],theta,raws["D"],qual[3*fit_ti],fit_ti,checks,name)
        endpoint_records.append(dict(name=name,qualifications=qual,signature=signature))
        for yi in LABEL_INDICES:
            values,info=independent_label(theta[:8],theta[8:],yi)
            info.update(endpoint=name,values=values,errors={})
            checks.require(info["quad_success"],name+f"/label{yi}:quad_success",info)
            for key,value in values.items():
                info["errors"][key]=checks.close(value,raws["D"][key][0,yi],"independent."+key,
                                                  name+f"/label{yi}",QUAD_SPEC["absolute_comparison_tolerance"])
            independent.append(info)
        if name=="P":Praw=raws["D"];Psignature=signature
        print(json.dumps(dict(audited_endpoint=name,fit_signature_passed=signature["passed"])),flush=True)
    checks.exact(len(result["endpoints"]),5,"result:five_endpoints")
    checks.exact(result["endpoint_blocks"],27,"result:27_blocks")
    checks.exact(result["endpoint_rule_blocks"],54,"result:54_rule_blocks")
    checks.exact(result["qualified_endpoint_blocks"],sum(q["qualified"] for e in endpoint_records for q in e["qualifications"]),"result:qualified_blocks")
    checks.exact(result["fit_signatures_passed"],sum(e["signature"]["passed"] for e in endpoint_records),"result:fit_signatures")
    probe=result["probe"];planpath=ART/"evaluation/probe/PROBE_PLAN.json";plan=read(planpath)
    checks.bind(planpath,probe["plan_artifact"]["sha256"],probe["plan_artifact"]["bytes"])
    g=Praw["gradient"][0,2];norm=float(np.linalg.norm(g));theta=np.asarray(lock["endpoints"][4]["theta"])
    expected_status="skipped_P_fit_signature_not_qualified" if not Psignature["passed"] else (
        "undefined_integrated_gradient_too_small" if norm<=1e-10 else "defined")
    checks.exact(plan["status"],expected_status,"probe:eligibility")
    probe_records=[]
    if expected_status=="defined":
        direction=-g/norm
        checks.exact(plan["integrated_gradient"],g,"probe:integrated_gradient")
        checks.exact(plan["gradient_L2"],norm,"probe:gradient_norm")
        checks.exact(plan["direction"],direction,"probe:direction")
        checks.exact(probe["direction"],direction,"probe:result_direction")
        checks.exact(plan["direction_locked_before_probe_evaluation"],True,"probe:direction_locked")
        checks.exact(len(plan["points"]),5,"probe:five_points")
        checks.exact(len(probe["points"]),5,"probe:five_results")
        for idx,(t,pp,row) in enumerate(zip((-.02,-.01,0.,.01,.02),plan["points"],probe["points"])):
            pt=theta+t*direction;feasible=box(pt)["feasible"]
            for obj in (pp,row):
                checks.exact(obj["index"],idx,"probe:point_index")
                checks.exact(obj["t"],t,"probe:fixed_t")
                checks.exact(obj["theta"],pt,"probe:unmodified_point")
                checks.exact(obj["box"],box(pt),"probe:point_box")
            checks.exact(row["step_reduced_or_projected"],False,"probe:no_projection")
            if not feasible:
                checks.exact(row["status"],"infeasible_not_evaluated","probe:infeasible_retained")
                continue
            stem=f"point_{idx:02d}";folder=ART/"evaluation/probe"
            checks.exact(row,read(folder/(stem+"_QUALIFICATION.json")),stem+":qualification_snapshot")
            _,qual,prob=check_pair(folder,stem,pt,{"population":source["teacher_pmf"]},source["teacher_pmf"],row,checks,resources)
            comparisons=dict(PMF_max_difference=float(np.max(abs(prob["C"]["pmf"]-prob["D"]["pmf"]))),
                C_PMF_mass_error=prob["C"]["pmf_mass_error"],D_PMF_mass_error=prob["D"]["pmf_mass_error"],
                KL_difference=abs(prob["C"]["raw_KL"]-prob["D"]["raw_KL"]))
            gates=dict(PMF_difference=comparisons["PMF_max_difference"]<=2e-7,
                C_PMF_mass=comparisons["C_PMF_mass_error"]<=2e-12,D_PMF_mass=comparisons["D_PMF_mass_error"]<=2e-12,
                KL_difference=comparisons["KL_difference"]<=1e-7)
            checks.exact(row["PMF_KL_comparisons"],comparisons,stem+":PMF_KL_comparisons")
            checks.exact(row["PMF_KL_gates"],gates,stem+":PMF_KL_gates")
            checks.exact(row["PMF_KL_failed_gates"],[k for k,v in gates.items() if not v],stem+":PMF_KL_failed_gates")
            qualified=all(gates.values()) and all(q["qualified"] for q in qual)
            checks.exact(row["qualified"],qualified,stem+":qualified")
            probe_records.append(dict(index=idx,t=t,qualified=qualified,qualifications=qual,PMF_KL_gates=gates))
    else:
        checks.exact(probe["points"],[],"probe:no_undefined_points")
        checks.exact(plan["evaluated"],False,"probe:not_evaluated")
    check_resources(resources[optimization_count:],result["evaluation_resources"],0,checks,"evaluation_total")
    for path,digest in list(checks.hashes.items()):checks.bind(ROOT/path,digest)
    return dict(passed=not checks.failures,source_sha256=sha(__file__),freeze_sha256=fh,audit_freeze_sha256=afh,
        input_hashes=checks.hashes,failures=checks.failures,checks=checks.count,numeric_elements_compared=checks.elements,
        numerical_maxima=checks.maxima,solver_checks=solver_records,endpoint_checks=endpoint_records,
        independent_label_checks=independent,probe_checks=probe_records,
        scope=dict(independent_numerical_execution_audit=True,scientific_peer_review=False,
            all_saved_call_and_final_arrays_reconstructed=len(resources),independent_endpoint_label_K8_cells=len(independent),
            fixed_labels=LABEL_INDICES,inner_rule="same prescribed D384/T11 rebuilt with numpy leggauss",
            different_outer_integrator=QUAD_SPEC,all_optimizer_updates_replayed=False,
            solver_arithmetic_and_acceptance_replayed=True,full_independent_inner_outer_recalculation=False,
            rigorous_Gaussian_or_Laplace_remainder_certificate=False,scientific_failures_retained=True,
            new_observations=0,new_random_draws=0,new_optimizer_updates=0,new_statistical_alpha=0),
        seconds=time.perf_counter()-started,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}))


def normal_rule(order=384, limit=11.):
    x, w = np.polynomial.legendre.leggauss(order)
    z = limit * x
    return z, limit * w * np.exp(-z*z/2) / math.sqrt(2*math.pi)


def label_features(b, a, y, z):
    logits = np.asarray(b)[None, :] + z[:, None] * np.asarray(a)[None, :]
    logp, logno = -np.logaddexp(0., -logits), -np.logaddexp(0., logits)
    p, no = np.exp(logp), np.exp(logno)
    r = np.asarray(y) - p
    A = np.exp(np.sum(np.where(np.asarray(y)[None, :] == 1, logp, logno), axis=1))
    G = np.concatenate([r, z[:, None]*r], axis=1)
    sp = p * no
    Bd = np.concatenate([sp, z[:, None]*sp, z[:, None]**2*sp], axis=1)
    return A, G, Bd


def expand_B(values, m):
    B = np.zeros((2*m, 2*m)); ix = np.arange(m)
    B[ix, ix] = values[:m]
    B[ix, m+ix] = B[m+ix, ix] = values[m:2*m]
    B[m+ix, m+ix] = values[2*m:]
    return B


def independent_label(b, a, label_index, K=8, *, order=384, limit=11.,
                      logu_min=-32., logu_max=64.):
    """Recompute one label, including the distinct-particle Hessian term.

    For G=grad log A, the distinct term integrates
    K(K-1) u^2 Psi^(K-2) [E(x exp(-ux)G)][E(x exp(-ux)G)]^T dv.
    It uses no c/d coefficients or Hessians supplied by the study code.
    """
    b, a = np.asarray(b), np.asarray(a); m = len(b); ntheta = 2*m
    y = np.array([(label_index >> j) & 1 for j in range(m-1, -1, -1)])
    z, w = normal_rule(order, limit)
    mass = math.fsum(float(t) for t in w)
    A, G, Bd = label_features(b, a, y, z)
    evidence = float(w @ A)
    scale = evidence / mass
    x = A / scale
    gg = (G[:, :, None]*G[:, None, :]).reshape(order, ntheta*ntheta)
    # Inner features are evaluated directly against positive weighted likelihoods.
    features = np.column_stack([G, Bd, gg, np.ones(order)])
    logmass = math.log(mass); bankmass = math.exp(K*logmass)

    def integrand(v):
        u = math.exp(v)
        ux = u*x
        ef = np.exp(-ux)
        deficit = float(w @ np.expm1(-ux)) / mass
        if abs(deficit) < .25:
            logpsi = logmass + math.log1p(deficit)
        else:
            psi = float(w @ ef)
            logpsi = math.log(psi) if psi > 0 else -math.inf
        wx = w*x*ef
        raw = wx @ features
        BG = raw[:ntheta]
        BB = raw[ntheta:ntheta+3*m]
        BGG = raw[ntheta+3*m:-1]
        B0 = raw[-1]
        d_factor = K*u*math.exp((K-1)*logpsi)
        c_factor = K*(K-1)*u*u*(1. if K == 2 else math.exp((K-2)*logpsi))
        left = K*logpsi; right = K*logmass-K*u
        # Stable signed difference of two positive exponentials.
        delta = left-right
        loss = 0. if delta == 0 else math.copysign(math.exp(max(left,right)) * -math.expm1(-abs(delta)), delta)
        return np.r_[loss, -d_factor*BG, d_factor*BB,
                     c_factor*B0*BGG, c_factor*np.outer(BG,BG).ravel(),
                     d_factor*B0, c_factor*B0*B0]

    points = [v for v in QUAD_SPEC["breakpoints"] if logu_min < v < logu_max]
    value, error, info = quad_vec(integrand, logu_min, logu_max, points=points,
        epsabs=QUAD_SPEC["epsabs"], epsrel=QUAD_SPEC["epsrel"], norm="max", full_output=True)
    j = 1
    grad = value[j:j+ntheta]; j += ntheta
    B = expand_B(value[j:j+3*m], m); j += 3*m
    same = value[j:j+ntheta*ntheta].reshape(ntheta,ntheta); j += ntheta*ntheta
    cross = value[j:j+ntheta*ntheta].reshape(ntheta,ntheta); j += ntheta*ntheta
    output = dict(label_loss=float(-bankmass*math.log(scale)+value[0]), label_gradient=grad,
        label_B=B, label_same_particle_term=same, label_cross_particle_term=cross,
        label_H_ba=B-same+cross, label_M=same[:m,:m],
        label_sigma_prime_average=np.diag(B)[:m], label_d_mass=float(value[j]),
        label_c_mass=float(value[j+1]))
    diagnostic = dict(label_index=label_index, K=K, normal_mass=mass, raw_bank_mass=bankmass,
        likelihood_scale=scale, evidence=evidence, quad_success=bool(info.success),
        quad_status=int(info.status), quad_message=str(info.message), estimated_error=float(error),
        integrand_evaluations=int(info.neval), adaptive_intervals=len(info.intervals))
    return output, diagnostic


def manufactured_pair_reference(b, a, label_index, order=64, limit=7.):
    """Direct two-node-bank enumeration, independent of any Laplace identity."""
    m = len(b); z,w = normal_rule(order,limit)
    y = np.array([(label_index >> j)&1 for j in range(m-1,-1,-1)])
    A,G,Bd = label_features(b,a,y,z)
    B = np.array([expand_B(row,m) for row in Bd])
    output = dict(label_loss=0., label_gradient=np.zeros(2*m), label_B=np.zeros((2*m,2*m)),
        label_same_particle_term=np.zeros((2*m,2*m)), label_cross_particle_term=np.zeros((2*m,2*m)),
        label_d_mass=0., label_c_mass=0.)
    for i in range(order):
        for j in range(order):
            p0 = A[i]/(A[i]+A[j]); p1=1-p0; weight=w[i]*w[j]
            output["label_loss"] -= weight*math.log((A[i]+A[j])/2)
            output["label_gradient"] -= weight*(p0*G[i]+p1*G[j])
            output["label_B"] += weight*(p0*B[i]+p1*B[j])
            output["label_same_particle_term"] += weight*p0*p1*(np.outer(G[i],G[i])+np.outer(G[j],G[j]))
            output["label_cross_particle_term"] += weight*p0*p1*(np.outer(G[i],G[j])+np.outer(G[j],G[i]))
            output["label_d_mass"] += weight
            output["label_c_mass"] += weight*2*p0*p1
    output["label_H_ba"] = output["label_B"]-output["label_same_particle_term"]+output["label_cross_particle_term"]
    output["label_M"] = output["label_same_particle_term"][:m,:m]
    output["label_sigma_prime_average"] = np.diag(output["label_B"])[:m]
    return output


def self_test():
    """No scientific files are opened and no random input is generated."""
    rows=[]
    for limit in (7.,1.2):
        b=np.array([.2,-.3]); a=np.array([.5,.7])
        exact=manufactured_pair_reference(b,a,1,limit=limit)
        numeric,info=independent_label(b,a,1,K=2,order=64,limit=limit)
        errors={k:float(np.max(abs(np.asarray(exact[k])-numeric[k]))) for k in exact}
        rows.append(dict(limit=limit,errors=errors,quad=info,
                         passed=info["quad_success"] and max(errors.values())<5e-10))
    # This import is confined to a manufactured fixture, never the audit run.
    import rankone_stationary_kernel as tested_kernel
    q=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1]])*(1.-3e-16)
    raw=tested_kernel.rankone_stationary(np.array([.12,-.24]),np.array([.45,.6]),q,
        K_values=(1,2,"inf"),normal_rule="legendre",normal_order=48,normal_limit=7.,
        logu_min=-32.,logu_max=64.,logu_step=.125)
    checks=Checks()
    reconstruct(raw,checks,"manufactured_aggregate")
    # Exercise final-output field wiring and failed-gate retention on a small,
    # intentionally inaccurate manufactured rule, not the research endpoints.
    import rankone_stationary_compute as tested_driver
    b=np.linspace(-.2,.2,8);a=np.linspace(.3,.5,8)
    q8=np.arange(1.,257.);q8/=q8.sum()
    fixture=tested_kernel.rankone_stationary(b,a,{"empirical":q8,"population":q8[::-1]},K_values=(8,64,"inf"),
        normal_rule="legendre",normal_order=16,normal_limit=3.,logu_min=-12.,logu_max=12.,logu_step=1.)
    tested_driver.project_result(fixture)
    arrays=tested_driver.kernel_arrays(fixture,np.r_[b,a])
    rebuilt=reconstruct(arrays,checks,"manufactured_final")
    summary=tested_driver.evaluation_summary(np.r_[b,a],fixture,q8[::-1],0.)
    check_final_summary(summary,arrays,rebuilt,q8[::-1],checks,"manufactured_final")
    qual=[dict(target=t,K=K,**tested_driver.qualification(fixture,fixture,ti,ki))
          for ti,t in enumerate(("empirical","population")) for ki,K in enumerate((8,64,"inf"))]
    restored=check_qualifications(qual,arrays,arrays,checks,"manufactured_final")
    signature=tested_driver.classification(np.r_[b,a],fixture,0,0,qual[0])
    check_signature(signature,np.r_[b,a],arrays,restored[0],0,checks,"manufactured_final")
    checks.require(not signature["qualified_full_local_minimum_signature"],"manufactured_failed_signature_preserved")
    # Inject a deterministic quadratic with a deliberately under-scaled Hessian
    # so direction capping and rejected backtracking trials are exercised.
    theta0=np.r_[np.zeros(8),np.full(8,.5)];center=theta0+.01
    def evaluator(theta,phase,hessian,context):
        delta=theta-center
        return dict(loss=np.array([[.5*(delta@delta)]]),gradient=delta[None,None,:],
                    H_ba=np.eye(16)[None,None,:,:]*.01 if hessian else None)
    trace=tested_driver.stationary_solve(theta0,evaluator,stages=(
        dict(name="O",max_accepted_steps=2,max_calls=30,gradient_tolerance=1e-12),))
    callmap={c["call_id"]:c for c in trace["calls"]}
    for step in trace["steps"]:check_newton_step(step,callmap,checks,"manufactured_Newton")
    checks.require(any(len(s["trials"])>1 for s in trace["steps"]),"manufactured_backtracking_exercised")
    return dict(passed=all(r["passed"] for r in rows) and not checks.failures, manufactured_only=True,
                scientific_inputs_read=False,new_random_draws=0,cases=rows,
                aggregation_assertions=checks.count,aggregation_failures=checks.failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test","--selftest",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_test()
        print(json.dumps(native(result),indent=2,allow_nan=False))
        return 0 if result["passed"] else 1
    if args.run and RECEIPT.exists():
        raise FileExistsError("AUDIT.json already exists; it will not be replaced")
    previous=None
    if args.check:
        previous=read(RECEIPT)
        if previous["source_sha256"]!=sha(__file__):
            raise ValueError("audit source changed after the saved receipt")
    try:
        result=perform()
    except BaseException as error:
        result=dict(passed=False,source_sha256=sha(__file__),failures=[dict(exception=str(error),traceback=traceback.format_exc())])
    if args.run:
        result["created_utc"]=datetime.now(timezone.utc).isoformat()
        with RECEIPT.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(native(result),f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")
    elif result["passed"]:
        # Exclude timing and live platform metadata; bind all actual inputs and
        # stable mathematical/solver outputs, including independent quadrature.
        keys=("passed","source_sha256","freeze_sha256","audit_freeze_sha256","input_hashes","failures",
              "checks","numeric_elements_compared","numerical_maxima","solver_checks","endpoint_checks",
              "independent_label_checks","probe_checks","scope")
        for key in keys:
            if native(result[key])!=previous[key]:
                raise ValueError("read-only repeat differs in stable result field: "+key)
    print(json.dumps(native({k:result[k] for k in ("passed","checks","seconds","failures") if k in result}),allow_nan=False),flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
