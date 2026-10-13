"""Independent execution/numerical audit of RankOneContinuation-v1.

Reuses the previously verified audit's quadrature and algebra helpers, not the
production kernel. The reuse is an explicit hashed dependency. No optimizer is
rerun and no Gaussian-integral error certificate is claimed.
"""
from __future__ import annotations

import os
for _var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_var]="1"

import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import scipy
import rankone_stationary_audit as base

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/rankone_continuation_v1_20261007"
RECEIPT=ART/"AUDIT.json"
COARSE=(9,10,12,16,24,32,64,128,"inf")
LABELS=(0,85,255)
read,sha,rel,native=base.read,base.sha,base.rel,base.native


def independent_posterior(b,a,label_index,order=384,limit=11.):
    """Direct per-label posterior loss and derivatives, with raw normal mass."""
    b,a=np.asarray(b),np.asarray(a);m=len(b)
    y=np.array([(label_index>>j)&1 for j in range(m-1,-1,-1)])
    z,w=base.normal_rule(order,limit)
    A,G,Bd=base.label_features(b,a,y,z)
    evidence=float(w@A);posterior=w*A/evidence
    meanG=posterior@G
    B=base.expand_B(posterior@Bd,m)
    same=np.sum(posterior[:,None,None]*G[:,:,None]*G[:,None,:],axis=0)
    cross=np.outer(meanG,meanG)
    return dict(label_loss=-math.log(evidence),label_gradient=-meanG,label_B=B,
        label_same_particle_term=same,label_cross_particle_term=cross,label_H_ba=B-same+cross,
        label_M=same[:m,:m],label_sigma_prime_average=np.diag(B)[:m],
        label_d_mass=float(posterior.sum()),label_c_mass=float(posterior.sum()))


def independent_label(b,a,label,K):
    if str(K)=="inf":
        return independent_posterior(b,a,label),dict(label_index=label,K="inf",method="independent posterior ratios")
    return base.independent_label(b,a,label,K=int(K))


def verify_trace(data,theta0,stages,checks,context):
    """Replay the stored solver's decisions; never call an optimizer/kernel."""
    calls={c["call_id"]:c for c in data["calls"]}
    checks.exact(list(calls),list(range(len(calls))),context+":call_ids")
    checks.exact(data["initial_theta"],theta0,context+":initial_snapshot")
    checks.exact(data["status"],"completed",context+":completed")
    checks.exact(len(data["attempted_calls"]),len(calls),context+":attempt_count")
    for attempt,c in zip(data["attempted_calls"],data["calls"]):
        checks.exact(attempt["status"],"completed",context+":attempt_status")
        for key in ("call_id","phase","role","phase_call_index","theta","compute_hessian"):
            checks.exact(attempt[key],c[key],context+":attempt_"+key)
    steps={s["current_call_id"]:s for s in data["steps"]}
    checks.exact(len(steps),len(data["steps"]),context+":unique_steps")
    checks.exact([s["step_id"] for s in data["steps"]],list(range(len(steps))),context+":step_ids")
    step_records=[base.check_newton_step(s,calls,checks,context+f"/step{s['step_id']}") for s in data["steps"]]
    last=np.asarray(theta0);accepted_ids=[];stage_records=[]
    for saved,stage in zip(data["stages"],stages):
        phase=stage["name"];maxcalls=stage["max_calls"];maxsteps=stage["max_accepted_steps"];tol=stage["gradient_tolerance"]
        for key,value in dict(phase=phase,max_calls=maxcalls,max_accepted_steps=maxsteps,gradient_tolerance=tol).items():
            checks.exact(saved[key],value,context+":stage_"+key)
        checks.exact(saved["start_theta"],last,context+":stage_start")
        phasecalls=[c for c in data["calls"] if c["phase"]==phase]
        checks.require(len(phasecalls)<=maxcalls,context+":call_budget")
        checks.exact([c["phase_call_index"] for c in phasecalls],list(range(1,len(phasecalls)+1)),context+":phase_call_indices")
        pos=0;accepted=0;reason=None
        while reason is None:
            if accepted>=maxsteps:reason="accepted_step_budget";break
            if pos>=maxcalls:reason="function_call_budget";break
            if pos>=len(phasecalls):
                checks.require(False,context+":missing_current_call");reason="missing_current_call";break
            c=phasecalls[pos];pos+=1
            checks.exact(c["role"],"current",context+":current_role")
            checks.exact(c["compute_hessian"],True,context+":current_H")
            checks.exact(c["theta"],last,context+":current_LAST")
            if np.max(abs(np.asarray(c["gradient"])))<=tol:reason="gradient_tolerance";break
            s=steps[c["call_id"]]
            checks.exact(s["phase"],phase,context+":step_phase")
            checks.exact(s["phase_step_index"],accepted,context+":phase_step_index")
            for key in ("theta","loss","gradient"):checks.exact(s[key],c[key],context+":step_snapshot_"+key)
            if s["initial_alpha"]<=0:reason="no_positive_feasible_box_step";break
            okay=False
            for trial in s["trials"]:
                checks.require(pos<len(phasecalls),context+":trial_exists")
                if pos>=len(phasecalls):break
                tc=phasecalls[pos];pos+=1
                checks.exact(tc["call_id"],trial["call_id"],context+":trial_sequence")
                for key in ("theta","loss","gradient","alpha","halvings"):
                    checks.exact(tc[key],trial[key],context+":trial_snapshot_"+key)
                if trial["accepted"]:
                    okay=True;accepted+=1;last=np.asarray(tc["theta"]);accepted_ids.append(tc["call_id"])
                    if np.max(abs(np.asarray(tc["gradient"])))<=tol:reason="gradient_tolerance_at_accepted_FG"
                    break
            if not okay:
                reason="function_call_budget" if pos>=maxcalls else "line_search_failed"
                if reason=="line_search_failed":checks.exact(len(s["trials"]),13,context+":backtracking_exhausted")
        checks.exact(pos,len(phasecalls),context+":all_calls_accounted")
        checks.exact(saved["stop_reason"],reason,context+":stop_reason")
        checks.exact(saved["call_count"],pos,context+":stage_calls")
        checks.exact(saved["accepted_step_count"],accepted,context+":stage_accepted")
        checks.exact(saved["final_theta"],last,context+":stage_final")
        checks.exact(saved["final_box"],base.box(last),context+":stage_box")
        stage_records.append(dict(phase=phase,calls=pos,accepted_steps=accepted,stop_reason=reason))
    checks.exact(len(data["stages"]),len(stages),context+":stage_count")
    checks.exact(data["final_theta"],last,context+":final_LAST")
    checks.exact(data["total_calls"],len(calls),context+":total_calls")
    checks.exact(data["accepted_step_count"],len(accepted_ids),context+":accepted_count")
    checks.exact([s["call_id"] for s in data["accepted_states"]],[None]+accepted_ids,context+":accepted_call_ids")
    for i,state in enumerate(data["accepted_states"]):
        checks.exact(state["state_id"],i,context+":state_id")
        if i==0:checks.exact(state["theta"],theta0,context+":initial_accepted_snapshot")
        else:
            c=calls[state["call_id"]]
            for key in ("theta","phase","loss","gradient","box"):
                checks.exact(state[key],c[key],context+":accepted_snapshot_"+key)
    return dict(stages=stage_records,calls=len(calls),accepted_steps=len(accepted_ids),steps=step_records)


def check_projection_summary(summary,raw,rebuilt,teacher,checks,context):
    a=raw["a"];V=raw["transverse_basis"];Q=base.helmert(8)
    u=a/np.linalg.norm(a);v=u.copy();v[0]+=1 if u[0]>=0 else -1
    checks.close(V,(np.eye(8)-2*np.outer(v,v)/(v@v))[:,1:],"transverse_basis",context,0.)
    checks.close(V.T@V,np.eye(7),"transverse_orthonormal",context)
    checks.close(V.T@a,np.zeros(7),"transverse_orthogonal_a",context)
    checks.close(raw["helmert_basis"],Q,"helmert_basis",context,0.)
    for name,H in (("H_ba",rebuilt["H_ba"]),("Hperp_projected",V.T@rebuilt["Hperp"]@V),
                   ("Hangle_projected",Q.T@rebuilt["Hangle"]@Q)):
        checks.close(raw[name],H,"projection."+name,context)
        S=(H+H.swapaxes(-1,-2))/2;vals=np.linalg.eigvalsh(S);vec=raw[name+"_eigenvectors"]
        checks.close(raw[name+"_eigenvalues"],vals,"spectrum."+name,context)
        checks.close(S@vec,vec*raw[name+"_eigenvalues"][...,None,:],"eigenpairs."+name,context)
        checks.close(vec.swapaxes(-1,-2)@vec,np.broadcast_to(np.eye(H.shape[-1]),vec.shape),"eigenvectors."+name,context)
    checks.close(raw["Hperp_projected_eigenvalues"],rebuilt["audit_Hperp_eigenvalues"],"independent_basis_spectrum",context)
    checks.exact(summary["theta"],raw["theta"],context+":theta")
    checks.exact(summary["target_names"],raw["target_names"],context+":target_names")
    checks.exact([str(k) for k in summary["K_values"]],raw["K_names"],context+":K_values")
    nk=len(raw["K_names"])
    checks.exact(len(summary["blocks"]),nk,context+":block_count")
    for ki,row in enumerate(summary["blocks"]):
        checks.exact((row["target"],str(row["K"])),("population",raw["K_names"][ki]),context+":block_identity")
        for key in ("loss","gradient","bias_gradient","radial_gradient","H_ba","Hperp","Hangle","directWard","ward_residual",
                    "H_ba_eigenvalues","H_ba_eigenvectors","Hperp_projected","Hperp_projected_eigenvalues","Hperp_projected_eigenvectors",
                    "Hangle_projected","Hangle_projected_eigenvalues","Hangle_projected_eigenvectors"):
            checks.exact(row[key],raw[key][0,ki],context+":json_raw_"+key)
        values=dict(gradient_Linf=np.max(abs(rebuilt["gradient"][0,ki])),raw_H_ba_symmetry_residual=rebuilt["H_ba_symmetry_residual"][0,ki],
            Ward_Linf=rebuilt["ward_residual_max"][0,ki],Cartesian_Ward_Linf=rebuilt["audit_Cartesian_Ward"][0,ki],
            near_rotation_residual_Linf=rebuilt["audit_normalized_rotation_residual"][0,ki],
            structural_common_angle_residual=rebuilt["common_angle_residual"][0,ki])
        for key,value in values.items():checks.close(row[key],value,"summary."+key,context)
        checks.exact(row["scalar_diagnostics"],base.scalar_diagnostics(raw,ki),context+":scalars")
    return base.check_probability(summary["probability"],raw,teacher,checks,context)


def check_qualifications(rows,C,D,checks,context):
    checks.exact(len(rows),len(C["K_names"]),context+":qualification_count")
    derived=[]
    for ki,row in enumerate(rows):
        checks.exact((row["target"],str(row["K"])),("population",C["K_names"][ki]),context+":qualification_identity")
        differences={key:float(np.max(abs(C[key][0,ki]-D[key][0,ki]))) for key in
            ("loss","gradient","H_ba","Hperp","Hangle","H_ba_eigenvalues","Hperp_projected_eigenvalues","Hangle_projected_eigenvalues")}
        gates=dict(loss=differences["loss"]<=1e-7,gradient=differences["gradient"]<=2e-7)
        for key in ("H_ba","Hperp","Hangle"):gates[key]=differences[key]<=2e-7
        for key in ("H_ba_eigenvalues","Hperp_projected_eigenvalues","Hangle_projected_eigenvalues"):gates[key]=differences[key]<=2e-6
        for name,raw in (("C",C),("D",D)):
            diag=base.scalar_diagnostics(raw,ki)
            gates.update({name+"_normal_mass":diag["normal_mass_error"]<=2e-12,
                name+"_d_relative_mass":diag["d_relative_mass_error"]<=2e-7,
                name+"_c_lower":diag["c_raw_lower_violation"]<=2e-7,
                name+"_c_upper":diag["c_raw_upper_violation"]<=2e-7,
                name+"_Ward":raw["ward_residual_max"][0,ki]<=2e-7})
            checks.exact(row["scalar_diagnostics"][name],diag,context+":qualification_scalars")
        checks.exact(row["comparisons"],differences,context+":comparisons")
        checks.exact(row["gates"],gates,context+":gates")
        passed=all(gates.values());failed=[k for k,v in gates.items() if not v]
        checks.exact(row["qualified"],passed,context+":qualified")
        checks.exact(row["failed_gates"],failed,context+":failed_gates")
        derived.append(dict(K=row["K"],qualified=passed,failed_gates=failed))
    return derived


def classify_values(theta,Dblock,qualified):
    a=np.asarray(theta)[8:];g=np.asarray(Dblock["gradient"]);Hp=np.asarray(Dblock["Hperp"])
    values=dict(gradient_Linf=float(np.max(abs(g))),box_margin=base.box(theta)["minimum_margin"],
        H_ba_minimum=float(Dblock["H_ba_eigenvalues"][0]),
        Hperp_minimum=float(Dblock["Hperp_projected_eigenvalues"][0]),
        Cartesian_Ward=float(np.max(abs(Hp@a-g[8:]))),rotation_residual=float(np.max(abs(Hp@(a/np.linalg.norm(a))))))
    gates=dict(numerical=qualified,gradient=values["gradient_Linf"]<=1e-8,box_margin=values["box_margin"]>=1e-3,
        H_ba=values["H_ba_minimum"]>1e-5,Cartesian_Ward=values["Cartesian_Ward"]<=2e-7,rotation=values["rotation_residual"]<=2e-7)
    radial=all(gates.values());e=values["Hperp_minimum"]
    sign="positive" if radial and e>1e-5 else "negative" if radial and e< -1e-5 else "unresolved"
    return dict(radial_qualified=radial,sign=sign,gates=gates,values=values)


def check_own(point,checks,context):
    result=classify_values(point["theta"],point["own_blocks"]["D"],point["own_qualification"]["qualified"])
    values=result["values"]
    fields=dict(D_gradient_Linf="gradient_Linf",minimum_box_margin="box_margin",D_H_ba_minimum="H_ba_minimum",
        D_transverse_minimum="Hperp_minimum",D_Cartesian_Ward_Linf="Cartesian_Ward",D_near_rotation_residual_Linf="rotation_residual")
    for key,valuekey in fields.items():checks.close(point[key],values[valuekey],"classification."+key,context)
    gates=dict(numerical_qualification=result["gates"]["numerical"],gradient=result["gates"]["gradient"],
        box_margin=result["gates"]["box_margin"],H_ba_positive=result["gates"]["H_ba"],
        Cartesian_Ward=result["gates"]["Cartesian_Ward"],near_rotation_mode=result["gates"]["rotation"])
    checks.exact(point["radial_qualification"],dict(qualified=result["radial_qualified"],gates=gates,
                 failed_gates=[k for k,v in gates.items() if not v]),context+":radial_qualification")
    checks.exact(point["transverse_label"],result["sign"],context+":transverse_label")
    eig=values["Hperp_minimum"];rawsign="positive" if eig>1e-5 else "negative" if eig< -1e-5 else "unresolved"
    checks.exact(point["raw_transverse_label_without_radial_qualification"],rawsign,context+":raw_sign")
    return result


def check_artifact(item,checks):
    checks.bind(ROOT/item["path"],item["sha256"],item.get("bytes"))


def check_evaluation(evaluation,folder,stem,theta,Ks,q,checks,resources):
    raws={};summaries={}
    for rule in ("C","D"):
        paths=evaluation["rule_artifacts"][rule]
        for key in ("array_artifact","summary_artifact"):check_artifact(paths[key],checks)
        path=folder/(stem+"_"+rule+".npz")
        checks.exact(paths["array_artifact"]["path"],rel(path),stem+":evaluation_path")
        raw,rebuilt=base.load_raw(path,checks,theta,["population"],[q],rule,resources)
        checks.exact(raw["K_names"],[str(k) for k in Ks],stem+":K_values")
        summary=read(path.with_suffix(".json"))
        checks.exact(summary["array_artifact"],paths["array_artifact"],stem+":array_binding")
        check_projection_summary(summary,raw,rebuilt,q,checks,rel(path))
        raws[rule],summaries[rule]=raw,summary
    quals=check_qualifications(evaluation["qualifications"],raws["C"],raws["D"],checks,stem)
    return raws,summaries,quals


def check_point(point,q,stages,fh,sh,checks,resources):
    pid=point["point_id"];folder=ART/"points"/pid;context="point/"+pid;K=point["K"]
    check_artifact(point["result_artifact"],checks)
    saved=read(folder/"RESULTS.json")
    checks.exact(saved,{k:v for k,v in point.items() if k!="result_artifact"},context+":public_snapshot")
    start=read(folder/"START_LOCK.json");checks.bind(folder/"START_LOCK.json")
    for key in ("point_id","K","kind","theta0","start_context"):checks.exact(start[key],point[key],context+":start_"+key)
    checks.exact(start["target"],q,context+":unmodified_target")
    checks.exact(start["freeze_sha256"],fh,context+":start_freeze")
    checks.exact(start["source_sha256"],sh,context+":start_source")
    checks.exact(start["locked_before_optimization"],True,context+":start_lock")
    check_artifact(point["parameter_lock_artifact"],checks)
    lock=read(folder/"PARAMETERS_LOCK.json")
    for key in ("point_id","K","theta"):checks.exact(lock[key],point[key],context+":lock_"+key)
    for key in ("start_lock_artifact","solver_artifact","tape_artifact"):check_artifact(lock[key],checks)
    for path,digest in lock["optimization_artifacts"].items():checks.bind(ROOT/path,digest)
    checks.exact(lock["locked_before_own_K_and_integrated_evaluation"],True,context+":parameters_locked_first")
    checks.exact(lock["parameter_selection_by_risk_or_curvature"],False,context+":no_reselection")
    checks.exact(lock["selected_state"],"finite_feasible_LAST",context+":selection_rule")
    solver=read(folder/"optimization/SOLVER.json")
    checks.exact(solver["K"],K,context+":solver_K")
    checks.exact(solver["freeze_sha256"],fh,context+":solver_freeze")
    checks.exact(solver["fit_target"],q,context+":solver_target")
    checks.exact(solver["final_theta"],point["theta"],context+":locked_LAST")
    trace=verify_trace(solver,point["theta0"],stages,checks,context)
    begin=len(resources)
    checks.exact(len(solver["call_artifacts"]),len(solver["calls"]),context+":call_artifact_count")
    for item,c in zip(solver["call_artifacts"],solver["calls"]):
        for field in ("array_artifact","summary_artifact"):check_artifact(item[field],checks)
        summary=read(ROOT/item["summary_artifact"]["path"])
        checks.exact(summary["K"],K,context+":call_K")
        for field in ("call_id","phase","role","phase_call_index","theta","loss","gradient","H_ba","compute_hessian"):
            checks.exact(summary[field],c[field],context+":call_snapshot_"+field)
        raw,_=base.load_raw(ROOT/item["array_artifact"]["path"],checks,c["theta"],["population"],[q],c["phase"],resources)
        checks.exact(raw["K_names"],[str(K)],context+":optimized_K")
        for field in ("loss","gradient"):checks.exact(c[field],raw[field][0,0],context+":raw_"+field)
        checks.exact(c["H_ba"],raw["H_ba"][0,0] if c["compute_hessian"] else None,context+":raw_H")
        checks.exact(c["box"],base.box(c["theta"]),context+":call_box")
    base.check_resources(resources[begin:],solver["resources"],trace["accepted_steps"],checks,context+"/optimization")
    checks.exact(point["optimization_resources"],solver["resources"],context+":optimization_resources")
    with np.load(folder/"optimization/TAPE.npz",allow_pickle=False) as f:tape={key:f[key] for key in f.files}
    for field,key in dict(call_id="call_id",call_theta="theta",call_loss="loss",call_gradient="gradient",phase="phase",role="role",compute_hessian="compute_hessian").items():
        checks.exact(tape[field],[c[key] for c in solver["calls"]],context+":tape_"+field)
    for i,c in enumerate(solver["calls"]):
        if c["compute_hessian"]:checks.exact(tape["call_H_ba"][i],c["H_ba"],context+":tape_H")
        else:checks.require(np.isnan(tape["call_H_ba"][i]).all(),context+":missing_H_mask")
    checks.exact(tape["accepted_theta"],[s["theta"] for s in solver["accepted_states"]],context+":tape_accepted")
    checks.exact(tape["accepted_call_id"],[-1 if s["call_id"] is None else s["call_id"] for s in solver["accepted_states"]],context+":tape_accepted_ids")
    checks.exact(point["stage_stop_reasons"],[s["stop_reason"] for s in solver["stages"]],context+":stops")
    eval_begin=len(resources);Ks=("inf",) if str(K)=="inf" else (K,"inf")
    raws,summaries,quals=check_evaluation(point["evaluation"],folder,"evaluation",point["theta"],Ks,q,checks,resources)
    for rule in ("C","D"):
        checks.exact(point["own_blocks"][rule],summaries[rule]["blocks"][0],context+":own_block")
        checks.exact(point["probability"][rule],summaries[rule]["probability"],context+":probability")
    checks.exact(point["own_qualification"],point["evaluation"]["qualifications"][0],context+":own_qualification")
    own=check_own(point,checks,context)
    base.check_resources(resources[eval_begin:],point["evaluation_resources"],0,checks,context+"/evaluation")
    base.check_resources(resources[begin:],point["resources"],trace["accepted_steps"],checks,context+"/total")
    sampled=[];theta=np.asarray(point["theta"])
    for label in LABELS:
        values,info=independent_label(theta[:8],theta[8:],label,K)
        checks.require(info.get("quad_success",True),context+f"/label{label}:quadrature_success")
        errors={key:checks.close(value,raws["D"][key][0,label],"independent."+key,context+f"/label{label}",5e-9) for key,value in values.items()}
        sampled.append(dict(label=label,K=K,values=values,errors=errors,diagnostics=info))
    return dict(point_id=pid,K=K,solver=trace,own=own,qualifications=quals,independent_label_checks=sampled)


def check_refinement(coarse,saved,checks):
    pair=None
    for left,right in zip(coarse,coarse[1:]):
        if left["K"]=="inf" or right["K"]=="inf":continue
        if left["radial_qualification"]["qualified"] and right["radial_qualification"]["qualified"] and left["transverse_label"]=="positive" and right["transverse_label"]=="negative":
            pair=(left,right);break
    plan=dict(maximum_refinements=6,
        coarse_order=[dict(point_id=p["point_id"],K=p["K"],radial_qualified=p["radial_qualification"]["qualified"],transverse_label=p["transverse_label"]) for p in coarse],
        selected_first_adjacent_pair=None if pair is None else [p["point_id"] for p in pair],
        selected_K_pair=None if pair is None else [p["K"] for p in pair])
    checks.exact(saved["plan"],plan,"refinement:initial_plan")
    checks.exact(read(ART/"REFINEMENT_PLAN.json"),plan,"refinement:plan_snapshot")
    checks.require(len(saved["points"])<=6,"refinement:budget")
    checks.exact(len(saved["decisions"]),len(saved["points"]),"refinement:decision_count")
    if pair is None:
        checks.exact(saved,dict(status="no_usable_finite_bracket",plan=plan,points=[],decisions=[],final_pair=None),"refinement:no_pair")
        return dict(status="no_usable_finite_bracket",iterations=0),None
    left,right=pair;reason="maximum_refinements_reached";records=[]
    for i,middle in enumerate(saved["points"]):
        lo,hi=left["K"],right["K"]
        checks.require(hi-lo>1,"refinement:no_solve_after_integer_adjacency")
        mid=(lo+hi)//2;fraction=(mid-lo)/(hi-lo)
        start=(1-fraction)*np.asarray(left["theta"])+fraction*np.asarray(right["theta"])
        expected=dict(index=i,point_id=f"refine_{i:02d}_K{mid:03d}",K=mid,
            lower_point_id=left["point_id"],upper_point_id=right["point_id"],lower_K=lo,upper_K=hi,
            lower_theta=left["theta"],upper_theta=right["theta"],interpolation_fraction=fraction,
            theta0=start,interpolation_coordinate="K")
        checks.exact(read(ART/"refinement"/f"midpoint_{i:02d}_PLAN.json"),expected,"refinement:midpoint_plan")
        checks.exact(middle["K"],mid,"refinement:integer_midpoint")
        checks.exact(middle["point_id"],expected["point_id"],"refinement:midpoint_id")
        checks.exact(middle["theta0"],start,"refinement:K_interpolated_start")
        checks.exact(middle["start_context"],expected,"refinement:midpoint_start_lock")
        action=("stop_radial_or_numerical_unresolved" if not middle["radial_qualification"]["qualified"] else
                "replace_lower" if middle["transverse_label"]=="positive" else
                "replace_upper" if middle["transverse_label"]=="negative" else "stop_transverse_unresolved")
        before=dict(lower_point_id=left["point_id"],upper_point_id=right["point_id"],lower_K=lo,upper_K=hi)
        if action=="replace_lower":left=middle
        elif action=="replace_upper":right=middle
        decision=dict(index=i,plan=expected,midpoint_point_id=middle["point_id"],midpoint_K=mid,
            midpoint_result_artifact=middle["result_artifact"],midpoint_radial_qualification=middle["radial_qualification"],
            midpoint_transverse_label=middle["transverse_label"],action=action,before=before,
            after=dict(lower_point_id=left["point_id"],upper_point_id=right["point_id"],lower_K=left["K"],upper_K=right["K"]))
        checks.exact(saved["decisions"][i],decision,"refinement:decision")
        checks.exact(read(ART/"refinement"/f"decision_{i:02d}.json"),decision,"refinement:decision_snapshot")
        records.append(dict(K=mid,action=action,before=before,after=decision["after"]))
        if action.startswith("stop_"):
            reason=action;checks.exact(i+1,len(saved["points"]),"refinement:stop_on_unresolved");break
        if right["K"]-left["K"]<=1:
            reason="adjacent_integers";checks.exact(i+1,len(saved["points"]),"refinement:stop_on_adjacent");break
    if not saved["points"] and right["K"]-left["K"]<=1:reason="adjacent_integers"
    if reason=="maximum_refinements_reached":checks.exact(len(saved["points"]),6,"refinement:all_six_required")
    checks.exact(saved["status"],reason,"refinement:status")
    checks.exact(saved["iterations"],len(saved["points"]),"refinement:iterations")
    checks.exact(saved["final_pair"],dict(lower=left,upper=right),"refinement:retained_pair")
    return dict(status=reason,iterations=len(records),decisions=records),dict(lower=left,upper=right)


def check_cross(saved,pair,q,lockhash,checks,resources):
    begin=len(resources)
    if pair is None:
        checks.exact(saved["status"],"skipped_no_usable_finite_bracket","cross:skipped")
        checks.exact(saved["new_target_blocks"],0,"cross:no_blocks")
        base.check_resources([],saved["resources"],0,checks,"cross")
        return dict(status="skipped_no_usable_finite_bracket")
    left,right=pair["lower"],pair["upper"]
    check_artifact(saved["result_artifact"],checks);check_artifact(saved["plan_artifact"],checks)
    checks.exact(read(ART/"cross/RESULTS.json"),{k:v for k,v in saved.items() if k!="result_artifact"},"cross:result_snapshot")
    plan=read(ART/"cross/PLAN.json");checks.exact(plan["endpoint_lock_sha256"],lockhash,"cross:all_parameters_locked")
    checks.exact(plan["direction_locked_before_cross_evaluation"],True,"cross:direction_locked")
    for key,value in dict(lower_point_id=left["point_id"],upper_point_id=right["point_id"],lower_K=left["K"],upper_K=right["K"]).items():
        checks.exact(plan[key],value,"cross:plan_"+key);checks.exact(saved[key],value,"cross:result_"+key)
    checks.exact(plan["theta_L"],left["theta"],"cross:theta_L")
    checks.exact(plan["theta_H"],right["theta"],"cross:theta_H")
    block=right["own_blocks"]["D"];Q=base.helmert(8)
    vec=np.asarray(block["Hangle_projected_eigenvectors"])[:,0]
    psi=Q@vec;index=int(np.argmax(abs(psi)));sign=1 if psi[index]>=0 else -1;psi*=sign
    vals=np.asarray(block["Hangle_projected_eigenvalues"])
    H=np.asarray(block["Hangle"]);P=Q.T@H@Q;S=(P+P.T)/2
    checks.close(np.linalg.eigvalsh(S),vals,"cross.negative_endpoint_spectrum","cross")
    checks.close(S@vec,vals[0]*vec,"cross.selected_eigenpair","cross")
    checks.close(psi@psi,1.,"cross.unit_direction","cross")
    checks.close(psi.sum(),0.,"cross.rotation_orthogonal","cross")
    direction=dict(direction=psi,helmert_eigenvector=sign*vec,sign_coordinate=index,
        D_minimum_angle_eigenvalue=float(vals[0]),D_angle_eigengap=float(vals[1]-vals[0]),
        direction_L2=float(np.linalg.norm(psi)),direction_sum=float(psi.sum()))
    for key,value in direction.items():
        checks.close(plan[key],value,"cross.plan_"+key,"cross")
        checks.close(saved["direction"][key],value,"cross.direction_"+key,"cross")
    blocks={"LL":left["own_blocks"],"HH":right["own_blocks"]}
    quals={"LL":left["own_qualification"],"HH":right["own_qualification"]}
    for cell,point,K in (("LH",left,right["K"]),("HL",right,left["K"])):
        raw,summaries,derived=check_evaluation(saved["cross_evaluations"][cell],ART/"cross",cell,point["theta"],(K,),q,checks,resources)
        blocks[cell]={r:s["blocks"][0] for r,s in summaries.items()}
        quals[cell]=saved["cross_evaluations"][cell]["qualifications"][0]
    values={"C":{},"D":{}}
    for cell in ("LL","LH","HL","HH"):
        row=saved["cells"][cell]
        metadata=dict(theta_source="lower" if cell[0]=="L" else "upper",K=left["K"] if cell[1]=="L" else right["K"],
                      reused_own_block=cell in ("LL","HH"))
        checks.exact(plan["cells"][cell],metadata,"cross:cell_plan")
        for key,value in metadata.items():checks.exact(row[key],value,"cross:cell_"+key)
        for rule in ("C","D"):
            matrix=np.asarray(blocks[cell][rule]["Hangle"]);val=float(psi@matrix@psi);values[rule][cell]=val
            checks.exact(row["Hangle"][rule],matrix,"cross:compensated_angle_matrix")
            checks.close(row[rule+"_curvature"],val,"cross.same_direction_curvature","cross/"+cell)
        checks.close(row["curvature_rule_difference"],abs(values["C"][cell]-values["D"][cell]),"cross.rule_difference","cross/"+cell)
        checks.exact(row["numerical_qualification"],quals[cell],"cross:cell_qualification")
    for rule,v in values.items():
        LL,LH,HL,HH=(v[k] for k in ("LL","LH","HL","HH"));total=HH-LL
        kfirst=dict(K_change=LH-LL,theta_change=HH-LH);tfirst=dict(theta_change=HL-LL,K_change=HH-HL)
        expected=dict(total=total,K_first=kfirst,theta_first=tfirst,K_first_sum=sum(kfirst.values()),theta_first_sum=sum(tfirst.values()),
            K_first_roundoff_residual=sum(kfirst.values())-total,theta_first_roundoff_residual=sum(tfirst.values())-total,
            interaction=HH-LH-HL+LL,descriptive_not_causal=True,unique_mechanism_attribution=False)
        for key,value in expected.items():
            if isinstance(value,dict):
                for k,x in value.items():checks.close(saved["algebra"][rule][key][k],x,"cross.algebra_"+key+"_"+k,"cross/"+rule)
            elif isinstance(value,bool):checks.exact(saved["algebra"][rule][key],value,"cross:algebra_scope")
            else:checks.close(saved["algebra"][rule][key],value,"cross.algebra_"+key,"cross/"+rule)
    checks.exact(saved["all_four_cells_numerically_qualified"],all(x["qualified"] for x in quals.values()),"cross:all_qualified")
    for key,value in dict(new_target_blocks=2,new_rule_blocks=4,new_optimizer_steps=0,noncollinear_function_evaluations=0).items():
        checks.exact(saved[key],value,"cross:"+key)
    base.check_resources(resources[begin:],saved["resources"],0,checks,"cross")
    return dict(status="completed",values=values,algebra=saved["algebra"],all_four_cells_qualified=all(x["qualified"] for x in quals.values()))


def perform():
    tick=time.perf_counter();checks=base.Checks();resources=[]
    if not (ART/"RESULTS.json").exists() or read(ART/"RESULTS.json").get("status")!="completed":
        raise RuntimeError("The fixed continuation and all dependent evaluations must complete before audit")
    frozen=read(ART/"FREEZE.json");source=read(ART/"SOURCE.json");af=read(ART/"AUDIT_FREEZE.json")
    fh=checks.bind(ART/"FREEZE.json");sh=checks.bind(ART/"SOURCE.json");afh=checks.bind(ART/"AUDIT_FREEZE.json")
    checks.exact(af["scientific_freeze_sha256"],fh,"audit_freeze:scientific_freeze")
    checks.exact(af["sources"].get(rel(__file__)),sha(__file__),"audit_freeze:new_audit")
    checks.exact(af["sources"].get(rel(base.__file__)),sha(base.__file__),"audit_freeze:reused_audit")
    checks.exact(frozen["source_sha256"],sh,"freeze:source")
    for mapping in (frozen["sources"],af["sources"],source["sources"]):
        for path,digest in mapping.items():checks.bind(ROOT/path,digest)
    config=frozen["config"]
    checks.exact(config["coarse_K"],COARSE,"fixed_coarse_grid")
    checks.exact(config["maximum_refinements"],6,"fixed_refinement_budget")
    stages=[dict(name="O",max_accepted_steps=12,max_calls=80,gradient_tolerance=1e-9),
            dict(name="C",max_accepted_steps=4,max_calls=40,gradient_tolerance=1e-10)]
    checks.exact(config["stages"],stages,"fixed_solver_budgets")
    q=source["teacher_pmf"];theta0=source["initial_theta"]
    old={key:read(ROOT/source["initial_K8"][key]) for key in ("C","D","qualification","endpoint_lock")}
    P=next(e for e in old["endpoint_lock"]["endpoints"] if e["name"]=="P")
    checks.exact(P["theta"],theta0,"old_P_initial_theta")
    checks.exact(P["fit_target"],q,"old_P_original_target")
    result=read(ART/"RESULTS.json");checks.bind(ART/"RESULTS.json")
    for path,digest in result["artifacts"].items():checks.bind(ROOT/path,digest)
    for key,value in dict(freeze_sha256=fh,source_sha256=sh).items():checks.exact(result[key],value,"result:"+key)
    initial=result["initial"]
    checks.exact(initial,read(ART/"INITIAL_K8.json"),"initial:snapshot")
    checks.exact(initial["theta"],theta0,"initial:theta")
    checks.exact((initial["K"],initial["point_id"]),(8,"initial_K8"),"initial:identity")
    for rule in ("C","D"):
        checks.exact(old[rule]["theta"],theta0,"old_P_public_theta")
        block=next(b for b in old[rule]["blocks"] if b["target"]=="population" and b["K"]==8)
        checks.exact(initial["own_blocks"][rule],block,"initial:public_block")
    oldqual=next(r for r in old["qualification"]["qualifications"] if r["target"]=="population" and r["K"]==8)
    checks.exact(initial["own_qualification"],oldqual,"initial:public_qualification")
    initial_class=check_own(initial,checks,"initial_K8")
    checks.exact(initial["old_arrays_read"],False,"initial:no_old_arrays")
    checks.exact(initial["new_kernel_calls"],0,"initial:no_K8_replay")
    coarse=result["coarse"]
    checks.exact([p["K"] for p in coarse],[8]+list(COARSE),"coarse:complete_order")
    checks.exact(coarse[0],initial,"coarse:initial")
    for index,(previous,p) in enumerate(zip(coarse,coarse[1:])):
        K=COARSE[index];pid="coarse_Kinf" if K=="inf" else f"coarse_K{K:03d}"
        checks.exact(p["point_id"],pid,"coarse:point_id")
        checks.exact(p["theta0"],previous["theta"],"coarse:preceding_LAST")
        context=dict(coarse_index=index,predecessor_point_id=previous["point_id"],predecessor_K=previous["K"],
            predecessor_theta=previous["theta"],start_rule="preceding_finite_feasible_LAST",predecessor_result_artifact=previous.get("result_artifact"))
        checks.exact(p["start_context"],context,"coarse:predecessor_context")
        checks.exact(p["kind"],"coarse","coarse:kind")
    cl=read(ART/"COARSE_LOCK.json")
    checks.exact(cl["rows"],coarse,"coarse:locked_complete_table")
    checks.exact(cl["full_sequence_completed"],True,"coarse:complete_before_refinement")
    checks.exact(cl["coarse_solve_count"],9,"coarse:solve_count")
    checks.exact(cl["curve_or_risk_based_early_stop"],False,"coarse:no_selection_stop")
    refinement=result["refinement"]
    checks.exact(refinement,read(ART/"BRACKET_FINAL.json"),"refinement:final_snapshot")
    checks.exact(result["points"],coarse[1:]+refinement["points"],"all_new_points_in_execution_order")
    point_records=[]
    for point in result["points"]:
        record=check_point(point,q,stages,fh,sh,checks,resources);point_records.append(record)
        print(json.dumps(dict(audited_point=point["point_id"],K=point["K"],transverse_label=record["own"]["sign"])),flush=True)
    refinement_record,pair=check_refinement(coarse,refinement,checks)
    for point in refinement["points"]:checks.exact(point["kind"],"refinement","refinement:kind")
    check_artifact(result["endpoint_lock_artifact"],checks)
    lockpath=ART/"ENDPOINTS_LOCK.json";lh=checks.bind(lockpath);lock=read(lockpath)
    checks.exact(lock["all_endpoints_locked_before_cross"],True,"endpoint_lock:before_cross")
    expected=[{key:p[key] for key in ("point_id","K","theta","parameter_lock_artifact","result_artifact")} for p in result["points"]]
    checks.exact(lock["endpoints"],expected,"endpoint_lock:all_snapshots")
    for key in ("old_K8_artifact","coarse_lock_artifact","bracket_final_artifact"):check_artifact(lock[key],checks)
    checks.exact(lock["freeze_sha256"],fh,"endpoint_lock:freeze")
    checks.exact(lock["source_sha256"],sh,"endpoint_lock:source")
    for obj in (result,lock):checks.exact(obj["new_endpoint_count"],len(point_records),"new_endpoint_count")
    checks.exact(result["coarse_solve_count"],9,"result:coarse_count")
    checks.exact(result["refinement_solve_count"],len(refinement["points"]),"result:refinement_count")
    cross=check_cross(result["cross"],pair,q,lh,checks,resources)
    base.check_resources(resources,result["resources"],sum(p["solver"]["accepted_steps"] for p in point_records),checks,"global")
    for path,digest in list(checks.hashes.items()):checks.bind(ROOT/path,digest)
    return dict(passed=not checks.failures,source_sha256=sha(__file__),reused_audit_sha256=sha(base.__file__),freeze_sha256=fh,
        audit_freeze_sha256=afh,input_hashes=checks.hashes,failures=checks.failures,checks=checks.count,
        numeric_elements_compared=checks.elements,numerical_maxima=checks.maxima,initial_classification=initial_class,
        point_checks=point_records,refinement_checks=refinement_record,cross_checks=cross,
        scope=dict(role="independent numerical/execution audit, not external scientific review",
            reused_verified_audit_functions=True,production_kernel_used_for_audit=False,old_NPZ_read=False,
            all_new_call_and_evaluation_arrays_reconstructed=len(resources),new_solved_endpoints=len(point_records),
            independent_own_K_endpoint_label_cells=3*len(point_records),fixed_labels=LABELS,
            inner_rule="same D384/T11 reconstructed with numpy leggauss",finite_K_outer_integrator=base.QUAD_SPEC,
            K_infinity_method="direct posterior-ratio derivatives",solver_decisions_replayed_without_optimization=True,
            full_independent_integral_replay=False,rigorous_Gaussian_or_Laplace_error_certificate=False,
            new_random_draws=0,new_observations=0,new_optimizer_steps=0,new_statistical_alpha=0,
            qualified_negative_or_unresolved_scientific_results_are_not_audit_failures=True),
        seconds=time.perf_counter()-tick,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}))


def self_test():
    b=np.array([.2,-.3]);a=np.array([.5,.7]);theta=np.r_[b,a]
    reference=base.manufactured_pair_reference(b,a,1)
    finite,info=base.independent_label(b,a,1,K=2,order=64,limit=7.)
    finite_errors={k:float(np.max(abs(np.asarray(reference[k])-finite[k]))) for k in reference}
    post=independent_posterior(b,a,1,64,7.)
    fd=np.empty((4,4));lossgrad=np.empty(4);step=1e-5
    for i in range(4):
        d=np.zeros(4);d[i]=step
        left=independent_posterior((theta-d)[:2],(theta-d)[2:],1,64,7.)
        right=independent_posterior((theta+d)[:2],(theta+d)[2:],1,64,7.)
        fd[:,i]=(right["label_gradient"]-left["label_gradient"])/(2*step)
        lossgrad[i]=(right["label_loss"]-left["label_loss"])/(2*step)
    herror=float(np.max(abs(fd-post["label_H_ba"])));gerror=float(np.max(abs(lossgrad-post["label_gradient"])))
    # Exercise variable-K output wiring and retained failed gates using only
    # a manufactured model and deliberately coarse normal quadrature.
    import rankone_stationary_compute as manufactured
    b8=np.linspace(-.2,.2,8);a8=np.linspace(.3,.5,8);q=np.arange(1.,257.);q/=q.sum()
    fixture=manufactured.kernel.rankone_stationary(b8,a8,{"population":q},K_values=(9,"inf"),normal_rule="legendre",
        normal_order=16,normal_limit=3.,logu_min=-12.,logu_max=12.,logu_step=1.)
    manufactured.project_result(fixture)
    raw=manufactured.kernel_arrays(fixture,np.r_[b8,a8]);checks=base.Checks()
    rebuilt=base.reconstruct(raw,checks,"manufactured_variable_K")
    summary=manufactured.evaluation_summary(np.r_[b8,a8],fixture,q,0.)
    check_projection_summary(summary,raw,rebuilt,q,checks,"manufactured_variable_K")
    rows=[dict(target="population",K=k,**manufactured.qualification(fixture,fixture,0,ki)) for ki,k in enumerate((9,"inf"))]
    quals=check_qualifications(rows,raw,raw,checks,"manufactured_variable_K")
    checks.require(not quals[0]["qualified"],"manufactured_failed_gate_retained")
    theta0=np.r_[np.zeros(8),np.full(8,.5)];center=theta0+.01
    def evaluate(theta,phase,hessian,context):
        d=theta-center
        return dict(loss=np.array([[.5*(d@d)]]),gradient=d[None,None,:],
                    H_ba=np.eye(16)[None,None,:,:]*.01 if hessian else None)
    stages=(dict(name="O",max_calls=30,max_accepted_steps=2,gradient_tolerance=1e-12),
            dict(name="C",max_calls=20,max_accepted_steps=1,gradient_tolerance=1e-12))
    tape=manufactured.stationary_solve(theta0,evaluate,stages=stages)
    verify_trace(tape,theta0,stages,checks,"manufactured_trace")
    # Manufacture saved refinement snapshots in memory, including an unresolved
    # midpoint. Neither the actual SOURCE nor a research endpoint is inspected.
    import rankone_continuation_compute as manufactured_continuation
    def point(K,label,pid=None):
        return dict(point_id=pid or f"fixture_K{K}",K=K,theta=np.full(16,float(K)/100),
                    radial_qualification=dict(qualified=label!="unresolved"),transverse_label=label,
                    result_artifact=dict(path="manufactured",sha256="0"*64,bytes=0))
    for stop_unresolved in (False,True):
        fixtures={}
        coarse=[point(8,"positive"),point(12,"negative"),point(64,"negative"),point("inf","negative")]
        # Infinity is metadata only for this decision fixture.
        coarse[-1]["theta"]=np.zeros(16)
        def writer(name,value):
            path=ART/"REFINEMENT_PLAN.json" if name=="initial" else ART/"refinement"/(name+"_PLAN.json")
            fixtures[str(path)]=native(value)
        def decision_writer(i,value):fixtures[str(ART/"refinement"/f"decision_{i:02d}.json")]=native(value)
        def solve(pid,K,start,kind,context):
            label="unresolved" if stop_unresolved else ("positive" if K<=10 else "negative")
            row=point(K,label,pid);row.update(theta0=start,start_context=context,kind=kind)
            return native(row)
        refined=manufactured_continuation.refine(native(coarse),solve,writer,decision_writer)
        with patch.object(sys.modules[__name__],"read",side_effect=lambda path:fixtures[str(path)]):
            check_refinement(native(coarse),refined,checks)
    coarse=native([point(8,"positive"),point(12,"unresolved"),point(16,"negative")])
    fixtures={}
    refined=manufactured_continuation.refine(coarse,lambda *args: (_ for _ in ()).throw(AssertionError("no-pair must not solve")),writer,decision_writer)
    with patch.object(sys.modules[__name__],"read",side_effect=lambda path:fixtures[str(path)]):
        check_refinement(coarse,refined,checks)
    return dict(passed=info["quad_success"] and max(finite_errors.values())<5e-10 and max(herror,gerror)<2e-8 and not checks.failures,
        manufactured_only=True,scientific_inputs_read=False,new_random_draws=0,
        reused_audit_sha256=sha(base.__file__),finite_K2_direct_enumeration_errors=finite_errors,
        posterior_H_gradient_difference=herror,posterior_gradient_loss_difference=gerror,
        execution_assertions=checks.count,execution_failures=checks.failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test","--selftest",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_test();print(json.dumps(native(result),indent=2,allow_nan=False));return 0 if result["passed"] else 1
    if args.run and RECEIPT.exists():raise FileExistsError("AUDIT.json exists and will not be replaced")
    previous=read(RECEIPT) if args.check else None
    if previous is not None and previous["source_sha256"]!=sha(__file__):raise ValueError("audit source changed after receipt")
    try:result=perform()
    except BaseException as error:
        result=dict(passed=False,source_sha256=sha(__file__),failures=[dict(exception=str(error),traceback=traceback.format_exc())])
    if args.run:
        result["created_utc"]=datetime.now(timezone.utc).isoformat()
        with RECEIPT.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(native(result),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    elif result["passed"]:
        for key in ("passed","source_sha256","reused_audit_sha256","freeze_sha256","audit_freeze_sha256","input_hashes","failures",
            "checks","numeric_elements_compared","numerical_maxima","initial_classification","point_checks","refinement_checks","cross_checks","scope"):
            if native(result[key])!=previous[key]:raise ValueError("repeat differs in stable field: "+key)
    print(json.dumps(native({k:result[k] for k in ("passed","checks","seconds","failures") if k in result}),allow_nan=False),flush=True)
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
