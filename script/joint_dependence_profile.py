"""Locked marginal-preserving profiles of the four joint-learning streams.

--self-test uses only explicit deterministic fixtures and writes no files.
--freeze requires all four completed/qualified training streams; --rep starts
one prescribed profile evaluation only after that freeze. No existing output
is replaced. This module never trains or uses teacher risk to choose profiles.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback

import numpy as np
import joint_dependence_kernel as kernel
import joint_dependence_train as training

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
DESIGN = ROOT / "docs/08_research_研究工作/JointDependence_v1_20261007/PROFILE_DESIGN.md"
PREDECLARATION = OUT / "PROFILE_PREDECLARATION.json"
FREEZE = OUT / "PROFILE_FREEZE.json"
PROFILE_CONFIG = dict(reps=4, dimension=8, latent=2, banks=4096, particles=[8,64],
                      seed=202610070300, seed_role=0, row_norm_threshold=1e-10,
                      marginal_tolerance=2e-7, evaluation_block=128,
                      bias_bounds=[-2.,2.], loading_bounds=[-2.,2.])
PROFILE_NAMES = ("REF", "H8", "H64")
TARGET_NAMES = ("empirical", "population")


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native(x):
    if isinstance(x,np.ndarray): return native(x.tolist())
    if isinstance(x,np.generic): return native(x.item())
    if isinstance(x,dict): return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):
        raise ValueError("nonfinite JSON value must be handled explicitly")
    return x


def save(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(native(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def save_arrays(path,**arrays):
    with Path(path).open("xb") as f: np.savez_compressed(f,**arrays)


def checked_parameters(candidate,m=8):
    theta=np.asarray(candidate["theta"],dtype=np.float64)
    b=np.asarray(candidate["b"],dtype=np.float64)
    W=np.asarray(candidate["W"],dtype=np.float64)
    if theta.shape!=(3*m,) or b.shape!=(m,) or W.shape!=(m,2):
        raise ValueError("candidate requires theta=b then row-major W, with shapes 24/8/8x2")
    if not np.isfinite(theta).all() or not np.array_equal(theta,np.r_[b,W.ravel()]):
        raise ValueError("candidate parameters are nonfinite or redundant fields disagree")
    return b.copy(),W.copy()


def box_record(b,W):
    bad_b=np.flatnonzero((b<-2)|(b>2))
    bad_W=np.argwhere((W<-2)|(W>2))
    return dict(feasible=not len(bad_b) and not len(bad_W),
                bias_violations=bad_b,loading_violations=bad_W,
                maximum_excess=max(0.,float(np.max(np.abs(b)))-2.,float(np.max(np.abs(W)))-2.),
                projected_or_repaired=False)


def construct_profiles(locked_candidates):
    """Pure geometry: accepts no teacher, PMF, risk, or score input."""
    by_name={c["name"]:c for c in locked_candidates}
    if len(by_name)!=len(locked_candidates) or not {"REF","K8","K64"}<=set(by_name):
        raise ValueError("unique locked REF/K8/K64 candidates required")
    rb,rW=checked_parameters(by_name["REF"])
    rn=np.linalg.norm(rW,axis=1)
    rows=[dict(name="REF",defined=True,b=rb,W=rW,theta=np.r_[rb,rW.ravel()],
               row_norms=rn,box=box_record(rb,rW),source_direction="REF")]
    for donor_name,profile_name in (("K8","H8"),("K64","H64")):
        _,donor=checked_parameters(by_name[donor_name]);dn=np.linalg.norm(donor,axis=1)
        bad_ref=np.flatnonzero(rn<=PROFILE_CONFIG["row_norm_threshold"])
        bad_donor=np.flatnonzero(dn<=PROFILE_CONFIG["row_norm_threshold"])
        if len(bad_ref) or len(bad_donor):
            rows.append(dict(name=profile_name,defined=False,source_direction=donor_name,
                reason="row norm does not exceed the fixed 1e-10 threshold",
                undefined_reference_rows=bad_ref,undefined_donor_rows=bad_donor,
                reference_row_norms=rn,donor_row_norms=dn))
            continue
        W=rn[:,None]*(donor/dn[:,None])
        rows.append(dict(name=profile_name,defined=True,b=rb.copy(),W=W,theta=np.r_[rb,W.ravel()],
            row_norms=np.linalg.norm(W,axis=1),reference_row_norms=rn,donor_row_norms=dn,
            maximum_reference_norm_difference=float(np.max(np.abs(np.linalg.norm(W,axis=1)-rn))),
            reference_biases_exactly_equal=bool(np.array_equal(rb,rows[0]["b"])),
            box=box_record(rb,W),source_direction=donor_name,
            teacher_alignment_used=False,common_rotation_alignment_used=False))
    return native(rows)


def source_folder(rep,attempt):
    base=OUT/f"rep_{rep:02d}"
    return base if attempt==1 else base/f"attempt_{attempt:03d}"


def load_training_metadata(rep,attempt):
    """Check eligibility and identities without selecting on any risk value."""
    folder=source_folder(rep,attempt)
    named={n:folder/n for n in ("PARAMS.json","CANDIDATES_LOCK.json","RESULTS.json","ENDPOINT_QUALIFICATION.json")}
    missing=[rel(p) for p in named.values() if not p.is_file()]
    if missing: raise ValueError("incomplete source stream: "+", ".join(missing))
    if (folder/"FAILURE.json").exists(): raise ValueError("source stream retains an unresolved FAILURE.json")
    p,l,r,q=(read(named[n]) for n in ("PARAMS.json","CANDIDATES_LOCK.json","RESULTS.json","ENDPOINT_QUALIFICATION.json"))
    if r["status"]!="completed" or r["rep"]!=rep or p["rep"]!=rep or l["rep"]!=rep:
        raise ValueError("source stream did not complete with the specified rep")
    if p["attempt"]!=attempt or r["attempt"]!=attempt or p["config"]!=training.CONFIG:
        raise ValueError("training attempt/configuration mismatch")
    if r["params_sha256"]!=sha(named["PARAMS.json"]) or l["params_sha256"]!=r["params_sha256"]:
        raise ValueError("source PARAMS binding mismatch")
    if r["candidates_lock_sha256"]!=sha(named["CANDIDATES_LOCK.json"]) or q["candidates_lock_sha256"]!=r["candidates_lock_sha256"]:
        raise ValueError("source candidate-lock binding mismatch")
    tf=training.freeze_path(attempt)
    if p["freeze_sha256"]!=sha(tf) or r["freeze_sha256"]!=sha(tf):
        raise ValueError("source training freeze mismatch")
    if p["source_sha256"]!=sha(training.__file__) or p["kernel_sha256"]!=sha(kernel.__file__):
        raise ValueError("current training/kernel code differs from the saved run")
    if r["candidates"]!=l["candidates"] or r["endpoints"]!=q["endpoints"]:
        raise ValueError("source candidates/endpoints differ from their locks")
    expected=("K8","K64","GH","REF","IND")
    if tuple(c["name"] for c in l["candidates"])!=expected:
        raise ValueError("unexpected source candidate names/order")
    if not q["all_qualified"] or not all(x["qualified"] for x in q["endpoints"]):
        raise ValueError("source endpoint numerical qualification failed")
    if tuple(x["name"] for x in q["endpoints"])!=expected:
        raise ValueError("source endpoint qualification names/order mismatch")
    if q["teacher_pmf_error"]>training.CONFIG["pmf_tolerance"] or q["teacher_mass_error"]>training.CONFIG["mass_tolerance"]:
        raise ValueError("source teacher numerical qualification failed")
    if r["training"]["completed_steps"]!=training.CONFIG["steps"] or not all(x["qualified"] for x in r["training"]["path_checks"]):
        raise ValueError("source training/path qualification incomplete")
    if not l["locked_before_teacher_risk_and_independent_cross_evaluation"] or p["teacher_used_by_training_function"]:
        raise ValueError("source learner/teacher lock boundary failed")
    frequency=np.asarray(p["frequency"],dtype=np.float64)
    if frequency.shape!=(256,) or np.any(frequency<0) or not np.isfinite(frequency).all() or abs(float(frequency.sum())-1)>1e-12:
        raise ValueError("invalid empirical pattern frequencies")
    if not np.array_equal(frequency,np.asarray(l["frequency"])) or not np.array_equal(frequency,np.asarray(r["frequency"])):
        raise ValueError("source frequency records differ")
    for c in l["candidates"]: checked_parameters(c)
    files={rel(path):sha(path) for path in named.values()}
    # Bind the source NPZs and all other saved records by their existing hashes.
    for entry in r["artifacts"].values():
        path=(ROOT/entry["path"]).resolve()
        if not path.is_relative_to(folder.resolve()): raise ValueError("source artifact escaped its stream folder")
        if path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:
            raise ValueError("source artifact hash/size mismatch: "+entry["path"])
        files[rel(path)]=entry["sha256"]
    for name,digest in (("INPUTS.npz",p["inputs_sha256"]),("TRAINING.npz",l["training_sha256"]),
                        ("REFERENCE_TRAINING.npz",l["reference_training_sha256"])):
        if files.get(rel(folder/name))!=digest:
            raise ValueError("source lock-to-NPZ binding mismatch: "+name)
    if p["inputs"]!=rel(folder/"INPUTS.npz"):
        raise ValueError("source input NPZ path mismatch")
    return dict(rep=rep,training_attempt=attempt,folder=rel(folder),sources=files,
                profiles=construct_profiles(l["candidates"]),frequency=frequency.tolist(),
                eligibility=dict(training_completed=True,path_checks_qualified=True,all_endpoints_qualified=True,
                    REF_numerically_qualified=True,REF_optimizer_success=bool(l["reference_optimizer"]["success"]),
                    REF_global_MLE_claimed=False),
                source_candidate_lock_created_utc=l["created_utc"])


def new_counts():
    return dict(new_observations=0,optimizer_updates=0,new_prior_vectors=0,new_Gaussian_scalars=0,
        GH_PMF_calls=0,GH_loss_gradient_calls=0,GH_PMF_nodes=0,GH_loss_gradient_nodes=0,
        GH_PMF_node_pattern_products=0,GH_loss_gradient_node_pattern_products=0,
        GH_logit_scalars=0,MC_kernel_calls=0,MC_bank_candidate_K_target_cells=0,
        MC_particle_pattern_products=0,MC_logit_scalars=0,failed_kernel_calls=0,
        teacher_GH_nodes_newly_evaluated=0,new_independent_training_streams=0)


def endpoint_reference(candidate,frequency,teacher_variants,counts,config):
    b,W=checked_parameters(candidate);labels=kernel.patterns(len(b)).astype(float)
    main=config["endpoint_orders"].index(config["main_reference_order"])
    P=np.asarray(teacher_variants[main][2],dtype=np.float64)
    variants=[]
    for order,angle in [(o,0.) for o in config["endpoint_orders"]]+[(config["rotated_order"],config["rotation"])]:
        try:
            q=kernel.model_pmf(b,W,order,angle)
        except BaseException:
            counts["failed_kernel_calls"]+=1;raise
        counts["GH_PMF_calls"]+=1;counts["GH_PMF_nodes"]+=order**2
        counts["GH_PMF_node_pattern_products"]+=order**2*len(labels)
        counts["GH_logit_scalars"]+=order**2*len(b)
        if not np.isfinite(q).all() or np.any(q<=0): raise FloatingPointError("nonpositive/nonfinite profile PMF")
        try:
            loss,gb,gW=kernel.exact_loss_grad(b,W,frequency,order,angle)
        except BaseException:
            counts["failed_kernel_calls"]+=1;raise
        counts["GH_loss_gradient_calls"]+=1;counts["GH_loss_gradient_nodes"]+=order**2
        counts["GH_loss_gradient_node_pattern_products"]+=order**2*int(np.count_nonzero(frequency))
        counts["GH_logit_scalars"]+=order**2*len(b)
        variants.append(dict(order=order,rotation=angle,pmf=q,empirical_NLL=loss,
            risk_KL=float(P@(np.log(P)-np.log(q))),gradient=np.r_[gb,gW.ravel()],marginals=q@labels))
    chosen=variants[main];q=chosen["pmf"]
    pe=max(float(np.max(abs(v["pmf"]-q))) for v in variants)
    mass=max(abs(float(v["pmf"].sum())-1) for v in variants)
    re=max(abs(v["risk_KL"]-chosen["risk_KL"]) for v in variants)
    te=max(abs(float(tq@(np.log(tq)-np.log(q)))-chosen["risk_KL"]) for _,_,tq in teacher_variants)
    le=max(abs(v["empirical_NLL"]-chosen["empirical_NLL"]) for v in variants)
    ge=max(float(np.max(abs(v["gradient"]-chosen["gradient"]))) for v in variants)
    qualified=(pe<=config["pmf_tolerance"] and mass<=config["mass_tolerance"]
        and max(re,te)<=config["risk_tolerance"] and le<=config["path_loss_tolerance"]
        and ge<=config["path_gradient_tolerance"])
    return native(dict(name=candidate["name"],qualified=qualified,main_order=config["main_reference_order"],
        empirical_NLL=chosen["empirical_NLL"],risk_KL=chosen["risk_KL"],pmf=q,marginals=chosen["marginals"],
        gradient=chosen["gradient"],variants=variants,pmf_max_discrepancy=pe,mass_max_error=mass,
        risk_max_discrepancy=max(re,te),NLL_max_discrepancy=le,gradient_max_discrepancy=ge,
        box=candidate["box"],original_box_claim_allowed=candidate["box"]["feasible"]))


def qualify_profiles(candidates,frequency,teacher_variants,counts,config):
    rows=[endpoint_reference(c,frequency,teacher_variants,counts,config) for c in candidates]
    reference=next(r for r in rows if r["name"]=="REF")
    ref_marginals=np.asarray(reference["marginals"])
    for row in rows:
        row["maximum_marginal_difference_from_REF"]=max(float(np.max(abs(np.asarray(v["marginals"])-ref_marginals))) for v in row["variants"])
        row["marginals_qualified"]=row["maximum_marginal_difference_from_REF"]<=PROFILE_CONFIG["marginal_tolerance"]
    return rows


def paired_summary(x):
    x=np.asarray(x,dtype=np.float64);sd=float(x.std(ddof=1))
    return dict(mean=float(x.mean()),sample_sd=sd,MCSE=sd/math.sqrt(len(x)),banks=len(x),
                scope="fixed locked endpoints and targets, not independent training repeats")


def cross_evaluate(candidates,frequency,P,prior,counts,block=128):
    losses=np.empty((len(prior),len(candidates),2,2),dtype=np.float64)
    for ci,candidate in enumerate(candidates):
        b,W=checked_parameters(candidate)
        for ki,K in enumerate((8,64)):
            for ti,target in enumerate((frequency,P)):
                try:
                    values=kernel.mc_bank_losses(b,W,target,prior[:,:K,:],bank_chunk_size=block)
                except BaseException:
                    counts["failed_kernel_calls"]+=1;raise
                losses[:,ci,ki,ti]=values
                counts["MC_kernel_calls"]+=1
                counts["MC_bank_candidate_K_target_cells"]+=len(prior)
                counts["MC_particle_pattern_products"]+=len(prior)*K*int(np.count_nonzero(target))
                counts["MC_logit_scalars"]+=len(prior)*K*len(b)
    if not np.isfinite(losses).all(): raise FloatingPointError("nonfinite profile MC losses")
    names=[c["name"] for c in candidates];ri=names.index("REF")
    comparisons=[name for name in ("H8","H64") if name in names]
    paired=np.stack([losses[:,names.index(n)]-losses[:,ri] for n in comparisons],axis=1)
    means=[];differences=[]
    for ki,K in enumerate((8,64)):
        for ti,target in enumerate(TARGET_NAMES):
            for ci,name in enumerate(names):
                means.append(dict(candidate=name,K=K,target=target,**paired_summary(losses[:,ci,ki,ti])))
            for ci,name in enumerate(comparisons):
                differences.append(dict(left=name,right="REF",K=K,target=target,**paired_summary(paired[:,ci,ki,ti])))
    return losses,paired,comparisons,dict(means=means,paired_differences=differences,
        statistical_family_claimed=False,selection_or_training_changed=False)


def self_test():
    """Pure manufactured qualification; no files or scientific RNG are used."""
    b=np.linspace(-.08,.08,8);angles=np.linspace(-.6,.7,8)
    W=.18*np.stack((np.cos(angles),np.sin(angles)),axis=1)
    def item(name,bb,ww):return dict(name=name,b=bb.tolist(),W=ww.tolist(),theta=np.r_[bb,ww.ravel()].tolist())
    sources=[item("REF",b,W),item("K8",b+.2,.07*np.stack((np.cos(angles*.6),np.sin(angles*.6)),axis=1)),
             item("K64",b-.1,.4*np.stack((np.cos(angles*.9),np.sin(angles*.9)),axis=1))]
    before=copy.deepcopy(sources);profiles=construct_profiles(sources)
    for row in profiles:
        assert row["defined"] and row["box"]["feasible"]
        np.testing.assert_array_equal(row["b"],b)
        np.testing.assert_allclose(np.linalg.norm(row["W"],axis=1),np.linalg.norm(W,axis=1),rtol=0,atol=1e-15)
    assert sources==before
    angle=.37;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    rotated=copy.deepcopy(sources)
    for j in (1,2):
        bb,ww=checked_parameters(rotated[j]);rotated[j]=item(rotated[j]["name"],bb,ww@rotation.T)
    rotated_profiles=construct_profiles(rotated)
    for j in (1,2):
        np.testing.assert_allclose(rotated_profiles[j]["W"],np.asarray(profiles[j]["W"])@rotation.T,rtol=0,atol=2e-16)
    zero=copy.deepcopy(sources);zero[1]=item("K8",b,np.zeros((8,2)))
    assert construct_profiles(zero)[1]["defined"] is False and construct_profiles(zero)[2]["defined"] is True
    large=copy.deepcopy(sources);large[0]=item("REF",b,np.full((8,2),2.));large[1]=item("K8",b,np.tile([1.,0.],(8,1)))
    assert construct_profiles(large)[1]["box"]["feasible"] is False
    config=copy.deepcopy(training.CONFIG);config.update(endpoint_orders=[8,10,12],main_reference_order=10,rotated_order=12)
    empirical=np.arange(1.,257.);empirical/=empirical.sum()
    teacher=[(o,a,kernel.model_pmf(b,W,o,a)) for o,a in [(8,0.),(10,0.),(12,0.),(12,config["rotation"])]]
    counts=new_counts();reference=qualify_profiles(profiles,empirical,teacher,counts,config)
    assert all(r["qualified"] and r["marginals_qualified"] for r in reference)
    prior=np.sin(np.arange(4*64*2,dtype=float)*.13).reshape(4,64,2)
    losses,paired,names,summary=cross_evaluate(profiles,empirical,teacher[1][2],prior,counts,block=2)
    assert losses.shape==(4,3,2,2) and paired.shape==(4,2,2,2) and names==["H8","H64"]
    np.testing.assert_array_equal(paired[:,0],losses[:,1]-losses[:,0])
    np.testing.assert_array_equal(paired[:,1],losses[:,2]-losses[:,0])
    direct=kernel.mc_bank_losses(b,np.asarray(profiles[1]["W"]),empirical,prior[:,:8],bank_chunk_size=2)
    np.testing.assert_array_equal(direct,losses[:,1,0,0])
    assert counts["MC_kernel_calls"]==12 and counts["new_prior_vectors"]==0
    assert counts["MC_particle_pattern_products"]==4*3*(8+64)*(256+256)
    assert len(summary["paired_differences"])==8
    return dict(passed=True,manufactured_only=True,new_scientific_random_draws=0,
        source_sha256=sha(__file__),kernel_sha256=sha(kernel.__file__),training_source_sha256=sha(training.__file__),
        prior_formula="sin(arange(4*64*2)*.13)",profiles_preserve_reference_bias_and_norm=True,
        donor_bias_and_scale_not_transferred=True,undefined_rows_explicit=True,out_of_box_not_projected=True,
        paired_axes_and_prefix_verified=True,counts_verified=True,common_donor_rotation_covariance=True,
        maximum_fixture_marginal_difference=max(r["maximum_marginal_difference_from_REF"] for r in reference))


def freeze(attempt):
    if FREEZE.exists(): raise FileExistsError("PROFILE_FREEZE already exists; it will not be replaced")
    try:
        pre=read(PREDECLARATION)
        if pre["design_sha256"]!=sha(DESIGN) or pre["training_result_files_present"]!=[]:
            raise ValueError("profile predeclaration does not bind the pre-training design")
        streams=[load_training_metadata(rep,attempt) for rep in range(4)]
        for stream in streams:
            if datetime.fromisoformat(pre["created_utc"])>datetime.fromisoformat(stream["source_candidate_lock_created_utc"]):
                raise ValueError("profile design was not recorded before the source output locks")
        qualification=self_test()
        sources=[Path(__file__),Path(kernel.__file__),Path(training.__file__),training.DESIGN,
                 training.freeze_path(attempt),DESIGN,PREDECLARATION]
        save(FREEZE,dict(created_utc=now(),config=PROFILE_CONFIG,reference_config=training.CONFIG,
            training_attempt=attempt,sources={rel(p):sha(p) for p in sources},streams=streams,
            manufactured_qualification=qualification,profile_parameters_locked_before_profile_risk=True,
            staged_exploratory=True,confirmatory=False,new_RNG_before_profile_freeze=0,
            original_angular_and_training_results_unchanged=True))
        print(json.dumps(dict(frozen=True,path=rel(FREEZE),sha256=sha(FREEZE))),flush=True)
    except BaseException as error:
        rejected=OUT/"PROFILE_FREEZE_REJECTED.json"
        if not rejected.exists():
            save(rejected,dict(created_utc=now(),reason=str(error),traceback=traceback.format_exc(),
                requested_training_attempt=attempt,new_prior_vectors=0,scientific_evaluation_started=False))
        raise


def verify_freeze():
    f=read(FREEZE)
    if f["config"]!=PROFILE_CONFIG or f["reference_config"]!=training.CONFIG:
        raise ValueError("profile/reference configuration differs from the freeze")
    for name,digest in f["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("changed frozen source: "+name)
    for stream in f["streams"]:
        for name,digest in stream["sources"].items():
            if sha(ROOT/name)!=digest:raise ValueError("changed frozen training artifact: "+name)
    if [s["rep"] for s in f["streams"]]!=list(range(4)):
        raise ValueError("all four source streams must be bound")
    return f


def run(rep):
    if rep not in range(4): raise ValueError("rep must be 0,1,2,3")
    frozen=verify_freeze();stream=frozen["streams"][rep];source=ROOT/stream["folder"]
    folder=OUT/"profile"/f"rep_{rep:02d}";folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();counts=new_counts()
    try:
        save(folder/"RUN_START.json",dict(created_utc=now(),rep=rep,profile_freeze_sha256=sha(FREEZE),
              source_stream=stream["folder"],new_prior_generated=False))
        source_lock=read(source/"CANDIDATES_LOCK.json")
        profiles=construct_profiles(source_lock["candidates"])
        if profiles!=stream["profiles"]:raise ValueError("profile parameters differ from frozen geometry")
        frequency=np.asarray(stream["frequency"],dtype=np.float64)
        candidates=[c for c in profiles if c["defined"]]
        lock=folder/"CANDIDATES_LOCK.json"
        save(lock,dict(created_utc=now(),rep=rep,candidates=profiles,frequency=frequency,
            source_candidate_lock_sha256=sha(source/"CANDIDATES_LOCK.json"),profile_freeze_sha256=sha(FREEZE),
            locked_before_profile_teacher_risk_and_prior=True,teacher_alignment_used=False))
        if len(candidates)==1:
            save(folder/"RESULTS.json",dict(status="skipped_all_profiles_undefined",rep=rep,
                profile_freeze_sha256=sha(FREEZE),source_stream=stream["folder"],
                profiles=profiles,candidates_lock_sha256=sha(lock),resources=counts,seconds=time.perf_counter()-started,
                reason="No H8/H64 profile has all required nonzero rows; no teacher risk or new prior evaluated."))
            print(json.dumps(dict(rep=rep,status="skipped_all_profiles_undefined",folder=rel(folder),new_prior_vectors=0)),flush=True)
            return
        # Teacher probabilities are consumed only after the profile lock above.
        original_qualification=read(source/"ENDPOINT_QUALIFICATION.json")
        teacher=[(v["order"],v["rotation"],np.asarray(v["pmf"],dtype=float)) for v in original_qualification["teacher_variants"]]
        main=training.CONFIG["endpoint_orders"].index(training.CONFIG["main_reference_order"])
        P=teacher[main][2]
        if not np.array_equal(P,np.asarray(original_qualification["teacher_pmf"])):
            raise ValueError("saved teacher main PMF mismatch")
        if P.shape!=(256,) or not np.isfinite(P).all() or np.any(P<=0) or abs(float(P.sum())-1)>training.CONFIG["mass_tolerance"]:
            raise ValueError("saved teacher PMF no longer meets numerical prerequisites")
        rows=qualify_profiles(candidates,frequency,teacher,counts,training.CONFIG)
        qualified=all(r["qualified"] and r["marginals_qualified"] for r in rows)
        save(folder/"ENDPOINT_QUALIFICATION.json",dict(candidates_lock_sha256=sha(lock),
            source_teacher_qualification_sha256=sha(source/"ENDPOINT_QUALIFICATION.json"),
            teacher_pmf=P,teacher_reference_reused=True,endpoints=rows,all_qualified=qualified,
            marginal_tolerance=PROFILE_CONFIG["marginal_tolerance"]))
        if not qualified:
            save(folder/"RESULTS.json",dict(status="skipped_numerical_qualification",rep=rep,
                profile_freeze_sha256=sha(FREEZE),source_stream=stream["folder"],
                profiles=profiles,endpoints=rows,candidates_lock_sha256=sha(lock),resources=counts,seconds=time.perf_counter()-started,
                reason="Fixed GH/rotation or marginal gate failed; no MC prior generated or gates changed."))
            print(json.dumps(dict(rep=rep,status="skipped_numerical_qualification",folder=rel(folder),new_prior_vectors=0)),flush=True)
            return
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([PROFILE_CONFIG["seed"],rep,0])))
        prior=rng.standard_normal((4096,64,2))
        counts["new_prior_vectors"]=4096*64;counts["new_Gaussian_scalars"]=4096*64*2
        save_arrays(folder/"EVALUATION_PRIOR.npz",prior=prior)
        losses,paired,comparison_names,cross=cross_evaluate(candidates,frequency,P,prior,counts,
            block=PROFILE_CONFIG["evaluation_block"])
        save_arrays(folder/"EVALUATION.npz",loss=losses,paired_loss=paired,prior=prior,
            candidate_names=np.asarray([c["name"] for c in candidates]),comparison_left=np.asarray(comparison_names),
            comparison_right=np.asarray(["REF"]*len(comparison_names)),particles=np.array([8,64]),
            target_names=np.asarray(TARGET_NAMES),frequency=frequency,teacher_pmf=P)
        by_name={r["name"]:r for r in rows};exact=[]
        for name in comparison_names:
            exact.append(dict(left=name,right="REF",risk_KL_difference=by_name[name]["risk_KL"]-by_name["REF"]["risk_KL"],
                empirical_NLL_difference=by_name[name]["empirical_NLL"]-by_name["REF"]["empirical_NLL"],
                marginal_difference=by_name[name]["maximum_marginal_difference_from_REF"],
                original_box_claim_allowed=by_name[name]["box"]["feasible"]))
        verify_freeze()
        save(folder/"RESULTS.json",dict(status="completed",rep=rep,profile_freeze_sha256=sha(FREEZE),
            candidates_lock_sha256=sha(lock),source_stream=stream["folder"],profiles=profiles,endpoints=rows,
            frequency=frequency,teacher_pmf=P,exact_differences=exact,cross_evaluation=cross,
            main_contrast="H8 minus REF at K8 empirical/population finite loss and observation KL",
            main_contrast_available="H8" in comparison_names,seed=[PROFILE_CONFIG["seed"],rep,0],
            evaluation_prior_sha256=array_hash(prior),resources=counts,seconds=time.perf_counter()-started,
            artifacts={p.name:dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()},
            claims=dict(staged_exploratory=True,formal_family=False,MCSE_conditions_on_locked_outputs=True,
                fixed_one_dimensional_marginals=True,joint_observation_law_preserved=False,
                all_training_risk_difference_explained=False,universal_K_improvement=False,
                reference_is_global_MLE=False,no_optimizer_updates=True,source_results_changed=False)))
        print(json.dumps(dict(completed=True,rep=rep,folder=rel(folder),resources=counts,
             seconds=time.perf_counter()-started)),flush=True)
    except BaseException as error:
        save(folder/"FAILURE.json",dict(rep=rep,profile_freeze_sha256=sha(FREEZE),reason=str(error),
            traceback=traceback.format_exc(),resources=counts,seconds=time.perf_counter()-started,
            prior_regenerated_or_retry_started=False,failed_kernel_internal_partial_work_not_counted=True))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test",action="store_true");mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--rep",type=int);parser.add_argument("--training-attempt",type=int,default=1,
        help="source training attempt for --freeze only; --rep always uses the recorded PROFILE_FREEZE")
    args=parser.parse_args()
    if args.training_attempt<1:parser.error("--training-attempt must be positive")
    if args.self_test:print(json.dumps(self_test(),sort_keys=True),flush=True)
    elif args.freeze:freeze(args.training_attempt)
    else:run(args.rep)
