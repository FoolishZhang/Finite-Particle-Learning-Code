"""Frozen per-observation SCBM loss-API experiment; no neural training.

schema is RNG-free. run requires the root's source/config/input freeze. Native
float32 API readouts remain primary; stable64 and zero/floor are controls only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t
import scbminv_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/scbm_invariance/RUN"
SOURCE=Path(__file__).resolve()
CONFIG=dict(streams=16,observations=4096,aux_observations=1024,batch_size=64,
    K_main=100,K_aux=400,paddings=[0,64,256],candidate_t=[-.75,0.,.75],
    seed_base=202610030100,roles=dict(data_gaussian=0,data_active_uniform=1,data_nuisance_uniform=2,
                                  prior_active=3,prior_nuisance=4),
    primary_alpha=.05,primary_family_size=4,primary_SE_thresholds=[5e-4,1e-3,2e-3,5e-4],
    numerical_rows=64,numerical_score_tolerance=5e-4,numerical_gradient_tolerance=5e-5,
    scale_control_paddings=[64,256],scale_control_values=[0.,1e-6],workers=1)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def save(p,x,new=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x" if new else "w",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def statistics(values,critical=None):
    a=np.asarray(values,float)
    out=dict(mean=float(a.mean()),SE=float(a.std(ddof=1)/math.sqrt(len(a))),streams=len(a))
    if critical is not None:out.update(lower=out["mean"]-critical*out["SE"],upper=out["mean"]+critical*out["SE"])
    return out

def normalize_gradient(result,padding):
    x=kernel.flatten_gradient(result).copy()
    if padding:x[-1]/=padding
    return x

def evaluate_batch(active,noise,y,yn,K,do_numerical):
    args=(active[:K],noise[:K],y,yn)
    values=np.empty((3,3));grad=np.empty((3,11));direction=np.empty(3)
    zeros=np.zeros((3,3),dtype=np.int64);ones=zeros.copy();numbers=zeros.copy();maxlog=np.empty((3,3))
    actual_api={};calls=0
    for ni,n in enumerate(CONFIG["paddings"]):
        for ti,t in enumerate(CONFIG["candidate_t"]):
            value=kernel.evaluate(*args,t=t,padding=n,need_gradient=t==0.)
            actual_api[ni,ti]=value;values[ni,ti]=value["value"]
            d=value["diagnostics"];calls+=d["official_API_calls"]
            zeros[ni,ti]=d["sigmoid_zero_count"];ones[ni,ti]=d["sigmoid_one_count"]
            numbers[ni,ti]=d["probability_evaluations"];maxlog[ni,ti]=d["maximum_absolute_logit"]
            if t==0.:
                grad[ni]=kernel.flatten_gradient(value);direction[ni]=value["gradient"]["correlation_direction"]
    factor_values=[];factor_gradient=None
    for t in CONFIG["candidate_t"]:
        value=kernel.evaluate(*args,t=t,padding=0,factorized=True,need_gradient=t==0.)
        calls+=value["diagnostics"]["official_API_calls"]
        factor_values.append(value["value"])
        if t==0.:factor_gradient=kernel.flatten_gradient(value)
    assert np.array_equal(np.asarray(factor_values),values[0]),"Same active API must be exactly reused in the factorized comparison"
    assert np.array_equal(factor_gradient[:10],grad[0,:10])
    numerical=[];controls=[];stable_calls=0
    if do_numerical:
        for ni,n in enumerate(CONFIG["paddings"]):
            for ti,t in enumerate(CONFIG["candidate_t"]):
                stable=kernel.evaluate(*args,t=t,padding=n,path="stable_logits64",need_gradient=t==0.)
                stable_calls+=1;api=actual_api[ni,ti]
                error=api["value"]-stable["value"]
                row=dict(K=K,padding=n,t=t,official32_score=api["value"],stable64_score=stable["value"],
                    score_difference=error,score_pass=abs(error)<=CONFIG["numerical_score_tolerance"])
                if t==0.:
                    diff=normalize_gradient(api,n)-normalize_gradient(stable,n)
                    row.update(official32_gradient=api["gradient"],stable64_gradient=stable["gradient"],
                        normalized_gradient_difference=diff.tolist(),maximum_normalized_gradient_difference=float(np.max(abs(diff))),
                        gradient_pass=bool(np.max(abs(diff))<=CONFIG["numerical_gradient_tolerance"]),
                        correlation_direction_difference=api["gradient"]["correlation_direction"]-stable["gradient"]["correlation_direction"])
                numerical.append(row)
        for scale in CONFIG["scale_control_values"]:
            for n in CONFIG["scale_control_paddings"]:
                for ti,t in enumerate(CONFIG["candidate_t"]):
                    controlled=kernel.evaluate(*args,t=t,padding=n,nuisance_scale=scale,need_gradient=t==0.)
                    calls+=controlled["diagnostics"]["official_API_calls"]
                    factor_score=factor_values[ti]+n*math.log(2.)
                    delta=controlled["value"]-factor_score
                    row=dict(K=K,padding=n,t=t,scale=scale,official32_score=controlled["value"],
                        factorized_score_from_active_API=factor_score,value_difference=delta,
                        value_qualified=abs(delta)<=CONFIG["numerical_score_tolerance"],
                        diagnostics=controlled["diagnostics"],scope="Predetermined numerical boundary control; not learned variance or an attained original zero-floor parameter")
                    if t==0.:
                        active_error=kernel.flatten_gradient(controlled)[:10]-factor_gradient[:10]
                        row.update(maximum_active_gradient_difference=float(np.max(abs(active_error))),
                            active_gradient_qualified=bool(np.max(abs(active_error))<=CONFIG["numerical_gradient_tolerance"]),
                            raw_scale_gradient_at_control=controlled["gradient"]["nuisance_scale"],
                            scale_gradient_scope="Fixed-realization boundary scale derivative is not part of the active factorization identity")
                    controls.append(row)
    arrays=dict(raw_values=values,raw_gradient=grad,raw_correlation_direction=direction,
        factorized_active_values=np.asarray(factor_values),factorized_active_gradient=factor_gradient,
        sigmoid_zero_counts=zeros,sigmoid_one_counts=ones,probability_evaluations=numbers,
        maximum_absolute_logits=maxlog)
    return arrays,numerical,controls,dict(official_calls=calls,stable64_calls=stable_calls)

def scope_summary(arrays,prefix,blocks):
    out={}
    for key in ["raw_values","raw_gradient","raw_correlation_direction","factorized_active_values","factorized_active_gradient"]:
        out[key]=np.asarray(arrays[prefix+"_"+key])[:blocks].mean(axis=0).tolist()
    val=np.asarray(out["raw_values"])
    out["raw_low_minus_truth"]=(val[:,0]-val[:,1]).tolist()
    out["raw_high_minus_truth"]=(val[:,2]-val[:,1]).tolist()
    factor=np.asarray(out["factorized_active_values"])
    out["factorized_low_minus_truth"]=float(factor[0]-factor[1])
    out["factorized_high_minus_truth"]=float(factor[2]-factor[1])
    out["factorized_score_with_analytic_nlog2"]=(factor[None,:]+np.asarray(CONFIG["paddings"])[:,None]*math.log(2)).tolist()
    out["rows"]=blocks*CONFIG["batch_size"]
    out["scope"]="Batch means equally weighted; factorized candidate constants cancel analytically, not by subtracting large rounded padded losses"
    return out

def summarize(records):
    critical=float(student_t.ppf(1-CONFIG["primary_alpha"]/(2*CONFIG["primary_family_size"]),15))
    primary_inputs=[[],[],[],[]]
    for r in records:
        s=r["summaries"]["K100_all"]
        primary_inputs[0].append(s["raw_high_minus_truth"][0])
        primary_inputs[1].append(s["raw_high_minus_truth"][2])
        primary_inputs[2].append(s["raw_correlation_direction"][2]-s["raw_correlation_direction"][0])
        primary_inputs[3].append(s["raw_gradient"][2][-1]/256)
    names=["K100_base_high_minus_truth","K100_pad256_high_minus_truth","K100_padding_correlation_gradient_shift","K100_pad256_nuisance_scale_gradient_per_coordinate"]
    directions=["positive","negative","negative","positive"]
    primary=[]
    for i,(name,expected,se_gate) in enumerate(zip(names,directions,CONFIG["primary_SE_thresholds"])):
        st=statistics(primary_inputs[i],critical)
        primary.append(dict(name=name,stream_values=primary_inputs[i],**st,expected_sign=expected,
            effect_passed=bool(st["lower"]>0 if expected=="positive" else st["upper"]<0),
            SE_threshold=se_gate,precision_passed=bool(st["SE"]<=se_gate)))
    tables=[]
    for scope in ("K100_all","K100_prefix1024","K400_prefix1024"):
        for ni,n in enumerate(CONFIG["paddings"]):
            rows=[r["summaries"][scope] for r in records]
            for ti,t in enumerate(CONFIG["candidate_t"]):
                tables.append(dict(scope=scope,padding=n,t=t,rows_per_stream=rows[0]["rows"],
                    raw_score=statistics([r["raw_values"][ni][ti] for r in rows]),
                    raw_contrast_to_truth=statistics([r["raw_values"][ni][ti]-r["raw_values"][ni][1] for r in rows]),
                    factorized_score=statistics([r["factorized_score_with_analytic_nlog2"][ni][ti] for r in rows]),
                    factorized_contrast_to_truth=statistics([r["factorized_active_values"][ti]-r["factorized_active_values"][1] for r in rows])))
    gradients=[]
    for scope in ("K100_all","K100_prefix1024","K400_prefix1024"):
        for ni,n in enumerate(CONFIG["paddings"]):
            g=np.asarray([r["summaries"][scope]["raw_gradient"][ni] for r in records])
            gradients.append(dict(scope=scope,padding=n,mu_components=[statistics(g[:,i]) for i in range(4)],
                cov_components=[statistics(g[:,i]) for i in range(4,10)],
                nuisance_scale=statistics(g[:,-1]),nuisance_scale_per_coordinate=statistics(g[:,-1]/n if n else np.zeros(16)),
                correlation_direction=statistics([r["summaries"][scope]["raw_correlation_direction"][ni] for r in records])))
    numerical=[dict(stream=r["stream"],**x) for r in records for x in r["numerical64"]]
    controls=[dict(stream=r["stream"],**x) for r in records for x in r["scale_controls"]]
    return dict(primary_family=primary,critical_t=critical,family_size=4,alpha=.05,df=15,
        descriptive_score_tables=tables,descriptive_gradient_tables=gradients,
        numerical64=numerical,scale_controls=controls,
        numerical_qualification=dict(score_count=len(numerical),score_passed=sum(r["score_pass"] for r in numerical),
            gradient_count=sum("gradient_pass" in r for r in numerical),gradient_passed=sum(r.get("gradient_pass",False) for r in numerical),
            maximum_score_difference=max(abs(r["score_difference"]) for r in numerical),
            maximum_normalized_gradient_difference=max(r.get("maximum_normalized_gradient_difference",0) for r in numerical)),
        scale_control_qualification=dict(value_count=len(controls),value_passed=sum(r["value_qualified"] for r in controls),
            active_gradient_count=sum("active_gradient_qualified" in r for r in controls),
            active_gradient_passed=sum(r.get("active_gradient_qualified",False) for r in controls)))

def schema():
    official=kernel.provenance()
    bound={official["official_losses"]:official["official_losses_sha256"],
           official["official_models"]:official["official_models_sha256"],
           official["default_config"]:official["default_config_sha256"]}
    print(json.dumps(dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),config=CONFIG,
        fixture=kernel.FIXTURE,official_inputs=bound,official_provenance=official,expected_freeze_fields=["source_sha256","kernel_sha256",
        "config","fixture","design_path","design_sha256","reference_path","reference_sha256","official_inputs","run_authorized","base_commit"],
        input_order="Roles0/1/2: full data Gaussian[N,4], active uniform[N,4], nuisance uniform[N,256]. Roles3/4: in each batch eps[Kmax,64,4], nu[Kmax,64,256]; first16 batches Kmax400 then48 batches Kmax100; K100 uses [:100]."),indent=2),flush=True)

def run():
    fp=OUT/"FREEZE.json";f=read(fp)
    assert f["run_authorized"] is True and f["config"]==CONFIG and f["fixture"]==kernel.FIXTURE
    assert f["source_sha256"]==sha(SOURCE) and f["kernel_sha256"]==sha(kernel.__file__)
    assert sha(ROOT/f["design_path"])==f["design_sha256"]
    rp=ROOT/f["reference_path"];assert sha(rp)==f["reference_sha256"]
    provenance=kernel.provenance()
    for name,digest in f["official_inputs"].items():assert sha(ROOT/name)==digest,name
    check=read(kernel.OUT/"SELFTEST.json");assert check["passed"] and check["source_sha256"]==sha(kernel.__file__)
    ref=read(rp);assert ref["qualification_passed"]
    assert ref["config"]["mean"]==kernel.FIXTURE["mu"] and ref["config"]["covariance"]==kernel.FIXTURE["C0"]
    assert not (OUT/"EXECUTION.json").exists() and not list(OUT.glob("stream_*.npz")),"No automatic scientific retry"
    save(OUT/"EXECUTION.json",dict(start_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(fp),official=provenance,automatic_retry=False),new=True)
    start=time.perf_counter();records=[];active_stream=None;finished_blocks=0
    try:
        for stream in range(CONFIG["streams"]):
            active_stream=stream;finished_blocks=0;tick=time.perf_counter()
            rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([CONFIG["seed_base"],stream,role]))) for name,role in CONFIG["roles"].items()}
            z=rng["data_gaussian"].standard_normal((4096,4));u=rng["data_active_uniform"].random((4096,4));un=rng["data_nuisance_uniform"].random((4096,256))
            logits=z@np.linalg.cholesky(np.asarray(kernel.FIXTURE["C0"])).T+np.asarray(kernel.FIXTURE["mu"])
            labels=(u<expit(logits)).astype(np.int8);noise_labels=(un<.5).astype(np.int8)
            hashes=dict(data_gaussian=kernel.ah(z),data_active_uniform=kernel.ah(u),data_nuisance_uniform=kernel.ah(un),
                        active_labels=kernel.ah(labels),nuisance_labels=kernel.ah(noise_labels))
            codes=labels@np.array([8,4,2,1],dtype=np.int64)
            patterns={"all":np.bincount(codes,minlength=16).tolist(),"prefix1024":np.bincount(codes[:1024],minlength=16).tolist()}
            data_inputs=dict(active_labels=labels,nuisance_labels=noise_labels,data_active_gaussian=z,data_active_uniform=u)
            del z,u,un,logits
            digests={name:hashlib.sha256() for name in ("prior_active","prior_nuisance")}
            arrays={};numerical=[];controls=[];official_calls=0;stable_calls=0;first_hashes={}
            for bi,begin in enumerate(range(0,4096,64)):
                end=begin+64;Kmax=400 if begin<1024 else 100
                e=rng["prior_active"].standard_normal((Kmax,64,4));nu=rng["prior_nuisance"].standard_normal((Kmax,64,256))
                for name,x in (("prior_active",e),("prior_nuisance",nu)):
                    b=np.ascontiguousarray(x).tobytes();digests[name].update(b)
                    if bi==0:first_hashes[name]=hashlib.sha256(b).hexdigest()
                if bi==0:
                    data_inputs.update(first_prior_active=e.copy(),first_prior_nuisance=nu.copy())
                    input_path=OUT/f"input_{stream:02}.npz";np.savez_compressed(input_path,**data_inputs)
                    del data_inputs
                for K in ([100,400] if begin<1024 else [100]):
                    values,num,con,counts=evaluate_batch(e,nu,labels[begin:end],noise_labels[begin:end],K,bi==0)
                    prefix=f"K{K}";nb=64 if K==100 else 16
                    for key,value in values.items():
                        name=prefix+"_"+key
                        if name not in arrays:arrays[name]=np.empty((nb,)+value.shape,dtype=value.dtype)
                        arrays[name][bi]=value
                    numerical.extend(num);controls.extend(con)
                    official_calls+=counts["official_calls"];stable_calls+=counts["stable64_calls"]
                finished_blocks=bi+1
            summaries={"K100_all":scope_summary(arrays,"K100",64),"K100_prefix1024":scope_summary(arrays,"K100",16),"K400_prefix1024":scope_summary(arrays,"K400",16)}
            oracle={}
            for name,count in patterns.items():
                oracle[name]=[dict(t=p["t"],numerical_reference_NLL=float(-np.dot(np.asarray(count),np.log(p["main_pmf"]))/sum(count))) for p in ref["records"]]
            path=OUT/f"stream_{stream:02}.npz";np.savez_compressed(path,**arrays)
            saturation={}
            for K,scope in ((100,"all4096"),(400,"prefix1024")):
                saturation[str(K)]=dict(scope=scope,sigmoid_zero_counts=arrays[f"K{K}_sigmoid_zero_counts"].sum(axis=0).tolist(),
                    sigmoid_one_counts=arrays[f"K{K}_sigmoid_one_counts"].sum(axis=0).tolist(),
                    probability_evaluations=arrays[f"K{K}_probability_evaluations"].sum(axis=0).tolist(),
                    maximum_absolute_logits=arrays[f"K{K}_maximum_absolute_logits"].max(axis=0).tolist())
            record=dict(stream=stream,source_sha256=sha(SOURCE),input_file=rel(input_path),input_sha256=sha(input_path),
                output_file=rel(path),output_sha256=sha(path),data_hashes=hashes,
                prior_hashes={name:h.hexdigest() for name,h in digests.items()},first_prior_batch_hashes=first_hashes,
                role_seed_sequences={name:[CONFIG["seed_base"],stream,role] for name,role in CONFIG["roles"].items()},
                summaries=summaries,pattern_counts=patterns,numerical_reference_scores=oracle,
                numerical64=numerical,scale_controls=controls,probability_diagnostics=saturation,
                official_API_calls=official_calls,stable64_control_calls=stable_calls,seconds=time.perf_counter()-tick)
            save(OUT/f"stream_{stream:02}.json",record,new=True);records.append(record)
            save(OUT/"PROGRESS.json",dict(completed_streams=len(records),seconds=time.perf_counter()-start))
            print(json.dumps(dict(completed_stream=stream,seconds=record["seconds"])),flush=True)
        assert f["source_sha256"]==sha(SOURCE) and f["kernel_sha256"]==sha(kernel.__file__)
        for name,digest in f["official_inputs"].items():assert sha(ROOT/name)==digest,name
        analysis=summarize(records)
        output=dict(source_sha256=sha(SOURCE),kernel_sha256=sha(kernel.__file__),freeze_sha256=sha(fp),reference_sha256=sha(rp),
            official=provenance,config=CONFIG,fixture=kernel.FIXTURE,records=records,**analysis,
            seconds=time.perf_counter()-start,
            ledger=dict(independent_streams=16,new_complete_observations=65536,main_prior_K=100,auxiliary_K=400,
                auxiliary_observations_per_stream=1024,prior_vectors=11468800,prior_Gaussian_scalars=2981888000,
                prior_active_Gaussian_scalars=45875200,prior_nuisance_Gaussian_scalars=2936012800,
                data_Gaussian_scalars=262144,data_active_uniforms=262144,data_nuisance_uniforms=16777216,
                data_uniforms=17039360,new_neural_network_fits=0,new_parameter_optimizations=0,
                official_API_calls=sum(r["official_API_calls"] for r in records),stable64_control_calls=sum(r["stable64_control_calls"] for r in records)),
            scope="Untouched official SCBM concept-score API at fixed model-compatible tensors; no encoder, target head, precision penalty or optimizer run. No claim of inevitable SCBM training failure or official checkpoint-byte reproduction.",
            numerical_scope="Native API32 is primary. Stable64 includes parameter/input rounding and expression changes; zero/floor only predetermined controls. Numerical reference PMFs are qualified, not strict integration certificates.")
        save(OUT/"RESULTS.json",output,new=True)
        print(json.dumps(dict(completed=True,primary_effects=[p["effect_passed"] for p in output["primary_family"]],
            primary_precision=[p["precision_passed"] for p in output["primary_family"]],
            numerical_qualification=output["numerical_qualification"],seconds=output["seconds"])),flush=True)
    except BaseException as error:
        save(OUT/"ERROR.json",dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            current_stream=active_stream,finished_blocks=finished_blocks,completed_streams=len(records),automatic_retry=False),new=True)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["schema","run"]);globals()[p.parse_args().phase]()
