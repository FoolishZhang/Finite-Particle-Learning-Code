"""Variance-matched nuisance population scores with fixed, role-split inputs.

check/bench use bounded deterministic arrays. Scientific RNG requires a frozen
MC_DESIGN, qualified scale reference, source binding and explicit run command.
Canonical nuisance signs serve a bank expectation, never finite-data training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy.special import expit, logsumexp
import wstructure_reference as reference_code

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/nuisance_structure/MC"
SOURCE=Path(__file__).resolve()
SIGNS=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=np.int8)
PRODUCTS=SIGNS.prod(axis=1)
DEFAULTS=dict(ns=[62,254],Ks=[64,256],families=["diffuse","boundedhetero","concentrated"],
    a_active=1.,R=.75,thetas=[0.,.6],rhos=[0.,.3,.6,.75],streams=16,banks_per_stream=4096,
    block=64,seed_base=202610023100,roles=dict(active=0,uniform=1,nuisance=2),
    price_stream_SE_gate=2e-4,score_stream_SE_gate=5e-4,primary_n=254,
    primary_family_size=12,primary_alpha=.05,audit_banks=4)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def write(p,x,new=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x" if new else "w",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def case_list(config):
    return [dict(n=n,K=K,family=family) for n in config["ns"] for K in config["Ks"] for family in config["families"]]

def teacher_weights(reference,config):
    known=reference["known_active_reference"]
    grid=np.asarray(known["rho801"]);rr=np.asarray(known["r801"])
    assert known["a"]==1. and known["R"]==config["R"] and known["original_r_qualification"]
    records=[];weights=[]
    for theta in config["thetas"]:
        index=int(np.argmin(abs(grid-theta)))
        assert abs(grid[index]-theta)<1e-14
        moment=0. if theta==0 else float(rr[index])
        probs=(1+moment*PRODUCTS)/4
        assert np.all(probs>0) and abs(float(probs.sum())-1)<2e-15
        records.append(dict(theta=theta,reference_node_rho=float(grid[index]),r=moment,
            exact_zero_used=theta==0,scope="Frozen numerical-reference teacher weights except exact theta=0; no new integration remainder certificate"))
        weights.append(probs)
    return np.asarray(weights),records

def profiles(reference,config):
    return {(n,family):reference_code.scale_vector(reference,family,n)
            for n in config["ns"] for family in config["families"]}

def noise_cache(noise,scale_profiles,cases):
    """Full nuisance factors retained, followed by rho-independent normalization."""
    totals={}
    for (n,family),scales in scale_profiles.items():
        totals[n,family]=-np.logaddexp(0.,-noise[:,:,:n]*scales[None,None,:]).sum(axis=2)
    cache=[];B=len(noise)
    ess=np.empty((B,len(cases)));max_weight=np.empty_like(ess);offset=np.empty_like(ess)
    for ci,case in enumerate(cases):
        logw=totals[case["n"],case["family"]][:,:case["K"]]
        norm=logsumexp(logw,axis=1)
        logq=logw-norm[:,None];q=np.exp(logq)
        ess[:,ci]=1/np.sum(q*q,axis=1)
        max_weight[:,ci]=np.max(q,axis=1)
        offset[:,ci]=math.log(case["K"])-norm
        cache.append(logq)
    return cache,ess,max_weight,offset

def mode_kernel(active,cache,cases,rho):
    """Actual path score/derivative plus off-diagonal Price estimate per mode."""
    B=len(active);rho=np.asarray(rho,float)
    if rho.ndim==0:rho=np.full(B,float(rho))
    assert rho.shape==(B,) and np.all(abs(rho)<1)
    z1,z2=active[:,:,0],active[:,:,1]
    root=np.sqrt(1-rho*rho)
    x2=rho[:,None]*z1+root[:,None]*z2
    dx=z1-rho[:,None]/root[:,None]*z2
    first_log={s:-np.logaddexp(0.,-s*z1) for s in (-1,1)}
    first_comp={s:expit(-s*z1) for s in (-1,1)}
    second_log={s:-np.logaddexp(0.,-s*x2) for s in (-1,1)}
    second_comp={s:expit(-s*x2) for s in (-1,1)}
    scores=np.empty((B,len(cases),4));price=np.empty_like(scores);path=np.empty_like(scores)
    for ci,case in enumerate(cases):
        k=case["K"]
        for mi,signs in enumerate(SIGNS):
            s1,s2=int(signs[0]),int(signs[1])
            logweight=cache[ci]+first_log[s1][:,:k]+second_log[s2][:,:k]
            norm=logsumexp(logweight,axis=1)
            p=np.exp(logweight-norm[:,None])
            scores[:,ci,mi]=-norm
            price[:,ci,mi]=-s1*s2*np.sum(p*(1-p)*first_comp[s1][:,:k]*second_comp[s2][:,:k],axis=1)
            path[:,ci,mi]=-s2*np.sum(p*second_comp[s2][:,:k]*dx[:,:k],axis=1)
    assert np.isfinite(scores).all() and np.isfinite(price).all() and np.isfinite(path).all()
    assert np.max(abs(price))<=1+2e-14
    return scores,price,path

def block_kernel(active,noise,uniform,scale_profiles,config,weights):
    cases=case_list(config);B=len(active)
    cache,ess,maxw,offset=noise_cache(noise,scale_profiles,cases)
    shape=(B,len(cases),len(config["rhos"]),4)
    score=np.empty(shape);price=np.empty(shape);path=np.empty(shape)
    for ri,rho in enumerate(config["rhos"]):
        score[:,:,ri],price[:,:,ri],path[:,:,ri]=mode_kernel(active,cache,cases,rho)
    score-=score[:,:,0:1].copy()
    random_price=np.empty((B,len(cases),len(config["thetas"]),4))
    random_rho=np.empty((B,len(config["thetas"])))
    gap=np.empty((B,len(cases),len(config["thetas"])))
    for ti,theta in enumerate(config["thetas"]):
        L=config["R"]-theta
        random_rho[:,ti]=theta+L*uniform
        _,random_price[:,:,ti],_=mode_kernel(active,cache,cases,random_rho[:,ti])
        gap[:,:,ti]=L*np.einsum("bcm,m->bc",random_price[:,:,ti],weights[ti])
        assert np.max(abs(gap[:,:,ti]))<=L+2e-14
    return dict(score_difference_modes=score,price_modes=price,path_modes=path,
        random_price_modes=random_price,random_rho=random_rho,gap=gap,
        nuisance_ESS=ess,nuisance_max_weight=maxw,nuisance_score_offset=offset)

def deterministic_inputs(B,K,n):
    active=1.3*np.sin(np.arange(B*K*2,dtype=float)*.13).reshape(B,K,2)
    noise=1.4*np.cos(np.arange(B*K*n,dtype=float)*.023).reshape(B,K,n)
    uniform=(np.arange(B)+.5)/B
    return active,noise,uniform

def fixture_profiles(ns):
    # Deliberately not variance-matched; computational fixtures only.
    return {(n,f):np.ones(n) if f=="diffuse" else np.tile([.5,1.4],n//2) if f=="boundedhetero"
            else np.concatenate(([2*math.sqrt(n)],np.full(n-1,1/n)))
            for n in ns for f in DEFAULTS["families"]}

def scalar_modes(active,noise,scales,k,rho):
    """Independent unnormalized scalar expression for a single bank and case."""
    ls=lambda x:-float(np.logaddexp(0.,-x))
    nuisance=[math.fsum(ls(float(scales[j]*noise[i,j])) for j in range(len(scales))) for i in range(k)]
    scores=[];price=[];path=[]
    for signs in SIGNS:
        s1,s2=map(int,signs)
        xs=[rho*float(a[0])+math.sqrt(1-rho*rho)*float(a[1]) for a in active[:k]]
        logs=[nuisance[i]+ls(s1*active[i,0])+ls(s2*xs[i]) for i in range(k)]
        peak=max(logs);unnorm=[math.exp(v-peak) for v in logs];total=math.fsum(unnorm);p=[v/total for v in unnorm]
        scores.append(math.log(k)-peak-math.log(total))
        price.append(-s1*s2*math.fsum(p[i]*(1-p[i])*float(expit(-s1*active[i,0]))*float(expit(-s2*xs[i])) for i in range(k)))
        path.append(-s2*math.fsum(p[i]*float(expit(-s2*xs[i]))*(active[i,0]-rho/math.sqrt(1-rho*rho)*active[i,1]) for i in range(k)))
    return np.array(scores),np.array(price),np.array(path)

def check():
    cfg=dict(DEFAULTS,ns=[6,10],Ks=[3,7]);cases=case_list(cfg);scales=fixture_profiles(cfg["ns"])
    active,noise,u=deterministic_inputs(3,7,10)
    weights=np.array([(1+t*PRODUCTS)/4 for t in (0.,.1)])
    out=block_kernel(active,noise,u,scales,cfg,weights)
    error=dict(score=0.,price=0.,path=0.,random_price=0.,path_FD=0.)
    cache,*_=noise_cache(noise,scales,cases)
    for bi in range(len(active)):
        for ci,c in enumerate(cases):
            anchor=scalar_modes(active[bi],noise[bi],scales[c["n"],c["family"]],c["K"],0.)[0]
            for ri,rho in enumerate(cfg["rhos"]):
                raw,pd,pathd=scalar_modes(active[bi],noise[bi],scales[c["n"],c["family"]],c["K"],rho)
                for name,got,want in [("score",out["score_difference_modes"][bi,ci,ri],raw-anchor),
                                      ("price",out["price_modes"][bi,ci,ri],pd),("path",out["path_modes"][bi,ci,ri],pathd)]:
                    error[name]=max(error[name],float(np.max(abs(got-want))))
            for ti in range(2):
                rr=out["random_rho"][bi,ti]
                want=scalar_modes(active[bi],noise[bi],scales[c["n"],c["family"]],c["K"],rr)[1]
                error["random_price"]=max(error["random_price"],float(np.max(abs(out["random_price_modes"][bi,ci,ti]-want))))
    h=2e-5
    for rho in (.11,.43,.71):
        vals=[mode_kernel(active,cache,cases,rho+j*h)[0] for j in (-2,-1,1,2)]
        finite_difference=(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h)
        true_path=mode_kernel(active,cache,cases,rho)[2]
        error["path_FD"]=max(error["path_FD"],float(np.max(abs(finite_difference-true_path))))
    assert max(error[k] for k in ("score","price","path","random_price"))<3e-12
    assert error["path_FD"]<5e-10
    zero_active=np.zeros_like(active);zero_noise=np.zeros_like(noise)
    zero=block_kernel(zero_active,zero_noise,u,scales,cfg,weights)
    expected=np.asarray([-PRODUCTS*(1-1/c["K"])/4 for c in cases])
    assert np.max(abs(zero["price_modes"]-expected[None,:,None,:]))<3e-14
    assert np.max(abs(zero["path_modes"]))==0 and np.max(abs(zero["score_difference_modes"]))==0
    for ci,c in enumerate(cases):
        assert np.max(abs(zero["nuisance_ESS"][:,ci]-c["K"]))<1e-12
    result=dict(passed=True,source_sha256=sha(SOURCE),reference_helper_sha256=sha(reference_code.__file__),
        maximum_errors=error,zero_bank_closed_formula_passed=True,new_Gaussian_draws=0,new_uniform_draws=0,
        model_integrals=0,new_training_fits=0,
        scope="Deterministic arrays only. FD checks the true path derivative; Price is checked against separate scalar/zero-bank formulas, never equated pathwise.")
    write(OUT/"SELFTEST.json",result,new=True);print(json.dumps(result),flush=True)

def bench():
    active,noise,u=deterministic_inputs(64,256,254)
    scales=fixture_profiles(DEFAULTS["ns"]);weights=np.array([(1+t*PRODUCTS)/4 for t in (0.,.1)])
    started=time.perf_counter();out=block_kernel(active,noise,u,scales,DEFAULTS,weights)
    elapsed=time.perf_counter()-started
    result=dict(source_sha256=sha(SOURCE),block=64,seconds=elapsed,
        projected_kernel_seconds=elapsed*DEFAULTS["streams"]*DEFAULTS["banks_per_stream"]/64,
        input_bytes=active.nbytes+noise.nbytes+u.nbytes,output_bytes=sum(x.nbytes for x in out.values()),
        checksum=float(sum(x.sum() for x in out.values())),new_Gaussian_draws=0,new_uniform_draws=0,
        model_integrals=0,new_training_fits=0,
        scope="Bounded deterministic, explicitly unmatched scale fixtures; timing excludes RNG/serialization/system contention, not a science pilot.")
    write(OUT/"BENCHMARK.json",result,new=True);print(json.dumps(result),flush=True)

def schema():
    print(json.dumps(dict(source_sha256=sha(SOURCE),reference_helper_sha256=sha(reference_code.__file__),
        config=DEFAULTS,expected_freeze_fields=["source_sha256","reference_helper_sha256","plan_path","plan_sha256",
        "reference_path","reference_sha256","config","run_authorized","base_commit"]),indent=2),flush=True)

def run():
    frozen_path=OUT/"FREEZE.json";f=read(frozen_path);cfg=f["config"]
    assert f["run_authorized"] is True and cfg==DEFAULTS
    assert f["source_sha256"]==sha(SOURCE) and f["reference_helper_sha256"]==sha(reference_code.__file__)
    assert sha(ROOT/f["plan_path"])==f["plan_sha256"]
    refpath=ROOT/f["reference_path"];assert sha(refpath)==f["reference_sha256"]
    ref=read(refpath);assert ref["qualification_passed"] is True
    test=read(OUT/"SELFTEST.json");assert test["passed"] and test["source_sha256"]==sha(SOURCE)
    assert not (OUT/"EXECUTION.json").exists() and not (OUT/"RESULTS.json").exists(),"No automatic resume/retry"
    assert not list(OUT.glob("stream_*.npz")) and not list(OUT.glob("input_*.npz"))
    scale_profiles=profiles(ref,cfg);cases=case_list(cfg);weights,weight_info=teacher_weights(ref,cfg)
    write(OUT/"EXECUTION.json",dict(start_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(SOURCE),
        freeze_sha256=sha(frozen_path),reference_sha256=sha(refpath),automatic_retry=False),new=True)
    records=[];gaps=[];started=time.perf_counter();current_stream=None;completed_banks=0
    try:
        for stream in range(cfg["streams"]):
            current_stream=stream;completed_banks=0;begin=time.perf_counter()
            rng={name:np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg["seed_base"],stream,role])))
                 for name,role in cfg["roles"].items()}
            digests={name:hashlib.sha256() for name in rng};first_hash={};acc=None
            block_seconds=[]
            for offset in range(0,cfg["banks_per_stream"],cfg["block"]):
                tick=time.perf_counter();B=cfg["block"]
                active=rng["active"].standard_normal((B,max(cfg["Ks"]),2))
                uniform=rng["uniform"].random(B)
                noise=rng["nuisance"].standard_normal((B,max(cfg["Ks"]),max(cfg["ns"])))
                for name,x in [("active",active),("uniform",uniform),("nuisance",noise)]:
                    raw=np.ascontiguousarray(x).tobytes();digests[name].update(raw)
                    if offset==0:first_hash[name]=hashlib.sha256(raw).hexdigest()
                if offset==0:
                    inp=OUT/f"input_{stream:02}.npz"
                    np.savez_compressed(inp,active=active[:cfg["audit_banks"]],uniform=uniform[:cfg["audit_banks"]],
                        nuisance=noise[:cfg["audit_banks"]],signs=SIGNS)
                result=block_kernel(active,noise,uniform,scale_profiles,cfg,weights)
                if acc is None:acc={key:np.empty((cfg["banks_per_stream"],)+value.shape[1:]) for key,value in result.items()}
                for key,value in result.items():acc[key][offset:offset+B]=value
                completed_banks=offset+B;block_seconds.append(time.perf_counter()-tick)
            dest=OUT/f"stream_{stream:02}.npz"
            np.savez_compressed(dest,**acc,signs=SIGNS,rhos=np.array(cfg["rhos"]),thetas=np.array(cfg["thetas"]))
            means={name:acc[name].mean(axis=0).tolist() for name in acc if name!="random_rho"}
            row=dict(stream=stream,role_seed_sequences={name:[cfg["seed_base"],stream,role] for name,role in cfg["roles"].items()},
                input_sha256={name:h.hexdigest() for name,h in digests.items()},first_block_sha256=first_hash,
                audit_input_file=rel(inp),audit_input_sha256=sha(inp),file=rel(dest),file_sha256=sha(dest),
                means=means,within_stream_gap_sample_variance=acc["gap"].var(axis=0,ddof=1).tolist(),
                elapsed_seconds=time.perf_counter()-begin,block_seconds=block_seconds)
            records.append(row);gaps.append(acc["gap"].copy())
            write(OUT/f"stream_{stream:02}.json",row,new=True)
            write(OUT/"PROGRESS.json",dict(completed_streams=len(records),elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(completed_stream=stream,seconds=row["elapsed_seconds"])),flush=True)
        M=cfg["streams"]*cfg["banks_per_stream"];allgap=np.concatenate(gaps,axis=0)
        means=allgap.mean(axis=0);variances=allgap.var(axis=0,ddof=1)
        logterm=math.log(4*cfg["primary_family_size"]/cfg["primary_alpha"])
        primary=[]
        for ci,case in enumerate(cases):
            if case["n"]!=cfg["primary_n"]:continue
            for ti,theta in enumerate(cfg["thetas"]):
                L=cfg["R"]-theta
                half=math.sqrt(2*float(variances[ci,ti])*logterm/M)+7*(2*L)*logterm/(3*(M-1))
                mean=float(means[ci,ti]);expected="positive" if case["family"]=="concentrated" else "negative"
                lower,upper=mean-half,mean+half
                primary.append(dict(**case,theta=theta,M=M,mean=mean,sample_variance=float(variances[ci,ti]),L=L,
                    logterm=logterm,half_width=half,lower=lower,upper=upper,expected_sign=expected,
                    passed=bool(lower>0 if expected=="positive" else upper<0),
                    scope="Empirical-Bernstein sampling interval for fixed reference-weight expected Price integral; excludes PRNG, float and Gaussian reference remainder certification."))
        assert len(primary)==12
        precision={}
        for name,field,threshold in [("price","price_modes",cfg["price_stream_SE_gate"]),
                                     ("score_difference","score_difference_modes",cfg["score_stream_SE_gate"]),
                                     ("path","path_modes",None)]:
            mode_means=np.asarray([r["means"][field] for r in records])
            teacher_means=np.einsum("scrm,tm->scrt",mode_means,weights)
            mean=teacher_means.mean(axis=0);se=teacher_means.std(axis=0,ddof=1)/math.sqrt(cfg["streams"])
            active_mask=np.ones_like(se,dtype=bool)
            if name=="score_difference":active_mask[:,0,:]=False
            precision[name]=dict(mean=mean.tolist(),stream_SE=se.tolist(),threshold=threshold,
                assessed_mask=active_mask.tolist(),pass_mask=None if threshold is None else (se<=threshold).tolist(),
                assessed_count=0 if threshold is None else int(active_mask.sum()),
                pass_count=None if threshold is None else int(((se<=threshold)&active_mask).sum()),
                scope="Sixteen-stream development precision diagnostic, separate from the per-bank primary family")
        output=dict(schema="wstructure-population-panel-v1",base_commit=f["base_commit"],source_sha256=sha(SOURCE),
            reference_helper_sha256=sha(reference_code.__file__),freeze_sha256=sha(frozen_path),reference_sha256=sha(refpath),
            config=cfg,cases=cases,mode_signs=SIGNS.tolist(),teacher_weights=weights.tolist(),teacher_weight_reference=weight_info,
            records=records,primary_family=primary,precision=precision,
            all_gap_mean=means.tolist(),all_gap_sample_variance=variances.tolist(),
            elapsed_seconds=time.perf_counter()-started,
            accounting=dict(independent_banks=65536,independent_seed_groups=16,prior_vectors=16777216,
                Gaussian_scalars=4294967296,active_Gaussian_scalars=33554432,nuisance_Gaussian_scalars=4261412864,
                uniform_scalars=65536,new_observations=0,new_data_fits=0,new_neural_models=0),
            scope="Population fixed-candidate scores/derivatives and bounded random-parameter Price-gap estimator; not real-row data training or a continuous-global-minimum certificate. Families/cases/teachers reuse each independent maximum bank.")
        write(OUT/"RESULTS.json",output,new=True)
        print(json.dumps(dict(completed=True,primary_passed=sum(r["passed"] for r in primary),primary_total=12,
            precision_counts={k:[v["pass_count"],v["assessed_count"]] for k,v in precision.items()},
            elapsed_seconds=output["elapsed_seconds"])),flush=True)
    except BaseException as error:
        write(OUT/"ERROR.json",dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            current_stream=current_stream,completed_banks_in_current_stream=completed_banks,
            completed_streams=len(records),automatic_retry=False),new=True)
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["check","bench","schema","run"])
    globals()[parser.parse_args().mode]()
