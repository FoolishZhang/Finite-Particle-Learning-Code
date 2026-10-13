"""Frozen POS_FULL raw/private-RB control, no fitting or natural data."""
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
for _key in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[_key] = "2"
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, gammaln, ndtr
from scipy.stats import binom, t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/positive_control_v1"
DOC=ROOT/"docs/08_research_研究工作/正族同分布控制_20260930"
BASE="ba25f839c87ee6e0c5ab642614bb8a50a2a5395a"
DESIGN=ROOT/"docs/08_research_研究工作/主定理定位与后续设计_20260929/POSITIVE_CONTROL_DESIGN.md"
UPPER=DESIGN.with_name("POSITIVE_MIXTURE_UPPER.md")
M2=ROOT/"artifacts/constraint_boundary_v2/M2"
METHODS=["raw10","private10","raw100","private100"]
CONTRASTS=["K10/raw_minus_F0","K10/private_minus_F0","K10/raw_minus_private",
           "K100/raw_minus_F0","K100/private_minus_F0","K100/raw_minus_private"]
CFG=dict(d=[24,128],sigma=4.,c=.9,K=[10,100],streams=8,observations=4096,batch=64,
    seed_base=20276101,common_offset=10000000,private_offset=20000000,
    GH_main=320,GH_audit=480,GH_chunk=256,
    particle_loglikelihood_gate=1e-6,stream_sensitivity_gate=1e-7,
    probability_identity_gate=1e-12,formula_gate=1e-12,pair_corr_gate=1e-9,
    quad_log_gate=1e-10,quad_interval=[-12.,12.],quad_epsabs=1e-12,
    quad_epsrel=1e-12,quad_limit=300,quad_targets=[-6.,-3.,-1.,0.,1.,3.,6.],
    alpha=.05,contrasts=12,raw_SE_gate=1.,private_SE_gate=.003,
    paired_SE_gate=1.,material=.005,primary_d=128,primary_K=100,
    raw_floor=None,dtype="float64",fits=0,natural_data=False)

class NumericalGateFailure(RuntimeError):
    pass

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_hash(state):return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
def state_copy(rng):return json.loads(json.dumps(rng.bit_generator.state))
def write(p,obj,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x" if exclusive else "w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def require(ok,message):
    if not ok:raise NumericalGateFailure(message)

def freeze():
    assert not OUT.exists(),"Refuse to overwrite package"
    source_design=subprocess.check_output(["git","show",f"{BASE}:{DESIGN.relative_to(ROOT).as_posix()}"],cwd=ROOT)
    assert source_design==DESIGN.read_bytes(),"Working design differs from authorized commit"
    assert read(M2/"PARAMETERS.json")["all_numerical_pass"]
    for name,h in read(M2/"PARAMETERS_FREEZE.json")["files"].items():
        assert sha(ROOT/name)==h,name
    OUT.mkdir()
    (OUT/".gitignore").write_text("/scores/\n*.npz\n",encoding="utf-8",newline="\n")
    write(OUT/"CONFIG.json",CFG)
    write(OUT/"ENVIRONMENT.json",dict(time_utc=now(),python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,dtype="float64",BLAS_threads=2,RNG="PCG64"))
    (OUT/"SOURCE_DESIGN.md").write_bytes(source_design)
    paths=[Path(__file__),ROOT/"script/positive_control_delivery.py",DOC/"PLAN.md",
        DESIGN,UPPER,OUT/"CONFIG.json",OUT/"ENVIRONMENT.json",OUT/"SOURCE_DESIGN.md",
        M2/"PARAMETERS.json",M2/"PARAMETERS_FREEZE.json",ROOT/"HANDOFF.md"]
    write(OUT/"FREEZE.json",dict(time_utc=now(),baseline=BASE,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        before_smoke=True,before_any_scientific_observation=True))
    write(OUT/"RUN_STATE.json",dict(status="frozen",active_jobs=[],time_utc=now()),exclusive=False)
    print(json.dumps(dict(freeze_sha256=sha(OUT/"FREEZE.json"),files=len(paths))),flush=True)

def check_freeze():
    assert read(OUT/"CONFIG.json")==CFG
    for name,h in read(OUT/"FREEZE.json")["files"].items():assert sha(ROOT/name)==h,name

@lru_cache(None)
def nodes(order):
    z,w=roots_hermitenorm(order)
    logw=np.full_like(w,-np.inf);positive=w>0
    logw[positive]=np.log(w[positive])-.5*np.log(2*np.pi)
    require(float(np.max(abs(z+z[::-1])))<1e-12,"GH nodes not symmetric")
    require(float(np.max(abs(w-w[::-1])))<1e-12,"GH weights not symmetric")
    return z,logw

def log_h(common,d,order,private_scale=None):
    original=np.asarray(common,dtype=float);flat=original.ravel()
    rho=CFG["c"]/(d-1)
    a=CFG["sigma"]*math.sqrt(rho)
    b=CFG["sigma"]*math.sqrt(1-rho) if private_scale is None else private_scale
    z,lw=nodes(order);hp=np.empty(len(flat));hn=np.empty(len(flat))
    symmetry=0.;largest_working_array=0
    for begin in range(0,len(flat),CFG["GH_chunk"]):
        x=a*flat[begin:begin+CFG["GH_chunk"],None]+b*z
        lp=log_expit(x);ln=log_expit(-x)
        hp[begin:begin+len(x)]=logsumexp(lp+lw,axis=1)
        hn[begin:begin+len(x)]=logsumexp(ln+lw,axis=1)
        # Exact GH sign-permutation identity on every queried common value.
        minus=logsumexp(ln[:,::-1]+lw,axis=1)
        symmetry=max(symmetry,float(np.max(abs(minus-hn[begin:begin+len(x)]))))
        largest_working_array=max(largest_working_array,x.nbytes)
    normalization=float(np.max(abs(np.logaddexp(hp,hn))))
    return hp.reshape(original.shape),hn.reshape(original.shape),dict(
        normalization_error=normalization,GH_sign_permutation_error=symmetry,
        largest_single_working_array_bytes=largest_working_array)

def logchoose(d,n):return gammaln(d+1)-gammaln(n+1)-gammaln(d-n+1)
def losses(logweights):return -logsumexp(logweights,axis=1)+np.log(logweights.shape[1])
def differences(score,d):
    f0=d*np.log(2)
    return np.column_stack((score[:,0]-f0,score[:,1]-f0,score[:,0]-score[:,1],
                            score[:,2]-f0,score[:,3]-f0,score[:,2]-score[:,3]))

def smoke():
    check_freeze();assert not (OUT/"SMOKE.json").exists()
    records=[];reference={p["d"]:p["spin_r"] for p in read(M2/"PARAMETERS.json")["rows"]}
    for d in CFG["d"]:
        rho=CFG["c"]/(d-1);a=CFG["sigma"]*math.sqrt(rho);b=CFG["sigma"]*math.sqrt(1-rho)
        z=np.array([-6.,-3.,-1.,0.,1.,3.,6.])
        hp,hn,diag=log_h(z,d,CFG["GH_main"]);mp,mn,_=log_h(-z,d,CFG["GH_main"])
        scalar_zero=abs(float(hp[3]+np.log(2)))
        symmetry=max(float(np.max(abs(hp-mn))),float(np.max(abs(hn-mp))))
        zmain,lwm=nodes(CFG["GH_main"]);pm,pn,_=log_h(zmain,d,CFG["GH_main"])
        zaudit,lwa=nodes(CFG["GH_audit"]);am,an,_=log_h(zaudit,d,CFG["GH_audit"])
        rm=float(np.exp(lwm)@((np.exp(pm)-np.exp(pn))**2))
        ra=float(np.exp(lwa)@((np.exp(am)-np.exp(an))**2))
        counts=np.array([0,d//3,d//2,d-1,d]);zz=np.linspace(-2,2,len(counts))
        h,q,_=log_h(zz,d,CFG["GH_main"])
        direct=counts*h+(d-counts)*q
        alternate=binom.logpmf(counts,d,np.exp(h))-logchoose(d,counts)
        binerr=float(np.max(abs(direct-alternate)))
        toy_eta=np.arange(2*d,dtype=float).reshape(2,d)/d-1
        toy_y=(np.arange(2*d).reshape(2,d)%2).astype(float)
        raw1=log_expit((2*toy_y-1)*toy_eta).sum(1)
        raw2=(toy_y*toy_eta-np.logaddexp(0.,toy_eta)).sum(1)
        rawerr=float(np.max(abs(raw1-raw2)))
        zero_h,zero_q,_=log_h(z,d,CFG["GH_main"],private_scale=0.)
        zeroprivate=max(float(np.max(abs(zero_h-log_expit(a*z)))),
                        float(np.max(abs(zero_q-log_expit(-a*z)))))
        row=dict(d=d,latent_diagonal=a*a+b*b,latent_offdiagonal=a*a,
            diagonal_error=abs(a*a+b*b-CFG["sigma"]**2),
            offdiagonal_error=abs(a*a-CFG["sigma"]**2*rho),
            half_probability_log_error=scalar_zero,symmetry_log_error=symmetry,
            normalization_error=diag["normalization_error"],pair_r_GH320=rm,pair_r_GH480=ra,
            pair_r_reference=reference[d],pair_error=max(abs(rm-reference[d]),abs(ra-reference[d])),
            binomial_pattern_error=binerr,raw_formula_error=rawerr,
            zero_private_identity_error=zeroprivate)
        row["passed"]=bool(row["diagonal_error"]<1e-12 and row["offdiagonal_error"]<1e-12
            and max(scalar_zero,symmetry,diag["normalization_error"],binerr,rawerr,zeroprivate)<1e-12
            and row["pair_error"]<CFG["pair_corr_gate"])
        records.append(row)
    result=dict(time_utc=now(),records=records,passed=all(r["passed"] for r in records),
        before_observation_simulation=True)
    write(OUT/"SMOKE.json",result)
    require(result["passed"],"Frozen smoke precision gate failed")
    return result

def quad_audit(common,d):
    flat=common.ravel();selected={int(np.argmin(flat)),int(np.argmax(flat))}
    for target in CFG["quad_targets"]:selected.add(int(np.argmin(abs(flat-target))))
    chosen=sorted(selected)
    hp,hn,_=log_h(flat[chosen],d,CFG["GH_audit"])
    neg_hp,neg_hn,_=log_h(-flat[chosen],d,CFG["GH_audit"])
    rho=CFG["c"]/(d-1);a=CFG["sigma"]*math.sqrt(rho);b=CFG["sigma"]*math.sqrt(1-rho)
    rows=[]
    for j,index in enumerate(chosen):
        z=float(flat[index]);results=[]
        for sign in [1,-1]:
            f=lambda e:math.exp(-e*e/2)/math.sqrt(2*math.pi)*expit(sign*(a*z+b*e))
            value=quad(f,*CFG["quad_interval"],epsabs=CFG["quad_epsabs"],
                       epsrel=CFG["quad_epsrel"],limit=CFG["quad_limit"],full_output=1)
            results.append(dict(probability=float(value[0]),estimated_error=float(value[1]),
                successful=len(value)==3,evaluations=int(value[2]["neval"]),
                message="" if len(value)==3 else str(value[3])))
        ok=all(r["successful"] and 0<r["probability"]<1 for r in results)
        errs=[abs(math.log(results[k]["probability"])-[hp[j],hn[j]][k]) if results[k]["probability"]>0 else 1e300 for k in range(2)]
        symmetry=max(abs(float(neg_hp[j]-hn[j])),abs(float(neg_hn[j]-hp[j])))
        rows.append(dict(d=d,flat_index=index,common_z=z,log_h_480=float(hp[j]),log_q_480=float(hn[j]),
            quad_h=results[0],quad_q=results[1],Gaussian_tail_probability_bound=float(2*ndtr(-12.)),
            log_h_error=errs[0],log_q_error=errs[1],direct_negative_input_symmetry_error=symmetry,
            passed=bool(ok and max(errs)<CFG["quad_log_gate"] and symmetry<CFG["probability_identity_gate"])))
    return rows

def one(d,stream):
    check_freeze()
    name=f"d{d}_stream{stream}";assert not (OUT/"streams"/f"{name}.json").exists()
    seed=CFG["seed_base"]+100*d+stream
    rngs=dict(teacher=np.random.default_rng(seed),common=np.random.default_rng(seed+CFG["common_offset"]),
              private=np.random.default_rng(seed+CFG["private_offset"]))
    initial={k:state_copy(v) for k,v in rngs.items()}
    rho=CFG["c"]/(d-1);a=CFG["sigma"]*math.sqrt(rho);b=CFG["sigma"]*math.sqrt(1-rho)
    main_rows=[];audit_rows=[];label_rows=[];common_rows=[]
    ticks=dict(teacher=0.,prior_generation=0.,raw_primary=0.,raw_formula_audit=0.,
               private_GH320=0.,private_GH480=0.,private_formula_audit=0.,quad_audit=0.)
    errors=dict(raw_formula=0.,binomial_pattern=0.,particle_loglikelihood=0.,
                log_h=0.,log_q=0.,normalization=0.,GH_sign_permutation=0.)
    peak_array=0;started=time.perf_counter()
    try:
        for begin in range(0,CFG["observations"],CFG["batch"]):
            n=min(CFG["batch"],CFG["observations"]-begin);tick=time.perf_counter()
            zteacher=rngs["teacher"].normal(size=(n,1))
            eps=rngs["teacher"].normal(size=(n,d))
            teacher_eta=a*zteacher+b*eps
            labels=(rngs["teacher"].random((n,d))<expit(teacher_eta)).astype(np.uint8)
            count=labels.sum(1).astype(int);ticks["teacher"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            common=rngs["common"].normal(size=(n,100))
            private=rngs["private"].normal(size=(n,100,d))
            shared_hash=array_sha(common)
            ticks["prior_generation"]+=time.perf_counter()-tick
            tick=time.perf_counter();eta=a*common[...,None]+b*private
            raw=log_expit((2*labels.astype(float)[:,None,:]-1)*eta).sum(2)
            raw10,raw100=losses(raw[:,:10]),losses(raw)
            ticks["raw_primary"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            alternate=(labels[:,None,:]*eta-np.logaddexp(0.,eta)).sum(2)
            errors["raw_formula"]=max(errors["raw_formula"],float(np.max(abs(raw-alternate))))
            ticks["raw_formula_audit"]+=time.perf_counter()-tick
            tick=time.perf_counter();hp,hn,diag=log_h(common,d,CFG["GH_main"])
            private_main=count[:,None]*hp+(d-count[:,None])*hn
            pm10,pm100=losses(private_main[:,:10]),losses(private_main)
            ticks["private_GH320"]+=time.perf_counter()-tick
            tick=time.perf_counter();ap,an,ad=log_h(common,d,CFG["GH_audit"])
            private_audit=count[:,None]*ap+(d-count[:,None])*an
            pa10,pa100=losses(private_audit[:,:10]),losses(private_audit)
            ticks["private_GH480"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            binomial=binom.logpmf(count[:,None],d,np.exp(hp))-logchoose(d,count)[:,None]
            errors["binomial_pattern"]=max(errors["binomial_pattern"],float(np.max(abs(binomial-private_main))))
            ticks["private_formula_audit"]+=time.perf_counter()-tick
            errors["particle_loglikelihood"]=max(errors["particle_loglikelihood"],float(np.max(abs(private_main-private_audit))))
            errors["log_h"]=max(errors["log_h"],float(np.max(abs(hp-ap))))
            errors["log_q"]=max(errors["log_q"],float(np.max(abs(hn-an))))
            errors["normalization"]=max(errors["normalization"],diag["normalization_error"],ad["normalization_error"])
            errors["GH_sign_permutation"]=max(errors["GH_sign_permutation"],diag["GH_sign_permutation_error"],ad["GH_sign_permutation_error"])
            peak_array=max(peak_array,eta.nbytes,diag["largest_single_working_array_bytes"],ad["largest_single_working_array_bytes"])
            values=np.column_stack((raw10,pm10,raw100,pm100))
            audit_values=np.column_stack((raw10,pa10,raw100,pa100))
            main_rows.append(values);audit_rows.append(audit_values);label_rows.append(labels);common_rows.append(common)
            require(np.isfinite(values).all() and np.isfinite(audit_values).all(),"Nonfinite score")
            require(shared_hash==array_sha(common),"Shared common bank was mutated")
            require(errors["raw_formula"]<CFG["formula_gate"],"Raw formula gate")
            require(errors["binomial_pattern"]<CFG["formula_gate"],"Binomial pattern gate")
            require(errors["particle_loglikelihood"]<CFG["particle_loglikelihood_gate"],"GH particle gate")
            require(max(errors["normalization"],errors["GH_sign_permutation"])<CFG["probability_identity_gate"],"Probability identity gate")
            if (begin+n)%1024==0:print(name,"observations",begin+n,"seconds",round(time.perf_counter()-started,1),flush=True)
        scores=np.concatenate(main_rows);audited=np.concatenate(audit_rows)
        labels=np.concatenate(label_rows);common=np.concatenate(common_rows)
        diffs=differences(scores,d);audit_diffs=differences(audited,d)
        mean_sensitivity=np.max(abs(scores.mean(0)-audited.mean(0)))
        diff_sensitivity=np.max(abs(diffs.mean(0)-audit_diffs.mean(0)))
        errors["stream_method_mean_sensitivity"]=float(mean_sensitivity)
        errors["stream_contrast_sensitivity"]=float(diff_sensitivity)
        tick=time.perf_counter();quads=quad_audit(common,d);ticks["quad_audit"]=time.perf_counter()-tick
        errors["quad_log_max"]=max(max(r["log_h_error"],r["log_q_error"]) for r in quads)
        ok=bool(max(mean_sensitivity,diff_sensitivity)<CFG["stream_sensitivity_gate"] and all(r["passed"] for r in quads))
        target=OUT/"scores"/f"{name}.npz";target.parent.mkdir(exist_ok=True)
        assert not target.exists()
        np.savez_compressed(target,losses=scores,audit_losses=audited,differences=diffs,
            audit_differences=audit_diffs,labels=labels,count=labels.sum(1),common_z=common)
        final={k:state_copy(v) for k,v in rngs.items()}
        receipt=dict(time_utc=now(),d=d,stream=stream,teacher_seed=seed,
            common_seed=seed+CFG["common_offset"],private_seed=seed+CFG["private_offset"],
            RNG="PCG64",initial_RNG_states=initial,final_RNG_states=final,
            initial_RNG_hashes={k:state_hash(v) for k,v in initial.items()},
            final_RNG_hashes={k:state_hash(v) for k,v in final.items()},
            requested_variates=dict(teacher_normal=4096*(d+1),teacher_uniform=4096*d,
                common_normal=4096*100,private_normal=4096*100*d),
            common_bank_sha256=array_sha(common),labels_sha256=array_sha(labels),
            common_bank_shared_by_construction=True,K10_is_full_K100_prefix=True,
            arrays_sha256=sha(target),methods=METHODS,contrasts=CONTRASTS,
            method_means=scores.mean(0).tolist(),contrast_means=diffs.mean(0).tolist(),
            precision=errors,quad_cases=quads,numerical_pass=ok,
            probability_floor=None,finite_scores=True,count_histogram=np.bincount(labels.sum(1).astype(int),minlength=d+1).tolist(),
            elapsed_seconds=time.perf_counter()-started,timing_seconds=ticks,
            logical_costs={str(k):dict(raw_Bernoulli_factors=4096*k*d,
                private_primary_logistic_node_evaluations=2*4096*k*320,
                private_additional_audit_logistic_node_evaluations=2*4096*k*480) for k in CFG["K"]},
            actual_shared_bank_costs=dict(raw_Bernoulli_factors=4096*100*d,
                private_primary_logistic_node_evaluations=2*4096*100*320,
                private_additional_audit_logistic_node_evaluations=2*4096*100*480),
            largest_single_working_array_bytes=peak_array,
            wallclock_is_not_per_method_benchmark=True)
        write(OUT/"streams"/f"{name}.json",receipt)
        require(ok,"Frozen stream sensitivity or quad gate failed")
        return receipt
    except Exception:
        if main_rows and not (OUT/"scores"/f"{name}.npz").exists():
            p=OUT/"scores"/f"PARTIAL_{name}.npz";p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,losses=np.concatenate(main_rows),audit_losses=np.concatenate(audit_rows),
                labels=np.concatenate(label_rows),common_z=np.concatenate(common_rows))
            write(OUT/"partial"/f"{name}.json",dict(time_utc=now(),d=d,stream=stream,
                observations=len(np.concatenate(main_rows)),errors=errors,
                arrays_sha256=sha(p),RNG_states={k:state_copy(v) for k,v in rngs.items()}))
        raise

def collect():
    check_freeze();assert not (OUT/"PRIMARY.csv").exists()
    methods=[];contrasts=[];precision=[];cost=[];quad_rows=[];receipts=[]
    for d in CFG["d"]:
        for stream in range(8):
            r=read(OUT/"streams"/f"d{d}_stream{stream}.json")
            assert r["numerical_pass"] and sha(OUT/"scores"/f"d{d}_stream{stream}.npz")==r["arrays_sha256"]
            receipts.append(r)
            methods.extend(dict(d=d,stream=stream,method=m,mean_F=r["method_means"][j],
                F0=d*np.log(2),mean_F_minus_F0=r["method_means"][j]-d*np.log(2)) for j,m in enumerate(METHODS))
            contrasts.extend(dict(d=d,stream=stream,contrast=m,mean=r["contrast_means"][j]) for j,m in enumerate(CONTRASTS))
            precision.append(dict(d=d,stream=stream,**r["precision"],numerical_pass=r["numerical_pass"]))
            cost.append(dict(d=d,stream=stream,elapsed_seconds=r["elapsed_seconds"],
                **r["timing_seconds"],largest_single_working_array_bytes=r["largest_single_working_array_bytes"]))
            for q in r["quad_cases"]:
                quad_rows.append(dict(d=d,stream=stream,flat_index=q["flat_index"],common_z=q["common_z"],
                    log_h_error=q["log_h_error"],log_q_error=q["log_q_error"],
                    negative_input_symmetry_error=q["direct_negative_input_symmetry_error"],passed=q["passed"]))
    critical=float(t.ppf(1-CFG["alpha"]/(2*CFG["contrasts"]),7));primary=[]
    for d in CFG["d"]:
        for name in CONTRASTS:
            values=np.array([r["mean"] for r in contrasts if r["d"]==d and r["contrast"]==name])
            mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(8))
            low,high=mean-critical*se,mean+critical*se
            gate=CFG["private_SE_gate"] if "/private_minus_F0" in name else 1.
            ok=se<=gate
            primary.append(dict(d=d,contrast=name,mean=mean,SE=se,tcrit=critical,
                lower=low,upper=high,SE_gate=gate,numerical_gates_pass=True,precision_pass=ok,
                material_positive=bool(ok and low>CFG["material"]),
                material_negative=bool(ok and high<-CFG["material"])))
    assert len(primary)==12 and len(methods)==64 and len(contrasts)==96
    for name,rows in [("STREAM_LOSSES.csv",methods),("STREAM_CONTRASTS.csv",contrasts),
        ("PRECISION.csv",precision),("COST.csv",cost),("QUAD_CHECKS.csv",quad_rows),("PRIMARY.csv",primary)]:
        csvwrite(OUT/name,rows)
    chosen={r["contrast"]:r for r in primary if r["d"]==128}
    tests=dict(raw100_true_worse=chosen["K100/raw_minus_F0"]["material_positive"],
        private100_true_better=chosen["K100/private_minus_F0"]["material_negative"],
        paired100_gain=chosen["K100/raw_minus_private"]["material_positive"])
    write(OUT/"DECISION.json",dict(time_utc=now(),primary_tests=tests,main_pass=all(tests.values()),
        precision_pass_count=sum(r["precision_pass"] for r in primary),contrasts=12,
        no_new_fits=True,no_natural_data=True,same_POS_FULL_Q=True,
        all_16_streams_retained=True,no_additional_samples_or_orders=True,
        not_global_optimality_or_same_wallclock_claim=True))
    write(OUT/"RUN_STATE.json",dict(status="complete_scoring",active_jobs=[],time_utc=now()),exclusive=False)
    files=[OUT/name for name in ["STREAM_LOSSES.csv","STREAM_CONTRASTS.csv","PRECISION.csv",
        "COST.csv","QUAD_CHECKS.csv","PRIMARY.csv","DECISION.json"]]
    files+=sorted((OUT/"streams").glob("*.json"))
    write(OUT/"RESULTS_FREEZE.json",dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    return primary

def run():
    check_freeze();assert read(OUT/"RUN_STATE.json")["status"]=="frozen","No silent restart"
    smoke()
    for d in CFG["d"]:
        for stream in range(8):
            write(OUT/"RUN_STATE.json",dict(status="running",d=d,stream=stream,pid=os.getpid(),time_utc=now()),exclusive=False)
            one(d,stream)
    collect()
    from script.positive_control_delivery import generate
    generate()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["freeze","run","check"])
    args=parser.parse_args()
    try:
        if args.phase=="check":
            from script.positive_control_delivery import check
            check()
        else:globals()[args.phase]()
    except Exception as exc:
        if args.phase=="run" and OUT.exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",dict(time_utc=now(),phase=args.phase,
                error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            write(OUT/"RUN_STATE.json",dict(status="stopped_failure",active_jobs=[],time_utc=now(),
                error_type=type(exc).__name__,error=str(exc)),exclusive=False)
        raise
