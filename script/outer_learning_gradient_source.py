"""Frozen-design reuse of old K4 banks for K2 label-RB gradient diagnostics.

qa uses fixed artificial arrays only. freeze/run require explicit orchestration;
the code never creates random samples or imports the old producer functions.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/08_research_研究工作/外层精度与有限样本伪相关_20261001"
OUT=ROOT/"artifacts/nuisance_learning/GRADIENT"
MC=ROOT/"artifacts/nuisance_dimensions/MC"
REF=ROOT/"artifacts/nuisance_dimensions/NUMERIC"
METHODS=("Price","raw_path","noise_CV")
LABELS=np.array(list(itertools.product((0.,1.),repeat=2)))
CFG={"schema":1,"K":2,"a":1.,"rho":.75,"theta":0.,"streams":8,
     "banks_per_stream":8192,"ns":[0,14,62,254,1022,4094],
     "outer_B":[16,64,256,1024,4096],"methods":list(METHODS),
     "Price_RB_bound":.125,"bound_tolerance":1e-12,
     "GH_orders":[64,96,128],"GH_constant_span_gate":1e-10,
     "old_K4_inputs_reused_after_development":True,
     "new_independent_banks":0,"new_labels":0,"RNG_calls":0,"fits":0,"locators":0,
     "statistical_CI":False,"new_success_gates":False}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,obj,replace=False):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open("w" if replace else "x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=serial,allow_nan=False);f.write("\n")
def need(ok,msg):
    if not ok:raise AssertionError(msg)
def verify(mapping):
    for p,h in mapping.items():need(sha(ROOT/p)==h,"hash mismatch: "+p)
def csv_save(path,records):
    with Path(path).open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def active_terms(z,rho):
    need(z.ndim==3 and z.shape[1:]==(2,2),"two-particle active shape")
    root=math.sqrt(1-rho*rho)
    eta=np.stack([z[:,:,0],rho*z[:,:,0]+root*z[:,:,1]],axis=-1)
    velocity=z[:,:,0]-rho*z[:,:,1]/root
    prob=expit(eta)
    logp=-np.logaddexp(0,-eta);logq=-np.logaddexp(0,eta)
    active=[];derivative=[];residual_product=[]
    for label in LABELS:
        active.append(np.sum(label*logp+(1-label)*logq,axis=-1))
        residual=label-prob
        derivative.append(residual[:,:,1]*velocity)
        residual_product.append(residual[:,:,0]*residual[:,:,1])
    return np.stack(active,axis=1),np.stack(derivative,axis=1),np.stack(residual_product,axis=1)

def sigmoid_difference(u,v):
    """sigmoid(u)-sigmoid(v), preserving the small positive-tail difference."""
    positive=(u>=0)&(v>=0)
    return np.where(positive,expit(-v)-expit(-u),expit(u)-expit(v))

def estimators(terms,s):
    active,derivative,residual_product=terms
    need(s.shape==(len(active),2),"nuisance sums shape")
    noise=s[:,0]-s[:,1]
    difference=noise[:,None]+active[:,:,0]-active[:,:,1]
    p1,p2=expit(difference),expit(-difference)
    price=-np.mean(p1*p2*np.sum(residual_product,axis=2),axis=1)
    raw=-np.mean(p1*derivative[:,:,0]+p2*derivative[:,:,1],axis=1)
    residual=sigmoid_difference(noise[:,None],difference)
    cv=np.mean(residual*(derivative[:,:,0]-derivative[:,:,1]),axis=1)
    value=np.stack([price,raw,cv],axis=-1)
    need(np.isfinite(value).all(),"nonfinite bank estimators")
    need(float(np.max(abs(price)))<=CFG["Price_RB_bound"]+CFG["bound_tolerance"],"Price label-RB range")
    return value

def artificial_losses(z,s,rho):
    active,_,_=active_terms(z,rho)
    centered=s-s.max(axis=1,keepdims=True)
    logmean=np.logaddexp(centered[:,None,0]+active[:,:,0],centered[:,None,1]+active[:,:,1])-math.log(2)
    raw=-logmean.mean(axis=1)
    q1,q2=expit(s[:,0]-s[:,1]),expit(s[:,1]-s[:,0])
    controlled=raw+np.mean(q1[:,None]*active[:,:,0]+q2[:,None]*active[:,:,1],axis=1)
    return raw,controlled

def artificial_qa():
    z=(np.arange(4*2*2,dtype=float).reshape(4,2,2)-7.5)/6
    s=np.array([[.2,-.1],[-.8,.3],[0,0],[2.,-1.]])
    rho=.37;h=1e-5;terms=active_terms(z,rho);g=estimators(terms,s)
    plus=artificial_losses(z,s,rho+h);minus=artificial_losses(z,s,rho-h)
    fd_raw=(plus[0]-minus[0])/(2*h);fd_cv=(plus[1]-minus[1])/(2*h)
    explicit=[]
    for b in range(len(z)):
        x=np.stack([z[b,:,0],rho*z[b,:,0]+math.sqrt(1-rho*rho)*z[b,:,1]],axis=-1)
        total=0.
        for label in LABELS:
            loglik=np.sum(label*(-np.logaddexp(0,-x))+(1-label)*(-np.logaddexp(0,x)),axis=1)
            h0,h1=s[b]+loglik
            norm=np.logaddexp(h0,h1)
            pi=np.exp(np.array([h0,h1])-norm)
            res=label-expit(x)
            total-=.25*sum(pi[k]*(1-pi[k])*res[k,0]*res[k,1] for k in range(2))
        explicit.append(total)
    _,derivative,_=terms
    q1,q2=expit(s[:,0]-s[:,1]),expit(s[:,1]-s[:,0])
    cv_identity=g[:,1]+np.mean(q1[:,None]*derivative[:,:,0]+q2[:,None]*derivative[:,:,1],axis=1)
    errors={"raw_path_vs_FD":float(abs(g[:,1]-fd_raw).max()),"noise_CV_vs_controlled_FD":float(abs(g[:,2]-fd_cv).max()),"Price_vs_explicit":float(abs(g[:,0]-explicit).max()),"CV_algebra":float(abs(g[:,2]-cv_identity).max())}
    need(errors["raw_path_vs_FD"]<1e-8 and errors["noise_CV_vs_controlled_FD"]<1e-8,"artificial FD")
    need(errors["Price_vs_explicit"]<1e-12 and errors["CV_algebra"]<1e-12,"artificial algebra")
    return {"passed":True,"errors":errors,"max_abs_Price":float(abs(g[:,0]).max()),"Price_not_required_equal_path_or_CV":True,"artificial_banks":4,"Gaussian_integrals":0,"old_INPUT_accessed":False,"RNG_calls":0,"science_statistics":0}

def qa():
    need(not (OUT/"FREEZE.json").exists(),"QA cannot mutate frozen package")
    result=artificial_qa();folder=OUT/"QA";folder.mkdir(parents=True,exist_ok=True)
    count=len(list(folder.glob("qa_*.json")))+1;path=folder/f"qa_{count:03d}.json"
    result.update(utc=utc(),source_sha256=sha(__file__),execution=count)
    save(path,result);save(OUT/"QA_LATEST.json",{"path":rel(path),"sha256":sha(path),"execution":count},replace=True)
    print(json.dumps(result,default=serial))

def old_sources():
    ms=read(MC/"RESULTS_FREEZE.json");fs=read(REF/"RESULTS_FREEZE.json")
    need(read(MC/"STATE.json")["status"]=="complete","old MC incomplete")
    need(read(REF/"STATE.json")["state"]=="complete","old Fourier incomplete")
    public={rel(MC/p):sha(MC/p) for p in ("FREEZE.json","RESULTS_FREEZE.json","STATE.json")}
    need(ms["source_freeze_sha256"]==sha(MC/"FREEZE.json"),"old MC source link")
    for p in ("CONFIG.json","FREEZE.json","GRID.json","CHECKS.json","RESULTS.json","STATE.json"):
        public[rel(REF/p)]=fs["files"][p]
    public[rel(REF/"RESULTS_FREEZE.json")]=sha(REF/"RESULTS_FREEZE.json")
    local={rel(MC/f"s{s}/INPUT.npz"):ms["files"][rel(MC/f"s{s}/INPUT.npz")] for s in range(8)}
    verify(public);verify(local)
    return public,local

def freeze():
    need(not (OUT/"FREEZE.json").exists(),"already frozen")
    q=read(OUT/"QA_LATEST.json");verify({q["path"]:q["sha256"]})
    unit=read(ROOT/q["path"]);need(unit["passed"] and unit["source_sha256"]==sha(__file__),"QA/source mismatch")
    public,local=old_sources()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"HANDOFF_INPUT.md").write_bytes((ROOT/"HANDOFF.md").read_bytes())
    save(OUT/"CONFIG.json",CFG)
    for p in (Path(__file__),DOC/"GRADIENT_DESIGN.md",OUT/"CONFIG.json",OUT/"HANDOFF_INPUT.md",OUT/"QA_LATEST.json",ROOT/q["path"]):public[rel(p)]=sha(p)
    save(OUT/"FREEZE.json",{"utc":utc(),"baseline":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"files":public,"local_inputs":local,"before_old_data_statistics":True,"before_new_Gaussian_reference":True,"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"threads":1}})
    print("Frozen source/design, artificial QA, old input hashes; no new statistics or integrals.")

def references():
    grid=read(REF/"GRID.json")["rows"];out=[]
    for n in CFG["ns"]:
        matches=[r for r in grid if (r["variant"],r["a"],r["theta"],r["n"],r["rho"])==("primary",1.,0.,n,.75)]
        need(len(matches)==1,"unique frozen gradient reference")
        r=matches[0];need(r["path_gradient"]!=0,"nonzero reference for sign/SNR")
        out.append({"n":n,"gradient":r["path_gradient"],"Price_gradient":r["Price_gradient"],"disagreement":abs(r["path_gradient"]-r["Price_gradient"]),"numerically_qualified_not_strict":True})
    return out

def gh_reference():
    data=[]
    for order in CFG["GH_orders"]:
        z,e=eigh_tridiagonal(np.zeros(order),np.sqrt(np.arange(1,order)),lapack_driver="stemr")
        w=e[0]**2;w/=math.fsum(w)
        moment=float(np.sum((np.tanh(z/2)**2).astype(np.longdouble)*w))
        data.append({"order":order,"E_tanh_squared":moment,"raw_path_RB_variance_limit":moment/(4*(1-CFG["rho"]**2))})
    span=max(r["raw_path_RB_variance_limit"] for r in data)-min(r["raw_path_RB_variance_limit"] for r in data)
    return {"rows":data,"primary_order":96,"span":span,"limit":CFG["GH_constant_span_gate"],"qualified":span<=CFG["GH_constant_span_gate"],"new_1D_integrals":3,"Gaussian_nodes":sum(CFG["GH_orders"]),"single_label_constant_not_used":True,"strict_certificate":False}

def summaries(values,refs,stream):
    ans=[]
    for j,n in enumerate(CFG["ns"]):
        for m,method in enumerate(METHODS):
            x=values[:,j,m].astype(np.longdouble);mean=float(x.mean());var=float(np.sum((x-x.mean())**2)/(len(x)-1))
            ans.append({"stream":stream,"n":n,"method":method,"banks":len(x),"reference_gradient":refs[j]["gradient"],"mean":mean,"mean_minus_reference":mean-refs[j]["gradient"],"variance_ddof1":var,"sqrt_n_variance":math.sqrt(n)*var if n and method!="raw_path" else None,"zero_fraction":float(np.mean(x==0)),"max_abs":float(np.max(abs(x))),"descriptive_only":True})
    return ans

def group_summaries(means,refs,B,stream):
    ans=[]
    for j,n in enumerate(CFG["ns"]):
        for m,method in enumerate(METHODS):
            x=means[:,j,m].astype(np.longdouble);ref=refs[j]["gradient"]
            ans.append({"stream":stream,"n":n,"method":method,"outer_B":B,"groups":len(x),"reference_gradient":ref,"mean_of_group_means":float(x.mean()),"MSE_to_fixed_reference":float(np.mean((x-ref)**2)),"mean_error":float(x.mean()-ref),"sign_hit_fraction":float(np.mean(x*ref>0)),"zero_fraction":float(np.mean(x==0)),"nested_n_and_B":True,"independent_confirmation":False})
    return ans

def run():
    frozen=read(OUT/"FREEZE.json");verify(frozen["files"]);verify(frozen["local_inputs"])
    need(read(OUT/"CONFIG.json")==CFG,"frozen config changed")
    need(not (OUT/"STATE.json").exists(),"existing run; no retry or overwrite")
    save(OUT/"STATE.json",{"status":"running","utc":utc()});started=time.perf_counter()
    completed=0;gh=None
    try:
        refs=references();save(OUT/"REFERENCES.json",{"rows":refs,"source":rel(REF/"GRID.json"),"source_sha256":sha(REF/"GRID.json"),"sample_mean_never_used_as_truth":True})
        gh=gh_reference();save(OUT/"GH_REFERENCE.json",gh)
        all_values=[];group_values={b:[] for b in CFG["outer_B"]};stream_rows=[];stream_groups=[]
        local=OUT/"local";local.mkdir();(OUT/".gitignore").write_text("local/\n",encoding="utf-8")
        for s in range(8):
            with np.load(MC/f"s{s}/INPUT.npz",allow_pickle=False) as old:
                need(old["active"].shape==(8192,4,2) and old["nuisance_sums"].shape==(8192,6,4),"frozen input shape")
                active=old["active"][:,:2,:].copy();sums=old["nuisance_sums"][:,:,:2].copy()
            need(np.isfinite(active).all() and np.isfinite(sums).all() and np.all(sums[:,0]==0),"input finite/n0 identity")
            terms=active_terms(active,CFG["rho"])
            values=np.stack([estimators(terms,sums[:,j,:]) for j in range(6)],axis=1)
            np.savez_compressed(local/f"s{s}_GRADIENTS.npz",gradients=values)
            all_values.append(values);sr=summaries(values,refs,s);stream_rows.extend(sr)
            group_archive={};sg=[]
            for B in CFG["outer_B"]:
                means=values.reshape(8192//B,B,6,3).mean(axis=1)
                group_archive[f"B{B}"]=means;group_values[B].append(means)
                sg.extend(group_summaries(means,refs,B,s))
            stream_groups.extend(sg);np.savez_compressed(local/f"s{s}_GROUP_MEANS.npz",**group_archive)
            save(OUT/f"STREAM_{s}.json",{"stream":s,"summary":sr,"group_summary":sg,"gradient_shape":list(values.shape),"source_input_sha256":sha(MC/f"s{s}/INPUT.npz")})
            completed+=1;print(f"stream {s}: reused first 2 particles; fixed statistics complete",flush=True)
        combined=np.concatenate(all_values);global_rows=summaries(combined,refs,"all")
        global_groups=[]
        for B in CFG["outer_B"]:
            global_groups.extend(group_summaries(np.concatenate(group_values[B]),refs,B,"all"))
        for row in global_groups:
            vr=next(x["variance_ddof1"] for x in global_rows if x["n"]==row["n"] and x["method"]==row["method"])
            row["reference_SNR_from_sample_variance"]=abs(row["reference_gradient"])*math.sqrt(row["outer_B"]/vr) if vr>0 else None
        csv_save(OUT/"ESTIMATOR_SUMMARY.csv",global_rows);csv_save(OUT/"OUTER_B_SUMMARY.csv",global_groups)
        csv_save(OUT/"STREAM_SUMMARY.csv",stream_rows);csv_save(OUT/"STREAM_OUTER_B.csv",stream_groups)
        counts={"reused_banks":65536,"old_original_K":4,"used_K":2,"reused_prior_vectors":131072,"streams":8,"new_independent_banks":0,"new_labels":0,"RNG_calls":0,"fits":0,"locators":0,"new_bank_n_combinations":393216,"label_bank_terms":1572864,"label_particle_weighted_terms":3145728,"active_label_particle_kernels":524288,"estimator_outputs":1179648,"active_transform_calls":8,"active_particle_transforms":131072,"new_1D_reference_integrals":3,"new_1D_Gaussian_nodes":288,"group_mean_numbers":98208,"global_estimator_rows":18,"global_outer_B_rows":90,"stream_outer_B_rows":720}
        need(len(global_rows)==18 and len(global_groups)==90 and len(stream_groups)==720,"output count")
        save(OUT/"COUNTS.json",counts)
        save(OUT/"RESULTS.json",{"estimators":global_rows,"outer_B":global_groups,"GH_reference_qualified":gh["qualified"],"original_inputs_reused_after_K4_development":True,"independent_confirmation":False,"statistical_CI":False,"prediction_or_success_gates":0,"fixed_reference_source":rel(REF/"GRID.json"),"sample_mean_used_as_truth":False,"no_general_SGD_or_variance_ordering_claim":True,"elapsed_seconds":time.perf_counter()-started})
        save(OUT/"STATE.json",{"status":"complete","utc":utc(),"GH_reference_qualified":gh["qualified"],"new_budgets_added":False},replace=True)
        public={rel(p):sha(p) for p in OUT.rglob("*") if p.is_file() and "local" not in p.parts and p.name!="RESULTS_FREEZE.json"}
        local_files={rel(p):sha(p) for p in local.glob("*.npz")}
        save(OUT/"RESULTS_FREEZE.json",{"utc":utc(),"source_freeze_sha256":sha(OUT/"FREEZE.json"),"files":public,"local_outputs":local_files})
        print(json.dumps({"complete":True,"GH_reference_qualified":gh["qualified"],"counts":counts}))
    except Exception as exc:
        save(OUT/"FAILURE.json",{"utc":utc(),"error":repr(exc),"traceback":traceback.format_exc(),"streams_completed":completed,"new_1D_integrals_executed":3 if gh is not None else 0,"no_automatic_retry":True})
        save(OUT/"STATE.json",{"status":"failed","utc":utc()},replace=True);raise

def check(workspace=False):
    f=read(OUT/"FREEZE.json");r=read(OUT/"RESULTS_FREEZE.json")
    need(r["source_freeze_sha256"]==sha(OUT/"FREEZE.json"),"result/source chain")
    verify(f["files"]);verify(r["files"])
    if workspace:verify(f["local_inputs"]);verify(r["local_outputs"])
    need(read(OUT/"STATE.json")["status"]=="complete","run incomplete")
    result=read(OUT/"RESULTS.json");need(len(result["estimators"])==18 and len(result["outer_B"])==90,"saved rows")
    print(json.dumps({"integrity_passed":True,"workspace":workspace,"GH_reference_qualified":result["GH_reference_qualified"],"statistics_or_integrals_reexecuted":False}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("qa","freeze","run","check"));p.add_argument("--workspace",action="store_true");args=p.parse_args()
    if args.action=="check":check(args.workspace)
    else:globals()[args.action]()
