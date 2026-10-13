"""Independent full-bank replay of frozen outer-gradient diagnostics; no RNG."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"
import argparse,csv,hashlib,itertools,json,math,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp,roots_hermitenorm

ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/"artifacts/nuisance_learning/GRADIENT"
OUT=ROOT/"artifacts/nuisance_learning/GRADIENT_AUDIT"
DOC=ROOT/"docs/08_research_研究工作/外层精度与有限样本伪相关_20261001/GRADIENT_AUDIT.md"
MARKER="<!-- OUTER_GRADIENT_AUDIT_PLAN_END -->"
MC=ROOT/"artifacts/nuisance_dimensions/MC"
REF=ROOT/"artifacts/nuisance_dimensions/NUMERIC"
NS=(0,14,62,254,1022,4094);BS=(16,64,256,1024,4096);METHODS=("Price","raw_path","noise_CV")
COUNTS={"new_scientific_samples":0,"RNG_calls":0,"fits":0,"locators":0,"streams_recomputed":0,"banks_recomputed":0,"bank_n_combinations":0,"label_bank_terms":0,"active_particle_transforms":0,"GH_integrals":0,"GH_nodes":0}
CHECKS=[];FAIL=[]

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,x):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(x,f,ensure_ascii=False,indent=2,default=serial,allow_nan=False);f.write("\n")
def need(ok,msg):
    if not ok:raise AssertionError(msg)
def verify(m):
    for p,h in m.items():need(sha(ROOT/p)==h,"hash mismatch "+p)
def phash():
    s=DOC.read_text(encoding="utf-8").replace("\r\n","\n")
    return hashlib.sha256((s.split(MARKER)[0]+MARKER+"\n").encode()).hexdigest()
def gate(name,error,limit):
    r={"name":name,"error":float(error),"limit":limit,"passed":bool(np.isfinite(error) and error<=limit)}
    CHECKS.append(r)
    if not r["passed"]:FAIL.append(r)

def sources():
    need(read(PROD/"STATE.json")["status"]=="complete","production incomplete")
    f=read(PROD/"FREEZE.json");r=read(PROD/"RESULTS_FREEZE.json");move=read(PROD/"SOURCE_RELOCATION.json")
    need(move["old_path"]=="script/nuisance_learning_gradient.py" and move["new_path"]=="script/outer_learning_gradient_source.py","unexpected relocation")
    need(move["source_freeze_sha256"]==sha(PROD/"FREEZE.json") and move["result_freeze_sha256"]==sha(PROD/"RESULTS_FREEZE.json"),"relocation seal links")
    need(f["files"][move["old_path"]]==move["unchanged_source_sha256"],"relocated source not original bytes")
    public={(move["new_path"] if p==move["old_path"] else p):h for p,h in f["files"].items()}
    public.update(r["files"])
    public[rel(PROD/"RESULTS_FREEZE.json")]=sha(PROD/"RESULTS_FREEZE.json")
    public[rel(PROD/"SOURCE_RELOCATION.json")]=sha(PROD/"SOURCE_RELOCATION.json")
    need(r["source_freeze_sha256"]==sha(PROD/"FREEZE.json"),"producer result chain")
    local={**f["local_inputs"],**r["local_outputs"]};verify(public);verify(local)
    return public,local,move

def bank_replay(z,sums):
    rho=.75;root=math.sqrt(1-rho*rho)
    x=np.stack([z[:,:,0],rho*z[:,:,0]+root*z[:,:,1]],axis=-1)
    velocity=z[:,:,0]-rho*z[:,:,1]/root
    active=[];dots=[];resprod=[]
    for y in itertools.product((-1.,1.),repeat=2):
        yy=np.asarray(y);res=yy*expit(-yy*x)
        active.append(-np.sum(np.logaddexp(0,-yy*x),axis=-1))
        dots.append(res[:,:,1]*velocity);resprod.append(np.prod(res,axis=-1))
    active=np.stack(active,axis=1);dots=np.stack(dots,axis=1);resprod=np.stack(resprod,axis=1)
    result=[]
    for j in range(6):
        s=sums[:,j,:];s=s-s.max(axis=1,keepdims=True)
        h=s[:,None,:]+active
        log_pi=h-logsumexp(h,axis=-1,keepdims=True);pi=np.exp(log_pi)
        price=-np.mean(np.exp(log_pi[:,:,0]+log_pi[:,:,1])*resprod.sum(axis=-1),axis=-1)
        path=-np.mean(np.sum(pi*dots,axis=-1),axis=-1)
        u=(s[:,0]-s[:,1])[:,None];delta=active[:,:,0]-active[:,:,1];v=u+delta
        ab=abs(delta);difference=np.zeros_like(delta);mask=ab>0
        # sigmoid(u)-sigmoid(v) = -sinh(delta/2)/(2 cosh(u/2)cosh(v/2)).
        logmag=ab[mask]/2+np.log(-np.expm1(-ab[mask]))-np.broadcast_to(np.logaddexp(u/2,-u/2),v.shape)[mask]-np.logaddexp(v[mask]/2,-v[mask]/2)
        difference[mask]=-np.sign(delta[mask])*np.exp(logmag)
        cv=np.mean(difference*(dots[:,:,0]-dots[:,:,1]),axis=-1)
        result.append(np.stack([price,path,cv],axis=-1))
    out=np.stack(result,axis=1);need(np.isfinite(out).all(),"nonfinite independent replay")
    gate("Price_RB_range",max(0.,float(abs(out[:,:,0]).max())-.125),1e-12)
    COUNTS["streams_recomputed"]+=1;COUNTS["banks_recomputed"]+=len(z)
    COUNTS["bank_n_combinations"]+=len(z)*6;COUNTS["label_bank_terms"]+=len(z)*6*4
    COUNTS["active_particle_transforms"]+=len(z)*2
    return out

def summarize(x,refs,stream,B=None):
    result=[]
    for j,n in enumerate(NS):
        for k,method in enumerate(METHODS):
            v=x[:,j,k].astype(np.longdouble);mean=float(v.mean());ref=refs[j]
            if B is None:
                var=float(np.sum((v-v.mean())**2)/(len(v)-1))
                row=dict(stream=stream,n=n,method=method,banks=len(v),reference_gradient=ref,mean=mean,mean_minus_reference=mean-ref,variance_ddof1=var,sqrt_n_variance=math.sqrt(n)*var if n and method!="raw_path" else None,zero_fraction=float(np.mean(v==0)),max_abs=float(abs(v).max()),descriptive_only=True)
            else:
                row=dict(stream=stream,n=n,method=method,outer_B=B,groups=len(v),reference_gradient=ref,mean_of_group_means=mean,MSE_to_fixed_reference=float(np.mean((v-ref)**2)),mean_error=float(v.mean()-ref),sign_hit_fraction=float(np.mean(v*ref>0)),zero_fraction=float(np.mean(v==0)),nested_n_and_B=True,independent_confirmation=False)
            result.append(row)
    return result

def compare_records(actual,saved,name,csv_mode=False):
    need(len(actual)==len(saved),name+" row count")
    maxerror=0.
    for a,b in zip(actual,saved):
        need(set(a)==set(b),name+" field names")
        for key,val in a.items():
            other=b[key]
            if csv_mode:
                if val is None:other=None if other=="" else other
                elif isinstance(val,bool):other={"True":True,"False":False}[other]
                elif isinstance(val,(int,float,np.number)):other=float(other)
            if val is None or isinstance(val,(str,bool)):
                need(val==other,name+" categorical "+key)
            elif key in ("n","banks","groups","outer_B","stream"):
                need(val==other,name+" integer "+key)
            else:
                err=abs(float(val)-float(other));maxerror=max(maxerror,err)
                if key in ("sign_hit_fraction","zero_fraction"):need(err==0,name+" exact sign/zero tally")
    gate(name,maxerror,5e-12)

def csv_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def gh_check():
    saved=read(PROD/"GH_REFERENCE.json");out=[]
    for order in (96,128):
        z,w=roots_hermitenorm(order);w=w/math.fsum(w)
        moment=float(np.sum((np.tanh(z/2)**2).astype(np.longdouble)*w))
        constant=moment/(4*(1-.75**2))
        errors=[abs(constant-r["raw_path_RB_variance_limit"]) for r in saved["rows"]]
        gate("GH_reference_"+str(order),max(errors),1e-10)
        out.append({"order":order,"E_tanh_squared":moment,"raw_path_RB_variance_limit":constant,"max_error_to_production":max(errors)})
        COUNTS["GH_integrals"]+=1;COUNTS["GH_nodes"]+=order
    gate("GH_96_128",abs(out[0]["raw_path_RB_variance_limit"]-out[1]["raw_path_RB_variance_limit"]),1e-10)
    return out

def write():
    need(not (OUT/"START.json").exists(),"existing audit attempt; preserve it")
    public,local,move=sources();cfg=read(PROD/"CONFIG.json")
    need(cfg["K"]==2 and cfg["rho"]==.75 and cfg["ns"]==list(NS) and cfg["outer_B"]==list(BS) and cfg["methods"]==list(METHODS),"fixed scope/config")
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/"START.json",{"utc":utc(),"source_sha256":sha(__file__),"source":rel(__file__),"plan_prefix_sha256":phash(),"public_inputs":public,"local_inputs_outputs":local,"source_relocation":move,"before_new_statistics_and_integrals":True})
    started=time.perf_counter();outputs=[]
    try:
        refgrid=read(REF/"GRID.json")["rows"];refs=[]
        for n in NS:
            rr=[r for r in refgrid if (r["variant"],r["a"],r["theta"],r["n"],r["rho"])==("primary",1.,0.,n,.75)]
            need(len(rr)==1,"unique numerical reference");refs.append(rr[0]["path_gradient"])
        need(refs==[r["gradient"] for r in read(PROD/"REFERENCES.json")["rows"]],"reference not data mean")
        allbank=[];groups={B:[] for B in BS};stream_rows=[];stream_groups=[]
        for s in range(8):
            with np.load(MC/f"s{s}/INPUT.npz",allow_pickle=False) as inp:
                z=inp["active"][:,:2,:].copy();noise=inp["nuisance_sums"][:,:,:2].copy()
            need(z.shape==(8192,2,2) and noise.shape==(8192,6,2),"input scope")
            value=bank_replay(z,noise)
            with np.load(PROD/f"local/s{s}_GRADIENTS.npz",allow_pickle=False) as saved:gate(f"s{s}_all_bank_values",float(abs(value-saved["gradients"]).max()),1e-10)
            stats=summarize(value,refs,s);original=read(PROD/f"STREAM_{s}.json")
            compare_records(stats,original["summary"],f"s{s}_summary")
            allbank.append(value);stream_rows.extend(stats);local_groups=[]
            with np.load(PROD/f"local/s{s}_GROUP_MEANS.npz",allow_pickle=False) as saved:
                for B in BS:
                    means=np.asarray(value.astype(np.longdouble).reshape(8192//B,B,6,3).mean(axis=1),dtype=float)
                    gate(f"s{s}_B{B}_all_group_means",float(abs(means-saved[f"B{B}"]).max()),1e-10)
                    groups[B].append(means);local_groups.extend(summarize(means,refs,s,B))
            compare_records(local_groups,original["group_summary"],f"s{s}_groups")
            stream_groups.extend(local_groups)
            print(json.dumps({"stream_completed":s,"failures":len(FAIL)}),flush=True)
        combined=np.concatenate(allbank);summary=summarize(combined,refs,"all");block=[]
        for B in BS:block.extend(summarize(np.concatenate(groups[B]),refs,"all",B))
        for row in block:
            variance=next(v["variance_ddof1"] for v in summary if v["n"]==row["n"] and v["method"]==row["method"])
            row["reference_SNR_from_sample_variance"]=abs(row["reference_gradient"])*math.sqrt(row["outer_B"]/variance) if variance>0 else None
        production=read(PROD/"RESULTS.json")
        compare_records(summary,production["estimators"],"all_estimator_summary")
        compare_records(block,production["outer_B"],"all_outer_B_summary")
        for filename,data in (("ESTIMATOR_SUMMARY.csv",summary),("OUTER_B_SUMMARY.csv",block),("STREAM_SUMMARY.csv",stream_rows),("STREAM_OUTER_B.csv",stream_groups)):
            compare_records(data,csv_rows(PROD/filename),filename,True)
        gh=gh_check()
        save(OUT/"RECOMPUTED.json",{"estimators":summary,"outer_B":block,"GH":gh,"stream_statistics_rows":len(stream_rows),"stream_outer_B_rows":len(stream_groups)})
        outputs.append(OUT/"RECOMPUTED.json")
        save(OUT/"CHECKS.json",{"checks":CHECKS,"all_passed":not FAIL});outputs.append(OUT/"CHECKS.json")
        save(OUT/"RECEIPT.json",{"utc":utc(),"state":"complete","passed":not FAIL,"start_sha256":sha(OUT/"START.json"),"files":{rel(p):sha(p) for p in outputs},"counts":COUNTS,"scope":{"all_bank_values":1179648,"local_array_files_compared":16,"group_mean_values":98208,"new_independent_confirmation":False,"strict_integral_certificate":False,"original_RNG_replayed":False},"checks":len(CHECKS),"failures":FAIL,"elapsed_seconds":time.perf_counter()-started})
        print(json.dumps({"passed":not FAIL,"checks":len(CHECKS),"counts":COUNTS,"failures":len(FAIL)}))
    except Exception as exc:
        save(OUT/"FAILURE.json",{"utc":utc(),"error":repr(exc),"traceback":traceback.format_exc(),"counts":COUNTS,"numeric_failures":FAIL});raise

def check(workspace=False):
    s=read(OUT/"START.json");r=read(OUT/"RECEIPT.json")
    need(s["source_sha256"]==sha(__file__) and s["plan_prefix_sha256"]==phash(),"audit source/plan")
    need(r["state"]=="complete" and r["start_sha256"]==sha(OUT/"START.json"),"audit receipt")
    verify(s["public_inputs"]);verify(r["files"])
    if workspace:verify(s["local_inputs_outputs"])
    need(r["passed"] and not r["failures"],"retained audit failure")
    print(json.dumps({"passed":True,"workspace":workspace,"statistics_integrals_RNG_reexecuted":False,"receipt_sha256":sha(OUT/"RECEIPT.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("write","check"));p.add_argument("--workspace",action="store_true");a=p.parse_args()
    write() if a.action=="write" else check(a.workspace)
