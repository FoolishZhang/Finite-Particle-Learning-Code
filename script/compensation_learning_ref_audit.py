"""Independent finite-point audit of frozen compensation_learning REF.

No producer numerical functions are imported. ``check`` checks recorded bytes;
only ``write`` executes the separately declared deterministic quadrature audit.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from datetime import datetime, timezone
from functools import lru_cache
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
from scipy.linalg import eigh_tridiagonal
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "artifacts/compensation_learning/REF"
OUT = ROOT / "artifacts/compensation_learning/REF_AUDIT"
OLD = ROOT / "artifacts/covariance_boundary/NUMERIC"
PLAN = ROOT / "docs/08_research_研究工作/补偿学习与连续目标_20261001/REF_REVIEW.md"
MARKER = "<!-- COMPENSATION_REF_AUDIT_PLAN_END -->"
POINTS = ((.5,.5),(.5,2.5),(2.5,.5),(2.5,2.5),(1.5,1.5),(1.25,2.))
COUNTS = {"new_scientific_labels":0,"new_RNG_calls":0,"new_fits":0,
          "private_node_pairs":0,"expit_scalar_evaluations":0,
          "kernel_vector_calls":0,"PMF_integral_executions":0,
          "outer_PMF_scalar_terms":0,"GH_rules_constructed":0}
FAILURES = []

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def save(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+"\n",encoding="utf-8")
def require(ok,msg):
    if not ok: raise AssertionError(msg)
def gate(name,value,limit):
    ok=bool(np.isfinite(value) and value<=limit)
    if not ok: FAILURES.append({"name":name,"value":float(value),"limit":limit})
    return ok
def verify(mapping):
    for p,h in mapping.items(): require(sha(ROOT/p)==h,"hash mismatch: "+p)
def plan_hash():
    s=PLAN.read_text(encoding="utf-8").replace("\r\n","\n")
    return hashlib.sha256((s.split(MARKER)[0]+MARKER+"\n").encode()).hexdigest()

def source_chain(workspace):
    f=read(REF/"FREEZE.json"); s=read(REF/"RESULTS_FREEZE.json")
    require(s["source_freeze_sha256"]==sha(REF/"FREEZE.json"),"source seal link")
    public={**f["files"],**s["files"],**read(REF/"INPUT_HASHES.json")}
    for p in (REF/"RESULTS_FREEZE.json",REF/"STATE.json"):
        public[rel(p)]=sha(p)
    verify(public)
    if workspace: verify(s["local_arrays"])
    require(read(REF/"QUALIFICATION.json")["all_qualified"],"original qualification false")
    require(read(REF/"STATE.json")["status"]=="complete","REF not complete")
    return public,s["local_arrays"]

@lru_cache(None)
def gh(order):
    x,v=eigh_tridiagonal(np.zeros(order),np.sqrt(np.arange(1,order)),lapack_driver="stemr")
    w=v[0]**2
    require(abs(math.fsum(w)-1)<1e-12,"GH mass")
    w=w/math.fsum(w)
    COUNTS["GH_rules_constructed"]+=1
    return x,w

def kernel(z,d,w,order):
    e,v=gh(order); z=np.asarray(z,dtype=float).reshape(-1)
    lam=w/math.sqrt(d-1); private=math.sqrt(1-lam*lam)
    x=lam*z[:,None]+private*e[None,:]
    p=expit(x); q=expit(-x)
    h=p@v; qbar=q@v
    path=(z[:,None]-lam*e[None,:]/private)/math.sqrt(d-1)
    hp=(p*q*path)@v
    hs=(p*q*(z[:,None]-lam*(1-2*p))/math.sqrt(d-1))@v
    require(np.all(h>0) and np.all(qbar>0),"nonpositive private probability")
    COUNTS["kernel_vector_calls"]+=1
    COUNTS["private_node_pairs"]+=x.size
    COUNTS["expit_scalar_evaluations"]+=2*x.size
    return h,qbar,hp,hs

def logvalues(k):
    h,q,hp,_=k
    return np.array([np.log(h),np.log(q),hp/h,-hp/q])

@lru_cache(None)
def binomial(d,w):
    z,_=gh(384); h,q,hp,_=kernel(z,d,w,256)
    a=np.ones((len(z),1)); da=np.zeros_like(a)
    for _ in range(d//2):
        b=np.zeros((len(z),a.shape[1]+1)); db=np.zeros_like(b)
        b[:,:-1]+=a*q[:,None]; b[:,1:]+=a*h[:,None]
        db[:,:-1]+=da*q[:,None]-a*hp[:,None]
        db[:,1:]+=da*h[:,None]+a*hp[:,None]
        a,da=b,db
    return a,da

@lru_cache(None)
def pmf(d,w0,w1):
    _,v=gh(384); a,da=binomial(d,w0); b,db=binomial(d,w1)
    raw=a.T@(v[:,None]*b)
    draw=np.stack([da.T@(v[:,None]*b),a.T@(v[:,None]*db)],axis=-1)
    mass=math.fsum(raw.ravel()); gm=draw.sum(axis=(0,1))
    p=raw/mass; gp=(draw-p[...,None]*gm)/mass
    require(np.all(p>0),"nonpositive PMF")
    COUNTS["PMF_integral_executions"]+=1
    COUNTS["outer_PMF_scalar_terms"]+=384*p.size*3
    return p,gp/p[...,None],mass,gm

def kl(p,q):
    p=np.asarray(p).ravel();q=np.asarray(q).ravel()
    p=p/math.fsum(p);q=q/math.fsum(q)
    u=(p-q)/(p+q);small=abs(u)<=.05
    us=u[small];correction=np.zeros_like(us)
    for k in range(1,17): correction+=us**(2*k+1)/(2*k+1)
    terms=np.empty_like(p)
    terms[small]=(p[small]+q[small])*(us*us+(1+us)*correction)
    terms[~small]=p[~small]*(np.log(p[~small])-np.log(q[~small]))-p[~small]+q[~small]
    return math.fsum(terms)

def saved_arrays(d,cfg):
    saved=read(REF/f"REFERENCE_GRID_d{d}.json")
    a=np.load(REF/f"local/reference_grid_d{d}.npz",allow_pickle=False)
    ws=a["w"];ps=a["pmf"];gs=a["grad_log_pattern"]
    require(ws.shape==(81,2) and len(ps)==81 and ps.shape[1]==3,"saved shape")
    require(np.array_equal(ws,np.asarray(saved["points"])),"saved point order")
    teacher=np.asarray(read(OLD/f"REFERENCE_d{d}.json")["pmf"][0])
    rows=[]
    for j,w in enumerate(ws):
        p,g=ps[j],gs[j]; st=saved["qualification"][j]["metrics"]
        risks=[kl(teacher,x) for x in p]
        metrics={"normalization":float(abs(p.sum(axis=(1,2))-1).max()),
          "normalized_gradient_sum":float(abs((p[...,None]*g).sum(axis=(1,2))).max()),
          "complement_error":float(abs(p-p[:,::-1,::-1]).max()),
          "PMF_L1_difference":max(float(abs(p[k]-p[l]).sum()) for k,l in itertools.combinations(range(3),2)),
          "PMF_log_difference":max(float(abs(np.log(p[k])-np.log(p[l])).max()) for k,l in itertools.combinations(range(3),2)),
          "PMF_log_gradient_difference":max(float(abs(g[k]-g[l]).max()) for k,l in itertools.combinations(range(3),2)),
          "reference_risks":risks,"reference_risk_span":max(risks)-min(risks)}
        other=int(np.flatnonzero(np.all(ws==w[::-1],axis=1))[0])
        metrics["group_swap_error"]=float(abs(p-ps[other].transpose(0,2,1)).max())
        metrics["group_swap_gradient_error"]=float(abs(g-gs[other].transpose(0,2,1,3)[...,::-1]).max())
        require(np.all(p>0),"saved PMF positivity")
        checks=[gate(f"d{d}/{j}/{k}",metrics[k],cfg[lim]) for k,lim in
          (("normalization","normalization_gate"),("normalized_gradient_sum","PMF_gradient_mass_gate"),
           ("complement_error","normalization_gate"),("group_swap_error","normalization_gate"),
           ("group_swap_gradient_error","PMF_log_gradient_gate"),("PMF_L1_difference","PMF_L1_gate"),
           ("PMF_log_difference","PMF_log_gate"),("PMF_log_gradient_difference","PMF_log_gradient_gate"),
           ("reference_risk_span","reference_risk_gate"))]
        residual=max(abs(metrics[k]-st[k]) for k in ("complement_error","PMF_L1_difference","PMF_log_difference","PMF_log_gradient_difference","reference_risk_span"))
        residual=max(residual,max(abs(x-y) for x,y in zip(risks,st["reference_risks"])))
        checks.append(gate(f"d{d}/{j}/public metric reproduction",residual,1e-12))
        # Only saved public values are available for raw, unnormalized mass.
        checks.append(gate(f"d{d}/{j}/recorded raw mass",st["PMF_mass_error"],cfg["PMF_mass_gate"]))
        checks.append(gate(f"d{d}/{j}/recorded raw gradient mass",st["PMF_gradient_mass_error"],cfg["PMF_gradient_mass_gate"]))
        rows.append({"d":d,"w":w,"metrics":metrics,"public_reproduction_max_error":residual,"passed":all(checks)})
    return rows,ws,ps,gs

def audit_kernel(d,cfg):
    z=np.linspace(-12,12,97); saved=read(REF/f"KERNEL_d{d}.json"); rows=[]
    for st in saved:
        low=logvalues(kernel(z,d,st["w"],64)); high=logvalues(kernel(z,d,st["w"],128))
        metrics={"log_difference":float(abs(low[:2]-high[:2]).max()),
          "log_gradient_difference":float(abs(low[2:]-high[2:]).max()),
          "normalization":max(float(abs(np.logaddexp(*x[:2])).max()) for x in (low,high)),
          "gradient_normalization":max(float(abs(np.exp(x[0])*x[2]+np.exp(x[1])*x[3]).max()) for x in (low,high)),
          "symmetry":max(float(abs(x[0]-x[1][::-1]).max()) for x in (low,high)),
          "gradient_symmetry":max(float(abs(x[2]-x[3][::-1]).max()) for x in (low,high))}
        checks=[gate(f"kernel/d{d}/{st['w']}/{k}",value,cfg["kernel_log_gate"] if k=="log_difference" else cfg["kernel_log_gradient_gate"] if k=="log_gradient_difference" else cfg["normalization_gate"]) for k,value in metrics.items()]
        rows.append({"d":d,"w":st["w"],**metrics,"public_statistic_difference":max(abs(metrics[k]-st[k]) for k in metrics),"passed":all(checks)})
    return rows

def audit_quad(d,cfg):
    rows=[]
    for st in read(REF/f"QUAD_d{d}.json"):
        h,q,hp,hs=map(lambda x:float(x[0]),kernel([st["z"]],d,st["w"],256))
        h0,q0,g0=st["values"]
        le=max(abs(math.log(h/h0)),abs(math.log(q/q0)))
        ge=max(abs(hp/h-g0/h0),abs(-hp/q+g0/q0),abs(hs/h-g0/h0),abs(-hs/q+g0/q0))
        checks=[gate(f"quad/d{d}/{st['w']}/{st['z']}/log",le,cfg["quad_log_gate"]),gate(f"quad/d{d}/{st['w']}/{st['z']}/gradient",ge,cfg["quad_log_gradient_gate"])]
        rows.append({"d":d,"w":st["w"],"z":st["z"],"values":[h,q,hp,hs],"log_difference":le,"log_gradient_difference":ge,"path_Stein_difference":abs(hp-hs),"passed":all(checks)})
    return rows

def rebuild(d,cfg,ws,ps,gs):
    rows=[]; stored=[]
    for w in POINTS:
        p,g,mass,gm=pmf(d,*w); j=int(np.flatnonzero(np.all(ws==w,axis=1))[0])
        errors={"PMF_L1":max(float(abs(p-x).sum()) for x in ps[j]),
          "log_PMF":max(float(abs(np.log(p)-np.log(x)).max()) for x in ps[j]),
          "log_gradient":max(float(abs(g-x).max()) for x in gs[j]),
          "raw_mass":abs(mass-1),"raw_gradient_mass":float(abs(gm).max())}
        checks=[gate(f"rebuild/d{d}/{w}/{k}",errors[k],cfg[lim]) for k,lim in (("PMF_L1","PMF_L1_gate"),("log_PMF","PMF_log_gate"),("log_gradient","PMF_log_gradient_gate"),("raw_mass","PMF_mass_gate"),("raw_gradient_mass","PMF_gradient_mass_gate"))]
        rows.append({"d":d,"w":w,"errors":errors,"passed":all(checks)})
        stored.append({"d":d,"w":w,"pmf":p,"grad_log_pattern":g,"raw_mass":mass,"raw_gradient_mass":gm})
    finite=[]; step=5e-5
    for w in POINTS[-2:]:
        _,g,_,_=pmf(d,*w)
        for k in (0,1):
            logs=[]
            for offset in (-2,-1,1,2):
                v=list(w);v[k]+=offset*step
                logs.append(np.log(pmf(d,*v)[0]))
            fd=(logs[0]-8*logs[1]+8*logs[2]-logs[3])/(12*step)
            error=float(abs(fd-g[...,k]).max())
            finite.append({"d":d,"w":w,"parameter":k,"step":step,"error":error,"passed":gate(f"FD/d{d}/{w}/{k}",error,cfg["finite_difference_log_gradient_gate"])})
    anchors=[];old=read(OLD/f"REFERENCE_d{d}.json")
    for st in read(REF/f"ANCHORS_d{d}.json"):
        p,g,mass,gm=pmf(d,*st["w"]); j=old["names"].index(st["name"])
        e=float(abs(p-np.asarray(old["pmf"][j])).sum());risk=kl(old["pmf"][0],p)
        delta=max(abs(risk-st["joint_KL"]),abs(risk-old["risk_to_teacher"][j]))
        checks=[gate(f"anchor/d{d}/{st['name']}/PMF",e,cfg["old_anchor_L1_gate"]),gate(f"anchor/d{d}/{st['name']}/KL",delta,cfg["old_comp_KL_gate"])]
        anchors.append({"d":d,"name":st["name"],"w":st["w"],"PMF_L1":e,"forward_KL_teacher_candidate":risk,"KL_difference":delta,"passed":all(checks)})
    return rows,stored,finite,anchors

def write():
    require(not (OUT/"START.json").exists(),"audit already started; preserve prior attempt")
    OUT.mkdir(parents=True,exist_ok=True)
    public,local=source_chain(True)
    start={"utc":now(),"source":rel(Path(__file__)),"source_sha256":sha(__file__),"plan_prefix_sha256":plan_hash(),"public_inputs":public,"local_inputs":local,"orders":{"private_kernel":[64,128],"quad_comparison":256,"PMF_private":256,"PMF_outer":384},"before_audit_integrals":True,"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"threads":1}}
    save(OUT/"START.json",start)
    begin=time.perf_counter();cfg=read(REF/"CONFIG.json");allrows={k:[] for k in ("saved_arrays","kernel","quad","rebuild","finite_difference","anchors")};outputs=[]
    try:
        for d in (16,64):
            rows,ws,ps,gs=saved_arrays(d,cfg);allrows["saved_arrays"]+=rows
            allrows["kernel"]+=audit_kernel(d,cfg);allrows["quad"]+=audit_quad(d,cfg)
            rows,stored,finite,anchors=rebuild(d,cfg,ws,ps,gs)
            allrows["rebuild"]+=rows;allrows["finite_difference"]+=finite;allrows["anchors"]+=anchors
            p=OUT/f"REBUILT_d{d}.json";save(p,stored);outputs.append(p)
            p=OUT/f"DETAIL_d{d}.json";save(p,{k:[x for x in v if x["d"]==d] for k,v in allrows.items()});outputs.append(p)
            print(json.dumps({"dimension_completed":d,"failures_so_far":len(FAILURES)},ensure_ascii=False),flush=True)
        summary={k:{"rows":len(v),"passed":sum(bool(x["passed"]) for x in v)} for k,v in allrows.items()}
        counts={**COUNTS,"saved_PMFs_checked":486,"reconstructed_base_PMFs":12,"finite_difference_directions":8,"perturbed_PMFs":32,"anchor_PMFs":4,"full_PMF_evaluation_requests":48,"binomial_tables_computed":binomial.cache_info().currsize}
        receipt={"utc":now(),"status":"complete","passed":not FAILURES,"start_sha256":sha(OUT/"START.json"),"summary":summary,"counts":counts,"failures":FAILURES,"elapsed_seconds":time.perf_counter()-begin,"files":{rel(p):sha(p) for p in outputs},"limits":["finite points and numerical agreement only, no rigorous quadrature remainder","raw masses for all saved PMFs were not stored; only public raw-mass gates checked","no complete repeat of producer finite differences or adaptive quad","no RUN outcomes accessed; zero scientific data, fits, or RNG"]}
        save(OUT/"INDEPENDENT_AUDIT.json",receipt)
        print(json.dumps(receipt,ensure_ascii=False,default=serial))
    except Exception as exc:
        save(OUT/"FAILURE.json",{"utc":now(),"error":repr(exc),"traceback":traceback.format_exc(),"counts":COUNTS,"numerical_failures":FAILURES,"completed":allrows})
        raise

def check(workspace):
    start=read(OUT/"START.json");r=read(OUT/"INDEPENDENT_AUDIT.json")
    require(r["status"]=="complete","audit incomplete")
    require(r["start_sha256"]==sha(OUT/"START.json"),"start hash")
    require(start["source_sha256"]==sha(__file__),"auditor source changed")
    require(start["plan_prefix_sha256"]==plan_hash(),"audit plan changed")
    verify(start["public_inputs"]);verify(r["files"])
    if workspace: verify(start["local_inputs"])
    require(r["passed"] and not r["failures"],"recorded audit failed")
    print(json.dumps({"passed":True,"workspace":workspace,"integrals_reexecuted":False,"receipt_sha256":sha(OUT/"INDEPENDENT_AUDIT.json")}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("write","check"));parser.add_argument("--workspace",action="store_true");args=parser.parse_args()
    if args.phase=="write":write()
    else:check(args.workspace)
