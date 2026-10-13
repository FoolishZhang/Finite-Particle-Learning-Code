"""Independent bounded Fourier/CF reconstruction and complete saved arithmetic."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
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

ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/"artifacts/nuisance_dimensions/NUMERIC"
OLD=ROOT/"artifacts/weak_emission/NUMERIC"
OUT=ROOT/"artifacts/nuisance_dimensions/FOURIER_AUDIT"
DOC=ROOT/"docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001/FOURIER_AUDIT.md"
MARKER="<!-- NUISANCE_FOURIER_AUDIT_PLAN_END -->"
PANELS=(0.,.125,.5,2.,6.,14.)
COUNT={"new_scientific_samples":0,"RNG_calls":0,"fits":0,"locators":0,
       "native_integrals":0,"native_Gaussian_nodes":0,"native_phase_terms":0,
       "nuisance_integrals":0,"nuisance_Gaussian_nodes":0,"nuisance_phase_terms":0,
       "GH_rules":0,"GL_rules":0,"evaluations":0,"Decimal_KLs":0}
FAIL=[]

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return read(p)["rows"]
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False,default=serial);f.write("\n")
def need(ok,msg):
    if not ok:raise AssertionError(msg)
def near(x,y,tol=1e-12):return abs(float(x)-float(y))<=tol
def verify(mapping):
    for p,h in mapping.items():need(sha(ROOT/p)==h,"hash mismatch: "+p)
def phash():
    s=DOC.read_text(encoding="utf-8").replace("\r\n","\n")
    return hashlib.sha256((s.split(MARKER)[0]+MARKER+"\n").encode()).hexdigest()
def gate(name,error,limit,**meta):
    ok=bool(np.isfinite(error) and error<=limit)
    r={"name":name,"error":float(error),"limit":float(limit),"passed":ok,**meta}
    if not ok:FAIL.append(r)
    return r

def sources():
    need((PROD/"RESULTS_FREEZE.json").exists(),"Fourier result seal required")
    need(read(PROD/"STATE.json")["state"]=="complete","Fourier production must be complete")
    f=read(PROD/"FREEZE.json");r=read(PROD/"RESULTS_FREEZE.json")
    public={rel(PROD/n):h for n,h in r["files"].items()}
    public.update({x["snapshot"]:x["sha256"] for x in f["inputs"]})
    public[rel(PROD/"UNIT_CHECK.json")]=f["unit_sha256"]
    public[rel(PROD/"CONFIG.json")]=f["config_sha256"]
    public[rel(PROD/"RESULTS_FREEZE.json")]=sha(PROD/"RESULTS_FREEZE.json")
    old=read(OLD/"RESULTS_FREEZE.json")
    for name in ("CONFIG.json","FREEZE.json","GRID.json","RESULTS.json","STATE.json"):
        public[rel(OLD/name)]=old["files"][name]
    public[rel(OLD/"RESULTS_FREEZE.json")]=sha(OLD/"RESULTS_FREEZE.json")
    verify(public)
    optional={rel(PROD/n):h for n,h in r["optional_workspace_arrays"].items()}
    return public,optional

def key(r):return (r["variant"],r["a"],r["theta"],r["n"],r["rho"])
def arithmetic(cfg,grid,evaluations,fd,checks,result,index):
    table={key(r):r for r in evaluations};gtable={key(r):r for r in grid}
    need(len(table)==len(evaluations)==1653 and len(grid)==1197 and len(fd)==152,"saved counts")
    need(all(table[k]==v for k,v in gtable.items()),"grid vs full cache")
    fdm={};fdout=[]
    for st in fd:
        v=[table[("primary",st["a"],st["theta"],st["n"],rho)]["delta_F"] for rho in st["rhos"]]
        need(v==st["values"],"FD saved values")
        if st["scheme"]=="central":coef=[1,-8,8,-1];den=12*st["step"]
        else:coef=[-25,48,-36,16,-3];den=12*st["step"]*(1 if st["rhos"][1]>st["rhos"][0] else -1)
        derivative=float(sum(np.longdouble(c)*np.longdouble(x) for c,x in zip(coef,v))/np.longdouble(den))
        need(near(derivative,st["derivative"],1e-15),"FD arithmetic")
        target=table[("primary",st["a"],st["theta"],st["n"],st["rho"])]
        need(st["path_gradient"]==target["path_gradient"],"FD target gradient")
        fdm[(st["a"],st["theta"],st["n"],st["rho"],st["step"])]=derivative
        fdout.append({"a":st["a"],"theta":st["theta"],"n":st["n"],"rho":st["rho"],"step":st["step"],"difference":abs(derivative-st["derivative"]),"passed":True})
    native={(r["variant"],r["a"],r["rho"]):r for r in index["native_rows"]}
    nuisance={(r["variant"],r["a"]):r for r in index["nuisance_rows"]}
    out=[];g=cfg["gates"]
    for st in checks:
        a=st["a"];n=st.get("n",0);scale=math.sqrt(max(1,n)) if a==1 else 1.
        name=st["name"]
        if name.startswith("variant_"):
            rs=[table[(v,a,st["theta"],n,st["rho"])] for v in cfg["variants"]]
            if name=="variant_pair_moment":err=max(float(np.ptp([r[k] for r in rs])) for k in ("teacher_spin_moment","candidate_spin_moment"));lim=g["pair"]
            elif name=="variant_joint_KL":err=float(np.ptp([r["joint_KL"] for r in rs]));lim=g["a1_KL"] if a==1 else g["weak_KL"]
            else:
                field=name[len("variant_"):];err=scale*float(np.ptp([r[field] for r in rs]));lim=g["a1_value_scaled" if field=="delta_F" else "a1_gradient_scaled"] if a==1 else g["weak_value" if field=="delta_F" else "weak_gradient"]
        elif name=="primary_FD_path":
            der=fdm[(a,st["theta"],n,st["rho"],st["step"])];r=table[("primary",a,st["theta"],n,st["rho"])]
            err=scale*abs(der-r["path_gradient"]);lim=g["a1_FD_scaled"] if a==1 else g["weak_FD"]
        elif name=="Price_path":
            r=table[(st["variant"],a,st["theta"],n,st["rho"])];err=scale*abs(r["Price_gradient"]-r["path_gradient"]);lim=g["a1_gradient_scaled"] if a==1 else g["weak_gradient"]
        elif name=="active_mean_drift":err=abs(native[(st["variant"],a,st["rho"])]["centered_A_mean"]);lim=g["mean_drift"]
        elif name=="CF_modulus_roundoff":
            r=nuisance[(st["variant"],a)];err=r["CF_maximum_positive_roundoff"];lim=g["CF_roundoff"]
            need(st["repaired_count"]==r["CF_positive_roundoff_count"],"CF repair count")
        else:raise AssertionError("unknown gate "+name)
        need(near(err,st["error"],1e-15),"error statistic "+name)
        need(lim==st["limit"] and st["passed"]==(st["error"]<=lim),"gate decision/limit")
        out.append({"name":name,"difference":abs(err-st["error"]),"original_passed":st["passed"]})
    predictions=[]
    for st in result["prediction_readouts"]:
        name=st["name"]
        if name=="near_tie_gradient":
            r=table[("primary",1.,st["theta"],4094,st["rho"])];actual=math.sqrt(4094)*r["Price_gradient"]
            expect=-(r["teacher_spin_moment"]+r["candidate_spin_moment"])/(4*math.sqrt(math.pi*r["variance_U"]))
            error=abs(actual-expect)/abs(expect);passed=error<=g["prediction_relative"]
            need(near(actual,st["actual"],1e-15) and near(expect,st["expected"],1e-15) and near(error,st["relative_error"],1e-12),"near tie readout")
        elif name=="large_n_endpoint_preference":
            value=table[("primary",1.,st["theta"],4094,.75)]["delta_F"]-table[("primary",1.,st["theta"],4094,st["reference_rho"])]["delta_F"]
            need(value==st["difference"],"endpoint preference");passed=value<0
        elif name=="null_sixth_order":
            actual=table[("primary",.05,0.,st["n"],.75)]["delta_F"]/.05**6
            expect=(10-3*(st["n"]+2))*.75**2/512;error=abs(actual-expect)/abs(expect)
            need(near(actual,st["actual"],1e-15) and expect==st["expected"] and near(error,st["relative_error"],1e-12),"sixth order readout")
            passed=error<=g["prediction_relative"]
        elif name=="fixed_noise_null_no_padding":
            value=table[("primary",1.,0.,0,.75)]["delta_F"];need(value==st["difference"],"null n0");passed=value>0
        else:raise AssertionError("unknown readout")
        need(passed==st["passed"],"readout decision")
        predictions.append({"name":name,"original_passed":passed})
    need(len(predictions)==10 and sum(p["original_passed"] for p in predictions)==result["prediction_passed"],"readout count")
    need(result["numerically_qualified"]==all(r["passed"] for r in checks),"global qualification")
    return {"checks":out,"FD":fdout,"readouts":predictions,"passed":True},table

@lru_cache(None)
def rules(order):
    z,v=eigh_tridiagonal(np.zeros(order),np.sqrt(np.arange(1,order)),lapack_driver="stemr")
    w=v[0]**2;w/=math.fsum(w)
    k=np.arange(1,order);gl,u=eigh_tridiagonal(np.zeros(order),k/np.sqrt(4*k*k-1),lapack_driver="stemr")
    gw=2*u[0]**2;gw*=2/math.fsum(gw)
    t=np.concatenate([(hi+lo)/2+(hi-lo)*gl/2 for lo,hi in zip(PANELS[:-1],PANELS[1:])])
    iw=np.concatenate([(hi-lo)*gw/2 for lo,hi in zip(PANELS[:-1],PANELS[1:])])
    COUNT["GH_rules"]+=1;COUNT["GL_rules"]+=1
    return z,w,t,iw

def weighted_complex(values,weights):
    return np.sum(values.astype(np.clongdouble)*weights.astype(np.longdouble)[None,:],axis=1).astype(np.complex128)

@lru_cache(None)
def nuisance(a,order):
    z,w,t,_=rules(order);ell=float(np.sum(np.logaddexp(a*z/2,-a*z/2).astype(np.longdouble)*w))
    score=-np.logaddexp(0,-a*z);mean=float(np.sum(score.astype(np.longdouble)*w));centered=score+ell
    inc=[]
    for first in range(0,len(t),32):inc.extend(weighted_complex(np.expm1(1j*t[first:first+32,None]*centered[None,:]),w))
    inc=np.asarray(inc);delta=2*inc.real+abs(inc)**2
    maximum=float(max(0.,delta.max()));count=int(np.sum(delta>0))
    gate("CF_modulus_roundoff",maximum,1e-12,a=a,order=order)
    delta=np.minimum(delta,0.)
    logmod=np.empty_like(delta);near_one=delta>-.5
    with np.errstate(divide="ignore",invalid="raise"):
        logmod[near_one]=np.log1p(delta[near_one])
        logmod[~near_one]=2*np.log(abs(1+inc[~near_one]))
    COUNT["nuisance_integrals"]+=1;COUNT["nuisance_Gaussian_nodes"]+=order;COUNT["nuisance_phase_terms"]+=order*len(t)
    return {"ell":ell,"mean":mean,"variance":float(np.sum((score-mean)**2*w)),"logmod":logmod,"roundoff_count":count,"roundoff_max":maximum}

@lru_cache(None)
def native(a,rho,order):
    z,w,t,_=rules(order);nu=nuisance(a,order)
    z1=np.repeat(z,order);z2=np.tile(z,order);mass=np.multiply.outer(w,w).ravel()
    x2=rho*z1+math.sqrt(1-rho*rho)*z2
    score=-np.logaddexp(0,-a*z1)-np.logaddexp(0,-a*x2)+2*nu["ell"]
    ap=a*expit(-a*x2)*(z1-rho*z2/math.sqrt(1-rho*rho))
    qq=expit(-a*z1)*expit(-a*x2)
    inc=[];marked_q=[];marked_prime=[]
    for first in range(0,len(t),32):
        ex=np.expm1(1j*t[first:first+32,None]*score[None,:])
        inc.extend(weighted_complex(ex,mass))
        marked_q.extend(float(qq@mass)+weighted_complex(ex,mass*qq))
        marked_prime.extend(float(ap@mass)+weighted_complex(ex,mass*ap))
    inc=np.asarray(inc);h=1+inc;marked_q=np.asarray(marked_q);marked_prime=np.asarray(marked_prime)
    # Derivative of |E exp(i t A)|^2, evaluated through complex products.
    path=-np.imag(marked_prime*np.conjugate(h))
    price=np.real(marked_q*np.conjugate(h))
    spin=(expit(a*z1)-expit(-a*z1))*(expit(a*x2)-expit(-a*x2))
    mean=float(score@mass)
    COUNT["native_integrals"]+=1;COUNT["native_Gaussian_nodes"]+=order**2;COUNT["native_phase_terms"]+=order**2*len(t)
    return {"inc":inc,"path":path,"price":price,"spin":float(np.sum(spin.astype(np.longdouble)*mass)),"mean":mean,"ap_mean":float(ap@mass)}

def dkl(rp,rq):
    with localcontext() as ctx:
        ctx.prec=80;p=(1+Decimal.from_float(float(rp)))/2;q=(1+Decimal.from_float(float(rq)))/2
        ans=p*(p/q).ln()+(1-p)*((1-p)/(1-q)).ln()
    COUNT["Decimal_KLs"]+=1
    need(ans>=0,"negative Decimal KL")
    return float(ans)

def evaluate(a,theta,n,rho,order):
    nu=nuisance(a,order);same=native(a,rho,order);diff=native(a,-rho,order);zero=native(a,0.,order)
    rt=0. if theta==0 else native(a,theta,order)["spin"]
    rq=0. if rho==0 else same["spin"];p=(1+rt)/2
    _,_,t,iw=rules(order);factor=np.ones_like(t) if n==0 else np.exp(n*nu["logmod"])
    def moddiff(a,b):
        # Direct complex-CF algebra retaining the real increment rather than
        # subtracting two rounded moduli close to one.
        return 2*(a.real-b.real)+np.real((a-b)*np.conjugate(a+b))
    num=p*moddiff(same["inc"],zero["inc"])+(1-p)*moddiff(diff["inc"],zero["inc"])
    kernel=np.sinh(math.pi*t)
    val=factor*num/(t*kernel)
    path=2*factor*(p*same["path"]-(1-p)*diff["path"])/kernel
    price=-2*a*a*t*factor*(p*same["price"]-(1-p)*diff["price"])/kernel
    integral=lambda x:float(np.sum(x.astype(np.longdouble)*iw.astype(np.longdouble)))
    COUNT["evaluations"]+=1
    return {"a":a,"theta":theta,"n":n,"rho":rho,"order":order,"delta_F":integral(val),"path_gradient":integral(path),"Price_gradient":integral(price),"teacher_spin_moment":rt,"candidate_spin_moment":rq,"joint_KL":dkl(rt,rq),"variance_U":nu["variance"],"active_mean_drift_same":same["mean"],"CF_positive_roundoff_count":nu["roundoff_count"],"CF_roundoff_max":nu["roundoff_max"]}

def reconstruct(cfg,table):
    cases=[(1.,th,n,r) for th in (0.,.15) for n in (0,14,4094) for r in (0.,.15,.3,.75)]+[(.05,0.,n,.75) for n in (0,1,2)]
    out=[];comparisons=[]
    for a,th,n,rho in cases:
        for name,order in (("primary",96),("high",128)):
            v=evaluate(a,th,n,rho,order);original=table[(name,a,th,n,rho)]
            scale=math.sqrt(max(1,n)) if a==1 else 1.
            ck=[]
            for field in ("delta_F","path_gradient","Price_gradient"):
                limit=1e-8 if a==1 else (5e-14 if field=="delta_F" else 5e-13)
                ck.append(gate(field,scale*abs(v[field]-original[field]),limit,a=a,theta=th,n=n,rho=rho,order=order))
            pe=max(abs(v[f]-original[f]) for f in ("teacher_spin_moment","candidate_spin_moment"))
            ck.append(gate("pair",pe,1e-12,a=a,theta=th,n=n,rho=rho,order=order))
            ke=abs(v["joint_KL"]-original["joint_KL"])
            ck.append(gate("KL",ke,1e-10 if a==1 else 1e-15,a=a,theta=th,n=n,rho=rho,order=order))
            ck.append(gate("variance_U",abs(v["variance_U"]-original["variance_U"]),1e-12,a=a,order=order))
            comparisons.extend(ck);out.append({**v,"comparisons":ck,"KL_relative_error":ke/v["joint_KL"] if v["joint_KL"]>0 else (0. if ke==0 else None)})
        print(json.dumps({"rebuilt":{"a":a,"theta":th,"n":n,"rho":rho},"failures":len(FAIL)}),flush=True)
    return out,comparisons

def old_control(rebuilt):
    old=rows(OLD/"GRID.json");lookup={(r["variant"],r["a"],r["rho"]):r for r in old}
    checks=[]
    for r in rebuilt:
        if r["n"]:continue
        a,rho=r["a"],r["rho"]
        for variant in ("low","primary","high"):
            plus=lookup[(variant,a,rho)];minus=lookup[(variant,a,-rho)];j=plus["K"].index(2)
            if r["theta"]==.15:delta=plus["score_minus_rho0"][j];path=plus["path_gradient"][j];price=plus["price_gradient"][j];identity="same_theta"
            else:delta=(plus["score_minus_rho0"][j]+minus["score_minus_rho0"][j])/2;path=(plus["path_gradient"][j]-minus["path_gradient"][j])/2;price=(plus["price_gradient"][j]-minus["price_gradient"][j])/2;identity="null_via_rho_sign_average"
            for field,value in (("delta_F",delta),("path_gradient",path),("Price_gradient",price)):
                limit=1e-8 if a==1 else (5e-14 if field=="delta_F" else 5e-13)
                checks.append(gate("old_weak_"+field,abs(r[field]-value),limit,a=a,theta=r["theta"],rho=rho,order=r["order"],old_variant=variant,identity=identity))
    return checks

def write():
    need(not (OUT/"START.json").exists(),"existing audit: preserve original attempt")
    public,optional=sources();cfg=read(PROD/"CONFIG.json")
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/"START.json",{"utc":utc(),"source":rel(__file__),"source_sha256":sha(__file__),"plan_prefix_sha256":phash(),"public_inputs":public,"optional_workspace_inputs":optional,"before_audit_integrals":True,"orders":[96,128],"cases":27,"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"threads":1}})
    start=time.perf_counter();outputs=[]
    try:
        grid=rows(PROD/"GRID.json");ev=rows(PROD/"EVALUATIONS.json");fd=rows(PROD/"FD.json");checks=rows(PROD/"CHECKS.json");result=read(PROD/"RESULTS.json")
        ar,table=arithmetic(cfg,grid,ev,fd,checks,result,read(PROD/"CACHE_INDEX.json"))
        save(OUT/"ARITHMETIC.json",ar);outputs.append(OUT/"ARITHMETIC.json")
        values,comparison=reconstruct(cfg,table)
        save(OUT/"REBUILT.json",{"rows":values});outputs.append(OUT/"REBUILT.json")
        old=old_control(values);save(OUT/"OLD_WEAK_CONTROL.json",{"checks":old});outputs.append(OUT/"OLD_WEAK_CONTROL.json")
        summary={"saved_grid_rows":len(grid),"saved_evaluation_rows":len(ev),"checks_recomputed":len(ar["checks"]),"FD_recomputed":len(fd),"readouts_recomputed":10,"independent_cases":27,"independent_point_orders":len(values),"comparison_gates":len(comparison),"old_weak_controls":len(old)}
        save(OUT/"RECEIPT.json",{"utc":utc(),"state":"complete","passed":not FAIL,"start_sha256":sha(OUT/"START.json"),"files":{rel(p):sha(p) for p in outputs},"summary":summary,"counts":COUNT,"failures":FAIL,"elapsed_seconds":time.perf_counter()-start,"scope":{"all_1197_integrals_rebuilt":False,"continuous_global_certificate":False,"strict_integration_certificate":False,"original_sources_unchanged":True}})
        print(json.dumps({"passed":not FAIL,"summary":summary,"counts":COUNT,"failures":len(FAIL)},default=serial),flush=True)
    except Exception as exc:
        save(OUT/"FAILURE.json",{"utc":utc(),"error":repr(exc),"traceback":traceback.format_exc(),"counts":COUNT,"numerical_failures":FAIL})
        raise

def check(workspace=False):
    s=read(OUT/"START.json");r=read(OUT/"RECEIPT.json")
    need(r["state"]=="complete" and r["start_sha256"]==sha(OUT/"START.json"),"receipt chain")
    need(s["source_sha256"]==sha(__file__) and s["plan_prefix_sha256"]==phash(),"source/plan changed")
    verify(s["public_inputs"]);verify(r["files"])
    if workspace:verify(s["optional_workspace_inputs"])
    need(r["passed"] and not r["failures"],"retained audit failure")
    print(json.dumps({"passed":True,"workspace":workspace,"integrals_reexecuted":False,"receipt_sha256":sha(OUT/"RECEIPT.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("write","check"));p.add_argument("--workspace",action="store_true");a=p.parse_args()
    if a.action=="write":write()
    else:check(a.workspace)
