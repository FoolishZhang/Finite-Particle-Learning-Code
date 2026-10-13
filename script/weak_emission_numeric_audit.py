"""Independent saved-arithmetic, GH80 moment and direct K2 tensor audit.

No producer numerical functions are imported. ``write`` is guarded by a completed
and sealed NUMERIC package; ``check`` never evaluates scientific integrals.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
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

ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/"artifacts/weak_emission/NUMERIC"
REPLAY=ROOT/"artifacts/weak_emission/REPLAY"
OUT=ROOT/"artifacts/weak_emission/INDEPENDENT"
DOC=ROOT/"docs/08_research_研究工作/弱发射展开与总体偏好_20261001/NUMERIC_AUDIT.md"
MARKER="<!-- WEAK_NUMERIC_AUDIT_PLAN_END -->"
COUNTS={"new_scientific_samples":0,"RNG_calls":0,"fits":0,"locator_executions":0,
        "pair_integrals":0,"pair_Gaussian_nodes":0,"direct_K2_native_integrals":0,
        "direct_K2_particle_pairs":0,"direct_single_particle_nodes":0,
        "GH_rules":0,"Decimal_KL_evaluations":0}
FAILURES=[]

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=serial,allow_nan=False);f.write("\n")
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def close(x,y,tol=1e-12):return abs(float(x)-float(y))<=tol
def gate(name,error,limit,**meta):
    passed=bool(np.isfinite(error) and error<=limit)
    row={"name":name,"error":float(error),"limit":float(limit),"passed":passed,**meta}
    if not passed:FAILURES.append(row)
    return row
def plan_hash():
    text=DOC.read_text(encoding="utf-8").replace("\r\n","\n")
    return hashlib.sha256((text.split(MARKER)[0]+MARKER+"\n").encode()).hexdigest()
def verify(mapping):
    for p,h in mapping.items():require(sha(ROOT/p)==h,"hash mismatch: "+p)

def source_inputs():
    require((PROD/"STATE.json").exists(),"NUMERIC not started")
    require(read(PROD/"STATE.json")["state"]=="complete","NUMERIC must be complete before audit integration")
    require((PROD/"RESULTS_FREEZE.json").exists(),"NUMERIC result seal missing")
    freeze=read(PROD/"FREEZE.json");seal=read(PROD/"RESULTS_FREEZE.json")
    mapping={rel(PROD/name):h for name,h in seal["files"].items()}
    for item in freeze["inputs"]:mapping[item["snapshot"]]=item["sha256"]
    mapping[rel(PROD/"CONFIG.json")]=freeze["config_sha256"]
    mapping[rel(PROD/"UNIT_CHECK.json")]=freeze["unit_check_sha256"]
    mapping[rel(PROD/"RESULTS_FREEZE.json")]=sha(PROD/"RESULTS_FREEZE.json")
    require((REPLAY/"COMPLETE.json").exists(),"saved-evaluation REPLAY incomplete")
    replay=read(REPLAY/"COMPLETE.json");start=read(REPLAY/"START.json")
    require(replay["all_exactly_equal_except_elapsed"],"REPLAY comparison failed")
    require(start["original_freeze_sha256"]==sha(PROD/"FREEZE.json"),"REPLAY original freeze")
    require(start["original_results_freeze_sha256"]==sha(PROD/"RESULTS_FREEZE.json"),"REPLAY original result seal")
    comparison=read(REPLAY/"COMPARISON.json")
    require(comparison["all_exactly_equal_except_elapsed"] and all(v["exactly_equal"] and not v["differences"] for v in comparison["files"].values()),"REPLAY source arithmetic mismatch")
    mapping.update({rel(REPLAY/name):h for name,h in replay["files"].items()})
    mapping[rel(REPLAY/"COMPLETE.json")]=sha(REPLAY/"COMPLETE.json")
    mapping[start["replay_source"]]=start["replay_source_sha256"]
    mapping[start["producer_source"]]=start["producer_sha256"]
    verify(mapping)
    return mapping

@lru_cache(None)
def gh(n):
    z,eigenvectors=eigh_tridiagonal(np.zeros(n),np.sqrt(np.arange(1,n)),lapack_driver="stemr")
    weight=eigenvectors[0]**2
    require(abs(math.fsum(weight)-1)<1e-12,"Jacobi GH mass")
    weight/=math.fsum(weight)
    COUNTS["GH_rules"]+=1
    return z,weight

@lru_cache(None)
def pair_moment(a,rho):
    z,w=gh(80);x=z[:,None];y=rho*x+math.sqrt(1-rho*rho)*z[None,:]
    def spin(t):
        ab=np.abs(t);return np.sign(t)*(-np.expm1(-ab))/(1+np.exp(-ab))
    values=spin(a*x)*spin(a*y)
    answer=float(np.sum(values.astype(np.longdouble)*np.multiply.outer(w,w).astype(np.longdouble)))
    COUNTS["pair_integrals"]+=1;COUNTS["pair_Gaussian_nodes"]+=80**2
    return answer

def decimal_kl(rp,rq):
    with localcontext() as ctx:
        ctx.prec=80
        p=(Decimal(1)+Decimal.from_float(float(rp)))/2
        q=(Decimal(1)+Decimal.from_float(float(rq)))/2
        answer=p*(p/q).ln()+(1-p)*((1-p)/(1-q)).ln()
    COUNTS["Decimal_KL_evaluations"]+=1
    value=float(answer)
    require(value>=0,"negative independent Decimal KL")
    return value

@lru_cache(None)
def direct_native(a,rho,n):
    """Two iid particles, each represented by an independent 2D GH rule."""
    z,w=gh(n);z1=np.repeat(z,n);z2=np.tile(z,n);weights=np.multiply.outer(w,w).ravel()
    root=math.sqrt(1-rho*rho);x2=rho*z1+root*z2
    logw=-np.logaddexp(0,-a*z1)-np.logaddexp(0,-a*x2)
    likelihood=np.exp(logw)
    dlogw=a*expit(-a*x2)*(z1-rho*z2/root)
    total_value=np.longdouble(0);total_gradient=np.longdouble(0)
    for first in range(0,len(weights),64):
        sl=slice(first,first+64)
        denominator=likelihood[sl,None]+likelihood[None,:]
        loss=math.log(2)-np.logaddexp(logw[sl,None],logw[None,:])
        grad=-(likelihood[sl,None]*dlogw[sl,None]+likelihood[None,:]*dlogw[None,:])/denominator
        mass=weights[sl,None]*weights[None,:]
        total_value+=np.sum(loss.astype(np.longdouble)*mass.astype(np.longdouble))
        total_gradient+=np.sum(grad.astype(np.longdouble)*mass.astype(np.longdouble))
    COUNTS["direct_K2_native_integrals"]+=1
    COUNTS["direct_K2_particle_pairs"]+=n**4
    COUNTS["direct_single_particle_nodes"]+=n*n
    return float(total_value),float(total_gradient)

def direct_score(a,rho,n,theta):
    p=(1+pair_moment(a,theta))/2
    fs,gs=direct_native(a,rho,n);fd,gd=direct_native(a,-rho,n);f0,_=direct_native(a,0.,n)
    value=p*fs+(1-p)*fd
    return {"a":a,"rho":rho,"order":n,"value":value,
            "value_minus_rho0":p*(fs-f0)+(1-p)*(fd-f0),
            "path_gradient":p*gs-(1-p)*gd,"teacher_p_same":p,
            "same_value":fs,"different_value":fd,"same_gradient":gs,"different_gradient":gd,"rho0_value":f0}

def load_rows(path):
    obj=read(path)
    return obj["rows"] if isinstance(obj,dict) else obj

class Tables:
    def __init__(self,grid,evaluations):
        self.rows={}
        for row in grid+evaluations:
            key=(float(row["a"]).hex(),float(row["rho"]).hex(),row["variant"])
            if key in self.rows:require(self.rows[key]==row,"inconsistent duplicate evaluation")
            self.rows[key]=row
    def get(self,a,rho,variant):
        key=(float(a).hex(),float(rho).hex(),variant)
        if key in self.rows:return self.rows[key]
        matches=[r for r in self.rows.values() if r["variant"]==variant and r["a"]==a and abs(r["rho"]-rho)<1e-14]
        require(len(matches)==1,"missing/ambiguous saved evaluation")
        return matches[0]

def arithmetic(cfg,table,grid,locs,checks,results,correction):
    ks=cfg["ks"];variants=list(cfg["variants"]);gates=cfg["gates"]
    require(len(grid)==315 and len(locs)==60,"grid/locator count")
    require({(r["a"],r["K"],r["variant"]) for r in locs}==set(itertools.product(cfg["emissions"],ks[1:],variants)),"locator configuration coverage")
    primary=[r for r in locs if r["variant"]=="primary"]
    require(primary==results["primary_locations"] and len(primary)==20,"primary locations")
    require(len(correction)==15,"correction count")
    locations=[]
    for row in locs:
        a,k,v=row["a"],row["K"],row["variant"];idx=ks.index(k)
        gg=[table.get(a,rho,v)["path_gradient"][idx] for rho in cfg["rho_grid"]]
        require(np.allclose(gg,row["grid_gradients"],rtol=0,atol=1e-15),"locator grid gradients")
        sign_intervals=[(l,r) for l,r,gl,gr in zip(cfg["rho_grid"][:-1],cfg["rho_grid"][1:],gg[:-1],gg[1:]) if gl*gr<0]
        require(sign_intervals==[(b["left"],b["right"]) for b in row["root_brackets"]],"root sign intervals")
        for bracket in row["root_brackets"]:
            require(bracket["left"]<=bracket["root"]<=bracket["right"],"root outside bracket")
        expected_candidates=sorted(set([cfg["rho_box"][0],cfg["rho_box"][1],*row["roots"]]))
        require(row["candidate_rhos"]==expected_candidates,"candidate set")
        values=[table.get(a,rho,v)["score_minus_rho0"][idx] for rho in expected_candidates]
        require(np.allclose(values,row["candidate_values"],rtol=0,atol=1e-15),"candidate scores")
        chosen=min(range(len(values)),key=lambda j:(values[j],expected_candidates[j]))
        require(row["rho"]==expected_candidates[chosen],"selected candidate")
        selected=table.get(a,row["rho"],v)
        require(close(row["score"],selected["score"][idx],1e-15),"selected score")
        require(close(row["gradient"],selected["path_gradient"][idx],1e-15),"selected gradient")
        require(close(row["joint_KL"],selected["joint_KL"],1e-15),"selected KL")
        vals=[table.get(a,rho,v)["score_minus_rho0"][idx] for rho in cfg["rho_grid"]]
        best=min(range(len(vals)),key=lambda j:(vals[j],cfg["rho_grid"][j]))
        require(row["grid_minimum_rho"]==cfg["rho_grid"][best],"grid minimum")
        require(close(row["candidate_minus_grid_minimum"],values[chosen]-vals[best],1e-15),"candidate vs grid")
        near=[rho for rho,value in zip(expected_candidates,values) if value<=values[chosen]+row["near_minimum_value_tolerance"]]
        require(near==row["near_minimum_candidate_rhos"] and row["unique_numerical_candidate"]==(len(near)==1),"near tie set")
        locations.append({"a":a,"K":k,"variant":v,"rho":row["rho"],"passed":True,"global_optimum_recertified":False})
    recomputed=[]
    for st in checks:
        name=st["name"];a=st["a"];rho=st.get("rho");k=st.get("K");idx=ks.index(k) if k is not None else None
        if name.startswith("variant_") and name!="variant_location":
            key=name[len("variant_"):];rr=[table.get(a,rho,v) for v in variants]
            error=float(np.ptp([r[key][idx] if idx is not None else r[key] for r in rr]))
        elif name=="Price_path":
            r=table.get(a,rho,st["variant"]);error=abs(r["numeric_path_gradient"][idx]-r["numeric_price_gradient"][idx])
        elif name=="K1_flat_integral":
            r=table.get(a,rho,st["variant"]);error=abs(r["numeric_score"][0]-r["K1_analytic"])
        elif name=="K1_numeric_gradient":error=abs(table.get(a,rho,st["variant"])["numeric_path_gradient"][0])
        elif name=="variant_location":error=float(np.ptp([r["rho"] for r in locs if r["a"]==a and r["K"]==k]))
        elif name=="located_candidate_vs_grid":
            r=next(x for x in locs if (x["a"],x["K"],x["variant"])==(a,k,st["variant"]))
            error=max(0.,r["candidate_minus_grid_minimum"])
        elif name=="primary_FD_path":
            fd=st["finite_difference"];values=[table.get(a,x,"primary")["score_minus_rho0"][idx] for x in fd["rhos"]]
            if fd["scheme"]=="central":coef=[1,-8,8,-1];denom=12*fd["h"]
            else:coef=[-25,48,-36,16,-3];denom=12*(fd["rhos"][1]-fd["rhos"][0])
            estimate=float(sum(np.longdouble(c)*np.longdouble(v) for c,v in zip(coef,values))/np.longdouble(denom))
            require(close(estimate,fd["estimate"],2e-13),"saved FD arithmetic")
            error=abs(estimate-table.get(a,rho,"primary")["path_gradient"][idx])
        elif name in ("corrected_pair_moment","corrected_joint_KL"):
            r=next(x for x in correction if x["a"]==a and x["K"]==k)
            if name=="corrected_pair_moment":error=max(float(np.ptp([v[key] for v in r["variants"]])) for key in ("teacher_spin_moment","candidate_spin_moment"))
            else:error=float(np.ptp([v["joint_KL"] for v in r["variants"]]))
        else:raise AssertionError("unhandled saved gate: "+name)
        require(close(error,st["error"],2e-13),"gate error reproduction: "+name)
        if name=="variant_score_minus_rho0" or name=="located_candidate_vs_grid":limit=max(gates["value_floor"],a**4*gates["value_scaled"])
        elif name in ("variant_path_gradient","Price_path","primary_FD_path"):limit=max(gates["gradient_floor"],a**4*gates["gradient_scaled"])
        elif name in ("variant_joint_KL","corrected_joint_KL"):limit=a**4*gates["risk_scaled"]
        elif name=="variant_location":limit=gates["position"]
        elif name=="K1_flat_integral":limit=gates["K1_value"]
        elif name=="K1_numeric_gradient":limit=gates["K1_gradient"]
        elif name=="corrected_pair_moment":limit=cfg["correction"]["moment_tolerance"]
        else:raise AssertionError("unhandled gate limit")
        require(st["limit"]==limit,"frozen gate limit mismatch")
        require(st["passed"]==(st["error"]<=st["limit"]),"saved gate decision")
        # The frozen decision is reproduced from the saved statistic; rounding in
        # independent FD arithmetic is recorded, not used to revise original gates.
        recomputed.append({"name":name,"a":a,"K":k,"rho":rho,"recomputed_error":error,"saved_error":st["error"],"difference":abs(error-st["error"]),"original_passed":st["passed"]})
    require(results["numerical_qualified"]==all(x["passed"] for x in checks),"overall numerical qualification")
    predictions=[]
    for st in results["prediction_gates"]:
        loc=next(x for x in primary if x["a"]==st["a"] and x["K"]==st["K"])
        a,k=loc["a"],loc["K"];lead=.75 if k==2 else k*cfg["theta"]/(k-2)
        coefficient=(lead-cfg["theta"])**2/32
        relative=abs(loc["joint_KL"]/a**4-coefficient)/coefficient
        position=(loc["rho"]==cfg["rho_box"][1] and all(g<0 for g in loc["grid_gradients"])) if k==2 else abs(loc["rho"]-lead)<=gates["prediction_position"]
        risk=relative<=gates["prediction_risk_relative"]
        require(close(st["leading_rho"],lead,1e-15) and close(st["leading_KL_over_a4"],coefficient,1e-15),"prediction analytic coefficients")
        require(position==st["location_or_pointwise_gradient_pass"] and risk==st["risk_prediction_pass"],"prediction gate")
        require(close(relative,st["KL_relative_difference"],1e-12),"prediction relative difference")
        predictions.append({"K":k,"location_pass":position,"risk_pass":risk,"relative_difference":relative})
    require(sum(int(x["location_pass"])+int(x["risk_pass"]) for x in predictions)==results["prediction_tests_passed"],"8 prediction count")
    for r in correction:
        require(r["rho_corrected"]==(r["K"]-2)*r["rho_raw"]/r["K"],"correction formula")
        require(r["feasible"] and cfg["rho_box"][0]<=r["rho_corrected"]<=cfg["rho_box"][1],"correction feasibility")
        primary_v=next(v for v in r["variants"] if v["variant"]=="primary")
        require(r["primary_joint_KL"]==primary_v["joint_KL"],"correction primary risk")
        require(close(r["primary_joint_KL_over_a8"],r["primary_joint_KL"]/r["a"]**8,1e-12),"correction scaled risk")
        require(close(r["corrected_over_raw_KL"],r["primary_joint_KL"]/r["raw_joint_KL"],1e-12),"correction risk ratio")
    tails=[]
    for a in cfg["emissions"]:
        m2=sum(math.comb(2,i)*math.comb(2,j)*math.exp(a*a*(i*i+j*j+2*.75*i*j)/2) for i in range(3) for j in range(3))
        saved=results["tail_bounds"][str(a)]
        require(close(m2,saved["negative_moment_bound"],1e-12*max(1,m2)),"negative moment bound")
        for row in saved["rows"]:
            k=row["K"];eps=k*math.exp(cfg["log_u_bounds"][0]);upper=k*math.exp(cfg["log_u_bounds"][1])
            expected={"t_lower":eps,"t_upper":upper,"value_lower":eps,"value_upper":m2/upper**2,"derivative_lower":a*a*(eps+eps*eps/(2*k))/16,"derivative_upper":a*a*(27*math.exp(-3)+8*math.exp(-2))*m2/upper**2}
            require(all(close(row[key],v,1e-14*max(abs(v),1e-100)) for key,v in expected.items()),"analytic tail arithmetic")
        tails.append({"a":a,"negative_moment_bound":m2,"passed":True})
    ledger=results["ledger"]
    for key,value in (("primary_location_programs",20),("all_variant_location_executions",60),("grid_rows",315),("grid_scores_including_K1",1575),("corrected_primary_points",15),("corrected_pair_variant_requests",45),("new_labels",0),("random_draws",0),("new_fits",0)):
        require(ledger[key]==value,"production ledger: "+key)
    return {"checks":recomputed,"locations":locations,"predictions":predictions,"analytic_tails":tails,"all_saved_arithmetic_passed":True}

def moments(cfg,table,correction):
    requests=[]
    for r in table.rows.values():
        requests.append({"source":"evaluate","a":r["a"],"rho":r["rho"],"variant":r["variant"],"teacher":r["teacher_spin_moment"],"candidate":r["candidate_spin_moment"],"risk":r["joint_KL"]})
    for r in correction:
        for v in r["variants"]:requests.append({"source":"corrected","a":r["a"],"rho":r["rho_corrected"],"K":r["K"],"variant":v["variant"],"teacher":v["teacher_spin_moment"],"candidate":v["candidate_spin_moment"],"risk":v["joint_KL"]})
    rows=[]
    for r in requests:
        rt=pair_moment(r["a"],cfg["theta"]);rq=pair_moment(r["a"],r["rho"]);risk=decimal_kl(rt,rq)
        pe=max(abs(rt-r["teacher"]),abs(rq-r["candidate"]));ke=abs(risk-r["risk"])
        ck1=gate("independent_pair",pe,1e-12,a=r["a"],rho=r["rho"],variant=r["variant"],source=r["source"])
        ck2=gate("independent_forward_KL",ke,r["a"]**4*cfg["gates"]["risk_scaled"],a=r["a"],rho=r["rho"],variant=r["variant"],source=r["source"])
        rows.append({**r,"independent_teacher":rt,"independent_candidate":rq,"independent_KL":risk,"pair_absolute_error":pe,"KL_absolute_error":ke,"KL_relative_error":ke/risk if risk>0 else (0. if ke==0 else None),"passed":ck1["passed"] and ck2["passed"]})
    return rows

def direct_checks(cfg,table):
    rows=[];convergence=[]
    for a in (.05,.2,1.):
        for rho in (-.75,.15,.75):
            values=[direct_score(a,rho,n,cfg["theta"]) for n in (40,64)]
            limv=max(5e-13,a**4*1e-7);limg=max(1e-13,a**4*1e-6)
            for key in ("value","value_minus_rho0","path_gradient"):
                convergence.append(gate("direct_order_"+key,abs(values[0][key]-values[1][key]),limg if key=="path_gradient" else limv,a=a,rho=rho))
            for value in values:
                comparisons=[]
                for v in cfg["variants"]:
                    original=table.get(a,rho,v);index=original["K"].index(2)
                    for key,source in (("value","score"),("value_minus_rho0","score_minus_rho0"),("path_gradient","path_gradient")):
                        comparisons.append(gate("direct_vs_Laplace_"+key,abs(value[key]-original[source][index]),limg if key=="path_gradient" else limv,a=a,rho=rho,order=value["order"],variant=v))
                rows.append({**value,"comparisons":comparisons,"passed":all(c["passed"] for c in comparisons)})
            print(json.dumps({"K2_point_completed":{"a":a,"rho":rho},"audit_failures":len(FAILURES)}),flush=True)
    return {"rows":rows,"convergence":convergence}

def write():
    require(not (OUT/"START.json").exists(),"Audit attempt already exists; preserve it")
    sources=source_inputs()
    cfg=read(PROD/"CONFIG.json");grid=load_rows(PROD/"GRID.json");locs=load_rows(PROD/"LOCATIONS.json")
    checks=load_rows(PROD/"CHECKS.json");results=read(PROD/"RESULTS.json");correction=load_rows(PROD/"CORRECTION.json")
    require((REPLAY/"EVALUATIONS.json").exists(),"full saved evaluations required for independent error-statistic replay")
    evaluations=load_rows(REPLAY/"EVALUATIONS.json")
    require(rel(REPLAY/"EVALUATIONS.json") in sources,"evaluation table not sealed")
    table=Tables(grid,evaluations)
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/"START.json",{"utc":utc(),"source_sha256":sha(__file__),"source":rel(__file__),"plan_prefix_sha256":plan_hash(),"inputs":sources,"orders":{"moments":80,"direct_K2":[40,64]},"block_size":64,"before_audit_integrals":True,"environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"threads":1}})
    started=time.perf_counter();outputs=[]
    try:
        ar=arithmetic(cfg,table,grid,locs,checks,results,correction)
        save(OUT/"ARITHMETIC.json",ar);outputs.append(OUT/"ARITHMETIC.json")
        mo=moments(cfg,table,correction)
        save(OUT/"MOMENTS.json",{"rows":mo});outputs.append(OUT/"MOMENTS.json")
        dr=direct_checks(cfg,table)
        save(OUT/"DIRECT_K2.json",dr);outputs.append(OUT/"DIRECT_K2.json")
        summary={"saved_evaluation_rows":len(table.rows),"saved_checks_recomputed":len(ar["checks"]),"location_executions_checked":len(ar["locations"]),"primary_locations_checked":len(results["primary_locations"]),"prediction_readings_checked":8,"moment_comparisons":len(mo),"moment_comparisons_passed":sum(r["passed"] for r in mo),"direct_K2_points":9,"direct_K2_point_orders":18,"direct_comparisons":sum(len(r["comparisons"]) for r in dr["rows"]),"direct_order_checks":len(dr["convergence"])}
        save(OUT/"RECEIPT.json",{"utc":utc(),"state":"complete","passed":not FAILURES,"start_sha256":sha(OUT/"START.json"),"files":{rel(p):sha(p) for p in outputs},"summary":summary,"counts":COUNTS,"failures":FAILURES,"elapsed_seconds":time.perf_counter()-started,"scope":{"K_ge3_direct_tensor_recomputed":False,"locator_rerun":False,"strict_quadrature_certificate":False,"all_original_failures_preserved":True}})
        print(json.dumps({"passed":not FAILURES,"summary":summary,"counts":COUNTS,"failures":len(FAILURES)},default=serial),flush=True)
    except Exception as exc:
        save(OUT/"FAILURE.json",{"utc":utc(),"error":repr(exc),"traceback":traceback.format_exc(),"counts":COUNTS,"numerical_failures":FAILURES})
        raise

def check():
    start=read(OUT/"START.json");r=read(OUT/"RECEIPT.json")
    require(r["state"]=="complete","audit incomplete")
    require(start["source_sha256"]==sha(__file__),"auditor source changed")
    require(start["plan_prefix_sha256"]==plan_hash(),"audit plan changed")
    require(r["start_sha256"]==sha(OUT/"START.json"),"audit start changed")
    verify(start["inputs"]);verify(r["files"])
    require(r["passed"] and not r["failures"],"audit has retained failures")
    print(json.dumps({"passed":True,"read_only":True,"integrals_reexecuted":False,"receipt_sha256":sha(OUT/"RECEIPT.json")}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("write","check"));args=parser.parse_args()
    if args.action=="write":write()
    else:check()
