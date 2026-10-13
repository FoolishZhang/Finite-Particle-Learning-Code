"""T: exhaustive d6 laminar-RC population capacity and deterministic identities."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="2"
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import argparse, hashlib, itertools, json, math, platform, sys, time, traceback
import numpy as np
import scipy
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/v3_select/T"
DOC=ROOT/"docs/08_research_研究工作/v3结构选择与查询诊断_20260930"
CORE_SHA="4f519607497cb4719a5096d5762dea96ee45337e722a28e8203ad7b6571d976d"
CFG=dict(d=6,r=3,epsilon=.9,trees=945,query_k=[1,5],diagnostic_d=8,diagnostic_r=4,
         diagnostic_epsilon=.9,projection_epsilon=.4,gap_gate=1e-8,
         mass_gate=1e-12,identity_gate=1e-11,analytic_gate=1e-12,
         scientific_observations=0,population_fit_calls=945,extra_optimizer_calls=0,
         root_required=True,full_binary_labelled_trees=True)
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def write(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w" if replace else "x",encoding="utf-8",newline="\n") as f:
        json.dump(serial(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def verify(ok,msg):
    if not ok:raise RuntimeError(msg)
def states(d):return ((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(int)
@lru_cache(None)
def trees(leaves):
    leaves=tuple(leaves)
    if len(leaves)==1:return ((),)
    first=leaves[0];rest=leaves[1:];out=[]
    for mask in range(1<<len(rest)):
        left=(first,)+tuple(v for i,v in enumerate(rest) if mask&(1<<i))
        if len(left)==len(leaves):continue
        right=tuple(v for v in leaves if v not in left)
        for tl in trees(left):
            for tr in trees(right):
                out.append(tuple(sorted((leaves,)+tl+tr,key=lambda n:(len(n),n))))
    verify(len(out)==len(set(out)),"duplicate rooted tree")
    return tuple(sorted(out))
def registry():
    alltrees=trees(tuple(range(CFG["d"])))
    verify(len(alltrees)==945,"tree enumeration count")
    verify(all(len(t)==5 and tuple(range(6)) in t for t in alltrees),"full binary internal-node count")
    return [dict(tree_id=i,nodes=[list(n) for n in t],node_masks=[sum(1<<j for j in n) for n in t]) for i,t in enumerate(alltrees)]
def freeze():
    verify(not OUT.exists(),"T output already exists")
    verify(sha(ROOT/"script/v3_select_core.py")==CORE_SHA,"new core differs from reviewed version")
    OUT.mkdir(parents=True)
    (OUT/"HANDOFF_INPUT.md").write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"CONFIG.json",CFG);write(OUT/"TREE_REGISTRY.json",registry())
    write(OUT/"ENVIRONMENT.json",dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    paths=[Path(__file__).resolve(),ROOT/"script/v3_select_core.py",ROOT/"script/v3_query_core.py",
           DOC/"PLAN_T.md",DOC/"THEORY_DESIGN.md",DOC/"NUMERICS.md",
           OUT/"CONFIG.json",OUT/"TREE_REGISTRY.json",OUT/"ENVIRONMENT.json",OUT/"HANDOFF_INPUT.md"]
    write(OUT/"FREEZE.json",dict(time_utc=now(),baseline_commit="391714d0b75bf3b65170a22d4c9c763e69255eb8",
          inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},scientific_population_fits_started=False))
    write(OUT/"STATE.json",dict(status="frozen",time_utc=now(),active_jobs=[]))
    print(json.dumps(dict(freeze=sha(OUT/"FREEZE.json"),trees=len(registry()))),flush=True)
def checkfreeze():
    verify(read(OUT/"CONFIG.json")==CFG,"configuration changed")
    for p,h in read(OUT/"FREEZE.json")["inputs"].items():verify(sha(ROOT/p)==h,p)
def blind_teacher(d,epsilon):
    c=states(d);h=np.prod(c[:,::2]-c[:,1::2],axis=1)
    raw=(1+epsilon*h)/(1<<d);mass=float(raw.sum())
    verify(np.all(raw>0) and abs(mass-1)<CFG["mass_gate"],"blind teacher positivity/mass")
    return raw/mass,c,h
def kappa(t):return .5*((1+t)*math.log1p(t)+(1-t)*math.log1p(-t))
def KL(p,q):return float(p@(np.log(p)-np.log(q)))
def fixed_query(p,q,observed,target):
    d=int(round(math.log2(len(p))));c=states(d)
    codes=c[:,observed]@(1<<np.arange(len(observed))) if observed else np.zeros(len(c),dtype=int)
    lab=2*codes+c[:,target];size=1<<len(observed)
    pj=np.bincount(lab,weights=p,minlength=2*size).reshape(-1,2)
    qj=np.bincount(lab,weights=q,minlength=2*size).reshape(-1,2)
    pe=pj.sum(1);qe=qj.sum(1);pc=pj/pe[:,None];qc=qj/qe[:,None]
    return dict(conditional_excess_KL=float(np.sum(pj*(np.log(pc)-np.log(qc)))),
                excess_Brier=float(pe@((pc[:,1]-qc[:,1])**2)),
                evidence_probability=pe,target_given_evidence=pc[:,1],prediction_given_evidence=qc[:,1])
def query_metrics(p,q,k):
    d=int(round(math.log2(len(p))));values=[]
    for observed in itertools.combinations(range(d),k):
        for target in range(d):
            if target not in observed:values.append(fixed_query(p,q,observed,target))
    return dict(conditional_excess_KL=math.fsum(v["conditional_excess_KL"] for v in values)/len(values),
                excess_Brier=math.fsum(v["excess_Brier"] for v in values)/len(values),
                query_configurations=len(values),scalar_Brier=True,complete_count_supplied=False)
def analytic_blind(d,epsilon):
    p,c,h=blind_teacher(d,epsilon);u=np.full(1<<d,1/(1<<d));r=d//2;n=c.sum(1);spin=2*c-1
    count_error=float(abs(np.bincount(n,weights=p)-np.bincount(n,weights=u)).max())
    pair_error=float(abs(spin.T@((p-u)[:,None]*spin)).max())
    conditional_error=0.
    for count in range(d+1):
        keep=n==count;pp=p[keep]/p[keep].sum();uu=u[keep]/u[keep].sum();ys=spin[keep]
        conditional_error=max(conditional_error,float(abs(ys.T@((pp-uu)[:,None]*ys)).max()))
    low_error=0.
    for order in range(1,r):
        for subset in itertools.combinations(range(d),order):
            code=c[:,subset]@(1<<np.arange(order))
            marginal=np.bincount(code,weights=p,minlength=1<<order)
            low_error=max(low_error,float(abs(marginal-1/(1<<order)).max()))
    joint=KL(p,u);loo=query_metrics(p,u,d-1);low=query_metrics(p,u,1)
    f_joint=2**(-r)*kappa(epsilon)
    f_loo=2**(-r)*(kappa(epsilon)-2*kappa(epsilon/2))
    f_brier=2**(-r)*epsilon**2/(2*(4-epsilon**2))
    formula_error=max(abs(joint-f_joint),abs(loo["conditional_excess_KL"]-f_loo),abs(loo["excess_Brier"]-f_brier))
    passed=max(count_error,pair_error,conditional_error,low_error,formula_error,abs(low["conditional_excess_KL"]),abs(low["excess_Brier"]))<CFG["analytic_gate"]
    return p,dict(d=d,r=r,epsilon=epsilon,full_support=True,count_distribution_error=count_error,
        unconditional_pair_residual_max=pair_error,conditional_given_N_pair_residual_max=conditional_error,
        all_marginals_order_less_than_r_max_error=low_error,joint_KL_uniform=joint,joint_KL_formula=f_joint,
        LOO=loo,LOO_log_formula=f_loo,LOO_Brier_formula=f_brier,k1=low,
        max_formula_error=formula_error,passed=bool(passed),
        conditional_N_pair_moments_not_claimed_zero=True)
def d3_projection():
    epsilon=CFG["projection_epsilon"];c=states(3);y=2*c-1;n=c.sum(1);u=np.full(8,1/8)
    p=(1+epsilon*(y[:,0]*y[:,1]-y[:,1]*y[:,2]))/8
    q=(1+epsilon*y[:,0]*y[:,1]-.5*epsilon*(y[:,0]*y[:,2]+y[:,1]*y[:,2]))/8
    j0=-.25*math.log1p(2*epsilon);j12=.25*math.log((1+2*epsilon)/(1-epsilon)**2)
    score=j12*y[:,0]*y[:,1]+j0*(y[:,0]*y[:,2]+y[:,1]*y[:,2])
    reconstructed=np.exp(score-logsumexp(score))
    moment_error=max(float(abs((p-q)@y).max()),float(abs(np.bincount(n,weights=p-q,minlength=4)).max()),
                     abs(float((p-q)@(y[:,0]*y[:,1]))))
    before=KL(p,u);after=KL(p,q);gain=KL(q,u);query_before=fixed_query(p,u,(0,),2);query_after=fixed_query(p,q,(0,),2)
    expected_log=-.5*math.log1p(-epsilon**2/4);expected_brier=epsilon**2/16
    error=max(float(abs(q-reconstructed).max()),moment_error,abs(before-after-gain),
              abs(query_before["conditional_excess_KL"]),abs(query_before["excess_Brier"]),
              abs(query_after["conditional_excess_KL"]-expected_log),abs(query_after["excess_Brier"]-expected_brier))
    result=dict(epsilon=epsilon,p=p,Q0=u,Qstar=q,joint_KL_before=before,joint_KL_after=after,joint_gain=gain,
        query_observed=[0],query_target=2,query_before=query_before,query_after=query_after,
        query_log_formula=expected_log,query_Brier_formula=expected_brier,
        Qstar_pair_coefficients=dict(J12=j12,J13=j0,J23=j0),
        original_and_added_moment_error=moment_error,max_identity_error=error,
        passed=bool(error<CFG["analytic_gate"] and after<before and query_after["conditional_excess_KL"]>0),
        population_solution_is_analytic_not_new_optimizer_call=True)
    return result
def run():
    checkfreeze();verify(read(OUT/"STATE.json")["status"]=="frozen","no silent scientific restart")
    import v3_select_core as core
    write(OUT/"START.json",dict(time_utc=now(),freeze_sha256=sha(OUT/"FREEZE.json")))
    p,diag6=analytic_blind(6,.9);p8,diag8=analytic_blind(8,.9);diag3=d3_projection()
    write(OUT/"ANALYTIC_CHECKS.json",dict(d6=diag6,d8=diag8,d3_projection=diag3))
    write(OUT/"TEACHERS.json",dict(d6_blind=p,d8_blind=p8))
    verify(diag6["passed"] and diag8["passed"] and diag3["passed"],"analytic identity gate")
    H=-float(p@np.log(p));records=[];pmfs=[]
    trees_saved=read(OUT/"TREE_REGISTRY.json");start=time.perf_counter()
    for spec in trees_saved:
        tree_id=spec["tree_id"]
        write(OUT/"STATE.json",dict(status="running_population_fits",tree_id=tree_id,time_utc=now(),pid=os.getpid()),True)
        began=time.perf_counter();result=core.fit_rc(p,spec["nodes"]);elapsed=time.perf_counter()-began
        q=np.asarray(result["q"]);valid=bool(np.isfinite(q).all() and np.all(q>0) and abs(float(q.sum())-1)<1e-11)
        gap=float(result["gap"]);objective=float(result["objective"])
        qualified=bool(valid and result["qualified"] and gap<=CFG["gap_gate"])
        joint=KL(p,q) if valid else None
        if valid:verify(abs(joint-(objective-H))<CFG["identity_gate"],"PMF/fit objective mismatch")
        record=dict(tree_id=tree_id,nodes=spec["nodes"],node_masks=spec["node_masks"],
            params=result["params"],parameter_count=result["parameter_count"],objective=objective,gap=gap,
            qualified=qualified,PMF_valid=valid,joint_KL=joint,
            joint_box_lower_raw=joint-gap if valid else None,
            queries={str(k):query_metrics(p,q,k) for k in CFG["query_k"]} if valid else None,
            optimizer_receipt=result["optimizer_receipt"],elapsed_seconds=elapsed,
            query_values_are_at_joint_fit_not_query_optima=True)
        with (OUT/"FIT_LOG.jsonl").open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(serial(record),ensure_ascii=False,allow_nan=False)+"\n")
        records.append(record);pmfs.append(q)
        if (tree_id+1)%50==0 or tree_id+1==945:
            print(json.dumps(dict(completed=tree_id+1,qualified=sum(x["qualified"] for x in records))),flush=True)
    write(OUT/"FITS.json",records);write(OUT/"PMFS.json",np.array(pmfs))
    np.savez_compressed(OUT/"PMFS.npz",q=np.array(pmfs),teacher=p)
    valid_records=[r for r in records if r["PMF_valid"]]
    upper_record=min(valid_records,key=lambda r:r["joint_KL"]) if valid_records else None
    all_valid=len(valid_records)==945;all_qualified=all(r["qualified"] for r in records)
    lower=min(r["joint_box_lower_raw"] for r in records) if all_valid else None
    summary=dict(population_fits=945,qualified=sum(r["qualified"] for r in records),
        failed_tree_ids=[r["tree_id"] for r in records if not r["qualified"]],
        best_joint_tree_id=upper_record["tree_id"] if upper_record else None,
        boxed_union_joint_KL_upper=upper_record["joint_KL"] if upper_record else None,
        boxed_union_joint_KL_lower_raw=lower,
        qualified_boxed_union_joint_KL_lower=max(0.,lower) if all_qualified else None,
        all_candidate_boxes_qualified=all_qualified,
        boxed_class_only_not_unbounded_numerical_lower_bound=True,
        minimum_query_among_joint_fit_endpoints={
            str(k):min(r["queries"][str(k)]["conditional_excess_KL"] for r in valid_records if r["qualified"])
            for k in CFG["query_k"]} if any(r["qualified"] for r in valid_records) else None,
        not_query_optimized=True,elapsed_fit_seconds=time.perf_counter()-start,
        scientific_random_observations=0,extra_optimizer_calls=0,
        array_sha256=sha(OUT/"PMFS.npz"),analytic_checks_passed=True)
    write(OUT/"SUMMARY.json",summary)
    outputs=[OUT/n for n in ["TREE_REGISTRY.json","ANALYTIC_CHECKS.json","TEACHERS.json","FITS.json","PMFS.json","SUMMARY.json","FIT_LOG.jsonl"]]
    write(OUT/"RESULTS_FREEZE.json",dict(time_utc=now(),files={x.relative_to(ROOT).as_posix():sha(x) for x in outputs},
          local_arrays={str((OUT/"PMFS.npz").relative_to(ROOT)).replace("\\","/"):sha(OUT/"PMFS.npz")}))
    write(OUT/"STATE.json",dict(status="complete",time_utc=now(),active_jobs=[]),True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["freeze","run"]);args=parser.parse_args()
    try:
        freeze() if args.phase=="freeze" else run()
    except Exception as error:
        if (OUT/"FREEZE.json").exists() and not (OUT/"RESULTS_FREEZE.json").exists():
            state=read(OUT/"STATE.json")
            write(OUT/f"FAILURE_{time.time_ns()}.json",dict(time_utc=now(),phase=args.phase,error=repr(error),traceback=traceback.format_exc(),
                last_state=state,scientific_restart_allowed=False))
            write(OUT/"STATE.json",dict(status="stopped_failure",time_utc=now(),active_jobs=[]),True)
        raise
