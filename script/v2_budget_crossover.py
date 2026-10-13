"""C: fixed high-budget crossover check, separate from immutable R2."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, os, platform, sys, time, traceback
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/structure_tradeoff_v2/C"
R2=ROOT/"artifacts/structure_tradeoff_v2/R2"
DOC=ROOT/"docs/08_research_研究工作/结构预算与对称化代价_v2_20260930"
CFG=dict(d=16,c=.9,profiles={"null":[4.,4.],"mild":[4.5,3.5],"contrast":[6.,2.]},
         K=[1000,100000],streams=8,banks=32,chunk=4096,seedbase=20301101,
         alpha=.05,contrasts=18,SE_gate=1e-4,mass_gate=1e-12,identity_gate=1e-10,
         propagated_reference_gate=1e-8,primary_K=100000,primary_profiles=["mild","contrast"],
         primary_direction_threshold=0.,joint_sensitivity_alpha_each=.025,
         primary_reference="adaptive",reference_names=["GH256","GH512","adaptive"],
         new_observed_labels=0,new_fits=0)
NAMES=["group_excess","single_excess","single_minus_group"]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(ok,msg):
    if not ok:raise RuntimeError(msg)
def write(p,value,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x" if exclusive else "w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvwrite(p,rows):
    with p.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geometry():
    m=CFG["d"]//2
    pairs=[(a,b) for a in range(m+1) for b in range(m+1)]
    ns=np.array([a+b for a,b in pairs])
    lg=np.log([math.comb(m,a)*math.comb(m,b) for a,b in pairs])
    lt=np.log([math.comb(CFG["d"],int(n)) for n in ns])
    return ns,lg,lt,np.eye(CFG["d"]+1)[ns]
def functional(p):
    n,lg,lt,ind=geometry();pn=p@ind;ps=pn[n]*np.exp(lg-lt)
    return -float(p@(np.log(p)-lg)),float(p@np.log(p/ps)),pn
def pb(eta):
    p=expit(eta);q=expit(-eta);mass=np.zeros((len(eta),eta.shape[1]+1));mass[:,0]=1
    for j in range(eta.shape[1]):
        mass[:,1:j+2]=mass[:,1:j+2]*q[:,j,None]+mass[:,:j+1]*p[:,j,None]
        mass[:,0]*=q[:,j]
    return mass
def group_pmf(eta):
    m=eta.shape[1]//2;a=pb(eta[:,:m]);b=pb(eta[:,m:])
    return (a[:,:,None]*b[:,None,:]).reshape(len(eta),-1)
def latent(z,scales):
    rho=CFG["c"]/(CFG["d"]-1);a=math.sqrt(1+rho)
    return (a*z-(a-math.sqrt(1-CFG["c"]))*z.mean(-1,keepdims=True))*np.repeat(scales,CFG["d"]//2)
def freeze():
    assert not OUT.exists()
    for name,expected in read(R2/"FREEZE.json")["files"].items():verify(sha(ROOT/name)==expected,name)
    for name,expected in read(R2/"RESULTS_FREEZE.json")["files"].items():verify(sha(ROOT/name)==expected,name)
    assert read(R2/"STATE.json")["status"]=="complete"
    OUT.mkdir(parents=True);(OUT/"local").mkdir();(OUT/"reference").mkdir()
    (OUT/".gitignore").write_text("/local/\n",encoding="utf-8",newline="\n")
    paths=[Path(__file__),DOC/"PLAN_C.md",R2/"FREEZE.json",R2/"RESULTS_FREEZE.json",
           R2/"PRIMARY.csv",R2/"STREAM_SCORES.csv",R2/"POSTHOC_AUDIT.json",ROOT/"script/v2_symmetry_reference_r2.py"]
    for profile in CFG["profiles"]:
        orig=R2/"reference"/f"{profile}_d16.json";verify(read(orig)["numerical_pass"],"bad R2 reference")
        copy=OUT/"reference"/orig.name;copy.write_bytes(orig.read_bytes());paths.extend([orig,copy])
    write(OUT/"CONFIG.json",CFG)
    write(OUT/"ENVIRONMENT.json",dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    paths.extend([OUT/"CONFIG.json",OUT/"ENVIRONMENT.json"])
    write(OUT/"FREEZE.json",dict(time_utc=now(),kind="independent_high_budget_followup",files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    write(OUT/"STATE.json",dict(status="frozen",time_utc=now(),active_jobs=[]))
    print(json.dumps(dict(freeze_sha256=sha(OUT/"FREEZE.json"))),flush=True)
def checkfreeze():
    verify(read(OUT/"CONFIG.json")==CFG,"config")
    for n,h in read(OUT/"FREEZE.json")["files"].items():verify(sha(ROOT/n)==h,n)
def smoke():
    import itertools
    eta=np.array([[-2.7,-.3,.8,1.2],[.2,.2,.2,.2]])
    labels=np.array(list(itertools.product([0.,1.],repeat=4)))
    probs=np.prod(np.where(labels[None,:,:]>0,expit(eta)[:,None,:],expit(-eta)[:,None,:]),axis=-1)
    ix=(3*labels[:,:2].sum(1)+labels[:,2:].sum(1)).astype(int)
    explicit=np.stack([np.bincount(ix,weights=p,minlength=9) for p in probs])
    err=float(np.max(abs(explicit-group_pmf(eta))))
    verify(err<1e-12,"d4 count smoke")
    # Prefix summation uses a fixed deterministic array, no science RNG.
    fixed=np.arange(17,dtype=float);prefix=0.;full=0.
    for offset in range(0,17,7):
        block=fixed[offset:offset+7];full+=float(block.sum())
        take=max(0,min(len(block),5-offset));prefix+=float(block[:take].sum())
    verify(prefix==fixed[:5].sum() and full==fixed.sum(),"prefix smoke")
    write(OUT/"SMOKE.json",dict(passed=True,count_error=err,deterministic_prefix_check=True,new_observations=0))
def span(x):
    return max(float(np.max(abs(x[i]-x[j]))) for i in range(len(x)) for j in range(i))
def stream_run(profile,idx,stream):
    ref=read(OUT/"reference"/f"{profile}_d16.json")
    refs=np.array(ref["variants"]);Hs=np.array([functional(p)[0] for p in refs]);Ds=np.array([functional(p)[1] for p in refs])
    ns,lg,lt,indicator=geometry()
    seed=CFG["seedbase"]+100*idx+stream;rng=np.random.Generator(np.random.PCG64(seed));initial=rng.bit_generator.state
    qgs=[];qns=[];losses=[];referenceFs=[];reference_losses=[];referenceJs=[];times=[]
    maxmass=0.;maxidentity=0.;sensitivity=0.;start=time.perf_counter()
    for bank in range(CFG["banks"]):
        bankstart=time.perf_counter();sumg=np.zeros((2,len(ns)));sumn=np.zeros((2,CFG["d"]+1))
        counts=np.zeros(2,dtype=np.int64)
        for offset in range(0,CFG["K"][-1],CFG["chunk"]):
            size=min(CFG["chunk"],CFG["K"][-1]-offset)
            eta=latent(rng.standard_normal((size,CFG["d"])),CFG["profiles"][profile])
            gp=group_pmf(eta);tp=pb(eta)
            maxmass=max(maxmass,float(np.max(abs(gp.sum(1)-1))),float(np.max(abs(tp.sum(1)-1))))
            maxidentity=max(maxidentity,float(np.max(abs(gp@indicator-tp))))
            for ki,k in enumerate(CFG["K"]):
                take=max(0,min(size,k-offset))
                if take:
                    sumg[ki]+=gp[:take].sum(0);sumn[ki]+=tp[:take].sum(0);counts[ki]+=take
        verify(counts.tolist()==CFG["K"],"prefix counts")
        qg=sumg/np.array(CFG["K"])[:,None];qn=sumn/np.array(CFG["K"])[:,None]
        verify(np.all(qg>0) and np.all(qn>0) and np.isfinite(qg).all() and np.isfinite(qn).all(),"nonpositive/nonfinite mass; no floors")
        maxmass=max(maxmass,float(np.max(abs(qg.sum(1)-1))),float(np.max(abs(qn.sum(1)-1))))
        qs=qn[:,ns]*np.exp(lg-lt)
        maxidentity=max(maxidentity,float(np.max(abs(qg@indicator-qn))),float(np.max(abs(qs@indicator-qn))))
        logs=np.stack([-np.log(qg)+lg,-np.log(qn[:,ns])+lt],axis=1)
        Fs=np.einsum("kqm,rm->rkq",logs,refs);excess=Fs-Hs[:,None,None]
        metrics=np.concatenate([excess,(excess[:,:,1]-excess[:,:,0])[:,:,None]],axis=2)
        jensen=excess[:,:,1]-Ds[:,None]
        sensitivity=max(sensitivity,span(Fs),span(metrics),span(jensen))
        qgs.append(qg);qns.append(qn);losses.append(metrics[2]);referenceFs.append(Fs);reference_losses.append(metrics);referenceJs.append(jensen)
        times.append(time.perf_counter()-bankstart)
    arr=np.array(losses);allFs=np.array(referenceFs);allmetrics=np.array(reference_losses);allJs=np.array(referenceJs)
    ok=maxmass<CFG["mass_gate"] and maxidentity<CFG["identity_gate"] and sensitivity<=CFG["propagated_reference_gate"]
    target=OUT/"local"/f"{profile}_s{stream}.npz";verify(not target.exists(),"overwrite array")
    np.savez_compressed(target,losses=arr,group_count_pmf=np.array(qgs),total_count_pmf=np.array(qns),
                        reference_F=allFs,reference_metrics=allmetrics,reference_single_Jensen=allJs,bank_seconds=np.array(times))
    record=dict(profile=profile,stream=stream,seed=seed,initial_RNG=initial,final_RNG=rng.bit_generator.state,
                means=arr.mean(0).tolist(),bank_count=32,prior_particles=3200000,exact_prefix_counts=CFG["K"],
                numerical_pass=bool(ok),mass_max_error=maxmass,total_count_identity_max_error=maxidentity,
                propagated_reference_max_difference=sensitivity,bank_seconds=times,elapsed_seconds=time.perf_counter()-start,
                array_sha256=sha(target),array_shapes={k:list(v.shape) for k,v in dict(losses=arr,reference_F=allFs,reference_metrics=allmetrics).items()},
                no_probability_floor=True)
    write(OUT/"streams"/f"{profile}_s{stream}.json",record)
    print(json.dumps(dict(phase="stream",profile=profile,stream=stream,passed=bool(ok),seconds=record["elapsed_seconds"])),flush=True)
    verify(ok,"C numerical gate failed; no expansion or restart")
def collect():
    rows=[];scores=[];decisions=[]
    crit=float(t.ppf(1-CFG["alpha"]/(2*CFG["contrasts"]),7))
    for profile in CFG["profiles"]:
        recs=[read(OUT/"streams"/f"{profile}_s{s}.json") for s in range(8)]
        values=np.array([r["means"] for r in recs])
        D=read(OUT/"reference"/f"{profile}_d16.json")["model_KL_single"]
        for ki,k in enumerate(CFG["K"]):
            for qi,name in enumerate(NAMES):
                v=values[:,ki,qi];mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(8));lo=mean-crit*se;hi=mean+crit*se
                passed=se<=CFG["SE_gate"]
                rows.append(dict(profile=profile,d=16,K=k,reading=name,mean=mean,SE=se,lower=lo,upper=hi,precision_pass=passed,
                                 family=18,confidence=.95,model_KL_single=D,single_Jensen=mean-D if qi==1 else None))
                scores.extend(dict(profile=profile,d=16,K=k,reading=name,stream=s,value=float(vv)) for s,vv in enumerate(v))
                if k==CFG["primary_K"] and profile in CFG["primary_profiles"] and name=="single_minus_group":
                    decisions.append(dict(profile=profile,mean=mean,lower=lo,upper=hi,precision_pass=passed,
                                          correct_group_preferred=passed and lo>0.,directional_not_material_threshold=True))
    csvwrite(OUT/"PRIMARY.csv",rows);csvwrite(OUT/"STREAM_SCORES.csv",scores)
    joint=[];r2rows=list(csv.DictReader((R2/"PRIMARY.csv").open(encoding="utf-8")))
    for package,items,family in [("R2",r2rows,72),("C",rows,18)]:
        jc=float(t.ppf(1-.025/(2*family),7))
        for row in items:
            mean=float(row["mean"]);se=float(row["SE"]);lo=mean-jc*se;hi=mean+jc*se
            isprimary=(int(row["d"])==16 and row["profile"] in CFG["primary_profiles"] and row["reading"]=="single_minus_group" and
                       int(row["K"])==(10 if package=="R2" else 100000))
            precision=(row["precision_pass"]=="True") if isinstance(row["precision_pass"],str) else row["precision_pass"]
            joint.append(dict(package=package,profile=row["profile"],d=int(row["d"]),K=int(row["K"]),reading=row["reading"],
                              mean=mean,SE=se,lower=lo,upper=hi,allocated_alpha=.025,package_family=family,
                              posthoc_joint_sensitivity=True,primary_reversal_endpoint=isprimary,precision_pass=precision,
                              desired_sign_pass=(precision and (hi<0 if package=="R2" else lo>0)) if isprimary else None))
    csvwrite(OUT/"JOINT_SENSITIVITY_90.csv",joint)
    selected=[x for x in joint if x["primary_reversal_endpoint"]]
    write(OUT/"DECISION.json",dict(primary=decisions,all_primary_pass=all(x["correct_group_preferred"] for x in decisions),
                                  precision_pass=sum(x["precision_pass"] for x in rows),contrasts=18,
                                  posthoc_joint_90_endpoints=selected,posthoc_joint_reversal_pass=all(x["desired_sign_pass"] for x in selected),
                                  posthoc_joint_not_original_R2_preregistration=True,streams=24,prior_banks=768,
                                  prior_particles=76800000,new_fits=0,new_observed_labels=0))
    return rows,joint
def audit(write_result=True):
    checkfreeze();maxerror=0.;array_count=0;arrays={}
    refs={}
    for profile in CFG["profiles"]:
        ref=read(OUT/"reference"/f"{profile}_d16.json");p=np.array(ref["mean"]);H,D,_=functional(p);refs[profile]=ref
        ns,lg,lt,ind=geometry()
        for stream in range(8):
            record=read(OUT/"streams"/f"{profile}_s{stream}.json");path=OUT/"local"/f"{profile}_s{stream}.npz"
            verify(sha(path)==record["array_sha256"],"array hash");array_count+=1
            z=np.load(path,allow_pickle=False);verify(z["losses"].shape==(32,2,3),"shape")
            for name in z.files:verify(np.isfinite(z[name]).all(),"nonfinite array")
            qg=z["group_count_pmf"];qn=z["total_count_pmf"]
            # A second calculation from saved PMFs, not from the recorded loss means.
            lgloss=-np.log(qg)+lg;lsloss=-np.log(qn[:,:,ns])+lt
            group=np.sum(lgloss*p,axis=-1)-H;single=np.sum(lsloss*p,axis=-1)-H
            metrics=np.stack([group,single,single-group],axis=-1)
            maxerror=max(maxerror,float(np.max(abs(metrics-z["losses"]))),float(np.max(abs(z["losses"].mean(0)-np.array(record["means"])))))
            verify(np.max(abs(qg@ind-qn))<CFG["identity_gate"],"saved total identity")
            arrays[(profile,stream)]=metrics.mean(0)
    rows=list(csv.DictReader((OUT/"PRIMARY.csv").open(encoding="utf-8")));verify(len(rows)==18,"18 rows")
    crit=float(t.ppf(1-.05/(2*18),7));failed=[]
    for row in rows:
        ki=CFG["K"].index(int(row["K"]));qi=NAMES.index(row["reading"])
        vals=[float(arrays[(row["profile"],s)][ki,qi]) for s in range(8)]
        mean=math.fsum(vals)/8;se=math.sqrt(math.fsum((x-mean)**2 for x in vals)/7/8)
        expected=[mean,se,mean-crit*se,mean+crit*se]
        maxerror=max(maxerror,max(abs(float(row[n])-v) for n,v in zip(["mean","SE","lower","upper"],expected)))
        verify((row["precision_pass"]=="True")==(se<=CFG["SE_gate"]),"SE gate")
        if se>CFG["SE_gate"]:failed.append({k:row[k] for k in ["profile","K","reading","SE"]})
    verify(maxerror<2e-12,"independent statistics replay")
    joint=list(csv.DictReader((OUT/"JOINT_SENSITIVITY_90.csv").open(encoding="utf-8")));verify(len(joint)==90,"90 sensitivity rows")
    for row in joint:
        crit=float(t.ppf(1-.025/(2*int(row["package_family"])),7));mean=float(row["mean"]);se=float(row["SE"])
        verify(abs(float(row["lower"])-(mean-crit*se))<2e-12 and abs(float(row["upper"])-(mean+crit*se))<2e-12,"joint interval")
    result=dict(time_utc=now(),passed=True,array_count=array_count,reference_count=3,statistics_rows=18,joint_sensitivity_rows=90,
                independent_replay_max_error=maxerror,precision_failures=failed,no_new_scientific_samples_or_fits=True)
    if write_result:write(OUT/"AUDIT.json",result)
    return result
def report():
    rows=list(csv.DictReader((OUT/"PRIMARY.csv").open(encoding="utf-8")))
    decision=read(OUT/"DECISION.json");audit_result=read(OUT/"AUDIT.json")
    lines=["# C：高预算固定复查结果","",
           "本包是R2完成后另立并冻结的预算复查，不修改原R2规则或结论。d16、原三profile、K1000/100000，24流×32bank，共768bank、76,800,000个先验潜粒子；0观察标签、0拟合。所有bank共用两个K前缀，仅评group/single。","",
           f"18项统计精度通过{decision['precision_pass']}/18；两个事前高预算方向主门{'均通过' if decision['all_primary_pass'] else '未全部通过'}。判定只是single−group下界>0，不是>.001实质改善。","",
           "| profile | K | 读数 | 均值 | SE | 同时95%下界 | 上界 | 精度 |",
           "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['profile']} | {r['K']} | {r['reading']} | {float(r['mean']):.10f} | {float(r['SE']):.3g} | {float(r['lower']):.10f} | {float(r['upper']):.10f} | {'通过' if r['precision_pass']=='True' else '失败'} |")
    lines+=["","## 事后联合敏感性","",
            "R2原72项和C18项分别用alpha=.025构造区间，union bound给出合计90项至少95%的t近似覆盖。这个联合检查在R2之后提出，不能说原R2预注册了90项，也不替换其原判定。参考误差资格单列，区间不认证严格全误差覆盖。","",
            "| 包 | profile | K | 配对均值 | 联合下界 | 联合上界 | 所需方向 |",
            "|---|---|---:|---:|---:|---:|---|"]
    for r in decision["posthoc_joint_90_endpoints"]:
        lines.append(f"| {r['package']} | {r['profile']} | {r['K']} | {r['mean']:.10f} | {r['lower']:.10f} | {r['upper']:.10f} | {'通过' if r['desired_sign_pass'] else '未通过'} |")
    lines+=["",f"四端点联合方向检查：{'通过' if decision['posthoc_joint_reversal_pass'] else '未通过'}。所有90行见JOINT_SENSITIVITY_90.csv。","",
            "只比较指定真参数处的两种评分。single积分目标为错对称化P，group积分目标为P；本包检验理论K趋于无穷时正确风险排序是否已在固定高预算出现。不由此给出最小反转K、训练收敛、全局优化或通用计算复杂度结论。null、K1000以及所有失败格均保留。","",
            f"独立审计从24份已保存count PMF重算逐bank风险/流均值及18区间，最大差{audit_result['independent_replay_max_error']:.3g}。数值参考引用R2三个已通过的确定性变体，不产生新参考样本。freeze SHA256={sha(OUT/'FREEZE.json')}。"]
    text="\n".join(lines)+"\n"
    for path in [DOC/"C_RESULTS.md",OUT/"READABLE.md"]:
        with path.open("x",encoding="utf-8",newline="\n") as f:f.write(text)
def seal():
    files=list((OUT/"streams").glob("*.json"))+[OUT/n for n in ["PRIMARY.csv","STREAM_SCORES.csv","JOINT_SENSITIVITY_90.csv","DECISION.json","AUDIT.json","READABLE.md","SMOKE.json"]]+[DOC/"C_RESULTS.md"]
    write(OUT/"RESULTS_FREEZE.json",dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    write(OUT/"STATE.json",dict(status="complete",time_utc=now(),active_jobs=[]),False)
    public=[p for p in OUT.rglob("*") if p.is_file() and "local" not in p.relative_to(OUT).parts and p.name!="MANIFEST.json"]
    write(OUT/"MANIFEST.json",dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in public},
                                  local_arrays_excluded=True,local_hash_receipts_in_streams=True))
def run():
    checkfreeze();verify(read(OUT/"STATE.json")["status"]=="frozen","no silent restart");smoke()
    for idx,profile in enumerate(CFG["profiles"]):
        for stream in range(8):
            write(OUT/"STATE.json",dict(status="running",profile=profile,stream=stream,pid=os.getpid(),time_utc=now()),False)
            stream_run(profile,idx,stream)
    collect();audit();report();seal()
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["freeze","run","check"]);args=parser.parse_args()
    try:
        if args.phase=="freeze":freeze()
        elif args.phase=="run":run()
        else:
            result=audit(False)
            for n,h in read(OUT/"RESULTS_FREEZE.json")["files"].items():verify(sha(ROOT/n)==h,n)
            for n,h in read(OUT/"MANIFEST.json")["files"].items():verify(sha(ROOT/n)==h,n)
            print(json.dumps(result),flush=True)
    except Exception as ex:
        if args.phase!="check" and (OUT/"FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",dict(time_utc=now(),phase=args.phase,error=repr(ex),traceback=traceback.format_exc()))
            write(OUT/"STATE.json",dict(status="stopped_failure",time_utc=now(),active_jobs=[]),False)
        raise

