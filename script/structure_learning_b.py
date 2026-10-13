"""B: frozen two-scale grid learning with released orientation and group misspecification."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, csv, hashlib, json, math, os, platform, sys, time, traceback
os.environ["OPENBLAS_NUM_THREADS"]="2";os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import t as student_t
import scale_learning_core as core
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/structure_learning_v2/B"
A=ROOT/"artifacts/structure_learning_v2/A"
DOC=ROOT/"docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930"
BASE="5cfbd418a48732ce33c5cde04b370fa5e098cd5f"
CFG=dict(d=16,c=.9,scales_grid=[j/2 for j in range(19)],r_values=[0,1,2,4],
         teacher=[6.,2.],witness=[3.,1.],K=[10,100],streams=16,banks=64,bank_batch=16,
         bank_seed=20340101,train_N=[4096,65536],objectives=["J","P","T","G10","G100"],
         mass_gate=1e-12,identity_gate=1e-10,anchor_L1_gate=1e-8,reference_sensitivity_gate=1e-7,
         finite_family=32,finite_SE_gate=.001,population_family=8,population_SE_gates=[.01,.005],
         alpha=.05,material=.001,T_near_tie_allowance=1e-7,
         oracle_bracket=[0.,9.],oracle_xtol=1e-12,oracle_rtol=1e-12,oracle_maxiter=100,
         oracle_residual_gate=1e-11,oracle_pair_gap_gate=1e-10,
         new_teacher_observations=0,new_neural_models=0)
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(ok,msg):
    if not ok:raise RuntimeError(msg)
def convert(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return x.relative_to(ROOT).as_posix()
    raise TypeError(type(x).__name__)
def write(p,x,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x" if exclusive else "w",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,default=convert,allow_nan=False);f.write("\n")
def csvwrite(p,rows):
    with p.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def hashes(paths):return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def grid():
    return [(u,v) for u in CFG["scales_grid"] for v in CFG["scales_grid"]]
def index(pair):return grid().index(tuple(pair))
def validate_A():
    teacher=read(A/"TEACHER.json")
    verify(teacher["r_values"]==CFG["r_values"],"A r order")
    p=np.array(teacher["mean"]);variants=np.array(teacher["variants"])
    mapped=np.array(teacher["mapped_counts_by_r"]);mv=np.array(teacher["mapped_variants_by_r"])
    verify(p.shape==(81,) and variants.shape==(3,81) and mapped.shape==(4,81) and mv.shape==(4,3,81),"A teacher shapes")
    verify(np.all(p>0) and np.all(mapped>0),"A teacher positivity")
    _,_,_,_,_,_,indicator=core.geometry(16)
    verify(np.max(abs(mapped.sum(1)-1))<CFG["mass_gate"],"A mapped mass")
    verify(np.max(abs(mapped@indicator-p@indicator))<CFG["identity_gate"],"A mapped total count identity")
    records=[]
    for s in range(16):
        r=read(A/"streams"/f"s{s:02}.json");counts=np.array(r["training_counts_by_r"])
        verify(r["stream"]==s and r["N"]==CFG["train_N"] and r["r_values"]==CFG["r_values"],"A stream contract")
        verify(counts.shape==(4,2,81) and np.issubdtype(counts.dtype,np.integer) and np.all(counts>=0),"A integer counts")
        verify(np.array_equal(counts.sum(2),np.repeat(np.array(CFG["train_N"])[None,:],4,axis=0)),"A training sizes")
        ncounts=counts@indicator.astype(np.int64)
        verify(np.array_equal(ncounts,np.repeat(ncounts[:1],4,axis=0)),"A data total-count invariance")
        records.append(r)
    return teacher,records
def freeze():
    verify(not OUT.exists(),"B output already exists")
    verify(read(A/"STATE.json")["status"]=="complete","A is not complete")
    for name,h in read(A/"DATA_FREEZE.json")["files"].items():verify(sha(ROOT/name)==h,"A data hash "+name)
    teacher,records=validate_A()
    OUT.mkdir(parents=True);(OUT/"local").mkdir();(OUT/"reference").mkdir()
    (OUT/".gitignore").write_text("/local/\n",encoding="utf-8",newline="\n")
    write(OUT/"CONFIG.json",CFG);write(OUT/"NUMERICS.json",core.numerics_contract())
    write(OUT/"ENVIRONMENT.json",dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    rows=[dict(index=i,u=u,v=v,mirror_index=index([v,u]),independent_reference=u<=v) for i,(u,v) in enumerate(grid())]
    csvwrite(OUT/"GRID.csv",rows)
    inputs=[Path(__file__),ROOT/"script/scale_learning_core.py",DOC/"PLAN_B.md",DOC/"THEORY.md",
            OUT/"CONFIG.json",OUT/"NUMERICS.json",OUT/"ENVIRONMENT.json",OUT/"GRID.csv",
            A/"DATA_FREEZE.json",A/"RESULTS_FREEZE.json",A/"TEACHER.json",A/"MAPPING.json"]
    inputs += [A/"streams"/f"s{s:02}.json" for s in range(16)]
    write(OUT/"FREEZE.json",dict(time_utc=now(),baseline_commit=BASE,files=hashes(inputs),
          A_data_receipt_covers_local_pattern_histograms=True))
    write(OUT/"STATE.json",dict(status="frozen",time_utc=now(),active_jobs=[]))
    print(json.dumps(dict(freeze_sha256=sha(OUT/"FREEZE.json"))),flush=True)
def checkfreeze():
    verify(read(OUT/"CONFIG.json")==CFG,"B config changed")
    for name,h in read(OUT/"FREEZE.json")["files"].items():verify(sha(ROOT/name)==h,"B freeze "+name)
def transpose_reference(ref,new_scales,source_index,source_hash):
    start=time.perf_counter()
    out=copy.deepcopy(ref)
    for name in ["mean","variants","raw_variants","symmetrized_group_p"]:
        x=np.array(out[name]);out[name]=x.reshape(x.shape[:-1]+(9,9)).swapaxes(-1,-2).reshape(x.shape)
    out["scales"]=list(new_scales)
    for name in ["variant_spin_moments","independent_moment_abs_errors"]:
        out[name]=np.array(out[name])[:,[1,0,2]]
    moments=out["independent_Hermite_moments"]
    for name in ["expected_spin_moments","pair_magnitudes_AA_BB_AB","pair_error_estimates_plus_analytic_tail","moment_error_estimates_plus_tail"]:
        values=moments[name];moments[name]=[values[1],values[0],values[2]]
    out.update(derived_by_exact_transpose=True,independent_reference=False,source_index=source_index,source_sha256=source_hash,
               source_elapsed_seconds=ref.get("elapsed_seconds"),independent_integral_seconds=0.)
    # Re-evaluate, without integration, the public derived-reference identities.
    na,nb,_,_,_,_,_=core.geometry(16)
    weights=np.stack([(2*na-8)**2,(2*nb-8)**2,(2*na-8)*(2*nb-8)],axis=1)
    err=float(np.max(abs(np.array(out["variants"])@weights-np.array(moments["expected_spin_moments"]))))
    verify(err<=core.numerics_contract()["moment_gate"],"transpose moment identity")
    out["transpose_moment_error"]=err
    out["elapsed_seconds"]=time.perf_counter()-start
    return out
def references():
    independent=0;derived=0
    for i,(u,v) in enumerate(grid()):
        if u>v:continue
        start=time.perf_counter();ref=core.reference(16,[u,v],.9)
        ref.update(index=i,u=u,v=v,independent_reference=True,derived_by_exact_transpose=False,elapsed_seconds=time.perf_counter()-start)
        path=OUT/"reference"/f"g{i:03}.json";write(path,ref);verify(ref["qualified"],f"unqualified candidate {u},{v}")
        independent+=1
        if u<v:
            j=index([v,u]);mirror=transpose_reference(ref,[v,u],i,sha(path));mirror.update(index=j,u=v,v=u)
            write(OUT/"reference"/f"g{j:03}.json",mirror);derived+=1
        print(json.dumps(dict(phase="reference",index=i,u=u,v=v,independent=independent,transposes=derived)),flush=True)
    verify(independent==190 and derived==171,"reference counts")
    info=load_reference()
    anchor_error=float(abs(info["pgrid"][index(CFG["teacher"])]-info["teacher"]["mean"]).sum())
    write(OUT/"REFERENCE_ANCHOR.json",dict(L1=anchor_error,passed=anchor_error<=CFG["anchor_L1_gate"]))
    verify(anchor_error<=CFG["anchor_L1_gate"],"teacher anchor error")
    information_obstruction(info)
    oracle(info)
    write(OUT/"REFERENCE_FREEZE.json",dict(time_utc=now(),independent_grid_references=190,
        transpose_grid_references=171,diagnostic_full_references=1,continuous_population_solutions=1,
        files=hashes(list((OUT/"reference").glob("*.json"))+[OUT/"REFERENCE_ANCHOR.json",OUT/"INFO_OBSTRUCTION.json",OUT/"ORACLE_REFERENCE.json",OUT/"ORACLE_DIAGNOSTIC.json"])))
def load_reference():
    teacher,_=validate_A()
    refs=[read(OUT/"reference"/f"g{i:03}.json") for i in range(361)]
    verify(all(r["qualified"] for r in refs),"unqualified candidate")
    variants=np.array([r["variants"] for r in refs]);pgrid=variants[:,2]
    _,_,ns,lg,lt,_,indicator=core.geometry(16)
    qn=variants@indicator
    # Mirror T candidates share the identical numerical probability vector.
    for i,(u,v) in enumerate(grid()):
        j=index([v,u])
        if i<=j:
            common=.5*(qn[i]+qn[j]);qn[i]=common;qn[j]=common
    verify(np.max(abs(qn.sum(2)-1))<CFG["mass_gate"],"symmetrized total PMF mass")
    for i,(u,v) in enumerate(grid()):verify(np.array_equal(qn[i],qn[index([v,u])]),"T mirror exact identity")
    J=-np.log(pgrid)+lg;T=-np.log(qn[:,2]);P=np.array([core.pair_loss_vectors(r) for r in refs])
    mapped=np.array(teacher["mapped_counts_by_r"]);mappedv=np.array(teacher["mapped_variants_by_r"])
    H=float(teacher["H_original"])
    hv=np.array([-float(p@(np.log(p)-lg)) for p in np.array(teacher["variants"])])
    verify(abs(hv[2]-H)<CFG["identity_gate"],"original teacher entropy identity")
    risk=mapped@J.T-H;Tpop=T@(np.array(teacher["mean"])@indicator)
    exactJ=mapped@J.T;exactP=mapped@P.T;bestJ=risk.min(1)
    sensitivity=0.
    for ri in range(4):
        candidates=[]
        for va in range(3):
            for vb in range(3):
                candidates.append(mappedv[ri,va]@(-np.log(variants[:,vb])+lg).T-hv[va])
        sensitivity=max(sensitivity,float(np.ptp(candidates,axis=0).max()))
    # Total-likelihood cross-reference sensitivity, including the shared symmetrization.
    totals=[]
    tv=np.array(teacher["variants"])@indicator
    for va in range(3):
        for vb in range(3):totals.append((-np.log(qn[:,vb]))@tv[va])
    sensitivity=max(sensitivity,float(np.ptp(totals,axis=0).max()))
    verify(sensitivity<=CFG["reference_sensitivity_gate"],"cross-reference risk sensitivity")
    return dict(teacher=teacher,refs=refs,variants=variants,pgrid=pgrid,qn=qn,J=J,P=P,T=T,H=H,
                mapped=mapped,mappedv=mappedv,risk=risk,exactJ=exactJ,exactP=exactP,exactT=Tpop,bestJ=bestJ,
                reference_sensitivity=sensitivity,indicator=indicator)
def oracle(info):
    p=np.array(info["teacher"]["mean"]);na,nb,_,_,_,_,_=core.geometry(16)
    totalspin=2*(na+nb)-16;rbar=float((16-p@(totalspin**2))/(16*15))
    calls=[]
    def equation(w):
        h=core.independent_moment_reference(16,[w,w],.9)
        value=float(h["pair_magnitudes_AA_BB_AB"][0])
        calls.append(dict(w=float(w),pair_magnitude=value,error_estimate=h["pair_error_estimates_plus_analytic_tail"][0]))
        return value-rbar
    w,solver=brentq(equation,*CFG["oracle_bracket"],xtol=CFG["oracle_xtol"],rtol=CFG["oracle_rtol"],
                   maxiter=CFG["oracle_maxiter"],full_output=True,disp=True)
    residual=abs(equation(w));verify(solver.converged and residual<CFG["oracle_residual_gate"],"continuous oracle root")
    ref=core.reference(16,[w,w],.9);write(OUT/"ORACLE_REFERENCE.json",ref);verify(ref["qualified"],"continuous oracle full reference")
    ri=CFG["r_values"].index(4);target=info["mapped"][ri]
    pair=float(core.pair_loss_vectors(ref)@target)
    lower=math.log(4)-.5*(1-rbar)*math.log1p(-rbar)-.5*(1+rbar)*math.log1p(rbar)
    verify(abs(pair-lower)<CFG["oracle_pair_gap_gate"],"continuous pair half-space optimum identity")
    lg=core.geometry(16)[3];joint=float((-np.log(ref["mean"])+lg)@target-info["H"])
    gridmin=float(info["exactP"][ri].min());ii=int(np.argmin(info["exactP"][ri]))
    verify(gridmin-lower>=-CFG["oracle_pair_gap_gate"],"grid pair loss below continuous halfspace lower bound")
    write(OUT/"ORACLE_DIAGNOSTIC.json",dict(r=4,continuous_scales=[float(w),float(w)],rbar=rbar,
        root_residual=residual,root_iterations=solver.iterations,hermite_evaluations=calls,
        pair_loss=pair,pair_halfspace_lower_bound=lower,continuous_oracle_joint_KL=joint,
        grid_pair_minimum=gridmin,grid_minus_continuous_pair_gap=gridmin-lower,
        grid_selected_index=ii,grid_selected_scales=list(grid()[ii]),grid_selected_diagonal=grid()[ii][0]==grid()[ii][1],
        not_added_to_fitting_grid=True,not_a_finite_grid_diagonal_theorem=True))
def information_obstruction(info):
    plus=index(CFG["teacher"]);minus=index(list(reversed(CFG["teacher"])))
    rows=[];count_error=0.;direction_error=0.
    for vi,name in enumerate(core.numerics_contract()["reference_variants"]):
        p=info["variants"][plus,vi];q=info["variants"][minus,vi];mid=.5*(p+q)
        a=float(p@np.log(p/mid));b=float(q@np.log(q/mid))
        forward=float(p@np.log(p/q));reverse=float(q@np.log(q/p))
        count_error=max(count_error,float(abs(p@info["indicator"]-q@info["indicator"]).max()))
        direction_error=max(direction_error,abs(a-b),abs(forward-reverse))
        rows.append(dict(variant=name,JS=.5*(a+b),KL_plus_mid=a,KL_minus_mid=b,
                         KL_plus_minus=forward,KL_minus_plus=reverse))
    metrics=np.array([[r[k] for k in ["JS","KL_plus_mid","KL_minus_mid","KL_plus_minus","KL_minus_plus"]] for r in rows])
    sensitivity=float(np.ptp(metrics,axis=0).max())
    ok=sensitivity<=CFG["reference_sensitivity_gate"] and count_error<CFG["identity_gate"] and direction_error<CFG["identity_gate"]
    result=dict(plus_scales=CFG["teacher"],minus_scales=list(reversed(CFG["teacher"])),variants=rows,
                primary_JS=rows[2]["JS"],raw_total_count_max_difference=count_error,
                shared_T_total_count_difference=float(abs(info["qn"][plus]-info["qn"][minus]).max()),
                swap_symmetry_max_error=direction_error,reference_sensitivity=sensitivity,numerical_pass=bool(ok),
                count_only_minimax_lower_bound_for_any_training_size=rows[2]["JS"],
                conditional_on_qualified_reference_not_strict_all_error_certificate=True,
                additional_teacher_observations=0,additional_references=0)
    write(OUT/"INFO_OBSTRUCTION.json",result);verify(ok,"information obstruction numerical gate")
def smoke():
    # No scientific RNG: exact finite arithmetic and mirror mappings.
    for i,pair in enumerate(grid()):verify(index(list(reversed(pair)))==19*(i%19)+i//19,"mirror index")
    _,_,_,_,_,_,ind=core.geometry(16)
    verify(ind.shape==(81,17) and np.array_equal(ind.sum(1),np.ones(81)),"count mapping")
    eta=np.array([[-2.7,-.3,.8,1.2],[.2,.2,.2,.2]])
    identity=float(abs(core.joint_mass(eta)@core.geometry(4)[6]-core.pb(eta)).max())
    verify(identity<CFG["mass_gate"],"deterministic group/total probability identity")
    write(OUT/"SMOKE.json",dict(passed=True,scientific_random_draws=0,probability_identity_error=identity))
def bank_stream(stream,info):
    seed=CFG["bank_seed"]+stream;rng=np.random.Generator(np.random.PCG64(seed));initial=rng.bit_generator.state
    means=np.zeros((361,2,81));normerr=0.;identity=0.;start=time.perf_counter()
    lg=core.geometry(16)[3];indicator=info["indicator"]
    for offset in range(0,64,16):
        z=rng.standard_normal((1600,16))
        for ti,pair in enumerate(grid()):
            eta=core.eta_from_z(z,pair,.9);mass=core.joint_mass(eta);totalmass=core.pb(eta)
            normerr=max(normerr,float(abs(mass.sum(1)-1).max()))
            identity=max(identity,float(abs(mass@indicator-totalmass).max()))
            mass=mass.reshape(16,100,81)
            for ki,k in enumerate(CFG["K"]):
                q=mass[:,:k].mean(1);verify(np.all(q>0) and np.isfinite(q).all(),"nonpositive G likelihood")
                normerr=max(normerr,float(abs(q.sum(1)-1).max()))
                means[ti,ki]+=(-np.log(q)+lg).sum(0)
    means/=64
    sensitivity=0.
    for ri in range(4):
        values=means@info["mappedv"][ri].T
        delta=values[index(CFG["witness"])]-values[index(CFG["teacher"])]
        sensitivity=max(sensitivity,float(np.ptp(values,axis=-1).max()),float(np.ptp(delta,axis=-1).max()))
    verify(normerr<CFG["mass_gate"] and identity<CFG["identity_gate"],"G probability mass/count identity")
    verify(sensitivity<=CFG["reference_sensitivity_gate"],"G reference sensitivity")
    a=read(A/"streams"/f"s{stream:02}.json");counts=np.array(a["training_counts_by_r"])
    pop=np.stack([means@info["mapped"][ri] for ri in range(4)])
    finite=np.stack([np.stack([means@(counts[ri,ni]/N) for ni,N in enumerate(CFG["train_N"])]) for ri in range(4)])
    target=OUT/"local"/f"s{stream:02}.npz";verify(not target.exists(),"no bank overwrite")
    np.savez_compressed(target,mean_pattern_loss=means,population_objectives=pop,finite_objectives=finite)
    write(OUT/"streams"/f"s{stream:02}.json",dict(stream=stream,seed=seed,initial_RNG=initial,final_RNG=rng.bit_generator.state,
        bank_count=64,unique_prior_particles=6400,candidate_particle_evaluations=2310400,
        population_objectives=pop,finite_objectives=finite,normalization_error=normerr,count_identity_error=identity,
        propagated_reference_sensitivity=sensitivity,numerical_pass=True,array_path=target,array_sha256=sha(target),
        A_stream_sha256=sha(A/"streams"/f"s{stream:02}.json"),elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(phase="bank_stream",stream=stream,seconds=time.perf_counter()-start)),flush=True)
def ci(values,family,gate):
    values=np.asarray(values);mean=float(values.mean());se=float(values.std(ddof=1)/4)
    crit=float(student_t.ppf(1-CFG["alpha"]/(2*family),15))
    return dict(mean=mean,SE=se,lower=mean-crit*se,upper=mean+crit*se,precision_pass=se<=gate)
def collect(info):
    records=[read(OUT/"streams"/f"s{s:02}.json") for s in range(16)]
    pop=np.array([r["population_objectives"] for r in records]);finite=np.array([r["finite_objectives"] for r in records])
    reference_rows=[];population_fits=[];finite_fits=[];ties=[];fixed=[];riskintervals=[];decisions=[]
    finite_risks=np.zeros((4,2,16,5))
    for ri,r in enumerate(CFG["r_values"]):
        bestj=int(np.argmin(info["exactJ"][ri]))
        for ti,(u,v) in enumerate(grid()):
            reference_rows.append(dict(r=r,index=ti,u=u,v=v,joint_KL=float(info["risk"][ri,ti]),
                excess_over_grid_J=float(info["risk"][ri,ti]-info["bestJ"][ri]),J=float(info["exactJ"][ri,ti]),
                P=float(info["exactP"][ri,ti]),T=float(info["exactT"][ti]),reference_sensitivity=info["reference_sensitivity"]))
        det={"J":info["exactJ"][ri],"P":info["exactP"][ri],"T":info["exactT"]}
        for obj in CFG["objectives"]:
            curve=det[obj] if obj in det else pop[:,ri,:,CFG["K"].index(int(obj[1:]))].mean(0)
            ii=int(np.argmin(curve))
            population_fits.append(dict(r=r,unit="aggregate_or_deterministic",objective=obj,index=ii,u=grid()[ii][0],v=grid()[ii][1],
                objective_value=float(curve[ii]),joint_KL=float(info["risk"][ri,ii]),excess_over_grid_J=float(info["risk"][ri,ii]-info["bestJ"][ri]),
                grid_selected_diagonal=grid()[ii][0]==grid()[ii][1],exact_continuous_optimum=False))
            if obj=="T":ties.append(tie_record(r,"population",None,curve,info,ri))
        for ki,k in enumerate(CFG["K"]):
            witness=index(CFG["witness"]);truth=index(CFG["teacher"])
            stat=ci(pop[:,ri,witness,ki]-pop[:,ri,truth,ki],8,CFG["population_SE_gates"][ki])
            fixed.append(dict(r=r,K=k,witness=CFG["witness"],anchor=CFG["teacher"],**stat))
            if r==0 and k==10:
                decisions.append(dict(kind="population_fixed_witness",r=r,passed=stat["precision_pass"] and stat["upper"]<-.001,**stat))
            for s in range(16):
                ii=int(np.argmin(pop[s,ri,:,ki]))
                population_fits.append(dict(r=r,unit=str(s),objective="G"+str(k),index=ii,u=grid()[ii][0],v=grid()[ii][1],
                    objective_value=float(pop[s,ri,ii,ki]),joint_KL=float(info["risk"][ri,ii]),
                    excess_over_grid_J=float(info["risk"][ri,ii]-info["bestJ"][ri]),
                    grid_selected_diagonal=grid()[ii][0]==grid()[ii][1],exact_continuous_optimum=False))
        for s in range(16):
            counts=np.array(read(A/"streams"/f"s{s:02}.json")["training_counts_by_r"])
            totalcounts=counts[0]@info["indicator"].astype(np.int64)
            for ni,N in enumerate(CFG["train_N"]):
                empirical=counts[ri,ni]/N
                curves={"J":info["J"]@empirical,"P":info["P"]@empirical,"T":info["T"]@(totalcounts[ni]/N),
                        "G10":finite[s,ri,ni,:,0],"G100":finite[s,ri,ni,:,1]}
                for oi,obj in enumerate(CFG["objectives"]):
                    curve=curves[obj];ii=int(np.argmin(curve));risk=float(info["risk"][ri,ii]);finite_risks[ri,ni,s,oi]=risk
                    finite_fits.append(dict(r=r,N=N,stream=s,objective=obj,index=ii,u=grid()[ii][0],v=grid()[ii][1],
                        train_objective=float(curve[ii]),joint_KL=risk,excess_over_grid_J=risk-float(info["bestJ"][ri]),
                        grid_minimum_gap=0.,grid_selected_diagonal=grid()[ii][0]==grid()[ii][1]))
                    if obj=="T":ties.append(tie_record(r,s,N,curve,info,ri))
        for ni,N in enumerate(CFG["train_N"]):
            for obj in ["P","G10","G100","T"]:
                values=finite_risks[ri,ni,:,CFG["objectives"].index(obj)]-finite_risks[ri,ni,:,0]
                stat=ci(values,32,.001);riskintervals.append(dict(r=r,N=N,comparison=obj+"-J",**stat))
                if r==0 and N==65536 and obj=="G10":
                    decisions.append(dict(kind="finite_data_excess",r=r,N=N,passed=stat["precision_pass"] and stat["lower"]>.001,**stat))
    verify(len(reference_rows)==1444 and len(finite_fits)==640 and len(riskintervals)==32 and len(fixed)==8,"B ledger counts")
    csvwrite(OUT/"REFERENCE_CURVES.csv",reference_rows);csvwrite(OUT/"POPULATION_FITS.csv",population_fits)
    csvwrite(OUT/"FINITE_FITS.csv",finite_fits);csvwrite(OUT/"FINITE_RISK_INTERVALS.csv",riskintervals)
    csvwrite(OUT/"POPULATION_FIXED_COMPARISONS.csv",fixed);write(OUT/"T_EQUIVALENCE_CLASSES.json",ties)
    oracle_diag=read(OUT/"ORACLE_DIAGNOSTIC.json")
    write(OUT/"DECISION.json",dict(primary=decisions,finite_precision_pass=sum(x["precision_pass"] for x in riskintervals),
        population_precision_pass=sum(x["precision_pass"] for x in fixed),finite_grid_fits=640,
        bank_only_population_grid_fits=128,aggregate_or_deterministic_population_endpoints=20,
        continuous_population_oracles=1,independent_grid_references=190,derived_transpose_grid_references=171,
        diagnostic_full_references=1,unique_prior_particles=102400,candidate_particle_evaluations=36966400,
        new_teacher_observations=0,r4_grid_pair_selected_diagonal=oracle_diag["grid_selected_diagonal"],
        r4_continuous_pair_theorem_not_a_grid_diagonal_guarantee=True,T_orientation_success_not_claimed=True))
    files=list((OUT/"reference").glob("*.json"))+list((OUT/"streams").glob("*.json"))+list(OUT.glob("*.csv"))
    files += [OUT/n for n in ["REFERENCE_ANCHOR.json","REFERENCE_FREEZE.json","TRAINING_INPUTS_FREEZE.json",
        "ORACLE_REFERENCE.json","ORACLE_DIAGNOSTIC.json","INFO_OBSTRUCTION.json","T_EQUIVALENCE_CLASSES.json","DECISION.json","SMOKE.json"]]
    write(OUT/"RESULTS_FREEZE.json",dict(time_utc=now(),files=hashes(files)))
    write(OUT/"STATE.json",dict(status="complete",time_utc=now(),active_jobs=[]),False)
def tie_record(r,stream,N,curve,info,ri):
    best=float(np.min(curve));exact=np.flatnonzero(curve==best);near=np.flatnonzero(curve<=best+CFG["T_near_tie_allowance"])
    return dict(r=r,stream=stream,N=N,exact_minimum=best,tie_break="first lexicographic index, smaller u",
        exact_indices=exact,near_indices=near,near_allowance=CFG["T_near_tie_allowance"],
        exact_candidates=[dict(index=int(i),u=grid()[i][0],v=grid()[i][1],joint_KL=float(info["risk"][ri,i]),
                               mirror_index=index(list(reversed(grid()[i]))),mirror_joint_KL=float(info["risk"][ri,index(list(reversed(grid()[i]))) ]))
                          for i in exact],orientation_success_not_claimed=True)
def run():
    checkfreeze();verify(read(OUT/"STATE.json")["status"]=="frozen","no silent restart")
    smoke();write(OUT/"STATE.json",dict(status="running_references",pid=os.getpid(),time_utc=now()),False)
    references();info=load_reference()
    for s in range(16):
        write(OUT/"STATE.json",dict(status="running_banks",stream=s,pid=os.getpid(),time_utc=now()),False);bank_stream(s,info)
    write(OUT/"TRAINING_INPUTS_FREEZE.json",dict(time_utc=now(),files=hashes(list((OUT/"streams").glob("*.json")))))
    collect(info)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["freeze","run"]);args=parser.parse_args()
    try:freeze() if args.phase=="freeze" else run()
    except Exception as ex:
        if (OUT/"FREEZE.json").exists() and not (OUT/"RESULTS_FREEZE.json").exists():
            write(OUT/f"FAILURE_{time.time_ns()}.json",dict(time_utc=now(),exception=repr(ex),traceback=traceback.format_exc()))
            write(OUT/"STATE.json",dict(status="stopped_failure",time_utc=now(),active_jobs=[]),False)
        raise
