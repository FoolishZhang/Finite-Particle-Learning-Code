"""Independent saved-parameter PMF/gradient/box-gap replay; no fitting or sampling."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="2"
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, itertools, json, math
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/v3_query"
TOL=1e-10
GAP_TOL=1e-8

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,message):
    if not ok:raise AssertionError(message)
def states(d):
    return ((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(float)
def normalized(logp):return np.exp(logp-logsumexp(logp))
def conditional_teachers():
    """Independent transcription of the frozen B teacher, not a new experiment."""
    x=states(8);spin=2*x-1;n=x.sum(1);beta=np.arange(25,100,10)/100
    dep=[];ind=[]
    for context in [-1,1]:
        p=normalized(-.35*(n-4)**2+context*(spin@beta))
        marg=p@x;dep.append(p)
        ind.append(normalized(np.sum(x*np.log(marg)+(1-x)*np.log1p(-marg),axis=1)))
    return {"dependent":dep,"independent":ind}
def reconstruct(params):
    family=params["family"];d=int(params["d"]);theta=np.asarray(params["theta"],dtype=float)
    x=states(d);spin=2*x-1;count=x.sum(1).astype(int)
    if family=="product":
        expected=d;bounds=np.full(expected,8.)
        score=sum(theta[i]*spin[:,i] for i in range(d))
        logq=score-logsumexp(score)
    elif family=="cardinality":
        expected=2*d-1;bounds=np.r_[np.full(d,8.),np.full(d-1,64.)]
        potential=np.r_[0.,0.,theta[d:]]
        score=sum(theta[i]*spin[:,i] for i in range(d))+potential[count]
        logq=score-logsumexp(score)
        check(params["count_potential"]==potential.tolist() and params["gauge"]=="v0=v1=0","cardinality aliases")
    elif family=="ising":
        pairs=list(itertools.combinations(range(d),2));expected=d+len(pairs);bounds=np.full(expected,8.)
        score=sum(theta[i]*spin[:,i] for i in range(d))
        for k,(i,j) in enumerate(pairs):score=score+theta[d+k]*spin[:,i]*spin[:,j]
        logq=score-logsumexp(score)
        check(params["pair_indices"]==[list(p) for p in pairs] and params["spin_pairs"]==theta[d:].tolist(),"Ising aliases")
    elif family in ("linear_ar","count_ar"):
        expected=d*(d+1)//2;bounds=np.full(expected,8.);logq=np.zeros(1<<d);offset=0
        check(params["order"]==list(range(d)),"AR order")
        for i in range(d):
            block=theta[offset:offset+i+1]
            check(params["logits"][i]==block.tolist(),"AR saved block")
            if family=="linear_ar":
                eta=np.full(1<<d,block[0])
                for j in range(i):eta=eta+block[j+1]*spin[:,j]
            else:
                eta=block[x[:,:i].sum(1).astype(int)]
            logq=logq+x[:,i]*eta-np.logaddexp(0.,eta)
            offset+=i+1
    else:raise ValueError("unknown family "+str(family))
    check(theta.shape==(expected,) and np.isfinite(theta).all(),"theta shape/finite")
    check(np.all(theta>=-bounds) and np.all(theta<=bounds),"declared box feasibility")
    if "spin_unary" in params:check(params["spin_unary"]==theta[:d].tolist(),"spin unary aliases")
    q=np.exp(logq)
    check(np.all(q>0) and np.isfinite(q).all() and abs(float(q.sum())-1)<1e-11,"PMF normalization/full support")
    return logq,q,x,spin,bounds

def independent_gradient(p,q,x,spin,family):
    d=x.shape[1];difference=q-p
    if family in ("product","cardinality","ising"):
        values=[float(difference@spin[:,i]) for i in range(d)]
        if family=="cardinality":
            n=x.sum(1).astype(int)
            values += [float(difference[n==k].sum()) for k in range(2,d+1)]
        elif family=="ising":
            values += [float(difference@(spin[:,i]*spin[:,j])) for i,j in itertools.combinations(range(d),2)]
        return np.array(values)
    # AR probabilities reconstructed from the full normalized q, not from sigmoid(theta).
    gradient=[]
    ids=np.arange(len(q))
    for i in range(d):
        if family=="linear_ar":
            prefix=ids% (1<<i)
            qm=np.bincount(prefix,weights=q,minlength=1<<i)
            q1=np.bincount(prefix,weights=q*x[:,i],minlength=1<<i)
            probability=q1/qm
            residual=p*(probability[prefix]-x[:,i])
            gradient.append(float(residual.sum()))
            gradient += [float(residual@spin[:,j]) for j in range(i)]
        else:
            prefix=x[:,:i].sum(1).astype(int)
            qm=np.bincount(prefix,weights=q,minlength=i+1)
            q1=np.bincount(prefix,weights=q*x[:,i],minlength=i+1)
            probability=q1/qm
            residual=p*(probability[prefix]-x[:,i])
            gradient.extend(np.bincount(prefix,weights=residual,minlength=i+1).tolist())
    return np.array(gradient)

def audit_row(identifier,row,target,truth,saved_q=None):
    raw_mass=float(np.sum(target));p=np.asarray(target,dtype=float)/raw_mass
    check(np.all(p>0) and abs(raw_mass-1)<=1e-12,"target positivity/mass")
    logq,q,x,spin,bounds=reconstruct(row["params"])
    family=row["family"];check(family==row["params"]["family"],"family metadata")
    gradient=independent_gradient(p,q,x,spin,family)
    theta=np.asarray(row["params"]["theta"]);receipt=row["optimizer_receipt"]
    check(np.array_equal(np.array(receipt["lower_bounds"]),-bounds) and np.array_equal(np.array(receipt["upper_bounds"]),bounds),"receipt box")
    objective=-float(p@logq)
    # Minimum of the linear supporting plane over the symmetric box, independent formula.
    gap=float(gradient@theta+np.abs(gradient)@bounds)
    errors=dict(objective=abs(objective-float(row["objective"])),
                gradient=float(np.max(abs(gradient-np.array(receipt["gradient"])))),
                box_gap=abs(gap-float(row["gap"])),
                raw_target_mass=abs(raw_mass-float(receipt["target_raw_mass"])))
    if saved_q is not None:errors["saved_PMF"]=float(np.max(abs(q-np.asarray(saved_q))))
    truth=np.asarray(truth,dtype=float)
    predicted_KL=float(truth@(np.log(truth)-logq))
    marginal_error=float(np.max(abs(truth@x-q@x)))
    errors["prediction_joint_KL"]=abs(predicted_KL-float(row["joint_KL"]))
    errors["prediction_marginal_error"]=abs(marginal_error-float(row["marginal_max_error"]))
    check(max(errors.values())<=TOL,identifier+": numerical replay "+str(errors))
    claimed=bool(row["qualified"]);recorded=float(row["gap"])<=GAP_TOL
    check(claimed==recorded and bool(receipt["converged"])==recorded,"qualification metadata")
    check(gap>=-TOL,"negative independent gap")
    independent_qualified=gap<=GAP_TOL
    boundary=independent_qualified!=claimed
    if boundary:check(abs(gap-GAP_TOL)<=TOL and abs(float(row["gap"])-GAP_TOL)<=TOL,"non-boundary qualification mismatch")
    check(receipt["hidden_restarts"]==0,"unexpected hidden restarts")
    return dict(id=identifier,family=family,dimension=x.shape[1],parameter_count=len(theta),
                errors=errors,independent_gradient=gradient.tolist(),independent_objective=objective,
                reconstructed_marginals=(q@x).tolist(),
                independent_box_gap=gap,recorded_box_gap=float(row["gap"]),
                recorded_qualified=claimed,independent_qualified=independent_qualified,
                qualification_roundoff_boundary=boundary,normalization_error=abs(float(q.sum())-1))

def run():
    a=read(OUT/"A_RESULTS.json");b=read(OUT/"B_RESULTS.json")
    check(a["population_fits"]==50 and len(a["rows"])==50,"A fit count")
    check(b["finite_fits"]==640 and len(b["rows"])==640 and len(b["population_rows"])==40 and len(b["pooled_rows"])==20,"B fit counts")
    for result in (a,b):check(sha(ROOT/result["array_file"])==result["array_sha256"],"saved array hash")
    pmfs=read(OUT/"A_PMFS.json");counts=read(OUT/"B_COUNTS.json");teacher=conditional_teachers()
    results=[];learned_marginals={}
    for row in a["rows"]:
        base=row["teacher"]+"_"+row["encoding"]
        p=np.asarray(pmfs["truth_"+base])
        results.append(audit_row("A/"+base+"/"+row["family"],row,p,p,pmfs["fit_"+base+"_"+row["family"]]))
    with np.load(ROOT/b["array_file"],allow_pickle=False) as archive:
        for row in b["rows"]:
            world=row["world"];rep=row["rep"];xi=row["x"];encoding=row["encoding"];family=row["family"]
            mask=0 if encoding=="native" else 21;perm=np.arange(256)^mask
            number=np.asarray(counts[f"counts_{world}_{rep}_{xi}"],dtype=float)
            check(number.shape==(256,) and np.all(number==np.floor(number)) and number.sum()==2048,"B integer training counts")
            target=(number[perm]+1/256)/2049
            truth=teacher[world][xi][perm]
            replay=audit_row(f"B/finite/{world}/{rep}/{encoding}/{xi}/{family}",row,target,truth,
                archive[f"q_{world}_{rep}_{encoding}_{xi}_{family}"])
            results.append(replay)
            if family in ("product","ising"):
                learned_marginals[(world,rep,encoding,xi,family)]=np.array(replay["reconstructed_marginals"])
    for kind,key in [("population","population_rows"),("pooled","pooled_rows")]:
        for row in b[key]:
            world=row["world"];encoding=row["encoding"];mask=0 if encoding=="native" else 21
            if kind=="population":p=teacher[world][row["x"]]
            else:p=(teacher[world][0]+teacher[world][1])/2
            p=p[np.arange(256)^mask]
            results.append(audit_row(f"B/{kind}/{world}/{encoding}/{row.get('x','pooled')}/{row['family']}",row,p,p))
    check(len(results)==750 and len({r["id"] for r in results})==750,"all fits uniquely replayed")
    check(sum(r["recorded_qualified"] for r in results[:50])==a["qualified"],"A qualification count")
    check(sum(r["recorded_qualified"] for r in results[50:])==b["qualified"],"B qualification count")
    maxerr={}
    for row in results:
        for k,v in row["errors"].items():maxerr[k]=max(maxerr.get(k,0.),v)
    learned_differences=[]
    for world in teacher:
        for rep in range(16):
            for encoding in ("native","flip21"):
                for xi in (0,1):
                    learned_differences.append(float(np.max(abs(learned_marginals[(world,rep,encoding,xi,"product")]
                        -learned_marginals[(world,rep,encoding,xi,"ising")]))))
    check(len(learned_differences)==128,"learned marginal paired cases")
    sources=[Path(__file__),OUT/"A_RESULTS.json",OUT/"A_PMFS.json",OUT/"A_FREEZE.json",
             OUT/"B_RESULTS.json",OUT/"B_COUNTS.json",OUT/"B_FREEZE.json",ROOT/a["array_file"],ROOT/b["array_file"]]
    for package in ["A","B"]:
        for path,expected in read(OUT/f"{package}_FREEZE.json")["inputs"].items():
            check(sha(ROOT/path)==expected,"frozen source "+path)
    return dict(passed=True,fits=750,A_fits=50,B_finite_fits=640,B_population_and_pooled_fits=60,
                qualification_roundoff_boundaries=sum(r["qualification_roundoff_boundary"] for r in results),
                qualified=sum(r["recorded_qualified"] for r in results),max_errors=maxerr,
                learned_marginal_Ising_product_maxdiff=max(learned_differences),learned_marginal_paired_cases=128,
                learned_marginal_diagnostic_posthoc=True,learned_marginal_check_not_preregistered_success_gate=True,
                declared_absolute_replay_tolerance=TOL,no_refits=True,no_random_draws=True,
                no_core_import=True,query_metrics_audited_separately=True,
                certificate_scope="Convex objective in frozen finite boxes, replayed with float64; not rigorous interval arithmetic",
                sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},rows=results)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--write",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    check(not(args.write and args.check),"choose write or check")
    result=run();path=OUT/"OPTIMALITY_AUDIT.json"
    if args.write:
        with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    if args.check:check(read(path)==result,"stored optimality audit differs")
    print(json.dumps({k:v for k,v in result.items() if k not in ("sources","rows")},ensure_ascii=False))
