"""Independent T parameter, completeness, convex-gap and query replay; no fits/RNG."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="2"
from pathlib import Path
import argparse, hashlib, itertools, json, math
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,msg):
    if not ok:raise AssertionError(msg)
def states(d):return ((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(int)
def kl(p,q):return float(p@(np.log(p)-np.log(q)))
def independent_queries(p,models,k):
    d=int(round(math.log2(len(p))));x=states(d);incidence=[];queries=0
    for O in itertools.combinations(range(d),k):
        for j in range(d):
            if j in O:continue
            queries+=1
            for bits in itertools.product([0,1],repeat=k):
                compatible=np.ones(len(p),dtype=bool)
                for i,b in zip(O,bits):compatible &= x[:,i]==b
                for b in [0,1]:incidence.append(compatible & (x[:,j]==b))
    matrix=np.array(incidence,dtype=float)
    pj=(matrix@p).reshape(queries,1<<k,2)
    qj=(models@matrix.T).reshape(len(models),queries,1<<k,2)
    pe=pj.sum(-1);qe=qj.sum(-1)
    pc=pj/pe[:,:,None];qc=qj/qe[:,:,:,None]
    logrisk=np.sum(pj[None,:,:,:]*(np.log(pc[None,:,:,:])-np.log(qc)),axis=(1,2,3))/queries
    brier=np.sum(pe[None,:,:]*(pc[None,:,:,1]-qc[:,:,:,1])**2,axis=(1,2))/queries
    return logrisk,brier
def validate_tree(nodes):
    nodes=[frozenset(n) for n in nodes];root=frozenset(range(6))
    check(len(nodes)==5 and root in nodes and len(set(nodes))==5,"tree root/internal count")
    for node in nodes:
        check(2<=len(node)<=6 and node<=root,"tree scope")
        for other in nodes:
            check(not(node&other) or node<=other or other<=node,"nonlaminar tree")
        proper=[s for s in nodes if s<node]
        children=[s for s in proper if not any(s<t<node for t in proper)]
        used=set().union(*children) if children else set()
        check(len(children)+len(node-used)==2,"not full binary")
    return tuple(sorted(tuple(sorted(n)) for n in nodes))
def audit():
    check(read(OUT/"STATE.json")["status"]=="complete","T incomplete")
    for receipt,key in [("FREEZE.json","inputs"),("RESULTS_FREEZE.json","files")]:
        for p,h in read(OUT/receipt)[key].items():check(sha(ROOT/p)==h,p)
    for p,h in read(OUT/"RESULTS_FREEZE.json")["local_arrays"].items():check(sha(ROOT/p)==h,p)
    registry=read(OUT/"TREE_REGISTRY.json");fits=read(OUT/"FITS.json")
    check(len(registry)==len(fits)==945,"candidate count")
    descriptions=[validate_tree(t["nodes"]) for t in registry]
    check(len(set(descriptions))==math.prod([1,3,5,7,9])==945,"tree completeness")
    data=read(OUT/"TEACHERS.json");p=np.array(data["d6_blind"]);stored=np.array(read(OUT/"PMFS.json"))
    check(stored.shape==(945,64),"public PMF shape")
    with np.load(OUT/"PMFS.npz",allow_pickle=False) as a:
        check(np.array_equal(a["q"],stored) and np.array_equal(a["teacher"],p),"array/public identity")
    x=states(6);y=2*x-1;H=-float(p@np.log(p));models=[];lower=[];upper=[];qualified=[];per_fit=[]
    maxima=dict(PMF=0.,objective=0.,gradient=0.,box_gap=0.,joint_KL=0.,query_KL=0.,query_Brier=0.)
    for tree,fit in zip(registry,fits):
        check(tree["tree_id"]==fit["tree_id"],"tree/fit order")
        check(tree["nodes"]==fit["nodes"],"tree nodes changed")
        params=fit["params"];nodes=sorted([tuple(n) for n in params["nodes"]],key=lambda n:(len(n),n))
        check(validate_tree(nodes)==descriptions[tree["tree_id"]],"parameter nodes")
        pieces=[y.astype(float)]
        for node in nodes:
            count=x[:,node].sum(1)
            pieces.append(np.array([count==n for n in range(2,len(node)+1)],dtype=float).T)
        F=np.ascontiguousarray(np.column_stack(pieces));theta=np.array(params["theta"])
        bound=np.r_[np.full(6,8.),np.full(F.shape[1]-6,64.)]
        check(theta.shape==(F.shape[1],) and np.all(abs(theta)<=bound),"box/shape")
        check(params["feature_sha256"]==hashlib.sha256(F.tobytes()).hexdigest(),"feature matrix hash")
        check(np.array_equal(params["lower_bounds"],-bound) and np.array_equal(params["upper_bounds"],bound),"box metadata")
        score=np.sum(F*theta[None,:],axis=1);logq=score-logsumexp(score);q=np.exp(logq);models.append(q)
        gradient=(q-p)@F
        gap=float(gradient@theta+np.abs(gradient)@bound)
        objective=-float(p@logq);joint=objective-H
        errors=dict(PMF=float(abs(q-stored[fit["tree_id"]]).max()),objective=abs(objective-fit["objective"]),
            gradient=float(abs(gradient-np.array(fit["optimizer_receipt"]["gradient"])).max()),
            box_gap=abs(gap-fit["gap"]),joint_KL=abs(joint-fit["joint_KL"]))
        for key,v in errors.items():maxima[key]=max(maxima[key],v)
        check(max(errors.values())<1e-10,"independent parameter replay")
        check(gap<=1e-8+1e-10 and fit["qualified"] and fit["PMF_valid"],"T candidate qualification")
        upper.append(joint);lower.append(joint-gap);qualified.append(fit["qualified"])
        per_fit.append(dict(tree_id=fit["tree_id"],independent_gap=gap,independent_joint_KL=joint,errors=errors))
    models=np.array(models)
    for k in [1,5]:
        lr,br=independent_queries(p,models,k)
        for i,fit in enumerate(fits):
            maxima["query_KL"]=max(maxima["query_KL"],abs(float(lr[i])-fit["queries"][str(k)]["conditional_excess_KL"]))
            maxima["query_Brier"]=max(maxima["query_Brier"],abs(float(br[i])-fit["queries"][str(k)]["excess_Brier"]))
    check(max(maxima.values())<1e-10,"independent query replay")
    summary=read(OUT/"SUMMARY.json")
    check(summary["qualified"]==945 and summary["failed_tree_ids"]==[],"summary qualifications")
    check(abs(min(upper)-summary["boxed_union_joint_KL_upper"])<1e-10,"union upper")
    check(abs(min(lower)-summary["boxed_union_joint_KL_lower_raw"])<1e-10,"union lower")
    # Independent analytic construction using spins rather than binary differences.
    analytic=read(OUT/"ANALYTIC_CHECKS.json");identity_error=0.
    def kap(z):return .5*((1+z)*math.log1p(z)+(1-z)*math.log1p(-z))
    for d in [6,8]:
        r=d//2;xx=states(d);yy=2*xx-1;hh=np.prod(yy[:,::2]-yy[:,1::2],axis=1)/2**r
        pp=(1+.9*hh)/2**d;uu=np.full(2**d,1/2**d)
        record=analytic[f"d{d}"]
        identity_error=max(identity_error,abs(kl(pp,uu)-2**(-r)*kap(.9)))
        lr,br=independent_queries(pp,uu[None,:],d-1)
        identity_error=max(identity_error,abs(float(lr[0])-2**(-r)*(kap(.9)-2*kap(.45))),
                           abs(float(br[0])-2**(-r)*.9**2/(2*(4-.9**2))))
        check(record["passed"],"analytic record qualification")
    diag=analytic["d3_projection"];p3=np.array(diag["p"]);q3=np.array(diag["Qstar"]);u3=np.full(8,1/8);x3=states(3)
    event=x3[:,0]
    pe=np.bincount(event,weights=p3);p1=np.bincount(event,weights=p3*x3[:,2])
    qe=np.bincount(event,weights=q3);q1=np.bincount(event,weights=q3*x3[:,2])
    pp=p1/pe;qq=q1/qe
    lr=float(np.sum(pe*(pp*np.log(pp/qq)+(1-pp)*np.log((1-pp)/(1-qq)))))
    br=float(pe@((pp-qq)**2))
    identity_error=max(identity_error,abs(lr+.5*math.log(1-.4**2/4)),abs(br-.4**2/16),
                       abs(kl(p3,u3)-kl(p3,q3)-kl(q3,u3)))
    check(identity_error<1e-12,"analytic formulas")
    return dict(passed=True,all_labelled_binary_trees=945,all_population_fits_replayed=945,
        qualified=945,max_errors=maxima,analytic_identity_max_error=identity_error,
        independently_replayed_union_lower=min(lower),independently_replayed_union_upper=min(upper),
        finite_box_only=True,no_new_fits=True,no_random_draws=True,no_core_import=True,
        audit_source_sha256=sha(Path(__file__)),results_freeze_sha256=sha(OUT/"RESULTS_FREEZE.json"),
        candidate_details=per_fit)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--write",action="store_true");p.add_argument("--check",action="store_true");args=p.parse_args()
    check(not(args.write and args.check),"choose write or check")
    result=audit();path=OUT/"AUDIT.json"
    if args.write:
        with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    if args.check:check(read(path)==result,"saved audit differs")
    print(json.dumps({k:v for k,v in result.items() if k!="candidate_details"},ensure_ascii=False))
