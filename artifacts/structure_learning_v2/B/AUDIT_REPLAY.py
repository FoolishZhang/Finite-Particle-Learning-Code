"""Independent B output replay; no random draws, integrals, or new fits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math
import numpy as np
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
A=ROOT/"artifacts/structure_learning_v2/A"
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def check(ok,msg):
    if not ok:raise AssertionError(msg)
def main(write=False):
    check(read(OUT/"STATE.json")["status"]=="complete","B not complete")
    for receipt in ["FREEZE.json","REFERENCE_FREEZE.json","RESULTS_FREEZE.json","TRAINING_INPUTS_FREEZE.json"]:
        for name,h in read(OUT/receipt)["files"].items():check(sha(ROOT/name)==h,name)
    cfg=read(OUT/"CONFIG.json");grid=[(u,v) for u in cfg["scales_grid"] for v in cfg["scales_grid"]]
    refs=[read(OUT/"reference"/f"g{i:03}.json") for i in range(361)]
    check(sum(r["independent_reference"] for r in refs)==190,"independent reference count")
    check(sum(r["derived_by_exact_transpose"] for r in refs)==171,"transpose count")
    probs=np.array([r["mean"] for r in refs]);variants=np.array([r["variants"] for r in refs])
    cells=[(a,b) for a in range(9) for b in range(9)];n=np.array([a+b for a,b in cells])
    ind=np.eye(17)[n];lg=np.log([math.comb(8,a)*math.comb(8,b) for a,b in cells])
    qn=variants@ind
    for i,(u,v) in enumerate(grid):
        j=grid.index((v,u))
        check(np.array_equal(probs[i].reshape(9,9).T,probs[j].reshape(9,9)),"mirror PMF")
        if i<=j:
            common=.5*(qn[i]+qn[j]);qn[i]=common;qn[j]=common
    T=-np.log(qn[:,2]);J=-np.log(probs)+lg
    # Direct pair enumeration of same/opposite counts, separate from core kernel.
    counts_same=np.array([[math.comb(a,2)+math.comb(8-a,2),
                           math.comb(b,2)+math.comb(8-b,2),
                           a*b+(8-a)*(8-b)] for a,b in cells])
    counts_diff=np.array([28,28,64])-counts_same
    P=[]
    for ref in refs:
        r=np.array(ref["independent_Hermite_moments"]["pair_magnitudes_AA_BB_AB"])
        P.append((120*math.log(4)-counts_same@np.log1p(-r)-counts_diff@np.log1p(r))/120)
    P=np.array(P)
    teacher=read(A/"TEACHER.json");mapped=np.array(teacher["mapped_counts_by_r"]);H=float(teacher["H_original"])
    risk=mapped@J.T-H;popT=T@(np.array(teacher["mean"])@ind)
    maxima=dict(reference_risk=0.,public_population=0.,public_finite=0.,selected_objective_gap=0.,
                selected_risk=0.,interval_replay=0.,fixed_interval_replay=0.)
    for row in rows(OUT/"REFERENCE_CURVES.csv"):
        ri=cfg["r_values"].index(int(row["r"]));ii=int(row["index"])
        maxima["reference_risk"]=max(maxima["reference_risk"],abs(float(row["joint_KL"])-risk[ri,ii]))
    fits=rows(OUT/"FINITE_FITS.csv");check(len(fits)==640,"finite fit count")
    lookup={(int(x["r"]),int(x["N"]),int(x["stream"]),x["objective"]):x for x in fits}
    finalrisks={};pops=[]
    for s in range(16):
        rec=read(OUT/"streams"/f"s{s:02}.json");path=ROOT/rec["array_path"]
        check(sha(path)==rec["array_sha256"],"array hash")
        z=np.load(path,allow_pickle=False);means=z["mean_pattern_loss"]
        check(means.shape==(361,2,81) and np.isfinite(means).all(),"mean shape/finite")
        ac=np.array(read(A/"streams"/f"s{s:02}.json")["training_counts_by_r"])
        expectedpop=np.stack([means@mapped[ri] for ri in range(4)])
        expectedfinite=np.stack([np.stack([means@(ac[ri,ni]/N) for ni,N in enumerate(cfg["train_N"])]) for ri in range(4)])
        maxima["public_population"]=max(maxima["public_population"],float(abs(expectedpop-np.array(rec["population_objectives"])).max()))
        maxima["public_finite"]=max(maxima["public_finite"],float(abs(expectedfinite-np.array(rec["finite_objectives"])).max()))
        pops.append(expectedpop)
        total=ac[0]@ind.astype(np.int64)
        for ri,r in enumerate(cfg["r_values"]):
            for ni,N in enumerate(cfg["train_N"]):
                empirical=ac[ri,ni]/N
                objectives={"J":J@empirical,"P":P@empirical,"T":T@(total[ni]/N),
                            "G10":expectedfinite[ri,ni,:,0],"G100":expectedfinite[ri,ni,:,1]}
                for objective,curve in objectives.items():
                    row=lookup[(r,N,s,objective)];ii=int(row["index"])
                    gap=float(curve[ii]-curve.min())
                    maxima["selected_objective_gap"]=max(maxima["selected_objective_gap"],abs(gap))
                    maxima["selected_risk"]=max(maxima["selected_risk"],abs(risk[ri,ii]-float(row["joint_KL"])))
                    finalrisks[(r,N,s,objective)]=float(risk[ri,ii])
                    if objective=="T":
                        mirror=grid.index(tuple(reversed(grid[ii])))
                        check(curve[ii]==curve[mirror],"T mirror likelihood exact")
                        check(grid[ii][0]<=grid[ii][1],"T did not choose smaller-u orientation")
        z.close()
    intervals=rows(OUT/"FINITE_RISK_INTERVALS.csv");check(len(intervals)==32,"interval count")
    critical=float(student_t.ppf(1-.05/(2*32),15))
    for row in intervals:
        r=int(row["r"]);N=int(row["N"]);left,right=row["comparison"].split("-")
        values=[finalrisks[(r,N,s,left)]-finalrisks[(r,N,s,right)] for s in range(16)]
        mean=math.fsum(values)/16;se=math.sqrt(math.fsum((x-mean)**2 for x in values)/15/16)
        for k,v in zip(["mean","SE","lower","upper"],[mean,se,mean-critical*se,mean+critical*se]):
            maxima["interval_replay"]=max(maxima["interval_replay"],abs(float(row[k])-v))
        check((row["precision_pass"]=="True")==(se<=.001),"finite precision")
    pops=np.array(pops);fixed=rows(OUT/"POPULATION_FIXED_COMPARISONS.csv");check(len(fixed)==8,"fixed comparisons")
    critical=float(student_t.ppf(1-.05/(2*8),15));truth=grid.index((6.,2.));witness=grid.index((3.,1.))
    for row in fixed:
        ri=cfg["r_values"].index(int(row["r"]));ki=cfg["K"].index(int(row["K"]))
        values=pops[:,ri,witness,ki]-pops[:,ri,truth,ki]
        mean=float(values.mean());se=float(values.std(ddof=1)/4)
        for k,v in zip(["mean","SE","lower","upper"],[mean,se,mean-critical*se,mean+critical*se]):
            maxima["fixed_interval_replay"]=max(maxima["fixed_interval_replay"],abs(float(row[k])-v))
    ties=read(OUT/"T_EQUIVALENCE_CLASSES.json");check(len(ties)==132,"T equivalence records")
    for record in ties:
        exact=set(record["exact_indices"])
        check(all(grid.index(tuple(reversed(grid[i]))) in exact for i in exact),"T class missing mirror")
        check(exact and min(exact)==min(x["index"] for x in record["exact_candidates"]),"T lexicographic tie")
    check(max(maxima.values())<2e-11,"independent replay precision")
    result=dict(time_utc=datetime.now(timezone.utc).isoformat(),passed=True,maxima=maxima,
                finite_fit_count=640,finite_interval_count=32,fixed_population_comparisons=8,T_equivalence_classes=132,
                array_hash_count=16,independent_grid_references=190,transpose_references=171,
                no_new_random_draws=True,no_new_reference_integrals=True,no_new_fits=True,
                audit_source_sha256=sha(Path(__file__)),results_freeze_sha256=sha(OUT/"RESULTS_FREEZE.json"))
    if write:
        with (OUT/"INDEPENDENT_AUDIT.json").open("x",encoding="utf-8",newline="\n") as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--write",action="store_true");args=parser.parse_args();main(args.write)
