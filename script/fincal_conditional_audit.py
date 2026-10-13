"""Independent audit of the explicitly posthoc conditional-target proxy.

Use fixed training pair moments, held-out pooled derivatives and the frozen
calibration only. No producer import, new random draw or Gaussian integration.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import os
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/finite_calibration"
OUT=BASE/"CONDITIONAL_TARGET"
PROTOCOL=OUT/"AUDIT_PROTOCOL.json"
DEST=OUT/"AUDIT.json"
TOL=1e-11
PTOL=2e-8


def read(path):return json.loads(Path(path).read_text("utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def fmean(values):return math.fsum(values)/len(values)


def horner(c,knots,point):
    interval=min(max(int(np.searchsorted(knots,point,side="right"))-1,0),len(knots)-2)
    offset=point-knots[interval]
    a,b,cc,d=c[:,interval]
    return float(((a*offset+b)*offset+cc)*offset+d)


def integral_from_left(c,knots,point):
    interval=min(max(int(np.searchsorted(knots,point,side="right"))-1,0),len(knots)-2)
    pieces=[]
    for j in range(interval+1):
        width=(point-knots[j]) if j==interval else (knots[j+1]-knots[j])
        a,b,cc,d=c[:,j]
        pieces.append(float(a*width**4/4+b*width**3/3+cc*width**2/2+d*width))
    return math.fsum(pieces)


def independent_minimum(knots,derivatives):
    curve=CubicSpline(knots,derivatives,bc_type="not-a-knot")
    c=np.asarray(curve.c)
    candidates=[float(knots[0]),float(knots[-1])]
    for j in range(len(knots)-1):
        coefficients=np.trim_zeros(c[:,j][::-1],"b")
        if len(coefficients)<=1:continue
        width=knots[j+1]-knots[j]
        for root in np.polynomial.polynomial.polyroots(coefficients):
            if abs(root.imag)<=1e-10 and -1e-12<=root.real<=width+1e-12:
                point=float(np.clip(knots[j]+root.real,knots[0],knots[-1]))
                if all(abs(point-old)>1e-10 for old in candidates):candidates.append(point)
    candidates.sort()
    origin=integral_from_left(c,knots,0.)
    values=[integral_from_left(c,knots,point)-origin for point in candidates]
    least=min(values)
    tied=[point for point,value in zip(candidates,values) if value<=least+1e-12]
    selected=min(tied,key=lambda point:(abs(point),point))
    return dict(rho=selected,derivative=horner(c,knots,selected),candidate_count=len(candidates),
                tied=tied,candidates=[dict(rho=x,potential=y) for x,y in zip(candidates,values)])


def fixed_reference_kl(t,r):
    value=(1+t)/2*math.log((1+t)/(1+r))+(1-t)/2*math.log((1-t)/(1-r))
    assert value>=-1e-13
    return max(0.,value)


def freeze():
    assert not PROTOCOL.exists() and not DEST.exists()
    paths=[OUT/"PROTOCOL.json",OUT/"RESULTS.json",BASE/"CAL/MODEL.json",
           BASE/"CAL/VALIDATION.json",BASE/"TRAIN/RESULTS.json",ROOT/"script/fincal_posthoc.py"]
    save_new(PROTOCOL,dict(
        schema="fincal-conditional-independent-audit-v1",source_sha256=sha(__file__),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        statistic_and_function_absolute_tolerance=TOL,point_absolute_tolerance=PTOL,
        polynomial_root_imaginary_tolerance=1e-10,root_merge_tolerance=1e-10,
        tie_tolerance=1e-12,tie_rule="smallest absolute rho, then signed rho",
        derivation="TRAIN exact-output t_N; pooled validation Uprime+t_N Vprime; finite penalty through stored cubic Horner values",
        optimizer="CubicSpline construction allowed; each segment independently solved by numpy.polynomial.polyroots; all roots/endpoints; explicit polynomial integral, not producer function",
        risk="Fixed MODEL PCHIP observation-moment table and independent Bernoulli log-ratio algebra; no new Gaussian integrals",
        audit_scope="192 assignments and data-determined unique (t_N,method) proxy optimizations; actual empirical-rho/MLE differences; all six summaries",
        actual_training_scope="TRAIN RESULTS is a frozen input; no training observations, bank arrays or empirical objective are regenerated",
        confidence_scope="Posthoc reuse of a four-validation-stream numerical proxy; neither exact B-infinity nor new independent confirmation; no causal percentages",
        preserve_failures=True,automatic_rerun=False,new_random_samples=0,new_Gaussian_integrals=0))
    print(json.dumps({"frozen":True,"protocol_sha256":sha(PROTOCOL)}),flush=True)


def run():
    assert not DEST.exists(),"Preserve existing audit"
    protocol=read(PROTOCOL);assert protocol["source_sha256"]==sha(__file__)
    for path,digest in protocol["inputs"].items():assert sha(ROOT/path)==digest,path
    started=time.perf_counter();comparisons=[];exacts=[];rows=[];summary_rows=[];cache={};failure=None

    def compare(name,actual,expected,tol=TOL):
        a,e=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
        assert a.shape==e.shape,(name,a.shape,e.shape)
        d=np.abs(a-e)
        comparisons.append(dict(name=name,count=int(d.size),tolerance=tol,
                                max_absolute_error=float(d.max()) if d.size else 0.,
                                failures=int(np.count_nonzero(d>tol))))
    def exact(name,actual,expected):exacts.append(dict(name=name,passed=bool(actual==expected)))

    try:
        producer_protocol,target=read(OUT/"PROTOCOL.json"),read(OUT/"RESULTS.json")
        model,validation,train=read(BASE/"CAL/MODEL.json"),read(BASE/"CAL/VALIDATION.json"),read(BASE/"TRAIN/RESULTS.json")
        source_hash=sha(ROOT/"script/fincal_posthoc.py")
        exact("producer source",producer_protocol["source_sha256"],source_hash)
        exact("result source",target["source_sha256"],source_hash)
        exact("result protocol",target["protocol_sha256"],sha(OUT/"PROTOCOL.json"))
        for path,digest in producer_protocol["inputs"].items():exact("producer input "+path,sha(ROOT/path),digest)
        exact("posthoc declaration",producer_protocol["posthoc"],True)
        exact("four validation streams",target["validation_streams"],4)
        exact("proxy not real training",target["conditional_expectation_proxy_not_real_empirical_training"],True)
        exact("no causal percentage",target["no_causal_percentage_claim"] and producer_protocol["no_causal_percentage_decomposition"],True)
        assert train["config"]["teachers"]==[0.,.3,.6] and train["config"]["sample_sizes"]==[1024,4096]
        assert len(train["records"])==16
        vcase=next(c for c in validation["cases"] if c["n"]==254)
        mcase=next(c for c in model["cases"] if c["n"]==254)
        knots=np.asarray(vcase["rho101"]);u=np.asarray(vcase["U_prime"]);v=np.asarray(vcase["V_prime"])
        c=np.asarray(mcase["gprime_spline_c"]);gx=np.asarray(mcase["gprime_spline_x"])
        penalty=np.array([horner(c,gx,float(point)) for point in knots])
        compare("finite gprime node identity",penalty,mcase["g_prime"])
        observation=PchipInterpolator(model["reference"]["rho801"],model["reference"]["r801"])
        expected_rows={(r["stream"],r["theta"],r["N"],r["method"]):r for r in target["rows"]}
        assert len(expected_rows)==192
        seen_t=set()
        for stream in train["records"]:
            outputs={(r["theta"],r["N"],r["method"]):r for r in stream["outputs"]}
            for theta in [0.,.3,.6]:
                teacher_moment=float(observation(theta))
                for N in [1024,4096]:
                    mle=outputs[(theta,N,"exact")];t=float(mle["empirical_pair"]);seen_t.add(t)
                    count=(Fraction.from_float(t)+1)*N/2
                    exact("actual pair-count admissibility "+str((stream["stream"],theta,N)),count.denominator==1 and 0<=count<=N,True)
                    desired=float(np.clip(t,float(observation(-.75)),float(observation(.75))))
                    compare("MLE moment equation "+str((stream["stream"],theta,N)),float(observation(mle["rho"])),desired)
                    for method in ["raw","finitecal"]:
                        ident=(stream["stream"],theta,N,method);saved=expected_rows[ident];actual=outputs[(theta,N,method)]
                        exact("shared actual t "+str(ident),actual["empirical_pair"],t)
                        ck=(t,method)
                        if ck not in cache:
                            values=u+t*v+(penalty if method=="finitecal" else 0.)
                            cache[ck]=independent_minimum(knots,values)
                        opt=cache[ck]
                        tag=str(ident)
                        compare("pair moment "+tag,t,saved["t_N"])
                        compare("proxy rho "+tag,opt["rho"],saved["proxy"]["rho"],PTOL)
                        compare("proxy derivative "+tag,opt["derivative"],saved["proxy"]["derivative"])
                        exact("proxy candidate count "+tag,opt["candidate_count"],saved["proxy"]["candidate_count"])
                        compare("proxy ties "+tag,opt["tied"],saved["proxy"]["tied"],PTOL)
                        proxy_risk=fixed_reference_kl(teacher_moment,float(observation(opt["rho"])))
                        gap_mle=opt["rho"]-mle["rho"];gap_emp=actual["rho"]-opt["rho"]
                        compare("proxy reference risk "+tag,proxy_risk,saved["proxy_reference_KL"])
                        exact("actual rho retained "+tag,actual["rho"],saved["actual_rho"])
                        exact("MLE rho retained "+tag,mle["rho"],saved["exact_MLE_rho"])
                        compare("proxy minus MLE "+tag,gap_mle,saved["proxy_minus_MLE"],PTOL)
                        compare("actual minus proxy "+tag,gap_emp,saved["actual_minus_proxy"],PTOL)
                        rows.append(dict(stream=stream["stream"],theta=theta,N=N,method=method,t_N=t,
                            proxy_rho=opt["rho"],proxy_reference_KL=proxy_risk,actual_rho=actual["rho"],exact_MLE_rho=mle["rho"],
                            proxy_minus_MLE=gap_mle,actual_minus_proxy=gap_emp))
        summary_map={(r["theta"],r["N"]):r for r in target["summary"]}
        for theta in [0.,.3,.6]:
            for N in [1024,4096]:
                subset=[r for r in rows if r["theta"]==theta and r["N"]==N and r["method"]=="finitecal"]
                assert len(subset)==16
                independent=dict(theta=theta,N=N,
                    proxy_mean_KL=fmean([r["proxy_reference_KL"] for r in subset]),
                    mean_abs_proxy_minus_MLE=fmean([abs(r["proxy_minus_MLE"]) for r in subset]),
                    mean_abs_actual_minus_proxy=fmean([abs(r["actual_minus_proxy"]) for r in subset]))
                for field in ["proxy_mean_KL","mean_abs_proxy_minus_MLE","mean_abs_actual_minus_proxy"]:
                    compare("summary "+str((theta,N,field)),independent[field],summary_map[(theta,N)][field])
                summary_rows.append(independent)
        exact("assignments",len(rows),target["assignments"])
        exact("unique optimizations",len(cache),target["unique_derivative_spline_optimizations"])
        exact("two methods per unique moment",len(cache),2*len(seen_t))
        exact("summary cells",len(summary_rows),6)
    except Exception:
        failure=traceback.format_exc()
    nf=sum(r["failures"] for r in comparisons);ef=sum(not r["passed"] for r in exacts)
    ledger=dict(new_random_samples=0,new_Gaussian_integrals=0,new_observed_data=0,new_training_fits=0,
                repeated_target_assignments=len(rows),repeated_unique_proxy_optimizations=len(cache),
                repeated_unique_empirical_pair_values=len(set(k[0] for k in cache)),summary_cells=len(summary_rows),
                validation_streams_reused=4,no_new_independent_confirmation=True)
    result=dict(status="passed" if failure is None and nf==0 and ef==0 else "failed",
        code_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),input_sha256=protocol["inputs"],
        elapsed_seconds=time.perf_counter()-started,numerical_comparisons=comparisons,exact_checks=exacts,
        numerical_failures=nf,exact_failures=ef,assignments=rows,summary=summary_rows,
        unique_proxy_optimizations=[dict(t_N=k[0],method=k[1],**value) for k,value in cache.items()],
        ledger=ledger,software_error=failure,
        scope="Numerical proxy from fixed four validation streams, conditional on actual observed pair moments; not exact B-infinity, empirical training or a causal percentage decomposition")
    save_new(DEST,result)
    print(json.dumps({"status":result["status"],"numerical_failures":nf,"exact_failures":ef,"ledger":ledger,"sha256":sha(DEST)}),flush=True)
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["freeze","run"])
    a=p.parse_args();freeze() if a.mode=="freeze" else run()
