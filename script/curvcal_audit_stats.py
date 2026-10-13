"""Independent saved-output statistics/curvature audit; no producer imports.

Scope/source freeze precedes ANALYSIS. Independent scalar sums, PCHIP slopes,
Horner evaluation and forward Student CDF inversion reconstruct the contract.
This is not an underlying likelihood/path-gradient or RNG replay audit.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import betainc

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/calibration_precision"
OUT = BASE/"STATS_AUDIT"
DOC = ROOT/"docs/08_research_研究工作/校准精度与跨尺度学习_20261002/STATS_AUDIT.md"
SOURCE = Path(__file__).resolve()
CASES = [f"n{n}_K{K}" for n in (62, 254) for K in (64, 256)]
THETAS, BS = [0., .3, .6], [128, 512]
METHODS = ["raw", "marg", "asymquad", "fullfinite", "localquad", "exact"]


def read(p):
    return json.loads(Path(p).read_text("utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def write_new(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def avg(x):
    return math.fsum(map(float, x))/len(x)


def var(x):
    m = avg(x)
    return math.fsum((float(v)-m)**2 for v in x)/(len(x)-1)


def summary(x, critical=None):
    m = avg(x)
    se = math.sqrt(var(x)/len(x)) if len(x)>1 else None
    result = dict(mean=m, SE=se, count=len(x))
    if critical is not None:
        result.update(lower=m-critical*se, upper=m+critical*se)
    return result


def student_critical():
    # Invert a forward beta-CDF identity; does not use scipy.stats.t.ppf/stdtrit.
    def cdf(t):
        return 1-.5*float(betainc(15/2, .5, 15/(15+t*t)))
    lo, hi = 0., 16.
    for _ in range(100):
        mid = (lo+hi)/2
        if cdf(mid) < .995:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def pchip_coefficients(knots, values):
    """Explicit PCHIP endpoint and weighted-harmonic interior slopes."""
    x, y = list(map(float, knots)), list(map(float, values))
    h = [b-a for a,b in zip(x[:-1],x[1:])]
    delta = [(b-a)/width for a,b,width in zip(y[:-1],y[1:],h)]
    assert len(x)>=3 and min(h)>0
    def endpoint(h0,h1,d0,d1):
        m = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
        if m*d0<=0:
            return 0.
        if d0*d1<0 and abs(m)>3*abs(d0):
            return 3*d0
        return m
    slope = [endpoint(h[0],h[1],delta[0],delta[1])]
    for i in range(1,len(x)-1):
        a,b = delta[i-1],delta[i]
        if a*b<=0:
            slope.append(0.)
        else:
            w1,w2 = 2*h[i]+h[i-1],h[i]+2*h[i-1]
            slope.append((w1+w2)/(w1/a+w2/b))
    slope.append(endpoint(h[-1],h[-2],delta[-1],delta[-2]))
    c = [[],[],[],[]]
    for i,width in enumerate(h):
        c[0].append((slope[i]+slope[i+1]-2*delta[i])/(width*width))
        c[1].append((3*delta[i]-2*slope[i]-slope[i+1])/width)
        c[2].append(slope[i]); c[3].append(y[i])
    return c,x


def cubic(c,x,q):
    assert x[0]<=q<=x[-1]
    i=min(max(bisect_right(x,q)-1,0),len(x)-2)
    u=q-x[i]
    return ((c[0][i]*u+c[1][i])*u+c[2][i])*u+c[3][i]


def curvature_exact(c,x):
    largest=Fraction(0)
    for i in range(len(x)-1):
        a,b,d = 3*Fraction.from_float(c[0][i]),2*Fraction.from_float(c[1][i]),Fraction.from_float(c[2][i])
        width=Fraction.from_float(x[i+1])-Fraction.from_float(x[i])
        points=[Fraction(0),width]
        if a and 0 < -b/(2*a) < width:
            points.append(-b/(2*a))
        largest=max(largest,*(abs((a*u+b)*u+d) for u in points))
    return largest


def observed_kl(t,r):
    return max(0.,.5*((1+t)*math.log1p((t-r)/(1+r))+(1-t)*math.log1p((r-t)/(1-r))))


class Checks:
    def __init__(self):
        self.items=[]
    def flag(self,name,ok):
        self.items.append(dict(name=name,passed=bool(ok)))
    def close(self,name,got,want,atol=3e-12,rtol=3e-10):
        if want is None or got is None:
            self.flag(name,got is None and want is None);return
        a,b=float(got),float(want)
        finite=math.isfinite(a) and math.isfinite(b)
        err=abs(a-b) if finite else 1e300
        tol=atol+rtol*abs(b)
        self.items.append(dict(name=name,passed=bool(finite and err<=tol),max_absolute_error=err,tolerance=tol))
    def fields(self,name,got,want):
        for key,value in want.items():
            if isinstance(value,dict):
                self.fields(name+"/"+key,got[key],value)
            elif isinstance(value,(str,bool)):
                self.flag(name+"/"+key,got[key]==value)
            else:
                self.close(name+"/"+key,got[key],value)


def selftest():
    from scipy.interpolate import PchipInterpolator
    from scipy.stats import t
    x=[-.75,-.2,0.,.13,.75]
    errors=[]
    for y in ([0.,.3,.5,.6,1.],[1.,.2,.2,.6,-1.],[0.,0.,.1,.1,.4]):
        c,xx=pchip_coefficients(x,y)
        ref=PchipInterpolator(x,y)
        errors.extend(abs(cubic(c,xx,float(q))-float(ref(q))) for q in np.linspace(-.75,.75,71))
    critical=student_critical()
    assert max(errors)<3e-14 and abs(critical-float(t.ppf(.995,15)))<3e-10
    answer=dict(passed=True,source_sha256=sha(SOURCE),pchip_max_error=max(errors),
                forward_CDF_critical=critical,inverse_API_critical=float(t.ppf(.995,15)),
                new_random_draws=0,new_model_integrals=0)
    write_new(OUT/"SELFTEST.json",answer)
    print(json.dumps(answer),flush=True)


def freeze():
    test=read(OUT/"SELFTEST.json")
    assert test["passed"] and test["source_sha256"]==sha(SOURCE)
    assert not (BASE/"ANALYSIS.json").exists(), "Freeze must precede analysis output"
    tf=read(BASE/"TRAIN/FREEZE.json")
    paths=[SOURCE,OUT/"SELFTEST.json",BASE/"TRAIN/FREEZE.json",ROOT/tf["plan_path"],
           ROOT/tf["reference_path"],ROOT/tf["local_model_path"],ROOT/"script/curvcal_analysis.py"]
    write_new(OUT/"PROTOCOL.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE),inputs={rel(p):sha(p) for p in paths},analysis_existed_at_freeze=False,
        expected_outputs=2304,expected_cells=144,expected_primary_gates=5,expected_error_rows=1920,
        expected_gradient_rows=1920,expected_curvature_rows=768,curvature_summary_slots=48,
        methods=METHODS,cases=CASES,teachers=THETAS,bank_prefixes=BS,
        numerical_tolerance=dict(absolute=3e-12,relative=3e-10),risk_absolute_tolerance=2e-14,
        scope="Scalar fsum statistics, forward-CDF t critical, independent PCHIP/Horner penalties and b/rprime, all saved-gradient diagnostics, rational known error checks conditional on saved base curvature.",
        excluded="No RNG replay or underlying likelihood/path-gradient recomputation; no strict quadrature or full floating-point certificate; production author auditing independent analysis is not blind production audit.",
        new_random_draws=0,new_model_integrals=0,automatic_retry=False))
    print(json.dumps(dict(frozen=True,source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/"PROTOCOL.json"))),flush=True)


def run():
    protocol=read(OUT/"PROTOCOL.json")
    assert protocol["source_sha256"]==sha(SOURCE)
    for name,digest in protocol["inputs"].items():
        assert sha(ROOT/name)==digest,name
    tr=read(BASE/"TRAIN/RESULTS.json");an=read(BASE/"ANALYSIS.json")
    full=read(BASE/"CAL/FULL/MODEL.json");local=read(BASE/"CAL/LOCAL/MODEL.json")
    paths=[BASE/"TRAIN/RESULTS.json",BASE/"ANALYSIS.json"]
    paths += [ROOT/c["file"] for f in tr["records"] for c in f["cases"]]
    paths += [ROOT/f["data_file"] for f in tr["records"]]
    write_new(OUT/"EXECUTION.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/"PROTOCOL.json"),
        inputs={rel(p):sha(p) for p in paths},automatic_retry=False))
    started=time.perf_counter();checks=Checks()
    try:
        reference=full["reference"]
        rcurve=pchip_coefficients(reference["rho801"],reference["r801"])
        rpcurve=pchip_coefficients(reference["rho801"],reference["rp801"])
        models={}
        for c in full["cases"]:
            key=f"n{c['n']}_K{c['K']}"
            lc=next(x for x in local["cases"] if x["n"]==c["n"] and x["K"]==c["K"])
            gp=(c["gprime_spline_c"],c["gprime_spline_x"])
            co=dict(asymquad=reference["A"]*reference["d"]*c["w"],localquad=lc["coefficient"])
            dc={"raw":Fraction(0),"marg":Fraction(0),"fullfinite":curvature_exact(*gp)}
            for method,value in co.items():
                dc[method]=2*abs(Fraction.from_float(value))
                checks.close("coefficient/"+key+"/"+method,tr["coefficients"][key][method],value)
            models[key]=dict(b=pchip_coefficients(c["rho101"],c["b_slope101"]),gp=gp,co=co,penalty_curvature=dc)
        rows=[row for f in tr["records"] for c in f["cases"] for row in c["outputs"]]
        lookup={(r["stream"],r["case"],r["theta"],r["B"],r["method"]):r for r in rows}
        checks.flag("2304 unique output rows",len(rows)==len(lookup)==2304)
        table_lookup={(r["case"],r["theta"],r["B"],r["method"]):r for r in an["tables"]}
        checks.flag("144 unique cells",len(table_lookup)==len(an["tables"])==144)
        def values(c,th,B,m,key):
            return [lookup[s,c,th,B,m][key] for s in range(16)]
        def errors(c,th,B,m):
            return [(a-b)**2 for a,b in zip(values(c,th,B,m,"rho"),values(c,th,B,"exact","rho"))]
        for key,row in table_lookup.items():
            c,th,B,m=key;rho=values(c,th,B,m,"rho")
            want=dict(rho=summary(rho),reference_joint_KL=summary(values(c,th,B,m,"reference_joint_KL")),
                squared_discrepancy_to_data_MLE=summary(errors(c,th,B,m)),endpoint_count=sum(abs(q)==.75 for q in rho))
            checks.fields("cell/"+str(key),row,want)
        for i,row in enumerate(rows):
            t=cubic(*rcurve,row["theta"]);r=cubic(*rcurve,row["rho"])
            checks.close("risk/"+str(i),row["reference_joint_KL"],observed_kl(t,r),atol=2e-14,rtol=0.)
        critical=student_critical()
        checks.close("critical",an["critical_t"],critical,atol=3e-10,rtol=0.)
        checks.flag("family",an["family_size"]==5 and an["alpha"]==.05 and an["df"]==15 and len(an["primary_gates"])==5)
        differences=[]
        for c in CASES:
            differences.append([avg([lookup[s,c,th,512,"localquad"]["reference_joint_KL"]-
                lookup[s,c,th,512,"fullfinite"]["reference_joint_KL"] for th in THETAS]) for s in range(16)])
        differences.append([avg([(lookup[s,c,th,128,"fullfinite"]["rho"]-lookup[s,c,th,128,"exact"]["rho"])**2-
            (lookup[s,c,th,512,"fullfinite"]["rho"]-lookup[s,c,th,512,"exact"]["rho"])**2
            for c in CASES for th in THETAS]) for s in range(16)])
        for i,(g,v) in enumerate(zip(an["primary_gates"],differences)):
            expected=summary(v,critical);threshold=.0001 if i<4 else 0.
            direction="upper_below" if i<4 else "lower_above"
            expected.update(threshold=threshold,direction=direction,
                passed=expected["upper"]<threshold if i<4 else expected["lower"]>threshold)
            checks.fields("gate/"+str(i),g,expected)
            for j,(a,b) in enumerate(zip(g["stream_differences"],v)):
                checks.close(f"gate_difference/{i}/{j}",a,b)
        h=max(Fraction.from_float(b)-Fraction.from_float(a) for a,b in zip(reference["rho801"][:-1],reference["rho801"][1:]))
        eps=Fraction(700)*Fraction(25,28)**192
        error_values=[]
        for f in tr["records"]:
            for c in f["cases"]:
                for i,row in enumerate(c["outputs"]):
                    if row["method"]=="exact":continue
                    name=f"error/{f['stream']}/{c['case']}/{i}"
                    D=Fraction.from_float(row["curvature_D"])
                    base=Fraction.from_float(c["base_curvature_D"][str(row["B"])])
                    checks.flag(name+"/curvature",D>=base+models[c["case"]]["penalty_curvature"][row["method"]])
                    checks.flag(name+"/mesh",Fraction.from_float(row["mesh_regret_bound_real_arithmetic"])>=D*h*h/8)
                    er=Fraction.from_float(row["interpolation_uncentered_sup_error_upper"])
                    checks.flag(name+"/interpolation",er>=eps if f["evaluation_mode"]=="chebyshev193" else er==0)
                    bound=row["continuous_objective_regret_bound_real_arithmetic"]
                    checks.flag(name+"/sum",Fraction.from_float(bound)>=Fraction.from_float(row["mesh_regret_bound_real_arithmetic"])+2*er)
                    error_values.append(bound)
        checks.fields("known_error",an["known_error_qualification"],dict(count=1920,maximum=max(error_values),threshold=3e-5,pass_count=sum(v<=3e-5 for v in error_values)))
        observed_grad={(r["stream"],r["case"],r["theta"],r["B"],r["method"]):r for r in an["gradient_rows"]}
        observed_diag={(r["stream"],r["case"],r["theta"],r["B"],r["method"]):r for r in an["curvature_diagnostic_rows"]}
        checks.flag("1920 gradient rows",len(observed_grad)==len(an["gradient_rows"])==1920)
        checks.flag("768 diagnostic rows",len(observed_diag)==len(an["curvature_diagnostic_rows"])==768)
        gradients=[];diagnostics=[];boundary_count=0
        for flow in tr["records"]:
            with np.load(ROOT/flow["data_file"]) as data:
                boundary_count+=sum(bool(v) for v in data["gradient_MLE_boundary"])
                canonical_q=data["gradient_rho"].copy()
            for case in flow["cases"]:
                c=case["case"];model=models[c]
                with np.load(ROOT/case["file"]) as data:
                    for ti,th in enumerate(THETAS):
                        q=float(data["gradient_rho"][ti]);boundary=bool(data["gradient_MLE_boundary"][ti])
                        checks.close(f"same_q/{flow['stream']}/{c}/{th}",q,canonical_q[ti],atol=0,rtol=0)
                        kap=cubic(*model["b"],q)*cubic(*rpcurve,q)
                        valid=math.isfinite(kap) and kap>0;gp=cubic(*model["gp"],q)
                        for B in BS:
                            for method in METHODS[:-1]:
                                sample=data["marg_path_gradient" if method=="marg" else "raw_path_gradient"][:B,ti]
                                penalty=0. if method in ("raw","marg") else gp if method=="fullfinite" else 2*model["co"][method]*q
                                mu=avg(sample)+penalty;v=var(sample);ub=mu*mu-v/B
                                key=(flow["stream"],c,th,B,method)
                                gr=dict(stream=flow["stream"],case=c,theta=th,B=B,method=method,q=q,boundary=boundary,
                                    gradient_mean=mu,sample_variance=v,variance_over_B=v/B,squared_mean=mu*mu,unbiased_squared_mean=ub)
                                checks.fields("gradient/"+str(key),observed_grad[key],gr);gradients.append(gr)
                                if method in ("fullfinite","localquad"):
                                    shift=lookup[key]["rho"]-q;usable=valid and not boundary
                                    pred=-mu/kap if usable else None
                                    dr=dict(**gr,kappa_reference=kap if math.isfinite(kap) else None,kappa_positive_finite=valid,
                                        interior_diagnostic_usable=usable,actual_shift=shift,actual_squared_shift=shift*shift,
                                        linearized_shift=pred,linearization_residual=shift-pred if usable else None,
                                        normalized_noise=v/(B*kap*kap) if usable else None,
                                        normalized_squared_mean=mu*mu/(kap*kap) if usable else None,
                                        unbiased_normalized_mean_square=ub/(kap*kap) if usable else None,
                                        local_minus_full_penalty_gradient=2*model["co"]["localquad"]*q-gp)
                                    checks.fields("diagnostic/"+str(key),observed_diag[key],dr);diagnostics.append(dr)
        summary_lookup={(r["case"],r["theta"],r["B"],r["method"]):r for r in an["curvature_summaries"]}
        rebuilt_summaries=[]
        for c in CASES:
            for th in THETAS:
                for B in BS:
                    for method in ("fullfinite","localquad"):
                        selected=[r for r in diagnostics if (r["case"],r["theta"],r["B"],r["method"])==(c,th,B,method) and r["interior_diagnostic_usable"]]
                        if not selected:continue
                        keys=["kappa_reference","actual_squared_shift","normalized_noise","normalized_squared_mean","unbiased_normalized_mean_square","local_minus_full_penalty_gradient"]
                        ss=dict(case=c,theta=th,B=B,method=method,interior_count=len(selected),
                            **{k:summary([r[k] for r in selected]) for k in keys})
                        ss["mean_squared_linearization_residual"]=avg([r["linearization_residual"]**2 for r in selected])
                        ss["actual_MSE_over_normalized_noise"]=ss["actual_squared_shift"]["mean"]/ss["normalized_noise"]["mean"]
                        checks.fields("summary/"+str((c,th,B,method)),summary_lookup[c,th,B,method],ss)
                        rebuilt_summaries.append(ss)
        checks.flag("summary_count",len(rebuilt_summaries)==len(summary_lookup)==len(an["curvature_summaries"]))
        checks.flag("boundary_count",an["boundary_unique_MLE"]==boundary_count)
        negative=sum(r["unbiased_squared_mean"]<0 for r in gradients)
        checks.flag("negative estimates retained",an["negative_unbiased_squared_gradient_count"]==negative)
        checks.flag("metadata outputs",an["output_count"]==2304 and an["new_random_samples"]==0)
        result=dict(passed=all(r["passed"] for r in checks.items),source_sha256=sha(SOURCE),
            protocol_sha256=sha(OUT/"PROTOCOL.json"),execution_sha256=sha(OUT/"EXECUTION.json"),
            checks=checks.items,check_count=len(checks.items),failure_count=sum(not r["passed"] for r in checks.items),
            primary_gates=[g["passed"] for g in an["primary_gates"]],
            verified_counts=dict(outputs=2304,cells=144,primary_gates=5,known_error_rows=len(error_values),gradient_rows=len(gradients),curvature_rows=len(diagnostics),curvature_summaries=len(rebuilt_summaries)),
            boundary_unique_MLE=boundary_count,negative_unbiased_squared_gradient_count=negative,
            independent_forward_CDF_critical=critical,elapsed_seconds=time.perf_counter()-started,
            new_random_draws=0,new_model_integrals=0,scope=protocol["scope"],excluded=protocol["excluded"])
        write_new(OUT/"RESULTS.json",result)
        print(json.dumps({k:v for k,v in result.items() if k!="checks"}),flush=True)
        assert result["passed"],"independent saved-statistics audit failed; retain receipt"
    except BaseException as error:
        if not (OUT/"ERROR.json").exists():
            write_new(OUT/"ERROR.json",dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def report():
    p,r=read(OUT/"PROTOCOL.json"),read(OUT/"RESULTS.json")
    assert r["source_sha256"]==sha(SOURCE) and r["protocol_sha256"]==sha(OUT/"PROTOCOL.json")
    assert not DOC.exists()
    text="\n".join(["# 保存统计与曲率诊断的独立算术审计","",
        "本审计在ANALYSIS产生前冻结源码与范围，未导入curvcal_analysis或生产helper。由TRAIN实现代理独立审根分析算术，不是生产盲审。没有新增科学随机样本或模型积分。","",
        "以math.fsum独立重建144格与五项预定主门；用前向Student CDF二分求t_.995,15。自行实现PCHIP斜率和Horner评价新FULL的b、已知r′和g′，从保存的真实路径梯度重建全部1920梯度、768曲率诊断及所有可用摘要。负无偏估计与边界点不删除。","",
        "1920已知误差项使用Fraction重验惩罚曲率、网格和插值之和，条件于生产保存的原始曲率上界；原始active输入及该上界的底层核验归TRAIN审计。此处不重放RNG或底层likelihood/path-gradient，不是严格积分或浮点全域证书。","",
        f"结论：passed={r['passed']}，{r['check_count']}项检查，{r['failure_count']}失败；耗时{r['elapsed_seconds']:.3f}秒。边界MLE {r['boundary_unique_MLE']}，负无偏梯度均值平方 {r['negative_unbiased_squared_gradient_count']}。","",
        "五门是四case分别teacher合并的local/full非劣，加一项全部case/teacher合并的MSE差。审计只核实既定开发近似统计，不提高其置信等级，也不覆盖重新校准变异。","",
        "```json",json.dumps(r["verified_counts"],ensure_ascii=False,indent=2),"```","",
        f"PROTOCOL SHA256 `{sha(OUT/'PROTOCOL.json')}`；RESULTS SHA256 `{sha(OUT/'RESULTS.json')}`。",""])
    DOC.write_text(text,encoding="utf-8",newline="\n")
    print(json.dumps(dict(report=rel(DOC),sha256=sha(DOC))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["selftest","freeze","run","report"])
    mode=parser.parse_args().mode
    globals()[mode]()
