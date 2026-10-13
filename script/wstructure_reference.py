"""Frozen numerical construction of variance-matched nuisance scale vectors.

check/schema do not call a Gaussian/model integrator. run requires a separate
root-authored REFERENCE/FREEZE.json and a source-bound REFERENCE_PLAN. No RNG.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import warnings

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/nuisance_structure/REFERENCE"
SOURCE=Path(__file__).resolve()
KNOWN=ROOT/"artifacts/finite_calibration/CAL/REFERENCE.json"
DEFAULTS=dict(ns=[62,254],a_active=1.,a_low=.5,hetero_bracket=[1.,2.],
    concentration_bracket_rule="[1, 2*sqrt(n)]",small_scale_rule="1/n",
    main_epsabs=2e-13,main_epsrel=2e-13,tight_epsabs=5e-14,tight_epsrel=5e-14,
    quad_limit=250,root_xtol=2e-13,root_rtol=1e-14,root_maxiter=100,
    max_unique_main_variance_evaluations=256,legendre_orders=[64,96],cutoff=12.,
    variance_check_atol=1e-13,variance_check_rtol=1e-11,match_relative_tolerance=1e-10)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def write_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def scale_vector(reference,family,n):
    """Pure read-only deployment interface; returns the frozen positive scales."""
    assert reference["qualification_passed"] is True,"Reference has not passed its fixed numerical qualifications"
    row=next(p for p in reference["profiles"] if p["family"]==family and p["n"]==n)
    values=np.asarray(row["scales"],dtype=np.float64)
    assert values.shape==(n,) and np.isfinite(values).all() and np.all(values>0)
    return values.copy()

def logcosh_stable(x):
    u=abs(float(x))
    if u<=.5:
        return math.log1p(2*math.sinh(u/2)**2)
    return u-math.log(2)+math.log1p(math.exp(-2*u))

def quad_variance(a,epsabs,epsrel,limit):
    """Positive-half quadrature of the mean and centered logcosh variance."""
    assert a>0 and math.isfinite(a)
    knots=sorted(set([0.,min(1.,1/a),1.,4.,12.]))+[math.inf]
    density=lambda x:math.sqrt(2/math.pi)*math.exp(-x*x/2)
    details=[];captured=[]
    def integrate(fn,name):
        values=[]
        for lo,hi in zip(knots[:-1],knots[1:]):
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always",IntegrationWarning)
                value,error=quad(fn,lo,hi,epsabs=epsabs/(len(knots)-1),epsrel=epsrel,limit=limit)
            captured.extend(str(w.message) for w in ws)
            details.append(dict(component=name,lower=lo,upper=None if math.isinf(hi) else hi,
                                value=value,reported_quad_error=error))
            values.append(value)
        return math.fsum(values)
    mean=integrate(lambda x:logcosh_stable(a*x/2)*density(x),"logcosh_mean")
    centered=integrate(lambda x:(logcosh_stable(a*x/2)-mean)**2*density(x),"centered_logcosh_variance")
    v=a*a/4+centered
    assert v>0 and math.isfinite(v)
    return dict(a=a,variance=v,mean_logcosh=mean,centered_logcosh_variance=centered,
                mean_logsigmoid=-math.log(2)-mean,quad_parts=details,warnings=captured,
                component_integrals=2,quad_subcalls=len(details))

def legendre_variance(a,order,cutoff):
    """Independent full plus/minus-logsigmoid moment expression on [0,L]."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    splits=sorted(set([0.,1.,2.,4.,8.,cutoff]+[t/a for t in (.25,.5,1.,2.,4.,8.) if 0<t/a<cutoff]))
    all_x=[];all_w=[]
    for lo,hi in zip(splits[:-1],splits[1:]):
        x=(lo+hi)/2+(hi-lo)*nodes/2
        all_x.append(x);all_w.append(weights*(hi-lo)/2*math.sqrt(2/math.pi)*np.exp(-x*x/2))
    x=np.concatenate(all_x);w=np.concatenate(all_w)
    plus=-np.logaddexp(0.,-a*x);minus=-np.logaddexp(0.,a*x)
    mean=math.fsum((w*(plus+minus)/2).tolist())
    # The omitted tail is tiny but is not silently made into exact mass one.
    variance=math.fsum((w*((plus-mean)**2+(minus-mean)**2)/2).tolist())
    return dict(a=a,order=order,cutoff=cutoff,variance=variance,mean_logsigmoid=mean,
        half_normal_mass=math.fsum(w.tolist()),node_evaluations=len(x),
        scope="Finite-cutoff, high-order numerical qualification; no strict quadrature remainder certificate.")

def check():
    xs=[0.,1e-12,1e-7,.01,.5,1.,16.,80.]
    values=[logcosh_stable(x) for x in xs]
    assert values[0]==0. and all(math.isfinite(v) and v>=0 for v in values)
    for x in xs:
        assert logcosh_stable(x)==logcosh_stable(-x)
        assert logcosh_stable(x)<=abs(x)+1e-15
    assert abs(logcosh_stable(.01)-math.log(math.cosh(.01)))<2e-16
    polynomial_errors=[]
    for order in DEFAULTS["legendre_orders"]:
        x,w=np.polynomial.legendre.leggauss(order)
        polynomial_errors += [abs(math.fsum(w.tolist())-2),abs(math.fsum((w*x*x).tolist())-2/3)]
    assert max(polynomial_errors)<2e-14
    result=dict(passed=True,source_sha256=sha(SOURCE),logcosh_test_values=dict(zip(map(str,xs),values)),
        polynomial_quadrature_weight_error=max(polynomial_errors),model_integrals_executed=0,
        Gaussian_draws=0,scale_roots_solved=0,
        scope="Stable elementary-function identities and quadrature weights for degree0/2 polynomials only; no v(a) evaluation.")
    write_new(OUT/"SELFTEST.json",result);print(json.dumps(result),flush=True)

def schema():
    print(json.dumps(dict(source_sha256=sha(SOURCE),config=DEFAULTS,expected_freeze_fields=[
        "source_sha256","plan_path","plan_sha256","known_reference_path","known_reference_sha256",
        "config","model_integrals_authorized","base_commit"],known_reference=rel(KNOWN)),indent=2),flush=True)

def run():
    freeze_path=OUT/"FREEZE.json";frozen=read(freeze_path)
    assert frozen["model_integrals_authorized"] is True
    assert frozen["source_sha256"]==sha(SOURCE)
    assert sha(ROOT/frozen["plan_path"])==frozen["plan_sha256"]
    known_path=ROOT/frozen["known_reference_path"]
    assert known_path.resolve()==KNOWN.resolve() and sha(known_path)==frozen["known_reference_sha256"]
    cfg=frozen["config"];assert cfg==DEFAULTS
    assert not (OUT/"EXECUTION.json").exists() and not (OUT/"MODEL.json").exists()
    test=read(OUT/"SELFTEST.json");assert test["passed"] and test["source_sha256"]==sha(SOURCE)
    write_new(OUT/"EXECUTION.json",dict(start_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE),freeze_sha256=sha(freeze_path),automatic_retry=False))
    started=time.perf_counter();cache={};solve_records=[]
    try:
        def main_v(a):
            a=float(a);key=a.hex()
            if key not in cache:
                assert len(cache)<cfg["max_unique_main_variance_evaluations"],"Fixed integration budget exhausted"
                cache[key]=quad_variance(a,cfg["main_epsabs"],cfg["main_epsrel"],cfg["quad_limit"])
            return cache[key]["variance"]
        v1=main_v(1.);vlo=main_v(cfg["a_low"])
        def solve(name,target,lo,hi):
            before=len(cache)
            lv,hv=main_v(lo),main_v(hi)
            assert lv<target<hv,(name,lv,target,hv)
            root,result=brentq(lambda x:main_v(x)-target,lo,hi,xtol=cfg["root_xtol"],
                              rtol=cfg["root_rtol"],maxiter=cfg["root_maxiter"],full_output=True,disp=True)
            assert result.converged
            solve_records.append(dict(name=name,target_variance=target,bracket=[lo,hi],scale=root,
                residual=main_v(root)-target,iterations=result.iterations,function_calls=result.function_calls,
                new_main_variance_evaluations=len(cache)-before))
            return root
        high=solve("boundedhetero_high",2*v1-vlo,*cfg["hetero_bracket"])
        profiles=[];validation_scales={1.,cfg["a_low"],high}
        for n in cfg["ns"]:
            small=1/n;vsmall=main_v(small)
            large=solve(f"concentrated_n{n}",n*v1-(n-1)*vsmall,1.,2*math.sqrt(n))
            validation_scales.update([small,large])
            for family,scales in [("diffuse",np.ones(n)),("boundedhetero",np.tile([cfg["a_low"],high],n//2)),
                                  ("concentrated",np.concatenate(([large],np.full(n-1,small))))]:
                assert len(scales)==n and np.all(scales>0)
                variances=[main_v(float(a)) for a in scales]
                total=math.fsum(variances)
                profiles.append(dict(n=n,family=family,scales=scales.tolist(),component_variances=variances,
                    total_variance=total,target_variance=n*v1,relative_variance_residual=(total-n*v1)/(n*v1),
                    maximum_variance_share=max(variances)/total,min_scale=float(min(scales)),max_scale=float(max(scales))))
        validation=[];checked={}
        for a in sorted(validation_scales):
            production=main_v(a)
            tight=quad_variance(a,cfg["tight_epsabs"],cfg["tight_epsrel"],cfg["quad_limit"])
            gl=[legendre_variance(a,order,cfg["cutoff"]) for order in cfg["legendre_orders"]]
            tolerance=cfg["variance_check_atol"]+cfg["variance_check_rtol"]*abs(production)
            errors=[abs(tight["variance"]-production)]+[abs(row["variance"]-production) for row in gl]
            passed=all(e<=tolerance for e in errors)
            validation.append(dict(a=a,main_variance=production,tight=tight,legendre=gl,
                comparison_errors=errors,tolerance=tolerance,passed=passed))
            checked[a.hex()]=gl[-1]["variance"]
        for profile in profiles:
            gl_total=math.fsum(checked[float(a).hex()] for a in profile["scales"])
            gl_target=profile["n"]*checked[float(1.).hex()]
            profile["independent_GL_total_variance"]=gl_total
            profile["independent_GL_relative_match_residual"]=(gl_total-gl_target)/gl_target
            profile["match_qualification_passed"]=bool(abs(profile["relative_variance_residual"])<=cfg["match_relative_tolerance"] and
                    abs(profile["independent_GL_relative_match_residual"])<=cfg["match_relative_tolerance"])
        known=read(known_path)
        main=next(row for row in known["orders"] if row["order"]==known["main_order"])
        oldv=float(known["inherited"]["var_logsigmoid"])
        oldv_error=abs(v1-oldv)
        oldv_pass=oldv_error<=cfg["variance_check_atol"]+cfg["variance_check_rtol"]*abs(v1)
        warning_count=sum(len(row["warnings"]) for row in cache.values())+sum(len(row["tight"]["warnings"]) for row in validation)
        qualification=all(row["passed"] for row in validation) and all(p["match_qualification_passed"] for p in profiles) and oldv_pass and warning_count==0
        output=dict(schema="wstructure-variance-reference-v1",source_sha256=sha(SOURCE),freeze_sha256=sha(freeze_path),
            known_reference_sha256=sha(known_path),base_commit=frozen["base_commit"],config=cfg,
            v1=v1,known_frozen_v1=oldv,known_v1_absolute_difference=oldv_error,known_v1_qualification_passed=oldv_pass,
            a_high=high,profiles=profiles,scale_solves=solve_records,main_variance_evaluations=list(cache.values()),
            validation=validation,warning_count=warning_count,qualification_passed=bool(qualification),
            known_active_reference=dict(a=1.,R=known["R"],rho801=known["rho801"],r801=main["r801"],H801=main["H801"],
                original_main_order=known["main_order"],original_r_qualification=known["r_qualification_pass"],
                scope="Copied frozen numerical active-model reference, no new r/H integration or strict remainder certificate"),
            accounting=dict(new_Gaussian_draws=0,new_observations=0,new_training_fits=0,constructed_scale_roots=3,
                main_variance_evaluations=len(cache),main_component_integrals=2*len(cache),
                main_quad_subcalls=sum(x["quad_subcalls"] for x in cache.values()),
                tight_variance_evaluations=len(validation),tight_component_integrals=2*len(validation),
                validation_GL_evaluations=len(validation)*len(cfg["legendre_orders"])),
            elapsed_seconds=time.perf_counter()-started,
            scope="Numerically variance-matched profiles; quadrature/root qualification is not an exact-equality or strict full-integral certificate. The concentrated profile deliberately does not retain a uniform positive nuisance-noise floor.")
        write_new(OUT/"MODEL.json",output)
        print(json.dumps(dict(completed=True,qualification_passed=bool(qualification),a_high=high,
            profiles=[{k:p[k] for k in ["n","family","min_scale","max_scale","relative_variance_residual","maximum_variance_share"]} for p in profiles],
            accounting=output["accounting"],elapsed_seconds=output["elapsed_seconds"])),flush=True)
    except BaseException as error:
        write_new(OUT/"ERROR.json",dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            completed_main_variance_evaluations=list(cache.values()),completed_scale_solves=solve_records,automatic_retry=False))
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["check","schema","run"])
    args=parser.parse_args();globals()[args.mode]()
