"""Independent four-case LOCAL/FULL calibration audit; no producer imports.

Freeze scope/source before replay. Rebuild every saved reduction and spline,
replay every Gaussian input, and independently score only first64 banks/stream.
No independent scientific sample or model integral is added by this audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PRODUCER = ROOT / "script/curvcal_calibrate.py"
BASE = ROOT / "artifacts/calibration_precision/CAL"
OUT = ROOT / "artifacts/calibration_precision/CAL_AUDIT"
REFERENCE = ROOT / "artifacts/finite_calibration/CAL/REFERENCE.json"
PROTOCOL = OUT / "PROTOCOL.json"
DOC = ROOT / "docs/08_research_研究工作/校准精度与跨尺度学习_20261002/CAL_AUDIT.md"
CASES = [(62,64),(62,256),(254,64),(254,256)]
SIGNS = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
LOCAL_RHO = np.array([-.12,-.06,-.03,.03,.06,.12])
H = np.array([.03,.06,.12])
SIZES = {"LOCAL":(8,1024,202610022000), "FULL":(16,2048,202610022100)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def freeze():
    assert not PROTOCOL.exists(), "Audit already frozen"
    producer_protocol = read(BASE/"PROTOCOL.json")
    paths = [PRODUCER,BASE/"PROTOCOL.json"]+[ROOT/x for x in producer_protocol["inputs"]]
    write_new(PROTOCOL, dict(schema="curvcal-independent-cal-audit-v1", source_sha256=sha(SOURCE),
        created_utc=datetime.now(timezone.utc).isoformat(), inputs={rel(x):sha(x) for x in paths},
        models_existed_at_freeze={pool:(BASE/pool/"MODEL.json").exists() for pool in SIZES},
        scope="Both pools full prior RNG; first64 bank per stream all4case/allrho/all4mode direct-probability Price; every saved mode mean and LOCAL G/c/SE/gate; FULL U/V/gprime/b and all stream/pooled SE, independent B-spline reconstruction, antiderivative values and exact stored-cubic curvature",
        Price_absolute_tolerance=5e-13, arithmetic_tolerance=2e-12, scaled_spline_coefficient_tolerance=2e-11,
        new_independent_scientific_samples=0, new_model_integrals=0, automatic_retry=False))
    print(json.dumps(dict(frozen=True, source_sha256=sha(SOURCE), protocol_sha256=sha(PROTOCOL))), flush=True)


class Checks:
    def __init__(self):
        self.rows=[]

    def flag(self,name,test):
        self.rows.append(dict(name=name,passed=bool(test)))

    def close(self,name,a,b,tol):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        valid=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error=float(np.max(abs(a-b))) if valid and a.size else (0. if valid else 1e300)
        self.rows.append(dict(name=name,passed=bool(valid and error<=tol),max_absolute_error=error,tolerance=tol))
        return error


def mean_axis0(values):
    values=np.asarray(values,dtype=float)
    flat=values.reshape(len(values),-1)
    means=[math.fsum(map(float,flat[:,j]))/len(values) for j in range(flat.shape[1])]
    return np.array(means).reshape(values.shape[1:])


def se_axis0(values):
    values=np.asarray(values,dtype=float)
    mu=mean_axis0(values).ravel();flat=values.reshape(len(values),-1);n=len(values)
    return np.array([math.sqrt(math.fsum((float(x)-mu[j])**2 for x in flat[:,j])/(n*(n-1)))
                     for j in range(flat.shape[1])]).reshape(values.shape[1:])


def direct_price(z,rhos):
    """Probability-domain expression; full K/n prefix selected independently."""
    result=np.empty((len(z),4,len(rhos),4))
    for ci,(n,k) in enumerate(CASES):
        zz=z[:,:k,:n+2]
        noise=np.prod(expit(zz[:,:,2:]),axis=2)
        assert np.isfinite(noise).all() and np.all(noise>0)
        for ri,rho in enumerate(rhos):
            x2=rho*zz[:,:,0]+math.sqrt(1-rho*rho)*zz[:,:,1]
            for si,(s1,s2) in enumerate(SIGNS):
                p1,p2=expit(s1*zz[:,:,0]),expit(s2*x2)
                likelihood=noise*p1*p2
                pi=likelihood/likelihood.sum(axis=1,keepdims=True)
                complement=(1-p1)*(1-p2)
                result[:,ci,ri,si]=-int(s1*s2)*(np.sum(pi*complement,axis=1)-np.sum(pi*pi*complement,axis=1))
    assert np.isfinite(result).all()
    return result


def projections(modes,rr):
    u=(modes[...,0]+modes[...,1]+modes[...,2]+modes[...,3])/4
    v=(modes[...,0]-modes[...,1]-modes[...,2]+modes[...,3])/4
    us=(u-u[...,::-1])/2;vs=(v+v[...,::-1])/2
    return dict(U_prime_unsym=u,V_prime_unsym=v,U_prime=us,V_prime=vs,b=-vs,g_prime=-us-rr*vs)


def curvature_fraction(coeff,knots):
    best=Fraction(0)
    for j in range(len(knots)-1):
        qa=3*Fraction.from_float(float(coeff[0,j]));qb=2*Fraction.from_float(float(coeff[1,j]));qc=Fraction.from_float(float(coeff[2,j]))
        width=Fraction.from_float(float(knots[j+1]))-Fraction.from_float(float(knots[j]))
        points=[Fraction(0),width]
        if qa and 0 < -qb/(2*qa) < width:
            points.append(-qb/(2*qa))
        best=max(best,*(abs((qa*x+qb)*x+qc) for x in points))
    return best


def audit():
    scope=read(PROTOCOL)
    assert scope["source_sha256"]==sha(SOURCE),"Audit source changed"
    for name,digest in scope["inputs"].items():
        assert sha(ROOT/name)==digest,f"Audit input changed: {name}"
    assert not (OUT/"EXECUTION.json").exists(),"No automatic retry"
    assert all((BASE/pool/"MODEL.json").is_file() for pool in SIZES),"Wait for both production models"
    paths=[]
    for pool,(streams,_,_) in SIZES.items():
        paths += [BASE/pool/name for name in ("MODEL.json","STREAMS.json","EXECUTION.json")]
        paths += [BASE/pool/f"stream_{i:02}.{ext}" for i in range(streams) for ext in ("json","npz")]
    write_new(OUT/"INPUT_RECEIPT.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
              protocol_sha256=sha(PROTOCOL),inputs={rel(x):sha(x) for x in paths}))
    write_new(OUT/"EXECUTION.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
              source_sha256=sha(SOURCE),protocol_sha256=sha(PROTOCOL),input_receipt_sha256=sha(OUT/"INPUT_RECEIPT.json")))
    checks=Checks();start=time.perf_counter();receipts=[];rebuilt={}
    try:
        p=read(BASE/"PROTOCOL.json");ref=read(REFERENCE)
        checks.flag("producer/source",p["source_sha256"]==sha(PRODUCER))
        for name,digest in p["inputs"].items():checks.flag("producer/input/"+name,sha(ROOT/name)==digest)
        checks.flag("producer/basic_model",p["a"]==1. and p["R"]==.75 and p["block"]==64 and p["max_prior_shape_per_block"]==[64,256,256])
        checks.flag("reference/qualification",ref["r_qualification_pass"] and ref["reference_qualification_pass"])
        rho801=np.array(ref["rho801"]);main=next(row for row in ref["orders"] if row["order"]==ref["main_order"])
        r801=np.array(main["r801"])
        rhos={"LOCAL":LOCAL_RHO,"FULL":rho801[::8]}
        ris=[int(np.argmin(abs(rho801-r))) for r in LOCAL_RHO]
        rs={"LOCAL":r801[ris],"FULL":r801[::8]}
        checks.close("reference/local_nodes",rho801[ris],LOCAL_RHO,1e-14)
        Js={case["K"]:case["J"] for case in ref["cases"]}
        expected_cases=[dict(n=n,K=k,J=Js[k],w=Js[k]/math.sqrt(n*ref["inherited"]["var_logsigmoid"])) for n,k in CASES]
        checks.flag("producer/cases",p["cases"]==expected_cases)
        checks.close("producer/local_rho",p["local_rho"],LOCAL_RHO,0.)
        checks.close("producer/full_rho",p["full_rho"],rho801[::8],0.)
        for pool,(streams,banks,seed) in SIZES.items():
            folder=BASE/pool;results=read(folder/"STREAMS.json");model=read(folder/"MODEL.json");execution=read(folder/"EXECUTION.json")
            checks.flag(pool+"/pool_size",p["pool_sizes"][pool]==dict(streams=streams,banks_per_stream=banks) and p["seeds"][pool]==seed)
            checks.flag(pool+"/model_parameters",model["a"]==1. and model["R"]==.75 and len(model["cases"])==4)
            for name,obj in (("stream_results",results),("model",model),("execution",execution)):
                checks.flag(pool+"/"+name+"/binding",obj["source_sha256"]==sha(PRODUCER)
                            and obj["protocol_sha256"]==sha(BASE/"PROTOCOL.json") and obj["reference_sha256"]==sha(REFERENCE))
            checks.flag(pool+"/records",len(results["records"])==streams)
            mode_means=[];local_means=[]
            for si in range(streams):
                meta=read(folder/f"stream_{si:02}.json");file=folder/f"stream_{si:02}.npz"
                checks.flag(f"{pool}/{si}/record",meta==results["records"][si] and meta["stream"]==si
                    and meta["seed"]==seed+si and meta["banks"]==banks and meta["file"]==rel(file) and meta["sha256"]==sha(file))
                with np.load(file,allow_pickle=False) as zsaved:
                    D=zsaved["D_modes"].copy()
                    for field,expected in (("rho",rhos[pool]),("r",rs[pool]),("cases",CASES),("signs",SIGNS)):
                        checks.close(f"{pool}/{si}/array_{field}",zsaved[field],expected,0.)
                    if pool=="LOCAL":
                        saved_G,saved_c=zsaved["G_bank"].copy(),zsaved["coefficients"].copy()
                        checks.close(f"{pool}/{si}/h",zsaved["h"],H,0.)
                checks.flag(f"{pool}/{si}/shape_finite",D.shape==(banks,4,len(rhos[pool]),4) and np.isfinite(D).all())
                checks.flag(f"{pool}/{si}/Price_sign",np.all(D*np.prod(SIGNS,axis=1)<=0.))
                generator=np.random.Generator(np.random.PCG64(seed+si));hasher=hashlib.sha256();first=None;error=None
                for offset in range(0,banks,64):
                    z=generator.standard_normal((64,256,256));raw=z.tobytes();hasher.update(raw)
                    if offset==0:
                        first=hashlib.sha256(raw).hexdigest()
                        error=checks.close(f"{pool}/{si}/independent_Price_first64",direct_price(z,rhos[pool]),D[:64],scope["Price_absolute_tolerance"])
                checks.flag(f"{pool}/{si}/full_RNG",hasher.hexdigest()==meta["prior_sha256"])
                checks.flag(f"{pool}/{si}/first_RNG",first==meta["first_block_prior_sha256"])
                mm=mean_axis0(D);mode_means.append(mm)
                checks.close(f"{pool}/{si}/all_mode_means",mm,meta["mode_means"],scope["arithmetic_tolerance"])
                if pool=="LOCAL":
                    U=(D[...,0]+D[...,1]+D[...,2]+D[...,3])/4
                    V=(D[...,0]-D[...,1]-D[...,2]+D[...,3])/4
                    G=-U-V*rs[pool]
                    cs=np.stack(((G[...,3]-G[...,2])/.12,(G[...,4]-G[...,1])/.24,(G[...,5]-G[...,0])/.48),axis=-1)
                    checks.close(f"{pool}/{si}/all_G",G,saved_G,scope["arithmetic_tolerance"])
                    checks.close(f"{pool}/{si}/all_c",cs,saved_c,scope["arithmetic_tolerance"])
                    cm=mean_axis0(cs);local_means.append(cm)
                    checks.close(f"{pool}/{si}/mean_c",cm,meta["local_coefficient_means"],scope["arithmetic_tolerance"])
                receipts.append(dict(pool=pool,stream=si,replayed_banks=banks,Price_banks=64,Price_entries=64*4*len(rhos[pool])*4,
                    Price_max_absolute_error=error,entire_prior_sha256=hasher.hexdigest(),first_prior_sha256=first))
                print(json.dumps(dict(audited_pool=pool,stream=si,Price_max_error=error)),flush=True)
            ledger=dict(banks=streams*banks,unique_max_prior_vectors=streams*banks*256,Gaussian_scalars=streams*banks*256*256,
                        bank_case_rho_mode_evaluations=streams*banks*4*len(rhos[pool])*4,new_observations=0,new_data_fits=0,new_model_integrals=0)
            for field,expected in ledger.items():checks.flag(pool+"/ledger/"+field,results["ledger"][field]==expected)
            if pool=="LOCAL":
                local_means=np.array(local_means);mu,se=mean_axis0(local_means),se_axis0(local_means)
                checks.flag("LOCAL/model_contract",model["teacher_specific_tuning"] is False and model["streams"]==8
                    and model["pool_results_sha256"]==sha(folder/"STREAMS.json"))
                gates=[]
                for ci,(n,k) in enumerate(CASES):
                    c=model["cases"][ci]
                    checks.flag(f"LOCAL/case{ci}/identity",c["n"]==n and c["K"]==k and c["h_main"]==.06 and c["coefficient_not_clipped_or_selected"])
                    checks.close(f"LOCAL/case{ci}/h_values",c["h_values"],H,0.)
                    for key,actual in (("pooled_c_h",mu[ci]),("stream_SE_c_h",se[ci]),("per_stream_c_h",local_means[:,ci]),
                                       ("coefficient",mu[ci,1]),("coefficient_SE",se[ci,1])):
                        checks.close(f"LOCAL/case{ci}/"+key,actual,c[key],scope["arithmetic_tolerance"])
                    vg=[(se[ci,1],2e-4,"main_stream_SE"),(abs(mu[ci,1]-mu[ci,0]),5e-5,"resolution_c06_c03")]
                    for value,threshold,name in vg:
                        gate=c["qualification"][name]
                        checks.close(f"LOCAL/case{ci}/{name}/value",value,gate["value"],scope["arithmetic_tolerance"])
                        checks.flag(f"LOCAL/case{ci}/{name}/pass",gate["threshold"]==threshold and gate["passed"]==bool(value<=threshold))
                    passed=all(v<=t for v,t,_ in vg);gates.append(passed)
                    checks.flag(f"LOCAL/case{ci}/all_pass",c["qualification_all_passed"]==passed)
                checks.flag("LOCAL/all_qualification",model["all_qualification_passed"]==all(gates))
                rebuilt[pool]=dict(coefficients=mu[:,1].tolist(),coefficient_SEs=se[:,1].tolist(),case_gates=[bool(x) for x in gates])
            else:
                projected=projections(np.array(mode_means),rs[pool]);sums={key:(mean_axis0(x),se_axis0(x)) for key,x in projected.items()}
                checks.flag("FULL/model_contract",model["training_sha256"]==sha(folder/"STREAMS.json") and model["training_streams"]==16
                    and model["teacher_free"] and model["shared_across_all_teachers"] and model["independent_of_LOCAL_pool"]
                    and model["training_qualification_is_not_uniform_confidence_coverage"])
                expected_reference={"rho801":rho801.tolist(),"r801":main["r801"],"rp801":main["rp801"],"H801":main["H801"],**ref["inherited"]}
                for key,value in expected_reference.items():checks.close("FULL/reference/"+key,model["reference"][key],value,0.)
                checks.flag("FULL/reference/hash",model["reference"]["known_reference_sha256"]==sha(REFERENCE))
                gpass,bpass=0,0;coefficient_errors=[]
                for ci,case in enumerate(expected_cases):
                    c=model["cases"][ci];grid=rhos[pool]
                    for key,value in case.items():checks.close(f"FULL/case{ci}/identity/"+key,c[key],value,0.)
                    checks.close(f"FULL/case{ci}/rho",c["rho101"],grid,0.)
                    for key,(mu,se) in sums.items():
                        checks.close(f"FULL/case{ci}/{key}/mean",mu[ci],c[key],scope["arithmetic_tolerance"])
                        checks.close(f"FULL/case{ci}/{key}/SE",se[ci],c["calibration_SE"][key],scope["arithmetic_tolerance"])
                        checks.close(f"FULL/case{ci}/{key}/streams",projected[key][:,ci],c["per_stream"][key],scope["arithmetic_tolerance"])
                    for key,actual in (("b_SE",sums["b"][1][ci]),("b_slope101",sums["b"][0][ci]),
                        ("b_slope_SE101",sums["b"][1][ci]),("stream_gprime101",projected["g_prime"][:,ci]),("stream_b101",projected["b"][:,ci])):
                        checks.close(f"FULL/case{ci}/"+key,actual,c[key],scope["arithmetic_tolerance"])
                    # Independent not-a-knot B-spline system, rather than the
                    # producer's CubicSpline construction in power basis.
                    spline=make_interp_spline(grid,sums["g_prime"][0][ci],k=3,bc_type="not-a-knot")
                    coeff=np.asarray(c["gprime_spline_c"]);knots=np.asarray(c["gprime_spline_x"])
                    checks.close(f"FULL/case{ci}/spline_knots",knots,grid,0.)
                    independent_coeff=np.stack((spline.derivative(3)(grid[:-1])/6,
                        spline.derivative(2)(grid[:-1])/2,spline.derivative()(grid[:-1]),spline(grid[:-1])))
                    scales=np.diff(grid)[None,:]**np.arange(3,-1,-1)[:,None]
                    coefficient_errors.append(checks.close(f"FULL/case{ci}/scaled_spline_coefficients",independent_coeff*scales,coeff*scales,
                                                          scope["scaled_spline_coefficient_tolerance"]))
                    anti=spline.antiderivative();gv=anti(grid)-anti(0.)
                    checks.close(f"FULL/case{ci}/antiderivative_values",gv,c["g_values"],scope["arithmetic_tolerance"])
                    # Origin shift depends on basis conventions; validate its
                    # role using direct exact-power segment integration.
                    segment=np.diff(knots)
                    areas=np.sum(coeff*segment[None,:]**np.arange(4,0,-1)[:,None]/np.arange(4,0,-1)[:,None],axis=0)
                    cumulative=np.concatenate(([0.],np.cumsum(areas)))
                    checks.close(f"FULL/case{ci}/power_antiderivative_origin",cumulative[len(grid)//2],c["g_antiderivative_origin_shift"],scope["arithmetic_tolerance"])
                    checks.close(f"FULL/case{ci}/power_integral_values",cumulative-cumulative[len(grid)//2],c["g_values"],scope["arithmetic_tolerance"])
                    stream_g=[]
                    for si in range(16):
                        stream_spline=make_interp_spline(grid,projected["g_prime"][si,ci],k=3,bc_type="not-a-knot").antiderivative()
                        stream_g.append(stream_spline(grid)-stream_spline(0.))
                    stream_g=np.array(stream_g)
                    checks.close(f"FULL/case{ci}/all_stream_g",stream_g,c["per_stream_g_values"],scope["arithmetic_tolerance"])
                    checks.close(f"FULL/case{ci}/g_values_SE",se_axis0(stream_g),c["g_values_SE"],scope["arithmetic_tolerance"])
                    checks.close(f"FULL/case{ci}/linear_mean_g",mean_axis0(stream_g),gv,scope["arithmetic_tolerance"])
                    checks.close(f"FULL/case{ci}/aggregation_receipt",float(np.max(abs(mean_axis0(np.asarray(c["per_stream_g_values"]))-np.asarray(c["g_values"])))),
                                 c["linear_aggregation_check_max"],scope["arithmetic_tolerance"])
                    exact=curvature_fraction(coeff,knots);receipt=c["g_second_sup_receipt"]
                    checks.flag(f"FULL/case{ci}/curvature_fraction",exact==Fraction(int(receipt["exact_numerator"]),int(receipt["exact_denominator"])))
                    checks.flag(f"FULL/case{ci}/curvature_outward",Fraction.from_float(c["g_second_sup"])>=exact and c["g_second_sup"]==receipt["upper_float"])
                    gflags=sums["g_prime"][1][ci]<=2e-4;bflags=sums["b"][0][ci]>0
                    checks.close(f"FULL/case{ci}/gprime_SE_flags",gflags,c["gprime_SE_pass_by_node"],0.)
                    checks.close(f"FULL/case{ci}/b_positive_flags",bflags,c["b_positive_by_node"],0.)
                    checks.flag(f"FULL/case{ci}/qualification_counts",c["gprime_SE_passed"]==int(gflags.sum()) and c["b_positive_passed"]==int(bflags.sum()))
                    gpass+=int(gflags.sum());bpass+=int(bflags.sum())
                checks.flag("FULL/total_qualification",model["gprime_SE_gate_count"]==404 and model["gprime_SE_gate_passed"]==gpass
                            and model["b_positive_node_count"]==404 and model["b_positive_nodes"]==bpass)
                rebuilt[pool]=dict(gprime_SE_pass=gpass,b_positive_nodes=bpass,scaled_spline_coefficient_max=max(coefficient_errors))
        result=dict(schema=1,passed=all(x["passed"] for x in checks.rows),source_sha256=sha(SOURCE),
            protocol_sha256=sha(PROTOCOL),input_receipt_sha256=sha(OUT/"INPUT_RECEIPT.json"),
            producer_sha256=sha(PRODUCER),checks=checks.rows,stream_receipts=receipts,rebuilt=rebuilt,
            elapsed_seconds=time.perf_counter()-start,producer_imported=False,
            accounting=dict(replayed_banks=40960,replayed_prior_vectors=10485760,replayed_Gaussian_scalars=2684354560,
                independently_Price_scored_banks=1536,independent_Price_entries=1703936,
                saved_Price_entries_reduced=53739520,new_independent_scientific_samples=0,new_model_integrals=0),
            scope="All RNG replayed; direct independent Price only first64/stream. All saved mode means, local coefficients, full projected curves, SEs, spline integrals and exact stored curvature checked. Not full underlying Price rerun or strict Gaussian integral/population b certification.")
        write_new(OUT/"RESULT.json",result)
        assert result["passed"],"Audit failed; preserve results"
        lines=["# LOCAL/FULL四case独立实现审计","","全部固定审计项通过。未导入生产校准程序或旧helper；源与范围先冻结，完整生产输入在审计算术前另绑定。","",
            "两池合计40960个bank的全部Gaussian输入已重放，逐流全数组和首块SHA均核对。底层Price独立重算限定每流首64bank、全部四case、对应全部rho与四模式，共1536bank、1703936项；使用直接sigmoid概率乘积与独立归一化。", "",
            f"独立Price最大绝对差 `{max(r['Price_max_absolute_error'] for r in receipts):.17g}`，预定阈值5e−13。其它bank只核全部保存模式的后续算术，未冒称全生产Price重算。","",
            "LOCAL的G、三个c_h、8流均值/SE/资格全重建。FULL的原始和投影U′/V′、g′、b、16流/pooled均值与SE全部重建；用不同的not-a-knot B-spline系统复核保存三次样条，再核独立反导、分段幂积分和全部逐流g值/SE。曲率receipt用保存binary系数和结点的精确Fraction重新求sup及向外性。","",
            "|case|LOCAL主系数|SE|LOCAL两门通过|","|---|---:|---:|---|"]
        for i,case in enumerate(CASES):lines.append(f"|{case}|{rebuilt['LOCAL']['coefficients'][i]:.12g}|{rebuilt['LOCAL']['coefficient_SEs'][i]:.12g}|{rebuilt['LOCAL']['case_gates'][i]}|")
        lines += ["",f"FULL g′SE资格 {rebuilt['FULL']['gprime_SE_pass']}/404；正b节点 {rebuilt['FULL']['b_positive_nodes']}/404。资格仍是开发诊断，不是统一置信带或严格总体b下界。","",
            "完整RNG重放对应2684354560个既有Gaussian标量；0新独立科学样本、0新模型积分。LOCAL仅读已知r，FULL只读允许的已知模型常数；无旧拟合g、训练观察或teacher风险调参，源访问范围另有CAL_CODE_REVIEW记录。","",
            f"审计源SHA `{sha(SOURCE)}`；审计协议 `{sha(PROTOCOL)}`；结果见 `artifacts/calibration_precision/CAL_AUDIT/RESULT.json`。",""]
        assert not DOC.exists(),"Refuse audit report overwrite"
        DOC.write_text("\n".join(lines),encoding="utf-8",newline="\n")
        print(json.dumps(dict(passed=True,checks=len(checks.rows),rebuilt=rebuilt)),flush=True)
    except Exception as exc:
        write_new(OUT/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),checks=checks.rows,stream_receipts=receipts))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["freeze","run"])
    args=parser.parse_args();freeze() if args.phase=="freeze" else audit()
