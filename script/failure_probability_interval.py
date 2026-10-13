"""Outward score-difference replay on exact saved binary64 audit inputs.

The sole scientific input is the already generated FailureProbability audit.
No Gaussian/S-generation or future native-program guarantee is made. The
arithmetic assumptions are those of numerical_sign_kernel: correctly rounded
separate IEEE binary64 basic operations, gradual underflow, nextafter and ldexp.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import traceback
import numpy as np
import numerical_sign_kernel as k

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/failure_probability_v1_20261006"
DOC = ROOT / "docs/08_research_研究工作/FailureProbability_v1_20261006"
INPUT = OUT / "AUDIT_BANKS.npz"
PREFIX = "SCORE_ARITHMETIC_"
N, K, BLOCK, TIME_LIMIT, ERROR_LIMIT = 262144, 100, 256, 1200, 1e-9
YS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=np.int8)
LOG_TERMS = 20
LOG_REMAINDER_EXACT = Fraction(9, 164 * 3**41)
LOG_REMAINDER_UPPER = math.nextafter(float(LOG_REMAINDER_EXACT), math.inf)
assert Fraction.from_float(LOG_REMAINDER_UPPER) >= LOG_REMAINDER_EXACT
private = ROOT / ".cache/finite_selection_tools"
if str(private) not in sys.path:
    sys.path.insert(0, str(private))
from flint import arb, ctx


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def save_new(path, record):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz_new(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def validate_interval(value, positive=False):
    if not np.all(np.isfinite(value.lo)) or not np.all(np.isfinite(value.hi)):
        raise ArithmeticError("nonfinite interval")
    if np.any(value.lo > value.hi):
        raise ArithmeticError("unordered interval")
    if positive and np.any(value.lo <= 0):
        raise ArithmeticError("positive lower endpoint lost")
    return value


def log_point(x):
    """Enclose log of exact positive binary64 x without a library logarithm."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("log_point requires finite strictly positive values")
    mantissa, exponent = np.frexp(x)
    m = np.ldexp(mantissa, 1)  # Exact power-of-two normalization, including subnormals.
    if np.any(m < 1) or np.any(m >= 2):
        raise ArithmeticError("frexp normalization left [1,2)")
    u = k.positive_div(k.sub(k.point(m), k.point(1.)), k.add(k.point(m), k.point(1.)))
    u = k.known_range(u, 0., math.nextafter(1./3., math.inf))
    u2 = k.known_range(k.mul(u, u), 0., math.nextafter(1./9., math.inf))
    polynomial = k.positive_div(k.point(1.), k.point(float(2*LOG_TERMS-1)))
    for j in range(LOG_TERMS-2, -1, -1):
        coefficient = k.positive_div(k.point(1.), k.point(float(2*j+1)))
        polynomial = k.add(coefficient, k.mul(u2, polynomial))
    series = k.mul(k.point(2.), k.mul(u, polynomial))
    series = k.add(series, k.Interval(np.asarray(0.), np.asarray(LOG_REMAINDER_UPPER)))
    exponent_term = k.mul(k.point(exponent-1), k.Interval(np.asarray(k.LN2_LO), np.asarray(k.LN2_HI)))
    return validate_interval(k.add(exponent_term, series))


def log_positive(value):
    """Monotone endpoint evaluation for a strictly positive interval."""
    validate_interval(value, positive=True)
    low, high = log_point(value.lo), log_point(value.hi)
    return validate_interval(k.Interval(low.lo, high.hi))


def root_constants():
    previous = ctx.prec
    ctx.prec = 192
    try:
        a = (arb(1)/10).sqrt()
        b = ((arb(14)/5).sqrt()-a)/3
        def enclosure(x):
            return k.Interval(np.asarray(math.nextafter(float(x.lower()), -math.inf)),
                              np.asarray(math.nextafter(float(x.upper()), math.inf)))
        ai, bi = enclosure(a), enclosure(b)
        assert arb(float(ai.lo)) <= a <= arb(float(ai.hi))
        assert arb(float(bi.lo)) <= b <= arb(float(bi.hi))
        return ai, bi
    finally:
        ctx.prec = previous


def sigmoid_interval_pair(x):
    validate_interval(x)
    sl, ql = k.sigmoid_pair(x.lo)
    sh, qh = k.sigmoid_pair(x.hi)
    return (validate_interval(k.Interval(sl.lo, sh.hi)),
            validate_interval(k.Interval(qh.lo, ql.hi)))


def bound_differences(z, s, constants=None):
    """Return B x 8 logmix(Cb)-logmix(I), with exact saved Z/S targets."""
    z = np.asarray(z, dtype=np.float64);s = np.asarray(s, dtype=np.float64)
    if z.ndim != 3 or z.shape[-1] != 3 or min(z.shape[:2]) < 1 or s.shape != z.shape[:2]:
        raise ValueError("expected z=(B,K,3), s=(B,K), B,K>=1")
    if not np.isfinite(z).all() or not np.isfinite(s).all():
        raise ValueError("all input points must be finite")
    a, b = root_constants() if constants is None else constants
    zi = k.point(z)
    total_z = k.pairwise_sum(zi, axis=2, keepdims=True)
    xb = k.add(k.mul(a, zi), k.mul(b, total_z))
    pi, qi = k.sigmoid_pair(z)
    pb, qb = sigmoid_interval_pair(xb)
    shift = np.max(s, axis=1, keepdims=True)
    relative = k.known_range(k.sub(k.point(s), k.point(shift)), -np.inf, 0.)
    weight = k.exp_nonpositive(relative)
    columns = []
    for label in YS:
        def mixture(plus, minus):
            probabilities = [k.take(plus if label[j] > 0 else minus, (..., j)) for j in range(3)]
            active = k.mul(k.mul(probabilities[0], probabilities[1]), probabilities[2])
            product = k.known_range(k.mul(weight, active), 0., 1.)
            return validate_interval(k.pairwise_sum(product, axis=1), positive=True)
        mi, mb = mixture(pi, qi), mixture(pb, qb)
        columns.append(k.sub(log_positive(mb), log_positive(mi)))
    return validate_interval(k.Interval(np.stack([v.lo for v in columns], axis=1),
                                        np.stack([v.hi for v in columns], axis=1)))


def arb_reference(z, s, bits=256):
    """Independent Arb evaluation with exact closed-form root, all eight labels."""
    previous = ctx.prec;ctx.prec = bits
    try:
        a = (arb(1)/10).sqrt();b = ((arb(14)/5).sqrt()-a)/3
        iz = [[arb(float(v)) for v in row] for row in z]
        bz = [[a*v+b*sum(row, arb(0)) for v in row] for row in iz]
        maximum = arb(float(np.max(s)))
        weights = [(arb(float(v))-maximum).exp() for v in s]
        def probabilities(rows):
            # Compute both signs directly; avoid cancellation of 1-sigma(x).
            return [[(1/(1+(-x).exp()), 1/(1+x.exp())) for x in row] for row in rows]
        ips, bps = probabilities(iz), probabilities(bz)
        result = []
        for label in YS:
            def mix(probs):
                return sum((w*row[0][int(label[0]<0)]*row[1][int(label[1]<0)]*row[2][int(label[2]<0)]
                            for w,row in zip(weights,probs)), arb(0))
            result.append(mix(bps).log()-mix(ips).log())
        return result
    finally:
        ctx.prec = previous


def qualification():
    """Manufactured inputs only; no scientific array access."""
    previous = ctx.prec;ctx.prec = 256
    try:
        assert np.dtype(np.float64).itemsize == 8
        assert float(np.nextafter(0., math.inf)) == k.MIN_SUBNORMAL
        assert float(np.ldexp(1., -1074)) == k.MIN_SUBNORMAL
        assert k.MIN_SUBNORMAL+k.MIN_SUBNORMAL == 2*k.MIN_SUBNORMAL
        values = np.array([k.MIN_SUBNORMAL,2*k.MIN_SUBNORMAL,2.**-1022,2.**-800,
                           1e-100,.25,.5,np.nextafter(1.,0.),1.,np.nextafter(1.,math.inf),
                           2.,8.,2.**100,np.finfo(np.float64).max])
        interval = log_point(values)
        logs = []
        for x,lo,hi in zip(values,interval.lo,interval.hi):
            reference = arb(float(x)).log()
            assert arb(float(lo)) <= reference <= arb(float(hi)), (x,lo,hi,str(reference))
            logs.append(dict(x=float(x),lower=float(lo),upper=float(hi),Arb=str(reference)))
        import nuisance_contrast_compute as legacy
        # Importing legacy selects its default precision; restore the fixture precision.
        ctx.prec = 256
        assert np.array_equal(YS, legacy.YS)
        fixtures = [(name,z,s) for name,z,s in k.manufactured_fixtures()
                    if len(z) in (1,2,3,100)]
        rows = []
        for name,z,s in fixtures:
            bounds = bound_differences(z[None,...],s[None,...])
            reference = arb_reference(z,s,256)
            legacy_rows = []
            for yi,value in enumerate(reference):
                lo,hi = float(bounds.lo[0,yi]),float(bounds.hi[0,yi])
                assert arb(lo) <= value <= arb(hi), (name,yi,str(value),lo,hi)
                counts = np.zeros(8,dtype=np.int64);counts[yi]=1
                iv = legacy.interval_pair((0,0,0),z[None,...],s[None,...],counts)[0][1]
                bv = legacy.interval_pair((9,9,9),z[None,...],s[None,...],counts)[0][1]
                old_reference = iv-bv
                assert old_reference.overlaps(value), (name,yi,"old Arb target mismatch")
                legacy_rows.append(str(old_reference))
            repeated = bound_differences(np.repeat(z[None,...],2,axis=0),np.repeat(s[None,...],2,axis=0))
            assert np.array_equal(repeated.lo,np.repeat(bounds.lo,2,axis=0))
            assert np.array_equal(repeated.hi,np.repeat(bounds.hi,2,axis=0))
            rows.append(dict(name=name,K=len(z),lower=bounds.lo[0].tolist(),upper=bounds.hi[0].tolist(),
                             Arb=[str(v) for v in reference],legacy_Arb=legacy_rows))
        rejected = []
        for name,call in [
            ("log_zero",lambda:log_point(0.)),
            ("log_negative",lambda:log_point(-1.)),
            ("log_infinite",lambda:log_point(math.inf)),
            ("nonfinite_Z",lambda:bound_differences(np.array([[[math.inf,0.,0.]]]),np.zeros((1,1)))),
            ("lost_positive_mixture",lambda:bound_differences(np.full((1,1,3),1000.),np.zeros((1,1))))]:
            try:
                call()
            except (ArithmeticError,ValueError):
                rejected.append(name)
            else:
                raise AssertionError("manufactured failure was not rejected: "+name)
        name,z,s = fixtures[-1]
        timings = []
        for _ in range(3):
            start=time.monotonic()
            bound_differences(np.repeat(z[None,...],BLOCK,axis=0),np.repeat(s[None,...],BLOCK,axis=0))
            timings.append(time.monotonic()-start)
        a,b = root_constants()
        return dict(status="passed",scientific_banks_read=0,new_random_draws=0,bits=256,
                    log_terms=LOG_TERMS,log_remainder_exact=str(LOG_REMAINDER_EXACT),
                    log_remainder_upper=LOG_REMAINDER_UPPER,
                    root_a=[float(a.lo),float(a.hi)],root_b=[float(b.lo),float(b.hi)],
                    log_fixtures=logs,score_fixtures=rows,explicit_failure_probes=rejected,
                    label_order=YS.tolist(),benchmark_seconds=timings,
                    full_pass_extrapolation_seconds=float(np.median(timings))*(N//BLOCK),
                    IEEE_assumptions_proved_by_fixtures=False)
    finally:
        ctx.prec = previous


def qualify():
    if (OUT / (PREFIX+"FREEZE.json")).exists():
        raise FileExistsError("source already frozen; no development qualification")
    attempt = 1
    while (OUT / f"{PREFIX}QUALIFICATION_ATTEMPT_{attempt:03}.json").exists():
        attempt += 1
    target = OUT / f"{PREFIX}QUALIFICATION_ATTEMPT_{attempt:03}.json"
    sources = [Path(__file__),Path(k.__file__),DOC/"SCORE_ARITHMETIC_CONTRACT.md",
               ROOT/"script/nuisance_contrast_compute.py",ROOT/"script/finite_selection_compute.py"]
    record = dict(started_utc=now(),source_sha256={rel(p):sha(p) for p in sources})
    try:
        record["qualification"] = qualification()
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        (OUT/f"{PREFIX}QUALIFICATION_SOURCE_{attempt:03}.py").write_bytes(Path(__file__).read_bytes())
        save_new(target,record)
        raise
    record["finished_utc"] = now()
    save_new(target,record)
    print(json.dumps(dict(qualified=True,receipt=rel(target),
                          benchmark=record["qualification"]["full_pass_extrapolation_seconds"])),flush=True)


def freeze():
    target = OUT/(PREFIX+"FREEZE.json")
    if target.exists() or (OUT/(PREFIX+"RUN_START.json")).exists():
        raise FileExistsError("refusing to replace score freeze/run")
    candidates = sorted(OUT.glob(PREFIX+"QUALIFICATION_ATTEMPT_*.json"))
    qualification_path = candidates[-1]
    qualified = read(qualification_path)
    assert qualified["status"] == "passed"
    for name,digest in qualified["source_sha256"].items():
        assert sha(ROOT/name) == digest, name
    result = read(OUT/"RESULTS.json")
    assert result["array"]["path"] == rel(INPUT)
    assert sha(INPUT) == result["array"]["sha256"]
    indices = np.linspace(0,N-1,32,dtype=int).tolist()
    assert indices == read(OUT/"FREEZE.json")["bank_audit_indices"]
    paths = [Path(__file__),Path(k.__file__),DOC/"SCORE_ARITHMETIC_CONTRACT.md",
             qualification_path,OUT/"RESULTS.json",OUT/"SCALAR_STATS.json",
             ROOT/"script/nuisance_contrast_compute.py",ROOT/"script/finite_selection_compute.py"]
    source_snapshot=OUT/(PREFIX+"FROZEN_SOURCE.py")
    with source_snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    record=dict(created_utc=now(),scope="exact_saved_binary64_Z_S_all_eight_label_anchor_differences",
                input_path=rel(INPUT),input_sha256=result["array"]["sha256"],input_bytes=INPUT.stat().st_size,
                source_sha256={rel(p):sha(p) for p in paths},source_snapshot=rel(source_snapshot),
                qualification_path=rel(qualification_path),N=N,K=K,block=BLOCK,
                timeout_seconds=TIME_LIMIT,eta_d=ERROR_LIMIT,log_terms=LOG_TERMS,
                label_order=YS.tolist(),Arb_indices=indices,Arb_bits=256,
                new_random_draws=0,new_statistical_comparisons=0,
                probability_lower_unchanged=result["inference"]["mathematical_program"]["probability_lower"])
    save_new(target,record)
    print(json.dumps(dict(frozen=True,freeze_sha256=sha(target),input_sha256=record["input_sha256"])),flush=True)


def verify_freeze():
    frozen = read(OUT/(PREFIX+"FREEZE.json"))
    for name,digest in frozen["source_sha256"].items():
        assert sha(ROOT/name) == digest, ("frozen source changed",name)
    assert sha(ROOT/frozen["source_snapshot"]) == sha(__file__)
    assert (frozen["N"],frozen["K"],frozen["block"],frozen["timeout_seconds"],frozen["eta_d"]) == (N,K,BLOCK,TIME_LIMIT,ERROR_LIMIT)
    return frozen


def write_results_doc(record):
    content=f"""# 全保存审计输入的锚点评分差算术包络

本步骤按独立的事后确定性合同执行，不修改原随机审计合同、标量统计、置信家族或 RESULTS。固定把保存的 binary64 Z/S 视为精确实数，对精确相关矩阵 Cb 的闭式平方根、logistic、无关相对权重、混合求和及对数全部外向计算。

状态：**{record['status']}**。完成 {record['completed_banks']:,} 个银行、K100、全部八标签，共 {record['label_differences']:,} 个点评分差。保存浮点差到数学目标的最大误差包络为 **{record['maximum_saved_difference_error_enclosed']:.17g}**，原门为 **10^-9**；最大区间宽度 {record['maximum_interval_width']:.17g}。运行用时 {record['wall_seconds']:.3f} 秒，上限20分钟，没有追加银行或修改误差门。

## 方法与独立核验

沿用已冻结的外向 binary64 后端，Cb 平方根采用 aI+b11 的192-bit Arb常数外包，a=sqrt(1/10)、b=(sqrt(14/5)-a)/3。正数对数由 frexp 归约到[1,2)，用20项 atanh 级数及严格尾界9/(164·3^41)。无关权重公共归一化在 log mix_Cb−log mix_I 中精确抵消。保存的原浮点差只作误差比较，不作为数学目标。

制造资格覆盖 subnormal、1附近、2的幂和大/小正数对数，以及K1/2/3/100全部八标签的评分差；同时与独立高精度Arb和旧Arb接口核对。非有限输入与非正混合下端明确失败。源码、合同、资格收据和科学输入哈希在全银行执行前冻结。

另在原预定的32个等距银行上，对256个标签差进行独立256-bit Arb交叉核验，所有高精度球均被全覆盖后端的相应区间包含：{record['Arb_diagnostics']['all_contained']}。这些诊断不替代后端自身的区间推导。

## 新补充的保证与仍然存在的条件

本结果补足本轮审计**全部保存输入上的点评分算术**误差门。原来仅32银行诊断的历史记录保持，升级由本文件与独立结果收据说明。原逐标签η_d不改，原0.70355036287487的数学程序概率下界及其99%审计置信度不重新计算、不提高；没有新增随机覆盖事件。

结论仍依赖分开的IEEE binary64基本运算、渐进下溢、nextafter和ldexp等后端假设；资格夹具不是对这些平台假设的证明。没有认证Gaussian生成、PRNG、保存S相对于精确无关变换的误差，也没有给未来任意输入、原生程序的统一ε_alg或无条件算法失败概率。原统计理论所用的理想iid模型与输入生成资格继续保留。

原RESULTS.json和SCALAR_STATS.json哈希核验未变。新资源只是原保存输入的区间重算：{record['label_differences']:,} 个点评分差、{2*record['label_differences']:,} 个锚点混合对数，另256项Arb交叉核验；新观测、RNG、先验银行、优化和求积均为0。全部逐例lower/upper/error保存于[区间产物](../../../{record['output']['path']})，源与输入绑定见[SCORE_ARITHMETIC_FREEZE](../../../artifacts/failure_probability_v1_20261006/SCORE_ARITHMETIC_FREEZE.json)，结果见[SCORE_ARITHMETIC_RESULTS](../../../artifacts/failure_probability_v1_20261006/SCORE_ARITHMETIC_RESULTS.json)。未改论文或封存本包。
"""
    (DOC/"SCORE_ARITHMETIC_RESULTS.md").write_text(content,encoding="utf-8",newline="\n")


def run():
    started=time.monotonic();completed=0
    frozen=verify_freeze()
    for name in ("RUN_START.json","RESULTS.json","INTERVALS.npz","FAILURE.json","PARTIAL_INTERVALS.npz"):
        if (OUT/(PREFIX+name)).exists():
            raise FileExistsError("refusing to overwrite "+PREFIX+name)
    assert sha(INPUT) == frozen["input_sha256"]
    save_new(OUT/(PREFIX+"RUN_START.json"),dict(started_utc=now(),freeze_sha256=sha(OUT/(PREFIX+"FREEZE.json")),
                                             input_sha256=frozen["input_sha256"],new_random_draws=0))
    lower=np.empty((N,8));upper=np.empty((N,8));errors=np.empty((N,8))
    try:
        with np.load(INPUT,allow_pickle=False) as arrays:
            z=arrays["z"];s=arrays["s"];reference=arrays["difference_by_label"];labels=arrays["label_order"]
        assert z.shape==(N,K,3) and s.shape==(N,K) and reference.shape==(N,8)
        assert np.array_equal(labels,YS) and np.isfinite(reference).all()
        constants=root_constants()
        for first in range(0,N,BLOCK):
            last=min(first+BLOCK,N)
            value=bound_differences(z[first:last],s[first:last],constants)
            lower[first:last]=value.lo;upper[first:last]=value.hi
            difference=k.sub(value,k.point(reference[first:last]))
            errors[first:last]=np.maximum(np.abs(difference.lo),np.abs(difference.hi))
            completed=last
            elapsed=time.monotonic()-started
            if elapsed>TIME_LIMIT:
                raise TimeoutError("fixed 20-minute score arithmetic budget exceeded")
            if last%32768==0:
                print(json.dumps(dict(stage="score_interval",completed=completed,total=N,seconds=elapsed,
                                      maximum_error=float(np.max(errors[:completed])))),flush=True)
        audits=[];previous=ctx.prec;ctx.prec=256
        try:
            for index in frozen["Arb_indices"]:
                values=arb_reference(z[index],s[index],256)
                rows=[]
                for yi,value in enumerate(values):
                    lo,hi=float(lower[index,yi]),float(upper[index,yi])
                    contained=bool(arb(lo)<=value<=arb(hi))
                    rows.append(dict(label_index=yi,lower=lo,upper=hi,Arb=str(value),contained=contained))
                    if not contained:
                        raise AssertionError(("Arb cross-check not contained",index,yi,lo,hi,str(value)))
                audits.append(dict(bank_index=index,rows=rows))
                if time.monotonic()-started>TIME_LIMIT:
                    raise TimeoutError("fixed score arithmetic budget exceeded in Arb diagnostics")
        finally:
            ctx.prec=previous
        audit=dict(bits=256,indices=frozen["Arb_indices"],bank_label_differences=256,all_contained=True,rows=audits)
        save_new(OUT/(PREFIX+"ARB_CHECKS.json"),audit)
        output=OUT/(PREFIX+"INTERVALS.npz")
        save_npz_new(output,index=np.arange(N,dtype=np.int64),label_order=YS,lower=lower,upper=upper,error_bound=errors)
        maximum=float(np.max(errors));width=float(np.max(k.up(upper-lower)))
        elapsed=time.monotonic()-started
        verify_freeze()
        assert sha(INPUT)==frozen["input_sha256"]
        passed=maximum<=ERROR_LIMIT and elapsed<=TIME_LIMIT
        record=dict(status="passed" if passed else "failed_gate",finished_utc=now(),
                    freeze_sha256=sha(OUT/(PREFIX+"FREEZE.json")),input_sha256=frozen["input_sha256"],
                    completed_banks=completed,total_banks=N,K=K,label_differences=N*8,wall_seconds=elapsed,
                    maximum_saved_difference_error_enclosed=maximum,maximum_interval_width=width,eta_d=ERROR_LIMIT,
                    all_saved_point_differences_enclosed=True,all_saved_points_error_gate_passed=maximum<=ERROR_LIMIT,
                    IEEE_arithmetic_assumptions_retained=True,Gaussian_generation_certified=False,
                    saved_S_generation_certified=False,PRNG_certified=False,uniform_future_epsilon_alg_proved=False,
                    original_probability_and_scalar_records_unchanged=True,probability_lower_unchanged=frozen["probability_lower_unchanged"],
                    new_random_draws=0,new_observations=0,new_prior_banks=0,new_statistical_comparisons=0,
                    new_optimizations=0,new_quadrature_nodes=0,
                    Arb_diagnostics={key:value for key,value in audit.items() if key!="rows"},
                    gates=dict(error_budget=maximum<=ERROR_LIMIT,Arb_containment=True,deadline=elapsed<=TIME_LIMIT),
                    output=dict(path=rel(output),sha256=sha(output),bytes=output.stat().st_size))
        save_new(OUT/(PREFIX+"RESULTS.json"),record)
        write_results_doc(record)
        print(json.dumps(dict(status=record["status"],completed_banks=completed,label_differences=N*8,
                              maximum_error=maximum,seconds=elapsed,probability_unchanged=True)),flush=True)
    except BaseException as error:
        if completed:
            save_npz_new(OUT/(PREFIX+"PARTIAL_INTERVALS.npz"),lower=lower[:completed],upper=upper[:completed],
                         error_bound=errors[:completed],label_order=YS)
        save_new(OUT/(PREFIX+"FAILURE.json"),dict(status="failed",completed_banks=completed,
                 total_banks=N,wall_seconds=time.monotonic()-started,error=repr(error),
                 traceback=traceback.format_exc(),full_coverage_claimed=False))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("qualify","freeze","run"))
    args=parser.parse_args()
    globals()[args.mode]()
