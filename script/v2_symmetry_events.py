"""Frozen two-group symmetry event experiment; no fitting or likelihood MC."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import beta, binom, norm

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "artifacts/structure_tradeoff_v2/E"
DOC = ROOT / "docs/08_research_研究工作/结构预算与对称化代价_v2_20260930"
OLD = ROOT / "docs/08_research_研究工作/非交换适用域_v2_20260930"
CFG = dict(dimensions=[16, 64, 128, 512],
           profiles=[dict(name="null", scales=[4., 4.]),
                     dict(name="mild", scales=[4.5, 3.5]),
                     dict(name="contrast", scales=[6., 2.])],
           c=.9, streams=8, observations_per_stream=65536, batch=2048,
           seed_base=20273001, alpha=.05, family_size=12,
           quad_upper=12., quad_epsabs=1e-12, quad_epsrel=1e-12,
           quad_limit=300, quad_gate=1e-10,
           primary_dimensions=[128], primary_profiles=["mild", "contrast"],
           seed_formula="base+100000*profile_index+100*d+stream_index",
           observation_unit="independent randomized Bernoulli sign-agreement event",
           tie_rule="two independent fair signs; consume both uniforms every observation",
           dtype="float64", threads=2, fits=0,
           no_material_size_threshold=True,
           no_failed_gate_sample_extension=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def guard_files(items):
    for item in items:
        path = ROOT / item["path"]
        assert path.exists(), f"missing frozen file: {path}"
        assert sha(path) == item["sha256"], f"frozen hash mismatch: {path}"


def file_record(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def freeze():
    assert not (HERE / "FREEZE.json").exists(), "freeze already exists"
    assert not list(HERE.glob("streams/*.json")), "scientific results already exist"
    inp = HERE / "source_inputs"
    inp.mkdir(parents=True, exist_ok=True)
    sources = [(ROOT / "HANDOFF.md", inp / "HANDOFF.md"),
               (OLD / "THEORY.md", inp / "PREVIOUS_THEORY.md"),
               (OLD / "INDEPENDENT_REVIEW.md", inp / "PREVIOUS_INDEPENDENT_REVIEW.md")]
    for src, dst in sources:
        assert not dst.exists(), dst
        shutil.copyfile(src, dst)
        assert src.read_bytes() == dst.read_bytes()
    write_new(HERE / "CONFIG.json", CFG)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    handoff = (inp / "HANDOFF.md").read_text(encoding="utf-8")
    assert "ce1aa834" in handoff, "unexpected external-review target"
    items = [Path(__file__), DOC / "PLAN_E.md", DOC / "MISMATCH_THEORY.md",
             HERE / "CONFIG.json"] + [dst for _, dst in sources]
    receipt = dict(created_utc=utc(), repository_head=head,
                   reviewed_commit="ce1aa8340028dbdfafc42187876bfc44999e2813",
                   handoff_usage="reference; mathematical candidate independently checked",
                   input_snapshot_reason="preserve exact bytes; future live HANDOFF may change",
                   source_files=[file_record(p) for p in items],
                   config=CFG, python=sys.version, numpy=np.__version__,
                   scipy=scipy.__version__, platform=platform.platform(),
                   estimated_primitives=dict(observations=6291456,
                       gaussian_draws=1132462080,
                       bernoulli_uniform_draws=1132462080,
                       tie_uniform_draws=12582912),
                   science_status="not yet integrated parameters or sampled observations")
    write_new(HERE / "FREEZE.json", receipt)
    print(json.dumps(dict(freeze=rel(HERE / "FREEZE.json"),
                          sha256=sha(HERE / "FREEZE.json"),
                          handoff_sha256=sha(inp / "HANDOFF.md")), ensure_ascii=False),
          flush=True)


def frozen_guard():
    fr = read(HERE / "FREEZE.json")
    assert fr["config"] == CFG == read(HERE / "CONFIG.json")
    guard_files(fr["source_files"])
    return fr


def klhalf(p):
    if p == 0. or p == 1.:
        return math.log(2.)
    return p * math.log(2*p) + (1-p) * math.log(2*(1-p))


def info_lower(lo, hi):
    if lo <= .5 <= hi:
        return 0.
    return klhalf(lo if lo > .5 else hi)


def deterministic_smoke():
    qtop = CFG["quad_upper"]
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        for s in sorted({s for p in CFG["profiles"] for s in p["scales"]}):
            first, e1 = quad(lambda z: 2*z*np.tanh(s*z/2)*norm.pdf(z),
                             0., qtop, epsabs=CFG["quad_epsabs"],
                             epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"])
            second, e2 = quad(lambda z: 4*s*expit(s*z)*expit(-s*z)*norm.pdf(z),
                              0., qtop, epsabs=CFG["quad_epsabs"],
                              epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"])
            tail1, tail2 = 2*norm.pdf(qtop), s*norm.sf(qtop)
            assert max(e1+tail1, e2+tail2, abs(first-second)) <= CFG["quad_gate"]
            assert 0 < first < 1
            values[str(s)] = dict(a=float(first), stein_a=float(second),
                absolute_disagreement=abs(first-second), quad_error_first=e1,
                quad_error_stein=e2, tail_bound_first=tail1, tail_bound_stein=tail2)
        ap, ape = quad(lambda z: 4*z*z*expit(4*z)*expit(-4*z)*norm.pdf(z),
                       0., qtop, epsabs=CFG["quad_epsabs"],
                       epsrel=CFG["quad_epsrel"], limit=CFG["quad_limit"])
        ap_tail = qtop*norm.pdf(qtop)+norm.sf(qtop)
        assert ape + ap_tail <= CFG["quad_gate"]
    profiles = []
    for p in CFG["profiles"]:
        aa, ab = [values[str(s)]["a"] for s in p["scales"]]
        plus, minus = (aa+ab)/2, (aa-ab)/2
        cov = np.eye(2) - CFG["c"]*np.outer([plus, minus], [plus, minus])
        assert np.linalg.eigvalsh(cov)[0] > 0
        zeta = cov[0, 1]/math.sqrt(cov[0, 0]*cov[1, 1])
        pinf = .5 + math.asin(zeta)/math.pi
        if p["name"] == "null":
            assert zeta == 0 and pinf == .5
        profiles.append(dict(profile=p["name"], a_A=aa, a_B=ab, a_plus=plus,
            a_minus=minus, limiting_covariance=cov.tolist(), zeta=float(zeta),
            limit_event_probability=pinf, limit_event_kl=klhalf(pinf),
            qualification="asymptotic prediction, not finite-d CP probability"))
    slices = {}
    for bits in itertools.product((0, 1), repeat=4):
        rawt, rawv = 2*sum(bits)-4, 2*(sum(bits[:2])-sum(bits[2:]))
        success = 0
        swap_success = 0
        for ct, cv in itertools.product((-1, 1), repeat=2):
            st = int(np.sign(rawt)) if rawt else ct
            sv = int(np.sign(rawv)) if rawv else cv
            swapped = int(np.sign(-rawv)) if rawv else cv
            success += int(st == sv)
            swap_success += int(st == swapped)
        assert success + swap_success == 4
        slices.setdefault(sum(bits), []).append(success)
    for cnt, successes in slices.items():
        assert sum(successes) == 2*len(successes), cnt
    projection_errors = {}
    for d in CFG["dimensions"]:
        alpha = math.sqrt(1+CFG["c"]/(d-1))
        b = alpha-math.sqrt(1-CFG["c"])
        mat = alpha*np.eye(d)-(b/d)*np.ones((d, d))
        target = (1+CFG["c"]/(d-1))*np.eye(d)-(CFG["c"]/(d-1))*np.ones((d, d))
        err = float(np.max(np.abs(mat @ mat.T - target)))
        assert err < 1e-12
        projection_errors[str(d)] = err
    aval = values["4.0"]["a"]
    local_coefficient = 2*CFG["c"]**2*aval**2*ap**2/(math.pi**2*(1-CFG["c"]*aval**2))
    return dict(created_utc=utc(), pass_all=True, scale_integrals=values,
                profiles=profiles, a_prime_at_four=ap,
                a_prime_error=ape, a_prime_tail_bound=ap_tail,
                local_delta_squared_event_kl_coefficient=local_coefficient,
                count_slice_exact_half_smoke={str(k): dict(patterns=len(v),
                    successes_over_all_coins=sum(v), coin_outcomes=4*len(v))
                    for k, v in slices.items()},
                projection_covariance_max_abs_errors=projection_errors,
                new_observations=0)


def stream_path(name, d, stream):
    return HERE / "streams" / f"{name}_d{d}_s{stream}.json"


def sample_stream(profile_index, profile, d, stream):
    seed = CFG["seed_base"]+100000*profile_index+100*d+stream
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.bit_generator.state
    n, batch, m = CFG["observations_per_stream"], CFG["batch"], d//2
    hist = np.zeros((m+1, m+1), dtype=np.int64)
    joint = np.zeros(4, dtype=np.int64)
    no_tie_joint = np.zeros(4, dtype=np.int64)
    tie_joint = np.zeros(4, dtype=np.int64)
    ties = np.zeros(3, dtype=np.int64)
    sums = np.zeros(5, dtype=np.int64)
    alpha = math.sqrt(1+CFG["c"]/(d-1))
    b = alpha-math.sqrt(1-CFG["c"])
    sA, sB = profile["scales"]
    begin = time.perf_counter()
    for offset in range(0, n, batch):
        size = min(batch, n-offset)
        z = rng.standard_normal((size, d), dtype=np.float64)
        means = z.mean(axis=1)
        z *= alpha
        z -= b*means[:, None]
        z[:, :m] *= sA
        z[:, m:] *= sB
        expit(z, out=z)
        labels = rng.random((size, d)) < z
        na = labels[:, :m].sum(axis=1, dtype=np.int64)
        nb = labels[:, m:].sum(axis=1, dtype=np.int64)
        rawt = 2*(na+nb)-d
        rawv = 2*(na-nb)
        coins = rng.random((size, 2)) < .5
        st = np.where(rawt == 0, coins[:, 0], rawt > 0)
        sv = np.where(rawv == 0, coins[:, 1], rawv > 0)
        code = st.astype(np.int64)*2+sv.astype(np.int64)
        has_tie = (rawt == 0) | (rawv == 0)
        joint += np.bincount(code, minlength=4)
        no_tie_joint += np.bincount(code[~has_tie], minlength=4)
        tie_joint += np.bincount(code[has_tie], minlength=4)
        ties += [np.count_nonzero(rawt == 0), np.count_nonzero(rawv == 0),
                 np.count_nonzero((rawt == 0) & (rawv == 0))]
        sums += [rawt.sum(), rawv.sum(), (rawt*rawt).sum(),
                 (rawv*rawv).sum(), (rawt*rawv).sum()]
        hist += np.bincount(na*(m+1)+nb, minlength=(m+1)**2).reshape(m+1, m+1)
    elapsed = time.perf_counter()-begin
    ai, bi = np.nonzero(hist)
    sparse = np.column_stack((ai, bi, hist[ai, bi])).tolist()
    return dict(profile=profile["name"], scales=profile["scales"], d=d,
        stream=stream, seed=seed, n=n, successes=int(joint[0]+joint[3]),
        sign_joint_order=["minus_minus", "minus_plus", "plus_minus", "plus_plus"],
        sign_joint=joint.tolist(), no_tie_sign_joint=no_tie_joint.tolist(),
        any_tie_sign_joint=tie_joint.tolist(),
        ties=dict(T=int(ties[0]), V=int(ties[1]), both=int(ties[2])),
        raw_moments_order=["sum_T", "sum_V", "sum_T_squared", "sum_V_squared", "sum_TV"],
        raw_moments=sums.tolist(), group_count_sparse=sparse,
        rng=dict(bit_generator="PCG64", initial_state=initial,
                 final_state=rng.bit_generator.state,
                 array_order="C", batch=batch,
                 draw_order="normal[size,d],uniform[size,d],uniform[size,2]"),
        elapsed_seconds=elapsed,
        primitives=dict(gaussian_draws=n*d, bernoulli_uniform_draws=n*d,
                        tie_uniform_draws=2*n))


def cp_beta(successes, n):
    tail = CFG["alpha"]/(2*CFG["family_size"])
    lo = 0. if successes == 0 else float(beta.ppf(tail, successes, n-successes+1))
    hi = 1. if successes == n else float(beta.ppf(1-tail, successes+1, n-successes))
    return lo, hi


def primary(row):
    return row["d"] in CFG["primary_dimensions"] and row["profile"] in CFG["primary_profiles"]


def collect():
    smoke = read(HERE / "SMOKE.json")
    rows = []
    for p in CFG["profiles"]:
        lim = next(x for x in smoke["profiles"] if x["profile"] == p["name"])
        for d in CFG["dimensions"]:
            data = [read(stream_path(p["name"], d, j)) for j in range(CFG["streams"])]
            n = sum(x["n"] for x in data)
            k = sum(x["successes"] for x in data)
            lo, hi = cp_beta(k, n)
            lower = info_lower(lo, hi)
            sums = np.sum(np.array([x["raw_moments"] for x in data], dtype=np.int64), axis=0)
            tmean, vmean = sums[:2]/n/math.sqrt(d)
            moments = dict(mean_T=float(tmean), mean_V=float(vmean),
                var_T=float(sums[2]/n/d-tmean*tmean),
                var_V=float(sums[3]/n/d-vmean*vmean),
                cov_TV=float(sums[4]/n/d-tmean*vmean))
            row = dict(profile=p["name"], scales=p["scales"], d=d, n=n, successes=k,
                p_hat=k/n, cp95_bonferroni_lower=lo, cp95_bonferroni_upper=hi,
                event_kl_plugin=klhalf(k/n), event_kl_simultaneous_lower=lower,
                exact_symmetrized_event_probability=.5,
                asymptotic_prediction=lim["limit_event_probability"],
                asymptotic_event_kl=lim["limit_event_kl"],
                tied_T=sum(x["ties"]["T"] for x in data),
                tied_V=sum(x["ties"]["V"] for x in data),
                tied_both=sum(x["ties"]["both"] for x in data),
                moments_descriptive= moments,
                elapsed_seconds=sum(x["elapsed_seconds"] for x in data))
            row["primary"] = primary(row)
            row["primary_pass"] = lower > 0 if primary(row) else None
            rows.append(row)
    return dict(created_utc=utc(), rows=rows, all_observations=sum(x["n"] for x in rows),
        primary_pass_all=all(x["primary_pass"] for x in rows if x["primary"]),
        family_size=CFG["family_size"], simultaneous_coverage=.95,
        uncertainty="iid finite-d event sampling; not 8 training repeats",
        fits=0, sample_extension=0,
        precision=dict(dtype="float64", integer_counts=True, exact_baseline=.5,
                       parameter_smoke_pass=smoke["pass_all"]),
        interpretation="lower bound on full-vector P versus its full symmetrization KL",
        no_full_pattern_kl_estimate=True)


def write_counts_csv(data):
    fields = ["profile", "d", "stream", "seed", "n", "successes", "elapsed_seconds",
              "tie_T", "tie_V", "tie_both", "gaussian_draws", "bernoulli_uniform_draws",
              "tie_uniform_draws"]
    path = HERE / "STREAM_COUNTS.csv"
    with path.open("x", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for x in data:
            row = {k: x[k] for k in fields[:7]}
            row.update(tie_T=x["ties"]["T"], tie_V=x["ties"]["V"],
                       tie_both=x["ties"]["both"], **x["primitives"])
            out.writerow(row)


def independent_audit():
    """Recompute with integer histogram loops and binomial-tail inversion."""
    frozen_guard()
    result = read(HERE / "RESULTS.json")
    with (HERE / "STREAM_COUNTS.csv").open(encoding="utf-8", newline="") as f:
        csvrows = list(csv.DictReader(f))
    assert len(csvrows) == 96
    mapping = {(r["profile"], int(r["d"]), int(r["stream"])): r for r in csvrows}
    assert len(mapping) == 96
    aggregated = {}
    seeds = set()
    primitive_totals = [0, 0, 0]
    for pi, prof in enumerate(CFG["profiles"]):
        for d in CFG["dimensions"]:
            ktotal, ntotal = 0, 0
            for j in range(CFG["streams"]):
                x = read(stream_path(prof["name"], d, j))
                assert x["seed"] == CFG["seed_base"]+100000*pi+100*d+j
                assert x["seed"] not in seeds
                seeds.add(x["seed"])
                assert x["profile"] == prof["name"] and x["d"] == d and x["stream"] == j
                assert x["scales"] == prof["scales"]
                assert x["rng"]["bit_generator"] == "PCG64"
                init = np.random.PCG64(x["seed"]).state
                assert x["rng"]["initial_state"] == init
                assert x["rng"]["final_state"] != init
                assert x["rng"]["batch"] == CFG["batch"]
                ncount, tzero, vzero, bothzero = 0, 0, 0, 0
                untied = [0]*4
                moments = [0]*5
                seen = set()
                for na, nb, freq in x["group_count_sparse"]:
                    assert isinstance(freq, int) and freq > 0
                    assert 0 <= na <= d//2 and 0 <= nb <= d//2
                    assert (na, nb) not in seen
                    seen.add((na, nb))
                    tt, vv = 2*(na+nb)-d, 2*(na-nb)
                    ncount += freq
                    tzero += freq*(tt == 0)
                    vzero += freq*(vv == 0)
                    bothzero += freq*(tt == 0 and vv == 0)
                    for idx, z in enumerate([tt, vv, tt*tt, vv*vv, tt*vv]):
                        moments[idx] += freq*z
                    if tt and vv:
                        untied[2*int(tt > 0)+int(vv > 0)] += freq
                assert ncount == x["n"] == CFG["observations_per_stream"]
                assert x["raw_moments"] == moments
                assert x["ties"] == dict(T=tzero, V=vzero, both=bothzero)
                assert untied == x["no_tie_sign_joint"]
                assert sum(x["any_tie_sign_joint"]) == tzero+vzero-bothzero
                assert all(v >= 0 and isinstance(v, int) for v in x["any_tie_sign_joint"])
                rebuilt = [a+b for a, b in zip(untied, x["any_tie_sign_joint"])]
                assert rebuilt == x["sign_joint"]
                k = rebuilt[0]+rebuilt[3]
                assert k == x["successes"]
                rr = mapping[(prof["name"], d, j)]
                for key in ["seed", "n", "successes"]:
                    assert int(rr[key]) == x[key]
                for key, val in zip(["tie_T", "tie_V", "tie_both"], [tzero, vzero, bothzero]):
                    assert int(rr[key]) == val
                assert float(rr["elapsed_seconds"]) == x["elapsed_seconds"]
                for idx, key in enumerate(["gaussian_draws", "bernoulli_uniform_draws", "tie_uniform_draws"]):
                    expected = ncount*d if idx < 2 else 2*ncount
                    assert x["primitives"][key] == int(rr[key]) == expected
                    primitive_totals[idx] += expected
                ktotal += k
                ntotal += ncount
            aggregated[(prof["name"], d)] = (ktotal, ntotal)
    tail = .05/24  # independent fixed 12-family tail
    audited_rows = []
    max_abs_cp_difference = 0.
    for r in result["rows"]:
        k, n = aggregated[(r["profile"], r["d"])]
        # Invert exact binomial tails, without beta.ppf or the production CP helper.
        lo = 0. if k == 0 else brentq(lambda p: binom.sf(k-1, n, p)-tail,
                                      0., 1., xtol=2e-14, rtol=4*np.finfo(float).eps)
        hi = 1. if k == n else brentq(lambda p: binom.cdf(k, n, p)-tail,
                                     0., 1., xtol=2e-14, rtol=4*np.finfo(float).eps)
        assert (k, n) == (r["successes"], r["n"])
        diff = max(abs(lo-r["cp95_bonferroni_lower"]), abs(hi-r["cp95_bonferroni_upper"]))
        max_abs_cp_difference = max(max_abs_cp_difference, diff)
        assert diff < 2e-11
        nearest = .5 if lo <= .5 <= hi else (lo if lo > .5 else hi)
        # Independent KL expression: log(2) minus binary entropy.
        lower = math.log(2.)+nearest*math.log(nearest)+(1-nearest)*math.log(1-nearest)
        if nearest == .5:
            lower = 0.
        assert abs(lower-r["event_kl_simultaneous_lower"]) < 2e-13
        isprimary = r["d"] == 128 and r["profile"] in ("mild", "contrast")
        assert r["primary"] == isprimary
        if isprimary:
            assert r["primary_pass"] == (lower > 0)
        assert r["p_hat"] == k/n
        audited_rows.append(dict(profile=r["profile"], d=r["d"], n=n, successes=k,
            cp_binomial_inversion=[lo, hi], independently_computed_information_lower=lower,
            primary=isprimary, primary_pass=(lower > 0 if isprimary else None)))
    assert len(audited_rows) == 12
    assert sum(n for _, n in aggregated.values()) == 6291456 == result["all_observations"]
    assert primitive_totals == [1132462080, 1132462080, 12582912]
    assert result["primary_pass_all"] == all(r["primary_pass"] for r in audited_rows if r["primary"])
    return dict(created_utc=utc(), pass_all=True, streams_checked=96,
                histogram_arithmetic_exact=True, rng_seeds_unique=len(seeds),
                cp_method="independent scipy.stats.binom tail inversion with brentq",
                max_abs_CP_difference=max_abs_cp_difference,
                total_observations=6291456, total_primitives=primitive_totals,
                rows=audited_rows,
                limitation="audits saved aggregates and RNG contract; no new observations or complete-label replay")


def report(results, audit):
    rows = results["rows"]
    lines = [
        "# E 结果：错误完整对称化删除的组分配信息",
        "",
        "本包属于后续 v2；已封存 v1 和前轮非交换解析包均未改。固定设计为 12 格、96 个独立 RNG 流、6,291,456 个独立观察，0 拟合、0 追加。最新 HANDOFF 的候选证明经过独立补齐，见 MISMATCH_THEORY.md；冻结的原始输入及哈希见 E/FREEZE.json。",
        "",
        "唯一随机化事件是总自旋 T 与组间自旋对比 V 同号。两个零值分别使用独立公平 coin，所以逐观察仍是 Bernoulli 变量。完整对称化分布下事件概率对每个有限 d 精确为 1/2；下表区间为12项 Bonferroni 同时95% Clopper–Pearson 区间。",
        "",
        "|profile|d|成功数 / 524288|事件概率|同时 CP 区间|完整 KL 的事件下界 (nats)|极限概率预测|",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"|{r['profile']}|{r['d']}|{r['successes']}|{r['p_hat']:.9f}|"
                     f"[{r['cp95_bonferroni_lower']:.9f}, {r['cp95_bonferroni_upper']:.9f}]|"
                     f"{r['event_kl_simultaneous_lower']:.9g}|{r['asymptotic_prediction']:.9f}|")
    lines += ["", "## 预定主终点与解释", ""]
    for r in rows:
        if r["primary"]:
            status = "通过" if r["primary_pass"] else "未通过"
            lines.append(f"- d=128、{r['profile']}：信息下界 {r['event_kl_simultaneous_lower']:.10g} nats，严格大于0主门{status}。")
    lines += [
        "",
        "没有使用任意实质大小阈值，也没有把其余维度替代主终点。d=512 为预定描述；null 对照全部保留。区间包含1/2不是一般无效或等效证明。微小正下界不能称为大实际影响。",
        "",
        "这些数值是有限维、完整模式 KL 的一个粗事件下界，不是完整 KL 点估计。完整对称化精确保留总标签计数分布，因此只看总计数会漏掉这里识别的组分配信息。固定异质性下非消失的解析结论与有限维实验相互补充；quad只给 d→∞ 极限预测，未当作有限维真概率。",
        "",
        "理论中小异质性的二次展开仅针对该粗事件下界；没有声称完整KL的精确二次尺度、任务收益、优化器行为或自然数据方法获胜。两组已知、等大、尺度固定且正是必要合同。",
        "",
        "## 数值与独立算术核验", "",
        f"固定参数双表达quad及尾界、d=4全模式/coin枚举和Gaussian投影协方差smoke全部通过。独立审计逐流用组计数稀疏直方图重算非平局符号、平局数量及T/V矩，结合实际coin计数恢复事件数；96流和12格总数完全一致。",
        "",
        f"独立以二项尾反演复算CP，和生产beta分位数结果最大绝对差 {audit['max_abs_CP_difference']:.3g}；12个信息下界及两个主判定一致。参数积分误差门是1e-10，统计区间反映独立观察采样，不是8次拟合或流间t区间。完整初末PCG64状态、batch和抽样顺序保留，未保存全部6M标签，也未重跑科学样本。",
        "",
        f"累计流采样墙钟 {sum(r['elapsed_seconds'] for r in rows):.3f} 秒；固定primitive总数为1,132,462,080个Gaussian、同数Bernoulli uniforms和12,582,912个coin uniforms。仅CPU、至多2线程。",
        "",
        "产物：artifacts/structure_tradeoff_v2/E/ 中的 CONFIG、FREEZE、SMOKE、STREAM_COUNTS、每流JSON、RESULTS、INDEPENDENT_AUDIT 与 RESULTS_FREEZE；源码 script/v2_symmetry_events.py。只读核验使用 .venv/Scripts/python -X utf8 script/v2_symmetry_events.py --check。没有新队列。", ""
    ]
    path = DOC / "EVENT_RESULTS.md"
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def run():
    frozen_guard()
    assert not (HERE / "SMOKE.json").exists(), "existing run; no silent restart"
    start = time.perf_counter()
    try:
        smoke = deterministic_smoke()
        write_new(HERE / "SMOKE.json", smoke)
        print("SMOKE passed; starting 96 frozen streams", flush=True)
        records = []
        for pi, prof in enumerate(CFG["profiles"]):
            for d in CFG["dimensions"]:
                for stream in range(CFG["streams"]):
                    out = sample_stream(pi, prof, d, stream)
                    write_new(stream_path(prof["name"], d, stream), out)
                    records.append(out)
                    print(f"stream {len(records):02d}/96 {prof['name']} d={d} "
                          f"s={stream} n={out['n']} elapsed={out['elapsed_seconds']:.3f}s",
                          flush=True)
        write_counts_csv(records)
        results = collect()
        write_new(HERE / "RESULTS.json", results)
        audit = independent_audit()
        write_new(HERE / "INDEPENDENT_AUDIT.json", audit)
        report(results, audit)
        frozen_guard()
        files = [p for p in HERE.rglob("*") if p.is_file()] + [DOC / "EVENT_RESULTS.md"]
        write_new(HERE / "RESULTS_FREEZE.json", dict(created_utc=utc(),
            source_freeze_sha256=sha(HERE / "FREEZE.json"),
            files=[file_record(p) for p in sorted(files)],
            wall_seconds=time.perf_counter()-start,
            all_science_complete=True, all_independent_audit_passed=True,
            primary_pass_all=results["primary_pass_all"],
            new_observations=results["all_observations"], fits=0))
        print(json.dumps(dict(complete=True, primary_pass_all=results["primary_pass_all"],
                              wall_seconds=time.perf_counter()-start)), flush=True)
    except Exception as err:
        failure = HERE / ("FAILURE_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json")
        write_new(failure, dict(created_utc=utc(), error=repr(err),
            source_sha256=sha(__file__), completed_streams=len(list(HERE.glob("streams/*.json"))),
            action="stopped; no new seeds/dimensions/parameters or restart automatically"))
        raise


def check():
    frozen_guard()
    final = read(HERE / "RESULTS_FREEZE.json")
    assert final["source_freeze_sha256"] == sha(HERE / "FREEZE.json")
    guard_files(final["files"])
    audit = independent_audit()
    assert final["all_science_complete"] and final["all_independent_audit_passed"]
    print(json.dumps(dict(check_pass=True, streams=audit["streams_checked"],
        observations=audit["total_observations"],
        cp_max_difference=audit["max_abs_CP_difference"],
        primary_pass_all=final["primary_pass_all"]), ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
    elif args.check:
        check()
    else:
        print(json.dumps(independent_audit(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
