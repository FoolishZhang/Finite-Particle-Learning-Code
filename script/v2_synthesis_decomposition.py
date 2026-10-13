"""Frozen post-hoc KL decomposition and fixed analytic substitutions; no fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, ndtr, roots_hermitenorm
from structure_learning_mapping import swap_kernel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v2_synthesis_20260930/decomposition"
DOC = ROOT / "docs/08_research_研究工作/v2收束与v3方向_20260930"
OLD_DOC = ROOT / "docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930"
C = ROOT / "artifacts/structure_learning_v2/C"
B = ROOT / "artifacts/structure_learning_v2/B"
CFG = dict(profile="contrast", N=65536, method="GLOBAL_G10", streams=16,
    d=16, c=.9, teacher_scales=[6., 2.], expected_merged=10, expected_unmerged=6,
    reviewed_commit="cab00907ffa5c676c2cd306e19d2fb052b24f76f",
    handoff_sha256="cc8194467fca06b4d0cc2351eab1d64ce31f9fc70131d66d13f1cffed970d64b",
    handoff_usage="reference; post-hoc derived diagnostic",
    identity_gate=1e-10, reference_sensitivity_gate=1e-7,
    a1_scales=[6., 2.], quad_upper=12., quad_epsabs=2e-13, quad_epsrel=2e-13,
    quad_limit=300, quad_error_gate=1e-11, a1_agreement_gate=1e-10,
    GH_orders=[2048, 4096], sufficient_N_dimensions=[16, 128],
    Fano_N=[256, 4096, 65536], new_samples=0, new_fits=0, new_banks=0)
BASE_COMPONENTS = ["full_joint_KL", "total_count_KL", "conditional_pattern_KL",
                   "within_selected_group_orbit_KL", "conditional_group_allocation_KL"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, obj, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, default=native, allow_nan=False)
        stream.write("\n")


def csvwrite(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_rows():
    with (C / "FITS.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if
                r["profile"] == "contrast" and r["N"] == "65536" and r["method"] == "GLOBAL_G10"]
    rows.sort(key=lambda row: int(row["stream"]))
    assert [int(row["stream"]) for row in rows] == list(range(16))
    assert sum(float(row["u"]) == float(row["v"]) for row in rows) == 10
    return rows


def verify_files(files):
    for name, expected in files.items():
        assert name != "HANDOFF.md", "do not bind the live review after snapshot"
        assert sha(ROOT / name) == expected, f"input hash mismatch: {name}"


def freeze():
    assert not OUT.exists(), "new output directory required"
    assert sha(ROOT / "HANDOFF.md") == CFG["handoff_sha256"]
    rows = selected_rows()
    indices = sorted({232} | {int(row["candidate_index"]) for row in rows})
    refs = [B / f"reference/g{i:03}.json" for i in indices]
    streams = [C / f"streams/contrast_s{s:02}.json" for s in range(16)]
    # Targeted membership checks against the old scientific receipts.
    bseal = read(B / "REFERENCE_FREEZE.json")["files"]
    cseal = read(C / "RESULTS_FREEZE.json")["files"]
    for path in refs:
        assert bseal[rel(path)] == sha(path)
    for path in streams+[C / "FITS.csv", C / "RISK_LIBRARY.json"]:
        assert cseal[rel(path)] == sha(path)
    OUT.mkdir(parents=True)
    (OUT / "source_inputs").mkdir()
    (OUT / "source_inputs/HANDOFF.md").write_bytes((ROOT / "HANDOFF.md").read_bytes())
    write(OUT / "CONFIG.json", CFG)
    write(OUT / "ENVIRONMENT.json", dict(python=sys.version, numpy=np.__version__,
        scipy=scipy.__version__, platform=platform.platform(), threads=2))
    inputs = [Path(__file__), ROOT / "script/structure_learning_mapping.py",
        DOC / "DECOMPOSITION_PLAN.md", OUT / "CONFIG.json", OUT / "ENVIRONMENT.json",
        OUT / "source_inputs/HANDOFF.md", C / "FITS.csv", C / "RISK_LIBRARY.json",
        C / "RESULTS_FREEZE.json", B / "REFERENCE_FREEZE.json",
        ROOT / "artifacts/structure_learning_v2/MANIFEST.json",
        OLD_DOC / "SAMPLE_VS_PARTICLE_SEPARATION.md", OLD_DOC / "E_REVIEW.md"]+refs+streams
    write(OUT / "FREEZE.json", dict(time_utc=now(),
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        reviewed_commit=CFG["reviewed_commit"], review_usage=CFG["handoff_usage"],
        files={rel(p): sha(p) for p in inputs}, candidate_indices=indices,
        before_decomposition_or_a1_integration=True, live_HANDOFF_excluded=True))
    write(OUT / "STATE.json", dict(status="frozen", active_jobs=[], time_utc=now()))
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / "FREEZE.json"))), flush=True)


def guard():
    assert read(OUT / "CONFIG.json") == CFG
    verify_files(read(OUT / "FREEZE.json")["files"])


def geometry():
    a = np.repeat(np.arange(9), 9)
    b = np.tile(np.arange(9), 9)
    total = a+b
    mult = np.array([math.comb(8, int(x))*math.comb(8, int(y)) for x, y in zip(a, b)], dtype=float)
    all_mult = np.array([math.comb(16, int(n)) for n in total], dtype=float)
    indicator = (total[:, None] == np.arange(17)[None, :]).astype(float)
    return total, mult, all_mult, indicator


def entropy(prob):
    return -math.fsum(float(p)*math.log(float(p)) for p in prob)


def kl(p, q):
    return math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(p, q))


def components(p_original, p_group, q_group, merged):
    total, mult, all_mult, indicator = geometry()
    for vector in (p_original, p_group, q_group):
        assert len(vector) == 81 and np.all(np.asarray(vector) > 0)
        assert abs(float(np.sum(vector))-1) < 1e-12
    H = entropy(p_original)+float(p_original @ np.log(mult))
    H_orbit = entropy(p_group)+float(p_group @ np.log(mult))
    pN, qN = p_group @ indicator, q_group @ indicator
    full = float(p_group @ (-np.log(q_group)+np.log(mult)))-H
    count = kl(pN, qN)
    orbit = H_orbit-H
    allocation = math.fsum(float(p_group[i])*math.log(
        float((p_group[i]/pN[total[i]])/(q_group[i]/qN[total[i]]))) for i in range(81))
    conditional = orbit+allocation
    double_count = kl(p_group, q_group)
    # Independent slice expression for E_N D(P(C|N)||Q(C|N)).
    q_conditional_cross_entropy = 0.
    for n in range(17):
        ix = np.flatnonzero(total == n)
        conditional_p_group = p_group[ix]/pN[n]
        log_q_full_given_n = np.log(q_group[ix])-np.log(mult[ix])-math.log(float(qN[n]))
        q_conditional_cross_entropy -= float(pN[n])*float(conditional_p_group @ log_q_full_given_n)
    conditional_alt = q_conditional_cross_entropy-(H-entropy(pN))
    full_sym_group = pN[total]*mult/all_mult
    H_sym = entropy(pN)+float(pN @ np.log([math.comb(16, n) for n in range(17)]))
    Dsym = H_sym-H
    residual = max(abs(full-count-conditional), abs(full-orbit-double_count),
                   abs(conditional-conditional_alt))
    result = dict(full_joint_KL=full, total_count_KL=count,
        conditional_pattern_KL=conditional, within_selected_group_orbit_KL=orbit,
        conditional_group_allocation_KL=allocation, double_count_KL=double_count,
        teacher_Dsym=Dsym, chain_identity_error=residual,
        merged_Dsym=None, merged_D_SP_to_Q=None, merged_identity_error=None)
    if merged:
        qsym_group = qN[total]*mult/all_mult
        assert np.max(abs(qsym_group-q_group)) < CFG["identity_gate"]
        reverse_piece = kl(full_sym_group, q_group)
        error = max(abs(full-Dsym-reverse_piece), abs(reverse_piece-count),
                    abs(conditional-Dsym))
        assert error <= CFG["identity_gate"]
        result.update(merged_Dsym=Dsym, merged_D_SP_to_Q=reverse_piece,
                      merged_identity_error=error)
    assert residual <= CFG["identity_gate"]
    assert min(full, count, conditional, orbit, allocation) > -CFG["identity_gate"]
    return result


def decompose():
    library = read(C / "RISK_LIBRARY.json")["contrast"]
    teacher = read(B / "reference/g232.json")
    assert teacher["qualified"]
    p0 = np.asarray(library["teacher_count_p"], dtype=float)
    pv = np.asarray(teacher["variants"], dtype=float)
    assert np.max(abs(p0-pv[2])) < 1e-12
    total, mult, _, _ = geometry()
    H = entropy(p0)+float(p0 @ np.log(mult))
    assert abs(H-library["H"]) < CFG["identity_gate"]
    results, sensitivity_rows = [], []
    for source in selected_rows():
        stream = int(source["stream"])
        index = int(source["candidate_index"])
        qref = read(B / f"reference/g{index:03}.json")
        assert qref["qualified"]
        qv = np.asarray(qref["variants"], dtype=float)
        u, v = float(source["u"]), float(source["v"])
        assert qref["scales"] == [u, v]
        true_mask, assumed_mask = int(source["true_mask"]), int(source["selected_mask"])
        receipt = read(C / f"streams/contrast_s{stream:02}.json")
        assert receipt["true_mask"] == true_mask
        r = 8-(true_mask & assumed_mask).bit_count()
        assert r == int(source["overlap_swap_r"])
        W = np.asarray(swap_kernel(r)["kernel"])
        p = p0 @ W
        assert np.max(abs(p-library["mapped_teacher_p"][r])) < CFG["identity_gate"]
        merged = u == v
        value = components(p0, p, qv[2], merged)
        errors = [abs(value["full_joint_KL"]-float(source["deployed_KL"])),
                  abs(value["total_count_KL"]-float(source["total_count_KL"])),
                  abs(value["full_joint_KL"]-library["candidate_joint_KL_by_swap"][r][index]),
                  abs(value["teacher_Dsym"]-library["count_only_bound"])]
        assert max(errors) <= CFG["identity_gate"]
        variants = [components(pp, pp @ W, qq, merged) for pp in pv for qq in qv]
        keys = BASE_COMPONENTS+(["merged_Dsym", "merged_D_SP_to_Q"] if merged else [])
        sensitivity = max(float(np.ptp([row[key] for row in variants])) for key in keys)
        assert sensitivity <= CFG["reference_sensitivity_gate"]
        results.append(dict(stream=stream, candidate_index=index, u=u, v=v,
            merged=merged, category="merged" if merged else "unmerged",
            true_mask=true_mask, selected_mask=assumed_mask, swap_r=r, **value,
            original_C_joint_KL=float(source["deployed_KL"]),
            original_C_total_count_KL=float(source["total_count_KL"]),
            source_replay_error=max(errors), cross_reference_sensitivity=sensitivity))
        sensitivity_rows.append(dict(stream=stream, candidate_index=index,
            max_component_range=sensitivity, teacher_variants=3, candidate_variants=3))
    assert len(results) == 16 and sum(row["merged"] for row in results) == 10
    summaries = []
    for group in ("all", "merged", "unmerged"):
        data = results if group == "all" else [row for row in results if row["category"] == group]
        keys = BASE_COMPONENTS+(["merged_Dsym", "merged_D_SP_to_Q"] if group == "merged" else [])
        for key in keys:
            values = [float(row[key]) for row in data]
            summaries.append(dict(group=group, n=len(data), component=key,
                mean=math.fsum(values)/len(values), minimum=min(values), maximum=max(values)))
    return results, summaries, sensitivity_rows


def analytic_substitution(Dsym, teacher_Dsym_variants):
    sH, sL, c = 6., 2., .9
    invsqrt2pi = 1/math.sqrt(2*math.pi)
    cutoff = CFG["quad_upper"]
    coefficients = {}
    teacher = read(B / "reference/g232.json")
    inherited = {float(r["scale"]): r["coefficients"][0]
                 for r in teacher["independent_Hermite_moments"]["coefficient_sources"]}
    for s in (sH, sL):
        direct = quad(lambda z: 2*z*math.tanh(s*z/2)*math.exp(-z*z/2)*invsqrt2pi,
                      0., cutoff, epsabs=CFG["quad_epsabs"], epsrel=CFG["quad_epsrel"],
                      limit=CFG["quad_limit"], full_output=1)
        stein = quad(lambda z: 4*s*expit(s*z)*expit(-s*z)*math.exp(-z*z/2)*invsqrt2pi,
                     0., cutoff, epsabs=CFG["quad_epsabs"], epsrel=CFG["quad_epsrel"],
                     limit=CFG["quad_limit"], full_output=1)
        assert len(direct) == len(stein) == 3, "fixed a1 quadrature did not converge"
        tail_direct = 2*math.exp(-cutoff*cutoff/2)*invsqrt2pi
        tail_stein = s*float(ndtr(-cutoff))
        assert max(direct[1]+tail_direct, stein[1]+tail_stein) <= CFG["quad_error_gate"]
        values = dict(quad=float(direct[0]), stein_quad=float(stein[0]))
        for order in CFG["GH_orders"]:
            nodes, weights = roots_hermitenorm(order)
            weights = weights*invsqrt2pi
            assert abs(float(weights.sum())-1) < 1e-12
            values["GH"+str(order)] = float(weights @ (nodes*np.tanh(s*nodes/2)))
        disagreement = max(values.values())-min(values.values())
        assert disagreement <= CFG["a1_agreement_gate"]
        assert abs(values["quad"]-inherited[s]) <= CFG["a1_agreement_gate"]
        coefficients[str(s)] = dict(values=values, inherited_reference_value=inherited[s],
            quad_error_estimate=direct[1], stein_error_estimate=stein[1],
            quad_tail_bound=tail_direct, stein_tail_bound=tail_stein,
            max_algorithm_disagreement=disagreement,
            quad_evaluations=int(direct[2]["neval"]), stein_evaluations=int(stein[2]["neval"]))
    M = .5*((sH/sL)**2*(1+c/3)/(1-c)-1)
    constants = {}
    for method in coefficients["6.0"]["values"]:
        Delta = coefficients["6.0"]["values"][method]**2-coefficients["2.0"]["values"][method]**2
        constants[method] = dict(Delta=Delta, kappa=c*Delta/6, M=M)
    sufficient = []
    for d in CFG["sufficient_N_dimensions"]:
        calculated = {}
        for name, values in constants.items():
            assert (c/(d-1))**2 <= values["Delta"]/2
            raw_N = 9*d/values["kappa"]**2*(4*math.log(d)+math.log1p(M))
            calculated[name] = dict(unrounded_N=raw_N, ceil_N=math.ceil(raw_N))
        sufficient.append(dict(d=d, dimension_condition_pass=True, by_algorithm=calculated,
            all_ceil_values_equal=len({row["ceil_N"] for row in calculated.values()}) == 1,
            expected_total_KL_upper=M/(4*(1+M)*d),
            simpler_expected_total_KL_upper=1/(4*d),
            known_scales_c_and_group_size=True))
    fano = []
    log_groups = math.log(math.comb(16, 8))
    for n in CFG["Fano_N"]:
        raw = 1-(n*Dsym+math.log(2))/log_groups
        variants = [1-(n*x+math.log(2))/log_groups for x in teacher_Dsym_variants]
        fano.append(dict(d=16, N=n, teacher_Dsym=Dsym, log_group_count=log_groups,
            raw_lower=raw, error_probability_lower=max(0., raw),
            bound_uninformative=(raw <= 0), raw_by_teacher_reference_variant=variants,
            target="uniform-prior/minimax exact group recovery; not density-estimation risk"))
    return dict(coefficients=coefficients, constants=constants, primary_algorithm="quad",
        sufficient_sample_sizes=sufficient, Fano=fano,
        posthoc_formula_evaluation_not_new_confirmation=True,
        new_Gaussian_scalar_coefficient_problems=2, new_full_joint_reference_integrals=0)


def audit_outputs():
    guard()
    rows, summary, _ = decompose()
    stored = read(OUT / "RESULTS.json")
    max_error = 0.
    assert len(stored["rows"]) == 16 and len(stored["group_summary"]) == len(summary)
    for actual, expected in zip(stored["rows"], rows):
        for key, value in expected.items():
            if isinstance(value, (float, np.floating)):
                max_error = max(max_error, abs(actual[key]-value))
            else:
                assert actual[key] == value
    for actual, expected in zip(stored["group_summary"], summary):
        for key, value in expected.items():
            if isinstance(value, float):
                max_error = max(max_error, abs(actual[key]-value))
            else:
                assert actual[key] == value
    analytic = read(OUT / "ANALYTIC_SUBSTITUTION.json")
    for row in analytic["sufficient_sample_sizes"]:
        for method, const in analytic["constants"].items():
            raw = 9*row["d"]/const["kappa"]**2*(4*math.log(row["d"])+math.log1p(const["M"]))
            assert row["by_algorithm"][method]["ceil_N"] == math.ceil(raw)
    for row in analytic["Fano"]:
        expected = 1-(row["N"]*row["teacher_Dsym"]+math.log(2))/math.log(math.comb(16, 8))
        max_error = max(max_error, abs(row["raw_lower"]-expected))
        assert row["error_probability_lower"] == max(0., row["raw_lower"])
    assert max_error <= CFG["identity_gate"]
    return dict(passed=True, time_utc=now(), rows=16, merged=10, unmerged=6,
        replay_error=max_error, check_does_not_repeat_scalar_integrals=True,
        source_and_two_chain_expressions_verified=True, no_new_samples_or_fits=True,
        scope="small derived-arithmetic self-check; not an old full-experiment rerun")


def report(rows, summaries, analytic):
    lookup = {(r["group"], r["component"]): r for r in summaries}
    lines = ["# 已保存G10输出的联合风险分解", "",
        "本页是最新HANDOFF之后的事后诊断，0新观察、0拟合、0bank。原C contrast/N65536的16个GLOBAL_G10全部保留，10个merged、6个unmerged；不将约.019576的总均值统一叫作identity loss。", "",
        "每例采用完整链式：joint KL=total-count KL+给定total count后的完整模式条件KL。后者又保留selected-group轨道内残差与双count条件分配KL；它们是分布错配项，不是因果份额。", "",
        "|组|n|完整KL：均值 [最小,最大]|总count KL：均值 [最小,最大]|条件模式KL：均值 [最小,最大]|",
        "|---|---:|---|---|---|"]
    for group in ("all", "merged", "unmerged"):
        cells = []
        for key in BASE_COMPONENTS[:3]:
            x = lookup[(group, key)]
            cells.append(f"{x['mean']:.10g} [{x['minimum']:.10g}, {x['maximum']:.10g}]")
        lines.append(f"|{group}|{lookup[(group,BASE_COMPONENTS[0])]['n']}|"+"|".join(cells)+"|")
    lines += ["", "仅merged另有D(P||Q)=D(P||SP)+D(SP||Q)，后一项等于total-count KL。unmerged未套用这条固定identity归因；全部细分列及组均值/范围分别见DECOMPOSITION.csv与GROUP_SUMMARY.csv。", "",
        "|stream|u,v|merged|r|joint KL|total-count KL|conditional-pattern KL|within-orbit KL|conditional-group KL|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['stream']}|{r['u']:g},{r['v']:g}|{r['merged']}|{r['swap_r']}|"
            +"|".join(f"{r[k]:.12g}" for k in BASE_COMPONENTS)+"|")
    lines += ["", "## 旧E公式的事后代入", "",
        "两尺度a1由quad/Stein/GH2048/GH4096交叉核对后代入原公式；不是新训练或新确认。c、两尺度与组大小已知的保守充分样本量，不预测实际65536观察的成功率。", "",
        "|d|主quad给出的充分N|四算法ceil一致|期望总KL上界|", "|---:|---:|---|---:|"]
    for r in analytic["sufficient_sample_sizes"]:
        lines.append(f"|{r['d']}|{r['by_algorithm']['quad']['ceil_N']}|{r['all_ceil_values_equal']}|{r['expected_total_KL_upper']:.10g}|")
    lines += ["", "Fano只针对均匀未知高组先验下的精确恢复；0表示下界无信息，不是错误率0。数值依赖旧qualified教师，未新增置信区间。", "",
        "|d|N|未截断右端|恢复错误率下界|", "|---:|---:|---:|---:|"]
    for r in analytic["Fano"]:
        lines.append(f"|16|{r['N']}|{r['raw_lower']:.10g}|{r['error_probability_lower']:.10g}|")
    lines += ["", "所有分量均以完整向量nats计。G10使用结构化group-count分量，其有限目标偏移并非旧product-mixture指数界的直接实验验证；这里也没有将有限bank选点当作精确F10的必然最优。", ""]
    (OUT / "READABLE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run():
    guard()
    assert read(OUT / "STATE.json")["status"] == "frozen", "no silent restart"
    rows, summaries, sensitivity = decompose()
    csvwrite(OUT / "DECOMPOSITION.csv", rows)
    csvwrite(OUT / "GROUP_SUMMARY.csv", summaries)
    write(OUT / "RESULTS.json", dict(rows=rows, group_summary=summaries, sensitivities=sensitivity,
        posthoc=True, new_samples=0, new_fits=0, new_banks=0))
    teacher = read(B / "reference/g232.json")
    Dsym_variants = [components(np.array(p), np.array(p), np.array(p), False)["teacher_Dsym"]
                     for p in teacher["variants"]]
    analytic = analytic_substitution(rows[0]["teacher_Dsym"], Dsym_variants)
    write(OUT / "ANALYTIC_SUBSTITUTION.json", analytic)
    write(OUT / "AUDIT.json", audit_outputs())
    report(rows, summaries, analytic)
    outputs = [OUT / name for name in ("DECOMPOSITION.csv", "GROUP_SUMMARY.csv", "RESULTS.json",
               "ANALYTIC_SUBSTITUTION.json", "AUDIT.json", "READABLE.md")]
    write(OUT / "RESULTS_FREEZE.json", dict(time_utc=now(), files={rel(p): sha(p) for p in outputs},
        source_freeze_sha256=sha(OUT / "FREEZE.json")))
    write(OUT / "STATE.json", dict(status="complete", active_jobs=[], time_utc=now()), replace=True)
    print(json.dumps(dict(complete=True, rows=16, merged=10, unmerged=6, new_samples=0, new_fits=0)), flush=True)


def check():
    guard()
    seal = read(OUT / "RESULTS_FREEZE.json")
    verify_files(seal["files"])
    assert seal["source_freeze_sha256"] == sha(OUT / "FREEZE.json")
    assert read(OUT / "STATE.json")["status"] == "complete"
    print(json.dumps(audit_outputs()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    try:
        freeze() if args.phase == "freeze" else run() if args.phase == "run" else check()
    except Exception as error:
        if args.phase == "run" and not (OUT / "RESULTS_FREEZE.json").exists():
            write(OUT / f"FAILURE_{time.time_ns()}.json", dict(time_utc=now(),
                error=repr(error), traceback=traceback.format_exc()))
            write(OUT / "STATE.json", dict(status="stopped_failure", active_jobs=[], time_utc=now()), replace=True)
        raise
