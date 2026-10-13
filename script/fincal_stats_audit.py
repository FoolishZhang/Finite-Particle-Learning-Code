"""Independent frozen-family statistics and exploratory quadratic replay audit.

Uses fsum, a beta-CDF bisection for t critical values, original output rows and
saved raw NPZ curves. No imports from production/statistics/replay functions.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import betainc, xlogy


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/finite_calibration"
TRAIN = BASE / "TRAIN"
POST = BASE / "POSTHOC_QUADRATIC"
SOURCE = Path(__file__).resolve()
PROTOCOL = TRAIN / "STATS_AUDIT_PROTOCOL.json"
RESULT = TRAIN / "STATS_AUDIT.json"
TOL = 1e-11


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(test, message):
    if not test:
        raise RuntimeError(message)


def freeze():
    require(not PROTOCOL.exists() and not RESULT.exists(), "Statistics audit already frozen/run")
    training = read(TRAIN / "FREEZE.json")
    result = read(TRAIN / "RESULTS.json")
    paths = [TRAIN / name for name in ("FREEZE.json", "RESULTS.json", "STATISTICS_PROTOCOL.json", "STATISTICS.json", "AUDIT.json")]
    paths += [POST / name for name in ("PROTOCOL.json", "EXECUTION_PROTOCOL.json", "RESULTS.json")]
    paths += [BASE / "CAL/MODEL.json", ROOT / "script/fincal_statistics.py", ROOT / "script/fincal_posthoc.py"]
    report = (ROOT / training["plan_path"]).parent / "STATISTICS_AUDIT.md"
    require(not report.exists(), "Statistics audit report exists")
    save_new(PROTOCOL, {
        "schema": "fincal-statistics-independent-audit-v1", "source_sha256": sha(SOURCE),
        "input_hashes": {rel(path): sha(path) for path in paths},
        "npz_hashes": {row["file"]: row["file_sha256"] for row in result["records"]},
        "report": rel(report), "independence": "Different internal implementation; no producer function imports; non-blind",
        "numerical_absolute_tolerance": TOL,
        "statistics": "math.fsum mean and centered variance over 16 original output rows; t quantiles by incomplete-beta CDF bisection, df15",
        "primary_scope": "Original four Bonferroni paired contrasts only; preserve scientific pass and fail separately from audit correctness",
        "posthoc_scope": "96 reselected saved raw curves, six descriptive units, paired mean differences and same-point counts; no new gate",
        "coefficient_source": "Only n254 frozen MODEL gprime cubic derivative at zero divided by 2; verify definition PROTOCOL and execution hashes",
        "posthoc_bounds": "Fraction checks of base D plus 2 abs(c), and D*h^2/8; no point-evaluation or integral certificate",
        "new_random_draws": 0, "new_scientific_samples": 0, "new_Gaussian_integrals": 0,
        "original_primary_family_modified": False, "automatic_retry": False, "preserve_all_failures": True,
    })
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


class Checkbook:
    def __init__(self):
        self.numeric, self.exact = [], []

    def number(self, name, got, want, tolerance=TOL):
        got, want = np.asarray(got), np.asarray(want)
        require(got.shape == want.shape, "Shape mismatch: "+name)
        finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
        diff = np.abs(got-want)
        self.numeric.append({"name": name, "count": int(diff.size), "absolute_tolerance": tolerance,
                             "maximum_absolute_difference": float(diff.max()) if diff.size and finite else None,
                             "failures": int(np.count_nonzero(diff > tolerance)) if finite else int(diff.size), "all_finite": finite})

    def flag(self, name, value):
        self.exact.append({"name": name, "passed": bool(value)})


def t_quantile(probability, df=15):
    """Invert F(t)=1-I_{df/(df+t^2)}(df/2,1/2)/2 for t>=0."""
    require(.5 < probability < 1, "Positive t quantile only")
    lo, hi = 0.0, 16.0
    require(1-.5*betainc(df/2, .5, df/(df+hi*hi)) >= probability, "Fixed t bracket insufficient")
    for _ in range(64):
        mid = (lo+hi)/2
        cdf = 1-.5*float(betainc(df/2, .5, df/(df+mid*mid)))
        if cdf < probability:
            lo = mid
        else:
            hi = mid
        if hi-lo <= 5e-15:
            break
    return (lo+hi)/2


def summarize(values, critical):
    values = [float(v) for v in values]
    n = len(values)
    mean = math.fsum(values)/n
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1)/n)
    return {"n": n, "values": values, "mean": mean, "SE": se,
            "lower": mean-critical*se, "upper": mean+critical*se, "zero_observed_SE": se == 0}


def check_summary(checks, name, got, expected):
    for key in ("values", "mean", "SE", "lower", "upper"):
        checks.number(name+"/"+key, got[key], expected[key])
    checks.flag(name+"/sample_count", got["n"] == expected["n"] == 16)
    checks.flag(name+"/zero_SE_flag", got["zero_observed_SE"] == expected["zero_observed_SE"])


def main_statistics(production, statistics, protocol_data, model, checks):
    rows = [row for stream in production["records"] for row in stream["outputs"]]
    lookup = {(row["stream"], row["theta"], row["N"], row["method"]): row for row in rows}
    checks.flag("primary_input/672_unique_rows", len(rows) == len(lookup) == 672)
    critical = t_quantile(.975)
    family = protocol_data["family"]
    checks.flag("primary_input/original_family_contract", family["alpha"] == .05 and family["contrasts"] == 4 and family["N"] == 4096)
    family_critical = t_quantile(1-family["alpha"]/(2*family["contrasts"]))
    checks.number("t_critical/descriptive", critical, statistics["t_critical_descriptive"])
    checks.number("t_critical/family", family_critical, statistics["t_critical_family"])
    reconstructed = []
    by_cell = {(r["theta"], r["N"], r["method"]): r for r in statistics["descriptive"]}
    checks.flag("descriptive/42_unique_cells", len(by_cell) == len(statistics["descriptive"]) == 42)
    for key, old in by_cell.items():
        theta, N, method = key
        chosen = [lookup[(stream, theta, N, method)] for stream in range(16)]
        new = summarize([r["reference_joint_KL"] for r in chosen], critical)
        name = f"description/{theta}/{N}/{method}"
        check_summary(checks, name, new, old)
        rhos = [r["rho"] for r in chosen]
        checks.number(name+"/rhos", rhos, old["rhos"])
        checks.number(name+"/mean_rho", math.fsum(rhos)/16, old["mean_rho"])
        endpoints = sum(abs(rho) == .75 for rho in rhos)
        checks.flag(name+"/endpoint_count", endpoints == old["endpoint_count"])
        reconstructed.append({**new, "theta": theta, "N": N, "method": method, "rhos": rhos,
                              "mean_rho": math.fsum(rhos)/16, "endpoint_count": endpoints})
    primary = []
    original_primary = {r["id"]: r for r in statistics["primary"]}
    gain = [lookup[(i, .3, 4096, "raw")]["reference_joint_KL"]-lookup[(i, .3, 4096, "finitecal")]["reference_joint_KL"] for i in range(16)]
    item = summarize(gain, family_critical)
    item.update(id="theta03_finite_improves_raw", theta=.3, N=4096, contrast="raw_minus_finite",
                threshold=family["gain_theta_03_margin"], pass_=item["lower"] > family["gain_theta_03_margin"])
    primary.append(item)
    for theta in (0.0, .3, .6):
        excess = [lookup[(i, theta, 4096, "finitecal")]["reference_joint_KL"]-lookup[(i, theta, 4096, "exact")]["reference_joint_KL"] for i in range(16)]
        item = summarize(excess, family_critical)
        item.update(id=f"theta{theta:g}_finite_noninferior_MLE", theta=theta, N=4096, contrast="finite_minus_exact",
                    threshold=family["finite_minus_MLE_noninferiority_margin"], pass_=item["upper"] < family["finite_minus_MLE_noninferiority_margin"])
        primary.append(item)
    for item in primary:
        old = original_primary[item["id"]]
        check_summary(checks, item["id"], item, old)
        for key in ("id", "theta", "N", "contrast", "threshold", "pass_"):
            checks.flag(item["id"]+"/"+key, item[key] == old[key])
    checks.flag("primary/recorded_count", statistics["primary_count"] == len(primary) == 4)
    checks.flag("primary/recorded_passes", statistics["primary_passed"] == sum(p["pass_"] for p in primary))
    # Preserve the observed two failed scientific gates; successful audit does
    # not imply all science gates passed.
    checks.flag("primary/original_two_pass_two_fail_preserved", sum(p["pass_"] for p in primary) == 2)
    certificates = {(r["stream"], r["theta"], r["N"], r["method"]): r for r in statistics["certificates"]}
    mesh = []
    for row in rows:
        if "continuous_objective_regret_bound_real_arithmetic" not in row:
            continue
        key = (row["stream"], row["theta"], row["N"], row["method"])
        old = certificates[key]
        bound = row["continuous_objective_regret_bound_real_arithmetic"]
        passed = bound <= protocol_data["mesh_gate"]
        checks.number("mesh/"+str(key)+"/value", bound, old["bound"])
        checks.flag("mesh/"+str(key)+"/gate", passed == old["pass_"])
        mesh.append({"stream": key[0], "theta": key[1], "N": key[2], "method": key[3], "bound": bound, "pass_": passed})
    checks.flag("mesh/counts", len(mesh) == statistics["mesh_count"] == len(certificates) == 480)
    checks.flag("mesh/pass_count", sum(r["pass_"] for r in mesh) == statistics["mesh_passed"])
    signals = []
    rho, rr = np.array(model["reference"]["rho801"]), np.array(model["reference"]["r801"])
    for old in statistics["signals"]:
        index = int(np.argmin(abs(rho-old["theta"])))
        require(abs(rho[index]-old["theta"]) < 1e-12, "Teacher is not a frozen reference node")
        t, N = float(rr[index]), old["N"]
        new = {"theta": old["theta"], "N": N, "r_theta": t, "sqrtN_r": math.sqrt(N)*t,
               "signal_over_pair_standard_error": math.sqrt(N)*t/math.sqrt(1-t*t)}
        for key in ("r_theta", "sqrtN_r", "signal_over_pair_standard_error"):
            checks.number(f"signals/{old['theta']}/{N}/{key}", new[key], old[key])
        signals.append(new)
    checks.flag("signals/six_cells", len(signals) == 6)
    checks.flag("statistics/conditional_calibration_scope", statistics["conditional_on_one_frozen_calibration"] is True and statistics["paired_t_is_development_approximation"] is True)
    return {"descriptive": reconstructed, "primary": primary, "mesh": mesh, "signals": signals,
            "critical_descriptive": critical, "critical_family": family_critical}, lookup


def fraction_coefficient(model):
    case = next(c for c in model["cases"] if c["n"] == 254 and c["K"] == 256)
    knots = np.array(case["gprime_spline_x"])
    index = int(np.searchsorted(knots, 0.0, side="right")-1)
    a, b, c = [Fraction.from_float(float(case["gprime_spline_c"][j][index])) for j in range(3)]
    local = -Fraction.from_float(float(knots[index]))
    return (3*a*local*local+2*b*local+c)/2


def bernoulli_risk(t, x):
    p, q = (1+t)/2, (1+x)/2
    return max(0.0, float(xlogy(p, p/q)+xlogy(1-p, (1-p)/(1-q))))


def posthoc_replay(production, model, policy, posthoc, checks, lookup):
    coefficient_exact = fraction_coefficient(model)
    coefficient = float(coefficient_exact)
    checks.flag("posthoc/coefficient_binary_exact", Fraction.from_float(coefficient) == coefficient_exact)
    checks.flag("posthoc/model_binding", policy["model_sha256"] == sha(BASE / "CAL/MODEL.json"))
    checks.number("posthoc/policy_coefficient", coefficient, policy["coefficient"], 0.0)
    checks.number("posthoc/result_coefficient", coefficient, posthoc["coefficient"], 0.0)
    checks.flag("posthoc/no_teacher_tuning_contract", policy["coefficient_uses_teacher_or_training_outcomes"] is False and policy["teacher_specific_tuning"] is False)
    checks.flag("posthoc/no_new_gate", policy["tests_or_new_success_gates"] is False and posthoc["original_primary_family_modified"] is False)
    rho = np.array(model["reference"]["rho801"])
    rr = np.array(model["reference"]["r801"])
    originals = {(r["stream"], r["theta"], r["N"]): r for r in posthoc["rows"]}
    rows = []
    for stream in production["records"]:
        with np.load(ROOT / stream["file"], allow_pickle=False) as arrays:
            raw = arrays["raw"]
            checks.number(f"posthoc/s{stream['stream']}/axis", arrays["rho"], rho, 0.0)
            for ti, theta in enumerate(production["config"]["teachers"]):
                true_index = int(np.argmin(abs(rho-theta)))
                require(abs(rho[true_index]-theta) < 1e-12, "Teacher reference node missing")
                t = float(rr[true_index])
                for ni, N in enumerate(production["config"]["sample_sizes"]):
                    mean = np.array([math.fsum(raw[:, ti, ni, index])/len(raw) for index in range(len(rho))])
                    scores = mean+coefficient*rho*rho
                    chosen = min(range(len(rho)), key=lambda index: float(scores[index]))
                    value = float(rho[chosen])
                    risk = bernoulli_risk(t, float(rr[chosen]))
                    finite = lookup[(stream["stream"], theta, N, "finitecal")]
                    original = originals[(stream["stream"], theta, N)]
                    name = f"posthoc/s{stream['stream']}/theta{theta}/N{N}"
                    checks.flag(name+"/index", chosen == original["grid_index"])
                    checks.number(name+"/rho", value, original["rho"], 0.0)
                    checks.number(name+"/risk", risk, original["reference_joint_KL"])
                    difference = risk-finite["reference_joint_KL"]
                    checks.number(name+"/paired_difference", difference, original["paired_risk_difference"])
                    same = chosen == finite["grid_index"]
                    checks.flag(name+"/same_point", same == original["same_point"])
                    checks.number(name+"/full_finite_rho", finite["rho"], original["full_finite_rho"], 0.0)
                    checks.number(name+"/full_finite_risk", finite["reference_joint_KL"], original["full_finite_risk"], 0.0)
                    D = Fraction.from_float(original["curvature_D"])
                    required_D = Fraction.from_float(stream["base_curvature_D"])+2*abs(coefficient_exact)
                    h = Fraction.from_float(finite["mesh_step"])
                    bound = Fraction.from_float(original["continuous_mesh_bound_real_arithmetic"])
                    checks.flag(name+"/D_outward", D >= required_D)
                    checks.flag(name+"/mesh_outward", bound >= D*h*h/8)
                    rows.append({"stream": stream["stream"], "theta": theta, "N": N, "grid_index": chosen,
                                 "rho": value, "reference_joint_KL": risk, "paired_risk_difference": difference,
                                 "same_point": same, "curvature_D": float(D), "mesh_bound": float(bound)})
    summaries = []
    for old in posthoc["summary"]:
        values = [row for row in rows if row["theta"] == old["theta"] and row["N"] == old["N"]]
        new = {"theta": old["theta"], "N": old["N"],
               "mean_reference_KL": math.fsum(row["reference_joint_KL"] for row in values)/16,
               "mean_rho": math.fsum(row["rho"] for row in values)/16,
               "mean_minus_full_finite": math.fsum(row["paired_risk_difference"] for row in values)/16,
               "exact_same_grid_point_count": sum(row["same_point"] for row in values)}
        for key in ("mean_reference_KL", "mean_rho", "mean_minus_full_finite"):
            checks.number(f"posthoc/summary/{new['theta']}/{new['N']}/{key}", new[key], old[key])
        checks.flag(f"posthoc/summary/{new['theta']}/{new['N']}/same_point_count", new["exact_same_grid_point_count"] == old["exact_same_grid_point_count"])
        summaries.append(new)
    checks.flag("posthoc/96_unique_rows", len(rows) == len(originals) == posthoc["additional_grid_selections"] == 96)
    checks.flag("posthoc/six_summary_cells", len(summaries) == 6)
    checks.flag("posthoc/55_same_points_preserved", sum(row["same_point"] for row in rows) == 55)
    return {"coefficient": coefficient, "coefficient_exact_numerator": str(coefficient_exact.numerator),
            "coefficient_exact_denominator": str(coefficient_exact.denominator),
            "rows": rows, "summaries": summaries, "same_point_count": sum(row["same_point"] for row in rows),
            "exploratory_only": True, "original_primary_family_modified": False}


def write_report(data, p):
    main = data["reconstructed_statistics"]
    lines = ["# 原统计家族与事后二次对照的独立审计", "",
             "本审计从原16基础流输出与保存的NPZ重算，不导入root的统计或事后分析函数。已知原结果，属于内部非盲审计；没有新增独立样本、随机数或Gaussian积分。", "",
             f"审计状态：{data['status']}；数值失败{data['numerical_failures']}，精确检查失败{data['exact_failures']}。统计公式重算通过不表示所有科学门通过。", "",
             "## 原四主门", "",
             "42个描述单元使用fsum求均值和中心化方差；df15临界值通过不完全beta表示的t CDF二分反演得到，未调用生产使用的t.ppf。四主门保持Bonferroni总alpha=.05、逐对比双侧alpha=.05/4，区间条件于同一个冻结CAL模型且只是开发近似。", "",
             "|原主门|均值|下界|上界|原门结果|", "|---|---:|---:|---:|---|"]
    if main:
        for row in main["primary"]:
            lines.append(f"|{row['id']}|{row['mean']:.9g}|{row['lower']:.9g}|{row['upper']:.9g}|{'通过' if row['pass_'] else '未通过'}|")
    lines += ["", "原家族保留2项通过、2项未通过；theta=.6的上界虽接近预定非劣阈值，也不追认为通过。零观察SE单独核flag，不被称为总体相等。480个mesh资格与6个观察信号单独复算，没有被当作效果门。", "",
              "## 事后二次复用", "",
              "实际定义记录为POSTHOC_QUADRATIC/PROTOCOL.json。它记载该规则在CAL导数审读后、root读取完整TRAIN风险前定义。审计确认系数只需冻结MODEL中n254的g′样条零点导数的一半，不使用teacher或训练数据；MODEL、定义协议、执行协议和结果hash链匹配。", "",
              "96个网格选择重新从原NPZ的raw曲线求均值后加同一个二次项得到；以参考r801节点直接算观察风险，避免复用生产PCHIP查询。6个描述单元、配对风险差和55个同选点均重建。所有新增D至少为D_base+2|c|，96个mesh界用Fraction检查向上性。", "",
              "这个规则重用了原数据及同一校准池，是明确的探索性对照，不并入四主门，也不产生新的确认性成功门。它没有重估校准，不能提供校准池无条件风险保证。", "",
              "## 账目", "", "```json", json.dumps(data["ledger"], ensure_ascii=False, indent=2), "```", "",
              f"协议SHA256：`{data['protocol_sha256']}`；脚本SHA256：`{data['source_sha256']}`。详细对比与保留结果在TRAIN/STATS_AUDIT.json。", ""]
    with (ROOT / p["report"]).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def run():
    require(not RESULT.exists(), "Statistics audit result exists; no overwrite")
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Audit source changed after freeze")
    checks = Checkbook()
    started = time.perf_counter()
    error = None
    main, extra = None, None
    try:
        for path, digest in p["input_hashes"].items():
            require(sha(ROOT/path) == digest, "Frozen input changed: "+path)
        for path, digest in p["npz_hashes"].items():
            checks.flag("npz/hash/"+path, sha(ROOT/path) == digest)
        production, statistics, stat_p = [read(TRAIN / name) for name in ("RESULTS.json", "STATISTICS.json", "STATISTICS_PROTOCOL.json")]
        training, model = read(TRAIN / "FREEZE.json"), read(BASE / "CAL/MODEL.json")
        policy, execution, posthoc = [read(POST / name) for name in ("PROTOCOL.json", "EXECUTION_PROTOCOL.json", "RESULTS.json")]
        checks.flag("statistics/source_hash", statistics["source_sha256"] == stat_p["source_sha256"] == sha(ROOT / "script/fincal_statistics.py"))
        checks.flag("statistics/protocol_hash", statistics["protocol_sha256"] == sha(TRAIN / "STATISTICS_PROTOCOL.json"))
        checks.flag("statistics/results_hash", statistics["results_sha256"] == sha(TRAIN / "RESULTS.json"))
        checks.flag("statistics/model_hash", statistics["reference_sha256"] == stat_p["model_sha256"] == sha(BASE / "CAL/MODEL.json"))
        checks.flag("statistics/family_from_FREEZE", stat_p["family"] == training["risk_family"] and stat_p["mesh_gate"] == training["mesh_discretization_gate"])
        checks.flag("statistics/freeze_hash", stat_p["freeze_sha256"] == sha(TRAIN / "FREEZE.json"))
        checks.flag("posthoc/source_hash", posthoc["source_sha256"] == execution["source_sha256"] == sha(ROOT / "script/fincal_posthoc.py"))
        checks.flag("posthoc/policy_hash", posthoc["protocol_sha256"] == execution["definition_protocol_sha256"] == sha(POST / "PROTOCOL.json"))
        checks.flag("posthoc/execution_hash", posthoc["execution_sha256"] == sha(POST / "EXECUTION_PROTOCOL.json"))
        for path, digest in execution["inputs"].items():
            checks.flag("posthoc/execution_input/"+path, sha(ROOT/path) == digest)
        main, lookup = main_statistics(production, statistics, stat_p, model, checks)
        extra = posthoc_replay(production, model, policy, posthoc, checks, lookup)
        for path, digest in p["input_hashes"].items():
            require(sha(ROOT/path) == digest, "Input changed during audit: "+path)
    except Exception:
        error = traceback.format_exc()
    numerical_failures = sum(r["failures"] for r in checks.numeric)
    exact_failures = sum(not r["passed"] for r in checks.exact)
    data = {"status": "passed" if not error and not numerical_failures and not exact_failures else "failed",
            "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "input_hashes": p["input_hashes"], "npz_hashes": p["npz_hashes"],
            "elapsed_seconds": time.perf_counter()-started, "numerical_checks": checks.numeric, "exact_checks": checks.exact,
            "numerical_failures": numerical_failures, "exact_failures": exact_failures, "software_error": error,
            "reconstructed_statistics": main, "reconstructed_posthoc": extra,
            "ledger": {"descriptive_cells": len(main["descriptive"]) if main else 0, "original_primary_gates": len(main["primary"]) if main else 0,
                       "original_primary_passed": sum(r["pass_"] for r in main["primary"]) if main else 0,
                       "original_primary_failed": sum(not r["pass_"] for r in main["primary"]) if main else 0,
                       "original_mesh_gates": len(main["mesh"]) if main else 0, "signal_cells": len(main["signals"]) if main else 0,
                       "posthoc_grid_selections_rebuilt": len(extra["rows"]) if extra else 0,
                       "posthoc_descriptive_cells": len(extra["summaries"]) if extra else 0,
                       "new_random_draws": 0, "new_scientific_samples": 0, "new_Gaussian_integrals": 0,
                       "posthoc_not_merged_into_primary_family": True},
            "approximate_t_intervals_not_distribution_free": True,
            "audit_success_does_not_requalify_failed_scientific_gates": True}
    save_new(RESULT, data)
    write_report(data, p)
    print(json.dumps({"status": data["status"], "numerical_groups": len(checks.numeric),
                      "numerical_scalar_comparisons": sum(r["count"] for r in checks.numeric), "exact_checks": len(checks.exact),
                      "numerical_failures": numerical_failures, "exact_failures": exact_failures, "software_error": bool(error),
                      "result_sha256": sha(RESULT)}, ensure_ascii=False))
    if data["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
