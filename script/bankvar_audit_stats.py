"""Independent statistics/gradient audit, scoped before ANALYSIS is produced.

No producer or old helper is imported. Scope/source are frozen first; completed
training arrays and ANALYSIS are hash-bound in a separate input receipt before
the audit arithmetic. No RNG or model integration is performed.
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
from scipy.special import stdtrit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/bank_precision/STATS_AUDIT"
BASE = ROOT / "artifacts/bank_precision"
PROTOCOL = OUT / "PROTOCOL.json"
DOC = ROOT / "docs/08_research_研究工作/粒子精度与局部校准_20261002/STATS_AUDIT.md"
THETAS, BS = [0., .3, .6], [32, 128, 512]
METHODS = ["raw", "marg", "asymquad", "fullfinite", "oldTaylor", "localquad", "exact"]
GRAD_FIELDS = ["gradient_mean", "sample_variance", "variance_over_B", "squared_mean", "unbiased_squared_population_mean"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")


def freeze():
    assert not PROTOCOL.exists(), "Audit scope already frozen"
    train = read(BASE/"TRAIN/FREEZE.json")
    paths = [ROOT/"script/bankvar_analysis.py", BASE/"TRAIN/FREEZE.json", ROOT/train["plan_path"],
             ROOT/train["reference_path"], ROOT/train["local_model_path"], ROOT/train["old_taylor_path"]]
    write_new(PROTOCOL, dict(schema="bankvar-independent-stats-audit-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(SOURCE),
        inputs={rel(p): sha(p) for p in paths}, analysis_existed_at_freeze=(BASE/"ANALYSIS.json").exists(),
        expected_output_rows=1008, expected_cells=63, expected_gradient_rows=864,
        expected_mesh_rows=864, numerical_tolerance=2e-12, KL_tolerance=2e-14,
        primary_family="Two paired t development gates: average3 teachers within each stream, then16 streams; t_(.9875,15)",
        scope="63 cells; both primary gates; 864 rational mesh/interpolation checks conditional on stored base curvature; all864 fixed-point gradient rows and all/interior summaries from saved NPZ and independently evaluated penalties",
        excluded="No prior RNG replay, no underlying likelihood/path-gradient recomputation, no Gaussian integral certification; these have separate audits",
        new_random_draws=0, new_integrals=0, automatic_retry=False))
    print(json.dumps(dict(frozen=True, protocol_sha256=sha(PROTOCOL), source_sha256=sha(SOURCE))), flush=True)


class Checks:
    def __init__(self):
        self.rows = []

    def flag(self, name, value):
        self.rows.append(dict(name=name, passed=bool(value)))

    def close(self, name, actual, expected, tolerance):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        err = float(np.max(abs(a-b))) if valid and a.size else (0. if valid else 1e300)
        self.rows.append(dict(name=name, passed=bool(valid and err <= tolerance), max_absolute_error=err, tolerance=tolerance))


def mean(xs):
    return math.fsum(float(x) for x in xs)/len(xs)


def variance(xs):
    mu = mean(xs)
    return math.fsum((float(x)-mu)**2 for x in xs)/(len(xs)-1)


def summary(xs, critical=None):
    mu, se = mean(xs), math.sqrt(variance(xs)/len(xs))
    result = dict(mean=mu, SE=se, n=len(xs))
    if critical is not None:
        result.update(lower=mu-critical*se, upper=mu+critical*se)
    return result


def cubic_value(coeff, knots, point):
    i = min(max(bisect_right(knots, point)-1, 0), len(knots)-2)
    x = point-knots[i]
    return ((coeff[0][i]*x+coeff[1][i])*x+coeff[2][i])*x+coeff[3][i]


def cubic_derivative_bound(coeff, knots):
    result = Fraction(0)
    for i in range(len(knots)-1):
        a, b, c = 3*Fraction.from_float(coeff[0][i]), 2*Fraction.from_float(coeff[1][i]), Fraction.from_float(coeff[2][i])
        width = Fraction.from_float(knots[i+1])-Fraction.from_float(knots[i])
        points = [Fraction(0), width]
        if a and 0 < -b/(2*a) < width:
            points.append(-b/(2*a))
        for x in points:
            result = max(result, abs((a*x+b)*x+c))
    return result


def check_stats(checks, name, got, expected, tolerance):
    checks.flag(name+"/n", got["n"] == expected["n"])
    for key in got:
        if key != "n":
            checks.close(name+"/"+key, got[key], expected[key], tolerance)


def run():
    scope = read(PROTOCOL)
    assert sha(SOURCE) == scope["source_sha256"], "Audit source changed"
    for name, digest in scope["inputs"].items():
        assert sha(ROOT/name) == digest, f"Frozen input changed: {name}"
    assert not (OUT/"EXECUTION.json").exists(), "No automatic audit retry"
    assert (BASE/"ANALYSIS.json").is_file() and (BASE/"TRAIN/RESULTS.json").is_file(), "Wait for completed production"
    train = read(BASE/"TRAIN/RESULTS.json")
    final_inputs = [BASE/"ANALYSIS.json", BASE/"TRAIN/RESULTS.json"]
    final_inputs += [ROOT/record["file"] for record in train["records"]]
    write_new(OUT/"INPUT_RECEIPT.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        audit_protocol_sha256=sha(PROTOCOL), inputs={rel(p): sha(p) for p in final_inputs}))
    write_new(OUT/"EXECUTION.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(SOURCE), protocol_sha256=sha(PROTOCOL), input_receipt_sha256=sha(OUT/"INPUT_RECEIPT.json")))
    checks, start = Checks(), time.perf_counter()
    try:
        analysis, freeze_train = read(BASE/"ANALYSIS.json"), read(BASE/"TRAIN/FREEZE.json")
        tol = scope["numerical_tolerance"]
        checks.flag("analysis/source_binding", analysis["source_sha256"] == sha(ROOT/"script/bankvar_analysis.py"))
        checks.flag("analysis/input_binding", analysis["input_sha256"] == sha(BASE/"TRAIN/RESULTS.json"))
        checks.flag("analysis/plan_binding", analysis["plan_sha256"] == freeze_train["plan_sha256"])
        checks.flag("train/config", train["config"] == freeze_train["config"] and train["methods"] == METHODS)
        checks.flag("train/source_binding", train["source_sha256"] == freeze_train["source_sha256"]
                    and train["freeze_sha256"] == sha(BASE/"TRAIN/FREEZE.json"))
        checks.flag("train/stream_count", len(train["records"]) == 16)
        rows = [row for rec in train["records"] for row in rec["outputs"]]
        lookup = {(row["stream"], row["theta"], row["B"], row["method"]): row for row in rows}
        expected_keys = {(s, th, B, method) for s in range(16) for th in THETAS for B in BS for method in METHODS}
        checks.flag("outputs/unique_complete", len(rows) == 1008 and set(lookup) == expected_keys and len(lookup) == 1008)
        checks.flag("outputs/count", analysis["output_count"] == 1008 and train["output_count"] == 1008
                    and analysis["unique_data_teacher_MLE"] == 48)
        model = read(ROOT/freeze_train["reference_path"])
        case = next(x for x in model["cases"] if x["n"] == 254 and x["K"] == 256)
        gp, knots = case["gprime_spline_c"], case["gprime_spline_x"]
        exact_gD = cubic_derivative_bound(gp, knots)
        checks.flag("fullfinite/curvature_reference", Fraction.from_float(case["g_second_sup"]) >= exact_gD)
        constants = model["reference"]
        coefficients = dict(asymquad=constants["A"]*constants["d"]*case["w"],
                            oldTaylor=read(ROOT/freeze_train["old_taylor_path"])["coefficient"],
                            localquad=read(ROOT/freeze_train["local_model_path"])["coefficient"])
        for method, coefficient in coefficients.items():
            checks.close("coefficient/"+method, coefficient, train["coefficients"][method], 0.)
        for key, row in lookup.items():
            t, r = row["teacher_r"], row["observed_r"]
            risk = ((1+t)*math.log((1+t)/(1+r))+(1-t)*math.log((1-t)/(1-r)))/2
            checks.close("row_KL/"+str(key), risk, row["reference_joint_KL"], scope["KL_tolerance"])
        def vals(th, B, method, field):
            return [lookup[s, th, B, method][field] for s in range(16)]
        def discrepancy(th, B, method):
            return [(lookup[s, th, B, method]["rho"]-lookup[s, th, B, "exact"]["rho"])**2 for s in range(16)]
        stored_cells = {(x["theta"], x["B"], x["method"]): x for x in analysis["tables"]}
        checks.flag("cells/count", len(analysis["tables"]) == 63 and len(stored_cells) == 63)
        for th in THETAS:
            for B in BS:
                for method in METHODS:
                    cell = stored_cells[th, B, method]
                    for name, values in (("rho", vals(th,B,method,"rho")),
                        ("reference_joint_KL", vals(th,B,method,"reference_joint_KL")),
                        ("squared_discrepancy_to_data_MLE", discrepancy(th,B,method))):
                        check_stats(checks, f"cell/{th}/{B}/{method}/{name}", summary(values), cell[name], tol)
                    checks.flag(f"cell/{th}/{B}/{method}/endpoints", cell["endpoint_count"] == sum(abs(x)==.75 for x in vals(th,B,method,"rho")))
        critical = float(stdtrit(15, .9875))
        checks.close("family/critical", critical, analysis["critical_t"], 1e-13)
        checks.flag("family/contract", analysis["family_size"] == 2 and analysis["alpha"] == .05 and analysis["df"] == 15
                    and analysis["calibration_uncertainty_in_CI"] is False and analysis["new_observations"] == 0)
        d1, d2 = [], []
        for s in range(16):
            d1.append(mean([(lookup[s,th,128,"fullfinite"]["rho"]-lookup[s,th,128,"exact"]["rho"])**2
                            -(lookup[s,th,512,"fullfinite"]["rho"]-lookup[s,th,512,"exact"]["rho"])**2 for th in THETAS]))
            d2.append(mean([lookup[s,th,512,"localquad"]["reference_joint_KL"]
                            -lookup[s,th,512,"fullfinite"]["reference_joint_KL"] for th in THETAS]))
        rebuilt_gates = []
        for values, name, threshold, direction, stored in zip((d1,d2),
            ("fullfinite_B128_minus_B512_MSE","local_minus_full_B512_KL"), (0.,.0001),
            ("lower_above","upper_below"), analysis["primary_gates"]):
            stat = summary(values, critical)
            passed = stat["lower"] > threshold if direction == "lower_above" else stat["upper"] < threshold
            checks.flag("gate/"+name+"/definition", stored["name"] == name and stored["threshold"] == threshold and stored["direction"] == direction)
            checks.close("gate/"+name+"/paired_streams", values, stored["stream_differences"], tol)
            check_stats(checks, "gate/"+name, stat, stored, tol)
            checks.flag("gate/"+name+"/passed", stored["passed"] == passed)
            rebuilt_gates.append(dict(name=name, **stat, threshold=threshold, direction=direction, passed=passed))
        stored_grad = {(x["stream"],x["theta"],x["B"],x["method"]):x for x in analysis["gradient_rows"]}
        checks.flag("gradient/count", len(stored_grad) == 864 and len(analysis["gradient_rows"]) == 864)
        rebuilt_grad, bounds, boundary_count, cheb_count = [], [], 0, 0
        eps_exact = Fraction(700)*Fraction(25,28)**192
        for record in train["records"]:
            s = record["stream"]
            path = ROOT/record["file"]
            checks.flag(f"stream{s}/archive_hash", sha(path) == record["file_sha256"])
            with np.load(path, allow_pickle=False) as z:
                rho_grid = z["rho"].copy()
                points, boundaries = z["gradient_rho"].copy(), z["gradient_MLE_boundary"].copy()
                raw, marg = z["raw_path_gradient"].copy(), z["marg_path_gradient"].copy()
            checks.flag(f"stream{s}/gradient_shapes", raw.shape == (512,3) and marg.shape == (512,3)
                        and np.isfinite(raw).all() and np.isfinite(marg).all())
            checks.close(f"stream{s}/MLE_points", points, record["gradient_rho"], 0.)
            checks.close(f"stream{s}/MLE_flags", boundaries, record["gradient_MLE_boundary"], 0.)
            boundary_count += sum(bool(x) for x in boundaries)
            cheb = record["evaluation_mode"] == "chebyshev193"
            cheb_count += int(cheb)
            checks.flag(f"stream{s}/evaluation_branch", record["active_gate"]["pass_all"] == cheb
                        and record["evaluation_mode"] in ("chebyshev193","direct801"))
            checks.flag(f"stream{s}/knots", set(knots).issubset(set(rho_grid)) and rho_grid[0]==-.75 and rho_grid[-1]==.75)
            mesh = max(Fraction.from_float(float(b))-Fraction.from_float(float(a)) for a,b in zip(rho_grid[:-1],rho_grid[1:]))
            for ti, th in enumerate(THETAS):
                q, boundary = float(points[ti]), bool(boundaries[ti])
                for B in BS:
                    checks.close(f"stream{s}/{th}/{B}/exact_baseline", lookup[s,th,B,"exact"]["rho"], q, 0.)
                    for method in METHODS[:-1]:
                        sample = marg[:B,ti] if method == "marg" else raw[:B,ti]
                        penalty = (0. if method in ("raw","marg") else
                                   cubic_value(gp,knots,q) if method == "fullfinite" else 2*coefficients[method]*q)
                        mu, var = mean(sample)+penalty, variance(sample)
                        row = dict(stream=s,theta=th,B=B,method=method,rho=q,boundary=boundary,
                            gradient_mean=mu,sample_variance=var,variance_over_B=var/B,
                            squared_mean=mu*mu,unbiased_squared_population_mean=mu*mu-var/B)
                        stored = stored_grad[s,th,B,method]
                        checks.flag(f"gradient/{s}/{th}/{B}/{method}/boundary", stored["boundary"] == boundary)
                        checks.close(f"gradient/{s}/{th}/{B}/{method}/point", stored["rho"], q, 0.)
                        for field in GRAD_FIELDS:
                            checks.close(f"gradient/{s}/{th}/{B}/{method}/{field}", row[field], stored[field], tol)
                        rebuilt_grad.append(row)
                        # Exact rational inequalities on every saved bound;
                        # the base raw curvature has a separate TRAIN audit.
                        output = lookup[s,th,B,method]
                        D = Fraction.from_float(output["curvature_D"])
                        baseD = Fraction.from_float(record["base_curvature_D"][str(B)])
                        penaltyD = (Fraction(0) if method in ("raw","marg") else
                                    exact_gD if method == "fullfinite" else 2*abs(Fraction.from_float(coefficients[method])))
                        eh = Fraction.from_float(output["mesh_step_upper"])
                        mesh_bound = Fraction.from_float(output["mesh_regret_bound_real_arithmetic"])
                        eps = Fraction.from_float(output["interpolation_uncentered_sup_error_upper"])
                        total = Fraction.from_float(output["continuous_objective_regret_bound_real_arithmetic"])
                        checks.flag(f"mesh/{s}/{th}/{B}/{method}", D>=baseD+penaltyD and eh>=mesh and mesh_bound>=D*eh*eh/8
                                    and eps >= (eps_exact if cheb else 0) and (cheb or eps==0)
                                    and Fraction.from_float(output["interpolation_minimizer_regret_upper"]) >= 2*eps
                                    and total>=mesh_bound+2*eps)
                        bounds.append(float(total))
        stored_sum = {(x["theta"],x["B"],x["method"],x["subset"]):x for x in analysis["gradient_summary"]}
        expected_summary_keys = set()
        for th in THETAS:
            for B in BS:
                for method in METHODS[:-1]:
                    for subset in ("all","interior"):
                        rr = [r for r in rebuilt_grad if r["theta"]==th and r["B"]==B and r["method"]==method
                              and (subset=="all" or not r["boundary"])]
                        if not rr:
                            continue
                        key = (th,B,method,subset)
                        expected_summary_keys.add(key)
                        stored = stored_sum[key]
                        checks.flag("gradient_summary/"+str(key)+"/count", stored["count"] == len(rr))
                        for field in GRAD_FIELDS:
                            checks.close("gradient_summary/"+str(key)+"/"+field, mean([r[field] for r in rr]), stored[field], tol)
        checks.flag("gradient_summary/complete", set(stored_sum) == expected_summary_keys and len(stored_sum)==len(analysis["gradient_summary"]))
        checks.flag("gradient/boundary_count", analysis["boundary_MLE_count"] == boundary_count)
        checks.flag("interpolation/stream_count", analysis["cheb_stream_count"] == cheb_count)
        negative = sum(x["unbiased_squared_population_mean"]<0 for x in rebuilt_grad)
        checks.flag("gradient/negative_preserved", sum(x["unbiased_squared_population_mean"]<0 for x in analysis["gradient_rows"]) == negative)
        qual = analysis["mesh_interpolation_qualification"]
        checks.flag("mesh/qualification", len(bounds)==864 and qual["count"]==864 and qual["threshold"]==3e-5
                    and qual["pass_count"] == sum(x<=3e-5 for x in bounds))
        checks.close("mesh/maximum", max(bounds), qual["maximum"], 0.)
        result = dict(schema=1, passed=all(x["passed"] for x in checks.rows), source_sha256=sha(SOURCE),
            protocol_sha256=sha(PROTOCOL), input_receipt_sha256=sha(OUT/"INPUT_RECEIPT.json"),
            analysis_sha256=sha(BASE/"ANALYSIS.json"), training_sha256=sha(BASE/"TRAIN/RESULTS.json"),
            checks=checks.rows, rebuilt_primary_gates=rebuilt_gates, rebuilt_critical_t=critical,
            cells=63, output_rows=1008, gradient_rows=len(rebuilt_grad), gradient_summaries=len(expected_summary_keys),
            negative_unbiased_squared_mean_rows=negative, boundary_MLE_count=boundary_count,
            mesh_rows=len(bounds), mesh_pass_count=sum(x<=3e-5 for x in bounds), mesh_maximum=max(bounds),
            producer_imported=False, new_random_draws=0, new_integrals=0, elapsed_seconds=time.perf_counter()-start,
            limitation="Saved-result arithmetic and fixed-point gradient aggregation, not underlying score/RNG replay. CI remains conditional development approximation; fixed-point ANOVA components and optimized-risk causal shares are not estimated.")
        write_new(OUT/"RESULT.json", result)
        assert result["passed"], "Audit failed; preserve report"
        lines = ["# 主统计与固定点梯度独立审计", "", "全部固定审计项通过。审计未导入生产 analysis 或旧 helper，未生成RNG、未做模型积分。", "",
                 "根据1008行输出独立重建63格的rho/KL/MSE均值、标准误和端点数；直接ratio-log复算逐行参考KL。两个主门先在每流平均三个teacher，再以16流计算paired t；使用独立 stdtrit 调用核对 t_.9875,15。", "",
                 "|主门|mean|SE|共同95%近似区间|通过|", "|---|---:|---:|---|---|"]
        for row in rebuilt_gates:
            lines.append(f"|{row['name']}|{row['mean']:.12g}|{row['SE']:.12g}|[{row['lower']:.12g}, {row['upper']:.12g}]|{row['passed']}|")
        lines += ["", "区间条件于本次旧数据与冻结校准，保持开发近似，不包含重新校准变异；不能从合并teacher门推及逐teacher非劣。", "",
                  f"16个NPZ直接读取原始raw/marg固定点梯度，并以独立Horner表达计算惩罚导数，重建全部864行及{len(expected_summary_keys)}个all/interior汇总。负的无偏均值平方估计共{negative}行，原样保留。48个不同数据/teacher数值MLE中边界{boundary_count}个，未把边界非零梯度当作校准误差。", "",
                  f"全部864个已保存mesh/interpolation界用有理不等式核验惩罚曲率、实际网格间距与2epsilon相加；{sum(x<=3e-5 for x in bounds)}/864达到预定3e-5资格，最大{max(bounds):.15g}。此处条件于训练记录的base曲率；底层active曲率与路径评分另由TRAIN审计负责。浮点zeta不在此解析项内。", "",
                  "这不是新数据复制、不是完整似然重算、不是严格积分认证，也没有估计POPULATION_DATA_THEORY的三个无条件ANOVA分量或任何优化后KL原因占比。", "",
                  f"审计源 SHA256 `{sha(SOURCE)}`；协议 `{sha(PROTOCOL)}`；生产ANALYSIS `{sha(BASE/'ANALYSIS.json')}`。", "",
                  "范围和源在生产ANALYSIS之前冻结；完成输入在审计算术前另以INPUT_RECEIPT.json绑定。完整检查收据见 `artifacts/bank_precision/STATS_AUDIT/RESULT.json`。", ""]
        assert not DOC.exists(), "Refuse report overwrite"
        DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(json.dumps(dict(passed=True, checks=len(checks.rows), negative_rows=negative, boundaries=boundary_count)), flush=True)
    except Exception as exc:
        write_new(OUT/"FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), checks=checks.rows))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze","run"])
    args = parser.parse_args()
    freeze() if args.phase=="freeze" else run()
