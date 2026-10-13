"""Independent a=1 audit via Price integration; never imports production code.

Freeze this script and the existing result bytes before executing --run.
This is an internal, non-blind numerical audit, not a rigorous quadrature certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import roots_hermitenorm, roots_legendre


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/regularization_learning/NUMERIC"
DOC = ROOT / "docs/08_research_研究工作/统一正则与真实相关学习_20261002/NUMERIC_INDEPENDENT_AUDIT.md"
PROTOCOL = OUT / "INDEPENDENT_AUDIT_PROTOCOL.json"
RESULT = OUT / "INDEPENDENT_AUDIT.json"
SOURCE = Path(__file__).resolve()
INPUTS = [OUT / "RUN.json", OUT / "AUDIT.json", OUT / "QUALIFICATION.json"]
TEACHERS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.70, 0.7375]
SETTINGS = [(80, 96), (112, 128)]
R = 0.75
A = 0.25
EPSABS = 2e-13
EPSREL = 2e-13


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    if PROTOCOL.exists() or RESULT.exists() or DOC.exists():
        raise RuntimeError("Audit protocol/result/report exists; do not overwrite or silently append budget")
    protocol = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": rel(SOURCE), "source_sha256": sha(SOURCE),
        "inputs": {rel(path): sha(path) for path in INPUTS},
        "independent_implementer": "/root/neighbor_check",
        "production_implementer_is_different_agent": True,
        "blind": False,
        "scope": "a=1,R=.75 original six teachers plus one posthoc numerical audit point",
        "six_prespecified_teachers_from_production": TEACHERS[:6],
        "posthoc_audit_teacher": TEACHERS[6],
        "posthoc_reason": "Already reported maximum on the production 61-teacher grid; audit the value only, not a new maximum confirmation",
        "all_audited_teachers": TEACHERS,
        "quadrature_variants": [{"gh": gh, "gl": gl} for gh, gl in SETTINGS],
        "methods": {
            "r_prime": "Price E[f'(X)f'(rho X+sqrt(1-rho^2)Z)] by roots_hermitenorm tensor nodes",
            "r": "Gauss-Legendre integral of r_prime from 0 to rho",
            "H": "Gauss-Legendre integral of (abs(rho)-u)*r_prime(u) from 0 to abs(rho)",
            "constants": "Four scipy.integrate.quad integrals on the real line: E f_prime, E f^2, E logsigmoid, Var logsigmoid",
            "quadratic_root": "Independent monotone bisection for 2d*rho-r(rho)=r(theta), clipped at R; bracket width <=2e-12, at most 50 iterations",
            "risk": "Adaptive one-dimensional integral from teacher moment t to candidate moment x of (u-t)/(1-u^2)",
        },
        "adaptive_settings": {"epsabs": EPSABS, "epsrel": EPSREL, "limit": 300},
        "comparison_targets": ["RUN.json a=1 order=64", "AUDIT.json Hermite a=1 order=64", "between both independent variants"],
        "absolute_tolerances": {"rho": 1e-6, "joint_KL": 1e-8, "all_other_scalar_fields": 1e-9},
        "no_imports_from_production_or_old_audit": True,
        "budgets": {"variants": 2, "teachers_per_variant": 7, "maximum_bisection_iterations_per_root": 50,
                    "automatic_escalation": False},
        "new_random_samples": 0, "new_labels": 0, "new_banks": 0, "new_data_fits": 0,
        "additional_numerical_integrals_counted_separately": True,
        "does_not_requalify_a2_or_a4": True,
        "strict_total_quadrature_certificate": False,
    }
    save_new(PROTOCOL, protocol)
    print(json.dumps({"protocol_frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


def normal_density(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class AdaptiveIntegrals:
    def __init__(self):
        self.records = []

    def integrate(self, name, fn, lower=-np.inf, upper=np.inf):
        answer = quad(fn, lower, upper, epsabs=EPSABS, epsrel=EPSREL, limit=300, full_output=1)
        value, estimated_error, info = answer[:3]
        self.records.append({"name": name, "value": float(value), "estimated_absolute_error": float(estimated_error),
                             "function_evaluations": int(info["neval"]), "subintervals": int(info["last"]),
                             "warning": answer[3] if len(answer) > 3 else None})
        return float(value)

    def risk(self, teacher_moment, candidate_moment, name):
        if teacher_moment == candidate_moment:
            return 0.0
        return self.integrate(name, lambda u: (u - teacher_moment) / (1 - u*u), teacher_moment, candidate_moment)


def constants(adaptive):
    fp = adaptive.integrate("E_f_prime", lambda x: 0.5 * (1-math.tanh(x/2)**2) * normal_density(x))
    vf = adaptive.integrate("E_f_squared", lambda x: math.tanh(x/2)**2 * normal_density(x))
    mean = adaptive.integrate("mean_logsigmoid", lambda x: -float(np.logaddexp(0.0, -x)) * normal_density(x))
    variance = adaptive.integrate("var_logsigmoid", lambda x: (-float(np.logaddexp(0.0, -x))-mean)**2 * normal_density(x))
    return {"A": A, "E_f_prime": fp, "d": fp*fp, "var_f": vf,
            "mean_logsigmoid": mean, "var_logsigmoid": variance,
            "gamma0": A*fp*fp, "gamma_local": A*fp*fp/2}


class PriceReconstruction:
    def __init__(self, gh_order, gl_order, common, adaptive):
        nodes, weights = roots_hermitenorm(gh_order)
        self.z = nodes
        self.weights = weights / math.sqrt(2*math.pi)
        self.xprime = 0.5 * (1-np.tanh(nodes/2)**2)
        self.gl_nodes, self.gl_weights = roots_legendre(gl_order)
        self.gh, self.gl = gh_order, gl_order
        self.common, self.adaptive = common, adaptive
        self.cache = {0.0: (0.0, 0.0)}
        self.counts = Counter()
        self.price_records = []
        self.root_records = []

    def price(self, values):
        correlations = np.asarray(values, dtype=float).reshape(-1)
        y = (correlations[:, None, None] * self.z[None, :, None]
             + np.sqrt(1-correlations**2)[:, None, None] * self.z[None, None, :])
        yprime = 0.5 * (1-np.tanh(y/2)**2)
        estimates = np.einsum("i,j,i,kij->k", self.weights, self.weights, self.xprime, yprime, optimize=False)
        self.counts["price_scalar_2d_integrals"] += len(correlations)
        self.counts["price_tensor_node_entries"] += len(correlations) * self.gh**2
        return estimates

    def moments(self, rho):
        sign = -1 if rho < 0 else 1
        extent = abs(float(rho))
        if extent not in self.cache:
            positions = extent * (self.gl_nodes+1)/2
            values = self.price(positions)
            weights = self.gl_weights*extent/2
            r = float(weights @ values)
            H = float((weights*(extent-positions)) @ values)
            self.cache[extent] = (r, H)
            self.counts["unique_moment_argument_integrations"] += 1
            self.counts["outer_scalar_1d_integrals"] += 2
        r, H = self.cache[extent]
        return sign*r, H

    def quadratic(self, t):
        d = self.common["d"]
        if t == 0:
            self.root_records.append({"t": t, "rho": 0.0, "mode": "symmetry", "iterations": 0})
            return 0.0
        endpoint = 2*d*R-self.moments(R)[0]
        if t >= endpoint:
            self.root_records.append({"t": t, "rho": R, "mode": "clipped", "endpoint_m": endpoint, "iterations": 0})
            return R
        lo, hi = 0.0, R
        iterations = 0
        while hi-lo > 2e-12 and iterations < 50:
            midpoint = (lo+hi)/2
            value = 2*d*midpoint-self.moments(midpoint)[0]-t
            if value < 0:
                lo = midpoint
            else:
                hi = midpoint
            iterations += 1
        if hi-lo > 2e-12:
            raise RuntimeError("Bisection exhausted fixed budget")
        rho = (lo+hi)/2
        self.root_records.append({"t": t, "rho": rho, "mode": "bisection", "iterations": iterations,
                                  "bracket": [lo, hi], "residual": 2*d*rho-self.moments(rho)[0]-t})
        self.counts["bisection_roots"] += 1
        self.counts["bisection_iterations"] += iterations
        return rho

    def run(self):
        r_R, H_R = self.moments(R)
        rprime_0, rprime_R = [float(x) for x in self.price([0, R])]
        scalar = {**self.common, "r_R": r_R, "H_R": H_R, "r_prime_R": rprime_R,
                  "gamma_global": A*H_R/R**2, "quadratic_curvature_min": A*(2*self.common["d"]-rprime_R),
                  "price_r_prime_0": rprime_0}
        rows = []
        for theta in TEACHERS:
            t, Ht = self.moments(theta)
            rho_quad = self.quadratic(t)
            rule_rhos = {"raw": -R if theta == 0 else R, "quadratic": rho_quad, "informed": theta, "zero": 0.0}
            rules = {}
            for name, rho in rule_rhos.items():
                r, H = self.moments(rho)
                risk = self.adaptive.risk(t, r, f"KL_gh{self.gh}_theta{theta}_{name}")
                rules[name] = {"rho": rho, "r": r, "H": H, "joint_KL": risk}
            rows.append({"theta": theta, "selection": "original_six" if theta != 0.7375 else "posthoc_audit_of_reported_grid_max",
                         "r_teacher": t, "H_teacher": Ht, "rules": rules})
        return {"a": 1.0, "R": R, "gh_order": self.gh, "gl_order": self.gl, "constants": scalar,
                "rows": rows, "root_records": self.root_records, "accounting": dict(self.counts)}


def compare_value(checks, family, field, current, reference):
    tolerance = 1e-6 if field.endswith("/rho") else 1e-8 if field.endswith("/joint_KL") else 1e-9
    difference = abs(current-reference)
    checks.append({"family": family, "field": field, "independent_value": current, "comparison_value": reference,
                   "absolute_difference": difference, "tolerance": tolerance, "pass": bool(difference <= tolerance)})


def compare_cell(checks, family, independent, reference):
    for key, value in independent["constants"].items():
        if key in reference["constants"]:
            compare_value(checks, family, "constants/"+key, value, reference["constants"][key])
    for row in independent["rows"]:
        other = next(r for r in reference["rows"] if abs(r["theta"]-row["theta"]) < 1e-11)
        prefix = f"theta/{row['theta']}"
        for key in ("r_teacher", "H_teacher"):
            if key in other:
                compare_value(checks, family, prefix+"/"+key, row[key], other[key])
        for rule, values in row["rules"].items():
            for key, value in values.items():
                if key in other["rules"][rule]:
                    compare_value(checks, family, prefix+"/"+rule+"/"+key, value, other["rules"][rule][key])


def report(result):
    primary = result["independent_cells"][0]
    lines = ["# a=1 数值参考的独立重建审计", "",
             "本审计由与生产及原 Hermite 复核不同的内部代理实现；已见生产报告及结果，因此不是盲审。未调用、导入或复制生产／原审计函数，只读取冻结结果用于事后数值比较。", "",
             "在执行任何新积分前，先冻结 `INDEPENDENT_AUDIT_PROTOCOL.json`，包含本脚本 SHA256、输入字节、教师、节点预算及资格阈值。", "",
             "## 范围与不同表达", "",
             "仅审计 a=1、R=.75。六个原教师为 0/.15/.30/.45/.60/.70；另加 .7375，因它已经被报告为原61点扫描的最坏点而事后选入。本审计只核该点读数，不重新扫描或认证连续最坏风险。a=2/a=4 的既有资格失败保持不变。", "",
             "独立实现先用 Price 的二维积分计算 r′，再用一维积分获得 r 和 H；两套节点为 GH80/GL96、GH112/GL128。d 与 log-sigmoid 均值／方差来自另写的一维 adaptive quad；二次根用独立二分，KL 通过观察矩风险导数积分计算。没有使用直接 tanh 乘积来重算 r，也没有使用 Hermite 级数重算 r/H。", "",
             "## 结果", "",
             f"全部 {len(result['checks'])} 项资格中 {result['passed']} 通过，{len(result['failures'])} 失败。阈值沿原计划：rho 绝对差≤1e−6、KL≤1e−8，其余所核标量≤1e−9。", "",
             "|比较家族|通过/总项|最大绝对差|", "|---|---:|---:|"]
    for family, item in result["family_summary"].items():
        lines.append(f"|{family}|{item['passed']}/{item['checks']}|{item['maximum_absolute_difference']:.4g}|")
    lines += ["", "独立主表达 GH80/GL96 的二次规则读数：", "",
              "|theta|rho|完整joint KL|审计选择|", "|---:|---:|---:|---|"]
    for row in primary["rows"]:
        q = row["rules"]["quadratic"]
        lines.append(f"|{row['theta']:.4f}|{q['rho']:.12f}|{q['joint_KL']:.12g}|{'原六点' if row['theta'] != .7375 else '事后审计点'}|")
    c = primary["constants"]
    lines += ["", f"独立 d={c['d']:.16g}，gamma_local={c['gamma_local']:.16g}，gamma_global={c['gamma_global']:.16g}，gamma0={c['gamma0']:.16g}，Var(log sigmoid Z)={c['var_logsigmoid']:.16g}。", "",
              "## 新增计算账目与限制", "", "```json", json.dumps(result["accounting"], ensure_ascii=False, indent=2), "```", "",
              "二维 Price 求积共享同一组 GH 节点；r/H 的外层积分共享 r′ 读数。重复节点及 root 迭代不是独立科学样本。新增的是审计数值积分，零随机观察、零粒子 bank、零训练拟合。", "",
              "节点变体一致、与生产／Hermite结果相符及 adaptive quad 的内部误差估计只构成数值资格，不是严格总积分余项证书。既有高发射强度失败未被本审计追认，本审计也不证明有限 n 粒子学习达到这些极限风险。", "",
              f"协议 SHA256：`{result['protocol_sha256']}`。脚本 SHA256：`{result['source_sha256']}`。所有逐项差异和积分账目保存在同目录 `INDEPENDENT_AUDIT.json`。", ""]
    with DOC.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def run():
    if RESULT.exists() or DOC.exists():
        raise RuntimeError("Audit result/report exists; refuse overwrite")
    protocol = read(PROTOCOL)
    if protocol["source_sha256"] != sha(SOURCE):
        raise RuntimeError("Source differs from frozen protocol")
    for path, digest in protocol["inputs"].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError("Frozen comparison input changed: "+path)
    adaptive = AdaptiveIntegrals()
    common = constants(adaptive)
    cells = [PriceReconstruction(gh, gl, common, adaptive).run() for gh, gl in SETTINGS]
    production = next(c for c in read(INPUTS[0])["cells"] if c["a"] == 1 and c["order"] == 64)
    hermite = next(c for c in read(INPUTS[1])["cells"] if c["a"] == 1 and c["order"] == 64)
    checks = []
    for cell in cells:
        label = f"Price_GH{cell['gh_order']}_GL{cell['gl_order']}"
        compare_cell(checks, label+"_vs_RUN64", cell, production)
        compare_cell(checks, label+"_vs_Hermite64", cell, hermite)
        compare_value(checks, label+"_Price_at_zero_vs_adaptive_d", "constants/d", cell["constants"]["price_r_prime_0"], common["d"])
    compare_cell(checks, "independent_variant_agreement", cells[0], cells[1])
    totals = Counter()
    for cell in cells:
        totals.update(cell["accounting"])
    family_summary = {}
    for family in sorted({check["family"] for check in checks}):
        rows = [r for r in checks if r["family"] == family]
        family_summary[family] = {"checks": len(rows), "passed": sum(r["pass"] for r in rows),
                                  "maximum_absolute_difference": max(r["absolute_difference"] for r in rows)}
    warnings = [r for r in adaptive.records if r["warning"]]
    result = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
              "protocol_sha256": sha(PROTOCOL), "inputs": protocol["inputs"], "independent_cells": cells,
              "independence": "Different internal agent and separately authored Price-integral/one-dimensional adaptive reconstruction; non-blind",
              "checks": checks, "passed": sum(c["pass"] for c in checks), "failures": [c for c in checks if not c["pass"]],
              "all_pass": all(c["pass"] for c in checks) and not warnings,
              "family_summary": family_summary, "adaptive_integral_records": adaptive.records,
              "adaptive_warnings": warnings,
              "accounting": {**dict(totals), "adaptive_scalar_1d_integrals": len(adaptive.records),
                             "adaptive_function_evaluations": sum(r["function_evaluations"] for r in adaptive.records),
                             "deterministic_rule_outputs": len(cells)*len(TEACHERS)*4,
                             "new_random_samples": 0, "new_labels": 0, "new_banks": 0, "new_data_fits": 0},
              "qualification_only_not_strict_total_integral_certificate": True,
              "does_not_requalify_a2_or_a4": True,
              "posthoc_point_does_not_confirm_grid_or_continuous_maximum": True,
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}}
    for path, digest in protocol["inputs"].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError("Comparison input changed during execution: "+path)
    save_new(RESULT, result)
    report(result)
    print(json.dumps({"all_pass": result["all_pass"], "checks": len(checks), "passed": result["passed"],
                      "adaptive_warnings": len(warnings), "accounting": result["accounting"],
                      "output_sha256": sha(RESULT)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
