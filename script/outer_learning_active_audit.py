"""Independent ACTIVE_TARGET finite-table audit; write/check, no RNG or integration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "artifacts/nuisance_learning/ACTIVE_TARGET"
OUT = ROOT / "artifacts/nuisance_learning/ACTIVE_TARGET_AUDIT"
DOC = ROOT / "docs/08_research_研究工作/外层精度与有限样本伪相关_20261001/ACTIVE_TARGET_AUDIT.md"
RHOS = [3 * i / 20 for i in range(-5, 6)]
NS, KS, BS, VARIANTS = [64, 256, 1024], [2, 4], [32, 512, 8192], ["low", "primary", "high"]
TOL = 1e-10
PLAN = {"schema": 1, "Decimal_precision": 80,
        "inversion": "solve same/different two linear equations from old D(rho), D(-rho), observed teacher pair; then s/N mixture",
        "state_argmin_sets": 8082, "score_values": 8082 * 11,
        "state_qualifications": 2694, "K2_null_interface_checks": 33,
        "expected_risk_groups": 18, "actual_lookups": 96, "RB_pairings": 288,
        "score_absolute_tolerance": TOL, "argmin_and_risk_class_comparison": "exact sets",
        "reference_risk_weights": "exact Fraction from fixed stored float risk and binomial integers",
        "new_RNG_calls": 0, "new_Gaussian_integrals": 0, "new_training_models": 0,
        "strict_Gaussian_integral_certificate": False,
        "failure_rule": "preserve and stop, no tolerance/coverage changes or automatic retry"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data, replace=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2); f.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def fraction(r):
    return Fraction(int(r["numerator"]), int(r["denominator"]))


def fr(q):
    return {"numerator": str(q.numerator), "denominator": str(q.denominator), "float_display": float(q)}


class Audit:
    def __init__(self):
        self.count = 0; self.maxima = {}

    def eq(self, a, b, label):
        self.count += 1
        if a != b:
            raise AssertionError(f"{label}: {a!r} != {b!r}")

    def close(self, a, b, label, category="score"):
        error = abs(float(a) - float(b))
        self.maxima[category] = max(error, self.maxima.get(category, 0.0))
        self.eq(math.isfinite(error) and error <= TOL, True, f"{label}: error={error}")


def inputs():
    return {x["original"]: x for x in read(SCI / "FREEZE.json")["inputs"]}


def source_input(mapping, original):
    return read(ROOT / mapping[original]["snapshot"])


def verify(audit):
    for name, expected in read(SCI / "RESULTS_FREEZE.json")["files"].items():
        audit.eq(sha(SCI / name), expected, "sealed active output " + name)
    freeze = read(SCI / "FREEZE.json")
    for row in freeze["inputs"]:
        audit.eq(sha(ROOT / row["snapshot"]), row["sha256"], "frozen input " + row["original"])
    audit.eq(sha(SCI / "CONFIG.json"), freeze["config_sha256"], "config seal")
    audit.eq(sha(SCI / "UNIT_CHECK.json"), freeze["unit_sha256"], "unit seal")


def reconstruct(audit, mapping):
    old = source_input(mapping, "artifacts/weak_emission/NUMERIC/GRID.json")["rows"]
    coefs = {(x["variant"], x["K"]): x for x in read(SCI / "COEFFICIENTS.json")["rows"]}
    audit.eq(len(coefs), 6, "six coefficient groups")
    audit.eq(read(SCI / "ENUMERATION.json")["count_state_argmins"], 8082, "saved enumeration count")
    audit.eq(read(SCI / "ENUMERATION.json")["signed_ties_not_forced_to_one_index"], True, "retain signed ties")
    produced = {(g["N"], g["K"], g["variant"]): g["states"] for g in read(SCI / "ENUMERATION.json")["groups"]}
    audit.eq(len(produced), 18, "18 state groups")
    expected, scores, solved_rows = {}, {}, []
    with localcontext() as ctx:
        ctx.prec = 80
        for variant in VARIANTS:
            grid = {r["rho"]: r for r in old if r["a"] == 1.0 and r["variant"] == variant}
            for K in KS:
                same, different = [], []
                c = coefs[(variant, K)]
                for i, rho in enumerate(RHOS):
                    plus, minus = grid[rho], grid[-rho]
                    teacher = Decimal.from_float(float(plus["teacher_spin_moment"]))
                    audit.eq(plus["teacher_spin_moment"], minus["teacher_spin_moment"], "mirror teacher observed moment")
                    d1 = Decimal.from_float(float(plus["score_minus_rho0"][plus["K"].index(K)]))
                    d2 = Decimal.from_float(float(minus["score_minus_rho0"][minus["K"].index(K)]))
                    p, q = (1 + teacher) / 2, (1 - teacher) / 2
                    determinant = p * p - q * q
                    audit.eq(determinant != 0, True, "nonzero observed-moment determinant")
                    ls = (p * d1 - q * d2) / determinant
                    ld = (p * d2 - q * d1) / determinant
                    same.append(ls); different.append(ld)
                    audit.close(p * ls + q * ld, d1, "first linear equation", "linear_residual")
                    audit.close(q * ls + p * ld, d2, "second linear equation", "linear_residual")
                    audit.close(c["even"][i], (ls + ld) / 2, "saved even coefficient", "coefficient")
                    audit.close(c["odd"][i], (ls - ld) / 2, "saved odd coefficient", "coefficient")
                    audit.eq(c["old_delta_F"][i], float(d1), "original D value")
                    audit.eq(c["teacher_spin_moment"], float(teacher), "observed divisor")
                solved_rows.append({"variant": variant, "K": K, "same_contrasts": [str(x) for x in same], "different_contrasts": [str(x) for x in different]})
                for N in NS:
                    published = produced[(N, K, variant)]
                    audit.eq(len(published), N + 1, "state length")
                    for s in range(N + 1):
                        curve = [(Decimal(s) * x + Decimal(N - s) * y) / Decimal(N) for x, y in zip(same, different)]
                        minimum = min(curve)
                        indices = [i for i, x in enumerate(curve) if x == minimum]
                        absolute = sorted({abs(i - 5) for i in indices})
                        state = published[s]
                        t = Fraction(2 * s - N, N)
                        for i in range(11):
                            saved_score = c["even"][i] + float(t) * c["odd"][i]
                            audit.close(saved_score, curve[i], "all reconstructed score points")
                        audit.eq(state["same_count"], s, "same-count ordering")
                        audit.close(state["empirical_pair"], float(t), "empirical pair", "pair")
                        audit.eq(state["argmin_indices"], indices, "Decimal argmin exact set")
                        audit.eq(state["argmin_rhos"], [RHOS[i] for i in indices], "argmin rho set")
                        audit.eq(state["absolute_grid_coordinates"], absolute, "null risk class")
                        mirror = len(indices) > 1 and sorted(indices) == sorted(10 - i for i in indices)
                        audit.eq(state["exact_signed_mirror_tie"], mirror, "signed tie flag")
                        audit.close(state["minimum_contrast"], minimum, "minimum score", "minimum")
                        key = (N, K, variant, s)
                        expected[key] = {"indices": indices, "absolute": absolute, "minimum": float(minimum)}
                        scores[key] = [float(x) for x in curve]
    audit.eq(len(expected), 8082, "all 8082 states")
    save(OUT / "LINEAR_SYSTEMS.json", {"Decimal_precision": 80, "rows": solved_rows})
    save(OUT / "STATE_RECONSTRUCTION.json", {"rows": [
        {"N": key[0], "K": key[1], "variant": key[2], "same_count": key[3], **value}
        for key, value in expected.items()]})
    return expected, scores


def qualification(audit, mapping, states, scores):
    public = read(SCI / "QUALIFICATION.json")
    rows = {(x["N"], x["K"], x["same_count"]): x for x in public["count_state_rows"]}
    audit.eq(len(rows), 2694, "2694 unique qualification rows")
    passed = {}
    for N in NS:
        for K in KS:
            for s in range(N + 1):
                curves = [scores[(N, K, v, s)] for v in VARIANTS]
                sets = [states[(N, K, v, s)]["indices"] for v in VARIANTS]
                absolute = [states[(N, K, v, s)]["absolute"] for v in VARIANTS]
                discrepancy = max(max(x[i] for x in curves) - min(x[i] for x in curves) for i in range(11))
                row = rows[(N, K, s)]
                same = all(x == sets[0] for x in sets)
                cls = all(x == absolute[0] for x in absolute)
                audit.close(row["max_pointwise_curve_difference"], discrepancy, "variant pointwise difference", "qualification")
                audit.eq(row["score_limit"], TOL, "unchanged score gate")
                audit.eq(row["score_qualified"], discrepancy <= TOL, "score qualified")
                audit.eq(row["variant_argmin_sets"], sets, "all variant signed sets")
                audit.eq(row["variant_absolute_argmin_sets"], absolute, "all variant risk classes")
                audit.eq(row["signed_sets_identical"], same, "signed set equality flag")
                audit.eq(row["choice_qualified"], cls, "risk-class qualification")
                passed[(N, K, s)] = cls and discrepancy <= TOL
                audit.eq(row["passed"], passed[(N, K, s)], "state combined qualification")
    fourier = source_input(mapping, "artifacts/nuisance_dimensions/NUMERIC/GRID.json")["rows"]
    other = {x["rho"]: x for x in fourier if x["a"] == 1 and x["theta"] == 0 and x["n"] == 0 and x["variant"] == "primary"}
    cross = {(r["variant"], r["rho"]): r for r in public["K2_null_cross_rows"]}
    audit.eq(len(cross), 33, "33 unique K2 interface rows")
    ok = []
    for v in VARIANTS:
        for i, rho in enumerate(RHOS):
            reconstructed = scores[(64, 2, v, 32)][i]
            value = other[rho]["delta_F"]
            error = abs(reconstructed - value)
            row = cross[(v, rho)]
            audit.close(row["reconstructed_null"], reconstructed, "null reconstruction", "K2_interface")
            audit.eq(row["frozen_Fourier_null"], value, "old Fourier value")
            audit.close(row["error"], error, "null interface error", "K2_interface")
            audit.eq(row["limit"], TOL, "unchanged K2 interface gate")
            audit.eq(row["passed"], error <= TOL, "null interface gate")
            ok.append(error <= TOL)
    audit.eq(public["all_passed"], all(passed.values()) and all(ok), "qualification aggregate")
    return passed, all(passed.values()) and all(ok)


def risk_expectations(audit, mapping, states):
    reference = source_input(mapping, "artifacts/nuisance_learning/REFERENCE.json")
    risk = [Fraction.from_float(float(x)) for x in reference["reference_joint_KL"]]
    mle_source = {x["N"]: x for x in source_input(mapping, "artifacts/nuisance_learning/MLE_ENUMERATION.json")["rows"]}
    mle = {}
    for N in NS:
        ids = mle_source[N]["indices_for_same_count_0_through_N"]
        audit.eq(len(ids), N + 1, "fixed MLE state count")
        weights = [math.comb(N, s) for s in range(N + 1)]
        ex = sum((w * risk[i] for w, i in zip(weights, ids)), Fraction(0)) / 2 ** N
        upper = Fraction(9 * sum(w * (i - 5) ** 2 for w, i in zip(weights, ids)), 12350 * 2 ** N)
        audit.eq(fraction(mle_source[N]["true_expected_KL_upper"]), upper, "old MLE analytic upper")
        mle[N] = (ex, upper)
    published = {(r["N"], r["K"], r["variant"]): r for r in read(SCI / "EXPECTED_RISKS.json")["rows"]}
    audit.eq(len(published), 18, "18 expected risk groups")
    result = []
    for N in NS:
        for K in KS:
            for v in VARIANTS:
                low, high = Fraction(0), Fraction(0)
                squares, multi = 0, []
                for s in range(N + 1):
                    state = states[(N, K, v, s)]
                    ids, w = state["indices"], math.comb(N, s)
                    low += w * min(risk[i] for i in ids)
                    high += w * max(risk[i] for i in ids)
                    squares += w * max((i - 5) ** 2 for i in ids)
                    if len(state["absolute"]) > 1:
                        multi.append(s)
                low /= 2 ** N; high /= 2 ** N
                upper = Fraction(9 * squares, 12350 * 2 ** N)
                ex_mle, up_mle = mle[N]
                row = published[(N, K, v)]
                for key, q in [("exact_reference_expected_KL_min", low), ("exact_reference_expected_KL_max", high),
                               ("true_expected_KL_upper_for_any_argmin_selection", upper),
                               ("reference_proxy_minus_table_MLE_min", low - ex_mle), ("reference_proxy_minus_table_MLE_max", high - ex_mle)]:
                    audit.eq(fraction(row[key]), q, "risk Fraction " + key)
                    audit.close(row[key]["float_display"], float(q), "risk float display", "risk_display")
                audit.eq(int(row["exact_binomial_max_square_sum"]), squares, "worst-case square sum")
                audit.eq(row["counts_with_multiple_magnitude_minima"], multi, "multiple magnitude tie counts")
                audit.eq(fraction(row["table_MLE"]["exact_reference_expected_KL"]), ex_mle, "MLE fixed-reference exact expectation")
                audit.eq(fraction(row["table_MLE"]["true_expected_KL_upper"]), up_mle, "MLE exact upper")
                audit.eq(row["table_MLE"]["argmin_recomputed"], False, "old MLE lookup scope")
                audit.eq(row["table_MLE"]["original_reference_expected_KL"], mle_source[N]["reference_expected_KL"], "original MLE reference")
                audit.eq(row["difference_of_analytic_upper_bounds_is_not_a_risk_gap_bound"], True, "upper bound scope")
                result.append({"N": N, "K": K, "variant": v, "reference_min": fr(low), "reference_max": fr(high),
                               "true_upper": fr(upper), "table_reference": fr(ex_mle), "table_true_upper": fr(up_mle)})
    save(OUT / "RISK_ARITHMETIC.json", {"rows": result})
    return result, reference


def lookups(audit, mapping, states, qualified, reference):
    targets = {(r["stream"], r["N"], r["K"]): r for r in read(SCI / "ACTUAL_TARGETS.json")["rows"]}
    pairs = {(r["stream"], r["N"], r["K"], r["B"]): r for r in read(SCI / "RB_COMPARISONS.json")["rows"]}
    audit.eq(len(targets), 96, "96 target lookups")
    audit.eq(len(pairs), 288, "288 RB pairings")
    risk = reference["reference_joint_KL"]
    summaries = []
    for stream in range(16):
        training = source_input(mapping, f"artifacts/nuisance_learning/s{stream:02d}/TRAINING.json")
        models = {x["id"]: x for x in training["curves"]}
        for ni, N in enumerate(NS):
            counts = training["active_pattern_counts_by_N"][ni]
            s = counts[0] + counts[3]
            audit.eq(sum(counts), N, "actual pattern total")
            audit.eq(s, models[f"table_MLE_N{N}"]["same_count"], "actual same-count lookup")
            for K in KS:
                state = states[(N, K, "primary", s)]
                ids = state["indices"]
                expected = {"stream": stream, "N": N, "K": K, "same_count": s,
                            "empirical_pair": (2 * s - N) / N,
                            "target_indices": ids, "target_rhos": [RHOS[i] for i in ids],
                            "target_absolute_grid_coordinates": state["absolute"],
                            "variant_target_sets": {v: states[(N, K, v, s)]["indices"] for v in VARIANTS},
                            "count_state_qualified": qualified[(N, K, s)],
                            "target_reference_risk_min": min(risk[i] for i in ids),
                            "target_reference_risk_max": max(risk[i] for i in ids)}
                audit.eq(targets[(stream, N, K)], expected, "complete target record")
                for B in BS:
                    rb = models[f"RB_N{N}_B{B}_K{K}"]
                    i = rb["index"]
                    expected_pair = {**expected, "B": B, "original_RB_index": i, "original_RB_rho": rb["rho"],
                                     "original_RB_reference_KL": risk[i], "exact_target_set_membership": i in ids,
                                     "target_absolute_rho_membership": abs(i - 5) in state["absolute"],
                                     "RB_minus_target_reference_risk_min": risk[i] - expected["target_reference_risk_max"],
                                     "RB_minus_target_reference_risk_max": risk[i] - expected["target_reference_risk_min"],
                                     "descriptive_not_independent_replicate": True}
                    audit.eq(pairs[(stream, N, K, B)], expected_pair, "complete RB pairing")
                summaries.append(expected)
    return summaries


def report(summary):
    text = f"""# ACTIVE_TARGET 独立有限表算术审计

审计通过。先保存 PLAN/START 的固定覆盖、审计源码和输入hash；没有导入生产计算函数。审计源码 SHA256：`{sha(__file__)}`。全部输入来自 ACTIVE_TARGET 的冻结快照与封存输出，未修改原件。

从旧weak表的 D*(rho)、D*(-rho)及各变体自己的观察pair矩，以80位Decimal直接解same/different两条线性方程，再按s/N混合。重建全部8082状态的88,902个分数，全部精确argmin集合与绝对rho风险类一致；最大分数差 `{summary['max_errors'].get('score', 0):.17g}`，门为1e-10。2694个跨变体资格和33个K2/null Fourier接口判断全部一致。

18组固定Binomial权重的参考期望风险上下端、最坏平局选择的解析upper和表格MLE对照均用独立Fraction重算，逐项精确一致。全部96个实际count lookup及288个原RB配对完整重建，保留带符号集合与风险区间，不按最好风险选平局。

0RNG、0新Gaussian积分、0新训练或标签。8082个状态是已有表的确定算术重建，不是新科学拟合。参考系数与风险仍只具有原数值资格；本审计不认证真实Gaussian B无穷目标或连续最优点。该包为观察原K2/RB结果后的事后诊断，不新增置信家族、不修改原门，也不把paired风险差称为因果百分比。

`check`只读核公开输入/输出hash与完成收据，不在每次检查中重算全部表算术；本包不依赖本地NPZ。没有失败、改门或自动重试。
"""
    with DOC.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write():
    if (OUT / "START.json").exists() or DOC.exists():
        raise FileExistsError("Audit already started/report exists; no replacement or retry.")
    if read(SCI / "STATE.json")["state"] != "complete":
        raise RuntimeError("Wait for complete production inputs.")
    mapping = inputs()
    source = mapping["script/outer_learning_active_target.py"]["sha256"]
    save(OUT / "PLAN.json", PLAN)
    save(OUT / "START.json", {"created": now(), "audit_source_sha256": sha(__file__),
                               "producer_source_sha256": source, "plan_sha256": sha(OUT / "PLAN.json"),
                               "source_freeze_sha256": sha(SCI / "FREEZE.json"),
                               "results_freeze_sha256": sha(SCI / "RESULTS_FREEZE.json"),
                               "inputs": mapping, "public_outputs": read(SCI / "RESULTS_FREEZE.json")["files"]})
    save(OUT / "STATE.json", {"state": "running", "started": now()})
    audit, began = Audit(), time.perf_counter()
    try:
        verify(audit)
        cfg = read(SCI / "CONFIG.json")
        audit.eq((cfg["rho_grid"], cfg["Ns"], cfg["Ks"], cfg["variants"], cfg["score_variant_tolerance"]),
                 (RHOS, NS, KS, VARIANTS, TOL), "fixed diagnostic contract")
        states, scores = reconstruct(audit, mapping)
        qualified, all_qualified = qualification(audit, mapping, states, scores)
        expectations, reference = risk_expectations(audit, mapping, states)
        actual = lookups(audit, mapping, states, qualified, reference)
        original = read(SCI / "RESULTS.json")
        audit.eq(original["qualified"], all_qualified, "results qualification")
        for name, number in [("count_state_argmins", 8082), ("unique_count_K_states", 2694),
                             ("primary_actual_count_lookups", 96), ("original_RB_pairings", 288),
                             ("old_MLE_state_index_lookups", 1347), ("new_Gaussian_integrals", 0),
                             ("new_labels", 0), ("new_banks", 0), ("new_training_models", 0), ("new_confidence_families", 0)]:
            audit.eq(original["ledger"][name], number, "ledger " + name)
        audit.eq(original["post_hoc"], True, "post-hoc scope")
        audit.eq(original["original_primary_and_CP_unchanged"], True, "original gates preserved")
        summary = {"passed": True, "numerical_qualification_reconfirmed": all_qualified,
                   "checks": audit.count, "max_errors": audit.maxima, "coverage": PLAN,
                   "primary_N1024_expectations": [r for r in expectations if r["N"] == 1024 and r["variant"] == "primary"],
                   "actual_lookups_rebuilt": len(actual), "wall_seconds": time.perf_counter() - began,
                   "failures": 0, "automatic_retries": 0}
        save(OUT / "SUMMARY.json", summary)
        report(summary)
        save(OUT / "STATE.json", {"state": "complete", "finished": now()}, replace=True)
        files = {p.name: sha(p) for p in sorted(OUT.glob("*.json")) if p.name != "RECEIPT.json"}
        save(OUT / "RECEIPT.json", {"passed": True, "audit_source_sha256": sha(__file__), "producer_source_sha256": source,
                                   "source_freeze_sha256": sha(SCI / "FREEZE.json"), "results_freeze_sha256": sha(SCI / "RESULTS_FREEZE.json"),
                                   "files": files, "report_sha256": sha(DOC), "checks": audit.count, "coverage": PLAN})
        print(json.dumps({"passed": True, "checks": audit.count, "max_errors": audit.maxima, "wall_seconds": summary["wall_seconds"]}))
    except Exception as exc:
        save(OUT / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc), "checks": audit.count,
                                   "max_errors": audit.maxima, "automatic_retry": False})
        save(OUT / "STATE.json", {"state": "failed", "finished": now()}, replace=True)
        raise


def check():
    receipt, start = read(OUT / "RECEIPT.json"), read(OUT / "START.json")
    audit = Audit()
    audit.eq(receipt["passed"], True, "passed receipt")
    audit.eq(read(OUT / "STATE.json")["state"], "complete", "complete state")
    audit.eq(sha(__file__), receipt["audit_source_sha256"], "audit source")
    audit.eq(sha(__file__), start["audit_source_sha256"], "START source")
    audit.eq(sha(SCI / "FREEZE.json"), receipt["source_freeze_sha256"], "source freeze")
    audit.eq(sha(SCI / "RESULTS_FREEZE.json"), receipt["results_freeze_sha256"], "results freeze")
    for name, expected in receipt["files"].items():
        audit.eq(sha(OUT / name), expected, "audit output " + name)
    audit.eq(sha(DOC), receipt["report_sha256"], "audit report")
    audit.eq(read(OUT / "PLAN.json"), PLAN, "fixed coverage")
    audit.eq(sha(OUT / "PLAN.json"), start["plan_sha256"], "plan seal")
    verify(audit)
    print(json.dumps({"passed": True, "checks": audit.count, "local_arrays_required": False,
                      "scope": "public frozen inputs/outputs and completed receipt; no table replay"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["write", "check"])
    parser.add_argument("--workspace", action="store_true", help="Accepted; this audit has only public inputs.")
    args = parser.parse_args()
    write() if args.mode == "write" else check()
