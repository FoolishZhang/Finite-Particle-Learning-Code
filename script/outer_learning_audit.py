"""Independent shared-bank learning audit: write/check [--workspace]."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "artifacts/nuisance_learning"
OUT = SCI / "LEARNING_AUDIT"
DOC = ROOT / "docs/08_research_研究工作/外层精度与有限样本伪相关_20261001/LEARNING_AUDIT.md"
SOURCE_SHA = "f77e390122ed200438ba383d61eac5d392d1acd10070507119bdc633c120292e"
NS, BS, KS = [64, 256, 1024], [32, 512, 8192], [2, 4]
RHOS = [3 * i / 20 for i in range(-5, 6)]
BANKS = [0, 1, 31, 32, 127, 128, 511, 512, 1023, 2047, 4095, 4096, 7000, 8000, 8190, 8191]
TOL = 1e-10
PLAN = {
    "schema": 1,
    "independent_of_producer_functions": True,
    "streams": 16,
    "fixed_bank_indices": BANKS,
    "all_actual_labels_per_fixed_bank": 1024,
    "all_rhos": RHOS, "Ns": NS, "Ks": KS, "Bs": BS,
    "original_score_replay": "sum coordinate log-sigmoid; cache unchanged nuisance; no noise GEMM/logcosh",
    "raw_contrast_values_replayed": 16896,
    "RB_contrast_values_replayed": 16896,
    "full_saved_bank_arithmetic": "576 raw/RB curves and 48 table training curves; 624 choices",
    "table_state_argmins": 1347,
    "table_state_reference": "80-digit Decimal logarithms of the exact stored float probabilities",
    "statistics": "all paired cells; single exact binomial; two downward dyadic CP; exact Fraction risk consequences",
    "full_input_RNG_replay": True,
    "Gaussian_scalars_replayed": 134217728,
    "label_bits_replayed": 4194304,
    "absolute_score_and_curve_tolerance": TOL,
    "selection_indices": "exact agreement",
    "RNG_values": "bitwise equality",
    "integer_and_Fraction_statistics": "exact equality",
    "new_independent_scientific_samples": 0,
    "new_Gaussian_integrals": 0,
    "new_scientific_fits": 0,
    "unreplayed_original_banks": 16 * (8192 - len(BANKS)),
    "failure_rule": "retain first failure and stop; no tolerance or coverage adaptation",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def rational(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def frac_record(q):
    return {"numerator": str(q.numerator), "denominator": str(q.denominator), "float_display": float(q)}


class Audit:
    def __init__(self):
        self.checks = 0
        self.maxima = {}

    def exact(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def close(self, actual, expected, label, category="float", tolerance=TOL):
        a, b = np.asarray(actual), np.asarray(expected)
        self.exact(a.shape == b.shape, label + ": shape")
        self.exact(bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b))), label + ": finite")
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.maxima[category] = max(self.maxima.get(category, 0.0), error)
        self.exact(error <= tolerance, f"{label}: error {error} > {tolerance}")


def public_inputs(audit, results_freeze):
    for name, expected in results_freeze["files"].items():
        audit.exact(digest(SCI / name) == expected, "science public hash " + name)
    freeze = read(SCI / "FREEZE.json")
    for item in freeze["inputs"]:
        audit.exact(digest(ROOT / item["snapshot"]) == item["sha256"], "frozen input " + item["snapshot"])
    source = next(x for x in freeze["inputs"] if x["original"] == "script/outer_learning_train.py")
    audit.exact(source["sha256"] == SOURCE_SHA, "fixed production source version")
    for name, key in [("CONFIG.json", "config_sha256"), ("REFERENCE.json", "reference_sha256"), ("UNIT_CHECK.json", "unit_sha256")]:
        audit.exact(digest(SCI / name) == freeze[key], "source freeze " + name)
    selection = read(SCI / "SELECTION_FREEZE.json")
    for path, expected in selection["files"].items():
        audit.exact(digest(ROOT / path) == expected, "selection seal " + path)


def independent_table(audit, reference):
    """Only fixed-table arithmetic: no candidate Gaussian integration."""
    result = {}
    published = {x["N"]: x for x in read(SCI / "MLE_ENUMERATION.json")["rows"]}
    with localcontext() as context:
        context.prec = 80
        moments = [Decimal.from_float(float(r)) for r in reference["spin_moments"]]
        plus = [(Decimal(1) + r).ln() for r in moments]
        minus = [(Decimal(1) - r).ln() for r in moments]
        for N in NS:
            indices, curves, gaps = [], [], []
            for s in range(N + 1):
                curve = [-(Decimal(s) * p + Decimal(N - s) * q) / Decimal(N) for p, q in zip(plus, minus)]
                ix = min(range(11), key=lambda i: (curve[i], i))
                ordered = sorted(curve)
                indices.append(ix)
                curves.append([float(x) for x in curve])
                gaps.append(ordered[1] - ordered[0])
            square_sum = sum(math.comb(N, s) * (ix - 5) ** 2 for s, ix in enumerate(indices))
            upper = Fraction(9 * square_sum, 12350 * 2 ** N)
            reference_mean = sum((Fraction(math.comb(N, s), 2 ** N) * Fraction.from_float(float(reference["reference_joint_KL"][ix])) for s, ix in enumerate(indices)), Fraction(0))
            pub = published[N]
            audit.exact(pub["indices_for_same_count_0_through_N"] == indices, f"all table states N{N}")
            audit.exact(int(pub["exact_binomial_integer_square_sum"]) == square_sum, f"table square sum N{N}")
            audit.exact(rational(pub["true_expected_KL_upper"]) == upper, f"table risk upper N{N}")
            audit.close(pub["reference_expected_KL"], float(reference_mean), f"table reference expectation N{N}", "reference_mean")
            audit.close(pub["binomial_float_mass_sum"], 1.0, f"table mass N{N}")
            audit.exact(pub["count_state_argmin_evaluations"] == N + 1, f"table state count N{N}")
            result[N] = {"indices": indices, "curves": curves, "upper": upper,
                         "minimum_decimal_runner_up_gap": str(min(gaps)), "square_sum": square_sum,
                         "reference_expected_KL": float(reference_mean)}
    save(OUT / "TABLE_STATES.json", {"count": 1347, "precision": 80, "rows": [
        {"N": N, "indices": v["indices"], "upper": frac_record(v["upper"]),
         "minimum_decimal_runner_up_gap": v["minimum_decimal_runner_up_gap"],
         "square_sum": str(v["square_sum"]), "reference_expected_KL": v["reference_expected_KL"]}
        for N, v in result.items()]})
    return result


def loss(log_likelihood, K):
    x = log_likelihood[:, :K]
    maximum = x.max(axis=1)
    return -maximum - np.log(np.exp(x - maximum[:, None]).sum(axis=1)) + math.log(K)


def replay_scores(audit, labels, prior, arrays, stream):
    """Only the 16 predeclared banks. Every label retains its actual coordinates."""
    spin = labels.astype(np.float64) * 2.0 - 1.0
    for b in BANKS:
        z = prior[b]
        nuisance = -np.logaddexp(0.0, -spin[:, None, 2:] * z[None, :, 2:]).sum(axis=2)
        baseline_active = -np.logaddexp(0.0, -spin[:, None, :2] * z[None, :, :2]).sum(axis=2)
        raw_base = {K: loss(nuisance + baseline_active, K) for K in KS}
        rb_base = {K: loss(baseline_active, K) for K in KS}
        for ki, K in enumerate(KS):
            for ni, N in enumerate(NS):
                audit.close(arrays["raw_rho0"][ni, ki, b], math.fsum(raw_base[K][:N]) / N,
                            f"s{stream} b{b} raw baseline N{N} K{K}", "raw_baseline")
                audit.close(arrays["RB_rho0"][ni, ki, b], math.fsum(rb_base[K][:N]) / N,
                            f"s{stream} b{b} RB baseline N{N} K{K}", "RB_baseline")
        for gi, rho in enumerate(RHOS):
            eta = np.stack([z[:, 0], rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]], axis=1)
            active = -np.logaddexp(0.0, -spin[:, None, :2] * eta[None, :, :]).sum(axis=2)
            for ki, K in enumerate(KS):
                raw_diff = loss(nuisance + active, K) - raw_base[K]
                rb_diff = loss(active, K) - rb_base[K]
                for ni, N in enumerate(NS):
                    audit.close(arrays["raw"][ni, ki, gi, b], math.fsum(raw_diff[:N]) / N,
                                f"s{stream} b{b} raw rho{gi} N{N} K{K}", "raw_original_scores")
                    audit.close(arrays["RB"][ni, ki, gi, b], math.fsum(rb_diff[:N]) / N,
                                f"s{stream} b{b} RB rho{gi} N{N} K{K}", "RB_original_scores")


def audit_stream(audit, stream, table):
    folder = SCI / f"s{stream:02d}"
    record = read(folder / "TRAINING.json")
    with np.load(folder / "INPUT.npz", allow_pickle=False) as data:
        labels, prior = data["labels"], data["prior"]
    with np.load(folder / "BANK_CONTRASTS.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in ("raw", "RB", "raw_rho0", "RB_rho0")}
    audit.exact(labels.shape == (1024, 256) and labels.dtype == np.uint8, "label shape/type")
    audit.exact(prior.shape == (8192, 4, 256) and prior.dtype == np.float64, "prior shape/type")
    audit.exact(bool(np.all(np.isfinite(prior))) and bool(np.all(labels <= 1)), "finite/binary inputs")
    for method in ("raw", "RB"):
        audit.exact(arrays[method].shape == (3, 2, 11, 8192), method + " array shape")
        audit.exact(arrays[method + "_rho0"].shape == (3, 2, 8192), method + " reference shape")
    for role, key, actual in [(0, "labels", labels), (1, "prior", prior)]:
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2037100101, stream, role])))
        audit.exact(generator.bit_generator.state == record["RNG_initial"][key], f"RNG initial s{stream} {key}")
        replay = (generator.integers(0, 2, size=(1024, 256), dtype=np.uint8) if role == 0
                  else generator.standard_normal((8192, 4, 256)))
        audit.exact(np.array_equal(actual.view(np.uint8), replay.view(np.uint8)), f"full bitwise RNG s{stream} {key}")
        audit.exact(generator.bit_generator.state == record["RNG_final"][key], f"RNG final s{stream} {key}")
        del replay
    models = {r["id"]: r for r in record["curves"]}
    audit.exact(len(models) == len(record["curves"]) == 39, f"39 unique models s{stream}")
    selections = {}
    for ni, N in enumerate(NS):
        count = [sum(int(2 * x[0] + x[1]) == p for x in labels[:N]) for p in range(4)]
        audit.exact(count == record["active_pattern_counts_by_N"][ni], "active counts")
        for ki, K in enumerate(KS):
            for B in BS:
                for method in ("raw", "RB"):
                    key = f"{method}_N{N}_B{B}_K{K}"
                    curve = [math.fsum(float(x) for x in arrays[method][ni, ki, g, :B]) / B for g in range(11)]
                    ref = math.fsum(float(x) for x in arrays[method + "_rho0"][ni, ki, :B]) / B
                    ix = min(range(11), key=lambda j: (curve[j], j))
                    saved = models[key]
                    audit.close(saved["contrast_curve"], curve, key + " curve", "saved_bank_curves")
                    audit.close(saved["rho0_objective"], ref, key + " rho0", "saved_bank_curves")
                    audit.close(saved["objective_curve"], [x + ref for x in curve], key + " full objective", "saved_bank_curves")
                    audit.exact(saved["index"] == ix and saved["grid_coordinate"] == ix - 5 and saved["rho"] == RHOS[ix], key + " index")
                    audit.exact(saved["method"] == method and saved["N"] == N and saved["B"] == B and saved["K"] == K, key + " metadata")
                    published_curve = saved["contrast_curve"]
                    minimum = min(published_curve)
                    audit.exact(saved["exact_ties"] == [j for j, x in enumerate(published_curve) if x == minimum], key + " exact ties")
                    audit.exact(saved["near_ties_diagnostic"] == [j for j, x in enumerate(published_curve) if x <= minimum + 1e-12], key + " near ties")
                    audit.exact(saved["near_ties_used_for_selection"] is False, key + " no near tie selection")
                    audit.close(saved["minimum_contrast"], curve[ix], key + " minimum", "saved_bank_curves")
                    selections[key] = ix
        same = int(np.sum(labels[:N, 0] == labels[:N, 1]))
        key = f"table_MLE_N{N}"
        saved, expected = models[key], table[N]
        ix = expected["indices"][same]
        audit.exact(saved["same_count"] == same and saved["index"] == ix and saved["grid_coordinate"] == ix - 5 and saved["rho"] == RHOS[ix], key + " table exact index/count")
        audit.close(saved["contrast_curve"], expected["curves"][same], key + " Decimal curve", "table_curves")
        audit.close(saved["objective_curve"], [x + math.log(4) for x in expected["curves"][same]], key + " Decimal full curve", "table_curves")
        selections[key] = ix
    replay_scores(audit, labels, prior, arrays, stream)
    summary = {"stream": stream, "passed": True, "choices": selections,
               "full_RNG_replay": {"Gaussian_scalars": 8192 * 4 * 256, "label_bits": 1024 * 256},
               "original_banks_replayed": BANKS, "raw_score_contrast_values": 1056,
               "RB_score_contrast_values": 1056, "remaining_original_banks_not_replayed": 8176}
    save(OUT / f"s{stream:02d}.json", summary)
    print(f"independent audit s{stream:02d}: RNG + 39 choices + fixed 16 banks passed", flush=True)
    return selections


def binomial_tail(p, successes, trials=16):
    """Independent exact probability-mass recurrence, not producer's integer polynomial."""
    if p == 0:
        return Fraction(int(successes == 0))
    if p == 1:
        return Fraction(1)
    mass = (1 - p) ** trials
    total = Fraction(0)
    for j in range(trials + 1):
        if j >= successes:
            total += mass
        if j < trials:
            mass *= Fraction(trials - j, j + 1) * p / (1 - p)
    return total


def cp(audit, row, count):
    lo, hi = Fraction(0), Fraction(1)
    if count:
        for _ in range(40):
            midpoint = (lo + hi) / 2
            if binomial_tail(midpoint, count) <= Fraction(1, 40):
                lo = midpoint
            else:
                hi = midpoint
        audit.exact(hi - lo == Fraction(1, 2 ** 40), "CP adjacent dyadic")
    audit.exact(rational(row["CP_lower"]) == lo, "CP exact lower")
    cert = row["dyadic_certificate"]
    audit.exact(cert["successes"] == count and cert["trials"] == 16 and cert["alpha"] == [1, 40], "CP metadata")
    audit.exact(Fraction(cert["lower_numerator"], cert["dyadic_denominator"]) == lo, "CP certificate lower")
    audit.exact(cert["dyadic_denominator"] == 2 ** 40, "CP denominator")
    if count:
        audit.exact(binomial_tail(lo, count) <= Fraction(1, 40) < binomial_tail(hi, count), "CP feasible/infeasible neighbors")
        rhs = 2 ** 640
        for name, p in [("lower_comparison_lhs", lo), ("upper_comparison_lhs", hi)]:
            integer = 40 * binomial_tail(p, count) * rhs
            audit.exact(integer.denominator == 1 and int(cert[name]) == integer.numerator, "CP independent integer certificate " + name)
        audit.exact(int(cert["comparison_rhs"]) == rhs and cert["upper_numerator"] == int(hi * 2 ** 40), "CP rhs/upper")
        audit.exact(cert["bisection_iterations"] == 40 and cert["zero_case"] is False, "CP iterations")
    else:
        audit.exact(cert["zero_case"] is True, "CP zero case")
    return lo


def all_statistics(audit, selections, table, reference):
    rows = read(SCI / "EVALUATION.json")["paired_cells"]
    indexed = {(x["stream"], x["n"], x["N"], x["B"], x["K"]): x for x in rows}
    audit.exact(len(indexed) == len(rows) == 576, "all paired cells unique")
    risks = reference["reference_joint_KL"]
    derived = {}
    for s, models in enumerate(selections):
        for n in (0, 254):
            for N in NS:
                for B in BS:
                    for K in KS:
                        rid = models[f"RB_N{N}_B{B}_K{K}"]
                        rawid = rid if n == 0 else models[f"raw_N{N}_B{B}_K{K}"]
                        mid = models[f"table_MLE_N{N}"]
                        gap = (rawid - 5) ** 2 - (rid - 5) ** 2
                        key = (s, n, N, B, K)
                        row = indexed[key]
                        for name, value in [("raw_index", rawid), ("RB_index", rid), ("table_MLE_index", mid),
                                            ("raw_grid_coordinate", rawid - 5), ("RB_grid_coordinate", rid - 5),
                                            ("integer_square_gap", gap), ("sufficient_true_KL_gain_gt_001", gap >= 4)]:
                            audit.exact(row[name] == value, f"paired {key} {name}")
                        for name, value in [("raw_reference_KL", risks[rawid]), ("RB_reference_KL", risks[rid]),
                                            ("table_MLE_reference_KL", risks[mid]), ("reference_raw_minus_RB", risks[rawid] - risks[rid])]:
                            audit.close(row[name], value, f"paired {key} {name}", "paired_risks")
                        if gap >= 0:
                            audit.exact(rational(row["analytic_gain_lower_if_nonnegative"]) == Fraction(gap, 3200), "paired analytic lower")
                        else:
                            audit.exact(row["analytic_gain_lower_if_nonnegative"] is None, "negative paired gap not assigned positive lower")
                        derived[key] = (rawid, rid, gap)
    result = read(SCI / "RESULTS.json")
    audit.exact(result["reference_MLE_expectations"] == read(SCI / "MLE_ENUMERATION.json")["rows"], "duplicated table expectations")
    test = result["primary_sign_test"]
    gaps = [derived[(s, 254, 1024, 8192, 4)][2] for s in range(16)]
    events = [g >= 4 for g in gaps]
    count = sum(events)
    pvalue = binomial_tail(Fraction(1, 2), count)
    audit.exact(test["successes"] == count and test["trials"] == 16 and test["events"] == events and test["integer_square_gaps"] == gaps, "primary events")
    audit.exact(rational(test["exact_one_sided_pvalue"]) == pvalue and test["alpha"] == [1, 20], "primary exact pvalue")
    audit.exact(test["passed"] == (pvalue <= Fraction(1, 20)), "primary gate")
    endpoints = []
    audit.exact(len(result["endpoint_CP_family"]) == 2, "two endpoint family")
    for row, K in zip(result["endpoint_CP_family"], KS):
        events = [abs(derived[(s, 254, 1024, 8192, K)][0] - 5) == 5 for s in range(16)]
        count = sum(events)
        audit.exact(row["K"] == K and row["events"] == events and row["successes"] == count and row["trials"] == 16, "endpoint events")
        lower = cp(audit, row, count)
        expected_upper = table[1024]["upper"]
        difference = lower / 128 - expected_upper
        audit.exact(rational(row["true_raw_expected_KL_lower"]) == lower / 128, "raw expected risk lower")
        audit.exact(rational(row["true_table_MLE_expected_KL_upper"]) == expected_upper, "table upper reused exactly")
        audit.exact(rational(row["true_expected_raw_minus_table_MLE_lower"]) == difference, "risk difference lower")
        audit.exact(row["derived_lower_positive"] == (difference > 0), "derived sign")
        endpoints.append({"K": K, "successes": count, "CP_lower": frac_record(lower), "mean_risk_gap_lower": frac_record(difference)})
    for K, row in zip(KS, result["concentration_illustrations"]):
        sigma2 = 2 * (1 - math.sqrt(1 - (3 / 4) ** 2))
        h = math.sqrt(2 * sigma2 * math.log(3 * 8192 * K / .05))
        L = math.log(12 * 11 / .05)
        tail = K * (h + sigma2 / h) * math.exp(-h * h / (2 * sigma2))
        bound = h * (math.sqrt(2 * L / 1024) + math.sqrt(2 * L / 8192)) + tail
        audit.close(row["error_bound_nats"], bound, "concentration illustration", "concentration")
        audit.close(row["tail_bias_bound"], tail, "concentration tail", "concentration")
    ledger = result["ledger"]
    fixed = {"independent_training_streams": 16, "role_RNG_streams": 32,
             "raw_RB_training_grid_optimizations": 576, "table_MLE_training_argmins": 48,
             "unique_training_model_outputs": 624, "raw_RB_cell_assignments": 1152,
             "count_state_analysis_argmins": 1347, "new_label_vectors": 16384,
             "new_label_bits": 4194304, "global_prior_banks": 131072, "prior_vectors": 524288,
             "Gaussian_scalars": 134217728, "new_reference_integrals": 0, "new_neural_models": 0,
             "continuous_optimizations": 0}
    audit.exact(ledger == fixed, "full resource ledger")
    audit.exact(result["families_not_combined_as_overall_95_percent"] is True, "confidence families separated")
    return {"primary": {"successes": sum(events0 for events0 in test["events"]), "trials": 16,
                        "pvalue": frac_record(pvalue), "passed": pvalue <= Fraction(1, 20),
                        "integer_square_gaps": gaps}, "endpoints": endpoints,
            "table_highN_true_risk_upper": frac_record(table[1024]["upper"]), "paired_cells": 576}


def write_report(summary):
    maximum = summary["max_errors"]
    stats = summary["statistics"]
    text = f"""# 真实共享 bank 学习的独立结果审计

独立审计通过。审计源码未导入生产计算函数；先封存 PLAN.json、START.json 与输入/源码 hash，再执行。审计源码 SHA256：`{digest(__file__)}`。生产源 SHA256：`{SOURCE_SHA}`。

完整核对16个训练流的32份本地数组与公开冻结；所有输入作逐值 RNG 重放，重新生成134,217,728个 Gaussian 标量和4,194,304个标签 bit。它们是原输入的重放，0新增独立科学样本、0新 Gaussian 求积、0新科学拟合。

全部保存 bank 数组重新聚合为576条 raw/RB曲线，另以80位 Decimal 对48个 table训练选点及全部1347个确定same-count状态作独立核对；共624个训练选择索引完全一致。期望风险上界、主二项、两项CP及风险推论全部用独立 Fraction 算术重算。

原始坐标评分只重建事前固定的每流16个 bank：`{BANKS}`。对这些256个 bank，使用全部1024条实际标签、全部11rho、K2/K4及三个N前缀，逐坐标求和 log-sigmoid，不使用生产 nuisance GEMM/logcosh。核对16,896个raw和16,896个RB bank对比值；其余130,816个bank只核保存数组算术与RNG输入，不宣称全部原始评分重放。基准分数另核。

最大原始raw评分差 `{maximum.get('raw_original_scores', 0):.17g}`，RB评分差 `{maximum.get('RB_original_scores', 0):.17g}`，保存bank聚合曲线差 `{maximum.get('saved_bank_curves', 0):.17g}`；事前绝对容差均为1e-10。选择、RNG、整数/Fraction采用精确相等。没有改门或重试。

主配对充分事件为{stats['primary']['successes']}/16，精确单侧p值 `{stats['primary']['pvalue']['numerator']}/{stats['primary']['pvalue']['denominator']}`，原门 `{stats['primary']['passed']}`。全部16个整数平方差保存，未删除非改善流。该门检验改善事件概率超过1/2，不检验平均风险差大于.001。

两个 endpoint CP家族的独立审计结果如下；它与主二项检验家族分开：

| K | endpoint成功 | CP保守下端 | raw−table程序期望风险下界 |
|---|---:|---:|---:|
"""
    for row in stats["endpoints"]:
        text += f"| {row['K']} | {row['successes']}/16 | {row['CP_lower']['float_display']:.12g} | {row['mean_risk_gap_lower']['float_display']:.12g} |\n"
    text += "\nCP的40位dyadic下点/相邻上点均通过精确有理尾概率核验；风险下界是同一CP覆盖事件的确定推论，不新增置信家族。参考风险表保持数值资格而非严格积分证书；解析风险证书不依赖该表的积分误差，但依赖理想iid训练stream合同。\n\n`check`核公开冻结、审计收据与源码；`check --workspace`另核32份本地数组hash。这两个只读入口不会每次重放上述全部算术或RNG。完整执行覆盖在PLAN/START/SUMMARY与16份逐流收据中保存。\n"
    with DOC.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write():
    if (OUT / "START.json").exists() or (OUT / "RECEIPT.json").exists():
        raise FileExistsError("Audit already started; preserve previous attempt, no automatic retry.")
    if DOC.exists():
        raise FileExistsError("Audit report already exists; do not replace it.")
    if read(SCI / "STATE.json")["state"] != "complete":
        raise RuntimeError("Production must be complete before audit START.")
    results_freeze = read(SCI / "RESULTS_FREEZE.json")
    save(OUT / "PLAN.json", PLAN)
    save(OUT / "START.json", {"started": now(), "audit_source_sha256": digest(__file__),
                               "plan_sha256": digest(OUT / "PLAN.json"),
                               "producer_source_sha256": SOURCE_SHA,
                               "source_freeze_sha256": digest(SCI / "FREEZE.json"),
                               "results_freeze_sha256": digest(SCI / "RESULTS_FREEZE.json"),
                               "public_inputs": results_freeze["files"],
                               "local_arrays": results_freeze["optional_local_arrays"]})
    save(OUT / "STATE.json", {"state": "running", "started": now()})
    began, audit = time.perf_counter(), Audit()
    try:
        public_inputs(audit, results_freeze)
        for name, expected in results_freeze["optional_local_arrays"].items():
            audit.exact(digest(SCI / name) == expected, "science local hash " + name)
        cfg = read(SCI / "CONFIG.json")
        audit.exact(cfg["Ns"] == NS and cfg["Bs"] == BS and cfg["Ks"] == KS and cfg["rho_grid"] == RHOS,
                    "fixed candidate and budget contract")
        audit.exact(cfg["streams"] == 16 and cfg["seed"] == 2037100101 and cfg["n_values"] == [0, 254], "fixed experiment contract")
        reference = read(SCI / "REFERENCE.json")
        table = independent_table(audit, reference)
        selections = [audit_stream(audit, s, table) for s in range(16)]
        stats = all_statistics(audit, selections, table, reference)
        summary = {"passed": True, "checks": audit.checks, "max_errors": audit.maxima, "statistics": stats,
                   "coverage": PLAN, "wall_seconds": time.perf_counter() - began,
                   "failures": 0, "automatic_retries": 0}
        save(OUT / "SUMMARY.json", summary)
        write_report(summary)
        save(OUT / "STATE.json", {"state": "complete", "finished": now()}, replace=True)
        files = {p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob("*.json")) if p.name != "RECEIPT.json"}
        save(OUT / "RECEIPT.json", {"passed": True, "finished": now(), "audit_source_sha256": digest(__file__),
                                   "producer_source_sha256": SOURCE_SHA,
                                   "source_freeze_sha256": digest(SCI / "FREEZE.json"),
                                   "results_freeze_sha256": digest(SCI / "RESULTS_FREEZE.json"),
                                   "files": files, "report_sha256": digest(DOC), "checks": audit.checks,
                                   "scope": PLAN})
        print(json.dumps({"passed": True, "checks": audit.checks, "max_errors": audit.maxima,
                          "wall_seconds": summary["wall_seconds"]}), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc), "time": now(),
                                   "completed_checks": audit.checks, "max_errors": audit.maxima,
                                   "automatic_retry": False})
        save(OUT / "STATE.json", {"state": "failed", "finished": now()}, replace=True)
        raise


def check(workspace=False):
    receipt, start = read(OUT / "RECEIPT.json"), read(OUT / "START.json")
    audit = Audit()
    audit.exact(receipt["passed"] is True and read(OUT / "STATE.json")["state"] == "complete", "audit complete")
    audit.exact(digest(__file__) == receipt["audit_source_sha256"] == start["audit_source_sha256"], "audit source")
    audit.exact(digest(SCI / "FREEZE.json") == receipt["source_freeze_sha256"] == start["source_freeze_sha256"], "science source freeze")
    audit.exact(digest(SCI / "RESULTS_FREEZE.json") == receipt["results_freeze_sha256"] == start["results_freeze_sha256"], "science results freeze")
    for name, expected in receipt["files"].items():
        audit.exact(digest(OUT / name) == expected, "audit file " + name)
    audit.exact(digest(DOC) == receipt["report_sha256"], "audit report")
    audit.exact(read(OUT / "PLAN.json") == PLAN and digest(OUT / "PLAN.json") == start["plan_sha256"], "audit fixed plan")
    public_inputs(audit, read(SCI / "RESULTS_FREEZE.json"))
    if workspace:
        for name, expected in start["local_arrays"].items():
            audit.exact(digest(SCI / name) == expected, "local array " + name)
    print(json.dumps({"passed": True, "workspace": workspace, "checks": audit.checks,
                      "scope": "public frozen hashes and completed receipt; workspace adds array hashes, no arithmetic/RNG replay"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["write", "check"])
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    write() if args.mode == "write" else check(args.workspace)
