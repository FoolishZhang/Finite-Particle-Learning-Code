"""Independent TRAIN RNG, partial raw scoring, full marginal and output audit.

No production imports; no new independent scientific data. Freeze after the
production RESULTS exists, then run once. Preserve every numerical failure.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import expit, logsumexp, xlogy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_calibration/TRAIN"
MODEL = ROOT / "artifacts/finite_calibration/CAL/MODEL.json"
PROTOCOL = OUT / "AUDIT_PROTOCOL.json"
RESULT = OUT / "AUDIT.json"
SOURCE = Path(__file__).resolve()
TOL = 1e-10
METHODS = ("raw", "marg", "quad", "asymcorr", "finitecal", "exact", "zero")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def freeze():
    require((OUT / "RESULTS.json").is_file(), "Wait for production RESULTS before freezing target hashes")
    require(not PROTOCOL.exists() and not RESULT.exists(), "Audit already frozen/run; no overwrite")
    production, training = read(OUT / "RESULTS.json"), read(OUT / "FREEZE.json")
    inputs = [OUT / "RESULTS.json", OUT / "FREEZE.json", MODEL, ROOT / training["plan_path"], ROOT / "script/fincal_train.py"]
    report = (ROOT / training["plan_path"]).parent / "TRAIN_AUDIT.md"
    require(not report.exists(), "Audit report already exists")
    write_new(PROTOCOL, {
        "schema": "fincal-independent-training-audit-v1", "source_sha256": sha(SOURCE),
        "input_hashes": {rel(path): sha(path) for path in inputs},
        "npz_hashes": {row["file"]: row["file_sha256"] for row in production["records"]},
        "report": rel(report), "independent_implementer": "/root/neighbor_check",
        "production_author_is_different_agent": True, "blind": False,
        "rng_scope": "All 16 complete DATA and prior streams, including nuisance Uniforms and complete Gaussian prior",
        "direct_raw_scope": {"observations": 64, "banks": 8, "teachers": 3, "rho_indices": list(range(0, 801, 40))},
        "cache_aggregate_scope": {"observations": 1024, "banks": 8, "teachers": 3, "rho_indices": list(range(0, 801, 40))},
        "cache_identity": "all-negative nuisance log likelihood plus positive-sign indicator matrix times z; independent implementation",
        "direct_raw_formula": "Sum all coordinate log-sigmoid factors, then logsumexp; compare actual per-observation contrasts to independent cache",
        "marg_scope": "Every N prefix, all 128 banks, all 801 rho, all 3 teachers; direct active sigmoid products and actual empirical mode frequencies",
        "output_scope": "All 672 methods from saved full raw curves and independently reconstructed marginal curves, model spline integrals, independent PCHIP and bisection MLE",
        "bound_scope": "All 480 real-arithmetic mesh bounds checked using Fraction, outward radicals and exact polynomial derivative extrema",
        "numerical_absolute_tolerance": TOL, "MLE_parameter_absolute_tolerance": TOL,
        "statistics_main_family": "Not recomputed here; separate summary/statistics audit",
        "new_independent_scientific_samples": 0, "replayed_input_is_new_data": False,
        "preserve_failures": True, "automatic_retry": False,
        "not_claimed": "Full raw training score replay, strict Gaussian integration/floating evaluation certification, or unconditional calibration-averaged risk coverage",
    })
    print(json.dumps({"audit_frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


class Checks:
    def __init__(self):
        self.numeric = []
        self.exact = []

    def compare(self, name, actual, expected, tolerance=TOL):
        left, right = np.asarray(actual), np.asarray(expected)
        require(left.shape == right.shape, "Comparison shape mismatch: " + name)
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        differences = np.abs(left-right)
        maximum = float(differences.max()) if differences.size and finite else 0.0 if not differences.size else None
        failures = int(np.count_nonzero(differences > tolerance)) if finite else int(differences.size)
        self.numeric.append({"name": name, "count": int(left.size), "absolute_tolerance": tolerance,
                             "maximum_absolute_difference": maximum, "failures": failures, "all_finite": finite})

    def flag(self, name, passed, detail=None):
        row = {"name": name, "passed": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        self.exact.append(row)


def replay(config, stream):
    nmax, n = max(config["sample_sizes"]), config["n"]
    seed_data, seed_prior = config["data_seed_base"]+stream, config["prior_seed_base"]+stream
    data = np.random.Generator(np.random.PCG64(seed_data))
    uniforms_noise = data.random((nmax, n))
    noise = np.where(uniforms_noise < .5, -1, 1).astype(np.int8)
    gaussian_active = data.standard_normal((nmax, 2))
    uniforms_active = data.random((nmax, 2))
    first = np.where(uniforms_active[:, 0] < expit(config["a"]*gaussian_active[:, 0]), 1, -1).astype(np.int8)
    second = []
    for theta in config["teachers"]:
        latent = theta*gaussian_active[:, 0]+math.sqrt(1-theta*theta)*gaussian_active[:, 1]
        second.append(np.where(uniforms_active[:, 1] < expit(config["a"]*latent), 1, -1))
    second = np.asarray(second, dtype=np.int8)
    prior_rng = np.random.Generator(np.random.PCG64(seed_prior))
    prior = prior_rng.standard_normal((config["B"], config["K"], n+2))
    hashes = {"data_seed": seed_data, "prior_seed": seed_prior,
              "noise_uniform_sha256": array_sha(uniforms_noise), "active_gaussian_sha256": array_sha(gaussian_active),
              "active_uniform_sha256": array_sha(uniforms_active), "prior_sha256": array_sha(prior),
              "y1_sha256": array_sha(first), "ynoise_sha256": array_sha(noise), "y2_sha256": array_sha(second)}
    return prior, first, noise, second, hashes


def cached_and_direct_raw(prior, first, noise, second, rho, saved_raw, saved_offset, checks, stream):
    """Independent cache validated on 64 rows; production compared on 1024."""
    indices = np.arange(0, 801, 40)
    candidates = rho[indices]
    zero = int(np.flatnonzero(candidates == 0)[0])
    positive_noise = np.asarray(noise[:1024] > 0, dtype=np.float64)
    compare_values = np.empty((8, 3, len(candidates)))
    direct_scalar_count = 0
    for bank in range(8):
        z = prior[bank]
        # log sigma(y z) = log sigma(-z) + 1_{y=+1} z.
        cache_log = positive_noise @ z[:, 2:].T
        cache_log -= np.logaddexp(0.0, z[:, 2:]).sum(axis=1)
        cache_log -= np.logaddexp(0.0, -first[:1024, None]*z[None, :, 0])
        row_max = cache_log.max(axis=1)
        weights = np.exp(cache_log-row_max[:, None])
        sums = weights.sum(axis=1)
        weights /= sums[:, None]
        offset = math.log(len(z))-row_max-np.log(sums)
        checks.compare(f"s{stream}/raw_offset_bank{bank}_N1024", offset, saved_offset[bank, :1024])
        x2 = z[:, 0, None]*candidates[None, :]+z[:, 1, None]*np.sqrt(1-candidates*candidates)[None, :]
        probs_positive = weights @ expit(x2)
        probs_negative = weights @ expit(-x2)
        cached_loss = np.empty((3, 1024, len(candidates)))
        for ti in range(3):
            probability = np.where(second[ti, :1024, None] > 0, probs_positive, probs_negative)
            cached_loss[ti] = -np.log(probability)
            mean = np.array([math.fsum(cached_loss[ti, :, ri])/1024 for ri in range(len(candidates))])
            compare_values[bank, ti] = mean-mean[zero]
        # Genuine full-coordinate expression, without the cache identity.
        direct_noise = np.zeros((64, len(z)))
        for start in range(0, noise.shape[1], 32):
            signs = noise[:64, start:start+32]
            active_z = z[:, 2+start:2+start+signs.shape[1]]
            direct_noise -= np.logaddexp(0.0, -signs[:, None, :]*active_z[None, :, :]).sum(axis=2)
        direct_first = -np.logaddexp(0.0, -first[:64, None]*z[None, :, 0])
        direct_losses = np.empty((3, 64, len(candidates)))
        for ti in range(3):
            for ri in range(len(candidates)):
                logs = direct_noise+direct_first-np.logaddexp(0.0, -second[ti, :64, None]*x2[None, :, ri])
                direct_losses[ti, :, ri] = math.log(len(z))-logsumexp(logs, axis=1)
        direct_scalar_count += direct_losses.size
        checks.compare(f"s{stream}/raw_direct_full_vs_cache_bank{bank}",
                       direct_losses, cached_loss[:, :64]+offset[None, :64, None])
        checks.compare(f"s{stream}/raw_direct_contrast_vs_cache_bank{bank}",
                       direct_losses-direct_losses[:, :, zero:zero+1],
                       cached_loss[:, :64]-cached_loss[:, :64, zero:zero+1])
    stored = np.asarray(saved_raw[:8, :, 0, :])[:, :, indices]
    checks.compare(f"s{stream}/raw_stored_N1024_first8_21rho", compare_values, stored)
    return {"direct_observation_bank_teacher_rho_values": direct_scalar_count,
            "direct_coordinate_logistic_terms": 8*64*256*254,
            "cache_aggregate_bank_teacher_rho_values": compare_values.size,
            "cache_observation_bank_teacher_rho_values": 8*1024*3*21}


def full_marginal(prior_active, first, second, sizes, rho):
    signs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    frequencies = np.empty((3, len(sizes), 4))
    for ti in range(3):
        for ni, N in enumerate(sizes):
            frequencies[ti, ni] = [np.count_nonzero((first[:N] == s1) & (second[ti, :N] == s2))/N for s1, s2 in signs]
    b, k, _ = prior_active.shape
    result = np.empty((b, 3, len(sizes), len(rho)))
    z1, z2 = prior_active[:, :, 0], prior_active[:, :, 1]
    first_probability = {s: expit(s*z1) for s in (-1, 1)}
    for begin in range(0, len(rho), 16):
        queries = rho[begin:begin+16]
        latent = z1[:, :, None]*queries+z2[:, :, None]*np.sqrt(1-queries*queries)
        second_probability = {s: expit(s*latent) for s in (-1, 1)}
        scores = np.empty((b, 4, len(queries)))
        for si, (s1, s2) in enumerate(signs):
            probabilities = np.mean(first_probability[s1][:, :, None]*second_probability[s2], axis=1)
            scores[:, si] = -np.log(probabilities)
        result[:, :, :, begin:begin+len(queries)] = np.einsum("bsq,tns->btnq", scores, frequencies, optimize=False)
    zero = int(np.flatnonzero(rho == 0)[0])
    result -= result[:, :, :, zero:zero+1].copy()
    return result, frequencies


class IndependentPchip:
    """PCHIP from its slope identities, without the production interpolator."""
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)
        h = np.diff(self.x)
        secants = np.diff(self.y)/h
        slopes = np.zeros_like(self.y)
        for index in range(1, len(y)-1):
            left, right = secants[index-1], secants[index]
            if left*right > 0:
                w1, w2 = 2*h[index]+h[index-1], h[index]+2*h[index-1]
                slopes[index] = (w1+w2)/(w1/left+w2/right)
        for edge, h0, h1, s0, s1 in ((0, h[0], h[1], secants[0], secants[1]),
                                      (-1, h[-1], h[-2], secants[-1], secants[-2])):
            value = ((2*h0+h1)*s0-h0*s1)/(h0+h1)
            if value*s0 <= 0:
                value = 0.0
            elif s0*s1 < 0 and abs(value) > 3*abs(s0):
                value = 3*s0
            slopes[edge] = value
        self.c = np.array([(slopes[:-1]+slopes[1:]-2*secants)/(h*h),
                           (3*secants-2*slopes[:-1]-slopes[1:])/h, slopes[:-1], self.y[:-1]])

    def __call__(self, value):
        values = np.asarray(value)
        indices = np.clip(np.searchsorted(self.x, values, side="right")-1, 0, len(self.x)-2)
        t = values-self.x[indices]
        c = self.c[:, indices]
        answer = ((c[0]*t+c[1])*t+c[2])*t+c[3]
        return float(answer) if values.ndim == 0 else answer

    def inverse(self, value):
        if value <= self.y[0]:
            return float(self.x[0])
        if value >= self.y[-1]:
            return float(self.x[-1])
        lo, hi = float(self.x[0]), float(self.x[-1])
        for _ in range(60):
            middle = (lo+hi)/2
            if self(middle) < value:
                lo = middle
            else:
                hi = middle
            if hi-lo <= 5e-13:
                break
        return (lo+hi)/2


def rational_integral_values(coefficients, knots, queries):
    knots_q = [Fraction.from_float(float(x)) for x in knots]
    coefs = [[Fraction.from_float(float(v)) for v in column] for column in np.asarray(coefficients).T]
    prefix = [Fraction(0)]

    def primitive(c, t):
        return c[0]*t**4/4+c[1]*t**3/3+c[2]*t*t/2+c[3]*t

    for index, c in enumerate(coefs):
        prefix.append(prefix[-1]+primitive(c, knots_q[index+1]-knots_q[index]))

    def value(x):
        index = min(max(int(np.searchsorted(knots, x, side="right")-1), 0), len(coefs)-1)
        return prefix[index]+primitive(coefs[index], Fraction.from_float(float(x))-knots_q[index])

    origin = value(0.0)
    return np.array([float(value(float(x))-origin) for x in queries])


def rational_derivative_sup(coefficients, knots):
    best = Fraction(0)
    for index, column in enumerate(np.asarray(coefficients).T):
        c = [Fraction.from_float(float(v)) for v in column]
        width = Fraction.from_float(float(knots[index+1]))-Fraction.from_float(float(knots[index]))
        endpoints = [Fraction(0), width]
        if c[0]:
            stationary = -c[1]/(3*c[0])
            if 0 < stationary < width:
                endpoints.append(stationary)
        for t in endpoints:
            best = max(best, abs(3*c[0]*t*t+2*c[1]*t+c[2]))
    return best


def radical_enclosure(value):
    x = math.sqrt(float(value))
    while Fraction.from_float(x)**2 < value:
        x = math.nextafter(x, math.inf)
    return Fraction.from_float(x)


def analytic_base_curvature(prior_active):
    cr = radical_enclosure(Fraction(9, 7))
    qr = radical_enclosure(Fraction(4096, 343))
    total = Fraction(0)
    for bank in prior_active:
        first = [Fraction.from_float(float(abs(x))) for x in bank[:, 0]]
        second = [Fraction.from_float(float(abs(x))) for x in bank[:, 1]]
        c = max(a+cr*b for a, b in zip(first, second))
        total += qr*max(second)+Fraction(5, 4)*c*c
    return total/len(prior_active)


def moment_risk(t, r):
    p, q = (1+t)/2, (1+r)/2
    return max(0.0, float(xlogy(p, p/q)+xlogy(1-p, (1-p)/(1-q))))


def reference_inputs(model, production, checks):
    ref = model["reference"]
    rho, rr = np.asarray(ref["rho801"]), np.asarray(ref["r801"])
    case = next(c for c in model["cases"] if c["n"] == 254 and c["K"] == 256)
    gp_c, gp_x = np.asarray(case["gprime_spline_c"]), np.asarray(case["gprime_spline_x"])
    r_cubic = CubicSpline(rho, rr, bc_type="not-a-knot")
    H = rational_integral_values(r_cubic.c, rho, rho)
    g = rational_integral_values(gp_c, gp_x, rho)
    H_D, g_D = rational_derivative_sup(r_cubic.c, rho), rational_derivative_sup(gp_c, gp_x)
    checks.compare("reference/H_exact_coefficient_integral_vs_MODEL", H, ref["H801"])
    checks.compare("reference/g_exact_coefficient_integral_vs_MODEL", g[::8], case["g_values"])
    checks.flag("reference/g_knots_are_training_knots", set(gp_x).issubset(set(rho)))
    checks.flag("reference/g_second_sup_outward", Fraction.from_float(case["g_second_sup"]) >= g_D)
    checks.flag("reference/H_second_sup_outward", Fraction.from_float(production["reference_interface"]["H_second_sup"]) >= H_D)
    A, d, w = ref["A"], ref["d"], case["w"]
    quad, asym = A*d*w, 2*A*w
    checks.compare("penalty/quad_coefficient", quad, production["penalty_coefficients"]["quad"])
    checks.compare("penalty/asym_coefficient", asym, production["penalty_coefficients"]["asymcorr"])
    penalties = {"raw": np.zeros_like(rho), "marg": np.zeros_like(rho), "quad": quad*rho*rho,
                 "asymcorr": asym*H, "finitecal": g}
    exact_penalty_D = {"raw": Fraction(0), "marg": Fraction(0), "quad": 2*Fraction.from_float(quad),
                       "asymcorr": Fraction.from_float(asym)*H_D, "finitecal": g_D}
    for name, value in exact_penalty_D.items():
        checks.flag("penalty/"+name+"_curvature_outward", Fraction.from_float(production["penalty_curvature_D"][name]) >= value)
    pchip = IndependentPchip(rho, rr)
    checks.compare("reference/independent_PCHIP_at_all_knots", pchip(rho), rr)
    return rho, pchip, penalties, exact_penalty_D


def output_checks(saved, rebuilt_marg, record, config, production, model_data, checks):
    rho, rfun, penalties, exact_penalty_D = model_data
    rows = {(row["theta"], row["N"], row["method"]): row for row in record["outputs"]}
    checks.flag(f"s{record['stream']}/unique_42_outputs", len(rows) == len(record["outputs"]) == 42)
    first, second = saved["y1"], saved["y2"]
    base_exact = analytic_base_curvature(saved["prior_active"])
    stored_base = Fraction.from_float(record["base_curvature_D"])
    checks.flag(f"s{record['stream']}/base_curvature_outward", stored_base >= base_exact)
    width_exact = max(Fraction.from_float(float(b))-Fraction.from_float(float(a)) for a, b in zip(rho[:-1], rho[1:]))
    reconstructions, mesh_rows = [], []
    for ti, theta in enumerate(config["teachers"]):
        teacher = rfun(theta)
        for ni, N in enumerate(config["sample_sizes"]):
            empirical = math.fsum(float(a*b) for a, b in zip(first[:N], second[ti, :N]))/N
            mle = rfun.inverse(empirical)
            for method in METHODS:
                original = rows[(theta, N, method)]
                label = f"s{record['stream']}/theta{theta}/N{N}/{method}"
                checks.compare(label+"/pair", empirical, original["empirical_pair"])
                checks.compare(label+"/teacher_r", teacher, original["teacher_r"])
                if method in penalties:
                    bank_curve = rebuilt_marg[:, ti, ni] if method == "marg" else saved["raw"][:, ti, ni]
                    means = np.array([math.fsum(bank_curve[:, index])/config["B"] for index in range(len(rho))])
                    scores = means+penalties[method]
                    chosen = min(range(len(rho)), key=lambda index: float(scores[index]))
                    estimate = float(rho[chosen])
                    checks.flag(label+"/grid_index", chosen == original["grid_index"])
                    checks.compare(label+"/selected_score", scores[chosen], original["selected_score"])
                    centered = [float(v)-means[chosen] for v in bank_curve[:, chosen]]
                    se = math.sqrt(math.fsum(v*v for v in centered)/(config["B"]-1)/config["B"])
                    checks.compare(label+"/bank_score_SE", se, original["raw_mesh_score_se"])
                    D = Fraction.from_float(original["curvature_D"])
                    h = Fraction.from_float(original["mesh_step"])
                    bound = Fraction.from_float(original["continuous_objective_regret_bound_real_arithmetic"])
                    checks.flag(label+"/curvature_total_outward", D >= stored_base+Fraction.from_float(production["penalty_curvature_D"][method]))
                    checks.flag(label+"/mesh_width_outward", h >= width_exact)
                    checks.flag(label+"/mesh_regret_outward", bound >= D*h*h/8)
                    checks.flag(label+"/mesh_covers_independent_real_bound", bound >= (base_exact+exact_penalty_D[method])*width_exact**2/8)
                    mesh_rows.append({"stream": record["stream"], "theta": theta, "N": N, "method": method,
                                      "stored_bound": float(bound), "gate": 3e-5, "gate_passed": bool(bound <= Fraction.from_float(3e-5))})
                else:
                    estimate = mle if method == "exact" else 0.0
                observed = rfun(estimate)
                risk = moment_risk(teacher, observed)
                checks.compare(label+"/rho", estimate, original["rho"])
                checks.compare(label+"/observed_r", observed, original["observed_r"])
                checks.compare(label+"/risk", risk, original["reference_joint_KL"])
                reconstructions.append({"stream": record["stream"], "theta": theta, "N": N, "method": method,
                                        "rho": estimate, "observed_r": observed, "reference_joint_KL": risk})
    return reconstructions, mesh_rows


def write_report(data, protocol_data):
    destination = ROOT / protocol_data["report"]
    lines = ["# TRAIN 独立 RNG、评分与选点审计", "",
             "本审计由与TRAIN实现者不同的内部代理完成，已读取生产源码以确认合同，未导入或调用其任何函数；不是外部盲审。协议在科学审计开始前冻结。所有随机调用只重放已有种子，零新增独立科学样本。", "",
             f"状态：{data['status']}。数值比较失败 {data['numerical_failures']}，精确检查失败 {data['exact_failures']}。任何科学资格未过与软件失败均保留，不自动追加预算。", "",
             "## 实际重建范围", "",
             "每条基础流重放全部nuisance Uniform、active Gaussian与Uniform、完整128×256×256 prior，核全部来源哈希、观测字节、prior_active及文件hash。共享teacher和N前缀没有重计独立样本。", "",
             "raw首先在每流首64观测、首8bank、3teacher、21个预定rho逐坐标计算完整log-sigmoid+logsumexp，并与另写的缓存恒等式作逐观察比较；随后把缓存扩到完整N1024，核首8bank/21rho的生产保存均值。64前缀从未拿来对比1024聚合。其余raw bank/候选只使用保存曲线作算术和选点核查，不称完整raw评分重放。", "",
             "marg以直接两位sigmoid概率乘积和实际模式频数重建全部N/B/801rho。全部672方法的选择和参考风险再用math.fsum、独立PCHIP斜率公式、二分MLE及冻结g′多项式的Fraction反导重算。", "",
             "全部480个网格近优输出的二阶包络、网格宽和Dh²/8用Fraction检查向上性。这个检查只验证真实算术离散化界，不认证浮点评分zeta、校准积分误差或总体最优性。主统计家族另行审计。", "",
             "## 计数与账目", "", "```json", json.dumps(data["ledger"], ensure_ascii=False, indent=2), "```", "",
             f"480项离散资格门通过 {sum(r['gate_passed'] for r in data['mesh_gates'])}/{len(data['mesh_gates'])}；科学门与本审计的向上性检查分别记账。", "",
             "数值通过表示在既定误差阈值内重建相符，不是严格全积分证书。一次共同CAL模型用于所有流，本审计没有重新学习校准，也不据此获得校准池无条件置信覆盖。", "",
             f"协议 SHA256：`{data['protocol_sha256']}`；脚本 SHA256：`{data['source_sha256']}`。完整逐项失败、源绑定、输出重建和mesh门保存在 `artifacts/finite_calibration/TRAIN/AUDIT.json`。", ""]
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def run():
    require(not RESULT.exists(), "Audit result exists; no retry/overwrite")
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Audit source differs from frozen protocol")
    checks, stream_rows, all_outputs, mesh_gates = Checks(), [], [], []
    replayed_streams = 0
    started = time.perf_counter()
    error = None
    try:
        for name, digest in p["input_hashes"].items():
            require(sha(ROOT/name) == digest, "Frozen input changed: "+name)
        production, training, model = read(OUT / "RESULTS.json"), read(OUT / "FREEZE.json"), read(MODEL)
        config = training["config"]
        checks.flag("input/source_chain", production["source_sha256"] == training["source_sha256"] == sha(ROOT / "script/fincal_train.py"))
        checks.flag("input/freeze_chain", production["freeze_sha256"] == sha(OUT / "FREEZE.json"))
        checks.flag("input/model_chain", production["reference_sha256"] == training["reference_sha256"] == sha(MODEL))
        checks.flag("input/config", production["config"] == config)
        checks.flag("input/16_streams", [r["stream"] for r in production["records"]] == list(range(16)))
        model_data = reference_inputs(model, production, checks)
        rho = model_data[0]
        for record in production["records"]:
            index = record["stream"]
            path = ROOT / record["file"]
            checks.flag(f"s{index}/npz_hash", sha(path) == record["file_sha256"] == p["npz_hashes"][record["file"]])
            checks.flag(f"s{index}/json_record", read(OUT / f"stream_{index:02d}.json") == record)
            prior, first, noise, second, provenance = replay(config, index)
            replayed_streams += 1
            for name, digest in provenance.items():
                checks.flag(f"s{index}/provenance/{name}", digest == record["provenance"][name])
            with np.load(path, allow_pickle=False) as file:
                saved = {key: file[key] for key in file.files}
            for name, value in (("y1", first), ("ynoise", noise), ("y2", second), ("prior_active", prior[:, :, :2]),
                                ("rho", rho), ("teachers", np.asarray(config["teachers"])), ("sample_sizes", np.asarray(config["sample_sizes"]))):
                checks.flag(f"s{index}/saved_bytes/{name}", array_sha(saved[name]) == array_sha(value))
            raw_scope = cached_and_direct_raw(prior, first, noise, second, rho, saved["raw"], saved["offset"], checks, index)
            rebuilt, frequencies = full_marginal(prior[:, :, :2], first, second, config["sample_sizes"], rho)
            checks.compare(f"s{index}/all_marginal_curves", rebuilt, saved["marg"])
            outputs, gates = output_checks(saved, rebuilt, record, config, production, model_data, checks)
            all_outputs.extend(outputs)
            mesh_gates.extend(gates)
            stream_rows.append({"stream": index, "file": record["file"], "file_sha256": sha(path), "replayed_provenance": provenance,
                                "prior_active_sha256": array_sha(prior[:, :, :2]), "mode_frequencies": frequencies.tolist(),
                                "raw_rebuild_scope_counts": raw_scope, "marginal_scalar_curve_comparisons": int(rebuilt.size)})
            print(json.dumps({"audited_stream": index, "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
        for name, digest in p["input_hashes"].items():
            require(sha(ROOT/name) == digest, "Input changed during audit: "+name)
        checks.flag("coverage/672_outputs", len(all_outputs) == 672)
        checks.flag("coverage/480_mesh_outputs", len(mesh_gates) == 480)
    except Exception:
        error = traceback.format_exc()
    numeric_failures = sum(row["failures"] for row in checks.numeric)
    exact_failures = sum(not row["passed"] for row in checks.exact)
    completed = len(stream_rows)
    data = {"status": "passed" if not error and not numeric_failures and not exact_failures else "failed",
            "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "input_hashes": p["input_hashes"],
            "npz_hashes": p["npz_hashes"], "elapsed_seconds": time.perf_counter()-started,
            "numeric_checks": checks.numeric, "exact_checks": checks.exact,
            "numerical_failures": numeric_failures, "exact_failures": exact_failures, "software_error": error,
            "streams": stream_rows, "reconstructed_outputs": all_outputs, "mesh_gates": mesh_gates,
            "ledger": {"completed_streams": completed, "complete_RNG_streams_replayed": replayed_streams,
                       "replayed_noise_uniforms": replayed_streams*4096*254,
                       "replayed_active_Gaussians": replayed_streams*4096*2, "replayed_active_uniforms": replayed_streams*4096*2,
                       "replayed_prior_Gaussian_scalars": replayed_streams*128*256*256,
                       "replayed_prior_banks": replayed_streams*128, "replayed_prior_vectors": replayed_streams*128*256,
                       "raw_direct_observation_bank_teacher_rho_values": completed*64*8*3*21,
                       "raw_cache_aggregate_bank_teacher_rho_values": completed*8*3*21,
                       "raw_cache_N1024_observation_bank_teacher_rho_values": completed*1024*8*3*21,
                       "marginal_scalar_curve_values": completed*128*3*2*801,
                       "method_outputs_rebuilt": len(all_outputs), "mesh_bounds_checked": len(mesh_gates),
                       "new_independent_scientific_samples": 0, "new_data_fits": 0, "new_Gaussian_integrals": 0},
            "full_raw_scoring_replayed": False, "marginal_all_saved_curves_rebuilt": completed == 16,
            "main_statistical_family_checked_here": False,
            "qualification_not_strict_integral_or_floating_evaluation_certificate": True}
    write_new(RESULT, data)
    write_report(data, p)
    print(json.dumps({"status": data["status"], "completed_streams": completed,
                      "numeric_groups": len(checks.numeric), "numeric_scalar_comparisons": sum(r["count"] for r in checks.numeric),
                      "exact_checks": len(checks.exact), "numerical_failures": numeric_failures, "exact_failures": exact_failures,
                      "software_error": bool(error), "output_sha256": sha(RESULT)}, ensure_ascii=False))
    if data["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
