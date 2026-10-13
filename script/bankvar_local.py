"""Independent local finite-n/K quadratic calibration, fixed-budget protocol.

No observed labels, old fitted g, or teacher-dependent input is read.  The only
scientific input is the known emission-model r801 table.  A source-bound PLAN
freeze is mandatory before the first random number is generated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT / "artifacts/bank_precision/LOCAL"
REFERENCE = ROOT / "artifacts/finite_calibration/CAL/REFERENCE.json"
PROTOCOL = OUT / "PROTOCOL.json"
A, R, N_NOISE, K = 1.0, 0.75, 254, 256
H = np.array([0.03, 0.06, 0.12], dtype=float)
RHO = np.array([-0.12, -0.06, -0.03, 0.03, 0.06, 0.12], dtype=float)
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int64)
PRODUCTS = SIGNS.prod(axis=1)
STREAMS, BANKS, BLOCK, SEED_BASE = 8, 1024, 64, 202610021700


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def reference_values():
    """Read only the known r801 emission model, not an old fitted penalty."""
    ref = read(REFERENCE)
    require(ref["a"] == A and ref["R"] == R, "Reference model mismatch")
    require(ref["reference_qualification_pass"] and ref["r_qualification_pass"], "Unqualified old reference")
    grid = np.asarray(ref["rho801"], dtype=float)
    table = next(row for row in ref["orders"] if row["order"] == ref["main_order"])
    indices = [int(np.argmin(np.abs(grid - value))) for value in RHO]
    require(np.max(np.abs(grid[indices] - RHO)) < 1e-14, "Local rho is not an existing reference node")
    result = np.asarray(table["r801"], dtype=float)[indices]
    require(np.max(np.abs(result + result[::-1])) < 1e-14, "Reference symmetry mismatch")
    return result, indices, ref["main_order"]


def price_block(z):
    """Bank × six rho × four-mode Price derivative, canonical nuisance signs."""
    require(z.ndim == 3 and z.shape[1:] == (K, N_NOISE + 2), "Bad Gaussian block shape")
    first, second = z[:, :, 0], z[:, :, 1]
    noise_loglik = -np.logaddexp(0.0, -z[:, :, 2:]).sum(axis=2)
    first_logs = {s: -np.logaddexp(0.0, -s * first) for s in (-1, 1)}
    first_comps = {s: expit(-s * first) for s in (-1, 1)}
    out = np.empty((len(z), len(RHO), len(SIGNS)), dtype=float)
    for j, rho in enumerate(RHO):
        x2 = rho * first + math.sqrt(1.0 - rho * rho) * second
        for m, (s1_np, s2_np) in enumerate(SIGNS):
            s1, s2 = int(s1_np), int(s2_np)
            logits = noise_loglik + first_logs[s1] - np.logaddexp(0.0, -s2 * x2)
            logits -= np.max(logits, axis=1, keepdims=True)
            np.exp(logits, out=logits)
            logits /= logits.sum(axis=1, keepdims=True)
            out[:, j, m] = -s1 * s2 * np.sum(
                logits * (1.0 - logits) * first_comps[s1] * expit(-s2 * x2), axis=1)
    return out


def coefficients(derivatives, rr):
    gp = -np.sum(derivatives * (1.0 + rr[:, None] * PRODUCTS), axis=-1) / 4.0
    pairs = [(int(np.flatnonzero(RHO == -h)[0]), int(np.flatnonzero(RHO == h)[0])) for h in H]
    cs = np.stack([(gp[..., pos] - gp[..., neg]) / (4.0 * h)
                   for h, (neg, pos) in zip(H, pairs)], axis=-1)
    return gp, cs


def scalar_price(z):
    """Separate scalar arithmetic for deterministic checking only; no RNG."""
    out = np.empty((len(z), len(RHO), len(SIGNS)))
    log_sigmoid = lambda x: -float(np.logaddexp(0.0, -x))
    for b in range(len(z)):
        noise = [math.fsum(log_sigmoid(float(v)) for v in z[b, k, 2:]) for k in range(K)]
        for j, rho in enumerate(RHO):
            xs = [(float(z[b, k, 0]), rho * float(z[b, k, 0]) + math.sqrt(1-rho*rho) * float(z[b, k, 1])) for k in range(K)]
            for m, (s1, s2) in enumerate(SIGNS):
                scores = [noise[k] + log_sigmoid(s1*x1) + log_sigmoid(s2*x2) for k, (x1, x2) in enumerate(xs)]
                mx = max(scores)
                weights = [math.exp(v-mx) for v in scores]
                total = math.fsum(weights)
                pi = [v/total for v in weights]
                out[b, j, m] = -int(s1*s2) * math.fsum(p*(1-p)*float(expit(-s1*x1))*float(expit(-s2*x2)) for p, (x1, x2) in zip(pi, xs))
    return out


def selftest():
    rr, indices, order = reference_values()
    z = np.zeros((2, K, N_NOISE + 2))
    z[1] = 1.5 * np.sin(np.arange(K*(N_NOISE + 2), dtype=float).reshape(K, N_NOISE + 2) / 17.0)
    vector, scalar = price_block(z), scalar_price(z)
    error = float(np.max(np.abs(vector-scalar)))
    zero_expected = -PRODUCTS * (1.0-1.0/K) / 4.0
    zero_error = float(np.max(np.abs(vector[0]-zero_expected)))
    gp, cs = coefficients(vector, rr)
    expected_c = np.array([(1.0-1.0/K)*rr[int(np.flatnonzero(RHO == h)[0])]/(8.0*h) for h in H])
    coefficient_error = float(np.max(np.abs(cs[0]-expected_c)))
    require(error < 5e-13 and zero_error < 5e-14 and coefficient_error < 5e-14, "Deterministic arithmetic test failed")
    require(np.all(vector * PRODUCTS <= 1e-15), "Derivative sign check failed")
    result = {"source_sha256": sha(SOURCE), "reference_sha256": sha(REFERENCE),
              "created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "scalar_vector_max_abs_error": error, "zero_bank_max_abs_error": zero_error,
              "zero_bank_coefficient_max_abs_error": coefficient_error,
              "reference_indices": indices, "reference_order": order,
              "new_random_samples": 0, "new_model_integrals": 0,
              "scope": "Two deterministic arrays, including all-zero and sinusoidal values; not new scientific banks"}
    write_new(OUT / "SELFTEST.json", result)
    print(json.dumps(result))


def freeze(plan):
    plan = Path(plan).resolve()
    require(plan.is_relative_to(ROOT) and plan.name == "PLAN.md" and plan.is_file(), "Use root-approved repository PLAN.md")
    test = read(OUT / "SELFTEST.json")
    require(test["passed"] and test["source_sha256"] == sha(SOURCE) and test["reference_sha256"] == sha(REFERENCE), "Selftest binding mismatch")
    rr, indices, order = reference_values()
    require(not PROTOCOL.exists() and not (OUT / "EXECUTION.json").exists(), "Already frozen or started")
    data = {"schema": "bankvar-local-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(SOURCE), "inputs": {rel(p): sha(p) for p in (plan, REFERENCE, OUT/"SELFTEST.json")},
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "report": rel(plan.parent / "LOCAL_REPORT.md"),
            "a": A, "R": R, "n": N_NOISE, "K": K, "rho": RHO.tolist(), "h": H.tolist(), "h_main": 0.06,
            "reference_indices": indices, "reference_order": order, "reference_r": rr.tolist(),
            "active_signs": SIGNS.tolist(), "streams": STREAMS, "banks_per_stream": BANKS,
            "block": BLOCK, "seed_base": SEED_BASE, "max_prior_shape_per_block": [BLOCK, K, N_NOISE+2],
            "rng": "numpy Generator(PCG64(seed_base+stream)); 16 sequential standard_normal((64,256,256)) blocks per stream",
            "G_bank": "-sum_y (1+r(rho)*y1*y2) D_y(rho)/4; D_y=-y1*y2 sum pi_k(1-pi_k) sigmoid(-y1 X1) sigmoid(-y2 X2)",
            "coefficient": "c_h=(G(h)-G(-h))/(4h); main h=.06 always; no clipping or h selection",
            "quadratic_penalty": "coefficient*rho^2; derivative 2*coefficient*rho",
            "teacher_specific_tuning": False, "reads_old_fitted_g": False,
            "qualification": {"main_across_8_streams_SE_max": 2e-4, "pooled_c06_minus_c03_abs_max": 5e-5,
                              "h012": "descriptive only, never replace h_main", "failure_policy": "retain all results; no budget extension or main coefficient replacement"},
            "subpools": "Four adjacent pairs of streams, each 2048 banks; descriptive calibration stability, not independent complete training confirmations",
            "prior_reuse": "All six rho and four active signs reuse each bank; no sign-averaged doubling of independent samples",
            "calibration_scope": "canonical nuisance population calibration, not complete-observation finite-data training",
            "accounting": {"new_observations": 0, "new_data_fits": 0, "new_integrals": 0, "banks": STREAMS*BANKS,
                           "prior_vectors": STREAMS*BANKS*K, "Gaussian_scalars": STREAMS*BANKS*K*(N_NOISE+2),
                           "bank_rho_mode_evaluations": STREAMS*BANKS*len(RHO)*len(SIGNS),
                           "old_full_CAL_training_banks_for_comparison": 32768, "bank_reduction_ratio": 4.0},
            "compute_scope": "Four times fewer prior banks than old full-CAL training pool; report wall clock and mode terms separately, not a measured speedup",
            "automatic_budget_extension": False}
    write_new(PROTOCOL, data)
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


def protocol():
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Source changed after freeze")
    for name, digest in p["inputs"].items():
        require(sha(ROOT/name) == digest, "Frozen input changed: " + name)
    return p


def run():
    p = protocol()
    require(not (OUT / "EXECUTION.json").exists(), "Already started; no automatic retry")
    rr, _, _ = reference_values()
    write_new(OUT / "EXECUTION.json", {"created_before_first_Gaussian_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "reference_sha256": sha(REFERENCE),
              "numpy_version": np.__version__, "scipy_version": scipy.__version__, "automatic_retry": False})
    records, all_cs = [], []
    started = time.perf_counter()
    for stream in range(STREAMS):
        rng = np.random.Generator(np.random.PCG64(SEED_BASE+stream))
        modes = np.empty((BANKS, len(RHO), len(SIGNS)))
        prior_hash = hashlib.sha256()
        first_hash = None
        rng_seconds, price_seconds = 0.0, 0.0
        stream_started = time.perf_counter()
        for offset in range(0, BANKS, BLOCK):
            t0 = time.perf_counter()
            z = rng.standard_normal((BLOCK, K, N_NOISE+2))
            rng_seconds += time.perf_counter()-t0
            raw = z.tobytes()
            prior_hash.update(raw)
            if offset == 0:
                first_hash = hashlib.sha256(raw).hexdigest()
            t0 = time.perf_counter()
            modes[offset:offset+BLOCK] = price_block(z)
            price_seconds += time.perf_counter()-t0
        require(np.isfinite(modes).all(), "Nonfinite Price output")
        gp, cs = coefficients(modes, rr)
        dest = OUT / f"stream_{stream:02d}.npz"
        require(not dest.exists(), "Refuse array overwrite")
        np.savez_compressed(dest, D_modes=modes, G_bank=gp, coefficients=cs, rho=RHO, h=H, signs=SIGNS, r=rr)
        record = {"stream": stream, "seed": SEED_BASE+stream, "banks": BANKS, "file": rel(dest), "sha256": sha(dest),
                  "prior_sha256": prior_hash.hexdigest(), "first_block_prior_sha256": first_hash,
                  "mean_G": gp.mean(axis=0).tolist(), "mean_c_h": cs.mean(axis=0).tolist(),
                  "elapsed_seconds": time.perf_counter()-stream_started, "Gaussian_generation_seconds": rng_seconds,
                  "price_evaluation_seconds": price_seconds}
        records.append(record)
        all_cs.append(cs)
        write_new(OUT / f"stream_{stream:02d}.json", record)
        print(json.dumps({"stream_completed": stream, "mean_c_h": record["mean_c_h"], "seconds": record["elapsed_seconds"]}), flush=True)
    stream_cs = np.array([row["mean_c_h"] for row in records])
    pooled = stream_cs.mean(axis=0)
    ses = stream_cs.std(axis=0, ddof=1)/math.sqrt(STREAMS)
    paired_resolution = stream_cs[:, 1]-stream_cs[:, 0]
    subpools = [{"subpool": i, "streams": [2*i, 2*i+1], "banks": 2*BANKS,
                 "c_h": stream_cs[2*i:2*i+2].mean(axis=0).tolist()} for i in range(STREAMS//2)]
    gates = {"main_stream_SE": {"value": float(ses[1]), "threshold": 2e-4, "passed": bool(ses[1] <= 2e-4)},
             "resolution_c06_c03": {"value": float(abs(pooled[1]-pooled[0])), "threshold": 5e-5,
                                     "passed": bool(abs(pooled[1]-pooled[0]) <= 5e-5)}}
    result = {"schema": "bankvar-local-model-v1", "a": A, "R": R, "n": N_NOISE, "K": K,
              "h_main": 0.06, "coefficient": float(pooled[1]), "coefficient_SE": float(ses[1]),
              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "reference_sha256": sha(REFERENCE),
              "reference_scope": "Known r801 table only; old fitted g and teacher observations never read",
              "penalty": "coefficient*rho^2", "h_values": H.tolist(), "pooled_c_h": pooled.tolist(),
              "stream_SE_c_h": ses.tolist(), "paired_c06_minus_c03_SE": float(paired_resolution.std(ddof=1)/math.sqrt(STREAMS)),
              "streams": records, "subpools_2048": subpools, "qualification": gates,
              "qualification_all_passed": all(row["passed"] for row in gates.values()),
              "main_coefficient_not_clipped_or_selected": True, "teacher_specific_tuning": False,
              "accounting": p["accounting"], "elapsed_seconds": time.perf_counter()-started,
              "compute_scope": p["compute_scope"], "scientific_scope": p["calibration_scope"]}
    write_new(OUT / "MODEL.json", result)
    print(json.dumps({"completed": True, "coefficient": result["coefficient"], "coefficient_SE": result["coefficient_SE"],
                      "qualification": gates, "model_sha256": sha(OUT/"MODEL.json"), "elapsed_seconds": result["elapsed_seconds"]}))


def report():
    p, m = protocol(), read(OUT/"MODEL.json")
    require(m["source_sha256"] == sha(SOURCE) and m["protocol_sha256"] == sha(PROTOCOL), "Model binding changed")
    dest = ROOT / p["report"]
    require(not dest.exists(), "Refuse report overwrite")
    lines = ["# 独立局部二次校准", "", "固定 a=1、R=.75、n=254、K=256，仅用已知发射模型 r801 和独立 prior 模拟；未读取旧拟合 g 或任何 teacher 观测。", "",
             "8 条独立流各 1024 bank，共 8192 bank。每 bank 的六个 rho 与四相关模式复用，canonical nuisance 仅用于总体校准。部署固定 h=.06 的 c_h=[G(h)−G(−h)]/(4h)，罚项为 c_h rho²；没有按 teacher、资格结果或风险选择 h，也没有截断系数。", "",
             "|h|pool c_h|8 流标准误|", "|---|---:|---:|"]
    for h, c, se in zip(m["h_values"], m["pooled_c_h"], m["stream_SE_c_h"]):
        lines.append(f"|{h:.2f}|{c:.12g}|{se:.8g}|")
    lines += ["", f"主系数 **{m['coefficient']:.12g}**。SE 为独立流均值的样本标准差除以 sqrt(8)，不是严格置信覆盖。", "",
              "|原资格门|读数|阈值|通过|", "|---|---:|---:|---|"]
    for name, row in m["qualification"].items():
        lines.append(f"|{name}|{row['value']:.9g}|{row['threshold']:.9g}|{row['passed']}|")
    lines += ["", "h=.12 仅描述，资格不构成参考积分余项证书或完整范围二次近似保证。四个相邻双流子池仅检查校准稳定性，后续若复用相同训练数据，其风险结果不作为四次独立完整训练确认。", "",
              "|2048-bank 子池|c_.03|c_.06|c_.12|", "|---|---:|---:|---:|"]
    for row in m["subpools_2048"]:
        lines.append(f"|{row['subpool']}|" + "|".join(f"{v:.12g}" for v in row["c_h"]) + "|")
    lines += ["", "## 账目与范围", "", f"本次耗时 {m['elapsed_seconds']:.3f} 秒；Gaussian 生成和 Price 计算分项在逐流 JSON 保存。", "",
              "与旧完整校准的 32768 个训练 bank 相比，prior bank 数为四分之一；两者计算的 rho 和模式项不同，不能据此宣称同样倍率的实测运行提速。新计算共 196608 个 bank–rho–mode 项，0 新观察、0 数据拟合、0 模型积分。", "", "```json", json.dumps(m["accounting"], ensure_ascii=False, indent=2), "```", "",
              f"MODEL SHA256 `{sha(OUT/'MODEL.json')}`；PROTOCOL SHA256 `{sha(PROTOCOL)}`。", ""]
    dest.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"report": rel(dest), "sha256": sha(dest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("selftest", "freeze", "run", "report"))
    parser.add_argument("--plan")
    args = parser.parse_args()
    if args.phase == "selftest":
        selftest()
    elif args.phase == "freeze":
        require(args.plan is not None, "freeze requires --plan")
        freeze(args.plan)
    elif args.phase == "run":
        run()
    else:
        report()
