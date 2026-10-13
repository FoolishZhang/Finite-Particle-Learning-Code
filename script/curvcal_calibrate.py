"""Fresh independent local/full calibration pools for four fixed (n,K) cases.

Only check/bench/freeze are RNG-free. The two explicitly invoked scientific
pools require the root PLAN and a source-bound protocol. Known emission-model
reference values are reused; old fitted calibration and observed labels are not.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUT = ROOT/"artifacts/calibration_precision/CAL"
REFERENCE = ROOT/"artifacts/finite_calibration/CAL/REFERENCE.json"
PROTOCOL = OUT/"PROTOCOL.json"
CASES = [(62, 64), (62, 256), (254, 64), (254, 256)]
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int64)
PRODUCTS = SIGNS.prod(axis=1)
R, A, BLOCK = .75, 1., 64
POOL_SIZES = {"LOCAL": (8, 1024), "FULL": (16, 2048)}
LOCAL_RHO = np.array([-.12, -.06, -.03, .03, .06, .12])
H = np.array([.03, .06, .12])
SEEDS = {"LOCAL": 202610022000, "FULL": 202610022100}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def known_reference():
    data = read(REFERENCE)
    require(data["a"] == A and data["R"] == R, "Reference model mismatch")
    require(data["r_qualification_pass"] and data["reference_qualification_pass"], "Reference qualification failed")
    main = next(row for row in data["orders"] if row["order"] == data["main_order"])
    rho, rr = np.array(data["rho801"]), np.array(main["r801"])
    require(len(rho) == 801 and np.all(np.diff(rho) > 0) and np.all(np.diff(rr) > 0), "Malformed known reference")
    require(np.max(abs(rho+rho[::-1])) < 1e-14 and np.max(abs(rr+rr[::-1])) < 1e-14, "Reference symmetry mismatch")
    js = {case["K"]: case["J"] for case in data["cases"]}
    inherited = data["inherited"]
    reference = {"rho801": rho.tolist(), "r801": rr.tolist(), "rp801": main["rp801"],
                 "H801": main["H801"], **inherited, "known_reference_sha256": sha(REFERENCE),
                 "scope": "Known emission-model Gaussian reference and prior extreme-value constants; no old fitted g or observations"}
    cases = [{"n": n, "K": k, "J": js[k], "w": js[k]/math.sqrt(n*inherited["var_logsigmoid"])} for n, k in CASES]
    return rho, rr, reference, cases


def local_reference():
    # LOCAL reads only model declarations, qualification, rho801 and r801.
    data = read(REFERENCE)
    require(data["a"] == A and data["R"] == R and data["r_qualification_pass"], "Local reference mismatch")
    rr = next(row["r801"] for row in data["orders"] if row["order"] == data["main_order"])
    rho = np.array(data["rho801"])
    index = np.array([np.argmin(abs(rho-value)) for value in LOCAL_RHO])
    require(np.max(abs(rho[index]-LOCAL_RHO)) < 1e-14, "Local points not old exact nodes")
    return np.asarray(rr)[index]


def price_block(z, rho):
    """B x four cases x rho x modes, preserving maximum-bank prefix coupling."""
    require(z.ndim == 3 and z.shape[1:] == (256, 256), "Unexpected prior shape")
    z1, z2 = z[..., 0], z[..., 1]
    first_logs = {s: -np.logaddexp(0., -s*z1) for s in (-1, 1)}
    first_comps = {s: expit(-s*z1) for s in (-1, 1)}
    noise = -np.logaddexp(0., -z[..., 2:])
    noise62, noise254 = noise[..., :62].sum(axis=2), noise.sum(axis=2)
    noises = {62: noise62, 254: noise254}
    result = np.empty((len(z), len(CASES), len(rho), 4))
    for ri, value in enumerate(rho):
        x2 = value*z1+math.sqrt(1-value*value)*z2
        second_logs = {s: -np.logaddexp(0., -s*x2) for s in (-1, 1)}
        second_comps = {s: expit(-s*x2) for s in (-1, 1)}
        for ci, (n, k) in enumerate(CASES):
            for si, signs in enumerate(SIGNS):
                s1, s2 = int(signs[0]), int(signs[1])
                logits = noises[n][:, :k]+first_logs[s1][:, :k]+second_logs[s2][:, :k]
                logits -= logits.max(axis=1, keepdims=True)
                np.exp(logits, out=logits)
                logits /= logits.sum(axis=1, keepdims=True)
                result[:, ci, ri, si] = -s1*s2*np.sum(logits*(1-logits)*first_comps[s1][:, :k]*second_comps[s2][:, :k], axis=1)
    return result


def local_coefficients(modes, rr):
    gp = -np.sum(modes*(1+rr[:, None]*PRODUCTS), axis=-1)/4
    coeff = np.stack([(gp[..., 3+i]-gp[..., 2-i])/(4*h) for i, h in enumerate(H)], axis=-1)
    return gp, coeff


def full_projection(modes, rr):
    u, v = modes.mean(axis=-1), np.mean(modes*PRODUCTS, axis=-1)
    us, vs = (u-u[..., ::-1])/2, (v+v[..., ::-1])/2
    gp = -us-rr*vs
    require(np.max(abs(gp+gp[..., ::-1])) < 1e-14, "Odd gprime construction failed")
    return {"U_prime_unsym": u, "V_prime_unsym": v, "U_prime": us, "V_prime": vs,
            "b": -vs, "g_prime": gp}


def mean_se(values):
    return values.mean(axis=0), values.std(axis=0, ddof=1)/math.sqrt(len(values))


def outward(value):
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    require(Fraction.from_float(result) >= value, "Outward conversion failed")
    return result


def curvature_receipt(spline):
    """Exact supremum of derivative of stored binary cubic gprime."""
    best = Fraction(0)
    for j in range(len(spline.x)-1):
        c0, c1, c2 = (Fraction.from_float(float(v)) for v in spline.c[:3, j])
        width = Fraction.from_float(float(spline.x[j+1]))-Fraction.from_float(float(spline.x[j]))
        points = [Fraction(0), width]
        if c0:
            vertex = -c1/(3*c0)
            if 0 < vertex < width:
                points.append(vertex)
        best = max(best, *(abs(3*c0*t*t+2*c1*t+c2) for t in points))
    return {"upper_float": outward(best), "exact_numerator": str(best.numerator), "exact_denominator": str(best.denominator),
            "scope": "Exact stored-polynomial curvature only, not model integral or calibration error"}


def scalar_price(z, rho):
    result = np.empty((len(z), len(CASES), len(rho), 4))
    ls = lambda x: -float(np.logaddexp(0., -x))
    for bi, bank in enumerate(z):
        for ci, (n, k) in enumerate(CASES):
            nuisance = [math.fsum(ls(float(v)) for v in bank[j, 2:2+n]) for j in range(k)]
            for ri, value in enumerate(rho):
                x = [value*bank[j, 0]+math.sqrt(1-value*value)*bank[j, 1] for j in range(k)]
                for si, (s1, s2) in enumerate(SIGNS):
                    logs = [nuisance[j]+ls(s1*bank[j, 0])+ls(s2*x[j]) for j in range(k)]
                    maximum = max(logs)
                    weights = [math.exp(v-maximum) for v in logs]
                    total = math.fsum(weights)
                    result[bi, ci, ri, si] = -int(s1*s2)*math.fsum((w/total)*(1-w/total)*float(expit(-s1*bank[j, 0]))*float(expit(-s2*x[j])) for j, w in enumerate(weights))
    return result


def check():
    z = np.zeros((2, 256, 256))
    z[1] = 1.2*np.sin(np.arange(256*256).reshape(256, 256)/11)
    direct, fast = scalar_price(z, LOCAL_RHO), price_block(z, LOCAL_RHO)
    error = float(np.max(abs(direct-fast)))
    require(error <= 5e-13, "Four-case scalar Price mismatch")
    zero = np.array([-PRODUCTS*(1-1/k)/4 for _, k in CASES])
    zero_error = float(np.max(abs(fast[0]-zero[:, None, :])))
    require(zero_error <= 1e-13, "Zero-bank closed formula mismatch")
    rr = local_reference()
    _, coefficients = local_coefficients(fast, rr)
    expected = np.array([[(1-1/k)*rr[3+i]/(8*h) for i, h in enumerate(H)] for _, k in CASES])
    require(np.max(abs(coefficients[0]-expected)) < 5e-13, "Local coefficient factor mismatch")
    rho, r801, _, _ = known_reference()
    small = price_block(z, rho[::80])
    pp = full_projection(small, r801[::80])
    require(np.min(pp["b"]) > 0, "Price b positivity failed")
    for ci in range(4):
        spline = CubicSpline(rho[::80], pp["g_prime"].mean(axis=0)[ci])
        bound = curvature_receipt(spline)
        require(np.max(abs(spline.derivative()(rho))) <= bound["upper_float"]+1e-14, "Curvature bound check failed")
    return {"passed": True, "scalar_price_max_abs": error, "zero_bank_max_abs": zero_error,
            "local_coefficient_max_abs": float(np.max(abs(coefficients[0]-expected))),
            "four_cases_tested": CASES, "symmetry_b_positive_and_curvature": True,
            "new_random_draws": 0, "new_model_integrals": 0}


def benchmark():
    require(not (OUT/"BENCHMARK.json").exists(), "Benchmark exists")
    tests = check()
    rho, _, _, _ = known_reference()
    z = 1.1*np.sin(np.arange(BLOCK*256*256, dtype=float).reshape(BLOCK, 256, 256)/17)
    timings, checksums = {}, {}
    for pool, grid in (("LOCAL", LOCAL_RHO), ("FULL", rho[::8])):
        start = time.perf_counter()
        values = price_block(z, grid)
        timings[pool] = time.perf_counter()-start
        checksums[pool] = float(values.sum())
    data = {"source_sha256": sha(SOURCE), "reference_sha256": sha(REFERENCE), "check": tests,
            "bounded_deterministic_shape": [BLOCK, 256, 256], "seconds_per_block": timings,
            "projected_price_seconds_by_final_pool_budget": {key: value*(math.prod(POOL_SIZES[key])/BLOCK) for key, value in timings.items()},
            "final_pool_sizes": POOL_SIZES,
            "checksum": checksums, "threads_requested": 1, "new_random_draws": 0,
            "scope": "Deterministic bounded arithmetic benchmark only; excludes Gaussian RNG, hashing, array compression and model assembly; not a scientific pilot"}
    write_new(OUT/"BENCHMARK.json", data)
    print(json.dumps(data))


def freeze(plan):
    plan = Path(plan).resolve()
    require(plan.name == "PLAN.md" and plan.is_file() and plan.is_relative_to(ROOT), "Need existing root PLAN.md")
    require(not PROTOCOL.exists(), "Already frozen")
    test = check()
    benchmark_data = read(OUT/"BENCHMARK.json")
    require(benchmark_data["source_sha256"] == sha(SOURCE), "Benchmark source binding mismatch")
    rho, _, _, cases = known_reference()
    data = {"schema": "curvcal-calibration-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sha(SOURCE), "inputs": {rel(path): sha(path) for path in (plan, REFERENCE, OUT/"BENCHMARK.json")},
            "report": rel(plan.parent/"CAL_IMPLEMENTATION.md"), "a": A, "R": R, "cases": cases,
            "local_rho": LOCAL_RHO.tolist(), "full_rho": rho[::8].tolist(), "h_values": H.tolist(), "h_main": .06,
            "pool_sizes": {name: {"streams": size[0], "banks_per_stream": size[1]} for name, size in POOL_SIZES.items()},
            "block": BLOCK, "max_prior_shape_per_block": [BLOCK, 256, 256],
            "seeds": SEEDS, "rng": "numpy Generator(PCG64(seed_base+stream)); LOCAL 16 and FULL 32 standard_normal((64,256,256)) calls per stream",
            "pool_independence": "LOCAL and FULL use disjoint fixed seed families; four n/K cases share each maximum bank's coordinate and particle prefixes",
            "projection": "U'=mean_y D_y, V'=mean_y y1y2 D_y; odd-project U', even-project V'; g'=-U'-r V'; b=-V'",
            "penalty": "FULL odd gprime values -> not-a-knot CubicSpline -> exact stored-polynomial antiderivative minus value at zero; LOCAL c_.06*rho^2",
            "local_qualification": {"main_stream_SE_max": 2e-4, "c06_minus_c03_absolute_max": 5e-5},
            "full_qualification": {"gprime_SE_per_node_max": 2e-4, "b_positive_all_nodes": True},
            "qualification_scope": "Numerical/MC development qualification; not simultaneous confidence bands or strict integral remainder certificates",
            "h012_scope": "Descriptive only; no h selection or positive clipping",
            "inputs_allowed": "Known REFERENCE.json r/rp/H/A/d/var/J only; no old fitted calibration or observations. LOCAL consumes only r/rho/model declarations.",
            "per_pool_accounting": {name: {"banks": math.prod(size), "prior_vectors": math.prod(size)*256,
                                    "Gaussian_scalars": math.prod(size)*256*256,
                                    "bank_case_rho_mode_evaluations": math.prod(size)*4*(6 if name == "LOCAL" else 101)*4}
                                    for name, size in POOL_SIZES.items()},
            "new_observations": 0, "new_data_fits": 0, "new_model_integrals": 0,
            "check": test, "automatic_budget_extension": False, "failure_policy": "Preserve all failures and main choices; no retry, extra streams, or h replacement"}
    write_new(PROTOCOL, data)
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


def protocol():
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Source changed after freeze")
    for path, digest in p["inputs"].items():
        require(sha(ROOT/path) == digest, "Frozen input changed: "+path)
    return p


def build_local(records, p):
    values = np.array([row["local_coefficient_means"] for row in records])
    means, ses = mean_se(values)
    cases = []
    for ci, (n, k) in enumerate(CASES):
        difference = abs(means[ci, 1]-means[ci, 0])
        gates = {"main_stream_SE": {"value": float(ses[ci, 1]), "threshold": 2e-4, "passed": bool(ses[ci, 1] <= 2e-4)},
                 "resolution_c06_c03": {"value": float(difference), "threshold": 5e-5, "passed": bool(difference <= 5e-5)}}
        cases.append({"n": n, "K": k, "h_main": .06, "coefficient": float(means[ci, 1]), "coefficient_SE": float(ses[ci, 1]),
                      "h_values": H.tolist(), "pooled_c_h": means[ci].tolist(), "stream_SE_c_h": ses[ci].tolist(),
                      "per_stream_c_h": values[:, ci].tolist(), "qualification": gates,
                      "qualification_all_passed": all(row["passed"] for row in gates.values()),
                      "coefficient_not_clipped_or_selected": True})
    model = {"schema": "curvcal-local-v1", "a": A, "R": R, "cases": cases,
             "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "reference_sha256": sha(REFERENCE),
             "pool_results_sha256": sha(OUT/"LOCAL/STREAMS.json"), "streams": len(records),
             "penalty": "case.coefficient*rho^2", "teacher_specific_tuning": False,
             "all_qualification_passed": all(c["qualification_all_passed"] for c in cases),
             "reference_scope": "Known r801 only; independent of FULL pool and all observed data"}
    write_new(OUT/"LOCAL/MODEL.json", model)
    return model


def build_full(records, p, rho, rr, reference, prior_cases):
    # The statistical unit is one independent stream; all cases/nodes/modes share it.
    streams = np.asarray([row["mode_means"] for row in records])
    projected = full_projection(streams, rr)
    summary = {key: mean_se(value) for key, value in projected.items()}
    cases = []
    for ci, case in enumerate(prior_cases):
        gp = CubicSpline(rho, summary["g_prime"][0][ci], bc_type="not-a-knot")
        anti = gp.antiderivative()
        sup = curvature_receipt(gp)
        all_g = []
        for si in range(len(records)):
            stream_anti = CubicSpline(rho, projected["g_prime"][si, ci], bc_type="not-a-knot").antiderivative()
            all_g.append(stream_anti(rho)-stream_anti(0.))
        all_g = np.array(all_g)
        _, gse = mean_se(all_g)
        data = {**case, "rho101": rho.tolist(), "gprime_spline_x": gp.x.tolist(), "gprime_spline_c": gp.c.tolist(),
                "g_antiderivative_origin_shift": float(anti(0.)), "g_values": (anti(rho)-anti(0.)).tolist(),
                "g_second_sup": sup["upper_float"], "g_second_sup_receipt": sup,
                "per_stream_g_values": all_g.tolist(), "g_values_SE": gse.tolist(),
                "calibration_SE": {key: se[ci].tolist() for key, (_, se) in summary.items()},
                "per_stream": {key: value[:, ci].tolist() for key, value in projected.items()},
                "gprime_SE_pass_by_node": (summary["g_prime"][1][ci] <= 2e-4).tolist(),
                "b_positive_by_node": (summary["b"][0][ci] > 0).tolist(),
                "b_SE": summary["b"][1][ci].tolist(),
                "b_slope101": summary["b"][0][ci].tolist(), "b_slope_SE101": summary["b"][1][ci].tolist(),
                "stream_gprime101": projected["g_prime"][:, ci].tolist(), "stream_b101": projected["b"][:, ci].tolist(),
                "linear_aggregation_check_max": float(np.max(abs(all_g.mean(axis=0)-(anti(rho)-anti(0.))))) }
        data.update({key: mean[ci].tolist() for key, (mean, _) in summary.items()})
        data["gprime_SE_passed"] = sum(data["gprime_SE_pass_by_node"])
        data["b_positive_passed"] = sum(data["b_positive_by_node"])
        cases.append(data)
    model = {"schema": "fincal-model-v1", "a": A, "R": R, "reference": reference, "cases": cases,
             "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "reference_sha256": sha(REFERENCE),
             "training_sha256": sha(OUT/"FULL/STREAMS.json"), "training_streams": len(records),
             "gprime_SE_gate_count": 4*101, "gprime_SE_gate_passed": sum(c["gprime_SE_passed"] for c in cases),
             "b_positive_node_count": 4*101, "b_positive_nodes": sum(c["b_positive_passed"] for c in cases),
             "teacher_free": True, "shared_across_all_teachers": True,
             "penalty_evaluation_contract": "PPoly(gprime_spline_c,gprime_spline_x).antiderivative()(rho) minus its value at zero",
             "independent_of_LOCAL_pool": True,
             "scope": "New finite FULL calibration comparator and numerical curvature diagnostic, not exact model calibration reference",
             "training_qualification_is_not_uniform_confidence_coverage": True}
    write_new(OUT/"FULL/MODEL.json", model)
    return model


def run(pool):
    p = protocol()
    folder = OUT/pool
    streams_count, banks_count = POOL_SIZES[pool]
    require(not (folder/"EXECUTION.json").exists(), "Pool already started; no automatic retry")
    if pool == "LOCAL":
        rho, rr = LOCAL_RHO, local_reference()
        reference, prior_cases = None, None
    else:
        grid, rgrid, reference, prior_cases = known_reference()
        rho, rr = grid[::8], rgrid[::8]
    write_new(folder/"EXECUTION.json", {"pool": pool, "created_before_first_Gaussian_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "reference_sha256": sha(REFERENCE),
              "seed_base": SEEDS[pool], "numpy_version": np.__version__, "scipy_version": scipy.__version__, "automatic_retry": False})
    started, records = time.perf_counter(), []
    for stream in range(streams_count):
        rng = np.random.Generator(np.random.PCG64(SEEDS[pool]+stream))
        modes = np.empty((banks_count, 4, len(rho), 4))
        digest, first_digest = hashlib.sha256(), None
        generation_seconds, scoring_seconds = 0., 0.
        tick = time.perf_counter()
        for begin in range(0, banks_count, BLOCK):
            phase = time.perf_counter()
            z = rng.standard_normal((BLOCK, 256, 256))
            generation_seconds += time.perf_counter()-phase
            raw = z.tobytes()
            digest.update(raw)
            if begin == 0:
                first_digest = hashlib.sha256(raw).hexdigest()
            phase = time.perf_counter()
            modes[begin:begin+BLOCK] = price_block(z, rho)
            scoring_seconds += time.perf_counter()-phase
        require(np.isfinite(modes).all(), "Nonfinite Price evaluation")
        dest = folder/f"stream_{stream:02d}.npz"
        require(not dest.exists(), "Refuse array overwrite")
        record = {"stream": stream, "seed": SEEDS[pool]+stream, "banks": banks_count, "file": rel(dest),
                  "prior_sha256": digest.hexdigest(), "first_block_prior_sha256": first_digest,
                  "mode_means": modes.mean(axis=0).tolist()}
        if pool == "LOCAL":
            gb, cb = local_coefficients(modes, rr)
            np.savez_compressed(dest, D_modes=modes, G_bank=gb, coefficients=cb, rho=rho, r=rr, h=H, cases=np.array(CASES), signs=SIGNS)
            record["local_coefficient_means"] = cb.mean(axis=0).tolist()
        else:
            np.savez_compressed(dest, D_modes=modes, rho=rho, r=rr, cases=np.array(CASES), signs=SIGNS)
        record.update({"sha256": sha(dest), "elapsed_seconds": time.perf_counter()-tick,
                       "Gaussian_generation_seconds": generation_seconds, "Price_scoring_seconds": scoring_seconds})
        records.append(record)
        write_new(folder/f"stream_{stream:02d}.json", record)
        print(json.dumps({"pool": pool, "stream_completed": stream, "seconds": record["elapsed_seconds"]}), flush=True)
    ledger = {"banks": streams_count*banks_count, "unique_max_prior_vectors": streams_count*banks_count*256,
              "Gaussian_scalars": streams_count*banks_count*256*256, "bank_case_rho_mode_evaluations": streams_count*banks_count*4*len(rho)*4,
              "new_observations": 0, "new_data_fits": 0, "new_model_integrals": 0,
              "case_reuse": "Case prefixes and modes/rho share the same maximum prior; not independent samples"}
    write_new(folder/"STREAMS.json", {"pool": pool, "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
              "reference_sha256": sha(REFERENCE), "records": records, "ledger": ledger,
              "elapsed_seconds": time.perf_counter()-started})
    if pool == "LOCAL":
        model = build_local(records, p)
        summary = [{"n": c["n"], "K": c["K"], "coefficient": c["coefficient"], "SE": c["coefficient_SE"], "qualified": c["qualification_all_passed"]} for c in model["cases"]]
    else:
        model = build_full(records, p, rho, rr, reference, prior_cases)
        summary = {"gprime_SE": [model["gprime_SE_gate_passed"], model["gprime_SE_gate_count"]],
                   "b_positive": [model["b_positive_nodes"], model["b_positive_node_count"]]}
    print(json.dumps({"pool_completed": pool, "model_sha256": sha(folder/"MODEL.json"), "summary": summary}))


def report():
    p = protocol()
    local, full = read(OUT/"LOCAL/MODEL.json"), read(OUT/"FULL/MODEL.json")
    dest = ROOT/p["report"]
    require(not dest.exists(), "Refuse report overwrite")
    lines = ["# 跨 n/K 的独立局部与完整校准实现", "", "LOCAL 为 8 流×1024 bank，FULL 为独立 16 流×2048 bank；四种 n/K 共用各 bank 的坐标和粒子前缀。两池独立记账，不把 case、节点或符号复用当新独立样本。", "",
             "只使用旧 REFERENCE.json 的已知模型 r/rp/H/A/d/var/J，不读旧拟合 g 或任何观测标签。LOCAL 仅消费 r，并固定 h=.06；FULL 是新有限样条参照和曲率诊断，不称精确校准。", "",
             "|n|K|LOCAL c_.06|8流 SE|SE/步长资格|FULL g′ SE通过|FULL正b节点|", "|---|---:|---:|---:|---|---:|---:|"]
    for lc, fc in zip(local["cases"], full["cases"]):
        require((lc["n"], lc["K"]) == (fc["n"], fc["K"]), "Case order mismatch")
        lines.append(f"|{lc['n']}|{lc['K']}|{lc['coefficient']:.12g}|{lc['coefficient_SE']:.8g}|{lc['qualification_all_passed']}|{fc['gprime_SE_passed']}/101|{fc['b_positive_passed']}/101|")
    lines += ["", "FULL 按四模式求 U′/V′，对 U′作奇投影、V′作偶投影，再令 g′=−U′−rV′；保存 b=−V′、逐流均值和标准误。对 g′ 的 not-a-knot 三次样条作分段精确反导，不另拟合 g 值。g″ 的 sup 由保存二进制系数与结点的精确有理运算向外界定，只认证该保存函数的曲率。", "",
              "LOCAL SE 为8个独立流均值的样本标准差除以sqrt(8)，FULL按16个独立流除以sqrt(16)；资格门不是严格同时置信带。步长差不保证整个参数区间二次替代误差；h=.12仅描述，未按教师、风险或资格换步长。所有资格失败保留，不加预算。", "",
              "|pool|bank|prior向量|Gaussian标量|bank-case-rho-mode项|实际秒|", "|---|---:|---:|---:|---:|---:|"]
    for pool in ("LOCAL", "FULL"):
        result = read(OUT/pool/"STREAMS.json")
        ledger = result["ledger"]
        lines.append(f"|{pool}|{ledger['banks']}|{ledger['unique_max_prior_vectors']}|{ledger['Gaussian_scalars']}|{ledger['bank_case_rho_mode_evaluations']}|{result['elapsed_seconds']:.3f}|")
    lines += ["", "每个池的逐流 Gaussian 生成、Price 评分和总耗时均另存。LOCAL 的 prior 数为 FULL 的四分之一；FULL 的101节点不属于 LOCAL 拟合必需成本，也不能把不同节点项比例直接当实测速度比。两池合计0新观测、0数据拟合、0模型积分。", "",
              f"协议SHA256 `{sha(PROTOCOL)}`；LOCAL模型 `{sha(OUT/'LOCAL/MODEL.json')}`；FULL模型 `{sha(OUT/'FULL/MODEL.json')}`。", ""]
    dest.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"report": rel(dest), "sha256": sha(dest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "bench", "freeze", "local", "full", "report"))
    parser.add_argument("--plan")
    args = parser.parse_args()
    if args.phase == "check":
        print(json.dumps(check()))
    elif args.phase == "bench":
        benchmark()
    elif args.phase == "freeze":
        require(args.plan is not None, "freeze requires --plan")
        freeze(args.plan)
    elif args.phase == "report":
        report()
    else:
        pool = args.phase.upper()
        try:
            run(pool)
        except Exception as error:
            if (OUT/pool/"EXECUTION.json").exists() and not (OUT/pool/"FAILURE.json").exists():
                write_new(OUT/pool/"FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                          "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
                          "no_automatic_retry": True})
            raise
