"""Independent cross-case TRAIN audit with complete fresh-input replay.

Reuses the earlier independent audit's pure arithmetic, never the production
scoring/selection/gradient/interpolation code. No previous audit runner or writer
is invoked. Raw reconstruction is explicitly limited to eight banks per case.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit
import bankvar_audit_train as ind

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
TRAIN = ROOT/"artifacts/calibration_precision/TRAIN"
OUT = ROOT/"artifacts/calibration_precision/TRAIN_AUDIT"
PROTOCOL = OUT/"PROTOCOL.json"
CASES = [(62, 64), (62, 256), (254, 64), (254, 256)]
TARGETS = np.array([-.75, -.6, -.3, -.03, 0., .03, .3, .6, .75])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def key(n, k):
    return f"n{n}_K{k}"


def interfaces(frozen):
    full = read(ROOT/frozen["reference_path"])
    local = read(ROOT/frozen["local_model_path"])
    reference = full["reference"]
    grid = np.array(reference["rho801"])
    rfun = PchipInterpolator(grid, reference["r801"], extrapolate=False)
    result = {}
    for n, k in CASES:
        case = next(c for c in full["cases"] if c["n"] == n and c["K"] == k)
        loc = next(c for c in local["cases"] if c["n"] == n and c["K"] == k)
        gx, gc = np.array(case["gprime_spline_x"]), np.array(case["gprime_spline_c"])
        exact_bound = ind.spline_curvature_exact(gx, gc)
        require(F.from_float(case["g_second_sup"]) >= exact_bound, "Stored spline curvature not outward")
        gvalues = ind.integral_spline(gx, gc, grid)-float(ind.integral_spline(gx, gc, 0.))
        coeff = {"asymquad": reference["A"]*reference["d"]*case["w"], "localquad": loc["coefficient"]}
        penalty = {"raw": np.zeros_like(grid), "marg": np.zeros_like(grid), "fullfinite": gvalues}
        ds = {"raw": 0., "marg": 0., "fullfinite": case["g_second_sup"]}
        for name, c in coeff.items():
            penalty[name] = c*grid**2
            ds[name] = ind.up(2*F.from_float(abs(c)))
        result[key(n, k)] = {"penalty": penalty, "penalty_D": ds, "coefficients": coeff}
    return grid, rfun, result


def replay(stream):
    rng = np.random.Generator(np.random.PCG64(202610022200+stream))
    noise_u = rng.random((4096, 254))
    yn = np.where(noise_u < .5, -1, 1).astype(np.int8)
    active_g = rng.standard_normal((4096, 2))
    active_u = rng.random((4096, 2))
    y1 = np.where(active_u[:, 0] < expit(active_g[:, 0]), 1, -1).astype(np.int8)
    y2 = np.stack([np.where(active_u[:, 1] < expit(t*active_g[:, 0]+math.sqrt(1-t*t)*active_g[:, 1]), 1, -1)
                   for t in (0., .3, .6)]).astype(np.int8)
    hashes = {"noise_uniform_sha256": ind.array_sha(noise_u), "active_gaussian_sha256": ind.array_sha(active_g),
              "active_uniform_sha256": ind.array_sha(active_u), "y1_sha256": ind.array_sha(y1),
              "ynoise_sha256": ind.array_sha(yn), "y2_sha256": ind.array_sha(y2)}
    rng = np.random.Generator(np.random.PCG64(202610022300+stream))
    digest = hashlib.sha256()
    case_hashes = {key(n, k): hashlib.sha256() for n, k in CASES}
    active = np.empty((512, 256, 2))
    prefix = None
    for begin in range(0, 512, 32):
        z = rng.standard_normal((32, 256, 256))
        digest.update(z.tobytes())
        active[begin:begin+32] = z[..., :2]
        if begin == 0:
            prefix = z[:8].copy()
        for n, k in CASES:
            case_hashes[key(n, k)].update(np.ascontiguousarray(z[:, :k, :n+2]).tobytes())
    hashes["prior_sha256"] = digest.hexdigest()
    return y1, y2, yn, active, prefix, hashes, {name: h.hexdigest() for name, h in case_hashes.items()}


def freeze():
    frozen = read(TRAIN/"FREEZE.json")
    require(not PROTOCOL.exists(), "Audit already frozen")
    paths = [TRAIN/"FREEZE.json", SOURCE, Path(ind.__file__), ROOT/"script/curvcal_train.py",
             ROOT/"script/fincal_train.py", ROOT/"script/bankvar_train.py", ROOT/frozen["plan_path"],
             ROOT/frozen["reference_path"], ROOT/frozen["local_model_path"]]
    require(sha(ROOT/"script/curvcal_train.py") == frozen["source_sha256"], "Production source mismatch")
    require(sha(ROOT/frozen["reference_path"]) == frozen["reference_sha256"], "FULL model mismatch")
    require(sha(ROOT/frozen["local_model_path"]) == frozen["local_model_sha256"], "LOCAL model mismatch")
    data = {"schema": "curvcal-independent-train-audit-v1", "created_before_reconstruction_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(SOURCE), "inputs": {rel(path): sha(path) for path in paths},
            "scope": {"fresh_inputs": "All 16 streams: complete new observation RNG and maximum prior RNG; all four prior/noise prefix hashes",
                      "raw": "First 8 banks per case, all 4096 original complete observation rows, fixed nine rho",
                      "marg": "All 512 banks, four patterns, all saved 193 or fallback801 nodes in all 64 stream/cases",
                      "interpolation": "All saved raw/marg node curves via independent DCT-I/Clenshaw to all801 points",
                      "gradient": "Raw first8 and marg all512 banks at all three saved numerical MLE points per case",
                      "selection": "1920 grid outputs, 384 repeated MLE rows (48 unique), reference KL, penalties, norm/mesh/interpolation bounds",
                      "statistics": "Primary statistical intervals and tests left to root",
                      "reuse": "Pure calculations from previous independent bankvar_audit_train; no production evaluators, selectors or derivatives invoked"},
            "raw_rho": TARGETS.tolist(), "raw_banks": 8, "replay_block": 32, "selftest": ind.selftest(),
            "tolerances": {"marg_nodes": 2e-12, "saved_interpolation": 3e-11, "raw_centered": "2*epsilon+1e-9",
                           "raw_offsets": 2e-10, "path_gradients": 2e-10, "MLE": 2e-11,
                           "KL": 3e-14, "scores": 3e-12, "bounds": 1e-14},
            "no_new_scientific_samples": True, "automatic_retry": False}
    write_new(PROTOCOL, data)
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE)}))


def run():
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Audit source changed")
    for path, digest in p["inputs"].items():
        require(sha(ROOT/path) == digest, "Frozen audit input changed: "+path)
    require((TRAIN/"RESULTS.json").exists(), "Wait until production complete")
    require(not (OUT/"EXECUTION.json").exists(), "Audit already started; no retry")
    frozen, results = read(TRAIN/"FREEZE.json"), read(TRAIN/"RESULTS.json")
    require(results["freeze_sha256"] == sha(TRAIN/"FREEZE.json"), "Production freeze mismatch")
    write_new(OUT/"EXECUTION.json", {"start_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
              "protocol_sha256": sha(PROTOCOL), "production_results_sha256": sha(TRAIN/"RESULTS.json")})
    grid, rfun, models = interfaces(frozen)
    require(results["coefficients"] == {name: m["coefficients"] for name, m in models.items()}, "Case coefficients mismatch")
    query_index = np.array([np.argmin(abs(grid-r)) for r in TARGETS])
    require(np.max(abs(grid[query_index]-TARGETS)) < 1e-14, "Audit points not on original grid")
    h = ind.up(max(F.from_float(float(v))-F.from_float(float(u)) for u, v in zip(grid[:-1], grid[1:])))
    eps = ind.up(F(700)*F(25, 28)**192)
    started, rows = time.perf_counter(), []
    for stream in range(16):
        tick = time.perf_counter()
        flow = next(r for r in results["records"] if r["stream"] == stream)
        require(sha(ROOT/flow["data_file"]) == flow["data_file_sha256"], "Production data bytes changed")
        y1, y2, yn, active, prefix, hashes, prefix_hashes = replay(stream)
        require(all(flow["provenance"][name] == value for name, value in hashes.items()), "Full new input RNG replay mismatch")
        with np.load(ROOT/flow["data_file"]) as data:
            for name, a in (("y1", y1), ("y2", y2), ("ynoise", yn), ("prior_active", active)):
                require(np.array_equal(a, data[name]), "Saved new input mismatch: "+name)
            points_saved = data["gradient_rho"].copy()
            boundaries_saved = data["gradient_MLE_boundary"].copy()
        maxnorm, gate = ind.ellipsoid_gate(active)
        require(maxnorm == flow["active_gate"]["max_squared_radius_upper"] and gate == flow["active_gate"]["pass_all"], "Norm gate mismatch")
        points, boundary, pairs = ind.bisect_mle(y1, y2, rfun)
        require(np.max(abs(points-points_saved)) <= p["tolerances"]["MLE"] and np.array_equal(boundary, boundaries_saved), "Independent numerical MLE mismatch")
        case_audits = []
        for n, k in CASES:
            casekey = key(n, k)
            record = next(c for c in flow["cases"] if c["case"] == casekey)
            model = models[casekey]
            require(record["prior_prefix_sha256"] == prefix_hashes[casekey], "Prior prefix mismatch")
            require(record["noise_prefix_sha256"] == ind.array_sha(yn[:, :n]), "Observed nuisance prefix mismatch")
            require(sha(ROOT/record["file"]) == record["file_sha256"], "Case output bytes changed")
            with np.load(ROOT/record["file"]) as saved:
                require(np.array_equal(saved["rho"], grid), "Stored candidate grid mismatch")
                require(np.array_equal(saved["gradient_rho"], points_saved), "Case evaluation points not shared")
                nodes = saved["nodes"].copy()
                raw, marg = saved["raw"].copy(), saved["marg"].copy()
                raw_nodes, marg_nodes = saved["raw_node_scores"].copy(), saved["marg_node_scores"].copy()
                raw_interp = ind.reconstruct_interpolation(nodes, raw_nodes, grid)
                raw_interp -= raw_interp[:, :, 400:401].copy()
                raw_interp_error = float(np.max(abs(raw_interp-raw)))
                direct_marg = ind.marg_rebuild(active[:, :k], y1, y2, nodes)
                marg_node_error = float(np.max(abs(direct_marg-marg_nodes)))
                marg_interp = ind.reconstruct_interpolation(nodes, direct_marg, grid)
                marg_interp -= marg_interp[:, :, 400:401].copy()
                marg_interp_error = float(np.max(abs(marg_interp-marg)))
                require(max(raw_interp_error, marg_interp_error) <= p["tolerances"]["saved_interpolation"], "Saved interpolation mismatch")
                require(marg_node_error <= p["tolerances"]["marg_nodes"], "Direct marg reconstruction mismatch")
                direct_raw, rawgrad, offsets = ind.full_rows_raw(prefix[:, :k, :n+2], y1, yn[:, :n], y2, grid[query_index], points_saved)
                direct_raw -= direct_raw[:, :, 4:5].copy()
                raw_error = float(np.max(abs(direct_raw-np.take(raw[:8], query_index, axis=2))))
                offset_error = float(np.max(abs(offsets-saved["offset"][:8])))
                raw_gradient_error = float(np.max(abs(rawgrad-saved["raw_path_gradient"][:8])))
                marggrad = ind.marg_gradient(active[:, :k], y1, y2, points_saved)
                marg_gradient_error = float(np.max(abs(marggrad-saved["marg_path_gradient"])))
                ieps = eps if gate else 0.
                require(raw_error <= 2*ieps+1e-9 and offset_error <= p["tolerances"]["raw_offsets"], "Direct raw limited-prefix mismatch")
                require(max(raw_gradient_error, marg_gradient_error) <= p["tolerances"]["path_gradients"], "Fixed-point gradient mismatch")
            bases = {str(b): ind.curvature(active[:b, :k]) for b in (128, 512)}
            require(bases == record["base_curvature_D"], "Base curvature mismatch")
            ds, ks, ss, rs = [], [], [], []
            selections, mles = 0, 0
            for output in record["outputs"]:
                ti = [0., .3, .6].index(output["theta"])
                tr = float(rfun(output["theta"]))
                require(output["empirical_pair"] == pairs[ti], "Empirical pair mismatch")
                if output["method"] == "exact":
                    estimate = float(points[ti]); mles += 1
                    require(output["MLE_at_boundary"] == bool(boundary[ti]), "Boundary flag mismatch")
                else:
                    method, b = output["method"], output["B"]
                    curve = marg[:b, ti] if method == "marg" else raw[:b, ti]
                    scores = curve.mean(axis=0)+model["penalty"][method]
                    chosen = min(enumerate(scores), key=lambda item: item[1])[0]
                    require(chosen == output["grid_index"], "Independent grid minimum mismatch")
                    estimate = float(grid[chosen]); selections += 1
                    ss.append(abs(float(scores[chosen])-output["selected_score"]))
                    d = float(np.nextafter(bases[str(b)]+model["penalty_D"][method], np.inf))
                    mesh = ind.up(F.from_float(d)*F.from_float(h)**2/8)
                    total = ind.up(F.from_float(mesh)+2*F.from_float(ieps))
                    ds.extend([abs(d-output["curvature_D"]), abs(mesh-output["mesh_regret_bound_real_arithmetic"]),
                               abs(total-output["continuous_objective_regret_bound_real_arithmetic"])])
                    require(output["mesh_step_upper"] == h and output["interpolation_uncentered_sup_error_upper"] == ieps, "Numerical bound ingredients differ")
                rs.append(abs(estimate-output["rho"]))
                ks.append(abs(ind.observed_kl(tr, float(rfun(estimate)))-output["reference_joint_KL"]))
            require((selections, mles) == (30, 6), "Case output count mismatch")
            require(max(rs) <= p["tolerances"]["MLE"] and max(ks) <= p["tolerances"]["KL"], "Output parameter or KL mismatch")
            require(max(ss) <= p["tolerances"]["scores"] and max(ds) <= p["tolerances"]["bounds"], "Output score or bound mismatch")
            case_audits.append({"case": casekey, "passed": True, "raw_prefix_score_max_abs": raw_error,
                "raw_offset_max_abs": offset_error, "raw_saved_interpolation_max_abs": raw_interp_error,
                "marg_direct_nodes_max_abs": marg_node_error, "marg_independent_interpolation_max_abs": marg_interp_error,
                "raw_prefix_path_gradient_max_abs": raw_gradient_error, "marg_all_path_gradient_max_abs": marg_gradient_error,
                "rho_max_abs": max(rs), "reference_KL_max_abs": max(ks), "selected_score_max_abs": max(ss),
                "bound_max_abs": max(ds), "grid_selections": selections, "repeated_MLE_rows": mles,
                "marg_bank_pattern_nodes": 512*4*len(nodes)})
        row = {"stream": stream, "passed": True, "fresh_data_and_maximum_prior_full_replay_pass": True,
               "all_four_prefix_hashes_pass": True, "active_gate": gate, "max_squared_radius_upper": maxnorm,
               "cases": case_audits, "elapsed_seconds": time.perf_counter()-tick}
        rows.append(row)
        write_new(OUT/f"stream_{stream:02d}.json", row)
        print(json.dumps({"audit_stream": stream, "four_cases_passed": True, "seconds": row["elapsed_seconds"]}), flush=True)
    result = {"source_sha256": sha(SOURCE), "independent_helper_sha256": sha(ind.__file__), "protocol_sha256": sha(PROTOCOL),
              "production_results_sha256": sha(TRAIN/"RESULTS.json"), "passed": True, "records": rows, "scope": p["scope"],
              "elapsed_seconds": time.perf_counter()-started,
              "accounting": {"replayed_maximum_prior_banks": 8192, "replayed_prior_Gaussian_scalars": 536870912,
                             "replayed_new_base_observations": 65536, "replayed_data_Gaussian_scalars": 131072,
                             "replayed_data_Uniform_scalars": 16777216, "raw_bank_case_prefixes": 512,
                             "raw_bank_case_teacher_points": 13824, "raw_fixed_point_derivatives": 1536,
                             "marg_fixed_point_derivatives": 98304,
                             "marg_bank_pattern_nodes": sum(c["marg_bank_pattern_nodes"] for row in rows for c in row["cases"]),
                             "grid_selections": 1920, "repeated_MLE_rows": 384, "unique_MLE_inputs": 48,
                             "new_independent_scientific_samples": 0}}
    write_new(OUT/"RESULTS.json", result)
    report = ROOT/Path(frozen["plan_path"]).parent/"TRAIN_AUDIT.md"
    require(not report.exists(), "Refuse audit report overwrite")
    text = ["# 新四格 TRAIN 的独立审计", "", "全部16基础流、64个流/case检查通过。完整重放新数据和最大prior的全部RNG，核全部观测数组及四种prefix哈希。", "",
            "raw限定每流每case前8bank，在9个事前固定rho保留全部N4096真实完整观测并以独立signed-log表/完整粒子logsumexp重建；marg全部512bank、全部193或fallback801节点由四模式sigmoid乘积重建。所有保存raw/marg插值由独立DCT-I/Clenshaw复核。", "",
            "固定数值MLE点的路径梯度：raw前8bank，marg全部；1920网格选点、384重复MLE行（48独立输入）、参考KL、norm向外门与全部网格/插值界均核验。主统计区间和检验由root单独复核。", "",
            "审计复用上一包独立审计的纯算术实现，没有导入生产评分/梯度/选择/插值，也没有调用旧审计runner或写入旧产物。复用算术源哈希绑定在协议中。完整RNG重放不等于全底层raw评分重跑；保存node曲线的全部重建不扩张该范围。", "",
            "```json", json.dumps(result["accounting"], ensure_ascii=False, indent=2), "```", "",
            f"重算耗时 {result['elapsed_seconds']:.3f} 秒。协议SHA `{sha(PROTOCOL)}`；结果SHA `{sha(OUT/'RESULTS.json')}`。", ""]
    report.write_text("\n".join(text), encoding="utf-8", newline="\n")
    print(json.dumps({"audit_complete": True, "results_sha256": sha(OUT/"RESULTS.json"), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("check", "freeze", "run"))
    args = parser.parse_args()
    if args.phase == "check":
        print(json.dumps(ind.selftest()))
    elif args.phase == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as error:
            if (OUT/"EXECUTION.json").exists() and not (OUT/"FAILURE.json").exists():
                write_new(OUT/"FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                          "source_sha256": sha(SOURCE), "created_utc": datetime.now(timezone.utc).isoformat(), "no_automatic_retry": True})
            raise
