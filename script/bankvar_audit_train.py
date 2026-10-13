"""Independent TRAIN audit: complete RNG, bounded full-row raw reconstruction.

No production evaluator, selector, derivative, interpolation, or helper is
imported.  The frozen scope deliberately does not recompute every raw score.
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
from scipy.fft import dct
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
TRAIN = ROOT/"artifacts/bank_precision/TRAIN"
OUT = ROOT/"artifacts/bank_precision/TRAIN_AUDIT"
PROTOCOL = OUT/"PROTOCOL.json"
TARGETS = np.array([-.75, -.6, -.3, -.03, 0., .03, .3, .6, .75])
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def up(value):
    value = F(value)
    result = float(value)
    if F.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    require(F.from_float(result) >= value, "Failed rational outward conversion")
    return result


def sqrt_up(value):
    value = F(value)
    result = math.sqrt(float(value))
    while F.from_float(result)**2 < value:
        result = math.nextafter(result, math.inf)
    return result


def ellipsoid_gate(active):
    x, y = np.abs(active[..., 0]), np.abs(active[..., 1])
    bounds = np.nextafter(np.nextafter(x*x, np.inf)+np.nextafter(y*y, np.inf), np.inf)
    return float(bounds.max()), bool(np.all(bounds <= 49.0))


def curvature(active):
    # Same proved analytic envelope, reconstructed without production helpers.
    cr, qr = sqrt_up(F(9, 7)), sqrt_up(F(4096, 343))
    x, y = np.abs(active[..., 0]), np.abs(active[..., 1])
    c = np.nextafter(x+np.nextafter(cr*y, np.inf), np.inf).max(axis=1)
    qq = np.nextafter(qr*y.max(axis=1), np.inf)
    vals = np.nextafter(qq+np.nextafter(1.25*np.nextafter(c*c, np.inf), np.inf), np.inf)
    return up(sum(map(lambda v: F.from_float(float(v)), vals), F(0))/len(vals))


def integral_spline(knots, coefficients, at):
    """Horner and interval integrals of stored cubic derivative; no PPoly."""
    knots, cc, at = np.asarray(knots), np.asarray(coefficients), np.asarray(at)
    interval = np.diff(knots)
    masses = cc[0]*interval**4/4+cc[1]*interval**3/3+cc[2]*interval**2/2+cc[3]*interval
    cumulative = np.r_[0., np.cumsum(masses)]
    idx = np.searchsorted(knots, at, side="right")-1
    idx = np.clip(idx, 0, len(knots)-2)
    dx = at-knots[idx]
    return cumulative[idx]+dx*(cc[3, idx]+dx*(cc[2, idx]/2+dx*(cc[1, idx]/3+dx*cc[0, idx]/4)))


def spline_curvature_exact(knots, coefficients):
    best = F(0)
    for j in range(len(knots)-1):
        h = F.from_float(float(knots[j+1]))-F.from_float(float(knots[j]))
        a, b, c = (F.from_float(float(coefficients[i][j]))*(3-i) for i in range(3))
        points = [F(0), h]
        if a:
            vertex = -b/(2*a)
            if 0 < vertex < h:
                points.append(vertex)
        best = max(best, *(abs((a*t+b)*t+c) for t in points))
    return best


def reference(frozen):
    model = read(ROOT/frozen["reference_path"])
    ref = model["reference"]
    case = next(c for c in model["cases"] if c["n"] == 254 and c["K"] == 256)
    grid = np.array(ref["rho801"])
    rfun = PchipInterpolator(grid, ref["r801"], extrapolate=False)
    gx, gc = np.array(case["gprime_spline_x"]), np.array(case["gprime_spline_c"])
    gvalues = integral_spline(gx, gc, grid)-float(integral_spline(gx, gc, 0.))
    exact_d = spline_curvature_exact(gx, gc)
    require(F.from_float(case["g_second_sup"]) >= exact_d, "Reference curvature not outward")
    coeff = {"asymquad": ref["A"]*ref["d"]*case["w"],
             "oldTaylor": read(ROOT/frozen["old_taylor_path"])["coefficient"],
             "localquad": read(ROOT/frozen["local_model_path"])["coefficient"]}
    penalty = {"raw": np.zeros_like(grid), "marg": np.zeros_like(grid), "fullfinite": gvalues}
    ds = {"raw": 0., "marg": 0., "fullfinite": case["g_second_sup"]}
    for name, c in coeff.items():
        penalty[name] = c*grid**2
        ds[name] = up(2*F.from_float(abs(c)))
    return grid, rfun, penalty, ds, coeff


def bisect_mle(y1, y2, rfun):
    result, boundaries, pairs = [], [], []
    for row in y2:
        pair = float(np.sum(y1.astype(np.int64)*row.astype(np.int64))/len(y1))
        if pair <= float(rfun(-.75)):
            point, boundary = -.75, True
        elif pair >= float(rfun(.75)):
            point, boundary = .75, True
        else:
            lo, hi = -.75, .75
            for _ in range(60):
                mid = (lo+hi)/2
                if float(rfun(mid)) < pair:
                    lo = mid
                else:
                    hi = mid
            point, boundary = (lo+hi)/2, False
        result.append(point)
        boundaries.append(boundary)
        pairs.append(pair)
    return np.array(result), np.array(boundaries), np.array(pairs)


def observed_kl(teacher, candidate):
    # Four-state law reduced to its two equiprobability classes.
    p, q = np.array([1+teacher, 1-teacher])/2, np.array([1+candidate, 1-candidate])/2
    return float(np.sum(p*(np.log(p)-np.log(q))))


def reconstruct_interpolation(nodes, values, grid):
    if len(nodes) == len(grid):
        require(np.array_equal(nodes, grid), "Unexpected direct-grid fallback")
        return values.copy()
    require(len(nodes) == 193, "Unexpected node count")
    # Reversed ascending node data are cosine-ordered; independent DCT-I route.
    cc = dct(values[..., ::-1], type=1, axis=-1)/192
    cc[..., 0] *= .5
    cc[..., -1] *= .5
    return np.polynomial.chebyshev.chebval(grid/.75, np.moveaxis(cc, -1, 0))


def marg_rebuild(active, y1, y2, rho):
    bcount, kcount = active.shape[:2]
    result = np.empty((bcount, len(y2), len(rho)))
    freq = np.array([[(np.logical_and(y1 == s1, row == s2)).mean() for s1, s2 in SIGNS] for row in y2])
    z1, z2 = active[..., 0], active[..., 1]
    for start in range(0, len(rho), 12):
        rr = rho[start:start+12]
        second = z1[..., None]*rr+z2[..., None]*np.sqrt(1-rr**2)
        pattern_loss = np.empty((bcount, 4, len(rr)))
        for i, (s1, s2) in enumerate(SIGNS):
            joint = expit(s1*z1)[..., None]*expit(s2*second)
            pattern_loss[:, i] = -np.log(joint.mean(axis=1))
        result[..., start:start+len(rr)] = np.einsum("bpq,tp->btq", pattern_loss, freq)
    return result


def marg_gradient(active, y1, y2, points):
    result = np.empty((len(active), len(y2)))
    x1, z2 = active[..., 0], active[..., 1]
    for t, point in enumerate(points):
        x2 = point*x1+math.sqrt(1-point*point)*z2
        dx = x1-point/math.sqrt(1-point*point)*z2
        total = np.zeros(len(active))
        for s1, s2 in SIGNS:
            first, second = expit(s1*x1), expit(s2*x2)
            freq = np.logical_and(y1 == s1, y2[t] == s2).mean()
            term = -(first*second*expit(-s2*x2)*s2*dx).sum(axis=1)/(first*second).sum(axis=1)
            total += freq*term
        result[:, t] = total
    return result


def full_rows_raw(prefix, y1, yn, y2, rho, points):
    """Independent signed-log table formulation, not production cache identity."""
    count, kcount = prefix.shape[:2]
    out = np.empty((count, len(y2), len(rho)))
    grads = np.empty((count, len(y2)))
    offsets = np.empty((count, len(y1)))
    pos, neg = (yn > 0).astype(float), (yn < 0).astype(float)
    for b in range(count):
        z = prefix[b]
        plus = -np.logaddexp(0., -z[:, 2:])
        minus = -np.logaddexp(0., z[:, 2:])
        # Uses every original nuisance sign, via two disjoint sign masks.
        base = pos @ plus.T+neg @ minus.T-np.logaddexp(0., -y1[:, None]*z[None, :, 0])
        offsets[b] = math.log(kcount)-logsumexp(base, axis=1)
        for j, q in enumerate(rho):
            x2 = q*z[:, 0]+math.sqrt(1-q*q)*z[:, 1]
            for t, second in enumerate(y2):
                logs = base-np.logaddexp(0., -second[:, None]*x2)
                out[b, t, j] = (math.log(kcount)-logsumexp(logs, axis=1)).mean()
        for t, q in enumerate(points):
            x2 = q*z[:, 0]+math.sqrt(1-q*q)*z[:, 1]
            dx = z[:, 0]-q/math.sqrt(1-q*q)*z[:, 1]
            logs = base-np.logaddexp(0., -y2[t, :, None]*x2)
            posterior = np.exp(logs-logsumexp(logs, axis=1)[:, None])
            derivative = y2[t, :, None]*dx*expit(-y2[t, :, None]*x2)
            grads[b, t] = -np.sum(posterior*derivative, axis=1).mean()
    return out, grads, offsets


def selftest():
    grid = np.linspace(-.75, .75, 801)
    nodes = -.75*np.cos(np.arange(193)*np.pi/192)
    values = np.stack((np.ones(193), nodes, nodes**5, np.cos(nodes)))
    got = reconstruct_interpolation(nodes, values, grid)
    expected = np.stack((np.ones(801), grid, grid**5, np.cos(grid)))
    err = float(np.max(abs(got-expected)))
    require(err < 2e-13, "DCT interpolation deterministic test failed")
    x = np.array([0., .5, 1.])
    cc = np.array([[1., 1.], [0., 1.5], [0., .75], [0., .125]])
    vals = integral_spline(x, cc, np.array([.25, .5, .75]))
    require(np.max(abs(vals-np.array([.25, .5, .75])**4/4)) < 1e-15, "Cubic integral test failed")
    require(spline_curvature_exact(x, cc) == 3, "Spline derivative extremum test failed")
    z = np.sin(np.arange(2*4*5, dtype=float).reshape(2, 4, 5)/3)
    y1 = np.array([-1, 1, 1, -1, 1], dtype=np.int8)
    yn = np.array([[-1, 1, -1], [1, 1, 1], [-1, -1, 1], [1, -1, -1], [1, -1, 1]], dtype=np.int8)
    y2 = np.array([y1, -y1, [1, 1, -1, -1, 1]], dtype=np.int8)
    points = np.array([-.1, .2, .5])
    raw, gr, _ = full_rows_raw(z, y1, yn, y2, np.array([-.2, 0., .6]), points)
    gm = marg_gradient(z[:, :, :2], y1, y2, points)
    raw_error, derivative_errors = 0., []
    for b in range(2):
        for t in range(3):
            for j, rho in enumerate((-.2, 0., .6)):
                x2 = rho*z[b, :, 0]+math.sqrt(1-rho*rho)*z[b, :, 1]
                row_values = []
                for row in range(5):
                    logs = []
                    for k in range(4):
                        all_x = [z[b, k, 0], x2[k], *z[b, k, 2:]]
                        all_y = [y1[row], y2[t, row], *yn[row]]
                        logs.append(sum(-float(np.logaddexp(0., -xx*yy)) for xx, yy in zip(all_x, all_y)))
                    row_values.append(math.log(4)-float(logsumexp(logs)))
                raw_error = max(raw_error, abs(raw[b, t, j]-sum(row_values)/5))
    for t, point in enumerate(points):
        pair = np.array([point-1e-5, point+1e-5])
        vr = full_rows_raw(z, y1, yn, y2, pair, points)[0]
        vm = marg_rebuild(z[:, :, :2], y1, y2, pair)
        derivative_errors.extend([float(np.max(abs((vr[:, t, 1]-vr[:, t, 0])/2e-5-gr[:, t]))),
                                  float(np.max(abs((vm[:, t, 1]-vm[:, t, 0])/2e-5-gm[:, t])))])
    require(raw_error < 2e-13 and max(derivative_errors) < 2e-9, "Direct full-row/gradient deterministic check failed")
    return {"DCT_polynomial_and_cosine_max_abs": err, "spline_integral_and_curvature": True,
            "raw_scalar_full_logsumexp_max_abs": raw_error, "raw_marg_gradient_finite_difference_max_abs": max(derivative_errors),
            "new_random_draws": 0, "production_code_imported": False}


def freeze():
    frozen = read(TRAIN/"FREEZE.json")
    require(not PROTOCOL.exists(), "Audit already frozen")
    inputs = {relative(TRAIN/"FREEZE.json"): sha(TRAIN/"FREEZE.json"),
              "script/bankvar_train.py": frozen["source_sha256"],
              "script/fincal_train.py": frozen["helper_sha256"],
              frozen["plan_path"]: frozen["plan_sha256"],
              frozen["reference_path"]: frozen["reference_sha256"],
              frozen["local_model_path"]: frozen["local_model_sha256"],
              frozen["old_taylor_path"]: frozen["old_taylor_sha256"]}
    inputs.update({row["path"]: row["sha256"] for row in frozen["data_inputs"]})
    for path, digest in inputs.items():
        require(sha(ROOT/path) == digest, "Frozen input mismatch: "+path)
    data = {"schema": "bankvar-train-audit-v1", "created_before_reconstruction_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(SOURCE), "inputs": inputs, "selftest": selftest(),
            "scope": {"prior_replay": "All 16 new seed streams, all 512x256x256 Gaussian values; hash and all active pairs",
                      "observed_data": "All 16 old NPZ hashes and original y1/y2/ynoise array hashes; no new labels",
                      "raw": "Each stream first 8 banks, complete N4096 nuisance rows, fixed nine rho; signed log-sigmoid tables and direct full-particle logsumexp",
                      "raw_interpolation": "Every saved raw node curve to every saved 801 point via independent DCT-I/Clenshaw, not full raw likelihood recomputation",
                      "marg": "All 512 banks in every stream, all saved nodes, four active patterns via direct sigmoid products; all saved interpolation",
                      "gradients": "All three prescribed numerical MLE points; raw first 8 banks, marg every bank; direct derivative expressions",
                      "selection": "All 864 grid choices, 144 repeated MLE rows (48 unique), penalties, reference KL, mesh/interpolation bounds",
                      "statistics": "No primary confidence intervals/tests; handled separately by root"},
            "raw_rho": TARGETS.tolist(), "raw_banks": 8, "replay_block": 32,
            "tolerances": {"marg_direct_vs_saved_nodes": 2e-12, "DCT_vs_saved_curves": 3e-11,
                           "raw_direct_centered_vs_interpolated": "2*analytic epsilon + 1e-9; fallback uses 1e-9",
                           "raw_offsets": 2e-10, "path_gradients": 2e-10,
                           "MLE_rho": 2e-11, "reference_KL": 3e-14, "selected_scores": 3e-12,
                           "bound_reconstruction": 1e-14},
            "failure_policy": "Preserve any failed execution and report first mismatch; no automatic repair of scientific outputs",
            "new_scientific_random_samples": 0, "RNG_replay_not_new_independent_evidence": True}
    write_new(PROTOCOL, data)
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "selftest": data["selftest"]}))


def audit():
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Audit source changed after freeze")
    for path, digest in p["inputs"].items():
        require(sha(ROOT/path) == digest, "Audit bound input changed: "+path)
    require((TRAIN/"RESULTS.json").exists(), "Wait until all production streams complete")
    require(not (OUT/"EXECUTION.json").exists(), "Audit already started; no automatic replay")
    frozen, results = read(TRAIN/"FREEZE.json"), read(TRAIN/"RESULTS.json")
    require(results["freeze_sha256"] == sha(TRAIN/"FREEZE.json"), "Production result freeze mismatch")
    write_new(OUT/"EXECUTION.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                   "audit_source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
                                   "production_results_sha256": sha(TRAIN/"RESULTS.json")})
    config = frozen["config"]
    grid, rfun, penalties, penalty_ds, coefficients = reference(frozen)
    require(results["coefficients"] == coefficients, "Penalty coefficient mismatch")
    indices = np.array([np.argmin(abs(grid-r)) for r in TARGETS])
    require(np.max(abs(grid[indices]-TARGETS)) < 1e-14, "Audit targets not on master grid")
    h = up(max(F.from_float(float(v))-F.from_float(float(u)) for u, v in zip(grid[:-1], grid[1:])))
    eps = up(F(700)*F(25, 28)**192)
    records, started = [], time.perf_counter()
    for stream in range(16):
        tick = time.perf_counter()
        row = next(v for v in results["records"] if v["stream"] == stream)
        require(sha(ROOT/row["file"]) == row["file_sha256"], "Saved stream bytes changed")
        data_row = next(v for v in frozen["data_inputs"] if v["stream"] == stream)
        with np.load(ROOT/data_row["path"]) as loaded:
            y1, y2, yn = (loaded[key].copy() for key in ("y1", "y2", "ynoise"))
        for key, arr in (("y1", y1), ("y2", y2), ("ynoise", yn)):
            require(array_sha(arr) == row["provenance"][key+"_sha256"], "Observed array changed")
        active = np.empty((512, 256, 2))
        prefix = None
        digest = hashlib.sha256()
        rng = np.random.Generator(np.random.PCG64(202610021800+stream))
        for begin in range(0, 512, 32):
            z = rng.standard_normal((32, 256, 256))
            digest.update(z.tobytes())
            active[begin:begin+32] = z[:, :, :2]
            if begin == 0:
                prefix = z[:8].copy()
        require(digest.hexdigest() == row["provenance"]["prior_sha256"], "Full prior replay mismatch")
        maxnorm, gate = ellipsoid_gate(active)
        require(maxnorm == row["active_gate"]["max_squared_radius_upper"] and gate == row["active_gate"]["pass_all"], "Active gate mismatch")
        points, boundary, pairs = bisect_mle(y1, y2, rfun)
        with np.load(ROOT/row["file"]) as saved:
            require(np.array_equal(active, saved["prior_active"]), "Active replay mismatch")
            nodes = saved["nodes"].copy()
            require(np.array_equal(grid, saved["rho"]), "Saved master grid mismatch")
            require(np.max(abs(points-saved["gradient_rho"])) <= p["tolerances"]["MLE_rho"], "Independent MLE mismatch")
            require(np.array_equal(boundary, saved["gradient_MLE_boundary"]), "MLE boundary mismatch")
            raw, marg = saved["raw"].copy(), saved["marg"].copy()
            raw_nodes, marg_nodes = saved["raw_node_scores"].copy(), saved["marg_node_scores"].copy()
            raw_interp = reconstruct_interpolation(nodes, raw_nodes, grid)
            raw_interp -= raw_interp[:, :, 400:401].copy()
            raw_interp_err = float(np.max(abs(raw_interp-raw)))
            marg_direct = marg_rebuild(active, y1, y2, nodes)
            marg_node_err = float(np.max(abs(marg_direct-marg_nodes)))
            marg_interp = reconstruct_interpolation(nodes, marg_direct, grid)
            marg_interp -= marg_interp[:, :, 400:401].copy()
            marg_interp_err = float(np.max(abs(marg_interp-marg)))
            require(raw_interp_err <= p["tolerances"]["DCT_vs_saved_curves"], "Raw saved interpolation mismatch")
            require(marg_node_err <= p["tolerances"]["marg_direct_vs_saved_nodes"], "Direct marg node mismatch")
            require(marg_interp_err <= p["tolerances"]["DCT_vs_saved_curves"], "Marg saved interpolation mismatch")
            actual_points = saved["gradient_rho"].copy()
            fullraw, rawgrad, offsets = full_rows_raw(prefix, y1, yn, y2, grid[indices], actual_points)
            fullraw -= fullraw[:, :, 4:5].copy()
            # Fixed take preserves bank/teacher/point axis ordering.
            raw_err = float(np.max(abs(fullraw-np.take(raw[:8], indices, axis=2))))
            offset_err = float(np.max(abs(offsets-saved["offset"][:8])))
            raw_gradient_err = float(np.max(abs(rawgrad-saved["raw_path_gradient"][:8])))
            marggrad = marg_gradient(active, y1, y2, actual_points)
            marg_gradient_err = float(np.max(abs(marggrad-saved["marg_path_gradient"])))
            ieps = eps if gate else 0.
            require(raw_err <= 2*ieps+1e-9, "Direct raw prefix score mismatch")
            require(offset_err <= p["tolerances"]["raw_offsets"], "Raw offset mismatch")
            require(max(raw_gradient_err, marg_gradient_err) <= p["tolerances"]["path_gradients"], "Direct fixed-point path gradient mismatch")
        bound_diffs, kldiffs, scorediffs, rhoerrors = [], [], [], []
        selected_count, mle_count = 0, 0
        bases = {str(b): curvature(active[:b]) for b in (32, 128, 512)}
        require(bases == row["base_curvature_D"], "Base curvature reconstruction mismatch")
        for output in row["outputs"]:
            ti = config["teachers"].index(output["theta"])
            teacher_r = float(rfun(output["theta"]))
            require(output["empirical_pair"] == float(pairs[ti]), "Empirical pair mismatch")
            if output["method"] == "exact":
                estimate = float(points[ti]); mle_count += 1
                require(output["MLE_at_boundary"] == bool(boundary[ti]), "Output MLE flag mismatch")
            else:
                method, b = output["method"], output["B"]
                curves = marg[:b, ti] if method == "marg" else raw[:b, ti]
                scores = curves.mean(axis=0)+penalties[method]
                minimum = min(enumerate(scores), key=lambda item: item[1])[0]
                require(minimum == output["grid_index"], "Independent grid choice mismatch")
                estimate = float(grid[minimum]); selected_count += 1
                scorediffs.append(abs(float(scores[minimum])-output["selected_score"]))
                dd = float(np.nextafter(bases[str(b)]+penalty_ds[method], np.inf))
                mesh = up(F.from_float(dd)*F.from_float(h)**2/8)
                total = up(F.from_float(mesh)+2*F.from_float(ieps))
                bound_diffs += [abs(dd-output["curvature_D"]), abs(mesh-output["mesh_regret_bound_real_arithmetic"]),
                                abs(total-output["continuous_objective_regret_bound_real_arithmetic"])]
                require(output["mesh_step_upper"] == h, "Mesh spacing mismatch")
                require(output["interpolation_uncentered_sup_error_upper"] == ieps, "Interpolation bound mismatch")
            rhoerrors.append(abs(estimate-output["rho"]))
            kldiffs.append(abs(observed_kl(teacher_r, float(rfun(estimate)))-output["reference_joint_KL"]))
        require(max(rhoerrors) <= p["tolerances"]["MLE_rho"], "Selected rho reconstruction mismatch")
        require(max(kldiffs) <= p["tolerances"]["reference_KL"], "Reference KL reconstruction mismatch")
        require(max(scorediffs) <= p["tolerances"]["selected_scores"], "Selected score reconstruction mismatch")
        require(max(bound_diffs) <= p["tolerances"]["bound_reconstruction"], "Bound reconstruction mismatch")
        require((selected_count, mle_count) == (54, 9), "Output count mismatch")
        record = {"stream": stream, "pass_all": True, "production_array_sha256": row["file_sha256"],
                  "prior_full_hash_pass": True, "observed_data_hash_pass": True,
                  "active_gate": gate, "max_squared_radius_upper": maxnorm,
                  "raw_prefix_full_rows_score_max_abs": raw_err, "raw_offset_max_abs": offset_err,
                  "raw_saved_interpolation_max_abs": raw_interp_err, "marg_direct_nodes_max_abs": marg_node_err,
                  "marg_independent_interpolation_max_abs": marg_interp_err,
                  "raw_prefix_path_gradient_max_abs": raw_gradient_err, "marg_all_path_gradient_max_abs": marg_gradient_err,
                  "reference_KL_max_abs": max(kldiffs), "selected_score_max_abs": max(scorediffs),
                  "rho_max_abs": max(rhoerrors), "bound_max_abs": max(bound_diffs),
                  "grid_choices_checked": selected_count, "repeated_MLE_rows_checked": mle_count,
                  "marg_bank_pattern_nodes": 512*4*len(nodes), "raw_bank_teacher_points": 8*3*len(TARGETS),
                  "elapsed_seconds": time.perf_counter()-tick}
        records.append(record)
        write_new(OUT/f"stream_{stream:02d}.json", record)
        print(json.dumps({"audit_stream": stream, "passed": True, "seconds": record["elapsed_seconds"],
                          "raw_error": raw_err, "marg_error": marg_node_err}), flush=True)
    output = {"source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
              "production_results_sha256": sha(TRAIN/"RESULTS.json"), "pass_all": True,
              "records": records, "scope": p["scope"], "elapsed_seconds": time.perf_counter()-started,
              "accounting": {"replayed_prior_banks": 8192, "replayed_prior_Gaussian_scalars": 536870912,
                             "raw_full_row_banks_recomputed": 128, "raw_full_row_candidate_teacher_scores": 3456,
                             "marg_bank_pattern_nodes": sum(r["marg_bank_pattern_nodes"] for r in records),
                             "raw_fixed_point_derivatives": 384, "marg_fixed_point_derivatives": 24576,
                             "grid_choices": 864, "repeated_MLE_rows": 144, "unique_MLE_inputs": 48,
                             "new_independent_scientific_samples": 0}}
    write_new(OUT/"RESULTS.json", output)
    report = ROOT/Path(frozen["plan_path"]).parent/"TRAIN_AUDIT.md"
    text = ["# 独立 TRAIN 审计", "", "审计源未导入生产评分、选点、梯度或插值函数；所有指定检查通过。", "",
            "完整重放16条新 prior 流并核全部 Gaussian 哈希和 active 输入；旧完整数据NPZ及观测数组哈希全核。raw仅前8 bank保留全部4096真实完整观测，在9个预定rho直接重算，使用正负log-sigmoid表及完整粒子logsumexp。marg全512bank、全部节点以四模式sigmoid乘积均值重建。", "",
            "所有保存的 raw 节点到801点曲线用独立DCT-I/Clenshaw重建；这不等于重新计算全部raw likelihood。marg同时核直接节点及独立插值。每流3个同数据数值MLE固定点，raw前8bank、marg全部bank直接重算路径梯度。", "",
            "全部864网格选点、144重复MLE展示行（48不同数据/teacher输入）、参考KL、输入norm向外门、网格与插值界独立核验。主统计区间及检验由root另核，本审计不重复宣称负责。", "",
            "误差资格仍不构成全域浮点ζ或Gaussian积分余项证书。完整RNG重放与限定raw评分重算范围分别记账，未把8192bank称为全部底层raw评分重跑。", "",
            "```json", json.dumps(output["accounting"], ensure_ascii=False, indent=2), "```", "",
            f"审计耗时 {output['elapsed_seconds']:.3f} 秒。协议SHA256 `{sha(PROTOCOL)}`。结果SHA256 `{sha(OUT/'RESULTS.json')}`。", ""]
    require(not report.exists(), "Refuse report overwrite")
    report.write_text("\n".join(text), encoding="utf-8", newline="\n")
    print(json.dumps({"audit_complete": True, "results_sha256": sha(OUT/"RESULTS.json"), "seconds": output["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("selftest", "freeze", "run"))
    args = parser.parse_args()
    if args.phase == "selftest":
        print(json.dumps(selftest()))
    elif args.phase == "freeze":
        freeze()
    else:
        try:
            audit()
        except Exception as error:
            if (OUT/"EXECUTION.json").exists() and not (OUT/"FAILURE.json").exists():
                write_new(OUT/"FAILURE.json", {"error": str(error), "traceback": traceback.format_exc(),
                          "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(SOURCE),
                          "no_automatic_retry": True})
            raise
