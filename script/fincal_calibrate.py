"""Teacher-free finite-n/K Price calibration with separate validation streams.

Implement, compile and inspect before freezing. Every numerical/random phase
requires a frozen protocol. The learned penalty is the exact antiderivative of
the frozen cubic spline of g_prime, never an independent spline fit to g_values.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
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
from scipy.interpolate import CubicSpline, PPoly
from scipy.special import expit, roots_hermitenorm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/finite_calibration/CAL"
PROTOCOL = OUT / "PROTOCOL.json"
SOURCE = Path(__file__).resolve()
OLD_REF = ROOT / "artifacts/regularization_learning/NUMERIC/GROWTH_REFERENCE.json"
OLD_J = ROOT / "artifacts/regularization_learning/GROWTH/J_REFERENCE.json"
R = 0.75
A = 0.25
RHO801 = np.arange(-400, 401, dtype=float) * (R/400)
RHO101 = RHO801[::8]
NS, KS = [62, 254], [64, 256]
SIGNS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=int)
SIGN_PRODUCTS = SIGNS.prod(axis=1)
BLOCK, BANKS, KMAX, DMAX = 64, 2048, 256, 256
TEACHERS = [0.0, 0.3, 0.6, 0.375]
SEEDS = {"train": 202610021100, "validation": 202610021200}
STREAMS = {"train": 16, "validation": 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(test, message):
    if not test:
        raise RuntimeError(message)


def protocol():
    p = read(PROTOCOL)
    require(p["source_sha256"] == sha(SOURCE), "Calibration source changed after freeze")
    for name, digest in p["inputs"].items():
        require(sha(ROOT/name) == digest, "Frozen input changed: " + name)
    return p


def freeze(plan):
    plan = Path(plan).resolve()
    require(plan.is_relative_to(ROOT) and plan.name == "PLAN.md" and plan.is_file(), "Use an existing repository PLAN.md")
    require(not PROTOCOL.exists() and not (OUT / "REFERENCE.json").exists(), "Calibration already frozen/started")
    report = plan.parent / "CAL_REPORT.md"
    require(not report.exists(), "Calibration report already exists")
    data = {
        "schema": "fincal-calibration-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(SOURCE), "inputs": {rel(p): sha(p) for p in (plan, OLD_REF, OLD_J)},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "report": rel(report), "a": 1.0, "R": R, "ns": NS, "ks": KS,
        "rho801": RHO801.tolist(), "rho101": RHO101.tolist(), "active_signs": SIGNS.tolist(),
        "streams": STREAMS, "banks_per_stream": BANKS, "block": BLOCK,
        "seed_bases": SEEDS, "max_prior_shape_per_block": [BLOCK, KMAX, DMAX],
        "rng": "numpy Generator(PCG64(seed_base+stream)); sequential standard_normal((64,256,256)) blocks",
        "prior_reuse": "Each n/K takes its coordinate and particle prefix of the same maximum block; rho/modes share prior",
        "population_only": "Canonical nuisance signs and four active modes estimate population/conditional-bank expectation, not finite-data full-bank training",
        "Price_derivative": "D_y=-y1*y2 sum_k pi_k(1-pi_k) sigmoid(-y1 X1) sigmoid(-y2 X2), a=1",
        "symmetry": "U_prime odd, V_prime even, then g_prime=-U_prime-r*V_prime; exact model symmetry, no extra independent samples",
        "penalty": "CubicSpline(g_prime101,bc_type=not-a-knot).antiderivative(), minus its value at zero",
        "g_second_sup": "Outward-rounded exact rational supremum of derivative of the stored g_prime cubic on each segment",
        "reference": {"GH_orders": [64, 96], "main_order": 64, "r_and_rprime": "direct tanh and Price Gaussian integrals",
                      "H": "antiderivative of r801 cubic spline, normalized at zero; numerical reference, not total integral certificate",
                      "d_A_var_J": "reuse immutable prior references; no new log-sigmoid or J integrals"},
        "qualification": {"reference64_vs96_absolute": 1e-9, "r64_vs96_absolute": 1e-9, "g51_vs101_absolute": 2e-6,
                          "training_gprime_SE_absolute": 1e-4,
                          "gprime51_vs101_absolute": 1e-5, "validation_4_streams": "descriptive, no uniform coverage claim"},
        "spline_resolution_check": "51-node nested subset vs 101 on 1001 prescribed points for qualification; additional piecewise stationary-point search is diagnostic only",
        "validation_teachers": TEACHERS, "teacher_specific_tuning": False,
        "spline_optimum": "Enumerate numerical real roots of validation derivative spline and endpoints; not true population or training globopt certificate",
        "spline_tie_tolerance": 1e-12, "spline_tie_rule": "smallest absolute rho, then signed rho; preserve all tied candidates",
        "validation_penalty_frozen_before_validation": True, "automatic_budget_extension": False,
        "planned_new_Gaussian_scalars": (16+4)*BANKS*KMAX*DMAX,
        "planned_unique_prior_vectors": (16+4)*BANKS*KMAX, "planned_banks": (16+4)*BANKS,
        "new_observations": 0, "new_observed_data_fits": 0,
    }
    write_new(PROTOCOL, data)
    print(json.dumps({"frozen": True, "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(SOURCE),
                      "Gaussian_scalars": data["planned_new_Gaussian_scalars"]}))


class ReferenceIntegrator:
    def __init__(self):
        self.rules = {}
        self.cache = {}
        self.counts = Counter()
        for order in (64, 96):
            z, w = roots_hermitenorm(order)
            w = w/math.sqrt(2*math.pi)
            self.rules[order] = (z, w[:, None]*w[None, :])

    def one(self, rho, order):
        magnitude = abs(float(rho))
        key = (magnitude, order)
        if key not in self.cache:
            z, weights = self.rules[order]
            first = np.tanh(z[:, None]/2)
            second = np.tanh((magnitude*z[:, None]+math.sqrt(1-magnitude*magnitude)*z[None, :])/2)
            rr = float(np.sum(weights*first*second))
            rp = float(np.sum(weights*(1-first*first)*(1-second*second))/4)
            self.cache[key] = (0.0 if magnitude == 0 else rr, rp)
            self.counts["two_dimensional_scalar_integrals"] += 2
            self.counts["shared_tensor_node_entries"] += order*order
        rr, rp = self.cache[key]
        return (-rr if rho < 0 else rr), rp

    def load_table(self, reference):
        for table in reference["orders"]:
            for rho, rr, rp in zip(RHO801, table["r801"], table["rp801"]):
                if rho >= 0:
                    self.cache[(float(rho), table["order"])] = (rr, rp)


def reference():
    p = protocol()
    require(not (OUT / "REFERENCE.json").exists(), "Reference already exists")
    worker = ReferenceIntegrator()
    old, old_j = read(OLD_REF), read(OLD_J)
    orders = []
    for order in (64, 96):
        values = np.array([worker.one(float(rho), order) for rho in RHO801])
        rr, rp = values[:, 0], values[:, 1]
        hs = CubicSpline(RHO801, rr, bc_type="not-a-knot").antiderivative()
        H = hs(RHO801)-hs(0.0)
        inherited_order = next(row for row in old["all_orders"] if row["order"] == order)
        orders.append({"order": order, "r801": rr.tolist(), "rp801": rp.tolist(), "H801": H.tolist(),
                       "d": inherited_order["d_r0"], "A": inherited_order["A"], "var_logsigmoid": inherited_order["var_logsigmoid"]})
    diff = {key: float(np.max(np.abs(np.array(orders[0][key])-orders[1][key]))) for key in ("r801", "rp801", "H801", "d", "var_logsigmoid")}
    inherited = {"d": old["d_r0"], "A": old["A"], "var_logsigmoid": old["var_logsigmoid"]}
    cases = []
    for n, k in zip(NS, KS):
        J = next(row["main"] for row in old_j["rows"] if row["K"] == k)
        cases.append({"n": n, "K": k, "J": J, "w": J/math.sqrt(n*inherited["var_logsigmoid"])})
    data = {"source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "a": 1.0, "R": R,
            "rho801": RHO801.tolist(), "orders": orders, "inherited": inherited, "cases": cases,
            "main_order": 64, "absolute_differences": diff,
            "r_qualification_pass": diff["r801"] <= p["qualification"]["r64_vs96_absolute"],
            "qualification_by_field": {key: value <= p["qualification"]["reference64_vs96_absolute"] for key, value in diff.items()},
            "reference_qualification_pass": all(value <= p["qualification"]["reference64_vs96_absolute"] for value in diff.values()),
            "H_scope": "Exact antiderivative of the specified r801 cubic; qualification is not a strict Gaussian integral certificate",
            "accounting": {**dict(worker.counts), "H_spline_antiderivatives": 2, "new_random_samples": 0}}
    write_new(OUT / "REFERENCE.json", data)
    print(json.dumps({"reference_complete": True, "differences": diff, "r_pass": data["r_qualification_pass"],
                      "all_reference_pass": data["reference_qualification_pass"], "accounting": data["accounting"]}))


def price_block(z):
    """Actual per-bank four-mode Price derivatives; canonical nuisance only."""
    result = np.empty((len(z), len(NS), len(RHO101), 4), dtype=float)
    z1, z2 = z[:, :, 0], z[:, :, 1]
    log_first = {s: -np.logaddexp(0.0, -s*z1) for s in (-1, 1)}
    comp_first = {s: expit(-s*z1) for s in (-1, 1)}
    cumulative = -np.logaddexp(0.0, -z[:, :, 2:])
    np.cumsum(cumulative, axis=2, out=cumulative)
    for ri, rho in enumerate(RHO101):
        x2 = rho*z1+math.sqrt(1-rho*rho)*z2
        log_second = {s: -np.logaddexp(0.0, -s*x2) for s in (-1, 1)}
        comp_second = {s: expit(-s*x2) for s in (-1, 1)}
        for ni, (n, k) in enumerate(zip(NS, KS)):
            noise = cumulative[:, :k, n-1]
            for si, (s1, s2) in enumerate(SIGNS):
                logits = noise+log_first[int(s1)][:, :k]+log_second[int(s2)][:, :k]
                logits -= logits.max(axis=1, keepdims=True)
                np.exp(logits, out=logits)
                logits /= logits.sum(axis=1, keepdims=True)
                result[:, ni, ri, si] = -int(s1*s2)*np.sum(
                    logits*(1-logits)*comp_first[int(s1)][:, :k]*comp_second[int(s2)][:, :k], axis=1)
    return result


def project_modes(mean_modes, r101):
    u = mean_modes.mean(axis=-1)
    v = np.sum(mean_modes*SIGN_PRODUCTS, axis=-1)/4
    us = (u-u[..., ::-1])/2
    vs = (v+v[..., ::-1])/2
    gp = -us-r101*vs
    return {"U_prime_unsym": u, "V_prime_unsym": v, "U_prime": us, "V_prime": vs, "g_prime": gp}


def mean_se(values):
    array = np.asarray(values)
    return array.mean(axis=0), array.std(axis=0, ddof=1)/math.sqrt(len(array))


def run_streams(pool):
    p = protocol()
    ref = read(OUT / "REFERENCE.json")
    require(ref["source_sha256"] == sha(SOURCE) and ref["protocol_sha256"] == sha(PROTOCOL), "Reference binding mismatch")
    require(ref["reference_qualification_pass"], "Reference qualification failed; stop without new random generation")
    target = OUT / ("TRAIN.json" if pool == "train" else "VALIDATION_STREAMS.json")
    require(not target.exists(), "Pool already completed")
    execution_path = OUT / ("TRAIN_EXECUTION.json" if pool == "train" else "VALIDATION_PROTOCOL.json")
    require(not execution_path.exists(), "Pool was already started; preserve its execution receipt, no automatic retry")
    reference_sha = sha(OUT / "REFERENCE.json")
    model_sha = None
    if pool == "validation":
        require((OUT / "MODEL.json").is_file(), "Freeze calibration model before validation")
        model_sha = sha(OUT / "MODEL.json")
    write_new(execution_path, {"pool": pool, "created_before_first_Gaussian_utc": datetime.now(timezone.utc).isoformat(),
                              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
                              "reference_sha256": reference_sha, "model_sha256": model_sha,
                              "streams": STREAMS[pool], "banks_per_stream": BANKS, "seed_base": SEEDS[pool],
                              "block_shape": [BLOCK, KMAX, DMAX], "automatic_retry": False})
    records = []
    started = time.perf_counter()
    for index in range(STREAMS[pool]):
        dest = OUT / f"{pool}_{index:02d}.npz"
        require(not dest.exists(), "Pool array already exists; no automatic retry: "+rel(dest))
        rng = np.random.Generator(np.random.PCG64(SEEDS[pool]+index))
        derivatives = np.empty((BANKS, len(NS), len(RHO101), 4), dtype=float)
        prior_hash = hashlib.sha256()
        first_hash = None
        for offset in range(0, BANKS, BLOCK):
            z = rng.standard_normal((BLOCK, KMAX, DMAX))
            prior_hash.update(z.tobytes())
            if offset == 0:
                first_hash = hashlib.sha256(z.tobytes()).hexdigest()
            derivatives[offset:offset+BLOCK] = price_block(z)
        require(np.isfinite(derivatives).all(), "Non-finite Price derivative")
        np.savez_compressed(dest, D_modes=derivatives, rho=RHO101, ns=np.array(NS), ks=np.array(KS), signs=SIGNS)
        records.append({"stream": index, "seed": SEEDS[pool]+index, "file": rel(dest), "file_sha256": sha(dest),
                        "full_prior_sha256": prior_hash.hexdigest(), "first64_prior_sha256": first_hash,
                        "D_modes_mean": derivatives.mean(axis=0).tolist()})
        print(json.dumps({"pool": pool, "completed_streams": len(records), "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    if model_sha:
        require(sha(OUT / "MODEL.json") == model_sha, "Calibration model changed during validation")
    require(sha(OUT / "REFERENCE.json") == reference_sha, "Reference changed during pool execution")
    protocol()  # Source and PLAN/old references must also remain frozen.
    ledger = {"new_Gaussian_scalars": STREAMS[pool]*BANKS*KMAX*DMAX,
              "unique_prior_vectors": STREAMS[pool]*BANKS*KMAX, "banks": STREAMS[pool]*BANKS,
              "independent_streams": STREAMS[pool], "new_observations": 0, "new_observed_data_fits": 0,
              "particle_rho_mode_entries": STREAMS[pool]*BANKS*sum(KS)*len(RHO101)*4}
    data = {"pool": pool, "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
            "execution_receipt_sha256": sha(execution_path), "reference_sha256": reference_sha, "model_sha256_at_start": model_sha,
            "records": records, "elapsed_seconds": time.perf_counter()-started, "ledger": ledger}
    write_new(target, data)
    if pool == "train":
        build_model(data, ref)
    else:
        summarize_validation(data, ref)


def outward_float(value):
    floating = float(value)
    if Fraction.from_float(floating) < value:
        floating = math.nextafter(floating, math.inf)
    return floating


def derivative_sup_exact(c, x):
    """Supremum of |gp'| for real polynomials with stored binary coefficients."""
    largest = Fraction(0)
    for j in range(len(x)-1):
        a, b, cc = [Fraction.from_float(float(c[i, j])) for i in range(3)]
        width = Fraction.from_float(float(x[j+1]))-Fraction.from_float(float(x[j]))
        points = [Fraction(0), width]
        if a:
            vertex = -b/(3*a)
            if 0 < vertex < width:
                points.append(vertex)
        for point in points:
            largest = max(largest, abs(3*a*point*point+2*b*point+cc))
    return {"upper_float": outward_float(largest), "exact_numerator": str(largest.numerator), "exact_denominator": str(largest.denominator),
            "scope": "Exact for stored binary cubic coefficients and knots; does not certify Monte Carlo or model-integral accuracy"}


def spline_penalty(gprime, x=RHO101):
    gp = CubicSpline(x, gprime, bc_type="not-a-knot")
    anti = gp.antiderivative()
    return gp, anti, float(anti(0.0))


def resolution_comparison(gprime):
    fine, anti_f, shift_f = spline_penalty(gprime)
    coarse, anti_c, shift_c = spline_penalty(gprime[::2], RHO101[::2])
    points_g = list(RHO101)
    points_gp = list(RHO101)
    for j in range(len(RHO101)-1):
        cj = j//2
        offset = RHO101[j]-coarse.x[cj]
        a, b, c, d = coarse.c[:, cj]
        shifted = np.array([a, 3*a*offset+b, 3*a*offset*offset+2*b*offset+c,
                            a*offset**3+b*offset**2+c*offset+d])
        diff = fine.c[:, j]-shifted
        width = RHO101[j+1]-RHO101[j]
        for coeff, points in ((diff, points_g), (np.array([3*diff[0], 2*diff[1], diff[2]]), points_gp)):
            nonzero = np.trim_zeros(coeff, "f")
            if len(nonzero) <= 1:
                continue
            for root in np.roots(nonzero):
                if abs(root.imag) <= 1e-10 and 0 <= root.real <= width:
                    points.append(float(RHO101[j]+root.real))
    dense = np.linspace(-R, R, 1001)
    points_g = np.array(points_g+list(dense))
    points_gp = np.array(points_gp+list(dense))
    gdifference = np.abs((anti_f(points_g)-shift_f)-(anti_c(points_g)-shift_c))
    gpdifference = np.abs(fine(points_gp)-coarse(points_gp))
    dense_g = np.abs((anti_f(dense)-shift_f)-(anti_c(dense)-shift_c))
    dense_gp = np.abs(fine(dense)-coarse(dense))
    return {"g_max_abs_difference": float(dense_g.max()), "gprime_max_abs_difference": float(dense_gp.max()),
            "qualification_mesh_points": 1001,
            "g_piecewise_search_max_abs_difference": float(gdifference.max()),
            "gprime_piecewise_search_max_abs_difference": float(gpdifference.max()),
            "g_max_location": float(points_g[int(np.argmax(gdifference))]), "candidate_points_g": len(points_g),
            "candidate_points_gprime": len(points_gp),
            "scope": "Qualification on 1001 prescribed points; piecewise stationary-point search is additional diagnostic, neither is a strict integral certificate"}


def build_model(train, ref):
    require(not (OUT / "MODEL.json").exists(), "Model already exists")
    p = protocol()
    r101 = np.array(ref["orders"][0]["r801"])[::8]
    streams = np.array([record["D_modes_mean"] for record in train["records"]])
    projected = project_modes(streams, r101)
    means, ses = {}, {}
    for key, values in projected.items():
        means[key], ses[key] = mean_se(values)
    all_g = np.empty((STREAMS["train"], len(NS), len(RHO101)))
    for si in range(STREAMS["train"]):
        for ni in range(len(NS)):
            _, antiderivative, shift = spline_penalty(projected["g_prime"][si, ni])
            all_g[si, ni] = antiderivative(RHO101)-shift
    means["g_values"], ses["g_values"] = mean_se(all_g)
    cases = []
    for ni, case in enumerate(ref["cases"]):
        gp, antiderivative, shift = spline_penalty(means["g_prime"][ni])
        g_values = antiderivative(RHO101)-shift
        sup = derivative_sup_exact(gp.c, gp.x)
        resolution = resolution_comparison(means["g_prime"][ni])
        case_data = {**case, "rho101": RHO101.tolist(), "U_prime": means["U_prime"][ni].tolist(),
                     "V_prime": means["V_prime"][ni].tolist(), "g_prime": means["g_prime"][ni].tolist(), "g_values": g_values.tolist(),
                     "gprime_spline_c": gp.c.tolist(), "gprime_spline_x": gp.x.tolist(),
                     "g_antiderivative_origin_shift": shift, "g_second_sup": sup["upper_float"], "g_second_sup_receipt": sup,
                     "calibration_SE": {key: ses[key][ni].tolist() for key in ses},
                     "per_stream": {key: projected[key][:, ni].tolist() for key in projected},
                     "per_stream_g_values": all_g[:, ni].tolist(), "resolution_51_vs101": resolution,
                     "gprime_SE_pass_by_node": (ses["g_prime"][ni] <= p["qualification"]["training_gprime_SE_absolute"]).tolist(),
                     "g_resolution_pass": resolution["g_max_abs_difference"] <= p["qualification"]["g51_vs101_absolute"],
                     "gprime_resolution_pass": resolution["gprime_max_abs_difference"] <= p["qualification"]["gprime51_vs101_absolute"],
                     "linear_aggregation_check_max": float(np.max(np.abs(g_values-means["g_values"][ni]))) }
        cases.append(case_data)
    main_ref = ref["orders"][0]
    model = {"schema": "fincal-model-v1", "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
             "training_sha256": sha(OUT / "TRAIN.json"), "reference_sha256": sha(OUT / "REFERENCE.json"),
             "a": 1.0, "R": R,
             "reference": {"rho801": RHO801.tolist(), "r801": main_ref["r801"], "rp801": main_ref["rp801"], "H801": main_ref["H801"], **ref["inherited"]},
             "cases": cases, "teacher_free": True, "shared_across_all_teachers": True,
             "penalty_evaluation_contract": "PPoly(gprime_spline_c,gprime_spline_x).antiderivative()(rho) minus its value at zero",
             "training_streams": STREAMS["train"], "gprime_SE_gate_count": len(NS)*len(RHO101),
             "gprime_SE_gate_passed": sum(sum(c["gprime_SE_pass_by_node"]) for c in cases),
             "g_resolution_gates_passed": sum(c["g_resolution_pass"] for c in cases),
             "gprime_resolution_gates_passed": sum(c["gprime_resolution_pass"] for c in cases),
             "training_qualification_is_not_uniform_confidence_coverage": True,
             "model_is_frozen_before_validation": True}
    write_new(OUT / "MODEL.json", model)
    print(json.dumps({"model_created": True, "model_sha256": sha(OUT / "MODEL.json"), "gprime_SE": [model["gprime_SE_gate_passed"], model["gprime_SE_gate_count"]],
                      "g_resolution": [model["g_resolution_gates_passed"], len(NS)],
                      "gprime_resolution": [model["gprime_resolution_gates_passed"], len(NS)]}))


def numeric_spline_minimum(derivatives):
    spline, anti, shift = spline_penalty(derivatives)
    if np.all(np.asarray(derivatives) == 0):
        return {"rho": 0.0, "flat_entire_interval": True, "candidates": [], "tied_candidates": [-R, 0.0, R]}
    candidates = [-R, R]
    for value in spline.roots(extrapolate=False):
        if np.isfinite(value) and -R-1e-12 <= value <= R+1e-12:
            value = float(np.clip(value, -R, R))
            if all(abs(value-old) > 1e-10 for old in candidates):
                candidates.append(value)
    candidates.sort()
    values = [float(anti(rho)-shift) for rho in candidates]
    minimum = min(values)
    tied = [rho for rho, value in zip(candidates, values) if value-minimum <= 1e-12]
    selected = min(tied, key=lambda rho: (abs(rho), rho))
    return {"rho": selected, "flat_entire_interval": False,
            "candidates": [{"rho": rho, "potential": value} for rho, value in zip(candidates, values)],
            "tied_candidates": tied, "selected_derivative": float(spline(selected)),
            "scope": "Numerical minimum of estimated derivative-spline potential, not exact finite population/empirical optimum"}


def observed_kl(t, x):
    if t == x:
        return 0.0
    value = ((1+t)*math.log1p((t-x)/(1+x))+(1-t)*math.log1p((x-t)/(1-x)))/2
    require(value >= -1e-14, "Negative KL exceeds rounding allowance")
    return max(0.0, value)


def summarize_validation(validation, ref):
    p, model = protocol(), read(OUT / "MODEL.json")
    require(validation["model_sha256_at_start"] == sha(OUT / "MODEL.json"), "Validation model binding")
    r101 = np.array(model["reference"]["r801"])[::8]
    modes = np.array([record["D_modes_mean"] for record in validation["records"]])
    projected = project_modes(modes, r101)
    averages = {key: mean_se(value) for key, value in projected.items()}
    moment = ReferenceIntegrator()
    moment.load_table(ref)
    cases = []
    for ni, trained in enumerate(model["cases"]):
        u, u_se = averages["U_prime"][0][ni], averages["U_prime"][1][ni]
        v, v_se = averages["V_prime"][0][ni], averages["V_prime"][1][ni]
        require(np.all(v < 0), "Nonnegative validation V_prime; retain run files and stop summary")
        penalties = {"raw": np.zeros_like(r101), "finite": np.array(trained["g_prime"]),
                     "asymptotic": 2*A*trained["w"]*r101,
                     "quadratic": 2*trained["w"]*A*model["reference"]["d"]*RHO101}
        residuals = {}
        for method, penalty in penalties.items():
            per_stream = penalty-projected["g_prime"][:, ni]
            mean, conditional_se = mean_se(per_stream)
            training_se = np.array(trained["calibration_SE"]["g_prime"]) if method == "finite" else np.zeros_like(mean)
            per_ratio = per_stream/(-projected["V_prime"][:, ni])
            ratio_mean, ratio_se = mean_se(per_ratio)
            residuals[method] = {"mean": mean.tolist(), "validation_conditional_SE": conditional_se.tolist(),
                                 "combined_train_validation_SE": np.sqrt(conditional_se**2+training_se**2).tolist(),
                                 "per_validation_stream": per_stream.tolist(),
                                 "delta_over_minus_Vprime_ratio_of_means": (mean/(-v)).tolist(),
                                 "implied_signed_moment_error_at_same_rho": (-mean/(-v)).tolist(),
                                 "per_stream_ratio_mean": ratio_mean.tolist(), "per_stream_ratio_SE": ratio_se.tolist()}
        optima = []
        for theta in TEACHERS:
            teacher_r, _ = moment.one(theta, 64)
            for method, penalty in penalties.items():
                selected = numeric_spline_minimum(u+teacher_r*v+penalty)
                rr64, _ = moment.one(selected["rho"], 64)
                rr96, _ = moment.one(selected["rho"], 96)
                selected.update({"theta": theta, "method": method, "r_teacher": teacher_r, "r_selected": rr64,
                                 "reference_joint_KL": observed_kl(teacher_r, rr64),
                                 "selected_r_GH64_vs96": abs(rr64-rr96),
                                 "selected_r_qualification_pass": abs(rr64-rr96) <= p["qualification"]["r64_vs96_absolute"]})
                selected["per_validation_stream_rhos"] = [numeric_spline_minimum(projected["U_prime"][si, ni]+teacher_r*projected["V_prime"][si, ni]+penalty)["rho"] for si in range(STREAMS["validation"])]
                optima.append(selected)
        key_points = []
        for rho in TEACHERS:
            index = int(np.argmin(abs(RHO101-rho)))
            require(abs(RHO101[index]-rho) < 1e-12, "Diagnostic point is not on frozen grid")
            key_points.append({"rho": rho, "minus_V_prime": float(-v[index]), "minus_V_prime_SE": float(v_se[index]),
                               "residuals": {name: {key: values[index] for key, values in row.items() if key != "per_validation_stream"} for name, row in residuals.items()}})
        cases.append({"n": trained["n"], "K": trained["K"], "w": trained["w"], "rho101": RHO101.tolist(),
                      "U_prime": u.tolist(), "V_prime": v.tolist(), "U_prime_SE": u_se.tolist(), "V_prime_SE": v_se.tolist(),
                      "minus_V_prime": (-v).tolist(), "residuals": residuals, "key_points": key_points, "spline_optima": optima})
    output = {"source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL), "model_sha256": sha(OUT / "MODEL.json"),
              "validation_streams_sha256": sha(OUT / "VALIDATION_STREAMS.json"), "reference_sha256": sha(OUT / "REFERENCE.json"),
              "cases": cases, "validation_streams": 4,
              "accounting": {"additional_2d_reference_scalar_integrals": moment.counts["two_dimensional_scalar_integrals"],
                             "additional_shared_reference_node_entries": moment.counts["shared_tensor_node_entries"],
                             "aggregate_derivative_spline_selections": len(NS)*len(TEACHERS)*4,
                             "per_validation_stream_derivative_spline_selections": len(NS)*len(TEACHERS)*4*STREAMS["validation"]},
              "validation_is_descriptive_no_uniform_coverage": True,
              "ratio_of_means_is_not_a_bias_certificate": True,
              "no_teacher_specific_penalty_or_postvalidation_refit": True}
    write_new(OUT / "VALIDATION.json", output)
    print(json.dumps({"validation_complete": True, "cases": len(cases), "accounting": output["accounting"]}))


def report():
    p, model, validation = protocol(), read(OUT / "MODEL.json"), read(OUT / "VALIDATION.json")
    require(validation["model_sha256"] == sha(OUT / "MODEL.json"), "Report model/validation mismatch")
    dest = ROOT / p["report"]
    require(not dest.exists(), "Calibration report already exists")
    lines = ["# 有限预算统一校准与独立验证", "",
             "本包用已知发射模型和 prior 模拟学习一个对所有教师共用的加性校准项。没有采集观测标签或训练观测数据；canonical 无关符号只用于总体／条件 bank 期望校准。", "",
             "训练16条独立流，验证另4条独立流，每流2048个bank；n62/K64与n254/K256复用最大Gaussian前缀。四相关块模式及所有rho共用同一bank。SE以流为独立单位，不能将rho、模式或前缀当额外重复。", "",
             "## 函数与资格", "",
             "部署g是101节点g′三次样条的精确反导并减去g(0)，没有另拟合g值样条。g_second_sup由所保存二进制样条系数按有理算术逐段求得并向上取整；它只保证这个已保存函数的二阶导数界，不保证高维模型积分误差。", "",
             f"训练g′的SE资格通过 {model['gprime_SE_gate_passed']}/{model['gprime_SE_gate_count']}；在1001个预定点上，51/101节点g差资格通过 {model['g_resolution_gates_passed']}/{len(NS)}，g′差资格通过 {model['gprime_resolution_gates_passed']}/{len(NS)}，阈值分别为2e−6和1e−5。任何未通过项均保留，不追加预算。", "",
             "|n/K|g51与101预定点最大差|g′51与101预定点最大差|保存函数g″上界|", "|---|---:|---:|---:|"]
    for case in model["cases"]:
        q = case["resolution_51_vs101"]
        lines.append(f"|{case['n']}/{case['K']}|{q['g_max_abs_difference']:.7g}|{q['gprime_max_abs_difference']:.7g}|{case['g_second_sup']:.7g}|")
    lines += ["", "r/r′参考使用801点GH64/96；H为r样条反导。节点对照和分段多项式极值搜索仅为数值资格，不是严格Gaussian全积分证书。", "",
              "## 独立验证", "",
              "有限校准在验证前已冻结。残差delta=所用g′−验证估计的精确有限g′，同时保存验证4流条件SE及有限校准训练／验证合成SE。delta/(−V′)仅为同rho处的归一化描述；相应驻点观察矩误差的符号与它相反，不能把ratio-of-means直接当作严格偏差证书。", "",
              "下表rho*是独立验证导数定义的样条势最小点，不是真实有限期望或任何观测训练目标的连续全局最优证书。", "",
              "|n/K|theta|规则|样条rho*|观察KL参考|", "|---|---:|---|---:|---:|"]
    for case in validation["cases"]:
        for row in case["spline_optima"]:
            lines.append(f"|{case['n']}/{case['K']}|{row['theta']:.3f}|{row['method']}|{row['rho']:.8f}|{row['reference_joint_KL']:.7g}|")
    lines += ["", "验证只有4条独立流，结果为描述性检查，没有严格全域置信覆盖，也没有根据教师或验证结果重调校准项。", "",
              "## 账目", "", "```json", json.dumps({"train": read(OUT / "TRAIN.json")["ledger"],
                      "validation": read(OUT / "VALIDATION_STREAMS.json")["ledger"],
                      "reference": read(OUT / "REFERENCE.json")["accounting"],
                      "validation_reference_and_spline": validation["accounting"]}, ensure_ascii=False, indent=2), "```", "",
              f"模型 SHA256：`{sha(OUT / 'MODEL.json')}`。协议 SHA256：`{sha(PROTOCOL)}`。所有逐流数组、残差、系数、平局候选及资格明细保存在 `artifacts/finite_calibration/CAL/`。", ""]
    dest.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"report": rel(dest), "sha256": sha(dest)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "reference", "train", "validate", "report"))
    parser.add_argument("--plan", help="Required only for freeze; report goes beside this PLAN.md")
    args = parser.parse_args()
    if args.phase == "freeze":
        require(args.plan is not None, "freeze requires --plan")
        freeze(args.plan)
    elif args.phase == "reference":
        reference()
    elif args.phase == "train":
        run_streams("train")
    elif args.phase == "validate":
        run_streams("validation")
    else:
        report()
