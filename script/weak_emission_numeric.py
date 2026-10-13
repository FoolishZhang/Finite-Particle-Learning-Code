"""Deterministic d=2 finite-particle population scoring; no RNG or fitting.

The scientific CLI --run path requires a freeze. The public engine is also
usable by auditors, who must enforce their own authorization/freeze contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logsumexp, roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/弱发射展开与总体偏好_20261001"
OUT = ROOT / "artifacts/weak_emission/NUMERIC"
KS = (1, 2, 3, 4, 8)
VARIANTS = {"low": (32, 256), "primary": (48, 384), "high": (64, 512)}
CONFIG = {
    "schema": 1, "d": 2, "theta": 0.15,
    "emissions": [0.05, 0.1, 0.2, 0.4, 1.0], "ks": list(KS),
    "rho_box": [-0.75, 0.75], "rho_grid": [3 * (i - 10) / 40 for i in range(21)],
    "variants": {k: list(v) for k, v in VARIANTS.items()}, "primary": "primary",
    "integration_variable": "log_u; t=K*u",
    "log_u_bounds": [-30.0 - math.log(8), 24.0], "node_chunk": 32,
    "brent_xtol": 1e-10, "brent_rtol": 1e-12, "brent_maxiter": 100,
    "correction": {"Ks": [3, 4, 8], "formula": "rho_corrected=(K-2)*rho_raw/K",
                   "moment_tolerance": 1e-12, "new_prediction_gate": False},
    "fd_steps": [0.01, 0.005],
    "gates": {"position": 1e-5, "value_scaled": 1e-7, "value_floor": 5e-13,
              "gradient_scaled": 1e-6, "gradient_floor": 1e-13,
              "risk_scaled": 1e-7, "risk_floor": 0.0,
              "K1_value": 5e-11, "K1_gradient": 1e-11,
              "prediction_position": 0.005, "prediction_risk_relative": 0.05},
    "new_random_draws": 0, "new_observed_labels": 0, "optimizer_fits": 0,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def exp_difference(log_a, log_b):
    """exp(log_a)-exp(log_b), without near-equality cancellation."""
    log_a, log_b = np.broadcast_arrays(log_a, log_b)
    positive = log_a >= log_b
    hi = np.maximum(log_a, log_b)
    lo = np.minimum(log_a, log_b)
    magnitude = np.exp(hi) * (-np.expm1(lo - hi))
    return np.where(positive, magnitude, -magnitude)


def binary_kl_from_spin(r_p, r_q):
    """KL of the same/different event, stably near equal probabilities."""
    q = (1.0 + float(r_q)) / 2.0
    delta = (float(r_p) - float(r_q)) / 2.0

    def remainder(x):
        if abs(x) < 0.01:
            return sum((-1.0) ** n * x ** n / (n * (n - 1)) for n in range(2, 14))
        return (1.0 + x) * math.log1p(x) - x

    return q * remainder(delta / q) + (1.0 - q) * remainder(-delta / (1.0 - q))


def gaussian_nodes(order):
    z, weight = roots_hermitenorm(int(order))
    weight = weight / np.sum(weight)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    w = np.multiply.outer(weight, weight).ravel()
    w /= w.sum()
    return z, weight, z1.ravel(), z2.ravel(), w


def emission_arrays(a, rho, z1, z2):
    if not (-1 < rho < 1):
        raise ValueError("Correlation must be strictly between -1 and 1.")
    root = math.sqrt(1.0 - rho * rho)
    x2 = rho * z1 + root * z2
    log_w = log_expit(a * z1) + log_expit(a * x2)
    w = np.exp(log_w)
    q1, q2 = expit(-a * z1), expit(-a * x2)
    # d log W / d rho for this finite Cholesky-node approximation.
    dlog_w = a * q2 * (z1 - rho / root * z2)
    spin_product = np.tanh(a * z1 / 2) * np.tanh(a * x2 / 2)
    return w, dlog_w, q1 * q2, spin_product


def transform_values(u, w, dlog_w, q_product, weights, a):
    """Finite weighted-node M, path derivative, and Price integrand.

    Price and path need not agree for a finite non-Gaussian node measure.
    """
    u = np.asarray(u, dtype=float)
    v = u[:, None] * w[None, :]
    ev = np.exp(-v)
    log_m = np.empty(len(u))
    small = u <= 1.0
    if np.any(small):
        dm = np.expm1(-v[small]) @ weights
        log_m[small] = np.log1p(dm)
    if np.any(~small):
        log_m[~small] = logsumexp(-v[~small] + np.log(weights)[None, :], axis=1)
    path = -(ev * v) @ (weights * dlog_w)
    price = a * a * (ev * v * (v - 1.0)) @ (weights * q_product)
    return log_m, path, price


def negative_moment(a, rho, p=2):
    return sum(math.comb(p, i) * math.comb(p, j)
               * math.exp(a * a * (i * i + j * j + 2 * rho * i * j) / 2)
               for i in range(p + 1) for j in range(p + 1))


def analytic_tail_bounds(a, rho_abs=0.75, ks=KS):
    """Rigorous analytic bounds for true population truncation, not quadrature."""
    lo, hi = CONFIG["log_u_bounds"]
    moment = negative_moment(a, rho_abs, 2)
    cp = 27 * math.exp(-3) + 8 * math.exp(-2)
    rows = []
    for k in ks:
        eps, upper = k * math.exp(lo), k * math.exp(hi)
        rows.append({"K": k, "t_lower": eps, "t_upper": upper,
                     "value_lower": eps, "value_upper": moment / upper ** 2,
                     "derivative_lower": a * a / 16 * (eps + eps * eps / (2 * k)),
                     "derivative_upper": a * a * cp * moment / upper ** 2})
    return {"negative_moment_bound": moment, "p": 2, "rows": rows}


class LaplaceEngine:
    def __init__(self, variant="primary", theta=0.15):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.theta = variant, float(theta)
        self.gh_order, self.gl_order = VARIANTS[variant]
        self.z, self.w1, self.z1, self.z2, self.weights = gaussian_nodes(self.gh_order)
        x, weight = roots_legendre(self.gl_order)
        lo, hi = CONFIG["log_u_bounds"]
        self.u = np.exp((hi + lo) / 2 + (hi - lo) * x / 2)
        self.integration_weights = weight * (hi - lo) / 2
        self.native_cache, self.teacher_cache, self.cache, self.pair_cache = {}, {}, {}, {}
        self.ledger = {"native_rho_evaluations": 0, "laplace_exponential_terms": 0,
                       "teacher_moment_evaluations": 0, "teacher_quadrature_nodes": 0,
                       "evaluate_requests": 0, "native_requests": 0,
                       "gaussian_pair_kernel_nodes": 0, "outer_transform_nodes": 0,
                       "pair_only_requests": 0, "pair_only_evaluations": 0,
                       "pair_only_quadrature_nodes": 0}

    def native(self, a, rho):
        self.ledger["native_requests"] += 1
        key = (float(a).hex(), float(rho).hex())
        if key in self.native_cache:
            return self.native_cache[key]
        w, dlogw, qprod, spin = emission_arrays(a, rho, self.z1, self.z2)
        lm, path, price = [], [], []
        for first in range(0, len(self.u), CONFIG["node_chunk"]):
            result = transform_values(self.u[first:first + CONFIG["node_chunk"]],
                                      w, dlogw, qprod, self.weights, a)
            for dest, val in zip((lm, path, price), result):
                dest.extend(val.tolist())
        lm, path, price = np.asarray(lm), np.asarray(path), np.asarray(price)
        k = np.asarray(KS)[:, None]
        log_laplace = k * lm[None, :]
        integrands = exp_difference(log_laplace, -k * self.u[None, :])
        factor = k * np.exp((k - 1) * lm[None, :])
        iw = self.integration_weights.astype(np.longdouble)
        integrate = lambda x: np.asarray(np.sum(x.astype(np.longdouble) * iw, axis=1), dtype=float)
        result = {"log_laplace": log_laplace, "score_numeric": integrate(integrands),
                  "path_numeric": integrate(factor * path), "price_numeric": integrate(factor * price),
                  "spin_moment": float(spin @ self.weights)}
        self.native_cache[key] = result
        self.pair_cache[key] = result["spin_moment"]
        self.ledger["native_rho_evaluations"] += 1
        self.ledger["gaussian_pair_kernel_nodes"] += self.gh_order ** 2
        self.ledger["outer_transform_nodes"] += self.gl_order
        self.ledger["laplace_exponential_terms"] += self.gh_order ** 2 * self.gl_order
        return result

    def pair_moment(self, a, rho):
        """Pair-only Gaussian quadrature: no Laplace transform or localization."""
        self.ledger["pair_only_requests"] += 1
        key = (float(a).hex(), float(rho).hex())
        if key not in self.pair_cache:
            _, _, _, spin = emission_arrays(a, rho, self.z1, self.z2)
            self.pair_cache[key] = float(spin @ self.weights)
            self.ledger["pair_only_evaluations"] += 1
            self.ledger["pair_only_quadrature_nodes"] += self.gh_order ** 2
        return self.pair_cache[key]

    def teacher(self, a):
        key = float(a).hex()
        if key not in self.teacher_cache:
            _, _, _, spin = emission_arrays(a, self.theta, self.z1, self.z2)
            r = float(spin @ self.weights)
            flat = float(2 * np.sum(self.w1 * np.logaddexp(a * self.z / 2, -a * self.z / 2)))
            self.teacher_cache[key] = {"spin_moment": r, "p_same": (1 + r) / 2,
                                       "K1_flat": flat}
            self.ledger["teacher_moment_evaluations"] += 1
            self.ledger["teacher_quadrature_nodes"] += self.gh_order ** 2 + self.gh_order
        return self.teacher_cache[key]

    def evaluate(self, a, rho):
        self.ledger["evaluate_requests"] += 1
        key = (float(a).hex(), float(rho).hex())
        if key in self.cache:
            return self.cache[key]
        teacher = self.teacher(a)
        same, diff, zero = self.native(a, rho), self.native(a, -rho), self.native(a, 0.0)
        p = teacher["p_same"]
        numeric = p * same["score_numeric"] + (1 - p) * diff["score_numeric"]
        path = p * same["path_numeric"] - (1 - p) * diff["path_numeric"]
        price = p * same["price_numeric"] - (1 - p) * diff["price_numeric"]
        iw = self.integration_weights.astype(np.longdouble)
        delta_nodes = (p * exp_difference(same["log_laplace"], zero["log_laplace"])
                       + (1 - p) * exp_difference(diff["log_laplace"], zero["log_laplace"]))
        delta = np.asarray(np.sum(delta_nodes.astype(np.longdouble) * iw, axis=1), dtype=float)
        score = numeric.copy()
        # K=1 is analytically flat. Keep all numerical controls separately.
        score[0], delta[0] = teacher["K1_flat"], 0.0
        public_path, public_price = path.copy(), price.copy()
        public_path[0], public_price[0] = 0.0, 0.0
        result = {"a": float(a), "rho": float(rho), "variant": self.variant,
                  "K": list(KS), "score": score.tolist(), "score_minus_rho0": delta.tolist(),
                  "path_gradient": public_path.tolist(), "price_gradient": public_price.tolist(),
                  "numeric_score": numeric.tolist(), "numeric_path_gradient": path.tolist(),
                  "numeric_price_gradient": price.tolist(),
                  "teacher_spin_moment": teacher["spin_moment"],
                  "candidate_spin_moment": same["spin_moment"],
                  "joint_KL": binary_kl_from_spin(teacher["spin_moment"], same["spin_moment"]),
                  "K1_analytic": teacher["K1_flat"]}
        self.cache[key] = result
        return result


def locate(engine, a, k):
    index = KS.index(k)
    grid = CONFIG["rho_grid"]
    grads = [engine.evaluate(a, x)["path_gradient"][index] for x in grid]
    roots, brackets = [], []
    for x, g in zip(grid, grads):
        if g == 0:
            roots.append(x)
    for left, right, gl, gr in zip(grid[:-1], grid[1:], grads[:-1], grads[1:]):
        if gl * gr < 0:
            f = lambda rho: engine.evaluate(a, rho)["path_gradient"][index]
            root, result = brentq(f, left, right, xtol=CONFIG["brent_xtol"],
                                  rtol=CONFIG["brent_rtol"], maxiter=CONFIG["brent_maxiter"],
                                  full_output=True, disp=False)
            if not result.converged:
                raise ArithmeticError("Predeclared Brent budget exhausted.")
            roots.append(float(root))
            brackets.append({"left": left, "right": right, "root": float(root),
                             "iterations": int(result.iterations), "calls": int(result.function_calls)})
    candidates = sorted(set([grid[0], grid[-1], *roots]))
    values = [engine.evaluate(a, x)["score_minus_rho0"][index] for x in candidates]
    chosen = min(range(len(candidates)), key=lambda j: (values[j], candidates[j]))
    point = engine.evaluate(a, candidates[chosen])
    grid_values = [engine.evaluate(a, x)["score_minus_rho0"][index] for x in grid]
    grid_best = min(range(len(grid)), key=lambda j: (grid_values[j], grid[j]))
    near_tol = max(CONFIG["gates"]["value_floor"], a ** 4 * CONFIG["gates"]["value_scaled"])
    near_min = [x for x, f in zip(candidates, values) if f <= values[chosen] + near_tol]
    return {"a": a, "K": k, "variant": engine.variant, "rho": candidates[chosen],
            "score": point["score"][index], "score_minus_rho0": values[chosen],
            "gradient": point["path_gradient"][index], "joint_KL": point["joint_KL"],
            "root_brackets": brackets, "grid_gradients": grads,
            "roots": roots, "no_grid_derivative_sign_change": not bool(brackets),
            "candidate_rhos": candidates, "candidate_values": values,
            "grid_minimum_rho": grid[grid_best], "grid_minimum_value": grid_values[grid_best],
            "candidate_minus_grid_minimum": values[chosen] - grid_values[grid_best],
            "near_minimum_candidate_rhos": near_min, "near_minimum_value_tolerance": near_tol,
            "unique_numerical_candidate": len(near_min) == 1,
            "selected_kind": "endpoint" if chosen in (0, len(candidates) - 1)
            else ("root" if candidates[chosen] in roots else "grid_point"),
            "global_minimum_certificate": False}


def five_point_fd(engine, a, k, rho, h):
    index = KS.index(k)
    low, high = CONFIG["rho_box"]
    if low <= rho - 2 * h and rho + 2 * h <= high:
        shifts, coef, denom, scheme = [-2, -1, 1, 2], [1, -8, 8, -1], 12 * h, "central"
    else:
        direction = 1 if rho + 4 * h <= high else -1
        shifts = [direction * j for j in range(5)]
        coef, denom, scheme = [-25, 48, -36, 16, -3], direction * 12 * h, "one_sided"
    values = [engine.evaluate(a, rho + h * s)["score_minus_rho0"][index] for s in shifts]
    estimate = float(np.dot(np.asarray(coef, dtype=np.longdouble), values) / denom)
    return {"h": h, "scheme": scheme, "estimate": estimate, "rhos": [rho + h * s for s in shifts]}


def artificial_checks():
    """No Gaussian quadrature, scientific candidates, RNG, or optimization."""
    z1 = np.array([-1.1, -0.2, 0.4, 1.3])
    z2 = np.array([0.7, -0.8, 0.1, -0.3])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    a, rho = 0.7, 0.2
    w, dlogw, qprod, _ = emission_arrays(a, rho, z1, z2)
    u = np.array([1e-8, 0.2, 2.0, 20.0])
    lm, dp, _ = transform_values(u, w, dlogw, qprod, weights, a)
    h = 1e-5
    plus = emission_arrays(a, rho + h, z1, z2)[0]
    minus = emission_arrays(a, rho - h, z1, z2)[0]
    fd = ((np.exp(-u[:, None] * plus) @ weights)
          - (np.exp(-u[:, None] * minus) @ weights)) / (2 * h)
    error = float(np.max(np.abs(fd - dp)))
    nodes, glw = roots_legendre(256)
    lo, hi = CONFIG["log_u_bounds"]
    quad_u = np.exp((lo + hi) / 2 + nodes * (hi - lo) / 2)
    quad_w = glw * (hi - lo) / 2
    log_m = logsumexp(-quad_u[:, None] * w + np.log(weights), axis=1)
    laplace = float(np.sum(exp_difference(2 * log_m, -2 * quad_u) * quad_w))
    direct = float(np.sum(-np.log((w[:, None] + w[None, :]) / 2) * np.outer(weights, weights)))
    const_lm = -quad_u / 4
    const_values = [float(np.sum(exp_difference(k * const_lm, -k * quad_u) * quad_w)) for k in KS]
    assert error < 2e-10
    assert abs(laplace - direct) < 2e-11
    assert max(abs(x - math.log(4)) for x in const_values) < 2e-11
    assert binary_kl_from_spin(0.03, 0.03) == 0
    return {"passed": True, "finite_node_path_fd_error": error,
            "finite_node_K2_laplace_direct_error": abs(laplace - direct),
            "constant_W_scores": const_values, "constant_W_truth": math.log(4),
            "finite_node_count": 4, "synthetic_GL_order": 256,
            "Gaussian_integrals": 0, "scientific_parameter_evaluations": 0,
            "random_draws": 0, "optimizer_fits": 0,
            "Price_identity_not_asserted_for_artificial_nodes": True}


def freeze():
    if (OUT / "FREEZE.json").exists():
        raise FileExistsError("Refusing to overwrite a freeze.")
    inputs = [Path(__file__), DOC / "NUMERIC_DESIGN.md", DOC / "PLAN_NUMERIC.md", ROOT / "HANDOFF.md"]
    if any(not p.exists() for p in inputs):
        raise FileNotFoundError("Required source, design, PLAN_NUMERIC, or HANDOFF is missing.")
    unit = read(OUT / "UNIT_CHECK.json")
    assert unit["passed"] and unit["source_sha256"] == sha(__file__)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "CONFIG.json", CONFIG)
    files = []
    for i, p in enumerate(inputs):
        snapshot = OUT / "source_inputs" / f"{i:02d}_{p.name}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as f:
            f.write(p.read_bytes())
        files.append({"original": p.relative_to(ROOT).as_posix(),
                      "snapshot": snapshot.relative_to(ROOT).as_posix(), "sha256": sha(snapshot)})
    save(OUT / "FREEZE.json", {"created": utc(), "inputs": files,
                                "config_sha256": sha(OUT / "CONFIG.json"),
                                "unit_check_sha256": sha(OUT / "UNIT_CHECK.json"),
                                "scientific_evaluations_before_freeze_in_this_program": 0})


def verify_freeze(require_live=False):
    receipt = read(OUT / "FREEZE.json")
    assert sha(OUT / "CONFIG.json") == receipt["config_sha256"]
    assert read(OUT / "CONFIG.json") == CONFIG
    assert sha(OUT / "UNIT_CHECK.json") == receipt["unit_check_sha256"]
    for item in receipt["inputs"]:
        assert sha(ROOT / item["snapshot"]) == item["sha256"]
        if require_live:
            assert sha(ROOT / item["original"]) == item["sha256"]
    return receipt


def make_checks(engines, locations, grid_rows):
    gates = CONFIG["gates"]
    records = []

    def add(name, error, limit, **meta):
        records.append({"name": name, "error": float(error), "limit": float(limit),
                        "passed": bool(error <= limit), **meta})

    for a in CONFIG["emissions"]:
        all_locations = [x for x in locations if x["a"] == a]
        # Evaluate all variants at identical selected points as well as the fixed grid.
        points = sorted(set(CONFIG["rho_grid"] + [x["rho"] for x in all_locations]))
        for rho in points:
            rows = [engines[v].evaluate(a, rho) for v in VARIANTS]
            for key, scaled, floor in [("score_minus_rho0", "value_scaled", "value_floor"),
                                       ("path_gradient", "gradient_scaled", "gradient_floor")]:
                array = np.array([r[key] for r in rows])
                for j, k in enumerate(KS):
                    add("variant_" + key, float(np.ptp(array[:, j])),
                        max(gates[scaled] * a ** 4, gates[floor]), a=a, rho=rho, K=k)
            add("variant_joint_KL", float(np.ptp([r["joint_KL"] for r in rows])),
                max(gates["risk_scaled"] * a ** 4, gates["risk_floor"]), a=a, rho=rho)
            for row in rows:
                for j, k in enumerate(KS):
                    add("Price_path", abs(row["numeric_path_gradient"][j] - row["numeric_price_gradient"][j]),
                        max(gates["gradient_scaled"] * a ** 4, gates["gradient_floor"]),
                        a=a, rho=rho, K=k, variant=row["variant"])
                add("K1_flat_integral", abs(row["numeric_score"][0] - row["K1_analytic"]),
                    gates["K1_value"],
                    a=a, rho=rho, variant=row["variant"])
                add("K1_numeric_gradient", abs(row["numeric_path_gradient"][0]),
                    gates["K1_gradient"],
                    a=a, rho=rho, variant=row["variant"])
        for k in KS[1:]:
            locs = [x for x in locations if x["a"] == a and x["K"] == k]
            add("variant_location", float(np.ptp([x["rho"] for x in locs])), gates["position"], a=a, K=k)
            for loc in locs:
                add("located_candidate_vs_grid", max(0.0, loc["candidate_minus_grid_minimum"]),
                    max(gates["value_scaled"] * a ** 4, gates["value_floor"]),
                    a=a, K=k, variant=loc["variant"])
            primary = next(x for x in locs if x["variant"] == "primary")
            for h in CONFIG["fd_steps"]:
                fd = five_point_fd(engines["primary"], a, k, primary["rho"], h)
                add("primary_FD_path", abs(fd["estimate"] - primary["gradient"]),
                    max(gates["gradient_scaled"] * a ** 4, gates["gradient_floor"]),
                    a=a, K=k, rho=primary["rho"], finite_difference=fd)
    return records


def correction_rows(engines, primary_locations):
    rows, checks = [], []
    for loc in primary_locations:
        a, k = loc["a"], loc["K"]
        if k not in CONFIG["correction"]["Ks"]:
            continue
        corrected = (k - 2) * loc["rho"] / k
        variants = []
        for name, engine in engines.items():
            rt = engine.teacher(a)["spin_moment"]
            rq = engine.pair_moment(a, corrected)
            variants.append({"variant": name, "teacher_spin_moment": rt,
                             "candidate_spin_moment": rq, "joint_KL": binary_kl_from_spin(rt, rq)})
        primary = next(v for v in variants if v["variant"] == "primary")
        risk_error = float(np.ptp([v["joint_KL"] for v in variants]))
        moment_error = max(float(np.ptp([v[key] for v in variants]))
                           for key in ("teacher_spin_moment", "candidate_spin_moment"))
        for name, err, limit in [("corrected_pair_moment", moment_error, CONFIG["correction"]["moment_tolerance"]),
                                 ("corrected_joint_KL", risk_error, a ** 4 * CONFIG["gates"]["risk_scaled"])]:
            checks.append({"name": name, "error": err, "limit": limit,
                           "passed": bool(err <= limit), "a": a, "K": k})
        rows.append({"a": a, "K": k, "rho_raw": loc["rho"], "rho_corrected": corrected,
                     "feasible": CONFIG["rho_box"][0] <= corrected <= CONFIG["rho_box"][1],
                     "primary_joint_KL": primary["joint_KL"],
                     "primary_joint_KL_over_a8": primary["joint_KL"] / a ** 8,
                     "raw_joint_KL": loc["joint_KL"],
                     "corrected_over_raw_KL": primary["joint_KL"] / loc["joint_KL"] if loc["joint_KL"] > 0 else None,
                     "variants": variants, "descriptive_only_no_prediction_gate": True})
    return rows, checks


def run():
    verify_freeze(require_live=True)
    if (OUT / "STATE.json").exists():
        raise FileExistsError("A run already exists; no automatic restart or added orders.")
    started = time.perf_counter()
    save(OUT / "STATE.json", {"state": "running", "started": utc()})
    grid, locations, checks = [], [], []
    try:
        engines = {v: LaplaceEngine(v, CONFIG["theta"]) for v in VARIANTS}
        for a in CONFIG["emissions"]:
            for variant, engine in engines.items():
                for rho in CONFIG["rho_grid"]:
                    grid.append(engine.evaluate(a, rho))
                print(f"grid a={a} variant={variant}", flush=True)
        save(OUT / "GRID.json", {"rows": grid})
        for a in CONFIG["emissions"]:
            for variant, engine in engines.items():
                for k in KS[1:]:
                    locations.append(locate(engine, a, k))
                print(f"locations a={a} variant={variant}", flush=True)
        save(OUT / "LOCATIONS.json", {"rows": locations})
        checks = make_checks(engines, locations, grid)
        primary = [x for x in locations if x["variant"] == "primary"]
        corrected, corrected_checks = correction_rows(engines, primary)
        checks.extend(corrected_checks)
        save(OUT / "CORRECTION.json", {"rows": corrected, "new_Laplace_evaluations": 0,
                                       "new_localizations": 0, "new_prediction_gates": 0})
        save(OUT / "CHECKS.json", {"rows": checks, "all_passed": all(x["passed"] for x in checks)})
        prediction = []
        for loc in primary:
            if loc["a"] != CONFIG["emissions"][0]:
                continue
            k, rho = loc["K"], loc["rho"]
            leading = 0.75 if k == 2 else k * CONFIG["theta"] / (k - 2)
            coeff = (leading - CONFIG["theta"]) ** 2 / 32
            relative = abs(loc["joint_KL"] / loc["a"] ** 4 - coeff) / coeff
            position = (rho == CONFIG["rho_box"][1] and all(g < 0 for g in loc["grid_gradients"])) if k == 2 else abs(rho - leading) <= CONFIG["gates"]["prediction_position"]
            prediction.append({"K": k, "a": loc["a"], "rho": rho, "leading_rho": leading,
                               "location_or_pointwise_gradient_pass": bool(position),
                               "leading_KL_over_a4": coeff, "KL_relative_difference": relative,
                               "risk_prediction_pass": relative <= CONFIG["gates"]["prediction_risk_relative"],
                               "passed": bool(position and relative <= CONFIG["gates"]["prediction_risk_relative"])})
        numerical_pass = all(x["passed"] for x in checks)
        prediction_count = sum(int(x["location_or_pointwise_gradient_pass"])
                               + int(x["risk_prediction_pass"]) for x in prediction)
        ledger = {"primary_location_programs": 20, "all_variant_location_executions": len(locations),
                  "grid_rows": len(grid), "grid_scores_including_K1": len(grid) * len(KS),
                  "corrected_primary_points": len(corrected), "corrected_pair_variant_requests": 3 * len(corrected),
                  "new_labels": 0, "random_draws": 0, "new_fits": 0,
                  "quadrature_engines": {k: e.ledger for k, e in engines.items()},
                  "elapsed_seconds": time.perf_counter() - started,
                  "Python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
        save(OUT / "RESULTS.json", {"numerical_qualified": numerical_pass,
                                    "prediction_gates": prediction, "primary_locations": primary,
                                    "prediction_tests_total": 8,
                                    "prediction_tests_passed": prediction_count,
                                    "K_joint_pass_total": 4,
                                    "K_joint_passed": sum(x["passed"] for x in prediction),
                                    "all_prediction_gates_pass": all(x["passed"] for x in prediction),
                                    "prediction_interpretable": numerical_pass,
                                    "tail_bounds": {str(a): analytic_tail_bounds(a) for a in CONFIG["emissions"]},
                                    "ledger": ledger,
                                    "claims": {"strict_quadrature_error_certificate": False,
                                               "continuous_global_optimum_certificate": False,
                                               "statistical_confidence_interval": False}})
        state = "complete" if numerical_pass else "numerical_failed"
        save(OUT / "STATE.json", {"state": state, "finished": utc(),
                                   "no_extra_orders_or_candidates": True}, exclusive=False)
        files = ["CONFIG.json", "FREEZE.json", "GRID.json", "LOCATIONS.json", "CORRECTION.json", "CHECKS.json", "RESULTS.json", "STATE.json"]
        save(OUT / "RESULTS_FREEZE.json", {"created": utc(), "files": {x: sha(OUT / x) for x in files}})
        print(json.dumps({"state": state, "qualification_failures": sum(not x["passed"] for x in checks),
                          "prediction_tests_passed": prediction_count, "prediction_tests_total": 8,
                          "K_joint_passed": sum(x["passed"] for x in prediction), "ledger": ledger}, ensure_ascii=False), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"time": utc(), "type": type(exc).__name__, "message": str(exc),
                                    "grid_rows_completed": len(grid), "locator_executions_completed": len(locations),
                                    "extra_budget_used": False})
        save(OUT / "STATE.json", {"state": "execution_failed", "finished": utc()}, exclusive=False)
        raise


def check():
    verify_freeze()
    receipt = read(OUT / "RESULTS_FREEZE.json")
    for name, expected in receipt["files"].items():
        assert sha(OUT / name) == expected, name
    grid, locs = read(OUT / "GRID.json")["rows"], read(OUT / "LOCATIONS.json")["rows"]
    results, checks = read(OUT / "RESULTS.json"), read(OUT / "CHECKS.json")["rows"]
    assert len(grid) == 5 * 21 * 3
    assert len(locs) == 5 * 4 * 3
    for row in checks:
        assert row["passed"] == (row["error"] <= row["limit"])
    assert results["numerical_qualified"] == all(x["passed"] for x in checks)
    assert len(results["primary_locations"]) == 20
    assert results["prediction_tests_total"] == 8
    assert results["prediction_tests_passed"] == sum(int(x["location_or_pointwise_gradient_pass"])
                                                     + int(x["risk_prediction_pass"])
                                                     for x in results["prediction_gates"])
    assert results["K_joint_pass_total"] == 4
    assert results["K_joint_passed"] == sum(x["passed"] for x in results["prediction_gates"])
    corrected = read(OUT / "CORRECTION.json")["rows"]
    assert len(corrected) == 15
    for row in corrected:
        assert row["rho_corrected"] == (row["K"] - 2) * row["rho_raw"] / row["K"]
        assert row["feasible"]
    assert results["ledger"]["new_labels"] == results["ledger"]["random_draws"] == results["ledger"]["new_fits"] == 0
    for item in locs:
        j = min(range(len(item["candidate_rhos"])), key=lambda i: (item["candidate_values"][i], item["candidate_rhos"][i]))
        assert item["rho"] == item["candidate_rhos"][j]
        assert item["score_minus_rho0"] == item["candidate_values"][j]
    print(json.dumps({"passed": True, "scope": "frozen bytes, saved qualification arithmetic, candidate selection, ledger; no reintegration", "state": read(OUT / "STATE.json")["state"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ("qa", "freeze", "run", "check"):
        group.add_argument("--" + action, action="store_true")
    args = parser.parse_args()
    if args.qa:
        result = artificial_checks()
        result["source_sha256"] = sha(__file__)
        result["created"] = utc()
        previous = OUT / "UNIT_CHECK.json"
        if (OUT / "FREEZE.json").exists():
            raise FileExistsError("Frozen QA cannot be replaced.")
        result["completed_QA_execution_index"] = 1
        if previous.exists():
            old = read(previous)
            result["completed_QA_execution_index"] = old.get("completed_QA_execution_index", 1) + 1
            save(OUT / "QA_HISTORY" / f"unit_{sha(previous)}.json", old)
        save(previous, result, exclusive=not previous.exists())
        print(json.dumps(result))
    elif args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        check()


if __name__ == "__main__":
    main()
