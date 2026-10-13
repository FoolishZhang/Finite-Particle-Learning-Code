"""K=2 deterministic Fourier population scoring with independent nuisance bits.

No source in an older frozen package is imported or edited. CLI science is
allowed only after a new freeze. Public pure APIs require the caller to respect
the same experiment authorization contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import scipy
from scipy.special import expit, log_expit, roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001"
OUT = ROOT / "artifacts/nuisance_dimensions/NUMERIC"
VARIANTS = {"low": (64, 64), "primary": (96, 96), "high": (128, 128)}
PANELS = [0.0, 0.125, 0.5, 2.0, 6.0, 14.0]
RHO_GRID = [3 * (i - 10) / 40 for i in range(21)]
CONFIG = {
    "schema": 1, "K": 2, "primary": "primary",
    "variants": {k: list(v) for k, v in VARIANTS.items()},
    "frequency_panels": PANELS, "rho_grid": RHO_GRID, "rho_box": [-0.75, 0.75],
    "cases": [{"a": 1.0, "theta": theta, "n": n}
              for theta in (0.0, 0.15) for n in (0, 1, 2, 14, 62, 254, 1022, 4094)]
             + [{"a": 0.05, "theta": 0.0, "n": n} for n in (0, 1, 2)],
    "node_chunk": 32, "new_labels": 0, "random_draws": 0,
    "finite_data_fits": 0, "locator_executions": 0,
    "fd_rhos": [-0.75, 0.0, 0.15, 0.75], "fd_steps": [0.01, 0.005],
    "gates": {"a1_value_scaled": 1e-8, "a1_gradient_scaled": 1e-8,
              "weak_value": 5e-14, "weak_gradient": 5e-13,
              "pair": 1e-12, "a1_KL": 1e-10, "weak_KL": 1e-15,
              "mean_drift": 1e-11, "CF_roundoff": 1e-12,
              "a1_FD_scaled": 1e-7, "weak_FD": 2e-12,
              "prediction_relative": 0.10},
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


def canonical_float(value):
    value = float(value)
    return 0.0 if value == 0 else value


def binary_kl_from_spin(rp, rq):
    q, delta = (1 + float(rq)) / 2, (float(rp) - float(rq)) / 2

    def rem(x):
        if abs(x) < 0.01:
            return sum((-1.0) ** j * x ** j / (j * (j - 1)) for j in range(2, 14))
        return (1 + x) * math.log1p(x) - x

    return q * rem(delta / q) + (1 - q) * rem(-delta / (1 - q))


def tail_bounds(a, cutoff=14.0, rho_max=0.75):
    e = math.exp(-math.pi * cutoff)
    denom = -math.expm1(-2 * math.pi * cutoff)
    moment = a * math.sqrt(2 / math.pi) / math.sqrt(1 - rho_max * rho_max)
    return {"frequency_cutoff": cutoff, "uniform_in_n": True,
            "value": 2 * e / (math.pi * cutoff * denom),
            "Price_gradient": 4 * a * a * e * (cutoff / math.pi + 1 / math.pi ** 2) / denom,
            "path_gradient": 4 * moment * e / (math.pi * denom),
            "bounds_cover_truncation_only": True}


def finite_phase_arrays(t, centered, weights, q_product=None, a_prime=None):
    """Finite positive-weight characteristic function; no distribution assumed."""
    phase = t[:, None] * centered[None, :]
    cos_minus_one = -2 * np.sin(phase / 2) ** 2
    sin_phase = np.sin(phase)
    out = {"h_real_increment": cos_minus_one @ weights, "h_imag": sin_phase @ weights}
    if q_product is not None:
        weighted_q = weights * q_product
        out["b_real"] = float(np.sum(weighted_q)) + cos_minus_one @ weighted_q
        out["b_imag"] = sin_phase @ weighted_q
    if a_prime is not None:
        weighted_prime = weights * a_prime
        out["A_prime_cos"] = float(np.sum(weighted_prime)) + cos_minus_one @ weighted_prime
        out["A_prime_sin"] = sin_phase @ weighted_prime
    return out


class FourierEngine:
    def __init__(self, variant="primary"):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.gh_order, self.gl_order = VARIANTS[variant]
        self.z, self.weights1 = roots_hermitenorm(self.gh_order)
        self.weights1 /= self.weights1.sum()
        z1, z2 = np.meshgrid(self.z, self.z, indexing="ij")
        self.z1, self.z2 = z1.ravel(), z2.ravel()
        self.weights2 = np.multiply.outer(self.weights1, self.weights1).ravel()
        self.weights2 /= self.weights2.sum()
        nodes, weights = roots_legendre(self.gl_order)
        self.t = np.concatenate([(lo + hi) / 2 + (hi - lo) * nodes / 2 for lo, hi in zip(PANELS[:-1], PANELS[1:])])
        self.iw = np.concatenate([(hi - lo) * weights / 2 for lo, hi in zip(PANELS[:-1], PANELS[1:])])
        self.sinh = np.sinh(math.pi * self.t)
        self.native_cache, self.nuisance_cache, self.cache = {}, {}, {}
        self.ledger = {"evaluate_requests": 0, "unique_evaluate_rows": 0,
                       "native_requests": 0, "native_evaluations": 0,
                       "native_Gaussian_nodes": 0, "native_phase_node_terms": 0,
                       "nuisance_requests": 0, "nuisance_evaluations": 0,
                       "nuisance_Gaussian_nodes": 0, "nuisance_phase_node_terms": 0}

    def nuisance(self, a):
        a = canonical_float(a)
        self.ledger["nuisance_requests"] += 1
        key = a.hex()
        if key in self.nuisance_cache:
            return self.nuisance_cache[key]
        ell = float(np.sum(self.weights1 * np.logaddexp(a * self.z / 2, -a * self.z / 2)))
        u = log_expit(a * self.z)
        centered = u + ell
        phase_parts = []
        for start in range(0, len(self.t), CONFIG["node_chunk"]):
            phase_parts.append(finite_phase_arrays(self.t[start:start + CONFIG["node_chunk"]], centered, self.weights1))
        re = np.concatenate([p["h_real_increment"] for p in phase_parts])
        im = np.concatenate([p["h_imag"] for p in phase_parts])
        delta = 2 * re + re * re + im * im
        positive_count = int(np.sum(delta > 0))
        maximum_positive = float(max(0.0, np.max(delta)))
        if maximum_positive > CONFIG["gates"]["CF_roundoff"]:
            raise ArithmeticError("CF modulus violation exceeds the frozen roundoff allowance.")
        # The exact modulus is at most one. Record every roundoff repair before
        # exponentiating n, so a positive numerical error is never amplified.
        delta = np.minimum(delta, 0.0)
        log_modulus2 = np.empty_like(delta)
        near_one = delta > -0.5
        log_modulus2[near_one] = np.log1p(delta[near_one])
        log_modulus2[~near_one] = np.log((1 + re[~near_one]) ** 2 + im[~near_one] ** 2)
        mean_u = float(u @ self.weights1)
        result = {"a": a, "ell": ell, "mean_U": mean_u,
                  "variance_U": float(((u - mean_u) ** 2) @ self.weights1),
                  "centering_mean_residual": float(centered @ self.weights1),
                  "CF_positive_roundoff_count": positive_count,
                  "CF_maximum_positive_roundoff": maximum_positive,
                  "cf_real_increment": re, "cf_imag": im, "log_modulus_squared": log_modulus2}
        self.nuisance_cache[key] = result
        self.ledger["nuisance_evaluations"] += 1
        self.ledger["nuisance_Gaussian_nodes"] += self.gh_order
        self.ledger["nuisance_phase_node_terms"] += self.gh_order * len(self.t)
        return result

    def native(self, a, rho):
        a, rho = canonical_float(a), canonical_float(rho)
        if not (-1 < rho < 1):
            raise ValueError("Correlation must be strictly within (-1,1).")
        self.ledger["native_requests"] += 1
        key = (a.hex(), rho.hex())
        if key in self.native_cache:
            return self.native_cache[key]
        nuisance = self.nuisance(a)
        root = math.sqrt(1 - rho * rho)
        x2 = rho * self.z1 + root * self.z2
        centered = log_expit(a * self.z1) + log_expit(a * x2) + 2 * nuisance["ell"]
        q1, q2 = expit(-a * self.z1), expit(-a * x2)
        q_product = q1 * q2
        a_prime = a * q2 * (self.z1 - rho / root * self.z2)
        phase_parts = []
        for start in range(0, len(self.t), CONFIG["node_chunk"]):
            phase_parts.append(finite_phase_arrays(self.t[start:start + CONFIG["node_chunk"]],
                                                  centered, self.weights2, q_product, a_prime))
        result = {name: np.concatenate([x[name] for x in phase_parts]) for name in phase_parts[0]}
        mean_a = float(centered @ self.weights2)
        mean_ap = float(a_prime @ self.weights2)
        result.update({"a": a, "rho": rho, "centered_A_mean": mean_a,
                       "variance_A": float(((centered - mean_a) ** 2) @ self.weights2),
                       "A_prime_mean": mean_ap,
                       "covariance_A_A_prime": float(((centered - mean_a) * (a_prime - mean_ap)) @ self.weights2),
                       "q_product_mean": float(q_product @ self.weights2),
                       "spin_moment": float((np.tanh(a * self.z1 / 2) * np.tanh(a * x2 / 2)) @ self.weights2)})
        self.native_cache[key] = result
        self.ledger["native_evaluations"] += 1
        self.ledger["native_Gaussian_nodes"] += self.gh_order ** 2
        self.ledger["native_phase_node_terms"] += self.gh_order ** 2 * len(self.t)
        return result

    @staticmethod
    def modulus_difference(native, zero):
        re, im = native["h_real_increment"], native["h_imag"]
        r0, i0 = zero["h_real_increment"], zero["h_imag"]
        return 2 * (re - r0) + (re * re + im * im) - (r0 * r0 + i0 * i0)

    @staticmethod
    def price_mark(native):
        return native["b_real"] * (1 + native["h_real_increment"]) + native["b_imag"] * native["h_imag"]

    @staticmethod
    def path_mark(native):
        return (-native["A_prime_sin"] * (1 + native["h_real_increment"])
                + native["A_prime_cos"] * native["h_imag"])

    def evaluate(self, a, theta, n, rho):
        a, theta, rho = map(canonical_float, (a, theta, rho))
        if int(n) != n or n < 0:
            raise ValueError("n must be a nonnegative integer.")
        n = int(n)
        self.ledger["evaluate_requests"] += 1
        key = (a.hex(), theta.hex(), n, rho.hex())
        if key in self.cache:
            return self.cache[key]
        nuisance = self.nuisance(a)
        same, diff, zero = self.native(a, rho), self.native(a, -rho), self.native(a, 0.0)
        rt = 0.0 if theta == 0 else self.native(a, theta)["spin_moment"]
        rq = 0.0 if rho == 0 else same["spin_moment"]
        ps, pd = (1 + rt) / 2, (1 - rt) / 2
        weight_n = np.ones_like(self.t) if n == 0 else np.exp(n * nuisance["log_modulus_squared"])
        numerator = ps * self.modulus_difference(same, zero) + pd * self.modulus_difference(diff, zero)
        value_integrand = weight_n * numerator / (self.t * self.sinh)
        price_integrand = -2 * a * a * self.t / self.sinh * weight_n * (ps * self.price_mark(same) - pd * self.price_mark(diff))
        path_integrand = 2 / self.sinh * weight_n * (ps * self.path_mark(same) - pd * self.path_mark(diff))
        integrate = lambda y: float(np.sum(y.astype(np.longdouble) * self.iw.astype(np.longdouble)))
        price, path = integrate(price_integrand), integrate(path_integrand)
        delta_value = integrate(value_integrand)
        mean_shift = (ps * same["centered_A_mean"] + pd * diff["centered_A_mean"]
                      - zero["centered_A_mean"])
        mean_derivative = ps * same["A_prime_mean"] - pd * diff["A_prime_mean"]
        asymptotic = -a * a * (rt + rq) / (4 * math.sqrt(math.pi * nuisance["variance_U"]))
        result = {"variant": self.variant, "a": a, "theta": theta, "n": n, "D": n + 2,
                  "rho": rho, "delta_F": delta_value,
                  "Price_gradient": price, "path_gradient": path,
                  "active_mean_drift_same": same["centered_A_mean"],
                  "active_mean_drift_diff": diff["centered_A_mean"],
                  "active_mean_drift_zero": zero["centered_A_mean"],
                  "discrete_active_mean_shift": mean_shift,
                  "discrete_active_mean_derivative": mean_derivative,
                  "same_node_raw_delta_F": delta_value - mean_shift,
                  "same_node_raw_path_gradient": path - mean_derivative,
                  "teacher_spin_moment": rt, "candidate_spin_moment": rq,
                  "joint_KL": binary_kl_from_spin(rt, rq),
                  "variance_U": nuisance["variance_U"],
                  "sqrt_n_Price_gradient": math.sqrt(n) * price if n else None,
                  "sqrt_n_path_gradient": math.sqrt(n) * path if n else None,
                  "asymptotic_scaled_gradient": asymptotic,
                  "value_integrand_at_zero": -(ps * (same["variance_A"] - zero["variance_A"])
                                                + pd * (diff["variance_A"] - zero["variance_A"])) / math.pi,
                  "Price_integrand_at_zero": -2 * a * a * (ps * same["q_product_mean"] - pd * diff["q_product_mean"]) / math.pi,
                  "path_integrand_at_zero": -2 * (ps * same["covariance_A_A_prime"] - pd * diff["covariance_A_A_prime"]) / math.pi}
        self.cache[key] = result
        self.ledger["unique_evaluate_rows"] += 1
        return result


def export_cache(engines, out_dir):
    """Save ALL already computed evaluation/native/CF values, no new calls."""
    arrays, native_rows, nuisance_rows = {}, [], []
    for variant, engine in engines.items():
        arrays[variant + "_frequency"] = engine.t
        arrays[variant + "_integration_weights"] = engine.iw
        arrays[variant + "_Gaussian_nodes"] = engine.z
        arrays[variant + "_Gaussian_weights"] = engine.weights1
        for i, val in enumerate(engine.native_cache.values()):
            scalars, mapping = {}, {}
            for name, item in val.items():
                if isinstance(item, np.ndarray):
                    key = f"{variant}_native_{i:04d}_{name}"
                    arrays[key], mapping[name] = item, key
                else:
                    scalars[name] = item
            native_rows.append({"variant": variant, **scalars, "array_keys": mapping})
        for i, val in enumerate(engine.nuisance_cache.values()):
            scalars, mapping = {}, {}
            for name, item in val.items():
                if isinstance(item, np.ndarray):
                    key = f"{variant}_nuisance_{i:04d}_{name}"
                    arrays[key], mapping[name] = item, key
                else:
                    scalars[name] = item
            nuisance_rows.append({"variant": variant, **scalars, "array_keys": mapping})
    target = Path(out_dir) / "CACHE.npz"
    if target.exists():
        raise FileExistsError(target)
    np.savez_compressed(target, **arrays)
    save(Path(out_dir) / "CACHE_INDEX.json", {"local_optional_numeric_cache": True,
                                             "Git_backup_of_npz_not_claimed": True, "npz_sha256": sha(target),
                                             "native_rows": native_rows, "nuisance_rows": nuisance_rows,
                                             "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
                                             "raw_phase_tensors_not_stored": True})
    rows = [row for engine in engines.values() for row in engine.cache.values()]
    save(Path(out_dir) / "EVALUATIONS.json", {"rows": rows, "count": len(rows),
                                             "unique_key": ["variant", "a", "theta", "n", "rho"]})


def five_point_fd(engine, a, theta, n, rho, step):
    if -0.75 <= rho - 2 * step and rho + 2 * step <= 0.75:
        offsets, coefficients, denominator, scheme = [-2, -1, 1, 2], [1, -8, 8, -1], 12 * step, "central"
    else:
        direction = 1 if rho + 4 * step <= 0.75 else -1
        offsets, coefficients = [direction * i for i in range(5)], [-25, 48, -36, 16, -3]
        denominator, scheme = direction * 12 * step, "one_sided"
    points = [rho + step * i for i in offsets]
    values = [engine.evaluate(a, theta, n, x)["delta_F"] for x in points]
    derivative = float(np.dot(np.asarray(coefficients, dtype=np.longdouble), values) / denominator)
    return {"step": step, "scheme": scheme, "rhos": points, "values": values, "derivative": derivative}


def artificial_checks():
    """Fixed four-point algebra/Fourier checks; zero Gaussian integrals."""
    z1, z2 = np.array([-1.1, -.2, .4, 1.3]), np.array([.7, -.8, .1, -.3])
    weights = np.array([.1, .2, .3, .4])
    a, rho = .7, .2
    gx, gw = roots_legendre(64)
    t = np.concatenate([(lo + hi) / 2 + (hi - lo) * gx / 2 for lo, hi in zip(PANELS[:-1], PANELS[1:])])
    iw = np.concatenate([(hi - lo) * gw / 2 for lo, hi in zip(PANELS[:-1], PANELS[1:])])

    def finite_model(r):
        root = math.sqrt(1 - r * r)
        x2 = r * z1 + root * z2
        aa = log_expit(a * z1) + log_expit(a * x2)
        ap = a * expit(-a * x2) * (z1 - r / root * z2)
        phase = finite_phase_arrays(t, aa + 1.4, weights, a_prime=ap)
        ll = np.logaddexp(aa[:, None], aa[None, :]) - math.log(2)
        mix = expit(aa[:, None] - aa[None, :])
        direct = -float(np.sum(np.outer(weights, weights) * ll))
        direct_gradient = -float(np.sum(np.outer(weights, weights)
                                        * (mix * ap[:, None] + (1 - mix) * ap[None, :])))
        return phase, direct, direct_gradient, float(aa @ weights), float(ap @ weights)

    phase, raw, raw_grad, mean, mean_grad = finite_model(rho)
    zero, raw_zero, _, mean_zero, _ = finite_model(0.0)
    fdiff = float(np.sum(iw * FourierEngine.modulus_difference(phase, zero) / (t * np.sinh(math.pi * t))))
    fgrad = float(np.sum(iw * 2 * FourierEngine.path_mark(phase) / np.sinh(math.pi * t)))
    correct_value = raw - raw_zero + mean - mean_zero
    correct_gradient = raw_grad + mean_grad
    h = 1e-5
    plus, minus = finite_model(rho + h), finite_model(rho - h)
    fd = ((plus[1] + plus[3]) - (minus[1] + minus[3])) / (2 * h)
    errors = {"Fourier_vs_direct_with_mean_CV": abs(fdiff - correct_value),
              "path_vs_direct_with_mean_CV": abs(fgrad - correct_gradient),
              "path_vs_FD_with_mean_CV": abs(fgrad - fd)}
    assert errors["Fourier_vs_direct_with_mean_CV"] < 1e-11
    assert errors["path_vs_direct_with_mean_CV"] < 1e-11
    assert errors["path_vs_FD_with_mean_CV"] < 1e-8
    assert abs(mean - mean_zero) > 1e-5  # Fixture would detect omission of the CV.
    return {"passed": True, "errors": errors, "mean_CV": mean - mean_zero,
            "gradient_mean_CV": mean_grad, "finite_nodes": 4,
            "direct_K2_particle_pairs": 16, "synthetic_GL_nodes": len(t),
            "Gaussian_integrals": 0, "scientific_model_evaluations": 0,
            "random_draws": 0, "fits": 0, "Price_identity_not_imposed": True}


def freeze():
    if (OUT / "FREEZE.json").exists():
        raise FileExistsError("Refusing to replace a freeze.")
    files = [Path(__file__), DOC / "NUMERIC_DESIGN.md", DOC / "PLAN_FOURIER.md",
             DOC / "NULL_PADDING_THEORY.md", ROOT / "HANDOFF.md"]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
    unit = read(OUT / "UNIT_CHECK.json")
    assert unit["passed"] and unit["source_sha256"] == sha(__file__)
    save(OUT / "CONFIG.json", CONFIG)
    items = []
    for i, p in enumerate(files):
        target = OUT / "source_inputs" / f"{i:02d}_{p.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(p.read_bytes())
        items.append({"original": p.relative_to(ROOT).as_posix(),
                      "snapshot": target.relative_to(ROOT).as_posix(), "sha256": sha(target)})
    save(OUT / "FREEZE.json", {"created": utc(), "inputs": items,
                                "config_sha256": sha(OUT / "CONFIG.json"),
                                "unit_sha256": sha(OUT / "UNIT_CHECK.json"),
                                "scientific_integrals_before_freeze": 0})


def verify_freeze(live=False):
    receipt = read(OUT / "FREEZE.json")
    assert read(OUT / "CONFIG.json") == CONFIG
    assert sha(OUT / "CONFIG.json") == receipt["config_sha256"]
    assert sha(OUT / "UNIT_CHECK.json") == receipt["unit_sha256"]
    for item in receipt["inputs"]:
        assert sha(ROOT / item["snapshot"]) == item["sha256"]
        if live:
            assert sha(ROOT / item["original"]) == item["sha256"]


def qualify(engines, grid):
    records, fd_rows = [], []
    gate = CONFIG["gates"]

    def add(name, error, limit, **meta):
        records.append({"name": name, "error": float(error), "limit": float(limit),
                        "passed": bool(error <= limit), **meta})

    index = {(r["variant"], r["a"], r["theta"], r["n"], r["rho"]): r for r in grid}
    for case in CONFIG["cases"]:
        a, theta, n = case["a"], case["theta"], case["n"]
        scale = math.sqrt(max(1, n)) if a == 1 else 1.0
        value_limit = gate["a1_value_scaled"] if a == 1 else gate["weak_value"]
        grad_limit = gate["a1_gradient_scaled"] if a == 1 else gate["weak_gradient"]
        for rho in RHO_GRID:
            rows = [index[(variant, a, theta, n, rho)] for variant in VARIANTS]
            meta = {**case, "rho": rho}
            add("variant_delta_F", scale * np.ptp([r["delta_F"] for r in rows]), value_limit, **meta)
            add("variant_path_gradient", scale * np.ptp([r["path_gradient"] for r in rows]), grad_limit, **meta)
            add("variant_pair_moment", max(np.ptp([r[k] for r in rows])
                                           for k in ("teacher_spin_moment", "candidate_spin_moment")), gate["pair"], **meta)
            add("variant_joint_KL", np.ptp([r["joint_KL"] for r in rows]),
                gate["a1_KL"] if a == 1 else gate["weak_KL"], **meta)
        for rho in CONFIG["fd_rhos"]:
            target = index[("primary", a, theta, n, rho)]
            for step in CONFIG["fd_steps"]:
                fd = five_point_fd(engines["primary"], a, theta, n, rho, step)
                fd_rows.append({**case, "rho": rho, "path_gradient": target["path_gradient"], **fd})
                add("primary_FD_path", scale * abs(fd["derivative"] - target["path_gradient"]),
                    gate["a1_FD_scaled"] if a == 1 else gate["weak_FD"],
                    **case, rho=rho, step=step)
    # Include every saved evaluation, including the primary-only FD perturbations.
    for variant, engine in engines.items():
        for row in engine.cache.values():
            scale = math.sqrt(max(1, row["n"])) if row["a"] == 1 else 1.0
            add("Price_path", scale * abs(row["Price_gradient"] - row["path_gradient"]),
                gate["a1_gradient_scaled"] if row["a"] == 1 else gate["weak_gradient"],
                **{k: row[k] for k in ("variant", "a", "theta", "n", "rho")})
        for row in engine.native_cache.values():
            add("active_mean_drift", abs(row["centered_A_mean"]), gate["mean_drift"],
                variant=variant, a=row["a"], rho=row["rho"])
        for row in engine.nuisance_cache.values():
            add("CF_modulus_roundoff", row["CF_maximum_positive_roundoff"], gate["CF_roundoff"],
                variant=variant, a=row["a"], repaired_count=row["CF_positive_roundoff_count"])
    return records, fd_rows


def prediction_readouts(grid):
    idx = {(r["a"], r["theta"], r["n"], r["rho"]): r for r in grid if r["variant"] == "primary"}
    rows = []
    for theta in (0.0, 0.15):
        for rho in (0.3, 0.75):
            val = idx[(1.0, theta, 4094, rho)]
            expected = val["asymptotic_scaled_gradient"]
            error = abs(val["sqrt_n_Price_gradient"] - expected) / abs(expected)
            rows.append({"name": "near_tie_gradient", "theta": theta, "rho": rho,
                         "actual": val["sqrt_n_Price_gradient"], "expected": expected,
                         "relative_error": error, "passed": error <= CONFIG["gates"]["prediction_relative"]})
    for theta, reference_rho in ((0.0, 0.0), (0.15, 0.15)):
        difference = idx[(1.0, theta, 4094, .75)]["delta_F"] - idx[(1.0, theta, 4094, reference_rho)]["delta_F"]
        rows.append({"name": "large_n_endpoint_preference", "theta": theta,
                     "reference_rho": reference_rho, "difference": difference, "passed": difference < 0})
    for n in (0, 1, 2):
        a, rho = .05, .75
        actual = idx[(a, 0.0, n, rho)]["delta_F"] / a ** 6
        expected = (10 - 3 * (n + 2)) * rho ** 2 / 512
        error = abs(actual - expected) / abs(expected)
        rows.append({"name": "null_sixth_order", "n": n, "a": a, "rho": rho,
                     "actual": actual, "expected": expected, "relative_error": error,
                     "passed": error <= CONFIG["gates"]["prediction_relative"]})
    difference = idx[(1.0, 0.0, 0, .75)]["delta_F"]
    rows.append({"name": "fixed_noise_null_no_padding", "difference": difference, "passed": difference > 0})
    assert len(rows) == 10
    return rows


def run():
    verify_freeze(live=True)
    if (OUT / "STATE.json").exists():
        raise FileExistsError("Existing run: no automatic restart or extra budget.")
    started = time.perf_counter()
    save(OUT / "STATE.json", {"state": "running", "started": utc()})
    grid = []
    try:
        engines = {v: FourierEngine(v) for v in VARIANTS}
        for variant, engine in engines.items():
            for case in CONFIG["cases"]:
                for rho in RHO_GRID:
                    grid.append(engine.evaluate(**case, rho=rho))
            print(f"fixed grid complete: {variant}", flush=True)
        save(OUT / "GRID.json", {"rows": grid, "count": len(grid)})
        checks, fd_rows = qualify(engines, grid)
        save(OUT / "FD.json", {"rows": fd_rows})
        save(OUT / "CHECKS.json", {"rows": checks, "all_passed": all(x["passed"] for x in checks)})
        predictions = prediction_readouts(grid)
        # No more engine calls after this export: it captures every computed row.
        export_cache(engines, OUT)
        ledger = {"fixed_grid_rows": len(grid), "primary_grid_rows": len(grid) // 3,
                  "FD_stencils": len(fd_rows), "all_evaluate_rows": sum(len(e.cache) for e in engines.values()),
                  "engines": {v: e.ledger for v, e in engines.items()},
                  "new_labels": 0, "random_draws": 0, "fits": 0, "locators": 0,
                  "wall_seconds": time.perf_counter() - started, "Python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__}
        qualified = all(x["passed"] for x in checks)
        save(OUT / "RESULTS.json", {"numerically_qualified": qualified,
                                    "prediction_readouts": predictions, "prediction_total": 10,
                                    "prediction_passed": sum(x["passed"] for x in predictions),
                                    "prediction_interpretable": qualified,
                                    "tail_bounds": {str(a): tail_bounds(a) for a in (1.0, .05)},
                                    "ledger": ledger, "statistical_CI": False,
                                    "continuous_global_certificate": False,
                                    "strict_full_quadrature_certificate": False})
        save(OUT / "STATE.json", {"state": "complete" if qualified else "numerical_failed",
                                   "finished": utc(), "no_added_budget": True}, exclusive=False)
        names = ["CONFIG.json", "FREEZE.json", "GRID.json", "FD.json", "CHECKS.json", "EVALUATIONS.json",
                 "CACHE_INDEX.json", "RESULTS.json", "STATE.json"]
        save(OUT / "RESULTS_FREEZE.json", {"created": utc(), "files": {n: sha(OUT / n) for n in names},
                                          "optional_workspace_arrays": {"CACHE.npz": sha(OUT / "CACHE.npz")}})
        print(json.dumps({"numerically_qualified": qualified, "qualification_count": len(checks),
                          "qualification_failures": sum(not x["passed"] for x in checks),
                          "prediction_passed": sum(x["passed"] for x in predictions), "ledger": ledger}), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"created": utc(), "type": type(exc).__name__, "message": str(exc),
                                    "completed_grid_rows": len(grid), "automatic_retry": False})
        save(OUT / "STATE.json", {"state": "execution_failed", "finished": utc()}, exclusive=False)
        raise


def check(workspace=False):
    verify_freeze()
    receipt = read(OUT / "RESULTS_FREEZE.json")
    for name, expected in receipt["files"].items():
        assert sha(OUT / name) == expected, name
    if workspace:
        for name, expected in receipt["optional_workspace_arrays"].items():
            assert sha(OUT / name) == expected, name
    grid = read(OUT / "GRID.json")["rows"]
    results, checks = read(OUT / "RESULTS.json"), read(OUT / "CHECKS.json")["rows"]
    assert len(grid) == 1197 and results["ledger"]["locators"] == 0
    assert len(read(OUT / "FD.json")["rows"]) == 152
    for row in checks:
        assert row["passed"] == (row["error"] <= row["limit"])
    assert results["numerically_qualified"] == all(x["passed"] for x in checks)
    assert results["prediction_readouts"] == prediction_readouts(grid)
    assert results["prediction_total"] == 10
    assert results["prediction_passed"] == sum(x["passed"] for x in results["prediction_readouts"])
    evaluations = read(OUT / "EVALUATIONS.json")
    assert evaluations["count"] == len(evaluations["rows"]) == results["ledger"]["all_evaluate_rows"]
    print(json.dumps({"passed": True, "workspace_arrays_checked": workspace,
                      "scope": "public source/result bytes, saved qualification and prediction arithmetic; no reintegration",
                      "state": read(OUT / "STATE.json")["state"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("qa", "freeze", "run", "check"):
        group.add_argument("--" + name, action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if args.qa:
        if (OUT / "FREEZE.json").exists():
            raise FileExistsError("Cannot replace frozen QA.")
        result = artificial_checks()
        result.update({"source_sha256": sha(__file__), "created": utc()})
        save(OUT / "UNIT_CHECK.json", result)
        print(json.dumps(result))
    elif args.freeze:
        freeze()
    elif args.run:
        run()
    else:
        check(args.workspace)


if __name__ == "__main__":
    main()
