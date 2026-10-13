"""Independent floating-point expression audit for fixed R02 model constants.

Pure compute_audit() API: no files, no project kernel imports, no random samples.
Execution of the fixed scientific targets is the root driver's responsibility.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import math
import platform
import time

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, ndtr


class _BudgetExpired(RuntimeError):
    pass


def _config(rho, K, hard_seconds):
    if isinstance(rho, (float, bool)):
        raise TypeError("rho must be an exact rational string, int or Fraction")
    rho = Fraction(rho)
    if not Fraction(-1, 2) < rho < 1:
        raise ValueError("three-dimensional equicorrelation must be positive definite")
    if type(K) is not int or K < 2:
        raise ValueError("K must be an integer at least 2")
    if isinstance(hard_seconds, bool) or not math.isfinite(hard_seconds) or hard_seconds <= 0:
        raise ValueError("hard_seconds must be finite and positive")
    return rho, K, float(hard_seconds)


def compute_audit(rho="9/20", K=100, hard_seconds=120):
    """Return partial or complete independent GH32/GH48 + adaptive-quad audit.

    Bounds/interval comparisons are deliberately not implemented here: these are
    ordinary float64 estimates, and root compares them with the strict constants.
    """
    rho, K, hard_seconds = _config(rho, K, hard_seconds)
    started = time.perf_counter()
    result = {
        "schema": "r02_finite_scale_independent_expression_audit_v1",
        "status": "running",
        "configuration": {
            "rho_exact": {"numerator": str(rho.numerator), "denominator": str(rho.denominator)},
            "K": K, "m": 3, "GH_orders": [32, 48], "label_probability": "1/8",
            "label_order": "lexicographic (-1,+1)^3, first observation most significant",
            "coordinate_method": "standard-normal tensor nodes multiplied by lower Cholesky transpose",
            "quadrature_weight_renormalization": False, "dtype": "float64",
            "normal_maximum_window": [-12, 12], "quad_epsabs": 1e-12,
            "quad_epsrel": 1e-12, "quad_limit": 200, "hard_seconds": hard_seconds,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "qualification": "independent floating-point expression check; quadrature errors are not rigorous certificates",
        "by_order": {}, "maximum": {}, "targets_by_order": {},
        "work_counts": {
            "GH_rules_constructed": 0, "active_Gaussian_vector_nodes": 0,
            "active_label_node_evaluations": 0, "active_scalar_moment_integrands": 0,
            "nuisance_Gaussian_scalar_nodes": 0, "nuisance_scalar_moment_integrands": 0,
            "adaptive_quad_calls_started": 0, "adaptive_quad_calls_completed": 0,
            "adaptive_function_evaluations": 0,
            "new_project_Dn_evaluations": 0, "new_random_samples": 0,
            "new_NPZ_reads": 0, "new_optimizations": 0,
        },
        "does_not_call_primary_factorization": True,
    }
    counts = result["work_counts"]

    def check_budget():
        if time.perf_counter() - started >= hard_seconds:
            raise _BudgetExpired("fixed independent-audit time budget exhausted")

    try:
        rho_float = float(rho)
        C = np.full((3, 3), rho_float, dtype=np.float64)
        np.fill_diagonal(C, 1.0)
        chol = np.linalg.cholesky(C)
        result["candidate_covariance_float64"] = C.tolist()
        result["cholesky_float64"] = chol.tolist()
        for order in (32, 48):
            check_budget()
            raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(order)
            nodes = math.sqrt(2.0) * raw_nodes
            weights = raw_weights / math.sqrt(math.pi)
            counts["GH_rules_constructed"] += 1

            # Direct log-sigmoid moments, without log-cosh/tanh transformations.
            noise = -np.logaddexp(0.0, -nodes)
            noise_raw = [float(np.dot(weights, noise ** j)) for j in (1, 2, 3)]
            ell, raw2, raw3 = noise_raw
            v = raw2 - ell ** 2
            kappa3 = raw3 - 3 * ell * raw2 + 2 * ell ** 3
            counts["nuisance_Gaussian_scalar_nodes"] += order
            counts["nuisance_scalar_moment_integrands"] += 3 * order
            item = {
                "order": order, "one_dimensional_weight_sum": float(np.sum(weights)),
                "nuisance_raw_logsigmoid_moments": {"E_L": ell, "E_L2": raw2, "E_L3": raw3},
                "labels": [], "targets": {"v": v, "kappa3_L": kappa3},
            }
            result["by_order"][str(order)] = item
            check_budget()

            standard = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
            xyz = standard @ chol.T
            tensor_weights = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).reshape(-1)
            node_count = order ** 3
            counts["active_Gaussian_vector_nodes"] += node_count
            item["three_dimensional_weight_sum"] = float(np.sum(tensor_weights))
            item["Gaussian_vector_nodes"] = node_count

            m_H = cbar = V_H = V_A = 0.0
            for label_index, spin in enumerate(product((-1, 1), repeat=3)):
                check_budget()
                transformed = xyz * np.asarray(spin, dtype=np.float64)[None, :]
                active_A = -np.sum(np.logaddexp(0.0, -transformed), axis=1)
                active_H = float(spin[0] * spin[1]) * expit(-transformed[:, 0]) * expit(-transformed[:, 1])
                mean_A = float(np.dot(tensor_weights, active_A))
                mean_H = float(np.dot(tensor_weights, active_H))
                moment_AH = float(np.dot(tensor_weights, active_A * active_H))
                moment_A2 = float(np.dot(tensor_weights, active_A * active_A))
                moment_H2 = float(np.dot(tensor_weights, active_H * active_H))
                covariance = moment_AH - mean_A * mean_H
                var_A = moment_A2 - mean_A ** 2
                var_H = moment_H2 - mean_H ** 2
                item["labels"].append({
                    "index": label_index, "spin": list(spin), "probability": 0.125,
                    "E_A": mean_A, "E_H": mean_H, "E_AH": moment_AH,
                    "E_A2": moment_A2, "E_H2": moment_H2,
                    "Cov_H_A_within_label": covariance,
                    "Var_H_within_label": var_H, "Var_A_within_label": var_A,
                })
                # Subtract each label's own means BEFORE averaging over labels.
                m_H += mean_H / 8
                cbar += covariance / 8
                V_H += var_H / 8
                V_A += var_A / 8
                counts["active_label_node_evaluations"] += node_count
                counts["active_scalar_moment_integrands"] += 5 * node_count
            item["targets"].update({"m_H": m_H, "cbar": cbar, "V_H": V_H, "V_A": V_A})
            item["status"] = "complete"
            check_budget()

        for name, exponent in (("J_K", 1), ("B_K", 2)):
            check_budget()
            calls_here = 0

            def integrand(z):
                nonlocal calls_here
                check_budget()
                calls_here += 1
                counts["adaptive_function_evaluations"] += 1
                density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
                return K * z ** exponent * density * float(ndtr(z)) ** (K - 1)

            counts["adaptive_quad_calls_started"] += 1
            quad_result = quad(integrand, -12.0, 12.0, epsabs=1e-12, epsrel=1e-12,
                               limit=200, full_output=1)
            raw_integral, estimated_error, info = quad_result[:3]
            # The -1 in B_K is outside the maximum-density integral.
            value = raw_integral - (1.0 if name == "B_K" else 0.0)
            entry = {
                "value": float(value), "raw_integral": float(raw_integral),
                "post_integral_subtraction": 1 if name == "B_K" else 0,
                "absolute_error_estimate": float(estimated_error),
                "error_estimate_is_rigorous": False, "neval": int(info["neval"]),
                "actual_callback_calls": calls_here, "subintervals_used": int(info["last"]),
                "omitted_outside_window": True, "tail_error_certified_by_this_audit": False,
                "integrand": "K*z*phi(z)*Phi(z)^(K-1)" if name == "J_K" else "K*z^2*phi(z)*Phi(z)^(K-1)",
            }
            if len(quad_result) > 3:
                entry["message"] = str(quad_result[3])
                result["maximum"][name] = entry
                raise RuntimeError("adaptive quadrature reported non-success: " + str(quad_result[3]))
            result["maximum"][name] = entry
            counts["adaptive_quad_calls_completed"] += 1
            check_budget()

        for order in (32, 48):
            targets = result["by_order"][str(order)]["targets"]
            targets.update({"J_K": result["maximum"]["J_K"]["value"], "B_K": result["maximum"]["B_K"]["value"]})
            targets["second_order_coefficient"] = targets["B_K"] / targets["v"] * (
                targets["m_H"] * targets["kappa3_L"] / (3 * targets["v"]) - targets["cbar"])
            result["targets_by_order"][str(order)] = dict(targets)
        result["absolute_GH32_GH48_differences"] = {
            name: abs(result["targets_by_order"]["32"][name] - result["targets_by_order"]["48"][name])
            for name in result["targets_by_order"]["32"]
        }
        result["status"] = "complete"
    except _BudgetExpired as exc:
        result["status"] = "budget_exceeded"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_seconds"] = time.perf_counter() - started
    result["budget_scope"] = "all compute work in API; excludes caller serialization and process/import startup"
    if result["status"] == "complete" and result["elapsed_seconds"] > hard_seconds:
        result["status"] = "budget_exceeded"
        result["failure"] = {"type": "_BudgetExpired", "message": "budget exceeded after final result composition"}
    return result
