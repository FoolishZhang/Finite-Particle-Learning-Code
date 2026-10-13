"""Independent ordinary-quadrature audit of the frozen R02 noise law.

This module has no file I/O and no project imports.  The target run belongs to
the root driver after source/input freezing; importing this file runs nothing.
"""
from __future__ import annotations

from fractions import Fraction
import math
import platform
import time

import numpy as np
import scipy
from scipy.special import roots_legendre


class _BudgetExpired(RuntimeError):
    pass


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("exact integer field required")
    result = Fraction(value)
    if result.denominator != 1:
        raise ValueError("integer field cannot have a fractional part")
    return result.numerator


def _endpoint(data):
    mantissa = _integer(data["mantissa"])
    exponent = _integer(data["exponent_of_two"])
    return Fraction(mantissa * 2 ** max(exponent, 0),
                    2 ** max(-exponent, 0))


def _input_midpoints(constants):
    """Decode exact endpoint midpoints before a single conversion to float64."""
    values, provenance = {}, {}
    for name in ("v", "d0", "kappa3_L"):
        lower = _endpoint(constants[name]["lower_exact_dyadic"])
        upper = _endpoint(constants[name]["upper_exact_dyadic"])
        if lower > upper:
            raise ValueError(name + " has inverted interval endpoints")
        if name == "v" and lower <= 0:
            raise ValueError("the complete input v interval must be positive")
        if name == "d0" and lower < 0:
            raise ValueError("the complete input d0 interval must be nonnegative")
        midpoint = (lower + upper) / 2
        value = float(midpoint)
        if not math.isfinite(value) or (name == "v" and value <= 0):
            raise ValueError(name + " midpoint is not representable for this audit")
        values[name] = value
        provenance[name] = {
            "midpoint_exact": {"numerator": str(midpoint.numerator),
                               "denominator": str(midpoint.denominator)},
            "midpoint_float64": value,
        }
    return values, provenance


def _node_scores(nodes, v, d0):
    # This is the original log-sigmoid expression.  It deliberately avoids the
    # primary implementation's x/2 - log(cosh(x/2)) + d0 factorization.
    log_likelihood = -np.logaddexp(0.0, -nodes)
    ell = -math.log(2.0) - d0
    standardized = (log_likelihood - ell) / math.sqrt(v)
    return log_likelihood, standardized


def compute_audit(old_constants, hard_seconds=120):
    """Return a bounded independent GL384/GL512 point-estimate audit.

    Input intervals are old CONSTANTS['composition']['constants'].  Each exact
    endpoint midpoint is converted to float64.  No interval propagation, strict
    quadrature bound, or strict tail certification is claimed here.  Comparison
    with the primary true intervals is the root driver's responsibility.
    """
    if (isinstance(hard_seconds, bool) or
            not isinstance(hard_seconds, (int, float)) or
            not math.isfinite(hard_seconds) or hard_seconds < 0):
        raise ValueError("hard_seconds must be finite and nonnegative")
    started = time.perf_counter()
    values, provenance = _input_midpoints(old_constants)
    frequencies = [Fraction(1, 4) + Fraction(j, 64) for j in range(273)]
    result = {
        "schema": "r02_remainder_independent_noise_audit_v1",
        "status": "running",
        "configuration": {
            "GL_orders": [384, 512], "window": [-12, 12],
            "Gaussian_weight": "GL_weight * exp(-x*x/2)/sqrt(2*pi)",
            "quadrature_weight_renormalization": False,
            "log_likelihood": "L=-logaddexp(0,-x)",
            "mean_log_likelihood": "ell=-log(2)-d0",
            "standardized_noise": "U=(L-ell)/sqrt(v)",
            "CF_convention": "E exp(+i*u*U)",
            "frequencies_exact": [str(u) for u in frequencies],
            "u_start_exact": "1/4", "u_step_exact": "1/64",
            "u_count": 273, "dtype": "float64/complex128",
            "hard_seconds": float(hard_seconds),
            "input_conversion": "exact endpoint midpoint, then float64",
            "kappa3_L_use": "provenance only; not used in these integrands",
        },
        "input_midpoints": provenance,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__},
        "qualification": {
            "independent_expression_point_estimate": True,
            "second_rigorous_certificate": False,
            "quadrature_error_certified": False,
            "omitted_Gaussian_tail_certified": False,
            "input_interval_uncertainty_propagated": False,
            "does_not_import_primary_noise_module": True,
        },
        "by_order": {},
        "work_counts": {
            "GL_rules_constructed": 0,
            "Gaussian_scalar_nodes": 0,
            "fourth_moment_integrand_evaluations": 0,
            "complex_exponential_evaluations": 0,
            "complex_CF_sums_completed": 0,
            "real_scalar_sums_completed": 0,
            "new_project_Dn_evaluations": 0,
            "new_random_samples": 0,
            "new_NPZ_reads": 0,
            "new_NPZ_writes": 0,
            "new_optimizations": 0,
        },
    }
    counts = result["work_counts"]

    def check_budget():
        if time.perf_counter() - started >= hard_seconds:
            raise _BudgetExpired("fixed independent-audit time budget exhausted")

    try:
        for order in (384, 512):
            check_budget()
            abscissas, raw_weights = roots_legendre(order)
            counts["GL_rules_constructed"] += 1
            check_budget()
            nodes = 12.0 * abscissas
            weights = (12.0 * raw_weights * np.exp(-nodes * nodes / 2.0)
                       / math.sqrt(2.0 * math.pi))
            _, standardized = _node_scores(nodes, values["v"], values["d0"])
            counts["Gaussian_scalar_nodes"] += order
            mu4 = float(np.dot(weights, standardized ** 4))
            counts["fourth_moment_integrand_evaluations"] += order
            counts["real_scalar_sums_completed"] += 1
            entry = {
                "order": order, "status": "running",
                "quality_mass": float(np.sum(weights)),
                "quality_mass_deviation_from_one": abs(float(np.sum(weights)) - 1.0),
                "mu4": mu4, "CF_real": [], "CF_imag": [], "CF_abs": [],
                "CF_rows_completed": 0,
            }
            result["by_order"][str(order)] = entry
            if not math.isfinite(mu4):
                raise ArithmeticError("fourth moment is not finite")
            for u in frequencies:
                check_budget()
                cf = complex(np.dot(weights, np.exp(1j * float(u) * standardized)))
                counts["complex_exponential_evaluations"] += order
                if not math.isfinite(cf.real) or not math.isfinite(cf.imag):
                    raise ArithmeticError("CF point estimate is not finite")
                entry["CF_real"].append(cf.real)
                entry["CF_imag"].append(cf.imag)
                entry["CF_abs"].append(abs(cf))
                entry["CF_rows_completed"] += 1
                counts["complex_CF_sums_completed"] += 1
                counts["real_scalar_sums_completed"] += 2
            entry["status"] = "complete"
            check_budget()

        low, high = (result["by_order"][str(order)] for order in (384, 512))
        result["absolute_GL384_GL512_differences"] = {
            "mu4": abs(low["mu4"] - high["mu4"]),
            "CF_real": [abs(x-y) for x, y in zip(low["CF_real"], high["CF_real"])],
            "CF_imag": [abs(x-y) for x, y in zip(low["CF_imag"], high["CF_imag"])],
            "CF_abs": [abs(x-y) for x, y in zip(low["CF_abs"], high["CF_abs"])],
            "quality_mass": abs(low["quality_mass"] - high["quality_mass"]),
        }
        result["status"] = "complete"
        check_budget()
    except _BudgetExpired as exc:
        result["status"] = "budget_exceeded"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_seconds"] = time.perf_counter() - started
    result["budget_scope"] = "API compute including input decoding; excludes imports and caller serialization"
    if result["status"] == "complete" and result["elapsed_seconds"] > hard_seconds:
        result["status"] = "budget_exceeded"
        result["failure"] = {"type": "_BudgetExpired",
                             "message": "budget exceeded during final result composition"}
    return result
