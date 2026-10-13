"""Rigorous scalar truncation bounds, scale=1, without scientific integration.

All decimal/rational input strings denote exact rationals. Floats are rejected.
Arb endpoints are exported as exact dyadics, never float certificates. This
module does not bound finite-interval quadrature error or prove a local witness.
"""
from __future__ import annotations

from fractions import Fraction
import operator
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/finite_selection_tools"))
from flint import arb, ctx
import flint


def rational(value, name="value"):
    if isinstance(value, (bool, float)):
        raise ValueError(f"{name} must be an exact rational string/int/Fraction")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid exact rational {name}") from exc


def integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an actual positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an actual positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be an actual positive integer")
    return result


def aq(value):
    value = rational(value)
    return arb(value.numerator) / value.denominator


def dyadic(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def ball_record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite Arb bound")
    return {"arb_display_only": str(value), "lower_exact_dyadic": dyadic(value.lower()),
            "upper_exact_dyadic": dyadic(value.upper())}


def dyadic_less_rational(endpoint, target):
    """Exact integer comparison without expanding enormous powers of two."""
    q = rational(target, "target")
    mantissa = int(endpoint["mantissa"])
    exponent = integer(abs(endpoint["exponent_of_two"]) + 1, "dyadic exponent") - 1
    exponent *= 1 if endpoint["exponent_of_two"] >= 0 else -1
    if mantissa < 0 or q < 0:
        raise ValueError("comparison requires nonnegative operands")
    left, right = mantissa * q.denominator, q.numerator
    if not left or not right:
        return left < right
    if exponent >= 0:
        left_bits, right_bits = left.bit_length() + exponent, right.bit_length()
        return left_bits < right_bits if left_bits != right_bits else (left << exponent) < right
    left_bits, right_bits = left.bit_length(), right.bit_length() - exponent
    return left_bits < right_bits if left_bits != right_bits else left < (right << -exponent)


def bound_box(m, K, b_l1, a_l1, normalL, teacherL, vmin, vmax, T, *,
              precision_bits=192, targets=None):
    """Return JSON data for a parameter box with l1 norm *upper* bounds.

    The physical Laplace variable is u=exp(v), likelihood_scale=1. Outputs
    cover gradient Euclidean norm and Hessian spectral norm in [b,a], and
    Hperp in unscaled transverse-loading coordinates (not Hangle).
    teacherL is a two-dimensional standard-Gaussian rectangle truncation.
    Target comparisons use exact upper dyadic endpoints and rational targets.
    """
    m, K = integer(m, "m"), integer(K, "K")
    precision_bits = integer(precision_bits, "precision_bits")
    if precision_bits < 64:
        raise ValueError("precision_bits must be at least 64")
    names = ("b_l1", "a_l1", "normalL", "teacherL", "vmin", "vmax", "T")
    inputs = {name: rational(value, name) for name, value in zip(
        names, (b_l1, a_l1, normalL, teacherL, vmin, vmax, T))}
    if inputs["b_l1"] < 0 or inputs["a_l1"] < 0:
        raise ValueError("l1 upper bounds must be nonnegative")
    if any(inputs[name] <= 0 for name in ("normalL", "teacherL", "T")):
        raise ValueError("Gaussian truncations and T must be positive")
    if inputs["vmin"] >= inputs["vmax"]:
        raise ValueError("vmin < vmax required")
    previous = ctx.prec
    ctx.prec = precision_bits
    try:
        x = {name: aq(value) for name, value in inputs.items()}
        quarter, e = arb(1)/4, arb(1).exp()
        cg, ch, cp = arb(m).sqrt(), 2*m+quarter, m+quarter
        ch_high, cp_high = ch+2*m/e, cp+2*m/e
        def gaussian_tail(L):
            phi = (-L*L/2).exp() / (2*arb.pi()).sqrt()
            return 2*phi/L, 2*phi*(L+1/L)
        pT, t2T = gaussian_tail(x["T"])
        pL, t2L = gaussian_tail(x["normalL"])
        pteacher, _ = gaussian_tail(x["teacherL"])
        eta = (-x["b_l1"]-x["T"]*x["a_l1"]).exp() / (arb(2)**m)
        u0, U = x["vmin"].exp(), x["vmax"].exp()
        damp = (-U*eta/2).exp()
        weighted = (1+K)*damp+pT**K+K*t2T*pT**(K-1)
        unweighted = damp+pT**K
        inner_weighted = K*(t2L+K*pL)
        inner_unweighted, teacher_mass = K*pL, 2*pteacher
        values = {
            "gradient": {"upper_u": cg*weighted, "lower_u": K*(2*arb(m)).sqrt()*u0,
                         "inner_gaussian": cg*inner_weighted,
                         "teacher_gaussian": cg*(1+K)*teacher_mass},
            "radial_hessian": {"upper_u": ch_high*weighted,
                         "lower_u": 2*K*(m+quarter)*u0+m*K*K*u0*u0,
                         "inner_gaussian": ch*inner_weighted,
                         "teacher_gaussian": ch*(1+K)*teacher_mass},
            "transverse_hessian": {"upper_u": cp_high*unweighted,
                         "lower_u": K*quarter*u0+m*K*(K-1)*u0*u0/2,
                         "inner_gaussian": cp*inner_unweighted,
                         "teacher_gaussian": cp*teacher_mass}}
        for parts in values.values():
            parts["total_truncation"] = sum(parts.values(), arb(0))
        output = {"schema_version": "1.0", "scope": "analytic_truncation_only_scale_one",
                  "m": m, "K": K, "actual_integer_validation": True,
                  "exact_rational_inputs": {k: str(v) for k, v in inputs.items()},
                  "precision_bits": precision_bits, "python_flint_version": flint.__version__,
                  "likelihood_scale_exact": "1", "physical_u": "exp(v); no division by K",
                  "constants": {k: ball_record(v) for k, v in {
                      "C_gradient": cg, "C_radial_global": ch, "C_transverse_global": cp,
                      "C_radial_upper_u": ch_high, "C_transverse_upper_u": cp_high,
                      "eta": eta, "u_min": u0, "u_max": U,
                      "p_T_upper": pT, "t2_T_upper": t2T,
                      "p_normalL_upper": pL, "t2_normalL_upper": t2L,
                      "teacher_missing_l1_upper": teacher_mass}.items()},
                  "bounds": {k: {j: ball_record(v) for j, v in parts.items()}
                             for k, parts in values.items()}, "comparisons": [],
                  "finite_interval_quadrature_certified": False,
                  "center_evaluated": False, "scientific_data_reads": 0,
                  "random_draws": 0, "acb_integral_calls": 0}
        for key, target in (targets or {}).items():
            if key not in values:
                raise ValueError(f"unknown target field: {key}")
            q = rational(target, "target")
            output["comparisons"].append({"bound": key+".total_truncation",
                "target_exact_rational": str(q), "strictly_below": dyadic_less_rational(
                    output["bounds"][key]["total_truncation"]["upper_exact_dyadic"], q),
                "comparison_method": "exact_integer_dyadic_vs_rational"})
        return output
    finally:
        ctx.prec = previous
