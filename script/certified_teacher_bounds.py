"""Scalar analytic L1 bounds for exact two-dimensional logistic teachers.

No PMF values, scientific Gaussian integration, or finite-grid likelihood sums
are evaluated here. Decimal strings are exact rationals. Endpoint dyadics and
integer target comparisons are the certificate data; displays are not.
"""
from __future__ import annotations
from fractions import Fraction
import operator
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/finite_selection_tools"))
import flint
from flint import arb, ctx

DEFAULT_ANGLES = ("-7/10", "-1/2", "-3/10", "-1/10",
                  "1/10", "3/10", "1/2", "7/10")


def exact_q(value):
    if isinstance(value, (float, bool)):
        raise ValueError("only exact rational str/int/Fraction inputs are accepted")
    try:
        return Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact rational") from exc


def actual_int(value):
    if isinstance(value, bool):
        raise ValueError("actual positive integer required")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("actual positive integer required") from exc
    if value <= 0:
        raise ValueError("actual positive integer required")
    return value


def aq(value):
    value = exact_q(value)
    return arb(value.numerator)/value.denominator


def endpoint(value):
    m, e = value.man_exp()
    return {"mantissa": str(m), "exponent_of_two": int(e)}


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite Arb result")
    return {"arb_display_only": str(value), "lower_exact_dyadic": endpoint(value.lower()),
            "upper_exact_dyadic": endpoint(value.upper())}


def dyadic_lt_q(ep, target):
    q = exact_q(target)
    m, e = int(ep["mantissa"]), operator.index(ep["exponent_of_two"])
    if m < 0 or q < 0:
        raise ValueError("comparison requires nonnegative operands")
    left, right = m*q.denominator, q.numerator
    if not left or not right:
        return left < right
    lb, rb = left.bit_length()+max(e, 0), right.bit_length()+max(-e, 0)
    if lb != rb:
        return lb < rb
    return (left << max(e, 0)) < (right << max(-e, 0))


def teacher_l1_bounds(*, h="1/8", limit="10", strip="1", amplitude="4/5",
                      angles=DEFAULT_ANGLES, precision_bits=192,
                      target="1/1000000000000"):
    """Exact-rational/trigonometric teacher, full-weight inclusive grid.

    Weight rows are amplitude*(cos(angle),sin(angle)); bias is exactly zero.
    The result bounds mathematical quadrature only. A subsequent certified
    finite sum must add its own L1 arithmetic enclosure and preserve endpoints.
    """
    hq, Lq, dq, aq0 = map(exact_q, (h, limit, strip, amplitude))
    angleq = tuple(exact_q(a) for a in angles)
    precision_bits = actual_int(precision_bits)
    if precision_bits < 64 or min(hq, Lq, dq) <= 0 or aq0 < 0 or not angleq:
        raise ValueError("positive h, limit, strip; nonnegative amplitude; nonempty angles; bits>=64")
    Nq = Lq/hq
    if Nq.denominator != 1:
        raise ValueError("limit/h must be an exact integer")
    N = actual_int(Nq.numerator)
    tq = exact_q(target)
    if tq <= 0:
        raise ValueError("target must be positive")
    previous = ctx.prec; ctx.prec = precision_bits
    try:
        ha, La, da, amp = map(aq, (hq, Lq, dq, aq0))
        row_bounds, C = [], arb(1)
        for angle in angleq:
            th = aq(angle)
            imaginary = da*amp*(abs(th.cos())+abs(th.sin()))
            if not imaginary < arb.pi():
                raise ValueError("pole-free strip condition was not rigorously established")
            cosine = (imaginary/2).cos()
            if not cosine > 0:
                raise ValueError("positive secant denominator not established")
            secant = 1/cosine
            C *= secant
            row_bounds.append({"angle_exact_rational": str(angle),
                               "imaginary_logit_bound": record(imaginary),
                               "row_modulus_sum_bound": record(secant),
                               "strictly_below_pi": True})
        M = (da*da).exp()*C
        q = (-2*arb.pi()*da/ha).exp()
        # Algebraically equivalent to ((1+q)/(1-q))**2 - 1;
        # this form avoids subtracting nearly equal numbers.
        alias = M*4*q/(1-q)**2
        phi0 = 1/(2*arb.pi()).sqrt()
        phiL = (-La*La/2).exp()*phi0
        pL = 2*phiL/La
        s = 1+ha*phi0
        finite_tail = 2*s*pL
        total = alias+finite_tail
        comparisons = []
        for name, val in (("infinite_grid_alias_L1", alias),
                          ("finite_grid_tail_L1", finite_tail),
                          ("total_quadrature_L1", total)):
            comparisons.append({"bound": name, "target_exact_rational": str(tq),
                                "strictly_below": dyadic_lt_q(endpoint(val.upper()), tq),
                                "method": "exact_integer_dyadic_vs_rational"})
        return {"schema_version": "1.0", "scope": "teacher_total_L1_analytic_quadrature_only",
                "precision_bits": precision_bits, "python_flint_version": flint.__version__,
                "exact_inputs": {"h": str(hq), "limit": str(Lq), "strip": str(dq),
                    "amplitude": str(aq0), "bias": "0",
                    "angles_radians": [str(a) for a in angleq]},
                "m": len(angleq), "label_count": 2**len(angleq),
                "grid": {"N_actual_integer": N, "minimum_index": -N, "maximum_index": N,
                         "points_per_axis": 2*N+1, "points_total": (2*N+1)**2,
                         "endpoint_weight": "full h^2, including all four corners",
                         "omitted_axis_indices_begin_at": N+1, "renormalized": False},
                "rows": row_bounds,
                "bounds": {name: record(value) for name, value in {
                    "label_modulus_product_C": C, "complex_shift_integral_M": M,
                    "alias_ratio_q": q, "one_dimensional_grid_mass_upper": s,
                    "two_sided_normal_tail_upper": pL,
                    "infinite_grid_alias_L1": alias,
                    "finite_grid_tail_L1": finite_tail,
                    "total_quadrature_L1": total}.items()},
                "comparisons": comparisons, "factor_256_used": False,
                "strict_all_L1_target": comparisons[-1]["strictly_below"],
                "target_exact_rational": str(tq),
                "PMF_evaluations": 0, "scientific_data_reads": 0, "random_draws": 0,
                "acb_integral_calls": 0, "finite_sum_arithmetic_certified": False,
                "binary64_teacher_weights_used": False}
    finally:
        ctx.prec = previous
