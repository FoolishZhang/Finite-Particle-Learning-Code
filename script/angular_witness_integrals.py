"""Pure full-label pair numerators and analytic error bounds for AngularWitness-v1.

This module has no file I/O, path mutation, project imports, RNG, optimization,
or old scientific-kernel calls.  Production callers must freeze exact inputs
and enforce the one-run contract before calling ``finite_pair_sum``.  A stopped
call returns its prefix sums; these are not true-integral enclosures.

Only the codecs exact/aq/record/endpoint_fraction/from_record/enlarge may be
shared by an independent expression audit.  The summation itself is the direct
i<j pair expression, not the independent U/V/T expression.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time

import flint
from flint import arb, ctx


M = 8
LABEL_COUNT = 1 << M
EXPECTED_FLINT_VERSION = "0.8.0"


def exact(value):
    """Accept explicit exact rationals, never binary floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("explicit int, rational string or Fraction required")
    return Fraction(value)


def aq(value):
    """Enclose one exact rational at the caller's current Arb precision."""
    value = exact(value)
    return arb(value.numerator) / value.denominator


def _endpoint(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def record(value):
    """Serialize outward Arb endpoints as exact dyadics; display is cosmetic."""
    if not isinstance(value, arb) or not value.is_finite():
        raise ArithmeticError("finite Arb value required")
    return {
        "arb_display_only": str(value),
        "lower_exact_dyadic": _endpoint(value.lower()),
        "upper_exact_dyadic": _endpoint(value.upper()),
    }


def endpoint_fraction(data):
    """Decode an endpoint without any floating-point conversion."""
    mantissa = exact(data["mantissa"])
    exponent = exact(data["exponent_of_two"])
    if mantissa.denominator != 1 or exponent.denominator != 1:
        raise ValueError("dyadic mantissa and exponent must be integers")
    m, e = mantissa.numerator, exponent.numerator
    return Fraction(m * (2 ** max(e, 0)), 2 ** max(-e, 0))


def from_record(data):
    """Enclose both saved exact endpoints at the current Arb precision."""
    lo = endpoint_fraction(data["lower_exact_dyadic"])
    hi = endpoint_fraction(data["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("inverted saved Arb endpoints")
    return aq(lo).union(aq(hi))


def enlarge(value, absolute_error):
    """Add a symmetric radius using a certified nonnegative error upper bound."""
    if not isinstance(value, arb) or not value.is_finite():
        raise ArithmeticError("finite Arb value required")
    if not isinstance(absolute_error, arb):
        absolute_error = aq(absolute_error)
    if not absolute_error.is_finite() or not absolute_error >= 0:
        raise ValueError("finite certified nonnegative absolute error required")
    return value + arb(0, absolute_error.upper())


def _rational_record(value):
    value = exact(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _inputs(a, u, step, N, bits):
    aa, uu = list(map(exact, a)), list(map(exact, u))
    step = exact(step)
    if len(aa) != M or len(uu) != M:
        raise ValueError("exactly eight loadings and eight direction integers required")
    if any(abs(x) > 1 for x in aa):
        raise ValueError("loadings must obey the exact envelope |a_i| <= 1")
    if any(x.denominator != 1 for x in uu):
        raise ValueError("u must contain exact integers")
    if sum(uu, Fraction(0)) != 0 or not any(uu):
        raise ValueError("nonzero sum-zero u required; common rotation is excluded")
    if step <= 0:
        raise ValueError("strictly positive exact step required")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("positive integer N required")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 32:
        raise ValueError("integer precision of at least 32 bits required")
    if flint.__version__ != EXPECTED_FLINT_VERSION:
        raise RuntimeError("AngularWitness-v1 requires python-flint 0.8.0")
    return aa, [x.numerator for x in uu], step


def _configuration(a, u, step, N, bits):
    return {
        "m": M,
        "b_exact": ["0"] * M,
        "a_exact": [_rational_record(x) for x in a],
        "u_exact": [str(x) for x in u],
        "S_exact": str(sum(x * x for x in u)),
        "direction": "d=u/sqrt(S); S=sum_i u_i^2; sum_i u_i=0",
        "unit_direction_used_without_rounding_sqrt": True,
        "step_exact": str(step),
        "N": N,
        "limit_exact": str(step * N),
        "endpoint_weights": "full",
        "precision_bits": bits,
        "python_flint_version": flint.__version__,
    }


def pair_bounds(a, u, step="1/8", N=88, strip="1", bits=192):
    """Uniform per-label integral error; no grid or sigmoid evaluations.

    The supported closed strip has exact width 0 < delta <= 1.  With |a_i|<=1,
    c=sec(delta/2), the one-side boundary integral is at most
    exp(delta^2/2) * c^8 * 8*c^2.  The default gives delta=1, h=1/8, L=11.
    """
    a, u, step = _inputs(a, u, step, N, bits)
    strip = exact(strip)
    if not 0 < strip <= 1:
        raise ValueError("the proved strip range is 0 < strip <= 1")
    previous = ctx.prec
    ctx.prec = bits
    try:
        delta, dx, limit = aq(strip), aq(step), aq(step * N)
        cosine = (delta / 2).cos()
        if not cosine > 0:
            raise ArithmeticError("positive cosine bound could not be certified")
        c = 1 / cosine
        C = c ** M
        pair_mass = sum((Fraction((u[i] - u[j]) ** 2, sum(x*x for x in u))
                         for i in range(M) for j in range(i + 1, M)), Fraction(0))
        if pair_mass != M:
            raise ArithmeticError("sum-zero unit-direction pair-mass identity failed")
        shift = (delta * delta / 2).exp()
        M_N = shift * C * M * c * c
        denominator = (2 * arb.pi() * delta / dx).exp() - 1
        if not denominator > 0:
            raise ArithmeticError("positive alias denominator could not be certified")
        phi_limit = (-limit * limit / 2).exp() / (2 * arb.pi()).sqrt()
        alias = 2 * M_N / denominator
        tail = M * 2 * phi_limit / limit
        result = _configuration(a, u, step, N, bits)
        result.update({
            "schema": "angular_witness_pair_bounds_v1",
            "strip_exact": str(strip),
            "node_count": 2 * N + 1,
            "label_uniform": True,
            "per_label_error_not_multiplied_by_label_count": True,
            "constants": {
                "loading_absolute_cap": record(arb(1)),
                "unit_direction_pair_mass": record(aq(pair_mass)),
                "P_real_absolute_cap": record(arb(M)),
                "c_sec_half_strip": record(c),
                "C_A": record(C),
                "P_strip_absolute_cap": record(M * c * c),
                "Gaussian_strip_factor": record(shift),
                "boundary_N_single_side": record(M_N),
                "alias_denominator": record(denominator),
                "phi_limit": record(phi_limit),
            },
            "errors": {
                "N_alias": record(alias),
                "N_grid_tail": record(tail),
                "N_total": record(alias + tail),
            },
            "finite_sum_arithmetic_included_separately_in_saved_balls": True,
            "new_integrals": 0,
            "new_grid_nodes": 0,
        })
        return result
    finally:
        ctx.prec = previous


def finite_pair_sum(a, u, step="1/8", N=88, bits=192, hard_seconds=600):
    """Return all 256 direct pair-expression finite sums, including any prefix.

    N_y = h sum_{k=-N}^N phi(kh) A_y(kh)
          sum_{i<j} a_i*a_j*(u_i-u_j)^2/S * (y_i-p_i)*(y_j-p_j).

    Observation zero is the most significant bit.  There is no symmetry
    reduction, label weighting, Q denominator, teacher input or renormalization.
    The monotonic-clock budget is checked between complete nodes.  Thus a final
    node or serialization can exceed the deadline; this is explicitly reported
    as budget_stopped even if every prescribed node was accumulated.
    """
    a, u, step = _inputs(a, u, step, N, bits)
    if (isinstance(hard_seconds, bool)
            or not isinstance(hard_seconds, (int, float))
            or not math.isfinite(hard_seconds) or hard_seconds <= 0):
        raise ValueError("finite positive wall-clock budget required")
    previous = ctx.prec
    ctx.prec = bits
    try:
        tick = time.perf_counter()
        S = sum(x * x for x in u)
        pairs = [(i, j, aq(a[i] * a[j] * Fraction((u[i] - u[j]) ** 2, S)))
                 for i in range(M) for j in range(i + 1, M)]
        labels = [tuple((label >> (M - 1 - i)) & 1 for i in range(M))
                  for label in range(LABEL_COUNT)]
        aa, dx = list(map(aq, a)), aq(step)
        normalizer = (2 * arb.pi()).sqrt()
        sums = [arb(0) for _ in range(LABEL_COUNT)]
        completed = 0
        for node in range(-N, N + 1):
            if time.perf_counter() - tick >= hard_seconds:
                break
            x = dx * node
            probabilities = [1 / (1 + (-ai * x).exp()) for ai in aa]
            residuals = [(-pi, 1 - pi) for pi in probabilities]
            products = [arb(1)]
            for pi in probabilities:
                products = [item for old in products
                            for item in (old * (1 - pi), old * pi)]
            weight = dx * (-x * x / 2).exp() / normalizer
            for label, product in enumerate(products):
                yy = labels[label]
                pair_value = sum((coefficient * residuals[i][yy[i]] * residuals[j][yy[j]]
                                  for i, j, coefficient in pairs), arb(0))
                sums[label] += weight * product * pair_value
            completed += 1
        finite_records = [record(value) for value in sums]
        elapsed = time.perf_counter() - tick
        within_budget = elapsed <= hard_seconds
        complete = completed == 2 * N + 1 and within_budget
        result = _configuration(a, u, step, N, bits)
        result.update({
            "schema": "angular_witness_pair_finite_sum_v1",
            "status": "completed" if complete else "budget_stopped",
            "budget_limit_exceeded": not within_budget,
            "label_count": LABEL_COUNT,
            "label_order": "integer labels 0..255; observation 0 is MSB",
            "N_definition": "E[A_y(Z) sum_i<j a_i a_j (u_i-u_j)^2/S (y_i-p_i)(y_j-p_j)]",
            "nodes_prescribed": 2 * N + 1,
            "nodes_completed": completed,
            "node_indices_completed": list(range(-N, -N + completed)),
            "N_finite_sum": finite_records,
            "wall_seconds": elapsed,
            "hard_budget_seconds": hard_seconds,
            "clock_budget_checked_between_nodes": True,
            "wall_seconds_include_setup_and_serialization": True,
            "real_sigmoid_evaluations": completed * M,
            "real_Gaussian_density_evaluations": completed,
            "label_probability_products": completed * LABEL_COUNT,
            "pair_terms_per_label_per_node": len(pairs),
            "pair_terms_evaluated_including_zero_coefficients": completed * LABEL_COUNT * len(pairs),
            "label_numerator_accumulations": completed * LABEL_COUNT,
            "finite_scalar_integrals_completed": LABEL_COUNT if complete else 0,
            "new_Q_denominator_integrals": 0,
            "new_F8_evaluations": 0,
            "new_kernel_calls": 0,
            "new_random_draws": 0,
            "new_optimizer_updates": 0,
            "scientific_NPZ_reads": 0,
            "automatic_retry_or_grid_extension": False,
            "finite_sum_only": True,
            "true_integral_enclosures_composed": False,
        })
        return result
    finally:
        ctx.prec = previous
