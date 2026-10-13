"""Independent full-angle-path finite sums for AngularWitness-v1.

The caller owns input/source freezing and one-shot scientific execution. This
module imports no project producer, reads no files, and has no scientific-run
CLI. ``--self-test`` only uses manufactured m=3 inputs and a three-node grid.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import time

from flint import arb, ctx


def exact(value):
    """Reject binary float input: the scientific caller must supply rationals."""
    if isinstance(value, (bool, float)):
        raise ValueError("explicit exact rational required")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def _endpoint(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def record(value):
    """Independent exact-dyadic codec; display text is not numerical evidence."""
    if not value.is_finite():
        raise ArithmeticError("nonfinite path-audit interval")
    return {
        "arb_display_only": str(value),
        "lower_exact_dyadic": _endpoint(value.lower()),
        "upper_exact_dyadic": _endpoint(value.upper()),
    }


def endpoint_fraction(data):
    mantissa = int(data["mantissa"])
    exponent = int(data["exponent_of_two"])
    if exponent >= 0:
        return Fraction(mantissa << exponent)
    return Fraction(mantissa, 1 << -exponent)


def from_record(data):
    lower = endpoint_fraction(data["lower_exact_dyadic"])
    upper = endpoint_fraction(data["upper_exact_dyadic"])
    if lower > upper:
        raise ValueError("inverted path-audit endpoints")
    return aq(lower).union(aq(upper))


def _parameters(a, u, step, N, bits):
    a = tuple(exact(x) for x in a)
    u = tuple(u)
    if not a or len(a) != len(u):
        raise ValueError("nonempty equal-length a and u required")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in u):
        raise ValueError("u must contain exact Python integers")
    S = sum(x * x for x in u)
    if not S or sum(u) != 0:
        raise ValueError("nonzero integer u with sum(u)=0 required")
    if any(abs(x) > 1 for x in a):
        raise ValueError("the fixed strip proof requires every |a_i| <= 1")
    step = exact(step)
    if step <= 0 or isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("positive exact step and positive integer N required")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 64:
        raise ValueError("integer precision of at least 64 bits required")
    return a, u, S, step


def _label_path_terms(a, u, S, x, probabilities, label):
    """Return A_y and N_y=-A_y'' after analytically integrating Z2.

    Multiplication deliberately follows integer label bits, never a shared
    prefix table or the pair-kernel expression. All arguments except u/S/label
    are Arb values. u is kept unnormalised to avoid a needless square root.
    """
    m = len(a)
    product = arb(1)
    first = arb(0)
    log_second_variance = arb(0)
    path_acceleration = arb(0)
    for j in range(m):
        p = probabilities[j]
        y = (label >> (m - 1 - j)) & 1
        if y:
            product *= p
            residual = 1 - p
        else:
            product *= 1 - p
            residual = -p
        au = a[j] * u[j]
        first += au * residual
        log_second_variance += au * au * p * (1 - p)
        path_acceleration += a[j] * (u[j] * u[j]) * residual
    Q2 = product * (first * first - log_second_variance - x * path_acceleration) / S
    return product, -Q2


def finite_path_audit(a, u, *, step="1/8", N=88, bits=192, hard_seconds=600):
    """Retain every finite Q and -Q'' interval without analytic enlargement.

    Production parameters are m=8, step=1/8 and N=88 (177 full-weight nodes).
    The generic m argument supports small manufactured verification cases.
    The time budget is checked between complete nodes; an over-budget result,
    including one that just finished its final node, is never ``completed``.
    """
    a, u, S, step = _parameters(a, u, step, N, bits)
    if (isinstance(hard_seconds, bool) or not isinstance(hard_seconds, (int, float))
            or not math.isfinite(hard_seconds) or hard_seconds <= 0):
        raise ValueError("finite positive wall-clock budget required")
    previous = ctx.prec
    ctx.prec = bits
    try:
        m, label_count = len(a), 1 << len(a)
        aa, dx = tuple(aq(x) for x in a), aq(step)
        Q = [arb(0) for _ in range(label_count)]
        numerator = [arb(0) for _ in range(label_count)]
        normalizer = (2 * arb.pi()).sqrt()
        completed = 0
        start = time.perf_counter()
        for node in range(-N, N + 1):
            if time.perf_counter() - start >= hard_seconds:
                break
            x = dx * node
            probabilities = tuple(1 / (1 + (-ai * x).exp()) for ai in aa)
            weight = dx * (-x * x / 2).exp() / normalizer
            for label in range(label_count):
                product, minus_second = _label_path_terms(aa, u, S, x, probabilities, label)
                Q[label] += weight * product
                numerator[label] += weight * minus_second
            completed += 1
        seconds = time.perf_counter() - start
        complete = completed == 2 * N + 1 and seconds <= hard_seconds
        return {
            "schema": "angular_witness_path_finite_sum_v1",
            "status": "completed" if complete else "budget_stopped",
            "precision_bits": bits,
            "m": m,
            "label_count": label_count,
            "label_order": "integer labels 0..2^m-1; observation 0 is MSB",
            "direction": "d_i = u_i / sqrt(sum_j u_j^2)",
            "u_integer": list(u),
            "S_integer": S,
            "zero_sum_u_checked": True,
            "step_exact": str(step),
            "N": N,
            "limit_exact": str(step * N),
            "endpoint_weights": "full",
            "nodes_prescribed": 2 * N + 1,
            "nodes_completed": completed,
            "Q_finite_sum": [record(value) for value in Q],
            "N_path_finite_sum": [record(value) for value in numerator],
            "numerator_sign": "N_path = -d^2 Q_y(theta(t))/dt^2 at t=0",
            "path_acceleration_included": True,
            "Z2_moments_integrated_analytically": {"E_Z2": 0, "E_Z2_squared": 1},
            "wall_seconds": seconds,
            "hard_budget_seconds": hard_seconds,
            "budget_check_granularity": "between complete nodes and after final node",
            "real_sigmoid_evaluations": completed * m,
            "real_Gaussian_density_evaluations": completed,
            "label_probability_factor_multiplications": completed * label_count * m,
            "label_path_row_contributions": completed * label_count * m,
            "label_probability_accumulations": completed * label_count,
            "label_directional_accumulations": completed * label_count,
            "finite_scalar_integrals_completed": 2 * label_count if complete else 0,
            "primary_pair_kernel_calls": 0,
            "new_kernel_calls": 0,
            "new_random_draws": 0,
            "new_optimizer_updates": 0,
            "scientific_NPZ_reads": 0,
            "finite_sum_only": True,
            "mass_renormalized": False,
        }
    finally:
        ctx.prec = previous


def path_bounds(a, u, *, step="1/8", N=88, bits=192):
    """Uniform analytic aliasing and omitted-grid-tail errors, with Amax=1.

    Only scalar Arb arithmetic is performed. These errors must be added to
    finite balls before comparing the path and pair integral representations.
    """
    a, u, S, step = _parameters(a, u, step, N, bits)
    limit = step * N
    if limit < 1:
        raise ValueError("the monotone x*phi(x) tail argument requires N*step >= 1")
    previous = ctx.prec
    ctx.prec = bits
    try:
        m = len(a)
        c = 1 / aq(Fraction(1, 2)).cos()
        C = c ** m
        M_Q = aq(Fraction(1, 2)).exp() * C
        M_N = M_Q * (m * c * c + c * c / 4 + c * arb(2).sqrt())
        denominator = (2 * arb.pi() / aq(step)).exp() - 1
        phi_limit = (-aq(limit) * aq(limit) / 2).exp() / (2 * arb.pi()).sqrt()
        Q_alias = 2 * M_Q / denominator
        N_alias = 2 * M_N / denominator
        Q_tail = 2 * phi_limit / aq(limit)
        N_tail = (m + aq(Fraction(1, 4))) * Q_tail + 2 * phi_limit
        return {
            "schema": "angular_witness_path_bounds_v1",
            "precision_bits": bits,
            "m": m,
            "step_exact": str(step),
            "N": N,
            "node_count": 2 * N + 1,
            "limit_exact": str(limit),
            "strip_exact": "1",
            "max_abs_a_cap_exact": "1",
            "direction_squared_norm_exact": "1",
            "S_integer": S,
            "endpoint_weights": "full",
            "label_uniform": True,
            "constants": {
                "c_sec_half": record(c),
                "C_product_cap": record(C),
                "boundary_Q_single_side": record(M_Q),
                "boundary_N_path_single_side": record(M_N),
            },
            "errors": {
                "Q_alias": record(Q_alias),
                "Q_grid_tail": record(Q_tail),
                "Q_total": record(Q_alias + Q_tail),
                "N_path_alias": record(N_alias),
                "N_path_grid_tail": record(N_tail),
                "N_path_total": record(N_alias + N_tail),
            },
            "finite_sum_arithmetic_included_separately_in_saved_balls": True,
            "new_integrals": 0,
        }
    finally:
        ctx.prec = previous


def _manufactured_factor_jet(a, u, z, p, label):
    """Rational factor-product differentiation, independent of log derivatives."""
    value = Fraction(1)
    first_z2 = second_constant = second_z2_squared = Fraction(0)
    m = len(a)
    for j in range(m):
        y = (label >> (m - j - 1)) & 1
        sign = 1 if y else -1
        f = p[j] if y else 1 - p[j]
        variance = p[j] * (1 - p[j])
        first = sign * variance * a[j] * u[j]
        second0 = -sign * variance * a[j] * u[j] ** 2 * z
        second2 = sign * variance * (1 - 2 * p[j]) * (a[j] * u[j]) ** 2
        second_constant, second_z2_squared = (
            second_constant * f + value * second0,
            second_z2_squared * f + 2 * first_z2 * first + value * second2,
        )
        first_z2 = first_z2 * f + value * first
        value *= f
    return value, -(second_constant + second_z2_squared) / sum(x * x for x in u)


def self_test():
    """Small algebra/API checks; never runs the target scientific grid."""
    a = tuple(map(Fraction, ("1/3", "-1/2", "2/5")))
    u, z = (1, 2, -3), Fraction(2, 3)
    probabilities = tuple(map(Fraction, ("1/5", "2/3", "3/7")))
    S = sum(x * x for x in u)
    old_precision = ctx.prec
    ctx.prec = 192
    try:
        exact_mass = exact_second = Fraction(0)
        for label in range(8):
            expected_q, expected_n = _manufactured_factor_jet(a, u, z, probabilities, label)
            residuals = [((label >> (2 - j)) & 1) - probabilities[j] for j in range(3)]
            first = sum((a[j] * u[j] * residuals[j] for j in range(3)), Fraction(0))
            variance = sum((a[j] ** 2 * u[j] ** 2 * probabilities[j] * (1 - probabilities[j])
                            for j in range(3)), Fraction(0))
            acceleration = z * sum((a[j] * u[j] ** 2 * residuals[j] for j in range(3)), Fraction(0))
            assert expected_n == -expected_q * (first ** 2 - variance - acceleration) / S
            q, n = _label_path_terms(tuple(map(aq, a)), u, S, aq(z), tuple(map(aq, probabilities)), label)
            for actual, expected in ((q, expected_q), (n, expected_n)):
                saved = record(actual)
                assert endpoint_fraction(saved["lower_exact_dyadic"]) <= expected
                assert expected <= endpoint_fraction(saved["upper_exact_dyadic"])
                assert from_record(saved).contains(actual)
            exact_mass += expected_q
            exact_second += expected_n
        assert exact_mass == 1 and exact_second == 0
        # At a=0 every conditional probability is 1/2 and the path is constant.
        for label in range(8):
            q, n = _label_path_terms((arb(0),) * 3, u, S, aq(z), (aq("1/2"),) * 3, label)
            assert q == aq("1/8") and n == 0
        ctx.prec = 89
        finite = finite_path_audit(a, u, step="1/2", N=1, bits=192, hard_seconds=10)
        assert ctx.prec == 89 and finite["status"] == "completed"
        assert finite["nodes_completed"] == 3 and finite["finite_scalar_integrals_completed"] == 16
        assert finite["label_probability_factor_multiplications"] == 72
        ctx.prec = 192
        Q = [from_record(item) for item in finite["Q_finite_sum"]]
        N_path = [from_record(item) for item in finite["N_path_finite_sum"]]
        gaussian_mass = sum((aq("1/2") * (-(aq(k) / 2) ** 2 / 2).exp() / (2 * arb.pi()).sqrt()
                             for k in (-1, 0, 1)), arb(0))
        assert sum(Q, arb(0)).overlaps(gaussian_mass)
        assert sum(N_path, arb(0)).contains(0)
        assert all(Q[j].overlaps(Q[7-j]) and N_path[j].overlaps(N_path[7-j]) for j in range(8))
        stopped = finite_path_audit(a, u, step="1/2", N=1, hard_seconds=1e-30)
        assert stopped["status"] == "budget_stopped" and stopped["nodes_completed"] == 0
        assert stopped["finite_scalar_integrals_completed"] == 0
        # Scalar analytic bounds, not a new integral or target grid.
        ctx.prec = 89
        budget = path_bounds(a, u, step="1/2", N=2, bits=192)
        assert ctx.prec == 89
        assert all(from_record(item) > 0 for item in budget["errors"].values())
        rejected = 0
        for invalid_a, invalid_u in ((a, (1, 1, 1)), (a, (0, 0, 0)),
                                     (a, (1.0, 2, -3)), ((1.1, 0, 0), u), ((2, 0, 0), u)):
            try:
                finite_path_audit(invalid_a, invalid_u, N=1)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("invalid exact inputs accepted")
        assert rejected == 5
        return {
            "status": "passed",
            "manufactured_m": 3,
            "factor_jet_labels_checked": 8,
            "exact_factor_jet_matches_log_derivative_path": True,
            "exact_probability_sum": "1",
            "exact_path_second_sum": "0",
            "zero_load_constant_path_labels_checked": 8,
            "small_grid_nodes_completed": 3,
            "small_grid_labels": 8,
            "label_complement_symmetry_checked": True,
            "finite_gaussian_mass_and_zero_path_mass_checked": True,
            "zero_node_budget_stop_checked": True,
            "precision_restoration_checked": True,
            "invalid_inputs_rejected": rejected,
            "target_scientific_nodes_completed": 0,
            "primary_pair_kernel_imported": False,
        }
    finally:
        ctx.prec = old_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", required=True, action="store_true")
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
