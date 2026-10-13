"""Strict teacher PMF and first angular derivative; no execution on import.

No historical producer, private cache, optimizer, RNG, or scientific files are
imported/read. Only --self-test is executable. A scientific caller must freeze
its own inputs before calling finite_grid_block. All returned data are JSON.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import time

import flint
from flint import arb, ctx


PHI = ("-7/10", "-1/2", "-3/10", "-1/10", "1/10", "3/10", "1/2", "7/10")
DEFAULT_CONFIG = dict(h="1/8", N=80, strip="1", amplitude="4/5",
                      base_angles=list(PHI), precision_bits=192)


def exact(value):
    if isinstance(value, (float, bool)):
        raise ValueError("exact rational string/int/Fraction required")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return dict(mantissa=str(mantissa), exponent_of_two=int(exponent))


def endpoint_fraction(value):
    if type(value["exponent_of_two"]) is not int:
        raise ValueError("integer dyadic exponent required")
    return Fraction(int(value["mantissa"])) * Fraction(2) ** value["exponent_of_two"]


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite interval")
    return dict(arb_display_only=str(value), lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))


def from_record(value):
    lo = endpoint_fraction(value["lower_exact_dyadic"])
    hi = endpoint_fraction(value["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("reversed endpoints")
    return aq(lo).union(aq(hi))


def configuration(config=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["base_angles"] = list(PHI)
    if config is not None:
        if set(config) - set(cfg):
            raise ValueError("unknown teacher configuration field")
        cfg.update(config)
    for key in ("h", "strip", "amplitude"):
        cfg[key] = str(exact(cfg[key]))
        if exact(cfg[key]) <= 0:
            raise ValueError("positive step, strip, amplitude required")
    if type(cfg["N"]) is not int or cfg["N"] < 1:
        raise ValueError("positive integer N required")
    if type(cfg["precision_bits"]) is not int or cfg["precision_bits"] < 64:
        raise ValueError("integer precision_bits>=64 required")
    cfg["base_angles"] = [str(exact(x)) for x in cfg["base_angles"]]
    if not 2 <= len(cfg["base_angles"]) <= 8:
        raise ValueError("two to eight teacher rows required")
    if exact(cfg["h"]) * cfg["N"] < 1:
        raise ValueError("L=N*h must be at least one for the stated tail proof")
    return cfg


def family(lambda_centre="0", config=None):
    cfg = configuration(config)
    lam = exact(lambda_centre)
    angles = [exact(x) for x in cfg["base_angles"]]
    angles[0] += lam
    return dict(config=cfg, lambda_centre_exact=str(lam),
                effective_angles_exact=[str(x) for x in angles],
                label_order="integer MSB is first teacher row; bit zero/one is y_i=0/1",
                bias_exact=["0"] * len(angles))


def analytic_bounds(lambda_centre="0", config=None):
    """Scalar analytic total L1 errors; evaluates no scientific target grid."""
    spec = family(lambda_centre, config)
    cfg = spec["config"]
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    try:
        h, eta, rho = [aq(cfg[x]) for x in ("h", "strip", "amplitude")]
        L = h * cfg["N"]
        secants, rows, trig = [], [], []
        for angle in spec["effective_angles_exact"]:
            theta = aq(angle)
            co, si = theta.cos(), theta.sin()
            beta = rho * eta * (abs(co) + abs(si)) / 2
            if not beta < arb.pi() / 2 or not beta.cos() > 0:
                raise ValueError("pole-free positive secant strip not established")
            secants.append(1 / beta.cos())
            trig.append((co, si))
            rows.append(dict(angle_exact=angle, beta=record(beta), secant=record(secants[-1])))
        B0 = arb(1)
        for secant in secants:
            B0 *= secant
        abs_normal_mean = (2 / arb.pi()).sqrt()
        first_tangent_l1 = abs(trig[0][0]) + abs(trig[0][1])
        B1 = rho / 2 * secants[0] ** 2 * first_tangent_l1 * (abs_normal_mean + eta)
        for secant in secants[1:]:
            B1 *= secant
        M0, M1 = (eta * eta).exp() * B0, (eta * eta).exp() * B1
        alias_ratio = (-2 * arb.pi() * eta / h).exp()
        alias_factor = 4 * alias_ratio / (1 - alias_ratio) ** 2
        phi0 = 1 / (2 * arb.pi()).sqrt()
        phi1 = (-aq("1/2")).exp() * phi0
        phiL = (-L * L / 2).exp() * phi0
        S0 = 1 + h * phi0
        S1 = abs_normal_mean + 4 * h * phi1
        T0, T1 = 2 * phiL / L, 2 * phiL
        q_alias, prime_alias = M0 * alias_factor, M1 * alias_factor
        q_tail = 2 * S0 * T0
        prime_tail = rho / 2 * first_tangent_l1 * (T1 * S0 + T0 * S1)
        q_error, prime_error = q_alias + q_tail, prime_alias + prime_tail
        qprime_l1_cap = rho / 2 * abs_normal_mean
        qsecond_l1_cap = rho ** 2 / (3 * arb(3).sqrt()) + qprime_l1_cap
        names = dict(B0=B0, B1=B1, shifted_integral_M0=M0, shifted_integral_M1=M1,
                     alias_ratio=alias_ratio, tensor_alias_factor=alias_factor,
                     whole_grid_mass_S0=S0, whole_grid_absolute_first_moment_S1=S1,
                     omitted_grid_mass_T0=T0, omitted_grid_absolute_first_moment_T1=T1,
                     q_alias_L1=q_alias, prime_alias_L1=prime_alias,
                     q_tail_L1=q_tail, prime_tail_L1=prime_tail,
                     q_total_L1=q_error, prime_total_L1=prime_error,
                     qprime_global_L1_cap=qprime_l1_cap, qsecond_global_L1_cap=qsecond_l1_cap)
        return dict(schema="r03_teacher_derivative_analytic_bounds_v1", family=spec,
                    rows=rows, bounds={key: record(value) for key, value in names.items()},
                    total_errors_below_1e_15=bool(q_error < aq("1e-15") and prime_error < aq("1e-15")),
                    tensor_alias_formula="4*r/(1-r)^2 = ((1+2*r/(1-r))^2-1)",
                    derivative_index=1, gaussian_dimension=2, label_factor_used=False,
                    finite_sum_arithmetic_included=False,
                    new_scientific_grid_points=0, new_target_integrals=0,
                    python_flint_version=flint.__version__)
    finally:
        ctx.prec = previous


def _point_vectors(probabilities, eta_prime):
    """First-factor derivative; labels use the first row as the MSB."""
    tail = [arb(1)]
    for p in probabilities[1:]:
        tail = [child for value in tail for child in (value * (1 - p), value * p)]
    p = probabilities[0]
    pprime = p * (1 - p) * eta_prime
    return ([(1 - p) * value for value in tail] + [p * value for value in tail],
            [-pprime * value for value in tail] + [pprime * value for value in tail])


def _accumulate(spec, inner_indices, outer_indices, deadline=None, manufactured=False):
    cfg = spec["config"]
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    tick = time.perf_counter()
    try:
        m = len(spec["effective_angles_exact"])
        step, rho = aq(cfg["h"]), aq(cfg["amplitude"])
        trig = [(aq(x).cos(), aq(x).sin()) for x in spec["effective_angles_exact"]]
        indices = sorted(set(inner_indices) | set(outer_indices))
        coords = {i: step * i for i in indices}
        weights = {i: step * (-coords[i] ** 2 / 2).exp() / (2 * arb.pi()).sqrt() for i in indices}
        q, prime = [arb(0) for _ in range(1 << m)], [arb(0) for _ in range(1 << m)]
        mass, points, completed_rows = arb(0), 0, []
        completed, last = True, None
        for i in outer_indices:
            row_complete = True
            for j in inner_indices:
                if deadline is not None and time.perf_counter() >= deadline:
                    completed = row_complete = False
                    break
                x, y = coords[i], coords[j]
                weight = weights[i] * weights[j]
                ps = [1 / (1 + (-rho * (co * x + si * y)).exp()) for co, si in trig]
                eta_prime = rho * (-trig[0][1] * x + trig[0][0] * y)
                qpoint, dpoint = _point_vectors(ps, eta_prime)
                for label in range(1 << m):
                    q[label] += weight * qpoint[label]
                    prime[label] += weight * dpoint[label]
                mass += weight
                points += 1
                last = [i, j]
            if row_complete:
                completed_rows.append(i)
            if not completed:
                break
        return dict(schema="r03_teacher_derivative_finite_block_v1", family=spec,
                    inner_indices=list(inner_indices), outer_indices=list(outer_indices),
                    completed_outer_indices=completed_rows, completed=completed,
                    last_completed_point=last, q_finite=[record(x) for x in q],
                    prime_finite=[record(x) for x in prime], grid_gaussian_mass=record(mass),
                    renormalized=False, true_integrals_enclosed=False,
                    manufactured=manufactured, elapsed_seconds=time.perf_counter() - tick,
                    work_counts=dict(grid_points=points, q_components=1 << m,
                                     prime_components=1 << m, logistic_exponentials=points * m,
                                     gaussian_1d_exponentials=len(indices),
                                     scientific_grid_points=0 if manufactured else points,
                                     manufactured_grid_points=points if manufactured else 0,
                                     random_draws=0, optimizer_updates=0))
    finally:
        ctx.prec = previous


def finite_grid_block(lambda_centre="0", outer_indices=None, config=None, deadline=None):
    """Compute one fresh finite-sum block; caller freezes and stores provenance.

    deadline, if provided, is an absolute time.perf_counter() deadline. Partial
    results are returned with completed=False; no retry or resume occurs here.
    """
    spec = family(lambda_centre, config)
    N = spec["config"]["N"]
    inner = tuple(range(-N, N + 1))
    outer = inner if outer_indices is None else tuple(outer_indices)
    if (not outer or any(type(x) is not int or not -N <= x <= N for x in outer)
            or len(set(outer)) != len(outer)):
        raise ValueError("distinct integer outer indices in [-N,N] required")
    return _accumulate(spec, inner, outer, deadline)


def aggregate_blocks(blocks):
    """Saved finite sums only; require exactly one complete full-weight square."""
    if not blocks:
        raise ValueError("nonempty blocks required")
    spec = blocks[0]["family"]
    if spec != family(spec["lambda_centre_exact"], spec["config"]):
        raise ValueError("family metadata drift")
    cfg, m = spec["config"], len(spec["effective_angles_exact"])
    expected = list(range(-cfg["N"], cfg["N"] + 1))
    outer = []
    for block in blocks:
        if (block["family"] != spec or block.get("completed") is not True
                or block.get("manufactured") is not False
                or block["inner_indices"] != expected
                or block["completed_outer_indices"] != block["outer_indices"]
                or block.get("renormalized") is not False):
            raise ValueError("incomplete, manufactured, mismatched, or renormalized block")
        if (len(block["q_finite"]) != 1 << m or len(block["prime_finite"]) != 1 << m
                or block["work_counts"]["grid_points"] != len(expected) * len(block["outer_indices"])):
            raise ValueError("finite block count drift")
        outer.extend(block["outer_indices"])
    if sorted(outer) != expected:
        raise ValueError("outer rows must cover full square exactly once")
    previous = ctx.prec
    ctx.prec = cfg["precision_bits"]
    try:
        q = [sum((from_record(block["q_finite"][i]) for block in blocks), arb(0)) for i in range(1 << m)]
        prime = [sum((from_record(block["prime_finite"][i]) for block in blocks), arb(0)) for i in range(1 << m)]
        mass = sum((from_record(block["grid_gaussian_mass"]) for block in blocks), arb(0))
        mask = (1 << m) - 1
        checks = dict(probability_sum_overlaps_grid_mass=sum(q, arb(0)).overlaps(mass),
                      derivative_sum_contains_zero=sum(prime, arb(0)).contains(0),
                      q_complement_overlap=all(q[i].overlaps(q[mask - i]) for i in range(1 << m)),
                      prime_complement_overlap=all(prime[i].overlaps(prime[mask - i]) for i in range(1 << m)))
        if not all(checks.values()):
            raise ArithmeticError("finite teacher identity failed")
        return dict(schema="r03_teacher_derivative_finite_v1", family=spec,
                    q_finite=[record(x) for x in q], prime_finite=[record(x) for x in prime],
                    grid_gaussian_mass=record(mass), checks=checks,
                    renormalized=False, true_integrals_enclosed=False,
                    completed=True, manufactured=False,
                    work_counts=dict(grid_points=len(expected) ** 2, scalar_finite_sums=2 * (1 << m),
                                     blocks=len(blocks), random_draws=0, optimizer_updates=0))
    finally:
        ctx.prec = previous


def _expand(value, radius):
    return (value - radius.upper()).union(value + radius.upper())


def compose(finite, bounds=None):
    """Pure saved-arithmetic composition; no new teacher or candidate grid."""
    if (finite.get("schema") != "r03_teacher_derivative_finite_v1"
            or finite.get("completed") is not True or finite.get("manufactured") is not False
            or finite.get("renormalized") is not False):
        raise ValueError("complete unnormalized aggregate required")
    spec = finite["family"]
    if spec != family(spec["lambda_centre_exact"], spec["config"]):
        raise ValueError("family metadata drift")
    if bounds is None:
        bounds = analytic_bounds(spec["lambda_centre_exact"], spec["config"])
    if bounds.get("family") != spec or bounds.get("schema") != "r03_teacher_derivative_analytic_bounds_v1":
        raise ValueError("bound/finite family mismatch")
    previous = ctx.prec
    ctx.prec = spec["config"]["precision_bits"]
    try:
        q, prime = [[from_record(x) for x in finite[key]] for key in ("q_finite", "prime_finite")]
        m = len(spec["effective_angles_exact"])
        if len(q) != 1 << m or len(prime) != 1 << m:
            raise ValueError("all original labels required")
        eq = from_record(bounds["bounds"]["q_total_L1"])
        ep = from_record(bounds["bounds"]["prime_total_L1"])
        qt, pt = [_expand(x, eq) for x in q], [_expand(x, ep) for x in prime]
        norm = _expand(sum((abs(x) for x in prime), arb(0)), ep)
        norm_lo = norm.lower()
        if norm_lo < 0:
            norm_lo = arb(0)
        norm = norm_lo.union(norm.upper())
        pairs = []
        for j in range(1, m):
            def coefficient(label):
                first = 2 * ((label >> (m - 1)) & 1) - 1
                other = 2 * ((label >> (m - 1 - j)) & 1) - 1
                return first * other
            qm = sum((coefficient(i) * q[i] for i in range(1 << m)), arb(0))
            dm = sum((coefficient(i) * prime[i] for i in range(1 << m)), arb(0))
            pairs.append(dict(rows_one_based=[1, j + 1],
                              spin_moment_true=record(_expand(qm, eq)),
                              derivative_true=record(_expand(dm, ep))))
        L2 = from_record(bounds["bounds"]["qsecond_global_L1_cap"])
        standard_rho = exact(spec["config"]["amplitude"]) == Fraction(4, 5)
        if standard_rho and not L2 < aq("0.442323"):
            raise ArithmeticError("convenient Taylor remainder constant not established")
        return dict(schema="r03_teacher_derivative_certificate_v1", family=spec,
                    q_true_component_intervals=[record(x) for x in qt],
                    prime_true_component_intervals=[record(x) for x in pt],
                    analytic_L1_errors=dict(q=record(eq), prime=record(ep)),
                    component_policy="each component receives the same global L1 error; it is not a per-label error to multiply by label count",
                    finite_sum_arithmetic_preserved=True, renormalized=False,
                    prime_true_L1=record(norm), first_row_spin_pairs=pairs,
                    sum_qprime_exact="0", all_single_spin_derivatives_exact="0",
                    qprime_global_L1_cap=bounds["bounds"]["qprime_global_L1_cap"],
                    qsecond_global_L1_cap=record(L2),
                    taylor_remainder_L1=dict(formula="0.5*L2*(lambda-lambda_centre)^2",
                                            half_L2=record(L2 / 2),
                                            convenient_half_L2_exact="0.2211615" if standard_rho else None,
                                            valid_for_all_real_lambda=True),
                    new_scientific_grid_points=0, new_target_integrals=0)
    finally:
        ctx.prec = previous


def self_test():
    """Manufactured algebra/2x2 grid only; never calls the study finite grid."""
    checks = {}
    p = [Fraction(2, 7), Fraction(3, 8), Fraction(4, 9), Fraction(5, 11),
         Fraction(6, 13), Fraction(7, 15), Fraction(8, 17), Fraction(9, 19)]
    eta = Fraction(7, 23)
    values, derivatives = [], []
    for label in range(256):
        bits = [(label >> (7 - i)) & 1 for i in range(8)]
        factors = [p[i] if bits[i] else 1 - p[i] for i in range(8)]
        tail = Fraction(1)
        for factor in factors[1:]:
            tail *= factor
        value = factors[0] * tail
        deriv = (2 * bits[0] - 1) * p[0] * (1 - p[0]) * eta * tail
        if deriv != value * (bits[0] - p[0]) * eta:
            raise AssertionError("independent score formula mismatch")
        values.append(value)
        derivatives.append(deriv)
    checks["fraction_all256_mass_one"] = sum(values) == 1
    checks["fraction_all256_derivative_mass_zero"] = sum(derivatives) == 0
    checks["fraction_other_seven_joint_marginal_derivative_zero"] = all(
        derivatives[i] + derivatives[128 + i] == 0 for i in range(128))
    checks["fraction_missing_label_sign_negative_control_fails"] = sum(abs(x) for x in derivatives) > 0
    checks["fraction_spinpair_derivatives"] = all(
        sum((2 * ((label >> 7) & 1) - 1) * (2 * ((label >> (7 - j)) & 1) - 1) * derivatives[label]
            for label in range(256)) == 2 * p[0] * (1 - p[0]) * eta * (2 * p[j] - 1)
        for j in range(1, 8))
    mirrored = []
    for label in range(256):
        bits = [(label >> (7 - i)) & 1 for i in range(8)]
        tail = Fraction(1)
        for i in range(1, 8):
            tail *= 1 - p[i] if bits[i] else p[i]
        mirrored.append((2 * bits[0] - 1) * p[0] * (1 - p[0]) * (-eta) * tail)
    symmetric_prime = [(x + y) / 2 for x, y in zip(derivatives, mirrored)]
    checks["fraction_all256_reflected_point_derivative"] = all(
        derivatives[i] == mirrored[255 - i] for i in range(256))
    checks["fraction_all256_complement_derivative"] = all(
        symmetric_prime[i] == symmetric_prime[255 - i] for i in range(256))
    checks["fraction_all_eight_single_spin_derivatives_zero"] = all(
        sum((2 * ((label >> (7 - j)) & 1) - 1) * symmetric_prime[label] for label in range(256)) == 0
        for j in range(8))
    test_config = dict(h="1/2", N=2, strip="1/2", amplitude="2/3",
                       base_angles=["-1/3", "2/5", "1/7"], precision_bits=128)
    spec = family("1/11", test_config)
    block = _accumulate(spec, (-1, 1), (-1, 1), manufactured=True)
    previous = ctx.prec
    ctx.prec = 128
    try:
        q, prime = [[from_record(x) for x in block[key]] for key in ("q_finite", "prime_finite")]
        direct_q, direct_d = [arb(0) for _ in range(8)], [arb(0) for _ in range(8)]
        rho, h = aq("2/3"), aq("1/2")
        trig = [(aq(x).cos(), aq(x).sin()) for x in spec["effective_angles_exact"]]
        for i in (-1, 1):
            for j in (-1, 1):
                x, y = h * i, h * j
                weight = h * h * (-(x * x + y * y) / 2).exp() / (2 * arb.pi())
                ps = [(1 + (rho * (co * x + si * y) / 2).tanh()) / 2 for co, si in trig]
                eta_prime = rho * (-trig[0][1] * x + trig[0][0] * y)
                for label in range(8):
                    bits = [(label >> (2 - k)) & 1 for k in range(3)]
                    product = arb(1)
                    for k in range(3):
                        product *= ps[k] if bits[k] else 1 - ps[k]
                    direct_q[label] += weight * product
                    direct_d[label] += weight * product * (bits[0] - ps[0]) * eta_prime
        checks["manufactured_2x2_independent_tanh_q_overlap"] = all(a.overlaps(b) for a, b in zip(q, direct_q))
        checks["manufactured_2x2_independent_score_derivative_overlap"] = all(a.overlaps(b) for a, b in zip(prime, direct_d))
        checks["manufactured_2x2_complement_derivative"] = all(prime[i].overlaps(prime[7 - i]) for i in range(8))
        checks["manufactured_2x2_derivative_zero"] = sum(prime, arb(0)).contains(0)
        checks["manufactured_2x2_mass"] = sum(q, arb(0)).overlaps(from_record(block["grid_gaussian_mass"]))
        r = Fraction(3, 17)
        checks["fraction_tensor_alias_identity"] = (1 + 2 * r / (1 - r)) ** 2 - 1 == 4 * r / (1 - r) ** 2
    finally:
        ctx.prec = previous
    scalar_bounds = analytic_bounds()
    checks["default_analytic_errors_below_1e_15"] = scalar_bounds["total_errors_below_1e_15"]
    for name, call in (("reject_float_lambda", lambda: family(0.0)),
                       ("reject_short_tail_domain", lambda: configuration(dict(h="1/8", N=1))),
                       ("reject_partial_manufactured_aggregate", lambda: aggregate_blocks([block]))):
        try:
            call()
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    return dict(schema="r03_teacher_derivative_manufactured_checks_v1", passed=True, checks=checks,
                default_analytic_L1_errors={name: scalar_bounds["bounds"][name] for name in ("q_total_L1", "prime_total_L1")},
                manufactured_grid_points=4, manufactured_dimension=3,
                fraction_label_count=256, scientific_target_grid_points=0,
                new_scientific_target_integrals=0, scientific_data_reads=0,
                python_flint_version=flint.__version__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False, indent=2, allow_nan=False))
