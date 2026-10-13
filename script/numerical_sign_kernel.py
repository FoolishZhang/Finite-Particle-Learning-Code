"""Outward binary64 enclosures for the exact saved-input n256 Price kernel.

No RNG, files, scientific-bank loading, or library exp/log calls occur here.
Correctness is conditional on separately executed correctly rounded IEEE
binary64 basic operations, gradual underflow, nextafter, and ldexp. Fixed
fixtures check implementation behavior; they are not the uniform proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PAIRS = ((0, 1), (0, 2), (1, 2))
C_PI = 29
C_Q = 35
TAYLOR_DEGREE = 20
TAIL_CUTOFF = -800.0
MIN_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")
LN2_LO = float.fromhex("0x1.62e42fefa39efp-1")
LN2_HI = math.nextafter(LN2_LO, math.inf)
INV_LN2_HINT = 1.4426950408889634  # Only chooses k; the resulting r is checked.
REMAINDER_EXACT = Fraction(3, math.factorial(21))
REMAINDER_UPPER = math.nextafter(float(REMAINDER_EXACT), math.inf)
assert Fraction.from_float(REMAINDER_UPPER) >= REMAINDER_EXACT


@dataclass(frozen=True, slots=True)
class Interval:
    lo: np.ndarray
    hi: np.ndarray


def point(value: Any) -> Interval:
    x = np.asarray(value, dtype=np.float64)
    return Interval(x, x)


def down(x: Any) -> np.ndarray:
    return np.nextafter(np.asarray(x, dtype=np.float64), -np.inf)


def up(x: Any) -> np.ndarray:
    return np.nextafter(np.asarray(x, dtype=np.float64), np.inf)


def add(a: Interval, b: Interval) -> Interval:
    return Interval(down(a.lo + b.lo), up(a.hi + b.hi))


def neg(a: Interval) -> Interval:
    return Interval(-a.hi, -a.lo)


def sub(a: Interval, b: Interval) -> Interval:
    return add(a, neg(b))


def mul(a: Interval, b: Interval) -> Interval:
    # Four explicitly materialized products; no fused multiply-add/reduction.
    p00 = a.lo * b.lo
    p01 = a.lo * b.hi
    p10 = a.hi * b.lo
    p11 = a.hi * b.hi
    return Interval(
        down(np.minimum(np.minimum(p00, p01), np.minimum(p10, p11))),
        up(np.maximum(np.maximum(p00, p01), np.maximum(p10, p11))),
    )


def positive_div(a: Interval, b: Interval) -> Interval:
    if np.any(b.lo <= 0) or not np.all(np.isfinite(b.hi)):
        raise ArithmeticError("division requires a finite strictly positive denominator interval")
    return Interval(
        down(np.minimum(a.lo / b.lo, a.lo / b.hi)),
        up(np.maximum(a.hi / b.lo, a.hi / b.hi)),
    )


def known_range(a: Interval, lower: float, upper: float) -> Interval:
    """Intersect with an independently proved exact range, never a guess."""
    lo = np.maximum(a.lo, lower)
    hi = np.minimum(a.hi, upper)
    if np.any(lo > hi):
        raise ArithmeticError("computed enclosure misses a proved exact range")
    return Interval(lo, hi)


def take(a: Interval, key: Any) -> Interval:
    return Interval(a.lo[key], a.hi[key])


def pairwise_sum(a: Interval, axis: int = -1, keepdims: bool = False) -> Interval:
    """Explicit pairwise outward addition; never delegates summation to np.sum."""
    lo = np.moveaxis(a.lo, axis, -1)
    hi = np.moveaxis(a.hi, axis, -1)
    if lo.shape[-1] < 1:
        raise ValueError("empty summation axis")
    while lo.shape[-1] > 1:
        count = lo.shape[-1]
        paired = count - count % 2
        new_lo = down(lo[..., :paired:2] + lo[..., 1:paired:2])
        new_hi = up(hi[..., :paired:2] + hi[..., 1:paired:2])
        if count % 2:
            new_lo = np.concatenate((new_lo, lo[..., -1:]), axis=-1)
            new_hi = np.concatenate((new_hi, hi[..., -1:]), axis=-1)
        lo, hi = new_lo, new_hi
    if keepdims:
        return Interval(np.moveaxis(lo, -1, axis), np.moveaxis(hi, -1, axis))
    return Interval(lo[..., 0], hi[..., 0])


def exp_nonpositive(x: Interval) -> Interval:
    """Enclose exp(x) for finite nonpositive interval inputs, without exp/log.

    k is just a reduction hint. Every reduced interval is checked in [-1,1].
    The degree20 Taylor remainder is at most e/21! < 3/21! there.
    """
    if not np.all(np.isfinite(x.lo)) or not np.all(np.isfinite(x.hi)):
        raise ValueError("exp interval input must be finite")
    if np.any(x.lo > x.hi) or np.any(x.hi > 0):
        raise ValueError("exp interval input must be ordered and nonpositive")
    work = Interval(np.maximum(x.lo, TAIL_CUTOFF), np.maximum(x.hi, TAIL_CUTOFF))
    midpoint_hint = (work.lo + work.hi) * 0.5
    k = np.rint(midpoint_hint * INV_LN2_HINT).astype(np.int32)
    reduced = sub(work, mul(point(k), Interval(np.asarray(LN2_LO), np.asarray(LN2_HI))))
    if np.any(reduced.lo < -1) or np.any(reduced.hi > 1):
        raise ArithmeticError("Taylor reduction interval left [-1,1]")
    polynomial = point(np.ones_like(work.lo))
    for j in range(TAYLOR_DEGREE, 0, -1):
        polynomial = add(point(1.0), positive_div(mul(reduced, polynomial), point(float(j))))
    polynomial = add(polynomial, Interval(np.asarray(-REMAINDER_UPPER), np.asarray(REMAINDER_UPPER)))
    if np.any(polynomial.lo <= 0):
        raise ArithmeticError("positive exponential lower bound lost")
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        lo = down(np.ldexp(polynomial.lo, k))
        hi = up(np.ldexp(polynomial.hi, k))
    lo = np.where(x.lo <= TAIL_CUTOFF, 0.0, lo)
    hi = np.where(x.hi <= TAIL_CUTOFF, MIN_SUBNORMAL, hi)
    return known_range(Interval(lo, hi), 0.0, 1.0)


def sigmoid_pair(z: np.ndarray) -> tuple[Interval, Interval]:
    """Return enclosures for sigma(z), sigma(-z), without subtracting from one."""
    t = exp_nonpositive(point(-np.abs(z)))
    denominator = add(point(1.0), t)
    large = known_range(positive_div(point(1.0), denominator), 0.0, 1.0)
    small = known_range(positive_div(t, denominator), 0.0, 1.0)
    positive = z >= 0
    sigma = Interval(np.where(positive, large.lo, small.lo), np.where(positive, large.hi, small.hi))
    q = Interval(np.where(positive, small.lo, large.lo), np.where(positive, small.hi, large.hi))
    return sigma, q


def curvature_from_probability_intervals(pi: Interval, q: Interval) -> Interval:
    """All three diagonal Price kernels; interval pi need not itself sum to one."""
    one = point(1.0)
    pi2 = mul(pi, pi)
    ompi = sub(one, pi)
    r = mul(q, sub(mul(point(2.0), q), one))
    columns = []
    for i, j in PAIRS:
        qi, qj = take(q, (..., i)), take(q, (..., j))
        ri, rj = take(r, (..., i)), take(r, (..., j))
        a = mul(qi, qj)
        aa = mul(a, a)
        A = pairwise_sum(mul(pi, a), axis=-1)
        B = pairwise_sum(mul(pi2, a), axis=-1)
        term1 = neg(mul(mul(pi, ompi), mul(ri, rj)))
        inner2 = add(mul(mul(qi, qi), rj), mul(mul(qj, qj), ri))
        term2 = mul(mul(mul(point(2.0), pi2), ompi), inner2)
        term3 = mul(mul(pi2, sub(one, mul(point(4.0), pi))), aa)
        local = pairwise_sum(add(add(term1, term2), term3), axis=-1)
        global_terms = add(sub(mul(A, A), mul(point(4.0), mul(A, B))), mul(point(6.0), mul(B, B)))
        columns.append(add(local, global_terms))
    result = Interval(np.stack([x.lo for x in columns], axis=-1), np.stack([x.hi for x in columns], axis=-1))
    if not np.all(np.isfinite(result.lo)) or not np.all(np.isfinite(result.hi)):
        raise ArithmeticError("non-finite curvature enclosure")
    return result


def bound_curvature(z: Any, s: Any) -> tuple[Interval, Interval]:
    """Return (three-pair, within-bank mean) intervals for exact binary64 Z/S.

    Shape contract: z=(B,K,3), s=(B,K); n256 only. All inputs are checked.
    A lost positive denominator is a failure, not a discarded bank.
    """
    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if z.ndim != 3 or z.shape[-1] != 3 or min(z.shape[:2]) < 1 or s.shape != z.shape[:2]:
        raise ValueError("expected z=(B,K,3), s=(B,K), B,K>=1")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(s)):
        raise ValueError("all saved inputs must be finite")
    sigma, q = sigmoid_pair(z)
    active = mul(mul(take(sigma, (..., 0)), take(sigma, (..., 1))), take(sigma, (..., 2)))
    shift = np.max(s, axis=1, keepdims=True)  # Selection of an exact saved value.
    relative = sub(point(s), point(shift))
    relative = known_range(relative, -np.inf, 0.0)
    nuisance = exp_nonpositive(relative)
    weights = known_range(mul(active, nuisance), 0.0, 1.0)
    denominator = pairwise_sum(weights, axis=1, keepdims=True)
    pi = known_range(positive_div(weights, denominator), 0.0, 1.0)
    pairs = curvature_from_probability_intervals(pi, q)
    average = positive_div(pairwise_sum(pairs, axis=1), point(3.0))
    return pairs, average


def manufactured_fixtures() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The contract's fixed inputs; no RNG and no scientific-array access."""
    rows = [
        ("K1_center", [[0., 0., 0.]], [0.]),
        ("K1_asymmetric", [[2., -3., 1.]], [7.]),
        ("K2_equal", [[.1, .8, -.5], [-.7, .2, 1.5]], [0., 0.]),
        ("K2_concentrated", [[.1, .8, -.5], [-.7, .2, 1.5]], [0., -9.]),
        ("K3_asymmetric", [[1.2, -.3, .7], [-.7, .8, -1.1], [.1, 1.5, -.2]], [0., -7., -.5]),
        ("K4_large_and_tail", [[32., -32., 0.], [-32., 32., .25], [0., 0., 0.], [-1000., 1000., -.25]], [0., -128., -512., -1000.]),
    ]
    k = np.arange(100, dtype=np.int64)
    z = np.stack(((k % 17 - 8) / 8., ((3*k) % 19 - 9) / 8., ((5*k) % 23 - 11) / 8.), axis=-1)
    s = -3. * ((17*k) % 29) / 4.
    result = [(name, np.asarray(z, dtype=np.float64), np.asarray(s, dtype=np.float64)) for name, z, s in rows]
    result.append(("K100_modular", z, s))
    return result


def qualification() -> dict[str, Any]:
    """Fixed-fixture correctness probes and three deterministic timing blocks."""
    private_tools = Path(__file__).resolve().parents[1] / ".cache" / "finite_selection_tools"
    if str(private_tools) not in sys.path:
        sys.path.insert(0, str(private_tools))
    from flint import arb, ctx
    from local_geometry_kernel import interval_curvature, price_curvature

    assert np.dtype(np.float64).itemsize == 8
    assert float(np.nextafter(np.float64(0), np.float64(np.inf))) == MIN_SUBNORMAL
    assert float(np.ldexp(np.float64(1), -1074)) == MIN_SUBNORMAL
    assert float(np.float64(MIN_SUBNORMAL) + np.float64(MIN_SUBNORMAL)) == 2 * MIN_SUBNORMAL
    assert float(np.ldexp(np.float64(1), -1022) * np.float64(.5)) > 0
    previous = ctx.prec
    ctx.prec = 256
    try:
        ln2 = arb(2).log()
        assert arb(LN2_LO) < ln2 < arb(LN2_HI)
        assert ln2 < arb(7) / 10
        exponent_inputs = np.asarray([-1000., -800., -799., -745., -710., -512., -128., -32., -1., -.7, -.5, -.01, -MIN_SUBNORMAL, 0.])
        ex = exp_nonpositive(point(exponent_inputs))
        exp_rows = []
        for x, lo, hi in zip(exponent_inputs, ex.lo, ex.hi):
            reference = arb(float(x)).exp()
            if -1e-70 < float(x) <= 0:
                # The 256-bit Arb ball around exp(-2^-1074) can straddle 1.
                # Exact convexity bounds resolve this known endpoint without
                # demanding that the returned ball itself lie below 1.
                assert Fraction.from_float(float(lo)) <= 1 + Fraction.from_float(float(x))
                assert float(hi) >= 1.0
                method = "exact_rational_1_plus_x_le_exp_x_le_1"
            else:
                assert arb(float(lo)) <= reference <= arb(float(hi)), (x, lo, hi, str(reference))
                method = "256_bit_Arb_containment"
            exp_rows.append(dict(x=float(x), lower=float(lo), upper=float(hi), reference=str(reference),check=method))
        rows = []
        maximum_radius = 0.
        for name, z, s in manufactured_fixtures():
            pairs, average = bound_curvature(z[None, ...], s[None, ...])
            references = interval_curvature(z, s, bits=192)[1]
            for j, reference in enumerate(references):
                assert arb(float(pairs.lo[0,j])) <= reference <= arb(float(pairs.hi[0,j])), (name, j)
            if len(z) == 1:
                assert np.all(pairs.lo <= 0) and np.all(pairs.hi >= 0)
            reference_float = price_curvature(z[None, ...], s[None, ...])[0, 1]
            radius = float(np.max(np.maximum(np.abs(pairs.lo[0]-reference_float), np.abs(pairs.hi[0]-reference_float))))
            maximum_radius = max(maximum_radius, radius)
            # Exact dyadic s are preserved by these shifts for the modular fixture.
            if name == "K100_modular":
                for shift in (0., 16., -1024.):
                    shifted, _ = bound_curvature(z[None, ...], (s+shift)[None, ...])
                    for j, reference in enumerate(references):
                        assert arb(float(shifted.lo[0,j])) <= reference <= arb(float(shifted.hi[0,j]))
                permutation = (2, 0, 1)
                permuted, _ = bound_curvature(np.take(z, permutation, axis=1)[None,...], s[None,...])
                indices = [PAIRS.index(tuple(sorted((permutation[i],permutation[j])))) for i,j in PAIRS]
                for j, index in enumerate(indices):
                    assert arb(float(permuted.lo[0,j])) <= references[index] <= arb(float(permuted.hi[0,j]))
            repeated, repeated_average = bound_curvature(np.repeat(z[None,...], 2, axis=0), np.repeat(s[None,...], 2, axis=0))
            assert np.array_equal(repeated.lo, np.repeat(pairs.lo, 2, axis=0))
            assert np.array_equal(repeated.hi, np.repeat(pairs.hi, 2, axis=0))
            assert np.array_equal(repeated_average.lo, np.repeat(average.lo, 2, axis=0))
            rows.append(dict(name=name, K=len(z), pair_lower=pairs.lo[0].tolist(), pair_upper=pairs.hi[0].tolist(), mean_lower=float(average.lo[0]), mean_upper=float(average.hi[0]), arb_reference=[str(v) for v in references], old_float_max_error_enclosed=radius))
        name,z,s = manufactured_fixtures()[-1]
        block_z=np.repeat(z[None,...],256,axis=0); block_s=np.repeat(s[None,...],256,axis=0)
        timings=[]
        for _ in range(3):
            started=time.perf_counter(); bound_curvature(block_z,block_s); timings.append(time.perf_counter()-started)
        return dict(status="passed", scope="manufactured_fixtures_only", random_draws=0, scientific_banks_read=0, certified_full_saved_pass=False, ideal_gaussian_pipeline_certified=False, IEEE_assumptions_proved_by_probes=False, C_pi=C_PI, C_q=C_Q, Taylor_degree=TAYLOR_DEGREE, ln2_interval=[LN2_LO,LN2_HI], ln2_arb=str(ln2), remainder_upper=REMAINDER_UPPER, exponent_fixtures=exp_rows, fixtures=rows, maximum_fixture_error_enclosed=maximum_radius, benchmark=dict(banks=256,K=100,repetitions=3,seconds=timings,median_seconds=float(np.median(timings)),full_262144_bank_extrapolation_seconds=float(np.median(timings))*1024.,not_a_scientific_runtime=True))
    finally:
        ctx.prec = previous
