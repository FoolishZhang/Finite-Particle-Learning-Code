"""Bounded-score statistics for locked directions; no study I/O or sampling.

Actual sample bounds use an unbiased sample variance. Population-variance
substitution into the empirical-Bernstein expression is an oracle planning
proxy only: it is neither a confidence bound nor a power calculation.
"""
from __future__ import annotations

import math
import numbers
from typing import Mapping, Sequence

import numpy as np

N_GRID = (1024, 4096, 16384, 65536, 262144, 1048576)
FAMILY_SCORE_COUNT = 48
LOOK_COUNT = 6
ALPHA = .05
DEFAULT_DELTA = ALPHA / (FAMILY_SCORE_COUNT * LOOK_COUNT)
MAX_N = 2 ** 30
PRIMARY_SOURCE = "https://arxiv.org/pdf/0907.3740"


def config_dump():
    return dict(schema="direction_validation_statistics_v1", N_grid=list(N_GRID),
                family_score_count=FAMILY_SCORE_COUNT, look_count=LOOK_COUNT,
                alpha=ALPHA, delta_per_score_and_look=DEFAULT_DELTA, max_n=MAX_N,
                range_uses_all_labels=True, population_mass_normalized_explicitly=True,
                empirical_variance_ddof=1, empirical_Bernstein_source=PRIMARY_SOURCE,
                source_locator="Theorem 4 and Corollary 5, PDF page 2",
                confidence_direction="one-sided lower bound on an improvement score",
                two_sided_CI_splits_delta=True, thresholds_share_the_same_confidence_event=True,
                oracle_planning_not_confidence=True, oracle_planning_not_power=True,
                new_samples_generated=0, new_random_draws=0)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _n(n, name="n"):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or int(n) < 2:
        raise ValueError(name + " must be an integer >= 2")
    return int(n)


def _delta(delta):
    value = _finite(delta, "delta")
    if not 0 < value < 1:
        raise ValueError("delta must lie strictly between zero and one")
    return value


def _nonnegative(value, name):
    value = _finite(value, name)
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    return value


def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(name + " must be a nonempty finite vector")
    return result


def safe_ratio(numerator, denominator):
    """Explicit undefined ratios; never encode a zero-variance SNR as infinity."""
    top = _finite(numerator, "numerator"); bottom = _finite(denominator, "denominator")
    if bottom == 0:
        return dict(value=None, defined=False, status="zero_denominator")
    value = top / bottom
    if not math.isfinite(value):
        return dict(value=None, defined=False, status="floating_overflow")
    return dict(value=value, defined=True, status="finite")


def weighted_moments(scores, raw_probabilities):
    """Moments under explicit pbar=P/sum(P); do not mutate either input.

    Range includes every supplied score, even labels with zero raw probability.
    The study caller supplies all 256 labels; small manufactured fixtures are
    accepted here. The raw mass/mean and normalization change remain visible.
    """
    x = _vector(scores, "scores")
    p = _vector(raw_probabilities, "raw_probabilities")
    if x.shape != p.shape or np.any(p < 0):
        raise ValueError("nonnegative raw probabilities must match the scores")
    mass = math.fsum(map(float, p))
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("raw probability mass must be positive and finite")
    pbar = p / mass
    raw_mean = math.fsum(float(pi) * float(xi) for pi, xi in zip(p, x))
    mean = math.fsum(float(pi) * float(xi) for pi, xi in zip(pbar, x))
    variance = math.fsum(float(pi) * (float(xi) - mean) ** 2 for pi, xi in zip(pbar, x))
    raw_centered = math.fsum(float(pi) * (float(xi) - mean) ** 2 for pi, xi in zip(p, x))
    low, high = float(np.min(x)), float(np.max(x))
    if low == high:
        # Exact constant-score identity, not a data-driven truncation of variance.
        mean, variance, raw_centered = low, 0., 0.
    for value in (raw_mean, mean, variance, raw_centered, high - low):
        if not math.isfinite(value):
            raise ValueError("moment arithmetic overflow")
    return dict(score_count=len(x), raw_mass=mass, raw_mass_error=mass - 1,
                raw_mean=raw_mean, mean=mean, variance=variance, std=math.sqrt(variance),
                raw_centered_second_moment=raw_centered, score_min=low, score_max=high,
                range=high - low, normalization_correction=mean - raw_mean,
                normalization_correction_absolute=abs(mean - raw_mean),
                normalization_correction_algebraic_bound=abs(1 - mass) * max(abs(low), abs(high)),
                normalization_correction_bound_is_exact_arithmetic_not_float_certificate=True,
                normalization_explicit=True, pbar=pbar.copy(), pbar_mass=math.fsum(map(float, pbar)),
                target_law="explicitly normalized saved probability vector",
                range_includes_zero_probability_labels=True,
                mean_over_sd=safe_ratio(mean, math.sqrt(variance)))


def hoeffding_radius(n, score_range, delta=DEFAULT_DELTA):
    """One-sided Hoeffding radius; the shared API deliberately requires n>=2."""
    n = _n(n); R = _nonnegative(score_range, "score_range"); delta = _delta(delta)
    return R * math.sqrt(-math.log(delta) / (2 * n))


def empirical_bernstein_radius(n, variance, score_range, delta=DEFAULT_DELTA):
    """Theorem-4 expression. Confidence use requires unbiased sample variance.

    Passing population variance only evaluates the planning expression. A zero
    observed variance with nonzero global range still has the positive 1/n term.
    """
    n = _n(n); V = _nonnegative(variance, "variance")
    R = _nonnegative(score_range, "score_range"); delta = _delta(delta)
    if R == 0 and V != 0:
        raise ValueError("zero score range requires zero variance")
    logterm = math.log(2.) - math.log(delta)
    return math.sqrt(2 * V * logterm / n) + 7 * R * logterm / (3 * (n - 1))


def _sample_summary(samples, lower, upper):
    x = _vector(samples, "samples"); n = _n(len(x))
    low = _finite(lower, "lower"); high = _finite(upper, "upper")
    if low > high or np.any(x < low) or np.any(x > high):
        raise ValueError("samples must be inside the externally fixed score bounds")
    mean = math.fsum(map(float, x)) / n
    variance = math.fsum((float(xi) - mean) ** 2 for xi in x) / (n - 1)
    if np.all(x == x[0]):
        mean, variance = float(x[0]), 0.
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise ValueError("sample moment overflow")
    return dict(n=n, mean=mean, sample_variance_unbiased=variance,
                score_min=low, score_max=high, range=high - low,
                range_source="externally fixed whole-score support, not sample extrema")


def sample_empirical_bernstein_bound(samples, lower, upper, delta=DEFAULT_DELTA, side="lower"):
    """Actual-sample one-sided bound for a locked bounded score and iid data.

    Bounds need to hold for the exact score; this function does not certify
    quadrature or floating-point score errors. Independence is a caller condition.
    """
    if side not in ("lower", "upper"):
        raise ValueError("side must be lower or upper")
    result = _sample_summary(samples, lower, upper); delta = _delta(delta)
    radius = empirical_bernstein_radius(result["n"], result["sample_variance_unbiased"], result["range"], delta)
    endpoint = result["mean"] + (radius if side == "upper" else -radius)
    return dict(**result, side=side, delta=delta, confidence=1 - delta, radius=radius,
                bound=endpoint, bound_not_clipped_to_support=True,
                variance_source="unbiased sample variance", conditional_on_locked_score_and_iid_validation=True,
                numerical_score_error_certified=False, method="empirical_bernstein_one_sided")


def sample_empirical_bernstein_ci(samples, lower, upper, delta=DEFAULT_DELTA):
    """A simultaneous two-sided CI uses delta/2 per side, not delta twice."""
    result = _sample_summary(samples, lower, upper); delta = _delta(delta)
    radius = empirical_bernstein_radius(result["n"], result["sample_variance_unbiased"], result["range"], delta / 2)
    return dict(**result, delta=delta, delta_per_side=delta / 2, confidence=1 - delta,
                radius=radius, lower=result["mean"] - radius, upper=result["mean"] + radius,
                variance_source="unbiased sample variance", conditional_on_locked_score_and_iid_validation=True,
                numerical_score_error_certified=False, method="empirical_bernstein_two_sided")


def _planning_radius(n, variance, score_range, method, delta):
    if method == "hoeffding":
        return hoeffding_radius(n, score_range, delta)
    if method == "eb_proxy":
        return empirical_bernstein_radius(n, variance, score_range, delta)
    raise ValueError("method must be hoeffding or eb_proxy")


def minimum_n(mean, target, variance, score_range, method, delta=DEFAULT_DELTA,
              max_n=MAX_N, radius_multiplier=1):
    """Minimum integer n with multiplier*planning_radius < mean-target.

    Monotone doubling/bisection, capped without automatic budget extension.
    This is the minimum for the chosen sufficient radius inequality, not a
    necessary sample size, confidence statement or test-power guarantee.
    """
    mu = _finite(mean, "mean"); target = _finite(target, "target")
    V = _nonnegative(variance, "variance"); R = _nonnegative(score_range, "score_range")
    delta = _delta(delta); cap = _n(max_n, "max_n")
    factor = _finite(radius_multiplier, "radius_multiplier")
    if factor <= 0:
        raise ValueError("radius_multiplier must be positive")
    _planning_radius(2, V, R, method, delta)
    margin = mu - target
    if not math.isfinite(margin):
        raise ValueError("nonfinite mean-target margin")
    common = dict(method=method, mean=mu, target=target, margin=margin, delta=delta,
                  population_variance=V, score_range=R, radius_multiplier=factor, max_n=cap,
                  strict_inequality=True, oracle_planning=True, confidence_bound=False,
                  power_guarantee=False, necessary_sample_size=False)

    def radius(n):
        return factor * _planning_radius(n, V, R, method, delta)

    if margin <= 0:
        return dict(**common, n=None, status="nonpositive_oracle_margin", radius_at_n=None,
                    previous_n=None, radius_at_previous_n=None, radius_at_cap=radius(cap))
    if radius(2) < margin:
        found = 2
    elif not radius(cap) < margin:
        return dict(**common, n=None, status="not_reached_by_cap", radius_at_n=None,
                    previous_n=None, radius_at_previous_n=None, radius_at_cap=radius(cap))
    else:
        lo, hi = 2, 4
        hi = min(hi, cap)
        while not radius(hi) < margin:
            lo, hi = hi, min(2 * hi, cap)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if radius(mid) < margin:
                hi = mid
            else:
                lo = mid
        found = hi
    previous = found - 1 if found > 2 else None
    return dict(**common, n=found, status="strict_planning_inequality_met", radius_at_n=radius(found),
                previous_n=previous, radius_at_previous_n=None if previous is None else radius(previous),
                radius_at_cap=radius(cap))


def target_values(kind, t=None):
    """Targets use positive improvements: -curvature or log Q_t-log Q_0."""
    if kind == "curvature":
        return {"zero": 0., "curvature_1e-4": 1e-4}
    if kind != "finite":
        raise ValueError("kind must be curvature or finite")
    t = _finite(t, "t")
    if t == 0:
        raise ValueError("finite-path planning uses a nonzero preselected t")
    return {"zero": 0., "matched_curvature_1e-4": .5 * t * t * 1e-4,
            "absolute_1e-5": 1e-5, "absolute_1e-4": 1e-4}


def planning_table(mean, variance, score_range, targets: Mapping[str, float],
                   Ngrid: Sequence[int] = N_GRID, delta=DEFAULT_DELTA, max_n=MAX_N):
    """Oracle width curves and sufficient-budget diagnostics; no observed CI.

    Multiple targets reuse one score event, so targets do not multiply delta.
    The separate two-Hoeffding-radius column is only a sufficient margin check.
    """
    mu = _finite(mean, "mean"); V = _nonnegative(variance, "variance")
    R = _nonnegative(score_range, "score_range"); delta = _delta(delta)
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("a nonempty named target mapping is required")
    clean = {str(k): _finite(v, "target") for k, v in targets.items()}
    if len(clean) != len(targets):
        raise ValueError("target names collide after string conversion")
    grid = [_n(n) for n in Ngrid]
    if not grid or grid != sorted(set(grid)):
        raise ValueError("Ngrid must contain unique ascending integers >=2")
    rows = []
    for n in grid:
        h = hoeffding_radius(n, R, delta)
        e = empirical_bernstein_radius(n, V, R, delta)
        rows.append(dict(n=n, hoeffding_radius=h, eb_proxy_radius=e,
                         oracle_mean_over_standard_error=safe_ratio(mu, math.sqrt(V / n)),
                         target_checks={name: dict(target=value, oracle_gap=mu - value,
                             mean_minus_hoeffding_exceeds_target=h < mu - value,
                             mean_minus_eb_proxy_exceeds_target=e < mu - value,
                             twice_hoeffding_below_gap=2 * h < mu - value)
                             for name, value in clean.items()}))
    minima = {name: {method: minimum_n(mu, value, V, R, method, delta, max_n)
                    for method in ("hoeffding", "eb_proxy")} for name, value in clean.items()}
    sufficient = {name: minimum_n(mu, value, V, R, "hoeffding", delta, max_n, radius_multiplier=2)
                  for name, value in clean.items()}
    return dict(mean=mu, population_variance=V, score_range=R, delta=delta,
                targets=clean, rows=rows, minimum_n=minima, twice_hoeffding_sufficient_n=sufficient,
                oracle_planning=True, confidence_bound=False, power_guarantee=False,
                targets_share_same_score_event=True, observed_validation_samples=0,
                eb_variance_source="normalized population planning variance, not sample variance")
