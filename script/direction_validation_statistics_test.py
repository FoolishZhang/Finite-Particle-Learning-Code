"""Manufactured tests only; never reads study scores, PMFs, or validation data."""
from __future__ import annotations

import json
import math
import time

import numpy as np

import direction_validation_statistics as s


def run_tests():
    tick = time.perf_counter(); passed = []

    def close(a, b, **kw):
        np.testing.assert_allclose(a, b, rtol=kw.get("rtol", 2e-13), atol=kw.get("atol", 1e-15))

    def rejects(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Expected explicit invalid-input rejection")

    x = np.array([-3., -.5, 1., 4.]); p = np.array([.15, .25, .35, .25]) * (1 + 3e-12)
    xc, pc = x.copy(), p.copy(); m = s.weighted_moments(x, p)
    assert np.array_equal(x, xc) and np.array_equal(p, pc)
    close(m["raw_mass"], math.fsum(p)); close(m["pbar"], p / math.fsum(p))
    close(m["raw_mean"], np.dot(p, x)); close(m["mean"], np.dot(p / math.fsum(p), x))
    close(m["normalization_correction"], m["mean"] - m["raw_mean"])
    close(m["variance"], np.dot(m["pbar"], (x - m["mean"]) ** 2))
    assert m["range"] == 7 and m["normalization_explicit"]
    passed.append("explicit_raw_mass_normalization_and_input_preservation")

    zero_support = s.weighted_moments([-100., 2., 8.], [0., 1., 0.])
    assert zero_support["mean"] == 2 and zero_support["variance"] == 0 and zero_support["range"] == 108
    constant = s.weighted_moments([.3] * 7, [.2] * 7)
    assert constant["mean"] == .3 and constant["range"] == constant["variance"] == 0
    assert s.hoeffding_radius(2, 0) == s.empirical_bernstein_radius(2, 0, 0) == 0
    assert s.empirical_bernstein_radius(100, 0, 2) > 0
    assert not constant["mean_over_sd"]["defined"] and constant["mean_over_sd"]["value"] is None
    assert s.safe_ratio(0, 0)["status"] == "zero_denominator"
    close(s.safe_ratio(3, 2)["value"], 1.5)
    passed.append("global_range_zero_mass_labels_and_degenerate_scores")

    for scale, shift in [(3., .8), (.002, 0.), (-2., 1.3)]:
        transformed = s.weighted_moments(scale * x + shift, p)
        close(transformed["mean"], scale * m["mean"] + shift)
        close(transformed["variance"], scale ** 2 * m["variance"])
        close(transformed["range"], abs(scale) * m["range"])
        close(s.hoeffding_radius(1024, transformed["range"]), abs(scale) * s.hoeffding_radius(1024, m["range"]))
        close(s.empirical_bernstein_radius(1024, transformed["variance"], transformed["range"]),
              abs(scale) * s.empirical_bernstein_radius(1024, m["variance"], m["range"]))
    passed.append("affine_weighted_moment_and_radius_equivariance")

    sample = np.array([-.4, .2, .8, .2, -.1]); n = len(sample); d = .031
    single = s.sample_empirical_bernstein_bound(sample, -1, 2, d)
    pairwise = sum((float(sample[i]) - float(sample[j])) ** 2 for i in range(n) for j in range(i + 1, n)) / (n * (n - 1))
    close(single["sample_variance_unbiased"], pairwise)
    close(single["sample_variance_unbiased"], np.var(sample, ddof=1))
    expected = math.sqrt(2 * pairwise * math.log(2 / d) / n) + 7 * 3 * math.log(2 / d) / (3 * (n - 1))
    close(single["radius"], expected); close(single["bound"], sample.mean() - expected)
    reflected = s.sample_empirical_bernstein_bound(-sample, -2, 1, d, side="upper")
    close(single["bound"], -reflected["bound"])
    passed.append("theorem4_pairwise_unbiased_variance_and_reflected_lower_bound")

    c = s.sample_empirical_bernstein_ci(sample, -1, 2, d)
    assert c["delta_per_side"] == d / 2
    one_half = s.sample_empirical_bernstein_bound(sample, -1, 2, d / 2)
    close(c["lower"], one_half["bound"])
    shifted = s.sample_empirical_bernstein_ci(7 * sample + 3, -4, 17, d)
    close([shifted["lower"], shifted["upper"]], 7 * np.array([c["lower"], c["upper"]]) + 3)
    assert c["radius"] > single["radius"]
    assert s.sample_empirical_bernstein_bound([.3, .3, .3], -1, 1)["sample_variance_unbiased"] == 0
    passed.append("two_sided_delta_split_and_sample_affine_equivariance")

    for badn in (0, 1, -1, 1.2, True):
        rejects(lambda badn=badn: s.hoeffding_radius(badn, 1))
        rejects(lambda badn=badn: s.empirical_bernstein_radius(badn, 1, 1))
    for badd in (0, 1, -.1, math.inf, math.nan):
        rejects(lambda badd=badd: s.hoeffding_radius(2, 1, badd))
    for fn in [lambda: s.weighted_moments([0, 1], [0, 0]),
               lambda: s.weighted_moments([0, 1], [-1, 2]),
               lambda: s.weighted_moments([0, 1], [1]),
               lambda: s.weighted_moments([0, math.inf], [.5, .5]),
               lambda: s.sample_empirical_bernstein_bound([.1], 0, 1),
               lambda: s.sample_empirical_bernstein_bound([-.1, .1], 0, 1),
               lambda: s.sample_empirical_bernstein_bound([0, 1], 1, 0),
               lambda: s.empirical_bernstein_radius(2, 1, 0),
               lambda: s.empirical_bernstein_radius(2, -1, 1),
               lambda: s.hoeffding_radius(2, -1)]:
        rejects(fn)
    passed.append("n1_invalid_and_domain_failures_are_explicit")

    for R, gap, delta in [(1.3, .17, .02), (.07, .015, s.DEFAULT_DELTA), (2., .8, .1)]:
        analytic = max(2, math.floor(R * R * math.log(1 / delta) / (2 * gap * gap)) + 1)
        result = s.minimum_n(gap, 0, .01, R, "hoeffding", delta)
        assert result["n"] == analytic
        assert result["radius_at_n"] < gap
        assert result["previous_n"] is None or result["radius_at_previous_n"] >= gap
    passed.append("Hoeffding_integer_minimum_matches_analytic_sufficient_radius_formula")

    for method in ("hoeffding", "eb_proxy"):
        radius = s._planning_radius(64, .025, .6, method, .03)
        exact_boundary = s.minimum_n(radius, 0, .025, .6, method, .03, max_n=1000)
        assert exact_boundary["n"] == 65
        too_small = s.minimum_n(radius, 0, .025, .6, method, .03, max_n=64)
        assert too_small["n"] is None and too_small["status"] == "not_reached_by_cap"
        found_at_cap = s.minimum_n(radius, 0, .025, .6, method, .03, max_n=65)
        assert found_at_cap["n"] == 65
    assert s.minimum_n(0, 0, 0, 0, "hoeffding")["n"] is None
    assert s.minimum_n(.1, 0, 0, 0, "eb_proxy")["n"] == 2
    assert s.minimum_n(-1, 0, 0, 1, "eb_proxy")["status"] == "nonpositive_oracle_margin"
    passed.append("strict_root_boundary_cap_and_no_solution_semantics")

    for method in ("hoeffding", "eb_proxy"):
        for scale in (.0002, 7.):
            a = s.minimum_n(.06, .01, .025, .6, method)
            b = s.minimum_n(scale * .06, scale * .01, scale ** 2 * .025, scale * .6, method)
            assert a["n"] == b["n"]
    doubled = s.minimum_n(.03, 0, 0, .6, "hoeffding", radius_multiplier=2)
    equivalent = s.minimum_n(.015, 0, 0, .6, "hoeffding")
    assert doubled["n"] == equivalent["n"]
    passed.append("planning_scale_equivariance_and_two_radius_sufficient_condition")

    targets = s.target_values("finite", .02)
    assert set(targets) == {"zero", "matched_curvature_1e-4", "absolute_1e-5", "absolute_1e-4"}
    close(targets["matched_curvature_1e-4"], .5 * .02 ** 2 * 1e-4)
    table = s.planning_table(.02, .01, 1., targets)
    assert [row["n"] for row in table["rows"]] == list(s.N_GRID)
    assert table["delta"] == .05 / (48 * 6) and table["targets_share_same_score_event"]
    assert table["oracle_planning"] and not table["confidence_bound"] and not table["power_guarantee"]
    close(table["rows"][0]["oracle_mean_over_standard_error"]["value"], .02 / math.sqrt(.01 / 1024))
    assert all(v["delta"] == s.DEFAULT_DELTA for group in table["minimum_n"].values() for v in group.values())
    rejects(lambda: s.planning_table(1, 1, 2, {"zero": 0}, Ngrid=[4, 4]))
    rejects(lambda: s.target_values("finite", 0))
    passed.append("fixed_family_budget_targets_and_nonconfidence_flags")

    # Analytic even positive finite-alphabet family; no model/kernel or study data.
    q0 = np.array([.2, .3, .5]); c2 = np.array([.03, -.01, -.02]); p0 = np.array([.4, .1, .5])
    curvature = -2 * c2 / q0
    expected_m = s.weighted_moments(-curvature, p0)
    errors = []
    for t in (.08, .04, .02):
        qt = q0 + t * t * c2
        z = np.log(qt) - np.log(q0)
        normalized = s.weighted_moments(2 * z / (t * t), p0)
        errors.append(np.max(abs(2 * z / (t * t) + curvature)))
        assert abs(normalized["mean"] - expected_m["mean"]) < .001
    assert errors[1] < .26 * errors[0] and errors[2] < .26 * errors[1]
    passed.append("manufactured_even_family_scaled_score_limit")
    return dict(passed=True, tests=len(passed), cases=passed, seconds=time.perf_counter() - tick,
                study_data_read=False, random_draws=0, new_validation_samples=0,
                scope="manufactured deterministic numerical/logic checks, not a scientific review")


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
