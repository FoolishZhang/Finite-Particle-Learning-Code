"""Zero-RNG exact Gaussian-polynomial check of paired local-risk arithmetic.

This checks a limiting Gaussian experiment, not finite-N model simulation,
quadrature, optimization, coverage or power. Only Python standard library.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ZERO = (0, 0, 0)


def add(*polynomials):
    out = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            out[exponent] = out.get(exponent, Q(0)) + coefficient
    return {e: c for e, c in out.items() if c}


def scale(polynomial, coefficient):
    return {e: coefficient*c for e, c in polynomial.items() if coefficient*c}


def multiply(first, second):
    out = {}
    for e, c in first.items():
        for f, d in second.items():
            g = tuple(x+y for x, y in zip(e, f))
            out[g] = out.get(g, Q(0)) + c*d
    return {e: c for e, c in out.items() if c}


def normal_expectation(polynomial):
    """Independent N(0,1) coordinates, exact integer double-factorial moments."""
    total = Q(0)
    for exponent, coefficient in polynomial.items():
        moment = 1
        for power in exponent:
            if power % 2:
                moment = 0
                break
            for odd in range(1, power, 2):
                moment *= odd
        total += coefficient*moment
    return total


def variance(polynomial):
    return normal_expectation(multiply(polynomial, polynomial))-normal_expectation(polynomial)**2


def gaussian_risks(r, a):
    """Build all three risks from their vector errors, not closed risk formulas."""
    group, mle = {}, {}
    for i, signal in enumerate(a):
        exponent = tuple(int(i == j) for j in range(3))
        epsilon = {exponent: Q(1)}
        group_error = add(scale(epsilon, r), {ZERO: -(1-r)*signal})
        group = add(group, scale(multiply(group_error, group_error), Q(1, 2)))
        mle = add(mle, scale(multiply(epsilon, epsilon), Q(1, 2)))
    independent = {ZERO: sum(signal**2 for signal in a)/2}
    return group, mle, independent


def check():
    count = 0

    def equal(actual, expected, label):
        nonlocal count
        count += 1
        if actual != expected:
            raise AssertionError(f"{label}: {actual} != {expected}")

    for r in (Q(1, 4), Q(1, 2), Q(3, 4), Q(7, 22)):
        for a in ((Q(0), Q(0), Q(0)), (Q(2), Q(-1), Q(2)), (Q(1, 2), Q(3, 2), Q(-2))):
            S = sum(x*x for x in a)
            group, mle, independent = gaussian_risks(r, a)
            difference_mle = add(group, scale(mle, -1))
            difference_independent = add(group, scale(independent, -1))
            equal(normal_expectation(group), ((1-r)**2*S+3*r*r)/2, "E group")
            equal(normal_expectation(mle), Q(3, 2), "E MLE")
            equal(normal_expectation(independent), S/2, "I risk")
            equal(normal_expectation(difference_mle), ((1-r)**2*S+3*r*r-3)/2, "paired mean GM")
            equal(normal_expectation(difference_independent), ((1-r)**2*S+3*r*r-S)/2, "paired mean GI")
            equal(variance(difference_mle), Q(3, 2)*(1-r*r)**2+r*r*(1-r)**2*S, "paired variance GM")
            equal(variance(difference_independent), Q(3, 2)*r**4+r*r*(1-r)**2*S, "paired variance GI")
        threshold_I = 3*r/(2-r)
        threshold_ML = 3*(1+r)/(1-r)
        equal(((1-r)**2*threshold_I+3*r*r-threshold_I)/2, Q(0), "I threshold")
        equal(((1-r)**2*threshold_ML+3*r*r-3)/2, Q(0), "MLE threshold")
    return count


def arithmetic_rows():
    j = .05558506019423109
    r = .3201223434446316
    rows = []
    for N, S in ((2048, .05), (2048, 2.), (2048, 12.), (256, 2.), (2048, .2), (2048, 8.)):
        group = ((1-r)**2*S+3*r*r)/2
        var_GM = 1.5*(1-r*r)**2+r*r*(1-r)**2*S
        var_GI = 1.5*r**4+r*r*(1-r)**2*S
        rows.append(dict(N=N, S=S, theta12_theta13=math.sqrt(S/(2*j*N)),
            min_eigenvalue=1-math.sqrt(S/(j*N)), group_Nrisk=group,
            MLE_Nrisk=1.5, I_Nrisk=S/2,
            paired_GM_mean=group-1.5, paired_GI_mean=group-S/2,
            paired_GM_variance=var_GM, paired_GI_variance=var_GI,
            Gaussian_nominal_t24_GM=(group-1.5)*math.sqrt(24/var_GM),
            Gaussian_nominal_t24_GI=(group-S/2)*math.sqrt(24/var_GI)))
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = Path(__file__).resolve()
    shutil.copyfile(source, out/source.name)
    source_hash = sha(source)
    if sha(out/source.name) != source_hash:
        raise RuntimeError("source copy identity failure")
    try:
        count = check()
        result = dict(schema="local_shrinkage_risk_algebra_manufacturing_v1", passed=True,
            exact_Fraction_checks=count, scientific_RNG_draws=0, scientific_scores=0,
            numerical_integral_evaluations=0, source_sha256=source_hash,
            scope="exact_normal_polynomial_moments_and_thresholds; floating_design_arithmetic",
            not_verified=["finite_N_risk", "coverage", "power", "algorithm_localization"],
            design_arithmetic=arithmetic_rows(), seconds=time.perf_counter()-started)
    except Exception as exc:
        result = dict(passed=False, error=repr(exc), scientific_RNG_draws=0, source_sha256=source_hash)
        (out/"FAILED.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        raise
    (out/"SUMMARY.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (out/"RUN.log").write_text(f"passed={result['passed']} exact_Fraction_checks={count} RNG=0 scores=0\n", encoding="utf-8")
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/"MANIFEST.json").write_text(json.dumps(dict(files=files), indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
