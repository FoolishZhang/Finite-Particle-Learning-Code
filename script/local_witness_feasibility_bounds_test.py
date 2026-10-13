"""Manufactured scalar identities only; no old center, arrays, RNG or quadrature."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

import local_witness_feasibility_bounds as bounds
from local_witness_feasibility_bounds import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "artifacts/local_witness_feasibility_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json"
RESULTS = []


class BoundsTests(unittest.TestCase):
    def test_exact_dyadic_comparisons(self):
        for mantissa in (0, 1, 3, 101):
            for exponent in range(-10, 11):
                from fractions import Fraction
                exact = Fraction(mantissa)*Fraction(2)**exponent
                for target in ("0", "1/3", "1", "1048576"):
                    self.assertEqual(bounds.dyadic_less_rational(
                        {"mantissa": str(mantissa), "exponent_of_two": exponent}, target),
                        exact < Fraction(target))
        self.assertTrue(bounds.dyadic_less_rational(
            {"mantissa": "17", "exponent_of_two": -10**24}, "1/1000000"))
        self.assertFalse(bounds.dyadic_less_rational(
            {"mantissa": "17", "exponent_of_two": 10**24}, "1/1000000"))
        RESULTS.append({"test": "exact_dyadic_comparisons", "ordinary_cases": 336,
                        "enormous_exponent_cases": 2, "status": "passed"})

    def test_validation_and_precision_restoration(self):
        args = [1, 2, "0", "0", "7", "7", "-12", "6", "3"]
        for index, value in ((0, True), (1, 2.0), (2, -1), (3, .0),
                             (4, 0), (5, "-1"), (8, "0")):
            bad = args.copy(); bad[index] = value
            with self.assertRaises(ValueError):
                bounds.bound_box(*bad)
        before = ctx.prec
        record = bounds.bound_box(*args, targets={"gradient": "1/100"})
        self.assertEqual(ctx.prec, before)
        self.assertTrue(record["comparisons"][0]["strictly_below"])
        with self.assertRaises(ValueError):
            bounds.bound_box(*args, targets={"unknown": "1"})
        self.assertEqual(ctx.prec, before)
        RESULTS.append({"test": "validation_and_precision_restoration", "rejected_inputs": 7,
                        "record": record, "status": "passed"})

    def test_constant_likelihood_exact_laplace_tails(self):
        # Manufactured logistic m=1,b=a=0,y=1; A=1/2 for every Gaussian.
        # Complete bank means analytically give radial H=(1/4,1/(4K)).
        # Hperp high tail can be negative; compare its absolute value.
        previous = ctx.prec; ctx.prec = 192
        try:
            count = 0; cases = []
            for K in (1, 2, 5):
                for vmin, vmax in (("-12", "2"), ("-8", "4")):
                    data = bounds.bound_box(1, K, "0", "0", "7", "7", vmin, vmax, "3")
                    u0, U = bounds.aq(vmin).exp(), bounds.aq(vmax).exp()
                    s0, s = K*u0/2, K*U/2
                    low_d, high_d = -(-s0).expm1(), (-s).exp()
                    high_c = (1-arb(1)/K)*(1+s)*high_d
                    low_perp_high_d = (-s0).exp()
                    low_perp_high_c = (1-arb(1)/K)*(1+s0)*low_perp_high_d
                    exact = {
                        "gradient": {"lower_u": low_d/2, "upper_u": high_d/2},
                        "radial_hessian": {"lower_u": (1-(1+s0)*(-s0).exp())/4,
                                           "upper_u": (1+s)*high_d/4},
                        "transverse_hessian": {"upper_u": abs((high_d-high_c)/4),
                           "lower_u": abs((arb(1)/K-low_perp_high_d+low_perp_high_c)/4)}}
                    # Compare exact Arb upper endpoints against exact dyadic bound
                    # upper endpoints using integer arithmetic (no float certificate).
                    comparisons = []
                    for field, parts in exact.items():
                        for part, value in parts.items():
                            bound = data["bounds"][field][part]
                            # Arb strict comparison establishes exact expression <
                            # its conservative analytic-bound ball, stronger than
                            # comparing rounded displays.
                            from fractions import Fraction
                            ep = bound["lower_exact_dyadic"]
                            rational_lower = Fraction(int(ep["mantissa"]))*Fraction(2)**ep["exponent_of_two"]
                            self.assertTrue(value < bounds.aq(rational_lower), (K, field, part))
                            comparisons.append({"field": field, "part": part,
                                "exact_expression_enclosure": bounds.ball_record(value),
                                "strictly_below_bound_lower_endpoint": True})
                            count += 1
                    cases.append({"m": 1, "K": K, "b": "0", "a": "0", "label": 1,
                                  "vmin": vmin, "vmax": vmax, "comparisons": comparisons})
            RESULTS.append({"test": "constant_likelihood_exact_laplace_tails",
                            "cases": cases, "comparisons": count, "status": "passed"})
        finally:
            ctx.prec = previous

    def test_gaussian_mills_bounds(self):
        previous = ctx.prec; ctx.prec = 192
        try:
            cases = []
            for L in (1, 3, 7):
                x = arb(L)
                phi = (-x*x/2).exp()/(2*arb.pi()).sqrt()
                pexact = (x/arb(2).sqrt()).erfc()
                t2exact = 2*x*phi+pexact
                pupper, t2upper = 2*phi/x, 2*phi*(x+1/x)
                self.assertTrue(pexact < pupper)
                self.assertTrue(t2exact < t2upper)
                cases.append({"L": L, "p_exact": bounds.ball_record(pexact),
                              "t2_exact": bounds.ball_record(t2exact),
                              "p_mills_bound": bounds.ball_record(pupper),
                              "t2_mills_bound": bounds.ball_record(t2upper),
                              "both_strict_comparisons": True})
            RESULTS.append({"test": "gaussian_mills_bounds", "cases": cases, "status": "passed"})
        finally:
            ctx.prec = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.write_receipt:
        sources = [ROOT/"script/local_witness_feasibility_bounds.py", Path(__file__).resolve(),
                   ROOT/"docs/08_research_研究工作/LocalWitnessFeasibility_v1_20261008/TAIL_BOUNDS.md"]
        receipt = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed", "action": "manufactured_analytic_tail_qualification",
            "precision_bits": 192, "tests_run": result.testsRun, "results": RESULTS,
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "scientific_data_reads": 0, "random_draws": 0, "acb_integral_calls": 0,
            "real_center_evaluations": 0, "finite_quadrature_error_certified": False,
            "unclosed": ["real parameter-box choices and rational budgets",
                         "finite-interval integration/quadrature error", "stationary-point existence",
                         "Hessian variation over a neighborhood", "strict local witness"]}
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        print(RECEIPT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
