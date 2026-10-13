"""Manufactured scalar checks only; no actual student gradient integration."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import unittest
import certified_f8_gradient_bounds as bounds
from certified_f8_gradient_bounds import aq, arb, ctx, record
from flint import acb

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/"artifacts/certified_f8_gradient_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json"
RESULTS = []


class ScalarTests(unittest.TestCase):
    def test_fixed_planning_budget(self):
        previous = ctx.prec
        data = bounds.gradient_l2_bounds(bounds.PLANNING_CENTER_PAIRS)
        self.assertEqual(ctx.prec, previous)
        self.assertTrue(data["comparison"]["strictly_below"])
        self.assertEqual(data["grid"]["inner_nodes"], 881)
        self.assertEqual(data["grid"]["outer_nodes"], 769)
        self.assertEqual(data["bounds"]["true_per_label_gradient_global_L2"]["lower_exact_dyadic"],
                         {"mantissa": "1", "exponent_of_two": 3})
        self.assertEqual(Fraction(data["exact_inputs"]["teacher_error_L1_upper"]),
                         Fraction("4.894e-21"))
        RESULTS.append({"test": "fixed_planning_budget", "status": "passed", "result": data,
                        "real_gradient_evaluations": 0})

    def test_unimodal_discrete_upper_tail(self):
        previous = ctx.prec; ctx.prec = 192
        try:
            h, e = arb(1)/8, arb(1).exp()
            cases = []
            for s0q in ("1/100", "1", "10"):
                s0 = aq(s0q)
                prefix = h*sum(((s0*(j*h).exp())*(-s0*(j*h).exp()).exp()
                                for j in range(1, 513)), arb(0))
                # Beyond node 512 this positive manufactured scalar density is
                # decreasing, so its right-grid tail is at most the integral.
                remainder = (-s0*(512*h).exp()).exp()
                grid_upper = prefix+remainder
                continuous = (-s0).exp()
                unimodal_upper = continuous+2*h/e
                self.assertTrue(grid_upper < unimodal_upper)
                monotone = Fraction(s0q) >= 1
                if monotone:
                    self.assertTrue(grid_upper < continuous)
                cases.append({"s_at_boundary_exact": s0q,
                    "grid_prefix_plus_remainder_upper": record(grid_upper),
                    "unimodal_bound": record(unimodal_upper),
                    "unimodal_strict_comparison": True,
                    "decreasing_case_strict_integral_comparison": monotone})
            RESULTS.append({"test": "unimodal_discrete_upper_tail", "status": "passed", "cases": cases})
        finally:
            ctx.prec = previous

    def test_constant_bank_outer_alias_fixture(self):
        # Invented A=1, scalar G=1,K=1. Integral of exp(v)*exp(-exp(v)) is 1.
        # This is a scalar identity fixture, not a logistic study gradient.
        previous = ctx.prec; ctx.prec = 192
        try:
            h, lo, hi, d = arb(1)/8, arb(-32), arb(10), arb(1)/2
            n = 336
            total = h*sum(((lo+j*h).exp()*(-(lo+j*h).exp()).exp()
                          for j in range(n+1)), arb(0))
            q = (-2*arb.pi()*d/h).exp()
            alias = 2*q/((1-q)*d.cos())
            lower = h*lo.exp()/h.expm1()
            upper = (-hi.exp()).exp()
            error = abs(total-1)
            self.assertTrue(error < alias+lower+upper)
            RESULTS.append({"test": "constant_bank_outer_alias_fixture", "status": "passed",
                "h": "1/8", "vmin": "-32", "vmax": "10", "full_weight_nodes": n+1,
                "exact_integral": "1", "finite_scalar_sum": record(total),
                "absolute_error": record(error), "analytic_bound": record(alias+lower+upper),
                "strict_comparison": True, "real_gradient_evaluations": 0})
        finally:
            ctx.prec = previous

    def test_complex_likelihood_sector_fixture(self):
        previous = ctx.prec; ctx.prec = 192
        try:
            d = arb(1)/8
            aa = [arb(1)/4, arb(1)/2, arb(1)/2, arb(1)/4]
            theta = sum(aa, arb(0))*d
            cases = 0
            for xq in ("-2", "0", "2"):
                z = acb(aq(xq), d)
                for label in itertools.product((-1, 1), repeat=4):
                    A = acb(1)
                    for sign, a in zip(label, aa):
                        A /= 1+(-sign*a*z).exp()
                    self.assertTrue(abs(A.arg()) < theta)
                    self.assertTrue(A.real > theta.cos()*abs(A))
                    cases += 1
            RESULTS.append({"test": "complex_likelihood_sector_fixture", "status": "passed",
                "manufactured_labels_times_real_positions": cases,
                "theta": record(theta), "all_strict_sector_comparisons": True,
                "acb_integral_calls": 0})
        finally:
            ctx.prec = previous

    def test_exact_inputs_and_comparisons(self):
        comparisons = 0
        for m in (0, 1, 17):
            for e in range(-12, 13):
                for q in ("0", "1/3", "1/100000000", "100"):
                    self.assertEqual(bounds.dyadic_lt_q(
                        {"mantissa": str(m), "exponent_of_two": e}, q),
                        Fraction(m)*Fraction(2)**e < Fraction(q))
                    comparisons += 1
        for kwargs in ({"K": 8.0}, {"hz": .025}, {"L": "3/2", "hz": "1"},
                       {"dz": "1"}, {"dv": "2"}, {"teacher_mass_upper": "-1"},
                       {"teacher_error_L1_upper": "-1"}, {"precision_bits": True}):
            with self.assertRaises(ValueError):
                bounds.gradient_l2_bounds(bounds.PLANNING_CENTER_PAIRS, **kwargs)
        RESULTS.append({"test": "exact_inputs_and_comparisons", "status": "passed",
                        "comparisons": comparisons, "rejected_inputs": 8})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScalarTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.write_receipt:
        sources = [ROOT/"script/certified_f8_gradient_bounds.py", Path(__file__).resolve(),
            ROOT/"docs/08_research_研究工作/CertifiedF8Gradient_v1_20261008/QUADRATURE_PROOF.md"]
        receipt = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "action": "manufactured_scalar_nested_gradient_budget", "status": "passed",
            "tests_run": result.testsRun, "precision_bits": 192, "results": RESULTS,
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "real_gradient_evaluations": 0, "scientific_data_reads": 0, "random_draws": 0,
            "acb_integral_calls": 0, "compute_arithmetic_included": False,
            "scientific_execution_frozen": False,
            "unclosed": ["actual finite-grid interval arithmetic",
                         "teacher mass/error comparison to previous certificate",
                         "true gradient values", "Hessian/existence certification"]}
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        print(RECEIPT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
