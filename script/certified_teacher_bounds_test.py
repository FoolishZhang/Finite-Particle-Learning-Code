"""Manufactured scalar qualification of teacher quadrature error formulas."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
import certified_teacher_bounds as bounds
from certified_teacher_bounds import arb, ctx, aq, record

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/"artifacts/certified_teacher_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json"
RESULTS = []


class ManufacturedTests(unittest.TestCase):
    def test_prior_scalar_budget(self):
        before = ctx.prec
        out = bounds.teacher_l1_bounds()
        self.assertEqual(ctx.prec, before)
        self.assertTrue(all(x["strictly_below"] for x in out["comparisons"]))
        self.assertEqual(out["grid"]["N_actual_integer"], 80)
        self.assertEqual(out["grid"]["points_total"], 25921)
        self.assertEqual(out["grid"]["omitted_axis_indices_begin_at"], 81)
        self.assertFalse(out["factor_256_used"])
        RESULTS.append({"test": "prior_scalar_budget", "status": "passed", "result": out,
                        "real_PMF_computed": False})

    def test_row_modulus_identity(self):
        previous = ctx.prec; ctx.prec = 192
        try:
            cases = []
            for t in (Fraction(1, 4), Fraction(1), Fraction(4)):
                for beta in (Fraction(1, 5), Fraction(3, 5)):
                    a, b = aq(t), aq(beta)
                    c = (b/2).cos()
                    denominator2 = (1-a)**2+4*a*c*c
                    identity_residual = denominator2-c*c*(1+a)**2-(1-c*c)*(1-a)**2
                    self.assertTrue(identity_residual.contains(0))
                    row = (1+a)/denominator2.sqrt()
                    secant = 1/c
                    if t != 1:
                        self.assertTrue(row < secant)
                    else:
                        self.assertTrue((row-secant).contains(0))
                    cases.append({"t_exact": str(t), "imag_logit_exact": str(beta),
                                  "row_modulus_sum": record(row), "secant": record(secant),
                                  "identity_residual_encloses_zero": True,
                                  "strict_comparison_or_exact_equality_case": True})
            RESULTS.append({"test": "row_modulus_identity", "status": "passed", "cases": cases})
        finally:
            ctx.prec = previous

    def test_zero_loading_gaussian_alias_and_endpoint_tail(self):
        # Manufactured teacher with zero loadings: all labels are constants.
        # Its total L1 grid alias is exactly S_h**2-1 for the Gaussian grid mass.
        # Use h=1 to make the analytic alias visible at 192 bits. No PMF sum.
        previous = ctx.prec; ctx.prec = 192
        try:
            h, L, N = arb(1), arb(3), 3
            out = bounds.teacher_l1_bounds(h="1", limit="3", amplitude="0", angles=("0",))
            r = (-2*arb.pi()**2).exp()
            Slo = 1+2*sum((r**(k*k) for k in range(1, 9)), arb(0))
            Shi = Slo+2*r**81/(1-r)
            exact_alias_upper = Shi**2-1
            theoretical_alias = arb(1).exp()*4*(-2*arb.pi()).exp()/(1-(-2*arb.pi()).exp())**2
            self.assertTrue(exact_alias_upper < theoretical_alias)
            phi = lambda x: (-x*x/2).exp()/(2*arb.pi()).sqrt()
            # Axis omitted points begin at 4, not 3. Sum a finite 4..20 prefix
            # and bound the remaining decreasing series by its tail integral.
            taulo = 2*h*sum((phi(arb(k)) for k in range(N+1, 21)), arb(0))
            tauhi = taulo+2*phi(arb(20))/20
            millsL = 2*phi(L)/L
            self.assertTrue(tauhi < millsL)
            fullmass_upper = 1+h*phi(arb(0))
            self.assertTrue(Shi < fullmass_upper)
            RESULTS.append({"test": "zero_loading_gaussian_alias_and_endpoint_tail",
                "status": "passed", "h": "1", "L": "3", "N": N,
                "first_omitted_index": 4, "alias_exact_upper": record(exact_alias_upper),
                "alias_analytic_bound": record(theoretical_alias),
                "discrete_axis_tail_upper": record(tauhi), "Mills_L_bound": record(millsL),
                "all_strict_comparisons": True, "PMF_evaluations": 0})
        finally:
            ctx.prec = previous

    def test_exact_comparisons_and_invalid_inputs(self):
        count = 0
        for m in (0, 1, 17):
            for e in range(-12, 13):
                for target in ("0", "1/3", "1/1000000000000", "100"):
                    ep = {"mantissa": str(m), "exponent_of_two": e}
                    self.assertEqual(bounds.dyadic_lt_q(ep, target),
                                     Fraction(m)*Fraction(2)**e < Fraction(target))
                    count += 1
        for kwargs in ({"h": .125}, {"limit": "1/3"}, {"strip": "4"},
                       {"precision_bits": 192.0}, {"angles": ()}, {"target": "0"}):
            with self.assertRaises(ValueError):
                bounds.teacher_l1_bounds(**kwargs)
        RESULTS.append({"test": "exact_comparisons_and_invalid_inputs", "status": "passed",
                        "comparisons": count, "rejected_inputs": 6})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.write_receipt:
        sources = [ROOT/"script/certified_teacher_bounds.py", Path(__file__).resolve(),
                   ROOT/"docs/08_research_研究工作/CertifiedTeacher_v1_20261008/QUADRATURE_PROOF.md"]
        out = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "action": "manufactured_scalar_teacher_bound_qualification", "status": "passed",
            "precision_bits": 192, "tests_run": result.testsRun, "results": RESULTS,
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "real_PMF_computed": False, "scientific_data_reads": 0, "random_draws": 0,
            "acb_integral_calls": 0, "finite_sum_arithmetic_certified": False,
            "scientific_execution_frozen": False,
            "unclosed": ["certified PMF finite-sum arithmetic", "later consumption of a PMF enclosure"]}
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        print(RECEIPT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
