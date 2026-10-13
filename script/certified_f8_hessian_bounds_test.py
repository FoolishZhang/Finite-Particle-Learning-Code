"""Manufactured operator/algebra/scalar checks, never a study Hessian."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
import certified_f8_hessian_bounds as bounds
from certified_f8_hessian_bounds import aq, arb, ctx, record

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/"artifacts/certified_f8_hessian_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json"
RESULTS = []


class ScalarTests(unittest.TestCase):
    def test_fixed_planning_operator_budget(self):
        previous = ctx.prec
        data = bounds.hessian_operator_bounds(bounds.PLANNING_CENTER_PAIRS)
        self.assertEqual(ctx.prec, previous)
        self.assertTrue(all(c["strictly_below"] for c in data["comparisons"]))
        self.assertEqual(data["Hba_dimension"], 16)
        self.assertEqual(data["Hp_dimension"], 8)
        self.assertEqual(data["grid"]["inner_nodes"], 881)
        self.assertEqual(data["grid"]["outer_nodes"], 769)
        self.assertEqual(Fraction(data["exact_inputs"]["teacher_error_L1_upper"]),
                         Fraction("4.894e-21"))
        self.assertEqual(data["bounds"]["teacher_global_Hba_operator"]["lower_exact_dyadic"],
                         {"mantissa": "585", "exponent_of_two": -2})
        RESULTS.append({"test": "fixed_planning_operator_budget", "status": "passed",
                        "result": data, "real_Hessian_evaluations": 0})

    def test_exact_distinct_particle_covariance_identity(self):
        w = [Fraction(1, 3), Fraction(2, 3)]
        G = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
        mean = [sum(w[k]*G[k][j] for k in range(2)) for j in range(2)]
        same = [[sum(w[k]*(1-w[k])*G[k][i]*G[k][j] for k in range(2))
                 for j in range(2)] for i in range(2)]
        cross = [[sum(w[k]*w[l]*G[k][i]*G[l][j] for k in range(2)
                      for l in range(2) if k != l) for j in range(2)] for i in range(2)]
        covariance = [[sum(w[k]*G[k][i]*G[k][j] for k in range(2))-mean[i]*mean[j]
                       for j in range(2)] for i in range(2)]
        self.assertEqual([[same[i][j]-cross[i][j] for j in range(2)] for i in range(2)],
                         covariance)
        self.assertEqual(covariance, [[Fraction(2, 9), Fraction(-2, 9)],
                                     [Fraction(-2, 9), Fraction(2, 9)]])
        self.assertEqual(covariance[0][0]*covariance[1][1]-covariance[0][1]**2, 0)
        RESULTS.append({"test": "exact_distinct_particle_covariance_identity", "status": "passed",
                        "covariance_exact": [[str(v) for v in row] for row in covariance],
                        "eigenvalues_exact": ["0", "4/9"], "PSD_verified_exactly": True})

    def test_constant_bank_outer_matrix_density(self):
        # Invented two-atom scalar bank: S=1, w=(1/3,2/3),G=(1,-1/2),
        # B=(1/4,1/5); Bbar=13/60,Cov=1/2. No logistic study data.
        previous = ctx.prec; ctx.prec = 192
        try:
            B, C = aq("13/60"), arb(1)/2
            h, lo, hi, d = arb(1)/8, arb(-32), arb(10), arb(1)/2
            total = arb(0)
            for j in range(337):
                s = (lo+j*h).exp()
                total += h*(-s).exp()*(s*B-s*s*C)
            exact = B-C
            q = (-2*arb.pi()*d/h).exp()
            M = B/d.cos()+C/d.cos()**2
            alias = 2*M*q/(1-q)
            lower = B*h*lo.exp()/h.expm1()+C*h*(2*lo).exp()/(2*h).expm1()
            shi = hi.exp()
            upper = (B+C*(1+shi))*(-shi).exp()
            error = abs(total-exact)
            self.assertTrue(error < alias+lower+upper)
            RESULTS.append({"test": "constant_bank_outer_matrix_density", "status": "passed",
                "finite_scalar_sum": record(total), "exact_integral": "-17/60",
                "absolute_error": record(error), "analytic_bound": record(alias+lower+upper),
                "strict_comparison": True, "real_Hessian_evaluations": 0})
        finally:
            ctx.prec = previous

    def test_second_moment_discrete_unimodal_tail(self):
        previous = ctx.prec; ctx.prec = 192
        try:
            h, e = arb(1)/8, arb(1).exp(); cases = []
            for s0q in ("1/100", "1", "2", "10"):
                s0 = aq(s0q); prefix = arb(0)
                for j in range(1, 513):
                    s = s0*(j*h).exp()
                    prefix += h*s*s*(-s).exp()
                s512 = s0*(512*h).exp()
                grid_upper = prefix+(1+s512)*(-s512).exp()
                continuous = (1+s0)*(-s0).exp()
                unimodal = continuous+8*h/e**2
                self.assertTrue(grid_upper < unimodal)
                monotone = Fraction(s0q) >= 2
                if monotone:
                    self.assertTrue(grid_upper < continuous)
                cases.append({"boundary_s_exact": s0q, "grid_upper": record(grid_upper),
                    "unimodal_upper": record(unimodal), "strict_comparison": True,
                    "decreasing_integral_comparison": monotone})
            RESULTS.append({"test": "second_moment_discrete_unimodal_tail", "status": "passed",
                            "cases": cases})
        finally:
            ctx.prec = previous

    def test_exact_inputs_and_comparisons(self):
        count = 0
        for m in (0, 1, 17):
            for e in range(-12, 13):
                for q in ("0", "1/3", "1/1000", "100"):
                    self.assertEqual(bounds.dyadic_lt_q(
                        {"mantissa": str(m), "exponent_of_two": e}, q),
                        Fraction(m)*Fraction(2)**e < Fraction(q))
                    count += 1
        for kwargs in ({"K": 8.0}, {"K": 1}, {"hz": .025}, {"L": "1"},
                       {"dz": "1"}, {"dv": "2"}, {"teacher_mass_upper": "-1"},
                       {"teacher_error_L1_upper": "-1"}, {"precision_bits": True}):
            with self.assertRaises(ValueError):
                bounds.hessian_operator_bounds(bounds.PLANNING_CENTER_PAIRS, **kwargs)
        RESULTS.append({"test": "exact_inputs_and_comparisons", "status": "passed",
                        "comparisons": count, "rejected_inputs": 9})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScalarTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.write_receipt:
        sources = [ROOT/"script/certified_f8_hessian_bounds.py", Path(__file__).resolve(),
            ROOT/"docs/08_research_研究工作/CertifiedF8Hessian_v1_20261008/QUADRATURE_PROOF.md"]
        receipt = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "action": "manufactured_scalar_Hessian_operator_budget", "status": "passed",
            "tests_run": result.testsRun, "precision_bits": 192, "results": RESULTS,
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "real_Hessian_evaluations": 0, "scientific_data_reads": 0, "random_draws": 0,
            "acb_integral_calls": 0, "compute_arithmetic_included": False,
            "old_saved_matrix_distance_included": False, "scientific_execution_frozen": False,
            "unclosed": ["actual finite-grid interval matrix arithmetic",
                         "teacher mass/error comparison to previous certificate",
                         "Arb distance to old saved matrices", "strict local Hessian signs"]}
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        with RECEIPT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
        print(RECEIPT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
