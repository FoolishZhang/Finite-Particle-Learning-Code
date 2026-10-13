"""Manufactured checks only; no real nominal output or scientific source reads."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import local_witness_feasibility_certificate as c


def diagonal(n, value):
    return [[value if i == j else 0 for j in range(n)] for i in range(n)]


def float_record(value):
    n, d = value.as_integer_ratio()
    return {"value": value, "hex": value.hex(), "numerator": str(n), "denominator": str(d)}


def fixture():
    g = [0.]*16
    gi = [0.]*8+[.002]*8
    theta = [0.]*8+[.5]*8
    return {"center": {"theta": theta, "original_theta": theta},
        "rules": {"D1": {"blocks": [
            {"K": 8, "gradient": g, "H_ba": diagonal(16, .1), "Hperp": diagonal(8, .02)},
            {"K": "inf", "gradient": gi}]}},
        "risk_direction": {"h_dyadic": [float_record(.002) for _ in range(8)]}}


class Manufactured(unittest.TestCase):
    def setUp(self):
        self.previous = c.ctx.prec
        c.ctx.prec = c.BITS

    def tearDown(self):
        c.ctx.prec = self.previous

    def test_positive_definite_LDL(self):
        result = c.ldlt_positive([[Q(2), Q(1)], [Q(1), Q(2)]])
        self.assertTrue(result["positive_definite_certified"])
        self.assertEqual(len(result["pivots"]), 2)

    def test_indefinite_rejected(self):
        result = c.ldlt_positive([[Q(1), Q(2)], [Q(2), Q(1)]])
        self.assertFalse(result["positive_definite_certified"])
        self.assertEqual(result["first_rejected_pivot"], 1)

    def test_singular_rejected(self):
        self.assertFalse(c.ldlt_positive([[Q(1), Q(1)], [Q(1), Q(1)]])["positive_definite_certified"])

    def test_near_singular_unresolved_sign_rejected(self):
        # Exact PD, but 192-bit intervals cannot resolve the cancellation.
        eps = Q(1, 2**400)
        result = c.ldlt_positive([[Q(1), Q(1)], [Q(1), Q(1)+eps]])
        self.assertFalse(result["positive_definite_certified"])
        self.assertEqual(result["first_rejected_pivot"], 1)

    def test_tiny_but_exact_diagonal_can_be_certified(self):
        self.assertTrue(c.ldlt_positive(diagonal(2, Q(1, 2**400)))["positive_definite_certified"])

    def test_asymmetric_even_below_precision_rejected(self):
        with self.assertRaises(ValueError):
            c.ldlt_positive([[Q(1), Q(0)], [Q(1, 2**400), Q(1)]])

    def test_generalized_norm_test_not_naive_identity_shift(self):
        h = diagonal(2, Q(1, 100))
        v = [[Q(2)], [Q(0)]]
        naive = c.shift(c.congruence(h, v), c.TRANSVERSE_FLOOR)
        correct = c.congruence(c.shift(h, c.TRANSVERSE_FLOOR), v)
        self.assertTrue(c.ldlt_positive(naive)["positive_definite_certified"])
        self.assertFalse(c.ldlt_positive(correct)["positive_definite_certified"])

    def test_fixed_rational_basis_exact_orthogonality(self):
        a = [Q(i+1, 8) for i in range(8)]
        v = c.fixed_slice_basis(a)
        self.assertTrue(all(sum((a[i]*v[i][j] for i in range(8)), Q(0)) == 0 for j in range(7)))
        self.assertNotEqual(sum((v[i][0]*v[i][0] for i in range(8)), Q(0)), Q(1))
        projected = c.congruence(c.shift(diagonal(8, Q(1, 50)), c.TRANSVERSE_FLOOR), v)
        self.assertTrue(c.ldlt_positive(projected)["positive_definite_certified"])

    def test_nonorthonormal_pair_lift_wrong_norm_factor(self):
        # g=(1,...,1): ||U+.T g||²=8, whereas ||S.T g||²=16.
        g = [Q(1)]*8
        correct_squared = sum(((g[i]+g[7-i])**2/2 for i in range(4)), Q(0))
        naive_squared = sum(((g[i]+g[7-i])**2 for i in range(4)), Q(0))
        self.assertEqual(correct_squared, 8)
        self.assertEqual(naive_squared, 16)
        self.assertTrue((c.projected_gradient_norm(g)-c.arb(8).sqrt()).contains(0))

    def test_L_safe_integer_exact_proof(self):
        proof = c.safe_lipschitz_proof()
        self.assertEqual(proof["L_F_safe_integer"], 14069)
        self.assertEqual(proof["exact_fraction_checks_passed"], [True]*3)

    def test_JSON_decimal_roundtrips_binary64(self):
        self.assertEqual(c.binary64(json.loads("0.1")), Q.from_float(.1))
        self.assertNotEqual(c.binary64(json.loads("0.1")), Q(1, 10))
        self.assertEqual(c.binary64(.1.hex()), Q.from_float(.1))

    def test_invalid_binary64_rejected(self):
        for v in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.binary64(v)

    def test_symmetric_matrix_rounding_error_recorded(self):
        matrix = [[1., .1], [math.nextafter(.1, math.inf), 1.]]
        sym, record = c.dyadic_matrix(matrix, 2)
        self.assertEqual(sym[0][1], sym[1][0])
        self.assertGreater(int(record["sym_rounding_error_Frobenius_bound"]["upper_exact_dyadic"]["mantissa"]), 0)
        self.assertEqual(record["dimension"], 2)

    def test_valid_manufactured_certificate(self):
        cert = c.make_certificate(fixture())
        self.assertTrue(cert["strict_saved_matrix_checks_passed"])
        self.assertTrue(cert["conditional_budget_arithmetic_passed"])
        self.assertFalse(cert["exact_stationary_point_existence_certified"])
        self.assertFalse(cert["true_F8_Hessian_positive_certified"])
        self.assertFalse(cert["true_risk_derivative_certified"])
        self.assertFalse(cert["C1_D1_difference_used_as_true_error"])
        self.assertEqual(cert, json.loads(json.dumps(cert)))

    def test_bad_saved_matrix_retains_failure(self):
        data = fixture()
        data["rules"]["D1"]["blocks"][0]["H_ba"] = diagonal(16, .049)
        cert = c.make_certificate(data)
        self.assertFalse(cert["strict_saved_matrix_checks_passed"])
        self.assertFalse(cert["radial_shifted_LDL"]["positive_definite_certified"])

    def test_asymmetric_center_rejected(self):
        data = fixture()
        data["center"]["theta"][8] = math.nextafter(.5, math.inf)
        with self.assertRaises(ValueError):
            c.make_certificate(data)

    def test_bias_center_nonzero_rejected(self):
        data = fixture()
        data["center"]["theta"][0] = 1e-100
        with self.assertRaises(ValueError):
            c.make_certificate(data)

    def test_missing_original_rejected(self):
        data = fixture()
        del data["center"]["original_theta"]
        with self.assertRaises(ValueError):
            c.make_certificate(data)

    def test_center_not_old_pair_mean_rejected(self):
        data = fixture()
        data["center"]["original_theta"] = [0.]*8+[.6]*8
        with self.assertRaises(ValueError):
            c.make_certificate(data)

    def test_frozen_h_integrity_and_pair_average(self):
        data = fixture()
        data["risk_direction"]["h_dyadic"][0]["numerator"] = "1"
        with self.assertRaises(ValueError):
            c.make_certificate(data)
        data = fixture()
        data["risk_direction"]["h_dyadic"] = [float_record(.003)]*8
        with self.assertRaises(ValueError):
            c.make_certificate(data)

    def test_risk_direction_keeps_rounded_pair_mean(self):
        data = fixture()
        ga = [.002]*8
        ga[0], ga[7] = .1, math.nextafter(.1, math.inf)
        data["rules"]["D1"]["blocks"][1]["gradient"] = [0.]*8+ga
        hh = [.5*(ga[i]+ga[7-i]) for i in range(4)]
        hh += list(reversed(hh))
        data["risk_direction"]["h_dyadic"] = [float_record(v) for v in hh]
        cert = c.make_certificate(data)
        got = cert["fixed_risk_direction"]["h_exact_rationals"][0]
        actual = Q(int(got["numerator"]), int(got["denominator"]))
        self.assertEqual(actual, Q.from_float(hh[0]))
        self.assertNotEqual(actual, (Q.from_float(ga[0])+Q.from_float(ga[7]))/2)

    def test_remaining_gradient_budget_can_reject(self):
        result = c.conditional_budget([Q(1, 2)]*8, c.aq(Q(1, 1000000)))
        self.assertFalse(result["checks"]["positive_future_gradient_error_budget"])
        self.assertTrue(result["checks"]["gradient_outward_condition"])

    def test_freeze_missing_or_wrong_baseline_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"FREEZE.json"
            with self.assertRaises(RuntimeError):
                c.require_freeze(path)
            path.write_text(json.dumps({"baseline": "wrong", "sources": {}}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                c.require_freeze(path)

    def test_freeze_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"FREEZE.json"
            path.write_text(json.dumps({"baseline": c.BASELINE, "sources": {}}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                c.require_freeze(path)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Manufactured)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"schema": "local_witness_certificate_manufactured_check_v1",
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [(str(case), details) for case, details in result.failures],
        "errors": [(str(case), details) for case, details in result.errors],
        "precision_bits": c.BITS, "python_flint_version": c.flint.__version__,
        "certificate_source_sha256": c.sha(c.__file__), "test_source_sha256": c.sha(__file__),
        "scope": "manufactured rational/dyadic matrices and scalar budgets only",
        "real_nominal_outputs_read": False, "scientific_NPZ_reads": 0,
        "new_kernel_calls": 0, "new_Gaussian_Laplace_integrals": 0,
        "exact_stationary_point_existence_certified": False}
    output = c.ART/"CERTIFICATE_MANUFACTURED_CHECK.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "new_kernel_calls": 0, "new_Gaussian_Laplace_integrals": 0}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
