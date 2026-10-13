"""Manufactured candidate grids/scalar bounds; no real center/teacher reads."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from fractions import Fraction as Q
from io import StringIO
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import certified_teacher_risk as r


def synthetic_candidate():
    # An algebra fixture with total Q mass one and sum B zero. It is not a
    # scientific model evaluation or a claim about the real saved center.
    a, h = [Q(0)]*8, [Q(-1)]*8
    q = r.aq(Q(1, 256))
    bs = [r.aq(Q(-1, 1000))]*128+[r.aq(Q(1, 1000))]*128
    return {"a_exact": [r.dyadic(x) for x in a], "h_exact": [r.dyadic(x) for x in h],
        "finite_sum": {"status": "completed", "Q_finite_sum": [r.record(q) for _ in range(256)],
            "B_h_finite_sum": [r.record(b) for b in bs]},
        "analytic_bounds": r.strip_bounds(a, h)}


def synthetic_teacher():
    return {"pmf_finite_sum": [r.record(r.aq(Q(3, 512))) for _ in range(128)]
        +[r.record(r.aq(Q(1, 512))) for _ in range(128)]}


class Manufactured(unittest.TestCase):
    def setUp(self):
        self.previous = r.ctx.prec
        r.ctx.prec = r.BITS

    def tearDown(self):
        r.ctx.prec = self.previous

    def test_single_observation_known_half_probability_and_zero_B(self):
        result = r.finite_candidate([Q(1, 2)], [Q(-1, 3)], step=Q(1), N=1)
        expected_mass = (r.arb(1)+2*(-r.arb(1)/2).exp())/(2*r.arb.pi()).sqrt()
        for value in result["Q_finite_sum"]:
            self.assertTrue(r.from_record(value).overlaps(expected_mass/2))
        for value in result["B_h_finite_sum"]:
            self.assertTrue(r.from_record(value).contains(0))
        self.assertEqual(result["nodes_completed"], 3)
        self.assertEqual(result["finite_scalar_integrals_completed"], 4)

    def test_zero_loadings_probability_equal_across_labels(self):
        result = r.finite_candidate([Q(0), Q(0)], [Q(1), Q(2)], step=Q(1), N=1)
        probabilities = [r.from_record(x) for x in result["Q_finite_sum"]]
        self.assertTrue(all(x.overlaps(probabilities[0]) for x in probabilities))
        self.assertTrue(all(r.from_record(x).contains(0) for x in result["B_h_finite_sum"]))

    def test_nonzero_B_derivative_sign_in_two_observation_fixture(self):
        result = r.finite_candidate([Q(1, 2)]*2, [Q(1)]*2, step=Q(1), N=1)
        values = [r.from_record(x) for x in result["B_h_finite_sum"]]
        self.assertTrue(values[0] > 0)
        self.assertTrue(values[3] > 0)
        self.assertTrue(values[1] < 0)
        self.assertTrue(values[2] < 0)
        self.assertTrue(sum(values, r.arb(0)).contains(0))

    def test_budget_stop_retains_partial_sums_without_promotion(self):
        with patch.object(r.time, "perf_counter", side_effect=[0., 2., 2.]):
            result = r.finite_candidate([Q(1)], [Q(1)], step=Q(1), N=1, hard_seconds=1)
        self.assertEqual(result["status"], "budget_stopped")
        self.assertEqual(result["nodes_completed"], 0)
        self.assertEqual(result["finite_scalar_integrals_completed"], 0)
        candidate = synthetic_candidate()
        candidate["finite_sum"]["status"] = "budget_stopped"
        with self.assertRaises(ValueError):
            r.candidate_ratios(candidate)

    def test_strip_poles_rejected_even_if_cosine_positive_again(self):
        for loading in (Q(4), Q(13)):
            with self.assertRaises(ValueError):
                r.strip_bounds([loading], [Q(1)])

    def test_strip_bound_normalization_and_tail_terms(self):
        budget = r.strip_bounds([Q(1, 2)]*8, [Q(-1, 10)]*8)
        self.assertEqual(budget["node_count"], 177)
        self.assertEqual(budget["endpoint_weights"], "full")
        hn = r.from_record(budget["constants"]["h_norm"])
        eb = r.from_record(budget["errors"]["B_h_total"])
        eu = r.from_record(budget["errors"]["B_unit_total"])
        self.assertTrue((eu-eb/hn).contains(0))
        self.assertTrue(r.from_record(budget["errors"]["Q_total"]) < r.aq(Q(1, 10**19)))
        self.assertTrue(r.from_record(budget["errors"]["B_unit_total"]) < r.aq(Q(1, 10**19)))

    def test_interval_endpoint_roundtrip_and_rejection(self):
        value = r.arb(1)/3
        saved = r.record(value)
        rebuilt = r.from_record(saved)
        self.assertTrue(rebuilt.contains(value))
        changed = dict(saved, lower_exact_dyadic=saved["upper_exact_dyadic"],
            upper_exact_dyadic=saved["lower_exact_dyadic"])
        with self.assertRaises(ValueError):
            r.from_record(changed)

    def test_error_enlargement_contains_both_endpoints(self):
        value, error = r.aq(Q(1)), r.aq(Q(1, 100))
        enlarged = r.enlarge(value, error)
        self.assertTrue(enlarged.contains(r.aq(Q(99, 100))))
        self.assertTrue(enlarged.contains(r.aq(Q(101, 100))))

    def test_direction_sign_and_L1_budget_not_repeated_256_times(self):
        teacher_bound = {"bounds": {"total_quadrature_L1": r.record(r.aq(Q(1, 1000)))}}
        result = r.compose(synthetic_candidate(), synthetic_teacher(), teacher_bound)
        self.assertTrue(result["true_center_risk_direction_upper_below_minus_point020"])
        rho = r.from_record(result["true_center_risk_direction_enclosure"])
        self.assertTrue(rho < 0)
        allowance = r.from_record(result["teacher_L1_transfer_allowance"])
        self.assertTrue(allowance < r.aq(Q(1, 10000)))
        self.assertTrue(result["teacher_total_L1_not_repeated_per_label"])
        self.assertFalse(result["true_F8_stationary_existence_certified"])
        self.assertFalse(result["teacher_or_candidate_mass_renormalized"])

    def test_changed_B_sign_does_not_pass_negative_direction_gate(self):
        candidate = synthetic_candidate()
        candidate["finite_sum"]["B_h_finite_sum"] = [r.record(-r.from_record(x))
            for x in candidate["finite_sum"]["B_h_finite_sum"]]
        teacher_bound = {"bounds": {"total_quadrature_L1": r.record(r.aq(Q(0)))}}
        result = r.compose(candidate, synthetic_teacher(), teacher_bound)
        self.assertFalse(result["true_center_risk_direction_upper_below_minus_point020"])
        self.assertFalse(result["center_risk_direction_certified_negative"])

    def test_candidate_Q_nonpositive_enclosure_rejected(self):
        candidate = synthetic_candidate()
        candidate["finite_sum"]["Q_finite_sum"][0] = r.record(r.arb(0))
        with self.assertRaises(ValueError):
            r.candidate_ratios(candidate)

    def test_saved_strip_budget_tampering_rejected(self):
        candidate = synthetic_candidate()
        candidate["analytic_bounds"]["N"] = 89
        with self.assertRaises(ValueError):
            r.candidate_ratios(candidate)

    def test_teacher_missing_label_rejected(self):
        teacher = synthetic_teacher()
        teacher["pmf_finite_sum"].pop()
        teacher_bound = {"bounds": {"total_quadrature_L1": r.record(r.aq(Q(0)))}}
        with self.assertRaises(ValueError):
            r.compose(synthetic_candidate(), teacher, teacher_bound)

    def test_check_arithmetic_pass_retains_false_scientific_target(self):
        result = {"true_center_risk_direction_upper_below_minus_point020": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manufactured_composition.json"
            path.write_text("{}", encoding="utf-8")
            stream = StringIO()
            with patch.object(r, "COMPOSED", path), patch.object(r.sys, "argv", ["risk.py", "--check"]), patch.object(r, "check_candidate", return_value={}), patch.object(r, "compose_files", return_value=result), patch.object(r, "read", return_value=result), redirect_stdout(stream):
                code = r.main()
            output = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(output["status"], "passed")
            self.assertFalse(output["true_center_risk_direction_upper_below_minus_point020"])


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Manufactured))
    receipt = {"schema": "certified_teacher_risk_manufactured_v1", "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": [(str(case), text) for case, text in result.failures],
        "errors": [(str(case), text) for case, text in result.errors],
        "precision_bits": r.BITS, "source_sha256": {"script/certified_teacher_risk.py": r.sha(r.__file__),
            "script/certified_teacher_risk_test.py": r.sha(__file__)},
        "manufactured_finite_grids_and_scalar_algebra_only": True,
        "real_candidate_or_teacher_input_reads": 0, "real_scientific_integrals": 0,
        "new_kernel_calls": 0, "new_random_draws": 0, "scientific_NPZ_reads": 0}
    r.OUT.mkdir(parents=True, exist_ok=True)
    (r.OUT/"RISK_MANUFACTURED_CHECK.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "real_scientific_integrals": 0, "new_kernel_calls": 0}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
