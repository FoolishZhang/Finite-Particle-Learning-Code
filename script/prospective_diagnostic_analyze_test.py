"""Explicit manufactured scalar scenarios; never load development arrays."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prospective_diagnostic_analyze as analyze
import numpy as np


def rows_fixture():
    rows = []
    for i, unit in enumerate(analyze.dev.units()):
        A, B = i % 4 == 0, i % 4 in (0, 1)
        gain = (.002 if A else -.0004 if B else .0007) + i * .00001
        row = dict(unit=unit["id"], D="D" + str(unit["teacher"] + 1), teacher_index=unit["teacher"], N=unit["N"], rep=unit["rep"],
            A=A, B=B, full_level="D3" if A else "D0", full_reasons=["suppression_and_release"] if A else ["low_resolution_band"],
            ref_reasons=["reference_negative"] if B else ["reference_unresolved"],
            D0_reason_categories=dict(computation_or_solver=[], resolution=[] if A else ["low_resolution_band"]),
            risk_KL=dict(LOW=.03 + gain, HIGH=.03, REF=.025), delta_R=gain,
            risk_qualified=True, risk_qualified_by_arm=dict(LOW=True, HIGH=True, REF=True),
            cost_seconds=dict(anchor=2. + .1 * i, kick_computation=.001, kick_generation=.002,
                common_C0=2.003 + .1 * i, full_diagnostic=10., reference_diagnostic=4., finite_extra_diagnostic=6.,
                LOW=1., HIGH=10., REF=7., input_generation_total=.1, data_generation=.02,
                shared_HIGH_prior_generation=.07, risk_evaluation=.5))
        row["policy_algebra"] = analyze.policy_algebra(row)
        rows.append(row)
    return rows


class AnalyzeTests(unittest.TestCase):
    def test_risk_policy_and_cost_identities(self):
        rows = rows_fixture()
        for row in rows:
            data = row["policy_algebra"]
            self.assertLess(max(abs(v) for v in data["identity_residuals"].values()), 1e-12)
            self.assertEqual(data["policies"]["FULL"]["risk"], row["risk_KL"]["HIGH" if row["A"] else "LOW"])
            self.assertEqual(data["policies"]["reference_only"]["risk"], row["risk_KL"]["HIGH" if row["B"] else "LOW"])
            self.assertEqual(data["policies"]["always_LOW"]["wall_seconds"], row["cost_seconds"]["common_C0"] + 1.)
        damaged = copy.deepcopy(rows[0]); damaged["A"], damaged["B"] = True, False
        with self.assertRaisesRegex(AssertionError, "imply"):
            analyze.policy_algebra(damaged)

    def test_all_denominators_and_information_increment(self):
        rows = rows_fixture(); s = analyze.summarize(rows, [1., 2., 3.])
        g = np.array([r["delta_R"] for r in rows]); A = np.array([r["A"] for r in rows]); B = np.array([r["B"] for r in rows])
        d = s["descriptive_full_denominator"]
        self.assertEqual(d["n"], 12)
        self.assertEqual(d["E_A_delta"], float(np.mean(A * g)))
        self.assertEqual(d["E_stay_delta"], float(np.mean((1 - A) * g)))
        self.assertEqual(d["I"], float(np.mean((A.astype(int) - B.astype(int)) * g)))
        self.assertEqual(d["G_raise"], float(g[A].mean()))
        self.assertTrue(s["primary_risk_totals"]["evaluable"])
        self.assertEqual(len(s["condition_rows"]), 6)
        self.assertTrue(all(c["n"] == 2 for c in s["condition_rows"]))
        self.assertEqual(sum(c["n"] for c in s["A_B_cells"]), 12)

    def test_unqualified_pair_blocks_primary_not_descriptions(self):
        rows = rows_fixture(); rows[5]["risk_qualified"] = False
        s = analyze.summarize(rows, [1., 2., 3.])
        self.assertFalse(s["primary_risk_totals"]["evaluable"])
        self.assertTrue(all(v is None for v in s["primary_risk_totals"]["values"].values()))
        self.assertEqual(s["descriptive_full_denominator"]["n"], 12)
        self.assertEqual(s["primary_risk_totals"]["failed_units"], [rows[5]["unit"]])
        self.assertTrue(all(p["risk_denominator"] == 12 for p in s["policy_risk_cost_table"]))
        self.assertTrue(next(p for p in s["policy_risk_cost_table"] if p["policy"] == "REF_fit")["risk_evaluable"])

    def test_empty_groups_no_zero_imputation(self):
        rows = rows_fixture()
        for row in rows:
            row["A"], row["B"] = False, False
            row["policy_algebra"] = analyze.policy_algebra(row)
        m = analyze.metrics(rows)
        self.assertIsNone(m["G_raise"]); self.assertIsNone(m["G_raise_minus_G_stay"])
        self.assertEqual(m["E_A_delta"], 0.)
        b = analyze.width_sensitivity(rows)
        self.assertEqual(b["empty_raise_draws"], 10000)
        self.assertEqual(b["result"]["G_raise"]["defined_draws"], 0)
        self.assertIsNone(b["result"]["G_raise"]["width"])
        self.assertFalse(b["undefined_draws_replaced"])
        self.assertEqual(b["result"]["I"]["defined_draws"], 10000)

    def test_fixed_complete_unit_bootstrap_reproducible(self):
        before = np.random.get_state(); rows = rows_fixture()
        first, second = analyze.width_sensitivity(rows), analyze.width_sensitivity(rows)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], [202610071601, 0, 99, 4])
        self.assertEqual(first["pseudo_units_per_draw"], 72)
        self.assertEqual(first["draws_per_cell"], 12)
        self.assertFalse(first["power_guarantee"])
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2:], after[2:])

    def test_cost_ceiling_includes_common_anchor_and_kick(self):
        rows = rows_fixture(); s = analyze.summarize(rows, [1., 2., 3.])
        expected = np.mean([(1 - int(r["A"])) * 9. - .2 * (r["cost_seconds"]["common_C0"] + 10.) for r in rows])
        self.assertEqual(s["cost_ceilings"]["FULL"]["necessary_20pct_mean_diagnostic_ceiling"], float(expected))
        self.assertFalse(s["cost_ceilings"]["FULL"]["measured_prototype_satisfies_cost_inequality"])
        for row in rows:
            row["cost_seconds"]["anchor"] += 1000.; row["cost_seconds"]["common_C0"] += 1000.
            row["policy_algebra"] = analyze.policy_algebra(row)
        expensive = analyze.summarize(rows, [1., 2., 3.])
        self.assertLess(expensive["free_diagnosis_all_LOW_saving_fraction"], .02)

    def test_missing_time_rejected_and_endpoint_norms_are_algebraic(self):
        with self.assertRaises(KeyError): analyze.required_number({}, "wall_seconds")
        with self.assertRaises(ValueError): analyze.required_number(dict(wall_seconds=float("nan")), "wall_seconds")
        geo = analyze.endpoint_geometry(dict(b=np.array([3., 4.]), W=np.array([[3., 0.], [0., 2.]])))
        self.assertEqual(geo["bias_L2"], 5.)
        self.assertAlmostEqual(geo["W_Frobenius"], np.sqrt(13.))
        self.assertAlmostEqual(geo["s2_over_s1"], 2. / 3., places=14)
        self.assertFalse(geo["path_causal_attribution"])
        zero = analyze.endpoint_geometry(dict(b=np.zeros(2), W=np.zeros((2, 2))))
        self.assertIsNone(zero["s2_over_s1"])

    def test_real_entry_rejects_missing_final_lock_before_restore(self):
        with tempfile.TemporaryDirectory(prefix="prospective_lock_fixture_") as td:
            with patch.object(analyze, "OUT", Path(td)), patch.object(analyze.dev, "restore", side_effect=AssertionError("must not restore")):
                with self.assertRaisesRegex(RuntimeError, "EVALUATION_LOCK"):
                    analyze.locked_inputs()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalyzeTests))
    print(json.dumps(dict(passed=result.wasSuccessful(), tests_run=result.testsRun, manufactured_only=True,
                         actual_development_inputs_used=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
