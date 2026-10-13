"""Manufactured K transport, continuation/refinement and four-cell checks."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import rankone_continuation_compute as driver
import numpy as np

DIAGNOSTICS = {}


def point(K, sign, *, radial=True, name=None, theta=None):
    return dict(point_id=name or "fixture_" + str(K), K=K, theta=([.1, -.1, .5, .6] if theta is None else list(theta)),
                radial_qualification=dict(qualified=radial), transverse_label=sign,
                resources=driver.base.new_counts())


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theta = np.array([.1, -.15, .35, .5])
        cls.q = np.array([.21, .24, .26, .29]) * (1. - 5e-16)
        cls.Ks = (9, 128, "inf")
        cls.base_config = copy.deepcopy(driver.base.config_dump())
        cls.base_sha = driver.sha(driver.base.__file__)
        cls.kernel_sha = driver.sha(driver.base.kernel.__file__)
        tick = time.perf_counter(); cls.results = {}
        for rule in ("C", "D"):
            r = driver.base.kernel.rankone_stationary(cls.theta[:2].copy(), cls.theta[2:].copy(),
                {"population": cls.q.copy()}, K_values=cls.Ks, **driver.base.RULES[rule])
            cls.results[rule] = driver.base.project_result(r)
        DIAGNOSTICS["actual_m2_K9_K128_inf_C_D_seconds"] = time.perf_counter() - tick

    def test_actual_large_K_and_integrated_kernel_qualification(self):
        C, D = self.results["C"], self.results["D"]
        self.assertEqual(tuple(C["K_values"]), self.Ks)
        for ki in range(3): self.assertTrue(driver.base.qualification(C, D, 0, ki)["qualified"])
        DIAGNOSTICS["C_D_loss_max_difference"] = float(np.max(abs(C["loss"] - D["loss"])))
        DIAGNOSTICS["C_D_gradient_max_difference"] = float(np.max(abs(C["gradient"] - D["gradient"])))
        DIAGNOSTICS["C_D_H_ba_max_difference"] = float(np.max(abs(C["H_ba"] - D["H_ba"])))
        self.assertEqual(driver.endpoint_Ks(9), (9, "inf"))
        self.assertEqual(driver.endpoint_Ks(128), (128, "inf"))
        self.assertEqual(driver.endpoint_Ks("inf"), ("inf",))
        np.testing.assert_array_equal(C["targets"][0], self.q)
        self.assertNotEqual(float(self.q.sum()), 1.)

    def test_actual_large_K_FG_finite_differences(self):
        tick = time.perf_counter(); step = 2e-4
        fg = np.empty((3, 4)); fh = np.empty((3, 4, 4))
        for j in range(4):
            plus, minus = self.theta.copy(), self.theta.copy()
            plus[j] += step; minus[j] -= step
            results = [driver.base.kernel.rankone_stationary(t[:2], t[2:], {"population": self.q.copy()},
                K_values=self.Ks, **driver.base.RULES["D"], compute_hessian=False) for t in (plus, minus)]
            fg[:, j] = (results[0]["loss"][0] - results[1]["loss"][0]) / (2 * step)
            fh[:, :, j] = (results[0]["gradient"][0] - results[1]["gradient"][0]) / (2 * step)
        gd = float(np.max(abs(fg - self.results["D"]["gradient"][0])))
        hd = float(np.max(abs(fh - self.results["D"]["H_ba"][0])))
        self.assertLess(gd, 2e-7); self.assertLess(hd, 2e-7)
        DIAGNOSTICS.update(K9_K128_inf_gradient_FD_max_error=gd, K9_K128_inf_H_ba_FD_max_error=hd,
                           eight_manufactured_FG_calls_seconds=time.perf_counter() - tick)

    def test_K_transport_into_unchanged_solver_and_snapshots(self):
        source = np.array([.4, -.3, .8, .6]); received = []
        for K in (9, 128, "inf"):
            center = np.array([.1, -.1, .5, .5]) + (0. if K == "inf" else 1. / K) * np.array([.01, 0., .01, 0.])
            buffer_g = np.empty((1, 1, 4)); buffer_H = np.empty((1, 1, 4, 4))
            def evaluator(theta, phase, hessian, context):
                received.append((K, phase, context["role"], theta.copy()))
                delta = theta - center; buffer_g[0, 0] = delta; buffer_H[0, 0] = np.eye(4)
                return dict(loss=np.array([[.5 * delta @ delta]]), gradient=buffer_g,
                            H_ba=buffer_H if hessian else None)
            trace = driver.base.stationary_solve(source.copy(), evaluator)
            np.testing.assert_allclose(trace["final_theta"], center, atol=2e-16, rtol=0)
            for row in trace["calls"]: np.testing.assert_array_equal(row["gradient"], row["theta"] - center)
            for left, right in zip(trace["accepted_states"], trace["accepted_states"][1:]):
                self.assertFalse(np.shares_memory(left["theta"], right["theta"]))
        self.assertEqual({r[0] for r in received}, {9, 128, "inf"})
        self.assertEqual(driver.base.config_dump(), self.base_config)
        self.assertEqual(driver.sha(driver.base.__file__), self.base_sha)
        self.assertEqual(driver.sha(driver.base.kernel.__file__), self.kernel_sha)

    def test_complete_coarse_LAST_chain_even_negative_or_unresolved(self):
        initial = point(8, "positive"); calls = []
        def solve(name, K, theta0, kind, context):
            calls.append(dict(name=name, K=K, theta0=theta0.copy(), context=copy.deepcopy(context)))
            answer = point(K, "negative" if len(calls) % 2 else "unresolved", radial=len(calls) % 2 == 1,
                name=name, theta=theta0 + np.array([.001, 0., .001, 0.]))
            theta0[:] = -123.  # Cannot mutate the predecessor snapshot.
            return answer
        old = copy.deepcopy(initial)
        rows = driver.coarse_chain(initial, solve)
        self.assertEqual([c["K"] for c in calls], list(driver.COARSE_K))
        self.assertEqual(len(rows), 10); self.assertEqual(initial, old)
        for i, call in enumerate(calls):
            np.testing.assert_array_equal(call["theta0"], rows[i]["theta"])
            self.assertEqual(call["context"]["predecessor_point_id"], rows[i]["point_id"])
        attempted = []
        def failing(name, K, theta0, kind, context):
            attempted.append(K)
            if K == 16: raise FloatingPointError("manufactured hard failure")
            return point(K, "positive", name=name, theta=theta0)
        with self.assertRaisesRegex(FloatingPointError, "manufactured hard failure"):
            driver.coarse_chain(initial, failing)
        self.assertEqual(attempted, [9, 10, 12, 16])

    def test_first_adjacent_finite_bracket_and_no_pair(self):
        self.assertIsNone(driver.select_first_bracket([point(8, "positive"), point(9, "negative", radial=False), point(10, "negative")]))
        self.assertIsNone(driver.select_first_bracket([point(8, "positive"), point("inf", "negative")]))
        rows = [point(8, "positive"), point(9, "negative"), point(10, "positive"), point(12, "negative")]
        pair = driver.select_first_bracket(rows)
        self.assertEqual([p["K"] for p in pair], [8, 9])
        calls = []
        report = driver.refine([point(8, "positive"), point("inf", "negative")], lambda *args: calls.append(args))
        self.assertEqual(report["status"], "no_usable_finite_bracket")
        self.assertEqual(calls, []); self.assertIsNone(report["final_pair"])

    def test_integer_refinement_starts_and_adjacent_stop(self):
        lo, hi = point(8, "positive", theta=[.1, -.1, .5, .6]), point(16, "negative", theta=[.2, 0., .7, .8])
        plans, decisions, calls = [], [], []
        def solve(name, K, theta0, kind, context):
            calls.append((K, theta0.copy(), context))
            expected = (1. - context["interpolation_fraction"]) * np.asarray(context["lower_theta"]) + context["interpolation_fraction"] * np.asarray(context["upper_theta"])
            np.testing.assert_array_equal(theta0, expected)
            return point(K, "positive" if K < 11 else "negative", name=name, theta=theta0 + .001)
        result = driver.refine([lo, hi], solve, lambda n, p: plans.append((n, p)), lambda i, d: decisions.append(d))
        self.assertEqual([c[0] for c in calls], [12, 10, 11])
        self.assertEqual(result["status"], "adjacent_integers")
        self.assertEqual([result["final_pair"][k]["K"] for k in ("lower", "upper")], [10, 11])
        self.assertEqual([d["action"] for d in decisions], ["replace_upper", "replace_lower", "replace_upper"])
        self.assertEqual(plans[0][0], "initial")
        self.assertEqual(result["iterations"], 3)

    def test_unresolved_retains_pair_and_six_call_limit(self):
        lo, hi = point(8, "positive"), point(16, "negative")
        for radial, sign, status in ((False, "negative", "stop_radial_or_numerical_unresolved"),
                                     (True, "unresolved", "stop_transverse_unresolved")):
            result = driver.refine([lo, hi], lambda name, K, theta0, kind, context: point(K, sign, radial=radial, name=name, theta=theta0))
            self.assertEqual(result["status"], status)
            self.assertEqual(result["final_pair"], dict(lower=lo, upper=hi))
            self.assertEqual(result["iterations"], 1)
        result = driver.refine([point(8, "positive"), point(2056, "negative")],
            lambda name, K, theta0, kind, context: point(K, "positive", name=name, theta=theta0))
        self.assertEqual(result["status"], "maximum_refinements_reached")
        self.assertEqual(result["iterations"], 6)

    def test_self_K_radial_gate_does_not_reject_negative_type(self):
        result = copy.deepcopy(self.results["D"])
        a = self.theta[2:]; V = driver.base.transverse_basis(a)
        result["gradient"][0, 0] = 0.; result["H_ba"][0, 0] = np.eye(4)
        for sign in (1., -1.):
            result["Hperp"][0, 0] = sign * V @ V.T
            driver.base.project_result(result)
            summary = driver.base.evaluation_summary(self.theta, result, self.q, .0)
            block = driver.find_block(summary, 9)
            classified = driver.own_classification(self.theta, block, dict(qualified=True))
            self.assertTrue(classified["radial_qualification"]["qualified"])
            self.assertEqual(classified["transverse_label"], "positive" if sign > 0 else "negative")
        block["gradient"][0] = 2e-8
        classified = driver.own_classification(self.theta, block, dict(qualified=True))
        self.assertFalse(classified["radial_qualification"]["qualified"])
        self.assertEqual(classified["transverse_label"], "unresolved")
        self.assertEqual(classified["raw_transverse_label_without_radial_qualification"], "negative")

    def test_direction_sign_tie_and_four_cell_algebra(self):
        block = dict(Hangle_projected_eigenvalues=[-.5], Hangle_projected_eigenvectors=[[-1.]])
        result = driver.same_angle_direction(block, np.array([.4, .5]))
        self.assertEqual(result["sign_coordinate"], 0)
        self.assertGreater(result["direction"][0], 0)
        self.assertAlmostEqual(np.linalg.norm(result["direction"]), 1.)
        self.assertAlmostEqual(sum(result["direction"]), 0.)
        decomposition = driver.cross_algebra(dict(LL=1., LH=3., HL=4., HH=9.))
        self.assertEqual(decomposition["total"], 8.)
        self.assertEqual(decomposition["K_first"], dict(K_change=2., theta_change=6.))
        self.assertEqual(decomposition["theta_first"], dict(theta_change=3., K_change=5.))
        self.assertEqual(decomposition["interaction"], 3.)

    def test_actual_pair_output_and_cross_fixed_K_calls(self):
        with tempfile.TemporaryDirectory(prefix="continuation_fixture_") as td:
            root = Path(td)
            counts = driver.base.new_counts()
            with patch.object(driver, "ROOT", root), patch.object(driver, "OUT", root):
                summaries, evaluation = driver.evaluate_pair(self.theta.copy(), (9, "inf"), self.q.copy(), root, "fixture", counts)
                self.assertEqual([q["K"] for q in evaluation["qualifications"]], [9, "inf"])
                self.assertEqual(counts["successful_kernel_calls"], 2)
                self.assertEqual(counts["target_K_blocks"], 4)
                for files in evaluation["rule_artifacts"].values():
                    for item in files.values(): self.assertEqual(driver.sha(root / item["path"]), item["sha256"])
                with np.load(root / "fixture_D.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["normal_quadrature_mass"].shape, ())
                    np.testing.assert_array_equal(arrays["K_names"], ["9", "inf"])
                own = {rule: driver.find_block(s, 9) for rule, s in summaries.items()}
                lower = point(9, "positive", name="low", theta=self.theta)
                upper = point(10, "negative", name="high", theta=self.theta + .01)
                lower.update(own_blocks=copy.deepcopy(own), own_qualification=dict(qualified=True))
                upper.update(own_blocks=copy.deepcopy(own), own_qualification=dict(qualified=True))
                calls = []
                def fake_pair(theta, Ks, q, folder, stem, counts):
                    calls.append((theta.copy(), Ks, stem))
                    answer = copy.deepcopy(summaries)
                    for summary in answer.values():
                        summary["blocks"] = [copy.deepcopy(summary["blocks"][0])]
                        summary["blocks"][0]["K"] = Ks[0]
                    counts["successful_kernel_calls"] += 2; counts["full_Hessian_calls"] += 2; counts["target_K_blocks"] += 2
                    return answer, dict(qualifications=[dict(qualified=False, K=Ks[0])], rule_artifacts={})
                with patch.object(driver, "evaluate_pair", side_effect=fake_pair):
                    result = driver.final_cross(dict(lower=lower, upper=upper), self.q, "manufactured_lock_hash")
                self.assertEqual([c[1] for c in calls], [(10,), (9,)])
                np.testing.assert_array_equal(calls[0][0], lower["theta"])
                np.testing.assert_array_equal(calls[1][0], upper["theta"])
                self.assertEqual(result["status"], "completed")
                self.assertFalse(result["all_four_cells_numerically_qualified"])
                self.assertEqual(result["resources"]["successful_kernel_calls"], 4)
                self.assertEqual(result["new_target_blocks"], 2)
                self.assertEqual(result["noncollinear_function_evaluations"], 0)


def run_tests():
    before = np.random.get_state(); tick = time.perf_counter(); log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContinuationTests))
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    return dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
        seconds=time.perf_counter() - tick, manufactured_only=True, random_state_unchanged=unchanged,
        old_study_arrays_read=False, study_teacher_or_fitted_theta_used=False,
        source_sha256=driver.sha(driver.__file__), test_sha256=driver.sha(__file__),
        base_source_sha256=driver.sha(driver.base.__file__), kernel_sha256=driver.sha(driver.base.kernel.__file__),
        diagnostics=DIAGNOSTICS, test_log=log.getvalue())


if __name__ == "__main__":
    report = run_tests(); print(json.dumps(report, sort_keys=True), flush=True)
    raise SystemExit(0 if report["passed"] else 1)
