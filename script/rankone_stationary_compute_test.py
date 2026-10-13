"""Manufactured-only qualification of solver, projections and delivery paths."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import rankone_stationary_compute as driver
import numpy as np

DIAGNOSTICS = {}


def quadratic_evaluator(center, H, *, reject_trials=False):
    center, H = np.asarray(center).copy(), np.asarray(H).copy()
    # Reused buffers deliberately test the driver's snapshot independence.
    gradient = np.empty((1, 1, len(center)))
    Hessian = np.empty((1, 1, len(center), len(center)))
    loss = np.empty((1, 1))
    def evaluate(theta, phase, hessian, context):
        delta = theta - center
        loss[0, 0] = .5 * delta @ driver.sym(H) @ delta
        if reject_trials and context["role"] == "trial": loss[0, 0] += 100.
        gradient[0, 0] = driver.sym(H) @ delta
        Hessian[0, 0] = H
        return dict(loss=loss, gradient=gradient, H_ba=Hessian if hessian else None)
    return evaluate


class StationaryDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theta = np.array([.15, -.1, .45, .35])
        cls.target = np.array([.1, .2, .3, .4]) * (1. - 5e-16)
        tick = time.perf_counter()
        cls.C = driver.project_result(driver.kernel.rankone_stationary(cls.theta[:2], cls.theta[2:],
            {"population": cls.target}, K_values=driver.KS, normal_rule="hermite", normal_order=40,
            logu_min=-28., logu_max=36., logu_step=.125))
        cls.D = driver.project_result(driver.kernel.rankone_stationary(cls.theta[:2], cls.theta[2:],
            {"population": cls.target}, K_values=driver.KS, normal_rule="legendre", normal_order=96, normal_limit=10.,
            logu_min=-28., logu_max=36., logu_step=.125))
        DIAGNOSTICS["two_manufactured_m2_kernel_calls_seconds"] = time.perf_counter() - tick

    def test_Householder_and_fixed_spectra(self):
        for a in (np.array([.3, .4]), np.array([-.3, .4, -.5]), np.array([0., 0., .5]), np.linspace(.1, .8, 8)):
            V = driver.transverse_basis(a)
            np.testing.assert_allclose(V.T @ V, np.eye(len(a) - 1), atol=5e-16, rtol=0)
            np.testing.assert_allclose(V.T @ a, 0., atol=4e-16, rtol=0)
            np.testing.assert_array_equal(V, driver.transverse_basis(a.copy()))
        for result in (self.C, self.D):
            V = result["transverse_basis"]
            for ki in range(3):
                H = result["Hperp"][0, ki]
                np.testing.assert_allclose(result["Hperp_projected"][0, ki], V.T @ H @ V, atol=2e-17, rtol=0)
                for name in ("H_ba", "Hperp_projected", "Hangle_projected"):
                    vals, vec = result[name + "_eigenvalues"][0, ki], result[name + "_eigenvectors"][0, ki]
                    np.testing.assert_allclose(vec @ np.diag(vals) @ vec.T, driver.sym(result[name][0, ki]), atol=5e-16, rtol=0)

    def test_quadratic_solver_and_independent_snapshots(self):
        initial = np.array([.4, -.3, .8, .6])
        center = np.array([.1, -.1, .5, .5])
        H = np.diag([2., 3., 1., 4.])
        original = initial.copy()
        trace = driver.stationary_solve(initial, quadratic_evaluator(center, H))
        np.testing.assert_allclose(trace["final_theta"], center, atol=2e-16, rtol=0)
        np.testing.assert_array_equal(initial, original)
        self.assertEqual(trace["stages"][0]["stop_reason"], "gradient_tolerance_at_accepted_FG")
        self.assertEqual(trace["stages"][1]["stop_reason"], "gradient_tolerance")
        self.assertEqual(trace["stages"][1]["call_count"], 1)
        for call in trace["calls"]:
            delta = call["theta"] - center
            np.testing.assert_allclose(call["gradient"], H @ delta, atol=2e-16, rtol=0)
            self.assertEqual(call["loss"], .5 * delta @ H @ delta)
        for left, right in zip(trace["accepted_states"], trace["accepted_states"][1:]):
            self.assertFalse(np.shares_memory(left["theta"], right["theta"]))
        for step in trace["steps"]:
            self.assertFalse(np.shares_memory(step["raw_H"], step["sym_H"]))
            self.assertLessEqual(float(np.max(abs(step["capped_direction"]))), .25)
            for trial in step["trials"]:
                np.testing.assert_array_equal(trial["theta"], step["theta"] + trial["alpha"] * step["capped_direction"])
        tape = driver.tape_arrays(trace)
        self.assertTrue(np.isnan(tape["call_H_ba"][~tape["compute_hessian"]]).all())
        self.assertTrue(np.isfinite(tape["call_H_ba"][tape["compute_hessian"]]).all())
        DIAGNOSTICS["quadratic_calls"] = trace["total_calls"]
        DIAGNOSTICS["quadratic_accepted_steps"] = trace["accepted_step_count"]

    def test_exact_budgets_and_C_starts_at_LAST(self):
        initial = np.array([.4, -.3, .8, .6])
        center = np.array([.1, -.1, .5, .5])
        stages = (dict(name="O", max_accepted_steps=12, max_calls=2, gradient_tolerance=1e-9),
                  dict(name="C", max_accepted_steps=4, max_calls=3, gradient_tolerance=1e-10))
        trace = driver.stationary_solve(initial, quadratic_evaluator(center, np.eye(4), reject_trials=True), stages)
        self.assertEqual([s["call_count"] for s in trace["stages"]], [2, 3])
        self.assertEqual([s["stop_reason"] for s in trace["stages"]], ["function_call_budget"] * 2)
        self.assertEqual(trace["total_calls"], 5)
        np.testing.assert_array_equal(trace["stages"][1]["start_theta"], trace["stages"][0]["final_theta"])
        np.testing.assert_array_equal(trace["final_theta"], initial)
        # Step budget is counted in accepts, not objective calls or trials.
        tiny_stage = (dict(name="O", max_accepted_steps=1, max_calls=80, gradient_tolerance=1e-20),)
        trace = driver.stationary_solve(initial, quadratic_evaluator(np.array([-1., -1., 1.8, 1.8]), np.eye(4)), tiny_stage)
        self.assertEqual(trace["accepted_step_count"], 1)
        self.assertEqual(trace["total_calls"], 2)
        self.assertEqual(trace["stages"][0]["stop_reason"], "accepted_step_budget")

    def test_thirteen_trials_and_failure_snapshot(self):
        initial = np.array([.4, -.3, .8, .6])
        center = np.array([.1, -.1, .5, .5])
        trace = driver.stationary_solve(initial, quadratic_evaluator(center, np.eye(4), reject_trials=True))
        self.assertEqual([s["call_count"] for s in trace["stages"]], [14, 14])
        self.assertEqual([s["stop_reason"] for s in trace["stages"]], ["line_search_failed"] * 2)
        for step in trace["steps"]:
            self.assertEqual(len(step["trials"]), 13)
            self.assertEqual(step["trials"][-1]["halvings"], 12)
            self.assertEqual(step["trials"][-1]["alpha"], step["initial_alpha"] / 4096.)
        old = quadratic_evaluator(center, np.eye(4))
        def failing(theta, phase, hessian, context):
            if context["role"] == "trial": raise FloatingPointError("manufactured failure")
            return old(theta, phase, hessian, context)
        partial = {}
        with self.assertRaisesRegex(FloatingPointError, "manufactured failure"):
            driver.stationary_solve(initial, failing, trace=partial)
        self.assertEqual(partial["attempted_calls"][-1]["status"], "failed")
        self.assertEqual(partial["attempted_calls"][-1]["role"], "trial")
        self.assertEqual(len(partial["calls"]), 1)
        self.assertFalse(np.shares_memory(partial["attempted_calls"][-1]["theta"], initial))

    def test_damping_symmetry_cap_and_box_fraction(self):
        initial = np.array([1.99, 0., .5, .6])
        H = np.diag([-2., 1., 1., 1.]); H[0, 1] = 1e-10; H[1, 0] = -1e-10
        g = np.array([-1., 0., 0., 0.])
        def manufactured(theta, phase, hessian, context):
            return dict(loss=np.array([[0. if context["role"] == "current" else -1.]]),
                        gradient=g[None, None].copy(), H_ba=H[None, None].copy() if hessian else None)
        stage = (dict(name="O", max_accepted_steps=1, max_calls=80, gradient_tolerance=1e-9),)
        trace = driver.stationary_solve(initial, manufactured, stage)
        step = trace["steps"][0]
        self.assertAlmostEqual(step["mu"], 2.0001)
        self.assertEqual(step["skew_max"], 2e-10)
        self.assertEqual(step["capped_direction"][0], .25)
        self.assertAlmostEqual(step["initial_alpha"], .99 * .01 / .25, places=14)
        self.assertAlmostEqual(trace["final_theta"][0], 1.9999, places=14)
        np.testing.assert_array_equal(step["raw_H"], H)
        np.testing.assert_array_equal(step["sym_H"], np.diag([-2., 1., 1., 1.]))

    def test_qualification_and_raw_mass_accounting(self):
        for ki in range(3): self.assertTrue(driver.qualification(self.C, self.D, 0, ki)["qualified"])
        DIAGNOSTICS["C_D_loss_max_difference"] = float(np.max(abs(self.C["loss"] - self.D["loss"])))
        DIAGNOSTICS["C_D_H_ba_max_difference"] = float(np.max(abs(self.C["H_ba"] - self.D["H_ba"])))
        corruptions = (("loss", (0, 0), 2e-7, "loss"), ("gradient", (0, 0, 0), 3e-7, "gradient"),
            ("H_ba", (0, 0, 0, 0), 3e-7, "H_ba"), ("Hperp", (0, 0, 0, 0), 3e-7, "Hperp"),
            ("Hangle", (0, 0, 0, 0), 3e-7, "Hangle"),
            ("H_ba_eigenvalues", (0, 0, 0), 3e-6, "H_ba_eigenvalues"),
            ("ward_residual_max", (0, 0), 3e-7, "C_Ward"))
        for key, index, amount, gate in corruptions:
            C = copy.deepcopy(self.C); C[key][index] += amount
            q = driver.qualification(C, self.D, 0, 0)
            self.assertFalse(q["qualified"]); self.assertIn(gate, q["failed_gates"])
        C = copy.deepcopy(self.C)
        C["normal_quadrature_mass"] = .9
        C["raw_bank_quadrature_mass"][0] = .9 ** 8
        C["label_d_mass"][0] = .9 ** 8
        C["label_c_mass"][0] = .8 * .9 ** 8
        C["raw_c_mass_upper"][0] = .875 * .9 ** 8
        q = driver.qualification(C, self.D, 0, 0)
        self.assertFalse(q["gates"]["C_normal_mass"])
        self.assertTrue(q["gates"]["C_d_relative_mass"])
        self.assertTrue(q["gates"]["C_c_upper"])
        self.assertLess(driver.scalar_diagnostics(C, 0)["d_relative_mass_error"], 1e-15)
        self.assertNotEqual(float(self.D["config"]["target_masses"][0]), 1.)

    def test_fit_signature_and_probe_fixed_direction(self):
        result = copy.deepcopy(self.D)
        result["gradient"][0, 0] = 0.
        result["H_ba"][0, 0] = np.eye(4)
        V = driver.transverse_basis(self.theta[2:])
        result["Hperp"][0, 0] = V @ V.T
        driver.project_result(result)
        q = dict(qualified=True)
        signature = driver.classification(self.theta, result, 0, 0, q)
        self.assertTrue(signature["qualified_full_local_minimum_signature"])
        self.assertEqual(signature["nonrotation_dimensions"], 5)
        result["gradient"][0, 2] = np.array([1., 2., 3., 4.])
        plan = driver.probe_plan(self.theta, result, signature)
        np.testing.assert_allclose(plan["direction"], -np.arange(1., 5.) / np.sqrt(30.), atol=2e-16, rtol=0)
        self.assertEqual([p["t"] for p in plan["points"]], list(driver.PROBE_T))
        result["gradient"][0, 0, 0] = 2e-8
        failed = driver.classification(self.theta, result, 0, 0, q)
        self.assertIn("gradient", failed["failed_gates"])
        self.assertEqual(driver.probe_plan(self.theta, result, failed)["status"], "skipped_P_fit_signature_not_qualified")
        result["gradient"][0, 2] = 0.
        self.assertEqual(driver.probe_plan(self.theta, result, signature)["status"], "undefined_integrated_gradient_too_small")
        result["gradient"][0, 2] = np.array([-1., 0., 0., 0.])
        at_edge = self.theta.copy(); at_edge[0] = 1.999
        plan = driver.probe_plan(at_edge, result, signature)
        self.assertFalse(plan["points"][-1]["box"]["feasible"])
        self.assertGreater(plan["points"][-1]["theta"][0], 2.)

    def test_kernel_public_arrays_and_raw_KL(self):
        report = driver.evaluation_summary(self.theta, self.D, self.target, .01)
        self.assertEqual(len(report["blocks"]), 3)
        p = np.exp(self.D["log_evidence"])
        self.assertEqual(report["probability"]["raw_KL"], float(self.target @ (np.log(self.target) - self.D["log_evidence"])))
        self.assertEqual(report["probability"]["pmf_mass"], float(p.sum()))
        json.dumps(report, allow_nan=False)
        counts = driver.new_counts(); driver.add_counts(counts, self.D, True)
        self.assertEqual(counts["successful_kernel_calls"], 1)
        self.assertEqual(counts["target_K_blocks"], 3)
        with tempfile.TemporaryDirectory(prefix="stationary_fixture_") as td:
            path = Path(td) / "CALL.npz"
            arrays = driver.kernel_arrays(self.D, self.theta)
            driver.save_npz(path, arrays)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertEqual(loaded["normal_quadrature_mass"].shape, ())
                for key, value in arrays.items(): np.testing.assert_array_equal(loaded[key], value)
            with self.assertRaises(FileExistsError): driver.save_npz(path, arrays)

    def test_paired_delivery_and_probe_without_study_inputs(self):
        def fixture_call(theta, targets, Ks, rule, hessian, counts):
            result = copy.deepcopy(self.C if rule == "C" else self.D)
            driver.add_counts(counts, result, True)
            return result, .001
        counts = driver.new_counts()
        with tempfile.TemporaryDirectory(prefix="stationary_delivery_fixture_") as td:
            folder = Path(td)
            with patch.object(driver, "ROOT", folder), patch.object(driver, "kernel_call", side_effect=fixture_call):
                results, summary = driver.paired_evaluation(self.theta, {"population": self.target}, self.target, folder, "fixture", counts)
                self.assertEqual(len(summary["qualifications"]), 3)
                self.assertTrue(all(q["qualified"] for q in summary["qualifications"]))
                for files in summary["rule_artifacts"].values():
                    for item in files.values(): self.assertEqual(driver.sha(folder / item["path"]), item["sha256"])
                probe_folder = folder / "probe"; probe_folder.mkdir()
                forced = dict(qualified_full_local_minimum_signature=True)
                probe = driver.run_probe(self.theta, self.D, forced, self.target, probe_folder, counts)
                self.assertEqual(probe["status"], "completed_prespecified_probe")
                self.assertEqual(len(probe["points"]), 5)
                self.assertTrue(all(p["qualified"] for p in probe["points"]))
                self.assertEqual(counts["successful_kernel_calls"], 12)
                self.assertEqual(counts["target_K_blocks"], 36)
                self.assertEqual(counts["new_random_draws"], 0)

    def test_all_five_endpoint_lock_after_manufactured_solves(self):
        center = np.array([.1, -.1, .5, .5])
        initial = np.array([.4, -.3, .8, .6])
        source = dict(teacher_pmf=self.target.tolist(), solves=[dict(name=name, theta0=initial.tolist(),
            target_name="population" if name == "P" else "empirical", target=self.target.tolist()) for name in driver.SOLVE_NAMES])
        def fake_call(theta, targets, Ks, rule, hessian, counts):
            r = copy.deepcopy(self.C)
            d = theta - center
            r["loss"] = np.array([[.5 * d @ d]])
            r["gradient"] = d[None, None].copy()
            r["H_ba"] = np.eye(4)[None, None].copy() if hessian else None
            r["target_names"] = tuple(targets)
            r["K_values"] = (8,)
            driver.add_counts(counts, r, hessian)
            return r, .001
        with tempfile.TemporaryDirectory(prefix="stationary_five_lock_fixture_") as td:
            base = Path(td); out = base / "artifacts"; out.mkdir()
            src, freeze, lock = out / "SOURCE.json", out / "FREEZE.json", out / "ENDPOINTS_LOCK.json"
            driver.save(src, source); driver.save(freeze, dict(manufactured=True))
            with patch.object(driver, "ROOT", base), patch.object(driver, "OUT", out), \
                    patch.object(driver, "SOURCE", src), patch.object(driver, "FREEZE", freeze), \
                    patch.object(driver, "ENDPOINTS", lock), patch.object(driver, "verify_freeze", return_value=({}, source)), \
                    patch.object(driver, "kernel_call", side_effect=fake_call):
                with self.assertRaises(FileNotFoundError): driver.load_endpoint_lock(source)
                with redirect_stdout(io.StringIO()): driver.optimize()
                endpoint_lock = driver.load_endpoint_lock(source)
                self.assertEqual([e["name"] for e in endpoint_lock["endpoints"]], list(driver.SOLVE_NAMES))
                self.assertTrue(endpoint_lock["all_five_locked_before_control_evaluations"])
                for endpoint in endpoint_lock["endpoints"]:
                    np.testing.assert_allclose(endpoint["theta"], center, atol=2e-16, rtol=0)
                    solver = driver.read(base / endpoint["solver_artifact"]["path"])
                    self.assertEqual(solver["status"], "completed")
                    self.assertEqual(len(solver["call_artifacts"]), solver["total_calls"])
                with self.assertRaises(FileExistsError): driver.optimize()
                changed = base / endpoint_lock["endpoints"][0]["solver_artifact"]["path"]
                changed.write_text(changed.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed saved optimization artifact"):
                    driver.load_endpoint_lock(source)


def run_tests():
    before = np.random.get_state(); tick = time.perf_counter(); log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StationaryDriverTests)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    return dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
        seconds=time.perf_counter() - tick, manufactured_only=True, random_state_unchanged=unchanged,
        study_teachers_or_old_endpoints_used=False, diagnostics=DIAGNOSTICS, test_log=log.getvalue(),
        source_sha256=driver.sha(driver.__file__), kernel_sha256=driver.sha(driver.kernel.__file__), test_sha256=driver.sha(__file__))


if __name__ == "__main__":
    report = run_tests(); print(json.dumps(report, sort_keys=True), flush=True)
    raise SystemExit(0 if report["passed"] else 1)
