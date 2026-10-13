"""Manufactured cache, fixed-panel, scalar-gate and outcome tests."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import noncollinear_probe_compute as driver
import numpy as np

DIAGNOSTICS = {}
SETTINGS = dict(normal_specs=[dict(rule="hermite", order=16), dict(rule="hermite", order=12)],
                logu_min=-24., logu_max=32., logu_step=.25)


def fixture_anchor(K=11):
    return dict(name="fixture_K" + str(K), K=K, theta=[.1, -.15, .35, .4],
                curvature=dict(C=.01, D=.011), risk_curvature=dict(C=-.02, D=-.021))


def old_reference(prepared):
    return dict(blocks=[dict(target="population", K=K, loss=float(prepared["absolute_loss"][0, ki])) for ki, K in enumerate(prepared["K_values"])],
                probability=dict(pmf=prepared["pmf"].tolist()))


def manufactured_scalar(t, F=2e-7, R=-3e-7):
    return dict(t=t, paired_F=F, paired_R=R, paired_infinite=R,
        absolute_loss=[5. + F, 4. + R], raw_KL=.1 + R,
        pmf=[.2, .25, .25, .3], normal_mass=1., model_mass=1.,
        all_outputs_finite=True, PMF_nonnegative=True,
        geometry=dict(bias_exactly_equal=True, row_norm_max_error=0.),
        marginal_max_difference=0., delta_consistency_error=[0., 0.], risk_consistency_error=0., infinite_risk_consistency_error=0.)


def scalar_gate_inputs(t=.02, F=2e-7, R=-3e-7):
    rows = {rule: manufactured_scalar(t, F, R) for rule in driver.RULES}
    baselines = {rule: dict(qualification=dict(qualified=True)) for rule in driver.RULES}
    even = {rule: driver.evenness(rows[rule], rows[rule]) for rule in driver.RULES}
    return rows, baselines, even


class ProbeDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = np.array([1., -1.]) / np.sqrt(2.)
        cls.q = np.array([.2, .23, .27, .3]) * (1. - 5e-16)
        anchor = fixture_anchor(); theta = np.asarray(anchor["theta"])
        cls.prepared = driver.kernel.prepare_baseline(theta[:2], driver.kernel.path_W(theta[2:], cls.d, 0.),
            {"population": cls.q.copy()}, K_values=(11, "inf"), **SETTINGS)

    def test_geometry_frozen_bias_row_norms_and_curvature_readouts(self):
        a = fixture_anchor(); theta = np.asarray(a["theta"])
        for t in (-.1, -.05, -.02, .02, .05, .1):
            W = driver.kernel.path_W(theta[2:], self.d, t)
            result = driver.kernel.evaluate_prepared(self.prepared, W)
            row = driver.point_summary(result, a, self.d, t)
            self.assertTrue(row["geometry"]["bias_exactly_equal"])
            self.assertLess(row["geometry"]["row_norm_max_error"], 2e-16)
            self.assertEqual(row["geometry"]["path_W_max_error"], 0.)
            self.assertEqual(row["quadratic_F"], 2. * row["paired_F"] / t ** 2)
            self.assertEqual(row["quadratic_R_minus_old"]["D"], row["quadratic_R"] - a["risk_curvature"]["D"])
            np.testing.assert_array_equal(result["targets"][0], self.q)
        minus = driver.kernel.path_W(theta[2:], self.d, -.1)
        plus = driver.kernel.path_W(theta[2:], self.d, .1)
        np.testing.assert_array_equal(minus[:, 0], plus[:, 0]); np.testing.assert_array_equal(minus[:, 1], -plus[:, 1])

    def test_logzero_exception_is_narrow_and_fingerprint_detects_mutation(self):
        value = dict(logPsi=np.array([0., -np.inf]), paired_integrand=np.array([0., 1.]))
        self.assertTrue(driver.all_finite(value))
        for x in (np.nan, np.inf):
            damaged = copy.deepcopy(value); damaged["logPsi"][1] = x
            self.assertFalse(driver.all_finite(damaged))
        damaged = copy.deepcopy(value); damaged["paired_integrand"][0] = -np.inf
        self.assertFalse(driver.all_finite(damaged))
        digest = driver.fingerprint(self.prepared)
        changed = copy.deepcopy(self.prepared); changed["W0"][0, 0] += 1e-10
        self.assertNotEqual(digest, driver.fingerprint(changed))
        self.assertEqual(digest, driver.fingerprint(self.prepared))

    def test_shared_baseline_and_exact_6_plus_36_panel(self):
        counts = driver.new_counts(); prepares, evaluations = [], []
        old = old_reference(self.prepared)
        def prep(*args, **kwargs):
            result = driver.kernel.prepare_baseline(*args, **kwargs)
            prepares.append(id(result)); return result
        def evaluate(prepared, W):
            evaluations.append((id(prepared), W.copy()))
            return driver.kernel.evaluate_prepared(prepared, W)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="probe_panel_fixture_") as td:
            root = Path(td)
            with patch.object(driver, "ROOT", root), redirect_stdout(io.StringIO()):
                for K in (11, 12):
                    anchor = fixture_anchor(K)
                    # The second manufactured reference has the same fixed rule.
                    p = driver.kernel.prepare_baseline(np.asarray(anchor["theta"])[:2], driver.kernel.path_W(np.asarray(anchor["theta"])[2:], self.d, 0.),
                        {"population": self.q}, K_values=(K, "inf"), **SETTINGS)
                    for rule in driver.RULES:
                        folder = root / anchor["name"] / rule; folder.mkdir(parents=True)
                        rows = driver.execute_rule(anchor, self.q, self.d, rule, folder, counts,
                            settings=SETTINGS, prepare=prep, evaluate=evaluate, old_D=old_reference(p))
                        self.assertEqual(list(rows), [name for name, _ in driver.POINTS])
                        self.assertEqual(rows["zero"]["paired_F"], 0.)
                        self.assertIsNone(rows["zero"]["quadratic_F"])
                        self.assertTrue(rows["zero"]["qualification"]["qualified"])
                        receipt = driver.read(folder / "RULE_COMPLETE.json")
                        self.assertTrue(receipt["prepared_unchanged"])
                        self.assertEqual(receipt["independently_evaluated_nonzero_points"], 6)
                        with np.load(folder / "BASELINE.npz", allow_pickle=False) as data:
                            self.assertIn("conditional_logA", data.files)
                            self.assertEqual(data["normal_quadrature_mass"].shape, ())
                        with np.load(folder / "m002.npz", allow_pickle=False) as data:
                            self.assertEqual(data["logPsi"].shape[0], 2)
                            self.assertIn("paired_integrand", data.files)
                            self.assertNotIn("conditional_logA", data.files)
                self.assertEqual(counts["baseline_preparations"], 6)
                self.assertEqual(counts["nonzero_evaluations"], 36)
                self.assertEqual(counts["reused_zero_records"], 6)
                self.assertEqual(len(prepares), 6); self.assertEqual(len(evaluations), 36)
                for group in range(6):
                    self.assertTrue(all(entry[0] == prepares[group] for entry in evaluations[group * 6:(group + 1) * 6]))
                    Wminus, Wplus = evaluations[group * 6][1], evaluations[group * 6 + 5][1]
                    np.testing.assert_array_equal(Wminus[:, 0], Wplus[:, 0])
                    np.testing.assert_array_equal(Wminus[:, 1], -Wplus[:, 1])
        DIAGNOSTICS["manufactured_complete_6_plus_36_seconds"] = time.perf_counter() - started
        DIAGNOSTICS["manufactured_high_level_factors"] = counts["high_level_factor_totals"]

    def test_prepared_mutation_fails_without_retry(self):
        calls = []
        def bad(prepared, W):
            result = driver.kernel.evaluate_prepared(prepared, W)
            calls.append(W.copy())
            if len(calls) == 6: prepared["W0"][0, 0] += 1e-8
            return result
        with tempfile.TemporaryDirectory(prefix="probe_mutation_fixture_") as td:
            folder = Path(td)
            with patch.object(driver, "ROOT", folder), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "prepared baseline mutated"):
                    driver.execute_rule(fixture_anchor(), self.q, self.d, "A", folder, driver.new_counts(),
                        settings=SETTINGS, evaluate=bad, old_D=old_reference(self.prepared))
                self.assertEqual(len(calls), 6)
                self.assertTrue((folder / "FAILURE.json").exists())
                self.assertFalse((folder / "RULE_COMPLETE.json").exists())

    def test_every_shared_gate_and_separate_effect_qualification(self):
        rows, baselines, even = scalar_gate_inputs()
        clean = driver.qualify_point(rows, baselines, even)
        self.assertTrue(clean["qualified"])
        self.assertEqual(clean["effects"]["F"]["sign"]["sign"], "positive")
        self.assertEqual(clean["effects"]["R"]["sign"]["sign"], "negative")
        corruptions = [("all_outputs_finite", False, "A_finite"), ("PMF_nonnegative", False, "A_PMF_nonnegative"),
            ("normal_mass", 1. + 3e-12, "A_normal_mass"), ("model_mass", 1. + 3e-12, "A_model_mass"),
            ("marginal_max_difference", 3e-10, "A_marginal"), ("risk_consistency_error", 6e-12, "A_risk_interface"),
            ("infinite_risk_consistency_error", 6e-12, "A_infinite_risk_interface"),
            ("delta_consistency_error", [6e-12, 0.], "A_delta_interface")]
        for key, value, gate in corruptions:
            modified = copy.deepcopy(rows); modified["A"][key] = value
            result = driver.qualify_point(modified, baselines, even)
            self.assertFalse(result["qualified"]); self.assertIn(gate, result["shared_failed_gates"])
        modified = copy.deepcopy(rows); modified["B"]["paired_F"] += 2e-10
        result = driver.qualify_point(modified, baselines, even)
        self.assertFalse(result["qualified_F"]); self.assertTrue(result["qualified_R"]); self.assertFalse(result["qualified"])
        modified_even = copy.deepcopy(even); modified_even["C"]["gates"]["R"] = False
        result = driver.qualify_point(rows, baselines, modified_even)
        self.assertTrue(result["qualified_F"]); self.assertFalse(result["qualified_R"])

    def test_absolute_and_relative_precision_and_sign_resolution(self):
        # Both absolute and relative conditions are required independently.
        self.assertFalse(driver.paired_precision(1e-12, 2e-12, "F", "BC")["gates"]["relative"])
        self.assertFalse(driver.paired_precision(1e-3, 1e-3 + 2e-10, "F", "BC")["gates"]["absolute"])
        self.assertEqual(driver.sign_readout(1e-9, 1e-9)["sign"], "unresolved")
        self.assertEqual(driver.sign_readout(2e-9, -2e-9)["sign"], "unresolved")
        self.assertEqual(driver.sign_readout(2e-9, 2e-9)["sign"], "positive")
        self.assertEqual(driver.sign_readout(-2e-9, -2e-9)["sign"], "negative")
        result = driver.sign_readout(2e-8, 2.2e-8)
        self.assertFalse(result["resolved"])

    def test_evenness_all_three_components(self):
        a, b = manufactured_scalar(-.02), manufactured_scalar(.02)
        self.assertTrue(driver.evenness(a, b)["qualified"])
        b["paired_F"] += 2e-11; b["paired_R"] += 2e-11; b["pmf"][0] += 3e-12
        result = driver.evenness(a, b)
        self.assertEqual(set(result["failed_gates"]), {"F", "R", "PMF"})

    def test_primary_cannot_be_replaced_by_sensitivity(self):
        anchors = []
        for K in (11, 12):
            points = []
            for point_id, t in driver.POINTS[1:]:
                rows, baselines, even = scalar_gate_inputs(t, F=2e-7 if K == 11 else -2e-7)
                points.append(dict(point_id=point_id, t=t, qualification=driver.qualify_point(rows, baselines, even)))
            anchors.append(dict(name="K" + str(K), K=K, points=points))
        self.assertEqual(driver.outcome_summary(anchors)["primary"]["status"], "supported")
        anchors[0]["points"][2]["qualification"]["qualified"] = False
        result = driver.outcome_summary(anchors)
        self.assertEqual(result["primary"]["status"], "unresolved_or_unqualified")
        self.assertTrue(all(s["supported"] for s in result["sensitivities"]))
        self.assertTrue(result["larger_amplitude_does_not_replace_primary"])
        anchors[0]["points"][2]["qualification"]["qualified"] = True
        anchors[0]["points"][2]["qualification"]["effects"]["F"]["sign"]["sign"] = "negative"
        self.assertEqual(driver.outcome_summary(anchors)["primary"]["status"], "resolved_different_combination")

    def test_parallel_and_serial_scientific_outputs_and_counts_equal(self):
        anchors = [fixture_anchor(11), fixture_anchor(12)]
        references = {}
        for anchor in anchors:
            theta = np.asarray(anchor["theta"])
            p = driver.kernel.prepare_baseline(theta[:2], driver.kernel.path_W(theta[2:], self.d, 0.),
                {"population": self.q}, K_values=(anchor["K"], "inf"), **SETTINGS)
            references[anchor["name"]] = old_reference(p)
        def rule_runner(anchor, q, d, rule, folder, counts):
            return driver.execute_rule(anchor, q, d, rule, folder, counts,
                settings=SETTINGS, old_D=references[anchor["name"]])
        def scientific(value):
            if isinstance(value, dict):
                return {key: scientific(item) for key, item in value.items()
                        if key != "seconds" and not key.endswith("_artifact")}
            if isinstance(value, list): return [scientific(item) for item in value]
            return value
        with tempfile.TemporaryDirectory(prefix="probe_parallel_fixture_") as td:
            root = Path(td)
            with patch.object(driver, "ROOT", root), redirect_stdout(io.StringIO()):
                serial = [driver.execute_anchor(a, self.q, self.d, root / "serial" / a["name"],
                    verify=lambda: None, rule_runner=rule_runner) for a in anchors]
                identities, identities_lock = [], threading.Lock()
                barrier = threading.Barrier(2)
                def worker(anchor):
                    with identities_lock: identities.append(threading.get_ident())
                    barrier.wait(timeout=10)
                    return driver.execute_anchor(anchor, self.q, self.d, root / "parallel" / anchor["name"],
                        verify=lambda: None, rule_runner=rule_runner)
                parallel = driver.parallel_anchors(anchors, worker)
                self.assertEqual(len(set(identities)), 2)
                self.assertEqual([r["anchor_name"] for r in parallel], [a["name"] for a in anchors])
                self.assertTrue(all(r["status"] == "completed" for r in parallel + serial))
                for sequential, concurrent in zip(serial, parallel):
                    self.assertEqual(scientific(sequential["summary"]), scientific(concurrent["summary"]))
                    self.assertEqual(sequential["resources"], concurrent["resources"])
                    self.assertIsNot(sequential["resources"], concurrent["resources"])
                before = copy.deepcopy([r["resources"] for r in parallel])
                total = driver.merge_counts([r["resources"] for r in parallel])
                self.assertEqual(total["baseline_preparations"], 6)
                self.assertEqual(total["nonzero_evaluations"], 36)
                self.assertEqual(total["reused_zero_records"], 6)
                self.assertEqual(before, [r["resources"] for r in parallel])

    def test_parallel_failure_waits_for_peer_and_overall_fails(self):
        anchors = [fixture_anchor(11), fixture_anchor(12)]
        source = dict(anchors=anchors, teacher_pmf=self.q.tolist(), direction=self.d.tolist())
        theta = np.asarray(anchors[1]["theta"])
        prepared = driver.kernel.prepare_baseline(theta[:2], driver.kernel.path_W(theta[2:], self.d, 0.),
            {"population": self.q}, K_values=(12, "inf"), **SETTINGS)
        reference = old_reference(prepared)
        original = driver.execute_anchor
        barrier = threading.Barrier(2)
        attempted = []; attempted_lock = threading.Lock()
        def runner(anchor, q, d, rule, folder, counts):
            with attempted_lock: attempted.append((anchor["name"], rule))
            if anchor["K"] == 11: raise RuntimeError("manufactured independent worker failure")
            return driver.execute_rule(anchor, q, d, rule, folder, counts, settings=SETTINGS, old_D=reference)
        def worker(anchor, q, d, folder):
            barrier.wait(timeout=10)
            return original(anchor, q, d, folder, verify=lambda: None, rule_runner=runner)
        with tempfile.TemporaryDirectory(prefix="probe_failure_fixture_") as td:
            root = Path(td); out = root / "artifacts"; out.mkdir()
            src, freeze = out / "SOURCE.json", out / "FREEZE.json"
            driver.save(src, source); driver.save(freeze, dict(manufactured=True))
            with patch.object(driver, "ROOT", root), patch.object(driver, "OUT", out), \
                    patch.object(driver, "SOURCE", src), patch.object(driver, "FREEZE", freeze), \
                    patch.object(driver, "verify_freeze", return_value=({}, source)), \
                    patch.object(driver, "execute_anchor", side_effect=worker), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "independent anchor workers failed"):
                    driver.run()
                self.assertFalse((out / "RESULTS.json").exists())
                self.assertTrue((out / "FAILURE.json").exists())
                self.assertTrue((out / "anchors" / anchors[0]["name"] / "FAILURE.json").exists())
                self.assertTrue((out / "anchors" / anchors[1]["name"] / "RESULTS.json").exists())
                workers = driver.read(out / "WORKERS.json")
                self.assertEqual([w["status"] for w in workers["outcomes"]], ["failed", "completed"])
                self.assertTrue(workers["all_started_workers_awaited"])
                self.assertFalse(workers["automatic_retry"])
                failure = driver.read(out / "FAILURE.json")
                self.assertEqual(failure["resources"]["baseline_preparations"], 3)
                self.assertEqual(failure["resources"]["nonzero_evaluations"], 18)
                self.assertEqual(sum(name == anchors[0]["name"] for name, _ in attempted), 1)
                self.assertEqual([rule for name, rule in attempted if name == anchors[1]["name"]], ["A", "B", "C"])


def run_tests():
    before = np.random.get_state(); tick = time.perf_counter(); log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProbeDriverTests))
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    return dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
        seconds=time.perf_counter() - tick, manufactured_only=True, random_state_unchanged=unchanged,
        study_teachers_or_anchors_used=False, old_scientific_arrays_read=False,
        source_sha256=driver.sha(driver.__file__), test_sha256=driver.sha(__file__), kernel_sha256=driver.sha(driver.kernel.__file__),
        diagnostics=DIAGNOSTICS, test_log=log.getvalue())


if __name__ == "__main__":
    report = run_tests(); print(json.dumps(report, sort_keys=True), flush=True)
    raise SystemExit(0 if report["passed"] else 1)
