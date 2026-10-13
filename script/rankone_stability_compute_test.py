"""Manufactured-only tests of the rank-one driver; no old data or RNG draws."""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import rankone_stability_compute as driver
import numpy as np

DIAGNOSTICS = {}


def candidate(name, b, W):
    return dict(name=name, b=b.tolist(), W=W.tolist(), theta=np.r_[b, W.ravel()].tolist())


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = np.array([.1, -.2, .25])
        cls.a = np.array([.3, .4, .5])
        empirical = np.arange(1., 9.) / 36.
        # Preserve a deliberately non-unit but qualified binary64 target mass.
        population = empirical[::-1].copy() * (1. - 5e-16)
        cls.targets = {"empirical": empirical, "population": population}
        tick = time.perf_counter()
        cls.C = driver.add_projection(driver.kernel.rankone_stability(cls.b, cls.a, cls.targets,
            K_values=driver.K_VALUES, normal_rule="hermite", normal_order=40,
            logu_min=-28., logu_max=36., logu_step=.125))
        cls.D = driver.add_projection(driver.kernel.rankone_stability(cls.b, cls.a, cls.targets,
            K_values=driver.K_VALUES, normal_rule="legendre", normal_order=96, normal_limit=10.,
            logu_min=-28., logu_max=36., logu_step=.125))
        DIAGNOSTICS["manufactured_two_kernel_calls_seconds"] = time.perf_counter() - tick

    def test_fixed_helmert_projection(self):
        for m in (2, 3, 8):
            Q = driver.helmert_basis(m)
            np.testing.assert_allclose(Q.T @ Q, np.eye(m - 1), atol=4e-16, rtol=0)
            np.testing.assert_allclose(Q.T @ np.ones(m), 0., atol=3e-16, rtol=0)
        Q = self.C["helmert_basis"]
        for ti in range(2):
            for ki in range(3):
                H = self.C["laplacian"][ti, ki]
                P = self.C["projected_hessian"][ti, ki]
                v, V = self.C["projected_eigenvalues"][ti, ki], self.C["projected_eigenvectors"][ti, ki]
                np.testing.assert_allclose(P, Q.T @ H @ Q, rtol=0, atol=1e-17)
                np.testing.assert_allclose(V @ np.diag(v) @ V.T, P, rtol=0, atol=2e-17)
                np.testing.assert_allclose(self.C["angle_eigenvectors"][ti, ki], Q @ V, rtol=0, atol=1e-16)
                np.testing.assert_allclose((Q @ V).T @ np.ones(3), 0., atol=4e-16, rtol=0)

    def test_anchors_use_only_fixed_biases_and_row_norms(self):
        b8 = np.linspace(-.3, .4, 8)
        bR = b8 + .15
        W8 = np.column_stack((np.linspace(.1, .2, 8), np.linspace(-.04, .03, 8)))
        WR = np.column_stack((np.linspace(.2, .3, 8), np.linspace(.06, -.01, 8)))
        sources = [candidate("K8", b8, W8), candidate("REF", bR, WR)]
        old = copy.deepcopy(sources)
        anchors = driver.construct_anchors(sources)
        self.assertEqual(sources, old)
        self.assertEqual([a["name"] for a in anchors],
            ["b8_s8", "b8_sREF", "bREF_s8", "bREF_sREF", "b8_lambda025", "b8_lambda050", "b8_lambda075"])
        s8, sR = np.linalg.norm(W8, axis=1), np.linalg.norm(WR, axis=1)
        for row in anchors:
            b = b8 if row["bias_source"] == "K8" else bR
            lam = row["lambda_value"]
            expected = (1. - lam) * s8 + lam * sR
            np.testing.assert_array_equal(row["b"], b)
            np.testing.assert_array_equal(row["a"], expected)
            np.testing.assert_array_equal(np.asarray(row["W"])[:, 0], row["a"])
            np.testing.assert_array_equal(np.asarray(row["W"])[:, 1], 0.)
            self.assertFalse(row["original_training_endpoint"])
        bad = copy.deepcopy(sources)
        zero = W8.copy(); zero[2] = 0.
        bad[0] = candidate("K8", b8, zero)
        with self.assertRaisesRegex(ValueError, "positive source row norms"):
            driver.construct_anchors(bad)

    def test_per_label_reweighting_and_independent_ward(self):
        q = np.asarray(list(self.targets.values()))
        maximum_reweight_difference = 0.
        for r in (self.C, self.D):
            M = np.einsum("tl,klij->tkij", q, r["label_M"])
            gb = np.einsum("tl,kli->tki", q, r["label_bias_gradient"])
            ga = np.einsum("tl,kli->tki", q, r["label_radial_gradient"])
            sp = np.einsum("tl,kli->tki", q, r["label_sigma_prime_average"])
            H = -M.copy()
            for i in range(3):
                H[..., i, i] += sp[..., i]
            L = np.zeros_like(H)
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = self.a[i] * self.a[j] * M[..., i, j]
                    L[..., i, j] = L[..., j, i] = -edge
                    L[..., i, i] += edge; L[..., j, j] += edge
            ward = H * self.a[None, None, :, None] * self.a[None, None, None, :]
            for i in range(3):
                ward[..., i, i] -= self.a[i] * ga[..., i]
            for name, value in (("laplacian", L), ("Hperp", H), ("ward_angle_hessian", ward),
                                ("bias_gradient", gb), ("radial_gradient", ga), ("ward_residual", ward - L)):
                maximum_reweight_difference = max(maximum_reweight_difference, float(np.max(abs(r[name] - value))))
                # Independent reduction associations may differ by a few ulps.
                np.testing.assert_allclose(r[name], value, atol=2 * np.finfo(float).eps, rtol=0)
            np.testing.assert_array_equal(r["config"]["target_masses"], q.sum(axis=1))
            np.testing.assert_allclose(r["target_d_mass"], np.einsum("tl,kl->tk", q, r["label_d_mass"]), atol=3e-16, rtol=0)
        DIAGNOSTICS["C_D_max_matrix_difference"] = float(np.max(abs(self.C["laplacian"] - self.D["laplacian"])))
        DIAGNOSTICS["maximum_independent_Ward_error"] = max(float(r["ward_residual_max"].max()) for r in (self.C, self.D))
        DIAGNOSTICS["maximum_independent_reweight_difference"] = maximum_reweight_difference
        self.assertNotEqual(float(q[1].sum()), 1.)

    def test_gate_failures_remain_failures(self):
        for ti in range(2):
            for ki in range(3):
                self.assertTrue(driver.qualify(self.C, self.D, ti, ki)["qualified"])
        # Each diagnostic is deliberately corrupted in isolation. The common
        # Laplacian zero mode remains intact in the Ward-only failure case.
        changes = [
            ("C_independent_Ward", "ward_residual_max", (0, 0), 3e-7),
            ("C_normal_mass_error", "normal_quadrature_mass", None, 1. + 3e-12),
            ("C_d_mass_max_error", "label_d_mass", (0, 7), 1. + 3e-7),
            ("C_c_mass_lower_violation", "label_c_mass", (0, 7), -3e-7),
            ("C_c_mass_upper_violation", "label_c_mass", (0, 7), .875 + 3e-7),
        ]
        for gate, field, index, value in changes:
            damaged = copy.deepcopy(self.C)
            if index is None:
                damaged[field] = value
            else:
                damaged[field][index] = value
            outcome = driver.qualify(damaged, self.D, 0, 0)
            self.assertFalse(outcome["qualified"])
            self.assertIn(gate, outcome["failed_gates"])
            self.assertEqual(outcome["classification"], "unresolved_numerical_qualification")
            np.testing.assert_array_equal(damaged["laplacian"], self.C["laplacian"])
        bad = copy.deepcopy(self.C)
        bad["laplacian"][0, 0, 0, 1] += 3e-7
        self.assertIn("matrix_difference", driver.qualify(bad, self.D, 0, 0)["failed_gates"])
        bad = copy.deepcopy(self.C)
        bad["projected_eigenvalues"][0, 0, 0] += 3e-6
        self.assertIn("projected_eigenvalue_difference", driver.qualify(bad, self.D, 0, 0)["failed_gates"])

    def test_fixed_threshold_and_all_coarse_discrepancies(self):
        e = driver.THRESHOLDS["sign_resolution"]
        for x in (-e, 0., e):
            self.assertEqual(driver.classify(x, True), "unresolved_at_fixed_sign_resolution")
        self.assertEqual(driver.classify(2 * e, True), "numerically_qualified_angular_local_minimum")
        self.assertEqual(driver.classify(-2 * e, True), "numerically_qualified_negative_angular_curvature")
        self.assertEqual(driver.classify(2 * e, False), "unresolved_numerical_qualification")
        rules = {"A": copy.deepcopy(self.C), "B": self.D, "C": self.C, "D": self.D}
        rules["A"]["laplacian"] += .1
        report = driver.anchor_summary(dict(name="manufactured"), rules)
        self.assertEqual(len(report["targets"]), 6)
        self.assertTrue(report["all_primary_matrices_qualified"])
        for row in report["targets"]:
            self.assertEqual(len(row["all_rule_pair_comparisons"]), 6)
            self.assertGreater(row["all_rule_pair_comparisons"]["A_C"]["laplacian_max_difference"], .09)

    def test_actual_counts_summaries_and_nonoverwrite(self):
        counts = driver.new_counts()
        for r in (self.C, self.D):
            driver.record_counts(counts, r)
        self.assertEqual(counts["successful_kernel_calls"], 2)
        self.assertEqual(counts["all_rule_target_K_matrices"], 12)
        self.assertEqual(counts["per_label_K_cells"], 48)
        self.assertEqual(counts["new_random_draws"], 0)
        self.assertEqual(counts["high_level_factor_totals"]["laplace_exp_minus_ux_factors"], 8 * 513 * (40 + 96))
        report = driver.rule_summary(self.C, self.a, .1)
        self.assertEqual(len(report["matrices"]), 6)
        self.assertTrue(all(row["structural_rowsum_is_not_Ward_verification"] for row in report["matrices"]))
        json.dumps(report, allow_nan=False)
        with tempfile.TemporaryDirectory(prefix="rankone_manufactured_") as td:
            path = Path(td)
            driver.save(path / "FIXTURE.json", report)
            with self.assertRaises(FileExistsError):
                driver.save(path / "FIXTURE.json", {})
            arrays = {k: v for k, v in self.C.items() if isinstance(v, np.ndarray)}
            driver.save_npz(path / "FIXTURE.npz", arrays)
            with self.assertRaises(FileExistsError):
                driver.save_npz(path / "FIXTURE.npz", arrays)
            with np.load(path / "FIXTURE.npz", allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), set(arrays))
                for name, value in arrays.items():
                    np.testing.assert_array_equal(loaded[name], value)

    def test_manufactured_run_delivery_and_failure_retention(self):
        # Only the output plumbing is exercised here: every kernel response is
        # the same already evaluated m=3 fixture, never a scientific anchor.
        anchors = [dict(name="fixture_" + str(i), b=self.b.tolist(), a=self.a.tolist()) for i in range(7)]
        streams = [dict(rep=i, folder="manufactured_source", anchors=anchors, sources={},
            frequency=self.targets["empirical"].tolist(), teacher_pmf=self.targets["population"].tolist(),
            target_masses=[float(q.sum()) for q in self.targets.values()]) for i in range(4)]
        frozen = dict(streams=streams)
        with tempfile.TemporaryDirectory(prefix="rankone_driver_delivery_fixture_") as td:
            base = Path(td)
            out = base / "artifacts"; out.mkdir()
            freeze = out / "FREEZE.json"
            driver.save(freeze, dict(manufactured=True))
            with patch.object(driver, "ROOT", base), patch.object(driver, "OUT", out), \
                    patch.object(driver, "FREEZE", freeze), \
                    patch.object(driver, "verify_freeze", return_value=frozen), \
                    patch.object(driver, "load_stream", side_effect=lambda rep: streams[rep]), \
                    patch.object(driver.kernel, "rankone_stability", side_effect=lambda *args, **kwargs: copy.deepcopy(self.C)) as fake:
                with redirect_stdout(io.StringIO()):
                    driver.run(0)
                self.assertEqual(fake.call_count, 28)
                result = driver.read(out / "rep_00" / "RESULTS.json")
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["primary_matrices"], 42)
                self.assertEqual(result["qualified_primary_matrices"], 42)
                self.assertEqual(result["resources"]["all_rule_target_K_matrices"], 168)
                self.assertEqual(len(list((out / "rep_00").glob("*.npz"))), 28)
                self.assertEqual(len(result["artifacts"]), 65)
                with self.assertRaises(FileExistsError):
                    driver.run(0)
                self.assertEqual(fake.call_count, 28)
                for entry in result["artifacts"].values():
                    self.assertEqual(driver.sha(base / entry["path"]), entry["sha256"])
                fake.side_effect = [copy.deepcopy(self.C), FloatingPointError("manufactured failure")]
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(FloatingPointError, "manufactured failure"):
                    driver.run(1)
                failure = driver.read(out / "rep_01" / "FAILURE.json")
                self.assertEqual(failure["resources"]["successful_kernel_calls"], 1)
                self.assertEqual(failure["resources"]["failed_kernel_calls"], 1)
                self.assertEqual(len(failure["completed_rule_outputs"]), 1)
                self.assertTrue(failure["partial_internal_work_of_failed_kernel_not_counted"])
                self.assertTrue((out / "rep_01" / "fixture_0_A.npz").exists())
                self.assertFalse((out / "rep_01" / "RESULTS.json").exists())


def run_tests():
    before = np.random.get_state()
    tick = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DriverTests)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = np.random.get_state()
    rng_unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    return dict(passed=result.wasSuccessful() and rng_unchanged, tests_run=result.testsRun,
        seconds=time.perf_counter() - tick, manufactured_only=True,
        study_teachers_or_old_endpoints_used=False, random_state_unchanged=rng_unchanged,
        source_sha256=driver.sha(driver.__file__), kernel_sha256=driver.sha(driver.kernel.__file__),
        test_sha256=driver.sha(__file__), diagnostics=DIAGNOSTICS, test_log=log.getvalue())


if __name__ == "__main__":
    outcome = run_tests()
    print(json.dumps(outcome, sort_keys=True), flush=True)
    raise SystemExit(0 if outcome["passed"] else 1)
