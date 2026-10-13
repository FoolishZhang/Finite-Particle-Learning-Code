"""Manufactured audit fixtures; no development teacher or saved unit is read."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prospective_diagnostic_audit as audit
import prospective_diagnostic_core as core
import numpy as np
from scipy.special import expit


def fixture_configuration():
    def rule(order):
        return dict(normal_rule="hermite", normal_order=order, normal_limit=10.,
                    logu_min=-24., logu_max=36., logu_step=.25, outer_rule="trapezoid")
    return dict(rules=dict(O=rule(12), C=rule(16), D=rule(20)),
        stages=[dict(name="O", max_accepted_steps=1, max_calls=6, gradient_tolerance=1e-9),
                dict(name="C", max_accepted_steps=1, max_calls=6, gradient_tolerance=1e-10)],
        arm_updates=3, arm_switch_step=2, checkpoint_every=1,
        reference_order=6, reference_options=dict(maxiter=2, maxfun=8, maxls=3, gtol=1e-7, ftol=1e-12))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = np.array([.16, .34, .34, .16])
        cls.cfg = fixture_configuration()
        cls.anchor = core.fit_anchor(cls.q, 100, manufactured_config=cls.cfg)
        cls.xi = np.array([.4, -.7])
        cls.start = core.neutral_kick(cls.anchor, cls.xi)
        cls.diag = core.diagnose(cls.anchor, cls.q, manufactured_config=cls.cfg)
        cls.prior = np.sin(np.arange(3 * 4 * 64 * 2) * .031).reshape(3, 4, 64, 2)
        cls.data = dict(q=cls.q, xi=cls.xi, prior=cls.prior, unit=dict(N=100))
        cls.arms = {K: core.train_arm(cls.q, cls.start, cls.prior, K, manufactured_config=cls.cfg) for K in (8, 64)}
        cls.ref = core.fit_reference(cls.q, cls.start, manufactured_config=cls.cfg)

    def assert_passed(self, checks):
        self.assertEqual(checks.failures, [], json.dumps(checks.failures, ensure_ascii=True)[:9000])

    def test_recode_anchor_newton_and_stable_kick(self):
        checks = audit.Checks()
        audit.check_initialization(self.data, self.anchor, checks, "m2")
        audit.check_newton_trace(self.anchor, checks, "m2")
        audit.check_kick(self.data, self.anchor, self.start, checks, "m2")
        self.assert_passed(checks)
        self.assertTrue(np.any(self.anchor["signs"] < 0))
        enormous = dict(self.data, xi=np.array([3e300, -3e300]))
        large_start = core.neutral_kick(self.anchor, enormous["xi"])
        checks = audit.Checks(); audit.check_kick(enormous, self.anchor, large_start, checks, "large_m2")
        self.assert_passed(checks)

    def test_all_saved_raw_aggregates_and_independent_diagnosis(self):
        checks = audit.Checks()
        audit.check_diagnosis(self.anchor, self.diag, checks, "m2")
        self.assert_passed(checks)
        damaged = copy.deepcopy(self.diag)
        damaged["finite_extra"]["raw"]["D"]["Hangle"][0, 0, 0, 1] += .01
        checks = audit.Checks(); audit.check_diagnosis(self.anchor, damaged, checks, "corrupt")
        self.assertTrue(checks.failures)

    def test_three_step_Adam_prefix_and_targeted_independent_gradients(self):
        for K, fit in self.arms.items():
            checks = audit.Checks()
            scope = audit.check_arm(self.data, self.start, fit, checks, f"m2_K{K}")
            self.assert_passed(checks)
            self.assertEqual(scope["saved_updates_replayed"], 3)
            self.assertEqual(scope["independently_recomputed_gradient_steps_zero_based"], [0])
            self.assertEqual(scope["processed_prior_vectors"], 3 * 4 * K)
        bad = copy.deepcopy(self.arms[8]); bad["trace"]["gradient_before_update"][0, 0] += .01
        checks = audit.Checks(); audit.check_arm(self.data, self.start, bad, checks, "corrupt_Adam", targeted=False)
        self.assertTrue(checks.failures)

    def test_reference_LAST_is_accepted_not_final_trial(self):
        checks = audit.Checks(); audit.check_reference(self.start, self.ref, checks, "m2_REF")
        self.assert_passed(checks)
        bad = copy.deepcopy(self.ref); bad["theta"] = bad["theta"] + .001
        checks = audit.Checks(); audit.check_reference(self.start, bad, checks, "wrong_LAST")
        self.assertTrue(checks.failures)

    def test_ordered_rule_and_B_not_cleared_by_finite_failure(self):
        ref = dict(anchor_finite_feasible=True, sign_resolved=True, box_margin=True, reference_matrix_qualified=True)
        full = dict(anchor_solver_completed=True, finite_matrices_available=True, all_full_matrix_qualifications=True,
                    radial_gradient=True, radial_positive=True, Cartesian_Ward=True, near_rotation_mode=True)
        low, reference, high = [.002, .002000001], [-.003, -.003000001], [-.004, -.004000001]
        differences = dict(low=1e-9, reference=1e-9, high=1e-9, reference_spectrum=1e-9)
        def compare(fp, hp=high, gaps=(.01, .01)):
            own = audit.independent_decision(low, reference, hp, differences, gaps, fp, ref)
            expected = core.classify_diagnostic(low_min=low, reference_min=reference, high_curvature=hp,
                low_disagreement=1e-9, reference_disagreement=1e-9, high_disagreement=1e-9,
                reference_spectrum_disagreement=1e-9, reference_gaps=gaps,
                full_prerequisites=fp, reference_prerequisites=ref)
            for key in own: self.assertEqual(own[key], expected[key])
            return own
        self.assertEqual(compare(full)["full_level"], "D3")
        failed = dict(full, all_full_matrix_qualifications=False)
        failure = compare(failed)
        self.assertFalse(failure["full_raise"]); self.assertTrue(failure["ref_raise"])
        self.assertEqual(failure["full_level"], "D0")
        unresolved = compare(full, gaps=(1e-9, 1e-9))
        self.assertEqual(unresolved["full_level"], "D0")
        self.assertTrue(unresolved["ref_raise"])

    def test_B_gate_ignores_cartesian_Hperp_but_FULL_does_not(self):
        C = copy.deepcopy(self.diag["reference_only"]["raw"]["C"])
        # This gate-selection fixture makes scalar prerequisites exact; it is
        # not a claim that the low-order integration fixture met those gates.
        C["normal_quadrature_mass"] = 1.
        C["raw_bank_quadrature_mass"][:] = 1.
        C["label_d_mass"][:] = 1.
        C["label_c_mass"][:] = 1.
        C["raw_c_mass_upper"][:] = 1.
        C["ward_residual_max"][:] = 0.
        D = copy.deepcopy(C)
        # Preserve mass/Ward fixtures so only Hperp disagreement is introduced.
        D["Hperp"][0, 0, 0, 0] += .01
        self.assertTrue(audit.numerical_quality(C, D, 0, reference_only=True))
        self.assertFalse(audit.numerical_quality(C, D, 0, reference_only=False))

    def test_generated_label_reconstruction_uses_saved_inputs(self):
        N = 7; z = np.linspace(-1., 1., N * 2).reshape(N, 2); u = np.linspace(.05, .95, N * 8).reshape(N, 8)
        teacher = dict(b=np.linspace(-.1, .1, 8), a=np.full(8, .2), phi=np.linspace(-.25, .25, 8))
        W = teacher["a"][:, None] * np.c_[np.cos(teacher["phi"]), np.sin(teacher["phi"])]
        labels = (u < expit(teacher["b"] + z @ W.T)).astype(np.int8)
        counts = np.bincount(labels @ 2 ** np.arange(7, -1, -1), minlength=256)
        unit = dict(id="manufactured", N=N, teacher=0, rep=0)
        data = dict(unit=unit, latent=z, uniforms=u, labels=labels, counts=counts, q=counts/N,
            prior=np.zeros((600, 4, 64, 2)), prior_vectors_generated=153600,
            Gaussian_scalars=2*N+8+600*4*64*2, uniform_scalars=8*N,
            seconds=.01, data_generation_seconds=.005, kick_generation_seconds=.001, prior_generation_seconds=.003)
        checks = audit.Checks(); audit.check_generated(unit, data, teacher, checks); self.assert_passed(checks)
        bad = copy.deepcopy(data); bad["labels"][0, 0] = 1 - bad["labels"][0, 0]
        checks = audit.Checks(); audit.check_generated(unit, bad, teacher, checks); self.assertTrue(checks.failures)

    def test_risk_and_scalar_policy_identities(self):
        p = np.array([.2, .25, .25, .3]); target = np.tile(p, (4, 1))
        pmfs = {"LOW": np.tile([.21, .24, .25, .3], (4, 1)), "HIGH": np.tile([.205, .245, .25, .3], (4, 1)), "REF": target.copy()}
        KL = {name: np.sum(target*(np.log(target)-np.log(P)), axis=1) for name,P in pmfs.items()}
        q = np.array([.21, .26, .24, .29])
        risk = dict(pmfs=pmfs, KL=KL, empirical_NLL={name:-np.log(P)@q for name,P in pmfs.items()},
            paired_gain=np.sum(target*(np.log(pmfs["HIGH"])-np.log(pmfs["LOW"])),axis=1), main_index=1,
            rules=[[40,0.],[60,0.],[80,0.],[72,np.pi/7]], qualified_by_arm={k:True for k in pmfs},
            errors={k:dict(mass_error=max(float(np.max(abs(P.sum(axis=1)-1))),float(np.max(abs(target.sum(axis=1)-1)))),pmf_max_difference=0.) for k,P in pmfs.items()},
            LOW_HIGH_risk_qualified=True)
        checks=audit.Checks();audit.check_risk(dict(q=q),target,risk,checks,"m2_risk");self.assert_passed(checks)
        from prospective_diagnostic_analyze_test import rows_fixture
        for row in rows_fixture():
            checks=audit.Checks();audit.check_policies(row,checks);self.assert_passed(checks)

    def test_global_lock_order_records(self):
        with tempfile.TemporaryDirectory(prefix="prospective_order_fixture_") as td:
            root=Path(td);out=root/'art';out.mkdir();start=datetime(2020,1,1,tzinfo=timezone.utc)
            for i,name in enumerate(("INPUTS_LOCK","DIAGNOSIS_LOCK","ENDPOINTS_LOCK","EVALUATION_LOCK")):
                file=out/f'unit_{i}.json';file.write_text(json.dumps(dict(created_utc=(start+timedelta(seconds=2*i+1)).isoformat())))
                (out/(name+'.json')).write_text(json.dumps(dict(created_utc=(start+timedelta(seconds=2*i+2)).isoformat(),files={str(file.relative_to(root)):{} })))
            with patch.object(audit,"ROOT",root),patch.object(audit,"OUT",out):
                checks=audit.Checks();audit.lock_order(checks);self.assert_passed(checks)
                (out/'unit_3.json').write_text(json.dumps(dict(created_utc=start.isoformat())))
                checks=audit.Checks();audit.lock_order(checks);self.assertTrue(checks.failures)


if __name__ == "__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    print(json.dumps(dict(passed=result.wasSuccessful(),tests_run=result.testsRun,manufactured_only=True,
                         real_development_inputs_used=False,no_optimizer_retraining_of_study=True)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
