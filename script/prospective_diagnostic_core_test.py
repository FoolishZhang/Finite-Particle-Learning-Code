"""Manufactured m2/m3 execution tests only; no development teachers/data."""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prospective_diagnostic_core as core
import numpy as np


def fixture_config():
    rules = {name: dict(normal_rule="hermite", normal_order=order, logu_min=-28.,
                         logu_max=48., logu_step=.25, outer_rule="trapezoid")
             for name, order in (("O", 16), ("C", 24), ("D", 32))}
    return dict(rules=rules,
                stages=[dict(name="O", max_accepted_steps=2, max_calls=6, gradient_tolerance=1e-8),
                        dict(name="C", max_accepted_steps=1, max_calls=4, gradient_tolerance=1e-9)],
                arm_updates=4, arm_switch_step=2, checkpoint_every=2,
                reference_order=6,
                reference_options=dict(maxiter=3, maxfun=6, maxls=3, gtol=1e-7, ftol=1e-12))


def fabricated_anchor():
    q = np.array([.35, .15, .15, .35])
    init = core.sign_initialization(q, 100)
    theta = np.array([.08, -.11, .34, .41])
    signs = init["signs"]
    return dict(q_original=q, q_recoded=init["q_recoded"], signs=signs, sign_resolved=True,
                theta_recoded=theta, theta_signed=np.r_[signs * theta[:2], signs * theta[2:]],
                b=signs * theta[:2], W=np.column_stack((signs * theta[2:], np.zeros(2))),
                finite_feasible=True, status="completed_gradient_stops", solver_execution_completed=True,
                used_initialization_fallback=False, failure=None, initialization=init)


def decisions(**overrides):
    arguments = dict(low_min=[.01, .01000001], reference_min=[-.02, -.02000002],
                     high_curvature=[-.005, -.00500001], low_disagreement=1e-8,
                     reference_disagreement=2e-8, high_disagreement=1e-8,
                     reference_spectrum_disagreement=2e-8, reference_gaps=[.01, .012],
                     reference_prerequisites=dict(anchor_finite_feasible=True, sign_resolved=True,
                                                  box_margin=True, reference_matrix_qualified=True),
                     full_prerequisites=dict(anchor_solver_completed=True, finite_matrices_available=True,
                                             radial_qualified=True, finite_qualified=True))
    arguments.update(overrides)
    return core.classify_diagnostic(**arguments)


class ProspectiveCoreTests(unittest.TestCase):
    diagnostics = {}

    def test_sign_recoding_and_initialization(self):
        q = np.array([.1, .4, .4, .1]); original = q.copy()
        result = core.sign_initialization(q, 20)
        self.assertTrue(result['sign_resolved'])
        self.assertEqual(int(np.prod(result['signs'])), -1)
        self.assertGreater(result['canonical_eigenvector'][result['canonical_index']], 0)
        self.assertGreater(result['eigenvalue_gap'], 1e-8)
        recoded, index = core.recode_frequency(q, result['signs'])
        np.testing.assert_array_equal(recoded[index], q)
        np.testing.assert_array_equal(core.recode_frequency(recoded, result['signs'])[0], q)
        self.assertEqual(recoded.sum(), q.sum())
        np.testing.assert_array_equal(q, original)
        self.assertFalse(result['target_smoothed'])
        self.assertFalse(result['target_renormalized'])
        skew = np.array([.2, .6, .1, .1]); r = core.sign_initialization(skew, 100)
        p = skew @ core.joint.patterns(2)
        expected = np.clip(np.log((p * 100 + .5) / (100 - p * 100 + .5)), -2, 2)
        # log(p)-log1p(-p) and log(odds) have distinct rounding paths.
        np.testing.assert_allclose(r['initial_bias_original'], expected, atol=4 * np.finfo(float).eps, rtol=0)

    def test_sign_fallbacks_are_not_data_smoothing(self):
        q = np.ones(8) / 8
        r = core.sign_initialization(q, 32)
        self.assertFalse(r['sign_resolved'])
        self.assertIn('leading_eigenvalue_gap_unresolved', r['reasons'])
        np.testing.assert_array_equal(r['signs'], np.ones(3))
        q = np.zeros(8); q[0] = 1
        r = core.sign_initialization(q, 32)
        np.testing.assert_array_equal(r['q_recoded'], q)
        np.testing.assert_array_equal(r['correlation'], np.zeros((3, 3)))
        np.testing.assert_array_equal(r['floored_variances'], np.full(3, 1e-8))
        np.testing.assert_array_equal(r['initial_bias_original'], np.full(3, -2.))

    def test_signed_likelihood_gradients_and_finite_differences(self):
        q = np.arange(1., 9.) / 36; signs = np.array([1, -1, 1])
        b = np.array([.2, -.1, .25]); W = np.array([[.31, .02], [-.42, .11], [.2, -.07]])
        banks = np.linspace(-1.3, 1.5, 3 * 8 * 2).reshape(3, 8, 2)
        qr, _ = core.recode_frequency(q, signs)
        original = core.joint.mc_loss_grad(b, W, q, banks)
        recoded = core.joint.mc_loss_grad(signs * b, signs[:, None] * W, qr, banks)
        self.assertAlmostEqual(original[0], recoded[0], delta=1e-15)
        np.testing.assert_allclose(original[1], signs * recoded[1], atol=2e-16, rtol=0)
        np.testing.assert_allclose(original[2], signs[:, None] * recoded[2], atol=3e-16, rtol=0)
        theta = np.r_[b, W.ravel()]; analytic = np.r_[original[1], original[2].ravel()]
        numeric = np.empty(len(theta)); step = 1e-5
        for j in range(len(theta)):
            high, low = theta.copy(), theta.copy(); high[j] += step; low[j] -= step
            numeric[j] = (core.joint.mc_loss_grad(high[:3], high[3:].reshape(3, 2), q, banks)[0]
                          - core.joint.mc_loss_grad(low[:3], low[3:].reshape(3, 2), q, banks)[0]) / (2 * step)
        error = float(np.max(abs(analytic - numeric)))
        self.diagnostics['signed_MC_gradient_FD_error'] = error
        self.assertLess(error, 1e-8)
        gh_orig = core.joint.exact_loss_grad(b, W, q, order=8)
        gh_rec = core.joint.exact_loss_grad(signs * b, signs[:, None] * W, qr, order=8)
        np.testing.assert_allclose(np.r_[gh_orig[0], gh_orig[1], gh_orig[2].ravel()],
                                   np.r_[gh_rec[0], signs * gh_rec[1], (signs[:, None] * gh_rec[2]).ravel()], atol=1e-15, rtol=0)

    def test_signed_angular_Ward_and_common_rotation(self):
        q = np.arange(1., 9.) / 36; signs = np.array([1, -1, 1])
        b, a = np.array([.14, -.23, .19]), signs * np.array([.31, .38, .27])
        qr, _ = core.recode_frequency(q, signs)
        rules = dict(normal_rule='hermite', normal_order=48, logu_min=-28, logu_max=48, logu_step=.25)
        orig = core.rankone.rankone_stationary(b, a, q, [2, 'inf'], **rules)
        rec = core.rankone.rankone_stationary(signs * b, signs * a, qr, [2, 'inf'], **rules)
        np.testing.assert_allclose(orig['Hangle'], rec['Hangle'], atol=2e-16, rtol=0)
        error = float(np.max(abs(orig['Hperp'] @ a - orig['radial_gradient'])))
        self.diagnostics['signed_Ward_error'] = error
        self.assertLess(error, 2e-10)
        W = np.column_stack((a, np.zeros(3)))
        banks = np.linspace(-1.1, 1.2, 4 * 8 * 2).reshape(4, 8, 2)
        angle = .31; R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        one = core.joint.mc_loss_grad(b, W, q, banks)
        two = core.joint.mc_loss_grad(b, W @ R.T, q, banks @ R.T)
        self.assertAlmostEqual(one[0], two[0], delta=1e-15)
        np.testing.assert_allclose(one[2] @ R.T, two[2], atol=2e-16, rtol=0)

    def test_ordered_classifier_all_levels_and_reasons(self):
        self.assertEqual(decisions()['full_level'], 'D3')
        self.assertTrue(decisions()['ref_raise'])
        self.assertEqual(decisions(high_curvature=[.002, .002])['full_level'], 'D2')
        self.assertEqual(decisions(low_min=[-.01, -.01])['full_level'], 'D1')
        self.assertEqual(decisions(reference_min=[.01, .01])['full_level'], 'D1')
        for kwargs, reason in ((dict(low_min=[.0001, .0001]), 'low_resolution_band'),
                               (dict(reference_min=[-.0001, -.0001]), 'reference_resolution_band'),
                               (dict(high_curvature=[.0001, .0001]), 'suppression_confirmed_release_unresolved'),
                               (dict(reference_gaps=[0., .1]), 'reference_direction_gap_unresolved')):
            r = decisions(**kwargs)
            self.assertEqual(r['full_level'], 'D0')
            self.assertIn(reason, r['D0_reason_categories']['resolution'])
        r = decisions(full_prerequisites=dict(finite_qualified=False, anchor_solver_completed=False))
        self.assertEqual(r['full_level'], 'D0'); self.assertTrue(r['ref_raise'])
        self.assertTrue(r['D0_reason_categories']['computation_or_solver'])
        r = decisions(reference_prerequisites=dict(sign_resolved=False))
        self.assertFalse(r['ref_raise']); self.assertTrue(r['D0_reason_categories']['resolution'])
        self.assertFalse(r['D0_reason_categories']['computation_or_solver'])

    def test_A_implies_B_for_manufactured_decision_grid(self):
        count = 0
        for low, ref, high, finite_ok, sign_ok, gap_ok in itertools.product((-.01, 0., .01), (-.01, 0., .01),
                                                                          (-.01, 0., .01), (False, True), (False, True), (False, True)):
            r = decisions(low_min=[low, low], reference_min=[ref, ref], high_curvature=[high, high],
                          reference_gaps=[.1, .1] if gap_ok else [0., 0.],
                          full_prerequisites=dict(finite_qualified=finite_ok),
                          reference_prerequisites=dict(sign_resolved=sign_ok))
            self.assertFalse(r['full_raise'] and not r['ref_raise']); count += 1
        self.diagnostics['manufactured_A_implies_B_cases'] = count

    def test_anchor_solver_snapshots_and_numerical_fallback(self):
        q = np.array([.35, .15, .15, .35]); before = q.copy()
        result = core.fit_anchor(q, 100, manufactured_config=fixture_config())
        self.assertTrue(result['finite_feasible']); self.assertTrue(result['solver_execution_completed'])
        self.assertEqual(result['cost']['kernel_calls'], len(result['calls']))
        self.assertTrue(result['config']['manufactured_only'])
        np.testing.assert_array_equal(q, before)
        stored = result['calls'][0]['theta_recoded'].copy()
        result['theta_recoded'][:] = 99
        np.testing.assert_array_equal(result['calls'][0]['theta_recoded'], stored)
        with patch.object(core.rankone, 'rankone_stationary', side_effect=FloatingPointError('manufactured failure')):
            failed = core.fit_anchor(q, 100, manufactured_config=fixture_config())
        self.assertTrue(failed['used_initialization_fallback']); self.assertFalse(failed['solver_execution_completed'])
        self.assertEqual(failed['fallback_reason'], 'no_finite_evaluated_accepted_state')
        np.testing.assert_array_equal(failed['theta_recoded'], failed['initialization']['theta0_recoded'])
        self.assertIsNotNone(failed['failure'])

    def test_diagnosis_cost_partition_and_finite_failure_preserves_B(self):
        anchor = fabricated_anchor(); q = anchor['q_original']; cfg = fixture_config()
        normal = core.diagnose(anchor, q, manufactured_config=cfg)
        self.assertEqual(normal['reference_only']['cost']['kernel_calls'], 2)
        self.assertEqual(normal['finite_extra']['cost']['kernel_calls'], 2)
        self.assertEqual(normal['cost']['always_policy_diagnostic_seconds'], 0)
        self.assertAlmostEqual(normal['cost']['reference_only']['wall_seconds'] + normal['cost']['finite_extra']['wall_seconds'],
                               normal['cost']['wall_seconds'], delta=1e-14)
        old = core.rankone.rankone_stationary
        def fail_finite(*args, **kwargs):
            if kwargs['K_values'] != ('inf',): raise FloatingPointError('manufactured finite-only failure')
            return old(*args, **kwargs)
        with patch.object(core.rankone, 'rankone_stationary', side_effect=fail_finite):
            failed = core.diagnose(anchor, q, manufactured_config=cfg)
        self.assertEqual(failed['ref_raise'], normal['ref_raise'])
        self.assertEqual(failed['full_level'], 'D0')
        self.assertEqual(failed['numerics']['low_min'], [None, None])
        self.assertIsNone(failed['numerics']['low_disagreement'])
        self.assertEqual(failed['finite_extra']['cost']['failed_kernel_calls'], 2)
        self.assertIn('reference_matrix_qualified', failed['reference_prerequisites'])

    def test_neutral_shared_kick_and_no_redraw(self):
        anchor = fabricated_anchor(); xi = np.array([-.5, .8]); before = xi.copy()
        start = core.neutral_kick(anchor, xi)
        expected = xi - xi.mean(); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(start['direction'], expected, atol=0, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(start['W'], axis=1), anchor['theta_recoded'][2:], atol=1e-16, rtol=0)
        np.testing.assert_array_equal(xi, before)
        self.assertFalse(start['resampled']); self.assertGreaterEqual(start['cost']['wall_seconds'], 0)
        with self.assertRaises(ValueError): core.neutral_kick(anchor, np.ones(2))
        with self.assertRaises(ValueError): core.neutral_kick(anchor, xi, .2)

    def test_neutral_kick_stable_small_nonzero_norm(self):
        anchor = fabricated_anchor(); xi = np.array([-3., 2.]) * 1e-300
        result = core.neutral_kick(anchor, xi)
        self.assertGreater(result['centered_norm'], 0.)
        self.assertEqual(result['normalization_method'], 'maxabs_scaled_L2')
        np.testing.assert_allclose(result['direction'], np.array([-1., 1.]) / np.sqrt(2), atol=3e-16, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(result['direction']), 1., delta=3e-16)
        self.assertFalse(result['resampled'])
        self.assertEqual(result['generated_random_vectors'], 0)
        with self.assertRaises(ValueError): core.neutral_kick(anchor, np.full(2, 1e-300))

    def test_LOW_HIGH_actual_particle_counts_and_Adam_replay(self):
        anchor = fabricated_anchor(); q = anchor['q_original']; start = core.neutral_kick(anchor, [-.5, .8])
        original_start = copy.deepcopy(start); pool = np.linspace(-1.5, 1.4, 4 * 4 * 64 * 2).reshape(4, 4, 64, 2)
        saved_pool = pool.copy(); cfg = fixture_config()
        result = {K: core.train_arm(q, start, pool, K, manufactured_config=cfg) for K in (8, 64)}
        self.assertEqual(result[8]['cost']['processed_prior_vectors'], 4 * 32)
        self.assertEqual(result[64]['cost']['processed_prior_vectors'], 4 * 256)
        for K, r in result.items():
            self.assertEqual(r['completed_updates'], 4); self.assertEqual(r['status'], 'completed_fixed_budget')
            self.assertEqual([c['step'] for c in r['checkpoints']], [0, 2, 4])
            trace = r['trace']; theta = trace['theta'][0].copy(); first = theta * 0; second = theta * 0
            for step, g in enumerate(trace['gradient_before_update']):
                first = .9 * first + .1 * g; second = .999 * second + .001 * g * g
                rate = .01 if step < 2 else .002
                theta = np.clip(theta - rate * (first / (1 - .9 ** (step + 1))) /
                                (np.sqrt(second / (1 - .999 ** (step + 1))) + 1e-8), -2, 2)
                np.testing.assert_allclose(theta, trace['theta'][step + 1], atol=3e-16, rtol=0)
            self.assertFalse(r['endpoint_objective_reevaluated'])
            self.assertEqual(r['cost']['label_node_products'], 4 * 4 * K * np.count_nonzero(q))
        np.testing.assert_array_equal(pool, saved_pool)
        np.testing.assert_array_equal(start['b'], original_start['b']); np.testing.assert_array_equal(start['W'], original_start['W'])
        result[8]['theta'][:] = 99
        self.assertTrue(np.all(abs(result[8]['checkpoints'][0]['theta']) < 2))

    def test_nonfinite_training_retains_last_update(self):
        anchor = fabricated_anchor(); start = core.neutral_kick(anchor, [-.5, .8]); q = anchor['q_original']
        pool = np.zeros((4, 4, 64, 2)); old = core.joint.mc_loss_grad; count = 0
        def objective(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise FloatingPointError('manufactured bad objective')
            return old(*args, **kwargs)
        with patch.object(core.joint, 'mc_loss_grad', side_effect=objective):
            r = core.train_arm(q, start, pool, 8, manufactured_config=fixture_config())
        self.assertEqual(r['completed_updates'], 1)
        self.assertEqual(r['status'], 'nonfinite_objective_stopped_last_finite')
        self.assertEqual(r['cost']['processed_prior_vectors'], 64)
        np.testing.assert_array_equal(r['theta'], r['trace']['theta'][-1])
        self.assertEqual(r['cost']['failed_kernel_calls'], 1)

    def test_reference_hard_cap_does_not_return_last_trial(self):
        anchor = fabricated_anchor(); start = core.neutral_kick(anchor, [-.5, .8]); cfg = fixture_config()
        cfg['reference_options']['maxfun'] = 2
        accepted_state = []
        def optimizer(fun, initial, *, callback, **kwargs):
            fun(initial.copy()); accepted = initial * .9; fun(accepted.copy()); callback(accepted)
            accepted_state.append(accepted.copy())
            fun(accepted * .8)  # Third request must not evaluate the objective.
            raise AssertionError('hard cap did not fire')
        with patch.object(core, 'minimize', side_effect=optimizer):
            r = core.fit_reference(anchor['q_original'], start, manufactured_config=cfg)
        self.assertEqual(r['status'], 'objective_budget_stop')
        self.assertEqual(r['cost']['attempted_calls'], 3); self.assertEqual(r['cost']['actual_evaluations'], 2)
        np.testing.assert_array_equal(r['theta'], accepted_state[0])
        self.assertFalse(r['restarted'])

    def test_reference_nonfinite_and_normal_real_optimizer(self):
        anchor = fabricated_anchor(); start = core.neutral_kick(anchor, [-.5, .8]); cfg = fixture_config()
        with patch.object(core.joint, 'exact_loss_grad', side_effect=FloatingPointError('manufactured')):
            failed = core.fit_reference(anchor['q_original'], start, manufactured_config=cfg)
        self.assertEqual(failed['status'], 'nonfinite_stopped_last_accepted')
        np.testing.assert_array_equal(failed['theta'], np.r_[start['b'], start['W'].ravel()])
        good = core.fit_reference(anchor['q_original'], start, manufactured_config=cfg)
        self.assertLessEqual(good['cost']['actual_evaluations'], 6)
        self.assertTrue(np.isfinite(good['theta']).all())
        self.assertIn(good['status'], ('converged_gradient', 'converged_function_tolerance', 'iteration_budget_stop',
                                     'objective_budget_stop', 'line_search_stop', 'optimizer_stopped_without_convergence'))
        self.assertEqual(good['cost']['label_node_products'], good['cost']['actual_evaluations'] * 36 * 4)

    def test_offline_AB_risk_and_cost_identities(self):
        gain = np.array([-.01, .003, 0., .001]); A = np.array([0, 1, 0, 1]); B = np.array([1, 1, 0, 1])
        L = np.array([.03, .04, .05, .02]); H = L - gain
        r = core.offline_policy_algebra(gain, A, B, anchor_seconds=1., reference_seconds=2.,
                                       finite_extra_seconds=5., low_seconds=7., high_seconds=12.)
        riskA = (1 - A) * L + A * H; riskB = (1 - B) * L + B * H
        np.testing.assert_allclose(r['reference_minus_full_risk'], riskB - riskA, atol=4e-18, rtol=0)
        np.testing.assert_allclose(r['full_minus_low_risk'], riskA - L, atol=4e-18, rtol=0)
        np.testing.assert_allclose(r['full_minus_high_cost'], r['full_policy_cost'] - r['always_high_cost'], atol=0, rtol=0)
        np.testing.assert_allclose(r['reference_minus_high_cost'], r['reference_policy_cost'] - r['always_high_cost'], atol=0, rtol=0)
        self.assertTrue(r['fewer_raises_alone_is_not_risk_improvement'])

    def test_manufactured_overrides_cannot_touch_m8(self):
        with self.assertRaises(ValueError): core.fit_anchor(np.ones(256) / 256, 1024, manufactured_config=fixture_config())
        self.assertEqual(core.configuration()['arm_updates'], 600)
        self.assertEqual(core.configuration()['reference_options']['maxfun'], 180)


if __name__ == '__main__':
    before = np.random.get_state()
    old_files = [Path(core.stationary.__file__), Path(core.rankone.__file__), Path(core.joint.__file__)]
    digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in old_files}
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProspectiveCoreTests))
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    old_unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == digests[str(p)] for p in old_files)
    print(json.dumps(dict(passed=result.wasSuccessful() and unchanged and old_unchanged, tests_run=result.testsRun,
                         seconds=time.perf_counter() - started, manufactured_m2_m3_only=True,
                         study_teacher_or_data_used=False, random_state_unchanged=unchanged,
                         frozen_sources_unchanged=old_unchanged, diagnostics=ProspectiveCoreTests.diagnostics), sort_keys=True))
    sys.exit(0 if result.wasSuccessful() and unchanged and old_unchanged else 1)
