"""Manufactured m2/m3 arrays only; no saved scientific object is restored."""
from __future__ import annotations
import copy
import itertools
import json
import math
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
import prospective_explanation_compute as compute


def manufactured_unit():
    m = 2; Y = compute.patterns(m); q = np.array([.4, .1, .2, .3]); signs = np.array([1, -1], dtype=np.int8)
    qr = compute.recode_weights(q, signs)
    d = np.array([-math.sqrt(.5), np.nextafter(math.sqrt(.5), 0.)])
    theta = np.array([.1, -.2, .5, .6])
    label_H = np.array([.01, -.02, .04, .03])[:, None, None] * np.array([[1., -1.], [-1., 1.]])
    label_G = np.arange(-8., 8.).reshape(4, 4) / 70.
    raw_ref = {}; raw_finite = {}
    for i, rule in enumerate(('C', 'D')):
        H = label_H * (1 + i * 1e-10); G = label_G * (1 + i * 1e-10)
        raw_ref[rule] = dict(K_values=['inf'], patterns=Y.copy(), label_Hangle=H[None].copy(), label_gradient=G[None].copy(),
                             Hangle=np.einsum('p,pij->ij', qr, H)[None, None], gradient=(qr @ G)[None, None])
        raw_finite[rule] = dict(K_values=[8, 64], gradient=np.array([[[.001, -.002, .003, -.004], [.1, -.2, .3, -.4]]]) * (1 + i * 1e-10))
    diagnosis = dict(full_raise=True, ref_raise=True, full_level='D3', full_reasons=['fixture'], ref_reasons=[],
                      D0_reason_categories=dict(computation_or_solver=[], resolution=[]),
                      thresholds=dict(low=5e-4, reference=5e-4, high=5e-4), signs=dict(reference='negative'),
                      numerics=dict(direction=d.copy()), reference_only=dict(raw=raw_ref), finite_extra=dict(raw=raw_finite))
    anchor = dict(theta_recoded=theta.copy(), signs=signs.copy(), q_original=q.copy(), q_recoded=qr.copy(),
                  box=dict(feasible=True, minimum_margin=.45))
    teachers = np.repeat(np.array([[.32, .18, .21, .29]]), 4, axis=0)
    perturb = np.array([1., -1., -1., 1.])
    for i in range(4): teachers[i] += (i - 1) * 1e-12 * perturb
    models = {arm: np.repeat(np.array([p]), 4, axis=0) for arm, p in
              [('LOW', [.30, .20, .24, .26]), ('HIGH', [.31, .19, .22, .28]), ('REF', [.32, .18, .20, .30])]}
    KL = {arm: np.sum(teachers * (np.log(teachers) - np.log(P)), axis=1) for arm, P in models.items()}
    gain = np.sum(teachers * (np.log(models['HIGH']) - np.log(models['LOW'])), axis=1)
    rules = [[4, 0.], [5, 0.], [6, 0.], [5, .2]]
    risk = dict(pmfs=models, KL=KL, paired_gain=gain, main_index=1, rules=rules,
                qualified_by_arm={arm: True for arm in models}, LOW_HIGH_risk_qualified=True)
    unit = dict(id='manufactured_m2', teacher=0, N=20, rep=0)
    return unit, q, anchor, diagnosis, risk, teachers, rules


class ExplanationComputeTests(unittest.TestCase):
    diagnostics = {}

    def test_raw_two_cell_marginals_and_identity_m3(self):
        P = np.arange(1., 9.) * .02
        Q = np.arange(8., 0., -1.) * .03
        result = compute.risk_decomposition(P, Q)
        expected_P = np.zeros((3, 2)); expected_Q = np.zeros((3, 2))
        for i, bits in enumerate(itertools.product((0, 1), repeat=3)):
            for j, bit in enumerate(bits): expected_P[j, bit] += P[i]; expected_Q[j, bit] += Q[i]
        np.testing.assert_allclose(result['teacher_marginals'], expected_P, atol=2e-16, rtol=0)
        np.testing.assert_allclose(result['model_marginals'], expected_Q, atol=2e-16, rtol=0)
        self.assertGreater(abs(result['teacher_marginals'][0, 0] - (1 - result['teacher_marginals'][0, 1])), .2)
        raw_KL = math.fsum(float(P[i] * math.log(P[i] / Q[i])) for i in range(8))
        self.assertAlmostEqual(result['raw_KL'], raw_KL, delta=3e-16)
        self.assertLess(abs(result['identity_residual']), 1e-14)
        self.assertTrue(result['identity_qualified']); self.assertFalse(result['masses_renormalized'])
        self.diagnostics['raw_nonunit_identity_residual'] = result['identity_residual']

    def test_dependence_score_is_not_a_KL_and_gain_identity(self):
        P = np.array([.4, .1, .1, .4]); low = np.array([.1, .4, .4, .1]); high = np.array([.25] * 4)
        account = compute.risk_decomposition(P, low)
        self.assertLess(account['dependence_score'], 0)
        self.assertFalse(account['dependence_score_is_dependency_KL'])
        gain = compute.gain_decomposition(P, low, high)
        direct = math.fsum(float(P[i] * math.log(high[i] / low[i])) for i in range(4))
        self.assertAlmostEqual(gain['direct_gain'], direct, delta=3e-16)
        self.assertLess(abs(gain['identity_residual']), 1e-14)
        self.assertAlmostEqual(gain['marginal_gain'], 0., delta=1e-15)
        self.assertFalse(gain['radial_angle_causal_decomposition'])

    def test_small_negative_raw_risk_is_not_clipped(self):
        P = np.full(4, .25) * (1 - 1e-12); Q = np.full(4, .25)
        result = compute.risk_decomposition(P, Q)
        self.assertLess(result['raw_KL'], 0.)
        self.assertFalse(result['rounding_negatives_clipped'])
        self.assertLess(abs(result['identity_residual']), 1e-14)

    def test_support_conflicts_preserve_infinity_and_undefined_decomposition(self):
        P = np.full(4, .25); Q = np.array([.5, .5, 0., 0.])
        result = compute.risk_decomposition(P, Q)
        self.assertTrue(math.isinf(result['raw_KL']))
        self.assertIsNone(result['dependence_score']); self.assertIsNone(result['identity_residual'])
        self.assertFalse(result['identity_qualified'])
        self.assertIn('undefined_support_conflict', result['identity_status'])
        joint_zero = compute.risk_decomposition(P, [.5, 0., 0., .5])
        self.assertEqual(joint_zero['dependence_score'], -math.inf)
        self.assertEqual(joint_zero['identity_status'], 'extended_positive_infinity_agreement')
        common = compute.risk_decomposition([1., 0., 0., 0.], [1., 0., 0., 0.])
        self.assertEqual(common['raw_KL'], 0.); self.assertTrue(common['identity_qualified'])
        json.dumps(compute.public(result), allow_nan=False)

    def test_analytic_independent_MLE_interior_and_boundary(self):
        q = np.arange(1., 9.) / 36
        result = compute.independent_bernoulli_mle(q)
        Y = np.array(list(itertools.product((0, 1), repeat=3)))
        p = np.array([sum(q[i] for i, y in enumerate(Y) if y[j]) for j in range(3)])
        expected = np.array([math.prod(p[j] if y[j] else 1 - p[j] for j in range(3)) for y in Y])
        np.testing.assert_allclose(result['pmf'], expected, atol=2e-16, rtol=0)
        self.assertTrue(result['analytic_fit']); self.assertTrue(result['no_smoothing'])
        self.assertEqual(result['optimizer_updates'], 0)
        boundary = compute.independent_bernoulli_mle([.25, .75, 0., 0.])
        self.assertTrue(boundary['boundary'])
        self.assertEqual(boundary['one_probabilities'][0], 0.)
        self.assertTrue(math.isinf(compute.relative_entropy(np.full(4, .25), boundary['pmf'])))
        self.assertTrue(math.isfinite(boundary['empirical_NLL']))

    def test_reweighting_all_combinations_and_original_direction_bytes(self):
        args = manufactured_unit(); original = copy.deepcopy(args)
        unit, q, anchor, diag, risk, teachers, rules = args
        result = compute.derive_unit(*args)
        signs = anchor['signs']; Y = compute.patterns(2)
        old_to_new = np.array([2 * (y[0] if signs[0] > 0 else 1 - y[0]) + (y[1] if signs[1] > 0 else 1 - y[1]) for y in Y])
        maximum = 0.
        for ri, rule in enumerate(('C', 'D')):
            label_H = diag['reference_only']['raw'][rule]['label_Hangle'][0]
            label_G = diag['reference_only']['raw'][rule]['label_gradient'][0]
            for ti, P in enumerate(teachers):
                H = np.zeros((2, 2)); G = np.zeros(4)
                for yi in range(4): H += P[yi] * label_H[old_to_new[yi]]; G += P[yi] * label_G[old_to_new[yi]]
                maximum = max(maximum, float(np.max(abs(H - result['population_Hangle'][ri, ti]))))
                np.testing.assert_allclose(result['population_Hangle'][ri, ti], H, atol=3e-17, rtol=0)
                np.testing.assert_allclose(result['population_gradient'][ri, ti], G, atol=3e-17, rtol=0)
                self.assertAlmostEqual(result['population_direction_curvature'][ri, ti], float(diag['numerics']['direction'] @ H @ diag['numerics']['direction']), delta=3e-17)
        self.diagnostics['manufactured_reweighting_max_error'] = maximum
        self.assertEqual(result['fixed_empirical_direction'].tobytes(), diag['numerics']['direction'].tobytes())
        self.assertTrue(result['original_A']); self.assertTrue(result['original_B'])
        for key, ki, block in [('8', 0, 'finite_extra'), ('64', 1, 'finite_extra'), ('inf', 0, 'reference_only')]:
            for ri, rule in enumerate(('C', 'D')):
                self.assertEqual(result['empirical_gradients'][key][ri].tobytes(), diag[block]['raw'][rule]['gradient'][0, ki].tobytes())
        compute.same_tree(args, original)

    def test_radial_direction_signed_norm_inner_product_equivalence(self):
        args = manufactured_unit(); result = compute.derive_unit(*args)
        channel = result['gradient_channel']; self.assertTrue(channel['defined'])
        signed_gradient = result['population_gradient_signed_original']
        derivative = np.einsum('rtj,j->rt', signed_gradient, channel['direction_signed_original'])
        np.testing.assert_allclose(derivative, channel['directional_derivatives'], atol=0, rtol=0)
        for rule in ('C', 'D'):
            norms = result['empirical_gradient_norms']['64'][rule]
            self.assertGreater(norms['bias']['L2'], 0.); self.assertGreater(norms['row_amplitude']['L2'], 0.)
        self.assertFalse(channel['finite_step_executed'])
        self.assertGreater(channel['maximum_forward_box_step'], 0.)

    def test_radial_direction_floor_and_missing_original_direction(self):
        args = list(manufactured_unit())
        args[3]['finite_extra']['raw']['D']['gradient'][0, 1] = [1e-10, 0., 0., 0.]
        args[3]['numerics']['direction'] = None
        result = compute.derive_unit(*args)
        self.assertFalse(result['gradient_channel']['defined'])
        self.assertIsNone(result['gradient_channel']['directional_derivatives'])
        self.assertIsNone(result['population_direction_curvature'])
        self.assertEqual(result['population_curvature_readout']['sign'], 'unresolved')
        self.assertFalse(result['population_direction_reselected'])

    def test_new_sign_band_requires_all_eight_not_just_main(self):
        values = np.full((2, 4), 1.05e-8); values[0, 0] = .95e-8
        result = compute.numerical_readout(values)
        self.assertGreater(result['main_value'], result['tau_num'])
        self.assertEqual(result['sign'], 'unresolved')
        self.assertFalse(result['all_variants_beyond_band'])
        self.assertEqual(compute.numerical_readout(np.full((2, 4), -2e-6))['sign'], 'negative')
        self.assertEqual(compute.numerical_readout(np.full((2, 4), 2e-6))['sign'], 'positive')
        self.assertEqual(compute.numerical_readout(np.full((2, 4), 1e-8))['sign'], 'unresolved')

    def test_grouping_and_leave_largest_remain_descriptive(self):
        rows = [dict(unit=f'fixture_{i}', N=4 if i < 6 else 8, A=i < 4, B=i < 5,
                     full_level='D3' if i < 4 else 'D0', saved_delta_R=[.1, -.02, -.03, -.01][i] if i < 4 else .002)
                for i in range(12)]
        original = copy.deepcopy(rows)
        summary = compute.descriptive_groups(rows)
        self.assertEqual(summary['all_units']['units'], 12)
        self.assertEqual(summary['by_N']['4']['units'], 6)
        self.assertEqual(summary['leave_largest_alarm']['removed_unit'], 'fixture_0')
        self.assertAlmostEqual(summary['leave_largest_alarm']['remaining_mean_saved_delta_R'], -.02, delta=1e-17)
        self.assertTrue(summary['leave_largest_alarm']['all_original_units_retained'])
        self.assertEqual(rows, original)

    def test_teacher_distance_is_saved_array_algebra_only(self):
        D1 = dict(pmfs=np.repeat(np.full((1, 4), .25), 4, axis=0), rules=[[4, 0]] * 4)
        D2 = dict(pmfs=np.repeat(np.array([[.3, .2, .2, .3]]), 4, axis=0), rules=[[4, 0]] * 4)
        result = compute.teacher_distance([D1, D2])
        expected = 2 * .3 * math.log(.3 / .25) + 2 * .2 * math.log(.2 / .25)
        np.testing.assert_allclose(result['values'], np.full(4, expected), atol=2e-16, rtol=0)
        self.assertTrue(result['numerical_value_is_not_a_certified_upper_bound'])
        self.assertEqual(result['new_teacher_PMF_calls'], 0)

    def test_public_nonfinite_encoding_and_immutable_outputs(self):
        self.assertEqual(compute.public(math.inf), {'__nonfinite_float__': 'inf'})
        args = manufactured_unit(); out = compute.derive_unit(*args)
        before = args[3]['numerics']['direction'].copy()
        out['fixed_empirical_direction'][:] = 99
        np.testing.assert_array_equal(args[3]['numerics']['direction'], before)
        self.assertFalse(np.shares_memory(out['population_gradient'], out['population_gradient_signed_original']))
        json.dumps(compute.public(out), allow_nan=False)

    def test_freeze_guard_and_static_config_do_not_read_scientific_arrays(self):
        with (patch.object(compute, 'read', side_effect=AssertionError('config must not read files')),
              patch.object(np, 'load', side_effect=AssertionError('config must not load arrays'))):
            config = compute.config_dump()
        self.assertEqual(config['producer_new_quadrature_calls'], 0)
        fake = dict(schema='prospective_explanation_freeze_v1', baseline=compute.BASE, handoff_sha256=compute.HANDOFF_SHA,
                    production_not_started=False, config=config, sources={})
        with patch.object(compute, 'read', return_value=fake), patch.object(compute, '_bind_sources', side_effect=AssertionError('must stop before source binding')):
            with self.assertRaisesRegex(AssertionError, 'new freeze/config'): compute.verify_freeze()


def run_tests():
    before = np.random.get_state(); tick = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExplanationComputeTests))
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    print(json.dumps(dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
                         seconds=time.perf_counter() - tick, manufactured_m2_m3_only=True,
                         real_saved_units_restored=False, new_quadrature_calls=0, new_optimizer_updates=0,
                         new_random_draws=0, random_state_unchanged=unchanged,
                         diagnostics=ExplanationComputeTests.diagnostics), sort_keys=True))
    return 0 if result.wasSuccessful() and unchanged else 1


if __name__ == '__main__': raise SystemExit(run_tests())
