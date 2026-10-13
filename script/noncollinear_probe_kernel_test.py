"""Small manufactured paired-likelihood fixtures; no scientific inputs."""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import copy
from decimal import Decimal, localcontext
import itertools
import json
import math
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
import noncollinear_probe_kernel as kernel


def independent_axis(spec):
    if spec['rule'] == 'hermite':
        z, w = hermegauss(spec['order'])
        return z, w / math.sqrt(2 * math.pi)
    z, w = leggauss(spec['order'])
    z = spec['limit'] * z
    return z, spec['limit'] * w * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi)


def direct_pair_bank(b, W0, W1, q, specs):
    axes = [independent_axis(s) for s in specs]
    n0, n1 = [len(z) for z, _ in axes]
    ix = np.indices((n0, n1, n0, n1)).reshape(4, -1).T
    bank = np.stack((np.column_stack((axes[0][0][ix[:, 0]], axes[1][0][ix[:, 1]])),
                     np.column_stack((axes[0][0][ix[:, 2]], axes[1][0][ix[:, 3]]))), axis=1)
    w = axes[0][1][ix[:, 0]] * axes[1][1][ix[:, 1]] * axes[0][1][ix[:, 2]] * axes[1][1][ix[:, 3]]
    Ys = np.array(list(itertools.product((0, 1), repeat=len(b))))
    logmeans = []
    for W in (W0, W1):
        eta = bank @ W.T + b
        logs = np.stack([np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta)).sum(axis=-1) for y in Ys], axis=-1)
        logmeans.append(logsumexp(logs, axis=1) - math.log(2))
    labels = np.column_stack((-w @ logmeans[0], -w @ logmeans[1]))
    delta_labels = w @ (logmeans[0] - logmeans[1])
    return dict(absolute_loss=np.asarray(q) @ labels, paired_delta=float(np.asarray(q) @ delta_labels),
                label_loss=labels, label_delta=delta_labels, bank_mass=float(w.sum()))


def direct_pmf(b, W, specs):
    axes = [independent_axis(s) for s in specs]
    z0, z1 = np.meshgrid(axes[0][0], axes[1][0], indexing='ij')
    z = np.column_stack((z0.ravel(), z1.ravel()))
    w = (axes[0][1][:, None] * axes[1][1][None, :]).ravel()
    eta = z @ W.T + b
    pmf = []
    for y in itertools.product((0, 1), repeat=len(b)):
        logs = np.where(np.array(y), -np.logaddexp(0, -eta), -np.logaddexp(0, eta)).sum(axis=1)
        pmf.append(float(w @ np.exp(logs)))
    return np.array(pmf)


class NoncollinearKernelTests(unittest.TestCase):
    diagnostics = {}
    specs = ({'rule': 'hermite', 'order': 7}, {'rule': 'hermite', 'order': 7})

    def test_direct_four_dimensional_K2(self):
        b = np.array([.21, -.18]); W0 = np.array([[.4, 0.], [.3, .15]])
        W1 = np.array([[.35, .2], [.31, -.08]]); q = np.array([.13, .22, .27, .38])
        start = time.perf_counter()
        result = kernel.paired_losses(b, W0, W1, q, (2, 'inf'), normal_specs=self.specs,
                                      logu_min=-28, logu_max=64, logu_step=.125)
        self.diagnostics['m2_K2_inf_preparation_and_evaluation_seconds'] = time.perf_counter() - start
        direct = direct_pair_bank(b, W0, W1, q, self.specs)
        errors = {key: float(np.max(abs(result[key][0, 0] - direct[key]))) for key in ('absolute_loss', 'paired_delta')}
        errors['label_delta'] = float(np.max(abs(result['label_delta'][0] - direct['label_delta'])))
        errors['label_loss'] = float(np.max(abs(result['label_loss'][0] - direct['label_loss'])))
        self.diagnostics['K2_direct_4D_errors'] = errors
        self.assertLess(max(errors.values()), 2e-11)
        self.assertLess(float(np.max(abs(result['delta_consistency_error']))), 2e-15)

    def test_nonunit_mass_is_raw_bank_mass(self):
        specs = ({'rule': 'legendre', 'order': 8, 'limit': 1.2}, {'rule': 'hermite', 'order': 7})
        b = np.array([.15, -.26]); W0 = np.array([[.35, 0.], [.41, 0.]])
        W1 = np.array([[.33, .11], [.4, -.09]]); q = np.array([.24, .19, .31, .26])
        baseline = kernel.prepare_baseline(b, W0, q, (2, 'inf'), normal_specs=specs,
                                           logu_min=-28, logu_max=64, logu_step=.125)
        result = kernel.evaluate_prepared(baseline, W1)
        direct = direct_pair_bank(b, W0, W1, q, specs)
        self.assertLess(baseline['normal_quadrature_mass'], .8)
        error = max(float(np.max(abs(result['absolute_loss'][0, 0] - direct['absolute_loss']))),
                    abs(float(result['paired_delta'][0, 0]) - direct['paired_delta']))
        self.diagnostics['raw_normal_mass_K2_error'] = error
        self.assertLess(error, 2e-11)
        self.assertAlmostEqual(baseline['raw_bank_quadrature_mass'][0], direct['bank_mass'], delta=2e-15)
        np.testing.assert_allclose(baseline['log_likelihood_scale'],
                                   baseline['log_evidence'] - math.log(baseline['normal_quadrature_mass']), atol=0, rtol=0)
        self.assertFalse(baseline['config']['normal_weights_renormalized'])

    def test_integrated_evidence_KL_and_marginals(self):
        b = np.array([.18, -.21]); W0 = np.array([[.38, 0.], [.44, 0.]])
        W1 = np.array([[.36, .13], [.41, -.16]])
        q = np.array([[.13, .22, .27, .38], [.25, .25, .25, .25]])
        result = kernel.paired_losses(b, W0, W1, q, (11, 12, 'inf'), normal_specs=self.specs)
        p = np.stack([direct_pmf(b, W, self.specs) for W in (W0, W1)])
        np.testing.assert_allclose(result['pmf'], p, atol=3e-16, rtol=0)
        expected_losses = -(q @ np.log(p).T)
        np.testing.assert_allclose(result['absolute_loss'][:, 2, :], expected_losses, atol=9e-16, rtol=0)
        expected_delta = -(q @ (np.log(p[1]) - np.log(p[0])))
        np.testing.assert_allclose(result['paired_risk_delta'], expected_delta, atol=8e-16, rtol=0)
        np.testing.assert_allclose(result['paired_delta'][:, 2], result['paired_risk_delta'], atol=0, rtol=0)
        expected_KL = (q * np.log(q)).sum(axis=1)[:, None] + expected_losses
        np.testing.assert_allclose(result['raw_KL'], expected_KL, atol=9e-16, rtol=0)
        np.testing.assert_allclose(result['marginals'], p @ result['patterns'], atol=3e-16, rtol=0)
        self.assertEqual(result['paired_integrand'].shape, (2, 4, len(result['outer_weights'])))
        reconstructed = np.einsum('tp,kpv,v->tk', q, result['paired_integrand'], result['outer_weights'])
        np.testing.assert_allclose(result['paired_delta'][:, :2], reconstructed, atol=3e-16, rtol=0)

    def test_path_evenness_norms_and_fixed_marginals(self):
        b, a, d = np.array([.12, -.17, .22]), np.array([.36, -.44, .31]), np.array([.5, -.7, .2])
        q = np.arange(1., 9.) / 36
        specs = ({'rule': 'hermite', 'order': 20}, {'rule': 'hermite', 'order': 18})
        W0 = kernel.path_W(a, d, 0.)
        baseline = kernel.prepare_baseline(b, W0, q, (11, 12, 'inf'), normal_specs=specs)
        plus = kernel.evaluate_prepared(baseline, kernel.path_W(a, d, .13))
        minus = kernel.evaluate_prepared(baseline, kernel.path_W(a, d, -.13))
        np.testing.assert_allclose(np.linalg.norm(plus['W1'], axis=1), abs(a), atol=1e-16, rtol=0)
        even_error = float(np.max(abs(plus['paired_delta'] - minus['paired_delta'])))
        marginal_error = float(np.max(abs(plus['marginals'][1] - plus['marginals'][0])))
        self.diagnostics['manufactured_path_evenness_error'] = even_error
        self.diagnostics['manufactured_path_marginal_error'] = marginal_error
        self.assertLess(even_error, 2e-15)
        self.assertLess(marginal_error, 3e-14)
        self.assertFalse(plus['qualifications']['symmetry_used_to_skip_evaluation'])
        zero = kernel.evaluate_prepared(baseline, W0)
        np.testing.assert_array_equal(zero['paired_delta'], np.zeros((1, 3)))
        np.testing.assert_array_equal(zero['paired_integrand'], np.zeros_like(zero['paired_integrand']))

    def test_mixed_axis_order_and_chunk_invariance(self):
        b, W0, W1, q = [.1, -.2], [[.3, 0.], [.4, 0.]], [[.28, .1], [.38, -.12]], [.1, .2, .3, .4]
        specs = ({'rule': 'legendre', 'order': 27, 'limit': 6.}, {'rule': 'hermite', 'order': 9})
        small = kernel.paired_losses(b, W0, W1, q, (2, 'inf'), normal_specs=specs, outer_chunk=7, label_chunk=1)
        large = kernel.paired_losses(b, W0, W1, q, (2, 'inf'), normal_specs=specs, outer_chunk=40, label_chunk=4)
        self.assertEqual(small['normal_nodes'].shape, (27 * 9, 2))
        np.testing.assert_array_equal(small['normal_nodes'][:9, 0], np.repeat(small['normal_axis_0_nodes'][0], 9))
        np.testing.assert_array_equal(small['normal_nodes'][:9, 1], small['normal_axis_1_nodes'])
        for key in ('pmf', 'absolute_loss', 'paired_delta', 'paired_risk_delta'):
            np.testing.assert_allclose(small[key], large[key], atol=2e-15, rtol=0)
        self.assertEqual(small['work_counts']['laplace_exp_minus_ux_factors'], 2 * 4 * 243 * len(small['outer_weights']))

    def test_prepared_and_outputs_are_snapshots(self):
        b = np.array([.1, -.2]); W0 = np.array([[.3, 0.], [.4, 0.]])
        W1 = np.array([[.28, .1], [.38, -.12]]); q = np.array([.1, .2, .3, .4])
        prepared = kernel.prepare_baseline(b, W0, q, (2, 'inf'), normal_specs=self.specs)
        reference = copy.deepcopy(prepared)
        result = kernel.evaluate_prepared(prepared, W1)
        def compare(a, b):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, dict):
                for key in a: compare(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                for aa, bb in zip(a, b): compare(aa, bb)
            else: self.assertEqual(a, b)
        compare(prepared, reference)
        def arrays(value):
            if isinstance(value, np.ndarray): return [value]
            if isinstance(value, dict): return [a for v in value.values() for a in arrays(v)]
            if isinstance(value, (list, tuple)): return [a for v in value for a in arrays(v)]
            return []
        outputs = arrays(result)
        inputs = arrays(prepared) + [b, W0, W1, q]
        for i, arr in enumerate(outputs):
            for other in outputs[i + 1:] + inputs:
                self.assertFalse(np.shares_memory(arr, other))
        self.assertEqual(prepared['work_counts']['baseline_preparations'], 1)
        self.assertEqual(result['work_counts']['baseline_preparations'], 0)
        self.assertEqual(result['work_counts']['perturbed_model_evaluations'], 1)
        result['normal_weights'][:] = 99
        compare(prepared, reference)

    def test_expm1_preserves_tiny_integrand_difference(self):
        left, right = -1e-12, -1.00000001e-12
        self.assertEqual(float(np.exp(left) - np.exp(right)), 0.)
        value = float(kernel._exp_difference(np.array(left), np.array(right)))
        with localcontext() as ctx:
            ctx.prec = 80
            expected = float(Decimal.from_float(left).exp() - Decimal.from_float(right).exp())
        self.assertNotEqual(value, 0.)
        self.assertLess(abs(value - expected), 2e-35)
        np.testing.assert_array_equal(kernel._exp_difference(np.array([-np.inf]), np.array([-np.inf])), [0.])

    def test_fast_path_against_all_logsumexp_reference(self):
        b, a, d = np.array([.21, -.18]), np.array([.4, .35]), np.array([.6, -.8])
        q = np.array([.13, .22, .27, .38])
        rules = (({'rule': 'hermite', 'order': 9}, {'rule': 'hermite', 'order': 7}),
                 ({'rule': 'legendre', 'order': 12, 'limit': 1.3}, {'rule': 'hermite', 'order': 7}))
        maxima = {}
        for specs in rules:
            for t in (.1, .0001):
                kwargs = dict(normal_specs=specs, logu_min=-32, logu_max=64, logu_step=.125)
                fast = kernel.angular_path_losses(b, a, d, t, q, (2, 11, 12, 'inf'), **kwargs)
                with patch.object(kernel, '_logpsi', kernel._logpsi_reference):
                    reference = kernel.angular_path_losses(b, a, d, t, q, (2, 11, 12, 'inf'), **kwargs)
                for key in ('label_loss', 'label_delta', 'absolute_loss', 'paired_delta', 'logPsi',
                            'paired_integrand', 'absolute_integrand', 'paired_risk_delta'):
                    error = float(np.max(abs(fast[key] - reference[key])))
                    maxima[key] = max(maxima.get(key, 0.), error)
                    self.assertLess(error, 3e-12)
        self.diagnostics['fast_vs_reference_maximum_errors'] = maxima

    def test_input_rejections(self):
        for q in ([.1] * 4, [.3, -.1, .4, .4]):
            with self.assertRaises(ValueError): kernel.prepare_baseline([0, 0], np.zeros((2, 2)), q)
        with self.assertRaises(ValueError): kernel.prepare_baseline([0, 0], np.zeros((2, 3)), np.ones(4) / 4)
        with self.assertRaises(ValueError): kernel.prepare_baseline([0, 0], np.zeros((2, 2)), np.ones(4) / 4, outer_chunk=0)
        with self.assertRaises(ValueError): kernel.normal_tensor([{'rule': 'hermite', 'order': 5}])
        with self.assertRaises(ValueError): kernel.normal_tensor([{'rule': 'legendre', 'order': 5, 'limit': -1}] * 2)
        with self.assertRaises(ValueError): kernel.path_W([1, 2], [1], .2)
        p = kernel.prepare_baseline([0, 0], np.zeros((2, 2)), np.ones(4) / 4, (2,), normal_specs=self.specs)
        with self.assertRaises(ValueError): kernel.evaluate_prepared(p, np.zeros((2, 3)))
        with self.assertRaises(ValueError): kernel.evaluate_prepared({}, np.zeros((2, 2)))


if __name__ == '__main__':
    before = np.random.get_state()
    tick = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NoncollinearKernelTests))
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    print(json.dumps(dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
                         seconds=time.perf_counter() - tick, manufactured_m2_m3_only=True,
                         study_teachers_or_old_endpoints_used=False, random_state_unchanged=unchanged,
                         diagnostics=NoncollinearKernelTests.diagnostics), sort_keys=True))
    sys.exit(0 if result.wasSuccessful() and unchanged else 1)
