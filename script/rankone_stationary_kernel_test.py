"""Invented m2/m3 fixtures only; no study teachers, endpoints or data files."""
from __future__ import annotations
import itertools
import json
import math
import sys
import time
import unittest
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, logsumexp
import rankone_stationary_kernel as kernel


def independent_normal(order, rule="hermite", limit=10.):
    if rule == "hermite":
        z, w = hermegauss(order)
        return z, w / math.sqrt(2 * math.pi)
    z, w = leggauss(order)
    z = limit * z
    return z, limit * w * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def direct_bank(b, a, q, K=2, order=40, rule="hermite", limit=10.):
    """Independent tensor bank expectation, from the log-sum Hessian formula."""
    x, w = independent_normal(order, rule, limit)
    index = np.indices((order,) * K).reshape(K, -1).T
    z, weight = x[index], np.prod(w[index], axis=1)
    m, size = len(b), len(index)
    logits = b + z[:, :, None] * a
    sigma, sp = expit(logits), expit(logits) * expit(-logits)
    ans = dict(loss=0., gradient=np.zeros(2 * m), H_ba=np.zeros((2 * m, 2 * m)),
               Hperp=np.zeros((m, m)), Hangle=np.zeros((m, m)))
    ix = np.arange(m)
    for y, prob in zip(itertools.product((0, 1), repeat=m), q):
        logs = np.where(np.array(y), -np.logaddexp(0, -logits), -np.logaddexp(0, logits)).sum(axis=-1)
        norm = logsumexp(logs, axis=1)
        pi = np.exp(logs - norm[:, None])
        r = np.array(y) - sigma
        G = np.concatenate((r, z[:, :, None] * r), axis=2)
        meanG = np.einsum('nk,nkd->nd', pi, G)
        B = np.zeros((size, 2 * m, 2 * m))
        B[:, ix, ix] = np.einsum('nk,nkj->nj', pi, sp)
        B[:, ix, m + ix] = B[:, m + ix, ix] = np.einsum('nk,nk,nkj->nj', pi, z, sp)
        B[:, m + ix, m + ix] = np.einsum('nk,nk,nkj->nj', pi, z * z, sp)
        H = B - np.einsum('nk,nki,nkj->nij', pi, G, G) + np.einsum('ni,nj->nij', meanG, meanG)
        M = np.einsum('n,nk,nki,nkj->ij', weight, pi * (1 - pi), r, r)
        hperp = np.diag(np.einsum('n,nk,nkj->j', weight, pi, sp)) - M
        edge = M * a[:, None] * a
        np.fill_diagonal(edge, 0)
        ans['loss'] -= prob * float(weight @ (norm - math.log(K)))
        ans['gradient'] -= prob * (weight @ meanG)
        ans['H_ba'] += prob * np.einsum('n,nij->ij', weight, H)
        ans['Hperp'] += prob * hperp
        ans['Hangle'] += prob * (np.diag(edge.sum(axis=1)) - edge)
    return ans


def direct_integrated(b, a, q, order=64):
    z, w = independent_normal(order)
    m = len(b)
    logits = b + z[:, None] * a
    sigma, sp = expit(logits), expit(logits) * expit(-logits)
    ans = dict(loss=0., gradient=np.zeros(2 * m), H_ba=np.zeros((2 * m, 2 * m)))
    ix = np.arange(m)
    for y, prob in zip(itertools.product((0, 1), repeat=m), q):
        logs = np.where(np.array(y), -np.logaddexp(0, -logits), -np.logaddexp(0, logits)).sum(axis=1)
        logp = logsumexp(np.log(w) + logs)
        posterior = np.exp(np.log(w) + logs - logp)
        r = np.array(y) - sigma
        G = np.concatenate((r, z[:, None] * r), axis=1)
        B = np.zeros((2 * m, 2 * m))
        B[ix, ix] = posterior @ sp
        B[ix, m + ix] = B[m + ix, ix] = (posterior * z) @ sp
        B[m + ix, m + ix] = (posterior * z * z) @ sp
        mean = posterior @ G
        ans['loss'] -= prob * logp
        ans['gradient'] -= prob * mean
        ans['H_ba'] += prob * (B - G.T @ (posterior[:, None] * G) + np.outer(mean, mean))
    return ans


class StationaryKernelTests(unittest.TestCase):
    diagnostics = {}

    def test_independent_two_particle_full_derivatives(self):
        b, a = np.array([-.23, .37, .11]), np.array([.42, -.61, .28])
        q = np.arange(1., 9.) / 36
        start = time.perf_counter()
        result = kernel.rankone_stationary(b, a, q, [2], normal_order=64)
        self.diagnostics['m3_K2_full_seconds'] = time.perf_counter() - start
        expected = direct_bank(b, a, q, order=40)
        errors = {key: float(np.max(np.abs(result[key][0, 0] - expected[key])))
                  for key in ('loss', 'gradient', 'H_ba', 'Hperp', 'Hangle')}
        self.diagnostics['K2_direct_errors'] = errors
        self.assertLess(max(errors.values()), 3e-8)
        self.assertLess(float(result['ward_residual_max'].max()), 3e-8)
        np.testing.assert_allclose(result['label_H_ba'], result['label_B'] - result['label_same_particle_term']
                                   + result['label_cross_particle_term'], atol=0, rtol=0)
        for key in ('H_ba', 'Hperp', 'Hangle', 'directWard'):
            np.testing.assert_allclose(result[key][0, 0], np.einsum('p,pij->ij', q, result['label_' + key][0]),
                                       atol=2e-16, rtol=0)

    def test_hessian_is_full_gradient_difference(self):
        b, a, q = np.array([.14, -.27]), np.array([.38, -.49]), np.array([.13, .21, .29, .37])
        center = kernel.rankone_stationary(b, a, q, [2], normal_order=64)
        theta = np.r_[b, a]
        numerical = np.empty((4, 4))
        loss_gradient = np.empty(4)
        step = 1e-5
        for j in range(4):
            direction = np.eye(4)[j] * step
            plus, minus = theta + direction, theta - direction
            high = kernel.rankone_stationary(plus[:2], plus[2:], q, [2], compute_hessian=False)
            low = kernel.rankone_stationary(minus[:2], minus[2:], q, [2], compute_hessian=False)
            numerical[:, j] = (high['gradient'][0, 0] - low['gradient'][0, 0]) / (2 * step)
            loss_gradient[j] = (high['loss'][0, 0] - low['loss'][0, 0]) / (2 * step)
        H_error = float(np.max(np.abs(numerical - center['H_ba'][0, 0])))
        g_error = float(np.max(np.abs(loss_gradient - center['gradient'][0, 0])))
        self.diagnostics['Hessian_gradient_FD_error'] = H_error
        self.diagnostics['gradient_loss_FD_error'] = g_error
        self.assertLess(H_error, 2e-8)
        self.assertLess(g_error, 2e-8)

    def test_integrated_posterior_and_self_target(self):
        b, a, q = np.array([.12, -.31, .2]), np.array([.33, -.46, .28]), np.arange(1., 9.) / 36
        result = kernel.rankone_stationary(b, a, q, [np.inf])
        direct = direct_integrated(b, a, q)
        for key in ('loss', 'gradient', 'H_ba'):
            np.testing.assert_allclose(result[key][0, 0], direct[key], atol=3e-15, rtol=0)
        self_q = np.exp(result['log_evidence'])
        exact = kernel.rankone_stationary(b, a, self_q, [np.inf])
        self.assertLess(float(np.max(np.abs(exact['gradient']))), 3e-16)
        self.assertLess(float(np.max(np.abs(exact['Hangle']))), 3e-16)
        scores = exact['label_gradient'][0]
        fisher = scores.T @ (self_q[:, None] * scores)
        np.testing.assert_allclose(exact['H_ba'][0, 0], fisher, atol=3e-16, rtol=0)
        self.assertGreater(np.linalg.eigvalsh(exact['H_ba'][0, 0]).min(), -2e-15)

    def test_K1_single_particle_convexity(self):
        b, a, q = np.array([-.17, .28]), np.array([.4, -.3]), np.array([.21, .28, .17, .34])
        result = kernel.rankone_stationary(b, a, {'first': q, 'uniform': np.ones(4) / 4}, [1])
        direct = direct_bank(b, a, q, K=1, order=64)
        for key in ('loss', 'gradient', 'H_ba'):
            np.testing.assert_allclose(result[key][0, 0], direct[key], atol=3e-15, rtol=0)
        np.testing.assert_allclose(result['H_ba'][0], result['H_ba'][1], atol=2e-16, rtol=0)
        np.testing.assert_array_equal(result['Hangle'], np.zeros((2, 1, 2, 2)))
        self.assertGreater(np.linalg.eigvalsh(result['H_ba'][0, 0]).min(), 0)

    def test_unnormalized_normal_mass_matches_raw_bank_measure(self):
        b, a, q = np.array([.19, -.26]), np.array([.35, -.42]), np.array([.21, .16, .34, .29])
        result = kernel.rankone_stationary(b, a, q, [1, 2], normal_rule='legendre', normal_order=48,
                                            normal_limit=1.7, logu_min=-28, logu_max=56, logu_step=.125)
        self.assertLess(result['normal_quadrature_mass'], .92)
        errors = {}
        for ki, K in enumerate((1, 2)):
            direct = direct_bank(b, a, q, K, order=48, rule='legendre', limit=1.7)
            for key in ('loss', 'gradient', 'H_ba'):
                errors[f'K{K}_{key}'] = float(np.max(np.abs(result[key][0, ki] - direct[key])))
        self.diagnostics['truncated_normal_raw_measure_errors'] = errors
        self.assertLess(max(errors.values()), 1e-11)
        self.assertLess(float(np.max(np.abs(result['label_d_mass_quadrature_error']))), 2e-12)
        self.assertGreater(float(np.max(np.abs(result['label_d_mass_error']))), .1)
        np.testing.assert_allclose(result['raw_c_mass_upper'], [0., .5 * result['normal_quadrature_mass'] ** 2],
                                   atol=2e-16, rtol=0)

    def test_rules_scale_and_multi_target(self):
        b, a = np.array([.13, -.28]), np.array([.31, -.43])
        q = np.array([[.2, .25, .31, .24], [.34, .11, .29, .26]])
        base = kernel.rankone_stationary(b, a, q, [2, 5, np.inf])
        scale = np.exp(base['log_evidence']) * np.array([.3, .7, 1.8, 3.1])
        other = kernel.rankone_stationary(b, a, q, [2, 5, np.inf], likelihood_scale=scale)
        legendre = kernel.rankone_stationary(b, a, q, [2, 5, np.inf], normal_rule='legendre',
                                              normal_order=128, normal_limit=8, logu_step=.125)
        for key in ('loss', 'gradient', 'H_ba', 'Hangle', 'Hperp'):
            np.testing.assert_allclose(base[key], other[key], atol=2e-8, rtol=0)
            np.testing.assert_allclose(base[key], legendre[key], atol=2e-8, rtol=0)
        single = kernel.rankone_stationary(b, a, q[1], [2, 5, np.inf])
        np.testing.assert_allclose(base['H_ba'][1], single['H_ba'][0], atol=2e-16, rtol=0)
        self.assertLess(float(base['H_ba_symmetry_residual'].max()), 2e-16)

    def test_lossgrad_mode_and_snapshot_ownership(self):
        b, a, q = np.array([.17, -.26]), np.array([.29, -.38]), np.array([.21, .16, .34, .29])
        originals = [t.copy() for t in (b, a, q)]
        full = kernel.rankone_stationary(b, a, q, [1, 2, np.inf], return_coefficients=True)
        start = time.perf_counter()
        fast = kernel.rankone_stationary(b, a, q, [1, 2, np.inf], compute_hessian=False, return_coefficients=True)
        self.diagnostics['m2_three_K_lossgrad_seconds'] = time.perf_counter() - start
        for key in ('loss', 'gradient', 'label_d_mass'):
            np.testing.assert_array_equal(full[key], fast[key])
        for key in ('H_ba', 'Hperp', 'Hangle', 'directWard', 'label_c_mass', 'c_coefficients'):
            self.assertIsNone(fast[key])
        self.assertEqual(fast['work_counts']['finite_c_coefficients'], 0)
        for current, original in zip((b, a, q), originals):
            np.testing.assert_array_equal(current, original)
        def arrays(value):
            if isinstance(value, np.ndarray):
                return [value]
            if isinstance(value, dict):
                return [x for v in value.values() for x in arrays(v)]
            return []
        saved = arrays(full)
        for j, array in enumerate(saved):
            for other in saved[j + 1:] + [b, a, q]:
                self.assertFalse(np.shares_memory(array, other))
        old_bias = full['bias_gradient'].copy()
        full['gradient'][...] = 999
        np.testing.assert_array_equal(full['bias_gradient'], old_bias)
        b[...] = 999
        np.testing.assert_array_equal(full['b'], originals[0])

    def test_invalid_inputs_and_visible_truncation(self):
        for targets in ([.1, .1, .1, .1], [.6, -.1, .2, .3]):
            with self.assertRaises(ValueError):
                kernel.rankone_stationary([0., 0.], [.1, .2], targets, [2])
        with self.assertRaises(ValueError):
            kernel.rankone_stationary([0., 0.], [.1, .2], np.ones(4) / 4, [2], likelihood_scale=0)
        result = kernel.rankone_stationary([.1, -.2], [.3, -.4], np.ones(4) / 4, [2],
                                            logu_min=-2, logu_max=1)
        self.assertGreater(float(np.max(np.abs(result['label_d_mass_quadrature_error']))), .01)
        self.assertFalse(result['qualifications']['coefficient_masses_renormalized'])
        self.assertFalse(result['qualifications']['signs_or_eigenvalues_clipped'])


if __name__ == '__main__':
    before = np.random.get_state()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StationaryKernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = np.random.get_state()
    unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    print(json.dumps(dict(passed=result.wasSuccessful() and unchanged, tests_run=result.testsRun,
                         seconds=time.perf_counter() - started, manufactured_m2_m3_only=True,
                         study_teachers_or_old_endpoints_used=False, random_state_unchanged=unchanged,
                         diagnostics=StationaryKernelTests.diagnostics), sort_keys=True))
    sys.exit(0 if result.wasSuccessful() and unchanged else 1)
