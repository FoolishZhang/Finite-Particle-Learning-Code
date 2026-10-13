"""Small invented m2/m3 fixtures, independent of study teachers or endpoints."""
from __future__ import annotations
import itertools
import json
import math
import sys
import time
import unittest
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logsumexp
import rankone_stability_kernel as kernel


def direct_k2(b, a, target, order=32):
    x, w = hermegauss(order)
    w /= math.sqrt(2 * math.pi)
    ix = np.indices((order, order)).reshape(2, -1).T
    z = x[ix]
    weights = np.prod(w[ix], axis=1)
    Y = np.array(list(itertools.product((0, 1), repeat=len(b))))
    eta = b + z[:, :, None] * a
    sigmoid = expit(eta)
    M = np.zeros((len(b), len(b)))
    bias, radial, diag = np.zeros(len(b)), np.zeros(len(b)), np.zeros(len(b))
    collision, loss = 0., 0.
    for y, q in zip(Y, target):
        logA = np.where(y, -np.logaddexp(0, -eta), -np.logaddexp(0, eta)).sum(axis=2)
        norm = logsumexp(logA, axis=1)
        pi = np.exp(logA - norm[:, None])
        r = y - sigmoid
        C = pi * (1 - pi)
        M += q * np.einsum("b,bk,bki,bkj->ij", weights, C, r, r)
        bias -= q * np.einsum("b,bk,bki->i", weights, pi, r)
        radial -= q * np.einsum("b,bk,bki,bk->i", weights, pi, r, z)
        diag += q * np.einsum("b,bk,bki->i", weights, pi, sigmoid * (1 - sigmoid))
        collision += q * float(weights @ C.sum(axis=1))
        loss -= q * float(weights @ (norm - math.log(2)))
    edges = M * a[:, None] * a[None, :]
    np.fill_diagonal(edges, 0)
    L = np.diag(edges.sum(axis=1)) - edges
    return dict(M=M, laplacian=L, bias_gradient=bias, radial_gradient=radial, sigma_prime=diag,
                collision=collision, loss=loss)


def angle_loss(b, a, target, angles, order=12, K=2):
    x, w = hermegauss(order)
    w /= math.sqrt(2 * math.pi)
    ix = np.indices((order,) * (2 * K)).reshape(2 * K, -1).T
    z = x[ix].reshape(-1, K, 2)
    weights = np.prod(w[ix], axis=1)
    W = a[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    eta = b + z @ W.T
    answer = 0.
    for y, q in zip(itertools.product((0, 1), repeat=len(b)), target):
        logs = np.where(np.asarray(y), -np.logaddexp(0, -eta), -np.logaddexp(0, eta)).sum(axis=2)
        answer -= q * float(weights @ (logsumexp(logs, axis=1) - math.log(K)))
    return answer


class RankOneKernelTests(unittest.TestCase):
    diagnostics = {}

    def test_normal_rules(self):
        for rule, order, limit, tolerance in (("hermite", 32, 10., 1e-13), ("legendre", 128, 8., 1e-11)):
            z, w = kernel.normal_nodes(order, rule=rule, limit=limit)
            self.assertLess(abs(w.sum() - 1), tolerance)
            self.assertLess(abs(w @ z), tolerance)
            self.assertLess(abs(w @ (z * z) - 1), tolerance)
            self.assertLess(abs(w @ (z ** 4) - 3), 1e-10)
        _, truncated = kernel.normal_nodes(96, rule="legendre", limit=2.)
        self.assertLess(truncated.sum(), .96)  # Deliberately not renormalized.

    def test_k2_against_direct_two_dimensional_bank_integral(self):
        b, a = np.array([-.23, .37, .11]), np.array([.42, -.61, .28])
        q = np.arange(1., 9.) / 36
        start = time.perf_counter()
        result = kernel.rankone_stability(b, a, q, [2], gh_order=64, compute_loss=True)
        self.diagnostics['m3_K2_kernel_seconds'] = time.perf_counter() - start
        direct = direct_k2(b, a, q, order=40)
        errors = {"laplacian": float(np.max(np.abs(result['laplacian'][0, 0] - direct['laplacian']))),
                  "bias": float(np.max(np.abs(result['bias_gradient'][0, 0] - direct['bias_gradient']))),
                  "radial": float(np.max(np.abs(result['radial_gradient'][0, 0] - direct['radial_gradient']))),
                  "loss": abs(float(result['loss'][0, 0]) - direct['loss']),
                  "collision": abs(float(result['target_c_mass'][0, 0]) - direct['collision'])}
        self.diagnostics['K2_direct_errors'] = errors
        self.assertLess(max(errors.values()), 2e-8)
        self.assertLess(float(result['ward_residual_max'].max()), 2e-8)
        np.testing.assert_allclose(result['bias_gradient'][0, 0],
                                   q @ result['label_bias_gradient'][0], atol=2e-16, rtol=0)
        # Independently differentiate the direct two-bank score with respect
        # to each bias; this also checks the sign and coordinate convention.
        step = 1e-5
        bias_fd = np.empty(len(b))
        for j in range(len(b)):
            direction = np.eye(len(b))[j]
            bias_fd[j] = (direct_k2(b + step * direction, a, q, 40)['loss']
                          - direct_k2(b - step * direction, a, q, 40)['loss']) / (2 * step)
        bias_fd_error = float(np.max(np.abs(result['bias_gradient'][0, 0] - bias_fd)))
        self.diagnostics['bias_gradient_finite_difference_error'] = bias_fd_error
        self.assertLess(bias_fd_error, 2e-8)

    def test_integrated_self_target_zero_angular_hessian(self):
        # Invented collinear model; use its own same-rule PMF so there is no
        # discrepancy between two different normal quadratures in this identity.
        b, a = np.array([.16, -.24, .31]), np.array([.39, -.52, .28])
        baseline = kernel.rankone_stability(b, a, np.ones(8) / 8, [np.inf], gh_order=64)
        q = np.exp(baseline['log_evidence'])  # No target mass correction.
        result = kernel.rankone_stability(b, a, q, [np.inf], gh_order=64)
        error = float(np.max(np.abs(result['laplacian'])))
        radial_error = float(np.max(np.abs(result['radial_gradient'])))
        self.diagnostics['integrated_self_target_L_error'] = error
        self.diagnostics['integrated_self_target_radial_error'] = radial_error
        self.assertLess(error, 2e-16)
        self.assertLess(radial_error, 2e-16)
        self.assertLess(float(np.max(np.abs(result['bias_gradient']))), 2e-16)
        self.assertLess(float(result['ward_residual_max'].max()), 2e-16)

    def test_angle_curvature_and_common_rotation(self):
        b, a = np.array([-.19, .31]), np.array([.36, .57])
        q = np.array([.13, .21, .29, .37])
        result = kernel.rankone_stability(b, a, q, [1, 2], gh_order=64)
        direction = np.array([.7, -.4])
        step = 1e-3
        center = angle_loss(b, a, q, np.zeros(2), order=14)
        plus = angle_loss(b, a, q, step * direction, order=14)
        minus = angle_loss(b, a, q, -step * direction, order=14)
        finite_difference = (plus - 2 * center + minus) / step ** 2
        predicted = float(direction @ result['laplacian'][0, 1] @ direction)
        error = abs(finite_difference - predicted)
        self.diagnostics['angle_curvature_error'] = error
        self.assertLess(error, 2e-7)
        rotation_error = abs(angle_loss(b, a, q, np.full(2, .27), order=14) - center)
        self.diagnostics['common_rotation_loss_error'] = rotation_error
        self.assertLess(rotation_error, 2e-10)
        np.testing.assert_array_equal(result['laplacian'][0, 0], np.zeros((2, 2)))
        self.assertLess(float(result['common_angle_residual'].max()), 1e-15)
        self.assertLess(abs(angle_loss(b, a, q, np.array([.2, -.4]), order=24, K=1)
                            - angle_loss(b, a, q, np.zeros(2), order=24, K=1)), 2e-12)

    def test_multi_targets_infinite_limit_and_signed_edges(self):
        b, a = np.array([-.17, .26]), np.array([.35, .62])
        q0, q1 = np.ones(4) / 4, np.array([0., 1., 0., 0.])
        result = kernel.rankone_stability(b, a, {'uniform': q0, 'mixed_label': q1}, [1, 2, 7, np.inf], return_coefficients=True)
        self.assertEqual(result['laplacian'].shape, (2, 4, 2, 2))
        self.assertEqual(result['K_values'], (1, 2, 7, 'inf'))
        for index, q in enumerate((q0, q1)):
            single = kernel.rankone_stability(b, a, q, [1, 2, 7, np.inf])
            np.testing.assert_allclose(single['laplacian'][0], result['laplacian'][index], atol=2e-15, rtol=0)
        self.assertLess(result['edge_weights'][1, 1, 0, 1], 0)
        self.assertLess(result['eigenvalues'][1, 1, 0], 0)  # No positive-part correction.
        self.assertLess(float(np.max(np.abs(result['label_d_mass_error']))), 1e-8)
        self.assertTrue(np.all(result['label_c_mass'] >= 0))
        self.assertLess(float(np.max(result['label_c_mass_upper_excess'])), 1e-9)
        np.testing.assert_allclose(result['c_coefficients'][3], result['d_coefficients'][3], atol=0, rtol=0)
        self.assertLess(float(result['ward_residual_max'].max()), 2e-8)

    def test_constant_likelihood_collision_bound(self):
        result = kernel.rankone_stability([-.17, .22, .37], [0., 0., 0.], np.ones(8) / 8,
                                         [1, 2, 7, 32, np.inf], compute_loss=True)
        expected = np.array([0., .5, 6 / 7, 31 / 32, 1.])
        np.testing.assert_allclose(result['label_c_mass'], np.repeat(expected[:, None], 8, axis=1), atol=2e-12, rtol=0)
        np.testing.assert_array_equal(result['laplacian'], np.zeros((1, 5, 3, 3)))
        self.assertFalse(result['qualifications']['coefficient_masses_renormalized'])
        self.assertLess(float(np.max(np.abs(result['label_d_mass_error']))), 2e-9)

    def test_arbitrary_positive_scale_and_rule_comparison(self):
        b, a, q = np.array([.12, -.29]), np.array([.27, -.48]), np.array([.19, .23, .31, .27])
        base = kernel.rankone_stability(b, a, q, [2, 5, np.inf], compute_loss=True)
        scale = np.exp(base['log_evidence']) * np.array([.3, .7, 1.8, 3.1])
        changed = kernel.rankone_stability(b, a, q, [2, 5, np.inf], likelihood_scale=scale, compute_loss=True)
        for name in ('laplacian', 'bias_gradient', 'radial_gradient', 'loss'):
            np.testing.assert_allclose(base[name], changed[name], atol=2e-8, rtol=0)
        legendre = kernel.rankone_stability(b, a, q, [2], normal_rule='legendre', normal_order=128,
                                            normal_limit=8., outer_rule='legendre', outer_order=512, compute_loss=True)
        fine = kernel.rankone_stability(b, a, q, [2], gh_order=80, logu_step=.125, compute_loss=True)
        for name in ('laplacian', 'bias_gradient', 'radial_gradient', 'loss'):
            np.testing.assert_allclose(legendre[name], fine[name], atol=2e-8, rtol=0)
        self.diagnostics['independent_rule_Ward_residual'] = float(legendre['ward_residual_max'].max())

    def test_bad_normalization_is_reported_not_repaired(self):
        result = kernel.rankone_stability([.1, -.2], [.3, .4], np.ones(4) / 4, [2],
                                         logu_min=-2, logu_max=1, logu_step=.25)
        self.assertGreater(float(np.max(np.abs(result['label_d_mass_error']))), .01)
        self.assertFalse(result['qualifications']['signs_or_eigenvalues_clipped'])
        with self.assertRaises(ValueError):
            kernel.rankone_stability([0, 0], [1, 1], [.1, .1, .1, .1], [2])
        with self.assertRaises(ValueError):
            kernel.rankone_stability([0, 0], [1, 1], np.ones(4) / 4, [2], likelihood_scale=[0, 1, 1, 1])


if __name__ == '__main__':
    before = np.random.get_state()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RankOneKernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = np.random.get_state()
    rng_unchanged = before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    print(json.dumps(dict(passed=result.wasSuccessful() and rng_unchanged, tests_run=result.testsRun,
          seconds=time.perf_counter() - started, manufactured_m2_m3_only=True,
          study_teachers_or_old_endpoints_used=False, random_state_unchanged=rng_unchanged,
          diagnostics=RankOneKernelTests.diagnostics), sort_keys=True))
    sys.exit(0 if result.wasSuccessful() and rng_unchanged else 1)
