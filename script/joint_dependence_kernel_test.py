"""Deterministic manufactured checks only; no scientific queue or RNG draws."""
from __future__ import annotations

import json
import math
import sys
import unittest

import numpy as np
from scipy.special import expit, logsumexp

import joint_dependence_kernel as kernel


def rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]], dtype=np.float64)


def fixture(m=3, bank_count=5, particles=7):
    b = np.linspace(-0.45, 0.55, m)
    W = (0.65 * np.sin(np.arange(2 * m).reshape(m, 2) + 0.3)
         + 0.2 * np.cos(0.7 * np.arange(2 * m).reshape(m, 2)))
    probabilities = np.arange(1, (1 << m) + 1, dtype=np.float64)
    probabilities /= probabilities.sum()
    index = np.arange(bank_count * particles * 2, dtype=np.float64)
    banks = (1.4 * np.sin(0.71 * index) + 0.3 * np.cos(0.13 * index)).reshape(bank_count, particles, 2)
    return b, W, probabilities, banks


def central_difference(function, b, W, step=1e-5):
    gradient_b, gradient_w = np.empty_like(b), np.empty_like(W)
    for index in range(len(b)):
        plus, minus = b.copy(), b.copy()
        plus[index] += step
        minus[index] -= step
        gradient_b[index] = (function(plus, W) - function(minus, W)) / (2 * step)
    for index in np.ndindex(W.shape):
        plus, minus = W.copy(), W.copy()
        plus[index] += step
        minus[index] -= step
        gradient_w[index] = (function(b, plus) - function(b, minus)) / (2 * step)
    return gradient_b, gradient_w


class JointDependenceKernelTests(unittest.TestCase):
    diagnostics = {}

    def test_patterns_lexicographic(self):
        expected = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.int8)
        np.testing.assert_array_equal(kernel.patterns(3), expected)

    def test_gh_normal_mass_and_moments(self):
        for order, dim, angle in ((2, 2, 0.0), (9, 2, 0.73), (8, 1, 0.0), (5, 3, 0.0)):
            nodes, logweights = kernel.gh_nodes(order, dim=dim, angle=angle)
            weights = np.exp(logweights)
            self.assertEqual(nodes.shape, (order ** dim, dim))
            self.assertLess(abs(weights.sum() - 1), 3e-14)
            np.testing.assert_allclose(weights @ nodes, np.zeros(dim), atol=3e-14, rtol=0)
            np.testing.assert_allclose(nodes.T @ (weights[:, None] * nodes), np.eye(dim), atol=3e-14, rtol=0)

    def test_gh_likelihood_and_gradient(self):
        b, W, q, _ = fixture()
        order, angle = 18, 0.37
        nodes, logweights = kernel.gh_nodes(order, angle=angle)
        sigma = expit(b + nodes @ W.T)
        # Independently form products in probability space on this moderate fixture.
        expected = np.array([np.exp(logweights) @ np.prod(np.where(y, sigma, 1 - sigma), axis=1)
                             for y in kernel.patterns(len(b))])
        observed = kernel.model_pmf(b, W, order, angle=angle)
        np.testing.assert_allclose(observed, expected, atol=3e-16, rtol=3e-14)
        self.assertLess(abs(observed.sum() - 1), 3e-14)
        loss, grad_b, grad_w = kernel.exact_loss_grad(b, W, q, order, angle=angle)
        self.assertAlmostEqual(loss, float(-q @ np.log(expected)), places=13)
        fd_b, fd_w = central_difference(lambda bb, ww: kernel.exact_loss_grad(bb, ww, q, order, angle=angle)[0], b, W)
        error = max(float(np.max(np.abs(grad_b - fd_b))), float(np.max(np.abs(grad_w - fd_w))))
        self.diagnostics['GH_gradient_max_finite_difference_error'] = error
        self.assertLess(error, 2e-8)

    def test_mc_gradient_fixed_banks(self):
        b, W, q, banks = fixture(m=4, bank_count=4, particles=6)
        _, grad_b, grad_w = kernel.mc_loss_grad(b, W, q, banks)
        fd_b, fd_w = central_difference(lambda bb, ww: kernel.mc_loss_grad(bb, ww, q, banks)[0], b, W)
        error = max(float(np.max(np.abs(grad_b - fd_b))), float(np.max(np.abs(grad_w - fd_w))))
        self.diagnostics['MC_gradient_max_finite_difference_error'] = error
        self.assertLess(error, 2e-8)

    def test_mc_k1_mean_log_identity(self):
        b, W, q, banks = fixture(bank_count=5, particles=1)
        logits = b + banks[:, 0] @ W.T
        expected = np.array([-sum(float(probability) * np.sum(np.where(y,
                              -np.logaddexp(0, -logits[bi]), -np.logaddexp(0, logits[bi])))
                              for y, probability in zip(kernel.patterns(len(b)), q))
                             for bi in range(len(banks))])
        loss, grad_b, grad_w = kernel.mc_loss_grad(b, W, q, banks)
        np.testing.assert_allclose(kernel.mc_bank_losses(b, W, q, banks), expected, atol=2e-15, rtol=0)
        self.assertAlmostEqual(loss, float(expected.mean()), places=14)
        residual = expit(logits) - q @ kernel.patterns(len(b))
        np.testing.assert_allclose(grad_b, residual.mean(axis=0), atol=2e-15, rtol=0)
        np.testing.assert_allclose(grad_w, residual.T @ banks[:, 0] / len(banks), atol=2e-15, rtol=0)

    def test_mc_bank_boundary_and_direct_score(self):
        b, W, q, _ = fixture()
        banks = np.array([[[-2., -1.], [-1., -2.]], [[0., .2], [.5, 1.]], [[2., 1.], [1., 2.]]])
        expected = []
        for bank in banks:
            logits = b + bank @ W.T
            loglik = np.array([np.sum(np.where(y, -np.logaddexp(0, -logits),
                                              -np.logaddexp(0, logits)), axis=1)
                              for y in kernel.patterns(len(b))])
            expected.append(float(-q @ (logsumexp(loglik, axis=1) - math.log(len(bank)))))
        observed = kernel.mc_bank_losses(b, W, q, banks)
        np.testing.assert_allclose(observed, expected, atol=2e-15, rtol=0)
        loss = kernel.mc_loss_grad(b, W, q, banks)[0]
        self.assertAlmostEqual(loss, float(observed.mean()), places=14)
        incorrectly_pooled_loss = kernel.mc_loss_grad(b, W, q, banks.reshape(1, -1, 2))[0]
        self.diagnostics['separate_bank_minus_pooled_loss'] = loss - incorrectly_pooled_loss
        self.assertGreater(loss - incorrectly_pooled_loss, 1e-4)

    def test_common_rotation_covariance(self):
        b, W, q, banks = fixture()
        R = rotation(0.91)
        loss, grad_b, grad_w = kernel.mc_loss_grad(b, W, q, banks)
        rotated = kernel.mc_loss_grad(b, W @ R.T, q, banks @ R.T)
        self.assertAlmostEqual(loss, rotated[0], places=14)
        np.testing.assert_allclose(grad_b, rotated[1], atol=3e-15, rtol=0)
        np.testing.assert_allclose(grad_w @ R.T, rotated[2], atol=3e-15, rtol=0)
        np.testing.assert_allclose(kernel.mc_bank_losses(b, W, q, banks),
                                   kernel.mc_bank_losses(b, W @ R.T, q, banks @ R.T), atol=3e-15, rtol=0)
        gh = kernel.exact_loss_grad(b, W, q, 16, angle=.23)
        gh_rotated = kernel.exact_loss_grad(b, W @ R.T, q, 16, angle=.23 + .91)
        self.assertAlmostEqual(gh[0], gh_rotated[0], places=14)
        np.testing.assert_allclose(gh[1], gh_rotated[1], atol=3e-15, rtol=0)
        np.testing.assert_allclose(gh[2] @ R.T, gh_rotated[2], atol=3e-15, rtol=0)

    def test_angular_chain_rule(self):
        b, W, q, banks = fixture()
        errors = []
        for evaluate in (lambda w: kernel.exact_loss_grad(b, w, q, 18),
                         lambda w: kernel.mc_loss_grad(b, w, q, banks)):
            expected = kernel.angular_grad(evaluate(W)[2], W)
            observed = np.empty(len(b))
            step = 1e-5
            for index in range(len(b)):
                plus, minus = W.copy(), W.copy()
                plus[index] = W[index] @ rotation(step).T
                minus[index] = W[index] @ rotation(-step).T
                observed[index] = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * step)
            errors.append(float(np.max(np.abs(observed - expected))))
        self.diagnostics['angular_gradient_max_finite_difference_error'] = max(errors)
        self.assertLess(max(errors), 2e-8)

    def test_chunks_zero_probabilities_and_no_input_mutation(self):
        b, W, q, banks = fixture(m=4)
        q[::2] = 0
        q /= q.sum()
        originals = [array.copy() for array in (b, W, q, banks)]
        reference = kernel.mc_loss_grad(b, W, q, banks, bank_chunk_size=1, pattern_chunk_size=1)
        for bc, pc in ((3, 4), (100, 100), (None, 32)):
            observed = kernel.mc_loss_grad(b, W, q, banks, bank_chunk_size=bc, pattern_chunk_size=pc)
            for actual, expected in zip(observed, reference):
                np.testing.assert_allclose(actual, expected, atol=4e-15, rtol=0)
        gh = kernel.exact_loss_grad(b, W, q, 12, pattern_chunk_size=1)
        gh_wide = kernel.exact_loss_grad(b, W, q, 12, pattern_chunk_size=64)
        for actual, expected in zip(gh_wide, gh):
            np.testing.assert_allclose(actual, expected, atol=4e-15, rtol=0)
        for actual, expected in zip((b, W, q, banks), originals):
            np.testing.assert_array_equal(actual, expected)

    def test_extreme_logits_remain_in_log_domain(self):
        b = np.array([-900., 900.])
        W = np.array([[.2, -.3], [-.1, .4]])
        q = np.ones(4) / 4
        banks = np.array([[[-1., .2], [.3, 1.]], [[2., -.5], [.2, -.4]]])
        for result in (kernel.exact_loss_grad(b, W, q, 12), kernel.mc_loss_grad(b, W, q, banks)):
            self.assertTrue(all(np.isfinite(value).all() for value in result))
        pmf = kernel.model_pmf(b, W, 12)
        self.assertTrue(np.any(pmf == 0))  # Documented probability-space underflow.
        self.assertLess(abs(pmf.sum() - 1), 3e-14)

    def test_invalid_inputs_are_not_silently_normalized(self):
        b, W, q, banks = fixture()
        with self.assertRaises(ValueError):
            kernel.patterns(0)
        with self.assertRaises(ValueError):
            kernel.gh_nodes(4, dim=3, angle=.2)
        with self.assertRaises(ValueError):
            kernel.exact_loss_grad(b, W, q / 2)
        with self.assertRaises(ValueError):
            kernel.mc_bank_losses(b, W, q, banks[:, :0])
        bad = banks.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            kernel.mc_loss_grad(b, W, q, bad)

    def test_global_legacy_rng_state_unchanged(self):
        before = np.random.get_state()
        b, W, q, banks = fixture()
        kernel.gh_nodes(6, angle=.4)
        kernel.model_pmf(b, W, 6)
        kernel.exact_loss_grad(b, W, q, 6)
        kernel.mc_loss_grad(b, W, q, banks)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(JointDependenceKernelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                          manufactured_only=True, new_random_draws=0,
                          diagnostics=JointDependenceKernelTests.diagnostics), sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
