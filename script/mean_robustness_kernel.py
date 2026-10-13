"""Deterministic full Price derivatives for a known shared mean at C = I.

This module draws no random numbers and writes no files.  A returned bank
kernel is an integrand: its Gaussian-bank expectation is the population
derivative.  Teacher labels are averaged explicitly using the supplied
product probabilities, rather than replaced by an all-positive label.
The nuisance conditions are (n0, n256), and covariance pairs are (01, 02, 12).
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp


PAIRS = ((0, 1), (0, 2), (1, 2))
CONDITIONS = ("n0", "n256")
LABELS = tuple(itertools.product((-1, 1), repeat=3))
GRADIENT_ABS_BOUND = 1.0
HESSIAN_DIAGONAL_LOWER = -33.0 / 8.0
HESSIAN_DIAGONAL_UPPER = 67.0 / 16.0
HESSIAN_OFFDIAGONAL_ABS_BOUND = 7.0
QUALIFICATION_TOLERANCE = 1e-10


def _validated_inputs(
    z: Any, s: Any, mu: Any, p: Any, *, batched: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(z, dtype=np.float64)
    nuisance = np.asarray(s, dtype=np.float64)
    means = np.asarray(mu, dtype=np.float64)
    probabilities = np.asarray(p, dtype=np.float64)
    expected_ndim = 3 if batched else 2
    if active.ndim != expected_ndim or active.shape[-1] != 3:
        shape = "(B, K, 3)" if batched else "(K, 3)"
        raise ValueError(f"z must have shape {shape}; got {active.shape}")
    if any(size < 1 for size in active.shape[:-1]):
        raise ValueError("z must contain at least one bank and one particle")
    if nuisance.shape != active.shape[:-1]:
        raise ValueError(f"s must have shape {active.shape[:-1]}; got {nuisance.shape}")
    if means.ndim == 0:
        means = np.full(3, float(means), dtype=np.float64)
    if means.shape != (3,):
        raise ValueError(f"mu must be a scalar or have shape (3,); got {means.shape}")
    if probabilities.shape != (3,):
        raise ValueError(f"p must have shape (3,); got {probabilities.shape}")
    if not all(np.all(np.isfinite(value)) for value in (active, nuisance, means, probabilities)):
        raise ValueError("z, s, mu, and p must contain finite float64 values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("p must contain probabilities in [0, 1]")
    return active, nuisance, means, probabilities


def _label_derivatives(
    coordinates: np.ndarray, nuisance: np.ndarray, labels: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed-label gradient and symmetric Hessian bank kernels.

    For L_ab = sum_k d_(k,a)d_(k,b), derivatives of -log(T), T=sum_k t_k,
    give G_ab = -sum pi(1-pi)q_a q_b.  For distinct pairs (a,b),(a,c),
    the same-particle fourth derivative is q_b q_c times

      -pi(1-pi)(1-2pi)r_a + pi^2(4-10pi+6pi^2)q_a^2.

    Summing different-particle derivatives produces
    A_ab A_ac - 2(B_ab A_ac + A_ab B_ac) + 6B_ab B_ac.  This sum includes
    the same-particle term pi^2(1-4pi+6pi^2)q_a^2 q_b q_c; subtracting it
    leaves the local coefficient 3pi^2(1-2pi) below.  Here q is the signed
    first likelihood ratio and r is the second likelihood ratio.
    """
    signed = coordinates * np.asarray(labels, dtype=np.float64)
    with np.errstate(over="ignore"):
        log_active = -np.logaddexp(0.0, -signed).sum(axis=-1)
    if not np.all(np.isfinite(log_active)):
        raise ValueError("active log likelihood exceeds the float64 range")
    unsigned_q = expit(-signed)
    q = unsigned_q * np.asarray(labels, dtype=np.float64)
    r = unsigned_q * (2.0 * unsigned_q - 1.0)
    # Subtract before adding so arbitrarily large constant nuisance offsets
    # cannot erase differences between active log likelihoods.
    with np.errstate(over="ignore"):
        shifted_nuisance = nuisance - np.max(nuisance, axis=1, keepdims=True)
    batch_size = coordinates.shape[0]
    gradient = np.empty((batch_size, 2, 3), dtype=np.float64)
    hessian = np.empty((batch_size, 2, 3, 3), dtype=np.float64)
    with np.errstate(over="ignore"):
        nuisance_log_weights = log_active + shifted_nuisance
    for condition, log_weights in enumerate((log_active, nuisance_log_weights)):
        pi = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        pi2 = pi * pi
        a = np.stack([q[..., first] * q[..., second] for first, second in PAIRS], axis=-1)
        A = np.sum(pi[..., None] * a, axis=1)
        B = np.sum(pi2[..., None] * a, axis=1)
        gradient[:, condition] = -np.sum((pi * (1.0 - pi))[..., None] * a, axis=1)
        for pair_index, (first, second) in enumerate(PAIRS):
            q1, q2 = q[..., first], q[..., second]
            r1, r2 = r[..., first], r[..., second]
            pair_a = a[..., pair_index]
            local = (
                -pi * (1.0 - pi) * r1 * r2
                + 2.0 * pi2 * (1.0 - pi) * (q1 * q1 * r2 + q2 * q2 * r1)
                + pi2 * (1.0 - 4.0 * pi) * pair_a * pair_a
            )
            pair_A, pair_B = A[:, pair_index], B[:, pair_index]
            hessian[:, condition, pair_index, pair_index] = (
                np.sum(local, axis=1) + pair_A * pair_A - 4.0 * pair_A * pair_B
                + 6.0 * pair_B * pair_B
            )
        for first_pair in range(3):
            for second_pair in range(first_pair + 1, 3):
                pair_p, pair_q = PAIRS[first_pair], PAIRS[second_pair]
                shared = next(index for index in pair_p if index in pair_q)
                other_p = next(index for index in pair_p if index != shared)
                other_q = next(index for index in pair_q if index != shared)
                local = q[..., other_p] * q[..., other_q] * (
                    -pi * (1.0 - pi) * (1.0 - 2.0 * pi) * r[..., shared]
                    + 3.0 * pi2 * (1.0 - 2.0 * pi) * q[..., shared] ** 2
                )
                value = (
                    np.sum(local, axis=1) + A[:, first_pair] * A[:, second_pair]
                    - 2.0 * (B[:, first_pair] * A[:, second_pair] + A[:, first_pair] * B[:, second_pair])
                    + 6.0 * B[:, first_pair] * B[:, second_pair]
                )
                hessian[:, condition, first_pair, second_pair] = value
                hessian[:, condition, second_pair, first_pair] = value
    return gradient, hessian


def mean_derivatives(z: Any, s: Any, mu: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return G with shape (B,2,3) and H with shape (B,2,3,3).

    ``z`` has shape (B,K,3), ``s`` has shape (B,K), and ``mu`` is a scalar
    or a length-three known mean shared by teacher and learner. ``p`` is
    the length-three vector of teacher probabilities P(Y_i=+1), supplied
    by the caller; this function does not integrate or estimate it.  The
    derivatives are with respect to off-diagonal covariance entries, with
    each entry mirrored in the symmetric covariance matrix.  The n0 arm
    sets nuisance log weights to a constant, and n256 uses ``s``.

    Bounds, preserved by the convex label average, are |G_i| <= 1,
    -33/8 <= H_ii <= 67/16, and |H_ij| <= 7 for i != j.  At zero mean
    with p_i=1/2, this label average need not equal an all-positive kernel
    on a particular bank; the Gaussian expectations have the symmetry.

    The off-diagonal local term is bounded by sum(pi + 3pi^2) <= 4.
    Within a fixed label, each a_p has a constant sign across particles.
    Thus B_p/A_p and B_q/A_q lie in [0,1] when nonzero, and the global
    term is A_p A_q[1-2(t+u)+6tu], whose absolute value is at most 3.
    """
    active, nuisance, means, probabilities = _validated_inputs(z, s, mu, p, batched=True)
    with np.errstate(over="ignore"):
        coordinates = active + means
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("z + mu exceeds the float64 range")
    gradient = np.zeros((active.shape[0], 2, 3), dtype=np.float64)
    hessian = np.zeros((active.shape[0], 2, 3, 3), dtype=np.float64)
    for labels in LABELS:
        label_weight = float(np.prod(np.where(np.asarray(labels) > 0, probabilities, 1.0 - probabilities)))
        if label_weight == 0.0:
            continue
        label_gradient, label_hessian = _label_derivatives(coordinates, nuisance, labels)
        gradient += label_weight * label_gradient
        hessian += label_weight * label_hessian
    return gradient, hessian


def interval_derivatives(
    z_one: Any, s_one: Any, mu: Any, p: Any, bits: int = 128
) -> tuple[list[list[Any]], list[list[list[Any]]]]:
    """Enclose one bank's G[2][3] and H[2][3][3] in Arb arithmetic.

    All input binary64 values, including mu and p, are treated as exact;
    the addition z + mu and the product teacher label weights are then
    formed in Arb.  The caller remains responsible for uncertainty in p.
    Only this function loads the private python-flint installation, and
    the previous Arb precision is restored even if evaluation fails.
    """
    active, nuisance, means, probabilities = _validated_inputs(z_one, s_one, mu, p, batched=False)
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or bits < 64:
        raise ValueError("bits must be an integer of at least 64")
    private_tools = Path(__file__).resolve().parents[1] / ".cache" / "finite_selection_tools"
    if str(private_tools) not in sys.path:
        sys.path.insert(0, str(private_tools))
    from flint import arb, ctx

    previous_precision = ctx.prec
    ctx.prec = int(bits)
    try:
        coordinates = [[arb(float(value)) + arb(float(mean)) for value, mean in zip(row, means)] for row in active]
        teacher_p = [arb(float(value)) for value in probabilities]
        shift = arb(float(np.max(nuisance)))
        nuisance_weights = [(arb(float(value)) - shift).exp() for value in nuisance]
        gradient = [[arb(0) for _ in PAIRS] for _ in CONDITIONS]
        hessian = [[[arb(0) for _ in PAIRS] for _ in PAIRS] for _ in CONDITIONS]
        for labels in LABELS:
            label_weight = arb(1)
            for label, probability in zip(labels, teacher_p):
                label_weight *= probability if label > 0 else 1 - probability
            signed = [[label * value for label, value in zip(labels, row)] for row in coordinates]
            sigmas = [[1 / (1 + (-value).exp()) for value in row] for row in signed]
            unsigned_q = [[1 / (1 + value.exp()) for value in row] for row in signed]
            q = [[label * value for label, value in zip(labels, row)] for row in unsigned_q]
            r = [[value * (2 * value - 1) for value in row] for row in unsigned_q]
            likelihood = [row[0] * row[1] * row[2] for row in sigmas]
            arms = (likelihood, [weight * value for weight, value in zip(nuisance_weights, likelihood)])
            for condition, weights in enumerate(arms):
                denominator = sum(weights, arb(0))
                pi = [weight / denominator for weight in weights]
                a = [[row[first] * row[second] for row in q] for first, second in PAIRS]
                A = [sum((weight * value for weight, value in zip(pi, pair_a)), arb(0)) for pair_a in a]
                B = [sum((weight * weight * value for weight, value in zip(pi, pair_a)), arb(0)) for pair_a in a]
                for pair_index, (first, second) in enumerate(PAIRS):
                    gradient[condition][pair_index] += label_weight * -sum(
                        (weight * (1 - weight) * value for weight, value in zip(pi, a[pair_index])), arb(0)
                    )
                    local = arb(0)
                    for weight, values, second_ratios, value in zip(pi, q, r, a[pair_index]):
                        q1, q2 = values[first], values[second]
                        r1, r2 = second_ratios[first], second_ratios[second]
                        local += (
                            -weight * (1 - weight) * r1 * r2
                            + 2 * weight * weight * (1 - weight) * (q1 * q1 * r2 + q2 * q2 * r1)
                            + weight * weight * (1 - 4 * weight) * value * value
                        )
                    pair_A, pair_B = A[pair_index], B[pair_index]
                    hessian[condition][pair_index][pair_index] += label_weight * (
                        local + pair_A * pair_A - 4 * pair_A * pair_B + 6 * pair_B * pair_B
                    )
                for first_pair in range(3):
                    for second_pair in range(first_pair + 1, 3):
                        pair_p, pair_q = PAIRS[first_pair], PAIRS[second_pair]
                        shared = next(index for index in pair_p if index in pair_q)
                        other_p = next(index for index in pair_p if index != shared)
                        other_q = next(index for index in pair_q if index != shared)
                        local = sum((
                            values[other_p] * values[other_q] * (
                                -weight * (1 - weight) * (1 - 2 * weight) * second_ratios[shared]
                                + 3 * weight * weight * (1 - 2 * weight) * values[shared] * values[shared]
                            )
                            for weight, values, second_ratios in zip(pi, q, r)
                        ), arb(0))
                        value = label_weight * (
                            local + A[first_pair] * A[second_pair]
                            - 2 * (B[first_pair] * A[second_pair] + A[first_pair] * B[second_pair])
                            + 6 * B[first_pair] * B[second_pair]
                        )
                        hessian[condition][first_pair][second_pair] += value
                        hessian[condition][second_pair][first_pair] += value
        return gradient, hessian
    finally:
        ctx.prec = previous_precision


def qualification() -> dict[str, Any]:
    """Check manufactured K1/2/3 fixtures against independent Price autograd.

    PyTorch differentiates the eight-label weighted -logsumexp objective
    directly with L_p and L_q L_p; it does not use the closed-form kernels.
    Includes all Hessian entries, fixed-label and averaged range checks,
    K1 vanishing, constant nuisance identity, permutations, and batching.
    There are no Gaussian calculations or RNG calls.
    """
    import torch
    import torch.nn.functional as torch_functional

    fixtures = (
        ("K1_zero", [[0.0, 0.0, 0.0]], [0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
        ("K1_asymmetric", [[2.0, -3.0, 1.0]], [7.0], [0.5, -0.75, 0.25], [0.2, 0.7, 0.9]),
        ("K2_zero_mean", [[0.1, 0.8, -0.5], [-0.7, 0.2, 1.5]], [0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
        ("K2_shifted", [[0.1, 0.8, -0.5], [-0.7, 0.2, 1.5]], [0.0, -9.0], [0.5, 0.5, 0.5], [0.3, 0.6, 0.8]),
        (
            "K3_asymmetric", [[1.2, -0.3, 0.7], [-0.7, 0.8, -1.1], [0.1, 1.5, -0.2]],
            [0.0, -7.0, -0.5], [-0.4, 0.8, 0.2], [0.15, 0.65, 0.85],
        ),
        (
            "K3_extreme_endpoint_p", [[-10.0, 8.0, 0.1], [10.0, -8.0, -0.1], [0.0, 0.0, 0.0]],
            [2.0, -2.0, 0.0], [0.125, -0.375, 0.75], [0.0, 1.0, 0.4],
        ),
    )
    tolerance = QUALIFICATION_TOLERANCE
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def direct_autograd(active: np.ndarray, nuisance: np.ndarray, means: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = torch.tensor(active, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor(LABELS, dtype=torch.float64)
        teacher_p = torch.tensor(probabilities, dtype=torch.float64)
        teacher_weights = torch.prod(torch.where(labels > 0, teacher_p, 1.0 - teacher_p), dim=1)
        signed = labels[:, None, :] * (coordinates[None, :, :] + torch.tensor(means, dtype=torch.float64))
        log_weights = torch_functional.logsigmoid(signed).sum(dim=2) + torch.tensor(nuisance, dtype=torch.float64)
        objective = torch.sum(teacher_weights * -torch.logsumexp(log_weights, dim=1))

        def price_operator(value: Any, pair: tuple[int, int]) -> Any:
            first = torch.autograd.grad(value, coordinates, create_graph=True, retain_graph=True)[0]
            terms = []
            for particle in range(len(active)):
                second = torch.autograd.grad(first[particle, pair[0]], coordinates, create_graph=True, retain_graph=True)[0]
                terms.append(second[particle, pair[1]])
            return sum(terms)

        price_gradient = [price_operator(objective, pair) for pair in PAIRS]
        price_hessian = [[price_operator(value, pair) for pair in PAIRS] for value in price_gradient]
        return (
            np.asarray([float(value.detach()) for value in price_gradient]),
            np.asarray([[float(value.detach()) for value in row] for row in price_hessian]),
        )

    def max_discrepancy(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
        return max(float(np.max(np.abs(a - b))) for a, b in zip(left, right))

    def check_ranges(gradient: np.ndarray, hessian: np.ndarray, name: str) -> None:
        diagonal = np.diagonal(hessian, axis1=-2, axis2=-1)
        offdiagonal = hessian[..., ~np.eye(3, dtype=bool)]
        assert np.all(np.abs(gradient) <= GRADIENT_ABS_BOUND), (name, "gradient_range")
        assert np.all(diagonal >= HESSIAN_DIAGONAL_LOWER), (name, "diagonal_lower")
        assert np.all(diagonal <= HESSIAN_DIAGONAL_UPPER), (name, "diagonal_upper")
        assert np.all(np.abs(offdiagonal) <= HESSIAN_OFFDIAGONAL_ABS_BOUND), (name, "offdiagonal_range")

    details = []
    maxima = {key: 0.0 for key in ("gradient_autograd", "hessian_autograd", "constant_nuisance", "permutation", "batch", "k1_zero")}
    try:
        for name, z_values, s_values, mu_values, p_values in fixtures:
            active = np.asarray(z_values, dtype=np.float64)
            nuisance = np.asarray(s_values, dtype=np.float64)
            means = np.asarray(mu_values, dtype=np.float64)
            probabilities = np.asarray(p_values, dtype=np.float64)
            actual = tuple(value[0] for value in mean_derivatives(active[None, ...], nuisance[None, ...], means, probabilities))
            references = [direct_autograd(active, arm, means, probabilities) for arm in (np.zeros_like(nuisance), nuisance)]
            reference = (np.stack([value[0] for value in references]), np.stack([value[1] for value in references]))
            errors = {
                "gradient_autograd": float(np.max(np.abs(actual[0] - reference[0]))),
                "hessian_autograd": float(np.max(np.abs(actual[1] - reference[1]))),
                "constant_nuisance": 0.0, "permutation": 0.0, "batch": 0.0, "k1_zero": 0.0,
            }
            check_ranges(*actual, name)
            for labels in LABELS:
                check_ranges(*_label_derivatives((active + means)[None, ...], nuisance[None, ...], labels), name + str(labels))
            assert np.array_equal(actual[1], np.swapaxes(actual[1], -1, -2)), (name, "hessian_symmetry")
            if len(active) == 1:
                errors["k1_zero"] = max(float(np.max(np.abs(value))) for value in actual)
            for constant in (0.0, 7.25, -11.0, 1000.0, -1000.0, 1e100):
                constant_values = mean_derivatives(active[None, ...], np.full((1, len(active)), constant), means, probabilities)
                error = max(float(np.max(np.abs(value[:, 0] - value[:, 1]))) for value in constant_values)
                errors["constant_nuisance"] = max(errors["constant_nuisance"], error)
            for permutation in itertools.permutations(range(3)):
                permuted = tuple(value[0] for value in mean_derivatives(
                    np.take(active, permutation, axis=1)[None, ...], nuisance[None, ...],
                    np.take(means, permutation), np.take(probabilities, permutation),
                ))
                indices = [PAIRS.index(tuple(sorted((permutation[i], permutation[j])))) for i, j in PAIRS]
                expected = (actual[0][:, indices], actual[1][:, indices][:, :, indices])
                errors["permutation"] = max(errors["permutation"], max_discrepancy(permuted, expected))
            doubled = mean_derivatives(np.stack((active, active)), np.stack((nuisance, nuisance)), means, probabilities)
            errors["batch"] = max_discrepancy(doubled, (actual[0][None, ...], actual[1][None, ...]))
            for key, error in errors.items():
                assert error <= tolerance, (name, key, error)
                maxima[key] = max(maxima[key], error)
            details.append({
                "name": name, "K": len(active), "active_z": active.tolist(), "log_nuisance": nuisance.tolist(),
                "mu": means.tolist(), "teacher_p": probabilities.tolist(),
                "gradient": actual[0].tolist(), "hessian": actual[1].tolist(),
                "autograd_gradient": reference[0].tolist(), "autograd_hessian": reference[1].tolist(),
                "max_abs_errors": errors,
            })
    finally:
        torch.set_num_threads(original_threads)
    return {
        "schema_version": "1.0", "status": "passed", "method": "fixed_fixtures_direct_second_and_fourth_price_autograd",
        "conditions": list(CONDITIONS), "pair_order": [list(pair) for pair in PAIRS],
        "float_dtype": "float64", "autograd_threads": 1, "random_draws": 0,
        "gaussian_integrations": 0, "fixture_count": len(fixtures), "label_count": len(LABELS),
        "autograd_gradient_scalar_comparisons": len(fixtures) * 2 * 3,
        "autograd_hessian_scalar_comparisons": len(fixtures) * 2 * 9,
        "fixed_label_range_checks": len(fixtures) * len(LABELS), "tolerance": tolerance,
        "per_bank_gradient_abs_bound": GRADIENT_ABS_BOUND,
        "per_bank_hessian_diagonal_range": [HESSIAN_DIAGONAL_LOWER, HESSIAN_DIAGONAL_UPPER],
        "per_bank_hessian_offdiagonal_abs_bound": HESSIAN_OFFDIAGONAL_ABS_BOUND,
        "max_abs_errors": maxima, "fixtures": details,
    }
