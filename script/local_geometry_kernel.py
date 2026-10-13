"""Deterministic Price curvature kernels for the matched n0/n256 audit.

This module never draws random numbers or writes files. ``price_curvature``
returns both nuisance conditions in the order (n0, n256), and the pair order
is (01, 02, 12). The arbitrary-precision dependency is loaded only when
``interval_curvature`` is called.
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
KERNEL_LOWER = -33.0 / 8.0
KERNEL_UPPER = 67.0 / 16.0
QUALIFICATION_TOLERANCE = 1e-10


def _validated_inputs(z: Any, s: Any, *, batched: bool) -> tuple[np.ndarray, np.ndarray]:
    """Convert finite inputs to float64 and enforce the public shape contract."""
    active = np.asarray(z, dtype=np.float64)
    nuisance = np.asarray(s, dtype=np.float64)
    expected_ndim = 3 if batched else 2
    if active.ndim != expected_ndim or active.shape[-1] != 3:
        shape = "(B, K, 3)" if batched else "(K, 3)"
        raise ValueError(f"z must have shape {shape}; got {active.shape}")
    if any(size < 1 for size in active.shape[:-1]):
        raise ValueError("z must contain at least one bank and one particle")
    if nuisance.shape != active.shape[:-1]:
        raise ValueError(f"s must have shape {active.shape[:-1]}; got {nuisance.shape}")
    if not np.all(np.isfinite(active)) or not np.all(np.isfinite(nuisance)):
        raise ValueError("z and s must contain finite float64 values")
    return active, nuisance


def price_curvature(z: Any, s: Any) -> np.ndarray:
    """Return the per-bank Price kernels with shape ``(B, 2, 3)``.

    Args:
        z: Active independent-normal coordinates, shape ``(B, K, 3)``.
        s: Log nuisance weights for n256, shape ``(B, K)``. The n0 arm
            replaces these weights by a constant within each bank.

    The function evaluates the fixed all-positive label kernel. Its expected
    value equals the corresponding diagonal population Hessian entry at I
    under the uniform teacher. It does not assert equality bank by bank to
    an eight-label average.
    """
    active, nuisance = _validated_inputs(z, s, batched=True)
    log_active = -np.logaddexp(0.0, -active).sum(axis=-1)
    q = expit(-active)
    r = q * (2.0 * q - 1.0)
    result = np.empty((active.shape[0], 2, len(PAIRS)), dtype=np.float64)

    for condition, log_weights in enumerate((log_active, log_active + nuisance)):
        pi = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        pi2 = pi * pi
        for pair_index, (first, second) in enumerate(PAIRS):
            q1, q2 = q[..., first], q[..., second]
            r1, r2 = r[..., first], r[..., second]
            a = q1 * q2
            a2 = a * a
            A = np.sum(pi * a, axis=1)
            B = np.sum(pi2 * a, axis=1)
            per_particle = (
                -pi * (1.0 - pi) * r1 * r2
                + 2.0 * pi2 * (1.0 - pi) * (q1 * q1 * r2 + q2 * q2 * r1)
                + pi2 * (1.0 - 4.0 * pi) * a2
            )
            result[:, condition, pair_index] = (
                np.sum(per_particle, axis=1) + A * A - 4.0 * A * B + 6.0 * B * B
            )
    return result


def interval_curvature(z_one: Any, s_one: Any, bits: int = 128) -> list[list[Any]]:
    """Enclose a single bank's ``(2, 3)`` kernel in Arb ball arithmetic.

    Inputs have shapes ``(K, 3)`` and ``(K,)`` and are treated as their exact
    binary64 values. The return value is a nested list of Arb values in the
    same condition/pair order as ``price_curvature``. Precision is restored
    after the call. Only this function loads the private python-flint install.

    Active likelihoods are products of Arb sigmoid values. Nuisance weights
    are exponentiated after subtracting the largest input log weight; no
    floating-point log likelihood enters the interval computation.
    """
    active, nuisance = _validated_inputs(z_one, s_one, batched=False)
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or bits < 64:
        raise ValueError("bits must be an integer of at least 64")
    private_tools = Path(__file__).resolve().parents[1] / ".cache" / "finite_selection_tools"
    if str(private_tools) not in sys.path:
        sys.path.insert(0, str(private_tools))
    from flint import arb, ctx  # Imported only for explicit interval qualification.

    previous_precision = ctx.prec
    ctx.prec = int(bits)
    try:
        az = [[arb(float(value)) for value in row] for row in active]
        sigmas = [[1 / (1 + (-value).exp()) for value in row] for row in az]
        q = [[1 / (1 + value.exp()) for value in row] for row in az]
        r = [[value * (2 * value - 1) for value in row] for row in q]
        likelihood = [row[0] * row[1] * row[2] for row in sigmas]
        shift = arb(float(np.max(nuisance)))
        nuisance_weights = [(arb(float(value)) - shift).exp() for value in nuisance]
        condition_weights = (
            likelihood,
            [weight * active_weight for weight, active_weight in zip(nuisance_weights, likelihood)],
        )
        result: list[list[Any]] = []
        for weights in condition_weights:
            denominator = sum(weights, arb(0))
            pi = [weight / denominator for weight in weights]
            row_result = []
            for first, second in PAIRS:
                a = [values[first] * values[second] for values in q]
                A = sum((weight * value for weight, value in zip(pi, a)), arb(0))
                B = sum((weight * weight * value for weight, value in zip(pi, a)), arb(0))
                local_terms = []
                for weight, values, second_derivatives, value in zip(pi, q, r, a):
                    q1, q2 = values[first], values[second]
                    r1, r2 = second_derivatives[first], second_derivatives[second]
                    weight2 = weight * weight
                    local_terms.append(
                        -weight * (1 - weight) * r1 * r2
                        + 2 * weight2 * (1 - weight) * (q1 * q1 * r2 + q2 * q2 * r1)
                        + weight2 * (1 - 4 * weight) * value * value
                    )
                row_result.append(sum(local_terms, arb(0)) + A * A - 4 * A * B + 6 * B * B)
            result.append(row_result)
        return result
    finally:
        ctx.prec = previous_precision


def qualification() -> dict[str, Any]:
    """Run six fixed deterministic fixtures against independent autograd.

    Checks all three pair derivatives for both arms, constant nuisance
    identity, all coordinate permutations, batching consistency, and the
    per-bank range [-33/8, 67/16]. No RNG is called. Raises AssertionError on
    an error exceeding 1e-10 or on a range failure. The return value contains
    JSON-compatible fixture data and maximum absolute discrepancies.
    """
    import torch
    import torch.nn.functional as torch_functional

    fixtures = (
        ("K1_zero", [[0.0, 0.0, 0.0]], [0.0]),
        ("K1_asymmetric", [[2.0, -3.0, 1.0]], [7.0]),
        ("K2_equal", [[0.1, 0.8, -0.5], [-0.7, 0.2, 1.5]], [0.0, 0.0]),
        ("K2_concentrated", [[0.1, 0.8, -0.5], [-0.7, 0.2, 1.5]], [0.0, -9.0]),
        (
            "K3_asymmetric",
            [[1.2, -0.3, 0.7], [-0.7, 0.8, -1.1], [0.1, 1.5, -0.2]],
            [0.0, -7.0, -0.5],
        ),
        (
            "K3_extreme",
            [[-10.0, 8.0, 0.1], [10.0, -8.0, -0.1], [0.0, 0.0, 0.0]],
            [2.0, -2.0, 0.0],
        ),
    )
    tolerance = QUALIFICATION_TOLERANCE
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def direct_autograd(active: np.ndarray, nuisance: np.ndarray, pair: tuple[int, int]) -> float:
        coordinates = torch.tensor(active, dtype=torch.float64, requires_grad=True)
        log_weights = torch.tensor(nuisance, dtype=torch.float64)
        objective = -torch.logsumexp(
            log_weights + torch_functional.logsigmoid(coordinates).sum(dim=1), dim=0
        )

        def price_operator(value: Any) -> Any:
            gradient = torch.autograd.grad(value, coordinates, create_graph=True)[0]
            terms = []
            for particle in range(len(active)):
                second = torch.autograd.grad(
                    gradient[particle, pair[0]],
                    coordinates,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                terms.append(second[particle, pair[1]])
            return sum(terms)

        return float(price_operator(price_operator(objective)).detach())

    details = []
    maximum_autograd_error = 0.0
    maximum_constant_error = 0.0
    maximum_permutation_error = 0.0
    maximum_batch_error = 0.0
    try:
        for name, active_values, nuisance_values in fixtures:
            active = np.asarray(active_values, dtype=np.float64)
            nuisance = np.asarray(nuisance_values, dtype=np.float64)
            actual = price_curvature(active[None, ...], nuisance[None, ...])[0]
            reference = np.asarray(
                [
                    [direct_autograd(active, arm_weights, pair) for pair in PAIRS]
                    for arm_weights in (np.zeros_like(nuisance), nuisance)
                ],
                dtype=np.float64,
            )
            autograd_error = float(np.max(np.abs(actual - reference)))
            assert autograd_error <= tolerance, (name, "autograd", autograd_error)
            assert np.all(actual >= KERNEL_LOWER) and np.all(actual <= KERNEL_UPPER), (name, actual)

            constant_error = 0.0
            for constant in (0.0, 7.25, -11.0, 1000.0, -1000.0):
                constant_result = price_curvature(
                    active[None, ...], np.full((1, len(active)), constant, dtype=np.float64)
                )[0]
                error = float(np.max(np.abs(constant_result[1] - constant_result[0])))
                constant_error = max(constant_error, error)
            assert constant_error <= tolerance, (name, "constant_nuisance", constant_error)

            permutation_error = 0.0
            for permutation in itertools.permutations(range(3)):
                permuted = price_curvature(
                    np.take(active, permutation, axis=1)[None, ...], nuisance[None, ...]
                )
                indices = [PAIRS.index(tuple(sorted((permutation[i], permutation[j])))) for i, j in PAIRS]
                expected = actual[:, indices]
                error = float(np.max(np.abs(permuted[0] - expected)))
                permutation_error = max(permutation_error, error)
            assert permutation_error <= tolerance, (name, "coordinate_permutation", permutation_error)

            doubled = price_curvature(np.stack((active, active)), np.stack((nuisance, nuisance)))
            batch_error = float(np.max(np.abs(doubled - actual[None, ...])))
            assert batch_error <= tolerance, (name, "batch_consistency", batch_error)
            maximum_autograd_error = max(maximum_autograd_error, autograd_error)
            maximum_constant_error = max(maximum_constant_error, constant_error)
            maximum_permutation_error = max(maximum_permutation_error, permutation_error)
            maximum_batch_error = max(maximum_batch_error, batch_error)
            details.append(
                {
                    "name": name,
                    "K": len(active),
                    "active_z": active.tolist(),
                    "log_nuisance": nuisance.tolist(),
                    "kernel": actual.tolist(),
                    "autograd": reference.tolist(),
                    "max_abs_error": autograd_error,
                    "constant_nuisance_max_abs_error": constant_error,
                    "pair_permutation_max_abs_error": permutation_error,
                    "batch_max_abs_error": batch_error,
                    "observed_min": float(np.min(actual)),
                    "observed_max": float(np.max(actual)),
                }
            )
    finally:
        torch.set_num_threads(original_threads)

    return {
        "schema_version": "1.0",
        "status": "passed",
        "method": "fixed_fixtures_direct_fourth_autograd",
        "conditions": list(CONDITIONS),
        "pair_order": [list(pair) for pair in PAIRS],
        "float_dtype": "float64",
        "autograd_threads": 1,
        "random_draws": 0,
        "fixture_count": len(fixtures),
        "autograd_scalar_comparisons": len(fixtures) * len(CONDITIONS) * len(PAIRS),
        "tolerance": tolerance,
        "per_bank_range": [KERNEL_LOWER, KERNEL_UPPER],
        "max_abs_error": maximum_autograd_error,
        "constant_nuisance_max_abs_error": maximum_constant_error,
        "pair_permutation_max_abs_error": maximum_permutation_error,
        "batch_max_abs_error": maximum_batch_error,
        "fixtures": details,
    }
