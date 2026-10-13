"""Float64 kernels for a dense Bernoulli model with two Gaussian latents.

Rows of W are emission loadings and logits are ``b + z @ W.T``. Patterns use
0/1 labels in lexicographic order. ``exact_loss_grad`` denotes the deterministic
Gauss--Hermite (GH) reference objective, not a certified exact Gaussian integral.
The caller must qualify quadrature order/rotation for its chosen parameter box.

Numerical boundaries
--------------------
* Emission logs use logaddexp; mixture logs use logsumexp. Neither likelihoods
  nor label probabilities are clipped or silently renormalized. In particular,
  model_pmf returns the raw GH integrals, with their floating-point mass error.
* model_pmf can underflow exceptionally small probabilities to zero. The loss
  and gradient routines operate in log space and do not take log(model_pmf).
  Logistic derivatives may saturate at machine precision for extreme logits.
* Finite inputs and finite computed logits/log mixtures are required. Overflow
  is reported instead of being converted into a usable-looking finite result.
* Pattern enumeration costs O(2**m). GH has order**dim nodes. MC processing is
  chunked over banks and patterns, never over a flattened B*K particle bank.
  Peak working storage is O(bank_chunk*K*(m+pattern_chunk)), beyond the inputs.
* Input probability mass must be within 1e-10 relative / 1e-12 absolute of one;
  the supplied values themselves are used unchanged.

No random number generation, global RNG manipulation, file I/O, or old research
module dependency is used here. The functions do not mutate caller arrays.
"""
from __future__ import annotations

import math
import operator
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logsumexp, roots_hermitenorm

DEFAULT_GH_ORDER = 24
DEFAULT_PATTERN_CHUNK_SIZE = 32
_MC_TARGET_ELEMENTS = 262144


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _real64(value: ArrayLike, name: str, *, check_finite: bool = True) -> NDArray[np.float64]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=np.float64)
    if check_finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def patterns(m: int) -> NDArray[np.int8]:
    """Return all 0/1 patterns, first coordinate most significant, shape (2**m,m)."""
    m = _positive_integer(m, "m")
    if m >= np.iinfo(np.intp).bits - 1:
        raise ValueError("2**m patterns cannot be indexed on this platform")
    integers = np.arange(1 << m, dtype=np.uint64)
    shifts = np.arange(m - 1, -1, -1, dtype=np.uint64)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def gh_nodes(order: int, dim: int = 2, angle: float = 0.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standard-normal tensor GH nodes and log weights, without mass correction.

    A two-dimensional row node is rotated as ``z @ R(angle).T``, where R is the
    usual counterclockwise rotation. Nonzero rotation is supported only at dim=2.
    Zero one-dimensional weights at extremely high orders remain log-weight -inf.
    """
    order = _positive_integer(order, "order")
    dim = _positive_integer(dim, "dim")
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    if dim != 2 and angle != 0.0:
        raise ValueError("a nonzero angle requires dim=2")
    one_nodes, one_weights = roots_hermitenorm(order)
    if not np.isfinite(one_nodes).all() or not np.isfinite(one_weights).all() or np.any(one_weights < 0):
        raise FloatingPointError("GH node/weight construction was not finite and nonnegative")
    with np.errstate(divide="ignore"):
        one_logweights = np.log(one_weights) - 0.5 * math.log(2.0 * math.pi)
    indices = np.indices((order,) * dim, dtype=np.intp).reshape(dim, -1).T
    nodes = np.asarray(one_nodes[indices], dtype=np.float64)
    logweights = np.asarray(one_logweights[indices].sum(axis=1), dtype=np.float64)
    if not np.isfinite(logsumexp(logweights)):
        raise FloatingPointError("GH weights have no finite positive mass")
    if angle != 0.0:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        nodes = nodes @ rotation.T
    return nodes, logweights


def _parameters(b: ArrayLike, W: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    b_array = _real64(b, "b")
    w_array = _real64(W, "W")
    if b_array.ndim != 1 or not len(b_array):
        raise ValueError("b must be a nonempty vector of length m")
    if w_array.shape != (len(b_array), 2):
        raise ValueError("W must have shape (m,2), matching b")
    return b_array, w_array


def _target(probs: ArrayLike, m: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probabilities = _real64(probs, "probs")
    if probabilities.shape != (1 << m,) or np.any(probabilities < 0):
        raise ValueError("probs must be a nonnegative vector of length 2**m")
    total = math.fsum(float(value) for value in probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("probs must sum to one (no automatic renormalization)")
    active = probabilities > 0
    return patterns(m)[active].astype(np.float64), probabilities[active]


def _pattern_chunks(labels: NDArray[np.float64], probabilities: NDArray[np.float64], size: int
                    ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    for start in range(0, len(labels), size):
        yield labels[start:start + size], probabilities[start:start + size]


def _emission_logs(logits: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not np.isfinite(logits).all():
        raise FloatingPointError("nonfinite logits; reduce the parameter/input range")
    # Both arrays are nonpositive: selecting them avoids cancellation of large
    # y*logit and softplus terms when a Bernoulli observation is highly likely.
    return -np.logaddexp(0.0, -logits), -np.logaddexp(0.0, logits)


def _log_likelihood(log_yes: NDArray[np.float64], log_no: NDArray[np.float64], labels: NDArray[np.float64]
                    ) -> NDArray[np.float64]:
    result = log_yes @ labels.T + log_no @ (1.0 - labels).T
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite emission log likelihood")
    return result


def model_pmf(b: ArrayLike, W: ArrayLike, order: int, angle: float = 0.0, *,
              pattern_chunk_size: int = DEFAULT_PATTERN_CHUNK_SIZE) -> NDArray[np.float64]:
    """Return raw GH probabilities in patterns(m) order; do not renormalize."""
    b_array, w_array = _parameters(b, W)
    chunk = _positive_integer(pattern_chunk_size, "pattern_chunk_size")
    nodes, logweights = gh_nodes(order, dim=2, angle=angle)
    labels = patterns(len(b_array)).astype(np.float64)
    probabilities = np.empty(len(labels), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        log_yes, log_no = _emission_logs(b_array + nodes @ w_array.T)
        for start in range(0, len(labels), chunk):
            block = labels[start:start + chunk]
            log_likelihood = _log_likelihood(log_yes, log_no, block)
            log_mass = logsumexp(log_likelihood + logweights[:, None], axis=0)
            probabilities[start:start + chunk] = np.exp(log_mass)
    return probabilities


def exact_loss_grad(b: ArrayLike, W: ArrayLike, probs: ArrayLike, order: int = DEFAULT_GH_ORDER,
                    angle: float = 0.0, *, pattern_chunk_size: int = DEFAULT_PATTERN_CHUNK_SIZE
                    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """GH cross entropy and its derivatives, with no PMF mass renormalization.

    The returned derivatives differentiate this fixed quadrature objective;
    order and angle are held fixed. Shapes are scalar, (m,), and (m,2).
    """
    b_array, w_array = _parameters(b, W)
    labels, target = _target(probs, len(b_array))
    chunk = _positive_integer(pattern_chunk_size, "pattern_chunk_size")
    nodes, logweights = gh_nodes(order, dim=2, angle=angle)
    grad_b = np.zeros_like(b_array)
    grad_w = np.zeros_like(w_array)
    loss = 0.0
    with np.errstate(over="raise", invalid="raise"):
        logits = b_array + nodes @ w_array.T
        log_yes, log_no = _emission_logs(logits)
        sigma = expit(logits)
        for block, probability in _pattern_chunks(labels, target, chunk):
            log_joint = _log_likelihood(log_yes, log_no, block) + logweights[:, None]
            log_mass = logsumexp(log_joint, axis=0)
            if not np.isfinite(log_mass).all():
                raise FloatingPointError("nonfinite GH mixture log likelihood")
            loss -= float(np.dot(probability, log_mass))
            weighted_posterior = np.exp(log_joint - log_mass[None, :]) * probability[None, :]
            residual = sigma * weighted_posterior.sum(axis=1)[:, None] - weighted_posterior @ block
            grad_b += residual.sum(axis=0)
            grad_w += residual.T @ nodes
    if not math.isfinite(loss) or not np.isfinite(grad_b).all() or not np.isfinite(grad_w).all():
        raise FloatingPointError("nonfinite GH loss or gradient")
    return loss, grad_b, grad_w


def _mc_evaluate(b: ArrayLike, W: ArrayLike, probs: ArrayLike, banks: ArrayLike, *,
                 need_gradient: bool, bank_chunk_size: int | None,
                 pattern_chunk_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    b_array, w_array = _parameters(b, W)
    labels, target = _target(probs, len(b_array))
    bank_array = _real64(banks, "banks", check_finite=False)
    if bank_array.ndim != 3 or bank_array.shape[2] != 2 or min(bank_array.shape[:2]) < 1:
        raise ValueError("banks must have shape (B,K,2) with positive B and K")
    count, particles, _ = bank_array.shape
    p_chunk = min(_positive_integer(pattern_chunk_size, "pattern_chunk_size"), len(labels))
    if bank_chunk_size is None:
        p_chunk = min(p_chunk, max(1, _MC_TARGET_ELEMENTS // particles))
        b_chunk = min(count, max(1, _MC_TARGET_ELEMENTS // (particles * max(p_chunk, len(b_array)))))
    else:
        b_chunk = min(count, _positive_integer(bank_chunk_size, "bank_chunk_size"))
    losses = np.zeros(count, dtype=np.float64)
    grad_b = np.zeros_like(b_array)
    grad_w = np.zeros_like(w_array)
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, count, b_chunk):
            stop = min(start + b_chunk, count)
            nodes = bank_array[start:stop]
            if not np.isfinite(nodes).all():
                raise ValueError("banks must be finite")
            logits = b_array + nodes @ w_array.T
            log_yes, log_no = _emission_logs(logits)
            sigma = expit(logits) if need_gradient else None
            for block, probability in _pattern_chunks(labels, target, p_chunk):
                log_likelihood = _log_likelihood(log_yes, log_no, block)
                # Particle normalization is within each B-index, separately for
                # every pattern. The B-index is averaged only after its log.
                log_sum = logsumexp(log_likelihood, axis=1)
                log_mean = log_sum - math.log(particles)
                losses[start:stop] -= log_mean @ probability
                if need_gradient:
                    weighted_posterior = np.exp(log_likelihood - log_sum[:, None, :]) * probability[None, None, :]
                    residual = sigma * weighted_posterior.sum(axis=2)[:, :, None] - weighted_posterior @ block
                    grad_b += residual.sum(axis=(0, 1))
                    grad_w += np.einsum("bkm,bkq->mq", residual, nodes, optimize=True)
    grad_b /= count
    grad_w /= count
    if not np.isfinite(losses).all() or not np.isfinite(grad_b).all() or not np.isfinite(grad_w).all():
        raise FloatingPointError("nonfinite MC loss or gradient")
    return losses, grad_b, grad_w


def mc_bank_losses(b: ArrayLike, W: ArrayLike, probs: ArrayLike, banks: ArrayLike, *,
                   bank_chunk_size: int | None = None,
                   pattern_chunk_size: int = DEFAULT_PATTERN_CHUNK_SIZE) -> NDArray[np.float64]:
    """Return B losses for paired-bank inference, without flattening B and K."""
    return _mc_evaluate(b, W, probs, banks, need_gradient=False,
                        bank_chunk_size=bank_chunk_size, pattern_chunk_size=pattern_chunk_size)[0]


def mc_loss_grad(b: ArrayLike, W: ArrayLike, probs: ArrayLike, banks: ArrayLike, *,
                 bank_chunk_size: int | None = None,
                 pattern_chunk_size: int = DEFAULT_PATTERN_CHUNK_SIZE
                 ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Bank-mean finite-particle loss and gradients, sharing each bank across Y."""
    losses, grad_b, grad_w = _mc_evaluate(b, W, probs, banks, need_gradient=True,
                                        bank_chunk_size=bank_chunk_size,
                                        pattern_chunk_size=pattern_chunk_size)
    return float(losses.mean()), grad_b, grad_w


def angular_grad(grad_W: ArrayLike, W: ArrayLike) -> NDArray[np.float64]:
    """Chain rule for counterclockwise row angles: g_j dot (-W_j2,W_j1)."""
    gradient = _real64(grad_W, "grad_W")
    weights = _real64(W, "W")
    if weights.ndim != 2 or weights.shape[1] != 2 or gradient.shape != weights.shape:
        raise ValueError("grad_W and W must have matching (m,2) shapes")
    with np.errstate(over="raise", invalid="raise"):
        return -gradient[:, 0] * weights[:, 1] + gradient[:, 1] * weights[:, 0]
