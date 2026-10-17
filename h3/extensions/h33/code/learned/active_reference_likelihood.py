"""Deterministic full-joint m3 observation likelihood and same-GH derivatives.

All private variables and the three active Gaussian coordinates are integrated.
Counts retain their binomial multiplicities. Cholesky derivatives differentiate
the very same finite positive GH sum used for the PMF, rather than substituting
a population Price identity for its numerical derivative. Imports never sample,
choose teachers/optimization settings, run an optimizer or write files.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time
import numpy as np
from scipy.special import expit, roots_hermitenorm
import signal_proposal_core as legacy
from informative_projection import project_coordinates

SCHEMA = "active_reference_likelihood_v1"
QUALIFICATION = "full_joint_formula_with_ordinary_positive_3D_active_and_1D_private_GH; same_order_Cholesky_derivative; not_interval_certificate"


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@lru_cache(maxsize=32)
def normal_rule(order):
    order = _integer(order, "GH order", 2)
    nodes, weights = roots_hermitenorm(order)
    weights = weights/math.sqrt(2*math.pi)
    if not np.isfinite(nodes).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FloatingPointError("Finite positive GH weights required")
    nodes.setflags(write=False); weights.setflags(write=False)
    return nodes, weights


def observation_indices(counts, observed_labels, *, M):
    """Full-joint per-axis order (-1,c=0..M),(+1,c=0..M)."""
    M = _integer(M, "M")
    counts, y = np.asarray(counts), np.asarray(observed_labels)
    if counts.ndim != 2 or len(counts) < 1 or counts.shape[1] != 3 or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > M):
        raise ValueError("counts must be integer nonempty (N,3) in 0..M")
    if y.shape != counts.shape or not np.isin(y, (-1, 1)).all():
        raise ValueError("observed_labels must be actual spins (N,3)")
    return counts.astype(np.int64)+((y.astype(np.int64)+1)//2)*(M+1)


@dataclass(frozen=True)
class ObservationPatterns:
    M: int
    sample_size: int
    state_indices: np.ndarray
    multiplicities: np.ndarray
    frequencies: np.ndarray
    inverse: np.ndarray


def prepare_observations(counts, observed_labels, *, M):
    """Compress actual observed patterns once, without teacher/data inference."""
    indices = observation_indices(counts, observed_labels, M=M)
    unique, inverse, multiplicities = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    frequency = multiplicities/len(indices)
    for value in (unique, inverse, multiplicities, frequency): value.setflags(write=False)
    return ObservationPatterns(int(M), len(indices), unique, multiplicities, frequency, inverse)


class ActiveReferenceLikelihood:
    """Fixed-model, fixed-order integrated likelihood with reusable pattern cache.

    The Gaussian rules, combinatorial coefficients and pair layout are cached.
    The first-coordinate factors are C-independent at exact unit diagonal;
    their actual Cholesky L00 is retained for roundoff-level accepted diagonals.
    The last C's whole PMF and derivatives are cached to share fun/jac calls.
    """
    def __init__(self, *, M, epsilon, tau, outer_order, inner_order,
                 pair_chunk=64, h_chunk=4096):
        self.M = _integer(M, "M")
        self.epsilon = np.asarray(epsilon, dtype=float).copy()
        if self.epsilon.ndim == 0: self.epsilon = np.full(3, self.epsilon)
        if self.epsilon.shape != (3,) or not np.isfinite(self.epsilon).all():
            raise ValueError("epsilon must be finite scalar or three group couplings")
        if isinstance(tau, bool) or not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be a finite nonnegative scalar")
        self.tau = float(tau)
        self.outer_order, self.inner_order = _integer(outer_order, "outer_order", 2), _integer(inner_order, "inner_order", 2)
        self.pair_chunk, self.h_chunk = _integer(pair_chunk, "pair_chunk"), _integer(h_chunk, "h_chunk")
        self.g, self.w = normal_rule(self.outer_order)
        self.private_g, self.private_w = normal_rule(self.inner_order)
        self.S = 2*(self.M+1)
        self.count_values = np.arange(self.M+1)
        self.combinations = np.asarray([math.comb(self.M, int(c)) for c in self.count_values], dtype=float)
        self.i1 = np.repeat(np.arange(self.outer_order), self.outer_order)
        self.i2 = np.tile(np.arange(self.outer_order), self.outer_order)
        self.pair_w = self.w[self.i1]*self.w[self.i2]
        self.epsilon.setflags(write=False); self.combinations.setflags(write=False)
        self._first_scale = None; self._first_factors = None
        self._last_key = None; self._last_law = None

    def _group_factors(self, x, group):
        """Positive local PMF and its derivative with respect to this x."""
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        h, hc, hp = (np.empty(len(flat)) for _ in range(3))
        e = self.epsilon[group]
        for start in range(0, len(flat), self.h_chunk):
            stop = min(start+self.h_chunk, len(flat))
            logits = e*flat[start:stop, None]+self.tau*self.private_g
            if not np.isfinite(logits).all(): raise FloatingPointError("Private quadrature logits overflow")
            p, complement = expit(logits), expit(-logits)
            h[start:stop] = p@self.private_w
            hc[start:stop] = complement@self.private_w
            hp[start:stop] = (p*complement)@self.private_w
        if not all(np.isfinite(v).all() for v in (h, hc, hp)) or np.any(h <= 0) or np.any(hc <= 0):
            raise FloatingPointError("Private GH probability underflow/nonfinite; no clipping")
        h, hc, hp = (value.reshape(x.shape) for value in (h, hc, hp))
        c = self.count_values
        bins = self.combinations*h[..., None]**c*hc[..., None]**(self.M-c)
        active_p, active_n = expit(x), expit(-x)
        nuisance_score = e*hp[..., None]*(c/h[..., None]-(self.M-c)/hc[..., None])
        negative, positive = active_n[..., None]*bins, active_p[..., None]*bins
        factors = np.concatenate((negative, positive), axis=-1)
        derivative = np.concatenate((negative*(nuisance_score-active_p[..., None]),
                                     positive*(nuisance_score+active_n[..., None])), axis=-1)
        if not np.isfinite(factors).all() or not np.isfinite(derivative).all():
            raise FloatingPointError("Nonfinite group PMF/derivative; no repair")
        return factors, derivative

    def law(self, C):
        """PMF[S,S,S], dPMF[physical_pair,S,S,S] for the identical finite rule."""
        C = legacy._matrix(C)
        if C.shape != (3, 3): raise ValueError("The integrated reference requires m3")
        key = C.tobytes()
        if key == self._last_key:
            return {**self._last_law, "cache_hit": True}
        started = time.perf_counter()
        L, dL = legacy.cholesky_with_derivatives(C)
        first_cached = self._first_scale == L[0, 0]
        if not first_cached:
            self._first_factors = self._group_factors(L[0, 0]*self.g, 0)[0]
            self._first_scale = L[0, 0]
        f1 = self._first_factors
        x2 = L[1, 0]*self.g[self.i1]+L[1, 1]*self.g[self.i2]
        dx2 = dL[:, 1, 0, None]*self.g[self.i1]+dL[:, 1, 1, None]*self.g[self.i2]
        f2, f2prime = self._group_factors(x2, 1)
        df2 = dx2[..., None]*f2prime[None, :, :]
        pairs = self.outer_order**2
        f3 = np.empty((pairs, self.S)); df3 = np.empty((3, pairs, self.S))
        for start in range(0, pairs, self.pair_chunk):
            stop = min(start+self.pair_chunk, pairs)
            g1, g2 = self.g[self.i1[start:stop]], self.g[self.i2[start:stop]]
            x3 = (L[2, 0]*g1+L[2, 1]*g2)[:, None]+L[2, 2]*self.g
            factors, derivative = self._group_factors(x3, 2)
            f3[start:stop] = np.einsum("bns,n->bs", factors, self.w)
            for p in range(3):
                dx3 = (dL[p, 2, 0]*g1+dL[p, 2, 1]*g2)[:, None]+dL[p, 2, 2]*self.g
                df3[p, start:stop] = np.einsum("bns,bn,n->bs", derivative, dx3, self.w)
        pmf = np.empty((self.S,)*3); gradient = np.empty((3,)+(self.S,)*3)
        for a in range(self.S):
            weighted = self.pair_w*f1[self.i1, a]
            pmf[a] = f2.T@(weighted[:, None]*f3)
            for p in range(3):
                gradient[p, a] = df2[p].T@(weighted[:, None]*f3)+f2.T@(weighted[:, None]*df3[p])
        if not np.isfinite(pmf).all() or np.any(pmf < 0) or not np.isfinite(gradient).all():
            raise FloatingPointError("Invalid GH PMF/derivative; no normalization or clipping")
        pmf.setflags(write=False); gradient.setflags(write=False)
        result = {"schema": SCHEMA, "joint_pmf": pmf, "pmf_gradient": gradient,
                  "mass": float(pmf.sum()), "mass_error": float(pmf.sum()-1),
                  "mass_gradient": gradient.sum(axis=(1, 2, 3)), "minimum_probability": float(pmf.min()),
                  "pair_order": legacy.pair_order(3), "M": self.M, "epsilon": self.epsilon.copy(), "tau": self.tau,
                  "outer_order": self.outer_order, "inner_order": self.inner_order,
                  "combination_probabilities_included": True, "automatic_PMF_normalization": False,
                  "gradient_method": "same_finite_GH_Cholesky_chain_rule",
                  "private_h_arguments_this_call": (0 if first_cached else self.outer_order)+self.outer_order**2+self.outer_order**3,
                  "private_node_evaluations_this_call": ((0 if first_cached else self.outer_order)+self.outer_order**2+self.outer_order**3)*self.inner_order,
                  "first_group_cache_hit": first_cached, "cache_hit": False,
                  "seconds": time.perf_counter()-started, "qualification": QUALIFICATION}
        self._last_key, self._last_law = key, result
        return result

    def empirical_objective(self, C, counts=None, observed_labels=None, *, patterns=None):
        """Mean joint empirical NLL and physical gradient, actual observations only."""
        if patterns is not None and (counts is not None or observed_labels is not None):
            raise ValueError("Provide prepared patterns or actual counts/Y, not both")
        if patterns is None: patterns = prepare_observations(counts, observed_labels, M=self.M)
        if not isinstance(patterns, ObservationPatterns) or patterns.M != self.M:
            raise ValueError("Prepared observation patterns have a different model")
        law = self.law(C)
        indices = tuple(patterns.state_indices.T)
        p, dp = law["joint_pmf"][indices], law["pmf_gradient"][(slice(None),)+indices]
        if np.any(p <= 0): raise FloatingPointError("Observed PMF probability underflow; no floor")
        value = float(-patterns.frequencies@np.log(p))
        gradient = -np.sum(dp*(patterns.frequencies/p)[None, :], axis=1)
        return {"nll": value, "gradient": gradient, "N": patterns.sample_size,
                "unique_observation_patterns": len(patterns.state_indices), "candidate_mass": law["mass"],
                "candidate_mass_error": law["mass_error"], "combination_constants_included": True,
                "law_cache_hit": law["cache_hit"], "data_role": "actual_observed_joint_counts_Y",
                "qualification": QUALIFICATION}

    def population_objective(self, C, teacher_pmf):
        """Offline raw teacher-PMF crossentropy; teacher mass errors retained."""
        teacher = np.asarray(teacher_pmf, dtype=float)
        if teacher.shape != (self.S,)*3 or not np.isfinite(teacher).all() or np.any(teacher < 0) or teacher.sum() <= 0:
            raise ValueError("teacher_pmf must be finite nonnegative nonempty full-joint PMF")
        law = self.law(C); positive = teacher > 0
        p = law["joint_pmf"][positive]
        if np.any(p <= 0): raise FloatingPointError("Teacher-positive candidate cell underflow; no floor")
        value = float(-np.sum(teacher[positive]*np.log(p)))
        gradient = -np.sum(law["pmf_gradient"][:, positive]*(teacher[positive]/p)[None, :], axis=1)
        return {"crossentropy": value, "gradient": gradient, "teacher_mass": float(teacher.sum()),
                "teacher_mass_error": float(teacher.sum()-1), "candidate_mass": law["mass"],
                "candidate_mass_error": law["mass_error"], "automatic_teacher_normalization": False,
                "combination_constants_included": True, "law_cache_hit": law["cache_hit"],
                "data_role": "offline_population_evaluation_only", "qualification": QUALIFICATION}


def projected_gradient_mapping(C, gradient, *, step=1., delta=.15):
    """Existing Euclidean projection for constraint stationarity, no optimizer."""
    c = legacy.correlation_coordinates(C); g = np.asarray(gradient, dtype=float)
    if c.shape != (3,) or g.shape != (3,) or not np.isfinite(g).all(): raise ValueError("Finite m3 physical gradient required")
    if not np.isscalar(step) or not np.isfinite(step) or step <= 0: raise ValueError("step must be positive and finite")
    projection = project_coordinates(c-step*g, delta=delta)
    mapping = (c-projection["coordinates"])/step
    return {"mapping": mapping, "norm": float(np.linalg.norm(mapping)), "projected_coordinates": projection["coordinates"],
            "projected_matrix": legacy.correlation_matrix(projection["coordinates"]), "projection": projection,
            "step": float(step), "delta": float(delta), "qualification": "numerical_Dykstra_projection; not_global_optimality_certificate"}
