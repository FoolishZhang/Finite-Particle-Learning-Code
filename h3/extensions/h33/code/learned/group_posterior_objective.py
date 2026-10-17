"""C-independent group-proposal finite-K scalar and its fixed-bank derivative.

No sampling, fitting, teacher generation or file writes. The caller supplies
samples from its *actual* product proposal and log(phi*ell/q) local weights.
For an ideal group posterior these are log p_i(O_i); a numerical inverse-CDF
table requires its own density correction. Never silently treat the table as
the ideal posterior. Count multiplicities belong in the supplied local weights.

For caller-owned group rejection, each independent trial uses a complete
6*K-dimensional proposal/accept bank, e.g. u[R,A,B,K,3,2]. The whole replicate
includes all independent trial nets, and is the SE unit. Common proposals and
uniforms may be accepted separately for each local observation state; a sample
accepted for one state must not silently be used for another state's law.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.special import logsumexp

SCHEMA = "group_posterior_objective_v1"
PAIRS = ((0, 1), (0, 2), (1, 2))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def empirical_patterns(counts, observed_labels, *, M):
    """Actual observations only; no teacher PMF or latent input accepted."""
    M = _positive_integer(M, "M")
    counts, labels = np.asarray(counts), np.asarray(observed_labels)
    if (counts.ndim != 2 or counts.shape[1] != 3 or len(counts) < 1 or
            counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > M)):
        raise ValueError("counts must be nonempty integer (N,3) in 0..M")
    if labels.shape != counts.shape or not np.isin(labels, (-1, 1)).all():
        raise ValueError("observed_labels must be actual spins (N,3)")
    indices = counts + ((labels + 1)//2)*(M+1)
    unique, inverse, multiplicities = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    return {"state_indices": unique.astype(np.int64), "weights": multiplicities/len(counts),
            "counts": unique % (M+1), "labels": 2*(unique//(M+1))-1,
            "multiplicities": multiplicities, "inverse": inverse, "sample_size": len(counts),
            "data_role": "empirical_actual_counts_Y", "automatic_normalization": False}


def population_patterns(joint_pmf, *, M):
    """All joint states with raw offline PMF weights; no clipping/normalization."""
    M = _positive_integer(M, "M")
    pmf = np.asarray(joint_pmf, float)
    S = 2*(M+1)
    if pmf.shape != (S, S, S) or not np.isfinite(pmf).all() or np.any(pmf < 0):
        raise ValueError("finite nonnegative full joint PMF required")
    if abs(float(pmf.sum())-1) > 1e-10:
        raise ValueError("raw PMF mass outside numerical tolerance; no normalization")
    states = np.indices(pmf.shape).reshape(3, -1).T
    return {"state_indices": states, "weights": pmf.ravel(), "raw_mass": float(pmf.sum()),
            "counts": states % (M+1), "labels": 2*(states//(M+1))-1,
            "data_role": "population_offline_PMF_not_learning_visible", "automatic_normalization": False}


def _correlation(C):
    C = np.asarray(C, float)
    if (C.shape != (3, 3) or not np.isfinite(C).all() or
            not np.allclose(C, C.T, rtol=0, atol=1e-12) or
            not np.allclose(np.diag(C), 1, rtol=0, atol=1e-12)):
        raise ValueError("finite symmetric unit-diagonal m3 correlation required")
    L = np.linalg.cholesky(C)
    inv = np.linalg.solve(C, np.eye(3))
    return inv, float(2*np.log(np.diag(L)).sum())


def _replicate_statistics(values, gradients):
    R = len(values)
    return {"value": float(values.mean()), "gradient": gradients.mean(axis=0),
            "replicate_values": values, "replicate_gradients": gradients,
            "replicate_se": float(values.std(ddof=1)/math.sqrt(R)) if R > 1 else None,
            "replicate_gradient_se": gradients.std(axis=0, ddof=1)/math.sqrt(R) if R > 1 else None}


def paired_statistics(left, right):
    """Match scramble IDs externally, then retain the covariance by subtraction."""
    lv, rv = np.asarray(left["replicate_values"]), np.asarray(right["replicate_values"])
    lg, rg = np.asarray(left["replicate_gradients"]), np.asarray(right["replicate_gradients"])
    if lv.shape != rv.shape or lg.shape != rg.shape or lg.shape != (len(lv), 3):
        raise ValueError("matched replicate shapes required")
    return {**_replicate_statistics(lv-rv, lg-rg),
            "uncertainty_unit": "matched_independent_scramble_difference_not_iid_internal_bank",
            "matching_scramble_identity_is_callers_responsibility": True}


@dataclass
class PreparedGroupBanks:
    """Compact conditional lookup: [replicate,B,K,group,observed_state].

    No candidate likelihood recomputation. The lookup contains samples for
    every local state, correctly conditioned through the caller's sampler.
    The object borrows caller arrays; they must remain fixed during fitting.
    Moment tables, if supplied, must describe the actual sampling proposal.
    A moment quadrature error changes the CV target/gradient, not only levels.
    """
    x_group: np.ndarray
    log_marginal_weight: np.ndarray
    state_indices: np.ndarray
    weights: np.ndarray
    proposal_kind: str = "caller_supplied_C_independent_product_proposal"
    proposal_mean: np.ndarray | None = None
    proposal_second_moment: np.ndarray | None = None

    def __post_init__(self):
        self.x_group = np.asarray(self.x_group, float)
        if (self.x_group.ndim != 5 or self.x_group.shape[3] != 3 or
                min(self.x_group.shape) < 1 or not np.isfinite(self.x_group).all()):
            raise ValueError("finite x_group[R,B,K,3,S] required")
        self.R, self.B, self.K, _, self.S = self.x_group.shape
        local = np.asarray(self.log_marginal_weight, float)
        if local.shape not in ((3, self.S), self.x_group.shape) or not np.isfinite(local).all():
            raise ValueError("local log(phi*ell/q) weights must be [3,S] or x_group shape")
        self._constant_local = local if local.shape == (3, self.S) else None
        self.log_marginal_weight = np.broadcast_to(local, self.x_group.shape)
        self.state_indices = np.asarray(self.state_indices)
        self.weights = np.asarray(self.weights, float)
        states = self.state_indices
        if (states.ndim != 2 or states.shape[1] != 3 or len(states) < 1 or
                states.dtype.kind not in "iu" or np.any(states < 0) or np.any(states >= self.S)):
            raise ValueError("integer state_indices[P,3] within conditional lookup required")
        if (self.weights.shape != (len(states),) or not np.isfinite(self.weights).all() or
                np.any(self.weights < 0) or abs(float(self.weights.sum())-1) > 1e-10):
            raise ValueError("nonnegative pattern weights of raw mass one required; no normalization")
        if (self.proposal_mean is None) != (self.proposal_second_moment is None):
            raise ValueError("both actual-proposal moment tables or neither required")
        if self.proposal_mean is not None:
            self.proposal_mean = np.asarray(self.proposal_mean, float)
            self.proposal_second_moment = np.asarray(self.proposal_second_moment, float)
            if (self.proposal_mean.shape != (3, self.S) or self.proposal_second_moment.shape != (3, self.S) or
                    not np.isfinite(self.proposal_mean).all() or not np.isfinite(self.proposal_second_moment).all() or
                    np.any(self.proposal_second_moment < self.proposal_mean**2-1e-12)):
                raise ValueError("finite consistent actual-proposal [3,S] moments required")
        self._weighted_moment_cache = None

    def _control_weighted_moments(self):
        """Exact contraction over pattern weights, not a new approximation.

        The compact lookup uses each local state's sample across patterns.
        Pairwise marginals therefore suffice for weighted x_i*x_j. This
        reduces the C-independent centering work from P to S**2 states and
        is cached only while the caller's borrowed arrays remain frozen.
        """
        if self._weighted_moment_cache is not None:
            return (*self._weighted_moment_cache, True)
        empirical = np.empty((self.R, self.B, 3, 3)); expected = np.empty((3, 3))
        for i in range(3):
            marginal = np.bincount(self.state_indices[:, i], weights=self.weights, minlength=self.S)
            empirical[..., i, i] = np.einsum("rtks,s->rt", self.x_group[..., i, :]**2, marginal)/self.K
            expected[i, i] = marginal@self.proposal_second_moment[i]
        for i, j in PAIRS:
            marginal = np.zeros((self.S, self.S))
            np.add.at(marginal, (self.state_indices[:, i], self.state_indices[:, j]), self.weights)
            empirical[..., i, j] = np.einsum("rtka,ac,rtkc->rt", self.x_group[..., i, :], marginal,
                                            self.x_group[..., j, :], optimize=True)/self.K
            empirical[..., j, i] = empirical[..., i, j]
            expected[i, j] = self.proposal_mean[i]@marginal@self.proposal_mean[j]
            expected[j, i] = expected[i, j]
        self._weighted_moment_cache = (empirical, expected)
        return empirical, expected, False

    def evaluate(self, C, *, control="raw", pattern_chunk=64, return_pattern_details=False):
        """Differentiate the exact returned fixed-bank scalar, never Price.

        log w = sum local_logweight - .5 logdet(C) - .5 x'(C^-1-I)x.
        For physical c_ij (both symmetric entries vary), score is
        (C^-1 x)_i(C^-1 x)_j - (C^-1)_ij. Thus grad f=-sum softmax(w)*score.

        Optional uniform_logratio adds mean log(phi_C/phi_I) minus its
        actual-proposal expectation. Its expectation is zero with exact
        moments; the fixed scalar and derivative both change. This is not
        the old prior logell control. Raw remains available in either mode.
        """
        started = time.perf_counter()
        chunk = _positive_integer(pattern_chunk, "pattern_chunk")
        if control not in ("raw", "uniform_logratio"):
            raise ValueError("control must be raw or uniform_logratio")
        if control != "raw" and self.proposal_mean is None:
            raise ValueError("uniform_logratio requires actual-proposal moment tables")
        inv, logdet = _correlation(C)
        A = inv-np.eye(3)
        shape = (self.R, self.B)
        bank_value, raw_bank_value = np.zeros(shape), np.zeros(shape)
        bank_gradient, raw_bank_gradient = np.zeros(shape+(3,)), np.zeros(shape+(3,))
        details_value, details_gradient = [], []
        min_logw, max_logw, max_normalized_weight, min_ESS = math.inf, -math.inf, 0., math.inf
        for start in range(0, len(self.weights), chunk):
            stop = min(start+chunk, len(self.weights)); states = self.state_indices[start:stop]
            # Explicit m3 algebra avoids stacked x/y/score particle tensors.
            # It changes floating-point operation order, not the scalar formula.
            x = [np.moveaxis(self.x_group[..., g, states[:, g]], 2, 3) for g in range(3)]
            if self._constant_local is None:
                local = sum(np.moveaxis(self.log_marginal_weight[..., g, states[:, g]], 2, 3) for g in range(3))
            else:
                local = sum(self._constant_local[g, states[:, g]] for g in range(3))[None, None, :, None]
            ratio = -.5*logdet
            for g in range(3): ratio = ratio-.5*A[g, g]*x[g]*x[g]
            for i, j in PAIRS: ratio = ratio-A[i, j]*x[i]*x[j]
            logw = local+ratio
            logsum = logsumexp(logw, axis=-1)
            pi = np.exp(logw-logsum[..., None])
            y = [sum(inv[g, h]*x[h] for h in range(3)) for g in range(3)]
            scalar = -logsum+math.log(self.K)
            gradient = np.empty(scalar.shape+(3,))
            mean_score = np.empty_like(gradient) if control != "raw" and return_pattern_details else None
            for d, (i, j) in enumerate(PAIRS):
                score = y[i]*y[j]-inv[i, j]
                gradient[..., d] = -np.sum(pi*score, axis=-1)
                if mean_score is not None: mean_score[..., d] = score.mean(axis=-1)
            weight = self.weights[start:stop]
            raw_bank_value += np.einsum("rbp,p->rb", scalar, weight)
            raw_bank_gradient += np.einsum("rbpd,p->rbd", gradient, weight)
            if control != "raw" and return_pattern_details:
                mean = np.stack([self.proposal_mean[g, states[:, g]] for g in range(3)], axis=-1)
                Q = mean[..., :, None]*mean[..., None, :]
                for g in range(3): Q[:, g, g] = self.proposal_second_moment[g, states[:, g]]
                expected_ratio = -.5*logdet-.5*np.einsum("ij,pji->p", A, Q)
                invQinv = np.einsum("ij,pjk,kl->pil", inv, Q, inv, optimize=True)
                expected_score = np.stack([invQinv[:, i, j]-inv[i, j] for i, j in PAIRS], axis=-1)
                scalar = scalar+ratio.mean(axis=-1)-expected_ratio
                gradient = gradient+mean_score-expected_score
            bank_value += np.einsum("rbp,p->rb", scalar, weight)
            bank_gradient += np.einsum("rbpd,p->rbd", gradient, weight)
            if return_pattern_details:
                details_value.append(scalar); details_gradient.append(gradient)
            min_logw = min(min_logw, float(logw.min())); max_logw = max(max_logw, float(logw.max()))
            max_normalized_weight = max(max_normalized_weight, float(pi.max()))
            min_ESS = min(min_ESS, float((1/np.sum(pi*pi, axis=-1)).min()))
        control_cache_hit = None
        if control != "raw" and not return_pattern_details:
            empirical_Q, expected_Q, control_cache_hit = self._control_weighted_moments()
            difference = empirical_Q-expected_Q
            bank_value -= .5*np.einsum("ij,rtji->rt", A, difference)
            centered = np.einsum("ij,rtjk,kl->rtil", inv, difference, inv, optimize=True)
            for d, (i, j) in enumerate(PAIRS): bank_gradient[..., d] += centered[..., i, j]
        if not all(np.isfinite(a).all() for a in (bank_value, bank_gradient, raw_bank_value, raw_bank_gradient)):
            raise FloatingPointError("nonfinite group-proposal scalar/gradient; no clipping")
        output = {"schema": SCHEMA, **_replicate_statistics(bank_value.mean(axis=1), bank_gradient.mean(axis=1)),
                  "bank_values": bank_value, "bank_gradients": bank_gradient,
                  "raw_bank_values": raw_bank_value, "raw_bank_gradients": raw_bank_gradient,
                  "control": control, "proposal_kind": self.proposal_kind,
                  "control_weighted_moment_cache_hit": control_cache_hit,
                  "replicates": self.R, "banks_per_replicate": self.B, "K": self.K,
                  "patterns": len(self.weights), "raw_pattern_mass": float(self.weights.sum()),
                  "particle_pattern_terms": self.R*self.B*self.K*len(self.weights),
                  "local_likelihood_evaluations_this_call": 0,
                  "gradient_qualification": "exact_fixed_C_independent_bank_Gaussian_density_ratio_derivative_not_Price",
                  "uncertainty_unit": "independent_scramble_replicate_not_internal_bank_or_particle",
                  "proposal_and_moment_accuracy_are_callers_responsibility": True,
                  "logdet_C": logdet, "min_log_weight": min_logw, "max_log_weight": max_logw,
                  "max_normalized_particle_weight": max_normalized_weight, "min_particle_ESS": min_ESS,
                  "seconds": time.perf_counter()-started, "automatic_normalization": False}
        raw = _replicate_statistics(raw_bank_value.mean(axis=1), raw_bank_gradient.mean(axis=1))
        for key, value in raw.items(): output["raw_"+key] = value
        if return_pattern_details:
            output["pattern_values"] = np.concatenate(details_value, axis=2)
            output["pattern_gradients"] = np.concatenate(details_gradient, axis=2)
        return output
