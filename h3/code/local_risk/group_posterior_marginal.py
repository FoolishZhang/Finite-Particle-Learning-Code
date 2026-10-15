"""One-dimensional actual-observation posterior and caller-owned rejection banks.

The target is continuous phi(x)*ell_h(O|x), where h uses the specified finite
positive private GH rule. The piecewise envelope is only a rejection proposal;
it is NOT substituted for this target. No RNG, teacher inference or optimizer.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, log_expit, ndtr, ndtri, roots_legendre, xlogy
from active_reference_likelihood import normal_rule

SCHEMA = "group_posterior_marginal_v1"
QUALIFICATION = ("continuous_rejection_target_with_finite_private_GH; "
                 "ordinary_GH_normalizers_and_moments; floating_point_and_finite_bits_not_strict_unbiased_certificate")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return int(value)


class RejectionExhausted(RuntimeError):
    """The complete attempt fails; partial accepted samples remain inspectable."""
    def __init__(self, result):
        self.result = result
        super().__init__(f"rejection exhausted: {result['unaccepted_coordinates']} coordinates; no retry or replacement")


class RejectionFailure(RuntimeError):
    """Sampling failure with requested-work ledger and already accepted values."""
    def __init__(self, message, result):
        self.result = result
        super().__init__(message)


class GroupPosteriorMarginal:
    """Cache all 2*(M+1) local posteriors in negative-label/count-first order.

    Two half-domain tables accumulate mass from their infinite tails toward x=0.
    Right coordinates are SF(x)=Phi(-x), never 1-Phi(x). Infinite tails are
    included as entire intervals, and inverse transforms do not clip uniforms.
    """
    def __init__(self, *, M, epsilon, tau, inner_order=256,
                 diagnostic_inner_order=192, normalizer_orders=(128, 192),
                 grid_size=4097, tail_x=8., chunk_size=4096):
        self.M = _integer(M, "M")
        self.epsilon = np.asarray(epsilon, dtype=float).copy()
        if self.epsilon.ndim == 0: self.epsilon = np.full(3, self.epsilon)
        if self.epsilon.shape != (3,) or not np.isfinite(self.epsilon).all():
            raise ValueError("epsilon must be finite scalar or length three")
        if isinstance(tau, bool) or not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and nonnegative")
        self.tau = float(tau)
        self.inner_order = _integer(inner_order, "inner_order", 2)
        self.diagnostic_inner_order = _integer(diagnostic_inner_order, "diagnostic_inner_order", 2)
        if len(normalizer_orders) != 2: raise ValueError("two normalizer orders required")
        self.normalizer_orders = tuple(_integer(n, "normalizer order", 2) for n in normalizer_orders)
        if self.normalizer_orders[0] >= self.normalizer_orders[1]:
            raise ValueError("normalizer orders must increase")
        self.grid_size = _integer(grid_size, "grid_size", 3)
        if self.grid_size % 2 != 1: raise ValueError("grid_size must be odd")
        if not np.isscalar(tail_x) or not np.isfinite(tail_x) or tail_x <= 0:
            raise ValueError("tail_x must be positive finite")
        self.tail_x = float(tail_x)
        self.chunk_size = _integer(chunk_size, "chunk_size")
        self.S = 2*(self.M+1)
        self.counts = np.arange(self.S) % (self.M+1)
        self.labels = np.where(np.arange(self.S) < self.M+1, -1, 1)
        self.log_choose = np.asarray([math.log(math.comb(self.M, int(c))) for c in self.counts])
        self._levels = {}
        self._cdf_cache = {}
        self._envelopes = {}
        self._build_envelopes()

    def probabilities(self, x, group, *, inner_order=None):
        """Separate positive h and hc sums, including analytic infinite limits."""
        group = _integer(group, "group", 0)
        if group > 2: raise ValueError("group outside 0..2")
        x = np.asarray(x, dtype=float)
        if np.isnan(x).any(): raise ValueError("x cannot contain NaN")
        nodes, weights = normal_rule(self.inner_order if inner_order is None else inner_order)
        e = self.epsilon[group]
        flat = x.ravel(); h = np.empty(flat.size); hc = np.empty(flat.size)
        finite = np.flatnonzero(np.isfinite(flat))
        for start in range(0, len(finite), self.chunk_size):
            ix = finite[start:start+self.chunk_size]
            logits = e*flat[ix, None]+self.tau*nodes
            if not np.isfinite(logits).all(): raise FloatingPointError("private logits overflow")
            h[ix] = expit(logits)@weights
            hc[ix] = expit(-logits)@weights
        infinite = ~np.isfinite(flat)
        if e == 0:
            h[infinite] = expit(self.tau*nodes)@weights
            hc[infinite] = expit(-self.tau*nodes)@weights
        else:
            positive = flat[infinite]*np.sign(e) > 0
            h[infinite] = np.where(positive, weights.sum(), 0.)
            hc[infinite] = np.where(positive, 0., weights.sum())
        if not np.isfinite(h).all() or not np.isfinite(hc).all() or np.any(h < 0) or np.any(hc < 0):
            raise FloatingPointError("invalid positive private rule")
        return h.reshape(x.shape), hc.reshape(x.shape)

    def log_likelihood(self, x, group, state=None, *, inner_order=None):
        """Full local observation probability, with count multiplicity retained."""
        x = np.asarray(x, dtype=float)
        h, hc = self.probabilities(x, group, inner_order=inner_order)
        states = np.arange(self.S) if state is None else np.asarray(state)
        if states.dtype.kind not in "iu" or np.any(states < 0) or np.any(states >= self.S):
            raise ValueError("integer states in 0..S-1 required")
        if state is None:
            return (log_expit(x[..., None]*self.labels)+self.log_choose
                    +xlogy(self.counts, h[..., None])+xlogy(self.M-self.counts, hc[..., None]))
        return (log_expit(x*self.labels[states])+self.log_choose[states]
                +xlogy(self.counts[states], h)+xlogy(self.M-self.counts[states], hc))

    def _level(self, outer_order, inner_order):
        key = (int(outer_order), int(inner_order))
        if key not in self._levels:
            x, w = normal_rule(outer_order)
            p, first, second = (np.empty((3, self.S)) for _ in range(3))
            for group in range(3):
                factors = np.exp(self.log_likelihood(x, group, inner_order=inner_order))
                p[group] = w@factors
                first[group] = (w*x)@factors
                second[group] = (w*x*x)@factors
            if not np.isfinite(p).all() or np.any(p <= 0):
                raise FloatingPointError("posterior normalizer underflow; no floor")
            mean, m2 = first/p, second/p
            variance = m2-mean*mean
            if not np.isfinite(mean).all() or not np.isfinite(m2).all() or np.any(variance < 0):
                raise FloatingPointError("invalid posterior moments; no repair")
            self._levels[key] = {"normalizer": p, "log_normalizer": np.log(p),
                "mean": mean, "second_moment": m2, "variance": variance,
                "observation_mass": p.sum(axis=1), "outer_order": int(outer_order),
                "inner_order": int(inner_order)}
            for value in self._levels[key].values():
                if isinstance(value, np.ndarray): value.setflags(write=False)
        return self._levels[key]

    def moments(self):
        """Same-target outer-order check plus a separate private-rule sensitivity."""
        lo, hi = self.normalizer_orders
        low = self._level(lo, self.inner_order)
        high = self._level(hi, self.inner_order)
        private = self._level(hi, self.diagnostic_inner_order)
        keys = ("normalizer", "log_normalizer", "mean", "second_moment", "variance")
        return {"schema": SCHEMA, **high,
                "same_target_low_outer": low, "diagnostic_private_level": private,
                "outer_absolute_differences": {k: np.abs(high[k]-low[k]) for k in keys},
                "private_absolute_differences": {k: np.abs(high[k]-private[k]) for k in keys},
                "automatic_normalization": False, "count_multiplicities_included": True,
                "target": "phi_times_specified_finite_private_GH_likelihood",
                "qualification": QUALIFICATION}

    def _build_envelopes(self):
        # No final v=1 coordinate: both halves use probabilities <=1/2.
        positive = np.linspace(0., self.tail_x, (self.grid_size+1)//2)
        t = np.r_[0., ndtr(-positive[::-1])]
        if not np.all(np.diff(t) > 0):
            raise FloatingPointError("non-distinct tail coordinates; change explicit grid, do not clip")
        self.tail_coordinates = t
        # A recorded fixed roundoff margin, not fitted or adaptive to observations.
        self.envelope_inflation = 1.+2.**-36
        for group in range(3):
            halves = []
            for sign in (-1, 1):
                nodes = sign*np.r_[np.inf, positive[::-1]]
                h, hc = self.probabilities(nodes, group)
                label = expit(nodes[:, None]*self.labels)
                logupper = (np.log(np.maximum(label[:-1], label[1:]))+self.log_choose
                    +xlogy(self.counts, np.maximum(h[:-1], h[1:])[:, None])
                    +xlogy(self.M-self.counts, np.maximum(hc[:-1], hc[1:])[:, None]))
                upper = np.exp(logupper)*self.envelope_inflation
                mass = np.diff(t)[:, None]*upper
                cumulative = np.cumsum(mass, axis=0)
                if (not np.isfinite(upper).all() or np.any(upper <= 0)
                        or np.any(np.diff(np.vstack((np.zeros(self.S), cumulative)), axis=0) <= 0)):
                    raise FloatingPointError("envelope mass not positive/resolvable; no discarded segments")
                halves.append({"upper": upper.T, "mass": mass.T,
                               "cumulative": cumulative.T, "total": cumulative[-1]})
            self._envelopes[group] = halves

    def proposal(self, uniforms, group, state):
        """Exact piecewise-constant envelope inverse in real arithmetic.

        Returns candidates and their rejection upper values, not posterior draws.
        Input interior uniforms are never clipped. Floating implementation is
        explicitly numerically qualified, as is the caller's finite-bit net.
        """
        u = np.asarray(uniforms, dtype=float)
        if not np.isfinite(u).all() or np.any(u <= 0) or np.any(u >= 1):
            raise ValueError("strictly interior finite proposal uniforms required; no clipping")
        group, state = _integer(group, "group", 0), _integer(state, "state", 0)
        if group > 2 or state >= self.S: raise ValueError("invalid group/state")
        halves = self._envelopes[group]
        left, right = (h["total"][state] for h in halves)
        total = left+right
        flat = u.ravel(); left_side = flat <= left/total
        output = np.empty(flat.size); bound = np.empty(flat.size)
        for side, mask in enumerate((left_side, ~left_side)):
            indices = np.flatnonzero(mask)
            if not len(indices): continue
            table = halves[side]
            # Complement of input u, not cancellation of a computed Gaussian CDF.
            target = (flat[indices] if side == 0 else 1.-flat[indices])*total
            cumulative = table["cumulative"][state]
            segment = np.searchsorted(cumulative, target, side="left")
            if np.any(segment >= len(cumulative)):
                raise FloatingPointError("proposal segment overflow; no repair")
            before = np.where(segment > 0, cumulative[np.maximum(segment-1, 0)], 0.)
            fraction = (target-before)/table["mass"][state, segment]
            if np.any(fraction < 0) or np.any(fraction > 1):
                raise FloatingPointError("proposal residual outside segment; no clipping")
            tail = self.tail_coordinates[segment]+fraction*np.diff(self.tail_coordinates)[segment]
            if np.any(tail <= 0) or np.any(tail > .5):
                raise FloatingPointError("unrepresentable posterior tail; no clipping/truncation")
            output[indices] = (1 if side == 0 else -1)*ndtri(tail)
            bound[indices] = table["upper"][state, segment]
        if not np.isfinite(output).all(): raise FloatingPointError("nonfinite inverse tail")
        return output.reshape(u.shape), bound.reshape(u.shape)

    def envelope_diagnostics(self):
        p = self.moments()["normalizer"]
        total = np.stack([sum(h["total"] for h in self._envelopes[i]) for i in range(3)])
        return {"envelope_integral": total, "expected_trials_GH_diagnostic": total/p,
                "tail_intervals_retained": True, "half_tail_coordinates": self.tail_coordinates.copy(),
                "envelope_inflation": self.envelope_inflation,
                "grid_size": self.grid_size, "tail_x": self.tail_x,
                "actual_half_segments": len(self.tail_coordinates)-1}

    def cdf(self, anchors=(-2., -1., 0., 1., 2.), *, outer_order=256, tail_x=10.):
        """Independent fixed positive GL integral on [-tail_x, anchor].

        The omitted left tail is reported, never discarded without a bound.
        Dividing by the ordinary high-order GH normalizer is numerically
        qualified; neither CDF values nor observation mass are clipped/repaired.
        """
        anchors = np.asarray(anchors, dtype=float)
        order = _integer(outer_order, "CDF GL order", 2)
        if (anchors.ndim != 1 or len(anchors) < 1 or not np.isfinite(anchors).all()
                or np.any(np.diff(anchors) <= 0)):
            raise ValueError("CDF anchors must be finite strictly increasing vector")
        if not np.isscalar(tail_x) or not np.isfinite(tail_x) or tail_x <= 0 or anchors.min() <= -tail_x:
            raise ValueError("positive finite tail_x and anchors > -tail_x required")
        tail_x = float(tail_x)
        key = (tuple(anchors), order, tail_x)
        if key not in self._cdf_cache:
            g, w = roots_legendre(order)
            if not np.isfinite(w).all() or np.any(w <= 0):
                raise FloatingPointError("positive finite CDF GL weights required")
            p = self.moments()["normalizer"]
            value = np.empty((3, self.S, len(anchors)))
            for a, anchor in enumerate(anchors):
                scale = (anchor+tail_x)/2
                x = scale*g+(anchor-tail_x)/2
                weighted_phi = scale*w*np.exp(-x*x/2)/math.sqrt(2*math.pi)
                for group in range(3):
                    ell = np.exp(self.log_likelihood(x, group))
                    value[group, :, a] = (weighted_phi@ell)/p[group]
            # h+hc equals the rule's raw weight mass in real arithmetic.
            # Max binomial mass over h/(h+hc) bounds each local probability.
            private_mass = float(normal_rule(self.inner_order)[1].sum())
            c = self.counts
            binomial_bound = np.exp(self.log_choose+xlogy(c, c/self.M)
                                   +xlogy(self.M-c, 1.-c/self.M))
            numerator_bound = np.broadcast_to(ndtr(-tail_x)*binomial_bound
                *private_mass**self.M*self.envelope_inflation, (3, self.S)).copy()
            conditional_bound = numerator_bound/p
            if not np.isfinite(value).all() or np.any(value < 0):
                raise FloatingPointError("invalid CDF quadrature; no clipping")
            saved_anchors = anchors.copy()
            for array in (value, saved_anchors, numerator_bound, conditional_bound):
                array.setflags(write=False)
            self._cdf_cache[key] = {"cdf": value, "anchors": saved_anchors,
                "outer_order": order, "quadrature": "independent_positive_Gauss_Legendre",
                "tail_x": tail_x, "tail_bound": conditional_bound,
                "left_tail_numerator_bound": numerator_bound,
                "normalizer_outer_order": self.normalizer_orders[1],
                "inner_order": self.inner_order, "automatic_normalization": False,
                "tail_qualification": "analytic_unintegrated_numerator_bound_divided_by_ordinary_GH_normalizer_not_interval_CDF_certificate",
                "qualification": QUALIFICATION}
        return self._cdf_cache[key]

    def sample_rejection(self, uniforms):
        """Caller-owned u[R,A,B,K,3,2]; no streams are created or replaced.

        Each (replicate,trial) must be an independently randomized 6K-dimensional
        net. Same base coordinates can couple all states. SE units are complete
        replicates INCLUDING all trial nets. Exhaustion fails the entire attempt.
        """
        u = None; x = accepted = candidates_by_trial = accepted_by_trial = None
        moments = None; R = A = B = K = 0
        location = {"replicate": None, "trial": None, "group": None, "state": None}
        def snapshot():
            return {"schema": SCHEMA, "x_group": x,
                "log_marginal_weight": None if moments is None else moments["log_normalizer"],
                "proposal_mean": None if moments is None else moments["mean"],
                "proposal_second_moment": None if moments is None else moments["second_moment"],
                "accepted_trial": accepted,
                "unaccepted_coordinates": None if accepted is None else int(np.sum(accepted == 0)),
                "candidates_by_trial": candidates_by_trial, "accepted_by_trial": accepted_by_trial,
                "candidate_coordinates": 0 if candidates_by_trial is None else int(candidates_by_trial.sum()),
                "candidate_coordinates_scope": "requested_log_likelihood_coordinates_registered_before_call_not_measured_CPU_completion",
                "requested_private_node_terms": 0 if candidates_by_trial is None else int(candidates_by_trial.sum())*self.inner_order,
                "caller_uniform_scalars": 0 if u is None else int(u.size), "maximum_trials": A,
                "failed_location": dict(location), "failed_location_indices": "zero_based",
                "posterior_normalizer_diagnostics": moments, "scientific_RNG_draws": 0,
                "state_order": "negative_label_count_0..M_then_positive_label_count_0..M",
                "net_dimensions_per_trial": 6*K,
                "uncertainty_unit": "complete_independent_replicate_including_all_trial_nets",
                "target": "continuous_phi_times_finite_private_GH_ell; envelope_not_target",
                "qualification": QUALIFICATION}
        try:
            u = np.asarray(uniforms, dtype=float)
            if u.ndim != 6 or min(u.shape) < 1 or u.shape[-2:] != (3, 2):
                raise ValueError("u must have shape [R,A,B,K,3,2]")
            R, A, B, K, _, _ = u.shape
            x = np.full((R, B, K, 3, self.S), np.nan)
            accepted = np.zeros(x.shape, dtype=np.int64)
            candidates_by_trial = np.zeros((R, A), dtype=np.int64)
            accepted_by_trial = np.zeros((R, A), dtype=np.int64)
            if not np.isfinite(u).all() or np.any(u <= 0) or np.any(u >= 1):
                bad = np.argwhere(~np.isfinite(u) | (u <= 0) | (u >= 1))[0]
                location.update(replicate=int(bad[0]), trial=int(bad[1]), group=int(bad[4]), state=None)
                raise ValueError("all caller-owned uniforms must be interior; no clipping")
            moments = self.moments()
            for r in range(R):
                for trial in range(A):
                    for group in range(3):
                        for state in range(self.S):
                            location.update(replicate=r, trial=trial, group=group, state=state)
                            target = x[r, :, :, group, state]
                            mask = accepted[r, :, :, group, state] == 0
                            if not mask.any(): continue
                            candidate, upper = self.proposal(u[r, trial, :, :, group, 0][mask], group, state)
                            # Requested work is committed before calling the GH kernel.
                            candidates_by_trial[r, trial] += len(candidate)
                            logaccept = self.log_likelihood(candidate, group, state)-np.log(upper)
                            if not np.isfinite(logaccept).all() or np.any(logaccept > 0):
                                raise FloatingPointError("rejection envelope violated; no acceptance clipping")
                            take = np.log(u[r, trial, :, :, group, 1][mask]) <= logaccept
                            accepted_by_trial[r, trial] += int(take.sum())
                            flat_ix = np.flatnonzero(mask)
                            target.flat[flat_ix[take]] = candidate[take]
                            accepted[r, :, :, group, state].flat[flat_ix[take]] = trial+1
                    if np.all(accepted[r] > 0): break
            if np.any(accepted == 0):
                first = np.argwhere(accepted == 0)[0]
                location.update(replicate=int(first[0]), trial=A-1, group=int(first[3]), state=int(first[4]))
                raise RejectionExhausted(snapshot())
            location.update(replicate=None, trial=None, group=None, state=None)
            return snapshot()
        except RejectionExhausted:
            raise
        except Exception as error:
            result = snapshot()
            result["failure_type"] = type(error).__name__
            result["failure_message"] = str(error)
            raise RejectionFailure(f"sampling failed at {location}: {error}", result) from error
