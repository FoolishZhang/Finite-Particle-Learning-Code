"""Caller-owned local simulation and deterministic defensive Gaussian mixtures.

No import-time RNG, integration, training or file writes. The proposal is shared
across the three homogeneous groups and never receives candidate/teacher C.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.special import expit, logsumexp, ndtri

SCHEMA = "learned_local_proposal_v1"
M = 8
STATES = 18
DEFAULT_CONSTRAINTS = {"weight_bounds": [.02, .98], "mean_bounds": [-5., 5.],
                       "std_bounds": [.15, 3.], "minimum_state_samples": 8}


def require(test, message):
    if not test:
        raise ValueError(message)


def _observations(x, state):
    x, state = np.asarray(x, float), np.asarray(state)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), "finite nonempty scalar training observations required")
    require(state.shape == x.shape and state.dtype.kind in "iu" and np.all((state >= 0) & (state < STATES)), "integer local states 0..17 required")
    return x, state


def _interior(u):
    require(np.isfinite(u).all() and np.all(u > 0) and np.all(u < 1), "strictly interior finite uniforms required; no clipping")


def simulate_local(n, *, active_rng, private_rng, label_rng, item_rng,
                   epsilon=1.4 / math.sqrt(8), tau=3., M=8):
    """Physical finite-M local model; all four RNG streams belong to caller.

    Explicit private variables are generated, not GH-approximated probabilities.
    A caller must save all returned arrays and its own stream states/identities.
    """
    require(isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n > 0, "positive integer n required")
    require(M == 8 and np.isfinite(epsilon) and np.isfinite(tau) and tau >= 0, "declared M8 local channel required")
    streams = (active_rng, private_rng, label_rng, item_rng)
    require(len({id(s) for s in streams}) == 4, "four distinct caller RNG stream objects required")
    x = np.asarray(active_rng.standard_normal(n), float)
    private_z = np.asarray(private_rng.standard_normal((n, M)), float)
    label_u = np.asarray(label_rng.random(n), float)
    item_u = np.asarray(item_rng.random((n, M)), float)
    require(x.shape == (n,) and private_z.shape == (n, M) and label_u.shape == (n,) and item_u.shape == (n, M), "caller RNG returned incorrect shape")
    require(np.isfinite(x).all() and np.isfinite(private_z).all(), "nonfinite caller Gaussian draws")
    require(np.isfinite(label_u).all() and np.isfinite(item_u).all() and
            np.all((label_u >= 0) & (label_u < 1)) and np.all((item_u >= 0) & (item_u < 1)), "simulation uniforms must lie in [0,1)")
    labels = np.where(label_u < expit(x), 1, -1).astype(np.int8)
    counts = (item_u < expit(epsilon * x[:, None] + tau * private_z)).sum(axis=1).astype(np.int64)
    state = counts + ((labels + 1) // 2) * (M + 1)
    return {"x": x, "counts": counts, "labels": labels, "state": state,
            "private_z": private_z, "label_u": label_u, "item_u": item_u}


class LocalProposal:
    """Normalized defensive raw2G or explicit reflection-average4G proposal."""
    def __init__(self, raw_parameters, *, kind="raw", defense=.05, source=None):
        require(kind in ("raw", "sym"), "kind must be raw or sym")
        require(np.isscalar(defense) and np.isfinite(defense) and 0 < defense < 1, "fixed defensive mass in (0,1) required")
        self.kind, self.defense = kind, float(defense)
        self.source = copy.deepcopy(source)
        raw = {name: np.array(raw_parameters[name], dtype=float, copy=True) for name in ("weights", "means", "stds")}
        require(all(a.shape == (STATES, 2) and np.isfinite(a).all() for a in raw.values()), "finite 18state two-Gaussian parameter arrays required")
        require(np.all(raw["weights"] > 0) and np.allclose(raw["weights"].sum(axis=1), 1, rtol=0, atol=2e-15), "positive normalized nondefense weights required")
        require(np.all(raw["stds"] > 0), "positive standard deviations required")
        for name, key in (("weights", "weight_bounds"), ("means", "mean_bounds"), ("stds", "std_bounds")):
            low, high = DEFAULT_CONSTRAINTS[key]
            require(np.all(raw[name] >= low) and np.all(raw[name] <= high),
                    "raw parameters outside the predeclared class constraints: " + name)
        self.raw_parameters = raw
        if kind == "raw":
            parameters = {name: array.copy() for name, array in raw.items()}
        else:
            parameters = {name: np.empty((STATES, 4)) for name in raw}
            for state in range(9):
                mirror = 17 - state
                parameters["weights"][state] = .5 * np.r_[raw["weights"][state], raw["weights"][mirror]]
                parameters["means"][state] = np.r_[raw["means"][state], -raw["means"][mirror]]
                parameters["stds"][state] = np.r_[raw["stds"][state], raw["stds"][mirror]]
                parameters["weights"][mirror] = parameters["weights"][state]
                parameters["means"][mirror] = -parameters["means"][state]
                parameters["stds"][mirror] = parameters["stds"][state]
        self.parameters = parameters
        for array in [*raw.values(), *parameters.values()]:
            array.setflags(write=False)

    def _components(self, state):
        weights = np.r_[self.defense, (1 - self.defense) * self.parameters["weights"][state]]
        means = np.r_[0., self.parameters["means"][state]]
        stds = np.r_[1., self.parameters["stds"][state]]
        return weights, means, stds

    def logpdf(self, x, state=None):
        """Broadcast x/state; absent state appends the18-state density axis."""
        x = np.asarray(x, float)
        require(np.isfinite(x).all(), "finite density arguments required")
        if state is None:
            x, state = np.broadcast_arrays(x[..., None], np.arange(STATES))
        else:
            state = np.asarray(state)
            require(state.dtype.kind in "iu" and np.all((state >= 0) & (state < STATES)), "integer states0..17 required")
            x, state = np.broadcast_arrays(x, state)
        if self.kind == "sym":
            x = np.where(state <= 8, x, -x)
            state = np.minimum(state, 17 - state)
        weights = self.parameters["weights"][state]
        means = self.parameters["means"][state]
        stds = self.parameters["stds"][state]
        component_log = np.log1p(-self.defense) + np.log(weights) - np.log(stds) - .5 * ((x[..., None] - means) / stds) ** 2 - .5 * math.log(2 * math.pi)
        defensive_log = math.log(self.defense) - .5 * x * x - .5 * math.log(2 * math.pi)
        result = logsumexp(np.concatenate((defensive_log[..., None], component_log), axis=-1), axis=-1)
        require(np.isfinite(result).all(), "nonfinite density result; no repair")
        return result

    def moments(self):
        p = self.parameters
        mean = (1 - self.defense) * np.sum(p["weights"] * p["means"], axis=1)
        second = self.defense + (1 - self.defense) * np.sum(p["weights"] * (p["stds"] ** 2 + p["means"] ** 2), axis=1)
        if self.kind == "sym":
            for state in range(9):
                mean[17-state] = -mean[state]
                second[17-state] = second[state]
        return {"mean": mean, "second_moment": second, "variance": second - mean * mean,
                "normalizer": np.ones(STATES), "analytic": True}

    def sample_from_uniforms(self, uniforms):
        """u[R,B,K,3,2] -> x[R,B,K,3,18], no internal stream or clipping.

        For sym, only nine canonical states are sampled and reflected exactly.
        State coupling is not an additional independent numerical replicate.
        """
        u = np.asarray(uniforms, float)
        require(u.ndim == 5 and min(u.shape) >= 1 and u.shape[-2:] == (3, 2), "uniform shape must be[R,B,K,3,2]")
        _interior(u)
        z = ndtri(u[..., 1])
        require(np.isfinite(z).all(), "nonfinite inverse-normal coordinates")
        output = np.empty(u.shape[:-1] + (STATES,))
        for state in range(9 if self.kind == "sym" else STATES):
            weights, means, stds = self._components(state)
            # At an exact cumulative boundary select the following component.
            component = np.sum(u[..., 0, None] >= np.cumsum(weights)[:-1], axis=-1)
            output[..., state] = means[component] + stds[component] * z
            if self.kind == "sym":
                output[..., 17-state] = -output[..., state]
        require(np.isfinite(output).all(), "nonfinite mixture sample; no repair")
        return output

    def to_record(self):
        return {"schema": SCHEMA, "kind": self.kind, "defense": self.defense,
                "raw_parameters": {k: v.tolist() for k, v in self.raw_parameters.items()},
                "parameters": {k: v.tolist() for k, v in self.parameters.items()}, "source": copy.deepcopy(self.source),
                "shared_homogeneous_groups": 3,
                "construction": "raw_two_Gaussian_plus_fixed_standard_normal" if self.kind == "raw" else
                "same_raw_density_explicit_reflection_average_four_Gaussian_plus_same_defense"}


class TrainingFailure(RuntimeError):
    def __init__(self, message, partial):
        self.partial = partial
        super().__init__(message)


def fit_path(x, state, *, epochs=80, checkpoints=(0, 5, 20, 80), defense=.05, constraints=None):
    """Deterministic constrained EM; save every epoch, never select a seed."""
    x, state = _observations(x, state)
    rules = copy.deepcopy(DEFAULT_CONSTRAINTS if constraints is None else constraints)
    require(rules == DEFAULT_CONSTRAINTS, "this version uses the explicitly fixed class constraints")
    require(isinstance(epochs, int) and not isinstance(epochs, bool) and epochs >= 0, "nonnegative integer epochs required")
    checkpoints = tuple(checkpoints)
    require(checkpoints and tuple(sorted(set(checkpoints))) == checkpoints and all(isinstance(e, int) and 0 <= e <= epochs for e in checkpoints), "ordered unique in-range checkpoints required")
    require(np.isfinite(defense) and 0 < defense < 1, "fixed defense mass required")
    local = [x[state == s] for s in range(STATES)]
    counts = np.array([len(values) for values in local])
    result = {"schema": SCHEMA, "kind": "raw_training_path", "defense": float(defense), "constraints": rules,
        "state_sample_counts": counts.tolist(), "training_rows": len(x), "epochs_requested": epochs,
        "checkpoint_epochs": list(checkpoints), "checkpoints": {}, "parameter_path": [], "loglikelihood_trace": [],
        "completed_state_updates": 0, "scientific_RNG_inside_fit": 0,
        "initializer": "empirical_linear_25_75_percentiles; population_empirical_SD; fixed_class_bounds; equal_learned_weights"}
    if np.any(counts < rules["minimum_state_samples"]):
        raise TrainingFailure("a local state has fewer than8 samples; no replenishment", result)
    params = {"weights": np.full((STATES, 2), .5), "means": np.empty((STATES, 2)), "stds": np.empty((STATES, 2))}
    for s, values in enumerate(local):
        params["means"][s] = np.clip(np.quantile(values, [.25, .75], method="linear"), *rules["mean_bounds"])
        params["stds"][s] = np.clip(values.std(ddof=0), *rules["std_bounds"])
    def save(epoch):
        proposal = LocalProposal(params, defense=defense)
        ll = [float(proposal.logpdf(values, s).sum()) for s, values in enumerate(local)]
        snapshot = {name: array.tolist() for name, array in params.items()}
        result["parameter_path"].append({"epoch": epoch, "raw_parameters": snapshot})
        result["loglikelihood_trace"].append({"epoch": epoch, "total": float(sum(ll)), "per_state": ll,
                                              "completed_state_updates": result["completed_state_updates"]})
        if epoch in checkpoints:
            result["checkpoints"][str(epoch)] = copy.deepcopy(snapshot)
    try:
        save(0)
        for epoch in range(1, epochs + 1):
            for s, values in enumerate(local):
                w, mu, sigma = params["weights"][s], params["means"][s], params["stds"][s]
                logits = np.column_stack((math.log(defense) - .5 * values ** 2,
                    np.log1p(-defense) + np.log(w) - np.log(sigma) - .5 * ((values[:, None] - mu) / sigma) ** 2))
                responsibility = np.exp(logits - logsumexp(logits, axis=1)[:, None])[:, 1:]
                mass = responsibility.sum(axis=0)
                require(np.isfinite(responsibility).all() and np.all(mass > 0), "degenerate learned responsibility mass; no replacement")
                new_mu = np.clip((responsibility * values[:, None]).sum(axis=0) / mass, *rules["mean_bounds"])
                new_std = np.clip(np.sqrt((responsibility * (values[:, None] - new_mu) ** 2).sum(axis=0) / mass), *rules["std_bounds"])
                w1 = float(np.clip(mass[0] / mass.sum(), *rules["weight_bounds"]))
                params["weights"][s], params["means"][s], params["stds"][s] = [w1, 1-w1], new_mu, new_std
                result["completed_state_updates"] += 1
            save(epoch)
        result["status"] = "completed_fixed_epochs"
        return result
    except Exception as error:
        result["last_partially_updated_parameters"] = {name: array.tolist() for name, array in params.items()}
        raise TrainingFailure(str(error), result) from error


def select_checkpoint(trainpath, validation_x, validation_state):
    """Choose one common epoch by mean(raw NLL,sym NLL); earliest exact tie."""
    x, state = _observations(validation_x, validation_state)
    require(trainpath.get("status") == "completed_fixed_epochs", "complete raw training path required")
    rows = []
    for epoch in sorted(map(int, trainpath["checkpoints"])):
        parameters = trainpath["checkpoints"][str(epoch)]
        scores = {}
        for kind in ("raw", "sym"):
            proposal = LocalProposal(parameters, kind=kind, defense=trainpath["defense"])
            scores[kind + "_mean_NLL"] = float(-proposal.logpdf(x, state).mean())
        rows.append({"epoch": epoch, **scores, "paired_mean_NLL": .5 * (scores["raw_mean_NLL"] + scores["sym_mean_NLL"])})
    require(rows, "at least one declared checkpoint required")
    selected = min(rows, key=lambda row: (row["paired_mean_NLL"], row["epoch"]))
    return {"schema": SCHEMA, "selected_epoch": selected["epoch"], "scores": rows, "validation_rows": len(x),
        "selection_rule": "mean_raw_and_its_reflection_average_validation_NLL; earliest_exact_tie",
        "raw_parameters": copy.deepcopy(trainpath["checkpoints"][str(selected["epoch"])]),
        "defense": trainpath["defense"], "same_checkpoint_for_raw_and_sym": True,
        "candidate_teacher_or_target_comparison_used": False}
