"""Read-only historical inputs and numerical kernels for budget sensitivity.

Imports never integrate, sample or fit. Historical runners and their gates are
unchanged; this module calls only their numerical/file primitives. No generator
of observations is exposed. A new contract must bind every source and input.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.stats import qmc

import active_reference_run as old
from active_reference_optimize import gradient_mapping
from informative_projection import project_coordinates
from group_posterior_marginal import GroupPosteriorMarginal
from group_posterior_objective import empirical_patterns
from group_posterior_quadratic import QuadraticPreparedGroupBanks, EXPONENTS, statistics

SCHEMA = "local_shrinkage_budget_sensitivity_v1"
REPO = Path(__file__).resolve().parents[1]
METHODS = ("integrated", "group8", "ridge")
read, write_json, write_npz, sha, serial = old.read, old.write_json, old.write_npz, old.sha, old.serial


def require(test, message):
    if not test:
        raise ValueError(message)


def resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (REPO / p).resolve()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    for value in result.values():
        value.setflags(write=False)
    return result


def historical_bank(folder, kind):
    require(kind in ("FIT_BANKS", "EVALUATION_BANKS"), "historical bank role required")
    R = 2 if kind == "FIT_BANKS" else 4
    data = arrays(folder / kind / "SAMPLES.npz")
    require(data["x_group"].shape == data["accepted_trial"].shape == (R, 512, 8, 3, 18),
            "original complete R/B/K bank shape differs")
    require(np.all((data["accepted_trial"] >= 1) & (data["accepted_trial"] <= 8)), "incomplete historical rejection samples")
    for name in ("log_marginal_weight", "proposal_mean", "proposal_second_moment"):
        require(data[name].shape == (3, 18), "historical proposal-moment shape differs")
    return data


def historical_case(contract, case_id):
    """Only saved observations/outputs; never read GENERATION_PROVENANCE."""
    original = read(resolve(contract["original_contract"]))
    matches = [c for c in original["cases"] if c["case_id"] == case_id]
    require(len(matches) == 1 and matches[0]["cell_id"] == "strong_N2048", "one original strong case required")
    folder = resolve(contract["old_run_root"]) / f"case_{case_id:05d}"
    visible = arrays(folder / "VISIBLE_DATA.npz")
    require(set(visible) == {"counts", "observed_labels"}, "only actual observations allowed")
    require(visible["counts"].shape == visible["observed_labels"].shape == (2048, 3), "original complete N2048 required")
    outputs = read(folder / "FIT_COMPLETE.json")["outputs"]
    require(len(outputs) == 6 and {(p["method"], p["initialization_index"]) for p in outputs} ==
            {(m, i) for m in METHODS for i in (0, 1)}, "all six original outputs required")
    for p in outputs:
        f = read(folder / ("FIT_" + p["method"]) / f"init{p['initialization_index']}_FIT.json")
        require(p["case_id"] == case_id and np.array_equal(p["coordinates"], f["coordinates"]), "P0 output and fit disagree")
    return matches[0], folder, visible, [{**p, "setting": "P0", "point_id": "P0_" + p["point_id"]} for p in outputs]


class Budget(old.Budget):
    """Requested/completed work counts, common wall limit, optional RSS limit."""
    def __init__(self, contract):
        super().__init__(contract)
        self.limits = contract.get("resources", {})
        self.observed_rss_bytes = 0
        self.output_directory = None
        self.last_disk_check = 0.
        self.observed_output_bytes = 0

    def check(self, name=None, amount=0):
        super().check(name, amount)
        # Optional psutil probe is operational monitoring, never a scientific RNG.
        if self.limits.get("max_rss_bytes"):
            import psutil
            rss = psutil.Process().memory_info().rss
            self.observed_rss_bytes = max(self.observed_rss_bytes, rss)
            require(rss <= self.limits["max_rss_bytes"], "frozen process RSS limit exceeded")
        if self.limits.get("max_output_bytes") and self.output_directory is not None and time.perf_counter() - self.last_disk_check >= 2:
            self.observed_output_bytes = sum(p.stat().st_size for p in self.output_directory.rglob("*") if p.is_file())
            self.last_disk_check = time.perf_counter()
            require(self.observed_output_bytes <= self.limits["max_output_bytes"], "frozen output storage limit exceeded")

    def snapshot(self):
        return {**super().snapshot(), "maximum_observed_rss_bytes": self.observed_rss_bytes,
                "last_observed_output_bytes": self.observed_output_bytes,
                "resource_monitor_scope": "RSS at checks; output size at checks separated by at least two seconds, not an OS quota",
                "new_observations": 0, "new_teacher_draws": 0,
                "uncertainty_unit": "complete_independent_randomization_not_B_K_or_pattern"}


def prepared(samples, patterns, *, R=None, B=None, moments=None):
    x = samples["x_group"]
    x = x[:R, :B] if R is not None or B is not None else x
    return QuadraticPreparedGroupBanks(
        x, samples["log_marginal_weight"] if moments is None else moments["log_normalizer"],
        patterns["state_indices"], patterns["weights"],
        proposal_kind="unchanged_finite_private_GH256_group_rejection",
        proposal_mean=samples["proposal_mean"] if moments is None else moments["mean"],
        proposal_second_moment=samples["proposal_second_moment"] if moments is None else moments["second_moment"])


def group_value(bank, coordinates, budget, contract):
    terms = bank.R * bank.B * bank.K * len(bank.weights)
    if bank._origin_hessian_cache is None:
        budget.add("control_construction_particle_pattern_terms", terms)
        bank.origin_hessian_control(pattern_chunk=contract["pattern_chunk"])
    budget.add("group_objective_calls")
    budget.add("particle_pattern_terms", terms)
    result = bank.evaluate(old.legacy.correlation_matrix(coordinates), control="uniform_logratio",
                           pattern_chunk=contract["pattern_chunk"])
    budget.check()
    return result


def save_result(prefix, result):
    write_npz(Path(prefix).with_suffix(".npz"), {k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    write_json(Path(prefix).with_suffix(".json"), {k: v for k, v in result.items() if not isinstance(v, np.ndarray)})


def moment_contractions(patterns, level):
    """Exact finite sums over the same empirical pattern weights."""
    states, weights = patterns["state_indices"], patterns["weights"]
    mu = np.stack([level["mean"][g, states[:, g]] for g in range(3)], axis=-1)
    nu = np.stack([level["second_moment"][g, states[:, g]] for g in range(3)], axis=-1)
    Q = np.einsum("p,pi,pj->ij", weights, mu, mu)
    np.fill_diagonal(Q, weights @ nu)
    pairs = ((0, 1), (0, 2), (1, 2))
    scores = np.stack([mu[:, i] * mu[:, j] for i, j in pairs], axis=-1)
    covariance = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            product = np.ones(len(states))
            for g, power in enumerate(EXPONENTS[i] + EXPONENTS[j]):
                if power == 1:
                    product *= mu[:, g]
                elif power == 2:
                    product *= nu[:, g]
            covariance[i, j] = weights @ (product - scores[:, i] * scores[:, j])
    logZ = sum(weights @ level["log_normalizer"][g, states[:, g]] for g in range(3))
    return Q, covariance, float(logZ)


def recenter_fixed_result(result, coordinates, patterns, original_level, new_level, K):
    """Exact algebraic change of numerical moments on an identical sample bank.

    The proposal's finite private GH rule and every accepted particle stay fixed.
    Only local normalizers and control centers change. No new objective sampling.
    Corrections are identical in each replicate, so between-replicate SE stays
    the same; finite-precision summation may differ from a full reevaluation.
    """
    Q0, cov0, z0 = moment_contractions(patterns, original_level)
    Q1, cov1, z1 = moment_contractions(patterns, new_level)
    c = np.asarray(coordinates, float)
    inv = np.linalg.inv(old.legacy.correlation_matrix(c))
    dQ, dD = Q1 - Q0, (1 - 1 / K) * (cov1 - cov0)
    dv = -(z1 - z0) + .5 * np.sum((inv - np.eye(3)) * dQ.T) - .5 * c @ dD @ c
    middle = inv @ dQ @ inv
    dg = -middle[[0, 0, 1], [1, 2, 2]] - dD @ c
    values = np.asarray(result["replicate_values"]) + dv
    gradients = np.asarray(result["replicate_gradients"]) + dg
    return {**statistics(values, gradients), "value_shift": float(dv), "gradient_shift": dg,
            "moment_change_only": True, "same_particles_same_numerical_private_proposal": True,
            "evaluation_method": "analytic_difference_of_same_controlled_scalar",
            "raw_value_shift": float(-(z1 - z0)), "raw_gradient_shift": np.zeros(3)}


def make_marginal(contract, budget, destination, *, levels):
    destination.mkdir(exist_ok=False)
    m = contract["model"]
    cfg = contract["marginal"]
    # Envelope requests for each sign/group; infinite endpoints need no GH sum.
    budget.add("marginal_private_node_terms", 6 * ((cfg["grid_size"] + 1) // 2) * cfg["inner_order"])
    sampler = GroupPosteriorMarginal(M=m["M"], epsilon=m["epsilon"], tau=m["tau"], **cfg)
    result = {}
    for order in levels:
        budget.add("marginal_private_node_terms", 3 * order * cfg["inner_order"])
        value = sampler._level(order, cfg["inner_order"])
        result[order] = value
        save_result(destination / f"MOMENTS_outer{order}_inner{cfg['inner_order']}", value)
    budget.check()
    return sampler, result


def make_banks(contract, sampler, settings, role, case_id, destination, budget, *, old_prefix=None):
    """Same legacy entropy and 6K nets, with explicit prefix identity checks."""
    destination.mkdir(exist_ok=False)
    R, B, K = settings["R"], settings["B"], contract["model"]["K"]
    A, bits = contract["sampling"]["maximum_trials"], contract["sampling"]["bits"]
    role_id = {"fit": 10, "evaluation": 20}[role]
    seed_family = contract["seed_family"] if role == "fit" else contract["evaluation_seed_family"]
    size = R * A * B * K * 6
    budget.check("qmc_uniform_scalars", size)
    budget.check("qmc_trial_nets", R * A)
    budget.check("posterior_sample_coordinates", R * B * K * 3 * sampler.S)
    u = np.full((R, A, B, K, 3, 2), np.nan)
    states = []
    try:
        for rep in range(R):
            for trial in range(A):
                entropy = [seed_family, role_id, case_id, K, rep, trial]
                rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
                net = qmc.Sobol(d=6 * K, scramble=True, bits=bits, rng=rng)
                value = net.random_base2(int(np.log2(B)))
                budget.counts["qmc_uniform_scalars"] += value.size
                budget.counts["qmc_trial_nets"] += 1
                u[rep, trial] = value.reshape(B, K, 3, 2)
                states.append({"replicate": rep, "trial": trial, "entropy": entropy,
                               "rng_state_after_net_generation": rng.bit_generator.state})
                require(np.all(value > 0) and np.all(value < 1), "noninterior uniform; no clipping")
                budget.check()
        write_npz(destination / "UNIFORMS.npz", {"uniforms": u})
        write_json(destination / "RNG.json", {"R": R, "B": B, "K": K, "role": role,
                   "seed_family": seed_family, "role_id": role_id,
                   "independent_from_original_evaluation_family": role == "evaluation" and seed_family != contract["seed_family"],
                   "states": states})
        if old_prefix is not None:
            previous = arrays(old_prefix / "UNIFORMS.npz")["uniforms"]
            same = np.array_equal(u[:previous.shape[0], :, :previous.shape[2]], previous)
            write_json(destination / "UNIFORM_PREFIX.json", {"old_path": str(old_prefix / "UNIFORMS.npz"),
                "old_sha256": sha(old_prefix / "UNIFORMS.npz"), "shape": previous.shape, "identical": same})
            require(same, "old Sobol prefix differs; no independent-bank fallback")
            del previous
        worst = R * A * B * K * 3 * sampler.S
        budget.check("rejection_candidate_coordinates", worst)
        budget.check("rejection_private_node_terms", worst * sampler.inner_order)
        # moments() inside the preserved sampler also requests low and private diagnostics.
        for outer, inner in ((128, 256), (192, 256), (192, 192)):
            if (outer, inner) not in sampler._levels:
                budget.add("marginal_private_node_terms", 3 * outer * inner)
        result = sampler.sample_rejection(u)
        budget.counts["rejection_candidate_coordinates"] += result["candidate_coordinates"]
        budget.counts["rejection_private_node_terms"] += result["candidate_coordinates"] * sampler.inner_order
        budget.counts["posterior_sample_coordinates"] += result["x_group"].size
        save_result(destination / "SAMPLES", result)
        # A completed draw must survive a wall/RSS/counter exception raised by
        # the following check. Never lose accepted particles after paying for them.
        budget.check("posterior_sample_coordinates")
        if old_prefix is not None:
            previous = arrays(old_prefix / "SAMPLES.npz")
            checks = {}
            for key in ("x_group", "accepted_trial"):
                shape = previous[key].shape
                checks[key] = np.array_equal(result[key][:shape[0], :shape[1]], previous[key])
            for key in ("log_marginal_weight", "proposal_mean", "proposal_second_moment"):
                checks[key] = np.array_equal(result[key], previous[key])
            write_json(destination / "SAMPLE_PREFIX.json", {"old_path": str(old_prefix / "SAMPLES.npz"),
                       "old_sha256": sha(old_prefix / "SAMPLES.npz"), "checks": checks, "identical": all(checks.values())})
            require(all(checks.values()), "accepted-particle or control-moment prefix differs; stop")
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        budget.check()
        return result
    except Exception as error:
        if not (destination / "UNIFORMS.npz").exists():
            write_npz(destination / "FAILED_PARTIAL_UNIFORMS.npz", {"uniforms": u})
        partial = getattr(error, "result", None)
        secondary_budget_error = None
        if partial is not None:
            budget.counts["rejection_candidate_coordinates"] += partial["candidate_coordinates"]
            budget.counts["rejection_private_node_terms"] += partial["candidate_coordinates"] * sampler.inner_order
            accepted = partial.get("accepted_trial")
            budget.counts["posterior_sample_coordinates"] += int(np.sum(accepted > 0)) if accepted is not None else 0
            save_result(destination / "FAILED_PARTIAL_SAMPLES", partial)
            try:
                budget.check("posterior_sample_coordinates")
            except Exception as secondary:
                # Preserve the sampler's original exception; a second resource
                # failure is evidence recorded alongside it, not its replacement.
                secondary_budget_error = repr(secondary)
        write_json(destination / "FAILED.json", {"error": repr(error), "traceback": traceback.format_exc(),
                   "secondary_budget_error": secondary_budget_error,
                   "states": states, "budget": budget.snapshot(), "automatic_retry": False})
        raise


def solve_recorded(objective, initial, destination, *, floor, budget, **settings):
    """Legacy solver arithmetic with append-only accepted checkpoints.

    The successful numerical trajectory matches active_reference_optimize.solve.
    A hard failure returns the last accepted point (or explicitly no output),
    never the lowest attempted point. A checkpoint is written before each new
    iteration, so it also survives an exception or process interruption.
    """
    destination.mkdir(exist_ok=False)
    c = np.asarray(initial, float).copy()
    require(c.shape == (3,) and np.linalg.eigvalsh(old.legacy.correlation_matrix(c))[0] >= floor - 1e-12,
            "initial coordinates outside domain")
    require(set(settings) == {"initial_step", "mapping_tolerance", "max_iterations", "max_evaluations",
            "max_backtracks", "armijo", "step_bounds"}, "explicit complete solver settings required")
    evaluations, trace = [], []
    started = time.perf_counter()
    accepted = None
    error = None
    status = "max_iterations"
    with (destination / "OBJECTIVE_TRACE.jsonl").open("x", encoding="utf-8") as log, \
         (destination / "ACCEPTED_TRACE.jsonl").open("x", encoding="utf-8") as accepted_log:
        def checkpoint_accepted(iteration, kind):
            nonlocal accepted
            # Acceptance is independent of whether the subsequent stationarity
            # diagnostic succeeds. Persist the actual accepted c/f/g first.
            accepted = {"iteration": iteration, "coordinates": c.tolist(), "value": f,
                "gradient": g.tolist(), "mapping_norm_inf": None, "min_eigenvalue": None,
                "evaluation_count": len(evaluations)}
            accepted_log.write(json.dumps({"event": kind, **accepted}, allow_nan=False) + "\n")
            accepted_log.flush()

        def evaluate(x):
            budget.check()
            if len(evaluations) >= settings["max_evaluations"]:
                return None
            budget.add("optimizer_objective_attempts")
            r = objective(x)
            f, g = float(r["value"]), np.asarray(r["gradient"], float)
            require(np.isfinite(f) and g.shape == (3,) and np.isfinite(g).all(), "nonfinite objective")
            evaluations.append({"coordinates": x.tolist(), "value": f, "gradient": g.tolist()})
            log.write(json.dumps({**evaluations[-1], "budget": budget.snapshot()}, default=serial, allow_nan=False) + "\n")
            log.flush()
            return f, g
        try:
            f, g = evaluate(c)
            checkpoint_accepted(0, "initial_objective_accepted")
            alpha = float(settings["initial_step"])
            for iteration in range(settings["max_iterations"] + 1):
                mapping = gradient_mapping(c, g, floor=floor)
                accepted = {"iteration": iteration, "coordinates": c.tolist(), "value": f,
                    "gradient": g.tolist(), "mapping_norm_inf": mapping["norm_inf"],
                    "min_eigenvalue": float(np.linalg.eigvalsh(old.legacy.correlation_matrix(c))[0]),
                    "evaluation_count": len(evaluations)}
                trace.append(accepted)
                accepted_log.write(json.dumps({"event": "stationarity_diagnostic", **accepted}, allow_nan=False) + "\n")
                accepted_log.flush()
                if mapping["norm_inf"] <= settings["mapping_tolerance"]:
                    status = "mapping_tolerance"
                    break
                if iteration == settings["max_iterations"]:
                    break
                took = False
                for bt in range(settings["max_backtracks"]):
                    projection = project_coordinates(c - alpha * g, delta=floor)
                    trial = np.asarray(projection["coordinates"])
                    displacement = trial - c
                    slope = float(g @ displacement)
                    candidate = evaluate(trial)
                    if candidate is None:
                        status = "evaluation_cap"
                        break
                    trial_f, trial_g = candidate
                    evaluations[-1].update(backtrack=bt, step=alpha, projected_slope=slope,
                                           projection_residual=float(projection["residual"]))
                    if trial_f <= f + settings["armijo"] * slope + 1e-13:
                        took = True
                        previous_g = g
                        c, f, g = trial.copy(), trial_f, trial_g
                        evaluations[-1]["accepted"] = True
                        checkpoint_accepted(iteration + 1, "armijo_accepted")
                        y = trial_g - previous_g
                        sy = float(displacement @ y)
                        next_step = float(displacement @ displacement) / sy if sy > 1e-18 else settings["initial_step"]
                        alpha = float(np.clip(next_step, *settings["step_bounds"]))
                        break
                    evaluations[-1]["accepted"] = False
                    alpha *= .5
                if not took:
                    if status != "evaluation_cap":
                        status = "line_search_limit"
                    break
        except Exception:
            error = traceback.format_exc()
            status = "hard_failure_last_accepted" if accepted is not None else "hard_failure_no_output"
    fit = {"schema": SCHEMA, "status": status, "has_valid_output": accepted is not None,
        "coordinates": None if accepted is None else accepted["coordinates"],
        "value": None if accepted is None else accepted["value"],
        "gradient": None if accepted is None else accepted["gradient"],
        "mapping_norm_inf": None if accepted is None else accepted["mapping_norm_inf"],
        "trace": trace, "evaluations": evaluations, "objective_calls": len(evaluations),
        "seconds": time.perf_counter() - started, "error": error,
        "output_rule": "last_accepted_per_original_start_no_risk_selection"}
    write_json(destination / "FIT.json", fit)
    return fit


def common_risk(contract, case, points, budget, destination):
    """Always both declared orders, including P0; no fresh risk observations."""
    destination.mkdir(exist_ok=False)
    engines = [old.engine(contract, orders) for orders in contract["risk_orders"]]
    teacher_laws, cached, rows = {}, {}, []
    tol = min(contract["qualification"]["kl_tol"], contract["qualification"]["scaled_risk_tol"] / case["N"])
    cfg = {**contract, "qualification": {**contract["qualification"], "kl_tol": tol}}
    for p in points:
        if p.get("coordinates") is None:
            rows.append({**p, "evaluable": False, "precision_qualified": False})
            continue
        point = {**p, "C": old.legacy.correlation_matrix(p["coordinates"])}
        key = np.asarray(point["coordinates"], float).tobytes()
        if key not in cached:
            result = old.risk_evaluation(cfg, budget, {"Cstar": case["Cstar"]}, point, engines, teacher_laws, destination)
            cached[key] = point["point_id"], result
        origin, result = cached[key]
        row = {**p, **result, "evaluable": True, "scaled_joint_kl": case["N"] * result["joint_kl"],
               "risk_reused_from": origin, "risk_kl_tolerance": tol, "included_without_diagnostic_filter": True}
        write_json(destination / (point["point_id"] + "_RISK.json"), row)
        rows.append(row)
    write_json(destination / "COMPLETE.json", {"rows": rows, "ordinary_order_check_not_error_certificate": True})
    return rows
