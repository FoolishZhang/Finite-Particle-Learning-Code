"""Freeze-gated development/validation driver for integrated and finite references.

Imports never draw or fit. Empirical fit/replay inputs are only actual counts/Y;
teacher latents are saved separately for generation provenance. Independent
Sobol scrambles, each point of dimension 3*K, define numerical-error units.
All starts retain their last accepted output; no best-start/risk selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.special import expit, ndtri
from scipy.stats import qmc

import active_reference_likelihood as integrated
import active_reference_finite as finite
import active_reference_optimize as optimize
import active_reference_sgd as original_sgd
import competition_core as competition
import selection_finite_core as observations
import signal_proposal_core as legacy

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "active_reference_run_v1"
REQUIRED_SOURCES = tuple("script/" + name for name in (
    "active_reference_run.py", "active_reference_likelihood.py", "active_reference_finite.py",
    "active_reference_optimize.py", "active_reference_sgd.py", "teacher_endpoint_core.py",
    "teacher_endpoint_training.py", "competition_core.py", "selection_finite_core.py",
    "signal_proposal_core.py", "informative_projection.py"))
ROLES = {"teacher": 1, "data": 2, "fitQMC": 10, "evalQMC": 20, "SGDindices": 30, "SGDactive": 31}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=serial, allow_nan=False)
        stream.write("\n")


def write_npz(path, values):
    with Path(path).open("xb") as stream: np.savez(stream, **values)


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def require(condition, message):
    if not condition: raise ValueError(message)


def reference_schedule(phase):
    entries = [(method, "empirical", N) for N in (256, 2048) for method in ("integrated", "finite8")]
    if phase == "development":
        return entries + [(method, "population", None) for method in ("integrated", "finite8")]
    return [("integrated", "empirical", 256), ("finite8", "empirical", 256), ("integrated", "empirical", 2048)]


def validate_contract(contract, *, verify_sources=True):
    require(not contract.get("draft_not_authorized_to_run", False), "draft contract is not executable")
    require(contract.get("schema") == SCHEMA and contract.get("phase") in ("development", "validation"), "driver schema/phase required")
    dev = contract["phase"] == "development"
    require(contract["seed_family"] == (202610149201 if dev else 202610149301), "frozen phase seed family required")
    require(contract["teacher_count"] == (4 if dev else 24), "fixed phase teacher count required")
    require(contract["initial_coordinates"] == [[0., 0., 0.], [.2, -.15, .1]], "both original starts retained")
    model = contract["model"]
    require(model["M"] == 8 and model["tau"] == 3. and model["a"] == 1.4 and model["spectral_floor"] == .15, "fixed observation model/domain required")
    require(np.allclose(model["epsilon"], np.full(3, 1.4 / math.sqrt(8)), rtol=0, atol=1e-15), "epsilon model mismatch")
    require(model["generation_inner_order"] == 192 and contract["fit_gh"] == [20, 128], "frozen generation/fit rules required")
    require(contract["risk_orders"] == [[20, 128], [28, 192], [36, 256]], "bounded original risk gate required")
    require(set(contract["solver"]) == {"initial_step", "mapping_tolerance", "max_iterations", "max_evaluations",
            "max_backtracks", "armijo", "step_bounds"},
            "solver must declare only its seven options; floor/deadline are driver-owned before RNG")
    f = contract["finite"]
    require(f["control"] in ("raw", "uniform_log") and f["qmc_bits"] == 52, "explicit finite surrogate and Sobol bits required")
    for key in ("fit_emp", "fit_pop", "eval_emp", "eval_pop", "K64_direction", "inner_check"):
        require(f[key]["R"] >= 1 and f[key]["B"] >= 1 and f[key]["B"] & (f[key]["B"] - 1) == 0, "power-of-two full-bank QMC B required")
    require(f["eval_emp"]["R"] >= 2 and f["eval_pop"]["R"] >= 2, "independent-scramble error units required")
    require(set(contract["caps"]) == {"seconds", "gaussian_scalars", "qmc_points", "qmc_normal_scalars",
            "ml_objective_calls", "pmf_law_calls", "finite_calls", "particle_pattern_terms", "sgd_bankcalls"}, "all nine frozen cap fields required")
    for name, value in contract["caps"].items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0, "positive frozen caps required")
    hashes = contract.get("source_sha256")
    require(isinstance(hashes, dict) and set(REQUIRED_SOURCES).issubset(hashes), "complete source hashes required before RNG")
    for name, expected in hashes.items():
        require(isinstance(expected, str) and len(expected) == 64, "empty/invalid source hash; no RNG")
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "source names must be safe repository-relative paths")
        path = resolve(name)
        require(path.is_relative_to(REPO), "source path outside repository")
        if verify_sources: require(sha(path) == expected, "source hash mismatch: " + name)
    for name in ("package_plan", "execution_design", "review_identity"):
        require(resolve(contract["references"][name]).is_file(), "missing frozen reference " + name)
    if dev: require(resolve(contract["references"]["development_teacher_selection"]).is_file(), "development selection required")
    else:
        require(read(resolve(contract["references"]["development_summary"]))["status"] == "completed", "completed development required before validation")
        require(resolve(contract["references"]["protocol_selection"]).is_file(), "prevalidation numerical/cost selection record required")
    return {"main_outputs": contract["teacher_count"] * (2 * len(reference_schedule(contract["phase"])) + 2),
            "extra_sgd_final_diagnostics": 8 if dev else 0, "fit_reference_outputs_are_last_accepted_per_start": True}


class Budget:
    def __init__(self, contract):
        self.caps = contract["caps"]
        self.started = time.perf_counter()
        self.counts = {name: 0 for name in self.caps if name != "seconds"}
        self.counts.update({"uniform_scalars": 0, "binomial_scalars": 0, "index_scalars": 0,
                           "pmf_cache_hits": 0, "qmc_scrambles": 0, "optimizer_updates": 0})

    def check(self, name=None, amount=0):
        if time.perf_counter() - self.started >= self.caps["seconds"]:
            raise RuntimeError("frozen phase wall cap reached")
        if name in self.caps and self.counts.get(name, 0) + amount > self.caps[name]:
            raise RuntimeError("frozen phase cap reached: " + name)

    def add(self, name, amount=1):
        self.check(name, amount); self.counts[name] = self.counts.get(name, 0) + int(amount)

    def snapshot(self):
        return {"counts": self.counts.copy(), "caps": self.caps, "seconds": time.perf_counter() - self.started,
            "QMC_uncertainty_unit": "independent_scramble_not_internal_bank_or_particle",
            "finite_terms_are_work_not_samples": True, "actual_PMF_cache_misses_separate_from_objective_calls": True}


class Streams:
    def __init__(self, contract, budget): self.contract, self.budget, self.streams = contract, budget, {}

    def get(self, role, teacher_id, *indices):
        key = (role, teacher_id, *indices)
        if key not in self.streams:
            entropy = [self.contract["seed_family"], ROLES[role], int(teacher_id), *map(int, indices)]
            self.streams[key] = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
        return self.streams[key]

    def draw(self, role, teacher_id, kind, size):
        rng = self.get(role, teacher_id)
        name = "gaussian_scalars" if kind == "standard_normal" else "uniform_scalars"
        number = int(np.prod(size))
        self.budget.check(name, number)
        value = getattr(rng, kind)(size)
        self.budget.add(name, number)
        return value

    def states(self): return {str(key): rng.bit_generator.state for key, rng in self.streams.items()}


def qmc_transform(uniform, K):
    uniform = np.asarray(uniform)
    require(uniform.ndim == 2 and uniform.shape[1] == 3 * K, "one Sobol point must be a complete d=3*K bank")
    z = ndtri(uniform).reshape(len(uniform), K, 3)
    require(np.isfinite(z).all(), "nonfinite Sobol Gaussian transform; no clipping or seed retry")
    return z


def make_qmc(contract, streams, budget, teacher_id, kind, K, destination):
    settings = contract["finite"][kind]
    R, B = settings["R"], settings["B"]
    role = "fitQMC" if kind.startswith("fit_") else "evalQMC"
    purpose = 0 if kind.endswith("emp") else 1 if kind.endswith("pop") else 2
    budget.check("qmc_points", R * B); budget.check("qmc_normal_scalars", R * B * K * 3)
    banks = []
    for replicate in range(R):
        rng = streams.get(role, teacher_id, K, replicate, purpose)
        net = qmc.Sobol(d=3 * K, scramble=True, bits=contract["finite"]["qmc_bits"], rng=rng)
        budget.add("qmc_scrambles")
        u = net.random_base2(int(math.log2(B)))
        budget.add("qmc_points", B)
        budget.add("qmc_normal_scalars", B * K * 3)
        write_npz(destination / f"{kind}_K{K}_scramble{replicate}_UNIFORM.npz", {"uniform": u})
        banks.append(qmc_transform(u, K))
    z = np.stack(banks)
    write_npz(destination / f"{kind}_K{K}_BANKS.npz", {"z": z})
    write_json(destination / f"{kind}_K{K}_RECEIPT.json", {"R": R, "B": B, "K": K, "Sobol_dimension": 3 * K,
        "role": role, "purpose": purpose, "bits": contract["finite"]["qmc_bits"],
        "seed_entropy": [[contract["seed_family"], ROLES[role], teacher_id, K, r, purpose] for r in range(R)],
        "reuse": "same_K_fit_or_eval_banks_shared_across_starts_and_nested_N; fit_and_eval_roles_independent"})
    return z


def generate_teachers(contract, streams, budget, out):
    if contract["phase"] == "development":
        selected = read(resolve(contract["references"]["development_teacher_selection"]))["teachers"]
        require([row["original_teacher_id"] for row in selected] == [9, 2, 11, 4], "frozen strength-ranked teacher ids required")
        teachers = [{"id": row["original_teacher_id"], **row, "role": "old_development_only"} for row in selected]
    else:
        teachers = []
        for teacher_id in range(24):
            G = streams.draw("teacher", teacher_id, "standard_normal", (3, 4))
            h = .25 + .5 * float(streams.draw("teacher", teacher_id, "random", (1,))[0])
            write_npz(out / f"TEACHER_DRAW_{teacher_id:02d}.npz", {"raw_G": G, "mixing_h": np.asarray(h)})
            gram = G @ G.T; diagonal = np.sqrt(np.diag(gram))
            require(np.all(diagonal > 0), "zero Wishart row; preserve failure without redraw")
            Cstar = (1-h) * np.eye(3) + h * gram / (diagonal[:,None] * diagonal[None,:])
            teachers.append({"id": teacher_id, "Cstar": Cstar, "raw_G": G, "mixing_h": h, "role": "new_independent_validation_teacher"})
    for teacher in teachers: legacy._matrix(teacher["Cstar"])
    write_json(out / "TEACHERS.json", {"phase": contract["phase"], "teachers": teachers,
        "saved_before_any_teacher_data_or_fit": True, "no_risk_result_selection": True})
    return teachers


def generate_visible_data(contract, streams, budget, teacher, destination):
    cfg = contract["model"]; tid = teacher["id"]; partial = {}
    try:
        x = streams.draw("data", tid, "standard_normal", (2048, 3)) @ np.linalg.cholesky(teacher["Cstar"]).T
        partial["Xstar"] = x
        h = competition.h_probabilities(x * np.asarray(cfg["epsilon"]), tau=cfg["tau"], inner_order=cfg["generation_inner_order"])
        budget.check(); rng = streams.get("data", tid)
        counts = rng.binomial(cfg["M"], h["h"]); budget.add("binomial_scalars", counts.size)
        partial["counts"] = counts
        uniforms = streams.draw("data", tid, "random", (2048, 3))
        Y = np.where(uniforms < expit(x), 1, -1).astype(np.int8)
        write_npz(destination / "VISIBLE_DATA.npz", {"counts": counts, "observed_labels": Y})
        write_npz(destination / "GENERATION_PROVENANCE.npz", {"Xstar": x, "label_uniforms": uniforms,
            "count_success_probability": h["h"], "count_failure_probability": h["complement"]})
        write_json(destination / "DATA_RECEIPT.json", {"N": 2048, "nested_prefix_N": 256,
            "teacher_latents_visible_to_empirical_fit": False, "generation_inner_order": cfg["generation_inner_order"],
            "rng_state": streams.get("data", tid).bit_generator.state,
            "qualification": "original_observation_formula_with_ordinary_GH_count_probabilities"})
        return {"counts": counts, "observed_labels": Y}
    except Exception:
        write_npz(destination / "FAILED_DATA_PREFIX.npz", partial)
        raise


def engine(contract, orders):
    m = contract["model"]
    return integrated.ActiveReferenceLikelihood(M=m["M"], epsilon=m["epsilon"], tau=m["tau"],
        outer_order=orders[0], inner_order=orders[1])


def counted_law(model, C, budget):
    # Frozen backend explicitly exposes exact-C byte cache; check cap before a
    # miss, and record actual returned cache state rather than charging hits.
    miss = model._last_key != np.asarray(C, dtype=float).tobytes()
    budget.check("pmf_law_calls", int(miss)); law = model.law(C)
    # The call can itself reach the wall cap; record its actual completed work
    # before the subsequent wall check, rather than losing the final count.
    name = "pmf_cache_hits" if law["cache_hit"] else "pmf_law_calls"
    budget.counts[name] += 1
    budget.check()
    return law


def ml_value(model, C, patterns, teacher_pmf, budget):
    budget.add("ml_objective_calls")
    miss = model._last_key != np.asarray(C, dtype=float).tobytes()
    budget.check("pmf_law_calls", int(miss))
    if patterns is not None:
        result = model.empirical_objective(C, patterns=patterns); value = result["nll"]
    else:
        result = model.population_objective(C, teacher_pmf); value = result["crossentropy"]
    name = "pmf_cache_hits" if result["law_cache_hit"] else "pmf_law_calls"
    budget.counts[name] += 1
    budget.check()
    return {"value": value, "gradient": result["gradient"], "metadata": result}


def finite_value(contract, z, C, patterns, budget, *, inner_order=None):
    m, f = contract["model"], contract["finite"]
    R, B, K, _ = z.shape; P = len(patterns["counts"])
    budget.check("particle_pattern_terms", R * B * K * P)
    budget.add("finite_calls"); budget.add("particle_pattern_terms", R * B * K * P)
    result = finite.evaluate(z, C, patterns["counts"], patterns["labels"], patterns["weights"],
        M=m["M"], epsilon=m["epsilon"], tau=m["tau"],
        inner_order=f["inner_order"] if inner_order is None else inner_order,
        control=f["control"], marginal_order=f["marginal_order"], pattern_chunk=f["pattern_chunk"])
    budget.check()
    return result


def point_record(teacher_id, method, role, N, init, coordinates, *, diagnostic=False, fit=None):
    label = "population" if N is None else f"N{N}"
    point_id = f"{method}_{label}_init{init}" if not diagnostic else f"sgd_final_diagnostic_init{init}"
    return {"point_id": point_id, "teacher_id": teacher_id, "method": method, "data_role": role, "N": N,
        "initialization_index": init, "coordinates": np.asarray(coordinates),
        "C": legacy.correlation_matrix(coordinates), "is_diagnostic": diagnostic,
        "fit_status": None if fit is None else fit["status"],
        "fit_mapping_norm_inf": None if fit is None else fit["mapping_norm_inf"],
        "output_rule": "fixed_reference_state" if method in ("initial", "teacher") else
            "last32_post_update_mean" if method == "sgd" and not diagnostic else
            "final_state_diagnostic_not_selected_output" if diagnostic else "last_accepted_per_start_not_best_start"}


def fit_reference(contract, budget, teacher_id, method, role, N, patterns, teacher_pmf, fit_z, model, destination):
    destination.mkdir(exist_ok=False)
    outputs = []
    for init, initial in enumerate(contract["initial_coordinates"]):
        with (destination / f"init{init}_OBJECTIVE_TRACE.jsonl").open("x", encoding="utf-8") as log:
            def objective(c):
                C = legacy.correlation_matrix(c)
                result = ml_value(model, C, patterns, teacher_pmf, budget) if method == "integrated" else finite_value(contract, fit_z, C, patterns, budget)
                record = {"coordinates": c, "value": result["value"], "gradient": result["gradient"],
                    "raw_value": result.get("raw_value"), "replicate_values": result.get("replicate_values"),
                    "replicate_gradients": result.get("replicate_gradients"), "budget": budget.snapshot()}
                log.write(json.dumps(record, default=serial, allow_nan=False) + "\n"); log.flush()
                return {"value": result["value"], "gradient": result["gradient"]}
            fit = optimize.solve(objective, initial, floor=contract["model"]["spectral_floor"],
                deadline=budget.started + contract["caps"]["seconds"], **contract["solver"])
        write_json(destination / f"init{init}_FIT.json", fit)
        point = point_record(teacher_id, method, role, N, init, fit["coordinates"], fit=fit)
        point["selected_scalar"] = "integrated_joint_NLL_or_CE" if method == "integrated" else contract["finite"]["control"] + "_fixed_QMC_surrogate"
        point["teacher_information_used_for_empirical_fit"] = False
        outputs.append(point)
    return outputs


def fit_sgd(contract, streams, budget, teacher_id, visible, destination):
    destination.mkdir(exist_ok=False); m = contract["model"]
    cfg = {"M": m["M"], "epsilon": m["epsilon"], "tau": m["tau"], "inner_order": 128, "K": 8,
           "steps": 256, "banks": 32, "learning_rate0": 1., "decay_scale": 64.,
           "spectral_floor": m["spectral_floor"], "output_average_last": 32}
    budget.check()
    indices = streams.get("SGDindices", teacher_id).integers(256, size=(256, 32)); budget.add("index_scalars", indices.size)
    write_npz(destination / "INDICES_PREFIX.npz", {"indices": indices})
    z = streams.draw("SGDactive", teacher_id, "standard_normal", (256, 32, 8, 3))
    write_npz(destination / "REPLAY_BANKS.npz", {"indices": indices, "z": z})
    write_json(destination / "CONFIG.json", cfg)
    result = original_sgd.replay(visible["counts"][:256], visible["observed_labels"][:256],
        contract["initial_coordinates"], indices, z, config=cfg,
        before_score=lambda B: budget.add("sgd_bankcalls", B))
    write_npz(destination / "TRACE.npz", result["trace"])
    budget.add("optimizer_updates", result["optimizer_updates"])
    outputs = [point_record(teacher_id, "sgd", "empirical", 256, init, c) for init, c in enumerate(result["coordinates"])]
    diagnostics = [point_record(teacher_id, "sgd", "empirical", 256, init, c, diagnostic=True)
                   for init, c in enumerate(result["trace"]["coordinates"][-1])] if contract["phase"] == "development" else []
    write_json(destination / "OUTPUTS.json", {"outputs": outputs, "diagnostic_points": diagnostics,
        "new_RNG_inside_replay": 0, "private_T_generated": False, "both_starts_share_indices_and_prior": True})
    write_json(destination / "TRACE_LOSS_CONVENTION.json", {"trace_loss_convention": result.get("trace_loss_convention",
        "original_collapsed_training_bank_loss_omits_count_combination_constants;_gradients_unchanged"),
        "new_integrated_and_finite_evaluation_include_count_combinations": True,
        "do_not_subtract_old_trace_absolute_loss_from_new_objective_levels": True})
    return outputs, diagnostics


def save_law(path, law):
    write_npz(path.with_suffix(".npz"), {name: value for name, value in law.items() if isinstance(value, np.ndarray)})
    write_json(path.with_suffix(".json"), {name: value for name, value in law.items() if not isinstance(value, np.ndarray)})


def precision_pass(d, q):
    return d["KL_order_difference"] <= q["kl_tol"] and d["PMF_order_max_difference"] <= q["pmf_tol"]


def risk_evaluation(contract, budget, teacher, point, engines, teacher_laws, destination):
    records, laws = [], []
    for level, orders in enumerate(contract["risk_orders"]):
        if level not in teacher_laws:
            teacher_laws[level] = counted_law(engines[level], teacher["Cstar"], budget)
            save_law(destination / f"TEACHER_GH{orders[0]}_{orders[1]}", teacher_laws[level])
        law = counted_law(engines[level], point["C"], budget); laws.append(law)
        prefix = point["point_id"] + f"_GH{orders[0]}_{orders[1]}"
        save_law(destination / prefix, law)
        risk = competition.joint_kl(teacher_laws[level]["joint_pmf"], law["joint_pmf"], M=contract["model"]["M"])
        records.append({"orders": orders, **risk, "PMF_path": str((destination / (prefix + ".npz")).resolve())})
        if level:
            d = {"KL_order_difference": abs(records[-1]["joint_kl"] - records[-2]["joint_kl"]),
                "PMF_order_max_difference": float(max(np.max(np.abs(law["joint_pmf"] - laws[-2]["joint_pmf"])),
                    np.max(np.abs(teacher_laws[level]["joint_pmf"] - teacher_laws[level-1]["joint_pmf"]))))}
            passed = precision_pass(d, contract["qualification"])
            if passed or level == 2: break
    return {"joint_kl": records[-1]["joint_kl"], "precision_qualified": passed,
        "risk_levels": records, "precision_diagnostics": d,
        "qualification": "ordinary_GH_order_gate_not_integral_certificate_no_normalization_or_KL_clipping"}


def objective_evaluation(contract, budget, point, initial_C, teacher_pmf, visible, models, qmc_eval, destination, baseline_cache):
    items = []
    contexts = [("empirical", 256), ("empirical", 2048), ("population", None)]
    for role, N in contexts:
        patterns = integrated.prepare_observations(visible["counts"][:N], visible["observed_labels"][:N], M=8) if N is not None else None
        same_initial = np.array_equal(point["C"], initial_C)
        starts = []
        for level in (0, 1):
            cache_key = (point["initialization_index"], role, N, level)
            if cache_key not in baseline_cache["integrated"]:
                baseline_cache["integrated"][cache_key] = ml_value(models[level], initial_C, patterns, teacher_pmf, budget)
            starts.append(baseline_cache["integrated"][cache_key])
        initial, higher_initial = starts
        result = initial if same_initial else ml_value(models[0], point["C"], patterns, teacher_pmf, budget)
        comparison = higher_initial if same_initial else ml_value(models[1], point["C"], patterns, teacher_pmf, budget)
        order_value = abs(comparison["value"] - result["value"])
        order_gradient = float(np.max(np.abs(comparison["gradient"] - result["gradient"])))
        ml_item = {"target": "integrated", "data_role": role, "N": N, "value": comparison["value"],
            "output_minus_matching_initial": comparison["value"] - higher_initial["value"],
            "gradient": comparison["gradient"], "value_order_difference": order_value,
            "gradient_order_max_difference": order_gradient,
            "precision_qualified": order_value <= contract["qualification"]["nll_order_tol"] and order_gradient <= contract["qualification"]["gradient_order_tol"],
            "projected_mapping": optimize.gradient_mapping(point["coordinates"], comparison["gradient"], floor=.15)}
        items.append(ml_item)
        if point["is_diagnostic"] and N != 256: continue
        fpatterns = finite.empirical_patterns(visible["counts"][:N], visible["observed_labels"][:N], M=8) if N is not None else finite.population_patterns(teacher_pmf, M=8)
        z = qmc_eval["eval_emp" if N is not None else "eval_pop"]
        cache_key = (point["initialization_index"], role, N)
        if cache_key not in baseline_cache["finite"]:
            baseline_cache["finite"][cache_key] = finite_value(contract, z, initial_C, fpatterns, budget)
        start = baseline_cache["finite"][cache_key]
        evaluated = start if same_initial else finite_value(contract, z, point["C"], fpatterns, budget)
        label = "population" if N is None else f"N{N}"
        write_npz(destination / (point["point_id"] + "_finite_" + label + ".npz"),
            {name: value for name, value in evaluated.items() if isinstance(value, np.ndarray)})
        settings = contract["finite"]["inner_check"]
        small_z = z[:settings["R"], :settings["B"]]
        primary_small = finite_value(contract, small_z, point["C"], fpatterns, budget)
        higher = finite_value(contract, small_z, point["C"], fpatterns, budget, inner_order=settings["inner_order"])
        inner_value = abs(primary_small["value"] - higher["value"])
        inner_grad = float(np.max(np.abs(primary_small["gradient"] - higher["gradient"])))
        mapping = optimize.gradient_mapping(point["coordinates"], evaluated["gradient"], floor=.15)
        delta = evaluated["replicate_values"] - start["replicate_values"]
        raw_delta = evaluated["raw_replicate_values"] - start["raw_replicate_values"]
        gradient_se = evaluated["replicate_gradient_se"]
        items.append({"target": "finite8", "data_role": role, "N": N, "control": evaluated["control"],
            "value": evaluated["value"], "raw_value": evaluated["raw_value"],
            "output_minus_matching_initial": float(delta.mean()), "paired_scramble_SE": float(delta.std(ddof=1)/math.sqrt(len(delta))),
            "raw_output_minus_matching_initial": float(raw_delta.mean()), "raw_paired_scramble_SE": float(raw_delta.std(ddof=1)/math.sqrt(len(raw_delta))),
            "gradient": evaluated["gradient"], "gradient_scramble_SE": gradient_se,
            "projected_mapping": mapping, "inner_value_order_difference": inner_value, "inner_gradient_order_difference": inner_grad,
            "inner_precision_qualified": inner_value <= contract["qualification"]["finite_inner_value_tol"] and inner_grad <= contract["qualification"]["finite_inner_gradient_tol"],
            "independent_gradient_qualified": bool(np.max(gradient_se) <= contract["qualification"]["finite_gradient_SE_tol"]),
            "independent_mapping_qualified": mapping["norm_inf"] <= contract["qualification"]["mapping_tol"],
            "scrambles": len(delta), "banks_per_scramble": z.shape[1],
            "qualification": "independent_eval_scrambles;_chosen_surrogate_exact_RP;_not_exact_true_FK_stationarity_or_global_optimality"})
    return items


def freeze(contract_path, contract, out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(contract_path, out / "CONTRACT.json")
    records = []
    for name, expected in contract["source_sha256"].items():
        target = out / "SEALED_SOURCE" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(name), target)
        require(sha(target) == expected, "source changed during freeze; no RNG")
        records.append({"path": target.relative_to(out).as_posix(), "sha256": expected})
    references = []
    for name, path in contract["references"].items():
        actual = resolve(path)
        target = out / "INPUT_REFERENCE" / (name + actual.suffix)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(actual, target)
        references.append({"role": name, "original_path": str(actual), "sha256": sha(actual), "copied_sha256": sha(target)})
    write_json(out / "FREEZE.json", {"schema": SCHEMA, "contract_sha256": sha(out / "CONTRACT.json"),
        "source_files": records, "references": references, "before_first_scientific_RNG": True,
        "rng_roles": ROLES, "QMC_entropy": "seed_family,role,teacher_id,K,scramble,emp0_pop1_direction2",
        "teacher_latents_are_generation_provenance_only_for_empirical_fit": True})


def run(contract_path, out):
    contract_path, out = Path(contract_path), Path(out).resolve()
    contract = read(contract_path); expected = validate_contract(contract)
    freeze(contract_path, contract, out)
    budget = Budget(contract); streams = Streams(contract, budget)
    outputs, diagnostics, teacher_data, teacher_weights, qmc_banks = [], [], {}, {}, {}
    active = {"stage": "teacher_generation"}
    try:
        teachers = generate_teachers(contract, streams, budget, out)
        for teacher in teachers:
            tid = teacher["id"]; active = {"stage": "data_and_fit", "teacher_id": tid}
            folder = out / f"teacher_{tid:02d}"; folder.mkdir()
            visible = generate_visible_data(contract, streams, budget, teacher, folder)
            teacher_data[tid] = visible
            qfolder = folder / "QMC"; qfolder.mkdir()
            banks = {kind: make_qmc(contract, streams, budget, tid, kind, 8, qfolder)
                     for kind in ("fit_emp", "fit_pop") if kind != "fit_pop" or contract["phase"] == "development"}
            qmc_banks[tid] = banks
            weight_model = engine(contract, [28, 192]); teacher_law = counted_law(weight_model, teacher["Cstar"], budget)
            teacher_weights[tid] = teacher_law["joint_pmf"]
            save_law(folder / "POPULATION_WEIGHT_GH28_192", teacher_law)
            fit_model = engine(contract, contract["fit_gh"])
            for method, role, N in reference_schedule(contract["phase"]):
                active.update(method=method, data_role=role, N=N)
                pattern = (integrated.prepare_observations(visible["counts"][:N], visible["observed_labels"][:N], M=8)
                    if method == "integrated" and N is not None else finite.empirical_patterns(visible["counts"][:N], visible["observed_labels"][:N], M=8)
                    if method == "finite8" and N is not None else finite.population_patterns(teacher_weights[tid], M=8) if method == "finite8" else None)
                label = "population" if N is None else f"N{N}"
                fit_z = banks.get("fit_emp" if N is not None else "fit_pop")
                outputs.extend(fit_reference(contract, budget, tid, method, role, N, pattern,
                    teacher_weights[tid] if N is None else None, fit_z, fit_model, folder / ("FIT_" + method + "_" + label)))
            sgd_outputs, final_points = fit_sgd(contract, streams, budget, tid, visible, folder / "SGD_N256")
            outputs.extend(sgd_outputs); diagnostics.extend(final_points)
            write_json(folder / "FIT_COMPLETE.json", {"teacher_id": tid, "outputs": [p for p in outputs if p["teacher_id"] == tid],
                "budget": budget.snapshot()})
            print(json.dumps({"stage": "fit", "teacher_id": tid, "completed_main_outputs": len(outputs), "budget": budget.snapshot()}, default=serial), flush=True)
        require(len(outputs) == expected["main_outputs"] and len(diagnostics) == expected["extra_sgd_final_diagnostics"], "fixed output schedule mismatch")
        write_json(out / "FROZEN_OUTPUTS.json", {"outputs": outputs, "output_selection": "every_fixed_start_last_accepted_or_prespecified_SGD_tail; no_best_start_or_risk_selection",
            "before_independent_QMC_or_high_order_output_evaluation": True, "budget": budget.snapshot()})
        write_json(out / "FROZEN_DIAGNOSTIC_POINTS.json", {"points": diagnostics, "role": "development_only_sgd_final_not_primary_method"})
        evaluations, direction_rows = [], []
        for teacher in teachers:
            tid = teacher["id"]; active = {"stage": "evaluation", "teacher_id": tid}
            folder = out / f"teacher_{tid:02d}"; eval_folder = folder / "EVALUATION"; eval_folder.mkdir()
            qfolder = folder / "QMC"
            qb = {kind: make_qmc(contract, streams, budget, tid, kind, 8, qfolder) for kind in ("eval_emp", "eval_pop")}
            models = [engine(contract, orders) for orders in contract["risk_orders"]]
            teacher_laws = {}
            baseline_cache = {"integrated": {}, "finite": {}}
            current = [point for point in outputs + diagnostics if point["teacher_id"] == tid]
            for init, c in enumerate(contract["initial_coordinates"]):
                current.append(point_record(tid, "initial", "baseline", None, init, c))
            current.append(point_record(tid, "teacher", "oracle", None, 0, legacy.correlation_coordinates(teacher["Cstar"])))
            teacher_eval = []
            for point in current:
                active["point_id"] = point["point_id"]
                risk = risk_evaluation(contract, budget, teacher, point, models, teacher_laws, eval_folder)
                initial_C = legacy.correlation_matrix(contract["initial_coordinates"][point["initialization_index"]])
                objectives = objective_evaluation(contract, budget, point, initial_C, teacher_weights[tid], teacher_data[tid], models, qb, eval_folder, baseline_cache)
                row = {**point, **risk, "objectives": objectives}
                own = [item for item in objectives if item["target"] == point["method"]
                       and item["data_role"] == point["data_role"] and item["N"] == point["N"]]
                row["reference_qualified"] = None
                if own and point["method"] in ("integrated", "finite8") and not point["is_diagnostic"]:
                    item = own[0]
                    precision = item["precision_qualified"] if point["method"] == "integrated" else (
                        item["inner_precision_qualified"] and item["independent_gradient_qualified"] and item["independent_mapping_qualified"])
                    row["reference_qualified"] = bool(point["fit_status"] == "mapping_tolerance" and precision
                        and item["projected_mapping"]["norm_inf"] <= contract["qualification"]["mapping_tol"])
                write_json(eval_folder / (point["point_id"] + "_EVALUATION.json"), row)
                teacher_eval.append(row); evaluations.append(row)
            if contract["phase"] == "development":
                high = make_qmc(contract, streams, budget, tid, "K64_direction", 64, qfolder)
                anchors = [point for point in current if point["method"] in ("initial", "teacher") or
                    point["method"] == "finite8" and point["data_role"] == "population"]
                require(len(anchors) == 5, "exact fixed K64 direction menu required")
                patterns = finite.population_patterns(teacher_weights[tid], M=8)
                for point in anchors:
                    # The first eight particles are a valid lower-dimensional
                    # marginal of this same scrambled complete K64 bank.
                    for K, z in ((8, high[:, :, :8, :]), (64, high)):
                        result = finite_value(contract, z, point["C"], patterns, budget)
                        name = point["point_id"] + f"_K{K}_DIRECTION"
                        write_npz(eval_folder / (name + ".npz"), {key: value for key, value in result.items() if isinstance(value, np.ndarray)})
                        direction_rows.append({"teacher_id": tid, "point_id": point["point_id"], "K": K,
                            "value": result["value"], "raw_value": result["raw_value"], "gradient": result["gradient"],
                            "gradient_scramble_SE": result["replicate_gradient_se"],
                            "paired_scramble_ids": list(range(high.shape[0])),
                            "paired_bank_source": f"QMC/K64_direction_K64_BANKS.npz",
                            "K8_is_first_eight_particles_of_same_K64_bank": True,
                            "mapping": optimize.gradient_mapping(point["coordinates"], result["gradient"], floor=.15),
                            "role": "development_fixed_point_direction_not_K64_optimization_or_validation_method"})
            write_json(folder / "EVALUATION_COMPLETE.json", {"teacher_id": tid, "points": teacher_eval, "budget": budget.snapshot()})
            print(json.dumps({"stage": "evaluation", "teacher_id": tid, "budget": budget.snapshot()}, default=serial), flush=True)
        report = {"schema": SCHEMA, "status": "completed", "phase": contract["phase"], "teachers": teachers,
            "outputs": outputs, "diagnostic_points": diagnostics, "evaluations": evaluations,
            "K64_fixed_direction": direction_rows, "budget": budget.snapshot(), "expected_schedule": expected,
            "uncertainty_unit": "independent_scramble_for_QMC;teacher_for_external_generalization",
            "teacher_risk_used_for_output_selection": False, "empirical_trainers_accept_only_visible_counts_Y": True,
            "qualification": "local_bounded_numerical_references;_all_starts_and_numerical_failures_retained;_not_global_optima"}
        write_json(out / "SUMMARY.json", report)
        write_json(out / "FINAL_RNG_STATES.json", streams.states())
        write_json(out / "MANIFEST.json", {"schema": SCHEMA, "files": [{"path": p.relative_to(out).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file()], "manifest_self_hash_excluded": True})
        return report
    except Exception:
        write_json(out / "FAILED.json", {"schema": SCHEMA, "status": "failed_preserved", "active": active,
            "traceback": traceback.format_exc(), "budget": budget.snapshot(), "completed_outputs": outputs,
            "rng_states": streams.states(), "automatic_retry": False})
        raise


def selftest():
    require(4 * (2 * len(reference_schedule("development")) + 2) == 56, "development schedule")
    require(24 * (2 * len(reference_schedule("validation")) + 2) == 192, "validation schedule")
    uniform = np.linspace(.05, .95, 4 * 24).reshape(4, 24)
    require(qmc_transform(uniform, 8).shape == (4, 8, 3), "complete-bank QMC shape")
    try: qmc_transform(uniform[:, :3], 8)
    except ValueError: pass
    else: raise AssertionError("dimension3 points cannot be grouped as K8 banks")
    require(precision_pass({"KL_order_difference": 0., "PMF_order_max_difference": 0.}, {"kl_tol": 1e-6, "pmf_tol": 1e-7}), "risk gate")
    return {"manufactured_only": True, "passed": True, "scientific_RNG": 0, "scientific_fit": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true"); modes.add_argument("--selftest", action="store_true")
    parser.add_argument("--contract", type=Path); parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.selftest: result = selftest()
    else:
        require(args.contract is not None, "contract required")
        if args.run:
            require(args.out is not None, "exclusive output required")
            result = run(args.contract, args.out)
            result = {"status": result["status"], "budget": result["budget"]}
        else: result = {"valid": True, "sampling": False, "writing": False, **validate_contract(read(args.contract))}
    print(json.dumps(result, indent=2, default=serial))


if __name__ == "__main__": main()
