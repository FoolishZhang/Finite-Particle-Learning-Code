"""Fresh-directory, contract-gated budget sensitivity on saved strong cases.

No execution happens on import. Phases never generate observations or modify
historical artifacts. A fit phase cannot create independent evaluation banks;
an evaluation phase requires a separately frozen list of all requested outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import scipy

import budget_sensitivity_core as core

REQUIRED_SOURCES = set(core.old.REQUIRED_SOURCES) | {
    "script/budget_sensitivity_core.py", "script/budget_sensitivity_run.py",
    "script/group_posterior_marginal.py", "script/group_posterior_objective.py",
    "script/group_posterior_quadratic.py"}
OPERATIONS = {"fixed_diagnostics": {"original_bank_stationarity", "moments", "common_risk"},
              "fit": {"fit"}, "evaluation": {"independent_stationarity", "common_risk", "moments"},
              "profile": {"profile"}}


def validate(cfg):
    core.require(cfg["schema"] == core.SCHEMA and cfg["phase"] in OPERATIONS, "explicit phase required")
    core.require(cfg.get("authorized_to_execute") is True and cfg.get("frozen") is True, "unfrozen/unauthorized contract")
    core.require(core.sha(core.REPO / "HANDOFF.md") == cfg["handoff_sha256"], "current feedback identity differs")
    original = core.read(core.resolve(cfg["original_contract"]))
    core.require(core.sha(core.resolve(cfg["original_contract"])) == cfg["original_contract_sha256"], "old contract differs")
    for key in ("model", "marginal", "sampling", "ridge", "initial_coordinates", "seed_family"):
        core.require(cfg[key] == original[key], "fixed original setting changed: " + key)
    core.require(cfg["seed_family"] == 202610159201 and cfg["evaluation_seed_family"] == 202610169201,
                 "original fit/new independent evaluation seed families required")
    core.require(cfg["case_ids"] and len(set(cfg["case_ids"])) == len(cfg["case_ids"]) and
                 all(i in range(2000, 2024) for i in cfg["case_ids"]), "unique original strong case IDs required")
    core.require(set(cfg["operations"]) <= OPERATIONS[cfg["phase"]] and cfg["operations"], "phase operations invalid")
    core.require(cfg["risk_orders"] == [[28, 192], [36, 256]], "common bounded risk rules required")
    core.require(cfg["moment_orders"] == [192, 256, 384], "fixed same-private-rule moment orders required")
    core.require(isinstance(cfg["pattern_chunk"], int) and 1 <= cfg["pattern_chunk"] <= 64, "bounded pattern chunk required")
    for key, cap in cfg["caps"].items():
        core.require(isinstance(cap, (int, float)) and not isinstance(cap, bool) and np.isfinite(cap) and cap > 0,
                     "positive finite cap required: " + key)
    required_caps = {"seconds", "group_objective_calls", "particle_pattern_terms", "control_construction_particle_pattern_terms",
        "ml_objective_calls", "pmf_law_calls", "marginal_private_node_terms", "optimizer_objective_attempts",
        "qmc_uniform_scalars", "qmc_trial_nets", "rejection_candidate_coordinates", "rejection_private_node_terms",
        "posterior_sample_coordinates"}
    core.require(required_caps <= set(cfg["caps"]), "complete work caps required")
    core.require(REQUIRED_SOURCES <= set(cfg["source_sha256"]), "numerical source closure missing")
    for field in ("source_sha256", "input_sha256"):
        for name, expected in cfg[field].items():
            path = Path(name)
            core.require(not path.is_absolute() and ".." not in path.parts, "repository-relative bound inputs required")
            core.require(core.sha(core.REPO / path) == expected, "bound file changed: " + name)
    root = core.resolve(cfg["old_run_root"])
    for case in cfg["case_ids"]:
        folder = root / f"case_{case:05d}"
        names = ["VISIBLE_DATA.npz", "FIT_COMPLETE.json", "FIT_BANKS/SAMPLES.npz", "FIT_BANKS/UNIFORMS.npz",
                 "EVALUATION_BANKS/SAMPLES.npz", "EVALUATION_BANKS/UNIFORMS.npz"]
        names += [f"FIT_{method}/init{i}_FIT.json" for method in core.METHODS for i in (0, 1)]
        for name in names:
            rel = (folder / name).relative_to(core.REPO).as_posix()
            core.require(rel in cfg["input_sha256"], "historical input not bound: " + rel)
    if cfg["phase"] == "fit":
        core.require(cfg["setting"] in ("O", "H") and cfg["operations"] == ["fit"], "single fit setting required")
        high = cfg["setting"] == "H"
        core.require(cfg["group"]["fit"] == ({"R": 4, "B": 2048} if high else {"R": 2, "B": 512}), "fit bank budget changed")
        core.require(cfg["fit_gh"] == ([28, 192] if high else [20, 128]), "fit GH setting changed")
        if high:
            core.require(cfg["solver_policy"] == "original" and cfg["solver"] == original["solver"],
                         "H requires the complete unchanged original solver policy")
        else:
            core.require(cfg["solver_policy"] == "tighter" and cfg["solver"] == {
                **original["solver"], "mapping_tolerance": 1e-9, "max_iterations": 300, "max_evaluations": 720},
                "O requires the explicitly tighter bounded solver policy")
    if cfg["phase"] == "profile":
        core.require(cfg["case_ids"] == [2000] and cfg["operations"] == ["profile"] and
                     cfg["group"]["fit"] == {"R": 4, "B": 2048}, "single fixed-case H-bank resource profile required")
    for case_id, name in cfg.get("fit_bank_reuse", {}).items():
        core.require(cfg["phase"] == "fit" and cfg["setting"] == "H" and int(case_id) in cfg["case_ids"] and int(case_id) == 2000,
                     "only the fixed case2000 profile bank may be reused by H")
        run = core.resolve(name)
        bindings = [run / "CONTRACT.json", run / "SUMMARY.json", run / "case_02000/FIT_BANKS/SAMPLES.npz",
                    run / "case_02000/FIT_BANKS/UNIFORMS.npz", run / "case_02000/FIT_BANKS/RNG.json",
                    run / "case_02000/FIT_BANKS/UNIFORM_PREFIX.json", run / "case_02000/FIT_BANKS/SAMPLE_PREFIX.json"]
        for path in bindings:
            core.require(path.relative_to(core.REPO).as_posix() in cfg["input_sha256"], "reused profile bank source unbound")
        profile = core.read(run / "CONTRACT.json")
        summary = core.read(run / "SUMMARY.json")
        core.require(profile["phase"] == "profile" and summary["status"] == "completed", "completed profile required for reuse")
        for key in ("seed_family", "model", "marginal", "sampling"):
            core.require(profile[key] == cfg[key], "reused profile sampling setting changed: " + key)
        for key in ("script/budget_sensitivity_core.py", "script/group_posterior_marginal.py"):
            core.require(profile["source_sha256"][key] == cfg["source_sha256"][key], "reused profile bank producer changed")
        core.require(core.read(run / "case_02000/FIT_BANKS/UNIFORM_PREFIX.json")["identical"] and
                     core.read(run / "case_02000/FIT_BANKS/SAMPLE_PREFIX.json")["identical"], "profile prefix was not verified")
    for check in cfg.get("moment_recompute_checks", []):
        core.require(cfg["phase"] == "fixed_diagnostics" and "moments" in cfg["operations"] and check ==
                     {"case_id": 2000, "point_id": "P0_group8_init0", "bank": "FIT_BANKS", "outer_order": 384},
                     "only frozen case2000/fit/init0/384 full moment recomputation permitted")
        core.require(check["case_id"] in cfg["case_ids"], "full-recompute check case not scheduled")
    core.require(len(cfg.get("moment_recompute_checks", [])) <= 1, "no duplicate full-recompute checks")
    if cfg["phase"] == "evaluation":
        core.require(cfg["group"]["eval"] == {"R": 8, "B": 2048} and cfg["prefix_B"] == 1024, "fixed independent reference budget required")
        core.require(cfg["frozen_output_files"], "output freeze required before independent bank generation")
        for name in cfg["frozen_output_files"]:
            core.require(name in cfg["input_sha256"], "unbound output freeze")
            frozen = core.read(core.resolve(name))
            core.require(frozen.get("outputs_frozen_before_independent_reference") is True, "not a fit-output freeze")
        core.require(cfg["expected_settings"] and len(set(cfg["expected_settings"])) == len(cfg["expected_settings"]), "expected output settings required")
        core.require("moments" not in cfg["operations"] or "independent_stationarity" in cfg["operations"],
                     "evaluation moment checks require the fixed reference bank")
    runtime = cfg["runtime"]
    core.require(runtime["python"] == sys.version.split()[0] and runtime["numpy"] == np.__version__ and
                 runtime["scipy"] == scipy.__version__, "runtime identity changed")
    return cfg


def freeze(path, out):
    cfg = validate(core.read(path))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, out / "CONTRACT.json")
    sources = {}
    for name, digest in cfg["source_sha256"].items():
        dest = out / "SEALED_SOURCE" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(core.REPO / name, dest)
        core.require(core.sha(dest) == digest, "source changed during freeze")
        sources[name] = digest
    core.write_json(out / "FREEZE.json", {"schema": core.SCHEMA, "phase": cfg["phase"],
        "contract_sha256": core.sha(path), "sources": sources, "input_sha256": cfg["input_sha256"],
        "before_scientific_integration_sampling_or_fitting": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    budget = core.Budget(cfg)
    budget.output_directory = out
    return cfg, budget


def manifest(out):
    core.write_json(out / "MANIFEST.json", {"schema": core.SCHEMA, "files": [
        {"path": p.relative_to(out).as_posix(), "sha256": core.sha(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*")) if p.is_file()], "manifest_self_hash_excluded": True})


def stationarity(point, result, cfg):
    mapping = core.gradient_mapping(point["coordinates"], result["gradient"], floor=cfg["model"]["spectral_floor"])
    se = float(np.max(result["replicate_gradient_se"]))
    fit_ok = point.get("fit_status") == "mapping_tolerance"
    mapping_ok = mapping["norm_inf"] <= cfg["qualification"]["mapping_tol"]
    se_ok = se <= cfg["qualification"]["finite_gradient_SE_tol"]
    return {"value": result["value"], "gradient": result["gradient"], "gradient_scramble_SE": result["replicate_gradient_se"],
            "projected_mapping": mapping, "maximum_gradient_SE": se, "fit_status_pass": fit_ok,
            "mapping_pass": bool(mapping_ok), "gradient_SE_pass": bool(se_ok),
            "composite_diagnostic_pass": bool(fit_ok and mapping_ok and se_ok),
            "diagnostic_never_inclusion_rule": True, "global_optimum_certified": False}


def level_from_samples(samples):
    return {"log_normalizer": samples["log_marginal_weight"], "mean": samples["proposal_mean"],
            "second_moment": samples["proposal_second_moment"]}


def evaluate_groups(cfg, samples, points, patterns, levels, budget, folder, *, prefix_B=None, bank_kind=None):
    bank = core.prepared(samples, patterns)
    old_level = level_from_samples(samples)
    for key in old_level:
        if 192 in levels:
            core.require(np.array_equal(old_level[key], levels[192][key]), "original and regenerated GH192 moments differ")
    rows = []
    for point in points:
        if point["method"] != "group8":
            continue
        if point.get("coordinates") is None:
            rows.append({"point": point, "evaluable": False})
            continue
        result = core.group_value(bank, point["coordinates"], budget, cfg)
        core.save_result(folder / (point["point_id"] + "_BASE"), result)
        row = {"point": point, "evaluable": True, "base": stationarity(point, result, cfg), "moment_levels": []}
        for order in cfg["moment_orders"] if "moments" in cfg["operations"] else []:
            changed = core.recenter_fixed_result(result, point["coordinates"], patterns, old_level, levels[order], cfg["model"]["K"])
            core.save_result(folder / (point["point_id"] + f"_MOMENTS{order}"), changed)
            row["moment_levels"].append({"outer_order": order, "same_inner_order": 256,
                **stationarity(point, changed, cfg), "value_shift": changed["value_shift"],
                "gradient_shift": changed["gradient_shift"], "evaluation_method": changed["evaluation_method"]})
            selected = {"case_id": point["case_id"], "point_id": point["point_id"],
                        "bank": bank_kind, "outer_order": order}
            if selected in cfg.get("moment_recompute_checks", []):
                # New prepared object, same sample arrays, explicit new moments;
                # never mutate the moment tables/caches of the production bank.
                full_bank = core.prepared(samples, patterns, moments=levels[order])
                full = core.group_value(full_bank, point["coordinates"], budget, cfg)
                core.save_result(folder / (point["point_id"] + f"_MOMENTS{order}_FULL_RECOMPUTE"), full)
                errors = {"value": abs(full["value"] - changed["value"]),
                    "gradient_max": float(np.max(np.abs(full["gradient"] - changed["gradient"]))),
                    "replicate_value_max": float(np.max(np.abs(full["replicate_values"] - changed["replicate_values"]))),
                    "replicate_gradient_max": float(np.max(np.abs(full["replicate_gradients"] - changed["replicate_gradients"]))),
                    "gradient_SE_max": float(np.max(np.abs(full["replicate_gradient_se"] - changed["replicate_gradient_se"])))}
                verification = {**selected, "absolute_errors": errors, "absolute_tolerance": 1e-11,
                    "passed": bool(max(errors.values()) <= 1e-11), "same_original_particles": True,
                    "full_recomputation_uses_separate_prepared_object": True,
                    "scope": "identity_of_two_finite_numerical_computations_not_true_moment_accuracy"}
                core.write_json(folder / "MOMENT_RECOMPUTE_VERIFICATION.json", verification)
                row["full_recompute_verification"] = verification
                core.require(verification["passed"], "analytic moment shift and full recomputation disagree")
        if prefix_B is not None:
            # Slice per-bank contributions from the exact same evaluation. The
            # controls have no B-dependent coefficients and use identical moments.
            low = core.statistics(result["bank_values"][:, :prefix_B].mean(axis=1),
                                  result["bank_gradients"][:, :prefix_B].mean(axis=1))
            delta = core.statistics(result["replicate_values"] - low["replicate_values"],
                                    result["replicate_gradients"] - low["replicate_gradients"])
            core.save_result(folder / (point["point_id"] + "_PREFIX"), low)
            core.save_result(folder / (point["point_id"] + "_FULL_MINUS_PREFIX"), delta)
            row["prefix"] = {"B": prefix_B, **stationarity(point, low, cfg)}
            row["full_minus_prefix"] = {k: delta[k] for k in ("value", "gradient", "replicate_se", "replicate_gradient_se")}
        rows.append(row)
    core.write_json(folder / "GROUP_DIAGNOSTICS.json", {"rows": rows})
    return rows


def fixed_case(cfg, case_id, budget, out, levels):
    case, historical, visible, points = core.historical_case(cfg, case_id)
    folder = out / f"case_{case_id:05d}"
    folder.mkdir()
    patterns = core.empirical_patterns(visible["counts"], visible["observed_labels"], M=cfg["model"]["M"])
    result = {"case_id": case_id, "pattern_count": len(patterns["weights"])}
    if "original_bank_stationarity" in cfg["operations"] or "moments" in cfg["operations"]:
        for kind in ("FIT_BANKS", "EVALUATION_BANKS"):
            target = folder / kind
            target.mkdir()
            samples = core.historical_bank(historical, kind)
            result[kind] = evaluate_groups(cfg, samples, points, patterns, levels, budget, target, bank_kind=kind)
    if "common_risk" in cfg["operations"]:
        result["risks"] = core.common_risk(cfg, case, points, budget, folder / "COMMON_RISK")
    core.write_json(folder / "COMPLETE.json", {**result, "budget": budget.snapshot()})
    return result


def fit_case(cfg, case_id, sampler, budget, out):
    case, historical, visible, _ = core.historical_case(cfg, case_id)
    folder = out / f"case_{case_id:05d}"
    folder.mkdir()
    patterns = core.empirical_patterns(visible["counts"], visible["observed_labels"], M=cfg["model"]["M"])
    if cfg["setting"] == "H":
        reuse = cfg.get("fit_bank_reuse", {}).get(str(case_id))
        if reuse is None:
            samples = core.make_banks(cfg, sampler, cfg["group"]["fit"], "fit", case_id, folder / "FIT_BANKS", budget,
                                      old_prefix=historical / "FIT_BANKS")
        else:
            bank_path = core.resolve(reuse) / f"case_{case_id:05d}/FIT_BANKS/SAMPLES.npz"
            samples = core.arrays(bank_path)
            core.require(samples["x_group"].shape == (4, 2048, 8, 3, 18), "reused H bank shape differs")
            core.write_json(folder / "FIT_BANK_REUSE.json", {"source_run": reuse, "path": str(bank_path),
                "sha256": core.sha(bank_path), "generated_in_this_phase": False,
                "source_contract_and_completed_prefix_receipts_bound": True})
    else:
        samples = core.historical_bank(historical, "FIT_BANKS")
    bank = core.prepared(samples, patterns)
    ml_patterns = core.old.integrated.prepare_observations(visible["counts"], visible["observed_labels"], M=cfg["model"]["M"])
    model = core.old.engine(cfg, cfg["fit_gh"])
    outputs = []
    for method in core.METHODS:
        for init, initial in enumerate(cfg["initial_coordinates"]):
            def objective(c):
                if method == "group8":
                    return core.group_value(bank, c, budget, cfg)
                result = core.old.ml_value(model, core.old.legacy.correlation_matrix(c), ml_patterns, None, budget)
                lam = cfg["ridge"]["lambda"] if method == "ridge" else 0.
                return {"value": result["value"] + .5 * lam * np.dot(c, c), "gradient": result["gradient"] + lam * c}
            fit = core.solve_recorded(objective, initial, folder / f"FIT_{method}_init{init}",
                floor=cfg["model"]["spectral_floor"], budget=budget, **cfg["solver"])
            p = {"case_id": case_id, "cell_id": case["cell_id"], "N": case["N"], "replicate": case["replicate"],
                "setting": cfg["setting"], "method": method, "initialization_index": init,
                "point_id": f"{cfg['setting']}_{method}_init{init}", "coordinates": fit["coordinates"],
                "fit_status": fit["status"], "fit_mapping_norm_inf": fit["mapping_norm_inf"],
                "has_valid_output": fit["has_valid_output"], "output_rule": fit["output_rule"],
                "teacher_information_used_for_fit": False}
            outputs.append(p)
            core.write_json(folder / (p["point_id"] + "_OUTPUT.json"), p)
            if fit["error"]:
                # Preserve last accepted output, then stop this frozen attempt.
                core.write_json(folder / "PARTIAL_OUTPUTS.json", {"outputs": outputs, "budget": budget.snapshot()})
                raise RuntimeError("hard fit failure retained; no retry or implicit budget increase")
    core.write_json(folder / "COMPLETE.json", {"outputs": outputs, "pattern_count": len(patterns["weights"]), "budget": budget.snapshot()})
    return outputs


def evaluation_points(cfg, case_id):
    case, folder, visible, p0 = core.historical_case(cfg, case_id)
    points = list(p0) if "P0" in cfg["expected_settings"] else []
    for name in cfg["frozen_output_files"]:
        points += [p for p in core.read(core.resolve(name))["outputs"] if p["case_id"] == case_id]
    expected = {(s, m, i) for s in cfg["expected_settings"] for m in core.METHODS for i in (0, 1)}
    core.require(len(points) == len(expected) and {(p["setting"], p["method"], p["initialization_index"]) for p in points} == expected,
                 "all specified frozen settings and original starts required; no filtering")
    return case, folder, visible, points


def evaluation_case(cfg, case_id, sampler, levels, budget, out):
    case, historical, visible, points = evaluation_points(cfg, case_id)
    folder = out / f"case_{case_id:05d}"
    folder.mkdir()
    result = {"case_id": case_id}
    if "independent_stationarity" in cfg["operations"]:
        samples = core.make_banks(cfg, sampler, cfg["group"]["eval"], "evaluation", case_id,
                                  folder / "REFERENCE_BANKS", budget)
        patterns = core.empirical_patterns(visible["counts"], visible["observed_labels"], M=cfg["model"]["M"])
        target = folder / "DIAGNOSTICS"
        target.mkdir()
        result["group_diagnostics"] = evaluate_groups(cfg, samples, points, patterns, levels, budget, target,
                                                       prefix_B=cfg["prefix_B"])
    if "common_risk" in cfg["operations"]:
        result["risks"] = core.common_risk(cfg, case, points, budget, folder / "COMMON_RISK")
    core.write_json(folder / "COMPLETE.json", {**result, "budget": budget.snapshot()})
    return result


def profile_case(cfg, sampler, budget, out):
    """Budget development only: one H bank and one pre-existing fixed point."""
    case, historical, visible, points = core.historical_case(cfg, 2000)
    folder = out / "case_02000"
    folder.mkdir()
    started = time.perf_counter()
    samples = core.make_banks(cfg, sampler, cfg["group"]["fit"], "fit", 2000,
                             folder / "FIT_BANKS", budget, old_prefix=historical / "FIT_BANKS")
    bank_seconds = time.perf_counter() - started
    patterns = core.empirical_patterns(visible["counts"], visible["observed_labels"], M=cfg["model"]["M"])
    point = next(p for p in points if p["point_id"] == "P0_group8_init0")
    bank = core.prepared(samples, patterns)
    started = time.perf_counter()
    result = core.group_value(bank, point["coordinates"], budget, cfg)
    objective_seconds = time.perf_counter() - started
    core.save_result(folder / "FIXED_POINT_GROUP", result)
    report = {"case_id": 2000, "point": point, "pattern_count": len(patterns["weights"]),
        "bank_generation_and_prefix_check_seconds": bank_seconds,
        "objective_and_first_control_construction_seconds": objective_seconds,
        "diagnostic": stationarity(point, result, cfg), "budget": budget.snapshot(),
        "scientific_role": "resource_development_not_confirmation_or_refit",
        "new_fits": 0, "new_risk_evaluations": 0, "new_visible_observations": 0,
        "same_fit_seed_family_as_frozen_future_H": cfg["seed_family"]}
    core.write_json(folder / "COMPLETE.json", report)
    return report


def run(path, out):
    cfg, budget = freeze(path, out)
    completed = []
    try:
        # All evaluation point identities are checked before any reference RNG.
        if cfg["phase"] == "evaluation":
            for case in cfg["case_ids"]:
                evaluation_points(cfg, case)
        sampler, levels = None, {}
        need_sampler = cfg["phase"] == "profile" or cfg["phase"] == "fit" and cfg["setting"] == "H" or "independent_stationarity" in cfg["operations"]
        if need_sampler or "moments" in cfg["operations"]:
            sampler, levels = core.make_marginal(cfg, budget, out / "MARGINAL", levels=cfg["moment_orders"] if "moments" in cfg["operations"] else [192])
        for case in cfg["case_ids"]:
            if cfg["phase"] == "fixed_diagnostics":
                item = fixed_case(cfg, case, budget, out, levels)
            elif cfg["phase"] == "fit":
                item = fit_case(cfg, case, sampler, budget, out)
            elif cfg["phase"] == "profile":
                item = profile_case(cfg, sampler, budget, out)
            else:
                item = evaluation_case(cfg, case, sampler, levels, budget, out)
            completed.append(item)
            print(json.dumps({"phase": cfg["phase"], "case_id": case, "status": "completed", "seconds": budget.snapshot()["seconds"]}), flush=True)
            budget.check()
        if cfg["phase"] == "fit":
            core.write_json(out / "FROZEN_OUTPUTS.json", {"outputs": [p for case in completed for p in case],
                "outputs_frozen_before_independent_reference": True, "all_original_starts_retained": True,
                "diagnostics_not_used_for_output_selection": True})
        summary = {"schema": core.SCHEMA, "phase": cfg["phase"], "status": "completed", "case_ids": cfg["case_ids"],
            "results": completed, "budget": budget.snapshot(), "new_observations": 0, "new_teacher_draws": 0,
            "new_scientific_claim": "retrospective_numerical_sensitivity_not_independent_confirmation"}
        core.write_json(out / "SUMMARY.json", summary)
        manifest(out)
        return summary
    except Exception:
        core.write_json(out / "FAILED.json", {"schema": core.SCHEMA, "phase": cfg["phase"], "completed_case_count": len(completed),
            "traceback": traceback.format_exc(), "budget": budget.snapshot(), "automatic_retry": False})
        manifest(out)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true", help="binding/shape/contract checks only; no scientific kernel calls")
    args = parser.parse_args()
    if args.check:
        validate(core.read(args.contract))
        print("contract and bindings valid; no integration/sampling/fitting")
    else:
        if args.out is None:
            parser.error("--out required for execution")
        run(args.contract.resolve(), args.out.resolve())
