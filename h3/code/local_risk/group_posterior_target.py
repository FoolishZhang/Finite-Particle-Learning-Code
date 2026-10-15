"""Frozen nine-teacher offline population target panel; no automatic execution.

Caller-owned common helpers freeze sources and generate bounded rejection banks.
All 36 last-accepted fits are saved before any independent evaluation bank.
The panel is fixed, not a random teacher sample; no risk-selected output rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import traceback
import numpy as np

import group_posterior_common as common
import group_posterior_objective as group

SCHEMA = "group_posterior_target_v1"
old = common.old


def validate(contract):
    old.require(contract["phase"] == "target", "target phase only")
    old.require(not contract.get("draft_not_authorized_to_run", False), "draft cannot execute")
    old.require(bool(contract.get("source_sha256")) and all(contract["source_sha256"].values()), "nonempty frozen source hashes required")
    teachers = contract["teachers"]
    old.require(len(teachers) == 9 and len({t["id"] for t in teachers}) == 9, "all nine fixed teachers required")
    old.require(contract["initial_coordinates"] == [[0., 0., 0.], [.2, -.15, .1]], "retain both original starts")
    old.require("floor" not in contract["solver"], "solver floor is supplied once from model")
    old.require(contract["workers"] == 4, "four fixed worker threads required")
    old.require(contract["group_control"] == "uniform_logratio", "frozen CV scalar required")
    for key in ("moment_tol", "coefficient_tol", "k1_diag_tol"):
        old.require(contract["qualification"][key] > 0, "explicit positive " + key + " required")
    for role in ("fit", "eval"):
        settings = contract["group"][role]
        old.require(settings["R"] >= 2 and settings["B"] > 0 and settings["B"] & (settings["B"]-1) == 0, "complete replicate banks required")
    for teacher in teachers:
        C = np.asarray(teacher["Cstar"], float)
        old.legacy._matrix(C)
        old.require(C.shape == (3, 3) and np.linalg.eigvalsh(C)[0] >= contract["model"]["spectral_floor"]-1e-12,
                    "fixed teacher must satisfy common spectral floor")
        for key in ("local_leading_coordinates", "quadratic_coordinates"):
            if teacher.get(key) is not None:
                c = np.asarray(teacher[key], float)
                old.require(c.shape == (3,) and np.isfinite(c).all(), "fixed prediction coordinates required")
                old.require(np.linalg.eigvalsh(old.legacy.correlation_matrix(c))[0] >= contract["model"]["spectral_floor"]-1e-12,
                            "prediction outside frozen spectral domain")
        for point in teacher.get("prior_frozen_points", []):
            source = Path(point["source_file"])
            if not source.is_absolute(): source = common.REPO/source
            old.require(common.sha(source) == point["source_sha256"], "old prior source hash differs")
            old.require(np.isfinite(point["joint_kl"]) and np.asarray(point["coordinates"]).shape == (3,), "frozen prior identity required")
    return contract


def qualify_marginal(contract, moments):
    """Ordinary double-order numerical gates, not continuous certificates."""
    maxima = {}
    for role in ("outer_absolute_differences", "private_absolute_differences"):
        maxima[role] = {key: float(np.max(np.abs(value))) for key, value in moments[role].items()}
    tol = contract["qualification"]["moment_tol"]
    old.require(all(value <= tol for fields in maxima.values() for value in fields.values()), "marginal normalizer/moment qualification failed")
    p, mu = np.asarray(moments["normalizer"]), np.asarray(moments["mean"])
    mass_error = float(np.max(np.abs(p.sum(axis=1)-1)))
    old.require(mass_error <= tol, "group observation mass qualification failed")
    v = np.sum(p*mu*mu, axis=1)
    coefficients = contract["theory_coefficients"]
    ct = contract["qualification"]["coefficient_tol"]
    old.require(np.max(np.abs(v-float(coefficients["v"]))) <= ct, "calibrated v differs from sampled target moments")
    J = float(coefficients["v"])**2; K = contract["model"]["K"]
    r = K*J/(1+(K-1)*J); lam = (1-J)/K
    old.require(abs(J-coefficients["J"]) <= ct and abs(r-coefficients["r_K8"]) <= ct,
                "calibrated local coefficients inconsistent")
    old.require(abs(J-contract["ridge"]["J"]) <= ct and abs(lam-contract["ridge"]["lambda"]) <= ct,
                "ridge is fixed by local Hessian, not risk selection")
    return {"qualified": True, "maxima": maxima, "observation_mass_error": mass_error,
            "v_by_group": v, "J": J, "r_K8": r, "ridge_lambda": lam,
            "qualification": "ordinary_GH_order_and_coefficient_agreement_not_interval_certificate"}


def worker_caps(contract):
    caps = {key: (value if key == "seconds" else 0 if key in common.SAMPLE_COUNTERS else int(value)//9)
            for key, value in contract["caps"].items()}
    for key in ("ml_objective_calls", "pmf_law_calls", "group_objective_calls", "particle_pattern_terms"):
        old.require(caps[key] > 0, "worker scoring cap must be positive: " + key)
    return caps


def coordinates(C):
    C = np.asarray(C)
    return C[[0, 0, 1], [1, 2, 2]]


def point(tid, point_id, c, method, *, initialization_index=None, fit=None, **extra):
    c = np.asarray(c, float)
    return {"point_id": point_id, "teacher_id": tid, "coordinates": c,
            "C": old.legacy.correlation_matrix(c), "method": method,
            "data_role": "population_offline", "N": None, "initialization_index": initialization_index,
            "is_fit_output": fit is not None, "fit_status": None if fit is None else fit["status"],
            "fit_mapping_norm_inf": None if fit is None else fit["mapping_norm_inf"],
            "output_rule": "fixed_reference_state" if fit is None else "last_accepted_per_original_start_no_best_start",
            **extra}


def fit_teacher(state, fit_samples, contract):
    teacher, budget, folder = state["teacher"], state["budget"], state["folder"]
    tid = teacher["id"]
    weight_orders = contract.get("population_weight_orders", [28, 192])
    weight_engine = old.engine(contract, weight_orders)
    law = old.counted_law(weight_engine, teacher["Cstar"], budget)
    old.save_law(folder/("POPULATION_WEIGHT_GH"+"_".join(map(str, weight_orders))), law)
    state["weight_law"], state["teacher_pmf"] = law, law["joint_pmf"]
    state["patterns"] = group.population_patterns(law["joint_pmf"], M=contract["model"]["M"])
    prepared = common.prepare(fit_samples, state["patterns"])
    model = old.engine(contract, contract["fit_gh"])
    outputs = []
    for method in ("group8", "ridge"):
        destination = folder/('FIT_'+method); destination.mkdir(exist_ok=False)
        for init, initial in enumerate(contract["initial_coordinates"]):
            with (destination/f"init{init}_OBJECTIVE_TRACE.jsonl").open("x", encoding="utf-8") as log:
                def objective(c):
                    if method == "group8":
                        result = common.group_value(prepared, c, budget, contract)
                    else:
                        base = old.ml_value(model, old.legacy.correlation_matrix(c), None, state["teacher_pmf"], budget)
                        lam = contract["ridge"]["lambda"]
                        result = {"value": base["value"]+.5*lam*np.dot(c, c), "gradient": base["gradient"]+lam*c}
                    log.write(json.dumps({"coordinates": c, "value": result["value"], "gradient": result["gradient"],
                        "raw_value": result.get("raw_value"), "replicate_values": result.get("replicate_values"),
                        "replicate_gradients": result.get("replicate_gradients"), "budget": budget.snapshot()}, default=common.serial, allow_nan=False)+"\n")
                    log.flush()
                    return {"value": result["value"], "gradient": result["gradient"]}
                fit = old.optimize.solve(objective, initial, floor=contract["model"]["spectral_floor"],
                    deadline=budget.started+contract["caps"]["seconds"], **contract["solver"])
            common.write_json(destination/f"init{init}_FIT.json", fit)
            outputs.append(point(tid, f"{method}_population_init{init}", fit["coordinates"], method,
                initialization_index=init, fit=fit,
                selected_scalar="uniform_logratio_fixed_group_bank" if method == "group8" else "integrated_CE_plus_fixed_ridge"))
    state["outputs"] = outputs
    common.write_json(folder/"FIT_COMPLETE.json", {"outputs": outputs, "budget": budget.snapshot()})
    return state


def analytic_K1(patterns, samples, contract):
    states, w = patterns["state_indices"], patterns["weights"]
    mu = np.stack([samples["proposal_mean"][g, states[:, g]] for g in range(3)], axis=-1)
    Q = np.einsum("p,pi,pj->ij", w, mu, mu)
    for g in range(3): Q[g, g] = w@samples["proposal_second_moment"][g, states[:, g]]
    error = float(np.max(np.abs(np.diag(Q)-1)))
    old.require(error <= contract["qualification"]["k1_diag_tol"], "population K1 diagonal identity failed; no empirical overwrite")
    C = Q.copy(); np.fill_diagonal(C, 1.)
    eigenvalues = np.linalg.eigvalsh(C)
    old.require(eigenvalues[0] >= contract["model"]["spectral_floor"]-1e-12,
                "analytic K1 unconstrained optimum outside common domain; no silent projection")
    return {"raw_Q": Q, "C": C, "coordinates": coordinates(C), "raw_diagonal_error": error,
            "eigenvalues": eigenvalues, "population_only_not_empirical_QN": True,
            "output_rule": "analytic_population_K1_Q_with_exact_unit_diagonal_identity",
            "no_K1_samples_or_optimizer": True}


def evaluation_points(state, samples, contract):
    teacher = state["teacher"]; tid = teacher["id"]
    k1 = analytic_K1(state["patterns"], samples, contract)
    common.write_json(state["folder"]/"ANALYTIC_K1.json", k1)
    points = [point(tid, "teacher_Cstar", coordinates(teacher["Cstar"]), "teacher"),
              point(tid, "initial_I", np.zeros(3), "initial"), *state["outputs"],
              point(tid, "analytic_K1", k1["coordinates"], "analytic_K1")]
    for key, label in (("local_leading_coordinates", "local_leading_prediction"), ("quadratic_coordinates", "quadratic_prediction")):
        if teacher.get(key) is not None:
            points.append(point(tid, label, teacher[key], "prediction", prediction_kind=key))
    for frozen in teacher.get("prior_frozen_points", []):
        points.append(point(tid, frozen["point_id"], frozen["coordinates"], "prior_frozen",
            prior_source_file=frozen["source_file"], prior_source_sha256=frozen["source_sha256"],
            prior_frozen_joint_kl=frozen["joint_kl"], prior_target_not_recomputed=True))
    old.require(len({p["point_id"] for p in points}) == len(points), "unique fixed point IDs required")
    return points


def evaluate_teacher(state, samples, contract):
    teacher, budget, folder = state["teacher"], state["budget"], state["folder"]
    destination = folder/"EVALUATION"; destination.mkdir(exist_ok=False)
    prepared = common.prepare(samples, state["patterns"])
    models = [old.engine(contract, orders) for orders in contract["risk_orders"]]
    teacher_laws = {}
    weight_orders = contract.get("population_weight_orders", [28, 192])
    if weight_orders in contract["risk_orders"]:
        teacher_laws[contract["risk_orders"].index(weight_orders)] = state["weight_law"]
    points = evaluation_points(state, samples, contract)
    common.write_json(destination/"FIXED_POINTS.json", {"points": points, "before_any_point_evaluation": True})
    group_cache, risk_cache, rows, results = {}, {}, [], {}
    for p in points:
        key = np.asarray(p["coordinates"], float).tobytes()
        if key not in group_cache:
            group_cache[key] = (p["point_id"], common.group_value(prepared, p["coordinates"], budget, contract))
        origin, result = group_cache[key]
        common.save_group_result(destination/(p["point_id"]+"_GROUP"), result)
        results[p["point_id"]] = result
        if key not in risk_cache:
            risk_cache[key] = (p["point_id"], old.risk_evaluation(contract, budget, teacher, p, models, teacher_laws, destination))
        risk_origin, risk = risk_cache[key]
        mapping = old.optimize.gradient_mapping(p["coordinates"], result["gradient"], floor=contract["model"]["spectral_floor"])
        se = result["replicate_gradient_se"]
        row = {**p, **risk, "group_result_reused_from": origin, "risk_reused_from": risk_origin,
               "group_value": result["value"], "group_raw_value": result["raw_value"],
               "group_gradient": result["gradient"], "group_gradient_scramble_SE": se,
               "group_raw_gradient": result["raw_gradient"], "group_raw_gradient_scramble_SE": result["raw_replicate_gradient_se"],
               "group_projected_mapping": mapping, "group_control": contract["group_control"],
               "scrambles": prepared.R, "banks_per_scramble": prepared.B,
               "reference_qualification_applies": p["method"] in ("group8", "ridge")}
        if p["method"] == "group8":
            row["reference_qualified"] = bool(p["fit_status"] == "mapping_tolerance" and
                mapping["norm_inf"] <= contract["qualification"]["mapping_tol"] and
                np.max(se) <= contract["qualification"]["finite_gradient_SE_tol"])
        elif p["method"] == "ridge":
            integrated = []
            lam, c = contract["ridge"]["lambda"], p["coordinates"]
            for engine in models[:2]:
                base = old.ml_value(engine, p["C"], None, state["teacher_pmf"], budget)
                integrated.append({"value": base["value"]+.5*lam*np.dot(c,c), "gradient": base["gradient"]+lam*c})
            high = integrated[-1]
            order_value = abs(high["value"]-integrated[0]["value"])
            order_gradient = float(np.max(np.abs(high["gradient"]-integrated[0]["gradient"])))
            ridge_mapping = old.optimize.gradient_mapping(c, high["gradient"], floor=contract["model"]["spectral_floor"])
            row["ridge_own_context"] = {**high, "projected_mapping": ridge_mapping,
                "value_order_difference": order_value, "gradient_order_max_difference": order_gradient, "lambda": lam}
            row["reference_qualified"] = bool(p["fit_status"] == "mapping_tolerance" and
                ridge_mapping["norm_inf"] <= contract["qualification"]["mapping_tol"] and
                order_value <= contract["qualification"]["nll_order_tol"] and
                order_gradient <= contract["qualification"]["gradient_order_tol"])
        else: row["reference_qualified"] = None
        if p["method"] == "prior_frozen":
            row["new_risk_minus_saved_prior_risk"] = row["joint_kl"]-p["prior_frozen_joint_kl"]
            row["prior_target_not_recomputed"] = True
        rows.append(row)
    baseline = results["teacher_Cstar"]
    for row in rows:
        r = results[row["point_id"]]
        paired = group.paired_statistics(r, baseline)
        raw = group.paired_statistics({"replicate_values": r["raw_replicate_values"], "replicate_gradients": r["raw_replicate_gradients"]},
            {"replicate_values": baseline["raw_replicate_values"], "replicate_gradients": baseline["raw_replicate_gradients"]})
        common.save_group_result(destination/(row["point_id"]+"_MINUS_CSTAR"),
            {**paired, **{"raw_"+k:v for k,v in raw.items()},
             "controlled_bank_delta": r["bank_values"]-baseline["bank_values"],
             "raw_bank_delta": r["raw_bank_values"]-baseline["raw_bank_values"]})
        row["group_minus_teacher"] = {"controlled_mean": paired["value"], "controlled_paired_scramble_SE": paired["replicate_se"],
                                      "raw_mean": raw["value"], "raw_paired_scramble_SE": raw["replicate_se"]}
        common.write_json(destination/(row["point_id"]+"_EVALUATION.json"), row)
    common.write_json(folder/"EVALUATION_COMPLETE.json", {"evaluations": rows, "budget": budget.snapshot()})
    return {"teacher_id": teacher["id"], "evaluations": rows}


def parallel(states, function, stage, workers):
    results, failures = [], []
    def guarded(state):
        try: return function(state)
        except Exception:
            error = {"schema": SCHEMA, "stage": stage, "teacher_id": state["teacher"]["id"],
                     "traceback": traceback.format_exc(), "budget": state["budget"].snapshot(), "automatic_retry": False}
            common.write_json(state["folder"]/('FAILED_'+stage.upper()+'.json'), error)
            raise
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='group-target-'+stage) as pool:
        futures = {pool.submit(guarded, state): state for state in states}
        for future in as_completed(futures):
            tid = futures[future]["teacher"]["id"]
            try:
                results.append(future.result())
                print(json.dumps({"stage": stage, "teacher_id": tid, "status": "completed"}), flush=True)
            except Exception:
                failures.append(traceback.format_exc())
                print(json.dumps({"stage": stage, "teacher_id": tid, "status": "failed"}), flush=True)
    if failures: raise RuntimeError("target worker failure(s): "+json.dumps(failures))
    return results


def run(contract_path, out):
    contract_path, out = Path(contract_path).resolve(), Path(out).resolve()
    validate(common.read(contract_path))
    contract, root_budget = common.start(contract_path, out, phase="target")
    states = []; active_stage = "marginal"
    try:
        sampler, moments = common.make_marginal(contract, out/"MARGINAL")
        marginal_qualification = qualify_marginal(contract, moments)
        common.write_json(out/"MARGINAL_QUALIFICATION.json", marginal_qualification)
        active_stage = "fit_bank_generation"
        fit_samples = common.make_banks(contract, sampler, contract["group"]["fit"], "fit", 0, out/"FIT_BANKS", root_budget)
        caps = worker_caps(contract)
        for teacher in contract["teachers"]:
            folder = out/f"teacher_{teacher['id']:02d}"; folder.mkdir(exist_ok=False)
            states.append({"teacher": teacher, "folder": folder, "budget": common.new_budget(contract, root_budget.started, caps)})
        active_stage = "fit"
        parallel(states, lambda state: fit_teacher(state, fit_samples, contract), "fit", contract["workers"])
        outputs = [point for state in states for point in state["outputs"]]
        old.require(len(outputs) == 36, "all 36 original-start fits required before independent bank generation")
        common.write_json(out/"FROZEN_OUTPUTS.json", {"schema": SCHEMA, "outputs": outputs,
            "before_independent_evaluation_bank_generation": True, "last_accepted_outputs_not_risk_selected": True})
        active_stage = "evaluation_bank_generation"
        eval_samples = common.make_banks(contract, sampler, contract["group"]["eval"], "evaluation", 0, out/"EVALUATION_BANKS", root_budget)
        active_stage = "evaluation"
        nested = parallel(states, lambda state: evaluate_teacher(state, eval_samples, contract), "eval", contract["workers"])
        order = {teacher["id"]:i for i,teacher in enumerate(contract["teachers"])}
        nested.sort(key=lambda item:order[item["teacher_id"]])
        evaluations = [p for item in nested for p in item["evaluations"]]
        budget = common.combine_budgets(contract, root_budget, [state["budget"] for state in states])
        report = {"schema": SCHEMA, "status": "completed", "phase": "target", "teacher_ids": list(order),
            "outputs": outputs, "evaluations": evaluations, "marginal_qualification": marginal_qualification,
            "budget": budget, "reference_qualified_count": sum(row["reference_qualified"] is True for row in evaluations),
            "risk_qualified_count": sum(row["precision_qualified"] for row in evaluations),
            "fit_outputs": 36, "teacher_sampling": 0, "visible_data_generation": 0,
            "K1_samples_and_fits": 0, "prior_target_recomputations": 0,
            "uncertainty_unit": "whole_independent_replicate_including_all_rejection_trial_nets",
            "qualification": "fixed_nine_teacher_population_panel_not_teacher_population_inference; local_numerical_references_not_global_certificates"}
        common.write_json(out/"SUMMARY.json", report); common.manifest(out)
        return report
    except Exception:
        common.write_json(out/"FAILED.json", {"schema": SCHEMA, "stage": active_stage,
            "traceback": traceback.format_exc(), "root_budget": root_budget.snapshot(),
            "worker_budgets": [{"teacher_id": state["teacher"]["id"], "budget": state["budget"].snapshot()} for state in states],
            "elapsed_seconds": time.perf_counter()-root_budget.started, "automatic_retry": False,
            "failed_attempt_preserved": True})
        common.manifest(out)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--contract', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = run(args.contract, args.out)
    print(json.dumps({"schema": SCHEMA, "status": result["status"], "outputs": len(result["outputs"])}, indent=2))


if __name__ == '__main__': main()
