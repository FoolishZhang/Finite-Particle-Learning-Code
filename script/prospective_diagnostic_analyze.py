"""Post-lock descriptive development analysis; no fitting or new observations.

The real CLI requires a verified global EVALUATION_LOCK before restoring any
study result. The 72-unit resampling is a fixed empirical-width sensitivity,
not a significance test, confidence guarantee, or launch decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import prospective_diagnostic_dev as dev
import numpy as np

ROOT, DOC, OUT = dev.ROOT, dev.DOC, dev.OUT
SUMMARY = OUT / "SUMMARY.json"
BOOTSTRAP_SEED = [202610071601, 0, 99, 4]
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_PER_CELL = 12
COST_KEYS = ("kernel_calls", "objective_calls", "successful_kernel_calls", "failed_kernel_calls",
             "full_hessian_calls", "loss_gradient_calls", "optimizer_updates", "processed_prior_vectors",
             "label_node_products", "generated_random_vectors")


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path): return json.loads(Path(path).read_text("utf-8"))
def native(x):
    if isinstance(x, np.ndarray): return native(x.tolist())
    if isinstance(x, np.generic): return native(x.item())
    if isinstance(x, dict): return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [native(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): raise ValueError("nonfinite public scalar; use explicit undefined status")
    return x
def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(native(value), f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
def text_save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f: f.write(value)


def required_number(mapping, key, *, nonnegative=True):
    if key not in mapping: raise KeyError("required measured field is absent: " + key)
    value = float(mapping[key])
    if not math.isfinite(value) or nonnegative and value < 0: raise ValueError("invalid measured field: " + key)
    return value


def expected_stems():
    paths = {"INPUTS_LOCK": [], "DIAGNOSIS_LOCK": [], "ENDPOINTS_LOCK": [], "EVALUATION_LOCK": []}
    for unit in dev.units():
        folder = OUT / "units" / unit["id"]
        paths["INPUTS_LOCK"].append(folder / "inputs")
        paths["DIAGNOSIS_LOCK"].extend(folder / name for name in ("anchor", "diagnosis", "start"))
        paths["ENDPOINTS_LOCK"].extend(folder / name for name in ("LOW", "HIGH", "REF"))
        paths["EVALUATION_LOCK"].append(folder / "risk")
    paths["EVALUATION_LOCK"].extend(OUT / f"teacher_D{i}" for i in (1, 2, 3))
    return paths


def locked_inputs():
    """Verify final lock first, then all previous locks and the fixed inventory."""
    final = OUT / "EVALUATION_LOCK.json"
    if not final.is_file(): raise RuntimeError("global EVALUATION_LOCK is required before analysis/audit")
    config = dev.verify()
    assert config["units"] == dev.units() and len(config["units"]) == 12
    hashes = {rel(OUT / "FREEZE.json"): sha(OUT / "FREEZE.json"), rel(OUT / "CONFIG.json"): sha(OUT / "CONFIG.json")}
    expected = expected_stems()
    for name in ("EVALUATION_LOCK", "INPUTS_LOCK", "DIAGNOSIS_LOCK", "ENDPOINTS_LOCK"):
        path = OUT / (name + ".json"); lock = read(path)
        assert lock["freeze_sha256"] == hashes[rel(OUT / "FREEZE.json")]
        inventory = {rel(stem.with_suffix(ext)) for stem in expected[name] for ext in (".json", ".npz")}
        assert set(lock["files"]) == inventory, (name, "unexpected or missing locked file")
        for key, row in lock["files"].items():
            assert row["path"] == key
            p = (ROOT / key).resolve(); assert p.is_relative_to(OUT.resolve())
            assert p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], key
            hashes[key] = row["sha256"]
        hashes[rel(path)] = sha(path)
    return config, hashes


def numeric_cost(cost):
    result = dict(wall_seconds=required_number(cost, "wall_seconds"))
    for key in COST_KEYS:
        value = required_number(cost, key)
        assert value.is_integer(), key
        result[key] = int(value)
    result["high_level_factor_totals"] = copy.deepcopy(cost["high_level_factor_totals"])
    return result


def merge_numeric_cost(left, right, wall_seconds):
    out = {key: left[key] + right[key] for key in COST_KEYS}
    out["wall_seconds"] = float(wall_seconds)
    names = set(left["high_level_factor_totals"]) | set(right["high_level_factor_totals"])
    # An absent operation category is a declared zero count in the union of
    # two complete primitive ledgers, not a missing measured-time fallback.
    out["high_level_factor_totals"] = {k: left["high_level_factor_totals"].get(k, 0) + right["high_level_factor_totals"].get(k, 0) for k in names}
    return out


def endpoint_geometry(fit):
    W, b = np.asarray(fit["W"]), np.asarray(fit["b"])
    singular = np.linalg.svd(W, compute_uv=False)
    return dict(W_Frobenius=float(np.linalg.norm(W)), bias_L2=float(np.linalg.norm(b)),
        W_singular_values=singular, s2_over_s1=float(singular[1] / singular[0]) if singular[0] > 0 else None,
        ratio_defined=bool(singular[0] > 0), algebra_of_saved_endpoint_only=True,
        used_for_selection_or_diagnosis=False, path_causal_attribution=False)


def policy_algebra(row):
    A, B = int(row["A"]), int(row["B"])
    if A > B: raise AssertionError("A must imply B")
    risk = row["risk_KL"]
    L, H, gain = float(risk["LOW"]), float(risk["HIGH"]), float(row["delta_R"])
    c = row["cost_seconds"]
    common, diag, refdiag, low, high = (c[key] for key in ("common_C0", "full_diagnostic", "reference_diagnostic", "LOW", "HIGH"))
    policies = {
        "always_LOW": dict(risk=L, wall_seconds=common + low),
        "always_HIGH": dict(risk=H, wall_seconds=common + high),
        "FULL": dict(risk=(1 - A) * L + A * H, wall_seconds=common + diag + (1 - A) * low + A * high),
        "reference_only": dict(risk=(1 - B) * L + B * H, wall_seconds=common + refdiag + (1 - B) * low + B * high),
        "REF_fit": dict(risk=float(risk["REF"]), wall_seconds=common + c["REF"]),
    }
    F, R = policies["FULL"], policies["reference_only"]
    residuals = dict(
        full_minus_low=(F["risk"] - L) - (-A * gain), full_minus_high=(F["risk"] - H) - ((1 - A) * gain),
        reference_minus_low=(R["risk"] - L) - (-B * gain), reference_minus_high=(R["risk"] - H) - ((1 - B) * gain),
        reference_minus_full=(R["risk"] - F["risk"]) - ((A - B) * gain),
        full_cost_minus_high=(F["wall_seconds"] - policies["always_HIGH"]["wall_seconds"]) - (diag - (1 - A) * (high - low)),
        reference_cost_minus_high=(R["wall_seconds"] - policies["always_HIGH"]["wall_seconds"]) - (refdiag - (1 - B) * (high - low)),
        full_cost_minus_reference=(F["wall_seconds"] - R["wall_seconds"]) - (diag - refdiag + (A - B) * (high - low)),
    )
    for key, value in residuals.items():
        tol = 2e-12 if not key.startswith(("full_cost", "reference_cost")) else 2e-10 * (1. + common + diag + refdiag + low + high)
        assert abs(value) <= tol, (key, value, tol)
    return dict(policies=policies, identity_residuals=residuals,
        FULL_terms=dict(A_delta=A * gain, stay_delta=(1 - A) * gain),
        reference_terms=dict(B_delta=B * gain, stay_delta=(1 - B) * gain),
        incremental_risk_gain=(A - B) * gain,
        FULL_20pct_diagnostic_ceiling=(1 - A) * (high - low) - .2 * (common + high),
        reference_20pct_diagnostic_ceiling=(1 - B) * (high - low) - .2 * (common + high))


def unit_row(unit, data, anchor, diagnosis, start, fits, risk):
    A, B = bool(diagnosis["full_raise"]), bool(diagnosis["ref_raise"])
    assert not A or B
    main = risk["main_index"]
    assert main == 1 and list(risk["rules"][main]) == [60, 0.]
    costs = {name: numeric_cost(fits[name]["cost"]) for name in ("LOW", "HIGH", "REF")}
    anchor_cost = numeric_cost(anchor["cost"])
    ref_cost = numeric_cost(diagnosis["cost"]["reference_only"])
    extra_cost = numeric_cost(diagnosis["cost"]["finite_extra"])
    full_seconds = required_number(diagnosis["cost"], "full_policy_diagnostic_seconds")
    ref_seconds = required_number(diagnosis["cost"], "reference_only_policy_diagnostic_seconds")
    assert math.isclose(full_seconds, ref_cost["wall_seconds"] + extra_cost["wall_seconds"], rel_tol=1e-12, abs_tol=1e-10)
    assert ref_seconds == ref_cost["wall_seconds"]
    kick = required_number(start["cost"], "wall_seconds")
    kick_generation = required_number(data, "kick_generation_seconds")
    c0 = anchor_cost["wall_seconds"] + kick + kick_generation
    times = dict(anchor=anchor_cost["wall_seconds"], kick_computation=kick, kick_generation=kick_generation,
        common_C0=c0, full_diagnostic=full_seconds, reference_diagnostic=ref_seconds,
        finite_extra_diagnostic=extra_cost["wall_seconds"],
        **{name: costs[name]["wall_seconds"] for name in costs},
        input_generation_total=required_number(data, "seconds"), data_generation=required_number(data, "data_generation_seconds"),
        shared_HIGH_prior_generation=required_number(data, "prior_generation_seconds"), risk_evaluation=required_number(risk, "seconds"))
    nums = diagnosis["numerics"]
    direction = nums.get("direction")
    alignment = None if direction is None else float(np.asarray(start["direction"]) @ np.asarray(direction))
    row = dict(unit=unit["id"], D="D" + str(unit["teacher"] + 1), teacher_index=unit["teacher"], N=unit["N"], rep=unit["rep"],
        A=A, B=B, full_level=diagnosis["full_level"], full_reasons=diagnosis["full_reasons"], ref_reasons=diagnosis["ref_reasons"],
        D0_reason_categories=diagnosis["D0_reason_categories"], thresholds=copy.deepcopy(diagnosis["thresholds"]),
        low_min=copy.deepcopy(nums["low_min"]), reference_min=copy.deepcopy(nums["reference_min"]),
        high_curvature=copy.deepcopy(nums["high_curvature"]), reference_gaps=copy.deepcopy(nums["reference_gaps"]),
        full_prerequisites=copy.deepcopy(diagnosis["full_prerequisites"]), reference_prerequisites=copy.deepcopy(diagnosis["reference_prerequisites"]),
        sign_resolved=bool(anchor["sign_resolved"]), anchor_status=anchor["status"],
        anchor_used_fallback=bool(anchor["used_initialization_fallback"]),
        endpoint_status={name: fits[name]["status"] for name in fits},
        endpoint_geometry={name: endpoint_geometry(fits[name]) for name in fits},
        kick_reference_direction_inner_product=alignment, alignment_defined=direction is not None,
        risk_KL={name: float(risk["KL"][name][main]) for name in ("LOW", "HIGH", "REF")},
        delta_R=float(risk["paired_gain"][main]), risk_qualified=bool(risk["LOW_HIGH_risk_qualified"]),
        risk_qualified_by_arm=copy.deepcopy(risk["qualified_by_arm"]), risk_errors=copy.deepcopy(risk["errors"]),
        paired_risk_rule_discrepancy=float(risk["paired_numerical_discrepancy"]), paired_risk_identity_error=float(risk["paired_identity_error"]),
        cost_seconds=times, costs=dict(anchor=anchor_cost, reference_diagnostic=ref_cost, finite_extra_diagnostic=extra_cost,
            full_diagnostic=merge_numeric_cost(ref_cost, extra_cost, full_seconds), **costs),
        generated_shared_prior_vectors=int(data["prior_vectors_generated"]), generated_kick_Gaussian_scalars=8,
        followup_order=["LOW", "HIGH", "REF"] if unit["rep"] == 0 else ["HIGH", "LOW", "REF"],
        risk_evaluation_order=60, descriptive_independent_development_unit=True)
    row["policy_algebra"] = policy_algebra(row)
    return native(row)


def group_descriptive(rows):
    if not rows: return dict(n=0, conditional_mean_defined=False, delta_mean=None, delta_min=None, delta_max=None, sample_sd=None)
    values = np.array([r["delta_R"] for r in rows])
    return dict(n=len(rows), conditional_mean_defined=True, delta_mean=float(values.mean()),
        delta_min=float(values.min()), delta_max=float(values.max()), sample_sd=float(values.std(ddof=1)) if len(rows) > 1 else None)


def metrics(rows):
    gain = np.array([r["delta_R"] for r in rows]); A = np.array([r["A"] for r in rows]); B = np.array([r["B"] for r in rows])
    assert np.all(A <= B)
    raise_rows, stay_rows = [r for r in rows if r["A"]], [r for r in rows if not r["A"]]
    G_raise = group_descriptive(raise_rows)["delta_mean"]; G_stay = group_descriptive(stay_rows)["delta_mean"]
    return dict(n=len(rows), A_count=int(A.sum()), B_count=int(B.sum()), D0_count=sum(r["full_level"] == "D0" for r in rows),
        E_A_delta=float(np.mean(A * gain)), E_stay_delta=float(np.mean((1 - A) * gain)),
        E_B_delta=float(np.mean(B * gain)), E_B_stay_delta=float(np.mean((1 - B) * gain)),
        I=float(np.mean((A.astype(int) - B.astype(int)) * gain)), G_raise=G_raise, G_stay=G_stay,
        G_raise_minus_G_stay=None if G_raise is None or G_stay is None else G_raise - G_stay,
        all_unit_gain=group_descriptive(rows), raise_group=group_descriptive(raise_rows), stay_group=group_descriptive(stay_rows))


def summarize(rows, teacher_seconds):
    assert len(rows) == 12 and len({r["unit"] for r in rows}) == 12
    descriptives = metrics(rows); evaluable = all(r["risk_qualified"] for r in rows)
    primary_names = ("E_A_delta", "E_stay_delta", "E_B_delta", "E_B_stay_delta", "I", "G_raise", "G_stay", "G_raise_minus_G_stay")
    primary = dict(evaluable=evaluable, failed_units=[r["unit"] for r in rows if not r["risk_qualified"]],
        values={key: descriptives[key] if evaluable else None for key in primary_names},
        reason=None if evaluable else "At least one LOW/HIGH numerical-risk qualification failed; primary risk totals are not evaluable.")
    cells = []
    for D in ("D1", "D2", "D3"):
        for N in (1024, 4096):
            group = [r for r in rows if r["D"] == D and r["N"] == N]
            assert len(group) == 2
            reason_counts = Counter(reason for r in group if r["full_level"] == "D0" for reason in r["full_reasons"])
            categories = {key: sum(r["full_level"] == "D0" and bool(r["D0_reason_categories"][key]) for r in group)
                          for key in ("computation_or_solver", "resolution")}
            cells.append(dict(D=D, N=N, **metrics(group), D0_reason_counts=dict(reason_counts), D0_category_counts=categories,
                all_risks_qualified=all(r["risk_qualified"] for r in group), all_units_retained=True))
    policies = []
    high_mean = float(np.mean([r["policy_algebra"]["policies"]["always_HIGH"]["wall_seconds"] for r in rows]))
    for name in ("always_LOW", "always_HIGH", "FULL", "reference_only", "REF_fit"):
        risk = [r["policy_algebra"]["policies"][name]["risk"] for r in rows]
        cost = [r["policy_algebra"]["policies"][name]["wall_seconds"] for r in rows]
        risk_ok = all(r["risk_qualified_by_arm"]["REF"] for r in rows) if name == "REF_fit" else evaluable
        policies.append(dict(policy=name, mean_risk_readout=float(np.mean(risk)), mean_wall_seconds=float(np.mean(cost)),
            total_wall_seconds=float(np.sum(cost)), risk_evaluable=risk_ok, risk_denominator=12,
            wall_saving_fraction_vs_always_HIGH=None if high_mean == 0 else 1. - float(np.mean(cost)) / high_mean,
            counterfactual_from_same_anchor_and_kick=True))
    ceilings = {}
    for name, dkey, idkey in (("FULL", "full_diagnostic", "FULL_20pct_diagnostic_ceiling"),
                             ("reference_only", "reference_diagnostic", "reference_20pct_diagnostic_ceiling")):
        diag = float(np.mean([r["cost_seconds"][dkey] for r in rows])); ceiling = float(np.mean([r["policy_algebra"][idkey] for r in rows]))
        ceilings[name] = dict(actual_mean_diagnostic_seconds=diag, necessary_20pct_mean_diagnostic_ceiling=ceiling,
            measured_prototype_satisfies_cost_inequality=diag <= ceiling,
            cost_inequality_only_not_a_risk_success_claim=True)
    always_low = next(p for p in policies if p["policy"] == "always_LOW")
    free_best = float(np.mean([r["cost_seconds"]["common_C0"] + min(r["cost_seconds"]["LOW"], r["cost_seconds"]["HIGH"]) for r in rows]))
    measured_research = sum(r["cost_seconds"]["input_generation_total"] + r["cost_seconds"]["anchor"] + r["cost_seconds"]["kick_computation"]
        + r["cost_seconds"]["full_diagnostic"] + sum(r["cost_seconds"][name] for name in ("LOW", "HIGH", "REF")) + r["cost_seconds"]["risk_evaluation"] for r in rows) + sum(teacher_seconds)
    return native(dict(unit_count=12, all_units_in_descriptive_denominator=True, primary_risk_totals=primary,
        descriptive_full_denominator=descriptives, condition_rows=cells,
        A_B_cells=[dict(A=A, B=B, **group_descriptive([r for r in rows if int(r["A"]) == A and int(r["B"]) == B])) for A in (0, 1) for B in (0, 1)],
        policy_risk_cost_table=policies, cost_ceilings=ceilings,
        free_diagnosis_all_LOW_saving_fraction=always_low["wall_saving_fraction_vs_always_HIGH"],
        free_diagnosis_best_per_unit_arm_saving_fraction=None if high_mean == 0 else 1. - free_best / high_mean,
        research_measured_component_wall_seconds=measured_research, teacher_PMF_seconds=teacher_seconds,
        cost_scope=dict(C0="anchor + neutral-kick computation + raw-xi generation", FULL="C0 + full diagnosis + selected LOW/HIGH",
            reference_only="C0 + reference-only diagnosis + B-selected LOW/HIGH", always="C0 + one arm; no diagnosis",
            research_total="input generation once + anchor + kick computation + full diagnosis once + all three fits + risk evaluation + teacher PMFs",
            shared_HIGH_pool_generation_charged_to_research_only=True, hypothetical_LOW_pool_generation_time_not_measured=True,
            imports_setup_serialization_and_analysis_not_in_function_wall=True, separately_timed_warmup=False,
            first_production_calls_included=True, fastest_repeat_selection=False,
            full_diagnostic_is_fixed_all_rule_prototype_not_optimal_short_circuit=True),
        no_development_significance_conclusion=True, new_formal_units=0))


def width_sensitivity(rows, *, draws=BOOTSTRAP_DRAWS):
    assert len(rows) == 12
    cells = [[i for i, r in enumerate(rows) if r["D"] == D and r["N"] == N] for D in ("D1", "D2", "D3") for N in (1024, 4096)]
    assert all(len(c) == 2 for c in cells)
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(BOOTSTRAP_SEED)))
    picks = np.asarray(cells)[np.arange(6)[None, :, None], rng.integers(0, 2, size=(draws, 6, BOOTSTRAP_PER_CELL))].reshape(draws, 72)
    A = np.array([r["A"] for r in rows], dtype=float)[picks]; B = np.array([r["B"] for r in rows], dtype=float)[picks]
    gain = np.array([r["delta_R"] for r in rows])[picks]
    ar, st = A.sum(axis=1), (1 - A).sum(axis=1)
    values = dict(E_A_delta=np.mean(A * gain, axis=1), E_stay_delta=np.mean((1 - A) * gain, axis=1),
        I=np.mean((A - B) * gain, axis=1), mean_delta=np.mean(gain, axis=1), A_fraction=A.mean(axis=1), B_fraction=B.mean(axis=1))
    g_raise = np.full(draws, np.nan); g_stay = np.full(draws, np.nan)
    np.divide((A * gain).sum(axis=1), ar, out=g_raise, where=ar > 0)
    np.divide(((1 - A) * gain).sum(axis=1), st, out=g_stay, where=st > 0)
    values.update(G_raise=g_raise, G_stay=g_stay, G_raise_minus_G_stay=g_raise - g_stay)
    for name in ("FULL", "reference_only", "always_LOW", "always_HIGH"):
        cost = np.array([r["policy_algebra"]["policies"][name]["wall_seconds"] for r in rows])[picks].mean(axis=1)
        values[name + "_mean_wall_seconds"] = cost
    denom = values["always_HIGH_mean_wall_seconds"]
    for name in ("FULL", "reference_only", "always_LOW"):
        values[name + "_saving_fraction"] = np.divide(denom - values[name + "_mean_wall_seconds"], denom,
            out=np.full(draws, np.nan), where=denom != 0)
    records = {}
    for name, value in values.items():
        finite = np.isfinite(value); quantiles = np.quantile(value[finite], [.025, .975], method="linear") if finite.any() else None
        records[name] = dict(defined_draws=int(finite.sum()), undefined_draws=int((~finite).sum()),
            percentile_2_5=None if quantiles is None else float(quantiles[0]),
            percentile_97_5=None if quantiles is None else float(quantiles[1]),
            width=None if quantiles is None else float(quantiles[1] - quantiles[0]),
            percentile_population="defined draws only; undefined counts retained, no replacements")
    return dict(seed=BOOTSTRAP_SEED, bit_generator="PCG64DXSM", draws=draws, cells=6, draws_per_cell=12,
        pseudo_units_per_draw=72, empirical_development_units_per_cell=2,
        joint_complete_unit_resampling=True, empty_raise_draws=int((ar == 0).sum()), empty_stay_draws=int((st == 0).sum()),
        undefined_draws_replaced=False, result=records, main_risk_evaluable=all(r["risk_qualified"] for r in rows),
        power_guarantee=False, confirmatory_interval=False, small_empirical_cells_can_understate_unseen_tails=True)


def collect():
    config, hashes = locked_inputs()
    rows = []
    for unit in dev.units():
        folder = OUT / "units" / unit["id"]
        data, anchor, diagnosis, start, risk = [dev.restore(folder / name) for name in ("inputs", "anchor", "diagnosis", "start", "risk")]
        fits = {name: dev.restore(folder / name) for name in ("LOW", "HIGH", "REF")}
        rows.append(unit_row(unit, data, anchor, diagnosis, start, fits, risk))
    teacher_seconds = [required_number(dev.restore(OUT / f"teacher_D{i}"), "seconds") for i in (1, 2, 3)]
    return config, hashes, rows, teacher_seconds


def stage_timing_summary():
    records, hashes = [], {}
    for path in sorted(OUT.glob("STAGE_TIMING_*.json")):
        record = read(path)
        if record["stage"] not in ("generate", "diagnose", "followup", "evaluate"): continue
        assert record["status"] in ("completed", "failed")
        assert record["includes_stage_import_verification_and_archival_IO"] is True
        assert record["excludes_initial_driver_process_import"] is True
        row = dict(path=rel(path), **record)
        row["wall_seconds"] = required_number(record, "wall_seconds")
        records.append(row); hashes[rel(path)] = sha(path)
    for stage in ("generate", "diagnose", "followup", "evaluate"):
        assert any(r["stage"] == stage and r["status"] == "completed" for r in records), "Missing completed stage timing: " + stage
    return dict(records=records, total_recorded_stage_wall_seconds=sum(r["wall_seconds"] for r in records),
        failures_included=True, includes_stage_import_verification_and_archival_IO=True,
        excludes_initial_driver_process_import=True, not_counterfactual_deployment_cost=True), hashes


def fmt(value): return "undefined" if value is None else f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value)


def readable(result):
    lines = ["# Independent development: all twelve fixed units", "",
        "These are descriptive development readouts, not confirmatory significance or power claims. Numerical failures remain in the denominator.", "",
        "|unit|A|B|level|LOW KL|HIGH KL|REF-fit KL|Delta R|LOW/HIGH risk qualified|",
        "|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in result["rows"]:
        lines.append("|" + "|".join(map(fmt, [r["unit"], int(r["A"]), int(r["B"]), r["full_level"],
            r["risk_KL"]["LOW"], r["risk_KL"]["HIGH"], r["risk_KL"]["REF"], r["delta_R"], r["risk_qualified"]])) + "|")
    lines += ["", "## Measured costs", "", "C0 includes anchor, neutral-kick computation and raw-xi generation. Full diagnosis is the fixed complete-rule prototype, not an optimized short-circuit implementation.", "",
        "|unit|anchor s|kick compute s|xi generate s|FULL diag s|REF-only diag s|LOW s|HIGH s|REF-fit s|LOW/HIGH processed priors|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["rows"]:
        c = r["cost_seconds"]
        lines.append("|" + "|".join(map(fmt, [r["unit"], c["anchor"], c["kick_computation"], c["kick_generation"],
            c["full_diagnostic"], c["reference_diagnostic"], c["LOW"], c["HIGH"], c["REF"],
            f"{r['costs']['LOW']['processed_prior_vectors']}/{r['costs']['HIGH']['processed_prior_vectors']}"])) + "|")
    lines += ["", "## Per-unit diagnostic and saved-endpoint details", ""]
    for r in result["rows"]:
        lines += [f"### {r['unit']}", "", f"Full reasons: {', '.join(r['full_reasons']) or 'none'}. Reference-only reasons: {', '.join(r['ref_reasons']) or 'none'}.",
            f"Low minima: {r['low_min']}; reference minima: {r['reference_min']}; high directional curvatures: {r['high_curvature']}; thresholds: {r['thresholds']}.",
            f"D0 categories: {r['D0_reason_categories']}. Kick/reference direction inner product: {fmt(r['kick_reference_direction_inner_product'])}; not used to select the kick.", "",
            "|arm|kernel calls|updates|W Frobenius|s2/s1|bias L2|endpoint status|", "|---|---:|---:|---:|---:|---:|---|"]
        for arm in ("LOW", "HIGH", "REF"):
            geo, c = r["endpoint_geometry"][arm], r["costs"][arm]
            lines.append("|" + "|".join(map(fmt, [arm, c["kernel_calls"], c["optimizer_updates"], geo["W_Frobenius"], geo["s2_over_s1"], geo["bias_L2"], r["endpoint_status"][arm]])) + "|")
        lines += ["", "|component|kernel calls|objective calls|label-node products|processed prior vectors|", "|---|---:|---:|---:|---:|"]
        for component in ("anchor", "full_diagnostic", "reference_diagnostic", "finite_extra_diagnostic"):
            c = r["costs"][component]
            lines.append("|" + "|".join(map(fmt, [component, c["kernel_calls"], c["objective_calls"], c["label_node_products"], c["processed_prior_vectors"]])) + "|")
    return "\n".join(lines) + "\n"


def result_markdown(result):
    s = result["summary"]; d = s["descriptive_full_denominator"]
    lines = ["# Independent development configuration 1: descriptive results", "",
        "This is an author execution/analysis draft for the fixed twelve development units. Scientific interpretation and any subsequent contract remain with the primary author. No formal sample was generated by this analysis.", "",
        f"All units: 12; FULL raises: {d['A_count']}; reference-only raises: {d['B_count']}; D0: {d['D0_count']}. A implies B on every unit.",
        f"Main LOW/HIGH risk totals evaluable: {s['primary_risk_totals']['evaluable']}. Failed units: {s['primary_risk_totals']['failed_units']}.",
        "Raw full-denominator descriptive values (not complete-case estimates): " + json.dumps({k:d[k] for k in ("E_A_delta","E_stay_delta","I","G_raise","G_stay","G_raise_minus_G_stay")}), "",
        "## Risk/cost table", "", "|policy|mean raw KL|risk evaluable|mean measured wall s|saving vs always HIGH|", "|---|---:|---|---:|---:|"]
    for row in s["policy_risk_cost_table"]:
        lines.append("|" + "|".join(map(fmt, [row["policy"], row["mean_risk_readout"], row["risk_evaluable"], row["mean_wall_seconds"], row["wall_saving_fraction_vs_always_HIGH"]])) + "|")
    lines += ["", "## Cost feasibility and uncertainty scope", "",
        "The twenty-percent cost condition is E[Cdiag] <= E[(1-A)(CH-CL)] - .2 E[C0+CH], with B substituted for reference-only.",
        "Measured ceilings: " + json.dumps(s["cost_ceilings"]),
        f"Free-diagnosis, all-LOW saving fraction: {fmt(s['free_diagnosis_all_LOW_saving_fraction'])}; free-diagnosis best observed arm per unit: {fmt(s['free_diagnosis_best_per_unit_arm_saving_fraction'])}.",
        f"Sum of measured research components: {fmt(s['research_measured_component_wall_seconds'])} seconds. This includes both arms, REF fit and risk evaluation, not a deployment policy cost.",
        f"Recorded generate/diagnose/followup/evaluate stage wall total: {fmt(result['research_stage_timing']['total_recorded_stage_wall_seconds'])} seconds, including stage import/verification/archival I/O and recorded failed calls, but excluding initial driver-process imports. This is distinct from the component sum and deployment costs.",
        "The fixed 10,000 resamples draw 12 complete units from each two-unit development cell. Percentiles describe only this empirical width sensitivity; empty conditional groups stay undefined and are never redrawn. Two observations per cell may miss important tail variation. This is not a power guarantee or significance conclusion.",
        "", "The complete unit table, diagnostic reasons, saved endpoint norms and all bootstrap widths are in the public SUMMARY.json and READABLE_UNIT_TABLE.md. The norms are algebraic descriptions of saved endpoints, not path-level causal attribution.", ""]
    return "\n".join(lines)


def make_figure(result, png, svg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["rows"]; x = np.arange(12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for name, marker in (("LOW", "o"), ("HIGH", "s"), ("REF", "^")):
        axes[0].plot(x, [r["risk_KL"][name] for r in rows], marker=marker, markersize=4, linewidth=.7, label=name)
    axes[0].set_xticks(x, [r["unit"] for r in rows], rotation=70, ha="right", fontsize=6)
    axes[0].set_ylabel("Observation KL (GH60 raw-weight readout)"); axes[0].set_title("All 12 development units"); axes[0].legend(fontsize=8)
    policies = result["summary"]["policy_risk_cost_table"]
    axes[1].bar(np.arange(len(policies)), [r["mean_wall_seconds"] for r in policies], color=["#577590", "#277da8", "#f8961e", "#90be6d", "#888888"])
    axes[1].set_xticks(np.arange(len(policies)), [r["policy"] for r in policies], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean measured function wall seconds"); axes[1].set_title("Costs include common anchor and neutral kick")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.suptitle("Descriptive independent development; numerical gates are not confidence bounds", fontsize=10)
    fig.savefig(png, dpi=170); fig.savefig(svg); plt.close(fig)


def compute_record():
    config, inputs, rows, teacher_seconds = collect()
    timings, timing_hashes = stage_timing_summary(); inputs.update(timing_hashes)
    return dict(schema="prospective_dev_analysis_v1", source_sha256=sha(__file__),
        driver_sha256=sha(dev.__file__), freeze_sha256=sha(OUT / "FREEZE.json"), evaluation_lock_sha256=sha(OUT / "EVALUATION_LOCK.json"),
        input_hashes=inputs, configuration_number=config["configuration_number"], stage="independent_development_not_formal",
        rows=rows, summary=summarize(rows, teacher_seconds), research_stage_timing=timings,
        bootstrap_width_sensitivity=width_sensitivity(rows),
        scientific_interpretation_owner="primary author", confirmatory_significance_claimed=False, formal_execution_decision=False)


def main(mode):
    if mode == "run" and SUMMARY.exists(): raise FileExistsError("SUMMARY exists; no overwrite or outcome-based revision")
    started = time.perf_counter(); record = compute_record()
    if mode == "check":
        saved = read(SUMMARY)
        assert {k:v for k,v in saved.items() if k not in ("created_utc", "analysis_seconds", "outputs")} == native(record)
        for name, digest in saved["outputs"].items(): assert sha(ROOT / name) == digest
        print(json.dumps(dict(passed=True, check=True, units=12, no_new_fit_or_observation=True)))
        return
    table, results_md = OUT / "READABLE_UNIT_TABLE.md", DOC / "RESULTS.md"
    png, svg = OUT / "development_risk_cost.png", OUT / "development_risk_cost.svg"
    for path in (table, results_md, png, svg):
        if path.exists(): raise FileExistsError(path)
    text_save(table, readable(record)); text_save(results_md, result_markdown(record)); make_figure(record, png, svg)
    _, after = locked_inputs(); _, timing_hashes = stage_timing_summary(); after.update(timing_hashes)
    assert after == record["input_hashes"], "locked inputs changed during analysis"
    record.update(created_utc=now(), analysis_seconds=time.perf_counter() - started, outputs={rel(p):sha(p) for p in (table, results_md, png, svg)})
    save(SUMMARY, record)
    print(json.dumps(dict(completed=True, units=12, primary_risk_evaluable=record["summary"]["primary_risk_totals"]["evaluable"], summary=rel(SUMMARY))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true"); mode.add_argument("--check", action="store_true")
    args = parser.parse_args(); main("run" if args.run else "check")
