"""Fixed finite-angle panel with shared baselines and direct paired effects.

No optimization, adaptive rule, new random draw or skipped negative side is
available. SOURCE/FREEZE are prepared by the coordinating author; --run executes
the frozen panel once. --self-test uses manufactured fixtures only.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
import noncollinear_probe_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/NoncollinearProbe_v1_20261007"
OUT = ROOT / "artifacts/noncollinear_probe_v1_20261007"
SOURCE, FREEZE = OUT / "SOURCE.json", OUT / "FREEZE.json"
POINTS = (("zero", 0.), ("m010", -.10), ("m005", -.05), ("m002", -.02),
          ("p002", .02), ("p005", .05), ("p010", .10))
RULES = {
    "A": dict(normal_specs=[dict(rule="hermite", order=96), dict(rule="hermite", order=24)], logu_min=-28., logu_max=56., logu_step=.25),
    "B": dict(normal_specs=[dict(rule="hermite", order=144), dict(rule="hermite", order=32)], logu_min=-32., logu_max=64., logu_step=.125),
    "C": dict(normal_specs=[dict(rule="legendre", order=384, limit=11.), dict(rule="hermite", order=40)], logu_min=-36., logu_max=72., logu_step=.125),
}
GATES = dict(mass=2e-12, row_norm=5e-14, marginal=2e-10,
    baseline_loss=1e-7, baseline_PMF=2e-10, absolute_loss=1e-7, raw_KL=1e-10, PMF=2e-10,
    interface=5e-12, F_BC_abs=1e-10, F_AC_abs=1e-9, R_BC_abs=1e-11, R_AC_abs=1e-10,
    BC_relative=.02, AC_relative=.10, relative_floor=1e-12,
    even_F=1e-11, even_R=1e-11, even_PMF=2e-12,
    sign_absolute_floor=1e-9, sign_difference_factor=20.)


def config_dump():
    return native(dict(schema="noncollinear_probe_config_v1", anchor_K=[11, 12],
        points=[dict(point_id=n, t=t) for n, t in POINTS], primary_amplitude=.02,
        sensitivity_amplitudes=[.05, .10], rules=RULES, gates=GATES,
        outer_chunk=32, label_chunk=8, anchor_workers=2,
        worker_order="independent anchors; fixed A/B/C and t order within each; SOURCE order at aggregation",
        baseline_preparations=6, nonzero_evaluations=36,
        main_F_effect="direct_paired_delta_for_own_finite_K", main_R_effect="direct_log_evidence_paired_risk_delta",
        Gaussian_or_target_weights_renormalized=False, optimizer_updates=0, new_random_draws=0))


def now(): return datetime.now(timezone.utc).isoformat()


def native(x):
    if isinstance(x, np.ndarray): return native(x.tolist())
    if isinstance(x, np.generic): return native(x.item())
    if isinstance(x, dict): return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [native(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): raise ValueError("nonfinite public JSON scalar")
    return x


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(native(value), f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def save_npz(path, values):
    with Path(path).open("xb") as f: np.savez_compressed(f, **values)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1048576), b""): h.update(data)
    return h.hexdigest()


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def artifact(path): return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def fingerprint(value):
    h = hashlib.sha256()
    def update(x):
        if isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
            h.update(memoryview(np.ascontiguousarray(x)).cast("B"))
        elif isinstance(x, dict):
            for key in sorted(x): h.update(str(key).encode()); update(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: update(item)
        else: h.update(json.dumps(native(x), sort_keys=True).encode())
    update(value)
    return h.hexdigest()


def all_finite(result):
    """All numeric outputs finite except valid logPsi=-inf (Psi exactly zero)."""
    for key, value in result.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if key == "logPsi":
                if np.isnan(value).any() or np.isposinf(value).any(): return False
            elif not np.isfinite(value).all(): return False
        if isinstance(value, (float, np.floating)) and not math.isfinite(value): return False
    return True


def arrays_for_save(result, anchor, direction, t):
    arrays = {k: v.copy() for k, v in result.items() if isinstance(v, np.ndarray)}
    # Kernel scalar quantities are retained as zero-dimensional arrays.
    for key in ("normal_quadrature_mass", "model_mass"):
        if key in result: arrays[key] = np.asarray(result[key]).copy()
    arrays.update(theta=np.asarray(anchor["theta"]).copy(), direction=np.asarray(direction).copy(), t=np.asarray(t),
        K_names=np.asarray([str(K) for K in result["K_values"]]),
        finite_K_values=np.asarray(result["finite_K_values"], dtype=int), target_names=np.asarray(result["target_names"]))
    return arrays


def geometry(anchor, direction, t, result_b, W):
    theta = np.asarray(anchor["theta"]); m = len(theta) // 2; a = theta[m:]
    return dict(bias_exactly_equal=bool(np.array_equal(result_b, theta[:m])),
        row_norm_max_error=float(np.max(abs(np.linalg.norm(W, axis=1) - a))),
        row_norms=np.linalg.norm(W, axis=1), singular_values=np.linalg.svd(W, compute_uv=False),
        path_W_max_error=float(np.max(abs(W - kernel.path_W(a, direction, t)))))


def baseline_summary(prepared, anchor, direction):
    return native(dict(t=0., reused_zero=True, K_values=prepared["K_values"],
        absolute_loss=prepared["absolute_loss"][0].copy(), raw_KL=float(prepared["raw_KL"][0]),
        pmf=prepared["pmf"].copy(), marginals=prepared["marginals"].copy(),
        model_mass=float(prepared["model_mass"]), normal_mass=float(prepared["normal_quadrature_mass"]),
        target_mass=float(prepared["q_mass"][0]), all_outputs_finite=all_finite(prepared),
        finite_gate_exception="logPsi may equal -infinity only to encode Psi=0",
        logPsi_zero_count=int(np.isneginf(prepared["logPsi"]).sum()),
        PMF_nonnegative=bool(np.all(prepared["pmf"] >= 0)),
        geometry=geometry(anchor, direction, 0., prepared["b"], prepared["W0"]),
        config=prepared["config"], work_counts=prepared["work_counts"],
        paired_F=0., paired_R=0., paired_infinite=0., zero_effect_is_definition=True,
        quadratic_F=None, quadratic_R=None, no_additional_zero_evaluation=True))


def point_summary(result, anchor, direction, t):
    ki = list(result["K_values"]).index(anchor["K"]); ii = list(result["K_values"]).index("inf")
    F, R = float(result["paired_delta"][0, ki]), float(result["paired_risk_delta"][0])
    qF, qR = 2. * F / (t * t), 2. * R / (t * t)
    return native(dict(t=t, K_values=result["K_values"], paired_F=F, paired_R=R,
        paired_infinite=float(result["paired_delta"][0, ii]), paired_delta=result["paired_delta"][0].copy(),
        absolute_loss=result["absolute_loss"][0, :, 1].copy(), baseline_absolute_loss=result["absolute_loss"][0, :, 0].copy(),
        absolute_difference=result["absolute_difference"][0].copy(), raw_KL=float(result["raw_KL"][0, 1]),
        baseline_raw_KL=float(result["raw_KL"][0, 0]),
        delta_consistency_error=result["delta_consistency_error"][0].copy(),
        risk_consistency_error=float(result["risk_consistency_error"][0]),
        infinite_risk_consistency_error=float(result["paired_delta"][0, ii] - result["paired_risk_delta"][0]),
        pmf=result["pmf"][1].copy(), baseline_pmf=result["pmf"][0].copy(),
        marginals=result["marginals"][1].copy(), baseline_marginals=result["marginals"][0].copy(),
        marginal_max_difference=float(np.max(abs(result["marginals"][1] - result["marginals"][0]))),
        model_mass=float(result["model_mass"][1]), normal_mass=float(result["normal_quadrature_mass"]),
        target_mass=float(result["q_mass"][0]), all_outputs_finite=all_finite(result),
        finite_gate_exception="logPsi may equal -infinity only to encode Psi=0",
        logPsi_zero_count=int(np.isneginf(result["logPsi"]).sum()),
        PMF_nonnegative=bool(np.all(result["pmf"] >= 0)),
        geometry=geometry(anchor, direction, t, result["b"], result["W1"]),
        quadratic_F=qF, quadratic_R=qR,
        old_curvature=copy.deepcopy(anchor["curvature"]), old_risk_curvature=copy.deepcopy(anchor["risk_curvature"]),
        quadratic_F_minus_old={rule: qF - value for rule, value in anchor["curvature"].items()},
        quadratic_R_minus_old={rule: qR - value for rule, value in anchor["risk_curvature"].items()},
        old_curvature_is_not_a_qualification_gate=True, work_counts=result["work_counts"]))


def qualify_baseline(record, old_D):
    expected_loss = np.array([next(b["loss"] for b in old_D["blocks"] if b["target"] == "population" and b["K"] == K) for K in record["K_values"]])
    old_p = np.asarray(old_D["probability"]["pmf"])
    comparisons = dict(old_1D_D_loss_max_difference=float(np.max(abs(np.asarray(record["absolute_loss"]) - expected_loss))),
        old_1D_D_PMF_max_difference=float(np.max(abs(np.asarray(record["pmf"]) - old_p))))
    gates = dict(outputs_finite=record["all_outputs_finite"], PMF_nonnegative=record["PMF_nonnegative"],
        normal_mass=abs(record["normal_mass"] - 1.) <= GATES["mass"], model_mass=abs(record["model_mass"] - 1.) <= GATES["mass"],
        bias_unchanged=record["geometry"]["bias_exactly_equal"], row_norm=record["geometry"]["row_norm_max_error"] <= GATES["row_norm"],
        old_1D_loss=comparisons["old_1D_D_loss_max_difference"] <= GATES["baseline_loss"],
        old_1D_PMF=comparisons["old_1D_D_PMF_max_difference"] <= GATES["baseline_PMF"])
    return dict(qualified=all(gates.values()), gates=gates, failed_gates=[k for k, ok in gates.items() if not ok], comparisons=comparisons)


def evenness(minus, plus):
    values = dict(F_difference=abs(minus["paired_F"] - plus["paired_F"]), R_difference=abs(minus["paired_R"] - plus["paired_R"]),
                  PMF_max_difference=float(np.max(abs(np.asarray(minus["pmf"]) - np.asarray(plus["pmf"])))))
    gates = dict(F=values["F_difference"] <= GATES["even_F"], R=values["R_difference"] <= GATES["even_R"],
                 PMF=values["PMF_max_difference"] <= GATES["even_PMF"])
    return dict(qualified=all(gates.values()), gates=gates, failed_gates=[k for k, ok in gates.items() if not ok], comparisons=values)


def paired_precision(left, right, effect, comparison):
    diff = abs(left - right)
    relative_limit = GATES[comparison + "_relative"] * max(abs(left), abs(right), GATES["relative_floor"])
    absolute_limit = GATES[effect + "_" + comparison + "_abs"]
    gates = dict(absolute=diff <= absolute_limit, relative=diff <= relative_limit)
    return dict(qualified=all(gates.values()), gates=gates, difference=diff,
                absolute_limit=absolute_limit, relative_limit=relative_limit)


def sign_readout(B, C):
    difference = abs(B - C)
    same = B * C > 0
    floor = max(GATES["sign_absolute_floor"], GATES["sign_difference_factor"] * difference)
    resolved = same and min(abs(B), abs(C)) > floor
    return dict(resolved=resolved, sign=("positive" if B > 0 else "negative") if resolved else "unresolved",
        B=B, C=C, B_C_difference=difference, same_nonzero_sign=same, required_minimum_magnitude=floor,
        probabilistic_or_rigorous_error_bound=False)


def qualify_point(records, baselines, even):
    """All gates remain visible; overall qualification requires both effects."""
    shared, comparisons = {}, {}
    for rule, r in records.items():
        shared[rule + "_baseline"] = baselines[rule]["qualification"]["qualified"]
        shared[rule + "_finite"] = r["all_outputs_finite"]
        shared[rule + "_PMF_nonnegative"] = r["PMF_nonnegative"]
        shared[rule + "_normal_mass"] = abs(r["normal_mass"] - 1.) <= GATES["mass"]
        shared[rule + "_model_mass"] = abs(r["model_mass"] - 1.) <= GATES["mass"]
        shared[rule + "_bias"] = r["geometry"]["bias_exactly_equal"]
        shared[rule + "_row_norm"] = r["geometry"]["row_norm_max_error"] <= GATES["row_norm"]
        shared[rule + "_marginal"] = r["marginal_max_difference"] <= GATES["marginal"]
        shared[rule + "_delta_interface"] = float(np.max(abs(np.asarray(r["delta_consistency_error"])))) <= GATES["interface"]
        shared[rule + "_risk_interface"] = abs(r["risk_consistency_error"]) <= GATES["interface"]
        shared[rule + "_infinite_risk_interface"] = abs(r["infinite_risk_consistency_error"]) <= GATES["interface"]
        shared[rule + "_even_PMF"] = even[rule]["gates"]["PMF"]
    for rule in ("A", "B"):
        row = dict(loss_max_difference=float(np.max(abs(np.asarray(records[rule]["absolute_loss"]) - records["C"]["absolute_loss"]))),
            raw_KL_difference=abs(records[rule]["raw_KL"] - records["C"]["raw_KL"]),
            PMF_max_difference=float(np.max(abs(np.asarray(records[rule]["pmf"]) - records["C"]["pmf"]))))
        comparisons[rule + "_C"] = row
        shared[rule + "_C_absolute_loss"] = row["loss_max_difference"] <= GATES["absolute_loss"]
        shared[rule + "_C_raw_KL"] = row["raw_KL_difference"] <= GATES["raw_KL"]
        shared[rule + "_C_PMF"] = row["PMF_max_difference"] <= GATES["PMF"]
    effects = {}
    for effect in ("F", "R"):
        precision = {pair: paired_precision(records[pair[0]]["paired_" + effect], records["C"]["paired_" + effect], effect, pair) for pair in ("BC", "AC")}
        egates = {"precision_" + pair: value["qualified"] for pair, value in precision.items()}
        egates.update({rule + "_evenness": e["gates"][effect] for rule, e in even.items()})
        qualified = all(shared.values()) and all(egates.values())
        effects[effect] = dict(qualified=qualified, gates=egates, failed_gates=[k for k, ok in egates.items() if not ok],
            precision=precision, sign=sign_readout(records["B"]["paired_" + effect], records["C"]["paired_" + effect]))
    return dict(qualified=all(e["qualified"] for e in effects.values()), qualified_F=effects["F"]["qualified"], qualified_R=effects["R"]["qualified"],
        shared_gates=shared, shared_failed_gates=[k for k, ok in shared.items() if not ok],
        absolute_comparisons=comparisons, evenness=copy.deepcopy(even), effects=effects,
        strict_numeric_error_bound_claimed=False, statistical_confidence_claimed=False)


def outcome_summary(anchors):
    amplitudes = {}
    for amplitude in (.02, .05, .10):
        rows = []
        for anchor in anchors:
            for side in (-1., 1.):
                point = next(p for p in anchor["points"] if p["t"] == side * amplitude)
                q = point["qualification"]; expected = "positive" if anchor["K"] == 11 else "negative"
                supported = q["qualified"] and q["effects"]["F"]["sign"]["sign"] == expected and q["effects"]["R"]["sign"]["sign"] == "negative"
                rows.append(dict(anchor_name=anchor["name"], K=anchor["K"], t=point["t"],
                    qualified=q["qualified"], F_sign=q["effects"]["F"]["sign"]["sign"], R_sign=q["effects"]["R"]["sign"]["sign"],
                    expected_F_sign=expected, expected_R_sign="negative", supports_expected_combination=supported))
        qualified_resolved = all(r["qualified"] and r["F_sign"] != "unresolved" and r["R_sign"] != "unresolved" for r in rows)
        supported = all(r["supports_expected_combination"] for r in rows)
        status = "supported" if supported else ("resolved_different_combination" if qualified_resolved else "unresolved_or_unqualified")
        amplitudes[str(amplitude)] = dict(amplitude=amplitude, status=status, cells=rows, supported=supported)
    return dict(primary_amplitude=.02, primary=amplitudes["0.02"], sensitivities=[amplitudes["0.05"], amplitudes["0.1"]],
        larger_amplitude_does_not_replace_primary=True, fixed_panel_complete_is_stopping_rule=True,
        further_rules_amplitudes_K_or_optimization_added=False)


def new_counts():
    return dict(baseline_preparations=0, nonzero_evaluations=0, reused_zero_records=0,
        failed_kernel_calls=0, high_level_factor_totals={}, new_observations=0,
        new_random_draws=0, optimizer_updates=0, statistical_alpha_spent=0)


def add_counts(counts, work, baseline):
    counts["baseline_preparations" if baseline else "nonzero_evaluations"] += 1
    for key, value in work.items():
        if isinstance(value, int): counts["high_level_factor_totals"][key] = counts["high_level_factor_totals"].get(key, 0) + value


def merge_counts(records):
    """Sum completed/failed workers' known local counters without shared mutation."""
    total = new_counts()
    for counts in records:
        if counts is None: continue
        for key, value in counts.items():
            if isinstance(value, int): total[key] += value
        for key, value in counts["high_level_factor_totals"].items():
            total["high_level_factor_totals"][key] = total["high_level_factor_totals"].get(key, 0) + value
    return total


def verify_freeze():
    frozen, source = read(FREEZE), read(SOURCE)
    if frozen["config"] != config_dump() or frozen["source_sha256"] != sha(SOURCE):
        raise ValueError("SOURCE/configuration does not match FREEZE")
    required = [Path(__file__), Path(__file__).with_name("noncollinear_probe_compute_test.py"),
        Path(kernel.__file__), Path(kernel.__file__).with_name("noncollinear_probe_kernel_test.py"),
        DOC / "DESIGN.md", DOC / "DERIVATION.md", DOC / "HANDOFF_DISPOSITION.md", SOURCE, OUT / "SCOPE.json"]
    for path in required:
        if rel(path) not in frozen["sources"]: raise ValueError("required source not frozen: " + rel(path))
    for mapping in (frozen["sources"], source["sources"]):
        for name, digest in mapping.items():
            if not isinstance(digest, str) or sha(ROOT / name) != digest: raise ValueError("changed frozen source: " + name)
    if source["schema"] != "noncollinear_probe_source_v1" or [a["K"] for a in source["anchors"]] != [11, 12]:
        raise ValueError("expected the fixed K11/K12 source anchors")
    q, d = np.asarray(source["teacher_pmf"]), np.asarray(source["direction"])
    if q.shape != (256,) or not np.isfinite(q).all() or np.any(q < 0) or abs(float(q.sum()) - 1.) > 1e-12:
        raise ValueError("invalid unchanged teacher weights")
    if d.shape != (8,) or not np.isfinite(d).all(): raise ValueError("invalid fixed direction")
    previous_source = "artifacts/rankone_continuation_v1_20261007/SOURCE.json"
    public_paths = [source["cross_plan"], source["cross_results"], previous_source]
    public_paths.extend(anchor["one_d"][rule] for anchor in source["anchors"] for rule in ("C", "D"))
    if not all(path in source["sources"] for path in public_paths):
        raise ValueError("a required previous public input is not source-bound")
    if not np.array_equal(q, np.asarray(read(ROOT / previous_source)["teacher_pmf"])):
        raise ValueError("teacher weights differ from the previous frozen source")
    plan, cross = read(ROOT / source["cross_plan"]), read(ROOT / source["cross_results"])
    if not np.array_equal(d, plan["direction"]) or [plan["lower_K"], plan["upper_K"]] != [11, 12]:
        raise ValueError("direction/K do not match old cross lock")
    for ai, anchor in enumerate(source["anchors"]):
        theta = np.asarray(anchor["theta"])
        expected = plan["theta_L" if ai == 0 else "theta_H"]
        if theta.shape != (16,) or not np.array_equal(theta, expected) or not np.isfinite(theta).all():
            raise ValueError("anchor differs from the old locked point")
        for rule in ("C", "D"):
            summary = read(ROOT / anchor["one_d"][rule])
            if not np.array_equal(summary["theta"], theta): raise ValueError("old 1D parameter mismatch")
            cell = cross["cells"]["LL" if ai == 0 else "HH"]
            if anchor["curvature"][rule] != cell[rule + "_curvature"]: raise ValueError("old finite curvature mismatch")
            infinity = next(b for b in summary["blocks"] if b["target"] == "population" and b["K"] == "inf")
            risk_c = float(d @ np.asarray(infinity["Hangle"]) @ d)
            if anchor["risk_curvature"][rule] != risk_c: raise ValueError("old risk curvature algebra mismatch")
    return frozen, copy.deepcopy(source)


def panel_lock(source):
    rows = []
    for anchor in source["anchors"]:
        theta = np.asarray(anchor["theta"]); m = len(theta) // 2
        rows.append(dict(name=anchor["name"], K=anchor["K"], theta=theta.copy(),
            points=[dict(point_id=name, t=t, b=theta[:m].copy(), W=kernel.path_W(theta[m:], source["direction"], t)) for name, t in POINTS]))
    return dict(created_utc=now(), anchors=rows, direction=copy.deepcopy(source["direction"]),
        rules=copy.deepcopy(RULES), source_sha256=sha(SOURCE), freeze_sha256=sha(FREEZE),
        locked_before_any_2D_evaluation=True, no_optimization=True)


def execute_rule(anchor, q, direction, rule, folder, counts, *, settings=None,
                 prepare=None, evaluate=None, old_D=None):
    """One preparation and six independently requested perturbations.

    Optional callables/settings support manufactured tests only. The public
    scientific --run entry point supplies the frozen RULES and kernel APIs.
    """
    settings = copy.deepcopy(RULES[rule] if settings is None else settings)
    prepare = kernel.prepare_baseline if prepare is None else prepare
    evaluate = kernel.evaluate_prepared if evaluate is None else evaluate
    theta = np.asarray(anchor["theta"]).copy(); m = len(theta) // 2
    b, a, d = theta[:m].copy(), theta[m:].copy(), np.asarray(direction).copy()
    started = time.perf_counter(); records = {}; active = "baseline"
    try:
        tick = time.perf_counter()
        try:
            prepared = prepare(b.copy(), kernel.path_W(a, d, 0.), {"population": np.asarray(q).copy()},
                K_values=(anchor["K"], "inf"), **settings, outer_chunk=32, label_chunk=8)
        except BaseException:
            counts["failed_kernel_calls"] += 1; raise
        seconds = time.perf_counter() - tick
        add_counts(counts, prepared["work_counts"], True)
        baseline_fingerprint = fingerprint(prepared)
        path = folder / "BASELINE.npz"; save_npz(path, arrays_for_save(prepared, anchor, d, 0.))
        baseline = baseline_summary(prepared, anchor, d)
        if old_D is None: old_D = read(ROOT / anchor["one_d"]["D"])
        baseline.update(rule=rule, anchor_name=anchor["name"], K=anchor["K"], point_id="zero", seconds=seconds,
            array_artifact=artifact(path), prepared_fingerprint=baseline_fingerprint)
        baseline["qualification"] = qualify_baseline(baseline, old_D)
        save(folder / "BASELINE.json", baseline)
        baseline["summary_artifact"] = artifact(folder / "BASELINE.json")
        records["zero"] = baseline
        counts["reused_zero_records"] += 1
        for point_id, t in POINTS[1:]:
            active = point_id; W = kernel.path_W(a, d, t); tick = time.perf_counter()
            try: result = evaluate(prepared, W.copy())
            except BaseException:
                counts["failed_kernel_calls"] += 1; raise
            seconds = time.perf_counter() - tick
            add_counts(counts, result["work_counts"], False)
            path = folder / (point_id + ".npz"); save_npz(path, arrays_for_save(result, anchor, d, t))
            record = point_summary(result, anchor, d, t)
            record.update(rule=rule, anchor_name=anchor["name"], K=anchor["K"], point_id=point_id,
                array_artifact=artifact(path), baseline_array_artifact=baseline["array_artifact"],
                prepared_fingerprint=baseline_fingerprint, seconds=seconds, independently_evaluated=True)
            save(folder / (point_id + ".json"), record)
            record["summary_artifact"] = artifact(folder / (point_id + ".json"))
            records[point_id] = record
            print(json.dumps(dict(anchor=anchor["name"], rule=rule, t=t, seconds=seconds)), flush=True)
        if fingerprint(prepared) != baseline_fingerprint: raise RuntimeError("prepared baseline mutated during its six evaluations")
        receipt = dict(rule=rule, anchor_name=anchor["name"], baseline_fingerprint=baseline_fingerprint,
            prepared_unchanged=True, baseline_preparations=1, independently_evaluated_nonzero_points=6,
            zero_additional_evaluations=0, point_artifacts={name: record["summary_artifact"] for name, record in records.items()},
            seconds=time.perf_counter() - started)
        save(folder / "RULE_COMPLETE.json", receipt)
        return records
    except BaseException as error:
        save(folder / "FAILURE.json", dict(anchor_name=anchor["name"], rule=rule, active=active,
            reason=str(error), traceback=traceback.format_exc(), completed_point_ids=list(records),
            seconds=time.perf_counter() - started, automatic_retry=False,
            failed_kernel_internal_partial_work_not_counted=True))
        raise


def summarize_anchor(anchor, by_rule):
    baselines = {rule: rows["zero"] for rule, rows in by_rule.items()}
    even = {}
    for suffix in ("002", "005", "010"):
        even[suffix] = {rule: evenness(rows["m" + suffix], rows["p" + suffix]) for rule, rows in by_rule.items()}
    points = [dict(point_id="zero", t=0., baseline_reused=True, rules=baselines,
        qualified=all(b["qualification"]["qualified"] for b in baselines.values()), signs_not_defined_at_zero=True)]
    for point_id, t in POINTS[1:]:
        records = {rule: rows[point_id] for rule, rows in by_rule.items()}
        points.append(dict(point_id=point_id, t=t, rules=records,
            qualification=qualify_point(records, baselines, even[point_id[1:]])))
    return dict(name=anchor["name"], K=anchor["K"], theta=copy.deepcopy(anchor["theta"]),
        curvature=copy.deepcopy(anchor["curvature"]), risk_curvature=copy.deepcopy(anchor["risk_curvature"]),
        baselines=baselines, evenness=even, points=points, all_nonzero_points_qualified=all(p["qualification"]["qualified"] for p in points[1:]))


def execute_anchor(anchor, q, direction, folder, *, verify=None, rule_runner=None):
    """One worker owns one anchor directory, count object and all local caches."""
    verify = verify_freeze if verify is None else verify
    rule_runner = execute_rule if rule_runner is None else rule_runner
    anchor, q, direction = copy.deepcopy(anchor), np.asarray(q).copy(), np.asarray(direction).copy()
    counts, by_rule = new_counts(), {}
    active = None; started = time.perf_counter()
    try:
        folder.mkdir(parents=True, exist_ok=False)
        for rule in RULES:
            active = rule; verify()
            rule_folder = folder / rule; rule_folder.mkdir(exist_ok=False)
            by_rule[rule] = rule_runner(anchor, q, direction, rule, rule_folder, counts)
            verify()
        summary = summarize_anchor(anchor, by_rule)
        path = folder / "RESULTS.json"; save(path, summary)
        summary["result_artifact"] = artifact(path)
        execution = dict(status="completed", anchor_name=anchor["name"], resources=counts,
            rules_completed=list(by_rule), seconds=time.perf_counter() - started,
            mutable_counts_or_baselines_shared_with_other_anchor=False)
        save(folder / "EXECUTION.json", execution)
        return dict(**execution, summary=summary, execution_artifact=artifact(folder / "EXECUTION.json"))
    except BaseException as error:
        failure = dict(status="failed", anchor_name=anchor["name"], active_rule=active,
            reason=str(error), traceback=traceback.format_exc(), resources=counts,
            rules_completed=list(by_rule), seconds=time.perf_counter() - started,
            automatic_retry=False, failed_kernel_internal_partial_work_not_counted=True,
            other_started_anchor_may_complete=True)
        # Each worker writes only its own failure receipt. A failure to write
        # this receipt is returned by the executor as an infrastructure error.
        if folder.exists() and not (folder / "FAILURE.json").exists(): save(folder / "FAILURE.json", failure)
        return dict(**failure, failure_artifact=artifact(folder / "FAILURE.json") if (folder / "FAILURE.json").is_file() else None)


def parallel_anchors(anchors, worker):
    """Start both independent jobs, await both, return in SOURCE order.

    No future is cancelled/retried when its peer fails. Worker errors are
    retained; unexpected infrastructure exceptions have explicitly unknown
    local counters rather than invented zero work.
    """
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="probe-anchor") as executor:
        futures = [executor.submit(worker, copy.deepcopy(anchor)) for anchor in anchors]
        outcomes = []
        for anchor, future in zip(anchors, futures):
            try: outcomes.append(future.result())
            except BaseException as error:
                outcomes.append(dict(status="failed", anchor_name=anchor["name"], reason=str(error),
                    traceback=traceback.format_exc(), resources=None, local_counters_unavailable=True,
                    automatic_retry=False, other_started_anchor_may_complete=True))
    return outcomes


def run():
    if any((OUT / name).exists() for name in ("RUN_START.json", "PANEL_LOCK.json", "anchors", "RESULTS.json")):
        raise FileExistsError("panel output exists and will not be replaced")
    try: _, source = verify_freeze()
    except BaseException as error:
        if not (OUT / "START_REJECTED.json").exists():
            save(OUT / "START_REJECTED.json", dict(created_utc=now(), reason=str(error), traceback=traceback.format_exc(), scientific_calls=0))
        raise
    started = time.perf_counter(); counts = new_counts(); summaries = []; worker_outcomes = []
    save(OUT / "RUN_START.json", dict(created_utc=now(), source_sha256=sha(SOURCE), freeze_sha256=sha(FREEZE),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform())))
    save(OUT / "PANEL_LOCK.json", panel_lock(source))
    try:
        def worker(anchor):
            return execute_anchor(anchor, source["teacher_pmf"], source["direction"], OUT / "anchors" / anchor["name"])
        worker_outcomes = parallel_anchors(source["anchors"], worker)
        counts = merge_counts([result["resources"] for result in worker_outcomes])
        summaries = [result["summary"] for result in worker_outcomes if result["status"] == "completed"]
        save(OUT / "WORKERS.json", dict(anchor_workers=2, source_order=[a["name"] for a in source["anchors"]],
            outcomes=[{key: value for key, value in result.items() if key != "summary"} for result in worker_outcomes],
            all_completed=all(result["status"] == "completed" for result in worker_outcomes),
            all_started_workers_awaited=True, automatic_retry=False))
        if any(result["status"] != "completed" for result in worker_outcomes):
            raise RuntimeError("one or more independent anchor workers failed; all started workers were awaited")
        if counts["baseline_preparations"] != 6 or counts["nonzero_evaluations"] != 36 or counts["reused_zero_records"] != 6:
            raise RuntimeError("fixed panel call counts do not match 6 preparations +36 perturbations")
        outcome = outcome_summary(summaries); verify_freeze()
        save(OUT / "RESULTS.json", dict(status="completed", created_utc=now(), source_sha256=sha(SOURCE), freeze_sha256=sha(FREEZE),
            panel_lock_artifact=artifact(OUT / "PANEL_LOCK.json"), anchors=summaries, outcome=outcome,
            workers_artifact=artifact(OUT / "WORKERS.json"), anchor_workers=2,
            resources=counts, seconds=time.perf_counter() - started,
            artifacts={rel(p): sha(p) for p in OUT.rglob("*") if p.is_file() and p != OUT / "RESULTS.json"},
            scope=dict(current_local_mechanism_diagnosis_closed_after_fixed_panel=True,
                direct_finite_loss_from_paired_integral=True, Gaussian_remainder_certified=False,
                exact_stationary_existence_proved=False, optimizer_updates=0, additional_2D_optimization=False,
                new_random_learning_streams=0, extra_rules_or_amplitudes=False,
                third_tier_enhancement_started=False, new_teachers_or_seeds=False,
                manuscript_changed=False, external_scientific_review_completed=False,
                next_scientific_review_and_manuscript_decision_belong_to_user=True)))
        print(json.dumps(dict(completed=True, primary_outcome=outcome["primary"]["status"], resources=counts)), flush=True)
    except BaseException as error:
        save(OUT / "FAILURE.json", dict(reason=str(error), traceback=traceback.format_exc(),
            completed_anchors=summaries, resources=counts, seconds=time.perf_counter() - started,
            worker_outcomes=worker_outcomes,
            resources_complete=all(result.get("resources") is not None for result in worker_outcomes),
            all_started_workers_awaited=True,
            automatic_retry=False, failed_kernel_internal_partial_work_not_counted=True))
        raise


def self_test():
    import noncollinear_probe_compute_test as tests
    return tests.run_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", action="store_true"); mode.add_argument("--self-test", action="store_true"); mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.config: print(json.dumps(config_dump(), sort_keys=True), flush=True)
    elif args.self_test:
        outcome = self_test(); print(json.dumps(outcome, sort_keys=True), flush=True)
        raise SystemExit(0 if outcome["passed"] else 1)
    else: run()
