"""Independent saved-JSON verification for the budget-sensitivity analysis.

No analyzer import, model kernels, random draws, fitting, NPZ loading, or changes
to evidence. The verifier reconstructs declared results from contract-bound
JSON/JSONL and compares the saved analysis. Numerical-risk qualifications remain
ordinary saved GH order checks, not independently recomputed integrals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
CASES = tuple(range(2000, 2024))
SETTINGS = ("P0", "H")
METHODS = ("integrated", "group8", "ridge")
N = 2048
ORDERS = [[28, 192], [36, 256]]
KL_TOL = 1e-4 / N
PMF_TOL = 1e-8
MAPPING_TOL = 3e-4
GRADIENT_SE_TOL = 1e-4
SCHEMA = "independent_budget_saved_analysis_verification_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def report_safe(value):
    """Keep malformed nonfinite evidence explicit without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_saved_value": repr(value)}
    if isinstance(value, dict):
        return {str(key): report_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [report_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [report_safe(item) for item in value]
    return value


def relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("repository-relative binding required: " + str(name))
    result = (REPO / path).resolve()
    if not result.is_relative_to(REPO.resolve()):
        raise ValueError("bound path escapes repository: " + str(name))
    return result


class Audit:
    def __init__(self):
        self.errors = []
        self.unresolved = []
        self.comparisons = 0

    def check(self, condition, where, expected=None, actual=None):
        self.comparisons += 1
        if not condition:
            self.errors.append({"where": where, "expected": expected, "actual": actual})

    def equal(self, actual, expected, where):
        self.check(actual == expected, where, expected, actual)

    def number(self, actual, expected, where):
        if expected is None:
            self.equal(actual, None, where)
        else:
            self.check(finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                       where, expected, actual)

    def vector(self, actual, expected, where):
        valid = isinstance(actual, list) and len(actual) == len(expected)
        self.check(valid, where + "/length", len(expected), None if not isinstance(actual, list) else len(actual))
        if valid:
            for index, value in enumerate(expected):
                self.number(actual[index], value, f"{where}/{index}")

    def pending(self, where, reason):
        self.unresolved.append({"where": where, "reason": reason})


def describe(values, interval=False):
    if len(values) != 24 or not all(finite(value) for value in values):
        return None
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(24)
    result = {"n": 24, "mean": mean, "sample_SD": sd, "SE": se, "df": 23}
    if interval:
        critical = float(t.ppf(0.9875, 23))
        result.update(critical_value=critical, family_alpha=.05, family_size=2,
                      lower=mean-critical*se, upper=mean+critical*se,
                      method="approximate_Bonferroni_simultaneous_two_sided_Student_t",
                      equivalence_conclusion=False)
    return result


def compare_description(audit, saved, expected, where):
    if expected is None:
        audit.equal(saved, None, where)
        return
    audit.check(isinstance(saved, dict), where, "description object", saved)
    if not isinstance(saved, dict):
        return
    for key, value in expected.items():
        if finite(value):
            audit.number(saved.get(key), value, where + "/" + key)
        else:
            audit.equal(saved.get(key), value, where + "/" + key)


def slot_eligible(row, case, setting, method, init):
    if row is None:
        return False
    diagnostic = row.get("precision_diagnostics", {})
    return (tuple(row.get(k) for k in ("case_id", "setting", "method", "initialization_index")) ==
            (case, setting, method, init) and row.get("N") == N and row.get("evaluable") is True and
            finite(row.get("joint_kl")) and row.get("precision_qualified") is True and
            [level.get("orders") for level in row.get("risk_levels", [])] == ORDERS and
            all(finite(diagnostic.get(key)) and 0 <= diagnostic[key] <= limit for key, limit in
                (("KL_order_difference", KL_TOL), ("PMF_order_max_difference", PMF_TOL))))


def reconstruct_risks(records, audit):
    slots, paired, extras = {}, [], []
    expected_keys = {(setting, method, init) for setting in SETTINGS for method in METHODS for init in (0, 1)}
    for case in CASES:
        record = records.get(case)
        if record is not None:
            audit.equal(record.get("case_id"), case, f"case/{case}/identity")
        index = {}
        for row in [] if record is None else record.get("risks", []):
            key = tuple(row.get(k) for k in ("setting", "method", "initialization_index"))
            index.setdefault(key, []).append(row)
            if key not in expected_keys:
                extras.append((case, row))
        means = {}
        case_ok = not any(key not in expected_keys for key in index)
        for setting in SETTINGS:
            for method in METHODS:
                risks = []
                for init in (0, 1):
                    rows = index.get((setting, method, init), [])
                    row = rows[0] if len(rows) == 1 else None
                    eligible = slot_eligible(row, case, setting, method, init)
                    key = (case, setting, method, init)
                    slots[key] = {"row": row, "eligible": eligible, "duplicates": rows if len(rows) > 1 else []}
                    case_ok = case_ok and eligible
                    risks.append(row.get("joint_kl") if row is not None and finite(row.get("joint_kl")) else None)
                    if row is not None and row.get("evaluable") is True:
                        levels = row.get("risk_levels", [])
                        if len(levels) == 2 and all(finite(level.get("joint_kl")) for level in levels):
                            audit.number(row.get("joint_kl"), levels[-1]["joint_kl"], f"risk/{key}/reported_high_order_KL")
                            audit.number(row.get("precision_diagnostics", {}).get("KL_order_difference"),
                                         abs(levels[1]["joint_kl"]-levels[0]["joint_kl"]), f"risk/{key}/saved_KL_order_arithmetic")
                        else:
                            audit.check(False, f"risk/{key}/two_finite_KL_levels", ORDERS, levels)
                means[f"{setting}_{method}_scaled_risk"] = N * statistics.mean(risks) if all(finite(x) for x in risks) else None
        def diff(a, b):
            return means[a]-means[b] if means[a] is not None and means[b] is not None else None
        dp0 = diff("P0_group8_scaled_risk", "P0_integrated_scaled_risk")
        dh = diff("H_group8_scaled_risk", "H_integrated_scaled_risk")
        paired.append({"case_id": case, **means, "D_P0": dp0, "D_H": dh,
                       "G": dh-dp0 if dh is not None and dp0 is not None else None,
                       "all_expected_risks_qualified": bool(case_ok)})
    complete = not extras and all(slot["eligible"] for slot in slots.values())
    if not complete:
        audit.pending("risks", "Some of the 288 expected risk slots are missing/unqualified, duplicated, or accompanied by unexpected rows.")
    primary = {}
    for quantity in ("D_H", "G"):
        settings = ("H",) if quantity == "D_H" else SETTINGS
        required = [slot for (_, setting, method, _), slot in slots.items()
                    if setting in settings and method in ("integrated", "group8")]
        qualified = all(slot["eligible"] for slot in required)
        values = [row[quantity] for row in paired]
        primary[quantity] = {"qualified": qualified, "required": len(required),
            "qualified_slots": sum(slot["eligible"] for slot in required), "values": values,
            "description": describe(values), "interval": describe(values, True) if qualified else None}
    return {"slots": slots, "paired": paired, "complete": complete, "primary": primary, "extras": extras}


def verify_risk_outputs(computed, analysis, summary, csv_rows, audit):
    audit.equal(analysis.get("all_24_cases_complete_and_all_288_risks_qualified"), computed["complete"], "risk/all_288_complete")
    audit.equal(summary.get("all_288_risks_qualified"), computed["complete"], "summary/all_288_complete")
    audit.equal([row for row in analysis.get("raw_risk_rows", []) if row.get("unexpected")],
                [{"case_id": case, "unexpected": True, "raw": row} for case, row in computed["extras"]],
                "risk/unexpected_records_preserved")
    raw_index = {}
    for row in analysis.get("raw_risk_rows", []):
        if not row.get("unexpected"):
            key = tuple(row.get(k) for k in ("case_id", "setting", "method", "initialization_index"))
            raw_index.setdefault(key, []).append(row)
    audit.equal(set(raw_index), set(computed["slots"]), "risk/288_slot_identities")
    for key, slot in computed["slots"].items():
        saved = raw_index.get(key, [])
        audit.equal(len(saved), 1, f"risk/{key}/single_analysis_slot")
        if len(saved) == 1:
            audit.equal(saved[0].get("raw"), slot["row"], f"risk/{key}/raw_record_preserved")
            audit.equal(saved[0].get("duplicate_rows_retained", []), slot["duplicates"], f"risk/{key}/duplicate_evidence")
            audit.equal(saved[0].get("valid_for_complete_analysis"), slot["eligible"], f"risk/{key}/qualification")
    for source_name, saved_rows in (("json", analysis.get("paired_case_rows", [])), ("csv", csv_rows)):
        indexed = {}
        for row in saved_rows:
            indexed.setdefault(int(row["case_id"]), []).append(row)
        audit.equal(set(indexed), set(CASES), f"paired/{source_name}/24_cases")
        for row in computed["paired"]:
            found = indexed.get(row["case_id"], [])
            audit.equal(len(found), 1, f"paired/{source_name}/{row['case_id']}/unique")
            if len(found) != 1:
                continue
            for key, expected in row.items():
                value = found[0].get(key)
                if source_name == "csv":
                    value = None if value == "" else value == "True" if isinstance(expected, bool) else float(value)
                if isinstance(expected, bool):
                    audit.equal(value, expected, f"paired/{source_name}/{row['case_id']}/{key}")
                else:
                    audit.number(value, expected, f"paired/{source_name}/{row['case_id']}/{key}")
    for name, expected in computed["primary"].items():
        for source_name, container in (("risk", analysis), ("summary", summary)):
            saved = container.get("primary", {}).get(name, {})
            prefix = source_name + "/" + name
            audit.equal(saved.get("required_risk_slots"), expected["required"], prefix + "/required_slots")
            audit.equal(saved.get("qualified_required_risk_slots"), expected["qualified_slots"], prefix + "/qualified_slots")
            audit.equal(saved.get("finite_case_count"), sum(x is not None for x in expected["values"]), prefix + "/finite_cases")
            audit.equal(saved.get("status"), "complete_qualified_exploratory_interval" if expected["qualified"] else "unresolved_no_primary_interval", prefix + "/status")
            saved_values = saved.get("case_values", [])
            audit.equal([row.get("case_id") for row in saved_values], list(CASES), prefix + "/ordered_cases")
            if len(saved_values) == 24:
                for case, row, value in zip(CASES, saved_values, expected["values"]):
                    audit.number(row.get("value"), value, f"{prefix}/{case}/value")
            compare_description(audit, saved.get("complete_vector_description"), expected["description"], prefix + "/description")
            compare_description(audit, saved.get("simultaneous_interval"), expected["interval"], prefix + "/interval")
            audit.equal(saved.get("diagnostic_failures_not_filtered"), True, prefix + "/no_diagnostic_filter")
    for key in ["D_P0"] + [f"{s}_{m}_scaled_risk" for s in SETTINGS for m in METHODS]:
        values = [row[key] for row in computed["paired"]]
        saved = analysis.get("descriptive", {}).get(key, {})
        compare_description(audit, saved.get("complete_vector_description"), describe(values), "descriptive/" + key)
        audit.equal(saved.get("finite_case_count"), sum(value is not None for value in values), "descriptive/" + key + "/finite_cases")
        audit.equal(saved.get("inferential_interval"), None, "descriptive/" + key + "/no_extra_interval")
    audit.equal(summary.get("risk_complete"), all(p["qualified"] for p in computed["primary"].values()), "summary/primary_complete")


def finite_vector(value, size=3):
    return isinstance(value, list) and len(value) == size and all(finite(x) for x in value)


def verify_diagnostic_identities(row, audit, where):
    base, prefix = row.get("base", {}), row.get("prefix", {})
    gradient, se = base.get("gradient"), base.get("gradient_scramble_SE")
    if not (finite_vector(gradient) and finite_vector(se)):
        audit.pending(where, "Missing finite GROUP gradient/SE vector.")
        return
    audit.check(all(x >= 0 for x in se), where + "/nonnegative_SE", "all >= 0", se)
    audit.number(base.get("maximum_gradient_SE"), max(se), where + "/maximum_SE")
    mapping = base.get("projected_mapping", {})
    if finite_vector(mapping.get("mapping")):
        audit.number(mapping.get("norm_inf"), max(abs(x) for x in mapping["mapping"]), where + "/mapping_norm")
    else:
        audit.pending(where + "/mapping", "No finite saved projection-mapping vector; projection itself is not recomputed.")
    delta = row.get("full_minus_prefix", {})
    if finite_vector(prefix.get("gradient")) and finite_vector(delta.get("gradient")):
        audit.vector(delta["gradient"], [a-b for a, b in zip(gradient, prefix["gradient"])], where + "/full_minus_prefix_gradient")
        if finite(base.get("value")) and finite(prefix.get("value")):
            audit.number(delta.get("value"), base["value"]-prefix["value"], where + "/full_minus_prefix_value")
        audit.equal(prefix.get("B"), 1024, where + "/prefix_B")
    else:
        audit.pending(where + "/prefix", "Missing finite saved prefix comparison.")
    levels = row.get("moment_levels", [])
    audit.equal(sorted(level.get("outer_order") for level in levels), [192, 256, 384], where + "/moment_orders")
    for level in levels:
        order = level.get("outer_order")
        audit.equal(level.get("same_inner_order"), 256, f"{where}/moment{order}/inner")
        shift = level.get("gradient_shift")
        if not finite_vector(shift) or not finite(level.get("value_shift")):
            audit.check(False, f"{where}/moment{order}/finite_shifts", "finite gradient/value shift", level)
            continue
        audit.vector(level.get("gradient"), [a+b for a, b in zip(gradient, shift)], f"{where}/moment{order}/gradient_identity")
        audit.number(level.get("value"), base["value"]+level["value_shift"], f"{where}/moment{order}/value_identity")
        audit.vector(level.get("gradient_scramble_SE"), se, f"{where}/moment{order}/unchanged_replicate_SE")
        if order == 192:
            audit.vector(shift, [0., 0., 0.], where + "/moment192/zero_gradient_shift")
            audit.number(level["value_shift"], 0., where + "/moment192/zero_value_shift")


def verify_diagnostics(records, saved, summary, audit):
    indexed, rows = {}, []
    for case in CASES:
        for row in (records.get(case) or {}).get("group_diagnostics", []):
            point = row.get("point", {})
            key = (case, point.get("setting"), point.get("initialization_index"))
            indexed.setdefault(key, []).append(row)
            rows.append((case, row))
    expected = {(case, setting, init) for case in CASES for setting in SETTINGS for init in (0, 1)}
    raw_saved = saved.get("rows", [])
    audit.equal([row.get("raw") for row in raw_saved], [row for _, row in rows], "diagnostics/all_raw_rows_preserved")
    extra_identity = set(indexed)-expected
    if extra_identity:
        audit.pending("diagnostics", "Unexpected diagnostic identities are retained; complete counts are not certified.")
    by_setting = {}
    for setting in SETTINGS:
        selected, consistent = [], not extra_identity
        for case in CASES:
            for init in (0, 1):
                found = indexed.get((case, setting, init), [])
                if len(found) != 1:
                    consistent = False
                for row in found:
                    point, base = row.get("point", {}), row.get("base", {})
                    if point.get("case_id") != case or point.get("method") != "group8":
                        consistent = False
                        continue
                    mapping = base.get("projected_mapping", {}).get("norm_inf")
                    se = base.get("maximum_gradient_SE")
                    numeric = finite(mapping) and finite(se)
                    own = point.get("fit_status") == "mapping_tolerance"
                    mp, sp = bool(numeric and mapping <= MAPPING_TOL), bool(numeric and se <= GRADIENT_SE_TOL)
                    if numeric:
                        audit.check(mapping >= 0 and se >= 0, f"diagnostics/{case}/{setting}/{init}/nonnegative_gates", ">=0", [mapping, se])
                    consistency = numeric and bool(base.get("mapping_pass")) == mp and bool(base.get("gradient_SE_pass")) == sp
                    consistent = consistent and consistency
                    if row.get("evaluable") is True:
                        audit.equal(base.get("fit_status_pass"), own, f"diagnostics/{case}/{setting}/{init}/own_flag")
                        audit.equal(base.get("composite_diagnostic_pass"), own and mp and sp, f"diagnostics/{case}/{setting}/{init}/composite_flag")
                        verify_diagnostic_identities(row, audit, f"diagnostics/{case}/{setting}/{init}")
                    category = ("neither_numeric_gate_fails" if mp and sp else "mapping_only" if not mp and sp else "SE_only" if mp and not sp else "both") if numeric else "missing_numeric_diagnostic"
                    selected.append({"case": case, "init": init, "own": own, "composite": own and mp and sp,
                                     "fit_status": point.get("fit_status"), "category": category, "raw": row})
        unique = {(r["case"], r["init"]) for r in selected}
        identity_complete = len(unique) == len(selected) == 48
        consistent = bool(consistent and identity_complete)
        categories = Counter(r["category"] for r in selected)
        counts = {"composite_pass": sum(r["composite"] for r in selected), "own_status_pass": sum(r["own"] for r in selected),
                  "own_status_counts": dict(Counter(str(r["fit_status"]) for r in selected)),
                  "numeric_failure_counts": {key: categories[key] for key in
                  ("neither_numeric_gate_fails", "mapping_only", "SE_only", "both", "missing_numeric_diagnostic")}}
        by_setting[setting] = {"complete": consistent, "counts": counts, "unique": len(unique), "observed": len(selected)}
        for name, target in (("diagnostics", saved.get("by_setting", {}).get(setting, {})),
                             ("summary", summary.get("diagnostics", {}).get(setting, {}))):
            where = name + "/" + setting
            audit.equal(target.get("observed_rows"), len(selected), where + "/observed_rows")
            audit.equal(target.get("unique_observed_outputs"), len(unique), where + "/unique_rows")
            audit.equal(target.get("complete_count_summary_usable"), consistent, where + "/count_qualification")
            audit.equal(target.get("identity_complete"), identity_complete, where + "/identity_complete")
            for key, value in counts.items():
                audit.equal(target.get(key), value if consistent else None, where + "/" + key)
            audit.equal(target.get("observed_counts_uncertified_if_incomplete"), None if consistent else counts, where + "/unqualified_counts_retained")
            prefix_values = [max(abs(x) for x in r["raw"]["full_minus_prefix"]["gradient"]) for r in selected
                             if finite_vector((r["raw"].get("full_minus_prefix") or {}).get("gradient"))]
            audit.equal(target.get("nested_prefix_rows"), len(prefix_values), where + "/prefix_rows")
            audit.number(target.get("nested_full_minus_prefix_gradient_max"), max(prefix_values, default=None), where + "/prefix_max")
            for order in (192, 256, 384):
                changes = [m for r in selected for m in r["raw"].get("moment_levels", []) if m.get("outer_order") == order]
                target_level = target.get("moment_changes", {}).get(str(order), {})
                audit.equal(target_level.get("row_count"), len(changes), f"{where}/moment{order}/count")
                if all(finite_vector(m.get("gradient_shift")) and finite(m.get("value_shift")) for m in changes):
                    audit.number(target_level.get("maximum_absolute_gradient_shift"), max((max(abs(x) for x in m["gradient_shift"]) for m in changes), default=None), f"{where}/moment{order}/gradient_max")
                    audit.number(target_level.get("maximum_absolute_value_shift"), max((abs(m["value_shift"]) for m in changes), default=None), f"{where}/moment{order}/value_max")
        if not consistent:
            audit.pending("diagnostics/" + setting, "The 48 identities and saved numerical gates are not complete and consistent.")
    audit.equal(summary.get("diagnostic_issues"), saved.get("issues"), "summary/diagnostic_issues_preserved")
    return by_setting


def verify_costs(cfg, saved, summary, audit):
    counts, phases, receipt_rows, parents = Counter(), Counter(), [], set()
    for binding in cfg.get("budget_receipts", []):
        path = relative(binding["path"])
        audit.check(path.name in ("SUMMARY.json", "FAILED.json"), "budget/terminal_receipt_only", "SUMMARY.json or FAILED.json", str(path))
        audit.check(path.parent not in parents, "budget/unique_phase_directory", "one terminal receipt per run", str(path.parent))
        parents.add(path.parent)
        record = read(path)
        budget = record["budget"]
        for name, value in budget["counts"].items():
            audit.check(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "budget/nonnegative_integer/" + name, ">=0 integer", value)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                counts[name] += value
        phases[record.get("phase")] += 1
        receipt_rows.append({**binding, "phase": record.get("phase"), "status": record.get("status", "failed"), "budget": budget})
        if record.get("status") != "completed":
            audit.pending("budget/" + binding["path"], "Failed or incomplete phase is retained in actual work counts.")
    expected_phases = {"profile": 1, "fixed_diagnostics": 1, "fit": 4, "evaluation": 4}
    if dict(phases) != expected_phases:
        audit.pending("budget/phase_coverage", "Expected 10 completed phase receipts: profile1, fixed_diagnostics1, fit4, evaluation4; observed " + str(dict(phases)))
    audit.equal(saved.get("phase_receipts"), receipt_rows, "budget/raw_terminal_receipts")
    audit.equal(saved.get("summed_actual_saved_work_counts"), dict(counts), "budget/independent_sum")
    identities, paths, trace_rows = set(), set(), []
    for name in cfg.get("accepted_trace_files", []):
        path = relative(name)
        match_case = re.fullmatch(r"case_(\d{5})", path.parent.parent.name)
        match_fit = re.fullmatch(r"FIT_(integrated|group8|ridge)_init([01])", path.parent.name)
        identity = (int(match_case[1]), match_fit[1], int(match_fit[2])) if match_case and match_fit and path.name == "ACCEPTED_TRACE.jsonl" else None
        audit.check(identity is not None and identity[0] in CASES, "trace/valid_identity", "case/method/start", name)
        audit.check(path not in paths and identity not in identities, "trace/no_duplicate_identity", "unique physical path and identity", name)
        paths.add(path)
        if identity is not None:
            identities.add(identity)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_counts = Counter(event.get("event") for event in events)
        audit.check(set(event_counts) <= {"armijo_accepted", "initial_objective_accepted", "stationarity_diagnostic"}, "trace/known_event_types", "three declared event kinds", dict(event_counts))
        trace_rows.append({"path": name, "line_count": len(events), "actual_accepted_updates": event_counts["armijo_accepted"],
                           "initial_objective_accepted": event_counts["initial_objective_accepted"], "stationarity_diagnostics": event_counts["stationarity_diagnostic"]})
    expected_traces = {(case, method, init) for case in CASES for method in METHODS for init in (0, 1)}
    complete = identities == expected_traces and len(paths) == len(trace_rows) == 144
    if not complete:
        audit.pending("trace/coverage", "Incomplete 144 original case/method/start H trace identities; no extrapolated update count.")
    total = sum(row["actual_accepted_updates"] for row in trace_rows)
    audit.equal(saved.get("accepted_trace_rows"), trace_rows, "trace/independent_event_counts")
    audit.equal(saved.get("accepted_trace_count"), len(trace_rows), "trace/file_count")
    audit.equal(saved.get("expected_H_fit_traces"), 144, "trace/declared_expected_count")
    audit.equal(saved.get("actual_accepted_updates"), total, "trace/update_total")
    audit.equal(summary.get("actual_accepted_updates"), total, "summary/update_total")
    audit.equal(saved.get("trace_coverage_complete"), complete, "trace/coverage_flag")
    audit.equal(saved.get("wall_seconds_not_summed_as_parallel_elapsed_time"), True, "budget/no_parallel_wall_sum")
    return {"phase_count": len(receipt_rows), "phase_types": dict(phases), "work_counts": dict(counts),
            "accepted_updates": total, "trace_count": len(trace_rows), "trace_complete": complete}


def bindings_and_manifest(cfg, contract_path, analysis, audit):
    audit.equal(cfg.get("schema"), "local_shrinkage_budget_analysis_v1", "contract/schema")
    audit.equal(cfg.get("frozen"), True, "contract/frozen")
    policy = cfg.get("policy", {})
    required_policy = {"case_ids": list(CASES), "settings": list(SETTINGS), "methods": list(METHODS), "initializations": [0, 1],
                       "N": N, "risk_orders": ORDERS, "kl_tolerance": KL_TOL, "pmf_tolerance": PMF_TOL,
                       "primary_quantities": ["D_H", "G"], "family_alpha": .05, "family_size": 2, "tails": 2, "df": 23,
                       "family_size_if_one_quantity_unresolved": 2, "diagnostic_filtering": False}
    for key, value in required_policy.items():
        audit.equal(policy.get(key), value, "policy/" + key)
    paths = cfg.get("evaluation_case_files", {})
    audit.equal(set(paths), {str(case) for case in CASES}, "contract/explicit_24_cases")
    required = [name for name in paths.values() if name is not None]
    required += [item["path"] for item in cfg.get("budget_receipts", [])] + cfg.get("accepted_trace_files", [])
    required.append(cfg["policy_frozen_before_new_risk"])
    for name in required:
        audit.check(name in cfg.get("input_sha256", {}), "binding/present/" + name, "bound input", None)
    for field in ("input_sha256", "source_sha256"):
        for name, digest in cfg.get(field, {}).items():
            try:
                actual = sha(relative(name))
            except (OSError, ValueError) as error:
                actual = "unreadable: " + str(error)
            audit.equal(actual, digest, field + "/" + name)
    audit.equal(sha(analysis / "CONTRACT.json"), sha(contract_path), "analysis/copied_contract_bytes")
    audit.equal(sha(analysis / "ANALYSIS_SOURCE.py"), cfg.get("source_sha256", {}).get("script/budget_sensitivity_analyze.py"), "analysis/frozen_analyzer_bytes")
    receipt = read(relative(cfg["policy_frozen_before_new_risk"]))
    audit.equal(receipt.get("policy"), policy, "policy/receipt_identity")
    audit.equal(receipt.get("frozen_before_new_risk"), True, "policy/pre_risk_provenance_flag")
    manifest = read(analysis / "MANIFEST.json")
    files = manifest.get("files", [])
    declared = [row.get("path") for row in files]
    audit.equal(len(declared), len(set(declared)), "manifest/no_duplicate_entries")
    audit.equal(set(declared), {p.name for p in analysis.iterdir() if p.is_file() and p.name != "MANIFEST.json"}, "manifest/exact_file_coverage")
    audit.equal(manifest.get("manifest_self_hash_excluded"), True, "manifest/self_exclusion")
    for row in files:
        name = row["path"]
        if Path(name).name != name or name == "MANIFEST.json":
            audit.check(False, "manifest/safe_flat_path", "one flat saved-analysis filename", name)
            continue
        path = analysis / name
        audit.equal(path.stat().st_size, row.get("bytes"), "manifest/bytes/" + name)
        audit.equal(sha(path), row.get("sha256"), "manifest/sha256/" + name)


def verify(contract_path, analysis):
    audit = Audit()
    report = {"schema": SCHEMA, "verifier_source_sha256": sha(Path(__file__)),
              "contract": str(contract_path), "analysis": str(analysis), "scope": {
                  "independent_analyzer_imported": False, "new_observations": 0, "new_model_integrations": 0,
                  "new_fits": 0, "NPZ_loaded": False,
                  "reconstructed": "288 saved risk slots; dataset two-start means; D_H/D_P0/G; fixed family2 t23 intervals; 96 GROUP diagnostic identities; ten phase work receipts;144 H accepted-event traces",
                  "not_verified": "Continuous integrals, PMF arrays, R8 replicate SE reconstruction, spectral projection, true proposal moments, global optimality, or external preregistration timestamps.",
                  "tolerances": {"absolute": 1e-12, "relative": 1e-10}}}
    try:
        cfg = read(contract_path)
        bindings_and_manifest(cfg, contract_path, analysis, audit)
        if audit.errors:
            report["dependent_arithmetic_skipped"] = "Binding/manifest/policy failure; no complete verification asserted."
        else:
            records = {case: read(relative(cfg["evaluation_case_files"][str(case)])) if cfg["evaluation_case_files"][str(case)] is not None else None for case in CASES}
            computed = reconstruct_risks(records, audit)
            risk, diagnostic = read(analysis / "RISK_ANALYSIS.json"), read(analysis / "DIAGNOSTICS.json")
            work, summary = read(analysis / "WORK_COUNTS.json"), read(analysis / "SUMMARY.json")
            with (analysis / "PAIRED_CASES.csv").open(encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            verify_risk_outputs(computed, risk, summary, csv_rows, audit)
            report["risk_scope"] = {"expected_slots": 288, "qualified_slots": sum(row["eligible"] for row in computed["slots"].values()),
                "all_288_qualified": computed["complete"], "primary": {key: {"required_slots": value["required"],
                "qualified_slots": value["qualified_slots"], "complete_qualified": value["qualified"]} for key, value in computed["primary"].items()}}
            report["diagnostic_scope"] = verify_diagnostics(records, diagnostic, summary, audit)
            report["cost_scope"] = verify_costs(cfg, work, summary, audit)
    except Exception as error:
        audit.errors.append({"where": "verification_exception", "type": type(error).__name__, "message": str(error)})
    status = "failed" if audit.errors else "consistent_but_unresolved" if audit.unresolved else "passed_complete"
    report.update(status=status, complete_verification_passed=status == "passed_complete", comparisons=audit.comparisons,
                  mismatches=audit.errors, unresolved=audit.unresolved)
    return report


class ManufacturedVerifierTests(unittest.TestCase):
    @staticmethod
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_failure_report_keeps_sets_and_nonfinite_evidence_serializable(self):
        evidence = {"expected": {(2000, "H", "group8", 0)}, "actual": float("nan")}
        value = report_safe(evidence)
        json.dumps(value, allow_nan=False)
        self.assertEqual(value["expected"], [[2000, "H", "group8", 0]])
        self.assertEqual(value["actual"], {"nonfinite_saved_value": "nan"})

    def test_dataset_pairing_and_independent_qualification(self):
        records = {}
        for j, case in enumerate(CASES):
            rows = []
            for setting in SETTINGS:
                for method in METHODS:
                    base = (1+.01*j) if method == "integrated" else (2+.02*j) if method == "group8" else 1.5
                    if setting == "H":
                        base += .3 if method == "group8" else .1
                    for init in (0, 1):
                        value = (base + (-.05 if init == 0 else .05)) / N
                        rows.append({"case_id": case, "N": N, "setting": setting, "method": method, "initialization_index": init,
                            "joint_kl": value, "evaluable": True, "precision_qualified": True,
                            "risk_levels": [{"orders": ORDERS[0], "joint_kl": value+1e-12}, {"orders": ORDERS[1], "joint_kl": value}],
                            "precision_diagnostics": {"KL_order_difference": 1e-12, "PMF_order_max_difference": 1e-12}})
            records[case] = {"case_id": case, "risks": rows}
        audit = Audit()
        result = reconstruct_risks(records, audit)
        self.assertFalse(audit.errors)
        self.assertEqual(len(result["slots"]), 288)
        self.assertAlmostEqual(result["primary"]["D_H"]["description"]["mean"], 1.315)
        self.assertAlmostEqual(result["primary"]["G"]["description"]["mean"], .2)
        self.assertEqual(result["primary"]["D_H"]["interval"]["family_size"], 2)
        records[2003]["risks"] = [r for r in records[2003]["risks"] if not (r["setting"] == "P0" and r["method"] == "group8" and r["initialization_index"] == 1)]
        result = reconstruct_risks(records, Audit())
        self.assertTrue(result["primary"]["D_H"]["qualified"])
        self.assertFalse(result["primary"]["G"]["qualified"])
        self.assertIsNone(result["primary"]["G"]["description"])

    def test_t23_family2_arithmetic(self):
        values = [float(j) for j in range(24)]
        result = describe(values, True)
        self.assertEqual(result["df"], 23)
        self.assertAlmostEqual(result["SE"], math.sqrt(50 / 24))
        self.assertAlmostEqual(result["lower"], 11.5-float(t.ppf(.9875, 23))*math.sqrt(50 / 24))
        self.assertIsNone(describe(values[:-1], True))

    def test_saved_diagnostic_arithmetic_and_tampering(self):
        gradient = [.0001, -.0002, .0003]
        se = [.00001, .00002, .00003]
        base = {"gradient": gradient, "gradient_scramble_SE": se, "maximum_gradient_SE": max(se), "value": 2.,
                "projected_mapping": {"mapping": gradient, "norm_inf": .0003}}
        row = {"base": base, "prefix": {"gradient": [0., 0., 0.], "value": 1.5, "B": 1024},
               "full_minus_prefix": {"gradient": gradient.copy(), "value": .5}, "moment_levels": []}
        for order, shift in ((192, 0.), (256, 1e-10), (384, 2e-10)):
            row["moment_levels"].append({"outer_order": order, "same_inner_order": 256, "gradient_shift": [shift]*3,
                "value_shift": shift, "gradient": [g+shift for g in gradient], "value": 2.+shift, "gradient_scramble_SE": se})
        audit = Audit()
        verify_diagnostic_identities(row, audit, "fixture")
        self.assertFalse(audit.errors)
        self.assertFalse(audit.unresolved)
        row["full_minus_prefix"]["gradient"][0] += 1e-4
        audit = Audit()
        verify_diagnostic_identities(row, audit, "fixture")
        self.assertTrue(audit.errors)

    def test_budget_events_partial_coverage_and_saved_counter_independence(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(globals(), {"REPO": Path(temp)}):
            receipts, phase_rows = [], []
            for phase, work in (("profile", 10), ("fixed_diagnostics", 20)):
                name = phase + "/SUMMARY.json"
                record = {"phase": phase, "status": "completed", "budget": {"counts": {"work": work, "optimizer_updates": 0}}}
                self.dump(REPO / name, record)
                receipts.append({"path": name})
                phase_rows.append({"path": name, **record})
            traces, trace_rows = [], []
            for init in (0, 1):
                name = f"fit/case_02000/FIT_integrated_init{init}/ACCEPTED_TRACE.jsonl"
                path = REPO / name
                path.parent.mkdir(parents=True)
                kinds = ["initial_objective_accepted", "stationarity_diagnostic", "armijo_accepted", "stationarity_diagnostic"]
                path.write_text("\n".join(json.dumps({"event": kind}) for kind in kinds), encoding="utf-8")
                traces.append(name)
                trace_rows.append({"path": name, "line_count": 4, "actual_accepted_updates": 1,
                                   "initial_objective_accepted": 1, "stationarity_diagnostics": 2})
            saved = {"phase_receipts": phase_rows, "summed_actual_saved_work_counts": {"work": 30, "optimizer_updates": 0},
                     "accepted_trace_rows": trace_rows, "actual_accepted_updates": 2, "accepted_trace_count": 2,
                     "expected_H_fit_traces": 144, "trace_coverage_complete": False,
                     "wall_seconds_not_summed_as_parallel_elapsed_time": True}
            audit = Audit()
            report = verify_costs({"budget_receipts": receipts, "accepted_trace_files": traces}, saved,
                                  {"actual_accepted_updates": 2}, audit)
            self.assertFalse(audit.errors)
            self.assertEqual(report["accepted_updates"], 2)
            self.assertEqual(report["work_counts"]["work"], 30)
            self.assertEqual(len(audit.unresolved), 2)

    def test_input_hash_and_analysis_manifest_are_checked(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(globals(), {"REPO": Path(temp)}):
            source = REPO / "script/budget_sensitivity_analyze.py"
            source.parent.mkdir()
            source.write_text("# manufactured non-executable source identity\n", encoding="utf-8")
            policy = {"case_ids": list(CASES), "settings": list(SETTINGS), "methods": list(METHODS), "initializations": [0, 1],
                      "N": N, "risk_orders": ORDERS, "kl_tolerance": KL_TOL, "pmf_tolerance": PMF_TOL,
                      "primary_quantities": ["D_H", "G"], "family_alpha": .05, "family_size": 2, "tails": 2,
                      "df": 23, "family_size_if_one_quantity_unresolved": 2, "diagnostic_filtering": False}
            self.dump(REPO / "policy.json", {"policy": policy, "frozen_before_new_risk": True})
            cfg = {"schema": "local_shrinkage_budget_analysis_v1", "frozen": True, "policy": policy,
                   "evaluation_case_files": {str(case): None for case in CASES}, "budget_receipts": [], "accepted_trace_files": [],
                   "policy_frozen_before_new_risk": "policy.json", "input_sha256": {"policy.json": sha(REPO / "policy.json")},
                   "source_sha256": {"script/budget_sensitivity_analyze.py": sha(source)}}
            contract = REPO / "contract.json"
            self.dump(contract, cfg)
            analysis = REPO / "analysis"
            analysis.mkdir()
            (analysis / "CONTRACT.json").write_bytes(contract.read_bytes())
            (analysis / "ANALYSIS_SOURCE.py").write_bytes(source.read_bytes())
            self.dump(analysis / "MANIFEST.json", {"files": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(analysis.iterdir())], "manifest_self_hash_excluded": True})
            audit = Audit()
            bindings_and_manifest(cfg, contract, analysis, audit)
            self.assertFalse(audit.errors)
            with (REPO / "policy.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            audit = Audit()
            bindings_and_manifest(cfg, contract, analysis, audit)
            self.assertTrue(any(item["where"] == "input_sha256/policy.json" for item in audit.errors))
            (analysis / "ANALYSIS_SOURCE.py").write_text("changed", encoding="utf-8")
            audit = Audit()
            bindings_and_manifest(cfg, contract, analysis, audit)
            self.assertTrue(any(item["where"] == "manifest/sha256/ANALYSIS_SOURCE.py" for item in audit.errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedVerifierTests))
        return 0 if result.wasSuccessful() else 1
    if not args.contract or not args.analysis or not args.out:
        parser.error("--contract, --analysis and fresh --out JSON are required")
    out, analysis = args.out.resolve(), args.analysis.resolve()
    if out.exists() or out.is_relative_to(analysis):
        parser.error("--out must be a new JSON outside the sealed analysis directory")
    report = verify(args.contract.resolve(), analysis)
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report_safe(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"], "mismatches": len(report["mismatches"]),
                      "unresolved": len(report["unresolved"]), "output": str(out)}))
    return 1 if report["mismatches"] else 2 if report["unresolved"] else 0


if __name__ == "__main__":
    sys.exit(main())
