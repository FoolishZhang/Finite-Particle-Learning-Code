"""Read-only analysis of frozen budget-sensitivity results; no science runner.

Only explicit contract-bound JSON/JSONL inputs are read. There is no discovery
of running directories, resampling, integration, fitting, diagnostic filtering,
or mutation of source results. --self-test uses entirely manufactured records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import unittest

import numpy as np
import scipy
from scipy.stats import t

SCHEMA = "local_shrinkage_budget_analysis_v1"
REPO = Path(__file__).resolve().parents[1]
CASES = list(range(2000, 2024))
SETTINGS = ("P0", "H")
METHODS = ("integrated", "group8", "ridge")
POLICY = {
    "case_ids": CASES, "settings": list(SETTINGS), "methods": list(METHODS),
    "initializations": [0, 1], "N": 2048,
    "unit": "original_dataset_case; two_start_risks_averaged_before_comparison",
    "risk_orders": [[28, 192], [36, 256]], "kl_tolerance": 1e-4 / 2048,
    "pmf_tolerance": 1e-8, "mapping_tolerance": 3e-4, "maximum_gradient_SE_tolerance": 1e-4,
    "primary_quantities": ["D_H", "G"], "secondary_descriptive": ["D_P0", "P0_H_ridge_risks"],
    "D_H": "2048*(mean_two_H_GROUP_KL-mean_two_H_integrated_ML_KL)",
    "D_P0": "2048*(mean_two_P0_GROUP_KL-mean_two_P0_integrated_ML_KL)",
    "G": "D_H-D_P0 within the identical dataset case",
    "family_alpha": .05, "family_size": 2, "tails": 2, "df": 23,
    "critical_value": "scipy.stats.t.ppf(1-.05/(2*2),23)",
    "interval": "mean +/- critical_value * sample_SD(ddof=1)/sqrt(24)",
    "missing_or_unqualified_rule": "retain_all_expected_rows; D_H_requires_its_96_H_GROUP_ML_slots; G_requires_192_P0_H_GROUP_ML_slots; ridge_96_slots_separately_descriptive",
    "family_size_if_one_quantity_unresolved": 2,
    "diagnostic_filtering": False, "zero_in_interval_means_equivalence": False,
    "inferential_scope": "retrospective_exploratory_paired_numerical_sensitivity; not_original_family4_confirmation",
    "risk_precision_scope": "ordinary_GH_order_difference_not_integral_error_certificate",
    "accepted_update_count": "count armijo_accepted events in each explicitly bound ACCEPTED_TRACE.jsonl",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(test, message):
    if not test:
        raise ValueError(message)


def resolve(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "repository-relative input required")
    return REPO / p


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def template():
    return {"schema": SCHEMA, "frozen": False, "policy": POLICY,
        "policy_frozen_before_new_risk": "REPLACE_WITH_PRE_RISK_POLICY_RECEIPT_PATH",
        "source_sha256": {"script/budget_sensitivity_analyze.py": sha(Path(__file__))},
        "evaluation_case_files": {str(case): None for case in CASES},
        "budget_receipts": [], "accepted_trace_files": [], "input_sha256": {},
        "instructions": "Bind saved completed/failed inputs after execution without changing this pre-risk policy; null case paths retain missing-case rows. Budget receipts must be one SUMMARY.json or FAILED.json per distinct phase run, not cumulative per-case snapshots."}


def validate(cfg):
    require(cfg["schema"] == SCHEMA and cfg["frozen"] is True, "frozen analysis contract required")
    require(cfg["policy"] == POLICY, "analysis policy differs from predeclared implementation")
    require(set(cfg["evaluation_case_files"]) == {str(case) for case in CASES}, "all 24 cases must have explicit path or null")
    require(cfg["source_sha256"].get("script/budget_sensitivity_analyze.py") == sha(Path(__file__)), "analysis source differs")
    for name, expected in cfg["source_sha256"].items():
        require(sha(resolve(name)) == expected, "source hash differs: " + name)
    required = [name for name in cfg["evaluation_case_files"].values() if name is not None]
    required += [r["path"] for r in cfg["budget_receipts"]] + cfg["accepted_trace_files"]
    required.append(cfg["policy_frozen_before_new_risk"])
    trace_identity_check = trace_coverage(cfg["accepted_trace_files"])
    require(trace_identity_check["identity_consistent"], "invalid, duplicate, or aliased accepted-trace identity")
    budget_paths = [Path(r["path"]) for r in cfg["budget_receipts"]]
    require(all(p.name in ("SUMMARY.json", "FAILED.json") for p in budget_paths), "only phase-level budget receipts may be summed")
    require(len({resolve(str(p)).resolve().parent for p in budget_paths}) == len(budget_paths), "duplicate/cumulative phase budget receipt")
    for name in required:
        require(name in cfg["input_sha256"], "input is not hash-bound: " + name)
    for name, expected in cfg["input_sha256"].items():
        require(sha(resolve(name)) == expected, "input hash differs: " + name)
    policy_receipt = read(resolve(cfg["policy_frozen_before_new_risk"]))
    require(policy_receipt.get("policy") == POLICY and policy_receipt.get("frozen_before_new_risk") is True,
            "matching pre-risk policy receipt required; temporal claim remains provenance, not self-certified by this check")
    return cfg


def summarize(values, *, interval=False):
    x = np.asarray(values, float)
    require(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(), "complete finite vector required")
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(len(x))
    result = {"n": len(x), "mean": float(x.mean()), "sample_SD": sd, "SE": float(se), "df": len(x)-1}
    if interval:
        require(len(x) == 24, "primary intervals require all 24 original cases")
        critical = float(t.ppf(1 - .05 / (2 * 2), 23))
        result.update(critical_value=critical, family_alpha=.05, family_size=2,
                      lower=result["mean"]-critical*se, upper=result["mean"]+critical*se,
                      method="approximate_Bonferroni_simultaneous_two_sided_Student_t",
                      equivalence_conclusion=False)
    return result


def risk_issue(row, case, setting, method, init):
    issues = []
    if row is None:
        return ["missing_expected_risk_row"]
    if (row.get("case_id"), row.get("setting"), row.get("method"), row.get("initialization_index")) != (case, setting, method, init):
        issues.append("row_identity_mismatch")
    if row.get("N") != 2048:
        issues.append("wrong_N")
    if row.get("evaluable") is not True or not finite(row.get("joint_kl")):
        issues.append("risk_not_finitely_evaluable")
    if row.get("precision_qualified") is not True:
        issues.append("risk_not_qualified")
    levels = row.get("risk_levels", [])
    if [r.get("orders") for r in levels] != POLICY["risk_orders"]:
        issues.append("common_risk_orders_missing_or_different")
    diagnostics = row.get("precision_diagnostics", {})
    for key, limit in (("KL_order_difference", POLICY["kl_tolerance"]), ("PMF_order_max_difference", POLICY["pmf_tolerance"])):
        value = diagnostics.get(key)
        if not finite(value) or value < 0 or value > limit:
            issues.append("failed_or_missing_" + key)
    return issues


def analyze_risks(case_records):
    raw, paired, all_issues = [], [], []
    for case in CASES:
        record = case_records.get(case)
        risks = [] if record is None else record.get("risks", [])
        indexed = {}
        for row in risks:
            key = (row.get("setting"), row.get("method"), row.get("initialization_index"))
            indexed.setdefault(key, []).append(row)
        averages = {}
        case_issues = []
        expected_keys = {(s, m, i) for s in SETTINGS for m in METHODS for i in (0, 1)}
        extras = [row for key, rows in indexed.items() if key not in expected_keys for row in rows]
        if extras:
            case_issues.append("unexpected_risk_rows")
            raw.extend({"case_id": case, "unexpected": True, "raw": row} for row in extras)
        for setting in SETTINGS:
            for method in METHODS:
                values = []
                for init in (0, 1):
                    rows = indexed.get((setting, method, init), [])
                    row = rows[0] if len(rows) == 1 else None
                    issues = risk_issue(row, case, setting, method, init)
                    if len(rows) > 1:
                        issues.append("duplicate_risk_identity")
                    case_issues += issues
                    raw.append({"case_id": case, "setting": setting, "method": method, "initialization_index": init,
                        "valid_for_complete_analysis": not issues, "issues": issues, "raw": row,
                        "duplicate_rows_retained": rows if len(rows) > 1 else []})
                    # A failed ordinary precision gate does not erase a saved
                    # numerical risk. It remains descriptive and blocks the CI.
                    values.append(row.get("joint_kl") if row is not None and finite(row.get("joint_kl")) else None)
                averages[f"{setting}_{method}_scaled_risk"] = None if any(v is None for v in values) else float(2048 * sum(values) / 2)
        def difference(a, b):
            return None if averages[a] is None or averages[b] is None else averages[a]-averages[b]
        dp0 = difference("P0_group8_scaled_risk", "P0_integrated_scaled_risk")
        dh = difference("H_group8_scaled_risk", "H_integrated_scaled_risk")
        change = None if dp0 is None or dh is None else dh-dp0
        paired.append({"case_id": case, **averages, "D_P0": dp0, "D_H": dh, "G": change,
                       "all_expected_risks_qualified": not case_issues, "issues": sorted(set(case_issues))})
        all_issues += [{"case_id": case, "issue": issue} for issue in sorted(set(case_issues))]
    complete = not all_issues
    primary = {}
    for quantity in ("D_H", "G"):
        needed_settings = ("H",) if quantity == "D_H" else SETTINGS
        required_rows = [r for r in raw if not r.get("unexpected") and r["setting"] in needed_settings and
                         r["method"] in ("integrated", "group8")]
        required_count = 96 if quantity == "D_H" else 192
        qualified = len(required_rows) == required_count and all(r["valid_for_complete_analysis"] for r in required_rows)
        values = [row[quantity] for row in paired]
        all_finite = all(v is not None for v in values)
        primary[quantity] = {"status": "complete_qualified_exploratory_interval" if qualified else "unresolved_no_primary_interval",
            "required_risk_slots": required_count, "qualified_required_risk_slots": sum(r["valid_for_complete_analysis"] for r in required_rows),
            "case_values": [{"case_id": row["case_id"], "value": row[quantity]} for row in paired],
            "finite_case_count": sum(v is not None for v in values),
            "complete_vector_description": summarize(values) if all_finite else None,
            "simultaneous_interval": summarize(values, interval=True) if qualified else None,
            "diagnostic_failures_not_filtered": True, "no_equivalence_test": True}
    description = {}
    for key in ["D_P0"] + [f"{s}_{m}_scaled_risk" for s in SETTINGS for m in METHODS]:
        values = [row[key] for row in paired]
        description[key] = {"complete_vector_description": summarize(values) if all(v is not None for v in values) else None,
            "finite_case_count": sum(v is not None for v in values), "inferential_interval": None,
            "qualification": "descriptive_only; no_subset_estimate_if_rows_missing"}
        if key.endswith("_scaled_risk"):
            setting, method, _, _ = key.split("_")
            relevant = [r for r in raw if not r.get("unexpected") and r["setting"] == setting and r["method"] == method]
            description[key]["qualified_risk_slots"] = sum(r["valid_for_complete_analysis"] for r in relevant)
            description[key]["expected_risk_slots"] = 48
    return {"all_24_cases_complete_and_all_288_risks_qualified": complete, "raw_risk_rows": raw,
            "paired_case_rows": paired, "primary": primary, "descriptive": description,
            "issues": all_issues, "scope": POLICY["inferential_scope"]}


def analyze_diagnostics(case_records):
    rows, seen, issues = [], set(), []
    for case in CASES:
        record = case_records.get(case) or {}
        for row in record.get("group_diagnostics", []):
            point = row.get("point", {})
            identity = (case, point.get("setting"), point.get("initialization_index"))
            if identity in seen:
                issues.append({"identity": identity, "issue": "duplicate_group_diagnostic"})
            valid = identity[1] in SETTINGS and identity[2] in (0, 1) and point.get("case_id") == case and point.get("method") == "group8"
            if valid:
                seen.add(identity)
            base = row.get("base", {})
            mapping = base.get("projected_mapping", {}).get("norm_inf")
            se = base.get("maximum_gradient_SE")
            finite_base = finite(mapping) and finite(se)
            own = point.get("fit_status") == "mapping_tolerance"
            mp = finite_base and mapping <= POLICY["mapping_tolerance"]
            sp = finite_base and se <= POLICY["maximum_gradient_SE_tolerance"]
            category = ("neither_numeric_gate_fails" if mp and sp else "mapping_only" if not mp and sp else
                        "SE_only" if mp and not sp else "both") if finite_base else "missing_numeric_diagnostic"
            consistency = finite_base and bool(base.get("mapping_pass")) == bool(mp) and bool(base.get("gradient_SE_pass")) == bool(sp)
            item = {"case_id": case, "setting": identity[1], "initialization_index": identity[2],
                "identity_valid": valid, "fit_status": point.get("fit_status"), "own_status_pass": own,
                "mapping_norm_inf": mapping, "maximum_gradient_SE": se, "mapping_pass": bool(mp), "SE_pass": bool(sp),
                "numeric_failure_category": category, "composite_pass": bool(valid and own and mp and sp),
                "saved_booleans_consistent": bool(consistency), "prefix": row.get("prefix"),
                "full_minus_prefix": row.get("full_minus_prefix"), "moment_levels": row.get("moment_levels", []),
                "raw": row}
            rows.append(item)
            if not valid or not consistency:
                issues.append({"identity": identity, "issue": "identity_or_saved_gate_inconsistency"})
    expected = {(case, s, i) for case in CASES for s in SETTINGS for i in (0, 1)}
    for identity in sorted(expected-seen):
        issues.append({"identity": identity, "issue": "missing_group_diagnostic"})
    result = {}
    for setting in SETTINGS:
        selected = [r for r in rows if r["setting"] == setting and r["identity_valid"]]
        relevant_issues = [r for r in issues if r["identity"][1] == setting or r["identity"][1] not in SETTINGS]
        unique_identities = {(r["case_id"], setting, r["initialization_index"]) for r in selected}
        identity_complete = len(unique_identities) == 48 and len(selected) == 48
        complete_consistent = identity_complete and not relevant_issues
        numeric = Counter(r["numeric_failure_category"] for r in selected)
        prefix_grad = [float(np.max(np.abs(r["full_minus_prefix"]["gradient"]))) for r in selected
                       if r["full_minus_prefix"] is not None]
        moment = {}
        for order in (192, 256, 384):
            changes = [m for r in selected for m in r["moment_levels"] if m.get("outer_order") == order]
            moment[str(order)] = {"row_count": len(changes),
                "maximum_absolute_gradient_shift": max((float(np.max(np.abs(m["gradient_shift"]))) for m in changes), default=None),
                "maximum_absolute_value_shift": max((abs(float(m["value_shift"])) for m in changes), default=None)}
        counts = {"composite_pass": sum(r["composite_pass"] for r in selected),
            "own_status_pass": sum(r["own_status_pass"] for r in selected),
            "own_status_counts": dict(Counter(str(r["fit_status"]) for r in selected)),
            "numeric_failure_counts": {name: numeric[name] for name in ("neither_numeric_gate_fails", "mapping_only", "SE_only", "both", "missing_numeric_diagnostic")}}
        result[setting] = {"expected_outputs": 48, "observed_rows": len(selected),
            "unique_expected_outputs": 48, "unique_observed_outputs": len(unique_identities),
            "identity_complete": identity_complete, "complete_and_consistent": complete_consistent,
            "complete_count_summary_usable": complete_consistent, "issues": relevant_issues,
            **{name: value if complete_consistent else None for name, value in counts.items()},
            "observed_counts_uncertified_if_incomplete": None if complete_consistent else counts,
            "nested_full_minus_prefix_gradient_max": max(prefix_grad, default=None),
            "nested_prefix_rows": len(prefix_grad), "moment_changes": moment,
            "diagnostic_is_not_global_optimum_certificate": True}
    return {"by_setting": result, "rows": rows, "issues": issues,
            "scope": "all_outputs_descriptive; numerical_replicates_not_new_datasets; no_filtering"}


def count_trace_events(events):
    return {"line_count": len(events),
        "actual_accepted_updates": sum(e.get("event") == "armijo_accepted" for e in events),
        "initial_objective_accepted": sum(e.get("event") == "initial_objective_accepted" for e in events),
        "stationarity_diagnostics": sum(e.get("event") == "stationarity_diagnostic" for e in events)}


def trace_coverage(names):
    """Validate the 24*3*2 path identities, independent of trace line counts."""
    identities, paths, issues, rows = set(), set(), [], []
    for name in names:
        path = resolve(name).resolve()
        identity = None
        method_match = re.fullmatch(r"FIT_(integrated|group8|ridge)_init([01])", path.parent.name)
        case_match = re.fullmatch(r"case_(\d{5})", path.parent.parent.name)
        if path.name == "ACCEPTED_TRACE.jsonl" and method_match and case_match:
            identity = (int(case_match[1]), method_match[1], int(method_match[2]))
        if identity is None or identity[0] not in CASES:
            issues.append({"path": name, "issue": "invalid_trace_case_method_initialization_path"})
        else:
            if identity in identities:
                issues.append({"path": name, "identity": identity, "issue": "duplicate_trace_identity"})
            identities.add(identity)
        if path in paths:
            issues.append({"path": name, "issue": "duplicate_or_aliased_physical_trace_path"})
        paths.add(path)
        rows.append({"path": name, "identity": identity})
    expected = {(case, method, init) for case in CASES for method in METHODS for init in (0, 1)}
    return {"expected_unique_identities": 144, "observed_unique_identities": len(identities),
        "identity_consistent": not issues, "coverage_complete": not issues and identities == expected,
        "missing_identities": sorted(expected-identities), "issues": issues, "rows": rows}


def analyze_budget(cfg):
    receipts, counts = [], Counter()
    for binding in cfg["budget_receipts"]:
        record = read(resolve(binding["path"]))
        budget = record["budget"]
        for name, value in budget["counts"].items():
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "noninteger or negative saved work count")
            counts[name] += value
        receipts.append({**binding, "phase": record.get("phase"), "status": record.get("status", "failed"), "budget": budget})
    trace_rows = []
    coverage = trace_coverage(cfg["accepted_trace_files"])
    for name in cfg["accepted_trace_files"]:
        events = []
        with resolve(name).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    events.append(json.loads(line))
        trace_rows.append({"path": name, **count_trace_events(events)})
    return {"phase_receipts": receipts, "summed_actual_saved_work_counts": dict(counts),
        "accepted_trace_rows": trace_rows, "actual_accepted_updates": sum(r["actual_accepted_updates"] for r in trace_rows),
        "accepted_trace_count": len(trace_rows), "expected_H_fit_traces": 144,
        "trace_coverage_complete": coverage["coverage_complete"], "trace_identity_coverage": coverage,
        "inherited_optimizer_updates_counter_is_not_actual_accepted_updates": True,
        "wall_seconds_not_summed_as_parallel_elapsed_time": True,
        "work_counts_are_not_independent_statistical_samples": True}


def run(contract_path, out):
    cfg = validate(read(contract_path))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(contract_path, out / "CONTRACT.json")
    shutil.copyfile(Path(__file__), out / "ANALYSIS_SOURCE.py")
    case_records = {case: read(resolve(cfg["evaluation_case_files"][str(case)]))
                    if cfg["evaluation_case_files"][str(case)] is not None else None for case in CASES}
    risks = analyze_risks(case_records)
    diagnostics = analyze_diagnostics(case_records)
    budget = analyze_budget(cfg)
    write(out / "RISK_ANALYSIS.json", risks)
    write(out / "DIAGNOSTICS.json", diagnostics)
    write(out / "WORK_COUNTS.json", budget)
    with (out / "PAIRED_CASES.csv").open("x", encoding="utf-8", newline="") as stream:
        keys = [k for k in risks["paired_case_rows"][0] if k != "issues"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in risks["paired_case_rows"]:
            writer.writerow({key: row[key] for key in keys})
    summary = {"schema": SCHEMA, "status": "complete_saved_analysis", "policy": POLICY,
        "analysis_runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "risk_complete": all(p["simultaneous_interval"] is not None for p in risks["primary"].values()),
        "all_288_risks_qualified": risks["all_24_cases_complete_and_all_288_risks_qualified"],
        "primary": risks["primary"], "diagnostics": diagnostics["by_setting"],
        "diagnostic_issues": diagnostics["issues"],
        "actual_accepted_updates": budget["actual_accepted_updates"],
        "new_observations": 0, "new_model_integrations": 0, "new_fits": 0,
        "saved_risk_statistics_recomputed": True, "no_equivalence_claim": True}
    write(out / "SUMMARY.json", summary)
    write(out / "MANIFEST.json", {"files": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(out.iterdir()) if p.is_file()], "manifest_self_hash_excluded": True})
    return summary


def manufactured_records():
    """Construct deterministic arithmetic fixtures, not simulated observations."""
    records = {}
    for j, case in enumerate(CASES):
        risks = []
        scaled = {("P0", "integrated"): 1+.01*j, ("P0", "group8"): 2+.015*j, ("P0", "ridge"): 1.8+.01*j,
                  ("H", "integrated"): 1.1+.01*j, ("H", "group8"): 2.3+.02*j, ("H", "ridge"): 1.9+.01*j}
        for (setting, method), value in scaled.items():
            for init in (0, 1):
                risks.append({"case_id": case, "N": 2048, "setting": setting, "method": method,
                    "initialization_index": init, "joint_kl": (value+(-.1 if init == 0 else .1))/2048,
                    "evaluable": True, "precision_qualified": True, "fit_status": "mapping_tolerance",
                    "risk_levels": [{"orders": order} for order in POLICY["risk_orders"]],
                    "precision_diagnostics": {"KL_order_difference": 1e-12, "PMF_order_max_difference": 1e-12}})
        records[case] = {"case_id": case, "risks": risks}
    return records


class ManufacturedAnalysisTests(unittest.TestCase):
    def test_accepted_update_counter_uses_events_not_inherited_zero(self):
        events = [{"event": name, "budget": {"counts": {"optimizer_updates": 0}}} for name in
            ("initial_objective_accepted", "stationarity_diagnostic", "armijo_accepted", "stationarity_diagnostic",
             "armijo_accepted", "stationarity_diagnostic")]
        counts = count_trace_events(events)
        self.assertEqual(counts["actual_accepted_updates"], 2)
        self.assertEqual(counts["initial_objective_accepted"], 1)
        self.assertEqual(counts["stationarity_diagnostics"], 3)

    def test_two_start_dataset_pairing_and_two_quantity_intervals(self):
        result = analyze_risks(manufactured_records())
        self.assertTrue(result["all_24_cases_complete_and_all_288_risks_qualified"])
        self.assertEqual(len(result["raw_risk_rows"]), 288)
        expected = 1.2+.01*np.arange(24)
        expected_change = .2+.005*np.arange(24)
        for name, values in (("D_H", expected), ("G", expected_change)):
            interval = result["primary"][name]["simultaneous_interval"]
            self.assertEqual(interval["df"], 23)
            self.assertAlmostEqual(interval["mean"], float(values.mean()), places=13)
            se = np.sqrt(np.sum((values-values.mean())**2)/23)/np.sqrt(24)
            self.assertAlmostEqual(interval["SE"], float(se), places=13)
            self.assertAlmostEqual(interval["lower"], float(values.mean()-t.ppf(.9875,23)*se), places=13)
        self.assertIsNone(result["descriptive"]["D_P0"]["inferential_interval"])

    def test_unqualified_ridge_retained_without_blocking_primary(self):
        records = manufactured_records()
        records[2007]["risks"][-1]["precision_qualified"] = False
        result = analyze_risks(records)
        self.assertFalse(result["all_24_cases_complete_and_all_288_risks_qualified"])
        self.assertEqual(len(result["raw_risk_rows"]), 288)
        self.assertEqual(len(result["paired_case_rows"]), 24)
        for item in result["primary"].values():
            self.assertIsNotNone(item["simultaneous_interval"])
            self.assertEqual(item["complete_vector_description"]["n"], 24)

    def test_missing_P0_blocks_G_not_DH_and_keeps_two_quantity_family(self):
        records = manufactured_records()
        records[2004]["risks"] = [r for r in records[2004]["risks"] if not
            (r["setting"] == "P0" and r["method"] == "group8" and r["initialization_index"] == 0)]
        result = analyze_risks(records)
        self.assertIsNone(result["primary"]["G"]["simultaneous_interval"])
        interval = result["primary"]["D_H"]["simultaneous_interval"]
        self.assertEqual(interval["family_size"], 2)
        self.assertAlmostEqual(interval["critical_value"], t.ppf(.9875, 23))

    def test_missing_case_has_full_slots_no_subset_mean_or_interval(self):
        records = manufactured_records()
        del records[2001]
        result = analyze_risks(records)
        self.assertEqual(len(result["raw_risk_rows"]), 288)
        self.assertEqual(result["primary"]["D_H"]["finite_case_count"], 23)
        self.assertIsNone(result["primary"]["D_H"]["complete_vector_description"])
        self.assertIsNone(result["primary"]["D_H"]["simultaneous_interval"])

    def test_diagnostic_failures_are_not_risk_filter(self):
        records = manufactured_records()
        rows = []
        for init, (mapping, se) in enumerate(((4e-4, 2e-4), (1e-8, 2e-4))):
            rows.append({"point": {"case_id": 2000, "setting": "H", "method": "group8", "initialization_index": init,
                         "fit_status": "mapping_tolerance"}, "evaluable": True,
                "base": {"projected_mapping": {"norm_inf": mapping}, "maximum_gradient_SE": se,
                         "mapping_pass": mapping <= 3e-4, "gradient_SE_pass": se <= 1e-4}, "moment_levels": []})
        records[2000]["group_diagnostics"] = rows
        diag = analyze_diagnostics(records)
        counts = diag["by_setting"]["H"]["observed_counts_uncertified_if_incomplete"]["numeric_failure_counts"]
        self.assertEqual(counts["both"], 1)
        self.assertEqual(counts["SE_only"], 1)
        self.assertTrue(analyze_risks(records)["all_24_cases_complete_and_all_288_risks_qualified"])

    def test_duplicate_diagnostic_does_not_present_certified_pass_total(self):
        records = manufactured_records()
        for case in CASES:
            records[case]["group_diagnostics"] = [{"point": {"case_id": case, "setting": setting, "method": "group8",
                "initialization_index": init, "fit_status": "mapping_tolerance"},
                "base": {"projected_mapping": {"norm_inf": 1e-8}, "maximum_gradient_SE": 1e-5,
                         "mapping_pass": True, "gradient_SE_pass": True}, "moment_levels": []}
                for setting in SETTINGS for init in (0, 1)]
        records[2000]["group_diagnostics"].append(records[2000]["group_diagnostics"][-1])
        result = analyze_diagnostics(records)
        self.assertEqual(len(result["rows"]), 97)
        self.assertIsNone(result["by_setting"]["H"]["composite_pass"])
        self.assertFalse(result["by_setting"]["H"]["complete_count_summary_usable"])
        self.assertEqual(result["by_setting"]["H"]["unique_observed_outputs"], 48)
        self.assertEqual(result["by_setting"]["P0"]["composite_pass"], 48)

    def test_trace_coverage_requires_unique_case_method_start_not_path_count(self):
        names = [f"artifacts/manufactured_only/case_{case:05d}/FIT_{method}_init{init}/ACCEPTED_TRACE.jsonl"
                 for case in CASES for method in METHODS for init in (0, 1)]
        self.assertTrue(trace_coverage(names)["coverage_complete"])
        names[-1] = names[0].replace("manufactured_only/", "manufactured_only/./")
        result = trace_coverage(names)
        self.assertEqual(len(names), 144)
        self.assertFalse(result["coverage_complete"])
        self.assertFalse(result["identity_consistent"])
        self.assertEqual(result["observed_unique_identities"], 143)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedAnalysisTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif args.template:
        if args.out:
            write(args.out, template())
        else:
            print(json.dumps(template(), ensure_ascii=False, indent=2))
    else:
        if not args.contract or not args.out:
            parser.error("--contract and --out required for saved analysis")
        result = run(args.contract.resolve(), args.out.resolve())
        print(json.dumps({"status": result["status"], "risk_complete": result["risk_complete"]}))
