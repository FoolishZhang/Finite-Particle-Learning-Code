"""Post-hoc CP sensitivity for frozen B population-grid selection.

No new Gaussian samples, reference integrals, fits, or producer imports.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"artifacts/objective_strengthening/B"
OUT = ROOT/"artifacts/objective_strengthening/B_sensitivity"
CFG = {
    "B_values": [64, 256, 1024], "candidate_count": 361, "streams": 16,
    "family_size": 1083, "family_alpha": 0.05, "one_sided": True,
    "mode": "population", "reference_margin": 1e-7,
    "tail_absolute_gate": 1e-12, "tail_relative_gate": 1e-8,
    "raw_bound": "r_min + sum_j lower_j*(r_ref_j-r_min)",
    "sensitivity_risk": "max(0,r_ref-1e-7)",
    "sensitivity_bound": "sum_j lower_j*sensitivity_risk_j",
    "post_hoc": True, "preregistered_confirmation": False,
    "new_scientific_samples": 0, "new_reference_integrations": 0,
    "new_fits": 0,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def readcsv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def inputs():
    return [OLD/"SOURCE_SEAL.json", OLD/"RESULTS_SEAL.json", OLD/"CONFIG.json",
            OLD/"ENDPOINTS.csv", OLD/"REFERENCE_RISK.csv", OLD/"DECISION.json"]


def verify_original():
    result = read(OLD/"RESULTS_SEAL.json")
    assert read(OLD/"STATE.json")["status"] == "complete"
    assert result["SOURCE_SEAL_sha256"] == sha(OLD/"SOURCE_SEAL.json")
    for name in ("CONFIG.json", "ENDPOINTS.csv", "REFERENCE_RISK.csv", "DECISION.json"):
        path = OLD/name
        assert sha(path) == result["files"][rel(path)], name
    return {rel(path): sha(path) for path in inputs()}


def prepare():
    assert not OUT.exists(), "no replacement of a prior post-hoc plan"
    hashes = verify_original()
    OUT.mkdir(parents=True)
    write(OUT/"INPUT_HASHES.json", {"files": hashes})
    write(OUT/"PLAN.json", {
        "utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc_classification": "sensitivity motivated by degenerate original t risk interval",
        "config": CFG, "audit_source_sha256": sha(__file__),
        "input_hashes_sha256": sha(OUT/"INPUT_HASHES.json"),
        "before_CP_computation": True,
        "nonnegative_raw_risk_assumption_not_made": True,
        "original_negative_roundoff_preserved": True,
        "scope": "expectation of fixed published numerical risk tables over new independent bank streams",
        "not_scope": ["exact true Gaussian KL without reference error assumption",
                      "original B preregistered primary gates", "all research combined 95 percent"],
        "approval_note": "Root approved signed-risk minimum-shift correction before this plan.",
    })
    print(json.dumps({"planned": True, "PLAN_sha256": sha(OUT/"PLAN.json")}))


def verify_plan():
    plan = read(OUT/"PLAN.json")
    assert plan["config"] == CFG and plan["audit_source_sha256"] == sha(__file__)
    assert plan["input_hashes_sha256"] == sha(OUT/"INPUT_HASHES.json")
    assert read(OUT/"INPUT_HASHES.json")["files"] == verify_original()


def calculate():
    verify_plan()
    assert not (OUT/"SUMMARY.json").exists()
    endpoint_rows = readcsv(OLD/"ENDPOINTS.csv")
    risk_rows = readcsv(OLD/"REFERENCE_RISK.csv")
    assert len(risk_rows) == 361 and {int(r["index"]) for r in risk_rows} == set(range(361))
    risk_rows = sorted(risk_rows, key=lambda row: int(row["index"]))
    risks = [float(r["true_joint_KL"]) for r in risk_rows]
    assert all(math.isfinite(r) for r in risks)
    rmin = min(risks)
    adjusted = [max(0.0, r-CFG["reference_margin"]) for r in risks]
    per_cell_alpha = CFG["family_alpha"]/CFG["family_size"]
    all_rows, aggregate = [], []
    max_abs_tail = max_rel_tail = max_k16_error = 0.0
    for B in CFG["B_values"]:
        selected = [r for r in endpoint_rows if r["mode"] == "population" and int(r["B"]) == B]
        assert len(selected) == 16 and {int(r["stream"]) for r in selected} == set(range(16))
        counts = [0]*361
        for row in selected:
            j = int(row["index"])
            assert float(row["u"]) == float(risk_rows[j]["u"])
            assert float(row["v"]) == float(risk_rows[j]["v"])
            counts[j] += 1
        assert sum(counts) == 16
        lower = []
        for j, k in enumerate(counts):
            l = 0.0 if k == 0 else float(beta.ppf(per_cell_alpha, k, 17-k))
            assert math.isfinite(l) and 0 <= l <= 1
            tail = None
            if k > 0:
                tail = float(binom.sf(k-1, 16, l))
                abs_error = abs(tail-per_cell_alpha)
                rel_error = abs_error/per_cell_alpha
                assert abs_error <= CFG["tail_absolute_gate"]
                assert rel_error <= CFG["tail_relative_gate"]
                max_abs_tail, max_rel_tail = max(max_abs_tail, abs_error), max(max_rel_tail, rel_error)
                if k == 16:
                    closed = per_cell_alpha**(1/16)
                    max_k16_error = max(max_k16_error, abs(l-closed))
                    assert abs(l-closed) <= 1e-14
            lower.append(l)
            all_rows.append({
                "B": B, "index": j, "u": float(risk_rows[j]["u"]), "v": float(risk_rows[j]["v"]),
                "selected_count": k, "n": 16, "one_sided_alpha": per_cell_alpha,
                "probability_lower": l, "binomial_upper_tail_at_lower": tail,
                "raw_reference_risk": risks[j],
                "original_reference_decimal": risk_rows[j]["true_joint_KL"],
                "margin_truncated_reference_risk": adjusted[j],
                "raw_nonnegative_shifted_contribution": l*(risks[j]-rmin),
                "sensitivity_contribution": l*adjusted[j],
            })
        total_l = math.fsum(lower)
        assert total_l <= 1 and total_l >= 0
        unshifted = math.fsum(l*r for l, r in zip(lower, risks))
        raw_bound = rmin+math.fsum(l*(r-rmin) for l, r in zip(lower, risks))
        correction_form = unshifted+rmin*(1-total_l)
        assert abs(raw_bound-correction_form) <= 1e-15
        sensitive_bound = math.fsum(l*r for l, r in zip(lower, adjusted))
        aggregate.append({
            "B": B, "n": 16, "sum_probability_lowers": total_l,
            "remaining_probability_mass_upper": 1-total_l,
            "nonzero_selection_categories": [
                {"index": j, "u": float(risk_rows[j]["u"]), "v": float(risk_rows[j]["v"]),
                 "count": k, "probability_lower": lower[j]}
                for j, k in enumerate(counts) if k],
            "raw_reference_minimum": rmin,
            "sum_l_times_raw_reference_not_standalone_bound": unshifted,
            "signed_raw_reference_expectation_lower": raw_bound,
            "signed_shift_correction": rmin*(1-total_l),
            "margin_truncated_table_expectation_lower": sensitive_bound,
            "empirical_mean_raw_reference_risk": math.fsum(k*r for k, r in zip(counts, risks))/16,
            "empirical_mean_margin_table_risk": math.fsum(k*r for k, r in zip(counts, adjusted))/16,
        })
    assert len(all_rows) == 1083
    with (OUT/"ALL_1083_PROBABILITY_LOWERS.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "post_hoc": True, "family_alpha": .05, "family_size": 1083,
        "probability_statement": "simultaneously for all 3 B and 361 categories, p_Bj >= l_Bj with at least 95 percent binomial-model coverage",
        "expected_risk_statement": "three lower bounds for fixed published risk tables, conditional on the frozen population-weight selection procedure",
        "raw_reference_negative_count": sum(r < 0 for r in risks),
        "raw_reference_minimum": rmin, "reference_margin": 1e-7,
        "reference_margin_is_not_certified_integration_error": True,
        "margin_table_not_claimed_pointwise_below_signed_raw_table": True,
        "true_Gaussian_KL_interpretation_requires_unverified_uniform_reference_error_bound": True,
        "original_B_primary_gates_unchanged": True,
        "not_preregistered_confirmation": True,
        "not_combined_with_old_t_families_or_all_research_95percent": True,
        "maximum_tail_absolute_error": max_abs_tail, "maximum_tail_relative_error": max_rel_tail,
        "k16_closed_form_max_error": max_k16_error, "aggregates": aggregate,
        "new_samples": 0, "new_fits": 0, "new_integrals": 0,
    }
    write(OUT/"SUMMARY.json", summary)
    files = [OUT/"PLAN.json", OUT/"INPUT_HASHES.json", OUT/"ALL_1083_PROBABILITY_LOWERS.csv",
             OUT/"SUMMARY.json"]
    write(OUT/"MANIFEST.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
        "files": {rel(p): sha(p) for p in files},
        "old_B_results_seal_sha256": sha(OLD/"RESULTS_SEAL.json")})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def check():
    verify_plan()
    manifest = read(OUT/"MANIFEST.json")
    assert manifest["audit_source_sha256"] == sha(__file__)
    assert manifest["old_B_results_seal_sha256"] == sha(OLD/"RESULTS_SEAL.json")
    for name, expected in manifest["files"].items():
        assert sha(ROOT/name) == expected
    print(json.dumps({"passed": True, "mode": "read_only_hash_check",
                      "probability_calculation_not_reexecuted": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "calculate", "check"))
    args = parser.parse_args()
    {"plan": prepare, "calculate": calculate, "check": check}[args.mode]()
