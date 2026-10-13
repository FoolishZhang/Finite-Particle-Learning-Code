"""Independent selected endpoint quadrature and full saved-PMF arithmetic audit."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import platform
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/scbm_scale_optimization/REFERENCE_AUDIT"
PLAN = ROOT / "docs/08_research_研究工作/可学习无关尺度与优化路径_20261003/REFERENCE_AUDIT_PLAN.md"
PATHS = {
    "reference_source": "script/scbmopt_reference.py",
    "reference_helper": "script/scbminv_reference.py",
    "reference_freeze": "artifacts/scbm_scale_optimization/REFERENCE/FREEZE.json",
    "reference_results": "artifacts/scbm_scale_optimization/REFERENCE/RESULTS.json",
    "training_source": "script/scbmopt_run.py",
    "training_kernel": "script/scbmopt_kernel.py",
    "training_freeze": "artifacts/scbm_scale_optimization/TRAIN/FREEZE.json",
    "training_results": "artifacts/scbm_scale_optimization/TRAIN/RESULTS.json",
    "experiment_design": "docs/08_research_研究工作/可学习无关尺度与优化路径_20261003/EXPERIMENT_DESIGN.md",
    "old_truth_model": "artifacts/scbm_invariance/REFERENCE/MODEL.json",
}
METHODS = ["FIXED", "FREE", "FACT"]
KINDS = ["TAIL", "LAST"]
MEAN = [-0.6, -0.2, 0.2, 0.6]
COVARIANCE = [
    [1.0, 0.25, -0.15, 0.10],
    [0.25, 1.0, 0.10, -0.15],
    [-0.15, 0.10, 1.0, 0.25],
    [0.10, -0.15, 0.25, 1.0],
]
CONFIG = {
    "order": 36, "chunk": 16384, "streams": 16, "final_outputs": 96,
    "methods": METHODS, "kinds": KINDS, "mean": MEAN, "covariance": COVARIANCE,
    "selection": "stream0_all_six_then_all96_min_t_then_all96_max_t_then_truth0",
    "deduplication": "float.hex", "maximum_selected": 9,
    "node_implementation": "numpy.polynomial.hermite_e.hermegauss",
    "covariance_root": "symmetric_principal_square_root",
    "maximum_PMF_difference": 5e-8, "normalization_tolerance": 2e-12,
    "root_tolerance": 2e-12, "arithmetic_tolerance": 5e-13,
    "production_orders": [24, 32, 40], "production_main_order": 32,
    "maximum_PMF_integrations": 9, "maximum_scalar_pattern_integrals": 144,
    "maximum_Gaussian_nodes": 9 * 36**4,
}
PROGRESS = {"completed_PMF_integrations": 0, "completed_Gaussian_nodes": 0,
            "in_flight": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def labels():
    return [[(number >> bit) & 1 for bit in (3, 2, 1, 0)] for number in range(16)]


def flatten_endpoints(train):
    records = train["records"]
    assert len(records) == 16 and {r["stream"] for r in records} == set(range(16))
    by_stream = {row["stream"]: row for row in records}
    result = []
    for stream in range(16):
        outputs = by_stream[stream]["outputs"]
        assert set(outputs) == set(METHODS)
        for method in METHODS:
            assert set(outputs[method]) == set(KINDS)
            for kind in KINDS:
                t = float(outputs[method][kind]["t"])
                assert math.isfinite(t) and -0.75 <= t <= 0.75
                result.append({"stream": stream, "method": method, "kind": kind,
                               "t": t, "key": t.hex()})
    assert len(result) == 96
    return result


def selection_from_training(train):
    endpoints = flatten_endpoints(train)
    requests = [
        {"t": row["t"], "reason": f"stream0_{row['method']}_{row['kind']}"}
        for row in endpoints if row["stream"] == 0
    ]
    for name, row in (
        ("minimum_t_among_96", min(endpoints, key=lambda r: r["t"])),
        ("maximum_t_among_96", max(endpoints, key=lambda r: r["t"])),
    ):
        requests.append({"t": row["t"], "reason": name})
    requests.append({"t": 0.0, "reason": "truth0"})
    selected = {}
    for request in requests:
        t = request["t"]
        key = t.hex()
        if key not in selected:
            selected[key] = {"t": t, "key": key, "reasons": []}
        selected[key]["reasons"].append(request["reason"])
    result = list(selected.values())
    assert 1 <= len(result) <= 9 and any(r["key"] == 0.0.hex() for r in result)
    return result


def kl(p, q):
    assert len(p) == len(q) == 16
    assert all(math.isfinite(x) and x > 0 for x in p + q)
    return math.fsum(x * math.log(x / y) for x, y in zip(p, q))


def load_and_validate_inputs():
    # Selection depends only on training parameters, never on risks.
    train = read(ROOT / PATHS["training_results"])
    selected = selection_from_training(train)
    training_freeze = read(ROOT / PATHS["training_freeze"])
    assert train["source_sha256"] == training_freeze["source_sha256"] == sha(ROOT / PATHS["training_source"])
    assert train["kernel_sha256"] == training_freeze["kernel_sha256"] == sha(ROOT / PATHS["training_kernel"])
    assert train["freeze_sha256"] == sha(ROOT / PATHS["training_freeze"])
    assert training_freeze["design_sha256"] == sha(ROOT / PATHS["experiment_design"])
    assert train["design_sha256"] == training_freeze["design_sha256"]
    assert train["config"] == training_freeze["config"]
    frozen = read(ROOT / PATHS["reference_freeze"])
    reference = read(ROOT / PATHS["reference_results"])
    assert frozen["source_sha256"] == reference["source_sha256"] == sha(ROOT / PATHS["reference_source"])
    assert reference["freeze_sha256"] == sha(ROOT / PATHS["reference_freeze"])
    assert reference["config"] == frozen["config"] and frozen["integrals_authorized"] is True
    for path, digest in frozen["inputs"].items():
        assert sha(ROOT / path) == digest, path
    assert reference["training_results_sha256"] == sha(ROOT / PATHS["training_results"])
    assert reference["truth_reference_sha256"] == sha(ROOT / PATHS["old_truth_model"])
    assert reference["config"]["orders"] == CONFIG["production_orders"]
    assert reference["config"]["main_order"] == 32
    assert reference["config"]["rotation"] == "cholesky"
    assert reference["config"]["max_PMF_change"] == CONFIG["maximum_PMF_difference"]
    assert reference["config"]["normalization_tolerance"] == CONFIG["normalization_tolerance"]
    assert reference["config"]["minimum_eigenvalue"] == 0.12
    truth_model = read(ROOT / PATHS["old_truth_model"])
    assert truth_model["source_sha256"] == sha(ROOT / PATHS["reference_helper"])
    assert truth_model["patterns"] == labels()
    assert truth_model["config"]["mean"] == MEAN and truth_model["config"]["covariance"] == COVARIANCE
    truth = next(r["main_pmf"] for r in truth_model["records"] if r["t"] == 0.0)
    return train, reference, truth, selected


def saved_arithmetic(train, reference, truth):
    expected_endpoints = flatten_endpoints(train)
    expected_keys = {row["key"] for row in expected_endpoints}
    rows = reference["records"]
    assert len(rows) == len(expected_keys)
    assert {row["key"] for row in rows} == expected_keys
    table = {}
    checks = []
    recalculated = {}
    tolerance = CONFIG["arithmetic_tolerance"]
    for row in rows:
        key = row["key"]
        assert key not in table and float(row["t"]).hex() == key
        table[key] = row
        assert math.isclose(row["minimum_eigenvalue"], (1.0 - row["t"]) / 2.0,
                            rel_tol=0.0, abs_tol=tolerance)
        variants = row["variants"]
        assert [r["order"] for r in variants] == [24, 32, 40]
        main = next(r["pmf"] for r in variants if r["order"] == 32)
        assert main == row["main_pmf"]
        qualifications = {q["order"]: q for q in row["qualifications"]}
        assert len(qualifications) == 3
        rebuilt = []
        for variant in variants:
            p = variant["pmf"]
            assert len(p) == 16 and all(math.isfinite(x) for x in p)
            assert variant["rotation"] == "cholesky"
            assert variant["gaussian_nodes"] == variant["order"]**4
            change = max(abs(a - b) for a, b in zip(p, main))
            norm = abs(math.fsum(p) - 1.0)
            positive = all(x > 0.0 for x in p)
            passed = (change <= CONFIG["maximum_PMF_difference"]
                      and norm <= CONFIG["normalization_tolerance"] and positive)
            stored = qualifications[variant["order"]]
            arithmetic_passed = (
                abs(change - stored["maximum_change"]) <= tolerance
                and abs(norm - stored["normalization_error"]) <= tolerance
                and positive == stored["positive"] and passed == stored["passed"])
            rebuilt.append(passed)
            checks.append({"key": key, "t": row["t"], "order": variant["order"],
                           "maximum_change": change, "normalization_error": norm,
                           "positive": positive, "qualification_passed": passed,
                           "arithmetic_passed": bool(arithmetic_passed)})
        assert row["passed"] == all(rebuilt)
        value = kl(truth, main)
        recalculated[key] = value
        checks.append({"key": key, "t": row["t"], "field": "reference_joint_KL",
                       "recomputed": value, "stored": row["reference_joint_KL"],
                       "absolute_difference": abs(value - row["reference_joint_KL"]),
                       "arithmetic_passed": abs(value - row["reference_joint_KL"]) <= tolerance})
    supplied = reference["endpoints"]
    assert len(supplied) == 96
    supplied_by_id = {}
    for row in supplied:
        identifier = (row["stream"], row["method"], row["kind"])
        assert identifier not in supplied_by_id
        supplied_by_id[identifier] = row
    output_checks = []
    for expected in expected_endpoints:
        identifier = (expected["stream"], expected["method"], expected["kind"])
        actual = supplied_by_id[identifier]
        assert all(actual[key] == value for key, value in expected.items())
        value = recalculated[expected["key"]]
        error = abs(actual["reference_joint_KL"] - value)
        output_checks.append({**expected, "recomputed_joint_KL": value,
                              "stored_joint_KL": actual["reference_joint_KL"],
                              "absolute_difference": error, "passed": error <= tolerance})
    qualified = all(r["passed"] for r in rows)
    assert reference["qualification_passed"] == qualified
    assert reference["primary_interpretation_qualified"] == qualified
    count = len(rows)
    expected_account = {
        "unique_endpoints": count, "final_outputs": 96, "PMF_integrations": 3 * count,
        "scalar_pattern_integrals": 48 * count,
        "Gaussian_nodes": count * sum(order**4 for order in (24, 32, 40)),
        "new_truth_integrals": 0, "new_random_draws": 0, "new_parameter_fits": 0,
    }
    assert reference["accounting"] == expected_account
    return {
        "passed": all(row["arithmetic_passed"] for row in checks)
                  and all(row["passed"] for row in output_checks),
        "scope": "All saved PMF qualification and all 96 output risk arithmetic; no new quadrature here and no primary t-statistic replication.",
        "unique_endpoints": count, "qualification_reconstructions": 3 * count,
        "output_risk_checks": len(output_checks), "production_qualification_passed": qualified,
        "checks": checks, "output_checks": output_checks, "production_accounting": expected_account,
    }


def independent_pmf(t):
    order = CONFIG["order"]
    covariance = np.eye(4) + (1.0 + t) * (np.asarray(COVARIANCE) - np.eye(4))
    eig, vec = np.linalg.eigh(covariance)
    root = (vec * np.sqrt(eig)) @ vec.T
    assert float(np.min(eig)) > 0.12
    symmetry = float(np.max(np.abs(root - root.T)))
    reconstruction = float(np.max(np.abs(root @ root.T - covariance)))
    assert symmetry <= CONFIG["root_tolerance"] and reconstruction <= CONFIG["root_tolerance"]
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / math.sqrt(2.0 * math.pi)
    strides = np.asarray([order**3, order**2, order, 1], dtype=np.int64)
    bits = np.asarray(labels(), dtype=np.float64)
    total = np.zeros(16, dtype=np.longdouble)
    for first in range(0, order**4, CONFIG["chunk"]):
        numbers = np.arange(first, min(first + CONFIG["chunk"], order**4), dtype=np.int64)
        index = (numbers[:, None] // strides[None, :]) % order
        logits = nodes[index] @ root.T + np.asarray(MEAN)
        all_zero = -np.logaddexp(0.0, logits).sum(axis=1)
        probabilities = np.exp(all_zero[:, None] + logits @ bits.T)
        weighted = probabilities * np.prod(weights[index], axis=1)[:, None]
        total += np.sum(weighted, axis=0, dtype=np.longdouble)
    return {"t": t, "key": float(t).hex(), "pmf": [float(x) for x in total],
            "order": order, "root": CONFIG["covariance_root"], "Gaussian_nodes": order**4,
            "eigenvalues": eig.tolist(), "symmetric_square_root": root.tolist(),
            "symmetry_error": symmetry, "reconstruction_error": reconstruction}


def selftest():
    synthetic = {"records": []}
    for stream in range(16):
        outputs = {}
        for mi, method in enumerate(METHODS):
            outputs[method] = {
                kind: {"t": (stream - 7) / 20.0 + mi / 100.0 + ki / 1000.0}
                for ki, kind in enumerate(KINDS)}
        synthetic["records"].append({"stream": stream, "outputs": outputs})
    first = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
    for index, (method, kind) in enumerate((m, k) for m in METHODS for k in KINDS):
        synthetic["records"][0]["outputs"][method][kind]["t"] = first[index]
    synthetic["records"][1]["outputs"]["FIXED"]["TAIL"]["t"] = -0.75
    synthetic["records"][1]["outputs"]["FIXED"]["LAST"]["t"] = 0.75
    chosen = selection_from_training(synthetic)
    assert [r["t"] for r in chosen] == first + [-0.75, 0.75, 0.0]
    assert len(flatten_endpoints(synthetic)) == 96
    synthetic["records"][0]["outputs"]["FIXED"]["TAIL"]["t"] = -0.0
    synthetic["records"][0]["outputs"]["FIXED"]["LAST"]["t"] = 0.0
    zeros = selection_from_training(synthetic)
    assert {float(-0.0).hex(), float(0.0).hex()}.issubset({r["key"] for r in zeros})
    marginal = [0.2, 0.4, 0.6, 0.8]
    p = [math.prod(a if bit else 1.0 - a for a, bit in zip(marginal, y)) for y in labels()]
    expected = math.fsum(a * math.log(2.0 * a) + (1.0 - a) * math.log(2.0 * (1.0 - a))
                         for a in marginal)
    assert abs(math.fsum(p) - 1.0) < 1e-15 and kl(p, p) == 0.0
    assert abs(kl(p, [1.0 / 16.0] * 16) - expected) < 1e-15
    # Hand-built finite-probability records exercise the production schema.
    # Order/node labels here are metadata only: no quadrature nodes are created.
    flat = flatten_endpoints(synthetic)
    unique = {r["key"]: r["t"] for r in flat}
    toy_records = []
    for key, value in unique.items():
        variants = [{"order": order, "rotation": "cholesky", "pmf": p,
                     "gaussian_nodes": order**4} for order in (24, 32, 40)]
        qualifications = [{"order": order, "maximum_change": 0.0,
                           "normalization_error": abs(math.fsum(p) - 1.0),
                           "positive": True, "passed": True} for order in (24, 32, 40)]
        toy_records.append({"key": key, "t": value, "minimum_eigenvalue": (1.0 - value) / 2.0,
                            "variants": variants, "qualifications": qualifications, "main_pmf": p,
                            "reference_joint_KL": 0.0, "passed": True})
    count = len(unique)
    toy_reference = {
        "records": toy_records,
        "endpoints": [{**r, "reference_joint_KL": 0.0} for r in flat],
        "qualification_passed": True, "primary_interpretation_qualified": True,
        "accounting": {
            "unique_endpoints": count, "final_outputs": 96, "PMF_integrations": 3 * count,
            "scalar_pattern_integrals": 48 * count,
            "Gaussian_nodes": count * sum(order**4 for order in (24, 32, 40)),
            "new_truth_integrals": 0, "new_random_draws": 0, "new_parameter_fits": 0}}
    rebuilt = saved_arithmetic(synthetic, toy_reference, p)
    assert rebuilt["passed"] and rebuilt["output_risk_checks"] == 96
    result = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(__file__), "plan_sha256": sha(PLAN), "config": CONFIG,
              "selection_schema_checked": True, "exact_hex_zero_identity_checked": True,
              "finite_probability_arithmetic_checked": True, "saved_PMF_schema_checked": True,
              "production_results_read": False,
              "new_PMF_integrations": 0, "new_quadrature_nodes": 0, "new_random_draws": 0}
    save(OUT / "SELFTEST.json", result)
    print(json.dumps(result, ensure_ascii=False))


def freeze():
    test = read(OUT / "SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(__file__) and test["plan_sha256"] == sha(PLAN)
    train, reference, truth, selected = load_and_validate_inputs()
    assert not (OUT / "EXECUTION.json").exists()
    # This validation is saved-PMF arithmetic only; new quadrature starts in run.
    arithmetic = saved_arithmetic(train, reference, truth)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(__file__),
        "plan_sha256": sha(PLAN), "selftest_sha256": sha(OUT / "SELFTEST.json"),
        "config": CONFIG, "selection": selected, "integrals_authorized": True,
        "bindings": {name: {"path": path, "sha256": sha(ROOT / path)} for name, path in PATHS.items()},
        "saved_arithmetic_checked_before_new_integrals": True,
        "saved_arithmetic_passed_before_new_integrals": arithmetic["passed"],
    }
    save(OUT / "FREEZE.json", record)
    print(json.dumps({"frozen": True, "selected_points": len(selected),
                      "Gaussian_nodes": len(selected) * 36**4,
                      "freeze_sha256": sha(OUT / "FREEZE.json")}))


def validate_freeze():
    frozen = read(OUT / "FREEZE.json")
    assert frozen["source_sha256"] == sha(__file__) and frozen["plan_sha256"] == sha(PLAN)
    assert frozen["selftest_sha256"] == sha(OUT / "SELFTEST.json")
    assert frozen["config"] == CONFIG and frozen["integrals_authorized"] is True
    assert set(frozen["bindings"]) == set(PATHS)
    for name, row in frozen["bindings"].items():
        assert row["path"] == PATHS[name] and row["sha256"] == sha(ROOT / row["path"])
    train, reference, truth, selected = load_and_validate_inputs()
    assert frozen["selection"] == selected
    return frozen, train, reference, truth


def assess_selected(rows, selected, reference, old_truth):
    assert [r["key"] for r in rows] == [s["key"] for s in selected]
    table = {r["key"]: r for r in reference["records"]}
    true36 = next(r["pmf"] for r in rows if r["key"] == 0.0.hex())
    output = []
    for row, request in zip(rows, selected):
        p = row["pmf"]
        assert len(p) == 16 and all(math.isfinite(x) for x in p)
        comparisons = []
        targets = []
        if row["key"] in table:
            targets.append(("production_endpoint_main32", table[row["key"]]["main_pmf"]))
        if "truth0" in request["reasons"]:
            targets.append(("old_truth_main32", old_truth))
        assert targets
        for name, target in targets:
            errors = [abs(a - b) for a, b in zip(p, target)]
            comparisons.append({"against": name, "absolute_differences": errors,
                                "maximum_absolute_difference": max(errors),
                                "passed": max(errors) <= CONFIG["maximum_PMF_difference"]})
        norm = abs(math.fsum(p) - 1.0)
        baseline_pmf = table[row["key"]]["main_pmf"] if row["key"] in table else old_truth
        independent_risk = kl(true36, p)
        original_risk = kl(old_truth, baseline_pmf)
        passed = (min(p) > 0.0 and norm <= CONFIG["normalization_tolerance"]
                  and row["symmetry_error"] <= CONFIG["root_tolerance"]
                  and row["reconstruction_error"] <= CONFIG["root_tolerance"]
                  and min(row["eigenvalues"]) > 0.12
                  and all(c["passed"] for c in comparisons))
        output.append({**row, "reasons": request["reasons"], "normalization_error": norm,
                       "minimum_probability": min(p), "comparisons": comparisons,
                       "joint_KL_new_truth36": independent_risk,
                       "joint_KL_old_reference": original_risk,
                       "joint_KL_difference": independent_risk - original_risk,
                       "passed": bool(passed)})
    return output


def run():
    frozen, train, reference, truth = validate_freeze()
    save(OUT / "EXECUTION.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
         "source_sha256": sha(__file__), "freeze_sha256": sha(OUT / "FREEZE.json"),
         "automatic_retry": False})
    start = time.perf_counter()
    try:
        raw_rows = []
        for item in frozen["selection"]:
            PROGRESS["in_flight"] = {"key": item["key"], "t": item["t"],
                                    "completed_nodes_unknown_if_interrupted": True}
            save(OUT / "PROGRESS.json", PROGRESS, exclusive=False)
            row = independent_pmf(item["t"])
            save(OUT / f"point_{len(raw_rows):02}.json", row)
            raw_rows.append(row)
            PROGRESS["completed_PMF_integrations"] += 1
            PROGRESS["completed_Gaussian_nodes"] += 36**4
            PROGRESS["in_flight"] = None
            save(OUT / "PROGRESS.json", PROGRESS, exclusive=False)
        audited = assess_selected(raw_rows, frozen["selection"], reference, truth)
        arithmetic = saved_arithmetic(train, reference, truth)
        validate_freeze()
        count = len(audited)
        result = {
            "passed": all(r["passed"] for r in audited) and arithmetic["passed"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(__file__), "plan_sha256": sha(PLAN),
            "freeze_sha256": sha(OUT / "FREEZE.json"), "bindings": frozen["bindings"],
            "config": CONFIG, "selection": frozen["selection"], "patterns": labels(),
            "records": audited, "saved_arithmetic": arithmetic,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "accounting": {"selected_PMF_integrations": count, "scalar_pattern_integrals": 16 * count,
                           "Gaussian_nodes": count * 36**4, "conditional_pattern_evaluations": 16 * count * 36**4,
                           "full_saved_output_risk_checks": 96, "new_random_draws": 0,
                           "new_parameter_fits": 0, "additional_orders_or_retries": 0},
            "seconds": time.perf_counter() - start,
            "scope": "All 96 saved output risk/qualification arithmetic; independent bottom-level quadrature only at the prespecified selected points. Numerical qualification, not a rigorous integration remainder certificate.",
        }
        save(OUT / "RESULTS.json", result)
        print(json.dumps({"passed": result["passed"], "accounting": result["accounting"],
                          "maximum_PMF_difference": max(c["maximum_absolute_difference"] for r in audited for c in r["comparisons"]),
                          "maximum_KL_difference": max(abs(r["joint_KL_difference"]) for r in audited),
                          "results_sha256": sha(OUT / "RESULTS.json"), "seconds": result["seconds"]}))
    except BaseException:
        if not (OUT / "FAILURE.json").exists():
            save(OUT / "FAILURE.json", {"utc": datetime.now(timezone.utc).isoformat(),
                 "traceback": traceback.format_exc(), "progress": PROGRESS, "automatic_retry": False})
        raise


def check():
    frozen, train, reference, truth = validate_freeze()
    result = read(OUT / "RESULTS.json")
    assert result["source_sha256"] == sha(__file__) and result["plan_sha256"] == sha(PLAN)
    assert result["freeze_sha256"] == sha(OUT / "FREEZE.json")
    assert result["bindings"] == frozen["bindings"] and result["config"] == CONFIG
    assert result["selection"] == frozen["selection"] and result["patterns"] == labels()
    assert result["saved_arithmetic"] == saved_arithmetic(train, reference, truth)
    raw = [read(OUT / f"point_{index:02}.json") for index in range(len(frozen["selection"]))]
    assert result["records"] == assess_selected(raw, frozen["selection"], reference, truth)
    count = len(raw)
    assert result["accounting"] == {
        "selected_PMF_integrations": count, "scalar_pattern_integrals": 16 * count,
        "Gaussian_nodes": count * 36**4, "conditional_pattern_evaluations": 16 * count * 36**4,
        "full_saved_output_risk_checks": 96, "new_random_draws": 0,
        "new_parameter_fits": 0, "additional_orders_or_retries": 0}
    assert result["passed"] == (all(r["passed"] for r in result["records"])
                                and result["saved_arithmetic"]["passed"])
    print(json.dumps({"check_passed": True, "audit_passed": result["passed"],
                      "new_integrations": 0, "results_sha256": sha(OUT / "RESULTS.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["selftest", "freeze", "run", "check"])
    arguments = parser.parse_args()
    {"selftest": selftest, "freeze": freeze, "run": run, "check": check}[arguments.mode]()
