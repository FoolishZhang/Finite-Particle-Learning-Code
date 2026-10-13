"""Independent, frozen 36^4 reference audit; no RNG and no production imports."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/scbm_invariance/REFERENCE_AUDIT"
PLAN = ROOT / "docs/08_research_研究工作/真实联合评分接口与无关扩展_20261003/REFERENCE_AUDIT_PLAN.md"
INPUTS = {
    "production_source": "script/scbminv_reference.py",
    "production_freeze": "artifacts/scbm_invariance/REFERENCE/FREEZE.json",
    "production_model": "artifacts/scbm_invariance/REFERENCE/MODEL.json",
    "production_reference_plan": "docs/08_research_研究工作/真实联合评分接口与无关扩展_20261003/REFERENCE_PLAN.md",
    "observable_readout": "artifacts/scbm_invariance/OBSERVABLE_READOUT.json",
}
MEAN = [-0.6, -0.2, 0.2, 0.6]
COVARIANCE = [
    [1.0, 0.25, -0.15, 0.10],
    [0.25, 1.0, 0.10, -0.15],
    [-0.15, 0.10, 1.0, 0.25],
    [0.10, -0.15, 0.25, 1.0],
]
CANDIDATES = [-0.75, 0.0, 0.75]
CONFIG = {
    "mean": MEAN,
    "covariance": COVARIANCE,
    "candidates": CANDIDATES,
    "order": 36,
    "chunk": 16384,
    "root": "symmetric_principal_square_root",
    "nodes": "numpy.polynomial.hermite_e.hermegauss",
    "maximum_PMF_difference": 5e-8,
    "normalization_tolerance": 2e-12,
    "root_residual_tolerance": 2e-12,
    "conditional_arithmetic_tolerance": 1e-13,
    "PMF_integrations": 3,
    "scalar_pattern_integrals": 48,
    "Gaussian_nodes": 3 * 36**4,
    "conditional_pattern_evaluations": 3 * 36**4 * 16,
    "conditional_probabilities": 72,
    "new_random_draws": 0,
    "new_fits": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def patterns():
    return [[(code >> shift) & 1 for shift in (3, 2, 1, 0)] for code in range(16)]


def validate_inputs():
    frozen = read(ROOT / INPUTS["production_freeze"])
    model = read(ROOT / INPUTS["production_model"])
    assert frozen["source_sha256"] == sha(ROOT / INPUTS["production_source"])
    assert frozen["plan_sha256"] == sha(ROOT / INPUTS["production_reference_plan"])
    assert frozen["plan_path"] == INPUTS["production_reference_plan"]
    assert frozen["integrals_authorized"] is True
    assert model["source_sha256"] == frozen["source_sha256"]
    assert model["freeze_sha256"] == sha(ROOT / INPUTS["production_freeze"])
    assert model["config"] == frozen["config"]
    for key, value in (("mean", MEAN), ("covariance", COVARIANCE), ("candidates", CANDIDATES)):
        assert model["config"][key] == value
    assert model["config"]["main_order"] == 32
    assert model["patterns"] == patterns()
    assert [row["t"] for row in model["records"]] == CANDIDATES
    for row in model["records"]:
        main = [v for v in row["variants"] if v["order"] == 32 and v["rotation"] == "cholesky"]
        assert len(main) == 1 and main[0]["pmf"] == row["main_pmf"]
    return model


def freeze():
    validate_inputs()
    assert not (OUT / "EXECUTION.json").exists()
    bound = {
        name: {"path": path, "sha256": sha(ROOT / path)}
        for name, path in INPUTS.items()
    }
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "source_sha256": sha(__file__),
        "plan_path": PLAN.relative_to(ROOT).as_posix(),
        "plan_sha256": sha(PLAN),
        "config": CONFIG,
        "bindings": bound,
        "integrals_authorized": True,
        "scope": "Independent fixed 36^4 symmetric-root PMFs and arithmetic-only post-hoc conditionals.",
    }
    create_json(OUT / "FREEZE.json", record)
    print(json.dumps({"frozen": True, "source_sha256": record["source_sha256"],
                      "freeze_sha256": sha(OUT / "FREEZE.json"),
                      "Gaussian_nodes": CONFIG["Gaussian_nodes"]}))


def validate_freeze():
    record = read(OUT / "FREEZE.json")
    assert record["source_sha256"] == sha(__file__)
    assert record["plan_sha256"] == sha(PLAN)
    assert record["config"] == CONFIG and record["integrals_authorized"] is True
    assert set(record["bindings"]) == set(INPUTS)
    for name, item in record["bindings"].items():
        assert item["path"] == INPUTS[name]
        assert item["sha256"] == sha(ROOT / item["path"])
    validate_inputs()
    return record


def independent_pmf(t):
    order = CONFIG["order"]
    covariance = np.eye(4) + (1.0 + t) * (np.asarray(COVARIANCE) - np.eye(4))
    eigvals, vectors = np.linalg.eigh(covariance)
    square_root = (vectors * np.sqrt(eigvals)) @ vectors.T
    symmetry_error = float(np.max(np.abs(square_root - square_root.T)))
    reconstruction_error = float(np.max(np.abs(square_root @ square_root.T - covariance)))
    assert np.min(eigvals) > 0.12
    assert symmetry_error <= CONFIG["root_residual_tolerance"]
    assert reconstruction_error <= CONFIG["root_residual_tolerance"]
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / math.sqrt(2.0 * math.pi)
    bits = np.asarray(patterns(), dtype=np.float64)
    strides = np.asarray([order**3, order**2, order, 1], dtype=np.int64)
    result = np.zeros(16, dtype=np.longdouble)
    for first in range(0, order**4, CONFIG["chunk"]):
        numbers = np.arange(first, min(first + CONFIG["chunk"], order**4), dtype=np.int64)
        indices = (numbers[:, None] // strides[None, :]) % order
        logits = nodes[indices] @ square_root.T + np.asarray(MEAN)
        log_all_zero = -np.logaddexp(0.0, logits).sum(axis=1)
        log_pattern_probabilities = log_all_zero[:, None] + logits @ bits.T
        quadrature_weights = np.prod(weights[indices], axis=1)
        weighted = np.exp(log_pattern_probabilities) * quadrature_weights[:, None]
        result += np.sum(weighted, axis=0, dtype=np.longdouble)
    return {
        "t": t,
        "order": order,
        "root": CONFIG["root"],
        "pmf": [float(x) for x in result],
        "Gaussian_nodes": order**4,
        "covariance": covariance.tolist(),
        "eigenvalues": eigvals.tolist(),
        "symmetric_square_root": square_root.tolist(),
        "symmetry_error": symmetry_error,
        "reconstruction_error": reconstruction_error,
    }


def kl(p, q):
    return math.fsum(float(x) * math.log(float(x) / float(y)) for x, y in zip(p, q))


def cross_entropy(p, q):
    return -math.fsum(float(x) * math.log(float(y)) for x, y in zip(p, q))


def conditional_audit(model):
    observed = read(ROOT / INPUTS["observable_readout"])
    supplied = {}
    for row in observed["conditional_probabilities"]:
        key = (row["t"], row["given_index"], row["query_index"], row["given_value"])
        assert key not in supplied
        supplied[key] = row["probability"]
    expected_keys = {
        (t, given, query, value)
        for t in CANDIDATES
        for given in range(4)
        for query in range(4) if query != given
        for value in (0, 1)
    }
    assert set(supplied) == expected_keys and len(supplied) == 72
    reference = {r["t"]: r["main_pmf"] for r in model["records"]}
    labels = patterns()
    records = []
    for t, given, query, value in sorted(expected_keys):
        p = reference[t]
        denominator = math.fsum(prob for prob, y in zip(p, labels) if y[given] == value)
        numerator = math.fsum(prob for prob, y in zip(p, labels) if y[given] == value and y[query] == 1)
        assert denominator > 0.0
        probability = numerator / denominator
        error = abs(probability - supplied[(t, given, query, value)])
        records.append({
            "t": t, "given_index": given, "query_index": query, "given_value": value,
            "numerator": numerator, "denominator": denominator,
            "probability": probability, "stored_probability": supplied[(t, given, query, value)],
            "absolute_difference": error,
            "passed": bool(error <= CONFIG["conditional_arithmetic_tolerance"]),
        })
    assert len(observed["highlight"]) == 3
    assert [r["t"] for r in observed["highlight"]] == CANDIDATES
    highlights = []
    for item in observed["highlight"]:
        key = (item["t"], item["given_index"], item["query_index"], item["given_value"])
        assert key == (item["t"], 0, 1, 1)
        expected = next(r for r in records if
                        (r["t"], r["given_index"], r["query_index"], r["given_value"]) == key)
        highlights.append({
            "t": item["t"],
            "probability": expected["probability"],
            "absolute_difference": abs(item["probability"] - expected["probability"]),
            "passed": bool(abs(item["probability"] - expected["probability"])
                           <= CONFIG["conditional_arithmetic_tolerance"]),
        })
    return {
        "scope": "Arithmetic-only conditioning of frozen observational PMFs, not SCBM intervention, causal effect, or a significance test.",
        "count": len(records),
        "records": records,
        "highlight": highlights,
        "maximum_absolute_difference": max(r["absolute_difference"] for r in records),
        "passed": all(r["passed"] for r in records + highlights),
    }


def run():
    frozen = validate_freeze()
    assert not (OUT / "EXECUTION.json").exists()
    create_json(OUT / "EXECUTION.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__), "freeze_sha256": sha(OUT / "FREEZE.json"),
        "no_repetition_or_additional_orders": True,
    })
    started = time.perf_counter()
    model = read(ROOT / INPUTS["production_model"])
    rows = [independent_pmf(t) for t in CANDIDATES]
    reference = {r["t"]: r for r in model["records"]}
    truth = next(r["pmf"] for r in rows if r["t"] == 0.0)
    for row in rows:
        saved = reference[row["t"]]
        row["normalization_error"] = abs(math.fsum(row["pmf"]) - 1.0)
        row["minimum_probability"] = min(row["pmf"])
        row["main32_absolute_differences"] = [
            abs(a - b) for a, b in zip(row["pmf"], saved["main_pmf"])]
        row["maximum_main32_absolute_difference"] = max(row["main32_absolute_differences"])
        row["joint_KL"] = kl(truth, row["pmf"])
        row["production_joint_KL"] = saved["reference_joint_KL"]
        row["joint_KL_difference"] = row["joint_KL"] - saved["reference_joint_KL"]
        row["cross_entropy"] = cross_entropy(truth, row["pmf"])
        row["production_cross_entropy"] = saved["exact_cross_entropy_reference"]
        row["cross_entropy_difference"] = row["cross_entropy"] - saved["exact_cross_entropy_reference"]
        row["passed"] = bool(
            row["minimum_probability"] > 0.0
            and row["normalization_error"] <= CONFIG["normalization_tolerance"]
            and row["maximum_main32_absolute_difference"] <= CONFIG["maximum_PMF_difference"])
    conditionals = conditional_audit(model)
    validate_freeze()
    result = {
        "passed": all(r["passed"] for r in rows) and conditionals["passed"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__), "freeze_sha256": sha(OUT / "FREEZE.json"),
        "plan_sha256": sha(PLAN), "bindings": frozen["bindings"], "config": CONFIG,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "patterns": patterns(), "records": rows, "conditional_arithmetic": conditionals,
        "accounting": {
            key: CONFIG[key] for key in
            ("PMF_integrations", "scalar_pattern_integrals", "Gaussian_nodes",
             "conditional_pattern_evaluations", "conditional_probabilities",
             "new_random_draws", "new_fits")},
        "seconds": time.perf_counter() - started,
        "scope": "Independent NumPy 36^4 symmetric-root numerical cross-check, not a rigorous quadrature remainder certificate. No main experiment scores or statistics read.",
    }
    create_json(OUT / "RESULTS.json", result)
    summary = [
        "# Independent reference audit", "",
        f"Passed: {result['passed']}. Fixed order 36, symmetric principal square root, NumPy hermegauss.",
        "",
        "| t | Joint KL | KL minus production | Max PMF difference | Normalization error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary.append(f"| {row['t']} | {row['joint_KL']:.16g} | {row['joint_KL_difference']:.6g} | "
                       f"{row['maximum_main32_absolute_difference']:.6g} | {row['normalization_error']:.6g} |")
    summary += [
        "",
        "All 72 ordered observational conditional probabilities and the three displayed highlights were rebuilt from the old production PMFs.",
        f"Maximum arithmetic difference: {conditionals['maximum_absolute_difference']:.6g}.",
        "This is post-hoc observational conditioning, not the SCBM intervention API or a causal test.",
        "",
        "New accounting: 3 PMFs, 48 pattern integrals, 5,038,848 Gaussian quadrature nodes; zero RNG and zero fits.",
        "No extra order, candidate, or model integration was added after seeing the result. Numerical agreement is not a rigorous integration remainder certificate.",
        "",
        f"Source SHA256: {sha(__file__)}",
        f"Freeze SHA256: {sha(OUT / 'FREEZE.json')}",
        f"Results SHA256: {sha(OUT / 'RESULTS.json')}",
    ]
    with (OUT / "SUMMARY.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(summary) + "\n")
    print(json.dumps({
        "passed": result["passed"],
        "maximum_PMF_difference": max(r["maximum_main32_absolute_difference"] for r in rows),
        "maximum_KL_difference": max(abs(r["joint_KL_difference"]) for r in rows),
        "conditional_count": conditionals["count"],
        "conditional_max_difference": conditionals["maximum_absolute_difference"],
        "accounting": result["accounting"], "seconds": result["seconds"],
    }))


def check():
    frozen = validate_freeze()
    result = read(OUT / "RESULTS.json")
    execution = read(OUT / "EXECUTION.json")
    assert result["source_sha256"] == frozen["source_sha256"] == execution["source_sha256"]
    assert result["freeze_sha256"] == execution["freeze_sha256"] == sha(OUT / "FREEZE.json")
    assert result["plan_sha256"] == sha(PLAN)
    assert result["bindings"] == frozen["bindings"] and result["config"] == CONFIG
    assert result["patterns"] == patterns()
    assert [row["t"] for row in result["records"]] == CANDIDATES
    assert sum(row["Gaussian_nodes"] for row in result["records"]) == CONFIG["Gaussian_nodes"]
    model = read(ROOT / INPUTS["production_model"])
    assert result["conditional_arithmetic"] == conditional_audit(model)
    assert result["accounting"] == {k: CONFIG[k] for k in
        ("PMF_integrations", "scalar_pattern_integrals", "Gaussian_nodes",
         "conditional_pattern_evaluations", "conditional_probabilities",
         "new_random_draws", "new_fits")}
    saved = {r["t"]: r for r in model["records"]}
    truth = next(r["pmf"] for r in result["records"] if r["t"] == 0.0)
    for row in result["records"]:
        assert row["normalization_error"] == abs(math.fsum(row["pmf"]) - 1.0)
        assert row["minimum_probability"] == min(row["pmf"])
        assert row["main32_absolute_differences"] == [
            abs(a - b) for a, b in zip(row["pmf"], saved[row["t"]]["main_pmf"])]
        assert row["maximum_main32_absolute_difference"] == max(row["main32_absolute_differences"])
        assert row["joint_KL"] == kl(truth, row["pmf"])
        assert row["production_joint_KL"] == saved[row["t"]]["reference_joint_KL"]
        assert row["joint_KL_difference"] == row["joint_KL"] - row["production_joint_KL"]
        assert row["cross_entropy"] == cross_entropy(truth, row["pmf"])
        assert row["production_cross_entropy"] == saved[row["t"]]["exact_cross_entropy_reference"]
        assert row["cross_entropy_difference"] == row["cross_entropy"] - row["production_cross_entropy"]
        assert row["passed"] == bool(
            row["minimum_probability"] > 0.0
            and row["normalization_error"] <= CONFIG["normalization_tolerance"]
            and row["maximum_main32_absolute_difference"] <= CONFIG["maximum_PMF_difference"])
    assert result["passed"] == (all(r["passed"] for r in result["records"])
                                and result["conditional_arithmetic"]["passed"])
    print(json.dumps({"check_passed": True, "audit_passed": result["passed"],
                      "results_sha256": sha(OUT / "RESULTS.json"),
                      "new_integrations_in_check": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run", "check"))
    arguments = parser.parse_args()
    {"freeze": freeze, "run": run, "check": check}[arguments.mode]()
