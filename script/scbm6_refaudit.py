"""Independent, prespecified nine-covariance GH36 audit; no scientific RNG.

Production quadrature and its helper are bound as provenance, never imported.
Only selected bottom-level PMFs are recomputed. The other agent owns full saved
reference arithmetic and training/direction audits.
"""
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

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/scbm_multicorrelation/REFERENCE_AUDIT"
PROTOCOL = OUT / "PROTOCOL.json"
PATHS = {
    "training_source": "script/scbm6_run.py",
    "training_kernel": "script/scbm6_kernel.py",
    "training_freeze": "artifacts/scbm_multicorrelation/TRAIN/FREEZE.json",
    "training_results": "artifacts/scbm_multicorrelation/TRAIN/RESULTS.json",
    "reference_source": "script/scbm6_reference.py",
    "reference_helper": "script/scbminv_reference.py",
    "reference_freeze": "artifacts/scbm_multicorrelation/REFERENCE/FREEZE.json",
    "reference_results": "artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json",
    "old_truth_model": "artifacts/scbm_invariance/REFERENCE/MODEL.json",
    "experiment_design": "docs/08_research_研究工作/六相关自由学习与方向机制_20261003/EXPERIMENT_DESIGN.md",
}
METHODS = ["FIXED", "FREE", "FACT"]
KINDS = ["TAIL", "LAST"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
MEAN = [-.6, -.2, .2, .6]
CONFIG = {
    "order": 36, "chunk": 16384,
    "selection": "stream0_FIX_FREE_FACT_each_TAIL_LAST_then_stream15_FIX_FREE_FACT_TAIL",
    "selected_endpoints": 9, "deduplication": False,
    "pair_order": [list(pair) for pair in PAIRS], "mean": MEAN,
    "node_implementation": "numpy.polynomial.hermite_e.hermegauss",
    "covariance_root": "principal_symmetric_eigen_square_root",
    "probability_implementation": "exp_all_zero_log_probability_plus_logits_dot_bits",
    "maximum_PMF_difference": 5e-8, "normalization_tolerance": 2e-12,
    "root_tolerance": 2e-12, "arithmetic_tolerance": 5e-13,
    "minimum_eigenvalue": .09999,
    "new_PMF_integrations": 9, "scalar_pattern_integrals": 144,
    "Gaussian_nodes": 9 * 36**4, "new_truth_integrals": 0,
    "new_random_draws": 0, "new_parameter_fits": 0,
    "additional_orders_or_retries": 0,
}
PROGRESS = {"completed_PMF_integrations": 0, "completed_Gaussian_nodes": 0,
            "in_flight": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def patterns():
    return [[(number >> bit) & 1 for bit in (3, 2, 1, 0)] for number in range(16)]


def covariance(rho):
    assert len(rho) == 6 and all(math.isfinite(x) for x in rho)
    result = np.eye(4)
    for value, (i, j) in zip(rho, PAIRS):
        result[i, j] = result[j, i] = value
    return result


def key_of(rho):
    return "|".join(float(value).hex() for value in rho)


def select(train):
    records = train["records"]
    assert len(records) == 16 and {row["stream"] for row in records} == set(range(16))
    by_stream = {row["stream"]: row for row in records}
    ids = [(0, method, kind) for method in METHODS for kind in KINDS]
    ids.extend((15, method, "TAIL") for method in METHODS)
    selected = []
    for stream, method, kind in ids:
        output = by_stream[stream]["outputs"][method][kind]
        rho = [float(x) for x in output["rho"]]
        C = covariance(rho)
        assert np.array_equal(C, np.asarray(output["covariance"], dtype=float))
        assert np.linalg.eigvalsh(C).min() >= CONFIG["minimum_eigenvalue"]
        selected.append({"stream": stream, "method": method, "kind": kind,
                         "rho": rho, "key": key_of(rho), "covariance": C.tolist()})
    assert len(selected) == CONFIG["selected_endpoints"]
    return selected


def kl(p, q):
    assert len(p) == len(q) == 16
    assert all(math.isfinite(x) and x > 0 for x in list(p) + list(q))
    return math.fsum(x * math.log(x / y) for x, y in zip(p, q))


def probability_table(logits):
    bits = np.asarray(patterns(), dtype=float)
    zero_log_probability = -np.logaddexp(0., logits).sum(axis=1)
    return np.exp(zero_log_probability[:, None] + logits @ bits.T)


def symmetric_root(C):
    values, vectors = np.linalg.eigh(C)
    assert values.min() >= CONFIG["minimum_eigenvalue"]
    root = (vectors * np.sqrt(values)) @ vectors.T
    symmetry = float(np.max(abs(root - root.T)))
    reconstruction = float(np.max(abs(root @ root.T - C)))
    assert max(symmetry, reconstruction) <= CONFIG["root_tolerance"]
    return root, values, symmetry, reconstruction


def validate_inputs():
    train = read(ROOT / PATHS["training_results"])
    selected = select(train)  # Training identities only; no risk-dependent choice.
    tf = read(ROOT / PATHS["training_freeze"])
    assert train["source_sha256"] == tf["source_sha256"] == sha(ROOT / PATHS["training_source"])
    assert train["kernel_sha256"] == tf["kernel_sha256"] == sha(ROOT / PATHS["training_kernel"])
    assert train["freeze_sha256"] == sha(ROOT / PATHS["training_freeze"])
    assert train["design_sha256"] == tf["design_sha256"] == sha(ROOT / PATHS["experiment_design"])
    reference = read(ROOT / PATHS["reference_results"])
    rf = read(ROOT / PATHS["reference_freeze"])
    assert reference["source_sha256"] == rf["source_sha256"] == sha(ROOT / PATHS["reference_source"])
    assert reference["helper_sha256"] == rf["helper_sha256"] == sha(ROOT / PATHS["reference_helper"])
    assert reference["freeze_sha256"] == sha(ROOT / PATHS["reference_freeze"])
    assert reference["config"] == rf["config"] and rf["integrals_authorized"] is True
    assert reference["config"]["orders"] == [24, 32, 40]
    assert reference["config"]["main_order"] == 32 and reference["config"]["root"] == "cholesky"
    assert reference["training_results_sha256"] == sha(ROOT / PATHS["training_results"])
    assert reference["truth_reference_sha256"] == sha(ROOT / PATHS["old_truth_model"])
    assert reference["pair_order"] == CONFIG["pair_order"]
    for path, digest in rf["inputs"].items():
        assert sha(ROOT / path) == digest, path
    old = read(ROOT / PATHS["old_truth_model"])
    assert old["qualification_passed"] and old["patterns"] == patterns()
    assert old["config"]["mean"] == MEAN
    truth = next(row["main_pmf"] for row in old["records"] if row["t"] == 0.)
    table = {row["key"]: row for row in reference["records"]}
    assert len(table) == len(reference["records"])
    endpoint_table = {(r["stream"], r["method"], r["kind"]): r for r in reference["endpoints"]}
    assert len(endpoint_table) == 96
    for request in selected:
        row = table[request["key"]]
        endpoint = endpoint_table[request["stream"], request["method"], request["kind"]]
        assert row["rho"] == endpoint["rho"] == request["rho"]
        assert endpoint["key"] == request["key"] and row["covariance"] == request["covariance"]
        assert row["main_pmf"] == next(v["pmf"] for v in row["variants"] if v["order"] == 32)
    return selected, reference, truth


def selftest():
    # No GH nodes and no model expectations: finite synthetic linear algebra only.
    synthetic = {"records": []}
    for stream in range(16):
        outputs = {}
        for mi, method in enumerate(METHODS):
            outputs[method] = {}
            for ki, kind in enumerate(KINDS):
                rho = [.001 * (stream + 1) + .0001 * mi + .00001 * ki + .000001 * j for j in range(6)]
                outputs[method][kind] = {"rho": rho, "covariance": covariance(rho).tolist()}
        synthetic["records"].append({"stream": stream, "outputs": outputs})
    chosen = select(synthetic)
    assert [(r["stream"], r["method"], r["kind"]) for r in chosen] == (
        [(0, m, k) for m in METHODS for k in KINDS] + [(15, m, "TAIL") for m in METHODS])
    assert key_of([0.] * 6) != key_of([-0.] + [0.] * 5)
    logits = np.array([[-2., -.3, .1, 1.2], [0., 0., 0., 0.]])
    table = probability_table(logits)
    direct = []
    for x in logits:
        direct.append([math.prod(1. / (1. + math.exp(-(2 * y - 1) * float(z)))
                                 for y, z in zip(pattern, x)) for pattern in patterns()])
    assert np.max(abs(table - np.asarray(direct))) < 1e-15
    assert np.max(abs(table.sum(axis=1) - 1.)) < 1e-15 and kl(table[0], table[0]) == 0.
    C = covariance([.25, -.15, .10, .10, -.15, .25])
    root, values, symmetry, reconstruction = symmetric_root(C)
    assert not np.allclose(root, np.linalg.cholesky(C))
    result = {"passed": True, "source_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
              "config": CONFIG, "selection_count": len(chosen), "symmetric_root_checked": True,
              "finite_probability_arithmetic_checked": True, "production_results_read": False,
              "new_PMF_integrations": 0, "new_Gaussian_nodes": 0, "new_random_draws": 0}
    save(OUT / "SELFTEST.json", result)
    print(json.dumps(result))


def freeze():
    test = read(OUT / "SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(__file__)
    assert test["protocol_sha256"] == sha(PROTOCOL) and test["config"] == CONFIG
    selected, reference, truth = validate_inputs()
    assert not (OUT / "EXECUTION.json").exists()
    value = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(__file__),
             "protocol_sha256": sha(PROTOCOL), "selftest_sha256": sha(OUT / "SELFTEST.json"),
             "config": CONFIG, "selection": selected, "integrals_authorized": True,
             "bindings": {name: {"path": path, "sha256": sha(ROOT / path)} for name, path in PATHS.items()},
             "production_qualification_passed": reference["qualification_passed"]}
    save(OUT / "FREEZE.json", value)
    print(json.dumps({"frozen": True, "selected_points": 9, "Gaussian_nodes": 9 * 36**4,
                      "freeze_sha256": sha(OUT / "FREEZE.json")}))


def validate_freeze():
    frozen = read(OUT / "FREEZE.json")
    assert frozen["source_sha256"] == sha(__file__) and frozen["protocol_sha256"] == sha(PROTOCOL)
    assert frozen["selftest_sha256"] == sha(OUT / "SELFTEST.json")
    assert frozen["config"] == CONFIG and frozen["integrals_authorized"] is True
    assert set(frozen["bindings"]) == set(PATHS)
    for name, row in frozen["bindings"].items():
        assert row["path"] == PATHS[name] and row["sha256"] == sha(ROOT / row["path"])
    selected, reference, truth = validate_inputs()
    assert frozen["selection"] == selected
    return frozen, reference, truth


def independent_pmf(request):
    order = CONFIG["order"]
    C = np.asarray(request["covariance"])
    root, eig, symmetry, reconstruction = symmetric_root(C)
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / math.sqrt(2. * math.pi)
    strides = np.asarray([order**3, order**2, order, 1], dtype=np.int64)
    total = np.zeros(16, dtype=np.longdouble)
    for first in range(0, order**4, CONFIG["chunk"]):
        last = min(first + CONFIG["chunk"], order**4)
        ids = np.arange(first, last, dtype=np.int64)
        index = (ids[:, None] // strides[None, :]) % order
        logits = nodes[index] @ root.T + np.asarray(MEAN)
        values = probability_table(logits) * np.prod(weights[index], axis=1)[:, None]
        total += np.sum(values, axis=0, dtype=np.longdouble)
        PROGRESS["in_flight"]["completed_nodes"] = last
    return {**request, "pmf": [float(x) for x in total], "order": order,
            "Gaussian_nodes": order**4, "symmetric_square_root": root.tolist(),
            "eigenvalues": eig.tolist(), "symmetry_error": symmetry,
            "reconstruction_error": reconstruction}


def assess(raw, request, reference, truth):
    assert all(raw[name] == value for name, value in request.items())
    target = next(row for row in reference["records"] if row["key"] == request["key"])
    p = raw["pmf"]
    assert len(p) == 16 and all(math.isfinite(x) for x in p)
    error = max(abs(a - b) for a, b in zip(p, target["main_pmf"]))
    normalization = abs(math.fsum(p) - 1.)
    risk = kl(truth, p)
    old_risk = kl(truth, target["main_pmf"])
    arithmetic_error = abs(old_risk - target["reference_joint_KL"])
    passed = (error <= CONFIG["maximum_PMF_difference"] and min(p) > 0.
              and normalization <= CONFIG["normalization_tolerance"]
              and max(raw["symmetry_error"], raw["reconstruction_error"]) <= CONFIG["root_tolerance"]
              and min(raw["eigenvalues"]) >= CONFIG["minimum_eigenvalue"]
              and arithmetic_error <= CONFIG["arithmetic_tolerance"])
    return {**raw, "maximum_PMF_difference": error, "normalization_error": normalization,
            "minimum_probability": min(p), "joint_KL_reused_old_truth": risk,
            "production_joint_KL_recomputed": old_risk, "joint_KL_difference": risk - old_risk,
            "selected_production_risk_arithmetic_error": arithmetic_error, "passed": bool(passed)}


def accounting():
    return {"selected_PMF_integrations": 9, "scalar_pattern_integrals": 144,
            "Gaussian_nodes": 9 * 36**4, "conditional_pattern_evaluations": 144 * 36**4,
            "selected_saved_risk_checks": 9, "new_truth_integrals": 0,
            "new_random_draws": 0, "new_parameter_fits": 0, "additional_orders_or_retries": 0}


def run():
    frozen, reference, truth = validate_freeze()
    save(OUT / "EXECUTION.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
         "source_sha256": sha(__file__), "freeze_sha256": sha(OUT / "FREEZE.json"), "automatic_retry": False})
    start = time.perf_counter()
    try:
        rows = []
        for index, request in enumerate(frozen["selection"]):
            PROGRESS["in_flight"] = {"index": index, "key": request["key"], "completed_nodes": 0}
            save(OUT / "PROGRESS.json", PROGRESS, exclusive=False)
            raw = independent_pmf(request)
            save(OUT / f"point_{index:02}.json", raw)
            rows.append(assess(raw, request, reference, truth))
            PROGRESS["completed_PMF_integrations"] += 1
            PROGRESS["completed_Gaussian_nodes"] += 36**4
            PROGRESS["in_flight"] = None
            save(OUT / "PROGRESS.json", PROGRESS, exclusive=False)
        validate_freeze()
        result = {"passed": all(row["passed"] for row in rows), "source_sha256": sha(__file__),
                  "protocol_sha256": sha(PROTOCOL), "freeze_sha256": sha(OUT / "FREEZE.json"),
                  "bindings": frozen["bindings"], "config": CONFIG, "selection": frozen["selection"],
                  "patterns": patterns(), "records": rows, "accounting": accounting(),
                  "maximum_PMF_difference": max(row["maximum_PMF_difference"] for row in rows),
                  "maximum_KL_difference": max(abs(row["joint_KL_difference"]) for row in rows),
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                  "seconds": time.perf_counter() - start,
                  "scope": "Independent GH36 symmetric-root PMFs only at nine prespecified training outputs; old qualified truth PMF reused. No full saved-statistics audit and no new truth integration. Numerical qualification, not a rigorous integration-remainder certificate."}
        save(OUT / "RESULTS.json", result)
        print(json.dumps({k: result[k] for k in ("passed", "maximum_PMF_difference", "maximum_KL_difference", "accounting", "seconds")}))
    except BaseException:
        if not (OUT / "FAILURE.json").exists():
            save(OUT / "FAILURE.json", {"utc": datetime.now(timezone.utc).isoformat(),
                 "traceback": traceback.format_exc(), "progress": PROGRESS, "automatic_retry": False})
        raise


def check():
    frozen, reference, truth = validate_freeze()
    result = read(OUT / "RESULTS.json")
    assert result["source_sha256"] == sha(__file__) and result["protocol_sha256"] == sha(PROTOCOL)
    assert result["freeze_sha256"] == sha(OUT / "FREEZE.json")
    assert result["bindings"] == frozen["bindings"] and result["config"] == CONFIG
    assert result["selection"] == frozen["selection"] and result["patterns"] == patterns()
    rows = [assess(read(OUT / f"point_{i:02}.json"), request, reference, truth)
            for i, request in enumerate(frozen["selection"])]
    assert result["records"] == rows and result["accounting"] == accounting()
    assert result["passed"] == all(row["passed"] for row in rows)
    assert result["maximum_PMF_difference"] == max(row["maximum_PMF_difference"] for row in rows)
    assert result["maximum_KL_difference"] == max(abs(row["joint_KL_difference"]) for row in rows)
    print(json.dumps({"check_passed": True, "audit_passed": result["passed"],
                      "new_integrations": 0, "results_sha256": sha(OUT / "RESULTS.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["selftest", "freeze", "run", "check"])
    args = parser.parse_args()
    globals()[args.mode]()
