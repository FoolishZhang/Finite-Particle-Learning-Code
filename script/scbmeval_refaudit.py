"""Independent six-point reference quadrature and complete saved-reference audit."""
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

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/scbm_frozen_evaluation/REFERENCE_AUDIT"
PLAN = ROOT / "docs/08_research_研究工作/冻结表示的条件与总体目标_20261003/REFERENCE_AUDIT_PLAN.md"
PATHS = {
    "selection": "artifacts/scbm_frozen_evaluation/MODEL_SELECTION.json",
    "reference_source": "script/scbmeval_reference.py",
    "production_helper": "script/scbminv_reference.py",
    "reference_freeze": "artifacts/scbm_frozen_evaluation/REFERENCE/FREEZE.json",
    "reference_results": "artifacts/scbm_frozen_evaluation/REFERENCE/RESULTS.json",
    "old_invariance": "artifacts/scbm_invariance/REFERENCE/MODEL.json",
    "old_optimization_reference": "artifacts/scbm_scale_optimization/REFERENCE/RESULTS.json",
    "old_training": "artifacts/scbm_scale_optimization/TRAIN/RESULTS.json",
    "experiment_design": "docs/08_research_研究工作/冻结表示的条件与总体目标_20261003/EXPERIMENT_DESIGN.md",
}
MEAN = [-0.6, -0.2, 0.2, 0.6]
COV = [[1., .25, -.15, .10], [.25, 1., .10, -.15],
       [-.15, .10, 1., .25], [.10, -.15, .25, 1.]]
PROFILES = [["TAIL", "learned"], ["TAIL", "half"], ["TAIL", "fact"],
            ["LAST", "learned"], ["LAST", "fact"]]
CONFIG = {
    "selected_grid_indices": [1, 5, 9, 14, 19, 10], "truth_index": 10,
    "order": 36, "chunk": 16384, "PMF_integrations": 6,
    "scalar_pattern_integrals": 96, "Gaussian_nodes": 6 * 36**4,
    "conditional_pattern_evaluations": 16 * 6 * 36**4,
    "max_PMF_difference": 5e-8, "normalization_tolerance": 2e-12,
    "root_tolerance": 2e-12, "arithmetic_tolerance": 5e-13,
    "node_implementation": "numpy.polynomial.hermite_e.hermegauss",
    "root": "symmetric_principal_square_root",
    "new_random_draws": 0, "new_fits": 0, "new_diagnostic_selections": 0,
}
PROGRESS = {"completed_PMF_integrations": 0, "completed_Gaussian_nodes": 0, "in_flight": None}


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


def patterns():
    return [[(code >> bit) & 1 for bit in (3, 2, 1, 0)] for code in range(16)]


def count_patterns(y):
    y = np.asarray(y)
    assert y.ndim == 2 and y.shape[1] == 4
    assert np.all((y == 0) | (y == 1))
    code = y[:, 0].astype(np.int64) * 8 + y[:, 1] * 4 + y[:, 2] * 2 + y[:, 3]
    return np.bincount(code.astype(np.int64), minlength=16).tolist()


def ce(weights, pmf):
    assert len(weights) == len(pmf) == 16 and all(p > 0 for p in pmf)
    return -math.fsum(w * math.log(p) for w, p in zip(weights, pmf))


def kl(weights, pmf):
    assert len(weights) == len(pmf) == 16 and all(w > 0 and p > 0 for w, p in zip(weights, pmf))
    return math.fsum(w * math.log(w / p) for w, p in zip(weights, pmf))


def selected_points(selection):
    grid = selection["grid_t"]
    assert len(grid) == 21 and grid[0] == -.75 and grid[10] == 0. and grid[20] == .75
    assert selection["zero_index"] == 10 and selection["contrast_index"] == 14
    return [{"grid_index": i, "t": grid[i], "key": float(grid[i]).hex()}
            for i in CONFIG["selected_grid_indices"]]


def load_inputs():
    selection = read(ROOT / PATHS["selection"])
    old = read(ROOT / PATHS["old_invariance"])
    opt = read(ROOT / PATHS["old_optimization_reference"])
    train = read(ROOT / PATHS["old_training"])
    reference = read(ROOT / PATHS["reference_results"])
    frozen = read(ROOT / PATHS["reference_freeze"])
    assert frozen["source_sha256"] == reference["source_sha256"] == sha(ROOT / PATHS["reference_source"])
    assert reference["freeze_sha256"] == sha(ROOT / PATHS["reference_freeze"])
    assert frozen["integrals_authorized"] is True and reference["config"] == frozen["config"]
    for path, digest in frozen["inputs"].items():
        assert sha(ROOT / path) == digest, path
    assert reference["selection_sha256"] == sha(ROOT / PATHS["selection"])
    assert selection["training_result_sha256"] == sha(ROOT / PATHS["old_training"])
    assert selection["truth_reference_sha256"] == sha(ROOT / PATHS["old_invariance"])
    assert old["source_sha256"] == sha(ROOT / PATHS["production_helper"])
    assert old["config"]["mean"] == MEAN and old["config"]["covariance"] == COV
    assert selection["patterns"] == old["patterns"] == patterns()
    truth = next(r["main_pmf"] for r in old["records"] if r["t"] == 0.)
    assert selection["population_weights"] == truth
    assert [r["stream"] for r in selection["models"]] == list(range(16))
    old_train = {r["stream"]: r for r in train["records"]}
    expected_paths = {f"artifacts/scbm_scale_optimization/TRAIN/input_{s:02}.npz" for s in range(16)}
    assert set(selection["old_input_NPZ"]) == expected_paths
    counts = []
    for model in selection["models"]:
        s = model["stream"]
        path = f"artifacts/scbm_scale_optimization/TRAIN/input_{s:02}.npz"
        assert sha(ROOT / path) == selection["old_input_NPZ"][path]
        with np.load(ROOT / path, allow_pickle=False) as arrays:
            active = arrays["active_labels"]
            assert active.shape == (4096, 4)
            rebuilt = count_patterns(active)
        assert model["N"] == sum(rebuilt) == 4096
        assert rebuilt == model["empirical_counts"]
        assert model["empirical_weights"] == [n / 4096. for n in rebuilt]
        for kind in ("TAIL", "LAST"):
            assert model[kind] == old_train[s]["outputs"]["FREE"][kind]
        counts.append({"stream": s, "counts": rebuilt, "N": 4096,
                       "input_sha256": selection["old_input_NPZ"][path], "passed": True})
    cfg = reference["config"]
    assert cfg["orders"] == [24, 32, 40] and cfg["main_order"] == 32
    assert cfg["max_PMF_change"] == CONFIG["max_PMF_difference"]
    assert cfg["normalization_tolerance"] == CONFIG["normalization_tolerance"]
    assert cfg["minimum_eigenvalue"] == .12
    assert cfg["new_grid_indices"] == [i for i in range(21) if i not in (0, 10, 20)]
    selected_points(selection)
    return selection, reference, old, opt, counts


def saved_arithmetic(selection, reference, old, opt, counts):
    tolerance = CONFIG["arithmetic_tolerance"]
    grid = selection["grid_t"]
    table = {}
    for i in (0, 10, 20):
        t = grid[i]
        row = next(r for r in old["records"] if float(r["t"]).hex() == float(t).hex())
        table[float(t).hex()] = {"t": t, "pmf": row["main_pmf"],
            "origin": "old_scbminv", "source_sha256": sha(ROOT / PATHS["old_invariance"])}
    records = reference["records"]
    assert [r["grid_index"] for r in records] == [i for i in range(21) if i not in (0, 10, 20)]
    qualification_checks = []
    stored_flat = []
    for row in records:
        i, t = row["grid_index"], row["t"]
        assert float(t).hex() == float(grid[i]).hex()
        variants = row["variants"]
        assert [v["order"] for v in variants] == [24, 32, 40]
        main = next(v["pmf"] for v in variants if v["order"] == 32)
        assert main == row["main_pmf"]
        stored = {q["order"]: q for q in row["qualifications"]}
        assert len(stored) == 3
        decisions = []
        for variant in variants:
            p = variant["pmf"]
            assert len(p) == 16 and all(math.isfinite(v) for v in p)
            assert variant["rotation"] == "cholesky" and variant["gaussian_nodes"] == variant["order"]**4
            change = max(abs(a - b) for a, b in zip(p, main))
            norm = abs(math.fsum(p) - 1.)
            positive = all(v > 0 for v in p)
            passed = change <= CONFIG["max_PMF_difference"] and norm <= CONFIG["normalization_tolerance"] and positive
            saved = stored[variant["order"]]
            assert saved["grid_index"] == i
            agrees = (abs(change - saved["max_change"]) <= tolerance
                      and abs(norm - saved["normalization_error"]) <= tolerance and passed == saved["passed"])
            qualification_checks.append({"grid_index": i, "order": variant["order"],
                "max_change": change, "normalization_error": norm, "positive": positive,
                "qualification_passed": passed, "arithmetic_passed": bool(agrees)})
            decisions.append(passed)
        assert row["passed"] == all(decisions)
        stored_flat.extend(row["qualifications"])
        table[float(t).hex()] = {"t": t, "pmf": main, "origin": "new_grid", "grid_index": i}
    assert reference["qualifications"] == stored_flat
    assert reference["qualification_passed"] == all(q["qualification_passed"] for q in qualification_checks)
    reused = []
    old_opt = {r["key"]: r for r in opt["records"]}
    for model in selection["models"]:
        for kind in ("TAIL", "LAST"):
            t = model[kind]["t"]
            key = float(t).hex()
            if key not in table:
                row = old_opt[key]
                assert row["passed"] and float(row["t"]).hex() == key
                table[key] = {"t": t, "pmf": row["main_pmf"], "origin": "old_scbmopt",
                              "source_sha256": sha(ROOT / PATHS["old_optimization_reference"])}
            reused.append({"stream": model["stream"], "kind": kind, "t": t, "key": key,
                           "origin": table[key]["origin"]})
    assert reference["PMF_table"] == table
    assert reference["actual_reused"] == reused and len(reused) == 32
    assert reference["profiles"] == PROFILES and reference["targets"] == ["P", "D"]
    population = selection["population_weights"]
    expected_ce = np.empty((16, 5, 22, 2), dtype=float)
    expected_kl = np.empty((16, 5, 22), dtype=float)
    for m, model in enumerate(selection["models"]):
        for c, (kind, _) in enumerate(PROFILES):
            for j, t in enumerate(grid + [model[kind]["t"]]):
                pmf = table[float(t).hex()]["pmf"]
                expected_ce[m, c, j] = [ce(population, pmf), ce(model["empirical_weights"], pmf)]
                expected_kl[m, c, j] = kl(population, pmf)
    expected_relative = expected_ce - expected_ce[:, :, 10:11, :]
    fields = [
        ("cross_entropy", expected_ce),
        ("relative_cross_entropy", expected_relative),
        ("reference_joint_KL", expected_kl),
    ]
    scalar_checks = []
    for name, expected in fields:
        actual = np.asarray(reference[name], dtype=float)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        error = float(np.max(np.abs(actual - expected)))
        scalar_checks.append({"field": name, "shape": list(actual.shape),
                              "count": int(actual.size), "maximum_absolute_difference": error,
                              "passed": error <= tolerance})
    expected_account = {
        "new_PMF_integrations": 54, "new_scalar_pattern_integrals": 864,
        "new_Gaussian_nodes": 70926336, "reused_grid_queries": 3,
        "reused_actual_queries": 32, "new_random_draws": 0, "new_fits": 0}
    assert reference["accounting"] == expected_account
    assert sum(v["gaussian_nodes"] for r in records for v in r["variants"]) == 70926336
    return {
        "passed": all(q["arithmetic_passed"] for q in qualification_checks)
                  and all(q["passed"] for q in scalar_checks),
        "empirical_count_checks": counts, "empirical_models_checked": 16,
        "PMF_table_entries_checked": len(table), "actual_reuse_mappings_checked": 32,
        "new_grid_records_checked": 18, "saved_qualification_checks": qualification_checks,
        "scalar_checks": scalar_checks, "total_scalar_checks": sum(q["count"] for q in scalar_checks),
        "production_accounting": expected_account,
        "scope": "All saved PMF reuse mappings, 54 qualifications, 16 empirical count vectors and 8800 CE/relative-CE/KL scalars; no new integrals in this reconstruction."}


def independent_pmf(t):
    order = CONFIG["order"]
    covariance = np.eye(4) + (1. + t) * (np.asarray(COV) - np.eye(4))
    eig, vec = np.linalg.eigh(covariance)
    root = (vec * np.sqrt(eig)) @ vec.T
    symmetry = float(np.max(np.abs(root - root.T)))
    reconstruction = float(np.max(np.abs(root @ root.T - covariance)))
    assert eig.min() > .12 and symmetry <= CONFIG["root_tolerance"] and reconstruction <= CONFIG["root_tolerance"]
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / math.sqrt(2. * math.pi)
    steps = np.asarray([order**3, order**2, order, 1], dtype=np.int64)
    bits = np.asarray(patterns(), dtype=float)
    result = np.zeros(16, dtype=np.longdouble)
    for start in range(0, order**4, CONFIG["chunk"]):
        numbers = np.arange(start, min(start + CONFIG["chunk"], order**4), dtype=np.int64)
        indices = (numbers[:, None] // steps[None, :]) % order
        logits = nodes[indices] @ root.T + np.asarray(MEAN)
        log_zero = -np.logaddexp(0., logits).sum(axis=1)
        values = np.exp(log_zero[:, None] + logits @ bits.T)
        values *= np.prod(weights[indices], axis=1)[:, None]
        result += np.sum(values, axis=0, dtype=np.longdouble)
    return {"t": t, "key": float(t).hex(), "order": order, "root": CONFIG["root"],
            "pmf": [float(v) for v in result], "Gaussian_nodes": order**4,
            "eigenvalues": eig.tolist(), "symmetric_square_root": root.tolist(),
            "symmetry_error": symmetry, "reconstruction_error": reconstruction}


def selftest():
    synthetic = {"grid_t": np.linspace(-.75, .75, 21).tolist(), "zero_index": 10, "contrast_index": 14}
    selected = selected_points(synthetic)
    assert [r["grid_index"] for r in selected] == [1, 5, 9, 14, 19, 10]
    assert len({r["key"] for r in selected}) == 6 and selected[-1]["t"] == 0.
    assert count_patterns(np.repeat(np.asarray(patterns(), dtype=np.int64), 256, axis=0)) == [256] * 16
    marginal = [.2, .4, .6, .8]
    p = [math.prod(a if bit else 1-a for a, bit in zip(marginal, y)) for y in patterns()]
    exact_ce = -math.fsum(a * math.log(a) + (1-a) * math.log(1-a) for a in marginal)
    assert abs(ce(p, p) - exact_ce) < 1e-15 and kl(p, p) == 0.
    assert abs(kl(p, [1/16] * 16) - (math.log(16) - exact_ce)) < 1e-15
    assert 16 * 5 * 22 * 2 * 2 + 16 * 5 * 22 == 8800
    record = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(__file__), "plan_sha256": sha(PLAN), "config": CONFIG,
              "index_pattern_count_CE_KL_checks": True, "production_reference_read": False,
              "new_integrals": 0, "new_quadrature_nodes": 0, "new_random_draws": 0}
    save(OUT / "SELFTEST.json", record)
    print(json.dumps(record, ensure_ascii=False))


def freeze():
    test = read(OUT / "SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(__file__) and test["plan_sha256"] == sha(PLAN)
    selection, reference, old, opt, counts = load_inputs()
    arithmetic = saved_arithmetic(selection, reference, old, opt, counts)
    assert not (OUT / "EXECUTION.json").exists()
    bound = {name: {"path": path, "sha256": sha(ROOT / path)} for name, path in PATHS.items()}
    for path, digest in selection["old_input_NPZ"].items():
        bound[path] = {"path": path, "sha256": digest}
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha(__file__),
        "plan_sha256": sha(PLAN), "selftest_sha256": sha(OUT / "SELFTEST.json"),
        "config": CONFIG, "selection": selected_points(selection), "bindings": bound,
        "integrals_authorized": True, "saved_arithmetic_passed_before_new_integrals": arithmetic["passed"]}
    save(OUT / "FREEZE.json", record)
    print(json.dumps({"frozen": True, "selected_points": 6,
                      "freeze_sha256": sha(OUT / "FREEZE.json"), "Gaussian_nodes": CONFIG["Gaussian_nodes"]}))


def validate_freeze():
    record = read(OUT / "FREEZE.json")
    assert record["source_sha256"] == sha(__file__) and record["plan_sha256"] == sha(PLAN)
    assert record["selftest_sha256"] == sha(OUT / "SELFTEST.json")
    assert record["config"] == CONFIG and record["integrals_authorized"] is True
    for row in record["bindings"].values():
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    values = load_inputs()
    assert record["selection"] == selected_points(values[0])
    expected_names = set(PATHS) | set(values[0]["old_input_NPZ"])
    assert set(record["bindings"]) == expected_names
    return record, values


def assess(rows, selection, reference):
    selected = selected_points(selection)
    assert [r["key"] for r in rows] == [r["key"] for r in selected]
    truth36 = next(r["pmf"] for r in rows if r["key"] == float(0.).hex())
    oldtruth = reference["PMF_table"][float(0.).hex()]["pmf"]
    result = []
    for row, chosen in zip(rows, selected):
        original = reference["PMF_table"][row["key"]]
        p = row["pmf"]
        errors = [abs(a-b) for a, b in zip(p, original["pmf"])]
        norm = abs(math.fsum(p) - 1.)
        value, saved = kl(truth36, p), kl(oldtruth, original["pmf"])
        passed = (min(p) > 0. and norm <= CONFIG["normalization_tolerance"]
                  and max(errors) <= CONFIG["max_PMF_difference"]
                  and row["symmetry_error"] <= CONFIG["root_tolerance"]
                  and row["reconstruction_error"] <= CONFIG["root_tolerance"]
                  and min(row["eigenvalues"]) > .12)
        result.append({**row, "grid_index": chosen["grid_index"], "compared_origin": original["origin"],
                       "normalization_error": norm, "minimum_probability": min(p),
                       "main32_absolute_differences": errors, "maximum_PMF_difference": max(errors),
                       "joint_KL_truth36": value, "joint_KL_reference": saved,
                       "joint_KL_difference": value - saved, "passed": bool(passed)})
    return result


def run():
    frozen, values = validate_freeze()
    selection, reference, old, opt, counts = values
    save(OUT / "EXECUTION.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
         "source_sha256": sha(__file__), "freeze_sha256": sha(OUT / "FREEZE.json"), "automatic_retry": False})
    started = time.perf_counter()
    try:
        rows = []
        for chosen in frozen["selection"]:
            PROGRESS["in_flight"] = {**chosen, "partial_nodes_unknown_if_interrupted": True}
            save(OUT / "PROGRESS.json", PROGRESS, False)
            row = independent_pmf(chosen["t"])
            save(OUT / f"point_{len(rows):02}.json", row)
            rows.append(row)
            PROGRESS["completed_PMF_integrations"] += 1
            PROGRESS["completed_Gaussian_nodes"] += 36**4
            PROGRESS["in_flight"] = None
            save(OUT / "PROGRESS.json", PROGRESS, False)
        audited = assess(rows, selection, reference)
        arithmetic = saved_arithmetic(selection, reference, old, opt, counts)
        validate_freeze()
        result = {
            "passed": all(r["passed"] for r in audited) and arithmetic["passed"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha(__file__), "plan_sha256": sha(PLAN),
            "freeze_sha256": sha(OUT / "FREEZE.json"), "bindings": frozen["bindings"],
            "config": CONFIG, "selection": frozen["selection"], "patterns": patterns(),
            "records": audited, "saved_arithmetic": arithmetic,
            "accounting": {key: CONFIG[key] for key in
                ("PMF_integrations", "scalar_pattern_integrals", "Gaussian_nodes",
                 "conditional_pattern_evaluations", "new_random_draws", "new_fits", "new_diagnostic_selections")},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "seconds": time.perf_counter() - started,
            "scope": "Six fixed bottom-level symmetric-root PMFs; all PMF reuse, 16 empirical count vectors and 8800 saved CE/KL scalars independently reconstructed. Numerical qualification only, not a strict quadrature remainder certificate."}
        save(OUT / "RESULTS.json", result)
        print(json.dumps({"passed": result["passed"], "accounting": result["accounting"],
                          "maximum_PMF_difference": max(r["maximum_PMF_difference"] for r in audited),
                          "maximum_KL_difference": max(abs(r["joint_KL_difference"]) for r in audited),
                          "saved_scalar_checks": arithmetic["total_scalar_checks"],
                          "results_sha256": sha(OUT / "RESULTS.json"), "seconds": result["seconds"]}))
    except BaseException:
        if not (OUT / "FAILURE.json").exists():
            save(OUT / "FAILURE.json", {"utc": datetime.now(timezone.utc).isoformat(),
                 "traceback": traceback.format_exc(), "progress": PROGRESS, "automatic_retry": False})
        raise


def check():
    frozen, values = validate_freeze()
    selection, reference, old, opt, counts = values
    result = read(OUT / "RESULTS.json")
    assert result["source_sha256"] == sha(__file__) and result["plan_sha256"] == sha(PLAN)
    assert result["freeze_sha256"] == sha(OUT / "FREEZE.json") and result["config"] == CONFIG
    assert result["bindings"] == frozen["bindings"] and result["selection"] == frozen["selection"]
    assert result["saved_arithmetic"] == saved_arithmetic(selection, reference, old, opt, counts)
    rows = [read(OUT / f"point_{i:02}.json") for i in range(6)]
    assert result["records"] == assess(rows, selection, reference)
    assert result["accounting"] == {key: CONFIG[key] for key in
        ("PMF_integrations", "scalar_pattern_integrals", "Gaussian_nodes",
         "conditional_pattern_evaluations", "new_random_draws", "new_fits", "new_diagnostic_selections")}
    assert result["passed"] == (all(r["passed"] for r in result["records"]) and result["saved_arithmetic"]["passed"])
    print(json.dumps({"check_passed": True, "audit_passed": result["passed"],
                      "new_integrals": 0, "results_sha256": sha(OUT / "RESULTS.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["selftest", "freeze", "run", "check"])
    args = parser.parse_args()
    {"selftest": selftest, "freeze": freeze, "run": run, "check": check}[args.mode]()
