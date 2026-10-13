"""Full deterministic producer-function replay, without altering NUMERIC.

This is a reproducibility replay and cache export, NOT an independent numerical
implementation. New scientific models/samples are zero; repeated arithmetic is
explicitly counted. --check verifies receipts only, without reintegration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "artifacts/weak_emission/NUMERIC"
OUT = ROOT / "artifacts/weak_emission/REPLAY"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def verify_original():
    freeze, results = read(NUM / "FREEZE.json"), read(NUM / "RESULTS_FREEZE.json")
    assert read(NUM / "STATE.json")["state"] == "complete"
    for name, expected in results["files"].items():
        assert sha(NUM / name) == expected, name
    for row in freeze["inputs"]:
        assert sha(ROOT / row["snapshot"]) == row["sha256"], row["snapshot"]
    assert sha(NUM / "CONFIG.json") == freeze["config_sha256"]
    assert sha(NUM / "UNIT_CHECK.json") == freeze["unit_check_sha256"]
    producer = next(x for x in freeze["inputs"] if x["original"] == "script/weak_emission_numeric.py")
    return freeze, results, producer


def compare(actual, expected, prefix="", skip=()):
    """Count exact JSON leaves and preserve all differences; no tolerances."""
    differences, leaves = [], 0

    def walk(x, y, path):
        nonlocal leaves
        if path in skip:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                differences.append({"path": path, "actual_keys": sorted(x), "original_keys": sorted(y)})
            for key in sorted(set(x) & set(y)):
                walk(x[key], y[key], f"{path}.{key}" if path else key)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                differences.append({"path": path, "actual_length": len(x), "original_length": len(y)})
            for j, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{path}[{j}]")
        else:
            leaves += 1
            if type(x) is not type(y) or x != y:
                differences.append({"path": path, "replayed": x, "original": y})

    walk(actual, expected, prefix)
    return {"exactly_equal": not differences, "scalar_leaves_compared": leaves,
            "differences": differences, "excluded_paths": list(skip)}


def write():
    if (OUT / "START.json").exists():
        raise FileExistsError("Replay already started; refusing overwrite or automatic rerun.")
    freeze, results_freeze, producer = verify_original()
    save(OUT / "START.json", {
        "created": utc(), "replay_source": Path(__file__).relative_to(ROOT).as_posix(),
        "replay_source_sha256": sha(__file__),
        "original_freeze_sha256": sha(NUM / "FREEZE.json"),
        "original_results_freeze_sha256": sha(NUM / "RESULTS_FREEZE.json"),
        "producer_source": producer["snapshot"], "producer_sha256": producer["sha256"],
        "original_results_files": results_freeze["files"],
        "plan": "Same-order grid, 60 locators, qualification, 15 pair-only corrections; exact-value comparison and full evaluate-cache export.",
        "independent_implementation": False, "new_scientific_models": 0,
        "new_random_draws": 0, "new_labels": 0,
        "original_files_never_written": True})
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("weak_emission_frozen_replay_producer", ROOT / producer["snapshot"])
        p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p)
        assert p.CONFIG == read(NUM / "CONFIG.json")
        engines = {v: p.LaplaceEngine(v, p.CONFIG["theta"]) for v in p.VARIANTS}
        grid, locations = [], []
        for a in p.CONFIG["emissions"]:
            for variant, engine in engines.items():
                for rho in p.CONFIG["rho_grid"]:
                    grid.append(engine.evaluate(a, rho))
            print(f"replay grid a={a}", flush=True)
        for a in p.CONFIG["emissions"]:
            for variant, engine in engines.items():
                for k in p.KS[1:]:
                    locations.append(p.locate(engine, a, k))
            print(f"replay locations a={a}", flush=True)
        checks = p.make_checks(engines, locations, grid)
        primary = [x for x in locations if x["variant"] == "primary"]
        corrected, correction_checks = p.correction_rows(engines, primary)
        checks.extend(correction_checks)

        prediction = []
        for loc in primary:
            if loc["a"] != p.CONFIG["emissions"][0]:
                continue
            k, rho = loc["K"], loc["rho"]
            leading = 0.75 if k == 2 else k * p.CONFIG["theta"] / (k - 2)
            coeff = (leading - p.CONFIG["theta"]) ** 2 / 32
            relative = abs(loc["joint_KL"] / loc["a"] ** 4 - coeff) / coeff
            position = (rho == p.CONFIG["rho_box"][1] and all(g < 0 for g in loc["grid_gradients"])) if k == 2 else abs(rho - leading) <= p.CONFIG["gates"]["prediction_position"]
            prediction.append({"K": k, "a": loc["a"], "rho": rho, "leading_rho": leading,
                               "location_or_pointwise_gradient_pass": bool(position),
                               "leading_KL_over_a4": coeff, "KL_relative_difference": relative,
                               "risk_prediction_pass": relative <= p.CONFIG["gates"]["prediction_risk_relative"],
                               "passed": bool(position and relative <= p.CONFIG["gates"]["prediction_risk_relative"])})
        numerical_pass = all(x["passed"] for x in checks)
        prediction_count = sum(int(x["location_or_pointwise_gradient_pass"])
                               + int(x["risk_prediction_pass"]) for x in prediction)
        elapsed = time.perf_counter() - started
        ledger = {"primary_location_programs": 20, "all_variant_location_executions": len(locations),
                  "grid_rows": len(grid), "grid_scores_including_K1": len(grid) * len(p.KS),
                  "corrected_primary_points": len(corrected), "corrected_pair_variant_requests": 3 * len(corrected),
                  "new_labels": 0, "random_draws": 0, "new_fits": 0,
                  "quadrature_engines": {k: e.ledger for k, e in engines.items()},
                  "elapsed_seconds": elapsed, "Python": platform.python_version(),
                  "numpy": p.np.__version__, "scipy": p.scipy.__version__}
        replay_results = {
            "numerical_qualified": numerical_pass, "prediction_gates": prediction,
            "primary_locations": primary, "prediction_tests_total": 8,
            "prediction_tests_passed": prediction_count, "K_joint_pass_total": 4,
            "K_joint_passed": sum(x["passed"] for x in prediction),
            "all_prediction_gates_pass": all(x["passed"] for x in prediction),
            "prediction_interpretable": numerical_pass,
            "tail_bounds": {str(a): p.analytic_tail_bounds(a) for a in p.CONFIG["emissions"]},
            "ledger": ledger, "claims": {"strict_quadrature_error_certificate": False,
                                           "continuous_global_optimum_certificate": False,
                                           "statistical_confidence_interval": False}}
        replay_objects = {
            "GRID.json": {"rows": grid}, "LOCATIONS.json": {"rows": locations},
            "CHECKS.json": {"rows": checks, "all_passed": numerical_pass},
            "CORRECTION.json": {"rows": corrected, "new_Laplace_evaluations": 0,
                                "new_localizations": 0, "new_prediction_gates": 0},
            "RESULTS.json": replay_results}
        comparisons = {name: compare(value, read(NUM / name), skip=("ledger.elapsed_seconds",) if name == "RESULTS.json" else ())
                       for name, value in replay_objects.items()}
        # Export caches after exactly the original computation sequence. Reading
        # dictionary values makes no new evaluation requests or calculations.
        evaluations = [row for engine in engines.values() for row in engine.cache.values()]
        save(OUT / "EVALUATIONS.json", {
            "schema": 1, "source": "Frozen producer evaluate caches from full deterministic replay",
            "rows": evaluations, "count": len(evaluations),
            "unique_key": ["variant", "a", "rho"],
            "independent_implementation": False,
            "includes": "Fixed grid, all root-search requests, all common selected-point variant evaluations, both FD stencils",
            "excludes": "Pair-only corrected-point evaluations; those are in original CORRECTION.json"})
        save(OUT / "REPLAY_RESULTS.json", replay_results)
        save(OUT / "COMPARISON.json", {"files": comparisons,
                                       "all_exactly_equal_except_elapsed": all(x["exactly_equal"] for x in comparisons.values()),
                                       "scalar_leaves_compared": sum(x["scalar_leaves_compared"] for x in comparisons.values())})
        save(OUT / "LEDGER.json", {
            "classification": "Repeated deterministic arithmetic using original production functions; no new scientific model, sample, or independent implementation",
            "repeated_locator_executions": 60, "repeated_primary_locator_executions": 20,
            "repeated_grid_scores": len(grid) * len(p.KS),
            "repeated_corrected_pair_requests": 45, "exported_evaluate_rows": len(evaluations),
            "replayed_producer_ledger": ledger,
            "original_elapsed_seconds": read(NUM / "RESULTS.json")["ledger"]["elapsed_seconds"],
            "new_scientific_models": 0, "new_random_draws": 0, "new_labels": 0})
        verify_original()
        names = ["START.json", "EVALUATIONS.json", "REPLAY_RESULTS.json", "COMPARISON.json", "LEDGER.json"]
        save(OUT / "COMPLETE.json", {"created": utc(),
                                     "all_exactly_equal_except_elapsed": all(x["exactly_equal"] for x in comparisons.values()),
                                     "files": {n: sha(OUT / n) for n in names}})
        assert all(x["exactly_equal"] for x in comparisons.values()), "Replay differs; differences retained."
        print(json.dumps({"passed": True, "exported_rows": len(evaluations), "repeated_locators": 60,
                          "elapsed_seconds": elapsed, "comparison_leaf_count": sum(x["scalar_leaves_compared"] for x in comparisons.values())}), flush=True)
    except Exception as exc:
        save(OUT / "FAILURE.json", {"created": utc(), "type": type(exc).__name__, "message": str(exc),
                                    "original_files_modified": False, "automatic_retry": False})
        raise


def check():
    verify_original()
    start, complete = read(OUT / "START.json"), read(OUT / "COMPLETE.json")
    assert sha(__file__) == start["replay_source_sha256"]
    assert sha(NUM / "FREEZE.json") == start["original_freeze_sha256"]
    assert sha(NUM / "RESULTS_FREEZE.json") == start["original_results_freeze_sha256"]
    assert sha(ROOT / start["producer_source"]) == start["producer_sha256"]
    for name, expected in complete["files"].items():
        assert sha(OUT / name) == expected, name
    assert complete["all_exactly_equal_except_elapsed"]
    comp = read(OUT / "COMPARISON.json")
    assert comp["all_exactly_equal_except_elapsed"]
    assert all(x["exactly_equal"] and not x["differences"] for x in comp["files"].values())
    rows = read(OUT / "EVALUATIONS.json")
    assert rows["count"] == len(rows["rows"])
    keys = [(r["variant"], r["a"], r["rho"]) for r in rows["rows"]]
    assert len(keys) == len(set(keys))
    assert read(OUT / "LEDGER.json")["repeated_locator_executions"] == 60
    print(json.dumps({"passed": True, "scope": "source/original/output hashes and already executed exact replay receipt; no reintegration",
                      "exported_rows": len(keys)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write() if args.write else check()
