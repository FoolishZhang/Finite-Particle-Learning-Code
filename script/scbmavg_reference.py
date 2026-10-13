"""Mechanical post-training candidates and prespecified Gaussian references.

prepare() performs no model integration. run() requires both the original and
derived freezes and evaluates every one of 136 new candidates at all 3 orders.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import time
import traceback
import numpy as np
import scbminv_reference as helper

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/scbm_bank_averaging"
OUT = BASE / "REFERENCE"
TRAIN = BASE / "TRAIN/RESULTS.json"
CONFIG = BASE / "CONFIG.json"
DATA = BASE / "DATA.json"
CANDIDATES = BASE / "CANDIDATES.json"
PLAN = ROOT / "docs/08_research_研究工作/同目标下的粒子梯度平均_20261003/PLAN.md"
SOURCE = Path(__file__).resolve()
RUN_SOURCE = ROOT / "script/scbmavg_run.py"
KERNEL_SOURCE = ROOT / "script/scbmavg_kernel.py"
OLD_TRUTH = ROOT / "artifacts/scbm_invariance/REFERENCE/MODEL.json"
HELPER_SHA = "79c911d1601ae9740b0e045a5d879394cfd4a82aaa9e48f72070fc46b5098a41"
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
ARMS = ["L1", "L4"]
MU = [-.6, -.2, .2, .6]
REFERENCE = {
    "new_candidates": 136, "old_candidates": 2, "orders": [24, 32, 40],
    "main_order": 32, "maximum_PMF_change": 5e-8, "normalization_tolerance": 2e-12,
    "workers": 4, "include_LAST": False, "include_dataset_arm_means": True,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def covariance(rho):
    assert len(rho) == 6 and np.isfinite(rho).all()
    C = np.eye(4)
    for value, (i, j) in zip(rho, PAIRS):
        C[i, j] = C[j, i] = value
    return C


def key_of(rho):
    return "|".join(float(v).hex() for v in rho)


def patterns():
    return [[(i >> j) & 1 for j in (3, 2, 1, 0)] for i in range(16)]


def configuration():
    cfg, data = read(CONFIG), read(DATA)
    assert cfg["data_ids"] == [0, 1, 2, 3] and cfg["repeats"] == 16 and cfg["arms"] == ARMS
    assert cfg["K"] == 100 and cfg["nuisance"] == 256 and cfg["reference"] == REFERENCE
    assert data["mu"] == MU and data["pair_order"] == [list(p) for p in PAIRS]
    assert data["patterns"] == patterns()
    assert [r["data_id"] for r in data["data"]] == cfg["data_ids"]
    for row in data["data"]:
        assert len(row["counts"]) == 16 and sum(row["counts"]) == 4096
        assert row["weights"] == [v / 4096 for v in row["counts"]]
    return cfg, data


def validate_root():
    f = read(BASE / "FREEZE.json")
    assert f["run_authorized"] is True
    required = [SOURCE, RUN_SOURCE, KERNEL_SOURCE, CONFIG, DATA, PLAN, OLD_TRUTH, Path(helper.__file__)]
    assert all(rel(p) in f["inputs"] for p in required)
    for name, digest in f["inputs"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(helper.__file__) == HELPER_SHA
    cfg, data = configuration()
    for name, digest in data["inputs"].items():
        assert sha(ROOT / name) == digest, name
    return f, cfg, data


def candidate_row(index, name, rho, kind, data_id=None, repeat=None, arm=None, **extra):
    values = [float(v) for v in rho]
    C = covariance(values)
    eigmin = float(np.linalg.eigvalsh(C).min())
    assert eigmin >= .09999
    return {
        "index": index, "name": name, "kind": kind, "rho": values,
        "covariance": C.tolist(), "minimum_eigenvalue": eigmin, "key": key_of(values),
        "data_id": data_id, "repeat": repeat, "arm": arm, **extra,
    }


def construct_candidates(train, cfg, data, old):
    records = train["records"]
    order = [(d, r) for d in cfg["data_ids"] for r in range(cfg["repeats"])]
    assert [(r["data_id"], r["repeat"]) for r in records] == order and len(records) == 64
    assert [r["id"] for r in records] == [f"d{d:02}_r{r:02}" for d, r in order]
    assert old["qualification_passed"] and old["patterns"] == patterns() and old["config"]["mean"] == MU
    candidates = []
    for name, t in [("truth", 0.), ("high", .75)]:
        row = next(r for r in old["records"] if r["t"] == t)
        variant = next(v for v in row["variants"] if v["order"] == 32 and v["rotation"] == "cholesky")
        C = np.asarray(variant["covariance"])
        rho = [float(C[i, j]) for i, j in PAIRS]
        candidates.append(candidate_row(len(candidates), name, rho, "old_reference",
            reused_pmf=row["main_pmf"], origin={"path": rel(OLD_TRUTH), "t": t}))
    assert candidates[0]["reused_pmf"] == data["population_weights"]
    for record in records:
        assert record["completed_steps"] == cfg["steps"]
        d, r = record["data_id"], record["repeat"]
        assert set(record["outputs"]) == set(ARMS)
        for arm in ARMS:
            state = record["outputs"][arm]["TAIL"]
            C = covariance(state["rho"])
            assert np.array_equal(C, np.asarray(state["covariance"]))
            assert len(state["scales"]) == 256 and all(float(v) == 1. for v in state["scales"])
            candidates.append(candidate_row(len(candidates), f"D{d}_R{r:02}_{arm}_TAIL",
                state["rho"], "TAIL", d, r, arm, reused_pmf=None,
                origin={"path": rel(TRAIN), "record_id": record["id"], "output": "TAIL"}))
    for d in cfg["data_ids"]:
        for arm in ARMS:
            members = [row for row in candidates if row["kind"] == "TAIL" and row["data_id"] == d and row["arm"] == arm]
            assert len(members) == 16
            mean_rho = [math.fsum(row["rho"][j] for row in members) / 16 for j in range(6)]
            candidates.append(candidate_row(len(candidates), f"D{d}_{arm}_MEAN_TAIL", mean_rho,
                "dataset_arm_mean", d, None, arm, reused_pmf=None,
                origin={"type": "arithmetic mean of sixteen same-data same-arm TAIL parameter vectors",
                        "member_candidate_indices": [row["index"] for row in members],
                        "not_cost_matched_deployment_method": True}))
    assert len(candidates) == 138 and sum(r["reused_pmf"] is None for r in candidates) == 136
    return candidates


def prepare():
    _, cfg, data = validate_root()
    train = read(TRAIN)
    assert train["source_sha256"] == sha(RUN_SOURCE) and train["kernel_sha256"] == sha(KERNEL_SOURCE)
    assert train["freeze_sha256"] == sha(BASE / "FREEZE.json")
    assert train["config_sha256"] == sha(CONFIG) and train["data_sha256"] == sha(DATA)
    rows = construct_candidates(train, cfg, data, read(OLD_TRUTH))
    pairs = [{"data_id": d, "repeat": r, "L1": 2+2*(16*d+r), "L4": 3+2*(16*d+r)}
             for d in cfg["data_ids"] for r in range(16)]
    means = [{"data_id": d, "L1": 130+2*d, "L4": 131+2*d} for d in cfg["data_ids"]]
    audit_indices = [2+2*(16*d)+a for d in cfg["data_ids"] for a in range(2)] + list(range(130,138))
    value = {
        "schema": "scbmavg-candidates-v1", "source_sha256": sha(SOURCE),
        "training_results_sha256": sha(TRAIN), "root_freeze_sha256": sha(BASE/"FREEZE.json"),
        "config_sha256": sha(CONFIG), "data_sha256": sha(DATA),
        "candidates": rows, "mu": MU, "pair_order": [list(p) for p in PAIRS], "patterns": patterns(),
        "population_weights": data["population_weights"], "data_weights": data["data"],
        "paired_TAIL_indices": pairs, "dataset_arm_mean_indices": means,
        "prespecified_independent_PMF_audit_indices": audit_indices,
        "inputs": {rel(p): sha(p) for p in [SOURCE, TRAIN, CONFIG, DATA, BASE/"FREEZE.json", OLD_TRUTH]},
        "new_model_integrals": 0, "new_random_draws": 0, "new_fits": 0,
        "scientific_candidate_selection": False, "include_LAST": False, "deduplicate_new_PMFS": False,
        "scope": "Mechanical fixed-order map from completed training. 128 TAIL states and eight conditional-prior mean-parameter probes. LAST is not integrated; means are not cost-matched new methods.",
    }
    save(CANDIDATES, value)
    print(json.dumps({"prepared": True, "candidates": len(rows), "new_PMFS": 136,
                      "audit_indices": audit_indices, "candidates_sha256": sha(CANDIDATES)}), flush=True)


def validate_derived():
    _, cfg, data = validate_root()
    derived = read(BASE/"DERIVED_FREEZE.json")
    assert derived["run_authorized"] is True and derived["root_freeze_sha256"] == sha(BASE/"FREEZE.json")
    assert all(rel(p) in derived["inputs"] for p in [TRAIN, CANDIDATES, BASE/"FREEZE.json"])
    for name, digest in derived["inputs"].items():
        assert sha(ROOT/name) == digest, name
    selection = read(CANDIDATES)
    assert selection["source_sha256"] == sha(SOURCE)
    assert selection["training_results_sha256"] == sha(TRAIN)
    assert selection["config_sha256"] == sha(CONFIG) and selection["data_sha256"] == sha(DATA)
    assert selection["root_freeze_sha256"] == sha(BASE/"FREEZE.json")
    for name, digest in selection["inputs"].items():
        assert sha(ROOT/name) == digest, name
    assert selection["candidates"] == construct_candidates(read(TRAIN), cfg, data, read(OLD_TRUTH))
    return derived, cfg, data, selection


def qualify(variants):
    main = next(v["pmf"] for v in variants if v["order"] == 32)
    checks = []
    for row in variants:
        p = row["pmf"]
        delta = max(abs(a-b) for a,b in zip(p,main))
        norm = abs(math.fsum(p)-1.)
        positive = len(p) == 16 and all(math.isfinite(v) and v > 0 for v in p)
        checks.append({"order": row["order"], "maximum_change": delta, "normalization_error": norm,
                       "positive": positive, "passed": bool(positive and delta<=5e-8 and norm<=2e-12)})
    return main, checks


def reference_arithmetic(pmf, data):
    truth = data["population_weights"]
    assert len(pmf) == 16 and all(v > 0 and math.isfinite(v) for v in pmf)
    logs = [math.log(v) for v in pmf]
    return {
        "reference_joint_KL": math.fsum(p*math.log(p/q) for p,q in zip(truth,pmf)),
        "reference_cross_entropy_P": -math.fsum(p*l for p,l in zip(truth,logs)),
        "reference_cross_entropy_D": [-math.fsum(p*l for p,l in zip(row["weights"],logs))
                                      for row in data["data"]],
    }


def evaluate_candidate(candidate, data, binding):
    index = candidate["index"]
    assert index >= 2 and candidate["reused_pmf"] is None
    assert sha(SOURCE) == binding["source_sha256"] and sha(helper.__file__) == HELPER_SHA
    progress_path = OUT/f"progress_{index:03}.json"
    progress = {"candidate_index": index, "completed_orders": [], "completed_Gaussian_nodes": 0,
                "in_flight": None, **binding}
    save(progress_path, progress)
    variants = []
    try:
        for order in REFERENCE["orders"]:
            progress["in_flight"] = {"order": order, "partial_nodes_unknown_if_interrupted": True}
            save(progress_path, progress, False)
            variant = helper.pmf(np.asarray(candidate["covariance"]), order, "cholesky")
            variants.append(variant)
            save(OUT/f"candidate_{index:03}_order_{order}.json", variant)
            progress["completed_orders"].append(order)
            progress["completed_Gaussian_nodes"] += order**4
            progress["in_flight"] = None
            save(progress_path, progress, False)
        main, checks = qualify(variants)
        row = {**candidate, **reference_arithmetic(main,data), "main_pmf": main,
               "variants": variants, "qualifications": checks, "PMF_reused": False,
               "passed": all(q["passed"] for q in checks), **binding}
        assert sha(SOURCE) == binding["source_sha256"] and sha(helper.__file__) == HELPER_SHA
        save(OUT/f"candidate_{index:03}.json", row)
        return row
    except BaseException:
        save(OUT/f"FAILURE_{index:03}.json", {"progress": progress, "traceback": traceback.format_exc(),
             "automatic_retry": False})
        raise


def reused_record(candidate, data, binding):
    pmf = candidate["reused_pmf"]
    norm = abs(math.fsum(pmf)-1.)
    passed = all(v > 0 and math.isfinite(v) for v in pmf) and norm <= 2e-12
    return {**candidate, **reference_arithmetic(pmf,data), "main_pmf": pmf, "variants": [],
            "qualifications": [{"reused": True, "positive": min(pmf)>0,
                                "normalization_error": norm, "passed": passed}],
            "PMF_reused": True, "passed": passed, **binding}


def accounting():
    return {"candidates": 138, "reused_PMFs": 2, "new_candidates": 136,
            "PMF_integrations": 408, "scalar_pattern_integrals": 6528,
            "Gaussian_nodes": 136*sum(o**4 for o in (24,32,40)),
            "duplicate_candidates_skipped": 0, "new_truth_or_high_integrals": 0,
            "new_random_draws": 0, "new_observations": 0, "new_fits": 0,
            "additional_orders_or_retries": 0, "LAST_integrations": 0}


def run():
    _, cfg, data, selection = validate_derived()
    test = read(OUT/"SELFTEST.json")
    assert test["passed"] and test["source_sha256"] == sha(SOURCE)
    assert test["config_sha256"] == sha(CONFIG) and test["data_sha256"] == sha(DATA)
    binding = {"source_sha256": sha(SOURCE), "helper_sha256": HELPER_SHA,
               "freeze_sha256": sha(BASE/"FREEZE.json"),
               "derived_freeze_sha256": sha(BASE/"DERIVED_FREEZE.json")}
    save(OUT/"EXECUTION.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
         **binding, "automatic_retry": False, "workers": 4})
    start = time.perf_counter()
    rows = {c["index"]: reused_record(c,data,binding) for c in selection["candidates"][:2]}
    for row in rows.values():
        save(OUT/f"candidate_{row['index']:03}.json",row)
    try:
        iterator = iter(selection["candidates"][2:])
        with ProcessPoolExecutor(max_workers=REFERENCE["workers"]) as pool:
            pending = {}
            for _ in range(REFERENCE["workers"]):
                candidate = next(iterator,None)
                if candidate is not None:
                    pending[pool.submit(evaluate_candidate,candidate,data,binding)] = candidate["index"]
            try:
                while pending:
                    done,_ = wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        rows[index] = future.result()
                        save(OUT/"PROGRESS.json", {"completed_new_candidates": len(rows)-2,
                            "completed_PMF_integrations": 3*(len(rows)-2),
                            "completed_Gaussian_nodes": (len(rows)-2)*sum(o**4 for o in (24,32,40)),
                            "active_candidate_indices": sorted(pending.values())},False)
                        candidate = next(iterator,None)
                        if candidate is not None:
                            pending[pool.submit(evaluate_candidate,candidate,data,binding)] = candidate["index"]
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        records = [rows[i] for i in range(138)]
        assert len(records) == 138 and len([r for r in records if not r["PMF_reused"]]) == 136
        account = accounting()
        assert sum(v["gaussian_nodes"] for r in records for v in r["variants"]) == account["Gaussian_nodes"] == 535887872
        assert sum(len(r["variants"]) for r in records) == account["PMF_integrations"] == 408
        validate_derived()
        result = {"schema": "scbmavg-reference-results-v1", **binding,
            "candidates_sha256": sha(CANDIDATES), "training_results_sha256": sha(TRAIN),
            "config_sha256": sha(CONFIG), "data_sha256": sha(DATA), "config": REFERENCE,
            "patterns": patterns(), "pair_order": [list(p) for p in PAIRS], "records": records,
            "qualification_passed": all(r["passed"] for r in records),
            "maximum_PMF_change": max(q["maximum_change"] for r in records[2:] for q in r["qualifications"]),
            "accounting": account, "seconds": time.perf_counter()-start,
            "scope": "TAIL-only primary risks and eight same-data/arm mean-parameter probes. All 136 new candidates evaluated at all three orders without deduplication. Numerical qualification, not a rigorous Gaussian-integration certificate; mean probes are not equal-cost methods."}
        save(OUT/"RESULTS.json",result)
        print(json.dumps({k:result[k] for k in ("qualification_passed","maximum_PMF_change","accounting","seconds")}),flush=True)
    except BaseException:
        if not (OUT/"FAILURE.json").exists():
            saved_progress = [read(p) for p in sorted(OUT.glob("progress_*.json"))]
            save(OUT/"FAILURE.json", {"traceback": traceback.format_exc(), "candidate_progress": saved_progress,
                 "completed_candidate_files": [rel(p) for p in sorted(OUT.glob("candidate_[0-9][0-9][0-9].json"))],
                 "automatic_retry": False})
        raise


def check():
    cfg, data = configuration()
    assert sha(helper.__file__) == HELPER_SHA
    for name,digest in data["inputs"].items():
        assert sha(ROOT/name) == digest,name
    manufactured = []
    for d in cfg["data_ids"]:
        for repeat in range(16):
            outputs = {}
            for arm_index,arm in enumerate(ARMS):
                rho = [.002*(d+1)+.0001*(repeat+1)+.00001*(arm_index+1)*(j+1) for j in range(6)]
                state = {"rho":rho,"covariance":covariance(rho).tolist(),"scales":[1.]*256,"eigmin":.9}
                last = {"rho":[.01]*6,"covariance":covariance([.01]*6).tolist(),"scales":[1.]*256,"eigmin":.9}
                outputs[arm] = {"TAIL":state,"LAST":last}
            manufactured.append({"id":f"d{d:02}_r{repeat:02}","data_id":d,"repeat":repeat,
                                 "completed_steps":512,"outputs":outputs})
    rows = construct_candidates({"records":manufactured},cfg,data,read(OLD_TRUTH))
    assert rows[2]["rho"] == manufactured[0]["outputs"]["L1"]["TAIL"]["rho"]
    assert rows[129]["rho"] == manufactured[-1]["outputs"]["L4"]["TAIL"]["rho"]
    assert rows[130]["origin"]["member_candidate_indices"] == list(range(2,34,2))
    assert rows[137]["origin"]["member_candidate_indices"] == list(range(99,130,2))
    toy = [1/16]*16
    _,quals = qualify([{"order":o,"pmf":toy} for o in [24,32,40]])
    assert all(q["passed"] for q in quals)
    assert accounting()["Gaussian_nodes"] == 535887872
    value = {"passed":True,"source_sha256":sha(SOURCE),"helper_sha256":HELPER_SHA,
             "config_sha256":sha(CONFIG),"data_sha256":sha(DATA),
             "manufactured_training_records":64,"fixed_candidate_slots":138,
             "mean_probe_member_counts":[16]*8,"LAST_excluded":True,
             "new_random_draws":0,"new_model_integrals":0,"new_GH_nodes_created":0,
             "planned_accounting":accounting()}
    save(OUT/"SELFTEST.json",value)
    print(json.dumps(value),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["schema","check","prepare","run"])
    mode = parser.parse_args().mode
    if mode == "schema":
        print(json.dumps({"source_sha256":sha(SOURCE),"reference":REFERENCE,"accounting":accounting()}))
    else:
        globals()[mode]()
