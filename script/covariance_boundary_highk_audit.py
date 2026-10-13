"""Separate HIGHK audit, reusing only the already independent audit helpers."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HELPER = ROOT/"script/covariance_boundary_profile_audit.py"
HELPER_SHA = "3bb1d5dde7d2bb7bd18ff57cf90a0fcce482b556bd65994f6ec72c680c4716dd"
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise AssertionError("independent helper version changed")
from script import covariance_boundary_profile_audit as independent
import numpy as np

OUT = ROOT/"artifacts/covariance_boundary/HIGHK"
PROFILE = OUT.parent/"PROFILE"
NUM = OUT.parent/"NUMERIC"
DOC = ROOT/"docs/08_research_研究工作/相关结构补偿与发射噪声_20261001"
RECEIPT = OUT/"INDEPENDENT_AUDIT.json"
SCHEMA = "covariance-boundary-highk-independent-audit-v1"
read, sha, rel, verify = independent.read, independent.sha, independent.rel, independent.verify


def replay_inputs(record, common_prefix, private_prefix, cfg, audit):
    d, s = record["d"], record["stream"]
    seed = cfg["seed"]+100000*d+1000*s
    audit.require((record["seed_common"], record["seed_private"]) == (seed, seed+1), "HIGHK seeds")
    common_rng = np.random.Generator(np.random.PCG64(seed))
    private_rng = np.random.Generator(np.random.PCG64(seed+1))
    audit.require([common_rng.bit_generator.state, private_rng.bit_generator.state] == record["RNG_initial"],
                  "HIGHK initial RNG states")
    K = max(cfg["Ks"])
    for bank in range(cfg["banks"]):
        common = common_rng.standard_normal((1, K))
        private = private_rng.standard_normal((1, K, d))
        if bank == 0:
            audit.require(np.array_equal(common, common_prefix), "HIGHK exact common prefix")
            audit.require(np.array_equal(private, private_prefix), "HIGHK exact private prefix")
    audit.require([common_rng.bit_generator.state, private_rng.bit_generator.state] == record["RNG_final"],
                  "HIGHK final RNG states")
    return dict(d=d, stream=s, normal_calls=2*cfg["banks"],
                normal_scalars=cfg["banks"]*K*(d+1), seed_common=seed, seed_private=seed+1)


def execute(workspace):
    if not workspace:
        raise AssertionError("initial --write requires --workspace")
    audit = independent.Audit()
    audit.require(read(OUT/"STATE.json")["status"] == "complete", "HIGHK complete")
    freeze, seal = read(OUT/"FREEZE.json"), read(OUT/"RESULTS_FREEZE.json")
    audit.require(freeze["proposed_after_original_PROFILE_results"] is True, "separate follow-up timing")
    audit.require(seal["source_freeze_sha256"] == sha(OUT/"FREEZE.json"), "HIGHK seal chain")
    files = {}
    for mapping in (freeze["files"], freeze["reference_files"], seal["files"],
                    read(PROFILE/"RESULTS_FREEZE.json")["files"]):
        for name, digest in mapping.items():
            audit.require(name not in files or files[name] == digest, "hash-map agreement")
            files[name] = digest
    for path in (OUT/"FREEZE.json", OUT/"RESULTS_FREEZE.json", Path(__file__), HELPER, DOC/"HIGHK_AUDIT.md"):
        files[rel(path)] = sha(path)
    verify(files)
    verify(seal["local_arrays"])
    cfg, profile_cfg = read(OUT/"CONFIG.json"), read(PROFILE/"CONFIG.json")
    audit.require(cfg["dimensions"] == [16, 64] and cfg["Ks"] == [256, 4096]
                  and cfg["names"] == ["comp", "over"], "HIGHK declared axes")
    audit.require(cfg["banks"] == 64 and cfg["streams"] == 8, "HIGHK scope")
    audit.require(cfg["family"] == 4 and cfg["alpha"] == .05, "separate four-contrast family")
    moment_document = read(independent.MOM)
    audit.require(moment_document["qualified"] is True, "old moment qualification")
    moment_values = moment_document["gh"][-1]["values"]
    original_seeds = set()
    for d in profile_cfg["dimensions"]:
        for s in range(profile_cfg["streams"]):
            row = read(PROFILE/"streams"/f"d{d}_s{s}.json")
            original_seeds.update((row["seed_common"], row["seed_private"]))
    stream_means, risks_by_d, replay_rows, direct_rows = {}, {}, [], []
    seeds = []
    fields = ("banks", "unique_base_prior_vectors", "base_gaussian_scalars",
              "count_particle_candidate_terms", "logsumexp_terms")
    ledger = {key: 0 for key in fields}
    for d in cfg["dimensions"]:
        all_params, probability, risks = independent.verify_reference(d, profile_cfg, moment_values, audit)
        ids = [independent.NAMES.index(name) for name in cfg["names"]]
        params = [all_params[j] for j in ids]
        risks_by_d[d] = float(risks[ids[1]]-risks[ids[0]])
        psis = np.array([math.fsum((d//2)*float(moment_values[str(float(s))]["ell"])
                                   for s in param["scales"]) for param in params])
        audit.require(psis[0] == psis[1], "common floor control variate")
        K = max(cfg["Ks"])
        for s in range(cfg["streams"]):
            record = read(OUT/"streams"/f"d{d}_s{s}.json")
            audit.require((record["d"], record["stream"]) == (d, s)
                          and record["Ks"] == cfg["Ks"] and record["names"] == cfg["names"], "stream identity")
            audit.require(record["array_sha256"] == sha(ROOT/record["array"]), "HIGHK local hash")
            with np.load(ROOT/record["array"], allow_pickle=False) as archive:
                scores = archive["scores"]
                common, private = archive["common_prefix"], archive["private_prefix"]
            audit.require(scores.shape == (64, 2, 2) and np.isfinite(scores).all(), "all HIGHK score values")
            means = independent.stable_means(scores)
            audit.near(record["means"], means, "complete HIGHK bank means", category="stream_means")
            stream_means[d, s] = means
            audit.require(np.max(scores-psis[None, None]) <= 1e-12, "HIGHK Jensen direction")
            audit.require(common.shape == (1, K) and private.shape == (1, K, d), "full first-bank shape")
            replay_rows.append(replay_inputs(record, common, private, cfg, audit))
            seeds.extend((record["seed_common"], record["seed_private"]))
            corrected, raw, baseline = independent.direct_prefix_scores(
                common, private, params, probability, psis, cfg["Ks"])
            audit.near(scores[:1], corrected, "full-count first bank direct log-sigmoid",
                       category="direct_score")
            audit.near(corrected, psis[None, None]+raw-baseline,
                       "first-bank control-variate identity", category="direct_control_variate")
            expected = dict(banks=64, unique_base_prior_vectors=64*K,
                            base_gaussian_scalars=64*K*(d+1),
                            count_particle_candidate_terms=2*64*K*(d//2+1)**2,
                            logsumexp_terms=2*64*sum(cfg["Ks"])*(d//2+1)**2)
            for key, value in expected.items():
                audit.require(record[key] == value, "HIGHK stream ledger/" + key)
                ledger[key] += value
            audit.require(record["prefix_banks_saved"] == 1 and not record["all_prior_inputs_saved"],
                          "HIGHK saved-input scope")
            direct_rows.append(dict(d=d, stream=s, banks=1, candidates=2, Ks=cfg["Ks"],
                                    count_states=(d//2+1)**2,
                                    count_particle_candidate_terms=2*K*(d//2+1)**2,
                                    logsumexp_terms=2*sum(cfg["Ks"])*(d//2+1)**2))
            print(json.dumps(dict(audit="HIGHK full bank means and first-bank score", d=d, stream=s)), flush=True)
    audit.require(len(set(seeds)) == len(seeds) == 32, "unique HIGHK seeds")
    audit.require(not original_seeds.intersection(seeds), "HIGHK seeds distinct from PROFILE")
    contrasts = read(OUT/"CONTRASTS.json")
    c_map = {(row["d"], row["K"]): row for row in contrasts}
    audit.require(len(contrasts) == len(c_map) == 4, "four HIGHK contrasts")
    mains, precision = {}, 0
    for d in cfg["dimensions"]:
        means = np.array([stream_means[d, s] for s in range(cfg["streams"])])
        for ki, K in enumerate(cfg["Ks"]):
            expected = independent.interval(means[:, ki, 1]-means[:, ki, 0],
                                            cfg["alpha"], cfg["family"], cfg["SE_gate"])
            row = c_map[d, K]
            audit.require(row["comparison"] == "over_minus_comp", "HIGHK difference direction")
            for key in ("values", "mean", "SE", "lower", "upper"):
                audit.near(row[key], expected[key], "HIGHK interval/" + key, category="interval")
            audit.require(row["precision_pass"] == expected["precision_pass"], "HIGHK precision gate")
            audit.require(row["t_is_approximation"] is True, "HIGHK approximate t")
            audit.near(row["reference_risk_difference"], risks_by_d[d], "HIGHK reference risk",
                       category="reference_risk")
            precision += int(expected["precision_pass"])
            if K == 4096:
                mains[d] = dict(
                    passed=bool(expected["precision_pass"] and expected["lower"] > cfg["score_margin"]
                                and risks_by_d[d] > cfg["risk_margin"]),
                    score_lower=expected["lower"], risk_difference=risks_by_d[d])
    decision = read(OUT/"DECISION.json")
    audit.require(len(decision["primary"]) == 2, "two HIGHK main readouts")
    for row in decision["primary"]:
        audit.require(row["K"] == 4096 and row["passed"] == mains[row["d"]]["passed"], "HIGHK primary gate")
        for key in ("score_lower", "risk_difference"):
            audit.near(row[key], mains[row["d"]][key], "HIGHK primary/" + key, category="interval")
    audit.require(decision["precision_pass"] == precision and decision["total"] == 4, "HIGHK precision total")
    audit.require(decision["independent_followup_not_original_PROFILE"] is True
                  and decision["all_failures_preserved"] is True
                  and decision["global_minimum_claim"] is False, "HIGHK interpretation")
    ledger.update(independent_streams=16, new_teacher_labels=0, new_fits=0, new_integrals=0, new_neural_models=0)
    audit.require(read(OUT/"COUNTS.json") == ledger, "HIGHK complete ledger")
    return dict(
        schema=SCHEMA, passed=True, checks=audit.checks, maximum_absolute_differences=audit.maximum,
        producer_results_freeze_sha256=sha(OUT/"RESULTS_FREEZE.json"),
        source_files=files, local_files=seal["local_arrays"],
        independent_helper=dict(path=rel(HELPER), sha256=HELPER_SHA, producer_calculations_imported=False),
        full_array_recalculation=dict(streams=16, banks=1024, score_cells=1024*2*2,
                                      contrasts=4, main_readouts=2),
        bounded_direct_replay=direct_rows, full_RNG_consumption_replay=replay_rows,
        audit_RNG_replay_calls=sum(r["normal_calls"] for r in replay_rows),
        audit_RNG_replayed_normal_scalars=sum(r["normal_scalars"] for r in replay_rows),
        audit_new_scientific_random_draws=0, audit_new_labels=0, audit_new_fits=0, audit_new_integrals=0,
        all_original_likelihood_terms_replayed=False,
        uncontrolled_and_baseline_not_saved_for_all_banks=True,
        scientific_ledger_verified=ledger,
        verified_outcomes=dict(precision_pass=precision, total=4,
                               primary_pass=sum(row["passed"] for row in mains.values()), primary_total=2,
                               per_dimension=mains))


def check(workspace):
    receipt = read(RECEIPT)
    if receipt["schema"] != SCHEMA or receipt["passed"] is not True:
        raise AssertionError("HIGHK audit has not passed")
    verify(receipt["source_files"])
    if workspace:
        verify(receipt["local_files"])
    if sha(OUT/"RESULTS_FREEZE.json") != receipt["producer_results_freeze_sha256"]:
        raise AssertionError("HIGHK result seal changed")
    print(json.dumps(dict(passed=True, workspace=workspace, receipt_sha256=sha(RECEIPT),
                          arithmetic_and_RNG_not_reexecuted=True)))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    args = parser.parse_args()
    if not args.write:
        check(args.workspace)
        return
    if RECEIPT.exists():
        raise AssertionError("audit receipt exists; no overwrite")
    if read(OUT/"STATE.json")["status"] != "complete":
        raise AssertionError("HIGHK must be complete before audit")
    started = time.perf_counter()
    try:
        receipt = execute(args.workspace)
        receipt["utc"] = datetime.now(timezone.utc).isoformat()
        receipt["elapsed_seconds"] = time.perf_counter()-started
        independent.write_new(RECEIPT, receipt)
        print(json.dumps(dict(passed=True, checks=receipt["checks"], receipt_sha256=sha(RECEIPT))), flush=True)
    except Exception as exc:
        independent.write_new(RECEIPT, dict(schema=SCHEMA, passed=False,
                                           utc=datetime.now(timezone.utc).isoformat(),
                                           audit_source_sha256=sha(Path(__file__)), error=repr(exc),
                                           traceback=traceback.format_exc(), producer_files_unchanged=True))
        raise


if __name__ == "__main__":
    main()
