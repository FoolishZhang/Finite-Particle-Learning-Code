"""Fresh frozen phases for one bounded learned-proposal precision study.

The development phase reuses saved samples/local weights and candidate banks.
It never calls the sampler or candidate kernel. Production phases have new,
fixed independent numerical streams. No science runs on import, --template or
--check; historical runners' validation/freeze entry points are never used.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
import scipy

import learned_local_run as old
from learned_local_proposal import LocalProposal
import learned_precision_core as core

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = "artifacts/learned_precision_v1_20261016"
OLD_ARTIFACT = "artifacts/learned_local_proposal_v1_20261016"
SCHEMA = "learned_precision_execution_v1"
BASELINE = "1e9ca3374665194c11d2d4a85ffcf945718f24d4"
HANDOFF_SHA = "43fe4a37bf2cfad61f0c584330c4d99c1ae27101bd67a260719867c04366fbe2"
SIZES, AMPLITUDES = (16384, 65536), (.1, .2)
PHASES = ("development", "coefficients", "validation")
P, R, K = 18**3, 16, 8
BASE_SOURCES = set(old.BASE_SOURCES) | {"script/learned_precision_core.py", "script/learned_precision_run.py"}
write, npz, sha, read, serial = old.write, old.npz, old.sha, old.read, old.serial
require = old.require


def resolve(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "safe repository-relative path required")
    target = (REPO/path).resolve()
    require(target.is_relative_to(REPO.resolve()), "bound path escapes repository")
    return target


def relative(path):
    return Path(path).resolve().relative_to(REPO.resolve()).as_posix()


def key_t(t):
    return "t"+str(float(t)).replace(".", "p")


def coefficient_record(folder, **extra):
    return {**extra, "statistics": relative(folder/"COEFFICIENT_STATISTICS.json"),
            "statistics_sha256": sha(folder/"COEFFICIENT_STATISTICS.json"),
            "banks": relative(folder/"COEFFICIENT_BANKS.npz"), "banks_sha256": sha(folder/"COEFFICIENT_BANKS.npz")}


def counter_limits(phase, rep):
    """Exact nominal requested work for the fixed two-proposal shard schedule."""
    require(phase in PHASES and rep in range(3), "declared phase/rep required")
    counters = {"seconds": 7200, "reference_score_entries": 3*P,
                "anchor_calls": 1, "anchor_particle_pattern_terms": 1,
                "q2_control_matrix_pattern_terms": 1}
    if phase == "development":
        counters.update(anchor_calls=4, anchor_particle_pattern_terms=2*R*(256+512)*K*P,
                        q2_control_matrix_pattern_terms=2*6*R*(256+512)*P*3,
                        saved_bank_loads=4, saved_difference_reuses=4,
                        saved_difference_bank_values=4*R*512)
    else:
        B = 256 if phase == "coefficients" else 512
        counters.update(qmc_uniform_scalars=2*R*B*K*6, qmc_nets=2*R,
                        posterior_coordinates=2*R*B*K*3*18,
                        proposal_density_values=2*R*B*K*3*18,
                        local_likelihood_x=2*R*B*K*3*18,
                        local_private_node_terms=2*R*B*K*3*18*256,
                        anchor_calls=2, anchor_particle_pattern_terms=2*R*B*K*P,
                        q2_control_matrix_pattern_terms=2*6*R*B*P*3)
        if phase == "validation":
            additional = 4 if rep == 0 else 0
            counters.update(candidate_calls=8+additional,
                            candidate_particle_pattern_terms=(8+additional)*R*B*K*P)
            if rep == 0:
                counters["reference_score_entries"] += 3*P
                counters["anchor_calls"] += 1
                counters["anchor_particle_pattern_terms"] += R*B*K*P
                counters["q2_control_matrix_pattern_terms"] += 6*R*B*P*3
    return counters


def bound(cfg, name):
    require(name in cfg["input_sha256"], "required input not bound: "+name)
    require(sha(resolve(name)) == cfg["input_sha256"][name], "bound input changed: "+name)
    return resolve(name)


def bound_record(cfg, record):
    for kind in ("json", "npz"):
        require(cfg["input_sha256"].get(record[kind]) == record[kind+"_sha256"], "QA bundle not bound: "+record[kind])
    return record


def load_bundle(cfg, record):
    bound_record(cfg, record)
    value = read(resolve(record["json"]))
    with np.load(resolve(record["npz"]), allow_pickle=False) as arrays:
        value.update({name: arrays[name] for name in arrays.files})
    return value


def frozen_pairs(cfg):
    frozen = read(bound(cfg, cfg["proposal_freeze"]))
    require(frozen.get("all_training_complete") is True and len(frozen.get("pairs", [])) == 6,
            "all six original trained pairs required")
    require({(p["rep"], p["N"]) for p in frozen["pairs"]} == {(r, n) for r in range(3) for n in SIZES},
            "fixed three repeats/two nested training sizes required")
    for pair in frozen["pairs"]:
        record = pair["proposals"]["sym"]
        require(cfg["input_sha256"].get(record["path"]) == record["sha256"], "frozen sym proposal not bound")
        proposal = read(bound(cfg, record["path"]))
        require(proposal["kind"] == "sym" and pair["defense"] == proposal["defense"] == .05 and
                proposal["raw_parameters"] == pair["raw_parameters"] and proposal["shared_homogeneous_groups"] == 3,
                "same frozen sym local proposal required")
        expected = LocalProposal(pair["raw_parameters"], kind="sym", defense=.05).to_record()
        require(proposal["parameters"] == expected["parameters"], "frozen sym construction differs")
    return frozen["pairs"]


def old_bank_paths(root, include_contrasts):
    names = ["COMPLETE.json", "BANK/COMPLETE.json", "BANK/RNG.json", "BANK/UNIFORMS.npz",
             "BANK/SAMPLES.npz", "BANK/LOCAL_WEIGHTS.npz", "BANK/ANALYTIC_PROPOSAL_MOMENTS.json"]
    if include_contrasts:
        names += [f"CONTRAST_{key_t(t)}.{extension}" for t in AMPLITUDES for extension in ("json", "npz")]
    return [(Path(root)/name).as_posix() for name in names]


def validate_old_bank(cfg, pair, root, phase):
    old_run = Path(root).parent.as_posix()
    B = 256 if phase == "coefficients" else 512
    family, role = (202610165301, 30) if phase == "coefficients" else (202610165401, 40)
    expected_root = f"{OLD_ARTIFACT}/{phase}_rep{pair['rep']}_001/rep_{pair['rep']}_N{pair['N']}_sym"
    require(root == expected_root, "development must use the declared original sym bank")
    contract = read(bound(cfg, old_run+"/CONTRACT.json"))
    freeze = read(bound(cfg, old_run+"/FREEZE.json"))
    manifest = read(bound(cfg, old_run+"/MANIFEST.json"))
    require(contract["phase"] == phase and contract["qa_index"] == cfg["qa_index"] and
            contract["proposal_freeze"] == cfg["proposal_freeze"], "old bank teacher/proposal lineage differs")
    require(freeze["contract_sha256"] == cfg["input_sha256"][old_run+"/CONTRACT.json"], "old FREEZE contract identity differs")
    manifest_files = {(Path(old_run)/r["path"]).as_posix(): r["sha256"] for r in manifest["files"]}
    for name in old_bank_paths(root, phase == "validation"):
        bound(cfg, name)
        require(manifest_files.get(name) == cfg["input_sha256"][name], "old bank member differs from sealed manifest")
    complete = read(resolve(root+"/BANK/COMPLETE.json"))
    proposal_record = read(resolve(pair["proposals"]["sym"]["path"]))
    for key in ("kind", "defense", "raw_parameters", "parameters", "shared_homogeneous_groups"):
        require(complete["proposal"][key] == proposal_record[key], "actual old bank proposal differs: "+key)
    source = complete["proposal"]["source"]
    require((source["rep"], source["N"], source["selected_epoch"]) == (pair["rep"], pair["N"], pair["selected_epoch"]), "old bank training source differs")
    rng = read(resolve(root+"/BANK/RNG.json"))
    require((rng["family"], rng["role"], rng["R"], rng["B"], rng["K"], rng["kind"]) == (family, role, R, B, K, "sym"), "old bank randomization role differs")
    require(len(rng["states"]) == R, "all old randomization records required")
    for r, record in enumerate(rng["states"]):
        require(record["randomization"] == r and record["entropy"] == [family, role, pair["rep"], pair["N"], 1, r], "old bank entropy differs")
    moments = read(resolve(root+"/BANK/ANALYTIC_PROPOSAL_MOMENTS.json"))
    actual = LocalProposal(pair["raw_parameters"], kind="sym", defense=.05).moments()
    for name in ("mean", "second_moment"):
        require(np.array_equal(moments[name], actual[name]), "old stored moments differ from actual proposal")
    if phase == "validation":
        for t in AMPLITUDES:
            contrast = read(resolve(root+f"/CONTRAST_{key_t(t)}.json"))
            require(contrast["t"] == t and contrast["alpha"] == core.ALPHA and contrast["pairing"] ==
                    "same_q_particles_and_same_teacher_PMF; shrink_candidate_minus_teacher_candidate", "saved baseline difference target differs")
            require(np.array_equal(contrast["candidate_coordinates"], [(core.ALPHA*t*core.V).tolist(), (t*core.V).tolist()]), "saved baseline candidates differ")
            for name in ("statistics", "raw_statistics"):
                require(contrast[name]["R"] == R and contrast[name]["B"] == B, "saved candidate R/B differs")


def validate_prediction_freeze(cfg):
    frozen = read(bound(cfg, cfg["prediction_freeze"]))
    require(frozen.get("schema") == "learned_precision_prediction_freeze_v1" and
            frozen.get("all_coefficients_complete") is True and frozen.get("all_predictions_frozen") is True and
            frozen.get("selected_estimator") == "q2" and frozen.get("before_validation_randomization") is True,
            "complete pre-validation q2 coefficient/prediction freeze required")
    require(frozen.get("development_decision") == cfg["development_decision"] and
            frozen.get("development_decision_sha256") == cfg["input_sha256"][cfg["development_decision"]],
            "prediction freeze uses a different development decision")
    require(cfg["input_sha256"].get(frozen.get("policy")) == frozen.get("policy_sha256") and
            isinstance(frozen.get("policy_sha256"), str), "prediction policy must be an actual bound input")
    outputs = frozen["coefficient_outputs"]
    require(len(outputs) == 6 and {(p["rep"], p["N"], p["kind"]) for p in outputs} ==
            {(r, n, "sym") for r in range(3) for n in SIZES}, "all six production coefficients required")
    identities = {}
    for record in outputs:
        for key in ("statistics", "banks"):
            require(cfg["input_sha256"].get(record[key]) == record[key+"_sha256"], "new coefficient input not bound")
        doc = read(bound(cfg, record["statistics"]))
        require((doc["rep"], doc["N"], doc["kind"]) == (record["rep"], record["N"], "sym") and
                doc["phase"] == "coefficients" and doc["outer_order"] == 36 and doc["inner_order"] == 256 and
                doc["reference_derivative"] == "same_finite_GH_Cholesky_chain_rule", "prediction coefficient has wrong teacher scalar/phase")
        identities[record["rep"], record["N"]] = record, doc
    predictions = frozen["predictions"]
    require(len(predictions) == 12 and {(p["rep"], p["N"], p["kind"], p["t"]) for p in predictions} ==
            {(r, n, "sym", t) for r in range(3) for n in SIZES for t in AMPLITUDES}, "all twelve fixed predictions required")
    for prediction in predictions:
        require(prediction["alpha"] == core.ALPHA and np.array_equal(prediction["direction"], core.V), "prediction alpha/direction differs")
        require(prediction.get("component_scaling") == "coefficients_before_t_scaling", "prediction components must be unscaled coefficients")
        record, doc = identities[prediction["rep"], prediction["N"]]
        require(prediction["coefficient_statistics"] == record["statistics"] and prediction["coefficient_statistics_sha256"] == record["statistics_sha256"], "prediction coefficient identity differs")
        for key in ("bank_id", "reference_identity", "reference_p0_sha256", "reference_dp0_sha256"):
            require(prediction[key] == doc[key], "prediction reference/bank identity differs: "+key)
        for key in ("replicates", "linear_replicates", "H_quadratic_replicates", "B_quadratic_replicates"):
            arr = np.asarray(prediction[key], float)
            require(arr.shape == (R,) and np.isfinite(arr).all(), "sixteen finite independent prediction components required")
        t = prediction["t"]
        total = t*np.asarray(prediction["linear_replicates"])+t*t*(np.asarray(prediction["H_quadratic_replicates"])+np.asarray(prediction["B_quadratic_replicates"]))
        require(np.allclose(total, prediction["replicates"], rtol=2e-11, atol=2e-13), "frozen prediction component sum differs")
    return frozen


def validate(cfg):
    require(cfg["schema"] == SCHEMA and cfg["phase"] in PHASES and cfg.get("frozen") is True and
            cfg.get("authorized_to_execute") is True, "explicit frozen authorized precision phase required")
    require(cfg["baseline_commit"] == BASELINE and cfg["handoff_sha256"] == HANDOFF_SHA and
            sha(REPO/"HANDOFF.md") == HANDOFF_SHA, "baseline/feedback identity differs")
    require(cfg["case_ids"] in ([0], [1], [2]) and cfg["training_N"] == list(SIZES) and cfg["proposal_kind"] == "sym", "one original repeat/two fixed sym proposals per shard")
    rep, phase = cfg["case_ids"][0], cfg["phase"]
    require(BASE_SOURCES <= set(cfg["source_sha256"]), "complete new and preserved scientific source closure required")
    for field in ("source_sha256", "input_sha256"):
        for name, digest in cfg[field].items():
            require(sha(resolve(name)) == digest, "bound file differs: "+name)
    runtime = cfg["runtime"]
    require((runtime["python"], runtime["numpy"], runtime["scipy"]) == (sys.version.split()[0], np.__version__, scipy.__version__), "runtime changed")
    require(all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")), "single-thread numeric libraries required")
    require(cfg["caps"] == counter_limits(phase, rep), "exact fixed nominal work/7200-second caps required")
    require(cfg["resources"] == {"max_rss_bytes": 3*1024**3, "max_output_bytes": 4*1024**3} and cfg["external_timeout_seconds"] == 7260, "fixed resources/outer timeout declaration required")
    require(cfg["K"] == K and cfg["alpha"] == core.ALPHA and cfg["t_values"] == list(AMPLITUDES) and
            np.array_equal(cfg["direction"], core.V), "fixed original K/alpha/t/direction required")
    require(isinstance(cfg["pattern_chunk"], int) and not isinstance(cfg["pattern_chunk"], bool) and 1 <= cfg["pattern_chunk"] <= 64, "bounded pattern_chunk required")
    require(cfg["teacher_policy"] == "same_saved_GH36_Cholesky_p0_dp0_and_pt", "one declared finite teacher scalar required")
    qa = read(bound(cfg, cfg["qa_index"]))
    require(qa.get("passed") is True, "saved QA must have passed")
    for t in (0., *AMPLITUDES):
        for order in (28, 36):
            record = bound_record(cfg, qa["teachers"][str(t)][str(order)])
            metadata = read(resolve(record["json"]))
            require(metadata["outer_order"] == order and metadata["inner_order"] == 256 and metadata["gradient_method"] == "same_finite_GH_Cholesky_chain_rule", "QA law is not the declared finite Cholesky family")
    pairs = frozen_pairs(cfg)
    if phase == "development":
        require(cfg["randomization"] is None and cfg.get("development_decision") is None and cfg.get("prediction_freeze") is None, "development must not use new scientific streams or future decisions")
        old_banks = cfg["old_banks"]
        require(len(old_banks) == 2 and {(p["rep"], p["N"]) for p in old_banks} == {(rep, n) for n in SIZES}, "both original sym bank pairs required")
        by_size = {p["N"]: p for p in pairs if p["rep"] == rep}
        for record in old_banks:
            validate_old_bank(cfg, by_size[record["N"]], record["coefficient_root"], "coefficients")
            validate_old_bank(cfg, by_size[record["N"]], record["validation_root"], "validation")
    else:
        require(not cfg.get("old_banks"), "production must not substitute old development banks")
        gate = read(bound(cfg, cfg["development_decision"]))
        require(gate.get("passed") is True and gate.get("selected_estimator") == "q2" and gate.get("all_six_cases_complete") is True, "fixed all-six development gate must pass before production")
        family, role, B = (202610166301, 60, 256) if phase == "coefficients" else (202610166401, 70, 512)
        require(cfg["randomization"] == {"family": family, "role": role, "R": R, "B": B, "bits": 52}, "fixed independent production role/R/B/bits required")
        if phase == "coefficients":
            require(cfg.get("prediction_freeze") is None, "coefficients must precede frozen predictions")
        else:
            validate_prediction_freeze(cfg)
            rule = read(bound(cfg, cfg["rule_policy"]))
            require(rule.get("frozen") is True, "frozen rule qualification policy required")
            require(cfg["rule_check"] == {"rep": 0, "N": 16384, "low_order": 28, "high_order": 36,
                    "max_replicate_normalized_D_difference": 1e-4, "max_replicate_W_same_difference": 1e-4}, "fixed complete-bank GH rule qualification required")
    return cfg


def freeze(path, out):
    cfg = validate(read(path))
    out = Path(out).resolve()
    require(out.is_relative_to(resolve(ARTIFACT)), "new precision phase directory required")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, out/"CONTRACT.json")
    for name, digest in cfg["source_sha256"].items():
        dest = out/"SEALED_SOURCE"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(name), dest)
        require(sha(dest) == digest, "source changed during freeze")
    write(out/"FREEZE.json", {"schema": SCHEMA, "phase": cfg["phase"], "contract_sha256": sha(path),
        "source_sha256": cfg["source_sha256"], "input_sha256": cfg["input_sha256"],
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "before_scientific_calls": True})
    return cfg, old.Budget(cfg, out)


def references(cfg, budget, out, order):
    qa = read(resolve(cfg["qa_index"]))
    origin_record = qa["teachers"]["0.0"][str(order)]
    origin_law = load_bundle(cfg, origin_record)
    budget.add("reference_score_entries", 3*P)
    identity = origin_record["npz"]+"#"+origin_record["npz_sha256"]
    ref = core.teacher_reference_from_law(origin_law, identity=identity)
    teachers, bindings = {}, {"origin": origin_record}
    for t in AMPLITUDES:
        record = qa["teachers"][str(t)][str(order)]
        teachers[key_t(t)] = np.asarray(load_bundle(cfg, record)["joint_pmf"], float).ravel()
        bindings[key_t(t)] = record
    payload = {**ref, **{key: value for key, value in teachers.items()}, "source_laws": bindings}
    old.bundle(out/f"REFERENCE_GH{order}", payload)
    return ref, teachers, bindings


def load_old_bank(cfg, pair, root, B, budget, out):
    budget.add("saved_bank_loads", 1)
    with np.load(resolve(root+"/BANK/SAMPLES.npz"), allow_pickle=False) as data:
        x = data["x"]
    with np.load(resolve(root+"/BANK/LOCAL_WEIGHTS.npz"), allow_pickle=False) as data:
        local = data["local"]
    require(x.shape == local.shape == (R, B, K, 3, 18) and np.isfinite(x).all() and np.isfinite(local).all(), "complete finite original bank required")
    moment_path = root+"/BANK/ANALYTIC_PROPOSAL_MOMENTS.json"
    moment = read(resolve(moment_path))
    qm, qs = (np.tile(np.asarray(moment[name], float), (3, 1)) for name in ("mean", "second_moment"))
    identity = {name: {"path": root+"/BANK/"+filename, "sha256": cfg["input_sha256"][root+"/BANK/"+filename]}
                for name, filename in (("samples", "SAMPLES.npz"), ("local", "LOCAL_WEIGHTS.npz"), ("moments", "ANALYTIC_PROPOSAL_MOMENTS.json"), ("complete", "COMPLETE.json"), ("rng", "RNG.json"))}
    identity["proposal"] = pair["proposals"]["sym"]
    bank_id = "saved:"+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write(out/"BANK_INPUT_IDENTITY.json", {"bank_id": bank_id, "reused_old_bank": True, "inputs": identity, "new_RNG": 0})
    x.setflags(write=False); local.setflags(write=False)
    return x, local, qm, qs, bank_id


def new_bank(cfg, pair, budget, out):
    x, local, qm, qs = old.proposal_bank(cfg, budget, pair, "sym", out/"BANK")
    identity = {name: {"path": relative(out/"BANK"/filename), "sha256": sha(out/"BANK"/filename)}
                for name, filename in (("samples", "SAMPLES.npz"), ("local", "LOCAL_WEIGHTS.npz"), ("moments", "ANALYTIC_PROPOSAL_MOMENTS.json"), ("complete", "COMPLETE.json"), ("rng", "RNG.json"))}
    identity["proposal"] = pair["proposals"]["sym"]
    bank_id = "new:"+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    write(out/"BANK_INPUT_IDENTITY.json", {"bank_id": bank_id, "reused_old_bank": False, "inputs": identity})
    return x, local, qm, qs, bank_id


def save_coefficients(folder, result, pair, cfg, bindings):
    arrays = {**result["banks"], "C_origin": result["C_origin"], **{"C_"+key: value for key, value in result["C_teachers"].items()}}
    npz(folder/"COEFFICIENT_BANKS.npz", arrays)
    metadata = {key: value for key, value in result.items() if key not in ("banks", "C_origin", "C_teachers")}
    metadata.update(rep=pair["rep"], N=pair["N"], kind="sym", phase=cfg["phase"], reference_bindings=bindings,
        symmetry_zero_scope="mirror-state sampling and complete enumeration impose reflection cancellations; no permutation projection",
        numerical_scalar="same finite GH Cholesky teacher probabilities and their own origin derivative")
    write(folder/"COEFFICIENT_STATISTICS.json", metadata)
    write(folder/"PREFIX_COMPARISON.json", {key: old.paired_prefix(value) for key, value in arrays.items()})
    return coefficient_record(folder, rep=pair["rep"], N=pair["N"], kind="sym", bank_id=result["bank_id"],
                              reference_identity=result["reference_identity"], outer_order=result["outer_order"])


def save_contrast(folder, candidate, remainder):
    tag = key_t(candidate["t"])
    npz(folder/f"CONTRAST_{tag}.npz", candidate["banks"])
    metadata = {key: value for key, value in candidate.items() if key != "banks"}
    metadata["prefix_comparison"] = {key: old.paired_prefix(value) for key, value in candidate["banks"].items()}
    write(folder/f"CONTRAST_{tag}.json", metadata)
    npz(folder/f"REMAINDER_{tag}.npz", remainder["banks"])
    rmeta = {key: value for key, value in remainder.items() if key != "banks"}
    rmeta["prefix_comparison"] = {key: old.paired_prefix(value) for key, value in remainder["banks"].items()}
    write(folder/f"REMAINDER_{tag}.json", rmeta)
    result = {"t": candidate["t"]}
    for key, basename in (("statistics", f"CONTRAST_{tag}.json"), ("banks", f"CONTRAST_{tag}.npz"),
                          ("remainder_statistics", f"REMAINDER_{tag}.json"), ("remainder_banks", f"REMAINDER_{tag}.npz")):
        result[key], result[key+"_sha256"] = relative(folder/basename), sha(folder/basename)
    return result


def independent_prediction(cfg, pair, t, candidate, folder):
    frozen = read(resolve(cfg["prediction_freeze"]))
    rows = [p for p in frozen["predictions"] if (p["rep"], p["N"], p["kind"], p["t"]) == (pair["rep"], pair["N"], "sym", t)]
    require(len(rows) == 1, "one pre-frozen independent prediction required")
    source = rows[0]
    predicted = np.asarray(source["replicates"], float)
    D = candidate["statistics"]["q2_D"]["replicates"]
    remainder = D-predicted
    def stat(values):
        return {"replicates": values, "mean": float(values.mean()), "SE": float(values.std(ddof=1)/np.sqrt(R)), "R": R}
    payload = {"rep": pair["rep"], "N": pair["N"], "kind": "sym", "t": t,
        "prediction_freeze": cfg["prediction_freeze"], "prediction_freeze_sha256": cfg["input_sha256"][cfg["prediction_freeze"]],
        "frozen_prediction": source, "statistics": {"P_ind": stat(predicted), "D_q2": stat(D),
            "remainder_ind": stat(remainder), "W_ind": stat(remainder/(t*t))},
        "pairing": "independent coefficient and validation randomizations matched by index0..15",
        "scope": "independent frozen prediction; distinct from same_bank_remainder"}
    path = folder/f"INDEPENDENT_{key_t(t)}.json"
    write(path, payload)
    return {"independent_statistics": relative(path), "independent_statistics_sha256": sha(path)}


def development(cfg, budget, out, refs):
    ref, teachers, bindings = refs
    pairs = {p["N"]: p for p in frozen_pairs(cfg) if p["rep"] == cfg["case_ids"][0]}
    outputs = []
    for record in cfg["old_banks"]:
        pair = pairs[record["N"]]
        folder = out/f"rep_{pair['rep']}_N{pair['N']}_sym"
        folder.mkdir()
        item = {"rep": pair["rep"], "N": pair["N"], "kind": "sym"}
        for category, B in (("coefficient", 256), ("validation", 512)):
            dest = folder/(category.upper()+"_BANK")
            dest.mkdir()
            x, local, qm, qs, bank_id = load_old_bank(cfg, pair, record[category+"_root"], B, budget, dest)
            coef = core.origin_coefficients(x, local, ref, qm, qs, teacher_weights=teachers, bank_id=bank_id, pattern_chunk=cfg["pattern_chunk"], budget=budget)
            index = save_coefficients(dest, coef, pair, cfg, bindings)
            if category == "coefficient":
                item["coefficient_bank"] = index
            else:
                contrasts = []
                for t in AMPLITUDES:
                    previous = record["validation_root"]+f"/CONTRAST_{key_t(t)}.npz"
                    with np.load(resolve(previous), allow_pickle=False) as data:
                        D, raw_D = data["banks"], data["raw_banks"]
                    require(D.shape == raw_D.shape == (R, B), "full original candidate banks required")
                    budget.add("saved_difference_reuses", 1)
                    budget.add("saved_difference_bank_values", D.size)
                    candidate = core.apply_candidate_control(D, coef, teacher_key=key_t(t), teacher_weights=teachers[key_t(t)], t=t, alpha=cfg["alpha"], direction=cfg["direction"], bank_id=bank_id)
                    candidate["banks"]["raw_D"] = raw_D
                    candidate["statistics"]["raw_D"] = core.stats(raw_D)
                    candidate["source_saved_candidate"] = {"path": previous, "sha256": cfg["input_sha256"][previous], "candidate_kernels_reexecuted": 0}
                    remainder = core.same_bank_remainder(candidate, coef)
                    contrasts.append(save_contrast(dest, candidate, remainder))
                item["validation_bank"] = {"coefficient": index, "contrasts": contrasts}
            budget.check()
        write(folder/"COMPLETE.json", item)
        outputs.append(item)
        print(json.dumps({"phase": "development", "rep": pair["rep"], "N": pair["N"], "status": "completed"}), flush=True)
    return {"outputs": outputs, "new_RNG": 0, "new_candidate_kernel_calls": 0, "scope": "posthoc_saved_bank_method_development_not_confirmation"}


def rule_comparison(folder, primary, alternate, cfg):
    rows, arrays = [], {}
    for t in AMPLITUDES:
        high_D, high_W = primary[t][0]["banks"]["q2_D"], primary[t][1]["banks"]["q2_normalized_remainder"]
        low_D, low_W = alternate[t][0]["banks"]["q2_D"], alternate[t][1]["banks"]["q2_normalized_remainder"]
        d, w = (high_D-low_D)/(t*t), high_W-low_W
        tag = key_t(t)
        arrays[tag+"_D_normalized_rule_difference"] = d
        arrays[tag+"_W_same_rule_difference"] = w
        ds, ws = core.stats(d), core.stats(w)
        max_D = float(np.max(abs(ds["replicates"])))
        max_W = float(np.max(abs(ws["replicates"])))
        rows.append({"t": t, "D_normalized_rule_difference": ds, "W_same_rule_difference": ws,
            "max_abs_complete_replicate_D_normalized_difference": max_D,
            "max_abs_complete_replicate_W_same_difference": max_W,
            "qualified": bool(max_D <= 1e-4 and max_W <= 1e-4), "threshold": 1e-4})
    npz(folder/"RULE_ORDER_COMPARISON.npz", arrays)
    write(folder/"RULE_ORDER_COMPARISON.json", {"rows": rows, "all_qualified": all(row["qualified"] for row in rows),
        "same_samples_but_each_teacher_rule_has_its_own_origin_and_derivative": True,
        "gate": "max over16 complete replicate differences, not mean-only or bank maximum",
        "rule_policy": cfg["rule_policy"], "rule_policy_sha256": cfg["input_sha256"][cfg["rule_policy"]],
        "ordinary_rule_comparison_not_integral_certificate": True, "output_inclusion_rule": False})
    return {"statistics": relative(folder/"RULE_ORDER_COMPARISON.json"), "statistics_sha256": sha(folder/"RULE_ORDER_COMPARISON.json"),
            "banks": relative(folder/"RULE_ORDER_COMPARISON.npz"), "banks_sha256": sha(folder/"RULE_ORDER_COMPARISON.npz")}


def production(cfg, budget, out, refs, refs28=None):
    ref, teachers, bindings = refs
    outputs = []
    for pair in frozen_pairs(cfg):
        if pair["rep"] != cfg["case_ids"][0]:
            continue
        folder = out/f"rep_{pair['rep']}_N{pair['N']}_sym"
        folder.mkdir()
        x, local, qm, qs, bank_id = new_bank(cfg, pair, budget, folder)
        coef = core.origin_coefficients(x, local, ref, qm, qs, teacher_weights=teachers, bank_id=bank_id, pattern_chunk=cfg["pattern_chunk"], budget=budget)
        index = save_coefficients(folder, coef, pair, cfg, bindings)
        item = {"rep": pair["rep"], "N": pair["N"], "kind": "sym", "coefficient": index}
        if cfg["phase"] == "validation":
            primary, contrasts = {}, []
            for t in AMPLITUDES:
                candidate = core.candidate_difference(x, local, ref, qm, qs, coef, teacher_key=key_t(t), teacher_weights=teachers[key_t(t)],
                    t=t, alpha=cfg["alpha"], direction=cfg["direction"], bank_id=bank_id, pattern_chunk=cfg["pattern_chunk"], budget=budget)
                remainder = core.same_bank_remainder(candidate, coef)
                primary[t] = candidate, remainder
                row = save_contrast(folder, candidate, remainder)
                row.update(independent_prediction(cfg, pair, t, candidate, folder))
                contrasts.append(row)
            item["contrasts"] = contrasts
            if pair["rep"] == 0 and pair["N"] == NLOW:
                lowref, lowteachers, lowbindings = refs28
                lowfolder = folder/"GH28"
                lowfolder.mkdir()
                lowcoef = core.origin_coefficients(x, local, lowref, qm, qs, teacher_weights=lowteachers, bank_id=bank_id, pattern_chunk=cfg["pattern_chunk"], budget=budget)
                lowindex = save_coefficients(lowfolder, lowcoef, pair, cfg, lowbindings)
                alternate, lowcontrasts = {}, []
                for t in AMPLITUDES:
                    candidate = core.candidate_difference(x, local, lowref, qm, qs, lowcoef, teacher_key=key_t(t), teacher_weights=lowteachers[key_t(t)],
                        t=t, alpha=cfg["alpha"], direction=cfg["direction"], bank_id=bank_id, pattern_chunk=cfg["pattern_chunk"], budget=budget)
                    remainder = core.same_bank_remainder(candidate, lowcoef)
                    alternate[t] = candidate, remainder
                    lowcontrasts.append(save_contrast(lowfolder, candidate, remainder))
                item["GH28_check"] = {"coefficient": lowindex, "contrasts": lowcontrasts, "comparison": rule_comparison(folder, primary, alternate, cfg)}
        write(folder/"COMPLETE.json", item)
        outputs.append(item)
        budget.check()
        print(json.dumps({"phase": cfg["phase"], "rep": pair["rep"], "N": pair["N"], "status": "completed"}), flush=True)
    require(len(outputs) == 2, "complete two-proposal shard required")
    if cfg["phase"] == "coefficients":
        write(out/"FROZEN_COEFFICIENT_OUTPUTS.json", {"all_shard_outputs_complete": True, "not_global_freeze": True,
            "before_validation_randomization": True, "outputs": [row["coefficient"] for row in outputs]})
    return {"outputs": outputs}


NLOW = SIZES[0]


def run(contract_path, out):
    cfg, budget = freeze(contract_path, out)
    try:
        refs = references(cfg, budget, out, 36)
        low = references(cfg, budget, out, 28) if cfg["phase"] == "validation" and cfg["case_ids"] == [0] else None
        result = development(cfg, budget, out, refs) if cfg["phase"] == "development" else production(cfg, budget, out, refs, low)
        report = {"schema": SCHEMA, "phase": cfg["phase"], "status": "completed", **result,
            "budget": budget.snapshot(), "no_new_training_or_observations": True, "automatic_followup": False,
            "count_scope": "nominal requested kernel entries; not CPU operations; read/hash/serialization and arithmetic summaries separate"}
        write(out/"SUMMARY.json", report)
        old.manifest(out)
        return report
    except Exception:
        write(out/"FAILED.json", {"schema": SCHEMA, "phase": cfg["phase"], "traceback": traceback.format_exc(),
            "budget": budget.snapshot(), "automatic_retry": False, "existing_random_inputs_and_partial_outputs_retained": True})
        old.manifest(out)
        raise


def template(phase, rep):
    """Read metadata/hashes only; missing future decisions remain explicit."""
    require(phase in PHASES and rep in range(3), "template phase/rep required")
    qa_path = OLD_ARTIFACT+"/qa_001/QA_INDEX.json"
    proposal_path = OLD_ARTIFACT+"/training_001/FROZEN_PROPOSALS.json"
    cfg = {"schema": SCHEMA, "phase": phase, "frozen": False, "authorized_to_execute": False,
        "baseline_commit": BASELINE, "handoff_sha256": HANDOFF_SHA, "case_ids": [rep], "training_N": list(SIZES),
        "proposal_kind": "sym", "K": K, "alpha": core.ALPHA, "direction": core.V.tolist(), "t_values": list(AMPLITUDES),
        "pattern_chunk": 8, "teacher_policy": "same_saved_GH36_Cholesky_p0_dp0_and_pt",
        "qa_index": qa_path, "proposal_freeze": proposal_path,
        "development_decision": None if phase == "development" else ARTIFACT+"/DEVELOPMENT_DECISION_001.json",
        "prediction_freeze": ARTIFACT+"/COEFFICIENTS_AND_PREDICTIONS_FROZEN_001.json" if phase == "validation" else None,
        "rule_policy": ARTIFACT+"/RULE_POLICY_001.json",
        "rule_check": {"rep": 0, "N": 16384, "low_order": 28, "high_order": 36,
                       "max_replicate_normalized_D_difference": 1e-4, "max_replicate_W_same_difference": 1e-4},
        "randomization": None if phase == "development" else {"family": 202610166301 if phase == "coefficients" else 202610166401,
            "role": 60 if phase == "coefficients" else 70, "R": R, "B": 256 if phase == "coefficients" else 512, "bits": 52},
        "old_banks": [{"rep": rep, "N": n,
            "coefficient_root": f"{OLD_ARTIFACT}/coefficients_rep{rep}_001/rep_{rep}_N{n}_sym",
            "validation_root": f"{OLD_ARTIFACT}/validation_rep{rep}_001/rep_{rep}_N{n}_sym"} for n in SIZES] if phase == "development" else [],
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "caps": counter_limits(phase, rep), "resources": {"max_rss_bytes": 3*1024**3, "max_output_bytes": 4*1024**3},
        "external_timeout_seconds": 7260, "source_sha256": {}, "input_sha256": {},
        "scope": "one fixed development and conditional single independent coefficient/validation round; no new data/training"}
    sources = set(BASE_SOURCES)
    for name in ("tests/test_learned_precision_core.py", "tests/test_learned_precision_run.py"):
        if resolve(name).is_file():
            sources.add(name)
    cfg["source_sha256"] = {name: sha(resolve(name)) for name in sorted(sources)}
    inputs = {"HANDOFF.md", "docs/08_research_研究工作/LearnedPrecision_v1_20261016/PLAN.md", qa_path, proposal_path, cfg["rule_policy"]}
    qa = read(resolve(qa_path))
    for t in (0., *AMPLITUDES):
        for order in (28, 36):
            record = qa["teachers"][str(t)][str(order)]
            inputs.update(record[key] for key in ("json", "npz"))
    for pair in read(resolve(proposal_path))["pairs"]:
        inputs.add(pair["proposals"]["sym"]["path"])
    for bank in cfg["old_banks"]:
        for category, phase_name in (("coefficient", "coefficients"), ("validation", "validation")):
            root = bank[category+"_root"]
            inputs.update(old_bank_paths(root, phase_name == "validation"))
            inputs.update((Path(root).parent/name).as_posix() for name in ("CONTRACT.json", "FREEZE.json", "MANIFEST.json"))
    for field in ("development_decision", "prediction_freeze"):
        if cfg[field] is not None:
            inputs.add(cfg[field])
    if cfg["prediction_freeze"] is not None and resolve(cfg["prediction_freeze"]).is_file():
        for row in read(resolve(cfg["prediction_freeze"]))["coefficient_outputs"]:
            inputs.update(row[key] for key in ("statistics", "banks"))
    cfg["input_sha256"] = {name: sha(resolve(name)) if resolve(name).is_file() else "REPLACE_WITH_ACTUAL_SHA256_BEFORE_FREEZE" for name in sorted(inputs)}
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--template", choices=PHASES)
    parser.add_argument("--rep", type=int, choices=range(3))
    args = parser.parse_args()
    if args.template:
        if args.rep is None or args.contract or args.check:
            parser.error("--template requires --rep and no contract/check")
        value = template(args.template, args.rep)
        if args.out:
            write(args.out, value)
        else:
            print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.check:
        if not args.contract or args.out:
            parser.error("--check requires --contract and no output")
        validate(read(args.contract))
        print("source/input/phase identities pass; no model, RNG, quadrature, fitting or objective execution")
    else:
        if not args.contract or not args.out:
            parser.error("--contract and fresh --out required")
        result = run(args.contract.resolve(), args.out.resolve())
        print(json.dumps({"phase": result["phase"], "status": result["status"]}))
