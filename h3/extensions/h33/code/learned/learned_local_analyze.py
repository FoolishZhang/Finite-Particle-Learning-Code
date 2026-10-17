"""Saved-JSON analysis for the frozen learned-local-proposal study.

No model, sampler, optimizer, NPZ, or scientific RNG is imported or executed.
All population quantities below are arithmetic on saved complete-randomization
statistics. Missing/invalid cells remain in the fixed analysis family.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy.stats import t as student_t

SCHEMA = "learned_local_saved_analysis_v1"
CONTRACT_SCHEMA = "learned_local_analysis_contract_v1"
REPO = Path(__file__).resolve().parents[1]
SOURCE = "script/learned_local_analyze.py"
R, FAMILY_SIZE, FAMILY_ALPHA, DF = 16, 24, .05, 15
ALPHA = .3201223434446316
KINDS = ("raw", "sym")
SIZES = (16384, 65536)
CV_KEYS = ("cv_g", "cv_H", "cv_B")
ALL_KEYS = ("raw_g", "cv_g", "raw_H", "cv_H", "raw_B", "cv_B")
PRODUCER_SOURCES = {"script/learned_local_run.py", "script/learned_local_proposal.py", "script/learned_local_objective.py",
                    "script/competition_core.py", "script/active_reference_likelihood.py", "script/group_posterior_objective.py",
                    "script/signal_proposal_core.py", "script/informative_projection.py"}
LIMITATIONS = [
    "Conditional numerical-randomization uncertainty given all frozen trained proposals; not training-population inference.",
    "RQMC replicate t intervals and Bonferroni family coverage are approximate, not exact finite-sample guarantees.",
    "Numerical replicate SE does not cover private-channel or outer-GH systematic integration error.",
    "Sym g/offdiagonal cancellation is imposed by reflected state sampling and full observation enumeration; zeros are implementation checks.",
    "H/B diagonal entries and candidate differences are computed from full weights, not assigned old exact-proposal coefficients.",
    "Positive estimated sym h and a reported b/h are not positive-definiteness, optimum, shrinkage, or global certificates.",
    "Raw b/h is not interpreted as a stationary branch through independence when its anchor gradient may be nonzero.",
    "Raw versus sym compares explicit reflection averaging of the same trained density; it also changes the mixture representation.",
    "Prediction-tolerance success is a predefined numerical tolerance, not a Taylor remainder theorem or risk equivalence.",
    "Negative-chain symmetry checks do not add an independent environment; numerical replicates do not add training repeats.",
    "Missing/nonfinite/wrong-R inputs remain unresolved slots without reducing the primary family size or replacing seeds.",
    "Only bound JSON statistics/metadata are read; particle arrays, NPZ files and model kernels are not inspected or recomputed.",
]


def require(test, message):
    if not test:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    """Keep malformed numeric inputs legible without emitting nonstandard JSON."""
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_input_value": repr(value)}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(name):
    require(isinstance(name, str) and name, "nonempty repository-relative path required")
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts, "safe repository-relative path required")
    target = (REPO / rel).resolve()
    require(target.is_relative_to(REPO.resolve()), "input escapes repository")
    return target


def proposal_keys():
    return [(rep, n, kind) for rep in range(3) for n in SIZES for kind in KINDS]


def critical_value():
    return float(student_t.ppf(1 - FAMILY_ALPHA / (2 * FAMILY_SIZE), DF))


def policy_check(policy):
    require(policy.get("schema") == "learned_local_analysis_policy_v1" and policy.get("frozen") is True,
            "frozen analysis policy required")
    expected = {"training_repeats": [0, 1, 2], "training_N": list(SIZES), "proposal_kinds": list(KINDS),
                "K": 8, "alpha": ALPHA, "sym_t": [.1, .2], "raw_t": [-.2, -.1, .1, .2],
                "coefficient_R": R, "validation_R": R, "coefficient_B": 256, "validation_B": 512}
    for key, value in expected.items():
        require(policy.get(key) == value, "frozen policy differs: " + key)
    v = np.asarray(policy["v"], float)
    require(v.shape == (3,) and np.allclose(v, [1/math.sqrt(2), 1/math.sqrt(2), 0], rtol=0, atol=1e-15),
            "fixed normalized chain direction required")
    fam = policy["primary_family"]
    require((fam["size"], fam["alpha"], fam["df"]) == (FAMILY_SIZE, FAMILY_ALPHA, DF), "fixed family24 required")
    return policy


class BoundJSON:
    """Read only explicitly SHA-bound JSON; identity failures are inspectable."""
    def __init__(self):
        self.audit = []
        self.documents = {}
        self.cache = {}

    def load(self, record, purpose):
        item = {"purpose": purpose, "record": record, "status": "unresolved"}
        self.audit.append(item)
        try:
            require(isinstance(record, dict), "bound JSON record required")
            path = resolve(record["path"])
            require(path.suffix.lower() == ".json", "only JSON inputs permitted")
            digest = record["sha256"]
            require(isinstance(digest, str) and len(digest) == 64, "SHA256 required")
            cache_key = (path.as_posix(), digest.lower())
            if cache_key in self.cache:
                item.update(status="verified", cached=True)
                return self.cache[cache_key], []
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            item["actual_sha256"] = actual
            require(actual == digest.lower(), "SHA256 mismatch")
            doc = json.loads(payload.decode("utf-8-sig"))
            require(isinstance(doc, dict), "JSON object required")
            self.cache[cache_key] = doc
            self.documents[record["path"]] = doc
            item.update(status="verified", bytes=len(payload), cached=False)
            return doc, []
        except (OSError, ValueError, KeyError, TypeError) as error:
            message = purpose + ": " + str(error)
            item["error"] = message
            return None, [message]


def put(index, key, document, errors, source):
    if key in index:
        index[key]["errors"].append("duplicate records for expected key: " + repr(key))
        index[key].setdefault("duplicate_records", []).append({"source": source, "document": document, "errors": errors})
    else:
        index[key] = {"document": document, "errors": list(errors), "source": source}


def get_key(record):
    return (record["rep"], record["N"], record["kind"])


def records_from(document, field, notes, label):
    records = document.get(field, [])
    if not isinstance(records, list):
        notes.append(label + " has no valid " + field + " list; all unprovided slots remain unresolved")
        return []
    return records


def canonical(name):
    return resolve(name).as_posix().casefold()


def same_binding(left, right):
    return canonical(left["path"]) == canonical(right["path"]) and left["sha256"].lower() == right["sha256"].lower()


def input_binding(producer_contract, binding):
    matches = [digest for name, digest in producer_contract["input_sha256"].items()
               if canonical(name) == canonical(binding["path"])]
    require(len(matches) == 1 and matches[0] == binding["sha256"], "producer input SHA does not bind " + binding["path"])


def manifest_index(root, manifest):
    require(isinstance(manifest.get("files"), list) and manifest.get("manifest_self_excluded") is True,
            "producer manifest file list required")
    index = {}
    for item in manifest["files"]:
        rel = Path(item["path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe manifest-relative path")
        path = resolve(root) / rel
        name = path.resolve().as_posix().casefold()
        require(path.resolve().is_relative_to(resolve(root)) and name not in index, "duplicate/escaping manifest identity")
        require(isinstance(item["sha256"], str) and len(item["sha256"]) == 64, "manifest SHA256 required")
        index[name] = item["sha256"]
    return index


def in_manifest(info, binding):
    require(info.get("manifest_index", {}).get(canonical(binding["path"])) == binding["sha256"],
            "JSON binding missing/different in producer manifest: " + binding["path"])


def root_file(root, binding, basename):
    require(resolve(binding["path"]) == resolve(root) / basename, "producer root " + basename + " required")


def producer_metadata(cfg, record, loader):
    """Validate recorded execution identities, without loading scientific arrays."""
    info = {"id": record.get("id"), "phase": record.get("phase"), "root": record.get("root"),
            "record": record, "errors": [], "status": "unresolved", "outputs": {}}
    docs = {}
    for field, basename in (("contract", "CONTRACT.json"), ("freeze", "FREEZE.json"), ("manifest", "MANIFEST.json")):
        try:
            binding = record[field]
            root_file(record["root"], binding, basename)
            docs[field], errors = loader.load(binding, "producer " + str(record.get("id")) + " " + field)
            info["errors"].extend(errors)
        except (KeyError, ValueError, TypeError) as error:
            docs[field] = None
            info["errors"].append("producer metadata: " + str(error))
    if any(docs.get(field) is None for field in ("contract", "freeze", "manifest")):
        return info
    try:
        contract, freeze = docs["contract"], docs["freeze"]
        info["contract"] = contract
        info["manifest_index"] = manifest_index(record["root"], docs["manifest"])
        in_manifest(info, record["contract"])
        in_manifest(info, record["freeze"])
        require(contract.get("schema") == "learned_local_execution_v1" and contract.get("phase") == record["phase"],
                "producer phase/schema mismatch")
        require(contract.get("frozen") is True and contract.get("authorized_to_execute") is True, "producer not frozen/authorized")
        require(freeze.get("schema") == "learned_local_execution_v1" and freeze.get("phase") == record["phase"],
                "producer FREEZE phase/schema mismatch")
        require(freeze.get("contract_sha256") == record["contract"]["sha256"], "producer FREEZE contract SHA mismatch")
        require(freeze.get("before_scientific_calls") is True, "producer FREEZE is not before scientific calls")
        require(freeze.get("source_sha256") == contract.get("source_sha256")
                and freeze.get("input_sha256") == contract.get("input_sha256"), "producer FREEZE maps differ from contract")
        for source, digest in contract["source_sha256"].items():
            binding = {"path": (Path(record["root"])/"SEALED_SOURCE"/source).as_posix(), "sha256": digest}
            in_manifest(info, binding)
        phase = record["phase"]
        if phase in ("coefficients", "validation"):
            require(contract["source_sha256"] == cfg["producer_source_sha256"], "producer differs from the fixed scientific source closure")
            require(contract.get("baseline_commit") == cfg["baseline_commit"], "producer baseline differs")
            case_ids = contract["case_ids"]
            require(len(case_ids) == 1 and case_ids[0] in (0, 1, 2), "one predetermined training repeat per statistical producer required")
            family, role, banks = (202610165301, 30, 256) if phase == "coefficients" else (202610165401, 40, 512)
            q = contract["randomization"]
            require((q["family"], q["role"], q["R"], q["B"], q["bits"], contract["K"]) == (family, role, R, banks, 52, 8),
                    "producer randomization family/role/R/B/bits/K mismatch")
            require(canonical(contract["proposal_freeze"]) == canonical(cfg["proposal_freeze"]["path"]),
                    "producer used a different proposal freeze")
            input_binding(contract, cfg["proposal_freeze"])
            require(canonical(contract["qa_index"]) == canonical(cfg["qa_index"]["path"]), "producer used a different QA index")
            input_binding(contract, cfg["qa_index"])
            if phase == "validation":
                require(canonical(contract["coefficient_freeze"]) == canonical(cfg["coefficient_freeze"]["path"]),
                        "validation used a different global coefficient freeze")
                input_binding(contract, cfg["coefficient_freeze"])
                require(contract.get("alpha") == ALPHA and contract.get("sym_t") == [.1, .2]
                        and contract.get("raw_t") == [-.2, -.1, .1, .2], "producer candidate policy differs")
        info["status"] = "verified"
    except (KeyError, ValueError, TypeError, IndexError) as error:
        info["errors"].append("producer identity/policy: " + str(error))
    if info["errors"]:
        info["status"] = "unresolved"
    return info


def verify_bank(info, output, key, model, selected_epoch, loader):
    issue = []
    bank_docs = {}
    try:
        rep, n, kind = key
        expected = resolve(info["root"]) / ("rep_" + str(rep) + "_N" + str(n) + "_" + kind)
        require(resolve(output["directory"]) == expected, "proposal output directory does not identify rep/N/kind")
        for field, basename in (("bank_complete", "COMPLETE.json"), ("bank_rng", "RNG.json")):
            binding = output[field]
            require(resolve(binding["path"]) == expected/"BANK"/basename, "BANK metadata outside declared proposal directory")
            in_manifest(info, binding)
            bank_docs[field], errors = loader.load(binding, "producer bank " + repr(key) + " " + field)
            issue.extend(errors)
        if issue:
            return issue
        actual = bank_docs["bank_complete"]["proposal"]
        require(model["document"] is not None and not model["errors"], "matching training proposal unavailable")
        frozen = model["document"]
        for field in ("schema", "kind", "defense", "raw_parameters", "parameters", "shared_homogeneous_groups", "construction"):
            require(actual.get(field) == frozen.get(field), "actual bank proposal differs from trained proposal: " + field)
        origin = actual["source"]
        require((origin.get("rep"), origin.get("N"), origin.get("selected_epoch")) == (rep, n, selected_epoch),
                "actual bank proposal training source differs")
        recorded_source = frozen["source"]
        require((recorded_source.get("rep"), recorded_source.get("N"), recorded_source.get("epoch")) == (rep, n, selected_epoch),
                "training proposal source differs from pair freeze")
        rng = bank_docs["bank_rng"]
        family, role, banks = (202610165301, 30, 256) if info["phase"] == "coefficients" else (202610165401, 40, 512)
        require((rng["family"], rng["role"], rng["R"], rng["B"], rng["K"], rng["kind"]) == (family, role, R, banks, 8, kind),
                "actual RNG family/role/R/B/K/kind differs")
        states = rng["states"]
        require(isinstance(states, list) and len(states) == R, "all16 actual randomization state records required")
        for r, state in enumerate(states):
            require(state["randomization"] == r and state["entropy"] == [family, role, rep, n, 0 if kind == "raw" else 1, r],
                    "actual RNG entropy differs at randomization " + str(r))
            require(state.get("rng_after", {}).get("bit_generator") == "PCG64DXSM", "actual RNG generator differs")
    except (KeyError, ValueError, TypeError, IndexError) as error:
        issue.append("actual proposal/RNG provenance: " + str(error))
    return issue


def provenance_inputs(cfg, loader, models, coefficients, contrasts, densities, pfreeze, cfreeze):
    """Cross-link frozen stages; an incorrect source makes affected slots unresolved."""
    producers, roots, notes = {}, {}, []
    output_index = {"coefficients": {}, "validation": {}}
    for record in cfg["producer_runs"]:
        info = producer_metadata(cfg, record, loader)
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in producers:
            if identifier in producers:
                producers[identifier]["errors"].append("duplicate producer id")
                producers[identifier]["status"] = "unresolved"
            notes.append("duplicate/missing producer id: " + repr(identifier))
            continue
        producers[identifier] = info
        try:
            root = canonical(record["root"])
            if root in roots:
                info["errors"].append("same canonical producer root declared twice")
                producers[roots[root]]["errors"].append("same canonical producer root declared twice")
                producers[roots[root]]["status"] = "unresolved"
                info["status"] = "unresolved"
            roots[root] = identifier
        except (KeyError, ValueError, TypeError) as error:
            info["errors"].append(str(error))
        if info["phase"] not in output_index:
            continue
        contract = info.get("contract", {})
        try:
            if pfreeze is None or cfreeze is None:
                raise ValueError("global proposal/coefficient freeze unavailable")
            for pair in pfreeze["pairs"]:
                for binding in pair["proposals"].values():
                    input_binding(contract, binding)
            if info["phase"] == "validation":
                for row in cfreeze["outputs"]:
                    for name in ("statistics", "banks"):
                        # Compare NPZ identities declared in JSON only; never read NPZ.
                        input_binding(contract, {"path": row[name], "sha256": row[name + "_sha256"]})
        except (KeyError, ValueError, TypeError) as error:
            info["errors"].append("producer global-freeze closure: " + str(error))
            info["status"] = "unresolved"
        seen = set()
        for output in record.get("outputs", []):
            try:
                key = get_key(output)
                output_errors = []
                require(key in proposal_keys(), "unexpected proposal identity")
                require(key[0] in contract.get("case_ids", []), "proposal repeat outside producer case_ids")
                require(key not in seen, "duplicate proposal directory in producer")
                seen.add(key)
                model = missing_entry(models, key, "proposal")
                selected = (model.get("source") or {}).get("selected_epoch")
                output_errors.extend(verify_bank(info, output, key, model, selected, loader))
                info["outputs"][repr(key)] = {"binding": output, "errors": output_errors,
                                               "status": "verified" if not output_errors else "unresolved"}
                output_index[info["phase"]].setdefault(key, []).append({"info": info, "output": output, "errors": output_errors})
            except (KeyError, ValueError, TypeError) as error:
                info["errors"].append("producer output index: " + str(error))
                info["status"] = "unresolved"
        wanted = {(r, n, k) for r in contract.get("case_ids", []) for n in SIZES for k in KINDS}
        if seen != wanted:
            info["errors"].append("producer output directory index does not cover case_ids x twoN x twoKinds")
            info["status"] = "unresolved"
    for phase, entries in (("coefficients", coefficients), ("validation", contrasts)):
        for key, entry in entries.items():
            qkey = key[:3]
            matches = output_index[phase].get(qkey, [])
            if len(matches) != 1:
                entry["errors"].append("no unique declared " + phase + " producer for " + repr(qkey))
                continue
            matched = matches[0]
            info, output = matched["info"], matched["output"]
            entry["errors"].extend(info["errors"] + matched["errors"])
            try:
                source = entry["source"] if phase == "coefficients" else entry["source"]["contrast"]
                binding = {"path": source["statistics"], "sha256": source["statistics_sha256"]}
                basename = "COEFFICIENT_STATISTICS.json" if phase == "coefficients" else (
                    "CONTRAST_t" + str(key[3]).replace("-", "m").replace(".", "p") + ".json")
                require(resolve(binding["path"]) == resolve(output["directory"])/basename,
                        "statistics file outside the declared run/proposal directory")
                in_manifest(info, binding)
                if phase == "validation":
                    summary_binding = entry["source"]["summary"]
                    root_file(info["root"], summary_binding, "SUMMARY.json")
                    in_manifest(info, summary_binding)
                entry["producer_provenance"] = {"producer_id": info["id"], "root": info["root"],
                                                  "output_directory": output["directory"],
                                                  "status": "verified" if not entry["errors"] else "unresolved"}
            except (KeyError, ValueError, TypeError) as error:
                entry["errors"].append("statistics provenance: " + str(error))
    for key, entry in densities.items():
        matches = output_index["coefficients"].get(key, [])
        if len(matches) != 1:
            entry["errors"].append("no unique coefficient producer for density metadata")
            continue
        match = matches[0]
        entry["errors"].extend(match["info"]["errors"] + match["errors"])
        try:
            require(resolve(entry["source"]["path"]) == resolve(match["output"]["directory"])/"DENSITY_DIAGNOSTICS.json",
                    "density metadata outside matching coefficient proposal directory")
            in_manifest(match["info"], entry["source"])
        except (KeyError, ValueError, TypeError) as error:
            entry["errors"].append("density provenance: " + str(error))
    for info in producers.values():
        if info["errors"]:
            info["status"] = "unresolved"
            notes.extend(["producer " + str(info["id"]) + ": " + error for error in info["errors"]])
    return producers, notes


def load_inputs(cfg, loader):
    models, coefficients, contrasts, densities = {}, {}, {}, {}
    notes = []
    pfreeze, errors = loader.load(cfg["proposal_freeze"], "proposal_freeze")
    notes.extend(errors)
    if pfreeze is not None:
        freeze_issue = []
        if pfreeze.get("all_training_complete") is not True:
            freeze_issue.append("proposal freeze does not declare all_training_complete")
        try:
            pairs = pfreeze["pairs"]
            require(len(pairs) == 6 and {(pair["rep"], pair["N"]) for pair in pairs} == {(r, n) for r in range(3) for n in SIZES},
                    "proposal freeze must contain all six fixed training pairs")
        except (KeyError, TypeError, ValueError) as error:
            freeze_issue.append(str(error))
        notes.extend(freeze_issue)
        for pair in records_from(pfreeze, "pairs", notes, "proposal freeze"):
            for kind in KINDS:
                try:
                    key = (pair["rep"], pair["N"], kind)
                    binding = pair["proposals"][kind]
                    doc, issue = loader.load(binding, "proposal " + repr(key))
                    issue.extend(freeze_issue)
                    if doc is not None:
                        if doc.get("kind") != kind or doc.get("defense") != .05 or pair.get("defense") != .05:
                            issue.append("proposal kind/defense mismatch")
                        if doc.get("shared_homogeneous_groups") != 3:
                            issue.append("proposal must share all three homogeneous groups")
                        if doc.get("raw_parameters") != pair.get("raw_parameters"):
                            issue.append("proposal raw parameters differ from bound freeze")
                    put(models, key, doc, issue, {"binding": binding, "selected_epoch": pair.get("selected_epoch")})
                except (KeyError, TypeError, ValueError) as error:
                    notes.append("malformed proposal-freeze entry: " + str(error))
    cfreeze, errors = loader.load(cfg["coefficient_freeze"], "coefficient_freeze")
    notes.extend(errors)
    if cfreeze is not None:
        freeze_issue = []
        if cfreeze.get("all_coefficients_complete") is not True:
            freeze_issue.append("coefficient freeze does not declare all_coefficients_complete")
        try:
            outputs = cfreeze["outputs"]
            require(len(outputs) == 12 and {get_key(record) for record in outputs} == set(proposal_keys()),
                    "global coefficient freeze must contain all12 fixed proposal outputs")
        except (KeyError, TypeError, ValueError) as error:
            freeze_issue.append(str(error))
        notes.extend(freeze_issue)
        for record in records_from(cfreeze, "outputs", notes, "coefficient freeze"):
            try:
                key = get_key(record)
                binding = {"path": record["statistics"], "sha256": record["statistics_sha256"]}
                doc, issue = loader.load(binding, "coefficient " + repr(key))
                issue.extend(freeze_issue)
                if record.get("role") != "independent_coefficient_evaluation":
                    issue.append("coefficient record is not independent_coefficient_evaluation")
                put(coefficients, key, doc, issue, record)
            except (KeyError, TypeError, ValueError) as error:
                notes.append("malformed coefficient-freeze entry: " + str(error))
    for binding in cfg["validation_summaries"]:
        summary, errors = loader.load(binding, "validation_summary")
        notes.extend(errors)
        if summary is None:
            continue
        phase_error = [] if summary.get("phase") == "validation" else ["summary phase is not validation"]
        for output in records_from(summary, "outputs", notes, "validation summary"):
            try:
                key = get_key(output)
                for record in output.get("contrasts", []):
                    ckey = (*key, float(record["t"]))
                    stats_binding = {"path": record["statistics"], "sha256": record["statistics_sha256"]}
                    doc, issue = loader.load(stats_binding, "contrast " + repr(ckey))
                    put(contrasts, ckey, doc, [*phase_error, *issue],
                        {"summary": binding, "output": {k: output[k] for k in ("rep", "N", "kind")}, "contrast": record})
            except (KeyError, TypeError, ValueError) as error:
                notes.append("malformed validation-summary entry: " + str(error))
    for binding in cfg.get("density_diagnostics", []):
        try:
            key = get_key(binding)
            doc, errors = loader.load(binding, "density_diagnostic " + repr(key))
            put(densities, key, doc, errors, binding)
        except (KeyError, TypeError, ValueError) as error:
            notes.append("malformed density-diagnostic binding: " + str(error))
    expected = set(proposal_keys())
    expected_contrasts = {(*key, t) for key in expected for t in ([.1, .2] if key[2] == "sym" else [-.2, -.1, .1, .2])}
    for name, index, wanted in (("models", models, expected), ("coefficients", coefficients, expected),
                                 ("densities", densities, expected), ("contrasts", contrasts, expected_contrasts)):
        for key in index.keys() - wanted:
            notes.append("unexpected " + name + " record retained in input archive but excluded from fixed slots: " + repr(key))
    producers, provenance_notes = provenance_inputs(cfg, loader, models, coefficients, contrasts, densities, pfreeze, cfreeze)
    notes.extend(provenance_notes)
    costs = cost_summary(cfg["phase_receipts"], loader, producers)
    costs["producer_provenance"] = [{key: value for key, value in info.items() if key != "manifest_index"}
                                     for info in producers.values()]
    return models, coefficients, contrasts, densities, costs, notes


def sample_summary(values):
    values = np.asarray(values, float)
    require(values.shape[0] == R and np.isfinite(values).all(), "16 finite complete randomizations required")
    with np.errstate(over="ignore", invalid="ignore"):
        sd, mean = values.std(axis=0, ddof=1), values.mean(axis=0)
    require(np.isfinite(sd).all() and np.isfinite(mean).all(), "nonfinite derived mean/SD")
    return {"R": R, "replicates": values, "mean": mean, "sample_SD": sd, "SE": sd / math.sqrt(R)}


def covariance(values, labels):
    values = np.asarray(values, float)
    require(values.shape == (R, len(labels)) and np.isfinite(values).all(), "finite covariance matrix input required")
    centered = values - values.mean(axis=0)
    cov = centered.T @ centered / DF
    require(np.isfinite(cov).all(), "nonfinite derived covariance")
    return {"labels": labels, "sample_covariance": cov, "covariance_of_mean": cov / R,
            "unit": "same complete-randomization index; cross-stream covariance is descriptive, not an independence test"}


def saved_stat(record, shape, banks):
    require(isinstance(record, dict) and record.get("R") == R, "saved statistic has wrong R")
    require(record.get("B") == banks, "saved statistic has wrong within-randomization B")
    arr = np.asarray(record["replicates"], float)
    require(arr.shape == (R, *shape) and np.isfinite(arr).all(), "saved replicate shape/nonfinite failure")
    summary = sample_summary(arr)
    for name in ("mean", "SE"):
        value = np.asarray(record[name], float)
        require(value.shape == shape and np.isfinite(value).all(), "saved " + name + " shape/nonfinite failure")
        require(np.allclose(value, summary[name], rtol=2e-12, atol=2e-14), "saved " + name + " inconsistent with its replicates")
    return summary


def missing_entry(index, key, label):
    item = index.get(key)
    if item is None:
        return {"document": None, "source": None, "errors": ["missing " + label + " for " + repr(key)]}
    return item


def coefficient_cell(key, entry, model):
    rep, n, kind = key
    cell = {"rep": rep, "N": n, "kind": kind, "status": "unresolved", "source": entry["source"],
            "errors": [*entry["errors"], *model["errors"]], "saved_statistics": entry["document"],
            "producer_provenance": entry.get("producer_provenance"),
            "model_metadata": model["source"], "symmetry_zeros_are_structural": kind == "sym",
            "ratio_interpretation": "conditional descriptive sym local-response estimate, not a certificate" if kind == "sym"
                                    else "not reported: raw anchor need not be stationary"}
    if entry["document"] is None or model["document"] is None or cell["errors"]:
        return cell
    try:
        doc = entry["document"]
        require(doc.get("kind") == kind, "coefficient kind mismatch")
        require(doc.get("control") == "uniform_logratio_actual_mixture_analytic_moments", "wrong coefficient control")
        stats = {name: saved_stat(doc["statistics"][name], (3,) if name.endswith("g") else (3, 3), 256)
                 for name in ALL_KEYS}
        g, h, b = (stats[name]["replicates"] for name in CV_KEYS)
        hreps, breps = np.trace(h, axis1=1, axis2=2) / 3, np.trace(b, axis1=1, axis2=2) / 3
        hs, bs = sample_summary(hreps), sample_summary(breps)
        cell.update(status="available", statistics=stats, h=hs, b=bs,
                    h_b_covariance=covariance(np.column_stack([hreps, breps]), ["h_trace_over3", "b_trace_over3"]))
        for prefix in ("cv", "raw"):
            arrays = [stats[prefix + "_" + suffix]["replicates"].reshape(R, -1) for suffix in ("g", "H", "B")]
            labels = [prefix + "_g_" + str(i) for i in range(3)]
            labels += [prefix + "_" + name + "_" + str(i) + str(j) for name in ("H", "B") for i in range(3) for j in range(3)]
            cell[prefix + "_component_covariance"] = covariance(np.concatenate(arrays, axis=1), labels)
        mask = ~np.eye(3, dtype=bool)
        cell["structure_diagnostics"] = {
            "maximum_abs_cv_g_replicate": float(np.max(abs(g))),
            "maximum_abs_cv_H_offdiagonal_replicate": float(np.max(abs(h[:, mask]))),
            "maximum_abs_cv_B_offdiagonal_replicate": float(np.max(abs(b[:, mask]))),
            "sym_origin_cancellation_imposed": kind == "sym",
            "not_an_independent_symmetry_discovery": kind == "sym"}
        cell["response_ratio"] = None
        if kind == "sym" and float(hs["mean"]) > 0:
            rhat = float(bs["mean"] / hs["mean"])
            delta = (breps - rhat * hreps) / float(hs["mean"])
            require(math.isfinite(rhat) and np.isfinite(delta).all(), "nonfinite ratio/delta linearization")
            cell["response_ratio"] = {"estimate": rhat, "delta_SE": float(delta.std(ddof=1) / math.sqrt(R)),
                                      "delta_replicates": delta, "positive_estimated_h_only": True,
                                      "certified_h_positive": False, "certified_shrinkage_or_optimum": False}
    except (KeyError, ValueError, TypeError, IndexError) as error:
        cell["errors"].append("coefficient statistics invalid: " + str(error))
        cell["status"] = "unresolved"
    return cell


def prediction_parts(coefficient, policy):
    g, h, b = [coefficient["statistics"][key]["replicates"] for key in CV_KEYS]
    v, alpha = np.asarray(policy["v"]), policy["alpha"]
    linear = (alpha - 1) * (g @ v)
    quadratic = .5 * (alpha**2 - 1) * np.einsum("i,rij,j->r", v, h, v)
    quadratic -= (alpha - 1) * np.einsum("i,rij,j->r", v, b, v)
    return linear, quadratic


def simultaneous(summary, critical):
    mean, se = float(summary["mean"]), float(summary["SE"])
    return {"lower": mean - critical * se, "upper": mean + critical * se,
            "critical_value": critical, "df": DF, "family_size": FAMILY_SIZE, "family_alpha": FAMILY_ALPHA,
            "method": "approximate_Bonferroni_two_sided_RQMC_replicate_t"}


def contrast_cell(key, entry, coefficient, policy, critical):
    rep, n, kind, tval = key
    cell = {"rep": rep, "N": n, "kind": kind, "t": tval, "status": "unresolved",
            "source": entry["source"], "saved_statistics": entry["document"], "errors": list(entry["errors"]),
            "producer_provenance": entry.get("producer_provenance"),
            "D_direction": "unresolved", "prediction_precision": "unresolved"}
    if coefficient["status"] != "available":
        cell["errors"].append("coefficient unavailable; prediction and primary paired cell unresolved")
    if entry["document"] is None or cell["errors"]:
        return cell
    try:
        doc = entry["document"]
        require(doc.get("t") == tval and doc.get("alpha") == policy["alpha"], "contrast t/alpha mismatch")
        v = np.asarray(policy["v"])
        coordinates = np.asarray(doc["candidate_coordinates"], float)
        expected = np.array([policy["alpha"] * tval * v, tval * v])
        require(coordinates.shape == (2, 3) and np.allclose(coordinates, expected, rtol=0, atol=2e-15), "candidate coordinates mismatch")
        require(doc.get("pairing") == "same_q_particles_and_same_teacher_PMF; shrink_candidate_minus_teacher_candidate",
                "candidate difference pairing/direction mismatch")
        d = saved_stat(doc["statistics"], (), 512)
        raw_d = saved_stat(doc["raw_statistics"], (), 512)
        linear, quadratic = prediction_parts(coefficient, policy)
        p = sample_summary(tval * linear + tval**2 * quadratic)
        residual = sample_summary(d["replicates"] - p["replicates"])
        w = sample_summary(residual["replicates"] / tval**2)
        cell.update(status="available", D=d, raw_uncontrolled_D=raw_d, prediction=p,
                    residual=residual, normalized_residual=w,
                    linear_coefficient=sample_summary(linear), quadratic_coefficient=sample_summary(quadratic),
                    covariance=covariance(np.column_stack([d["replicates"], p["replicates"], residual["replicates"], w["replicates"]]),
                                          ["D", "P", "D_minus_P", "(D_minus_P)/t_squared"]))
        if kind == "sym":
            dci, wci = simultaneous(d, critical), simultaneous(w, critical)
            cell.update(D_simultaneous_interval=dci, normalized_residual_simultaneous_interval=wci)
            cell["D_direction"] = "negative_supports_fixed_shrink_candidate" if dci["upper"] < 0 else (
                "positive_supports_teacher_candidate" if dci["lower"] > 0 else "unresolved_direction")
            cell["prediction_precision"] = "within_predefined_tolerance" if wci["lower"] >= -.01 and wci["upper"] <= .01 else "not_within_predefined_tolerance"
        else:
            cell.update(D_direction="descriptive_only", prediction_precision="descriptive_only", simultaneous_interval=None)
    except (KeyError, ValueError, TypeError, IndexError) as error:
        cell["errors"].append("contrast statistics invalid: " + str(error))
        cell["status"] = "unresolved"
    return cell


def odd_even_cell(rep, n, tval, positive, negative, coefficient, policy):
    cell = {"rep": rep, "N": n, "kind": "raw", "abs_t": tval, "status": "unresolved", "errors": [],
            "definition": "odd=(D(+t)-D(-t))/(2t); even=(D(+t)+D(-t))/(2t^2); same validation replicate indices",
            "inference": "descriptive mean/SE only; no simultaneous intervals or selected significance conclusions"}
    if positive["status"] != "available" or negative["status"] != "available" or coefficient["status"] != "available":
        cell["errors"].append("both signed contrasts and full coefficients required")
        return cell
    plus, minus = positive["D"]["replicates"], negative["D"]["replicates"]
    linear, quadratic = prediction_parts(coefficient, policy)
    odd_part, even_part = (plus - minus) / 2, (plus + minus) / 2
    odd, even = odd_part / tval, even_part / tval**2
    arrays = {"D_positive": plus, "D_negative": minus, "odd_part": odd_part, "even_part": even_part,
              "odd_coefficient": odd, "even_coefficient": even,
              "predicted_odd_coefficient": linear, "predicted_even_coefficient": quadratic,
              "odd_coefficient_residual": odd - linear, "even_coefficient_residual": even - quadratic}
    cell.update(status="available", statistics={name: sample_summary(arr) for name, arr in arrays.items()},
                covariance=covariance(np.column_stack(list(arrays.values())), list(arrays)))
    return cell


def density_cell(key, entry):
    cell = dict(zip(("rep", "N", "kind"), key))
    cell.update(status="unresolved", source=entry["source"], errors=list(entry["errors"]),
                saved_diagnostic=entry["document"], ordinary_diagnostic_qualified=None, output_inclusion_rule=False)
    if entry["document"] is None or cell["errors"]:
        return cell
    try:
        doc = entry["document"]
        require(len(doc["levels"]) == 2, "two density rules required")
        for level, order in zip(doc["levels"], (256, 384)):
            require(level["outer_order"] == order and level["private_inner_order"] == 256, "density rule identity differs")
            for name in ("KL_q0_to_q", "second_ratio_moment", "q_mass", "q0_mass"):
                arr = np.asarray(level[name], float)
                require(arr.shape == (18,) and np.isfinite(arr).all(), "all18 finite density states required: " + name)
        require(isinstance(doc["ordinary_diagnostic_qualified"], bool), "explicit density qualification required")
        require(doc.get("never_output_inclusion_rule") is True, "density diagnostic must not filter outputs")
        cell.update(status="available", ordinary_diagnostic_qualified=doc["ordinary_diagnostic_qualified"],
                    qualification=doc.get("qualification"), high_rule=doc["levels"][-1],
                    order_differences=doc["outer_order_max_differences"], mass_errors=doc["high_rule_mass_errors"])
    except (KeyError, ValueError, TypeError, IndexError) as error:
        cell["errors"].append("density metadata invalid: " + str(error))
    return cell


def cost_summary(records, loader, producers):
    totals, phases, errors = {}, [], []
    seen_roots = set()
    for record in records:
        phase_id, path = record.get("phase_id"), record.get("path")
        row = {"phase_id": phase_id, "source": record, "status": "unresolved"}
        phases.append(row)
        try:
            require(isinstance(phase_id, str) and phase_id, "explicit phase_id required")
            info = producers.get(record.get("producer_id"))
            require(info is not None and not info["errors"], "verified producer identity required for terminal cost receipt")
            terminal = resolve(path)
            require(terminal.parent == resolve(info["root"]) and terminal.name in ("SUMMARY.json", "FAILED.json"),
                    "only root SUMMARY.json or FAILED.json may supply terminal cost")
            run_root = canonical(info["root"])
            require(run_root not in seen_roots, "same canonical producer run already counted, regardless of phase_id/path spelling")
            in_manifest(info, record)
            doc, issue = loader.load(record, "phase_receipt " + phase_id)
            errors.extend(issue)
            require(doc is not None and not issue, "terminal receipt identity unavailable")
            require(doc.get("phase") == info["phase"], "terminal receipt phase differs from producer")
            if terminal.name == "SUMMARY.json":
                require(doc.get("status") == "completed", "SUMMARY is not a completed terminal phase")
            budget = doc["budget"]
            counts = budget["counts"]
            require(isinstance(counts, dict) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()),
                    "nonnegative integer work counters required")
            seconds = budget["seconds"]
            require(isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0, "finite phase duration required")
            if "expected_phase" in record:
                require(doc.get("phase") == record["expected_phase"], "receipt phase mismatch")
            for name, value in counts.items():
                totals[name] = totals.get(name, 0) + value
            seen_roots.add(run_root)
            row.update(status="counted_once", producer_id=info["id"], canonical_run_root=run_root,
                       phase=doc.get("phase"), execution_status=doc.get("status"), budget=budget)
        except (ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
            errors.append("phase receipt invalid " + phase_id + ": " + str(error))
    process_seconds = sum(row["budget"]["seconds"] for row in phases if row["status"] == "counted_once")
    expected_roots = {canonical(info["root"]) for info in producers.values() if isinstance(info.get("root"), str) and info["root"]}
    missing_roots = sorted(expected_roots - seen_roots)
    errors.extend("no verified unique terminal cost receipt for declared producer run: " + root for root in missing_roots)
    return {"counts": totals, "phases": phases, "errors": errors, "receipt_count": len(records),
            "expected_declared_producer_runs": len(expected_roots), "missing_terminal_run_roots": missing_roots,
            "unique_phase_receipts_counted": sum(row["status"] == "counted_once" for row in phases),
            "sum_process_phase_duration_seconds": process_seconds, "overall_elapsed_wall_seconds": None,
            "duration_scope": "Sum of per-process phase durations is not elapsed wall time when phases run concurrently.",
            "counting_scope": "Explicit terminal phase receipts only, each once; no nested checkpoints/COMPLETE snapshots summed."}


def frequencies(items, field):
    out = {}
    for item in items:
        value = str(item[field])
        out[value] = out.get(value, 0) + 1
    return out


def assemble(policy, models, coefficient_entries, contrast_entries, density_entries, costs, notes):
    critical = critical_value()
    coeff = {key: coefficient_cell(key, missing_entry(coefficient_entries, key, "coefficient"),
                                   missing_entry(models, key, "proposal")) for key in proposal_keys()}
    primary, raw, lookup = [], [], {}
    for key in proposal_keys():
        for tval in (policy["sym_t"] if key[2] == "sym" else policy["raw_t"]):
            ckey = (*key, tval)
            cell = contrast_cell(ckey, missing_entry(contrast_entries, ckey, "contrast"), coeff[key], policy, critical)
            lookup[ckey] = cell
            (primary if key[2] == "sym" else raw).append(cell)
    odd_even = [odd_even_cell(rep, n, tval, lookup[(rep, n, "raw", tval)], lookup[(rep, n, "raw", -tval)],
                             coeff[(rep, n, "raw")], policy)
                for rep in range(3) for n in SIZES for tval in (.1, .2)]
    density = [density_cell(key, missing_entry(density_entries, key, "density diagnostic")) for key in proposal_keys()]
    n_complete = sum(cell["status"] == "available" for cell in primary)
    other_unresolved = (any(cell["status"] != "available" for cell in [*raw, *coeff.values(), *density])
                        or bool(notes) or bool(costs.get("errors")))
    status = "completed_with_unresolved_primary_slots" if n_complete < 12 else (
        "completed_with_unresolved_inputs" if other_unresolved else "completed")
    summary = {"schema": SCHEMA, "status": status,
               "expected_primary_cells": 12, "expected_primary_quantities": FAMILY_SIZE,
               "available_primary_cells": n_complete, "unresolved_primary_quantities": 2 * (12 - n_complete),
               "primary_family_size_never_reduced": FAMILY_SIZE, "family_alpha": FAMILY_ALPHA, "df": DF,
               "critical_value": critical, "candidate_direction_counts": frequencies(primary, "D_direction"),
               "prediction_precision_counts": frequencies(primary, "prediction_precision"),
               "coefficient_status_counts": frequencies(list(coeff.values()), "status"),
               "raw_contrast_status_counts": frequencies(raw, "status"), "raw_odd_even_status_counts": frequencies(odd_even, "status"),
               "density_status_counts": frequencies(density, "status"),
               "density_qualified_counts": {"true": sum(row["ordinary_diagnostic_qualified"] is True for row in density),
                                            "false": sum(row["ordinary_diagnostic_qualified"] is False for row in density),
                                            "unresolved": sum(row["ordinary_diagnostic_qualified"] is None for row in density)},
               "sym_positive_estimated_h_ratio_count": sum(row.get("response_ratio") is not None for row in coeff.values()),
               "input_notes": notes, "costs": costs, "limitations": LIMITATIONS,
               "producer_provenance_status_counts": frequencies(costs.get("producer_provenance", []), "status"),
               "completion_status_scope": "Complete expected primary/raw/coefficient/density slots plus recorded provenance and cost identities; density qualification failure itself never excludes an output.",
               "conditional_numerical_uncertainty_not_training_population": True,
               "independence_basis": "SHA-bound producer CONTRACT/FREEZE/MANIFEST, common proposal/coefficient freeze links, actual BANK/COMPLETE proposal records and all16 BANK/RNG entropy tuples; sample correlation is not evidence of independence",
               "residual_SE": "sd((D_r-P_r)/t^2, ddof=1)/sqrt16, preserving paired index and incidental cross-stream covariance",
               "manuscript_modified": False, "no_new_science_executed": True}
    return {"summary": summary, "primary_cells": primary, "raw_cells": raw,
            "coefficient_cells": list(coeff.values()), "raw_odd_even": odd_even, "density_diagnostics": density,
            "policy": policy, "all_expected_cells_retained": True}


def flattened_coefficients(cells):
    rows = []
    for cell in cells:
        row = {k: cell[k] for k in ("rep", "N", "kind", "status")}
        row["errors"] = " | ".join(cell["errors"])
        if cell["status"] == "available":
            for name in ALL_KEYS:
                for statistic in ("mean", "SE"):
                    arr = np.asarray(cell["statistics"][name][statistic])
                    for index in np.ndindex(arr.shape):
                        row[name + "_" + statistic + "_" + "".join(map(str, index))] = float(arr[index])
            for name in ("h", "b"):
                row[name + "_mean"], row[name + "_SE"] = float(cell[name]["mean"]), float(cell[name]["SE"])
            ratio = cell.get("response_ratio")
            row["sym_response_ratio"] = ratio["estimate"] if ratio else None
            row["sym_response_ratio_delta_SE"] = ratio["delta_SE"] if ratio else None
        rows.append(row)
    return rows


def flattened_contrasts(cells):
    rows = []
    for cell in cells:
        row = {k: cell[k] for k in ("rep", "N", "kind", "t", "status", "D_direction", "prediction_precision")}
        row["errors"] = " | ".join(cell["errors"])
        if cell["status"] == "available":
            for name in ("D", "raw_uncontrolled_D", "prediction", "residual", "normalized_residual"):
                for statistic in ("mean", "SE"):
                    row[name + "_" + statistic] = float(cell[name][statistic])
            for name in ("D", "normalized_residual"):
                interval = cell.get(name + "_simultaneous_interval")
                if interval:
                    row[name + "_lower"], row[name + "_upper"] = interval["lower"], interval["upper"]
            if cell["kind"] == "sym":
                row.update(family_size=FAMILY_SIZE, df=DF)
        rows.append(row)
    return rows


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_contract(cfg, loader):
    require(cfg.get("schema") == CONTRACT_SCHEMA and cfg.get("frozen") is True and cfg.get("authorized_to_execute") is True,
            "frozen authorized saved-analysis contract required")
    require(len(cfg.get("baseline_commit", "")) == 40, "baseline commit identity required")
    require(cfg.get("source_sha256") == {SOURCE: sha(REPO / SOURCE)}, "analysis source identity differs")
    policy, errors = loader.load(cfg["policy"], "analysis_policy")
    require(not errors and policy is not None, "analysis policy identity unavailable: " + repr(errors))
    policy_check(policy)
    plan = cfg["plan"]
    require(plan["sha256"] == policy.get("plan_sha256") and sha(resolve(plan["path"])) == plan["sha256"],
            "policy PLAN SHA must match the explicitly bound actual PLAN file")
    loader.audit.append({"purpose": "frozen_plan_identity", "record": plan, "status": "verified",
                         "actual_sha256": plan["sha256"], "scope": "text identity only, no execution"})
    for name in ("proposal_freeze", "coefficient_freeze", "qa_index"):
        require(isinstance(cfg.get(name), dict), name + " binding required")
    qa, errors = loader.load(cfg["qa_index"], "common_qa_index")
    require(qa is not None and not errors and qa.get("passed") is True, "bound successful shared QA index required")
    for name in ("validation_summaries", "phase_receipts", "density_diagnostics", "producer_runs"):
        require(isinstance(cfg.get(name), list), name + " explicit list required")
    closure = cfg["producer_source_sha256"]
    require(isinstance(closure, dict) and PRODUCER_SOURCES <= set(closure), "fixed complete scientific producer source closure required")
    require(all(isinstance(digest, str) and len(digest) == 64 for digest in closure.values()), "producer source SHA256 identities required")
    require(all(isinstance(record, dict) for record in cfg["producer_runs"]), "producer records must be objects")
    prov = cfg["stream_provenance"]
    require(prov.get("coefficient_family") == 202610165301 and prov.get("coefficient_role") == 30
            and prov.get("validation_family") == 202610165401 and prov.get("validation_role") == 40,
            "fixed independent numerical stream identities required")
    require(prov.get("independent_coefficient_validation_streams") is True
            and prov.get("all_coefficients_frozen_before_validation") is True
            and prov.get("pairing") == "randomization_index_0_to_15", "explicit stream/freeze/index provenance required")
    return policy


def run(contract_path, out, check=False):
    cfg = read(contract_path)
    loader = BoundJSON()
    policy = validate_contract(cfg, loader)
    if check:
        inputs = load_inputs(cfg, loader)
        identity_failures = any(item["status"] != "verified" for item in loader.audit)
        semantic_failures = any(entry["errors"] for index in inputs[:4] for entry in index.values())
        cost_failures = bool(inputs[4]["errors"])
        wanted = set(proposal_keys())
        wanted_contrasts = {(*key, tval) for key in wanted for tval in ([.1, .2] if key[2] == "sym" else [-.2, -.1, .1, .2])}
        schedule_complete = (set(inputs[0]) == wanted and set(inputs[1]) == wanted and set(inputs[2]) == wanted_contrasts
                             and set(inputs[3]) == wanted)
        return {"schema": SCHEMA, "source_policy_contract_valid": True, "json_identity_counts": frequencies(loader.audit, "status"),
                "all_declared_input_provenance_valid": schedule_complete and not (identity_failures or semantic_failures or cost_failures or inputs[-1]),
                "complete_expected_input_schedule": schedule_complete,
                "coefficient_records_with_provenance_errors": sum(bool(entry["errors"]) for entry in inputs[1].values()),
                "contrast_records_with_provenance_errors": sum(bool(entry["errors"]) for entry in inputs[2].values()),
                "producer_provenance_status_counts": frequencies(inputs[4].get("producer_provenance", []), "status"),
                "input_notes": inputs[-1], "no_scientific_statistics_recomputed": True, "no_science_executed": True}
    out = Path(out).resolve()
    require(out.is_relative_to(REPO.resolve()), "fresh output directory must be inside repository")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(contract_path, out / "CONTRACT.json")
    shutil.copyfile(REPO / SOURCE, out / "SOURCE.py")
    require(sha(out / "SOURCE.py") == cfg["source_sha256"][SOURCE], "source changed during freeze")
    write(out / "FREEZE.json", {"utc": utc(), "contract_sha256": sha(contract_path), "source_sha256": cfg["source_sha256"],
                               "policy": cfg["policy"], "before_saved_statistics_arithmetic": True, "no_model_execution": True})
    try:
        inputs = load_inputs(cfg, loader)
        result = assemble(policy, *inputs)
        result["stream_provenance"] = cfg["stream_provenance"]
        result["input_audit"] = loader.audit
        result["bound_json_input_archive"] = loader.documents
        result["nonfinite_archive_encoding"] = "nonfinite inputs are retained as {nonfinite_input_value: string}; unresolved cells are not imputed"
        result["summary"]["json_identity_counts"] = frequencies(loader.audit, "status")
        result["summary"]["analysis_utc"] = utc()
        write(out / "ANALYSIS.json", result)
        write(out / "SUMMARY.json", result["summary"])
        write_csv(out / "PRIMARY_CELLS.csv", flattened_contrasts(result["primary_cells"]))
        write_csv(out / "RAW_CELLS.csv", flattened_contrasts(result["raw_cells"]))
        write_csv(out / "COEFFICIENT_CELLS.csv", flattened_coefficients(result["coefficient_cells"]))
        write(out / "INPUT_AUDIT.json", loader.audit)
        write(out / "MANIFEST.json", {"schema": SCHEMA, "files": [
            {"path": path.relative_to(out).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(out.rglob("*")) if path.is_file()], "manifest_self_excluded": True})
        return result["summary"]
    except Exception:
        write(out / "FAILED.json", {"utc": utc(), "traceback": traceback.format_exc(), "input_audit": loader.audit,
                                    "automatic_retry": False, "no_scientific_execution": True})
        raise


def template():
    empty = {"path": "", "sha256": ""}
    runs = []
    for phase, rep in [("qa", None), ("training", None), ("profile", None)] + [(p, r) for p in ("coefficients", "validation") for r in range(3)]:
        run = {"id": phase if rep is None else phase + "_rep" + str(rep), "phase": phase, "root": "",
               "contract": dict(empty), "freeze": dict(empty), "manifest": dict(empty)}
        if phase in ("coefficients", "validation"):
            run["outputs"] = [{"rep": rep, "N": n, "kind": kind, "directory": "",
                               "bank_complete": dict(empty), "bank_rng": dict(empty)} for n in SIZES for kind in KINDS]
        runs.append(run)
    return {"schema": CONTRACT_SCHEMA, "frozen": True, "authorized_to_execute": True,
            "baseline_commit": "5cd5559f88aa8d8f779fc49cc93d1bfd7ead983b",
            "source_sha256": {SOURCE: sha(REPO / SOURCE)},
            "policy": {"path": "artifacts/learned_local_proposal_v1_20261016/ANALYSIS_POLICY_001.json", "sha256": ""},
            "plan": {"path": "docs/08_research_研究工作/LearnedLocalProposal_v1_20261016/PLAN.md", "sha256": ""},
            "qa_index": dict(empty),
            "producer_source_sha256": {name: "" for name in sorted(PRODUCER_SOURCES)},
            "producer_runs": runs,
            "proposal_freeze": dict(empty), "coefficient_freeze": dict(empty),
            "validation_summaries": [dict(empty) for _ in range(3)],
            "phase_receipts": [{**empty, "phase_id": run["id"], "producer_id": run["id"], "expected_phase": run["phase"]} for run in runs],
            "density_diagnostics": [{"rep": r, "N": n, "kind": k, **empty} for r, n, k in proposal_keys()],
            "stream_provenance": {"coefficient_family": 202610165301, "coefficient_role": 30,
                                  "validation_family": 202610165401, "validation_role": 40,
                                  "independent_coefficient_validation_streams": True,
                                  "all_coefficients_frozen_before_validation": True,
                                  "pairing": "randomization_index_0_to_15"}}


def manufactured_policy():
    return {"schema": "learned_local_analysis_policy_v1", "frozen": True,
            "training_repeats": [0, 1, 2], "training_N": list(SIZES), "proposal_kinds": list(KINDS),
            "K": 8, "alpha": ALPHA, "v": [1/math.sqrt(2), 1/math.sqrt(2), 0.],
            "sym_t": [.1, .2], "raw_t": [-.2, -.1, .1, .2], "coefficient_R": R, "validation_R": R,
            "coefficient_B": 256, "validation_B": 512,
            "primary_family": {"size": FAMILY_SIZE, "alpha": FAMILY_ALPHA, "df": DF}}


def as_saved(arr, banks):
    result = sample_summary(arr)
    return {key: result[key] for key in ("R", "replicates", "mean", "SE")} | {"B": banks}


def fixtures():
    """Deterministic invented tensors; no actual channel/data/observations."""
    policy = manufactured_policy()
    u = (np.arange(R, dtype=float) - 7.5) / 100
    models, coefficients, contrasts = {}, {}, {}
    for key in proposal_keys():
        kind = key[2]
        g = np.column_stack([.1 + u, -.02 - .3*u, .03 + .7*u]) if kind == "raw" else np.zeros((R, 3))
        h = np.broadcast_to(np.eye(3), (R, 3, 3)).copy()
        b = .2*h.copy()
        h += u[:, None, None]*np.eye(3)
        b += .3*u[:, None, None]*np.eye(3)
        if kind == "raw":
            h[:, 0, 1] = h[:, 1, 0] = .4 + .2*u
            b[:, 0, 1] = b[:, 1, 0] = -.1 + .1*u
        values = {"raw_g": g+.01, "cv_g": g, "raw_H": h+.02, "cv_H": h,
                  "raw_B": b+.03, "cv_B": b}
        doc = {"kind": kind, "control": "uniform_logratio_actual_mixture_analytic_moments",
               "statistics": {name: as_saved(arr, 256) for name, arr in values.items()}}
        models[key] = {"document": {"manufactured": True}, "source": None, "errors": []}
        coefficients[key] = {"document": doc, "source": None, "errors": []}
        ccell = coefficient_cell(key, coefficients[key], models[key])
        linear, quadratic = prediction_parts(ccell, policy)
        for tval in (policy["sym_t"] if kind == "sym" else policy["raw_t"]):
            # Arbitrary deterministic arrays have no stochastic-independence
            # assertion. Their nonzero sample covariance distinguishes the
            # prescribed index-paired SE from a different variance estimator.
            d = tval*linear + tval**2*quadratic + tval**2*(.002 + u[::-1]/10)
            v = np.asarray(policy["v"])
            contrast = {"t": tval, "alpha": ALPHA, "candidate_coordinates": [ALPHA*tval*v, tval*v],
                        "pairing": "same_q_particles_and_same_teacher_PMF; shrink_candidate_minus_teacher_candidate",
                        "statistics": as_saved(d, 512), "raw_statistics": as_saved(d+.003, 512)}
            contrasts[(*key, tval)] = {"document": contrast, "source": None, "errors": []}
    return policy, models, coefficients, contrasts


class ManufacturedJSON:
    """In-memory, SHA-consistent metadata only; never touches actual results."""
    def __init__(self):
        self.documents = {}

    def add(self, path, document):
        import copy
        document = copy.deepcopy(document)
        digest = hashlib.sha256(json.dumps(safe(document), sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.documents[path] = (document, digest)
        return {"path": path, "sha256": digest}

    def load(self, binding, purpose):
        import copy
        entry = self.documents.get(binding["path"])
        if entry is None or entry[1] != binding["sha256"]:
            return None, [purpose + ": manufactured metadata identity mismatch"]
        return copy.deepcopy(entry[0]), []


def provenance_fixture():
    """A complete invented six-producer metadata graph, without numerical data generation."""
    import copy
    _, models, coefficients, contrasts = fixtures()
    loader = ManufacturedJSON()
    base = "artifacts/MANUFACTURED_ONLY_NOT_WRITTEN"
    source_map = {name: "e"*64 for name in sorted(PRODUCER_SOURCES)}
    cfg = {"baseline_commit": "a"*40, "producer_source_sha256": source_map, "producer_runs": []}
    cfg["qa_index"] = loader.add(base+"/qa/INDEX.json", {"passed": True})
    pairs = []
    for rep in range(3):
        for n in SIZES:
            raw = {"weights": [[.5, .5]], "means": [[0., 1.]], "stds": [[1., 1.]]}
            pair = {"rep": rep, "N": n, "selected_epoch": 20, "defense": .05, "raw_parameters": raw, "proposals": {}}
            for kind in KINDS:
                key = (rep, n, kind)
                model = {"schema": "learned_local_proposal_v1", "kind": kind, "defense": .05,
                         "raw_parameters": raw, "parameters": copy.deepcopy(raw), "shared_homogeneous_groups": 3,
                         "construction": "manufactured_"+kind, "source": {"rep": rep, "N": n, "epoch": 20}}
                binding = loader.add(base+f"/training/rep_{rep}_N{n}/PROPOSAL_{kind}.json", model)
                models[key] = {"document": model, "errors": [], "source": {"binding": binding, "selected_epoch": 20}}
                pair["proposals"][kind] = binding
            pairs.append(pair)
    pfreeze = {"all_training_complete": True, "pairs": pairs}
    cfg["proposal_freeze"] = loader.add(base+"/training/FROZEN_PROPOSALS.json", pfreeze)
    coefficient_rows = []
    for key, entry in coefficients.items():
        rep, n, kind = key
        directory = base+f"/coefficients_{rep}/rep_{rep}_N{n}_{kind}"
        binding = loader.add(directory+"/COEFFICIENT_STATISTICS.json", entry["document"])
        record = {"rep": rep, "N": n, "kind": kind, "statistics": binding["path"], "statistics_sha256": binding["sha256"],
                  "banks": directory+"/COEFFICIENT_BANKS.npz", "banks_sha256": "b"*64,
                  "role": "independent_coefficient_evaluation"}
        entry["source"] = record
        coefficient_rows.append(record)
    cfreeze = {"all_coefficients_complete": True, "outputs": coefficient_rows}
    cfg["coefficient_freeze"] = loader.add(base+"/FROZEN_COEFFICIENTS.json", cfreeze)
    for phase in ("coefficients", "validation"):
        for rep in range(3):
            root = base+f"/{phase}_{rep}"
            mapping = {cfg[name]["path"]: cfg[name]["sha256"] for name in ("qa_index", "proposal_freeze")}
            for pair in pairs:
                for binding in pair["proposals"].values():
                    mapping[binding["path"]] = binding["sha256"]
            family, role, banks = (202610165301, 30, 256) if phase == "coefficients" else (202610165401, 40, 512)
            contract = {"schema": "learned_local_execution_v1", "phase": phase, "frozen": True, "authorized_to_execute": True,
                        "baseline_commit": cfg["baseline_commit"], "source_sha256": source_map, "input_sha256": mapping,
                        "case_ids": [rep], "K": 8, "qa_index": cfg["qa_index"]["path"],
                        "proposal_freeze": cfg["proposal_freeze"]["path"],
                        "randomization": {"family": family, "role": role, "R": R, "B": banks, "bits": 52}}
            if phase == "validation":
                contract.update(coefficient_freeze=cfg["coefficient_freeze"]["path"], alpha=ALPHA,
                                sym_t=[.1, .2], raw_t=[-.2, -.1, .1, .2])
                mapping[cfg["coefficient_freeze"]["path"]] = cfg["coefficient_freeze"]["sha256"]
                for row in coefficient_rows:
                    for name in ("statistics", "banks"):
                        mapping[row[name]] = row[name+"_sha256"]
            cb = loader.add(root+"/CONTRACT.json", contract)
            fb = loader.add(root+"/FREEZE.json", {"schema": "learned_local_execution_v1", "phase": phase,
                            "contract_sha256": cb["sha256"], "source_sha256": source_map, "input_sha256": mapping,
                            "before_scientific_calls": True})
            run = {"id": phase+str(rep), "phase": phase, "root": root, "contract": cb, "freeze": fb, "outputs": []}
            members = [cb, fb]
            summary_outputs = []
            for n in SIZES:
                for kind in KINDS:
                    key = (rep, n, kind)
                    directory = root+f"/rep_{rep}_N{n}_{kind}"
                    actual = copy.deepcopy(models[key]["document"])
                    actual["source"] = {"rep": rep, "N": n, "selected_epoch": 20}
                    complete = loader.add(directory+"/BANK/COMPLETE.json", {"proposal": actual})
                    rng = {"family": family, "role": role, "R": R, "B": banks, "K": 8, "kind": kind,
                           "states": [{"randomization": r, "entropy": [family, role, rep, n, 0 if kind == "raw" else 1, r],
                                       "rng_after": {"bit_generator": "PCG64DXSM"}} for r in range(R)]}
                    rb = loader.add(directory+"/BANK/RNG.json", rng)
                    run["outputs"].append({"rep": rep, "N": n, "kind": kind, "directory": directory,
                                           "bank_complete": complete, "bank_rng": rb})
                    members += [complete, rb]
                    if phase == "coefficients":
                        row = coefficients[key]["source"]
                        members.append({"path": row["statistics"], "sha256": row["statistics_sha256"]})
                        summary_outputs.append(row)
                    else:
                        rows = []
                        for tval in ([.1, .2] if kind == "sym" else [-.2, -.1, .1, .2]):
                            ckey = (*key, tval)
                            basename = "CONTRAST_t"+str(tval).replace("-", "m").replace(".", "p")
                            binding = loader.add(directory+"/"+basename+".json", contrasts[ckey]["document"])
                            row = {"t": tval, "statistics": binding["path"], "statistics_sha256": binding["sha256"],
                                   "banks": directory+"/"+basename+".npz", "banks_sha256": "b"*64}
                            members.append(binding)
                            rows.append(row)
                            contrasts[ckey]["source"] = {"output": {"rep": rep, "N": n, "kind": kind}, "contrast": row}
                        summary_outputs.append({"rep": rep, "N": n, "kind": kind, "contrasts": rows})
            summary = loader.add(root+"/SUMMARY.json", {"phase": phase, "status": "completed", "outputs": summary_outputs,
                                "budget": {"counts": {"terms": 7}, "seconds": 3.}})
            members.append(summary)
            if phase == "validation":
                for ckey, entry in contrasts.items():
                    if ckey[0] == rep:
                        entry["source"]["summary"] = summary
            files = [{"path": Path(binding["path"]).relative_to(root).as_posix(), "sha256": binding["sha256"]} for binding in members]
            files += [{"path": "SEALED_SOURCE/"+name, "sha256": digest} for name, digest in source_map.items()]
            run["manifest"] = loader.add(root+"/MANIFEST.json", {"files": files, "manifest_self_excluded": True})
            cfg["producer_runs"].append(run)
    return cfg, loader, models, coefficients, contrasts, pfreeze, cfreeze


def self_test():
    tests = []
    policy, models, coefficients, contrasts = fixtures()
    policy_check(policy)
    cv = critical_value()
    require(abs(float(student_t.sf(cv, DF)) - FAMILY_ALPHA/(2*FAMILY_SIZE)) < 2e-14, "family24 two-sided tail")
    require(cv > float(student_t.ppf(1-.05/(2*12), DF)), "family24 cannot be replaced by12")
    tests.append("family24_critical_and_tail")
    result = assemble(policy, models, coefficients, contrasts, {}, {}, [])
    require(len(result["primary_cells"]) == 12 and len(result["raw_cells"]) == 24 and len(result["coefficient_cells"]) == 12,
            "complete fixed schedule")
    cell = result["primary_cells"][0]
    expected_w = (cell["D"]["replicates"] - cell["prediction"]["replicates"])/cell["t"]**2
    require(np.array_equal(expected_w, cell["normalized_residual"]["replicates"]), "residual pairing identity")
    require(abs(float(cell["normalized_residual"]["SE"]) - float(expected_w.std(ddof=1)/4)) < 1e-16, "residual exact index SE")
    separate = math.sqrt(float(cell["D"]["SE"])**2 + float(cell["prediction"]["SE"])**2)/cell["t"]**2
    require(abs(separate-float(cell["normalized_residual"]["SE"])) > 1e-4, "test must distinguish paired-index and separate variance SE")
    tests.append("index_paired_residual_SE_not_Welch")
    reduced = dict(contrasts)
    reduced.pop((0, 16384, "sym", .1))
    incomplete = assemble(policy, models, coefficients, reduced, {}, {}, [])
    require(len(incomplete["primary_cells"]) == 12 and incomplete["summary"]["unresolved_primary_quantities"] == 2,
            "missing slot retained")
    require(incomplete["summary"]["primary_family_size_never_reduced"] == 24 and incomplete["summary"]["critical_value"] == cv,
            "missing does not change multiplicity")
    tests.append("missing_slot_preserves_family24")
    c = result["coefficient_cells"][0]
    _, q = prediction_parts(c, policy)
    hs = c["h"]["replicates"]; bs = c["b"]["replicates"]
    diagonal_only = .5*(ALPHA**2-1)*hs - (ALPHA-1)*bs
    require(np.max(abs(q-diagonal_only)) > .1, "raw offdiagonal terms essential")
    tests.append("raw_full_H_B_offdiagonal_prediction")
    odd_even = result["raw_odd_even"][0]
    rawlookup = {(row["rep"], row["N"], row["t"]): row for row in result["raw_cells"]}
    plus = rawlookup[(0, 16384, .1)]["D"]["replicates"]
    minus = rawlookup[(0, 16384, -.1)]["D"]["replicates"]
    require(np.array_equal(odd_even["statistics"]["odd_coefficient"]["replicates"], (plus-minus)/.2), "odd paired indices")
    require(np.allclose(odd_even["statistics"]["even_coefficient"]["replicates"], (plus+minus)/.02, rtol=0, atol=1e-14), "even paired indices")
    require(np.asarray(odd_even["covariance"]["sample_covariance"]).shape == (10, 10), "complete odd/even covariance")
    tests.append("raw_odd_even_pairing_and_covariance")
    import copy
    bad = copy.deepcopy(contrasts)
    bad[(0, 16384, "sym", .1)]["document"]["statistics"]["R"] = 15
    bad[(0, 16384, "sym", .2)]["document"]["statistics"]["replicates"][0] = np.nan
    invalid = assemble(policy, models, coefficients, bad, {}, {}, [])
    require(invalid["summary"]["unresolved_primary_quantities"] == 4, "wrongR/nonfinite slots retained")
    tests.append("wrong_R_and_nonfinite_remain_unresolved")
    sym = next(row for row in result["coefficient_cells"] if row["kind"] == "sym")
    ratio = sym["response_ratio"]
    h, b = sym["h"]["replicates"], sym["b"]["replicates"]
    expected_delta = ((b-ratio["estimate"]*h)/h.mean()).std(ddof=1)/4
    require(abs(ratio["delta_SE"]-expected_delta) < 1e-16 and c["response_ratio"] is None, "sym ratio delta and raw not interpreted")
    tests.append("sym_ratio_delta_SE_and_raw_no_response_claim")
    neg = copy.deepcopy(coefficients[(0, 16384, "sym")])
    neg["document"]["statistics"]["cv_H"] = as_saved(-np.broadcast_to(np.eye(3), (R, 3, 3)).copy(), 256)
    ncell = coefficient_cell((0, 16384, "sym"), neg, models[(0, 16384, "sym")])
    require(ncell["status"] == "available" and ncell["response_ratio"] is None, "nonpositive h retained without ratio")
    tests.append("nonpositive_estimated_h_not_removed_or_divided")
    cfg, loader, pmodels, pcoeff, pcontrast, pfreeze, cfreeze = provenance_fixture()
    producers, notes = provenance_inputs(cfg, loader, pmodels, pcoeff, pcontrast, {}, pfreeze, cfreeze)
    require(not notes and all(not row["errors"] for row in pcoeff.values())
            and all(not row["errors"] for row in pcontrast.values()), "complete manufactured producer graph must verify")
    tests.append("complete_producer_contract_freeze_manifest_proposal_RNG_chain")
    info = producers["coefficients0"]
    output = cfg["producer_runs"][0]["outputs"][0]
    bad_output = copy.deepcopy(output)
    complete_doc, _ = loader.load(output["bank_complete"], "fixture")
    complete_doc["proposal"]["raw_parameters"]["means"][0][0] = .123
    # Keep the expected path while changing both the bound SHA and manifest
    # SHA, so this catches the semantic model mismatch rather than a hash bug.
    bad_output["bank_complete"] = loader.add(output["bank_complete"]["path"], complete_doc)
    bad_info = copy.deepcopy(info)
    bad_info["manifest_index"][canonical(bad_output["bank_complete"]["path"])] = bad_output["bank_complete"]["sha256"]
    issues = verify_bank(bad_info, bad_output, (0, 16384, "raw"), pmodels[(0, 16384, "raw")], 20, loader)
    require(any("raw_parameters" in issue for issue in issues), "SHA-valid wrong actual proposal must fail")
    tests.append("wrong_actual_proposal_rejected_despite_consistent_file_SHA")
    # Restore a fresh graph for the independent wrong-seed test.
    cfg, loader, pmodels, pcoeff, pcontrast, pfreeze, cfreeze = provenance_fixture()
    info = producer_metadata(cfg, cfg["producer_runs"][0], loader)
    output = copy.deepcopy(cfg["producer_runs"][0]["outputs"][0])
    rngdoc, _ = loader.load(output["bank_rng"], "fixture")
    rngdoc["states"][7]["entropy"][0] += 1
    output["bank_rng"] = loader.add(output["bank_rng"]["path"], rngdoc)
    info["manifest_index"][canonical(output["bank_rng"]["path"])] = output["bank_rng"]["sha256"]
    issues = verify_bank(info, output, (0, 16384, "raw"), pmodels[(0, 16384, "raw")], 20, loader)
    require(any("entropy" in issue for issue in issues), "SHA-valid wrong RNG entropy must fail")
    tests.append("wrong_actual_RNG_entropy_rejected_despite_consistent_file_SHA")
    cfg, loader, pmodels, pcoeff, pcontrast, pfreeze, cfreeze = provenance_fixture()
    run = cfg["producer_runs"][0]
    output = run["outputs"][1]  # sym/lowN: both primary t slots must fail closed.
    rngdoc, _ = loader.load(output["bank_rng"], "fixture")
    rngdoc["states"][4]["entropy"][3] += 1
    output["bank_rng"] = loader.add(output["bank_rng"]["path"], rngdoc)
    manifest_doc, _ = loader.load(run["manifest"], "fixture")
    rel = Path(output["bank_rng"]["path"]).relative_to(run["root"]).as_posix()
    next(item for item in manifest_doc["files"] if item["path"] == rel)["sha256"] = output["bank_rng"]["sha256"]
    run["manifest"] = loader.add(run["manifest"]["path"], manifest_doc)
    producers, pnotes = provenance_inputs(cfg, loader, pmodels, pcoeff, pcontrast, {}, pfreeze, cfreeze)
    rejected = assemble(policy, pmodels, pcoeff, pcontrast, {}, {}, pnotes)
    affected = [row for row in rejected["primary_cells"] if row["rep"] == 0 and row["N"] == 16384]
    require(len(affected) == 2 and all(row["status"] == "unresolved" for row in affected)
            and rejected["summary"]["status"] != "completed", "source failure must propagate into unavailable cells")
    tests.append("provenance_failure_prevents_available_or_complete_success")
    cfg, loader, pmodels, pcoeff, pcontrast, pfreeze, cfreeze = provenance_fixture()
    run = copy.deepcopy(cfg["producer_runs"][0])
    frozen, _ = loader.load(run["freeze"], "fixture")
    frozen["contract_sha256"] = "0"*64
    run["freeze"] = loader.add(run["freeze"]["path"], frozen)
    manifest_doc, _ = loader.load(run["manifest"], "fixture")
    next(item for item in manifest_doc["files"] if item["path"] == "FREEZE.json")["sha256"] = run["freeze"]["sha256"]
    run["manifest"] = loader.add(run["manifest"]["path"], manifest_doc)
    badproducer = producer_metadata(cfg, run, loader)
    require(any("FREEZE contract SHA" in issue for issue in badproducer["errors"]), "wrong linked producer contract hash must fail")
    tests.append("wrong_producer_FREEZE_contract_hash_rejected")
    cfg, loader, pmodels, pcoeff, pcontrast, pfreeze, cfreeze = provenance_fixture()
    producers, _ = provenance_inputs(cfg, loader, pmodels, pcoeff, pcontrast, {}, pfreeze, cfreeze)
    records = []
    for identifier in ("coefficients0", "coefficients1"):
        info = producers[identifier]
        path = info["root"]+"/SUMMARY.json"
        records.append({"phase_id": identifier, "producer_id": identifier, "path": path,
                        "sha256": info["manifest_index"][canonical(path)]})
    alias = {**records[0], "phase_id": "renamed", "path": producers["coefficients0"]["root"]+"/./SUMMARY.json"}
    bad_snapshot = {**records[0], "phase_id": "nested", "path": producers["coefficients0"]["root"]+"/rep_0_N16384_raw/COMPLETE.json"}
    cost_producers = {name: producers[name] for name in ("coefficients0", "coefficients1")}
    cost = cost_summary([*records, alias, bad_snapshot], loader, cost_producers)
    require(cost["counts"]["terms"] == 14 and cost["unique_phase_receipts_counted"] == 2 and cost["overall_elapsed_wall_seconds"] is None,
            "cost receipts once, sum not elapsed wall")
    require(len(cost["errors"]) == 2, "alias and nested cumulative snapshot both rejected")
    tests.append("canonical_root_terminal_cost_only_no_alias_or_nested_COMPLETE")
    incomplete_cost = cost_summary(records[:1], loader, cost_producers)
    require(len(incomplete_cost["missing_terminal_run_roots"]) == 1 and incomplete_cost["errors"],
            "every declared producer requires a unique terminal receipt")
    tests.append("missing_declared_producer_terminal_cost_detected")
    return {"schema": SCHEMA, "status": "passed", "tests": tests, "test_count": len(tests),
            "critical_value": cv, "manufactured_deterministic_tensors_only": True,
            "scientific_RNG_model_integrals_optimizations": 0, "real_result_files_read": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.template:
        require(not (args.self_test and args.template), "choose one standalone mode")
        require(args.contract is None and args.out is None and not args.check, "standalone mode takes no contract/output")
        result = self_test() if args.self_test else template()
    else:
        if args.contract is None or (not args.check and args.out is None):
            parser.error("--contract and --out required, or --contract with --check")
        result = run(args.contract.resolve(), args.out, args.check)
    print(json.dumps(safe(result), ensure_ascii=False, indent=2, allow_nan=False))
