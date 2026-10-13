"""Freeze and execute the single authorized, bounded R03 internal audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import r03_independent_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/r03_teacher_family_v1_20261011"
DOC = "docs/08_research_研究工作/R03TeacherFamily_v1_20261011/"
SOURCE_FILES = ["script/r03_audit_run.py", "script/r03_independent_audit.py",
                "script/r03_candidate_interval.py", DOC + "AUDIT_METHOD.md",
                DOC + "AUDIT_CONTRACT.md"] + list(audit.SOURCE_FILES)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / name).exists() for name in ("AUDIT_INPUTS.json", "AUDIT_FREEZE.json", "AUDIT_START.json", "AUDIT_RESULT.json")):
        raise FileExistsError("audit inputs/freeze/start/result already exists; no overwrite or new run")
    inputs = dict(schema="r03_independent_audit_run_inputs_v1", specification=audit.prepared_inputs(),
                  wall_seconds=120, run_count=1, blas_threads=1,
                  purpose="internal implementation and numerical verification at original a0; no extension of the certified lambda range",
                  certified_route_closure="|lambda|<=1/1000; no new continuation chain",
                  forbidden=["automatic retry", "rule change", "label selection change", "old source modification", "manuscript modification"],
                  public_python_flint_required=True)
    write_new(OUT / "AUDIT_INPUTS.json", inputs)
    names = SOURCE_FILES + ["artifacts/r03_teacher_family_v1_20261011/AUDIT_INPUTS.json"]
    lock = dict(schema="r03_independent_audit_freeze_v1", frozen_utc=now(),
                bindings={name: sha(ROOT / name) for name in names},
                audit_target_calls_before_freeze=0, scientific_run_authorized_count=1)
    write_new(OUT / "AUDIT_FREEZE.json", lock)
    print(json.dumps(dict(frozen=True, source_count=len(names), wall_seconds=120,
                          labels=list(audit.LABELS), teacher_orders=list(audit.ORDERS)), ensure_ascii=False))


def verify():
    lock = read(OUT / "AUDIT_FREEZE.json")
    expected_names = set(SOURCE_FILES + ["artifacts/r03_teacher_family_v1_20261011/AUDIT_INPUTS.json"])
    if lock.get("schema") != "r03_independent_audit_freeze_v1" or set(lock["bindings"]) != expected_names:
        raise ValueError("audit freeze schema or source list drift")
    for name, digest in lock["bindings"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("frozen audit source changed: " + name)
    inputs = read(OUT / "AUDIT_INPUTS.json")
    if (inputs.get("schema") != "r03_independent_audit_run_inputs_v1"
            or inputs["specification"] != audit.prepared_inputs()
            or inputs["wall_seconds"] != 120 or inputs["run_count"] != 1
            or inputs["blas_threads"] != 1):
        raise ValueError("frozen audit inputs drift")
    return inputs


def summary(result):
    teacher = []
    for run in result.get("teacher_runs", []):
        teacher.append(dict(order=run["order"], status=run["status"],
                            reference_L1={name: x["L1_worst_absolute_difference_display"]
                                          for name, x in run.get("strict_reference_comparisons", {}).items()}))
    point = []
    for row in result.get("candidate_comparisons", {}).get("rows", []):
        point.append(dict(label=row["label_index"], passed=row["passed"],
                          max_entry_errors={name: x["max_worst_absolute_difference_display"]
                                            for name, x in row["comparisons"].items()}))
    return dict(status=result["status"], passed=result["passed"], seconds=result["elapsed_seconds"],
                counts=result["actual_counts"], teacher=teacher,
                two_rule_L1={name: row["L1_display"] for name, row in result.get("teacher_two_rule_difference", {}).items()},
                point=point, old_new_q_overlap=result.get("old_teacher_interval_check", {}).get("all_256_overlap"),
                error=result.get("error"))


def run():
    inputs = verify()
    if any((OUT / name).exists() for name in ("AUDIT_IMPLEMENTATION_CHECK.json", "AUDIT_START.json", "AUDIT_RESULT.json")):
        raise FileExistsError("audit execution evidence already exists; no automatic retry")
    flint_file = Path(audit.candidate.flint.__file__).resolve()
    if flint_file.is_relative_to(ROOT) or ".cache" in flint_file.parts:
        raise RuntimeError("audit requires source-external public python-flint, not a repository cache")
    try:
        check = audit.self_test()
    except Exception as exc:
        write_new(OUT / "AUDIT_IMPLEMENTATION_CHECK.json", dict(passed=False, completed_utc=now(),
                    error=dict(type=type(exc).__name__, message=str(exc)), new_target_calls=0))
        raise
    check["completed_utc"] = now()
    check["freeze_sha256"] = sha(OUT / "AUDIT_FREEZE.json")
    write_new(OUT / "AUDIT_IMPLEMENTATION_CHECK.json", check)
    if check.get("passed") is not True:
        raise RuntimeError("audit implementation check failed; scientific targets not called")
    # Recheck the locked inputs after the zero-target self-test and immediately
    # before the exclusive start marker and the one permitted target call.
    verify()
    start = dict(schema="r03_independent_audit_start_v1", started_utc=now(),
                 freeze_sha256=sha(OUT / "AUDIT_FREEZE.json"), executable=sys.executable,
                 python_flint_file=str(flint_file), python_flint_version=audit.candidate.flint.__version__,
                 blas_environment={key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                 wall_seconds=120, authorized_run_index=1, automatic_retry=False)
    write_new(OUT / "AUDIT_START.json", start)
    result = audit.produce_audit(inputs["specification"], wall_seconds=inputs["wall_seconds"])
    result["run_provenance"] = dict(finished_utc=now(), freeze_sha256=start["freeze_sha256"],
                                     start_sha256=sha(OUT / "AUDIT_START.json"),
                                     implementation_check_sha256=sha(OUT / "AUDIT_IMPLEMENTATION_CHECK.json"))
    write_new(OUT / "AUDIT_RESULT.json", result)
    print(json.dumps(summary(result), ensure_ascii=False, indent=2, allow_nan=False))
    return bool(result["passed"])


def check_saved():
    verify()
    checks = read(OUT / "AUDIT_IMPLEMENTATION_CHECK.json")
    start = read(OUT / "AUDIT_START.json")
    result = read(OUT / "AUDIT_RESULT.json")
    if (checks.get("passed") is not True
            or start["freeze_sha256"] != sha(OUT / "AUDIT_FREEZE.json")
            or result["run_provenance"]["start_sha256"] != sha(OUT / "AUDIT_START.json")
            or result["run_provenance"]["implementation_check_sha256"] != sha(OUT / "AUDIT_IMPLEMENTATION_CHECK.json")):
        raise ValueError("audit saved provenance mismatch")
    print(json.dumps(dict(saved_evidence_valid=True, **summary(result)), ensure_ascii=False, indent=2, allow_nan=False))
    return bool(result["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(0 if (run() if args.run else check_saved()) else 2)
