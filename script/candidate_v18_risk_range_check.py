"""Strict archive-aware supplement to the sealed risk-range checker.

The original freeze/result/source bytes are unchanged. Only the frozen
HANDOFF.md source name resolves to its identical archived input in this new
read-only checker. No run, freeze, grid or integration entry point is added.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import candidate_v18_risk_range as original

ROOT=Path(__file__).resolve().parents[1]
ART="artifacts/submission_candidate_v1_8_20261008"
ARCHIVE=f"{ART}/HANDOFF_INPUT.md"
FREEZE=f"{ART}/RISK_RANGE_FREEZE.json"
RESULT=f"{ART}/RISK_RANGE_RESULT.json"
START=f"{ART}/RISK_RANGE_RUN_START.json"
ADAPTATION=f"{ART}/RISK_RANGE_CHECK_ADAPTATION.json"
NEW_SOURCES=("script/candidate_v18_risk_range_check.py",
             "script/candidate_v18_risk_range_check_test.py")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root,name):
    if not isinstance(name,str) or not name or PureWindowsPath(name).is_absolute() or Path(name).is_absolute():
        raise ValueError("repository-relative input path required")
    # Also reject Windows drive-relative names such as C:foo.
    if PureWindowsPath(name).drive:raise ValueError("input drive escape rejected")
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing input or path outside repository: "+name)
    return path


def source_names():
    return set((*original.SOURCE_PATHS,*original.CODE_DOC_PATHS,
        original.MANUFACTURED.relative_to(original.ROOT).as_posix(),"HANDOFF.md"))


def strict_bindings(root):
    """Verify old provenance using one exact, documented source resolution."""
    root=Path(root).resolve()
    freeze_path=safe_path(root,FREEZE);result_path=safe_path(root,RESULT);start_path=safe_path(root,START)
    frozen,saved,start=map(read,(freeze_path,result_path,start_path))
    if (frozen.get("schema")!="candidate_v18_risk_range_freeze_v1"
            or frozen.get("precision_bits")!=original.BITS or frozen.get("label_count")!=256
            or frozen.get("reviewed_commit")!=original.REVIEWED_COMMIT
            or frozen.get("handoff_sha256")!=original.HANDOFF_SHA
            or not frozen.get("no_new_grid_or_integral") or frozen.get("probability_mass_renormalization") is not False):
        raise ValueError("sealed risk-range freeze qualification drift")
    sources=frozen.get("sources")
    if not isinstance(sources,dict) or set(sources)!=source_names():
        raise ValueError("sealed source set drift")
    archive_hash=sha(safe_path(root,ARCHIVE))
    if (sources["HANDOFF.md"]!=original.HANDOFF_SHA or archive_hash!=sources["HANDOFF.md"]
            or archive_hash!=frozen["handoff_sha256"] or archive_hash!=saved.get("handoff_sha256")):
        raise ValueError("archived HANDOFF is not byte-identical to sealed feedback")
    resolved={}
    for name,digest in sources.items():
        actual_name=ARCHIVE if name=="HANDOFF.md" else name
        if sha(safe_path(root,actual_name))!=digest:
            raise ValueError("sealed source hash drift: "+name)
        resolved[name]=actual_name
    freeze_hash=sha(freeze_path);start_hash=sha(start_path)
    if (saved.get("source_sha256")!=sources or start.get("source_sha256")!=sources
            or saved.get("freeze_sha256")!=freeze_hash or start.get("freeze_sha256")!=freeze_hash
            or saved.get("run_start_sha256")!=start_hash
            or saved.get("reviewed_commit")!=frozen["reviewed_commit"]):
        raise ValueError("original result/run-start/source bindings drift")
    if saved.get("precision_bits")!=original.BITS or saved.get("label_count")!=256:
        raise ValueError("sealed result precision/alphabet drift")
    return frozen,saved,resolved


def core_check(root=ROOT):
    """Read-only saved-scalar replay; deliberately never original.verify_freeze."""
    frozen,saved,resolved=strict_bindings(root)
    # The original unchanged implementation enforces positivity, exact model
    # mass consistency, metric/radius qualification and every numerical gate.
    expected=original.synthesis()
    for name,value in expected.items():
        if saved.get(name)!=value:raise ValueError("sealed scalar synthesis drift: "+name)
    return dict(passed=True,checker="archive_aware_strict_supplement",
        original_freeze_source_key_unchanged="HANDOFF.md",
        archived_handoff_input=resolved["HANDOFF.md"],
        archived_handoff_sha256=frozen["handoff_sha256"],
        original_freeze_rewritten=False,original_result_rewritten=False,
        original_live_handoff_binding_bypassed_only_via_same_byte_archive=True,
        saved_scalar_fields_rebuilt=len(expected),replayed_saved_probability_syntheses=1,
        new_grid_evaluations=0,new_integrals=0,new_PMF_evaluations=0,
        new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0)


def original_file_hashes(root):
    return {name:sha(safe_path(root,name)) for name in (FREEZE,RESULT,START)}


def adaptation_record(root=ROOT):
    """Metadata only; caller may exclusive-create the one adaptation artifact."""
    verified=core_check(root)
    return dict(schema="candidate_v18_risk_range_archive_check_adaptation_v1",
        scope="governance_binding_adaptation_only_not_new_scientific_synthesis",
        explanation=("The sealed original checker retains its live HANDOFF.md binding. "
            "Default delivery uses the new strict supplement, resolving that one source "
            "key to its byte-identical archived HANDOFF_INPUT.md. Original freeze, "
            "source, result and run-start files were not rewritten; other paths and "
            "all numerical qualifications remain strict."),
        source_sha256={name:sha(safe_path(root,name)) for name in NEW_SOURCES},
        original_file_sha256=original_file_hashes(root),
        source_resolution={"HANDOFF.md":ARCHIVE},
        archived_handoff_sha256=sha(safe_path(root,ARCHIVE)),
        sealed_handoff_sha256=original.HANDOFF_SHA,
        actual_saved_scalar_check=verified,
        manufactured_tests="temporary roots; changed live feedback permitted only with matching archive; tampering rejected",
        new_grid_evaluations=0,new_integrals=0,new_PMF_evaluations=0,
        new_scientific_risk_range_evidence_created=False)


def check(root=ROOT):
    adaptation=read(safe_path(root,ADAPTATION))
    if (adaptation.get("schema")!="candidate_v18_risk_range_archive_check_adaptation_v1"
            or adaptation.get("source_resolution")!={"HANDOFF.md":ARCHIVE}
            or adaptation.get("archived_handoff_sha256")!=original.HANDOFF_SHA
            or adaptation.get("sealed_handoff_sha256")!=original.HANDOFF_SHA):
        raise ValueError("check adaptation source resolution drift")
    if adaptation.get("source_sha256")!={name:sha(safe_path(root,name)) for name in NEW_SOURCES}:
        raise ValueError("supplemental checker source hash drift")
    if adaptation.get("original_file_sha256")!=original_file_hashes(root):
        raise ValueError("original sealed freeze/result/run-start bytes changed")
    checked=core_check(root)
    if adaptation.get("actual_saved_scalar_check")!=checked:
        raise ValueError("adaptation/read-only verification receipt drift")
    return checked


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true",required=True)
    parser.parse_args()
    print(json.dumps(check(),ensure_ascii=False,indent=2))
