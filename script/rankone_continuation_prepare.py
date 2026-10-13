"""Prepare fixed public inputs and a one-time pre-execution freeze."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_continuation_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneContinuation_v1_20261007"
OLD=ROOT/"artifacts/rankone_stationary_v1_20261007"
BASE="f8354584fd321f909c8319000485c89cb303188d"
PRIOR_SHA="405d07e987981d353ee7648ff6d0ab1253b6c17ff3993893cb131b58542a4600"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def save(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def source():
    OUT.mkdir(exist_ok=False)
    assert sha(OLD/"DELIVERY_MANIFEST.json")==PRIOR_SHA
    lock=read(OLD/"ENDPOINTS_LOCK.json");q=read(OLD/"SOURCE.json")["teacher_pmf"]
    endpoint=next(e for e in lock["endpoints"] if e["name"]=="P")
    anchor=dict(C=rel(OLD/"evaluation/P_C.json"),D=rel(OLD/"evaluation/P_D.json"),
                qualification=rel(OLD/"evaluation/P_QUALIFICATION.json"),
                endpoint_lock=rel(OLD/"ENDPOINTS_LOCK.json"))
    assert np.array_equal(endpoint["fit_target"],q)
    for key in ("C","D"):
        assert np.array_equal(read(ROOT/anchor[key])["theta"],endpoint["theta"])
    assert read(ROOT/anchor["qualification"])["fitted_signature"]["qualified_full_local_minimum_signature"]
    inputs=[*(ROOT/name for name in anchor.values()),OLD/"SOURCE.json",OLD/"DELIVERY_MANIFEST.json"]
    save(OUT/"SOURCE.json",dict(schema="rankone_continuation_source_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        initial_theta=endpoint["theta"],teacher_pmf=q,initial_K8=anchor,
        sources={rel(p):sha(p) for p in inputs},
        target_weights_renormalized=False,teacher_mass=float(np.sum(q)),
        initial_K8_reused_without_new_optimization_or_quadrature=True))
    save(OUT/"SCOPE.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        user_authorized_continuation=True,handoff_sha256=sha(ROOT/"HANDOFF.md"),handoff_unchanged=True,
        coarse_K=[9,10,12,16,24,32,64,128,"inf"],maximum_refinement_solves=6,
        old_arrays_replayed=False,new_observations=0,new_random_draws=0,statistical_alpha_spent=0,
        manuscript_modified=False,new_transfer_prompt_created=False,design_sha256=sha(DOC/"DESIGN.md")))

def freeze():
    import rankone_continuation_compute as driver
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==BASE
    assert not (OUT/"ENDPOINTS_LOCK.json").exists()
    inputs=[Path(__file__).resolve(),ROOT/"script/rankone_continuation_compute.py",
        ROOT/"script/rankone_continuation_compute_test.py",
        ROOT/"script/rankone_stationary_kernel.py",ROOT/"script/rankone_stationary_compute.py",
        DOC/"DESIGN.md",DOC/"INTERPRETATION.md",OUT/"SOURCE.json",OUT/"SCOPE.json",
        OUT/"COMPUTE_TEST.json",OLD/"DELIVERY_MANIFEST.json"]
    save(OUT/"FREEZE.json",dict(schema="rankone_continuation_freeze_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        config=driver.config_dump(),source_sha256=sha(OUT/"SOURCE.json"),
        sources={rel(p):sha(p) for p in inputs},science_not_started_at_freeze=True,
        new_observations=0,new_random_draws=0,statistical_alpha_spent=0,
        manuscript_modified=False,new_transfer_prompt_created=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("source","freeze"))
    args=parser.parse_args();(source if args.mode=="source" else freeze)()
    print(json.dumps(dict(passed=True,mode=args.mode,
        sha256=sha(OUT/("SOURCE.json" if args.mode=="source" else "FREEZE.json")))))
