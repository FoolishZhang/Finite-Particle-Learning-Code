"""One-time source reconstruction and pre-execution freeze; no scientific solve."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_stationary_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneStationary_v1_20261007"
BASE="7279ad63cb68335aaf6c29d3fa5ba5c5e8b2e07f"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def source():
    assert not (OUT/"FREEZE.json").exists()
    sources={};solves=[];teacher=None
    for rep in range(4):
        folder=ROOT/"artifacts/joint_dependence_v1_20261007"/f"rep_{rep:02d}"
        lock=folder/"CANDIDATES_LOCK.json";qual=folder/"ENDPOINT_QUALIFICATION.json"
        sources.update({rel(p):sha(p) for p in (lock,qual)})
        data=read(lock);saved=read(qual)
        candidate=next(c for c in data["candidates"] if c["name"]=="K8")
        b=np.array(candidate["b"]);W=np.array(candidate["W"])
        assert b.shape==(8,) and W.shape==(8,2)
        frequency=np.asarray(data["frequency"])
        current=np.asarray(saved["teacher_pmf"])
        if teacher is None:teacher=current.copy()
        else:assert np.array_equal(teacher,current)
        assert frequency.shape==(256,) and (frequency>0).all()
        assert abs(frequency.sum()-1)<1e-12
        solves.append(dict(name=f"E{rep}",source_rep=rep,target_name="empirical",
            theta0=np.r_[b,np.linalg.norm(W,axis=1)].tolist(),target=frequency.tolist()))
    assert teacher.shape==(256,) and (teacher>0).all() and abs(teacher.sum()-1)<1e-12
    solves.append(dict(name="P",source_rep=0,target_name="population",
        theta0=list(solves[0]["theta0"]),target=teacher.tolist()))
    write(OUT/"SOURCE.json",dict(schema="rankone_stationary_source_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        sources=sources,teacher_pmf=teacher.tolist(),solves=solves,
        teacher_mass=float(teacher.sum()),targets_renormalized=False,
        start_rule="original_K8_bias_and_W_row_norms; population_start_rep00",
        independent_new_data_replicates=0,new_observations=0,new_random_draws=0))

def freeze():
    import rankone_stationary_compute as driver
    assert not (OUT/"ENDPOINTS_LOCK.json").exists()
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==BASE
    inputs=[DOC/"DESIGN.md",DOC/"PROOF.md",OUT/"SCOPE.json",OUT/"SOURCE.json",
            OUT/"KERNEL_TEST.json",OUT/"COMPUTE_TEST.json",
            ROOT/"script/rankone_stationary_kernel.py",
            ROOT/"script/rankone_stationary_kernel_test.py",
            ROOT/"script/rankone_stationary_compute.py",
            ROOT/"script/rankone_stationary_compute_test.py",Path(__file__).resolve(),
            ROOT/"artifacts/rankone_stability_v1_20261007/DELIVERY_MANIFEST.json"]
    config=driver.config_dump()
    if isinstance(config,str):config=json.loads(config)
    write(OUT/"FREEZE.json",dict(schema="rankone_stationary_freeze_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        source_sha256=sha(OUT/"SOURCE.json"),config=config,
        sources={rel(p):sha(p) for p in inputs},
        science_not_started_at_freeze=True,endpoints_not_selected_by_risk=True,
        new_observations=0,new_random_draws=0,statistical_alpha_spent=0,
        manuscript_modified=False,new_transfer_prompt_created=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["source","freeze"])
    args=parser.parse_args()
    (source if args.mode=="source" else freeze)()
    print(json.dumps(dict(passed=True,mode=args.mode,
        sha256=sha(OUT/("SOURCE.json" if args.mode=="source" else "FREEZE.json")))))
