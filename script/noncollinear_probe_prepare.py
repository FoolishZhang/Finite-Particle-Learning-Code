"""Prepare immutable public anchors/feedback and freeze the matched path panel."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/noncollinear_probe_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/NoncollinearProbe_v1_20261007"
OLD=ROOT/"artifacts/rankone_continuation_v1_20261007"
BASE="94fec1e1a2166ed7342752ab8c3bd43e39951714"
PRIOR_SHA="69333cee55272028bf0dd40a2ac3177d9d132a07c028e72105ab12fa46a45105"
HANDOFF_SHA="071614d0b001da6c55e351325653cbf45890256b2da8533bf96a975cfdfba9ab"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def save(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def source():
    OUT.mkdir(exist_ok=False)
    assert sha(OLD/"DELIVERY_MANIFEST.json")==PRIOR_SHA
    assert sha(ROOT/"HANDOFF.md")==HANDOFF_SHA
    with (OUT/"HANDOFF_INPUT.md").open("xb") as f:f.write((ROOT/"HANDOFF.md").read_bytes())
    plan_path=OLD/"cross/PLAN.json";cross_path=OLD/"cross/RESULTS.json"
    plan=read(plan_path);cross=read(cross_path);d=np.array(plan["direction"])
    assert plan["lower_K"]==11 and plan["upper_K"]==12
    teacher=read(OLD/"SOURCE.json")["teacher_pmf"]
    inputs=[plan_path,cross_path,OLD/"SOURCE.json",OLD/"ENDPOINTS_LOCK.json",
            OLD/"DELIVERY_MANIFEST.json"]
    anchors=[]
    for name,K,point,theta_key,cell in [
        ("K11",11,plan["lower_point_id"],"theta_L","LL"),
        ("K12",12,plan["upper_point_id"],"theta_H","HH")]:
        folder=OLD/"points"/point
        paths={rule:folder/f"evaluation_{rule}.json" for rule in ("C","D")}
        inputs.extend(paths.values());inputs.append(folder/"PARAMETERS_LOCK.json")
        theta=plan[theta_key];risk={}
        for rule,p in paths.items():
            data=read(p);assert np.array_equal(data["theta"],theta)
            infinity=next(row for row in data["blocks"] if row["K"]=="inf")
            risk[rule]=float(d@np.asarray(infinity["Hangle"])@d)
        anchors.append(dict(name=name,K=K,theta=theta,source_point_id=point,
            one_d={rule:rel(p) for rule,p in paths.items()},
            curvature={rule:cross["cells"][cell][rule+"_curvature"] for rule in ("C","D")},
            risk_curvature=risk))
    save(OUT/"SOURCE.json",dict(schema="noncollinear_probe_source_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        teacher_pmf=teacher,direction=d.tolist(),cross_plan=rel(plan_path),
        cross_results=rel(cross_path),anchors=anchors,
        sources={rel(p):sha(p) for p in inputs},teacher_mass=float(np.sum(teacher)),
        target_weights_renormalized=False,new_kernel_evaluations_for_source=0,
        risk_curvature_readout="fixed direction contracted against prior saved Hangle_inf",
        handoff_sha256=HANDOFF_SHA,reviewed_commit=BASE))
    save(OUT/"SCOPE.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        user_authorized_route_closure=True,handoff_sha256=HANDOFF_SHA,reviewed_commit=BASE,
        handoff_use="reference",feedback_snapshot_sha256=sha(OUT/"HANDOFF_INPUT.md"),
        external_review_linked_checks_files_obtained=False,
        primary_abs_t=.02,secondary_abs_t=[.05,.10],planned_baselines=6,planned_nonzero_evaluations=36,
        new_observations=0,new_random_draws=0,optimizer_updates=0,statistical_alpha_spent=0,
        two_dimensional_optimization_started=False,third_tier_enhancement_started=False,
        manuscript_modified=False,new_transfer_prompt_created=False,
        design_sha256=sha(DOC/"DESIGN.md")))

def freeze():
    import noncollinear_probe_compute as driver
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==BASE
    inputs=[Path(__file__).resolve(),ROOT/"script/noncollinear_probe_kernel.py",
        ROOT/"script/noncollinear_probe_kernel_test.py",ROOT/"script/noncollinear_probe_compute.py",
        ROOT/"script/noncollinear_probe_compute_test.py",DOC/"DESIGN.md",DOC/"DERIVATION.md",
        DOC/"HANDOFF_DISPOSITION.md",OUT/"SOURCE.json",OUT/"SCOPE.json",OUT/"HANDOFF_INPUT.md",
        OUT/"KERNEL_TEST.json",OUT/"COMPUTE_TEST.json",OLD/"DELIVERY_MANIFEST.json"]
    save(OUT/"FREEZE.json",dict(schema="noncollinear_probe_freeze_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        source_sha256=sha(OUT/"SOURCE.json"),config=driver.config_dump(),
        sources={rel(p):sha(p) for p in inputs},science_not_started_at_freeze=True,
        new_observations=0,new_random_draws=0,optimizer_updates=0,statistical_alpha_spent=0,
        manuscript_modified=False,third_tier_enhancement_started=False,new_transfer_prompt_created=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("source","freeze"))
    args=parser.parse_args();(source if args.mode=="source" else freeze)()
    print(json.dumps(dict(passed=True,mode=args.mode,
        sha256=sha(OUT/("SOURCE.json" if args.mode=="source" else "FREEZE.json")))))
