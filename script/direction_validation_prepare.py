"""Freeze the finite direction-validation feasibility panel before calculation."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/direction_validation_v1_20261007'
DOC=ROOT/'docs/08_research_研究工作/DirectionValidation_v1_20261007'
BASE='df31ecd3d93c50f72859d364f1e5fd22faf0eed4'
HANDOFF='11c9d0d8518c8f7fff6b1abd74bde35abe4e00a2ff49d75228d1126c9c1f8f89'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def bind(p):return dict(path=rel(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def main():
    import direction_validation_compute as compute
    import scipy
    assert not (OUT/'FREEZE.json').exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert sha(ROOT/'HANDOFF.md')==sha(OUT/'HANDOFF_INPUT.md')==HANDOFF
    checks=[OUT/(s+'_MANUFACTURED_CHECK.json') for s in ('STATISTICS','COMPUTE','AUDIT')]
    for path in checks:
        r=read(path);assert r['passed'],path
        for name,value in r.get('sources',{}).items():assert sha(ROOT/name)==(value if isinstance(value,str) else value['sha256']),name
    sources=[*compute.required_freeze_sources(),Path(__file__),DOC/'MATH.md',*checks,
        ROOT/'script/direction_validation_audit.py',ROOT/'script/direction_validation_audit_test.py',
        ROOT/'script/prospective_explanation_independent.py',ROOT/'script/prospective_explanation_independent_test.py']
    record=dict(schema='direction_validation_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        handoff_sha256=HANDOFF,production_not_started=True,config=compute.config_dump(),
        sources={rel(p):bind(p) for p in sources},
        independent_config=dict(GL_PMF_calls=48,GL_order=128,GL_limit=10.,GL_rotation=0.,
            PMF_mass_tolerance=2e-12,PMF_difference_tolerance=2e-8,paired_logscore_tolerance=1e-9,
            paired_mean_tolerance=1e-10,automatic_order_escalation=False),
        environment=dict(python=platform.python_version(),numpy=compute.np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        implementation_sources_are_bound_working_tree_files_not_claimed_in_baseline=True,
        stage='oracle_feasibility_on_previous_development_objects',new_independent_validation=False,
        new_training=False,new_observations=0,new_random_draws=0,new_optimizer_updates=0,
        old_A_B_d_unchanged=True,producer_PMF_calls=192,independent_PMF_calls=48,
        manuscript_modified=False,alpha_spent_on_new_random_validation=0,
        next_stage_requires_separate_fixed_contract=True)
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(frozen=True,freeze_sha256=sha(OUT/'FREEZE.json'),producer_PMF_calls=192,independent_PMF_calls=48,new_samples=0)))
if __name__=='__main__':main()
