"""Freeze the bounded saved-object explanation before any study calculation."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/prospective_explanation_v1_20261007'
DOC=ROOT/'docs/08_research_研究工作/ProspectiveExplanation_v1_20261007'
OLD=ROOT/'artifacts/prospective_diagnostic_dev_v1_20261007'
BASE='8478793de147fa4d2162a63e5ad1fb2cfc3c9380'
HANDOFF='f6a58dfad46e02ddf424d69600cc7066aca52a795abd38670252cb656cca0089'
DEV_MANIFEST='2b537dd6e86f9547a1e540dd429b14a86a46c04d327ddb97c5b58fb783a9abd8'
PAPER_MANIFEST='c47d8f3f732458ca96830f729a4c588685f5c56f2c634af062b8659e41da410d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8'))
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def binding(path):return dict(path=rel(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def save(path,record):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    import prospective_explanation_compute as compute
    import scipy
    assert not (OUT/'FREEZE.json').exists(), 'Never overwrite a scientific freeze'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert sha(ROOT/'HANDOFF.md')==sha(OUT/'HANDOFF_INPUT.md')==HANDOFF
    assert sha(OLD/'DELIVERY_MANIFEST.json')==DEV_MANIFEST
    paper=ROOT/'artifacts/submission_candidate_v1_6_1_20261007/DELIVERY_MANIFEST.json'
    assert sha(paper)==PAPER_MANIFEST
    for name in ('COMPUTE_MANUFACTURED_CHECK.json','INDEPENDENT_MANUFACTURED_CHECK.json'):
        assert (OUT/name).is_file(), 'Finish the manufactured implementation tests first: '+name
        check=read(OUT/name);assert check['passed'],name
        if 'sources' in check:
            for path,digest in check['sources'].items():assert sha(ROOT/path)==digest,path
        else:
            assert sha(ROOT/'script/prospective_explanation_independent.py')==check['source_sha256']
            assert sha(ROOT/'script/prospective_explanation_independent_test.py')==check['test_sha256']
    paths=[DOC/'CONTRACT.md',OUT/'HANDOFF_INPUT.md',Path(__file__),
        *(ROOT/'script'/name for name in ('prospective_explanation_compute.py','prospective_explanation_compute_test.py',
          'prospective_explanation_independent.py','prospective_explanation_independent_test.py',
          'prospective_diagnostic_dev.py','prospective_diagnostic_core.py','rankone_stationary_kernel.py',
          'rankone_stationary_compute.py','joint_dependence_kernel.py')),
        OUT/'COMPUTE_MANUFACTURED_CHECK.json',OUT/'INDEPENDENT_MANUFACTURED_CHECK.json',
        OLD/'DELIVERY_MANIFEST.json',paper,
        *(OLD/(name+'.json') for name in ('CONFIG','SUMMARY','FREEZE','INPUTS_LOCK','DIAGNOSIS_LOCK','ENDPOINTS_LOCK','EVALUATION_LOCK'))]
    old=read(OLD/'DELIVERY_MANIFEST.json')
    raw=old['raw_evidence']
    for name,row in raw.items():
        p=ROOT/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],name
    save(OUT/'SOURCE.json',dict(schema='prospective_explanation_sources_v1',baseline=BASE,
        predecessor_development_manifest_sha256=DEV_MANIFEST,predecessor_manuscript_manifest_sha256=PAPER_MANIFEST,
        old_raw_bindings=raw,scope='Only original saved development objects; no new observation/prior/trajectory'))
    paths.append(OUT/'SOURCE.json')
    record=dict(schema='prospective_explanation_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=BASE,handoff_sha256=HANDOFF,sources={rel(p):binding(p) for p in paths},config=compute.config_dump(),
        environment=dict(python=platform.python_version(),numpy=compute.np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        independent_config=dict(GL_orders=[96,128,128],GL_limit=10.,GL_rotations=['0','0','pi/7'],GL_PMF_calls=6,
            GH_anchor_order=80,GH_K_infinity_calls=12,normal_mass_gate=2e-12,PMF_gate=1e-12,KL_gate=1e-10,
            anchor_matrix_gradient_gate=1e-9,automatic_order_escalation=False),
        production_not_started=True,post_hoc_explanation=True,new_independent_validation=False,
        implementation_sources_are_bound_working_tree_files_not_claimed_in_baseline_commit=True,
        teacher_information_for_explanation_only=True,new_observations=0,new_random_priors=0,new_optimizer_updates=0,
        analytic_independent_Bernoulli_MLE_fits=12,main_compute_new_quadrature_calls=0,
        limited_independent_quadrature_calls=18,original_A_B_and_directions_unchanged=True,
        new_explanation_not_automatically_integrated_into_v162=True)
    save(OUT/'FREEZE.json',record)
    print(json.dumps(dict(frozen=True,baseline=BASE,old_raw_files_bound=len(raw),new_observations=0,
        new_optimizer_updates=0,independent_quadrature_call_cap=18,freeze_sha256=sha(OUT/'FREEZE.json'))))

if __name__=='__main__':main()
