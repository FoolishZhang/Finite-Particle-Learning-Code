"""Freeze exact teacher and fixed candidate quadrature before scientific runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/certified_teacher_v1_20261008'
DOC=ROOT/'docs/08_research_研究工作/CertifiedTeacher_v1_20261008'
BASE='4aa755b6163a2db95d9be94f74c4c9d3ccb40e9c'
HANDOFF='3e2703ead9948ba86d115335b0c31aeac2a903517c0064e846151e77678e6e2f'
PRIOR='artifacts/local_witness_feasibility_v1_20261008/DELIVERY_MANIFEST.json'
PRIOR_SHA='ba1928d56531fa7949d76dd124ebf5e51a45ddcd008177904047664c0904d1be'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='research'
    for name in ('FREEZE.json','RUN_START.json','FINITE_GRID_SUM.json','CANDIDATE_RISK_RUN_START.json','AUDIT_START.json'):
        assert not (OUT/name).exists(),name
    assert sha(ROOT/'HANDOFF.md')==HANDOFF and sha(ROOT/PRIOR)==PRIOR_SHA
    import certified_teacher_compute as compute
    assert compute.read(compute.INPUTS)==compute.prepared_inputs('all_labels')
    paths=[DOC/name for name in ('CONTRACT.md','HANDOFF_DISPOSITION.md','QUADRATURE_PROOF.md','IMPLEMENTATION.md','TRANSFER_BUDGET.md')]
    paths+=[OUT/'INPUTS.json',ROOT/PRIOR,ROOT/compute.OLD_SOURCE,
        ROOT/'artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json',Path(__file__).resolve()]
    for name in ('bounds','compute','audit','finalize','risk'):
        paths.extend(ROOT/'script'/f'certified_teacher_{name}{suffix}.py' for suffix in ('','_test'))
    for name in ('BOUNDS','COMPUTE','AUDIT','FINALIZE','RISK'):
        p=OUT/(name+'_MANUFACTURED_CHECK.json');j=json.loads(p.read_text(encoding='utf-8'))
        assert j.get('passed') is True or j.get('status')=='passed',p
        paths.append(p)
    for p in paths:assert p.is_file(),p
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    paths.append(OUT/'HANDOFF_INPUT.md')
    freeze=dict(schema='certified_teacher_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=BASE,handoff_sha256=HANDOFF,reviewed_commit='1cfd58ddaeba096e39974f783a80f8a09153c24f',feedback_use='reference_unchanged',
        prior_manifest_sha256=PRIOR_SHA,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
        real_teacher_grid_evaluations_before_freeze=0,real_candidate_grid_evaluations_before_freeze=0,
        producer_grid_points=25921,producer_labels=256,producer_blocks=11,producer_hard_seconds=600,
        audit_grid_points=25921,audit_labels=[0,15,85,90,165,170,240,255],audit_hard_seconds=300,
        candidate_grid_points=177,candidate_labels=256,candidate_hard_seconds=60,
        precision_bits=192,full_endpoint_weights=True,renormalization=False,
        teacher_L1_target_exact='1/1000000000000',saved_GH60_L1_target_exact='1/10000000000',
        risk_direction_upper_target_exact='-1/50',new_random_draws=0,new_optimizer_updates=0,
        full_F8_gradient_Hessian_integrals_executed=False,strict_stationary_existence_certified=False,
        manuscript_modified=False,read_only_delivery_source_bound_only_at_final_seal=True)
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(freeze,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,sources=len(freeze['sources']),sha256=sha(OUT/'FREEZE.json'))))
if __name__=='__main__':main()
