"""Freeze the bounded F8 gradient prototype before actual grid evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/certified_f8_gradient_v1_20261008';DOC=ROOT/'docs/08_research_研究工作/CertifiedF8Gradient_v1_20261008'
BASE='a29ebe39b557cfc476d14fe973af63d499354732';HANDOFF='3e2703ead9948ba86d115335b0c31aeac2a903517c0064e846151e77678e6e2f'
PRIOR='artifacts/certified_teacher_v1_20261008/DELIVERY_MANIFEST.json';PRIOR_SHA='541ef0a498b0d275e78ab0c418275404c7beff2021e59e275da079d76c3c24d4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='research'
    for name in ('FREEZE.json','RUN_START.json','FINITE_GRADIENTS.json','AUDIT_START.json'):assert not (OUT/name).exists(),name
    assert not list(OUT.glob('rep_*.json'))
    assert sha(ROOT/'HANDOFF.md')==HANDOFF and sha(ROOT/PRIOR)==PRIOR_SHA
    import certified_f8_gradient_compute as compute
    import certified_f8_gradient_bounds as bounds
    from flint import ctx
    assert compute.common.read(OUT/'INPUTS.json')==compute.prepared_inputs('1/40','arb_mat',128,1800)
    ctx.prec=192
    teacher=compute.common.read(ROOT/'artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json')
    assert compute.common.from_endpoint(teacher['total_L1_error_from_finite_midpoints']['upper_exact_dyadic'])<compute.common.aq(bounds.TEACHER_ERROR)
    paths=[DOC/n for n in ('CONTRACT.md','HANDOFF_DISPOSITION.md','QUADRATURE_PROOF.md','SYMMETRY_AND_SCOPE.md','IMPLEMENTATION.md')]
    paths += [OUT/'INPUTS.json',Path(__file__).resolve(),ROOT/PRIOR,ROOT/compute.NOMINAL,
        ROOT/'script/certified_teacher_compute.py',ROOT/'artifacts/certified_teacher_v1_20261008/FINITE_GRID_SUM.json',ROOT/'artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json']
    for name in ('bounds','symmetry','compute','audit','finalize'):
        paths += [ROOT/'script'/f'certified_f8_gradient_{name}{suffix}.py' for suffix in ('','_test')]
    for name in ('BOUNDS','SYMMETRY','COMPUTE','AUDIT','FINALIZE'):
        p=OUT/(name+'_MANUFACTURED_CHECK.json');j=json.loads(p.read_text(encoding='utf-8'))
        assert j.get('passed') is True or j.get('status')=='passed',p
        paths.append(p)
    for p in paths:assert p.is_file(),p
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    paths.append(OUT/'HANDOFF_INPUT.md')
    frozen=dict(schema='certified_f8_gradient_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        handoff_sha256=HANDOFF,reviewed_commit='1cfd58ddaeba096e39974f783a80f8a09153c24f',feedback_use='reference_unchanged',prior_manifest_sha256=PRIOR_SHA,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
        real_center_factor_evaluations_before_freeze=0,representatives=41,isometric_gradient_dimension=4,K=8,
        inner_nodes=881,outer_nodes=769,producer_factor_budget=41*881*769,producer_hard_seconds=1800,
        audit_counts=[[0,0,0,0],[1,1,1,1]],audit_factor_budget=2*881*769,audit_hard_seconds=600,
        precision_bits=192,backend='arb_mat',saturation_threshold=128,saturation_is_interval_enclosure=True,
        teacher_error_L1_exact=bounds.TEACHER_ERROR,target_gradient_L2_exact='1/100000000',
        Hessian_integrals_started=False,true_F8_stationary_existence_certified=False,manuscript_modified=False,new_random_draws=0,new_optimizer_updates=0,
        delivery_only_sources_bound_at_final_manifest=True)
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(frozen,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,sources=len(frozen['sources']),sha256=sha(OUT/'FREEZE.json'))))
if __name__=='__main__':main()
