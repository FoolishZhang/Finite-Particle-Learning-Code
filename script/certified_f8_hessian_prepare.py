"""Freeze full-block Hessian computation and conditional witness synthesis."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/certified_f8_hessian_v1_20261008';DOC=ROOT/'docs/08_research_研究工作/CertifiedF8Hessian_v1_20261008'
BASE='3f3814072f2b2ca7de29c78d5cfbdfdc33d8d954';HANDOFF='3e2703ead9948ba86d115335b0c31aeac2a903517c0064e846151e77678e6e2f'
PRIOR='artifacts/certified_f8_gradient_v1_20261008/DELIVERY_MANIFEST.json';PRIOR_SHA='784d935ffe6be666f7cdd5a2182aec4c20403aa82f5e67f276c735ccc67b6740'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='research'
    for n in ('FREEZE.json','RUN_START.json','AUDIT_START.json','FINITE_HESSIANS.json'):assert not (OUT/n).exists(),n
    assert not list(OUT.glob('rep_*.json')) and sha(ROOT/'HANDOFF.md')==HANDOFF and sha(ROOT/PRIOR)==PRIOR_SHA
    import certified_f8_hessian_compute as compute
    import certified_f8_hessian_finalize as finalize
    inputs=compute.common.read(OUT/'INPUTS.json');assert inputs==compute.prepared_inputs()
    paths=[DOC/n for n in ('CONTRACT.md','HANDOFF_DISPOSITION.md','QUADRATURE_PROOF.md','MOMENTS_AND_BLOCKS.md','IMPLEMENTATION.md','WITNESS_ARGUMENT.md')]
    paths += [OUT/'INPUTS.json',ROOT/PRIOR,Path(__file__).resolve()]
    paths += [ROOT/n for n in inputs['input_source_sha256']]
    paths += [finalize.SAVED,finalize.GRAD,finalize.GRAD_INPUTS,finalize.RISK,finalize.RISK_INPUTS,finalize.TEACHER_CERT]
    for stem in ('bounds','moments','compute','audit','finalize'):
        paths += [ROOT/'script'/f'certified_f8_hessian_{stem}{suffix}.py' for suffix in ('','_test')]
    for prefix in ('BOUNDS','MOMENTS','COMPUTE','AUDIT','FINALIZE'):
        p=OUT/(prefix+'_MANUFACTURED_CHECK.json');record=json.loads(p.read_text(encoding='utf-8'))
        assert record.get('passed') is True or record.get('status')=='passed',p
        paths.append(p)
    for p in paths:assert p.is_file(),p
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    paths.append(OUT/'HANDOFF_INPUT.md')
    frozen=dict(schema='certified_f8_hessian_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),baseline=BASE,
        handoff_sha256=HANDOFF,reviewed_commit='1cfd58ddaeba096e39974f783a80f8a09153c24f',feedback_use='reference_unchanged',prior_manifest_sha256=PRIOR_SHA,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
        real_Hessian_factor_evaluations_before_freeze=0,K=8,precision_bits=192,representatives=41,block_names=['Hbb','Haa','Hperp'],block_dimension=8,
        inner_nodes=881,outer_nodes=769,producer_factors=27777049,producer_hard_seconds=3600,
        audit_counts=[[0,0,0,0],[1,1,1,1]],audit_factors=1354978,audit_matrix_components=384,audit_hard_seconds=600,
        contraction_columns=35,total_scalar_moments=36,saturation_threshold=128,
        Hessian_operator_error_gate_exact='1/1000',gradient_cap_exact='1893/2500000000000',physical_radius_exact='1/2000000',
        risk_descent_cap_exact='201723/10000000',conflict_step_exact='1/10000000',
        witness_claim_requires_all_applicable_verified_antecedents=True,new_random_draws=0,new_optimizer_updates=0,manuscript_modified=False,
        delivery_checker_bound_only_at_final_seal=True)
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(frozen,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,sources=len(frozen['sources']),sha256=sha(OUT/'FREEZE.json'))))
if __name__=='__main__':main()
