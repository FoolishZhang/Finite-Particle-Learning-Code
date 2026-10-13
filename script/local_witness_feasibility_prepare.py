"""Freeze the bounded single-center feasibility study before its first kernel call."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/local_witness_feasibility_v1_20261008'
DOC = ROOT / 'docs/08_research_研究工作/LocalWitnessFeasibility_v1_20261008'
BASE = 'e924b13643e09a9ad190b71c03cabeac501b3971'
HANDOFF_SHA = '3e2703ead9948ba86d115335b0c31aeac2a903517c0064e846151e77678e6e2f'
PRIOR = 'artifacts/asymptotic_bridge_v1_20261008/DELIVERY_MANIFEST.json'
PRIOR_SHA = '9469d4738aa6dcc4dc2a681d7defa41d778bdc432458cd146dabf27aeba6e1e6'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == BASE
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() == 'research'
    assert not (OUT/'FREEZE.json').exists() and not (OUT/'RUN_START.json').exists()
    assert not list(OUT.glob('*.npz')) and not (OUT/'NOMINAL_SUMMARY.json').exists()
    assert sha(ROOT/'HANDOFF.md') == HANDOFF_SHA and sha(ROOT/PRIOR) == PRIOR_SHA
    from local_witness_feasibility_compute import validate_inputs
    inputs = validate_inputs()
    paths = [DOC/name for name in ('CONTRACT.md','GEOMETRY.md','TAIL_BOUNDS.md','HANDOFF_DISPOSITION.md')]
    paths += [OUT/name for name in ('INPUTS.json','GEOMETRY_CHECK_READOUT.json','COMPUTE_MANUFACTURED_CHECK.json',
        'CERTIFICATE_MANUFACTURED_CHECK.json','BOUNDS_MANUFACTURED_CHECK.json','TAIL_WRAPPER_MANUFACTURED_CHECK.json')]
    for stem in ('compute','certificate','bounds','tail_run'):
        paths.extend(ROOT/'script'/f'local_witness_feasibility_{stem}{suffix}.py' for suffix in ('','_test'))
    paths += [Path(__file__).resolve(), ROOT/PRIOR]
    paths += [ROOT/name for name in inputs['source_files']]
    paths += [ROOT/'artifacts/submission_candidate_v1_7_1_20261008/DELIVERY_MANIFEST.json']
    for path in paths:
        assert path.is_file(), path
    for name in ('COMPUTE_MANUFACTURED_CHECK.json','CERTIFICATE_MANUFACTURED_CHECK.json',
                 'BOUNDS_MANUFACTURED_CHECK.json','TAIL_WRAPPER_MANUFACTURED_CHECK.json'):
        receipt = json.loads((OUT/name).read_text(encoding='utf-8'))
        assert receipt.get('passed') is True or receipt.get('status') == 'passed', name
    with (OUT/'HANDOFF_INPUT.md').open('xb') as file:
        file.write((ROOT/'HANDOFF.md').read_bytes())
    paths += [OUT/'HANDOFF_INPUT.md']
    data = dict(schema='local_witness_feasibility_freeze_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=BASE, handoff_sha256=HANDOFF_SHA, reviewed_commit='1cfd58ddaeba096e39974f783a80f8a09153c24f',
        feedback_use='reference', prior_bridge_manifest_sha256=PRIOR_SHA,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
        real_center_kernel_calls_before_freeze=0, new_scientific_RNG_calls=0,
        prescribed_kernel_calls=2, rules=['C1','D1'], K_values=[8,'inf'], center_count=1,
        radius_exact='1/2000000', likelihood_scale_exact='1', matrix_floors_exact=['1/20','7/500'],
        gradient_total_target_exact='1/100000000', hessian_operator_error_target_exact='1/1000',
        truncation_target_exact='1/1000000000', scalar_precision_bits=192,
        manuscript_modified=False, full_interval_integral_or_root_certificate_started=False,
        scientific_freeze_excludes_later_delivery_checker_and_result_prose=True)
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as file:
        json.dump(data,file,ensure_ascii=False,indent=2,allow_nan=False); file.write('\n')
    print(json.dumps(dict(frozen=True, sources=len(data['sources']), sha256=sha(OUT/'FREEZE.json'))))

if __name__ == '__main__':
    main()
