"""Freeze source provenance before constructing the sole R01 direction."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'artifacts/angular_witness_v1_20261010'
DOC = 'docs/08_research_研究工作/AngularWitness_v1_20261010/'
BASE = 'fd19c05ee181e95e8c2ce4040943b01ee18907e5'
TEACH = 'artifacts/certified_teacher_v1_20261008/'
SOURCES = {
    'old_direction': 'artifacts/rankone_stationary_v1_20261007/evaluation/P_D.json',
    'saved_center_matrix': 'artifacts/local_witness_feasibility_v1_20261008/D1.json',
    'center': TEACH+'CANDIDATE_RISK_INTEGRALS.json',
    'teacher_inputs': TEACH+'INPUTS.json',
    'teacher_finite': TEACH+'FINITE_GRID_SUM.json',
    'teacher_certificate': TEACH+'TEACHER_CERTIFICATE.json',
    'Q_certificate': TEACH+'TEACHER_RISK_CERTIFICATE.json',
    'root_certificate': 'artifacts/certified_f8_hessian_v1_20261008/HESSIAN_AND_WITNESS_CERTIFICATE.json',
    'center_identity': 'artifacts/local_witness_feasibility_v1_20261008/INPUTS.json',
    'hessian_inputs': 'artifacts/certified_f8_hessian_v1_20261008/INPUTS.json',
    'gradient_inputs': 'artifacts/certified_f8_gradient_v1_20261008/INPUTS.json',
    'paper_manifest': 'artifacts/submission_candidate_v1_8_4_20261010/DELIVERY_MANIFEST.json',
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def write_new(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def verify_bindings(bindings):
    for name, digest in bindings.items():
        require(sha(ROOT/name) == digest, 'Source changed: '+name)

def freeze_selection():
    files = list(SOURCES.values()) + [DOC+'CONTRACT.md', 'script/angular_witness_setup.py', 'HANDOFF.md']
    bindings = {p: sha(ROOT/p) for p in files}
    require(bindings['HANDOFF.md'] == '762dd8160e2abad43e110a7f60c5e055a8f025bc98edd4ab542226acea88b7f2', 'Feedback identity')
    write_new(PACKAGE/'SELECTION_FREEZE.json', dict(schema='angular_selection_freeze_v1',
        frozen_utc=now(), base_commit=BASE, bindings=bindings,
        target_nodes_before_freeze=0, project_direction_selected_before_freeze=False))
    with (PACKAGE/'HANDOFF_INPUT.md').open('xb') as f:
        f.write((ROOT/'HANDOFF.md').read_bytes())

def select():
    import numpy as np
    frozen = read(PACKAGE/'SELECTION_FREEZE.json')
    verify_bindings(frozen['bindings'])
    require(not (PACKAGE/'DIRECTION.json').exists(), 'Direction already constructed')
    pd = read(ROOT/SOURCES['old_direction'])
    blocks = [b for b in pd['blocks'] if b['target'] == 'population' and b['K'] == 'inf']
    require(len(blocks) == 1, 'Unique old population Kinf block')
    block = blocks[0]
    vals = np.asarray(block['Hangle_projected_eigenvalues'], dtype=np.float64)
    require(np.isfinite(vals).all() and vals[0] < -1e-4 and vals[1]-vals[0] > 1e-6, 'Old candidate qualification')
    projected = np.asarray(block['Hangle_projected_eigenvectors'], dtype=np.float64)[:, 0]
    lifted = np.asarray(pd['helmert_basis'], dtype=np.float64) @ projected
    lifted -= np.mean(lifted)
    lifted /= np.linalg.norm(lifted)
    if lifted[np.argmax(np.abs(lifted))] < 0:
        lifted *= -1
    require(np.isfinite(lifted).all(), 'Nonfinite direction')
    u = [int(x) for x in np.rint(lifted[:7]*(2**24))]
    u.append(-sum(u))
    S = sum(x*x for x in u)
    require(S > 0 and sum(u) == 0 and len(set(u)) > 1, 'Exact direction qualification')
    center = read(ROOT/SOURCES['center'])
    a = [Fraction(int(x['numerator']), int(x['denominator'])) for x in center['a_exact']]
    D = read(ROOT/SOURCES['saved_center_matrix'])
    db = [b for b in D['blocks'] if b['K'] == 'inf']
    require(len(db) == 1, 'Unique centre matrix')
    df = np.asarray(u, dtype=np.float64)/np.sqrt(float(S))
    rayleigh = float(df @ np.asarray(db[0]['Hangle']) @ df)
    direction = dict(schema='angular_direction_v1', selected_utc=now(),
        selection_freeze_sha256=sha(PACKAGE/'SELECTION_FREEZE.json'),
        source= SOURCES['old_direction'], source_sha256=sha(ROOT/SOURCES['old_direction']),
        numpy_version=np.__version__, source_eigenvalues=vals.tolist(),
        saved_eigenvector_column=0, canonical_lifted_float64=lifted.tolist(),
        rounding='first seven nearest-even at 2^24; eighth = negative sum',
        u_integer=u, S_integer=S, sum_u=0, direction='d=u/sqrt(S)',
        center_saved_matrix_rayleigh_motivation_only=rayleigh,
        new_integrals=0, candidate_selection_is_exploratory=True,
        scientific_sign_certificate=False)
    write_new(PACKAGE/'DIRECTION.json', direction)
    write_new(PACKAGE/'INPUTS.json', dict(schema='angular_inputs_v1', frozen_utc=now(),
        a_exact=list(map(str,a)), b_exact=['0']*8, u_integer=u, S_integer=S,
        step_exact='1/8', N=88, bits=192, strip_exact='1', hard_seconds=600,
        root_distance_cap_exact='1.805e-8', scalar_L_cap_exact='417',
        transfer_cap_exact='7.52685e-6', teacher_L1_cap_exact='4.894e-21',
        center_width_gate_exact='1e-8', source_paths=SOURCES,
        direction_sha256=sha(PACKAGE/'DIRECTION.json'),
        selection_freeze_sha256=sha(PACKAGE/'SELECTION_FREEZE.json')))
    print(json.dumps(dict(u=u, S=S, saved_matrix_rayleigh=rayleigh, new_integrals=0)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser.add_mutually_exclusive_group(required=True)
    p.add_argument('--freeze-selection', action='store_true')
    p.add_argument('--select', action='store_true')
    args=parser.parse_args()
    freeze_selection() if args.freeze_selection else select()

if __name__ == '__main__':
    main()
