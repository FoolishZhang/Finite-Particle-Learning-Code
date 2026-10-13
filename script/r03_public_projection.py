"""Publish a bounded readout of local raw development; no scientific calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/r03_teacher_family_v1_20261011'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def produce():
    raw = json.loads((OUT / 'DEVELOPMENT.json').read_text(encoding='utf-8'))
    public = {k: v for k, v in raw.items() if k != 'points'}
    public['schema'] = 'r03_development_public_projection_v1'
    public['points'] = []
    kept = ['target_names', 'K_values', 'b', 'a', 'targets', 'loss', 'gradient',
            'bias_gradient', 'radial_gradient', 'H_ba', 'Hperp', 'Hangle',
            'ward_residual_max', 'H_ba_symmetry_residual', 'common_angle_residual',
            'config', 'work_counts', 'qualifications']
    for point in raw['points']:
        p = {k: v for k, v in point.items() if k != 'evaluations'}
        p['evaluations'] = {
            rule: {'summary': ev['summary'], 'aggregate_result': {k: ev['result'][k] for k in kept}}
            for rule, ev in point['evaluations'].items()}
        public['points'].append(p)
    rawfiles = [OUT / 'DEVELOPMENT.json', *sorted(OUT.glob('DEV_POINT_*.json'))]
    index = dict(schema='r03_local_raw_index_v1', files={
        p.relative_to(ROOT).as_posix(): dict(sha256=digest(p), bytes=p.stat().st_size)
        for p in rawfiles}, raw_files_kept_locally_unchanged=True,
        raw_files_in_git=False, raw_files_required_for_saved_strict_certificate_check=False,
        source_bindings={Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(__file__))})
    public['source_raw'] = index['files']['artifacts/r03_teacher_family_v1_20261011/DEVELOPMENT.json']
    public['projection_policy'] = dict(all_seven_points=True, all_root_traces=True,
        all_teacher_q_rules=True, all_aggregate_C_D_matrices=True, all_qualifications=True,
        omitted='final per-label tensors and quadrature coefficient arrays; unchanged local raw retains them',
        exact_public_GL_label_audit='AUDIT_RESULT.json embeds three saved comparison labels',
        second_independent_scientific_execution=False, new_target_calls=0)
    public['source_bindings'] = index['source_bindings']
    return public, index


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--write',action='store_true');g.add_argument('--check',action='store_true')
    a=p.parse_args(); public,index=produce()
    for name,value in [('DEVELOPMENT_PUBLIC.json',public),('DEV_RAW_INDEX.json',index)]:
        path=OUT/name
        if a.write:
            with path.open('x',encoding='utf-8',newline='\n') as f:
                json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        else:
            assert json.loads(path.read_text(encoding='utf-8'))==value,name
    print(json.dumps(dict(passed=True,local_raw_files=len(index['files']),
        public_points=len(public['points']),new_target_calls=0)))


if __name__=='__main__':main()
