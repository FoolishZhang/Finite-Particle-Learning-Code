"""Prespecified endpoint PMF qualification and paired risk statistics; no RNG."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from scipy.stats import t as student_t
import scbminv_reference as helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scbm_scale_optimization/REFERENCE'
TRAIN = ROOT / 'artifacts/scbm_scale_optimization/TRAIN/RESULTS.json'
DOC = ROOT / 'docs/08_research_研究工作/可学习无关尺度与优化路径_20261003'
TRUTH = ROOT / 'artifacts/scbm_invariance/REFERENCE/MODEL.json'
METHODS = ['FIXED', 'FREE', 'FACT']
KINDS = ['TAIL', 'LAST']
CFG = dict(orders=[24, 32, 40], main_order=32, rotation='cholesky',
           max_PMF_change=5e-8, normalization_tolerance=2e-12,
           minimum_eigenvalue=.12, candidate_range=[-.75, .75],
           streams=16, family_alpha=.05, family_size=3,
           noninferiority_margin=.0002,
           improvement_SE=.0002, noninferiority_SE=.00005)
PROGRESS = dict(completed_PMF_integrations=0, completed_Gaussian_nodes=0,
                completed_unique_endpoints=0, in_flight=None)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def read(p):
    return json.loads(Path(p).read_text('utf8'))


def save(p, value, exclusive=True):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def freeze():
    production = read(TRAIN)
    assert len(production['records']) == CFG['streams']
    files = [Path(__file__), Path(helper.__file__), DOC / 'EXPERIMENT_DESIGN.md',
             TRAIN, TRAIN.parent / 'FREEZE.json', TRUTH]
    save(OUT / 'FREEZE.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         source_sha256=sha(__file__), config=CFG,
         inputs={rel(p): sha(p) for p in files},
         endpoint_policy='All TAIL and LAST outputs; float.hex deduplication, no risk-based selection.',
         quadrature_policy_fixed_before_training=True, integrals_authorized=True))


def run():
    frozen = read(OUT / 'FREEZE.json')
    assert frozen['source_sha256'] == sha(__file__) and frozen['config'] == CFG
    for p, h in frozen['inputs'].items():
        assert sha(ROOT / p) == h, p
    assert frozen['integrals_authorized']
    save(OUT / 'EXECUTION.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
         source_sha256=sha(__file__), freeze_sha256=sha(OUT / 'FREEZE.json'),
         automatic_retry=False))
    started = time.perf_counter()
    train = read(TRAIN)
    true_model = read(TRUTH)
    assert np.array_equal(helper.MU, true_model['config']['mean'])
    assert np.array_equal(helper.COV, true_model['config']['covariance'])
    truth = np.asarray(next(r['main_pmf'] for r in true_model['records'] if r['t'] == 0.))
    endpoints = []
    unique = {}
    for row in train['records']:
        for method in METHODS:
            for kind in KINDS:
                value = float(row['outputs'][method][kind]['t'])
                assert -.75 <= value <= .75 and np.isfinite(value)
                key = value.hex()
                unique[key] = value
                endpoints.append(dict(stream=row['stream'], method=method, kind=kind,
                                      t=value, key=key))
    assert len(endpoints) == 96 and len(train['records']) == 16
    records = []
    for key, value in unique.items():
        cov = np.eye(4) + (1 + value) * (helper.COV - np.eye(4))
        eigmin = float(np.linalg.eigvalsh(cov).min())
        assert eigmin > CFG['minimum_eigenvalue']
        variants = []
        for order in CFG['orders']:
            PROGRESS['in_flight'] = dict(key=key, t=value, order=order,
                                        completed_nodes_unknown_if_interrupted=True)
            save(OUT / 'PROGRESS.json', PROGRESS, exclusive=False)
            variants.append(helper.pmf(cov, order, 'cholesky'))
            PROGRESS['completed_PMF_integrations'] += 1
            PROGRESS['completed_Gaussian_nodes'] += order**4
            PROGRESS['in_flight'] = None
            save(OUT / 'PROGRESS.json', PROGRESS, exclusive=False)
        main = np.asarray(next(v['pmf'] for v in variants if v['order'] == 32))
        qualifications = []
        for variant in variants:
            p = np.asarray(variant['pmf'])
            change = float(np.max(abs(p - main)))
            norm = abs(float(p.sum()) - 1.)
            qualifications.append(dict(order=variant['order'], maximum_change=change,
                normalization_error=norm, positive=bool(np.all(p > 0)),
                passed=bool(change <= CFG['max_PMF_change'] and
                            norm <= CFG['normalization_tolerance'] and np.all(p > 0))))
        risk = float(np.sum(truth * np.log(truth / main)))
        records.append(dict(key=key, t=value, minimum_eigenvalue=eigmin,
            main_pmf=main.tolist(), reference_joint_KL=risk, variants=variants,
            qualifications=qualifications, passed=all(x['passed'] for x in qualifications)))
        save(OUT / f'endpoint_{len(records)-1:03}.json', records[-1])
        PROGRESS['completed_unique_endpoints'] = len(records)
        save(OUT / 'PROGRESS.json', PROGRESS, exclusive=False)
    table = {r['key']: r for r in records}
    for row in endpoints:
        row['reference_joint_KL'] = table[row['key']]['reference_joint_KL']
    risks = {(r['stream'], r['method'], r['kind']): r['reference_joint_KL'] for r in endpoints}
    critical = float(student_t.ppf(1 - CFG['family_alpha'] / (2 * CFG['family_size']), 15))
    primary = []
    for name, left, right, direction, margin, gate in [
        ('FIXED_minus_FREE', 'FIXED', 'FREE', 'positive', 0., CFG['improvement_SE']),
        ('FIXED_minus_FACT', 'FIXED', 'FACT', 'positive', 0., CFG['improvement_SE']),
        ('FREE_minus_FACT_noninferiority', 'FREE', 'FACT', 'upper_below_margin',
         CFG['noninferiority_margin'], CFG['noninferiority_SE'])]:
        values = np.array([risks[(s, left, 'TAIL')] - risks[(s, right, 'TAIL')] for s in range(16)])
        mean = float(values.mean())
        se = float(values.std(ddof=1) / 4)
        lower, upper = mean - critical * se, mean + critical * se
        primary.append(dict(name=name, kind='TAIL', stream_values=values.tolist(),
            mean=mean, SE=se, lower=lower, upper=upper, expected=direction, margin=margin,
            effect_passed=bool(lower > 0 if direction == 'positive' else upper < margin),
            SE_threshold=gate, precision_passed=bool(se <= gate)))
    descriptive = []
    for kind in KINDS:
        for method in METHODS:
            v = np.array([risks[(s, method, kind)] for s in range(16)])
            t = [r['t'] for r in endpoints if r['method'] == method and r['kind'] == kind]
            descriptive.append(dict(kind=kind, method=method, mean_joint_KL=float(v.mean()),
                SE_joint_KL=float(v.std(ddof=1) / 4), mean_t=float(np.mean(t)),
                min_t=float(min(t)), max_t=float(max(t)), endpoint_t=t))
    passed = all(r['passed'] for r in records)
    output = dict(source_sha256=sha(__file__), freeze_sha256=sha(OUT / 'FREEZE.json'),
        config=CFG, truth_reference_sha256=sha(TRUTH), training_results_sha256=sha(TRAIN),
        records=records, endpoints=endpoints, qualification_passed=passed,
        primary_family=primary, critical_t=critical,
        primary_interpretation_qualified=passed,
        primary_effects_passed=sum(x['effect_passed'] for x in primary),
        primary_precision_passed=sum(x['precision_passed'] for x in primary),
        descriptive=descriptive,
        accounting=dict(unique_endpoints=len(records), final_outputs=96,
            PMF_integrations=3 * len(records), scalar_pattern_integrals=48 * len(records),
            Gaussian_nodes=len(records) * sum(x**4 for x in CFG['orders']),
            new_truth_integrals=0, new_random_draws=0, new_parameter_fits=0),
        seconds=time.perf_counter() - started,
        scope='Direct endpoint numerical PMFs, not interpolation or a strict integration certificate; paired development t intervals, not exact finite-sample coverage.')
    save(OUT / 'RESULTS.json', output)
    print(json.dumps({k: output[k] for k in ['qualification_passed', 'primary_effects_passed',
          'primary_precision_passed', 'descriptive', 'accounting', 'seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['schema', 'freeze', 'run'])
    a = p.parse_args()
    if a.mode == 'schema':
        print(json.dumps(dict(config=CFG, source_sha256=sha(__file__),
              helper_sha256=sha(helper.__file__))))
    else:
        try:
            globals()[a.mode]()
        except BaseException:
            if a.mode == 'run' and not (OUT / 'FAILURE.json').exists():
                save(OUT / 'FAILURE.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                     traceback=traceback.format_exc(), progress=PROGRESS, automatic_retry=False))
            raise
