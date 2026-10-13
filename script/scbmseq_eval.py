"""Independent per-data expected-score pools for fixed new-data outputs.

Each (data_id, group) owns an independent stream and scores its ten local
models on shared inputs. Development checks use deterministic arrays only.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import hashlib
import json
import math
import time
import traceback
import numpy as np
from scipy.special import expit
import scbm6eval_mc as helper
import scbmseq_reference as reference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/scbm_sequential_generalization'
OUT = BASE/'EVAL'
SOURCE = Path(__file__).resolve()
CONFIG = BASE/'CONFIG.json'
SPEC = BASE/'SPEC.json'
HELPER_SHA = 'ced8d96024d95c2b81abf50076bba9901ffae8639cbcdf3fbde4fe0e20e1edd1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value, new=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if new else 'w', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def local_indices(data_id):
    assert isinstance(data_id, int) and 0 <= data_id < 16
    return [0, 1] + list(range(2+8*data_id, 10+8*data_id))


def group_name(data_id, group):
    assert 0 <= data_id < 16 and 0 <= group < 8
    return f'group_d{data_id:02}_g{group:02}'


def configuration():
    cfg = read(CONFIG)['evaluation']
    expected = dict(data_ids=list(range(16)), groups=8, banks_per_group=512,
        K=100, nuisance=256, chunk=64, workers=4, seed_base=202610040200,
        roles={'active': 0, 'noise': 1}, seed_format='[seed_base,data_id,group,role]',
        profiles=['raw', 'FACT'], candidates=130, local_candidates=10,
        saved_prefix_banks=2, alpha=.025, family_size=2, df=15, SE_threshold=.0005,
        primary_order=['raw_P_OUT4_minus_IN4', 'raw_P_SEQ4_minus_OUT4'],
        two_sided=True, direction_open=True, canonical_nuisance_label=0,
        native_data_ids=[0, 15], native_groups=[0])
    assert cfg == expected, 'Evaluation configuration differs from fixed contract'
    assert sha(helper.__file__) == HELPER_SHA
    spec = read(SPEC)
    assert spec['mu'] == [-.6, -.2, .2, .6]
    assert spec['patterns'] == helper.PATTERNS.astype(int).tolist()
    return cfg


def contract():
    cfg = configuration()
    _, _, _, selection = reference.validate_derived()
    freeze = read(BASE/'FREEZE.json')
    for path in [SOURCE, Path(helper.__file__), SPEC, CONFIG]:
        assert freeze['inputs'][rel(path)] == sha(path)
    rows = selection['candidates']
    assert len(rows) == 130 and [row['index'] for row in rows] == list(range(130))
    assert [rows[i]['name'] for i in [0, 1]] == ['truth', 'I']
    for d in cfg['data_ids']:
        for r in range(2):
            for a, arm in enumerate(['IN4', 'OUT4', 'SEQ4', 'FACT']):
                row = rows[2+(d*2+r)*4+a]
                assert (row['data_id'], row['repeat'], row['arm']) == (d, r, arm)
    return cfg, selection


def bindings():
    return dict(source_sha256=sha(SOURCE), helper_sha256=HELPER_SHA,
        config_sha256=sha(CONFIG), spec_sha256=sha(SPEC),
        data_sha256=sha(BASE/'DATA.json'), candidates_sha256=sha(BASE/'CANDIDATES.json'),
        freeze_sha256=sha(BASE/'FREEZE.json'),
        data_freeze_sha256=sha(BASE/'DATA_FREEZE.json'),
        derived_freeze_sha256=sha(BASE/'DERIVED_FREEZE.json'))


def accounting():
    banks = 16*8*512
    return dict(data_sets=16, groups=128, groups_per_data=8,
        independent_observation_banks=banks, prior_vectors=banks*100,
        Gaussian_scalars=banks*100*260,
        mathematical_bank_pattern_scores=banks*2*10*16,
        original_API_calls=640, new_observations=0, new_training=0,
        new_model_integrals=0, new_neural_network_fits=0,
        automatic_retries=0)


def selftest():
    assert not (BASE/'FREEZE.json').exists(), 'Do not rerun development after scientific freeze'
    cfg = configuration()
    spec = read(SPEC)
    a = np.sin(np.arange(2*7*4)*.03).reshape(2, 7, 4)
    n = np.cos(np.arange(2*7*3)*.041).reshape(2, 7, 3)
    matrices = np.stack([np.eye(4), np.eye(4)*.8+.2*np.ones((4, 4)),
                        np.eye(4)*1.1-.1*np.ones((4, 4))])
    lowers = np.linalg.cholesky(matrices)
    values, _ = helper.evaluate_block(a, n, lowers, spec['mu'])
    direct = np.empty_like(values)
    for b in range(2):
        for j in range(3):
            z = a[b]@lowers[j].T + spec['mu']
            for y, bits in enumerate(helper.PATTERNS):
                p = expit((2*bits-1)*z).prod(axis=1)
                u = expit(-n[b]).prod(axis=1)
                direct[b, 0, j, y] = -math.log(float((p*u).mean()))
                direct[b, 1, j, y] = 3*math.log(2)-math.log(float(p.mean()))
    error = float(np.max(abs(values-direct)))
    assert error < 2e-14
    zero, _ = helper.evaluate_block(a, np.zeros_like(n), lowers, spec['mu'])
    assert np.max(abs(zero[:, 0]-zero[:, 1])) < 2e-14
    absent, _ = helper.evaluate_block(a, n[:, :, :0], lowers, spec['mu'])
    assert np.max(abs(values[:, 1]-absent[:, 1]-3*math.log(2))) < 2e-14
    names = [group_name(d, g) for d in cfg['data_ids'] for g in range(8)]
    seeds = [[cfg['seed_base'], d, g, role] for d in cfg['data_ids']
             for g in range(8) for role in [0, 1]]
    assert len(names) == len(set(names)) == 128
    assert len(seeds) == len({tuple(s) for s in seeds}) == 256
    assert local_indices(0) == list(range(10))
    assert local_indices(15) == [0, 1]+list(range(122, 130))
    assert sorted(i for d in range(16) for i in local_indices(d)[2:]) == list(range(2, 130))
    # Distinguish independent bank log losses from a pooled inner particle loss.
    scores = direct[:, 0, 0, 0]
    pooled = -math.log(float(np.exp(-scores).mean()))
    assert abs(pooled-float(scores.mean())) > 1e-8
    local = np.arange(130*16, dtype=float).reshape(130, 16)[local_indices(15)]
    assert local.shape == (10, 16) and np.array_equal(local[2], np.arange(122*16, 123*16))
    value = dict(passed=True, source_sha256=sha(SOURCE), helper_sha256=HELPER_SHA,
        reference_source_sha256=sha(reference.__file__), config_sha256=sha(CONFIG),
        spec_sha256=sha(SPEC), maximum_direct_error=error,
        group_filenames_unique=True, per_data_role_seed_keys_unique=True,
        local_candidate_maps_checked=16, inner_bank_log_order_checked=True,
        zero_nuisance_identity_checked=True, FACT_elimination_checked=True,
        planned_accounting=accounting(), new_RNG=0, new_model_integrals=0,
        new_training=0, original_API_calls=0)
    save(OUT/'SELFTEST.json', value)
    print(json.dumps(value), flush=True)


def run_group(data_id, group):
    cfg, selection = contract()
    assert data_id in cfg['data_ids'] and 0 <= group < cfg['groups']
    candidate_indices = local_indices(data_id)
    matrices = np.asarray([selection['candidates'][i]['covariance'] for i in candidate_indices])
    lowers = np.linalg.cholesky(matrices)
    mu = read(SPEC)['mu']
    entropy = {role: [cfg['seed_base'], data_id, group, role_id]
               for role, role_id in cfg['roles'].items()}
    rng = {role: np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
           for role, seed in entropy.items()}
    initial = {role: generator.bit_generator.state for role, generator in rng.items()}
    digest = {role: hashlib.sha256() for role in rng}
    blocks, prefix, controls = [], {}, None
    name = group_name(data_id, group)
    bind = bindings()
    started = time.perf_counter()
    progress = dict(data_id=data_id, group=group, candidate_indices=candidate_indices,
        completed_blocks=0, consumed_banks=0, **bind)
    save(OUT/f'progress_{name}.json', progress)
    try:
        for block in range(cfg['banks_per_group']//cfg['chunk']):
            a = rng['active'].standard_normal((64, 100, 4))
            n = rng['noise'].standard_normal((64, 100, 256))
            digest['active'].update(a.tobytes(order='C'))
            digest['noise'].update(n.tobytes(order='C'))
            values, _ = helper.evaluate_block(a, n, lowers, mu)
            blocks.append(values.mean(axis=0))
            if block == 0:
                prefix = dict(active_eps=a[:2].copy(), noise_eps=n[:2].copy(),
                    individual_total_score=values[:2].copy())
                if data_id in cfg['native_data_ids'] and group in cfg['native_groups']:
                    controls = helper.native_controls(a[:2], n[:2], values[:2], matrices,
                                                      mu, list(range(10)), 1e-4)
                    for check in controls['checks']:
                        check['global_candidate_index'] = candidate_indices[check['candidate_index']]
                    assert controls['official_API_calls'] == 320 and controls['passed']
            progress.update(completed_blocks=block+1, consumed_banks=(block+1)*64)
            save(OUT/f'progress_{name}.json', progress, False)
        prefix['block_means'] = np.stack(blocks)
        assert prefix['block_means'].shape == (8, 2, 10, 16)
        means = prefix['block_means'].mean(axis=0)
        assert means.shape == (2, 10, 16) and np.isfinite(means).all()
        p = OUT/f'{name}_prefix.npz'
        with p.open('xb') as stream:
            np.savez_compressed(stream, **prefix)
        assert bindings() == bind, 'Execution inputs changed during evaluation group'
        row = dict(data_id=data_id, group=group, candidate_indices=candidate_indices,
            means=means.tolist(), prefix_npz=rel(p), prefix_sha256=sha(p),
            prefix_array_sha256={key: ah(value) for key, value in prefix.items()},
            role_seed_sequences=entropy, RNG_initial_states=initial,
            RNG_final_states={role: generator.bit_generator.state for role, generator in rng.items()},
            role_input_sha256={role: value.hexdigest() for role, value in digest.items()},
            consumed_banks=512, completed_blocks=8, numerical_controls=controls,
            profile_order=cfg['profiles'], **bind, seconds=time.perf_counter()-started)
        save(OUT/f'{name}.json', row)
        return row
    except BaseException:
        save(OUT/f'FAILURE_{name}.json', dict(progress=progress, traceback=traceback.format_exc(),
            RNG_initial_states=initial,
            RNG_current_states={role: generator.bit_generator.state for role, generator in rng.items()},
            role_input_sha256={role: value.hexdigest() for role, value in digest.items()},
            automatic_retry=False))
        raise


def run():
    cfg, _ = contract()
    test = read(OUT/'SELFTEST.json')
    assert test['passed'] and test['source_sha256'] == sha(SOURCE)
    assert test['config_sha256'] == sha(CONFIG) and test['spec_sha256'] == sha(SPEC)
    assert test['reference_source_sha256'] == sha(reference.__file__)
    assert not any(list(OUT.glob(pattern)) for pattern in
        ['EXECUTION.json', 'group_*.json', 'group_*_prefix.npz', 'progress_*.json', 'FAILURE*.json'])
    bind = bindings()
    save(OUT/'EXECUTION.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
         **bind, workers=4, automatic_retry=False))
    tasks = [(d, g) for d in cfg['data_ids'] for g in range(cfg['groups'])]
    started = time.perf_counter()
    rows, active, next_task = {}, {}, 0
    pool = ProcessPoolExecutor(max_workers=4)
    try:
        while next_task < 4:
            pair = tasks[next_task]
            active[pool.submit(run_group, *pair)] = pair
            next_task += 1
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            fresh = [future.result() for future in done]
            for future in done:
                del active[future]
            for row in fresh:
                key = (row['data_id'], row['group'])
                rows[key] = row
                print(json.dumps(dict(completed_data_id=key[0], group=key[1])), flush=True)
            save(OUT/'PROGRESS.json', dict(completed=[list(k) for k in sorted(rows)],
                running=[list(k) for k in sorted(active.values())]), False)
            while len(active) < 4 and next_task < len(tasks):
                pair = tasks[next_task]
                active[pool.submit(run_group, *pair)] = pair
                next_task += 1
        contract()
        assert bindings() == bind
        ordered = [rows[pair] for pair in tasks]
        controls = [dict(data_id=row['data_id'], group=row['group'], **row['numerical_controls'])
                    for row in ordered if row['numerical_controls'] is not None]
        assert [(q['data_id'], q['group']) for q in controls] == [(0, 0), (15, 0)]
        qual = dict(records=controls, checks=[c for q in controls for c in q['checks']],
            official_API_calls=sum(q['official_API_calls'] for q in controls),
            pattern_means_checked=sum(q['pattern_means_checked'] for q in controls),
            maximum_absolute_difference=max(q['maximum_absolute_difference'] for q in controls),
            passed=all(q['passed'] for q in controls), tolerance=1e-4)
        assert qual['official_API_calls'] == 640 and qual['passed']
        result = dict(schema='scbmseq-evaluation-results-v1', **bind, groups=ordered,
            numerical_qualification=qual, profile_order=cfg['profiles'],
            accounting=accounting(), seconds=time.perf_counter()-started,
            scope='Mathematical64 finite-K scores; independent evaluation pool per new-data outer unit. '
                  'Group MC is conditional on trained local candidates; program-level inference uses '
                  '16 iid data units and averages8 groups and2 training repeats within data. '
                  'Native32 qualification onlyd0/15 group0 first2banks ten local matrices.')
        save(OUT/'RESULTS.json', result)
        save(OUT/'STATE.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
            results_sha256=sha(OUT/'RESULTS.json')))
        print(json.dumps(dict(complete=True, seconds=result['seconds'], native_passed=qual['passed'],
                              results_sha256=sha(OUT/'RESULTS.json'))), flush=True)
    except BaseException:
        for future in active:
            future.cancel()
        save(OUT/'FAILURE.json', dict(traceback=traceback.format_exc(),
            completed=[list(k) for k in sorted(rows)],
            already_active=[list(k) for k in sorted(active.values())], automatic_retry=False))
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['schema', 'check', 'run'])
    mode = parser.parse_args().mode
    if mode == 'schema':
        print(json.dumps(dict(source_sha256=sha(SOURCE), evaluation=configuration(), accounting=accounting())))
    elif mode == 'check':
        selftest()
    else:
        run()
