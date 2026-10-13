"""Frozen paired evaluation; pool probabilities before applying the log score."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri, expit
from script.dependence_v4_core import (
    data, cases, means, score, normalize, product, bootstrap, csvwrite,
    now, sha, read, write,
)
from script.shared_uncertainty_holdout_inference import blocks, take, batch_grid
from script.shared_uncertainty_core import emission

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/dependence_boundary_v1'
E = RUN / 'E'
DOC = ROOT / 'docs/08_research_研究工作/边际依赖接口_20260929'
METHODS = ['NP', 'EQ', 'NPp', 'EQp', 'IB']
CFG = dict(
    roles=['I', 'O'], K=[4, 100], samplers=['iid', 'antithetic'],
    initial_sd=1., reps=6, streams=4, nodes=1024, half_nodes=512,
    sobol_seed=20269501, cdf_grid=2049, cdf_check_grid=4097,
    cdf_target_NLL_max_difference=1e-4, cdf_partner_extreme_cases=32,
    MC_SE_gate=.0005, node_sensitivity_gate=.0005,
    probability_floor=1e-12, contrasts=8, bootstrap=10000,
    bootstrap_seed=20269502, material_nats=.005,
    primary='four-scramble pooled probability, then log; 4096 total nodes',
    sensitivity='pool first 512 nodes of each scramble; 2048 total nodes',
    target_floor_gate='no true-target floor in pool, half-pool, streams, or CDF checks',
    numerical_scope='preset robustness gates; not a rigorous finite-MC bias bound',
)


def seal(name, paths, **extra):
    write(E / name, dict(time=now(), files={p.relative_to(ROOT).as_posix(): sha(p)
                                           for p in paths}, **extra))


def check(name):
    for p, h in read(E / name)['files'].items():
        assert sha(ROOT / p) == h, p
    check_all_models_sealed()


def identifiers():
    return [f'{r}_K{k}_{s}_sd1' for r in CFG['roles']
            for k in CFG['K'] for s in CFG['samplers']]


def check_all_models_sealed():
    """All 96 terminal models must remain intact before test-array access."""
    manifest = read(RUN / 'N/MODELS_FREEZE.json')
    audit = read(RUN / 'N/AUDIT.json')
    assert manifest['terminal_covariance_models'] == 96
    assert manifest['mean_models'] == 6 and manifest['new_mean_models'] == 0
    assert manifest['C_T_evaluation_started'] is False
    assert audit['passed'] and audit['models'] == 96 and audit['configurations'] == 16
    assert audit['all_finite'] and audit['all_covariances_positive_definite_from_factor']
    expected = {}
    for role in CFG['roles']:
        for k in CFG['K']:
            for sampler in CFG['samplers']:
                for tag, sd in [('001', .01), ('1', 1.)]:
                    ident = f'{role}_K{k}_{sampler}_sd{tag}'
                    name = (RUN / 'N/models' / f'{ident}.json').relative_to(ROOT).as_posix()
                    expected[name] = dict(role=role, K=k, sampler=sampler, initial_sd=sd)
    assert set(manifest['files']) == set(expected) and len(expected) == 16
    for path, config in expected.items():
        assert sha(ROOT / path) == manifest['files'][path], path
        model = read(ROOT / path)
        assert model['config'] == config and model['total_steps'] == 2000
        assert model['seeds'] == list(range(20266001, 20266007))
        factors = np.asarray(model['factors'])
        assert factors.shape == (6, 24, 24) and np.isfinite(factors).all()
    return [ROOT / path for path in sorted(expected)]


def freeze():
    assert not (E / 'FREEZE.json').exists()
    models = check_all_models_sealed()
    write(E / 'CONFIG.json', CFG)
    paths = [Path(__file__), DOC / 'PLAN_E.md', E / 'CONFIG.json',
             RUN / 'N/MODELS_FREEZE.json', RUN / 'N/AUDIT.json',
             ROOT / 'script/dependence_boundary_natural.py',
             ROOT / 'script/dependence_v4_core.py',
             ROOT / 'script/shared_uncertainty_holdout_inference.py',
             ROOT / 'script/shared_uncertainty_core.py',
             ROOT / 'artifacts/feedback_uncertainty_v3/cache/cub_means.npz',
             ROOT / 'artifacts/feedback_uncertainty_v3/D/IMAGE_ROLES.json',
             ROOT / 'artifacts/feedback_uncertainty_v3/D/LABEL_DEFINITION.json'] + models
    seal('FREEZE.json', paths, all_terminal_models_checked=96)


def predict(p, u, grid_n, half_nodes, grid=None, normal=None):
    """Return NP/EQ integrals, with no partner labels accepted."""
    assert 0 < half_nodes <= len(u) and np.all((u > 0) & (u < 1))
    z = ndtri(u) if normal is None else normal
    assert z.shape == u.shape
    eps = np.einsum('kj,bij->bki', z[:, 1:], p['L'])
    mu, load = p['mu'][:, None, :], p['b'][:, None, :]
    gz, gc = batch_grid(p, grid_n) if grid is None else grid
    assert gz.shape == gc.shape == (len(p['y']), grid_n)
    post = np.array([np.interp(u[:, 0], gc[i], gz[i]) for i in range(len(p['y']))])
    prior = emission(mu + load * z[None, :, 0, None] + eps)
    posterior = emission(mu + load * post[:, :, None] + eps)
    full = np.stack([prior.mean(1), posterior.mean(1)], axis=1)
    half = np.stack([prior[:, :half_nodes].mean(1),
                     posterior[:, :half_nodes].mean(1)], axis=1)
    assert np.isfinite(full).all() and np.isfinite(half).all()
    return full, half


def marginal_product(q):
    return product(np.column_stack([q[:, 2] + q[:, 3], q[:, 1] + q[:, 3]]))


def all_methods(base, independent):
    """Product controls are formed after the requested probability pooling."""
    return np.stack([base[:, 0], base[:, 1], marginal_product(base[:, 0]),
                     marginal_product(base[:, 1]), independent], axis=1)


def target_probabilities(q, labels):
    states = (2 * labels[:, 0] + labels[:, 1]).astype(int)
    return q[np.arange(len(q))[:, None], np.arange(len(METHODS))[None, :], states[:, None]]


def score_methods(q, labels):
    """Declared 1e-12 floor; separately count true-target hits."""
    assert q.shape == (len(labels), len(METHODS), 4) and np.isfinite(q).all()
    true_target_floors = (target_probabilities(q, labels) <= CFG['probability_floor']).sum(0)
    scored, normalized, entries = [], [], []
    for j in range(len(METHODS)):
        norm, count = normalize(q[:, j])
        normalized.append(norm)
        scored.append(score(q[:, j], labels))
        entries.append(count)
    return (np.stack(scored, axis=1), np.stack(normalized, axis=1),
            np.asarray(entries, dtype=int), true_target_floors)


def influence_values(stream_probabilities, pooled_probabilities, labels):
    target = target_probabilities(pooled_probabilities, labels)
    result = np.stack([-(target_probabilities(q, labels) - target) / target
                       for q in stream_probabilities])
    assert np.isfinite(result).all()
    return result


def cdf_cases(p, query):
    """Selection uses query/feedback and logits, never partner labels."""
    selected = set()
    for concept in np.unique(query):
        for feedback in [0, 1]:
            ix = np.flatnonzero((query == concept) & (p['y'] == feedback))
            if len(ix):
                selected.add(int(ix[np.argmin(p['query_mu'][ix])]))
                selected.add(int(ix[np.argmax(p['query_mu'][ix])]))
    extremeness = np.max(np.abs(p['mu']), axis=1)
    selected.update(np.argsort(-extremeness, kind='stable')[:CFG['cdf_partner_extreme_cases']].tolist())
    return np.asarray(sorted(selected), dtype=int)


def audit_cdf(p, query, labels, independent, u):
    selected = cdf_cases(p, query)
    pp = take(p, selected)
    low, _ = predict(pp, u, CFG['cdf_grid'], CFG['half_nodes'])
    high, _ = predict(pp, u, CFG['cdf_check_grid'], CFG['half_nodes'])
    qlow, qhigh = all_methods(low, independent[selected]), all_methods(high, independent[selected])
    slow, _, _, fl = score_methods(qlow, labels[selected])
    shigh, _, _, fh = score_methods(qhigh, labels[selected])
    error = np.max(np.abs(slow[:, :, 0] - shigh[:, :, 0]), axis=0)
    probability_error = np.max(np.abs(qlow - qhigh), axis=(0, 2))
    return dict(
        case_indices=selected.tolist(), covered_cases=len(selected), total_cases=len(query),
        selection='query x feedback min/max query_mu plus top32 partner abs(mu); deduplicated',
        selection_used_partner_labels=False, checked_stream=0,
        grid_target_NLL_error={m: float(error[j]) for j, m in enumerate(METHODS)},
        grid_probability_error={m: float(probability_error[j]) for j, m in enumerate(METHODS)},
        target_floors={m: int(fl[j] + fh[j]) for j, m in enumerate(METHODS)},
        method_pass={m: bool(error[j] < CFG['cdf_target_NLL_max_difference'] and fl[j] + fh[j] == 0)
                     for j, m in enumerate(METHODS)},
        scope='parameter-extreme subset only, not a uniform CDF or NLL error bound',
    )


def one(identifier):
    check('FREEZE.json')  # All 96 models, before means/cases/test-label access.
    assert identifier in identifiers()
    out = E / 'scores' / identifier
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'COMPLETE.json').exists()
    model = read(RUN / 'N/models' / f'{identifier}.json')
    mu = means('old')
    ix, local, query, partners, feedback, labels = cases('T')
    assert len(ix) == 5794 and len(local) == 3 * len(ix)
    power = int(math.log2(CFG['nodes']))
    assert 2 ** power == CFG['nodes']
    us = [qmc.Sobol(3, scramble=True, seed=CFG['sobol_seed'] + i).random_base2(power)
          for i in range(CFG['streams'])]
    normals = [ndtri(u) for u in us]
    summaries, started = [], time.perf_counter()
    for rep in range(CFG['reps']):
        path = out / f'{rep}.npz'
        assert not path.exists()
        factor = np.asarray(model['factors'][rep])
        p = blocks(mu[rep, ix[local]], factor @ factor.T, query, partners, feedback)
        independent = product(expit(np.take_along_axis(mu[rep, ix[local]], partners, axis=1)))
        cdf = audit_cdf(p, query, labels, independent, us[0])
        stream_scores = np.zeros((CFG['streams'], len(local), len(METHODS), 3))
        pool_scores = np.zeros((len(local), len(METHODS), 3))
        half_scores = np.zeros_like(pool_scores)
        influences = np.zeros((CFG['streams'], len(local), len(METHODS)))
        floor_entries = {k: np.zeros(len(METHODS), dtype=int) for k in ['pool', 'half', 'streams']}
        target_floors = {k: np.zeros(len(METHODS), dtype=int) for k in ['pool', 'half', 'streams']}
        for begin in range(0, len(local), 128):
            sl = slice(begin, begin + 128)
            pp, yy, ib = take(p, sl), labels[sl], independent[sl]
            grid = batch_grid(pp, CFG['cdf_grid'])
            full_bases, half_bases, stream_probabilities = [], [], []
            for st, u in enumerate(us):
                full, half = predict(pp, u, CFG['cdf_grid'], CFG['half_nodes'],
                                     grid=grid, normal=normals[st])
                full_bases.append(full)
                half_bases.append(half)
                value, normalized, floor, tfloor = score_methods(all_methods(full, ib), yy)
                stream_scores[st, sl] = value
                stream_probabilities.append(normalized)
                floor_entries['streams'] += floor
                target_floors['streams'] += tfloor
            pooled = all_methods(np.mean(full_bases, axis=0), ib)
            half_pooled = all_methods(np.mean(half_bases, axis=0), ib)
            value, normalized_pool, floor, tfloor = score_methods(pooled, yy)
            pool_scores[sl] = value
            floor_entries['pool'] += floor
            target_floors['pool'] += tfloor
            value, _, floor, tfloor = score_methods(half_pooled, yy)
            half_scores[sl] = value
            floor_entries['half'] += floor
            target_floors['half'] += tfloor
            influences[:, sl] = influence_values(stream_probabilities, normalized_pool, yy)
        arrays = dict(
            pooled_scores=pool_scores.reshape(len(ix), 3, len(METHODS), 3).mean(1),
            half_pooled_scores=half_scores.reshape(len(ix), 3, len(METHODS), 3).mean(1),
            influence=influences.reshape(CFG['streams'], len(ix), 3, len(METHODS)).mean(2),
            stream_scores=stream_scores.reshape(CFG['streams'], len(ix), 3, len(METHODS), 3).mean(2),
        )
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(path, **arrays)
        method_pass = {
            m: bool(cdf['method_pass'][m] and sum(a[j] for a in target_floors.values()) == 0)
            for j, m in enumerate(METHODS)
        }
        summaries.append(dict(
            rep=rep, cdf_audit=cdf, numerical_method_pass=method_pass,
            floor_entries={k: {m: int(a[j]) for j, m in enumerate(METHODS)} for k, a in floor_entries.items()},
            true_target_floors={k: {m: int(a[j]) for j, m in enumerate(METHODS)} for k, a in target_floors.items()},
            sha=sha(path),
        ))
        print('boundary natural scored', identifier, rep,
              {m: round(float(arrays['pooled_scores'][:, j, 0].mean()), 6)
               for j, m in enumerate(METHODS)}, round(time.perf_counter() - started, 1), flush=True)
    write(out / 'COMPLETE.json', dict(time=now(), id=identifier, models=summaries,
                                     seconds=time.perf_counter() - started))


def paired_statistics(iid, anti, method_index):
    """Arrays have rep, [stream], image, method, [metric] axes."""
    difference = (iid['pooled_scores'][:, :, method_index, 0]
                  - anti['pooled_scores'][:, :, method_index, 0]).mean(0)
    half_difference = (iid['half_pooled_scores'][:, :, method_index, 0]
                       - anti['half_pooled_scores'][:, :, method_index, 0]).mean(0)
    inf = (iid['influence'][:, :, :, method_index]
           - anti['influence'][:, :, :, method_index]).mean((0, 2))
    stream_risk_difference = (iid['stream_scores'][:, :, :, method_index, 0]
                              - anti['stream_scores'][:, :, :, method_index, 0]).mean((0, 2))
    return difference, dict(
        half_pooled_difference=float(half_difference.mean()),
        node_sensitivity=abs(float(difference.mean() - half_difference.mean())),
        MC_SE=float(inf.std(ddof=1) / math.sqrt(len(inf))),
        stream_influences=inf.tolist(),
        mean_stream_risk_difference=float(stream_risk_difference.mean()),
        paired_pool_vs_mean_NLL_gap=float(stream_risk_difference.mean() - difference.mean()),
    )


def collect():
    check('FREEZE.json')
    assert not (E / 'COMPLETE.json').exists()
    scores, receipts, summary, streams = {}, {}, [], []
    for identifier in identifiers():
        out = E / 'scores' / identifier
        receipt = read(out / 'COMPLETE.json')
        assert receipt['id'] == identifier and len(receipt['models']) == CFG['reps']
        receipts[identifier] = receipt
        arrays = []
        for rep in range(CFG['reps']):
            assert receipt['models'][rep]['rep'] == rep
            path = out / f'{rep}.npz'
            assert sha(path) == receipt['models'][rep]['sha']
            with np.load(path) as archive:
                a = {key: archive[key] for key in archive.files}
            assert a['pooled_scores'].shape == a['half_pooled_scores'].shape == (5794, 5, 3)
            assert a['stream_scores'].shape == (4, 5794, 5, 3)
            assert a['influence'].shape == (4, 5794, 5)
            arrays.append(a)
            for j, method in enumerate(METHODS):
                metrics = a['pooled_scores'][:, j].mean(0)
                mean_stream = float(a['stream_scores'][:, :, j, 0].mean())
                summary.append(dict(
                    id=identifier, rep=rep, method=method, NLL=float(metrics[0]),
                    Brier=float(metrics[1]), parity_Brier=float(metrics[2]),
                    half_pooled_NLL=float(a['half_pooled_scores'][:, j, 0].mean()),
                    mean_stream_NLL=mean_stream, mean_NLL_minus_pooled_NLL=mean_stream - float(metrics[0]),
                ))
                for st in range(CFG['streams']):
                    streams.append(dict(id=identifier, rep=rep, stream=st, method=method,
                                        NLL=float(a['stream_scores'][st, :, j, 0].mean()),
                                        influence=float(a['influence'][st, :, j].mean())))
        scores[identifier] = {key: np.stack([a[key] for a in arrays]) for key in arrays[0]}
    names, differences, diagnostics, numerical_ok = [], [], [], []
    for role in CFG['roles']:
        for k in CFG['K']:
            ii, ai = f'{role}_K{k}_iid_sd1', f'{role}_K{k}_antithetic_sd1'
            for method in ['NP', 'EQ']:
                diff, diag = paired_statistics(scores[ii], scores[ai], METHODS.index(method))
                differences.append(diff)
                diagnostics.append(diag)
                names.append(f'{role}/K{k}/{method}/iid_minus_antithetic')
                numerical_ok.append(all(r['numerical_method_pass'][method]
                                        for ident in [ii, ai] for r in receipts[ident]['models']))
    diffs = np.stack(differences, axis=1)
    assert diffs.shape == (5794, CFG['contrasts'])
    boot, lo, hi = bootstrap(diffs, CFG['bootstrap_seed'])
    assert boot.shape == (CFG['bootstrap'], CFG['contrasts'])
    primary = []
    for j, name in enumerate(names):
        diag = diagnostics[j]
        ok = bool(numerical_ok[j] and diag['MC_SE'] < CFG['MC_SE_gate']
                  and diag['node_sensitivity'] < CFG['node_sensitivity_gate'])
        primary.append(dict(
            contrast=name, mean=float(diffs[:, j].mean()), lower=float(lo[j]), upper=float(hi[j]),
            **{k: v for k, v in diag.items() if k != 'stream_influences'},
            cdf_and_floor_pass=numerical_ok[j], precision_pass=ok,
            material_pass=bool(ok and lo[j] > CFG['material_nats']),
        ))
    ix, *_ = cases('T')
    rows, _, _ = data()
    image_rows = []
    pooled = {k: a['pooled_scores'].mean(0) for k, a in scores.items()}
    for i, global_i in enumerate(ix):
        row = dict(image_id=rows[global_i]['image_id'], group_id=rows[global_i]['group_id'])
        for identifier, a in pooled.items():
            for j, method in enumerate(METHODS):
                row[f'{identifier}/{method}/NLL'] = float(a[i, j, 0])
        image_rows.append(row)
    csvwrite(E / 'SUMMARY.csv', summary)
    csvwrite(E / 'STREAMS.csv', streams)
    csvwrite(E / 'IMAGE_METRICS.csv', image_rows)
    csvwrite(E / 'PRIMARY.csv', primary)
    write(E / 'BOOTSTRAP.json', dict(seed=CFG['bootstrap_seed'], contrasts=names, values=boot.tolist()))
    write(E / 'NUMERICAL_DIAGNOSTICS.json', dict(
        contrasts={n: d for n, d in zip(names, diagnostics)},
        models={ident: r['models'] for ident, r in receipts.items()},
        claim='preset numerical robustness gates only; no rigorous finite-MC error or bias bound',
    ))
    seal('RESULTS_FREEZE.json', [E / name for name in
                                ['SUMMARY.csv', 'STREAMS.csv', 'IMAGE_METRICS.csv', 'PRIMARY.csv',
                                 'BOOTSTRAP.json', 'NUMERICAL_DIAGNOSTICS.json']])
    write(E / 'COMPLETE.json', dict(time=now(), status='complete', primary=primary,
                                    all_primary_precision=all(p['precision_pass'] for p in primary)))
    print(primary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'one', 'collect'])
    parser.add_argument('--id')
    args = parser.parse_args()
    if args.phase == 'one':
        one(args.id)
    else:
        globals()[args.phase]()
