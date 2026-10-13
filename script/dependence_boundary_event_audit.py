"""Independent three-concept GH audit of a fixed subset of sealed CUB models."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import math
import numpy as np
from scipy.special import expit, roots_hermitenorm
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/dependence_boundary_v1'
OUT = RUN / 'N_EVENT_AUDIT'
OLD = ROOT / 'artifacts/feedback_uncertainty_v3'
IDS = [f'{r}_K{k}_{s}_sd1' for r in ['I', 'O'] for k in [4, 100]
       for s in ['iid', 'antithetic']]
CONTRACT = dict(
    rep=0, identifiers=IDS, queries=[0, 8, 16], feedback=[0, 1],
    case_selection='Within each query/feedback among original T cases, minimize '
                   'SHA256(UTF8("boundary-audit:20269603|" + decimal image_id)); '
                   'break hash ties by image_id then original case_index.',
    selection_reads_partner_labels=False, max_model_cases=48,
    reference='Direct 3D Gaussian GH of P(query=y, partner state); denominator independent 1D GH',
    GH_orders=[64, 96], denominator_orders=[160, 240],
    E_scrambles=4, E_nodes_per_scramble=1024, E_sobol_seed=20269501,
    E_cdf_grids=[2049, 4097],
    GH_target_NLL_convergence_gate=1e-6, E_vs_GH_target_NLL_gate=.002,
    E_probability_floor=1e-12,
    classification='Use direct, unnormalized GH event-ratio target NLL; also retain '
                   'E-style floored/renormalized NLL and Brier for both methods. '
                   'Nonpositive or floor-level true target probability cannot pass.',
    scope='Local fixed-case numerical audit only, not a complete-risk error guarantee',
    failure_policy='Preserve every planned cell, including nonconvergence; no extra orders or cases',
    scientific_model_or_strategy_selection=False,
    replaces_E_primary_scores=False,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def upstream_check():
    nseal = read(RUN / 'N/MODELS_FREEZE.json')
    assert nseal['terminal_covariance_models'] == 96 and len(nseal['files']) == 16
    assert read(RUN / 'N/AUDIT.json')['passed']
    for manifest in [nseal, read(RUN / 'E/FREEZE.json')]:
        for path, digest in manifest['files'].items():
            assert sha(ROOT / path) == digest, path


def freeze():
    assert not OUT.exists(), 'Refuse to overwrite an existing event audit'
    upstream_check()
    write(OUT / 'CONTRACT.json', CONTRACT)
    inputs = [Path(__file__), OUT / 'CONTRACT.json', RUN / 'N/MODELS_FREEZE.json',
              RUN / 'N/AUDIT.json', RUN / 'E/FREEZE.json', RUN / 'E/CONFIG.json',
              ROOT / 'script/dependence_boundary_evaluate.py',
              OLD / 'D/IMAGE_ROLES.json', OLD / 'D/LABEL_DEFINITION.json',
              OLD / 'cache/cub_means.npz']
    inputs += [RUN / 'N/models' / f'{ident}.json' for ident in IDS]
    write(OUT / 'FREEZE.json', dict(time_utc=now(),
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        selection_or_scoring_started=False, upstream_models_sealed=96))
    print('event audit frozen before selecting or scoring T cases', flush=True)


def check():
    for path, digest in read(OUT / 'FREEZE.json')['files'].items():
        assert sha(ROOT / path) == digest, path
    upstream_check()


def selected_cases():
    # Independently reconstruct original test cases; do not use E.cases().
    rows = read(OLD / 'D/IMAGE_ROLES.json')
    definition = read(OLD / 'D/LABEL_DEFINITION.json')
    prototype = np.asarray(definition['class_labels'])
    test = [i for i, row in enumerate(rows) if row['role'] == 'test_locked']
    candidates = {(q, y): [] for q in CONTRACT['queries'] for y in CONTRACT['feedback']}
    for local, global_index in enumerate(test):
        row = rows[global_index]
        for query_position, query in enumerate(row['query_ids']):
            if query not in CONTRACT['queries']:
                continue
            feedback = int(prototype[row['class_id'] - 1, query])
            ident = int(row['image_id'])
            key = hashlib.sha256(f'boundary-audit:20269603|{ident}'.encode('utf-8')).hexdigest()
            candidates[(query, feedback)].append(dict(
                image_id=ident, global_index=global_index, case_index=3 * local + query_position,
                query=query, feedback=feedback, selection_hash=key, group_id=row['group_id']))
    selection, absent = [], []
    for key, values in candidates.items():
        if not values:
            absent.append(dict(query=key[0], feedback=key[1]))
            continue
        winner = min(values, key=lambda x: (x['selection_hash'], x['image_id'], x['case_index']))
        selection.append(winner)
    # Partner labels are attached only after the selected IDs are fixed.
    for case in selection:
        row = rows[case['global_index']]
        case['partners'] = definition['triplets'][case['query']]['partners']
        assert case['query'] not in case['partners'] and len(set(case['partners'])) == 2
        case['partner_labels'] = prototype[row['class_id'] - 1, case['partners']].astype(int).tolist()
    return selection, absent


@lru_cache(maxsize=2)
def tensor(order):
    nodes, weights = roots_hermitenorm(order)
    weights /= math.sqrt(2 * math.pi)
    index = np.indices((order, order, order)).reshape(3, -1).T
    return nodes[index], np.prod(weights[index], axis=1)


def denominator(mu, variance, feedback, order):
    z, w = roots_hermitenorm(order)
    w /= math.sqrt(2 * math.pi)
    return float(w @ expit((2 * feedback - 1) * (mu + math.sqrt(variance) * z)))


def direct_integral(mu, covariance, feedback, order, denominator_order):
    z, w = tensor(order)
    chol = np.linalg.cholesky(covariance)
    eta = mu[None, :] + z @ chol.T
    observed_query = expit((2 * feedback - 1) * eta[:, 0])
    pp, pn = expit(eta[:, 1:]), expit(-eta[:, 1:])
    states = np.stack([pn[:, 0] * pn[:, 1], pn[:, 0] * pp[:, 1],
                       pp[:, 0] * pn[:, 1], pp[:, 0] * pp[:, 1]], axis=1)
    prior = w @ states
    numerator = (w * observed_query) @ states
    den = denominator(mu[0], covariance[0, 0], feedback, denominator_order)
    assert den > 0 and np.isfinite(den)
    conditional = numerator / den
    return dict(NP=prior.tolist(), EQ=conditional.tolist(), numerator=numerator.tolist(),
                denominator=den, numerator_query_mass=float(numerator.sum()),
                conditional_mass=float(conditional.sum()), prior_mass=float(prior.sum()),
                GH_order=order, denominator_order=denominator_order)


def metrics(probability, labels):
    q = np.asarray(probability, dtype=float)
    state = 2 * labels[0] + labels[1]
    target = float(q[state])
    raw_nll = -math.log(target) if target > 0 and math.isfinite(target) else None
    marginal = np.asarray([q[2] + q[3], q[1] + q[3]])
    clipped = np.maximum(q, CONTRACT['E_probability_floor'])
    clipped /= clipped.sum()
    cmarg = np.asarray([clipped[2] + clipped[3], clipped[1] + clipped[3]])
    return dict(raw_target_probability=target, raw_target_NLL=raw_nll,
                raw_Brier=float(np.mean((marginal - labels) ** 2)),
                floored_normalized_NLL=float(-math.log(clipped[state])),
                floored_normalized_Brier=float(np.mean((cmarg - labels) ** 2)),
                true_target_floor=bool(target <= CONTRACT['E_probability_floor']),
                all_floor_entries=int(np.sum(q < CONTRACT['E_probability_floor'])),
                probability_sum=float(q.sum()))


def score_e(mu_all, covariance, case, us, grid):
    # E is the implementation under test, never the GH reference.
    from script.dependence_boundary_evaluate import predict
    from script.shared_uncertainty_holdout_inference import blocks
    query = np.asarray([case['query']])
    partners = np.asarray([case['partners']])
    feedback = np.asarray([case['feedback']])
    block = blocks(mu_all[None, :], covariance, query, partners, feedback)
    values = [predict(block, u, grid, 512)[0][0] for u in us]
    pool = np.mean(values, axis=0)
    return dict(NP=pool[0].tolist(), EQ=pool[1].tolist(),
                grid=grid, total_nodes=4096, stream_probabilities=np.asarray(values).tolist())


def run():
    check()
    assert not (OUT / 'CASE_SELECTION.json').exists()
    cases, absent = selected_cases()
    write(OUT / 'CASE_SELECTION.json', dict(time_utc=now(), cases=cases, absent_groups=absent,
                                           partner_labels_used_for_selection=False))
    assert len(cases) <= 6
    mu = np.load(OLD / 'cache/cub_means.npz')['means'][0].astype(float)
    us = [qmc.Sobol(3, scramble=True, seed=CONTRACT['E_sobol_seed'] + stream).random_base2(10)
          for stream in range(4)]
    results, rows = [], []
    for identifier in IDS:
        model = read(RUN / 'N/models' / f'{identifier}.json')
        factor = np.asarray(model['factors'][0])
        covariance = factor @ factor.T
        for case in cases:
            m = mu[case['global_index']]
            axes = [case['query'], *case['partners']]
            submu, subcov = m[axes], covariance[np.ix_(axes, axes)]
            reference = [direct_integral(submu, subcov, case['feedback'], order, denorder)
                         for order, denorder in zip(CONTRACT['GH_orders'], CONTRACT['denominator_orders'])]
            tested = [score_e(m, covariance, case, us, grid) for grid in CONTRACT['E_cdf_grids']]
            detail = dict(identifier=identifier, rep=0, case=case, mu=submu.tolist(),
                          covariance=subcov.tolist(), reference=reference, tested=tested, comparisons={})
            for method in ['NP', 'EQ']:
                rm = [metrics(r[method], case['partner_labels']) for r in reference]
                em = [metrics(r[method], case['partner_labels']) for r in tested]
                valid = all(x['raw_target_NLL'] is not None for x in rm + em)
                gh_delta = abs(rm[1]['raw_target_NLL'] - rm[0]['raw_target_NLL']) if valid else None
                e_delta = abs(em[0]['raw_target_NLL'] - rm[1]['raw_target_NLL']) if valid else None
                cdf_delta = abs(em[1]['raw_target_NLL'] - em[0]['raw_target_NLL']) if valid else None
                not_floored = all(not x['true_target_floor'] for x in rm + em)
                converged = bool(valid and not_floored and gh_delta < CONTRACT['GH_target_NLL_convergence_gate'])
                agree = bool(converged and e_delta < CONTRACT['E_vs_GH_target_NLL_gate'])
                comparison = dict(
                    reference_metrics=rm, E_metrics=em, GH_target_NLL_delta=gh_delta,
                    E_vs_GH_target_NLL_delta=e_delta, CDF_target_NLL_delta=cdf_delta,
                    E_vs_GH_Brier_delta=abs(em[0]['floored_normalized_Brier'] - rm[1]['floored_normalized_Brier']),
                    true_target_no_floor=not_floored, reference_convergence_pass=converged,
                    E_agreement_pass=agree,
                    classification='local_agreement' if agree else ('E_disagreement' if converged else 'reference_unresolved'),
                )
                detail['comparisons'][method] = comparison
                rows.append(dict(identifier=identifier, rep=0, image_id=case['image_id'],
                    query=case['query'], feedback=case['feedback'], method=method,
                    GH64_NLL=rm[0]['raw_target_NLL'], GH96_NLL=rm[1]['raw_target_NLL'],
                    E_pool_NLL=em[0]['raw_target_NLL'], GH_target_NLL_delta=gh_delta,
                    E_vs_GH_target_NLL_delta=e_delta, CDF_target_NLL_delta=cdf_delta,
                    E_Brier=em[0]['floored_normalized_Brier'], GH96_Brier=rm[1]['floored_normalized_Brier'],
                    true_target_no_floor=not_floored, reference_convergence_pass=converged,
                    E_agreement_pass=agree, classification=comparison['classification']))
            results.append(detail)
        print('audited event cases', identifier, len(cases), flush=True)
    write(OUT / 'RESULTS.json', dict(time_utc=now(), model_cases=len(results), results=results))
    with (OUT / 'SUMMARY.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(r['classification'] == name for r in rows)
              for name in ['local_agreement', 'E_disagreement', 'reference_unresolved']}
    complete = dict(time_utc=now(), status='completed_including_all_numerical_failures',
        identifiers=8, rep=0, distinct_cases=len(cases), model_cases=len(results), method_cases=len(rows),
        classification_counts=counts, no_extra_orders_or_cases=True, E_primary_unchanged=True,
        scope=CONTRACT['scope'], files={p.name: sha(p) for p in
        [OUT / 'CASE_SELECTION.json', OUT / 'RESULTS.json', OUT / 'SUMMARY.csv']})
    write(OUT / 'COMPLETE.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'run'])
    args = parser.parse_args()
    globals()[args.phase]()
