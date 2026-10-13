"""Independent REC1 audit; no producer imports or new scientific random banks.

Recompute all original-data counts, closed-form baseline identities, saved-PMF
risks, proposal rules, and paired-bank uncertainty. Replays only prespecified
existing RNG chunks for a direct-probability scoring check. --check is read-only.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rec_v1_20261004/REC1'
ARMS = ('NEAR_FREE', 'UNIT_FREE')
STATES = ('BASE', 'MU', 'RHO')
MODES = ('retained', 'eliminated')
TARGETS = ('F_D', 'F_P')
PATTERNS = np.array(list(itertools.product((0, 1), repeat=4)))
PAIRS = tuple(itertools.combinations(range(4), 2))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol=2e-12):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error <= tol, (error, tol)
    return error


def kl(P, Q):
    return math.fsum(float(p) * (math.log(float(p)) - math.log(float(q)))
                     for p, q in zip(P, Q))


def product(p):
    return np.array([math.prod(float(p[i]) if y[i] else 1 - float(p[i])
                               for i in range(4)) for y in PATTERNS])


def audit():
    result = read(OUT / 'RESULTS.json')
    freeze = read(OUT / 'FREEZE.json')
    assert result['freeze_sha256'] == sha(OUT / 'FREEZE.json')
    for group in ('sources', 'inputs'):
        for path, digest in freeze[group].items():
            assert sha(ROOT / path) == digest, path
    old_truth = read(ROOT / 'artifacts/active_scale_boundary/REFERENCE/RESULTS.json')
    P = np.array(next(r for r in old_truth['records'] if r['name'] == 'mixed')['teacher_pmf'])
    p = P @ PATTERNS
    tc = kl(P, product(p))
    close(result['teacher_P'], P)
    close(result['teacher_p'], p)
    close(result['true_product_risk'], tc)
    assert [r['stream'] for r in result['records']] == list(range(16))
    all_models, bank_arrays, baseline_risks = [], [], []
    max_score_error = max_risk_error = max_preserved_marginal_error = 0.
    max_direct_score_error = 0.
    for stream, record in enumerate(result['records']):
        old = read(ROOT / f'artifacts/joint_scale_learning/TRAIN/stream_{stream:02}.json')
        with np.load(ROOT / old['input_file'], allow_pickle=False) as data:
            labels = data['active_labels']
            counts = np.sum(labels, axis=0)
            empirical = np.array([np.count_nonzero(np.all(labels == y, axis=1)) / len(labels)
                                  for y in PATTERNS])
        assert labels.shape == (4096, 4)
        assert counts.tolist() == old['data_counts']
        fitted = (counts + .5) / 4097.
        baseline_risk = kl(P, product(fitted))
        marginal_cost = math.fsum(float(pi) * math.log(float(pi / qi)) +
                                 float(1 - pi) * math.log(float((1 - pi) / (1 - qi)))
                                 for pi, qi in zip(p, fitted))
        close(baseline_risk, tc + marginal_cost)
        baseline_risks.append(baseline_risk)
        for path_key, hash_key in (('model_file', 'model_sha256'), ('score_file', 'score_sha256')):
            assert sha(ROOT / record[path_key]) == record[hash_key]
        saved = read(ROOT / record['model_file'])
        close(saved['empirical'], empirical)
        baseline = saved['baseline']
        close(baseline['counts'], counts, 0.)
        close(baseline['p'], fitted)
        close(baseline['pmf'], product(fitted))
        close(baseline['risk'], baseline_risk)
        close(baseline['empirical_NLL'], -empirical @ np.log(product(fitted)))
        close(baseline['population_NLL'], -P @ np.log(product(fitted)))
        assert len(saved['records']) == 6
        for arm in ARMS:
            base = next(r for r in saved['records'] if r['arm'] == arm and r['state'] == 'BASE')
            original = old['outputs'][arm]['TAIL']
            for key in ('mu', 's', 'rho', 'tau', 'covariance'):
                close(base['model'][key], original[key])
            for name in ('MU', 'RHO'):
                candidate = next(r for r in saved['records'] if r['arm'] == arm and r['state'] == name)
                info = base['construction']['directions'][name]
                g = np.array(info['gradient'])
                gmax = float(np.max(abs(g)))
                step = np.zeros_like(g) if gmax <= 1e-14 else -.05 * g / gmax
                close(info['normalized_proposal_delta'], step)
                assert not info['teacher_used'] and not info['evaluation_bank_used']
                close(candidate['model']['s'], base['model']['s'], 0.)
                close(candidate['model']['tau'], base['model']['tau'], 0.)
                if name == 'MU':
                    close(candidate['model']['mu'], np.clip(np.array(base['model']['mu']) + step, -2, 2))
                    close(candidate['model']['rho'], base['model']['rho'], 0.)
                else:
                    proposed = np.array(base['model']['rho']) + step
                    C = np.eye(4)
                    for x, (i, j) in zip(proposed, PAIRS):
                        C[i, j] = C[j, i] = x
                    emin = float(np.linalg.eigvalsh(C).min())
                    factor = (1 - .1001) / (1 - emin) if emin < .1 else 1.
                    close(candidate['model']['rho'], proposed * factor)
                    close(candidate['model']['mu'], base['model']['mu'], 0.)
                    max_preserved_marginal_error = max(max_preserved_marginal_error,
                        close(np.array(candidate['main_pmf']) @ PATTERNS,
                              np.array(base['main_pmf']) @ PATTERNS, 2e-7))
        for row in saved['records']:
            Q = np.array(row['main_pmf'])
            assert np.all(Q > 0)
            close(Q.sum(), 1.)
            risk = kl(P, Q)
            max_risk_error = max(max_risk_error, close(row['risk'], risk))
            close(row['risk_minus_same_data_independent'], risk - baseline_risk)
            close(row['risk_minus_true_product'], risk - tc)
            close(row['own_product_risk'], kl(P, product(Q @ PATTERNS)))
            close(row['empirical_exact_NLL'], -empirical @ np.log(Q))
            close(row['population_exact_NLL'], -P @ np.log(Q))
            state = row['model']
            C = np.eye(4)
            for x, (i, j) in zip(state['rho'], PAIRS):
                C[i, j] = C[j, i] = x
            scale = np.array(state['s'])
            close(state['covariance'], np.outer(scale, scale) * C)
            assert np.linalg.eigvalsh(C).min() >= .1 - 1e-12
            assert row['qualification_passed'] == all(q['passed'] for q in row['qualifications'])
        all_models += saved['records']
        with np.load(ROOT / record['score_file'], allow_pickle=False) as data:
            banks = data['reduced_scores'].copy()
        assert banks.shape == (2048, 2, 3, 2, 2) and np.isfinite(banks).all()
        bank_arrays.append(banks)
        if stream in (0, 4, 8, 12):
            ra = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610046110, stream, 0])))
            rn = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610046110, stream, 1])))
            active = ra.standard_normal((32, 100, 4))
            nuisance = rn.standard_normal((32, 100, 256))
            assert hashlib.sha256(active.tobytes()).hexdigest() == record['bank_hashes'][0]['active']
            assert hashlib.sha256(nuisance.tobytes()).hexdigest() == record['bank_hashes'][0]['nuisance']
            for row in saved['records']:
                m = row['model']
                logits = active[0] @ np.linalg.cholesky(m['covariance']).T + m['mu']
                probabilities = 1 / (1 + np.exp(-logits))
                nuisance_probability = 1 / (1 + np.exp(-m['tau'] * nuisance[0]))
                nuisance_likelihood = np.prod(nuisance_probability, axis=1)
                pattern_scores = np.empty((2, 16))
                for j, y in enumerate(PATTERNS):
                    likelihood = np.prod(np.where(y, probabilities, 1 - probabilities), axis=1)
                    pattern_scores[0, j] = -math.log(math.fsum(likelihood * nuisance_likelihood) / 100) - 256 * math.log(2)
                    pattern_scores[1, j] = -math.log(math.fsum(likelihood) / 100)
                expected = np.stack([pattern_scores @ empirical, pattern_scores @ P], axis=-1)
                ai, si = ARMS.index(row['arm']), STATES.index(row['state'])
                max_direct_score_error = max(max_direct_score_error, close(expected, banks[0, ai, si]))
        assert len(record['bank_hashes']) == 64
        assert [h['start'] for h in record['bank_hashes']] == list(range(0, 2048, 32))
        assert len(record['score_rows']) == 24
        for row in record['score_rows']:
            ai, si, mi, ti = ARMS.index(row['arm']), STATES.index(row['state']), MODES.index(row['mode']), TARGETS.index(row['target'])
            values = banks[:, ai, si, mi, ti]
            delta = values - banks[:, ai, 0, mi, ti]
            for key, expected in (('reduced_mean', values.mean()), ('actual_mean', values.mean() + 256 * math.log(2)),
                                  ('MC_SE', values.std(ddof=1) / math.sqrt(2048)),
                                  ('delta_mean', delta.mean()), ('delta_bank_variance', delta.var(ddof=1))):
                max_score_error = max(max_score_error, close(row[key], expected))
    close(result['baseline_risks'], baseline_risks)
    close(result['baseline_mean_risk'], np.mean(baseline_risks))
    critical = float(t.ppf(1 - .05 / 32, 2047))
    close(result['MC_tcritical'], critical)
    assert len(result['primary_MC_family']) == 16
    for row in result['primary_MC_family']:
        ai, si, mi, ti = ARMS.index(row['arm']), STATES.index(row['candidate']), MODES.index(row['mode']), TARGETS.index(row['target'])
        differences = np.stack([bank[:, ai, si, mi, ti] - bank[:, ai, 0, mi, ti] for bank in bank_arrays])
        means = differences.mean(axis=1)
        mean = float(means.mean())
        se = float(np.sqrt(differences.var(axis=1, ddof=1).sum() / 2048) / 16)
        interval = [mean - critical * se, mean + critical * se]
        close(row['stream_paired_means'], means)
        close(row['delta'], mean)
        close(row['MC_SE'], se)
        close(row['interval'], interval)
        expected_direction = 'decrease' if interval[1] < 0 else 'increase' if interval[0] > 0 else 'unresolved'
        assert row['direction'] == expected_direction
        assert row['precision_passed'] == (se <= 2e-5)
    for row in result['descriptive_risk']:
        group = sorted([r for r in all_models if r['arm'] == row['arm'] and r['state'] == row['state']], key=lambda r: r['stream'])
        risks = [kl(P, r['main_pmf']) for r in group]
        deltas = [risk - next(base['risk'] for base in all_models if base['stream'] == r['stream'] and base['arm'] == r['arm'] and base['state'] == 'BASE') for risk, r in zip(risks, group)]
        close(row['risks'], risks)
        close(row['mean_risk'], np.mean(risks))
        close(row['paired_delta'], deltas)
        close(row['mean_delta'], np.mean(deltas))
        close(row['delta_range'], [min(deltas), max(deltas)])
        close(row['mean_versus_same_data_independent'], np.mean(np.array(risks) - baseline_risks))
        assert row['risk_decrease_resolved'] == (np.mean(deltas) < -2e-6)
    return dict(passed=True, source_sha256=sha(__file__), result_sha256=sha(OUT / 'RESULTS.json'),
                streams=16, original_outputs=32, candidates=64, baseline_recomputations=16,
                independent_PMFS_risk_recomputations=96, primary_MC_cells=16,
                max_saved_risk_difference=max_risk_error, max_saved_score_arithmetic_difference=max_score_error,
                direct_probability_pattern_scores=768, replayed_Gaussian=4 * 32 * 100 * 260,
                max_direct_probability_score_difference=max_direct_score_error,
                max_RHO_preserved_marginal_difference=max_preserved_marginal_error,
                scope='Independent arithmetic from original data, saved PMFs and all saved banks, plus prespecified first-bank direct probability score replay; no producer imports or new scientific banks/integrals; does not reproduce all scoring banks or quadrature kernels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = audit()
    target = OUT / 'INDEPENDENT_ARITHMETIC.json'
    if args.check:
        saved = read(target)
        assert saved['result'] == result
    else:
        with target.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), result=result), f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(result, ensure_ascii=False))
