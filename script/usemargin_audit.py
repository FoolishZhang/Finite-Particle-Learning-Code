"""Independent Decimal arithmetic audit of saved USE margins; no producer imports."""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/use_margin_v4_20261005'
EPS = Decimal('1e-12')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def decimal(x):
    return Decimal(str(x))


def choose(x):
    low = min(x)
    return next(i for i, value in enumerate(x) if value - low <= EPS)


def audit():
    result = read(OUT / 'RESULTS.json')
    seen = set()
    errors = []
    verified = []
    def close(actual, expected):
        error = abs(decimal(actual) - expected)
        errors.append(float(error))
        assert error < Decimal('2e-12'), (actual, str(expected))
    with localcontext() as context:
        context.prec = 50
        for row in result['records']:
            key = (row['study'], row['case'], row['rep'], row['method'])
            assert key not in seen
            seen.add(key)
            selection = read(ROOT / row['source_selection'])
            original = read(ROOT / row['source_result'])
            assert original['selection_sha256'] == sha(ROOT / row['source_selection'])
            raw = selection.get('scores', selection.get('score_matrix'))
            method_index = selection['methods'].index(row['method'])
            ell = list(map(decimal, raw[4]))
            approx = list(map(decimal, raw[method_index]))
            risk = list(map(decimal, original['candidate_risks' if row['study'] == 'USE-v2' else 'candidate_test_NLL']))
            test = list(map(decimal, original['candidate_test_NLL']))
            w, jhat = choose(ell), choose(approx)
            assert (w, jhat) == (row['reference_winner'], row['selected'])
            assert jhat == selection['choices'][method_index] == original['choices'][method_index]
            assert row['changed'] == (jhat != w)
            assert row['reference_scores'] == raw[4] and row['approximate_scores'] == raw[method_index]
            error = [a - b for a, b in zip(approx, ell)]
            rivals = [j for j in range(4) if j != w]
            for actual, expected in zip(row['realized_errors'], error):
                close(actual, expected)
            assert row['candidate_risk'] == original['candidate_risks' if row['study'] == 'USE-v2' else 'candidate_test_NLL']
            assert row['candidate_test_NLL'] == original['candidate_test_NLL']
            delta = min(ell[j] - ell[w] for j in rivals)
            amin = min(approx[j] - approx[w] for j in rivals)
            spread = max(error) - min(error)
            amplitude = max(abs(e) for e in error)
            close(row['reference_min_gap'], delta)
            close(row['approximate_min_gap'], amin)
            close(row['error_range'], spread)
            close(row['max_absolute_error'], amplitude)
            close(row['max_winner_relative_error'], max(abs(error[j] - error[w]) for j in rivals))
            close(row['joint_error_range'], max(error[1:]) - min(error[1:]))
            close(row['joint_centered_max_error'], (max(error[1:]) - min(error[1:])) / 2)
            assert row['absolute_bound_certificate'] == (delta > 2 * amplitude + EPS)
            assert row['range_bound_certificate'] == (delta > spread + EPS)
            if row['range_bound_certificate'] or row['absolute_bound_certificate']:
                assert jhat == w
            assert row['reference_near_tie'] == (delta <= EPS)
            assert row['approximate_near_tie'] == (abs(amin) <= EPS)
            assert row['strict_crossings'] == sum(approx[j] - approx[w] < -EPS for j in rivals)
            assert row['reference_runner_up'] == min(rivals, key=lambda j: (ell[j], j))
            assert row['approximate_strongest_rival'] == min(rivals, key=lambda j: (approx[j], j))
            runner = row['reference_runner_up']
            close(row['runner_reference_gap'], ell[runner] - ell[w])
            close(row['runner_relative_error'], error[runner] - error[w])
            close(row['runner_approximate_gap'], approx[runner] - approx[w])
            assert len(row['pairs']) == 3
            for pair, j in zip(row['pairs'], rivals):
                assert pair['candidate'] == j
                g, d, a = ell[j] - ell[w], error[j] - error[w], approx[j] - approx[w]
                assert g + d == a
                close(pair['reference_gap'], g)
                close(pair['relative_error'], d)
                close(pair['approximate_gap'], a)
                close(pair['risk_difference'], risk[j] - risk[w])
                close(pair['test_difference'], test[j] - test[w])
            close(row['selected_risk_difference'], risk[jhat] - risk[w])
            close(row['selected_test_difference'], test[jhat] - test[w])
            close(row['selected_regret'], risk[jhat] - min(risk))
            close(row['reference_regret'], risk[w] - min(risk))
            verified.append((row, risk[jhat] - risk[w], test[jhat] - test[w]))
        assert len(seen) == 512
        for group in result['groups']:
            subset = [v for v in verified if v[0]['case'] == group['case'] and v[0]['method'] == group['method']]
            assert len(subset) == group['n'] == 32
            differences = [v[1] for v in subset]
            assert group['changed'] == sum(r[0]['changed'] for r in subset)
            for label, count in [('harmed', sum(d > EPS for d in differences)),
                                 ('helped', sum(d < -EPS for d in differences)),
                                 ('zero', sum(abs(d) <= EPS for d in differences))]:
                assert group[label] == count
            assert group['exceed_old_individual_epsilon'] == sum(d > decimal(group['old_epsilon']) for d in differences)
            for field in ('absolute_bound_certificate', 'range_bound_certificate', 'reference_near_tie', 'approximate_near_tie'):
                assert group[field] == sum(v[0][field] for v in subset)
            for field, stats in group.items():
                if not isinstance(stats, dict):
                    continue
                values = [decimal(v[0][field]) for v in subset]
                close(stats['min'], min(values)); close(stats['max'], max(values))
                close(stats['median'], statistics.median(values)); close(stats['mean'], sum(values) / 32)
    return dict(passed=True, decisions=512, pairs=1536, groups=16,
                maximum_decimal_discrepancy=max(errors), arithmetic_scalars_checked=len(errors),
                result_sha256=sha(OUT / 'RESULTS.json'), source_sha256=sha(Path(__file__)),
                scope='Independent Decimal differences, selection/ties, certificates, risks and group summaries; no resampling or new statistical inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'check'))
    command = parser.parse_args().command
    output = audit()
    if command == 'run':
        with (OUT / 'ARITHMETIC_AUDIT.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(output, f, ensure_ascii=False, indent=2); f.write('\n')
    else:
        assert output == read(OUT / 'ARITHMETIC_AUDIT.json')
    print(json.dumps(output))
