"""Post-hoc deterministic margin/error/risk analysis of all saved USE-v2/v3 decisions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/use_margin_v4_20261005'
DOC = ROOT / 'docs/08_research_研究工作/USE_v4_margin_20261005'
TOL = 1e-12
METHODS = ['RAW100', 'OUT16', 'INNER1600', 'JVI100']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rel(p):
    return p.relative_to(ROOT).as_posix()


def select(values):
    floor = min(values)
    return next(i for i, v in enumerate(values) if v <= floor + TOL)


def measures(reference, approximate, risk, test):
    w, selected = select(reference), select(approximate)
    rivals = [j for j in range(len(reference)) if j != w]
    errors = [h - l for h, l in zip(approximate, reference)]
    gaps = [reference[j] - reference[w] for j in rivals]
    relative_errors = [errors[j] - errors[w] for j in rivals]
    approximate_gaps = [approximate[j] - approximate[w] for j in rivals]
    delta, margin = min(gaps), min(approximate_gaps)
    amplitude = max(abs(e) for e in errors)
    spread = max(errors) - min(errors)
    joint_spread = max(errors[1:]) - min(errors[1:])
    runner = min(rivals, key=lambda j: (reference[j], j))
    strongest = min(rivals, key=lambda j: (approximate[j], j))
    pairs = [dict(candidate=j, reference_gap=g, relative_error=d, approximate_gap=a,
                  risk_difference=risk[j] - risk[w], test_difference=test[j] - test[w])
             for j, g, d, a in zip(rivals, gaps, relative_errors, approximate_gaps)]
    return dict(reference_winner=w, selected=selected, changed=selected != w,
                reference_runner_up=runner, approximate_strongest_rival=strongest,
                reference_scores=reference, approximate_scores=approximate,
                candidate_risk=risk, candidate_test_NLL=test, realized_errors=errors,
                reference_min_gap=delta, approximate_min_gap=margin,
                max_absolute_error=amplitude, error_range=spread,
                max_winner_relative_error=max(abs(d) for d in relative_errors),
                joint_error_range=joint_spread, joint_centered_max_error=joint_spread / 2,
                absolute_bound_certificate=delta > 2 * amplitude + TOL,
                range_bound_certificate=delta > spread + TOL,
                reference_near_tie=delta <= TOL, approximate_near_tie=abs(margin) <= TOL,
                strict_crossings=sum(a < -TOL for a in approximate_gaps), pairs=pairs,
                selected_risk_difference=risk[selected] - risk[w],
                selected_test_difference=test[selected] - test[w],
                selected_regret=risk[selected] - min(risk),
                reference_regret=risk[w] - min(risk),
                runner_reference_gap=reference[runner] - reference[w],
                runner_relative_error=errors[runner] - errors[w],
                runner_approximate_gap=approximate[runner] - approximate[w])


def qualification():
    examples = [
        ('common_offset', [0., .2, .4, .6], [1., 1.2, 1.4, 1.6], 0),
        ('small_error', [0., 1., 2., 3.], [.01, 1.02, 2.03, 3.04], 0),
        ('reversal', [0., .2, .4, .6], [.3, .2, .4, .6], 1),
        ('near_tie_first', [1., 0., 2., 3.], [1e-13, 0., 2., 3.], 0),
        ('sufficient_not_necessary', [0., .2, .4, .6], [0., 2.2, 3.4, 4.6], 0),
        ('better_risk_after_switch', [0., .2, .4, .6], [.3, .2, .4, .6], 1),
    ]
    for name, ref, approx, expected in examples:
        m = measures(ref, approx, [1., 0., 2., 3.], [1., 0., 2., 3.])
        assert m['selected'] == expected, name
        for p in m['pairs']:
            assert abs(p['reference_gap'] + p['relative_error'] - p['approximate_gap']) < 2e-15
        assert not (m['absolute_bound_certificate'] or m['range_bound_certificate']) or not m['changed']
        if name == 'common_offset':
            assert m['range_bound_certificate'] and not m['absolute_bound_certificate']
        if name == 'sufficient_not_necessary':
            assert not m['range_bound_certificate'] and not m['changed']
        if name == 'better_risk_after_switch':
            assert m['selected_risk_difference'] < 0
    return dict(passed=True, manufactured_cases=len(examples), new_scientific_observations=0)


def stream_paths():
    for case in ('NULL', 'WEAK', 'STRONG'):
        for rep in range(32):
            yield 'USE-v2', case, rep, ROOT / 'artifacts/use_selection_v2_20261005' / case / f'rep_{rep:02}'
    for rep in range(32):
        yield 'USE-v3', 'Optdigits', rep, ROOT / 'artifacts/use_digits_v3_20261005' / f'rep_{rep:02}'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'FREEZE.json').exists()
    assert sha(ROOT / 'HANDOFF.md') == '35935d5bc6304be536efee93a668ae507ca4926d0e5e85b9c6d694ce32c2f616'
    (OUT / 'HANDOFF_INPUT.md').write_bytes((ROOT / 'HANDOFF.md').read_bytes())
    sources = [Path(__file__), ROOT / 'script/usemargin_audit.py', DOC / 'CONTRACT.md', DOC / 'HANDOFF_DISPOSITION.md']
    inputs = [OUT / 'HANDOFF_INPUT.md', ROOT / 'artifacts/research_advance_20261005/DELIVERY_MANIFEST.json']
    for _, _, _, folder in stream_paths():
        inputs += [folder / 'SELECTION.json', folder / 'RESULT.json']
    save(OUT / 'QUALIFICATION.json', qualification())
    save(OUT / 'FREEZE.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline='d063bfb78ffd749414ac6d80aa55515373fb76db', post_hoc_existing_outputs=True,
        sources={rel(p): sha(p) for p in sources}, inputs={rel(p): sha(p) for p in inputs},
        qualification_sha256=sha(OUT / 'QUALIFICATION.json'), streams=128, decisions=512,
        pairwise_comparisons=1536, tie_tolerance=TOL, new_training=0, new_prior=0, new_quadrature=0))
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / 'FREEZE.json'))))


def validate():
    freeze_record = read(OUT / 'FREEZE.json')
    for group in ('sources', 'inputs'):
        for name, digest in freeze_record[group].items():
            assert sha(ROOT / name) == digest, name
    assert sha(OUT / 'QUALIFICATION.json') == freeze_record['qualification_sha256']
    assert qualification() == read(OUT / 'QUALIFICATION.json')


def describe(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values),
                mean=math.fsum(values) / len(values))


def analyze():
    rows = []
    for study, case, rep, folder in stream_paths():
        selection, result = read(folder / 'SELECTION.json'), read(folder / 'RESULT.json')
        assert result['selection_sha256'] == sha(folder / 'SELECTION.json')
        scores = selection['score_matrix' if study == 'USE-v2' else 'scores']
        names = selection['candidate_names' if study == 'USE-v2' else 'candidates']
        assert selection['methods'][:4] == METHODS
        assert select(scores[4]) == selection['choices'][4] == result['choices'][4]
        risk = result['candidate_risks' if study == 'USE-v2' else 'candidate_test_NLL']
        test = result['candidate_test_NLL']
        for i, method in enumerate(METHODS):
            m = measures(scores[4], scores[i], risk, test)
            assert m['selected'] == selection['choices'][i] == result['choices'][i]
            assert not (m['absolute_bound_certificate'] or m['range_bound_certificate']) or not m['changed']
            if not m['approximate_near_tie']:
                assert m['changed'] == (m['approximate_min_gap'] < 0)
            rows.append(dict(study=study, case=case, rep=rep, method=method, candidates=names,
                             source_selection=rel(folder / 'SELECTION.json'), source_result=rel(folder / 'RESULT.json'),
                             risk_kind='teacher_KL' if study == 'USE-v2' else 'official_test_NLL', **m))
    groups = []
    for case in ('NULL', 'WEAK', 'STRONG', 'Optdigits'):
        for method in METHODS:
            subset = [r for r in rows if r['case'] == case and r['method'] == method]
            assert len(subset) == 32
            differences = [r['selected_risk_difference'] for r in subset]
            epsilon = .05 if case == 'Optdigits' else .0002
            group = dict(case=case, method=method, n=32, changed=sum(r['changed'] for r in subset),
                         harmed=sum(x > TOL for x in differences), helped=sum(x < -TOL for x in differences),
                         zero=sum(abs(x) <= TOL for x in differences),
                         exceed_old_individual_epsilon=sum(x > epsilon for x in differences),
                         old_epsilon=epsilon, risk_kind=subset[0]['risk_kind'])
            for key in ('absolute_bound_certificate', 'range_bound_certificate', 'reference_near_tie', 'approximate_near_tie'):
                group[key] = sum(r[key] for r in subset)
            for key in ('reference_min_gap', 'approximate_min_gap', 'max_absolute_error', 'error_range',
                        'joint_error_range', 'runner_relative_error', 'runner_approximate_gap',
                        'selected_risk_difference', 'selected_test_difference', 'selected_regret'):
                group[key] = describe([r[key] for r in subset])
            groups.append(group)
    assert len(rows) == 512
    return dict(records=rows, groups=groups, streams=128, decisions=len(rows), rival_pairs=1536,
                scope='Post-hoc deterministic analysis of realized single-pool errors; no expectation-bias identification or new statistical family')


def run():
    validate()
    start = time.perf_counter()
    result = analyze()
    result.update(freeze_sha256=sha(OUT / 'FREEZE.json'), seconds=time.perf_counter() - start,
                  accounting=dict(new_observations=0, new_training=0, new_prior_vectors=0, new_quadrature_nodes=0))
    save(OUT / 'RESULTS.json', result)
    print(json.dumps(dict(completed=True, streams=128, decisions=512, seconds=result['seconds'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'run', 'check'))
    command = p.parse_args().command
    if command == 'check':
        validate()
        result = read(OUT / 'RESULTS.json')
        for key, value in analyze().items():
            assert result[key] == value, key
        assert result['freeze_sha256'] == sha(OUT / 'FREEZE.json')
        print(json.dumps(dict(passed=True, decisions=512)))
    else:
        globals()[command]()
