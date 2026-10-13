"""Post-hoc finite-sample sensitivity for the degenerate pair-vs-joint t rows."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scale_learning_v2'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def calculate():
    fits = rows(OUT / 'FINITE_FITS.csv'); curves = rows(OUT / 'REFERENCE_CURVES.csv')
    result = []
    for profile in ['null', 'contrast']:
        chosen = {(int(r['stream']), r['objective']): int(r['selected_index']) for r in fits
                  if r['profile'] == profile and int(r['N']) == 65536 and r['objective'] in ('J', 'P')}
        disagreements = sum(chosen[s, 'J'] != chosen[s, 'P'] for s in range(16))
        assert len(chosen) == 32 and disagreements == 0
        risks = [float(r['true_Q_KL']) for r in curves if r['profile'] == profile]
        # Numeric allowance is a qualified-reference sensitivity, not a rigorous integration certificate.
        risk_range = max(risks) - min(risks) + 2e-7
        upper_probability = -math.expm1(math.log(.025) / 16)
        result.append(dict(profile=profile, N=65536, paired_units=16, observed_different_selections=0,
            alpha_one_sided=.025, disagreement_probability_upper=upper_probability,
            candidate_risk_range_plus_numerical_allowance=risk_range,
            mean_risk_difference_absolute_bound=risk_range * upper_probability,
            strict_001_equivalence_certified=bool(risk_range * upper_probability <= .001)))
    sources = [Path(__file__), OUT / 'FINITE_FITS.csv', OUT / 'REFERENCE_CURVES.csv', OUT / 'RESULTS_FREEZE.json']
    return dict(posthoc_sensitivity_not_original_primary=True, observations_reused_no_new_samples=True,
        original_t_primary_decisions_unchanged=True, conditional_on_qualified_reference=True,
        simultaneous_confidence_for_two_fixed_profile_bounds=.95,
        joint_coverage_with_original_t_families_not_claimed=True,
        method='zero-disagreement exact binomial bound times finite candidate risk range',
        rows=result, sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    path = OUT / 'PAIR_SENSITIVITY.json'; result = calculate()
    if args.check:
        saved = json.loads(path.read_text(encoding='utf-8')); saved.pop('time_utc')
        assert saved == result
    else:
        assert not (OUT / 'MANIFEST.json').exists(), 'already sealed'
        with path.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(dict(time_utc=datetime.now(timezone.utc).isoformat(), **result), f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(result['rows']))
