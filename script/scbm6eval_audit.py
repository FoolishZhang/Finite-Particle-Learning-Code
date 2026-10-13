"""Independent frozen-six-target run audit; no producer score/statistic calls.

check: deterministic arrays/old-input provenance and producer-statistic comparison.
run: authorized full input replay, restricted score reconstruction, saved
statistics, and exactly the prespecified independent reference quadratures.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import t as student

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/scbm_six_frozen_targets'
OUT = BASE / 'AUDIT'
DOC = ROOT / 'docs/08_research_研究工作/六相关冻结输出与有限目标_20261003'
PAIRS = list(itertools.combinations(range(4), 2))
BITS = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.int64)
LOSS = ROOT / 'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
SCOPE = dict(groups=32, banks_per_group=1024, K=100, chunk=64,
    replay_Gaussian_scalars=851968000, prefix_banks_per_group=4,
    prefix_score_cells=143360, independent_native_API_calls=192,
    independent_V_candidates=[0, 1, 2, 17, 34], V_GL=40, V_GH=48,
    independent_V_bivariate_nodes=2764800, independent_mean_PMF_order=36,
    independent_mean_PMF_nodes=1679616, new_independent_random_draws=0,
    score_absolute_tolerance=2e-10, stored_arithmetic_absolute_tolerance=2e-10,
    native_replay_absolute_tolerance=2e-6, V_absolute_tolerance=1e-8,
    PMF_absolute_tolerance=5e-8, no_automatic_retry=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, data, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def close(a, b, tol=2e-10):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    error = float(np.max(np.abs(x-y))) if x.size else 0.
    assert error <= tol, (error, tol)
    return error


def verify_freeze():
    f = read(BASE / 'FREEZE.json')
    assert f['run_authorized'] is True
    for path, digest in f['inputs'].items():
        p = (ROOT/path).resolve()
        assert p.is_relative_to(ROOT) and sha(p) == digest, path
    for p in [Path(__file__), BASE/'CONFIG.json', BASE/'CANDIDATES.json', DOC/'PLAN.md', LOSS]:
        assert f['inputs'][rel(p)] == sha(p), p
    assert not (BASE/'VERSION_LOCK.json').exists()
    return f


def candidate_check():
    cfg, c = read(BASE/'CONFIG.json'), read(BASE/'CANDIDATES.json')
    old = read(ROOT/'artifacts/scbm_multicorrelation/TRAIN/RESULTS.json')
    ref = read(ROOT/'artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json')
    truth = read(ROOT/'artifacts/scbm_invariance/REFERENCE/MODEL.json')
    for path, digest in c['inputs'].items():
        assert sha(ROOT/path) == digest, path
    assert c['patterns'] == BITS.tolist() and c['pair_order'] == [list(p) for p in PAIRS]
    assert c['mean'] == c['mu'] == [-.6, -.2, .2, .6]
    assert [r['stream'] for r in old['records']] == list(range(16))
    rows = c['candidates']
    assert [r['index'] for r in rows] == list(range(35))
    names = ['truth', 'high'] + [f'{method}_TAIL_{s:02}' for method in ['FIXED', 'FACT'] for s in range(16)] + ['MEAN_FIXED']
    assert [r['name'] for r in rows] == names
    oldref = {tuple(r['rho']): r for r in ref['records']}
    for j, row in enumerate(rows):
        C = np.eye(4)
        for v, (u, w) in zip(row['rho'], PAIRS):
            C[u, w] = C[w, u] = v
        assert np.array_equal(C, row['covariance'])
        close(np.linalg.eigvalsh(C)[0], row['minimum_eigenvalue'], 2e-13)
        assert np.linalg.eigvalsh(C)[0] >= .09999
        if j < 2:
            source = next(r for r in truth['records'] if r['t'] == [0., .75][j])
            assert row['reused_pmf'] == source['main_pmf']
            close(row['rho'], (1+[0., .75][j])*np.array([.25, -.15, .1, .1, -.15, .25]), 0.)
        elif j < 34:
            method, s = ('FIXED', j-2) if j < 18 else ('FACT', j-18)
            assert row['stream'] == s and row['rho'] == old['records'][s]['outputs'][method]['TAIL']['rho']
            source = oldref[tuple(row['rho'])]
            assert source['passed'] and row['reused_pmf'] == source['main_pmf']
        else:
            mean = [math.fsum(r['outputs']['FIXED']['TAIL']['rho'][k] for r in old['records'])/16 for k in range(6)]
            close(row['rho'], mean, 2e-15)
            assert row['reused_pmf'] is None
    assert c['population_weights'] == rows[0]['reused_pmf']
    assert abs(math.fsum(c['population_weights'])-1) <= 2e-12
    for s, row in enumerate(c['data_weights']):
        assert row['stream'] == s
        source = old['records'][s]
        assert sha(ROOT/source['input_file']) == source['input_sha256']
        with np.load(ROOT/source['input_file'], allow_pickle=False) as f:
            labels = f['active_labels'].astype(np.int64)
        assert labels.shape == (4096, 4)
        counts = [int(np.all(labels == pattern, axis=1).sum()) for pattern in BITS]
        assert counts == row['counts'] and sum(counts) == 4096
        assert row['weights'] == [x/4096 for x in counts]
    return cfg, c, dict(candidate_count=35, old_PMF_mappings=34, old_data_sets=16,
                        label_rows=65536, new_RNG=0, new_model_integrals=0)


def independent_scores(active, noise, c):
    """Signed-label log-sigmoid products; no producer's x@patterns formula."""
    B, K, _ = active.shape
    out = np.empty((B, 2, 35, 16))
    lognoise = np.log(expit(-noise)) if np.max(abs(noise)) < 40 else -np.logaddexp(0., noise)
    lognoise = lognoise.sum(axis=2)
    for j, row in enumerate(c['candidates']):
        x = active @ np.linalg.cholesky(np.array(row['covariance'])).T + np.array(c['mu'])
        for y, bits in enumerate(BITS):
            a = -np.logaddexp(0., -(2*bits-1)*x).sum(axis=2)
            out[:, 0, j, y] = math.log(K)-logsumexp(a+lognoise, axis=1)
            out[:, 1, j, y] = noise.shape[2]*math.log(2.)+math.log(K)-logsumexp(a, axis=1)
    return out


def explicit_contrasts(means, c):
    """Explicit group/model/pattern loops avoid ambiguous 16x16 broadcasting."""
    p = c['population_weights']; d = [r['weights'] for r in c['data_weights']]
    a = np.asarray(means)
    result = np.empty((11, len(a)))
    for group, values in enumerate(a):
        k = 0
        for profile in range(2):
            for empirical in [False, True]:
                for comparator in ['truth', 'FACT']:
                    per_model = []
                    for s in range(16):
                        other = 0 if comparator == 'truth' else 18+s
                        weights = d[s] if empirical else p
                        per_model.append(math.fsum(weights[y]*(values[profile, 2+s, y]-values[profile, other, y]) for y in range(16)))
                    result[k, group] = math.fsum(per_model)/16
                    k += 1
        result[8, group] = math.fsum(p[y]*(values[0, 1, y]-values[0, 0, y]) for y in range(16))
        for profile in range(2):
            result[9+profile, group] = math.fsum(p[y]*(math.fsum(values[profile, 2+s, y] for s in range(16))/16-values[profile, 34, y]) for y in range(16))
    return result


def estimate(values):
    v = [float(x) for x in values]
    m = math.fsum(v)/len(v)
    se = math.sqrt(math.fsum((x-m)**2 for x in v)/(len(v)*(len(v)-1)))
    return m, se


def native_audit(active, noise, scores, records, c, cfg):
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('_scbm6eval_independent_official', LOSS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loss = module.SCBLoss(num_classes=2, alpha=1., config=SimpleNamespace(
        reg_precision='l1', reg_weight=1., num_monte_carlo=100, training_mode='joint'))
    largest = 0.; cells = 0
    for ci in cfg['native_control_candidate_indices']:
        for profile, name in enumerate(['raw', 'FACT']):
            record = next(x for x in records['checks'] if x['candidate_index'] == ci and x['profile'] == name)
            eps = torch.tensor(active, dtype=torch.float32)
            C = torch.tensor(c['candidates'][ci]['covariance'], dtype=torch.float32)
            x = eps @ torch.linalg.cholesky(C).T + torch.tensor(c['mu'], dtype=torch.float32)
            logits = torch.cat([x, torch.tensor(noise, dtype=torch.float32)], dim=2) if profile == 0 else x
            probability = torch.sigmoid(logits.permute(0, 2, 1))
            actual = []
            with torch.no_grad():
                for bits in BITS:
                    y = np.tile(bits, (4, 1))
                    if profile == 0:
                        y = np.column_stack([y, np.zeros((4, 256), dtype=np.int64)])
                    value = loss.compute_concept_loss(probability, torch.tensor(y, dtype=torch.int64))
                    if profile == 1:
                        value = value + 256*math.log(2.)
                    actual.append(float(value))
            largest = max(largest, close(actual, record['native_total_score'], SCOPE['native_replay_absolute_tolerance']))
            mean = scores[:, profile, ci, :].mean(axis=0)
            close(mean, record['mathematical_total_score'])
            delta = np.array(actual)-mean
            close(delta, record['difference'], SCOPE['native_replay_absolute_tolerance'])
            assert record['passed'] == (np.max(abs(np.array(record['difference']))) <= cfg['numerical_tolerance'])
            cells += 16
    assert cells == 192
    return dict(calls=cells, maximum_native_replay_difference=largest,
                production_passed=records['passed'], independent_new_random_draws=0)


def mc_audit(cfg, c, mc):
    records = mc['groups']
    assert [x['group'] for x in records] == list(range(32))
    assert mc['numerical_qualification'] == records[0]['numerical_controls']
    assert mc['accounting']['Gaussian_scalars'] == 851968000
    maxscore = 0.; maxmeans = 0.; native = None
    for group, row in enumerate(records):
        p = ROOT/row['prefix_npz']
        assert sha(p) == row['prefix_sha256']
        with np.load(p, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        assert set(arrays) == {'active_eps', 'noise_eps', 'individual_total_score', 'block_means'}
        for name, value in arrays.items():
            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == row['prefix_array_sha256'][name]
        for role, dim in [('active', 4), ('noise', 256)]:
            seed = [cfg['seed_base'], group, cfg['roles'][role]]
            assert row['role_seed_sequences'][role] == seed
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            assert rng.bit_generator.state == row['RNG_initial_states'][role]
            digest = hashlib.sha256()
            for chunk in range(16):
                values = rng.standard_normal((64, 100, dim), dtype=np.float64)
                digest.update(values.tobytes(order='C'))
                if chunk == 0:
                    assert np.array_equal(values[:4], arrays[role+'_eps'])
            assert digest.hexdigest() == row['role_input_sha256'][role]
            assert rng.bit_generator.state == row['RNG_final_states'][role]
        assert row['completed_blocks'] == 16 and row['consumed_banks'] == 1024
        values = independent_scores(arrays['active_eps'], arrays['noise_eps'], c)
        maxscore = max(maxscore, close(values, arrays['individual_total_score'], SCOPE['score_absolute_tolerance']))
        assert arrays['block_means'].shape == (16, 2, 35, 16)
        means = np.array([math.fsum(x)/16 for x in arrays['block_means'].reshape(16, -1).T]).reshape(2, 35, 16)
        maxmeans = max(maxmeans, close(means, row['means']))
        if group == 0:
            native = native_audit(arrays['active_eps'], arrays['noise_eps'], values, row['numerical_controls'], c, cfg)
        else:
            assert row['numerical_controls'] is None
        save(OUT/f'group_{group:02}.json', dict(passed=True, group=group,
            prefix_input_sha256=sha(p), replay_roles=2, score_cells=4*2*35*16,
            maximum_score_error=float(np.max(abs(values-arrays['individual_total_score'])))))
        print(json.dumps(dict(audited_group=group)), flush=True)
    return (dict(groups=32, roles=64, Gaussian_scalars=851968000, all_hashes_and_states_match=True,
                 new_independent_draws=0),
            dict(cases=143360, maximum_score_error=maxscore, maximum_block_mean_error=maxmeans,
                 scope='First4 banks/group all35 candidates,2 profiles,16patterns; all saved block arithmetic.'), native)


def statistics_audit(cfg, c, mc, ref, result):
    means = np.array([r['means'] for r in mc['groups']])
    vectors = explicit_contrasts(means, c)
    critical = float(student.isf(.05/(2*11), 31))
    close(critical, result['critical_t'], 2e-12)
    pmf = np.array([r['main_pmf'] for r in ref['records']])
    exact = -np.log(pmf)
    integral_vectors = explicit_contrasts(np.tile(exact, (1, 2, 1, 1)), c)[:, 0]
    effects = precision = combined = 0
    for j, row in enumerate(result['primary_family']):
        assert row['name'] == cfg['primary_order'][j]
        assert row['expected_sign'] == cfg['primary_expected_sign'][j]
        close(vectors[j], row['group_values'])
        m, se = estimate(vectors[j]); lo, hi = m-critical*se, m+critical*se
        for key, value in [('mean', m), ('SE', se), ('lower', lo), ('upper', hi),
                           ('integrated_loglikelihood_contrast', integral_vectors[j]),
                           ('MC_distortion_contrast', m-integral_vectors[j])]:
            close(value, row[key])
        close([lo-integral_vectors[j], hi-integral_vectors[j]], row['MC_distortion_interval'])
        ep = hi < 0 if row['expected_sign'] == 'negative' else lo > 0
        pp = se <= .0002
        assert row['effect_passed'] == ep and row['precision_passed'] == pp
        assert row['resolved_sign'] == ('negative' if hi < 0 else 'positive' if lo > 0 else 'unresolved')
        effects += ep; precision += pp; combined += ep and pp
    p = np.array(c['population_weights']); d = np.array([r['weights'] for r in c['data_weights']])
    V = [r['V_main'] for r in ref['records']]
    kl = [math.fsum(float(w)*math.log(float(w)/float(q)) for w, q in zip(p, r)) for r in pmf]
    for j, row in enumerate(result['candidates']):
        assert row['index'] == j and row['rho'] == c['candidates'][j]['rho']
        close(row['reference_joint_KL'], kl[j]); close(row['V'], V[j])
        close(row['integrated_P'], math.fsum(p[y]*exact[j, y] for y in range(16)))
        close(row['integrated_D_by_stream'], [math.fsum(d[s, y]*exact[j, y] for y in range(16)) for s in range(16)])
        for profile in range(2):
            z = [math.fsum(p[y]*a[profile, j, y] for y in range(16)) for a in means]
            m, se = estimate(z)
            close(row['mean_score_P'][profile], m); close(row['SE_score_P'][profile], se)
    assert len(result['per_model_descriptive']) == 128
    for row in result['per_model_descriptive']:
        s = row['stream']; profile = cfg['profiles'].index(row['profile'])
        other = 0 if row['comparison'] == 'FIXED_minus_truth' else 18+s
        w = p if row['weights'] == 'P' else d[s]
        z = [math.fsum(w[y]*(a[profile, 2+s, y]-a[profile, other, y]) for y in range(16)) for a in means]
        m, se = estimate(z); close(m, row['mean']); close(se, row['SE'])
        close(row['integrated_difference'], math.fsum(w[y]*(exact[2+s, y]-exact[other, y]) for y in range(16)))
        close(row['V_population_reference_difference'], V[2+s]-V[other])
    expectedV = [math.fsum(V[2:18])/16-V[0], math.fsum(V[2+s]-V[18+s] for s in range(16))/16,
                 V[1]-V[0], math.fsum(V[2:18])/16-V[34]]
    close(list(result['V_panel_reference'].values()), expectedV)
    close(list(result['mean_fixed_risk_comparison'].values()), [math.fsum(kl[2:18])/16, kl[34], math.fsum(kl[2:18])/16-kl[34]])
    assert result['primary_effects_passed'] == effects and result['primary_precision_passed'] == precision and result['primary_combined_passed'] == combined
    assert result['reference_qualification'] == dict(PMF=ref['PMF_qualification_passed'], V=ref['V_qualification_passed'], all=ref['qualification_passed'])
    assert result['native_numerical_qualification'] == mc['numerical_qualification']
    return dict(passed=True, primary_contrasts=11, group_values=352, candidates=35,
        per_model_descriptive=128, effects=int(effects), precision=int(precision), combined=int(combined),
        qualification_failures_not_overridden=True, new_tests=0)


def center(c):
    p = c['population_weights']; spins = 2*BITS-1
    b = np.array([math.fsum(float(w)*int(y[i]) for w, y in zip(p, spins)) for i in range(4)])
    k = np.array([math.fsum(float(w)*int(y[i]*y[j]) for w, y in zip(p, spins))-b[i]*b[j] for i, j in PAIRS])
    return b, k


def reference_arithmetic(c, ref):
    b, k = center(c); close(ref['V_center']['spin_means'], b, 2e-15); close(ref['V_center']['true_spin_covariances'], k, 2e-15)
    mean_reference = read(BASE/'REFERENCE/MEAN_FIXED_PMF.json')
    mean_main = next(v['pmf'] for v in mean_reference['variants'] if v['order'] == 32)
    assert mean_main == mean_reference['main_pmf'] == ref['records'][34]['main_pmf']
    pmf_flags = []; new_pmf_deltas = []
    for v, q in zip(mean_reference['variants'], mean_reference['qualifications']):
        pmf = v['pmf']; delta = max(abs(a-b) for a,b in zip(pmf, mean_main))
        norm = abs(math.fsum(pmf)-1); positive = len(pmf) == 16 and all(math.isfinite(x) and x > 0 for x in pmf)
        close(delta, q['maximum_change']); close(norm, q['normalization_error'])
        assert q['positive'] == positive and q['passed'] == (positive and delta <= 5e-8 and norm <= 2e-12)
        pmf_flags.append(q['passed']);new_pmf_deltas.append(delta)
    assert mean_reference['passed'] == all(pmf_flags)
    maxerror = 0.; all_pmf = []; all_V = []; V_deltas = []
    for index, row in enumerate(ref['records']):
        assert row['index'] == index and row['rho'] == c['candidates'][index]['rho']
        if index < 34:
            assert row['main_pmf'] == c['candidates'][index]['reused_pmf']
            assert row['PMF_reused'] is True
            q = row['qualifications']['PMF'][0]
            norm = abs(math.fsum(row['main_pmf'])-1)
            positive = all(math.isfinite(x) and x > 0 for x in row['main_pmf'])
            close(norm,q['normalization_error']);assert q['positive']==positive and q['passed']==(positive and norm<=2e-12)
            assert sha(ROOT/q['source_path']) == q['source_sha256']
        else:
            assert row['PMF_reused'] is False and row['qualifications']['PMF'] == mean_reference['qualifications']
        p = row['main_pmf']; truth = c['population_weights']
        close(row['reference_joint_KL'], math.fsum(w*math.log(w/q) for w,q in zip(truth,p)))
        close(row['reference_cross_entropy_P'], -math.fsum(w*math.log(q) for w,q in zip(truth,p)))
        close(row['reference_cross_entropy_D'], [-math.fsum(w*math.log(q) for w,q in zip(d['weights'],p)) for d in c['data_weights']])
        for variant in row['V_variants']:
            integrals = []
            for pair, coefficient in zip(variant['pairs'], row['rho']):
                i, j = pair['pair']; assert pair['correlation'] == coefficient
                vals = np.array(pair['inner_spin_pair_moments'])-b[i]*b[j]
                close(vals, pair['inner_centered_covariances'])
                integral = coefficient*math.fsum(w*x for w,x in zip(pair['outer_weights_on_minus1_plus1'], vals))/2
                close(integral, pair['integral_0_to_c'])
                integrals.append(integral)
            value = -math.fsum(kk*cc+ii for kk,cc,ii in zip(k,row['rho'],integrals))/4
            maxerror = max(maxerror, close(value, variant['V']))
            if [variant['GL_order'], variant['GH_order']] == [24,32]:close(value,row['V_main'])
        for variant, q in zip(row['V_variants'], row['qualifications']['V']):
            delta = abs(variant['V']-row['V_main']);close(delta,q['maximum_change'])
            assert q['passed'] == (delta <= 1e-8)
            V_deltas.append(delta)
        flags=[q['passed'] for q in row['qualifications']['PMF']+row['qualifications']['V']]
        assert row['passed'] == all(flags)
        all_pmf.extend(q['passed'] for q in row['qualifications']['PMF'])
        all_V.extend(q['passed'] for q in row['qualifications']['V'])
    assert ref['PMF_qualification_passed'] == all(all_pmf) and ref['V_qualification_passed'] == all(all_V)
    assert ref['qualification_passed'] == all(all_pmf+all_V)
    close(ref['maximum_V_change'],max(V_deltas));close(ref['maximum_new_PMF_change'],max(new_pmf_deltas))
    for key,value in dict(new_PMF_integrations=3,new_four_dimensional_Gaussian_nodes=3940352,
        V_pair_rule_integrals=630,inner_pair_GH_expectations=15120,new_bivariate_Gaussian_nodes=17848320).items():
        assert ref['accounting'][key] == value
    return dict(candidates=35, V_rule_values=105, V_pair_integrals=630,
                maximum_saved_arithmetic_error=maxerror)


def independent_V(row, c):
    # Independent polynomial quadrature generators and symmetric roots.
    z, wz = np.polynomial.hermite.hermgauss(48)
    z = np.sqrt(2.)*z; wz = wz/math.sqrt(math.pi)
    gx, gw = np.polynomial.legendre.leggauss(40)
    eps = np.array(list(itertools.product(z, repeat=2)))
    weights = np.outer(wz,wz).reshape(-1)
    b, k = center(c); integrals = []
    for index, (i,j) in enumerate(PAIRS):
        correlation = row['rho'][index]; values = []
        for node in gx:
            u = correlation*(node+1)/2
            C = np.array([[1.,u],[u,1.]])
            eig, U = np.linalg.eigh(C)
            S = (U*np.sqrt(eig))@U.T
            x = eps@S.T + np.array([c['mu'][i],c['mu'][j]])
            values.append(float(np.dot(weights,np.tanh(x[:,0]/2)*np.tanh(x[:,1]/2)))-b[i]*b[j])
        integrals.append(correlation*math.fsum(float(w)*v for w,v in zip(gw,values))/2)
    return -math.fsum(float(kj)*cc+ii for kj,cc,ii in zip(k,row['rho'],integrals))/4


def independent_mean_pmf(c):
    q = 36; z, w = np.polynomial.hermite.hermgauss(q)
    z = np.sqrt(2.)*z; w = w/math.sqrt(math.pi)
    C = np.array(c['candidates'][34]['covariance']);eig,U=np.linalg.eigh(C);S=(U*np.sqrt(eig))@U.T
    total = np.zeros(16, dtype=np.longdouble)
    for start in range(0, q**4, 65536):
        indices = np.arange(start,min(start+65536,q**4))
        ids = np.column_stack([(indices//q**power)%q for power in [3,2,1,0]])
        x = z[ids]@S.T + np.array(c['mu']); weights = w[ids].prod(axis=1)
        prob = expit(x)
        for yi,bits in enumerate(BITS):
            likelihood = np.where(bits,prob,1-prob).prod(axis=1)
            total[yi] += np.sum(weights*likelihood,dtype=np.longdouble)
    return total.astype(float)


def independent_reference(c, ref):
    values = []
    for index in [0,1,2,17,34]:
        save(OUT/'REFERENCE_PROGRESS.json',dict(in_flight=dict(kind='V',index=index),completed=len(values)),False)
        value = independent_V(c['candidates'][index],c)
        error = abs(value-ref['records'][index]['V_main'])
        values.append(dict(index=index,value=value,production=ref['records'][index]['V_main'],difference=error,passed=error<=1e-8))
        save(OUT/f'V_{index:02}.json',values[-1])
    save(OUT/'REFERENCE_PROGRESS.json',dict(in_flight=dict(kind='meanPMF',order=36),completed_V=len(values)),False)
    pmf = independent_mean_pmf(c)
    error = float(np.max(abs(pmf-np.array(ref['records'][34]['main_pmf']))))
    result = dict(passed=all(v['passed'] for v in values) and error<=5e-8,
        V_records=values,mean_PMF=pmf.tolist(),mean_PMF_maximum_difference=error,
        mean_PMF_normalization_error=abs(math.fsum(pmf)-1),
        new_V_candidate_integrals=5,new_V_pair_rule_integrals=30,new_bivariate_nodes=2764800,
        new_PMF_integrals=1,new_PMF_pattern_integrals=16,new_four_dimensional_nodes=1679616,
        roots='symmetric covariance roots; NumPy hermgauss/leggauss, not production routines',
        no_extra_rules_or_retries=True)
    save(OUT/'REFERENCE_INDEPENDENT.json',result)
    return result


def check():
    assert not (BASE/'FREEZE.json').exists()
    cfg,c,prep=candidate_check()
    # Deliberately nonexchangeable data weights test stream/pattern axes.
    weights=dict(population_weights=[(j+1)/136 for j in range(16)],
                 data_weights=[dict(weights=[float(y==(3*s+1)%16) for y in range(16)]) for s in range(16)])
    a=np.empty((32,2,35,16))
    for g,h,j,y in itertools.product(range(32),range(2),range(35),range(16)):
        a[g,h,j,y]=(j+1)*(y+2)/17+(h+1)*(j-2)**2/113+g*(j*y+1)/211
    spec=importlib.util.spec_from_file_location('_scbm6eval_stats_readonly',ROOT/'script/scbm6eval_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    error=close(explicit_contrasts(a,weights),module.contrast_vectors(a,weights),2e-12)
    active=np.sin(np.arange(2*7*4)*.071).reshape(2,7,4)
    zero=np.zeros((2,7,256));out=independent_scores(active,zero,c)
    assert np.max(abs(out[:,0]-out[:,1]))<1e-12
    record=dict(passed=True,source_sha256=sha(__file__),scope=SCOPE,
        preparation=prep,asymmetric_axis_check_error=error,new_GH_or_GL_nodes=0,
        new_scientific_RNG=0,new_model_integrals=0,new_training=0,
        inputs={rel(p):sha(p) for p in [BASE/'CONFIG.json',BASE/'CANDIDATES.json',DOC/'PLAN.md',ROOT/'script/scbm6eval_results.py']})
    save(OUT/'SELFTEST.json',record)
    print(json.dumps(record),flush=True)


def run():
    verify_freeze();cfg,c,prep=candidate_check();test=read(OUT/'SELFTEST.json')
    assert test['passed'] and test['source_sha256']==sha(__file__)
    assert not (OUT/'EXECUTION.json').exists() and not (OUT/'RESULTS.json').exists()
    inputs={rel(p):sha(p) for p in [BASE/'FREEZE.json',BASE/'MC/RESULTS.json',BASE/'REFERENCE/RESULTS.json',BASE/'RESULTS.json',BASE/'CONFIG.json',BASE/'CANDIDATES.json']}
    mc,ref,result=[read(BASE/p) for p in ['MC/RESULTS.json','REFERENCE/RESULTS.json','RESULTS.json']]
    for row in mc['groups']:inputs[row['prefix_npz']]=row['prefix_sha256']
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),input_files=inputs,scope=SCOPE,automatic_retry=False))
    start=time.perf_counter()
    try:
        replay,prefix,native=mc_audit(cfg,c,mc)
        statistics=statistics_audit(cfg,c,mc,ref,result)
        arithmetic=reference_arithmetic(c,ref)
        independent=independent_reference(c,ref)
        verify_freeze()
        for p,h in inputs.items():assert sha(ROOT/p)==h,p
        outcome=dict(passed=bool(independent['passed']),remaining_required_fixes=[] if independent['passed'] else ['Independent reference tolerance failed; no retry or extra quadrature authorized.'],
            source_sha256=sha(__file__),freeze_sha256=sha(BASE/'FREEZE.json'),input_files=inputs,
            preparation_audit=prep,production_RNG_replay=replay,prefix_score_audit=prefix,
            native_API_replay=native,statistics_audit=statistics,reference_audit=dict(saved_arithmetic=arithmetic,independent=independent),
            seconds=time.perf_counter()-start,new_independent_scientific_draws=0,
            new_model_integrals=dict(V_candidates=5,V_pair_rules=30,PMFs=1,bivariate_nodes=2764800,four_dimensional_nodes=1679616),
            scope='All production inputs replayed, all saved arithmetic/statistics checked; independent bottom-level scores limited to first4 banks/group. No full scientific scoring rerun or training. New audit quadratures are explicitly budgeted and counted.')
        save(OUT/'RESULTS.json',outcome)
        print(json.dumps(dict(passed=outcome['passed'],seconds=outcome['seconds'],statistics=statistics,results_sha256=sha(OUT/'RESULTS.json'))),flush=True)
    except BaseException:
        save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start,
                                   scope=SCOPE,automatic_retry=False))
        raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','run']);args=parser.parse_args()
    check() if args.mode=='check' else run()
