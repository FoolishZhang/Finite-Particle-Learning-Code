"""Frozen finite-grid scale learning; no writes to prior research packages."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json, math, os, platform, sys, time, traceback
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import scipy
from scipy.special import log_expit
from scipy.stats import t as student_t
import scale_learning_core as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scale_learning_v2'
DOC = ROOT / 'docs/08_research_研究工作/有限粒子尺度学习_v2_20260930'
BASE = '65ec5b5248f66cc6f02d7d93c5e6f418e7978159'
CFG = dict(d=16, c=.9, profiles={'null': [4., 4.], 'contrast': [6., 2.]},
    grid=[j / 20 for j in range(31)], truth_index=20, witness_index=10,
    K=[10, 100, 1000], streams=16, banks=64, bank_batch=16,
    bank_seed=20310101, data_seed=20320101, train_N=[4096, 65536],
    mass_gate=1e-12, identity_gate=1e-10, anchor_L1_gate=1e-8,
    reference_sensitivity_gate=1e-7, population_family=360, fit_family=20,
    population_SE_gates=[.01, .005, .002], fit_SE_gate=.001,
    alpha=.05, material=.001, objectives=['J', 'P', 'G10', 'G100', 'G1000', 'S10', 'S100', 'S1000'])

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def convert(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return x.relative_to(ROOT).as_posix()
    raise TypeError(type(x).__name__)
def write(p, x, exclusive=True):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, default=convert, allow_nan=False); f.write('\n')
def csvwrite(p, rows):
    with p.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def verify(ok, message):
    if not ok: raise RuntimeError(message)
def hashes(paths): return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def freeze():
    verify(not OUT.exists(), 'output directory already exists')
    OUT.mkdir(parents=True); (OUT / 'local').mkdir(); (OUT / 'source_inputs').mkdir()
    (OUT / '.gitignore').write_text('/local/\n', encoding='utf-8', newline='\n')
    (OUT / 'source_inputs/HANDOFF.md').write_bytes((ROOT / 'HANDOFF.md').read_bytes())
    write(OUT / 'CONFIG.json', CFG)
    write(OUT / 'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()))
    inputs = [Path(__file__), ROOT / 'script/scale_learning_core.py', DOC / 'PLAN.md', DOC / 'THEORY.md', DOC / 'NUMERICS.md',
        OUT / 'CONFIG.json', OUT / 'ENVIRONMENT.json', OUT / 'source_inputs/HANDOFF.md',
        ROOT / 'artifacts/structure_tradeoff_v2/MANIFEST.json', ROOT / 'artifacts/paper_v1_final/MANIFEST.json',
        ROOT / 'artifacts/paper_v1_final/VERSION_LOCK.json']
    inputs += [ROOT / f'artifacts/structure_tradeoff_v2/R2/reference/{p}_d16.json' for p in CFG['profiles']]
    write(OUT / 'FREEZE.json', dict(time_utc=now(), baseline_commit=BASE, files=hashes(inputs)))
    write(OUT / 'STATE.json', dict(status='frozen', time_utc=now(), active_jobs=[]))
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / 'FREEZE.json'))), flush=True)

def checkfreeze():
    verify(read(OUT / 'CONFIG.json') == CFG, 'config changed')
    for name, expected in read(OUT / 'FREEZE.json')['files'].items(): verify(sha(ROOT / name) == expected, 'freeze changed: ' + name)

def smoke():
    d = 4; a, b, n, lg, lt, labels, ind = core.geometry(d)
    all_labels = np.array(list(itertools.product([0., 1.], repeat=d)))
    ia = all_labels[:, :2].sum(1).astype(int); ib = all_labels[:, 2:].sum(1).astype(int)
    eta = np.array([[-2.7, -.3, .8, 1.2], [.2, .2, .2, .2]])
    direct = np.exp(log_expit((2 * all_labels[None, :, :] - 1) * eta[:, None, :]).sum(-1))
    joint = core.joint_mass(eta); whole = core.pb(eta)
    errs = [float(np.max(abs(joint @ ind - whole)))]
    for j in range(2): errs.append(float(np.max(abs(np.bincount(3 * ia + ib, weights=direct[j], minlength=9) - joint[j]))))
    z = np.eye(d)
    factor = core.eta_from_z(z, [1., 1.], CFG['c'])
    rho = CFG['c'] / (d - 1)
    errs.append(float(np.max(abs(factor.T @ factor - ((1 + rho) * np.eye(d) - rho * np.ones((d, d)))))))
    # Arithmetic fixture: specified pair tables test the counting kernel only;
    # it is not a new Gaussian reference or a scientific observation.
    fixture = core.reference(4, [0., 0.], CFG['c'])
    fixture['independent_Hermite_moments']['pair_magnitudes_AA_BB_AB'] = [.1, .15, .2]
    pair_details = core.pair_loss_vectors(fixture, return_details=True)
    probabilities = pair_details['pair_probabilities']
    for label in all_labels.astype(int):
        direct_pair = 0.
        for i in range(4):
            for j in range(i + 1, 4):
                pairtype = 0 if j < 2 else (1 if i >= 2 else 2)
                direct_pair -= math.log(probabilities[pairtype, 2 * label[i] + label[j]]) / 6
        cell = 3 * int(label[:2].sum()) + int(label[2:].sum())
        errs.append(abs(direct_pair - pair_details['loss_vector'][cell]))
    verify(max(errs) < 1e-12, 'deterministic probability smoke failed')
    write(OUT / 'SMOKE.json', dict(passed=True, maximum_error=max(errs), enumerated_labels=16, science_samples=0))

def references():
    for profile, scales in CFG['profiles'].items():
        for ti, amplitude in enumerate(CFG['grid']):
            start = time.perf_counter()
            ref = core.reference(CFG['d'], np.array(scales) * amplitude, CFG['c'])
            ref['profile'] = profile; ref['amplitude'] = amplitude
            ref['elapsed_seconds'] = time.perf_counter() - start
            write(OUT / 'reference' / f'{profile}_t{ti:02}.json', ref)
            verify(ref['qualified'], f'unqualified reference {profile} t={amplitude}')
            if ti == CFG['truth_index']:
                old = read(ROOT / f'artifacts/structure_tradeoff_v2/R2/reference/{profile}_d16.json')
                error = float(np.abs(np.asarray(ref['mean']) - old['mean']).sum())
                write(OUT / 'reference' / f'{profile}_OLD_ANCHOR.json', dict(L1=error, passed=error <= CFG['anchor_L1_gate']))
                verify(error <= CFG['anchor_L1_gate'], 'previous reference anchor differs')
            print(json.dumps(dict(phase='reference', profile=profile, t=amplitude, qualified=True, seconds=ref['elapsed_seconds'])), flush=True)

def load_profile(profile):
    refs = [read(OUT / 'reference' / f'{profile}_t{i:02}.json') for i in range(31)]
    verify(all(r['qualified'] for r in refs), 'unqualified reference')
    variants = np.array([r['variants'] for r in refs])
    p = variants[20, 2]
    _, _, n, lg, lt, _, ind = core.geometry(CFG['d'])
    joint = -np.log(variants[:, 2]) + lg
    qn = variants[:, 2] @ ind
    single = -np.log(qn[:, n]) + lt
    pair = np.array([core.pair_loss_vectors(r) for r in refs])
    H = -float(p @ (np.log(p) - lg))
    kl = joint @ p - H; skl = single @ p - H
    countkl = -np.log(qn) @ (p @ ind) + float((p @ ind) @ np.log(p @ ind))
    allrisks = []
    for vi in range(3):
        pv = variants[20, vi]; hv = -float(pv @ (np.log(pv) - lg))
        for vj in range(3):
            qv = variants[:, vj]; nv = qv @ ind
            allrisks.append(np.stack([(-np.log(qv) + lg) @ pv - hv, (-np.log(nv[:, n]) + lt) @ pv - hv]))
    sens = float(np.ptp(allrisks, axis=0).max())
    verify(sens <= CFG['reference_sensitivity_gate'], 'candidate risk reference sensitivity')
    return dict(refs=refs, variants=variants, p=p, H=H, joint=joint, single=single, pair=pair,
                kl=kl, skl=skl, countkl=countkl, reference_sensitivity=sens)

def bank_stream(profile, pi, stream):
    info = load_profile(profile); p = info['p']; scales = np.array(CFG['profiles'][profile])
    d = CFG['d']; _, _, n, lg, lt, _, ind = core.geometry(d)
    seed = CFG['bank_seed'] + 1000 * pi + stream
    rng = np.random.Generator(np.random.PCG64(seed)); initial = rng.bit_generator.state
    means = np.zeros((31, 3, 2, len(p))); bank_scores = np.zeros((64, 31, 3, 2))
    normerr = 0.; identity = 0.; start = time.perf_counter()
    for offset in range(0, 64, 16):
        z = rng.standard_normal((16000, d))
        for ti, amplitude in enumerate(CFG['grid']):
            eta = core.eta_from_z(z, scales * amplitude, CFG['c'])
            j = core.joint_mass(eta); pn = core.pb(eta)
            normerr = max(normerr, float(np.max(abs(j.sum(1) - 1))), float(np.max(abs(pn.sum(1) - 1))))
            identity = max(identity, float(np.max(abs(j @ ind - pn))))
            j = j.reshape(16, 1000, -1); pn = pn.reshape(16, 1000, d + 1)
            for ki, k in enumerate(CFG['K']):
                qg = j[:, :k].mean(1); qn = pn[:, :k].mean(1)
                verify(np.all(qg > 0) and np.all(qn > 0), 'nonpositive particle likelihood')
                qs = qn[:, n] * np.exp(lg - lt)
                identity = max(identity, float(np.max(abs(qg @ ind - qn))), float(np.max(abs(qs @ ind - qn))))
                loss = np.stack([-np.log(qg) + lg, -np.log(qn[:, n]) + lt], axis=1)
                means[ti, ki] += loss.sum(0)
                bank_scores[offset:offset + 16, ti, ki] = loss @ p
    means /= 64
    sensvalues = np.einsum('tkma,va->vtkm', means, info['variants'][20])
    differences = sensvalues - sensvalues[:, 20:21]
    sens = max(float(np.ptp(sensvalues, axis=0).max()), float(np.ptp(differences, axis=0).max()))
    verify(normerr < CFG['mass_gate'] and identity < CFG['identity_gate'], 'probability mass/count identity failure')
    verify(sens <= CFG['reference_sensitivity_gate'], 'bank reference sensitivity failed')
    dseed = CFG['data_seed'] + 1000 * pi + stream
    data_rng = np.random.Generator(np.random.PCG64(dseed)); dinitial = data_rng.bit_generator.state
    small = data_rng.multinomial(4096, p); large = small + data_rng.multinomial(61440, p)
    counts = np.stack([small, large]); verify(np.array_equal(counts.sum(1), CFG['train_N']), 'training count size')
    target = OUT / 'local' / f'{profile}_s{stream:02}.npz'
    np.savez_compressed(target, mean_pattern_loss=means, bank_population_scores=bank_scores, training_counts=counts)
    write(OUT / 'streams' / f'{profile}_s{stream:02}.json', dict(profile=profile, stream=stream,
        bank_seed=seed, bank_initial_RNG=initial, bank_final_RNG=rng.bit_generator.state,
        data_seed=dseed, data_initial_RNG=dinitial, data_final_RNG=data_rng.bit_generator.state,
        training_counts=counts, bank_count=64, unique_prior_particles=64000, candidate_particle_evaluations=1984000,
        population_objectives=means @ p,
        finite_particle_objectives=np.stack([means @ (counts[ni] / N) for ni, N in enumerate(CFG['train_N'])]),
        count_observations=65536, normalization_error=normerr, count_identity_error=identity,
        propagated_reference_sensitivity=sens, numerical_pass=True,
        array_path=target, array_sha256=sha(target), elapsed_seconds=time.perf_counter() - start))
    print(json.dumps(dict(phase='stream', profile=profile, stream=stream, seconds=time.perf_counter() - start,
        numerical_pass=True)), flush=True)

def ci(values, family, gate):
    values = np.asarray(values); mean = float(values.mean()); se = float(values.std(ddof=1) / 4)
    critical = float(student_t.ppf(1 - CFG['alpha'] / (2 * family), 15))
    return dict(mean=mean, SE=se, lower=mean - critical * se, upper=mean + critical * se, precision_pass=se <= gate)

def collect():
    curves = []; poprows = []; popfits = []; finitefits = []; bandsets = []; riskrows = []; primaries = []; rawlocal = []
    for profile in CFG['profiles']:
        info = load_profile(profile); p = info['p']; H = info['H']
        arrays = [np.load(OUT / 'local' / f'{profile}_s{s:02}.npz') for s in range(16)]
        logs = np.array([a['mean_pattern_loss'] for a in arrays])
        pop = logs @ p
        fixed = {'J': info['joint'], 'P': info['pair']}
        objlogs = np.stack([np.broadcast_to(fixed['J'], (16, 31, len(p))), np.broadcast_to(fixed['P'], (16, 31, len(p)))] +
            [logs[:, :, ki, mi] for mi in range(2) for ki in range(3)], axis=2)
        # shape streams x grid x objectives x count; all train choices below use counts alone.
        for ti, amplitude in enumerate(CFG['grid']):
            curves.append(dict(profile=profile, t=amplitude, true_Q_KL=float(info['kl'][ti]),
                deployed_S_KL=float(info['skl'][ti]), total_count_KL=float(info['countkl'][ti]),
                exact_joint_loss=float(info['joint'][ti] @ p), pair_loss=float(info['pair'][ti] @ p),
                exact_single_loss=float(info['single'][ti] @ p), reference_sensitivity=info['reference_sensitivity']))
        for oi, objective in enumerate(CFG['objectives']):
            estimate = (objlogs[:, :, oi] @ p).mean(0); ti = int(np.argmin(estimate))
            popfits.append(dict(profile=profile, unit='aggregate', objective=objective, t=CFG['grid'][ti],
                selected_index=ti, objective_value=float(estimate[ti]), true_Q_KL=float(info['kl'][ti]),
                deployed_KL=float(info['skl'][ti] if objective.startswith('S') else info['kl'][ti])))
        for mi, method in enumerate(['G', 'S']):
            for ki, k in enumerate(CFG['K']):
                bands = [None] * 31; bands[20] = dict(lower=0., upper=0., precision_pass=True)
                for ti, amplitude in enumerate(CFG['grid']):
                    if ti == 20: continue
                    stat = ci(pop[:, ti, ki, mi] - pop[:, 20, ki, mi], 360, CFG['population_SE_gates'][ki]); bands[ti] = stat
                    row = dict(profile=profile, method=method, K=k, t=amplitude, **stat)
                    poprows.append(row)
                    if method == 'G' and k == 10 and ti == 10:
                        primaries.append(dict(kind='population_fixed_witness', profile=profile,
                            pass_primary=bool(stat['precision_pass'] and stat['upper'] < -.001 and info['kl'][10] > .001),
                            difference=stat['mean'], upper=stat['upper'], witness_KL=float(info['kl'][10])))
                cap = min(b['upper'] for b in bands)
                selected = [i for i, b in enumerate(bands) if b['lower'] <= cap]
                bandsets.append(dict(profile=profile, method=method, K=k, indices=selected,
                    grid=[CFG['grid'][i] for i in selected], all_precision_pass=all(b['precision_pass'] for b in bands),
                    truth_excluded=20 not in selected, min_upper=cap, min_true_Q_KL=float(info['kl'][selected].min()),
                    max_true_Q_KL=float(info['kl'][selected].max()), confidence_family=360,
                    min_true_Q_KL_minus_qualification_allowance=max(0., float(info['kl'][selected].min()) - CFG['reference_sensitivity_gate']),
                    numerical_allowance=CFG['reference_sensitivity_gate'], conditional_on_qualified_reference=True,
                    t_approximation_not_finite_sample_certificate=True))
                for stream in range(16):
                    ti = int(np.argmin(pop[stream, :, ki, mi]))
                    popfits.append(dict(profile=profile, unit=str(stream), objective=method + str(k), t=CFG['grid'][ti],
                        selected_index=ti, objective_value=float(pop[stream, ti, ki, mi]), true_Q_KL=float(info['kl'][ti]),
                        deployed_KL=float(info['skl'][ti] if method == 'S' else info['kl'][ti])))
        finite_risks = np.zeros((16, 2, 8))
        for stream, array in enumerate(arrays):
            for ni, N in enumerate(CFG['train_N']):
                counts = array['training_counts'][ni]; empirical = counts / N
                losses = objlogs[stream] @ empirical
                for oi, objective in enumerate(CFG['objectives']):
                    ti = int(np.argmin(losses[:, oi])); deployed = info['skl'][ti] if objective.startswith('S') else info['kl'][ti]
                    finite_risks[stream, ni, oi] = deployed
                    finitefits.append(dict(profile=profile, stream=stream, N=N, objective=objective, t=CFG['grid'][ti],
                        selected_index=ti, train_objective=float(losses[ti, oi]), true_Q_KL=float(info['kl'][ti]),
                        deployed_KL=float(deployed), total_count_KL=float(info['countkl'][ti]), grid_minimum_gap=0.))
        comparisons = [('G10', 'J'), ('G100', 'J'), ('G1000', 'J'), ('S10', 'G10'), ('P', 'J')]
        for ni, N in enumerate(CFG['train_N']):
            for left, right in comparisons:
                stat = ci(finite_risks[:, ni, CFG['objectives'].index(left)] - finite_risks[:, ni, CFG['objectives'].index(right)], 20, .001)
                riskrows.append(dict(profile=profile, N=N, comparison=left + '-' + right, **stat))
                if N == 65536 and left in ('G10', 'P'):
                    passed = stat['precision_pass'] and (stat['lower'] > .001 if left == 'G10' else stat['lower'] >= -.001 and stat['upper'] <= .001)
                    primaries.append(dict(kind='finite_data_excess' if left == 'G10' else 'pair_equivalence', profile=profile,
                        pass_primary=bool(passed), difference=stat['mean'], lower=stat['lower'], upper=stat['upper']))
        # Raw local derivative diagnostic: no raw fit or continuous/global certificate.
        na, nb, _, _, _, _, _ = core.geometry(16); sa = 2 * na - 8; sb = 2 * nb - 8
        mAA = float((p @ sa ** 2 - 8) / 56); mBB = float((p @ sb ** 2 - 8) / 56); mAB = float(p @ (sa * sb) / 64)
        scaleA, scaleB = CFG['profiles'][profile]; rho = .9 / 15
        trace = 8 * (scaleA ** 2 + scaleB ** 2)
        B = -rho * (56 * scaleA ** 2 * mAA + 56 * scaleB ** 2 * mBB + 128 * scaleA * scaleB * mAB)
        rawlocal.append(dict(profile=profile, trace_S=trace, B=B, critical_K=1 + trace / B,
            derivatives={str(k): (trace - (k - 1) * B) / (8 * k) for k in CFG['K']},
            parameter='right derivative in v=t^2 at zero', raw_fits=0))
        for a in arrays: a.close()
    verify(len(finitefits) == 512 and len(poprows) == 360 and len(riskrows) == 20 and len(popfits) == 208, 'ledger row count')
    csvwrite(OUT / 'REFERENCE_CURVES.csv', curves); csvwrite(OUT / 'POPULATION_BANDS.csv', poprows)
    csvwrite(OUT / 'POPULATION_FITS.csv', popfits); csvwrite(OUT / 'FINITE_FITS.csv', finitefits)
    csvwrite(OUT / 'FINITE_RISK_INTERVALS.csv', riskrows)
    write(OUT / 'SELECTION_SETS.json', bandsets); write(OUT / 'RAW_LOCAL_DIAGNOSTIC.json', rawlocal)
    write(OUT / 'DECISION.json', dict(primary=primaries, finite_grid_fits=512, bank_only_grid_fits=192,
        aggregate_population_fits=16, population_precision_pass=sum(r['precision_pass'] for r in poprows),
        finite_precision_pass=sum(r['precision_pass'] for r in riskrows), unique_prior_particles=2048000,
        candidate_particle_evaluations=63488000, count_observations=2097152, new_neural_models=0,
        finite_grid_not_continuous_global_optimum=True, separate_confidence_families=True))
    files = list((OUT / 'reference').glob('*.json')) + list((OUT / 'streams').glob('*.json'))
    files += list(OUT.glob('*.csv')) + [OUT / n for n in ['SELECTION_SETS.json', 'RAW_LOCAL_DIAGNOSTIC.json', 'DECISION.json', 'SMOKE.json']]
    write(OUT / 'RESULTS_FREEZE.json', dict(time_utc=now(), files=hashes(files)))
    write(OUT / 'STATE.json', dict(status='complete', time_utc=now(), active_jobs=[]), False)
    print(json.dumps(read(OUT / 'DECISION.json')), flush=True)

def run():
    checkfreeze(); verify(read(OUT / 'STATE.json')['status'] == 'frozen', 'No silent scientific restart')
    smoke()
    write(OUT / 'STATE.json', dict(status='running_references', time_utc=now(), pid=os.getpid()), False)
    references()
    for pi, profile in enumerate(CFG['profiles']):
        for stream in range(16):
            write(OUT / 'STATE.json', dict(status='running_banks', profile=profile, stream=stream, time_utc=now(), pid=os.getpid()), False)
            bank_stream(profile, pi, stream)
    # All finite-data inputs and computational surfaces exist before endpoint evaluation.
    write(OUT / 'TRAINING_INPUTS_FREEZE.json', dict(time_utc=now(), files=hashes(list((OUT / 'streams').glob('*.json')))))
    collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'run']); args = parser.parse_args()
    try:
        freeze() if args.phase == 'freeze' else run()
    except Exception as ex:
        if (OUT / 'FREEZE.json').exists() and not (OUT / 'RESULTS_FREEZE.json').exists():
            write(OUT / f'FAILURE_{time.time_ns()}.json', dict(time_utc=now(), exception=repr(ex), traceback=traceback.format_exc()))
            write(OUT / 'STATE.json', dict(status='stopped_failure', time_utc=now(), active_jobs=[]), False)
        raise
