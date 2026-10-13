"""REC-v2 fixed-state capacity/scale diagnosis; all science requires a new freeze.

Reuses the sealed REC1 quadrature and finite score kernels without modifying them.
No computation or file output happens on import. demo needs no historical arrays.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
import traceback
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import expit, roots_hermitenorm
from scipy.stats import t
import rec1_diagnostics as kernels
import rec1_verify as independent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rec_scale_v2_20261005'
DOC = ROOT / 'docs/08_research_研究工作/REC_v2_20261005'
OLD = ROOT / 'artifacts/rec_v1_20261004/REC1'
JOINT = ROOT / 'artifacts/joint_scale_learning'
Y = np.array(list(itertools.product((0, 1), repeat=4)))
PAIRS = tuple(itertools.combinations(range(4), 2))
ARMS = ('NEAR_FREE', 'UNIT_FREE')
STATES = ('BASE', 'MU_SAFE', 'SCALE2')
MODES = ('retained', 'eliminated')
TARGETS = ('F_D', 'F_P')
HEAD = '9bdb4d532d187d9d9195d106663eb04d9d8f6185'
HANDOFF = '853358fe26a70335d6a5fb627372433694de9e410ebbccf0d6f9e49623e0c531'
CONFIG = dict(streams=16, banks=2048, K=100, nuisance=256, chunk=32, seed=202610051210,
              rng='PCG64DXSM', workers=4, threads=1, mean_step=4., scale_multiplier=2.,
              mean_bounds=[-2., 2.], scale_bounds=[.001, 2.], delta=.1,
              marginal_order=64, marginal_audit_order=96, marginal_tolerance=5e-11,
              root_xtol=2e-14, root_rtol=1e-14, root_max_calls=102,
              orders=[32, 40, 48], main_order=40, pmf_tolerance=2e-7,
              KL_tolerance=2e-6, mass_tolerance=2e-12, risk_resolution=4e-6,
              empirical_resolution=1e-10, family=16, alpha=.05, df=2047, SE_limit=2e-5,
              audit_streams=[0, 4, 8, 12], audit_order=44, audit_state='SCALE2')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def marginal(mu, scale, order):
    x, w = roots_hermitenorm(order)
    return float(w @ expit(float(mu) + float(scale) * x) / math.sqrt(2 * math.pi))


def capacity(scales, P):
    spins = 2 * Y - 1
    means = P @ spins
    covariance = (spins - means).T @ (P[:, None] * (spins - means))
    rows = []
    for i, j in PAIRS:
        bound = .9 * float(scales[i]) * float(scales[j]) / 4
        teacher = float(covariance[i, j])
        gap = max(0., abs(teacher) - bound)
        rows.append(dict(pair=[i, j], teacher_spin_covariance=teacher, capacity_upper_bound=bound,
                         bound_over_teacher=bound / abs(teacher) if teacher else None,
                         gap=gap, pair_KL_lower_bound=gap ** 2 / 8,
                         excludes_exact_pair_match=gap > 1e-12))
    return dict(pairs=rows, maximum_KL_lower_bound=max(r['pair_KL_lower_bound'] for r in rows),
                scope='Uniform in means and allowed C at these fixed scales; maximum, never sum, of pair bounds')


def candidates(base, gradient):
    mu = np.array(base['mu'])
    s = np.array(base['s'])
    mean = kernels.model(np.clip(mu - 4 * np.array(gradient), -2, 2), s, base['rho'], base['tau'])
    q0 = np.array([marginal(m, a, 64) for m, a in zip(mu, s)])
    proposed_scales = np.minimum(2., 2 * s)
    matched = []
    roots = []
    feasible = True
    for index, (target, scale) in enumerate(zip(q0, proposed_scales)):
        calls = 0
        def residual(m):
            nonlocal calls
            calls += 1
            assert calls <= 102, 'Frozen marginal root function-call budget exceeded'
            return marginal(m, scale, 64) - target
        left, right = residual(-2), residual(2)
        ok = left <= 0 <= right
        if ok:
            value = brentq(residual, -2., 2., xtol=2e-14, rtol=1e-14, maxiter=100)
            matched.append(float(value))
        else:
            feasible = False
            value = None
            matched.append(float(mu[index]))
        roots.append(dict(coordinate=index, calls=calls, feasible=bool(ok),
                          endpoint_residuals=[left, right], root=value))
    scaled = kernels.model(matched, proposed_scales, base['rho'], base['tau']) if feasible else base
    audit_base = np.array([marginal(m, a, 96) for m, a in zip(mu, s)])
    audit_scale = np.array([marginal(m, a, 96) for m, a in zip(scaled['mu'], scaled['s'])])
    delta = np.array(mean['mu']) - mu
    return dict(BASE=base, MU_SAFE=mean, SCALE2=scaled), dict(
        gradient_source='Frozen REC1 GH24 empirical mean gradient', gradient=list(gradient),
        mean_actual_step=delta.tolist(), exact_gradient_descent_bound=-float(delta @ delta) / 8,
        mean_linear_prediction=float(np.array(gradient) @ delta),
        marginal_target64=q0.tolist(), marginal_base96=audit_base.tolist(),
        marginal_scale96=audit_scale.tolist(), marginal_max_error=float(np.max(abs(audit_scale - q0))),
        marginal_base_order_error=float(np.max(abs(audit_base - q0))),
        scale_feasible=bool(feasible), roots=roots, proposed_scales=proposed_scales.tolist(),
        teacher_used=False, evaluation_bank_used=False)


def direct_scores(state, active, noise):
    z = active @ np.linalg.cholesky(state['covariance']).T + state['mu']
    a = 1 / (1 + np.exp(-z))
    w = np.prod(1 / (1 + np.exp(-state['tau'] * noise)), axis=2)
    out = np.zeros((len(active), 2, 16))
    for k, y in enumerate(Y):
        likelihood = np.prod(np.where(y, a, 1 - a), axis=2)
        out[:, 0, k] = -np.log(np.mean(likelihood * w, axis=1)) - noise.shape[2] * math.log(2)
        out[:, 1, k] = -np.log(likelihood.mean(axis=1))
    return out


def manufactured():
    state = kernels.model([-.3, -.1, .2, .4], [.3, .4, .5, .6], [.2, -.1, .05, .06, -.15, .25], .05)
    empirical = np.arange(1., 17.) / 136.
    quad = kernels.quadrature(state, 24, True)
    g = -np.array(quad['dmu']).T @ (empirical / np.array(quad['pmf']))
    states, meta = candidates(state, g)
    assert meta['scale_feasible'] and meta['marginal_max_error'] <= 5e-11
    q_safe = independent.independent_pmf(states['MU_SAFE']['mu'], np.array(state['covariance']), 24)['pmf']
    delta_nll = float(-empirical @ np.log(q_safe) + empirical @ np.log(quad['pmf']))
    assert delta_nll <= meta['exact_gradient_descent_bound'] + 1e-9
    # Independent finite differences of GH NLL gradients check the Hessian upper bound.
    H = np.zeros((4, 4))
    for i in range(4):
        gradients = []
        for sign in (-1., 1.):
            m = np.array(state['mu']); m[i] += sign * 1e-4
            trial = kernels.model(m, state['s'], state['rho'], state['tau'])
            q = kernels.quadrature(trial, 20, True)
            gradients.append(-np.array(q['dmu']).T @ (empirical / np.array(q['pmf'])))
        H[:, i] = (gradients[1] - gradients[0]) / 2e-4
    symmetry_error = float(np.max(abs(H - H.T)))
    hmax = float(np.linalg.eigvalsh((H + H.T) / 2).max())
    assert symmetry_error < 1e-8 and hmax <= .25 + 1e-8
    edge = kernels.model([1.99] * 4, [1.] * 4, [0.] * 6, .05)
    _, bad = candidates(edge, [0.] * 4)
    assert not bad['scale_feasible']
    active = np.sin(np.arange(3 * 7 * 4).reshape(3, 7, 4) * .37)
    noise = np.cos(np.arange(3 * 7 * 5).reshape(3, 7, 5) * .29)
    score_error = float(np.max(abs(kernels.scores(state, active, noise) - direct_scores(state, active, noise))))
    assert score_error < 1e-12
    x, w = roots_hermitenorm(40); w = w / math.sqrt(2 * math.pi)
    covariance_checks = []
    for rho in (-.9, 0., .9):
        zi = -.2 + .2 * x[:, None]
        zj = .3 + .5 * (rho * x[:, None] + math.sqrt(1 - rho ** 2) * x[None, :])
        a, b = np.tanh(zi / 2), np.tanh(zj / 2)
        weight = np.outer(w, w)
        cov = float(np.sum(weight * a * b) - np.sum(weight * a) * np.sum(weight * b))
        bound = .2 * .5 * .9 / 4
        assert abs(cov) <= bound + 1e-13
        covariance_checks.append(dict(rho=rho, covariance=cov, bound=bound))
    return dict(passed=True, scope='Deterministic manufactured tensors; no historical arrays, scientific RNG or training',
                mean_NLL_change=delta_nll, exact_gradient_bound=meta['exact_gradient_descent_bound'],
                max_Hessian_eigenvalue=hmax, Hessian_symmetry_error=symmetry_error,
                matched_marginal_error=meta['marginal_max_error'], infeasible_branch_checked=True,
                direct_score_error=score_error, covariance_checks=covariance_checks,
                illustrative_spin_capacity=.9 * .07 * .10 / 4,
                covariance_Lipschitz_constant_for_Bernoulli=1., spin_KL_gap_factor=1/8)


def sources_inputs():
    sources = [Path(__file__), ROOT / 'script/recscale_audit.py',
               Path(kernels.__file__), Path(independent.__file__),
               DOC / 'CONTRACT.md', DOC / 'THEORY.md', DOC / 'HANDOFF_DISPOSITION.md']
    inputs = [ROOT / 'HANDOFF.md', OUT / 'HANDOFF_INPUT.md', kernels.TRUTH,
              OLD / 'RESULTS.json', ROOT / 'artifacts/rec_v1_20261004/DELIVERY_MANIFEST.json']
    for j in range(16):
        p = JOINT / 'TRAIN' / f'stream_{j:02}.json'
        inputs += [p, ROOT / read(p)['input_file'], OLD / f'models_{j:02}.json']
    return {rel(p): sha(p) for p in sources}, {rel(p): sha(p) for p in inputs}


def qualify():
    start = time.perf_counter()
    try:
        result = manufactured()
        result.update(source_sha256=sha(__file__), reused_kernel_sha256=sha(kernels.__file__),
                      independent_kernel_sha256=sha(independent.__file__), seconds=time.perf_counter()-start,
                      environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
        save(OUT / 'QUALIFICATION.json', result)
        print(json.dumps(result))
    except Exception:
        number = len(list(OUT.glob('QUALIFICATION_FAILURE_*.json')))
        save(OUT / f'QUALIFICATION_FAILURE_{number:02}.json', dict(traceback=traceback.format_exc(), source_sha256=sha(__file__)))
        raise


def freeze():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == HEAD
    assert sha(ROOT / 'HANDOFF.md') == HANDOFF
    qualification = read(OUT / 'QUALIFICATION.json')
    assert qualification['passed'] and qualification['source_sha256'] == sha(__file__)
    sources, inputs = sources_inputs()
    save(OUT / 'FREEZE.json', dict(authorized=True, created_utc=datetime.now(timezone.utc).isoformat(),
         baseline=HEAD, handoff_sha256=HANDOFF, handoff_use='reference', config=CONFIG, sources=sources,
         inputs=inputs, qualification_sha256=sha(OUT / 'QUALIFICATION.json'),
         environment=qualification['environment']))
    print(json.dumps(dict(frozen=True, sha256=sha(OUT / 'FREEZE.json'))))


def validate():
    f = read(OUT / 'FREEZE.json')
    assert f['config'] == CONFIG and f['authorized']
    for group in ('sources', 'inputs'):
        for p, digest in f[group].items():
            assert sha(ROOT / p) == digest, p
    assert sha(OUT / 'QUALIFICATION.json') == f['qualification_sha256']
    return f


def stream_run(stream):
    tick = time.perf_counter()
    try:
        old = read(OLD / f'models_{stream:02}.json')
        source = read(JOINT / 'TRAIN' / f'stream_{stream:02}.json')
        assert sha(ROOT / source['input_file']) == source['input_sha256']
        with np.load(ROOT / source['input_file'], allow_pickle=False) as data:
            empirical = np.bincount(data['active_labels'] @ np.array([8,4,2,1]), minlength=16) / 4096.
            assert data['active_labels'].sum(axis=0).tolist() == old['baseline']['counts']
        assert np.max(abs(empirical-old['empirical'])) < 1e-15
        P, p, true_product = kernels.teacher()
        rows = []
        for arm in ARMS:
            original = next(r for r in old['records'] if r['arm']==arm and r['state']=='BASE')
            base = original['model']
            g = original['construction']['directions']['MU']['gradient']
            states, meta = candidates(base, g)
            for name in STATES:
                state = states[name]
                variants = original['variants'] if name=='BASE' else [kernels.quadrature(state,o) for o in (32,40,48)]
                Q = np.array(next(v['pmf'] for v in variants if v['order']==40))
                R = kernels.risk(P,Q)
                qualification = []
                for v in variants:
                    change = float(np.max(abs(np.array(v['pmf'])-Q)))
                    rc = abs(kernels.risk(P,v['pmf'])-R)
                    passed = change<=2e-7 and rc<=2e-6 and v['mass_error']<=2e-12 and min(v['pmf'])>0
                    qualification.append(dict(order=v['order'], max_PMF_change=change, KL_change=rc, mass_error=v['mass_error'], passed=bool(passed)))
                marginal_error = float(np.max(abs(Q@Y-np.array(original['main_pmf'])@Y)))
                if name=='SCALE2':
                    marginal_pass = marginal_error<=2e-7 and meta['marginal_max_error']<=5e-11 and meta['marginal_base_order_error']<=5e-11
                else:
                    marginal_pass = True
                nll = float(-empirical@np.log(Q))
                delta_nll = nll-original['empirical_exact_NLL']
                rows.append(dict(stream=stream,arm=arm,state=name,model=state,main_pmf=Q.tolist(),variants=variants,
                    risk=R, delta_risk=R-original['risk'], empirical_NLL=nll, empirical_NLL_change=delta_nll,
                    mean_descent_status=('decrease' if delta_nll < -1e-10 else 'increase' if delta_nll>1e-10 else 'unresolved') if name=='MU_SAFE' else None,
                    population_NLL=float(-P@np.log(Q)), marginal_change=marginal_error,
                    marginal_passed=bool(marginal_pass), qualification=qualification,
                    qualified=bool(all(v['passed'] for v in qualification) and marginal_pass),
                    construction=meta if name=='BASE' else None, capacity=capacity(state['s'],P) if name!='MU_SAFE' else None))
        modelpath=OUT/f'models_{stream:02}.json'
        save(modelpath,dict(stream=stream,empirical=empirical.tolist(),reused_baseline=old['baseline'],records=rows))
        ra=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],stream,0])))
        rn=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG['seed'],stream,1])))
        banks=np.empty((2048,2,3,2,2))
        hashes=[]
        for start in range(0,2048,32):
            a=ra.standard_normal((32,100,4)); n=rn.standard_normal((32,100,256))
            hashes.append(dict(start=start,active=hashlib.sha256(a.tobytes()).hexdigest(),nuisance=hashlib.sha256(n.tobytes()).hexdigest()))
            for ai,arm in enumerate(ARMS):
                tau=next(r['model']['tau'] for r in rows if r['arm']==arm)
                logw=(math.log(2)-np.logaddexp(0.,-tau*n)).sum(axis=2)
                for si,name in enumerate(STATES):
                    state=next(r['model'] for r in rows if r['arm']==arm and r['state']==name)
                    score=kernels.scores(state,a,n,log_nuisance_relative=logw)
                    banks[start:start+32,ai,si]=score@np.stack((empirical,P),axis=1)
        scorepath=OUT/f'scores_{stream:02}.npz'
        assert not scorepath.exists()
        np.savez_compressed(scorepath,reduced_scores=banks)
        stats=[]
        for ai,arm in enumerate(ARMS):
            for si,name in enumerate(STATES):
                for mi,mode in enumerate(MODES):
                    for ti,target in enumerate(TARGETS):
                        values=banks[:,ai,si,mi,ti]; difference=values-banks[:,ai,0,mi,ti]
                        stats.append(dict(arm=arm,state=name,mode=mode,target=target,mean=float(values.mean()),
                            delta=float(difference.mean()),bank_variance=float(difference.var(ddof=1))))
        record=dict(stream=stream,model_file=rel(modelpath),model_sha256=sha(modelpath),score_file=rel(scorepath),
                    score_sha256=sha(scorepath),stats=stats,bank_hashes=hashes,qualified=all(r['qualified'] for r in rows),seconds=time.perf_counter()-tick)
        save(OUT/f'stream_{stream:02}.json',record)
        return record
    except Exception:
        save(OUT/f'ERROR_stream_{stream:02}.json',dict(stream=stream,traceback=traceback.format_exc(),automatic_retry=False))
        raise


def summary(records):
    models=[r for stream in records for r in read(ROOT/stream['model_file'])['records']]
    critical=float(t.ppf(1-.05/32,2047))
    family=[]
    for arm,name,mode,target in itertools.product(ARMS,STATES[1:],MODES,TARGETS):
        selected=[next(r for r in s['stats'] if (r['arm'],r['state'],r['mode'],r['target'])==(arm,name,mode,target)) for s in records]
        mean=float(np.mean([r['delta'] for r in selected]));se=float(math.sqrt(sum(r['bank_variance']/2048 for r in selected))/16)
        interval=[mean-critical*se,mean+critical*se]
        family.append(dict(arm=arm,candidate=name,mode=mode,target=target,delta=mean,SE=se,interval=interval,
            direction='decrease' if interval[1]<0 else 'increase' if interval[0]>0 else 'unresolved',precision_passed=se<=2e-5))
    risk_rows=[]
    for arm,name in itertools.product(ARMS,STATES):
        group=[r for r in models if r['arm']==arm and r['state']==name]
        deltas=[r['delta_risk'] for r in group]
        risk_rows.append(dict(arm=arm,state=name,mean_risk=float(np.mean([r['risk'] for r in group])),
            paired_delta=deltas,mean_delta=float(np.mean(deltas)),delta_range=[min(deltas),max(deltas)],
            risk_decrease_resolved=float(np.mean(deltas)) < -4e-6,
            empirical_NLL_mean_change=float(np.mean([r['empirical_NLL_change'] for r in group])),
            empirical_descent_counts={key:sum(r['mean_descent_status']==key for r in group) for key in ('decrease','increase','unresolved')},
            max_marginal_change=max(r['marginal_change'] for r in group),qualified=all(r['qualified'] for r in group)))
    caps=[]
    for arm in ARMS:
        selected=[r for r in models if r['arm']==arm and r['state']=='BASE']
        pairs=[p for r in selected for p in r['capacity']['pairs']]
        caps.append(dict(arm=arm,excluded_pairs=sum(p['excludes_exact_pair_match'] for p in pairs),total_pairs=96,
            models_with_any_gap=sum(any(p['excludes_exact_pair_match'] for p in r['capacity']['pairs']) for r in selected),
            capacity_ratio_range=[min(p['bound_over_teacher'] for p in pairs),max(p['bound_over_teacher'] for p in pairs)],
            KL_lower_bound_range=[min(r['capacity']['maximum_KL_lower_bound'] for r in selected),max(r['capacity']['maximum_KL_lower_bound'] for r in selected)]))
    P,p,prod=kernels.teacher()
    return dict(config=CONFIG,freeze_sha256=sha(OUT/'FREEZE.json'),records=records,teacher_P=P.tolist(),
        true_product_risk=kernels.risk(P,prod),reused_baseline_mean=read(OLD/'RESULTS.json')['baseline_mean_risk'],
        family=family,risk=risk_rows,capacity=caps,qualified=all(r['qualified'] for r in records),
        feasible_scale_candidates=sum(r['construction']['scale_feasible'] for r in models if r['state']=='BASE'),
        marginal_root_calls=sum(sum(v['calls'] for v in r['construction']['roots']) for r in models if r['state']=='BASE'),
        accounting=dict(new_observations=0,new_training_trajectories=0,new_baseline_fits=0,base_models=32,candidates=64,
             new_GH_PMF=192,new_GH_nodes=64*sum(n**4 for n in (32,40,48)),audit_PMF=8,audit_GH_nodes=8*44**4,
             marginal_base64_calls=128,marginal_audit96_calls=256,prior_vectors=16*2048*100,Gaussian=16*2048*100*260,
             pattern_scores=16*2048*2*3*2*16,active_logits=16*2048*100*4*6,nuisance_logits=16*2048*100*256*2),
        scope='Conditional saved-panel diagnosis, not new training-population inference; inequalities use frozen numerical teacher; no rigorous GH remainder certificate')


def run():
    validate();tick=time.perf_counter()
    save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(OUT/'FREEZE.json'),automatic_retry=False))
    try:
        records=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(stream_run,s) for s in range(16)]):
                record=future.result();records.append(record)
                print(json.dumps(dict(stream=record['stream'],qualified=record['qualified'],seconds=record['seconds'])),flush=True)
        records.sort(key=lambda r:r['stream']);validate()
        result=summary(records);result['seconds']=time.perf_counter()-tick
        save(OUT/'RESULTS.json',result)
        print(json.dumps(dict(completed=True,qualified=result['qualified'],seconds=result['seconds'])),flush=True)
    except Exception:
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('demo','qualify','freeze','run'))
    command=parser.parse_args().command
    if command=='demo':
        print(json.dumps(manufactured(),indent=2))
    else:
        globals()[command]()
