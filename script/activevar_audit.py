"""Independent active-scale input, score, reference and statistics audit.

No producer numerical routine is imported.  The only project computation loaded
is the unchanged official SCBM concept-loss implementation for bounded controls.
Scientific run requires a separately authorized, byte-bound audit plan.
"""
from __future__ import annotations
import os
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '1'
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import t as student_t

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
DOC = ROOT / 'docs/08_research_研究工作/有效尺度自由度与有限曲率_20261004'
BASE = ROOT / 'artifacts/active_scale_boundary'
OUT = BASE / 'AUDIT'
OFFICIAL = ROOT / 'reproducibility_snapshots/cbm_public_sources/SCBM/58c3e602cb11fb4ea9106226e0f9e58f50d5d10a/models/losses.py'
PATTERNS = np.asarray(list(itertools.product((0, 1), repeat=4)), dtype=np.int64)
N_VALUES = (0, 4, 16, 64, 256)
T_VALUES = (.1, .2, .4)
TOL = dict(arithmetic=2e-11, prefix_score=2e-11,
           native_contrast=1e-4, hessian=2e-12,
           reference_pmf=2e-8, reference_KL=2e-7)
_LOSS = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def require(value, label):
    if not value:
        raise RuntimeError(label)


def near(actual, expected, tol=TOL['arithmetic'], label='arithmetic'):
    left, right = np.asarray(actual, float), np.asarray(expected, float)
    require(left.shape == right.shape, label + ' shape')
    require(np.isfinite(left).all() and np.isfinite(right).all(), label + ' finite')
    err = float(np.max(np.abs(left-right))) if left.size else 0.
    require(err <= tol, f'{label}: {err} > {tol}')
    return err


def official_loss():
    global _LOSS
    if _LOSS is None:
        spec = importlib.util.spec_from_file_location('activevar_independent_official', OFFICIAL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOSS = module.SCBLoss(alpha=1., config=SimpleNamespace(
            reg_precision='l1', reg_weight=1., num_monte_carlo=100,
            training_mode='joint'))
    return _LOSS


def pattern_losses(active_logits, noise_logits):
    """NumPy/SciPy independent score: [bank,K,4] -> [bank,16]."""
    active_logits = np.asarray(active_logits, float)
    signed = active_logits[:, :, None, :] * (2*PATTERNS-1)[None, None, :, :]
    terms = -np.logaddexp(0., -signed).sum(axis=-1)
    if noise_logits.shape[-1]:
        terms += -np.logaddexp(0., -noise_logits).sum(axis=-1)[:, :, None]
    return math.log(active_logits.shape[1]) - logsumexp(terms, axis=1)


def native_pattern_losses(active_logits, noise_logits):
    """One row per official call: SCBLoss reduces its batch to one scalar."""
    active_logits = torch.tensor(active_logits, dtype=torch.float32)
    noise_logits = torch.tensor(noise_logits, dtype=torch.float32)
    whole = torch.cat((active_logits, noise_logits), dim=-1)
    probabilities = torch.sigmoid(whole).T.unsqueeze(0)
    scores = []
    for pattern in PATTERNS:
        yy = np.r_[pattern, np.ones(noise_logits.shape[-1], dtype=np.int64)]
        value = official_loss().compute_concept_loss(
            probabilities, torch.tensor(yy, dtype=torch.int64).unsqueeze(0))
        scores.append(float(value))
    return np.asarray(scores), dict(
        calls=16, sigmoid_zero=int((probabilities == 0).sum()),
        sigmoid_one=int((probabilities == 1).sum()),
        unique_sigmoid_entries=int(probabilities.numel()),
        maximum_absolute_logit=float(whole.abs().max()))


def normal_hermite_pmf(mu, covariance, order=36):
    """Probabilists' Hermite rule and symmetric eigenroot, not Cholesky."""
    mu, covariance = np.asarray(mu, float), np.asarray(covariance, float)
    vals, vectors = np.linalg.eigh(covariance)
    require(vals.min() >= -1e-12, 'reference covariance PSD')
    symmetric_root = (vectors * np.sqrt(np.maximum(vals, 0.))) @ vectors.T
    xx, ww = roots_hermitenorm(order)
    ww = ww / math.sqrt(2*math.pi)
    pmf = np.zeros(16)
    count = order**4
    for start in range(0, count, 4096):
        ids = np.arange(start, min(start+4096, count), dtype=np.int64)
        indices = np.stack(np.unravel_index(ids, (order,)*4), axis=1)
        gaussian = xx[indices]
        weights = np.prod(ww[indices], axis=1)
        probabilities = expit(gaussian @ symmetric_root.T + mu)
        for pi, pattern in enumerate(PATTERNS):
            cell = np.prod(np.where(pattern[None, :], probabilities,
                                    1-probabilities), axis=1)
            pmf[pi] += float(weights @ cell)
    require(np.isfinite(pmf).all() and (pmf > 0).all(), 'audit PMF positive')
    mass = float(math.fsum(pmf))
    require(abs(mass-1) <= 2e-12, 'audit PMF mass')
    return dict(order=order, Gaussian_nodes=count, raw_mass=mass,
                raw_pmf=pmf.tolist(), pmf=(pmf/mass).tolist(),
                root='symmetric_eigenroot', quadrature='scipy_roots_hermitenorm')


def KL(p, q):
    return math.fsum(float(x)*math.log(float(x/y)) for x, y in zip(p, q) if x > 0)


def moments(p):
    pp = np.asarray(p, float)
    marginal = pp @ PATTERNS
    centered = PATTERNS-marginal
    gamma = (centered.T*pp) @ centered
    d = marginal*(1-marginal)
    corr = gamma/np.sqrt(d[:, None]*d[None, :])
    return marginal, d, gamma, corr


def hessian_control():
    """Manufactured deterministic arrays; not normal RNG or model integration."""
    z = (np.sin(np.arange(28).reshape(7, 4)*.37)
         + .2*np.cos(np.arange(28).reshape(7, 4)*.13))
    weights = np.asarray([1., 2., 4., 3., 2., 6., 5.]); weights /= weights.sum()
    p = np.arange(1., 17.); p /= p.sum()
    marg, d, gamma, _ = moments(p)
    mu0 = np.log(marg/(1-marg))
    tz, tw = torch.tensor(z), torch.tensor(weights)
    ty, tp = torch.tensor(PATTERNS, dtype=torch.float64), torch.tensor(p)

    def objective(state):
        mu, scale = state[:4], state[4:]
        logits = mu[None, :]+scale[None, :]*tz
        logs = (ty[:, None, :]*logits[None, :, :]
                -torch.nn.functional.softplus(logits)[None, :, :]).sum(dim=-1)
        return -(tp*torch.logsumexp(logs+torch.log(tw)[None, :], dim=1)).sum()

    base = torch.tensor(np.r_[mu0, np.zeros(4)], requires_grad=True)
    observed = torch.autograd.functional.hessian(objective, base).detach().numpy()
    m1 = weights @ z
    m2 = (z.T*weights) @ z
    scale = np.diag(d*np.diag(m2))-gamma*(m2-np.outer(m1, m1))
    expected = np.block([[np.diag(d), np.diag(d*m1)],
                         [np.diag(d*m1), scale]])
    error = near(observed, expected, TOL['hessian'], 'deterministic bank Hessian')
    return dict(passed=True, maximum_Hessian_error=error,
                mean_scale_block_error=near(observed[:4, 4:], np.diag(d*m1),
                                           TOL['hessian'], 'mean-scale block'),
                deterministic_particles=7, patterns=16,
                new_random_draws=0, model_integrals=0, official_API_calls=0)


def validate_root():
    frozen = read(BASE/'FREEZE.json')
    require(frozen.get('science_authorized') is True, 'root science authorization')
    require(frozen['source_sha256'] == sha(ROOT/'script/activevar_compute.py'), 'producer source freeze')
    require(frozen['spec_sha256'] == sha(BASE/'SPEC.json'), 'spec freeze')
    for name, digest in frozen['input_sha256'].items():
        require(sha(ROOT/name) == digest, 'root input '+name)
    spec = read(BASE/'SPEC.json')
    require(spec['mc']['seed_base'] == 202610044000, 'seed contract')
    require(spec['mc']['bit_generator'] == 'PCG64DXSM', 'generator contract')
    require(spec['mc']['streams'] == 16 and spec['mc']['banks_per_stream'] == 2048,
            'scientific panel')
    require(spec['mc']['chunk'] == 32 and spec['mc']['K'] == 100, 'draw shape')
    require(spec['mc']['nuisance_values'] == list(N_VALUES), 'n panel')
    require(spec['t_values'] == [0., .1, .2, .4], 't panel')
    require(spec['patterns'] == PATTERNS.tolist(), 'pattern enumeration')
    return spec


def freeze_audit():
    validate_root()
    paths = [SOURCE, OFFICIAL, BASE/'FREEZE.json', BASE/'SPEC.json',
             BASE/'REFERENCE/RESULTS.json', BASE/'SUMMARY/RESULTS.json',
             ROOT/'script/activevar_compute.py', DOC/'PROTOCOL.md', DOC/'THEORY.md']
    for stream in range(16):
        folder = BASE/'MC'/f'stream_{stream:02d}'
        paths += [folder/'RESULTS.json', folder/'BANKS.npz']
    plan = dict(created_utc=now(), audit_authorized=True,
                root_freeze_sha256=sha(BASE/'FREEZE.json'),
                inputs={rel(p): sha(p) for p in paths}, tolerances=TOL,
                scope=dict(all_input_hash_replay=True, all_saved_statistics=True,
                           prefix_banks_per_stream=2, native_first_bank=True,
                           native_n=[0, 256], native_t=[0., .2],
                           native_scalar_pattern_calls=4096,
                           expected_reference_PMFS=16, reference_order=36,
                           reference_Gaussian_nodes=16*36**4,
                           independent_scientific_draws=0, parameter_fits=0,
                           automatic_retry=False),
                deterministic_control=hessian_control(),
                environment=dict(numpy=np.__version__, torch=torch.__version__,
                                 torch_threads=torch.get_num_threads(),
                                 thread_environment={k:os.environ[k] for k in
                                  ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    save(OUT/'PROTOCOL.json', plan)
    print(json.dumps(dict(audit_protocol_sha256=sha(OUT/'PROTOCOL.json'),
                          source_sha256=sha(SOURCE))), flush=True)


def bind_audit():
    validate_root()
    plan = read(OUT/'PROTOCOL.json')
    require(plan.get('audit_authorized') is True, 'audit authorization')
    require(plan['root_freeze_sha256'] == sha(BASE/'FREEZE.json'), 'root binding')
    require(plan['inputs'][rel(SOURCE)] == sha(SOURCE), 'audit source binding')
    require(plan['tolerances'] == TOL, 'audit tolerances')
    for name, digest in plan['inputs'].items():
        require(sha(ROOT/name) == digest, 'audit input changed '+name)
    return plan


def reference_arithmetic(ref, spec):
    require(ref['source_sha256'] == sha(ROOT/'script/activevar_compute.py') and
            ref['freeze_sha256'] == sha(BASE/'FREEZE.json'), 'reference provenance')
    require(ref['patterns'] == PATTERNS.tolist(), 'reference pattern order')
    expected_qualifications = []
    errors = []
    for ti, (record, teacher) in enumerate(zip(ref['records'], spec['teachers'])):
        require(record['name'] == teacher['name'], 'teacher order')
        near(record['covariance'], teacher['covariance'], 0., 'teacher covariance')
        variants = record['teacher_variants']
        require([x['order'] for x in variants] == [24, 32, 40], 'teacher orders')
        p = np.asarray(variants[2]['pmf']); p32 = np.asarray(variants[1]['pmf'])
        near(p, record['teacher_pmf'], 0., 'teacher main')
        for vv in variants:
            raw = np.asarray(vv['raw_pmf']); mass = float(math.fsum(raw))
            errors.append(near(mass, vv['raw_mass'], label='teacher raw mass'))
            near(raw/mass, vv['pmf'], label='teacher normalization')
            near(abs(mass-1), vv['mass_error'], label='teacher mass error')
            require((raw > 0).all() and vv['Gaussian_nodes'] == vv['order']**4, 'teacher nodes/positive')
        drow = record['derived']; p_marg, dd, gamma_num, ry_num = moments(p)
        gamma = np.diag(dd) if ti == 0 else gamma_num
        ry = np.eye(4) if ti == 0 else ry_num
        eigenvalues, eigenvectors = np.linalg.eigh(ry)
        lam = 1. if ti == 0 else float(eigenvalues[-1])
        vector = np.ones(4)/2 if ti == 0 else eigenvectors[:, -1]
        if vector[np.flatnonzero(np.abs(vector) > 1e-15)[0]] < 0:
            vector = -vector
        signs = np.where(vector >= 0, 1., -1.)
        cc = .1*np.eye(4)+.9*np.outer(signs, signs)
        nn = float(np.sum(vector**2/dd)); direction = np.abs(vector)/np.sqrt(dd*nn)
        bb = .1+.9*lam
        product = np.prod(np.where(PATTERNS, p_marg, 1-p_marg), axis=1)
        product /= math.fsum(product)
        expected = dict(p=p_marg, Gamma=gamma, R_Y=ry,
                        numerical_Gamma_before_null_identity=gamma_num,
                        numerical_R_Y_before_null_identity=ry_num,
                        null_identity_correction_max_abs=np.max(np.abs(ry_num-ry)),
                        lambda_max_R_Y=lam, top_eigenvector=vector, B=bb,
                        collision_threshold=1-1/bb, normalization_squared=nn,
                        scale_direction=direction, direction_sign=signs,
                        direction_correlation=cc,
                        direction_correlation_eigenvalues=np.linalg.eigvalsh(cc),
                        mu0=np.log(p_marg/(1-p_marg)), product_pmf=product,
                        product_KL=KL(p, product), exact_scale_curvature=(1-bb)/nn)
        for name, value in expected.items():
            errors.append(near(value, drow[name], label='derived '+name))
        tq = dict(teacher=record['name'], object='teacher', t=None,
                  PMF_change_40_minus_32=float(np.max(np.abs(p-p32))),
                  PMF_change_40_minus_24_descriptive=float(np.max(np.abs(p-variants[0]['pmf']))),
                  mass_passed=all(v['mass_error'] <= 2e-12 for v in variants))
        tq['passed'] = bool(tq['PMF_change_40_minus_32'] <= 2e-8 and tq['mass_passed'])
        expected_qualifications.append(tq)
        for si, candidate in enumerate(record['candidates']):
            require(candidate['t'] == spec['t_values'][si], 'candidate t order')
            q = np.asarray(candidate['pmf']); risk = KL(p, q)
            errors.append(near(risk, candidate['joint_KL'], label='reference risk'))
            near(risk-KL(p, product), candidate['KL_minus_product'], label='reference risk contrast')
            if si == 0:
                near(q, product, label='zero-scale product')
                require(candidate['Gaussian_nodes'] == 0 and candidate['variants'] == [], 'zero analytic')
                continue
            cv = candidate['variants']; require([x['order'] for x in cv] == [24,32,40], 'candidate orders')
            near(q, cv[2]['pmf'], 0., 'candidate main')
            near(candidate['scale'], spec['t_values'][si]*direction, label='candidate scales')
            for vv in cv:
                raw = np.asarray(vv['raw_pmf']); mass = float(math.fsum(raw))
                near(mass, vv['raw_mass'], label='candidate mass')
                near(raw/mass, vv['pmf'], label='candidate normalization')
                near(abs(mass-1), vv['mass_error'], label='candidate mass error')
                require((raw > 0).all() and vv['Gaussian_nodes'] == vv['order']**4, 'candidate nodes/positive')
            q32 = np.asarray(cv[1]['pmf'])
            qq = dict(teacher=record['name'], object='candidate', t=candidate['t'],
                      PMF_change_40_minus_32=float(np.max(np.abs(q-q32))),
                      PMF_change_40_minus_24_descriptive=float(np.max(np.abs(q-cv[0]['pmf']))),
                      KL_change_candidate_only=abs(risk-KL(p,q32)),
                      KL_change_teacher_and_candidate=abs(risk-KL(p32,q32)),
                      KL_contrast_change_teacher_and_candidate=abs((risk-KL(p,product))-(KL(p32,q32)-KL(p32,product))),
                      mass_passed=all(v['mass_error'] <= 2e-12 for v in cv))
            qq['passed'] = bool(qq['PMF_change_40_minus_32'] <= 2e-8 and qq['mass_passed'] and
                                max(qq[k] for k in ('KL_change_candidate_only','KL_change_teacher_and_candidate',
                                                  'KL_contrast_change_teacher_and_candidate')) <= 2e-7)
            expected_qualifications.append(qq)
    require(len(ref['records']) == 4 and len(ref['qualifications']) == 16, 'reference scope')
    for got, expected in zip(ref['qualifications'], expected_qualifications):
        require(set(got) == set(expected), 'qualification fields')
        for key, value in expected.items():
            if isinstance(value, float): near(got[key], value, label='qualification '+key)
            else: require(got[key] == value, 'qualification '+key)
    require(ref['qualification_passed'] == all(x['passed'] for x in expected_qualifications), 'reference qualification aggregate')
    expected_accounting = dict(PMF_integrations=48, scalar_pattern_integrals=768,
                              Gaussian_nodes=16*sum(o**4 for o in (24,32,40)),
                              analytic_product_PMFS=4, new_random_draws=0,
                              new_observations=0, new_fits=0)
    require(ref['accounting'] == expected_accounting, 'reference accounting')
    return dict(passed=True, maximum_arithmetic_error=max(errors), teacher_count=4,
                candidate_count=16, qualifications=16, new_PMFS=0)


def independent_reference(ref):
    rows = []
    for record in ref['records']:
        d = record['derived']; p_main = np.asarray(record['teacher_pmf'])
        teacher = normal_hermite_pmf([-.6,-.2,.2,.6], record['covariance'])
        p_audit = np.asarray(teacher['pmf'])
        teacher['PMF_error'] = near(p_audit, p_main, TOL['reference_pmf'], 'independent teacher PMF')
        candidates = []
        base = np.asarray(d['product_pmf'])
        for candidate in record['candidates'][1:]:
            scale = candidate['t']*np.asarray(d['scale_direction'])
            covariance = np.asarray(d['direction_correlation'])*np.outer(scale,scale)
            rule = normal_hermite_pmf(d['mu0'], covariance)
            q_audit = np.asarray(rule['pmf']); q_main = np.asarray(candidate['pmf'])
            rule['t'] = candidate['t']
            rule['PMF_error'] = near(q_audit,q_main,TOL['reference_pmf'],'independent candidate PMF')
            rule['candidate_only_KL_error'] = near(KL(p_main,q_audit), candidate['joint_KL'],
                                                   TOL['reference_KL'],'candidate-only KL')
            rule['teacher_and_candidate_KL_error'] = near(KL(p_audit,q_audit), candidate['joint_KL'],
                                                          TOL['reference_KL'],'full audit KL')
            rule['contrast_KL_error'] = near(KL(p_audit,q_audit)-KL(p_audit,base),
                                             candidate['KL_minus_product'],TOL['reference_KL'],
                                             'full audit contrast')
            candidates.append(rule)
        rows.append(dict(teacher_name=record['name'], teacher=teacher,candidates=candidates))
    return dict(passed=True, records=rows, PMF_integrations=16,
                Gaussian_nodes=16*36**4, scalar_pattern_integrals=16*16,
                numerical_qualification_not_rigorous_error_bound=True)


def stream_audit(stream, ref):
    folder=BASE/'MC'/f'stream_{stream:02d}'; meta=read(folder/'RESULTS.json')
    npzpath=folder/'BANKS.npz'
    require(meta['stream']==stream and meta['npz_sha256']==sha(npzpath),'stream metadata/hash')
    require(meta['source_sha256']==sha(ROOT/'script/activevar_compute.py') and
            meta['freeze_sha256']==sha(BASE/'FREEZE.json') and
            meta['reference_sha256']==sha(BASE/'REFERENCE/RESULTS.json'),'stream binding')
    require(meta['seed']==dict(entropy=202610044000+stream,bit_generator='PCG64DXSM',
                              nuisance_spawn_key=[0],active_spawn_key=[1]),'stream seed')
    with np.load(npzpath,allow_pickle=False) as saved:
        require(set(saved.files)=={'contrasts','collisions','prefix_active','prefix_nuisance'},'NPZ fields')
        values={key:saved[key].copy() for key in saved.files}
    require(values['contrasts'].shape==(2048,4,4,5) and values['collisions'].shape==(2048,5),'NPZ axes')
    for key in ('contrasts','collisions'):
        require(array_sha(values[key])==meta['array_sha256'][key],'saved array hash '+key)
        require(np.isfinite(values[key]).all(),'saved finite '+key)
    require(meta['contrast_axis_order']==['bank','teacher','t','nuisance'] and
            meta['collision_axis_order']==['bank','nuisance'],'axes metadata')
    require(meta['teacher_order']==[r['name'] for r in ref['records']] and
            meta['t_values']==[0.,.1,.2,.4] and meta['nuisance_values']==list(N_VALUES),'panel metadata')
    rng={name:np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
         202610044000+stream,spawn_key=(role,)))) for name,role in [('nuisance',0),('active',1)]}
    hashes={name:hashlib.sha256() for name in rng}; maximum={name:0. for name in rng}; prefix={}
    for block in range(64):
        for name,dimensions in [('nuisance',256),('active',4)]:
            array=rng[name].standard_normal((32,100,dimensions),dtype=np.float64)
            hashes[name].update(array.tobytes(order='C'))
            maximum[name]=max(maximum[name],float(np.abs(array).max()))
            if block==0:prefix[name]=array[:2].copy()
    require(meta['input_sha256']=={name+'_C_order_float64':value.hexdigest() for name,value in hashes.items()},
            'complete input replay hash')
    for name in rng:
        require(np.array_equal(prefix[name],values['prefix_'+name]),'prefix '+name)
    near(maximum['active'],meta['maximum_absolute_active_base_normal'],0.,'active maximum')
    near(maximum['nuisance'],meta['maximum_absolute_nuisance_normal'],0.,'noise maximum')
    near(values['contrasts'].mean(axis=0),meta['mean_contrasts'],label='stream contrast mean')
    near(values['collisions'].mean(axis=0),meta['mean_collisions'],label='stream collision mean')
    near(values['contrasts'][:,:,0,:],np.zeros((2048,4,5)),0.,'zero t contrasts')
    near(values['collisions'][:,0],np.full(2048,.01),0.,'n0 analytic collision')
    require(np.all(values['collisions']>=.01-1e-14) and np.all(values['collisions']<=1+1e-14),'collision bounded')
    expected_cost=dict(banks=2048,prior_vectors=204800,active_Gaussian=819200,
                       nuisance_Gaussian=52428800,total_Gaussian=53248000,
                       canonical_patterns_not_sampled=16,
                       active_likelihood_ratios=2048*100*4*3*16,
                       weighted_log_score_cells=2048*4*3*5*16,
                       new_observations=0,fits=0,NN_training=0,native_API_calls=0)
    require(meta['accounting']==expected_cost,'stream cost')
    scipy_error=collision_error=0.; native_rows=[]; native_calls=0
    zero_count=one_count=unique_entries=0
    for ni,n in enumerate(N_VALUES):
        noise=prefix['nuisance'][:,:,:n]
        logweights=-np.logaddexp(0.,-noise).sum(axis=2)
        weights=np.exp(logweights-logsumexp(logweights,axis=1)[:,None])
        collision_error=max(collision_error,near(np.sum(weights**2,axis=1),
                  values['collisions'][:2,ni],TOL['prefix_score'],'prefix collision'))
        for ti,teacher in enumerate(ref['records']):
            d=teacher['derived']; mu=np.asarray(d['mu0']); direction=np.asarray(d['scale_direction'])
            transformed=prefix['active']@np.linalg.cholesky(np.asarray(d['direction_correlation'])).T
            base=np.broadcast_to(mu,(2,100,4))
            base_loss=pattern_losses(base,noise)
            for si,tvalue in enumerate((0.,.1,.2,.4)):
                logits=mu+tvalue*direction*transformed
                contrast=(pattern_losses(logits,noise)-base_loss)@np.asarray(teacher['teacher_pmf'])
                scipy_error=max(scipy_error,near(contrast,values['contrasts'][:2,ti,si,ni],
                                  TOL['prefix_score'],'prefix independent log score'))
            if n in (0,256):
                losses=[]; counters=[]
                for tvalue in (0.,.2):
                    logits=mu+tvalue*direction*transformed[0]
                    scored,count=native_pattern_losses(logits,noise[0]);losses.append(scored);counters.append(count)
                    native_calls+=count['calls'];zero_count+=count['sigmoid_zero'];one_count+=count['sigmoid_one']
                    unique_entries+=count['unique_sigmoid_entries']
                mathematical=pattern_losses((mu+.2*direction*transformed[0])[None],noise[:1])[0]-base_loss[0]
                observed=losses[1]-losses[0]
                err=near(observed,mathematical,TOL['native_contrast'],'native scalar pattern contrast')
                weighted=float(observed@np.asarray(teacher['teacher_pmf']))
                weighted_error=near(weighted,values['contrasts'][0,ti,2,ni],TOL['native_contrast'],'native weighted contrast')
                native_rows.append(dict(teacher=teacher['name'],nuisance=n,t=.2,
                    pattern_contrasts_native32=observed.tolist(),pattern_contrasts_math64=mathematical.tolist(),
                    weighted_native32=weighted,weighted_math64=float(values['contrasts'][0,ti,2,ni]),
                    maximum_pattern_error=err,weighted_error=weighted_error,probability_controls=counters))
    require(native_calls==256 and len(native_rows)==8,'native stream count')
    return dict(stream=stream,passed=True,all_input_hash_replay=True,
                maximum_prefix_score_error=scipy_error,maximum_prefix_collision_error=collision_error,
                native_rows=native_rows,official_API_calls=native_calls,
                native_pattern_contrasts=128,native_weighted_contrasts=8,
                sigmoid_zero=zero_count,sigmoid_one=one_count,
                unique_native_probability_entries=unique_entries,
                replayed_Gaussian_scalars=53248000,new_independent_draws=0),values['contrasts'].mean(axis=0),values['collisions']


def statistics_audit(ref, stream_means, all_collisions):
    result=read(BASE/'SUMMARY/RESULTS.json')
    require(result['source_sha256']==sha(ROOT/'script/activevar_compute.py') and
            result['freeze_sha256']==sha(BASE/'FREEZE.json') and
            result['reference_sha256']==sha(BASE/'REFERENCE/RESULTS.json'),'summary binding')
    require(len(result['streams'])==16,'summary streams')
    for stream,row in enumerate(result['streams']):
        folder=BASE/'MC'/f'stream_{stream:02d}'
        require(row==dict(stream=stream,metadata_sha256=sha(folder/'RESULTS.json'),
                         npz_sha256=sha(folder/'BANKS.npz')),'summary stream binding')
    primary=result['raw_primary_family'];collision=result['collision_family']
    require(primary['q']==8 and primary['alpha']==.025 and primary['df']==15 and
            primary['max_SE']==.00015,'primary contract')
    require(collision['q']==4 and collision['alpha']==.025 and
            collision['independent_banks']==32768,'collision contract')
    critical=float(student_t.isf(.025/16,15))
    near(critical,primary['critical_t'],label='t critical')
    half=.99*math.sqrt(math.log(320.)/65536)
    near(half,collision['halfwidth_before_clipping'],label='Hoeffding halfwidth')
    errors=[]
    for ni,n in enumerate(N_VALUES):
        row=collision['results'][ni];require(row['nuisance']==n and row['exact']==(n==0),'collision identity')
        mean=.01 if n==0 else math.fsum(all_collisions[:,ni])/32768
        lo=mean if n==0 else max(.01,mean-half)
        hi=mean if n==0 else min(1.,mean+half)
        for key,value in [('mean',mean),('lower',lo),('upper',hi),('halfwidth_before_clipping',0. if n==0 else half)]:
            errors.append(near(value,row[key],label='collision '+key))
        require(len(row['curvatures'])==4,'curvature count')
        for ti,teacher in enumerate(ref['records']):
            d=teacher['derived'];b=d['B'];norm=d['normalization_squared'];cc=row['curvatures'][ti]
            require(cc['teacher']==teacher['name'],'curvature teacher')
            for key,x in [('mean',mean),('lower',lo),('upper',hi)]:
                near((1-(1-x)*b)/norm,cc[key],label='derived curvature '+key)
            near((1-b)/norm,cc['exact_risk_curvature'],label='risk curvature')
            require(cc['raw_uniform_C_strict_min_qualified']==((1-lo)*b<1), 'raw min flag')
            require(cc['raw_constructed_descent_qualified']==((1-hi)*b>1), 'raw descent flag')
    mapped={}
    for row in primary['results']+result['descriptive_curves']:
        ident=(row['teacher'],row['t'],row['nuisance']);require(ident not in mapped,'duplicate summary row');mapped[ident]=row
    require(len(mapped)==80 and len(primary['results'])==8 and len(result['descriptive_curves'])==72,'summary count')
    effects=precision=0
    for ti,teacher in enumerate(ref['records']):
        for si,tvalue in enumerate((0.,.1,.2,.4)):
            for ni,n in enumerate(N_VALUES):
                row=mapped[teacher['name'],tvalue,n]
                vv=np.asarray(stream_means)[:,ti,si,ni];mean=math.fsum(vv)/16
                se=math.sqrt(math.fsum(float((x-mean)**2) for x in vv)/15/16)
                for key,value in [('stream_means',vv),('raw_score_contrast',mean),
                                  ('SE_across_16_stream_means',se),
                                  ('reference_KL_contrast',teacher['candidates'][si]['KL_minus_product'])]:
                    errors.append(near(value,row[key],label='summary '+key))
                if tvalue==.2 and n in (0,256):
                    lo=mean-critical*se;hi=mean+critical*se
                    near(lo,row['lower'],label='primary lower');near(hi,row['upper'],label='primary upper')
                    require(row['precision_passed']==(se<=.00015) and row['max_SE']==.00015,'precision gate')
                    precision+=int(row['precision_passed'])
                    if ti==0:require(row['direction_gate'] is None and row['direction_passed'] is None,'null not gated')
                    else:
                        expected=hi<0 if n==0 else lo>0
                        require(row['direction_gate']==('negative' if n==0 else 'positive') and
                                row['direction_passed']==expected,'direction gate')
                        effects+=int(expected)
    require(primary['direction_passed']==effects and primary['direction_total']==6 and
            primary['precision_passed']==precision and primary['precision_total']==8,'gate counts')
    require(result['accounting']==dict(new_prior_banks=32768,prior_vectors=3276800,
            active_Gaussian=13107200,nuisance_Gaussian=838860800,total_Gaussian=851968000,
            new_observations=0,new_fits=0,NN_training=0,native_API_calls=0),'summary costs')
    return dict(passed=True, maximum_arithmetic_error=max(errors),
                primary_direction_passed=effects,primary_direction_total=6,
                primary_precision_passed=precision,primary_precision_total=8,
                collision_intervals=4,analytic_collision_baselines=1,
                primary_interval_count=8,descriptive_rows=72,
                independent_streams=16,independent_collision_banks=32768,
                scope='All saved arrays and statistical arithmetic; not all bottom-level raw scores.')


def run_audit():
    plan=bind_audit(); started=time.perf_counter()
    save(OUT/'EXECUTION.json',dict(start_utc=now(),source_sha256=sha(SOURCE),
          protocol_sha256=sha(OUT/'PROTOCOL.json'),root_freeze_sha256=sha(BASE/'FREEZE.json'),
          automatic_retry=False))
    spec=read(BASE/'SPEC.json');ref=read(BASE/'REFERENCE/RESULTS.json')
    deterministic=hessian_control()
    arithmetic=reference_arithmetic(ref,spec)
    save(OUT/'REFERENCE_ARITHMETIC.json',arithmetic)
    streams=[];means=[];collisions=[]
    for stream in range(16):
        tick=time.perf_counter();row,average,cc=stream_audit(stream,ref)
        row['seconds']=time.perf_counter()-tick
        save(OUT/f'stream_{stream:02d}.json',row)
        streams.append(row);means.append(average);collisions.append(cc)
        print(json.dumps(dict(audit_stream=stream,passed=True,seconds=row['seconds'],
                    maximum_native_pattern_error=max(x['maximum_pattern_error'] for x in row['native_rows']))),flush=True)
    stats=statistics_audit(ref,np.asarray(means),np.concatenate(collisions))
    save(OUT/'STATISTICS.json',stats)
    reference=independent_reference(ref)
    save(OUT/'REFERENCE_INDEPENDENT.json',reference)
    native_calls=sum(x['official_API_calls'] for x in streams)
    require(native_calls==4096,'all native calls')
    result=dict(passed=True,source_sha256=sha(SOURCE),
                protocol_sha256=sha(OUT/'PROTOCOL.json'),
                root_freeze_sha256=sha(BASE/'FREEZE.json'),
                summary_sha256=sha(BASE/'SUMMARY/RESULTS.json'),
                reference_sha256=sha(BASE/'REFERENCE/RESULTS.json'),
                deterministic=deterministic,statistics=stats,
                receipts={rel(OUT/name):sha(OUT/name) for name in
                    ['REFERENCE_ARITHMETIC.json','STATISTICS.json','REFERENCE_INDEPENDENT.json']+
                    [f'stream_{j:02d}.json' for j in range(16)]},
                maxima=dict(prefix_score=max(x['maximum_prefix_score_error'] for x in streams),
                            prefix_collision=max(x['maximum_prefix_collision_error'] for x in streams),
                            native_pattern_contrast=max(y['maximum_pattern_error'] for x in streams for y in x['native_rows']),
                            native_weighted_contrast=max(y['weighted_error'] for x in streams for y in x['native_rows']),
                            reference_PMF=max(max(x['teacher']['PMF_error'],*(y['PMF_error'] for y in x['candidates'])) for x in reference['records']),
                            reference_KL=max(max(y[k] for k in ('candidate_only_KL_error','teacher_and_candidate_KL_error','contrast_KL_error'))
                                             for x in reference['records'] for y in x['candidates'])),
                accounting=dict(replayed_Gaussian_scalars=sum(x['replayed_Gaussian_scalars'] for x in streams),
                    original_inputs_reused=True,new_independent_scientific_draws=0,new_observations=0,
                    parameter_fits=0,new_neural_network_fits=0,
                    original_API_calls=native_calls,native_scalar_pattern_contrasts=2048,
                    native_teacher_weighted_contrasts=128,
                    independently_scored_prefix_banks=32,
                    independent_positive_t_prefix_contrasts=32*4*3*5,
                    independent_prefix_pattern_contrasts=32*4*3*5*16,
                    audit_PMF_integrations=16,audit_Gaussian_nodes=16*36**4,
                    audit_scalar_pattern_integrals=256,
                    sigmoid_zero=sum(x['sigmoid_zero'] for x in streams),
                    sigmoid_one=sum(x['sigmoid_one'] for x in streams),
                    unique_native_probability_entries=sum(x['unique_native_probability_entries'] for x in streams)),
                seconds=time.perf_counter()-started,
                scope='Full input hash replay/all saved arithmetic; bottom scores only first2banks/stream; native firstbank n0/n256 t0/.2;16 additional GH36 PMFs. Not complete main-score rerun or native32 certificate.')
    save(OUT/'RESULTS.json',result)
    print(json.dumps(dict(audit_passed=True,seconds=result['seconds'],maxima=result['maxima'],
                         accounting=result['accounting'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','freeze','run'])
    args=parser.parse_args()
    if args.mode=='check':
        print(json.dumps(hessian_control()),flush=True)
    elif args.mode=='freeze':
        freeze_audit()
    else:
        try:
            run_audit()
        except BaseException:
            if not (OUT/'FAILURE.json').exists():
                save(OUT/'FAILURE.json',dict(utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise


