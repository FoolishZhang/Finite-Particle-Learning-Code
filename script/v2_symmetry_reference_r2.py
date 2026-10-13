"""R2: deterministic Fourier reference; scientific bank design unchanged from R."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json, math, os, platform, sys, time, traceback
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import scipy
from scipy.integrate import quad, quad_vec
from scipy.special import expit, log_expit, logsumexp, gammaln, roots_hermitenorm, ndtr, eval_hermitenorm
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/structure_tradeoff_v2/R2'
OLD = ROOT / 'artifacts/structure_tradeoff_v2/R'
DOC = ROOT / 'docs/08_research_研究工作/结构预算与对称化代价_v2_20260930'
BASE = 'ce1aa8340028dbdfafc42187876bfc44999e2813'
CFG = dict(d=[8, 16], profiles={'null': [4., 4.], 'mild': [4.5, 3.5], 'contrast': [6., 2.]}, c=.9,
           ref_variants=['GH256', 'GH512', 'adaptive'], ref_primary='adaptive', ref_GH_orders=[256, 512],
           inner_standard_limit=12., inner_epsabs=2e-13, inner_epsrel=2e-13, inner_limit=300,
           outer_standard_limit=9., outer_epsabs=2e-12, outer_epsrel=2e-12, outer_limit=4000,
           ref_variant_L1_gate=1e-8, ref_function_difference_gate=1e-5,
           outer_error_estimate_gate=1e-10, inner_propagated_error_estimate_gate=1e-9,
           moment_gate=1e-8, hermite_odd_orders=[1, 3, 5, 7, 9, 11, 13], hermite_standard_limit=12.,
           hermite_epsabs=2e-13, hermite_epsrel=2e-13, hermite_limit=300,
           K=[1, 10, 100], streams=8, banks=256, bank_batch=16, bank_seed=20291101,
           propagated_reference_gate=1e-4, mass_gate=1e-12, identity_gate=1e-10,
           raw_SE_gate=1., group_single_SE_gate=.01, paired_SE_gate=.005,
           alpha=.05, contrasts=72, material=.001, primary_d=16, primary_K=10,
           primary_profiles=['mild', 'contrast'], fits=0, observed_label_samples=0)
NAMES = ['raw_excess', 'group_excess', 'single_excess', 'single_minus_group']

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x, exclusive=True):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def csvwrite(p, rows):
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def verify(ok, message):
    if not ok: raise RuntimeError(message)

def freeze():
    assert not OUT.exists()
    oldstate = read(OLD / 'STATE.json')
    oldfailures = list(OLD.glob('FAILURE_*.json'))
    assert oldstate['status'] == 'stopped_failure' and len(oldfailures) == 1
    assert not list((OLD / 'local').glob('*.npz')) and not list((OLD / 'streams').glob('*.json'))
    for n, h in read(OLD / 'FREEZE.json')['files'].items(): assert sha(ROOT / n) == h, n
    OUT.mkdir(parents=True); (OUT / 'local').mkdir(); (OUT / 'source_inputs').mkdir()
    (OUT / '.gitignore').write_text('/local/\n', encoding='utf-8', newline='\n')
    (OUT / 'source_inputs/HANDOFF.md').write_bytes((OLD / 'source_inputs/HANDOFF.md').read_bytes())
    write(OUT / 'CONFIG.json', CFG)
    write(OUT / 'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()))
    oldpaths = [OLD / 'FREEZE.json', OLD / 'STATE.json', oldfailures[0], OLD / 'reference/null_d8.json',
                ROOT / 'script/v2_symmetry_reference.py', DOC / 'PLAN_R.md']
    write(OUT / 'ORIGINAL_R_RECEIPT.json', dict(time_utc=now(), original_status=oldstate['status'],
          original_bank_files=0, original_bank_scoring_started=False,
          numerical_algorithm_repair_only=True, old_failure_is_not_reclassified=True,
          files={p.relative_to(ROOT).as_posix(): sha(p) for p in oldpaths}))
    paths = [Path(__file__), DOC / 'PLAN_R2.md', DOC / 'NUMERICAL_REFERENCE.md', OUT / 'CONFIG.json',
             OUT / 'ENVIRONMENT.json', OUT / 'ORIGINAL_R_RECEIPT.json', OUT / 'source_inputs/HANDOFF.md',
             ROOT / 'docs/08_research_研究工作/非交换适用域_v2_20260930/THEORY.md',
             ROOT / 'artifacts/nonexchange_v2/MANIFEST.json'] + oldpaths
    write(OUT / 'FREEZE.json', dict(time_utc=now(), base=BASE, files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}))
    write(OUT / 'STATE.json', dict(status='frozen', time_utc=now(), active_jobs=[]))
    print(json.dumps({'freeze': sha(OUT / 'FREEZE.json'), 'original_R_freeze': sha(OLD / 'FREEZE.json')}), flush=True)

def checkfreeze():
    assert read(OUT / 'CONFIG.json') == CFG
    for n, h in read(OUT / 'FREEZE.json')['files'].items(): assert sha(ROOT / n) == h, n

def geometry(d):
    m = d // 2
    a, b = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing='ij'); a = a.ravel(); b = b.ravel(); n = a + b
    lg = (gammaln(m + 1) - gammaln(a + 1) - gammaln(m - a + 1) + gammaln(m + 1) - gammaln(b + 1) - gammaln(m - b + 1))
    lt = gammaln(d + 1) - gammaln(n + 1) - gammaln(d - n + 1)
    labels = np.concatenate([(np.arange(m)[None, :] < a[:, None]), (np.arange(m)[None, :] < b[:, None])], axis=1).astype(float)
    return a, b, n, lg, lt, labels, np.eye(d + 1)[n]

def eta_from_z(z, scales):
    d = z.shape[-1]; rho = CFG['c'] / (d - 1)
    u = np.sqrt(1 + rho) * z - (np.sqrt(1 + rho) - np.sqrt(1 - CFG['c'])) * z.mean(-1, keepdims=True)
    return u * np.repeat(scales, d // 2)

def pb(eta):
    p = expit(eta); q = expit(-eta); mass = np.zeros((len(eta), eta.shape[1] + 1)); mass[:, 0] = 1
    for j in range(eta.shape[1]):
        mass[:, 1:j + 2] = mass[:, 1:j + 2] * q[:, j, None] + mass[:, :j + 1] * p[:, j, None]
        mass[:, 0] *= q[:, j]
    return mass

def joint_mass(eta):
    m = eta.shape[-1] // 2; aa = pb(eta[:, :m]); bb = pb(eta[:, m:])
    return (aa[:, :, None] * bb[:, None, :]).reshape(len(eta), -1)

def functionals(p, d):
    _, _, n, lg, lt, _, ind = geometry(d); p = np.asarray(p); verify(np.all(p > 0), 'nonpositive reference mass')
    pn = p @ ind; ps = pn[n] * np.exp(lg - lt)
    return -float(p @ (np.log(p) - lg)), float(p @ np.log(p / ps)), pn, ps

def smoke():
    errors = []
    for d in [8, 16]:
        rho = CFG['c'] / (d - 1); a = np.sqrt(1 + rho); b = a - np.sqrt(1 - CFG['c'])
        factor = a * np.eye(d) - b * np.ones((d, d)) / d
        target = (1 + rho) * np.eye(d) - rho * np.ones((d, d))
        errors.append(float(np.max(abs(factor @ factor.T - target))))
        invformula = np.eye(d) / (1 + rho) + rho * np.ones((d, d)) / ((1 + rho) * (1 - CFG['c']))
        errors.append(float(np.max(abs(invformula @ target - np.eye(d)))))
    eta = np.array([[-2.7, -.3, .8, 1.2], [.2, .2, .2, .2]])
    labels = np.array(list(itertools.product([0., 1.], repeat=4))); a = labels[:, :2].sum(1).astype(int); b = labels[:, 2:].sum(1).astype(int)
    weights = np.exp(log_expit((2 * labels[None, :, :] - 1) * eta[:, None, :]).sum(-1))
    joint = joint_mass(eta); whole = pb(eta); _, _, _, _, _, _, ind = geometry(4)
    for j in range(2): errors.append(float(np.max(abs(np.bincount(3 * a + b, weights=weights[j], minlength=9) - joint[j]))))
    errors.append(float(np.max(abs(joint @ ind - whole))))
    H, D, _, _ = functionals(np.bincount(3 * a + b, minlength=9) / 16, 4); errors.extend([abs(H - 4 * math.log(2)), abs(D)])
    verify(max(errors) < 1e-12, 'deterministic probability smoke failure')
    write(OUT / 'SMOKE.json', dict(passed=True, maximum_error=max(errors), enumerated_labels=16, new_scientific_observations=0))

def pairwise_max(values, l1=False):
    values = np.asarray(values)
    return max(float(np.abs(values[i] - values[j]).sum() if l1 else np.max(np.abs(values[i] - values[j])))
               for i in range(len(values)) for j in range(i))

HERMITE_CACHE = {}
def independent_moment_reference(scales, d):
    """Mehler expansion independent of the Fourier imaginary-part implementation."""
    for scale in scales:
        if scale not in HERMITE_CACHE:
            coefficients = []; errors = []; neval = 0
            for order in CFG['hermite_odd_orders']:
                norm = math.sqrt(math.factorial(order)) * math.sqrt(2 * math.pi)
                def f(z): return math.tanh(scale * z / 2) * float(eval_hermitenorm(order, z)) * math.exp(-z * z / 2) / norm
                ans = quad(f, 0., CFG['hermite_standard_limit'], epsabs=CFG['hermite_epsabs'], epsrel=CFG['hermite_epsrel'],
                           limit=CFG['hermite_limit'], full_output=1)
                verify(len(ans) == 3, 'independent Hermite quadrature convergence failure')
                coefficients.append(2 * ans[0]); errors.append(2 * ans[1]); neval += int(ans[2]['neval'])
            HERMITE_CACHE[scale] = dict(scale=scale, coefficients=coefficients, quadrature_error_estimates=errors, evaluations=neval)
    rho = CFG['c'] / (d - 1); m = d // 2
    powers = rho ** np.array(CFG['hermite_odd_orders']); tail = rho ** 15
    coefftail = math.sqrt(2 * float(ndtr(-CFG['hermite_standard_limit'])))
    pairs = []; pairerrors = []
    for s, tscale in [(scales[0], scales[0]), (scales[1], scales[1]), (scales[0], scales[1])]:
        aa = np.array(HERMITE_CACHE[s]['coefficients']); bb = np.array(HERMITE_CACHE[tscale]['coefficients'])
        ea = np.array(HERMITE_CACHE[s]['quadrature_error_estimates']) + coefftail
        eb = np.array(HERMITE_CACHE[tscale]['quadrature_error_estimates']) + coefftail
        pairs.append(float((aa * bb) @ powers))
        pairerrors.append(float((abs(aa) * eb + abs(bb) * ea + ea * eb) @ powers + tail))
    raa, rbb, rab = pairs
    expected = np.array([m - m * (m - 1) * raa, m - m * (m - 1) * rbb, -m * m * rab])
    errors = np.array(pairerrors) * [m * (m - 1), m * (m - 1), m * m]
    return dict(expected_spin_moments=expected.tolist(), pair_magnitudes_AA_BB_AB=pairs,
                pair_error_estimates_plus_analytic_tail=pairerrors, moment_error_estimates_plus_tail=errors.tolist(),
                mehler_analytic_remainder_bound=tail, coefficient_analytic_tail_bound=coefftail,
                coefficient_sources=[HERMITE_CACHE[s] for s in dict.fromkeys(scales)],
                quadrature_estimates_not_rigorous_certificates=True)

def reference(profile, d, idx):
    start = time.perf_counter(); scales = CFG['profiles'][profile]
    rho = CFG['c'] / (d - 1); a = 1 + rho; tau = rho / (a * (1 - CFG['c'])); normalizer = math.sqrt(a / (1 - CFG['c']))
    na, nb, _, lg, _, _, _ = geometry(d); m = d // 2; multiplicity = np.exp(lg)
    inner_tail_u = float(ndtr(-CFG['inner_standard_limit']))
    outer_L1_tail = normalizer * 2 * float(ndtr(-CFG['outer_standard_limit']))
    cache = {}; stats = dict(inner_calls=0, inner_evaluations=0, inner_max_error_estimate=0.,
                            max_factor_abs=0., max_integrand_mass_identity_error=0.)
    invsqrt2pi = 1 / math.sqrt(2 * math.pi)
    def uvalue(scale, w):
        key = (float(scale), float(abs(w)))
        if key not in cache:
            ws = key[1]
            def f(x): return math.tanh(scale * math.sqrt(a) * x / 2) * math.sin(ws * math.sqrt(a) * x) * math.exp(-x * x / 2) * invsqrt2pi
            ans = quad(f, 0., CFG['inner_standard_limit'], epsabs=CFG['inner_epsabs'], epsrel=CFG['inner_epsrel'], limit=CFG['inner_limit'], full_output=1)
            verify(len(ans) == 3, 'inner quad convergence failure: ' + (str(ans[3]) if len(ans) > 3 else ''))
            integral, error, info = ans
            # H = 2*integral, while Im(u) = H/2; error here is the u coefficient estimate.
            cache[key] = complex(.5 * math.exp(-a * ws * ws / 2), integral)
            stats['inner_calls'] += 1; stats['inner_evaluations'] += int(info['neval'])
            stats['inner_max_error_estimate'] = max(stats['inner_max_error_estimate'], float(error))
        u = cache[key] if w >= 0 else cache[key].conjugate()
        stats['max_factor_abs'] = max(stats['max_factor_abs'], abs(u))
        return u
    def integrand(t0):
        w = math.sqrt(tau) * t0; ua = uvalue(scales[0], w); ub = uvalue(scales[1], w)
        value = normalizer * multiplicity * ua ** na * ua.conjugate() ** (m - na) * ub ** nb * ub.conjugate() ** (m - nb)
        expected = normalizer * math.exp(-a * d * w * w / 2)
        stats['max_integrand_mass_identity_error'] = max(stats['max_integrand_mass_identity_error'], float(abs(value.sum() - expected)))
        return value
    variants = []; checks = []
    for order in CFG['ref_GH_orders']:
        nodes, weights = roots_hermitenorm(order); weights = weights / math.sqrt(2 * math.pi)
        raw = np.zeros(len(na), dtype=complex)
        for node, weight in zip(nodes, weights):
            if weight != 0: raw += weight * integrand(float(node))
        variants.append(raw.real)
        checks.append(dict(name='GH' + str(order), outer_nodes=order, maximum_imaginary_residual=float(np.max(abs(raw.imag))),
                           raw_mass_error=abs(float(raw.real.sum()) - 1), raw_min_probability=float(raw.real.min())))
    def density_integrand(t0): return integrand(float(t0)).real * math.exp(-t0 * t0 / 2) * invsqrt2pi
    qvalue, qerror, qinfo = quad_vec(density_integrand, -CFG['outer_standard_limit'], CFG['outer_standard_limit'],
                                    epsabs=CFG['outer_epsabs'], epsrel=CFG['outer_epsrel'], norm='max',
                                    limit=CFG['outer_limit'], quadrature='gk21', full_output=True)
    variants.append(qvalue)
    checks.append(dict(name='adaptive', outer_evaluations=int(qinfo.neval), status=int(qinfo.status), success=bool(qinfo.success),
                       message=qinfo.message, error_estimate=float(qerror), error_norm='PMF max norm',
                       L1_error_estimate=float(qerror) * len(na), raw_mass_error=abs(float(qvalue.sum()) - 1),
                       raw_min_probability=float(qvalue.min())))
    rawvariants = np.array(variants)
    validmass = all(x['raw_mass_error'] < CFG['mass_gate'] for x in checks)
    positive = bool(np.all(rawvariants > 0))
    finite = bool(np.isfinite(rawvariants).all())
    delta_est = stats['inner_max_error_estimate'] + inner_tail_u
    inner_L1_est = normalizer * 2 * d * delta_est * (1 + 2 * delta_est) ** (d - 1)
    approx_outer_L1_est = outer_L1_tail * (1 + 2 * delta_est) ** d
    # No clipping or normalization of an unqualified mass; write its diagnostic before stopping.
    result = dict(profile=profile, d=d, scales=scales, primary=CFG['ref_primary'], checks=checks, stats=stats,
                  raw_variants=rawvariants.tolist(), variant_names=CFG['ref_variants'],
                  inner_u_analytic_tail_bound=inner_tail_u, outer_exact_L1_analytic_tail_bound=outer_L1_tail,
                  inner_L1_error_estimate_plus_tail=inner_L1_est, approximate_outer_L1_tail_estimate=approx_outer_L1_est,
                  probability_finite=finite, raw_mass_pass=validmass, probability_strict_positive=positive,
                  deterministic_variants_are_not_independent_replicates=True, reference_standard_error=None,
                  numerical_qualification_not_rigorous_all_error_certificate=True,
                  complex_cancellation_not_positive_product_mixture=True)
    moment_ref = independent_moment_reference(scales, d)
    spinA = 2 * na - m; spinB = 2 * nb - m
    moment_weights = np.stack([spinA ** 2, spinB ** 2, spinA * spinB], axis=1)
    result['independent_Hermite_moments'] = moment_ref
    prelim = validmass and positive and finite
    if prelim:
        probs = rawvariants / rawvariants.sum(1, keepdims=True)  # recorded roundoff-only normalization
        funcs = np.array([functionals(p, d)[:2] for p in probs]); p = probs[2]
        H, D, pn, ps = functionals(p, d)
        l1 = pairwise_max(probs, True); fdiff = pairwise_max(funcs)
        moments = probs @ moment_weights
        moment_errors = np.abs(moments - np.array(moment_ref['expected_spin_moments']))
        max_moment_error = float(moment_errors.max())
        result.update(variants=probs.tolist(), mean=p.tolist(), H=H, model_KL_single=D, total_p=pn.tolist(),
                      symmetrized_group_p=ps.tolist(), variant_functionals=funcs.tolist(), pairwise_L1=l1,
                      pairwise_max_function_difference=fdiff, variant_spin_moments=moments.tolist(),
                      independent_moment_abs_errors=moment_errors.tolist(), independent_moment_max_error=max_moment_error)
        ok = (l1 <= CFG['ref_variant_L1_gate'] and fdiff <= CFG['ref_function_difference_gate'] and qinfo.success
              and qerror <= CFG['outer_error_estimate_gate'] and inner_L1_est <= CFG['inner_propagated_error_estimate_gate']
              and max(x.get('maximum_imaginary_residual', 0.) for x in checks) < CFG['mass_gate']
              and stats['max_integrand_mass_identity_error'] < CFG['identity_gate']
              and max_moment_error <= CFG['moment_gate']
              and max(moment_ref['moment_error_estimates_plus_tail']) <= CFG['moment_gate'])
    else: ok = False
    result.update(numerical_pass=bool(ok), elapsed_seconds=time.perf_counter() - start)
    write(OUT / 'reference' / f'{profile}_d{d}.json', result)
    print(json.dumps({'phase': 'reference', 'profile': profile, 'd': d, 'L1': result.get('pairwise_L1'),
                      'function_diff': result.get('pairwise_max_function_difference'), 'pass': bool(ok)}), flush=True)
    verify(ok, 'R2 reference numerical gate failed; stop without changing orders, nodes, or rules')

def scores_from_logs(logs, p, d):
    H, _, _, _ = functionals(p, d); v = np.einsum('...qm,m->...q', logs, p) - H
    return np.concatenate([v, (v[..., 2] - v[..., 1])[..., None]], axis=-1)

def bank_stream(profile, d, idx, stream):
    ref = read(OUT / 'reference' / f'{profile}_d{d}.json'); verify(ref['numerical_pass'], 'bad reference')
    p = np.array(ref['mean']); refs = np.array(ref['variants'])
    H, D, pn, _ = functionals(p, d); a, b, n, lg, lt, labels, indicator = geometry(d)
    seed = CFG['bank_seed'] + 10000 * d + 100 * idx + stream; rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.bit_generator.state; out = []; logmeans = np.zeros((3, 3, len(p))); totalrisk = []
    maxidentity = 0.; normerror = 0.; formulaerror = 0.; start = time.perf_counter()
    for offset in range(0, CFG['banks'], CFG['bank_batch']):
        nb = CFG['bank_batch']; eta = eta_from_z(rng.standard_normal((nb * 100, d)), CFG['profiles'][profile])
        j = joint_mass(eta); pnt = pb(eta)
        maxidentity = max(maxidentity, float(np.max(abs(j @ indicator - pnt))))
        normerror = max(normerror, float(np.max(abs(j.sum(1) - 1))), float(np.max(abs(pnt.sum(1) - 1))))
        rawlog = eta @ labels.T - np.logaddexp(0., eta).sum(1)[:, None]
        if offset == 0:
            other = log_expit((2 * labels[None, :, :] - 1) * eta[:4, None, :]).sum(-1)
            formulaerror = max(formulaerror, float(np.max(abs(other - rawlog[:4]))))
        rawlog = rawlog.reshape(nb, 100, -1); j = j.reshape(nb, 100, -1); pnt = pnt.reshape(nb, 100, d + 1)
        vals = []; tvals = []
        for ki, k in enumerate(CFG['K']):
            qg = j[:, :k].mean(1); qn = pnt[:, :k].mean(1)
            verify(np.all(qg > 0) and np.all(qn > 0), 'zero likelihood; no floors')
            logs = np.stack([-logsumexp(rawlog[:, :k], axis=1) + math.log(k), -np.log(qg) + lg, -np.log(qn[:, n]) + lt], axis=1)
            vals.append(scores_from_logs(logs, p, d)); logmeans[ki] += logs.sum(0); tvals.append(-np.log(qn) @ pn)
            qs_group = qn[:, n] * np.exp(lg - lt)
            maxidentity = max(maxidentity, float(np.max(abs(qg @ indicator - qn))), float(np.max(abs(qs_group @ indicator - qn))))
        out.append(np.stack(vals, axis=1)); totalrisk.append(np.stack(tvals, axis=1))
    losses = np.concatenate(out); totals = np.concatenate(totalrisk); logmeans /= CFG['banks']; mean = losses.mean(0)
    reference_F = np.array([np.einsum('kqm,m->kq', logmeans, r) for r in refs])
    reference_readings = np.array([scores_from_logs(logmeans, r, d) for r in refs])
    reference_D = np.array([functionals(r, d)[1] for r in refs])
    jensen_single_reference = reference_readings[:, :, 2] - reference_D[:, None]
    Fs = pairwise_max(reference_F); readings_sens = pairwise_max(reference_readings); js = pairwise_max(jensen_single_reference)
    sens = max(Fs, readings_sens, js)
    ok = maxidentity < CFG['identity_gate'] and normerror < CFG['mass_gate'] and formulaerror < CFG['identity_gate'] and sens <= CFG['propagated_reference_gate']
    target = OUT / 'local' / f'{profile}_d{d}_s{stream}.npz'
    verify(not target.exists(), 'refusing to overwrite a bank array')
    np.savez_compressed(target, losses=losses, total_count_loss=totals, mean_log_pattern_loss=logmeans,
                        reference_readings=reference_readings, reference_F=reference_F, single_Jensen_reference=jensen_single_reference)
    result = dict(profile=profile, d=d, stream=stream, seed=seed, initial_RNG=initial, final_RNG=rng.bit_generator.state,
                  means=mean.tolist(), total_count_means=totals.mean(0).tolist(), bank_count=256, prior_particles=25600,
                  count_identity_max_error=maxidentity, normalization_error=normerror, raw_formula_error=formulaerror,
                  propagated_reference_sensitivity=sens, F_reference_sensitivity=Fs, readings_reference_sensitivity=readings_sens,
                  single_Jensen_reference_sensitivity=js, reference_variant_F=reference_F.tolist(),
                  reference_variant_readings=reference_readings.tolist(), reference_variant_single_Jensen=jensen_single_reference.tolist(),
                  numerical_pass=bool(ok), array_sha256=sha(target), reference_standard_error=None,
                  canonical_raw_only_population_equivalence=True, elapsed_seconds=time.perf_counter() - start)
    write(OUT / 'streams' / f'{profile}_d{d}_s{stream}.json', result)
    print(json.dumps({'phase': 'banks', 'profile': profile, 'd': d, 'stream': stream, 'numerical_pass': bool(ok), 'reference_sensitivity': sens}), flush=True)
    verify(ok, 'Bank numerical/reference gate failed; stop')

def collect():
    rows = []; decision = []; stream_rows = []; crit = float(t.ppf(1 - CFG['alpha'] / (2 * 72), 7))
    for d in CFG['d']:
        for profile in CFG['profiles']:
            records = [read(OUT / 'streams' / f'{profile}_d{d}_s{s}.json') for s in range(8)]
            means = np.array([r['means'] for r in records]); D = read(OUT / 'reference' / f'{profile}_d{d}.json')['model_KL_single']
            for ki, k in enumerate(CFG['K']):
                for q, name in enumerate(NAMES):
                    values = means[:, ki, q]; avg = float(values.mean()); se = float(values.std(ddof=1) / math.sqrt(8)); lo = avg - crit * se; hi = avg + crit * se
                    gate = 1. if q == 0 else (.005 if q == 3 else .01); prec = se <= gate
                    row = dict(profile=profile, d=d, K=k, reading=name, mean=avg, SE=se, lower=lo, upper=hi, precision_pass=prec,
                               reference_qualified=True, confidence_family=72, confidence=.95,
                               model_KL_single_reference=D, derived_single_Jensen=avg - D if q == 2 else None)
                    rows.append(row)
                    stream_rows.extend(dict(profile=profile, d=d, K=k, reading=name, stream=s, value=float(v)) for s, v in enumerate(values))
                    if d == 16 and k == 10 and profile in CFG['primary_profiles'] and name == 'single_minus_group':
                        decision.append(dict(profile=profile, wrong_symmetry_preferred=prec and hi < -CFG['material'], mean=avg, upper=hi,
                                             model_change_strict_positive_from_theory=True, model_KL_reference=D))
    assert len(rows) == 72 and len(decision) == 2
    csvwrite(OUT / 'PRIMARY.csv', rows); csvwrite(OUT / 'STREAM_SCORES.csv', stream_rows)
    write(OUT / 'DECISION.json', dict(primary=decision, all_primary_pass=all(x['wrong_symmetry_preferred'] for x in decision),
          precision_pass=sum(x['precision_pass'] for x in rows), contrasts=72, streams=48, prior_banks=12288,
          prior_particles=1228800, deterministic_reference_variants=18, reference_stochastic_replicates=0,
          new_fits=0, new_observed_labels=0, no_claim_of_rigorous_reference_or_new_calibration_method=True))
    files = list((OUT / 'reference').glob('*.json')) + list((OUT / 'streams').glob('*.json')) + [OUT / 'PRIMARY.csv', OUT / 'STREAM_SCORES.csv', OUT / 'DECISION.json']
    write(OUT / 'RESULTS_FREEZE.json', dict(time_utc=now(), files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    write(OUT / 'STATE.json', dict(status='complete', time_utc=now(), active_jobs=[]), False)

def run():
    checkfreeze(); assert read(OUT / 'STATE.json')['status'] == 'frozen', 'No silent restart'
    smoke(); write(OUT / 'STATE.json', dict(status='running_reference', pid=os.getpid(), time_utc=now()), False)
    for d in CFG['d']:
        for idx, profile in enumerate(CFG['profiles']): reference(profile, d, idx)
    for d in CFG['d']:
        for idx, profile in enumerate(CFG['profiles']):
            for stream in range(8):
                write(OUT / 'STATE.json', dict(status='running_banks', d=d, profile=profile, stream=stream, pid=os.getpid(), time_utc=now()), False)
                bank_stream(profile, d, idx, stream)
    collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'run']); args = parser.parse_args()
    try:
        if args.phase == 'freeze': freeze()
        else: run()
    except Exception as ex:
        if (OUT / 'FREEZE.json').exists():
            write(OUT / f'FAILURE_{time.time_ns()}.json', dict(time_utc=now(), phase=args.phase, error=repr(ex), traceback=traceback.format_exc()))
            write(OUT / 'STATE.json', dict(status='stopped_failure', time_utc=now(), active_jobs=[]), False)
        raise
