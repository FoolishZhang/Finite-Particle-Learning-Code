"""Frozen independent failure-probability audit; qualified mathematical program.

The seed, thresholds, targets and family are fixed in CONTRACT.md. Binary64
sample moments are accumulated as exact integers before 128-bit Arb inference.
This does not certify Gaussian generation, S accumulation, or every score node.
"""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
import time
import traceback
import numpy as np
import failure_probability_compute as development
from flint import arb, ctx

ROOT = development.ROOT
DOC = ROOT / 'docs/08_research_研究工作/FailureProbability_v1_20261006'
OUT = ROOT / 'artifacts/failure_probability_v1_20261006'
SEED = 202610060501
L, K, BLOCK, N_DATA, B_SELECT, T = 262144, 100, 256, 65536, 16384, 6
MAX_SECONDS = 1200
BASE = 'b06ee9b039236a20734cf8a86dc2bd4a09feae7e'
FORMAL_DIAGNOSTIC_TOLERANCE = 1e-10
sha, read, rel, now = development.sha, development.read, development.rel, development.now
save = development.save_new
old = development.legacy.old
legacy = development.legacy
ctx.prec = 128


def ball(value):
    return old.ball(value)


def lower(value):
    return math.nextafter(float(value.lower()), -math.inf)


def upper(value):
    return math.nextafter(float(value.upper()), math.inf)


def exact_stats(values):
    """Exact binary64 sum / sum of squares on one common power-of-two scale."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) > 1 and np.isfinite(values).all()
    ratios = [float(value).as_integer_ratio() for value in values]
    exponents = [denominator.bit_length() - 1 for _, denominator in ratios]
    exponent = max(exponents)
    numerators = [numerator << (exponent - e) for (numerator, _), e in zip(ratios, exponents)]
    total = sum(numerators)
    squares = sum(value * value for value in numerators)
    count = len(values)
    variance_numerator = count * squares - total * total
    assert variance_numerator >= 0
    denominator = 1 << exponent
    mean = arb(total) / (arb(count) * arb(denominator))
    variance = arb(variance_numerator) / (arb(count) * arb(count - 1) * arb(denominator) ** 2)
    sd = variance.sqrt()
    return dict(count=count, common_binary_exponent=exponent,
                sum_numerator=str(total), sumsq_numerator=str(squares),
                sample_variance_numerator=str(variance_numerator),
                input_float64_sha256=hashlib.sha256(values.astype('<f8', copy=False).tobytes()).hexdigest(),
                mean=ball(mean), sample_variance=ball(variance), sample_sd=ball(sd),
                observed_min=float(values.min()), observed_max=float(values.max()),
                exact_aggregation=True), mean, variance, sd


def restore_stats(record):
    n = record['count']
    denominator = arb(1 << record['common_binary_exponent'])
    total, squares = int(record['sum_numerator']), int(record['sumsq_numerator'])
    variance_numerator = n * squares - total * total
    assert str(variance_numerator) == record['sample_variance_numerator']
    mean = arb(total) / (n * denominator)
    variance = arb(variance_numerator) / (arb(n) * (n - 1) * denominator ** 2)
    return mean, variance, variance.sqrt()


def constants():
    a_left = 1 - (arb(1) / 10).sqrt()
    a_right = (arb(28) / 10).sqrt() - 1
    assert a_left > a_right
    a = a_left
    m = 3 * a * T
    dimension = 3 * K
    phi = (-arb(T) ** 2 / 2).exp() / (2 * arb.pi()).sqrt()
    p_tail = 2 * dimension * phi / T
    q_tail = 2 * dimension * phi * (T + arb(dimension) / T)
    tau2 = 3 * a * a * q_tail
    tau1 = (tau2 * p_tail).sqrt()
    bk = arb(8) / 3 * (arb(K).log() + 3 * arb(2).log())
    residual = 3 * a * a * bk
    eps_x = arb('1e-9') + arb('1e-12')
    eps_delta = 2 * arb('1e-9') + arb('1e-12')
    eps_square = (2 * m + eps_x) * eps_x + arb('1e-12')
    risk = arb(9) / 100 * (-arb(1)).exp() / 32
    return dict(A=a, M=m, D=arb(dimension), phi_T=phi, p_T=p_tail, Q_T=q_tail,
                tau1=tau1, tau2=tau2, b_K=bk, residual_second_moment_upper=residual,
                epsilon_X=eps_x, epsilon_D=eps_delta, epsilon_X_squared=eps_square,
                risk_grid=risk, range_X=2 * m, range_X_squared=m * m, range_D=4 * m,
                alpha=arb(1) / 100, point_alpha=arb(1) / 300,
                log_one_sided=arb(600).log(), log_two_sided=arb(1200).log())


def eb_width(sd, epsilon, range_length, log_factor, count):
    sd_inflation = (arb(count) / (count - 1)).sqrt() * epsilon
    width = (arb(2) * log_factor / count).sqrt() * (sd + sd_inflation)
    width += 7 * range_length * log_factor / (3 * (count - 1))
    return width, sd_inflation


def cantelli_lower(g_lower, v_upper, epsilon_algorithm='0'):
    threshold = arb('1e-12') + arb(epsilon_algorithm)
    margin = arb(g_lower) - threshold
    if not margin > 0:
        return dict(threshold=ball(threshold), margin=ball(margin), probability_lower=0.,
                    raw_probability=ball(arb(0)), positive_margin=False)
    probability = margin * margin / (arb(v_upper) + margin * margin)
    return dict(threshold=ball(threshold), margin=ball(margin), probability_lower=max(0., lower(probability)),
                raw_probability=ball(probability), positive_margin=True)


def infer(records):
    c = constants()
    xm, _, xs = restore_stats(records['X'])
    qm, _, qs = restore_stats(records['X_squared'])
    dm, _, ds = restore_stats(records['D_T'])
    assert all(row['count'] == L for row in records.values())
    specs = [('X', xs, c['epsilon_X'], c['range_X'], c['log_one_sided']),
             ('X_squared', qs, c['epsilon_X_squared'], c['range_X_squared'], c['log_one_sided']),
             ('D_T', ds, c['epsilon_D'], c['range_D'], c['log_two_sided'])]
    widths, inflation, components = {}, {}, {}
    for name, sd, epsilon, span, log_factor in specs:
        widths[name], inflation[name] = eb_width(sd, epsilon, span, log_factor, L)
        components[name] = dict(numerical_mean_allowance=ball(epsilon),
                                numerical_sd_inflation=ball(inflation[name]),
                                true_range_length=ball(span), log_factor=ball(log_factor),
                                EB_halfwidth=ball(widths[name]))
    g_expression = xm - c['epsilon_X'] - widths['X'] - c['tau1']
    u2_expression = qm + c['epsilon_X_squared'] + widths['X_squared'] + c['tau2']
    delta_low = dm - c['epsilon_D'] - widths['D_T'] - 2 * c['tau1']
    delta_high = dm + c['epsilon_D'] + widths['D_T'] + 2 * c['tau1']
    g_lower, u2 = lower(g_expression), upper(u2_expression)
    delta_interval = [lower(delta_low), upper(delta_high)]
    u_delta = max(abs(x) for x in delta_interval)
    variance_parts = dict(data=3 * arb(u_delta) ** 2 / (16 * N_DATA),
                          bank=arb(u2) / B_SELECT,
                          interaction=c['residual_second_moment_upper'] / (N_DATA * B_SELECT))
    v_expression = sum(variance_parts.values(), arb(0))
    v_upper = upper(v_expression)
    mathematical = cantelli_lower(g_lower, v_upper)
    native_conditional = cantelli_lower(g_lower, v_upper, '1e-8')
    return dict(constants={name: ball(value) for name, value in c.items()},
                confidence_components=components,
                g_lower=g_lower, g_lower_expression=ball(g_expression),
                bank_second_moment_upper=u2, bank_second_moment_expression=ball(u2_expression),
                delta_interval=delta_interval, delta_absolute_upper=u_delta,
                variance_parts={name: ball(value) for name, value in variance_parts.items()},
                variance_upper=v_upper, variance_expression=ball(v_expression),
                mathematical_program=mathematical,
                native_program_conditional_sensitivity=dict(epsilon_algorithm='1e-8',
                    uniform_error_bound_established=False, unconditional_native_probability_claim=False,
                    **native_conditional),
                audit_alpha=.01, family_components=3, point_alpha=1 / 300,
                sampling_unit='one new bank; eight labels and both anchors are coupled',
                g_from_old_family_used=False, old_development_samples_in_confidence_statistics=0,
                statistical_qualification='ideal iid model and declared per-label numerical allowances; not certified at every node')


def qualify():
    assert not (OUT / 'QUALIFICATION.json').exists()
    fixtures = [np.array([-1.25, .5, 2., 2.75]), np.full(7, .1),
                np.array([2. ** 20, -2. ** 20, 2. ** -20, -2. ** -20])]
    rows = []
    for values in fixtures:
        record, mean, variance, _ = exact_stats(values)
        fractions = [Fraction.from_float(float(x)) for x in values]
        exact_mean = sum(fractions) / len(fractions)
        exact_variance = sum((x - exact_mean) ** 2 for x in fractions) / (len(fractions) - 1)
        assert (mean - arb(exact_mean.numerator) / exact_mean.denominator).contains(0)
        assert (variance - arb(exact_variance.numerator) / exact_variance.denominator).contains(0)
        restored = restore_stats(record)
        assert (restored[0] - mean).contains(0) and (restored[1] - variance).contains(0)
        rows.append(dict(values=values.tolist(), mean_fraction=str(exact_mean),
                         variance_fraction=str(exact_variance), record=record))
    assert rows[0]['mean_fraction'] == '1' and rows[0]['variance_fraction'] == '25/8'
    width, _ = eb_width(arb(2), arb(1) / 8, arb(5), arb(600).log(), 1000)
    inflated_variance = (2 + (arb(1000) / 999).sqrt() / 8) ** 2
    reference = (2 * inflated_variance * arb(600).log() / 1000).sqrt() + 35 * arb(600).log() / (3 * 999)
    assert (width - reference).contains(0)
    c = constants()
    risk_other = (arb(1) / 2) * (((-arb(1) / 2).exp() / 4) * (arb(3) / 10)) ** 2
    assert (c['risk_grid'] - risk_other).contains(0)
    assert c['risk_grid'] > arb('.00103466') and c['risk_grid'] < arb('.00103466093')
    fixed = cantelli_lower(1., 4., '-1e-12')
    assert fixed['probability_lower'] <= .2 and fixed['raw_probability']['upper'] >= .2
    assert cantelli_lower(0., 1.)['probability_lower'] == 0.
    assert cantelli_lower(1., 1.)['probability_lower'] > cantelli_lower(1., 4.)['probability_lower']
    candidates = old.grid()
    assert len(candidates) == 143 and candidates.count((0, 0, 0)) == 1
    assert all(any(abs(x) >= 3 for x in candidate) for candidate in candidates if candidate != (0, 0, 0))
    development_qualification = read(OUT / 'MANUFACTURED_QUALIFICATION.json')
    assert development_qualification['passed']
    assert development_qualification['code_sha256'] == sha(development.__file__)
    result = dict(created_utc=now(), code_sha256=sha(__file__), contract_sha256=sha(DOC / 'CONTRACT.md'),
                  proof_sha256=sha(DOC / 'PROOF.md'), bits=128, passed=True,
                  exact_moment_fixtures=rows, EB_formula_fixture_difference=ball(width - reference),
                  risk_constant=ball(c['risk_grid']), risk_formula_difference=ball(c['risk_grid'] - risk_other),
                  grid_candidates=143, grid_only_low_risk_point_is_I=True,
                  Cantelli_rational_fixture=fixed, new_RNG_draws=0,
                  reused_development_qualification_sha256=sha(OUT / 'MANUFACTURED_QUALIFICATION.json'))
    save(OUT / 'QUALIFICATION.json', result)
    print(json.dumps(dict(qualified=True, code_sha256=result['code_sha256'],
                          exact_moment_fixtures=len(rows), new_RNG_draws=0)), flush=True)


def freeze():
    assert not (OUT / 'FREEZE.json').exists() and not (OUT / 'RUN_START.json').exists()
    qualification = read(OUT / 'QUALIFICATION.json')
    assert qualification['passed'] and qualification['code_sha256'] == sha(__file__)
    assert qualification['contract_sha256'] == sha(DOC / 'CONTRACT.md')
    assert qualification['proof_sha256'] == sha(DOC / 'PROOF.md')
    development_inputs = development.verify_inputs(OUT)
    source = ROOT / development_inputs['source_array']
    assert sha(source) == development_inputs['expected_source_array_sha256']
    # Check executable source and old run/freeze receipts, not planning documents
    # that naturally declare the still-unused new seed.
    found = []
    for path in (ROOT / 'script').glob('*.py'):
        if path.resolve() != Path(__file__).resolve() and str(SEED) in path.read_text('utf-8', errors='replace'):
            found.append(rel(path))
    for pattern in ('*FREEZE*.json', '*RUN*.json'):
        for path in (ROOT / 'artifacts').rglob(pattern):
            if str(SEED) in path.read_text('utf-8', errors='replace'):
                found.append(rel(path))
    assert not found, ('New seed already used', found)
    paths = [Path(__file__), DOC / 'CONTRACT.md', DOC / 'PROOF.md', DOC / 'DEVELOPMENT_CONTRACT.md',
             Path(development.__file__), Path(legacy.__file__), Path(old.__file__),
             OUT / 'HANDOFF_INPUT.md', OUT / 'SCOPE.json', OUT / 'QUALIFICATION.json',
             OUT / 'MANUFACTURED_QUALIFICATION.json', OUT / 'DEVELOPMENT_INPUTS.json',
             OUT / 'DEVELOPMENT_RESULTS.json', development.DEFAULT_REFERENCE]
    record = dict(created_utc=now(), baseline=BASE, sources={rel(p): sha(p) for p in paths},
                  old_development_array=dict(path=rel(source), sha256=sha(source), bytes=source.stat().st_size),
                  seed=SEED, seed_roles=dict(active=1, nuisance=2), seed_unused_in_old_sources=True,
                  generator='PCG64DXSM(SeedSequence([seed, role]))', L=L, K=K, block=BLOCK, T=T,
                  N_data=N_DATA, B_selection=B_SELECT, alpha=.01, point_alpha=1 / 300,
                  quantities=['X', 'X_squared', 'D_T'], numerical_eta_d='1e-9',
                  native_algorithm_sensitivity_only='1e-8', manufactured_qualification_passed=True,
                  no_new_scientific_rng_before_freeze=True, maximum_run_seconds=MAX_SECONDS,
                  bank_audit_indices=np.linspace(0, L - 1, 32, dtype=int).tolist(),
                  old_samples_used_for_confidence=0, adaptive_budget_or_seed_changes=False)
    save(OUT / 'FREEZE.json', record)
    print(json.dumps(dict(frozen=True, freeze_sha256=sha(OUT / 'FREEZE.json'), seed=SEED,
                          no_new_scientific_rng_before_freeze=True)), flush=True)


def verify_freeze():
    frozen = read(OUT / 'FREEZE.json')
    for name, digest in frozen['sources'].items():
        assert sha(ROOT / name) == digest, ('Frozen source changed', name)
    assert (frozen['seed'], frozen['L'], frozen['K'], frozen['T'], frozen['block']) == (SEED, L, K, T, BLOCK)
    return frozen


def diagnostic_bank(z, s, point_differences):
    rows = []
    for yi, label in enumerate(development.YS):
        counts = np.zeros(8, dtype=np.int64)
        counts[yi] = 1
        vi = legacy.interval_pair((0, 0, 0), z[None], s[None], counts)[0][1]
        vb = legacy.interval_pair((9, 9, 9), z[None], s[None], counts)[0][1]
        discrepancy = abs(arb(float(point_differences[yi])) - (vi - vb))
        error_upper = upper(discrepancy)
        rows.append(dict(label_index=yi, label=label.tolist(), binary64=float(point_differences[yi]),
                         interval=ball(vi - vb), error_upper=error_upper,
                         passed=error_upper <= FORMAL_DIAGNOSTIC_TOLERANCE))
    assert all(row['passed'] for row in rows), rows
    return rows


def write_results_docs(result):
    inference = result['inference']
    records = result['scalar_statistics']
    parts = inference['variance_parts']
    p = inference['mathematical_program']['probability_lower']
    p_native = inference['native_program_conditional_sensitivity']['probability_lower']
    rows = '\n'.join(f"|{name}|{record['mean']['mid']:.12g}|{record['sample_variance']['mid']:.12g}|"
                     for name, record in records.items())
    content = f'''# FailureProbability-v1 独立审计结果

本页由冻结实现从RESULTS生成。它是供主代理和后续外部审查使用的研究结果，不改动现有论文，也不是对整条数值流水线的认证。

固定的新审计使用 {L:,} 个独立银行，K={K}，T={T}，seed={SEED}，role1/2分别生成活跃与无关Gaussian。旧开发银行未进入本轮置信统计。目标程序仍为N={N_DATA:,}个标签、B={B_SELECT:,}个共享银行、143个固定候选、n256和10^-12容差；本轮未运行该搜索或抽标签。

在理想iid模型和合同声明的逐标签数值误差资格下，以至少99%的**审计置信度**，该数学评分程序在重新抽取标签和银行时输出风险至少精确常数 `.09 exp(-1)/32` 的概率，下界为 **{p:.12g}**。风险常数严格大于.00103466；它的向上近似.00103466093未用作严格门。这不是“新运行以99%概率失败”。

## 三个预定统计量

|截断量|样本均值|无偏样本方差|
|---|---:|---:|
{rows}

截断银行数：{result['truncated_banks']}。保存float64序列的sum/sumsq以共同二进制分母的整数精确累计，再由128-bit Arb计算均值、标准差、EB半宽、尾界与概率端点。三项分别分配α/3；均值和二阶矩为单侧，D_T为双侧，使用log600/log1200。

## 置信预算与程序概率

- g下界：{inference['g_lower']:.17g}。
- 银行二阶矩上界：{inference['bank_second_moment_upper']:.17g}。
- Δ区间：{inference['delta_interval']}，绝对值上界：{inference['delta_absolute_upper']:.17g}。
- 程序方差上界：{inference['variance_upper']:.17g}。
- 数据/银行/交互三项上界贡献中值分别为 {parts['data']['mid']:.12g}、{parts['bank']['mid']:.12g}、{parts['interaction']['mid']:.12g}；JSON保存各项向外区间。
- 仅在另有统一 `ε_alg<=10^-8` 保证时，原生浮点程序的条件性敏感性值为 {p_native:.12g}。本轮没有建立该统一条件，也没有给原生程序无条件概率下界。

## 验证、资源与未认证边界

制造验证包含精确矩、EB、风险常数与Cantelli夹具。新银行按预定32个等距索引进行了全部8标签的128-bit Arb定向检查，最大误差上界 {result['arithmetic_diagnostics']['max_error_upper']:.12g}，门为10^-10；该检查不覆盖全节点。实际新增先验粒子26,214,400、Gaussian标量6,789,529,600、候选×银行×标签评分4,194,304次，另有256个Arb点差诊断。无新观测、训练、优化或求积。

运行用时 {result['wall_seconds']:.3f} 秒，冻结SHA256为 `{result['freeze_sha256']}`。所有输入Z/S、八标签点差、截断掩码和三标量序列均保存；无关Gaussian的逐坐标值没有存储，可按冻结RNG再生。

尚未认证：Gaussian生成的理想iid性质、S累计的全流水线误差、全部浮点评分节点的10^-9逐标签误差界，以及新运行原生程序的统一ε_alg。有限Arb诊断支持数值资格，不能替代这些命题。此结果也不证明n0程序恢复、任意优化器失败、任意细网格失败或连续最小化算法的失败概率。

完整数值、向外端点、逐例产物哈希和资源账目见[RESULTS.json](../../../artifacts/failure_probability_v1_20261006/RESULTS.json)，精确整数矩记录见[SCALAR_STATS.json](../../../artifacts/failure_probability_v1_20261006/SCALAR_STATS.json)。
'''
    (DOC / 'RESULTS.md').write_text(content, encoding='utf-8', newline='\n')
    (DOC / 'README.md').write_text('''# FailureProbability-v1：固定有限搜索程序的独立概率审计

先读[执行合同](CONTRACT.md)、[概率推导](PROOF.md)和[结果与资格](RESULTS.md)。开发过程见[开发合同](DEVELOPMENT_CONTRACT.md)，开发旧银行与正式独立审计的样本角色分开。

本包新增一个预定α=.01审计家族，给已固定n256数学搜索程序的新运行风险概率下界，保留数值资格。99%表示审计置信度，不是算法失败概率。无原生浮点程序的无条件保证，无一般优化器保证；稿件尚未据此更新。

入口产物：[FREEZE](../../../artifacts/failure_probability_v1_20261006/FREEZE.json)、[QUALIFICATION](../../../artifacts/failure_probability_v1_20261006/QUALIFICATION.json)、[RESULTS](../../../artifacts/failure_probability_v1_20261006/RESULTS.json)。大数组保存在本地并记录哈希，不要求外部审查者重复全量计算。
''', encoding='utf-8', newline='\n')


def run():
    frozen = verify_freeze()
    assert not (OUT / 'RUN_START.json').exists(), 'One formal run only; retain any failed run'
    started = time.monotonic()
    save(OUT / 'RUN_START.json', dict(utc=now(), freeze_sha256=sha(OUT / 'FREEZE.json'),
                                    code_sha256=sha(__file__), seed=SEED, new_scientific_run=True))
    zr = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([SEED, 1])))
    sr = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([SEED, 2])))
    z = np.empty((L, K, 3), dtype=np.float64)
    s = np.empty((L, K), dtype=np.float64)
    differences = np.empty((L, 8), dtype=np.float64)
    truncated = np.empty(L, dtype=bool)
    completed = 0
    try:
        for lo in range(0, L, BLOCK):
            hi = lo + BLOCK
            zz = zr.standard_normal((BLOCK, K, 3))
            ss = -np.logaddexp(0, -sr.standard_normal((BLOCK, K, 256))).sum(axis=2)
            point = development.point_differences(zz, ss)
            z[lo:hi], s[lo:hi], differences[lo:hi] = zz, ss, point
            truncated[lo:hi] = np.max(np.abs(zz), axis=(1, 2)) > T
            completed = hi
            assert time.monotonic() - started < MAX_SECONDS, 'Frozen 20-minute run budget exceeded'
            if hi % (BLOCK * 64) == 0 or hi == L:
                print(json.dumps(dict(stage='new_audit_banks', completed=hi, total=L,
                                      wall_seconds=time.monotonic() - started)), flush=True)
        x = np.where(truncated, 0., differences.mean(axis=1))
        delta = np.where(truncated, 0., differences[:, [0, 7]].mean(axis=1) - differences[:, 1:7].mean(axis=1))
        square = x * x
        arrays_path = OUT / 'AUDIT_BANKS.npz'
        development.save_npz_new(arrays_path, z=z, s=s, difference_by_label=differences,
                                  truncated=truncated, label_order=development.YS,
                                  X=x, X_squared=square, D_T=delta)
        scalar_values = {'X': x, 'X_squared': square, 'D_T': delta}
        records = {name: exact_stats(values)[0] for name, values in scalar_values.items()}
        save(OUT / 'SCALAR_STATS.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'FREEZE.json'),
             bits=128, exact_binary64_integer_moments=True, records=records, truncated_banks=int(truncated.sum())))
        inference = infer(records)
        c = constants()
        range_gates = dict(X=all(abs(arb(float(value))) <= c['M'] + c['epsilon_X'] for value in (x.min(), x.max())),
                           X_squared=bool(arb(float(square.min())) >= 0 and arb(float(square.max())) <= c['M'] ** 2 + c['epsilon_X_squared']),
                           D_T=all(abs(arb(float(value))) <= 2 * c['M'] + c['epsilon_D'] for value in (delta.min(), delta.max())))
        assert all(range_gates.values())
        audit_rows = []
        for index in frozen['bank_audit_indices']:
            rows = diagnostic_bank(z[index], s[index], differences[index])
            audit_rows.append(dict(bank_index=index, rows=rows))
            assert time.monotonic() - started < MAX_SECONDS
        audit = dict(bits=128, indices=frozen['bank_audit_indices'], rows=audit_rows,
                     bank_label_differences=256, interval_pair_calls=512,
                     max_error_upper=max(row['error_upper'] for bank in audit_rows for row in bank['rows']),
                     tolerance=FORMAL_DIAGNOSTIC_TOLERANCE, all_passed=True,
                     all_scientific_nodes_checked=False,
                     input_scope='saved binary64 Z/S plus matrix-root allowance only')
        save(OUT / 'ARITHMETIC_DIAGNOSTICS.json', audit)
        # Feed only old descriptive development statistics into a separate
        # comparison; never combine them with the new confidence family.
        previous = read(OUT / 'DEVELOPMENT_RESULTS.json')
        previous_t6 = next(row for row in previous['truncation_diagnostics'] if row['T'] == T)['summaries']
        comparison = {name: dict(old_development_mean=previous_t6[name]['mean'],
                     new_audit_mean=records[name]['mean']['mid'],
                     difference=records[name]['mean']['mid'] - previous_t6[name]['mean'])
                      for name in records}
        informational = inference['mathematical_program']['probability_lower'] >= .1
        result = dict(finished_utc=now(), wall_seconds=time.monotonic() - started,
                      baseline=BASE, freeze_sha256=sha(OUT / 'FREEZE.json'), code_sha256=sha(__file__),
                      scalar_statistics=records, inference=inference, truncated_banks=int(truncated.sum()),
                      arithmetic_diagnostics={key: value for key, value in audit.items() if key != 'rows'},
                      development_comparison=comparison, gates=dict(manufactured_qualification=True,
                          ranges=all(range_gates.values()), fixed_Arb_diagnostics=True,
                          positive_g_lower=inference['mathematical_program']['positive_margin'],
                          informative_probability_at_least_point_one=informational),
                      resources=dict(new_observations=0, new_prior_vectors=L * K,
                          new_Gaussian_scalars=L * K * 259, new_active_Gaussian_scalars=L * K * 3,
                          new_nuisance_Gaussian_scalars=L * K * 256, audit_banks=L, K=K,
                          candidate_bank_label_logscore_evaluations=2 * L * 8,
                          particle_emission_evaluations=2 * L * 8 * K,
                          coupled_kernel_block_calls=L // BLOCK, optimizer_updates=0, quadrature_nodes=0,
                          new_statistical_components=3, new_family_alpha=.01,
                          Arb_bank_label_differences=256, Arb_interval_pair_calls=512,
                          Arb_candidate_particle_label_products=512 * K * 8),
                      array=dict(path=rel(arrays_path), sha256=sha(arrays_path), bytes=arrays_path.stat().st_size),
                      scalar_stats_sha256=sha(OUT / 'SCALAR_STATS.json'),
                      arithmetic_diagnostics_sha256=sha(OUT / 'ARITHMETIC_DIAGNOSTICS.json'),
                      claims=dict(qualified_mathematical_program_probability=True,
                          audit_confidence=.99, audit_confidence_is_algorithm_failure_probability=False,
                          uniform_pointwise_eta_d_proved=False, Gaussian_and_S_pipeline_certified=False,
                          unconditional_native_program_probability=False, general_optimizer_guarantee=False,
                          new_n0_recovery_guarantee=False, no_old_samples_in_new_family=True,
                          original_search_program_modified=False, manuscript_modified=False))
        save(OUT / 'RESULTS.json', result)
        write_results_docs(result)
        verify_freeze()
        print(json.dumps(development.native(dict(completed=True,
                          g_lower=inference['g_lower'], variance_upper=inference['variance_upper'],
                          variance_parts=inference['variance_parts'],
                          probability_lower=inference['mathematical_program']['probability_lower'],
                          conditional_native_sensitivity=inference['native_program_conditional_sensitivity']['probability_lower'],
                          truncated_banks=int(truncated.sum()), max_Arb_error=audit['max_error_upper'],
                          wall_seconds=result['wall_seconds'], gates=result['gates'])), ensure_ascii=True), flush=True)
    except BaseException:
        if completed and not (OUT / 'AUDIT_BANKS.npz').exists():
            development.save_npz_new(OUT / f'PARTIAL_BANKS_{completed}.npz', z=z[:completed], s=s[:completed],
                                       difference_by_label=differences[:completed], truncated=truncated[:completed],
                                       label_order=development.YS)
        save(OUT / f'PARTIAL_RUN_{time.time_ns()}.json', dict(utc=now(), completed_banks=completed,
             wall_seconds=time.monotonic() - started, seed=SEED, freeze_sha256=sha(OUT / 'FREEZE.json'),
             active_rng_state=zr.bit_generator.state, nuisance_rng_state=sr.bit_generator.state,
             no_seed_change_or_budget_extension=True))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('qualify', 'freeze', 'run'))
    args = parser.parse_args()
    try:
        globals()[args.mode]()
    except BaseException as error:
        save(OUT / f'FAILURE_{args.mode}_{time.time_ns()}.json',
             dict(utc=now(), code_sha256=sha(__file__), error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
