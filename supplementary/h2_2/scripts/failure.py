"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

import math


from flint import arb, ctx

L, K, BLOCK, N_DATA, B_SELECT, T = 262144, 100, 256, 65536, 16384, 6


def ball(value):
    return dict(mid=float(value.mid()),lower=lower(value),upper=upper(value),enclosure=str(value))

def lower(value):
    return math.nextafter(float(value.lower()), -math.inf)

def upper(value):
    return math.nextafter(float(value.upper()), math.inf)

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
