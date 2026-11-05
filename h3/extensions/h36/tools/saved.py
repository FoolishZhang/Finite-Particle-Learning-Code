# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Reaggregate selected saved matrices/scalars; no model producers or integration."""
from pathlib import Path
import json
import math

import numpy as np
from scipy.stats import t as student_t

STRUCTURES = ('L', 'A', 'B', 'JNT')
CONTRASTS = {
    'A_minus_L_all': ('A', 'L', 'all', 'directional'),
    'B_minus_A_within': ('B', 'A', 'within', 'directional'),
    'B_minus_A_cross_equivalence': ('B', 'A', 'cross', 'equivalence'),
    'JNT_minus_B_cross': ('JNT', 'B', 'cross', 'directional'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class Audit:
    def __init__(self):
        self.count, self.failures, self.maximum_absolute_error = 0, [], 0.

    def require(self, value, label):
        self.count += 1
        if not value:
            self.failures.append(label)

    def close(self, actual, expected, label, tolerance=5e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.require(False, label + ': shape differs')
            return
        err = float(np.max(np.abs(actual - expected))) if actual.size else 0.
        self.maximum_absolute_error = max(self.maximum_absolute_error, err)
        self.require(np.isfinite(err) and err <= tolerance, label)


def design_arrays(design):
    d, K = design['d'], design['K']
    tau = np.array(design['tau'])
    graph = np.zeros((d, d))
    for i, j in design['teacher_edges']:
        graph[i, j] = graph[j, i] = 1.
    G = graph / design['teacher_graph_divisor']
    edges = [(i, j) for i in range(d) for j in range(i + 1, d)]
    g = np.array([G[i, j] for i, j in edges])
    v = 1 / (1 + tau)
    J = np.array([v[i] * v[j] for i, j in edges])
    block = {i: b for b, cells in enumerate(design['blocks']) for i in cells}
    within = np.array([block[i] == block[j] for i, j in edges])
    masks = {'all': g != 0, 'within': within & (g != 0), 'cross': ~within & (g != 0)}
    product = np.array([(1 - v[i]) * (1 - v[j]) for i, j in edges])
    residuals = {'L': 1 - J, 'A': product, 'B': np.where(within, 0., product), 'JNT': np.zeros_like(J)}
    theory = {q: {s: float(np.sum(J[m] * g[m]**2 * J[m] / (J[m] + R[m] / K)) /
                                np.sum(J[m] * g[m]**2)) for s, m in masks.items()}
              for q, R in residuals.items()}
    return edges, g, J, masks, theory, residuals


def summary(values, critical, center=0.):
    values = np.asarray(values)
    mean, sd = float(np.mean(values)) + center, float(np.std(values, ddof=1))
    se = sd / math.sqrt(len(values))
    halfwidth = critical * se
    return {'mean': mean, 'sample_SD': sd, 'MCSE': se, 'R': len(values),
            'approximate_simultaneous_halfwidth': halfwidth,
            'approximate_simultaneous_interval': [mean - halfwidth, mean + halfwidth]}


def neural(folder, audit):
    folder = Path(folder)
    design = read(folder / 'data/neural_design.json')
    records = read(folder / 'data/neural_records.json')['records']
    saved = read(folder / 'data/fresh_statistics.json')
    ids = saved['expected_data_ids']
    audit.require(ids == list(range(300, 312)) and len(records) == 96, 'fresh fixed census')
    by_key = {(r['data_id'], r['structure'], int(np.sign(r['eta']))): r for r in records}
    audit.require(len(by_key) == 96, 'unique dataset/structure/sign endpoints')
    edges, g, J, masks, theory, residuals = design_arrays(design)
    responses = {}
    for data_id in ids:
        responses[data_id] = {}
        for q in STRUCTURES:
            pos, neg = [by_key[data_id, q, sign] for sign in (1, -1)]
            for row in (pos, neg):
                audit.require(row['checkpoint_step'] == 28000 and row['start'] == 0
                              and row['available'] and row['metadata_binding_verified']
                              and not row['fit_failed'] and not row['fallback_used']
                              and row['checkpoint_role'] == 'lastlegal', 'saved fixed endpoint role')
                C = np.asarray(row['correlation'])
                audit.require(C.shape == (8, 8) and np.isfinite(C).all(), 'saved C dimensions')
                audit.close(C, C.T, 'saved C symmetry')
                audit.close(C.diagonal(), np.ones(8), 'saved C diagonal')
                audit.require(np.linalg.eigvalsh(C).min() > 0, 'saved C positive definite')
            audit.require(pos['seed'] == neg['seed'], 'paired training seed')
            D = (np.array(pos['correlation']) - np.array(neg['correlation'])) / (2 * design['amplitude'])
            dv = np.array([D[i, j] for i, j in edges])
            rho = {s: float(np.sum(J[m] * g[m] * dv[m]) / np.sum(J[m] * g[m]**2)) for s, m in masks.items()}
            for s, value in rho.items():
                audit.close(value, saved['per_data'][str(data_id)][q]['rho'][s], 'C to saved rho')
            responses[data_id][q] = rho
    statistics = {}
    critical = float(student_t.ppf(1 - .05 / 8, len(ids) - 1))
    for key, (a, b, group, kind) in CONTRASTS.items():
        values = [responses[i][a][group] - responses[i][b][group] for i in ids]
        st = summary(values, critical)
        original = saved['primary'][key]
        audit.close(values, [original['values_by_data_id'][str(i)] for i in ids], 'saved paired values')
        audit.close(st['mean'], original['mean'], 'fresh mean')
        audit.close(st['sample_SD'], original['paired_SD'], 'fresh paired SD')
        audit.close(st['approximate_simultaneous_interval'], original['simultaneous_95_CI'], 'fresh simultaneous CI')
        audit.close(critical, original['critical_t'], 'fresh t critical')
        lo, hi = st['approximate_simultaneous_interval']
        decision = lo >= -.05 and hi <= .05 if kind == 'equivalence' else st['mean'] >= .05 and lo > 0
        audit.require(decision, 'predeclared fresh structure rule')
        statistics[key] = {'prediction': theory[a][group] - theory[b][group],
                           'mean': st['mean'], 'paired_SD': st['sample_SD'],
                           'simultaneous_95_CI': st['approximate_simultaneous_interval'],
                           'critical_t': critical, 'R': len(ids), 'kind': kind, 'margin': .05,
                           'rule_met': decision, 'values_by_data_id': dict(zip(map(str, ids), values))}
    old = read(folder / 'data/old_equal_updates_statistics.json')
    for key, original in old['primary'].items():
        st = summary(list(original['values_by_data_id'].values()), critical)
        audit.close(st['mean'], original['mean'], 'old posthoc mean')
        audit.close(st['sample_SD'], original['paired_SD'], 'old posthoc SD')
        audit.close(st['approximate_simultaneous_interval'], original['simultaneous_95_CI'], 'old posthoc CI')
    return {'statistics': statistics, 'theory': {'J': J.tolist(), 'rho': theory,
            'residual_diagonals': {q: r.tolist() for q, r in residuals.items()}},
            'data_ids': ids, 'endpoint_count': len(records),
            'scope': 'saved d8 K4 resource-amended fixed-update independent confirmation'}


def risk64(folder, audit):
    saved = read(Path(folder) / 'data/r64.json')
    records = saved['records']
    audit.require(len(records) == 256 and len({r['dataset_id'] for r in records}) == 256, 'R64 dataset census')
    critical = float(student_t.ppf(1 - .05 / 8, 63))
    audit.close(critical, saved['student_critical'], 'R64 Student critical')
    audit.require(saved['inference_df'] == 63, 'R64 df')
    result = []
    for condition, theory in zip(saved['conditions'], saved['theory_comparison']):
        rows = [r for r in records if r['condition_id'] == condition['condition_id']]
        audit.require(len(rows) == 64 and [r['dataset_id'] for r in rows] == condition['dataset_ids'], 'R64 fixed condition roster')
        raw = np.array([r['arms']['A']['N_KL'] - r['arms']['L']['N_KL'] for r in rows])
        centers = [r['expected_C_N'] for r in rows]
        audit.close(centers, [condition['primary']['finite64_center']] * 64, 'same finite64 center')
        controlled_residual = raw - np.array([r['C_N'] for r in rows])
        primary = summary(controlled_residual, critical, centers[0])
        raw_summary = summary(raw, critical)
        for label, actual in (('primary', primary), ('raw', raw_summary)):
            for field, value in actual.items():
                audit.close(value, condition[label][field], 'R64 ' + label + '/' + field)
        N, j, eta = condition['N'], condition['j'], np.array(condition['eta'])
        v = math.sqrt(j)
        rL = 8 * j / (1 + 7 * j)
        rA = 8 * j / (8 * j + (1 - v)**2)
        leading = .5 * ((1 - rA)**2 - (1 - rL)**2) * condition['S'] + 1.5 * (rA**2 - rL**2)
        audit.close(leading, theory['DeltaInfinity'], 'R64 leading risk difference')
        arms = {}
        for q in ('L', 'A'):
            errors = np.array([r['arms'][q]['theta'] for r in rows]) - eta
            quadratic = .5 * N * j * np.sum(errors**2, axis=1)
            bias = float(.5 * N * j * np.sum(errors.mean(axis=0)**2))
            variance = float(.5 * N * j * np.mean(np.sum((errors - errors.mean(axis=0))**2, axis=1)))
            original = condition['explanatory']['arms'][q]
            audit.close(errors.mean(axis=0), original['mean_parameter_error'], 'R64 parameter mean/' + q)
            audit.close(np.cov(errors.T, ddof=1), original['sample_parameter_covariance'], 'R64 parameter covariance/' + q)
            # The historical scalar has unscaled parameter-squared units;
            # the paper's B has the additional factor N*j/2.
            audit.close(bias, .5 * N * j * original['bias_squared_plugin'], 'R64 scaled plug-in bias/' + q)
            audit.close(quadratic.mean(), original['Q_N']['mean'], 'R64 quadratic risk/' + q)
            audit.close(bias + variance, original['quadratic_mean_finite_R_identity'], 'R64 finite-R identity/' + q)
            for r in rows:
                flags = r['arms'][q]
                audit.require(flags['converged'] and flags['numeric_qualification']
                              and not flags['boundary'] and not flags['fallback'], 'R64 retained numerical flags/' + q)
            arms[q] = {'B': bias, 'V': variance, 'Q': float(quadratic.mean())}
        result.append({'condition_id': condition['condition_id'], 'N': N, 'S': condition['S'],
                       'leading_delta': leading, 'primary': primary, 'raw': raw_summary,
                       'bias_variance': arms, 'delta_B': arms['A']['B'] - arms['L']['B'],
                       'delta_V': arms['A']['V'] - arms['L']['V'],
                       'delta_Q': arms['A']['Q'] - arms['L']['Q']})
    return result


def risk_table(rows, zh=False):
    header = (r'| 条件 $(N,S)$ | 理论 $\Delta_\infty(S)$ | 校正均值 $\Delta_N^{\rm ctrl}$ | 近似同时95%区间 |' if zh else
              r'| Condition $(N,S)$ | Theory $\Delta_\infty(S)$ | Controlled $\Delta_N^{\rm ctrl}$ | Approx. simultaneous 95% CI |') + '\n'
    lines = [header, '|---|---:|---:|---:|\n']
    for row in rows:
        low, high = row['primary']['approximate_simultaneous_interval']
        lines.append(f"| $({row['N']},{row['S']:g})$ | ${row['leading_delta']:+.5f}$ | ${row['primary']['mean']:+.5f}$ | $[{low:+.5f},{high:+.5f}]$ |\n")
    return ''.join(lines)
