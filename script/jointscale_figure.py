"""Render frozen joint-scale results; no sampling, fitting, or integration.

This file deliberately does not import either scientific producer. All plotted
numbers are copied or arithmetically summarized from completed saved outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/joint_scale_learning'
DOC = ROOT / 'docs/08_research_研究工作/两组尺度共同学习与恢复边界_20261004'
FIG = DOC / 'figures'
ARMS = ['NEAR_FIXED', 'NEAR_FREE', 'NEAR_ZERO',
        'UNIT_FIXED', 'UNIT_FREE', 'UNIT_ZERO']
METHOD_COLORS = {'FIXED': '#8b96a5', 'FREE': '#2171b5', 'ZERO': '#3a8b64'}
START_COLORS = {'NEAR': '#2171b5', 'UNIT': '#d97922'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + '\n', encoding='utf8')


def collect():
    train_path = OUT / 'TRAIN/RESULTS.json'
    ref_path = OUT / 'REFERENCE/RESULTS.json'
    train, reference = read(train_path), read(ref_path)
    assert train['config']['arms'] == ARMS
    assert [r['stream'] for r in train['records']] == list(range(16))
    assert all(r['completed_steps'] == 512 for r in train['records'])
    assert len(reference['records']) == 96
    assert reference['qualification_passed'], 'Reference qualification failed'
    sources = {relative(train_path): sha(train_path), relative(ref_path): sha(ref_path)}
    lookup = {(r['stream'], r['arm']): r for r in reference['records']}
    assert len(lookup) == 96
    risk = {}
    for arm in ARMS:
        risk[arm] = [float(lookup[s, arm]['reference_joint_KL']) for s in range(16)]
        assert all(v > 0 for v in risk[arm]), 'Log risk panel requires positive KL'
        assert np.allclose(risk[arm], reference['descriptive'][arm]['risks'],
                           rtol=0, atol=1e-15)
        assert abs(np.mean(risk[arm]) - reference['descriptive'][arm]['mean_joint_KL']) < 1e-15
    expected = [f'{start}_{left}_minus_{right}' for start in ('NEAR', 'UNIT')
                for left, right in (('FREE', 'FIXED'), ('FREE', 'PRODUCT'), ('ZERO', 'PRODUCT'))]
    assert [p['name'] for p in reference['primary']] == expected
    primary = []
    for row in reference['primary']:
        copied = {k: row[k] for k in ('name', 'start', 'left', 'right', 'mean', 'SE',
                                    'lower', 'upper', 'effect_passed', 'precision_passed',
                                    'stream_values')}
        assert copied['lower'] <= copied['mean'] <= copied['upper']
        assert abs(np.mean(copied['stream_values']) - copied['mean']) < 1e-15
        primary.append(copied)
    params = []
    for row in train['records']:
        path = ROOT / row['output_file']
        assert sha(path) == row['output_sha256']
        sources[relative(path)] = sha(path)
        with np.load(path, allow_pickle=False) as tape:
            p = np.asarray(tape['params'], dtype=np.float64)
        assert p.shape == (6, 513, 15) and np.isfinite(p).all()
        params.append(p)
    all_params = np.stack(params)
    trajectories = {}
    for start in ('NEAR', 'UNIT'):
        index = ARMS.index(f'{start}_FREE')
        tau = all_params[:, index, :, 14]
        active_variance = np.mean(all_params[:, index, :, 4:8] ** 2, axis=2)
        trajectories[start] = {
            'tau_by_stream': tau.tolist(), 'mean_tau': tau.mean(axis=0).tolist(),
            'mean_active_variance_by_stream': active_variance.tolist(),
            'mean_active_variance': active_variance.mean(axis=0).tolist()}
    return dict(sources=sources, arms=ARMS, stream_ids=list(range(16)), risk=risk,
                mean_risk={a: float(np.mean(risk[a])) for a in ARMS},
                truth_product_risk=reference['truth_product_risk'], primary=primary,
                trajectories=trajectories, steps=list(range(513)), tail_window=[449, 512],
                interval_scope=reference['interval_scope'],
                mapping={
                    'risk': 'REFERENCE.records reference_joint_KL; all 96 TAIL outputs, no LAST or checkpoint risk',
                    'contrast': 'REFERENCE.primary existing mean/lower/upper and original effect/precision flags; no new CI',
                    'tau': 'TRAIN.stream_XX.npz params[arm,step,14]; all 16 streams, FREE NEAR/UNIT only',
                    'active_variance': 'mean(params[arm,step,4:8]**2) across 4 active coordinates, then stream mean',
                    'TAIL': 'Scientific output averages physical mu and full covariance at postupdates449..512; not a predictive mixture',
                    'scatter_offsets': 'linspace(-0.16,0.16,16) in stream order; deterministic and not new RNG'},
                new_scientific_RNG=0, new_fits=0, new_model_integrals=0,
                new_confidence_intervals=0)


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', color='#e4e7eb', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def render(data):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'jointscale-frozen-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 10.0))
    fig.subplots_adjust(left=.075, right=.975, bottom=.195, top=.865,
                        wspace=.43, hspace=.49)
    fig.suptitle('Joint scale learning: risk outputs and parameter paths',
                 x=.075, y=.962, ha='left', fontsize=19, fontweight='bold')
    fig.text(.075, .919, 'K=100  |  256 nuisance variables  |  16 paired data/prior streams  |  512 updates',
             fontsize=12, color='#424b57')
    ax = axes[0, 0]
    positions = [0, 1, 2, 3.6, 4.6, 5.6]
    offsets = np.linspace(-.16, .16, 16)
    for x, arm in zip(positions, ARMS):
        method = arm.split('_')[1]
        ax.scatter(x + offsets, data['risk'][arm], s=27, alpha=.7,
                   color=METHOD_COLORS[method], edgecolor='white', linewidth=.45, zorder=3)
        ax.scatter(x, data['mean_risk'][arm], marker='D', s=63,
                   facecolor='#19232f', edgecolor='white', linewidth=.8, zorder=4)
    ax.axhline(data['truth_product_risk'], color='#a04349', linestyle='--', lw=1.35,
               label='True product-marginal risk', zorder=2)
    ax.set_xticks(positions, ['FIXED', 'FREE', 'ZERO'] * 2)
    ax.set_yscale('log')
    ax.set_ylabel('TAIL joint KL (nats; log scale)')
    ax.set_title('A  All six arms: 16 outputs and their means', loc='left', pad=13)
    ax.text(.205, -.17, 'NEAR: initial active s=0.1', transform=ax.transAxes,
            ha='center', fontsize=10)
    ax.text(.80, -.17, 'UNIT: initial active s=1', transform=ax.transAxes,
            ha='center', fontsize=10)
    ax.legend(loc='lower left', fontsize=8.5, frameon=False)
    ax.text(.02, .12, 'Each dot: one stream; diamond: mean',
            transform=ax.transAxes, fontsize=9, color='#505966')
    style_axis(ax)

    ax = axes[0, 1]
    rows = data['primary']
    ys = np.arange(5, -1, -1)
    labels = []
    for y, row in zip(ys, rows):
        both = row['effect_passed'] and row['precision_passed']
        color = '#2171b5' if both else '#bc6726'
        ax.errorbar(row['mean'] * 1000, y,
                    xerr=np.array([[row['mean'] - row['lower']], [row['upper'] - row['mean']]]) * 1000,
                    fmt='o', ms=5.6, capsize=4, color=color, elinewidth=1.7)
        effect = 'pass' if row['effect_passed'] else 'FAIL'
        precision = 'pass' if row['precision_passed'] else 'FAIL'
        labels.append(f"{row['start']}: {row['left']} − {row['right']}\nEffect {effect}; precision {precision}")
    ax.axvline(0, color='#5b6370', lw=1.1, linestyle='--')
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_ylim(-.6, 5.6)
    ax.set_xlabel('Paired risk difference (× 10⁻³ nats); negative favored')
    ax.set_title('B  All six original comparisons', loc='left', pad=13)
    ax.text(.02, -.23, 'Original shared t family: α=0.05, q=6, df=15\nDevelopment approximation; numerical integration error excluded',
            transform=ax.transAxes, fontsize=9, color='#505966', va='top')
    style_axis(ax)
    ax.grid(axis='x', color='#e4e7eb', linewidth=.7)
    ax.grid(axis='y', visible=False)

    for ax, kind, title, ylabel in (
        (axes[1, 0], 'tau', 'C  FREE nuisance scale paths', 'Shared nuisance scale τ'),
        (axes[1, 1], 'mean_active_variance', 'D  FREE active variance paths', 'Mean active variance: (Σ sᵢ²) / 4')):
        for start in ('NEAR', 'UNIT'):
            payload = data['trajectories'][start]
            values = payload['tau_by_stream' if kind == 'tau' else 'mean_active_variance_by_stream']
            mean = payload['mean_tau' if kind == 'tau' else 'mean_active_variance']
            color = START_COLORS[start]
            for row in values:
                ax.plot(data['steps'], row, color=color, alpha=.14, lw=.6)
            ax.plot(data['steps'], mean, color=color, lw=2.3,
                    label=f"{start} FREE (initial s={'0.1' if start == 'NEAR' else '1'})")
        ax.axvspan(449, 512, color='#596675', alpha=.10, zorder=0)
        ax.set_xlim(0, 512)
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_xlabel('Post-update step (0 = initialization)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', pad=13)
        ax.legend(loc='best', fontsize=9, frameon=False)
        ax.set_ylim(bottom=0)
        style_axis(ax)
    axes[1, 0].axhline(1, color='#7d8793', linestyle=':', lw=1.0)
    axes[1, 0].axhline(0, color='#7d8793', linestyle=':', lw=1.0)
    fig.text(.075, .095,
             'FIXED: τ=1; FREE: learn τ; ZERO: τ=0. All arms learn active means, scales, and correlations.\n'
             'Paths are descriptive saved parameters: thin lines = all 16 streams, thick lines = means; shading = TAIL steps 449–512.\n'
             'No checkpoint risk integrals, trajectory confidence bands, collision-window confirmation, or additional scientific runs.',
             fontsize=9.5, color='#414c59', linespacing=1.5, va='top')
    for suffix in ('png', 'svg'):
        target = FIG / f'joint_scale_learning.{suffix}'
        options = {'dpi': 190} if suffix == 'png' else {'metadata': {'Date': None}}
        fig.savefig(target, facecolor='white', **options)
    plt.close(fig)


def main():
    data = collect()
    FIG.mkdir(parents=True, exist_ok=True)
    names = ['joint_scale_learning.png', 'joint_scale_learning.svg',
             'FIGURE_DATA.json', 'FIGURE_MANIFEST.json']
    existing = [FIG / name for name in names if (FIG / name).exists()]
    if existing:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history = FIG / 'history' / stamp
        history.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, history / path.name)
    save_json(FIG / 'FIGURE_DATA.json', data)
    render(data)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    script=relative(__file__), script_sha256=sha(__file__),
                    data_sha256=sha(FIG / 'FIGURE_DATA.json'), sources=data['sources'],
                    outputs={name: sha(FIG / name) for name in names[:2]},
                    python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
                    scientific_scope='Frozen-result arithmetic and rendering only; no scientific random draws, fits, integrals, or new intervals',
                    visual_QA='Separate FIGURE_REVIEW.md records actual image inspection')
    save_json(FIG / 'FIGURE_MANIFEST.json', manifest)
    print(json.dumps({'created': [relative(FIG / name) for name in names],
                      'manifest_sha256': sha(FIG / 'FIGURE_MANIFEST.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
