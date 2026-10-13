"""Fixed descriptive figures from completed, frozen six-correlation results.

This script reads JSON only. It neither imports scientific evaluators nor draws
samples, fits parameters, integrates models, or constructs confidence intervals.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scbm_multicorrelation'
DEST = OUT / 'figures'
SOURCE = Path(__file__).resolve()
METHODS = ['FIXED', 'FREE', 'FACT']
COLORS = {'FIXED': '#BA4855', 'FREE': '#386FA7', 'FACT': '#168472'}
LABELS = {'FIXED': 'Fixed scales', 'FREE': 'Learned scales', 'FACT': 'Factorized'}
PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    with Path(path).open('x', encoding='utf8', newline='\n') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def verify(bindings):
    for name, digest in bindings.items():
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and sha(p) == digest, name


def close(a, b):
    assert np.allclose(a, b, rtol=3e-12, atol=3e-14), (a, b)


def load_complete():
    assert not (OUT / 'VERSION_LOCK.json').exists(), 'Package already sealed'
    for phase in ('TRAIN', 'REFERENCE', 'DIRECTION'):
        assert not (OUT / phase / 'ERROR.json').exists()
        assert not (OUT / phase / 'FAILURE.json').exists()
    names = ['TRAIN/RESULTS.json', 'TRAIN/FREEZE.json', 'TRAIN/STATE.json',
             'REFERENCE/RESULTS.json', 'REFERENCE/FREEZE.json',
             'REFERENCE/PROGRESS.json', 'DIRECTION/RESULTS.json',
             'DIRECTION/FREEZE.json']
    paths = [OUT / n for n in names]
    train, tf, ts, ref, rf, rp, direction, df = [read(p) for p in paths]
    inputs = {rel(p): sha(p) for p in paths}
    assert ts['status'] == 'complete' and ts['results_sha256'] == sha(paths[0])
    assert train['freeze_sha256'] == sha(paths[1])
    assert train['config'] == tf['config']
    assert train['source_sha256'] == tf['source_sha256'] == sha(ROOT / 'script/scbm6_run.py')
    assert train['kernel_sha256'] == tf['kernel_sha256'] == sha(ROOT / 'script/scbm6_kernel.py')
    assert ref['training_results_sha256'] == sha(paths[0])
    assert ref['freeze_sha256'] == sha(paths[4])
    assert ref['source_sha256'] == rf['source_sha256'] == tf['reference_source_sha256']
    assert direction['freeze_sha256'] == sha(paths[7])
    assert direction['source_sha256'] == df['source_sha256']
    assert direction['config'] == df['config']
    for bindings in (rf['inputs'], df['inputs'], tf['official_inputs']):
        verify(bindings)
        inputs.update(bindings)
    inputs[rel(ROOT / 'script/scbm6_run.py')] = train['source_sha256']
    inputs[rel(ROOT / 'script/scbm6_kernel.py')] = train['kernel_sha256']
    assert train['config']['pair_order'] == ref['pair_order'] == direction['config']['pairs'] == PAIRS
    assert train['config']['methods'] == METHODS and train['config']['steps'] == 512
    assert [r['stream'] for r in train['records']] == list(range(16))
    assert all(r['completed_steps'] == 512 for r in train['records'])
    assert len(ref['endpoints']) == ref['accounting']['trained_outputs'] == 96
    assert len(ref['records']) == ref['accounting']['unique_covariances'] == 97
    assert rp['completed_unique_covariances'] == 97 and rp['in_flight'] is None
    assert rp['completed_PMF_integrations'] == ref['accounting']['PMF_integrations'] == 291
    assert ref['qualification_passed'] and all(r['passed'] for r in ref['records'])
    assert len(ref['primary_family']) == 3
    assert [r['replicate'] for r in direction['records']] == list(range(32))
    assert all(r['consumed_banks'] == 512 and r['completed_blocks'] == 8 for r in direction['records'])
    assert direction['config']['matrices'] == ['identity', 'truth']
    assert direction['config']['profiles'] == ['scale1', 'scale02', 'FACT']
    assert len(direction['primary_family']) == 18
    for result in (ref, direction):
        assert result['primary_effects_passed'] == sum(x['effect_passed'] for x in result['primary_family'])
        assert result['primary_precision_passed'] == sum(x['precision_passed'] for x in result['primary_family'])
    return train, ref, direction, inputs


def make_points(train, ref, direction):
    truth = np.array([train['data_model']['covariance'][i][j] for i, j in PAIRS])
    signs = np.sign(truth)
    risk = []
    offsets = np.linspace(-.055, .055, 16)
    for mi, method in enumerate(METHODS):
        rows = [next(r for r in ref['endpoints'] if r['stream'] == s and r['method'] == method and r['kind'] == 'TAIL') for s in range(16)]
        values = np.array([r['reference_joint_KL'] for r in rows])
        rho = np.array([r['rho'] for r in rows])
        for s, row in enumerate(rows):
            close(row['rho'], train['records'][s]['outputs'][method]['TAIL']['rho'])
        original = next(r for r in ref['descriptive'] if r['method'] == method and r['kind'] == 'TAIL')
        close(values.mean(), original['mean_joint_KL'])
        close(rho.mean(axis=0), original['mean_rho'])
        risk.append(dict(method=method,stream_joint_KL=values.tolist(),mean_joint_KL=float(values.mean()),
                         stream_rho=rho.tolist(),mean_rho=rho.mean(axis=0).tolist(),
                         risk_display_x=(mi+offsets).tolist(),
                         rho_display_x=(np.arange(6)[None,:]+(mi-1)*.23+offsets[:,None]*.65).tolist()))
    scalar = {}
    vectors = {}
    for key in ['score', 'centered_score', 'overlap', 'mean_ess', 'mean_inverse_ess']:
        arr = np.array([r['means'][key] for r in direction['records']])
        assert arr.shape == (32,2,3) and np.isfinite(arr).all()
        close(arr.mean(axis=0), direction['descriptive'][key]['mean'])
        scalar[key] = dict(replicate_values=arr.tolist(),mean=arr.mean(axis=0).tolist())
    for key in ['gradient', 'mean_mark_term', 'coupling_term', 'markmc']:
        arr = np.array([r['means'][key] for r in direction['records']])
        assert arr.shape == (32,2,3,6) and np.isfinite(arr).all()
        close(arr.mean(axis=0), direction['descriptive'][key]['mean'])
        vectors[key] = dict(replicate_values=arr.tolist(),mean=arr.mean(axis=0).tolist())
    close(np.array(vectors['gradient']['replicate_values']),
          np.array(vectors['mean_mark_term']['replicate_values'])+np.array(vectors['coupling_term']['replicate_values']))
    projected = {}
    for key in ['gradient', 'mean_mark_term', 'coupling_term']:
        arr = np.array(vectors[key]['replicate_values'])[:,1,:,:]
        per_group = (arr*signs).mean(axis=-1)
        projected[key] = dict(replicate_values=per_group.tolist(),mean=per_group.mean(axis=0).tolist())
    intervals = [r for r in direction['primary_family'] if r['name'] == 'signed_scale1']
    assert [r['pair'] for r in intervals] == PAIRS
    for j, row in enumerate(intervals):
        values = np.array(vectors['gradient']['replicate_values'])[:,1,0,j]*signs[j]
        close(values, row['replicate_values'])
        close(values.mean(), row['mean'])
    return dict(
        pair_order=PAIRS,stream_order=list(range(16)),replicate_order=list(range(32)),
        method_order=METHODS,truth_rho=truth.tolist(),truth_signs=signs.tolist(),
        identity_reference_joint_KL=ref['identity_reference']['reference_joint_KL'],
        training_TAIL=risk,training_primary_family=ref['primary_family'],
        training_effects_passed=ref['primary_effects_passed'],training_precision_passed=ref['primary_precision_passed'],
        direction_matrix_order=['identity','truth'],direction_profile_order=['scale1','scale02','FACT'],
        direction_scalar=scalar,direction_vector=vectors,
        descriptive_fixed_projection=projected,
        projection_definition='(1/6) sum_j sign(truth_rho_j) component_j at truth covariance; not six independent samples',
        displayed_intervals=intervals,all_original_direction_intervals=direction['primary_family'],
        direction_effects_passed=direction['primary_effects_passed'],direction_precision_passed=direction['primary_precision_passed'],
        statistical_units={'training':'16 independent training streams','direction':'32 independent new MC groups, each 512 banks; not six independent pairs'},
        scope=[
            'All 16 TAIL risks and all six physical correlation coordinates are retained.',
            'Risk is the mean of per-stream reference joint KL, never the risk of the mean covariance.',
            'Horizontal offsets are deterministic display offsets only.',
            'Left decomposition is descriptive: no additional interval or statistical gate.',
            'Overlap uses full likelihood particle weights; it is mass minus E(1/ESS), not 1 minus 1/E(ESS).',
            'Right intervals are the six signed scale1 absolute levels from the original 18-test direction family, not absolute values.',
            'Prescribed scale0.2 is distinct from learned FREE scales.',
            'Training and direction confidence families and replication units remain separate.'
        ])


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':11,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
        'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','svg.hashsalt':'scbm6-fixed-descriptive-20261003'})


def finish(fig, name):
    outputs = {}
    for suffix in ('png','svg'):
        path = DEST / f'{name}.{suffix}'
        assert not path.exists(), path
        fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white',
                    metadata={'Software':'scbm6_figures.py'} if suffix=='png' else {'Date':None})
        outputs[rel(path)] = sha(path)
    plt.close(fig)
    return outputs


def training_figure(points):
    fig, axes = plt.subplots(1,2,figsize=(11.1,4.0),gridspec_kw={'width_ratios':[1,1.45]})
    left,right = axes
    rows = points['training_TAIL']
    for s in range(16):
        left.plot([row['risk_display_x'][s] for row in rows],
                  [1000*row['stream_joint_KL'][s] for row in rows],color='#ADB4BA',alpha=.45,lw=.75,zorder=1)
    for mi,row in enumerate(rows):
        method=row['method'];color=COLORS[method]
        left.scatter(row['risk_display_x'],1000*np.array(row['stream_joint_KL']),color=color,s=20,alpha=.8,zorder=3)
        left.scatter([mi],[1000*row['mean_joint_KL']],color=color,marker='D',edgecolor='white',s=70,zorder=4)
    left.axhline(1000*points['identity_reference_joint_KL'],color='#5B6268',lw=1.1,ls='--',label='Identity initialization')
    left.set(xticks=range(3),xticklabels=[LABELS[m] for m in METHODS],ylabel=r'Reference joint KL ($\times 10^{-3}$)',
             title='a  Risk across 16 training streams',ylim=(0,None))
    left.margins(x=.17)
    left.legend(loc='upper right',frameon=False)
    for mi,row in enumerate(rows):
        rho=np.array(row['stream_rho']);x=np.array(row['rho_display_x']);color=COLORS[row['method']]
        right.scatter(x.flatten(),rho.flatten(),s=14,color=color,alpha=.5,zorder=2)
        right.scatter(np.arange(6)+(mi-1)*.23,row['mean_rho'],s=48,color=color,marker='D',edgecolor='white',lw=.6,zorder=4)
    for j,truth in enumerate(points['truth_rho']):
        right.plot([j-.36,j+.36],[truth,truth],color='#242A30',lw=1.5,zorder=3)
    right.axhline(0,color='#CCD1D5',lw=.7,zorder=0)
    right.set(xticks=range(6),xticklabels=[f'{i+1}{j+1}' for i,j in PAIRS],xlabel='Active coordinate pair',
              ylabel=r'TAIL physical correlation $\rho_{ij}$',title='b  Full six-coordinate learning')
    right.legend(handles=[Line2D([0],[0],marker='D',color=COLORS[m],lw=0,label=LABELS[m]) for m in METHODS]+
                 [Line2D([0],[0],color='#242A30',lw=1.5,label='Truth')],loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False)
    right.set_ylim(-.9,1.03)
    right.set_xlim(-.55,5.55)
    for ax in axes:ax.grid(axis='y',color='#E4E7E9',lw=.6,zorder=0)
    fig.subplots_adjust(wspace=.35,bottom=.17,top=.89)
    return finish(fig,'six_correlation_learning')


def direction_figure(points):
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),gridspec_kw={'width_ratios':[1.14,1]})
    left,right=axes;x=np.arange(3)
    projected=points['descriptive_fixed_projection']
    meanmark=1000*np.array(projected['mean_mark_term']['mean'])
    coupling=1000*np.array(projected['coupling_term']['mean'])
    gradient=1000*np.array(projected['gradient']['mean'])
    left.bar(x-.11,meanmark,width=.22,color='#BC6B58',label='Mean-mark term',zorder=3)
    left.bar(x+.11,coupling,width=.22,color='#598CA5',label='Coupling term',zorder=3)
    left.scatter(x,gradient,marker='D',color='#252C33',s=45,label='Total gradient',zorder=4)
    left.axhline(0,color='#636B72',lw=.8)
    overlap=np.array(points['direction_scalar']['overlap']['mean'])[1]
    left.set(xticks=x,xticklabels=[r'$\tau=1$',r'$\tau=0.2$','Factorized'],
             ylabel=r'Signed projection ($\times 10^{-3}$)',title='a  Opposing terms at the true covariance')
    left.set_ylim(-16.8,19.0)
    for j,value in enumerate(overlap):
        left.text(j,-15.5,f'A = {value:.3f}',ha='center',va='center',fontsize=9,color='#545B61')
    left.legend(loc='upper left',frameon=False,ncol=2,columnspacing=1.0,handlelength=1.2)
    rows=points['displayed_intervals'];mean=1000*np.array([r['mean'] for r in rows])
    low=1000*np.array([r['lower'] for r in rows]);high=1000*np.array([r['upper'] for r in rows])
    right.errorbar(np.arange(6),mean,yerr=np.array([mean-low,high-mean]),fmt='o',markersize=5,
                   capsize=3,color='#386FA7',elinewidth=1.5,zorder=3)
    right.axhline(0,color='#636B72',lw=.8)
    right.set(xticks=range(6),xticklabels=[f'{i+1}{j+1}' for i,j in PAIRS],xlabel='Active coordinate pair',
              ylabel=r'$\mathrm{sign}(\rho^\star_{ij})\,\partial F/\partial\rho_{ij}$ ($\times 10^{-3}$)',
              title=r'b  Six signed components at $\tau=1$')
    right.set_xlim(-.5,5.5)
    right.set_ylim(-4.65,.35)
    right.text(.5,.98,'Original 18-test family intervals',transform=right.transAxes,ha='center',va='top',fontsize=9,color='#545B61')
    for ax in axes:ax.grid(axis='y',color='#E4E7E9',lw=.6,zorder=0)
    fig.subplots_adjust(wspace=.39,bottom=.17,top=.89)
    return finish(fig,'direction_decomposition')


def main():
    for path in [OUT/'FIGURE_POINTS.json',OUT/'FIGURES.json']:
        assert not path.exists(),f'Refuse overwrite: {path}'
    train,ref,direction,inputs=load_complete()
    points=make_points(train,ref,direction)
    points['source']=rel(SOURCE);points['source_sha256']=sha(SOURCE);points['inputs']=inputs
    save(OUT/'FIGURE_POINTS.json',points)
    DEST.mkdir(exist_ok=True)
    style()
    outputs=training_figure(points)
    outputs.update(direction_figure(points))
    verify(inputs)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=rel(SOURCE),source_sha256=sha(SOURCE),
        inputs=inputs,points={rel(OUT/'FIGURE_POINTS.json'):sha(OUT/'FIGURE_POINTS.json')},
        figures=outputs,figure_count=2,new_RNG=0,new_model_integrals=0,new_fits=0,new_intervals=0,
        families_separate=True,
        family_status={'training':{'effects':ref['primary_effects_passed'],'precision':ref['primary_precision_passed'],'size':3},
                       'direction':{'effects':direction['primary_effects_passed'],'precision':direction['primary_precision_passed'],'size':18}},
        selected_panels='Two panels per figure, fixed before plotting; all training streams and all six correlation coordinates retained.',
        visual_review='Separate FIGURE_QA.json records actual PNG inspection; generation alone is not visual review.',
        scope=points['scope'])
    save(OUT/'FIGURES.json',metadata)
    print(json.dumps({'figure_count':2,'source_sha256':sha(SOURCE),'figures_sha256':sha(OUT/'FIGURES.json'),
                      'points_sha256':sha(OUT/'FIGURE_POINTS.json'),'figures':outputs},indent=2))


if __name__=='__main__':
    main()
