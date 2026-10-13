"""Prespecified frozen-panel figures from completed MC statistics only.

The diagnostic band and P4 interval are reused verbatim. No new confidence
family, candidate selection, parameter fit, random input or integral is added.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation'
DEST=OUT/'figures'
SOURCE=Path(__file__).resolve()
COLORS={'learned':'#386FA7','half':'#B68127','fact':'#168472','reference':'#404047','selected':'#B68127'}
LABELS={'learned':r'Learned scales ($\alpha=1$)','half':r'Half scales ($\alpha=1/2$)',
        'fact':'Factorized','reference':'Likelihood (numerical reference)'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))


def save(p,value):
    with Path(p).open('x',encoding='utf8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def verify(inputs):
    for name,digest in inputs.items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name


def close(a,b):assert np.allclose(a,b,rtol=3e-12,atol=3e-14),(a,b)


def load_complete():
    assert not (OUT/'VERSION_LOCK.json').exists(),'Package already sealed'
    for phase in ('EVAL','REFERENCE','STATISTICS'):
        assert not (OUT/phase/'ERROR.json').exists() and not (OUT/phase/'FAILURE.json').exists()
    files=[OUT/'MODEL_SELECTION.json',OUT/'EVAL/RESULTS.json',OUT/'EVAL/FREEZE.json',OUT/'EVAL/STATE.json',
           OUT/'REFERENCE/RESULTS.json',OUT/'REFERENCE/FREEZE.json',
           OUT/'STATISTICS/RESULTS.json',OUT/'STATISTICS/FREEZE.json']
    selection,evaluation,ef,state,reference,rf,stats,sf=[read(p) for p in files]
    assert state['status']=='complete' and state['results_sha256']==sha(files[1])
    assert evaluation['freeze_sha256']==sha(files[2])
    assert evaluation['selection_sha256']==reference['selection_sha256']==stats['selection_sha256']==sha(files[0])
    assert stats['evaluation_results_sha256']==sha(files[1]) and stats['reference_results_sha256']==sha(files[4])
    assert reference['freeze_sha256']==sha(files[5]) and stats['freeze_sha256']==sha(files[7])
    verify(sf['inputs']);verify(rf['inputs'])
    assert stats['policy']==sf['policy'] and sf['postprocessing_authorized']
    assert stats['source_sha256']==sf['source_sha256']==ef['summary_source_sha256']==sha(ROOT/'script/scbmeval_summary.py')
    assert reference['source_sha256']==rf['source_sha256']==ef['reference_source_sha256']==sha(ROOT/'script/scbmeval_reference.py')
    assert [r['replicate'] for r in evaluation['records']]==list(range(32))
    assert len(selection['models'])==16 and len(selection['grid_t'])==21
    assert len(stats['primary_family'])==stats['policy']['primary_family_size']==4
    assert stats['policy']['diagnostic_family_size']==7480
    inputs={rel(p):sha(p) for p in files};inputs.update(sf['inputs']);inputs.update(rf['inputs'])
    return selection,reference,stats,inputs


def points(selection,reference,stats):
    grid=np.asarray(selection['grid_t'],dtype=float)
    bands=stats['diagnostic_bands']['relative_score']
    for field in ('mean','SE','lower','upper'):assert np.asarray(bands[field]).shape==(17,5,22,2)
    ce=np.asarray(reference['relative_cross_entropy'],dtype=float)
    assert ce.shape==(16,5,22,2) and np.isfinite(ce).all()
    panels=[]
    for target_index,target in enumerate(['P','D']):
        curves=[]
        for profile_index,profile in enumerate(['learned','half','fact']):
            row=dict(profile=profile,profile_index=profile_index)
            for field in ('mean','SE','lower','upper'):
                # Index16 is the prespecified equal-weight old-model panel.
                # Index21 mixes heterogeneous actual t values and is excluded.
                row[field]=np.asarray(bands[field])[16,profile_index,:21,target_index].tolist()
            curves.append(row)
        likelihood=ce[:,0,:21,target_index].mean(axis=0)
        for i in (1,2):close(ce[:,i,:21,target_index],ce[:,0,:21,target_index])
        panels.append(dict(target=target,target_index=target_index,curves=curves,
                           likelihood_relative_reference=likelihood.tolist()))
    chosen=stats['selected'];indices=chosen['selected_point_indices'];selected_t=chosen['selected_t']
    assert len(indices)==len(selected_t)==16
    actual=[m['LAST']['t'] for m in selection['models']]
    expected=[actual[i] if j==21 else float(grid[j]) for i,j in enumerate(indices)]
    close(selected_t,expected)
    gains=np.asarray(chosen['validation_per_model_gains'],dtype=float)
    assert gains.shape==(16,16) and np.isfinite(gains).all()
    p4=next(r for r in stats['primary_family'] if r['name']=='P4_LAST_conditional_selected_improvement')
    assert p4['replicates']==16 and p4['df']==15
    close(p4['replicate_values'],gains.mean(axis=1));close(p4['mean'],gains.mean())
    return dict(grid_t=grid.tolist(),relative_score_panels=panels,
        diagnostic_band=dict(source_field='diagnostic_bands.relative_score',panel_index=16,
            plotted_point_indices=list(range(21)),excluded_heterogeneous_actual_point_index=21,
            family_size=7480,critical_t=bands['critical_t'],replicates=32,
            scope='Prespecified simultaneous diagnostic t approximation, not a newly fitted interval or exact coverage statement.'),
        likelihood_reference_qualified=reference['qualification_passed'],
        LAST=dict(model_indices=list(range(16)),actual_t=actual,selected_point_indices=indices,selected_t=selected_t,
            discovery_replicates=chosen['selection_replicates'],validation_replicates=chosen['validation_replicates'],
            validation_per_model_gains=gains.tolist(),descriptive_mean_gain_per_model=gains.mean(axis=0).tolist(),
            overall_P4=p4,gain_definition='F_D(original LAST t; fixed LAST scales) minus F_D(selected candidate t; same fixed scales)',
            scope=chosen['scope']),
        original_primary_family=stats['primary_family'],
        statistical_scope='MC units are the 32 fresh evaluation groups, with 16 reserved for P4 validation. The historical 16 models are a fixed paired panel, not independent fresh training replications. Original primary and diagnostic families remain separate.',
        scientific_scope='Frozen low-dimensional models and mathematical finite-particle expectations; not full SCBM training, new optimization trajectories, or a continuous optimality certificate.')


def style():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'grid.alpha':.17,'svg.hashsalt':'scbmeval-prespecified-two-figures','savefig.facecolor':'white'})


def export(fig,name):
    outputs={}
    for suffix in ('png','svg'):
        p=DEST/f'{name}.{suffix}'
        assert not p.exists(),'Archive existing figures/source/metadata before any display revision'
        fig.savefig(p,dpi=235,metadata={'Date':None} if suffix=='svg' else None)
        outputs[rel(p)]=sha(p)
    plt.close(fig);return outputs


def objectives_figure(p):
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.35),sharey=True)
    fig.subplots_adjust(left=.082,right=.985,top=.84,bottom=.29,wspace=.21)
    grid=np.asarray(p['grid_t'])
    for ax,panel in zip(axes,p['relative_score_panels']):
        for curve in panel['curves']:
            color=COLORS[curve['profile']]
            ax.fill_between(grid,curve['lower'],curve['upper'],color=color,alpha=.13,linewidth=0)
            ax.plot(grid,curve['mean'],color=color,linewidth=1.8,label=LABELS[curve['profile']])
        ax.plot(grid,panel['likelihood_relative_reference'],color=COLORS['reference'],linestyle='--',linewidth=1.5,label=LABELS['reference'])
        ax.axhline(0,color='#777777',linewidth=.6);ax.axvline(0,color='#999999',linestyle=':',linewidth=.7)
        ax.set(title='(a) Population target' if panel['target']=='P' else '(b) Conditional on old data',
               xlabel='Correlation amplitude increment, t',xticks=[-.75,-.375,0,.375,.75],xlim=(-.75,.75))
        ax.grid(axis='y');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    axes[0].set_ylabel(r'Panel-mean relative objective: $F(t)-F(0)$')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.092),ncol=2,frameon=False)
    fig.suptitle('Frozen TAIL scales: population and data-conditional objectives',y=.985,fontsize=11)
    fig.text(.5,.039,'Shading reuses the prespecified 7,480-entry diagnostic t family (32 new MC groups). All curves use the same 21 t values.',
             ha='center',va='bottom',fontsize=7.4)
    status='passed' if p['likelihood_reference_qualified'] else 'FAILED'
    fig.text(.5,.012,f'Actual-output points are excluded. Likelihood reference qualification: {status}; no exact quadrature certificate is claimed.',
             ha='center',va='bottom',fontsize=7.3)
    return export(fig,'conditional_population_objectives')


def validation_figure(p):
    last=p['LAST'];x=np.arange(16);actual=np.asarray(last['actual_t']);selected=np.asarray(last['selected_t'])
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.35),gridspec_kw={'width_ratios':[1.,1.1]})
    fig.subplots_adjust(left=.082,right=.985,top=.84,bottom=.30,wspace=.28)
    for i in x:axes[0].plot([i,i],[actual[i],selected[i]],color='#999999',linewidth=.9,zorder=1)
    axes[0].scatter(x,actual,facecolors='white',edgecolors=COLORS['learned'],s=36,linewidth=1.2,zorder=2,label='Original LAST')
    axes[0].scatter(x,selected,color=COLORS['selected'],s=21,marker='D',zorder=3,label='Selected candidate')
    axes[0].axhline(0,color='#888888',linewidth=.7,linestyle=':')
    axes[0].set(title='(a) Selection on the first 16 MC groups',ylabel='Physical t',xlabel='Frozen model index',
                xticks=[0,3,6,9,12,15],xlim=(-.6,15.6),ylim=(-.79,.79))
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,fontsize=7.8)
    gains=np.asarray(last['descriptive_mean_gain_per_model']);p4=last['overall_P4'];xx=18
    axes[1].scatter(x,gains,color=COLORS['learned'],s=25,zorder=3)
    axes[1].axhline(0,color='#555555',linewidth=.8)
    axes[1].axvline(16.5,color='#bbbbbb',linewidth=.7,linestyle=':')
    axes[1].errorbar(xx,p4['mean'],yerr=[[p4['mean']-p4['lower']],[p4['upper']-p4['mean']]],
                    fmt='D',color=COLORS['selected'],markersize=6,elinewidth=1.5,capsize=4,zorder=4)
    axes[1].set(title='(b) Independent validation groups 16–31',ylabel='Conditional score gain (positive = improvement)',
                xlabel='Frozen model index; panel mean at right',xticks=[0,3,6,9,12,15,18],
                xticklabels=['0','3','6','9','12','15','P4'],xlim=(-.6,19.))
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    for ax in axes:ax.grid(axis='y')
    fig.suptitle('Frozen LAST scales: candidate selection and independent validation',y=.985,fontsize=11)
    fig.text(.5,.065,'Score gain = original LAST loss minus selected-candidate loss, keeping each model’s LAST scales fixed.',
             ha='center',va='bottom',fontsize=7.5)
    fig.text(.5,.036,'Per-model gains are descriptive means. The overall bar reuses the original P4 interval across new MC groups, not across old models.',
             ha='center',va='bottom',fontsize=7.3)
    fig.text(.5,.010,'All candidate choices and validation gains are retained; the finite candidate comparison is not a continuous optimum or retraining.',
             ha='center',va='bottom',fontsize=7.3)
    return export(fig,'LAST_selection_validation')


def run():
    assert not (OUT/'FIGURES.json').exists()
    source_hash=sha(SOURCE)
    selection,reference,stats,inputs=load_complete();p=points(selection,reference,stats)
    style();point_path=OUT/'FIGURE_POINTS.json';save(point_path,p)
    outputs=objectives_figure(p);outputs.update(validation_figure(p))
    assert sha(SOURCE)==source_hash;verify(inputs)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=rel(SOURCE),source_sha256=source_hash,
        inputs=inputs,points={rel(point_path):sha(point_path)},figures=outputs,figure_count=2,
        new_random_draws=0,new_integrals=0,new_fits=0,new_candidate_selections=0,new_intervals=0,new_statistical_families=0,
        original_primary_family_size=4,original_diagnostic_family_size=7480,
        reference_qualification_passed=reference['qualification_passed'],
        excluded_actual_point_index=21,main_curve_grid_points=21,
        statistical_scope=p['statistical_scope'],scientific_scope=p['scientific_scope'],
        visual_QA='Separate actual PNG viewing is required; rendering alone is not visual approval')
    save(OUT/'FIGURES.json',metadata);print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run']);parser.parse_args();run()
