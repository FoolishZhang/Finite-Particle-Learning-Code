"""Prespecified descriptive figures from completed optimization/reference outputs.

No partial results are read for panel selection. No new random draw, fitted
model, model integral, confidence interval, or statistical family is produced.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_scale_optimization'
DEST=OUT/'figures'
SOURCE=Path(__file__).resolve()
METHODS=['FIXED','FREE','FACT']
COLORS={'FIXED':'#BA4855','FREE':'#386FA7','FACT':'#168472'}
LABELS={'FIXED':'Fixed scales','FREE':'Learned scales','FACT':'Factorized'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf8'))


def save(p,value):
    with Path(p).open('x',encoding='utf8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def verify(bindings):
    for name,digest in bindings.items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name


def close(a,b):
    assert np.allclose(a,b,rtol=3e-12,atol=2e-14),(a,b)


def load_complete():
    assert not (OUT/'VERSION_LOCK.json').exists(),'Package already sealed'
    assert not (OUT/'TRAIN/ERROR.json').exists() and not (OUT/'REFERENCE/FAILURE.json').exists()
    paths=[OUT/'TRAIN/RESULTS.json',OUT/'TRAIN/FREEZE.json',OUT/'TRAIN/STATE.json',
           OUT/'REFERENCE/RESULTS.json',OUT/'REFERENCE/FREEZE.json']
    train,freeze,state,reference,ref_freeze=[read(p) for p in paths]
    assert state['status']=='complete' and state['results_sha256']==sha(paths[0])
    assert train['freeze_sha256']==sha(paths[1]) and train['config']==freeze['config']
    assert train['source_sha256']==freeze['source_sha256']==sha(ROOT/'script/scbmopt_run.py')
    assert train['kernel_sha256']==freeze['kernel_sha256']==sha(ROOT/'script/scbmopt_kernel.py')
    assert reference['training_results_sha256']==sha(paths[0])
    assert reference['freeze_sha256']==sha(paths[4])
    assert reference['source_sha256']==ref_freeze['source_sha256']==freeze['reference_source_sha256']
    assert reference['source_sha256']==sha(ROOT/freeze['reference_source_path'])
    verify(ref_freeze['inputs'])
    assert [r['stream'] for r in train['records']]==list(range(16))
    assert all(r['completed_steps']==512 for r in train['records'])
    assert train['config']['steps']==512 and train['config']['tail_postupdates']==64
    assert train['config']['methods']==METHODS and train['config']['nuisance']==256
    assert len(reference['endpoints'])==96 and len(reference['primary_family'])==3
    assert reference['config']['family_size']==3 and reference['config']['family_alpha']==.05
    assert reference['accounting']['final_outputs']==96
    assert len(reference['records'])==reference['accounting']['unique_endpoints']
    inputs={rel(p):sha(p) for p in paths}
    inputs.update(ref_freeze['inputs'])
    inputs.update({rel(ROOT/'script/scbmopt_run.py'):train['source_sha256'],
                   rel(ROOT/'script/scbmopt_kernel.py'):train['kernel_sha256']})
    tcurves=[];rmscurves=[];tape_inputs={}
    for record in train['records']:
        tape_path=ROOT/record['output_file']
        assert sha(tape_path)==record['output_sha256']
        tape_inputs[record['output_file']]=record['output_sha256']
        with np.load(tape_path,allow_pickle=False) as tape:
            t=np.asarray(tape['t'],dtype=np.float64)
            tau=np.asarray(tape['free_tau'],dtype=np.float64)
            assert t.shape==(3,513) and tau.shape==(513,256)
            assert np.isfinite(t).all() and np.isfinite(tau).all()
            assert (abs(t)<=.75).all() and (tau>0).all()
            assert np.all(t[:,0]==0.) and np.all(tau[0]==1.)
            for mi,method in enumerate(METHODS):
                close(record['outputs'][method]['TAIL']['t'],t[mi,-64:].mean())
                close(record['outputs'][method]['LAST']['t'],t[mi,-1])
            close(record['outputs']['FREE']['TAIL']['scales'],tau[-64:].mean(axis=0))
            close(record['outputs']['FREE']['LAST']['scales'],tau[-1])
            assert all(record['outputs']['FIXED'][kind]['scales']==[1.]*256 for kind in ('TAIL','LAST'))
            assert all(record['outputs']['FACT'][kind]['scales']==[0.]*256 for kind in ('TAIL','LAST'))
            tcurves.append(t)
            rmscurves.append(np.sqrt(np.mean(tau*tau,axis=1)))
    assert len(tape_inputs)==16
    return train,reference,inputs,tape_inputs,np.asarray(tcurves),np.asarray(rmscurves)


def points(train,reference,tcurves,rmscurves):
    risks=[]
    offsets=np.linspace(-.055,.055,16)
    for kind in ('TAIL','LAST'):
        table=[]
        for mi,method in enumerate(METHODS):
            rows=[next(r for r in reference['endpoints'] if r['stream']==s and r['method']==method and r['kind']==kind) for s in range(16)]
            values=[float(r['reference_joint_KL']) for r in rows]
            assert all(math.isfinite(v) for v in values)
            for row in rows:
                close(row['t'],train['records'][row['stream']]['outputs'][method][kind]['t'])
            mean=float(np.mean(values))
            descriptive=next(r for r in reference['descriptive'] if r['method']==method and r['kind']==kind)
            close(mean,descriptive['mean_joint_KL'])
            table.append(dict(method=method,stream_values=values,mean=mean,
                              stream_t=[r['t'] for r in rows],display_x=(mi+offsets).tolist()))
        risks.append(dict(kind=kind,methods=table))
    return dict(postupdate_steps=list(range(513)),method_order=METHODS,stream_order=list(range(16)),
        t_by_stream_method_step=tcurves.tolist(),mean_t_by_method_step=tcurves.mean(axis=0).tolist(),
        FREE_tau_RMS_by_stream_step=rmscurves.tolist(),mean_FREE_tau_RMS_by_step=rmscurves.mean(axis=0).tolist(),
        scale_definition='For each stream and step: sqrt(mean over the 256 physical nuisance standard deviations squared); thick line is the mean of these 16 RMS values.',
        FACT_scale_display=None,FACT_scale_interpretation='NA: analytically eliminated. Saved output zeros are a convention, not learned floor-attaining scales.',
        endpoint_risks=risks,primary_family=reference['primary_family'],
        primary_interpretation_qualified=reference['primary_interpretation_qualified'],
        reference_qualification_passed=reference['qualification_passed'],
        risk_reference_scope=reference['scope'],
        statistical_scope='Original TAIL three-test Bonferroni t development family is reported separately; no intervals or significance tests added by these figures. LAST is descriptive for the same 16 streams.',
        display_offset='Endpoint plots use a deterministic stream offset in [-.055,.055], shared across methods, only to reduce overlap. No numerical response value is changed.',
        scientific_scope='Controlled low-dimensional optimization through the official concept-loss API; not full SCBM encoder/task-head training or a general convergence guarantee.')


def style():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'grid.alpha':.17,'svg.hashsalt':'scbmopt-prespecified-two-figures','savefig.facecolor':'white'})


def export(fig,name):
    output={}
    for suffix in ('png','svg'):
        path=DEST/f'{name}.{suffix}'
        assert not path.exists(),'Preserve the previous source, figures and metadata before any display revision'
        fig.savefig(path,dpi=235,metadata={'Date':None} if suffix=='svg' else None)
        output[rel(path)]=sha(path)
    plt.close(fig)
    return output


def trajectory_figure(p):
    steps=np.asarray(p['postupdate_steps']);t=np.asarray(p['t_by_stream_method_step'])
    rms=np.asarray(p['FREE_tau_RMS_by_stream_step'])
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1))
    fig.subplots_adjust(left=.078,right=.985,top=.845,bottom=.25,wspace=.29)
    for mi,method in enumerate(METHODS):
        for row in t[:,mi]:axes[0].plot(steps,row,color=COLORS[method],alpha=.12,linewidth=.6)
        axes[0].plot(steps,t[:,mi].mean(axis=0),color=COLORS[method],linewidth=2,label=LABELS[method])
    axes[0].axhline(0,color='#555555',linewidth=.8,linestyle=':')
    axes[0].set(title='(a) Correlation parameter',ylabel='Physical t',ylim=(-.79,.79),yticks=[-.75,-.5,-.25,0,.25,.5,.75])
    for row in rms:axes[1].plot(steps,row,color=COLORS['FREE'],alpha=.18,linewidth=.7)
    axes[1].plot(steps,rms.mean(axis=0),color=COLORS['FREE'],linewidth=2)
    axes[1].axhline(1,color='#888888',linewidth=.8,linestyle=':')
    axes[1].set(title='(b) Learned nuisance scales',ylabel=r'Physical scale RMS: $(n^{-1}\sum_i\tau_i^2)^{1/2}$')
    for ax in axes:
        ax.set(xlabel='Completed parameter updates',xticks=[0,128,256,384,512],xlim=(0,512))
        ax.grid(axis='y')
    fig.legend(handles=[Line2D([0],[0],color=COLORS[m],linewidth=2,label=LABELS[m]) for m in METHODS],
               loc='lower center',bbox_to_anchor=(.5,.084),ncol=3,frameon=False)
    fig.suptitle('Optimization paths under shared data and prior inputs',y=.985,fontsize=11)
    fig.text(.5,.02,'Thick lines: means across 16 streams. Faint lines: every stream. Descriptive trajectories; factorized scales are NA.',
             ha='center',va='bottom',fontsize=7.5)
    return export(fig,'optimization_paths')


def risk_figure(p):
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.05),sharey=True)
    fig.subplots_adjust(left=.086,right=.985,top=.845,bottom=.27,wspace=.22)
    for ax,panel in zip(axes,p['endpoint_risks']):
        rows=panel['methods']
        for stream in range(16):
            x=[r['display_x'][stream] for r in rows];y=[r['stream_values'][stream] for r in rows]
            ax.plot(x,y,color='#777777',alpha=.22,linewidth=.65,zorder=1)
        for mi,row in enumerate(rows):
            color=COLORS[row['method']]
            ax.scatter(row['display_x'],row['stream_values'],color=color,alpha=.68,s=18,zorder=2)
            ax.scatter(mi,row['mean'],color=color,edgecolors='white',linewidth=.6,marker='D',s=55,zorder=4)
            ax.plot([mi-.13,mi+.13],[row['mean'],row['mean']],color=color,linewidth=1.4,zorder=3)
        title='(a) TAIL: average of last 64 states' if panel['kind']=='TAIL' else '(b) LAST: final state (descriptive)'
        ax.set(title=title,xticks=[0,1,2],xticklabels=[LABELS[m] for m in METHODS],xlim=(-.23,2.23))
        ax.grid(axis='y');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    axes[0].set_ylabel(r'Reference joint KL $D(Q_*\Vert Q_t)$')
    fig.suptitle('Paired model risks from the same 16 optimization streams',y=.985,fontsize=11)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#777777',linewidth=.6,markersize=4,label='Paired stream'),
                        Line2D([0],[0],marker='D',color='#444444',linestyle='none',markersize=6,label='Mean')],
               loc='lower center',bbox_to_anchor=(.5,.103),ncol=2,frameon=False)
    qualification='passed' if p['reference_qualification_passed'] else 'FAILED; risk interpretation unqualified'
    fig.text(.5,.044,f'Numerical PMF qualification: {qualification}. Original three-test TAIL family is reported separately; no added intervals.',
             ha='center',va='bottom',fontsize=7.2)
    fig.text(.5,.014,'TAIL and LAST reuse the same streams. Small deterministic horizontal offsets separate paired points; all risk values are unchanged.',
             ha='center',va='bottom',fontsize=7.2)
    return export(fig,'paired_endpoint_risk')


def run():
    assert not (OUT/'FIGURES.json').exists(),'Figure outputs already exist'
    source_hash=sha(SOURCE)
    train,reference,inputs,tapes,tcurves,rmscurves=load_complete()
    p=points(train,reference,tcurves,rmscurves)
    style();point_path=OUT/'FIGURE_POINTS.json';save(point_path,p)
    output=trajectory_figure(p);output.update(risk_figure(p))
    assert sha(SOURCE)==source_hash
    verify(inputs);verify(tapes)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=rel(SOURCE),source_sha256=source_hash,
        inputs=inputs,input_tape_NPZ_sha256=tapes,points={rel(point_path):sha(point_path)},figures=output,
        figure_count=2,trajectory_streams=16,trajectory_methods=3,trajectory_states=513,
        endpoint_outputs=96,read_local_tapes=16,input_NPZ_not_read=True,
        reference_qualification_passed=reference['qualification_passed'],
        new_scientific_RNG_draws=0,new_model_integrals=0,new_fits=0,new_intervals=0,new_test_families=0,
        statistical_scope=p['statistical_scope'],scientific_scope=p['scientific_scope'],
        visual_QA='Separate actual PNG viewing remains required; successful rendering is not visual approval')
    save(OUT/'FIGURES.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run']);parser.parse_args();run()
