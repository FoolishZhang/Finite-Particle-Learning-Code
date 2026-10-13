"""All-case figures from the completed fixed-budget experiment; no RNG."""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/calibration_precision';DEST=OUT/'figures'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (OUT/'FIGURES.json').exists();DEST.mkdir(exist_ok=True)
    a=read(OUT/'ANALYSIS.json');tables={(v['case'],v['theta'],v['B'],v['method']):v for v in a['tables']}
    cases=[f'n{n}_K{k}' for n in (62,254) for k in (64,256)];theta=[0.,.3,.6]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'curvcal-fixed-v1'})
    methods=['raw','asymquad','fullfinite','localquad','marg','exact'];colors=['#ad4142','#9262ae','#226bb0','#ed922c','#19846b','#333333']
    labels=['Raw','Asymptotic quadratic','Full finite correction','Local quadratic','Marginalized','Same-data MLE'];files=[]
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,c in zip(axes.flat,cases):
        for m,col,lab in zip(methods,colors,labels):
            y=[tables[c,th,512,m]['reference_joint_KL']['mean'] for th in theta]
            ax.plot(theta,y,color=col,label=lab,lw=1.6,marker='^' if m=='localquad' else 'o',ms=5,ls='--' if m in ('localquad','exact') else '-')
        ax.set_title(c.replace('_',', ').replace('n','n = ').replace('K','K = '));ax.set_yscale('log');ax.set_xticks(theta);ax.grid(alpha=.18)
    for ax in axes[-1]:ax.set_xlabel('Teacher correlation')
    for ax in axes[:,0]:ax.set_ylabel('Mean reference joint KL')
    handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
    fig.suptitle('Fresh-data cross-case comparison at B = 512',y=.985);fig.subplots_adjust(left=.09,right=.98,top=.92,bottom=.16,hspace=.24,wspace=.14)
    for ext in ('png','svg'):
        p=DEST/f'cross_case_risk.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    cols=['#2774ad','#169c85','#b34957','#99742c'];fig,axes=plt.subplots(1,2,figsize=(12.6,4.8))
    for c,col in zip(cases,cols):
        for m in ('fullfinite','localquad'):
            y=[np.mean([tables[c,th,B,m]['squared_discrepancy_to_data_MLE']['mean'] for th in theta]) for B in (128,512)]
            axes[0].plot([128,512],y,color=col,ls='-' if m=='fullfinite' else '--',marker='o' if m=='fullfinite' else '^',label=c.replace('_',', ') if m=='fullfinite' else None,lw=1.8)
    axes[0].set_xscale('log',base=2);axes[0].set_yscale('log');axes[0].set_xticks([128,512],['128','512']);axes[0].set_xlabel('Independent shared training banks B');axes[0].set_ylabel('Mean squared error to same-data MLE');axes[0].set_title('Solid: full correction; dashed: local quadratic');axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.18)
    allpoints=[]
    for c,col in zip(cases,cols):
        rr=[v for v in a['curvature_diagnostic_rows'] if v['case']==c and v['B']==512 and v['method']=='fullfinite' and v['interior_diagnostic_usable']]
        xs=[v['linearized_shift'] for v in rr];ys=[v['actual_shift'] for v in rr];allpoints+=xs+ys
        axes[1].scatter(xs,ys,color=col,label=c.replace('_',', '),s=23,alpha=.72)
    limit=max(abs(float(v)) for v in allpoints)*1.08;axes[1].plot([-limit,limit],[-limit,limit],ls=':',color='#555555',lw=1.3)
    axes[1].set_xlim(-limit,limit);axes[1].set_ylim(-limit,limit);axes[1].set_aspect('equal',adjustable='box');axes[1].set_xlabel('Same-bank linearization: -gradient / curvature');axes[1].set_ylabel('Actual fitted correlation - data MLE');axes[1].set_title('Interior data MLE, full correction, B = 512');axes[1].grid(alpha=.18)
    fig.tight_layout(pad=1.5)
    for ext in ('png','svg'):
        p=DEST/f'precision_and_curvature.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    obj=dict(source_sha256=sha(__file__),inputs={'artifacts/calibration_precision/ANALYSIS.json':sha(OUT/'ANALYSIS.json')},outputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},new_scientific_computation=False,notes='All four cases shown. Conditional on fixed calibration pools. Right scatter is a same-bank diagnostic, not independent prediction or a risk guarantee; excludes boundary data MLE only.')
    with (OUT/'FIGURES.json').open('x',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'figures':len(files)}))
if __name__=='__main__':main()
