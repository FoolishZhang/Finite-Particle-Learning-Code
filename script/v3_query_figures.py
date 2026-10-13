"""Static figures from frozen results only; no sampling or fitting."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3_query';FIG=OUT/'figures'
FAMS=['product','cardinality','ising','linear_ar','count_ar']
LABELS=['Product','Unary + count','Pairwise Ising','Linear AR','Count-state AR']
def read(name):return json.loads((OUT/name).read_text('utf-8'))
def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def population():
    rows=read('A_RESULTS.json')['rows'];teachers=['indep','count','allocation','allocation_sym','parity']
    keys=[(teacher,code) for teacher in teachers for code in ['native','flip21']]
    fig,axes=plt.subplots(1,2,figsize=(13,7.2))
    for ax,metric,title in zip(axes,['joint','query'],['Full joint KL (nats / vector)','Conditional excess log loss (nats / bit, k = 7)']):
        z=np.zeros((len(keys),len(FAMS)))
        for i,(teacher,code) in enumerate(keys):
            for j,fam in enumerate(FAMS):
                r=next(r for r in rows if r['teacher']==teacher and r['encoding']==code and r['family']==fam)
                z[i,j]=max(0.,r['joint_KL'] if metric=='joint' else r['queries']['7']['conditional_excess_KL'])
        im=ax.imshow(z,cmap='Blues',vmin=0,vmax=max(z.max(),.1),aspect='auto')
        for i in range(len(keys)):
            for j in range(len(FAMS)):
                value=z[i,j];label='~0' if value<5e-6 else (f'{value:.1e}' if value<.001 else f'{value:.3f}')
                ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if value>z.max()*.6 else 'black')
        ax.set_xticks(range(5),LABELS,rotation=35,ha='right');ax.set_yticks(range(10),[f'{t} / {c}' for t,c in keys]);ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Population fits: model class and label encoding change what can be predicted',fontsize=13)
    fig.tight_layout();save(fig,'population_capability')
def finite():
    data=read('B_RESULTS.json');rows=data['rows'];fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=False)
    colors=['#4d4d4d','#0072b2','#009e73','#d55e00','#cc79a7']
    for ax,world in zip(axes,['independent','dependent']):
        for f,(fam,label,color) in enumerate(zip(FAMS,LABELS,colors)):
            for c,code in enumerate(['native','flip21']):
                vals=[np.mean([r['queries']['7']['conditional_excess_KL'] for r in rows if r['world']==world and r['encoding']==code and r['family']==fam and r['rep']==rep]) for rep in range(16)]
                pos=c+(f-2)*.13
                ax.scatter(pos+np.linspace(-.035,.035,16),vals,s=10,color=color,alpha=.35)
                for rep in range(16):
                    qualified=all(r['qualified'] for r in rows if r['world']==world and r['encoding']==code and r['family']==fam and r['rep']==rep)
                    if not qualified:ax.plot(pos+np.linspace(-.035,.035,16)[rep],vals[rep],'x',color='black',markersize=6)
                ax.plot(pos,np.mean(vals),'_',markersize=15,color=color,label=label if c==0 else None)
        ax.set_xticks([0,1],['Native labels','Partial bit flip']);ax.set_xlim(-.5,1.5);ax.set_ylim(bottom=0)
        ax.set_title(f'Conditionally {world} given X');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Conditional excess log loss (nats / bit, k = 7)')
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Matched conditional means: 16 training replicates, N = 4096; x = numerical gate failure',fontsize=11)
    fig.tight_layout();save(fig,'conditional_mean_control')
def inference():
    with (OUT/'C/SUMMARY.csv').open(encoding='utf-8',newline='') as source:rows=list(csv.DictReader(source))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for d,color in [(8,'#0072b2'),(32,'#009e73'),(64,'#d55e00')]:
        sel=[r for r in rows if int(r['d'])==d and int(r['observed'])==3*d//4 and r['layout']=='mixed']
        x=np.array([int(r['K']) for r in sel]);y=np.array([float(r['mean_excess_Brier']) for r in sel])
        lo=np.array([float(r['lower']) for r in sel]);hi=np.array([float(r['upper']) for r in sel])
        axes[0].errorbar(x,y,yerr=np.array([y-lo,hi-y]),color=color,marker='o',capsize=3,label=f'd={d}, observed={3*d//4}')
    axes[0].set_xscale('log',base=2);axes[0].set_yscale('log');axes[0].set_xticks([16,64,256],['16','64','256'])
    axes[0].set_xlabel('Particles K (nested within each query)');axes[0].set_ylabel('Same-model conditional excess Brier')
    axes[0].set_title('Mixed positions, 75% observed');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    sel=[r for r in rows if int(r['d'])==64 and int(r['observed'])==48 and r['layout']=='mixed']
    for r in sel:
        x=1000*float(r['mean_is_path_seconds']);y=float(r['mean_excess_Brier']);lo=float(r['lower']);hi=float(r['upper'])
        axes[1].errorbar(x,y,yerr=[[y-lo],[hi-y]],color='#d55e00',marker='o',capsize=3)
        axes[1].annotate('IS K='+r['K'],(x,y),xytext=(7,0),textcoords='offset points',fontsize=9)
    x=1000*float(sel[0]['mean_dp_seconds']);axes[1].plot(x,0,'s',color='#0072b2')
    axes[1].annotate('Exact DP',(x,0),xytext=(-50,12),textcoords='offset points',fontsize=9)
    axes[1].set_xlabel('Mean query path time (ms; RNG table excluded)');axes[1].set_ylabel('Same-model conditional excess Brier')
    axes[1].set_title('d=64, 48 observed; descriptive timing');axes[1].set_xlim(.44,.84);axes[1].set_ylim(-.005,.135);axes[1].grid(alpha=.2)
    fig.suptitle('Correct likelihood weighting still has finite-budget error; exact DP is not always faster',fontsize=12)
    fig.tight_layout();save(fig,'same_model_inference')
if __name__=='__main__':population();finite();inference();print('Created A/B/C figures from frozen results; 0 fits or samples.')
