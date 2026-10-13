"""Render frozen result summaries; no stochastic input or scientific refit."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_bank_averaging'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'VERSION_LOCK.json').exists() and not (OUT/'FIGURES.json').exists(),'Refuse to overwrite an existing figure record or sealed package'
    r=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'));rho=np.array(r['rho_by_data_repeat_arm']);kl=np.array(r['KL_by_data_repeat_arm'])
    risk=kl.mean(axis=0)*1000;distances=(.5*((rho[:,::2]-rho[:,1::2])**2).sum(axis=3)).mean(axis=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'scbmavg-fixed-target'})
    fig,axes=plt.subplots(1,3,figsize=(12.4,4.9),gridspec_kw={'width_ratios':[1,1,1.15]});colors=['#ad4f42','#247b81']
    for ax,values,title,ylabel,row,mult in [(axes[0],risk,'(a) Conditional learning risk','Joint KL × 1000',r['training_primary'][0],1000),(axes[1],distances,'(b) Within-data parameter dispersion','Half squared parameter distance',r['training_primary'][1],1)]:
        offsets=np.linspace(-.075,.075,len(values))
        for j,v in enumerate(values):ax.plot([offsets[j],1+offsets[j]],v,color='.82',lw=.8,zorder=1)
        for a in range(2):
            ax.scatter(a+offsets,values[:,a],color=colors[a],s=20,zorder=2)
            ax.plot([a-.12,a+.12],[values[:,a].mean()]*2,color='black',lw=2,zorder=3)
        ax.set_xticks([0,1],['L1','L4']);ax.set_ylabel(ylabel)
        caption=f"Δ = {row['mean']*mult:+.3f}  [{row['lower']*mult:+.3f}, {row['upper']*mult:+.3f}]"
        ax.set_title(title+'\n'+caption,loc='left',fontsize=10,linespacing=1.5)
        ax.set_xlim(-.3,1.3);ax.set_xlabel('16 prior columns' if ax is axes[0] else '8 disjoint column pairs')
    rows=r['evaluation_primary'];m=np.array([x['mean'] for x in rows])*1000;err=np.array([[x['mean']-x['lower'] for x in rows],[x['upper']-x['mean'] for x in rows]])*1000
    axes[2].errorbar(m,[0,1],xerr=err,fmt='o',capsize=4,color='#604e87',ms=6);axes[2].axvline(0,color='.6',lw=.8)
    axes[2].set_yticks([0,1],['Data D','Population P']);axes[2].invert_yaxis();axes[2].set_ylim(1.6,-.6)
    axes[2].set_xlabel('Raw score: L4 − L1, × 1000\nnegative means lower L4 score');axes[2].set_title('(c) Independent finite-target scores',loc='left',fontsize=11)
    fig.text(.5,.018,'Four fixed old datasets, K = 100, 512 updates. L4 uses four times the sampling per update.\nSeparate 97.5% simultaneous development t families: training (a,b), and conditional-on-trained-panel evaluation (c).',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,.96]);folder=OUT/'figures';folder.mkdir(exist_ok=True);outputs=[]
    for ext in ['png','svg']:
        p=folder/f'outer_averaging.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'matplotlib'});outputs.append(p)
    plt.close(fig)
    record=dict(source_sha256=sha(Path(__file__)),inputs={'artifacts/scbm_bank_averaging/RESULTS.json':sha(OUT/'RESULTS.json')},outputs={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},new_RNG=0,new_model_integrals=0,new_training=0,new_intervals=0,scope='Previously frozen intervals and paired risk/dispersion units; no new inference.')
    with (OUT/'FIGURES.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
