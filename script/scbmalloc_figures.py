"""Scientific figure from frozen matched-budget readouts only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_budget_allocation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'VERSION_LOCK.json').exists() and not (OUT/'FIGURES.json').exists()
    r=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'));data=json.loads((OUT/'DATA.json').read_text(encoding='utf-8'))
    rho=np.array(r['rho_by_data_repeat_arm']);e=rho-data['truth_rho'];kl=np.array(r['KL_by_data_repeat_arm']).mean(axis=0)*1000
    dist=(.5*((e[:,::2]-e[:,1::2])**2).sum(axis=3)).mean(axis=0);cross=(e[:,::2]*e[:,1::2]).sum(axis=3).mean(axis=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'scbmalloc'})
    fig,axes=plt.subplots(1,4,figsize=(15.4,4.9),gridspec_kw={'width_ratios':[1,1,1,1.25]});colors=['#ad4f42','#247b81']
    items=[(kl,'(a) Observation risk','Joint KL × 1000',1000,'16 prior-quartet columns'),(dist,'(b) Conditional dispersion','Half squared distance',1,'8 disjoint column pairs'),(cross,'(c) Conditional center error','Cross-product estimate',1,'8 disjoint column pairs')]
    for ax,(v,title,ylabel,mult,xlabel),row in zip(axes,items,r['training_primary']):
        jitter=np.linspace(-.075,.075,len(v))
        for j,x in enumerate(v):ax.plot([jitter[j],1+jitter[j]],x,color='.82',lw=.8,zorder=1)
        for a in range(2):
            ax.scatter(a+jitter,v[:,a],color=colors[a],s=19,zorder=2);ax.plot([a-.12,a+.12],[v[:,a].mean()]*2,color='black',lw=2,zorder=3)
        ax.set_xlim(-.3,1.3);ax.set_xticks([0,1],['IN4','OUT4']);ax.set_ylabel(ylabel);ax.set_xlabel(xlabel)
        ax.set_title(title+'\n'+f"Δ = {row['mean']*mult:+.3f} [{row['lower']*mult:+.3f}, {row['upper']*mult:+.3f}]",loc='left',fontsize=10,linespacing=1.5)
        if v.min()<0:ax.axhline(0,color='.6',lw=.6)
    q=r['evaluation_primary'];m=np.array([x['mean'] for x in q])*1000;err=np.array([[x['mean']-x['lower'] for x in q],[x['upper']-x['mean'] for x in q]])*1000
    axes[3].errorbar(m,[0,1],xerr=err,fmt='o',capsize=4,color='#604e87',ms=6);axes[3].axvline(0,color='.6',lw=.8)
    axes[3].set_yticks([0,1],['Data D','Population P']);axes[3].set_ylim(1.6,-.6);axes[3].set_xlabel('Raw score OUT4 − IN4, × 1000\npositive: OUT4 scores worse');axes[3].set_title('(d) Independent target scores',loc='left',fontsize=10)
    fig.text(.5,.018,'Matched prior inputs and concept-loss calls per strategy; optimizer updates and wall time are not asserted equal.\nSeparate 97.5% simultaneous development t families: training q3 (a–c), conditional-on-output evaluation q2 (d). Center-error estimates may be negative.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.11,1,.98]);folder=OUT/'figures';folder.mkdir(exist_ok=True);paths=[]
    for ext in ['png','svg']:
        p=folder/f'budget_allocation.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'matplotlib'});paths.append(p)
    plt.close(fig)
    with (OUT/'FIGURES.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(source_sha256=sha(Path(__file__)),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'RESULTS.json',OUT/'DATA.json']},outputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},new_RNG=0,new_model_integrals=0,new_training=0,new_intervals=0),f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
