"""Plot frozen new-data results; generates no observations, scores or tests."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_sequential_generalization'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (BASE/'VERSION_LOCK.json').exists() and not (BASE/'FIGURES.json').exists()
    r=json.loads((BASE/'RESULTS.json').read_text(encoding='utf8'));kl=np.array(r['KL_by_data_repeat_arm']).mean(axis=1)*1000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'scbmseq'})
    fig,axes=plt.subplots(1,3,figsize=(16,5.4),gridspec_kw={'width_ratios':[1.3,1.35,1]});colors=['#ae594a','#277f82','#b48c23','#53677e'];jitter=np.linspace(-.08,.08,16)
    x=[0,1,2,3.35]
    for d,row in enumerate(kl):axes[0].plot(np.array(x[:3])+jitter[d],row[:3],color='.85',lw=.8,zorder=1)
    for a in range(4):
        axes[0].scatter(x[a]+jitter,kl[:,a],s=24,color=colors[a],zorder=2)
        axes[0].plot([x[a]-.14,x[a]+.14],[kl[:,a].mean()]*2,color='black',lw=2,zorder=3)
    axes[0].axhline(r['identity_reference_risk']*1000,ls='--',color='#915a91',lw=1.2,label='Untrained identity I')
    axes[0].set_xticks(x,['IN4','OUT4','SEQ4','FACT\n(reference)']);axes[0].set_ylabel('Observation joint KL × 1000');axes[0].set_ylim(bottom=0)
    axes[0].set_title('(a) Risk on 16 new datasets',loc='left');axes[0].legend(fontsize=8,loc='upper right');axes[0].set_xlabel('Each dot averages 2 training quartets')
    labels=['OUT4 − IN4','SEQ4 − OUT4']
    for j,label in enumerate(labels):
        for k,(rows,color,text) in enumerate([(r['training_primary'],'#277f82','Risk'),(r['evaluation_primary'],'#715493','Raw P score')]):
            row=rows[j];m=row['mean']*1000;err=np.array([[row['mean']-row['lower']],[row['upper']-row['mean']]])*1000
            axes[1].errorbar(m,j+(k-.5)*.18,xerr=err,fmt='o',capsize=4,color=color,label=text if j==0 else None,ms=6)
    axes[1].axvline(0,color='.6',lw=.8);axes[1].set_yticks([0,1],labels);axes[1].set_ylim(1.5,-.6);axes[1].set_title('(b) Risk and original-score contrasts',loc='left');axes[1].set_xlabel('Difference × 1000; lower is better');axes[1].legend(fontsize=8,loc='best')
    for j,row in enumerate(r['gap_derived']):
        m=row['mean']*1000;err=np.array([[row['mean']-row['lower']],[row['upper']-row['mean']]])*1000
        axes[2].errorbar(m,j,xerr=err,fmt='o',capsize=4,color='#a36642',ms=6)
    axes[2].axvline(0,color='.6',lw=.8);axes[2].set_yticks([0,1],labels);axes[2].set_ylim(1.5,-.6);axes[2].set_title('(c) Expected log-estimation bias',loc='left');axes[2].set_xlabel('ΔJ = ΔF − ΔRisk, × 1000\nIntervals derived from the two families')
    fig.text(.5,.027,'IN4 / OUT4 / SEQ4 match prior inputs and score calls; update counts are 1 / 4 / 4 per macrostep. FACT is a quality reference.\nRisk q5 and raw P q2 use separate 97.5% simultaneous development t families, each with 16 independent new-data units. Numerical-reference error is excluded.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.12,1,.99]);folder=BASE/'figures';folder.mkdir(exist_ok=True);paths=[]
    for ext in ['png','svg']:
        p=folder/f'new_data_sequential.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'matplotlib'});paths.append(p)
    plt.close(fig)
    rec=dict(source_sha256=sha(__file__),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [BASE/'RESULTS.json',BASE/'CONFIG.json']},outputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},new_RNG=0,new_model_integrals=0,new_training=0,new_intervals_or_tests=0)
    with (BASE/'FIGURES.json').open('x',encoding='utf8') as f:json.dump(rec,f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
