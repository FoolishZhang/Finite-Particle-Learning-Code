"""Plot frozen target contrasts and risks, without scientific recomputation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_six_frozen_targets'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=OUT/'RESULTS.json';r=json.loads(p.read_text(encoding='utf-8'));rows={x['name']:x for x in r['primary_family']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'scbm6eval'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.1,1]})
    labels=['FIXED − truth (P)','FIXED − FACT (P)','FIXED − truth (D)','FIXED − FACT (D)']
    suffix=['P_FIXED_minus_truth','P_FIXED_minus_FACT','D_FIXED_minus_truth','D_FIXED_minus_FACT']
    for profile,color,offset in [('raw','#a5483d',-.11),('FACT','#267579',.11)]:
        rr=[rows[profile+'_'+s] for s in suffix];m=np.array([x['mean'] for x in rr])*1000
        error=np.array([[x['mean']-x['lower'] for x in rr],[x['upper']-x['mean'] for x in rr]])*1000
        axes[0].errorbar(m,np.arange(4)+offset,xerr=error,fmt='o',ms=5,capsize=3,color=color,label=profile+' score')
    axes[0].axvline(0,color='.5',lw=.8);axes[0].set_yticks(range(4),labels);axes[0].invert_yaxis()
    axes[0].set_xlabel('Expected score contrast × 1000\nnegative: FIXED is preferred')
    axes[0].set_title('(a) Frozen panel, independent priors',loc='left',fontsize=11);axes[0].legend(fontsize=9,loc='best')
    risks=np.array([x['reference_joint_KL'] for x in r['candidates']]);x=np.arange(16)
    for i in range(16):axes[1].plot([i,i],risks[[i+2,i+18]]*1000,color='.8',lw=.8,zorder=1)
    axes[1].scatter(x,risks[2:18]*1000,color='#a5483d',s=23,label='FIXED TAIL',zorder=2)
    axes[1].scatter(x,risks[18:34]*1000,color='#267579',s=23,label='FACT TAIL',zorder=2)
    axes[1].axhline(risks[34]*1000,color='#6c5594',ls='--',label='Mean FIXED matrix (probe)')
    axes[1].set_xlabel('Old training stream (0-based)');axes[1].set_ylabel('Integrated joint KL × 1000');axes[1].set_xticks([0,5,10,15])
    axes[1].set_title('(b) Same frozen matrices, integrated risk',loc='left',fontsize=11);axes[1].legend(fontsize=8,loc='best')
    fig.text(.5,.015,'Score bars: simultaneous 95% development t intervals (11-comparison family), conditional on the old panel.\nThe mean matrix combines 16 old outputs; it is a geometric probe, not a single-stream training baseline.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,1]);folder=OUT/'figures';folder.mkdir(exist_ok=True)
    outputs=[]
    for ext in ['png','svg']:
        f=folder/f'frozen_targets.{ext}';fig.savefig(f,dpi=180,metadata={'Date':None} if ext=='svg' else {'Software':'matplotlib'});outputs.append(f)
    plt.close(fig)
    record=dict(source_sha256=sha(Path(__file__)),inputs={p.relative_to(ROOT).as_posix():sha(p)},outputs={f.relative_to(ROOT).as_posix():sha(f) for f in outputs},new_RNG=0,new_model_integrals=0,new_fits=0,new_confidence_intervals=0,scope='Frozen existing simultaneous intervals and numerically qualified risk references replotted.')
    with (OUT/'FIGURES.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
