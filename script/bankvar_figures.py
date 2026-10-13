"""Figures from sealed-budget outputs only; no random numbers or extra fits."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bank_precision';DEST=OUT/'figures'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (OUT/'FIGURES.json').exists()
    an=read(OUT/'ANALYSIS.json');local=read(OUT/'LOCAL/MODEL.json');train=read(OUT/'TRAIN/RESULTS.json')
    DEST.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'bank-precision-fixed-v1'})
    bs=[32,128,512];colors=dict(raw='#a33b38',marg='#16836d',asymquad='#9671ad',fullfinite='#1764ad',localquad='#e78922',exact='#333333')
    labels=dict(raw='Raw',marg='Marginalized',asymquad='Asymptotic quadratic',fullfinite='Full finite calibration',localquad='Local quadratic',exact='Same-data MLE')
    table={(r['theta'],r['B'],r['method']):r for r in an['tables']}
    files=[]
    fig,ax=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    for a,th in zip(ax,[0.,.3,.6]):
        for m in colors:
            v=[table[th,B,m]['reference_joint_KL']['mean'] for B in bs]
            a.plot(bs,v,color=colors[m],marker='o' if m!='localquad' else '^',ls='--' if m in ('localquad','exact') else '-',label=labels[m],lw=1.7,ms=5)
        a.set_xscale('log',base=2);a.set_yscale('log');a.set_xticks(bs,[str(x) for x in bs]);a.set_title(f'Teacher correlation {th:g}');a.grid(alpha=.18);a.set_xlabel('Independent shared banks B')
    ax[0].set_ylabel('Mean reference joint KL')
    handles,labs=ax[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02))
    fig.suptitle('Fresh-bank precision study on the same 16 complete datasets',y=.99)
    fig.subplots_adjust(left=.085,right=.99,top=.87,bottom=.29,wspace=.12)
    for ext in ('png','svg'):
        p=DEST/f'risk_by_bank.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.5))
    for m in ('fullfinite','localquad','marg'):
        vals=[np.mean([table[th,B,m]['squared_discrepancy_to_data_MLE']['mean'] for th in (0.,.3,.6)]) for B in bs]
        ax[0].plot(bs,vals,marker='^' if m=='localquad' else 'o',ls='--' if m=='localquad' else '-',label=labels[m],color=colors[m],lw=1.8)
    ax[0].set_xscale('log',base=2);ax[0].set_yscale('log');ax[0].set_xticks(bs,[str(x) for x in bs]);ax[0].set_xlabel('Independent shared banks B');ax[0].set_ylabel('MSE to the same-data MLE');ax[0].set_title('Teacher average within each base stream');ax[0].grid(alpha=.18);ax[0].legend(frameon=False,fontsize=9)
    x=np.array(local['h_values']);y=np.array(local['pooled_c_h']);se=np.array(local['stream_SE_c_h'])
    ax[1].errorbar(x,y,yerr=se,fmt='o-',capsize=5,color=colors['localquad'],label='Local calibration: mean ± 1 SE')
    ax[1].axhline(train['coefficients']['oldTaylor'],color=colors['fullfinite'],ls=':',label='Taylor coefficient of old full curve')
    ax[1].scatter([.06],[local['coefficient']],s=90,facecolors='none',edgecolors='#202020',zorder=5,label='Prespecified deployed h = .06')
    ax[1].set_xticks(x,[str(v) for v in x]);ax[1].set_xlabel('Central-difference step h');ax[1].set_ylabel('Quadratic coefficient c');ax[1].set_title('Six model points, eight independent prior streams');ax[1].grid(alpha=.18);ax[1].ticklabel_format(axis='y',style='plain',useOffset=False);ax[1].legend(frameon=False,fontsize=8,loc='lower left')
    fig.tight_layout(pad=1.6)
    for ext in ('png','svg'):
        p=DEST/f'precision_and_local_coefficient.{ext}';fig.savefig(p,dpi=180,metadata={'Date':None} if ext=='svg' else None);files.append(p)
    plt.close(fig)
    result=dict(source_sha256=sha(__file__),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in (OUT/'ANALYSIS.json',OUT/'LOCAL/MODEL.json',OUT/'TRAIN/RESULTS.json')},outputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},new_scientific_computation=False,notes='KL is a qualified numerical reference; reused data and shared calibration, not fresh-data replication. Calibration error bars are stream SE, not confidence coverage.')
    with (OUT/'FIGURES.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'figures':len(files)}))
if __name__=='__main__':main()
