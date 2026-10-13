"""Publication plot from sealed public scalars only; no scientific arrays."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_8_20261008'
OLD=DOC.parent/'submission_candidate_v1_7_1_20261008'
OUT=ROOT/'artifacts/submission_candidate_v1_8_20261008'
SOURCE=ROOT/'artifacts/asymptotic_bridge_v1_20261008/AUTHOR_READOUT.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()
def main(check=False):
    if check:
        record=json.loads((OUT/'FIGURES.json').read_text(encoding='utf-8'))
        for group in ('sources','outputs'):
            for name,digest in record[group].items():assert sha(ROOT/name)==digest,name
        print(json.dumps(dict(passed=True,new_scientific_computation=False,new_integrals=0)));return
    assert not (OUT/'DELIVERY_MANIFEST.json').exists()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    data=json.loads(SOURCE.read_text(encoding='utf-8'));panel=data['per_n'];x=np.arange(4)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    for lang in ('en','zh'):
        font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc') if lang=='zh' else FontProperties(family='DejaVu Sans')
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained',gridspec_kw={'width_ratios':[1.12,1]})
        def plot(ax,key,label,color,marker,offset):
            mean=np.array([r[key]['mean'] for r in panel]);lo=np.array([r[key]['interval'][0] for r in panel]);hi=np.array([r[key]['interval'][1] for r in panel])
            ax.errorbar(x+offset,mean,yerr=np.vstack((mean-lo,hi-mean)),color=color,marker=marker,
                linewidth=1.2,markersize=4.5,capsize=2.7,label=label)
        labels=('Gap / leading term','Gradient / leading term','Overlap / scale') if lang=='en' else ('目标差 / 首项','梯度 / 首项','重叠量 / 尺度')
        for key,label,color,marker,offset in zip(('A','Bgrad','omega_over_w'),labels,('#16718a','#ce6031','#686868'),('o','s','^'),(-.09,0,.09)):
            plot(axes[0],key,label,color,marker,offset)
        axes[0].axhline(1,color='#9a9a9a',linestyle='--',linewidth=.7);axes[0].axhline(0,color='#aeb7bf',linewidth=.6)
        axes[0].set_ylim(-.25,1.27);axes[0].legend(loc='lower right',prop=font,frameon=False)
        axes[0].set_title('(a) Finite-scale ratios' if lang=='en' else '(a) 有限规模比值',fontproperties=font)
        axes[0].set_ylabel('Normalized value' if lang=='en' else '归一化量',fontproperties=font)
        plot(axes[1],'O',r'$O_n$  overlap' if lang=='en' else r'$O_n$  重叠残差','#16718a','o',-.07)
        plot(axes[1],'J',r'$J_n$  marked coupling' if lang=='en' else r'$J_n$  标记耦合','#8d488b','s',.07)
        mh=data['frozen_references']['m_H45'];axes[1].axhline(mh,color='#9b4944',linestyle=':',linewidth=1)
        axes[1].text(3.15,mh+.00035,r'$m_H$',ha='right',color='#9b4944',fontsize=9)
        axes[1].axhline(0,color='#aeb7bf',linewidth=.6);axes[1].set_ylim(-.0018,.028)
        axes[1].set_title('(b) Fixed-coordinate residual' if lang=='en' else '(b) 固定坐标残差',fontproperties=font)
        axes[1].set_ylabel(r'$D_n/w_n+m_H=O_n+J_n$')
        axes[1].legend(loc='upper right',prop=font,frameon=False)
        for ax in axes:
            ax.set_xticks(x,[str(r['n']) for r in panel]);ax.set_xlabel(r'Nuisance dimension $n$' if lang=='en' else r'干扰维数 $n$',fontproperties=font)
            ax.set_xlim(-.26,3.26);ax.grid(axis='y',color='#e7ebef',linewidth=.5);ax.set_axisbelow(True)
        fig.savefig(DOC/f'figures/bridge_summary_{lang}.png',dpi=300);plt.close(fig)
    oldfigures=sorted(p for p in (OLD/'figures').iterdir() if p.is_file())
    assert len(oldfigures)==20
    for p in oldfigures:assert sha(p)==sha(DOC/'figures'/p.name),p.name
    files=sorted(p for p in (DOC/'figures').iterdir() if p.is_file())
    record=dict(sources={rel(SOURCE):sha(SOURCE),rel(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        outputs={rel(p):sha(p) for p in files},old_20_figures_preserved=True,new_scientific_computation=False,
        new_integrals=0,new_random_draws=0,scientific_array_reads=0,
        plot_scope='same sealed four-n panel; conditional joint99% intervals for26 targets; frozen numeric references')
    (OUT/'FIGURES.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(generated=True,figures=len(files),new_scientific_computation=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');main(p.parse_args().check)
