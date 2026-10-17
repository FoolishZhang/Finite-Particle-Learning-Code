"""Render all prespecified saved panels; no model calls or scientific sampling."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.analysis.read_text(encoding='utf-8'))
    cells=sorted(data['cells'],key=lambda c:(c['rep'],c['N'],c['t']))
    assert len(cells)==12 and not args.out.exists()
    args.out.mkdir(parents=True)
    shutil.copyfile(Path(__file__),args.out/'PLOT_SOURCE.py')
    fig,axes=plt.subplots(1,2,figsize=(12,6.9),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
    labels=[f"Pool {c['rep']}, train={c['N']:,}, t={c['t']:.1f}" for c in cells]
    y=np.arange(12)
    axes[0].axvspan(-.01,.01,color='#e3efe6',zorder=0,label='Prespecified tolerance')
    for ax in axes:
        ax.axvline(0,color='#7b8490',linewidth=.8)
        ax.set_axisbelow(True);ax.grid(axis='x',color='#e5e7eb',linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
    for j,c in enumerate(cells):
        if not all(key in c for key in ('W_ind','W_same')):
            axes[0].text(.98,j,'unresolved',transform=axes[0].get_yaxis_transform(),ha='right',fontsize=8)
        for key,shift,color,marker,label in [('W_ind',-.12,'#3567a0','o','Independent prediction'),('W_same',.12,'#b55a32','s','Same-bank remainder')]:
            if key not in c:continue
            ci=c[key+'_interval'];value=c[key]['mean'];err=np.array([[value-ci['lower']],[ci['upper']-value]])
            axes[0].errorbar([value],[j+shift],xerr=err,fmt=marker,color=color,capsize=2,markersize=4,label=label if j==0 else None)
        if 'W_same' in c:
            ci=c['W_same_interval'];value=c['W_same']['mean']
            axes[1].errorbar([value],[j],xerr=[[value-ci['lower']],[ci['upper']-value]],fmt='s',color='#b55a32',capsize=2,markersize=4)
        else:
            axes[1].text(.98,j,'unresolved',transform=axes[1].get_yaxis_transform(),ha='right',fontsize=8)
    axes[0].set_yticks(y,labels,fontsize=9);axes[0].invert_yaxis()
    bounds=[c[key+'_interval'][edge] for c in cells for key in ('W_ind','W_same')
            if key+'_interval' in c for edge in ('lower','upper')]
    lo,hi=min([-.01,*bounds]),max([.01,*bounds]);pad=.07*(hi-lo)
    axes[0].set_xlim(lo-pad,hi+pad)
    axes[0].set_xlabel('Normalized error (D - P) / t²')
    axes[1].set_xlabel('Same-bank (D - P) / t²; expanded scale')
    axes[0].set_title('Two distinct measurements',loc='left',fontweight='bold',fontsize=12)
    axes[1].set_title('Finite-amplitude remainder',loc='left',fontweight='bold',fontsize=12)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.60,.945),ncol=3,fontsize=9,frameon=False)
    axes[1].ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    fig.suptitle('Frozen learned local proposals: independent prediction and Taylor remainder',y=.99,fontsize=13)
    fig.text(.02,.015,'All 12 prespecified panels. Bars: approximate simultaneous 95% intervals across 36 quantities (16 randomizations).\nConditional on six frozen proposals from three training pools; not training-population uncertainty.',fontsize=9,color='#394453')
    fig.tight_layout(rect=(0,.07,1,.86))
    for suffix in ('png','svg'):fig.savefig(args.out/f'precision_and_remainder.{suffix}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    record={'analysis_sha256':sha(args.analysis),'source_sha256':sha(Path(__file__)),
            'all_prespecified_cells':12,'family_size':36,'right_panel_expanded_scale':True,
            'no_scientific_calls':True,'files':{q.name:sha(q) for q in args.out.iterdir() if q.is_file()}}
    with (args.out/'FIGURE_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))


if __name__=='__main__':main()
