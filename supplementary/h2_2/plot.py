"""Regenerate the two paper panels from supplied public summaries only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(pkg,name):
    return json.loads((ROOT/'artifacts'/pkg/name).read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Use an output directory outside the supplement')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    gap=read('nuisance_contrast_v1_20261006','POPULATION.json')['results'][:2]
    curvature=read('local_geometry_v1_20261006','STATISTICS.json')['results'][:2]
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.15),layout='constrained')
    for ax,rows,title,xlabel in zip(axes,(gap,curvature),('Population anchor gap','Curvature at the independent truth'),(r'$g_n=F_n(I)-F_n(C_b)$',r'$h_n$')):
        for y,row,color in zip((1,0),rows,('#2d6d96','#bd583a')):
            m=row['mean'];lo,hi=row['interval'];ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color=color,capsize=4)
        ax.axvline(0,color='gray',linestyle='--');ax.set_yticks([1,0],[r'$n=0$',r'$n=256$']);ax.set_ylim(-.55,1.55);ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'figure1.png',dpi=300);plt.close(fig)
    data=read('asymptotic_bridge_v1_20261008','AUTHOR_READOUT.json');panel=data['per_n'];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    def curve(ax,key,label,color,offset):
        m=np.array([r[key]['mean'] for r in panel]);lo=np.array([r[key]['interval'][0] for r in panel]);hi=np.array([r[key]['interval'][1] for r in panel])
        ax.errorbar(x+offset,m,yerr=np.vstack((m-lo,hi-m)),label=label,color=color,marker='o',capsize=3,linewidth=1)
    for key,label,color,offset in zip(('A','Bgrad','omega_over_w'),('Gap / leading term','Gradient / leading term','Overlap / scale'),('#16718a','#ce6031','#686868'),(-.09,0,.09)):
        curve(axes[0],key,label,color,offset)
    axes[0].axhline(1,color='gray',linestyle='--');axes[0].set_ylim(-.25,1.27);axes[0].set_title('Finite-scale ratios')
    for key,label,color,offset in zip(('O','J'),(r'$O_n$ overlap',r'$J_n$ marked coupling'),('#16718a','#8d488b'),(-.07,.07)):
        curve(axes[1],key,label,color,offset)
    axes[1].axhline(data['frozen_references']['m_H45'],color='#9b4944',linestyle=':');axes[1].set_ylim(-.0018,.028);axes[1].set_title('Fixed-coordinate residual')
    for ax in axes:
        ax.axhline(0,color='gray',linewidth=.5);ax.set_xticks(x,[str(r['n']) for r in panel]);ax.set_xlabel(r'Nuisance dimension $n$');ax.legend(fontsize=7,frameon=False);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'figure2.png',dpi=300);plt.close(fig)
    print(json.dumps(dict(passed=True,figures=2,saved_data_only=True,new_integrals=0,new_samples=0)))


if __name__=='__main__':main()
