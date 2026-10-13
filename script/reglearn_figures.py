"""Plot only stored, frozen calculations; no new scientific evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/regularization_learning'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))


def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    numerical=OUT/'NUMERIC/RUN.json';growth=OUT/'GROWTH/SUMMARY.json'
    cell=next(c for c in read(numerical)['cells'] if c['a']==1 and c['order']==64)
    rows=cell['rows'];g=read(growth)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'raw':'#de7412','quadratic':'#1975b5','marg':'#16866c','informed':'#7754a3'}
    fig,axs=plt.subplots(1,2,figsize=(12.4,4.7),layout='constrained')
    theta=np.array([r['theta'] for r in rows])
    for rule,label in [('raw','原始极限'),('zero','恒零'),('quadratic','统一二次极限')]:
        values=np.array([r['rules'][rule]['joint_KL'] for r in rows]);mask=values>1e-14
        axs[0].plot(theta[mask],values[mask],'.-',lw=1.7,ms=3,label=label,
            color=colors.get(rule,'#6c747d'))
    axs[0].axhline(1/71136,color='#1975b5',ls='--',lw=1,label='二次全区间解析上界')
    axs[0].set(yscale='log',xlabel='真实相关参数 θ',ylabel='完整联合 KL',title='(a) 极限中：统一二次规则风险很小')
    axs[0].text(.025,.02,r'$\leq 10^{-14}$ 的点未置于对数轴',transform=axs[0].transAxes,fontsize=9)
    axs[0].legend(fontsize=9,loc='lower right');axs[0].grid(alpha=.18)
    for method,label in [('raw','原始'),('marg','同 K 边际化'),('quadratic','二次 / 知情渐近修正')]:
        rr=[r for r in g['selections'] if r['theta']==.6 and r['method']==method]
        axs[1].plot([r['K'] for r in rr],[r['rho'] for r in rr],'o-',lw=2,label=label,color=colors[method])
    axs[1].axhline(.6,color='#424a53',ls='--',lw=1,label='真实 θ = .60')
    axs[1].set(xscale='log',xticks=[16,64,256],xticklabels=['16','64','256'],ylim=(0,.8),
               xlabel='增长预算 K = n + 2',ylabel='固定候选面板的均值评分选点',
               title='(b) 有限维：渐近修正仍可过度收缩')
    axs[1].xaxis.set_minor_locator(NullLocator())
    axs[1].legend(fontsize=9,loc='lower right');axs[1].grid(alpha=.18)
    outputs={}
    for ext in ['png','svg']:
        p=figdir/('limit_vs_finite.'+ext);fig.savefig(p,dpi=180);outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.25),layout='constrained',sharey=True)
    for ax,theta in zip(axs,[0,.3,.6]):
        for method,label in [('raw','原始'),('marg','边际化'),('quadratic','统一二次'),('informed','知情渐近修正')]:
            c=next(c for c in g['curves'] if c['n']==254 and c['theta']==theta and c['method']==method)
            ax.plot(c['rho'],c['normalized_mean'],label=label,color=colors[method],
                linestyle='--' if method=='informed' else '-',marker='.',ms=3,lw=1.6)
        ax.axvline(theta,color='#747a80',ls=':',lw=1)
        ax.axhline(0,color='#969da4',lw=.6)
        ax.set(title=f'θ = {theta:.2f}',xlabel='候选相关 ρ');ax.grid(alpha=.15)
    axs[0].set_ylabel(r'相对 ρ = 0 的均值评分差 / $w_n$')
    axs[1].legend(ncol=2,fontsize=9,loc='upper center')
    fig.suptitle('n = 254，K = 256：固定候选总体评分面板（16 条独立先验流）',fontsize=13)
    for ext in ['png','svg']:
        p=figdir/('growth_profiles.'+ext);fig.savefig(p,dpi=180);outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    plt.close(fig)
    (OUT/'FIGURES.json').write_text(json.dumps(dict(
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [numerical,growth]},outputs=outputs,
        source_sha256=sha(__file__),new_scientific_computation=False,
        scope='Stored 61-point a1 limit scan and 21-candidate growth panel; no continuous finite optimum or new risk integration.'),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
