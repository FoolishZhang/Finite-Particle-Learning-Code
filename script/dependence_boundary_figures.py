"""Readable, deterministic views of the frozen boundary experiments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/dependence_boundary_v1';OUT=RUN/'figures'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    rows=read(RUN/'S/RESULTS.json');OUT.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.1),layout='constrained')
    colors=['#8b5fbf','#167d9a','#ce6e27'];ks=np.arange(2,101)
    for rho,color in zip([.05,.108,.25],colors):
        axes[0].plot(ks,(1-(ks-1)*rho)/(4*ks),label=f'r = {rho:g}',color=color)
        axes[0].axhline(-rho/4,color=color,ls=':',lw=1.2)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xscale='log',xlabel='Training samples K',ylabel='Population loss derivative at zero',title='Exact local boundary')
    axes[0].legend(fontsize=8);axes[0].text(.03,.04,'Solid: iid     Dotted: antithetic',transform=axes[0].transAxes,fontsize=8)
    for sampler,style,color in [('iid','o-','#b14e44'),('antithetic','s-','#167d9a')]:
        subset=[r for r in rows if r['K']==10 and r['sampler']==sampler]
        axes[1].plot([r['rho'] for r in subset],[r['model_correlation'] for r in subset],style,color=color,label=sampler)
        ss=[r for r in rows if r['rho']==.108 and r['sampler']==sampler]
        axes[2].plot([r['K'] for r in ss],[r['feedback_gain'] for r in ss],style,color=color,label=sampler)
    axes[1].plot([0,.25],[0,.25],':',color='gray',label='true correlation')
    axes[1].set(xlabel='True observed correlation',ylabel='Fitted observed correlation',title='Same K = 10, different learned dependence');axes[1].legend(fontsize=8)
    oracle=next(r['feedback_gain']+r['KL'] for r in rows if r['rho']==.108)
    axes[2].axhline(oracle,ls=':',color='gray',label='oracle information gain')
    axes[2].axhline(.005,ls='--',color='gray',lw=.8,label='registered material scale')
    axes[2].set(xscale='log',xlabel='Training samples K',ylabel='Single-feedback gain (nats)',title='Feedback utility: true r = .108');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.savefig(OUT/'population_boundary.png',dpi=170)
    fig.savefig(OUT/'population_boundary.svg')
    p=OUT/'population_boundary.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    lines=['# 人口有限采样目标：完整结果','',
        '所有4状态按真实概率精确加权，无有限标签训练误差。Gaussian积分为冻结的随机化准蒙特卡洛，拟合后另用新流核验。',
        '盒内数值拟合不构成全局优化解析证书；成对目标仍有有限K偏差。','',
        '| True r | K | Sampler | Latent variance t | Observed r | Feedback NLL | Feedback gain | KL | Numerical gate |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['rho']:.3f} | {r['K']} | {r['sampler']} | {r['t']:.6f} | {r['model_correlation']:.6f} | {r['conditional_NLL']:.6f} | {r['feedback_gain']:.6f} | {r['KL']:.6f} | {r['fit_precision_pass']} |")
    (ROOT/'docs/08_research_研究工作/边际依赖接口_20260929/POPULATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print({'figure':str(OUT/'population_boundary.png'),'cases':len(rows)})
if __name__=='__main__':main()
