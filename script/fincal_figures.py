"""Figures derived exclusively from stored calibration and training outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/finite_calibration'


def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    input_names=['CAL/MODEL.json','CAL/VALIDATION.json','TRAIN/STATISTICS.json',
                 'POSTHOC_QUADRATIC/PROTOCOL.json','POSTHOC_QUADRATIC/RESULTS.json','CONDITIONAL_TARGET/RESULTS.json']
    model,val,stats,qp,qr,cond=[read(OUT/p) for p in input_names]
    cm=next(c for c in model['cases'] if c['n']==254);cv=next(c for c in val['cases'] if c['n']==254)
    rho=np.array(cm['rho101']);r=np.array(model['reference']['r801'])[::8];positive=rho>=0
    gref=-np.array(cv['U_prime'])-r*np.array(cv['V_prime'])
    gamma=model['reference']['A']*model['reference']['d']*cm['w']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,
        'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'raw':'#da721d','marg':'#148a6d','quad':'#8461aa','finitecal':'#1575b5',
            'finite_taylor_quad':'#1575b5','exact':'#69727a','zero':'#292f34'}
    fig,ax=plt.subplots(1,3,figsize=(15.4,4.8),layout='constrained')
    ax[0].plot(rho[positive],gref[positive],color='#272d35',lw=2,label='独立验证导数估计')
    ax[0].plot(rho[positive],np.array(cm['g_prime'])[positive],color=colors['finitecal'],lw=1.7,label='有限校准')
    ax[0].plot(rho[positive],2*gamma*rho[positive],color=colors['quad'],lw=1.7,label='渐近二次')
    ax[0].plot(rho[positive],2*qp['coefficient']*rho[positive],color=colors['finitecal'],ls='--',lw=1.3,label='有限二次近似*')
    ax[0].set(xlabel='候选相关 ρ',ylabel='参数校正项的导数',title='(a) 有限维需要较弱的校正幅度')
    ax[0].legend(fontsize=9);ax[0].grid(alpha=.17)
    teachers=[0.,.3,.6]
    for method,label in [('raw','原始'),('marg','边际化'),('quad','渐近二次'),('finitecal','有限校准'),
                         ('finite_taylor_quad','有限二次近似*'),('exact','精确似然参考'),('zero','恒零')]:
        ys=[]
        for th in teachers:
            source=qr['summary'] if method=='finite_taylor_quad' else stats['descriptive']
            row=next(x for x in source if x['theta']==th and x['N']==4096 and (method=='finite_taylor_quad' or x['method']==method))
            y=row['mean_reference_KL'] if method=='finite_taylor_quad' else row['mean'];ys.append(y if y>0 else np.nan)
        ax[1].plot(teachers,ys,marker='o',ms=4,lw=1.7,color=colors[method],
            ls='--' if method in ['zero','finite_taylor_quad'] else '-',label=label)
    ax[1].set(yscale='log',xticks=teachers,xlabel='真实相关 θ',ylabel='16流平均参考 joint KL',
              title='(b) N = 4096 的完整观测模拟训练')
    ax[1].legend(fontsize=8.1,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3);ax[1].grid(alpha=.17)
    ax[1].text(.02,.025,'恒零在 θ=0 的风险为0，未置于对数轴',transform=ax[1].transAxes,fontsize=8)
    for th in teachers:
        row=next(x for x in stats['primary'] if x['theta']==th and x['contrast']=='finite_minus_exact')
        ax[2].errorbar(th,row['mean'],yerr=[[row['mean']-row['lower']],[row['upper']-row['mean']]],
            fmt='o',color='#1775aa' if row['pass_'] else '#ca641d',capsize=4)
    ax[2].axhline(.0002,color='#a85627',ls='--',lw=1.3,label='非劣门限 .0002')
    ax[2].axhline(0,color='#9a9fa5',lw=.8)
    ax[2].set(xticks=teachers,xlabel='真实相关 θ',ylabel='有限校准 − 精确似然参考',
              title='(c) 两项正相关非劣门未通过')
    ax[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[2].legend(fontsize=8.5);ax[2].grid(alpha=.17)
    outputs={};dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        p=dest/f'calibration_training.{ext}';fig.savefig(p,dpi=180);outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    for N,color in [(1024,'#977950'),(4096,'#177eb1')]:
        s=[next(x for x in cond['summary'] if x['theta']==th and x['N']==N) for th in teachers]
        ax[0].plot(teachers,[x['mean_abs_proxy_minus_MLE'] for x in s],'o--',color=color,label=f'N{N}：条件代理−MLE')
        ax[0].plot(teachers,[x['mean_abs_actual_minus_proxy'] for x in s],'s-',color=color,label=f'N{N}：实际−条件代理')
    ax[0].set(yscale='log',xticks=teachers,xlabel='真实相关 θ',ylabel='平均参数绝对差',title='(a) 事后误差分层诊断')
    ax[0].legend(fontsize=8.5);ax[0].grid(alpha=.17)
    for N,color in [(1024,'#977950'),(4096,'#177eb1')]:
        s=[next(x for x in qr['summary'] if x['theta']==th and x['N']==N) for th in teachers]
        ax[1].plot(teachers,[x['mean_minus_full_finite'] for x in s],'o-',color=color,label=f'N = {N}')
    ax[1].axhline(0,color='#9a9fa5',lw=.8)
    ax[1].set(xticks=teachers,xlabel='真实相关 θ',ylabel='二次近似 − 完整有限校准（均值KL）',
              title='(b) 事后二次近似接近完整曲线')
    ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[1].legend(fontsize=9);ax[1].grid(alpha=.17)
    for ext in ['png','svg']:
        p=dest/f'posthoc_diagnostics.{ext}';fig.savefig(p,dpi=180);outputs[p.relative_to(ROOT).as_posix()]=sha(p)
    plt.close(fig)
    (OUT/'FIGURES.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        inputs={(OUT/p).relative_to(ROOT).as_posix():sha(OUT/p) for p in input_names},outputs=outputs,
        new_scientific_computation=False,
        scope='Saved outputs only; starred quadratic and conditional-target diagnostics are posthoc reused-data analyses; t intervals are conditional development approximations.'),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
