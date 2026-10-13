"""Derive readable tables and a scientific plot from sealed J/Q/M/G readouts.

No learner, RNG, integration, or scientific scoring module is imported.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from datetime import datetime, timezone
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/rc_quant'
OUT=BASE/'figures'
DOC=ROOT/'docs/08_research_研究工作/RC定量界与多块查询_20261001/RESULTS.md'
FILES=dict(J='J/SUMMARY_b100.json',Jdecision='J/DECISION.json',Q='Q/SUMMARY.json',
    M='M/SUMMARY.json',Mprimary='M/PRIMARY.json',Mcounts='M/COUNTS.json',
    G='G/RESULTS.json',Gdecision='G/DECISION.json',
    Jindependent='INDEPENDENT/J_SUMMARY.json',Qindependent='INDEPENDENT/Q_SUMMARY.json')
FLOOR=Fraction(87,56320)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def frac(x):return Fraction(*x)
def fnum(x):return f'{x:.9g}'
def outward(x,direction,places=15):
    x=Fraction(x)
    with localcontext() as context:
        context.prec=120;context.rounding=direction
        number=Decimal(x.numerator)/Decimal(x.denominator)
        return format(number.quantize(Decimal(1).scaleb(-places),rounding=direction),'f')
def interval(low,high):return f'[{outward(low,ROUND_FLOOR)}, {outward(high,ROUND_CEILING)}]'


def load():
    data={};inputs={}
    for key,name in FILES.items():
        path=BASE/name;package=name.split('/')[0]
        seal=BASE/package/('MANIFEST.json' if package=='INDEPENDENT' else 'RESULTS_FREEZE.json')
        expected=read(seal)['files'][path.relative_to(ROOT).as_posix()]
        assert sha(path)==expected,name
        data[key]=read(path);inputs[path.relative_to(ROOT).as_posix()]=expected
        inputs[seal.relative_to(ROOT).as_posix()]=sha(seal)
    assert data['Jdecision']['qualified'] and data['Q']['all_precisions_width_pass']
    assert data['Gdecision']['qualified'] and data['Gdecision']['interval_certificate'] is False
    assert len(data['Mprimary'])==4 and all(frac(r['analytic_RC_floor'])==FLOOR for r in data['Mprimary'])
    return data,inputs


def plot_data(data):
    rows=data['M'];ds=sorted({r['D'] for r in rows});ns=sorted({r['n1'] for r in rows})
    assert ds==[12,24,48,96] and ns==[2048,8192,32768]
    primary={r['D']:r for r in data['Mprimary']}
    trajectories=[]
    for d in ds:
        r=[next(row for row in rows if (row['D'],row['n1'],row['method'])==(d,n,'MOM')) for n in ns]
        trajectories.append(dict(D=d,n1=ns,n2=ns,mean_query_KL=[x['mean_query_kl'] for x in r],
            exact_structure_recoveries=[x['exact_structure_recoveries'] for x in r],
            high_N_interval=primary[d]['interval']))
    names=['MOM','MLE','HIST','PRODUCT','UNIFORM']
    high=[next(r for r in rows if (r['D'],r['n1'],r['method'])==(96,32768,name)) for name in names]
    return dict(trajectories=trajectories,high_N_D96=[dict(method=r['method'],mean_query_KL=r['mean_query_kl']) for r in high],
        D96_MOM_interval=primary[96]['interval'],ordinary_RC_analytic_lower=[FLOOR.numerator,FLOOR.denominator],
        lower_is_theory_not_fitted_baseline=True,repetitions=16,
        intervals='only the existing family4 simultaneous t development intervals for high-N MOM',
        new_samples=0,new_fits=0,new_integrations=0)


def figure(payload):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11.5,
        'axes.labelsize':10,'svg.fonttype':'none','svg.hashsalt':'rc_quant_20261001',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),gridspec_kw={'width_ratios':[1.14,1.]})
    fig.subplots_adjust(left=.073,right=.985,bottom=.25,top=.84,wspace=.24)
    palette=['#0072B2','#009E73','#D55E00','#6B4C9A'];markers=['o','s','^','D']
    floor=float(FLOOR)
    for ax in axes:
        ax.set_yscale('log');ax.set_ylim(6e-6,.075)
        ax.yaxis.set_major_locator(LogLocator(base=10));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y',which='major',color='#d9dde3',linewidth=.7,zorder=0)
        ax.axhline(floor,color='#252a31',linestyle=(0,(5,3)),linewidth=1.4,zorder=1)
    ax=axes[0]
    for row,color,marker in zip(payload['trajectories'],palette,markers):
        ax.plot(row['n1'],row['mean_query_KL'],label=f"D = {row['D']}",color=color,marker=marker,
            linewidth=1.8,markersize=6.3,markerfacecolor='white',markeredgewidth=1.5,zorder=3)
        mean=row['mean_query_KL'][-1];low,high=row['high_N_interval']
        ax.errorbar(32768,mean,yerr=[[mean-low],[high-mean]],fmt='none',color=color,capsize=3,linewidth=1.1,zorder=2)
    ax.set_xscale('log',base=2);ax.set_xticks([2048,8192,32768],['2,048','8,192','32,768'])
    ax.set_xlim(1600,42000)
    ax.set_ylabel('Mean excess LOO log loss (nats / target)')
    ax.set_xlabel('Vectors per split: n₁ = n₂\n(total training vectors = 2n₁)')
    ax.set_title('A  Unknown-block MOM learning',loc='left',fontweight='bold')
    ax.legend(loc='lower left',bbox_to_anchor=(.01,.055),frameon=True,framealpha=.96,ncol=2,
        edgecolor='#e2e5ea',fontsize=9,columnspacing=1.2,handlelength=2.1)
    ax.annotate('All D: 0/16 recovered\nGlobal uniform fallback',xy=(2048,payload['trajectories'][0]['mean_query_KL'][0]),
        xytext=(2450,.053),fontsize=9,color='#333b46',ha='left',va='center',
        arrowprops={'arrowstyle':'-','color':'#68717b','linewidth':.8})
    middle_offsets={12:(12,7),24:(12,-9),48:(12,-10),96:(12,8)}
    for row,color in zip(payload['trajectories'],palette):
        ax.annotate(f"{row['exact_structure_recoveries'][1]}/16",xy=(8192,row['mean_query_KL'][1]),
            xytext=middle_offsets[row['D']],textcoords='offset points',fontsize=8.8,color=color)
    ax.text(.98,.017,'At n = 32,768: all D recover 16/16',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=8.7,color='#48525f')
    ax.text(.03,floor*1.15,'Ordinary RC: proven lower bound',transform=ax.get_yaxis_transform(),
        ha='left',va='bottom',fontsize=8.7,color='#252a31',bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
    ax=axes[1]
    colors=['#0072B2','#009E73','#6B4C9A','#697789','#20252c']
    names=[r['method'] for r in payload['high_N_D96']]
    values=[r['mean_query_KL'] for r in payload['high_N_D96']]
    for x,(name,value,color) in enumerate(zip(names,values,colors)):
        ax.scatter(x,value,s=60,marker='D' if name=='UNIFORM' else 'o',
            facecolors='white' if name=='UNIFORM' else color,edgecolors=color,linewidths=1.5,zorder=3)
        offset=22 if name=='MOM' else 10
        ax.annotate(f'{value:.3g}',xy=(x,value),xytext=(0,offset),textcoords='offset points',
            ha='center',fontsize=9,color='#303b47')
    low,high=payload['D96_MOM_interval'];ax.errorbar(0,values[0],yerr=[[values[0]-low],[high-values[0]]],
        fmt='none',color=colors[0],capsize=4,linewidth=1.3,zorder=2)
    ax.set_xticks(range(5),['MOM','MLE','HIST','PRODUCT','Uniform']);ax.set_xlim(-.55,4.6)
    ax.set_xlabel('Fixed methods on the saved high-N data')
    ax.set_title('B  D = 96, n₁ = n₂ = 32,768',loc='left',fontweight='bold')
    ax.text(.03,floor*1.15,'87/56320 ≈ 0.001545  ·  theory, not a fit',transform=ax.get_yaxis_transform(),
        ha='left',va='bottom',fontsize=8.7,color='#252a31',bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
    fig.suptitle('Unknown high-order blocks: data recovery and an ordinary-RC query limit',
        fontsize=14,fontweight='bold',x=.073,ha='left',y=.965)
    fig.text(.073,.903,'Known three-pair template; hidden blocks and matchings; |ε| = 0.75 / 0.90 in equal proportions.',fontsize=9.5,color='#46505d')
    fig.text(.073,.087,'Means: 16 independent repetitions. Error bars: frozen simultaneous 95% t intervals for high-N MOM only.',fontsize=9,color='#46505d')
    fig.text(.073,.040,'Targets are uniform over all D coordinates; every other bit is observed. Dashed line is an analytic lower bound, not a fitted RC baseline.',fontsize=8.7,color='#46505d')
    fig.savefig(OUT/'unknown_blocks_query.png',dpi=180)
    fig.savefig(OUT/'unknown_blocks_query.svg',metadata={'Date':None,'Creator':'Matplotlib; sealed-result presentation only'})
    plt.close(fig)


def markdown(data):
    j=data['J'];q=data['Q'];primary=data['Mprimary'];m=data['M'];g=data['G'];gd=data['Gdecision'];counts=data['Mcounts']
    jl,jh=frac(j['lower_bound']),frac(j['upper_bound']);ql,qh=[frac(x) for x in q['intersection_global_interval']]
    text=['# J／Q／M／G完整可读结果','',
        '本页由已封存JSON派生，0新样本、拟合或积分。joint KL为整向量nats，query为全坐标平均LOO的nats/target；两者不混为同一单位。图中的理论下界不是拟合模型。','',
        '## J与Q：无界普通RC的严格区间','',
        '教师固定为六变量差分分布、ε=9/10。945棵满二叉树覆盖全部普通单一无符号laminar RC；不含signed count、混合或潜变量。下表显示端点向外舍入至15位小数，证书宽度来自完整有理端点。','',
        '|包与目标|严格全树最优区间（向外显示）|实际有理证书宽度|科研拟合／状态|',
        '|---|---|---:|---|',
        f'|J：完整joint KL|{interval(jl,jh)}|{float(jh-jl):.6e}|复用旧945终点；0新增拟合；945证书通过|',
        f'|Q：均匀六目标LOO|{interval(ql,qh)}|{float(qh-ql):.6e}|945新增query拟合；{q["numeric_qualified"]}/945数值门通过|','',
        f'J独立有理审计的区间宽为{float(frac(data["Jindependent"]["width"])):.6e}，Q为{float(frac(data["Qindependent"]["width"])):.6e}，均与生产证书相交。这里使用严格可行对偶及正有理模型的上下界；不是把小梯度当作无界证书。','',
        '|Q证书精度|全树区间宽度|≤1e−8|','|---|---:|---|']
    for bits,r in q['precision_levels'].items():text.append(f'|{bits} bits|{float(frac(r["width"])):.6e}|{"通过" if r["desired_width_pass"] else "未通过"}|')
    v=q['precision_levels']['160'];sl,sh=map(frac,v['global_summary_min_interval']);bl,bh=map(frac,v['summary_at_numeric_query_best_tree'])
    text += ['',f'全树最小摘要CMI区间：{interval(sl,sh)}；数值query最优代表树{q["numeric_best_representative_tree"]}上的摘要CMI区间：{interval(bl,bh)}。两者与RC最优风险不是同一个量，不能用不同树的global minima之差作因果归因。','',
        '严格端点见[J原始证书摘要](../../../artifacts/rc_quant/J/SUMMARY_b100.json)、[Q原始证书摘要](../../../artifacts/rc_quant/Q/SUMMARY.json)及[独立审计](../../../artifacts/rc_quant/INDEPENDENT/AUDIT.json)。','',
        '## M：保留全部恢复失败与三个样本预算','',
        '教师由独立六位差分块组成，绝对幅度.75/.9各占一半、符号随机；学习器知道三对差分模板和幅度范围，不知道块／matching。所有坐标均有作用，target均匀取全部D位并观察其余全部bit。每格16个独立重复，n1和n2为两个独立训练split，总向量数为n1+n2；不同前缀和方法共享数据。结构验证失败时MOM/MLE/HIST全部回退uniform，不能把这些平台删掉或改称数值失败。','',
        '|D|n1=n2|有效结构／16|完全恢复／16|MOM query KL|MOM joint KL|回退重复数|',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in m if r['method']=='MOM'),key=lambda r:(r['D'],r['n1'])):
        text.append(f'|{r["D"]}|{r["n1"]}|{r["valid_discoveries"]}/16|{r["exact_structure_recoveries"]}/16|{fnum(r["mean_query_kl"])}|{fnum(r["mean_joint_kl"])}|{16-r["valid_discoveries"]}|')
    text += ['', '### 最大样本格的全部方法','',
        '以下均为n1=n2=32768的16重复均值。MOM/MLE使用已知正确差分模板；HIST是所学六位块的64状态表；PRODUCT合并两个split估计边际；Uniform是固定公平参照。MLE/MOM的点估计差是次要读数，没有另加显著性门。','',
        '|D|方法|平均query KL|平均joint KL|','|---:|---|---:|---:|']
    for d in [12,24,48,96]:
        for method in ['MOM','MLE','HIST','PRODUCT','UNIFORM']:
            r=next(r for r in m if (r['D'],r['n1'],r['method'])==(d,32768,method))
            text.append(f'|{d}|{method}|{fnum(r["mean_query_kl"])}|{fnum(r["mean_joint_kl"])}|')
    text += ['', '### 原四项主判断','',
        f'原冻结门保持：高N MOM的4项Bonferroni同时95% t开发区间上端小于87/56320≈{float(FLOOR):.12f}，且该格观察到16/16完全恢复。区间不是严格总体覆盖，16/16也不证明真实恢复概率为1。','',
        '|D|MOM均值|原同时t区间|观察恢复|原主门|','|---:|---:|---|---:|---|']
    for r in primary:text.append(f'|{r["D"]}|{fnum(r["mean"])}|[{fnum(r["interval"][0])}, {fnum(r["interval"][1])}]|{r["observed_exact_recoveries"]}/16|{"通过" if r["passed"] else "未通过"}|')
    text += ['', f'原主门{sum(r["passed"] for r in primary)}/4通过。没有新增对比、阈值、维数或样本。强Q证书的其它lift参照不替换本页主门。','',
        '![未知块恢复、均值查询风险与普通RC理论下界](../../../artifacts/rc_quant/figures/unknown_blocks_query.png)','',
        '图A保留低样本global fallback平台，标注中样本的实际恢复数；图B为D96高N五个已保存均值。误差棒只画原四项高N MOM区间，其余点没有新置信区间。两图虚线均为87/56320的解析RC下界，不代表拟合过一个RC。可下载[矢量SVG](../../../artifacts/rc_quant/figures/unknown_blocks_query.svg)。','',
        '## G：Gaussian-logistic联系的三棵cherry树','',
        '固定d3、rho=−.45、scales=(.8,1.4,4)。下表坐标改用1起始（原JSON为0起始）。Γ的严格正性来自解析证明；这里的小数是固定求积收敛核对后的展示，不是严格数值区间证书。joint投影终点上的query值不是query目标全族最优值。','',
        '|cherry|外侧目标|Γ下界的数值展示|解析joint投影的joint KL|该joint终点的平均LOO KL|','|---|---:|---:|---:|---:|']
    for r in g['projections']:
        cherry='{'+','.join(str(v+1) for v in r['cherry'])+'}'
        text.append(f'|{cherry}|{r["outside_target"]+1}|{fnum(r["Gamma_lower"])}|{fnum(r["joint_KL_at_I_projection"])}|{fnum(r["mean_LOO_KL_at_I_projection"])}|')
    text += ['',f'三树min Γ的数值量级为{g["all_tree_query_lower_min"]:.12g}。三pair最大|GH512−GH256|={gd["moment_errors"]["GH512_minus_GH256"]:.6e}，最大|GH512−GL512|={gd["moment_errors"]["GH512_minus_GL512"]:.6e}，均通过原1e−10门；该门没有给出求积离散余项的严格界。G为0新拟合、0随机观察，未增加阶数或参数。','',
        '## 计数和验证范围','',
        f'M生成{counts["training_vector_draws"]:,}条完整训练向量（{counts["training_scalar_bits"]:,}个标量bit），0随机test／validation。192次discovery中{counts["valid_discoveries"]}次有效；{counts["global_uniform_fallback_outputs"]}份global fallback不计新参数拟合。共有{counts["total_joint_fit_endpoints"]}份joint拟合端点（{counts["product_joint_fits"]} PRODUCT＋{counts["learned_block_joint_fits"]}块模型），估计{counts["estimated_six_bit_factors"]}个六位factor和{counts["estimated_product_bernoulli_factors"]}个Bernoulli factor。','',
        '本轮新增科研拟合为Q的945个总体query问题加M的477份闭式joint端点；J证书构造、G解析投影、证书读回与本页作图均不新增科研拟合。M恢复失败／回退全部保留；G仅有收敛核对，J/Q才有严格有理数值证书。','',
        '派生脚本：`script/rc_quant_figures.py`。本页及PNG/SVG均绑定已保存输入hash，生成清单见[figures/MANIFEST.json](../../../artifacts/rc_quant/figures/MANIFEST.json)。脚本不导入学习、求积或科学评分模块。','']
    return '\n'.join(text)


def build():
    data,inputs=load();payload=plot_data(data);OUT.mkdir(parents=True,exist_ok=True)
    figure(payload);DOC.write_text(markdown(data),encoding='utf-8',newline='\n')
    (OUT/'PLOT_DATA.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs=[DOC,OUT/'unknown_blocks_query.png',OUT/'unknown_blocks_query.svg',OUT/'PLOT_DATA.json']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),inputs=inputs,
        outputs={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},new_samples=0,new_fits=0,new_integrations=0,
        figure_uses_only_saved_summary_and_primary_intervals=True)
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    check()


def check():
    data,inputs=load();manifest=read(OUT/'MANIFEST.json')
    assert manifest['source_sha256']==sha(__file__) and manifest['inputs']==inputs
    for name,value in manifest['outputs'].items():assert sha(ROOT/name)==value,name
    assert DOC.read_text(encoding='utf-8')==markdown(data)
    assert read(OUT/'PLOT_DATA.json')==plot_data(data)
    with Image.open(OUT/'unknown_blocks_query.png') as pic:
        assert pic.size==(2340,1026);pic.verify()
    root=ET.parse(OUT/'unknown_blocks_query.svg').getroot();assert root.tag.endswith('svg')
    print(json.dumps(dict(passed=True,PNG=[2340,1026],saved_summary_rows=len(data['M']),original_primary_tests=4,
        original_primary_passed=sum(r['passed'] for r in data['Mprimary']),new_samples=0,new_fits=0,new_integrations=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['build','check']);args=parser.parse_args()
    build() if args.phase=='build' else check()
