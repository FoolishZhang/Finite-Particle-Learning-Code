"""Render fixed-result tables and a scientific plot; no new model evaluation."""
from pathlib import Path
import json
import os
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/rec_k_v3_20261005'
DOC=ROOT/'docs/08_research_研究工作/REC_v3_20261005'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/reck_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text(encoding='utf-8'))
r=read(OUT/'RESULTS.json');audit=read(OUT/'ARITHMETIC_AUDIT.json');moment=read(OUT/'MOMENT_AUDIT.json')
arms=('NEAR_FREE','UNIT_FREE');ks=(100,400,1600)
assert not (DOC/'RESULTS.md').exists()
figdir=DOC/'figures';figdir.mkdir(exist_ok=True)
def cell(kind,arm,k,target):return next(x for x in r['family'] if (x['kind'],x['arm'],x['K'],x['target'])==(kind,arm,k,target))
def formatted(x):
    lo,hi=x['interval'];return f'{x["mean"]*1e6:+.3f} [{lo*1e6:+.3f}, {hi*1e6:+.3f}]'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
for ax,arm in zip(axes,arms):
    rows=[cell('endpoint',arm,k,'F_P') for k in ks]
    exact=next(x for x in r['exact'] if x['arm']==arm)
    mean=np.array([x['mean'] for x in rows])*1e6
    err=np.array([[x['mean']-x['interval'][0] for x in rows],[x['interval'][1]-x['mean'] for x in rows]])*1e6
    curve=np.geomspace(100,1600,180)
    ax.plot(curve,(exact['exact_loss_delta'][1]+exact['coefficient_delta'][1]/curve)*1e6,
            color='#d18a00',linestyle='--',linewidth=1.7,label='Second-order 1/K approximation')
    ax.errorbar(ks,mean,yerr=err,fmt='o',color='#176899',capsize=4,markersize=6,label='MC mean + simultaneous intervals',zorder=3)
    ax.axhline(exact['exact_loss_delta'][1]*1e6,color='#515b65',linestyle=':',linewidth=1.5,label='Fixed exact-risk difference')
    ax.axhline(0,color='#ad4a4a',linewidth=.9,alpha=.65)
    ax.set_xscale('log');ax.set_xticks(ks,labels=[str(k) for k in ks]);ax.minorticks_off()
    ax.set_xlabel('Active particles K');ax.set_title(arm.replace('_FREE','')+' initialization')
    ax.grid(axis='y',alpha=.17);ax.set_ylim(-250,380)
axes[0].set_ylabel('Candidate minus baseline score (×10⁻⁶)')
fig.suptitle('Fixed endpoints: increasing K reverses the population-score preference',fontsize=12,y=.98)
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=1,frameon=False,fontsize=9,bbox_to_anchor=(.52,.015))
fig.subplots_adjust(left=.10,right=.98,top=.84,bottom=.31,wspace=.14)
fig.savefig(figdir/'active_K_preference.png',dpi=180,facecolor='white')
fig.savefig(figdir/'active_K_preference.svg',facecolor='white')
plt.close(fig)

lines=['# REC-v3：参数与风险不变，active粒子数改变尺度候选的总体评分排序','',
'2026-10-05。按固定合同完成K100/400/1600的对照。**两种初始化下，K100的总体有限目标仍惩罚SCALE2；K400和K1600均转为支持同一个候选。** 模型参数、观察分布、风险和数据频率始终不变；改变的是消去nuisance后的active计算准则。','',
'这是对REC-v2剩余active有限粒子偏差的有限网格诊断，不是训练恢复实验。K400是本次预定网格中首个确认改判的值，不是最小阈值或通用推荐。风险恶化个例没有消失，也没有获得新的训练总体结论。','',
'![固定端点的active粒子数与总体评分偏好](figures/active_K_preference.png)','',
'误差条为固定16项Bonferroni家族的MC开发t近似区间，仅处理条件于这批保存端点的评价误差；虚线为预定二阶近似，点线为不随K改变的旧精确风险差。图没有新的风险积分或未冻结K评价。','',
'## 1. 完整预定K网格','',
'每流1024个独立bank，K使用同一1600粒子池的嵌套前缀；同流两初值/两端点共享prior。16份旧数据是固定面板，不把32个BASE或SCALE2端点当作新的独立训练重复。下表单位为10⁻⁶，负值支持候选。','',
'|初值|K|ΔF_D [MC CI]|ΔF_P [MC CI]|',
'|---|---:|---:|---:|']
for arm in arms:
    for k in ks:lines.append(f'|{arm}|{k}|{formatted(cell("endpoint",arm,k,"F_D"))}|{formatted(cell("endpoint",arm,k,"F_P"))}|')
lines+=['',
'K100新评价独立重现了上轮总体评分的反向排序，没有覆盖旧数值。K400/1600的四个总体区间都全负。原数据条件F_D在三个K上均支持候选，翻转发生在F_P；因此不能声称K100的训练目标一概拒绝恢复。','',
'跨K的四个直接对照也是原16项家族的一部分，方差按同bank配对差计算：','',
'|初值|ΔF_P(K)−ΔF_P(100)|配对变化及MC CI（10⁻⁶）|',
'|---|---:|---:|']
for arm in arms:
    for k in (400,1600):lines.append(f'|{arm}|K={k}|{formatted(cell("K_contrast",arm,k,"F_P"))}|')
lines+=['',
'这些结果确认了固定端点上的计算准则干预会改变面板平均偏好，排除了“只要参数固定，评价精度提高不能改变此处期望排序”的解释。它没有确认每个端点都同向，也没有新训练或泛化比较。','',
'## 2. 二阶矩解释与1/K量级','',
'精确身份是ΔF_T,K=Δell_T+Δb_T,K。新积分M2=EA²，按B_T=(1/2)sum T_y(M2_y/q_y²−1)得到二阶近似ΔF≈Δell+ΔB/K；M2没有按标签和归一化。这是经典Jensen/Taylor工具的定点诊断，不是新的通用定理或有限K误差界。','',
'|初值|固定精确Δell_D|固定精确ΔR=Δell_P|ΔB_D|ΔB_P|',
'|---|---:|---:|---:|---:|']
for x in r['exact']:
    lines.append(f'|{x["arm"]}|{x["exact_loss_delta"][0]:+.10f}|{x["exact_loss_delta"][1]:+.10f}|{x["coefficient_delta"][0]:.9f}|{x["coefficient_delta"][1]:.9f}|')
lines+=['',
'总体偏差增量随K减小，K乘增量与二阶矩系数处于同一量级：','',
'|初值|K|Δb_P,K|K×Δb_P,K|ΔF_P−二阶预测|',
'|---|---:|---:|---:|---:|']
for x in r['bias']:
    if x['target']=='F_P':lines.append(f'|{x["arm"]}|{x["K"]}|{x["bias_increment"]:.10f}|{x["K_times_bias"]:.9f}|{x["MC_minus_prediction"]:+.10f}|')
compatible=0
for x in r['bias']:
    interval=cell('endpoint',x['arm'],x['K'],x['target'])['interval']
    compatible+=interval[0]<=x['predicted_delta']<=interval[1]
lines+=['',
f'描述性核对中，{compatible}/12个预定二阶近似读数落入相应MC区间。这个兼容性不建立严格余项界；K增大后偏差很小，K乘MC误差会放大相对波动，不能把高K的系数读数当作精确常数。没有额外拟合率、选K或增加统计家族。','',
'Jensen保证每个固定模型的偏差非负且随K趋零，并不保证两个模型的偏差差或偏好单调。本次只确认三个预定K的排序；未测量中间K或精确翻转阈值。','',
'## 3. 与两轮既有结果放在一起','',
'|证据层|已经得到的判断|仍不能推出|',
'|---|---|---|',
'|REC-v1|FREE在冻结面板平均上优于同数据独立拟合；RHO候选可共同改善目标/风险|所有恢复自由度都被目标接受，或原程序已优化充分|',
'|REC-v2|固定小尺度限制精确依赖表达；保持边际放大尺度平均降风险却升F_P，消去nuisance仍冲突|能力下界足以排除胜过产品；唯一历史原因已经识别|',
'|REC-v3|固定端点只增加active K，K400/1600使总体评分改判；二阶矩与1/K解释相容|高K训练必然恢复、每个模型受益、同成本算法胜出或通用K阈值|','',
'这些判断分别涉及可表达模型集合、有限计算目标和有限程序实际输出，不能合并成唯一的风险因果分解。特别是本轮只用已消去nuisance的数学评分，不覆盖保留无关权重的原生float32训练、head/precision或未知块的完整SCBM。','',
'SCALE2原有每组2个风险恶化个例、平均仍高于真实产品风险的事实都保持。本轮风险没有再下降，因为根本没有改变端点；变的是评分是否愿意接受它。','',
'## 4. 核验、预算与复现','',
f'- 64模型×3阶的一阶/二阶矩资格通过。固定8个GH44对称根检查通过，M2最大差{max(x["second_error"] for x in moment["records"]):.3g}，系数最大差{max(x["coefficient_error"] for x in moment["records"]):.3g}。',
f'- 全部16项MC统计和系数复算通过；直接首bank评分最大差{audit["max_direct_score_difference"]:.3g}。16/16绝对精度通过，0方向未决；最大SE={max(x["SE"] for x in r["family"]):.6g}，低于1e−5。',
'- 这些是不同表达式的本地核验，不是外部专家复现、严格GH余项或新训练总体置信保证。统计df1023、q16、alpha.05，跨K依赖已由配对方差处理。',
'- 0新观察/训练/候选/基线；64固定模型。252,182,528主矩节点、29,984,768独立矩节点；26,214,400新prior向量/104,857,600 Gaussian，3,145,728个pattern评分。1024bank×16流不增加独立训练样本数。',
f'- 主计算{r["seconds"]:.2f}秒，独立矩审计{moment["seconds"]:.2f}秒；K变大意味着每bank更多粒子，不能称同端到端成本。制造、4流哈希重放及公开demo成本另记。',
'- 新科学输入全部来自Git内的JSON和代码，不需要原观察数组。只复制一个核验脚本和两个公开JSON的隔离目录，实际流0首bank复核通过；它不是全新依赖环境或全部面板的外部复现。见[最小复现](MINIMAL_REPRO.md)。','',
'## 5. 本周期结论与工作取舍','',
'这一轮完成了上轮预定的区分：在固定尺度端点，剩余active有限K偏差能改变总体偏好，并可在预定更高K上减小到足以改判。继续在同一面板追加更多粒子数已不是当前最有价值的工作；不拟合精确临界K，也不将经典渐近性质扩张为论文主创新。','',
'当前恢复诊断可以收束成一条短证据链：保留同数据基线、能力边界和固定端点K翻转，明确它们如何修正“全部是碰撞/全部是优化”的叙述。接下来作者工作应评估这条链是否强化原有符号偏好—谱—风险主线；若再做新训练或用途实验，需要一个能新增判断的合同，而非以FREE必须失败或某个修复必须成功为目标。旧论文不回写，不自动生成v10。']
with (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'report_and_figure_written':True,'second_order_predictions_in_existing_intervals':int(compatible),'new_scientific_evaluations':0}))
