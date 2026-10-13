"""Render completed REC-v5 cross-objective scores without new scientific evaluation."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_cross_v5_20261005';DOC=ROOT/'docs/08_research_研究工作/REC_v5_cross_20261005'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/reccross_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text(encoding='utf-8'))
r=read(OUT/'RESULTS.json');a=read(OUT/'ARITHMETIC_AUDIT.json');ind=read(OUT/'INDEPENDENT_AUDIT.json')
assert a['passed'] and ind['passed'] and not (DOC/'RESULTS.md').exists()
derived=[]
for e in r['exact']:
    for ti,target in enumerate(('F_D','F_P')):
        for k in ('K100','K400'):
            cell=next(x for x in r['family'] if (x['case'],x['target'],x['contrast'])==(e['case'],target,k))
            derived.append(dict(case=e['case'],target=target,K=k,exact_delta=e['mean_exact_deltas'][ti],
                                finite_delta=cell['mean'],bias_difference=cell['mean']-e['mean_exact_deltas'][ti]))
with (OUT/'BIAS_DIFFERENCE.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(dict(result_sha256=hashlib.sha256((OUT/'RESULTS.json').read_bytes()).hexdigest(),records=derived,
                   scope='Descriptive identity delta B_K = delta F_K - delta exact loss; no new tests or causal shares'),f,ensure_ascii=False,indent=2);f.write('\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4),sharey=True)
for ax,case in zip(axes,('NULL','MIXED')):
    e=next(x for x in r['exact'] if x['case']==case)
    for ti,(target,color,label) in enumerate((('F_D','#b45d25','Saved training data'),('F_P','#176899','Teacher distribution'))):
        xs=[-.055 if ti==0 else .055,1+(-.055 if ti==0 else .055),2+(-.055 if ti==0 else .055)]
        cells=[next(x for x in r['family'] if (x['case'],x['target'],x['contrast'])==(case,target,k)) for k in ('K100','K400')]
        vals=[c['mean']*1000 for c in cells]+[e['mean_exact_deltas'][ti]*1000]
        ax.plot(xs,vals,color=color,alpha=.65,linewidth=1)
        for x,c in zip(xs,cells):
            mean=c['mean']*1000;lo,hi=[v*1000 for v in c['interval']]
            ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=5,markersize=6)
        ax.scatter(xs[2],vals[2],marker='D',s=36,color=color,label=label)
    ax.axhline(0,color='#333333',linewidth=.8)
    ax.set_xticks([0,1,2],['F100','F400','GH reference'])
    ax.set_xlim(-.3,2.3);ax.set_title(case+' teacher');ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Evaluation criterion (fixed endpoints)')
axes[0].set_ylabel('Loss difference: INNER - OUTER (x 1e-3)')
axes[1].legend(frameon=False,loc='upper right')
fig.text(.5,.035,'Positive favors OUTER. Error bars: fixed-panel MC q12 intervals; GH points reuse saved PMFs, with no new data interval.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.1,right=.98,bottom=.22,top=.88,wspace=.12)
(DOC/'figures').mkdir(exist_ok=True)
fig.savefig(DOC/'figures/cross_objective.png',dpi=180,facecolor='white')
fig.savefig(DOC/'figures/cross_objective.svg',facecolor='white');plt.close(fig)

lines=['# REC-v5：两种有限目标平均都偏好OUTER，MIXED风险仍由INNER更低','',
'**冻结REC-v4的全部128个TAIL后，F100和F400在NULL/MIXED的同数据与总体面板平均上均偏好OUTER。没有得到“各自训练目标支持各自终点”的平均分离。** MIXED的高精度观察风险仍由INNER较低；NULL则OUTER较低。新MC评价因此改变解释重点，不改旧学习结果。','',
'![交叉目标与高精度参考](figures/cross_objective.png)','',
'纵轴统一为INNER−OUTER，正数支持OUTER。有限目标误差条为固定面板的q12 MC区间；高精度参考复用GH40，不是新训练总体区间。连接线仅连接三个实际评价准则，不推断任意K曲线或翻转阈值。','',
'## 1. 主要交叉评价：八项终点差','',
'每条件32对已保存模型，全部保留；每流4096个新独立bank，每bank400个四维基础prior。F100为四块100损失外平均，F400为同400点内平均，估计各自期望目标。','',
'|条件|权重|评价|平均差INNER−OUTER|q12 MC区间|MC SE|偏好|',
'|---|---|---|---:|---|---:|---|']
for x in r['family']:
    if x['contrast']=='K400-K100':continue
    lo,hi=x['interval'];lines.append(f"|{x['case']}|{x['target']}|{x['contrast']}|{x['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|{x['SE']:.2g}|{x['direction']}|")
lines+=['','F_D对应原数据的经验目标，F_P用同教师分布加权。两种权重都给出同向的面板平均排名。MIXED的F_D400差虽只有约0.0000473，区间仍明确为正，不能把它写成未分辨或INNER已经更好。','',
'## 2. 高精度观察NLL与原风险','',
'|条件|同数据NLL差|总体NLL差＝原教师KL差|说明|','|---|---:|---:|---|']
for x in r['exact']:
    lines.append(f"|{x['case']}|{x['mean_exact_deltas'][0]:+.9f}|{x['mean_exact_deltas'][1]:+.9f}|"+('INNER训练NLL更低，但总体风险更高|' if x['case']=='NULL' else 'INNER训练NLL和总体风险均更低|'))
lines+=['','这保留了REC-v4的MIXED约21.8%改善与NULL约42.7%恶化。NULL中训练拟合更好、总体风险更差与过拟合解释相容，但不是本轮识别出的唯一原因。MIXED中有限目标排名与观察风险仍相反。','',
'## 3. 提高K向INNER移动，但尚未改变平均排名','',
'|条件|权重|Δ400−Δ100|q12 MC区间|','|---|---|---:|---|']
for x in r['family']:
    if x['contrast']!='K400-K100':continue
    lo,hi=x['interval'];lines.append(f"|{x['case']}|{x['target']}|{x['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|")
lines+=['','这四项负值表示相对评分向INNER移动，不表示F400已经选择INNER；其F400终点差仍为正。令B_K=F_K−L_ref，恒等式ΔB_K=ΔF_K−ΔL_ref给出如下额外惩罚差，纯属派生解释而非新独立检验或因果份额。','',
'|条件|权重|K|INNER的额外有限评分惩罚差ΔB_K|','|---|---|---|---:|']
for x in derived:lines.append(f"|{x['case']}|{x['target']}|{x['K']}|{x['bias_difference']:+.9f}|")
lines+=['','例如MIXED同数据NLL改善约0.0003243，但K400下额外评分惩罚约0.0003716，留下+0.0000473的有限目标差。对该面板，观察拟合收益尚不足以抵消有限评价惩罚；这不是关于全局最优解的结论。','',
'## 4. 平均排名不等于每个流都同向','',
'以下只按逐流MC点估计计数，没有256项逐流同时区间保证。','',
'|条件|权重|两目标都偏OUTER|各目标偏自己的终点|两目标都偏INNER|反向分离|','|---|---|---:|---:|---:|---:|']
for x in r['classification']:
    c=x['point_sign_counts'];lines.append(f"|{x['case']}|{x['target']}|{c['both_OUTER']}|{c['aligned_split']}|{c['both_INNER']}|{c['reverse_split']}|")
lines+=['','MIXED的F_D有11/32对出现各自目标支持各自终点的点估计分离，另21/32同偏OUTER。不能用整体平均否认这些异质性，也不能把点估计计数当逐流显著性证据。','',
'## 5. 结论与边界','',
'交叉评价没有支持最直接的平均解释——F100选OUTER、F400选INNER。两个有限目标平均都更喜欢OUTER；INNER却在MIXED取得更低观察风险。这使优化路径、有限预算、噪声和TAIL读出成为下一步必须保留的竞争解释。它不证明INNER没有优化自身目标，也不证明某个优化器实现错误或OUTER全局更优。两程序访问状态、Adam矩及修复不同，且TAIL平均模型不是直接优化的状态。','',
'本轮选择的是旧研究后提出的新诊断，独立的是评价prior，不是训练模型或观察数据。MC区间只条件于这个冻结面板，不能替代原q6训练流区间，不能与其拼成新的总体确认。没有为结果追加bank、K、模型或训练。','',
'## 6. 核验与资源','',
f"全部12项SE门通过、0项方向未分辨。所有262144bank的保存分数/Jensen/统计与标签面板审计通过，最大算术差{a['max_score_stat_error']:.3g}；预定8流前2bank、两臂与两准则共1024分数直接概率复算，最大差{ind['max_error']:.3g}。原GH32/40/48对新数据加权损失均通过资格；没有新增积分。",'',
f"主阶段{r['seconds']:.2f}秒；104857600个共享prior向量、419430400 Gaussian，审计另重放6400向量/25600 Gaussian，不增加独立单位。0新观察、0训练、0模型、0优化、0积分节点。源码、输入、数据/参考绑定见[合同](CONTRACT.md)与FREEZE；本地64份评分数组仍被Git忽略。",'',
'作者措辞及下一步见[AUTHOR_UPDATE](AUTHOR_UPDATE.md)，原REC-v4和所有USE结果不回写。']
with (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(report=True,figures=2,new_evaluation=0)))
