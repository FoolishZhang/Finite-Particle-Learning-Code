"""Render the completed fixed-readout comparison, without additional science."""
from pathlib import Path
import os,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_readout_v6_20261005';DOC=ROOT/'docs/08_research_研究工作/REC_v6_readout_20261005'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/recreadout_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(p.read_text(encoding='utf-8'))
r=read(OUT/'RESULTS.json');a=read(OUT/'ARITHMETIC_AUDIT.json');ind=read(OUT/'INDEPENDENT_AUDIT.json')
render_only='--render-only' in sys.argv
assert a['passed'] and ind['completed'] and (render_only or not (DOC/'RESULTS.md').exists())
names={'both_prefer_OUTER':'都偏OUTER','both_prefer_INNER':'都偏INNER','aligned_target_split':'各目标偏自己的终点',
       'reverse_target_split':'反向分离','unresolved':'方向未分辨','unresolved_precision':'精度未过门'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,6.4))
for row,case in enumerate(('NULL','MIXED')):
    for k,color in ((100,'#b45d25'),(400,'#176899')):
        cells=[next(c for c in r['family'] if (c['case'],c['target'],c['K'],c['contrast'])==(case,'F_D',k,label)) for label in ('LAST','TAIL')]
        ys=[c['mean']*1000 for c in cells];axes[row,0].plot([0,1],ys,color=color,linewidth=1,label=f'F{k}')
        for j,c in enumerate(cells):
            mean=c['mean']*1000;lo,hi=[x*1000 for x in c['interval']]
            axes[row,0].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=5,markersize=6)
    exact=next(x for x in r['exact'] if x['case']==case)
    axes[row,1].plot([0,1],[x*1000 for x in exact['arm_risk_deltas']],'-D',color='#6e528b',linewidth=1,markersize=6)
    for col in range(2):
        ax=axes[row,col];ax.axhline(0,color='#555555',linewidth=.7);ax.set_xticks([0,1],['LAST','TAIL']);ax.set_xlim(-.2,1.2);ax.grid(axis='y',alpha=.15)
        ax.set_title(case+(': training-data target difference' if col==0 else ': teacher KL difference'))
        ax.set_ylabel('INNER - OUTER (x 1e-3)')
    if row==0:axes[row,0].legend(frameon=False)
zoom=axes[1,0].inset_axes([.45,.28,.49,.49])
for j,label in enumerate(('LAST','TAIL')):
    c=next(c for c in r['family'] if (c['case'],c['target'],c['K'],c['contrast'])==('MIXED','F_D',400,label))
    mean=c['mean']*1e6;lo,hi=[v*1e6 for v in c['interval']]
    zoom.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color='#176899',capsize=4,markersize=4)
zoom.axhline(0,color='#555555',linewidth=.7);zoom.set_xticks([0,1],['LAST','TAIL']);zoom.set_xlim(-.3,1.3)
zoom.set_title('F400 detail (x 1e-6)',fontsize=8);zoom.tick_params(labelsize=8);zoom.grid(axis='y',alpha=.15)
fig.subplots_adjust(left=.10,right=.98,top=.92,bottom=.13,hspace=.48,wspace=.35)
fig.text(.5,.03,'Positive favors OUTER. Left: fixed-panel MC q24 intervals. Right: quadrature risk differences, no new data intervals.',ha='center',fontsize=8.5)
(DOC/'figures').mkdir(exist_ok=True)
fig.savefig(DOC/'figures/readout_sensitivity.png',dpi=180,facecolor='white');fig.savefig(DOC/'figures/readout_sensitivity.svg',facecolor='white');plt.close(fig)
if render_only:
    print(json.dumps(dict(figures_updated=2,scientific_results_unchanged=True)));raise SystemExit

lines=['# REC-v6：LAST与TAIL的固定读出敏感性','',
'**读出会改变MIXED同数据F400的微小平均排名：LAST略偏INNER，TAIL偏OUTER。但NULL和两个条件的总体F100/F400均仍偏OUTER，MIXED改善/NULL恶化的风险方向也保留。** 因此，TAIL可解释这一局部排名差异，不能解释全部目标—风险关系。','',
'本轮比较同一批128条训练轨迹的LAST与原定TAIL，共256个读出。没有新训练或checkpoint搜索；LAST新增风险求积，四个模型状态共享新的独立评价粒子。主要问题是REC-v5的面板平均目标排名是否依赖TAIL平均。','',
'![读出敏感性](figures/readout_sensitivity.png)','',
'统一差额为INNER−OUTER，正数偏OUTER。左列用原训练标签频率加权F100/F400，误差条是固定面板MC区间；右列为教师KL差，没有新的数据抽样区间。','',
'## 1. 目标排名是否随读出改变','',
'|条件|权重|LAST的F100/F400判断|TAIL的F100/F400判断|','|---|---|---|---|']
for case in ('NULL','MIXED'):
    for target in ('F_D','F_P'):
        cs=[next(x for x in r['classification'] if (x['case'],x['target'],x['readout'])==(case,target,rd)) for rd in ('LAST','TAIL')]
        lines.append(f"|{case}|{target}|{names[cs[0]['result']]}|{names[cs[1]['result']]}|")
lines+=['','以上是每条件32流平均的条件MC判断，不是全部流逐个方向保证。即使交互项非零，也需区分“差额改变”和“排名翻转”。','',
'MIXED的同数据F400在LAST为−0.000003498，q24区间[−0.000005821,−0.000001175]；TAIL为+0.000045790，区间[+0.000043647,+0.000047934]。对应交互为−0.000049289。LAST的方向虽已分辨，优势只有约3.5e−6 nats/观察，不能夸大为实质优化优越性或新的训练总体结论。','',
'## 2. 全部24项MC比较','',
'LAST/TAIL列分别是该读出的INNER−OUTER；INTERACTION是Δ_LAST−Δ_TAIL。q24 Bonferroni alpha .05、df4095，SE门1e−5。','',
'|条件|权重|K|比较|平均差|q24区间|SE|','|---|---|---:|---|---:|---|---:|']
for x in r['family']:
    lo,hi=x['interval'];lines.append(f"|{x['case']}|{x['target']}|{x['K']}|{x['contrast']}|{x['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|{x['SE']:.2g}|")
lines+=['','## 3. 高精度风险与同数据拟合','',
'|条件|读出|OUTER平均KL|INNER平均KL|INNER−OUTER KL|INNER−OUTER同数据NLL|','|---|---|---:|---:|---:|---:|']
for x in r['exact']:
    for ri,rd in enumerate(('LAST','TAIL')):
        lines.append(f"|{x['case']}|{rd}|{x['mean_risks'][ri][0]:.9f}|{x['mean_risks'][ri][1]:.9f}|{x['arm_risk_deltas'][ri]:+.9f}|{x['arm_loss_deltas'][ri][0]:+.9f}|")
lines+=['','下表是逐训练臂的LAST−TAIL风险变化：正值表示该臂的LAST在本面板平均风险较高。它不构成平均方法的普遍优越性或新训练总体检验。','',
'|条件|OUTER的LAST−TAIL KL|INNER的LAST−TAIL KL|','|---|---:|---:|']
for x in r['exact']:lines.append(f"|{x['case']}|{x['readout_risk_deltas'][0]:+.9f}|{x['readout_risk_deltas'][1]:+.9f}|")
lines+=['','## 4. 逐流异质性','',
'以下只统计配对差点估计符号，不给逐流同时区间。','',
'|条件|权重|读出|同偏OUTER|各自目标支持自己终点|同偏INNER|反向分离|','|---|---|---|---:|---:|---:|---:|']
for x in r['classification']:
    c=x['point_sign_counts'];lines.append(f"|{x['case']}|{x['target']}|{x['readout']}|{c['both_OUTER']}|{c['aligned_split']}|{c['both_INNER']}|{c['reverse_split']}|")
lines+=['','## 5. 核验、预算与解释范围','',
f"主参考全部通过：{r['qualified']}；独立GH44参考全部通过：{ind['passed']}。24项MC中{sum(x['precision_passed'] for x in r['family'])}项通过SE门，{sum(x['direction']=='unresolved' for x in r['family'])}项区间跨零。原轨迹提取、标签、全部262144bank的保存分数/Jensen及统计算术通过，最大差{a['max_score_stat_error']:.3g}；2048个独立直接概率分数最大差{ind['max_score_error']:.3g}。",'',
f"主运行{r['seconds']:.2f}秒；104857600个共享prior向量（419430400Gaussian），128个LAST的GH32/40/48共1141374976节点，8个独立GH44共29984768节点。MC审计另重放6400向量，不增加独立样本。TAIL原参考不重算，无新观察、训练或优化。",'',
'两读出属于相同旧轨迹，结果只能解释这个固定有限程序。新的MC区间不替代REC-v4训练流区间，LAST与TAIL风险读数也没有被包装成新的总体确认。没有挑选最优checkpoint或调整平均窗口；原科学结果及失败保持。具体作者判断与后续取舍见[AUTHOR_UPDATE](AUTHOR_UPDATE.md)。']
if not r['qualified'] or not ind['passed']:lines.insert(2,'**风险参考未全部通过，相关风险读数不能据此作确定结论。原失败记录保留。**')
with (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(report=True,figures=2,new_science=0)))
