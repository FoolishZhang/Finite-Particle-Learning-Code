"""Readable tables and scientific figures from completed frozen T/A/B/B2 only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3_select'
DOC=ROOT/'docs/08_research_研究工作/v3结构选择与查询诊断_20260930'
def read(name):return json.loads((OUT/name).read_text('utf-8'))
def csvread(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def fmt(v):return f'{float(v):.9g}'
def save(fig,name):
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def tables():
    a=read('A/RESULTS.json');fits=read('A/FITS.json');methods=list(a['population'][0]['candidates'])
    lines=['# 结构选择全部结果','',
           '本文仅由冻结JSON／CSV派生。joint为总向量nats，query为每次单bit nats；浮点零附近负数不解释成负KL。不同包的教师、信息合同、置信家族分别保留。','',
           '## A 总体能力锚','', '|教师|编码|候选|joint KL|k7 query KL|k3 query KL|合格|','|---|---|---|---:|---:|---:|---|']
    for row in a['population']:
        for m,v in row['candidates'].items():lines.append(f"|{row['teacher']}|{row['encoding']}|{m}|{fmt(v['joint_KL'])}|{fmt(v['query7'])}|{fmt(v['query3'])}|{v['qualified']}|")
    lines+=['','## A 全部40项主对比','',
            '固定Ntrain8192／Nval512，16独立重复、40项Bonferroni同时95% t开发近似；正值为左侧规则风险减右侧规则风险。不是严格覆盖或等效性证书；[0,0]只表示这些重复给出相同风险。','',
            '|教师|编码|对比|平均改善|下端|上端|数值合格|下端>0.001|','|---|---|---|---:|---:|---:|---|---|']
    for r in a['contrasts']:
        lines.append(f"|{r['teacher']}|{r['encoding']}|{r['contrast']}|{fmt(r['mean'])}|{fmt(r['interval'][0])}|{fmt(r['interval'][1])}|{r['qualified']}|{r['practical_positive']}|")
    lines+=['','## A 两个训练预算和两个验证预算的全部程序均值','',
            '只有上节40项属于主门；此表保留较低预算、固定方法与选择程序的全部读数。','',
            '|教师|编码|Ntrain|Nval|程序|平均query7|平均query3|平均joint KL|','|---|---|---:|---:|---|---:|---:|---:|']
    teachers=list(dict.fromkeys(u['teacher'] for u in a['units']))
    for teacher in teachers:
        for code in ['native','flip21']:
            for n in [512,8192]:
                units=[u for u in a['units'] if (u['teacher'],u['encoding'],u['ntrain'])==(teacher,code,n)]
                for m in methods:
                    vals=[u['candidates'][m] for u in units]
                    lines.append(f"|{teacher}|{code}|{n}|NA|{m}|"+'|'.join(fmt(np.mean([v[k] for v in vals])) for k in ['query7','query3','joint_KL'])+'|')
                for nv in [128,512]:
                    for m in ['joint_holdout','query_holdout','moment_alarm']:
                        vals=[next(s for s in u['selections'] if s['nval']==nv)[m]['risk'] for u in units]
                        lines.append(f"|{teacher}|{code}|{n}|{nv}|{m}|"+'|'.join(fmt(np.mean([v[k] for v in vals])) for k in ['query7','query3','joint_KL'])+'|')
    lines+=['','## B 全部64项区间','',
            '32重复、64项同时t近似，全部方向和主门保持原冻结。b1/4在单目标query上的重复读数不是独立确认。','',
            '|d|Ntrain|b|Nval|对比|改善|下端|上端|主门|实质通过|','|---:|---:|---:|---:|---|---:|---:|---:|---|---|']
    for r in csvread('B/INTERVALS.csv'):
        lines.append('|'+ '|'.join(r[k] if k in ['d','Ntrain','b','Nval','contrast','primary','practical_gain'] else fmt(r[k]) for k in ['d','Ntrain','b','Nval','contrast','mean_gain','lower','upper','primary','practical_gain'])+'|')
    lines+=['','## B2 同总标签的全部8项区间','',
            '看到B后另行冻结的开发性复用；没有新抽样。池化后没有独立验证，速度也不必更快。','',
            '|d|旧Ntrain|旧Nval|Npool|HQ−pooledTMI|下端|上端|主门|实质通过|','|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in csvread('B2/INTERVALS.csv'):
        lines.append('|'+ '|'.join(r[k] if k in ['d','Ntrain','Nval','Npool','primary','practical_gain'] else fmt(r[k]) for k in ['d','Ntrain','Nval','Npool','mean_gain','lower','upper','primary','practical_gain'])+'|')
    t=read('T/SUMMARY.json');lines+=['','## T 全候选能力参考','',
        f"全部{t['population_fits']}棵d6二叉laminar树均数值合格；finite-box联合最优KL的浮点上下界为[{fmt(t['qualified_boxed_union_joint_KL_lower'])}, {fmt(t['boxed_union_joint_KL_upper'])}]。不是无界RC的定量下界，不是query优化界。全部节点、参数与逐树读数在T/FITS.json、TREE_REGISTRY.json及PMFS.json公开。",'',
        f"A有{len(fits)}个真实拟合调用、2970个候选分配；71总体、2344有限数据。T945总体，B640闭式，B2额外1280闭式，共5280科学拟合。参数缓存引用和核验重放不增加独立模型数。",'']
    (DOC/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

def figures():
    a=read('A/RESULTS.json');analytic=read('T/ANALYTIC_CHECKS.json');tfits=read('T/FITS.json');summary=read('T/SUMMARY.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    d3=analytic['d3_projection'];x=np.arange(2)
    axes[0].bar(x-.17,[d3['joint_KL_before'],d3['joint_KL_after']],width=.32,label='Joint KL',color='#0072b2')
    axes[0].bar(x+.17,[0,d3['query_after']['conditional_excess_KL']],width=.32,label='Specified query KL',color='#d55e00')
    axes[0].set_xticks(x,['CAR projection','Add pair + exact refit']);axes[0].tick_params(axis='x',labelsize=9)
    axes[0].set_ylabel('KL (joint/vector and query/bit differ)');axes[0].set_title('d=3: joint improves, query worsens');axes[0].legend(fontsize=8)
    for col,(name,label,color) in enumerate([('moment_alarm','Moment alarm','#d55e00'),('query_holdout','Query holdout','#0072b2')]):
        units=[u for u in a['units'] if (u['teacher'],u['encoding'],u['ntrain'])==('blindspot','native',8192)]
        vals=[next(s for s in u['selections'] if s['nval']==512)[name]['risk']['query7'] for u in units]
        axes[1].scatter(col+np.linspace(-.08,.08,16),vals,color=color,s=18,alpha=.65)
        axes[1].plot(col,np.mean(vals),'_',color='black',markersize=22)
    axes[1].set_xticks([0,1],['Moment alarm','Query holdout']);axes[1].set_ylim(bottom=0)
    axes[1].set_title('d=8 blindspot: N=8192, V=512');axes[1].set_ylabel('Conditional excess log loss (nats/bit)')
    # These 945 points enumerate structures; they are not independent random replicates.
    key='joint_KL' if 'joint_KL' in tfits[0] else 'KL'
    values=sorted(float(r[key]) for r in tfits)
    axes[2].plot(range(1,946),values,color='#009e73',linewidth=2)
    axes[2].axhline(analytic['d6']['joint_KL_uniform'],color='#777777',linestyle='--',label='Uniform model')
    axes[2].axhline(0,color='black',linewidth=.7)
    axes[2].set_xlabel('All 945 labelled binary trees, sorted');axes[2].set_ylabel('Joint KL (nats/vector)')
    axes[2].set_title('d=6: every frozen RC box is insufficient');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'query_and_structure_limits')
    b=csvread('B/SUMMARY.csv');pooled=csvread('B2/SCORES.csv');fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    methods=['JOINT_MI','TARGET_MI','HOLDOUT_JOINT','HOLDOUT_QUERY']
    selected=[next(r for r in b if (r['d'],r['Ntrain'],r['b'],r['Nval'],r['method'])==('64','1024','1','512',m)) for m in methods]
    axes[0].bar(range(4),[float(r['mean_query_KL']) for r in selected],color=['#d55e00','#0072b2','#d55e00','#0072b2'])
    axes[0].set_xticks(range(4),['Joint MI','Target MI','Joint holdout','Query holdout'],rotation=20,ha='right')
    axes[0].set_ylabel('Conditional excess log loss (nats/bit)');axes[0].set_title('d=64, N=1024, V=512, one retained edge')
    for i,(m,label,color) in enumerate([('ORIGINAL_HOLDOUT_QUERY','64 train + 512 validation','#d55e00'),('POOLED_TARGET_MI','576 train; fixed Target MI','#0072b2')]):
        vals=[float(r['query_KL']) for r in pooled if (r['d'],r['Ntrain'],r['Nval'],r['b'],r['method'])==('64','64','512','1',m)]
        assert len(vals)==32
        axes[1].scatter(i+np.linspace(-.09,.09,32),vals,color=color,s=17,alpha=.65);axes[1].plot(i,np.mean(vals),'_',color='black',markersize=23)
    axes[1].set_xticks([0,1],['64 train + 512 validation','576 train; fixed Target MI'],fontsize=9)
    axes[1].set_ylim(bottom=-.002);axes[1].set_ylabel('Conditional excess log loss (nats/bit)');axes[1].set_title('Same 576 labels; pooling leaves no validation')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'objective_and_label_budget')
if __name__=='__main__':
    assert not (OUT/'MANIFEST.json').exists(),'sealed delivery: create a new version instead of overwriting figures/tables'
    tables();figures();print('Created complete readable tables and two figures; no fits or random draws.')
