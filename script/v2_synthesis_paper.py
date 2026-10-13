"""Assemble a reviewable Chinese v2 working manuscript without changing v1."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/07_papers_论文工作/ParticleJointSupervision/v2_working_20260930'
RESEARCH=ROOT/'docs/08_research_研究工作/v2收束与v3方向_20260930'
OUT=ROOT/'artifacts/v2_synthesis_20260930'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(s.rstrip()+'\n')
def proof_text():
    source=RESEARCH/'THEORY_UNKNOWN_SCALES.md'
    text=source.read_text(encoding='utf-8')
    def link(match):
        target=match.group(1)
        if '://' in target or target.startswith('#'):return match.group(0)
        path,sep,anchor=target.partition('#')
        absolute=(source.parent/path).resolve()
        relative=Path(os.path.relpath(absolute,PAPER)).as_posix()
        return ']('+relative+(sep+anchor if sep else '')+')'
    text=re.sub(r'\]\(([^)]+)\)',link,text)
    text=text.replace('\\[','$$').replace('\\]','$$')
    text=re.sub(r'\\\((.*?)\\\)',lambda m:'$'+m.group(1)+'$',text,flags=re.S)
    return text.rstrip()+'\n'

def build():
    assert not (OUT/'MANIFEST.json').exists(),'delivery snapshot is already sealed'
    assert not (PAPER/'SOURCE_RECEIPT.json').exists(),'paper snapshot exists; do not overwrite silently'
    write(PAPER/'PROOF_UNKNOWN_SCALES.md',proof_text())
    write(PAPER/'README.md','''# v2.0 中文工作初稿

[正文](MANUSCRIPT_ZH.md) · [新证明附件](PROOF_UNKNOWN_SCALES.md) · [证据与限制](EVIDENCE.md)

本稿围绕可学习性、有限评分表示及统计粗化三条不同证据组织。它是同一研究的v2工作版本，尚未投稿定稿；此次提供Markdown正文和证明附件，没有生成新的PDF或英文版本。v1中英文定稿、PDF与历史科学冻结完整保留。

本轮未知尺度定理针对split连续反演估计器，不追认旧固定网格程序；G学习实验不是原product-mixture界的直接验证。后续方法化设想独立见[研究方向](../../../08_research_研究工作/v2收束与v3方向_20260930/RESEARCH_DIRECTION.md)，不作为当前稿件的已完成贡献。

v2.0作为本次可审查快照记录来源；后续作者修改可另建v2工作快照，不改写v1，也不意味着v2科学内容已经终稿。
''')
    write(PAPER/'EVIDENCE.md','''# v2.0 证据与主张对应

本文总joint KL均为完整向量nats。正文中的统计家族、网格、程序和理论估计器分别解释；既有科学结果保持原冻结，工作稿不产生新训练。

|主张|精确来源|不可越过的边界|
|---|---|---|
|任意K-product观察KL下界与原代理保证缺口|[混合表示证明](../../../08_research_研究工作/全局约束与采样门槛_20260929/MIXTURE_BOUND.md)、[v1全文](../v1_final_20260930/MANUSCRIPT_ZH.md)|只直接约束该有限product类；不直接约束宏观group scorer|
|两尺度CLT与非消失信息|[非交换理论](../../../08_research_研究工作/非交换适用域_v2_20260930/THEORY.md)、[错对称化信息](../../../08_research_研究工作/结构预算与对称化代价_v2_20260930/MISMATCH_THEORY.md)|固定合法参数点，不把弱收敛当熵收敛|
|旧已知尺度ROW、Fano与表示对比|[已封存E](../../../08_research_研究工作/结构先验释放与分组学习_v2_20260930/SAMPLE_VS_PARTICLE_SEPARATION.md)|Fano只约束精确组恢复；样本N与表示K资源不同|
|新未知尺度总风险|[原新证明](../../../08_research_研究工作/v2收束与v3方向_20260930/THEORY_UNKNOWN_SCALES.md)、[独立审查](../../../08_research_研究工作/v2收束与v3方向_20260930/THEORY_REVIEW.md)|已知c、正紧区间、尺度分离、样本split和连续反演；固定网格不是此估计器|
|group有限目标、未知组学习与先验错配|[A/B/C完整结果](../../../08_research_研究工作/结构先验释放与分组学习_v2_20260930/RESULTS.md)|既有4项精度失败保留，B有限主门未过；A/C有bank复用|
|原16个G10输出的风险分解|[本轮派生表](../../../../artifacts/v2_synthesis_20260930/decomposition/READABLE.md)|后验诊断，无新随机观察或拟合；不作因果份额|
|保守样本公式及Fano代入|[解析代入](../../../../artifacts/v2_synthesis_20260930/decomposition/ANALYTIC_SUBSTITUTION.json)|四种确定性积分不是随机重复，不验证实际65536样本充分|
|独立验证控制|[D结果](../../../08_research_研究工作/结构先验释放与分组学习_v2_20260930/D_RESULTS.md)|仅相对训练H0的pair增益；四倍独立验证预算；不认证joint或整个同质类|
|变量聚类近邻与原创性范围|[定向文献核查](../../../08_research_研究工作/v2收束与v3方向_20260930/LITERATURE_POSITION.md)|不认证ROW首创／最优，不将PECOK不满足条件当方法胜利|

旧A/B/C共2240个有限数据拟合，D另32个H0拟合与32次验证选择。旧训练／验证观察、潜粒子及数组继续以原记录为准；本轮不重复计算或增加这些数量。

本文没有提供新的自然任务收益、预算匹配方法对照、未知c保证、连续原F_K全局最优或一般推断复杂性结论。最近邻比较依据可读的作者稿及已核对元数据，完整文献覆盖和投稿格式仍需后续作者工作。
''')
    files=[PAPER/'MANUSCRIPT_ZH.md',PAPER/'PROOF_UNKNOWN_SCALES.md',PAPER/'README.md',PAPER/'EVIDENCE.md']
    sources=[RESEARCH/'THEORY_UNKNOWN_SCALES.md',RESEARCH/'THEORY_REVIEW.md',RESEARCH/'LITERATURE_POSITION.md',
        OUT/'decomposition/RESULTS_FREEZE.json',ROOT/'artifacts/structure_learning_v2/MANIFEST.json',
        ROOT/'artifacts/paper_v1_final/VERSION_LOCK.json',Path(__file__)]
    receipt=dict(time_utc=datetime.now(timezone.utc).isoformat(),version='v2.0-working-zh',paper_final=False,
        source_baseline='cab00907ffa5c676c2cd306e19d2fb052b24f76f',new_stochastic_experiments=0,new_model_fits=0,
        deliverable_format='Markdown manuscript and proof appendix; no new PDF',
        source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        output_files={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write(PAPER/'SOURCE_RECEIPT.json',json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(dict(paper_snapshot_created=True,version=receipt['version'],new_training=0)))

def check():
    r=json.loads((PAPER/'SOURCE_RECEIPT.json').read_text('utf-8'))
    for k in ['source_files','output_files']:
        for n,h in r[k].items():assert sha(ROOT/n)==h,n
    assert (PAPER/'PROOF_UNKNOWN_SCALES.md').read_text('utf-8')==proof_text()
    text=(PAPER/'MANUSCRIPT_ZH.md').read_text('utf-8')
    assert not any(x in text for x in ['TODO','TBD',':chatgpt-content-reference'])
    assert '一般不能' in text and 'O(1/\\log d)' in text and '未过' in text
    print(json.dumps(dict(paper_check_pass=True,outputs=len(r['output_files']),version=r['version'],v1_not_modified=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','check']);args=p.parse_args()
    build() if args.phase=='build' else check()
