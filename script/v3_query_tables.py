"""Readable complete A/B tables, deterministically derived from result JSON."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3_query'
DOC=ROOT/'docs/08_research_研究工作/v3表示与条件查询_20260930'
def read(name):return json.loads((OUT/name).read_text('utf-8'))
def f(x):return f'{x:.9g}'
def main():
    a=read('A_RESULTS.json');b=read('B_RESULTS.json');audit=read('A_AUDIT.json')
    lines=['# A与B的完整可读结果','',
      '本表从冻结JSON派生，无新增拟合／随机数。joint KL单位为总向量nats，query KL为单bit查询的nats；k为已观察位置数。约1e−15的负数保留为浮点残差，不解释为负KL。',
      '', '## A 总体能力矩阵','',
      '|教师|编码|模型|joint KL|k3 query KL|k7 query KL|gap|合格|',
      '|---|---|---|---:|---:|---:|---:|---|']
    for r in a['rows']:
        lines.append('|'+ '|'.join([r['teacher'],r['encoding'],r['family'],f(r['joint_KL']),f(r['queries']['3']['conditional_excess_KL']),f(r['queries']['7']['conditional_excess_KL']),f(r['gap']),str(r['qualified'])])+'|')
    lines+=['','## A 计数摘要的信息下界','','以下是只读取已观察count的最优query log regret，即条件互信息，对固定O/j混合平均。它是CAR下界，不是所有联合模型的下界。','',
            '|教师|编码|k|count-blind KL floor|count-blind Brier floor|','|---|---|---:|---:|---:|']
    for r in audit['blindness_diagnostics']:
        lines.append(f"|{r['teacher']}|{r['encoding']}|{r['k']}|{f(r['count_blind_KL_floor'])}|{f(r['count_blind_Brier_floor'])}|")
    lines+=['','## B 全部32项开发对比','',
            '正值表示product风险减候选风险；每个重复先对两个X平均。区间为事前32项Bonferroni双侧95% t近似，n=16独立训练重复，不是严格有限样本覆盖。精度未过的对比即使区间很正也记未决。实质门为下端>0.001且全部相关拟合合格。','',
            '|世界|编码|候选|k|平均改善|区间下端|区间上端|数值合格|实质收益通过|','|---|---|---|---:|---:|---:|---:|---|---|']
    for r in b['contrasts']:
        lo,hi=r['simultaneous_t_interval']
        lines.append(f"|{r['world']}|{r['encoding']}|{r['family']}|{r['k']}|{f(r['product_minus_model'])}|{f(lo)}|{f(hi)}|{r['contrast_qualified']}|{r['practical_gain']}|")
    lines+=['','## B 原精度失败完整列表','',
            '|世界|编码|重复|X索引|模型|原gap|','|---|---|---:|---:|---|---:|']
    for r in b['rows']:
        if not r['qualified']:lines.append(f"|{r['world']}|{r['encoding']}|{r['rep']}|{r['x']}|{r['family']}|{f(r['gap'])}|")
    lines+=['','全部失败原样保留，没有追加优化／种子救门；`x=0/1`分别代表X=−1/+1。700个B问题中695合格，24/32对比具有全部数值资格，8/32通过事前实质门。','',
            '## B 总体条件锚与pooled诊断','','pooled故意忽略X，是不同信息合同，不与使用X的预测器作公平排名。','',
            '|类别|世界|编码|X索引|模型|joint KL|k3 query KL|k7 query KL|合格|','|---|---|---|---|---|---:|---:|---:|---|']
    for kind,key in [('conditional_population','population_rows'),('pooled','pooled_rows')]:
        for r in b[key]:lines.append(f"|{kind}|{r['world']}|{r['encoding']}|{r.get('x','NA')}|{r['family']}|{f(r['joint_KL'])}|{f(r['queries']['3']['conditional_excess_KL'])}|{f(r['queries']['7']['conditional_excess_KL'])}|{r['qualified']}|")
    lines+=['','有限640个条件分布的逐模型参数、优化路径、所有风险见`artifacts/v3_query/B_RESULTS.json`；训练直方图在B_COUNTS.json。A全部PMF也以JSON公开。C另见C_RESULTS，不与A/B的置信家族混合。','']
    (DOC/'RESULTS_AB.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Wrote complete A/B readable tables; no fits or samples.')
    with (OUT/'C/SUMMARY.csv').open(encoding='utf-8',newline='') as source:c=list(csv.DictReader(source))
    lines=['# C 同模型查询的全部36个读数','',
           '主指标是同一正确count-state AR模型下的标量excess Brier；每格512查询，16独立流。36项Bonferroni同时95% t近似，不设强制收益门。Prefix与mixed的target分布不同，差值不是纯位置因果效应。','',
           '|d|观察数|布局|K|excess Brier|区间下端|区间上端|ESS/K|DP毫秒|IS路径毫秒|最大随机表毫秒|prefix直接毫秒|',
           '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in c:
        vals=[row['d'],row['observed'],row['layout'],row['K']]+[f(float(row[key])) for key in ['mean_excess_Brier','lower','upper','mean_ESS_fraction']]
        vals+=[f(1000*float(row[key])) if row[key] else 'NA' for key in ['mean_dp_seconds','mean_is_path_seconds','mean_uniform_generation_seconds','mean_prefix_direct_seconds']]
        lines.append('|'+ '|'.join(vals)+'|')
    lines+=['','DP为精确同模型查询，IS是完整路径likelihood weighting。IS路径时间未包含单列的最大256粒子随机数表；它被各K重用，不可称零成本，也不能将这一表的全部生成成本说成独立K16实现必需成本。时间含当前Python实现开销、未隔离同机干扰，不作算法速度定理或严格同预算胜利。','',
            '2052次预定枚举验证；全部6144查询以另一个scaled forward DP及保存路径权重重放。公开数据：`artifacts/v3_query/C/SUMMARY.csv`、`STREAM_SCORES.csv`、`streams/*.json`；本地完整proposal路径登记hash。','']
    (DOC/'RESULTS_C.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Wrote complete C table from frozen CSV; no fits or samples.')
if __name__=='__main__':main()
