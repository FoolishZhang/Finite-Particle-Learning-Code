"""Deterministic report tables from the completed original-API experiment."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_invariance'
DOC=ROOT/'docs/08_research_研究工作/真实联合评分接口与无关扩展_20261003'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x):return f'{x:.10g}'
def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    run=read(OUT/'RUN/RESULTS.json');ref=read(OUT/'REFERENCE/MODEL.json')
    zeros=ones=0;maxlog=0.
    for row in run['records']:
        for q in row['probability_diagnostics'].values():
            zeros+=int(np.asarray(q['sigmoid_zero_counts']).sum());ones+=int(np.asarray(q['sigmoid_one_counts']).sum())
            maxlog=max(maxlog,float(np.asarray(q['maximum_absolute_logits']).max()))
    patterns=np.asarray(ref['patterns']);conditional=[]
    for row in ref['records']:
        p=np.asarray(row['main_pmf'])
        for i in range(4):
            for j in range(4):
                if i==j:continue
                for observed in [0,1]:
                    mask=patterns[:,i]==observed
                    conditional.append(dict(t=row['t'],given_index=i,query_index=j,given_value=observed,
                        probability=float(p[mask&(patterns[:,j]==1)].sum()/p[mask].sum())))
    scope='Post hoc interpretation of already-frozen 16-pattern reference PMFs; all ordered pairs retained, no new integration, simulation, intervention API or significance test.'
    interpretation=dict(scope=scope,conditional_probabilities=conditional,highlight=[r for r in conditional if r['given_index']==0 and r['query_index']==1 and r['given_value']==1])
    (OUT/'OBSERVABLE_READOUT.json').write_text(json.dumps(interpretation,indent=2)+'\n',encoding='utf8')
    base=run['primary_family'][0]['stream_values'];padded=run['primary_family'][1]['stream_values']
    summary=dict(source_sha256=sha(__file__),inputs={str((OUT/x).relative_to(ROOT)).replace('\\','/'):sha(OUT/x) for x in ['RUN/RESULTS.json','REFERENCE/MODEL.json']},
        main_effects_passed=sum(p['effect_passed'] for p in run['primary_family']),main_precision_passed=sum(p['precision_passed'] for p in run['primary_family']),
        sigmoid_zero_count=zeros,sigmoid_one_count=ones,maximum_absolute_logit=maxlog,
        descriptive_stream_counts=dict(base_high_worse=sum(v>0 for v in base),padded_high_better=sum(v<0 for v in padded),
            both_signs=sum(a>0 and b<0 for a,b in zip(base,padded)),padded_reverse_streams=[i for i,v in enumerate(padded) if v>0]),
        numerical_qualification=run['numerical_qualification'],scale_control_qualification=run['scale_control_qualification'],
        reference_qualification=ref['qualification_passed'],reference_KL={str(r['t']):r['reference_joint_KL'] for r in ref['records']},
        conditional_readout_sha256=sha(OUT/'OBSERVABLE_READOUT.json'),conditional_readout_scope=scope,
        new_summary_random_draws=0,new_summary_integrals=0,new_tests=0)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    s='''# 完整主要读数

主结果来自未修改的官方SCBLoss.compute_concept_loss，Torch2.5.1 CPU、float32。一个四概念非零均值mixed-sign SPD fixture，65536条完整观测、16独立流；t是相关矩阵off-diagonal的倍率增量，t=.75表示相关幅度乘1.75，不是单个相关系数等于.75。无网络训练或参数优化。

## 1 原四项主家族

K100，每流4096观测。两侧Bonferroni t开发近似，家族α=.05、df15；不是精确有限样本覆盖。

|预定比较|均值|SE|区间下端|区间上端|效应门|SE门|
|---|---:|---:|---:|---:|---|---|
'''
    for p in run['primary_family']:s+=f"|{p['name']}|{f(p['mean'])}|{f(p['SE'])}|{f(p['lower'])}|{f(p['upper'])}|{'pass' if p['effect_passed'] else 'fail'}|{'pass' if p['precision_passed'] else 'fail'}|\n"
    s+=f"\n临界值{f(run['critical_t'])}。效应{summary['main_effects_passed']}/4、SE资格{summary['main_precision_passed']}/4。n0有{summary['descriptive_stream_counts']['base_high_worse']}/16流high评分较差，n256有{summary['descriptive_stream_counts']['padded_high_better']}/16流high评分更好；两种符号在同一流同时出现{summary['descriptive_stream_counts']['both_signs']}/16。反向stream6/7均保留，不声明逐流全部反转。\n"
    s+='''
## 2 所有预定候选对比

每行均为candidate-minus-truth；负值偏好该候选。K400只使用每流前1024条，比较K影响时对应K100_prefix1024，不能与K100_all的不同N混为一个精度实验。下表仅均值/SE，没有给n64、辅助K或低候选新增显著性门。factorized各n值来自同active原API，nlog2解析抵消。

|观测范围|n|t|raw均值|raw SE|factorized均值|factorized SE|
|---|---:|---:|---:|---:|---:|---:|
'''
    for p in run['descriptive_score_tables']:
        if p['t']==0:continue
        r=p['raw_contrast_to_truth'];g=p['factorized_contrast_to_truth']
        s+=f"|{p['scope']}|{p['padding']}|{p['t']}|{f(r['mean'])}|{f(r['SE'])}|{f(g['mean'])}|{f(g['SE'])}|\n"
    s+='''
辅助K400/n256的均值仍负，但该子集误差较大，未设新的成功门，不能确证增加K是否恢复排序，也不声称K增加使偏差更坏。固定K的factorized还可有active粒子偏差；这里只证明其对无关扩展的实现不变性。

![原API排序与共同前缀K对照](../../../artifacts/scbm_invariance/figures/api_candidate_ranking.png)

## 3 预定梯度读数

均在t0；相关方向为D与六个对称协方差坐标梯度的内积。十个active分量、全部SE和每流值保存在原RESULTS，不按结果挑分量。n0没有nuisance尺度，归一化显示NA；实现数组中的0仅为空变量占位。

|范围|n|raw相关方向均值|SE|nuisance尺度梯度/n|SE|
|---|---:|---:|---:|---:|---:|
'''
    for p in run['descriptive_gradient_tables']:
        g=p['correlation_direction'];q=p['nuisance_scale_per_coordinate'];n=p['padding']
        s+=f"|{p['scope']}|{n}|{f(g['mean'])}|{f(g['SE'])}|{f(q['mean']) if n else 'NA'}|{f(q['SE']) if n else 'NA'}|\n"
    s+='''
![方向梯度及可行尺度方向](../../../artifacts/scbm_invariance/figures/api_gradient_directions.png)

尺度梯度的正号与理想总体严格单调性一致，提供受限下降方向；没有运行优化器，不能叫作“学得了尺度修复”。

## 4 数值资格与原API范围

'''
    q=run['numerical_qualification'];s+=f"每流首64观测、两K/全部n/t的API32与独立stable64比较：score {q['score_passed']}/{q['score_count']}、梯度{q['gradient_passed']}/{q['gradient_count']}通过，最大差分别{f(q['maximum_score_difference'])}及{f(q['maximum_normalized_gradient_difference'])}（nuisance分量先按n归一）。差异包括输入/参数舍入和表达变化；stable64不是官方第二种dtype路径。\n\n"
    q=run['scale_control_qualification'];s+=f"zero/floor控制的值{q['value_passed']}/{q['value_count']}、active梯度{q['active_gradient_passed']}/{q['active_gradient_count']}通过。主输入sigmoid零/一计数为{zeros}/{ones}，最大|logit|={f(maxlog)}。完整成功JSON要求有限数值；未出现科学异常或自动重试。\n\n"
    for scale in [0.,1e-6]:
        rows=[r for r in run['scale_controls'] if r['scale']==scale]
        s+=f"尺度{scale:g}的最大constant/factorized值差{f(max(abs(r['value_difference']) for r in rows))}、最大active梯度差{f(max(r.get('maximum_active_gradient_difference',0) for r in rows))}。\n\n"
    s+='''理想数学floor界3.2e−11 nats不等于上述原API的浮点误差承诺。正确解析分解与在大张量中填入常数.5的数学身份，也不要求浮点求和逐bit相同。资格前缀不认证全部底层计算误差。

## 5 不变统计任务的参考与事后解释

'''
    for row in ref['records']:s+=f"t={row['t']}相对truth的参考joint KL={f(row['reference_joint_KL'])}。\n\n"
    s+='''这些风险不随n改变。12次PMF积分的主/变阶/不同根资格通过，最大PMF差约6.31e−12；不是严格余项证书。

以下仅为读取结果后的观察意义解释，使用已冻结16模式PMF，无新积分或统计门：

|t|P(Y2=1 given Y1=1)|
|---:|---:|
'''
    for row in interpretation['highlight']:s+=f"|{row['t']}|{f(row['probability'])}|\n"
    s+='''全部72个有序pair条件概率保存在OBSERVABLE_READOUT.json，例子固定展示前两概念。它是指定模型的观察条件概率，不是SCBM原生干预模块或因果效应测试。

## 6 账目与原始入口

'''
    l=run['ledger'];s+=f"新增完整观测{l['new_complete_observations']}；prior向量{l['prior_vectors']}、prior Gaussian {l['prior_Gaussian_scalars']}；数据Gaussian {l['data_Gaussian_scalars']}、Uniform {l['data_uniforms']}。主科学调用原API {l['official_API_calls']}次，独立stable64控制{l['stable64_control_calls']}次；无NN拟合、无参数优化。实际RUN耗时{f(run['seconds'])}秒，不是完整SCBM端到端吞吐测量。\n\n"
    s+='''[RUN全部读数](../../../artifacts/scbm_invariance/RUN/RESULTS.json) · [确定性参考](../../../artifacts/scbm_invariance/REFERENCE/MODEL.json) · [来源合同](SOURCE_CONTRACT.md)。审计源、补充积分、输入重放和限定底层复算的实际范围另见AUDIT_PLAN及审计收据。32个大NPZ按原规则留本地，公开JSON/hash不是完整二进制备份。旧稿、旧科学和失败保持。
'''
    (DOC/'RESULTS.md').write_text(s,encoding='utf8',newline='\n')
    print(json.dumps(dict(primary=summary['main_effects_passed'],numeric=q,zeros=zeros,ones=ones,maxlog=maxlog,posthoc_conditional=interpretation['highlight'])))
if __name__=='__main__':main()
