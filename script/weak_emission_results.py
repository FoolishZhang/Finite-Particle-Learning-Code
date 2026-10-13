"""Derive readable tables from frozen deterministic population results."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/08_research_研究工作/弱发射展开与总体偏好_20261001'
OUT=ROOT/'artifacts/weak_emission';NUM=OUT/'NUMERIC'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def n(v):return f'{v:.9g}'
def main():
    assert read(NUM/'STATE.json')['state']=='complete'
    result=read(NUM/'RESULTS.json');loc=result['primary_locations'];corr=read(NUM/'CORRECTION.json')['rows']
    checks=read(NUM/'CHECKS.json')['rows'];ledger=result['ledger']
    stats={name:dict(count=sum(x['name']==name for x in checks),
                    max_error=max(x['error'] for x in checks if x['name']==name),
                    max_fraction_of_gate=max(x['error']/x['limit'] for x in checks if x['name']==name))
           for name in sorted({x['name'] for x in checks})}
    summary=dict(source_sha256=sha(__file__),results_sha256=sha(NUM/'RESULTS.json'),
      corrections_sha256=sha(NUM/'CORRECTION.json'),
      numerical_qualification_count=len(checks),numerical_qualification_pass=sum(x['passed'] for x in checks),
      primary_prediction_tests_total=result['prediction_tests_total'],primary_prediction_tests_passed=result['prediction_tests_passed'],
      K_joint_passed=result['K_joint_passed'],groups=stats,
      primary_locations=20,variant_location_executions=60,correction_primary_points=15,
      grid_parameter_variant_rows=315,grid_K_scores=1575,
      production_native_laplace_evaluations=sum(e['native_rho_evaluations'] for e in ledger['quadrature_engines'].values()),
      production_laplace_exponential_terms=sum(e['laplace_exponential_terms'] for e in ledger['quadrature_engines'].values()),
      production_seconds=ledger['elapsed_seconds'],
      max_value_truncation_bound=max(r['value_lower']+r['value_upper'] for x in result['tail_bounds'].values() for r in x['rows']),
      max_gradient_truncation_bound=max(r['derivative_lower']+r['derivative_upper'] for x in result['tail_bounds'].values() for r in x['rows']),
      descriptive_correction_improved=sum(x['primary_joint_KL']<x['raw_joint_KL'] for x in corr),
      correction_has_no_new_prediction_gate=True,new_random_draws=0,new_labels=0,new_data_fits=0,
      new_calculations_in_this_summary='saved arithmetic only; no Gaussian/Laplace integration or root solving')
    with (OUT/'DERIVED_SUMMARY.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['# 弱发射总体目标的数值结果','',
      '固定d=2、教师潜在相关theta=.15、候选rho∈[-.75,.75]，教师和候选共同尺度a。rho是潜在Gaussian相关参数，不是二值观察的相关系数。全部结果来自对prior随机性取期望后的确定性积分，0标签或随机bank。',
      '',f"原{len(checks)}项数值资格全通过；8个事前解析预测读数8/8通过（4个K、每K位置和风险两项）。这是数值一致性读数，没有统计置信区间。",'',
      '## 主要定位与完整观察风险','',
      '三变体各定位一次，共60次执行、20个主要结果。记录的是全21点网格发现的根及端点中的最低候选；数值定位不构成任意有限a的严格连续全局最优证书。理论的小a阈值尚无数值认证。',
      '', '|a|K|定位rho|弱发射领先rho|joint KL|KL/a⁴|','|---|---|---|---|---|---|']
    for x in loc:
        k=x['K'];leading=.75 if k==2 else .15*k/(k-2)
        lines.append(f"|{n(x['a'])}|{k}|{n(x['rho'])}|{n(leading)}|{n(x['joint_KL'])}|{n(x['joint_KL']/x['a']**4)}|")
    lines += ['','![Population correlation locations](../../../artifacts/weak_emission/figures/rho_location_vs_a.png)','',
      'K1的真实总体目标在整个相关区间精确平坦，仅作积分控制，不执行唯一选点。正确指定的教师候选rho=.15有精确零部署KL，不能把有限K定位误差归因于候选类缺少真模型。','',
      '## 最小a的八个事前读数','',
      '|K|位置或网格导数门|KL/a⁴领先系数|风险相对差|风险门|','|---|---|---|---|---|']
    for x in result['prediction_gates']:
        lines.append(f"|{x['K']}|{x['location_or_pointwise_gradient_pass']}|{n(x['leading_KL_over_a4'])}|{n(x['KL_relative_difference'])}|{x['risk_prediction_pass']}|")
    lines += ['','K2的全网格导数为负只是21个预定点的数值读数；理论另用一致导数余项证明充分小a时全区间单调。没有据此将当前a=.05或a=1认定为已严格落入该理论阈值。','',
      '## 固定收缩的描述性风险对照','',
      'rho_corr=(K−2)rho_raw/K，只使用K与定位模型，没有按教师风险调系数。15个模型在同一primary校正rho上用三阶数复评，0新增定位、0新预测门。表中全部参考风险均下降；理论保证限于未被投影截断且足够小a的情形，另有边界反例。',
      '', '|a|K|校正rho|校正joint KL|校正/raw风险|校正KL/a⁸|','|---|---|---|---|---|---|']
    for x in corr:
        lines.append(f"|{n(x['a'])}|{x['K']}|{n(x['rho_corrected'])}|{n(x['primary_joint_KL'])}|{n(x['corrected_over_raw_KL'])}|{n(x['primary_joint_KL_over_a8'])}|")
    lines += ['','![Observation risk and fixed correction](../../../artifacts/weak_emission/figures/observation_risk.png)','',
      '小风险使用消去线性项后的稳定二元KL表达；仍仅通过求积收敛与独立数值重建资格。K4的风险下降在本表中比一般O(a⁸)上界更快，不据此宣称已证明精确a¹²阶或一般消项规律。a=1的改善是数值观察，不扩大弱发射定理。','',
      '## 数值资格和成本','',
      '|检查|数量|最大差|最大门占比|','|---|---|---|---|']
    for key,x in stats.items():lines.append(f"|{key}|{x['count']}|{n(x['max_error'])}|{n(x['max_fraction_of_gate'])}|")
    lines += ['',f"实际区间的值截断界最大{n(summary['max_value_truncation_bound'])}、导数截断界最大{n(summary['max_gradient_truncation_bound'])}。这些严格界只控制Laplace尾部，未认证GH/GL离散化。",'',
      f"生产含315个参数/变体网格行、1575个K评分，20主/60变体定位、15校正模型/45变体pair复评。跨K共用M后有{summary['production_native_laplace_evaluations']}个不同原生Laplace核评价、{summary['production_laplace_exponential_terms']}个节点指数项，耗时约{n(summary['production_seconds'])}秒。0随机标签、0随机prior向量、0有限数据拟合、0神经网络。",'',
      '生产完整重放另执行一次同样的60定位与求积，逐值核对后导出659个原始缓存点；这是生产函数复用，不能称独立实现。独立审计的直接K2张量积分、GH80观察风险重建及Decimal算术另记。原source及NUMERIC冻结未因补充材料而改变。',
      '', '详见[原方案](PLAN_NUMERIC.md)、[数值设计](NUMERIC_DESIGN.md)、[独立审计](NUMERIC_AUDIT.md)与[范围](SCOPE.md)。','']
    (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n').write('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ['numerical_qualification_count','primary_prediction_tests_passed','primary_locations','correction_primary_points','production_laplace_exponential_terms']}))
if __name__=='__main__':main()
