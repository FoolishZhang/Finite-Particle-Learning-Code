"""Postprocessing of fixed evaluation statistics; no new tests or draws."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_frozen_evaluation'
DOC=ROOT/'docs/08_research_研究工作/冻结表示的条件与总体目标_20261003'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def run():
    assert not (OUT/'VERSION_LOCK.json').exists()
    stats=read(OUT/'STATISTICS/RESULTS.json');ev=read(OUT/'EVAL/RESULTS.json')
    ref=read(OUT/'REFERENCE/RESULTS.json');sel=read(OUT/'MODEL_SELECTION.json')
    primary=stats['primary_family'];p1,p2,p3,p4=primary
    half={key:p1[key]+p3[key] for key in ('mean','lower','upper')}
    excess={key:p2[key]+p3[key] for key in ('mean','lower','upper')}
    derived=dict(source_sha256=sha(__file__),statistics_sha256=sha(OUT/'STATISTICS/RESULTS.json'),
        half_candidate_contrast=half,half_excess_contrast=excess,
        half_candidate_order_positive_on_joint_event=half['lower']>0,
        scope='Post hoc interval arithmetic from the original four-test simultaneous development intervals. c_half=P1+P3; e_half=P2+P3. No tighter unadjusted test or additional family. Conditional on the original frozen panel and numerical population weights.',
        no_whole_curve_or_minimizer_recovery=True,new_tests=0,new_draws=0)
    save(OUT/'DERIVED_INTERVALS.json',derived)
    kl=np.array(ref['reference_joint_KL']);chosen=stats['selected']['selected_point_indices']
    old=kl[:,3,21];new=np.array([kl[m,3,j] for m,j in enumerate(chosen)])
    risk=dict(source_sha256=sha(__file__),statistics_sha256=sha(OUT/'STATISTICS/RESULTS.json'),
        reference_sha256=sha(OUT/'REFERENCE/RESULTS.json'),
        old_LAST_mean=float(old.mean()),selected_mean=float(new.mean()),
        per_model=[dict(model=m,old_LAST_t=sel['models'][m]['LAST']['t'],selected_t=stats['selected']['selected_t'][m],
            old_LAST_joint_KL=float(old[m]),selected_joint_KL=float(new[m]),difference=float(new[m]-old[m])) for m in range(16)],
        improved=int((new<old).sum()),worse=int((new>old).sum()),equal=int((new==old).sum()),
        scope='Post hoc descriptive risk readout for the sixteen already selected diagnostic comparators. Reuses qualified PMFs; no new training, integration, or risk test. Does not predict the effect of continuing joint FREE optimization.',
        new_tests=0,new_integrals=0,new_fits=0,new_diagnostic_selections=0)
    save(OUT/'EXPLORATORY_SELECTED_RISK.json',risk)
    bands=stats['diagnostic_bands'];profile_names=['TAIL learned','TAIL half','TAIL FACT','LAST learned','LAST FACT']
    score=np.array(bands['relative_score']['mean']);price=np.array(bands['price_t']['mean'])
    table=[]
    for c,name in enumerate(profile_names):
        table.append(dict(profile=name,P_t14_contrast=float(score[16,c,14,0]),D_t14_contrast=float(score[16,c,14,1]),
            P_Price_at_truth=float(price[16,c,10,0]),D_Price_at_truth=float(price[16,c,10,1])))
    mean_gain=np.array(stats['selected']['validation_per_model_gains']).mean(axis=0)
    summary=dict(source_sha256=sha(__file__),
        inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
          [OUT/'EVAL/RESULTS.json',OUT/'REFERENCE/RESULTS.json',OUT/'STATISTICS/RESULTS.json',OUT/'MODEL_SELECTION.json']},
        derived_intervals_sha256=sha(OUT/'DERIVED_INTERVALS.json'),
        exploratory_risk_sha256=sha(OUT/'EXPLORATORY_SELECTED_RISK.json'),
        primary_effects_passed=stats['primary_effects_passed'],primary_precision_passed=stats['primary_precision_passed'],
        primary_family_size=4,diagnostic_family_size=7480,statistical_families_separate=True,
        new_evaluation_replicates=32,old_training_models_fixed=16,
        conditional_on_old_panel_not_new_training_confirmation=True,
        numerical_qualification=ev['numerical_qualification'],reference_qualification=ref['qualification_passed'],
        evaluation_ledger=ev['ledger'],reference_ledger=ref['accounting'],
        evaluation_seconds=ev['seconds'],reference_seconds=ref['seconds'],
        profile_readouts=table,reference_KL_at_t14=float(kl[0,0,14]),
        validation_model_mean_gain=mean_gain.tolist(),
        validation_negative_models=np.flatnonzero(mean_gain<0).tolist(),
        diagnostic_candidate_selections=16,new_original_training_runs=0,new_neural_fits=0,
        new_summary_RNG=0,new_summary_integrals=0,new_summary_tests=0)
    save(OUT/'SUMMARY.json',summary)
    lines=['# 冻结表示独立评估的全部主要读数','',
        '固定16个旧FREE模型及原数据；独立单位为32个全新MC组，每组512个K100 bank。'
        'P表示冻结数值生成权重，D表示各自旧数据频数；区间条件于这一panel，不是新的训练算法总体区间。','',
        '## 四项预定主比较','',
        '|比较|平均值|SE|区间下端|区间上端|df|效应|精度|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for row in primary:
        lines.append(f'|{row["name"]}|{row["mean"]:.10f}|{row["SE"]:.10f}|{row["lower"]:.10f}|{row["upper"]:.10f}|{row["df"]}|{row["effect_passed"]}|{row["precision_passed"]}|')
    lines+=['','P1/P2要求上端<0，P3/P4要求下端>0；全部SE门5e−5。'
        'P1…P3使用32组，P4只用后16组验证前16组选出的比较点。'
        '一个四项Bonferroni双侧t开发近似家族；不称精确覆盖，也不把P1…P3的区间说成条件于选点。','',
        '## 同一候选对的曲线读数','',
        '|表示|P中t14−truth|D中t14−truth|P的truth处Price|D的truth处Price|',
        '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append('|'+row['profile']+'|'+ '|'.join(f'{row[k]:.10f}' for k in ['P_t14_contrast','D_t14_contrast','P_Price_at_truth','D_Price_at_truth'])+'|')
    lines+=['',f't14实际保存值为{sel["grid_t"][14]:.17g}，表示潜相关幅度乘1.3；其参考joint KL为{kl[0,0,14]:.12f}。'
        '表中梯度为描述读数，完整7480点同时带另存STATISTICS/RESULTS；其family与四主门分开，不合称总报告95%。'
        'Price与path是期望相同的不同估计量，不能把本评估Price方差当原训练路径梯度噪声。','',
        '## 原共同区间的派生解释','',
        f'由c_half=P1+P3，半尺度候选差均值{half["mean"]:.10f}，保守区间[{half["lower"]:.10f},{half["upper"]:.10f}]。'
        '这是事后相加原共同区间的逻辑推论，继承原开发近似事件，不是新增检验。只描述t14与truth的这对候选，不宣称全曲线或极小点恢复。','',
        f'同理e_half=P2+P3，均值{excess["mean"]:.10f}，保守区间[{excess["lower"]:.10f},{excess["upper"]:.10f}]。'
        '没有从P3正值自动推断所有参数点的绝对偏差变小。','',
        '## LAST选点及独立验证','',
        '|旧model|实际LAST t|诊断t|后16组平均D改进|原LAST参考KL|诊断参考KL|',
        '|---:|---:|---:|---:|---:|---:|']
    for m,row in enumerate(risk['per_model']):
        lines.append(f'|{m}|{row["old_LAST_t"]:.8f}|{row["selected_t"]:.8f}|{mean_gain[m]:.10f}|{row["old_LAST_joint_KL"]:.10f}|{row["selected_joint_KL"]:.10f}|')
    lines+=['',f'负的逐model验证均值为{summary["validation_negative_models"]}；保留全部反向与零值。'
        'P4只保证panel平均条件切片改进，不是每个model都改善，也不是连续全局优化证书。','',
        f'风险列为事后描述：原LAST平均{old.mean():.10f}，诊断候选平均{new.mean():.10f}，'
        f'{risk["improved"]}改善/{risk["worse"]}变差/{risk["equal"]}相同。'
        '它复用已合格PMF，不新增拟合/积分/风险门，不是继续联合训练FREE的结果或因果预测。'
        '条件目标分数更低与真实观察风险更低必须分别核查。','',
        '## 数值资格与账目','',
        'API数值资格：'+json.dumps(ev['numerical_qualification'],ensure_ascii=False)+'。','',
        '新评估账目：'+json.dumps(ev['ledger'],ensure_ascii=False)+'。','',
        '参考账目：'+json.dumps(ref['accounting'],ensure_ascii=False)+'。'
        f'主评估{ev["seconds"]:.3f}s，参考{ref["seconds"]:.3f}s；独立审计成本另列。','',
        '0新观测、0原训练更新、0NN；P4产生16个诊断候选选择，已单列。'
        '新曲线是理想数学准则的数值MC评价；canonical/API32对照仅为前缀数值资格，'
        '不提供原API32全域精度或Gaussian参考严格余项证书。','',
        '[报告](REPORT.md) · [设计](EXPERIMENT_DESIGN.md) · [理论](EVALUATION_THEORY.md) · [审查](REVIEW.md)','']
    with (DOC/'RESULTS.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(primary_effects=summary['primary_effects_passed'],primary_precision=summary['primary_precision_passed'],
          half=half,selected_risk_means=[float(old.mean()),float(new.mean())]),ensure_ascii=False))
if __name__=='__main__':run()
