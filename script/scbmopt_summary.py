"""Summarize frozen training outputs and qualified endpoint risks; no new science."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/scbm_scale_optimization'
DOC = ROOT / 'docs/08_research_研究工作/可学习无关尺度与优化路径_20261003'


def read(p):
    return json.loads(Path(p).read_text('utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def run():
    assert not (OUT / 'VERSION_LOCK.json').exists()
    train = read(OUT / 'TRAIN/RESULTS.json')
    ref = read(OUT / 'REFERENCE/RESULTS.json')
    assert ref['training_results_sha256'] == sha(OUT / 'TRAIN/RESULTS.json')
    scales = []
    for r in train['records']:
        for kind in ['TAIL', 'LAST']:
            tau = np.array(r['outputs']['FREE'][kind]['scales'])
            scales.append(dict(stream=r['stream'], kind=kind,
                mean=float(tau.mean()), RMS=float(np.sqrt(np.mean(tau**2))),
                minimum=float(tau.min()), maximum=float(tau.max()),
                q25=float(np.quantile(tau, .25)), median=float(np.median(tau)),
                q75=float(np.quantile(tau, .75)), sum_squares_over_8=float(np.sum(tau**2)/8),
                at_or_below_nominal_floor_count=int(np.sum(tau <= 1e-6))))
    diagnostics = {}
    for method in ['FIXED', 'FREE', 'FACT']:
        rr = [r['diagnostics'][method] for r in train['records']]
        diagnostics[method] = dict(
            sigmoid_zero_count=sum(x['sigmoid_zero_count'] for x in rr),
            sigmoid_one_count=sum(x['sigmoid_one_count'] for x in rr),
            maximum_absolute_logit=max(x['maximum_absolute_logit'] for x in rr),
            boundary_postupdate_count=sum(x['boundary_postupdate_count'] for x in rr),
            projected_count=sum(x['projected_count'] for x in rr),
            all_finite=all(x['all_finite'] for x in rr),
            all_feasible=all(x['all_feasible'] for x in rr))
    summary = dict(source_sha256=sha(__file__),
        inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in
                [OUT/'TRAIN/RESULTS.json', OUT/'REFERENCE/RESULTS.json']},
        primary_effects_passed=ref['primary_effects_passed'],
        primary_precision_passed=ref['primary_precision_passed'], primary_total=3,
        reference_qualification=ref['qualification_passed'],
        numerical_qualification=train['numerical_qualification'],
        scale_summaries=scales, diagnostics=diagnostics,
        training_ledger=train['ledger'], reference_ledger=ref['accounting'],
        training_seconds=train['seconds'], reference_seconds=ref['seconds'],
        development_t_intervals_not_exact_coverage=True,
        TAIL_and_LAST_not_independent=True,
        parameter_average_risk_not_average_risk=True,
        new_summary_RNG=0, new_summary_integrals=0, new_summary_fits=0,
        scope='Observed risks concern a fixed low-dimensional optimization program, not a full SCBM training result; training scores are not fresh population assessments.')
    write(OUT/'SUMMARY.json', summary)
    lines = ['# 受控尺度优化的完整读数', '',
        '16个独立数据/prior流；每流4096固定完整观测、512次批64更新、K100。'
        'TAIL为最后64个postupdate物理参数的平均，LAST为单个末点；两者不是独立实验。'
        '风险为保存输出t的完整观察joint KL，单位nats，未按维归一。', '',
        '## 预定主家族', '',
        f'三项共同Bonferroni双侧t开发近似，df15，family α=.05，临界值{ref["critical_t"]:.12g}。'
        '它不是精确有限样本覆盖。非劣margin .0002不是等价或零风险要求。', '',
        '|TAIL比较|平均差|SE|区间下端|区间上端|效应门|SE门|',
        '|---|---:|---:|---:|---:|---|---|']
    for x in ref['primary_family']:
        lines.append(f'|{x["name"]}|{x["mean"]:.10f}|{x["SE"]:.10f}|{x["lower"]:.10f}|{x["upper"]:.10f}|{x["effect_passed"]}|{x["precision_passed"]}|')
    lines += ['', '前两项要求区间下端>0，SE≤.0002；第三项要求区间上端<.0002，SE≤.00005。'
              '任何失败保留，不从LAST或其他时点追认。', '',
              '## 输出与风险', '',
              '|输出|程序|平均t|t最小|t最大|平均joint KL|KL的流间SE|',
              '|---|---|---:|---:|---:|---:|---:|']
    for x in ref['descriptive']:
        lines.append(f'|{x["kind"]}|{x["method"]}|{x["mean_t"]:.8f}|{x["min_t"]:.8f}|{x["max_t"]:.8f}|{x["mean_joint_KL"]:.10f}|{x["SE_joint_KL"]:.10f}|')
    lines += ['', 'FACT在固定K100下仍有active粒子与有限优化误差，不是精确MLE。'
              'FREE多256个参数、FACT少计算无关概率；同更新步数不是同算术成本或时间。', '',
              '## 全部逐流TAIL读数', '',
              '|流|FIXED t|FREE t|FACT t|FIXED KL|FREE KL|FACT KL|FREE尺度RMS|Στ²/8|',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    lookup = {(x['stream'], x['kind'], x['method']): x for x in ref['endpoints']}
    for s in range(16):
        row = [lookup[(s, 'TAIL', m)] for m in ['FIXED','FREE','FACT']]
        sc = next(x for x in scales if x['stream']==s and x['kind']=='TAIL')
        lines.append('|'+str(s)+'|'+'|'.join(f'{x["t"]:.8f}' for x in row)+'|'+
                     '|'.join(f'{x["reference_joint_KL"]:.10f}' for x in row)+
                     f'|{sc["RMS"]:.8f}|{sc["sum_squares_over_8"]:.8f}|')
    lines += ['', '尺度平方和界是固定输出在新prior下的无关评分偏差上界，不是观察KL、active梯度误差或收敛证书。'
              'FACT尺度为NA；代码buffer或输出零数组是占位，不是学得参数。'
              '全部256个TAIL/LAST尺度、梯度数值对照及反向结果保存在TRAIN JSON/NPZ。', '',
              '## 数值资格与成本', '',
              '预定原API32/stable-logit64对照：`'+json.dumps(train['numerical_qualification'],ensure_ascii=False)+'`。', '',
              '生产概率与参数诊断：`'+json.dumps(diagnostics,ensure_ascii=False)+'`。', '',
              f'风险参考资格：{ref["qualification_passed"]}；实际积分账目：`'+json.dumps(ref['accounting'])+'`。'
              '对96输出的实际float.hex去重；所有唯一t直接GH24/32/40，不作插值、不裁剪极小KL。'
              '变阶通过是数值资格，不是严格积分余项证书。', '',
              '训练科学账目：`'+json.dumps(train['ledger'])+'`。'
              f'训练墙钟{train["seconds"]:.3f}s（两worker），参考墙钟{ref["seconds"]:.3f}s。'
              '重复呈现、方法间配对和TAIL/LAST不重复计新数据。独立审计计算另登记。', '',
              '参数学习实际使用官方概念评分API，未使用encoder、task head、precision项或自然数据。'
              '没有fresh API assessment；训练loss/梯度不当成独立总体评分结果。', '',
              '[报告](REPORT.md) · [理论](SCALE_DYNAMICS_THEORY.md) · [预定设计](EXPERIMENT_DESIGN.md)', '']
    with (DOC/'RESULTS.md').open('x',encoding='utf8',newline='\n') as f:
        f.write('\n'.join(lines))
    print(json.dumps(dict(primary_effects=summary['primary_effects_passed'],
         primary_precision=summary['primary_precision_passed'],
         reference_passed=summary['reference_qualification'],
         descriptive=ref['descriptive']),ensure_ascii=False))


if __name__ == '__main__':
    run()
