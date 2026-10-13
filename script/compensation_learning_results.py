"""Deterministic presentation of frozen RUN evidence; no new statistical family."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/08_research_研究工作/补偿学习与连续目标_20261001'
OUT=ROOT/'artifacts/compensation_learning';RUN=OUT/'RUN'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):return f'{x:.7g}'
def main():
    assert read(RUN/'STATE.json')['status']=='complete' and (RUN/'RESULTS_FREEZE.json').exists()
    r=read(RUN/'RESULTS.json');rows=r['risks']
    selected=list(itertools.chain.from_iterable(read(p) for p in RUN.glob('d*/SELECTIONS.json')))
    methods=['MOM','quasiMLE','raw','col'];summary=[]
    for d,N in itertools.product([16,64],[256,4096]):
        for method in methods:
            rr=[x for x in rows if x['d']==d and x['N']==N and x['method']==method]
            assert len(rr)==8
            risk=np.array([x['joint_KL'] for x in rr]);w=np.array([x['w'] for x in rr])
            summary.append(dict(d=d,N=N,method=method,mean_KL=float(risk.mean()),min_KL=float(risk.min()),max_KL=float(risk.max()),
                                mean_w=w.mean(0).tolist(),optimization_qualified=sum(x['optimization_qualified'] for x in rr)))
    fits=[x for x in selected if x['method']!='MOM']
    costs={m:dict(requests=sum(x.get('requests',0) for x in fits if x['method']==m),
                  actual=sum(x.get('actual',0) for x in fits if x['method']==m)) for m in methods[1:]}
    ledger=dict(summary=summary,objective_evaluations=costs,
        quasiMLE_full_reference_calls=costs['quasiMLE']['actual'],endpoint_three_variant_reference_calls=r['endpoint_reference_integrations'],
        frozen_algebra_reference_calls=5,prefreeze_development_reference_calls=10,
        col_private_quadrature_calls=costs['col']['actual'],
        solver_successes=sum(z['success'] for x in fits for z in x['starts']),solver_calls=864,
        selected_solver_successes=sum(x['starts'][x['selected_start']]['success'] for x in fits),
        optimization_diagnostic_pass=sum(x['optimization_qualified'] for x in fits),
        max_projected_gradient=max(x['pg'] for x in fits),max_fd_error=max(x['fd_error'] for x in fits),
        max_diagnostic_grid_improvement=max(x['grid_improvement'] for x in fits),
        raw_highN_upper_boundary_counts={str(d):sum(x['d']==d and x['N']==4096 and x['method']=='raw' and any(w==2.5 for w in x['w']) for x in fits) for d in [16,64]},
        risk_precision_pass=sum(x['precision'] for x in r['risk_contrasts']),score_precision_pass=sum(x['precision'] for x in r['score_contrasts']),
        risk_main_pass=sum(r['risk_main_gates']),score_main_pass=sum(r['score_main_gates']),
        run_wall_seconds=read(RUN/'STATE.json')['seconds'],new_scientific_samples=0,new_fits=0,new_integrals=0,
        source_results_sha256=sha(RUN/'RESULTS.json'),source_script_sha256=sha(__file__))
    with (OUT/'DERIVED_SUMMARY.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(ledger,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['# 补偿学习与连续目标：实验结果','',
      '全部128输出先选定并封存，再计算风险和独立评分。本表为已冻结RUN的确定性派生，没有新增拟合、积分、样本或检验家族。',
      '',f"原8项风险主门{ledger['risk_main_pass']}/8通过，另2项独立评分主门{ledger['score_main_pass']}/2通过。风险精度{ledger['risk_precision_pass']}/24、评分精度{ledger['score_precision_pass']}/16通过；所有失败保留。",'',
      '## 四程序的完整观察风险','',
      'KL为完整d位向量的总nats；每行8条独立训练流。范围是8个实测端点的min–max，不是置信区间。MOM的“合格”表示无似然优化要求；另三程序为原驻点/FD/网格诊断，非全局最优证书。','',
      '|d|N|方法|平均KL|端点KL范围|平均w_H,w_L|诊断合格|','|---|---|---|---|---|---|---|']
    for x in summary:
        lines.append(f"|{x['d']}|{x['N']}|{x['method']}|{num(x['mean_KL'])}|{num(x['min_KL'])}–{num(x['max_KL'])}|{num(x['mean_w'][0])}, {num(x['mean_w'][1])}|{x['optimization_qualified']}/8|")
    lines += ['','![Full joint risk](../../../artifacts/compensation_learning/figures/joint_risk.png)','',
      '## 原风险对比家族','',
      '各差值为前方法减后方法。24项共享alpha=.025的Bonferroni双侧t近似、df7，SE门为.002。主门仅N4096；raw−MOM/raw−col要求下端>.002，MOM/col−quasiMLE要求上端<.001（单侧非劣，不是等效）。','',
      '|d|N|差值|均值|区间|SE|精度|主门|','|---|---|---|---|---|---|---|---|']
    for x in r['risk_contrasts']:
        lines.append(f"|{x['d']}|{x['N']}|{x['pair']}|{num(x['mean'])}|[{num(x['low'])}, {num(x['high'])}]|{num(x['SE'])}|{x['precision']}|{x.get('main_gate','—')}|")
    lines += ['','## 独立评分家族','',
      '每流另用B512、K=d的独立prior银行，只评价已选模型。raw使用canonical恒等式及解析单粒子控制变量；训练未使用该压缩或控制变量。16项另有alpha=.025，SE门.003。N4096 raw−MOM要求上端<−.001。该家族与风险家族不合并。','',
      '|d|N|目标|差值|均值|区间|SE|精度|主门|','|---|---|---|---|---|---|---|---|---|']
    for x in r['score_contrasts']:
        lines.append(f"|{x['d']}|{x['N']}|{x['objective']}|{x['pair']}|{num(x['mean'])}|[{num(x['low'])}, {num(x['high'])}]|{num(x['SE'])}|{x['precision']}|{x.get('main_gate','—')}|")
    lines += ['','## 优化和计算账目','',
      f"96个连续程序的原诊断{ledger['optimization_diagnostic_pass']}/96通过；864次求解器调用中success={ledger['solver_successes']}，所选96次中success={ledger['selected_solver_successes']}。全部状态和终点均保存。投影梯度最大{num(ledger['max_projected_gradient'])}、五点差分误差最大{num(ledger['max_fd_error'])}、诊断网格改善最大{num(ledger['max_diagnostic_grid_improvement'])}。",'',
      f"65536个新训练向量、2621440个bit；16独立训练流。32个MOM输出+96连续程序=128输出；864个起点不是独立重复。训练/确认prior向量分别40960/327680，共368640，候选复用不重计。0确认标签、0新神经网络。",'',
      f"准似然优化实际调用{ledger['quasiMLE_full_reference_calls']}个完整候选参考；终点三变体另{ledger['endpoint_three_variant_reference_calls']}次。冻结ALGEBRA5次、冻前开发10次另记；col目标的私噪声求积以及REF/独立审计另列，不能将终点字段当全部积分。RUN全过程{num(ledger['run_wall_seconds'])}秒；方法行秒数不含共用raw缓存构建，不用于端到端速度比较。",'',
      '详情见[范围](SCOPE.md)、[开发与记账](DEVELOPMENT_LOG.md)、[原方案](PLAN_RUN.md)及[独立审计](RUN_AUDIT.md)。参考PMF和最终端点通过数值资格不等于严格积分余项或连续全域误差证书。','']
    (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n').write('\n'.join(lines))
    print(json.dumps(ledger,ensure_ascii=False))
if __name__=='__main__':main()
