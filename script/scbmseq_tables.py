"""Readable frozen outer-unit tables and prespecified descriptive influence checks."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/scbm_sequential_generalization';DOC=ROOT/'docs/08_research_研究工作/新数据与连续更新对照_20261004'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def f(x):return format(float(x),'.12g')
def main():
    assert not (BASE/'VERSION_LOCK.json').exists() and not (DOC/'RESULTS.md').exists()
    r=read(BASE/'RESULTS.json');tr=read(BASE/'TRAIN/RESULTS.json');ev=read(BASE/'EVAL/RESULTS.json');ref=read(BASE/'REFERENCE/RESULTS.json');old=read(BASE/'OLD_PANEL_DIAGNOSTIC.json')
    rows=r['training_primary']+r['evaluation_primary'];influence=[]
    for row in rows:
        v=np.array(row['values']);loo=(v.sum()-v)/15
        influence.append(dict(name=row['name'],negative_units=int((v<0).sum()),positive_units=int((v>0).sum()),zero_units=int((v==0).sum()),leave_one_data_out_mean_min=float(loo.min()),leave_one_data_out_mean_max=float(loo.max()),scope='Prespecified descriptive influence check; no deletion, new CI, selection or test.'))
    s=['# 新数据外层：全部主量与描述性读数','','本页仅重排保存数值，并计算预先规定的逐数据符号与leave-one-data-out点均值范围。0新RNG、训练、评分、积分或独立检验。风险以16个新数据单位覆盖数据/训练随机性；P评分另覆盖各数据独立评价随机性。每份数据先平均2个训练四元组，评价先在每份数据内部平均8组；不将32训练组或128评价组当独立外层单位。','','[机器统计](../../../artifacts/scbm_sequential_generalization/RESULTS.json) · [训练记录](../../../artifacts/scbm_sequential_generalization/TRAIN/RESULTS.json) · [参考积分](../../../artifacts/scbm_sequential_generalization/REFERENCE/RESULTS.json) · [独立评价](../../../artifacts/scbm_sequential_generalization/EVAL/RESULTS.json)','','## 1. 七项原主量','','|差|均值|SE|共同下界|共同上界|方向|SE过门|','|---|---:|---:|---:|---:|---|---|']
    for row in rows:s.append('|'+ '|'.join([row['name']]+[f(row[k]) for k in ['mean','SE','lower','upper']]+[row['direction'],str(row['precision_passed'])])+'|')
    s+=['','风险alpha=.025/q5，评分另alpha=.025/q2，均df15、双侧开发t近似。风险两程序差SE门.0005，三项相对I差.0002；两P评分SE门.0005。精度与方向分别登记；未分辨不代表无效、等效或非劣。数值参考误差不在区间内。','','|策略|平均观察joint KL|','|---|---:|']
    for arm,value in zip(['IN4','OUT4','SEQ4','FACT'],r['mean_KL_by_arm']):s.append(f'|{arm}|{f(value)}|')
    s.append('|不训练I（旧参考）|'+f(r['identity_reference_risk'])+'|')
    s+=['','FACT是同数据高质量程序参考，不是等评分成本竞争臂或精确MLE。truth/I参考复用；所有KL针对每次实际部署参数模型后平均，非跨quartet另平均参数的风险。','','## 2. 全部16个外层单位','','|数据d|风险OUT−IN|风险SEQ−OUT|风险OUT−I|风险SEQ−I|风险FACT−I|raw P OUT−IN|raw P SEQ−OUT|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in range(16):s.append('|'+str(d)+'|'+'|'.join(f(row['values'][d]) for row in rows)+'|')
    s+=['','## 3. 所有128个实际TAIL风险','','|数据|quartet|IN4|OUT4|SEQ4|FACT|','|---|---:|---:|---:|---:|---:|']
    for d in range(16):
        for rep in range(2):s.append(f'|{d}|{rep}|'+ '|'.join(f(v) for v in r['KL_by_data_repeat_arm'][d][rep])+'|')
    s+=['','## 4. 对数估计偏差：只由原家族派生','','|差|ΔJ点值|派生下界 L_F−U_R|派生上界 U_F−L_R|','|---|---:|---:|---:|']
    for row in r['gap_derived']:s.append('|'+row['name']+'|'+'|'.join(f(row[k]) for k in ['mean','lower','upper'])+'|')
    s+=['','逐数据ΔJ=ΔF−ΔR；共享训练输出使两者相关，不独立相加SE。派生界继承两原家族的联合开发近似，不是新主门。Jensen gap由随机似然分布决定，不是噪声方差或优化器因果份额。','','旧固定面板单独诊断：ΔJ='+f(old['log_estimation_Jensen_gap_difference'])+'，原P区间平移固定数值风险为['+', '.join(f(v) for v in old['conditional_shifted_interval'])+']；仅条件于旧面板，不替代新外层不确定性。','','## 5. 逐数据影响描述（不删数据）','','|原主量|负/正/零单位数|删去任一单数据后的点均值最小|最大|','|---|---|---:|---:|']
    for row in influence:s.append(f"|{row['name']}|{row['negative_units']}/{row['positive_units']}/{row['zero_units']}|{f(row['leave_one_data_out_mean_min'])}|{f(row['leave_one_data_out_mean_max'])}|")
    s+=['','这里只检查点均值是否由一个单位改变符号，不重跑区间、不把符号数当新检验，不把稳定符号当作所有数据同效。','','## 6. 每数据评分及参数描述','','|数据|策略|raw P|raw D|FACT P|FACT D|raw J_P|参数半平方距|参数交叉积|参数MSE|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in r['parameter_descriptive']:s.append('|'+str(row['data_id'])+'|'+row['arm']+'|'+'|'.join(f(row[k]) for k in ['raw_P','raw_D','FACT_P','FACT_D','raw_J_P','half_squared_distance','cross_product_error','parameter_MSE'])+'|')
    s+=['','每数据仅两训练重复，参数半平方距与交叉积之和为参数MSE；负值保留，非KL因果分解。D/FACT评分及参数两量没有新增主门。完整rho、每组评分和组内MC标准误在机器统计/评价记录；组内MC SE不能代替总外层SE。','','## 7. 计算与资格','','|调用/共享执行计时|IN4|OUT4|SEQ4|FACT|','|---|---:|---:|---:|---:|']
    for key in ['original_API_calls','optimizer_steps','repair_calls','native_forward_backward_seconds','optimizer_and_repair_seconds']:s.append('|'+key+'|'+'|'.join(f(tr['strategy_timing_totals'][a][key]) for a in ['IN4','OUT4','SEQ4','FACT'])+'|')
    s+=['','每策略调用计时为并行共享执行中的时间片之和，可能重叠，不含所有生成/hash/IO/资格开销；不提供独立端到端速度比。实测TRAIN墙钟'+f(tr['seconds'])+'秒，REFERENCE '+f(ref['seconds'])+'秒，EVAL '+f(ev['seconds'])+'秒。','','新数据65536观察、262144 Gaussian、17039360 Uniform。32prior四元组、224私有优化、128策略TAIL。训练4194304逐观察bank、419430400向量、109051904000 Gaussian；三raw各65536原评分、FACT另16384，Adam/修复合计163840。训练1280聚合控制和1664逐bank数学64调用按原资格检查。','','评价65536bank、6553600向量、1703936000 Gaussian；640原API资格，主评分数学64。参考384新PMF、504365056结点，另2旧truth/I复用；独审另16PMF、26873856结点。数据及prior输入全重放、保存更新与统计全核，底层训练限定160状态/208bank调用、评价81920前缀单元和640原API，不称全训练或全底层评分重跑。实际执行结果及失败历史见[审查](REVIEW.md)。','','开发确定性资格另账；若资格/独审与表中科学效应的状态不同，以各原始收据为准。没有按风险或区间追加步数、数据、prior组或积分阶数。']
    output=DOC/'RESULTS.md';output.write_text('\n'.join(s)+'\n',encoding='utf8')
    record=dict(source_sha256=sha(__file__),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [BASE/'RESULTS.json',BASE/'TRAIN/RESULTS.json',BASE/'REFERENCE/RESULTS.json',BASE/'EVAL/RESULTS.json',BASE/'OLD_PANEL_DIAGNOSTIC.json']},outputs={output.relative_to(ROOT).as_posix():sha(output)},influence_descriptive=influence,new_RNG=0,new_model_integrals=0,new_training=0,new_intervals_or_tests=0)
    with (BASE/'TABLES.json').open('x',encoding='utf8') as out:json.dump(record,out,ensure_ascii=False,indent=2);out.write('\n')
if __name__=='__main__':main()
