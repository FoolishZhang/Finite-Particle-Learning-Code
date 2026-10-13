"""Present completed conditional experiment results, preserving both families."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_bank_averaging';DOC=ROOT/'docs/08_research_研究工作/同目标下的粒子梯度平均_20261003'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,s):(DOC/name).write_text(s,encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'VERSION_LOCK.json').exists(),'The research package is sealed'
    r=read(OUT/'RESULTS.json');train=read(OUT/'TRAIN/RESULTS.json');ref=read(OUT/'REFERENCE/RESULTS.json');ev=read(OUT/'EVAL/RESULTS.json')
    cols=['sample_trace_covariance_unbiased','debiased_conditional_mean_parameter_error','sample_parameter_MSE','mean_matrix_KL']
    summary={arm:{key:float(np.mean([x[key] for x in r['parameter_descriptive'] if x['arm']==arm])) for key in cols} for arm in ['L1','L4']}
    repairs={arm:sum(x['diagnostics'][arm]['feasibility_repair_count'] for x in train['records']) for arm in ['L1','L4']}
    risk0,risk4=r['mean_KL_by_arm'];v0,v4=r['mean_disjoint_conditional_dispersion_by_arm']
    post=dict(label='Post-hoc analytic prediction, not a new experiment or test',source_result_sha256=sha(OUT/'RESULTS.json'),conditional_parameter_summary=summary,formula='E||mean of 4 iid same-data L1 outputs - truth||^2 = B_L1 + trCov_L1/4',unbiased_parameter_MSE_estimate=summary['L1']['debiased_conditional_mean_parameter_error']+summary['L1']['sample_trace_covariance_unbiased']/4,observed_single_L4_parameter_MSE=summary['L4']['sample_parameter_MSE'],new_RNG=0,new_model_integrals=0,new_training=0,new_confidence_intervals=0,new_scientific_gates=0,scope='Does not compute or guarantee KL of a four-output mean. The actual mean matrices evaluated in this study use sixteen outputs; they are not four-cost comparators.')
    with (OUT/'POST_HOC_PARAMETER_PREDICTION.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(post,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    table='|家族/比较 L4−L1|均值|SE|家族共同开发区间|方向|精度|\n|---|---:|---:|---|---|---|\n'
    for family,key in [('训练','training_primary'),('条件评价','evaluation_primary')]:
        for x in r[key]:table+=f"|{family}: {x['name']}|{x['mean']:.10f}|{x['SE']:.10f}|[{x['lower']:.10f}, {x['upper']:.10f}]|{x['direction']}|{'过' if x['precision_passed'] else '未过'}|\n"
    pt='|data|arm|mean KL|R−1样本trace covariance|去偏条件均值参数误差|参数MSE|16参数均值矩阵KL|\n|---|---|---:|---:|---:|---:|---:|\n'
    for x in r['parameter_descriptive']:pt+=f"|{x['data_id']}|{x['arm']}|{x['mean_KL']:.10f}|{x[cols[0]]:.10f}|{x[cols[1]]:.10f}|{x[cols[2]]:.10f}|{x[cols[3]]:.10f}|\n"
    write('RESULTS.md','# 全部主要读数\n\n两个预定家族分别alpha=.025、q2；各97.5%共同开发t区间，总alpha预算.05。训练风险使用16个prior列（df15），离散度使用8个不重叠列对（df7）；评价使用32新MC组（df31），条件于已训练面板。后者不是对训练prior重复的期望目标变化的区间。\n\n'+table+'\n## 预定参数与均值矩阵描述\n\n本表没有新增置信区间。主离散推断使用不重叠列对，以下全R样本trace covariance是同一对象的另一个无偏估计，有限样本值不必相同。去偏均值误差允许为负，本轮8项均非负。参数MSE恒等式不是KL因果分解。\n\n'+pt+'\n## 修复和初始梯度描述\n\n每臂均32768次更新，修复次数L1 '+str(repairs['L1'])+'，L4 '+str(repairs['L4'])+'；次数变化不是修复的因果效应。初始I/各数据首批次的经验梯度trace variance比依次为 '+', '.join(f"{x['ratio']:.8f}" for x in r['initial_gradient_descriptive'])+'；这些相关比值仅描述，不作为1/4的新确认门。\n\n完整16/8/32个主单位、模型向量及逐数据评分见[RESULTS.json](../../../artifacts/scbm_bank_averaging/RESULTS.json)。原始128TAIL/128LAST、数值控制与RNG账目见[TRAIN](../../../artifacts/scbm_bank_averaging/TRAIN/RESULTS.json)；LAST未作风险积分。\n')
    report=f'''# 同一期望目标下，外层平均降低风险与离散，但中心点估计更偏

**本轮得到一项受控的正向结果：不改变每个固定参数/批次的期望K100目标，仅把独立bank梯度在更新前作四份平均，就降低了给定旧数据的学习风险和参数离散。** 两项训练主区间及精度要求都通过。与此同时，预定描述显示四份数据的条件均值参数误差和均值矩阵KL点估计都升高；后两者没有新增置信区间，不能与主结果等强度表述。

v7及全部旧科学保持封存，新结果仅进入本研究包。HANDOFF没有新更新，仍评议cf4f8f51；本轮基准9d48bbda8aa0196341b89b1cc2306957d0e5299f。ROOT冻结865b331dd472620589fff31f990a0fbfdea72f1a07a6005f8eacaab64a97a074先于所有训练；候选按规则机械构建后，以DERIVED冻结ae805837b612ca70f152a81e6816eca21bc095a7881400b49413a7f72650d6a4绑定参考和评分。

## 1. 控制的是哪一层

固定原data_id 0..3的4份完整数据，每份N4096。每份16次全新prior重复，L1与L4配对，从I学习六相关。均值/单位方差/块位置/无关尺度1固定；K100、批64、512步、Adam、lr=.02和原可行性修复全部相同。L4在同一状态分别计算四个原K100损失及梯度，外平均之后才做一次Adam、一次修复；L1共享第一份输入。

在实算术下，同参数/同批次的期望损失和原始梯度均值保持，原始梯度协方差除以4。它没有把K改成400。但Adam的二阶矩、预条件、修复和轨迹会响应分布变化，因此不是保持平均更新不变的“纯方差”干预。批次按固定日程循环，每步对应当前批次目标，不能当完整D梯度的无偏抽样。

L4每次更新使用四倍独立采样和评分，未作为等成本方法。128个六参数优化均完成，0新观察、0NN训练。TAIL仍取最后64个物理参数；LAST仅保存，不改主选择。

## 2. 风险与条件离散均下降

|主量|L1|L4|L4−L1共同开发区间|
|---|---:|---:|---|
|mean joint KL|{risk0:.10f}|{risk4:.10f}|[−.0022798127, −.0001787031]|
|条件trace covariance的独立列对估计|{v0:.10f}|{v4:.10f}|[−.2641225019, −.0806681716]|

风险差−.0012292579，SE .0003703535，达到预定.0005；离散差−.1723953368，SE .0275019651，达到.05。两个方向均为降低。相对下降约{100*(risk0-risk4)/risk0:.2f}%与{100*(v0-v4)/v0:.2f}%只作描述性比例。

风险推断先在每个prior列中平均4份固定数据，得到16个iid条件单位；离散推断用8个不重叠prior列对。不是把64任务当新iid数据集，也不是把TAIL64个状态或全部重叠成对距离当独立样本。区间为alpha=.025、q2的开发t近似，结论条件于这4份旧数据。

![固定目标下的训练与评价](../../../artifacts/scbm_bank_averaging/figures/outer_averaging.png)

## 3. 这批输出也取得了原目标的进展

用全新32组×1024bank对冻结的128个TAIL及8个均值探针评分。raw F_D的L4−L1为−.0014092789，区间[−.0017861090,−.0010324488]；raw F_P为−.0011348532，区间[−.0015121670,−.0007575394]。两项SE约.000142，均达到预定.0002。

这第二家族也用alpha=.025、q2，但只条件于此次实际训练输出与旧标签频数。它支持这个面板的原有限目标平均评分降低，不是对训练prior总体期望评分变化的区间。共享评价池不能被重解释成16个独立训练列的评分重复。两个家族的范围和原失败均保持。

目前可确认的窄结论是：该外平均程序提高了这个受控设置中的条件风险与离散表现，本轮实际输出还得到更低的原目标评分。目标本身未被校正，也没有证明优化完成或风险趋零。

## 4. 改善并非简单围绕同一个中心收缩

使用全部16个prior重复的预定参数恒等分解，平均R−1样本trace covariance从{summary['L1'][cols[0]]:.10f}降到{summary['L4'][cols[0]]:.10f}，而去偏条件均值参数误差点估计从{summary['L1'][cols[1]]:.10f}升到{summary['L4'][cols[1]]:.10f}。四份数据各自都呈现这一方向。它们相加恰为样本参数MSE；该恒等式没有把KL变化因果分解。

每数据16个输出的平均矩阵，其参考KL也四份都升高，平均{summary['L1'][cols[3]]:.10f}→{summary['L4'][cols[3]]:.10f}。这些是预定描述，没有新置信区间；不宣称条件均值偏移已经获得独立统计确认。均值模型不是混合预测器，也不是单次训练等成本基线。

修复次数则从{repairs['L1']}增加到{repairs['L4']}，分母每臂同为32768更新。原始梯度平均不等于减少触边，也不能从修复次数归因风险变化。初始同点的经验梯度方差比.2197至.3086与理论1/4相容，但仅为描述；路径分开后不可继续套用1/4。

这解释了为什么下一步要研究计算放在哪里，而不是把本轮称为目标纠偏或把旧FREE收益分摊成某个噪声百分比。

## 5. 解析补充：固定经验权重仍有相同极限曲率

训练运行期间新增[条件势记录](CONDITIONAL_POTENTIAL_NOTE.md)，未改变预设或产生新抽样。对固定归一标签权重q，正确条件矩分量是A_{{q,ij}}=E_q[(S_i−b_i)(S_j−b_j)]，一般不只是经验协方差。原标签统一近并列界给出条件势V_q；q只改变其线性项，Hessian以及同一q下固定面板的Jensen差保持。

这不证明当前n256/K100有限目标全局凹，更不证明每份固定数据都有相对精确经验MLE的额外MC风险。记录给出point-mass控制：K≥2时，精确经验目标与有限raw条件期望可具有同一个唯一极点最小值。未归一权重还需保留质量因子。详见[独立数学审读](NOTE_REVIEW.md)。

## 6. 预算、核验和后续

训练使用8388608独立观察bank、838860800prior向量、218103808000Gaussian，163840次原API调用；耗时{train['seconds']:.1f}秒。512个聚合及1280个逐bank数值资格全过。评价另32768bank/3276800向量/851968000Gaussian，320个原API32对照通过；主评价仍是数学64，非整体native32证书。

136新C的408个PMF使用535887872四维节点，全部预定阶数资格通过；truth/high两PMF复用。独审另16个GH36对称根PMF/26873856节点。均为数值资格，不是严格余项证书。全训练和评价输入重放、所有保存平均/Adam/修复与统计独核；底层训练限32aggregate状态/80bank，评价限282624标量case，不称全部底层重跑。详见[独审](REVIEW_INDEPENDENT.md)。

开发控制另120个原API/20个math64调用，无科学RNG。准备脚本一次括号错误在解析阶段退出，未写输入；修复后确定性准备成功。首张图的标题/区间文字重叠已修复并保留历史，不计科学失败。原科学包/v7保持。

下一优先是新独立prior四元组下的计算匹配对照：同样四份K100输入，分配给一条L4路径或四条L1路径，后者仅在输出端平均参数。主指标仍需joint KL与独立目标评分，四条均值不是本轮已算的16条均值。[事后参数预测](../../../artifacts/scbm_bank_averaging/POST_HOC_PARAMETER_PREDICTION.json)只提供Bhat+S/4的代数线索，不是新实验、KL保证或已验证方法优势。固定预条件对照随后用于进一步拆解Adam作用；新预算尚未冻结或运行。见[NEXT](NEXT.md)。
'''
    write('REPORT.md',report)
    write('MAIN_RESULT_ZH.md','''# 同一期望目标下，风险与条件离散得到改善

固定4份旧数据，各16次新prior重复；比较单份K100梯度与四份独立K100梯度外平均，两者都只作一次原Adam/修复。128条优化完成，期望当前批次目标不变，L4采样/评分成本为四倍。

mean joint KL .007042179→.005812921，差−.001229258，共同开发区间[−.002279813,−.000178703]。同数据内参数离散独立列对估计.383091588→.210696251，差区间[−.264122502,−.080668172]。两项预定精度均过。

另32组独立评价显示本次输出的raw D/P评分都下降，两个区间及精度也过；它们条件于已训练面板，不是训练prior总体评分差的区间。

预定描述显示，四份数据的去偏条件均值参数误差及16输出均值矩阵KL点估计反而都上升。此项没有新置信区间，不与风险/离散主结论等强度表述。不能称为目标纠偏、纯方差份额或FREE收益归因。下一步优先新prior下计算匹配的四条L1输出平均，再拆解Adam预条件作用。v7不改。[报告](REPORT.md)。
''')
    write('MAIN_RESULT_EN.md','''# Same-target outer averaging improves conditional risk and dispersion

On four fixed old datasets, sixteen fresh prior-repeat columns compare one K100-bank gradient with the average of four independent K100-bank gradients before one unchanged Adam/feasibility update. The expected fixed-parameter, current-batch criterion is preserved. L4 uses four times the sampling/scoring per update.

Mean integrated KL decreases from 0.007042179 to 0.005812921; the difference is −0.001229258 with simultaneous development interval [−0.002279813,−0.000178703]. Conditional parameter dispersion, estimated with eight disjoint repeat-column pairs, decreases from 0.383091588 to 0.210696251, with difference interval [−0.264122502,−0.080668172]. Both prespecified precision targets are met.

A separate independent MC family places the realized-panel raw D and P score differences below zero and meets its precision targets. These intervals condition on the trained matrices and are not training-prior population score intervals.

Prespecified descriptive estimates move differently: debiased conditional mean-parameter error and KL at each sixteen-output mean matrix increase on all four datasets. No new interval confirms that center shift. The intervention is not target correction, a pure variance-only dynamical change, or a causal allocation of the old FREE gain. Next priority is a new-prior, computation-matched comparison with four independent L1 trajectories averaged at output; fixed preconditioning is a subsequent mechanism control. v7 remains immutable. See [the report](REPORT.md).
''')
    print(json.dumps(dict(parameter_summary=summary,repairs=repairs,posthoc=post['unbiased_parameter_MSE_estimate']),ensure_ascii=False))
if __name__=='__main__':main()
