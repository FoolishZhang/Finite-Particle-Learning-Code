"""Write a human report from frozen results; no new scientific computation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/scbm_six_frozen_targets';DOC=ROOT/'docs/08_research_研究工作/六相关冻结输出与有限目标_20261003'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,text):(DOC/name).write_text(text,encoding='utf-8',newline='\n')
def main():
    r=read(OUT/'RESULTS.json');mc=read(OUT/'MC/RESULTS.json');ref=read(OUT/'REFERENCE/RESULTS.json');rows=r['primary_family'];q={x['name']:x for x in rows}
    table='|预定比较|均值|SE|共同开发区间|方向门|精度门|\n|---|---:|---:|---|---|---|\n'
    for x in rows:table+=f"|{x['name']}|{x['mean']:.10f}|{x['SE']:.10f}|[{x['lower']:.10f}, {x['upper']:.10f}]|{'过' if x['effect_passed'] else '未过'}|{'过' if x['precision_passed'] else '未过'}|\n"
    integral='|比较|积分似然差 ΔL|有限评分失真差 ΔF−ΔL|同一区间平移|\n|---|---:|---:|---|\n'
    for x in rows:
        lo,hi=x['MC_distortion_interval'];integral+=f"|{x['name']}|{x['integrated_loglikelihood_contrast']:.10f}|{x['MC_distortion_contrast']:.10f}|[{lo:.10f}, {hi:.10f}]|\n"
    ctable='|候选|参考joint KL|参考势V|raw P均值|FACT P均值|\n|---|---:|---:|---:|---:|\n'
    for x in r['candidates']:ctable+=f"|{x['name']}|{x['reference_joint_KL']:.10f}|{x['V']:.10f}|{x['mean_score_P'][0]:.10f}|{x['mean_score_P'][1]:.10f}|\n"
    write('RESULTS.md','# 全部预定读数\n\n主家族q11、alpha .05、df31，临界t='+f"{r['critical_t']:.10f}"+'。区间条件于旧数据/输出及数值P权重；开发近似，非新训练总体确认。SE门统一.0002。效应10/11、精度5/11、合并5/11；六项raw精度全未达，raw D FIXED−FACT方向亦未分辨。\n\n'+table+'\n## 同一随机对比的积分平移\n\n以下不是新独立家族，未纳入未知求积余项，也不是因果份额。FACT评分仍有active有限K偏差。\n\n'+integral+'\n## 固定35候选\n\n总评分含共同无关项；不同评分的绝对量不用于幅度归因。V未按w缩放，不等于有限F。\n\n'+ctable+'\n完整32组向量、128项逐模型描述与来源见[RESULTS.json](../../../artifacts/scbm_six_frozen_targets/RESULTS.json)。逐模型只给均值/SE，不增加未经校正的显著性结论。\n')
    report='''# 实际坏输出也得到有限粒子目标的奖励

**新证据把目标失真从预设的一维候选，推进到了实际六相关学习输出。** 对固定的16个旧FIXED TAIL，新的总体评分P在面板平均意义下偏好这些高风险输出，而同K因子分解评分与积分模型风险给出相反排序。方向得到预定开发区间支持，但所有raw比较的SE都未达到预定要求；本包不宣称主门全部通过。

本研究在[v7封存](../../07_papers_论文工作/ParticleJointSupervision/v7_working_20261003/README.md)之后单独冻结。v7 manifest为801306f7797b3cb4231cf81b5575b2b6c70db3a5e5f4b87d5cd939eb58e2621d；新科学冻结为08c384cb9336af21a910d9018cf3af817af503943fa8ce5a16c26efebf2e91fe。本报告全部新结果不进入v7。HANDOFF原文保持，外部sandbox未取得。

## 1. 实际输出的风险与目标方向相反

固定原数据、16个FIXED TAIL、同数据FACT TAIL、truth、旧high和一个FIXED均值矩阵，共35个C。用32组×1024bank、K100的全新prior分别评价原数据条件目标D与参考总体目标P。每个D比较先使用该流自己的标签频数，再平均16流；推断单位是32组新MC。

|总体P中的候选差|raw有限评分差及共同区间|同K因子分解评分差|积分风险差|
|---|---|---:|---:|
|FIXED−truth|−.0009588146 [−.0017040382, −.0002135911]|+.0060752698|+.0063134213|
|FIXED−FACT|−.0009078556 [−.0015682044, −.0002475069]|+.0051744032|+.0053793679|

分数越低越好。因子分解的两项区间分别为[.0060617534,.0060887862]与[.0051623052,.0051865012]。对这批固定矩阵，有限raw评分方向与精确统计任务的风险方向确实相反；这不是仅由选择了一条已知真实相关方向得到的例子。

这是面板平均的条件结论，不是对全部16个矩阵逐一建立共同显著性。它说明实际训练产生的候选中存在与目标失真相符的几何方向；它不证明训练已优化到总体最小点，也不能分摊原训练风险有多少由偏差、梯度噪声、数据或修复造成。

原数据条件目标D也偏好FIXED相对truth，差−.0013378606，区间[−.0020862333,−.0005894878]。D下FIXED−FACT为−.0006058628，但区间[−.0012689187,+.0000571930]跨零，不能认定其排序。D−P仅替换固定候选的标签权重，不能代表数据选择对训练的全部因果作用。

**实际输出还保留P总体准则下的面板平均改进余量。** 从原q11共同区间事后作矩形相减，得到FIXED面板平均raw P评分减预定high评分为+.0018781847，保守区间[.0003403590,.0034160105]。因此high比这些实际输出的平均评分更低；其参考KL .001724195也低于FIXED风险均值。目标偏差与这种P改进余量同时存在，不能说FIXED已经优化好了错误总体目标，也不直接给原D训练目标的优化误差。这只是原共同事件的代数推论，见[单列记录](../../../artifacts/scbm_six_frozen_targets/POST_HOC_DERIVED.json)，不新增家族、不追认门，也不逐一证明16模型次优。high使用已知真实方向，是可行优化参照，不能冒充可部署新方法。

![冻结输出的评分与风险](../../../artifacts/scbm_six_frozen_targets/figures/frozen_targets.png)

## 2. 跨流离散本身对应一个反向评分几何

新增meanFIXED矩阵的参考joint KL为.0001429581；原16个FIXED的风险均值为.0063134213，两者差+.0061704631。这是参数均值矩阵的风险，不是模型混合风险；它汇集16个训练流，只能作为几何探针。

在同一固定点集上，平均_i F(C_i)−F(mean_i C_i) 的raw P差为−.0008615250，共同区间[−.0015054665,−.0002175835]；同K因子分解差为+.0059443712，区间[.0059307229,.0059580195]。于是低风险的均值矩阵在raw评分中反而不如分散输出的平均评分。

这比单看参数MSE分解更直接：当前有限目标在这个预定面板上对离散给出相反于积分风险的排序。但一个有限点集的Jensen差不能证明有限目标在全相关域上凹，也不能证明训练噪声制造了这些输出。

新[理论](THEORY.md)补充定量联系：极限势满足J_V≤−β·dispersion/8≤−dispersion/64，其中β=.25 exp[−(1+.6²)/2]。真实相关的线性项在Jensen差中消去；固定数值中心造成的线性误差也消去，pair求积误差仍存在。本轮数值J_V=−.0088740334，满足更紧解析上界−.0069701512。V未按w缩放，不用它的幅度解释当前F差的比例。

## 3. 精度和适用范围仍限制结论

11项主比较的方向门10/11、精度门5/11、合并5/11。全部6项raw比较SE为.0002105至.0002591，超过预定.0002；不追加bank或更改门限。唯一方向未分辨项为raw D的FIXED−FACT。同K因子分解5项方向与精度都通过。high−truth的raw P差−.0028369994提供原目标失真的控制，也未达精度门。旧训练、旧方向包的失败均保持。

主准则采用数学float64。原SCBM API32的192个预定实际输入对照通过，另有192个冻结前确定性对照与192个审计重调用；这不构成所有主评分的native32证书。canonical无关标签只因新prior期望中的公平符号对称成立，不复现原训练实现。

区间条件于旧数据、冻结输出及数值P权重，是开发t近似。三阶求积及独立规则的一致性只是数值资格，没有严格全积分余项证书。TAIL部署平均不是优化器LAST。本轮没有新观察、训练或神经网络实验，没有根据新结果选点。

## 4. 计算与独立核验

生产32768个新bank、3276800个prior向量、851968000个高斯标量。新增均值矩阵3PMF/3940352四维节点，V为35C×3规则/17848320二维节点；34个旧PMF原样复用。独审另5个V/2764800二维节点和1个PMF/1679616四维节点。生产PMF最大变化2.24036e−12、V最大变化2.63175e−12，均过预定资格。

全部新输入重放，底层独立评分限定每组首4bank，共143360标量case；全部保存block均值、11项统计、逐模型描述和参考算术另核。不称全部主评分重跑。最终审计状态与误差见[独审](REVIEW_INDEPENDENT.md)及[审查入口](REVIEW.md)。

## 5. 下一步转向随机优化机制

这轮已回答较窄但关键的问题：实际高风险输出确实可获得更低的有限总体评分。下一阶段优先固定训练数据和批次顺序，只重复训练prior，量化给定数据下的参数离散与风险；另将可行性修复作为有针对性的控制，而不是从修复次数直接归因。暂不加倍旧迭代、追旧门或把均值探针包装成方法。详见[NEXT](NEXT.md)。
'''
    write('REPORT.md',report)
    write('MAIN_RESULT_ZH.md','''# 有限评分奖励实际坏输出，也奖励这个面板的参数离散

原16个六相关FIXED TAIL的参考mean jointKL为.0063134213。同数据FACT为.0009340534，参数均值矩阵为.0001429581。新的独立MC总体评分却偏好FIXED：相对truth差−.0009588146，相对FACT差−.0009078556，q11共同开发区间均在零下。相同K的因子分解评分两差为正。

对预定均值矩阵探针，平均raw评分减均值矩阵评分为−.0008615250，区间[−.0015054665,−.0002175835]，而积分风险Jensen差为+.0061704631。同一固定点集出现反向排序。新解析界J_V≤−β·dispersion/8连接此现象与极限势的强凹性；它不证明当前有限F全局凹。

效应10/11、精度5/11；全部raw SE未达.0002，D下FIXED−FACT还未分辨。区间条件于旧面板/数值P权重，不是新训练总体确认；均值模型不是等成本算法；不作偏差/优化噪声/数据/修复的因果分摊。v7已先行封存，新结果只保存在本研究包。[完整报告](REPORT.md)。

原共同区间的事后代数推论还给出FIXED面板−high评分区间[.0003403590,.0034160105]：已预定high比实际输出平均分数更低。因此当前证据支持目标偏差与P总体准则下的面板平均改进余量并存，不支持已到有限总体最小点，也不直接给原D训练目标的优化误差。
''')
    write('MAIN_RESULT_EN.md','''# Actual learned matrices exhibit a finite-score/risk reversal

We freeze 16 old six-correlation FIXED TAIL matrices, their same-data FACT counterparts, truth, a prespecified high-correlation control, and the mean FIXED matrix. New independent priors give a population-weighted raw-score contrast of −0.000958815 for FIXED minus truth and −0.000907856 for FIXED minus FACT. Both simultaneous development intervals are negative, whereas their integrated-risk contrasts and the same-K factorized-score contrasts are positive.

The raw-score Jensen contrast over the fixed panel is −0.000861525, with interval [−0.001505467,−0.000217583]. The integrated-risk counterpart is +0.006170463: mean individual KL is 0.006313421, while KL at the mean matrix is 0.000142958. A new strong-concavity bound J_V≤−β dispersion/8 explains the limiting-potential direction; it does not prove global concavity of the present finite objective.

The 11-comparison family meets 10 effect targets and 5 precision targets. All six raw-score SEs exceed the prespecified 0.0002 threshold; the data-conditioned FIXED-versus-FACT contrast also remains unresolved. Intervals condition on the old panel and qualified numerical P weights. The mean matrix is a multi-stream geometric probe, not a cost-matched learning method. No causal decomposition of optimization noise, data, target bias, or repair is identified. This study began after v7 was sealed and is excluded from that manuscript. See [the full report](REPORT.md).

A post-hoc rectangle implication of the original simultaneous intervals places the FIXED-panel-minus-high raw P contrast in [0.000340359,0.003416011]. Thus the preselected feasible high candidate still scores better than the average actual output. Target distortion and average objective-improvement room coexist; the outputs are not certified finite-population minimizers. This implication is not a new independently tested family or a per-matrix guarantee.
''')
    print('Human result files written from frozen results; independent review pending.')
if __name__=='__main__':main()
