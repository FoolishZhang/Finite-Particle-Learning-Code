"""Write the completed finite-calibration research report and navigation payloads."""
from pathlib import Path
import base64,hashlib,json,platform,subprocess,sys
import numpy,scipy

ROOT=Path(__file__).resolve().parents[1]
HOME='docs/08_research_研究工作/有限校准与真实样本训练_20261002'
DOC=ROOT/HOME
OUT=ROOT/'artifacts/finite_calibration'
BASE='c3d215e9866411103bba409efcce1c4603c52604'
HANDOFF='f0f31f4f8ec095f4a57d1b19e0bf41bd2b4e5e868b270a6dba25a509a8a0677d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def put(p,s):p.write_text(s.strip()+'\n',encoding='utf-8',newline='\n')
def save(p,s):put(p,json.dumps(s,ensure_ascii=False,indent=2))


def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    assert sha(ROOT/'HANDOFF.md')==sha(OUT/'HANDOFF_INPUT.md')==HANDOFF
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==BASE
    stats=read(OUT/'TRAIN/STATISTICS.json');quad=read(OUT/'POSTHOC_QUADRATIC/RESULTS.json')
    model=read(OUT/'CAL/MODEL.json');train=read(OUT/'TRAIN/RESULTS.json')
    put(DOC/'README.md','''# 有限校准与真实样本训练

[研究报告](REPORT.md) · [全部读数](RESULTS.md) · [审查入口](REVIEW.md) · [中文核心](MAIN_RESULT_ZH.md) · [English](MAIN_RESULT_EN.md) · [下一步](NEXT.md)

本包把上一轮有限K校准从解析参照落实为不读取真实theta的模型校准，并进入逐行保留无关观测的完整共享bank训练。原四项开发判读2通过、2未通过；完整有限校准恢复了真实相关的大部分强度，但未确认两个正教师相对MLE非劣。事后二次近似几乎复现完整曲线，不能据此要求复杂修正。

理论与审计：[计算恒等式和误差分层](COMPUTATION_THEORY.md)、[数学审读](COMPUTATION_AUDIT.md)、[CAL审计](CAL_AUDIT.md)、[训练审计](TRAIN_AUDIT.md)、[统计审计](STATISTICS_AUDIT.md)、[条件代理审计](CONDITIONAL_AUDIT.md)、[端点诊断审计](ENDPOINT_AUDIT.md)。TRAIN_REPORT保留预运行实现记录，最终科学结论以本包REPORT/RESULTS为准。

完整论文仍为原样封存v4.1；本包没有新稿或新PDF。只读`.venv/Scripts/python script/fincal_delivery.py --check`；--workspace另核当前导航/HANDOFF及36个本地数组。历史checker只用默认模式。默认交付检查核字节、算术收据与文档，不重新运行科学，也不认证严格Gaussian积分或全部浮点误差。
''')
    put(DOC/'REPORT.md',r'''# 有限校准恢复了真实相关 但仍有随机评分误差

本轮的实质进展是：**同一条不读取真实相关值的有限校准，能够在真实完整观测的模拟训练中恢复被渐近修正压低的相关强度；一条二次近似已能接近完整校准曲线的效果。** 因而当前问题更接近“如何校准有限维幅度、如何控制剩余随机评分”，不是“是否必须用复杂正则”。

这仍是已知两位相关块、固定$a=1,R=.75$的Gaussian-logistic受控实验，不是自然数据结果或一般算法优越性。完整校准依赖模型积分与额外prior模拟；便宜的同K边际化和精确active似然是保留的强基线。

![n254/K256，训练B128。左为教师无关校正导数，验证仅4独立流的估计；中为N4096时16流均值参考风险；右为原四项家族中三项有限校准对MLE非劣比较，t区间是条件于冻结校准的开发近似。星号二次规则是事后重用分析，不属于主家族。](../../../artifacts/finite_calibration/figures/calibration_training.png)

## 1 从正确的有限目标估计校准

使用独立校准prior，按Price表达估计$U',V'$，构造$\widehat g'=-\widehat U'-r\widehat V'$。对称化后在101点拟合导数样条，部署函数是其反导并减$g(0)$；整个校准池在验证与训练前固定，所有theta和N共用同一函数。它不使用本轮训练观测、真实theta或教师风险调系数。

独立验证中，$n254/K256$的theta .30/.60对应有限校准代理最小点约.302287/.604654，参考KL约$7.71\times10^{-8}$/$3.30\times10^{-7}$；相同验证下渐近知情项约选.156183/.313744。theta .30/.60处，有限校准的导数残差约$-3.03\times10^{-5}$/$-6.26\times10^{-5}$，而渐近项约$+.00368/+.00734$。这与此前“渐近幅度在有限维偏强”的方向一致。

这些是4条独立验证流估出的样条势，不是精确总体最优点或全域置信证书。训练202个导数SE门、g与g′的51/101对照各2门均通过；参考GH64/96也通过，但不能用这些资格替代严格校准误差界。[COMPUTATION_THEORY](COMPUTATION_THEORY.md)给出保守条件界及误差分层，没有假定当前有限预算已满足一个有用的全域严格误差保证。

## 2 真实完整观测训练与连续求解控制

训练固定$n=254,K=256,B=128$，16基础流，每流4096病例，N1024为其前缀。theta0/.30/.60共享基础随机数但各自有正确的观测分布，故不能把teacher或N当独立重复。训练保留每条真实Ynoise；为加速，仅把noise与第一位似然归一化成rho无关权重，再计算第二位的两个概率。这与原完整似然的rho对比逐实现相同，不是canonical噪声替代。

前五个方法在801点返回计算网格最小值。根据完整目标曲率与所有样条断点，480个输出的真算术**离散化项**均小于$3\times10^{-5}$，最大约$1.84985\times10^{-5}$。该界不要求目标凸，但实际返回点对真算术目标的完整regret还含浮点点值误差$2\zeta$；这里没有认证全部$\zeta$或总体误差，不能把它写成严格连续全局最优证书。独立直接评分差达到$10^{-13}$量级，作为单独的数值资格。

N4096时的16流平均参考joint KL为：

|真实theta|raw|同K marg|渐近二次|渐近知情|有限校准|精确似然参考|
|---:|---:|---:|---:|---:|---:|---:|
|0|.00841757|.00021225|.00007758|.00007758|.00028371|.00020657|
|.30|.00390159|.00021765|.00047989|.00047989|.00034640|.00021013|
|.60|.00034757|.00016253|.00139659|.00140032|.00024246|.00016243|

theta .60时，渐近二次平均相关估计为.30199，有限校准为.58078，精确似然参考为.59048；theta .30时相应为.13418/.26367/.27972。有限校准明显保留更多真实相关，同时仍高于marg/MLE的参考风险。null下更强的渐近收缩反而有更低风险，不能宣称完整有限校准对每个教师都占优。

### 原四项开发判读保持两通过两未通过

theta .30、N4096的raw−finite平均KL差为.00355519，Bonferroni配对t区间约[.00151311,.00559727]，通过预定>.001改善门。null的finite−MLE上界约.000171741，通过<.0002非劣门。

两个正教师的非劣门未通过：theta .30上界.000321423；theta .60上界.000201322，虽仅略超阈值也不追认。均值差本身分别为.000136270和.000080034；未确认非劣不等于已证明劣于该margin。原门、失败及全部逐流结果保持，没有追加数据或更改阈值。

这些区间只有16基础流，属于预定的小样本t开发近似；全部训练共用一个冻结校准池，因此区间不含重估校准的变异。raw null的零观测方差也不视作严格总体等式。完整低N、恒零基线、全部区间与数据观察信号在[RESULTS](RESULTS.md)。

## 3 简单二次近似已经接近完整有限校准

读完CAL导数形状后、读取完整训练风险汇总前，我们另行冻结了一个探索规则：$c=\widehat g''(0)/2$，使用raw+$c\rho^2$。它只从同一个模型校准取得系数，不读teacher或训练效果，但属于原计划之外的事后重用分析。原672输出和四主门不改，新增96个原曲线选点，不是96次独立新训练。

有限维系数为.008344015，此前渐近系数为.014532496。N4096时，有限二次在theta0/.30/.60的平均参考KL约.00028495/.00035562/.00024451；相对完整有限曲线的均值差绝对值均小于$9.22\times10^{-6}$。这不证明二者总体等效，但当前对照不支持必须引入复杂的非二次形状。实际取得这个局部系数仍用了完整CAL池，不能另声称已实现更低校准成本。

![两个明确事后诊断。左把实际finite输出、四验证流条件bank期望代理和相同数据MLE分开；右比较有限二次与完整有限曲线的均值风险。图线只连接三个固定teacher，不能作因果百分比分解或新确认。](../../../artifacts/finite_calibration/figures/posthoc_diagnostics.png)

## 4 数据误差与随机评分误差不能混淆

用独立验证的U′/V′和实际$t_N$重建的条件bank期望代理，在N4096下与同数据MLE的平均参数绝对差约.00066/.00216/.00427；实际finite输出与该代理的差约.04266/.04923/.04381。两层差距的量级不同，但代理仅来自4验证流，不是精确无限bank目标，也不支持把风险按因果百分比分摊。

另一项固定输入诊断可用解析符号：因为U偶、V奇且$V'<0$，给定数据的精确条件期望端点差为$2t_NV(R)$，符号恰为$-\operatorname{sign}(t_N)$。本轮theta .30、N4096有一流raw选择负端点，但其pair和为−24，即$t_N=-.005859375$；条件期望本来就偏好负端点。因此不能把该次错误直接归为bank排序反转。全部正教师高N单元的端点反向计数为零，低N/null则存在反向个例。此处没有新频率检验或总体发生率保证。

上述对照支持继续区分“校准平均目标”和“控制实际随机评分”，而不是用一种单一原因解释每条失败轨迹。

## 5 工作量与复核范围

CAL训练/验证共40960bank、10485760条prior向量、2684354560个Gaussian标量；训练另有2048bank、524288条prior向量、134217728个Gaussian。观测数据为65536基础病例、196608个共享输入的teacher向量，active Gaussian131072、总Uniform16777216；低N及teacher复用不重复计独立样本。无新神经网络。

原672输出为480个有离散界的网格输出、96数值MLE、96恒零；另有96事后二次重用选点。CAL验证32 pooled和128逐流样条选择、条件代理192 assignments/158独特优化均另列，不算独立训练。新增Gaussian参考积分1676个二维标量积分，旧d/Var/J复用另记。

全部CAL与TRAIN RNG已独立重放。CAL仅每流首64、共1280bank独立重算Price导数，其余核保存算术；TRAIN直接完整raw核限定64观测/8bank/21rho，再用独立缓存核完整N1024、首8bank、21rho。marg全部保存曲线、672输出及480向上界独立重建。统计、事后二次、条件代理、端点诊断分别审计，未冒称全raw评分重放或严格总积分证书。

一次开发SyntaxError和首次科学入口误用--run均在新科学输入产生前退出，修复/改用正确run入口后执行；记录保留，不算科学精度失败。所有旧科学失败、v4.1稿件/PDF与HANDOFF原字节保持。本包全部科学与审计结束，没有自动追加队列。
''')
    names={'raw':'raw','marg':'marg','quad':'渐近二次','asymcorr':'渐近知情','finitecal':'有限校准','exact':'MLE参考','zero':'恒零'}
    lines=['# 全部训练读数','', '所有区间条件于同一个冻结CAL函数；描述t区间不是同时覆盖或分布无关保证。主四门单独使用Bonferroni临界值。','',
        '|N|theta|方法|平均参考KL|SE|平均rho|零观测SE|','|---:|---:|---|---:|---:|---:|---|']
    for r in stats['descriptive']:
        lines.append(f"|{r['N']}|{r['theta']:.2f}|{names[r['method']]}|{r['mean']:.10g}|{r['SE']:.7g}|{r['mean_rho']:.8f}|{r['zero_observed_SE']}|")
    lines+=['','## 原四项开发门','', '|比较|均值|下界|上界|门限|通过|','|---|---:|---:|---:|---:|---|']
    for r in stats['primary']:lines.append(f"|{r['id']}|{r['mean']:.10g}|{r['lower']:.10g}|{r['upper']:.10g}|{r['threshold']}|{r['pass_']}|")
    lines+=['','## 观察层信号','', '|theta|N|r(theta)|sqrt(N)r(theta)|信号/标准误|','|---:|---:|---:|---:|---:|']
    for r in stats['signals']:lines.append(f"|{r['theta']}|{r['N']}|{r['r_theta']:.9f}|{r['sqrtN_r']:.7f}|{r['signal_over_pair_standard_error']:.7f}|")
    lines+=['','## 事后二次近似','', '|theta|N|均值参考KL|均值rho|相对完整finite均值KL差|相同网格点/16|','|---:|---:|---:|---:|---:|---:|']
    for r in quad['summary']:lines.append(f"|{r['theta']}|{r['N']}|{r['mean_reference_KL']:.10g}|{r['mean_rho']:.8f}|{r['mean_minus_full_finite']:.10g}|{r['exact_same_grid_point_count']}|")
    lines+=['','逐流672结果在TRAIN/RESULTS.json；原四门与全部SE在STATISTICS.json；96额外选择在POSTHOC_QUADRATIC。参考风险接近0的浮点残差不赋独立科学意义。']
    put(DOC/'RESULTS.md','\n'.join(lines))
    put(DOC/'MAIN_RESULT_ZH.md',r'''# 有限维幅度校准进入真实样本训练

在已知两位相关块、a1/R.75的Gaussian-logistic模型中，我们从独立prior池估计一条所有teacher共用的有限n/K校准函数，并在n254/K256/B128下运行真正保留每条无关观测的共享bank训练。N4096、theta.6时，渐近二次的平均相关估计约.302，有限校准约.581，精确似然参考约.590；相应均值joint KL为.001397/.000242/.000162。theta.3时raw到finite平均KL从.003902降至.000346，通过原定改善门。

原四项配对t开发判读2通过2未通过：两个正teacher对MLE的非劣上界分别.000321423/.000201322，均未低于.0002；不追认失败，也不把未确认非劣说成已证明有害。训练区间条件于一个冻结CAL池。

独立的事后重用发现，有限校准局部二阶系数.008344（渐近系数.014532）的二次规则，N4096三个teacher均值KL与完整曲线相差不到9.22e−6。本轮不支持复杂形状有必要，也未实际减少校准成本。

480个网格输出都附完整目标的向外离散化界，最大1.85e−5；这不包含全部浮点评分、校准或bank误差。实际经验输出与条件bank期望代理仍有差距，校准均值不等于消除随机评分误差；数据不足也必须相对同数据MLE单独辨认。
''')
    put(DOC/'MAIN_RESULT_EN.md',r'''# Finite-dimension calibration restores correlation in full-observation training

In the known two-coordinate Gaussian-logistic model with a=1 and R=.75, one teacher-independent calibration function was estimated from a separate prior pool and shared across all data-generating correlations and training repetitions. Training retained every nuisance observation and reused complete prior banks, with n=254,K=256,B=128. At N=4096 and theta=.6, the mean fitted correlation was approximately .302 under the asymptotic quadratic correction, .581 under finite calibration, and .590 under the active-likelihood reference. Their mean reference joint KL values were .001397,.000242,.000162. At theta=.3, finite calibration reduced mean raw KL from .003902 to .000346.

Two of four prespecified paired-t development gates passed. Noninferiority to the likelihood reference was not established for the two positive teachers; the upper bounds .000321423 and .000201322 did not fall below .0002. Intervals are conditional on the single frozen calibration pool and are not distribution-free guarantees.

An explicitly exploratory reuse of stored curves used the local quadratic coefficient ghat''(0)/2=.008344, rather than the asymptotic .014532. At N=4096, its mean risks differed from the full finite-calibration curve by less than 9.22e-6 across all three teachers. This does not establish population equivalence or a cheaper implemented calibration procedure, but gives no evidence that a complicated penalty shape is needed here.

All 480 grid-based outputs have outward real-arithmetic discretization bounds below 1.85e-5 for the full implemented objectives. These bounds exclude unverified point-evaluation error, calibration uncertainty, and bank error. The results separate correcting the mean objective from controlling its realized stochastic fluctuations; no general algorithmic advantage over marginalization or the cheap known-block likelihood is claimed.
''')
    put(DOC/'REVIEW.md','''# 有限校准与真实样本训练审查入口

先读[REPORT](REPORT.md)、[RESULTS](RESULTS.md)及两图。当前完整稿仍v4.1，本包不是新论文版本。

最值得审查的事项：

1. [COMPUTATION_THEORY](COMPUTATION_THEORY.md)与[数学审读](COMPUTATION_AUDIT.md)：缓存权重是完整raw目标的逐实现代数重写，没有删除真实噪声行；有限g导数是固定teacher偏导，不是对角全导数。有限校准的正确方向不等于全局强凸。
2. [CAL_REPORT](CAL_REPORT.md)和[CAL_AUDIT](CAL_AUDIT.md)：一个teacher-free函数在所有验证与训练前固定，独立prior及样条资格不等于严格全域MC误差。CAL的额外模型信息/成本不得隐藏。
3. [TRAIN_AUDIT](TRAIN_AUDIT.md)：真实观测生成、跨teacher/N复用、全部RNG与限定raw重评分。完整marg重建不等于全raw重建。
4. [STATISTICS_AUDIT](STATISTICS_AUDIT.md)：原四门保留2/4，两项正相关非劣未过；小样本t仅开发近似、且条件于共享CAL。全部672原输出独立核验。
5. 三个探索分析与原家族分开：有限二次重用、[条件代理](CONDITIONAL_AUDIT.md)、[端点符号](ENDPOINT_AUDIT.md)。它们没有新独立样本，不能代替确认性结果或分摊因果百分比。负端点个例必须先核实际t_N。

本轮HANDOFF没有更新，仍评议旧22833fac；没有新外部审批或审查确认。本轮内部实现和独立复核不视为外部专家盲审。旧失败、旧包及v4.1字节保持；当前无自动追加队列。后续思路见[NEXT](NEXT.md)。
''')
    put(DOC/'NEXT.md','''# 下一步先解释剩余随机误差 而非增加修正复杂度

本轮已经把正确的有限n/K校准带入真实样本训练；一个局部二次近似也接近完整函数。因此下一步不应以更复杂的函数击败简单规则为目标。

优先在同一观测集上另冻小规模B变化，使用完全独立新增bank或预定嵌套前缀，比较实际经验曲线与条件bank期望代理。保持校准池不变，以区分训练prior精度与校准精度；同时保留同K marg和数值active MLE。新预算不得用于追认本轮两个未过非劣门。

如果B增加足以明显缩小差距，则论文可以更清楚地区分目标偏差与随机评分误差；如果不能，需复核校准的全域精度和有限样本相互作用，而不能只凭四流代理宣布原因已确定。可先建立固定n/K的梯度误差界，避免单凭优化器停止条件解释结果。

另一条有价值但独立的路线是只用局部校准估计一个二次系数，检验是否真的能降低offline成本。本轮事后二次仍借用了完整校准池，尚未证明更便宜；新的低成本版本需另冻且不使用真实theta调参。

已有结果足以支撑下一版作者写作：保持独立扩展破坏学习不变性的主线，补有限校准的正面能力与剩余随机性，保留简单基线成功及非劣失败。无需把这套受控模型包装为通用新算法。未知结构、自由协方差和自然任务继续按科学价值排序，而非同时展开。

本包已收口，以上实验尚未冻结或启动。v4.1完整稿仍封存，后续整合另建版本。
''')
    put(DOC/'POSTHOC_SCOPE.md','''# 原判读与探索分析分开

原PLAN和TRAIN FREEZE定义四项开发门，结果保持两通过两未通过。追加分析均未改变训练输入、CAL函数、原672输出、门限或预算。

- 有限二次规则：读CAL形状后、主代理读取完整训练风险汇总前，冻结c=ghat″(0)/2。随后复用保存raw曲线产生96额外选择；没有重新训练CAL、没有按teacher调参，也没有新成功门。它不是新的独立验证。
- 条件目标代理：看到原风险汇总后，将每份实际数据的t_N代入独立验证U′/V′，重建192个raw/finite分配（158不同状态优化）。它是四验证流数值代理，不是实际经验目标或严格无限bank证书。
- 端点符号：看到结果后，用U偶/V奇/V′<0的解析身份比较96份固定数据的期望与实际端点排序。按t_N=0期望平局单列，不做新显著性或总体发生率检验。

三者都支持解释和后续设计，不能合并为原家族之外的“又一次确认”。所有来源、代码、协议及独立重建均保存。
''')
    # Pre-run implementation report is historical, not a stale claim about the package.
    p=DOC/'TRAIN_REPORT.md';old=p.read_text('utf-8')
    if not old.startswith('> 阶段说明：'):
        put(p,'> 阶段说明：下文是科学运行前的实现/资源记录，按原文保留。科学运行现已完成；最终672输出、四主门和审计结论见REPORT、RESULTS及TRAIN_AUDIT。本段不改变冻结源码或benchmark。\n\n'+old)
    save(OUT/'FIGURE_QA.json',dict(passed=True,actual_viewed_pngs={p:d for p,d in read(OUT/'FIGURES.json')['outputs'].items() if p.endswith('.png')},
        repaired_issues=['Moved risk-panel legend outside plotted data; clarified independent derivative estimate and simulated-observation label'],
        remaining_required_fixes=[],new_scientific_computation=False))
    save(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,platform=platform.platform(),
        note='RNG/input/hash replay verified in this environment; no cross-platform PDF or floating byte invariance claim'))
    ledger=dict(CALtrain=read(OUT/'CAL/TRAIN.json')['ledger'],CALvalidation=read(OUT/'CAL/VALIDATION_STREAMS.json')['ledger'],
        CALreference=read(OUT/'CAL/REFERENCE.json')['accounting'],CALvalidation_numeric=read(OUT/'CAL/VALIDATION.json')['accounting'],
        TRAINfreeze_data=read(OUT/'TRAIN/FREEZE.json')['data_ledger'],TRAINfreeze_output=read(OUT/'TRAIN/FREEZE.json')['output_ledger'],
        CALaudit=read(OUT/'CAL/AUDIT.json')['ledger'],TRAINaudit=read(OUT/'TRAIN/AUDIT.json')['ledger'])
    paths=['artifacts/regularization_learning/MANIFEST.json','artifacts/regularization_learning/VERSION_LOCK.json',
        'docs/08_research_研究工作/统一正则与真实相关学习_20261002/FINITE_K_CALIBRATION.md',
        'artifacts/paper_v4_1_20261002/MANIFEST.json','artifacts/paper_v4_1_20261002/VERSION_LOCK.json',
        'output/pdf/particle_joint_supervision_v4_1_zh.pdf','script/reglearn_delivery.py']
    save(OUT/'DELIVERY_INPUT.json',dict(base_commit=BASE,handoff_sha256=HANDOFF,handoff_is_new=False,
        handoff_reviewed_old_commit='22833facae52ab88ef7444f94e47fc9e92a9d060',
        old_files={p:sha(ROOT/p) for p in paths},science_ledger=ledger,new_manuscript=False,
        posthoc_ledger=dict(extra_quadratic_selections=96,conditional_assignments=192,conditional_unique_optimizations=158,
            endpoint_rows=96,new_observations_or_RNG=0,new_Gaussian_integrals=0,original_primary_family_modified=False)))
    save(OUT/'PRIOR_CHECKS.json',dict(checks=[dict(script='script/reglearn_delivery.py',args=['--check'],returncode=0,
        script_sha256=sha(ROOT/'script/reglearn_delivery.py'),result=dict(passed=True,version='regularization-learning-20261002',workspace=False))]))
    save(OUT/'AUTHOR_REVIEW.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(DOC.glob('*.md'))},
        full_theory_read=True,independent_cal_train_statistics_audits=True,original_failures_preserved=True,
        posthoc_results_separate=True,remaining_required_fixes=[],external_blind_review=False,novelty_certified=False))
    frame=f'''# 当前论文论证框架

**主问题：统计任务不变时，有限prior评分如何扭曲学习，简单修正能恢复什么，剩余随机误差是什么。** 当前完整稿仍[v4.1](docs/07_papers_论文工作/ParticleJointSupervision/v4_1_working_20261002/README.md)；[本轮科学报告]({HOME}/REPORT.md)未写入新稿。

|环节|可守结论|边界|
|---|---|---|
|原失真与连续学习|旧包已给连续近优学习分离、统一二次全教师极限KL≤1/71136、有限K条件校准参照|已知相关块/固定噪声；不等于有限B逐实现MLE|
|有限校准实际实现|独立prior校准的同一g适用于全部teacher/N；validation近真实相关|4流样条代理；202+2+2资格不等于全域严格误差|
|真实完整观测训练|N4096/theta.6，渐近二次mean rho .302→finite .581（MLE .590）；theta.3 raw KL .003902→.000346|16基础流/共享校准，原开发四门2pass2fail；两个正teacher非劣未确认|
|简单规则仍强|事后finite Taylor二次c=.008344，highN mean KL与完整曲线差<9.22e−6|96原曲线重用，不是独立确认或更低成本实现|
|误差需要分层|480离散界≤1.85e−5；条件代理近MLE，实际finite仍偏离代理|ζ/bank/校准误差未被离散界覆盖；不能作因果百分比|

真实Ynoise始终保留；缓存权重只作逐实现等价加速。θ.3高N负端点个例的t_N为负，不可误归bank排序反转。一般校准/Bregman有直接先例，没有新通用算法或运行时优势声明。

[下一步]({HOME}/NEXT.md)优先独立B精度与低成本单系数校准，旧非劣失败不追认。全部科学/审计收口，无自动追加队列。
'''
    put(OUT/'FRAMEWORK_SNAPSHOT.md',frame)
    head=f'''# 最新研究状态：有限校准进入真实完整观测训练

[报告]({HOME}/REPORT.md) · [全部读数]({HOME}/RESULTS.md) · [审查]({HOME}/REVIEW.md) · [中文核心]({HOME}/MAIN_RESULT_ZH.md) · [English]({HOME}/MAIN_RESULT_EN.md)。产物artifacts/finite_calibration，完整稿v4.1/PDF未改。

独立prior估有限g′并部署样条反导，202导数SE/2+2分辨率与参考资格通过；4流独立验证近真实相关但非严格总体证书。逐行完整观测训练n254/K256/B128/N1024,4096，16基础流；N4096/theta.6有限校准mean rho .581（渐近二次.302，MLE.590）。theta.3 mean jointKL raw.003902→finite.000346，原改善门过；四主门2pass2fail，两个正teacher对MLE非劣未确认（上界.000321423/.000201322> .0002）。t为条件于同一个CAL池的开发近似。

480输出有完整目标向外离散化项界，最大1.84985e−5；ζ/MC/校准误差不在该界内，不称严格连续globopt。独立缓存恒等式保留真实Ynoise与共享bank。事后有限二次c=.008344接近完整曲线，96额外选择不入主家族；条件代理与端点诊断均为事后复用、非因果百分比。θ.3高N负端点个例t_N<0，不误判为bank排序反转。

CAL40960bank、训练2048bank；prior向量共11010048，prior Gaussian2818572288；65536基础病例/196608共享输入teacher向量，active Gaussian131072、Uniform16777216。原672输出=480网格近优+96MLE+96恒零，另96二次重用。完整RNG重放；CAL仅1280bank独立Price，TRAIN raw限定前缀/首8bank核，marg全保存曲线核；统计和探索分析独立审计。无新NN。

基准`{BASE}`；HANDOFF未更新，仍`{HANDOFF}`评议旧22833fac，原字节保留，不虚构新反馈。全部原门/失败与旧科学/v4.1保持。只读`.venv/Scripts/python script/fincal_delivery.py --check`；--workspace另核36本地NPZ与当前导航/HANDOFF，旧checker不用历史--workspace。科学全部结束，无自动追加队列；下一步另冻B精度/局部单系数校准，不追认本轮未过门。

---

'''
    payload={}
    for name in ['README.md','REVIEW_BRIEF.md','AGENTS.md','RESEARCH_ROADMAP.md','docs/00_project_项目管理/START_HERE_当前工作入口.md']:
        old=(ROOT/name).read_bytes();assert not old.startswith('# 最新研究状态：有限校准'.encode())
        payload[name]=(head if '/' not in name else head.replace('](','](../../')).encode()+old
    payload['ARGUMENT_FRAMEWORK.md']=frame.encode()
    n='docs/00_project_项目管理/CURRENT_WORK_STATE.json';s=read(ROOT/n)
    s.setdefault('previous_research_states',{})['regularization_learning_20261002']=s['current_research']
    s['updated_at']='2026-10-02'
    s['current_research']=dict(id='finite_calibration_20261002',status='complete_calibration_real_observation_training_and_audits',
        base_commit=BASE,home=HOME+'/README.md',report=HOME+'/REPORT.md',results=HOME+'/RESULTS.md',review=HOME+'/REVIEW.md',next=HOME+'/NEXT.md',
        core_statement_ZH=HOME+'/MAIN_RESULT_ZH.md',core_statement_EN=HOME+'/MAIN_RESULT_EN.md',artifact='artifacts/finite_calibration',
        handoff_raw_sha256=HANDOFF,handoff_new_feedback=False,handoff_target='22833facae52ab88ef7444f94e47fc9e92a9d060',
        teacher_free_calibration_shared=True,conditional_on_single_calibration=True,independent_training_base_streams=16,
        new_base_observation_cases=65536,new_teacher_observation_vectors=196608,teacher_vectors_share_inputs=True,
        new_prior_banks=43008,new_prior_vectors=11010048,new_prior_Gaussian_scalars=2818572288,new_data_Gaussian_scalars=131072,
        new_uniform_draws=16777216,primary_method_outputs=672,posthoc_quadratic_selections=96,mesh_checks=480,mesh_passed=480,
        primary_effect_gates=4,primary_effect_passed=2,positive_teacher_noninferiority_not_confirmed=[.3,.6],
        new_neural_models=0,full_raw_score_replay=False,full_RNG_replayed=True,active_jobs=[],automatic_extension_queue=[],
        manuscript_changed=False,current_paper_unchanged='v4.1-working',new_PDF=False,submission_completed=False,novelty_certified=False)
    s['authority']['current_research_selection']=HOME+'/REPORT.md'
    s['execution_gate'].update(status='current_package_complete_no_scientific_queue',baseline_commit=BASE,current_phase='ICML_strengthening_finite_calibration_training_complete')
    s['review_workflow'].update(detailed_entry=HOME+'/REVIEW.md',last_consumed_feedback_sha256=HANDOFF,feedback_use_mode='unchanged prior review preserved; current work authorized by user continuation')
    s['continuing_user_goal']['current_stage']='finite_calibration_20261002_completed';s['active_tasks']=[];s['pending_tasks']=[]
    payload[n]=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode()
    payload['.gitattributes']=(ROOT/'.gitattributes').read_bytes()+('\n# Finite calibration and actual full-observation training evidence.\n'
        'script/fincal_*.py -text whitespace=cr-at-eol\n'+HOME+'/** -text whitespace=cr-at-eol\n'
        'artifacts/finite_calibration/** -text whitespace=cr-at-eol\n'
        'artifacts/finite_calibration/HANDOFF_INPUT.md conflict-marker-size=128 whitespace=-blank-at-eol,-blank-at-eof,cr-at-eol\n'
        'artifacts/finite_calibration/figures/*.svg whitespace=-blank-at-eol,-blank-at-eof,cr-at-eol\n').encode()
    payload['.gitignore']=(ROOT/'.gitignore').read_bytes()
    print(json.dumps({n:base64.b64encode(v).decode() for n,v in payload.items()}))


if __name__=='__main__':main()
