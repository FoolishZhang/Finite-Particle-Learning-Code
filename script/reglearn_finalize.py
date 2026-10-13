"""Assemble the completed research package and return byte-safe navigation writes."""
from pathlib import Path
import base64,hashlib,json,platform,subprocess,sys
import numpy,scipy

ROOT=Path(__file__).resolve().parents[1]
HOME='docs/08_research_研究工作/统一正则与真实相关学习_20261002'
DOC=ROOT/HOME
OUT=ROOT/'artifacts/regularization_learning'
BASE='22833facae52ab88ef7444f94e47fc9e92a9d060'
HANDOFF='f0f31f4f8ec095f4a57d1b19e0bf41bd2b4e5e868b270a6dba25a509a8a0677d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def put(p,t):p.write_text(t.strip()+'\n',encoding='utf-8',newline='\n')
def save(p,d):put(p,json.dumps(d,ensure_ascii=False,indent=2))


def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    assert sha(ROOT/'HANDOFF.md')==sha(OUT/'HANDOFF_INPUT.md')==HANDOFF
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==BASE
    nr=read(OUT/'NUMERIC/RUN.json');summ=read(OUT/'GROWTH/SUMMARY.json')
    put(DOC/'README.md','''# 统一正则与真实相关学习

[研究报告](REPORT.md) · [全部数值读数](RESULTS.md) · [审查入口](REVIEW.md) · [中文核心](MAIN_RESULT_ZH.md) · [English statement](MAIN_RESULT_EN.md) · [后续事项](NEXT.md)

本包接受“简单二次规则可能非常有效”的提醒，完成全参数区间风险界、连续共享粒子近优学习、独立极限积分以及增长K的固定候选总体评分核查。极限中统一二次非常强；有限维直接套渐近系数仍可能过度收缩，两者必须分开。

理论：[正则与风险界](REGULARIZATION_THEORY.md)、[连续学习](CONTINUOUS_LEARNING.md)、[事后有限K校准桥](FINITE_K_CALIBRATION.md)。每份均有独立数学审读；计算与审计范围见REPORT。原PLAN及所有资格失败保留，不追加预算。

当前完整论文仍为原样封存的v4.1，本包不是新稿。没有新增观测样本、数据拟合或神经网络；有新Gaussian先验模拟与确定积分。只读`.venv/Scripts/python script/reglearn_delivery.py --check`；`--workspace`另核当前导航/HANDOFF及16本地NPZ。默认不重跑科学或声称数学首创。
''')
    put(DOC/'REPORT.md',r'''# 统一二次规则很强，但极限校准不等于有限维校准

本轮最重要的进展是把高级模型的提醒变成了一个正面定理：**在当前$a=1,R=3/4$模型中，不使用真实相关值的统一二次系数，对整个真实相关区间具有严格的小风险上界。** 同时，增长粒子数的有限维面板表明，直接套用渐近系数仍可能过度收缩；更复杂的渐近知情项没有自动解决这一点。

![统一二次极限风险与有限维固定候选选点。左图为a1的61点极限参考；右图固定theta=.6，横轴K=n+2。二次与知情渐近修正在该有限候选网格上选点重合，并非其目标函数相同。](../../../artifacts/regularization_learning/figures/limit_vs_finite.png)

## 1 统一二次规则的全教师保证

记$A=a^2/4,d=r_a'(0),H(\rho)=\int_0^\rho r_a(u)du$，使用固定规则$\lambda_n=w_nAd$。该规则只用已知发射模型、维数与粒子数，不读取教师参数。

Hermite展开的非负奇数系数给出非线性剩余$h(\rho)=r_a(\rho)-d\rho$。在$a=1$时，Price界与Gaussian积分不等式直接给极限二次目标的强凸模数至少$1/48$。再由$E(f_a'')^2\le a^4/27$，得到$h(3/4)\le1/384$，因而

\[
\sup_{\theta\in[-3/4,3/4]}\mathrm{KL}\bigl(Q_\theta\Vert Q_{\rho_0(\theta)}\bigr)
\le \frac{1}{71136}<1.406\times10^{-5}.
\]

这里$\rho_0$是二次修正的极限最小点，KL是完整观测向量的联合风险。上界不依赖数值求积，是[正则理论](REGULARIZATION_THEORY.md)的解析结果；计算规则本身仍需$d$的模型信息。它不是有限$n,N,B$训练风险保证。

独立数值重建复现了反馈的六点结果。$\theta=.30,.60,.70$时，二次极限最小点约为$.300828,.606813,.710972$，对应参考KL约$1.01\times10^{-8},7.08\times10^{-7},1.87\times10^{-6}$。61点扫描最大约$2.44\times10^{-6}$；该扫描不是连续最大值证书，连续范围由上面的解析上界承担。

零相关的局部和全局阈值分别约$.02134611$与$.02143925$；统一系数$Ad\approx.04269222$。局部稳定与全局修复不同，且在局部临界等号处零点仍为严格局部最大。单一二次系数不能对所有教师精确一致，**这一事实不足以判定它实际无效**。

## 2 学习保证已从固定网格延伸到连续区间

[连续学习理论](CONTINUOUS_LEARNING.md)保留真实完整观测与跨观测共享的完整bank。采用一维等距覆盖，只对独立bank集中，不将$NB$交叉项当独立样本。

沿$K=n+2,B=\lceil n^{1+\eta}\rceil$、固定正$a$、已知相关块及预定$N_n\to\infty$，iid观测与bank独立时：原始连续近优点在固定正教师下趋向$R$，零教师下趋近两端集合；知情渐近修正和边际化都恢复真实相关。统一二次规则收敛到其唯一极限最小点，因此在当前$a=1$范围保留上述小风险界。

raw、二次与知情修正要求真实经验目标的总优化误差$o(w_n)$；边际化只需$o(1)$。结论约束全部相应近优集合，不是任意局部求解器或SGD保证，也不声称有限$n$的经验极小点已经精确贴边。数据误差、bank误差、渐近余项、数值误差和求解误差分别计量。正则后的固定bank导数包络包括完整惩罚项；网格离散界为$\min(Lh/2,Dh^2/8)$，不替代其他误差。

## 3 增长K的有限维面板没有继承极限精度

新模拟使用16条独立先验流，每流1024bank，$n=14,62,254$且$K=n+2$，三个教师为$0,.30,.60$。每条流对四种相关块观测模式按已知概率平均；Gaussian符号对称性只用于这个**总体评分估计**。没有生成训练观测，也没有运行实际有限样本训练。21个预定候选的面板不认证连续全局最优。

在$n=254,K=256$时，16流均值评分的候选选点和参考风险为：

|真实theta|原始选点／KL|同K边际化选点／KL|统一二次选点／KL|知情渐近修正选点／KL|
|---:|---:|---:|---:|---:|
|0|.75／.008418|0／0|0／0|0／0|
|.30|.75／.003068|.30／0|.15／.000330|.15／.000330|
|.60|.75／.000348|.60／约0|.375／.000760|.375／.000760|

因此不能从极限的极小风险推断当前维度已经得到相同效果。对$\theta=.60$，渐近修正的面板参考风险甚至高于原始选点。两个修正在此网格上相同选点，也说明当前数据不支持“更复杂的非二次渐近项明显更好”。另一方面，$n=14,62$的零教师原始面板只选$.075$，并未在所有测试维度都确认端点现象。全部条件在[RESULTS](RESULTS.md)，不只展示$n=254$。

360个相邻对比的预定SE资格均通过；SE阈值和逐点t区间只是开发精度描述，不是全部曲线同时置信带、所选最小点保证或有限总体端点阈值证书。

![三个教师的有限维总体评分剖面。四种方法均画相对rho0的均值差除以w_n；线段仅连接固定候选，垂线是真实参数。误差统计保存在SUMMARY，不以图线作为严格连续目标。](../../../artifacts/regularization_learning/figures/growth_profiles.png)

## 4 事后解析桥区分了有限校准和渐近领先项

看到有限面板后，我们另行推导[有限K校准](FINITE_K_CALIBRATION.md)，不改原PLAN、门或预算。任意固定$n\ge0,K\ge2$，对bank积分后的目标可写为$U(\rho)+tV(\rho)$。Price恒等式严格给$V'<0$。教师无关参数项

\[
g_{n,K}(\rho)=-[U(\rho)-U(0)]-\int_0^\rho r_a(u)V'(u)du
\]

使校正目标导数成为$V'(t-r_a)$，所以其条件期望最小点对任意数据均值$t$都等于连续矩反演／精确MLE。该项在保留原数据线性项的全教师加性合同下仅差常数唯一，而$g_{n,K}/w_n\to2AH$。原渐近修正只是这个有限校准的领先项。

这是一条机制边界，**没有实际计算有限$g$或校准残差，也未量化解释本次面板偏移的因果比例**。它需要有限$n,K$的模型积分能力；本例精确似然早已可用，不能据此声称计算优势或新通用校准算法。一般均值一致评分与Bregman原理沿用Gneiting等已有理论。条件期望同极小点不是有限bank逐实现身份。

## 5 资格失败、审计及工作量

NUMERIC主$a=1$生产2222/2222、另一表达2972/2972通过；不同实现者对原六点及事后扫描点做Price积分核验，650/650通过。$a=.5$也通过；$a=2,4$的714项生产精度失败和1055项另一表达失败完整保留，未追加阶数，不能把这些副分支用于已认证的强发射结论。所有求积资格都不等于严格总积分余项证书。

GROWTH生成16384新bank、4194304条prior向量、1073741824个Gaussian标量。候选、教师与n/K前缀复用不重复计独立样本；0新观测、0数据拟合、0神经网络。完整RNG已重放；每流首64、合计1024bank以直接概率乘法独立重评分，raw最大差$5.69\times10^{-14}$，其余15360bank只核保存算术。全部36面板、576逐流选点、360门与固定对比统计另由独立脚本复核通过。确定积分、求根和审计重算单独列入机器账本，不作独立训练重复。

三份数学文件均经独立交叉审读，修复等距覆盖量词、控制字节和有限校准anchor表述。作图经历缺字和mathtext语法修复，属于制作问题，没有重跑科学。全部原失败与原HANDOFF字节保持。v4.1稿件/PDF原样封存，新结果尚未进入论文；本包科学和审计已收口，无自动追加队列。
''')
    # All nine population settings are displayed, including near-null and unfavorable cells.
    headers=['# 完整数值读数','', '本页均为已冻结结果的派生显示。有限面板选点不是有限数据训练或连续最优证书。','',
        '## 增长K固定候选面板','', '|theta|n / K|raw rho / KL|marg rho / KL|quad rho / KL|informed rho / KL|',
        '|---:|---:|---:|---:|---:|---:|']
    for theta in [0,.3,.6]:
        for n in [14,62,254]:
            cells=[]
            for method in ['raw','marg','quadratic','informed']:
                r=next(r for r in summ['selections'] if r['theta']==theta and r['n']==n and r['method']==method)
                cells.append(f"{r['rho']:.3f} / {r['reference_joint_KL']:.9g}")
            headers.append(f'|{theta:.2f}|{n} / {n+2}|'+ '|'.join(cells)+'|')
    headers+=['','## a1统一二次极限六点','', '|theta|r(theta)|rho_quad|参考joint KL|N1024信号/标准误|','|---:|---:|---:|---:|---:|']
    c=next(c for c in nr['cells'] if c['a']==1 and c['order']==64)
    for theta in [0,.15,.3,.45,.6,.7]:
        row=next(r for r in c['rows'] if abs(r['theta']-theta)<1e-12);r=row['rules']['quadratic'];t=row['r_teacher']
        headers.append(f"|{theta:.2f}|{t:.9f}|{r['rho']:.9f}|{r['joint_KL']:.10g}|{32*t/(1-t*t)**.5:.5f}|")
    headers+=['','61点、48/64/96全部变体与副a在NUMERIC/RUN.json；原714/1055失败在QUALIFICATION/AUDIT。精确风险零附近的浮点残差不单独赋科学意义。',
              '', '完整曲线、16流SE与逐点开发t区间在GROWTH/SUMMARY.json。有限差可与w_n比较，但当前没有高维极限余项的可计算有限阈值。']
    put(DOC/'RESULTS.md','\n'.join(headers))
    put(DOC/'MAIN_RESULT_ZH.md',r'''# 统一修正的能力与有限维边界

在固定非退化噪声、已知两位相关块的Gaussian-logistic模型中，统一二次系数$\gamma_0=(a^2/4)r_a'(0)$可以很有效。当前$a=1,R=3/4$时，其高维极限目标至少$1/48$强凸，且所有$\theta\in[-R,R]$的完整联合KL均不超过$1/71136$。这是解析上界，不是固定有限训练预算保证；二次规则的微小残余不一致不能被当作实际无效。

真实共享完整bank的连续近优学习也有分离：沿$K=n+2,B=\lceil n^{1+\eta}\rceil$，iid观测和bank独立、预定N发散，raw正教师趋R/null趋端点集合；同模型marg与知情渐近修正恢复教师，统一二次趋其小风险极限点。raw派生规则要求优化误差$o(w_n)$，marg只需$o(1)$；不赋予任意SGD保证。

有限增长K的固定候选总体评分给出不同层次的提醒：在n254/K256、theta.6，raw选.75，同K边际化选.6，两种渐近修正都选.375。它们在此面板过度收缩；不是新有限样本训练或连续全局最优证明，也不否定其极限性质。

任意有限n≥0/K≥2的条件bank期望还存在精确教师无关加性校正，与连续精确MLE同极小点；渐近知情项仅是其领先部分。这解释了应核查有限校准误差，但未实际计算该有限校正，也不构成优于已可计算精确似然的新算法。
''')
    put(DOC/'MAIN_RESULT_EN.md',r'''# Uniform correction can be effective while asymptotic calibration remains insufficient at finite dimension

In the known two-coordinate Gaussian-logistic model, the teacher-independent quadratic coefficient gamma0=(a²/4)r_a′(0) is a strong baseline. For a=1 and R=3/4, its high-dimensional limiting objective has strong convexity at least 1/48, and its integrated joint KL is at most 1/71136 uniformly over theta in [-R,R]. This analytic guarantee concerns limiting minimizers, not a finite n,N,B training procedure. Residual inconsistency therefore does not imply practical failure.

We extend the shared-full-bank learning argument from a fixed finite grid to continuous approximate minimization. With K=n+2, B=ceil(n^(1+eta)), independent iid observations and a prescribed diverging observation count, raw near-minimizers approach the positive boundary for a fixed positive teacher and the boundary set under the null. Marginalization and the informed asymptotic correction recover the teacher; the quadratic rule approaches its limiting minimizer. Raw-derived objectives require optimization error o(w_n), whereas marginalization requires o(1). These are not SGD or finite-dimensional endpoint guarantees.

A separately frozen growing-K population panel exposes a finite-dimensional boundary. At n=254,K=256,theta=.6, mean-score selection on 21 fixed candidates gives .75 for raw, .6 for marginalization, and .375 for both asymptotic corrections. This is not observed-data training or a continuous optimum certificate. A subsequent analytic construction identifies an exact teacher-independent correction for any fixed n and K>=2 at the conditional-bank-expectation level; its leading asymptotic term is 2Aw_nH. The exact finite correction was not computed, and access to the required model integrals already permits the cheap active likelihood baseline. No general algorithmic or computational advantage is claimed.
''')
    put(DOC/'REVIEW.md','''# 本轮审查入口

先读[REPORT](REPORT.md)与两幅图，再按[RESULTS](RESULTS.md)查看全部条件。当前完整论文仍是v4.1；本包不是新稿。

建议重点审查：

1. [REGULARIZATION_THEORY](REGULARIZATION_THEORY.md)的Hermite尾界、全教师1/71136风险及[独立审读](REGULARIZATION_AUDIT.md)。它是否准确回应“二次虽不精确一致但很有实际价值”。
2. [CONTINUOUS_LEARNING](CONTINUOUS_LEARNING.md)及[审读](CONTINUOUS_AUDIT.md)：真实共享bank、等距覆盖、连续marg偏差、近优集合和固定非零theta量词。N可慢增长不是未知结构学习率。
3. [有限K桥](FINITE_K_CALIBRATION.md)与[审读](FINITE_K_AUDIT.md)：条件期望身份/全区间加性合同唯一性、C1桥；不能当作有限B逐实现或计算优势。
4. 极限与有限面板必须分开。NUMERIC的主a1核查包括不同实现者[独立积分](NUMERIC_INDEPENDENT_AUDIT.md)；副a2/4失败不可追认。GROWTH是已知教师权重的总体估计，没有生成训练数据，21候选不等于连续globopt。
5. [GROWTH_AUDIT](GROWTH_AUDIT.md)的实际范围：全RNG，前1024bank独立重评分，其余保存算术；统计和教师权重另复核。开发t区间不是确认性或同时覆盖保证。

高级模型提醒已采纳，原HANDOFF包含新反馈及重复历史尾段，原字节不改。外部sandbox附件未取得，所有本包代码和数学独立建立；内部交叉审读不冒称外部专家盲审。一般Bregman/一致评分工具有既有原理。本轮最小新增认识是修正的能力、连续学习后果与有限维校准误差之间的明确边界，而非预设一类规则失败。

下一步见[NEXT](NEXT.md)，不需要外部审批；用户可以选择是否将本入口发给审稿人。本轮没有自行对外联系。
''')
    put(DOC/'NEXT.md','''# 下一步应先辨认有限校准误差

本轮改变了资源优先级：统一二次在当前极限范围已经很强，不值得为了方法标签继续寻找更复杂惩罚把它击败。有限面板中二次和非二次渐近项同时过修正，下一步应研究它们距离有限校准还差什么。

优先以少数固定n/K和教师无关条件目标，核查U′、V′以及相对于−V′的校准误差；比较渐近项、有限校准近似和直接边际化/精确似然所需的模型信息与成本。有限K桥已经证明一个精确参照存在，但本轮没有数值计算它。后续即使能够近似，也不得忽略本例精确active似然更便宜的竞争者。

随后另冻真实有限样本连续训练。教师同时含0和正内点；样本量按观测矩r(theta)及相同数据的精确似然基线规划，而非要求每流恢复潜在参数。逐项报告固定经验求解误差、bank误差与数据误差。当前增长K面板用总体四模式平均，不能作为这个训练阶段的替身。

英文完整稿可以并行开始，保持v4.1骨架，只吸收最有解释力的两点：简单统一修正的正面能力，以及有限维与极限校准的区别。新的连续学习可替代原固定网格限制；局部/全局阈值、条件校准唯一性等放到证明和边界中，不再展开多个平行故事。

固定噪声来源明确的应用设置仍值得研究，但未知结构、异质噪声和自由协方差继续后置。下一阶段任何新实验须另冻预算与判读规则；本包全部运行结束，没有自动追加队列。
''')
    put(DOC/'HANDOFF_DISPOSITION.md',f'''# 最新审查意见的处置

原HANDOFF SHA256 `{HANDOFF}`，针对父提交`{BASE}`；包含新反馈和重复历史尾段，原字节及损坏公式保持。未取得外部sandbox脚本/结果，未将其当已复核输入。

核心提醒采纳：不从精确不一致推出实际无效。我们独立复现a1数值并给出更强的全教师解析风险界，将gamma0作为统一事前强基线。局部/全局阈值、知情修正与连续MLE关系也完成证明；一般一致评分归既有文献。未按真实theta分别调参。

连续覆盖扩展已正式完成，不仅列为未来计划。有限增长K预算按原PLAN执行结束；没有见过修正失败后调低惩罚或增加n直到成功。全部面板、数值精度失败和不利结果保留。

有限K校准桥是在看完GROWTH后追加的纯解析推导，标为事后；不改原冻结，不新增科学积分，也不把公式存在当作已运行的修复方法。英文完整稿建议保留为可并行下一步，本轮没有改动v4.1。

外部认可不作为录用概率或新实验确认。当前结论是简单规则的极限能力很强、有限校准仍需检查，这两个事实共同决定下一阶段方向。
''')
    save(OUT/'POSTHOC_THEORY_RECORD.json',dict(trigger='GROWTH fixed-panel quadratic and informed asymptotic corrections over-shrink positive theta',
        observed_summary_sha256=sha(OUT/'GROWTH/SUMMARY.json'),new_file=(DOC/'FINITE_K_CALIBRATION.md').relative_to(ROOT).as_posix(),
        source_sha256=sha(DOC/'FINITE_K_CALIBRATION.md'),changed_original_plan=False,new_random_draws=0,new_integrals=0,
        finite_correction_computed=False,finite_residual_measured=False))
    save(OUT/'FIGURE_QA.json',dict(passed=True,actual_viewed_pngs={n:h for n,h in read(OUT/'FIGURES.json')['outputs'].items() if n.endswith('.png')},
        repaired_production_issues=['font lacks subscript-n; switched to mathtext','log minor tick labels overlapped; disabled minor ticks','mathtext le unsupported; changed to leq'],
        scientific_results_changed=False,remaining_required_fixes=[]))
    save(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,platform=platform.platform(),
        reproducibility='seed/call-order and hashes recorded for current environment; no universal cross-platform bitwise claim'))
    ledger=dict(numeric_production=nr['accounting'],numeric_alternate=read(OUT/'NUMERIC/AUDIT.json')['accounting'],
        numeric_independent=read(OUT/'NUMERIC/INDEPENDENT_AUDIT.json')['accounting'],growth=read(OUT/'GROWTH/FREEZE.json'),
        growth_audit_replay=read(OUT/'GROWTH/AUDIT.json')['ledger'],growth_stats_audit=read(OUT/'GROWTH/STATS_AUDIT.json')['ledger'],
        J_reference_new_scalar_integrals=read(OUT/'GROWTH/J_REFERENCE.json')['new_scalar_integrals'])
    oldpaths=['artifacts/paper_v4_1_20261002/MANIFEST.json','artifacts/paper_v4_1_20261002/VERSION_LOCK.json',
        'docs/07_papers_论文工作/ParticleJointSupervision/v4_1_working_20261002/MANUSCRIPT_ZH.md',
        'docs/07_papers_论文工作/ParticleJointSupervision/v4_1_working_20261002/APPENDIX_THEORY_ZH.md',
        'output/pdf/particle_joint_supervision_v4_1_zh.pdf','script/paper_v41_delivery.py']
    save(OUT/'DELIVERY_INPUT.json',dict(base_commit=BASE,handoff_sha256=HANDOFF,old_files={p:sha(ROOT/p) for p in oldpaths},
        new_samples=0,new_data_fits=0,science_ledger=ledger,
        samples_definition='new_samples means observed data; nonzero Gaussian prior simulation is separately enumerated in science_ledger'))
    save(OUT/'PRIOR_CHECKS.json',dict(checks=[dict(script='script/paper_v41_delivery.py',args=['--check'],returncode=0,
        script_sha256=sha(ROOT/'script/paper_v41_delivery.py'),result=dict(passed=True,pages=21,numbered_equations=62,workspace=False))]))
    save(OUT/'AUTHOR_REVIEW.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(DOC.glob('*.md'))},
        independent_math_audits=['REGULARIZATION_AUDIT.md','CONTINUOUS_AUDIT.md','FINITE_K_AUDIT.md'],
        root_full_theory_read=True,root_scientific_interpretation_review=True,remaining_required_fixes=[],
        external_blind_review=False,novelty_certified=False))
    frame=f'''# 当前论文论证框架

**主问题：精确统计任务不变时，有限prior粒子学习怎样改变相关，以及多简单的规则能修正它。** 当前稿件仍[v4.1](docs/07_papers_论文工作/ParticleJointSupervision/v4_1_working_20261002/README.md)；[新科研报告]({HOME}/REPORT.md)尚未进入稿件。

|环节|当前最小可守结论|边界|
|---|---|---|
|原始目标的学习偏移|新增连续真实共享bank近优理论：正theta趋R，null趋两端；marg/corr恢复真值|已知相关块、固定噪声，K=n+2/B=n^(1+eta)，iid数据且预定N发散；raw派生需epsilon=o(w)|
|简单修正的正面能力|a1/R.75统一gamma0=A r′(0)不使用theta，全教师极限joint KL≤1/71136|极限最小点保证，不是有限训练预算；不能把残余不一致当实际无效|
|有限维边界|n254/K256/theta.6面板raw选.75、marg.6、两渐近修正.375|总体评分、21固定候选、16独立prior流；无新训练、连续最优或同时置信保证|
|有限校准参照|任意n≥0/K≥2有teacher-free加性校正，在条件bank期望层与连续MLE同min；其领先项是2AwH|事后解析桥；精确有限校正/残差未计算；model-aware且无计算优势|

数值主a1独立核准；a2/a4的714生产与1055另一表达失败保留。一般梯度退化、Bregman/均值一致评分均有直接先例，新增解释限具体失真、连续学习与修正边界。

[下一步]({HOME}/NEXT.md)优先有限校准误差与真实有限样本验证，接受统一二次成功；不再预设复杂修正获胜。旧稿/旧科学原样，当前无科学运行或自动追加队列。
'''
    put(OUT/'FRAMEWORK_SNAPSHOT.md',frame)
    head=f'''# 最新研究状态：统一正则能力、连续学习与有限校准边界完成

[报告]({HOME}/REPORT.md) · [结果]({HOME}/RESULTS.md) · [审查]({HOME}/REVIEW.md) · [中文核心]({HOME}/MAIN_RESULT_ZH.md) · [English]({HOME}/MAIN_RESULT_EN.md)。产物artifacts/regularization_learning；完整稿v4.1/PDF仍原样封存。

新增解析全教师界：a1/R.75统一gamma0=A r′(0)的极限joint KL≤1/71136，强凸≥1/48；不能以残余不一致判简单正则实际失败。连续真实共享bank近优学习覆盖正theta，raw趋R/null趋端点，marg/corr恢复；raw派生误差需o(w)，非SGD/有限n精确贴边。

NUMERIC主a1生产2222/2222、另一表达2972/2972、不同实现关键积分650/650通过；61点最大参考二次KL约2.44e-6，非连续最大证书。a2/a4的714/1055精度失败保持，不追加节点。GROWTH n14/62/254且K=n+2，360相邻SE门通过；n254/theta.6固定面板raw.75、marg.6、两渐近修正.375。总体评分面板不是有限数据训练/连续globopt。

事后[有限K桥]({HOME}/FINITE_K_CALIBRATION.md)证明任意n≥0/K≥2的teacher-free精确条件期望校正与连续MLE同min，领先项为2AwH；有限校正/残差没有数值计算，不量化归因本次偏移或声称新通用算法。

16独立prior流、16384新bank、4194304 prior向量、1073741824 Gaussian标量；0新观测/数据拟合/NN。完整RNG重放；前1024bank独立评分，其余保存算术；36面板/576流选点及统计独立复核，积分与重算另记。三数学独立审读收口。

基准`{BASE}`；新HANDOFF SHA256 `{HANDOFF}`评议该提交，原字节和重复尾段保持，外部sandbox附件未取得。旧稿和科学不改，本包未写入新论文。只读`.venv/Scripts/python script/reglearn_delivery.py --check`；--workspace另核本地16NPZ和当前导航/HANDOFF，旧checker不带--workspace。所有运行结束，无自动追加队列。下一优先有限校准误差与另冻真实训练，不预设二次规则失败；下方历史与用户治理保持。

---

'''
    payload={}
    for name in ['README.md','REVIEW_BRIEF.md','AGENTS.md','RESEARCH_ROADMAP.md','docs/00_project_项目管理/START_HERE_当前工作入口.md']:
        old=(ROOT/name).read_bytes();assert not old.startswith('# 最新研究状态：统一正则'.encode())
        payload[name]=(head if '/' not in name else head.replace('](','](../../')).encode()+old
    payload['ARGUMENT_FRAMEWORK.md']=frame.encode()
    statepath='docs/00_project_项目管理/CURRENT_WORK_STATE.json';s=read(ROOT/statepath)
    s.setdefault('previous_research_states',{})['paper_v4_1_20261002']=s['current_research']
    s['updated_at']='2026-10-02'
    s['current_research']=dict(id='regularization_learning_20261002',status='complete_theory_population_panels_and_audits',base_commit=BASE,
        home=HOME+'/README.md',report=HOME+'/REPORT.md',results=HOME+'/RESULTS.md',review=HOME+'/REVIEW.md',next=HOME+'/NEXT.md',
        core_statement_ZH=HOME+'/MAIN_RESULT_ZH.md',core_statement_EN=HOME+'/MAIN_RESULT_EN.md',artifact='artifacts/regularization_learning',
        handoff_raw_sha256=HANDOFF,handoff_target=BASE,handoff_new_feedback=True,
        uniform_quadratic_limit_KL_bound='1/71136 at a=1,R=3/4, all theta in interval',continuous_near_optimal_shared_learning=True,
        finite_bank_exact_endpoint_claim=False,SGD_guarantee=False,finite_calibration_theory_posthoc=True,finite_calibration_computed=False,
        growth_independent_prior_streams=16,new_banks=16384,new_prior_vectors=4194304,new_Gaussian_scalars=1073741824,
        new_observed_samples=0,new_data_fits=0,new_neural_models=0,growth_fixed_candidates=21,growth_SE_checks=360,growth_SE_passed=360,
        numeric_precision_failures=714,alternate_precision_failures=1055,independent_numeric_checks_passed=650,
        full_Gaussian_RNG_replayed=True,independently_rescored_banks=1024,full_score_replay=False,
        manuscript_changed=False,current_paper_unchanged='v4.1-working',active_jobs=[],automatic_extension_queue=[],
        submission_completed=False,novelty_or_acceptance_certified=False)
    s['authority']['current_research_selection']=HOME+'/REPORT.md'
    s['execution_gate'].update(status='current_package_complete_no_scientific_queue',baseline_commit=BASE,
        current_phase='ICML_strengthening_regularization_boundary_complete')
    s['review_workflow'].update(detailed_entry=HOME+'/REVIEW.md',last_consumed_feedback_sha256=HANDOFF,
        feedback_use_mode='strong quadratic baseline reminder adopted; independent theory and numerical checks; raw preserved')
    s['continuing_user_goal']['current_stage']='regularization_learning_20261002_completed'
    s['active_tasks']=[];s['pending_tasks']=[]
    payload[statepath]=(json.dumps(s,ensure_ascii=False,indent=2)+'\n').encode()
    payload['.gitattributes']=(ROOT/'.gitattributes').read_bytes()+('\n# Uniform regularization and continuous learning evidence.\n'
        'script/reglearn_*.py -text whitespace=cr-at-eol\n'+HOME+'/** -text whitespace=cr-at-eol\n'
        'artifacts/regularization_learning/** -text whitespace=cr-at-eol\n'
        'artifacts/regularization_learning/HANDOFF_INPUT.md conflict-marker-size=128 whitespace=-blank-at-eol,-blank-at-eof,cr-at-eol\n'
        'artifacts/regularization_learning/figures/*.svg whitespace=-blank-at-eol,-blank-at-eof,cr-at-eol\n').encode()
    payload['.gitignore']=(ROOT/'.gitignore').read_bytes()
    print(json.dumps({name:base64.b64encode(b).decode() for name,b in payload.items()}))


if __name__=='__main__':main()
