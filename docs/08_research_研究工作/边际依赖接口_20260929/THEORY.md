# 固定边际接口与有限采样的局部依赖门槛

本文件给出解析命题及其可证伪数值合同，不将尚未执行的数值案例写成结果。适用对象是 Gaussian-logistic 二元联合模型；证明不依赖本项目旧实验是否成功。所有命题均须与末节的已有工作及适用限制一起使用。

## 1. 对象和两种不同问题

固定一个信息状态或表示 $h$。真实分布为 $P(C\mid h)$，$C\in\{0,1\}^d$，所有边际 $p_j=P(C_j=1\mid h)\in(0,1)$。下文省略 $h$。令 $g(u)=(1+e^{-u})^{-1}$，并令

\[
 \mu_j=\operatorname{logit}(p_j),\qquad
 \eta\sim N(\mu,\Sigma),\qquad
 C_j\mid\eta\stackrel{\mathrm{ind}}\sim\operatorname{Bernoulli}(g(\eta_j)).
\]

其观察分布记为 $Q_{\mu,\Sigma}$。在 $\Sigma=0$ 时，$Q_0(C)=\prod_jp_j^{C_j}(1-p_j)^{1-C_j}$，具有正确的边际，但不一定具有正确的联合依赖。

这里有两个问题，不能合为一项：

1. **表示接口问题**：固定作为边际概率 logit 的 $\mu$，只加协方差，能否同时保持原边际并表达依赖？
2. **训练目标问题**：即使允许均值补偿，有限 iid 采样的联合概念目标能否从独立边界获得学习弱依赖的下降方向？

“边际正确”在这些命题中指给定同一 $h$ 的正确概率，不仅是通常的可靠性条件 $E[C_j\mid \widehat p_j]=\widehat p_j$。后者允许在同一预测概率层内存在不同的真实条件概率。低 ECE 或整体温度校准不能认证本节前提。对类级属性 CUB，完整图像的真实标签又可能近乎确定；自然实验中应说明 $h$ 究竟是特征、模型输入还是已压缩预测，不能直接把经验校准提升为逐图像真概率。

## 2. 命题一：非中心 logit 的严格收缩

定义 $m(\mu,s)=E_Zg(\mu+sZ)$，$Z\sim N(0,1)$，$s\ge0$。有

\[
 m(\mu,s)-\frac12
 =\frac{\sinh\mu}{2}\,
 E_Z\frac{1}{\cosh\mu+\cosh(sZ)}. \tag{1}
\]

因此，对任意有限 $\mu\ne0$：

- $m(\mu,0)=g(\mu)$；
- $s>0$ 时，$m(\mu,s)$ 严格位于 $g(\mu)$ 与 $1/2$ 之间；
- $|m(\mu,s)-1/2|$ 随 $s\ge0$ 严格递减，且趋于零。

**证明。** 直接化简得到

\[
 g(\mu+u)+g(\mu-u)-1
 =\frac{\sinh\mu}{\cosh\mu+\cosh u}.
\]

利用 $Z\overset d=-Z$ 即得 (1)。当 $s$ 增大时，分母对 $Z\ne0$ 严格增大，而高斯分布满足 $P(Z=0)=0$。极限由有界收敛给出。这个证明也适用于对称且在非零值上有正概率的加性噪声；不需要把它包装为高斯专有的新一般定理。

**零均值例外。** $\mu=0$ 时，对所有 $s$，$m(0,s)=1/2$。因此“任意正方差都会损坏边际”是错误陈述，必须排除中心概率 $1/2$。

### 推论一：固定边际 logit 的不兼容性

若固定 $\mu_j=\operatorname{logit}(p_j)\ne0$，并要求 $Q(C_j=1)=p_j$ 精确成立，则命题一强迫 $\Sigma_{jj}=0$。半正定矩阵满足 $\Sigma_{jk}^2\le\Sigma_{jj}\Sigma_{kk}$，所以整行非对角元素也为零。于是该概念在 $Q$ 中与其余概念独立。

若所有 $p_j\ne1/2$，同时精确保留全部原边际只能取 $\Sigma=0$，得到乘积分布。若存在 $p_j=1/2$，非独立结构仍可能保留在这些中心概念的子块中。

这是一项**固定上游边际 logit 的接口限制**。允许重新拟合均值、改变发射链接或采用边际化参数化，会改变前提；不能据此否定可联合更新均值的整个 SCBM 模型族。

## 3. 命题二：精确联合似然主动接受边际失准

取任意固定 $S\succeq0$，考察 $\Sigma=tS$，$t\ge0$。真实概念协方差记为 $\Gamma=\operatorname{Cov}_P(C)$，$D=\operatorname{diag}\{p_j(1-p_j)\}$。令精确人口联合 NLL 为

\[
 R_\infty(t)=E_P[-\log Q_{\mu,tS}(C)].
\]

则

\[
 R_\infty'(0+)
 =-\sum_{i<j}S_{ij}\Gamma_{ij}
 =\frac12\operatorname{tr}\{S(D-\Gamma)\}. \tag{2}
\]

**证明。** 对每一状态 $c$，令

\[
 f_c(\mu)=\prod_jg(\mu_j)^{c_j}[1-g(\mu_j)]^{1-c_j},\qquad
 r_j=c_j-p_j.
\]

有 $\partial_j f_c/f_c=r_j$，$\partial_{jj}f_c/f_c=(1-2p_j)r_j$，以及 $i\ne j$ 时 $\partial_{ij}f_c/f_c=r_ir_j$。高斯平滑的边界展开为

\[
 Q_{\mu,tS}(c)=f_c(\mu)+\frac t2\operatorname{tr}\{S\nabla^2f_c(\mu)\}+O(t^2).
\]

有限 $d,\mu,S$ 下 logistic 乘积的所需导数有界，展开可逐状态积分；所有 $f_c(\mu)>0$，所以对数展开也合法。因此

\[
 \left.\frac{d}{dt}\log Q_{\mu,tS}(c)\right|_{0+}
 =\frac12\sum_jS_{jj}(1-2p_j)(c_j-p_j)
   +\sum_{i<j}S_{ij}(c_i-p_i)(c_j-p_j).
\]

在 $P$ 下取期望时，对角项因边际正确而消失，交叉项为真实协方差，得到 (2)。

**严格下降的存在性。** 若某一对 $\Gamma_{ij}\ne0$，可取

\[
 S=I+a\,\operatorname{sign}(\Gamma_{ij})(e_ie_j^T+e_je_i^T),\qquad 0<a<1.
\]

该 $S$ 严格正定，且 $R_\infty'(0+)=-a|\Gamma_{ij}|<0$。故即使上游边际完全正确、数据无限、联合似然精确，也存在增加方差并改善联合风险的方向。对参与的非中心概念，这同时引入命题一的边际失准。这里的失准不是有限 MC 或上游过拟合所必需造成的。

这不意味着最优解一定取得任意预定幅度的失准；小 $t$ 的失准可任意小。命题只保证局部下降方向，且只用两两协方差。纯高阶依赖可以有 $\Gamma_{ij}=0$ 而仍非独立，例如三个公平二值变量的奇偶约束；此时本一阶判据为零，不认证独立性。

### 局部边际损失的二阶系数

令 $q_j(t)=Q_{\mu,tS}(C_j=1)$。则

\[
 q_j(t)=p_j+a_jt+O(t^2),\qquad
 a_j=\frac{S_{jj}}2p_j(1-p_j)(1-2p_j). \tag{3}
\]

逐概念 NLL 相对正确概率的超额为

\[
 D_{\mathrm{KL}}(\operatorname{Ber}(p_j)\Vert\operatorname{Ber}(q_j(t)))
 =\frac{S_{jj}^2}{8}p_j(1-p_j)(1-2p_j)^2t^2+O(t^3). \tag{4}
\]

按正类概率计算的单项 Brier 超额是 $(q_j(t)-p_j)^2=a_j^2t^2+O(t^3)$。若采用两个类别概率的平方误差之和，系数再乘二。

所以在 (2) 严格为负的方向，联合 NLL 收益对 $t$ 为一阶，而边际 NLL/Brier 代价为二阶。概率偏差本身仍是一阶，不能误写成“边际概率到二阶才改变”。这给出了联合评分与边际评分偏好不同模型的局部机制，但没有证明任意有限操作点或自然数据上会排序翻转。

## 4. 命题三：有限 iid 采样有精确的局部谱门槛

令 $Z_1,\ldots,Z_K\stackrel{\mathrm{iid}}\sim N(0,S)$，独立于 $C\sim P$。定义标准先验采样联合目标

\[
 F_K(t)=E_{P,Z_{1:K}}\left[-\log\left\{\frac1K\sum_{k=1}^Kf_C(\mu+\sqrt t Z_k)\right\}\right].
\]

不加概率下限、不附加精度正则、任务项或重要性提议。则对任意有限整数 $K\ge1$，

\[
 \begin{aligned}
 F_K'(0+)
 &=R_\infty'(0+)+\frac1{2K}\operatorname{tr}(S\Gamma)\\
 &=\frac12\operatorname{tr}\left\{S\left[D-\left(1-\frac1K\right)\Gamma\right]\right\}. \tag{5}
 \end{aligned}
\]

这是 $t=0$ 的精确右导数，不是“大 $K$”近似，也不是有限步训练必然收敛到某处的声明。

**证明。** 令 $s=\sqrt t$，$\bar Z=K^{-1}\sum_kZ_k$，并令 $H_c=\nabla^2f_c(\mu)/f_c(\mu)$。在固定 $c,Z_{1:K}$ 下，对数的二阶展开给

\[
 \begin{aligned}
 -\log\frac1K\sum_kf_c(\mu+sZ_k)
 =&-\log f_c(\mu)-s\,r^T\bar Z\\
 &-\frac{s^2}{2K}\sum_kZ_k^TH_cZ_k
  +\frac{s^2}{2}(r^T\bar Z)^2+O(s^3).
 \end{aligned}
\]

对 Gaussian 样本取期望，线性项为零，$E\bar Z\bar Z^T=S/K$，二次项得到精确目标导数加 $r^TSr/(2K)$。再用 $E_Pr=0$、$E_Prr^T=\Gamma$ 得 (5)。交换微分与期望可由 log-sum 的前三阶导数控制：每个 log-likelihood 的 logistic 导数有界，所余因子是有限 Gaussian 样本范数的有限阶多项式，具有可积上界。也可先在 $s$ 下求两次导数，再由样本联合符号对称性及 $t=s^2$ 得右导数，避免把逐随机路径的 $\sqrt t$ 项误认为不存在。

### 谱阈值

定义真实二元概念的相关矩阵 $R_C=D^{-1/2}\Gamma D^{-1/2}$。当 $K>1$ 时，存在 $S\succeq0$ 使 $F_K'(0+)<0$，**当且仅当**

\[
 \lambda_{\max}(R_C)>\frac K{K-1}. \tag{6}
\]

**证明。** 记 $A_K=D-(1-1/K)\Gamma$。PSD 锥自对偶，所以存在 PSD $S$ 使 $\operatorname{tr}(SA_K)<0$ 当且仅当 $A_K$ 有负特征方向。而

\[
 A_K=D^{1/2}\left[I-\left(1-\frac1K\right)R_C\right]D^{1/2}.
\]

合同变换保持惯性，得到 (6)。若要求 $S\succ0$，可给严格下降的秩一方向加足够小的正对角扰动，结论不变。

- $K=1$：$F_1'(0+)=\operatorname{tr}(SD)/2>0$ 对任意非零 PSD $S$ 成立，无下降方向。
- $\lambda_{\max}=K/(K-1)$：存在非零零导数方向，但没有严格负一阶方向；二阶及更高阶未决。
- 阈值以下：所有方向一阶非下降；**不等于已证实零协方差是全局最优，也不等于无法从非零初始化学到依赖**。
- $d=2$：$\lambda_{\max}=1+|\operatorname{Corr}_P(C_1,C_2)|$，故下降方向存在当且仅当 $|\operatorname{Corr}_P(C_1,C_2)|>1/(K-1)$。

这将学习依赖的一阶信号与有限采样的 Jensen 代价明确分开。它不要求固定方差，因而与旧“固定方差附加 nuisance”案例是不同命题。所有 $K$ 的公式都针对 iid 先验样本和精确期望；对一个固定随机流的目标、优化器轨迹、截断/下限实现，必须单独核对。

## 5. 均值补偿能修接口，不能消掉 iid 门槛

对任意指定 $p\in(0,1)$ 与有限 $v\ge0$，存在唯一 $b(p,v)\in\mathbb R$ 满足

\[
 E_Zg(b(p,v)+\sqrt v Z)=p. \tag{7}
\]

这是因为左侧对 $b$ 连续严格递增，并在 $b\to\pm\infty$ 时趋于零/一。沿 $v_j=tS_{jj}$ 的精确保边际补偿有

\[
 b_j(t)=\mu_j-\frac{tS_{jj}}2(1-2p_j)+O(t^2). \tag{8}
\]

式 (8) 由隐函数微分及 (3) 得到。代入逐状态 score 后，其均值漂移恰消去命题二的对角 score，所以剩余一阶项只有交叉项。

更一般地，任何 $\mu(t)=\mu+t a+o(t)$ 的确定性光滑均值更新，在人口一阶导数中只增加 $-a^TE_P(C-p)=0$。因此 (2)、(5)、(6) 对这种均值更新仍成立；(7) 的边际补偿是特例。iid 采样平均的 $\sqrt t$ 波动也不受这项 $O(t)$ 均值补偿影响。

所以保持边际与消除有限采样局部障碍是两件事。不能把均值补偿失败归因于它未精确保边际，也不能期待它自动解决所有 MC 训练偏差。

(7) 是已知的 **marginally specified logistic-normal / marginalized multilevel model** 参数化，不是本项目的新方法。此处采用它的用途是隔离边际与依赖的接口，并检验有限采样门槛是否仍在。

## 6. 采样反事实：成对样本移除一阶门槛

若偶数 $K=2L$，样本采用 $Z_1,-Z_1,\ldots,Z_L,-Z_L$，每个 $Z_\ell\sim N(0,S)$，则样本平均严格为零。沿同一模型路径，在命题三的展开中 $r^T\bar Z=0$ 逐样本成立，因此

\[
 (F^{\mathrm{anti}}_K)'(0+)=R_\infty'(0+)
 =-\sum_{i<j}S_{ij}\Gamma_{ij}. \tag{9}
\]

边际补偿或其他上述 $O(t)$ 均值漂移不改变 (9)。每个样本的边缘分布仍是所需 Gaussian，二阶平均的期望仍正确；改变的是样本间依赖与一阶随机均值误差。

这只移除 **$t=0$ 的一阶 Jensen 项**，不使有限 $K$ 目标在任意 $t>0$ 精确、无偏，也不保证全局最优、更好自然任务风险或所有训练阶段的方差更低。反向成对采样是标准 antithetic 工具，本项目不能宣称发明该采样方法。其研究价值若成立，应在于精确预测并消除本模型的局部依赖门槛。

## 7. 固定非零均值下，协方差不具有旧等价族的任意性

若已知固定的每个 $\mu_j\ne0$，完整观察分布唯一确定模型内的 $\Sigma$：每个边际通过命题一的严格单调性唯一确定方差；给定两边际方差，Price 恒等式给出

\[
 \frac{\partial}{\partial c}\,
 E[g(\eta_i)g(\eta_j)]
 =E[g'(\eta_i)g'(\eta_j)]>0,
 \quad c=\Sigma_{ij}, \tag{10}
\]

在 $-\sqrt{v_iv_j}<c<\sqrt{v_iv_j}$ 内严格递增，连续延伸到 PSD 边界。因此每个二元联合又唯一确定对应协方差。零方差行由 PSD 约束确定。

这说明旧自由均值/零均值的观察等价族不能直接移植到本固定非零均值前提。均值补偿虽然可保持边际，却不保证潜在尺度可识别：例如二概念允许任意方差并按 (7) 补偿均值时，固定边际只剩一个观察相关自由度，仍可能存在连续的潜在等价表示。本文只需要观察层面的正确预测，不以 latent 参数“像真值”代替观察风险。

### 允许边际误差下的一个可计算界

固定非中心 $\mu_j$，若要求 $|m(\mu_j,\sqrt{v_j})-p_j|\le\epsilon_j<|p_j-1/2|$，命题一唯一给出上限 $v_j\le V_j(\epsilon_j)$。由 (10) 与 $0<g'\le1/4$，

\[
 |\operatorname{Cov}_Q(C_i,C_j)|
 \le\frac{|\Sigma_{ij}|}{16}
 \le\frac{\sqrt{V_i(\epsilon_i)V_j(\epsilon_j)}}{16}. \tag{11}
\]

这是方便但可能很松的必要条件。给定精确方差时，更紧的二元相关上下端点可在 $\Sigma_{ij}=\pm\sqrt{v_iv_j}$ 用一维 Gaussian 积分算出；不能不证明就断言所有误差容许区间的最紧端点必在两个方差上限同时取得。若边际误差容许到达距 $1/2$ 的全部距离，方差上限不再有限，(11) 失去此形式的约束。

## 8. 最小数值核验案例的预定义

以下定义是供独立实现核验解析式的合同，不含训练和已运行结果。

**案例 A，非中心相关。** 取 $p_1=.8,p_2=.7$，真实 $P(11)=.64,P(10)=.16,P(01)=.06,P(00)=.14$。因此 $\Gamma_{12}=.08,D=\operatorname{diag}(.16,.21)$。固定 $S=\begin{pmatrix}1&.9\\.9&1\end{pmatrix}\succ0$。无需数值积分即可预言：

\[
 R_\infty'(0+)=-.072,\qquad
 F_K'(0+)=-.072+\frac{.257}{K},\qquad
 (F_K^{\mathrm{anti}})'(0+)=-.072.
\]

边际概率导数为 $(-.048,-.042)$；两项边际 KL 之和的二阶系数是 $.0072+.0042=.0114$。边际补偿版本的边际概率恒为 $(.8,.7)$，但上述人口联合及有限 iid 一阶导数不变。

**案例 B，独立反例。** 保留 $p,S$，改用真乘积 $P(11)=.56,P(10)=.24,P(01)=.14,P(00)=.06$。精确和 antithetic 的一阶导数均为零，iid 为 $.185/K>0$。由于此时 $Q_0=P$，精确联合风险的全局最小已经在原分布达到；任何严格低于其熵的数值结果都是实现错误。

**案例 C，中心例外。** 取 $p_1=p_2=.5$，$P(11)=P(00)=.35$，$P(10)=P(01)=.15$，沿案例 A 的 $S$。两边际在任何方差下均精确为 $.5$，同时可学习依赖。精确/antithetic 一阶导数为 $-.09$，iid 为 $-.09+.34/K$。这是针对不兼容命题前提的必保留反例。

独立实现应核对：积分结果随 $t\downarrow0$ 的斜率；逐样本平均的有限 $K$ 目标而非 log-of-exact-mean；固定均值与补偿均值；iid 与相同总样本数的成对采样；以及 (6) 的谱阈值。有限差分与 Monte Carlo 会有消差/采样误差，不能拿单随机流的斜率替代人口导数。具体步长、样本数、数值容差应在执行包中冻结，并完整保留所有预定格点，不按符合理论的格点筛选。

## 9. 新颖性边界与最强反驳

Heagerty 1999 已提出边际化 logistic-normal 参数化，使边际均值与关联模型分开。因此均值补偿、边际和条件 logit 的区别不能作为新的通用方法。原文：[Marginally Specified Logistic-Normal Models for Longitudinal Binary Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0006-341X.1999.00688.x)。后续可读实例明确采用一维积分约束求连接截距：[Practical Marginalized Multilevel Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC3865434/)。

Gaussian 协方差导数、高斯平滑与 (10) 属于 Price 类恒等式的标准工具；可核对 [A general version of Price's theorem](https://arxiv.org/abs/1710.03576)。正定/半正定约束、PSD 锥对偶和 Bernoulli KL 局部展开也是标准结果。

固定模拟样本数的 log-likelihood 存在 Jensen 偏差，且采样设计改变偏差，是已有 simulated-likelihood 问题；不能把“log 后取平均不同于先平均再 log”称为发现。[Train 的 Halton Sequences for Mixed Logit](https://escholarship.org/uc/item/6zs694tp)明确讨论模拟概率的对数偏差与采样设计。

当前可能有具体研究增量的是：在明确的 Gaussian-logistic 概念接口中，给出精确的人口局部矩阵 $A_K$、可检验谱阈值、均值补偿不改变阈值而成对采样改变阈值的区分，并检验这些预测在实际训练中有无实质后果。这尚不是原创性认证；还须对照 variance-component score tests、模拟似然边界偏差、IWAE/posterior collapse 和 antithetic 近邻。

最强的拒绝理由包括：第一，固定上游均值的接口若只是本项目人为适配，就不能代表原方法的一般缺陷；第二，谱阈值只控制独立边界的局部方向，若实际初始化/终点远离该边界，可能没有实践解释力；第三，$K=100$ 的二元阈值仅 $1/99$，其实际效应可能很小；第四，已有方差分量/模拟似然文献可能已含同一数学结论或直接推论。新实验必须允许这些反驳成立，不能由一个理论正确案例直接宣布找到论文核心。

本文未证明反馈策略排序、任务头性能、自然数据泛化或因果效应。后续若使用反馈/任务指标，必须给它们独立的实验定义；不能从 (2)–(9) 自动推出。
