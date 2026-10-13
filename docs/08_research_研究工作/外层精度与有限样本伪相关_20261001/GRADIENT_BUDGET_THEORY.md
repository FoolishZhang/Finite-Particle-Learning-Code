# K=2 的近并列梯度与外层 bank 预算

本页为基准 `33f2d1c5378e269563a96cbfdd6eb0b2c32d9a7b` 后的新解析工作。0新科学积分、随机样本、拟合或优化。旧包及其冻结保持。研究对象是一个固定参数处的总体梯度估计精度，不是有限标签学习、经验优化器或同时变化发射尺度的保证。

## 1. 模型与两个标签合同

固定发射尺度 $a>0$、教师 active 相关 $\theta\in(-1,1)$、候选相关 $\rho\in[-R,R]$，其中 $0<R<1$。active 块含两个坐标；另有 n 个教师与候选均完全独立的 nuisance 坐标。所有基础 latent 方差为1，均值为0，二值发射为 $\sigma(aX)$。固定粒子数 $K=2$。以下渐近只令 $n\to\infty$，允许外层独立 bank 数 $B=B_n\ge1$ 变化。

使用自旋标签 $Y_i\in\{-1,1\}$。定义

$$
r_a(u)=E_u[\tanh(aX_1/2)\tanh(aX_2/2)],\quad
r_\star=r_a(\theta),\quad r=r_a(\rho).
\tag{GB1}
$$

教师 active 标签概率为 $p_y=(1+r_\star y_1y_2)/4$。对同一 bank 的两个 prior 粒子，令

$$
A_{k,y}=\sum_{i=1}^2\log\sigma(ay_iX_{ki}),\quad
\Delta_y=A_{1,y}-A_{2,y},\quad
H_{k,y}=y_1y_2\prod_{i=1}^2\sigma(-ay_iX_{ki}),\quad
M_y=H_{1,y}+H_{2,y}.
\tag{GB2}
$$

两粒子独立且各自服从相关为 rho 的二维标准 Gaussian。把 nuisance 标签符号吸收进各独立 prior 坐标，令

$$
U=\log\sigma(aZ),\quad v=\operatorname{Var}U>0,\quad
D_n=\sum_{l=1}^n(U_{1l}-U_{2l}),\quad
h(x)=\sigma(x)(1-\sigma(x)).
\tag{GB3}
$$

$D_n$ 与全部 active 变量独立。原总体目标为标准 iid 等权 prior 的负 log-mean likelihood；记其关于 rho 的导数为 $\mu_n=F'_{n,2}(\rho)$。

本页区分两种独立 bank 合同：

1. **单标签合同。** 每个 bank 独立抽取一次 $Y\sim p$，同一 bank 的两个粒子共用这个标签。bank 之间的 Y、active prior 和 nuisance prior 均独立。简记 $\Delta=\Delta_Y,M=M_Y$。
2. **四标签 RB 合同。** 每个 bank 只抽取 prior；四个 active 标签共用同一 active bank 和 $D_n$，以已知总体权重 $p_y$ 平均。它是第一种合同对标签的条件期望。null 教师 $\theta=0$ 时 $p_y=1/4$，可直接计算四项，或在单标签合同中直接模拟两个公平自旋，无教师求积误差。

RB 使用总体标签权重。复用一个有限 N 数据集、用经验权重替代 $p_y$、或先在同一批 bank 上选参数，均不是本页的合同。

## 2. 噪声差的密度引理

记 $f_n$ 为 $D_n$ 的密度，$L_n=\|f_n\|_\infty$，并令

$$
c_v=\frac1{\sqrt{4\pi v}}.
\tag{GB4}
$$

**引理1。** 存在只依赖固定 a 的有限常数 L，使对全部 $n\ge1$，$L_n\le L/\sqrt n$；对每个固定实数 x，

$$
\sqrt n f_n(x)\longrightarrow c_v.
\tag{GB5}
$$

**证明。** U 在 u<0 的密度为
$\varphi((u-\log(1-e^u))/a)/[a(1-e^u)]$，在 u≥0 延拓为0。左尾为 Gaussian 型；右端点的密度及各阶导数比任意 u 的幂更快趋零。因此密度为 Schwartz 函数。其特征函数 psi 平方可积，且在任意远离0的频率区域上模严格小于1。$U_1-U_2$ 的特征函数为 $|\psi|^2$，方差为2v。

Fourier 反演中，在0附近用 $|\psi(t)|^2=1-vt^2+O(|t|^3)$。按 sqrt(n) 缩放频率后，小频率被 Gaussian 可积函数支配并趋于相应 Gaussian 特征函数。在其余区域使用 $|\psi|^{2n}\le q^{2n-2}|\psi|^2$，q<1，尾积分指数衰减。这给标准化密度的一致局部极限及一致上界，因而得到 GB5 和充分大 n 的 $L_n$ 界。有限个剩余 n 的密度有界，可增大 L 包含。□

后面只需要 GB5 与统一密度上界。若 Z 表示与 $D_n$ 独立、分布不随 n 变化的 active 标记，且 $E\int|g(x,Z)|dx<\infty$，则支配收敛给

$$
\sqrt n\,E g(D_n,Z)
\longrightarrow c_v\int E g(x,Z)dx.
\tag{GB6}
$$

同样结论适用于平方以及下面可积的特征函数差。随机的 active 平移无需作统一截断：对每个固定标记平移先取点态极限，再由 $\sqrt n f_n\le L$ 支配。

## 3. 单标签 Price 梯度的精确常数

固定对角的 Gaussian Price 身份给出单 bank 无偏梯度

$$
G_n=-a^2h(D_n+\Delta)M,\qquad E G_n=\mu_n.
\tag{GB7}
$$

两个对称非对角协方差条目抵消 Price 式中的1/2；$\pi_1\pi_2=h(D_n+\Delta)$，故没有额外2倍系数。因为 $|M|\le2$、$h\le1/4$，有 $|G_n|\le a^2/2$。

设

$$
q_1=EM=\frac{r_\star+r}{2},\qquad q_2=EM^2>0,
\quad g_\star=-a^2c_vq_1,\quad V_\star=\frac{a^4c_vq_2}{6}.
\tag{GB8}
$$

正性是严格的：给定标签，同一 bank 的两个 H 有相同的符号且绝对值严格为正，故 M 几乎处处不为0。为核对 q2，记

$$
u_a=E\tanh^2(aZ/2),\quad
z_a(\rho)=E_\rho[\tanh^2(aX_1/2)\tanh^2(aX_2/2)].
$$

则可显式展开为

$$
q_2=\frac{2+2u_a+z_a(\rho)+6r_\star r+r^2}{8}.
\tag{GB9}
$$

具体地，给定 $\tau=Y_1Y_2$，吸收标签后的 prior 相关为 tau rho，且 $E(H_k\mid\tau)=(\tau+r)/4$。单粒子平方项的标签平均为 $[1+2u_a+z_a(\rho)+4r_\star r]/16$。两个独立粒子的条件交叉项为 $E(\tau+r)^2/16$。把两个平方项与两倍交叉项相加即得 GB9。

**定理1（矩渐近）。**

$$
\sqrt n\,\mu_n\to g_\star,\qquad
\sqrt n\,E G_n^2\to V_\star,\qquad
\sqrt n\,\operatorname{Var}G_n\to V_\star>0.
\tag{GB10}
$$

**证明。** 在 GB6 中代入 GB7，作变量替换 $t=x+\Delta$。使用

$$
\int_{\mathbb R}h(t)dt=1,\qquad
\int_{\mathbb R}h(t)^2dt
=\int_0^1 p(1-p)dp=\frac16.
$$

相应一、二次可积包络由 M 有界保证。最后 $\mu_n^2=O(n^{-1})$，在二阶矩的 $n^{-1/2}$ 尺度上可忽略。□

下文讨论相对精度时，明确要求 $g_\star\ne0$。例如 null 教师且 rho≠0 满足此条件；null/rho=0 的精确总体梯度为0，不能定义相对误差。更一般的 $r_\star+r=0$ 也不由本页控制其相对精度。

## 4. 外层 B 的三个区间

令 $\bar G_{n,B}=B^{-1}\sum_{b=1}^B G_{n,b}$，bank 独立。以下 B 可以是任意正整数序列。

**定理2（小预算）。** 若 $B/\sqrt n\to0$，则

$$
\sqrt n\,\bar G_{n,B}\longrightarrow0\quad\text{依概率}.
\tag{GB11}
$$

**证明。** 固定窗口宽度 T>0。条件于 active 标记，单 bank 满足 $|D_n+\Delta|\le T$ 的概率至多 $2TL_n$；全部 B 个 bank 中出现一次这样的近并列的概率至多 $2TBL_n\to0$。窗口外，

$$
\sqrt n E\bigl[|G_n|1_{\{|D_n+\Delta|>T\}}\bigr]
\le2a^2L\int_{|x|>T}h(x)dx.
$$

对窗口外的平均使用 Markov 界，先 n→∞ 再 T→∞ 即证。无需额外的 log n 因子。若 $g_\star\ne0$，平均值的期望乘 sqrt(n) 仍趋于非零 $g_\star$；这与概率极限0不矛盾，因为这批归一化随机变量不一致可积。□

**定理3（临界预算）。** 若 $B/\sqrt n\to\beta\in(0,\infty)$，则 $\sqrt n\bar G_{n,B}$ 依分布趋于随机变量 $W_\beta$，其特征指数为

$$
\log E e^{itW_\beta}
=\beta c_v\int_{\mathbb R}
E\left[\exp\left\{-\frac{it a^2}{\beta}h(x)M\right\}-1\right]dx.
\tag{GB12}
$$

等价地，令 N 为强度 $\beta c_v\,dx\,\mathcal L(M)(dm)$ 的 Poisson 随机测度，则

$$
W_\beta=-\frac{a^2}{\beta}\int h(x)m\,N(dx,dm).
\tag{GB13}
$$

该积分绝对可积，是有限变差的 shot-noise 极限；整个实轴上的强度无限，不能称为有限强度的复合 Poisson 分布。它满足

$$
E W_\beta=g_\star,\qquad
\operatorname{Var}W_\beta=V_\star/\beta>0.
\tag{GB14}
$$

**证明。** 对单 bank 的特征函数差，使用 GB6；外部归一化 $\sqrt n/B\to1/\beta$ 必须保留在指数内。由 $|e^{iz}-1|\le|z|$ 和 h 可积，支配收敛合法；平移 x+Delta 消去 Delta 后得到 GB12。单 bank 特征函数差为 O(n^{-1/2})，所以把 B 次乘积取对数时的二阶余项为 O(B/n)=o(1)。GB13 的有限绝对一阶矩与二阶矩分别由积分 h 和 h²给出。由于 M 几乎处处非零，其 Lévy 测度不为零，分布非退化。□

**定理4（大预算）。** 若 $B/\sqrt n\to\infty$，则

$$
\sqrt{\frac B{\sqrt n}}\left(\sqrt n\bar G_{n,B}-\sqrt n\mu_n\right)
\ \Longrightarrow\ N(0,V_\star).
\tag{GB15}
$$

**证明。** 每项中心化后的绝对值不超过 a²，而行内方差和
$B\operatorname{Var}G_n\sim BV_\star/\sqrt n\to\infty$。因此独立三角阵的 Lindeberg 条件直接成立。用 GB10 换回尺度即得 GB15。这里中心是有限 n 的真实梯度 $\mu_n$，不能在没有误差率时将 CLT 中的 $\sqrt n\mu_n$ 替换成 $g_\star$。□

**推论1（相对一致性的充要预算）。** 若 $g_\star\ne0$，则

$$
E\left(\frac{\bar G_{n,B}}{\mu_n}-1\right)^2
\sim\frac{V_\star}{g_\star^2}\frac{\sqrt n}{B}.
\tag{GB16}
$$

相对 L² 一致性、相对概率一致性均当且仅当 $B/\sqrt n\to\infty$。充分性由 GB16 和 Chebyshev 得到。若该比值不趋∞，取一个有界比值子列，再取收敛子列：极限0时 GB11 排除收敛到非零 $g_\star$；有限正极限时 GB12–14 给出非退化极限，同样排除。大预算下还可把 GB15 除以 $\sqrt n\mu_n$，得到相对 CLT，极限方差为 $V_\star/g_\star^2$。

这说明仅有 B→∞ 不够；需要有效近并列数量 B/sqrt(n) 增长。反过来，这不是估计器在所有参数处的必要预算下界，因为它排除了均值主项为0的点，并固定了估计器合同。

## 5. 有限 B 的有界浓度

由密度上界，对于所有 $n\ge1$，

$$
\operatorname{Var}G_n\le E G_n^2\le\frac{a^4q_2L_n}{6},
\qquad |G_n-\mu_n|\le a^2.
$$

标准 Bernstein 不等式因此给出对任意 x>0，

$$
P(|\bar G_{n,B}-\mu_n|\ge x)
\le2\exp\left\{-\frac{Bx^2}{a^4q_2L_n/3+(2/3)a^2x}\right\}.
\tag{GB17}
$$

若已知非零 $\mu_n$，把 $x=\epsilon|\mu_n|$ 代入即可得到有限 B 的相对误差充分预算。结合 GB10 与引理1，对固定 $0<\epsilon\le1$、$0<\delta<1$，充分大 n 时存在依赖固定模型参数的常数，使

$$
B\ge C\sqrt n\,\epsilon^{-2}\log(2/\delta)
\tag{GB18}
$$

足以保证相对误差至多 epsilon 的概率至少1−delta。这不声称已计算 L、实际梯度下界或渐近阈值 n0；GB18 不能直接当作已认证的具体实验预算。尤其当 $r_\star+r$ 很小时，其常数可很大。样本标准差恰好很小、甚至没有观察到近并列，也不能替代 GB17 的有界余项或相应有效事件量要求。

## 6. 四标签 RB：相同均值、不同二阶常数

令 Z 表示两个 active prior 粒子，定义

$$
g_{\rm RB}(x,Z)=-a^2\sum_y p_y h(x+\Delta_y)M_y,
\qquad G^{\rm RB}_n=g_{\rm RB}(D_n,Z),
$$
$$
Q_{\rm RB}=E_Z\int_{\mathbb R}g_{\rm RB}(x,Z)^2dx,
\qquad V^{\rm RB}_\star=c_vQ_{\rm RB}.
\tag{GB19}
$$

这一定义保留所有交叉标签、不同 active 平移的交叉项，**不能**把它替换成单标签的 $a^4q_2/6$。因为该估计器是对标签的条件期望，

$$
E G^{\rm RB}_n=\mu_n,\quad |G^{\rm RB}_n|\le a^2/2,\quad
0<Q_{\rm RB}\le a^4q_2/6,
$$
$$
\sqrt n\operatorname{Var}G^{\rm RB}_n\to V^{\rm RB}_\star.
\tag{GB20}
$$

可积性和上界由逐点 Jensen 与四个平移 h² 的积分得到。为证明严格正性，取两个 active 粒子相同，均为 $x=(x_1,x_2)$，则所有 Delta_y 为0，且

$$
\sum_y p_yM_y
=\frac12\{r_\star+\tanh(ax_1/2)\tanh(ax_2/2)\}.
\tag{GB21}
$$

可选 x 使右边非零。由于 a>0、|rho|<1，两个 prior 粒子的联合 Gaussian 在四维上全支持；在上述点的一个开邻域以及噪声参数 x 的一个有限区间上，核仍非零。于是平方积分的期望严格正。这里“两个粒子恰相同”的零概率事件只用于寻找连续函数非零的一点，正概率来自开邻域。

GB11、GB15、GB16 及相对概率充要结论均对 RB 成立，将 $V_\star$ 换为 $V^{\rm RB}_\star$。临界区间的精确特征指数应写为

$$
\beta c_v\int E_Z\left[
\exp\{it g_{\rm RB}(x,Z)/\beta\}-1\right]dx.
\tag{GB22}
$$

这是以整个 active bank 为标记的 Poisson 积分，不能仅用 M 的边缘分布替代。其均值仍为 $g_\star$，方差为 $V^{\rm RB}_\star/\beta>0$。

为完整核对小预算证明，在有限 D 窗口内使用 $P(|D_n|\le T)\le2TL_n$；窗口外由
$E\int_{|x|>T}|g_{\rm RB}(x,Z)|dx\to0$ 控制，先 n→∞ 再 T→∞。该尾积分成立，因为全部实轴的一阶积分由有限个 h 平移与有界 M 控制。临界证明则对 GB22 使用同一个可积一阶包络；大预算证明仍由有界性和 GB20 给出。因此 RB 改善常数而不改变这里的预算阶。

有限 B 时，GB17 的分母相应改为 $2L_nQ_{\rm RB}+(2/3)a^2x$。同一四标签必须使用同一 D 与 active bank；若改用四套独立 prior，这已经是不同的成本和方差合同。

## 7. 标准 Cholesky 路径梯度：无 CV 的 n 级预算

固定采用以下参数耦合，而不是声称所有重参数化方差相同：

$$
X_{k1}=Z_{k1},\quad X_{k2}=\rho Z_{k1}+\sqrt{1-\rho^2}Z_{k2},\quad
V_k=\partial_\rho X_{k2}=Z_{k1}-\frac\rho{\sqrt{1-\rho^2}}Z_{k2}.
\tag{GB23}
$$

V_k 与 X_{k2} 为独立 Gaussian，$EV_k^2=(1-\rho^2)^{-1}$。定义

$$
L_{k,y}=\partial_\rho A_{k,y}
=ay_2\sigma(-ay_2X_{k2})V_k,
\quad P_{n,y}=-\pi_{1,y}L_{1,y}-\pi_{2,y}L_{2,y}.
\tag{GB24}
$$

对每个固定标签，$E L_{k,y}=0$；路径微分合法且 $E P_{n,Y}=\mu_n$。令 $J_n=1$ 当 $D_n>0$，否则 $J_n=2$。D_n 对称且连续，故 J_n 均匀且独立于所有 active 标记。当 n 增长，有限 active 平移对 winner 的影响趋零，从而
$P_{n,Y}+L_{J_n,Y}\to0$ 于 L²。证明可条件于 active 标记使用 $P(|D_n|\le T)\to0$，再由 $|L_1|+|L_2|$ 的 Gaussian 二阶包络支配。于是

$$
\operatorname{Var}P_{n,Y}\longrightarrow
V_{\rm path}=\frac{a^2E\sigma(aZ)^2}{1-\rho^2}
=\frac{a^2(1+u_a)}{4(1-\rho^2)}>0.
\tag{GB25}
$$

对 null 教师的四标签路径平均，winner 对四标签渐近相同，而

$$
\sum_y\frac14 L_{k,y}
=-\frac a2\tanh(aX_{k2}/2)V_k.
$$

故其无 CV 方差极限为

$$
V_{{\rm path},{\rm RB}}
=\frac{a^2u_a}{4(1-\rho^2)}>0.
\tag{GB26}
$$

这与 Price RB 的 $n^{-1/2}$ 方差不同。a>0 固定使 u_a>0，不能在本式中同时令 a→0 后仍沿用常数阶比较。

当 $g_\star\ne0$ 时，两种无 CV 路径估计器分别满足

$$
E\left(\frac{\bar P_{n,B}}{\mu_n}-1\right)^2
\sim\frac{V_{\rm path}}{g_\star^2}\frac nB
\quad\text{或}\quad
\frac{V_{{\rm path},{\rm RB}}}{g_\star^2}\frac nB.
\tag{GB27}
$$

因此相对 L² 一致性要求且只要求 B/n→∞。相对概率一致性也有同一充要条件：路径变量具有统一 Gaussian 矩包络、方差趋于正数，B→∞ 时三角阵 CLT 适用。若 B/n 有有限子列极限，CLT 排除在 $n^{-1/2}$ 均值尺度上的集中；若 B 沿子列有界，则其极限是有限个独立非退化连续路径变量的平均，仍不能集中到趋零的均值窗口。充分性直接由 GB27 得到。

此处比较的是同样数量的独立 bank，不是同样运行时间；尤其四标签平均有额外算术。它不提供不同梯度方法的普遍最优性下界。

## 8. 一个噪声 softmax 控制变量：Price 不是唯一降方差途径

令 $w_n=\sigma(D_n)$，只用 nuisance 分数、不含 active 分数。由于它独立于 active 标记且每个 $E L_{k,y}=0$，
$w_nL_{1,y}+(1-w_n)L_{2,y}$ 是零均值控制变量。把它加到 GB24 后得到

$$
P^{\rm CV}_{n,y}
=-\{\sigma(D_n+\Delta_y)-\sigma(D_n)\}
(L_{1,y}-L_{2,y}),\qquad E P^{\rm CV}_{n,Y}=\mu_n.
\tag{GB28}
$$

对任意实数 Delta，平移的单调性给出 L¹差为 |Delta|；Minkowski 不等式与 $\|h\|_2^2=1/6$ 给出另一个平方积分界：

$$
\int[\sigma(x+\Delta)-\sigma(x)]^2dx
\le\min\{|\Delta|,\Delta^2/6\}.
\tag{GB29}
$$

所以

$$
E(P^{\rm CV}_{n,Y})^2
\le L_n E\left[(L_{1,Y}-L_{2,Y})^2
\min\{|\Delta_Y|,\Delta_Y^2/6\}\right]
=O(n^{-1/2}).
\tag{GB30}
$$

最后一个期望有限：log-sigmoid 至多线性增长，L_k 有 Gaussian 线性包络，固定严格正定范围保证所需多项式矩有限。四标签平均再由 Jensen 继承这个上界。

因此在 $g_\star\ne0$ 时，B≫sqrt(n) 也足以使该 CV 版本相对 L² 一致。这里不进一步声称其方差下界、最优常数或全部临界极限，也未比较数值实现成本。这个简单构造足以说明：第7节的 n 级需求属于未加控制变量的标准路径估计器，不能包装成所有 pathwise 方法的限制，Price 也不是唯一能消除常数级 winner 波动的途径。

## 9. 本页能与不能支持的结论

在固定非零均值主项点，Price 的外层相对精度尺度是 B/sqrt(n)，而所比较的无 CV Cholesky 路径的外层相对精度尺度是 B/n。低预算 Price 平均趋近0是少见近并列造成的概率现象，不是总体梯度为0，也不是优化达到驻点。临界预算的非 Gaussian 极限进一步说明仅看很大的 B 数值而忽略 n 不足以判断普通近似区间是否可靠。

全部结论固定 K=2、a、rho 与教师相关。没有证明关于这些参数同时变化的统一率，没有有限 n 实际极小点证书，也没有 N 标签与 B bank 复用下的学习保证。标签 RB 使用已知总体权重；学习后的选点需要独立复评或另行处理选择依赖。该页是经典局部极限、Price、Poisson 三角阵及方差缩减在本模型中的组合，不以通用工具或控制变量本身主张新意。
