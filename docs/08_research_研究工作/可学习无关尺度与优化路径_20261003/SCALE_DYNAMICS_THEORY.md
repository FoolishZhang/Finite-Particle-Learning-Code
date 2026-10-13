# 可学习无关尺度：局部漂移、路径噪声与参数化边界

日期：2026-10-03。新增解析文档，0RNG、0模型期望数值积分。本轮训练只有FIXED、FREE、FACT三臂；第6节反向配对只是未实验的解析候选。旧论文、旧理论与旧科学结果均保持冻结。

## 1. 固定模型与本页对象

沿用上一包合同 (historical reference not distributed)的四维active模型：均值为 \((-.6,-.2,.2,.6)\)，协方差 \(C(t)=I+(1+t)D\)，其中D为旧mixed-sign真实协方差减I，物理参数 \(t\in[-.75,.75]\)。真实t为0。附加n个独立零均值Gaussian-logistic通道，候选标准差向量为 \(\tau=(\tau_1,\ldots,\tau_n)\)。主训练n=256、K=100，每个观察独立K个prior粒子；不同方法共用每步的基础输入。

因Gaussian对称性，无关观察位始终独立公平，任意尺度均有
\[
Q_{t,\tau}(y,u)=Q_t^A(y)2^{-n}.
\]
所以学习尺度不改变精确观察分布；其作用对象是有限粒子评分的噪声与偏差。令 \(F_K(t,\tau)\) 为完整观察与prior均平均后的理想实算术负log粒子评分，\(F_K^{\rm fact}(t)\) 为直接因子分解评分，后者仍有active有限K误差。上一包已证明

$$
0\le F_K(t,\tau)-F_K^{\rm fact}(t)
\le\frac18\sum_i\tau_i^2,\qquad
\partial_{\tau_i}F_K(t,\tau)>0\quad(\tau_i>0).
\tag{S.1}
$$

本页分析严格正的总体尺度导数为何不等于有限步Adam必定缩小尺度。除特别注明，局部极限固定n、K、t及其它尺度；不能将其余项常数默认为增长n时统一。理想式也不自动认证float32计算。

## 2. 单个尺度的小量系数

固定一个完整观察、active粒子和除第i坐标外的所有无关输入。把其粒子似然归一化为 \(\beta_k>0\)、\(\sum_k\beta_k=1\)。第i坐标Gaussian与这些权重独立；固定0/1标签的符号可吸收到标准Gaussian \(G_k\) 中。把第i尺度从0改为τ带来的逐实现评分差为

$$
h_\beta(\tau,G)
=-\log\sum_{k=1}^K\beta_k\,2\sigma(\tau G_k),
\qquad
q_\beta=\sum_k\beta_k^2\in[1/K,1].
\tag{S.2}
$$

记 \(m=\sum_k\beta_kG_k\)。在τ=0处，括号内均值为1，其一阶导数为m/2、二阶导数为0，故

$$
h_\beta'(0,G)=-m/2,\qquad
h_\beta''(0,G)=m^2/4,\qquad
E_Gm^2=q_\beta.
\tag{S.3}
$$

Gaussian对称性给 \(E h_\beta(\tau,G)=E h_\beta(-\tau,G)\)。有限K下可微分四次并交换期望，且余项对整个权重单纯形一致可控。一个明确的控制如下：令 \(M=\max_k|G_k|\)、\(a_k(\tau)=\log[2\sigma(\tau G_k)]\)。有
\[
|a_k'|\le M,\quad |a_k''|\le M^2/4,\quad
|a_k'''|\le M^3/4,\quad |a_k''''|\le M^4.
\]
对 \(S=\sum_k\beta_ke^{a_k}\)，得到
\(|S'/S|\le M\)、\(|S''/S|\le5M^2/4\)、
\(|S'''/S|\le2M^3\)、\(|S''''/S|\le5M^4\)。
把这些界代入 \((\log S)''''\) 的商导数公式，得
\[
|h_\beta''''|\le
(5+8+75/16+15+6)M^4<40M^4.
\]
Gaussian最大值有有限四阶矩，因此控制收敛及Taylor余项均可用。由(S.3)和奇数阶均值为0，

$$
\begin{aligned}
E_G h_\beta(\tau,G)
&=\frac{q_\beta}{8}\tau^2+R_0,\qquad
|R_0|\le\frac{40}{24}E M^4\,\tau^4,\\
E_G h_\beta'(\tau,G)
&=\frac{q_\beta}{4}\tau+R_1,\qquad
|R_1|\le\frac{40}{6}E M^4\,|\tau|^3 .
\end{aligned}
\tag{S.4}
$$

这里其它尺度不必接近0：它们只改变条件权重β。进一步平均其余随机性，令
\(q_i(t,\tau_{-i})=E q_\beta\)，仍在 \([1/K,1]\)，(S.4)以q_i替代qβ成立。它是排除第i坐标后的逆ESS的期望
\(E[1/\mathrm{ESS}_{-i}]\)，不能写成 \(1/E[\mathrm{ESS}_{-i}]\)，也不能在τ=1时直接以全粒子ESS代入小τ系数。

若全部尺度趋0，则剩余权重是active粒子归一化权重α。记
\(q_A(t)=E\sum_k\alpha_k^2\)。固定n、K时，同样的四阶控制给

$$
F_K(t,\tau)-F_K^{\rm fact}(t)
=\frac{q_A(t)}8\|\tau\|_2^2+O_{n,K}(\|\tau\|_2^4).
\tag{S.5}
$$

证明可沿 \(\lambda\mapsto\lambda\tau\) 使用Taylor：每个log因子的r阶导数受 \(\|\tau\|_2^r\max_k\|G_k\|_2^r\) 控制，四阶期望有限；零点Hessian的期望为 \(q_A I/4\)，各混合项因独立中心Gaussian而消失。此处的Gaussian维数是n，余项常数一般随n增大。(S.1)的总平方尺度上界则无此局部限制。

## 3. 普通路径梯度的局部信噪比

对(S.2)直接重参数化微分，得到原粒子评分所用的物理尺度路径梯度 \(g_\tau=h_\beta'(\tau,G)\)。由(S.3)，在τ=0处条件于β精确有

$$
E(g_\tau\mid\beta)=0,\qquad
\operatorname{Var}(g_\tau\mid\beta)=q_\beta/4.
\tag{S.6}
$$

平方梯度的期望随τ为偶函数；其二阶导数受 \(C M^4\) 控制，因此与(S.4)合并得

$$
E g_\tau=\frac{q_i}4\tau+O_K(\tau^3),\qquad
\operatorname{Var}(g_\tau)=\frac{q_i}4+O_K(\tau^2).
\tag{S.7}
$$

余项仍可在β上平均；τ=0时的条件均值恒为0，故没有漏掉额外的条件均值方差。总体下降信号随τ消失，单次物理路径梯度的波动却保持非零。

对M个独立观察/prior输入的平均梯度，在同一固定参数点，均值与标准误之比为

$$
\frac{E\overline g_\tau}
{\sqrt{\operatorname{Var}(\overline g_\tau)}}
=\frac{\tau\sqrt{M q_i}}2+o(\tau)
\quad(\tau\downarrow0,\ M,K\text{固定}).
\tag{S.8}
$$

固定实际标签而只刷新各行prior时也有相应式，q_i用行系数平均替代。此式描述一个固定点梯度估计；它不把M、τ的相对量级变成学习器必要复杂度，也不排除动量、跨步累积或其它方差控制。优化轨迹上的参数和梯度历史是相关的，不能将512步当512次独立固定点确认。

## 4. 官方softplus参数化与floor

独立无关块中的Cholesky对角就是标准差。官方映射为
\[
\tau=\operatorname{softplus}(u)+\varepsilon,\qquad
\varepsilon=10^{-6}.
\]
其精确实算术链式因子为

$$
\chi(u)=\frac{d\tau}{du}
=\sigma(u)=1-\exp[-(\tau-\varepsilon)].
\tag{S.9}
$$

固定参数点的raw路径梯度是 \(g_u=\chi g_\tau\)，因此
\[
Eg_u=\chi Eg_\tau,\qquad
\operatorname{SD}(g_u)=\chi\operatorname{SD}(g_\tau).
\]
只要χ>0，单点均值/标准差比不因这个确定性重参数化而改变。

**理想零floor或中间尺度。** 当 \(\varepsilon=0,\tau\to0\)，或讨论
\(\varepsilon\ll\tau\ll1\) 的双尺度近似时，\(\chi\sim\tau\)，从(S.7)得到

$$
Eg_u\sim q_i\tau^2/4,\qquad
\operatorname{SD}(g_u)\sim \tau\sqrt{q_i}/2.
\tag{S.10}
$$

均值比标准差更快消退。它是softplus局部梯度的性质，不是Adam更新时间率。

**固定代码floor。** 保持 \(\varepsilon>0\)，让 \(u\to-\infty\)，实际是
\(\tau\downarrow\varepsilon\)、\(\chi\sim\tau-\varepsilon\)。
固定其它参数，若floor处物理梯度均值和标准差分别记为
\(d_i(\varepsilon)>0\)、\(s_i(\varepsilon)>0\)，连续性给

$$
Eg_u\sim(\tau-\varepsilon)d_i(\varepsilon),\qquad
\operatorname{SD}(g_u)\sim(\tau-\varepsilon)s_i(\varepsilon).
\tag{S.11}
$$

标准差严格正可由Gaussian全支撑和路径梯度关于该坐标输入不是常函数得到。此时信噪比趋向floor处的 \(d_i(\varepsilon)/s_i(\varepsilon)\)，不能把(S.10)不加限定地解释为代码中的τ继续趋0。若另考察小ε且其它尺度固定，(S.7)才给该比约为 \(\varepsilon\sqrt{q_i}/2\)。float32还可能把极小softplus项舍入掉而直接等于代码floor；本页实算术展开没有为该舍入提供精度保证。

**一个严格但受限的速度参照。** 仅在零floor、t及其它尺度固定时，考虑精确总体欧氏raw梯度流
\(\dot u=-\eta\partial_uF\)，不是Adam或SGD。尺度严格下降并趋0，且
\[
\dot\tau=-\eta\chi^2\partial_\tau F
\sim-\eta q_i\tau^3/4,\qquad
\tau(s)\sim\sqrt{\frac{2}{\eta q_i s}}\quad(s\to\infty).
\]
最后一步由 \(d(\tau^{-2})/ds\to\eta q_i/2\) 积分得到。正尺度处总体导数严格正，排除了该一维流停在正内点；导数有界也保证有限时间内不会跑到负无穷。若直接在物理尺度作精确梯度流，则 \(\dot\tau\sim-\eta q_i\tau/4\)，呈局部指数型衰减。这只说明参数化能改变理想动力学，不能把任一率赋给同时学习t和256尺度的有限步Adam。

尤其，在共同小尺度处，沿所有坐标共同变化的物理尺度方向，其总体导数领先项为 \(nq_A\tau/4\)；单个物理尺度坐标的对应领先项为 \(q_A\tau/4\)。若共享一个raw参数，或分别使用256个独立raw参数，这些物理导数均须再乘softplus链式因子χ；在零floor或 \(\varepsilon\ll\tau\ll1\) 时，raw梯度领先项分别为 \(nq_A\tau^2/4\) 和 \(q_A\tau^2/4\)。二者不是同一个受控程序。Adam的矩归一和epsilon又会改变梯度尺度的作用，不能直接把这个n倍梯度差称为Adam的n倍速度差。

## 5. 总平方尺度与增长维数的充分条件

由(S.1)，对任何确定的尺度向量
\[
\sup_t|F_K(t,\tau)-F_K^{\rm fact}(t)|
\le\|\tau\|_2^2/8.
\]
因此 \(\sum_i\tau_{n,i}^2\to0\) 是让无关部分的期望评分误差统一趋0的充分条件；共同尺度时变为 \(n\tau_n^2\to0\)。它不是必要条件，未控制active有限K误差，也不是尺度学习算法的收敛保证。若要保护量级为w_n的消退排序间隙，更强的 \(\sum_i\tau_{n,i}^2=o(w_n)\) 是一个直接充分条件。

固定代码floor ε时，\(\sum_i\tau_i^2\ge n\varepsilon^2\)，所以不能仅靠此界证明任意增长n下一致性。本轮n=256是固定维数，逼近floor时该上界为 \(3.2\times10^{-11}\)；这仍不等于512步训练能够达到该尺度，也不等于浮点评价达到该误差。

报告尺度不能只看均值或中位数：少数大尺度可支配平方和。应保留平方和、RMS、范围/分位数及256个最终物理尺度。对训练后固定输出，独立新prior的条件评分仍可用上述统一界；不能把同一训练prior上选出的随机参数直接代入固定参数的无偏MC陈述，而忽略选择依赖。

## 6. 未实验候选：反向配对能取消什么噪声

取相同active输入和实际标签，分别以G和−G计算两个各含K粒子的完整loss，再平均。两套无关输入的边缘分布都正确，故这种配对保持同一个期望目标 \(F_K\) 及其期望梯度。必须先各自取log再平均；把两个ensemble拼成2K再取log会改成另一个期望目标。

### 6.1 只翻转当前坐标

若只把第i坐标翻转，其余输入不变，则(S.2)中的β保持相同。两个物理梯度的平均为
\[
g_\tau^{\rm ant}
=\{h_\beta'(\tau,G)-h_\beta'(-\tau,G)\}/2
=\tau(\beta^\top G)^2/4+O_G(\tau^3).
\]
四阶包络还给L²余项控制，因为Gaussian最大值有有限八阶矩。条件于β，

$$
E g_\tau^{\rm ant}
=q_\beta\tau/4+O_K(\tau^3),\qquad
\operatorname{Var}(g_\tau^{\rm ant})
=q_\beta^2\tau^2/8+O_K(\tau^4).
\tag{S.12}
$$

这里使用 \(\beta^\top G\sim N(0,q_\beta)\)。若β也随机，leading variance为
\(\tau^2\{3E(q_\beta^2)-(Eq_\beta)^2\}/16\)，不能用条件方差均值替代全部方差。
在零floor或 \(\varepsilon\ll\tau\ll1\) 时，raw梯度均值与标准差都为τ²阶；这种单坐标翻转消去的是该坐标τ=0处的随机一阶项。对256个坐标分别构造这样的梯度不是两个完整loss就能自动实现的计算合同。

### 6.2 一次翻转整个无关块

如果G的全部n列一起反号，而只令 \(\tau_i\to0\)、其它尺度保持非零，那么第二套ensemble中的其它因子也改变了权重β。此时τ_i=0处的配对梯度一般为
\[
-\tfrac14\sum_k
[\beta_k(G_{-i})-\beta_k(-G_{-i})]G_{ki},
\]
并不为0。因此“全块反向配对总能取消每个小尺度的一阶噪声”是错误的。

若全部尺度共同变小，例如固定n、固定向量v且 \(\tau=a v\)、\(a\to0\)，全块翻转才消去零点的全部一阶随机项。为给出剩余维数效应，取最清楚的共同尺度 \(\tau_i=a\)。条件于active权重α，令
\[
q=\sum_k\alpha_k^2,\qquad
A=\operatorname{diag}(\alpha)-\alpha\alpha^\top,\qquad
G_i=(G_{1i},\ldots,G_{Ki})^\top.
\]
零尺度处逐实现loss Hessian为
\[
\mathcal H_{ii}=(\alpha^\top G_i)^2/4,\qquad
\mathcal H_{ij}=-G_i^\top A G_j/4\quad(i\ne j).
\]
于是第i个物理尺度的全块配对梯度满足
\(g_i^{\rm all-ant}=a\sum_j\mathcal H_{ij}+O_{n,K,G}(a^3)\)。
对独立Gaussian各列使用二阶及四阶矩，有

$$
\begin{aligned}
E\!\left(\sum_j\mathcal H_{ij}\,\middle|\,\alpha\right)&=q/4,\\
\operatorname{Var}\!\left(\sum_j\mathcal H_{ij}\,\middle|\,\alpha\right)
&=\frac{q^2}{8}+\frac{n-1}{16}\operatorname{tr}(A^2),\\
\operatorname{tr}(A^2)&=q-2\sum_k\alpha_k^3+q^2.
\end{aligned}
\tag{S.13}
$$

证明中 \(\operatorname{Var}[(\alpha^\top G_i)^2]=2q^2\)，
\(E(G_i^\top A G_j)^2=\operatorname{tr}(A^2)\)；
不同j的交叉项条件于G_i后均为0，且与对角项的协方差也为0。故反向配对后的每坐标剩余噪声仍可随n累积。在α均匀的示例中，leading单点均值/标准差比为
\[
\frac1{\sqrt{\,2+(n-1)(K-1)\,}},
\]
与仅翻当前坐标、条件权重固定时的 \(1/\sqrt2\) 不同。该算式是解析例子，不是本轮active权重均匀的经验断言。

在共同小尺度且零floor的限定下，乘上softplus链式因子后，均值和标准差均为a²阶，但常数可受n影响；尺度按不同速度下降时也不能自动维持每坐标的相对精度。固定代码floor处仍须用(S.9)—(S.11)的链式因子，不能把a趋0直接赋给代码。原active随机性在两次loss中被共享，也没有获得两个独立active样本的方差减少。

本轮不训练ANT臂。这些公式只说明未来的方差控制应检查适用尺度、梯度协方差与成本，不能宣称简单G/−G配对一定解决噪声或使Adam收敛。

## 7. 对当前三臂训练的约束

当前计划以512步、16个独立训练流、N4096、每步64观察、K100比较FIXED、FREE与FACT；t和各raw尺度均使用预定Adam lr .02、betas(.9,.999)、epsilon \(10^{-8}\)，无weight decay、梯度裁剪或schedule。t投影回区间后保留Adam moments，最后64步物理参数平均为主输出，LAST仅作诊断。这些是低维受控选择，不是官方网络默认学习率 .0001 的完整优化配置复现。

理论提供的只是同目标、下降方向与局部随机性解释。它不证明三臂的512步输出等于各自总体最优点；不把FREE相对FIXED的改善等同于完全消除偏移；不要求固定K100的FACT恰好等于truth。主问题应由独立流的预定风险比较回答，保留未恢复、反向或精度不足结果。

源依据为冻结官方SCBM commit 58c3e602cb11fb4ea9106226e0f9e58f50d5d10a 的models.py（SHA256 e3d95d557ed6afe95f842f8831c3e849e1b1d56b750150f00c363d41c04221a1）及上一包理论合同。实际代码资格、训练预算与参考积分另由本包冻结协议规定；本文只新增解析推导。

2026-10-03独立审读后修订：仅澄清第4节末的共享/独立尺度比较，区分物理导数与须乘χ的raw导数；S.1—S.13公式、实验合同及科学冻结均未改。原作者版SHA256为96cfb661945942f57d1e39cd2bf093f8968672eaf6c7a3fd631fbbf945b283a4。
