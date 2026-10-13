# 有界异质无关发射：多项式粒子数下的一致相关方向

日期：2026-10-02。基准 `ecf844bcffb8f0b4810a87cfb743654913787a40`。本文件仅作解析推导，不生成RNG、数值积分、拟合或训练结果。它将旧[多项式K证明](../无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_THEORY.md)及v5.1附录A.2 (historical reference not distributed)中的iid无关和改为有界异质三角阵。关键不是普通CLT，而是对全部允许尺度数组一致的倾斜局部密度、相对上尾和粒子对积分。

## 1. 模型、量词与结论

固定 $0<R<1$、$0<a_-\le a_+<\infty$、$C_0,p>0$。每个n选择一个确定性数组 $\boldsymbol a_n=(a_{n1},\ldots,a_{nn})\in[a_-,a_+]^n$。该数组为已知模型输入，不依赖候选相关或抽取的粒子；不要求其经验分布随n收敛。教师与候选使用相同数组。

active块发射尺度固定为一，只学习已知两位的相关 $\rho\in I=[-R,R]$。令全部基础Gaussian独立，

$$
X_1=Z_1,\quad X_2=\rho Z_1+\sqrt{1-\rho^2}Z_2,\qquad
P(Y_1=y_1\mid X)=\sigma(y_1X_1),\quad
P(Y_2=y_2\mid X)=\sigma(y_2X_2),
\tag{H.1}
$$
$$
P(Y_{i+2}=y_{i+2}\mid Z)=\sigma(a_{ni}y_{i+2}Z_{i+2}),
\qquad 1\le i\le n.
\tag{H.2}
$$

各发射条件独立，$y_i\in\{-1,1\}$，真实相关 $\theta\in I$ 固定。标准iid等权先验粒子独立于观测。每个粒子在无关坐标上使用同一个尺度数组；不同粒子之间的无关总分仍同分布。

定义

$$
L_a=\log\sigma(aZ),\quad \mu(a)=EL_a,\quad v(a)=\operatorname{Var}L_a,
\quad m_n=\sum_{i=1}^n\mu(a_{ni}),\quad
s_n^2=\sum_{i=1}^n v(a_{ni}),\qquad Z\sim N(0,1).
\tag{H.3}
$$

因为 $L_a=aZ/2-\log[2\cosh(aZ/2)]$，奇偶性使两项协方差为零，故 $v(a)\ge a^2/4$。又 $|L_a|\le\log2+a_+|Z|$，所以存在只依赖 $a_-,a_+$ 的常数 $0<c_v<C_v<\infty$，使 $c_v n\le s_n^2\le C_v n$。

记 $F_{n,K}^{\theta,\boldsymbol a}(\rho)$ 为完整观测的负log先验粒子均值似然对观测及K个粒子取期望。令

$$
r(\rho)=E_\rho[\tanh(X_1/2)\tanh(X_2/2)],\qquad
I_K=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{K-2}\,dz,\qquad
J_K=K(K-1)I_K=E\max_{k\le K}G_k,
\tag{H.4}
$$

其中 $G_k$ 独立标准正态，K为整数且至少为二。

**定理H.1（有界异质数组的一致相关矩之和方向）。** 有

$$
\sup_{\boldsymbol a\in[a_-,a_+]^n}
\sup_{2\le K\le C_0n^p}
\sup_{\rho\in I}
\left|
\frac{s_n}{J_K}\frac{d}{d\rho}F_{n,K}^{\theta,\boldsymbol a}(\rho)
+\frac{r(\theta)+r(\rho)}4
\right|\longrightarrow0.
\tag{H.5}
$$

K的上确界仅取整数。该结论是归一化梯度的一致绝对误差；不在 $r(\theta)+r(\rho)=0$ 处声称相对梯度误差。全部尺度可随n变化，但必须留在同一个固定正紧区间内。这里只证明总体目标及其近优集合，不附带有限N/B、SGD或当前数值预算的保证。

无关观测始终为独立公平位，因此积分观测律仍为

$$
Q_\rho^{(n)}(y)=2^{-n}\frac{1+y_1y_2r(\rho)}4.
\tag{H.6}
$$

故改变已知尺度数组不改变精确相关学习任务；本定理说明固定正紧区间内的异质性不改变领先的错误方向，只改变 $J_K/s_n$ 的消退尺度。后文给出实际总体极小点与近优集合的不同量词。

## 2. 单坐标倾斜族的一致正则性

对 $u<0$，$L_a$ 的密度是

$$
f_a(u)=\frac{\varphi(z_a(u))}{a(1-e^u)},\qquad
z_a(u)=\frac{u-\log(1-e^u)}a;
\qquad f_a(u)=0\quad(u\ge0).
\tag{H.7}
$$

这些密度在整个 $a\in[a_-,a_+]$ 上构成一致Schwartz族。具体地：当 $u\le-1$，$z_a(u)=u/a+O(e^u)$，任意固定阶密度导数被一个统一的多项式乘Gaussian尾控制；当 $u\uparrow0$，令 $t=-\log(-u)$，第j阶导数至多增加 $e^{(j+1)t}$ 与t的固定次幂，而Gaussian部分由 $\exp[-t^2/(2a_+^2)+O(t)]$ 控制。它比任意这些幂更快趋零，因此零延拓的各阶导数在边界同为零。各Schwartz半范数均可用与a无关的常数控制。

取固定 $0<\lambda_0<1/4$，定义中心化单坐标累积量

$$
\kappa_a(\lambda)=\log Ee^{\lambda(L_a-\mu(a))},
\qquad |\lambda|\le\lambda_0.
\tag{H.8}
$$

$|L_a|\le\log2+a_+|Z|$ 保证任意固定阶指数加权矩在 $(a,\lambda)$ 的紧集上一致有限。倾斜归一化因子连续且严格正，中心化时还由Jensen不小于一。由控制收敛，$\kappa_a$ 的各固定阶导数联合连续且一致有界。倾斜分布始终具有非退化连续密度，故其方差在该紧集上有统一正下界：

$$
0<c_2\le\kappa_a''(\lambda)\le C_2,
\qquad \sup_{a,|\lambda|\le\lambda_0}|\kappa_a^{(j)}(\lambda)|\le C_j
\quad\text{对每个固定 }j.
\tag{H.9}
$$

以 $e^{\lambda(L_a-\mu(a))}$ 倾斜后，再减去倾斜均值。其密度的一致Schwartz界仍成立：远左尾只多一个有界参数的指数线性因子，近零尾不变，中心化平移在参数紧集上一致有界。

令 $\psi_{a,\lambda}(u)$ 为这个中心化倾斜变量的特征函数。对任意固定 $\varepsilon>0$，

$$
\sup_{a,|\lambda|\le\lambda_0,|u|\ge\varepsilon}
|\psi_{a,\lambda}(u)|\le q_\varepsilon<1,
\qquad
\sup_{a,|\lambda|\le\lambda_0}\int_{\mathbb R}
|\psi_{a,\lambda}(u)|^2du\le C.
\tag{H.10}
$$

第一式不能仅从逐分布非格点性直接断言。这里先在 $\varepsilon\le|u|\le T$ 上使用联合连续性、参数紧性和连续非退化密度给出的严格模小于一；再用一致密度导数L1界分部积分，得 $|\psi_{a,\lambda}(u)|\le C/|u|$，选大T覆盖远频率。第二式可由这一统一衰减界直接积分，或由Fourier等距性得到。

## 3. 三角阵的倾斜局部密度与相对上尾

条件于任意固定完整观测，无关标签符号可在粒子集期望中逐坐标吸收到Gaussian先验中。因此每个粒子的无关总分在减去公共常数m_n后为

$$
S_{n,\boldsymbol a}=\sum_{i=1}^n[L_{a_{ni},i}-\mu(a_{ni})].
\tag{H.11}
$$

和内各项独立但一般不同分布；不同粒子使用相同数组，所以各粒子的S仍iid。令 $G_c$ 为独立标准Gumbel减去其均值。其对数矩母函数记为 $g_0(\lambda)=\log Ee^{\lambda G_c}$，在 $|\lambda|\le\lambda_0$ 上所有所需矩有限，并有 $g_0(0)=g_0'(0)=0$。记

$$
\mathcal K_{n,\boldsymbol a}(\lambda)
=\sum_{i=1}^n\kappa_{a_{ni}}(\lambda)+g_0(\lambda),
\quad \tau_{n,\boldsymbol a,\lambda}^2=\mathcal K_{n,\boldsymbol a}''(\lambda),
\quad W_{n,\boldsymbol a}=\frac{S_{n,\boldsymbol a}+G_c}{s_n}.
\tag{H.12}
$$

倾斜保持各坐标独立，且由式(H.9)，$cn\le\tau^2\le Cn$，三阶累积量总和为O(n)，全部对数组及倾斜参数一致。将倾斜和的密度记为 $f_{n,\boldsymbol a,\lambda}$。

**引理H.2（倾斜中心局部界）。** 对全部允许数组与 $|\lambda|\le\lambda_0$，

$$
f_{n,\boldsymbol a,\lambda}(\mathcal K_{n,\boldsymbol a}'(\lambda))
=\frac{1+O(n^{-1/2})}{\sqrt{2\pi}\,\tau_{n,\boldsymbol a,\lambda}},
\qquad
\sup_x s_n f_{n,\boldsymbol a,\lambda}(x)\le C.
\tag{H.13}
$$

**证明。** 中心化并按倾斜标准差τ标准化，其特征函数为

$$
\Psi_{n,\boldsymbol a,\lambda}(t)
=\chi_\lambda(t/\tau)\prod_{i=1}^n\psi_{a_{ni},\lambda}(t/\tau),
\tag{H.14}
$$

其中 $\chi_\lambda$ 是中心化倾斜Gumbel的特征函数，模不超过一。各因子的三阶绝对矩一致有界，所以在统一的零邻域可选连续对数分支；由方差相加，对固定充分小δ，

$$
\log\Psi_{n,\boldsymbol a,\lambda}(t)
=-t^2/2+O(|t|^3/\sqrt n),\qquad |t|\le\delta\sqrt n.
\tag{H.15}
$$

取δ使余项实部不超过 $t^2/4$，实际对数与 $-t^2/2$ 之间插值的实部至多为 $-ct^2$。由指数函数的积分差公式，该区间上 $\Psi$ 与 $e^{-t^2/2}$ 的L1距离至多 $C n^{-1/2}\int |t|^3e^{-ct^2}dt=O(n^{-1/2})$。

高频部分是非iid替换的关键。令 $u=t/\tau$；当 $|t|>\delta\sqrt n$ 时，$|u|$具有统一正下界ε。式(H.10)使任意n−2个因子的乘积不超过 $q_\varepsilon^{n-2}$，其余两个不同的因子用Cauchy–Schwarz积分：

$$
\begin{aligned}
\int_{|t|>\delta\sqrt n}|\Psi_{n,\boldsymbol a,\lambda}(t)|dt
&\le \tau q_\varepsilon^{n-2}
\int_{\mathbb R}|\psi_{a_{n1},\lambda}(u)\psi_{a_{n2},\lambda}(u)|du\\
&\le C\sqrt n\,q_\varepsilon^{n-2}.
\end{aligned}
\tag{H.16}
$$

此处n充分大，因而可取两个因子。不能在非同分布数组中写成单个 $|\psi|^n$，也不能在无限频率区间只积分常数 $q^n$。再加正态尾积分可得全轴L1差为 $O(n^{-1/2})$。Fourier反演给标准化密度与φ的一致绝对差 $O(n^{-1/2})$，在零点得到第一式；同一绝对积分界及 $s_n/\tau\asymp1$ 给第二式。□

将 $W_{n,\boldsymbol a}$ 的密度与CDF记为 $q_{n,\boldsymbol a},H_{n,\boldsymbol a}$。

**引理H.3（相对密度／上尾）。** 对每个固定M>0，一致于所有允许数组，

$$
\sup_{|z|\le M\sqrt{\log n}}
\left|\frac{q_{n,\boldsymbol a}(z)}{\varphi(z)}-1\right|
=O_M\!\left(\frac{1+(\log n)^{3/2}}{\sqrt n}\right),
\tag{H.17}
$$
$$
\sup_{-1\le z\le M\sqrt{\log n}}
\left|\frac{1-H_{n,\boldsymbol a}(z)}{1-\Phi(z)}-1\right|=o(1),
\quad\sup_z q_{n,\boldsymbol a}(z)\le C,
\tag{H.18}
$$
$$
q_{n,\boldsymbol a}(z)\le C_M n^{-M^2/2}
\exp\{-M\sqrt{\log n}(z-M\sqrt{\log n})+o(1)\},
\quad z\ge M\sqrt{\log n}.
\tag{H.19}
$$

**证明。** 以下省略数组下标。对 $x=s_nz$、$|z|\le M\sqrt{\log n}$，正方差下界保证方程 $\mathcal K_n'(\lambda)=x$ 在统一的小倾斜邻域内有唯一解。这里 $\mathcal K_n'(0)=0$、$\mathcal K_n''(0)=s_n^2+O(1)$，并有统一 $|\mathcal K_n'''|\le Cn$。先用单调性得 $|\lambda|\le C|z|/\sqrt n$，再Taylor展开可得

$$
\lambda=\frac z{s_n}+O_M((1+z^2)/n),\qquad
\lambda x-\mathcal K_n(\lambda)
=z^2/2+O_M((1+|z|^3)/\sqrt n),\qquad
\frac{s_n}{\sqrt{\mathcal K_n''(\lambda)}}
=1+O_M((1+|z|)/\sqrt n).
\tag{H.20}
$$

均匀性来自式(H.9)，并不要求平均方差 $s_n^2/n$ 有极限。由精确去倾斜身份 $q_n(z)=s_ne^{\mathcal K_n(\lambda)-\lambda x}f_{n,\lambda}(x)$，式(H.13)与(H.20)给相对密度式(H.17)。

对远尾固定取 $h_n=M\sqrt{\log n}/s_n$，则 $\mathcal K_n(h_n)=M^2\log n/2+o(1)$，误差仍对数组一致。式(H.13)给所有实z上的界 $q_n(z)\le C\exp\{\mathcal K_n(h_n)-h_ns_nz\}$，从而得到式(H.19)。由于这个未改写的固定倾斜界对所有z成立，它也允许在阈值附近作小幅平移。

最后取固定 $M'>M$，将式(H.17)积分到 $M'\sqrt{\log n}$，再用式(H.19)控制余尾。分母的最小量级由Mills下界给为 $n^{-M^2/2}/\sqrt{\log n}$，余尾与它的比为 $O(n^{-(M'^2-M^2)/2})$；密度的相对误差本身也一致趋零。由此得到式(H.18)的相对上尾。零倾斜下的式(H.13)给全局密度界。□

## 4. 对多项式K一致的粒子对核

给定active分数 $A_1,\ldots,A_K$，softmax权重使用 $S_{nk}+A_k$，其中S在粒子k之间iid。Gumbel最大值身份及关于另一个active分数的导数给出精确式

$$
E_S[\pi_i\pi_j]
=\frac1{s_n}\int_{\mathbb R}q_{n,\boldsymbol a}(z-\delta_i)
q_{n,\boldsymbol a}(z-\delta_j)
\prod_{l\ne i,j}H_{n,\boldsymbol a}(z-\delta_l)dz,
\qquad\delta_l=A_l/s_n,\quad i\ne j.
\tag{H.21}
$$

身份只需要不同粒子的总分同分布，不需要粒子内部的n个坐标同分布。微分交换由有界softmax导数、密度上界乘可积密度保证；Gumbel已卷入$q,H$，无需引入不合法的随机最大值二阶导数。

**引理H.4（相对粒子对质量）。** 固定C_1>0。对允许数组、整数 $2\le K\le C_0n^p$、$\max_l|A_l|\le C_1\sqrt{\log n}$ 及 $i\ne j$，一致有

$$
\frac{s_nE_S[\pi_i\pi_j]}{I_K}\longrightarrow1.
\tag{H.22}
$$

**证明。** 选固定 $M^2>4p+4$。由 $s_n\asymp\sqrt n$，所有平移为 $O(\sqrt{\log n/n})$。在 $0\le z\le M\sqrt{\log n}$ 上，式(H.17)–(H.18)与正态平移公式给

$$
q_n(z-\delta)=\varphi(z)(1+o(1)),\qquad
1-H_n(z-\delta)=1-\Phi(z) (historical reference not distributed)),
\tag{H.23}
$$

全部误差一致。密度对数比为 $z\delta-\delta^2/2$；正态上尾对数比由 $C|\delta|(1+z+|\delta|)$ 控制。二者均为 $O(\log n/\sqrt n)$。对稍微超出所写区间的平移，使用略大固定M的引理即可。

对 $u=1-\Phi(z)\le1/2$，$\log(1-(1+O(\varepsilon))u)/\log(1-u)=1+O(\varepsilon)$ 一致成立。因此存在对全部数组适用的 $\varepsilon_n\to0$，使 $\Phi(z)^{1+\varepsilon_n}\le H_n(z-\delta)\le\Phi(z)^{1-\varepsilon_n}$。式(H.21)的中间区间被 $1+o(1)$ 倍的

$$
\int_0^{M\sqrt{\log n}}\varphi(z)^2
\Phi(z)^{(K-2)(1\pm\varepsilon_n)}dz
\tag{H.24}
$$

上下夹住。这保留了约1/K的相对尾精度，没有累乘K个逐点相对误差。

对实x>1也以式(H.4)的积分定义 $I_x,J_x=x(x-1)I_x$。分部积分给 $J_x$ 为CDF $\Phi^x$ 的均值；Mills界及最大值尾积分给 $J_x\sim\sqrt{2\log x}$。令 $x=2+(K-2)(1\pm\varepsilon_n)$，则有界K用连续性，发散K用 $x/K\to1$ 与该渐近，得到 $I_x/I_K\to1$。反证取子序列可使这一比较对所有允许K一致。

正远尾的两密度乘积积分由式(H.19)控制为 $Cn^{-M^2}/\sqrt{\log n}$；小平移只增加 $e^{O(\log n/\sqrt n)}$。因为 $I_K=J_K/[K(K-1)]$、$J_K\ge J_2>0$、$K^2\le Cn^{2p}$，它相对 $I_K$ 消失，正态比较积分的同侧余尾也如此。

若K发散，负半轴上所有平移CDF最终不超过固定 $q<1$，因此该部分至多为 $Cq^{K-2}$。除以 $I_K$ 后仍趋零，即使K任意慢地发散。若K有界，则保留负半轴：紧区间上密度和CDF一致趋于正态，区间外用全局密度界乘尾概率。$EW_n=0$、$EW_n^2=1+\operatorname{Var}(G_c)/s_n^2=1+O(1/n)$，故Chebyshev能对所有数组一致压小该尾。两类子序列给式(H.22)。□

## 5. Price导数与active截断

active分数及导数标记为

$$
A_k=\log\sigma(Y_1X_{k1})+\log\sigma(Y_2X_{k2}),\qquad
T_k=Y_1Y_2\sigma(-Y_1X_{k1})\sigma(-Y_2X_{k2}),\quad |T_k|\le1.
\tag{H.25}
$$

两个对称协方差条目抵消通常的1/2，非对角Price身份给

$$
\frac{dF_{n,K}^{\theta,\boldsymbol a}}{d\rho}
=-E\sum_k\pi_k(1-\pi_k)T_k
=-E\sum_{k<l}\pi_k\pi_l(T_k+T_l).
\tag{H.26}
$$

对每个固定n/K，完整负log似然由单粒子的Gaussian线性包络及logK控制，混合导数有界，因此身份与微分交换合法。无关尺度不参与对active相关的微分。

令 $\Lambda=1+R$。有 $|A_k|\le2\log2+\sqrt2\|X_k\|$ 及 $\|X_k\|\le\sqrt\Lambda\|Z_{k,act}\|$。二维下 $Ee^{\|Z\|^2/4}=2$，故阈值

$$
T_n^{act}=2\log2+2\sqrt{2\Lambda(p+3)\log n}
\quad\Longrightarrow\quad
P\{\max_k|A_k|>T_n^{act}\}\le2C_0n^{-3}
\tag{H.27}
$$

对候选、数组和观测符号一致。坏事件对式(H.26)原始评分导数的贡献至多为该概率，因为 $\sum_k\pi_k(1-\pi_k)\le1$；再乘 $s_n/J_K\le C\sqrt n/J_2$ 后仍消失。

在好事件上，条件于全部active变量应用引理H.4。相对粒子对误差统一为o(1)，以 $|T_k+T_l|\le2$ 控制并求和，误差为 $o(J_K/s_n)$，而非把一个未缩放的o(1)乘K²。移除好事件后，主项为

$$
\frac{dF_{n,K}^{\theta,\boldsymbol a}}{d\rho}
=-\frac{I_K}{s_n}(K-1)\sum_k ET_k+o(J_K/s_n).
\tag{H.28}
$$

候选粒子与观测独立。记 $t_i=\tanh(X_i/2)$，则

$$
T_k=\frac{Y_1Y_2-Y_1t_2-Y_2t_1+t_1t_2}{4},
\qquad ET_k=\frac{r(\theta)+r(\rho)}4.
\tag{H.29}
$$

代入式(H.28)及 $J_K=K(K-1)I_K$ 即证定理H.1。误差在全部允许数组、候选与K上一致，符号来自近并列粒子对，不是将softmax直接换成单个winner。

## 6. 实际总体端点与近优集合

令 $w_{n,K,\boldsymbol a}=J_K/s_n$、$H(\rho)=\int_0^\rho r(u)du$。式(H.5)从零积分给出统一C1极限

$$
\frac{F_{n,K}^{\theta,\boldsymbol a}(\rho)-F_{n,K}^{\theta,\boldsymbol a}(0)}{w_{n,K,\boldsymbol a}}
\longrightarrow V_\theta(\rho):=-\frac14[r(\theta)\rho+H(\rho)].
\tag{H.30}
$$

r为奇函数且 $r'>0$，所以H偶且严格凸，$V_\theta$严格凹。θ>0时唯一极小点为R；θ<0时为−R；θ=0时极小点集合为两端点。

**推论H.5（实际总体极小点）。** 对每个固定θ≠0，存在n_0，使n≥n_0时，对全部允许数组与整数K，原总体目标的唯一全局极小点恰为 $\operatorname{sign}(\theta)R$。对固定θ=0，充分大n时极小点恰为 $\{-R,R\}$。

证明先用式(H.30)的值间隙排除远离极小端点的闭集，再用式(H.5)在小端点邻域内的严格导数符号排除邻域内部。θ=0时分别排除中部与两个邻域内部，并由精确符号对称性 $F^{0,\boldsymbol a}(\rho)=F^{0,\boldsymbol a}(-\rho)$ 得两端相等。这是实际总体精确端点结论，不是仅从极小点收敛直接推出有限n贴边；n_0不被声明可计算或对接近零的非零θ统一。

对任意允许数组/K序列，定义整个ε_n近优集合。若

$$
\varepsilon_n=o(w_n)=o(J_{K_n}/s_n),
\tag{H.31}
$$

则该集合到上述极小点集合的最大距离趋零。若对所有数组/K给一族误差并要求其 $\varepsilon/w$ 的上确界趋零，则结论也对这些近优集合一致。由于 $s_n\asymp\sqrt n$，多项式K下 $w_n\to0$，真实参数自身的raw regret也为O(w_n)=o(1)。因此任意o(1)近优容差不足以得到这一结论。

积分模型的完整联合风险恒为

$$
D(Q_\theta^{(n)}\Vert Q_\rho^{(n)})
=\operatorname{kl}\!\left(\frac{1+r(\theta)}2\middle\Vert\frac{1+r(\rho)}2\right).
\tag{H.32}
$$

对内点教师，近优集合的风险一致趋于错误端点的严格正值；零教师的两端点风险相同。θ=±R时同号端点就是真值，不在正风险失败声明中。实际有限数据/粒子评分是否选到端点，需要另外的抽样与优化误差控制，本文件没有据此认证任何现实训练。

## 7. 适用边界与新增账目

该推广允许任意确定性有界异质数组，不要求方差均值或尺度比例收敛。它没有覆盖尺度趋零、单个尺度发散、粒子之间使用不同尺度数组、无关项彼此相关、任意噪声分布、均值自由学习或未知active结构。固定正紧区间是本证明的充分条件，未证明它是必要条件。

尤其，方差匹配的“一项强尺度＋其余趋小尺度”不满足本定理的统一紧区间条件；若该分支恢复精确方向，它提供普适性边界，而非与本定理矛盾。仅知道总方差 $s_n^2$ 发散或量级相同，不能替代上面的相对密度与尾条件。

新增账目：0科学RNG、0数值模型积分、0拟合、0训练。本文包含新的三角阵解析证明；有限样本学习、数值资格和另一集中噪声分支由相应独立材料另行处理。
