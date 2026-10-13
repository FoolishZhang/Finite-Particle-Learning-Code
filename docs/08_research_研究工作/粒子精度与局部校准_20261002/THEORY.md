# 粒子精度与局部校准：计算误差、固定点梯度与单系数估计

日期：2026-10-02。本文件提供本包计算和诊断的数学依据，不包含新随机样本、数值积分或训练结果。模型固定为 $a=1$、$\rho\in I=[-3/4,3/4]$，只学习已知两坐标相关块。完整观测行、完整共享 prior bank 及 nuisance 符号均保留。

下文三部分分别控制候选评分的插值误差、固定数据固定参数处的训练粒子波动、以及局部二次系数的估计误差。它们不是优化后 KL 的因果分解，也不改变旧 [完整 bank 计算身份与误差接口](../有限校准与真实样本训练_20261002/COMPUTATION_THEORY.md)。

## 1. 保留完整观测的 Chebyshev 加速及其确定性误差

### 1.1 插值对象与移除常数

固定实际数据 $\mathcal D=(Y_j)_{j=1}^N$ 和第 $b$ 个 bank 的 $K$ 条向量。记

$$
X_{bk2}(\rho)=\rho Z_{bk1}+\sqrt{1-\rho^2}Z_{bk2},
\qquad
\ell_{jbk}=\log\sigma(Y_{j1}Z_{bk1})+
\sum_{i=3}^{n+2}\log\sigma(Y_{ji}Z_{bki}),
\tag{S.1}
$$

并令 $\omega_{jbk}=e^{\ell_{jbk}}/\sum_l e^{\ell_{jbl}}$。这些权重严格正、和为 1，且与候选 $\rho$ 无关。对 $s\in\{-1,1\}$ 定义

$$
p_{jb,s}(\rho)=\sum_k\omega_{jbk}\sigma(sX_{bk2}(\rho)),
\qquad f_{jb,s}(\rho)=-\log p_{jb,s}(\rho).
\tag{S.2}
$$

原始完整负 log 粒子似然为

$$
-\log\left\{K^{-1}\sum_k\prod_i\sigma(Y_{ji}X_{bki}(\rho))\right\}
=-\log\left\{K^{-1}\sum_k e^{\ell_{jbk}}\right\}
+f_{jb,Y_{j2}}(\rho).
\tag{S.3}
$$

第一项不随 $\rho$ 变化，故可移除而不改变评分差、最小点或同容差近优集合。**下面的统一复数模界只用于移除该项后的 raw 损失**；含 $n$ 个 nuisance 的原始常数不能放进 $M=21$。权重仍由每一条实际完整观测计算，不能用 canonical nuisance 行代替它。

对 marg，同一身份使用 $\ell^{marg}_{jbk}=\log\sigma(Y_{j1}Z_{bk1})$，另有已移除的 $n\log2$。其第一位损失常数

$$
c^{marg}_{jb}=-\log\left\{K^{-1}\sum_k\sigma(Y_{j1}Z_{bk1})\right\}
\tag{S.4}
$$

可以保留；它是下面 $M=21$ 中最多 8 的附加项。raw 与 marg 的权重表不同。

### 1.2 一次可检查的 active 输入条件

假设拟采用插值的各 bank 满足

$$
Z_{bk1}^2+Z_{bk2}^2\le49\quad\text{对所有 }b,k.
\tag{S.5}
$$

这是对已抽取输入的计算分支条件，不是截断 Gaussian 模型。若某个规定执行单元不满足，应在该单元回退到原 801 点直接评价，保留原输入和预算；不得重抽或删去失败 bank。若需以严格实算术方式使用本命题，应按已保存 float64 输入的精确有理值或向外取整上界核验 (S.5)。

取 Bernstein 椭圆参数 $q=28/25$，将标准 $[-1,1]$ 椭圆缩放到 $I$。其半轴为

$$
a_e=\frac34\frac{q+q^{-1}}2=\frac{4227}{5600},
\qquad b_e=\frac34\frac{q-q^{-1}}2=\frac{477}{5600}.
\tag{S.6}
$$

闭椭圆内任意 $z=x+iy$ 有 $|x|\le a_e<1$、$|y|\le b_e$。由于 $\operatorname{Re}(1-z^2)=1-x^2+y^2>0$，可以在整个椭圆的邻域一致采用主平方根。令 $\sqrt{1-z^2}=u+iv$，则 $u>0$ 且

$$
u^2-v^2=1-x^2+y^2,\quad uv=-xy,\quad
u\ge\sqrt{1-x^2},\quad
|v|\le\frac{|xy|}{\sqrt{1-x^2}}.
\tag{S.7}
$$

同时 $|1-z^2|\le1+|z|^2$ 给 $u^2\le1+y^2$。对 (S.1) 的该解析延拓应用 Cauchy–Schwarz，有

$$
\begin{aligned}
|\operatorname{Im}X_{bk2}(z)|
&\le7\sqrt{y^2+v^2}
\le\frac{7b_e}{\sqrt{1-a_e^2}}<1,\\
|\operatorname{Re}X_{bk2}(z)|
&\le7\sqrt{x^2+u^2}
\le7\sqrt{1+a_e^2+b_e^2}<\frac{28}3.
\end{aligned}
\tag{S.8}
$$

最后两项严格不等式可全用有理数核验：它们的左侧平方分别为 $11148921/13492471<1$ 和 $24727529/320000<784/9$。因此平方根分支点 $\pm1$ 和后续 logistic 极点均未进入该区域。

### 1.3 混合概率不会穿过对数分支

对 $w=u+iv$ 且 $|v|\le1$，$\sigma(w)=(1+e^{-u}e^{-iv})^{-1}$。因 $\cos v>0$，其分母实部大于 1，从而 $|\sigma(w)|\le1$。分母是正实数 1 与辐角 $-v$ 的向量之和，故 $|\arg\sigma(w)|\le|v|\le1$。另一方面

$$
|\sigma(w)|\ge\frac1{1+e^{-u}}\ge\frac12e^{-|u|},
\qquad
\operatorname{Re}\sigma(w)
\ge|\sigma(w)|\cos1\ge\frac14e^{-|u|}.
\tag{S.9}
$$

正权混合 (S.2) 因而位于同一个闭扇形 $|\arg p|\le1$ 内，并满足

$$
|p_{jb,s}(z)|\le1,\qquad
\operatorname{Re}p_{jb,s}(z)\ge\tfrac14e^{-28/3}>0.
\tag{S.10}
$$

所以其主对数在椭圆邻域解析，没有零点或负实轴分支问题，而且

$$
|{-\operatorname{Log}p_{jb,s}(z)}|
\le-\log|p_{jb,s}(z)|+|\arg p_{jb,s}(z)|
\le\frac{28}3+\log4+1<13.
\tag{S.11}
$$

由 (S.5)，$c^{marg}_{jb}\le\log(1+e^7)\le7+\log2<8$。因此 raw 的已去常数损失以 13 为模界，marg 的去掉 $n\log2$ 后损失以 **$M=21$** 为共同模界。任何观测平均、教师样本符号选择和 bank 平均仍保持解析性与此模界；平均不要求样本独立。

### 1.4 193 点的统一插值误差

令 $f$ 为上述任一种 bank 损失或其平均，令 $P_m f$ 为在

$$
\rho_j=\tfrac34\cos(j\pi/m),\quad 0\le j\le m,
\tag{S.12}
$$

上的次数不超过 $m$ 的精确 Lobatto 插值多项式。则

$$
\|f-P_mf\|_{\infty,I}
\le\frac{4M q^{-m}}{q-1}.
\tag{S.13}
$$

这是经典 Chebyshev 解析函数插值界，而非本研究的新近似原理；可核对 Trefethen《Approximation Theory and Approximation Practice》定理 8.1–8.2 的[作者配套章节源码](https://raw.githubusercontent.com/chebfun/ATAP/master/chap8.m)及[作者书籍入口](https://people.maths.ox.ac.uk/trefethen/ATAP/)。下面给出所需证明，说明插值而非仅级数截断为何带系数 4。

**证明。** 令 $\widetilde f(t)=f(3t/4)$，作 Joukowsky 代换 $t=(w+w^{-1})/2$。所得函数在 $q^{-1}<|w|<q$ 解析、关于 $w\leftrightarrow w^{-1}$ 对称，模不超过 $M$。Laurent 系数的 Cauchy 界给其 Chebyshev 展开 $\widetilde f(t)=\sum_{k\ge0}a_kT_k(t)$ 满足 $|a_k|\le2M q^{-k}$，$k\ge1$。在 $m+1$ 个 Lobatto 点，$T_k$ 与某个 $T_{\nu_m(k)}$ 取相同值，其中 $0\le\nu_m(k)\le m$，由 $k$ 对 $2m$ 取余后反射得到；$k\le m$ 时 $\nu_m(k)=k$。级数绝对收敛，逐项插值有效。因此

$$
\|\widetilde f-P_m\widetilde f\|_\infty
\le\sum_{k>m}|a_k|\|T_k-T_{\nu_m(k)}\|_\infty
\le4M\sum_{k=m+1}^\infty q^{-k}
=\frac{4M q^{-m}}{q-1}.
\tag{S.14}
$$

这也是缩放区间上的界。□

取 $m=192,M=21$，所需 expensive bank 评分节点为 193 个，且有精确有理上界

$$
\epsilon_{Cheb}=700(25/28)^{192}
=2.484492484391\ldots\times10^{-7}
<2.49\times10^{-7}.
\tag{S.15}
$$

此处减少的是 expensive 节点评分数，不是 Gaussian 输入、bank 数或模型维数；构建权重和插值也有开销，不能仅凭 $801/193$ 声称总运行时间加速倍数。

若实际保存中心化曲线 $f(\rho)-f(0)$，不应直接宣称它本身也受同一个复数模界控制。正确理由是插值保持常数：$P_m(f-f(0))=P_mf-f(0)$，故其**插值误差**与 $f$ 的完全一致。先分别近似 raw、marg 再取两者差时，单个评分差的误差界则可能为 $2\epsilon_{Cheb}$。

### 1.5 仍在 801 个真实候选上选点

将插值多项式评价在原 801 点 master 网格 $\mathcal G$ 上，并在这些点增加实际部署的惩罚 $p(\rho)$。有限校准样条、二次惩罚及其它已冻结惩罚直接使用其存储函数，**不作 Chebyshev 插值**。记真算术完整经验目标 $F=f+p$，近似目标 $\widetilde F=P_mf+p$。若返回点在计算网格上有误差 $\epsilon_{alg}$，且实现相对于理想插值及惩罚评价的误差不超过 $\zeta$，则

$$
F(\widehat\rho)-\min_{\rho\in I}F(\rho)
\le \frac{D h_{\mathcal G}^2}{8}
+2\epsilon_{Cheb}+2\zeta+\epsilon_{alg}.
\tag{S.16}
$$

这里 $D$ 控制**完整目标**在每段的二阶导数；$h_{\mathcal G}$ 是实际候选值的最大间距上界。master 网格须包含端点和所有惩罚样条结点，以便逐段应用曲率界。$D$ 包含惩罚的曲率项，按旧计算理论对已保存多项式作向外界；不能只对 raw 求 $D$。证明先以两次统一函数值误差把近似网格选点转为真网格 regret，再加逐段线性插值误差 $D h_{\mathcal G}^2/8$。此证明不需要目标凸。

节点位置、sigmoid/log、矩阵乘法、DCT 系数、Clenshaw 求值及累加的浮点误差均属于 $\zeta$。高精度抽查、节点回代、独立点评价与实际残差可给数值资格；没有严格舍入控制时，不应把 (S.16) 中的已知解析项称为完整的严格连续全局最优证书。函数值插值界也**不自动给出导数误差界**；下节的路径梯度须直接评价其解析表达。

## 2. 固定数据 MLE 处的精确梯度分解

固定完整数据 $\mathcal D$，令 $t_N=N^{-1}\sum_jY_{j1}Y_{j2}$，并令

$$
q_N=r^{-1}\!\left(\operatorname{clip}\{t_N,-r(R),r(R)\}\right),
\qquad R=3/4,
\tag{S.17}
$$

为相同数据的精确连续相关块 MLE。它只依赖数据、不依赖待评价的训练 bank。若实现采用参考表反演得到近似点，下述分解在该固定点仍然成立，但不能把该点无条件称为精确 MLE，或自动使用内点零梯度身份。

令 $H_b(q)$ 为第 $b$ 个完整共享 bank 的**实际经验 raw 目标路径导数**，每个 $H_b$ 先对所有实际观测行平均。直接计算时

$$
\begin{aligned}
X'_{bk2}(q)&=Z_{bk1}-\frac{q}{\sqrt{1-q^2}}Z_{bk2},\\
H_b(q)&=-\frac1N\sum_j
\frac{\sum_k\omega_{jbk}\,Y_{j2}\,X'_{bk2}(q)
\,\sigma(Y_{j2}X_{bk2}(q))\sigma(-Y_{j2}X_{bk2}(q))}
{\sum_k\omega_{jbk}\sigma(Y_{j2}X_{bk2}(q))}.
\end{aligned}
\tag{S.18}
$$

这里同一 bank 内的各观测项相关，不能把 $NB$ 项当作独立。条件于数据，不同 bank 是 iid；$|H_b(q)|\le M_b/\sqrt{1-R^2}$，其中 $M_b=\max_k\sqrt{Z_{bk1}^2+Z_{bk2}^2}$，故有有限二阶矩。定义

$$
\mu_{\mathcal D}(q)=E[H_b(q)\mid\mathcal D],
\qquad v_{\mathcal D}(q)=\operatorname{Var}[H_b(q)\mid\mathcal D].
\tag{S.19}
$$

进一步条件于独立估计后冻结的校准池 $\mathcal C$，其惩罚 $\widehat g$ 已成为确定函数。则

$$
E\!\left[\left\{B^{-1}\sum_bH_b(q)+\widehat g'(q)\right\}^2
\middle|\mathcal D,\mathcal C\right]
=\left\{\mu_{\mathcal D}(q)+\widehat g'(q)\right\}^2
+\frac{v_{\mathcal D}(q)}B.
\tag{S.20}
$$

**证明。** 条件于数据与冻结校准，平均导数的均值是括号中的均值项，方差是 $v_{\mathcal D}/B$；用二阶矩等于均值平方加方差。独立性只需要 bank 间成立。□

因此，对 raw、raw 加完整有限校准、raw 加局部二次惩罚，**在同一固定参数处，确定加性惩罚不会改变训练 bank 导数方差**。这不适用于改变粒子权重的 marg，也不表示这些方法在各自随机选点处具有相同梯度分布或相同风险。只有一次 bank 组的样本均值平方并不是均值平方的无偏估计：若 $S_H^2$ 是无偏样本方差，则 $\bar H^2-S_H^2/B$ 才是 $\mu_{\mathcal D}^2$ 的无偏估计，并可能为负。

设理想有限校准由条件 bank 期望

$$
\Delta(t,\rho)=\alpha(\rho)+t\beta(\rho),\quad
g'(\rho)=-\alpha'(\rho)-r(\rho)\beta'(\rho),\quad
b(\rho)=-\beta'(\rho)>0
\tag{S.21}
$$

定义。则旧有限校准身份给

$$
\mu_{\mathcal D}(q)+\widehat g'(q)
=b(q)[r(q)-t_N]+[\widehat g'(q)-g'(q)].
\tag{S.22}
$$

在内点精确 MLE 处 $r(q_N)=t_N$，第一项为零；在右端点且 $t_N>r(R)$ 时它严格为负，在左端点且 $t_N<-r(R)$ 时它严格为正。这是合法的约束最小点梯度，并非校准错误。即使惩罚已完全正确，边界 MLE 处的平方均值也可能非零。

如果还对独立校准池随机性平均，则有额外项

$$
E[G_B(q)^2\mid\mathcal D]
=\{\mu_{\mathcal D}(q)+E_{\mathcal C}\widehat g'(q)\}^2
+\operatorname{Var}_{\mathcal C}(\widehat g'(q))
+\frac{v_{\mathcal D}(q)}B.
\tag{S.23}
$$

同一校准池被全部训练流复用时，按训练流作的比较条件于该池，不能声称已包含第二项的重新校准变异。

这些分解只针对规定的固定参数和数据条件。不能将 $\widehat\rho$ 这种用同一 bank 选出的随机参数直接代入 iid 固定点公式，也不能把几项比例解释为最终 KL 的“校准原因占比”和“MC 原因占比”。相同数据下增加 $B$ 的配对实验可以诊断其影响，仍需报告实际选点及风险。

## 3. 用同 bank 中心差分估计局部二次系数

### 3.1 估计对象与二阶步长偏差

记 $G(\rho)=g'(\rho)$，这里 $G$ 是理想有限校准导数，不是先前的 raw 梯度。局部二次系数为

$$
c_0=\tfrac12G'(0)=\tfrac12g''(0),\qquad
c_h=\frac{G(h)-G(-h)}{4h}.
\tag{S.24}
$$

理想 $g$ 偶、$G$ 奇，但差分写法保留两边以便实现共同随机数。若 $G\in C^3([-h_0,h_0])$，且 $M_3=\sup_{|u|\le h_0}|G'''(u)|$，Taylor 余项给

$$
|c_h-c_0|\le M_3h^2/12,\qquad0<h\le h_0.
\tag{S.25}
$$

固定有限 $n,K$ 及严格内部相关区间下，Gaussian moments 和有界 logistic 导数保证所需可微性，但本式不声称已为当前模型给出实用的 $M_3$ 数值证书。预定 $h\in\{0.03,0.06,0.12\}$、主步长 $0.06$；其它两者用于报告分辨率和随机误差敏感性，不按测试教师风险更换主步长。

### 3.2 Price 校准抽样与共同随机数

对一个独立校准 bank $Z$，令 $D_{Z,y}(\rho)$ 是四种 active 模式 $y=(y_1,y_2)$ 的 Price 导数估计器，含规定的完整 nuisance 粒子权重：

$$
D_{Z,y}(\rho)=-y_1y_2\sum_k\pi_{k,y}(\rho)[1-\pi_{k,y}(\rho)]
\sigma(-y_1X_{k1})\sigma(-y_2X_{k2}(\rho)).
\tag{S.26}
$$

定义逐 bank 校准导数样本

$$
A_Z(\rho)=-\frac14\sum_y[1+r(\rho)y_1y_2]D_{Z,y}(\rho),
\qquad E A_Z(\rho)=G(\rho).
\tag{S.27}
$$

Price 样本和路径导数样本不逐 bank 相等，只有各自期望对应相同目标导数；本段只将 Price 表达用于校准。$r(\rho)$ 是已知模型函数，与训练教师无关；其数值近似误差必须另记。

用同一组 $B_c$ 个 bank 在 $h$ 与 $-h$ 评价，得到

$$
\widehat c_h=\frac1{B_c}\sum_{b=1}^{B_c}C_{h,Z_b},\qquad
C_{h,Z}=\frac{A_Z(h)-A_Z(-h)}{4h},\qquad
E\widehat c_h=c_h.
\tag{S.28}
$$

共同随机数的作用可直接量化。若 $A_Z$ 有绝对连续路径且

$$
\sup_{|u|\le h_0}E|A_Z'(u)|^2\le L_2^2<\infty,
\tag{S.29}
$$

则由积分表达和 Jensen 不等式，

$$
C_{h,Z}=\frac12\frac1{2h}\int_{-h}^hA_Z'(u)\,du,
\qquad E|C_{h,Z}|^2\le\frac{L_2^2}4,
\qquad\operatorname{Var}(\widehat c_h)\le\frac{L_2^2}{4B_c}.
\tag{S.30}
$$

因此不必出现独立两侧差分的 $1/h^2$ 方差上界。若进一步 $A_Z'(u)\to A_Z'(0)$ 在 $L^2$ 中成立，则 $C_{h,Z}\to A_Z'(0)/2$ 在 $L^2$ 中成立。这是共同随机数与路径平滑性的标准作用，不是新方差缩减原理；也不保证任意两个不光滑或错误耦合的估计器都有该性质。

本模型满足 (S.29)。令 $M_K=\max_k\sqrt{Z_{k1}^2+Z_{k2}^2}$、$L_Z=M_K/\sqrt{1-h_0^2}$。每个粒子的候选 log 权重导数绝对值不超过 $L_Z$，故 $\sum_k|\pi_k'|\le2L_Z$。式 (S.26) 里 $|D_{Z,y}|\le1$、$|D_{Z,y}'|\le(9/4)L_Z$；这里第一项来自 $\pi_k(1-\pi_k)$ 的求导，第二项来自 $|\sigma'|\le1/4$。又 $r'\le1/4$，于是

$$
|A_Z'(u)|\le\frac14+\frac94 L_Z,
\qquad
L_2^2\le\frac18+
\frac{81(2\log K+2)}{8(1-h_0^2)}.
\tag{S.31}
$$

最后一步使用二维 Gaussian 最大半径的尾界 $EM_K^2\le2\log K+2$。该包络不依赖 nuisance 维数，因为 nuisance 只进入归一化正权且不随候选相关变化。它通常很粗，提供存在性和小步长稳定性，不是本次估计达到某个经验 SE 的证明。

### 3.3 不能从局部系数直接推断全区间修复

即使 $\widehat c_h$ 的 MC 标准误差很小，$\widehat c_h\rho^2$ 与完整 $g(\rho)$ 仍可能有全区间形状差异。式 (S.25) 控制有限差分逼近 $c_0$；它不控制用一个系数代替整个校准函数时的风险。需要分别报告步长差异、独立校准变异、训练 bank 波动，以及固定所有规则后的选点和风险。若强行取正部或用风险选择步长，就定义了另一个程序，应另行声明，不能仍称 (S.28) 的无偏估计器。

在用参考 $\widehat r$ 代替 (S.27) 中精确 $r$ 时，若只知道逐点误差 $\delta_r$，保守系数误差为 $\delta_r/(2h)$，另外叠加导数评价的数值误差；除非已证明参考误差也有平滑耦合，不能把所有数值误差自动纳入 (S.30) 的无爆炸结论。

## 4. 本文件的验证边界

- 解析检查包括主平方根和主对数分支、统一模界、Lobatto alias 常数、网格 regret 传递、固定点条件二阶矩身份以及局部系数的 $L^2$ 差分界。
- 常数 (S.6)、(S.8)、(S.15) 已用精确有理数算术独立复核；小数仅用于展示。该复核零随机样本、零积分、零训练。
- 本文件没有给浮点误差 $\zeta$、校准曲线全域误差或当前有限预算最终风险的严格数值证书。独立重建和实验结果应在本包结果与审计文件中另外报告。
