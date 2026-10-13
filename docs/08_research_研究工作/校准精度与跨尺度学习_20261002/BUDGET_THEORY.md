# 连续有限校准的两种充分预算：训练 bank 与校准 bank

日期：2026-10-02。本补充在本包科学协议之后独立推导，不改变 PLAN、固定预算、统计门或已运行程序。零新 RNG、零数值积分、零训练。证明使用经典条件期望、Jensen、积分 Cauchy–Schwarz 与 Gaussian 指数矩；结论限于本已知两位相关块、固定发射、完整共享 bank 模型，不作为新的通用 Monte Carlo 理论或必要预算下界。

核心结果是：在已证明的高维方向适用域中，记消退目标尺度为 $w_n=J_{K_n}/\sqrt{nv}$。一个**理想连续校准程序**的充分路径是

$$
B_nw_n^2\longrightarrow\infty,\qquad
C_nw_n\longrightarrow\infty.
\tag{B.1}
$$

这里 B 是真实完整观测训练所用的独立 bank 数，C 是独立连续 Price 校准的 bank 数。固定 K 时分别要求 $B\gg n$、$C\gg\sqrt n$；增长 K 时分别为 $B\gg n/J_K^2$、$C\gg\sqrt n/J_K$。这些是两个指定程序的充分条件，不表示较小预算不可能成功，也不自动适用于当前固定101结点校准、固定801候选网格或 SGD。

## 1. 合同、误差与积分不等式

固定 $a=1$、$I=[-R,R]$、$0<R<1$，有限 $n\ge0,K\ge2$。给定任意完整数据集 $\mathcal D=(Y_j)_{j=1}^N$，训练 bank $Z_b$ 是独立标准 Gaussian 数组，彼此 iid 且独立于数据。每个 bank 被全部观测行和全部候选共用。记单行完整粒子负 log-likelihood 的路径导数为 $h_\rho(Y,Z)$，并定义

$$
H_{\mathcal D,Z}(\rho)=\frac1N\sum_jh_\rho(Y_j,Z),\qquad
e_B(\rho)=\widehat F_B(\rho)-E_Z[\widehat F_B(\rho)\mid\mathcal D].
\tag{B.2}
$$

参数无关的 raw 常数可按同一个规则移除；它不影响下述振幅。记 $\operatorname{osc}f=\sup_I f-\inf_I f$。对任意绝对连续 f，

$$
\operatorname{osc}f\le\int_{-R}^R|f'(u)|\,du,
\qquad
(\operatorname{osc}f)^2\le2R\int_{-R}^R|f'(u)|^2du.
\tag{B.3}
$$

因此不必先覆盖参数网格或分别控制各结点。不同参数处共享同一个 bank 完全允许，(B.3)不需要这些点独立。

## 2. 完整训练路径导数的两个二阶矩界

### 2.1 最大半径界：已有包络的直接推论

令 $M_K=\max_{k\le K}\sqrt{Z_{k1}^2+Z_{k2}^2}$。实际路径为 $X_{k2}(\rho)=\rho Z_{k1}+\sqrt{1-\rho^2}Z_{k2}$，故

$$
|h_\rho(Y,Z)|\le\frac{M_K}{\sqrt{1-R^2}},\qquad
|H_{\mathcal D,Z}(\rho)|\le\frac{M_K}{\sqrt{1-R^2}}.
\tag{B.4}
$$

二维标准 Gaussian 半径平方的尾给 $P(M_K^2>x)\le\min(1,Ke^{-x/2})$；积分可得 $EM_K^2\le2\log K+2$。这保证对每个固定完整数据集，损失与其路径导数的期望可交换微分，且导数期望在紧区间连续。训练目标本身可含很大的 nuisance 常数，但上述导数包络不含该常数。

### 2.2 对 nuisance 条件化可去掉 log K

进一步固定一条实际观察行 Y，条件于该 bank 的所有 nuisance prior 坐标。令

$$
a_k=\frac{\prod_{i=3}^{n+2}\sigma(Y_iZ_{ki})}
{\sum_l\prod_{i=3}^{n+2}\sigma(Y_iZ_{li})},\qquad
f_k=\sigma(Y_1Z_{k1})\sigma(Y_2X_{k2}(\rho)),
\quad S=\sum_ka_kf_k,
\quad\pi_k=\frac{a_kf_k}{S}.
\tag{B.5}
$$

n=0时约定空乘积为1。这里 $a_k>0$、和为1且独立于全部 active Gaussian 输入。**a只含 nuisance**；不能把包含第一位 active 似然的实现缓存权重直接用作该独立性论据。令 $T_k=X_{k2}'(\rho)$，则

$$
h_\rho=-\sum_k\pi_kY_2T_k\sigma(-Y_2X_{k2}),\qquad
h_\rho^2\le\sum_k\pi_kT_k^2
\le\left(\sum_ia_if_iT_i^2\right)\left(\sum_j\frac{a_j}{f_j}\right).
\tag{B.6}
$$

第一步是对正权 $\pi$ 的 Jensen；第二步来自 $1/S\le\sum_ja_j/f_j$。条件于 nuisance 后，不同粒子的 active 输入独立。将最后一项拆成对角与非对角，得到

$$
E_{active}h_\rho^2
\le\sum_i a_i^2ET_i^2
+\sum_{i\ne j}a_ia_jE[f_iT_i^2]E[1/f_j].
\tag{B.7}
$$

有 $ET_i^2=(1-\rho^2)^{-1}$、$0<f_i\le1$，并由 Gaussian 指数矩直接算得

$$
E\frac1{f_i}=1+2e^{1/2}+e^{1+Y_1Y_2\rho}
\le C_R:=1+2e^{1/2}+e^{1+R}.
\tag{B.8}
$$

所以 (B.7) 不超过 $C_R/(1-R^2)$。该界对条件 nuisance 权重的所有可能值均成立，再取 nuisance 期望仍成立。对实际数据行平均使用 Jensen，不需假设同 bank 的各观察评分独立，得

$$
E_Z[H_{\mathcal D,Z}(\rho)^2\mid\mathcal D]
\le V_{R,K}:=\frac{\min\{2\log K+2,C_R\}}{1-R^2}.
\tag{B.9}
$$

此界对任意固定完整数据集和所有 $\rho\in I$ 成立，常数不依赖 N 或 n；采用 $C_R$ 一支时也不依赖 K。它没有把真实训练行替换成 canonical nuisance 行，没有改变实际 raw 路径梯度，也没有使用 teacher 相关值。

### 2.3 对完整连续经验目标的振幅控制

条件于数据，独立 B 个 bank 给

$$
E[e_B'(\rho)^2\mid\mathcal D]
=\frac{\operatorname{Var}(H_{\mathcal D,Z}(\rho)\mid\mathcal D)}B
\le\frac{V_{R,K}}B.
\tag{B.10}
$$

由 (B.3)、Tonelli 与 (B.9)，有

$$
\boxed{E[(\operatorname{osc}e_B)^2\mid\mathcal D]
\le\frac{4R^2V_{R,K}}B.}
\tag{B.11}
$$

均值交换由 (B.4) 的可积包络保证，非负平方积分可直接使用 Tonelli。这个概率界对每个固定数据集具有同一个上界，也因此可对随机数据再平均；它不是“同一组 bank 实现同时对全部可能数据集具有小误差”的保证。

## 3. 连续 Price 校准具有更小的二阶矩尺度

### 3.1 定义一个理想连续校准程序

校准使用另一个独立 prior 池。给一个最大 bank Z，按同一 Cholesky 规则在全部 $\rho\in I$ 评价四个 active 模式 $y=(y_1,y_2)$ 的 Price 样本。校准的 nuisance 符号可统一取+1，因为这里估计的是 bank 期望，而不是替代实际训练行。记

$$
\begin{aligned}
D_{Z,y}(\rho)&=-s_y d_{Z,y}(\rho),\qquad s_y=y_1y_2,\\
d_{Z,y}(\rho)&=\sum_k\pi_{k,y}(1-\pi_{k,y})
\sigma(-y_1X_{k1})\sigma(-y_2X_{k2})\in[0,1],\\
P_\rho(y)&=\frac{1+r(\rho)s_y}{4},\qquad
G_Z(\rho)=-\sum_yP_\rho(y)D_{Z,y}(\rho).
\end{aligned}
\tag{B.12}
$$

有限校准身份给 $E G_Z(\rho)=g'(\rho)$。这个无偏性是先对候选参数求 Price 导数，再以已知模型的 $P_\rho$ 加权；不是对同时改变生成参数的对角目标作全导数。$P_\rho$ 是与该 bank 无关的确定概率。

定义连续估计

$$
g_C(\rho)=\int_0^\rho\frac1C\sum_{c=1}^CG_{Z_c}(u)\,du,
\qquad e_C=g_C-g,
\tag{B.13}
$$

其中两者在0归一化。每个样本函数连续且绝对值不超过1，故积分、期望交换合法，且 $E g_C=g$。这里积分为理想连续程序的一部分；没有声称C个bank配合固定101结点便能无误差实现它。

### 3.2 与有限目标斜率 b 绑定的矩界

由 $D_y^2\le|D_y|=d_y$、对概率 $P_\rho$ 的 Jensen 及 $P_\rho(y)\le(1+r_R)/4$，有

$$
E G_Z(\rho)^2
\le\sum_yP_\rho(y)E D_{Z,y}(\rho)^2
\le\frac{1+r_R}{4}\sum_yE d_{Z,y}(\rho)
=(1+r_R)b(\rho).
\tag{B.14}
$$

最后一步精确使用 $b=-V'=(1/4)\sum_yE d_y$，没有遗漏四模式的1/4。另一方面 $|G_Z|\le1$。所以可用

$$
W_{n,K}=\min\{1,(1+r_R)b^*\},\qquad b^*=\max_Ib,
\quad E G_Z(\rho)^2\le W_{n,K}.
\tag{B.15}
$$

本模型 $g'$ 奇、b偶。如果每个 bank 改用共同随机数的奇投影 $G_Z^{odd}(\rho)=[G_Z(\rho)-G_Z(-\rho)]/2$，其期望仍为 $g'(\rho)$；由 $(x-y)^2/4\le(x^2+y^2)/2$ 与b偶性，(B.14)—(B.15)仍成立。这不要求两侧独立，也不把符号复用当新增bank。

如同训练误差，(B.3)给

$$
\boxed{E[(\operatorname{osc}e_C)^2]
\le\frac{4R^2W_{n,K}}C
\le\frac{4R^2}C.}
\tag{B.16}
$$

更细的中间形式为 $(2R/C)\int_I\min\{1,(1+r_R)b(u)\}du$。独立校准池使这些界条件于任何实际数据也不变。原始粗界仅需C增长；比较消退目标时，(B.14)给出的 $b^*$ 因子是更强预算的关键。

## 4. 传递到实际连续近优输出的风险

记实际部署的校准为 $\widetilde g_C=g_C+u_C$，其中 $u_C$ 收集从理想连续程序到实现函数的误差；例如结点近似、积分/样条、模型参考和舍入误差。令输出相对于真算术实际连续目标 $\widehat F_B+\widetilde g_C$ 的regret至多η。则 [THEORY.md](THEORY.md) 的全局函数误差传递可以取

$$
E_0=\eta+\operatorname{osc}e_B+
\operatorname{osc}e_C+\operatorname{osc}u_C.
\tag{B.17}
$$

它使用实际全局近优性及已证明的二次增长，没有假设校准目标全域强凸，也没有把局部梯度线性代理当成真实风险。

对任意正确指定教师 $\theta\in I$，假设完整观测iid，令 $t_\star=r(\theta)$。若η和实现振幅具有有限期望，则由THEORY的精确似然传递、Cauchy–Schwarz及 $E(t-t_\star)^2=(1-t_\star^2)/N$，有

$$
\boxed{
E\,\mathrm{KL}(Q_\theta\|Q_{\widehat\rho})
\le\frac1{1-r_R^2}\left[
\frac{2}{\lambda_*}\left\{
E\eta+E\operatorname{osc}u_C
+2R\sqrt{\frac{V_{R,K}}B}
+2R\sqrt{\frac{W_{n,K}}C}\right\}
+\frac{1-t_\star^2}{N}\right].
}
\tag{B.18}
$$

$\lambda_*=\min_I b(1-r^2)/r'>0$ 是理想有限目标的斜率比，不能用一个未认证的样条最小值替代它。式(B.18)是该理想程序及误差合同下的有限预算上界，可能很松；不是本次四格数值精度的自动认证或必要收敛率。

若η与实现误差仅在概率至少 $1-\delta$ 的事件上被某个确定值控制，应按该事件拆期望并添加 $R_{max}\delta$，其中可取 $R_{max}=2r_R^2/(1-r_R^2)$。不能把高概率的近优/实现资格当作无条件误差等式。MC与校准的(B.11)、(B.16)是无条件矩控制，可在拆分后保守用于好事件，不需额外假设其与数据独立。

## 5. 渐近充分预算：B对w平方，C对w

现在让n增长，固定发射与R，并限于已经证明一致方向的整数范围 $2\le K_n\le C_0n^p$。沿用

$$
v=\operatorname{Var}(\log\sigma(Z))>0,\quad
w_n=J_{K_n}/\sqrt{nv},\quad
J_K=E\max_{k\le K}Z_k.
\tag{B.19}
$$

旧有限校准桥接已给 $b_n/w_n\to A=1/4$ 一致，所以 $b_n^*\asymp b_{*,n}\asymp\lambda_{*,n}\asymp w_n$。在此范围 $w_n\to0$。故 (B.11)、(B.16)直接推出

$$
E\left[\left(\frac{\operatorname{osc}e_{B_n}}{w_n}\right)^2\middle|\mathcal D\right]
\le\frac{4R^2V_{R,K_n}}{B_nw_n^2},\qquad
E\left[\left(\frac{\operatorname{osc}e_{C_n}}{w_n}\right)^2\right]
=O\!\left(\frac1{C_nw_n}\right).
\tag{B.20}
$$

使用(B.9)中不依赖K的分支，$B_nw_n^2\to\infty$保证第一项趋0；$C_nw_n\to\infty$保证第二项趋0。若只用较粗的最大半径界，利用整数K≥2时 $J_K^2\asymp\log K$ 的统一比较，可得到较强的要求 $B_n/n\to\infty$；若只用 $|G|\le1$，则得到较强的要求 $C_nw_n^2\to\infty$。这些旧粗界仍然有效，但无需作为本补充的主要预算。

**定理（连续近优一致性的充分路径）。** 在上述模型与K范围内，假定训练池、校准池和iid数据相互独立，$N_n\to\infty$，并且

$$
B_nw_n^2\to\infty,\qquad C_nw_n\to\infty,\qquad
\eta_n+\operatorname{osc}u_{C_n}=o_p(w_n).
\tag{B.21}
$$

对任何满足该连续近优合同的可测输出，有

$$
|\widehat\rho_n-\theta|\to_p0,\qquad
\mathrm{KL}(Q_\theta^{(n)}\|Q_{\widehat\rho_n}^{(n)})\to_p0,
\qquad
E\,\mathrm{KL}(Q_\theta^{(n)}\|Q_{\widehat\rho_n}^{(n)})\to0.
\tag{B.22}
$$

**证明。** (B.20)和Markov给两个MC振幅均为 $o_p(w_n)$，(B.21)补齐实现和求解误差。THEORY中的全域误差传递遂使输出到同数据clipped MLE的距离趋0；iid数据给经验相关趋于 $r(\theta)$，从而clipped MLE趋θ。全joint KL只有两位active项，nuisance完全相同而抵消，故连续性给KL依概率趋0。该KL对全部n和I内参数均被固定的 $R_{max}$ 控制，因而一致可积，推出期望趋0。最后一步没有擅自对局部一阶展开的平方余项取期望。□

N可以沿预定序列任意慢地趋∞，因为训练振幅矩界条件于任意固定数据并对N统一；这不是未知结构或高维参数学习的样本复杂度结论。

若进一步 $E\eta_n+E\operatorname{osc}u_{C_n}=o(w_n)$，(B.18)可写成粗但严格的渐近上界

$$
E\,\mathrm{KL}
\le O\!\left(
\frac{E\eta_n+E\operatorname{osc}u_{C_n}}{w_n}
+\frac1{\sqrt{B_nw_n^2}}
+\frac1{\sqrt{C_nw_n}}
+\frac1{N_n}\right).
\tag{B.23}
$$

只假设(B.21)的依概率误差时，仍有(B.22)的期望一致性，但不能不加条件声称(B.23)中的误差期望速度。此上界由全域函数误差得到，未认证任何项的最优阶。

## 6. 预算换算与实际实现边界

因为v固定，(B.21)的两项bank条件可改写为

$$
B_n\gg\frac{n}{J_{K_n}^2},\qquad
C_n\gg\frac{\sqrt n}{J_{K_n}}.
\tag{B.24}
$$

固定K时 $J_K$ 为正常数，得到n与sqrt(n)两档；当 $K_n\asymp n^p$ 且p>0，$J_{K_n}^2\sim2p\log n$，对应 $n/\log n$ 与 $\sqrt{n/\log n}$。例如可以令两项右侧各乘一个预定趋∞因子；定理并未指定该因子必须是某个幂或对数，也没有证明其必要性。

每个bank有K条(n+2)维prior向量，采样账目分别为 $B_nK_n(n+2)$ 与 $C_nK_n(n+2)$ 个Gaussian标量。连续校准函数的评价/积分、实际数据评分、求解和已知r的计算另有成本；少bank不等于同倍率的实测运行时间优势。

本补充不能自动追认当前程序的渐近一致性：

- 固定101结点样条必须另证其相对于连续 $g_C$ 的实现振幅是 $o_p(w_n)$。结点SE小或两种分辨率接近不构成这条全域结论。
- 单系数 `localquad` 除MC误差外还有函数形状近似残差；只有另行控制该残差才可使用(B.21)，不能从C预算条件直接断言二次规则精确一致。
- 固定801候选网格对任意连续teacher不自动满足 $\eta_n=o_p(w_n)$。需要相应求解精度或细化候选，而不是将现有有限网格视为连续极小点。
- 旧B/C有限实验的经验改善、SE和t区间与本定理不同；不能按本定理给它们新增通过门、严格置信覆盖或必要预算判定。
- 当前推导比较的是指定完整raw训练与模型已知Price校准的充分预算。它不排除其它方差缩减、评分改变或更便宜精确似然的程序，也不证明最优性、不可改进性或SGD保证。

## 7. 一个明确的有限网格充分实现

为避免定理只依赖抽象η，这里给出一个**精确已知r点值和实算术**下的有限网格构造；没有实际执行它。取 $m_n=\lceil Rn\rceil$，用对称网格 $\mathcal G_n=\{jR/m_n:-m_n\le j\le m_n\}$，包含0和端点，最大间距 $h_n\le1/n$、点数为O(n)。

CAL在这些结点计算 $\overline G_C=C^{-1}\sum_cG_{Z_c}$（也可先作奇投影），作分段线性插值 $S_C$，部署其精确分段二次反导 $\widetilde g_C(\rho)=\int_0^\rho S_C(u)du$。由Price样本的路径求导，若 $M_{K,c}$ 为该bank的active最大半径，则

$$
|G_{Z_c}'(\rho)|\le\frac14+
\frac{9M_{K,c}}{4\sqrt{1-R^2}},\qquad
\operatorname{Lip}(\overline G_C)\le
L_C:=\frac14+\frac9{4\sqrt{1-R^2}}\frac1C\sum_cM_{K,c}.
\tag{B.25}
$$

为核常数，单粒子候选log权重导数不超过 $M_K/\sqrt{1-R^2}$，所以 $\sum_k|\pi_k'|$ 不超过其两倍。对 $d_y$ 求导时，$\pi(1-\pi)$ 一项至多贡献两倍包络，第二位sigmoid一项至多贡献包络的1/4，合为9/4；模式权重的导数绝对值之和为 $|r'|\le1/4$。奇投影不增大这一统一包络。

对Lipschitz函数，分段线性插值的sup误差至多 $L_Ch_n/2$。积分振幅因而满足

$$
\operatorname{osc}(\widetilde g_C-g_C)
\le R L_Ch_n
=O_p\!\left(\frac{\sqrt{1+\log K_n}}n\right)
=o_p(w_n).
\tag{B.26}
$$

这里 $EL_C=O(\sqrt{1+\log K_n})$ 来自最大半径矩，故对任意C都可用Markov得到该 $O_p$；末步使用K≥2时 $J_K^2\asymp1+\log K$，不需隐含的结点独立性。

TRAIN在相同网格（或包含所有CAL结点的更细网格）逐点评价真实完整raw目标加上这个部署惩罚，取真算术最小值。raw的逐段二阶包络可取

$$
D_B=\frac5{4(1-R^2)}\frac1B\sum_bM_{K,b}^2
+\frac1{(1-R^2)^{3/2}}\frac1B\sum_bM_{K,b},
\qquad ED_B=O(1+\log K_n).
\tag{B.27}
$$

校准反导在每个结点区间的二阶导数为 $S_C'$，其绝对值不超过 $L_C$。因TRAIN网格包含全部CAL结点，每个网格小区间内完整目标的二阶包络为 $D_B+L_C$。逐段线性插值界遂给所选点相对于该连续部署目标的regret

$$
\eta_n\le\frac{(D_B+L_C)h_n^2}{8}
=O_p\!\left(\frac{1+\log K_n}{n^2}\right)=o_p(w_n).
\tag{B.28}
$$

最后一步在既定多项式K范围成立。该二阶论证允许惩罚在结点处的二阶导数跳变，因为每个区间分别处理且结点全部包含，不需要全域强凸。

于是这个有限网格构造自动满足(B.21)的实现与求解要求，再配合 $B_nw_n^2\to\infty$、$C_nw_n\to\infty$、$N_n\to\infty$，得到(B.22)。可以选择多项式B、C、N以及O(n)候选网格形成有限实算术程序；该说法不包括精确模型r点值的取得成本、浮点位复杂度或最优计算复杂度。实际参考表误差、sigmoid/log、权重、积分和求值舍入仍需另控制为 $o_p(w_n)$；没有追认当前固定101/801程序或任意连续优化器。

因此本补充增强的是一个可验证的理论方向：在保持固定非退化噪声和真实共享bank训练合同的同时，连续校准的充分bank增长可比训练更慢；实际四case程序是否已进入对应精度范围，仍由独立实验和其数值边界判断。
