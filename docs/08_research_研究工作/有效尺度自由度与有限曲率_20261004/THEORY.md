# 有效均值与尺度释放：有限曲率门槛和总体尺度塌缩

日期：2026-10-04。本页为本轮解析推导；0 RNG、0 模型积分、0 参数优化。保留 v8 和全部旧科学。先将结论写成可独立核验的条件命题，不认证首创，不预设实际优化器会到达总体近优点。

本轮回答的限制是有效变量的均值与尺度原先固定。释放它们后，相关矩阵不再是唯一会变化的参数。下面区分两项结果：有限粒子准则在产品边际附近的精确二阶条件，以及无关维数增长时的统一主阶极限。后者不把旧固定尺度的细阶谱定理直接套到新参数类。

**与项目已有结果的关系。** 2026-09-29 的[固定边际接口理论](../边际依赖接口_20260929/THEORY.md)已经给出均匀 iid 粒子时的 \(D-(1-1/K)\Sigma_Y\) 局部形式、谱阈值、均值补偿和纯高阶依赖例外；自由因子局部极小推论 (historical reference not distributed)已经证明自由均值交叉块为零及严格局部极小。下面第2节是这些结果在独立无关权重下的直接扩展，不再次作为本轮原创发现。新增分析集中于 \(1/K\) 被无关权重碰撞 \(\chi\) 替换、\(\delta\) 相关矩阵类中的精确可行门槛，以及释放有效均值与尺度后的高维总体塌缩和观察风险。

## 1. 模型、边界点与独立性

固定有效维数 \(m\ge2\)。真实有效观察 \(Y\sim P\) 取值于 \(\{0,1\}^m\)，边际为 \(p_i=E_PY_i\in(0,1)\)。令

\[
D=\operatorname{diag}\{p_i(1-p_i)\},\quad
\Sigma_Y=\operatorname{Cov}_P(Y),\quad
R_Y=D^{-1/2}\Sigma_YD^{-1/2},\quad
\mu_i^0=\operatorname{logit}(p_i).
\tag{S.1}
\]

候选模型为 \(X_i=\mu_i+s_iZ_i\)，其中 \(Z\sim N(0,C)\)、\(s_i\ge0\)，给定 X 后各位独立 Bernoulli-logistic 发射。相关矩阵类为

\[
\mathcal E_\delta^{(m)}=
\{C=C^\top:\operatorname{diag}C=1,\ C\succeq\delta I\},
\qquad 0<\delta<1.
\tag{S.2}
\]

本项目对应 \(\delta=.1\)。记候选观察概率为 \(Q_{\mu,s,C}\)，以及真实观察风险 \(R(\mu,s,C)=D_{\rm KL}(P\Vert Q_{\mu,s,C})\)。有效核的 log 为

\[
A_k(Y;\mu,s,C)=\sum_i\{Y_i(\mu_i+s_iZ_{ki})-
\operatorname{softplus}(\mu_i+s_iZ_{ki})\}\le0.
\tag{S.3}
\]

给定有限 \(K\ge1\)，\(Z_1,\ldots,Z_K\) 独立、同分布，且独立于 Y 和所有无关权重。\(W_1,\ldots,W_K\) 几乎处处有限且严格为正；定义

\[
\pi_k=\frac{W_k}{\sum_\ell W_\ell},\qquad
\chi=E\sum_k\pi_k^2,\qquad \omega=1-\chi.
\tag{S.4}
\]

有限曲率部分不要求 W 服从特定分布，也不要求 W 各分量彼此独立；但需要整个 W 向量与 \((Y,Z_1,\ldots,Z_K)\) 独立。标准无关块 prior 采样满足该条件。若权重规律依赖有效观察标签，下面的单个标量 \(\omega\) 一般不足以描述 Hessian。

为避免与无关目标绝对常数的可积性混淆，直接定义中心化原目标

\[
\overline F_K(\mu,s,C)
=E\left[-\log\sum_k\pi_k e^{A_k(Y;\mu,s,C)}\right].
\tag{S.5}
\]

当无关负 log 目标有限时，这正是完整原目标减去同 bank 的无关目标。其定义在有限均值、尺度下有 Gaussian 线性包络。这里 \(\pi_k\) 是**无关权重**；离开 \(s=0\) 后，完整权重另含 \(e^{A_k}\)，不可与本节的碰撞量混用。

在 \((\mu,s)=(\mu^0,0)\)，候选观察模型为产品边际

\[
Q_0(y)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i},
\qquad \overline F_K(\mu^0,0,C)=\sum_i h(p_i),
\tag{S.6}
\]

不依赖 C。因此这是一个 C 不可识别的参数流形，不是唯一的完整参数点。若实现限定严格正尺度，\(s=0\) 是闭包边界；有限正 floor 的结论另列。

## 2. 有限粒子的精确二阶式

**命题 1（产品边际处的 Hessian）。** 对每个固定 C，在上述点，\(\nabla_\mu\overline F_K=\nabla_s\overline F_K=0\)，且

\[
\nabla_{\mu\mu}^2\overline F_K=D,\qquad
\nabla_{\mu s}^2\overline F_K=0,\qquad
H_s^F(C):=\nabla_{ss}^2\overline F_K
=D-\omega(\Sigma_Y\circ C).
\tag{S.7}
\]

符号 \(\circ\) 为逐元素乘积。精确观察风险在同一点满足

\[
\nabla R=0,\qquad
\nabla_{\mu\mu}^2R=D,\qquad
\nabla_{\mu s}^2R=0,\qquad
H_s^R(C)=D-\Sigma_Y\circ C.
\tag{S.8}
\]

**证明。** 记 \(r_i=Y_i-p_i\)。零尺度处
\(\partial_{s_i}A_k=r_iZ_{ki}\)，
\(\partial_{s_i}\partial_{s_j}A_k=-D_{ii}Z_{ki}^2\mathbf1_{i=j}\)。负 log-sum-exp 的 Hessian 是负的加权二阶导数减去加权一阶导数的协方差。固定 Y、W，利用 Gaussian 粒子彼此独立、均值零及 \(E Z_{ki}Z_{kj}=c_{ij}\)，有

\[
E_Z\left[\sum_k\pi_k Z_{ki}Z_{kj}
-(\sum_k\pi_kZ_{ki})(\sum_\ell\pi_\ell Z_{\ell j})\right]
=(1-\sum_k\pi_k^2)c_{ij}.
\tag{S.9}
\]

再对 Y、W 取期望即得尺度块；均值的一阶导在粒子间相同，故其加权协方差为零；交叉块因 Gaussian 零均值而消失。精确目标对单个 Gaussian 的积分代替粒子平均，同一计算中协方差项系数成为 1，得到 (S.8)。有限 K 下各导数由 Gaussian 多项式矩控制，允许微分交换期望。结论亦可先将 s 延伸为有符号实向量求导，再限制回非负锥。证毕。

注意精确风险的尺度 Hessian 对角为零，而原目标的尺度 Hessian 对角为 \(\chi D\)。这不是以真值 Fisher 代替任意状态风险 Hessian；这里的点是产品边际，通常不是真实联合分布。

## 3. 全相关矩阵类上的尖锐有限曲率门槛

令 \(\lambda=\lambda_{\max}(R_Y)\ge1\)，

\[
B=\delta+(1-\delta)\lambda.
\tag{S.10}
\]

**引理 2（Schur 乘积的极值）。**

\[
\sup_{C\in\mathcal E_\delta^{(m)}}
\lambda_{\max}(R_Y\circ C)=B.
\tag{S.11}
\]

**证明。** 每个 C 可写为 \(C=\delta I+(1-\delta)T\)，T 为相关矩阵。由 \(\lambda I-R_Y\succeq0\) 和 Schur 乘积定理，
\(\lambda I-R_Y\circ T=(\lambda I-R_Y)\circ T\succeq0\)，给上界。任取 \(q_i\in\{-1,1\}\)，令 \(T=qq^\top\)，则 \(R_Y\circ T=\operatorname{diag}(q)R_Y\operatorname{diag}(q)\)，谱保持，达到上界。证毕。

取单位顶端特征向量 v，并令 \(q_i=\operatorname{sign}(v_i)\)，零坐标的符号任取。定义

\[
C^\dagger=\delta I+(1-\delta)qq^\top,\qquad
N=\|D^{-1/2}|v|\|_2,\qquad
a=\frac{D^{-1/2}|v|}{N}\ge0.
\tag{S.12}
\]

C 为正定，\(\lambda_{\min}(C^\dagger)=\delta\)；a 为合法、单位长度的非负尺度激活方向。顶端特征值重根不影响存在性，选择任一固定单位顶端向量即可。沿 \(\mu=\mu^0,s=ta,C=C^\dagger\)，有

\[
\left.\frac{d^2\overline F_K}{dt^2}\right|_{t=0}
=\frac{1-\omega B}{N^2},\qquad
\left.\frac{d^2R}{dt^2}\right|_{t=0}
=\frac{1-B}{N^2}.
\tag{S.13}
\]

**定理 3（允许均值与全部尺度变化的局部相反偏好）。** 若 \(\lambda>1\) 且

\[
\chi>1-\frac1B
\quad\Longleftrightarrow\quad \omega B<1,
\tag{S.14}
\]

则原目标在 \((\mu^0,0)\) 的均值—尺度方向具有一致正曲率，均匀覆盖整个相关矩阵类；产品边际流形为局部最小流形。同时，(S.12) 的可行尺度激活对充分小的 \(t>0\) 严格降低观察风险，却严格提高原目标。

**证明。** 白化后的尺度 Hessian 为
\(D^{-1/2}H_s^F D^{-1/2}=I-\omega(R_Y\circ C)\)，故 (S.11) 给最小特征值至少 \(1-\omega B>0\)。连同均值块 D、零交叉块，得到整个均值—尺度 Hessian 的一致正下界。Gaussian 包络和 C 的紧性给 Hessian 在邻域内的一致连续性，从而原目标增加至少常数乘 \(\|\mu-\mu^0\|^2+\|s\|^2\)。沿 (S.12)，(S.13) 的两个曲率严格异号，且一阶项为零，Taylor 展开给结论。证毕。

此处释放均值不自动吸收风险改进：任何保持观察边际的光滑补偿路径 \(\mu_i(t)=\mu^0_i+O(t^2)\) 都有相同的尺度二阶式，因为基点的均值梯度为零。该补偿路径的存在由隐函数定理给出：\(E\sigma(\mu_i(t)+ta_iG)=p_i\)，且对均值的导数为 \(D_{ii}>0\)。

门槛的反面也应明确：当 \(\omega B>1\)，(S.12) 同时降低原目标和风险；这只是此局部方向不再相反，不是全局恢复或学习器保证。等号时二阶信息不足。\(\lambda=1\) 等价于 \(R_Y=I\)，即各对观察不相关；它不等价于观察联合独立。若存在纯高阶依赖，本定理的精确风险二阶式为零，必须另查更高阶，不能把本条件写成对任意依赖教师都成立。

这一门槛只在规定的产品边际基点、独立无关权重和尺度激活中成立。离开该点后，完整权重与有效标记耦合；旧结论“重叠量本身不能决定一般相关梯度方向”保持。

## 4. 无关权重退化下的统一主阶极限

本节固定 \(m,\delta\)，允许 \(\mu\in\mathcal M\)、\(s\in[0,S]^m\)、\(C\in\mathcal E_\delta^{(m)}\)，其中均值盒 \(\mathcal M\) 紧且包含 \(\mu^0\)，S 有限。令 n 个独立公平 Bernoulli 无关观察以单位 Gaussian-logistic prior 表示，因对称性可固定其标签；则

\[
W_{nk}=\prod_{j=1}^n\sigma(G_{kj}),\qquad
G_{kj}\stackrel{\rm iid}{\sim}N(0,1).
\tag{S.15}
\]

这些输入与有效变量、教师观察独立。此处无关表示固定为正的单位尺度；若无关尺度本身也可学习并随 n 变小，下面的塌缩定理不覆盖该联合优化问题，不能用它否定旧 FREE 无关尺度实验的缓解结果。取整数 \(2\le K\le C_0n^b\)，固定 \(C_0,b>0\)。令 \(v_0=\operatorname{Var}\log\sigma(G)>0\)，\(J_K=E\max_{k\le K}G_k\)，\(w_{n,K}=J_K/\sqrt{nv_0}\)。

**引理 4（直接继承旧近并列引理的无关重叠）。**

\[
\omega_{n,K}:=E[1-\sum_k\pi_{nk}^2]
=(1+o(1))w_{n,K},
\tag{S.16}
\]

误差对允许的 K 一致，特别地 \(\sup_K\omega_{n,K}=O(\sqrt{\log n/n})\)。

**依据。** v8 英文稿 (historical reference not distributed) 附录 A 的 (A.12)–(A.13)，取全部有效 log 分数 \(A_j=0\)。每对 \(k\ne\ell\) 的权重乘积期望为 \((1+o(1))I_K/\sqrt{nv_0}\)，其中 \(K(K-1)I_K=J_K\)。累加所有有序对即 (S.16)。这里仅用旧相对近并列证明，没有新模拟、没有将高斯权重近似当作实际无关分布。K=1 时 \(\omega=0\)，下面的单粒子极限实际逐 n 精确成立，可另列但不改变原 K≥2 主范围。

**引理 5（从无关重叠到有效 log 目标的逆矩界）。** 记有效正核 \(f_Y(Z;\theta)=e^{A(Y;\theta)}\)，其中 \(\theta=(\mu,s,C)\)，并令

\[
M(\theta)=E_{Y\sim P}\!\left[
E_Z f_Y(Z;\theta)\,E_Z f_Y(Z;\theta)^{-1}-1\right].
\tag{S.17a}
\]

逆矩中的期望针对另一份同分布 Gaussian，并不是 \((E_Zf_Y)^{-1}\)。对任意 K 及第1节的独立权重，精确地有

\[
0\le L(\mu,s)-\overline F_K(\mu,s,C)
\le\omega M(\theta),
\qquad
L(\mu,s)=\sum_i\{E\operatorname{softplus}(\mu_i+s_iG)-p_i\mu_i\}.
\tag{S.17}
\]

**证明。** 固定 Y、\(\pi\)，以 \(f_k=f_Y(Z_k;\theta)>0\) 记各独立有效核。由加权 Jensen 及 \(\log x\le x-1\)，逐 bank 有

\[
0\le\log\sum_k\pi_kf_k-\sum_k\pi_k\log f_k
\le(\sum_k\pi_kf_k)(\sum_j\pi_jf_j^{-1})-1.
\tag{S.18}
\]

条件期望下，右边对角项 \(k=j\) 为 \(\sum_k\pi_k^2\)，交叉项为
\((1-\sum_k\pi_k^2)E_Zf_Y\,E_Zf_Y^{-1}\)。因此总上界恰为
\((1-\sum_k\pi_k^2)(E_Zf_Y E_Zf_Y^{-1}-1)\)。左边的期望等于 \(L-\overline F_K\)，因为 \(\pi\) 独立于所有有效粒子及 Y。再平均 Y、\(\pi\) 得 (S.17)。证毕。

**统一逆矩资格。** 若 \(|\mu_i|\le U\)、\(s_i\le S\)，展开
\(f_y^{-1}=\prod_i\{1+e^{-(2y_i-1)(\mu_i+s_iZ_i)}\}\)，每个子集的期望由 Gaussian MGF 给出；其均值项不超过 mU，方差项不超过 \(m^2S^2\)。故

\[
\sup_{y,\theta}E_Z f_y^{-1}
\le 2^m\exp\{mU+m^2S^2/2\}<\infty.
\tag{S.18a}
\]

因为 \(0<f_y\le1\)，(S.18a) 给 \(M_*:=\sup_\theta M(\theta)<\infty\)，且
\(0\le L-\overline F_K\le M_*\omega\) 一致成立。此界可能数值很松，但作为统一渐近收敛界足够；不把它当作当前有限激活大小的实用误差证书。

**定理 6（自由尺度的统一主阶目标与产品边际塌缩）。**

\[
\sup_{2\le K\le C_0n^b}\sup_{\mu,s,C}
|\overline F_{n,K}(\mu,s,C)-L(\mu,s)|\longrightarrow0.
\tag{S.19}
\]

若 \(\theta_n=(\mu_n,s_n,C_n)\) 的原总体目标 regret 为 \(o(1)\)，则

\[
\mu_n\longrightarrow\mu^0,\qquad s_n\longrightarrow0,
\qquad Q_{\theta_n}\longrightarrow Q_0,
\qquad R(\theta_n)\longrightarrow D_{\rm KL}(P\Vert Q_0)
=\sum_i h(p_i)-H(P).
\tag{S.20}
\]

**证明。** 由引理4、5，统一误差为 \(O(w_{n,K})\)，且允许 K 中的最大值趋零。这里仅把 \(A=0\) 输入旧近并列引理；不需要重建带有效分数的相对尾近似，也不需要新的截断增长窗口。

严格 Jensen 不等式和 Bernoulli 交叉熵最小化给

\[
L(\mu,s)\ge\sum_i\{\operatorname{softplus}(\mu_i)-p_i\mu_i\}
\ge\sum_i h(p_i),
\tag{S.21}
\]

且当且仅当全部 \(s_i=0\)、\(\mu_i=\mu_i^0\) 等号成立。C 仍完全平坦。紧性和统一函数收敛把 o(1) regret 转成到这一最小流形的距离趋零。观察模式有限；\(Q_\theta(y)>0\) 连续，且在整个紧参数类上具有一致正下界。因此候选概率和 KL 连续，得到 (S.20)。证毕。

对随机参数输出，若总体 regret 为 \(o_p(1)\)，相同结论依概率成立；风险在紧类上有界，故期望也收敛到 (S.20) 的值。允许 \(s>0\) 而仅把0作为闭包时，只要允许任意小尺度且近优相对该闭包的下确界定义，同一极限成立。

右端是 P 相对其产品边际的总相关，任意非独立 P 都严格为正，包括两两独立但有高阶依赖的情况。这比定理3的二阶可辨范围更广。若要把它表述为模型本可达到零风险却被原目标推离，必须另假设 P 可由本自由参数类内的某个参数实现；定理本身容许任意 P，不能省略这一正确指定条件。

与旧定理的区别：固定有效尺度时，主阶 L 对相关矩阵不变，需要 \(o(w)\) 精度辨认细阶谱偏好；释放尺度后，非退化有效尺度已在主阶受到严格惩罚，所以这里只需 \(o(1)\) 总体近优误差。结论是依赖表达塌缩，不是旧谱极点结论在新参数类中的逐字延续。它不证明有限 n 的全局最优点已经为0，也不证明 Adam 会实现该精度。

## 5. 正尺度 floor：受限恢复而非零尺度结论

固定 \(\varepsilon>0\)，把尺度类改为 \([\varepsilon,S]^m\)。令 \(\mu_p(s)\) 是唯一解
\(E\sigma(\mu_p(s)+sG)=p\)。均值盒需包含每个 \(\mu_{p_i}(\varepsilon)\)。存在性、唯一性由均值方向的严格单调性及0/1端点极限给出。

对每一坐标，先将 L 对均值最小化。包络定理和 Stein 身份给

\[
\frac d{ds}\min_\mu
\{E\operatorname{softplus}(\mu+sG)-p\mu\}
=E[G\sigma(\mu_p(s)+sG)]
=sE\sigma'(\mu_p(s)+sG)>0\quad(s>0).
\tag{S.22}
\]

因此该 floor 类中 L 的唯一均值—尺度最小点是
\(s_i=\varepsilon\)、\(\mu_i=\mu_{p_i}(\varepsilon)\)，而 C 仍任意。定理6的统一收敛给原总体 o(1) 近优序列趋向这一流形。均值盒未包含这些点时，不可沿用该精确端点识别。

在极限流形上，候选边际正好为 p。Gaussian Price 公式与 \(0<\sigma'\le1/4\) 给

\[
|\operatorname{Cov}_Q(Y_i,Y_j)|
\le\varepsilon^2|c_{ij}|/16
\le\varepsilon^2(1-\delta)/16.
\tag{S.23}
\]

若用自旋 \(S_i=2Y_i-1\)，则协方差上界乘4，为
\(\varepsilon^2(1-\delta)/4\)。这一区分不能丢掉。

令真实自旋协方差 \(\kappa_{ij}=\operatorname{Cov}_P(S_i,S_j)\)。候选和真实边际一致，故 pair 自旋乘积期望之差即协方差之差。总变差和 Pinsker 不等式于是给任何相应近优序列

\[
\liminf_n R(\theta_n)
\ge\frac12
\left(\max\{|\kappa_{ij}|-\varepsilon^2(1-\delta)/4,0\}\right)^2.
\tag{S.24}
\]

可选任意一对，或取各对下界的最大值；不能将重叠 pair 下界直接相加。有限 n 的均值未必精确匹配，此式依靠近优极限和连续性，不能不加误差项移作任意有限输出的下界。

## 6. 贡献边界与可验证含义

1. 有效尺度的自由度并不自动消除有限粒子失真。无关权重独立且足够集中时，产品边际附近原目标稳定，而某些合法依赖激活严格改善观察风险。这是一个带具体二阶门槛的局部模型边界，不是优化器因果分解。
2. 允许全部有效均值与尺度学习后，旧相关矩阵细阶问题可以转化为主阶依赖塌缩。唯一性只涉及均值和尺度，C 在零尺度处不可识别。正 floor 下则有可量化的观察相关能力上限。
3. 一般 log 估计偏差、Jensen 不等式、Schur 乘积、Gaussian Price/Stein 身份及 Pinsker 都是既有工具。本页不将这些单独包装为新定理贡献。项目09-29已具有均匀权重的局部谱门槛与自由均值局部极小；本轮有限分析是其带权、谱下限约束的扩展，主要新增全局结论是与旧非 Gaussian 无关近并列极限连接的自由参数类行为。引理5是初等权重 Jensen-gap 控制，也不单独认证新的一般不等式。
4. 模型继续假定有效块位置、无关块独立性、prior 采样和无关表示约束已知。没有释放有效—无关交叉协方差，没有任务头、自然数据、后验提议或端到端神经网络结论。
5. 有限 Hessian 是总体期望导数。数值有限差分、Monte Carlo 碰撞估计、模型积分资格和实际训练，是各自另行定义的证据；不能因本页解析条件成立便宣布某个有限实验通过。
6. 本页未创建新检验或追认旧门。前期 OUT4/IN4 风险—原评分反向、新数据对 I 的改善与 FACT 的较低风险仍属于旧实验事实，与这里释放尺度后的新参数类分开。
