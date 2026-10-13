# 有界可训练均值下的总体近优失真与单事件补偿边界

本文件独立扩展固定正尺度下限的总体目标理论。允许任意坐标均值与任意坐标尺度训练，保留已知的等相关矩阵结构；不释放任意协方差。证明保留标准化总自旋的全部均值漂移，不能由平均均值平方趋零直接推出未中心化 CLT。没有新随机观察、拟合或数值积分。全部 log 为自然底，优化误差与 joint KL 均按完整向量的总 nats 计。

## 1. 对象与主结论

固定 \(0<a<b<\infty\)、\(M>0\)、\(0<c<1\)，沿偶数 \(d\ge4\) 定义
\[
\rho_d=\frac{c}{d-1},\qquad
R_d^-=(1+\rho_d)I-\rho_d\mathbf1\mathbf1^\top,\qquad
R_d^+=(1-\rho_d)I+\rho_d\mathbf1\mathbf1^\top .
\tag{M1}
\]
对每个符号 \(\xi\in\{-,+\}\)，使用同一参数域
\[
\Theta_d^M=[-M,M]^d\times[a,b]^d,\qquad
\theta=(\mu,s),\quad
\eta=\mu+D_sU,\quad U\sim N(0,R_d^\xi).
\tag{M2}
\]
给定 \(\eta\)，自旋 \(Y_i\in\{-1,1\}\) 条件独立且
\(P(Y_i=y_i\mid\eta)=\sigma(y_i\eta_i)\)。记精确积分分布为
\(Q_{\theta,d}^\xi\)。参数域包括两个组均值或两个组尺度的限制类，但本文件证明直接处理更大的逐坐标参数域。每个候选均满足
\[
a^2(1-c)I\preceq D_sR_d^\xi D_s\preceq\Lambda I,
\qquad \Lambda=b^2(1+c).
\tag{M3}
\]
这是 covariance 的双侧谱界，不是把自由 Cholesky 对角元下限当成谱下限。

教师仍为旧主问题的同一个零均值两尺度分布：
\[
P_d^{\star,\xi}=Q_{0,s^\star,d}^\xi,\qquad
s_i^\star\in\{s_L^\star,s_H^\star\},\quad
\#\{i:s_i^\star=s_L^\star\}=d/2,\quad
a<s_L^\star<s_H^\star<b.
\tag{M4}
\]
两个真尺度固定，不随 d 变化，真分组任意。教师属于 (M2)，且具有全局自旋翻转对称性，特别是 \(E_{P^\star}Y_i=0\)。

给每个候选取 K 个 iid prior 粒子，完整 bank 独立于评分标签 \(Y\sim P^\star\)，并定义
\[
\begin{aligned}
f_\eta(y)&=\prod_i\sigma(y_i\eta_i),&
\widehat Q_{\theta,K}^{\rm raw}(y)&=\frac1K\sum_{k=1}^Kf_{\eta_k}(y),\\
F_{d,K}^{\rm raw}(P^\star;\theta)&=
E[-\log\widehat Q_{\theta,K}^{\rm raw}(Y)],&
R_d(P^\star;\theta)&=E_{P^\star}[-\log Q_{\theta,d}^\xi(Y)].
\end{aligned}
\tag{M5}
\]
目标 F 同时对标签和 bank 取总体期望；K、训练样本数以及用于估计 F 的独立 bank 数不是同一个预算。

本文件证明三件事：

1. 当 \(\log K_d=o(d)\)、总优化误差 \(\varepsilon_d=o(d)\) 时，全部 raw 总体近最优点具有平均尺度收缩和平均均值平方消失；标准化总自旋在**减去其精确均值后**仍趋向固定 floor 方差。
2. 负族的一个中央事件仍统一失真。正族的这个单事件确可被 \(O(d^{-1/2})\) 均值补偿，但两个预先固定的左右尾事件至少一个失真，因此全部近最优点仍有总 joint KL 正下界。
3. 正族的每候选同 Q 私噪声边际化在整个 (M2) 上仍给总 KL 恢复界。取同一 \(K_d=d\)、同一 \(\varepsilon_d\to0\)，raw 与边际化目标的近最优集合具有不同的预测后果，包括两边的实际总体极小点。

## 2. 非零均值的统一目标界

置
\[
L(\mu,s)=E_Z\log[2\cosh((\mu+sZ)/2)],\qquad
\ell(s)=L(0,s),\qquad Z\sim N(0,1),
\]
\[
\Psi_d(\theta)=\sum_iL(\mu_i,s_i),\qquad
t_{d,K}=\sqrt{2\Lambda d\log K}.
\tag{M6}
\]

**引理 1（零观察边际下的主项）。** 对任意满足
\(E_{P^\star}Y_i=0\) 的标签分布及整个 (M2)，
\[
0\le\Psi_d(\theta)-F_{d,K}^{\rm raw}(P^\star;\theta)
\le t_{d,K}.
\tag{M7}
\]
不要求教师的观察坐标独立；也不要求教师属于模型类。若标签边际非零，则相同证明的主项变为
\(\Psi_d(\theta)-\frac12\sum_i\mu_iE_{P^\star}Y_i\)，不能继续使用 (M7) 的零边际主项。

**证明。** 对固定 y，
\[
h_y(\eta)=-\log f_\eta(y)
=\sum_i\log[2\cosh(\eta_i/2)]-\tfrac12y^\top\eta,
\qquad
E_\eta h_y=\Psi_d(\theta)-\tfrac12y^\top\mu .
\]
每个 \(\partial_{\eta_i}h_y\) 的绝对值至多 1，所以
\(h_y(\mu+\Sigma^{1/2}Z)\) 是 \(\sqrt{\Lambda d}\)-Lipschitz。
平移 \(\mu\) 不改变这个常数。标准 Gaussian log-Sobolev/Herbst 界给
\[
\log E\exp\{t(h_y-Eh_y)\}\le t^2\Lambda d/2 .
\]
于是 \(E\max_{k\le K}(Eh_y-h_y(\eta_k))\le t_{d,K}\)，其中 K=1 时左侧为零。结合逐 bank 不等式
\[
\min_kh_y(\eta_k)\le
-\log\!\left(K^{-1}\sum_ke^{-h_y(\eta_k)}\right)
\le K^{-1}\sum_kh_y(\eta_k),
\]
得到 \(0\le E_\eta h_y-E_{\rm bank}[-\log\widehat Q(y)]\le t_{d,K}\)。
最后对独立的 Y 平均，\(\sum_i\mu_iE Y_i=0\) 即给 (M7)。这里不能将 \(E_\eta h_y\) 误写为与 y 无关；消去线性项是在标签平均之后发生的。\(\square\)

集中、最大值界与 Jensen 均为标准工具，Gaussian 集中的归一化来源见文末。结论保持标准 iid 等权 prior 合同，不推广到标签依赖 proposal 或权重。

## 3. 全部近最优点的明确定位与存在性

定义严格正常数
\[
\kappa_a=\ell'(a)
=\tfrac12E[Z\tanh(aZ/2)]>0,\qquad
\kappa_M=\frac{P(|Z|\le1)}4
\operatorname{sech}^2((M+b)/2)>0.
\tag{M8}
\]
在 \(|\mu|\le M\)、\(a\le s\le b\) 上，
\[
\partial_{\mu\mu}L(\mu,s)
=\tfrac14E\operatorname{sech}^2((\mu+sZ)/2)\ge\kappa_M.
\]
因为 L 关于 \(\mu\) 为偶函数，且 \(\ell''(s)>0\)，
\[
L(\mu,s)-\ell(a)
\ge \kappa_a(s-a)+\tfrac12\kappa_M\mu^2.
\tag{M9}
\]
这个显式 \(\kappa_M\) 只用 \(|Z|\le1\) 的部分积分给下界；不需要新的数值求积。

**定理 2（自由有界均值、逐坐标尺度的近优集合）。** 固定 d,K 时，F 在 (M2) 达到最小值。定义非空集合
\[
\mathcal A_d(\varepsilon)=
\{\theta\in\Theta_d^M:F_{d,K}^{\rm raw}(P^\star;\theta)
\le\min_{\vartheta\in\Theta_d^M}F_{d,K}^{\rm raw}(P^\star;\vartheta)
+\varepsilon\}.
\]
每个 \(\theta\in\mathcal A_d(\varepsilon)\) 都满足
\[
\kappa_a\,\frac1d\sum_i(s_i-a)
+\frac{\kappa_M}{2d}\sum_i\mu_i^2
\le\delta_{d,K,\varepsilon}:=
\sqrt{\frac{2\Lambda\log K}{d}}+\frac{\varepsilon}{d}.
\tag{M10}
\]
所以 \(\log K_d=o(d)\)、\(\varepsilon_d=o(d)\) 时，两个平均量在整个近优集合上一致趋零。一般不能由此推出
\(\max_i|\mu_i|\to0\)、所有坐标尺度一致趋 a，或标准化总自旋的均值趋零。

**证明。** 对固定 d,K，用相同标准 Gaussian bank 写出参数连续的 \(\eta_k=\mu+D_s(R_d^\xi)^{1/2}Z_k\)。每个标签、bank 的评分连续，且非负并不超过
\[
d(\log2+M)+\frac bK\sum_{k,i}|((R_d^\xi)^{1/2}Z_k)_i|.
\]
该上界与参数无关且可积。支配收敛给 F 连续；(M2) 紧，故达到最小值。
比较可行点 \((0,a\mathbf1)\)，由 (M7) 得
\[
\Psi_d(\theta)-t_{d,K}\le F(\theta)
\le\min F+\varepsilon
\le F(0,a\mathbf1)+\varepsilon
\le d\ell(a)+\varepsilon .
\]
代入 (M9) 并除以 d 即为 (M10)。\(\square\)

K=1 时 F 精确等于 \(\Psi_d\)，其唯一参数最优点为
\(\mu=0,s=a\mathbf1\)。K>1 的结论只定位渐近平均参数，不证明有限 d 精确贴边、均值恰为零或目标凸性。有限维的 (M10) 也可能不比原参数区间更有信息。

若只允许两个组均值和两个组尺度且比较点可行，上述同一比较直接给
\[
\frac{\kappa_a}{2}[(u-a)+(v-a)]
+\frac{\kappa_M}{4}(\mu_H^2+\mu_L^2)
\le\delta_{d,K,\varepsilon}.
\tag{M11}
\]
因此旧两个尺度定位得到保留，但均值的可得量级仍不足以删去总和漂移。
事实上下面的精确漂移满足
\(\lvert\beta_d\rvert\le\frac12\sqrt{\sum_i\mu_i^2}
\le\sqrt{(t_{d,K}+\varepsilon)/(2\kappa_M)}\)。
这个界可以发散；它不是 \(\beta_d=o(1)\) 的证明。

## 4. Cesàro 参数收敛下的中心化三角阵 CLT

令
\[
f_{i,d}(z)=\tanh((\mu_{i,d}+s_{i,d}z)/2),\qquad
b_{i,d}=E f_{i,d}(Z),\qquad
\beta_d=\frac1{\sqrt d}\sum_i b_{i,d},
\]
\[
S_d=\frac1{\sqrt d}\sum_iY_i,\qquad
\alpha(a)=E[Z\tanh(aZ/2)],\quad
v(a)=E[\tanh(aZ/2)^2],\quad
V_a^\xi=1+\xi c\alpha(a)^2 .
\tag{M12}
\]
公式中的 \(\xi\) 在代数表达里取 \(-1\) 或 \(+1\)。
每个 \(U_i\) 的边际精确为标准正态，因此 \(\beta_d=E_{Q_{\theta,d}^\xi}S_d\)，不是 Taylor 近似的均值。

**引理 3（保留任意漂移的 CLT）。** 对任意确定性参数三角阵，只要
\[
\frac1d\sum_i(s_{i,d}-a)\longrightarrow0,\qquad
\frac1d\sum_i\mu_{i,d}^2\longrightarrow0,
\tag{M13}
\]
并保持 (M2) 的固定界，就有
\[
S_d-\beta_d\Rightarrow N(0,V_a^\xi),\qquad
\sup_{x\in\mathbb R}
\left|Q_{\theta_d,d}^\xi(S_d-\beta_d\le x)
-\Phi(x/\sqrt{V_a^\xi})\right|\longrightarrow0.
\tag{M14}
\]
这里不对 \(\beta_d\) 加收敛、有界或趋零假设。

**证明：参数平均控制。** Cauchy–Schwarz 给
\(d^{-1}\sum_i|\mu_i|\to0\)。sigmoid 自旋函数的均值定理给
\[
|f_{i,d}(z)-\tanh(az/2)|
\le\tfrac12(|\mu_{i,d}|+|s_{i,d}-a||z|).
\]
对 z 积分并对 i 平均，右侧趋零。函数绝对值至多 1，因此平方的平均 L1 差也趋零。另一方面
\(|b_{i,d}|\le|\mu_{i,d}|/2\)，因为同一尺度在 \(\mu=0\) 时均值为零。所以
\[
\begin{aligned}
\frac1d\sum_iE f_{i,d}(Z)^2&\to v(a),&
\frac1d\sum_i b_{i,d}^2&\to0,\\
\frac1d\sum_i\operatorname{Var}(f_{i,d}(Z))&\to v(a),&
\frac1d\sum_iE f'_{i,d}(Z)&\to\alpha(a).
\end{aligned}
\tag{M15}
\]
最后一个收敛由
\[
|f'_{i,d}(z)-\tfrac a2\operatorname{sech}^2(az/2)|
\le\tfrac12|s_{i,d}-a|
+\tfrac b2(|\mu_{i,d}|+|s_{i,d}-a||z|)
\]
积分得到；同时用了 Gaussian 分部积分
\(E[\frac a2\operatorname{sech}^2(aZ/2)]=\alpha(a)\)。
空间导数统一满足 \(|f'_{i,d}|\le b/2\)、\(|f''_{i,d}|\le b^2/2\)。

**独立 Gaussian 行。** 取每行 iid 标准正态 \(Z_i\)，定义
\[
A_d=\frac1{\sqrt d}\sum_i(f_{i,d}(Z_i)-b_{i,d}),
\qquad W_d=\frac1{\sqrt d}\sum_i Z_i .
\]
任意固定线性组合的三阶矩 Lyapunov 余项为 \(O(d^{-1/2})\)，因为 f 有界且 Gaussian 三阶绝对矩有限。由 (M15) 和分部积分，
\[
(A_d,W_d)\Rightarrow N\!\left(0,
\begin{pmatrix}v(a)&\alpha(a)\\\alpha(a)&1\end{pmatrix}\right).
\tag{M16}
\]
也可直接对独立行的特征函数逐项展开；不要求参数逐坐标一致收敛。

**负族相关修正。** 精确表示
\[
U_i=\gamma_dZ_i-\lambda_d\bar Z_d,\qquad
\gamma_d=\sqrt{1+\rho_d},\quad
\lambda_d=\gamma_d-\sqrt{1-c},\quad
\bar Z_d=d^{-1}\sum_iZ_i .
\]
设 \(\lambda=1-\sqrt{1-c}\)。有
\(\sum_i(U_i-Z_i)^2=O_p(1)\)，因为
\[
\sum_i(U_i-Z_i)^2
\le2(\gamma_d-1)^2\sum_iZ_i^2+2\lambda_d^2d\bar Z_d^2.
\]
因此各 Taylor 二阶余项之和除以 \(\sqrt d\) 为 \(o_p(1)\)。
含 \(\gamma_d-1=O(d^{-1})\) 的一阶项也为 \(o_p(1)\)。
有界独立三角阵 LLN 与 (M15) 给
\(d^{-1}\sum_if'_{i,d}(Z_i)\to\alpha(a)\) 依概率。故
\[
\frac1{\sqrt d}\sum_i(f_{i,d}(U_i)-b_{i,d})
=A_d-\lambda\alpha(a)W_d+o_p(1)
\Rightarrow N(0,v(a)-c\alpha(a)^2).
\tag{M17}
\]
用到 \(2\lambda-\lambda^2=c\)。

**正族相关修正。** 取独立于 Z 行的标准正态 W，令
\(U_i=\sqrt{1-\rho_d}Z_i+\sqrt{\rho_d}W\)。
同样的位移平方和、Taylor 及导数 LLN 给
\[
\frac1{\sqrt d}\sum_i(f_{i,d}(U_i)-b_{i,d})
=A_d+\sqrt c\,\alpha(a)W+o_p(1)
\Rightarrow N(0,v(a)+c\alpha(a)^2).
\tag{M18}
\]
A_d 与 W 对每个 d 独立。

**条件 Bernoulli 噪声。** 令
\(\epsilon_i=Y_i-f_{i,d}(U_i)\)。给定全部 Gaussian 变量，它们独立、中心且绝对值至多 2。\(f_{i,d}^2\) 的空间 Lipschitz 常数至多 b；由位移平方和及 Cauchy–Schwarz，
\[
\frac1d\sum_i|f_{i,d}(U_i)^2-f_{i,d}(Z_i)^2|=o_p(1).
\]
再由独立有界行 LLN 与 (M15)，条件平均方差趋于 \(1-v(a)\)。
对固定 t，逐项条件特征函数展开并取乘积给
\[
E\!\left[e^{itd^{-1/2}\sum_i\epsilon_i}\mid\mathcal G_d\right]
\longrightarrow e^{-t^2(1-v(a))/2}
\quad\text{依概率且在 L1 中}.
\tag{M19}
\]
三阶余项之和为 \(O(d^{-1/2})\)，取对数的二阶余项为 \(O(d^{-1})\)；L1 收敛由模长至多 1 得到。乘任何模长至多 1 的 Gaussian 可测特征函数再平均，可知该噪声极限与 (M17)/(M18) 的极限独立。两项相加给 (M14) 的弱收敛。由于 \(\alpha(a)^2\le v(a)<1\) 且 \(c<1\)，极限方差严格为正。连续正态 CDF 下，弱收敛推出全实轴 CDF 一致收敛：在大紧区间上取有限网格并利用单调性，区间外用两端尾概率控制即可。\(\square\)

**漂移的两个情形。** 若某子序列上 \(\beta_d\to\beta\in\mathbb R\)，则
\(S_d\Rightarrow N(\beta,V_a^\xi)\)；若
\(\beta_d\to+\infty\) 或 \(-\infty\)，中心化变量的紧性使
\(S_d\to+\infty\) 或 \(-\infty\) 依概率。任意漂移序列可再取有限极限或单向发散的子序列。下文使用 (M14) 的全阈值一致形式，因此无需为候选选择某个漂移极限。

**集合上的统一量词。** 固定确定性的 \(K_d,\varepsilon_d\)，满足
\(\log K_d=o(d)\)、\(\varepsilon_d=o(d)\)，令
\(\mathcal A_d=\mathcal A_d(\varepsilon_d)\)。则
\[
\eta_d:=
\sup_{\theta\in\mathcal A_d}\sup_{x\in\mathbb R}
\left|Q_{\theta,d}^\xi(S_d-\beta_d(\theta)\le x)
-\Phi(x/\sqrt{V_a^\xi})\right|\longrightarrow0.
\tag{M20}
\]
否则可选择一条违反收敛的确定性参数子序列；它由 (M10) 满足 (M13)，与引理矛盾。这是对整个近优集合的一致结论，不只是先选定一个参数再证明点态收敛。

## 5. 预先固定的两个尾事件保持预测障碍

对一般 \(s>0\)，沿用函数记号
\(\alpha(s)=E[Z\tanh(sZ/2)]\)。期望内微分给
\(\alpha'(s)=\tfrac12E[Z^2\operatorname{sech}^2(sZ/2)]>0\)。
置
\[
\bar\alpha_\star=\tfrac12[\alpha(s_H^\star)+\alpha(s_L^\star)]
>\alpha(a),\qquad
V_\star^\xi=1+\xi c\bar\alpha_\star^2,
\]
\[
T_d^+=\{S_d>1\},\quad T_d^-=\{S_d<-1\},\qquad
r_\star^\xi=1-\Phi(1/\sqrt{V_\star^\xi}),\quad
r_a^\xi=1-\Phi(1/\sqrt{V_a^\xi}).
\tag{M21}
\]
事件 \(T_d^\pm\) 不依赖拟合参数、真实组或 bank，是事前固定的两个事件。
教师的固定两尺度 CLT 给
\[
P_d^{\star,\xi}(T_d^+)=P_d^{\star,\xi}(T_d^-)=:r_d^\star
\longrightarrow r_\star^\xi.
\]
该 CLT 可由第4节同一证明得到：\(\mu=0\) 下两个独立行二阶矩的平均为
\([v(s_H^\star)+v(s_L^\star)]/2\)，导数平均为
\(\bar\alpha_\star\)，Bernoulli 噪声仍将前者消去。等号来自有限 d 精确的全局翻转对称性，不需要在边界随机破平。

因为正态右尾在正方差上严格增加，
\[
r_\star^-<r_a^-,\qquad r_\star^+>r_a^+,\qquad
\Delta_\xi:=|r_\star^\xi-r_a^\xi|>0 .
\tag{M22}
\]
对候选令 \(q_d^\pm(\theta)=Q_{\theta,d}^\xi(T_d^\pm)\)，定义固定两事件的平均超额评分
\[
\begin{aligned}
\mathcal E_{\rm Br}^{\rm pair}(\theta)
&=\tfrac12[(q_d^+-r_d^\star)^2+(q_d^--r_d^\star)^2],\\
\mathcal E_{\log}^{\rm pair}(\theta)
&=\tfrac12[\operatorname{kl}(r_d^\star\Vert q_d^+)
+\operatorname{kl}(r_d^\star\Vert q_d^-)] .
\end{aligned}
\tag{M23}
\]
可将它解释为先独立公平选取一个事件索引，再评价该事件的标量 Brier 或 binary log-loss；不是单概念 LOO，也不是看到结果后才挑一个事件报告。

**定理 4（全部均值自由近优点的预测失真）。** 对两种符号均有
\[
\begin{aligned}
\liminf_d\inf_{\theta\in\mathcal A_d}
\mathcal E_{\rm Br}^{\rm pair}(\theta)&\ge\tfrac12\Delta_\xi^2>0,\\
\liminf_d\inf_{\theta\in\mathcal A_d}
\mathcal E_{\log}^{\rm pair}(\theta)
&\ge\tfrac12\operatorname{kl}(r_\star^\xi\Vert r_a^\xi)>0,\\
\liminf_d\inf_{\theta\in\mathcal A_d}
D(P_d^{\star,\xi}\Vert Q_{\theta,d}^\xi)
&\ge\operatorname{kl}(r_\star^\xi\Vert r_a^\xi)>0 .
\end{aligned}
\tag{M24}
\]
常数只是明确的正下界，不声称是最优常数。

**证明。** 用 (M20)，两条候选尾概率都与
\(N(\beta_d,V_a^\xi)\) 的对应尾概率相差至多 \(\eta_d\)。严格小于的 CDF 左极限也有同一界，因为正态 CDF 连续。

正族中，若 \(\beta_d\ge0\)，左尾 \(q_d^-\le r_a^++\eta_d\)；若
\(\beta_d\le0\)，右尾 \(q_d^+\le r_a^++\eta_d\)。
教师的两个尾概率都趋向更大的 \(r_\star^+\)。因此至少一个预先固定事件的绝对偏差不小于
\(\Delta_+-o(1)\)，且其 binary KL 不小于
\(\operatorname{kl}(r_\star^+\Vert r_a^+)-o(1)\)。

负族中，若 \(\beta_d\ge0\)，右尾 \(q_d^+\ge r_a^--\eta_d\)；若
\(\beta_d\le0\)，左尾 \(q_d^-\ge r_a^--\eta_d\)。
教师的两个尾概率都趋向更小的 \(r_\star^-\)，故得到同样的偏差和 KL 下界。
这些 \(o(1)\) 在近优集合上一致。KL 的端点比较用
\(\partial_q\operatorname{kl}(p\Vert q)=(q-p)/[q(1-q)]\)；被比较的 \(r_a^\xi,r_\star^\xi\) 均严格位于 (0,1)。若另一侧候选概率趋近 0 或 1，只会增大相应 KL，不需要把所有候选概率限制在内点。

两事件平均评分中另一个非负项可以丢弃，得到前两行。每个单事件都是同一 \(P,Q\) 的固定粗化，DPI 给 joint KL 至少为两项 binary KL 的最大值，因而最后一行没有额外的 \(1/2\)。证明不要求固定哪一个事件是较差者。\(\square\)

特别地，结论覆盖任意实际总体极小点序列，也覆盖在 \(\mathcal A_d\) 中随机选择的模型：集合上的一致下界逐个所选模型适用。若算法只以概率进入该集合，则必须另外给出该概率，不能自动视作任意训练程序的结论。

## 6. 负族单事件仍有效；正族单事件确有补偿

考虑旧中央事件
\(C_d=\{|S_d|\le1\}\)，并置
\[
p_\star^\xi=2\Phi(1/\sqrt{V_\star^\xi})-1,\qquad
p_a^\xi=2\Phi(1/\sqrt{V_a^\xi})-1.
\tag{M25}
\]

**负族的统一单事件结论。** 正态密度关于零对称且随绝对值递减，所以固定长度区间在居中时具有最大概率。由 (M20)，对任意漂移、整个近优集合，
\[
Q_{\theta,d}^-(C_d)\le p_a^-+2\eta_d,\qquad p_\star^->p_a^-.
\]
因此其单事件 Brier、binary log-loss 及 joint KL 分别有一致下界
\((p_\star^--p_a^-)^2\)、
\(\operatorname{kl}(p_\star^-\Vert p_a^-)\)、
\(\operatorname{kl}(p_\star^-\Vert p_a^-)\)，均指 \(\liminf_d\inf_{\theta\in\mathcal A_d}\)。这个结论包括有界、发散和振荡漂移；不把固定 a 模型当作独立模型。

**命题 5（正族单中央事件的真实补偿边界）。** 对正族教师，存在固定 \(t_0>0\)，使
\[
s_i=a,\qquad \mu_i=t_0/\sqrt d\quad(1\le i\le d)
\tag{M26}
\]
的候选最终属于 (M2)，且其中央事件概率趋于 \(p_\star^+\)。对任何
\(\log K_d=o(d)\)，这些候选的 raw 总体目标 regret 为 \(o(d)\)。
它们因此反驳“所有 \(o(d)\) 近优点的正族单中央事件都有原来的正下界”；不反驳定理4的两事件或 joint KL 下界。

**证明。** 记
\[
\tau_a=\left.\partial_\mu E\tanh((\mu+aZ)/2)\right|_{\mu=0}
=\tfrac12E\operatorname{sech}^2(aZ/2)=\alpha(a)/a>0.
\]
对 (M26)，精确漂移满足
\(\beta_d=\sqrt d\,E\tanh((t_0/\sqrt d+aZ)/2)\to\tau_a t_0\)。
于是引理3给
\(S_d\Rightarrow N(\tau_at_0,V_a^+)\)。
函数
\[
j_a(t)=
\Phi((1-\tau_at)/\sqrt{V_a^+})
-\Phi((-1-\tau_at)/\sqrt{V_a^+})
\]
在 \(t\ge0\) 上连续，从 \(j_a(0)=p_a^+>p_\star^+\) 严格下降至零。因此有唯一 \(t_0>0\) 使 \(j_a(t_0)=p_\star^+\)。

由 \(\partial_\mu L(0,a)=0\)，Taylor 展开给
\[
\Psi_d(t_0\mathbf1/\sqrt d,a\mathbf1)-d\ell(a)
\longrightarrow \tfrac12\partial_{\mu\mu}L(0,a)t_0^2
=\frac{\alpha(a)}{4a}t_0^2>0.
\tag{M27}
\]
又所有候选的 F 至少为 \(d\ell(a)-t_{d,K}\)，该特定候选的 F 至多为其 \(\Psi_d\)。故
\[
0\le F(\theta_d)-\min_{\Theta_d^M}F
\le t_{d,K}+O(1)=o(d).
\tag{M28}
\]
这正给出所述的近优尺度，而不提供 \(o(1)\) regret。K=1 时真实 regret 恰趋于 (M27) 的严格正常数，尤其不能将本反例说成精确总体极小点或误差趋零的近最优点。\(\square\)

这里“补偿”只指一个固定对称中央事件。该候选具有非零标准化漂移，左右尾被不同方向改变；定理4仍保证它的平均两事件风险与 joint KL 不消失。没有证成完整分布修复，也没有否定正族精确极小点仍可能保有单中央事件障碍；后者本文件未判定。

## 7. 同一均值自由候选类上的正族修复

正族每个候选写成
\[
\eta_i=\mu_i+s_i\{\sqrt{\rho_d}Z+\sqrt{1-\rho_d}E_i\},
\]
其中 Z 与全部 \(E_i\) iid 标准正态。逐坐标精确积分私噪声，定义
\[
h_{\theta,i}(z)=
E_E\sigma(\mu_i+s_i[\sqrt{\rho_d}z+\sqrt{1-\rho_d}E]),\qquad
\mathsf P_{\theta,z}(y)=
\prod_i\operatorname{Ber}((y_i+1)/2;h_{\theta,i}(z)).
\tag{M29}
\]
对每个候选都有 \(Q_{\theta,d}^+=E_Z\mathsf P_{\theta,Z}\)。
取 iid 公共因子 \(Z_1,\ldots,Z_K\)，bank 独立于标签，用
\(\widehat Q_{\theta,K}^{\rm col}=K^{-1}\sum_k\mathsf P_{\theta,Z_k}\)
定义总体评分 \(F^{\rm col}=E[-\log\widehat Q^{\rm col}(Y)]\)。
这是 private-noise marginalization/private-RB，不是负族 count-RB，也不是确定性量化评分。

**定理 6（同模型恢复在均值自由类中保留）。** 对同一正族教师 (M4)、同一 (M2)，\(F^{\rm col}\) 在固定 d,K 达到最小值。任何总误差不超过 \(\varepsilon\) 的 col 总体近优点满足
\[
D(P_d^{\star,+}\Vert Q_{\widetilde\theta,d}^+)
\le \varepsilon+\frac{c}{1-c}\omega_K,\qquad K\ge2,
\tag{M30}
\]
其中
\[
\omega_K=\frac{12\log K}{\sqrt K}
+6\sqrt{\frac2\pi}\frac{\sqrt{2\log K}+1/\sqrt{2\log K}}K
\longrightarrow0.
\tag{M31}
\]

**证明：同 Q 与 kernel 传输。** col 是同一候选 raw likelihood 估计器给定公共因子 bank 的条件期望，所以无偏且
\(R(P^\star;\theta)\le F^{\rm col}(P^\star;\theta)\le F^{\rm raw}(P^\star;\theta)\)。
固定候选，两个公共值 z,t 的条件 Gaussian 协方差都是
\((1-\rho_d)D_s^2\)，其均值差是
\(\sqrt{\rho_d}D_s\mathbf1(z-t)\)。可训练均值 \(\mu\) 在**同一候选**的这项差里完全消去。因此 Gaussian KL 及共同 logistic 观察通道的 DPI 给
\[
D(\mathsf P_{\theta,z}\Vert\mathsf P_{\theta,t})
\le\frac{d\rho_d}{2(1-\rho_d)}(z-t)^2
\le\frac c{1-c}(z-t)^2 .
\tag{M32}
\]
对任意有有限二阶矩的混合分布 \(\nu,\nu'\)，用任意共同耦合 \(\gamma(dz,dt)\) 构造两个联合分布
\(\gamma\mathsf P_{\theta,z}\) 与 \(\gamma\mathsf P_{\theta,t}\)。
KL 链式法则后边际化、再对耦合取下确界，得到
\[
D(\nu\mathsf P_\theta\Vert\nu'\mathsf P_\theta)
\le\frac c{1-c}W_2^2(\nu,\nu').
\tag{M33}
\]
不比较连续 mixing law 与离散经验 mixing law 的直接 KL。

**经验 W2 的充分上界。** 令 \(\nu=N(0,1)\)，\(\widehat\nu_K\) 是 K 个 iid 样本的经验分布。将两者截断到 \([-t,t]\)，首尾截断耦合的期望平方误差各为
\(e(t)=E(|Z|-t)_+^2\)。中间有
\[
W_2^2(\nu_t,\widehat\nu_{K,t})
\le2tW_1(\nu_t,\widehat\nu_{K,t}),\qquad
EW_1\le\int_{-t}^t\frac{dx}{2\sqrt K}
=\frac t{\sqrt K}.
\]
第一式用一维分位数耦合，第二式用一维 CDF 表示及经验 Bernoulli CDF 的方差不超过 \(1/(4K)\)。W2 三角不等式和三项平方和界给
\[
EW_2^2(\nu,\widehat\nu_K)
\le6e(t)+6t^2/\sqrt K.
\]
而
\(e(t)=2[(1+t^2)\Phi(-t)-t\varphi(t)]
\le2\varphi(t)(t+t^{-1})\)。
取 \(t=\sqrt{2\log K}\) 即得 (M31) 的上界。这是保守充分率，不称最优率。

在真参数处逐 bank 使用 (M33) 后取期望：
\[
0\le F^{\rm col}(Q_{\theta^\star};\theta^\star)-H(Q_{\theta^\star})
=E D(Q_{\theta^\star}\Vert\widehat Q_{\theta^\star,K}^{\rm col})
\le\frac c{1-c}\omega_K.
\]
均值和尺度的常数均未进入右侧。对固定教师，这是**真参数处**的 bias 上界，不是错误候选的统一 Jensen-gap 上界。用所有候选的 Jensen 下界以及真参数可行性，
\[
R(P^\star;\widetilde\theta)-H(P^\star)
\le F^{\rm col}(P^\star;\widetilde\theta)-H(P^\star)
\le\varepsilon+F^{\rm col}(P^\star;\theta^\star)-H(P^\star),
\]
得到 (M30)。
最后，kernel likelihood 连续且严格正；条件 Jensen 给其负 log 一个与参数无关的可积上界
\(d(\log2+M)+bd(\sqrt{\rho_d}|z|+\sqrt{1-\rho_d}E|Z|)\)。
支配收敛与紧性给固定 d,K 的 col 极小值存在。\(\square\)

**同教师、同更大模型类的对照。** 在正族取共同 \(K_d=d\)、共同确定性总容差 \(\varepsilon_d\to0\)。raw 的全部近优点满足 (M24) 的固定两事件和总 KL 正下界；col 的全部近优点由 (M30) 总 KL 趋零。任何事件序列的 binary log-loss 超额由 DPI 趋零，其 scalar Brier 超额也由
\(\operatorname{kl}(p\Vert q)\ge2(p-q)^2\) 趋零。两边均包括实际总体极小点，均值、逐坐标尺度与 covariance floor 完全相同。

raw 障碍可容许更宽的 \(\varepsilon_d=o(d)\)，col 一致性需要
\(\varepsilon_d=o(1)\)；上面的共同容差选择避免了混用。结论不意味着相同 K 的运行时间相同，私噪声积分及梯度仍有成本。本文件不提供有限标签或有限 bank 的一致估计、优化收敛或算法复杂度保证。

## 8. 范围、未决问题与来源

- 释放的是任意有界 logit 均值和有固定正上下界的逐坐标尺度，已知 \(R_d^\xi(c)\) 仍不变。没有证明任意 covariance/自由 Cholesky 的全部近优点也失真，亦没有同时重证可训练 c 的扩展。
- 教师的零观察边际是主项以 \(\mu=0\) 为中心的关键。非零边际时线性项改变参数定位；本文件不作直接推广。
- 平均 \(\mu_i^2\to0\) 只用于中心化 CLT 的 Cesàro 条件。精确漂移 \(\beta_d\) 一直保留，可能有界非零或发散；没有从平均参数收缩偷换成整个高维分布的 KL 收敛。
- 正族单中央事件的补偿是 \(o(d)\) 近优反例，不是 \(\varepsilon=o(1)\) 或实际极小点的反例。两尾事件的平均评分事前固定，不由 oracle 挑选报告终点。
- 负族单事件、两族两事件、joint KL 和正族 col 风险单位不同，不能统一称平均单标签或 LOO 风险。没有新的自然任务成功结论。
- 所有结论为人口目标定理；K>1 没有有限维精确贴边或均值恰为零证明。所有常数界都可能在有限 d 过粗。

标准工具采用 Gaussian log-Sobolev/Herbst、Lyapunov 三角阵 CLT、Gaussian 分部积分、条件特征函数、KL chain rule/DPI、Jensen 与一维 W2/CDF 耦合。Gaussian 集中归一化可见 Michel Ledoux (1999), *Concentration of measure and logarithmic Sobolev inequalities*, Séminaire de Probabilités XXXIII, pp.120–216，尤其引言 pp.123–124 与 §2.3：[NUMDAM 一手全文](https://www.numdam.org/item/SPS_1999__33__120_0.pdf)。本文件不把这些一般工具宣称为原创。

新增内容、前序证明的边界及读取哈希见同目录 MEAN_SOURCE_NOTE.md。正文关键推导已写出，不要求读者用旧仓库文件补齐均值自由的证明。新意与重要性仍须按整条主结果及直接文献近邻审查。
