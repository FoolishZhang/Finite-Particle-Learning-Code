# 任意有界谱相关结构下的发射噪声障碍

本文件给出新的解析结论，不改写旧冻结。它允许候选 Gaussian 的全部相关结构变化，并允许逐坐标有界均值和尺度训练。证明不要求候选潜在均值总和满足 CLT；关键是条件独立 Bernoulli 发射留下的噪声限制了区间集中概率。结果带有明确的教师信号条件，并非“所有相关结构变化都不能补偿”的无条件主张。没有新随机观察、拟合或科学数值积分。

全部 log 为自然底；raw 优化误差和 joint KL 都是完整向量的总 nats。模型预测指对所选参数精确积分后的分布，而非另一个有限粒子部署估计器。

## 1. 任意协方差候选类及比较点

固定
\[
0<a<b<\infty,\qquad M>0,\qquad
0<\nu\le a^2\le\Lambda<\infty .
\]
沿偶数 \(d\ge4\)，令
\[
\Theta_d^\nu=
\left\{(\mu,\Sigma):
|\mu_i|\le M,\quad
\nu I\preceq\Sigma\preceq\Lambda I,\quad
a^2\le\Sigma_{ii}\le b^2\ (1\le i\le d)\right\}.
\tag{N1}
\]
这里 \(\Sigma\) 是完整、可训练的协方差矩阵，不指定其特征向量、符号、等相关结构或因子载荷。统一谱上界 \(\Lambda\) 与 d 无关。令
\[
s_i=\sqrt{\Sigma_{ii}},\qquad
\eta\sim N(\mu,\Sigma),\qquad
Q_\theta(y)=E_\eta\prod_{i=1}^d\sigma(y_i\eta_i),
\quad y\in\{-1,1\}^d .
\tag{N2}
\]
给定 \(\eta\)，所有 \(Y_i\) 独立，\(E[Y_i\mid\eta]=\tanh(\eta_i/2)\)。

本类含比较点
\(\theta_a=(0,a^2I)\)，每个模型均严格正定且有固定正谱下限 \(\nu\)。
坐标标准差下限 a 与谱下限 \(\nu\) 是不同条件，二者不能互换。特别地，自由 Cholesky 对角元素的下限本身不等于 (N1) 的两个谱界。

以下参数定位只要求教师 \(P_d^\star\) 的观察边际为零：
\[
E_{P_d^\star}Y_i=0\quad\text{对每个 }i.
\tag{N3}
\]
预测后果使用第7节给出的负相关 Gaussian-logistic 教师，并另要求它属于 (N1)，从而结论发生在正确指定的模型类内。

取 K 个 iid prior 粒子，完整 bank 独立于当前标签 \(Y\sim P_d^\star\)。定义
\[
\widehat Q_{\theta,K}^{\rm raw}(y)=
\frac1K\sum_{k=1}^K\prod_i\sigma(y_i\eta_{ki}),\qquad
F_{d,K}(\theta)=E_{Y,\eta_{1:K}}[-\log\widehat Q_{\theta,K}^{\rm raw}(Y)].
\tag{N4}
\]
K 是对数内部的粒子数；本文件不涉及训练标签数或用于估计 F 的独立 bank 平均数。

## 2. 统一主项仍不依赖相关结构

记
\[
L(\mu,s)=E_Z\log[2\cosh((\mu+sZ)/2)],\quad
\ell(s)=L(0,s),\quad
\Psi_d(\theta)=\sum_iL(\mu_i,s_i),\quad Z\sim N(0,1).
\]
固定标签 y 的粒子损失为
\[
h_y(\eta)=\sum_i\log[2\cosh(\eta_i/2)]-\tfrac12y^\top\eta,\qquad
E_\eta h_y=\Psi_d(\theta)-\tfrac12y^\top\mu .
\tag{N5}
\]
每个梯度分量 \(\partial_{\eta_i}h_y\) 的绝对值至多1，所以
\(h_y(\mu+\Sigma^{1/2}Z)\) 对标准 Gaussian Z 是
\(\sqrt{\Lambda d}\)-Lipschitz。

标准 Gaussian log-Sobolev/Herbst 不等式给
\[
\log E\exp\{t(h_y-Eh_y)\}\le t^2\Lambda d/2 .
\]
对 K 个粒子作指数最大值界，并使用
\[
\min_kh_y(\eta_k)
\le-\log\left(K^{-1}\sum_ke^{-h_y(\eta_k)}\right)
\le K^{-1}\sum_kh_y(\eta_k),
\]
得到对每个 y，
\(0\le E_\eta h_y-E_{\rm bank}[-\log\widehat Q(y)]
\le\sqrt{2\Lambda d\log K}\)。
对 Y 平均后由 (N3) 消去线性项。因此整个 (N1) 上
\[
0\le\Psi_d(\theta)-F_{d,K}(\theta)
\le t_{d,K}:=\sqrt{2\Lambda d\log K}.
\tag{N6}
\]
K1 时 \(t_{d,1}=0\)。非零教师边际会留下
\(-\frac12\sum_i\mu_iE Y_i\)，不能继续使用这里以零均值为中心的定位。

令
\[
\kappa_a=\ell'(a)=\tfrac12E[Z\tanh(aZ/2)]>0,\qquad
\kappa_M=\frac{P(|Z|\le1)}4
\operatorname{sech}^2((M+b)/2)>0 .
\tag{N7}
\]
因为 L 关于 \(\mu\) 为偶函数，且在固定参数矩形上
\(\partial_{\mu\mu}L\ge\kappa_M\)，同时 \(\ell''(s)>0\)，所以
\[
L(\mu_i,s_i)-\ell(a)
\ge\kappa_a(s_i-a)+\tfrac12\kappa_M\mu_i^2.
\tag{N8}
\]

**定理1（一般协方差的全部总体近优点）。**
对固定 d,K，F 在 (N1) 达到最小值。对任意 \(\varepsilon\ge0\)，若
\[
\mathcal A_d(\varepsilon)=
\{\theta\in\Theta_d^\nu:F_{d,K}(\theta)
\le\min_{\vartheta\in\Theta_d^\nu}F_{d,K}(\vartheta)+\varepsilon\},
\]
则每个 \(\theta\in\mathcal A_d(\varepsilon)\) 都有
\[
\kappa_a\,\frac1d\sum_i(s_i-a)
+\frac{\kappa_M}{2d}\sum_i\mu_i^2
\le\delta_{d,K,\varepsilon}:=
\sqrt{\frac{2\Lambda\log K}{d}}+\frac{\varepsilon}{d}.
\tag{N9}
\]

**证明。** 固定 d 下，(N1) 是有限维闭有界集合，故紧。
对称正半定矩阵平方根连续，因此可用共同 Z bank 写出参数连续的粒子。逐 bank 评分非负，并由
\[
d(\log2+M)+\frac{\sqrt{\Lambda d}}K\sum_{k=1}^K\|Z_k\|_2
\]
统一支配。该随机上界可积，故 F 连续并达到最小值。
再比较可行点 \(\theta_a\)：
\[
\Psi_d(\theta)-t_{d,K}
\le F(\theta)\le F(\theta_a)+\varepsilon
\le d\ell(a)+\varepsilon.
\]
代入 (N8) 后除以 d 得 (N9)。\(\square\)

这里没有识别相关矩阵，也没有由尺度和均值的平均收缩直接宣称观察风险收敛。后面的证明专门控制发射噪声的区间集中概率。

## 3. 平均条件发射方差的期望与集中

置
\[
g(x)=\operatorname{sech}^2(x/2),\qquad
\tau_a^2=E g(aZ)\in(0,1),\qquad
V_d(\eta)=\frac1d\sum_i g(\eta_i).
\tag{N10}
\]
\(\tau_a^2\) 在本文件表示一个方差；它不是旧均值补偿证明中均值响应导数的同名记号。
直接求导可得
\[
g'(x)=-g(x)\tanh(x/2),\quad
|g'(x)|\le L_g:=\frac{2}{3\sqrt3},\quad
g''(x)=g(x)-\tfrac32g(x)^2,\quad |g''(x)|\le\tfrac12 .
\tag{N11}
\]

Gaussian 边际给
\(E V_d=d^{-1}\sum_i E g(\mu_i+s_iZ)\)，与非对角 covariance 无关。
对固定 s，\(E g(\mu+sZ)\) 关于 \(\mu\) 为偶函数。用 (N11) 的二阶界，
\[
|E g(\mu+sZ)-E g(sZ)|\le\tfrac14\mu^2.
\]
尺度方向则有
\[
|E g(sZ)-E g(aZ)|
\le L_g E|Z|(s-a).
\]
令
\[
C_\tau=
\max\left\{\frac{L_g\sqrt{2/\pi}}{\kappa_a},
\frac1{2\kappa_M}\right\},\qquad
v_d^{\rm P}=\frac{\Lambda L_g^2}{d}=\frac{4\Lambda}{27d}.
\tag{N12}
\]
将这两个误差作为 (N9) 左侧两项的非负线性组合，得到整个近优集合上的明确界
\[
|E_\theta V_d-\tau_a^2|\le C_\tau\delta_{d,K,\varepsilon}.
\tag{N13}
\]
使用关于均值的偶性使这里是 \(O(\delta)\)，而不是只靠 Lipschitz/Cauchy–Schwarz 得到的 \(O(\sqrt\delta)\)。

**引理2（任意有界谱 covariance 的噪声方差集中）。**
对整个 (N1)，
\[
\operatorname{Var}_\theta(V_d)
\le E[\nabla V_d(\eta)^\top\Sigma\nabla V_d(\eta)]
\le v_d^{\rm P}.
\tag{N14}
\]
因此，对 \(\theta\in\mathcal A_d(\varepsilon)\)，
\[
E_\theta(V_d-\tau_a^2)^2
\le v_d^{\rm P}+(C_\tau\delta_{d,K,\varepsilon})^2 .
\tag{N15}
\]

**证明。** Gaussian Poincaré 不等式可对
\(\eta=\mu+\Sigma^{1/2}Z\) 使用。每个
\(\partial_{\eta_i}V_d=g'(\eta_i)/d\)，故
\(\|\nabla V_d\|_2^2\le L_g^2/d\)，再用 \(\Sigma\preceq\Lambda I\) 得 (N14)。这也显示了谱上界为什么必须对 d 统一。Poincaré 的归一化可由 Gaussian log-Sobolev 对 \(e^{tV_d}\) 在 \(t=0\) 的二阶展开直接得到；V_d 有界，展开合法。方差加平方偏差给 (N15)。\(\square\)

给任意 \(x>0\)，(N13)–(N14) 还给
\[
\sup_{\theta\in\mathcal A_d(\varepsilon)}
P_\theta\!\left(V_d<\tau_a^2-C_\tau\delta_{d,K,\varepsilon}-x\right)
\le \min\{1,v_d^{\rm P}/x^2\}.
\tag{N16}
\]
若使用同一 Gaussian log-Sobolev/Herbst 结果，V_d 对标准 Gaussian 输入的 Lipschitz 平方也至多 \(v_d^{\rm P}\)，因此可将右端进一步替换为
\(\exp[-x^2/(2v_d^{\rm P})]\)。主结论只用较弱的 (N16) 已足够。

## 4. 条件 Berry–Esseen：控制区间概率而不只控制方差

给定任意有限 \(\eta\)，令
\[
m_i(\eta)=\tanh(\eta_i/2),\quad
v_i(\eta)=1-m_i(\eta)^2=g(\eta_i),\quad
T_d(\eta)=d^{-1/2}\sum_i m_i(\eta),
\]
\[
S_d=d^{-1/2}\sum_iY_i,\qquad
W_d=d^{-1/2}\sum_i(Y_i-m_i(\eta)).
\tag{N17}
\]
此时 \(S_d=T_d(\eta)+W_d\)。T_d 是任意 Gaussian 可测随机漂移；它不需要趋零、有界、满足 CLT，或与 V_d 独立。

使用以下标准独立、非同分布 Berry–Esseen 形式：若 X_i 独立、中心，\(B^2=\sum_iEX_i^2>0\)，则
\[
\sup_z\left|P\!\left(\frac{\sum_iX_i}{B}\le z\right)-\Phi(z)\right|
\le C_{\rm BE}\frac{\sum_iE|X_i|^3}{B^3}.
\tag{N18}
\]
\(C_{\rm BE}\) 为任一有效绝对常数，本文可保守取 \(C_{\rm BE}=1\)，不追求最优。该非同分布版本及常数范围见文末的一手来源；不是将 iid 版本误用于不同 Bernoulli 参数。

在 (N17) 中，给定 \(\eta\) 后各变量条件独立、中心。
写 \(p_i=(1+m_i)/2\)，直接对两个取值计算：
\[
E[|Y_i-m_i|^3\mid\eta]
=8p_i(1-p_i)[p_i^2+(1-p_i)^2]
=2v_i-v_i^2\le2v_i .
\tag{N19}
\]
而总条件方差为 \(dV_d>0\)，所以逐个 \(\eta\) 都有
\[
\sup_z\left|
P(W_d\le z\mid\eta)-\Phi(z/\sqrt{V_d})
\right|
\le\frac{2C_{\rm BE}}{\sqrt{dV_d}}.
\tag{N20}
\]

对任意固定半宽 \(h>0\) 和任意区间中心 t，从 CDF 两端相减得
\[
P(|S_d-t|\le h\mid\eta)
\le G_h(V_d)+\frac{4C_{\rm BE}}{\sqrt{dV_d}},
\qquad
G_h(v)=2\Phi(h/\sqrt v)-1 .
\tag{N21}
\]
理由是方差 v 的正态分布对固定长度 \(2h\) 的区间，在区间关于其均值居中时概率最大。条件漂移只平移区间，不改变这个最大值。闭区间左端使用 CDF 左极限；正态 CDF 连续，所以两端的误差仍各只有一次 (N20)，无需附加格点概率假设。

这一环节不能用“总方差至少为某个正数”替代。一般随机变量完全可以以概率 \(1-\epsilon\) 取零，以剩余概率取两个越来越远的点，在方差保持固定时令中央质量趋1。这里能排除这种推理漏洞，是因为给定 logits 后的独立 Bernoulli 和有定量的正态近似，而不是方差下界本身。

## 5. 有限维、全近优集合的集中函数上界

**定理3（任意相关结构和潜在漂移的统一区间界）。**
固定 d,K,\(\varepsilon\)，并令 \(\delta=\delta_{d,K,\varepsilon}\)。
对任意 \(h>0,x>0\)，只要
\[
v_0=\tau_a^2-C_\tau\delta-x>0,
\]
就有
\[
\sup_{\theta\in\mathcal A_d(\varepsilon)}
\sup_{t\in\mathbb R}Q_\theta(|S_d-t|\le h)
\le U_d(h,x):=
\min\left\{1,\,
G_h(v_0)+\frac{4C_{\rm BE}}{\sqrt{dv_0}}
+\frac{v_d^{\rm P}}{x^2}\right\}.
\tag{N22}
\]
若 \(v_0\le0\)，本式只保留平凡上界1，不将它用于正的风险结论。

**证明。** 在 \(V_d\ge v_0\) 的条件实现上，G_h 关于正方差递减，且 Berry–Esseen 项也随方差递减，所以 (N21) 至多为
\(G_h(v_0)+4C_{\rm BE}/\sqrt{dv_0}\)。在其余实现上使用概率上界1，再用 (N16)。所有常数、坏事件界和区间中心控制均与候选无关，因此可同时对整个近优集合和所有 t 取上确界。\(\square\)

这是一条直接的有限维上界；不是先假设每个候选有 CLT，再以点态收敛推测统一性。候选 covariance、潜在条件均值、无条件均值漂移和它们之间的依赖都不进入正态极限假设。

若 \(\log K_d=o(d)\)、\(\varepsilon_d=o(d)\)，则 \(\delta_d\to0\)。
例如固定取 \(x_d=(v_d^{\rm P})^{1/3}\)，则最终 \(v_0>0\)，并且
\[
\limsup_{d\to\infty}
\sup_{\theta\in\mathcal A_d(\varepsilon_d)}
\sup_{t\in\mathbb R}Q_\theta(|S_d-t|\le h)
\le G_h(\tau_a^2).
\tag{N23}
\]
还可显式写出收敛余项。若
\(C_\tau\delta_d+x_d\le\tau_a^2/2\)，令
\[
D_{h,a}=\frac{h}{\sqrt{2\pi}(\tau_a^2/2)^{3/2}} .
\]
因为 \(|G_h'(v)|\le D_{h,a}\) 对 \(v\ge\tau_a^2/2\) 成立，(N22) 给
\[
\sup_{\theta,t}Q_\theta(|S_d-t|\le h)
\le G_h(\tau_a^2)
+D_{h,a}C_\tau\delta_d
+(D_{h,a}+1)(v_d^{\rm P})^{1/3}
+4C_{\rm BE}\sqrt{\frac{2}{d\tau_a^2}} .
\tag{N24}
\]
上确界的 \(\theta\) 仍限于 \(\mathcal A_d(\varepsilon_d)\)。若右端超过1，实际使用时截为1。

可选的 Gaussian 集中强化：以
\(x_d=\sqrt{2v_d^{\rm P}\log d}\) 使用 (N16) 后的指数界，则 (N24) 中
\((D_{h,a}+1)(v_d^{\rm P})^{1/3}\) 可替换为
\(D_{h,a}\sqrt{2v_d^{\rm P}\log d}+d^{-1}\)。
因此在同样的正方差余量条件下，误差为
\(O(\delta_d+\sqrt{\log d/d})\)。这是现有 Gaussian 集中工具给出的充分界，不声称有限预算下 sharp。

## 6. 有限事件风险界与渐近分离

取固定的、无需学习任何组身份的中央事件
\[
C_{d,h}=\{|S_d|\le h\},\qquad
p_d^\star=P_d^\star(C_{d,h}),\quad q_\theta=Q_\theta(C_{d,h}).
\]
若定理3的有限上界 \(U=U_d(h,x)<p_d^\star\)，则每个近优候选都满足
\[
\begin{aligned}
\mathcal E_{\rm Brier}(C_{d,h};\theta)
&=(q_\theta-p_d^\star)^2\ge(p_d^\star-U)^2,\\
\mathcal E_{\log}(C_{d,h};\theta)
&=\operatorname{kl}(p_d^\star\Vert q_\theta)
\ge\operatorname{kl}(p_d^\star\Vert U),\\
D(P_d^\star\Vert Q_\theta)
&\ge\operatorname{kl}(p_d^\star\Vert U).
\end{aligned}
\tag{N25}
\]
第二行使用 binary KL 对 \(q<p\) 单调递减，第三行是事件粗化的 DPI。
若 \(U\ge p_d^\star\)，由这条比较只能报告下界0。
有限式中的 \(p_d^\star\) 是真实有限维概率，不能无误差地替换成下面的 CLT 极限。

**推论4（比发射噪声更集中的教师不可补偿）。**
设教师满足 (N3)、属于 (N1)，并且
\[
S_d\ \text{在 }P_d^\star\text{ 下}\Rightarrow N(0,V_\star),
\qquad 0<V_\star<\tau_a^2 .
\tag{N26}
\]
对任意固定 \(h>0\)，记
\[
p_\star=G_h(V_\star),\qquad p_{\rm noise}=G_h(\tau_a^2),
\qquad \Delta_h=p_\star-p_{\rm noise}>0.
\]
则在 \(\log K_d=o(d),\varepsilon_d=o(d)\) 下，
\[
\begin{aligned}
\liminf_d\inf_{\theta\in\mathcal A_d(\varepsilon_d)}
\mathcal E_{\rm Brier}(C_{d,h};\theta)&\ge\Delta_h^2>0,\\
\liminf_d\inf_{\theta\in\mathcal A_d(\varepsilon_d)}
\mathcal E_{\log}(C_{d,h};\theta)
&\ge\operatorname{kl}(p_\star\Vert p_{\rm noise})>0,\\
\liminf_d\inf_{\theta\in\mathcal A_d(\varepsilon_d)}
D(P_d^\star\Vert Q_\theta)
&\ge\operatorname{kl}(p_\star\Vert p_{\rm noise})>0 .
\end{aligned}
\tag{N27}
\]

**证明。** 正态极限在 \(\pm h\) 无原子，故 \(p_d^\star\to p_\star\)。
取第5节的确定性 \(x_d\)，其统一上界趋于 \(p_{\rm noise}<p_\star\)，于是对充分大 d 使用 (N25)，再用 binary KL 在这两个内点的连续性。\(\square\)

结论包括所有实际总体全局极小点。随机选择近优参数也不改变这个集合上一致的下界；若某个训练程序仅以一定概率进入近优集合，仍必须另外证明进入概率。这里没有 SGD、连续优化算法或自由 covariance 的有限样本学习保证。

## 7. 原负族教师满足所需 CLT 的自足核对

令 \(0<c<1\) 固定，两个真实尺度满足
\(a<s_L^\star<s_H^\star<b\)，各占一半。设
\[
\rho_d=\frac c{d-1},\qquad
\Sigma_d^\star=D_{s^\star}
[(1+\rho_d)I-\rho_d\mathbf1\mathbf1^\top]D_{s^\star},
\qquad \mu^\star=0 .
\tag{N28}
\]
保证它属于 ambient class 的一个明确充分选择是
\[
0<\nu\le\min\{a^2,(s_L^\star)^2(1-c)\},\qquad
\Lambda\ge (s_H^\star)^2(1+c/3).
\tag{N29}
\]
这使用 d≥4；对该教师无需正族的另一谱条件。也可使用更宽的 \(\Lambda=b^2(1+c)\)。
真实尺度和谱参数固定，因此该选择确实允许固定正谱 floor。

定义
\[
f_s(z)=\tanh(sz/2),\quad
\alpha(s)=E[Zf_s(Z)],\quad v(s)=Ef_s(Z)^2,\quad
\bar\alpha_\star=\frac{\alpha(s_H^\star)+\alpha(s_L^\star)}2,
\quad \bar v_\star=\frac{v(s_H^\star)+v(s_L^\star)}2 .
\]
取 iid 标准正态 \(Z_i\)，记
\[
A_d=d^{-1/2}\sum_if_{s_i^\star}(Z_i),\qquad
W_d=d^{-1/2}\sum_iZ_i .
\]
独立行的三阶矩统一有界，Lyapunov 三角阵 CLT 给
\[
(A_d,W_d)\Rightarrow
N\!\left(0,\begin{pmatrix}
\bar v_\star&\bar\alpha_\star\\
\bar\alpha_\star&1
\end{pmatrix}\right).
\tag{N30}
\]
协方差项等于 Gaussian 分部积分后的导数平均。

标准化 logits 的精确表示为
\[
U_i=\gamma_dZ_i-\lambda_d\bar Z_d,\qquad
\gamma_d=\sqrt{1+\rho_d},\quad
\lambda_d=\gamma_d-\sqrt{1-c},\quad
\bar Z_d=d^{-1}\sum_iZ_i .
\]
它的 covariance 是 (N28) 中括号内的矩阵。
有 \(\sum_i(U_i-Z_i)^2=O_p(1)\)，且
\(\gamma_d-1=O(d^{-1})\)。两种 \(f_s\) 的二阶导数统一有界，所以 Taylor 余项之和除以 \(\sqrt d\) 为 \(o_p(1)\)。
导数的独立行 LLN 给
\(d^{-1}\sum_if_{s_i^\star}'(Z_i)\to\bar\alpha_\star\)。
置 \(\lambda=1-\sqrt{1-c}\)，便得到教师的 latent 条件均值
\[
d^{-1/2}\sum_if_{s_i^\star}(U_i)
=A_d-\lambda\bar\alpha_\star W_d+o_p(1)
\Rightarrow N(0,\bar v_\star-c\bar\alpha_\star^2).
\tag{N31}
\]

教师的条件发射方差期望精确为 \(1-\bar v_\star\)。
(N14) 的 Poincaré 证明同样适用于教师，只用其固定谱上界，因此其 \(V_d\) 依概率趋于 \(1-\bar v_\star>0\)。
给定 Gaussian 变量，中心 Bernoulli 噪声的条件特征函数逐项展开给
\[
E\!\left[e^{itd^{-1/2}\sum_i(Y_i-f_{s_i^\star}(U_i))}
\mid\eta\right]
=\exp(-t^2V_d/2)+o(1),
\]
其中每个固定 t 的余项由 \(|Y_i-f_{s_i^\star}(U_i)|\le2\) 统一控制为 \(O_t(d^{-1/2})\)。因模长有界，该条件特征函数在 L1 中趋于
\(\exp[-t^2(1-\bar v_\star)/2]\)。
乘 Gaussian 可测的特征函数再平均，证明噪声极限与 (N31) 独立。故
\[
S_d\Rightarrow N(0,V_\star^-),\qquad
V_\star^-=1-c\bar\alpha_\star^2>0 .
\tag{N32}
\]
教师的零边际由奇函数对称性精确成立。至此推论4对原两尺度负族是自足的，不要求读者到旧文件补 CLT。

其额外信号条件可写成
\[
1-c\bar\alpha_\star^2<\tau_a^2
\quad\Longleftrightarrow\quad
c\bar\alpha_\star^2>v(a).
\tag{N33}
\]
这比旧固定相关矩阵结果所需的 \(\bar\alpha_\star>\alpha(a)\) 更强，不能删去或追认旧全部教师自动满足。

**条件并非空集，但不是普遍条件。**
当 s→∞，支配收敛给
\(\alpha(s)\to E|Z|=\sqrt{2/\pi}\)；当 a↓0，
\(\tau_a^2\to1\)。
对任意固定 c>0，先取充分小但固定的 a>0，再取两个充分大、不同且有限的真实尺度，即可使 (N33) 成立；随后固定更大的 b 和满足 (N29) 的正 \(\nu,\Lambda\)。
整个构造中这些常数都不依赖 d。
反之，若 \(\tau_a^2\le1-2c/\pi\)，这个特定两尺度负族不可能满足 (N33)。

对原正族教师，\(V_\star^+=1+c\bar\alpha_\star^2>1>\tau_a^2\)，本判据不成立。因此它不排除正族通过调整载荷而补偿尺度收缩，也不将这种补偿是否实际学到作为结论。证明只认证满足 (N26) 的集中型教师障碍。

## 8. PSD、严格正定与结论范围

1. 主类 (N1) 取固定 \(\nu>0\)，所以所有候选严格正定且参数集闭紧。零奇异点不是证明所需条件，实际总体极小值存在。
2. 将 \(\nu\) 设为0、保留其余闭约束时，参数集仍紧；Gaussian 平方根、集中和 Poincaré 可通过 \(\mu+\Sigma^{1/2}Z\) 使用，条件 Bernoulli 方差仍严格正，因此整条结论也覆盖这个 PSD 闭包。
3. 若只要求 \(\Sigma\succ0\) 却不设统一正谱下限，则这个严格正定集合一般不闭。所有定量界仍对以 infimum 定义的 \(\varepsilon\) 近优集合有效，只要 floor 比较点可行；\(\varepsilon>0\) 时该集合非空，但 \(\varepsilon=0\) 时不能无条件声称 infimum 在该开放类达到。若某个实际最小值确实存在，它当然也受界约束。
4. 若另加约束排除了所有 \(\mu=0,\Sigma_{ii}=a^2\) 的比较点，则 (N9) 不能直接照用。对于本文件的 ambient class，\((0,a^2I)\) 的可行性由参数条件明示保证。正确指定还须另核教师在类内。
5. 没有规定候选相关符号或因子结构，但必须有维数统一的谱上界、固定有界均值与标准差、零教师观察边际，以及给定 logits 后的独立 Bernoulli 发射。相关结构任意不等于 covariance 无界或任意观察通道。
6. 本结果限制的是 raw 评分近优点进入的低尺度子类，不是说完整候选类无法表示教师。教师本就在完整类内。没有将正族旧 private-RB 恢复界自动推广为这里任意 covariance 类或负族的恢复算法。
7. 方差比较 (N33) 是充分而非必要条件。条件不成立、或有限式 (N22) 等于1，只表示该界无信息，不能宣布模型已修复或有限实验一定失败。

## 标准来源

Gaussian log-Sobolev/Herbst 与 Poincaré 的归一化沿用 Michel Ledoux (1999), *Concentration of measure and logarithmic Sobolev inequalities*, Séminaire de Probabilités XXXIII, pp.120–216，尤其引言与 §2.3：[NUMDAM 一手全文](https://www.numdam.org/item/SPS_1999__33__120_0.pdf)。

独立非同分布 Berry–Esseen 的 (N18) 形式及 general-case 绝对常数可见 Iosif Pinelis, *On the nonuniform Berry–Esseen bound*, arXiv:1301.2828v5 (2013)，§1，PDF pp.1–2，式(1.1)及其后的常数说明：[固定版本全文](https://arxiv.org/pdf/1301.2828v5)。本文只保守使用 \(C_{\rm BE}=1\)，不使用 iid 专属常数或声称最优常数。

Gaussian 集中、Poincaré、Berry–Esseen、条件特征函数和 KL DPI 均为标准工具。本文件的新增模型论证是将 raw 的全类参数定位与发射噪声的区间集中限制连接，给出带可核条件的一般相关结构障碍；原创性仍须另按直接近邻审查。来源哈希、读取范围和未核验的外部数值见 NOISE_SOURCE_NOTE.md。
