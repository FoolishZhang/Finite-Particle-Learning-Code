# 固定正尺度下限的总体目标塌缩与同模型修复

本文件给出新的解析证明，不回写 v3.1 或旧科学冻结。结论针对明确的零均值、已知相关矩阵、紧尺度区间候选类，以及独立于当前标签的等权 prior 评分。没有新数值积分、样本、拟合或优化轨迹；有限维证据和原创性判断须另行处理。全部 log 为自然底，优化误差和 KL 均为完整向量的总 nats。

## 1. 对象、候选类与两个评分器

先定义一般原始 prior 评分。给定候选的中心 Gaussian
\(\eta\sim N(0,\Sigma_\theta)\)，对自旋标签 \(y\in\{-1,1\}^d\) 令
\[
f_\eta(y)=\prod_{i=1}^d \sigma(y_i\eta_i),\qquad
h_y(\eta)=-\log f_\eta(y)
=\sum_i\log[2\cosh(\eta_i/2)]-\tfrac12y^\top\eta.
\tag{1}
\]
\(\sigma\) 为 sigmoid。取 K 个 iid prior 粒子，整个 bank 独立于当前总体评分标签 \(Y\sim P^\star\)，定义
\[
\widehat Q_{\theta,K}^{\rm raw}(y)=K^{-1}\sum_{k=1}^Kf_{\eta_k}(y),\qquad
F_{d,K}^{\rm raw}(P^\star;\theta)=
E_{Y,\eta_{1:K}}[-\log\widehat Q_{\theta,K}^{\rm raw}(Y)].
\tag{2}
\]
精确积分模型为 \(Q_\theta=E_\eta f_\eta\)，风险为
\(R_d(P^\star;\theta)=E_{P^\star}[-\log Q_\theta(Y)]\)。
这是 population 目标；训练样本数 N、近似这个期望的 bank 数 \(B_{\rm MC}\)、以及每个 bank 的 K 均未混为同一个预算。

主候选类固定 \(0<a<b<\infty,\ 0<c<1\)，沿偶数 \(d=2m\ge4\)，并置
\[
\rho_d=\frac{c}{d-1},\qquad
R_d^-=(1+\rho_d)I-\rho_d\mathbf1\mathbf1^\top,\qquad
R_d^+=(1-\rho_d)I+\rho_d\mathbf1\mathbf1^\top.
\tag{3}
\]
令 \(\mathcal H_d\) 为非空的 m 元坐标子集集合。它可以只有一个已知分组，也可包含全部等大分组；始终是有限集合。设
\[
\Theta_d=\mathcal H_d\times[a,b]^2,\quad
\theta=(H,u,v),\quad
s_i(\theta)=u\,1_H(i)+v\,1_{H^c}(i),\quad
D_\theta=\operatorname{diag}(s_i(\theta)).
\]
对于每个符号 \(\xi\in\{-,+\}\)，候选 logits 协方差为
\(\Sigma_{\theta,d}^{\xi}=D_\theta R_d^\xi D_\theta\)，均值固定为零。两个尺度允许相等，也没有候选分离约束。

所有候选满足同一、维数无关的双侧谱界：
\[
a^2(1-c)I\preceq\Sigma_{\theta,d}^{\xi}\preceq
\Lambda I,\qquad \Lambda=b^2(1+c).
\tag{4}
\]
负族的最大特征值还可用 \(b^2(1+c/3)\)，但下文为两个符号使用共同的 \(\Lambda\)。这里 a 是 logit 标准差的下限，不是零方差边界。

预测后果部分固定真实内点
\[
a<s_L^\star<s_H^\star<b,\qquad H_d^\star\in\mathcal H_d,
\]
尺度值不随 d 改变，真分布分别为
\(P_d^{\star,\xi}=Q_{H_d^\star,s_H^\star,s_L^\star}^{\xi}\)。
两个符号使用相同的真实组、尺度与 c；每一维观察边际均为 1/2。真模型在各自候选类中，但本文件不分析学习算法如何搜索这些候选。

## 2. 整个原始目标的统一近似

定义
\[
\ell(s)=E_{G\sim N(0,1)}\log[2\cosh(sG/2)],\qquad
\Psi_d(\theta)=\sum_i\ell(\sqrt{\Sigma_{\theta,ii}}).
\tag{5}
\]

**定理 1（零均值 Gaussian prior 的统一主项）。**
设任意候选类满足 \(\lambda_{\max}(\Sigma_\theta)\le\Lambda<\infty\)，评分采用 (2)。对任意标签分布 \(P^\star\)、任何 \(K\ge1\) 和每个候选，
\[
0\le\Psi_d(\theta)-F_{d,K}^{\rm raw}(P^\star;\theta)
\le t_{d,K}:=\sqrt{2\Lambda d\log K}.
\tag{6}
\]
同一个 \(\Lambda\) 适用于全类时，此界也对全类统一。它不需要观察标签之间独立，也不要求 \(P^\star\) 是 Gaussian-logistic 分布。

**证明。** 每个固定 y 的 (1) 对 Gaussian 坐标的梯度分量是
\[
\partial_{\eta_i}h_y=\tfrac12\tanh(\eta_i/2)-\tfrac12y_i,
\]
绝对值至多 1。因此 h_y 是 \(\sqrt d\)-Lipschitz；写
\(\eta=\Sigma_\theta^{1/2}Z\) 后，对标准 Gaussian Z 的 Lipschitz 常数至多 \(\sqrt{\Lambda d}\)。中心化的 Gaussian Lipschitz 函数满足经典 Laplace 界
\[
\log E\exp\{t(h_y-Eh_y)\}\le t^2\Lambda d/2,\qquad t\in\mathbb R.
\tag{7}
\]
这是 Gaussian log-Sobolev/Herbst 的标准结果，见 Ledoux (1999)，引言 pp.123–124 与 §2.3；不作为新工具主张。具体地，Gaussian log-Sobolev 的
\(\operatorname{Ent}(e^g)\le\tfrac12E[e^g\|\nabla g\|^2]\)
对 \(g=t(h_y-Eh_y)\) 给 Laplace 对数 \(\varphi\) 的
\(t\varphi'(t)-\varphi(t)\le t^2\Lambda d/2\)；
由 \(\varphi(0)=\varphi'(0)=0\) 积分可得 (7)。对无界 Lipschitz 函数可先截断和平滑，再由 Gaussian 可积性取极限。

零均值保证对每个 y 都有 \(Eh_y=\Psi_d(\theta)\)，没有 y 依赖。于是对 K 个粒子，
\[
\begin{aligned}
E\max_{k\le K}(\Psi_d-h_y(\eta_k))
&\le\frac1t\log\sum_{k=1}^KEe^{t(\Psi_d-h_y(\eta_k))}\\
&\le\frac{\log K}{t}+\frac{t\Lambda d}{2}.
\end{aligned}
\]
K>1 时优化 t 得 \(t_{d,K}\)；K=1 时中心项均值为零。逐 bank 有
\[
\min_kh_y(\eta_k)\le
-\log(K^{-1}\sum_ke^{-h_y(\eta_k)})
\le K^{-1}\sum_kh_y(\eta_k).
\]
先对 bank、再对任意独立标签 Y 取期望，即得 (6)。\(\square\)

因此若 \(\log K_d=o(d)\)，
\[
\sup_{\theta\in\Theta_d}
\left|d^{-1}F_{d,K_d}^{\rm raw}(P^\star;\theta)
-d^{-1}\Psi_d(\theta)\right|\longrightarrow0.
\tag{8}
\]
这里比较的是 F 与单粒子期望主项 \(\Psi\)；\(\Psi-F\) 不是模型的 Jensen gap \(F-R\)。该机制也不专属于负相关。虽然最大值的这一证明实际不要求粒子之间独立，本文仍维持标准 iid 等权 prior 合同，不推广到任意数据依赖权重或标签依赖 proposal。

## 3. 全部总体近最优点趋向固定尺度下限

对 (3)–(4) 的两尺度类，
\(\Psi_d(H,u,v)=d[\ell(u)+\ell(v)]/2\)。微分可在期望内进行，且
\[
\ell'(s)=\tfrac12E[G\tanh(sG/2)]>0\quad(s>0),\qquad
\ell''(s)=\tfrac14E[G^2\operatorname{sech}^2(sG/2)]>0.
\tag{9}
\]
记 \(\kappa_a=\ell'(a)>0\)。

**定理 2（固定正下限下的总体近最优点）。**
对任一符号、任意标签分布及每个固定 d,K，F 在 \(\Theta_d\) 上达到最小值。若
\[
F_{d,K}^{\rm raw}(P^\star;\theta)
\le\min_{\vartheta\in\Theta_d}F_{d,K}^{\rm raw}(P^\star;\vartheta)
+\varepsilon,\qquad \varepsilon\ge0,
\]
则
\[
0\le(u-a)+(v-a)\le
\min\left\{2(b-a),\
\frac{2[\sqrt{2\Lambda d\log K}+\varepsilon]}{d\kappa_a}\right\}.
\tag{10}
\]
因此沿 \(\log K_d=o(d)\)、\(\varepsilon_d=o(d)\) 的任意总体近最优序列，两尺度都趋于 a，且 \(|u_d-v_d|\to0\)。特别地，它覆盖所有实际总体全局极小点序列，不要求唯一性。

**证明：存在性。** 对固定 H，用共同标准 Gaussian 粒子表示
\(\eta_k=D_\theta(R_d^\xi)^{1/2}Z_k\)。每个有限标签与粒子实现的评分对 u,v 连续。并且
\[
0\le-\log\widehat Q_{\theta,K}^{\rm raw}(y)
\le K^{-1}\sum_k h_y(\eta_k)
\le d\log2+\frac bK\sum_{k,i}|((R_d^\xi)^{1/2}Z_k)_i|.
\]
右侧对固定 d,K 可积且不依赖 u,v,H；最后一点来自用同一个 \(R_d^\xi\)，尺度仅逐坐标乘且均不超过 b。支配收敛给 F 连续。有限 H 集合与紧方形 [a,b]^2 因此保证最小值存在。

**证明：定位。** \((H,a,a)\) 对任意 H 都可行，并代表相同模型。由 (6)，
\[
\begin{aligned}
\frac d2[\ell(u)+\ell(v)]-t_{d,K}
&\le F(\theta)\le\min F+\varepsilon\\
&\le F(H,a,a)+\varepsilon\le d\ell(a)+\varepsilon .
\end{aligned}
\]
又 \(\ell(s)-\ell(a)\ge\kappa_a(s-a)\)；整理并结合区间的平凡上界即得 (10)。\(\square\)

这个有限维上界可能完全无信息：当其第二项不小于 \(2(b-a)\) 时，未优于参数区间本身。K>1 时没有证明任意有限 d 的极小点精确等于 (a,a)，也没有证明 F 对尺度逐点单调。K=1 是特殊情形，F=\(\Psi\)，故该受限类中的尺度最优值确为 (a,a)。任何情形下，都没有推出 SGD 的轨迹或有限 bank 经验优化器一定达到总体近最优条件。

对固定真实内点，令
\(\Delta_\ell=[\ell(s_H^\star)+\ell(s_L^\star)]/2-\ell(a)>0\)。
同一比较还给
\(F^{\rm raw}(\theta^\star)-F^{\rm raw}(H,a,a)\ge d\Delta_\ell-t_{d,K}\)。
当右侧为正时，真参数已被严格排除为总体最优；若该粗界非正，则不据它宣称有限维排序。

## 4. 变化两尺度的 CLT：不能直接代入固定参数结论

记
\[
f_s(z)=\tanh(sz/2),\quad
\alpha(s)=E[Zf_s(Z)]=E[f_s'(Z)],\quad
v(s)=E[f_s(Z)^2].
\tag{11}
\]
最后一个 alpha 恒等式由 Gaussian 分部积分得到。\(\alpha(s)>0\) 且
\(\alpha'(s)=\tfrac12E[Z^2\operatorname{sech}^2(sZ/2)]>0\)，并有
\(\alpha(s)^2\le v(s)<1\)。

**引理 3（两种符号、变化参数的总自旋 CLT）。**
设 \(u_d,v_d\in[a,b]\)，且 \(u_d\to u_\infty,\ v_d\to v_\infty\)。任意大小 d/2 的分组 H_d 下，对精确积分模型 \(Q_{H_d,u_d,v_d}^\xi\)，
\[
\frac1{\sqrt d}\sum_{i=1}^dY_i
\Rightarrow N(0,V_\xi),\qquad
V_-=1-c\bar\alpha^2,\quad V_+=1+c\bar\alpha^2,
\tag{12}
\]
其中 \(\bar\alpha=[\alpha(u_\infty)+\alpha(v_\infty)]/2\)。
不要求 \(u_d-u_\infty\) 或 \(v_d-v_\infty\) 为 \(o(d^{-1/2})\)。

**共同准备。** (3) 对所有坐标置换不变，而总自旋也是对称统计量，故可以逐 d 将 H_d 重排为前 m 个坐标而不改其分布。记相应两个函数为 \(f_{g,d}\)，并令
\[
\bar\alpha_d=\tfrac12[\alpha(u_d)+\alpha(v_d)]\to\bar\alpha,\qquad
\bar v_d=\tfrac12[v(u_d)+v(v_d)]\to\bar v .
\]
对 s 属于 [a,b]，f、前两阶空间导数及 \(f^2\) 的一阶导数均统一有界；s 连续变化还给 f 及其导数的 Gaussian L2 连续性。每一行中 \(Ef_{g,d}(Z)=0\) 精确成立，因而没有被 \(\sqrt d\) 放大的均值漂移。

以下每行 \(Z_i\) 为 iid 标准 Gaussian，
\[
A_d=d^{-1/2}\sum_i f_{g(i),d}(Z_i),\qquad
W_d=d^{-1/2}\sum_iZ_i.
\]
任意固定线性组合的三阶绝对矩和归一化余项为 \(O(d^{-1/2})\)。因此 Lyapunov 三角阵 CLT（或逐项展开独立特征函数）给
\[
(A_d,W_d)\Rightarrow N\!\left(0,
\begin{pmatrix}\bar v&\bar\alpha\\\bar\alpha&1\end{pmatrix}\right).
\tag{13}
\]
这一步使用了变化的 \(\bar v_d,\bar\alpha_d\) 的极限，而不是把固定尺度 CLT 直接代入。

**负族的 latent 条件均值。** 精确表示为
\[
U_i=\gamma_dZ_i-\beta_d\bar Z_d,\quad
\gamma_d=\sqrt{1+\rho_d},\quad
\beta_d=\gamma_d-\sqrt{1-c},\quad
\bar Z_d=d^{-1}\sum_iZ_i .
\]
它的协方差就是 \(R_d^-\)。记 \(\beta=1-\sqrt{1-c}\)，则
\(\beta_d\to\beta\)、\(2\beta-\beta^2=c\)。位移
\(\Delta_i=U_i-Z_i\) 满足 \(\sum_i\Delta_i^2=O_p(1)\)：
\[
\sum_i\Delta_i^2\le
2(\gamma_d-1)^2\sum_iZ_i^2+2\beta_d^2d\bar Z_d^2.
\]
Taylor 二阶余项之和除以 \(\sqrt d\) 因而趋零。含
\(\gamma_d-1=O(d^{-1})\) 的一阶项也为 \(o_p(1)\)，由有界导数及 \(\sum_i|Z_i|=O_p(d)\) 可见。
独立三角阵的有界导数 LLN 给
\[
d^{-1}\sum_i f_{g(i),d}'(Z_i)-\bar\alpha_d\to0
\quad\text{依概率}.
\]
故
\[
d^{-1/2}\sum_i f_{g(i),d}(U_i)
=A_d-\beta\bar\alpha W_d+o_p(1)
\Rightarrow N(0,\bar v-c\bar\alpha^2).
\tag{14}
\]

**正族的 latent 条件均值。** 令 W 为独立于全部 Z_i 的标准 Gaussian，
\[
U_i=\sqrt{1-\rho_d}Z_i+\sqrt{\rho_d}W .
\]
这精确产生 \(R_d^+\)。同样 \(\sum_i(U_i-Z_i)^2=O_p(1)\)，两种 f 的统一二阶导数使 Taylor 余项消失。由 \(\sqrt{d\rho_d}\to\sqrt c\) 与上述导数 LLN，
\[
d^{-1/2}\sum_i f_{g(i),d}(U_i)
=A_d+\sqrt c\,\bar\alpha W+o_p(1)
\Rightarrow N(0,\bar v+c\bar\alpha^2).
\tag{15}
\]
A_d 与 W 对每个 d 独立，极限亦独立。

**两族的 Bernoulli 条件噪声。** 给定整行 Gaussian 变量，置
\(\epsilon_i=Y_i-f_{g(i),d}(U_i)\)。这些变量独立、中心、绝对值至多 2，条件方差为 \(1-f_{g(i),d}(U_i)^2\)。由于 \(f^2\) 的统一 Lipschitz 界，
\[
\frac1d\sum_i|f_{g(i),d}(U_i)^2-f_{g(i),d}(Z_i)^2|
\le C\sqrt{\frac1d\sum_i(U_i-Z_i)^2}\to0 .
\]
对 Z 行作有界三角阵 LLN，得到平均条件方差趋于 \(1-\bar v\)。对任意固定实数 t，逐项条件特征函数展开为
\[
E[e^{it\epsilon_i/\sqrt d}\mid\mathcal G_d]
=1-\frac{t^2}{2d}\operatorname{Var}(\epsilon_i\mid\mathcal G_d)
+O(d^{-3/2}),
\]
余项由 \(|\epsilon_i|\le2\) 统一控制。将乘积取对数，其二次余项为 \(O(d^{-1})\)，所以
\[
E[e^{itd^{-1/2}\sum_i\epsilon_i}\mid\mathcal G_d]
\longrightarrow e^{-t^2(1-\bar v)/2}
\quad\text{依概率且在 L1 中}.
\tag{16}
\]
L1 收敛来自模长至多 1。乘任意模长至多 1 的 Gaussian 可测特征函数再取期望，说明极限噪声与 (14)/(15) 的极限独立。两项相加即给 (12)。由于 \(\bar\alpha^2<1,\ c<1\)，负号的极限方差也严格正。\(\square\)

由同一“任意序列”证明还有一个所需的局部统一形式：对任何 \(\delta_d\downarrow0\)，(12) 在 \(|u-a|+|v-a|\le\delta_d\) 的候选上对固定正态连续事件一致成立。否则可选出违背一致性的候选序列，但它必满足 u_d,v_d→a，与引理矛盾。该结论不要求声称整个紧参数类的统一 CLT。

## 5. 同教师的全局数量事件预测后果

定义同一个、无需真实组身份的事件
\[
\mathcal A_d=\left\{\left|\sum_iY_i\right|\le\sqrt d\right\}.
\tag{17}
\]
令
\[
\bar\alpha_\star=
\frac{\alpha(s_H^\star)+\alpha(s_L^\star)}2>\alpha(a)>0,\qquad
p_\xi(x)=2\Phi\!\left((1+\xi c x^2)^{-1/2}\right)-1,
\tag{18}
\]
其中 \(\xi=+\) 取 +1，\(\xi=-\) 取 −1。记
\(p_\star^\xi=p_\xi(\bar\alpha_\star)\)、\(p_a^\xi=p_\xi(\alpha(a))\)，两者均在 (0,1)。

**定理 4（全部 raw 总体近最优点的事件失真）。**
固定内点教师 \(P_d^{\star,\xi}\)，取满足定理 2 且
\(\log K_d=o(d),\ \varepsilon_d=o(d)\) 的任意候选序列 \(\theta_d\)。用其精确积分模型预测事件，令
\[
p_d^\star=P_d^{\star,\xi}(\mathcal A_d),\qquad
q_d=Q_{\theta_d}^{\xi}(\mathcal A_d).
\]
则
\[
p_d^\star\to p_\star^\xi,\quad q_d\to p_a^\xi,
\qquad
p_\star^->p_a^->2\Phi(1)-1,\quad
p_\star^+<p_a^+<2\Phi(1)-1 .
\tag{19}
\]
特别地，该 fixed-floor 候选并非独立分布，不能使用旧独立候选的事件差常数。

**证明。** 定理 2 使 u_d,v_d→a；引理 3 分别用于真教师固定尺度和候选变化尺度。正态极限在事件边界 ±1 无质量，因此概率收敛成立。函数 \(v\mapsto2\Phi(v^{-1/2})-1\) 对正 v 严格递减，加上 \(\bar\alpha_\star>\alpha(a)\)，给出两种符号相反的严格排序。\(\square\)

对事件标签 \(B_d=1_{\mathcal A_d}\)，预测数值 q 的 scalar Brier 超额为
\[
E(B_d-q)^2-\min_tE(B_d-t)^2=(q-p_d^\star)^2.
\]
因此这些 raw 近最优模型的 Brier 与事件 log-loss 超额分别满足
\[
\begin{aligned}
(q_d-p_d^\star)^2&\longrightarrow(p_a^\xi-p_\star^\xi)^2>0,\\
\operatorname{kl}(p_d^\star\Vert q_d)&\longrightarrow
\operatorname{kl}(p_\star^\xi\Vert p_a^\xi)>0 .
\end{aligned}
\tag{20}
\]
二元 KL 连续性可用，因为两极限都严格位于 (0,1)。事件 DPI 还给
\[
\liminf_dD_{\rm joint}(P_d^{\star,\xi}\Vert Q_{\theta_d}^{\xi})
\ge\operatorname{kl}(p_\star^\xi\Vert p_a^\xi)>0.
\tag{21}
\]
这说明即使对所选参数作精确积分，事件预测仍错。它不是平均单概念 LOO 下界，也不把尺度趋向 a 直接偷换成整个高维分布在 KL 中趋向某个极限模型。

若从总体近最优集合中随机选点且 \(\varepsilon_d\) 是确定性的统一容差，同一结论对事件概率由第4节的局部统一性给出；不需要选择器确定。若只对随机优化误差给概率控制，须明确这种控制再使用相应收敛，不自动包含任意训练程序。

## 6. 正族两尺度的每候选同模型私噪声边际化

以下仅讨论正族，保留完全相同的 \(\Theta_d,[a,b],c,\mu=0\)。对每个候选 \(\theta\)，用独立标准 Gaussian Z、E_i 表示
\[
\eta_i=s_i(\theta)\{\sqrt{\rho_d}Z+\sqrt{1-\rho_d}E_i\}.
\]
定义
\[
h_{d,s}(z)=E_E\sigma\!\left(s[\sqrt{\rho_d}z+\sqrt{1-\rho_d}E]\right),\qquad
\mathsf P_{\theta,z}(y)=
\prod_i\operatorname{Ber}\!\left(\frac{y_i+1}2;h_{d,s_i(\theta)}(z)\right).
\tag{22}
\]
私噪声逐坐标独立，所以对**每个候选**，
\[
Q_\theta^+(y)=E_Z\mathsf P_{\theta,Z}(y).
\tag{23}
\]
两尺度仅使各坐标的 h 不同；单个 kernel 仍逐坐标 product，积分模型没有改变。

取 \(Z_1,\ldots,Z_K\) iid 标准 Gaussian 并独立于当前 Y，定义
\[
\widehat Q_{\theta,K}^{\rm col}(y)=K^{-1}\sum_k\mathsf P_{\theta,Z_k}(y),
\quad
F_{d,K}^{\rm col}(P^\star;\theta)=
E[-\log\widehat Q_{\theta,K}^{\rm col}(Y)].
\tag{24}
\]
col 表示 private-noise-collapsed，也即此处的 private-RB。它没有给负族定义修复器，不等于组内 count-RB。

**引理 5（同 Q、Jensen 与连续性）。**
对所有候选，\(E\widehat Q_{\theta,K}^{\rm col}=Q_\theta^+\)，且对任意独立标签分布，
\[
R_d(P^\star;\theta)\le F_{d,K}^{\rm col}(P^\star;\theta)
\le F_{d,K}^{\rm raw}(P^\star;\theta).
\tag{25}
\]
固定 d,K 时 \(F^{\rm col}\) 在同一紧候选类达到最小值。

**证明。** 在共同 Z bank 上，私噪声独立地用于每个 raw 粒子，故对固定 y，
\[
\widehat Q_{\theta,K}^{\rm col}(y;Z_{1:K})
=E[\widehat Q_{\theta,K}^{\rm raw}(y;Z_{1:K},E)\mid Z_{1:K}].
\]
这是 likelihood 估计器的条件期望，不是负对数评分的条件期望。条件 Jensen 给 (25) 的第二个不等式；第一次无偏性及对整个 bank 的 Jensen 给第一个。h 对 s 连续且在有限 z 时严格位于 (0,1)。由条件 Jensen 和 (1)，单 kernel 的负 log 有统一可积上界
\[
-\log\mathsf P_{\theta,z}(y)
\le d\log2+bd\{\sqrt{\rho_d}|z|+\sqrt{1-\rho_d}\,E|G|\}.
\]
bank 的负 log 不超过各项负 log 平均，故可用支配收敛证明连续性；紧性和有限 H 随后给出极小值存在。\(\square\)

## 7. 正族的统一真参数 bias 上界

令 \(\nu=N(0,1)\)，\(\widehat\nu_K=K^{-1}\sum_k\delta_{Z_k}\)。给定同一候选的 z 或 t，logits 的条件协方差均为
\((1-\rho_d)D_\theta^2\)，均值差为
\(\sqrt{\rho_d}(z-t)(s_i(\theta))_i\)。所以 Gaussian KL 中两尺度逐坐标消去：
\[
D(\mathsf P_{\theta,z}\Vert\mathsf P_{\theta,t})
\le A_d^+(z-t)^2,\qquad
A_d^+=\frac{d\rho_d}{2(1-\rho_d)}
=\frac{cd}{2(d-1-c)}\le C_c:=\frac c{1-c}.
\tag{26}
\]
第一个不等式使用共同 logistic-Bernoulli 通道的 DPI；a>0 保证相关 Gaussian 条件协方差非退化。

给任意混合测度 \(\nu_1,\nu_2\) 及其共同耦合 \(\gamma(dz,dt)\)，比较
\(\gamma(dz,dt)\mathsf P_{\theta,z}(dy)\) 与
\(\gamma(dz,dt)\mathsf P_{\theta,t}(dy)\)。KL 链式法则及边际化给
\[
D(\nu_1\mathsf P_\theta\Vert\nu_2\mathsf P_\theta)
\le A_d^+E_\gamma(Z-T)^2,
\]
对耦合取 infimum 即为
\[
D(\nu_1\mathsf P_\theta\Vert\nu_2\mathsf P_\theta)
\le A_d^+W_2^2(\nu_1,\nu_2).
\tag{27}
\]
只要求混合测度有有限二阶矩；离散经验测度也适用，不使用可能无限的 continuous/discrete mixing KL。

为完整给出经验 W2 界，令 \(\nu_t,\widehat\nu_{K,t}\) 分别将真与经验分布截断到 [-t,t]，\(t>0\)，并记
\[
e(t)=E(|Z|-t)_+^2
=2[(1+t^2)\Phi(-t)-t\varphi(t)]
\le2\varphi(t)(t+t^{-1}).
\]
首尾截断耦合的期望平方误差各至多 e(t)。中间两个分布支持在 [-t,t]，一维分位数耦合与 CDF 恒等式给
\[
W_2^2(\nu_t,\widehat\nu_{K,t})
\le2tW_1(\nu_t,\widehat\nu_{K,t})
=2t\int_{-t}^{t}|\widehat F_{K,t}(x)-F_t(x)|\,dx.
\]
固定 x 的经验 CDF 是 K 个 iid Bernoulli 指示的平均，故均方根误差至多 \(1/(2\sqrt K)\)，包括截断端点有原子的情况。W2 三角不等式再用三项平方和界，得到
\[
EW_2^2(\nu,\widehat\nu_K)\le6e(t)+\frac{6t^2}{\sqrt K}.
\]
K≥2 时取 \(t_K=\sqrt{2\log K}\)，于是
\[
\omega_K=
\frac{12\log K}{\sqrt K}
+6\sqrt{\frac2\pi}\frac{t_K+t_K^{-1}}K
\longrightarrow0,\qquad
EW_2^2(\nu,\widehat\nu_K)\le\omega_K.
\tag{28}
\]
这只是一个保守充分界，不是最优经验 W2 收敛率。

**定理 6（两尺度真参数处的统一 bias）。**
每个正族真参数 \(\theta^\star\in\Theta_d\) 都满足
\[
\begin{aligned}
0\le F_{d,K}^{\rm col}(Q_{\theta^\star}^+;\theta^\star)
-H(Q_{\theta^\star}^+)
&=E D_{\rm joint}
(Q_{\theta^\star}^+\Vert\widehat Q_{\theta^\star,K}^{\rm col})\\
&\le A_d^+\omega_K\le C_c\omega_K .
\end{aligned}
\tag{29}
\]
因此上界对 d、组位置及两尺度真值统一，常数也不依赖 a,b。它仅是在“真分布与评分参数一致”这条对角线上的统一 bias；不声称对固定 \(P^\star\) 和所有错误候选都给同样 Jensen gap 界。

**证明。** 逐 bank 用 (27)，取 \(\nu_1=\nu,\nu_2=\widehat\nu_K\)，再取 bank 期望并用 (28)。标签与整个 bank 独立保证第一行的总体 NLL/KL 恒等式。\(\square\)

## 8. 同候选类中的近最优风险恢复

**定理 7（正族同模型修复）。**
取同一个正族教师 \(P_d^{\star,+}=Q_{\theta_d^\star}^+\) 与同一个 \(\Theta_d\)。若
\[
F_{d,K}^{\rm col}(P_d^{\star,+};\widetilde\theta)
\le\min_{\theta\in\Theta_d}F_{d,K}^{\rm col}(P_d^{\star,+};\theta)+\varepsilon,
\]
则对 K≥2，
\[
D_{\rm joint}(P_d^{\star,+}\Vert Q_{\widetilde\theta}^+)
\le\varepsilon+C_c\omega_K .
\tag{30}
\]

**证明。** 所有候选的 Jensen 下界只用于第一步，真参数的统一 bias 只用于最后一步：
\[
\begin{aligned}
D_{\rm joint}(P^\star\Vert Q_{\widetilde\theta})
&=R(P^\star;\widetilde\theta)-H(P^\star)\\
&\le F^{\rm col}(P^\star;\widetilde\theta)-H(P^\star)\\
&\le\varepsilon+\min_\theta F^{\rm col}(P^\star;\theta)-H(P^\star)\\
&\le\varepsilon+F^{\rm col}(P^\star;\theta^\star)-H(P^\star)\\
&\le\varepsilon+C_c\omega_K .
\end{aligned}
\]
不需要上界错误候选的 bias，也不需要优化器识别唯一真参数。\(\square\)

任意事件的 binary log-loss 超额由 DPI 至多为 (30) 的右端；其 scalar Brier 超额还至多为该右端的一半，因为
\(\operatorname{kl}(p\Vert q)\ge2(p-q)^2\)。后二元不等式也可由固定 q 时 KL 关于 p 的二阶导数 \(1/[p(1-p)]\ge4\) 直接得到。

因此可在同一正族、相同固定区间与相关矩阵上作严格对照。选择任何 \(K_d\to\infty\) 且 \(\log K_d=o(d)\)：

- raw 的全部总体近最优点，只要总优化误差 \(\varepsilon_d^{\rm raw}=o(d)\)，便具有 (19)–(21) 的非消失事件错误，特别包括精确总体极小点。
- col 的总体近最优点，若总优化误差 \(\varepsilon_d^{\rm col}=o(1)\)，则 (30) 使总联合 KL 趋零，包括 (17) 在内的任意事件序列的两种超额评分亦趋零，特别包括精确总体极小点。

两种近最优容差的单位都是总 nats，但所需收敛尺度不同，不能将 raw 的 o(d) 直接换到恢复界中。取两边精确极小点时，两者优化误差同为零，区别仅来自评分方式，候选模型类和真实模型不变。同 K 也不等于同运行时间：col 必须计算精确一维私噪声积分，其实际求积、梯度和反向传播成本未在此分析。

## 9. 结论边界与旧结果的关系

1. 本文的 covariance 谱下界由 (4) 直接保证，并非让独立候选的方差随 d 趋零。所有尺度趋 a 仍保留相关结构；这与 v3.1 的 \(d^{-2}I\) 候选不同。
2. 这只是已知 \(R_d^\pm\)、零均值、两等大尺度且允许合并的候选类结论。固定谱下界不是“原自由 Cholesky 参数全类”的同义词；该全类可能有额外均值/相关自由度或不受统一谱上界控制。若禁止 u=v，(a,a) 比较点不可用。
3. 对 K>1，(10) 不承诺有限 d 精确贴边或给定实验维数出现强效应。粗尺度界、正族 \(\omega_K\) 界和有限维区间都可能无实际信息，不可依理论方向删去相反数值结果。
4. 当前结论是总体目标及其近最优点。没有给有限标签 N、有限独立 bank 平均数 \(B_{\rm MC}\) 的一致逼近或优化误差证书，也不证明 SGD 达到这些集合。旧 group 目标增加 \(B_{\rm MC}\) 的控制须与新 raw 结论分开。
5. 正族两尺度的 private-RB 保持每候选同 Q；这个事实加上 (29) 才给出恢复保证。同 Q 本身并不够，也不能将这里的 product-kernel 传输界套给负族 count/group-RB。
6. raw 的尺度主项机制同时覆盖正负族；负族另外存在有限 product-mixture 表示障碍，这是另一层结论。本文没有把符号特化、Gaussian 集中、Jensen、标准 argmin 论证或 KL 数据处理宣称为新一般定理。

### 标准来源

- Michel Ledoux (1999), *Concentration of measure and logarithmic Sobolev inequalities*, Séminaire de Probabilités XXXIII, pp.120–216。Gaussian log-Sobolev 与 Herbst Laplace 界见引言 pp.123–124，系统讨论见 §2.3：[NUMDAM 一手全文](https://www.numdam.org/item/SPS_1999__33__120_0.pdf)。
- 条件特征函数、Lyapunov 三角阵 CLT、Gaussian 分部积分、KL chain rule/DPI、一维 Wasserstein 的分位数/CDF 表示及 Jensen 均为标准工具。上文给出本问题所需条件和推导，没有用新模拟代替证明。
- 旧同尺度 positive-kernel coupling 与经验 W2 推导的项目来源、变化两尺度负族 CLT 的来源，以及本轮新增/重组范围，见同目录 THEORY_SOURCE_NOTE.md 的逐文件哈希；本文证明不要求读者依赖那些仓库链接补齐关键步骤。
