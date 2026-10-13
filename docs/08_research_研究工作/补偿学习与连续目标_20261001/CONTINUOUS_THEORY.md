# 连续载荷目标的梯度与近似正确模型中的风险恢复

本文件只作解析推导，不运行新科学随机数、拟合或积分，不改旧冻结。固定尺度的候选 latent 参数与原两尺度教师不同；本文件不假定有限 d 正确指定，也不由 latent 参数不同断言所有观察分布必不相同。核心风险界只需一个可行的近似表示。

全部目标和 KL 均为完整向量的总 nats。经验目标采用固定共享 bank；对数内的 K 与对数外的 B 不混用。

## 1. 连续两载荷类及谱界

沿偶数 \(d\ge16\)，\(m=d/2\)，两个组 H,L 已知。设 \(a=1\)；推导中保留 a 便于核对导数。参数域为
\[
\mathcal W=[L,U]^2,\qquad L=\tfrac12,\ U=\tfrac52,\qquad
\lambda_g=\frac{w_g}{\sqrt{d-1}},\quad q_g=\sqrt{1-\lambda_g^2}.
\]
候选 logits 为
\[
\eta_i=a(\lambda_gZ+q_gE_i),\qquad i\in g,
\tag{CT1}
\]
其中 Z 与各 \(E_i\) 独立标准正态。均值固定0，所有边际标准差固定为 a，给定 logits 独立 Bernoulli 发射。\(Q_w\) 表示精确积分模型。

全域 \(q_g^2\ge1-U^2/15=7/12\)，且
\[
\frac{7a^2}{12}I\preceq\Sigma_w\preceq
a^2(1+U^2)I=\frac{29a^2}{4}I.
\tag{CT2}
\]
下界来自对角私噪声部分；上界由非负载荷下每行非对角和至多 \(a^2U^2\) 及对称矩阵的行和范数界得到。这是一个连续、严格正定但受限的两参数类，不是任意 covariance 学习。

## 2. raw 目标的完整路径梯度

令 \(r=(d-1)^{-1/2}\)，对坐标 i 所属的组 g，
\[
\dot\eta_i=\frac{\partial\eta_i}{\partial w_g}
=ar\left(Z-\frac{\lambda_g}{q_g}E_i\right),\qquad
\ddot\eta_i=-ar^2q_g^{-3}E_i .
\tag{CT3}
\]
调整公共载荷时必须同时调整私噪声方差；遗漏第二项改变模型，也改变梯度。

对二元标签 \(c_i=(y_i+1)/2\)，第 k 个粒子的完整模式 log likelihood 为
\(\ell_k=\sum_i[c_i\log\sigma(\eta_{ki})+(1-c_i)\log\sigma(-\eta_{ki})]\)。
记
\[
A_{kg}=\partial_g\ell_k
=\sum_{i\in g}(c_i-\sigma(\eta_{ki}))\dot\eta_{ki},
\]
\[
H_{k,gg}=\partial_g^2\ell_k
=\sum_{i\in g}\left[
-\sigma(\eta_{ki})(1-\sigma(\eta_{ki}))\dot\eta_{ki}^2
+(c_i-\sigma(\eta_{ki}))\ddot\eta_{ki}\right],
\quad H_{k,HL}=0 .
\tag{CT4}
\]
对单 bank 损失
\[
\mathcal L_K=-\log\left(K^{-1}\sum_ke^{\ell_k}\right),
\qquad \pi_k=\frac{e^{\ell_k}}{\sum_je^{\ell_j}},
\]
有
\[
\nabla\mathcal L_K=-E_\pi A_k,\qquad
\nabla^2\mathcal L_K=-E_\pi H_k-\operatorname{Cov}_\pi(A_k).
\tag{CT5}
\]
经验梯度再对实际 N 个标签和 B 个 bank 平均。Hessian 中的 covariance 项为负半定，且 \(\ddot\eta\) 项没有固定符号；不能据二元 logistic 似然的简单凸性推断这个载荷目标凸。

对固定 d,K，紧参数域与 \(q_g\) 的正下界使这些导数由可积的 Gaussian 多项式统一支配，故可在总体期望中微分。

**K1只在总体层面平坦。** 每个候选的 logit 边际都为 \(N(0,a^2)\)，所以任意教师标签分布下
\[
F^{\rm raw}_{d,1}(P;w)=d\ell(a),\qquad
\ell(a)=E\log[2\cosh(aZ/2)] .
\tag{CT6}
\]
固定有限 bank 的经验梯度一般不为零，不能把 CT6 作为逐实现梯度零的 QA 条件。对 K>1，本文件没有证明 comp 附近的人口梯度符号、凸性或全局最优点位置。

## 3. col 与数值准似然的梯度

私噪声边际化核为
\[
h_g(w_g,z)=E_E\sigma(a[\lambda_gz+q_gE]).
\tag{CT7}
\]
直接路径微分与 Gaussian 分部积分给两个等价表达：
\[
\begin{aligned}
\dot h_g
&=ar\,E\left[\sigma'(\eta)
\left(z-\frac{\lambda_g}{q_g}E\right)\right]\\
&=ar\left[zE\sigma'(\eta)-a\lambda_gE\sigma''(\eta)\right],\\
\ddot h_g&=E[\sigma''(\eta)\dot\eta^2+\sigma'(\eta)\ddot\eta].
\end{aligned}
\tag{CT8}
\]
第二个一阶表达利用 \(E[E\sigma'(\eta)]=aq_gE\sigma''(\eta)\)，可用于与路径积分的独立数值核验。

给定组count \(n_g\)，单公共因子粒子的 log likelihood 为
\[
\ell_z=\sum_g[n_g\log h_g+(m-n_g)\log(1-h_g)] .
\]
其一阶与二阶导数是
\[
\partial_g\ell_z=
\frac{n_g-mh_g}{h_g(1-h_g)}\dot h_g,
\]
\[
\partial_g^2\ell_z
=n_g\left[\frac{\ddot h_g}{h_g}-\frac{\dot h_g^2}{h_g^2}\right]
+(m-n_g)\left[-\frac{\ddot h_g}{1-h_g}
-\frac{\dot h_g^2}{(1-h_g)^2}\right],
\quad \partial_H\partial_L\ell_z=0 .
\tag{CT9}
\]
col bank 的梯度/Hessian仍由 CT5 的 posterior 权重式得到。它用 iid 公共因子样本；每个候选保持原来的精确 Q。

完整数值准似然把公共因子也积分，梯度公式只需把离散 bank 权重替换为关于 z 的 likelihood-tilted Gaussian 权重。有限正权求积实现使用“求积权重×likelihood”的归一化权重。若程序另外将原始 count PMF 除以参数相关的总质量，梯度也须包含这个归一化常数的导数，不能只改函数值。

col 和精确似然都可按实际标签的双count直方图精确压缩。raw 的有限共享bank评分通常依赖标签在组内的位置，不能以canonical标签替代真实训练标签。

## 4. 全域导数界及每标签 col bias

因为 \(|\sigma''|\le\sigma'\le\min(\sigma,1-\sigma)\)，CT8 给
\[
|\partial_{w_g}\log h_g|,\quad
|\partial_{w_g}\log(1-h_g)|
\le\frac a{\sqrt{d-1}}
\left(|z|+\frac{aU}{\sqrt{d-1}}\right).
\tag{CT10}
\]
单核负 log likelihood 对 w 的梯度 l1 范数因此至多
\[
L_d(z)=\frac{ad}{\sqrt{d-1}}|z|
+\frac{a^2dU}{d-1}.
\tag{CT11}
\]
这是相对于参数的 l∞ 距离的 Lipschitz 界；对 z 的 Lipschitz 平方另有
\[
A_d=\frac{a^2d^2U^2}{d-1}.
\tag{CT12}
\]
CT11与CT12是不同方向的导数，不能混用。

令单核损失为 \(H_w(y,Z)\)，其期望为 \(\Psi_w^{\rm col}\)。
零均值对称性使该期望不依赖标签 y，但它一般不同于 raw 的 \(d\ell(a)\)。
写
\[
m_{w,K}(y)=E_{\rm bank}\mathcal L_K^{\rm col}(y),\qquad
b_{w,K}(y)=m_{w,K}(y)+\log Q_w(y).
\]
Jensen及单核平均上界给 \(0\le b_{w,K}\le EH_w+\log E e^{-H_w}\)。
Gaussian Lipschitz log-MGF 的常数 CT12 则给
\[
0\le b_{w,K}(y)\le \mathcal B_d:=A_d/2
\quad\text{对所有 }w,y,K\ge1.
\tag{CT13}
\]
这是 pointwise bias 的保守 \(O(d)\) 上界，不是说所有错误候选的实际 bias 都小。

对同一候选的两个公共值 z,t，私噪声条件 Gaussian 的 KL 系数为
\[
C_w=\frac12\sum_i\frac{\lambda_i^2}{1-\lambda_i^2}
\le\frac{dU^2}{2(d-1-U^2)}
\le C_{\mathcal W}:=\frac{16U^2}{2(15-U^2)}=\frac{40}{7}.
\tag{CT14}
\]
尺度 a 消去。经观察通道 DPI、共同 mixing 耦合及对耦合取最小，有
\(D(\nu\mathsf P_w\Vert\nu'\mathsf P_w)\le C_{\mathcal W}W_2^2(\nu,\nu')\)。

对标准 Gaussian 的 K 点 iid 经验测度，K≥2 时可用
\[
\omega_K=\frac{12\log K}{\sqrt K}
+6\sqrt{\frac2\pi}
\frac{\sqrt{2\log K}+1/\sqrt{2\log K}}K
\longrightarrow0.
\tag{CT15}
\]
其推导是将真/经验测度都截断到 \([-t,t]\)：首尾平方误差各为
\(e(t)=E(|Z|-t)_+^2\)，中间 \(W_2^2\le2tW_1\)，经验CDF点wise标准差≤\(1/(2\sqrt K)\)，所以
\(EW_2^2\le6e(t)+6t^2/\sqrt K\)。
用 \(e(t)\le2\varphi(t)(t+t^{-1})\)、\(t=\sqrt{2\log K}\) 即得 CT15。
于是对每个候选的自身观察分布，
\[
E_{Q_w}b_{w,K}(Y)
=F^{\rm col}_{d,K}(Q_w;w)-H(Q_w)
\le C_{\mathcal W}\omega_K .
\tag{CT16}
\]
这是对角线上的 bias 控制；下一节单独处理真实 P 与 Q 不同。

## 5. 不假定正确指定的 Hellinger bias 转移

**引理1。** 对任意完整有限字母分布 P,Q及函数 \(0\le b\le\mathcal B\)，
\[
E_Pb\le\left(\sqrt{E_Qb}+\sqrt{\mathcal B\,D(P\Vert Q)}\right)^2.
\tag{CT17}
\]

**证明。** 加权l2三角不等式给
\[
\|\sqrt b\sqrt P\|_2
\le\|\sqrt b\sqrt Q\|_2
+\sqrt{\mathcal B}\,\|\sqrt P-\sqrt Q\|_2.
\]
又
\(D(P\Vert Q)\ge2[1-\sum_y\sqrt{P(y)Q(y)}]
=\|\sqrt P-\sqrt Q\|_2^2\)，由 \(-\log x\ge1-x\) 得到。
两边平方即为 CT17。这里没有使用不存在的一般 KL 三角不等式。\(\square\)

**定理2（近似正确连续类的总体 col 保证）。**
设存在一个可行 \(w_d^0\in\mathcal W\)，且
\[
r_d=D(P_d^\star\Vert Q_{w_d^0}).
\]
若 \(\widehat w\) 满足
\(F^{\rm col}(P_d^\star;\widehat w)
\le F^{\rm col}(P_d^\star;w_d^0)+\varepsilon\)，特别是全域总体 \(\varepsilon\) 近优点，则
\[
D(P_d^\star\Vert Q_{\widehat w})
\le \varepsilon+r_d+
\left(\sqrt{C_{\mathcal W}\omega_K}+\sqrt{\mathcal B_dr_d}\right)^2.
\tag{CT18}
\]

**证明。** 定义 \(R(P;w)=-E_P\log Q_w(Y)\)。对所有候选，用 Jensen 得
\(R(P^\star;w)\le F^{\rm col}(P^\star;w)\)。
在比较点，
\[
F^{\rm col}(P^\star;w_d^0)-H(P^\star)
=r_d+E_{P^\star}b_{w_d^0,K}.
\]
将 CT13、CT16 代入 CT17，再作给定目标比较，即得 CT18。\(\square\)

若 \(r_d=O(d^{-2})\)，则 \(\mathcal B_dr_d=O(d^{-1})\)。
取 \(K=d\)、总比较误差 \(\varepsilon_d\to0\)，右端趋零。
即使实际有限 d 恰存在某个观察等价的 w，这个结论也仍有效；本证明不需要证明严格失配。

**原两尺度教师的可行近似。** 对本轮 a=1、c=.9、尺度(6,2)的弱公共因子教师，取
\[
w_g^0=\sqrt c\,\frac{\alpha(s_g^\star)}{\alpha(1)}.
\tag{CT19}
\]
它的可行性还可解析核对，不必依赖浮点载荷：\(\log\cosh x\le x^2/2\) 给
\(\operatorname{sech}^2(Z/2)\ge e^{-Z^2/4}\)，故
\[
\alpha(1)=\tfrac12E\operatorname{sech}^2(Z/2)
\ge1/\sqrt6,\qquad
\alpha(s)\le E|Z|=\sqrt{2/\pi}.
\]
再用 \(\alpha\) 对正尺度严格增加，得到
\[
\sqrt{.9}\le w_g^0\le\sqrt{10.8/\pi},
\]
这个区间严格包含于 (.5,2.5)。旧 COMPENSATION_THEORY 的全实轴 Taylor/分母控制因此给本类内
\(r_d=O(d^{-2})\)；其足够大维度条件也由 \(R^2\le12/\pi<4\) 在 d≥16满足。这是教师依赖的**证明比较点**，不可作为训练输入或据此称学习已完成。

## 6. 连续经验目标的统一桥接

令 N 个 iid 标签独立于 B 个 iid 公共因子 bank，每bank含 K 个 iid标准Gaussian公共粒子；全部标签和候选共享这些bank。定义理想（私噪声精确积分）经验目标 \(\widehat F^{\rm col}(w)\)。

给定任意 \(\zeta\in(0,1)\)，取确定性规则网格 \(\Gamma_h\subset\mathcal W\)，每点距某网格点的 l∞ 距离不超过 h/2，网格大小 M。零均值核的单粒子期望与标签无关，Gaussian集中与单粒子最小/平均夹界给
\[
m_{w,K}(y)\in[\Psi_w^{\rm col}-\sqrt{2A_d\log K},\Psi_w^{\rm col}].
\]
给定标签，共享 B 个bank的Gaussian Lipschitz平方是 \(A_d/B\)，不是 \(A_d/(NB)\)；标签条件均值的Hoeffding方差代理为 \(A_d\log K/(2N)\)。条件MGF先后取期望，再对网格作双侧union bound，以至少 \(1-\zeta/2\) 的概率，
\[
\sup_{w\in\Gamma_h}|\widehat F(w)-F(w)|
\le e_\Gamma:=
\sqrt{2A_d\left[\frac1B+\frac{\log K}{2N}\right]
\log\frac{4M}{\zeta}}.
\tag{CT20}
\]
候选之间共享随机数无需独立，MGF分解也不要求bank残差与标签条件均值独立。

另外，以至少 \(1-\zeta/2\) 的概率，
\[
\max_{b,k}|Z_{bk}|\le T_\zeta=
\sqrt{2\log(4BK/\zeta)}.
\]
由 CT11 及 log-mean posterior 梯度，整个经验目标在 w 的 l∞ 距离下 Lipschitz，常数至多
\[
L_{\rm emp}=\frac{ad}{\sqrt{d-1}}T_\zeta+\frac{a^2dU}{d-1}.
\]
总体目标同样有确定性常数
\[
L_{\rm pop}=
\frac{ad}{\sqrt{d-1}}\sqrt{2\log(2K)}
+\frac{a^2dU}{d-1},
\tag{CT21}
\]
因为 \(E\max_{k\le K}|Z_k|\le\sqrt{2\log(2K)}\)。
对任意 w 使用最近网格点，两个事件的 union bound 给概率至少 \(1-\zeta\) 的全域界
\[
\sup_{w\in\mathcal W}|\widehat F(w)-F(w)|
\le e_{\rm cont}:=e_\Gamma+\frac h2(L_{\rm emp}+L_{\rm pop}).
\tag{CT22}
\]
这里 \(\zeta/2\) 各用于网格浓度和Gaussian最大值，未把第三个事件隐含纳入同一预算。

若数值目标 \(\widetilde F\) 对整个连续域满足严格
\(\sup_w|\widetilde F(w)-\widehat F(w)|\le\eta\)，且实际选点是其全域
\(\varepsilon_{\rm alg}\) 近优点，则在 CT22 的事件上，
\[
F(\widehat w)\le F(w_d^0)+
2e_{\rm cont}+\varepsilon_{\rm alg}+2\eta .
\]
因此定理2中的 \(\varepsilon\) 可取这三项之和。
如果数值误差仅以另一个概率事件保证，需单独追加失败预算；求积收敛检查不是严格的全域 \(\eta\) 证书。

## 7. 一条渐密网格路径，而非固定9起点保证

取
\[
K_d=d,\quad N_d=B_d=d^2,\quad
\zeta_d=d^{-2},\quad h_d=d^{-2},\quad
M_d=(2d^2+1)^2 .
\tag{CT23}
\]
则 \(A_d=O(d)\)，\(e_\Gamma=O(\log d/\sqrt d)\)，网格扩展项为
\(O(d^{-3/2}\sqrt{\log d})\)。若 \(\eta_d,\varepsilon_{{\rm alg},d}\to0\)，连续全域经验近优点结合 CT18 给总joint KL在概率中趋零。

也可以直接枚举逐步加密的 \(\Gamma_{h_d}\) 并取全网格ERM：它相对连续理想经验最小值的误差至多
\(L_{\rm emp}h_d/2\)，加数值误差后仍趋零。这里网格点数与 N/B/K 为多项式充分路径，不宣称最优率或私噪声数值积分的端到端位复杂度已经被认证。

期望收敛也成立：任意 w 的 raw 单粒子期望是 \(d\ell(a)\)，故
\[
0\le D(P^\star\Vert Q_w)\le R(P^\star;w)\le d\ell(a).
\]
坏事件贡献至多 \(\zeta_dd\ell(a)=O(d^{-1})\)。
若数值误差另有概率失败，则需其为 \(o(1/d)\) 才可沿用这一期望结论。

**上述全域近优前提不由固定9起点L-BFGS-B认证。**
投影梯度小是数值一阶驻点诊断；网格交叉检查、多起点相同终点或独立评分较好都不能自动给出 \(\varepsilon_{\rm alg}\) 的全局上界。实际本轮连续学习实验必须据此收窄解释，不将 CT23 追认到固定计算预算。

## 8. 本文件不证明的内容

- 不证明当前盒内 raw 人口极小点的位置、K>1补偿点的局部最优性或总体梯度方向。
- 不证明所有9起点运行达到连续全局最优，亦不把独立部署风险复评当作优化证书。
- 不假定教师在固定尺度候选类中精确可表示；也不宣称已经排除了有限 d 的某个观察等价参数。
- 不把参考PMF/导数的数值收敛资格改称严格积分误差界。
- 不把矩学习、L-BFGS-B、Gaussian集中、Hellinger不等式或网格ERM作为新通用算法。

来源为本项目已封存的 COMPENSATION_THEORY、FINITE_THEORY 及正族 private-kernel/W2 推导；本文件的新增接口是近似表示误差与pointwise bias的加权Hellinger传递，以及上述明确的连续域覆盖步骤。源哈希和具体实现合同见 CONTINUOUS_DESIGN.md。
