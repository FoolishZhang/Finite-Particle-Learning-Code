# 无均值边界的发射噪声定理

本文件给出本轮最强的噪声障碍版本。它保留已完成的NOISE_THEORY及其独立审读，不改写它们；利用一个logistic通道恒等式，将噪声控制常数改为2，并完全去掉候选均值的有界盒约束。候选协方差仍有维数统一的谱上界和固定正下界；未覆盖没有这些约束的原自由Cholesky实现。0新随机样本、拟合或数值积分。

## 1 通道恒等式

令
$$
l(x)=\log[2\cosh(x/2)],\qquad g(x)=\operatorname{sech}^2(x/2).
$$
直接微分有$l''=g/4$、$g''=g-3g^2/2$，所以偶函数
$$
J(x)=g(x)+2l(x),\qquad J''(x)=\tfrac32g(x)(1-g(x))\ge0
\tag{H1}
$$
是凸函数。对任意$\mu\in\mathbb R$、$s\ge a>0$、$Z\sim N(0,1)$，
$$
EJ(\mu+sZ)\ge EJ(sZ)\ge EJ(aZ).
$$
第一个不等式来自$\mu\mapsto EJ(\mu+sZ)$的偶凸性。第二个可由偶凸函数随绝对值非减而逐点比较，也可由Gaussian方差增大的凸序得到。定义
$$
L(\mu,s)=El(\mu+sZ),\quad \ell(a)=L(0,a),\quad
\tau_a^2=Eg(aZ)>0.
$$
于是得到全实轴、无均值上界的比较：
$$
\boxed{Eg(\mu+sZ)\ge\tau_a^2-2[L(\mu,s)-\ell(a)].}
\tag{H2}
$$
右侧允许为负；它是单粒子损失接近最小时才有用的通道约束。无需先证明平均$\mu_i^2$趋零，也没有用Gaussian尾部截断代替不等式。

## 2 任意均值与协方差的总体近优集合

固定$0<\nu\le a^2\le\Lambda<\infty$，其中参数都不随$d$变化。考虑
$$
\mathcal T_d=\{(\mu,\Sigma):\mu\in\mathbb R^d,
\quad \nu I\preceq\Sigma\preceq\Lambda I,
\quad \Sigma_{ii}\ge a^2\ (1\le i\le d)\}.
\tag{H3}
$$
所有协方差条目可训练，没有固定符号、特征向量、等相关或因子载荷；对角上界由谱上界自动给出。取$\eta\sim N(\mu,\Sigma)$，给定$\eta$时自旋$Y_i\in\{-1,1\}$独立，成功概率$\sigma(\eta_i)$，精确观察模型为$Q_\theta$。教师$P_d$满足$E_{P_d}Y_i=0$。标准等权iid prior粒子独立于评分标签，raw总体目标为
$$
F_{d,K}(\theta)=E_{Y\sim P_d,\eta_{1:K}}
\left[-\log\left\{K^{-1}\sum_k\prod_i\sigma(Y_i\eta_{ki})\right\}\right].
$$
令$s_i=\sqrt{\Sigma_{ii}}$、$\Psi_d=\sum_iL(\mu_i,s_i)$。固定标签的负loglikelihood对标准Gaussian输入为$\sqrt{\Lambda d}$-Lipschitz；其均值为$\Psi_d-Y^T\mu/2$。Gaussian最大值界与粒子最小/平均夹界，最后对零边际教师平均，给出
$$
0\le\Psi_d(\theta)-F_{d,K}(\theta)
\le t_{d,K}:=\sqrt{2\Lambda d\log K}.
\tag{H4}
$$
这一步对任意有限$\mu$成立，均值平移不改变集中常数。完整推导及标准Gaussian集中来源见[NOISE_THEORY](NOISE_THEORY.md) §2和来源说明 (historical reference not distributed)。非零教师边际时线性项不会消失，不适用本定理。

**极小值存在。** $l$凸且$l(x)\ge|x|/2$，所以
$$
F_{d,K}(\mu,\Sigma)\ge\tfrac12\|\mu\|_1-t_{d,K}.
\tag{H5}
$$
对每个固定$d,K$，这给出关于均值的统一强制增长性。协方差集合是紧的；在任何有界均值集上，矩阵平方根连续且逐bank损失由
$d\log2+\|\mu\|_1+\sqrt{\Lambda d}K^{-1}\sum_k\|Z_k\|_2$
统一可积支配，故$F$连续。有限sublevel由(H5)限制在某个均值紧盒内，故$F$在(H3)达到最小值。没有将整个无界参数集误称为紧集。

对$\varepsilon\ge0$定义所有总体近优点
$\mathcal A_d(\varepsilon)=\{\theta:F_{d,K}(\theta)\le\min F_{d,K}+\varepsilon\}$。
与可行点$(0,a^2I)$比较，得到
$$
0\le\frac{\Psi_d(\theta)}d-\ell(a)
\le\delta_{d,K,\varepsilon}:=
\sqrt{\frac{2\Lambda\log K}{d}}+\frac\varepsilon d,
\qquad\theta\in\mathcal A_d(\varepsilon).
\tag{H6}
$$
左侧由$L(\mu,s)\ge\ell(s)\ge\ell(a)$成立。所有量均是整向量总nats；$\varepsilon$不是每维误差。

## 3 目标近优性直接迫使发射噪声保留

令
$$
V_d(\eta)=\frac1d\sum_i g(\eta_i),\qquad
S_d=\frac1{\sqrt d}\sum_iY_i.
$$
$V_d$是给定logits时$S_d$的条件方差。逐坐标使用(H2)再平均，得到
$$
E_\theta V_d\ge\tau_a^2-2\delta_{d,K,\varepsilon},
\qquad\theta\in\mathcal A_d(\varepsilon).
\tag{H7}
$$
这是比有界均值参数定位更直接的控制。无界均值类中，不宣称平均$\mu_i^2\to0$；少量大均值可以破坏该参数结论，而(H7)仍成立。

记$L_g=\sup|g'|=2/(3\sqrt3)$。Gaussian Poincare给
$$
\operatorname{Var}_\theta(V_d)
\le E[\nabla V_d^T\Sigma\nabla V_d]
\le\frac{\Lambda L_g^2}{d}.
\tag{H8}
$$
相同的Lipschitz界还给Gaussian下尾集中。因此对任何$u>0$，统一有
$$
P_\theta(V_d<\tau_a^2-2\delta-u)
\le r_d(u):=\min\left\{1,\frac{\Lambda L_g^2}{du^2},
\exp\left[-\frac{du^2}{2\Lambda L_g^2}\right]\right\}.
\tag{H9}
$$
统一谱上界在此不可省略；仅控制对角方差并不足以得到这项集中。

## 4 有限维的所有区间中心上界

给定任意logits，设$m_i=E(Y_i\mid\eta)$、$v_i=1-m_i^2$。条件独立、中心的$X_i=Y_i-m_i$满足
$E(|X_i|^3\mid\eta)=2v_i-v_i^2\le2v_i$。
非同分布Berry–Esseen因此给标准化噪声和的CDF误差至多
$2C_{\rm BE}/\sqrt{dV_d}$，可用保守常数$C_{\rm BE}=1$；原始来源及适用条件已核于NOISE_SOURCE_NOTE。

给定$\eta$，$S_d$近似$N(T_d,V_d)$，其中$T_d=d^{-1/2}\sum_im_i$可以是任意值。任何中心$r$、半宽$h>0$的正态区间质量，均不超过居中时的$G_h(V_d)=2\Phi(h/\sqrt{V_d})-1$。区间的两个CDF端点使误差至多$4C_{\rm BE}/\sqrt{dV_d}$。这里不需要$T_d$的CLT，不需要它有界，也不需要$T_d$与$V_d$独立。

**定理1。** 对任何$u>0$使$v_0=\tau_a^2-2\delta_{d,K,\varepsilon}-u>0$，
$$
\boxed{
\sup_{\theta\in\mathcal A_d(\varepsilon)}\sup_{r\in\mathbb R}
Q_\theta(|S_d-r|\le h)
\le U_d(h,u):=\min\left\{1,G_h(v_0)
+\frac{4C_{\rm BE}}{\sqrt{dv_0}}+r_d(u)\right\}.}
\tag{H10}
$$
证明只需在$V_d\ge v_0$上使用条件界，在补集上以1上界，再用(H9)。严格不等CDF的左极限由正态连续性处理。若$v_0\le0$或右侧等于1，当前预算下此界无信息，不将极限值代入冒充有限证书。

## 5 预测后果与正确指定的负族实例

若$\log K_d=o(d)$、$\varepsilon_d=o(d)$，取
$u_d=\sqrt{2\Lambda L_g^2\log d/d}$，则(H9)的指数项为$d^{-1}$，(H10)给
$$
\limsup_d\sup_{\theta\in\mathcal A_d(\varepsilon_d)}\sup_r
Q_\theta(|S_d-r|\le h)\le G_h(\tau_a^2).
\tag{H11}
$$
若教师同一固定事件$A_d=\{|S_d|\le h\}$的概率趋$p_\star>G_h(\tau_a^2)$，令$q_a=G_h(\tau_a^2)$，则
$$
\liminf_d\inf_{\theta\in\mathcal A_d}
\mathcal E_{\rm Brier}(A_d)\ge(p_\star-q_a)^2>0,
\qquad
\liminf_d\inf_{\theta\in\mathcal A_d}
D(P_d\Vert Q_\theta)\ge\operatorname{kl}(p_\star\Vert q_a)>0.
\tag{H12}
$$
事件log-loss也有同一binary KL下界。这由$q\le q_a+o(1)$、二元KL关于$q<p$的单调性和事件DPI得到；无需高维分布本身弱收敛。有限维版本只在真实$p_d=P_d(A_d)>U_d$时使用$(p_d-U_d)^2$及$\operatorname{kl}(p_d\Vert U_d)$，不能用$p_\star$替换$p_d$。

原零均值负相关两尺度教师的总自旋CLT在NOISE_THEORY §7已自足证明：
$$
S_d\Rightarrow N(0,V_\star^-),\qquad
V_\star^-=1-c\bar\alpha_\star^2,
\quad \bar\alpha_\star=\tfrac12[\alpha(s_H^\star)+\alpha(s_L^\star)].
$$
当$V_\star^-<\tau_a^2$时，取$h=1$满足(H12)。选择$\nu\le(s_L^\star)^2(1-c)$、$\Lambda\ge(s_H^\star)^2(1+c)$及$a<s_L^\star$，教师与比较点均属于(H3)。因此即使候选的全部均值与相关结构可训练，正确指定下的全部足够近总体极小点仍有预测损失。

该信号条件是充分条件，不是必要条件，也不意味着“所有负相关都失败”。K1时$F=\Psi$，其最小值$d\ell(a)$恰由$\mu=0,\operatorname{diag}\Sigma=a^2\mathbf1$的所有可行协方差达到。

正族补偿见证需要另核谱域，不能对任意(H3)参数声称好/坏极小点并存。例如$\nu=a^2$时，floor对角加$\Sigma\succeq a^2I$只允许$\Sigma=a^2I$。对于COMPENSATION_THEORY (historical reference not distributed)中的$a,b,c,R,d_0$，若另取$\nu\le a^2\min\{1-c,1/2\}$以及$\Lambda\ge\max\{b^2(1+c),a^2(1+2cR^2)\}$，并令$d\ge d_0$，则teacher、comp及fixedRfloor均可行；该文件的好/坏K1精确极小点也保留在本无均值界类中。正族补偿的完整观察KL可趋零，与本定理要求的更集中教师信号不冲突。

## 6 本轮新增与限制

核心新增步骤是(H1)–(H2)把原始评分的单粒子主项直接连接到条件发射噪声，不经由有界均值盒中的强凸常数。Gaussian集中、Poincare、Berry–Esseen、正态区间集中及KL粗化都是标准工具，本文件不以工具重新命名申报新颖性。

本定理依然要求标准等权prior评分、零观察边际教师、维数统一谱上界、固定正尺度下限和独立Bernoulli发射。它是总体目标结论，未证明有限样本自由协方差SGD的行为。主包的先前有界均值证明及审阅保持，作为推导历史和独立交叉检查；最终最强噪声陈述应引用本文件及其专门审阅。
