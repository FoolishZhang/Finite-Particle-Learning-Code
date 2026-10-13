# 共线模型的角度Hessian与行长补偿

本轮作者推导；不沿用弱载荷展开，不声称原谱底/n增长定理已覆盖本模型。下面身份对任意固定有限K、有限b、正行长a和固定有限目标分布q(y)成立。数值符号判断另有独立资格，不由代码检查自动证明。

## 1. 角度子问题

Y∈{0,1}^m。给定二维标准Gaussian Z=(z,x)，

\[
A_y(z,x;\phi)=\prod_{j=1}^m
\sigma\{b_j+a_j(\cos\phi_j z+\sin\phi_j x)\}^{y_j}
[1-\sigma\{b_j+a_j(\cos\phi_j z+\sin\phi_j x)\}]^{1-y_j}.
\]

目标是F_K(φ)=sum_y q_y E[-log(K^-1 sum_k A_y(Z_k;φ))]。在φ=0，logit为b_j+a_j z_k。令

\[
p_{kj}=\sigma(b_j+a_jz_k),\quad r_{kj}=y_j-p_{kj},\quad
D_k=\operatorname{diag}\{p_{kj}(1-p_{kj})\},\quad
\pi_k=\frac{A_y(z_k)}{\sum_l A_y(z_l)}.
\]

此时π与第二Gaussian坐标x_k独立。对角度路径φ=tψ，第一次变化为delta W_{j2}=a_jψ_j；第二次变化为delta² W_{j1}=−a_jψ_j²。关于x的奇对称性给所有角度一阶导为0。它不要求b/a驻定。

## 2. Cartesian横向Hessian及必要补偿

先沿直线W_j(t)=(a_j,t v_j)变化。logsumexp二阶求导并条件平均独立x_k（均值0、方差1），得到

\[
H_\perp=\sum_yq_yE\left[\sum_k\pi_kD_k-
\sum_k\pi_k(1-\pi_k)r_kr_k^T\right],\qquad
 g_a=-\sum_yq_yE\sum_k\pi_kr_kz_k,\qquad
 g_b=-\sum_yq_yE\sum_k\pi_kr_k.
\tag{RS1}
\]

但这条直线会在二阶改变行范数，不能直接叫固定边际角度曲率。正确角度Hessian为

\[
H_\phi=\operatorname{diag}(a)H_\perp\operatorname{diag}(a)
-\operatorname{diag}(a\odot g_a).
\tag{RS2}
\]

省掉最后一项，会在b/a非驻点上把行长效应误计为纯角度效应。本研究保留g_b/g_a及补偿项，不将固定角度驻定误称全参数驻定。

## 3. Gaussian分部积分与signed Laplacian

固定y时，对每个z_k，

\[
\partial_{z_k}(\pi_kr_{ki})
=\pi_k(1-\pi_k)(a^Tr_k)r_{ki}-\pi_ka_i p_{ki}(1-p_{ki}).
\]

Gaussian分部积分E[z_k f]=E[f']给H_perp a=g_a。故H_phi 1=0；这是共同潜变量旋转的精确零模。对i≠j定义

\[
w_{ij}=a_i a_j\sum_yq_yE\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj}.
\]

则

\[
(H_\phi)_{ij}=-w_{ij}\ (i\ne j),\qquad
(H_\phi)_{ii}=\sum_{j\ne i}w_{ij},\qquad
\psi^TH_\phi\psi=\sum_{i<j}w_{ij}(\psi_i-\psi_j)^2.
\tag{RS3}
\]

边权可正可负，不能预设这就是正半定图Laplacian。K1时pi=1，角度Hessian精确为0，与单粒子准则不识别相对角度相符。实现同时计算RS2和RS3，检查Ward残差；只强制RS3行和为0不能检验内层积分准确性。

取固定正交基Q，其列空间是1的正交补。如果精确Q^T H_phi Q正定，则共线点在去掉共同旋转后是固定b/a角度子问题的严格局部极小；若有负特征值，则有角度下降方向。都不自动给出b/a驻点、全参数极小、有限步优化器收敛或新运行风险保证。

## 4. 有限K的Laplace降维

固定y，取任意正常数c_y，定义x(z)=A_y(z)/c_y。令

\[
\Psi(u)=E e^{-ux},\quad B_0(u)=E[x e^{-ux}],\quad
B_{rr}(u)=E[x e^{-ux}rr^T],\quad B_f(u)=E[x e^{-ux}f(z)].
\]

由1/s=integral_0^infinity exp(−us)du及1/s²=integral_0^infinity u exp(−us)du、iid粒子交换性，

\[
E\sum_k\pi_k f(z_k)=K\int_0^\infty B_f(u)\Psi(u)^{K-1}\,du,
\]

\[
E\sum_k\pi_k(1-\pi_k)r_kr_k^T
=K(K-1)\int_0^\infty u B_{rr}(u)B_0(u)\Psi(u)^{K-2}\,du.
\tag{RS4}
\]

第二式把sum pi(1−pi)写成sum_{k≠l} x_k x_l/(sum x)^2；残差在第k粒子上。这正是交叉粒子项，不能误写成K E[x² rr/(sum x)²]。对有符号残差逐正负部分作Fubini，绝对值由sum pi(1−pi)≤1−1/K控制。g_a的z项用E sum pi |z|≤E max|z_k|<infinity。c_y不必等于真实EA，缩放在比值中抵消。

令u=exp(v)，第一个积分用u dv，第二个用u² dv。内层一维Gaussian求积后，保存节点上的系数

\[
d_l=\omega_l x_l K\int_{-\infty}^\infty u e^{-ux_l}\Psi(u)^{K-1}\,dv,
\]

\[
c_l=\omega_l x_l K(K-1)\int_{-\infty}^\infty
u^2 e^{-ux_l}B_0(u)\Psi(u)^{K-2}\,dv.
\tag{RS5}
\]

其中RS5第二行的u²是辅助变量的平方。于是sum_l d_l≈1，sum_l c_l≈E sum pi(1−pi)∈[0,1−1/K]，且M≈sum_l c_l r_l r_l^T。代码保留未归一化的数值质量/界残差，不能靠重归一或截断负值通过资格。

对于积分likelihood目标，用一维后验权重omega_l A_l/(sum omega A)；RS1的第二项替为E_posterior rr^T，没有有限K的(1−pi)因子。RS2/RS3及Ward身份仍成立。如果目标q恰是该共线模型自身分布，则条件Bernoulli残差的非对角二阶矩为0，积分目标的角度Hessian为0；这是另一实现夹具，不能误要求其严格正定。

## 5. 合法性与解释边界

logistic实导数有界，pi在[0,1]，相关导数只引入有限次Gaussian坐标。固定K和参数紧邻域内可用可积Gaussian矩支配交换期望/微分；目标字母表有限。积分likelihood的每个状态概率为正，局部下界足以处理其对数导数。Laplace积分身份只依赖正likelihood，不引入新的随机审计家族。

角度路径保持每个logit的N(b_j,a_j²)边际以及每个观察Bernoulli边际，但改变联合依赖；它不是观察等价扩维。规范全正共线anchor使用a=||W_j||构造，不逐行翻转观测标签、不拿原W第一列当幅度，也不使用teacher对齐。它不是把原K8终态直接重命名成精确rank-one点。

解析身份本身不给数值积分的误差证书。本轮按预定Gaussian/外积分规则与Ward门核验矩阵；跨规则吻合仍不等于严格的内层求积余项。数值符号只在声明资格下解读。行长插值两端若异号，连续性至多支持存在性夹区，不证明唯一、单调或真实训练路径必然跨过该点。
