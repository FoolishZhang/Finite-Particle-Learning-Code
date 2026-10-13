# 固定对角 raw 目标的精确 Price 梯度

这是本阶段的通用微分工具与独立数值梯度接口，0新积分、随机样本或拟合。Price 恒等式本身不作为新颖性主张。本文件的独立候选局部结论，与 WEAK_THEORY (historical reference not distributed) 的弱发射实际极小点定理分开陈述；不修改旧连续学习或 fixed-floor 冻结。

## 1. 目标、残差与 Gaussian 微分

令教师标签 $Y\in\{-1,1\}^d$ 具有任意固定分布 P，$b_i=(Y_i+1)/2$。K 为有限正整数，iid prior 粒子 $\eta_k\sim N(\mu,\Sigma)$ 独立于 Y，且 $\Sigma\succ0$。定义

$$
\ell_k=\sum_i\log\sigma(Y_i\eta_{ki}),\qquad
\pi_k=\frac{e^{\ell_k}}{\sum_l e^{\ell_l}},\qquad
\mathcal L=-\log\left(K^{-1}\sum_ke^{\ell_k}\right),
$$
$$
r_{ki}=b_i-\sigma(\eta_{ki}),\qquad
s_{ki}=\sigma(\eta_{ki})[1-\sigma(\eta_{ki})],
\qquad F(\mu,\Sigma)=E\mathcal L.
\tag{P1}
$$

同一粒子块的 Hessian 是

$$
\nabla_{\eta_k}\mathcal L=-\pi_kr_k,\qquad
\nabla_{\eta_k}^2\mathcal L
=\pi_k\operatorname{diag}(s_k)-\pi_k(1-\pi_k)r_kr_k^\top.
\tag{P2}
$$

对 $k\ne l$ 的交叉块为 $\pi_k\pi_l r_kr_l^\top$。这些交叉块存在，但标准 iid 粒子协方差是 $I_K\otimes\Sigma$，沿共同候选协方差变化时，没有跨粒子的协方差导数，所以它们不进入下式。对 antithetic 或其它相关粒子不能直接照搬该删项。

固定均值，沿可微 SPD 路径 $\Sigma(t)$，记 $H=\Sigma'(t)$。Gaussian 分部积分（Price 公式）给

$$
\frac{dF}{dt}
=\frac12\sum_{k=1}^K E\operatorname{tr}\!\left[
H\nabla_{\eta_k}^2\mathcal L\right]
=\frac12\sum_k E\left[
\pi_k\sum_iH_{ii}s_{ki}
-\pi_k(1-\pi_k)r_k^\top Hr_k\right].
\tag{P3}
$$

可从 Gaussian 密度导数 $\partial_t\varphi_\Sigma=\frac12\sum_{ij}H_{ij}\partial_{ij}\varphi_\Sigma$，对每个粒子块两次分部积分得到。残差和权重有界，损失只有线性增长；在任意严格谱紧的路径邻域内，可积包络足以处理微分及边界项。教师 P 不必有零边际，但必须不随当前协方差路径变化。

## 2. 固定对角路径和系数约定

若 $\operatorname{diag}H=0$，则

$$
\boxed{
DF(\Sigma)[H]=-
\frac12\sum_kE[\pi_k(1-\pi_k)r_k^\top Hr_k]
=-\sum_{i<j}H_{ij}\,G_{ij}(\Sigma),
}
\tag{P4}
$$

其中

$$
G_{ij}(\Sigma)=\sum_kE[\pi_k(1-\pi_k)r_{ki}r_{kj}].
\tag{P5}
$$

若把 $\Sigma_{ij}=\Sigma_{ji}$ 作为一个标量坐标，其偏导正是 $-G_{ij}$，没有额外的 1/2。非平凡零对角 H 必为不定矩阵，因此 P4 中的负号本身不确定梯度符号。

对 K1，$\pi_1=1$，P4 对所有固定对角路径恒为零。这是精确总体恒等式：K1 的期望损失只依赖各 logit 边际。有限 bank 的路径导数不必逐实现为零，不能将总体平坦性设为每次采样梯度零的检查条件。

## 3. 零均值独立候选处的严格符号

现取 $\mu=0$、$\Sigma=D=\operatorname{diag}(v_1,\ldots,v_d)$，其中 $v_i>0$。令
$X_{ki}=Y_i\eta_{ki}$。条件于任何标签 Y，X 都具有同一分布 $N(0,D)$，故 X 与 Y 独立。又

$$
r_{ki}=Y_i\sigma(-X_{ki}),\qquad
\pi_k=\frac{\prod_i\sigma(X_{ki})}{\sum_l\prod_i\sigma(X_{li})}.
$$

所以对 $i\ne j$，

$$
G_{ij}(D)=\beta_{ij}(D,K)M_{ij},\qquad
M_{ij}=E_P(Y_iY_j),
$$
$$
\beta_{ij}(D,K)=E_X\sum_k\pi_k(1-\pi_k)
\sigma(-X_{ki})\sigma(-X_{kj}).
\tag{P6}
$$

有限 logits 下所有似然严格正。于是 K>1 时每个 $0<\pi_k<1$，并且 sigmoid 因子在 (0,1)，给

$$
0<\beta_{ij}(D,K)<1-1/K.
\tag{P7}
$$

上界利用 $\sum_k\pi_k(1-\pi_k)=1-\sum_k\pi_k^2\le1-1/K$。K1 则 $\beta_{ij}=0$。严格正性是固定有限 d、K、D 的陈述，没有给出维数一致的正下界。

**推论1。** 在零均值独立候选处，对称非对角协方差偏导为

$$
\boxed{\partial_{\Sigma_{ij}}F=-\beta_{ij}(D,K)M_{ij}.}
\tag{P8}
$$

M 是自旋二阶矩，不是一般非零边际教师的中心化 covariance。若 M 的某个非对角元素非零，且允许完整的固定对角协方差邻域，则取 $H_{ij}=M_{ij}$、$H_{ii}=0$ 给严格下降导数 $-\sum_{i<j}\beta_{ij}M_{ij}^2<0$。足够小的路径步长保持 SPD；额外谱或结构约束仍须允许该方向。特别地，独立点不是这个开放固定对角类的驻点或局部极小点。

若所有非对角 M 都为零，一阶结论只是零；不能据此判断更高阶依赖或全局最优。若独立点落在额外约束的边界，也不能忽略方向可行性。

## 4. 自由载荷与两组共享载荷的 Hessian

先取同质对角 $D=a^2I$。置

$$
\Sigma(\lambda)=a^2\{\operatorname{diag}(1-\lambda_i^2)+\lambda\lambda^\top\},
\qquad \lambda\in(-1,1)^d.
\tag{P9}
$$

每个对角恒为 $a^2$；非对角为 $a^2\lambda_i\lambda_j$。P6 中的 $\beta_{ij}$ 此时因 prior 坐标交换性而对所有不同 i,j 相同，记为 $\beta_{d,K,a}$。在 $\lambda=0$，协方差的一阶载荷导数为零，因此该点总是载荷驻点。协方差 Taylor 展开和 P8 给

$$
F(\Sigma(\lambda))=F(a^2I)
-a^2\beta_{d,K,a}\sum_{i<j}M_{ij}\lambda_i\lambda_j
+O(\|\lambda\|^4),
$$
$$
\boxed{\nabla_\lambda^2F(0)=-a^2\beta_{d,K,a}M_{\rm off}.}
\tag{P10}
$$

余项阶数为4，因为 $\Sigma(\lambda)-a^2I$ 是二次映射。只要 $M_{off}\ne0$ 且 K>1，右侧是非零、对称、迹为零的矩阵，必同时有正和负特征值。所以这是 unrestricted signed-loading 空间中的严格鞍点，而不是负定极大点。沿仅一个坐标载荷改变的轴，模型协方差事实上完全不变，存在精确平坦方向并不阻止其它方向的负曲率。

在非负载荷锥中，若至少有一个 $M_{ij}>0$，取仅 i,j 两个正载荷就给可行严格下降方向。若 teacher pair 全正，则提高共同正载荷有下降方向；不能把这个方向结论扩成 signed Hessian 负定。

若有两组各 m 个坐标，同组 pair 矩分别为 $r_{HH},r_{LL}$、跨组为 $r_{HL}$，并让组内共享 $\lambda_H,\lambda_L$，则限制到该二维子空间得到

$$
\nabla^2_{(\lambda_H,\lambda_L)}F(0)
=-a^2\beta_{d,K,a}
\begin{pmatrix}
m(m-1)r_{HH}&m^2r_{HL}\\
m^2r_{HL}&m(m-1)r_{LL}
\end{pmatrix}.
\tag{P11}
$$

若改用 $\lambda_g=w_g/\sqrt{d-1}$，Hessian 再除以 d−1。二维共享参数的对角项一般不为零，不能沿用全坐标 Hessian 的 trace-zero 论证。正 pair 情形至少有共同正方向的严格负曲率，但整个二维 Hessian 的惯性还取决于显示矩阵的行列式。旧 RUN 的 $w\in[.5,2.5]^2$ 不包含原点，本推论不能当成那个盒内点的局部证书。

## 5. 一般载荷的精确梯度接口及边界

对于 P9 的任意内部载荷，链式法则和 P4 给

$$
\frac{\partial F}{\partial\lambda_i}
=-a^2\sum_{j\ne i}\lambda_jG_{ij}(\Sigma(\lambda)).
\tag{P12}
$$

同样，若教师与候选都在两组内交换，组内/跨组的 G 分别记为 $G_{HH},G_{LL},G_{HL}$，则

$$
\frac{\partial F}{\partial w_H}
=-\frac{a^2}{d-1}
\{m(m-1)w_HG_{HH}+m^2w_LG_{HL}\},
\tag{P13}
$$

低组公式交换 H/L 即得。这提供与直接路径微分不同的独立积分梯度表达，且自动包含“增公共方差同时减私方差”的固定对角合同。它是总体期望公式，不要求一个有限 bank 的 Price 积分估计与同一 bank 的路径估计逐值相等。

一般非对角候选处，$X\mid Y\sim N(0,D_Y\Sigma D_Y)$ 会依赖 Y，所以不能再把 G 分离为正的 teacher pair 系数。即使所有 teacher pair 为正，也不能据 P8 推断 P13 在 comp 附近必为负、comp 必非驻点或载荷越大越好。

对二维弱发射候选 $\eta=aX$、$\operatorname{Cov}X=C_\rho$，P4 的精确接口为

$$
\partial_\rho F_{a,K}=-a^2\sum_kE[\pi_k(1-\pi_k)r_{k1}r_{k2}].
\tag{P14}
$$

这里教师权重是在当前 a 下的真实观察分布；求 candidate rho 导数时教师保持不变。将独立候选写为 $X=aZ$，中心 Gaussian 符号对称及可积的二阶微分包络使 $\beta_{d,K,a}$ 为 a 的偶 $C^2$ 函数，因此 $\beta_{d,K,a}=(K-1)/(4K)+O(a^2)$。同时 $E_{P_a}Y_iY_j=a^2T_{ij}/4+O(a^4)$。P8 因而给交叉系数 $-a^4(K-1)T_{ij}/(16K)+O(a^6)$，与 WEAK_THEORY W11 独立吻合；它单独不能推出 W13 中的二次惩罚系数。

Price 工具的价值是精确区分协方差梯度、载荷 Hessian和可行方向，并为将来的独立梯度数值核验提供接口。超出独立点的实际总体极小点结论，在本阶段由另一个弱发射展开及其一致余项承担，而非由局部符号外推。
