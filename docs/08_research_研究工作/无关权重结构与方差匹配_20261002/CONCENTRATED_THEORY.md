# 同总对数方差下，集中无关权重恢复精确总体目标

日期：2026-10-02。本推导固定相关块发射尺度为1、候选区间\(I=[-R,R]\)、\(0<R<1\)。它只改变独立且不依赖候选参数的无关发射尺度；没有新增RNG、积分或训练。恢复结论针对总体raw评分，有限完整数据和共享粒子集的学习推论见LEARNING_SEPARATION.md (historical reference not distributed)。

## 1. 方差匹配尺度的存在性与量级

令\(G\sim N(0,1)\)、\(\sigma(x)=(1+e^{-x})^{-1}\)，定义

$$
v(a)=\operatorname{Var}[\log\sigma(aG)],\qquad
h_a(x)=\log\cosh(ax/2),\qquad a\ge0.
\tag{C.1}
$$

对称性使奇函数\(aG/2\)与偶函数\(h_a(G)\)不相关，因此

$$
\log\sigma(aG)=\frac{aG}{2}-\log2-h_a(G),\qquad
v(a)=\frac{a^2}{4}+\operatorname{Var}h_a(G),
\quad \frac{a^2}{4}\le v(a)\le\frac{a^2}{2}.
\tag{C.2}
$$

上界使用\(0\le h_a(G)\le a|G|/2\)。函数v连续，并在a>0严格递增：令\(X=|G|\)，则

$$
v'(a)=\frac a2+
2\operatorname{Cov}\!\left(h_a(X),\frac X2\tanh(aX/2)\right)\ge\frac a2>0.
\tag{C.3}
$$

两个协方差因子均随X不减，因而其协方差非负；这也可由独立副本\(X'\)的乘积差表达直接看出。微分由Gaussian二阶矩与局部一致线性包络支配。又\(v(0)=0\)、\(v(a)\to\infty\)。

固定整数n≥1，取n−1个尺度为1/n，另一个尺度\(A_n\)由

$$
v(A_n)+(n-1)v(1/n)=n v(1)
\tag{C.4}
$$

定义。因为\(v(1/n)\le v(1)\)，右端减去弱尺度项至少为\(v(1)>0\)，因此连续严格单调性给出唯一\(A_n>0\)。记\(v_1=v(1)\)。由(C.2)，

$$
2n v_1-\frac{n-1}{n^2}\le A_n^2\le4n v_1.
\tag{C.5}
$$

于是\(A_n=\Theta(\sqrt n)\)，且每个有限n的全部尺度严格为正。这里并没有维数一致的正下限或有限上界：弱尺度趋0，强尺度发散。

设\(G_i\)相互独立，定义归一化无关权重

$$
W_n=\left\{\prod_{i=1}^{n-1}2\sigma(G_i/n)\right\}2\sigma(A_nG_n).
\tag{C.6}
$$

由Gaussian对称性，\(EW_n=1\)，且

$$
\operatorname{Var}(\log W_n)
=(n-1)v(1/n)+v(A_n)=n v_1.
\tag{C.7}
$$

这与n个尺度全为1的diffuse族具有相同总对数方差，但还不意味着两族具有相同权重结构。

## 2. 归一化权重二阶矩和对数矩

对任意a≥0，\(2\sigma(aG)=1+\tanh(aG/2)\)，故

$$
E[2\sigma(aG)]^2=1+E\tanh^2(aG/2)
\le1+\min\{1,a^2/4\}.
\tag{C.8}
$$

独立性于是给出

$$
EW_n^2\le
2\left(1+\frac1{4n^2}\right)^{n-1}
\le2\exp\!\left(\frac{n-1}{4n^2}\right)
\le2e^{1/(4n)}.
\tag{C.9}
$$

此外，\(|\log(2\sigma(x))|\le|x|\)。用\(L^2\)三角不等式，

$$
\|\log W_n\|_2
\le A_n+\frac{n-1}{n}\le A_n+1.
\tag{C.10}
$$

因此归一化权重的二阶矩一致有界，而其对数的二阶矩可以为O(n)。两种矩控制不同：下面的对数目标证明同时使用它们。

## 3. 正归一化粒子均值的对数误差引理

**引理。** 设\(T_1,\ldots,T_K\)是严格正iid变量，
\(ET_1=1\)、\(ET_1^2\le M<\infty\)、\(\|\log T_1\|_2\le L<\infty\)，其中M≥1。记\(\overline T_K=K^{-1}\sum_kT_k\)。则

$$
P(\overline T_K<1/2)\le e^{-K/(8M)},
\tag{C.11}
$$

并有

$$
\begin{aligned}
0\le E[-\log\overline T_K]
&\le \frac{2(M-1)}K+(\log K+L)e^{-K/(16M)},\\
E|\log\overline T_K|
&\le2\sqrt{\frac{M-1}{K}}+(\log K+L)e^{-K/(16M)}.
\end{aligned}
\tag{C.12}
$$

**证明。** 对x≥0，\(e^{-x}\le1-x+x^2/2\)。取\(s=1/(2M)\)，负Laplace变换和Chernoff给

$$
\begin{aligned}
P\!\left(\sum_kT_k<K/2\right)
&\le e^{sK/2}(Ee^{-sT_1})^K\\
&\le\exp\{K(-s/2+s^2M/2)\}
=e^{-K/(8M)}.
\end{aligned}
\tag{C.13}
$$

严格正性及\(\overline T_K\ge T_1/K\)使
\((-\log\overline T_K)^+\le\log K+|\log T_1|\)；
而\((\log\overline T_K)^+\le\overline T_K\)。因此对数可积，Jensen可用。

令\(H(x)=-\log x+x-1\ge0\)。当x≥1/2时，Taylor界给
\(H(x)\le2(x-1)^2\)，并且\(|\log x|\le2|x-1|\)。
当\(\overline T_K<1/2\)时，

$$
H(\overline T_K)\le-\log\overline T_K
\le\log K-\log T_1.
\tag{C.14}
$$

对坏事件用Cauchy–Schwarz与(C.11)，其贡献至多
\((\log K+L)e^{-K/(16M)}\)。
在好事件上，\(E(\overline T_K-1)^2\le(M-1)/K\)，
\(E|\overline T_K-1|\le\sqrt{(M-1)/K}\)。
最后用\(E[-\log\overline T_K]=EH(\overline T_K)\)，得到两条界。□

这个负Laplace证明比直接Paley–Zygmund计数更便于固定阈值1/2。后者若只计\(T_k\ge1/2\)的粒子，直接给出的均值下界通常为1/M量级，不能直接宣称均值超过1/2。

**对数尾条件不可省略。** 严格正与有限二阶矩本身不足以保证期望对数误差有限。例如取\(V\ge e\)，满足\(P(V>u)=1/\log u\)（u≥e），令\(T_0=e^{-V}\)、\(T=T_0/ET_0\)。T严格正且有界，故有有限二阶矩；但对任何固定K，所有\(V_k>u\)的事件概率为\((\log u)^{-K}\)。该事件使\(-\log\overline T_K\ge u+\log ET_0\)，尾积分发散，所以\(E[-\log\overline T_K]=\infty\)。集中Gaussian族满足的(C.10)正是排除此类问题的额外控制。

## 4. 应用于完整Gaussian-logistic评分

相关块为\((X_1,X_2)\sim N(0,C_\rho)\)，其中
\(C_\rho=\left(\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right)\)，
与全部无关因子独立。对固定相关块观测\(y=(y_1,y_2)\)，令

$$
f_{\rho,y}(X)=\sigma(y_1X_1)\sigma(y_2X_2),\quad
\mu_{\rho,y}=Ef_{\rho,y}
=\frac{1+y_1y_2r(\rho)}4,\quad
\mu_*=\frac{1-r(R)}4>0.
\tag{C.15}
$$

其中\(r\)为已知相关矩函数，奇且严格递增，\(1/6\le r'\le1/4\)。
任意固定完整观测的无关符号可以在**单行的prior期望**中吸收进相应Gaussian坐标；它们不会改变W的分布。实际共享粒子评分中的各行符号仍须保留。

归一化完整单粒子权重

$$
T_{\rho,y}=\frac{W_n f_{\rho,y}}{\mu_{\rho,y}}
\tag{C.16}
$$

满足\(ET_{\rho,y}=1\)。因为\(0<f<1\)、\(Ef^2\le Ef=\mu_{\rho,y}\)，

$$
ET_{\rho,y}^2\le M_n:=
\frac{2e^{(n-1)/(4n^2)}}{\mu_*}
\le M_*:=\frac{2e^{1/4}}{\mu_*}.
\tag{C.17}
$$

由\(-\log\sigma(x)\le\log2+|x|\)、两个active边际方差为1，以及(C.10)，

$$
\|\log T_{\rho,y}\|_2\le A_n+c_R
\le L_n:=2\sqrt{n v_1}+c_R,\qquad
c_R=3+2\log2+|\log\mu_*|.
\tag{C.18}
$$

这些界对全部ρ、四种观测模式和完整无关符号一致；也独立于生成参数θ。它们不要求在有限n处近似Gaussian对数权重。

定义完整raw总体评分

$$
F^{conc,\theta}_{n,K}(\rho)
=E_{Y\sim Q_\theta^{(n)},Z}
\left[-\log\left\{\frac1K\sum_k
f_{\rho,Y_{1:2}}(X_k)
\prod_{i=1}^{n}\sigma(a_{n,i}Y_{i+2}G_{ki})\right\}\right],
\tag{C.19}
$$

其中\(a_{n,i}=1/n\)（i<n）、\(a_{n,n}=A_n\)，每个粒子均为独立prior抽取。
积分观测律仍为\(Q_\rho^{(n)}(Y)=2^{-n}\mu_{\rho,Y_{1:2}}\)，故与diffuse族完全相同。记
\(\ell^\theta_{act}(\rho)=E_{Y\sim Q_\theta}[-\log\mu_{\rho,Y_{1:2}}]\)。
逐模式应用引理得到

$$
0\le F^{conc,\theta}_{n,K}(\rho)-n\log2-\ell^\theta_{act}(\rho)
\le \Delta_{n,K}:=
\frac{2(M_*-1)}K+(\log K+L_n)e^{-K/(16M_*)}.
\tag{C.20}
$$

因此，K=n+2时，该差对ρ和θ∈I统一趋0，并且
\(\Delta_{n,n+2}=O(n^{-1})+O((\sqrt n+\log n)e^{-c_R' n})\)，
其中\(c_R'>0\)是仅依赖固定模型的常数。这里减去的是完整确定常数nlog2；极限是active精确交叉熵，减其在θ处的值即为完整joint KL。

## 5. 总体极小点与可守的范围

固定n,K的总体目标在紧区间连续并有极小点。连续性可由单粒子损失的Gaussian线性包络在I上一致支配；n和尺度在这一步均固定。若候选\(\widehat\rho_n\)满足总体regret至多η_n，则由(C.20)中误差**非负**的方向，

$$
\mathcal R_\theta(\widehat\rho_n)
\le\Delta_{n,K}+\eta_n,\qquad
|\widehat\rho_n-\theta|
\le\frac{\sqrt{2(\Delta_{n,K}+\eta_n)}}{\min_I r'}.
\tag{C.21}
$$

不需要给两端误差各加一次Δ。参数界使用
\(\mathcal R_\theta(\rho)\ge|r(\rho)-r(\theta)|^2/2\)。
因此K=n+2、η_n→0时，全部总体近优点统一恢复θ，包含边界θ；这不是导数一致收敛、有限n唯一极小点或SGD保证。

该构造说明总\(\operatorname{Var}(\log W)\)本身不足以决定失真。在集中族，最强因子的方差占比满足
\(v(A_n)/(nv_1)\to1\)，而diffuse族中每项占比为1/n。集中族的归一化二阶矩保持有界；diffuse族则有
\(EW_{diff}^2=(1+E\tanh^2(G/2))^n\)，按指数增长。

这项对照释放了原固定同质无关尺度条件：每个有限维度的尺度仍正，但弱尺度趋0、强尺度增大。它不推翻原固定噪声定理，也不证明任意异质噪声均可恢复。其严格结论是：在完全相同的积分观测任务、相同K和相同总对数方差下，不同无关权重结构可以导致不同的raw总体极小点极限。
