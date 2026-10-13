# 多项式粒子数下的无关维度偏好：固定噪声的一致C¹极限

本页是新解析扩展，基准 `04f6f0e09e3aff396d0b3651ae3d158fc4ed4662`。0新科学积分、随机样本、拟合或数值参数搜索；不改本包已冻结的Fourier/MC或旧研究。以下始终固定发射尺度a>0、active块大小m和相关谱范围，让独立nuisance坐标数n增长。粒子数允许与维度同比，但active块不增长：这不是旧两大组、稠密弱公共因子教师的定理。

## 1. 模型、归一化与主要结论

固定m≥2。教师基础协方差为 $T\oplus I_n$，候选为 $C\oplus I_n$，均值为0，全部坐标的条件Bernoulli发射为 $\sigma(aX_i)$。令 $\mathcal C$ 为单位对角active协方差的一个固定紧集合，明确假设 $T\in\mathcal C$，候选 $C\in\mathcal C$，所以教师在同一候选类内、模型正确指定。该集合满足

\[
0<\nu I_m\preceq C\preceq\Lambda I_m<\infty.
\tag{PK1}
\]

可在该集合的稍大严格正定邻域内取非对角坐标导数。标准等权iid prior粒子独立于教师标签。令 $F_{n,K}(C)$ 为完整标签的negative log-mean likelihood再对K个粒子取期望；K在log内，不将外层重复数合并入K。

定义

\[
U=\log\sigma(aZ),\quad \mu=EU,\quad v=\operatorname{Var}U>0,
\quad s_n=\sqrt{nv},\quad Z\sim N(0,1),
\tag{PK2}
\]
\[
r_a(\rho)=E_\rho\bigl[\tanh(aX_1/2)\tanh(aX_2/2)\bigr],
\]
\[
I_K=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{K-2}dz,
\qquad J_K=K(K-1)I_K.
\tag{PK3}
\]

这里 $\varphi,\Phi$ 是标准正态密度/CDF。对整数K≥2，$J_K=E\max_{k\le K}Z_k$，且 $J_K\sim\sqrt{2\log K}$。下面也将用PK3定义实数K>1时的 $I_K,J_K$。

**定理1（多项式K的一致梯度）。** 固定 $b>0,0<C_0<\infty$。对每个active非对角坐标i<j，

\[
\boxed{
\sup_{2\le K\le C_0n^b}\sup_{C\text{满足PK1}}
\left|\frac{s_n}{J_K}\partial_{C_{ij}}F_{n,K}(C)
+\frac{a^2}{4}\{r_a(T_{ij})+r_a(C_{ij})\}\right|\longrightarrow0.}
\tag{PK4}
\]

K的上确界限于整数；不含K1。误差是归一化梯度的**一致绝对误差**，不在领先项为0的地方声称相对梯度误差。固定a与PK1给实际logit协方差固定正下界 $a^2\min(\nu,1)$，没有将a送往0。

定理1的标量实际极小点后果在第7节完整证明。技术核心是下一节至第5节的相对密度/尾界和pair积分，它们处理 $1/K$ 量级的上尾；普通绝对Berry–Esseen误差不足以替代这些步骤。

## 2. Gaussian-logistic单坐标与Gumbel卷积

将每个nuisance标签符号吸收进其独立Gaussian prior坐标。条件于任意完整教师标签，nuisance的log-likelihood贡献均为n个独立U之和，分布相同且独立于active标签与active prior。对bank k，忽略所有bank共有的nμ后，可写其log-likelihood为

\[
S_{nk}+A_k,\quad S_{nk}=\sum_{l=1}^n(U_{kl}-\mu),\quad
A_k=\sum_{i=1}^m\log\sigma(aY_iX_{ki}).
\tag{PK5}
\]

各 $S_{nk}$ iid，且独立于全部active变量。

令G为标准Gumbel，CDF为 $\exp(-e^{-x})$，均值为Euler常数γ。令 $G_c=G-\gamma$，并记

\[
W_n=\frac{S_n+G_c}{s_n},\qquad q_n,H_n\text{为其密度和CDF}.
\tag{PK6}
\]

Gumbel已完整卷入 $q_n,H_n$，不另作随机截断。其矩母函数在λ<1为

\[
Ee^{\lambda G_c}=e^{-\gamma\lambda}\Gamma(1-\lambda).
\tag{PK7}
\]

此式可由G的密度直接作变量替换 $u=e^{-x}$ 得到。特别是零邻域内所有所需矩与导数有限。

U的密度可显式写出。对u<0，令 $z(u)=\{u-\log(1-e^u)\}/a$，则

\[
f_U(u)=\frac{\varphi(z(u))}{a(1-e^u)},\qquad f_U(u)=0\quad(u\ge0).
\tag{PK8}
\]

当u趋向−∞时，z(u)与u/a之差指数小，故密度及其任意固定阶导数具有Gaussian尾；当u从左趋0时，每次微分至多产生若干 $|u|^{-1}$ 因子和log幂，而Gaussian项含 $\exp[-(\log|u|)^2/(2a^2)+O(|\log|u||)]$，比任何这些幂更快趋0。因此延拓密度为Schwartz函数。乘以 $e^{\lambda(u-\mu)}$ 并归一化后，对λ在任一固定紧邻域内，上述Schwartz界一致。U的矩母函数对所有实λ有限。

下面固定一个足够小的 $\lambda_0>0$，例如小于1/4。设

\[
\kappa(\lambda)=\log Ee^{\lambda(U-\mu)},\quad
g(\lambda)=\log Ee^{\lambda G_c},\quad
B_n(\lambda)=n\kappa(\lambda)+g(\lambda).
\tag{PK9}
\]

有 $\kappa(0)=\kappa'(0)=g(0)=g'(0)=0$、$\kappa''(0)=v$。在 $[-\lambda_0,\lambda_0]$，各固定阶导数一致有界，且 $\kappa''$ 有严格正下界。

## 3. 倾斜局部密度引理与相对上尾

**引理1。** 对任意固定L>0，

\[
\sup_{|z|\le L\sqrt{\log n}}
\left|\frac{q_n(z)}{\varphi(z)}-1\right|
\le O_L\left(\frac{1+(\log n)^{3/2}}{\sqrt n}\right)=o(1).
\tag{PK10}
\]

并且

\[
\sup_{-1\le z\le L\sqrt{\log n}}
\left|\frac{1-H_n(z)}{1-\Phi(z)}-1\right|=o(1).
\tag{PK11}
\]

另有 $\sup_xq_n(x)\le C$，以及任意固定L的右尾密度包络：对 $z\ge L\sqrt{\log n}$，

\[
q_n(z)\le C_L n^{-L^2/2}
\exp\{-L\sqrt{\log n}(z-L\sqrt{\log n})+o(1)\}.
\tag{PK12}
\]

所有常数依赖固定a及固定L，不依赖候选、K或n。

**证明：先给倾斜中心处的局部估计。** 按λ倾斜 $S_n+G_c$ 后，仍是n个独立、同分布的倾斜U与一个独立倾斜Gumbel的和。其均值为 $B_n'(\lambda)$，方差 $\tau_{n,\lambda}^2=B_n''(\lambda)$ 在cn与Cn之间。PK8给倾斜U密度的一致Schwartz界。因其为非退化连续密度，对任意η>0，中心化特征函数 $\psi_\lambda$ 满足

\[
\sup_{|\lambda|\le\lambda_0,\ |t|\ge\eta}|\psi_\lambda(t)|<1,
\qquad \sup_{|\lambda|\le\lambda_0}\int|\psi_\lambda(t)|^2dt<\infty.
\tag{PK13}
\]

第一式在有限t区间由连续性、紧性及非格点性得到；大t部分由密度导数的一致L¹界分部积分得到。第二式由一致L²密度界和Fourier等距性得到。倾斜Gumbel的三阶绝对矩在该λ区间也一致有限。

将倾斜和中心化并按τ标准化。在 $|t|\le\delta\sqrt n$ 上，特征函数对数的三阶展开给

\[
\log\Psi_{n,\lambda}(t)=-t^2/2+O(|t|^3/\sqrt n),
\]

而选择固定小δ后，实际对数与二次式之间插值的实部均不超过 $-ct^2$。因此该区间上与 $e^{-t^2/2}$ 的L¹差不超过
$C n^{-1/2}\int |t|^3e^{-ct^2}dt=O(n^{-1/2})$。
在其补集上，变回未标准化频率，PK13将特征函数L¹积分控制为 $O(\sqrt n\,r^{n-2})$，其中r<1；使用 $|\psi|^n\le r^{n-2}|\psi|^2$ 避免在无限区间积分一个常数。Fourier反演遂给倾斜和在其均值处的密度

\[
f_{n,\lambda}(B_n'(\lambda))
=\frac{1}{\sqrt{2\pi}\,\tau_{n,\lambda}}
\{1+O(n^{-1/2})\},
\tag{PK14}
\]

且相同绝对积分估计给 $\sup_x s_n f_{n,\lambda}(x)\le C$，均对λ一致。这里已展示所需局部极限定理的频率分割与误差，未仅援引普通CLT。

**指数倾斜到moderate区间。** 对 $x=s_nz$、$|z|\le L\sqrt{\log n}$，取唯一小解 $B_n'(\lambda)=x$。由PK9的Taylor展开和正方差下界，

\[
\lambda=\frac{z}{\sqrt{nv}}+O_L((1+z^2)/n),\quad
\lambda x-B_n(\lambda)=\frac{z^2}{2}+O_L((1+|z|^3)/\sqrt n),
\]
\[
\frac{s_n}{\sqrt{B_n''(\lambda)}}=1+O_L((1+|z|)/\sqrt n).
\tag{PK15}
\]

密度的精确去倾斜身份与PK14给
$q_n(z)=s_n e^{B_n(\lambda)-\lambda x}f_{n,\lambda}(x)$；代入PK15即得PK10。

**远尾与相对上尾。** 令 $h_n=L\sqrt{\log n}/s_n$，不再为每个更大的z改变倾斜参数。由PK9，
$B_n(h_n)=L^2\log n/2+o(1)$。倾斜密度的统一上界给

\[
q_n(z)\le C\exp\{B_n(h_n)-h_ns_nz\},
\]

即PK12。积分还给 $1-H_n(L\sqrt{\log n})\le C_Ln^{-L^2/2}/\sqrt{\log n}$。为证PK11，先在更大的固定 $L'>L$ 上应用PK10，对 $[z,L'\sqrt{\log n}]$ 积分，再用PK12控制剩余尾。正态Mills界给
$1-\Phi(L\sqrt{\log n})\ge c_Ln^{-L^2/2}/\sqrt{\log n}$；剩余尾与这个最小分母之比为 $O(n^{-(L'^2-L^2)/2})$。这证明整个区间的**相对**上尾误差趋零。对−1≤z≤0，同一积分证明成立，分母还远离0。取λ=0的统一密度上界则给 $\sup q_n\le C$。□

## 4. 精确Gumbel pair积分

对确定active位移向量 $A=(A_1,\ldots,A_K)$，令
$\pi_i=\exp(S_{ni}+A_i)/\sum_l\exp(S_{nl}+A_l)$。独立标准Gumbel加到各分数后，i成为最大者的条件概率等于 $\pi_i$；可直接用其CDF和替换 $u=e^{-x}$ 核算。于是

\[
E_S\pi_i=P(S_{ni}+G_{ci}+A_i\text{最大}).
\]

对 $A_j$（j≠i）求导，左边导数是 $-E_S\pi_i\pi_j$；右边以最大者的密度积分求导。得到精确身份

\[
\boxed{E_S\pi_i\pi_j
=\frac1{s_n}\int_{\mathbb R}
q_n(z-\delta_i)q_n(z-\delta_j)
\prod_{l\ne i,j}H_n(z-\delta_l)dz,\qquad\delta_l=A_l/s_n.}
\tag{PK16}
\]

softmax导数有界；右边的密度乘积可由一个密度的上界乘另一个可积密度控制，故上述交换合法。Gumbel已卷入密度，证明不需要对随机max的Dirac型二阶导数作错误的有界性假设。

**引理2（增长K的pair核）。** 固定 $b>0,0<C_0,C_1<\infty$。若 $2\le K\le C_0n^b$ 且 $\max_l|A_l|\le C_1\sqrt{\log n}$，则

\[
\sup_{K,A,i\ne j}\left|
\frac{s_nE_S\pi_i\pi_j}{I_K}-1\right|\longrightarrow0.
\tag{PK17}
\]

**证明。** 选固定大L，例如 $L^2>4b+4$，并记 $\Delta_n=C_1\sqrt{\log n}/s_n$。在 $0\le z\le L\sqrt{\log n}$、$|\delta|\le\Delta_n$ 上，PK10、PK11及正态密度/Mills比的平移界给

\[
q_n(z-\delta)=\varphi(z)(1+o(1)),\quad
1-H_n(z-\delta)=(1-\Phi(z))(1+o(1)),
\tag{PK18}
\]

误差一致。具体，密度平移比是 $\exp(z\delta-\delta^2/2)$；上尾平移的对数比绝对值至多 $C|\delta|(1+z+|\delta|)$。二者都由 $O(\log n/\sqrt n)$ 控制。

因为正半轴的正态上尾p不超过1/2，对 $p\in(0,1/2]$ 及小η，
$\log(1-(1+O(\eta))p)/\log(1-p)=1+O(\eta)$ 一致成立。因此存在 $\varepsilon_n\to0$，使

\[
\Phi(z)^{1+\varepsilon_n}
\le H_n(z-\delta)\le\Phi(z)^{1-\varepsilon_n}.
\tag{PK19}
\]

这一步保留了 $1/K$ 级别的相对上尾信息；不能改成累乘 $(1+\varepsilon_n)^K$。PK16在该正区间上的积分遂夹在 $(1\pm o(1))$ 乘

\[
\int_0^{L\sqrt{\log n}}\varphi(z)^2
\Phi(z)^{(K-2)(1\pm\varepsilon_n)}dz
\tag{PK20}
\]

之间。

为比较这两个积分，先核PK3中的J身份。对实数x>1，CDF $\Phi^x$ 定义一个概率分布；分部积分给其均值

\[
J_x=x\int z\varphi(z)\Phi(z)^{x-1}dz
=x(x-1)I_x.
\]

在 $t_\pm=(1\pm\epsilon)\sqrt{2\log x}$ 处用正态Mills界，得到该分布位于 $[t_-,t_+]$ 的概率趋1；其右侧期望余项由 $x\int_{t_+}^\infty(1-\Phi(t))dt$ 控制，负部期望由 $2^{-(x-1)}\int_0^\infty\Phi(-t)dt$ 控制。因此 $J_x\sim\sqrt{2\log x}$。取 $x=2+(K-2)(1\pm\varepsilon_n)$ 后，

\[
\frac{I_{2+(K-2)(1\pm\varepsilon_n)}}{I_K}\to1
\tag{PK21}
\]

对这里的K一致：任意K序列若有有界子列，用连续性；若趋∞，用 $x/K\to1$ 与J的上述渐近。这个子列论证排除了所有可能的一致性失败序列。

还须去掉PK20的截区。对于 $z>L\sqrt{\log n}$，固定倾斜包络PK12在平移 $|\delta|\le\Delta_n$ 后只多出 $e^{O(\log n/\sqrt n)}$。故该区间的实际pair积分不超过 $C n^{-L^2}/\sqrt{\log n}$。除以 $I_K=J_K/[K(K-1)]$ 后仍趋0，因为 $K^2\le Cn^{2b}$ 且 $J_K\ge J_2>0$。相同估计控制正态比较积分的右尾。

当K→∞时，负半轴满足 $H_n(z-\delta)\le H_n(\Delta_n)\le c<1$（可取固定c=.6，当n充分大）。结合 $\sup q_n\le C$，其实际pair积分至多 $Cc^{K-2}$；除以I_K后仍趋0，因为 $K^2c^{K-2}/J_K\to0$。正态比较积分的负半轴同理。

若K有界，不能丢掉负半轴。此时q及H在每个固定紧区间上一致趋于 $\varphi,\Phi$，平移也一致趋0；区间外用 $\sup q_n$ 乘尾概率控制。$W_n$ 的均值为0、方差为 $1+O(1/n)$，故该尾可用Chebyshev界一致压小。这给整个实轴积分趋于I_K。结合有界与发散K两种子列、PK20–PK21，即证PK17。□

## 5. active坏事件与Price梯度的一致控制

对active非对角坐标i<j，设

\[
H_k=Y_iY_j\sigma(-aY_iX_{ki})\sigma(-aY_jX_{kj}),\qquad |H_k|\le1.
\]

固定对角Gaussian协方差的Price公式给

\[
\partial_{C_{ij}}F_{n,K}
=-a^2E\sum_k\pi_k(1-\pi_k)H_k
=-a^2E\sum_{k<l}\pi_k\pi_l(H_k+H_l).
\tag{PK22}
\]

两个对称非对角元抵消协方差微分中的1/2。公式可直接由每bank log-likelihood对两个不同active坐标求导并对Gaussian密度两次分部积分得到；nuisance参数不参与该导数。

由 $|A_k|\le m\log2+a\sqrt m\|X_k\|$ 和 $\|X_k\|\le\sqrt\Lambda\|Z_k\|$，可以取

\[
T_n=m\log2+2a\sqrt{m\Lambda(b+3)\log n}.
\tag{PK23}
\]

Gaussian积分给 $Ee^{\|Z\|^2/4}=2^{m/2}$，Markov与union bound于是给

\[
\sup_C P\{\max_{k\le K}|A_k|>T_n\}
\le C_0 2^{m/2}n^{-3}.
\tag{PK24}
\]

此界对标签也一致，不需要各bank标签独立。原Price表达中
$\sum_k\pi_k(1-\pi_k)\le1$，所以该坏active事件对归一化梯度的影响至多 $C(s_n/J_K)n^{-3}=o(1)$。这里使用的是active事件；没有将这一界误用于一个显式截断的Gumbel二阶密度。

在好active事件上，条件于全部active变量后应用引理2。其pair核相对误差一致为o(1)，乘上 $|H_k+H_l|\le2$ 并对 $K(K-1)/2$ 项求和，误差为 $o(J_K/s_n)$，而不是o(1)再乘K²。去掉好事件限制产生的对应标记误差也由PK24控制。因此

\[
\partial_{C_{ij}}F_{n,K}
=-\frac{a^2I_K}{s_n}(K-1)\sum_k EH_k
+o(J_K/s_n).
\tag{PK25}
\]

候选prior独立于教师。写 $t_i=\tanh(aX_i/2)$，则
$H_k=(Y_iY_j-Y_it_j-Y_jt_i+t_it_j)/4$。候选均值0给 $Et_i=Et_j=0$，教师的自旋pair矩为 $r_a(T_{ij})$。所以

\[
EH_k=\frac{r_a(T_{ij})+r_a(C_{ij})}{4}.
\tag{PK26}
\]

代入PK25并用PK3，证得定理1。注意出现的是teacher与candidate pair矩之和，不是二者之差；该结论来自近并列积分，而非把softmax直接替换成单个winner。

## 6. 标量势的一致收敛

以下专门取active块大小2，教师相关固定为 $\theta\in[0,R]$，候选 $\rho\in[-R,R]$，其中 $0<R<1$。记同一尺度下的教师pair矩为 $r_\theta=r_a(\theta)$。PK4积分给

\[
\sup_{2\le K\le C_0n^b}\sup_{|\rho|\le R}
\left|\frac{s_n}{J_K}\{F_{n,K}(\rho)-F_{n,K}(0)\}
-V_\theta(\rho)\right|\to0,
\tag{PK27}
\]

其中

\[
V_\theta(\rho)=-\frac{a^2}{4}
\left[r_\theta\rho+\int_0^\rho r_a(u)du\right].
\tag{PK28}
\]

同时归一化一阶导数一致趋于 $V_\theta'$。r为奇函数，而且Gaussian Price公式给

\[
r_a'(\rho)=\frac{a^2}{4}E_\rho[
\operatorname{sech}^2(aX_1/2)\operatorname{sech}^2(aX_2/2)]>0.
\tag{PK29}
\]

因此V严格凹。若θ>0，
$V_\theta(R)-V_\theta(-R)=-a^2r_\theta R/2<0$，故V唯一最小点是R。若θ=0，V偶对称，恰有两个最小点±R。

## 7. 从C¹到实际端点：无需C²结论

**定理2。** 对固定 $a,R,b,C_0$ 和固定θ>0，存在n0，使所有 $n\ge n_0$、所有整数 $2\le K\le C_0n^b$，原总体目标 $F_{n,K}$ 的唯一全局极小点恰为R。对θ=0，充分大n时恰有两个全局极小点±R。

**证明。** 原目标连续，紧区间上存在极小点。先设θ>0，取固定小δ，使 $V_\theta'$ 在 $[R-\delta,R]$ 上严格小于某个负常数。因为V的唯一最小点为R，闭集 $[-R,R-\delta]$ 与最小值之间有严格正gap。PK27的一致收敛先将原目标的全部全局极小点排除出该闭集。再由PK4，在端点邻域 $[R-\delta,R]$ 上原导数对所有允许K严格为负；故邻域内唯一可能的极小点是R。这里第一步用值gap排除远处，第二步用梯度符号排除近处，不能仅由极小点收敛就宣称有限n精确贴边。

当θ=0，V在去掉两个端点邻域后的闭中部具有严格正gap，故先排除中部。左端点邻域的极限导数严格为正，右端点邻域严格为负；一致C¹据此排除两个邻域的内部。原教师独立且Gaussian符号对称，精确有 $F_{n,K}(\rho)=F_{n,K}(-\rho)$，所以两端值相等，且没有其它极小点。□

对于 $0\le\theta<R$，这些错误端点的完整观察KL不随n变化。θ>0时为

\[
\operatorname{kl}\left(\frac{1+r_a(\theta)}2\middle\Vert\frac{1+r_a(R)}2\right)>0;
\tag{PK30}
\]

θ=0时两端给同一个严格正值。严格正性由PK29的严格单调性保证；当θ=R，端点恰是真模型，不能套用正风险结论。active同/异标签事件也有固定正的Brier excess（概率差的平方）。这些不是均匀全维query或全局count风险下界。

## 8. 全近优集合、K与维度同比及极限次序

对任一允许的整数序列K_n，定义

\[
\mathcal A_n(\epsilon_n)=\{\rho:F_{n,K_n}(\rho)
\le\min_{|u|\le R}F_{n,K_n}(u)+\epsilon_n\}.
\]

若

\[
\epsilon_n=o(J_{K_n}/\sqrt n),
\tag{PK31}
\]

则PK27的紧集argmin论证给
$\sup_{\rho\in\mathcal A_n}\operatorname{dist}(\rho,M_\theta)\to0$，其中θ>0时 $M_\theta=\{R\}$，θ=0时 $M_0=\{-R,R\}$。完整观察KL连续，故这些近优集合的风险一致趋于PK30对应的正值（θ<R）。这是一整个近优集合的结论，不只挑出一个坏候选。

PK31的尺度不能省略。由PK27，所有固定候选之间的raw目标差都是 $O(J_{K_n}/\sqrt n)$；多项式K下该量趋0。因此真参数本身也有o(1)的raw regret。仅假设 $\epsilon_n\to0$ 不足以排除真模型，更不能沿用旧研究中的 $\epsilon=o(d)$ 条件。

取 $K_n=n+2=D$ 即满足 $K_n\le3n$。此时

\[
J_{K_n}\sim\sqrt{2\log n},\qquad
\epsilon_n=o\!\left(\sqrt{\frac{\log n}{n}}\right)
\tag{PK32}
\]

足以控制全部近优集合；定理2也覆盖其实际极小点。极限始终固定a>0、固定active大小/教师和R，让n与K按该关系共同增长，没有先令a趋0，也没有先固定K后直接代入K=D。粒子增长已在倾斜相对密度、上尾和pair核中统一处理。

本页只证明总体目标。它不提供随n增长的有限标签数N、外层bank数B或数值优化预算率，也不追认固定多起点为全域优化。若对每个固定n、K先令理想经验程序的N、B趋∞，那是另一个嵌套学习极限，不能未经预算控制交换为同时增长结论。

## 9. 参数的两个迭代极限不交换；绝对KL仍都趋零

固定 $0<\theta<R<1$，取 $K_n=n+2$，n≥1；其余仍是第6节的两位active块加独立nuisance。对每个n,a任选一个原总体全局极小点 $\rho^*_{n,a}$。下面两个内层极限都处在最终唯一极小点的范围，所以不依赖中间参数处的选择。

先固定n再令a趋0。使用已封存的 WEAK_THEORY.md (historical reference not distributed)（完整仓库路径：`docs/08_research_研究工作/弱发射展开与总体偏好_20261001/WEAK_THEORY.md`）中的一致C²四阶展开，将它限制到只改变 $C_{12}=\rho$ 的凸紧区间。固定 $K_n>2$ 时，其四阶二次项严格凸；投影与两个变分不等式仍适用。其余非对角参数和教师参数均为0，故

\[
\lim_{a\downarrow0}\rho^*_{n,a}
=\operatorname{clip}_{[-R,R]}\!\left(\frac{n+2}{n}\theta\right).
\tag{PK33}
\]

这保留了小n时膨胀目标可能越过R的情形，没有省掉可行投影。再令n趋∞，该值趋于θ。反之，对每个固定a>0，定理2已覆盖 $K_n=n+2$，所以先令n趋∞时极小点最终精确等于R，再令a趋0仍为R。于是

\[
\boxed{
\lim_{n\to\infty}\lim_{a\downarrow0}\rho^*_{n,a}=\theta,
\qquad
\lim_{a\downarrow0}\lim_{n\to\infty}\rho^*_{n,a}=R.}
\tag{PK34}
\]

这是**参数偏好**的非交换，不是a趋零后仍保留固定大小观察损失的结论。完整观察KL始终等于两位active块的KL；在固定rho区间上，$r_a(\rho)=a^2\rho/4+O(a^4)$ 一致成立，故该KL一致为 $O(a^4)$。因此上述两种迭代次序的**未归一化绝对KL最终都为0**。固定a先取高维极限时风险严格为正，与随后再令a趋零使其消失并不矛盾。

最后再次固定风险合同：对于每个随机prior bank，归一化预测器为 $\widehat Q_C$，精确有 $F(C)=H(P)+E_{\rm bank}D(P\Vert\widehat Q_C)$。本页的PK30评价 $Q_C=E_{\rm bank}\widehat Q_C$；错误端点对积分模型的风险更高，不等于raw在其自身随机预测合同上更差。本结论覆盖固定active块与独立nuisance的正确指定、固定正谱族，不覆盖旧稠密双组教师、无约束协方差全类或所有相关粒子方案。
