# 附录A　证明与表示边界

定理1由A.1--A.2证明，定理2由A.3证明。附录F的星号表示固定教师，与正文总体评分为同一对象。正文定理2的\(\varepsilon_*\)为势函数最小点集上的最小风险，(A.19)的\(R_*\)是它的显式下界。A.4--A.9给出补充表示与尺度边界。

全文使用自然对数。风险指一个完整观察的KL散度，不按维数归一化。A.1–A.6始终固定 active 维数 \(m\ge2\)、均值向量 \(\mu\in\mathbb R^m\) 和谱下界 \(0<\delta<1\)。自旋标签为 \(Y_i\in\{-1,1\}\)，原API的二元标签为 \(Y_i^{(01)}=(Y_i+1)/2\)。粒子样本独立于被评分的观察。下述总体结论不假定有限步优化已经达到所要求的近优误差。

## A.1　模型、精确任务与有限粒子导数

学习矩阵 \(C\) 的上三角元素 \(c_{ij}\)，\(i<j\)，可行域为

$$
\begin{gathered}
\mathcal E_\delta^{(m)}
=\{C=C^\top:\operatorname{diag}C=\mathbf1,\ C\succeq\delta I\},\\
|c_{ij}|\le1-\delta,\qquad
\delta\le\lambda_{\min}(C)\le\lambda_{\max}(C)\le m-(m-1)\delta .
\end{gathered}
\tag{A.1}
$$

它是单位对角对称矩阵仿射空间中的凸紧集，具有非空相对内部。令 \(X\sim N(\mu,C)\)，条件发射相互独立，且 \(P(Y_i=y_i\mid X_i)=\sigma(y_iX_i)\)。独立无关logit为 \(\tau_jG_j\)，\(G_j\sim N(0,1)\)，使用相同logistic发射。由对称性，

$$
\begin{aligned}
Q_C^A(y)&=E_C\prod_{i=1}^m\sigma(y_iX_i),&
Q_{C,\tau}^{(n)}(y,u)&=2^{-n}Q_C^A(y),\\
\widehat Q_{C,\tau,K}^{(n)}(y,u)
&=\frac1K\sum_{k=1}^K
\prod_{i=1}^m\sigma(y_iX_{ki})\prod_{j=1}^n\sigma(u_j\tau_jG_{kj}),\\
F_{n,K}^{*,\tau}(C)&=E_{(Y,U)\sim Q_{C_*}^{(n)},\,Z}
[-\log\widehat Q_{C,\tau,K}^{(n)}(Y,U)],&
\mathcal R_*(C)&=D(Q_{C_*}^A\Vert Q_C^A).
\end{aligned}
\tag{A.2}
$$

真实与候选模型的均值和边际方差一致。固定 \(n\) 时，改变无关尺度保持整个观察分布；跨不同 \(n\) 时样本空间不同，保持不变的是active似然比与完整joint KL。每个实现的粒子预测分布都归一化，其期望等于 \(Q_C\)，但其期望对数评分仍不同于积分模型的对数评分。

这与纯潜变量坐标变换不同。对可逆可测变换 \(T_C\)，将先验推前，相应地将发射变为 \(p'_C(y\mid z')=p_C(y\mid T_C^{-1}z')\)，并配对运输粒子。对每个实现和每个 \(C\)，

$$
\frac1K\sum_kp'_C(y\mid T_C(Z_k))
=\frac1K\sum_kp_C(y\mid Z_k).
\tag{A.3}
$$

在可微情形中，计入全部运输导数后，对同一个物理参数 \(C\) 的路径导数也保持一致。改变优化坐标仍须使用链式法则，Adam更新未必不变。不配对采样时，相同的是估计量分布，而非逐实现数值。改变Gaussian无关尺度却保留未补偿的logistic发射，属于观察等价的表示改变，不是式(A.3)。

记 \(b_i=E Y_i=E\tanh((\mu_i+G)/2)\)，以及 \(\kappa_{ij}(c)=E_CY_iY_j-b_ib_j\)。固定边际均值和方差后，后者只依赖对应pair相关 \(c\)。Gaussian分部积分给出

$$
\kappa_{ij}(0)=0,\qquad
\kappa_{ij}'(c)=\frac14E_c[
\operatorname{sech}^2(X_i/2)\operatorname{sech}^2(X_j/2)]>0 .
\tag{A.4}
$$

对一个完整bank，令 \(\pi_k\) 为完整粒子似然的归一化权重，\(\gamma_k=\pi_k(1-\pi_k)\)。其混合logit Hessian与Gaussian非对角Price公式给出

$$
\begin{aligned}
\partial_{c_{ij}}F_{n,K}^{*,\tau}(C)
&=-E\sum_k\gamma_kH_{k,ij},\\
H_{k,ij}
&=Y_iY_j\sigma(-Y_iX_{ki})\sigma(-Y_jX_{kj})\\
&=(Y_i^{(01)}-\sigma(X_{ki}))(Y_j^{(01)}-\sigma(X_{kj})),\\
EH_{k,ij}&=\frac{\kappa_{*,ij}+\kappa_{ij}(c_{ij})}{4},
\qquad \kappa_{*,ij}=\kappa_{ij}(c_{*,ij}).
\end{aligned}
\tag{A.5}
$$

具体而言，第 \(k\) 个粒子的混合导数为 \(-\gamma_kH_{k,ij}\)；同时改变两个对称协方差元素，恰好消去Price公式中的 \(1/2\)。Gaussian线性包络保证损失可积并允许求导交换，混合导数则有界。将 \(H\) 展开为 \([Y_iY_j-Y_iT_j-Y_jT_i+T_iT_j]/4\)，其中 \(T_i=\tanh(X_{ki}/2)\)，再利用观察与粒子独立，即得其期望。Price样本通常不等于某个实现的Cholesky路径导数。

## A.2　一致近并列分析与矩阵极限势

我们稍作推广，对预定无关尺度数组 \(a_{nj}\in[a_-,a_+]\) 证明方向结果，其中 \(0<a_-\le a_+<\infty\) 固定。主定理的单位尺度包含在内。定义

$$
\begin{gathered}
L_a=\log\sigma(aG),\quad v(a)=\operatorname{Var}L_a,\quad
s_n^2=\sum_{j=1}^nv(a_{nj}),\\
I_K=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{K-2}\,dz,\qquad
J_K=K(K-1)I_K=E\max_{k\le K}G_k,\qquad w_{n,K}=J_K/s_n .
\end{gathered}
\tag{A.6}
$$

由于 \(L_a=aG/2-\log2-\log\cosh(aG/2)\)，奇偶正交性给出 \(v(a)\ge a^2/4\)，Gaussian包络则保证 \(s_n^2\asymp n\) 一致成立。分部积分证明 \(J_K\) 的身份。下文的实指数论证给出 \(J_K\sim\sqrt{2\log K}\)，且对整数 \(K\ge2\)，有 \(J_K^2\asymp1+\log K\)。

下述各步明确保持常数的一致性。密度估计中的常数可依赖固定的 \(a_-,a_+\) 与所列截断水平，但不依赖 \(n\)、具体无关数组或其排列。粒子对估计的常数还可依赖固定的 \(C_0,p,C_1\)。只有active分数包络会引入对固定 \(m,\mu,\delta\) 的依赖。每个 \(o(1)\) 均按此一致意义使用；不声称式(A.13)具有显式收敛模量。

### 步骤1　统一倾斜密度：低频与高频

\(L_a\) 的密度在 \(x\ge0\) 处补零后为

$$
f_a(x)=\frac{\varphi(z_a(x))}{a(1-e^x)},\qquad
z_a(x)=\frac{x-\log(1-e^x)}a,\qquad x<0 .
\tag{A.7}
$$

在 \(a\in[a_-,a_+]\) 上，这些密度及任意固定阶导数具有一致的快速衰减包络。左尾由Gaussian乘多项式控制。接近零时令 \(t=-\log(-x)\)，任意固定次微分最多引入 \(e^{ct}\) 与多项式因子，仍被 \(\exp[-t^2/(2a_+^2)+O(t)]\) 压住。因此补零延拓仍光滑。在足够小的固定区间内进行指数倾斜后，同样成立。

令 \(k_a(\lambda)=\log E e^{\lambda(L_a-EL_a)}\)。在 \(|\lambda|\le\lambda_0<1/4\) 上，其任意固定阶导数一致有界；倾斜方差在此紧区间内由连续性与非退化性一致远离零。对中心化倾斜特征函数 \(\psi_{a,\lambda}\)，由于倾斜分布具有密度，每个非零频率处的模严格小于一。连续性使该严格界在紧频率环带上一致成立，高频的一致分部积分再给出衰减，将两段连接后得到

$$
0<c\le k_a''(\lambda)\le C,\qquad
\sup_{a,\lambda,|u|\ge\varepsilon}|\psi_{a,\lambda}(u)|\le q_\varepsilon<1,
\qquad
\sup_{a,\lambda}\int|\psi_{a,\lambda}(u)|^2\,du<\infty .
\tag{A.8}
$$

令 \(S_n=\sum_j(L_{a_{nj},j}-EL_{a_{nj}})\)，加入独立的中心化标准Gumbel变量 \(E\)。其对数矩母函数 \(g_0\) 在 \(\lambda<1\) 上解析。记 \(\mathcal K_n=\sum_jk_{a_{nj}}+g_0\)，并令 \(f_{n,\lambda}\) 为 \(S_n+E\) 指数倾斜后的密度。记 \(B_{n,\lambda}^2=\mathcal K_n''(\lambda)\asymp n\)，以 \(\mathcal K_n'(\lambda)\) 为中心、除以 \(B_{n,\lambda}\) 后的特征函数记为 \(\Psi_{n,\lambda}\)。一致三阶累积量界给出

$$
\begin{gathered}
\log\Psi_{n,\lambda}(u)=-u^2/2+R_{n,\lambda}(u),\\
|R_{n,\lambda}(u)|\le C|u|^3/\sqrt n,\\
\int_{|u|\le\varepsilon\sqrt n}
|\Psi_{n,\lambda}(u)-e^{-u^2/2}|\,du\le Cn^{-1/2}.
\end{gathered}
$$

取足够小的固定 \(\varepsilon\)，指数及其与 \(-u^2/2\) 插值的实部均至多为 \(-cu^2\)。故被积函数之差由 \(C|u|^3e^{-cu^2}/\sqrt n\) 控制，即使积分范围增长，上述积分估计仍成立。在补集上，\(|u|/B_{n,\lambda}\) 一致远离零。保留两个不同无关坐标因子，其余 \(n-2\) 个使用式(A.8)，再用Cauchy–Schwarz，得到

$$
\int_{|u|>\varepsilon\sqrt n}|\Psi_{n,\lambda}(u)|\,du
\le B_{n,\lambda}q^{n-2}
\int_{\mathbb R}|\psi_{a_{n1},\lambda}(t)\psi_{a_{n2},\lambda}(t)|\,dt
\le C\sqrt n\,q^{n-2}.
$$

倾斜Gumbel因子的模不超过一，因而不影响该上界。正态特征函数在低频范围外的积分指数小。Fourier反演于是给出

$$
f_{n,\lambda}(\mathcal K_n'(\lambda))
=\frac{1+O(n^{-1/2})}{\sqrt{2\pi\mathcal K_n''(\lambda)}},
\qquad
\sup_x s_nf_{n,\lambda}(x)\le C .
\tag{A.9}
$$

具体而言，标准化密度与 \(\varphi\) 之差对其自变量一致为 \(O(n^{-1/2})\)；在中心处，由 \(\varphi(0)>0\)，这也是相对误差。同一个Fourier积分给出全域密度上界。至此，式(A.9)两部分对所有预定数组和小倾斜均一致成立。

### 步骤2　相对密度与上尾估计

对 \(x=s_nz\)、\(|z|\le M\sqrt{\log n}\)，严格凸性保证 \(\mathcal K_n'(\lambda)=x\) 有唯一的小参数解。在零处Taylor展开，使用 \(\mathcal K_n'(0)=0\)、\(\mathcal K_n''(0)=s_n^2+\operatorname{Var}E\) 及一致 \(O(n)\) 的三阶导数，得到
\(\lambda=z/s_n+O_M((1+z^2)/n)\)、
\(\lambda x-\mathcal K_n(\lambda)=z^2/2+O_M((1+|z|^3)/\sqrt n)\)，以及
\(s_n/\sqrt{\mathcal K_n''(\lambda)}=1+O_M((1+|z|)/\sqrt n)\)。
令 \(q_n,H_n\) 为 \((S_n+E)/s_n\) 的密度与CDF。精确解除倾斜，即 \(q_n(z)=s_ne^{\mathcal K_n(\lambda)-\lambda s_nz}f_{n,\lambda}(s_nz)\)，结合式(A.9)，证明下式第一行：

$$
\begin{aligned}
\sup_{|z|\le M\sqrt{\log n}}\left|\frac{q_n(z)}{\varphi(z)}-1\right|
&=O_M\!\left(\frac{1+(\log n)^{3/2}}{\sqrt n}\right),\\
\sup_{-1\le z\le M\sqrt{\log n}}
\left|\frac{1-H_n(z)}{1-\Phi(z)}-1\right|&=o(1).
\end{aligned}
\tag{A.10}
$$

证明第二行时，绝对CLT不足以完成相对控制。在式(A.9)的全域密度界中取倾斜参数 \(h=M\sqrt{\log n}/s_n\)。同样的累积量展开给出，对 \(z\ge M\sqrt{\log n}\)，

$$
q_n(z)\le C_Mn^{-M^2/2}
\exp\{-M\sqrt{\log n}(z-M\sqrt{\log n})+o(1)\}.
\tag{A.11}
$$

固定 \(M'>M\)，记 \(T=M\sqrt{\log n}\)、\(T'=M'\sqrt{\log n}\)，以 \(\rho_n(M')\) 表示 \([-T',T']\) 上的相对密度误差。将该估计积分至 \(T'\)，之后使用取 \(M'\) 的式(A.11)，得到

$$
\left|\frac{1-H_n(z)}{1-\Phi(z)}-1\right|
\le \rho_n(M')+
\frac{\int_{T'}^\infty[q_n(t)+\varphi(t)]\,dt}{1-\Phi(T)}
=o(1),\qquad -1\le z\le T.
$$

这里Mills下界给出 \(1-\Phi(T)\ge c_M n^{-M^2/2}/\sqrt{\log n}\)，而遗漏尾部的分子至多为 \(C_{M'} n^{-M'^2/2}/\sqrt{\log n}\)。由 \(M'>M\)，二者比值趋零。这就证明了相对上尾结论，且一致覆盖到 \(z=-1\)。式(A.9)在零倾斜时还给出 \(\sup_zq_n(z)\le C\)。标准化变量中心化，其二阶矩为 \(1+\operatorname{Var}E/s_n^2\)，一致有界。在固定紧区间上积分相对密度估计，再用此矩界控制尾部，可得CDF在紧区间上一致收敛到 \(\Phi\)。

### 步骤3　条件粒子对身份与小平移后的CDF幂

固定完整标签，将无关标签符号吸收到Gaussian期望中。条件于active对数分数 \(A_1,\ldots,A_K\)，粒子权重是 \(S_{nk}+A_k\) 的softmax。对Gumbel最大值表示关于另一个分数求导，得到

$$
E_S[\pi_k\pi_\ell\mid A]
=\frac1{s_n}\int q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz,\qquad d_j=A_j/s_n .
\tag{A.12}
$$

softmax选择概率的导数是 \(-\pi_k\pi_\ell\)；对相应最大值积分求导则使一个CDF变为密度。softmax导数有界，密度乘积可积，因而两种求导均合法。我们将证明，对任意固定的 \(C_1,C_0,p>0\)，

$$
\sup_{\substack{2\le K\le C_0n^p\\
\max_j|A_j|\le C_1\sqrt{\log n}\\k\ne\ell}}
\left|\frac{s_nE_S[\pi_k\pi_\ell\mid A]}{I_K}-1\right|\longrightarrow0 .
\tag{A.13}
$$

记 \(D_n=C_1\sqrt{\log n}/s_n\)，故 \(|d_j|\le D_n\)，并取固定 \(M\) 使 \(M^2>4p+4\)。在 \(0\le z\le T=M\sqrt{\log n}\) 上，恒等式 \(\log[\varphi(z-d)/\varphi(z)]=zd-d^2/2\) 一致为 \(O(\log n/\sqrt n)\)。Mills界给出 \(t\ge-1\) 时 \(\varphi(t)/(1-\Phi(t))\le C(1+t_+)\)，对这个对数上尾导数积分，得到相同阶的正态上尾平移控制。将式(A.10)用于稍大的固定窗口，得到

$$
\sup_{\substack{0\le z\le T\\ |d|\le D_n}}
\left|\frac{q_n(z-d)}{\varphi(z)}-1\right|=o(1),\qquad
\sup_{\substack{0\le z\le T\\ |d|\le D_n}}
\left|\frac{1-H_n(z-d)}{1-\Phi(z)}-1\right|=o(1).
$$

从上尾到幂的步骤可明确核对。记 \(u=1-\Phi(z)\)、\(u'=1-H_n(z-d)\)，此时 \(0<u\le1/2\)，且 \(u'/u=1+o(1)\) 一致成立。例如在 \(0<u\le1/2\)、\(|u'/u-1|\le1/4\) 上，利用 \(\log(1-u)\) 的导数和 \(u\le-\log(1-u)\)，可将 \(|\log(1-u')/\log(1-u)-1|\) 由常数乘 \(|u'/u-1|\) 控制。因此存在对全部数组和平移一致的确定性 \(\varepsilon_n\to0\)，使

$$
\Phi(z)^{1+\varepsilon_n}\le H_n(z-d)\le\Phi(z)^{1-\varepsilon_n},\qquad
\Phi(z)^{(K-2)(1+\varepsilon_n)}
\le\prod_{j\ne k,\ell}H_n(z-d_j)
\le\Phi(z)^{(K-2)(1-\varepsilon_n)}.
$$

两条不等式对所有 \(0\le z\le T\)、\(|d|\le D_n\) 成立。乘以负数 \(\log\Phi(z)\) 时不等号反向，决定了幂的排列。乘积只改变指数，不要求逐点误差小于 \(1/K\)。

### 步骤4　实指数正态粒子对积分与一致比值

对实数 \(x>1\)，将式(A.6)延拓为

$$
I_x=\int_{\mathbb R}\varphi(z)^2\Phi(z)^{x-2}\,dz,\qquad
J_x=x\int_{\mathbb R}z\varphi(z)\Phi(z)^{x-1}\,dz=x(x-1)I_x.
$$

CDF \(\Phi^x\) 定义一个变量 \(M_x\)，其均值为 \(J_x\)。分部积分的边界项为零，因为 \(\varphi(z)\Phi(z)^{x-1}\) 在两端均趋零。在 \(x>1\) 的每个紧子区间上，Mills界为 \(\varphi^2\Phi^{x-2}\) 提供可积包络，故 \(I_x\) 在其上连续且严格为正。

为保持自包含，记 \(b_x=\sqrt{2\log x}\)，取固定 \(0<\eta<1\)，并写 \(M_x^+=\max(M_x,0)\)、\(M_x^-=\max(-M_x,0)\)。Mills界与 \(1-\Phi(t)^x\le x[1-\Phi(t)]\) 给出

$$
\begin{aligned}
E M_x^-&\le 2^{-(x-1)}\int_0^\infty\Phi(-t)\,dt,\\
E M_x^+&\le b_x+x\int_{b_x}^\infty[1-\Phi(t)]\,dt=b_x+O(b_x^{-2}),\\
P(M_x\le(1-\eta)b_x)
&\le\exp\{-x[1-\Phi((1-\eta)b_x)]\}\longrightarrow0.
\end{aligned}
$$

第二行使用 \(x\varphi(b_x)=(2\pi)^{-1/2}\)。最后一行成立，是因为Mills下界使 \(x[1-\Phi((1-\eta)b_x)]\to\infty\)。因此 \(J_x\le b_x+O(b_x^{-2})\)，且 \(J_x\ge(1-\eta)b_x(1-o(1))-o(1)\)，证明了实数 \(x\to\infty\) 时 \(J_x\sim\sqrt{2\log x}\)。

这些事实使步骤3所需比值对允许的整数 \(K\) 一致成立。若一致性失败，可选取 \(n\to\infty\)、\(K_n\)、\(\theta_n\) 的违反序列，其误差始终大于某个固定正数。有界的 \(K_n\) 子列有进一步恒等于某个整数 \(K\) 的子列；此时 \(x_n\to K\)，连续性给出 \(I_{x_n}/I_K\to1\)。否则取 \(K_n\to\infty\) 的子列。由 \(x_n/K_n\to1\)，实指数渐近给出

$$
\frac{I_{x_n}}{I_{K_n}}
=\frac{J_{x_n}}{J_{K_n}}
\frac{K_n(K_n-1)}{x_n(x_n-1)}\longrightarrow1,\qquad
x_n=2+(K_n-2)(1+\theta_n),\quad |\theta_n|\le\varepsilon_n.
$$

两种情形均与固定正误差矛盾。特别地，这覆盖 \(\theta_n=\pm\varepsilon_n\)。该论证不要求 \(K_n\) 以某个速率增长。

### 步骤5　粒子对相对尾界与有界粒子数

正远尾相对于 \(I_K\) 一致可忽略。如式(A.11)，在 \(M\sqrt{\log n}/s_n\) 精确解除倾斜时，也可对 \(z\ge T\) 的平移后密度取得带 \(e^{TD_n}\) 因子的上界。对两个密度上界的乘积积分，得到

$$
\int_T^\infty q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz
\le \frac{C n^{-M^2}e^{2TD_n+o(1)}}{T}=o(I_K),\qquad
I_K\ge c\frac{\sqrt{1+\log K}}{K^2}.
$$

下界来自 \(J_K^2\asymp1+\log K\) 与式(A.6)，有限多个小整数纳入常数。由于 \(K\le C_0n^p\)、\(TD_n=o(1)\)，所列正尾的相对贡献至多为 \(C n^{2p-M^2}/\sqrt{\log n}\to0\)。指数为 \((K-2)(1\pm\varepsilon_n)\) 的正态粒子对积分也具有相同上尾界，因为其CDF因子至多为一。

对负半轴，由紧区间CDF收敛和 \(D_n\to0\)，对所有 \(z\le0\)、所有允许平移和充分大的 \(n\)，均有 \(H_n(z-d_j)\le q_*<1\)。使用 \(\sup q_n\le C\)，再对另一个密度积分，得到

$$
\begin{gathered}
\int_{-\infty}^0q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz\le Cq_*^{K-2},\\
\frac{Cq_*^{K-2}}{I_K}
\le C\frac{K^2q_*^{K-2}}{\sqrt{1+\log K}}\longrightarrow0
\quad(K\longrightarrow\infty).
\end{gathered}
$$

这个极限只依赖 \(K\)，所以也覆盖任意慢发散的序列 \(K_n\)。负半轴正态粒子对积分同理由 \(C2^{-(x-2)}\) 控制；当 \(x=2+(K-2)(1\pm\varepsilon_n)\)、\(K\to\infty\) 时，它为 \(o(I_K)\)。因此对这样的序列，可将正中间区间的正态积分换成全轴积分。步骤3的密度估计和CDF幂界将其夹在 \((1+o(1))I_{2+(K-2)(1+\varepsilon_n)}\) 与 \((1+o(1))I_{2+(K-2)(1-\varepsilon_n)}\) 之间。步骤4和两侧尾界证明了每条发散序列上的式(A.13)。

对有界 \(K\)，不能丢弃负半轴：它对 \(I_K\) 有贡献，尤其在 \(K=2\) 时。取固定整数子列。在每个 \([-B,B]\) 上，平移后的密度与CDF一致收敛到对应正态量。对 \(B>D_n\)，剩余尾部满足

$$
\begin{gathered}
\int_{|z|>B}q_n(z-d_k)q_n(z-d_\ell)
\prod_{j\ne k,\ell}H_n(z-d_j)\,dz
\le C P(|W_n+d_k|>B)\le\frac{C}{(B-D_n)^2},\\
W_n=(S_n+E)/s_n.
\end{gathered}
$$

第二个不等式是Chebyshev不等式，使用步骤2的一致二阶矩界。对应正态尾部也随 \(B\to\infty\) 消失。先固定 \(B\) 令 \(n\to\infty\)，再令 \(B\to\infty\)，便得到完整积分趋于固定正数 \(I_K\)。最后，式(A.13)的任何违反序列均有固定整数子列或发散子列，上述论证已排除二者。全部估计对数组和分数界一致，故得到所述上确界结论，而非仅逐点极限。

### 步骤6　active分数截断、去截断与矩阵一致性

对 \(X_k=\mu+C^{1/2}Z_k\)，记 \(\Lambda=m-(m-1)\delta\)。对全部 \(C\)、标签和粒子，一致有

$$
|A_k|\le m\log2+\sqrt m\,\|\mu\|+\sqrt{m\Lambda}\,\|Z_k\|,
\qquad E e^{\|Z_k\|^2/4}=2^{m/2}.
\tag{A.14}
$$

具体而言，\(-\log\sigma(t)\le\log2+|t|\)，欧氏范数到坐标和的界贡献 \(\sqrt m\)，且 \(\|C^{1/2}\|_{\mathrm{op}}\le\sqrt\Lambda\)。Markov不等式与联合界表明，事件 \(\mathcal B_n=\{\max_k\|Z_k\|\le2\sqrt{(p+3)\log n}\}\) 的补事件概率至多为 \(C_02^{m/2}n^{-3}\)。在该事件上，式(A.14)对充分大的 \(n\) 给出 \(\max_k|A_k|\le C_1\sqrt{\log n}\)，其中一个固定 \(C_1\) 只依赖 \(m,\mu,\delta,p\)。这才是对 \(C\) 一致性的来源；仅凭矩阵类紧性不能将逐点粒子对估计升级为一致估计。

条件于全部active粒子和标签，它们确定分数与标记，而无关和仍独立。将式(A.5)改写为 \(-E\sum_{k<\ell}\pi_k\pi_\ell(H_{k,ij}+H_{\ell,ij})\)。在 \(\mathcal B_n\) 上，式(A.13)使每个条件粒子对质量为 \((I_K/s_n)(1+r_{k\ell}(A))\)。由于 \(|H_{k,ij}|\le1\)，总误差满足

$$
\left|\frac{I_K}{s_n}E\!\left[
\mathbf1_{\mathcal B_n}\sum_{k<\ell}
 r_{k\ell}(A)(H_{k,ij}+H_{\ell,ij})\right]\right|
\le \eta_n\frac{K(K-1)I_K}{s_n}=\eta_n w_{n,K},\qquad
\sup_{A,k\ne\ell}|r_{k\ell}(A)|\le\eta_n\longrightarrow0.
$$

由粒子交换性，去掉事件后的主项恰为 \(-K(K-1)I_KEH_{k,ij}/s_n=-w_{n,K}EH_{k,ij}\)。从主项移除指示函数的代价至多为 \(w_{n,K}P(\mathcal B_n^c)\)。对补事件上的实际导数，\(\sum_k\pi_k(1-\pi_k)\le1\) 给出至多 \(P(\mathcal B_n^c)=O(n^{-3})\) 的绝对误差。对 \(K\ge2\) 一致有 \(J_K\ge J_2>0\)、\(s_n\le C\sqrt n\)，故 \(w_{n,K}\ge c/\sqrt n\)，从而后一误差为 \(o(w_{n,K})\)。代入式(A.5)的标记期望，证明

$$
\sup_{\substack{\mathbf a_n\in[a_-,a_+]^n,\ 2\le K\le C_0n^p\\
C\in\mathcal E_\delta^{(m)}}}
\left|\frac{\partial_{c_{ij}}F_{n,K}^{*,\mathbf a_n}(C)}{w_{n,K}}
+\frac{\kappa_{*,ij}+\kappa_{ij}(c_{ij})}{4}\right|\to0 .
\tag{A.15}
$$

\(K\) 的上确界只取整数。这是缩放后的绝对误差结论，也覆盖梯度为零的位置。回到单位无关尺度，则 \(s_n=\sqrt{nv(1)}\)。沿始终位于凸可行域内的线段 \(I+t(C-I)\) 积分，得到对全部 \(C\) 和允许的 \(K\) 一致成立的

$$
\begin{aligned}
\frac{F_{n,K}^*(C)-F_{n,K}^*(I)}{w_{n,K}}&\longrightarrow V(C),\\
V(C)&=-\frac14\sum_{i<j}
\left[\kappa_{*,ij}c_{ij}+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\right].
\end{aligned}
\tag{A.16}
$$

坐标个数固定且有限，并有一致界，故积分后的一致误差仍趋零。这只证明梯度与函数值结论，不能再对余项求导以推断Hessian收敛。这里不允许 \(m\) 或 \(\mu\) 随 \(n\) 增长。

## A.3　极点谱结构与观察风险

\(V\) 关于自由坐标的Hessian为对角矩阵，其条目为 \(-\kappa_{ij}'(c_{ij})/4<0\)，故 \(V\) 严格凹。严格凹函数在凸集上的极小点必为极点：若它是非平凡凸组合，其值将严格高于端点值的加权平均。仿射双射 \(R=(C-\delta I)/(1-\delta)\) 将可行域映到实相关矩阵集合。

为保持自包含，给出所需经典秩界的短证明。若 \(R=UU^\top\) 的秩为 \(r\)，可取满列秩的 \(U\in\mathbb R^{m\times r}\)，其各行 \(u_i\) 长度为一。若 \(r(r+1)/2>m\)，则线性方程 \(u_i^\top S u_i=0\) 存在非零对称解 \(S\)。取足够小的 \(\varepsilon>0\)，矩阵 \(U(I\pm\varepsilon S)U^\top\) 是两个不同的相关矩阵，平均为 \(R\)；不同性来自映射 \(S\mapsto USU^\top\) 的单射性。因此极点 \(R\) 必满足

$$
\begin{gathered}
\mathcal M=\operatorname*{argmin}_{\mathcal E_\delta^{(m)}}V,\qquad
r_{\max}=\left\lfloor\frac{\sqrt{8m+1}-1}{2}\right\rfloor,\\
C\in\mathcal M\quad\Longrightarrow\quad
\operatorname{rank}\frac{C-\delta I}{1-\delta}\le r_{\max},\qquad
\lambda_1(C)=\cdots=\lambda_{m-r_{\max}}(C)=\delta .
\end{gathered}
\tag{A.17}
$$

特征值按从小到大排列。这个秩限制属于[Li与Tam（1994）](https://doi.org/10.1137/S0895479892240683)已有的极端相关矩阵几何；这里将它用于粒子诱导的极限势。\(C\) 本身始终正定且满秩。

所有active模式概率都严格为正并连续依赖 \(C\)，因此在紧参数类上一致远离零。式(A.4)使pair观察矩识别每个 \(c_{ij}\)，所以只有 \(C=C_*\) 时才有 \(Q_C^A=Q_{C_*}^A\)。每个相对内点 \(C_*\) 都被排除在 \(\mathcal M\) 之外，包括 \(C_*=I\)：此时极限梯度虽为零，严格凹性仍排除其为极小点。

还可给出一般的显式下界。利用 \(\log\cosh x\le x^2/2\) 与Jensen，

$$
\kappa_{ij}'(c)\ge\frac14
E e^{-(X_i^2+X_j^2)/4}
\ge\frac14e^{-(1+\|\mu\|_\infty^2)/2}
=:\beta_\mu>0 .
\tag{A.18}
$$

令 \(q=m-r_{\max}\ge1\)，\(\Delta_*=\lambda_q(C_*)-\delta>0\)。对 \(C\in\mathcal M\)，Weyl不等式及零对角差矩阵的行和界给出
\(\Delta_*\le\|C-C_*\|_{\mathrm{op}}\le(m-1)\max_{i<j}|c_{ij}-c_{*,ij}|\)。
于是某个自旋乘积的均值差至少为 \(\beta_\mu\Delta_* /(m-1)\)。将完整观察映到这个取值为 \(\{-1,1\}\) 的统计量，再用Bernoulli Pinsker，得到

$$
\inf_{C\in\mathcal M}\mathcal R_*(C)
\ge R_*:=\frac12\left[\frac{\beta_\mu\Delta_*}{m-1}\right]^2>0 .
\tag{A.19}
$$

这里没有将重叠pair风险相加，也没有把潜在Gaussian KL下界转成观察KL下界。

对任意预定且允许的 \(K_n\)，由紧性和一致极限(A.16)，

$$
\begin{gathered}
F_{n,K_n}^*(\widehat C_n)-\inf_C F_{n,K_n}^*(C)=o(w_n)
\quad\Longrightarrow\quad
\operatorname{dist}(\widehat C_n,\mathcal M)\to0,\quad
\liminf_n\mathcal R_*(\widehat C_n)\ge R_* .
\end{gathered}
\tag{A.20}
$$

具体而言，极限势的近优误差至多为缩放后的目标近优误差加两倍一致近似误差。离开 \(\mathcal M\) 的任意邻域后，连续极限势都有严格正的间隙。若总体近优误差随机且为 \(o_p(w_n)\)，则距离依概率趋零；对任意 \(\varepsilon>0\)，有 \(P(\mathcal R_*(\widehat C_n)<R_*-\varepsilon)\to0\)，并且 \(\liminf E\mathcal R_*(\widehat C_n)\ge R_*\)。多个极小点未必有相同KL。普通 \(o(1)\) 目标误差不足以推出这些结论，因为 \(w_n\to0\)。

### 四active实验构造

受控实验使用

$$
\begin{gathered}
\mu=(-.6,-.2,.2,.6),\qquad
C_*=\begin{pmatrix}
1&.25&-.15&.10\\.25&1&.10&-.15\\
-.15&.10&1&.25\\.10&-.15&.25&1
\end{pmatrix},\\
\delta=.1,\qquad
\operatorname{spec}(C_*)=(.5,1,1.2,1.3).
\end{gathered}
\tag{A.21}
$$

此时 \(r_{\max}=2\)，所以每个 \(C\in\mathcal M\) 都有两个特征值等于 .1。第二小特征值与真值相差 .9，迫使某个相关误差不小于 .3。式(A.18)给出 \(\beta_\mu=\tfrac14e^{-.68}>1/8\)；最后一步为纯解析比较，因为 \(\log2>2(1/3+1/81)=56/81>.68\)。因此

$$
\mathcal R_*(C)\ge\frac12(.3/8)^2=\frac9{12800}
\quad(C\in\mathcal M),
\qquad
\liminf_n\mathcal R_*(\widehat C_n)\ge\frac9{12800}
\tag{A.22}
$$

其中近优序列满足式(A.20)，随机近优情形有相应结论。这不宣称有限 \(n\) 的总体极小点恰有两条特征值在下界，也不为所报告的512步Adam程序提供谱或风险保证。

初始化同样影响结果的解释。在truth处，缩放后的粒子梯度趋于 \(-\kappa_{*,ij}/2\)，精确似然的梯度则为零。在独立矩阵处，令 \(h_i=E\sigma'(\mu_i+G)>0\)，有

$$
\left.\partial_{c_{ij}}E_*[-\log Q_C^A(Y)]\right|_{C=I}
=-\frac{4h_ih_j\kappa_{*,ij}}{(1-b_i^2)(1-b_j^2)} .
\tag{A.23}
$$

在 \(I\) 处因子分解后，相应得分为
\(4h_ih_j(Y_i-b_i)(Y_j-b_j)/[(1-b_i^2)(1-b_j^2)]\)，从而证明式(A.23)。其符号与raw在 \(I\) 处的领先方向相同。因此，起初朝真实符号移动本身不是错误方向；极限势的全局偏好受到矩阵可行性的约束，不能由各坐标独立取端点描述。

## A.4　两active直观例子与匹配方差的支撑结果

取 \(m=2,\mu=0,\rho\in[-R,R]\)，其中 \(0<R<1\)，令固定真实相关为 \(\theta\in[-R,R]\)，并定义 \(r(\rho)=E_\rho[\tanh(X_1/2)\tanh(X_2/2)]\)。则

$$
\begin{gathered}
Q_\rho^A(y)=\frac{1+y_1y_2r(\rho)}4,\qquad
\mathcal R_\theta(\rho)=\operatorname{kl}\!\left(
\frac{1+r(\theta)}2\middle\Vert\frac{1+r(\rho)}2\right),\\
\ell_s'(\rho)=\frac{r'(\rho)[r(\rho)-s]}{1-r(\rho)^2},\qquad
\frac{(F_{n,K}^\theta)'(\rho)}{w_{n,K}}\to-\frac{r(\theta)+r(\rho)}4 .
\end{gathered}
\tag{A.24}
$$

这里 \(s\) 是经验一致性矩，不是旧四active实验的径向参数。由式(A.4)，\(r\) 为奇函数且严格递增。因此raw极限势为 \(-[r(\theta)\rho+\int_0^\rho r(v)\,dv]/4\)；当 \(\theta\ne0\) 时，其最小点为 \(\operatorname{sign}(\theta)R\)，当 \(\theta=0\) 时则为两端点集合。内点真参数具有正的近优风险；不宣称真实边界点失效。

相同的总log权重方差并不强制相同的极限。令 \(v(a)=\operatorname{Var}\log\sigma(aG)\)。恒等式
\(v(a)=a^2/4+\operatorname{Var}\log\cosh(aG/2)\) 与Gaussian Poincaré给出 \(a^2/4\le v(a)\le a^2/2\)。由连续性及
\(v'(a)=a/2+2\operatorname{Cov}(\log\cosh(a|G|/2),|G|\tanh(a|G|/2)/2)\ge a/2\)，
存在唯一正数 \(A_n\) 满足

$$
\begin{gathered}
v(A_n)+(n-1)v(1/n)=nv(1),\qquad A_n=\Theta(\sqrt n),\\
W_n=2\sigma(A_nG_n)\prod_{j<n}2\sigma(G_j/n),\qquad
EW_n=1,\quad EW_n^2\le2e^{1/(4n)},\quad
\|\log W_n\|_2\le A_n+1 .
\end{gathered}
\tag{A.25}
$$

二阶矩界来自
\(E[(2\sigma(aG))^2]=1+E\tanh^2(aG/2)\le1+\min(1,a^2/4)\)。
对数界来自 \(|\log(2\sigma(x))|\le|x|\) 和 \(L^2\) 三角不等式。观察分布与总log方差均与分散的单位尺度表示完全相同。

必须显式控制负对数尾。若独立同分布的 \(T_k>0\) 满足 \(ET=1\)、\(ET^2\le M\)、\(\|\log T\|_2\le L\)，则

$$
\begin{aligned}
P(\bar T_K<1/2)&\le e^{-K/(8M)},\\
0\le E[-\log\bar T_K]&\le
\frac{2(M-1)}K+(\log K+L)e^{-K/(16M)}=:\Delta_K,\\
E|\log\bar T_K|&\le
2\sqrt{\frac{M-1}K}+(\log K+L)e^{-K/(16M)}=:\varepsilon_K .
\end{aligned}
\tag{A.26}
$$

使用 \(e^{-x}\le1-x+x^2/2\)，并在 \(1/(2M)\) 处取负Laplace界，即得第一行。由于 \(\bar T_K\ge T_1/K\)，且 \((\log\bar T_K)^+\le\bar T_K\)，对数可积。在 \(\bar T_K\ge1/2\) 上使用 \(-\log x+x-1\le2(x-1)^2\) 与 \(|\log x|\le2|x-1|\)；补事件上使用 \(-\log\bar T_K\le\log K-\log T_1\) 和Cauchy–Schwarz。最后，
\(E[-\log\bar T_K]=E[-\log\bar T_K+\bar T_K-1]\)。
仅有正性与二阶矩并不足以控制这一负对数尾。

对任意固定的 \(m,\mu,\delta\)，记 \(f_C(y\mid Z)=\prod_i\sigma(y_i[\mu+C^{1/2}Z]_i)\)，并将式(A.26)用于
\(T=W_nf_C(y\mid Z)/Q_C^A(y)\)。
紧性给出 \(\min_{C,y}Q_C^A(y)>0\)，且 \(f^2\le f\)；Gaussian对数包络使常数一致满足 \(M=O(1)\)、\(L=O(\sqrt n)\)。故

$$
\begin{gathered}
0\le F_{n,K}^{*,\mathrm{conc}}(C)-n\log2-E_*[-\log Q_C^A(Y)]
\le\Delta_{n,K},\\
\Delta_{n,n+2}=O(n^{-1})+O((\sqrt n+\log n)e^{-cn}).
\end{gathered}
\tag{A.27}
$$

对总体 \(\eta\)-近优点，与 \(C_*\) 比较即可得 \(\mathcal R_*(\widehat C)\le\Delta_{n,K}+\eta\)。这证明了集中表示在完整矩阵类上的总体恢复，而式(A.20)给出分散表示的总体失效。集中尺度离开了任何固定正紧区间。

为明确范围，既有的实际共享bank学习分离仍限于一维。这里 \(\widehat F\) 对 \(N\) 行完整观察和 \(B\) 个完整prior bank的损失取平均，bank在全部数据行及候选之间共享。对任意这样的数据集和大小为 \(M_n\) 的网格 \(\mathcal G_n\)，记 \(\ell_{\mathcal D}^{A}=N^{-1}\sum_j[-\log Q_\rho^A(Y_j)]\)。由式(A.26)、三角不等式和最大值不超过总和，
\(E[\max_{\rho\in\mathcal G_n}|\widehat F^{\mathrm{conc}}(\rho)-n\log2-\ell_{\mathcal D}^{A}(\rho)|\mid\mathcal D]\le M_n\varepsilon_{n,K}\)。
该界适用于任意预定 \(B\ge1\)，不要求数据行或候选之间独立。若 \(K=n+2\)、\(M_n=O(n^{1/4})\)、网格宽度趋零，且与bank独立的iid数据满足 \(N_n\to\infty\)，网格近优误差为 \(o_p(1)\)，则集中表示学习一致。证明是四种active模式概率一致为正，因此经验频数给出精确似然的一致大数律，再结合网格近似和上述最大误差界。风险有界，所以期望风险也收敛。

分散表示需要更细的精度。仅条件于无关权重 \(a_k\)，两active路径梯度 \(h\) 满足

\[
h^2\le(\sum a_kf_k(X'_{k2})^2)(\sum a_k/f_k).
\]

分别处理相同与不同粒子下标，并使用 \(f\le1\)、
\(E(X_2')^2=(1-\rho^2)^{-1}\) 和
\(E(1/f)\le C_R=1+2e^{1/2}+e^{1+R}\)，得到 \(Eh^2\le C_R/(1-R^2)\)。
由 \(|h|\le\max_k\|(Z_{k1},Z_{k2})\|/\sqrt{1-R^2}\) 可交换求导。对任意数据行平均使用Jensen，再用积分Cauchy–Schwarz，得到

$$
E[(\operatorname{osc}_{[-R,R]}e_B)^2\mid\mathcal D]
\le\frac{4R^2C_R}{B(1-R^2)},\qquad
e_B=\widehat F-E(\widehat F\mid\mathcal D).
\tag{A.28}
$$

符号对称性使条件均值关于经验一致性矩 \(s\) 为仿射函数；在两个不同的固定真实相关处应用式(A.15)，即可对 \(s\in[-1,1]\) 一致识别两个系数。当 \(B=n,K=n+2\) 时，\(nw_n^2\asymp\log n\)，式(A.28)因而给出 \(\operatorname{osc}e_B=o_p(w_n)\)。在包含端点的预定网格上，分散表示的网格近优误差 \(o_p(w_n)\) 因此得到式(A.24)的端点极限；在相同数据和bank数下，集中表示的网格近优误差 \(o_p(1)\) 给出零风险。这是一项限定的一维共享bank学习比较；式(A.28)不是多维函数上确界的控制，也不是Adam定理。

## A.5　保持观察模型的尺度路径与机制恒等式

条件于active粒子和一个观察，令 \(\alpha_k>0\) 为归一化active似然。独立公平无关通道给出均值为一的权重 \(W_k(\tau)=\prod_j2\sigma(u_j\tau_jG_{kj})\)。两次使用Jensen可得

$$
\begin{gathered}
0\le E[-\log\sum_k\alpha_kW_k(\tau)]
\le\sum_jE\log\cosh(\tau_jG/2)\le\frac18\|\tau\|_2^2,\\
\widehat F^{\mathrm{fact}}_{n,K}=n\log2+\widehat F^{\mathrm{raw}}_{0,K}.
\end{gathered}
\tag{A.29}
$$

最后一个身份在配对active粒子时逐实现成立；因子分解仍保留active的有限 \(K\) 偏差。为证明单个正尺度上的严格递增性，条件于全部其他输入和幅度 \(r_k=|G_k|\)。其和可写为 \(\sum_kc_k(1+\epsilon_ku_k)\)，其中 \(u_k=\tanh(\tau r_k/2)\)，\(\epsilon_k\) 为独立公平符号。固定其他符号，令 \(A=c_k+\sum_{\ell\ne k}c_\ell(1+\epsilon_\ell u_\ell)>c_ku_k\)，则

\[
E_{\epsilon_k}[-\log(A+\epsilon_kc_ku_k)]
=-\tfrac12\log(A^2-c_k^2u_k^2),\qquad
\partial_{u_k}=\frac{c_k^2u_k}{A^2-c_k^2u_k^2}>0 .
\]

当 \(\tau>0\) 时，\(du_k/d\tau>0\) 几乎处处成立。对各偏导求和再平均即可；Gaussian最大值控制路径导数，允许交换求导与期望。因此每个正的无关尺度都提供一个保持观察模型不变、向零下降的总体方向。

这一边界并不限于公平边际。对任意固定 \(p\in(0,1)\)，关于 \(\mu\) 的严格单调性给出唯一的光滑偶函数 \(\mu_p(\tau)\)，满足 \(E\sigma(\mu_p(\tau)+\tau G)=p\)。记 \(X_\tau=\mu_p(\tau)+\tau G\)。当 \(\tau\ge0\)，且可微凸函数 \(\psi\) 满足交换求导所需的可积性时，隐函数定理与Gaussian分部积分给出

$$
\begin{gathered}
\mu_p(0)=\log\frac p{1-p},\qquad
\mu_p'(\tau)=-\tau\frac{E\sigma''(X_\tau)}{E\sigma'(X_\tau)},\qquad
|\mu_p'(\tau)|\le\tau,\\
\frac d{d\tau}E\psi(\sigma(X_\tau))
=a_\tau\operatorname{Cov}_{\nu_\tau}(\psi'(\sigma(X_\tau)),G)\ge0,\\
d\nu_\tau=\frac{\sigma'(X_\tau)}{a_\tau}\,d\Phi,\quad a_\tau=E\sigma'(X_\tau).
\end{gathered}
\tag{A.30}
$$

均值约束使 \(G\) 在倾斜分布下被中心化。对凸函数 \(\psi\)，两项都是 \(G\) 的递增函数，故协方差非负；若 \(\psi\) 严格凸且 \(\tau>0\)，则严格为正。将此结论逐粒子用于
\(-\log\sum_k\alpha_kW_{y,k}\)，其中 \(W_1=\sigma(X)/p\)、\(W_0=(1-\sigma(X))/(1-p)\)，再沿共同尺度路径将偏导求和。条件函数严格凸，其求导后的被积量被 \(\tau+\max_k|G_k|\) 控制。故期望评分沿正尺度严格增加，同时保持Bernoulli \(p\)。这是带均值补偿的总导数，不是固定均值偏导或优化器保证。非公平无关标签也不会因此获得公平标签的canonical化身份。

局部地，固定 \(p,K\) 及除当前变化无关坐标以外的全部输入，令 \(\omega_k\) 为其归一化似然权重，\(q=\sum_k\omega_k^2\)，\(c_1=1-p,c_0=-p\)，并定义 \(h_y=-\log\sum_k\omega_kW_{y,k}\)。均值保持和 \(W_y'(0)=c_yG\) 给出
\(Eh_y(\tau)=q c_y^2\tau^2/2+O(\tau^4)\)、
\(Eh_y'(\tau)=q c_y^2\tau+O(\tau^3)\)，以及
\(\operatorname{Var}h_y'(0)=q c_y^2\)。
偶性和Gaussian矩包络保证余项。当 \(\tau=\operatorname{softplus}(u)+\varepsilon\) 时，均值与标准差都乘上精确因子 \(1-e^{-(\tau-\varepsilon)}\)。零floor或 \(\varepsilon\ll\tau\ll1\) 下，raw梯度均值为 \(\tau^2\) 阶，标准差为 \(\tau\) 阶；小raw梯度不能证明恢复。固定正floor及增长的无关维数需要分别控制。

最后，对任意固定参考中心 \(h_{ij}^{\mathrm{ref}}\)，方向实验使用精确恒等式

$$
\begin{aligned}
G_{ij}&=-h_{ij}^{\mathrm{ref}}A-\sum_k\gamma_k(H_{k,ij}-h_{ij}^{\mathrm{ref}}),\\
A&=\sum_k\gamma_k=1-\frac1{\mathrm{ESS}},\qquad EA=1-E(1/\mathrm{ESS}).
\end{aligned}
\tag{A.31}
$$

这里使用完整似然权重，而非noise-only权重；一般而言 \(EA\ne1-1/E(\mathrm{ESS})\)。改变参考中心会使两项反向移动而总和不变，因此分项比例不是因果贡献。固定 \(n\) 并在truth处令 \(K\to\infty\)，期望评分梯度趋零，而 \(EA\to1\)，中心化耦合于是抵消平均标记项。这由有界似然的大数律及 \(\sum_y\partial_{c_{ij}}Q_C(y)=0\) 得到。它解释了为什么仅凭重叠不能判断梯度方向，但没有把分解升级为新的符号判据。

对有限 \(n\ge0\) 与 \(K\ge2\)，另一个两active对照仍然成立：将条件bank均值写作 \(F_s=U+sV\)，Price标记具有固定的一致性符号，且 \(\pi_k(1-\pi_k)>0\)，因此 \(F_+'<0<F_-'\)，并有 \(b=-V'>0\)。确定性加性修正 \(g'=-U'-rV'\) 使 \((F_s+g)'=b(r-s)\)，从而与精确截断似然具有相同极小点。这是优化点一致，不是无偏对数评分或归一化密度。将整个准则替换为任意正权重乘 \(r-s\) 的积分，也有相同极小点；两种构造都不提供通用模拟预算或矩阵学习算法。

## A.6. 条件标签权重与有限数据边界

保持 \(m,\mu,\delta\) 固定，无关尺度为单位，且无关通道独立、公平。令 \(q\) 为 \(2^m\) 个active spin模式上的任意概率向量，允许零分量。比较候选矩阵时保持它不变，并令其独立于新评分粒子。对固定active模式 \(s\)，记 \(\phi_{s,n,K}(C)\) 为任一固定无关标签模式下raw评分的prior期望，令 \(F_{q,n,K}=\sum_s q_s\phi_{s,n,K}\)。Gaussian反射可以在该期望中消去无关标签符号，但不替换实际训练运行的输入。记 \(b_{q,i}=E_qS_i\)，正确的中心矩为

$$
\begin{aligned}
A_{q,ij}&=E_q[(S_i-b_i)(S_j-b_j)]\\
&=\operatorname{Cov}_q(S_i,S_j)+(b_{q,i}-b_i)(b_{q,j}-b_j),\\
E_{q,C}H_{k,ij}&=\frac{A_{q,ij}+\kappa_{ij}(c_{ij})}{4}.
\end{aligned}
\tag{A.32}
$$

当 \(q\) 是经验频数时，\(\operatorname{Cov}_q\) 的分母为 \(N\)，一般不能省略经验边际偏移。展开式(A.5)中的标记并使用标签与粒子的独立性，即得式(A.32)。

近并列论证对标签一致：反射后的无关和具有相同分布，式(A.14)的包络不依赖标签，且 \(|H_{k,ij}|\le1\)。相同截断对每个标签均有 \(O(n^{-3})\) 的补事件概率。在好事件上，相对粒子对误差求和为 \(o(w_{n,K})\)；在补事件上，\(\sum_k\gamma_k\le1\) 控制导数误差。先逐一证明有限个标签的界，再按 \(q\) 平均，不需要除以任何标签概率。因此，对概率单纯形、\(C\in\mathcal E_\delta^{(m)}\) 及整数 \(2\le K\le C_0n^p\) 一致地有

$$
\begin{gathered}
\frac{\partial_{c_{ij}}F_{q,n,K}(C)}{w_{n,K}}
+\frac{A_{q,ij}+\kappa_{ij}(c_{ij})}{4}\longrightarrow0,\\
\frac{F_{q,n,K}(C)-F_{q,n,K}(I)}{w_{n,K}}-V_q(C)\longrightarrow0,\\
V_q(C)=-\frac14\sum_{i<j}
\left[A_{q,ij}c_{ij}+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\right].
\end{gathered}
\tag{A.33}
$$

如式(A.16)一样沿线段积分，即得值极限。只有线性项依赖 \(q\)。对同一个 \(q\)、凸权重 \(\alpha_\ell\)、候选面板 \(C_\ell\) 及其物理均值 \(\bar C\)，以 \(\rho_\ell\) 表示自由相关向量，并记 \(H_{ij}(c)=\int_0^c\kappa_{ij}(u)\,du\)。则

$$
\begin{aligned}
\nabla^2V_q&=-\tfrac14\operatorname{diag}(\kappa_{ij}')
\preceq-\tfrac{\beta_\mu}{4}I,\\
\mathcal J_{V_q}
&:=\sum_\ell\alpha_\ell V_q(C_\ell)-V_q(\bar C)\\
&=-\tfrac14\sum_{i<j}\left[
\sum_\ell\alpha_\ell H_{ij}(c_{ij}^{(\ell)})-H_{ij}(\bar c_{ij})\right]\\
&\le-\tfrac{\beta_\mu}{8}\sum_\ell\alpha_\ell
\|\rho_\ell-\bar\rho\|^2.
\end{aligned}
\tag{A.34}
$$

此固定面板的差与 \(q\) 无关，但极小点和训练生成的面板可以依赖 \(q\)。若不同候选使用不同标签权重，线性项就不能消去。这是极限势的结论，不是有限粒子目标的全域凹性或Hessian收敛结论。

对每个固定 \(q\)，严格凹性与式(A.17)使 \(V_q\) 的极小点位于极点，并保持相同的下界特征值数量界。条件prior期望的 \(o(w)\)-近优点逼近该固定最小集合。统一值收敛并不推出对任意 \(q_n\) 到移动最小集合的距离一致趋零，因为近并列极小点可以切换。若 \(q_n\to q\)，紧性论证给出到极限 \(q\) 的最小集合的距离收敛，不一定给出到每个移动集合的距离收敛。

极点几何本身也不证明相对于精确经验似然的额外模拟误差。为说明这一点，令 \(q\) 集中于单个spin模式 \(s\)。对固定正归一权重 \(a_k\)，要求其独立于 \(C\) 和active粒子；定义 \(F_s^a(C)=E[-\log\sum_k a_k\prod_i\sigma(s_iX_{ki})]\)，其中 \(X_k\sim N(\mu,C)\) 独立同分布。若 \(K\ge2\)，由于 \(\gamma_k>0\)，\(F_s^a\) 的Price导数严格具有 \(-s_is_j\) 的符号。精确模式概率满足

$$
\begin{gathered}
\partial_{c_{ij}}Q_C^A(s)=s_is_jE_C\left[
\sigma'(s_iX_i)\sigma'(s_jX_j)
\prod_{\ell\ne i,j}\sigma(s_\ell X_\ell)\right],\\
C_s=\delta I+(1-\delta)ss^\top,\\
\operatorname*{argmin}_{C\in\mathcal E_\delta^{(m)}}[-\log Q_C^A(s)]
=\operatorname*{argmin}_{C\in\mathcal E_\delta^{(m)}}F_s^a(C)=\{C_s\}.
\end{gathered}
\tag{A.35}
$$

这是因为 \(C_s\) 同时最大化所有signed correlation \(s_is_jc_{ij}\)。从任意其他 \(C\) 到 \(C_s\) 的可行线段上，所有坐标贡献都不增，至少一个严格下降。先条件于正无关权重再积分，可以为当前有限 \(n\) 的logistic模型保留这一结论。若推广到任意随机权重，还须确保所比较的目标或其差有定义。当 \(K=1\) 时，期望raw评分只依赖固定边际，对 \(C\) 为常数，因此raw唯一性失效。这些符号都不是对每个实现的路径梯度作出的声明。

最后，未归一化数值权重需要质量因子。若 \(\widetilde q\ge0\) 的总质量为 \(a>0\)，且 \(\widetilde A_{ij}=\sum_s\widetilde q_s(s_i-b_i)(s_j-b_j)\)，线性性给出

$$
\begin{gathered}
\frac{\partial_{c_{ij}}F_{\widetilde q,n,K}(C)}{w_{n,K}}
\longrightarrow-\frac{\widetilde A_{ij}+a\kappa_{ij}(c_{ij})}{4},\\
\nabla^2V_{\widetilde q}=-\frac a4\operatorname{diag}(\kappa_{ij}').
\end{gathered}
\tag{A.36}
$$

固定面板差也乘以 \(a\)。这些身份不授权修改冻结的数值权重，不宣称不同归一方式的面板具有机器层面的精确相等关系，也不将有限数据的边界偏好解释成相对于经验最大似然的普遍额外风险。

## A.7　自由有效均值与尺度：有限曲率门槛

本节释放有效均值和边际尺度。A.7–A.9的\(Y\in\{0,1\}^m\)是**二元标签**，教师P可为边际\(p_i\in(0,1)\)的任意联合律。定义

$$
\begin{gathered}
X_i=\mu_i+s_iZ_i,\quad Z\sim N(0,C),\quad s_i\ge0,\quad C\in\mathcal E_\delta^{(m)},\\
\mu^0=\operatorname{logit}p,\quad D=\operatorname{diag}\{p_i(1-p_i)\},\quad
\Gamma=\operatorname{Cov}_P(Y),\quad R_Y=D^{-1/2}\Gamma D^{-1/2},\\
f_\theta(y,Z)=\prod_i\sigma\{(2y_i-1)(\mu_i+s_iZ_i)\},\qquad
\overline F_K(\theta)=E[-\log\sum_k\pi_k f_\theta(Y,Z_k)].
\end{gathered}
\tag{A.37}
$$

其中\(\theta=(\mu,s,C)\)，有效粒子iid，整个正无关权重向量\(\pi_k=W_k/\sum_\ell W_\ell\)独立于\((Y,Z_1,\ldots,Z_K)\)。以下局部式不要求不同粒子的W独立。令\(\chi=E\sum_k\pi_k^2\)、\(\omega=1-\chi\)。这是无关块权重，不是(A.31)的完整权重；中心化目标避免额外要求任意无关归一常数可积。

在\((\mu,s)=(\mu^0,0)\)，候选是真实边际的乘积\(Q_0\)。中心化原目标和精确观察风险的均值/尺度梯度均为零。固定C，有

$$
\begin{aligned}
\nabla^2_{\mu\mu}\overline F_K&=D,&\nabla^2_{\mu s}\overline F_K&=0,&
\nabla^2_{ss}\overline F_K&=D-\omega(\Gamma\circ C),\\
\nabla^2_{\mu\mu}\mathcal R&=D,&\nabla^2_{\mu s}\mathcal R&=0,&
\nabla^2_{ss}\mathcal R&=D-\Gamma\circ C.
\end{aligned}
\tag{A.38}
$$

核验时置\(r_i=Y_i-p_i\)。log核的一阶尺度导数为\(r_iZ_{ki}\)，二阶为\(-D_{ii}Z_{ki}^2\mathbf1_{i=j}\)。负log-sum-exp的Hessian减去一阶导的加权协方差。条件于\(Y,\pi\)，粒子独立给
\(E_Z[\sum_k\pi_kZ_{ki}Z_{kj}-(\sum_k\pi_kZ_{ki})(\sum_k\pi_kZ_{kj})]=(1-\sum_k\pi_k^2)c_{ij}\)。
再平均即得第一行。均值score在粒子间相同，交叉块因Gaussian零均值消失；精确积分对应系数1，得到第二行。Gaussian多项式包络允许交换微分与期望。可先对有符号尺度微分，再限制回非负锥。本式推广已有均匀权重\(\chi=1/K\)的局部公式，均匀权重特例不单独作为新颖性。

整个相关类具有尖锐可达门槛。令

$$
\lambda=\lambda_{\max}(R_Y),\qquad B=\delta+(1-\delta)\lambda,
\qquad \sup_{C\in\mathcal E_\delta^{(m)}}\lambda_{\max}(R_Y\circ C)=B.
\tag{A.39}
$$

每个C可写为\(\delta I+(1-\delta)T\)，T为相关矩阵。对\((\lambda I-R_Y)\circ T\)应用Schur乘积定理，得\(R_Y\circ T\preceq\lambda I\)。取\(T=qq^\top\)时，它等价于用\(\operatorname{diag}q\)共轭\(R_Y\)，达到上界。选一个单位最大特征向量v，定义

$$
q_i=\operatorname{sign}(v_i),\quad
C^\dagger=\delta I+(1-\delta)qq^\top,\quad
N=\|D^{-1/2}|v|\|_2,\quad a=D^{-1/2}|v|/N\ge0.
\tag{A.40}
$$

零坐标符号任取。此构造使候选负曲率方向成为物理可行的非负尺度激活。沿\(\mu=\mu^0,s=ta,C=C^\dagger\)，

$$
\overline F_K''(0)=\frac{1-(1-\chi)B}{N^2},\qquad
\mathcal R''(0)=\frac{1-B}{N^2},\qquad
\chi>1-1/B\ \Longleftrightarrow\ (1-\chi)B<1.
\tag{A.41}
$$

若\(\lambda>1\)且严格不等式成立，白化后的原目标Hessian对全部C一致正定，间隔至少\(1-(1-\chi)B\)。紧性和Gaussian导数包络将正性延伸到邻域：产品边际流形在C以外方向局部最小，而充分小的正激活(A.40)降低观察风险、提高原评分。均值块仍正定，故释放均值不消除此结论。由隐函数定理存在保持边际的光滑补偿\(\mu(t)=\mu^0+O(t^2)\)；因均值梯度为零，它具有相同尺度曲率。

当\((1-\chi)B>1\)，构造方向同时降低二者，但不证明全局恢复。等号时二阶式不足。\(\lambda=1\)表示两两不相关，不代表联合独立；纯高阶依赖不由此二阶检验识别。零尺度处C不识别，因此极小是流形，不是唯一完整参数。另一个固定C的无约束负特征向量未必位于尺度非负锥；离开此基点后，无关碰撞也不能单独决定一般梯度符号。

## A.8　自由有效尺度的主阶塌缩定理

固定\(m,\delta\)，取包含\(\mu^0\)的紧均值盒及\(s\in[0,S]^m\)，其中\(|\mu_i|\le U\)。关键界仅用初等不等式。有效正核iid并独立于\(\pi\)时，加权Jensen与\(\log x\le x-1\)给

$$
\begin{gathered}
0\le L(\mu,s)-\overline F_K(\theta)\le\omega M(\theta),\\
L(\mu,s)=\sum_i\{E\operatorname{softplus}(\mu_i+s_iG)-p_i\mu_i\},\\
M(\theta)=E_Y[(E_Zf_\theta(Y,Z))(E_Zf_\theta(Y,Z)^{-1})-1].
\end{gathered}
\tag{A.42}
$$

逆矩指\(E(1/f)\)，不是\(1/Ef\)。记\(S_f=\sum_k\pi_kf_k\)，逐实现Jensen gap不超过\(S_f\sum_k\pi_k/f_k-1\)。取期望后，同索引项贡献\(\sum_k\pi_k^2\)，异索引项贡献\((1-\sum_k\pi_k^2)Ef\,E(1/f)\)，即得(A.42)。

对独立单位Gaussian-logistic无关权重，将(A.12)–(A.13)的所有有效log分数设为0，得到重叠渐近。按坐标子集展开\(f^{-1}\)并使用Gaussian矩母函数，得到统一逆矩资格：

$$
\begin{gathered}
\sup_{y,\theta}E_Zf_\theta(y,Z)^{-1}\le2^m e^{mU+m^2S^2/2}<\infty,\\
\omega_{n,K}=(1+o(1))w_{n,K},\qquad
\sup_{2\le K\le C_0n^b,\,\theta}|\overline F_{n,K}(\theta)-L(\mu,s)|
=O\!\left(\sqrt{\log n/n}\right)\to0 .
\end{gathered}
\tag{A.43}
$$

其中\(C_0,b>0\)固定，K为整数，重叠误差在该范围一致。这里通过已经证明的近并列估计处理实际非Gaussian无关权重，没有替换成Gaussian log权重。逆矩界用于一致渐近资格，不是任意有限激活下实用的数值误差证书。

严格Jensen和Bernoulli交叉熵最小化给
\(L\ge\sum_i[\operatorname{softplus}(\mu_i)-p_i\mu_i]\ge\sum_i h(p_i)\)，且恰在\(\mu=\mu^0,s=0\)等号成立。因此对任意允许的\(K_n\)，

$$
\begin{gathered}
F_{n,K_n}(\theta_n)-\inf_\theta F_{n,K_n}(\theta)=o(1)\\
\Longrightarrow\quad\mu_n\to\mu^0,\ s_n\to0,\ Q_{\theta_n}\to Q_0,\qquad
\mathcal R(\theta_n)\to D(P\Vert Q_0)=\sum_i h(p_i)-H(P).
\end{gathered}
\tag{A.44}
$$

无关目标常数在regret中消失。紧性、一致收敛及唯一均值/尺度极小，给出到产品流形的收敛。有限模式概率连续、严格正且有一致正下界，故KL连续。随机\(o_p(1)\) regret下结论依概率成立；风险有界，因此期望亦收敛。若零尺度仅属于闭包，则使用闭包下确界，并要求允许任意小正尺度。

极限风险为总相关，对所有非独立P都正，包括纯高阶依赖。若称其为偏离本可达到的零风险，还须假定存在可行\(\theta_*\)实现\(P=Q_{\theta_*}\)。主阶结论只需\(o(1)\) regret，而固定尺度的细阶谱偏好要求\(o(w)\)。它不识别有限n的全局极小点，也不证明优化器收敛；尤其，此定理全程把无关尺度固定为1。

本段限制有效尺度为\(s\in[\varepsilon,S]^m\)，其中\(0<\varepsilon\le S\)。若有效尺度有固定下界\(\varepsilon>0\)，假定均值盒包含唯一根\(E\sigma(\mu_{p_i}(\varepsilon)+\varepsilon G)=p_i\)。包络求导给\(d\min_\mu L_i/ds=sE\sigma'(\mu_{p_i}(s)+sG)>0\)，故近优输出趋向\(s_i=\varepsilon,\mu_i=\mu_{p_i}(\varepsilon)\)，C仍自由。该极限流形上由Price公式和\(\sigma'\le1/4\)得

$$
\begin{gathered}
|\operatorname{Cov}_Q(Y_i,Y_j)|\le\varepsilon^2(1-\delta)/16,\\
\liminf_n\mathcal R(\theta_n)\ge\tfrac12
\left(\max\{|\kappa_{*,ij}|-\varepsilon^2(1-\delta)/4,0\}\right)^2,
\quad \kappa_{*,ij}=\operatorname{Cov}_P(2Y_i-1,2Y_j-1).
\end{gathered}
\tag{A.45}
$$

风险界在边际收敛后，对一对自旋乘积矩应用Pinsker。可以取各对下界的最大值，不能相加重叠pair下界。这是floor类的渐近结论，不是边际尚未匹配的任意有限输出界。

## A.9　无关尺度也自由时的边界

本节固定有限粒子数\(K\ge2\)；下文提及\(K=1\)时，仅作另行延拓。

令\(n\ge1\)个独立公平无关坐标共享\(\tau\ge0\)，以\(F_{A,K}(\theta)\)表示有效块有限粒子评分。A.5的条件Jensen界和严格尺度单调性对任意有效\(\theta\)成立，因此

$$
\begin{gathered}
B_{n,K}(\theta,\tau):=F_{n,K}(\theta,\tau)-n\log2-F_{A,K}(\theta),\quad
0\le B_{n,K}\le n\tau^2/8,\quad \partial_\tau F_{n,K}>0\ (\tau>0),\\
\inf_{\theta,\tau\ge0}F_{n,K}=n\log2+\inf_\theta F_{A,K},\qquad
r(\theta,\tau)=r_A(\theta)+B_{n,K}(\theta,\tau).
\end{gathered}
\tag{A.46}
$$

其中r与\(r_A\)分别是相应总体regret，均非负。若极小点存在，共同总体极小点具有\(\tau=0\)及有效有限K目标的极小点。固定n、K、紧有效参数和有界\(\tau\)时，共同regret趋零迫使\(\tau\to0\)：连续性使每个紧子集\(\tau\ge a>0\)上的B有正下界。但没有对增长n、K一致的速率。若\(\tau\ge\varepsilon>0\)，最优边界变为\(\varepsilon\)；带正floor的softplus在有限raw坐标下可能仅有下确界。

以下比较假定真实参数\(\theta_*\)位于所讨论的可行类中。正确指定\(P=Q_{\theta_*}\)时，Jensen给出有用但标准的风险比较：

$$
e_K(\theta_*)=F_{A,K}(\theta_*)-H(P)\ge0,\qquad
\mathcal R(\theta)\le r+e_K(\theta_*)
\quad\text{or}\quad r+e_K(\theta_*)+n\varepsilon^2/8
\tag{A.47}
$$

两种上界分别对应零/正无关floor，各r相对于各自可行类定义。将共同最优值与\((\theta_*,0)\)或\((\theta_*,\varepsilon)\)比较，再用\(F\ge H(P)+\mathcal R+n\log2\)即可。正核、有限有效字母表及Gaussian对数矩允许在固定\(\theta_*\)处应用(A.26)，给\(e_K\to0\)。因而零floor时\(K\to\infty,r\to0\)足够恢复理想总体风险；正floor另加\(n\varepsilon^2\to0\)足够。固定K仍有有效块模拟偏差，K=1时其最优本身就是产品模型。这些均不是有限训练保证。

在\((\mu,s,\tau)=(\mu^0,0,0)\)，上述坐标的完整Hessian为

$$
\nabla^2_{(\mu,s,\tau)}F_{n,K}
=\operatorname{blockdiag}\{D,\ D-(1-1/K)(\Gamma\circ C),\ n/(4K)\}.
\tag{A.48}
$$

C方向及其交叉块为0。每个独立无关坐标的中心化log核导数为G/2，曲率为\(1/(4K)\)；独立性消去交叉项，共同尺度累加n项。精确观察风险的尺度块为\(D-\Gamma\circ C\)，无关块为0。

\newpage

有限n的无关碰撞端点也需区分。对iid Gaussian \(G_{kj}\)，

$$
\begin{gathered}
\chi_{n,K}(\tau)=E\sum_k\left[
\frac{\prod_j\sigma(\tau G_{kj})}{\sum_\ell\prod_j\sigma(\tau G_{\ell j})}\right]^2,
\quad \chi_{n,K}(0)=1/K,\\
\lim_{\tau\to\infty}\chi_{n,K}(\tau)
=\chi_\infty(n,K)=P(M=0)+\sum_{r=1}^K\frac{P(M=r)}r,
\quad M\sim\operatorname{Binomial}(K,2^{-n}).
\end{gathered}
\tag{A.49}
$$

有界支配收敛给连续性，有限正\(\tau\)给\(1/K<\chi<1\)。当\(M\ge1\)，只有全正Gaussian行保留极限权重，且均分；当\(M=0\)，各行\(\sum_j\min(G_{kj},0)<0\)几乎必然有唯一最大值，故一行占优。碰撞有界，再用支配收敛得(A.49)。特别地，

$$
\chi_\infty(n,K)\ge1-\binom K2\,4^{-n}.
\tag{A.50}
$$

这是因为\(1-\chi_\infty\le P(M\ge2)\le E\binom M2\)。端点一般不等于1：固定n、尺度无穷不同于固定正尺度、n增长。

在产品基点及任意有限\(\tau\)，(A.38)使用\(\chi_{n,K}(\tau)\)。因此仅由连续性即可得

$$
\begin{gathered}
(1-1/K)B>1\ \Longrightarrow\ \exists\eta>0:
\overline F_K''(0;\tau)<0\quad(0\le\tau<\eta),\\
\chi_\infty(n,K)>1-1/B\ \Longrightarrow\ \exists T<\infty:
H_s^F(C;\tau)\succ0\quad(C\in\mathcal E_\delta^{(m)},\ \tau>T).
\end{gathered}
\tag{A.51}
$$

第一行曲率沿(A.40)，精确风险曲率保持负。若两个条件都满足，至少过门一次；不推出\(\chi\)单调、交点唯一或有限幅度保证。在\(\tau>0\)，共同目标仍有无关下降方向，故有效尺度正曲率不表示共同驻点。

最后，必须指明局部拓扑。在\(C=I,\tau=0\)，有效原目标Hessian为\(D/K\succ0\)，即使(A.51)第一条件在类内另一C处成立。\((\mu,s,C,\tau)\)的小欧氏邻域因此可能存在局部极小流形。但s=0时全部C代表同一产品模型，可先沿等目标路径转到\(C^\dagger\)，再下降。若采用协方差因子\(\Sigma=AA^\top\)，路径\(A=t\operatorname{diag}(a)L_{C^\dagger}\)已位于A=0的任意小邻域；当该曲率为负时，排除相同的局部极小断言。观察分布也沿下降路径趋向\(Q_0\)。这是坐标相关边界，不是普遍优化器陷阱。实验使用正物理尺度及指定修复规则，不能由这些局部恒等式继承收敛或困陷定理。

