# 附录G. 匹配均值未必保持有限粒子驻定性

## G.1 二粒子反例

考虑三个logistic观测、单位Gaussian边际方差、相互独立的教师坐标，以及已知匹配均值 $\mu_t=(t,t,0)$。记 $\sigma(x)=(1+e^{-x})^{-1}$，$P_t$ 为 $\{-1,1\}^3$ 上的教师标签分布。取两个独立候选粒子 $X_k\sim N(\mu_t,C)$，无无关坐标，定义

$$
\begin{aligned}
F_{0,2}(C;\mu_t)
&=E_{Y\sim P_t}E_X\left[-\log\left\{\frac12\sum_{k=1}^2
\prod_{i=1}^3\sigma(Y_iX_{ki})\right\}\right],\\
g(t)&=\left.\partial_{c_{12}}F_{0,2}(C;\mu_t)\right|_{C=I}.
\end{aligned}
\tag{G.1}
$$

对候选相关求导时，教师分布保持固定。其余两个候选相关为零。

**命题G.1。** 存在常数 $\Gamma>0$ 使

$$
\lim_{t\to+\infty}e^{2t}g(t)=-e\Gamma<0.
\tag{G.2}
$$

因此存在有限 $T$，使所有实数 $t>T$ 均有 $g(t)<0$。此外，$g$ 是非恒零整函数在实轴上的限制，是偶函数，且 $g(0)=0$。故存在 $\epsilon_0>0$，使 $0<|t|<\epsilon_0$ 时 $g(t)\ne0$。这里未量化两个阈值，也未确定首个非零Taylor项的阶数或原点附近的符号。

## G.2 四种标签的贡献及极限

令 $\varepsilon=e^{-t}$、$R_{ki}=e^{-Z_{ki}}$（$i=1,2$）、$B_k=\sigma(Z_{k3})$，其中所有 $Z_{ki}$ 相互独立且服从标准Gaussian分布。记 $a=ER_{ki}=e^{1/2}$、$p_t=E\sigma(t+Z)$、$q_t=1-p_t$。支配收敛给出 $p_t\to1$ 和 $q_t/\varepsilon\to a$。对第三个标签，反射 $Z_{k3}$ 不改变 $B_k$ 的分布；两个标签概率之和为1，因此没有额外的因子2。

对 $s=(s_1,s_2)\in\{+,-\}^2$ 和 $\delta_i=1_{\{s_i=-\}}$，定义

$$
\begin{aligned}
\pi_k^s(\varepsilon)&=
\frac{B_kR_{k1}^{\delta_1}R_{k2}^{\delta_2}/[(1+\varepsilon R_{k1})(1+\varepsilon R_{k2})]}
{\sum_{\ell=1}^2B_\ell R_{\ell1}^{\delta_1}R_{\ell2}^{\delta_2}/[(1+\varepsilon R_{\ell1})(1+\varepsilon R_{\ell2})]},\\
d_{ki}^+&=\frac{\varepsilon R_{ki}}{1+\varepsilon R_{ki}},\qquad
 d_{ki}^-=-\frac1{1+\varepsilon R_{ki}}.
\end{aligned}
\tag{G.3}
$$

权重中的公共因子 $\varepsilon^{\delta_1+\delta_2}$ 已约去。对负对数粒子平均的混合导数应用Price恒等式[@price1958]，得到

$$
\begin{aligned}
G^s&=-\sum_{k=1}^2\pi_k^s(1-\pi_k^s)d_{k1}^{s_1}d_{k2}^{s_2},\\
g(t)&=p_t^2EG^{++}+p_tq_t(EG^{+-}+EG^{-+})+q_t^2EG^{--}.
\end{aligned}
\tag{G.4}
$$

对 $0<\varepsilon\le1$，有效的支配界为

$$
\begin{aligned}
\varepsilon^{-2}|G^{++}|&\le\sum_kR_{k1}R_{k2},&
\varepsilon^{-1}|G^{+-}|&\le\sum_kR_{k1},\\
\varepsilon^{-1}|G^{-+}|&\le\sum_kR_{k2},&
|G^{--}|&\le1.
\end{aligned}
\tag{G.5}
$$

这些lognormal包络可积。由于 $B_k>0$ 几乎必然成立，权重逐点收敛到与 $B_kR_{k1}^{\delta_1}R_{k2}^{\delta_2}$ 成比例的权重，不需要 $B$ 的逆矩。取独立标准Gaussian变量 $U_1,U_2$，并使其独立于 $B_1,B_2$，引入碰撞率

$$
T_k(v)=B_ke^{\sqrt vU_k},\qquad
c(v)=E\sum_{k=1}^2\left(\frac{T_k(v)}{T_1(v)+T_2(v)}\right)^2,
\qquad v\ge0.
\tag{G.6}
$$

乘以一个或两个独立的 $R$ 因子分别对应 $v=1$ 或 $v=2$。应用式(G.5)后，利用每个未进入权重的 $R$ 坐标的独立性，可得

$$
\begin{aligned}
\varepsilon^{-2}EG^{++}&\longrightarrow-a^2[1-c(0)],\\
\varepsilon^{-1}EG^{+-},\ \varepsilon^{-1}EG^{-+}
&\longrightarrow a[1-c(1)],\\
EG^{--}&\longrightarrow-[1-c(2)].
\end{aligned}
\tag{G.7}
$$

特别地，式(G.4)中的教师尾概率不可省略：

$$
\begin{aligned}
\lim_{t\to\infty}e^{2t}g(t)
&=-a^2[1-c(0)]+2a^2[1-c(1)]-a^2[1-c(2)]\\
&=e\{c(0)-2c(1)+c(2)\}.
\end{aligned}
\tag{G.8}
$$

## G.3 给出严格符号的Fourier表示

记 $L=\log B$、$\phi_L(\omega)=Ee^{i\omega L}$。二粒子的对数优势比满足

$$
\begin{aligned}
D(v)&=L_1-L_2+\sqrt v(U_1-U_2),\\
Ee^{i\omega D(v)}&=|\phi_L(\omega)|^2e^{-v\omega^2},\\
c(v)&=1-\tfrac12E\operatorname{sech}^2(D(v)/2).
\end{aligned}
\tag{G.9}
$$

非负模平方来自 $L_1,L_2$ 独立同分布，不要求 $L$ 本身对称。采用约定 $\widehat f(\omega)=\int_{\mathbb R}f(x)e^{-i\omega x}\,dx$，对 $f(x)=\operatorname{sech}^2(x/2)$ 作代换 $u=e^x$，得到

$$
\widehat f(\omega)
=4\int_0^\infty\frac{u^{-i\omega}}{(1+u)^2}\,du
=4\Gamma(1-i\omega)\Gamma(1+i\omega)
=\frac{4\pi\omega}{\sinh(\pi\omega)}.
\tag{G.10}
$$

这里Beta积分参数的实部为1，Gamma递推与反射公式适用[@nistdlmf, Sections 5.12 and 5.5]。零点的连续值为4。由于 $\widehat f$ 可积且 $|\phi_L|\le1$，Fourier反演与期望可以交换。因此

$$
\begin{aligned}
c(v)&=1-\int_{\mathbb R}\frac{\omega}{\sinh(\pi\omega)}
|\phi_L(\omega)|^2e^{-v\omega^2}\,d\omega,\\
c(0)-2c(1)+c(2)&=-\Gamma,\\
\Gamma&=\int_{\mathbb R}\frac{\omega}{\sinh(\pi\omega)}
|\phi_L(\omega)|^2(1-e^{-\omega^2})^2\,d\omega>0.
\end{aligned}
\tag{G.11}
$$

严格正性来自解析论证：$\omega/\sinh(\pi\omega)>0$（零点取值 $1/\pi$），而特征函数在原点连续且 $\phi_L(0)=1$，使其模平方在原点某邻域内为正。其余因子在非零点为正。式(G.8)与(G.11)证明了式(G.2)，不依赖数值积分或有限 $t$ 证书。

## G.4 带显式包络的整函数延拓

对实数 $x\in\mathbb R^6$ 和 $y\in\{-1,1\}^3$，固定有界的Price核：

$$
\begin{aligned}
\pi_k^y(x)&=\frac{\prod_{i=1}^3\sigma(y_ix_{ki})}
{\sum_{\ell=1}^2\prod_{i=1}^3\sigma(y_ix_{\ell i})},\\
H_y(x)&=-\sum_{k=1}^2\pi_k^y(x)(1-\pi_k^y(x))
y_1\sigma(-y_1x_{k1})y_2\sigma(-y_2x_{k2}),\\
|H_y(x)|&\le1.
\end{aligned}
\tag{G.12}
$$

将均值移入Gaussian密度。对 $z=a+ib$ 且 $|z|\le R$，基本不等式 $(x-a)^2\ge x^2/2-a^2$ 给出

$$
\left|e^{-(x-z)^2/2}\right|
=e^{-(x-a)^2/2+b^2/2}
\le e^{R^2/2}e^{-x^2/4}.
\tag{G.13}
$$

共有四个平移坐标和两个未平移坐标。因此复化密度及其核积分满足

$$
\begin{aligned}
\rho_z(x)&=(2\pi)^{-3}\exp\left[-\frac12\sum_{k=1}^2
\{(x_{k1}-z)^2+(x_{k2}-z)^2+x_{k3}^2\}\right],\\
|H_y(x)\rho_z(x)|&\le(2\pi)^{-3}e^{2R^2}e^{-\|x\|^2/4},\\
h_y(z)&=\int_{\mathbb R^6}H_y(x)\rho_z(x)\,dx.
\end{aligned}
\tag{G.14}
$$

该上界可积，并在每个复紧圆盘上一致成立。被积函数关于 $z$ 整，因此带支配的全纯参数积分使每个 $h_y$ 都是整函数。同一维度为1的论证表明 $p(z)=\int_{\mathbb R}\sigma(x)\varphi(x-z)\,dx$ 也是整函数，其中 $\varphi$ 是在其自变量上延拓的标准Gaussian密度。每个教师权重为 $w_y(z)=\tfrac12\prod_{i=1}^2p(z)^{1_{\{y_i=1\}}}[1-p(z)]^{1_{\{y_i=-1\}}}$。有限和 $\sum_yw_y(z)h_y(z)$ 因而是 $g$ 的整函数延拓。该构造只在实数 $x$ 处计算logistic函数，没有假定复log-sigmoid为整函数。

同时反射前两个坐标保持 $c_{12}$ 不变，并将 $t$ 变为 $-t$，故 $g(-t)=g(t)$。零均值处的单坐标反射给出 $g(0)=0$。式(G.2)排除了恒零函数。零点孤立性与偶性于是给出某个整数 $r\ge1$ 和实数 $A\ne0$，满足

$$
g(t)=At^{2r}+O(t^{2r+2}),\qquad
0<|t|<\epsilon_0\ \Longrightarrow\ g(t)\ne0.
\tag{G.15}
$$

这排除了已知匹配均值下的通用驻定身份，既没有证明二次系数 $b_2$ 非零，也不能移植到 $K=100$。特别地，它不解决附录F所用非对称均值点的六个梯度区间均含零的问题，不证明每个有限 $K$ 都非驻定，也不否定至多一个均值坐标非零时的精确对称性结论。
