# 附录 C. 有限选择、总体风险间隔与局部几何

## C.1. 固定的配对权重程序

正文图1展示这一比较：100粒子，无干扰与256干扰变量。(a)总体目标差$g_n=F_n(I)-F_n(C_b)$，正值表示相关候选优于独立真值，单位为每条完整观察的nats。(b)相关Hessian公共对角元$h_n$的符号决定真值局部类型。各图使用262,144个银行，分别属于预定99%同时家族，联合至少98%，保留数值资格。图下选择来自16,384外层银行、65,536标签的一次保存运行，不是新运行概率界。

本附录始终取 $m=3$、已知均值为零、$C_*=I$、$K=100$、$C\in\mathcal E_{.1}^{(3)}$，以及 $\mathcal R(C)=D(Q_I\Vert Q_C)$。简记 $F_n=F_{n,100}$。去掉与候选无关的无关块常数后，评分为

$$
\begin{aligned}
\phi_{n,y}(C;Z,a)&=-\log\sum_{k=1}^{100}a_{nk}\prod_{i=1}^3\sigma\bigl(y_i(C^{1/2}Z_k)_i\bigr),\\
\widehat F_n(C)&=B^{-1}\sum_{b=1}^B\sum_yq_y\phi_{n,y}(C;Z_b,a_b),\\
F_n(C)&=\mathbb E\,8^{-1}\sum_y\phi_{n,y}(C;Z,a).
\end{aligned}
\tag{C.1}
$$

65,536 个标签及频率 $q_y$ 复用附录 H 的输入。另生成 $B=16,384$ 个共享选择银行，每个银行含 100 个独立标准三维 Gaussian 向量。$n=0$ 时，$a_{0k}=1/100$；$n=256$ 时，$a_{256,k}=\exp(S_k)/\sum_l\exp(S_l)$，其中 $S_k$ 是 256 个独立标准 Gaussian 的 log-sigmoid 项之和，且与活跃输入独立。两分支之间只改变这一权重输入。

网格包含 $\{-.9,-.6,-.3,0,.3,.6,.9\}^3$ 中满足 $C\ge.1I$ 的 143 个三元组。选择规则返回与最小 binary64 评分相差不超过 $10^{-12}$ 的词典序首点，**没有回退规则**。评分和输出均在计算风险前锁定。锚点 $C_b$ 的三个非对角元均为 $.9$。两分支的数据、活跃粒子、网格、评分函数和选择规则完全相同；$K=100$ 在本轮运行前固定。这是共享银行的有限搜索，不等同于一般的逐观测重采样与优化流程。特别地，$K$ 表示每个对数内部的粒子数，不是学习的总预算。

|权重条件|返回点|数值观察风险|
|---|---|---:|
|$n=0$|$I$|0|
|$n=256$|$C_b$|.031103092779|

对保存的全部选择输入作 128-bit 球算术重算，得到 $n=256$ 的经验连续区域余量至少为 $.00390670332106984$：每个 $\mathcal R(C)\le10^{-6}$ 的参数评分都高于 $C_b$。该结论把保存的 binary64 输入视为精确数学输入，并计入平方根残差余量（附录 H）；它是确定性结论，不是重抽数据的失败概率。所报风险值来自相互核对的 Gaussian 求积，没有认证求积余项。旧 $K=2$ 中两条件均失败的结果仍保留在附录 H，不能作为预算之间的受控比较。

## C.2. 从联合风险外包到总体间隔

令 $\kappa(c)=\mathbb E[\tanh(X_1/2)\tanh(X_2/2)]$，其中两个标准 Gaussian 边际的相关为 $c$。中心对称性消去奇数阶 spin 矩。Walsh 反演与正交性精确给出

$$
\begin{gathered}
Q_C(y)=\frac{1+u_y}{8},\qquad u_y=\sum_{i<j}\kappa(c_{ij})y_iy_j,\\
\mathbb E_{Q_I}u=0,\qquad\mathbb E_{Q_I}u^2=\sum_{i<j}\kappa(c_{ij})^2,\qquad
\mathcal R(C)=-\mathbb E_{Q_I}\log(1+u).
\end{gathered}
\tag{C.2}
$$

对 $r_0=10^{-6}$ 及 $\mathcal R(C)\le r_0$，将 Pinsker 用于事件 $\{y,-y\}$，得到 $|u_y|\le a=\sqrt{8r_0}<1$。对正负两种 $u$，并由 Price 恒等式 [@price1958]，有

$$
\begin{aligned}
-\log(1+u)&=-u+u^2\int_0^1\frac{1-t}{(1+tu)^2}\,dt
\ge-u+\frac{u^2}{2(1+a)^2},\\
\kappa'(c)&=\tfrac14\mathbb E[\operatorname{sech}^2(X_1/2)\operatorname{sech}^2(X_2/2)]\\
&\ge\tfrac14\mathbb E e^{-(X_1^2+X_2^2)/4}\ge\beta:=e^{-1/2}/4.
\end{aligned}
\tag{C.3}
$$

末两个不等式使用 $\log\cosh v\le v^2/2$ 和 Jensen。由于 $\kappa(0)=0$，对 (C.3) 取平均并应用 (C.2)，得到

$$
\mathcal R(C)\le r_0\ \Longrightarrow\ 
\|C-I\|_F^2\le\frac{4(1+\sqrt{8r_0})^2r_0}{\beta^2}=:d^2,
\qquad d\simeq.013227076469306579.
\tag{C.4}
$$

为控制曲率，令 $S(t)=\sqrt{I+tD}$、$M_b=\max_k\|Z_{bk}\|_2$，以及 $\ell_k=\sum_i\log\sigma(y_i(S(t)Z_{bk})_i)$。用 $\pi$ 表示完整归一权重，则 $\phi''=-\sum_k\pi_k\ell_k''-\operatorname{Var}_{\pi}(\ell_k')$。logistic 导数界与附录 H 已证明的平方根导数界给出：当 $\|D\|_F\le d<1$ 时，

$$
\begin{aligned}
|\phi''|&\le\left\{\frac{(13/4)M_b^2}{4\lambda}+\frac{\sqrt3M_b}{4\lambda^{3/2}}\right\}\|D\|_F^2,\quad\lambda=1-d,\\
H(U;\lambda)&=\frac{(13/4)U}{4\lambda}+\frac{\sqrt{3U}}{4\lambda^{3/2}},\quad
b_K=\tfrac83(\log K+3\log2)\ge\mathbb E M_b^2,\\
\inf_{\mathcal R(C)\le r_0}\widehat F_n(C)&\ge\widehat F_n(I)-d\|\nabla\widehat F_n(I)\|_2/\sqrt2-\tfrac12H(U;1-d)d^2,\\
\inf_{\mathcal R(C)\le r_0}F_n(C)&\ge F_n(I)-c_d,\qquad c_d=\tfrac12H(b_K;1-d)d^2.
\end{aligned}
\tag{C.5}
$$

其中 $U\ge B^{-1}\sum_bM_b^2$。Gaussian 界来自 $\exp(tM_b^2)\le\sum_k\exp(t\|Z_{bk}\|^2)$，取 $t=3/8$。Jensen 控制 $M_b$ 的一阶矩。坐标符号对称性使总体在 $I$ 的梯度为零；经验梯度不必为零。Taylor 定理给出末两行。使用最大范数避免了粗糙的线性 $K$ 曲率代价。

独立审计使用 $L=262,144$ 个配对银行和真实标签权重 $1/8$；学习器不使用这些真实权重。预设对比为 $g_n=F_n(I)-F_n(C_b)$ 及 $\Delta=g_{256}-g_0$。令 $A=\|\sqrt{C_b}-I\|_{\rm op}$、$T=8$、$M=3AT$。若银行中任一活跃 Gaussian 坐标绝对值超过 $T$，便将该银行点差置零。单分支截断差落在 $[-M,M]$，配对差落在 $[-2M,2M]$。Cauchy--Schwarz、Gaussian 矩界及 Mills 不等式给出

$$
\begin{gathered}
t_T=\sqrt3A\sqrt{b_K\,2K(3)\varphi(T)/T},\qquad\alpha_j=.01/3,\\
\operatorname{hw}_j=\sqrt{\frac{2s_j^2\log(4/\alpha_j)}L}
+\frac{7R_j\log(4/\alpha_j)}{3(L-1)}+t_j+\epsilon_j,
\end{gathered}
\tag{C.6}
$$

其中 $s_j^2$ 是无偏样本方差，$R_j$ 是范围长度。按显示精度，单分支 $(R_j,t_j,\epsilon_j)=(32.821067231192,\; .000003077999,\;10^{-9})$；$\Delta$ 的三项均加倍，其方差直接用配对差计算。经验 Bernstein 不等式 [@maurer2009, Theorem 4] 与并合界给出一个有数值资格的 99% 家族事件。没有银行触发截断。每分支另对 32 个固定银行作球算术核验；这些诊断不证明每个总体节点都满足数值余量。覆盖解释依赖规定的 iid Gaussian 抽样模型及数值误差资格。 附录E的全节点评分外包针对附录D独立的新运行矩审计，并非这里较早的锚点差银行。

|对比量|均值|99% 家族下界|99% 家族上界|
|---|---:|---:|---:|
|$g_0$|-.030106637632|-.032232312506|-.027980962758|
|$g_{256}$|.005905979917|.002342327475|.009469632359|
|$\Delta$|.036012617550|.030377739156|.041647495943|

端点向外舍入。原预设外包使用 $d_{\rm old}=\sqrt{12r_0}/\beta$，总体代价为 $.004361765095074056$。其区域下界为 $-.002019437619389396$：该门失败，**原七门中六门通过**。这一记录保持不变。

式 (C.4) 是**事后确定性加强**，复用同一组已经同时覆盖的对比。它给出 $c_d\simeq.0014471050001073832$；在原覆盖事件上，

$$
\begin{aligned}
\inf_{\mathcal R(C)\le r_0}\{F_{256}(C)-F_{256}(C_b)\}&\ge\eta,
\qquad\eta=.0008952224755772784>0,\\
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<\eta
&\ \Longrightarrow\ \mathcal R(C)>10^{-6}.
\end{aligned}
\tag{C.7}
$$

向外下界由原 $g_{256}$ 下端点减去新的确定性代价得到。没有新增随机对比或误差预算。该蕴含成立，是因为总体下确界不大于 $F_{256}(C_b)$。目标在紧参数类上连续，并有可积包络 $3\log2+3M_b$，所以最小点存在。每个最小点的风险都大于 $10^{-6}$；不能据此推出每个最小点的风险都是 $.0311$，也不能推出 $C_b$ 全局最优。这里未认证高维渐近余项或一般优化器的 regret。

附录D的独立矩审计在自己的联合事件上控制同一个总体点差。附录D.5复用上述确定性半径和代价，在支持新运行风险界的同一事件上给出大于$.00134$的区域余量。该后续同事件推论不改动原审计表、失败门和较早余量。

## C.3. 正曲率控制的解析证明

坐标变号与置换保持两个总体目标不变。在自由坐标 $(c_{12},c_{13},c_{23})$ 中，它们精确推出

$$
\nabla F_{n,K}(I)=0,\qquad\nabla^2F_{n,K}(I)=h_{n,K}I_3.
\tag{C.8}
$$

下面证明 $h_{0,100}>.00410$。令 $X_i$ 为独立标准 Gaussian、$p_i=\sigma(X_i)$、$W=p_1p_2p_3$，并定义

$$
\begin{aligned}
A(s)&=\mathbb E e^{-sW},\quad B(s)=\mathbb E[X_1X_2e^{-sW}],\\
D(s)&=\mathbb E[(X_1^2-1)(X_2^2-1)e^{-sW}],\\
h_{0,K}&=K\int_0^\infty A(s)^{K-2}\{A(s)D(s)+(K-1)B(s)^2\}\,\frac{ds}{s}.
\end{aligned}
\tag{C.9}
$$

该恒等式来自对 $-\log\sum_kW_k$ 的 Laplace 表达式求两次导数；单粒子的 Gaussian 密度 score 分别为 $X_1X_2$ 和 $(X_1^2-1)(X_2^2-1)$。反射 Gaussian 坐标可以吸收任意标签，并保持这一对角二阶导数不变。为说明交换合法，令 $S=\sum_kW_k$，以 $J$ 表示联合二阶密度 score。此时 $\mathbb EJ=\mathbb E(JS)=0$，所以零点附近 $|\mathbb EJe^{-sS}|\le s^2\mathbb E(|J|S^2)/2$。对 $T>0$，尾部绝对积分不超过 $\mathbb E[|J|\{E_1(T)-\log W_1\}]<\infty$，其中 $E_1(v)=\int_v^\infty e^{-u}du/u$。这里使用 $S\ge W_1$ 及 $-\log\sigma(x)\le\log2+|x|$。Gaussian 密度导数存在局部一致的多项式乘 Gaussian 包络，同时保证二阶连续可微。

令 $m_r=\mathbb E\sigma(X)^r$、$d_r=\mathbb E[(X^2-1)\sigma(X)^r]$。对称性给出 $d_0=d_1=0$。对 $r\ge2$，偶函数 $f_r=(p^r+(1-p)^r)/2$ 随 $|X|$ 严格增加，因此 $d_r>0$。在 $x\ge0$ 上，

$$
\begin{aligned}
\left[\frac{r+1}{r}f_r-f_{r+1}\right]'&=\frac{r+1}{2}p'p(1-p)\{p^{r-2}-(1-p)^{r-2}\}\ge0,\\
0<d_{r+1}&\le\frac{r+1}{r}d_r,\qquad
D(s)=\sum_{r=2}^\infty\frac{(-s)^r}{r!}d_r^2m_r\ge0\quad(0\le s\le4/3).
\end{aligned}
\tag{C.10}
$$

同向单调函数的协方差给出 $d_r$ 的比较不等式。这个绝对收敛的交错级数，相邻项绝对值之比不超过 $s(r+1)/r^2\le3s/4$，由此得到其符号；未假设 $D$ 在整个正轴上非负。

由 $\operatorname{sech}^2(X/2)\ge e^{-X^2/4}$，记 $a_0=1/\sqrt{24}\le a=\mathbb E\sigma'(X)$。Jensen 给出 $A(s)\ge e^{-s/8}$，分部积分给出

$$
\begin{gathered}
B(s)=s\mathbb E[\sigma'(X_1)\sigma'(X_2)p_3(sW-1)e^{-sW}],\\
|B(s)|\ge\tfrac12a^2s(1-s)e^{-s}\quad(0\le s\le1).
\end{gathered}
\tag{C.11}
$$

只在 $[0,1/8]$ 上积分这一正贡献。对剩余负贡献，取 $T=4/3$，并使用 $m_2=1/2-a$、$e^{-u}\le1-u+u^2/2$ 以及 $\|(X_1^2-1)(X_2^2-1)\|_2=2$，得到

$$
\begin{aligned}
P&=\frac{100\cdot99}{2304}(7/8)^2e^{-57/32}/128,\\
A(T)&\le q:=1-T/8+(T^2/2)(1/2-a_0)^3,\\
\int_T^\infty|D(s)|\frac{ds}{s}&\le2\{e^{-T}/T+3(\log2+1)\},\\
N&=200q^{99}\{3(\log2+1)+e^{-T}/T\},\qquad h_{0,100}\ge P-N.
\end{aligned}
\tag{C.12}
$$

尾界来自 $E_1(TW)\le E_1(T)-\log W$、Cauchy--Schwarz 及 Minkowski。式 (C.10) 控制 $D$ 积分的其余部分。

严格数值门只使用有理数运算。取 $a_L=204124/10^6$、$r=57/32$，并令

$$
\begin{aligned}
E_U&=\sum_{j=0}^{12}\frac{r^j}{j!}+\frac{r^{13}/13!}{1-r/14}>e^r,\qquad
q_U=1-\tfrac16+\tfrac89(\tfrac12-a_L)^3,\\
P_L&=\frac{100\cdot99}{2304}(7/8)^2/(128E_U),\quad
N_U=200q_U^{99}\{3(694/1000+1)+1/5\},\\
24a_L^2&<1,\quad\sum_{j=0}^{6}\frac{(694/1000)^j}{j!}>2,\quad
\sum_{j=0}^{5}\frac{(4/3)^j}{j!}>15/4,\\
h_{0,100}&\ge P_L-N_U>41/10000=.00410.
\end{aligned}
\tag{C.13}
$$

指数尾项由几何级数控制。两个有限和分别证明 $\log2<.694$ 以及 $e^{-4/3}/(4/3)<1/5$；最后的严格比较是精确有理数不等式（$P_L-N_U\simeq.004101508638$）。因此 $I$ 是 $n=0$ 的严格局部极小点，不带 Monte Carlo 资格。这里没有证明它全局最优。

## C.4. 有界曲率审计及其范围

对 $n=256$，使用新的统计量和新银行。在 $I$ 固定标签 $(1,1,1)$，用 $\pi_k$ 表示 $a_k\prod_i\sigma(Z_{ki})$ 的归一化权重。定义 $q_{ki}=\sigma(-Z_{ki})$、$r_{ki}=q_{ki}(2q_{ki}-1)$、$a_k^{ij}=q_{ki}q_{kj}$、$A_{ij}=\sum_k\pi_ka_k^{ij}$ 和 $B_{ij}=\sum_k\pi_k^2a_k^{ij}$。对 Price 算子 $\sum_k\partial_{Z_{ki}}\partial_{Z_{kj}}$ 平方，得到

$$
\begin{aligned}
H_{ij}={}&\sum_k\left[-\pi_k(1-\pi_k)r_{ki}r_{kj}
+2\pi_k^2(1-\pi_k)(q_{ki}^2r_{kj}+q_{kj}^2r_{ki})\right.\\
&\left.\hspace{8mm}+\pi_k^2(1-4\pi_k)q_{ki}^2q_{kj}^2\right]
+A_{ij}^2-4A_{ij}B_{ij}+6B_{ij}^2,\\
h_{n,K}&=\mathbb EH_{ij},\qquad -33/8\le H_{ij}\le67/16.
\end{aligned}
\tag{C.14}
$$

显示的四项分别落在 $[-1,1/8]$、$[-1/8,1]$、$[-3,1/16]$ 和 $[0,3]$；这里使用 $-1/8\le r\le1$、$\sum\pi^2(1-\pi)\le1/4$ 以及 $0\le B_{ij}\le A_{ij}\le1$。logistic 导数和归一权重有界；条件于无关权重后可合法使用 Price 求导。$K=1$ 时该多项式精确相消。该估计器与平方根路径 Hessian 的期望相同，逐银行取值不必相同。

另生成 262,144 个独立配对银行。三个 pair 核在**银行内**平均，不能作为三个独立观测。对 $h_{0,100}$、$h_{256,100}$ 及其差应用 (C.6)，尾项取零，范围长度分别为 $133/16$、$133/16$、$133/8$，数值余量为 $10^{-9}$、$10^{-9}$、$2\times10^{-9}$。这是另一个 $\alpha=.01$ 家族。

|曲率对比量|均值|99% 家族下界|99% 家族上界|
|---|---:|---:|---:|
|$h_{0,100}$|.028200713775|.027644213668|.028757213882|
|$h_{256,100}$|-.004265410162|-.004912712428|-.003618107895|
|$h_{256,100}-h_{0,100}$|-.032466123936|-.033641746822|-.031290501050|

端点向外舍入。人工构造的小 $K$ 自动微分检查核验该多项式；32 个固定科学银行给出 192 项球算术核验，最大差为 $8.35\times10^{-15}$。后续确定性计算对全部262,144个银行的保存输入n256核作外包，覆盖三个方向，并将误差传播到样本均值与标准差（附录E）。原表与原家族保持不变。原诊断和这项保存输入计算都没有建立统一的Gaussian/log-sigmoid输入生成误差界，因此总体符号仍保留该资格。

在这一事件上，(C.8) 说明 $I$ 是 $n=256$ 的严格局部极大点。每个充分小的非零相关扰动都降低其有限总体目标，同时增加精确观察风险；$\kappa'(c)\ge\beta$ 保证只有 $I$ 的风险为零。$I$ 的梯度精确为零，所以从该点初始化的精确确定性优化器不必移动。本曲率审计没有运行优化器。

在相应抽样与数值条件下，本附录两个随机家族的保守联合覆盖至少为98%。一并纳入附录F的均值稳健性家族与附录D的新运行矩家族后至少为96%；再加附录H的较早家族则至少为95%。该覆盖只针对这些指定家族，不是全文全部历史实验。这不要求家族之间独立。确定性加强、解析正曲率证明及附录E的保存输入算术不花费额外概率预算。

## C.5. 已保留有限实例的首项量级校准

固定同一个$m=3$独立教师、$n=256$、$K=100$及非对角元均为$.9$的$C_b$，对(A.4)、(A.6)和(A.16)的系数进行确定性求积，得到

$$
\begin{gathered}
v(1)\simeq.271514501801,\quad J_{100}\simeq2.507593636442,\\
w_{256,100}\simeq.300774255953,\quad V(C_b)\simeq-.052200051030.
\end{gathered}
$$

这里$V(I)=0$。目标差首项为$w_{256,100}[V(I)-V(C_b)]$。另对极限梯度场形式求导，得到$h^{\rm formal}_{256,100}=-w_{256,100}\kappa'(0)/4$，其中$\kappa'(0)=\tfrac14(E\operatorname{sech}^2(G/2))^2\simeq.170768891292$。

|比较对象|首项或形式值|已保留有限审计均值|原有数值资格的99%家族区间|
|---|---:|---:|---|
|$g_{256}$，目标差|.015700431509|.005905979917|[.002342327475, .009469632359]|
|$h_{256,100}$，形式二阶比较|−.012840721555|−.004265410162|[−.004912712428, −.003618107895]|

目标差单位为每个完整观测的nats，曲率单位为每个自由相关坐标平方的nats。有限目标差均值是首项的37.62%，或首项是该均值的2.658倍；形式曲率幅度是有限均值幅度的3.010倍。两者均在对应原区间之外。两个区间属于不同家族，原条件下联合覆盖至少98%。有限目标差与曲率条目分别沿用C.2和C.4的审计。这些是与估计量的比较，不是渐近误差证书；定理未给出本$n$下的小相对余项，有限后果另有独立的数值资格。归一化梯度的一致收敛不能直接再求导得出Hessian收敛，因此曲率行只称**形式二阶比较**，不称定理1已经证明的曲率预测。首项目标差比较的是$g_{256}$，不是无关权重交互量$g_{256}-g_0$。

确定性Hermite40/64/96配合相关方向Legendre32/48/64、另行参数化的正态加权Legendre128及全实线一维求积，对上述系数的差异均在$1.2\times10^{-13}$内。这些规则敏感性比较不是严格求积余项界；该比较不增加Monte Carlo数据或训练。

## C.6. 四个固定规模：重叠量与带标记学习场

本比较保持C.1的独立三变量教师、$K=100$、$I/C_b$及自由坐标约定。在$n=64,256,1024,4096$上使用131,072个iid完整银行，活跃输入与干扰前缀共享。各规模使用同一批银行，不是独立训练重复。令$C(\rho)$全部非对角元为$\rho$，$w_n$为(7)的参考，冻结的$m_H^{\mathrm{ref}}\simeq.0192712788530$近似$\kappa(.45)/4$。定义

$$
\begin{gathered}
g_n=F_n(I)-F_n(C_b),\quad D_n=\partial_{c_{12}}F_n(C(.45)),\quad
\omega_n=E\sum_k\gamma_{nk},\\
\gamma_{nk}=\pi_{nk}(1-\pi_{nk}),\quad
A_n=\frac{g_n}{w_n\{V(I)-V(C_b)\}},\quad B_n=-\frac{D_n}{m_H^{\mathrm{ref}}w_n}.
\end{gathered}
\tag{C.15}
$$

**主目标与有界核。** 每个完整银行独立抽取一次均匀路径坐标，在各$n$之间共享。Price求导、正谱下界紧路径上的Gaussian包络及Fubini给出

$$
\begin{aligned}
g_n&=-\int_0^{.9}\sum_{i<j}\partial_{c_{ij}}F_n(C(\rho))\,d\rho=EX_n,\\
X_n&=.9\,\frac18\sum_y\sum_k\gamma_{nk}(C(U))\sum_{i<j}H_{k,ij}(C(U)),
\qquad U\sim\operatorname{Uniform}(0,.9),\\
X_n&\in[-.675s,1.35s],\quad D_{n,\mathrm{bank}}\in[-s/2,s/2],\quad
\omega_{n,\mathrm{bank}}\in[0,s],\qquad s=1-1/K=.99.
\end{aligned}
\tag{C.16}
$$

八个标签以$1/8$枚举。自由$c_{12}$同时改变两个对称协方差项，因此Price导数没有额外$1/2$；路径导数为三个自由坐标之和。路径支持中，两个全同号标签的三标记和在$[0,3]$内，六个混合标签在$[-1,1]$内，给出(C.16)较窄的平均支持。这是数学采样模型下目标差的无偏估计，不是某次实现的平方根路径导数随机变量。

**固定坐标分解。** 在$C(.45)$处，各银行使用同一个冻结参考$m_H^{\mathrm{ref}}$，

$$
\begin{aligned}
\mathcal C_n&=E\sum_k\gamma_{nk}(H_{k,12}-m_H^{\mathrm{ref}}),\qquad D_n=-m_H^{\mathrm{ref}}\omega_n-\mathcal C_n,\\
\frac{D_n}{w_n}+m_H^{\mathrm{ref}}&=\underbrace{m_H^{\mathrm{ref}}\left(1-\frac{\omega_n}{w_n}\right)}_{O_n}
+\underbrace{\left(-\frac{\mathcal C_n}{w_n}\right)}_{J_n},\\
\mathcal C_{n,\mathrm{bank}}&\in[-(1/2+m_H^{\mathrm{ref}})s,(1-m_H^{\mathrm{ref}})s/2].
\end{aligned}
\tag{C.17}
$$

银行形式都平均八个标签。该身份对任意固定$m_H^{\mathrm{ref}}$成立；把它解释为渐近参考时仍继承数值资格。下表残差使用同一绝对归一化尺度。完整权重同时含活跃与干扰评分，因此$J_n$是标记耦合残差，不是对所有非Gaussian偏离深层来源的识别。

|$n$|$A_n$：均值[区间]|$B_n$：均值[区间]|$\omega_n/w_n$：均值[区间]|
|---:|---:|---:|---:|
|64|-0.14273 [-0.16133, -0.12413]|-0.14859 [-0.17300, -0.12419]|0.99160 [0.98741, 0.99580]|
|256|0.33789 [0.30066, 0.37511]|0.32796 [0.27900, 0.37692]|1.02633 [1.01697, 1.03569]|
|1024|0.65135 [0.58246, 0.72023]|0.65173 [0.56171, 0.74175]|1.01815 [1.00267, 1.03363]|
|4096|0.82920 [0.70351, 0.95489]|0.83086 [0.66679, 0.99493]|1.00822 [0.98451, 1.03193]|

|$n$|$O_n$：均值[区间]|$J_n$：均值[区间]|
|---:|---:|---:|
|64|0.00016179 [0.00008101, 0.00024257]|0.02197309 [0.02146427, 0.02248191]|
|256|-0.00050746 [-0.00068772, -0.00032720]|0.01345856 [0.01246015, 0.01445698]|
|1024|-0.00034978 [-0.00064800, -0.00005156]|0.00706136 [0.00528073, 0.00884199]|
|4096|-0.00015841 [-0.00061528, 0.00029846]|0.00341803 [0.00022369, 0.00661238]|

在$n=64$，条件区间支持$g_n<0$及$D_n>0$，都反于对应首项，尽管标量重叠已接近其尺度。耦合项$J_{64}\simeq.02197309$还大于$m_H^{\mathrm{ref}}$，而$O_{64}\simeq.00016179$。64、256、1024处的联合区间确认$J_n>|O_n|$，4096处则未确认这一排序。结论针对该坐标与分解，不覆盖整条$I\to C_b$路径，也不是训练风险的因果分解。

**全部目标与同时区间。** 家族包括四规模各五个原始目标及六个相邻配对差，共26目标。原始目标为路径目标差、固定坐标梯度、重叠、耦合和次级直接锚点损失差。用无偏样本方差$\widehat v_j$和预定支持长度$R_j$，双侧经验Bernstein半径为

$$
\begin{gathered}
\delta_j=.01/26,\qquad B=131072,\\
\operatorname{hw}_j=\sqrt{\frac{2\widehat v_j\log(4/\delta_j)}B}
+\frac{7R_j\log(4/\delta_j)}{3(B-1)}+t_j+\epsilon_j.
\end{gathered}
\tag{C.18}
$$

每个原始目标取$\epsilon_j=10^{-9}$，仿射变换及差按系数缩放该容差。前四个支持见(C.16)--(C.17)，尾项为零。直接差沿用C.2：若任一活跃$|Z|>8$，该银行值置零；支持为$[-M,M]$，$M=16.410533615595888$，尾容差为$3.077999197742064\times10^{-6}$。不截断干扰对数权重。直接差控制不替换主路径估计器。

|$n$|目标|均值|样本方差|联合下端|联合上端|
|---:|---|---:|---:|---:|---:|
|64|$g_n$|-0.00448185566|0.0004559644|-0.00506565|-0.00389807|
|64|$D_n$|0.00172259379|0.0001016962|0.00143977|0.00200542|
|64|$\omega_n$|0.596498369|0.03940728|0.59397700|0.59901974|
|64|$\mathcal C_n$|-0.0132178802|0.0001418427|-0.01352396|-0.01291180|
|64|$g_n^{\mathrm{dir}}$|-0.00452243934|0.01260892|-0.01126389|0.00221901|
|256|$g_n$|0.00530496451|0.0004581111|0.00472058|0.00588935|
|256|$D_n$|-0.00190094564|0.0001032908|-0.00218471|-0.00161718|
|256|$\omega_n$|0.308694352|0.04976738|0.30588104|0.31150766|
|256|$\mathcal C_n$|-0.0040479893|0.0001304913|-0.00434829|-0.00374769|
|256|$g_n^{\mathrm{dir}}$|0.0055723925|0.04136146|-0.00225117|0.01339595|
|1024|$g_n$|0.00511321392|0.0003143611|0.00457247|0.00565396|
|1024|$D_n$|-0.00188881694|6.786048e-05|-0.00214970|-0.00162793|
|1024|$\omega_n$|0.15311671|0.03318471|0.15078953|0.15544389|
|1024|$\mathcal C_n$|-0.00106193787|7.545536e-05|-0.00132973|-0.00079415|
|1024|$g_n^{\mathrm{dir}}$|0.00613213357|0.06823211|-0.00237854|0.01464281|
|4096|$g_n$|0.00325468235|0.0001887566|0.00276135|0.00374801|
|4096|$D_n$|-0.00120397332|3.956504e-05|-0.00144172|-0.00096623|
|4096|$\omega_n$|0.0758116501|0.01858578|0.07402902|0.07759428|
|4096|$\mathcal C_n$|-.000257014127|4.050285e-05|-0.00049721|-0.00001682|
|4096|$g_n^{\mathrm{dir}}$|0.00365793433|0.08548928|-0.00522308|0.01253895|

均值和方差为显示近似，区间端点向外舍入。相邻差先逐银行构造，再估计方差。若$\alpha_n$为对应仿射尺度，方差为$\alpha_b^2\widehat v_b+\alpha_a^2\widehat v_a-2\alpha_a\alpha_b\widehat{\operatorname{Cov}}(X_a,X_b)$，支持为两缩放支持的差，数值容差相加。不把协方差替换为零，也不预设配对一定降低方差。

|量|后一个减前一个$n$|均值|联合区间|原始配对协方差|配对方差|
|---|---|---:|---|---:|---:|
|$A$|256-64|0.4806169|[0.43092, 0.53031]|-3.293773e-06|2.33423|
|$A$|1024-256|0.3134604|[0.21962, 0.40730]|1.566845e-05|6.705305|
|$A$|4096-1024|0.1778487|[0.00305, 0.35265]|9.030132e-06|16.76677|
|$B$|256-64|0.4765523|[0.41116, 0.54194]|5.843457e-07|3.813732|
|$B$|1024-256|0.3237732|[0.20000, 0.44755]|1.30637e-06|10.99817|
|$B$|4096-1024|0.1791244|[-0.05090, 0.40915]|6.744675e-07|26.60021|

完整20目标与26目标协方差矩阵保留在机器可读伴随结果中，表中给出六项比较所需全部配对协方差。$A_n$三次相邻增加均被分辨，最后下端仅约.00305；$B_n$最后一次增加仍未分辨。4096处的两项比值区间未包含于$[.9,1.1]$，故未确认10%相对精度。更宽的直接差区间及所有未决比较保留，没有根据结果挑选。

这些是理想iid完整银行采样、声明数值容差和冻结参考系数条件下的99%联合区间；不是统一原生浮点或精确系数证书，也不是渐近余项定理。它在C.4此前五个指定家族之外另占一个.01家族；不要求家族相互独立，union bound给合计至少94%的条件联合覆盖。I.8的确定性局部证书不增加alpha。

此前C.5目标差估计为首项的37.62%，本面板在256处为33.79%；二者来自不同银行和估计器，新区间位于旧目标差区间内。正文图2统一使用新共享面板，不把一个批次结果回写为另一个。本比较增加有限带标记场的解释，不给普适收敛速率、Hessian预测或粒子预算规则。

图中原有标记$m_H$表示本节固定数值参考$m_H^{\mathrm{ref}}$，区别于C.7起解析展开中的精确期望。

## C.7. 有限尺度修正的精确标签条件核 {#fixed-k-correction}

C.7--C.14证明(11)--(12)中的固定$K$修正及有限规模推论。对象仍为同一零均值模型；这里使用确定性模型积分，C.6的条件Monte Carlo区间保持原资格。先固定$K\ge2$、有限$m$、正定相关矩阵$C$及不随$C$变化的标签律$p_y=P(Y=y)$。令$Z\sim N(0,1)$、$X\sim N(0,C)$，记

$$
\begin{aligned}
L&=\log\sigma(Z),\quad \ell=EL,\quad v=\operatorname{Var}L,\\
\kappa_3&=E(L-\ell)^3,\quad s=\sqrt{nv},\\
A_y&=\sum_{i=1}^m\log\sigma(y_iX_i),\quad T_y=A_y-m\ell,\\
H_y&=y_1y_2\sigma(-y_1X_1)\sigma(-y_2X_2),\\
m_y&=E_XH_y,\quad c_y=E_X[(H_y-m_y)T_y],\\
m_H&=E_Ym_Y,\qquad \bar c=E_Yc_Y.
\end{aligned}
\tag{C.19}
$$

同一银行的所有粒子共享标签，标签与候选粒子独立。保留的独立教师实例中，标签为八标签均匀分布。令$\xi$为独立的中心化标准Gumbel，$S_n=\sum_{j=1}^n(L_j-\ell)$，$g,G$为$(S_n+\xi)/s$的密度和CDF。以下所有核均保留此Gumbel。对每个固定标签定义

$$
\begin{aligned}
f_y(z)&=E_Xg(z-T_y/s),\quad F_y(z)=E_XG(z-T_y/s),\\
h_y(z)&=E_X[H_yg(z-T_y/s)],\quad q=K-2,\\
D_n&=-\frac{K(K-1)}s\sum_yp_y\int_{\mathbb R}h_yf_yF_y^q.
\end{aligned}
\tag{C.20}
$$

末式由A.2的有序配对Price公式和Gumbel最大值表示得到。每个标签积分内共同平移$m\ell/s$，恰好将活跃得分中心化。先使用给定标签后的粒子独立性，再作最终标签平均；若先混合标签密度再相乘，就改变了目标。特别地，$\bar c$是同一粒子内、条件于标签的标记—得分协方差，不是粒子间协方差或训练噪声统计量。

以$\phi,\Phi$表示标准Gaussian密度与CDF。令$M_K=\max_{k\le K}Z_k$，其中$Z_k$为iid标准Gaussian；分部积分给出

$$
\begin{aligned}
I_K&=\int\phi^2\Phi^q,\quad T_K^{\rm G}=\int z\phi^2\Phi^q,\\
J_K&=EM_K=K(K-1)I_K,\\
B_K&=EM_K^2-1=K(K-1)T_K^{\rm G},\\
B_K&=\frac{K(K-1)(K-2)}2\int\phi^3\Phi^{K-3}\quad(K\ge3),\\
B_2&=0.
\end{aligned}
\tag{C.21}
$$

这里$T_y$是活跃中心得分，而$T_K^{\rm G}$与$B_K$是Gaussian次序统计常数。上述恒等式依次使用$\phi'=-z\phi$、$z^2\phi=\phi-(z\phi)'$和$(\phi^2)'=-2z\phi^2$，所有边界项均消失。

## C.8. 有符号次阶项的证明与解释

对上述每个固定$K,m,C$，存在有限常数$C_0,n_0$，使得$n\ge n_0$时下式余项的绝对值不超过$C_0n^{-3/2}$：

$$
\begin{aligned}
D_n&=-\frac{m_HJ_K}{s}
+\frac{B_K}{s^2}\left(\frac{m_H\kappa_3}{3v}-\bar c\right)
+O_{K,m,C}(n^{-3/2}),\\
D_n^{(2)}&=-\frac{m_HJ_K}{s}+\frac{b_2}{n},\qquad
b_2=\frac{B_K}{v}\left(\frac{m_H\kappa_3}{3v}-\bar c\right).
\end{aligned}
\tag{C.22}
$$

这是一项固定$K$结论，不宣称对增长的$K$或整个参数类一致。上述存在性常数本身不能认证四个有限规模中的任一个；后续小节另行给出显式有限界。

为使非线性核的控制自足，以下给出所需密度范数论证。记$\epsilon=n^{-1/2}$、$\alpha=\kappa_3/(6v^{3/2})$、$e=(z^3-3z)\phi$及$\mathcal P=(1-z^2)\phi$，于是$\mathcal P'=e$。对$p=1,\infty$及充分大$n$，

$$
\begin{aligned}
\|g-\phi-\epsilon\alpha e\|_p&=O(n^{-1}),\\
\|g'-\phi'\|_p&=O(n^{-1/2}),\qquad \|g''\|_p=O(1).
\end{aligned}
\tag{C.23}
$$

证明如下。$L$在$x<0$的密度为$p_L(x)=\phi(x-\log(1-e^x))/(1-e^x)$，在$x\ge0$延拓为零。在$-\infty$，任意固定阶导数由Gaussian衰减乘多项式控制；在$0-$令$t=-\log(-x)$，每阶导数由$\exp(-t^2/2+O(t))$乘多项式控制，因而趋零。零延拓因此是Schwartz函数。于是$U=(L-\ell)/\sqrt v$具有所有矩，特征函数$\psi$及其导数快速衰减，且每个$\delta>0$均有$\sup_{|u|\ge\delta}|\psi(u)|<1$；非退化连续密度在非零频率不能满足$|\psi(u)|=1$。令$p_n$为$S_n/s$的密度，局部对数展开及其前两阶导数给出

$$
\begin{aligned}
n\log\psi(t/\sqrt n)&=-t^2/2+\epsilon\alpha(it)^3+O(t^4/n),\\
R_n(t)&=\psi(t/\sqrt n)^n-e^{-t^2/2}(1+\epsilon\alpha(it)^3),\\
\int_{\mathbb R}\left|\partial_t^j\{(it)^rR_n(t)\}\right|dt
&=O(n^{-1}),\qquad 0\le r,j\le2.
\end{aligned}
\tag{C.24}
$$

末行的尾部控制如下。在$|t|\le n^{1/12}$上，Taylor余项及前两阶导数由$n^{-1}$乘固定多项式和$e^{-ct^2}$控制，其中包括三次项的平方。在$n^{1/12}<|t|\le\delta\sqrt n$上，取固定小$\delta$使$|\psi(u)|\le e^{-cu^2}$且$\psi(u)\ne0$；求导乘积至多增加$n,t$的多项式因子，仍被Gaussian尾压制。在$|t|>\delta\sqrt n$上，保留至多两个受导因子及两个未受导因子，其余至少$n-4$个因子贡献某个$r_\delta<1$的$r_\delta^{n-4}$；保留的Schwartz因子换元后具有有限加权积分，所有$n$的幂均被该指数压制。三段估计共同证明(C.24)，没有留下未控制的Fourier尾。

Fourier反演及频率上的两次分部积分于是给出$\sup_z(1+z^2)|p_n^{(r)}-\phi^{(r)}-\epsilon\alpha e^{(r)}|=O(n^{-1})$，同时得到$L^1$和$L^\infty$界。由$E\xi=0$、$E\xi^2=\pi^2/6$，与$\xi/s$卷积给出$\|g-p_n\|_p\le E\xi^2\|p_n''\|_p/(2s^2)$、$\|g'-p_n'\|_p\le E|\xi|\|p_n''\|_p/s$及$\|g''\|_p\le\|p_n''\|_p$，从而证明(C.23)。因此Gumbel不贡献(C.22)的$n^{-1}$系数，但仍进入余项。

Gaussian活跃矩和$|H_y|\le1$保证积分Taylor余项合法。由中心化$ET_y=0$，在所列两种密度范数下均有

$$
\begin{aligned}
\|f_y-g\|_{1,\infty}+\|F_y-G\|_\infty&=O(s^{-2}),\\
h_y-m_yf_y&=-c_yg'/s+O_{1,\infty}(s^{-2}).
\end{aligned}
\tag{C.25}
$$

令$\mathcal I(f)=\int f^2F^q$；固定$q$时，有界密度与$L^1+L^\infty$误差控制所有乘积。它在$\phi$处沿$e$方向的线性变分为

$$
\begin{aligned}
D\mathcal I_\phi[e]&=2\int\phi e\Phi^q+q\int\phi^2\mathcal P\Phi^{q-1}
=-2T_K^{\rm G},\\
\mathcal I(f_y)&=I_K-\frac{\kappa_3}{3vs}T_K^{\rm G}+O_K(n^{-1}),\\
\int(h_y-m_yf_y)f_yF_y^q&=\frac{c_y}{s}T_K^{\rm G}+O_K(n^{-1}).
\end{aligned}
\tag{C.26}
$$

首个恒等式通过积分$[(z^2-1)\phi^2\Phi^q]'$得到；末行的正号来自$\phi'=-z\phi$。每个一阶密度或CDF扰动均为$O(n^{-1/2})$，故乘积余项为$O_K(n^{-1})$。当$q=0$时直接省略乘$q$的项，不使用CDF负幂；当$q=1$时，CDF因子为线性，其二阶及更高变分消失。两种情形都由同一个有限多项式展开得到，不使用后文仅对$q=98$引入的$q-1$除法。代回(C.20)并使用(C.21)，即得(C.22)。

最后，$E_XA_y=m\ell$与自由相关坐标无关。由Price微分及$A_y$的坐标可加性，

$$
\begin{aligned}
\partial_{c_{12}}E_XT_y^2&=2m_y,\qquad
\partial_{c_{12}}E_XT_y^3=6c_y,\\
V_A&=E_YE_XT_Y^2,\quad T_A=E_YE_XT_Y^3,\\
m_H&=\tfrac12\partial_{c_{12}}V_A,\qquad
\bar c=\tfrac16\partial_{c_{12}}T_A.
\end{aligned}
\tag{C.27}
$$

自由非对角Price算子的系数为一，因为两个对称协方差元素同时变化。次阶项因此组合了nuisance偏斜与标签条件活跃第三矩的变化。$K=2$时因$B_2=0$而消失。只看标量重叠不足以取得该项。

## C.9. 五项显式有限余项预算

以下有限证书固定$m=3$、$C=C(\rho)$且$\rho=9/20$、独立教师、$K=100$和$q=98$。置换及整体反射将八标签和精确约化为两类。在(C.19)--(C.20)中，将$H_y$换成三条自由边标记的平均：

$$
\begin{aligned}
\widetilde H_y&=\tfrac13(H_{12,y}+H_{13,y}+H_{23,y}),\\
p_+&=1/4\ (y=+++),\qquad p_0=3/4\ (y=++-),\\
m_+&=m_H+1/4,\qquad m_0=m_H-1/12,\\
c_y&=E[(\widetilde H_y-m_y)T_y],\quad v_y=ET_y^2.
\end{aligned}
\tag{C.28}
$$

在六个混合标签中，固定边有两次同号、四次异号；置换到代表标签后，恰好是一条同号边和两条异号边的平均。因此最终导数仍为自由$c_{12}$偏导，不是三边径向导数。

令$\lambda=\kappa_3/(6vs)$。以下各量均使用含Gumbel的真实噪声范数的严格上帽：

$$
\begin{aligned}
\|g^{(j)}\|_\infty&\le L_j\quad(j=0,1,2),\\
\|G-\Phi\|_\infty&\le\delta_G,\quad \|g-\phi\|_\infty\le\delta_f,\\
\|g'-\phi'\|_\infty&\le\delta_p,\quad \|g'-\phi'\|_1\le V_{\rm diff},\\
\|g-\phi\|_2^2&\le a_2^2,\quad \|g-\phi-\lambda e\|_\infty\le R_0,\\
\|G-\Phi-\lambda\mathcal P\|_\infty&\le R_{\rm CDF}.
\end{aligned}
\tag{C.29}
$$

$$
\begin{aligned}
M_2&=\sum_yp_yE[|\widetilde H_y-m_y|T_y^2],\\
M_{mv}&=\sum_yp_y|m_y|v_y,\quad M_{cv}=\sum_yp_y|c_y|v_y,\\
v_{\max}&=\max_yv_y,\qquad \Delta=\delta_G+L_1v_{\max}/(2s^2).
\end{aligned}
\tag{C.30}
$$

对非负整数$r$和非负实数$d$，定义保留极值CDF因子的Gaussian比较权重：

$$
\begin{aligned}
W_1(r,d)&=\int_{\mathbb R}|\phi'|\min(\Phi+d,1)^r,\\
W_{1,+}(r,d)&=\int_0^\infty|\phi'|\min(\Phi+d,1)^r,\\
W_2(r,d)&=\int_{\mathbb R}|\phi''|\min(\Phi+d,1)^r,\\
\mathcal W_r&=W_1(r,\Delta)+V_{\rm diff}.
\end{aligned}
\tag{C.31}
$$

五个非负预算为

$$
\begin{aligned}
\mathsf E_1&=\frac{KL_2M_2}{2s^3},\\
\mathsf E_2&=\frac{K(K-1)M_{mv}}{2s^3}
\left[\frac{L_2}{K-1}+L_1\mathcal W_q\right],\\
\mathsf E_3&=\frac{K(K-1)M_{cv}}{2s^4}
\left[L_2\mathcal W_q+qL_1L_0\mathcal W_{q-1}\right].
\end{aligned}
\tag{C.32}
$$

$$
\begin{aligned}
\mathsf E_4&=\frac{K(K-1)|\bar c|}{s^2}
\left[\frac{\delta_p}{K-1}+\delta_GW_2(q,\delta_G)\right],\\
\mathsf E_5&=\frac{K(K-1)|m_H|}{s}\left\{
\frac{2R_0+J_KR_{\rm CDF}}{K-1}+a_2^2+2\delta_f\delta_G\right.\\
&\hspace{18mm}\left.+\frac q2\delta_G^2[W_{1,+}(q-1,\delta_G)+\delta_f]\right\},\\
|D_n-D_n^{(2)}|&\le\mathsf E_n:=\sum_{j=1}^5\mathsf E_j.
\end{aligned}
\tag{C.33}
$$

只要所列有限输入成立，该推论对每个$n\ge1$成立；下文噪声构造在四个指定规模认证这些输入。预算$\mathsf E_j$与CDF校正函数$\mathcal P$使用不同记号。

下面给出传播证明，明确符号与常数。积分Taylor展开给出

$$
\begin{aligned}
\|f_y-g\|_\infty&\le\delta_{f,y}:=L_2v_y/(2s^2),\\
\|F_y-G\|_\infty&\le\delta_{F,y}:=L_1v_y/(2s^2),\\
h_y-m_yf_y&=-c_yg'/s+e_y,\\
\|e_y\|_\infty&\le L_2E[|\widetilde H_y-m_y|T_y^2]/(2s^2).
\end{aligned}
\tag{C.34}
$$

由于$\int f_yF_y^q=1/(K-1)$，(C.20)中$e_y$的贡献由$\mathsf E_1$控制。记$I_g=\int g^2G^q$、$J_g=\int g'gG^q$及$J_\phi=\int\phi'\phi\Phi^q=-T_K^{\rm G}$。两项精确分部积分恒等式为

$$
\begin{aligned}
\mathcal I(f_y)-I_g
&=\int(f_y-g)f_yF_y^q
-\frac1{q+1}\int g'(F_y^{q+1}-G^{q+1}),\\
J_g-J_\phi
&=\int(g'-\phi')gG^q
-\frac1{q+1}\int\phi''(G^{q+1}-\Phi^{q+1}).
\end{aligned}
\tag{C.35}
$$

光滑有界密度在无穷远趋零；导数界保证边界项与积分合法。使用$|u^a-v^a|\le a|u-v|\max(u,v)^{a-1}$及$\max(F_y,G)\le\min(\Phi+\Delta,1)$，得$\int|g'|\max(F_y,G)^r\le\mathcal W_r$，从而

$$
\begin{aligned}
|\mathcal I(f_y)-I_g|&\le\frac{\delta_{f,y}}{K-1}+\delta_{F,y}\mathcal W_q,\\
\left|\int g'f_yF_y^q-J_g\right|
&\le\delta_{f,y}\mathcal W_q+q\delta_{F,y}L_0\mathcal W_{q-1},\\
|J_g-J_\phi|&\le\frac{\delta_p}{K-1}+\delta_GW_2(q,\delta_G).
\end{aligned}
\tag{C.36}
$$

分别乘(C.20)的相应系数，即得$\mathsf E_2,\mathsf E_3,\mathsf E_4$。共同核先完成标签平均，其系数保留为有符号的$m_H,\bar c$，没有换成绝对值平均。

对$\mathsf E_5$，令$d=g-\phi$、$D=G-\Phi$，使用$0\le t\le1$上的正密度插值$f_t=\phi+td$、$F_t=\Phi+tD$，不假定Edgeworth截断为正。线性残差及精确二阶变分满足

$$
\begin{aligned}
|D\mathcal I_\phi[d]-\lambda D\mathcal I_\phi[e]|
&\le\frac{2R_0+J_KR_{\rm CDF}}{K-1},\\
\frac{d^2}{dt^2}\mathcal I(f_t)
&=2\int d^2F_t^q+4q\int f_tdDF_t^{q-1}\\
&\quad+q(q-1)\int f_t^2D^2F_t^{q-2}.
\end{aligned}
\tag{C.37}
$$

首界使用$\int\phi\Phi^q=1/(K-1)$及$q\int\phi^2\Phi^{q-1}=J_{K-1}/(K-1)\le J_K/(K-1)$。二阶变分前两项的绝对上帽为$2a_2^2$与$4\delta_f\delta_G$。对末项，

$$
\begin{aligned}
\int f_t^2F_t^{q-2}
&=\int(f_t-\phi)f_tF_t^{q-2}-\frac1{q-1}\int\phi'F_t^{q-1}\\
&\le\frac{\delta_f+W_{1,+}(q-1,\delta_G)}{q-1},\\
|I_g-I_K-D\mathcal I_\phi[d]|
&\le a_2^2+2\delta_f\delta_G\\
&\quad+\frac q2\delta_G^2[\delta_f+W_{1,+}(q-1,\delta_G)].
\end{aligned}
\tag{C.38}
$$

负半轴上$\phi'>0$，因此带负号的贡献非正，可以舍去；这正是使用正半轴权重的依据，不能先丢掉符号再这样做。末行保留Taylor因子$\int_0^1(1-t)dt=1/2$。合并(C.37)--(C.38)，再使用(C.21)、(C.26)，即得$\mathsf E_5$并闭合(C.33)。含$q-1$的除法仅在固定$q=98$处使用。

## C.10. 真实噪声的有限Fourier界

使用约定$\widehat f(t)=\int e^{itz}f(z)\,dz$。记$U=(L-\ell)/\sqrt v$、$\mu_3=EU^3$、$\mu_4=EU^4$、$\alpha=\mu_3/6$、$P_n(t)=\psi(t/\sqrt n)^n$及$Q_n(t)=e^{-t^2/2}(1+\alpha(it)^3/\sqrt n)$。后者是$\phi+\lambda e$的变换，不对应非负概率密度。实际生产分区采用$\delta=1/4$、$a=9/2$，低、中、高频分界为$\delta\sqrt n,a\sqrt n$。

**低频。** 定义

$$
\begin{aligned}
q_\delta&=\tfrac12+|\mu_3|\delta/6+\mu_4\delta^2/24,
\quad z_\delta=\delta^2q_\delta,\\
C_\delta&=\mu_4/24+\frac{q_\delta^2}{2(1-z_\delta)},\\
D_\delta&=\frac{\mu_4/6+(1+|\mu_3|\delta/2)q_\delta}{1-z_\delta},\\
c&=\tfrac12-C_\delta\delta^2,\quad
c_{\rm mod}=\tfrac12\{1-(2\mu_4+6)\delta^2/24\}.
\end{aligned}
\tag{C.39}
$$

认证第四矩为$\mu_4\simeq4.53778313038698$；保存区间给出$z_\delta<.034973$、$c>.4780439$、$c_{\rm mod}>.4803703$。三阶Taylor展开、$|\log(1+w)-w|\le |w|^2/[2(1-|w|)]$及$\psi'/\psi$给出$\log\psi(u)=-u^2/2+\alpha(iu)^3+r(u)$，其中$|r(u)|\le C_\delta|u|^4$、$|r'(u)|\le D_\delta|u|^3$。这些有限界不需要第五矩。对$0\le t\le\delta\sqrt n$，

$$
\begin{aligned}
|P_n-Q_n|&\le A_0(t):=e^{-ct^2}\frac{C_\delta t^4}{n}
+e^{-t^2/2}\frac{\alpha^2t^6}{2n},\\
|P_n'-Q_n'|&\le A_1(t):=tA_0(t)+e^{-t^2/2}\frac{3\alpha^2t^5}{n}\\
&\quad+e^{-ct^2}\left(\frac{3|\alpha|C_\delta t^6}{n^{3/2}}
+\frac{D_\delta t^3}{n}\right),\\
|P_n(t)|&\le e^{-\max(c,c_{\rm mod})t^2}.
\end{aligned}
\tag{C.40}
$$

三次相位为纯虚数，故$|e^{ib}-1-ib|\le b^2/2$保留相位而不付出$e^{|b|}$损失。额外模界来自对$U-U'$使用四阶余弦界，其第四矩为$2\mu_4+6$。正系数向上取端点，衰减率向下取端点。

**中高频。** 在273个频率$u_j=1/4+j/64$、$j=0,\ldots,272$处，认证复有限和给出$|\psi(u_j)|$的上帽$b_j$。由$|\psi'|\le E|U|\le1$，272个小区间分别有合法上帽

$$
\begin{aligned}
\rho_j&=\min\{1,\max(b_j,b_{j+1})+1/128\}<1,\\
|\psi(u)|&\le \min(1,B/|u|),\qquad
B=\frac{2\sqrt v(1+e^{1/2})}{\sqrt{2\pi}}<1.102.
\end{aligned}
\tag{C.41}
$$

高频界的证明为：$p_L(\log\sigma(x))=\phi(x)(1+e^x)$，其关于$x$的对数导数是$-x+\sigma(x)$，严格递减且两端异号。密度因此单峰、两端趋零，故$\|p_L'\|_1=2\sup p_L\le2(1+e^{1/2})/\sqrt{2\pi}$；缩放及分部积分给出(C.41)。这给出显式尾常数。每个中频小区间有$|P_n|\le\rho_j^n$、$|P_n'|\le\sqrt n\rho_j^{n-1}$。高频矩满足

$$
\begin{aligned}
\int_{a\sqrt n}^{\infty}t^p(B\sqrt n/t)^kdt
&=\frac{n^{(p+1)/2}B^ka^{p-k+1}}{k-p-1},\quad k>p+1,\\
I_p(c;t_0)&:=\int_{t_0}^{\infty}t^pe^{-ct^2}dt,\\
I_0(c;t_0)&=\frac{\sqrt\pi}{2\sqrt c}\operatorname{erfc}(\sqrt c\,t_0),
\quad I_1(c;t_0)=\frac{e^{-ct_0^2}}{2c},\\
I_p(c;t_0)&=\frac{t_0^{p-1}e^{-ct_0^2}+(p-1)I_{p-2}(c;t_0)}{2c}\quad(p\ge2).
\end{aligned}
\tag{C.42}
$$

中频小区间的多项式积分是精确端点幂差。(C.42)处理$n,n-1,2n,2n-2$次幂及全部Gaussian多项式尾。低频正项包络可保守延伸至无穷，其平方保留所有交叉项。Gaussian比较尾从$t_0=\delta\sqrt n$开始仅计一次，不在每个中频小区间重复计入。这些步骤不需要Fourier数值求积。

**Gumbel与范数转换。** 对$\chi_n(t)=Ee^{it\xi/s}$，中心化及$E\xi^2=\pi^2/6$给出

$$
\begin{aligned}
|\chi_n|&\le1,\quad |\chi_n-1|\le b(t):=\pi^2t^2/(12nv),\\
|\chi_n'|&\le d(t):=\pi^2|t|/(6nv),\\
R_g&:=\chi_nP_n-Q_n,\\
|R_g|&\le A_g:=A_0+b|Q_n|,\\
|R_g'|&\le A_g':=A_1+d\,e^{-\max(c,c_{\rm mod})t^2}+b|Q_n'|
\quad(0\le t\le\delta\sqrt n).
\end{aligned}
\tag{C.43}
$$

这里$A_g'$表示$|R_g'|$的上帽，不是包络$A_g$的导数。其余频率使用$|R_g|\le M+|Q_n|$及$|R_g'|\le M_1+dM+|Q_n'|$，其中$M,M_1$是前述$|P_n|,|P_n'|$上帽。中心化Gumbel因此进入每个有限误差范数。

令$r_j=g^{(j)}-(\phi+\lambda e)^{(j)}$。反演、CDF反演及Parseval给出下列充分界：

$$
\begin{aligned}
\|r_j\|_\infty&\le\pi^{-1}\int_0^\infty t^jA_g(t)dt\quad(j=0,1,2),\\
\|G-\Phi-\lambda\mathcal P\|_\infty&\le\pi^{-1}\int_0^\infty A_g(t)/t\,dt,\\
\|r_0\|_2^2&\le\pi^{-1}\int_0^\infty A_g(t)^2dt,\\
\|r_1\|_1&\le\left[\int_0^\infty\{t^2A_g^2+(A_g+tA_g')^2\}\,dt\right]^{1/2},\\
\|g^{(j)}\|_\infty&\le L_{j,\rm direct}:=\pi^{-1}\int_0^\infty t^jM(t)dt.
\end{aligned}
\tag{C.44}
$$

对$L^1$一行，使用权重$(1+z^2)^{-1}$的Cauchy--Schwarz及Parseval给出$\|r_1\|_1^2\le\tfrac12(\|tR_g\|_2^2+\|(tR_g)'\|_2^2)$。对称性将其化为所列正频积分，不再附加$\pi$因子。CDF积分在零处可积：Gumbel误差从$t^2$开始，其余误差从$t^4$开始。在中高频，实际预算先取CF块与Gaussian块的向量$L^2$帽的Minkowski和再平方，归一化相同。直接密度帽使用$|\chi_n|\le1$。

令$R_1,R_{02},R_{11}$分别为所得$\|r_1\|_\infty,\|r_0\|_2,\|r_1\|_1$上帽。(C.29)的输入通过保留一阶Edgeworth项恢复为

$$
\begin{aligned}
\delta_f&=|\lambda|\|e\|_\infty+R_0,\quad
\delta_p=|\lambda|\|e'\|_\infty+R_1,\\
\delta_G&=\min(1,|\lambda|\phi(0)+R_{\rm CDF}),\\
V_{\rm diff}&=|\lambda|\|e'\|_1+R_{11},\quad
 a_2^2=(|\lambda|\|e\|_2+R_{02})^2,\\
L_0&=\min(L_{0,\rm direct},\phi(0)+\delta_f),\\
L_1&=\min(L_{1,\rm direct},\phi(1)+\delta_p),\qquad L_2=L_{2,\rm direct}.
\end{aligned}
\tag{C.45}
$$

所有比较均取上端点。所需Hermite范数是闭式标量表达。令$a_*=\sqrt{3-\sqrt6}$、$b_*=\sqrt{3+\sqrt6}$，$\operatorname{He}_j$为概率论约定的Hermite多项式，则

$$
\begin{aligned}
\|e\|_\infty&=\max_{z\in\{a_*,b_*\}}|\operatorname{He}_3(z)\phi(z)|,\\
\|e'\|_\infty&=\max_{z\in\{0,\sqrt{5-\sqrt{10}},\sqrt{5+\sqrt{10}}\}}
|\operatorname{He}_4(z)\phi(z)|,\\
\|e'\|_1&=4\sqrt6[a_*\phi(a_*)+b_*\phi(b_*)],\qquad
\|e\|_2^2=\frac{15}{16\sqrt\pi}.
\end{aligned}
\tag{C.46}
$$

这些公式由$e'=-\operatorname{He}_4\phi$、$e''=\operatorname{He}_5\phi$的零点、极值间总变差以及Gaussian偶矩得到。

**有限CF表与第四矩的认证。** 原Gaussian变量采用全权梯形和，$h=1/16$、$x_j=jh$、$|j|\le192$、截断$R=12$、条带半宽$\eta=1$及192位球算术。记$d_0=E\log\cosh(Z/2)$，于是$L-\ell=x/2-\log\cosh(x/2)+d_0$。在$|\operatorname{Im}x|\le\eta$上有$|\operatorname{Im}\log\sigma(x)|\le\eta$。条带积分帽及所得误差为

$$
\begin{aligned}
M_\psi(u)&=\exp(\eta^2/2+|u|\eta/\sqrt{v_-}),\\
\varepsilon_\psi(u)&=\frac{2M_\psi(u)}{e^{2\pi\eta/h}-1}+\frac{2\phi(R)}R,\\
B_4^*&=d_{0,+}+\eta+\log\sec(\eta/2),\\
M_4&=e^{\eta^2/2}E(|Z|+B_4^*)^4,\\
\varepsilon_4&=\frac{2M_4}{e^{2\pi\eta/h}-1}
+\sum_{k=0}^4{4\choose k}d_{0,+}^{4-k}P_k(R).
\end{aligned}
\tag{C.47}
$$

第四矩条带界使用$|L-\ell|\le|\operatorname{Re}x|+B_4^*$，其期望由绝对Gaussian矩$1,\sqrt{2/\pi},1,2\sqrt{2/\pi},3$展开。无限梯形混叠界是两条条带边界的积分帽之和除以$e^{2\pi\eta/h}-1$。对遗漏的真实轴格点尾，由$x\ge R\ge2$、$k\le4$时$x^k\phi(x)$的单调性，

$$
\begin{aligned}
P_0(R)&=2\phi(R)/R,\quad P_1(R)=2\phi(R),\\
P_2(R)&=2\phi(R)(R+1/R),\\
P_3(R)&=2\phi(R)(R^2+2),\\
P_4(R)&=2\phi(R)(R^3+3R+3/R).
\end{aligned}
\tag{C.48}
$$

原始第四矩先加$\varepsilon_4$，再除以保存的$v^2$区间。每个复有限和加入复模误差$\varepsilon_\psi$，真模帽为$|\text{finite sum}|+\varepsilon_\psi$。全部输入dyadic区间与算术半径均保留。385个共享Gaussian节点产生273个复CF和及一个实第四矩和，即547个实标量有限和和105105次复指数。这些是模型噪声计算，不是直接求值$D_n$。

## C.11. 不增加活跃积分的矩上帽

全部活跃输入来自已认证标量$V_A,V_H,s_0,m_H,\bar c$。这里$V_H=E_Y\operatorname{Var}_XH_{12,Y}$、$s_0=E\operatorname{sech}^2(Z/2)$。记$d(x)=\log\cosh(x/2)$、$\mathcal D=\sum_{i=1}^3[d(X_i)-Ed(Z)]$、$S_y=y^TX/2$及$a_y=Cy/2$。于是$T_y=S_y-\mathcal D$；奇偶对称性给出$ES_y\mathcal D=0$。令$\lambda_C=1+2\rho$、$\kappa=4m_H$及$t_2=1-s_0$。

$$
\begin{aligned}
V_d&=E\mathcal D^2=V_A-3/4,\\
\sigma_+^2&=3\lambda_C/4,\quad \sigma_0^2=(3-2\rho)/4,\\
v_+&=\sigma_+^2+V_d,\qquad v_0=\sigma_0^2+V_d.
\end{aligned}
\tag{C.49}
$$

对类协方差，沿三边$\rho$路径的Price微分仅作为矩恒等式使用。两次分部积分给出$ES_y^2\mathcal D=(s_0/4)\|a_y\|^2$，因而

$$
\begin{aligned}
ET_y^3&=-\tfrac{3s_0}4\|a_y\|^2-E\mathcal D^3,\\
\frac d{d\rho}ET_y^3&=6\sum_{i<j}E[T_yH_{ij,y}]=18c_y,\\
\|a_+\|^2-\|a_0\|^2&=4\rho+2\rho^2,\\
c_+&=\bar c-\frac{s_0(1+\rho)}8,\qquad
c_0=\bar c+\frac{s_0(1+\rho)}{24}.
\end{aligned}
\tag{C.50}
$$

系数18来自立方混合导数的系数6乘三条边。除以该系数并使用$\bar c=(c_++3c_0)/4$即得末行，没有把径向导数代入$D_n$。

对第四矩，写$X=C^{1/2}Z_3$。由$|d'|\le1/2$，有$\|\nabla_{Z_3}\mathcal D\|^2\le3\lambda_C/4$。将Gaussian Poincare不等式用于$\mathcal D^2$，得

$$
E\mathcal D^4-V_d^2\le4E[\mathcal D^2\|\nabla_{Z_3}\mathcal D\|^2]
\le3\lambda_CV_d,\qquad Q_4:=V_d^2+3\lambda_CV_d.
\tag{C.51}
$$

这里的Poincare不等式可由$\operatorname{Var}u=2\int_0^\infty E\|\nabla P_tu\|^2dt$、$\nabla P_tu=e^{-t}P_t\nabla u$及Jensen得到。Gaussian矩保证所用函数与梯度均平方可积。

混合项需要保留符号。两次分部积分给出

$$
\begin{aligned}
ES_y^2\mathcal D^2
&=\sigma_y^2V_d+2E(a_y\cdot\nabla\mathcal D)^2
+2E[\mathcal D(a_y\cdot\nabla)^2\mathcal D],\\
E[\mathcal D(a_y\cdot\nabla)^2\mathcal D]&\le0.
\end{aligned}
\tag{C.52}
$$

为证第二行，条件于$X_i=x$。每个$E[d(X_j)\mid X_i=x]$都是$x$的偶凸函数，因此随$|x|$非减；而$d''(x)=\tfrac14\operatorname{sech}^2(x/2)$随$|x|$非增。对$|X_i|$的独立副本$R,R'$，协方差恒等式$2\operatorname{Cov}(a(R),b(R))=E[(a(R)-a(R'))(b(R)-b(R'))]\le0$证明$\operatorname{Cov}(\sum_jd(X_j),d''(X_i))\le0$。$\mathcal D$的Hessian为对角矩阵，乘非负系数$(a_y)_i^2$求和即得所需符号。

$$
\begin{aligned}
Q_y&:=E(a_y\cdot\nabla\mathcal D)^2
=\tfrac14\{(t_2-\kappa)\|a_y\|^2+\kappa(\mathbf1^Ta_y)^2\},\\
ET_y^4&\le3(\sigma_y^2)^2+6\sigma_y^2V_d+12Q_y+Q_4,\\
\overline T_4&:=\frac{27+36\rho^2}{16}
+\left(\frac92+3\lambda_C\right)V_d+V_d^2\\
&\quad+\frac94(1+2\rho^2)t_2+\frac92\rho(2+\rho)\kappa,\\
\sum_yp_yET_y^4&\le\overline T_4.
\end{aligned}
\tag{C.53}
$$

整体反射使含奇数次$S_y$的项消失。末两行所用标签平均为$E_y\sigma_y^2=3/4$、$E_y(\sigma_y^2)^2=(9+12\rho^2)/16$、$E_y\|a_y\|^2=(3+6\rho^2)/4$及$E_y(\mathbf1^Ta_y)^2=3\lambda_C^2/4$。对三边平均使用Jensen，再由标签置换对称性得$\sum_yp_y\operatorname{Var}(\widetilde H_y)\le V_H$。在$(y,X)$上使用Cauchy--Schwarz即得

$$
M_2\le\sqrt{V_H\overline T_4}<.307468813.
\tag{C.54}
$$

这是余项所需矩的上帽，不是真实$M_2$的测量值。结合(C.28)、(C.49)、(C.50)，即可提供(C.30)的全部活跃输入，无需新增活跃Gaussian积分。

## C.12. Gaussian权重包围

每个规模需要四个权重：$W_1(98,\Delta)$、$W_1(97,\Delta)$、$W_2(98,\delta_G)$及$W_{1,+}(97,\delta_G)$。计算仅使用$[-12,12]$上间距$1/128$的3073节点网格中的$\phi,\Phi$，精度为192位。对小区间$[a,b]$，权重$w(z)=\min(\Phi(z)+d,1)^r$非减；取$w_- =\min(\underline\Phi(a)+d,1)^r$及$w_+=\min(\overline\Phi(b)+d,1)^r$，并将CDF包围裁剪到$[0,1]$。网格包含$-1,0,1$，没有小区间跨过未处理的导数变号点。

$$
\begin{aligned}
\int_a^b|\phi'|&=|\phi(b)-\phi(a)|,\\
\int_a^b|\phi''|&=|-b\phi(b)+a\phi(a)|,\\
M_-w_-&\le\int_a^bw(z)|\phi^{(j)}(z)|dz\le M_+w_+,\\
\int_{|z|>R}|\phi'|&=2\phi(R),\quad \int_R^\infty|\phi'|=\phi(R),\\
\int_{|z|>R}|\phi''|&=2R\phi(R)\qquad(R\ge2).
\end{aligned}
\tag{C.55}
$$

前两行分别用于不跨$0$及$\{-1,1\}$的小区间；$[M_-,M_+]$是相应导数质量的区间。下、上乘积各自求和，并仅向上和加入所列尾界，即包围各权重。传播使用上端点以及$d$的上帽。因此16个全实线权重界均由保存的Gaussian值及单调性得到，没有通过网格吻合来推断求积余项。

## C.13. 模型常数与完整四规模合成

严格常数来自模型积分，不是拟合的梯度系数。(C.19)的低维求值使用$t(x)=\tanh(x/2)$、$d(x)=\log\cosh(x/2)$以及$X_i=\sqrt\rho U_0+\sqrt{1-\rho}V_i$，其中各变量为独立标准Gaussian。在$X=\sqrt\rho u+\sqrt{1-\rho}V$处定义$\tau(u)=E_Vt(X)$、$\beta(u)=E_Vd(X)$、$\gamma(u)=E_V[t(X)d(X)]$、$\nu(u)=E_Vt(X)^2$。标签平均及Gaussian分部积分给出

$$
\begin{aligned}
d_0&=Ed(Z),\quad s_0=1-Et(Z)^2,\quad v=\tfrac14+\operatorname{Var}d(Z),\\
\kappa_3&=-3s_0/16-E[(d(Z)-d_0)^3],\quad \kappa=E\tau(U_0)^2,\\
m_H&=\kappa/4,\\
\bar c&=-\rho s_0/8-\{2E[\tau\gamma]+E[\tau^2\beta]-3d_0\kappa\}/4,\\
V_H&=\{2Et(Z)^2+E\nu(U_0)^2-\kappa^2\}/16,\\
V_A&=3/4+3(v-1/4)+6\{E\beta(U_0)^2-d_0^2\}.
\end{aligned}
\tag{C.56}
$$

这里每个外期望均对$U_0$取值。第三个活跃坐标贡献$E[\tau^2\beta]-d_0\kappa$，没有被舍去。这些表达在等相关点求值自由坐标矩，不沿三边路径求导。

认证求值使用全权Gaussian梯形和，间距$1/16$、截断$10$、条带半宽$3/4$及192位精度。由$|y|\le\pi/2$时的$|t(x+iy)|\le1$及$|d(x+iy)|\le|x|/2+\log\sec(|y|/2)+|y|/2$，得到与(C.47)相同的混叠公式及显式多项式Gaussian条带、尾包络。每个条件内积分先加自己的混叠与尾界再形成乘积；外和随后加入自身误差。(C.21)的Gaussian次序统计积分单独处理，使用$|\Phi(x+i\eta)|\le1+\eta e^{\eta^2/2}/\sqrt{2\pi}$。完整包络及精确保存区间由配套常数计算提供，没有将普通浮点求积规则间差异作为严格误差界。

|常数|约值|
|---|---:|
|$v$|.271514501800558731839|
|$\kappa_3$|−.16208683226381804873|
|$m_H$|.01927127885302918387252|
|$\bar c$|−.0547240726452943675562|
|$J_{100}$|2.50759363644168|
|$B_{100}$|5.47243065910866|
|$V_H$|.02360133572724289470720|
|$V_A$|.84051923403528451804|

$$
\begin{aligned}
b_2^{\rm nuisance}&=B_{100}m_H\kappa_3/(3v^2)\simeq-.0772914085493864,\\
b_2^{\rm mark}&=-B_{100}\bar c/v\simeq1.102974945903173,\\
1.02568353735378&<b_2<1.02568353735380.
\end{aligned}
\tag{C.57}
$$

较大的正修正主要来自有符号的标记—得分项。这是解析展开中的机制，不是训练结果的因果归因。表中小数仅供阅读；严格系数区间及下表最终外包均使用完整保存dyadic输入。

|$n$|$D_n^{(2)}$约值|$\mathsf E_n$严格上帽|$D_n$的向外包围|判向|
|---:|---:|---:|---|---|
|64|+.00443369615|99.704837|[−99.70041, 99.70928]|未分辨|
|256|−.00178972824|.35027010|[−.35205983, .34848037]|未分辨|
|1024|−.00189650820|.011695409|[−.013591917, .009798901]|未分辨|
|4096|−.00119866512|.000846727778|[−.002045393, −.0003519373]|严格负|

下表完整保留五项预算表达式的近似读数；其中任何一项都不是实际误差的下界。

|$n$|$\mathsf E_1$|$\mathsf E_2$|$\mathsf E_3$|$\mathsf E_4$|$\mathsf E_5$|
|---:|---:|---:|---:|---:|---:|
|64|.135264699|12.7311527|82.8304548|.445542322|3.56242253|
|256|.0114049517|.0790174116|.223704548|.00979174301|.0263514419|
|1024|.00140485780|.00341057177|.00422225078|.00071844314|.00193928492|
|4096|.00017560722|.00026772948|.00014413379|.00006758914|.00019166814|

64尺度的界合法但很松，不能解释其旧正导数估计；其二阶中心也仍在C.6旧区间外。在256和1024处，中心为负不等于真实符号已确定。在1024处，即使令$\mathsf E_2=\mathsf E_3=0$并保留其余预算，也有$\mathsf E_1+\mathsf E_4+\mathsf E_5\simeq.00406258585509>|D_{1024}^{(2)}|$；$\mathsf E_5$单项已超过该窗口。因此只降低这两项不能使当前固定配方通过。这既不是实际误差下界，也不表示其他方法不可能。

在保留的4096尺度点，完整有限合成证明

$$
\mathsf E_{4096}<.000846727778,\qquad
-.002045393<D_{4096}<-.0003519373<0.
\tag{C.58}
$$

这也给出与原首项场的直接认证联系。令$D_n^{(0)}=-m_HJ_{100}/\sqrt{nv}$，则

$$
\begin{aligned}
D_{4096}^{(0)}&\simeq-.00144907613957,\qquad b_2/4096\simeq.000250411019862,\\
|D_{4096}-D_{4096}^{(0)}|
&\le |b_2|/4096+\mathsf E_{4096}\\
&<.001097138798<|D_{4096}^{(0)}|.
\end{aligned}
\tag{C.59}
$$

首项相对上预算约为75.7%，二阶余项帽约为二阶中心绝对值的70.6%。这是符号层面的有限适用性见证，不是10%近似精度保证。断言仅针对$n=4096$，不是阈值或所有更大$n$的结论；也没有建立全矩阵有限余项、参数区域或优化器后果。

旧四尺度面板在推导修正前已经可见。模型常数虽未对其梯度回归，对照仍属于知情的事后分析。新的确定性包围不是置信区间，不消耗alpha，也不升级旧理想iid或原生算术资格。C.6保留原$J_{100}$数值参考及其来源。

(C.58)--(C.59)的严格性来自解析包络、保存球算术及精确端点合成。模型常数认证使用1296个标量和、103041个嵌套节点；之后的噪声和权重计算使用C.10、C.12中的计数。独立普通Gaussian-Hermite与Gaussian-Legendre表达检查一致性，其中覆盖全部273个CF值，但不是第二套严格证书。独立有理重构从保存端点核对16个权重包围、20个预算表达式及4个最终区间，不重新证明全部解析包络。没有直接$D_n$积分、新随机银行或优化。配套记录保存完整有限和、误差包络及算术收据。备选分位数窗口接口不属于这套全实线证书：其中央窗口未计算，其单独尾预算也没有在此叠加。

## C.14. 已认证有限点上的局部观察风险推论

对三个中心Gaussian logit，整体反射使全部奇数阶spin矩消失。由条件独立性展开三个Bernoulli因子，得到下列第一个身份。教师在八标签上均匀，记$\kappa(c)=E_c[\tanh(X_1/2)\tanh(X_2/2)]$。每个二维边际只依赖自身相关，因此自由$c_{12}$变化不改变$\kappa(c_{13}),\kappa(c_{23})$：

$$
\begin{aligned}
Q_C(y)&=\tfrac18\left(1+\sum_{i<j}\kappa(c_{ij})y_iy_j\right),\\
\left.\partial_{c_{12}}\mathcal R(C)\right|_{C=C(\rho)}
&=-\frac{\kappa'(\rho)}8
\left(\frac2{1+3\kappa(\rho)}-\frac2{1-\kappa(\rho)}\right)\\
&=\frac{\kappa'(\rho)\kappa(\rho)}{(1-\kappa(\rho))(1+3\kappa(\rho))}>0,
\qquad \rho=9/20.
\end{aligned}
\tag{C.60}
$$

两条全同号标签的$y_1y_2$之和为2，三条pair乘积之和为3；其余六条对应为−2和−1。对均匀教师交叉熵求导即得(C.60)，KL只差一个常数。(5)给出$\kappa'(c)>0$；由$\kappa(0)=0$及$|\tanh|<1$，有$0<\kappa(9/20)<1$。这是解析正性论证，没有新增$\kappa'$数值积分。沿$\rho$同时改变三条边时，该导数需另乘3。

令$A=e_1e_2^\top+e_2e_1^\top$、$C_h=C(9/20)+hA$。因$\|A\|_{\rm op}=1$及$\lambda_{\min}(C(9/20))=11/20$，$|h|<9/20$保证$C_h$仍在论文的$\delta=.1$相关矩阵类内部；Gaussian边际及公平观察边际均保持。该界只保证可行性。结合可微性、(C.58)和(C.60)，存在$\varepsilon>0$，使每个$0<h<\varepsilon$均可行且满足$F_{4096,100}^*(C_h)<F_{4096,100}^*(C(9/20))$及$\mathcal R(C_h)>\mathcal R(C(9/20))$。未给出显式单调性半径或有限风险增量。结论仅为单侧、局部且指定坐标，不声称驻点或优化器后果。

## C.15. 有限程序的完整陈述 {#finite-evidence-summary}

本节分别考察保存运行的选择、连续低风险区的近优性及新运行的正风险概率。固定$m=3$、$\mu=0$、$C_*=I$、$\delta=.1$、$K=100$。两个程序共享65,536个标签、活跃粒子、143个可行候选及$10^{-12}$词典序容差。一次匹配运行中，将均匀权重替换为256独立干扰因子的权重，选择从$I$（零KL）变为等相关$C_b$：非对角元.9、谱(.1,.1,2.8)、KL .03110309。选择不使用风险。100是内层粒子数；一组先验粒子称为粒子银行，评分按经验标签频率在16,384个共享外层银行上平均。

去掉与候选无关的干扰常数后，记$F_n=F_{n,100}$。对连续低风险集$\mathcal R(C)\le r_0=10^{-6}$，Price--Pinsker及矩阵平方根导数界给出$\widehat F(C)\ge\widehat F(I)-L_r$。因此，不差于$C_b$的输出满足

$$
\widehat F(I)-\widehat F(C_b)>L_r
\quad\Longrightarrow\quad \mathcal R(\widehat C)>r_0.
$$

区间重放给出保存输入的经验区域余量.003906703321。这排除了该准则下整个连续低风险集，而不只是网格上的$I$。

原总体低风险排除界不够紧，失败保留于附录C。确定性细化给出总体代价界$c_d<.001447106$；另行银行矩审计对$F_{256}(I)-F_{256}(C_b)$的界减去$c_d$，在同一99%事件上留下大于$.00134$的余量（附录D.5）。取$\eta=.00134$，保留声明的采样和输入资格，

$$
\begin{gathered}
\inf_{\mathcal R(C)\le10^{-6}}F_{256}(C)
>F_{256}(C_b)+\eta,\\
F_{256}(\widehat C)-\inf_C F_{256}(C)<\eta
\ \Longrightarrow\ \mathcal R(\widehat C)>10^{-6}.
\end{gathered}
$$

这既不证明$C_b$全局最优，也不把它的点风险.03110309赋予所有极小点。

在声明的iid采样与输入数值资格下，对具有143候选、$K=100$、$N=65{,}536$、$B=16{,}384$及相同容差的数学干扰程序的**新运行**，使用另行生成的$L=262{,}144$银行样本的矩审计以至少99%审计置信度给出

$$
\begin{gathered}
\Pr_{\rm fresh}\{\mathcal R(\widehat C)\ge r_{\rm grid}\}\ge .7035,\\
r_{\rm grid}=\frac{0.09\,\exp(-1)}{32}>.00103466.
\end{gathered}
$$

新运行界的70.35%是指定程序新运行的概率下界，99%是审计置信度；前者不是实测频率，另行生成样本也不表示第三方审计。每个非$I$网格点有绝对值至少.3的相关，Price--Pinsker给出$r_{\rm grid}$，经验锚点差为正则排除唯一低风险点。附录D用Cantelli不等式、共享银行方差与矩界，未将$NB$个配对视为独立样本。总体区域与新运行陈述共享审计事件，分别约束总体近优和指定新运行，不覆盖连续优化器、更细网格或均匀分支恢复。未来原生浮点运行还需统一误差保证。
