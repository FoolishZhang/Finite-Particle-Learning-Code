# 两类尺度的NEG族：下界仍成立，单总count控制不再同Q

这是SCOPE.md登记的独立v2后续解析工作，日期2026-09-30。没有新模拟、数值积分、拟合或自然实验。**不得将本文件并入已封存v1的摘要、正文、图或证明。**

所有对数和KL使用自然对数；TC定义为边际熵之和减去完整联合熵。

## 1. 固定模型与符号

沿偶数 $d\ge4$，令 $m=d/2$，$A,B$ 为固定的两个m元素集合。固定 $0<c<1$、有限且不同的 $s_A,s_B>0$，均不随d变化。轻微打破交换性可一次固定 $s_A=s+\delta,s_B=s-\delta$，$0<\delta<s/2$；不扫描或数值选择delta。

\[
 \rho_d=\frac c{d-1},\quad U\sim N(0,R_d),\quad
 R_d=(1+\rho_d)I-\rho_d\mathbf1\mathbf1^\top.
\]

对 $i\in g$，$g\in\{A,B\}$，令 $\eta_i=s_gU_i$，且
\[
 C_i\mid U\stackrel{\mathrm{ind}}{\sim}\operatorname{Ber}(g_0(s_gU_i)),
 \quad g_0=\operatorname{sigmoid},\qquad Y_i=2C_i-1.
\]
观察联合记为 $P_d$。定义两个固定奇函数及其矩
\[
 f_g(z)=\tanh(s_gz/2),\quad
 a_g=E[Zf_g(Z)]=E f'_g(Z),\quad
 v_g=E f_g(Z)^2,\qquad Z\sim N(0,1).
\]
Gaussian分部积分给中间等式。正尺度下 $0<a_g<1$、$0<v_g<1$；前者的正性来自 $Zf_g(Z)>0$，而Cauchy–Schwarz给 $a_g^2\le v_g<1$。记
\[
 \bar a=(a_A+a_B)/2,\qquad \bar v=(v_A+v_B)/2,\qquad
 \beta_*=c\bar a^2\in(0,c).
\]
各观察边际仍为1/2。$R_d$ 特征值为 $1-c$ 及 $1+\rho_d$，严格正定；有限尺度下每个二值模式有正概率。模型在组内置换群 $S_A\times S_B$ 下不变，下一节的下界不需要完整交换性。

## 2. BL与finite-product-mixture下界：系数精确保留

取 $L_d=\lambda_{\max}(R_d)=1+c/(d-1)$。任意非负观察函数 $q_i$ 满足
\[
 \begin{aligned}
 E_{P_d}\prod_iq_i(C_i)^{1/L_d}
 &=E_U\prod_iE[q_i(C_i)^{1/L_d}\mid U_i]\\
 &\le E_U\prod_i\{E[q_i(C_i)\mid U_i]\}^{1/L_d}\\
 &\le\prod_i(E_{P_{d,i}}q_i(C_i))^{1/L_d}.
 \end{aligned} \tag{1}
\]
第一步用发射条件独立，第二步用凹性，最后一步用Gaussian-BL及 $L_dI-R_d\succeq0$。各channel不需要相同。使用标准化U的R，而不是非等方差logit协方差或观察相关矩阵的最大特征值。

相对熵变分式给任意观察分布V及C边际为P_d的任意辅助变量J：
\[
 \sum_iD(V_i\Vert P_{d,i})\le L_dD(V\Vert P_d),\qquad
 \sum_iI(C_i;J)\le L_dI(C;J). \tag{2}
\]
对任意 $Q(c)=\sum_{j=1}^K w_j\prod_iq_{ij}(c_i)$，若Q在P_d正质量处为0则KL无穷；否则令 $\widetilde P(c,j)=P_d(c)Q(j\mid c)$，有
\[
 \begin{aligned}
 D(P_d\Vert Q)&\ge TC_{\widetilde P}(C\mid J)\\
 &=TC(P_d)+I(C;J)-\sum_iI(C_i;J)\\
 &\ge TC(P_d)-(L_d-1)\log K.
 \end{aligned}
\]
因此
\[
 \boxed{D(P_d\Vert Q)\ge\max\{0,TC(P_d)-c\log K/(d-1)\}.} \tag{3}
\]

系数不变的原因是原channel论证本来就不要求各channel相同，不是“小扰动应当不影响结论”的推测。此部分是旧Gaussian-BL适用条件的直接核对，不宣称新的通用BL定理。原依据见既有混合界 (historical reference not distributed)，其原文件保持。

## 3. 两类型三角阵的总spin CLT

这里不能直接套用全部函数相同的iid展开。每行取iid标准Gaussian $Z_1,\ldots,Z_d$，并令
\[
 \bar Z_d=d^{-1}\sum_iZ_i,\quad
 \alpha_d=\sqrt{1+\rho_d},\quad b_d=\alpha_d-\sqrt{1-c},\quad
 U_i=\alpha_dZ_i-b_d\bar Z_d.
\]
直接计算Cov(U)得到恰好R_d。记 $b=1-\sqrt{1-c}$，则 $b_d\to b$ 且 $2b-b^2=c$。

### 3.1 条件均值部分

类型记为g(i)。两函数的前两阶导数统一有界。位移
$\Delta_i=(\alpha_d-1)Z_i-b_d\bar Z_d$ 满足
\[
 \sum_i\Delta_i^2\le
 2(\alpha_d-1)^2\sum_iZ_i^2+2b_d^2d\bar Z_d^2=O_p(1).
\]
因此Taylor二阶余项的和除sqrt(d)趋零。含 $(\alpha_d-1)$ 的一阶项也是 $O_p(d^{-1/2})$，所以
\[
 \frac1{\sqrt d}\sum_i f_{g(i)}(U_i)
 =\frac1{\sqrt d}\sum_i f_{g(i)}(Z_i)
 -b_d\left(\frac1d\sum_i f'_{g(i)}(Z_i)\right)
       \frac1{\sqrt d}\sum_iZ_i+o_p(1).
\]
两个组各自的大数律给导数平均趋于bar a。因此
\[
 \frac1{\sqrt d}\sum_i f_{g(i)}(U_i)
 =\frac1{\sqrt d}\sum_i f_{g(i)}(Z_i)
 -b\bar a\frac1{\sqrt d}\sum_iZ_i+o_p(1). \tag{4}
\]
三角阵 $(f_{g(i)}(Z_i),Z_i)$ 只有两个固定类型、相互独立且三阶绝对矩统一有界。对任意线性组合应用Lyapunov CLT，得到
\[
 \frac1{\sqrt d}\sum_i(f_{g(i)}(Z_i),Z_i)
 \Rightarrow N\!\left(0,
 \begin{pmatrix}\bar v&\bar a\\\bar a&1\end{pmatrix}\right).
\]
(4)的极限方差是 $\bar v-c\bar a^2$。

### 3.2 Bernoulli条件噪声

令 $\epsilon_i=Y_i-f_{g(i)}(U_i)$。给定整行Z，它们独立、中心、绝对值<=2，条件方差为 $1-f_{g(i)}(U_i)^2$。由于 $f_g^2$ 导数统一有界，
\[
 d^{-1}\sum_i|f_{g(i)}(U_i)^2-f_{g(i)}(Z_i)^2|
 \le C\sqrt{d^{-1}\sum_i\Delta_i^2}=o_p(1).
\]
组内大数律故给平均条件方差趋于 $1-\bar v>0$。条件特征函数展开的三阶余项总计 $O(d^{-1/2})$，对数展开的二次余项总计 $O(d^{-1})$。于是
\[
 E\!\left[\exp\left(itd^{-1/2}\sum_i\epsilon_i\right)\middle|Z\right]
 \longrightarrow \exp[-t^2(1-\bar v)/2]
\]
依概率且在L1中成立（绝对值始终<=1）。与任意有界Z可测特征函数相乘取期望，给出条件噪声极限与(4)的极限独立。

两项相加，得到
\[
 \boxed{\frac{S_d}{\sqrt d}\Rightarrow N(0,1-\beta_*),\quad
 S_d=\sum_iY_i,\quad
 \beta_*=c\left(\frac{a_A+a_B}{2}\right)^2.} \tag{5}
\]
不能把beta_*写成 $c(a_A^2+a_B^2)/2$，也不能保留原统一尺度的 $ca(s)^2$。公共校正项依赖的是平均一阶Hermite系数，再平方。

## 4. 非交换模型的TC仍为Theta(1)

令 $Q_0=\operatorname{Ber}(1/2)^{\otimes d}$，并取 $E_d=\{|S_d|\le\sqrt d\}$。模型CLT和独立二项CLT给
\[
 P_d(E_d)\to p_*:=2\Phi((1-\beta_*)^{-1/2})-1,\qquad
 Q_0(E_d)\to p_0:=2\Phi(1)-1.
\]
两极限严格位于(0,1)且不相等。DPI及有限字母KL连续性于是给
\[
 \liminf_dTC(P_d)\ge
 D(\operatorname{Ber}(p_*)\Vert\operatorname{Ber}(p_0))>0. \tag{6}
\]
将同一组不同坐标channel分别用于 $N(0,R_d)$ 和 $N(0,I)$，后者产生恰好Q_0；因此
\[
 TC(P_d)\le-\frac12\log\det R_d
 =-\frac12\{\log(1-c)+(d-1)\log(1+c/(d-1))\}.
\]
右侧趋于有限常数 $[-\log(1-c)-c]/2$，故
\[
 \boxed{TC(P_d)=\Theta(1).} \tag{7}
\]
没有由CLT直接推熵收敛，也没有把旧M2的TC/block-MI数值下界移植到本分布。

结合(3)，任取 $0<\varepsilon<\delta<\liminf_dTC(P_d)$，充分大d下任何误差<=epsilon的K-product mixture都要求
$\log K\ge(\delta-\varepsilon)(d-1)/c$。有限混合障碍不依赖完整交换性。

对不使用当前C的prior-bank评分形式，仍有
$F_K(\theta)\ge d\log2-c\log K/(d-1)$。若另外使用前轮允许独立PSD闭包点的候选模型类，原近最优代理regret推论可条件性地沿用；不能从本教师族的固定对角设定，擅自推出任意受限实现具有完全相同的可达独立参数。

## 5. 观察分布确实不再全交换

对正相关rho的Gaussian $(X,Y)$，定义
\[
 r(s,t;\rho)=E[\tanh(sX/2)\tanh(tY/2)].
\]
此函数对s、t分别严格递增。证明：$m_t(x)=E[\tanh(tY/2)\mid X=x]$ 与x严格同号；而
$\partial_s\tanh(sx/2)=(x/2)\operatorname{sech}^2(sx/2)$ 也与x同号。被积导数由 $|x|/2$ 控制，所以可以交换导数与期望，得到严格正导数。

不失一般性令 $s_A>s_B$，对rho=rho_d记
\[
 r_{AA}=r(s_A,s_A;\rho_d),\quad
 r_{AB}=r(s_A,s_B;\rho_d),\quad
 r_{BB}=r(s_B,s_B;\rho_d).
\]
于是 $r_{AA}>r_{AB}>r_{BB}>0$。原负Gaussian配合奇发射给实际spin相关分别为 $-r_{AA},-r_{AB},-r_{BB}$。两个组至少各有一对，故d>=4时P_d不是全置换交换分布。

有限类型Hermite展开还给 $r_{gh}=a_ga_h\rho_d+O(\rho_d^3)$；余项由L2系数和及Cauchy–Schwarz控制。它仅说明固定少量pair的差可以随d衰减，不能把三个r当作一个统一r。

## 6. 单总count-RB现在估计对称化模型

逐粒子的原单count代数恒等式依旧是
\[
 G_n(\eta)=\frac{\Pr(N=n\mid\eta)}{\binom dn}
 =\frac1{\binom dn}\sum_{|a|=n}f_a(\eta).
\]
但真实prior期望只能给
\[
 \boxed{EG_{|c|}(\eta)=
 \widetilde P_d(c):=\frac{P_d(N=|c|)}{\binom d{|c|}}.} \tag{8}
\]
tilde P是完整置换对称化。第5节证明二者在每个有限偶数d>=4都不同，完整支持故保证
\[
 D(P_d\Vert\widetilde P_d)
 =E_N D\!\left(P_d(C\mid N)\middle\Vert
 \operatorname{Unif}\{c:|c|=N\}\right)>0. \tag{9}
\]
原单count算法不再对原模式概率无偏，也不再由原prior的置换条件期望取得RB目标顺序。delta小不能将它变成严格同Q。

评价也可能掩盖此问题：tilde P精确保留N的分布，任何只读N的评价都看不到(9)，包括上述central事件。还存在精确分解
\[
 TC(P_d)=D(P_d\Vert\widetilde P_d)
          +D(P_d(N)\Vert\operatorname{Bin}(d,1/2)).
\]
这不认证第一项的渐近非消失、有限d大小或实质尺度；本包不计算这些量。

### 固定查询集的反馈反例

取不同i,k属于A，j属于B，固定查询集合{i,j}、未查询伙伴k。全局bit-flip对称使所有奇阶spin矩为0。三个spin的有限状态展开给
\[
 E[Y_k\mid Y_i=a,Y_j=b]
 =\frac{aE(Y_kY_i)+bE(Y_kY_j)}
        {1+abE(Y_iY_j)}.
\]
故
\[
 E[Y_k\mid Y_i=1,Y_j=-1]
 =\frac{-r_{AA}+r_{AB}}{1+r_{AB}}<0,\qquad
 E[Y_k\mid Y_i=-1,Y_j=1]
 =\frac{r_{AA}-r_{AB}}{1+r_{AB}}>0.
\]
两种反馈的总count都为1，但伙伴成功概率相差
$(r_{AA}-r_{AB})/(1+r_{AB})>0$。仅保留查询总count不再保持未查询预测。这个固定伙伴差可为O(1/d)，不能称常数尺度反馈损害。

## 7. 两个group counts修正保留同Q

记 $N_A=\sum_{i\in A}C_i,N_B=\sum_{i\in B}C_i$。组内交换性给
\[
 \boxed{P_d(c)=
 \frac{P_d(N_A=n_A,N_B=n_B)}
      {\binom m{n_A}\binom m{n_B}}.} \tag{10}
\]
定义
\[
 H_{n_A,n_B}(\eta)=
 \frac{\Pr(N_A=n_A\mid\eta_A)\Pr(N_B=n_B\mid\eta_B)}
      {\binom m{n_A}\binom m{n_B}}. \tag{11}
\]
条件于eta两块发射独立，因此分子是条件两个count的联合概率。(11)等于对 $S_A\times S_B$ 的组内排列平均。prior对该群不变，所以它是原f_c对组内orbit信息的条件期望，期望等于原P_d(c)。

更具体地，群平均本身组内不变；对任意组内不变有界测试函数psi，换变量利用prior不变性，得到 $E[f_{\rm avg}\psi]=E[f_c\psi]$，这正是条件期望的定义。这里无需把两组latent误当作相互独立。

对iid prior粒子，逐粒子orbit条件化与Jensen给
\[
 E[-\log\widehat P_{K,\rm groupRB}(c)]
 \le E[-\log\widehat P_{K,\rm raw}(c)].
\]
同Q、likelihood无偏和这个期望顺序均成立；有限K log-likelihood仍有偏。任意耦合粒子只满足单粒子边际无偏性时，不能自动得到整bank的RB顺序，还须核对coupling的组内置换不变性。

两个Poisson-binomial递推成本为 $O(m^2+m^2)=O(d^2)$；每个粒子的组内分布一般为非product，在本非退化teacher下几乎处处如此，但零logit等退化特例可以是product。这不与(3)矛盾。候选模型离开已知组内对称类时，同Q论证也会失效，不是通用非交换CBM修复。

这里修正的是沿logit eta坐标组内置换所获得的Poisson-binomial简化。标准化U本身仍有更大的置换对称群；若利用它，必须在似然中保留每个位置对应的尺度和标签，不能把“置换U”偷换成上节的单总count公式。本包不排除其他精确对称性估计器，也不分析其实现或成本。

### 固定子集反馈

固定查询集合T，令 $T_A=T\cap A,T_B=T\cap B,R=T^c$。只在两个已查询子组内置换、保持R不动仍是模型对称性，因此
\[
 P_d(C_R\mid C_T=a)
 =P_d(C_R\mid N_{T_A}=|a_{T_A}|,\,
                    N_{T_B}=|a_{T_B}|).
\]
相同论证适用于未查询latent边缘。不覆盖自适应/信息性查询、未知分组，或任意任务头对已查询位置的直接使用。

### 不自动继承旧全局投影的K1结论

在logit eta=0，group-RB归一化一阶score为
\[
 (n_A/m-1/2)\mathbf1_A+(n_B/m-1/2)\mathbf1_B.
\]
它落在两个组常数向量的span，而非单一全局常数方向。因此旧投影
$I-\mathbf1\mathbf1^\top/d$ 不会消去组间对比方向。只有同时湮灭两个组常数向量的协方差方向，才直接满足“逐粒子一阶项为零”。本包不据此宣称新的全部方向K1修复、最优K或训练收敛。

## 8. 结束边界

已闭合：异质独立channel继承原BL系数；两类型CLT与新beta_*；TC非消失；单总count同Q身份的精确失败与固定查询反例；group-count修正的同Q身份。

未开展：新科学数值、旧TC/latent信息数字的迁移、非交换惩罚的最优渐近常数、任意不同尺度的近似聚类RB、一般网络、完整局部谱门或实用效率。若所有尺度都不同且没有已知非平凡对称群，group-RB可退化到恒等操作，不能将本两组构造宣传成通用非交换修复。
