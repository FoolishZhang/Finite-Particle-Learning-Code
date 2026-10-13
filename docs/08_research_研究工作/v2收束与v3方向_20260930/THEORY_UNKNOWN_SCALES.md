# 未知两尺度时的统一总joint KL扩展

这是新的解析扩展，文件独立于旧SAMPLE_VS_PARTICLE_SEPARATION (historical reference not distributed)。旧E、A/B/C及其冻结均不改。本文0新科学数值、0样本、0拟合；不把已有固定网格程序的有限实验当作下面连续估计器的证明。

结论可以闭合：在已知c、已知紧区间[a,b]、已知正分离下界delta及已知组大小下，先用独立样本恢复组，再用另一批样本估计两个尺度，可得到统一的完整向量KL风险界。两个样本批次都取O(d log d)时风险趋零，但本证明只给O(1/log d)，不能直接延续旧已知尺度的O(1/d)风险率。要在这里保证O(1/d)，第二批取O(d²)即可。

全部log为自然对数，所有KL都是完整d维二值向量的总nats，**没有除以d**。

## 1. 已知条件、参数类与记号

固定已知常数

\[
0<c<1,\qquad 0<a<b<\infty,\qquad 0<\delta\le b-a.
\]

令d=2m为偶数，d>=4，

\[
\rho=\frac{c}{d-1},\qquad
R=(1+\rho)I-\rho\mathbf1\mathbf1^\top.
\]

未知参数为

\[
\theta=(H,s_H,s_L),\quad |H|=m,\quad
a\le s_L<s_H\le b,\quad s_H-s_L\ge\delta.
\]

D_theta为把s_H放在H、s_L放在其余坐标的对角矩阵。模型为

\[
U\sim N(0,R),\quad \eta=D_\theta U,\quad
C_i\mid\eta\stackrel{\rm ind}{\sim}
\operatorname{Ber}(\operatorname{sigmoid}(\eta_i)).
\]

其观察分布为P_theta，Y_i=2C_i−1。各一维观察边际恒为1/2，但未知组及尺度仍影响联合分布。训练观察按完整向量独立；一个向量中的坐标和pair都不假设独立。

对相关为正rho的二元标准Gaussian(X,Y)，定义

\[
f_s(z)=\tanh(sz/2),\qquad
r_d(s)=E[f_s(X)f_s(Y)],\qquad
a_1(s)=E[Zf_s(Z)].
\]

原负相关模型同一真组内的spin pair相关为−r_d(s)，故组内总spin S_g满足

\[
E_\theta S_g^2=m-m(m-1)r_d(s_g).
\tag{1}
\]

为避免混淆，下面明确区分

\[
\partial_s f_s(z)=\frac z2\operatorname{sech}^2(sz/2),
\qquad
\partial_z f_s(z)=\frac s2\operatorname{sech}^2(sz/2).
\tag{2}
\]

前者用于尺度方向的Hermite导数，L²范数至多1/2；后者用于Gaussian空间梯度，其绝对值至多b/2。不能交换这两个界。

## 2. split-sample算法

取两批互相独立的数据，样本数分别为正整数n1、n2。

第一批运行ROW：

\[
\widehat b_i=-\frac1{n_1}\sum_{n=1}^{n_1}
Y_i^{(n)}\sum_{j\ne i}Y_j^{(n)}.
\]

选择m个最大值为Hhat，平局用预定规则。第二批不参与这个选择。对估计组g=Hhat及其补集，计算

\[
\widehat r_g=
\frac{m-n_2^{-1}\sum_{n=1}^{n_2}
(\sum_{i\in g}Y_i^{(n)})^2}{m(m-1)}.
\tag{3}
\]

原始rhat可能超出模型可达范围，所以算法必须先在相关空间投影：

\[
\widetilde r_g=
\Pi_{[r_d(a),r_d(b)]}(\widehat r_g),
\qquad
\widehat s_g=r_d^{-1}(\widetilde r_g).
\tag{4}
\]

即**先clip到可达区间，再作连续单调反演**；不是对一个可能不存在的inverse先求值再clip。第4节证明在定理所用维数条件下，r_d在[a,b]上严格递增，故该定义唯一且始终输出[a,b]内尺度。

预测分布为

\[
\widehat P=P_{\widehat H,\widehat s_{\widehat H},
                              \widehat s_{\widehat H^c}}
\]

所定义的完整Gaussian-logistic观察模型。这里不再用有限K product mixture替代它。两个估计尺度在有限样本中不必保持真实大小顺序或delta间隔；本算法不静默重新标记坐标，风险证明只需要二者均被限制在[a,b]。

样本分割使“ROW组恢复正确”事件只依赖第一批数据。条件于第一批和该事件，第二批仍是原模型的iid观察，所用组就是固定真组。这避免把数据驱动选组后的组内矩当作未经选择的样本矩。

## 3. 明确常数与定理

记

\[
h=a_1(a)\,
 E[Z^2\operatorname{sech}^2(bZ/2)]>0,
\qquad
\kappa_0=\frac{ch\delta}{6},
\]

\[
\Lambda=\frac{b^2(1+c/3)}4,\qquad
C_4=32+40\Lambda^2,
\]

\[
C_G=\frac{1+(b/a)^2+4c^2/[9(1-c)]}{2a^2},
\qquad
A=\frac{36C_GC_4}{c^2h^2},
\qquad B=C_G(b-a)^2.
\tag{5}
\]

这些常数只依赖已知的c、a、b、delta，不依赖未知H、未知真实尺度或d。要求d为充分大的偶数，使

\[
\rho^2\le\frac h2,\qquad
\rho^2\le\frac{h\delta}{2}.
\tag{6}
\]

对固定上述参数，(6)对所有充分大偶数d成立；本文不未经检查就把它自动套给某个有限实验维数。

**定理。** 对(6)成立的d和任意n1,n2>=1，上述算法满足

\[
\boxed{\displaystyle
\sup_{\theta}
E_\theta D(P_\theta\Vert\widehat P)
\le A\frac d{n_2}
+\frac{Bd^3}{4}
 \exp\!\left[-\frac{n_1\kappa_0^2}{9d}\right].}
\tag{7}
\]

sup覆盖第1节的整个参数类；期望同时覆盖两批训练数据。以下逐项证明，包括尺度反演稳定性、四阶矩与Gaussian KL转换。

## 4. 统一ROW间隔与尺度反演导数

首先，

\[
a_1'(s)=\frac12E[Z^2\operatorname{sech}^2(sZ/2)]>0.
\]

对s属于[a,b]，

\[
2a_1(s)a_1'(s)
=a_1(s)E[Z^2\operatorname{sech}^2(sZ/2)]
\ge h.
\tag{8}
\]

因此真实两个尺度的首阶Hermite能量差至少为

\[
a_1(s_H)^2-a_1(s_L)^2
=\int_{s_L}^{s_H}2a_1(s)a_1'(s)\,ds
\ge h\delta.
\tag{9}
\]

### 4.1 ROW组恢复界

旧E已证明：若该能量差至少为Delta0、rho²<=Delta0/2，则ROW人口行间隔至少为c Delta0/6，且

\[
\Pr_\theta(\widehat H\ne H)
\le\frac{d^2}{4}
 \exp\!\left[-\frac{n_1(c\Delta_0/6)^2}{9d}\right].
\]

这里取Delta0=h delta即可。为明确适用对象，其关键量为跨真组坐标i,j的

\[
W=-(Y_i-Y_j)\sum_{k\ne i,j}Y_k.
\]

其均值是(m−1)(r_HH−r_LL)，至少为kappa0；负pair相关给EW²<=4(d−2)，且|W|<=2(d−2)。中心化Bernstein分母至多9d，对m²个高低比较作union bound，不要求同一向量中的比较独立。因此

\[
\Pr_\theta(\widehat H\ne H)
\le\frac{d^2}{4}e^{-n_1\kappa_0^2/(9d)}.
\tag{10}
\]

此界对整个紧参数类统一成立。详细Bernstein MGF及常数推导保留在旧E第2–4节，没有改写旧文件。

### 4.2 r_d的尺度导数不能只写成首阶近似

以规范化概率论Hermite基h_l展开f_s，系数为a_l(s)。f_s为奇函数，故

\[
r_d(s)=\rho a_1(s)^2+
 \sum_{\ell\ge3,\ \ell\ {\rm odd}}\rho^\ell a_\ell(s)^2.
\]

L²意义下s到f_s为连续可微映射：由(2)，
\(|\partial_s f_s(z)|\le|z|/2\)，Gaussian二阶矩可积。因此可在L²中对Mehler二次型求导；尾部算子的范数至多rho³。Parseval给

\[
\|f_s\|_2\le1,\qquad
\|\partial_s f_s\|_2\le\frac12.
\]

于是尾部导数的绝对值不超过

\[
2\rho^3\|f_{s,\ge3}\|_2
          \|\partial_s f_{s,\ge3}\|_2
\le\rho^3.
\]

由(8)和(6)，

\[
\boxed{\displaystyle
r_d'(s)\ge\rho h-\rho^3\ge\frac{\rho h}{2},
\qquad s\in[a,b].}
\tag{11}
\]

这是对整个紧区间的实际导数下界，不是忽略余项后的近似式。连续反演因此满足

\[
|r_d^{-1}(x)-r_d^{-1}(y)|
\le\frac{2}{\rho h}|x-y|,
\quad x,y\in[r_d(a),r_d(b)].
\tag{12}
\]

## 5. Gaussian Poincaré与组内四阶矩

本节仅使用基本Gaussian谱gap，不依赖完整Gaussian集中定理。

**Poincaré引理。** 若Z为标准多元Gaussian，F及其弱梯度平方可积，则

\[
\operatorname{Var}(F(Z))\le E\|\nabla F(Z)\|^2.
\tag{13}
\]

可由多元规范化Hermite展开自足说明：若
\(F=\sum_\alpha c_\alpha h_\alpha\)，则
\(\operatorname{Var}F=\sum_{|\alpha|\ge1}c_\alpha^2\)，而
\(E\|\nabla F\|^2=\sum_\alpha|\alpha|c_\alpha^2\)。
先对多项式成立，再由Gaussian Sobolev逼近推广。下面的F和F²都是有界光滑函数，梯度平方可积，因此适用条件满足。

给定一个真组g、其真实尺度s，写

\[
S_g=\sum_{i\in g}Y_i=F_g(U)+\Xi_g,
\]

\[
F_g(U)=\sum_{i\in g}f_s(U_i),\qquad
\Xi_g=\sum_{i\in g}(Y_i-f_s(U_i)).
\]

### 5.1 条件Bernoulli噪声

给定U，各噪声项独立、中心、绝对值至多2，方差至多1，四阶矩至多4。独立中心变量的四阶矩展开给

\[
E(\Xi_g^4\mid U)
\le4m+6\binom m2
=3m^2+m\le4m^2.
\tag{14}
\]

这里仅使用给定latent后的发射条件独立，不假设观察坐标在边际上独立。

### 5.2 latent条件均值

写U=R^{1/2}Z。F_g的均值为0，由空间导数界
\(|\partial_z f_s|\le b/2\)及lambda_max(R)=1+rho<=1+c/3，

\[
\|\nabla_ZF_g\|^2
\le\frac{b^2m}{4}(1+c/3)=\Lambda m=:L^2.
\]

第一次应用(13)得\(EF_g^2\le L^2\)。再对F_g²应用一次：

\[
\operatorname{Var}(F_g^2)
\le E\|2F_g\nabla F_g\|^2
\le4L^2EF_g^2\le4L^4.
\]

因此

\[
EF_g^4
=\operatorname{Var}(F_g^2)+(EF_g^2)^2
\le5L^4=5\Lambda^2m^2.
\tag{15}
\]

由实数不等式(x+y)^4<=8(x^4+y^4)，(14)–(15)给出

\[
\boxed{ES_g^4\le(32+40\Lambda^2)m^2=C_4m^2.}
\tag{16}
\]

该界对所有真实尺度[a,b]、所有H及所考虑d统一成立。没有使用错误的“pair独立”或把S_g的最坏范围m直接平方成松的m⁴阶控制。

## 6. 两个尺度各自的均方误差是O(1/n2)

在组已正确给出的条件下，(1)使(3)对r_d(s_g)无偏。由独立训练向量及(16)，

\[
E(\widehat r_g-r_d(s_g))^2
=\frac{\operatorname{Var}(S_g^2)}
 {n_2m^2(m-1)^2}
\le\frac{C_4}{n_2(m-1)^2}.
\tag{17}
\]

真实r_d(s_g)位于clip区间，投影不增大其距离。再用(12)，

\[
E(\widehat s_g-s_g)^2
\le\frac{4C_4}
 {n_2\rho^2h^2(m-1)^2}.
\]

d>=4时，

\[
(m-1)\rho=\frac{c(d-2)}{2(d-1)}\ge\frac c3.
\]

所以每个尺度均满足

\[
\boxed{\displaystyle
E(\widehat s_g-s_g)^2
\le\frac{36C_4}{c^2h^2n_2}.}
\tag{18}
\]

这里的重要抵消是：反演导数随d增长到O(d)，但组内二阶矩的相关估计方差有O(d^{-2}/n2)因子。不能仅凭参数可识别性或一致性跳过这两个维数因素。

两组的估计无需互相独立，后面只将两个均方误差的期望相加。由于样本分割，(18)在给定第一批数据且ROW正确时仍成立。

## 7. 不同尺度的Gaussian KL：精确ratio公式

对任意真实坐标尺度s_i及估计坐标尺度hat_s_i，都属于[a,b]。不论分组是否恢复正确，置

\[
e_i=\frac{s_i}{\widehat s_i},\qquad
q_d=\frac{\rho^2}{(1+\rho)(1-c)}.
\]

真实/预测logit协方差为
\(\Sigma=DRD\)、\(\widehat\Sigma=\widehat D R\widehat D\)。
由

\[
R^{-1}=\frac1{1+\rho}I+
 \frac{\rho}{(1+\rho)(1-c)}\mathbf1\mathbf1^\top
\]

可直接相乘并用trace循环性得到

\[
\operatorname{tr}(\widehat\Sigma^{-1}\Sigma)
=\sum_i e_i^2+
q_d\left[d\sum_i e_i^2-\left(\sum_i e_i\right)^2\right].
\]

而log determinant比为\(2\sum_i\log e_i\)。故精确Gaussian KL为

\[
\boxed{\begin{aligned}
D(N(0,\Sigma)\Vert N(0,\widehat\Sigma))
={}&\frac12\sum_i(e_i^2-1-2\log e_i)\\
&+\frac{q_d}{2}
 \left[d\sum_i e_i^2-\left(\sum_i e_i\right)^2\right].
\end{aligned}}
\tag{19}
\]

未知尺度时determinant不再相等，不能直接复用旧E的det相消论证；上式保留了其必要的log项。

对e属于[a/b,b/a]，函数phi(e)=e²−1−2log e在1处函数和导数均为0，且

\[
\phi''(e)\le2[1+(b/a)^2].
\]

又

\[
d\sum_i e_i^2-(\sum_i e_i)^2
=d\sum_i(e_i-1)^2-(\sum_i(e_i-1))^2
\le d\sum_i(e_i-1)^2,
\]

\[
q_dd
=\frac{c^2d}{(d-1)^2(1+\rho)(1-c)}
\le\frac{4c^2}{9(1-c)}.
\]

结合\(|e_i-1|\le|s_i-\widehat s_i|/a\)，(19)给

\[
D(N(0,\Sigma)\Vert N(0,\widehat\Sigma))
\le C_G\sum_i(s_i-\widehat s_i)^2.
\tag{20}
\]

在logits之后使用相同的逐坐标logistic-Bernoulli通道，DPI因此给

\[
\boxed{\displaystyle
D(P_\theta\Vert\widehat P)
\le C_G\sum_i(s_i-\widehat s_i)^2.}
\tag{21}
\]

共同通道作用于eta，而不是把不同尺度下U到C的映射误当作同一个channel。KL方向始终为真实分布到预测分布。

## 8. 正确组与错误组两种事件的完整风险分解

令E为第一批数据恢复组正确的事件。给定第一批数据且E成立，第二批仍独立。此时

\[
\sum_i(s_i-\widehat s_i)^2
=m[(s_H-\widehat s_H)^2+(s_L-\widehat s_L)^2].
\]

由(18)、(21)，正确组部分的期望风险至多

\[
C_Gm\frac{72C_4}{c^2h^2n_2}
=\frac{36C_GC_4}{c^2h^2}\frac d{n_2}
=A\frac d{n_2}.
\tag{22}
\]

在E的补集上，不声称(18)仍以真实组尺度为中心。但clip保证所有坐标尺度都在[a,b]，所以逐个训练输出都有

\[
D(P_\theta\Vert\widehat P)
\le C_Gd(b-a)^2=Bd.
\tag{23}
\]

因此

\[
\begin{aligned}
E_\theta D(P_\theta\Vert\widehat P)
&=E_\theta[\mathbf1_E D]+
  E_\theta[\mathbf1_{E^c}D]\\
&\le A\frac d{n_2}
 +Bd\,\Pr_\theta(E^c),
\end{aligned}
\]

再代入(10)得到(7)。所有常数及维数条件均统一于第1节参数类，因而可以取sup_theta。

## 9. 两种样本配置给不同的风险率

第一种配置取

\[
n_1=\left\lceil
\frac{9d}{\kappa_0^2}
 [4\log d+\log(1+B)]
\right\rceil,\qquad
n_2=\lceil d\log d\rceil.
\]

则

\[
\sup_\theta E_\theta D(P_\theta\Vert\widehat P)
\le\frac{A}{\log d}+\frac1{4d}\longrightarrow0.
\tag{24}
\]

总样本数仍是O(d log d)，但本证明只给O(1/log d)总KL率。它没有证明这是最优率，也没有把该结论偷换成旧E的O(1/d)。

若目标是恢复同样的1/(4d)上界，可改为

\[
n_1=\left\lceil
\frac{9d}{\kappa_0^2}
 [4\log d+\log(2(1+B))]
\right\rceil,\qquad
n_2=\lceil8Ad^2\rceil.
\]

此时正确组与错误组两项各至多1/(8d)，从而

\[
\boxed{\displaystyle
\sup_\theta E_\theta D(P_\theta\Vert\widehat P)
\le\frac1{4d},\qquad n_1+n_2=O(d^2).}
\tag{25}
\]

样本阶数中的常数允许依赖固定c、a、b、delta；尺度接近、a趋零、b增大或c趋边界时，本文不作统一常数承诺。

## 10. 数值反演、旧网格及计算范围

(4)是连续、精确函数层面的估计器。假如实现给出的尺度在clip后与理想连续inverse最多差epsilon_inv，则由平方误差不等式，同样推导可得

\[
\sup_\theta E D(P_\theta\Vert\widehat P_{\rm num})
\le 2A\frac d{n_2}
 +2C_Gd\,\varepsilon_{\rm inv}^2
 +\frac{Bd^3}{4}e^{-n_1\kappa_0^2/(9d)}.
\tag{26}
\]

因此数值误差也需要进入总KL预算。固定不缩小的尺度网格不能直接套用本连续估计器的统一渐近证明；要将其实现为离散近似，必须控制网格或反演误差，使相应d epsilon_inv²项不破坏目标率。已有C的0,.5,...,9网格及固定真实尺度有限实验没有提供这个保证。

ROW和第二批组内S_g²的数据处理可用O((n1+n2)d)操作、O(d)工作空间完成，不需要d²个pair矩阵。之后只有两个已知函数的标量反演；本文没有分析这些反演的数值位复杂度，也没有把参数学习成本等同于任意joint likelihood或后验查询成本。

输出是紧凑的Gaussian-logistic模型，不是列举所有2^d模式，也不是有限K乘积分量的部署近似。若额外限制最终预测器为K-product mixture，旧统一观察KL下界仍会作用于这一额外表示限制。

## 11. 对v2收束与v3的含义

这个扩展移除了旧E“真实两个尺度值已知”的条件；保留的是已知c、已知组大小、固定紧区间及固定正分离下界。它不是未知c、未知组数、无分离、一般异质协方差或自然数据的定理。

对任意固定合法尺度对，旧TC非消失与有限K product-mixture下界可同本定理并列，继续得到“可由带标记数据学习的紧凑模型”与“某类有限乘积近似”之间的分离。这里不从旧逐参数CLT未经证明地跳到整个紧参数类的uniform CLT；建立分离用一个固定合法子族就已经足够。

样本split是当前证明的重要组成部分。若同一数据同时选组又反演尺度，不能再直接对选中的组调用(17)–(18)，需要新的选择后或统一控制。若要研究未知c，也不能直接复用本反演函数和Gaussian同R公式。这些是后续独立问题，不是本文件已经解决的内容。

本推导使用的Gaussian Poincaré、Mehler展开、Bernstein和KL数据处理均为标准工具；新增内容是该指定参数类下，维数因素、两个样本阶段与总joint KL被明确地连起来。没有新增科学数值，也没有将解析结果追认为任何旧实验主门。
