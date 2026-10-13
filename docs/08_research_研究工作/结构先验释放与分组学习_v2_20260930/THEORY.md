# 结构先验释放：识别、count盲区与错误分组下的pair合并

本文件是5cfbd418后的新理论包，不改旧v1/v2。模型仍为固定负等相关Gaussian加独立logistic二值channel，但释放两个组尺度，允许未知方向，并考察给错分组。以下不报告尚未运行的新数值。KL链式、Jensen–Shannon与错设复合似然的一般原理不作为新通用定理；增量是它们在本明确结构族中的可核验联系。

## 1. 两个尺度可由标记pair识别，单边际仍无信息

设d=2m>=4，0<c<1，rho=c/(d−1)，R=(1+rho)I−rho 11ᵀ。给定两组A/B，各m个坐标，

\[
U\sim N(0,R),\quad
C_i\mid U\ \text{独立},\quad
\Pr(C_i=1\mid U)=\operatorname{sigmoid}(s_{g(i)}U_i).
\]

参数为两个有限非负尺度(sA,sB)，真值在正尺度域。Y_i=2C_i−1。所有单边际仍为1/2；零均值设置下单概念边际目标无法识别任一尺度。

对正相关rho的标准Gaussian(X,Y)，记

\[
r(s,t)=E[\tanh(sX/2)\tanh(tY/2)].
\]

当另一尺度为正时，r分别对s/t严格递增。证明见旧非交换THEORY，核心是E[tanh(tY/2)|X=x]与x严格同号，而对s的导数为(x/2)sech²(sx/2)；可积控制允许交换求导。对角函数r(s,s)从0严格递增，因此观察pair矩

\[
E(Y_iY_j)=
-r(s_A,s_A),\ -r(s_A,s_B),\ -r(s_B,s_B)
\]

按AA/AB/BB分别取值。两个组各至少一对，AA确定sA、BB确定sB；这是**带已知坐标组标记**的全局识别，不只是未知次序集合的识别。完整joint相等必有pair相等，故joint也识别。

全部pair平均目标的人口超额值为各pair KL的平均。正确分组、固定c、候选包含真值时，它和J都唯一在真参数最小。该结论不等于有限样本效率相同，也不允许把一个向量内的所有pair当作独立样本。若c也未知，旧首阶Hermite退化问题仍存在，本包不放开c。

## 2. 总count不能识别尺度方向：任意训练样本数的下界

令P+=P_{sA,sB}，P−=P_{sB,sA}，sA≠sB，A/B的坐标标签固定。交换两组坐标的置换把P+变成P−，却保持N=ΣC_i。因此

\[
(P_+)_N=(P_-)_N,\qquad P_+\ne P_-.
\]

这不是无关紧要的latent标签交换：两个分布作用于相同的、有语义标记的观察坐标，其AA与BB的pair矩不同。若事先强制sA>=sB，就重新加入了方向先验，不能把由此得到的朝向归功于count数据。

设算法只读取任意n个训练total counts及与教师假设独立的随机种子，输出任意观察分布Qhat。它在P+/P−两世界下的输出分布完全相同。令M=(P++P−)/2，则对每个固定输出Q，

\[
\frac12D(P_+\Vert Q)+\frac12D(P_-\Vert Q)
=\operatorname{JS}(P_+,P_-)+D(M\Vert Q).
\]

若Q在正质量点为0，右侧/左侧允许无穷，不影响结论。对共同输出分布取期望，得到

\[
\boxed{
\max_{\nu\in\{+,-\}}E_\nu D(P_\nu\Vert\widehat Q)
\ge \operatorname{JS}(P_+,P_-)>0.}
\]

该下界对任意n、任意算法和任意输出模型类成立；固定错误tie-break不是其原因。算法若用带坐标标签的pair信息选择方向，便不再属于count-only程序。这里也没有证明count在连续二维域中“恰好只有交换这一个歧义”；本包只使用已证明的至少两点不可识别。

### 非消失事件桥

采用旧MISMATCH_THEORY中的统计量

\[
T_d=\frac{\sum_A Y_i+\sum_B Y_i}{\sqrt d},\qquad
V_d=\frac{\sum_A Y_i-\sum_B Y_i}{\sqrt d},
\]

零值各用独立公平符号处理，令E为二者同号。交换A/B使T不变、V取反，故有限维就有P−(E)=1−P+(E)。在M下事件概率恰为1/2。对两项KL使用同一个事件通道，得到

\[
\operatorname{JS}(P_+,P_-)
\ge \operatorname{kl}(P_+(E)\Vert1/2).
\]

固定不同正尺度、固定c沿偶数d增长时，旧两类型联合CLT给

\[
P_+(E)\to \frac12+\frac{\arcsin\zeta}{\pi},\qquad
\zeta=-\frac{c a_+a_-}
 {\sqrt{(1-ca_+^2)(1-ca_-^2)}}\ne0,
\]

其中a(s)=E[Z tanh(sZ/2)]，a_+=(a(sA)+a(sB))/2，a_-=(a(sA)−a(sB))/2。于是上述count-only最坏joint风险的liminf严格为正，哪怕训练数量n=n(d)任意增长。

这不声称固定数量的带标记pair样本就足以在增长维数下恢复尺度；J/P的识别及通常一致性首先是固定d结论。也不把JS、DPI或两点不可识别本身包装成新一般方法。

## 3. 给错分组或比例时，三个精确目标在拟合什么

令G为候选假定分区内置换群，Q_theta^G为该假定分组的候选模型。它在G下不变。令P_G为真实P的G群平均，则

\[
\boxed{D(P\Vert Q_\theta^G)
=D(P\Vert P_G)+D(P_G\Vert Q_\theta^G).}
\]

因为log(P_G/Q)在群作用下不变，其P期望等于P_G期望。对二组count，P_G在每个假定count轨道内均匀，所以第二项等于假定双count分布的KL。第一项是结构先验错误造成的不可去除下界。

如果某个假定组同时包含一个真A坐标i和真B坐标j，取不同的第三坐标k。严格单调性使真实E YiYk与E YjYk不同，而组内置换i/j不变的任何Q必须使二者相同。因此不同真尺度下P不是G不变的，D(P||P_G)>0。整组A/B对换只是一种重新命名；当两个尺度均自由时，它不改变候选模型类，不能当作结构错设正例。

J最小化完整joint KL，即上述投影项。P最小化各标记pair KL的平均，在错设下不保证与J的伪真点相同。T最小化总count KL；对相同大小的候选两组，其总count分布只依赖尺度multiset，与哪些坐标被分到该组无关。因而T对结构误置也完全盲。

T在本包只用于估计参数，随后部署Q_theta^G，**不是旧single的S(Q)**。其风险分账是

\[
D(P\Vert Q_\theta^G)
=D(P_N\Vert Q_{\theta,N})
+E_{P_N}D(P(C\mid N)\Vert Q_\theta^G(C\mid N)).
\]

count项为零也不保证条件分配正确。只有真正部署完整对称化S(Q)时，才能使用旧D_sym(P)+count-KL公式；不能将这两份合同混用。

若固定了错误尺度比例但分组正确，J/P/T各自在错误射线类中取其自己的投影；不存在一般理由使三者伪真点重合。释放两个尺度是本轮用来检验先前成功依赖多少结构先验的必要变化。

### 不需要新高维积分的错误分组教师权重

从真A/B各交换r个坐标。条件于真count(N_A=a,N_B=b)，被交换部分的成功数

\[
X_A\sim\operatorname{Hypergeom}(m,a,r),\quad
X_B\sim\operatorname{Hypergeom}(m,b,r)
\]

相互独立；这是给定真count后的两个组均匀分配身份，不要求潜变量组独立。假定count为

\[
N'_A=a-X_A+X_B,\quad N'_B=b-X_B+X_A.
\]

由这个有限Markov核可把原81格教师PMF变换成每个错误分组的81格PMF；它保持总count。其L1误差不会放大原PMF的L1误差。A包另用完整pattern整数hist进行同样确定映射，使各结构读到的是同一批观察，不能将四个r当独立样本。

候选Q及G粒子估计器的结构始终按假定分组定义。重加权错误分组教师count并不等于把候选q错误平均成另一个模型。

## 4. 50%混组时，精确pair人口目标必合并两个尺度

令m为偶数，假定每个组都含m/2真A及m/2真B，即r=m/2。本包m=8、r=4。记真实正幅度

\[
r_A=r(s_A,s_A),\quad r_B=r(s_B,s_B),\quad r_{AB}=r(s_A,s_B).
\]

Mehler/Hermite表示给

\[
D_r=r_A+r_B-2r_{AB}
=\sum_{\ell\ge1,\ \ell\ {\rm odd}}\rho^\ell
 (a_\ell(s_A)-a_\ell(s_B))^2>0.
\]

严格性可由一阶Hermite系数a_1(s)=E[Z tanh(sZ/2)]严格递增得到。错误组内的两类平均pair幅度相同，记alpha；错误组间平均幅度记beta。计数得到

\[
\alpha=\frac{(m-2)(r_A+r_B)+2m r_{AB}}{4(m-1)},\qquad
\beta=\frac{r_A+r_B+2r_{AB}}4,\qquad
\beta-\alpha=\frac{D_r}{4(m-1)}>0.
\]

虽然同一错误组内的真实各pair不完全相同，但候选给它们同一个概率；cross-entropy对真实概率线性，所以以上平均值精确描述人口pair目标。

对候选(u,v)，令x=r(u,u)、y=r(v,v)、z=r(u,v)。Hermite正定核给

\[
x+y\ge2z.
\]

pair负log的标量cross-entropy（真实负幅度a、候选负幅度x）为

\[
\ell_a(x)=\log4-\frac{1-a}{2}\log(1-x)
                     -\frac{1+a}{2}\log(1+x),\qquad
\ell'_a(x)=\frac{x-a}{1-x^2}.
\]

它在(-1,1)严格凸。取w=choose(m,2)，v_0=m²，人口目标乘以总pair数后为

\[
w\ell_\alpha(x)+w\ell_\alpha(y)+v_0\ell_\beta(z).
\]

将真正可达集合放宽到半空间x+y>=2z。由严格凸性和x/y对称，唯一最优有x=y；再写a=x=y，约束a>=z。无约束最优(alpha,beta)违反alpha<beta，故约束激活，a=z=rbar，其中

\[
\bar r=\frac{2w\alpha+v_0\beta}{2w+v_0}
=\frac{\binom m2(r_A+r_B)+m^2 r_{AB}}{\binom{2m}2}.
\]

这个rbar就是全部真实pair的平均幅度。它严格位于r(sB,sB)与r(sA,sA)之间（若sA>sB）。对角r(w_*,w_*)严格单调，故存在唯一w_*∈(sB,sA)使其等于rbar；候选u=v=w_*达到半空间下界。于是

\[
\boxed{\arg\min_{u,v\ge0} L_{\rm pair}^{r=m/2}(u,v)=\{(w_*,w_*)\}.}
\]

这是无限数据、精确pair目标在结构错设下的尺度合并；不是MC噪声或优化失败。它不证明J在同样错误分组下也必合并，后者是新比较对象。

### 连续解不等于离散网格解

上述结论针对连续参数域。若w_*不在0,.5,...,9中，不能保证19×19网格最优点恰为对角；非对角候选可能更接近连续最优的相关幅度。B如实记录该现象，不将其作为定理失败或新的成功门。

B在固定[0,9]上另求r(w,w)=rbar的唯一root，并对(w,w)计算一份qualified完整参考；它不加入361拟合网格。因为所有候选pair满足半空间约束，连续oracle的pair-loss同时是严格解析下界，可报告网格gap。此处额外的是1个确定性population解与1份诊断参考，不是新增教师样本或拟合重复。

## 5. B学习与推断范围

B对同一A数据的r=0,1,2,4映射，各用J/P/T/G10/G100在完整361点网格学习。T数值共享镜像total PMF，并报告完整最小等价类及各镜像部署风险；其方向不能由微小浮点差或固定tie-break产生“成功”。

G的期望仍满足F_K=H(P)+D(P||Q)+J_K。这里结构错设下的模型KL、有限数据误差、finite-bank积分偏差和noise argmin选择效应是四个不同对象。高精度J的当前网格最小KL作为可核验基线，不等于连续空间最优。已有旧共同幅度结果不能被当作本二维问题的通过结果。

16个独立数据/粒子流是统计单元，四r和两个N为配对读取，所有grid候选和K前缀共用bank。32项有限数据区间与8项固定总体比较按各自冻结家族的t近似解释；没有把361个候选当独立样本，也没有为noise argmin声称总体最优置信证书。

本轮可以推进的主张是：释放方向先验后，粗统计存在与样本数无关的识别障碍；给错分组后，精确低阶监督本身也可有可证明的伪真结构；正确候选模型的有限粒子目标另有计算偏差。它不说明pair一般无效、总count所有参数均不可识别，或已获得自然数据上的通用替代训练方法。
