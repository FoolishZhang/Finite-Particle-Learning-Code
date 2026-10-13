---
title: "有限先验粒子似然目标可能偏好提高风险的依赖"
lang: zh-CN
---

# 摘要

潜变量模型常这样拟合：在独立的先验抽样上取$K$个条件似然的平均来估计边际似然，再最小化这一估计的期望负对数。固定$K$时，它是一个独立的准则，而不是对数似然的带噪版本：把它估计得更准，并不会把它变成对数似然。我们问它偏好哪些依赖结构，这些偏好是否与积分模型的KL风险一致。在一个Gaussian--logistic模型中，我们追加$n$个被观察的公平硬币。积分之后，它们不改变任何候选的风险；在粒子平均内部，它们各粒子的似然却改变了哪些粒子占主导。在活跃Gaussian边际固定、$K\ge2$至多按$n$的多项式增长时，随$n$增大，我们对实际的非Gaussian权重一致地找出首个区分相关结构的项。它是一个严格凹势，奖励活跃对数似然在先验抽样间更大的方差。能够分辨这一收缩项的总体近极小点趋向相关矩阵类的谱边界，并保持正风险。在没有追加块、全部参数自由的情形，计算机辅助证明给出一个8粒子总体局部极小：其单变量边际全部正确，联合KL却约为$0.0104$；从该点出发，沿一条保持边际的路径，每个充分小的非零步都降低风险、抬高目标。因此，准确拟合有限目标，或正确拟合边际，都不能决定学到的是哪种依赖。

# 引言

许多潜变量模型的边际似然$p_\theta(y)=E_{Z}\,p_\theta(y\mid Z)$难以计算。一种简单的替代是抽取$K$个独立先验粒子$Z_1,\ldots,Z_K$，最小化其平均似然的期望负对数。它出现在模拟极大似然[@lee1995]、以先验抽样学习二元潜变量模型[@tipping1998]，以及提议分布固定为先验时的重要性加权界[@burda2016]中。由于对数是凹函数，这一目标有偏，且偏差依赖参数[@maddison2017; @sung2007]。因此固定$K$时，它是不同于似然的另一个准则。增加外层重复只会更准确地估计同一个有限$K$目标，不会把它变成积分对数似然。

一般的偏差结论留下了一个问题，这正是本文要回答的：*有限先验粒子目标偏好哪种依赖结构？这种偏好何时与积分模型$Q_\theta$的KL风险$D(P\Vert Q_\theta)$相冲突？*

![公平硬币对照。三个二元变量的潜在logit相互独立（$C_*=I$），每个对数内用$K=100$个粒子。(a)真值与等相关候选$C_b$之间的总体目标差$g_n=F_n(I)-F_n(C_b)$；正值表示有限目标偏好$C_b$。(b)目标在真值处的曲率$h_n$，三个自由相关上$\nabla^2F_n(I)=h_nI_3$；负值使真值成为局部极大。$n=256$时，追加公平硬币的各粒子似然留在粒子平均内部；活跃观察、活跃粒子和候选与$n=0$共享。误差条为各面板在理想iid抽样和声明数值资格下的99%同时区间（附录C.2、C.4）；$h_0$为正另有解析证明（C.3）。图下注报告一次匹配的143点网格选择（C.1）。](figures/finite_instance.png){width=100%}

**一个受控比较。** 取三个二元观察，其潜在Gaussian logit相互独立，并使用$K=100$个粒子。再追加$n$个被观察的二元变量，它们是与其他一切独立的公平硬币。每个硬币有自己的潜在logit $G_j\sim N(0,1)$，结果$u_j\in\{-1,1\}$的发射概率为$\sigma(u_jG_j)$。在先验上平均后，每个硬币在每个候选下的概率都是$1/2$。因此每个候选的积分模型等于原模型乘以$2^{-n}$，所有风险都不变。在粒子平均内部，硬币的作用却不同。粒子$k$带有因子$\prod_j\sigma(u_jG_{kj})$，其先验均值虽为$2^{-n}$，取值却随粒子而变。这些因子决定哪些粒子主导平均，从而决定活跃似然如何竞争。若在平均*之前*先把硬币积掉，它们只贡献共同常数$2^{-n}$，什么也不改变。

图1在相同的活跃观察、活跃粒子和候选上比较$n=0$与$n=256$。没有硬币时，有限目标偏好真实的独立模型，而不是所有相关均为$.9$的等相关候选$C_b$，真值是严格局部极小。加入256个硬币后偏好反转：$C_b$得分更好，真值变成局部极大，匹配的网格搜索返回$C_b$，其风险为$.031$ nats。

我们把这一构造看作对照，而不是推荐的做法。追加块已知独立，其似然精确可分解。它的价值在于：改变估计器，同时保持活跃似然比和每个候选的KL风险。因此图1中发生的变化，都是这一准则自身的偏好。

**为什么需要精细尺度。** 随$n$增大，干扰对数权重在$\sqrt n$尺度上展开，平均由干扰权重最大的粒子主导。活跃边际固定时，一个先验粒子的期望损失不依赖候选相关矩阵，因此目标的主导项完全看不到依赖。直观地说，在这一渐近区间内，领先的选择项来自两个领先粒子近乎并列的事件，而这类事件的频率为$w_{n,K}\asymp\sqrt{\log K/n}$阶。于是选择发生在随$n$收缩的尺度上，这带来两个后果。把目标近似到$o(1)$的方法，可能对选中哪种相关毫无判断力；近似必须相对$w_{n,K}$准确。而识别这一偏好，需要对实际的非Gaussian对数似然权重、在所有候选上一致地控制近乎并列事件。对数权重的Gaussian模型可以形式地预测大$K$系数，却不提供这种控制（第6节）。

定理1提供了这种控制。所得的势严格凹，且有直接的解读：它奖励使观察的活跃对数似然在先验抽样间变化更大的相关矩阵。对真实相关为零的两个变量，精确似然把拟合的相关拉回零，极限粒子目标却把它推离零。定理2把凹性与相关矩阵的经典几何结合：能分辨$w_{n,K}$尺度的近极小点被推向该类的谱边界，其风险保持远离零。

**没有追加块时。** 硬币让机制变得可见，但留下一个自然的问题。假设没有指定的独立块，所有偏置和载荷都可学习，而有限目标的某个局部极小已经再现了每个观察边际。此时目标是否至少不会反对能降低风险的依赖变化？定理3对一个$K=8$的八条目、双因子模型给出否定回答。计算机辅助证明认证：精确总体目标在全部24个参数中、模共同旋转后，有一个秩一严格局部极小。所有边际都正确，联合KL约为$.0104$。沿一条只旋转载荷行、不改变行长的路径，每个充分小的非零步都降低风险、抬高目标。两部分因此回答同一准则上前后相继的问题：先是受控族中关于*选择*的渐近陈述，再是没有该控制时关于几何的有限、*局部*陈述。二者互不推出。

```{=latex}
\par\dimen0=\pagegoal\advance\dimen0 by -\pagetotal\ifdim\dimen0<6\baselineskip\newpage\fi
```

**贡献。**

- 对实际非Gaussian权重的一致选择尺度定理，覆盖每个固定$K\ge2$以及按$n$多项式增长的$K$，并给出极限势的条件方差解读（定理1）。
- 它的几何后果：对regret为$o(w_{n,K})$的总体近极小点，谱边界偏好与正的极限风险（定理2）。
- 有限尺度分析：带符号的固定$K$修正分离出标量重叠看不到的同粒子耦合，显式误差界在$n=4096$认证一个坐标的首项符号（第4节）。
- 一个计算机辅助证书：边际正确、风险为正的局部极小，带有同根、保持边际的冲突，并在一个小的显式教师族上延续（定理3及其推论）。

主要理论研究以先验粒子定义的总体目标；第4.3节另给一个指定有限程序的保证。这些结果都不描述随机优化器收敛到何处。

# 先验粒子目标与受控扩维

对潜变量$Z\sim p(z)$和条件似然$p_\theta(y\mid z)$，$K$粒子目标为$E_{Y\sim P}E_{Z_{1:K}}[-\log K^{-1}\sum_kp_\theta(Y\mid Z_k)]$，其中iid先验粒子独立于$Y$。$K=\infty$时它成为积分交叉熵。第一个模型只把活跃相关作为未知量。固定$m\ge2$、有限均值$\mu$及$0<\delta<1$：

$$
\begin{aligned}
\mathcal E_\delta^{(m)}
&=\{C=C^\top:\operatorname{diag}C=\mathbf1,\ C\succeq\delta I\},\\
X_A&=\mu+L_CZ,\qquad L_CL_C^\top=C,\\
X_{m+j}&=\tau_jG_j,\\
P(Y_i=y_i\mid X_i)&=\sigma(y_iX_i),\qquad y_i\in\{-1,1\}.
\end{aligned}
\tag{1}
$$

应用$L_C$之前，所有Gaussian输入均为独立标准正态；给定logit后观察条件独立。活跃方差与均值已知，活跃与干扰logit不相关。记$U$为干扰观察，并假设教师$C_*$位于相对内部，$C_*\succ\delta I$。由于每个干扰硬币在积分后都是公平的，

$$
Q_{C,\tau}^{(n)}(y,u)=2^{-n}Q_C^A(y),\qquad
\mathcal R(C)=D(Q_{C_*}^A\Vert Q_C^A).
\tag{2}
$$

这也是完整观察的KL风险。固定$n$改变干扰尺度，整个观察分布不变；改变$n$，活跃似然比和KL不变。取独立于观察的$K$个iid先验粒子$(Z_k,G_k)$，

$$
\begin{aligned}
\widehat Q_{C,\tau,K}^{(n)}(y,u)
&=K^{-1}\sum_{k=1}^Kp_{C,\tau}(y,u\mid Z_k,G_k),\\
F_{n,K}(C;\tau)&=E_{Y,U,Z,G}[-\log\widehat Q_{C,\tau,K}^{(n)}(Y,U)].
\end{aligned}
\tag{3}
$$

每次实现的混合$\widehat Q$都是归一化分布，故$F-H(P)=E_{Z,G}D(P\Vert\widehat Q)$。有限目标评价的是随机混合，风险评价的是它的平均。略去固定参数，二者之差是经典的Jensen项：

$$
\begin{aligned}
F_P(C)&=H(P)+\mathcal R(C)+\mathcal J_P(C),\\
\mathcal J_P(C)&=E_P[\log Q_C\\
&\qquad -E_{Z,G}\log\widehat Q_C]\ge0.
\end{aligned}
\tag{4}
$$

这里$P=Q_{C_*}^{(n)}$。$\mathcal J_P$非负是标准事实；对学习重要的是$\mathcal J_P$如何随$C$变化，而非负性本身并不决定这一点。这与对数评分的适当性并不矛盾，因为$F$是随机预测规则的期望评分，而不是$Q_C$的评分。另有两个事实界定了扩维做什么、不做什么。用可逆潜变量映射同时输运先验、发射和粒子，$\widehat Q$的每次实现都不变（附录A.1）；追加被观察的坐标则会改变估计器。并且$K=1$时目标只依赖固定边际，因而对$C$为常数：任何依赖选择都需要$K\ge2$。

# 有限目标偏好哪些相关

## 选择尺度

记$b_i=EY_i$，$\kappa_{ij}(c)$为潜在相关为$c$时$Y_i,Y_j$的协方差。固定边际使$b_i$与$C$无关，Price恒等式[@price1958]给出

$$
\begin{aligned}
\kappa_{ij}(0)&=0,\\
\kappa_{ij}'(c)&=\tfrac14 E_C[\operatorname{sech}^2(X_i/2)\\
&\qquad\quad{}\times\operatorname{sech}^2(X_j/2)]>0.
\end{aligned}
\tag{5}
$$

所以观察协方差识别潜在相关。对目标求导时，每个粒子的*依赖标记*按该粒子在平均中所占份额加权。令$\pi_k$为归一化的**完整**粒子似然（含干扰因子），则

$$
\begin{aligned}
\partial_{c_{ij}}F_{n,K}
&=-E\sum_k\pi_k(1-\pi_k)H_{k,ij},\\
H_{k,ij}&=Y_iY_j\sigma(-Y_iX_{ki})\sigma(-Y_jX_{kj}),\\
EH_{k,ij}&=\{\kappa_{ij}(C_*)+\kappa_{ij}(C)\}/4.
\end{aligned}
\tag{6}
$$

标记的均值成立，是因为标签与候选粒子独立。但在有限$n$下，粒子的权重与其标记相互耦合，这一点在第4节会起作用。

令干扰尺度全为一，记$F_{n,K}^*(C)=F_{n,K}(C;\mathbf1)$，并定义

$$
\begin{gathered}
v=\operatorname{Var}\log\sigma(G),\qquad J_K=E\max_{k\le K}G_k,\\
w_{n,K}=J_K/\sqrt{nv}.
\end{gathered}
\tag{7}
$$

这里$J_K$是$K$个标准Gaussian最大值的期望，$J_K\sim\sqrt{2\log K}$，且对整数$K\ge2$有$J_K^2\asymp1+\log K$（附录A.2）。因此在下文的多项式范围内$w_{n,K}\asymp\sqrt{\log K/n}$，固定$K$时$w_{n,K}\asymp n^{-1/2}$。一个启发式说明这一尺度：若$K$个干扰对数权重是标准差为$\sqrt{nv}$的Gaussian，最大两者之差在零处的密度为$K(K-1)\int\phi^2\Phi^{K-2}/\sqrt{nv}=w_{n,K}$。$O(1)$的活跃对数似然差只在这种近乎并列时才会改变主导粒子。

## 极限势

**定理1（一致方向与势）。** 固定$m,\mu,\delta$、$C_*\in\operatorname{relint}\mathcal E_\delta^{(m)}$及$C_0,p>0$。当$n\to\infty$，对$C\in\mathcal E_\delta^{(m)}$及整数$2\le K\le C_0n^p$一致有

$$
\begin{aligned}
\frac{\partial_{c_{ij}}F_{n,K}^*(C)}{w_{n,K}}
&\longrightarrow-\frac{\kappa_{ij}(C_*)+\kappa_{ij}(C)}4,\\
\frac{F_{n,K}^*(C)-F_{n,K}^*(I)}{w_{n,K}}
&\longrightarrow V(C),\\
V(C)&=-\frac14\sum_{i<j}\Bigl[\kappa_{ij}(C_*)c_{ij}\\
&\qquad\qquad+\int_0^{c_{ij}}\kappa_{ij}(u)\,du\Bigr].
\end{aligned}
\tag{8}
$$

证明（附录A.1--A.2）直接处理实际的log-sigmoid干扰权重。它借助对多项式$K$成立的一致中等尾部与密度估计，控制近乎并列的干扰对数权重对，并在固定维数的相关类上给出活跃评分的共同包络。紧性本身给不出收缩尺度上的控制。沿可行线段从$I$到$C$积分梯度得到$V$。

以下三种解读使(8)具体化。

*Jensen项吸收了风险。* $F^*_{n,K}$的差以$w_{n,K}$的速率消失，而风险差$\mathcal R(C)-\mathcal R(I)$与$n$无关。由(4)，Jensen项随$C$的变化在主导阶上抵消了风险差。在$w_{n,K}$尺度上留下的是$V$，它对相关结构的排序可以不同于观察风险。

*$V$奖励什么。* 令$A_C(y,Z)=\sum_i\log\sigma(y_iX_i)$为标签$y$在一次先验抽样下的活跃对数似然，$T(C)=E_Y\operatorname{Var}_ZA_C(Y,Z)$为它在先验抽样间的方差，再对真实标签平均。Price恒等式给出$\partial_{c_{ij}}T=\{\kappa_{ij}(C_*)+\kappa_{ij}(C)\}/2$，于是

$$
V(C)=-\tfrac12\{T(C)-T(I)\}.
$$

极限目标偏好这样的相关矩阵：在它之下，观察的活跃对数似然在先验抽样间变化更大。这是活跃对数似然的条件方差，不是似然本身、干扰权重或随机梯度的方差（附录B.1）。

*两个变量。* 对中心化变量、真实相关$\theta$与spin矩$r(\rho)$，精确交叉熵的导数为$r'(\rho)\{r(\rho)-r(\theta)\}/\{1-r(\rho)^2\}$，其下降把$\rho$推向$\theta$。$\theta=0$时，极限粒子梯度$-\kappa(\rho)/4$符号相反，下降把非零相关推离零。更一般地，由(5)，$V$在相关坐标中的Hessian为对角且各元严格为负：$V$严格凹。

## 谱边界与风险

紧凸集上的严格凹函数只在极点处取到最小值。对相关矩阵，这些点受一个经典秩界约束[@litam1994]。

**定理2（谱偏好与风险）。** 在定理1的条件下，令$\mathcal M=\arg\min_{\mathcal E_\delta^{(m)}}V$及$r_{\max}=\lfloor(\sqrt{8m+1}-1)/2\rfloor$。每个$C\in\mathcal M$满足

$$
\operatorname{rank}\left(\frac{C-\delta I}{1-\delta}\right)\le r_{\max}.
\tag{9}
$$

因此至少$m-r_{\max}$个特征值等于$\delta$，而$C$仍正定。对最终满足$2\le K_n\le C_0n^p$的整数序列，若$\widehat C_n\in\mathcal E_\delta^{(m)}$的总体regret满足$F_{n,K_n}^*(\widehat C_n)-\inf_C F_{n,K_n}^*(C)=o(w_{n,K_n})$，则

$$
\begin{gathered}
\operatorname{dist}(\widehat C_n,\mathcal M)\to0,\\
\liminf_n\mathcal R(\widehat C_n)
\ge\varepsilon_*,\\
\varepsilon_*:=\min_{C\in\mathcal M}\mathcal R(C)>0.
\end{gathered}
\tag{10}
$$

常数依赖$(m,\mu,\delta,C_*)$。(9)中的仿射映射把$\mathcal E_\delta^{(m)}$送到一个相关矩阵集合，其中秩为$r$的极点满足$r(r+1)/2\le m$。(5)的单调性使观察矩识别潜在相关，紧性使$\mathcal M$在观察KL上与内部教师分离（附录A.3）。例如在四变量实例(A.21)中，$\mu=(-.6,-.2,.2,.6)$、$\delta=.1$、教师特征值为$(.5,1,1.2,1.3)$时，至少两个极限特征值等于$.1$，且$\liminf_n\mathcal R(\widehat C_n)\ge9/12800$。

这里的“谱边界”指特征值落在下界$\delta$。被选中的矩阵仍正定，单个相关也不必全部增大。对regret为$o_p(w_{n,K_n})$的随机输出，到$\mathcal M$的距离依概率趋于零，且对每个$\eta>0$有$P\{\mathcal R(\widehat C_n)<\varepsilon_*-\eta\}\to0$。定理关心的是总体regret，而不是每个有限数据集：经验标签分布会改变$V$的线性项，只含单一标签的数据集可以让精确极小点与粒子极小点重合（附录A.6）。

**其他参数制度。** 两个变体说明制度很重要。若活跃均值和尺度也自由，均值取值于包含真实单变量概率logit的紧盒，尺度取值于$[0,S]^m$，硬币仍为单位尺度，则$o(1)$的总体regret已经把活跃尺度推向零。观察分布收敛到真实边际之积，风险趋于教师的总相关（附录A.8）。若改为共同干扰尺度可学习且可以为零，联合最小化会关闭硬币：额外的干扰惩罚消失，剩下带有自身偏差的活跃有限$K$目标（附录A.9）。定理1--2把干扰尺度固定为一，正是这一点使硬币成为对照。

# 渐近图景如何对接有限尺度

定理1在任何有限$n$下都不给余项界。因此我们细看一个实例：$m=3$、$\mu=0$、$C_*=I$、$\delta=.1$、$K=100$。令$C(\rho)$的所有非对角元为$\rho$，$C_b=C(.9)$。在$n=64,256,1024,4096$，我们把目标差$g_n=F_n(I)-F_n(C_b)$和$C(.45)$处自由$c_{12}$导数$D_n$与其首项预测比较，其中$F_n=F_{n,100}^*$、$w_n=w_{n,100}$。比值$A_n=g_n/\{w_n[V(I)-V(C_b)]\}$与$B_n=-D_n/(m_H^{\mathrm{ref}}w_n)$在首项精确时等于一；$m_H^{\mathrm{ref}}$是$\kappa(.45)/4$的固定数值参考。

## 权重重叠不是学习信号

一个自然的捷径是追踪标量权重重叠$\omega_n=E\sum_k\pi_k(1-\pi_k)$，它衡量平均中有多少份额在粒子间共享。若标记与权重独立，(6)给出$D_n=-m_H\omega_n$。$n=64$时重叠紧贴其首项尺度，$\omega_n/w_n\simeq.992$。然而$A_n\simeq-.143$、$B_n\simeq-.149$：目标差和导数都与其首项预测*反号*，条件区间排除零。恒等式$D_n=-m_H^{\mathrm{ref}}\omega_n-\mathcal C_n$（其中$\mathcal C_n=E\sum_k\pi_k(1-\pi_k)(H_{k,12}-m_H^{\mathrm{ref}})$）把重叠误差$O_n$与权重—标记耦合$J_n$分开（图2b）。在64、256和1024，耦合的幅度占主导。到4096，$A_n$与$B_n$的点估计约为$.83$，但区间未认证10%的相对精度。完整权重也含活跃评分，所以这一拆分是恒等式，而不是因果分解（附录C.6）。

![100个粒子下的有限尺度；$n$为追加公平硬币的个数。(a)目标差、自由坐标梯度和标量重叠相对其首项预测的比值；负比值表示与预测反号。(b) $C(.45)$处的梯度残差，分为重叠误差$O_n$与权重—标记耦合$J_n$；图中$m_H$指固定数值参考$m_H^{\mathrm{ref}}$。误差条为26个目标在理想iid抽样和声明数值资格下的条件99%同时区间；131,072个银行在不同$n$间共享，$B_n$最后一次配对增量未分辨（附录C.6）。](figures/bridge_summary_zh.png){width=100%}

## 带符号的修正与认证的方向

标量重叠漏掉的部分可以在固定$K$下识别。记$L=\log\sigma(G)$、$\kappa_3=E(L-EL)^3$、$s_n=\sqrt{nv}$及$B_K=EM_K^2-1$，其中$M_K=\max_{k\le K}G_k$。对一个候选粒子，令$A_Y=\sum_i\log\sigma(Y_iX_i)$、$H_Y=H_{1,12}$（来自(6)）、$m_H=EH_Y$及$\bar c=E_Y\operatorname{Cov}_X(H_Y,A_Y)$。

::: {#fixed-k-proposition}

**命题（固定K修正）。** 在单位干扰尺度的中心化模型(1)中，固定$m$、正定候选$C$、教师标签律和整数$K\ge2$。当$n\to\infty$时，

$$
\begin{aligned}
D_n:=\partial_{c_{12}}F_{n,K}^*(C)
&=-\frac{m_HJ_K}{s_n}\\
&\quad+\frac{B_K}{s_n^2}
 \left(\frac{m_H\kappa_3}{3v}-\bar c\right)\\
&\quad+O_{K,m,C}(n^{-3/2}).
\end{aligned}
\tag{11}
$$

:::

新的成分是$\bar c$：*同一个粒子内部*、其依赖标记与活跃对数似然之间的协方差，先在给定标签下计算，再对标签平均。它不是粒子之间或训练梯度之间的协方差。活跃似然高的粒子获得更多权重；当它们的标记系统性地不同，加权后的标记就会偏移。对上述实例的$C=C(.45)$，$n^{-1}$系数为$b_2\simeq1.026$，分为干扰偏度部分$\simeq-.077$与标记—评分部分$\simeq1.103$（附录C.13）。因此修正与首项方向相反，且几乎全部来自同粒子耦合。这些系数是模型积分，而不是对梯度面板的拟合，不过推导修正时该面板已经可见。该命题在固定$K$下成立，不替代定理1的一致陈述。

::: {#finite-direction-corollary}

**有限规模推论。** 在$m=3$、$\mu=0$、$C_*=I$、$K=100$、$C=C(.45)$与单个自由$c_{12}$坐标处，显式误差界给出

$$
\begin{gathered}
-.002045393<D_{4096}<-.0003519373,\\
|D_{4096}-D_{4096}^{(0)}|<.001097138798\\
<|D_{4096}^{(0)}|,\qquad
D_n^{(0)}=-m_HJ_{100}/s_n.
\end{gathered}
\tag{12}
$$

:::

在$n=4096$，首项对这一坐标给出正确的符号。误差界约为$|D_{4096}^{(0)}|$的75.7%，因此这一证书关于符号，而不是10%的精度。在64、256和1024，同样的做法不能分辨符号。这些包围是确定性的，使用实际的Gumbel平滑干扰噪声和经认证的模型矩（[附录C.7起](#fixed-k-correction)）。

认证的符号有风险后果。由(5)和三变量观察律，$\partial_{c_{12}}\mathcal R=\kappa'(.45)\kappa/\{(1-\kappa)(1+3\kappa)\}>0$，其中$\kappa=\kappa(.45)>0$。因此$c_{12}$每个充分小的可行增加都降低$F^*_{4096,100}$，同时提高风险。这是一个有限的、坐标层面的目标—风险冲突，由其自身的显式余项界确立，而不是由定理2推出。

## $n=256$处的有限程序结论

图1的实例在$n=256$、$K=100$还给出有限程序陈述（附录C.15、D、E）。在理想iid抽样与声明数值条件下的99%审计事件上，类中每个总体regret低于$.00134$的相关矩阵，无论是否位于网格上，风险都高于$10^{-6}$；这在一个未被证明有算法能达到的近优条件下，排除了一个连续的低风险区域。在同一事件上，固定143点网格程序的新运行，以至少$.7035$的概率选出风险至少为$0.09/(32e)$的模型；这是该程序的概率下界，而不是观测频率。

# 没有追加块时：一个认证的局部冲突

## 全部参数自由的双因子模型

现在去掉对照。没有指定的独立块；所有偏置和载荷都被学习：

$$
\begin{aligned}
Z&\sim N(0,I_2),\qquad \theta=(b,W),\\
Y_j\mid Z&\sim\operatorname{Bernoulli}
 \{\sigma(b_j+w_j^\top Z)\},\\
F_K^q(\theta)&=E_{Y\sim q,Z_{1:K}}\Bigl[
-\log\Bigl\{K^{-1}\sum_{k=1}^K
p_\theta(Y\mid Z_k)\Bigr\}\Bigr].
\end{aligned}
\tag{13}
$$

这里$b$为logit偏置，$Y_j\in\{0,1\}$，先验粒子$Z_{1:K}$为iid且独立于$Y$；给定$Z$后各条目条件独立。目标$q$为经验律或教师律，$F_\infty^q$为积分交叉熵。教师有八个条目、零偏置，行长均为$.8$，角度为$(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$，因此所有载荷两两内积为正。这一模型类不同于第3节：logit协方差$WW^\top$至多秩二、没有谱下界，载荷秩也不等于$\operatorname{Cov}(Y)$的秩。

## 边际正确、风险为正，并排斥一个改进

**定理3（计算机辅助的局部依赖冲突）。** 对(13)中的精确教师，令$q_*$为其观察律，$\mathcal R(\theta)=D(q_*\Vert Q_\theta)$。存在秩一内部参数$\theta_*=(0,a_*,0)$，$a_{*,i}>0$且$\nabla F_8^{q_*}(\theta_*)=0$，它在完整24维参数空间中、模共同潜变量旋转后是$F_8^{q_*}$的严格局部极小。每个观察边际都正确，$Q_{\theta_*}(Y_j=1)=q_*(Y_j=1)=1/2$，但$\mathcal R(\theta_*)\simeq.01043>0$。

存在固定的$d=d_{\mathrm{ang}}^{(8)}\in\mathbf1^\perp\subset\mathbb R^8$、$\|d\|_2=1$，使路径$\theta_{\mathrm{ang}}(t)=(0,W(t))$满足

$$
\begin{aligned}
W_i(t)&=a_{*,i}\bigl(\cos(td_i),\sin(td_i)\bigr),\\
r(t)&=\mathcal R(\theta_{\mathrm{ang}}(t)),\\
r'(0)&=0,\qquad r''(0)<-.00450377.
\end{aligned}
\tag{14}
$$

存在$\varepsilon>0$，使每个$0<|t|<\varepsilon$都有

$$
\begin{gathered}
F_8^{q_*}(\theta_{\mathrm{ang}}(t))>F_8^{q_*}(\theta_*),\\
\mathcal R(\theta_{\mathrm{ang}}(t))<\mathcal R(\theta_*),\\
\operatorname{rank}W(t)=2.
\end{gathered}
\tag{15}
$$

这条路径只改变依赖。行长固定，所以每个候选logit保持其Gaussian边际$N(0,a_{*,i}^2)$，每个观察仍是公平硬币，而$\operatorname{Cov}(X_i(t),X_j(t))=a_{*,i}a_{*,j}\cos\{t(d_i-d_j)\}$随之改变。这些保持不变的logit方差是候选自己的，不是教师的$.8^2$。由于该根是严格局部极小，而非常量的角路径离开旋转轨道，目标沿路径上升。风险下降，是因为它沿路径的曲率为负。这一冲突是二阶的、局部的：沿这条路径的每个充分小非零步都降低风险、抬高目标（附录I.10）。固定行长定义了这条比较路径，并不是对学习问题的约束。[附录I.8](#full-local-statement)中的完整陈述还认证了一个一阶风险导数为负的方向$d_{\mathrm{risk}}$；沿它走一个认证的$10^{-7}$步，会抬高$F_8$、降低风险并保持观察边际公平，同时改变行长。

作为尺度参照：在四个各含$4096$个教师样本的训练流中，独立Bernoulli拟合的KL约为$.107$，$K=8$的Adam训练结束于约$.011$（附录I.4）；共线子模型内积分目标的一个重新拟合驻点，KL约为$.0070$（I.6）。这些是数值比较，不是认证的最优值。

**证书概要。** 对称性把求根降为四个坐标，区间根包含证明在一个二进中心附近存在精确驻点。完整空间稳定性需要所有方向：全部16个偏置/径向方向和7个非旋转横向方向都有认证的正曲率；只检查角方向是不够的（[附录I.2](#full-hessian-criterion)、[I.8](#full-local-statement)）。(14)沿路径的风险曲率包含路径加速度。它借助Gaussian分部积分、解析求积误差界[@trefethen2014]和中点—半径算术[@johansson2017]在全部256个标签上被包围，再通过行长上的Lipschitz界从中心传递到真根（[I.10](#same-root-angle)）。

::: {#quantitative-persistence-corollary}

**推论（定量局部持续性）。** 保持教师的八个偏置为零、行长为$4/5$，只把其第一个角度改为$-.7+\lambda$，$|\lambda|\le.001$。记其观察律为$q_\lambda$，存在与定理3的根相连的光滑分支$\theta_\lambda=(0,a(\lambda),0)$，每一点都是其自身$F_8^{q_\lambda}$在完整空间中模旋转的严格局部极小。用同一固定$d$、并在(14)中以$a(\lambda)$替换$a_*$，对每个教师和每个充分小的非零$|t|<\varepsilon_\lambda$，得到一条秩二路径：它抬高该教师的有限目标、降低其自身风险$D(q_\lambda\Vert Q_\theta)$，并保持候选Gaussian logit边际和正确的公平观察边际。

:::

这一教师族打破了原教师的对称性，但范围很小：端点观察律的$L^1$差约为$1.6\times10^{-4}$（[附录I.11](#teacher-family-full-statement)）。对更一般、未量化的教师扰动，[附录I.9](#local-persistence)证明附近仍有一个秩一严格局部极小，并带有一阶风险下降方向。

## 弱载荷下的对齐直觉

固定$K>2$时的弱载荷展开给出直觉，但不是定量解释（附录I.1）。行长固定、载荷为$\varepsilon$阶时，在与角度无关的项之外，有限目标为$\varepsilon^4a_K\|c-\{K/(K-2)\}c_*\|^2+O(\varepsilon^6)$，其中$a_K>0$见(I.2)，$c$收集去掉共同尺度$\varepsilon$后各行的两两内积。它奖励相对教师被*放大*的内积。行长固定时，各行只能通过对齐来放大内积。在第3节的独立教师情形，极限势同样偏好教师所没有的依赖。两个展开分别推导，彼此不解释对方。共线构型的弱载荷稳定边界$K\simeq4.33$，并不预测载荷为$.8$时的行为。

数值学习与这一图景相容，但并不确立它（附录I.4--I.7）。在四个数据流中，$K=8$训练得到近乎共线的载荷，$K=64$和积分参照则没有。在每个$K$重新拟合后，共线点的局部类型在数值上于$K=11$与$12$之间改变；那里的角度探针只使风险变化约$.02\%$，这一效应量属于这些探针，而不属于认证的$K=8$冲突。

# 相关工作

**Monte Carlo目标与模拟似然。** 无偏似然估计量的对数是有偏的，且偏差依赖参数。这一点已在离散选择模型的模拟极大似然[@lee1995]、Monte Carlo似然推断[@sung2007]（其中在对数内部提高模拟精度与对固定$K$的对数目标取平均是两个不同的极限）以及Monte Carlo变分目标[@maddison2017]中确立。对非负无偏的归一化常数估计量，AESMC的Claim 1把期望对数间隙写成扩展抽样空间上的KL散度[@le2018aesmc]；该散度不是本文所研究积分模型的观察风险。专就依赖而言， @geweke1997 的摘要报告了GHK模拟似然实验中序列相关估计的向下偏差；我们把它当作经验先例，而不声称本文机制解释了它。

**权重退化与梯度渐近。** 高维重要性抽样[@agapiou2017; @chatterjee2018]、粒子滤波崩塌[@bengtsson2008; @bickel2008; @snyder2008]和多样本梯度估计量[@rainforth2018; @tucker2019]都已有充分研究。大$K$时，$w_{n,K}$与 @bickel2008 中最大权重的期望超额同阶；判断依赖选择需要在这一尺度上的控制。与定理1最接近的结果是 @daudel2026 （D.4.1节，式(208)）。在对数权重满足Gaussian随机场条件、并可交换导数与期望时，次主导的平均梯度服从对数权重标准差的导数。借助第3节的条件方差恒等式，该公式可形式地预测本文的大$K$系数。其证明用到标准化对数权重场与其参数导数之间的独立性，而近似Gaussian性并不保持这种独立性（其3.4节）。 @daudel2023 （定理6）处理非Gaussian的iid对数权重和，给出标量目标渐近；该标量余项不能简单地求导成在更小选择尺度上一致的相关场。定理1增加的正是这种控制：针对实际的非Gaussian权重，在整个相关类以及$2\le K\le C_0n^p$上一致成立，包括每个固定$K\ge2$。定理2再把它与相关矩阵的经典极点几何[@litam1994]结合，得到观察风险结论。附录B给出定理层面的比较。

**潜变量学习与局部几何。** 以先验抽样似然学习二元潜变量模型早于本文[@tipping1998]。重要性加权自编码器在目标改变时会改变其学到的表示[@burda2016, Section 5.2 and Table 2]；本文的准则是其提议分布固定为先验时的界取负。学习提议分布的版本已用于项目因子分析[@urban2021]。线性潜变量模型中的崩塌与局部稳定性已有分析[@lucas2019; @wangziyin2022]；对线性VAE， @lucas2019 （定理1）排除了相对边际似然额外的ELBO局部极大，并指出固定观察方差会影响pPCA的稳定性。定理3则在一个非线性模型中认证有限粒子目标的完整空间局部极小，以及一个保持边际、沿之目标排斥风险改进的方向；它不是关于一般低秩性的论断。

# 讨论

**这些结果改变了什么判断。** 固定$K$时，先验粒子目标有自己的依赖偏好，把它估计得更准，并不会把这些偏好移向积分似然。在受控族中，这些偏好是$w_{n,K}$阶的，却决定了选中哪些相关，并导致正的极限风险（定理1--2）。在有限尺度实例中，标量重叠可以紧贴其首项预测，而相关梯度却与自己的首项预测反号（第4节）。在完整空间局部极小处边际正确，并不意味着目标欢迎能降低风险的依赖变化（定理3）。

**范围。** 各结果的保证类型不同：定理1--2是一致渐近陈述；(12)中的推论是单个$n$上的确定性包围；第4.3节依赖带声明数值条件的99%审计事件；定理3是计算机辅助的局部证明。它们都使用先验粒子；除第4.3节的网格程序界外，研究对象都是总体目标。第一部分要求已知独立、单位尺度的追加块，固定的活跃维数与边际，以及相对$w_{n,K}$很小的regret。第二部分是一个模型及一个小的显式教师族。一个自然的后续问题是：学习得到的、适应后验的提议分布[@burda2016]会消除、缩小还是反转这一偏好。

**可复核性。** 随附的匿名补充材料提供保存的区间输入、证书与有限程序界的算术检查，以及图1--2的数据和作图脚本。其README说明运行命令与核验范围。这些检查重建保存证据，不重新运行生产求积或训练。

# 影响声明

本研究旨在加深对基于模拟的概率学习的理解，其潜在益处是更可靠地区分随机训练准则与积分模型的风险。研究使用受控合成观测，尚未建立部署收益。应用这些结果时，需要保留其假设与有限预算限制，包括文中报告的恢复边界和反对照。

# 附录阅读导引

保留的附录给出正文结果的证明与证据，并按连续字母编号。下表列出正文每项内容的支持位置。

|正文内容|在附录何处成立|支持类型|
|---|---|---|
|图1，公平硬币对照（第1节）|C.1匹配选择；C.2目标差审计；C.3 $n=0$解析曲率；C.4 $n=256$曲率审计|$n=0$解析界；各带数值资格的99%审计家族；一次保存的匹配运行|
|模型、分解与表示事实（第2节）|A.1|解析|
|选择尺度与定理1（第3.1--3.2节）|A.2；尺度启发式及其与粒子滤波崩塌的关系见B.2|一致渐近证明|
|$V$的条件方差解读|B.1|精确恒等式|
|定理2与四变量例子|A.3（A.21--A.22）|解析|
|其他参数制度|A.8（活跃尺度自由）、A.9（干扰尺度可学习）；其余边界见A.4--A.7|解析|
|图2，重叠与耦合（第4.1节）|C.5--C.6|条件99%同时区间，26个目标|
|[固定$K$修正](#fixed-k-proposition)与[有限规模推论](#finite-direction-corollary)（第4.2节）|[C.7起](#fixed-k-correction)：证明C.8，误差预算C.9--C.12，常数与合成C.13，风险导数C.14|渐近证明；$n=4096$的确定性包围|
|有限程序结论（第4.3节）|[C.15](#finite-evidence-summary)；D（新运行界）；E（保存输入算术）|99%审计事件、理想iid抽样、声明的数值条件|
|模型(13)与弱载荷启发式（第5.1、5.3节）|[I.1](#dense-model)|带定性余项的展开|
|定理3（第5.2节）|[I.2](#full-hessian-criterion)判据（[完整导数](#full-derivatives)）；[I.3](#laplace-representation)表示；[I.8](#proof-theorem-3)真根与[完整空间陈述](#full-local-statement)；[I.10](#same-root-angle)同根角路径|计算机辅助证明|
|[定量持续性推论](#quantitative-persistence-corollary)（第5.2节）|[I.11](#quantitative-teacher-family)及[完整族陈述](#teacher-family-full-statement)|在$\lvert\lambda\rvert\le.001$上的计算机辅助证明|
|定性持续性（第5.2节）|[I.9](#local-persistence)|隐函数论证；邻域未量化|
|数值学习与K11/K12探针（第5.3节）|I.4--I.7|数值结果，附求积资格说明|

**只在附录报告的材料。** 已知均值稳健性与观察等价尺度（F）；说明匹配均值不必保持有限粒子驻点性的两粒子反例（G）；以及早期有限选择及其负对照（附录H）。

A.2保留完整的一致证明。保存端点与证书合成的检查不重新计算概率、梯度或Hessian网格。

\clearpage

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

\clearpage

# 附录 B. 文献定位与条件方差解释

## B.1. 极限势的精确解释

保持第2节的固定Gaussian边际与教师。记

$$
A_C(y,Z)=\sum_{i=1}^m\log\sigma(y_iX_i),\qquad
T(C)=E_{Y\sim Q_{C_*}^A}\!\left[\operatorname{Var}_Z A_C(Y,Z)\right].
\tag{B.1}
$$

对每个固定$y$，均值$E_ZA_C(y,Z)$仅依赖固定一维边际，故其对自由非对角协方差的导数为零。因$A_C$是逐坐标和，$A_C^2$的混合导数为$2(\partial_{x_i}A_C)(\partial_{x_j}A_C)$。Gaussian分部积分给Price恒等式[@price1958]；平方与导数项在紧类上由可积Gaussian包络一致控制。使用式(6)，

$$
\partial_{c_{ij}}T(C)=2EH_{ij}
=\frac{\kappa_{ij}(C_*)+\kappa_{ij}(C)}2,
\qquad V(C)=-\frac12\{T(C)-T(I)\}.
\tag{B.2}
$$

第二个身份沿$I$起始的可行线段积分并用$V(I)=0$得到。它解释既有势函数，不是独立发现方差驱动选择。对象为给定观测的*活跃对数似然*方差，再按真实观测平均；不同于无关log权重方差、随机梯度方差或预测概率方差。辅助数值核验用均值$(-.6,.4)$、教师相关$.25$及候选相关$(-.8,-.3,0,.4,.8)$。32/48/64阶直接条件方差求积与64点Legendre积分的协方差表达之差至多$5.85\times10^{-14}$；身份由上文解析证明支持。

## B.2. 最近渐近结果的预测与缺口

Daudel--Roueff v2命题1证明的式(208)，在$\partial_\psi E\Lambda$之外包含次阶均值梯度项$\sqrt{2\log K}\,\partial_\psi B$ [@daudel2026]。该均值梯度公式令粒子数与$B$趋无穷，并要求$\log K=o(B^2)$。相关定位为Gaussian log权重随机场条件，以及Daudel--Roueff的第D.4.1节、式(208)。形式上匹配本模型给定观测的log权重均值与方差，有

$$
\begin{gathered}
B_C(y)^2=nv+\operatorname{Var}_Z A_C(y,Z),\\
-\sqrt{2\log K}\,E_Y\partial_{c_{ij}}B_C(Y)
=-\frac{\sqrt{2\log K}}{2\sqrt{nv}}\partial_{c_{ij}}T(C)+O\!\left(\frac{\sqrt{\log K}}{n^{3/2}}\right).
\end{gathered}
\tag{B.3}
$$

当$n\to\infty$且$K\to\infty$保持在多项式范围内时，它预测式(8)的系数。这是机制联系，而不是对实际模型直接应用：logistic对数权重不是Gaussian随机场；普通CLT不提供近极值区域相对双粒子核精度，也不允许对未控余项求导。精确Gaussian性承重于D.4.1节式(208)证明：归一化权重与标准化对数权重场的中心化导数独立，因此该导数项期望为零。D.3节的Gaussian场性质提供独立性；第3.4节式(49)之后说明，仅有近似Gaussian性时会丢失该性质。此处用条件内容与节号定位，避免版本依赖的条件标签。全相关类一致性和固定$K\ge2$也须建立。此外，以匹配矩的Gaussian替代log权重，一般会改变精确指数积分，后者为$\log E e^\Lambda=E\Lambda+\operatorname{Var}(\Lambda)/2$；故形式替换未必保持统计模型。

Daudel等定理6（A2及式(37)--(38)）已在$\log K=o(d^{1/3})$下处理非Gaussian log权重和[@daudel2023]；标量展开不是一致$o(w)$参数差或梯度定理。类似地，Bickel等命题3.1要求附录A.1--A.2的相对尾/密度条件，以及$(\log K\log d)/d\to0$ [@bickel2008]。将其粒子数对应到$K$、评分标准差对应到$\sqrt{nv}$时，最大权重之外质量的期望与$w_{n,K}$具有相同的大$K$首阶尺度$\sqrt{2\log K}/\sqrt{nv}$；该尺度并非新发现，但不直接给带mark的条件双粒子估计(A.13)。更早退化结果[@bengtsson2008; @snyder2008]属于背景。以上比较不排除旧框架未来扩展；本文不声称恢复这一特定参数选择必需指数粒子数。

## B.3　与近邻结果的定理级比较

本文区别在于实际非Gaussian模型的参数选择尺度一致分析及其观察风险后果；B.1条件方差身份将机制联系到已有高失配梯度理论。一般Jensen偏差、估计器影响学习、塌缩及相关矩阵极点几何，均为既有工具或现象。下表比较各结果实际分析的对象，不声称旧框架不可能扩展到本模型。

|结果|关键假设/分析对象|已有结论及本文区别|
|---|---|---|
|Monte Carlo objectives / FIVO [@maddison2017]|正的无偏似然估计器及期望对数目标，包括粒子过滤估计器。|给出一般目标构造与下界性质；本文识别观察等价无关表示诱导的带符号相关场。|
|AESMC的Claim 1 [@le2018aesmc]|扩展采样空间提议下的非负无偏归一化常数估计器。|期望对数间隙是到估计器诱导目标的扩展空间KL，不是积分候选的观察KL；不识别本文一致有符号相关场。|
|Daudel等定理6 [@daudel2023]|高维VR-IWAE下界与权重模型，指定条件下包含非Gaussian log权重和。|A2及式(37)--(38)、$\log K=o(d^{1/3})$给出非精确log-normal模型的标量目标gap渐近；标量变平本身不提供本文完整相关场和风险结论。|
|Daudel–Roueff v2 [@daudel2026]|Gaussian log权重随机场条件；导数/期望交换；Daudel--Roueff第D.4.1节、命题1式(208)。|标准差导数经B.1形式上预测本文大K系数；实际非Gaussian条件双粒子控制、全C一致性和固定K仍需额外分析，不能对未控CLT余项直接求导。|
|粒子滤波退化 [@bengtsson2008; @bickel2008; @snyder2008]|Bickel等命题3.1：附录相对尾/密度条件及$(\log K\log d)/d$趋零。|最大权重之外质量及极值尺度已有；不直接控制条件于活跃评分平移的带mark重叠。|
|模拟似然 [@lee1995]|离散选择模拟极大似然及对数导数偏差。|经典偏差背景，不是当前增长维数一致场的定理。|
|多期probit模拟 [@geweke1997]|多项多期probit模型中的GHK模拟似然。|发表摘要报告序列相关估计的向下偏差；不由此推出普遍符号或干扰增长机制，本文未重新核对其完整实验分析。|
|相关矩阵极点 [@litam1994]|实相关矩阵及其极点几何。|秩限制是已有几何；本文模型特定的一步是使粒子诱导严格凹势的极小点落入极点集，再将距离转为观察KL。|
|潜变量塌缩 [@lucas2019; @wangziyin2022]|所分析的线性潜变量模型及似然/变分目标。|塌缩及方差学习的限制早已存在；本文尺度边界结合独立无关碰撞、受限相关类与Gaussian-logistic观察KL极限。均匀权重Hessian只是特例，不另占新颖性。|
|本文固定尺度定理|固定有效维数与Gaussian边际；独立正紧无关尺度；多项式K；总体regret为o(w)。|一致首个有效带符号场、严格凹矩阵势、floor谱几何及正观察风险界；不推出有限优化器极限。|
|本文自由有效尺度边界|紧有效均值/尺度，固定单位无关尺度，多项式K；总体regret为o(1)。|统一主阶目标趋向边际目标，风险趋向教师总相关；共同释放无关尺度须另用A.9，有限程序结果为不同层次证据。|

\clearpage

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

\clearpage

# 附录 D. 固定有限搜索的新运行风险

## D.1. 数学程序及其风险分离

固定中心化三变量 Gaussian-logistic 模型、独立教师 $C_*=I$、单位边际方差、$K=100$ 及 $n=256$。候选集合 $\mathcal C_{\rm grid}$ 包含 $\{-.9,-.6,-.3,0,.3,.6,.9\}^3$ 中使单位对角相关矩阵满足 $C\succeq.1I$ 的143个三元组。真值 $I$ 和非对角元均为 $.9$ 的等相关锚点 $C_b$ 都在该集合中。下文均使用正定对称平方根。

每次新运行从 $\{-1,1\}^3$ 的均匀分布抽取 $N=65,536$ 个 iid 活跃标签，另独立抽取 $B=16,384$ 个 iid 银行。银行 $W_b$ 包含 $K$ 个标准三维活跃 Gaussian 向量，每个粒子另有256个独立标准 Gaussian 无关坐标。所有观测和候选共享同一批银行。定义

$$
\begin{aligned}
S_k&=\sum_{j=1}^{256}\log\sigma(G_{kj}),\qquad
a_k(W)=\frac{e^{S_k}}{\sum_{l=1}^{K}e^{S_l}},\\
\phi(C;y,W)&=-\log\sum_{k=1}^{K}a_k(W)
 \prod_{j=1}^{3}\sigma\!\left(y_j(C^{1/2}Z_k)_j\right),\\
\widehat F(C)&=\frac1{NB}\sum_{\ell=1}^{N}\sum_{b=1}^{B}\phi(C;Y_\ell,W_b),\\
d(y,w)&=\phi(I;y,w)-\phi(C_b;y,w),\quad
g=\mathbb E d(Y,W),\quad
\widehat g=\frac1{NB}\sum_{\ell,b}d(Y_\ell,W_b).
\end{aligned}
\tag{D.1}
$$

数学程序返回与最小数学评分相差不超过 $\mathrm{tol}=10^{-12}$ 的词典序首个候选，不使用观察风险比较或回退规则。这个定义将数学求值与其浮点实现区分开来。

每个非独立网格点至少有一项 $|c_{ij}|\ge.3$。对观测对 spin 矩 $\kappa$，附录C已证明 $\kappa(0)=0$ 及 $\kappa'(c)\ge\beta=e^{-1/2}/4$。有界的观测对 spin 统计量使全变差至少为 $|\kappa(c_{ij})|/2$；再沿真值到候选的 KL 方向使用 Pinsker，得到

$$
\begin{aligned}
\mathcal R(C)&\ge\tfrac12\beta^2(.3)^2
=r_{\rm grid}:=\frac{0.09\,\exp(-1)}{32}>.00103466
 \quad(C\in\mathcal C_{\rm grid}\setminus\{I\}),\\
\widehat g>\mathrm{tol}
&\ \Longrightarrow\
\widehat F(I)>\min_{C\in\mathcal C_{\rm grid}}\widehat F(C)+\mathrm{tol}
\ \Longrightarrow\ \mathcal R(\widehat C)\ge r_{\rm grid}.
\end{aligned}
\tag{D.2}
$$

第一个蕴含使用 $\min_C\widehat F(C)\le\widehat F(C_b)$，不需要判定其余142点中谁胜出。全文使用精确风险常数；向上舍入的近似值 $.00103466093$ 不是已经认证的风险下界门。

## D.2. 共享银行方差与 Gaussian 尾界

令 $m(y)=\mathbb E_Wd(y,W)$、$q(w)=\mathbb E_Yd(Y,w)$。交叉平均的分解为

$$
\begin{aligned}
d(y,w)&=g+a(y)+b(w)+r(y,w),\qquad
a(y)=m(y)-g,\quad b(w)=q(w)-g,\\
\mathbb E[r(Y,W)\mid Y]&=\mathbb E[r(Y,W)\mid W]=0,\\
\operatorname{Var}(\widehat g)
&=\frac{\sigma_Y^2}{N}+\frac{\sigma_W^2}{B}+\frac{\sigma_r^2}{NB},\\
\sigma_Y^2&=\mathbb E a(Y)^2,\qquad
\sigma_W^2=\mathbb E b(W)^2,\qquad
\sigma_r^2=\mathbb E r(Y,W)^2.
\end{aligned}
\tag{D.3}
$$

条件中心化使残差协方差在任一索引不同时消失，三项之间也相互正交。这证明了 (D.3)，没有将 $NB$ 个交叉项当作独立观测。

坐标置换保持 $C_b$ 及其平方根不变，全局符号反射保持 Gaussian 银行分布不变。因此 $m(y)$ 只有两个取值：两个全同号标签对应 $m_{\rm same}$，其余六个标签对应 $m_{\rm split}$。于是

$$
g=\tfrac14m_{\rm same}+\tfrac34m_{\rm split},\qquad
\Delta=m_{\rm same}-m_{\rm split},\qquad
\sigma_Y^2=\tfrac3{16}\Delta^2.
\tag{D.4}
$$

这只是条件期望的对称性。同一银行实际得到的八个点差不必只有两个值，程序也不对输入作对称化。

$\log\sigma$ 是1-Lipschitz函数，log-sum-exp 关于最大范数同样是1-Lipschitz。两个锚点使用完全相同的无关权重，因此下面的包络不依赖这些权重。附录C取 $t=3/8$ 的 Gaussian 指数矩论证给出

$$
\begin{aligned}
A&=\|\sqrt{C_b}-I\|_{\rm op}
=\max(1-\sqrt{.1},\sqrt{2.8}-1),\qquad
M_Z=\max_{k\le K}\|Z_k\|_2,\\
|d(y,W)|&\le\sqrt3\,A M_Z,\qquad
\mathbb E M_Z^2\le b_K:=\tfrac83(\log K+3\log2),\\
\sigma_r^2&\le\mathbb E d(Y,W)^2\le3A^2b_K.
\end{aligned}
\tag{D.5}
$$

最后一个界来自 (D.3) 的正交性，直接控制残差项，不需要再增加随机审计目标。

取 $D=3K=300$、$T=6$，以及 $E_T=\{\max_{k,j}|Z_{kj}|\le T\}$。定义银行的类别差 $u(w)$ 为两个同号标签点差的平均，减去六个异号标签点差的平均；于是 $\mathbb Eu(W)=\Delta$。三个截断审计变量及其范围为

$$
\begin{gathered}
X=q(W)\mathbf1_{E_T},\qquad
D_T=u(W)\mathbf1_{E_T},\qquad M=3AT,\\
X\in[-M,M],\qquad X^2\in[0,M^2],\qquad D_T\in[-2M,2M].
\end{gathered}
\tag{D.6}
$$

以 $\varphi$ 和 $\overline\Phi$ 表示标准 Gaussian 密度和上尾概率。坐标联合界与 Mills 不等式控制尾概率。为控制二阶矩，用全部 $D$ 个坐标的平方和上界 $M_Z^2$，再用坐标越界指示之和上界尾事件指示。对角项使用 $\mathbb E[Z^2\mathbf1_{\{|Z|>T\}}]=2(T\varphi(T)+\overline\Phi(T))$，非对角项使用独立性；没有假设越界事件互斥。因此

$$
\begin{aligned}
\mathbb P(E_T^c)&\le p_T:=2D\varphi(T)/T,\\
\mathbb E[M_Z^2\mathbf1_{E_T^c}]
&\le2D\{T\varphi(T)+D\overline\Phi(T)\}
\le Q_T:=2D\varphi(T)(T+D/T),\\
\tau_2&:=3A^2Q_T,\qquad \tau_1:=\sqrt{\tau_2p_T},\\
|g-\mathbb EX|&\le\tau_1,\qquad
|\Delta-\mathbb ED_T|\le2\tau_1,\qquad
\sigma_W^2\le\mathbb E q(W)^2\le\mathbb EX^2+\tau_2.
\end{aligned}
\tag{D.7}
$$

Cauchy--Schwarz 给出均值尾界。数值上 $\tau_1\simeq1.31901920513\times10^{-5}$、$\tau_2\simeq2.86347137776\times10^{-4}$；实际计算对这些常数作向外外包。

## D.3. 使用另行生成银行的三分量矩审计

这里是作者使用另行生成银行样本的计算，不是第三方审计。审计使用 $L=262,144$ 个新银行，每个银行内评价全部八标签与两个锚点。这些样本与未来程序的 $N$ 个标签、$B$ 个银行分开。此前的独立银行只用于开发和选择截断设计，没有进入新置信统计。正式运行在生成新银行前固定 $T=6$、三个目标和误差分配。可再生实现采用 PCG64DXSM，种子序列为 $[202610060501,j]$，活跃与无关部分使用独立的 $j=1,2$ 流。概率解释以规定的 iid Gaussian 模型为前提，可再生性本身不证明这一前提。

对每条保存的 binary64 标量序列 $\widetilde V_1,\ldots,\widetilde V_L$，选择共同的二进制分母，并精确累积整数：

$$
\widetilde V_b=2^{-e}z_b,\quad S_1=\sum_bz_b,\quad S_2=\sum_bz_b^2,\qquad
\overline{\widetilde V}=\frac{S_1}{L2^e},\quad
\widetilde s_V^2=\frac{LS_2-S_1^2}{L(L-1)2^{2e}}.
\tag{D.8}
$$

$X$、$X^2$、$D_T$ 的指数分别为56、95、68。之后用高精度球算术外包标准差、置信端点和最终概率。表中的样本矩是舍入后的描述值，不是真实方差的界。

|审计变量|样本均值|无偏样本方差|
|---|---:|---:|
|$X$|.00562444984697731|.04106054244035259|
|$X^2$|.04109202024289478|.011140404592328061|
|$D_T$|.23838858527527407|.05025621128454873|

没有银行触发截断。三项各分配 $\alpha_j=.01/3=1/300$：对 $X$ 的均值作单侧下界，对 $X^2$ 的均值作单侧上界，对 $D_T$ 的均值作双侧区间。对范围长度 $R_j$，将数值样本误差同时传播到均值和标准差，经验 Bernstein 不等式 [@maurer2009, Theorem 4] 使用

$$
\begin{aligned}
\epsilon_X&=10^{-9}+10^{-12},\qquad
\epsilon_D=2\cdot10^{-9}+10^{-12},\\
\epsilon_2&=(2M+\epsilon_X)\epsilon_X+10^{-12},\\
B_j&=\sqrt{\frac{2\ell_j}{L}}\!
 \left(\widetilde s_j+\sqrt{\frac{L}{L-1}}\epsilon_j\right)
 +\frac{7R_j\ell_j}{3(L-1)},\\
(R_X,R_2,R_D)&=(2M,M^2,4M),\qquad
(\ell_X,\ell_2,\ell_D)=(\log600,\log600,\log1200).
\end{aligned}
\tag{D.9}
$$

这里 $|\widetilde V_b-V_b|\le\epsilon_j$ 是相应数值条件。中心化 Euclidean 范数给出 $|s_j-\widetilde s_j|\le\sqrt{L/(L-1)}\epsilon_j$。因此 (D.9) 使用精确数学范围，放宽样本标准差，并在均值端点另加入 $\epsilon_j$。附录E给出完整算术方法和仍保留的输入资格。特别地，后续对全部2,097,152个保存的银行×标签点差进行了外包；当保存的 $Z/S$ 视为精确数时，最大算术误差低于 $3.331\times10^{-14}$，原 $10^{-9}$ 余量保持不变。这次确定性重算不引入新置信家族，也不认证输入生成或所有未来运行。

在依相应条件具有至少 $.99$ 概率的同时审计事件上，向外端点定义为

$$
\begin{aligned}
g_L&=\overline{\widetilde X}-\epsilon_X-B_X-\tau_1,\\
U_2&=\overline{\widetilde{X^2}}+\epsilon_2+B_2+\tau_2,\\
\Delta_-&=\overline{\widetilde D_T}-\epsilon_D-B_D-2\tau_1,\qquad
\Delta_+=\overline{\widetilde D_T}+\epsilon_D+B_D+2\tau_1,\\
U_\Delta&=\max(|\Delta_-|,|\Delta_+|),\\
g&\ge g_L,\quad \sigma_W^2\le U_2,\quad |\Delta|\le U_\Delta,\qquad
\operatorname{Var}(\widehat g)\le V_U:=
\frac{3U_\Delta^2}{16N}+\frac{U_2}{B}+\frac{3A^2b_K}{NB}.
\end{aligned}
\tag{D.10}
$$

银行项使用总体二阶矩上界，不以没有资格的样本方差替代真实方差。下表保守显示认证端点；实际推断使用完整的向外端点。

|量|保守显示值|
|---|---:|
|$g_L$|.00279405023657203|
|$U_2$|.05074113097360336|
|$U_\Delta$|.24317070176659291|
|$V_U$|$3.289456534426161\times10^{-6}$|
|$p_L$|.70355036|

下表给出 $V_U$ 的三项贡献，均向上显示。它们是该方差**上界**的组成，不是经验方差分解或因果归因。

|上界组成|向上显示值|
|---|---:|
|数据：$3U_\Delta^2/(16N)$|$1.691779809884757\times10^{-7}$|
|银行：$U_2/B$|$3.096992857275596\times10^{-6}$|
|残差：$3A^2b_K/(NB)$|$2.328569616208936\times10^{-8}$|

## D.4. 概率陈述及其边界

由于 $g_L>\mathrm{tol}$，对新运行交叉平均使用单侧 Cantelli，再结合 (D.2)，在同一审计事件上有

$$
\begin{aligned}
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C)\ge r_{\rm grid}\}
&\ge\mathbb P_{\rm fresh}\{\widehat g>\mathrm{tol}\}\\
&\ge\frac{(g_L-\mathrm{tol})^2}{V_U+(g_L-\mathrm{tol})^2}
\ge p_L>.7035.
\end{aligned}
\tag{D.11}
$$

具体地，Cantelli 将 $\mathbb P\{\widehat g\le\mathrm{tol}\}$ 上界为 $V/(V+(g-\mathrm{tol})^2)$；该式随 $V$ 增加、随 $g>\mathrm{tol}$ 减小。记录的下端点约为 $.703550362875$，表格和正文采用保守向下的显示值。

**99%是对新运行风险概率下界的审计置信度。** 70.35%是一个下界，不是观察到的失败频率。本审计没有反复执行143点搜索，也没有抽取新的训练标签；它建立的是这一充分条件论证所需的矩界。KL 的非负性还给出确定性推论

$$
\mathbb E_{\rm fresh}\mathcal R(\widehat C)
\ge .7035\,r_{\rm grid}>.00072788.
\tag{D.12}
$$

它使用同一审计事件，不花费额外误差预算。

结果只针对 $n=256$ 程序本身，没有证明至少70.35%的匹配新运行会从正确的 $n=0$ 输出变成错误的 $n=256$ 输出，也不主张每个错误输出都等于 $C_b$ 或具有 $.0311$ 风险。更细的网格可能含有其他低风险点，使单锚点捷径失效。附录C的连续区域结论与本附录的粗网格概率结论提供不同保证；二者都不是 Adam 或任意连续优化的保证。

浮点程序需要另一个条件。假设在新运行概率至少为 $1-\delta_{\rm alg}$ 的事件上，所有候选评分都有限、程序按规定容差返回可行网格点，并且 $|\widehat g_{\rm fp}-\widehat g|\le\epsilon_{\rm alg}$。令 $t_{\rm alg}=\mathrm{tol}+\epsilon_{\rm alg}<g_L$，由联合界得到

$$
\mathbb P_{\rm fresh}\{\mathcal R(\widehat C_{\rm fp})\ge r_{\rm grid}\}
\ge\left[
\frac{(g_L-t_{\rm alg})^2}{V_U+(g_L-t_{\rm alg})^2}
-\delta_{\rm alg}\right]_+,\qquad [x]_+=\max(x,0).
\tag{D.13}
$$

左侧事件只计入满足风险门的有效网格输出；未成功返回的执行不计入。 如果有效输出与误差保证一致成立（$\delta_{\rm alg}=0$），$\epsilon_{\rm alg}\le.00022$ 可保留超过 $.66824>2/3$ 的概率下界。这只是条件性敏感性陈述：当前没有建立这样的统一保证。保存输入上的算术外包不能认证 Gaussian 生成、无关变换的累积、理想 iid 抽样律或上述未来程序条件。

本审计是新增的99%家族。将它与附录C、F中的锚点差、中心曲率和已知均值三个家族合并，保守联合覆盖至少为96%；再加入附录H较早的有限选择家族，则至少为95%。这些界只覆盖所列家族，不是论文所有历史实验的整体覆盖率，也不要求家族之间独立。全部保存输入的算术加强和 (D.12) 都不增加统计比较。

\newpage

## D.5. 同一审计事件上的连续区域推论

式 (D.1) 中的 $g$ 正是附录 C 的 $F_{256}(I)-F_{256}(C_b)$。两种定义都将相同的归一权重损失对均匀三位教师标签和同一 Gaussian 银行律取平均，使用已知零均值、单位边际方差、$K=100$、$n=256$ 和相关为 $.9$ 的等相关锚点。与候选无关的无关权重归一化常数在点差及总体 regret 中抵消。$T=6$ 只截断审计统计量，不改变总体目标：式 (D.10) 中的均值尾修正已使 $g_L$ 成为未截断 $g$ 的下界。

式 (C.4)–(C.5) 的半径与总体代价 $c_d=H(b_{100};1-d)d^2/2$ 都是确定性的，不需要额外随机事件。在本节矩审计的同一99%联合事件上，保留其输入、数值与 iid 模型资格，应用该区域不等式得到

$$
\begin{aligned}
\inf_{\substack{C\in\mathcal E_{.1}^{(3)}\\\mathcal R(C)\le10^{-6}}}
\{F_{256}(C)-F_{256}(C_b)\}
&\ge g_L-c_d>.00134,\\
F_{256}(C)-\inf_{C'\in\mathcal E_{.1}^{(3)}}F_{256}(C')<.00134
&\ \Longrightarrow\ \mathcal R(C)>10^{-6}.
\end{aligned}
\tag{D.14}
$$

向外标量算术用已有 $g$ 的下端点和 $c_d$ 的上端点核实上述严格余量。第二行来自总体下确界不大于 $F_{256}(C_b)$。因此，支持新运行概率陈述的审计事件，也支持连续区域蕴涵（两者完整陈述见C.15），其保守 regret 门为 $.00134$。

这是同一事件的确定性推论，没有新增统计比较或alpha分配，也没有给出连续优化器的新运行失败概率。附录 C 中原来约 $.0008952$ 的余量和最初七门六过的记录保持不变。四个或五个指定家族的联合覆盖仍分别为96%和95%，本推论不改变其范围。

\clearpage

# 附录 E 保存审计输入的算术包络

## E.1 三种不同层次的保证

有限总体结果采用理想 iid Gaussian 采样模型。我们将这一概率假设与两个数值问题分开。第一，**保存输入的算术**把每个保存的 binary64 活跃 Gaussian 坐标 $Z_{ki}$ 和无关 log-weight 和 $S_k$ 当成精确实数，再包络对应数学 sigmoid、权重、曲率或评分。第二，**输入变换误差**关心这些保存数对目标输入的近似，包括256项 log-sigmoid 如何生成并累加为 $S_k$。完整算术包络处理第一层，不把第二层误差设为零。理想 iid Gaussian 律仍是明确的统计模型假设；没有声称伪随机实现与该定律等价。

两次确定性计算现已覆盖附录 C 保存的全部中心 $n=256$ 曲率核，以及附录 D 新概率审计保存的全部银行—标签锚点差。它们为这些指定数学量提供数值支持，不保证未来每次原生浮点执行的算术误差；其统计解释继续保留输入与采样资格。

## E.2 统一曲率敏感性

对相关对 $(i,j)$，令 $q_{ki}=\sigma(-Z_{ki})$、$r_{ki}=q_{ki}(2q_{ki}-1)$，$\pi_k\propto e^{S_k}\prod_i\sigma(Z_{ki})$ 表示中心全正标签曲率所用的精确归一化权重。式 (C.14) 的对角 Price 核为

$$
\begin{aligned}
H_{ij}(\pi,q)={}&-\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj}\\
&+2\sum_k\pi_k^2(1-\pi_k)(q_{ki}^2r_{kj}+q_{kj}^2r_{ki})\\
&+\sum_k\pi_k^2(1-4\pi_k)q_{ki}^2q_{kj}^2+P(A,B),\\
A={}&\sum_k\pi_kq_{ki}q_{kj},\quad B=\sum_k\pi_k^2q_{ki}q_{kj},\quad
P(A,B)=A^2-4AB+6B^2.
\end{aligned}
\tag{E.1}
$$

在概率单纯形与 $0\le q\le1$ 上，有 $0\le B\le A\le1$、$|r|\le1$、$|r'|\le3$。下表分别用权重变化的 $\ell_1$ 范数和全部 logistic 输入变化的最大范数，控制各项导数贡献：

|式 (E.1) 中的项|权重常数|Logistic 输入常数|
|---|---:|---:|
|第一局部和|1|6|
|第二局部和|4|5|
|第三局部和|10|12|
|$P(A,B)$|14|12|

权重列使用 $|1-2\pi|\le1$、$|2\pi-3\pi^2|\le1$、$|2\pi-12\pi^2|\le10$。输入列中，$r_ir_j$、$q_i^2r_j+q_j^2r_i$、$q_i^2q_j^2$ 对两坐标的导数绝对值之和分别至多为 $6,10,4$；对应权重和分别至多为 $1,1/2,3$，其中 $2\sum\pi_k^2(1-\pi_k)\le1/2$，$\sum\pi_k^2|1-4\pi_k|\le3$。

整体多项式必须包含 $B$ 对权重平方的依赖。记 $a_k=q_{ki}q_{kj}$，其导数与最终界为

$$
\begin{gathered}
\partial_{\pi_k}P=a_k\{(2-8\pi_k)A+(-4+24\pi_k)B\},\\
\partial_{q_{ki}}P=\pi_kq_{kj}\{(2-4\pi_k)A+(-4+12\pi_k)B\},\\
|H_{ij}(\widetilde\pi,\widetilde q)-H_{ij}(\pi,q)|
\le29\|\widetilde\pi-\pi\|_1+35\|\widetilde q-q\|_\infty.
\end{gathered}
\tag{E.2}
$$

当 $A>0$ 时，各括号除以 $A$，再令 $t=B/A\in[0,1]$。第一个双线性式的四角值为 $2,-6,-2,14$，故绝对值至多14；第二个的四角值为 $2,-2,-2,6$，跨粒子累加 $\pi_kq_{kj}$ 后，每个坐标至多贡献6，再计两个坐标。$A=0$ 时直接成立。沿单纯形与立方体中的线段积分导数界，即证式 (E.2)。同银行内三个相关对求平均仍保留常数29与35，既不将常数乘三，也不增加独立观察。

对 log 权重 $\ell$ 和扰动 $e$，softmax Jacobian 给出 $|e-Ee|$ 的期望。若 $m=\min e_k$、$M=\max e_k$，则 $\operatorname{Var}(e)\le(M-Ee)(Ee-m)\le(M-m)^2/4$。沿 $\ell+te$ 积分得到

$$
\begin{gathered}
\|\operatorname{softmax}(\ell+e)-\operatorname{softmax}(\ell)\|_1
\le\int_0^1E_{\operatorname{softmax}(\ell+te)}|e-Ee|\,dt\\
\le\tfrac12\operatorname{osc}(e)
=\inf_c\|e-c\mathbf1\|_\infty.
\end{gathered}
\tag{E.3}
$$

因此公共 log-weight 偏移精确抵消。这些界要求权重位于精确单纯形，舍入后权重未必满足此前提；区间实现包络精确归一化本身，不假设舍入权重已经精确归一。

## E.3 样本统计量的误差传播

令 $v_b$ 为精确目标银行核，$t_b$ 为保存的浮点参考，且全部 $L$ 个银行均有 $|v_b-t_b|\le\epsilon_b$。取去均值投影 $J=I_L-\mathbf1\mathbf1^T/L$，由无偏样本方差得到的标准差 $s(t)=\|Jt\|_2/\sqrt{L-1}$ 满足

$$
\begin{gathered}
|\bar v-\bar t|\le\bar\epsilon,\qquad
|s(v)-s(t)|\le\frac{\|J(v-t)\|_2}{\sqrt{L-1}}
\le d_s:=\sqrt{\frac{\sum_b\epsilon_b^2}{L-1}},\\
|s(v)^2-s(t)^2|\le2s(t)d_s+d_s^2.
\end{gathered}
\tag{E.4}
$$

所以数值误差同时影响经验 Bernstein 界中的均值与标准差。令 $\bar t_U,s_U,\bar\epsilon_U,d_{s,U}$ 为这些量的可靠上包络，$R$ 为精确目标核的范围长度。在相应经验 Bernstein 事件上，

$$
h\le\bar t_U+\bar\epsilon_U
+\sqrt{\frac{2\log(4/\alpha_j)}L}(s_U+d_{s,U})
+\frac{7R\log(4/\alpha_j)}{3(L-1)}.
\tag{E.5}
$$

下端对称构造。中心曲率使用 $L=262144$、$R=133/16$、$\alpha_j=.01/3$，因此对数因子为 $\log1200$。因为精确目标的样本统计量已被包络，范围仍取该目标的范围；这里没有只给均值加余量而保留未经控制的方差计算。若每银行误差都不超过 $\epsilon$，均值与标准差引入的额外量至多为 $\{1+\sqrt{2\log1200/(L-1)}\}\epsilon$，约为 $1.007355\epsilon$；可靠归约还覆盖其自身舍入。

对评分审计，在其固定截断指示量下，逐标签算术误差至多为 $\eta$ 时，八标签均值误差至多为 $\eta$，两类标签条件均值之差的误差至多为 $2\eta$。若精确截断均值满足 $|X|\le M$，其平方统计量误差至多为 $2M\eta+\eta^2$。这些界可代入式 (E.4) 控制各标量统计；完整评分计算后，附录 D 原数值余量仍保持。

## E.4 完整面板的向外算术

实现假定 IEEE binary64 的加、减、乘、除分别执行并正确舍入，使用渐进下溢、可靠 nextafter 及二进制指数缩放。每次端点运算再向外扩一个可表示步长，归约采用显式成对相加树。制造探针覆盖一般数值、subnormal及非法输入拒绝，但不是这些平台假设的证明。非法区间、非有限输入或丢失正分母下界时明确失败，不删样本。

指数计算先固定一个经 Arb 验证的 $\log2$ binary64 区间。只有整个归约区间 $r=x-k\log2$ 位于 $[-1,1]$ 时才使用该整数 $k$。区间 Horner 计算给出

$$
\begin{gathered}
e^x=2^ke^r,\qquad
e^r\in\left[\sum_{j=0}^{20}\frac{r^j}{j!}-\frac3{21!},\;
\sum_{j=0}^{20}\frac{r^j}{j!}+\frac3{21!}\right],\quad |r|\le1,\\
x\le-800\quad\Longrightarrow\quad 0<e^x<2^{-1074}.
\end{gathered}
\tag{E.6}
$$

Taylor 余项使用 $e<3$；最后一式来自 $\log2<.7$ 和 $800>1074(.7)$，从而得到严格的 subnormal 包络。二的幂缩放后同样向外舍入。由 $u=e^{-|z|}$，两个 sigmoid 符号分别用 $u/(1+u)$ 和 $1/(1+u)$ 计算，避免从一相减。曲率实现直接包络式 (E.1) 的全部项。

正数对数用精确二进制归一化写成 $x=2^em$、$1\le m<2$。令 $u=(m-1)/(m+1)\in[0,1/3)$，使用

$$
\begin{gathered}
\log x=e\log2+2\sum_{j=0}^{19}\frac{u^{2j+1}}{2j+1}+R_{20},\\
0\le R_{20}\le\frac{2u^{41}}{41(1-u^2)}
\le\frac9{164\,3^{41}}.
\end{gathered}
\tag{E.7}
$$

余项由正尾项的几何控制得到。对正区间输入，可利用单调性分别评价端点。这些节点不把普通库 exp/log 的输出作为精确目标；少量常数以及统计归约中的标量 log／平方根操作另用 Arb 包络。

评分审计中的等相关锚点使用精确闭式平方根，不以舍入特征分解作为数学目标。记 $T_C=\sqrt C$、$S_{\max}=\max_kS_k$，有

$$
\begin{gathered}
C_b=\tfrac1{10}I+\tfrac9{10}\mathbf1\mathbf1^T,\qquad
T_{C_b}=aI+b\mathbf1\mathbf1^T,\\
a=\sqrt{1/10},\qquad b=\{\sqrt{14/5}-a\}/3,\\
M_C(y;Z,S)=\sum_ke^{S_k-S_{\max}}\prod_{i=1}^3\sigma\{y_i(T_CZ_k)_i\},\\
d_y(Z,S)=\log M_{C_b}(y;Z,S)-\log M_I(y;Z,S).
\end{gathered}
\tag{E.8}
$$

常数 $a,b$ 使用192-bit Arb 包络。无关权重的公共归一化分母与任何公共权重偏移都在评分差中精确抵消。全部八个标签均被评价，保存的评分差仅用于误差比较。

## E.5 覆盖范围、结果与解释

每次完整计算前固定算术方法、制造检查、源码版本和保存输入身份。这些是对已完成审计追加的事后确定性计算，原统计合同与结果保持。中心曲率银行使用seed202610060301，新概率审计银行使用seed202610060501；两者均由 PCG64DXSM SeedSequence 的role1／2生成活跃／无关正态量，按256银行分块。算术计算没有新增随机数。带索引的lower／upper、原参考值、误差包络和内容哈希支持完整本地复核，并把每次计算绑定到其输入。

|算术目标|保存输入的完整覆盖|最大误差包络，向上舍入|
|---|---:|---:|
|中心 $n=256$ 曲率|262144银行；786432个方向核|每银行平均 $4.382\times10^{-13}$|
|新概率审计的锚点差|262144银行；2097152个银行—标签点差|每点差 $3.331\times10^{-14}$|

记录中的未舍入最大值约为 $4.3818074169088566\times10^{-13}$ 和 $3.33066907387547\times10^{-14}$。曲率平均误差界至多为 $2.268\times10^{-14}$，标准差扰动界至多为 $3.630\times10^{-14}$。在式 (E.5) 中传播这些量并包络全部归约误差后，得到向外的条件区间

$$
h_{256,100}\in[-.004912711427809,\;-.003618108895746].
\tag{E.9}
$$

该区间继续附带声明的输入与采样资格。评分计算在全部保存点上满足原 $10^{-9}$ 逐标签算术余量；原精确标量统计与附录 D 的概率下界均不改变。两次完整计算分别用时126.438秒和339.672秒。

另在32个固定银行上独立用192-bit Arb检查96项 $n=256$ 曲率，用256-bit Arb检查256个标签差，全部参考球都落在完整计算的相应区间中。这些是辅助实现核验；全覆盖来自对每个保存输入执行向外算法，不是由抽查外推。更早的匹配权重点差审计仍保留自身32银行诊断，附录 F 的18分量非对称均值导数审计也不在本次两组覆盖范围内。

两次计算都作用于已有样本，属于确定性加强，不增加alpha。它们不证明确定性的期望符号，也没有给原生算法在未来数据与银行上的统一误差界。

双锚点评分的输入误差接口为

$$
|d_y(\widetilde Z,\widetilde S)-d_y(Z,S)|
\le\operatorname{osc}(\widetilde S-S)
+3(1+\sqrt{2.8})\|\widetilde Z-Z\|_\infty.
\tag{E.10}
$$

每个候选的对数混合量变化均介于最小与最大无关 log-weight 误差之间，由此得到振幅项。矩阵 $T_{C_b}$ 元素非负、行和为 $\sqrt{2.8}$，$\log\sigma$ 与 log-sum-exp 具有所需的单位 Lipschitz 界，所以式 (E.10) 对全部标签与银行成立。它是条件性桥接式，不是对未知输入误差的数值界。未来原生程序的 $\epsilon_{\rm alg}$ 条件另见附录 D，不能仅由保存输入的完整覆盖推出。

\clearpage

# 附录 F 已知均值与观察等价尺度

## F.1 匹配均值与光滑性

固定有限 $K$、三个单位 Gaussian 边际方差，以及教师与候选共享的已知均值 $\mu$。只拟合 $\theta=(c_{12},c_{13},c_{23})$，$C(\theta)$ 是单位对角相关矩阵，教师为 $C_*=I$。令 $p_i=E\sigma(\mu_i+X)$，其中 $X\sim N(0,1)$，并记 $q_i(+1)=p_i$、$q_i(-1)=1-p_i$。归一非负权重 $a_k$ 之和为一，独立于活跃 Gaussian 与标签，其分布不依赖 $(\theta,\mu)$。定义

$$
\begin{gathered}
Q_{I,\mu}(y)=\prod_iq_i(y_i),\\
\Phi_y(X,\mu,a)=-\log\sum_{k=1}^Ka_k\prod_{i=1}^3\sigma\{y_i(\mu_i+X_{ki})\},\\
F_n(\theta,\mu)=\sum_yQ_{I,\mu}(y)E_{a,X}\Phi_y(X,\mu,a),\quad X_k\stackrel{\rm iid}{\sim}N(0,C(\theta)),\\
G_n=\nabla_\theta F_n,\qquad
Q_{C,\mu}(y)=E_{X\sim N(0,C)}\prod_i\sigma\{y_i(\mu_i+X_i)\},\\
\mathcal R(\theta,\mu)=D(Q_{I,\mu}\Vert Q_{C(\theta),\mu}).
\end{gathered}
\tag{F.1}
$$

其中 $n=0$ 对应 $a_k=1/K$，$n=256$ 使用独立无关权重。归一化仅去掉与参数无关的加性项。无需权重逆矩，因为对所有权重统一有

$$
0\le\Phi_y(X,\mu,a)\le3\log2+\|\mu\|_1+\sum_i\max_k|X_{ki}|.
\tag{F.2}
$$

在正定协方差的紧邻域内，Gaussian 密度导数具有多项式乘 Gaussian 的包络。$\log\sigma$ 的每个正阶导数都有界；对数和的导数是这些导数与归一后验权重的有限组合。因此可在期望内进行任意固定阶微分，包括教师概率的微分，得到 $F_n$ 关于两类参数联合 $C^\infty$。这一结论针对已知匹配均值，不包括联合拟合均值或均值估计误差。

对 $D=\operatorname{diag}(d_1,d_2,d_3)$、$d_i\in\{-1,1\}$，令 $S_D=\operatorname{diag}(d_1d_2,d_1d_3,d_2d_3)$。Gaussian 变号与标签重标给出

$$
F_n(S_D\theta,D\mu)=F_n(\theta,\mu),\qquad
G_n(S_D\theta,D\mu)=S_DG_n(\theta,\mu).
\tag{F.3}
$$

在真值处，$G_{n,ij}(0,\mu)$ 对 $\mu_i,\mu_j$ 分别为奇函数，对其余均值为偶函数。两次光滑除法与坐标置换对称性给出

$$
G_{n,ij}(0,\mu)=\mu_i\mu_jb_{n,ij}(\mu)
=b_n\mu_i\mu_j+O(\|\mu\|^4).
\tag{F.4}
$$

每个光滑函数 $b_{n,ij}$ 对所有均值坐标分别为偶函数，且 $b_{n,ij}(0)=b_n$ 相同。因此，至多一个非零均值坐标时，任意有限均值幅度下的 $I$ 都精确驻定；极值类型的保持仅有局部小均值保证。一般匹配均值不保证 $I$ 驻定，附录 G 给出反例，但不解决$K=100$ 时的相应系数。

## F.2 附近驻点分支与精确风险

在 $\mu=0$，附录 C 给出 $G_n(0,0)=0$、$D_\theta G_n(0,0)=h_nI_3$：在 $K=100$，解析地有 $h_0>.00410$；$h_{256}<0$ 依赖中心曲率事件及其数值资格。只要 $h_n\ne0$，隐函数定理就给出真值邻域内唯一的光滑驻点分支，Hessian 连续性保持其严格局部极小或极大类型。符号等变性与局部唯一性进一步给出

$$
\begin{gathered}
\theta_{n,ij}(\mu)=\mu_i\mu_jt_{n,ij}(\mu)
=-\frac{b_n}{h_n}\mu_i\mu_j+O(\|\mu\|^4),\\
\theta_n(\mu)=O(\|\mu\|^2).
\end{gathered}
\tag{F.5}
$$

邻域可取在 $C\succ.1I$ 内。唯一性是局部的，邻域不必对所有 $K$ 一致，也不认证指定的中等均值落在其中。

令 $v_i=E\sigma'(\mu_i+X)>0$。Price 恒等式在 $I$ 处给出概率导数 $y_iy_jv_iv_j\prod_{\ell\ne i,j}q_\ell(y_\ell)$。教师乘积律下有 $E[Y_i/q_i(Y_i)]=0$ 和 $E[\{Y_i/q_i(Y_i)\}^2]=1/[p_i(1-p_i)]$。因此精确观察风险的 Hessian 是对角 Fisher 信息：

$$
\begin{gathered}
\nabla_\theta^2\mathcal R(0,\mu)=\operatorname{diag}_{i<j}r_{ij}(\mu),\qquad
r_{ij}(\mu)=\frac{v_i^2v_j^2}{p_i(1-p_i)p_j(1-p_j)}>0,\\
\mathcal R(\theta,\mu)=\frac12\sum_{i<j}r_{ij}(\mu)\theta_{ij}^2+O(\|\theta\|^3),\\
\mathcal R(\theta_n(\mu),\mu)
=\frac{r_*}{2}\left(\frac{b_n}{h_n}\right)^2\sum_{i<j}\mu_i^2\mu_j^2
+O(\|\mu\|^6)=O(\|\mu\|^4),\\
r_*=16\{E\sigma'(X)\}^4.
\end{gathered}
\tag{F.6}
$$

系数 $b_n$ 尚未证明非零。风险阶数是上界，不能写成 $\Theta(\|\mu\|^4)$；均值坐标轴上的风险精确为零。更一般地，每个可行 $C\ne I$ 都有严格正风险：二元概率 $E[\sigma(\mu_i+X_i)\sigma(\mu_j+X_j)]$ 关于相关的导数 $E[\sigma'(\mu_i+X_i)\sigma'(\mu_j+X_j)]$ 严格为正，所以非零相关会改变某个二元观察边际。

## F.3 非对称均值处的完整导数审计

固定点 $\mu_*=(-.2,.1,.3)$、$K=100$ 使用 262,144 个全新 iid 银行，两种权重条件共享活跃 Gaussian。均值、粒子数与银行数均未依据这些结果选择。教师概率约为 $(.458758831114,.520651702363,.561706803459)$，以 128-bit Acb 在 $[-12,12]$ 上积分，加 Mills 尾界包络，并与 GH64/96 交叉核验。每银行都按教师概率平均**全部八个标签**，这里不能用单个全正标签替代。

固定 $y$，令 $x_{ki}=\mu_i+Z_{ki}$，$\pi_k$ 正比于 $a_k\prod_i\sigma(y_ix_{ki})$。定义

$$
\begin{gathered}
q_{ki}=y_i\sigma(-y_ix_{ki}),\quad
r_{ki}=\sigma(-y_ix_{ki})\{2\sigma(-y_ix_{ki})-1\},\\
A_p=\sum_k\pi_kq_{ki}q_{kj},\quad B_p=\sum_k\pi_k^2q_{ki}q_{kj},\quad p=(i,j).
\end{gathered}
\tag{F.7}
$$

Price 算子 $L_p=\sum_k\partial_{x_{ki}}\partial_{x_{kj}}$ 没有额外的 $1/2$ 因子。它给出下列梯度核，以及不同相关对 $p=(a,b)$、$q=(a,c)$ 的非对角 Hessian 核：

$$
\begin{aligned}
\mathsf G_p={}&-\sum_k\pi_k(1-\pi_k)q_{ki}q_{kj},\\
\mathsf H_{pq}={}&\sum_kq_{kb}q_{kc}\left[-\pi_k(1-\pi_k)(1-2\pi_k)r_{ka}
+3\pi_k^2(1-2\pi_k)q_{ka}^2\right]\\
&+A_pA_q-2(B_pA_q+A_pB_q)+6B_pB_q.
\end{aligned}
\tag{F.8}
$$

对角核采用式 (C.14) 的多项式，只需代入此处有符号的 $q_{ki}$。式 (F.8) 来自负对数和的四阶导数分割，再合并同粒子与异粒子项。由 $|q_{ki}|\le1$、$r_{ki}\in[-1/8,1]$，梯度范围是 $[-1,1]$，对角核范围是 $[-33/8,67/16]$。每个相关对的全部 $q_{ki}q_{kj}$ 都具有相同标签符号，故 $B_p=sA_p$、$s\in[0,1]$。由 $|1-2s-2t+6st|\le3$，式 (F.8) 最后三项的绝对值至多为 $3$，局部求和的绝对值至多为 $4$，从而 $\mathsf H_{pq}\in[-7,7]$。标签平均保留这些界。

两臂各三个梯度与六个唯一 Hessian 元素，共用一个 $\alpha=.01$ 的 18 分量家族。记银行无偏方差为 $s_j^2$、范围长度为 $L_j$、$N=262144$，经验 Bernstein 区间为

$$
\widehat m_j\ \pm\left\{\sqrt{\frac{2s_j^2\log7200}{N}}
+\frac{7L_j\log7200}{3(N-1)}+\varepsilon_j\right\},\qquad
L_j\in\{2,133/16,14\}.
\tag{F.9}
$$

其中 $\varepsilon_j=10^{-9}+M_j\delta_Q$，$M_j\in\{1,67/16,7\}$ 是核绝对值上界，$\delta_Q\le5.56\times10^{-16}$ 是教师概率舍入的误差界。独立单位是银行，不能把标签、坐标或两臂视为额外独立样本。全部 18 个区间如下，端点向外舍入至小数点后 12 位；Hessian 坐标顺序为 $(12,13,23)$。

|分量|$n=0$ 区间|$n=256$ 区间|
|---|---|---|
|$g_{12}$|[-0.000184565302, 0.000179529158]|[-0.000253339714, 0.000229858472]|
|$g_{13}$|[-0.000195341459, 0.000168423016]|[-0.000270877759, 0.000210500179]|
|$g_{23}$|[-0.000177779284, 0.000186080091]|[-0.000235312039, 0.000246075882]|
|$H_{12,12}$|[0.027349272457, 0.028693509363]|[-0.004963286860, -0.003481915149]|
|$H_{13,13}$|[0.027054242599, 0.028398354055]|[-0.004906087758, -0.003426349185]|
|$H_{23,23}$|[0.027163138163, 0.028507306983]|[-0.004915420006, -0.003435231666]|
|$H_{12,13}$|[-0.001116578916, 0.001113107665]|[-0.001128737702, 0.001143888725]|
|$H_{12,23}$|[-0.001109989535, 0.001119719329]|[-0.001123088450, 0.001149345972]|
|$H_{13,23}$|[-0.001113738311, 0.001115930316]|[-0.001136064635, 0.001136714384]|

用完整、未舍入的同时区间计算 Gershgorin 界，得到

$$
\lambda_{\min}(H_0)\ge .02482173336837,\qquad
\lambda_{\max}(H_{256})\le-.00114574607762.
\tag{F.10}
$$

六个梯度区间都含零，既未证明零，也未证明非零，不能据此认定 $\mu_*$ 处的 $I$ 是极值点。在 32 个预定银行上，用 128-bit Arb 重算了 576 个导数分量，最大差异为 $2.19\times10^{-16}$。这些检查支持显式数值余量，但不构成所有 Gaussian、log-sigmoid 或权重计算的全局浮点误差证明；总体结论保留该数值资格。

## F.4 风险冲突与尺度连续性

真值处的负定性使某个邻域内有 $\nabla_\theta^2F_{256}\preceq-\lambda I_3$、$\lambda>0$。记 $g=G_{256}(0,\mu)$，得到

$$
\begin{gathered}
F_{256}(\theta,\mu)-F_{256}(0,\mu)
\le g^\top\theta-\frac\lambda2\|\theta\|^2,\\
F_{256}(\theta,\mu)+F_{256}(-\theta,\mu)-2F_{256}(0,\mu)
\le-\lambda\|\theta\|^2<0\quad(\theta\ne0).
\end{gathered}
\tag{F.11}
$$

因此，穿过 $I$ 的每条足够短的无向直线至少一侧降低有限目标，两侧的非零扰动都提高精确风险。只有 $g=0$ 时，才能保证每个方向都下降。该结论在新事件上适用于 $\mu_*$，也在中心事件上适用于充分小的均值，不给优化器保证。

固定 $K=100$，保留全部 256 个无关观察，将其 logit 改为 $\tau G_j$，$0\le\tau\le1$。每个观察仍是独立公平位，故

$$
\begin{gathered}
Q_{C,\mu,\tau}(y,u)=Q_{C,\mu}(y)2^{-256},\\
a_k(\tau)=\frac{\prod_{j=1}^{256}\sigma(\tau G_{kj})}
{\sum_\ell\prod_{j=1}^{256}\sigma(\tau G_{\ell j})}.
\end{gathered}
\tag{F.12}
$$

$\tau=0$ 时权重均匀，但观察仍然存在。逐点连续性与不依赖权重的包络使相关 Hessian 关于 $\tau$ 连续。$\mu=0$ 时 Hessian 为 $h(\tau)I_3$，有 $h(0)>.00410$，且在中心事件上 $h(1)<0$，因此内部至少有一个零曲率尺度。对充分小的一般均值，或在自身事件上的 $\mu_*$，两端 Hessian 分别正定、负定，所以连续特征值必然过零。真值不驻定时，这是 Hessian 符号类型的变化，不能称为驻点分岔。这里没有尺度扫描、零点定位、唯一性、单调性或全路径驻点分支延拓结论。

匹配权重总体审计、中心曲率审计与非对称均值审计各有一个 99% 家族事件，三者保守联合覆盖至少 97%；加入更早的 $K=2$ 有限选择家族后至少为 96%，无需假定独立。风险、分支与尺度的确定性推论使用各自已有事件，不增加概率预算。这些概率表述均保留对应审计的数值资格。

\clearpage

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

\clearpage

# 附录 H. 早期有限选择及其负对照

## H.1. 选择条件与区域证书的职责

对$\mathcal M$的邻域$U$，定义$\Delta_U=\inf_{C\notin U}V(C)-\min_CV(C)>0$；令$e=\sup_C|F(C)-F(I)-wV(C)|$、$\zeta=\operatorname{osc}_C(\widehat F-F)$、$\eta=\widehat F(\widehat C)-\inf_C\widehat F(C)$。与任一$V$极小点比较，得到常见充分条件

$$
2e+\zeta+\eta<w\Delta_U\ \Longrightarrow\ \widehat C\in U.
\tag{H.1}
$$

因为$U$外输出相对该极小点的经验差至少为$w\Delta_U-2e-\zeta$，与$\eta$上界矛盾。仅写条件并不产生有限数值证书。本研究直接给整片低风险区域上的经验准则下界，再单独审计有限总体比较；不从随机重启或网格最小值估计全局$e$、$\zeta$或优化regret。

## H.2. 连续低风险集合的同时下界

本节固定$m=3$、$\mu=0$、$C_*=I$、$\delta=.1$及$K=2$。令$q_y$为任意经验活跃标签频率，$Z_{bk}\in\mathbb R^3$及正无关权重$a_{bk}$为任意固定有限粒子库输入。对称根$S(C)=C^{1/2}$耦合候选。只去掉与候选无关的加法常数后，

$$
\widehat F(C)=\frac1B\sum_{b=1}^{B}\sum_yq_y\phi_y(S(C)Z_b),\qquad
\phi_y(X)=-\log\sum_{k=1}^K a_{bk}\prod_{i=1}^m\sigma(y_iX_{ki}),\qquad
a_{bk}=\frac{e^{S_{bk}^{\rm nuis}}}{\sum_\ell e^{S_{b\ell}^{\rm nuis}}}.
\tag{H.2}
$$

无关权重仍是随机似然因子，没有像FACT那样解析移除。取$r_0=10^{-6}$，令$\mathcal G_{r_0}=\{C\in\mathcal E_{.1}^{(3)}:\mathcal R(C)\le r_0\}$。Price、$\log\cosh u\le u^2/2$和Jensen给$\kappa'(c)\ge\beta=e^{-1/2}/4$。对$y_iy_j$使用Pinsker，并用真实pair矩为零，得到$|\kappa(c_{ij})|\le\sqrt{2r_0}$。因此

$$
C\in\mathcal G_{r_0}\ \Longrightarrow\
\|C-I\|_F\le d:=\frac{\sqrt{12r_0}}{\beta}=.02284535206697714\ldots,
\qquad\lambda:=1-d>0.
\tag{H.3}
$$

对$D=C-I$，$I+tD$上各矩阵最小特征值至少为$\lambda$。根身份求导给$SS'+S'S=D$和$SS''+S''S=-2(S')^2$；Sylvester逆算子界给

$$
\|S'\|_F\le\frac{\|D\|_F}{2\sqrt\lambda},\qquad
\|S''\|_F\le\frac{\|D\|_F^2}{4\lambda^{3/2}},\qquad
\|\nabla_X\phi_y\|_2\le\sqrt m,\quad
\|\nabla_X^2\phi_y\|_{\rm op}\le m+\tfrac14.
\tag{H.4}
$$

最后一界由加权发射Hessian（范数至多$1/4$）与log-sum-exp梯度协方差（范数至多$m$）得到。若$a_b^2=\sum_k\|Z_{bk}\|^2$、$A_2=B^{-1}\sum_ba_b^2$，链式法则再用$B^{-1}\sum_b a_b\le\sqrt{A_2}$控制二阶方向导数。记$g$为$I$处三个自由坐标导数，Taylor公式给

$$
H(A_2)=\frac{(m+1/4)A_2}{4\lambda}
+\frac{\sqrt{mA_2}}{4\lambda^{3/2}},\qquad
L_r=\frac{\|g\|_2d}{\sqrt2}+\frac{H(A_2)d^2}{2},\qquad
\inf_{C\in\mathcal G_{r_0}}\widehat F(C)\ge\widehat F(I)-L_r.
\tag{H.5}
$$

因子$1/\sqrt2$将自由非对角坐标转换为Frobenius范数。该界同时覆盖连续低风险集合的每个矩阵，无论是否落在搜索网格。于是，

$$
\widehat g:=\widehat F(I)-\widehat F(C_b)-L_r>0,\qquad
\widehat F(\widehat C)\le\widehat F(C_b)
\quad\Longrightarrow\quad \mathcal R(\widehat C)>r_0.
\tag{H.6}
$$

## H.3. 单独的有限总体比较

真实均匀标签分布及Gaussian先验使$F(C)$对$C$的同时符号共轭不变，故两无关条件下$I$的自由坐标梯度都为零。对(H.4)取期望，用$Ea_b^2=Km=6$和$Ea_b\le\sqrt6$，得到$A_2=6$时的界：

$$
\inf_{C\in\mathcal G_{r_0}}F(C)-F(C_b)
\ge g-\frac{H(6)d^2}{2},\qquad
g=F(I)-F(C_b),\qquad \frac{H(6)d^2}{2}=.00158844598226789635\ldots.
\tag{H.7}
$$

独立审计只估这一预定点差；用真实$1/8$权重精确平均八个活跃模式，这些权重不进入学习器。若$D_b=h_b(I)-h_b(C_b)$、$A=\|C_b^{1/2}-I\|_{\rm op}=\max(1-\sqrt{.1},\sqrt{2.8}-1)$，Lipschitz性给$|D_b|\le\sqrt3 A\max_k\|Z_{bk}\|$。取$T=8$、$E_b=\{\max_{k,i}|Z_{bki}|\le T\}$、$D_b^t=D_b1_{E_b}$及$M=3AT$，Gaussian Mills和Cauchy界给

$$
|D_b^t|\le M=16.410533615595888\ldots,\qquad
|ED_b-ED_b^t|\le t_T:=\sqrt3 A\sqrt{6\cdot12\varphi(T)/T}
<2.526\times10^{-7}.
\tag{H.8}
$$

对$L=2097152$个独立银行，缩放后的双侧经验Bernstein不等式[@maurer2009, Theorem 4]给下列半宽。$s^2$为$D_b^t$的无偏样本方差，$\alpha_j=.005$；在iid抽样模型及所述数值余量下，对两条件联合的总错误概率至多$.01$：

$$
b_L=\sqrt{\frac{2s^2\log(4/\alpha_j)}{L}}
+\frac{7(2M)\log(4/\alpha_j)}{3(L-1)}+t_T+10^{-9},
\qquad g\in[\overline D^t-b_L,\overline D^t+b_L].
\tag{H.9}
$$

## H.4. 固定程序、输入与完整结果

单份训练数据含独立教师生成的65536个iid三位观测。无关标签用Gaussian符号对称性积分，而每个银行仍保留$n$个随机无关似然因子。每个相关坐标取$\{-9,-6,-3,0,3,6,9\}/10$；以整数检查$10(C-.1I)$全部主子式，剩143个可行候选。double最小分数$10^{-12}$以内的词典序首点作提案；区间比较确认其不差于固定参照$(.9,.9,.9)$，否则返回参照。输出锁定后才算风险和谱；不以真实相关方向或风险oracle选择候选。

PCG64DXSM SeedSequence为`[202610060101,role]`：role0/1/2依次为训练标签、活跃Gaussian银行和无关Gaussian因子，3/4为独立审计的活跃/无关Gaussian。按[银行,粒子,坐标]顺序，每块4096生成。两条件共享标签和活跃prior；$n=0$无关权重设为$1/2$，$n=256$在同活跃输入上保留256个独立无关因子。训练银行65536，独立审计银行2097152。执行后没有更换种子、维数、粒子数、风险阈值或统计家族。共享输入只计一次：4325376个prior向量，各含3个活跃和256个无关Gaussian坐标，以及65536个活跃观测模式；无神经训练或梯度优化更新。

实现的界如下，单位为nats/观测：

|n|Empirical point gap|Linear cost|Quadratic cost|Empirical region lower|
|---|---|---|---|---|
|0|.003594355617|.000015488961|.001585398366|.001993468290|
|256|.004741770692|.000019724039|.001585398366|.003136648287|

独立点差审计给出：

|n|Population point mean|99% family interval|Population region lower|
|---|---|---|---|
|0|.002478573642|[.001864904388, .003092242896]|.000276458406|
|256|.003266233361|[.002280561286, .004251905435]|.000692115304|

显示值作了舍入，所有正性判定使用未舍入区间。两提案本来就等于$C_b$，无回退或比较后的改选。两输出谱为$(.1,.1,2.8)$，数值观察KL为$.031103092779$，真实独立参照风险为零。解析证书只证明$\mathcal R>10^{-6}$，不证明$.0311$的紧下界。尤其$n=0$控制也排除了低风险区域。本研究为指定程序分辨有限错误选择，但没有识别新增无关因素为必要原因，没有确定完整全局最小集，也未建立常见随机优化器失效。

## H.5. 数值核验与保留的故障

制造输入将double评分和自由坐标导数与128-bit Arb球算术比较，评分/梯度差小于$3\times10^{-17}$，中心有限差分梯度误差小于$2.4\times10^{-11}$。保存训练库的每个银行均在$I$、$C_b$和返回点作球算术重算，并重算$I$梯度。对该实现程序证书，binary64银行输入作为精确数处理。对正定的舍入对称根$S_0$，球残差界给$\|S_0-C^{1/2}\|_F\le\|S_0^2-C\|_F/\sqrt\delta$；分数余量至多为该界乘$\sqrt{mA_2}$。$S_0$正定性由Sylvester主子式核对。

总体审计保存全部配对银行差并重算均值/方差，每条件预定32个等距银行以Arb核对。它们为(H.9)的$10^{-9}$显式数值余量提供资格检查，但不证明全局浮点误差界，也不是每个总体节点的区间求积；因此总体覆盖声明保留这一数值条件。确定性经验区域证书不依赖这项抽查。

选择后，观测概率使用精确的中心化三自旋表达$Q_C(y)=\{1+\sum_{i<j}\kappa(c_{ij})y_iy_j\}/8$，奇数阶自旋矩由对称性为零。pair Gaussian求积32/48/64阶另对照独立三维32/48阶；最大概率差小于$3.21\times10^{-13}$，风险变化小于$3.23\times10^{-13}$，质量误差小于$1.12\times10^{-16}$。这是数值资格，不是严格Gaussian积分余项证书；区域证明依赖Price与Pinsker，而不是求积误差声明。

原运行完成数据、网格提案、总体逐银行差和全量区间选择证书后，在方差汇总的NumPy布尔值写盘处失败。另行冻结的接续修正标量类型，仅以同一银行和已锁定选择完成未保存的诊断。原错误、保存输入/输出及恢复成本均保留，未新增数据、prior或学习输出。计算使用NumPy/SciPy double，球算术使用python-flint0.8.0的128-bit Arb。原无关Gaussian坐标可由固定RNG重建，其充分log和、活跃银行、标签和逐银行差保存在本地；哈希不能替代这些数组。

\clearpage

# 附录 I. 无指定可删块的联合学习与局部稳定性

本附录证明两件相连但不同的事：有限目标在完整参数空间中有严格局部极小；从同一根沿保持候选边际的角路径，仍能降低联合风险而提高有限目标。先由I.1–I.3的[模型](#dense-model)、[完整导数](#full-derivatives)和[Laplace表示](#laplace-representation)固定对象，再由I.8建立[严格局部见证](#proof-theorem-3)，I.10证明[同根固定边际角路径冲突](#same-root-angle)。I.11给出[明确教师族中的定量持续性](#quantitative-teacher-family)。I.9的[定性延续](#local-persistence)针对原径向风险方向，允许更一般的教师扰动。I.4–I.7的学习与重拟合比较为这些结论提供数值背景；经验拟合和K11/K12的结论仍属数值层级。

## I.1. 模型、目标与弱载荷接口 {#dense-model}

本附录研究一个有限维模型，与定理1–2的无关维度增长设定不同。取$m=8$、$Z\sim N(0,I_2)$，全部偏置和载荷均可变化：

$$
\begin{aligned}
A_y(z;b,W)&=\prod_{j=1}^m\sigma(b_j+W_jz)^{y_j}
[1-\sigma(b_j+W_jz)]^{1-y_j},\qquad y\in\{0,1\}^m,\\
p_\theta(y)&=E_ZA_y(Z;\theta),\qquad
F_K^q(\theta)=\sum_yq_yE\left[-\log\left\{K^{-1}\sum_{k=1}^KA_y(Z_k;\theta)\right\}\right],\\
F_\infty^q(\theta)&=-\sum_yq_y\log p_\theta(y),\qquad
R(\theta)=D_{\mathrm{KL}}(q_*\Vert p_\theta).
\end{aligned}
\tag{I.1}
$$

对候选参数求导时，目标$q$固定：它可以是经验状态频率，也可以是教师分布$q_*$。教师满足$b_*=0$，载荷行为$W_{*,j}=.8(\cos\phi_{*,j},\sin\phi_{*,j})$，其中$\phi_*=(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$。所有行非零，所有非对角载荷内积均为正；Gaussian logistic的单调性还给出非零的观察两两依赖。这里没有指定的独立观察块可供删除。Bernoulli潜变量建模已有基础[@tipping1998]；本附录的贡献针对具体目标几何，不是新模型或一般潜变量塌缩现象[@burda2016; @lucas2019; @urban2021]。

由于$\operatorname{rank}(WW^\top)\le2$，这个参数类不同于定理1–2中施加正谱下界的完整相关矩阵类。其谱结论和无关维度增长速率均不移用到本模型。

仅在弱载荷计算中使用自旋$Y_j\in\{-1,1\}$、零偏置、$W_j^\top=\varepsilon v_j(\phi)$及$v_j=s_j(\cos\phi_j,\sin\phi_j)^\top$，其中$s_j>0$固定。记$c=(v_i^\top v_j)_{i<j}$，教师使用相同的$\varepsilon,s$及角度$\phi_*$。对固定$m,K$，在紧角度邻域上连同两阶角度导数一致地有

$$
\begin{aligned}
F_{K,\varepsilon}(\phi)-F_{K,\varepsilon}(\phi_*)
&=\varepsilon^4\{a_K(\|c\|^2-\|c_*\|^2)-b_Kc_*^\top(c-c_*)\}+O(\varepsilon^6),\\
a_K&=\frac{(K-1)(K-2)}{32K^2},\qquad b_K=\frac{K-1}{16K}.
\end{aligned}
\tag{I.2}
$$

沿通过教师的光滑路径，记$\dot c=dc/dt|_0$。相应风险和方向导数为

$$
\begin{aligned}
R_\varepsilon(\phi)&=\frac{\varepsilon^4}{32}\|c-c_*\|^2+O(\varepsilon^6),\\
F_{K,\varepsilon}'(0)&=-\frac{K-1}{8K^2}\varepsilon^4c_*^\top\dot c+O(\varepsilon^6),\\
R_\varepsilon'(0)&=0,\qquad
R_\varepsilon''(0)=\frac{\varepsilon^4}{16}\|\dot c\|^2+O(\varepsilon^6).
\end{aligned}
\tag{I.3}
$$

下面给出系数推导。令$x_j=v_j^\top Z$，展开$2^mA_y=1+\varepsilon h_1+\varepsilon^2h_2+\varepsilon^3h_3+\cdots$：

$$
\begin{aligned}
h_1&=\tfrac12\sum_jy_jx_j,\qquad h_2=\tfrac14\sum_{i<j}y_iy_jx_ix_j,\\
h_3&=\tfrac18\sum_{i<j<\ell}y_iy_jy_\ell x_ix_jx_\ell-\tfrac1{24}\sum_jy_jx_j^3,\\
[-\log(1+\textstyle\sum_{r\ge1}\varepsilon^r\bar h_r)]_{\varepsilon^4}
&=-\bar h_4+\bar h_1\bar h_3+\tfrac12\bar h_2^2-\bar h_1^2\bar h_2+\tfrac14\bar h_1^4.
\end{aligned}
\tag{I.4}
$$

横线表示对$K$个粒子取平均。均匀自旋正交性与Gaussian四阶矩给出每个$c_{ij}^2$的下列系数；仅依赖固定行范数的项在差分中消去。

|式（I.4）中的项|$c_{ij}^2$的系数|
|---|---:|
|$-E\bar h_4$|$0$|
|$E\bar h_1\bar h_3$|$0$|
|$E\bar h_2^2/2$|$(K+1)/(32K)$|
|$-E\bar h_1^2\bar h_2$|$-(K+1)/(8K^2)$|
|$E\bar h_1^4/4$|$3/(16K^2)$|

特别地，

$$
\begin{aligned}
E(\overline{x_ix_j})^2&=(1+K^{-1})c_{ij}^2+K^{-1}s_i^2s_j^2,\\
E(\bar x_i\bar x_j\overline{x_ix_j})&=K^{-2}\{s_i^2s_j^2+(K+1)c_{ij}^2\},\\
E_Z[-\bar h_2+\bar h_1^2/2]
&=\frac{\sum_js_j^2}{8K}-\frac{K-1}{4K}\sum_{i<j}y_iy_jc_{ij}.
\end{aligned}
\tag{I.5}
$$

教师矩$E(Y_iY_j)=\varepsilon^2c_{*,ij}/4+O(\varepsilon^4)$给出$b_K$项。展开有限字母表上的KL并使用同一自旋正交性，即得（I.3）。在$K=2$时，精确的双粒子log-cosh身份独立给出两个$\|c\|^2$系数的相消，确认$a_2=0$、$b_2=1/32$。

这个余项不是对无界Gaussian输入逐点一致的Taylor界。log-sigmoid的实导数有界；log-sum-exp的导数是权重位于$[0,1]$的加权累积量。在所需的六阶载荷导数及两阶角度导数范围内，紧参数集上可用$\max_k\|Z_k\|$的多项式支配。Gaussian矩保证可以交换求导与期望。同时反转全部潜粒子使目标关于$\varepsilon$为偶函数。这些事实给出所述余项；没有提供显式六阶常数、随$K$增长的一致性或适用于$\varepsilon=.8$的阈值。

当$K>2$时，(I.2)平方完成后，除去与候选角度无关的项，得到$\varepsilon^4a_K\|c-\{K/(K-2)\}c_*\|^2+O(\varepsilon^6)$：这个固定行长弱载荷近似倾向拟合放大的成对内积。对文中教师角度和$s_j=1$，令$L_*$为边权$\cos(\phi_{*,i}-\phi_{*,j})$的Laplacian。共线候选在$\mathbf1^\perp$上的首项角度Hessian为$\varepsilon^4\{b_KL_*-2a_K(8)I\}$；最小特征值的形式变号边界为$K=16/(8-\lambda_{\min}(L_*|_{\mathbf1^\perp}))\simeq4.32690638$，由一个固定特征分解得到，没有扫描$K$。这是固定等行长、共同载荷充分小时首项系数的实数边界，不是载荷$.8$下的定量预测。I.6中通过资格的11/12分离对应完整有限载荷目标，且每个$K$都重新拟合偏置和行长。弱展开提供偏好直觉，尚未定量预测重新拟合后的转变；两个数字都不确定普适临界粒子数。

取$s_j=1$、$\phi(t)=(1-t)\phi_*$，则所有$c_{*,ij}>0$且$\dot c_{ij}=(\phi_{*,i}-\phi_{*,j})\sin(\phi_{*,i}-\phi_{*,j})>0$。因此足够弱的载荷下，有限目标下降而局部风险增加。$K=1$对角度精确不变。有限$\varepsilon$的核对保留两个未通过精度门的情形：

|$\varepsilon$|$F_2(.05)-F_2(0)$|$R(.05)$|$F_2'(0)$|GH16/20导数差|资格|
|---:|----------:|----------:|----------:|----------:|--------|
|.125|$-2.46194897\,10^{-6}$|$1.53201221\,10^{-7}$|$-5.01126089\,10^{-5}$|$9.92045\,10^{-18}$|通过|
|.25|$-3.78093733\,10^{-5}$|$2.20782999\,10^{-6}$|$-7.68065055\,10^{-4}$|$1.01103\,10^{-12}$|通过|
|.5|$-4.60084001\,10^{-4}$|$2.51342565\,10^{-5}$|$-.00932588949$|$1.40079\,10^{-7}$|未通过|
|.8|$-.00187535140$|$1.00148195\,10^{-4}$|$-.0379671263$|$2.04217\,10^{-5}$|未通过|

这些结果使用GH阶数8/12/16/20的完整四维$K=2$银行；导数与配对损失的资格要求16/20相邻阶差不超过$10^{-7}$。独立的二维教师PMF核对在$.8$通过，因而允许继续学习，但不使该点的四维角度计算获得资格。

## I.2. 完整导数与条件性局部类型准则 {#full-derivatives}

::: {#full-hessian-criterion}

**命题3（秩一Hessian结构）。** 固定$q$与$K$，并允许积分目标，令$W=(a,0)$且$a_j>0$。在坐标$(b,u,v)$中，记$H_{ba}$为偏置与第一列载荷块、$H_\perp$为第二列载荷块、$g_a=\partial_a F_K^q$，以及$D_a=\operatorname{diag}(a)$，则

$$
\begin{aligned}
H_{\rm full}&=\operatorname{diag}(H_{ba},H_\perp),\\
H_\phi&=D_aH_\perp D_a-\operatorname{diag}(a\odot g_a),\\
H_\perp a&=g_a,\qquad H_\phi\mathbf1=0.
\end{aligned}
$$

:::

回到二元位编码。在$W_j=(a_j,0)$、$a_j>0$处，记

$$
\begin{aligned}
p_k&=\sigma(b+az_k),\quad r_k=y-p_k,\quad D_k=\operatorname{diag}\{p_{kj}(1-p_{kj})\},
\quad\pi_k=\frac{A_y(z_k)}{\sum_\ell A_y(z_\ell)},\\
G_k&=\binom{r_k}{z_kr_k},\qquad
B_k=\begin{pmatrix}D_k&z_kD_k\\z_kD_k&z_k^2D_k\end{pmatrix}.
\end{aligned}
\tag{I.6}
$$

下列期望包括$y\sim q$及独立同分布粒子。按$(b,a)$坐标顺序对log-sum-exp求导得到

$$
\begin{aligned}
g_{ba}&=-E\sum_k\pi_kG_k,\\
H_{ba}&=E\sum_k\pi_kB_k-E\sum_k\pi_k(1-\pi_k)G_kG_k^\top\\
&\quad+E\sum_{k\ne\ell}\pi_k\pi_\ell G_kG_\ell^\top.
\end{aligned}
\tag{I.7}
$$

最后一项不可省略：负Hessian等于加权条件负Hessian减去粒子得分协方差。跨粒子外积必须先于标签平均形成。

在完整坐标$(b,u,v)$下，$W_j=(u_j,v_j)$。第二Gaussian坐标的反射给出$F(b,u,v)=F(b,u,-v)$。在$(u,v)=(a,0)$处，

$$
\begin{aligned}
g_v&=0,\qquad H_{(b,u),v}=0,\qquad
H_{\rm full}=\operatorname{diag}(H_{ba},H_\perp),\\
H_\perp&=E\left[\sum_k\pi_kD_k-\sum_k\pi_k(1-\pi_k)r_kr_k^\top\right].
\end{aligned}
\tag{I.8}
$$

独立、中心化的第二粒子坐标只消去横向块中的跨粒子项。保持行范数的角度路径还具有二阶变化$\ddot W_{j1}=-a_j\dot\phi_j^2$，故

$$
H_\phi=\operatorname{diag}(a)H_\perp\operatorname{diag}(a)
-\operatorname{diag}(a\odot g_a),\qquad H_\perp a=g_a,\qquad H_\phi\mathbf1=0.
\tag{I.9}
$$

完整起见，对下式应用Gaussian分部积分：

$$
\begin{aligned}
\partial_{z_k}(\pi_kr_{ki})
&=\pi_k(1-\pi_k)(a^\top r_k)r_{ki}-\pi_ka_i(D_k)_{ii},\\
E[z_k\pi_kr_{ki}]&=E[\partial_{z_k}(\pi_kr_{ki})].
\end{aligned}
\tag{I.10}
$$

这证明了（I.9）的Ward关系。等价的有符号Laplacian为

$$
\begin{aligned}
w_{ij}&=a_ia_jE\sum_k\pi_k(1-\pi_k)r_{ki}r_{kj},\qquad i\ne j,\\
(H_\phi)_{ij}&=-w_{ij},\qquad (H_\phi)_{ii}=\sum_{j\ne i}w_{ij},\\
d^\top H_\phi d&=\sum_{i<j}w_{ij}(d_i-d_j)^2.
\end{aligned}
\tag{I.11}
$$

边权可以为负；仅使行和结构性地等于零，不能验证求积。$K=1$时$H_\phi=0$是精确身份。

在**真正**的驻点$g_b=g_a=0$处，令$V$的列为$a^\perp$的一组正交归一基。充分条件为

$$
\lambda_{\min}(H_{ba})>0,\qquad
\lambda_{\min}(V^\top H_\perp V)>0.
\tag{I.12}
$$

切片$a^\top v=0$与共同旋转轨道横截。由（I.8），条件（I.12）使切片上的Hessian正定，从而得到模去旋转后的严格局部极小；$m=8$时有23个正曲率的非旋转方向。横向负方向则给出局部下降。离散反射不增加连续零模。这些条件性结论不会把一个很小的浮点梯度变成经认证的精确驻点。I.8为总体K8问题补上误差界与根包含；四个经验拟合及K11/K12延续仍保持数值证据层级。紧邻域上的Gaussian多项式支配保证上述求导合法；每个积分标签概率为正，对$F_\infty^q$也可作相应的局部论证。

## I.3. 单粒子Laplace计算与配对差 {#laplace-representation}

对每个标签取任意$c_y>0$，令$x=A_y/c_y$、$u=e^v$，定义

$$
\Psi_y(u)=E e^{-ux},\qquad B_{0,y}(u)=E[xe^{-ux}],\qquad M_{G,y}(u)=E[xe^{-ux}G].
\tag{I.13}
$$

利用$S^{-1}=\int_0^\infty e^{-uS}du$、$S^{-2}=\int_0^\infty ue^{-uS}du$以及粒子交换性，可把银行期望降为单粒子潜变量积分与一个辅助积分。对离散normal节点$(z_\ell,\omega_\ell)$，将内层期望替换为加权和：

$$
\begin{aligned}
d_{y\ell}&=\omega_\ell x_\ell K\int_{\mathbb R}u e^{-ux_\ell}\Psi_y(u)^{K-1}\,dv,\\
c_{y\ell}&=\omega_\ell x_\ell K(K-1)\int_{\mathbb R}u^2e^{-ux_\ell}B_{0,y}(u)\Psi_y(u)^{K-2}\,dv.
\end{aligned}
\tag{I.14}
$$

逐标签导数因而为

$$
\begin{aligned}
g_{ba,y}&=-\sum_\ell d_{y\ell}G_\ell,\\
H_{ba,y}&=\sum_\ell d_{y\ell}B_\ell-\sum_\ell c_{y\ell}G_\ell G_\ell^\top\\
&\quad+K(K-1)\int_{\mathbb R}u^2M_{G,y}(u)M_{G,y}(u)^\top\Psi_y(u)^{K-2}\,dv.
\end{aligned}
\tag{I.15}
$$

最后再按$q_y$平均。有符号项的Fubini交换由$\sum_k\pi_k(1-\pi_k)\le1-1/K$及可积Gaussian得分矩保证。$K=1$时$g=-EG$、$H=EB$。对积分似然，后验为$\pi_y(dz)\propto A_y(z)\varphi(z)dz$，有

$$
\begin{aligned}
g_{ba,\infty,y}&=-E_{\pi_y}G,\\
H_{ba,\infty,y}&=E_{\pi_y}B-E_{\pi_y}GG^\top
+(E_{\pi_y}G)(E_{\pi_y}G)^\top,\\
H_{\perp,\infty,y}&=E_{\pi_y}D-E_{\pi_y}rr^\top.
\end{aligned}
\tag{I.16}
$$

对目标平均后，反射及Ward关系仍成立。当目标恰为该积分模型自身分布时，共线点的角度Hessian为零；这项核对不要求严格正定。对于原质量为$M$的数值normal测度$\nu$，保留其原权重及有限银行质量$M^K$：

$$
\begin{aligned}
F_{K,y}^{(\nu)}&=-M^K\log c_y+
\int_{\mathbb R}\{\Psi_y^{(\nu)}(e^v)^K-M^Ke^{-Ke^v}\}\,dv,\\
\sum_\ell d_{y\ell}&=M^K,\qquad
0\le\sum_\ell c_{y\ell}\le M^K(1-1/K).
\end{aligned}
\tag{I.17}
$$

等式针对未截断的辅助积分；程序检查相应数值残差。积分后验权重的质量则为1。normal权重和目标权重均不静默重归一。

对于角度方向$d_{\mathrm{ang}}^{(12)}\in\mathbb R^8$的匹配非共线路径，共同尺度在基线处固定：

$$
W_j(t)=a_j(\cos(td_{\mathrm{ang},j}^{(12)}),\sin(td_{\mathrm{ang},j}^{(12)})),\qquad
c_y=\frac{\int A_{y,0}\,d\nu}{M},\qquad
\Psi_{y,t}^{(\nu)}(u)=\int e^{-uA_{y,t}/c_y}\,d\nu.
\tag{I.18}
$$

主差额直接计算为

$$
\begin{aligned}
\Delta F_K^{(\nu)}(t)&=\sum_yq_y\int_{\mathbb R}
\{\Psi_{y,t}^{(\nu)}(e^v)^K-\Psi_{y,0}^{(\nu)}(e^v)^K\}\,dv,\\
\Delta R^{(\nu)}(t)&=-\sum_yq_y\{\log p_t^{(\nu)}(y)-\log p_0^{(\nu)}(y)\}.
\end{aligned}
\tag{I.19}
$$

由Frullani身份，共同常数消去。只有内层积分变为二维，并非$2K$维。指数差使用保留符号的`expm1`；绝对损失相减只作接口核对。小$u$计算使用原质量及`log1p`／`expm1`，极小和回退到log-sum-exp。在真实Gaussian律下，固定行范数保持每个一维观察边际，反射使两种差额关于$t$为偶函数。局部光滑性给出

$$
F_K(t)-F_K(0)=\tfrac12t^2(d_{\mathrm{ang}}^{(12)})^\top H_\phi d_{\mathrm{ang}}^{(12)}+O(t^4),\qquad
R(t)-R(0)=\tfrac12t^2(d_{\mathrm{ang}}^{(12)})^\top H_{\phi,\infty}d_{\mathrm{ang}}^{(12)}+O(t^4).
\tag{I.20}
$$

下面的有限差额并非由（I.20）生成。

## I.4. 完整学习与混合的固定边际对照

四条探索流各含4096条来自$.8$教师的完整观察。所有方法共享数据与初始化：Jeffreys平滑边际logit，以及独立$N(0,.25^2)$载荷元素。偏置和载荷元素均约束在$[-2,2]$。投影Adam执行1200次更新，前800步学习率$.02$、其后$.005$，$(\beta_1,\beta_2)=(.9,.999)$，稳定项$10^{-8}$；输出为LAST。每步K8与K64共享128个新的二维prior向量，分别组成16或2个银行。每个银行先独立取log-mean再平均；合并银行会优化另一目标。GH20给出积分经验目标对照。相同粒子数不等于相同梯度噪声或总成本。

REF从GH-Adam LAST开始，用GH40 L-BFGS-B，最多300次迭代／450次调用，梯度容差$10^{-7}$、函数容差$10^{-12}$。四流均已满足梯度门：只调用一次，没有新增迭代。IND使用Jeffreys平滑拟合独立Bernoulli边际。教师只进入输出锁定后的总体评价，不进入训练。数据／prior流使用PCG64DXSM及SeedSequence $[202610070100,r,\mathrm{role}]$；角色0–4分别对应潜变量、发射均匀数、初始化、训练prior及独立评价。

|流|K8 KL|K64 KL|REF KL|IND KL|
|---|---:|---:|---:|---:|
|0|.010714173|.000882779|.001013446|.106851719|
|1|.013518917|.005017770|.004563076|.108354984|
|2|.008767421|.003466091|.002591933|.107382320|
|3|.011357337|.002199853|.002356989|.107318617|
|均值|.011089462|.002891623|.002631361|.107476910|

每流K64的风险都低于K8，但其中两流还低于REF：有限数据下，更精确的积分并非逐例最优。K8单位行Gram矩阵的第二特征值为$1.20\,10^{-5}$–$2.30\,10^{-4}$，REF为1.3403–1.5988，K64为1.0785–1.4567。这指的是近一因子的**载荷结构**，不是秩一的观察协方差。

每流使用4096个新的64粒子银行配对评价固定模型；$F_8$取前8个粒子。下表为K8减REF，括号内是条件MCSE，单位为每条八位观察的nats。

|流|经验$F_8$差|总体$F_8$差|
|---|---:|---:|
|0|−.017111839 (.000420685)|−.019030398 (.000421690)|
|1|−.017389742 (.000861898)|−.017994448 (.000870445)|
|2|−.016952623 (.000519001)|−.019887290 (.000506188)|
|3|−.017387109 (.000383751)|−.018698806 (.000387336)|

积分经验NLL和总体KL都偏好REF；$F_{64}$评价也在所有流及两类目标下偏好REF。这些是固定输出之间的排序比较，不是全局最优或训练总体置信结论。

分阶段profile对照固定REF的偏置和行范数，将单位方向换成K8或K64方向，得到H8／H64，不作教师对齐或重新拟合。另行冻结的银行家族$[202610070300,r,0]$使用相同评价预算。全部候选仍在可行域内，边际最大差为$6.67\,10^{-16}$。

|流|H8−REF KL|经验$F_8$差（MCSE）|总体$F_8$差（MCSE）|
|---|---:|---:|---:|
|0|+.015649258|−.000384622 (.000474800)|−.001914842 (.000475939)|
|1|+.015691309|−.000283583 (.000921181)|+.000805674 (.000943760)|
|2|+.015529662|−.002605471 (.000575264)|−.000903313 (.000572604)|
|3|+.016110010|+.001209140 (.000399319)|−.000149413 (.000400167)|

H8在四流都增加风险，但$F_8$排序混合，且一些差值与MCSE同量级。$F_{64}$一致惩罚H8。H64减REF的KL为$(−.000009563,+.000896783,+.000587691,+.000160217)$；H64平均KL为$.00304014$，H8为$.01837642$。这个改善对照及后续曲率结果都没有消除H8比较的不一致；没有追加银行追求特定符号。

## I.5. 行长对照与完整驻定数值签名

规范共线锚点使用正行范数，而不是原载荷第一列。每流交叉组合K8／REF偏置与K8／REF行范数；固定K8偏置时，还在$\lambda=.25,.5,.75$处插值行范数。在K8偏置及行范数下，角度Hessian投影至$\mathbf1^\perp$后的最小特征值为：

|流|经验$F_8$|总体$F_8$|总体$F_{64}$|风险|
|---|---:|---:|---:|---:|
|0|+.002854452|+.003744359|−.004021564|−.005337262|
|1|+.004466493|+.002889122|−.006012170|−.007533965|
|2|+.004168796|+.002341562|−.008817385|−.010706974|
|3|+.003203522|+.003711700|−.003994969|−.005288578|

只换成REF行范数后，经验$F_8$最小值变为$−.01958$到$−.01101$，总体最小值变为$−.02006$到$−.01332$。本面板中只换偏置不改变端点类型。插值给出$(0,.25)$或$(.25,.5)$的数值异号区间，不保证唯一或单调。这些锚点的偏置／径向梯度非零；角度稳定本身不是完整驻点结论。

为核对这一差别，对四个经验$F_8$问题E0–E3和一个总体问题P优化全部$(b,a)$，约束$b\in[-2,2]^8$、$a\in[.05,2]^8$。它们从相应K8规范锚点开始，P使用流0。P是使用教师的机制诊断，不是第五份独立数据。

|终点|$\|g_{ba}\|_\infty$|$\lambda_{\min}(H_{ba})$|$\lambda_{\min}(V^\top H_\perp V)$|KL|
|---|---:|---:|---:|---:|
|E0|$9.93946\,10^{-11}$|.05029121|.01062942|.01110942|
|E1|$4.53156\,10^{-11}$|.05191026|.01891921|.01373345|
|E2|$5.02400\,10^{-14}$|.05465116|.01576894|.00908874|
|E3|$2.96443\,10^{-12}$|.05410205|.01389994|.01229271|
|P|$2.93578\,10^{-12}$|.05396628|.01485717|.01043357|

五点都通过完整局部极小的数值签名。表中仍是原近似终点的数值读数。总体P目标另有I.8认证的附近精确局部极小；该证书不外推到E0–E3。在P处，积分风险梯度范数为$.02456458$。沿其固定负单位$(b,a)$方向，预定五个有限步均通过相应资格：

|步长|$\Delta F_8$|$\Delta R$|
|---:|---:|---:|
|−.02|$+1.49974827\,10^{-5}$|$+5.09524051\,10^{-4}$|
|−.01|$+3.74871128\,10^{-6}$|$+2.50202674\,10^{-4}$|
|0|0|0|
|+.01|$+3.74715392\,10^{-6}$|$-2.41091767\,10^{-4}$|
|+.02|$+1.49850334\,10^{-5}$|$-4.73081829\,10^{-4}$|

正向$.02$步把KL从$.01043357299$降到$.00996049116$，同时升高$F_8$。在同一五个终点上，总体$F_{64}$和风险的角度块具有负方向；这两个目标没有在那里被优化到驻定。

## I.6. 对每个K重新拟合与同方向比较

随后对每个总体问题重新拟合，从前一个可行LAST开始，完成整个粗序列中的K=9、10、12、16、24、32、64、128及积分目标。之后才对首个相邻且合格的有限正→负对作整数细化，最多六个中点，起点按端点参数插值。实际只在11处细化一次，得到保留对$[11,12]$。下表是Cartesian横向特征值，不是角度方向曲率。

|K|$\|g_{ba}\|_\infty$|$\lambda_{\min}(H_{ba})$|横向最小值|负模数|KL|
|---:|---:|---:|---:|---:|---:|
|8，复用|$2.936\,10^{-12}$|.0539663|+.01485717|0|.010433573|
|9|$4.910\,10^{-14}$|.0530561|+.00959400|0|.009461482|
|10|$5.902\,10^{-11}$|.0524036|+.00530421|0|.008842072|
|11|$2.513\,10^{-12}$|.0519187|+.00174653|0|.008426627|
|12|$4.369\,10^{-14}$|.0515476|−.00124838|1|.008136147|
|16|$3.982\,10^{-14}$|.0506760|−.00960756|1|.007557032|
|24|$3.617\,10^{-14}$|.0500081|−.01810993|1|.007228793|
|32|$3.951\,10^{-11}$|.0497429|−.02239912|1|.007134125|
|64|$3.200\,10^{-14}$|.0494177|−.02886195|1|.007055936|
|128|$9.837\,10^{-12}$|.0492813|−.03210040|1|.007039262|
|$\infty$|$7.083\,10^{-12}$|.0491572|−.03533998|2|.007034194|

所有新点均通过径向驻定资格。积分极限仍受共线子模型约束，其正风险不是完整两因子模型的恢复上限。没有继续追踪积分极限的第二负模。整数类型区分不是关于唯一分支、单调转换或普适临界K的定理。

方向$d_{\mathrm{ang}}^{(12)}\in\mathbb R^8$取K12终点的最小角度特征向量，从固定Helmert基映回，并令最大绝对分量为正。舍入分量： $(-.5316182,-.3864170,-.2460486,-.0867412,.0867412,.2460486,.3864170,.5316182)$；实际始终使用与$\mathbf1$正交的全精度单位向量。角度特征值间隙为$.0122054$。此角度方向不同于I.8.5的物理风险方向$d_{\mathrm{risk}}$。四格均使用同一$(d_{\mathrm{ang}}^{(12)})^\top H_\phi d_{\mathrm{ang}}^{(12)}$，得到：

|参数|$F_{11}$|$F_{12}$|
|---|---:|---:|
|$\theta_{11}$|+.000542300179|−.000056581709|
|$\theta_{12}$|+.000226560907|−.000394546595|

先换K的两项贡献为$−.000598881888$和$−.000337964886$；先换参数的两项为$−.000315739271$和$−.000621107502$。总和均为$−.000936846774$，交互项为$−.000022225614$。这是依赖顺序的四格代数，而非唯一因果分解。交叉格并未对其交叉目标驻定，仍须保留径向补偿。

## I.7. 直接非共线行为、数值资格与边界

两个锁定终点使用同一$d_{\mathrm{ang}}^{(12)}$及（I.18）。主幅度为$|t|=.02$，$.05,.10$是固定敏感性幅度。正负两侧全部独立评价，零点共享基线。主规则读数如下：

|K|t|$\Delta F_K$|$\Delta R$|$2\Delta F_K/t^2$|$2\Delta R/t^2$|
|---:|-----:|-------------:|-------------:|-------------:|-------------:|
|11|−.10|$+2.90960527\,10^{-6}$|$-3.62277988\,10^{-5}$|+.000581921053|−.00724555975|
|11|−.05|$+6.90272432\,10^{-7}$|$-9.10994254\,10^{-6}$|+.000552217946|−.00728795403|
|11|−.02|$+1.08777517\,10^{-7}$|$-1.45996822\,10^{-6}$|+.000543887585|−.00729984112|
|11|+.02|$+1.08777517\,10^{-7}$|$-1.45996822\,10^{-6}$|+.000543887585|−.00729984112|
|11|+.05|$+6.90272432\,10^{-7}$|$-9.10994254\,10^{-6}$|+.000552217946|−.00728795403|
|11|+.10|$+2.90960527\,10^{-6}$|$-3.62277988\,10^{-5}$|+.000581921053|−.00724555975|
|12|−.10|$-1.76055405\,10^{-6}$|$-3.91948957\,10^{-5}$|−.000352110811|−.00783897915|
|12|−.05|$-4.79905287\,10^{-7}$|$-9.85357108\,10^{-6}$|−.000383924230|−.00788285687|
|12|−.02|$-7.85692828\,10^{-8}$|$-1.57903202\,10^{-6}$|−.000392846414|−.00789516009|
|12|+.02|$-7.85692828\,10^{-8}$|$-1.57903202\,10^{-6}$|−.000392846414|−.00789516009|
|12|+.05|$-4.79905287\,10^{-7}$|$-9.85357108\,10^{-6}$|−.000383924230|−.00788285687|
|12|+.10|$-1.76055405\,10^{-6}$|$-3.91948957\,10^{-5}$|−.000352110811|−.00783897915|

两锚点的零点差额均按定义为0。在$.02$处，$2\Delta R/t^2$为$−.00729984112$和$−.00789516009$。有限目标的二阶读数与保存曲率分别偏离.293%／.431%；在$.10$处，偏离增至7.31%／10.76%。Taylor接近程度不是资格门。主幅度的风险改善仅约为基线KL的.017%／.019%：这是机制核对，不是显著恢复或一个新优化的两因子解。

**数值方案。** PMF及学习终点使用GH40/60/80，以及旋转$\pi/7$的GH72张量规则：质量容差$2\,10^{-12}$、PMF差$2\,10^{-7}$、风险差$2\,10^{-6}$，NLL／梯度容差$10^{-5}/10^{-4}$。固定边际曲率对照比较$[-10,10]$上的正态加权Legendre384与$[-11,11]$上的Legendre512，log-u网格分别为$[-24,56]/.125$和$[-28,64]/.125$。另保存Hermite96与Legendre256／$[-10,10]$核对，两者均使用$[-24,48]/.25$。矩阵、谱、Ward及质量门采用下文对应标准。全部固定锚点曲率资格通过。

驻定求解采用两个固定阶段：Legendre128／$[-10,10]$、log-u $[-24,48]/.25$，最多12个接受步／80次调用；随后Legendre256／$[-10,10]$、$[-28,56]/.25$，最多4步／40次调用。梯度停止门为$10^{-9}/10^{-10}$。Newton方向使用$H_s=(H+H^\top)/2$及移位$\max(0,10^{-4}-\lambda_{\min}(H_s))I$，无穷范数限幅$.25$。初始步长乘子为$\min(1,.99\alpha_{\rm box})$，其中$\alpha_{\rm box}$是最大可行乘子，之后至多二分12次。Armijo常数为$10^{-4}$，固定损失松弛为$5\,10^{-13}$。资格使用**未加移位**的Hessian；对称化仅消除算术反对称误差，不裁剪特征值。

驻定／延续终点评价比较Legendre256／$[-10,10]$、log-u $[-28,56]/.25$与Legendre384／$[-11,11]$、$[-32,64]/.125$。门为：损失$10^{-7}$，梯度／矩阵／Ward及原系数质量残差$2\,10^{-7}$，谱$2\,10^{-6}$，normal质量$2\,10^{-12}$。自身目标的签名还要求梯度无穷范数不超过$10^{-8}$、box边距至少$10^{-3}$、径向最小特征值大于$10^{-5}$，以及Cartesian Ward／旋转残差不超过$2\,10^{-7}$。横向符号阈值为$\pm10^{-5}$。所报告的驻定、延续和四格资格均通过。

直接路径使用三套分别指定的规则：

|规则|第一Gaussian轴|第二轴|log-u区间／步长|
|---|---|---|---|
|A|GH96|GH24|$[-28,56]/.25$|
|B，主值|GH144|GH32|$[-32,64]/.125$|
|C，核对|正态加权Legendre384, $[-11,11]$|GH40|$[-36,72]/.125$|

配对$\Delta F$要求B／C绝对差$\le10^{-10}$且相对差$\le2\%$，A／C为$\le10^{-9}$且$\le10\%$；风险对应门为$10^{-11}/2\%$和$10^{-10}/10\%$。相对比较以两边幅度及$10^{-12}$的最大值为基准。符号还要求B／C同号，且最小幅度超过$\max(10^{-9},20|\Delta_B-\Delta_C|)$。每个规则核对范数误差$\le5\,10^{-14}$、边际变化$\le2\,10^{-10}$、两种效应的偶性残差$\le10^{-11}$及PMF偶性$\le2\,10^{-12}$。

直接路径还要求normal／模型质量误差$\le2\,10^{-12}$、偏置不变且PMF非负。A／B相对C的绝对损失、原KL和PMF门分别为$10^{-7},10^{-10},2\,10^{-10}$。二维基线与保存的一维参考比较，损失／PMF容差为$10^{-7}/2\,10^{-10}$。直接差与相减差的接口容差为$5\,10^{-12}$。这些预定筛查既不是统计置信界，也不是严格积分误差界；此前$10^{-7}$的绝对损失门单独不足以分辨主效应。

全部六个基线规则和十二个非零参数点均通过。B／C配对差的最大偏差为：$\Delta F$的$5.620\,10^{-17}$及$\Delta R$的$1.551\,10^{-16}$；最大边际变化为$3.886\,10^{-16}$。独立执行核验重建保存量的聚合、求解决策和资格，并使用另行实现的选定标签积分。直接路径中，全部十四个C规则PMF均独立重建；对两锚点正向$.02,.10$处的标签0、85、255，另作自适应辅助积分，最大逐标签差额偏差为$2.776\,10^{-16}$。负侧积分没有另行独立重复，但每个保存的负侧结果均参加聚合与偶性核对。这不是完整独立重新优化，也不是严格Gaussian／辅助积分余项证书。

保留的训练记录勘误涉及十六个中间参数字段，它们因别名而指向GH-LAST。当时落盘的路径文件及保存轨迹给出正确参数；损失、梯度、资格标志、更新、锁定输出和profile构造均未受影响。保留原记录及失败审计尝试，以新增补正元数据处理，而未重写科学输出。

四条训练流仍是仅有的独立学习重复。确定性后续核对没有增加新的统计置信家族，也不继承旧附录的覆盖保证。固定有限路径面板已完成当前局部机制检验，没有追加二维优化、第二负模搜索或更多K／教师／种子扫描。四个经验拟合与K11/K12延续的严格驻点存在性仍未建立。下节认证总体K8问题；全局恢复、Adam吸引及独立条件下的事前预测仍未建立。

![学习输出与局部几何。(a)四流观察KL，纵轴为对数，黑线为均值。(b)去除共同旋转后的最小Cartesian横向曲率；横轴是离散粒子数，连线不表示连续分岔，纵轴不是角度Hessian特征值。(c,d)沿$d_{\mathrm{ang}}^{(12)}$直接计算K11/K12的目标差与风险差，单位为每条八位观察的nats，分别按$10^{-6}$、$10^{-5}$缩放。这些有限幅度数值结果区别于同根K8证书(14)--(15)。](figures/joint_mechanism.png){width=100%}

## I.8. 总体局部极小与风险冲突的认证证明 {#proof-theorem-3}

本节对（I.1）的精确总体目标、K8情形证明定理3。见证是期望目标的真实驻点，由根包含定位，不把打印出的数值终点当作精确根。教师、中心及证明中的每个范数均指向同一个模型。

::: {#full-local-statement}

**定理3的完整定量陈述。**

**(i) 完整空间局部极小与正风险。**

对(13)指定的精确教师，令$q_*$为其观察分布，$\mathcal R(\theta)=D(q_*\Vert Q_\theta)$。存在共线内部参数$\theta_*=(0,a_*,0)$、$a_{*,i}>0$及固定物理单位方向$d_{\mathrm{risk}}\in\mathbb R^{24}$，使$\theta_*$在完整24维参数空间中模共同潜变量旋转后为$F_8^{q_*}$的严格局部极小。对[附录I.8](#proof-theorem-3)指定的精确dyadic中心$\theta_0$，

$$
\begin{gathered}
\|\theta_*-\theta_0\|_2<1.805\times10^{-8},\\
\nabla F_8^{q_*}(\theta_*)=0,\qquad
D\mathcal R(\theta_*)[d_{\mathrm{risk}}]<-.0201723.
\end{gathered}
$$

每个观察边际均正确，$Q_{\theta_*}(Y_j=1)=q_*(Y_j=1)=1/2$，但$0.0104335716\le\mathcal R(\theta_*)\le0.0104335744$。沿$d_{\mathrm{risk}}$的固定$10^{-7}$步提高$F_8$、严格降低风险，并保持这些观察边际；行长与Gaussian logit方差发生变化。

```{=latex}
\Needspace{5\baselineskip}
```

**(ii) 同根角冲突。**

还存在固定角方向$d=d_{\mathrm{ang}}^{(8)}\in\mathbf1^\perp\subset\mathbb R^8$、$\|d\|_2=1$，使路径$\theta_{\mathrm{ang}}(t)=(0,W(t))$满足

$$
\begin{aligned}
W_i(t)&=a_{*,i}\bigl(\cos(td_i),\sin(td_i)\bigr),\\
r(t)&=\mathcal R(\theta_{\mathrm{ang}}(t)),\\
r'(0)&=0,\qquad r''(0)<-.00450377.
\end{aligned}
$$

存在$\varepsilon>0$，使每个$0<|t|<\varepsilon$均满足

$$
\begin{gathered}
F_8^{q_*}(\theta_{\mathrm{ang}}(t))>F_8^{q_*}(\theta_*),\\
\mathcal R(\theta_{\mathrm{ang}}(t))<\mathcal R(\theta_*),\\
\operatorname{rank}W(t)=2.
\end{gathered}
$$

路径保持全部行长、logit边际$N(0,a_{*,i}^2)$及公平观察边际。单位归一化在角空间中成立，并非对Cartesian路径速度归一化；所保持的候选logit方差不同于教师的$.8^2$。

:::

### I.8.1. 精确输入、对称性与完整参数空间

教师行长与角度是精确有理数。候选中心将旧总体终点的成对对称化数值视为精确dyadic；其前四个正行长如下，后四个为反序。中心没有重新优化。载荷两列记为u、v；向量使用物理Euclidean范数，矩阵使用operator范数。

$$
\begin{aligned}
W_{*,j}&=\tfrac45(\cos\phi_{*,j},\sin\phi_{*,j}),\quad b_*=0,\quad
\phi_*=(-\tfrac7{10},-\tfrac12,-\tfrac3{10},-\tfrac1{10},\tfrac1{10},\tfrac3{10},\tfrac12,\tfrac7{10}),\\
\theta_0&=(0,a_0,0),\qquad a_{0,j}=a_{0,9-j},\\
(a_{0,1},\ldots,a_{0,4})&=
\left(\frac{4236614515730759}{9007199254740992},
\frac{1274451243427741}{2251799813685248},
\frac{181129089615167}{281474976710656},
\frac{775431105319425}{1125899906842624}\right).
\end{aligned}
\tag{I.21}
$$

令P反序八个观察。教师具有标签补集与反序对称性；后者对应第二潜变量反号。标签及全部粒子同时反号给整个偏置向量的偶性，不是逐个偏置独立偶性。第二潜变量反射对任意目标均成立。因此，使用四个等距坐标时，

$$
\begin{aligned}
q_*(\mathbf1-y)&=q_*(y)=q_*(Py),\qquad
F_8^{q_*}(-b,W)=F_8^{q_*}(b,W),\\
F_8^q(b,u,-v)&=F_8^q(b,u,v),\qquad
U_+=\frac1{\sqrt2}(e_1+e_8,e_2+e_7,e_3+e_6,e_4+e_5),\\
a&=U_+\xi,\qquad f(\xi)=F_8^{q_*}(0,U_+\xi,0),\qquad
\nabla_bF_8^{q_*}=\nabla_vF_8^{q_*}=0,\quad \nabla_aF_8^{q_*}=U_+\nabla f.
\end{aligned}
\tag{I.22}
$$

最后一行在成对、零偏置、共线子空间成立，给四坐标梯度与完整24坐标梯度的范数相等。局部类型检查仍保留八个偏置、四个对称和四个反对称行长、七个非旋转横向方向。下文Hessian均先在未约束完整模型中求导，再在成对中心评价。

仅为计算，将标签归为四个成对计数；Gaussian粒子反号使每类与其补类等价，留下41个按lexicographic选择的代表。各类权重从原教师概率求和，不假定教师在单对内部可交换。

$$
\begin{aligned}
c_j&=y_j+y_{9-j}\in\{0,1,2\},\qquad
A_c(z)=\prod_{j=1}^4\sigma(a_{0,j}z)^{c_j}[1-\sigma(a_{0,j}z)]^{2-c_j},\\
G_{\xi,c}(z)&=\frac z{\sqrt2}\{c-2\sigma(a_{0,1:4}z)\},\qquad
A_{2-c}(-z)=A_c(z),\quad G_{\xi,2-c}(-z)=G_{\xi,c}(z),\\
\mu_c&=-E\sum_{k=1}^8\pi_kG_{\xi,c}(Z_k)=\mu_{2-c},\qquad
\nabla f=\sum_{r\in\mathcal C}w_r\mu_r,\\
\mathcal C&=\{\min(c,2-c):c\in\{0,1,2\}^4\},\quad|\mathcal C|=41,\quad
w_r=\sum_{y:\min(c(y),2-c(y))=r}q_*(y).
\end{aligned}
\tag{I.23}
$$

类梯度相等使用反号后的Gaussian银行，不是保持同一个固定银行不动。Hessian还必须保留完整标签残差。

### I.8.2. 实际求积的Hessian密度

似然尺度固定为1，使用物理Laplace变量。以（I.6）的完整16坐标得分和条件负Hessian定义以下内层矩；在共线点，期望均为一维标准Gaussian期望。这里$B=B_y(z)$是单粒子条件负Hessian，$\mathcal B=\mathcal B_y(v)$是其$t$加权Gaussian矩；（I.24）省略标签及辅助坐标依赖。

$$
\begin{aligned}
u&=e^v,\quad t=uA_y(z)e^{-uA_y(z)},\quad\Psi=E e^{-uA_y(Z)},\quad T_0=Et,\\
\mathcal R&=E[tG],\quad\mathcal Q=E[tGG^\top],\quad\mathcal B=E[tB],\quad
\mathcal Q_p=E[trr^\top],\quad\mathcal B_p=E[t\operatorname{diag}\sigma'(a_0Z)],\\
I_{ba,y}(v)&=K\Psi^{K-1}\mathcal B-K(K-1)\Psi^{K-2}(T_0\mathcal Q-\mathcal R\mathcal R^\top),\\
I_{p,y}(v)&=K\Psi^{K-1}\mathcal B_p-K(K-1)\Psi^{K-2}T_0\mathcal Q_p,\qquad K=8.
\end{aligned}
\tag{I.24}
$$

将密度在实辅助轴上积分，再按标签平均，得到完整径向和横向块。首项为条件负Hessian；次项包含同粒子项减异粒子得分外积。横向块的异粒子外积因不同粒子的第二坐标独立且中心化而消失。I.3中的倒数及平方倒数身份给上述幂次与系数，无额外K或u缩放。

必须区分同／异粒子密度与直接对梯度密度作参数导数。对实银行定义以下和及似然权重；所得covariance为半正定。

$$
\begin{aligned}
S&=\sum_kA_k,\quad w_k=A_k/S,\quad s=e^vS,\\
\bar B&=\sum_kw_kB_k,\quad\bar G=\sum_kw_kG_k,\quad\bar Q=\sum_kw_kG_kG_k^\top,\\
I_{ba}(v)&=E[e^{-s}\{s\bar B-s^2(\bar Q-\bar G\bar G^\top)\}],\\
I_{\rm dir}(v)&=E[e^{-s}\{s(\bar B-\bar Q)+s^2\bar G\bar G^\top\}],\\
I_{ba}(v)-I_{\rm dir}(v)&=E[\partial_v(se^{-s})\bar Q],\\
\int_{\mathbb R}(I_{ba}-I_{\rm dir})\,dv&=0.
\end{aligned}
\tag{I.25}
$$

每个正银行和的边界项为零，Gaussian得分矩保证期望与积分交换。两密度的全实积分相同，有限格值不同。以下误差包络对应首个密度，正是证书计算的密度。

补侧标签使用其有效补标签，自补计数只计一次。这保持三个对角块；偏置—径向混块变号，只有使用精确教师对称性后才为零。逐代表构造加权标签矩及内层概率矩：

$$
\begin{aligned}
W&=\sum q_y,\quad M_i=\sum q_y\widetilde y_i,\quad S_{ij}=\sum q_y\widetilde y_i\widetilde y_j,\\
T_l&=E[tz^l],\quad P_{l,i}=E[tz^lp_i(z)],\quad PP_{l,ij}=E[tz^lp_i(z)p_j(z)],\quad p_i(z)=\sigma(a_{0,i}z),\\
D_l&=W\operatorname{diag}(P_l-\operatorname{diag}PP_l),\quad l=0,2,\\
J_{bb}&=W(T_0PP_0-P_0P_0^\top),\\
J_{aa}&=(T_0T_2-T_1^2)S-T_0(MP_2^\top+P_2M^\top)
+T_1(MP_1^\top+P_1M^\top)+W(T_0PP_2-P_1P_1^\top),\\
J_p&=T_0(T_0S-MP_0^\top-P_0M^\top+WPP_0),\\
I_{bb}&=K\Psi^{K-1}D_0-K(K-1)\Psi^{K-2}J_{bb},\quad
I_{aa}=K\Psi^{K-1}D_2-K(K-1)\Psi^{K-2}J_{aa},\\
I_p&=K\Psi^{K-1}D_0-K(K-1)\Psi^{K-2}J_p.
\end{aligned}
\tag{I.26}
$$

标签矩在首个covariance差中代数消去，在另外两块仍保留。它们是八坐标向量及矩阵，公式不除以类质量。四个不同概率使内层标量矩数量为36（包含Laplace变换），却不降低Hessian维数；broken-pair及反对称行长方向的信息仍在。

### I.8.3. 区间求积、离散尾与合成

计算使用192-bit midpoint-radius算术[@johansson2017]，有限和用精确dyadic端点外包，再加解析无限格及遗漏格误差。浮点规则相符不作为误差界。对于轮廓端边界消失、边界包络可积的解析条带，Fourier轮廓平移及Poisson周期化给[@trefethen2014]

$$
\left\|\int_{\mathbb R}f(x)\,dx-h\sum_{k\in\mathbb Z}f(kh)\right\|
\le\frac{M_++M_-}{e^{2\pi d/h}-1},\qquad
M_\pm\ge\int_{\mathbb R}\|f(x\pm id)\|\,dx.
\tag{I.27}
$$

同一论证适用于向量Euclidean范数和矩阵operator范数。本模型各阶实轴导数具有Gaussian多项式或辅助轴快速衰减包络，故函数属Schwartz类；Fourier级数绝对收敛也验证周期化身份。仅有解析性并不足够。

教师使用以下矩形内全权二维格。在polystrip中，各logit避开sigmoid极点，所有标签似然绝对值之和因子化，因此直接控制完整概率向量，不把逐标签界乘256。

具体地，$\sum_y|A_y(x+it)|\le\prod_j\sec(B_j/2)$，且$|\varphi(x_i+it_i)|=e^{t_i^2/2}\varphi(x_i)$。两个坐标的Fourier平移给几何alias和之积，删去零频即得下一式的因子。

$$
\begin{aligned}
\theta_{\rm T}&=(b_*,W_*),\quad h_t=\tfrac18,\quad L_t=10,\quad d_t=1,\\
q_y^f&=h_t^2\sum_{|k_1|,|k_2|\le80}\varphi(h_tk_1)\varphi(h_tk_2)A_y(h_tk;\theta_{\rm T}),\\
B_j&=d_t(|W_{*,j1}|+|W_{*,j2}|)<\pi,\quad
M_t=e^{d_t^2}\prod_j\sec(B_j/2),\quad \rho_t=e^{-2\pi d_t/h_t},\\
\|q_*-q^f\|_1&\le\delta_t:=M_t\frac{4\rho_t}{(1-\rho_t)^2}
+2(1+h_t\varphi(0))\frac{2\varphi(L_t)}{L_t}<4.894\,10^{-21}.
\end{aligned}
\tag{I.28}
$$

第二项是遗漏Gaussian格点质量界：一轴无限格质量至多1加中央格权重，每个遗漏右尾由相邻连续尾控制。保留的边与角都取全权。有限和算术半径另保留在概率球内。

候选内外格及全权端点固定如下。定义载荷总和、扇形角及secant常数。各sigmoid因子的argument在零与logit虚部之间，故似然乘积可选连续argument，其幅度不超过扇形角。因此整个内层条带中实部为正，对每个实辅助坐标统一成立。

$$
\begin{aligned}
h_z&=\tfrac1{40},\ L_z=11,\ d_z=\tfrac18,\qquad
h_v=\tfrac18,\ v_-=-32,\ v_+=64,\ d_v=\tfrac12,\ W_v=96.125,\\
C_a&=\sum_i|a_{0,i}|,\quad\beta=C_ad_z<\pi/2,\quad
c_i=\sec(|a_{0,i}|d_z/2),\ C_R^2=\sum_i c_i^2,\ c_*^2=\max_i c_i^2,\\
\operatorname{Re}A_y(x+it)&\ge|A_y(x+it)|\cos\beta>0,\quad
|e^{-uA_y}|\le1,\quad |uA_ye^{-uA_y}|\le(e\cos\beta)^{-1},\\
F_z&=\frac{e^{d_z^2/2}}{e\cos\beta},\quad J_z=2+d_z^2,\quad
\rho_z=e^{-2\pi d_z/h_z},\qquad
\delta_X=\frac{2M_X\rho_z}{1-\rho_z}+T_X.
\end{aligned}
\tag{I.29}
$$

内层881点、外层769点。条带内残差及条件Hessian范数由secant控制，Gaussian边界积分给下表包络。中列按对象为标量、向量或operator边界积分界；末列为遗漏内层格界。

有限和实现可将$e^{-x}$和$xe^{-x}$外包为$[0,e^{-128}]$和$[0,128e^{-128}]$，条件是区间输入证明$x=uA\ge128$。这些是保守饱和区间，不是替换成零值；半径仍保留在最终有限和球中。

|内层矩|$M_X$|$T_X$|
|---|---|---|
|$\Psi$|$e^{d_z^2/2}$|$p_L$|
|$T_0$|$F_z$|$p_L/e$|
|$\mathcal R$|$F_z\sqrt{C_R^2J_z}$|$2\sqrt{2m}\varphi(L_z)/e$|
|$\mathcal Q$|$F_z C_R^2J_z$|$m j_L/e$|
|$\mathcal B$|$F_z c_*^2J_z/4$|$j_L/(4e)$|
|$\mathcal Q_p$|$F_z C_R^2$|$m p_L/e$|
|$\mathcal B_p$|$F_z c_*^2/4$|$p_L/(4e)$|

尾部符号为$p_L=2\varphi(L_z)/L_z$和$j_L=2\varphi(L_z)(L_z+2/L_z)$，$m=8$；由保留端点之外的单调右矩形比较以及Gaussian一、二阶矩得到。解析密度使用普通转置而非共轭转置；复秩一矩阵的operator范数仍由向量范数平方控制。

统一内层误差必须穿过嵌套幂及矩阵乘积。仅在以下估计中，用小写字母记真实矩的范数上界，采用以下乘积误差算子。

$$
\begin{aligned}
P&=1+\delta_\Psi,\quad
\mathcal E_n(b,\delta b)=P^n\delta b+nP^{n-1}\delta_\Psi b,\\
C&=T_0\mathcal Q-\mathcal R\mathcal R^\top,\quad
\|C\|\le tq+r^2,\quad
\delta_C\le\delta_{T_0}q+(t+\delta_{T_0})\delta_{\mathcal Q}+2r\delta_{\mathcal R}+\delta_{\mathcal R}^2,\\
C_p&=T_0\mathcal Q_p,\quad\|C_p\|\le tq_p,\quad
\delta_{C_p}\le\delta_{T_0}q_p+(t+\delta_{T_0})\delta_{\mathcal Q_p},\\
E_{{\rm in},ba}&=W_v\{K\mathcal E_{K-1}(b,\delta_{\mathcal B})
+K(K-1)\mathcal E_{K-2}(tq+r^2,\delta_C)\},\\
E_{{\rm in},p}&=W_v\{K\mathcal E_{K-1}(b_p,\delta_{\mathcal B_p})
+K(K-1)\mathcal E_{K-2}(tq_p,\delta_{C_p})\}.
\end{aligned}
\tag{I.30}
$$

因子$W_v=769h_v=96.125$是有限外层和的权重总量，端点使用完整权重。$E_{{\rm in},ba}$和$E_{{\rm in},p}$比较这个有限和中的精确内层矩与近似内层矩；外层alias与遗漏格点尾另计。

可用实轴界为$t=1/e$、$r=\sqrt{2m}/e$、$q=2m/e$、$b=1/(2e)$、$q_p=m/e$、$b_p=1/(4e)$。幂的telescoping与矩阵三角不等式给（I.30），不静默删除接近1的因子。四梯度计算对$D=E[tG_\xi]$用同一论证，$\|D\|\le E_G/e$，向量包络如下。

$$
\begin{aligned}
E_G&=\sqrt m\sqrt{2/\pi},\quad
M_D=F_z\sqrt{C_R^2(1+d_z^2)},\quad
\delta_D=\frac{2M_D\rho_z}{1-\rho_z}+2\sqrt m\varphi(L_z)/e,\\
E_{{\rm in},g}&=KW_v\{P^{K-1}\delta_D+(K-1)P^{K-2}\delta_\Psi E_G/e\}.
\end{aligned}
\tag{I.31}
$$

在外层条带中，实银行似然权重仍为实数，只有辅助标量变成复数。换为银行缩放后的辅助半径，积分银行密度绝对值，得到

$$
\begin{aligned}
M_{ba}&=(1+K)\{(4\cos d_v)^{-1}+m(\cos d_v)^{-2}\},\\
M_p&=(4\cos d_v)^{-1}+m(\cos d_v)^{-2},\quad M_g=KE_G/\cos d_v,\\
E_{{\rm alias},X}&=\frac{2M_X\rho_v}{1-\rho_v},\qquad
\rho_v=e^{-2\pi d_v/h_v},\quad X\in\{ba,p,g\}.
\end{aligned}
\tag{I.32}
$$

在下端，密度及导数指数衰减。上端按与辅助坐标成比例的潜变量阈值拆分银行：有界潜变量部分的似然和给双指数衰减，补集由Gaussian尾给快速衰减。求导只产生银行缩放辅助半径的多项式乘阻尼指数，同一拆分适用于每个固定阶导数；因此外层（I.27）的轮廓端边界与Schwartz条件成立。

遗漏外层点需要离散尾界。以下两个银行核的总variation，使右格尾至多连续尾加“步长乘variation”；若边界已经越过峰值，单调右矩形可用更紧的连续尾界。

$$
\begin{aligned}
f_1(v)&=se^{-s},\quad f_2(v)=s^2e^{-s},\quad
\int_{\mathbb R}|f_1'(v)|\,dv=2/e,\quad\int_{\mathbb R}|f_2'(v)|\,dv=8/e^2,\\
l_1&=\frac{h_ve^{v_-}}{e^{h_v}-1},\quad l_2=\frac{h_ve^{2v_-}}{e^{2h_v}-1},\quad
E_{{\rm low},p}\le Kl_1/4+mK^2l_2,\quad E_{{\rm low},ba}\le(1+K)E_{{\rm low},p},\\
T&=4,\quad U=e^{v_+},\quad\eta=2^{-m}e^{-C_aT},\quad U\eta>2,\\
p_T&=2\varphi(T)/T,\quad t_{2T}=2\varphi(T)(T+1/T),\quad
C_{\rm bad}=\tfrac14(1+2h_v/e)+m(1+8h_v/e^2),\\
G_p&=\{\tfrac14+m(1+U\eta)\}e^{-U\eta},\quad
E_{{\rm up},p}\le G_p+C_{\rm bad}p_T^K,\\
E_{{\rm up},ba}&\le(1+K)G_p+C_{\rm bad}(p_T^K+Kt_{2T}p_T^{K-1}).
\end{aligned}
\tag{I.33}
$$

这里$Z_k$是共线似然及其一维内层积分使用的标量第一潜坐标。定义$G=\{\min_{1\le k\le8}|Z_k|\le T\}$及$G^c=\bigcap_{k=1}^8\{|Z_k|>T\}$。在$G$上，似然和满足$S=\sum_k A_y(Z_k)\ge\eta$。在$G^c$上，独立性与双侧Gaussian尾给$\Pr(G^c)\le p_T^K$及$E[(1+M^2)\mathbf1_{G^c}]\le p_T^K+Kt_{2T}p_T^{K-1}$，其中$M=\max_k|Z_k|$。因此bad集合保留格点修正，不把连续Laplace尾直接挪作每个遗漏格点的界。梯度的相应界为

$$
\begin{aligned}
E_{{\rm low},g}&\le KE_Gl_1,\\
E_{{\rm up},g}&\le\sqrt m\{\sqrt K e^{-U\eta}+(1+2h_v/e)2K\varphi(T)p_T^{K-1}\}.
\end{aligned}
\tag{I.34}
$$

将内层替代、外层alias及两个离散尾相加，得到逐标签统一误差。有限教师质量至多1加其L1误差。以下逐标签真梯度界及保守Hessian界把教师误差每个向量／矩阵范数只传入一次。

$$
\begin{aligned}
E_X^{\rm unif}&=E_{{\rm in},X}+E_{{\rm alias},X}+E_{{\rm low},X}+E_{{\rm up},X},\\
\|g_{\xi,y}\|&\le\sqrt m E\max_k|Z_k|\le\sqrt{mK}=8,\\
\|H_{ba,y}\|&\le(2m+\tfrac14)(1+K)=146.25,\quad\|H_{p,y}\|\le m+\tfrac14=8.25,\\
E_g&\le(1+\delta_t)E_g^{\rm unif}+8\delta_t,\quad
E_{ba}\le(1+\delta_t)E_{ba}^{\rm unif}+146.25\delta_t,\\
E_p&\le(1+\delta_t)E_p^{\rm unif}+8.25\delta_t.
\end{aligned}
\tag{I.35}
$$

Hessian使用逐标签偏置／径向principal blocks组成的block diagonal以及横向块；折叠正好保持这些对象。精确教师混块为零，不要求有限教师概率逐位对称。有限矩阵区间保留权重及算术不确定性；与保存dyadic矩阵比较时，Frobenius上界支配operator范数。

最终向外舍入界如下。保存矩阵是前序对称binary64矩阵的精确dyadic解释，其正性由区间LDL证明。固定横向切片的有理基按Gram矩阵检查，不假定它正交归一。

有理基的列为$V_j=e_j-(a_{0,j}/a_{0,8})e_8$，$1\le j\le7$。检查$V^\top(S_p-.014I)V\succ0$，相对于$V^\top V$认证物理曲率界。

$$
\begin{aligned}
\|\nabla F_8^{q_*}(\theta_0)\|_2&<7.572\,10^{-10},\\
\|H_{ba}(\theta_0)-S_{ba}\|_{\rm op}&<1.138\,10^{-8}<.001,\\
\|H_p(\theta_0)-S_p\|_{\rm op}&<2.531\,10^{-9}<.001,\\
S_{ba}&\succ.050I_{16},\qquad
w^\top S_pw>.014\|w\|_2^2\quad(0\ne w\perp a_0).
\end{aligned}
\tag{I.36}
$$

|分量|梯度L2|径向Hessian operator|横向Hessian operator|
|---|---:|---:|---:|
|内层替代|2.49106e−10|9.04057e−9|2.27090e−9|
|外层alias|5.00387e−10|2.33628e−9|2.59587e−10|
|外层下尾|2.14647e−13|2.14005e−13|2.37783e−14|
|数学总误差，含教师|<7.497076e−10|<1.137708e−8|<2.530514e−9|
|有限球范数或到保存矩阵距离|<7.490176e−12|<1.456e−13|<2.663e−14|

表中小数概括上界，（I.36）由精确dyadic端点及有理数比较决定；未单列的上尾已包含在总数中。已经按向量或operator范数给出的界，不再乘41类、256标签或矩阵元素数量。

### I.8.4. 根包含与23方向局部极小

完整参数的一致Hessian变化界把中心证书传入邻域。以下银行包络包含两个潜坐标；条件log-kernel的一、二、三阶导数由相应幂控制。log-sum-exp的三阶导数由平均三阶导数、三个covariance项及中心三阶矩组成，其界给出所列常数；Gaussian矩保证期望内求导。

$$
\begin{gathered}
q_Z=\sqrt{1+\max_{k\le K}\|Z_k\|^2},\\
\|D\ell_k\|\le\sqrt m q_Z,\quad\|D^2\ell_k\|\le q_Z^2/4,\quad\|D^3\ell_k\|\le q_Z^3/4,\\
L_F\le(\tfrac14+\tfrac32\sqrt m+8m^{3/2})
[1+2dK+dK(dK+2)]^{3/4}<14069,\quad m=K=8,\ d=2,\\
\|H_F(\theta)-H_F(\theta')\|_{\rm op}\le14069\|\theta-\theta'\|_2.
\end{gathered}
\tag{I.37}
$$

例如每个covariance项至多为一、二阶导数界乘积的两倍，中心三阶矩至多为一阶界立方的八倍。银行包络平方由1加十六自由度chi-square变量支配。整数上限以有理数上界核对，不由数值导数拟合。使用预定物理半径，

$$
\begin{aligned}
r&=5\,10^{-7},\qquad
m_r=m_{16}=.050-.001-14069r=.0419655,\\
m_T&=.014-.001-14069r=.0059655,\qquad
\epsilon_g=7.572\,10^{-10}<m_rr=2.098275\,10^{-8}.
\end{aligned}
\tag{I.38}
$$

四坐标闭球内受限目标强凸。球面上沿中心到边界的线段积分Hessian，得到以下向外不等式。闭球极小点不可能在边界，因为向内移动可下降；故它是内部受限根，且由强凸性在该球内唯一。（I.22）的精确对称性将其提升为完整驻点，强单调性进一步给位置。

$$
\begin{aligned}
\delta^\top\nabla f(\xi_0+\delta)&\ge-\epsilon_g r+m_rr^2>0\quad(\|\delta\|=r),\\
\nabla F_8^{q_*}(\theta_*)&=0,\qquad\theta_*=(0,a_*,0),\qquad
\|\theta_*-\theta_0\|_2<\epsilon_g/m_r<1.805\,10^{-8}.
\end{aligned}
\tag{I.39}
$$

固定切片为$a_0^\top v=0$，其沿共同潜变量旋转的约束导数是$a_0^\top u$，在小邻域中为正。隐函数定理给唯一的小旋转使邻点进入切片。真根处，反射使径向／横向混块为零；（I.38）使16个径向及七个切片横向方向为正。Taylor定理给严格切片极小，旋转不变性将其提升为模共同旋转严格局部极小。旋转不变性要求24维Hessian沿共同旋转方向奇异；这里的严格极小是模共同旋转的，16个偏置／径向方向和七个横向方向的正曲率均已认证。

### I.8.5. 风险方向、固定小步与范围

同一中心及一个固定成对dyadic向量定义数学单位方向$d_{\mathrm{risk}}\in\mathbb R^{24}$。候选概率与方向导数由一维177个全权节点外包，步长八分之一、区间负十一到十一；再加条带及Gaussian尾误差，教师概率误差通过后验比值传入。Jensen不等式给物理球内统一正概率下界，后验负Hessian身份给风险曲率界。

（I.40）中定义方向的向量$h$不同于格步长$h_t,h_z,h_v$及（I.27）的一般标量步长。其分量来自数值径向风险梯度的成对对称化；认证只使用固定的精确dyadic，未假定它等于真实梯度。其前四个分量为以下精确dyadic；令$h_{9-j}=h_j$，并使用数学范数$\|h\|_2=\sqrt{\sum_{j=1}^8h_j^2}$。

|分量|精确dyadic|
|---|---|
|$h_1$|$-8589911184852041/1152921504606846976$|
|$h_2$|$-2443582330405755/288230376151711744$|
|$h_3$|$-5282230754430109/576460752303423488$|
|$h_4$|$-2739850785036709/288230376151711744$|

$$
\begin{aligned}
d_{\mathrm{risk}}&=(0,-h/\|h\|_2,0),\\
B_{h,y}&=E[A_y(Z)Z\sum_i h_i\{y_i-\sigma(a_{0,i}Z)\}],\\
DR(\theta_0)[d_{\mathrm{risk}}]&=\sum_yq_*(y)\frac{B_{h,y}}{Q_y(\theta_0)\|h\|_2}\\
&\in[-.024564580310120128,-.024564580310120124],\\
Q_y(\theta)&\ge q_{\min}:=2^{-8}
\exp\{-3r/2-[r^2+(\|a_0\|_2+r)^2]/8\},\\
\|H_R(\theta)\|&\le M_R:=24/q_{\min}<8784.371,\\
DR(\theta)[d_{\mathrm{risk}}]&<-\nu,\quad\nu=.0201723,\qquad\|\theta-\theta_0\|_2\le r.
\end{aligned}
\tag{I.40}
$$

分子$B_{h,y}$是$Q_y$沿$+(0,h,0)$的导数，故$DQ_y(\theta_0)[d_{\mathrm{risk}}]=-B_{h,y}/\|h\|_2$。风险中的负log概率贡献第二个负号，得到（I.40）中的正号。

为完整起见，候选概率在宽度一条带的边界范数至多为Gaussian平移因子乘行长secant之积；分子的界再乘secant加权方向绝对值之和及二的平方根。（I.27）控制格误差；遗漏尾分别至多为端点Gaussian密度的两倍除以端点、以及该密度两倍乘方向绝对值之和。正概率下界使区间相除合法，教师L1余量只乘最大绝对后验比值一次。

曲率界的验证如下：逐标签负log概率Hessian是后验平均条件负Hessian减后验得分covariance。两项均半正定，差的范数至多二者范数的较大值。似然至多1，将prior得分矩除以统一概率下界，得到上述界。方向没有在评价根之后重新选择。

预定步长从未知精确根出发仍在已认证球内。它保持零偏置，故Gaussian中心对称性使各观察Bernoulli边际严格公平；行logit方差则随行长改变。这不是I.7的固定行范数角度路径。径向曲率及统一风险导数给

$$
\begin{aligned}
t&=10^{-7},\quad \|\theta_*-\theta_0\|_2+t<r,\\
F_8^{q_*}(\theta_*+td_{\mathrm{risk}})-F_8^{q_*}(\theta_*)&\ge\tfrac12m_{16}t^2=2.098275\,10^{-16}>0,\\
R(\theta_*)-R(\theta_*+td_{\mathrm{risk}})&>\nu t=2.01723\,10^{-9},\qquad
R(\theta_*)>2.01723\,10^{-9}.
\end{aligned}
\tag{I.41}
$$

这是已认证邻域的解析推论，不是相近float64损失相减。它认证一个总体局部极小处的严格冲突，不估计典型训练改善大小。KL非负给最后一个不等式。

**保存概率区间给风险量级。** 上述小步界仍是核心存在性证书。教师与候选概率区间还允许确定性外包中心KL，再传入精确根。观察得分信息由完整数据信息支配，因此以下导数界只使用概率比值。

$$
\begin{aligned}
\chi^2(q_*\Vert Q_0)&=\sum_y\frac{(q_*(y)-Q_y(\theta_0))^2}{Q_y(\theta_0)},\quad
I_{\rm obs}(\theta_0)\preceq I_{\rm complete}(\theta_0)\preceq\tfrac14I_{24},\\
\|\nabla R(\theta_0)\|_2&\le\tfrac12\sqrt{\chi^2(q_*\Vert Q_0)},\\
|R(\theta_*)-R(\theta_0)|&\le\tfrac12\sqrt{\chi^2(q_*\Vert Q_0)}\,\eta_*+\tfrac12M_R\eta_*^2,\quad
\eta_*=\epsilon_g/m_r.
\end{aligned}
\tag{I.42}
$$

完整数据条件得分的跨行信息为零，同行信息至多为增广Gaussian输入外积期望的四分之一，该期望是单位矩阵。条件期望给观察得分，Jensen不等式使信息不增。观察得分的模型概率加权均值为零，对目标减模型权重使用Cauchy–Schwarz即得（I.42）。目标概率不重归一化，也不把旧浮点梯度代作完整梯度范数。

仅用保存端点的区间合成给中心风险约0.01043357299286302、chi-square散度约0.021791388777069481、完整梯度界小于0.073809534、根传递余量小于1.334e−9。向外有理数舍入给严格包含区间

$$
R(\theta_*)\in[.0104335716,\ .0104335744].
\tag{I.43}
$$

这一额外界是保存概率区间与认证根距离的确定性合成，没有新积分、拟合或随机样本；它解释了为何不能将（I.41）的保守正风险下界读成实际风险量级。

证据由完整区间有限和及上述解析误差组成。另行实现对八个选定教师标签的完整网格、41个梯度／Hessian类中的两个完整网格作独立重算，不是所有类独立复算。全部端点不确定性进入最终证书。此前经验拟合、弱载荷K2精度失败、K11/K12数值延续及混合profile比较均保持原范围。本结果不认证全局最优、优化器进入邻域的概率、普适临界K或微小保证步长的数值可分辨性。

## I.9. 目标扰动下的定性稳定性 {#local-persistence}

**推论I.1（教师分布的定性持续性）。** 精确教师概率向量在概率单纯形内部有一个邻域，其中每个目标的K8准则都具有附近的秩一模共同旋转严格局部极小，并沿同一固定方向$d_{\mathrm{risk}}\in\mathbb R^{24}$具有负积分风险导数。扰动概率不必保持补集或反序对称性。结论也适用于原两因子模型中所有充分邻近教师参数所诱导的观察分布。该邻域不量化。

**证明。** 使用固定切片$\mathcal S=\{a_0^\top v=0\}$上的全部23个坐标，不使用四坐标对称分支。令$E_{\mathcal S}$的列为切空间的正交归一基，对总和为1的概率使用仿射坐标。有限粒子目标对目标权重线性、对参数光滑，各阶导数由Gaussian多项式支配。以认证根定义

$$
\begin{aligned}
\mathcal F(s,q)&=E_{\mathcal S}^\top\nabla_\theta F_8^q(\theta_*+E_{\mathcal S}s),\\
\mathcal F(0,q_*)&=0,\qquad
D_s\mathcal F(0,q_*)=E_{\mathcal S}^\top H_F(\theta_*)E_{\mathcal S}\succ0.
\end{aligned}
\tag{I.44}
$$

隐函数定理给唯一的局部光滑切片驻定分支$\theta(q)$。取以$\theta_*=(0,a_*,0)$为中心、在$T(b,u,v)=(b,u,-v)$下不变的充分小切片邻域，使其包含在定理的唯一性邻域内；再缩小目标邻域，使分支始终留在其中。对任意目标，潜变量反射均保持目标及固定切片。因此切片驻点的反射是同一邻域内的另一个切片驻点，唯一性使二者相等。连续性使$u(q)$保持非零，故

$$
\begin{aligned}
F_8^q(T\theta)&=F_8^q(\theta),\qquad T\mathcal S=\mathcal S,\qquad T\theta(q)=\theta(q),\\
v(q)&=0,\qquad \operatorname{rank}W(q)=1.
\end{aligned}
$$

切片驻定意味着环境梯度与切片法向量成比例；旋转不变性使梯度与轨道切向量正交，而原横截条件持续保持，使法向量与此切向量的内积非零。因此环境梯度为零，不对扰动目标施加零偏置或成对行长约束。切片正曲率由连续性保持，给模旋转严格局部极小。

$$
\begin{aligned}
\theta(q)&=\theta_*+E_{\mathcal S}s(q),\qquad
\nabla F_8^q(\theta(q))=0,\\
R_q(\theta)&=D_{\rm KL}(q\Vert p_\theta),\qquad
D R_q(\theta(q))[d_{\mathrm{risk}}]<0.
\end{aligned}
\tag{I.45}
$$

最后一个严格不等式由风险方向导数的连续性得到，因为在单纯形内部教师向量和认证根附近，各概率仍为正；它也给正风险。对正确指定的教师子类，每个观察概率随教师参数连续：标签似然连续且被1支配，可用支配收敛定理。因此概率邻域的逆像是原教师参数的一个邻域，结论在其中同样成立。这不声称每个附近概率向量都属于原两因子模型。

这是非退化性的定性推论，不是新教师网格实验或量化适用区间，不增加K范围。附近分支仍为秩一，但其偏置及成对行长不必保持原对称性。尤其不将I.8.5原认证根及其风险步上的公平观察边际推广到扰动教师或其根。

## I.10. 同一精确根上的固定边际角路径冲突 {#same-root-angle}

本节在I.8的同一精确教师$q_*$、中心$\theta_0=(0,a_0,0)$及真根$\theta_*=(0,a_*,0)$上加强局部见证，不重新拟合根。新路径保持各行长度，仅改变相对角度；它诊断的是完整模型中的局部极小，并非对原学习问题增加约束。

### I.10.1. 方向与完整路径导数

定义整数向量及数学单位角方向

$$
\begin{aligned}
(u_1,u_2,u_3,u_4)&=(-8440215,-6758298,-4585241,-1673686),\\
u_{9-i}&=-u_i\quad(1\le i\le4),\qquad S=\sum_i u_i^2=281474961915412,\\
d=d_{\mathrm{ang}}^{(8)}&=u/\sqrt S,\qquad \|d\|_2=1,\qquad \mathbf1^\top d=0.
\end{aligned}
\tag{I.46}
$$

该方向由先前P点保存的风险角矩阵数值信息启发；其精确构造在新增积分评价前固定，评价后没有更换。这是探索性的存在性见证，不是盲选方向。所有二次系数均使用$(u_i-u_j)^2/S$等精确比值，不舍入$\sqrt S$。角空间的单位归一化不同于Cartesian路径速度的单位归一化。此方向也不同于I.8.5的$d_{\mathrm{risk}}$及I.6–I.7的$d_{\mathrm{ang}}^{(12)}$。

对共线载荷$a$，取独立标准Gaussian变量$Z,X$，令$\eta_i(t)=a_i\{Z\cos(td_i)+X\sin(td_i)\}$。记$p_i(z)=\sigma(a_i z)$、$r_i=y_i-p_i(z)$、$D_i=p_i(1-p_i)$、$A_y(z)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$及$Q_y=E A_y(Z)$。本段另记$U=\sum_i a_i d_i r_i$、$V=\sum_i a_i d_i^2r_i$、$G=\sum_i a_i r_i$和$T=\sum_i a_i^2d_i^2D_i$。对似然求导并积掉$X$，得到

$$
\begin{aligned}
W_i(t)&=a_i(\cos(td_i),\sin(td_i)),\qquad b_i(t)=0,\\
\dot\eta_i(0)&=a_i d_iX,\qquad \ddot\eta_i(0)=-a_i d_i^2Z,\\
Q_y'(0)&=0,\qquad Q_y''(0)=E B_y(Z),\\
B_y(z)&=A_y(z)\{U^2-T-zV\}.
\end{aligned}
\tag{I.47}
$$

第一导数由$X$反射对称性为零，故这不是一阶负风险方向。加速度项$-zV$不能删去：仅使用Cartesian横向Hessian会对另一条路径求导。实sigmoid导数有界以及Gaussian多项式包络保证上述微分合法。

令$P_y(z)=\sum_{i<j}a_ia_j(d_i-d_j)^2r_ir_j$。展开给出$P_y=GV-U^2$、$A_y'=A_yG$及$V'=-T$，因而

$$
\begin{aligned}
B_y+A_yP_y&=(A_yV)'-zA_yV
=\{\varphi A_yV\}'/\varphi,\\
Q_y''(0)&=-E[A_y(Z)P_y(Z)],\qquad N_y:=E[A_y(Z)P_y(Z)],\\
\mathcal H_R(a;d)&:=\left.\frac{d^2R(\theta(t))}{dt^2}\right|_{t=0}
=\sum_yq_*(y)\frac{N_y}{Q_y}
=\sum_yq_*(y)E_{\pi_y}P_y,\\
\pi_y(dz)&=A_y(z)\varphi(z)\,dz/Q_y.
\end{aligned}
\tag{I.48}
$$

由$A_y\le1$及$V$有界，$\varphi A_yV$在实轴两端消失。第一行是Gaussian权下的全导数，第二行仅在积分后成立；并不声称被积函数$B_y$和$-A_yP_y$逐点相等。对负log概率求导提供第二个负号，$Q_y'(0)=0$消去一阶导数平方项。常量$d$给$P_y=0$，符合共同旋转不变性。

### I.10.2. 中心外包与另一完整路径表达

在（I.21）的精确中心，$|a_{0,i}|<1$且$\sum_{i<j}(d_i-d_j)^2=8$，故$|P_y|\le8$。全部256标签逐一评价。在闭条带$|\operatorname{Im}z|\le1$上令$c=\sec(1/2)$。对$s\ge0$、$|v|\le1$，不等式$|1+s e^{iv}|\ge(1+s)\cos(v/2)$将每个复Bernoulli因子界为对应实轴值的$c$倍。因此$|A_y|\le c^8$、$|r_i|\le c$且$|D_i|\le c^2/4$。两种分子表达各有自己的单侧边界积分及遗漏格点界：

$$
\begin{aligned}
h&=1/8,\quad N=88,\quad L=Nh=11,\quad c=\sec(1/2),\\
M_{\rm pair}&=e^{1/2}c^8\,8c^2,\qquad M_Q=e^{1/2}c^8,\\
M_{\rm path}&=e^{1/2}c^8\{8c^2+c^2/4+c\sqrt2\},\\
T_{\rm pair}&=16\varphi(L)/L,\qquad T_Q=2\varphi(L)/L,\\
T_{\rm path}&=(8+1/4)2\varphi(L)/L+2\varphi(L),\\
\epsilon_j&=\frac{2M_j}{e^{2\pi/h}-1}+T_j,\qquad j\in\{{\rm pair},{\rm path},Q\}.
\end{aligned}
\tag{I.49}
$$

配对式的被积函数为$\varphi A_yP_y$，另一表达为$-\varphi B_y$。条带边界上$|U|\le\sqrt8c$、$|T|\le c^2/4$、$|V|\le c$且$E|Z\pm i|\le\sqrt2$，由此得到$M_{\rm path}$。轮廓平移给Fourier衰减$M_j e^{-|\xi|}$，对非零Poisson频率求和给alias项，与（I.27）一致；Gaussian包络使轮廓端边界消失。实轴上两种分子因子的绝对值分别不超过$8$及$8+1/4+|z|$。由$\varphi(x)$和$x\varphi(x)$在$x\ge L\ge1$上的单调性，遗漏全权节点$|k|>N$由相邻连续尾外包，再用$\int_L^\infty x\varphi(x)dx=\varphi(L)$及$\int_L^\infty\varphi(x)dx\le\varphi(L)/L$得到所列尾界。

每个177节点和的两端均取全权，使用192-bit Arb算术。向外算术区间与解析误差分别保留。配对式每标签误差约为$1.44001098659947\,10^{-20}$，它不是最终曲率的误差。另一实现由（I.47）构造完整路径二阶导数，保留加速度，独立按整数标签相乘概率因子，使用（I.49）中自己的误差界，并重新计算$Q_y$。

令$[F_y]$为配对有限和区间，$[Q_y]$为I.8.5已使用的严格正真实概率外包。令$\widehat q_y$为保存教师有限和区间两个向外端点的精确中点，不对这些中点重归一化。原教师证书给出$\|q_*-\widehat q\|_1<\epsilon_q:=4.894\,10^{-21}$。区间合成为

$$
\begin{aligned}
[N_y]&=[F_y]+[-\epsilon_{\rm pair}^{\rm up},\epsilon_{\rm pair}^{\rm up}],\\
[\mathcal H_0]&=\sum_y\widehat q_y[N_y]/[Q_y]+[-8\epsilon_q,8\epsilon_q],\\
|N_y/Q_y|&=|E_{\pi_y}P_y|\le8,\qquad
8\epsilon_q=3.9152\,10^{-20}.
\end{aligned}
\tag{I.50}
$$

其中$\epsilon^{\rm up}$为认证的向外上端点。教师余量通过真实后验比值界只计一次；每标签分子误差则穿过相除和加权。另一中心外包换用完整路径有限和及$\epsilon_{\rm path}$，但仍使用原$[Q_y]$。新计算的概率用于交叉核对，不替换该合成中的分母。

分别加入对应解析误差后，全部256对分子区间相交，全部256个重算概率区间也与原区间相交。这些检查比较同一积分的外包，不要求有限和相同，也不替代解析误差证明。两种实现共享Arb、教师输入和根证书。另有精确有理算术检查独立复合保存端点，不重算超越函数或积分。因此这些是不同表达与实现的核对，而不是不同算术后端的端到端独立认证。

### I.10.3. 向认证真根的标量转移

所需的是固定$q_*,b=0,d$下单个标量$\mathcal H_R(a;d)$的变化界，不是完整风险Hessian的界。设所考虑线段上$\max_i|a_i|\le A$，共有$m$行且$\|d\|_2=1$。在实轴积分$(zA_y\varphi)'$，得到

$$
\begin{aligned}
E_{\pi_y}Z^2&=1+E_{\pi_y}[Z\,\partial_z\log A_y],\qquad
|\partial_z\log A_y|\le mA,\\
\mu_y^2&\le E_{\pi_y}Z^2\le1+mA\mu_y,\qquad \mu_y:=E_{\pi_y}|Z|,\\
\mu_y&\le M_1:=\frac{mA+\sqrt{m^2A^2+4}}2.
\end{aligned}
\tag{I.51}
$$

Gaussian衰减使边界项消失，Cauchy–Schwarz不等式给第二行。这个后验矩估计不需要$Q_y$的数值下界。记$c_{ij}=(d_i-d_j)^2$，则$\sum_{i<j}c_{ij}\le m$且$|P_y|\le mA^2$。对单位载荷变化方向$h_0$直接微分，得到

$$
\begin{aligned}
D_aP_y[h_0]&=\sum_{i<j}c_{ij}\big[(h_{0,i}a_j+a_ih_{0,j})r_ir_j\\
&\hspace{20mm}-za_ia_j(D_ih_{0,i}r_j+r_iD_jh_{0,j})\big],\\
|D_aP_y[h_0]|&\le2mA+(mA^2/2)|z|,\qquad
|D_a\log A_y[h_0]|\le\sqrt m|z|,\\
D_a E_{\pi_y}P_y[h_0]&=E_{\pi_y}D_aP_y[h_0]
+\operatorname{Cov}_{\pi_y}(P_y,D_a\log A_y[h_0]),\\
|\operatorname{Cov}_{\pi_y}(P_y,D_a\log A_y[h_0])|
&\le2m^{3/2}A^2 E_{\pi_y}|Z|.
\end{aligned}
\tag{I.52}
$$

协方差保留了后验分母的微分；用$|P_y|\le mA^2$分别界定它的两项即得末行。按固定概率向量加权，再沿载荷线段积分导数界，得到

$$
\begin{aligned}
|\mathcal H_R(a;d)-\mathcal H_R(a';d)|&\le L_{\rm ang}\|a-a'\|_2,\\
L_{\rm ang}&=2mA+(mA^2/2+2m^{3/2}A^2)M_1,\\
L_{\rm ang}\big|_{m=8,A=1}
&=16+(4+16\sqrt8)(8+\sqrt{68})/2<417.
\end{aligned}
\tag{I.53}
$$

最后的帽值可精确有理核对：$\sqrt8<2829/1000$及$\sqrt{68}<8247/1000$给上界$52024513/125000<417$。$a_0\pm(1.805\,10^{-8})\mathbf1$各坐标均在$(0,1)$内。由（I.39），连接中心与同一真根的线段满足$A=1$，且没有偏置或教师变化。因此

$$
|\mathcal H_R(a_*;d)-\mathcal H_R(a_0;d)|
<417(1.805\,10^{-8})=7.52685\,10^{-6}.
\tag{I.54}
$$

这里没有将（I.40）的风险Hessian范数帽或（I.37）的有限目标变化常数当作风险曲率变化界。合成（I.50）及（I.54），得到以下保守十进制外包，每一行均包含两种表达对应的保存dyadic证书：

| 对象 | 下端点 | 上端点 |
|---|---:|---:|
| $\mathcal H_R(a_0;d)$ | $-0.004511298531248709$ | $-0.004511298531248699$ |
| $\mathcal H_R(a_*;d)$ | $-0.004518825382$ | $-0.004503771681$ |

### I.10.4. 固定边际的局部冲突

最后在（I.47）中取$a=a_*$。上述外包与反射对称性给$R'(0)=0$及$R''(0)<-0.00450377$。二阶可微性使充分小的双侧非零步均降低风险。又因$d$非常量且各$a_{*,i}>0$，存在两行的子式$a_{*,i}a_{*,j}\sin(t(d_j-d_i))$，在所有充分小非零$t$上非零。这些秩二点不在原秩一根的共同旋转轨道内；I.8.4已证的完整空间模共同旋转严格局部极小性质遂给出：存在$\varepsilon>0$，使

$$
\begin{gathered}
R'(0)=0,\qquad R''(0)<-0.00450377,\\
0<|t|<\varepsilon\quad\Longrightarrow\quad
R(\theta(t))<R(\theta_*),\\
F_8^{q_*}(\theta(t))>F_8^{q_*}(\theta_*),\quad\operatorname{rank}W(t)=2.
\end{gathered}
\tag{I.55}
$$

整条路径保持每个候选logit分布$N(0,a_{*,i}^2)$及每个观察边际$\operatorname{Bernoulli}(1/2)$。后者正确匹配教师，候选logit方差则不必等于教师的$(4/5)^2$。非对角协方差按$a_{*,i}a_{*,j}\cos(t(d_i-d_j))$变化，因此这是依赖变化而非共同旋转。它在原精确根处给出定理3的固定logit边际结论，并保留（I.41）所提供的另一项径向指定步长保证。

本节不指定数值$\varepsilon$、有限步长收益、完整角谱证书或K11/K12转变。尤其不能将曲率界当作$t=.02$的认证损失变化。I.9仍是针对径向方向的定性结论；仅凭I.10不将其全部角路径性质推广到扰动教师。I.11对一个指定零偏置族给出这一延伸。精确输入、有限区间及证据映射随AngularWitness-v1证书保存；本节证明依赖所列标量恒等式、积分包络和转移论证，不依赖执行日志。

## I.11. 变化教师族中的定量持续性 {#quantitative-teacher-family}

::: {#teacher-family-full-statement}

**推论（明确教师族中的定量局部持续性）。** 保持教师偏置为零、全部教师行长为$4/5$，仅将第一条教师角度改为$-.7+\lambda$。令$q_\lambda$为其观察律，$\mathcal R_\lambda(\theta)=D(q_\lambda\Vert Q_\theta)$。在区间$|\lambda|\le10^{-3}$上，存在与定理3真根相连接的一条光滑分支$\theta_\lambda=(0,a(\lambda),0)$，其中每个$\theta_\lambda$在完整参数空间中模共同潜变量旋转后均为对应$F_8^{q_\lambda}$的严格局部极小。使用(14)中同一固定方向$d=d_{\mathrm{ang}}^{(8)}$，令$\theta_\lambda(t)=(0,W_\lambda(t))$、$f_\lambda(t)=F_8^{q_\lambda}(\theta_\lambda(t))$及$r_\lambda(t)=\mathcal R_\lambda(\theta_\lambda(t))$，则

$$
\begin{aligned}
W_{\lambda,i}(t)&=a_i(\lambda)(\cos(td_i),\sin(td_i)),\\
f_\lambda'(0)&=r_\lambda'(0)=0,\\
f_\lambda''(0)&>.0011,\qquad
r_\lambda''(0)\le-.004351070278<0.
\end{aligned}
$$

对每个$\lambda$，存在$\varepsilon_\lambda>0$，使所有$0<|t|<\varepsilon_\lambda$均提高$F_8^{q_\lambda}$、降低其自身真实风险$\mathcal R_\lambda$，并使$\operatorname{rank}W_\lambda(t)=2$。路径保持候选logit边际$N(0,a_i(\lambda)^2)$和正确的公平观察边际。

:::

### I.11.1. 带联合余项的教师展开

保持(13)的教师，只改变其第一条角度，记$\rho=4/5$、$\phi=(-.7,-.5,-.3,-.1,.1,.3,.5,.7)$。对$z\in\mathbb R^2$，以$A_y^*(z;\lambda)$表示其Bernoulli乘积。仅对第一个因子求导得到

$$
\begin{aligned}
t_i^*&=\rho\{\cos(\phi_i+\lambda\mathbf1_{i=1})z_1
+\sin(\phi_i+\lambda\mathbf1_{i=1})z_2\},\\
\partial_\lambda A_y^*&=A_y^*(y_1-\sigma(t_1^*))\partial_\lambda t_1^*,
\qquad q_\lambda(y)=E A_y^*(Z;\lambda),\\
q_\lambda&=q_0+\lambda q'_0+e_\lambda,\qquad
\sum_y e_{\lambda,y}=0,\\
\|q''_\lambda\|_1&\le\frac{\rho^2}{3\sqrt3}
+\frac\rho2\sqrt{2/\pi}<.442323,\qquad
\|e_\lambda\|_1\le .2211615\lambda^2.
\end{aligned}
\tag{I.56}
$$

二阶导数界对所有实数$\lambda$成立：先对第一位的两个标签导数求绝对值并求和，再对其余七个因子求和，使用$\sup|\sigma''|=1/(6\sqrt3)$、$\sup\sigma'=1/4$、$E(\partial_\lambda t_1^*)^2=\rho^2$及$E|\partial_\lambda^2t_1^*|=\rho\sqrt{2/\pi}$。精确概率质量给出余项零和。Gaussian反射保留互补对称和公平边际，但不保留反序对称。

全部256个$q_0,q'_0$分量由间距$\delta=1/8$、截断$L=10$和192-bit算术的二维全权梯形规则包围。对模不超过$\eta=1$的复坐标平移，令$\beta_i=\rho\eta(|\cos\phi_i|+|\sin\phi_i|)/2$。两个Bernoulli因子的模长和至多为$\sec\beta_i$；导数因子由$\rho\sec^2\beta_1/2$乘一个线性Gaussian包络控制。令$s=e^{-2\pi\eta/\delta}$、$\kappa=|\sin\phi_1|+|\cos\phi_1|$，显式边界积分与省略节点的上界给出

$$
\begin{aligned}
M_q&=e^{\eta^2}\prod_i\sec\beta_i,\\
M_{q'}&=e^{\eta^2}\frac\rho2\sec^2\beta_1
\left(\prod_{i>1}\sec\beta_i\right)\kappa(\sqrt{2/\pi}+\eta),\\
S_0&=1+\delta\varphi(0),\quad S_1=\sqrt{2/\pi}+4\delta\varphi(1),\\
T_0&=2\varphi(L)/L,\quad T_1=2\varphi(L),\\
T_q&=2S_0T_0,\qquad T_{q'}=(\rho/2)\kappa(T_1S_0+T_0S_1),\\
\epsilon_j&\le M_j\frac{4s}{(1-s)^2}+T_j,\qquad j\in\{q,q'\},\\
\|q_0-q_0^f\|_1&<4.894\times10^{-21},\qquad
\|q'_0-(q'_0)^f\|_1<5.927\times10^{-21}.
\end{aligned}
\tag{I.57}
$$

alias系数同时求和单轴和双轴频率。离散尾在截断之外使用Gaussian单调尾，在整轴一阶绝对矩上使用$|z|\varphi(z)$的总变差。这些是对标签联合的$L^1$界，只计一次；算术区间另行保留。得到的分量包围没有经过平均化或重归一。

### I.11.2. 八个自由候选行长参数

新候选不能用反序配对约化。我们在I.3和I.8.3的同一Laplace表示中保留全部八个自由候选行长参数$a_i$。在$u=e^v$处，令$p_i=\sigma(a_i z)$、$\widetilde p_i=p_i-1/2$、$\widetilde y_i=y_i-1/2$、$t=uA_y e^{-uA_y}$及$\Psi=E e^{-uA_y}$。完整中心化矩为

$$
\begin{aligned}
T_l&=E[tz^l],\quad \widetilde P_l=E[tz^l\widetilde p],\quad
\widetilde{PP}_l=E[tz^l\widetilde p\widetilde p^\top],\\
R_1&=T_1\widetilde y-\widetilde P_1,\\
\Gamma_l&=T_l\widetilde y\widetilde y^\top
-\widetilde y\widetilde P_l^\top-\widetilde P_l\widetilde y^\top
+\widetilde{PP}_l,\\
D_l&=\operatorname{diag}(T_l/4-\operatorname{diag}\widetilde{PP}_l),\qquad
c_K=K\Psi^{K-1},\quad d_K=K(K-1)\Psi^{K-2}.
\end{aligned}
\tag{I.58}
$$

$T,\widetilde P$使用$l=0,1,2$，$\widetilde{PP}$使用$l=0,2$，通过对称性保留全部矩阵元素。条件梯度和Hessian块是下列各式在全实$v$轴上的积分：

$$
\begin{aligned}
g_y(v)&=-c_KR_1,\\
H_{bb,y}(v)&=c_KD_0-d_K(T_0\widetilde{PP}_0-\widetilde P_0\widetilde P_0^\top),\\
H_{aa,y}(v)&=c_KD_2-d_K(T_0\Gamma_2-R_1R_1^\top),\\
H_{\perp,y}(v)&=c_KD_0-d_KT_0\Gamma_0,\qquad K=8.
\end{aligned}
\tag{I.59}
$$

中心化是矩的精确代数改写。第二Gaussian坐标均值为零，因此横向块没有平均得分平方项。反射$A_{1-y}(z)=A_y(-z)$使这四类量在互补标签下不变，因而可以用权重$q_y+q_{1-y}$合成128个代表。这里不使用41类配对约化。I.8的围道和离散尾界使用八个行长参数各自的上端点，计算$\sum_i\bar a_i$及$\sum_i\sec^2(\bar a_i d_z/2)$。相应的统一梯度误差和算子范数误差记为$\epsilon_g,\epsilon_H,\epsilon_P$；区间依赖膨胀及算术半径仍保留在有限矩阵中。

### I.11.3. 覆盖整个区间的同一根盒

令$a_0$为(I.21)中的精确中心，$h=.001$、$X=a_0+\prod_i[-r_i,r_i]$、$r=(.00075,.00015,\ldots,.00015)$及$t_h=.2211615h^2$。用$g_y^f,H_y^f$表示有限条件包围。为保留教师导数的符号，先合成

$$
\begin{aligned}
G_1^f(X)&=\sum_y[q'_{0,y}]g_y^f(X),\\
H_{j,\mathrm{fin}}(X)&=\sum_y[q_{0,y}]H_{j,y}^f(X)
+[-h,h]\sum_y[q'_{0,y}]H_{j,y}^f(X),\\
E_H&=\epsilon_H(1+hQ_1)+146.25t_h,\qquad
E_P=\epsilon_P(1+hQ_1)+8.25t_h,\quad Q_1\ge\|q'_0\|_1.
\end{aligned}
\tag{I.60}
$$

其中$j=bb,aa,\perp$；$E_H$适用于完整偏置／径向Hessian的两个主子块。教师分量区间已包含(I.57)。条件量的全局上帽沿用I.8的8、146.25和8.25。若在求和之前，就为每个线性化权重分别乘上不相关的$\lambda$区间，会丢失可用的符号抵消。

使用证书中的固定有理对称预条件矩阵$A$，其正定性由严格区间LDL核验，令$T_\lambda(a)=a-AG(a,\lambda)$。对行向量$A_i$，令$b_i$为$(Ag_y^f(X))_i$联合值域的中点。余项质量为零，允许对每个标签减去同一个常数，因而

$$
\begin{aligned}
\widehat C_i&=\min\left\{8\|A_i\|_2,
\max_y\sup|(Ag_y^f(X))_i-b_i|+\|A_i\|_2\epsilon_g\right\},\\
Y_i&=7.572\times10^{-10}\|A_i\|_2
+h\sup|(AG_1^f(X))_i|
+h\|A_i\|_2\epsilon_gQ_1+t_h\widehat C_i.
\end{aligned}
\tag{I.61}
$$

旧中心的非零残差仍然计入。尽管$G_1^f$覆盖整个$X$，$Y_i$只需要控制$a_0$处的位移。对$T_\lambda$的导数，加入算子范数误差预算得到

$$
\begin{aligned}
Z_i&=\sum_j\sup|(I-AH_{aa,\mathrm{fin}}(X))_{ij}|r_j
+\|A_i\|_2E_H\|r\|_2,\\
\max_i Z_i/r_i&<.264019,\qquad
\max_i(Y_i+Z_i)/r_i<.753428,\\
\min_i(r_i-Y_i-Z_i)&>8.37210\times10^{-5}.
\end{aligned}
\tag{I.62}
$$

在半径加权最大范数下，Banach定理对每个$\lambda$给出$X$中唯一的径向根，且收缩一致成立。$A$可逆使不动点等价于根，一致收缩给出连续性。由(I.39)，旧真根在$X$内部，唯一性因而确定$a(0)=a_*$。下文的径向Hessian一致可逆，隐函数定理遂给出光滑性；各局部分支由唯一性相接。

### I.11.4. 从原矩阵不确定性取得完整稳定性

对每个原始对称$8\times8$有限区间矩阵，以$l_{ij},u_{ij}$表示精确端点，在投影之前构造中点：

$$
\begin{aligned}
(M_0)_{ij}&=(l_{ij}+u_{ij})/2,\qquad B_{ij}=(u_{ij}-l_{ij})/2,\\
\|H_{j,\mathrm{fin}}-M_0\|_{\rm op}&\le\|B\|_F
\le\bar\rho_F,\qquad \sum_{i,j}B_{ij}^2\le\bar\rho_F^2.
\end{aligned}
\tag{I.63}
$$

最后的平方比较使用精确有理算术，包含全部64个元素。令$U_0$为由$u_0=a_0/\|a_0\|$、$v_0=u_0+e_1$构造的Householder矩阵的后七列。其精确正交性及$U_0^\top a_0=0$给出单位算子范数。对每个块各自的$M_0,\bar\rho_F$，严格区间LDL证明

$$
\begin{aligned}
M_{0,bb}-(.1+E_H+\bar\rho_{F,bb})I_8&\succ0,\\
M_{0,aa}-(.02+E_H+\bar\rho_{F,aa})I_8&\succ0,\\
U_0^\top M_{0,\perp}U_0-(.005+E_P+\bar\rho_{F,\perp})I_7&\succ0.
\end{aligned}
\tag{I.64}
$$

先界定原矩阵的不确定性，可以避免先投影整个区间矩阵造成的额外依赖损失。门槛保持不变。与I.2相同，标签互补与潜坐标反射使其他梯度分量和混合块为零。在真根处，旋转不变性给出Ward恒等式：

$$
\begin{gathered}
\nabla F_8^{q_\lambda}(0,a(\lambda),0)=0,\qquad
H_\perp a(\lambda)=0,\qquad a_0^\top a(\lambda)>0,\\
H_{bb}\succ .1I_8,\qquad H_{aa}\succ .02I_8,\qquad
U_0^\top H_\perp U_0\succ .005I_7.
\end{gathered}
\tag{I.65}
$$

固定切片在整个$X$上与旋转方向横截，因此切片正性给出完整空间中模共同旋转的严格局部极小。数值.005是$a_0^\perp$上的下界，不声称其原值直接适用于变化的$a(\lambda)^\perp$。

### I.11.5. 每个教师自身的角风险

使用(I.46)定义的固定角方向$d=d_{\mathrm{ang}}^{(8)}$，并采用I.10中保留路径加速度的完整角路径。对每个$a\in X$，分别外包全部256项

$$
\begin{aligned}
Q_y(a)&=E A_y(Z;a)>0,\\
N_y(a)&=E[A_y(Z;a)P_y(Z;a,d)],\qquad
\rho_y(a)=N_y(a)/Q_y(a),\\
\sum_y A_y(z;a)P_y(z;a,d)&=0,\qquad \sum_y N_y(a)=0.
\end{aligned}
\tag{I.66}
$$

节点恒等式来自独立Bernoulli残差交叉乘积的求和。由于$0<a_i<1$，(I.49)中的177节点规则以及分开的概率／分子界直接适用于$X$的每个成员。这样可直接包围候选的移动，不必支付417乘位移的转移代价。正分母、解析误差和算术区间全部计入。令$B_*$为$\rho_y(X)$联合值域的中点，则

$$
\begin{aligned}
R_0(a)&=\sum_y q_{0,y}\rho_y(a)
=\sum_y(q_{0,y}-Q_y(a))\rho_y(a),\\
[r_\lambda''(0)]&\subset [R_0(X)]
+[-h,h]\sum_y[q'_{0,y}]\rho_y(X)\\
&\hspace{8mm}+[-t_h,t_h]\max_y\sup|\rho_y(X)-B_*|\\
&\subset[-.004671548834,-.004351070278].
\end{aligned}
\tag{I.67}
$$

对$R_0$的两个合法包围取交集可收紧其界；仅凭重叠不能认证任意输入。两种中心化都使用精确零质量恒等式，没有改变目标概率。反射给出$r_\lambda'(0)=0$。为控制有限目标曲率，令$v=\operatorname{diag}(a)d$、$c=(a_0^\top v)/(a_0^\top a)$及$w=v-ca$。由(I.65)，

$$
\begin{aligned}
w&\in a_0^\perp,\qquad
f_\lambda''(0)=v^\top H_\perp v=w^\top H_\perp w\\
&>.005\|w\|^2\ge .005a_{\min}^2\|d-c\mathbf1\|^2
\ge .005a_{\min}^2>.0011,\\
a_{\min}&=\min_i(a_{0,i}-r_i),\qquad
\|d-c\mathbf1\|^2=1+8c^2.
\end{aligned}
\tag{I.68}
$$

驻定性给出$f_\lambda'(0)=0$。二阶可微性由此证明教师族推论，以及各教师自身充分小双侧步上的后果。正行长和非常量$d$给出秩二，固定行长则保持候选Gaussian logit及公平观察边际。这里不推出共同的数值步长半径或整个族的KL区间。

### I.11.6. 非平凡性与认证边界

Taylor不等式及$\|q'_0\|_1$的包围表明观察律确实变化：

$$
\begin{aligned}
.159347929859&\le\frac{\|q_\lambda-q_0\|_1}{|\lambda|}
\le .159790252860\quad(0<|\lambda|\le .001),\\
\|q_{\pm.001}-q_0\|_1&\in[.000159347929,.000159790253].
\end{aligned}
\tag{I.69}
$$

七个第一行spin成对矩的导数在加入$.442323h$预算后仍严格为正；其余七个观察量的联合律保持不变。因此，原先相等的$(1,2)$与$(8,7)$矩在每个非零$\lambda$下分离，破坏反序对称。为包围隐式导数$a'=-H_{aa}^{-1}G_\lambda$，逐坐标选择条件梯度联合值域的中点$g_C$，令$C_g=\min\{8,\max_y\sup\|g_y^f(X)-g_C\|_2+\epsilon_g\}$。由于$q'_\lambda-q'_0$总质量为零且$L^1$范数至多为$.442323h$，用$G_1^f$近似$G_\lambda$的误差至多为$\epsilon_gQ_1+C_g\,.442323h$。将有限$H_{aa}$包围的每个元素扩大$E_H$后，进行区间线性求解，得到

$$
\begin{aligned}
a'_1(\lambda)-a'_8(\lambda)&\in[.429945195966,.565673464386],\\
\frac{a_1(\lambda)-a_8(\lambda)}{\lambda}
&\in[.429945195966,.565673464386]\quad(\lambda\ne0),\\
|a_1(\pm.001)-a_8(\pm.001)|&\in[.000429945195,.000565673465].
\end{aligned}
\tag{I.70}
$$

其中$a_1(0)=a_8(0)$来自旧真根的精确对称性。因此，该分支离开了配对行长子空间。三个预定盒半宽的结果保留如下；“未决”表示外包不足，不是反例。

|半宽$h$|原根包含|原$bb/aa/\perp$ LDL|自身风险符号|中点谱界之后|
|---:|---|---|---|---|
|$.01$|未决|通过／未决／未决|负|未决|
|$.001$|通过|通过／通过／未决|负|完整证书|
|$.0001$|通过|通过／通过／通过|负|完整证书|

中点论证是在保存矩阵上进行的后续确定性细化，原失败和全部门槛均保留。到达$\pm.1$的七个浮点开发点不认证该整个区间。(I.69)中的端点观察律变化将已证结果定位于较小的定量局部尺度。精确区间与表达式核验保存在R03TeacherFamily-v1中；本证明不依赖开发轨迹。
