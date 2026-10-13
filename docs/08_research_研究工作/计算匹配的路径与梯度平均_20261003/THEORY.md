# 相同模拟预算的两种分配：更新内平均与输出端参数平均

本合同针对固定四份旧完整数据、六个物理相关参数、K100及单位无关尺度。IN4是一条训练路径，每步在同一参数处平均四份独立K100损失/梯度后作一次Adam与可行性修复。OUT4是四条各用一份流的独立训练路径，最后等权平均四个物理TAIL参数。两策略复用同一个新prior四元组以配对，但各自执行自己的程序。全部论断条件于旧数据；本文不新增RNG、模型积分或训练。

## 1. 保持的对象与没有保持的对象

设固定完整数据为 \(D_d\)，第t步的固定循环批次为 \(\mathcal B_{d,t}\)。记一份K100粒子输入在参数 \(\rho\) 处的损失/路径梯度为 \(\ell_K(\rho;\mathcal B,Z)\)、\(g_K(\rho;\mathcal B,Z)\)。四份Z彼此独立。IN4使用

$$
\bar g_4(\rho;\mathcal B)
=\frac14\sum_{j=0}^3g_K(\rho;\mathcal B,Z_j),\qquad
E\bar g_4=\nabla F_{\mathcal B,K}(\rho),\qquad
\operatorname{Cov}\bar g_4=\frac14\Sigma_{\mathcal B}(\rho).
\tag{A.1}
$$

在固定均值、单位对角、谱下界.1的紧矩阵类上，Cholesky导数有界，log-sigmoid导数有界；路径梯度被常数乘active高斯粒子的最大范数控制，所以这些二阶矩存在。该模型固定K和维数，不借此声称增长维数或Adam收敛率。

每个成员路径也使用同一固定参数/批次的期望目标。但运行中的四个成员参数通常不同，不能把四个成员当前梯度平均当作IN4当前状态的梯度，不能共享已算梯度。IN4与成员路径的Adam矩、proposal和修复分别更新。外平均不是把400粒子放进一次log，也不是在同一条路径作四次更新。

批次是固定循环，不是每步随机抽取完整数据的无偏梯度；冻结一个参数后，对全部固定批次平均才给完整 \(F_{D_d,K}\)。实际主实现为float32，实算术身份与预定数值资格分开。

## 2. OUT4严格保持单成员的条件参数均值

令 \(\widehat\rho_{d,j}\) 为第j条单bank路径的TAIL，包含同样的初始化、512步、数据日程、Adam和修复规则。TAIL是最后64个post-update物理相关向量的均值，不是最后一次迭代。独立的是四条完整prior流；各路径没有共享优化器状态，固定数据本身不破坏条件独立性。因此给定 \(D_d\)，四个成员独立同分布。记

$$
\mu_d=E_{\rm prior}[\widehat\rho_{d,j}\mid D_d],
\qquad
\Sigma_d=\operatorname{Cov}_{\rm prior}(\widehat\rho_{d,j}\mid D_d),
\qquad
\widehat\rho_{d,\mathrm{OUT4}}=\frac14\sum_{j=0}^3\widehat\rho_{d,j}.
\tag{A.2}
$$

则准确地有

$$
\begin{aligned}
E[\widehat\rho_{d,\mathrm{OUT4}}\mid D_d]&=\mu_d,\\
\operatorname{Cov}(\widehat\rho_{d,\mathrm{OUT4}}\mid D_d)&=\Sigma_d/4,\\
E[\|\widehat\rho_{d,\mathrm{OUT4}}-\rho_*\|^2\mid D_d]
&=\|\mu_d-\rho_*\|^2+\operatorname{tr}\Sigma_d/4.
\end{aligned}
\tag{A.3}
$$

式(A.3)针对参数分布，不是观察PMF或joint KL。给定新quartet后，OUT4是一个参数均值模型 \(Q_{\bar C}\)，一般不等于四个预测分布的混合 \(\frac14\sum_jQ_{C_j}\)。因而不能用混合预测器的KL凸性替OUT4建立风险优势。本轮不积分成员PMF，也不估计成员风险。

相关矩阵 \(C(\rho)\) 对六向量线性，集合
\(\mathcal E_{.1}=\{C=C^\top:\operatorname{diag}C=1,\ C\succeq .1I\}\)
凸。因此只要成员物理状态可行，TAIL和OUT4在实算术下均可行；均值的最小特征值不小于成员最小特征值的均值。实现中的向外容差和实际谱检查另列，不能由凸性取消float32资格检查。

IN4的终点分布没有对应的式(A.3)保证。即使两策略使用同样四份随机输入，非线性目标、Adam和修复也使“先平均再优化”与“独立优化后平均”不同。本文不预设哪个策略风险更低。

## 3. 配对、独立单位与主estimand

每份数据有16个独立新prior四元组，r为quartet编号；同r内部IN4与OUT4配对，跨r以及跨固定数据使用独立流。设两策略输出为 \(\widehat\rho_{d,A,r}\)，\(A\in\{\mathrm{IN4},\mathrm{OUT4}\}\)。主风险对象是

$$
\Delta_{\mathcal R\mid D}
=\frac14\sum_{d=0}^3
E_{\rm prior}\!\left[
\mathcal R(\widehat\rho_{d,\mathrm{OUT4}})
-\mathcal R(\widehat\rho_{d,\mathrm{IN4}})\mid D
\right].
\tag{A.4}
$$

每列先平均四份数据的配对风险差，产生16个iid条件单位；风险方向开放。四份数据不是四个新的随机数据样本，四个成员也不能作为四个独立OUT4输出。

对每策略令条件协方差为 \(\Sigma_{d,A}\)。两个独立quartet输出满足

$$
E\!\left[
\frac12\|\widehat\rho_{d,A,r}-\widehat\rho_{d,A,r'}\|^2\mid D
\right]=\operatorname{tr}\Sigma_{d,A},\qquad r\ne r'.
\tag{A.5}
$$

预定8个不重叠列对(0,1)、…、(14,15)各自形成OUT4−IN4差，再平均四份数据，得到8个iid条件单位。它们估计条件trace covariance差，不是参数平方误差差。不得把所有重叠列对当作独立重复。

第三项训练主量是条件参数均值的平方误差差。对同一个不重叠列对，记 \(x=\widehat\rho_{d,A,2j}\)、\(y=\widehat\rho_{d,A,2j+1}\)，定义交叉积 \(W_{d,A,j}\)。由两列条件独立且同分布，

$$
\begin{aligned}
W_{d,A,j}&=(x-\rho_*)^\top(y-\rho_*),\\
E[W_{d,A,j}\mid D]&=\|E[\widehat\rho_{d,A}\mid D]-\rho_*\|^2,\\
\frac12\bigl(\|x-\rho_*\|^2+\|y-\rho_*\|^2\bigr)
&=\frac12\|x-y\|^2+W_{d,A,j}.
\end{aligned}
\tag{A.6}
$$

每对先作OUT4−IN4的交叉积差，再平均四数据，得到8个iid条件单位。交叉积可以为负，必须保留；这不意味着真实平方均值误差为负。同一组8对产生的离散估计与交叉积估计逐实现相加，恰等于全部16输出的样本参数MSE，两个主统计量彼此相关。Bonferroni不要求家族内独立，但不能把它们叙述为两个独立因果通道，更不能将其和分解成KL份额。

训练家族alpha=.025、q=3，双侧开发t：风险16列/df15、离散8对/df7、条件均值误差8对/df7；SE阈值分别.0005、.05、.05。方向按预冻结规则开放分类。独立评价仍另alpha=.025、q=2。第三量在任何新科学RNG前规定，不以旧描述或本轮结果倒推确认标准。解析恒等式不替代对应统计资格。

## 4. 描述性参数分解与成员核查

对每数据、每策略的R=16个输出，定义

$$
\begin{aligned}
\bar\rho_{d,A}&=\frac1R\sum_r\widehat\rho_{d,A,r},\\
S_{d,A}&=\frac1{R-1}\sum_r\|\widehat\rho_{d,A,r}-\bar\rho_{d,A}\|^2,\\
\widehat B_{d,A}&=\|\bar\rho_{d,A}-\rho_*\|^2-S_{d,A}/R.
\end{aligned}
\tag{A.7}
$$

这里 \(ES=\operatorname{tr}\Sigma_{d,A}\)，\(E\widehat B=\|E\widehat\rho_{d,A}-\rho_*\|^2\)，且逐样本参数MSE恰为 \(S+\widehat B\)。负的去偏估计原样保留。它与第三主量的平均交叉积针对同一条件均值误差，但使用不同的离散估计，有限实现不必相等；本节全R分解仅描述，不能替换8对主门。这是参数平方距离恒等式，不是KL的因果分解，也不分摊目标偏差、数据误差和有限优化误差。

成员参数还给出一个逐实现核查：

$$
\frac1{16}\sum_{r=0}^{15}\widehat\rho_{d,\mathrm{OUT4},r}
=\frac1{64}\sum_{r=0}^{15}\sum_{j=0}^3\widehat\rho_{d,j,r}.
\tag{A.8}
$$

式(A.8)仅涉及确定性算术和浮点求和容差。若额外报告64成员样本协方差，须明确它与16个OUT4样本协方差在有限实现中并不严格相差四倍；1/4是条件总体身份。成员与OUT4的去偏均值误差估计也因各自方差估计不同而未必逐实现相等，不以该差异虚构均值保持身份失效。

## 5. 独立目标评价与参考范围

新评价prior池独立于全部训练流，对冻结输出同时评价原raw \(F_D\) 与数值参考权重 \(F_P\)。每个数据使用自己的原active经验频数，先平均该数据的16次输出，再平均四数据。共享评价池提供配对候选差；其MC组是评价区间的单位，128个相关候选不是128个独立评价重复。

这类区间条件于本次实现的训练面板，不覆盖训练prior总体的评分变化。两种不确定性与各自家族必须分开；不能把新MC确认的面板排序宣称为训练程序总体差已获同等区间。公平nuisance的canonical标签恒等式只成立于新prior期望，不重放原完整标签的训练耦合。

参考候选固定为两份旧truth/high PMF，128个主TAIL参数模型和8个“每数据/策略16输出均值”模型，共138项。136个新矩阵按24/32/40阶全部积分、主32；不根据分数挑选、不去重少跑、不追加阶数。成员PMF与LAST均不积分。8个均值探针涉及16个quartet输出，不等于主OUT4的四成员平均，也不是另一个等成本方法。积分阶数一致性只给数值资格，不给严格余项或概率覆盖证书。

## 6. 计算合同与研究边界

每个quartet每步生成四份独立整批MC输入；IN4作四次评分，四个成员分别再作一次评分，所以配对研究共八次原API评分。两种候选策略各用四份K100采样/评分预算，但OUT4需要四次Adam/修复，IN4一次；整个quartet有五条优化路径。D4×R16因此是64个paired quartets、320个优化程序，不是128个程序。另记优化器开销、存储、实际墙钟和并行资源，不能把模拟预算匹配写成时间或全部计算成本严格相等。

共同输入用于配对，不是八份独立抽样；各路径参数分开后仍须在各自状态评分。所有非有限、数值资格、修复与反向结果按原规则保留，不通过增加步数、成员、quartet或评价bank追门。

本轮比较有限步程序的计算分配，不证明目标被修正，不证明IN4/OUT4收敛、普遍优于对方或为通用计算最优方法。旧v7、先前主门和先前描述保持；当前旧数据条件结论不升级为自由尺度、神经网络或完整SCBM学习结论。
