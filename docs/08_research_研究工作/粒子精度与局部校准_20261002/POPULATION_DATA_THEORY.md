# 校准后为何仍需增加独立 bank：交叉抽样分解与固定 B 边界

本补充在本包 LOCAL 已完成、TRAIN 尚未结束且未读取主统计时提出。它不改事前方案或主检验，不依赖本轮成功与否，不新增随机计算。使用的正交分解与固定Monte Carlo预算的一致性边界是经典原理；贡献限于把本模型的有限校准、完整观测和共享bank学习连在一起，不宣称一般统计方法创新。

## 1. 数据与 bank 两个预算有不同作用

固定有限 n≥0、K≥2、a=1、R<1 和内点 teacher θ∈(−R,R)。观测 Y_j iid 来自正确指定 Q_θ；bank Z_b iid，且与全部观测独立。所有观测共享 B 个bank。固定参数 q，记完整负log粒子似然的路径导数 h_q(Y,Z)。这一定义保留逐行 nuisance，不用 canonical 行替代。记 m=E h_q，

\[
u_q(Y)=E_Zh_q(Y,Z)-m,\quad
v_q(Z)=E_Yh_q(Y,Z)-m,\quad
e_q(Y,Z)=h_q(Y,Z)-m-u_q(Y)-v_q(Z).
\]

三者均值为0，且 E[e_q|Y]=E[e_q|Z]=0。对平均梯度 \(\bar h_{N,B}=(NB)^{-1}\sum_{j,b}h_q(Y_j,Z_b)\)，有精确恒等式

\[
\operatorname{Var}(\bar h_{N,B})
=\frac{\sigma_Y^2(q)}N+
\frac{\sigma_Z^2(q)}B+
\frac{\sigma_e^2(q)}{NB},
\tag{P.1}
\]

其中三方差分别为 Var u、Var v、Var e。证明把平均拆成 u 的 N 项均值、v 的 B 项均值及 e 的 NB 项均值：条件均值为零使全部交叉协方差为零；e中即使两项共享行或bank，协方差仍为零。本模型 \(|h_q|\le M_K/\sqrt{1-R^2}\)，\(M_K=\max_k\sqrt{Z_{k1}^2+Z_{k2}^2}\)，所以二阶矩有限。

给目标加任意确定性可微罚项不改变上述方差。若先估计再固定校准池，则条件于它同样成立；若重新对独立校准池随机性平均，还须添加该点校准导数的方差。不能把 NB 个评分项当作 NB 个独立样本。

## 2. 理想有限校准显式留下哪些项

有限条件期望身份为

\[
E_Zh_q(Y,Z)=U'(q)+Y_1Y_2 V'(q),\qquad
b(q)=-V'(q)>0,\qquad
g'(q)=-U'(q)-r(q)V'(q).
\tag{P.2}
\]

因此 \(E[\bar h_{N,B}+g'(q)]=b(q)[r(q)-r(\theta)]\)。在固定真值 q=θ 处，平均校准梯度为0，并且数据主效应可以精确写为

\[
\sigma_Y^2(\theta)=b(\theta)^2[1-r(\theta)^2],
\qquad
E[\{\bar h_{N,B}+g'(\theta)\}^2]
=\frac{b(\theta)^2[1-r(\theta)^2]}N+
\frac{\sigma_Z^2(\theta)}B+
\frac{\sigma_e^2(\theta)}{NB}.
\tag{P.3}
\]

该式是对新iid数据和bank同时平均的理论身份。本次实验复用旧数据且在数值数据MLE处取梯度，其实测对象是 THEORY.md 的条件方差身份；不能直接把实测读数称为 (P.3) 三项的估计。尤其没有本轮数据支持给三个方差分量赋数值或百分比。

## 3. 固定 B 的噪声项确实严格为正

对上述任意固定 θ，有 \(\sigma_Z^2(\theta)>0\)。以下证明不依赖粒子退化事件具有实用概率，也不声称实用下界。

考虑一个bank的所有粒子都趋近同一有限向量 z=(z_1,z_2,0,…,0)。此极限下粒子混合等于单粒子，且观测第二位边际为1/2，所以

\[
E_Yh_\theta(Y,Z)
\longrightarrow
\left(z_1-\frac{\theta}{\sqrt{1-\theta^2}}z_2\right)
\left[\sigma\!\left(\theta z_1+\sqrt{1-\theta^2}z_2\right)-\frac12\right].
\tag{P.4}
\]

z=(0,0,…)给0。θ≠0时取(z_1,z_2)=(1,0)，得到非零值；θ=0时取(1,1)，也得到非零值。完整观测状态有限，粒子概率严格正，故 bank→E_Yh 的映射连续。两个极限值的充分小邻域内函数值区间不相交，而有限维标准Gaussian在每个这样的开集上概率严格正。因此 E_Yh 不是几乎处处常数，方差严格正。

由(P.3)，固定 B 时令 N→∞仍留下严格正的梯度二阶矩 \(\sigma_Z^2(\theta)/B\)。增加数据不能取代增加独立bank，完全准确的平均目标校准也不能消除这一项。此结论不量化当前N、B是否足够，不证明本轮有限B间的平均KL必然单调。

## 4. 对无限数据、固定 B 最优点的直接后果

固定 B、n、K，定义真正对观测积分、但保留该次 B 个随机bank的目标

\[
L_{B,\theta}(\rho)
=B^{-1}\sum_b E_{Y\sim Q_\theta}[-\log \widehat Q_{Z_b,\rho}(Y)]+p(\rho),
\tag{P.5}
\]

p为任意独立于这些bank的确定性C¹罚项，包括理想g；也可以条件于独立校准池取其固定C¹输出。这里p固定，不是随N增大的惩罚序列；不提供对所有p统一的正风险下界。该目标在紧区间连续、有极小点；在θ邻域可微。由上节，\(\operatorname{Var}(L'_{B,\theta}(\theta))=\sigma_Z^2(\theta)/B>0\)，故其导数非零的事件概率严格正。在这个事件上，内点θ不是任何全局极小点。因此任意可测全局极小点选择 \(\widehat\rho_B\) 满足

\[
P(\widehat\rho_B\ne\theta)>0,\qquad
E\,\mathrm{KL}(Q_\theta\|Q_{\widehat\rho_B})>0.
\tag{P.6}
\]

最后一步使用r严格递增、Q由active pair矩识别，以及KL非负且仅在ρ=θ为零。区间紧且R<1确保风险有限。结论针对(P.5)的实际最优点，不是SGD、数值网格选点或有限N自动达到全局最优的保证。

这与旧“先对bank积分后总体目标失真”是两个不同边界。理想g能修正后者；固定B仍保留前者。它同样不表示复杂校准优于简单二次规则，亦不否定B→∞时的Monte Carlo一致性。本轮B干预和局部单系数对照检验的是两者在当前预算下的相对重要性。
