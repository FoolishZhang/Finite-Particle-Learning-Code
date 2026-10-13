# 从正确概率命题到固定自然数据目标

这里解释SPECTRUM计算的对象，不能将自然数据的预测概率p自动称为真实条件概率。

对固定的有限样本(x_n,c_n)及冻结均值mu_n，令p_n=sigmoid(mu_n)，r_n=c_n-p_n，并定义

\[
D_p=\operatorname{diag}\{N^{-1}\sum_n p_{nj}(1-p_{nj})\},\quad
M=N^{-1}\sum_n r_nr_n^T.
\]

M是不减去残差均值的二阶矩，不是通常中心化的协方差。它同时包含概率失准和条件依赖的影响。对每个n用K个iid Gaussian噪声，再对噪声和有限样本取期望，固定mu的零协方差右导数为

\[
 F_K'(0+)=\tfrac12\operatorname{tr}\{S[D_p-(1-1/K)M]\}.
\]

证明可直接用log Bernoulli的Hessian为-diag[p_n(1-p_n)]，所以似然的归一化Hessian为r_nr_n^T-diag[p_n(1-p_n)]。logmean的额外项为r_n^TSr_n/(2K)，平均即得上式。无需假定p_n正确，也无需观察同一个x的多次标签。

因此本项目SPECTRUM的D_p^{-1/2} M D_p^{-1/2}是**归一化残差二阶矩**。其最大特征值给出这个经验训练目标的局部条件，不是被独立认证的真实概念相关矩阵。它不附带抽样置信区间。对Z/-Z的配对目标，右导数为.5 tr S(D_p-M)。这些等式为实数logistic/logmean计算的结论，源码float32饱和或概率截断须另行记录。

## 为什么不能把自然矩阵的全部效应称为依赖

即使M的非对角全为0，只要某个M_jj>D_p,jj，固定均值的精确目标也会沿该维方差下降。这是边际重校准的用途。反过来，M_jj<D_p,jj会增加方差代价。自然试验用NPp−NP与EQp−EQ诊断联合相对边际乘积的额外预测用途，但风险恒等式仍不是因果归因。

可以另作一个已有方法的机制控制：保持每个预测p_n不变，调整Gaussian均值。其局部补偿是

\[
 \mu_{nj}(t)=\mu_{nj}-\tfrac12tS_{jj}(1-2p_{nj})+O(t^2).
\]

该补偿消去归一化Hessian的对角项，因此在上述经验数据上的iid导数变为

\[
 \widetilde F_K'(0+)=\tfrac12\operatorname{tr}\{S[D_M-(1-1/K)M]\},\quad D_M=\operatorname{diag}(M).
\]

配对版本为.5 tr S(D_M-M)，对角项恰为0。与THEORY中“正确p时补偿不改变阈值”一致，因为正确概率下M的对角等于D_p；不正确p的自然数据则不能直接套用同一阈值。

本节后一个补偿版本目前只是解析扩展，未包含在N实际拟合中。任何数值补偿或copula比较都须另立方案，不能把已有RAW训练当成已经做了该控制。
