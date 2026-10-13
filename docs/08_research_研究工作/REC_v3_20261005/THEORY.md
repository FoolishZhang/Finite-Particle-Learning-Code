# 固定端点：Jensen偏差、粒子数与候选排序

2026-10-05。本页用经典MC目标性质解释REC-v3的设计，不将它们声明为新的通用定理。

对固定模型θ、标签y，A_k(y)>0 iid，qθ(y)=EA_k(y)，定义f_K(θ,y)=E[−log(K⁻¹sum A_k)]。给定标签权重T（原数据经验频率或冻结教师），F_T,K=sum T_y f_K，ell_T=−sum T_y log qθ(y)。nuisance已解析消去，256log2是与模型无关的共同常数。

## 精确身份和极限

Jensen给b_T,K=F_T,K−ell_T≥0。对K+1粒子均值取K粒子的所有leave-one-out均值，凸性与交换性给F_T,K+1≤F_T,K。Gaussian-logistic固定有限参数下，−log A由Gaussian绝对值之和控制，任意p>1的矩有限。由于−log(平均A)≤平均(−log A)，再用凸性得到统一L^p界，因此一致可积；结合大数定律，K增大时F_T,K趋ell_T。

对两个端点SCALE2与BASE，

\[
\Delta F_{T,K}=\Delta\ell_T+\Delta b_{T,K}.
\]

T=P时Δell_P=ΔKL(P||Q)=ΔR。若固定面板平均ΔR<0，则充分大K的平均ΔF_P,K也为负。这个极限不提供具体有限K、不证明训练恢复，也不能由每个b单调推断Δb或ΔF全程单调。REC-v3只在事先固定的100/400/1600比较符号和配对变化。

## 二阶解释

对W=A/q，EW=1、Var(W)=M2/q²−1。对−log在样本均值1附近作二阶展开，领先项为

\[
b_{T,K}\approx\frac{B_T}{K},\qquad
B_T=\frac12\sum_yT_y\left(\frac{E A_y^2}{q_y^2}-1\right).
\]

故ΔF_T,K约为Δell_T+ΔB_T/K。当前Gaussian-logistic点有有限矩，但本轮没有建立对所用K的严格余项常数或一致界；该式只作解释性数值对照。M2无需且不可归一化为标签PMF；分母使用已有合格q，独立检查新一阶积分与它的一致性。

本实验改变计算准则中的K，模型、观察分布、数据频率和风险均固定。因此排序变化可归于这一评价准则干预，而不是新训练、边际修复、模型选择或新数据。它不能量化原训练中K的历史因果份额；未比较保留nuisance的原生float32程序、同计算成本训练或最终泛化。
