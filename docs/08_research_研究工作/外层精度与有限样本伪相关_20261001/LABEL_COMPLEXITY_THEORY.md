# 标签数的收紧：raw 端点选择不要求经验 pair 接近零

本页是追加解析结果，基准 `33f2d1c5378e269563a96cbfdd6eb0b2c32d9a7b`。不修改已冻结的 SHARED_ERM、GRADIENT 理论或任何实验，0新随机样本、积分、拟合和数值搜索。旧 SHARED_ERM 中 N=B 为多项式的充分路径仍成立；本页证明同一真实共享银行算法的标签数 N 可以任意慢地趋于无穷。

## 1. 真实训练合同与条件期望

固定发射尺度 $a>0$、$0<R<1$，有限相关网格 $\mathcal G\subset[-R,R]$ 包含 $0,\pm R$，大小为 M。只有前两位 active 相关 rho 可变；全部基础Gaussian坐标的边际尺度为1（logit尺度为a），均值为0，其余 n 位在候选和教师中均为独立公平观察因子。真实教师为 rho=0，维数 D=n+2。N个完整自旋标签 $Y_j\in\{-1,1\}^{D}$ 与 B 个完整 Gaussian prior bank 独立；每 bank 含 K 个粒子，全部标签和候选共用这些 bank。

真实 raw 损失仍为

$$
L_\rho(Y,Z_b)=-\log\left\{K^{-1}\sum_{k=1}^K
\prod_{i=1}^{D}\sigma(Y_i\eta_{bki}(\rho))\right\},\qquad
\widehat F^{raw}(\rho)=\frac1{NB}\sum_{j,b}L_\rho(Y_j,Z_b),
\tag{LC1}
$$

其中 $\eta=a(Z_1,\rho Z_1+\sqrt{1-\rho^2}Z_2,Z_3,\ldots)$。RB在每个粒子的似然内部精确边际化 n 个 nuisance 位，使用同一 bank 的 active 部分和同一批实际标签，每个候选的积分观察律 Q 不变。两种程序均作同一有限网格的 computed 目标 ERM；下述解析结论采用精确算术/精确网格最小值。

先固定任意完整数据集 $\mathcal Y=(Y_1,\ldots,Y_N)$，不要求它来自教师。令

$$
t_N=N^{-1}\sum_jY_{j1}Y_{j2}\in[-1,1].
\tag{LC2}
$$

对**单条标签的 bank 期望**，独立 Gaussian 的符号对称性消去所有 nuisance 标签符号；active 标签同时翻转也不改变其期望。因此 raw 的条件平均对比只依赖 active spin product：存在函数 $A_{n,K},C_{n,K}$，使

$$
E_Z[\widehat F^{raw}(\rho)-\widehat F^{raw}(0)\mid\mathcal Y]
=\Delta^{raw}_{n,K}(t_N,\rho)
=A_{n,K}(\rho)+t_NC_{n,K}(\rho).
\tag{LC3}
$$

具体地，若 $\ell_\pm(\rho)$ 为 active product 等于±1时的单标签期望损失对比，则 $A=(\ell_++\ell_-)/2$、$C=(\ell_+-\ell_-)/2$。

LC3仅是对 bank 积分后的身份。计算真实训练目标时，仍须保留每条完整标签在 nuisance 项中的符号。不能对不同标签分别变换 bank 后，又把变换后的同一个 canonical 噪声和强行共享；那会改变经验目标的联合分布。本页集中论证始终应用于原始完整共享 bank。

## 2. 两个合法 anchor 推出全部经验 t 的一致势

记

$$
r_a(u)=E_u[\tanh(aX_1/2)\tanh(aX_2/2)],\quad
s_n=\sqrt{n\operatorname{Var}(\log\sigma(aZ))},\quad
J_K=E\max_{k\le K}G_k,
\tag{LC4}
$$

其中 $G_k$ 为 iid 标准正态。引用已冻结 POLYNOMIAL_K_THEORY 的一致值极限。选两个固定合法教师 anchor：latent 相关0，以及 latent 相关 R/2。后者的**观察 pair 矩**为

$$
t_1=r_a(R/2)>0,
\tag{LC5}
$$

并不是 R/2 本身。两个 anchor 均属于同一个严格正定连续相关区间，不要求 R/2 是训练网格中的候选。

对整数 $2\le K\le C_0n^b$，旧定理分别给 t=0、t=t1 时

$$
\frac{s_n}{J_K}\Delta^{raw}_{n,K}(t,\rho)
\longrightarrow V_t(\rho):=-\frac{a^2}{4}
\left[t\rho+\int_0^\rho r_a(u)du\right]
\tag{LC6}
$$

在整个 rho 区间和该 K 范围上一致。由 LC3 的精确仿射身份，

$$
C_{n,K}=\frac{\Delta^{raw}_{n,K}(t_1,\cdot)-\Delta^{raw}_{n,K}(0,\cdot)}{t_1}.
$$

所以若两个 anchor 的最大归一化误差分别为 e0、e1，则全部 $t\in[-1,1]$ 的误差至多 $e_0+(e_0+e_1)/t_1$。t1固定且严格正，得到

$$
\sup_{|t|\le1,\ |\rho|\le R,\ 2\le K\le C_0n^b}
\left|\frac{s_n}{J_K}\Delta^{raw}_{n,K}(t,\rho)-V_t(\rho)\right|\to0.
\tag{LC7}
$$

这不是把旧定理未经证明地扩展到随 n 变化的教师；它是由两个固定教师的结论和一个精确仿射身份得到的有限维推论。

Price 公式给 $r_a'(\rho)>0$，连续性和紧区间因此给正常数

$$
\lambda=\frac{a^2}{4}\min_{|u|\le R}r_a'(u)>0,
\qquad V_t''(\rho)\le-\lambda.
\tag{LC8}
$$

强凹插值不等式对任何 t 同时成立：

$$
V_t(\rho)\ge\frac{R-\rho}{2R}V_t(-R)
+\frac{R+\rho}{2R}V_t(R)
+\frac\lambda2(R^2-\rho^2)
\ge\min\{V_t(-R),V_t(R)\}+\frac\lambda2(R^2-\rho^2).
\tag{LC9}
$$

因此固定网格的所有非端点都有一个与 t 无关的严格 gap：

$$
g_0=\frac\lambda2\min_{\rho\in\mathcal G\setminus\{-R,R\}}(R^2-\rho^2)>0.
\tag{LC10}
$$

LC7遂使任意完整数据集的 raw 条件 bank 期望，在充分大 n 时，非端点相对最佳端点的 gap 至少为 $(g_0/2)J_K/s_n$。无需先让 t_N 接近0；经验 pair 的符号只可能改变两个端点的相对偏好。

## 3. 只对 B 集中的条件界

令 $f^q_\rho=L^q_\rho-L^q_0$，q为raw或RB。逐标签、逐bank的固定网格包络为

$$
|f^q_\rho(Y,Z_b)|\le H_b
=s\max_{k\le K}\|(Z_{bk1},Z_{bk2})\|,\qquad
s=a\sqrt{2(1-\sqrt{1-R^2})}.
\tag{LC11}
$$

这也控制数据平均 $N^{-1}\sum_jf^q_\rho(Y_j,Z_b)$，与 N、完整标签及 nuisance 维数均无关。条件于任意确定的完整数据集，不同 bank 的这些平均仍 iid；同一 bank 内的 N 项不被当成独立。

取 $0<\delta<1$，设

$$
h=s\sqrt{2\log(2BK/\delta)},\quad L=\log(8M/\delta),
$$
$$
e_{B,K,\delta}=h\sqrt{2L/B}
+K\left(h+\frac{s^2}{h}\right)e^{-h^2/(2s^2)}.
\tag{LC12}
$$

以条件概率至少1−delta，raw/RB两个网格同时满足

$$
\max_{q,\rho\in\mathcal G}
\left|\widehat\Delta^q(\rho)
-E_Z[\widehat\Delta^q(\rho)\mid\mathcal Y]\right|
\le e_{B,K,\delta}.
\tag{LC13}
$$

**证明。** 分析中截断每个 bank 的数据平均，乘以 $1_{H_b\le h}$，算法本身不截断。Gaussian 半径尾与 union bound 使任一实际 bank 越界的概率不超过 $BK e^{-h^2/(2s^2)}=\delta/2$。截断变量范围[-h,h]，对 B 个 iid bank 使用双侧 Hoeffding，再对两个方法的 M 个网格函数 union，失败概率为 $4M e^{-L}=\delta/2$。截断的总体偏差不超过 $E[H1_{H>h}]\le K(h+s^2/h)e^{-h^2/(2s^2)}$。合并即得 LC13。□

这里的统一量词是：对每个任意确定的 N 和完整数据集 Y，其条件失败概率均不超过同一个 delta，即 $\sup_{N,\mathcal Y}P_Z(\text{失败}\mid\mathcal Y)\le\delta$。它**不是**某一个 bank 实现以概率1−delta同时对所有可能的数据集成功。将此条件界对任意独立数据分布积分是合法的，不需要对全部数据集另作 union。

## 4. raw 的端点结论不要求标签增多

固定 eta>0，取

$$
K_n=n+2,\qquad B_n=\lceil n^{1+\eta}\rceil,
\qquad\delta_n=n^{-2}\quad(n\ge2).
\tag{LC14}
$$

LC12给 $e_n=O(\log n/n^{(1+\eta)/2})=o(J_{K_n}/s_n)$。将 LC13 的经验误差与 LC10 的条件期望网格 gap 比较，存在不依赖 N 或 Y 的 n0，使全部 $n\ge n_0$ 有

$$
\inf_{N\ge1,\mathcal Y}
P_Z\{\widehat\rho_n^{raw}\in\{-R,R\}\mid\mathcal Y\}
\ge1-\delta_n.
\tag{LC15}
$$

确切地，LC7使条件期望 gap 最终至少 $g_0J_K/(2s_n)$；LC13使任意两个候选的比较误差不超过2e_n，后者最终严格更小。所以非端点不能成为经验网格最小值，精确算术下的平局也只能发生于端点之间。

因此即使 N 固定，raw 仍有上述端点结论。它不要求样本 pair 矩 t_N 一致估计0。这是原始共享银行经验目标的结论，不是用总体目标或梯度估计器代替训练。

对真实独立教师，记完整积分模型风险

$$
\mathcal R_a(\rho)=D(Q_0\Vert Q_\rho)
=-\tfrac12\log(1-r_a(\rho)^2),\quad
R_{\max}=\mathcal R_a(R)>0.
\tag{LC16}
$$

它与 n 无关，在整个网格上位于[0,Rmax]，两个端点风险相同。故 LC15 还给对任意数据集的条件期望风险夹界
$(1-\delta_n)R_{\max}\le E_Z\mathcal R_a(\widehat\rho_n^{raw})\le R_{\max}$。这里 n0 未被数值确定，不用于认证当前有限 n 的精确端点概率。

## 5. RB 恢复只需 N 任意慢地趋于无穷

现在要求真实训练标签 iid 来自独立教师，且与全部 bank 独立。令 N=N_n 为预先指定的任意正整数序列，不是依 bank 自适应决定的停时。对每个 active 标签与每个固定网格候选，K粒子 active 似然均值的负 log 由单粒子负 log 的平均控制；后者具有有界二阶矩。SLLN与一致可积性给其 bank 期望趋于精确 active 负 log-likelihood。标签四种、网格有限，因此该收敛对所有经验四类权重一致，尤其不要求 N 与 K 同阶。

精确条件交叉熵对比有闭式

$$
\Delta^{exact}(t_N,\rho)
=\mathcal R_a(\rho)-t_N\operatorname{atanh}(r_a(\rho)).
\tag{LC17}
$$

这是将 same/different 权重 $(1\pm t_N)/2$ 代入 $Q_\rho(y)=(1+y_1y_2r_a(\rho))/4$ 得到的恒等式，n个公平因子的 log-loss 在对比中完全抵消。

定义固定正常数

$$
g_1=\min_{\rho\in\mathcal G\setminus\{0\}}\mathcal R_a(\rho)>0,
\quad A_1=\max_{\rho\in\mathcal G}|\operatorname{atanh}(r_a(\rho))|>0,
\quad \kappa=\min\{1/2,g_1/(4A_1)\}>0.
\tag{LC18}
$$

在 $|t_N|\le\kappa$ 时，所有非零候选的精确对比至少为3g1/4。上述有限网格 K→∞ 收敛和 LC13 的 e_n→0，使得在充分大 n 时，RB 经验目标在0处有严格唯一最小值。阈值 n0可以依赖固定模型和网格，但不依赖 N。

独立教师下，$Y_{j1}Y_{j2}$ 为 iid 公平自旋，Hoeffding给
$P(|t_N|>\kappa)\le2e^{-N\kappa^2/2}$。因此在同一 LC13 bank 事件上，充分大 n 时

$$
P\{\widehat\rho_n^{raw}\in\{-R,R\},\ \widehat\rho_n^{RB}=0\}
\ge1-\delta_n-2e^{-N_n\kappa^2/2}.
\tag{LC19}
$$

不要求数据事件与 bank 事件独立：条件 bank 失败概率已经对所有数据集统一受控。

只要 $N_n\to\infty$，无论多慢，LC19趋于1；风险有界也给

$$
E\mathcal R_a(\widehat\rho_n^{raw})\to R_{\max}>0,
\qquad
E\mathcal R_a(\widehat\rho_n^{RB})
\le R_{\max}\min\{1,\delta_n+2e^{-N_n\kappa^2/2}\}\to0.
\tag{LC20}
$$

raw与RB使用完全相同的N条完整标签、B个完整bank、K=n+2和候选网格。这个结论收紧的是标签量词，不是给某个已运行预算补一张更严格的数值证书。有限N的概率界仍可能空；一般三角数组只得到这里陈述的概率和期望结论，不自动得到程序最终几乎处处成功。

## 6. 范围与来源

本页给出B=n^{1+eta}的充分路径，不证明它是 sharp 必要预算，也不证明固定N能够让一般标签驱动的RB网格ERM一致。N计完整标签向量；向量维数D仍增长，不能把较慢的向量样本数误报为总bit成本不增长。已知active位置、固定有限网格、固定a/R、零均值、同模型nuisance边际化均保留。未知active结构、连续ERM、共享标签下的其它梯度估计器与固定K实验不由此追认。

原始随机预测器评分与积分模型风险仍是不同合同。本页证明实际网格学习后的积分风险分离，不称raw在其自身随机预测合同上次优。科学实现的固定浮点误差若不随n控制，不能自动继承精确算术的渐近网格结论。

主要引用的冻结来源为 [POLYNOMIAL_K_THEORY.md](../无关维度与固定噪声总体偏好_20261001/POLYNOMIAL_K_THEORY.md)，SHA256 `e88b5e4de6c4ac4210708bb366d9f299a124b6ec118e199461a56fda04d08d9d`；本包先前路径见 [SHARED_ERM_THEORY.md](SHARED_ERM_THEORY.md)，SHA256 `8e377c20fd68f42261f56db7278fb72eca1980f4022a01a67011decc2fd6fd99`。两源均未修改。新增推论使用经典条件集中与有限网格gap，不以这些通用工具主张首创。
