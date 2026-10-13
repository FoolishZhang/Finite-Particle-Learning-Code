# 同一个有限K目标下的外层平均：可识别量与推断边界

本页为新包scbm_bank_averaging的解析合同。旧四份完整数据固定为data_id 0..3；每份16列新prior重复，比较L1与L4。K恒为100，无关尺度恒为1，六个物理相关从I开始。v7与全部旧科学不改。本页不调用新RNG、拟合或模型积分。

## 1. 改的是MC平均层，不是粒子目标

对固定参数 \(C\) 和固定批次 \(\mathcal B\)，令
\(\ell_K(C;\mathcal B,Z)\) 为一份batch-wide MC输入的负log粒子均值损失，批内每个观察有自己独立的K条prior。定义

$$
\bar\ell_{L,K}=\frac1L\sum_{\ell=1}^L\ell_K(C;\mathcal B,Z_\ell),
\qquad
g_{L,K}=\frac1L\sum_{\ell=1}^Lg_K(C;\mathcal B,Z_\ell),
\qquad Z_\ell\ {\rm iid}.
\tag{A.1}
$$

所有 \(L\) 项在同一个参数处评价，然后只作一次Adam更新与一次可行性修复。两臂共享bank0以配对，L4的其余三个bank独立；这不改变各臂边缘分布。

在实算术下，记固定点均值和协方差为
\(\mu_{\mathcal B}(C)=Eg_K(C;\mathcal B,Z)\)、
\(\Sigma_{\mathcal B}(C)=\operatorname{Cov}g_K(C;\mathcal B,Z)\)，则

$$
E\bar\ell_{L,K}=F_{\mathcal B,K}(C),\qquad
Eg_{L,K}=\mu_{\mathcal B}(C)=\nabla F_{\mathcal B,K}(C),\qquad
\operatorname{Cov}g_{L,K}=\frac{\Sigma_{\mathcal B}(C)}L .
\tag{A.2}
$$

这里的可积性成立：在单位对角、谱下界.1的紧类上，Cholesky映射的一阶导数一致有界。log-sum-exp的路径梯度是单粒子梯度的正权平均，log-sigmoid导数有界，所以梯度范数被常数乘 \(\max_{k\le K}\|Z_{k,\mathrm{active}}\|\) 控制，具有有限二阶矩。该包固定K与无关维数；无须宣称增长维数或SGD收敛率。

如果每观察的第 \(\ell\) 个bank均值似然为 \(a_\ell>0\)，外平均是
\(L^{-1}\sum_\ell[-\log a_\ell]\)，而合并LK粒子是
\(-\log[L^{-1}\sum_\ell a_\ell]\)。后者由Jensen不大于前者，且通常不同。因此pooling LK改变了目标；循环中做L次更新也不是式(A.1)。

本轮批次按固定顺序循环。式(A.2)逐步对应当前批次目标，不是每一步都无偏估计完整 \(F_{\mathcal D,K}\) 的梯度。只有冻结同一个C后，均匀平均全部64个固定批次，才得到完整数据目标。

## 2. 最强反驳：同一期望梯度不等于同一优化动力学

外平均除了缩小协方差，也改变更高阶分布特征。尤其

$$
E[g_{L,K}\odot g_{L,K}]
=\mu_{\mathcal B}(C)\odot\mu_{\mathcal B}(C)
+\frac{\operatorname{diag}\Sigma_{\mathcal B}(C)}L .
\tag{A.3}
$$

所以即使C、上一时刻的Adam状态和批次相同，Adam二阶矩状态的条件期望也随L变化。除法预条件、保留动量的可行性修复与后续轨迹均是非线性的，平均更新不必相同。研究可称为“固定程序下的MC外平均干预”，不能称为只改方差而不改任何其他动力学的干预，更不能给风险改善分配一个纯噪声百分比。

在随机轨迹 \(C_t^{(L)}\) 上，条件于固定数据和该时刻批次，

$$
\operatorname{Cov}(g_{L,K}(C_t^{(L)}))
=E\!\left[\frac{\Sigma_{\mathcal B_t}(C_t^{(L)})}{L}\right]
+\operatorname{Cov}(\mu_{\mathcal B_t}(C_t^{(L)})).
\tag{A.4}
$$

两臂的状态分布不同，式(A.4)不是跨轨迹的1/L比例。不能将两条路径在不同参数处的梯度差称为平均场方向被校正；相同参数/批次下的平均场仍是同一个 \(F_{\mathcal B,K}\)。

L4消耗四份独立prior输入，L1复用第一份；每paired step共五次原API评分而不是五份独立随机输入。额外计算是干预的一部分，不能包装为等成本优势。修复规则本轮固定，修复计数只作描述。

## 3. 条件学习分布与风险estimand

令 \(\widehat\rho_{d,L,r}\) 为固定数据 \(D_d\) 上第r列新prior、arm L的TAIL六向量。TAIL是最后64个post-update物理参数的平均，不是64次独立观察，也不是LAST。所有非prior输入与规则固定后，它是训练prior的确定函数。

各列r独立同分布；列内两臂配对，跨数据的prior也独立。四份数据作为固定层，不当作四个新抽样数据集。定义

$$
\Delta_{\mathcal R\mid D}
=\frac14\sum_{d=0}^3
E_{\rm prior}\!\left[
\mathcal R(\widehat\rho_{d,4})-\mathcal R(\widehat\rho_{d,1})\mid D
\right].
\tag{A.5}
$$

其无偏估计由每列先平均四份数据的配对风险差构成，共16个iid条件单位。预定开发t区间使用df15。参考风险通过数值PMF计算，积分资格另列，不把资格当作覆盖概率证书。

风险方向开放：更稳定地沿同一个错误准则前进，可能改善或加剧积分风险。一个方向不预设为实现成功。训练家族alpha=.025、q=2；风险SE目标.0005，不能与下一节离散度的单位或门槛混用。

## 4. 条件离散：独立列对与去偏参数分解

对每个固定 \(D_d,L\)，定义
\(\mu_{d,L}=E_{\rm prior}\widehat\rho_{d,L}\)，
\(\Sigma_{d,L}=\operatorname{Cov}_{\rm prior}\widehat\rho_{d,L}\)。
两个独立列的半平方距离满足

$$
E\!\left[\frac12\|\widehat\rho_{d,L,r}
-\widehat\rho_{d,L,r'}\|^2\mid D\right]
=\operatorname{tr}\Sigma_{d,L},\qquad r\ne r'.
\tag{A.6}
$$

因此预定使用(0,1)、…、(14,15)八个不重叠列对，先作L4−L1差，再平均四份数据，得到8个iid条件单位。其均值无偏估计
\(\Delta_V=4^{-1}\sum_d[\operatorname{tr}\Sigma_{d,4}-\operatorname{tr}\Sigma_{d,1}]\)。
开发t区间用df7，SE目标.05；负方向为预期，正向或未分辨同样保留。不能把全部120个相互重叠列对当作120个独立重复。

另一项预定描述使用全部R=16列。令

$$
\begin{aligned}
\bar\rho_{d,L}&=\frac1R\sum_r\widehat\rho_{d,L,r},\\
S_{d,L}&=\frac1{R-1}\sum_r
\|\widehat\rho_{d,L,r}-\bar\rho_{d,L}\|^2,\\
\widehat B_{d,L}
&=\|\bar\rho_{d,L}-\rho_*\|^2-\frac{S_{d,L}}R .
\end{aligned}
\tag{A.7}
$$

则 \(ES_{d,L}=\operatorname{tr}\Sigma_{d,L}\)，
\(E\widehat B_{d,L}=\|\mu_{d,L}-\rho_*\|^2\)，且逐样本有

$$
\frac1R\sum_r\|\widehat\rho_{d,L,r}-\rho_*\|^2
=S_{d,L}+\widehat B_{d,L}.
\tag{A.8}
$$

式(A.8)使用完整六向量，不除以六。去偏的 \(\widehat B\) 可能为负，必须保留；这不是负的真实平方误差。它度量给定D的算法均值到真实参数的距离，不等于有限粒子目标本身的偏差，也包含固定数据与有限优化程序的作用。

全R样本协方差与八个不重叠距离都是同一条件trace covariance的无偏估计，但有限样本值可不同。主推断固定用后者，不能依据结果换成前者；式(A.7)–(A.8)不新增未经预定的置信区间，更不是KL的因果分解。

每数据/arm的16-rho均值矩阵由凸性保证可行，八个均值PMF仅作条件prior几何探针。它们不是平均风险或混合预测器风险，也不是与单次训练等成本的算法。

## 5. 独立F_D评价与训练推断必须分层

训练后固定128个TAIL及八个均值探针。对active模式y，令 \(\phi_y^q(C)\) 为新prior期望评分；q为raw或同K FACT。公平无关标签仅在该期望内可canonical化。于是

$$
F_{D_d}^q(C)=\sum_y\widehat p_d(y)\phi_y^q(C),
\qquad F_P^q(C)=\sum_y p_*(y)\phi_y^q(C).
\tag{A.9}
$$

raw的K仍为100，所以两臂终点使用相同的有限目标评价。P用旧合格数值权重、不重新归一化；D使用各自原数据频数。FACT评分保留active有限K偏差，不是精确似然。

评价阶段32组×1024个新bank在全部138候选间共用，主区间对应

$$
\Delta_D^{\rm panel}
=\frac1{4R}\sum_{d,r}
\{F_{D_d}^{\rm raw}(\widehat C_{d,4,r})
-F_{D_d}^{\rm raw}(\widehat C_{d,1,r})\},
\tag{A.10}
$$

以及其P版本。df31、alpha=.025的q2区间条件于这批实际训练输出与旧数据，不是对训练prior重复的
\(E_{\rm train}\Delta_D^{\rm panel}\) 的区间。共用评价池会使按训练列拆出的评分估计彼此相关，不能直接把它们当16个独立评分样本。

训练风险家族与评价家族分别有alpha=.025预算；总预算.05仍仅在各自开发近似成立时有相应联合含义，且两类estimand不同。若日后要求训练和评价两层合并的不确定性，需要独立每列评价池，或另定crossed两因素推断；本轮不暗中加入。

目标更低、风险更低、参数离散更低是三个不同结论。即使三者同向，也不把总风险变化分摊为纯噪声或目标偏差份额；如果它们反向，这同样回答了干预是否更稳定地追随错误准则的问题。

## 6. 最小实现控制与本轮验证边界

确定性控制应使用四个不同bank、同一个参数和同一批次，逐项核对外平均loss/六梯度，确保没有pool-K、循环内多次更新或只保留最后一bank。每个bank的K保持100。实际float32程序另作math64资格检查，式(A.2)不被称为严格浮点恒等式。

首步的I和每份数据首批次在16列间相同，可从原训练记录比较64个单bank梯度与16个四bank平均梯度的样本trace covariance。其比值相关且有抽样误差，只作描述，不要求它精确等于1/4，不追加RNG追门。后续状态不同，不再具备这个固定点解释。

最强未排除因素是Adam自适应预条件及修复与噪声分布的交互，而不是目标均值改变。当前设计有意保留这些环节，识别的是整个固定程序对外平均干预的反应。若需要更窄的机制，下一次才考虑冻结预条件或单一可行性对照；不在本轮同时更改学习率、K、步数或多项参数化。

参考计算只为固定的128个TAIL及八个均值新矩阵运行预定GH24/32/40，共136×3=408 PMF、535887872四维节点；即使重复也不去重。LAST不积分。truth/high两PMF旧值复用，high只作oracle诊断而不进入学习器。新候选由训练结果机械生成，第二阶段DERIVED_FREEZE之后才允许新参考或评价计算。所有新结果留在本包，不回改封存v7。
