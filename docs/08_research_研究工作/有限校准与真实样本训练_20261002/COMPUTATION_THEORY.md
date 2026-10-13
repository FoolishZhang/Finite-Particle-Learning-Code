# 真实完整 bank 训练的等价计算、连续求解与校准误差

日期：2026-10-02。本文证明拟议训练实现的计算身份并给出条件误差接口，零新随机抽样、零数值积分、零训练。拟议条件为 $a=1,n=254,K=256,N\in\{1024,4096\},B=128$；下文恒等式在一般固定 $a>0$、$|\rho|\le R<1$ 下成立。有限预算是否足以识别正确目标、改善风险，必须由新实验和其误差检查判断。

已有输入为上一包 连续共享 bank 理论 (historical reference not distributed) 及 正则理论 (historical reference not distributed)。本文保留真实完整观测行，不以总体四模式／canonical nuisance 计算替代经验训练。

## 1. 一个只消除重复计算的精确身份

令 $Y_j\in\{-1,1\}^{n+2}$ 为第 $j$ 条固定完整观测，$Z_{bk}\in\mathbb R^{n+2}$ 为第 $b$ 个训练 bank 的第 $k$ 条固定标准 Gaussian 向量。候选采用

$$
X_{bk1}=Z_{bk1},\quad
X_{bk2}(\rho)=\rho Z_{bk1}+\sqrt{1-\rho^2}Z_{bk2},\quad
X_{bki}=Z_{bki}\quad(i>2).
\tag{T.1}
$$

先计算与候选 $\rho$ 无关的量

$$
\begin{aligned}
\ell_{jbk}
&=\sum_{i=3}^{n+2}\log\sigma(aY_{ji}Z_{bki})
+\log\sigma(aY_{j1}Z_{bk1}),\\
C_{jb}&=\sum_{k=1}^K e^{\ell_{jbk}}>0,
\qquad \omega_{jbk}=e^{\ell_{jbk}}/C_{jb},\\
p_{jb}(\rho)&=\sum_k\omega_{jbk}\sigma(aX_{bk2}(\rho)).
\end{aligned}
\tag{T.2}
$$

所有有限输入下 $0<p_{jb}(\rho)<1$。完整粒子似然的逐观测、逐 bank 身份为

$$
\frac1K\sum_k\prod_{i=1}^{n+2}\sigma(aY_{ji}X_{bki}(\rho))
=\frac{C_{jb}}K
\begin{cases}
p_{jb}(\rho),&Y_{j2}=1,\\
1-p_{jb}(\rho),&Y_{j2}=-1.
\end{cases}
\tag{T.3}
$$

**证明。** 提出 $e^{\ell_{jbk}}$，再除以 $C_{jb}$ 得正位分支。负位用 $\sigma(-x)=1-\sigma(x)$ 与 $\sum_k\omega_{jbk}=1$。□

因此原始训练目标与下式只差 $\rho$ 无关项：

$$
\widehat L(\rho)=\frac1{NB}\sum_{j,b}
\left[-1_{Y_{j2}=1}\log p_{jb}(\rho)
-1_{Y_{j2}=-1}\log(1-p_{jb}(\rho))\right].
\tag{T.4}
$$

差项为 $-(NB)^{-1}\sum_{j,b}\log(C_{jb}/K)$。故任意两个参数的评分差、全部最小点、同容差近优集合完全相同。各方法若增加同一个参数惩罚，该身份仍成立；常数可以逐数据集、逐 $N$ 前缀不同，但对该次参数比较没有影响。

这里的权重是**给定实际完整观测行后**的计算量。它保留全部 $Y_{j,noise}$，并没有把不同观测的 Gaussian 噪声符号统一化，也没有更换 bank 或 posterior proposal。虽然形式上变成第二位的加权 Bernoulli loss，其作用仅是精确重写原完整向量经验目标，不改变论文的观测密度学习合同。

### 1.1 稳定实现与缓存边界

用 $\ell_{jbk}-\max_k\ell_{jbk}$ 归一化可避免直接生成极小的 $C_{jb}$；如果只需评分差，就不必保存常数。数学上的 $1-p$ 身份可以用于解释，但数值实现建议另外计算

$$
p^+_{jb}(\rho)=\sum_k\omega_{jbk}\sigma(aX_{bk2}(\rho)),\qquad
p^-_{jb}(\rho)=\sum_k\omega_{jbk}\sigma(-aX_{bk2}(\rho)),
\tag{T.5}
$$

分别取 $-\log p^+$ 或 $-\log p^-$，以免接近 1 时的减法丢失有效位。真算术中两者之和为 1；浮点残差可以作为诊断，但不能仅凭该残差认证全部目标误差。

无关部分还可以精确写成

$$
\sum_{i>2}\log\sigma(aY_{ji}Z_{bki})
=\frac a2\sum_{i>2}Y_{ji}Z_{bki}
-\sum_{i>2}\log[2\cosh(aZ_{bki}/2)].
\tag{T.6}
$$

第一项是实际观测矩阵与 prior 矩阵的乘积，第二项只依赖 prior。式 (T.6) 可以加速预计算；它依然保留不同观测行的符号。不能把第一项省略、用一个 canonical 行代替，或只留下权重最大的粒子而仍称计算等价。

同一个 $\omega_{jbk}$ 可服务全部 $\rho$。对每个 bank，候选第二位 sigmoid 表与 $\omega$ 矩阵相乘，就得到一批候选的全部 $p^+$/$p^-$。共享第一位和 nuisance 观测的不同教师可进一步复用这些概率，仅第二位符号选择及损失求和不同。所有 $N$ 前缀用同一最大 $N$ 行的前缀和得到，不能对较小 $N$ 重新抽取银行却仍称完全配对。

存下 $4096\times128\times256$ 个 float64 权重约占 1 GiB。按 bank、样本行或候选分块后累积各教师和 $N$ 前缀的损失，可以维持相同真算术目标；浮点累加顺序变化属于数值差异，应复核。此身份减少重复计算，不证明运行时间优于所有基线。

marg 同样可使用 (T.2)—(T.5)，但将 $\ell_{jbk}$ 中的 nuisance 项全部删除；共同的 $n\log2$ 不影响选点。这与 raw 缓存是两个不同的确定权重表，不能因它们的候选第二位表相同而混用。

## 2. 跨教师共同随机数的正确生成与统计单位

对每个观测行独立生成 $G_{j1},G_{j2}\sim N(0,1)$、$U_{j1},U_{j2}\sim U(0,1)$，并生成独立的 nuisance 公平自旋 $Y_{j,i}$，$i>2$。所有这些数据输入独立于训练 prior 和校准 prior。令

$$
\begin{aligned}
Y_{j1}&=2\,1_{U_{j1}\le\sigma(aG_{j1})}-1,\\
X^{\star,\theta}_{j2}&=\theta G_{j1}+\sqrt{1-\theta^2}G_{j2},\\
Y^{\theta}_{j2}&=2\,1_{U_{j2}\le\sigma(aX^{\star,\theta}_{j2})}-1.
\end{aligned}
\tag{T.7}
$$

**命题 T.1。** 对每一个固定教师 $\theta$，按 (T.7) 生成的完整观测行 iid 来自目标 $P_\theta^{(n)}$。不同教师的数据集一般相关；较小 $N$ 是较大 $N$ 的嵌套前缀。

**证明。** 对每个固定 $\theta$，$(G_{j1},X^{\star,\theta}_{j2})$ 是所需单位边际相关 Gaussian。两个独立 Uniform 给定潜变量后生成条件独立的两位 Bernoulli。nuisance 观测独立公平，恰是原 Gaussian-logistic nuisance 的积分律，因此可以直接生成公平位而不必生成未用于观察的教师 nuisance 潜变量。行间独立性来自底层输入逐行独立。跨教师则复用了 $G$、$U$ 及全部共同观测，故不是独立数据集。□

这允许用同一 $\omega$ 同时评价多个教师；不是把真实 $Y_2$ 换成教师期望 $r(\theta)$。训练必须使用实际生成的 $Y^{\theta}_{j2}$。实际观测信号为 $r(\theta)$，而非直接等于潜在相关 $\theta$。

应分别记账：每个独立训练流的底层数据行数、各教师实际构造的观测向量数、共享 nuisance 位和第一位、教师特定第二位。不能把 $T$ 个教师和两个 $N$ 前缀当作 $2T$ 次独立确认；方法差、教师差可作为配对量按真正独立流计算。如果一个校准池被全部训练流共享，跨训练流区间条件于该校准曲线，不包含重新估计校准的变异。

## 3. 固定完整 bank 的连续求解误差

令

$$
M_b=\max_{k\le K}\|(Z_{bk1},Z_{bk2})\|,
\quad \overline M=B^{-1}\sum_bM_b,
\quad \overline{M^2}=B^{-1}\sum_bM_b^2.
\tag{T.8}
$$

对 raw 或 marg 的真算术经验目标，可取统一一、二阶导数包络

$$
L_0=\frac a{\sqrt{1-R^2}}\overline M,
\qquad
D_0=\frac{5a^2}{4(1-R^2)}\overline{M^2}
+\frac a{(1-R^2)^{3/2}}\overline M.
\tag{T.9}
$$

该界对全部实际观测行和 $N$ 前缀同时成立。证明不把 $NB$ 项当作独立：单粒子 log 权重的导数只来自第二位，

$$
|X_2'|\le M_b/\sqrt{1-R^2},\quad
|X_2''|\le M_b/(1-R^2)^{3/2},
\tag{T.10}
$$

而 $|\ell_k'|\le a|X_2'|$、$|\ell_k''|\le a^2|X_2'|^2/4+a|X_2''|$。负 log-sum-exp 的二阶导数是 $-E_\pi\ell_k''-\operatorname{Var}_\pi(\ell_k')$；用 $\operatorname{Var}_\pi(\ell_k')\le\max_k|\ell_k'|^2$ 后平均 $j,b$ 得 (T.9)。权重中完整噪声行始终保留。计算身份 (T.3) 只删了参数无关常数，所以同一个包络控制缓存版本。

记 $A=a^2/4$、$H(\rho)=\int_0^\rho r(u)du$、$w=J_K/\sqrt{nv_a}$。若实际加到经验目标上的 finite 校准为 $\widehat g$，其导数为样条函数 $\widehat d$，则**完整目标**包络为：

| 目标 | 一阶包络 | 二阶包络 |
|---|---|---|
| raw 或 marg | $L_0$ | $D_0$ |
| raw $+\lambda\rho^2$ | $L_0+2|\lambda|R$ | $D_0+2|\lambda|$ |
| raw $+2AwH(\rho)$ | $L_0+2A|w|r(R)$ | $D_0+2A|w|\sup r'\le D_0+2A^2|w|$ |
| raw $+\widehat g(\rho)$ | $L_0+\sup_I|\widehat d|$ | $D_0+\sup_I|\widehat d'|$ |

若实现对 finite 校准另乘系数，该系数的绝对值也必须乘到最后一行两项中。若 asymcorr 实际使用样条近似 $H$，应使用该**实际实现的惩罚**导数包络，或把它与精确惩罚的差纳入数值误差；不能混用理想包络与不同的数值函数。

### 3.1 样条包络可以从分段多项式检查

对 CubicSpline 梯度 $\widehat d$，每段是三次多项式；$\sup|\widehat d|$ 可检查该段两个端点及其导数的区间内实根。$\widehat d'$ 为二次多项式，$\sup|\widehat d'|$ 可检查端点及其一阶导数的区间内实根。所有结点和边界都须包含，不只在原采样结点取最大值。采用显式奇对称化后，仍可将反射分段多项式合并到同一对称结点集合上执行这一检查。

真算术或经过向外取整的根／值界可给严格包络；常规浮点根和加安全系数只提供数值资格，不能自动变成严格舍入认证。原始样条在结点准确也不限制其结点之间的过冲。

### 3.2 网格到连续经验最小值的界

设网格覆盖整个 $[-R,R]$、包含端点、最大间距不超过 $h$，完整目标的上述包络为 $L,D$。则

$$
0\le\min_{\mathcal G_h}\widehat F-\min_{[-R,R]}\widehat F
\le d_{mesh}:=\min\{Lh/2,Dh^2/8\}.
\tag{T.11}
$$

一阶界来自最近网格点；二阶界来自逐段线性插值误差 $D(x-u)(v-x)/2$。该证明不需要目标凸、连续最小点为内点，或某个优化器成功退出。不同惩罚通常有不同的 $D$；应按所优化的完整目标分别算界。

若每个已评价点的计算误差经控制不超过 $\zeta$，返回点相对计算网格最小值的误差为 $\epsilon_{alg}$，则相对真算术**连续经验目标**的 regret 至多

$$
\epsilon_{solve}\le\epsilon_{alg}+2\zeta+d_{mesh}.
\tag{T.12}
$$

不在网格上的局部求解器输出也可用网格最小值减去 $d_{mesh}$ 作为全局下界，加入该输出的评分误差后计算 regret。若无法严格控制 $\zeta$，报告应分列解析离散化界和实际数值资格，不将两者统称为严格连续全局证书。

## 4. 有限 $(n,K)$ 校准的理想对象与方向

本节为校准误差给出明确参照。固定 $n,K\ge2$，用相关块观测自旋积的期望 $t$ 参数化 bank 期望，去掉参数无关基准后记

$$
\Delta_{n,K}(t,\rho)=\alpha(\rho)+t\beta(\rho).
\tag{T.13}
$$

四种 active 模式的总体权重是 $(1+t y_1y_2)/4$；nuisance 对 bank 取期望后由 Gaussian 对称性消去观测符号。真实完整数据条件下，$t=t_N=N^{-1}\sum_jY_{j1}Y_{j2}$。这个仿射身份只在 bank 期望层成立，不取代实际训练中的噪声行。

定义教师无关的理想有限校准

$$
g(0)=0,\qquad
g'(\rho)=-\alpha'(\rho)-r(\rho)\beta'(\rho),\qquad
b(\rho)=-\beta'(\rho).
\tag{T.14}
$$

它使用已知发射模型和给定 $(n,K)$，没有使用未知教师 $\theta$。这一定义的含义是抵消“候选参数与模型生成参数同取 $\rho$”时的总体评分导数，而不是对每个训练教师另调参数。

这里的导数是先固定生成参数、仅对候选参数求偏导，再在生成参数等于候选参数处评价；不是沿对角线对 $F_{\rho}(\rho)$ 求全导数。后者还会产生生成权重变化项，不能作为 (T.14) 的估计器。

**命题 T.2。** 固定 $n,K\ge2$，$b$ 在 $[-R,R]$ 上连续且严格正。因此 $b_*:=\min_I b>0$，且

$$
\partial_\rho[\Delta_{n,K}(t,\rho)+g(\rho)]
=b(\rho)[r(\rho)-t].
\tag{T.15}
$$

**证明。** 给定 active 模式 $y$，固定对角 Price 微分给 raw 总体评分导数

$$
F_y'(\rho)=-a^2 y_1y_2 E\sum_k\pi_{k,y}(1-\pi_{k,y})
\sigma(-ay_1X_{k1})\sigma(-ay_2X_{k2}).
\tag{T.16}
$$

$\pi_{k,y}$ 是含完整 nuisance 的粒子权重。提取模式权重中 $t$ 的系数得到

$$
b(\rho)=\frac{a^2}{4}\sum_yE\sum_k\pi_{k,y}(1-\pi_{k,y})
\sigma(-ay_1X_{k1})\sigma(-ay_2X_{k2})>0.
\tag{T.17}
$$

$K\ge2$ 时各有限输入下 $0<\pi_k<1$，余因子严格正。被积函数有界且协方差参数处于紧的严格正定区间，故期望连续；紧性给 $b_*>0$。由 $\sum_k\pi_k(1-\pi_k)\le1$ 还有 $b(\rho)\le a^2$。代入 (T.14) 得 (T.15)。□

理想有限校准目标在连续区间的唯一最小点为

$$
q(t)=r^{-1}\!\left(\operatorname{clip}\{t,-r(R),r(R)\}\right),
\tag{T.18}
$$

恰为相同数据的精确连续 MLE／矩反演。它并不自动给出新计算优势，因为计算校准使用了已知模型信息，直接精确相关块似然也可作为基线。

这里必须区分**正确方向**与**强凸性**。二阶导数为 $b r'+b'(r-t)$，第二项没有已知固定符号；不能直接断言整个有限校准目标强凸。但是令 $m=\min_I r'>0$，积分 (T.15) 可得足够的二次增长：

$$
[\Delta(t,\rho)+g(\rho)]-[\Delta(t,q(t))+g(q(t))]
\ge\frac{b_*m}{2}|\rho-q(t)|^2.
\tag{T.19}
$$

内点情形由 $|r(u)-r(q)|\ge m|u-q|$ 积分；截断到右端点时 $t\ge r(R)$，在 $u<R$ 有 $t-r(u)\ge r(R)-r(u)\ge m(R-u)$，左端点对称。因此 (T.19) 不遗漏不可由候选教师实现的经验 $t$。

对 $g'$ 的奇对称化是与理想对象相符的：模式交换给 $\alpha$ 偶、$\beta$ 奇，故 $\alpha'$ 奇、$\beta'$ 偶，(T.14) 的 $g'$ 奇，锚定后的 $g$ 偶。$b_*>0$ 的存在性并不给当前 $(n,K)$ 一个已经计算或认证的良好数值下界；如果没有这种下界，下面的参数／风险式是条件界，不是实际精度认证。

## 5. 独立校准、奇对称化与全域误差

校准 prior 与数据、训练 prior 独立。设在事先固定的对称结点上估计 (T.14) 的导数。可以使用四 active 模式的 Price 表达，无需新观测：单个校准 bank 的 $g'$ 估计为 (T.16) 按已知模型 $t=r(\rho)$ 加权后取负号。因为它等于四模式有界导数的凸组合，绝对值不超过 $a^2$。它保持原有限目标的总体导数，不是用 pathwise derivative 和 Price derivative 的样本值逐 bank 相等来论证。

例如 $B_c$ 个独立校准 bank、$M_c$ 个预定结点，以相同 bank 共用所有结点，Hoeffding 加 union bound 给

$$
\max_{\rho\in\mathcal G_c}|\widehat d_{node}(\rho)-g'(\rho)|
\le a^2\sqrt{\frac{2\log(2M_c/\delta_c)}{B_c}}
\tag{T.20}
$$

的概率至少 $1-\delta_c$。这里只是一个保守充分界；跨结点复用不妨碍 union bound，也不使结点独立。如果数值 $r$ 只有统一误差 $\delta_r$，且用于加权的值限制在 $[-1,1]$，额外偏差最多 $a^2\delta_r$；实际导数计算误差仍须另外计入。

对任意初始样条 $S$，建议明确构造

$$
\widehat d(\rho)=\frac{S(\rho)-S(-\rho)}2,\qquad
\widehat g(\rho)=\int_0^\rho\widehat d(u)\,du.
\tag{T.21}
$$

这样奇导数、偶校准及 $\widehat g(0)=0$ 在真算术中成立。对真实奇函数的全域误差，奇投影不增大 sup 范数。若先在结点奇对称化后直接用 CubicSpline，需固定结点、边界条件并验证该插值操作保持奇性；不应默认任意端点导数条件都保持对称性。样条积分可由分段多项式反导数计算，无需为每个候选重新做数值积分。

### 5.1 结点误差不是全域误差

记校准结点最大间距为 $h_c$，插值后结点误差已控制为 $\delta_{node}$。如果已知 $\sup_I|g''|\le C_g$，则一个不依赖高阶样条误差定理的充分界是

$$
\|\widehat g'-g'\|_\infty
=\|\widehat d-g'\|_\infty
\le\delta_{node}+\frac{h_c}{2}\left(\sup_I|\widehat d'|+C_g\right)
=:\delta_{cal}.
\tag{T.22}
$$

这是把任意点连接到最近结点后的两个一阶变化界；原始结点估计先奇投影时，其节点误差不增大。若结点没有覆盖两个端点，该 $h_c/2$ 覆盖界不再自动成立。

可以从前述 pathwise 包络得到一个明确但可能很粗的 $C_g$。令

$$
u_K=\sqrt{2\log K}+(2\log K)^{-1/2},\quad
\overline D_K=
\frac{5a^2}{4(1-R^2)}(2\log K+2)
+\frac a{(1-R^2)^{3/2}}u_K.
\tag{T.23}
$$

二维 Gaussian 最大半径的尾界给 $EM_b\le u_K$、$EM_b^2\le2\log K+2$。于是 $|\alpha''|,|\beta''|\le\overline D_K$。由 $g''=-\alpha''-r'\beta'-r\beta''$、$r'\le A$、$|\beta'|=b\le a^2$ 得

$$
C_g=(1+r(R))\overline D_K+Aa^2
\tag{T.24}
$$

是有效统一界。不需要用同一 noisy 校准样条来“估计”真实 $g''$ 并循环认证自身。此粗界与 (T.20) 在可用预算下可能非常保守；本文件没有声明它们会通过拟定精度门。双份校准接近、结点误差小或插值曲线平滑本身，都不能替代一个成立的全域 $\delta_{cal}$。

## 6. 校准误差到参数和风险的条件界

为避免把不同误差混在一起，先固定真实数据集，记 $t=t_N$，理想有限校准条件目标为 $J_t(\rho)=\Delta_{n,K}(t,\rho)+g(\rho)$，其最小点为 $q(t)$。令训练 raw 对比与其条件 bank 期望满足

$$
\sup_I|\widehat\Delta^{raw}(\rho)-\Delta_{n,K}(t,\rho)|\le e_{MC}.
\tag{T.25}
$$

这一条件可由独立理论界、另行冻结的数值证据或保守诊断讨论；固定 $B=128$ 本身不是它已足够小的证据。

若 $\|\widehat g'-g'\|_\infty\le\delta_{cal}$ 且两者在 0 锚定，则 $\|\widehat g-g\|_\infty\le R\delta_{cal}$。设最终输出对于真算术 $\widehat\Delta^{raw}+\widehat g$ 的连续经验 regret 至多 $\epsilon_{solve}$（包括 (T.12) 中实际能够保证的项目），令 $c=b_*m$。由 (T.19) 得

$$
|\widehat\rho-q(t)|
\le\sqrt{\frac{2\{2e_{MC}+2R\delta_{cal}+\epsilon_{solve}\}}c}.
\tag{T.26}
$$

更精细地，校准误差在 $\widehat\rho$ 与 $q(t)$ 间的差至多 $\delta_{cal}|\widehat\rho-q(t)|$。解二次不等式可得

$$
|\widehat\rho-q(t)|\le
\frac{\delta_{cal}+\sqrt{\delta_{cal}^2+2c(2e_{MC}+\epsilon_{solve})}}c.
\tag{T.27}
$$

这两个界都没有假设受扰经验目标凸。若仅有校准导数扰动、没有 MC／求解误差，且求到受扰目标的实际最小点，则由 (T.15) 的导数符号还可收紧为 $|\widehat\rho-q(t)|\le\delta_{cal}/c$，包括约束端点；它不自动适用于只有函数值容差的求解器。

令 $M_r=\sup_I r'$、$q_R=r(R)$、$t_\star=r(\theta)$。对 (T.26) 或 (T.27) 任一已成立上界 $d_{comp}$，都有

$$
\begin{aligned}
|\widehat\rho-\theta|&\le d_{comp}+|t_N-t_\star|/m,\\
\mathcal R_\theta(\widehat\rho)
&\le\frac{\{M_r d_{comp}+|t_N-t_\star|\}^2}{2(1-q_R^2)}.
\end{aligned}
\tag{T.28}
$$

证明使用 $r(q(t))=\operatorname{clip}_{[-q_R,q_R]}t$ 对 $t_\star$ 的非扩张性，以及 Bernoulli KL 的统一上界。数据是 iid 教师观测时 $E(t_N-t_\star)^2=(1-t_\star^2)/N$，因此在一个成立的确定性 $d_{comp}$ 界下，$L^2$ 三角不等式可把 (T.28) 转成对应期望界。若计算界仅在高概率事件成立，还必须另加其失败概率乘风险上界，不能无条件忽略坏事件。

## 7. 误差分层和有限预算可支持的结论

| 层次 | 对象 | 必须单列的误差或限制 |
|---|---|---|
| 数据 | $t_N$ 相对 $r(\theta)$ | 真实观测不足；教师／前缀复用的相关性 |
| 训练 prior | 真正完整行经验目标相对给定数据的 bank 期望 | $e_{MC}$；独立单位是 bank，不是 $NB$ 项 |
| 有限校准 | 独立 prior 估计的 $g'$ 与理想有限 $g'$ | 结点抽样、模型数值、样条全域误差；单校准池的条件性 |
| 浮点评分 | 缓存概率、惩罚及累计和相对真算术 | $\zeta$ 或明确只为数值资格；不能默认为零 |
| 连续求解 | 同一经验目标上的搜索／网格误差 | 完整目标的 $D h^2/8$、候选输出 gap，见 (T.12) |
| 渐近替代 | 用 $2AwH$ 或统一二次替代理想有限 $g$ | 有限 $(n,K)$ 的目标差，不能归入求解误差，也不保证较小 |

理想有限校准在固定 $(n,K)$ 具有正确方向，并不意味着有限校准池已估准它，或有限训练 bank 足以学习；渐近修正则还受到有限维近似误差影响。在当前拟议预算下，应同时展示实际训练目标近优界、数据精确 MLE／矩反演、积分模型风险及各方法间配对差，允许 raw、简单二次、asymcorr 或 finite 校准的结果互有优劣。

缓存身份和 (T.9)—(T.12) 是固定实际输入下的数学结论；(T.25)—(T.28) 是带明确前提的风险接口。本文没有执行本包预算，没有认证任一现实 $\delta_{cal}$、$e_{MC}$ 或浮点余项，也没有声称实际训练将改善、连续全局求解已完成或新方法优于精确似然基线。
