# R03 一般八行长候选核：表示、误差界与制造检查

2026-10-11。本文件记录作者实现与人工检查，不是外部科研审核，也不代表 R03 目标已取得证书。本子任务只新增 [r03_candidate_interval.py](../../../script/r03_candidate_interval.py)；不改旧 producer、不读旧 NPZ、不执行目标梯度/Hessian、不优化或抽样。调用方须另行冻结科学输入、配置、源文件和预算后才执行目标。

## 对象、输入与八维坐标

候选为零 bias、共线行向量 $w_i=(a_i,0)$，$i=0,\ldots,7$。八个 $a_i$ 独立，可为精确有理数、原生 Arb 区间或保存的 exact-dyadic endpoint 记录；整个区间必须位于 $(0,1]$。**不要求 $a_i=a_{7-i}$，不使用四维 paired coordinates，不使用 41 paired-count 类。**

每个标签保存 $g_{a,y}\in\mathbb R^8$ 与原始 $8\times8$ 的 $H_{bb,y},H_{aa,y},H_{\perp,y}$。这些都是先对完整参数 $(b,a,v)$ 求导，再在上述共线点评价。单标签的 bias 梯度与 $b$–$a$ 混块一般不为零，本核不返回它们；只有精确教师整体标签互补对称时，总体对应项才因反射为零。横向与 $(b,a)$ 的混块由第二 Gaussian 坐标反射为零。

标签采用大端二进制：$y_i=(\mathrm{label}>>(7-i))\&1$。互补索引为 $255-\mathrm{label}$。任何互补归并都不假设教师类内等权或行交换对称。

## 有限和公式

令 $p_i(z)=\sigma(a_i z)$，$r_i=y_i-p_i$，$D=\operatorname{diag}(p_i(1-p_i))$，

\[
A_y(z)=\prod_i p_i(z)^{y_i}(1-p_i(z))^{1-y_i},\quad
u=e^v,\quad t=uA_y(z)e^{-uA_y(z)},\quad\Psi=E e^{-uA_y(Z)}.
\]

真实积分中的 $E$ 是一维标准 Gaussian 期望；有限程序将其全部替换为同一个 full-weight 对称 Gaussian 梯形和。没有 likelihood rescale、Gaussian mass 重归一或教师权重重归一。

固定采用中心化基 $\widetilde p_i=p_i-1/2$、$\widetilde y_i=y_i-1/2$。定义 $T_l=E[tz^l]$、$\widetilde P_{l,i}=E[tz^l\widetilde p_i]$（$l=0,1,2$），以及 $\widetilde{PP}_{l,ij}=E[tz^l\widetilde p_i\widetilde p_j]$（$l=0,2$）。共 $3+3\cdot8+2\cdot36=99$ 个矩列，再加独立 $\Psi$ 一列，合计 100 个标量构件。$\widetilde{PP}$ 只省去对称下三角，不省去任何行坐标。

令 $c_K=K\Psi^{K-1}$、$d_K=K(K-1)\Psi^{K-2}$，$R_1=T_1\widetilde y-\widetilde P_1$，以及

\[
\Gamma_l=T_l\widetilde y\widetilde y^T-\widetilde y\widetilde P_l^T-\widetilde P_l\widetilde y^T+\widetilde{PP}_l,
\quad D_l=\operatorname{diag}(T_l/4-\operatorname{diag}\widetilde{PP}_l),
\]

则对 $dv$ 的 integrand 为

\[
g_a(v)=-c_KR_1,
\]

\[
H_{bb}(v)=c_KD_0-d_K(T_0\widetilde{PP}_0-\widetilde P_0\widetilde P_0^T),
\]

\[
H_{aa}(v)=c_KD_2-d_K(T_0\Gamma_2-R_1R_1^T),\qquad
H_\perp(v)=c_KD_0-d_KT_0\Gamma_0.
\]

$H_{bb}$ 使用 $\operatorname{Cov}(y-p)=\operatorname{Cov}(p)$ 的精确消项；$H_{aa}$ 保留全部 $y_iy_j$ 项。横向坐标的独立零均值使跨粒子 mean-score 项消失，因此它没有 $R_0R_0^T$ 项。

中心化在每个精确有限网格上是代数恒等式：$r=\widetilde y-\widetilde p$、$p(1-p)=1/4-\widetilde p^2$，且 covariance 对常数平移不变。预计算先形成中心化 $\widetilde p$ 和其逐节点乘积，再积分，以减少先积累未中心化 $P/PP$ 后相减产生的区间依赖膨胀。Bernoulli likelihood 始终使用原始 $p$ 和 $1-p$ 两标签因子；没有改模型或求积密度。因此下文解析误差预算保持不变，中心化只改变合法有限和区间的运算表达，不保证所有输入盒都会得到更窄区间。

因子由 $1/S=\int e^{-uS}du$、$1/S^2=\int u e^{-uS}du$ 和 $du=u\,dv$ 给出，其中 $S=\sum_{k=1}^K A_k$。两个 distinct-particle 项各有两个 $uA$；上式不能再乘一次 $u$ 或 $K$。这是 旧 MOMENTS_AND_BLOCKS 的 H1–H7 (historical reference not distributed) 的逐原始标签版本，不是对 gradient 外层 integrand 直接求导所得的另一种有限密度。两种密度全实积分相同，但有限网格不同，误差界不能混用。

同一 $e^{-uA}$ 同时用于 $\Psi$、全部矩、梯度和 Hessian。`arb_mat` 实现一个 $1\times N_z$ 向量乘 $N_z\times99$ 矩阵。$uA\ge128$ 时可用合法区间 $e^{-uA}\in[0,2^{-184}]$ 与 $uAe^{-uA}\in[0,2^{-177}]$；程序先以 Arb 严格证明两个固定 cap。饱和不是截零，区间的 outward rounding 半径全部保留；无需 $t^2e^{-t}$ 新指数。

## 128 互补代表与原 256 权重审计

对于任何固定的八行长，$A_{1-y}(z)=A_y(-z)$。对称 Gaussian 有限网格也保留该恒等式。反射下 bias 得分变号，行长得分不变；上述三块和 $g_a$ 都不变。因此默认只算标签 $0,\ldots,127$，用 $q_y+q_{255-y}$ 加权即可复原这四种量的 256 标签和。

`aggregate` 强制调用方提供非空的 `teacher_complement_symmetry` 解析声明。声明应来自教师零 bias 和中心对称 latent prior 的解析性质，不以两个近似区间重叠充当证明。原来的 256 个 $q$ 输入记录、其转换后区间、总质量与 128 个折叠权重全部返回；不把成对权重平均后覆盖原记录。有限权重的互补区间可以因算术半径而不逐位相同。

四种不变量的折叠等式本身不要求 $q_y=q_{1-y}$；声明的额外作用，是允许调用方把总体 bias 梯度和 $b$–$a$ 混块定为精确零。若无法解析声明教师对称性，本 API 拒绝把这四种量包装为该对称目标的聚合。`label_mode="all_labels"` 可算完整 256 条作独立实现核对，但不解除省略其他导数项的科学条件。

## 通用八行长解析预算

数学来源为旧 梯度求积证明 (historical reference not distributed) 和 Hessian 求积证明 (historical reference not distributed)。其中配对只用于写出相同长度的重复计数；逐行 score 包络和积分证明不需要行相等。对新输入必须明确替换为

\[
m=8,\quad C_a=\sum_{i=0}^7\bar a_i,\quad
C_R^2=\sum_{i=0}^7\sec^2(\bar a_i d_z/2),\quad
s_*^2=\max_i\sec^2(\bar a_i d_z/2),
\]

其中 $\bar a_i$ 是 Arb 输入的严格上端点。这里没有外加系数 2，尤其不能把八行数组送入旧 `m=2*len(a)` 接口得到 16 行。使用上端点使以下解析界对整个输入参数盒成立；实际有限和仍使用完整输入区间。区间依赖膨胀是合法外包，不作为新的数学截断误差另加一次。

API 严格检查 $L\ge2$、正步长/strip、整数网格边界，

\[
\theta=C_ad_z<\pi/2,\qquad d_v<\pi/2,\qquad
e^{v_+}\eta>2,\quad\eta=2^{-8}e^{-C_aT}.
\]

梯度 score 满足 $\|zr\|_2\le\sqrt8|z|$。在 $|\operatorname{Im}z|\le d_z$ 中，likelihood 位于角宽 $\theta$ 的右半平面，$|uAe^{-uA}|\le1/(e\cos\theta)$。因此内层边界包络为

\[
M_\Psi=e^{d_z^2/2},\quad
M_D=\frac{e^{d_z^2/2}\sqrt{C_R^2(1+d_z^2)}}{e\cos\theta}.
\]

对任一矩 $X$，内层 alias 为 $2M_Xq_z/(1-q_z)$，$q_z=e^{-2\pi d_z/h_z}$。梯度尾分别为 $2\phi(L)/L$ 与 $2\sqrt8\phi(L)/e$；幂和乘积传播保留 $(1+\delta_\Psi)^{K-1}$。外层梯度 boundary $M_v=K\sqrt8\sqrt{2/\pi}/\cos d_v$，alias 为 $2M_vq_v/(1-q_v)$。离散下尾使用几何和；上尾按至少一个粒子 $|Z|\le T$ 与全 $K$ 粒子超 $T$ 分开，bad 部分保留 $1+2h_v/e$ 的离散单峰余量。

Hessian 完整 $(b,a)$ score 为 $G=(r,zr)$。令 $F=e^{d_z^2/2}/(e\cos\theta)$、$J=2+d_z^2$；七个内层矩的 boundary 与离散尾如下，$p_L=2\phi(L)/L$、$j_L=2\phi(L)(L+2/L)$。

|矩|boundary 范数包络|离散尾包络|
|---|---|---|
|$\Psi$|$e^{d_z^2/2}$|$p_L$|
|$T_0$|$F$|$p_L/e$|
|$R=E[tG]$|$F\sqrt{C_R^2J}$|$2\sqrt{16}\phi(L)/e$|
|$Q=E[tGG^T]$|$FC_R^2J$|$8j_L/e$|
|$B_0=E[tB]$|$Fs_*^2J/4$|$j_L/(4e)$|
|$Q_p=E[trr^T]$|$FC_R^2$|$8p_L/e$|
|$B_p=E[tD]$|$Fs_*^2/4$|$p_L/(4e)$|

各矩 alias 同前。`analytic_error_bounds` 返回七种矩的 boundary、alias、tail 和总误差，随后用完整乘积扰动不等式传播到 $T_0Q-RR^T$ 和 $T_0Q_p$，没有把逐元素界冒充 operator 界。外层 Hessian boundary 为

\[
M_p=\frac1{4\cos d_v}+\frac8{\cos^2d_v},\qquad M_{ba}=(1+K)M_p.
\]

外层 alias 仍为 $2Mq_v/(1-q_v)$。令 $\ell_j=h_ve^{jv_-}/(e^{jh_v}-1)$，下尾 $E_{p,-}=K\ell_1/4+8K^2\ell_2$、$E_{ba,-}=(1+K)E_{p,-}$。上尾使用 $se^{-s}$ 和 $s^2e^{-s}$ 的总变差 $2/e$ 和 $8/e^2$，因此 bad 系数为

\[
C_{bad}=\tfrac14(1+2h_v/e)+8(1+8h_v/e^2).
\]

good 部分为 $G_p=\{1/4+8(1+U\eta)\}e^{-U\eta}$、$G_{ba}=(1+K)G_p$。设 $\bar p_T=2\phi(T)/T$、$\bar t_{2T}=2\phi(T)(T+1/T)$；bad 部分分别为 $C_{bad}\bar p_T^K$ 和 $C_{bad}(\bar p_T^K+K\bar t_{2T}\bar p_T^{K-1})$。这些控制省略的离散无限网格；不能改用连续 Laplace 尾代替。

标量预算统一覆盖所有原标签，没有 128、256 或元素数倍增。设精确有限教师权重 $q^f\ge0$、质量上界 $m_f$、$\|p_*-q^f\|_1\le\epsilon_T$，必须使用

\[
\sum p_* H-\sum q^fH^{grid}
=\sum q^f(H-H^{grid})+\sum(p_*-q^f)H.
\]

真实单标签全域上界为 $\sqrt{8K}$（梯度 L2）、$(16+1/4)(1+K)$（完整 Hba operator）、$8+1/4$（横向 operator）。K=8 时是 8、146.25、8.25。最终预算分别为 $m_f E_{uniform}+\epsilon_T C_{global}$。原有限 $q$ 球的算术半径已经保留在聚合中；$\epsilon_T$ 应界定真实教师与球中所包的精确有限权重之差，不能以球 midpoint 代替有限权重。

$H_{bb}$ 与 $H_{aa}$ 是 full Hba 的 principal blocks，各自适用同一 Hba operator 预算；在精确 mixed 为零时组合 block diagonal 取两块误差的最大值，不再乘 $8$ 或 $\sqrt{128}$。这些预算没有自动证明驻点存在、强凸性、横向曲率或任何最终谱门。

## API、超时与保存责任

```python
cache = precompute(a_values, config)
block = finite_label_block(cache, range(0, 8), hard_seconds=remaining_seconds)
# Caller saves each completed block exactly, including every conditional row.
result = aggregate(all_blocks, q_256,
    teacher_complement_symmetry="caller-owned analytic proof statement")
budget = analytic_error_bounds(a_values, config,
    teacher_error_L1_upper="...", teacher_mass_upper="...")
```

`config` 字段为 `K, precision_bits, hz, L, dz, hv, vmin, vmax, dv, T, saturation_threshold, label_mode`。默认 K8、192bit、$h_z=1/40,L=11,d_z=1/8,h_v=1/8,[v_-,v_+]=[-32,64],d_v=1/2,T=4$。这些是可替换默认值，**不是本轮科学冻结**。未知字段、浮点有理参数、非整数网格或非法输入盒直接拒绝。全部入口临时设置并恢复 Arb 精度。

`finite_label_block` 允许任意互不重复的合法标签子集。它在外层点及每 32 个内层因素检查 deadline；一次矩阵乘法和一个外层矩阵更新是原子操作，因此这是合作式 wall budget，不是操作系统强制终止。超时抛 `CandidateBudgetStop`，`.partial` 返回已完成标签和当前标签已完成外层前缀；半个内层点只计资源，不进入前缀值。调用方须保存部分记录并停止，不能把前缀当完整积分。`aggregate` 拒绝缺标签、重复标签、未完成块、错模型/配置或错形状。模块没有文件读写、历史运行入口、自动恢复或自动重试。

完整块保留输入记录、输入区间、配置哈希签名、Gaussian 网格质量、全部逐标签的原生 endpoint 值和实际计数。缓存、块和聚合 schema 明确记录 `moment_basis="p_i_minus_one_half"`、`label_coefficient_basis="y_i_minus_one_half"`；两字段也进入模型签名，聚合拒绝不匹配的矩基。中心化形式固定使用，不新增配置开关。聚合只做后处理，条件块仍须作为独立公开 JSON 保存，不能只保留教师加权后的结果。

默认网格的精确工作量是每标签 $881\cdot769=677489$ 个 Laplace 因素；128 标签共 86,718,592 个因素，256 标签为其两倍。每个外层点有 99 列矩收缩，$\Psi$ 另算；共享预计算的 logistic 指数为 $16\cdot881=14096$ 次，Gaussian 指数 881 次。这些是公式计数，不是实测耗时。一般八行的列数比旧 paired 核多，运行前应另冻 wall budget；当前人工 3×3 测试不能作为全网格的可靠耗时测量。

## 已执行制造检查与边界

公开 `python-flint==0.8.0` 环境运行 `python script/r03_candidate_interval.py --self-test` 通过。人工行长为 $a_i=(i+1)/16$，明确不具反序配对；Gaussian 和 Laplace 都仅 3 个点，K=3。所有对象现场构造，不读取科学输入。

检查包括：5 个点值标签与独立逐点 same/cross 表达比较；2 对互补标签反射；1 个区间行长标签包含原点的梯度及三块矩阵；完整 128 代表对人工四原子对称 $q$ 的聚合及原 256 权重保留；中心化 schema；参数拒绝；合作式 deadline 部分输出；八行参数盒的纯标量预算 m8 和全域 146.25 系数。独立 `_direct_manufactured` 参考明确保持原始未中心化的 $r=y-p$、$p(1-p)$ 表达，没有同时改成中心化实现。共 1200 个独立数值区间重叠比较；生产路径人工标签调用134条、因素1206次；另有7条独立 direct 参考标签、63个参考因素。上述计数不含 scalar budget 的指数表达，不称为科学目标积分。

本子任务实际科学目标评估为 **0**，旧 NPZ 读取0、随机数0、优化步0。仍需调用方完成教师证书、科学输入/选择/停止规则冻结、实际有限区间计算与保存、解析预算合成和所需的后续定理验证。宽参数盒可能使有限区间很宽；合法外包不保证足以通过任何数值门。
