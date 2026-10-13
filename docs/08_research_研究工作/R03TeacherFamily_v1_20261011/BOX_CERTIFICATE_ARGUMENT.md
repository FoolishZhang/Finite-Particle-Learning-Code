# R03：同一中心的连续教师盒证书

本页说明 `script/r03_continuous_compose.py` 的纯保存量合成。它不计算教师或候选积分，不优化、不生成随机数、不读写文件。实际证书必须由完整区间输入和来源绑定支持；制造夹具通过只确认实现路径，不证明真实研究对象通过。

## 对象、固定输入与门限

教师是八行零偏置、行长 `4/5` 的二维 Gaussian logistic 模型，角度为 `(-.7,-.5,-.3,-.1,.1,.3,.5,.7)+lambda e1`。以旧精确有理中心 `a0` 固定

\[
\lambda\in[-h,h],\qquad X=a_0+\prod_{i=1}^8[-r_i,r_i],
\quad r=(.75,.15,.15,.15,.15,.15,.15,.15)h.
\]

`h` 仅允许预先指定的 `.01,.001,.0001`；每次调用只合成一个盒，不在函数内自动缩盒、搜索门限或追加积分。`A` 是由开发基点 D 规则 Haa 逆矩阵对称化后冻结的精确有理 8×8 矩阵。浮点产生的每项须按其完整二进制值转为精确有理数，不能通过短小数重建；合成先检查其精确对称性和区间 LDL 正定性。正定性保证 A 可逆。

旧严格证书提供 `||G(a0,0)||2<7.572e-10` 及旧真根距 a0 小于 `1.805e-8`。G 是八维径向 F8 梯度；旧完整梯度帽当然也包围该子向量。程序默认这两个保存上界，但将实际输入数和来源哈希保留。`a0` 不是被设定为精确根，旧梯度误差从未置零。

候选保存数据是在覆盖 X 的同一盒上完整 128 个互补标签代表的条件径向梯度及 Hbb、Haa、Hperp 区间。128 是 256 标签的互补折叠，不使用行配对或旧 41 类约化。来源必须全部完成、无 partial 行、外层节点齐全、同一模型签名和配置。解析误差来自 `r03_candidate_interval.analytic_error_bounds`，固定 `teacher_error_L1_upper=0`、`teacher_mass_upper=1`。

须注意 Arb 的区间半径会向外舍入：解析界的 `a_upper_exact` 必须覆盖有限和实际保存 `a_enclosures` 的精确 dyadic 上端，而不只覆盖理想 `a0+r`。程序用保存端点的 Fraction 比较核对这些包含关系，避免把重新装载引入的进一步微小膨胀误当成原始计算域。

## 带符号教师展开与误差

真实教师证书提供所有 256 项 `q0_y`、`q′0_y` 的真值区间、`Q1>=||q′0||1` 和全实数范围的 `||q″lambda||1<.442323`。因此

\[
q_\lambda=q_0+\lambda q'_0+e_\lambda,
\qquad \|e_\lambda\|_1\le .2211615h^2=:t_h,
\qquad \sum_y e_{\lambda,y}=0.
\]

教师证书的各组件区间已包含教师有限和与解析误差，不在这里再次添加同一教师误差。真实质量恒为一、导数总质量为零来自模型；数值区间重叠只是矛盾检查，不取代理论恒等式。零 bias 保证 `q_y=q_(255-y)` 及对应导数相等。

合成分别计算

\[
G_1(X)=\sum_yq'_{0,y}g_y(X),\quad
H_0(X)=\sum_yq_{0,y}H_y(X),\quad
H_1(X)=\sum_yq'_{0,y}H_y(X),
\]

然后取 `Hfinite=H0+[-h,h]H1`。每个代表以 `q_y+q_(255-y)` 折叠；q′ 的负号保留。绝不先把每项 `q0_y+lambda q′0_y` 变成互不相关的区间权重而损失符号消去。

记条件梯度 L2 误差为 `epsg`，条件完整 ba Hessian operator 误差为 `epsH`，条件横向 operator 误差为 `epsP`。使用旧全局帽 `||g_y||2<=8`、`||Hba_y||op<=146.25`、`||Hperp_y||op<=8.25`，得到

\[
E_H=\varepsilon_H(1+hQ_1)+146.25t_h,
\qquad E_P=\varepsilon_P(1+hQ_1)+8.25t_h.
\]

这些是 operator 误差；主收缩检验中不将其无谓转换为 64 项逐元素误差。g1 的有限和误差为 `epsg Q1`。所有保存区间的算术误差与依赖膨胀已保留在其自身区间中。

## 收缩、存在性与同一连续分支

令 `Tlambda(a)=a-A G(a,lambda)`。对 A 的第 i 行记 `Ai`。教师 Taylor 余项严格满足总质量为零，可以先消去条件梯度中各标签共有的部分。对已保存条件有限梯度计算 `s_yi=(A g_y(X))_i`，取其所有标签区间值域的精确中点 `b_i`，并定义

\[
C_i=\max_y\sup|s_{yi}-b_i|+\|A_i\|_2\varepsilon_g,
\qquad \widehat C_i=\min\{C_i,8\|A_i\|_2\}.
\]

对每个固定 a，`sum e_lambda,y=0` 消去同一 b_i，故 `|Ai sum e_lambda,y gtrue_y(a)|<=t_h C_i`；条件积分误差通过 `||Ai|| epsg` 加入。旧全局上界独立成立，因此两者取小仍严格。128 个互补代表已覆盖所有标签的 s 值域。程序保留中点、有限值域、Ci、旧帽、采用的帽，以及新旧余项成本；共同平移所有条件梯度不改变中心化帽。

程序保存下面每一项的精确 dyadic 包围及余量：

\[
Y_i=7.572\times10^{-10}\|A_i\|_2
 +h\sup|(AG_1(X))_i|
 +h\|A_i\|_2\varepsilon_gQ_1+t_h\widehat C_i,
\]

\[
Z_i=\sum_j\sup|(I-AH_{aa,\mathrm{finite}}(X))_{ij}|r_j
 +\|A_i\|_2E_H\|r\|_2.
\]

虽然 G1 在整个 X 上包围，第一式只用于包围中心 `Tlambda(a0)-a0`，因此成立。对每个 i 必须严格证明 `Y_i+Z_i<r_i`，并且 `max_i(Z_i/r_i)<1`。后一数是半径加权无穷范数下的统一 Lipschitz 上界；前者给严格自映射。Banach 定理遂给每个 `lambda in [-h,h]` 在同一 X 内唯一根。A 可逆使不动点等价于 G=0。

上述中点只用于真实教师 Taylor 余项的解析零质量，绝不通过浮点或区间数值总和猜测零质量；原有限梯度矩、带符号 G1 和 Hessian 组合都未替换。分支的一阶漂移仍完整保留在 `h sup|AG1|` 中，没有重新居中分支位置。旧 `8||Ai||2` 余项成本与完整旧 Y 同时保存作独立比较；Hessian 的 `146.25/8.25` 余项帽保持原定值。

统一收缩及 `(a,lambda)` 连续性给根关于 lambda 连续。旧真根距离帽小于全部 r_i，且旧根本身是此教师在 lambda=0 的径向根，故唯一性将新连续分支与旧真根识别为同一分支。它不声明 X 外无其他根，也不声明一般求解器必定收敛。

## 完整局部稳定性

固定预定目标 `mu_bb=.1`、`mu_aa=.02`、`mu_perp_slice=.005`。分别对有限区间 Hbb、Haa 减去 `(mu+EH)I`，以及下述投影减去 `(.005+EP)I` 做严格区间 LDL。每个正主元下界都保存；一个主元未严格正就记录 `positive_lower_bound_unresolved`，不据此声称真实 Hessian 非正，不自动降低 mu。

令 U0 是精确 a0 的确定性 Householder 正交补：`u=a0/||a0||`、`v=u+e1`，取 `I-2vvT/(vTv)` 后七列。Arb 表达包围该精确矩阵；正交性及 `U0^T a0=0` 由公式解析成立，不由小数正交残差决定。投影矩阵为 `U0^T Hperp_finite U0`。operator 误差投影后仍不超过 EP。

整盒 `a0 dot a>0` 保证该固定切片与每个根的共同旋转方向横截。根处 Ward 恒等式给 `Hperp a=0`；切片上严格正定因而给旋转商上的正定性。不能把 `.005` 不经转换声称为变化的 `a^perp` 正交补上的统一最小特征值。

精确教师互补对称及 z 反射使 b=0 处 bias 梯度和 bias/radial 混合块为零；独立第二 Gaussian 坐标的反射使 transverse 与 ba 的混合块为零。因此 Hbb、Haa 和横向切片的三项正定认证共同给完整模型中模共同旋转的严格局部极小，而不只是径向极小。

## 固定纯角风险曲率

风险输入由调用者另行提供全 256 项真条件曲率 `rho_y(a)`，覆盖同一 X，固定旧方向 `d=u/sqrt(S)`。这里 rho 的语义是角路径下每标签负 log 模型概率的二阶导数；在秩一处反射给 `Q′_y(0)=0`，故 `rho_y=-Q″_y/Q_y=N_y/Q_y`。输入必须已经包围正分母、有限和、尾项与算术的全部误差，本合成不再猜测遗漏项。

由整个 rho 范围的最小下端和最大上端取其中点 B0，计算

\[
R''_\lambda(0)\in\sum_yq_{0,y}\rho_y(X)
 +[-h,h]\sum_yq'_{0,y}\rho_y(X)
 +[-t_h,t_h]\max_y\sup|\rho_y(X)-B_0|.
\]

中心化合法性来自 `sum e_lambda=0`；B0 可以是任何固定实数，保存区间范围中点给一个确定选择。只有最终包围的上端严格小于零才通过。它是每个 lambda 自己的真实教师风险，未将不同教师风险混合比较。若存在性、稳定性和风险三部分均通过，则每个此分支根上，固定行长的小角扰动保持候选 logit 与正确观察边际，并产生局部目标上升和自身真实风险下降；具体非零步长仍需另行量化。

若风险输入另外提供真实候选概率 `Q_true_intervals[256]`、真实分子 `N_true_intervals[256]`，并显式声明 `own_target_numerator_sum_zero=true`，可以对零阶项使用同一模型目标的恒等式。固定任意 a 和 Gaussian 节点，Bernoulli 条件概率恒满足 `sum_y A_y=1`；对固定角路径作二阶微分，再对独立横向 Gaussian 作矩收缩，得到条件分子多项式的逐节点恒等式 `sum_y A_y P_y=0`。可积性允许积分，故 `sum_y N_y(a)=0` 对盒内每个 a 都严格成立。因此

\[
R_0(a)=\sum_yq_{0,y}\frac{N_y(a)}{Q_y(a)}
=\sum_y\bigl(q_{0,y}-Q_y(a)\bigr)\frac{N_y(a)}{Q_y(a)}.
\]

这对应 R01 已验证的 own-target 零曲率恒等式，并不要求 `a=a0` 或将 q0 随 a 变化。q0 始终是固定的 lambda0 真实教师；Q 是盒内当前 a 的候选模型概率。只有这一零阶项使用改写，`sum q′0 rho` 的线性项和先前 Taylor 尾项保持原式。

实现要求全部 Q 严格正，并检查 Q 总质量包围一、N 总和包围零。从保存 Q/N 独立重建商后，与原 rho 做逐标签重叠一致性检查；端点重新装载可能引入微小外扩，因此不要求旧 rho 严格包含再次计算的商。两组各自已经由来源绑定生产程序证明为真包围，故取二者合法交集用于中心化式。单纯重叠绝不认证任意输入 rho 的数学有效性。交付层必须重建并绑定原风险生产合成，程序输出对此有明确标记。

结果同时保存 `R0_original`、`R0_centered`，并将合法交集存于 `R0` 后用于总风险式；若两个包围不相交，视为输入/算术矛盾而拒绝。若无显式 own-target 声明，仍允许原公式，`R0_centered=null`，不默默假设缺失的恒等式。这里的改写在真实目标数值执行前加入，没有以新的数值门结果为依据追认旧失败。

没有风险数据时状态是 `candidate_partial`，`passed=false`；根、收缩、稳定性的各项真实资格仍独立报告。数据齐全但任一门未过时返回 `unresolved`。只有损坏输入、缺标签或错误绑定才抛异常，科学不充分不能冒充程序错误。

## 可选反序破坏读数

程序另外包围隐式导数 `a′=-Haa^-1 G_lambda`。对每坐标取全部条件有限梯度区间值域的固定精确中点组成向量 gC，定义

\[
C_g=\min\left\{8,\ \max_y\sqrt{\sum_i\sup|g_{y,i}(X)-gC_i|^2}+\varepsilon_g\right\}.
\]

因为 `q′lambda-q′0` 的真实总质量为零，共同 gC 消去。其右端使用原带符号 G1 加上 L2 误差

`epsg Q1 + Cg*.442323*h`，

后项来自 `||q′lambda-q′0||1`，旧 `epsg Q1+8*.442323*h` 上界也独立保存。真实 Haa 的每项扩大 EH 后交给 Arb 区间求解；operator 帽蕴含每项误差帽，代价是更保守。若求解成功且 `a′1-a′8` 下端正，则保留该附加证据；否则记为 unresolved，不阻碍核心存在性/稳定性/风险认证。由此进一步推导所有非零 lambda 的反序破坏还需结合旧 lambda0 真根的配对对称事实。

## 接口和制造核验

`compute(compactinputs,candidateblocks,teacher,risk=None)` 返回严格可 JSON 序列化的结果，所有关键矩阵、Y/Z/余量、LDL 主元及风险区间保存 dyadic 端点。输入对象的规范 JSON SHA256 被保存；实际文件路径与字节来源绑定由调用者在冻结和交付层处理。

- `compactinputs`：`h_exact`、`radii_exact[8]`、`a0_exact[8]`、`A_exact[8][8]`、`candidate_analytic_bounds`；可选 `precision_bits`（默认 192）、`old_gradient_L2_cap_exact`、`old_root_distance_cap_exact`。
- `candidateblocks`：原 `r03_candidate_interval.finite_label_block` 的完整返回列表。
- `teacher`：原 `r03_teacher_derivative.compose` 的真值证书返回。
- `risk`：`schema="r03_risk_box_ratios_v1"`、`true_ratios_enclosed=true`、`rho_true_intervals[256]`、`a_enclosures[8]`、`direction={u_integer,S_integer}`；可保留 `source_bindings` 等审计字段。启用 own-target 改写还须提供 `own_target_numerator_sum_zero=true`、`Q_true_intervals[256]`、`N_true_intervals[256]`。h 统一来自 compactinputs，不采用风险对象内另一个 h。

直接运行模块只执行十六项制造保存量检查：八维线性根、常负风险、常 rho 中心化零余项、JSON 往返、缺风险不冒充完成、带符号教师导数及正反序导数、宽 Hessian 未决、非正定 A 未过、partial 输入拒绝、错误角方向拒绝、own-target 两式相同非零值、own-target 交集缩小共同不确定性、错误 rho 商绑定拒绝、梯度共同平移不变、新转移帽不大于旧帽、旧全局帽可独立胜出。制造数据覆盖人工区间盒，不调用任何积分核、不读真实科学输入，不写文件。
