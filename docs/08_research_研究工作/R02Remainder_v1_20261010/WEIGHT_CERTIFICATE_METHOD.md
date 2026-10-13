# R02：Gaussian 比较权重证书与保存量传播器

2026-10-10。实现为 [r02_remainder_propagate.py](../../../script/r02_remainder_propagate.py)。本文件说明 FINITE_PROPAGATION.md (historical reference not distributed) 的 P1–P33 如何使用保存量合成。代码不导入生产 noise 模块，不读文件，不运行 MC、NPZ、优化或实际 `D_n` 评分。

正式 Gaussian 比较网格为 `h=1/128,N=1536,L=12,192 bit`，共 3073 个节点；该网格必须由主执行器完成冻结后才运行。本文编写时只执行了五节点制造网格，**没有执行正式 3073 节点，也没有对实际噪声证书运行四规模传播**。

## 1. 两个纯函数及输出

`make_weight_grid(step='1/128', N=1536, bits=192, hard_seconds=120)` 只生成标准 Gaussian 的 `φ(z),Φ(z)` 保存区间。输出 schema 为 `r02_remainder_weight_grid_v1`，包含：

- `status`：`complete`、`budget_stop` 或 `runtime_failure`；未完成行和运行失败不被掩盖。
- `configuration`：精确步长、N、L、精度和 python-flint 版本。
- `rows`：每行 `index,z_exact,phi,Phi`，超越函数结果保存为精确 dyadic 下上端点。
- `nodes_prescribed,nodes_completed,work_counts` 和实际时间；时间不含调用者写盘。

每个节点用 Arb 计算

\[
\phi(z)=e^{-z^2/2}/\sqrt{2\pi},\qquad
\Phi(z)=\tfrac12\operatorname{erfc}(-z/\sqrt2).
\]

输入必须使用显式有理数；拒绝浮点步长。有限网格必须覆盖 `[-L,L],L≥2` 并包含 `−1,0,1`。制造网格可用 `h=1,N=2`，但不得将其当正式网格的替代执行。

`propagate(old_constants, noise_bounds, grid)` 只读取三个传入对象并执行保存点运算和闭式标量合成。其中：

- `old_constants` 是旧 `CONSTANTS.json['composition']['constants']`。
- `noise_bounds` 是单独保存的 `r02_remainder_noise.compose_bounds` 输出；本模块不调用或导入这个函数。
- `grid` 是上述完整 Gaussian 保存网格。

传播器要求固定 n 序列 `[64,256,1024,4096]`。返回 schema `r02_remainder_finite_propagation_v1`。每个 `rows` 项包含 `n`、`budgets.E1` 至 `E5`、`E_total`、`E_total_upper`、`D_second_order`、`theoretical_D_interval`、`bound_qualification_passed`、`negative_sign_certified`、`status`，以及四项 Gaussian 比较权重的完整保存预算。`theoretical_D_interval` 与 `D_n_theoretical_enclosure` 是同一内容的两个入口。

每行另有 `propagation_inputs`，原样保存 E1–E5 实际使用的 `s,L0,L1,L2,deltaG,deltaf,deltap,Vdiff,a2_squared,R0,RCDF,Delta,M2,Mmv,Mcv`、四项 W 上帽和两项 Wg 帽，以及 `m_H,cbar,J_K,v,b2`；`M2/Mmv/Mcv` 同时保存表达式区间和单独的上端帽。`s_lower_exact,s_upper_exact` 提供尺度的精确有理端点，不能把分母 s 的上端误作安全下界。顶层保留全部活跃表达式和 Hermite 闭式常数，便于独立有理区间重构组合因子。

资格与符号分开：完整合法输入可得到一个很宽但有效的区间，此时资格通过、符号仍为 `unresolved`。只有严格区间的上端小于零才设置 `negative_sign_certified=true`。不存在按预测符号、旧 MC 结果或预期容差调整的通过门。

输入 noise 证书须由外层检查器单独验证，输出明确标注 `noise_certificate_must_be_verified_separately=true`。本传播器只验证传入结构、有限性和所需标量资格，不声称重新证明 noise 模块的解析界。

## 2. 保留权重的上下 Riemann 界

对固定 `d≥0`，令 `w(z)=min(Φ(z)+d,1)^r`，这是非减函数。每个 cell `[a,b]` 使用端点 CDF 的保守下上界：

\[
w_-=\min(\underline\Phi(a)+d,1)^r,
\qquad
w_+=\min(\overline\Phi(b)+d,1)^r.
\]

CDF 区间可与数学上已知的 `[0,1]` 相交，避免低精度区间因舍入略越过这个范围。下上权重同时裁剪到 `[0,1]`。

当 cell 不跨越 0 时，

\[
\int_a^b|\phi'|=|\phi(b)-\phi(a)|.
\]

当 cell 不跨越 `−1,1` 时，

\[
\int_a^b|\phi''|=|\phi'(b)-\phi'(a)|
=|-b\phi(b)+a\phi(a)|.
\]

这些导数质量由保存的 `φ` 区间经有理端点运算取得，不重新评价导数或任何网格函数。若相减后的区间跨零，绝对值质量的下界为零、上界为两个端点绝对值的最大值；不会把有符号差错误当成正质量。

若质量区间是 `[M−,M+]`，则 cell 的加权积分在

\[
[M_-w_-,M_+w_+]
\]

内。所有 cell 下界和、上界和分别用 Arb 运算；最终取下界和的精确下端、上界和的精确上端，作为全网格区间端点。由于网格包含 `−1,0,1`，没有 cell 跨越被遗漏的导数符号变化点。

实线窗口外使用 `0≤w≤1`。对 `L≥2`，

\[
\int_{|z|>L}|\phi'|=2\phi(L),\qquad
\int_L^\infty|\phi'|=\phi(L),\qquad
\int_{|z|>L}|\phi''|=2L\phi(L).
\]

所以 `W1` 加 `2φ(L)` 的上帽，`W1plus` 加 `φ(L)`，`W2` 加 `2Lφ(L)`；尾部下界安全取零。正式 `L=12` 时最后一项是 `24φ(12)`。

每个 W 记录包含：`kind,exponent,delta_upper_exact,support`、上下 Riemann 和、`tail_upper,enclosure,upper,riemann_width_upper,total_enclosure_width_exact,saved_cells_used`。这里的区间是**给定 d 的 Gaussian 比较积分**的区间。它的下端不代表实际 nuisance 误差或梯度误差的下界。

传播中的 d 使用相应噪声/活跃预算的上端，因此比较权重只向保守方向扩大。四项权重为：

| 保存键 | 数学对象 |
|---|---|
| `W1_q_Delta` | `W1(98,Δ)` |
| `W1_q_minus1_Delta` | `W1(97,Δ)` |
| `W2_q_deltaG` | `W2(98,δG)` |
| `W1plus_q_minus1_deltaG` | `W1plus(97,δG)` |

全部使用 `upper` 参与有限传播，Riemann 区间宽度和尾帽保留供审读，不按结果缩小网格或调高精度。

## 3. 从 Edgeworth 残差恢复实际噪声范数

noise 模块给出的残差针对 `g−φ−λe`，其中 `e=He3φ`、`λ=κ3/(6v√(nv))`。不能直接把其残差范数当成 `g−φ` 的范数。

令 `a=√(3−√6),b=√(3+√6)`。`e′=−He4φ`，故

\[
\|e\|_\infty=\max(|\operatorname{He}_3(a)\phi(a)|,
|\operatorname{He}_3(b)\phi(b)|).
\]

`e′` 的绝对极值从 `He5` 的实根取得，即

\[
\|e'\|_\infty=
\max_{z\in\{0,\sqrt{5-\sqrt{10}},\sqrt{5+\sqrt{10}}\}}
|\operatorname{He}_4(z)\phi(z)|.
\]

高斯尾中这些函数趋零，负根因奇偶性给出相同绝对值。根据 e 在这些极值点之间的总变差，以及 Gaussian 偶矩，

\[
\|e'\|_1=4\sqrt6[a\phi(a)+b\phi(b)],\qquad
\|e\|_2^2=\frac{15}{16\sqrt\pi}.
\]

另外 `||(1−z²)φ||∞=φ(0)`，`||φ′||∞=φ(1)=1/√(2πe)`。以上是闭式标量计算，不是新生产积分或网格。

noise 行的字段与本传播器变量对应为：

| noise 字段 | 变量 |
|---|---|
| `error_density_derivative_Linf['0']` | `r0inf=R0` |
| `error_density_derivative_Linf['1']` | `r1inf` |
| `error_CDF_Linf` | `RCDF` |
| `error_density_L2` | `r0L2` |
| `error_density_prime_L1` | `r1L1` |

每项均取保存区间的**上端**作为误差帽。实际范数的充分上帽是

\[
\begin{aligned}
\delta_f&=|\lambda|\|e\|_\infty+r0inf,\\
\delta_p&=|\lambda|\|e'\|_\infty+r1inf,\\
\delta_G&=\min(1,|\lambda|\phi(0)+R_{\rm CDF}),\\
V_{\rm diff}&=|\lambda|\|e'\|_1+r1L1,\\
a_2^2&=(|\lambda|\|e\|_2+r0L2)^2.
\end{aligned}
\]

使用 Minkowski 不等式和三角不等式得到这些界。直接噪声导数帽与 Gaussian 比较帽取较小者时，也只比较其上端：

\[
L_0=\min(L_{0,\rm direct},\phi(0)+\delta_f),\qquad
L_1=\min(L_{1,\rm direct},\phi(1)+\delta_p),\qquad
L_2=L_{2,\rm direct}.
\]

这不会将舍入区间的中点或下端错误作为上帽。该模块不选择或修改 noise 证明中的衰减率。

## 4. 活跃常数、有限区间和报告含义

活跃常数完全按 `FINITE_PROPAGATION.md` 的 P24、P27、P32、P33 从旧严格标量计算：类方差、类均值、类协方差、`Mmv,Mcv`、`B4_bar`、`M2` 以及 `vmax`。传播器保留各表达式的区间，并核对方差正性和旧二阶系数的重构区间相交；没有新 active 积分。

按 P5–P9 计算 `E1,…,E5` 后，令 E 是其和的严格上端。最终输出

\[
D_n\in D_n^{(2)}+[-E,E],\qquad
D_n^{(2)}=-m_HJ_K/s+b_2/n.
\]

每项误差与 `|D_n^(2)|` 的比值、总比值均保留；若二阶预测区间包含零，相应比值输出空值，避免除以未证非零的量。首项比较预算另记为

\[
E_{\rm first}=|b_2|/n+E.
\]

`E1`–`E5`、`E_total` 和上述比值的下端只是**上界表达式的区间下端**，不是实际误差或实际相对误差的下界。`theoretical_D_interval` 是有限理论包络，不是新模拟置信区间。输出同时明确 `target_D_n_evaluated=false`。

## 5. 冻结前制造测试记录

使用公开 `python-flint==0.8.0` 环境执行一次测试程序，全部通过：

- `h=1,N=2,L=2` 的五个 Gaussian 节点；未调用默认生产参数。
- 六项质量恒等式：对 `W1,W1plus,W2` 分别取 `r=0,d=0` 和 `r=98,d=1`，权重恒为一；保存区间均包含 `2φ(0),φ(0),4φ(1)`。
- 两次相同的制造传播重构，每次四行；使用明确制造的标量和零 Edgeworth 残差，核对 E4、E5 精确为零、输出确定性、保存网格未重算、精度恢复以及资格/符号字段存在。此测试不代表任何实际有限 n 的结果。
- 零秒预算产生零节点并保存 `budget_stop`。
- 五项非法输入拒绝：不含 `±1` 的网格、`L<2`、浮点步长、负误差上帽和被改乱的节点序号。

本程序共新增五个制造 Gaussian 节点；生产节点为零，实际噪声函数调用为零，实际 `D_n` 评分为零。传播器重构只使用保存点和闭式标量，不会在检查时重放正式 Gaussian 网格。
