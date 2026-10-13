# AngularWitness-v1：纯角度分子的有限和与解析误差

本说明核对 `MATH.md` (historical reference not distributed) 中 (AW.3) 的分子积分，配套实现为 [`angular_witness_integrals.py`](../../../script/angular_witness_integrals.py)。它不使用旧载荷方向 $B_h$ 的误差界，不计算新的 $Q_y$ 分母，也不认证项目教师下的曲率符号。生产输入、唯一固定方向、冻结和实际执行状态由本轮合同及收据负责；本说明中的制造验证不运行项目 177 节点网格。

## 1. 固定对象与精确输入

固定 $m=8,b=0$，输入 $a_i$ 为显式有理数，且 $|a_i|\le1$。取非零整数向量 $u\in\mathbb Z^8$，要求 $\sum_i u_i=0$，令

$$
S=\sum_i u_i^2>0,\qquad d=u/\sqrt S,\qquad
c_{ij}=(u_i-u_j)^2/S=(d_i-d_j)^2.
$$

代码使用精确有理 $c_{ij}$；不先把 $d$ 或 $\sqrt S$ 舍入。由单位范数及零和条件，

$$
\sum_{i<j}c_{ij}=m\|d\|_2^2-(\mathbf1^\top d)^2=8.
$$

令 $p_i(z)=\sigma(a_i z)$、$r_i=y_i-p_i(z)$ 及 $A_y(z)=\prod_i p_i(z)^{y_i}(1-p_i(z))^{1-y_i}$。只积

$$
P_y(z)=\sum_{i<j}a_i a_jc_{ij}r_i(z)r_j(z),\qquad
N_y=\int_{\mathbb R}\varphi(x)A_y(x)P_y(x)\,dx.
$$

实轴上 $0<A_y\le1,|r_i|\le1$，故 $|P_y|\le8$、$|A_yP_y|\le8$。载荷可以为负或零；包络与实现不依赖正载荷、反序对称或教师对称。当前项目中心的精确有理身份须由调用者绑定旧 `CANDIDATE_RISK_INTEGRALS.a_exact`。

全部 256 标签按整数 $0,\ldots,255$ 排序，第 0 个观察是最高位。既不合并互补标签，也不假定类内等权。

## 2. 闭条带上的 sigmoid 与分子包络

取 $0<\delta\le1$；默认 $\delta=1$。由于 $|a_i|\le1$，闭条带 $|\operatorname{Im}z|\le\delta$ 不含 sigmoid 的极点。对 $t\ge0$ 和 $|v|\le\delta$，

$$
|1+t e^{iv}|^2=(1+t)^2-4t\sin^2(v/2)
\ge(1+t)^2\cos^2(v/2).
$$

这里用 $4t\le(1+t)^2$。将 $t=e^{-a_i x}$ 代入，同时处理 $\sigma$ 和 $1-\sigma$ 的分子，得到

$$
|p_i(x\pm i\delta)|\le c\,p_i(x),\qquad
|1-p_i(x\pm i\delta)|\le c\,[1-p_i(x)],\qquad
c=\sec(\delta/2).
$$

于是 $|r_i(x\pm i\delta)|\le c$，而

$$
|A_y(x\pm i\delta)|\le c^8 A_y(x)\le c^8,
\quad |P_y(x\pm i\delta)|\le8c^2.
$$

Gaussian 的复移位模满足 $|\varphi(x\pm i\delta)|=e^{\delta^2/2}\varphi(x)$。因此对 $f_y(z)=\varphi(z)A_y(z)P_y(z)$，两侧分别有

$$
\int_{\mathbb R}|f_y(x\pm i\delta)|\,dx
\le M_N:=e^{\delta^2/2}c^8\,8c^2.
\tag{AN.1}
$$

这里 $M_N$ 是单侧边界积分上界，已经对整条实轴积分；无须再乘区间长度。默认值使用 $c=\sec(1/2)$，与 $a,u$ 的具体坐标无关，只依赖上述精确包络。

## 3. Gaussian 加权 Poisson 梯形误差

对 Fourier 约定 $\widehat f(\xi)=\int f(x)e^{-i\xi x}\,dx$，按 $\xi$ 的正负向对应条带边界移动积分路径。条带内各 logistic 因子有统一界，Gaussian 使竖边随实部趋于正负无穷而消失。由 (AN.1)，

$$
|\widehat f_y(\xi)|\le M_N e^{-\delta|\xi|}.
$$

这些函数及实轴导数具有 Gaussian 衰减，格点和与 Fourier 格点和绝对收敛，故可用 Poisson 公式

$$
h\sum_{k\in\mathbb Z}f_y(kh)
=\sum_{\ell\in\mathbb Z}\widehat f_y(2\pi\ell/h).
$$

除去 $\ell=0$ 的真积分，再对正负频率的几何级数求和，得

$$
\left|N_y-h\sum_{k\in\mathbb Z}f_y(kh)\right|
\le\frac{2M_N}{e^{2\pi\delta/h}-1}
=:\epsilon_{\rm alias}.
\tag{AN.2}
$$

系数 2 来自两侧频率；(AN.1) 已是单侧界，不能再额外翻倍，也不能只保留单侧频率。

## 4. 全权端点的遗漏格点尾

实际和为 $F_y=h\sum_{k=-N}^{N}f_y(kh)$，$L=Nh>0$，两个端点均取全权。遗漏的是 $|k|\ge N+1$。Gaussian 密度在 $[0,\infty)$ 单调下降，每个右端矩形满足

$$
h\varphi(kh)\le\int_{(k-1)h}^{kh}\varphi(x)\,dx\quad(k\ge N+1).
$$

因此

$$
\left|h\sum_{|k|>N}f_y(kh)\right|
\le16h\sum_{k=N+1}^{\infty}\varphi(kh)
\le16\int_L^\infty\varphi(x)\,dx
\le\frac{16\varphi(L)}L
=:\epsilon_{\rm tail}.
\tag{AN.3}
$$

最后一步由 $x/L\ge1$ 及 $\int_L^\infty x\varphi(x)dx=\varphi(L)$ 得到。此证明恰好对应全权的 $\pm L$；不能将有限和换成半权端点而复用同一差额解释。

默认 $h=1/8,N=88,L=11,\delta=1$，192-bit Arb 对这些解析表达的严格区间给出约

| 项 | 数值显示（完整真端点由 `pair_bounds` 返回） |
| --- | ---: |
| $\epsilon_{\rm alias}$ | $1.4400106784075753\times10^{-20}$ |
| $\epsilon_{\rm tail}$ | $3.0819189141954233\times10^{-27}$ |
| $\epsilon_N=\epsilon_{\rm alias}+\epsilon_{\rm tail}$ | $1.4400109865994667\times10^{-20}$ |

这些是每个标签共同适用的绝对分子误差，未乘 256。把分子除以旧真 $Q_y$、按教师加权以及教师 L1 误差转移，必须另作区间合成；不能直接把 $\epsilon_N$ 当成最终风险曲率误差。

## 5. Arb 有限和、编解码和预算状态

`finite_pair_sum(a,u,step='1/8',N=88,bits=192,hard_seconds=600)` 直接枚举所有 $i<j$，逐节点逐标签累加 $h\varphi A_yP_y$；所有 28 个配对均保留，包括精确零系数。它只返回 `N_finite_sum[256]`，不积 $Q$，不访问教师、科学数组、旧模块、随机数或优化器，也不读写任何文件。模块不修改 `sys.path`。要求外部公开 `python-flint==0.8.0`；两项计算函数在 `finally` 中恢复调用前的 `ctx.prec`，默认运算精度为 192 bit。

`record` 保存 Arb 向外端点的整数尾数与二进制指数；`arb_display_only` 不用于判断。`from_record` 以精确有理数重建并作包络。若保存有限和球为 $[F_y]$，真积分球由

$$
[N_y]=[F_y]+[-\epsilon_N^{\rm up},\epsilon_N^{\rm up}]
$$

构造；`enlarge` 明确取误差球的上端点。有限和的算术半径已经留在 $[F_y]$，不把它替换成显示小数，也不藏入解析误差预算。`pair_bounds` 只算解析常数与误差；不产生任何积分节点。

预算用单调时钟，在每个完整节点开始前检查；单节点和最后的端点序列化不可抢占。只要完成节点不足，或包括准备和序列化的实际耗时超过预算，状态即为 `budget_stopped`。结果保留 `node_indices_completed`、全部部分和、计数和实际秒数，供调用者按原样落盘；不自动重试、加密网格或扩大尾区。即使恰好完成所有节点但超时，也不标为合格完成。`budget_stopped` 的部分和不得提升为真积分证书。

## 6. 制造验证及其范围

制造测试由源码树之外的公开 PyPI `python-flint 0.8.0` 环境执行，日志保存在本地 `tmp/angular_witness_v1/pair_manufactured_test.log`，对应制造脚本同目录。32 项检查全部通过：

- 在八个非项目有理载荷、整数零和方向和五节点小网格上，全部 256 配对和与独立展开 $(SV-U^2)/S_u$ 的区间相交，差的绝对区间上界小于 $10^{-48}$。该独立表达只作制造测试，不充当项目的独立完整路径审计。
- 核对零偏置互补关系、全标签分子总和包含零、整数方向缩放不变、同时反转观察坐标时的最高位标签映射，以及全零载荷的精确零结果。
- 核对端点往返与对称误差扩大、正常结束及解析界计算后的精度恢复、极小预算返回零节点部分和且不重试；拒绝零方向、共同旋转、非整数方向、float 科学输入、越界载荷、不合格条带及其他非法配置。
- 静态核对模块没有项目或数组依赖；有限和节点计数与 28 配对计数一致。

每次完整测试中，有限和函数共访问 18 个制造节点，独立表达另访问 5 个制造节点，共 23 次制造节点访问；项目节点访问、科学数组读取、RNG、优化、F8 及旧科学内核调用均为 0。这些测试支持实现和输入边界，并不提供项目中心分子、项目风险曲率、真根负号或生产网格执行结果。
