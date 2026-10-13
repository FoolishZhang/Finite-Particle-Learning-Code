# R02Remainder-v1：有限噪声证书方法与实现接口

2026-10-10。实现为 `script/r02_remainder_noise.py`。本文给出将 Fourier 规划变成固定执行方案的接口；是否实际执行及结果由本轮冻结、运行和合成记录另行说明。此模块不读写文件，不导入旧科学代码，不计算 `D_n`、MC、NPZ 或活跃标记矩。

## 1. 固定生产参数与旧输入

传入前轮 `CONSTANTS.json` 的 `composition.constants` 中三个完整 dyadic 区间：`v,d0,kappa3_L`。该依赖已经认证 `d0=E log cosh(G/2)`、`v=Var log sigma(G)`；保持它们与同一模型绑定。

生产配置是 `delta=1/4`、最高标准化频率 `a=9/2`、频率间距 `1/64`，共 273 点和 272 段。原 Gaussian 全权梯形 `h=1/16,N=192,L=12`，复条带半宽 `eta=1`，192 bit，墙钟 600 秒。这将规划中的 `delta=1/2` 缩到 `1/4`，其余低频公式保持通用。没有根据目标梯度正负选参。

385 个 Gaussian 节点复用 `logcosh`，同时计算原始第四中心矩和 273 个复 CF 有限和，至多 105,105 次复指数。按实标量和计数是 `1+2*273=547` 个有限和；每个复 CF 的两个实分量分别计数。没有 Fourier 数值求积。

输入和输出中模型量的浮点数一律不作为精确数学输入；有理参数只接受整数、有理字符串或 `Fraction`。显示小数仅供阅读。只有墙钟时间使用普通时间标量。

## 2. 第四矩与 CF 误差

使用恒等式 `L−ell=x/2−logcosh(x/2)+d0`。先保存原始有限和 `raw4_finite`，再加解析误差形成 `raw4_true`，最后除以保存的 `v²` 得到 `mu4_true`，不把输入区间半径忽略。

在复条带上

\[
|L-\ell|\le |x|+B_4,
\quad B_4=d_{0,+}+\eta/2+\log\sec(\eta/2)+\eta/2.
\]

单侧边界积分帽是 `exp(eta²/2) E(|Z|+B4)^4`。绝对标准 Gaussian 矩为 `1,sqrt(2/pi),1,2sqrt(2/pi),3`，二项式展开给出完全标量表达。全权无限梯形混叠帽为 `2M/(exp(2pi eta/h)−1)`。

真实轴上 `|L−ell|<=|x|+d0_upper`。对 `L>=2`，单项 `x^k phi(x)` 在尾部不增，`k<=4`，所以遗漏的格点尾由尾积分上帽控制。所用双侧多项式尾是

\[
P_0=2\phi(L)/L,\;P_1=2\phi(L),\;
P_2=2\phi(L)(L+1/L),\;
P_3=2\phi(L)(L^2+2),\;
P_4=2\phi(L)(L^3+3L+3/L).
\]

第四矩余尾为 `sum binom(4,k) d0_upper^(4−k) Pk`。有限 Arb 算术半径保持在有限和区间内，不重复或丢弃。

每个标准化频率 `u` 的 CF 是 `E exp(iu(L−ell)/sqrt(v))`。条带内 `|Im log sigma(x+iy)|<=|y|`，故单侧积分帽为

\[
M_\psi(u)=\exp\{\eta^2/2+|u|\eta/\sqrt{v_-}\}.
\]

混叠帽 `2Mpsi/(exp(2pi eta/h)−1)`，遗漏真实轴尾 `2phi(L)/L`。这是复模误差；对实虚部各加该帽合法。用于模上帽时直接使用 `abs(finite_complex)+error`，避免把复误差矩形再次放大；可与普适帽 1 取最小。

`E|U|<=sqrt(EU²)=1` 给 `|psi'|<=1`。相邻段的帽为

\[
\rho_j=\min\{1,\max(b_j,b_{j+1})+1/128\},
\]

其中 `bj` 是节点真模的上帽。保存全部 272 段。资格门为全部 `rho_j<1`、`mu4>0` 且第四矩区间宽度 `<1e−8`。这仅为噪声证书资格，不是有限梯度判向门。

## 3. Fourier 解析合成

低频完全采用 `FOURIER_PLAN.md` F5–F10，使用执行 `delta=1/4`。第四矩、`|mu3|` 与正项系数取保守上端，衰减率 `c,c_mod` 取保守下端；若对数分支或正衰减门不满足就失败，不调参重跑。

低频误差帽写成有限正项列表 `sum coefficient*t^power*exp(−decay*t²)`。`A_g`、`A_g'` 包含原 `A0,A1` 以及 Gumbel 帽

\[
b(t)=\pi^2t^2/(12nv),\qquad d(t)=\pi^2t/(6nv).
\]

不使用 `min` 仍是合法上帽，有利于精确正项合成。低频原 CF 模用 `exp(−max(c,c_mod)t²)`；两者均是已经向下圆整的有效率。

低频正项积分可保守延伸到无穷。令 `I_p(c)=int_0^infty t^p exp(−ct²)dt`，实现使用

\[
I_0=\sqrt\pi/(2\sqrt c),\quad I_1=1/(2c),\quad
I_{p+2}=(p+1)I_p/(2c).
\]

平方将所有正交叉项完整展开后积分。没有把区间中值代替预算上端。

中频逐段使用 `rho_j^n`；CF 导数为 `sqrt(n)rho_j^(n−1)`。积分 `t^p` 在标准化频率段乘 `sqrt(n)` 后的端点之间作解析幂差。高频用规划 F1 的 BV 常数

\[
B=2\sqrt{v_+}(1+e^{1/2})/\sqrt{2\pi},
\quad |P_n(t)|\le(B\sqrt n/t)^n.
\]

统一矩函数 `tail_power_moment(p,k)` 对中频及高频返回 `int t^p rho(t)^k` 上帽。高频表达为

\[
n^{(p+1)/2}B^k a^{p-k+1}/(k-p-1),\qquad k>p+1.
\]

`sqrt(n)^(p+1)` 实现半整数幂，不输入浮点指数。Edgeworth 的 Gaussian 尾只从 `t0=delta sqrt(n)` 算一次，通过 `erfc` 与递推

\[
I_0(c;t_0)=\sqrt\pi\,\operatorname{erfc}(\sqrt c\,t_0)/(2\sqrt c),\quad
I_1(c;t_0)=e^{-ct_0^2}/(2c),
\]

\[
I_p(c;t_0)=\{t_0^{p-1}e^{-ct_0^2}+(p-1)I_{p-2}(c;t_0)\}/(2c).
\]

中高频误差直接按 `Pcap+Edgecap` 处理，没有在每段重复计入整个 Gaussian 尾。CDF 的 `Pcap/t` 保守用 `Pcap/t0`；Edge 常数项同样用 `1/t0`，三次项则精确降幂。这比逐段对数积分略保守，但不产生额外数值求积。

## 4. 返回的范数与归一化

返回 `g_n`（含 Gumbel）减一阶 Edgeworth 后的：密度零、一、二阶导数 `L∞`，CDF `L∞`，密度 `L2`，密度一阶导数 `L1`。另返回精确密度自身前两阶导数 `L∞` 上帽。后者使用 `|chi_n|<=1`，不额外加入 Gumbel 误差。

正负频率对称。密度导数 `L∞` 为正频加权绝对积分除 `pi`；Parseval 给密度 `L2²=(1/pi)int_0^infty |R_g|²`。中高频先对 CF 与 Gaussian 两块取 Minkowski：`sqrt(int Pcap²)+sqrt(int Edgecap²)`，再平方。

一阶导数 `L1` 由 Sobolev/Parseval 使用向量 `(t R_g,R_g+t R_g')` 的正频平方积分开根，不再乘 `pi` 因子。其 CF 尾向量封套是

\[
(tP_n,\;P_n+\sqrt n\,tP_{n-1}+\pi^2t^2P_n/(6nv)),
\]

Gaussian 尾向量是 `(t Edge,Edge+t Edge')`。两块向量 `L2` 范数按 Minkowski 相加。低频向量平方帽延到无穷后，与真实尾上帽平方相加，重复覆盖只增加保守性。

这些输出是“所报告上预算”的区间。使用上端作为误差保证；不得把区间下端写成实际误差下界。成功仅表示有限噪声函数界成立，不表示 `D_n` 符号已知。

## 5. 接口、停止与核验边界

```python
saved = finite_noise(constants, step="1/16", N=192,
                     u_start="1/4", u_step="1/64", u_count=273,
                     strip="1", bits=192, hard_seconds=600)
base = compose_saved(saved)             # 仅保存量和解析标量
bounds = compose_bounds(saved, (64,256,1024,4096))
```

`finite_noise` 保存 `configuration,input_constants,raw4_finite,raw4_analytic_error,raw4_true,mu4_true,cf_rows,work_counts`。每个 CF 行含频率、完成节点数、复有限和实虚 dyadic 区间、模区间、解析误差及真区间/真模帽。`composition` 保存重建第四矩、全部节点帽、全部段帽与资格门。

墙钟零预算和中途到期返回 `budget_stop`、已完成节点/CF 行及当前不完整行；不会自动增加精度、网格或预算。运行中普通异常记录 `runtime_failure`、阶段和已保存局部有限和，不自动重试。参数/输入/解析合成错误仍可抛给调用方，由调用方保留失败收据。计时包含内部保存量合成，不含调用方磁盘写入。未完成或资格失败输入不进入 `compose_bounds`。

编写阶段只做了非项目小夹具：5 个 Gaussian 节点、2 个频率（10 次复指数），以及零预算、codec、Gaussian 矩递推等纯标量检查。夹具使用 `v=1,d0=0,kappa3=0` 的制造输入，故不会当作本模型矩证书；资格失败也原样返回。正式 385×273 网格与第四矩目标均留待根任务冻结后执行。

另用五行人工零 CF 表和人工第四矩 3 测试四个 n 的完整合成接口。这张表明确不是概率模型的有效 CF 证据，只检查保存量解码、范数合成和输出字段；没有调用模型积分。正项平方交叉项与高频幂尾分别和闭式表达交叉核对通过，非法 n 被拒绝，所有调用恢复原 Arb 精度。正式结论不得引用这些制造数据。
