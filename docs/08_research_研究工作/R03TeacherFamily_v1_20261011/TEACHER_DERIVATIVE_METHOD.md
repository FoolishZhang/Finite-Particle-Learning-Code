# R03 教师线性接口：全标签 PMF、角导数与二维严格求积

日期：2026-10-11。实现为 [r03_teacher_derivative.py](../../../script/r03_teacher_derivative.py)。本文给出接口和数学误差证明；本文编写/自检阶段没有执行正式25921节点教师有限和。只有256标签 Fraction 制造代数、一个三观察变量的2x2制造网格、以及不评价目标的解析标量预算。正式生产须由本轮主执行器先冻结来源、参数与预算，再调用此模块并保存结果。

模块直接使用公开 python-flint，不导入旧 producer、私有 `.cache`、科学数组或旧教师数值。模块导入时不计算目标、不写文件；命令行只有 `--self-test`。下面所有误差是确定性区间界，不是统计置信区间。

## 1. 固定对象、标签和接口

默认 m=8，rho=4/5，原角度

\[
 \phi=(-7/10,-1/2,-3/10,-1/10,1/10,3/10,1/2,7/10),
 \quad \theta_i(\lambda)=\phi_i+\lambda\mathbf1_{i=1},
\]

教师 bias 恒为0，第 i 行载荷是 rho(cos theta_i,sin theta_i)。本轮生产中心首先冻结为 lambda_c=0；API 支持任意明确的有理中心，不接受 binary64 浮点输入。默认 h=1/8、N=80、L=Nh=10、条带 eta=1、192bit，全权正方形节点 (kh,lh)、-N<=k,l<=N，共25921节点。

标签整数的最高位是 y_1，最低位是 y_8，输出顺序0,...,255。全部256个 PMF 与全部256个导数均独立累加保存，不通过互补/反序平均修改结果，不把有限 Gaussian 质量或 PMF 重归一。一般中心依然具有互补对称，但 q'_lambda 通常没有原反序对称。

对 z=(x,y)，定义

\[
 t_i=\rho(\cos\theta_i\,x+\sin\theta_i\,y),\quad
 p_i=\sigma(t_i),\quad
 \dot t_1=\rho(-\sin\theta_1\,x+\cos\theta_1\,y),
\]
\[
 A_s(z)=\prod_i p_i^{s_i}(1-p_i)^{1-s_i},\qquad
 \dot A_s(z)=(2s_1-1)p_1(1-p_1)\dot t_1
 \prod_{i>1}p_i^{s_i}(1-p_i)^{1-s_i}.
\tag{T1}
\]

目标为 q_c(s)=E A_s、q'_c(s)=E dot A_s。代码生产端用 T1 第一因子导数；制造独立端用 `A_s*(s_1-p_1)*dot t_1`，且 Bernoulli 概率换为 tanh 表达。高斯正态密度是 phi(x)phi(y)，有限和分别为

\[
 q^f_c(s)=h^2\sum_{k,l=-N}^{N}\varphi(kh)\varphi(lh)A_s(kh,lh),\quad
 d^f_c(s)=h^2\sum_{k,l=-N}^{N}\varphi(kh)\varphi(lh)\dot A_s(kh,lh).
\tag{T2}
\]

主要 API：

```python
analytic_bounds(lambda_centre="0", config=None)
finite_grid_block(lambda_centre="0", outer_indices=None, config=None, deadline=None)
aggregate_blocks(blocks)
compose(finite, bounds=None)
self_test()
```

`config` 默认字段为 `h,N,strip,amplitude,base_angles,precision_bits`。`finite_grid_block` 可以按外层行切块；deadline 是 `time.perf_counter()` 的绝对截止时刻，在节点之间检查。超时返回已累加的原始部分结果，`completed=False`，没有自动续算或重试。`aggregate_blocks` 仅接受完整、无重复、覆盖所有外层行的同一正方形；不接受制造或部分块。调用者负责 exclusive 保存、冻结哈希、资源预算及异常记录。

## 2. 复条带的标签 L1 包络

允许两坐标分别平移 z_1=x+i tau_1、z_2=y+i tau_2，|tau_j|<=eta。令

\[
 \beta_i=\frac{\rho\eta}{2}(|\cos\theta_i|+|\sin\theta_i|)<\pi/2,
 \quad s_i=\sec\beta_i,\quad B_0=\prod_i s_i.
\tag{T3}
\]

这里 s_i 只是本节 secant 常数，不是 T1 的标签。代码严格核 beta_i<pi/2。复 logistic 的两标签模长和满足

\[
 |\sigma(t_i)|+|\sigma(-t_i)|\le\sec(|\operatorname{Im}t_i|/2)\le s_i,
\tag{T4}
\]

故 sum_label |A_label|<=B0。又

\[
 |\sigma'(t_1)|=\frac1{4|\cosh(t_1/2)|^2}
 \le\tfrac14\sec^2\beta_1,
\]

因为 |cosh((a+ib)/2)|^2=sinh^2(a/2)+cos^2(b/2)。对两个第一标签求和得到因子2，再用

\[
 |\dot t_1|\le\rho\{|\sin\theta_1|(|x|+\eta)
                  +|\cos\theta_1|(|y|+\eta)\}
\]

与 E|Z|=sqrt(2/pi)，得到全部导数标签在移轴后的积分包络

\[
 B_1=\frac\rho2\sec^2\beta_1\prod_{i>1}\sec\beta_i
 (|\sin\theta_1|+|\cos\theta_1|)(\sqrt{2/\pi}+\eta).
\tag{T5}
\]

Gaussian 复平移模长乘 exp((tau_1^2+tau_2^2)/2)<=exp(eta^2)。因此，包含单轴平移、双轴同号或异号平移的统一标签 L1 边界积分帽为

\[
 M_0=e^{\eta^2}B_0,\qquad M_1=e^{\eta^2}B_1.
\tag{T6}
\]

rho=4/5 时 T5 的前系数为.4。这里对标签求和已经完成，后续不能再乘256。T3 的严格余量保证闭条带包含在稍大的无极点区域中；概率因子在该条带有界，导数多至线性实坐标包络。Gaussian 衰减使逐轴 Cauchy 移轴的远端竖边积分消失，也使 Fubini 与下面的 Poisson 求和合法。

## 3. 二维无限梯形 alias 系数：明确保留单轴和双轴项

对向量函数 f=(phi(x)phi(y)A_label)_label，或其角导数版本，取角频率 Fourier 变换。对非零双指标 (k,l)，按各指标的符号分别将相应坐标移到 +/-i eta；指标为0的坐标保持实轴。T6 与三角不等式给

\[
 \sum_{\rm label}|\widehat f_{\rm label}(2\pi k/h,2\pi l/h)|
 \le M\exp[-(2\pi\eta/h)(|k|+|l|)].
\tag{T7}
\]

需要同时移两轴的项由同一个 M 覆盖，不能只证明两条实轴边界后默认双移轴也成立；T3–T6 已明确覆盖全部符号组合。由于闭条带内有 Gaussian 衰减，实轴函数和相应各阶导数快速衰减，Poisson 展开绝对收敛；可逐分量求和，再按标签 L1 合成。

设 r=exp(-2 pi eta/h)。无限全权网格与真实积分之差满足

\[
 E_{\rm alias}\le M\sum_{(k,l)\ne(0,0)}r^{|k|+|l|}
 =M\left[\left(1+\frac{2r}{1-r}\right)^2-1\right]
 =M\frac{4r}{(1-r)^2}.
\tag{T8}
\]

其中单轴频率贡献4r/(1-r)，双非零频率贡献4r^2/(1-r)^2。二者相加是最后的系数；不再添加额外的双边或四角倍数。代码用4r/(1-r)^2避免两个接近1的数相减；制造 Fraction 检查也核该恒等式。

## 4. 真正的离散网格尾界

T8 控制的是整个无穷网格。截断误差必须控制该网格省略的节点，而不仅是连续 Gaussian 尾。定义

\[
 S_0=h\sum_{k\in\mathbb Z}\varphi(kh),\quad
 S_1=h\sum_{k\in\mathbb Z}|kh|\varphi(kh),
\]
\[
 T_0=h\sum_{|k|>N}\varphi(kh),\quad
 T_1=h\sum_{|k|>N}|kh|\varphi(kh).
\tag{T9}
\]

每个已保留边界节点 kh=+/-L 都具有完整权重；遗漏从 +/- (L+h) 开始。Gaussian 在正半轴递减，因此右矩形比较和 Mills 上帽给

\[
 S_0\le1+h\varphi(0),\qquad T_0\le2\int_L^\infty\varphi(x)dx\le2\varphi(L)/L.
\tag{T10}
\]

函数 x phi(x) 只在 x>=1 时递减；接口因此要求 L>=1。在省略区域可用右矩形比较：

\[
 T_1\le2\int_L^\infty x\varphi(x)dx=2\varphi(L).
\tag{T11}
\]

整个实轴上不能把 S1 直接换成 E|Z|，因为 |x|phi(x) 在0到1增加。我们改用其总变差。对可积 BV 函数 f，逐网格 cell 比较有

\[
 \left|h\sum_{k\in\mathbb Z}f(kh)-\int f(x)dx\right|\le h\,TV(f).
\]

在 f(x)=|x|phi(x) 时，两个峰均为 phi(1)，总变差是4phi(1)，所以

\[
 S_1\le\sqrt{2/\pi}+4h\varphi(1).
\tag{T12}
\]

实轴全部标签的概率和为1；用“第一轴超界或第二轴超界”的并集上界，PMF 的无穷网格省略 L1 质量至多

\[
 E_{q,\rm tail}\le2S_0T_0.
\tag{T13}
\]

实轴上 T1 的导数标签模长和有更直接的界

\[
 \sum_{\rm label}|\dot A_{\rm label}(x,y)|
 =2p_1(1-p_1)|\dot t_1|
 \le\frac\rho2(|\sin\theta_1||x|+|\cos\theta_1||y|).
\tag{T14}
\]

省略正方形外节点的两个轴并集分别给 T1 S0 与 S1 T0，故

\[
 E_{d,\rm tail}\le\frac\rho2(|\sin\theta_1|+|\cos\theta_1|)
 (T_1S_0+T_0S_1).
\tag{T15}
\]

交角区域在并集中被重复计数仅使上界更松。T10/T11 的积分比较只用于已证明单调的外侧；T12 负责内侧/整轴的非单调部分，未将所有离散尾草率替换成连续积分尾。

## 5. 有限和、解析误差与真实分量的合成

令

\[
 \epsilon_q=M_0\frac{4r}{(1-r)^2}+2S_0T_0,\qquad
 \epsilon_d=M_1\frac{4r}{(1-r)^2}
 +\frac\rho2(|\sin\theta_1|+|\cos\theta_1|)(T_1S_0+T_0S_1).
\tag{T16}
\]

它们严格控制 ||q_c-q_c^f||_1、||q'_c-d_c^f||_1。192bit Arb 计算包住真实有限和 q^f,d^f 的全部算术误差；解析 T16 另加一次。默认 lambda_c=0 的标量预算为

|对象|总 L1 解析求积误差上界的显示近似|
|---|---:|
|q_c|4.89365537e-21|
|q'_c|5.92655488e-21|

正式证书以输出 exact dyadic 上端点为准，不能将此表舍入显示作为原始端点。这些数的计算没有评价任何25921节点教师目标，只计算 T3–T16 的显式标量。

`compose` 给每个 q 分量的有限和球加 +/-epsilon_q，每个导数分量球加 +/-epsilon_d。这是合法但稍松的分量包围，因为单分量误差不超过总 L1 误差；同时保留总 L1 预算，后续不可误将一个 epsilon 乘256后称它是原先的联合预算。

导数 L1 范数使用反三角不等式作更紧的联合合成：

\[
 \|q'_c\|_1\in[\|d_c^f\|_1-\epsilon_d,\|d_c^f\|_1+\epsilon_d]\cap[0,\infty).
\tag{T17}
\]

代码对有限和原始球先求绝对值与总和，再扩 epsilon_d；没有对已逐分量扩完的球重新求和，因而没有不必要的256倍解析误差。对七个 spin pair (1,j)，j=2,...,8，系数 chi_j(label)=(2y_1-1)(2y_j-1) 绝对值为1，所以

\[
 \partial_\lambda E_{q_\lambda}[(2Y_1-1)(2Y_j-1)]|_{\lambda_c}
 \in\sum_{\rm label}\chi_j(\mathrm{label})[d_c^f(\mathrm{label})]
 +[-\epsilon_d,\epsilon_d].
\tag{T18}
\]

`compose` 返回这些导数及对应 q_c 的 spin 矩包围，并保留 label 编号。它不重新积分、不对有限结果重归一化。

## 6. 线性教师近似怎样进入连续根管

第一 Bernoulli 因子导数与 Gaussian 矩给全实 lambda 的统一界，详见 [CONTINUATION_ARGUMENT C9–C11](CONTINUATION_ARGUMENT.md)：

\[
 \|q'_\lambda\|_1\le\frac\rho2\sqrt{2/\pi},\qquad
 \|q''_\lambda\|_1\le\frac{\rho^2}{3\sqrt3}+\frac\rho2\sqrt{2/\pi}.
\tag{T19}
\]

rho=4/5 时第二项总和严格<.442323，所以对任意实 lambda（包括本轮[-.1,.1]）

\[
 \|q_\lambda-q_c-(\lambda-\lambda_c)q'_c\|_1
 \le.2211615(\lambda-\lambda_c)^2.
\tag{T20}
\]

若后续直接用精确数学有限和构成线性权重 q_c^f+s d_c^f，s=lambda-lambda_c，则与真实 q_lambda 的总误差为

\[
 \|q_\lambda-(q_c^f+s d_c^f)\|_1
 \le\epsilon_q+|s|\epsilon_d+.2211615s^2.
\tag{T21}
\]

有限系数的保存区间运算误差继续保留在区间里。T21 是整个标签向量的 L1 余项，可乘单标签梯度/Hessian/角曲率的统一帽一次；不必将二阶余项逐标签独立扩散后再支付256倍。若需要严格 q 坐标盒，也可保守逐分量扩同一个 T21，但必须明确这种盒包裹会较松。

线性权重 q_c^f+s d_c^f 不应不经检查就当非负且归一的 PMF。对依赖“加权质量=1”或“权重非负”的候选求积误差，须检查整个 s 段的权重区间，或直接用其总绝对质量严格上界，再加 T21。q 的真实精确性质与近似线性权重的性质不能混用。T20/T21 也不处理 a(lambda) 的移动；候选根管的区间、收缩与稳定性仍要另行认证。

## 7. 解析恒等式与实际制造核验

对每个真实/制造节点，sum_label A_label=1、sum_label dot A_label=0；对每个其余七位标签，两个第一位的导数互相抵消。因此只由 Y_2,...,Y_8 决定的全部边缘分布对 lambda 不变。这是逐节点等式，不依赖 Gaussian 网格精度。

零偏置的反射还给 A_(1-label)(-z)=A_label(z)、dot A_(1-label)(-z)=dot A_label(z)。完整对称正方形的 q 与 q' 因而都有互补对称；所有单 spin 导数为0。旧反序对称不用于导数生产或检查。

`self_test` 的执行范围：

- 用八个指定有理 p_i 和有理 dot t_1 枚举256标签，逐标签核第一因子式与独立 score 式、总质量、总导数、七维完整边缘导数、七个 pair 导数闭式。
- 将制造节点的 p_i 同时换成1-p_i、dot t_1 换成负值，精确核全部256标签的反射关系、对称两节点组合的互补导数、八个单 spin 导数为0。漏掉第一标签符号的负对照必须失败。
- 在三变量、非项目角度/行长的2x2制造网格比较 logistic 第一因子端与 tanh/score 独立端；仅4个制造节点、8标签，不计算项目教师。
- 核 T8 的有理恒等式、默认解析误差阈值、浮点输入拒绝、L<1拒绝、部分/制造块不能伪装完整生产输入。

公开环境运行：

```powershell
<公开python解释器> -B script/r03_teacher_derivative.py --self-test
```

自检返回 JSON、没有文件写入；主执行器可将该返回值与脚本/本文件哈希绑定保存。制造通过支持实现与恒等式，不能替代未执行的正式有限和。正式交付还须有全块覆盖、q/q'的原始区间、解析合成、预算/来源锁及相应停止记录。
