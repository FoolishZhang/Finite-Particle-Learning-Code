# CertifiedTeacher-v1：精确 teacher 整向量 L1 求积误差

本文件给出作者的解析求积预算及制造标量核验，不是外部 peer review。范围只包括精确 teacher 的无限均匀 trapezoid alias 误差与有限格点截断误差。前置标量预算属于 planning，没有计算真实 256 项 PMF、科学积分或随机数，没有读取旧数组，也未冻结科学执行。后续真实有限和若使用 Arb，必须另计算术 enclosure 的整向量 L1 误差。

## 精确对象与格点定义

Teacher latent 是二维独立标准 Gaussian，bias 精确为零，八个加载行精确为

\[
w_j=\frac45(\cos\theta_j,\sin\theta_j),\qquad
\theta_j\in\{-7/10,-1/2,-3/10,-1/10,1/10,3/10,1/2,7/10\}.
\]

角度是精确有理数弧度，cos/sin 以严格 Arb 外包计算；不把旧 binary64 加载行当作此对象。令 $\sigma(t)=(1+e^{-t})^{-1}$、$s_j=2y_j-1$，对全部 $y\in\{0,1\}^8$ 定义

\[
A_y(z)=\prod_{j=1}^8\sigma(s_jw_j\cdot z),\qquad
F_y(z)=\phi(z_1)\phi(z_2)A_y(z),\qquad
p_y=\int_{\mathbb R^2}F_y(x)\,dx.
\]

其中 $\phi(t)=(2\pi)^{-1/2}e^{-t^2/2}$。均匀步长 $h=1/8$、$L=10$、实际整数 $N=L/h=80$。无限和及其有限截断为

\[
Q^\infty_y=h^2\sum_{k\in\mathbb Z^2}F_y(hk),\qquad
Q^N_y=h^2\sum_{k_1=-80}^{80}\sum_{k_2=-80}^{80}F_y(hk).
\]

所有边界点和角点都乘完整 $h^2$ 权重。这是无限全实 trapezoid 的矩形截断，不是有限区间复合梯形规则；不能对端点半权。161×161 共 25,921 格点，不做 PMF 重归一化。遗漏从任一轴 $|k_i|=81$ 开始。

## 复条带条件与全标签包络

对 $t=x+i\beta$、$a=e^x>0$，直接代数给

\[
|\sigma(t)|+|\sigma(-t)|
=\frac{1+a}{\sqrt{(1-a)^2+4a\cos^2(\beta/2)}}.
\]

若 $|\beta|<\pi$，则 $\cos(\beta/2)>0$，且

\[
(1-a)^2+4a\cos^2(\beta/2)
-\cos^2(\beta/2)(1+a)^2
=\sin^2(\beta/2)(1-a)^2\ge0.
\]

所以行的两类标签模长之和至多 $\sec(|\beta|/2)$。选闭 polystrip $|\operatorname{Im}z_i|\le d$、$d=1$。各 logit 满足

\[
|\operatorname{Im}(w_j\cdot z)|
\le B_j=d(|w_{j1}|+|w_{j2}|),\qquad B_j<\pi.
\]

192-bit Arb 标量核验所有 $B_j<\pi$，最大值约 $1.12724789962$。Logistic 极点仅可能在 logit 虚部等于奇数倍 $\pi$ 且实部为零处，因此本闭条带及其一个开放邻域内没有极点。对任意合法复平移 $t=(t_1,t_2)$，完整标签求和因子化：

\[
\sum_y|A_y(x+it)|
=\prod_j\bigl(|\sigma(w_j\cdot(x+it))|
             +|\sigma(-w_j\cdot(x+it))|\bigr)
\le C=\prod_j\sec(B_j/2).
\]

Gaussian 复模长为

\[
|\phi(x_1+it_1)\phi(x_2+it_2)|
=e^{(t_1^2+t_2^2)/2}\phi(x_1)\phi(x_2)
\le e^{d^2}\phi(x_1)\phi(x_2).
\]

因此

\[
\int_{\mathbb R^2}\sum_y|F_y(x+it)|\,dx\le M=e^{d^2}C.
\]

本候选 $M$ 约为 $8.21708645275$。Gaussian 包络也给出条带内一致快速衰减；移动轮廓时无穷远边界积分趋零。

## Fourier 平移与二维 Poisson alias

采用角频率 Fourier 约定

\[
\widehat F_y(\omega)=\int_{\mathbb R^2}F_y(x)e^{-i\omega\cdot x}\,dx.
\]

对各非零 $\omega_i$ 把第 i 坐标轮廓移到 $x_i-id\,\operatorname{sign}\omega_i$；零频坐标可不移动。逐轴 Cauchy 轮廓平移合法，因前述 pole-free 闭条带、Gaussian 可积包络及边界消失。由移位的指数因子，

\[
\sum_y|\widehat F_y(\omega)|
\le M e^{-d(|\omega_1|+|\omega_2|)}.
\]

实轴上 $F_y$ 连同各阶导数均快速衰减：实 sigmoid 及其各阶导数有界，加载固定，Gaussian 导数是 polynomial×Gaussian。因此 $F_y$ 属于 Schwartz 类，Poisson 求和可用；频率 alias 和也由上界绝对可和。二维 Poisson 身份为

\[
Q^\infty_y=\sum_{\ell\in\mathbb Z^2}\widehat F_y(2\pi\ell/h).
\]

这里也可直接用 periodization 验证身份：$P_h(t)=\sum_{k\in\mathbb Z^2}F_y(t+hk)$ 的各阶导数在一个实 fundamental square 上一致收敛，因 Schwartz 衰减。其 Fourier series 系数由逐项积分和换元精确为 $h^{-2}\widehat F_y(2\pi\ell/h)$。上述指数界使该 Fourier series 绝对一致收敛；在 $t=0$ 取值并乘 $h^2$ 即得此身份。因此不需要对仅有 L1/strip 假设的更宽函数类另作 Poisson 有效性推断。

零频项是 $p_y$。先按标签取三角不等式，再对非零格点求和，令 $q=e^{-2\pi d/h}$，得到

\[
\sum_y|Q^\infty_y-p_y|
\le M\sum_{\ell\ne(0,0)}q^{|\ell_1|+|\ell_2|}
=M\left[\left(\frac{1+q}{1-q}\right)^2-1\right]
=M\frac{4q}{(1-q)^2}.
\]

代码用最后一个稳定等价形式，避免接近 1 的平方减 1。这里没有 256 倍因子；关键是 Fourier 界在积分前就对完整标签模长求和。

一维实轴 strip/Fourier/Poisson 误差框架可核 [Trefethen–Weideman 2014 原文，§5、Theorem 5.1](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2014_149.pdf)，出版标识 [SIAM DOI 10.1137/130932132](https://epubs.siam.org/doi/10.1137/130932132)。此处二维、整向量式由以上逐轴移位和绝对可和 alias 直接推导，不把一维标量定理误称为现成二维向量结论。

## 有限格点尾及端点处理

实点上 $A_y(x)\ge0$ 且 $\sum_y A_y(x)=1$。令

\[
S_h=h\sum_{k\in\mathbb Z}\phi(kh),\qquad
I_N=h\sum_{k=-N}^N\phi(kh),\qquad
\tau_N=S_h-I_N=2h\sum_{k=N+1}^\infty\phi(kh).
\]

实 Gaussian 在 $[0,\infty)$ 单调递减。对每个遗漏的正 k，矩形右端点界给

\[
h\phi(kh)\le\int_{(k-1)h}^{kh}\phi(x)\,dx.
\]

从 $k=N+1$ 求和恰好以 $Nh=L$ 为积分下端，因此

\[
\tau_N\le2\int_L^\infty\phi(x)\,dx
\le\bar p(L)=2\phi(L)/L.
\]

这是 inclusive $\pm L$ 的尾界。把被保留的 $k=N$ 也算到尾中、或者把端点改成半权，都会改变规则和误差；此证明不采用这两种改法。同样把 $k\ge1$ 的单调右端点矩形相加，给

\[
S_h=h\phi(0)+2h\sum_{k=1}^\infty\phi(kh)
\le1+h\phi(0).
\]

二维遗漏向量各项非负，标签求和精确等于遗漏的 Gaussian 格点质量：

\[
\sum_y(Q^\infty_y-Q^N_y)
=S_h^2-I_N^2
=2S_h\tau_N-\tau_N^2
\le2(1+h\phi(0))\bar p(L).
\]

这里也没有 256 倍因子，不需要逐标签尾积分。Mills 界直接由 $\phi'(x)=-x\phi(x)$ 与 $x\ge L>0$ 得到：$\int_L^\infty\phi\le\int_L^\infty x\phi/L=\phi(L)/L$。

## 整向量预算与严格标量输出

数学有限格点 PMF 与精确 PMF 的 L1 误差合计为

\[
\|p-Q^N\|_1\le E_{\rm alias}+E_{\rm gridtail},\quad
E_{\rm alias}=M\frac{4q}{(1-q)^2},\quad
E_{\rm gridtail}=2(1+h\phi(0))\,\frac{2\phi(L)}L.
\]

前置 192-bit Arb 标量显示约为：

|分量|显示近似（不作为证书）|
|---|---:|
|无限格点 alias L1|$4.86134212239\times10^{-21}$|
|有限格点 tail L1|$3.23132448683\times10^{-23}$|
|总数学 quadrature L1|$4.89365536726\times10^{-21}$|

script/certified_teacher_bounds.py 的 teacher_l1_bounds 使用 exact rational 输入与 Arb 192-bit 对 trig、exp、sqrt 外包，输出每量的精确 dyadic 上下端点。严格预算比较对 dyadic 上端点与有理数 $10^{-12}$ 用整数比较完成，不把 float/十进制字符串当证书。strict_all_L1_target 指总数学 quadrature L1 上端点严格小于该目标。

这留下算术预算，但尚未计算真实 PMF。后续 finite sum 的 Arb enclosure 必须独立求整向量误差 $E_{\rm arithmetic}$；最终保证只在 $E_{\rm alias}+E_{\rm gridtail}+E_{\rm arithmetic}\le10^{-12}$ 时成立。后续必须保持本精确 teacher、grid、完整标签集合与 full endpoint weights。它不认证旧 binary64 teacher、旧有限阶规则、其它 latent 分布或后续 student/风险积分。

## 制造标量核验及未闭合项

script/certified_teacher_bounds_test.py 只检查：

- 候选解析标量预算、pole-free 条件、整数格点/端点定义与 exact target 比较。
- 制造行参数 $a=1/4,1,4$、$\beta=1/5,3/5$ 的复 sigmoid 模长代数恒等式及 secant 外包；$a=1$ 为精确等号情形。
- 零加载制造 teacher：由解析 Gaussian Fourier 系数外包其无限 alias，而非计算 PMF；另以 $h=1,L=3$ 的有限标量 Gaussian 尾前缀加剩余积分界检查 inclusive 端点逻辑。
- 300 个 exact dyadic/rational 比较及输入拒绝、Arb precision 恢复。

运行 .venv/Scripts/python.exe script/certified_teacher_bounds_test.py --write-receipt，以 exclusive-create 保存 artifacts/certified_teacher_v1_20261008/BOUNDS_MANUFACTURED_CHECK.json。已有收据时只运行无写入测试。收据保存来源哈希、全部 Arb 外包及比较；真实 PMF 评估、科学数据读取、随机数和 acb.integral 调用均为零。

未闭合项是后续真实有限和的算术 enclosure 及 PMF enclosure 的下游使用。本轮未做真实积分，不声称已得到完整 teacher 数值向量。
