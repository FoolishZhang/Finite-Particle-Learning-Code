# n0 局部曲率的独立比值证明

本页给出一个不使用新抽样、评分或数值求积的独立证明。它只用 logistic 函数的初等界、Gaussian 分部积分、Cauchy–Schwarz 与 AM–GM。证明针对等权、独立先验粒子的 n0 控制，不针对带无关随机权重的 n256 情形。

**结论。** 在三维单位边际、中心 Gaussian logits、独立均匀标签和等权 K 粒子设定中，I 是总体有限粒子目标的驻点，其三个相关系数坐标的 Hessian 为 h_K I_3。对于 K=100，

\[
h_{100}>0.000240.
\]

下面的闭式下界约为 0.0002404670786254；用纯有理数验证可得到严格下界 0.0002404666795238。因此 I 是 K100 n0 总体目标的严格局部极小点。相同初等界还证明所有整数 K≥100 的严格局部极小性。本结论不证明 I 是连续类的全局最小点，不声称这个下界接近真实曲率，也不改变旧 K2 结果。

## 1. 目标、对称性与导数

令 q(x)=σ(x)，C 是三维相关矩阵，X_k iid∼N(0,C)，并令

\[
F_K(C)=\mathbb E_X\left[-\frac18\sum_{y\in\{-1,1\}^3}
\log\left\{\frac1K\sum_{k=1}^K\prod_{i=1}^3q(y_iX_{ki})\right\}\right].
\]

坐标为 θ=(C₁₂,C₁₃,C₂₃)。标签均匀性给出 F_K(DCD)=F_K(C)，其中 D 为任意对角符号矩阵；坐标置换也保持目标。符号不变性令 I 处所有一次导数及不同相关系数间的二次混合导数为零；置换不变性令三个对角二次导数相等。因此

\[
\nabla_\theta F_K(I)=0,\qquad \nabla_\theta^2F_K(I)=h_K I_3.
\]

为求 C₁₂ 方向的二次导数，可取每个粒子的独立标准正态 X,Y,Z，并用 Y_ρ=ρX+√(1−ρ²)Y 耦合。标签变换把全正标签的 ρ 变成 ±ρ，所以所有标签在 ρ=0 具有相同二阶导数。定义

\[
W=q(X)q(Y)q(Z),\quad
D=q(X)q'(Y)Xq(Z),\quad
V=q(X)q(Z)\{q''(Y)X^2-q'(Y)Y\}.
\]

它们分别是全正标签单粒子似然在 ρ=0 的值、一阶导数及二阶导数。令 S=ΣₖWₖ，D_Σ=ΣₖDₖ，V_Σ=ΣₖVₖ，则

\[
\boxed{h_K=\mathbb E\frac{D_\Sigma^2}{S^2}
-\mathbb E\frac{V_\Sigma}{S}.}\tag{1}
\]

在 ρ=0，|D/W|≤|X|，|V/W|≤X²+|Y|；在 ρ 的一个固定小邻域，相关比值导数同样由 Gaussian 坐标的多项式控制。有限 K 的 log-sum 导数是相应比值的加权组合，故可以在期望内微分。更一般地，在 I 附近取光滑矩阵平方根耦合，得到本节使用的 C² 正则性。

记 μ=EW=1/8，d=ED。由 Gaussian 分部积分，E[Xq(X)]=Eq'(X)，故

\[
d=\frac12\{\mathbb E q'(X)\}^2,\qquad EV=0.\tag{2}
\]

后一个等式使用 q'' 为奇函数、Yq'(Y) 为奇函数。

## 2. 比值的通用下界

写 v=Var(W)、e₂=EV²。Cauchy–Schwarz 给出

\[
\mathbb E\frac{D_\Sigma^2}{S^2}
\ge\frac{(ED_\Sigma)^2}{ES^2}
=\frac{d^2}{\mu^2+v/K}.\tag{3}
\]

令 T=Σₖ₌₂ᴷWₖ。由 EV₁=0 且 T 与 (W₁,V₁) 独立，

\[
\begin{aligned}
\left|\mathbb E\frac{V_\Sigma}{S}\right|
&=K\left|\mathbb E\left[V_1\left\{\frac1{T+W_1}-\frac1{T+\mu}\right\}\right]\right|\\
&\le K\mathbb E|V_1(W_1-\mu)|\,ET^{-2}\\
&\le K\sqrt{e_2v}\,ET^{-2}.
\end{aligned}\tag{4}
\]

因此，只需一个逆矩上界。设 n=K−1≥1。AM–GM 和粒子独立性给出

\[
ET^{-2}\le\frac1{n^2}\{EW^{-2/n}\}^n.\tag{5}
\]

由 cosh u≤exp(u²/2)，对 r=2/n<4，

\[
\begin{aligned}
Eq(X)^{-r}
&=2^r E\{e^{-rX/2}\cosh^r(X/2)\}\\
&\le 2^r(1-r/4)^{-1/2}
\exp\left\{\frac{r^2}{8(1-r/4)}\right\}.
\end{aligned}
\]

W 是三个独立 q 因子的乘积。将上式代入 (5)，得到

\[
ET^{-2}\le U_n:={64\over n^2}
\left(1-\frac1{2n}\right)^{-3n/2}
\exp\left\{\frac3{2n(1-1/(2n))}\right\}.\tag{6}
\]

所有这些矩均有限。式 (1)–(6) 给出

\[
h_K\ge\frac{d^2}{\mu^2+v/K}-K\sqrt{e_2v}\,U_{K-1}.\tag{7}
\]

## 3. 单粒子矩的初等常数

q(x)=(1+tanh(x/2))/2，且 |tanh(x/2)|≤|x|/2。对称性消去奇项，因此

\[
Eq(X)^2\le\frac14+\frac1{16}EX^2=\frac5{16},
\]

从而

\[
v\le\left(\frac5{16}\right)^3-\frac1{64}
=\frac{61}{4096}=:v_* .\tag{8}
\]

又 q'(x)=(1−tanh²(x/2))/4≥1/4−x²/16，故

\[
Eq'(X)\ge\frac3{16},\qquad d\ge\frac9{512}=:d_* .\tag{9}
\]

下面保守控制 e₂。首先，

\[
E[X^4q(X)^2]\le\tfrac12 EX^4=\frac32,
\qquad E[X^2q(X)^2]\le\frac14EX^2+\frac1{16}EX^4=\frac7{16}.
\tag{10}
\]

第一式利用 q(x)²+q(−x)²≤1。另有 supₓq''(x)²=1/108。

为控制 M=E[Y²q'(Y)²]，注意 cosh²(Y/2)≥1+Y²/4，因此 q'(Y)≤1/(Y²+4)。令 J=E[1/(Y²+4)]。Cauchy–Schwarz 给出

\[
E\frac{Y^2}{Y^2+4}
\ge\frac{(EY^2)^2}{E[Y^2(Y^2+4)]}=\frac17,
\quad J\le\frac3{14}.
\]

对 g(Y)=Y/(Y²+4) 作 Gaussian 分部积分，可得

\[
E\frac{Y^2}{(Y^2+4)^2}=\frac{5J-1}{2}\le\frac1{28}.
\]

故 M≤1/28。因为 q''(y)=−q'(y)tanh(y/2)，还有

\[
-2yq''(y)q'(y)
=2y\tanh(y/2)q'(y)^2\le y^2q'(y)^2.
\]

展开 EV²，并按 X,Y,Z 独立性分离各因子，得到

\[
\begin{aligned}
e_2
&\le Eq(Z)^2\left\{
\frac{E[X^4q(X)^2]}{108}
+\left(Eq(X)^2+E[X^2q(X)^2]\right)M\right\}\\
&\le\frac5{16}\left\{\frac1{72}+\frac3{112}\right\}
=:e_* .
\end{aligned}\tag{11}
\]

代入 (7)，得到完全显式的下界

\[
\boxed{
h_K\ge L_K:={ (9/512)^2\over 1/64+61/(4096K)}
-K\sqrt{\frac{61}{4096}\frac5{16}
\left(\frac1{72}+\frac3{112}\right)}\,U_{K-1}.}
\tag{12}
\]

这里只对不等式右侧使用了上界常数；没有把上界常数当成真实矩。

## 4. K100 的纯有理数正号证书

式 (12) 的通常数值显示为 L₁₀₀≈0.0002404670786254。正号和保守的六位小数门可完全以有理数运算核验，无需求积或浮点误差假设。

取 x=1/198，并定义

\[
\ell=\sum_{j=1}^4\frac{x^j}{j}+\frac{x^5}{5(1-x)},
\quad z=\frac{297}{2}\ell+\frac3{197},
\]

\[
P=\sum_{j=0}^{12}\frac{z^j}{j!}
+\frac{z^{13}}{13!}\frac1{1-z/14}.
\]

对数级数尾部的系数均不超过 1/5，故 −log(1−x)≤ℓ。指数级数从第 13 项开始的相邻项之比不超过 z/14，且 z<1，因此 exp(z)≤P。由 (6)，

\[
U_{99}\le\frac{64}{9801}P
<\frac{140627215}{10^{10}},
\qquad
\sqrt{v_*e_*}<\frac{13758517}{10^9}.
\tag{13}
\]

式 (13) 的两项严格有理数比较，以及下述结论，可直接运行文末代码核验：

\[
\begin{aligned}
h_{100}
&>\frac{2025}{103376}
-100\frac{13758517}{10^9}\frac{140627215}{10^{10}}\\
&=\frac{31073104328071709}{129220000000000000000}\\
&>\frac{240}{10^6}.
\end{aligned}\tag{14}
\]

这是严格正数的确定性解析证书，不是新置信区间。

```python
from fractions import Fraction as Q
from math import factorial

v = Q(61, 4096)
e = Q(5, 16) * (Q(1, 72) + Q(3, 112))
x = Q(1, 198)
ell = sum(x**j / j for j in range(1, 5)) + x**5 / (5 * (1 - x))
z = Q(297, 2) * ell + Q(3, 197)
assert 0 < z < 1
P = sum(z**j / factorial(j) for j in range(13))
P += z**13 / factorial(13) / (1 - z / 14)
u = Q(140627215, 10**10)
s = Q(13758517, 10**9)
assert Q(64, 9801) * P < u
assert v * e < s * s
h_lower = Q(2025, 103376) - 100 * s * u
assert h_lower == Q(31073104328071709, 129220000000000000000)
assert h_lower > Q(240, 10**6)
```

## 5. K≥100 与结论范围

式 (12) 第一项随 K 严格增加。第二项的 K 依赖因子为

\[
\frac{n+1}{n^2}
\exp\left\{-\frac{3n}{2}\log\left(1-\frac1{2n}\right)
+\frac3{2n-1}\right\},\qquad n=K-1.
\]

(n+1)/n² 随 n>0 减少，3/(2n−1) 随 n>1/2 减少。设 t=1/(2n)，剩余指数项的导数为

\[
\frac32\left\{-\log(1-t)-\frac{t}{1-t}\right\}<0,
\]

其中不等式由幂级数或积分直接得到。因此第二项随 n 减少，L_K 随 K 增加。K100 的严格正号证书推出所有整数 K≥100 均有 h_K>0。

由 I 是相关矩阵类的内部点、F_K 在其邻域 C²，且 Hessian 正定，I 为严格局部极小点。这里没有给出显式邻域半径，也没有排除更远处存在更低目标值。不能把局部结论写成 n0 连续总体全局最优，不能用它替代 n256 的独立分析，更不能据此断言任意有限优化器返回 I。

本证明由局部比值公式 (1) 出发，可作为 Laplace/Price 路线的独立数学复核。验证只包含标准正态矩恒等式和文中有理数比较，无新增科学样本、先验粒子、似然评分、求积节点或稿件改动。
